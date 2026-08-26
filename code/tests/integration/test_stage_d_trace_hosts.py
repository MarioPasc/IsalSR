"""Integration tests: the Stage-D tracer fires inside the live dedup path.

SP-5 requires the D2 instrumentation to be demonstrated on **both** hosts, not
just unit-tested against a synthetic stream.  Each test runs a real, very short
search and then asserts on the artefacts the run left behind, plus the matching
no-op case with the environment unset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.models.stage_d_trace import (
    CANDIDATES_FILE,
    CANON_COST_FILE,
    FALLBACK_MD_FILE,
    SPOT_CHECK_FILE,
    STREAM_SIZE_FILE,
)

pytestmark = pytest.mark.integration

ARTEFACTS = (
    CANDIDATES_FILE,
    CANON_COST_FILE,
    FALLBACK_MD_FILE,
    SPOT_CHECK_FILE,
    STREAM_SIZE_FILE,
)

MAX_TIME_S = 5.0


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a tiny Nguyen-1-style problem: ``y = x^3 + x^2 + x``."""
    rng = np.random.default_rng(0)
    x_train = rng.uniform(-1.0, 1.0, size=(40, 1))
    x_test = rng.uniform(-1.0, 1.0, size=(20, 1))

    def target(x: np.ndarray) -> np.ndarray:
        return (x[:, 0] ** 3 + x[:, 0] ** 2 + x[:, 0]).astype(float)

    return x_train, target(x_train), x_test, target(x_test)


def _enable(monkeypatch: pytest.MonkeyPatch, out_dir: Path, rate: int = 1) -> None:
    monkeypatch.setenv("ISALSR_STAGE_D_TRACE", "1")
    monkeypatch.setenv("ISALSR_STAGE_D_TRACE_DIR", str(out_dir))
    monkeypatch.setenv("ISALSR_STAGE_D_TRACE_SAMPLE_RATE", str(rate))


def _disable(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ISALSR_STAGE_D_TRACE",
        "ISALSR_STAGE_D_TRACE_DIR",
        "ISALSR_STAGE_D_TRACE_SAMPLE_RATE",
    ):
        monkeypatch.delenv(name, raising=False)


def _rows(trace_dir: Path) -> list[dict[str, Any]]:
    text = (trace_dir / CANDIDATES_FILE).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _assert_live_stream(trace_dir: Path) -> list[dict[str, Any]]:
    """Assert the artefacts of a live traced run and return the records."""
    for name in ARTEFACTS:
        assert (trace_dir / name).is_file(), f"missing {name}"
    rows = _rows(trace_dir)
    assert rows, "the tracer produced no candidates from a live run"

    replayable = [r for r in rows if r["serialisation"]]
    assert replayable, "no record carries a replayable serialisation"
    for row in replayable:
        assert row["k"] is not None and row["k"] >= 0
        assert isinstance(row["labels"], dict) and row["labels"]
        assert row["digest_insertion"] is not None
        assert row["digest_topological"] is not None
        assert row["digest_topological_commutative"] is not None
        assert row["t_canon_s"] > 0.0

    canonical = [r for r in replayable if r["canonical"]]
    assert canonical, "no record carries a canonical string"
    assert any(r["t_eval_s"] > 0.0 for r in canonical), "T_eval was never measured"

    hist = json.loads((trace_dir / CANON_COST_FILE).read_text())
    assert hist["by_k"], "the k-stratified histogram is empty"
    assert hist["n_sampled"] == len(rows)

    spot = json.loads((trace_dir / SPOT_CHECK_FILE).read_text())
    assert spot["n_checked"] > 0
    assert spot["n_mismatch"] == 0, spot["checks"]
    assert spot["clean"] is True
    return rows


# --------------------------------------------------------------------------- #
# Bingo
# --------------------------------------------------------------------------- #


class TestBingoHost:
    """The tracer must fire inside ``IsalSREvaluation._serial_eval``."""

    def _run(self, dataset: Any, config: dict[str, Any]) -> None:
        from experiments.models.bingo.isalsr_runner import IsalSRBingoRunner

        x_train, y_train, x_test, y_test = dataset
        runner = IsalSRBingoRunner(config=config)
        runner.fit(x_train, y_train, x_test, y_test, seed=1, config=config)

    @pytest.fixture
    def config(self) -> dict[str, Any]:
        return {
            "population_size": 24,
            "stack_size": 12,
            "max_generations": 20,
            "max_time": MAX_TIME_S,
            "operators": ["+", "-", "*"],
            "use_simplification": False,
            "problem_name": "Nguyen-1",
            "shadow_hash": False,
        }

    def test_trace_fires_in_live_dedup_path(
        self,
        dataset: Any,
        config: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trace_dir = tmp_path / "c2_trace"
        _enable(monkeypatch, trace_dir, rate=1)
        self._run(dataset, config)
        rows = _assert_live_stream(trace_dir)
        hist = json.loads((trace_dir / CANON_COST_FILE).read_text())
        assert hist["run"]["method"] == "bingo"
        assert hist["run"]["seed"] == 1
        assert hist["run"]["problem"] == "Nguyen-1"
        assert any(r["dedup_hit"] for r in rows), "no duplicate observed in a 5 s Bingo run"

    def test_sampling_rate_is_honoured_live(
        self,
        dataset: Any,
        config: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trace_dir = tmp_path / "c2_trace"
        _enable(monkeypatch, trace_dir, rate=7)
        self._run(dataset, config)
        indices = [r["i"] for r in _rows(trace_dir)]
        assert indices == list(range(0, 7 * len(indices), 7))

    def test_no_op_when_env_unset(
        self,
        dataset: Any,
        config: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _disable(monkeypatch)
        self._run(dataset, config)
        assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# UDFS
# --------------------------------------------------------------------------- #


class TestUdfsHost:
    """The tracer must fire inside the monkey-patched ``evaluate_cgraph``."""

    def _run(self, dataset: Any, config: dict[str, Any]) -> None:
        from experiments.models.udfs.isalsr_runner import IsalSRUDFSRunner

        x_train, y_train, x_test, y_test = dataset
        runner = IsalSRUDFSRunner(config=config)
        runner.fit(x_train, y_train, x_test, y_test, seed=1, config=config)

    @pytest.fixture
    def config(self) -> dict[str, Any]:
        return {
            "k": 1,
            "n_calc_nodes": 2,
            "max_orders": 200,
            "processes": 1,
            "max_time": MAX_TIME_S,
            "problem_name": "Nguyen-1",
            "shadow_hash": False,
            "verbose": 0,
        }

    def test_trace_fires_in_live_dedup_path(
        self,
        dataset: Any,
        config: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trace_dir = tmp_path / "c2_trace"
        _enable(monkeypatch, trace_dir, rate=1)
        self._run(dataset, config)
        _assert_live_stream(trace_dir)
        hist = json.loads((trace_dir / CANON_COST_FILE).read_text())
        assert hist["run"]["method"] == "udfs"
        assert hist["run"]["seed"] == 1

    def test_no_op_when_env_unset(
        self,
        dataset: Any,
        config: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _disable(monkeypatch)
        self._run(dataset, config)
        assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# End-to-end: live stream -> Mode-1 replay
# --------------------------------------------------------------------------- #


class TestLiveStreamReplays:
    """A stream produced by a real Bingo run must replay cleanly through D3."""

    def test_replay_of_a_live_bingo_stream(
        self, dataset: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from experiments.models.bingo.isalsr_runner import IsalSRBingoRunner
        from experiments.scripts import stage_d_mode1_replay as m1

        trace_dir = tmp_path / "c2_trace"
        _enable(monkeypatch, trace_dir, rate=1)
        config = {
            "population_size": 24,
            "stack_size": 12,
            "max_generations": 20,
            "max_time": MAX_TIME_S,
            "operators": ["+", "-", "*"],
            "use_simplification": False,
            "problem_name": "Nguyen-1",
            "shadow_hash": False,
        }
        x_train, y_train, x_test, y_test = dataset
        IsalSRBingoRunner(config=config).fit(
            x_train, y_train, x_test, y_test, seed=1, config=config
        )

        out_json = tmp_path / "replay.json"
        out_md = tmp_path / "replay.md"
        code = m1.main(
            [
                "--trace-dir",
                str(trace_dir),
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ]
        )
        report = json.loads(out_json.read_text())
        assert report["hash_soundness_ok"] is True, report["streams"][0]["hash_soundness"]
        assert report["isalsr_soundness_ok"] is True
        assert report["replay_fidelity_ok"] is True
        assert code == 0
        overall = report["streams"][0]["ratios"]["overall"]
        assert overall["n"] > 0
        assert overall["rho_iso"] >= overall["rho_total"] - 1e-12
        assert report["streams"][0]["ratios"]["monotonicity_ok"] is True
        assert out_md.read_text()
