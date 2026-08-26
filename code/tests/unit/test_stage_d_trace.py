"""Unit tests for the Stage-D D2 trace persistence layer.

Covers the no-op contract when the environment flags are unset, the schema of the
five artefacts, the deterministic sampling rule, and the spot check's ability to
catch a corrupted recorded canonical string.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from experiments.models.fallback_ledger import FallbackLedger
from experiments.models.stage_d_trace import (
    CANDIDATES_FILE,
    CANON_COST_FILE,
    FALLBACK_MD_FILE,
    FALLBACK_PATHS,
    SPOT_CHECK_FILE,
    STREAM_SIZE_FILE,
    StageDTraceConfig,
    StageDTracer,
    StorageBudget,
    canonical_digest,
    unreachable_nonvar,
)
from isalsr.baselines.fixed_order_hash import FixedOrder, fixed_order_digest, serialise
from isalsr.core.canonical import fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

ARTEFACTS = (
    CANDIDATES_FILE,
    CANON_COST_FILE,
    FALLBACK_MD_FILE,
    SPOT_CHECK_FILE,
    STREAM_SIZE_FILE,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _dag_add(n_vars: int = 2) -> LabeledDAG:
    """Return ``x_0 + x_1`` as a LabeledDAG."""
    dag = LabeledDAG(max_nodes=8)
    for i in range(n_vars):
        dag.add_node(NodeType.VAR, var_index=i)
    node = dag.add_node(NodeType.ADD)
    for i in range(n_vars):
        dag.add_edge(i, node)
    return dag


def _dag_mul() -> LabeledDAG:
    """Return ``x_0 * x_1`` as a LabeledDAG."""
    dag = LabeledDAG(max_nodes=8)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    node = dag.add_node(NodeType.MUL)
    dag.add_edge(0, node)
    dag.add_edge(1, node)
    return dag


def _dag_sin_of_add() -> LabeledDAG:
    """Return ``sin(x_0 + x_1)`` as a LabeledDAG."""
    dag = _dag_add()
    node = dag.add_node(NodeType.SIN)
    dag.add_edge(2, node)
    return dag


def _dag_orphan_const() -> LabeledDAG:
    """Return a DAG with a CONST that no VAR reaches (RTF precondition violated)."""
    dag = LabeledDAG(max_nodes=8)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.CONST)
    return dag


@pytest.fixture
def enabled_env(tmp_path: Path) -> dict[str, str]:
    """Return a Stage-D environment enabling the tracer at rate 1."""
    return {
        "ISALSR_STAGE_D_TRACE": "1",
        "ISALSR_STAGE_D_TRACE_DIR": str(tmp_path / "c2_trace"),
        "ISALSR_STAGE_D_TRACE_SAMPLE_RATE": "1",
    }


def _feed(tracer: StageDTracer, dags: list[LabeledDAG], **kwargs: Any) -> None:
    """Push *dags* through the tracer's per-candidate protocol."""
    for dag in dags:
        tracer.begin()
        tracer.note_eval_time(1e-4)
        tracer.record(
            dag=dag,
            representation=fast_canonical_string(dag),
            t_canon=5e-4,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# No-op contract
# --------------------------------------------------------------------------- #


class TestDisabled:
    """The tracer must be inert unless the environment says otherwise."""

    @pytest.mark.parametrize(
        "env",
        [
            {},
            {"ISALSR_STAGE_D_TRACE": "0"},
            {"ISALSR_STAGE_D_TRACE": "false"},
            {"ISALSR_STAGE_D_TRACE": ""},
            {"ISALSR_STAGE_D_TRACE": "no"},
            # Enabled but with no destination: still disabled, not a crash.
            {"ISALSR_STAGE_D_TRACE": "1"},
            {"ISALSR_STAGE_D_TRACE": "1", "ISALSR_STAGE_D_TRACE_DIR": "   "},
        ],
    )
    def test_config_disabled(self, env: dict[str, str]) -> None:
        cfg = StageDTraceConfig.from_env(env)
        assert cfg.enabled is False
        assert cfg.out_dir is None

    def test_no_files_created(self, tmp_path: Path) -> None:
        tracer = StageDTracer.from_env({})
        assert tracer.enabled is False
        _feed(tracer, [_dag_add(), _dag_mul()])
        tracer.close(ledger=FallbackLedger(), run={"method": "bingo"})
        assert list(tmp_path.iterdir()) == []

    def test_hooks_return_false_and_do_nothing(self) -> None:
        tracer = StageDTracer.from_env({})
        assert tracer.begin() is False
        assert tracer.sampling is False
        tracer.note_eval_time(1.0)
        tracer.record(dag=_dag_add(), representation="V+")
        assert tracer.n_seen == 0
        assert tracer.n_sampled == 0

    def test_overhead_is_negligible(self) -> None:
        """A disabled begin/record round trip must stay well under 10 us."""
        tracer = StageDTracer.from_env({})
        n = 20_000
        t0 = time.perf_counter()
        for _ in range(n):
            tracer.begin()
            tracer.record()
        elapsed = (time.perf_counter() - t0) / n
        # Bingo's measured per-candidate cost is ~1.27 ms (EXECUTION-PLAN 11.1);
        # 10 us is a 0.8 % ceiling and the real figure is two orders below it.
        assert elapsed < 1e-5, f"disabled tracer costs {elapsed * 1e6:.2f} us/candidate"


# --------------------------------------------------------------------------- #
# Enabled behaviour
# --------------------------------------------------------------------------- #


class TestSampling:
    """The 1-in-N rule must be deterministic and RNG-free."""

    @pytest.mark.parametrize("rate", [1, 2, 3, 10])
    def test_deterministic_grid(self, tmp_path: Path, rate: int) -> None:
        env = {
            "ISALSR_STAGE_D_TRACE": "1",
            "ISALSR_STAGE_D_TRACE_DIR": str(tmp_path / f"t{rate}"),
            "ISALSR_STAGE_D_TRACE_SAMPLE_RATE": str(rate),
        }
        tracer = StageDTracer.from_env(env)
        taken = [tracer.begin() for _ in range(30)]
        for i, flag in enumerate(taken):
            assert flag is (i % rate == 0)
        assert tracer.n_seen == 30

    @pytest.mark.parametrize("raw", ["0", "-4", "banana", ""])
    def test_bad_rate_falls_back_to_default(self, tmp_path: Path, raw: str) -> None:
        cfg = StageDTraceConfig.from_env(
            {
                "ISALSR_STAGE_D_TRACE": "1",
                "ISALSR_STAGE_D_TRACE_DIR": str(tmp_path),
                "ISALSR_STAGE_D_TRACE_SAMPLE_RATE": raw,
            }
        )
        assert cfg.enabled is True
        assert cfg.sample_rate == 100

    def test_records_only_sampled_candidates(self, tmp_path: Path) -> None:
        env = {
            "ISALSR_STAGE_D_TRACE": "1",
            "ISALSR_STAGE_D_TRACE_DIR": str(tmp_path / "tr"),
            "ISALSR_STAGE_D_TRACE_SAMPLE_RATE": "5",
        }
        tracer = StageDTracer.from_env(env)
        _feed(tracer, [_dag_add()] * 20)
        tracer.close()
        rows = _read_rows(tmp_path / "tr")
        assert [r["i"] for r in rows] == [0, 5, 10, 15]


class TestArtefacts:
    """All five artefacts, with the documented schema."""

    @pytest.fixture
    def traced(self, enabled_env: dict[str, str]) -> Path:
        tracer = StageDTracer.from_env(enabled_env)
        _feed(tracer, [_dag_add(), _dag_mul(), _dag_sin_of_add(), _dag_add()])
        ledger = FallbackLedger()
        tracer.close(ledger=ledger, run={"method": "bingo", "seed": 1, "problem": "Pagie-1"})
        return Path(enabled_env["ISALSR_STAGE_D_TRACE_DIR"])

    def test_all_five_present(self, traced: Path) -> None:
        for name in ARTEFACTS:
            assert (traced / name).is_file(), name

    def test_candidate_schema(self, traced: Path) -> None:
        rows = _read_rows(traced)
        assert len(rows) == 4
        required = {
            "i",
            "k",
            "labels",
            "serialisation",
            "digest_insertion",
            "digest_topological",
            "digest_topological_commutative",
            "canonical",
            "canonical_digest",
            "representation_kind",
            "representation_hash",
            "t_canon_s",
            "t_eval_s",
            "fallback",
            "dedup_hit",
            "violated_post",
        }
        for row in rows:
            assert required <= set(row)

    def test_recorded_keys_match_recomputed(self, traced: Path) -> None:
        dags = [_dag_add(), _dag_mul(), _dag_sin_of_add(), _dag_add()]
        for row, dag in zip(_read_rows(traced), dags, strict=True):
            assert row["serialisation"] == serialise(dag, FixedOrder.INSERTION)
            for order in FixedOrder:
                assert row[f"digest_{order.value}"] == fixed_order_digest(dag, order)
            assert row["canonical"] == fast_canonical_string(dag)
            assert row["canonical_digest"] == canonical_digest(row["canonical"])

    def test_label_multiset_and_k(self, traced: Path) -> None:
        rows = _read_rows(traced)
        assert rows[0]["labels"] == {"ADD": 1, "VAR": 2}
        assert rows[0]["k"] == 1
        assert rows[2]["labels"] == {"ADD": 1, "SIN": 1, "VAR": 2}
        assert rows[2]["k"] == 2

    def test_timings_recorded(self, traced: Path) -> None:
        for row in _read_rows(traced):
            assert row["t_canon_s"] == pytest.approx(5e-4)
            assert row["t_eval_s"] == pytest.approx(1e-4)

    def test_canon_cost_hist_stratified_by_k(self, traced: Path) -> None:
        payload = json.loads((traced / CANON_COST_FILE).read_text())
        assert payload["schema"] == "stage_d_canon_cost_hist/1"
        assert payload["run"] == {"method": "bingo", "seed": 1, "problem": "Pagie-1"}
        assert payload["sample_rate"] == 1
        assert payload["n_sampled"] == 4
        assert set(payload["by_k"]) == {"1", "2"}
        assert payload["by_k"]["1"]["n"] == 3
        assert payload["by_k"]["2"]["n"] == 1
        canon = payload["by_k"]["1"]["t_canon_s"]
        assert canon["n"] == 3
        assert canon["mean_s"] == pytest.approx(5e-4)
        assert sum(canon["bins"]) == 3
        assert len(canon["bins"]) == len(payload["bin_edges_s"]) + 1

    def test_fallback_md_has_five_rates(self, traced: Path) -> None:
        text = (traced / FALLBACK_MD_FILE).read_text()
        for path in FALLBACK_PATHS:
            assert f"| `{path}` |" in text
            assert f"### `{path}`" in text

    def test_stream_size_reports_both_multipliers(self, traced: Path) -> None:
        text = (traced / STREAM_SIZE_FILE).read_text()
        assert "bytes / persisted candidate" in text
        assert "Full-rate counterfactual" in text
        assert "Campaign counterfactual (8,400 runs)" in text
        assert "94,600 inodes" in text
        # D2 is one cell: 5 files, which must be recorded as fitting.
        assert "files for D2 as specified (1 cell) | 5" in text

    def test_stream_size_measures_real_bytes(self, traced: Path) -> None:
        measured = (traced / CANDIDATES_FILE).stat().st_size
        assert f"{measured:,}" in (traced / STREAM_SIZE_FILE).read_text()

    def test_spot_check_clean(self, traced: Path) -> None:
        payload = json.loads((traced / SPOT_CHECK_FILE).read_text())
        assert payload["schema"] == "stage_d_spot_check/1"
        assert payload["n_requested"] == 20
        assert payload["n_checked"] == 4
        assert payload["n_mismatch"] == 0
        assert payload["clean"] is True
        assert payload["replay_engine"] == "python"
        assert payload["production_engine"] in {"cpp", "python"}
        assert all(c["status"] == "match" for c in payload["checks"])

    def test_close_is_idempotent(self, traced: Path) -> None:
        before = {name: (traced / name).read_bytes() for name in ARTEFACTS}
        # A second close (e.g. via atexit) must not truncate or duplicate.
        tracer = StageDTracer(StageDTraceConfig())
        tracer.close()
        after = {name: (traced / name).read_bytes() for name in ARTEFACTS}
        assert before == after


class TestSpotCheckCatchesCorruption:
    """A corrupted recorded canonical must be flagged, loudly."""

    def test_corrupted_canonical_is_caught(
        self,
        enabled_env: dict[str, str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        tracer = StageDTracer.from_env(enabled_env)
        dags = [_dag_add(), _dag_mul(), _dag_sin_of_add()]
        for i, dag in enumerate(dags):
            tracer.begin()
            representation = fast_canonical_string(dag)
            if i == 1:
                # Deliberate corruption: the run "recorded" a string that the
                # production engine did not emit for this DAG.
                representation = representation + "W"
            tracer.record(dag=dag, representation=representation, t_canon=1e-4)

        with caplog.at_level("ERROR"):
            tracer.close()

        out_dir = Path(enabled_env["ISALSR_STAGE_D_TRACE_DIR"])
        payload = json.loads((out_dir / SPOT_CHECK_FILE).read_text())
        assert payload["n_mismatch"] == 1
        assert payload["clean"] is False
        bad = [c for c in payload["checks"] if c["status"] == "mismatch"]
        assert len(bad) == 1
        assert bad[0]["canonical_recorded"] != bad[0]["canonical_replay"]
        assert "Stage-D spot check FAILED" in caplog.text

    def test_undecodable_serialisation_is_an_error_not_a_crash(
        self, enabled_env: dict[str, str]
    ) -> None:
        tracer = StageDTracer.from_env(enabled_env)
        tracer.begin()
        tracer.record(dag=_dag_add(), representation=fast_canonical_string(_dag_add()))
        # Corrupt the reservoir entry's serialisation directly, mimicking a
        # truncated write.
        tracer._reservoir[0]["serialisation"] = "3|garbage"  # noqa: SLF001
        tracer.close()
        out_dir = Path(enabled_env["ISALSR_STAGE_D_TRACE_DIR"])
        payload = json.loads((out_dir / SPOT_CHECK_FILE).read_text())
        assert payload["checks"][0]["status"] == "error"
        assert payload["clean"] is False


class TestFallbackPathsAndExamples:
    """Fallback paths and the worked examples the deliverable requires."""

    @pytest.mark.parametrize("path", ["timeout", "conversion_failure", "canon_raised", "atlas_hit"])
    def test_control_flow_path_gets_a_worked_example(
        self, enabled_env: dict[str, str], path: str
    ) -> None:
        tracer = StageDTracer.from_env(enabled_env)
        tracer.begin()
        tracer.record(dag=_dag_add(), fallback=path)
        tracer.close()
        out_dir = Path(enabled_env["ISALSR_STAGE_D_TRACE_DIR"])
        text = (out_dir / FALLBACK_MD_FILE).read_text()
        section = text.split(f"### `{path}`")[1]
        assert "candidate `i=0`" in section
        assert _read_rows(out_dir)[0]["fallback"] == path

    def test_unknown_path_is_downgraded_not_raised(self, enabled_env: dict[str, str]) -> None:
        tracer = StageDTracer.from_env(enabled_env)
        tracer.begin()
        tracer.record(dag=_dag_add(), fallback="not_a_path")
        tracer.close()
        out_dir = Path(enabled_env["ISALSR_STAGE_D_TRACE_DIR"])
        assert _read_rows(out_dir)[0]["fallback"] == "none"

    def test_violated_post_example_names_the_unreachable_nodes(
        self, enabled_env: dict[str, str]
    ) -> None:
        dag = _dag_orphan_const()
        assert unreachable_nonvar(dag) == [1]
        tracer = StageDTracer.from_env(enabled_env)
        tracer.begin()
        tracer.record(dag=dag)
        tracer.close()
        out_dir = Path(enabled_env["ISALSR_STAGE_D_TRACE_DIR"])
        assert _read_rows(out_dir)[0]["violated_post"] is True
        section = (out_dir / FALLBACK_MD_FILE).read_text().split("### `violated_post`")[1]
        assert "unreachable from any VAR" in section
        assert "`[1]`" in section

    def test_absent_path_says_so_explicitly(self, enabled_env: dict[str, str]) -> None:
        tracer = StageDTracer.from_env(enabled_env)
        _feed(tracer, [_dag_add()])
        tracer.close()
        out_dir = Path(enabled_env["ISALSR_STAGE_D_TRACE_DIR"])
        section = (out_dir / FALLBACK_MD_FILE).read_text().split("### `timeout`")[1]
        assert "No event of this path was observed" in section

    def test_rates_come_from_the_ledger(
        self, enabled_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ISALSR_LEDGER_ENABLED", "1")
        ledger = FallbackLedger()
        ledger.record_pre(_dag_add())
        ledger.record_post(_dag_add())
        ledger.record_timeout(_dag_add())
        tracer = StageDTracer.from_env(enabled_env)
        _feed(tracer, [_dag_add()])
        tracer.close(ledger=ledger)
        out_dir = Path(enabled_env["ISALSR_STAGE_D_TRACE_DIR"])
        text = (out_dir / FALLBACK_MD_FILE).read_text()
        assert "| `timeout` | 1 | 1 |" in text
        assert "- `n_seen` (ledger): 1" in text


class TestEdgeCases:
    """Degenerate inputs must not take the run down."""

    def test_conversion_failure_row_has_no_dag_fields(self, enabled_env: dict[str, str]) -> None:
        tracer = StageDTracer.from_env(enabled_env)
        tracer.begin()
        tracer.record(fallback="conversion_failure")
        tracer.close()
        row = _read_rows(Path(enabled_env["ISALSR_STAGE_D_TRACE_DIR"]))[0]
        assert row["k"] is None
        assert row["serialisation"] is None
        assert row["violated_post"] is None
        assert row["fallback"] == "conversion_failure"

    def test_empty_stream_still_writes_four_artefacts(self, enabled_env: dict[str, str]) -> None:
        tracer = StageDTracer.from_env(enabled_env)
        tracer.close()
        out_dir = Path(enabled_env["ISALSR_STAGE_D_TRACE_DIR"])
        for name in ARTEFACTS:
            assert (out_dir / name).is_file()
        payload = json.loads((out_dir / SPOT_CHECK_FILE).read_text())
        assert payload["n_checked"] == 0
        assert payload["clean"] is False

    def test_non_canonical_arm_records_no_canonical(self, enabled_env: dict[str, str]) -> None:
        dag = _dag_add()
        tracer = StageDTracer.from_env(enabled_env)
        tracer.begin()
        tracer.record(
            dag=dag,
            representation=serialise(dag, FixedOrder.INSERTION),
            representation_kind="hash",
        )
        tracer.close()
        out_dir = Path(enabled_env["ISALSR_STAGE_D_TRACE_DIR"])
        row = _read_rows(out_dir)[0]
        assert row["representation_kind"] == "hash"
        assert row["canonical"] is None
        # The spot check has nothing canonical to verify and says so.
        assert json.loads((out_dir / SPOT_CHECK_FILE).read_text())["n_checked"] == 0

    def test_unwritable_directory_disables_rather_than_raises(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory")
        tracer = StageDTracer.from_env(
            {
                "ISALSR_STAGE_D_TRACE": "1",
                "ISALSR_STAGE_D_TRACE_DIR": str(blocker / "c2_trace"),
            }
        )
        assert tracer.enabled is False
        _feed(tracer, [_dag_add()])
        tracer.close()

    def test_reservoir_draw_is_reproducible(self, tmp_path: Path) -> None:
        dags = [_dag_add(), _dag_mul(), _dag_sin_of_add()] * 20

        def run(where: str) -> list[int]:
            tracer = StageDTracer.from_env(
                {
                    "ISALSR_STAGE_D_TRACE": "1",
                    "ISALSR_STAGE_D_TRACE_DIR": str(tmp_path / where),
                    "ISALSR_STAGE_D_TRACE_SAMPLE_RATE": "1",
                }
            )
            _feed(tracer, dags)
            tracer.close()
            payload = json.loads((tmp_path / where / SPOT_CHECK_FILE).read_text())
            return [c["i"] for c in payload["checks"]]

        assert run("a") == run("b")

    def test_storage_budget_verdict_flips_when_headroom_is_small(self, tmp_path: Path) -> None:
        cfg = StageDTraceConfig(
            enabled=True,
            out_dir=tmp_path / "tight",
            sample_rate=1,
            budget=StorageBudget(inode_headroom=100),
        )
        tracer = StageDTracer(cfg)
        _feed(tracer, [_dag_add()])
        tracer.close()
        text = (tmp_path / "tight" / STREAM_SIZE_FILE).read_text()
        assert "| 42,000 | 100 inodes | EXCEEDS |" in text


def _read_rows(trace_dir: Path) -> list[dict[str, Any]]:
    """Return the parsed records of ``candidates.jsonl``."""
    text = (trace_dir / CANDIDATES_FILE).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
