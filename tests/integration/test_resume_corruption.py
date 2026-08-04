"""Resume and idempotency of the orchestrator (check B8).

B8 asks for three behaviours to be *observed*, not assumed:

1. a fresh cell runs and writes a valid ``run_log.json``;
2. re-invoking the same cell **skips** it;
3. a cell whose ``run_log.json`` is corrupt is **detected, deleted and re-run**.

The corruption modelled here is the one that actually happens on Picasso: an OOM
or a wallclock kill lands mid-``json.dump``, leaving a syntactically truncated
file that ``Path.exists()`` cannot distinguish from a good one. The orchestrator
therefore validates content rather than existence
(``orchestrator.run_experiment``, the resume guard).

These tests drive the real orchestrator end-to-end on one Nguyen-1 cell with a
5-second search budget. They are integration tests because the skip/re-run
decision is a property of the orchestrator loop, not of any single function.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

pytest.importorskip("numpy")
pytest.importorskip("bingo")

from experiments.models import orchestrator  # noqa: E402

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CONFIG = os.path.join(_REPO, "experiments", "configs", "bingo_nguyen.yaml")
_PROBLEM = "Nguyen-1"
_SEED = 1
_VARIANT = "baseline"
_MAX_TIME = 5.0


def _invoke(output_dir: str) -> int:
    """Run one orchestrator cell, returning its exit status.

    Parameters
    ----------
    output_dir
        Results root for this invocation.

    Returns
    -------
    int
        The orchestrator's return code.
    """
    args = orchestrator.build_parser().parse_args(
        [
            "--config",
            _CONFIG,
            "--output-dir",
            output_dir,
            "--seeds",
            str(_SEED),
            "--problems",
            _PROBLEM,
            "--variants",
            _VARIANT,
            "--max-time",
            str(_MAX_TIME),
            "--postprocess",
            "skip",
        ]
    )
    return orchestrator.run_experiment(_CONFIG, args)


def _run_log_path(output_dir: str) -> str:
    """Locate the single ``run_log.json`` beneath a results root."""
    hits = []
    for root, _dirs, files in os.walk(output_dir):
        if "run_log.json" in files:
            hits.append(os.path.join(root, "run_log.json"))
    assert len(hits) == 1, f"expected exactly one run_log.json, found {hits}"
    return hits[0]


@pytest.fixture(scope="module")
def completed_cell(tmp_path_factory) -> str:
    """A results root holding one completed Nguyen-1 baseline cell."""
    out = str(tmp_path_factory.mktemp("b8_resume"))
    assert _invoke(out) == 0
    return out


# ---------------------------------------------------------------------------
# 1. Fresh run
# ---------------------------------------------------------------------------


def test_fresh_run_writes_valid_run_log(completed_cell: str) -> None:
    path = _run_log_path(completed_cell)
    payload = json.loads(open(path).read())
    assert payload["metadata"]["problem"] == _PROBLEM
    assert payload["metadata"]["seed"] == _SEED
    assert os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# 2. Re-submit skips an intact cell
# ---------------------------------------------------------------------------


def test_intact_cell_is_skipped(completed_cell: str, caplog) -> None:
    """An intact completed cell must be skipped, not silently re-run."""
    path = _run_log_path(completed_cell)
    before_mtime = os.path.getmtime(path)
    before_bytes = open(path, "rb").read()

    with caplog.at_level("INFO"):
        assert _invoke(completed_cell) == 0

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "Skipping" in messages, f"no skip logged; got: {messages[-500:]}"
    assert "Corrupt" not in messages
    assert os.path.getmtime(path) == before_mtime, "skipped cell was rewritten"
    assert open(path, "rb").read() == before_bytes


def test_skip_does_not_depend_on_a_second_process(completed_cell: str) -> None:
    """Idempotency holds across repeated invocations, not just the second."""
    path = _run_log_path(completed_cell)
    mtime = os.path.getmtime(path)
    for _ in range(2):
        assert _invoke(completed_cell) == 0
    assert os.path.getmtime(path) == mtime


# ---------------------------------------------------------------------------
# 3. Corruption is detected, deleted and re-run
# ---------------------------------------------------------------------------


def _truncate_mid_json(path: str) -> int:
    """Truncate a file to half its length, emulating a kill mid-write.

    Returns
    -------
    int
        The truncated size in bytes.
    """
    raw = open(path, "rb").read()
    half = len(raw) // 2
    with open(path, "wb") as handle:
        handle.write(raw[:half])
    return half


def test_truncated_run_log_is_detected_deleted_and_rerun(tmp_path_factory, caplog) -> None:
    out = str(tmp_path_factory.mktemp("b8_corrupt"))
    assert _invoke(out) == 0
    path = _run_log_path(out)
    good_size = os.path.getsize(path)

    truncated_size = _truncate_mid_json(path)
    assert truncated_size < good_size
    with pytest.raises(json.JSONDecodeError):
        json.loads(open(path).read())

    with caplog.at_level("WARNING"):
        assert _invoke(out) == 0

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "Corrupt run_log detected" in messages, f"no detection logged: {messages[-500:]}"

    # Re-run, not merely repaired in place: the file parses again and is whole.
    payload = json.loads(open(path).read())
    assert payload["metadata"]["problem"] == _PROBLEM
    assert os.path.getsize(path) > truncated_size


def test_empty_run_log_is_treated_as_corrupt(tmp_path_factory, caplog) -> None:
    """A zero-byte run_log (the classic OOM artefact) must not count as done."""
    out = str(tmp_path_factory.mktemp("b8_empty"))
    assert _invoke(out) == 0
    path = _run_log_path(out)
    open(path, "wb").close()
    assert os.path.getsize(path) == 0

    with caplog.at_level("WARNING"):
        assert _invoke(out) == 0

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "Corrupt run_log detected" in messages
    assert json.loads(open(path).read())["metadata"]["problem"] == _PROBLEM


def test_structurally_valid_but_schema_wrong_is_rejected(tmp_path_factory, caplog) -> None:
    """Valid JSON that is not a RunLog must also trigger the re-run path.

    This is the case ``Path.exists()`` and a bare ``json.loads`` would both wave
    through; only content validation catches it.
    """
    out = str(tmp_path_factory.mktemp("b8_schema"))
    assert _invoke(out) == 0
    path = _run_log_path(out)
    with open(path, "w") as handle:
        json.dump({"not": "a run log"}, handle)

    with caplog.at_level("WARNING"):
        assert _invoke(out) == 0

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "Corrupt run_log detected" in messages
    assert json.loads(open(path).read())["metadata"]["problem"] == _PROBLEM
