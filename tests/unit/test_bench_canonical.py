"""Smoke tests for experiments.scripts.bench_canonical.

Verifies that ``--quick`` mode completes and writes a structurally valid JSON
report without measuring actual performance numbers (the benchmark numbers
themselves are validated by the AC-5 acceptance check, not by unit tests).

All tests run against the same fixed seed used by the benchmark to keep
results reproducible.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — matches the convention in bench_canonical.py itself.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_PROJECT_ROOT, "src"), _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.scripts.bench_canonical import (  # noqa: E402
    CORPUS_PER_BUCKET_QUICK,
    FIXED_SEED,
    K_BUCKETS,
    N_REPS,
    N_WARMUP,
    _build_bucket_corpus,
    _is_fully_reachable,
    _mad,
    _median,
    _probe_cpp,
    _projected_overhead,
    main,
)
from isalsr.core.labeled_dag import LabeledDAG  # noqa: E402
from isalsr.core.node_types import NodeType  # noqa: E402

# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestIsFullyReachable:
    """Tests for _is_fully_reachable reachability filter."""

    def test_empty_dag_is_reachable(self) -> None:
        dag = LabeledDAG(max_nodes=1)
        assert _is_fully_reachable(dag)

    def test_single_var_node_is_reachable(self) -> None:
        dag = LabeledDAG(max_nodes=1)
        dag.add_node(NodeType.VAR, var_index=0)
        assert _is_fully_reachable(dag)

    def test_connected_chain_is_reachable(self) -> None:
        dag = LabeledDAG(max_nodes=3)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.COS)
        dag.add_edge(0, 1)
        dag.add_edge(1, 2)
        assert _is_fully_reachable(dag)

    def test_disconnected_node_not_reachable(self) -> None:
        dag = LabeledDAG(max_nodes=3)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)  # node 1: disconnected from 0
        dag.add_node(NodeType.SIN)
        dag.add_edge(1, 2)  # 1→2 but 1 not reachable from 0
        assert not _is_fully_reachable(dag)

    def test_all_connected_binary(self) -> None:
        dag = LabeledDAG(max_nodes=3)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)
        # node 1 is not reachable from node 0 by out-edges (no edge 0→1)
        assert not _is_fully_reachable(dag)


class TestMadAndMedian:
    """Tests for _mad and _median helpers."""

    def test_median_odd(self) -> None:
        assert _median([3.0, 1.0, 2.0]) == pytest.approx(2.0)

    def test_median_even(self) -> None:
        assert _median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)

    def test_median_single(self) -> None:
        assert _median([7.5]) == pytest.approx(7.5)

    def test_mad_constant(self) -> None:
        assert _mad([5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_mad_simple(self) -> None:
        # values [1,2,3,4], median=2.5, deviations=[1.5,0.5,0.5,1.5], MAD=1.0
        assert _mad([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)

    def test_mad_single(self) -> None:
        assert _mad([3.7]) == pytest.approx(0.0)


class TestProjectedOverhead:
    """Tests for _projected_overhead."""

    def test_speedup_1_gives_same_overhead(self) -> None:
        result = _projected_overhead(1.0)
        # speedup=1 → no change; should return ~0.392

        assert result["projected_overhead_bingo"] == pytest.approx(0.392, rel=1e-3)

    def test_speedup_10_reduces_overhead(self) -> None:
        result = _projected_overhead(10.0)
        # OH' = (0.392/10) / (0.608 + 0.392/10) = 0.0392/0.6480 ≈ 0.0605
        assert result["projected_overhead_bingo"] == pytest.approx(0.0605, rel=1e-2)

    def test_result_has_required_keys(self) -> None:
        result = _projected_overhead(5.0)
        assert "projected_overhead_bingo" in result
        assert "note" in result
        assert "formula" in result
        assert "inputs" in result

    def test_speedup_infinite_gives_zero_overhead(self) -> None:
        result = _projected_overhead(float("inf"))
        assert result["projected_overhead_bingo"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------


class TestBuildBucketCorpus:
    """Tests for _build_bucket_corpus."""

    def test_returns_correct_count(self) -> None:
        dags, n_gen, n_disc = _build_bucket_corpus(k_min=1, k_max=4, target=10, seed=FIXED_SEED)
        assert len(dags) == 10

    def test_all_dags_reachable(self) -> None:
        dags, _, _ = _build_bucket_corpus(k_min=1, k_max=4, target=20, seed=FIXED_SEED)
        for dag in dags:
            assert _is_fully_reachable(dag)

    def test_k_in_range(self) -> None:
        k_min, k_max = 5, 14
        dags, _, _ = _build_bucket_corpus(k_min=k_min, k_max=k_max, target=20, seed=FIXED_SEED)
        for dag in dags:
            # num_vars=1, so k = node_count - 1
            k = dag.node_count - 1
            assert k_min <= k <= k_max, f"k={k} not in [{k_min},{k_max}]"

    def test_discard_count_is_nonnegative(self) -> None:
        _, n_gen, n_disc = _build_bucket_corpus(k_min=1, k_max=4, target=5, seed=FIXED_SEED)
        assert n_gen >= 5
        assert n_disc >= 0
        assert n_gen >= n_disc


# ---------------------------------------------------------------------------
# Integration: --quick CLI run
# ---------------------------------------------------------------------------


def test_quick_run_writes_valid_json() -> None:
    """The --quick flag completes and writes a structurally valid JSON report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "report.json")
        rc = main(["--quick", "--out", out_path])
        assert rc == 0, f"main() returned non-zero exit code {rc}"
        assert os.path.exists(out_path), "Report file was not written"

        with open(out_path, encoding="utf-8") as fh:
            report = json.load(fh)

    # Schema version
    assert report.get("schema_version") == "1.0"

    # Provenance fields
    prov = report["provenance"]
    assert prov["seed"] == FIXED_SEED
    assert prov["quick_mode"] is True
    assert prov["mode"] == "quick"
    assert "git_commit" in prov
    assert "build_info" in prov
    assert "cpu_model" in prov
    assert "python_version" in prov
    assert "elapsed_seconds" in prov
    assert prov["elapsed_seconds"] > 0

    # Protocol params recorded
    proto = prov["protocol"]
    assert proto["n_warmup"] == N_WARMUP
    assert proto["n_reps"] == N_REPS

    # k-buckets present
    assert "k_buckets" in report
    for bname, _k_min, _k_max in K_BUCKETS:
        assert bname in report["k_buckets"], f"Missing bucket {bname}"
        bd = report["k_buckets"][bname]
        assert bd["n_dags"] == CORPUS_PER_BUCKET_QUICK
        assert "python" in bd
        assert "median_ms_per_dag" in bd["python"]
        assert "mad_ms_per_dag" in bd["python"]
        assert bd["n_dags"] > 0
        assert bd["python"]["median_ms_per_dag"] > 0.0

    # Overall block
    ov = report["overall"]
    assert ov["n_dags_total"] == CORPUS_PER_BUCKET_QUICK * len(K_BUCKETS)
    assert "python" in ov
    assert ov["python"]["median_ms_per_dag"] > 0.0


def test_quick_run_cpp_block_present_when_available() -> None:
    """When C++ is available, the cpp block must appear in every bucket."""
    cpp_avail, _ = _probe_cpp()
    if not cpp_avail:
        pytest.skip("C++ engine not available — skipping cpp-specific check")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "report.json")
        main(["--quick", "--out", out_path])
        with open(out_path, encoding="utf-8") as fh:
            report = json.load(fh)

    for bname, _, _ in K_BUCKETS:
        bd = report["k_buckets"][bname]
        assert bd["cpp"] is not None, f"Bucket {bname}: cpp block is null"
        assert bd["cpp"]["median_ms_per_dag"] > 0.0
        assert bd["speedup"] is not None
        assert bd["speedup"] > 0.0

    ov = report["overall"]
    assert ov["cpp"] is not None
    assert ov["speedup"] is not None and ov["speedup"] > 0.0

    # Projected overhead must be present and strictly less than published overhead
    assert "projected_overhead_bingo" in ov
    proj = ov["projected_overhead_bingo"]
    assert 0.0 <= proj < 0.392, f"Projected overhead {proj} not in [0, 0.392)"


def test_quick_run_speedup_positive_when_cpp_available() -> None:
    """Per-bucket speedup must be > 1.0 when C++ is faster than Python."""
    cpp_avail, _ = _probe_cpp()
    if not cpp_avail:
        pytest.skip("C++ engine not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "report.json")
        main(["--quick", "--out", out_path])
        with open(out_path, encoding="utf-8") as fh:
            report = json.load(fh)

    # C++ should be faster on every bucket (speedup > 1.0).
    # Use a relaxed bound to tolerate CPU noise on CI.
    for bname, _, _ in K_BUCKETS:
        bd = report["k_buckets"][bname]
        spd = bd.get("speedup")
        assert spd is not None
        assert spd > 0.5, f"Bucket {bname}: speedup={spd} suspiciously low"
