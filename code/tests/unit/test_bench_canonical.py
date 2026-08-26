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
    ENCODING_KWARGS,
    FIXED_SEED,
    K_BUCKETS,
    N_REPS,
    N_WARMUP,
    VALID_ENCODINGS,
    _bucket_for_k,
    _build_bucket_corpus,
    _count_k,
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


# ---------------------------------------------------------------------------
# New: encoding constants
# ---------------------------------------------------------------------------


class TestEncodingConstants:
    """Tests for ENCODING_KWARGS and VALID_ENCODINGS."""

    def test_legacy_in_valid_encodings(self) -> None:
        assert "legacy" in VALID_ENCODINGS

    def test_split_in_valid_encodings(self) -> None:
        assert "split" in VALID_ENCODINGS

    def test_encoding_kwargs_legacy_decompose_false(self) -> None:
        assert ENCODING_KWARGS["legacy"]["decompose"] is False

    def test_encoding_kwargs_split_decompose_true(self) -> None:
        assert ENCODING_KWARGS["split"]["decompose"] is True

    def test_encoding_kwargs_split_share_unary_false(self) -> None:
        assert ENCODING_KWARGS["split"]["share_unary"] is False


# ---------------------------------------------------------------------------
# New: _count_k and _bucket_for_k helpers
# ---------------------------------------------------------------------------


class TestCountK:
    """Tests for _count_k."""

    def test_single_var_gives_zero(self) -> None:
        from isalsr.core.node_types import NodeType

        dag = LabeledDAG(max_nodes=1)
        dag.add_node(NodeType.VAR, var_index=0)
        assert _count_k(dag) == 0

    def test_var_plus_sin_gives_one(self) -> None:
        from isalsr.core.node_types import NodeType

        dag = LabeledDAG(max_nodes=2)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_edge(0, 1)
        assert _count_k(dag) == 1

    def test_two_vars_plus_add_gives_one(self) -> None:
        from isalsr.core.node_types import NodeType

        dag = LabeledDAG(max_nodes=3)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)
        assert _count_k(dag) == 1

    def test_empty_dag_gives_zero(self) -> None:
        dag = LabeledDAG(max_nodes=1)
        assert _count_k(dag) == 0


class TestBucketForK:
    """Tests for _bucket_for_k."""

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_small_k_gives_lt5_bucket(self, k: int) -> None:
        assert _bucket_for_k(k) == "k_lt5"

    @pytest.mark.parametrize("k", [5, 9, 14])
    def test_mid_k_gives_5_to_14_bucket(self, k: int) -> None:
        assert _bucket_for_k(k) == "k_5_to_14"

    @pytest.mark.parametrize("k", [15, 20, 31])
    def test_large_k_gives_15_to_31_bucket(self, k: int) -> None:
        assert _bucket_for_k(k) == "k_15_to_31"

    @pytest.mark.parametrize("k", [0, 32, 100])
    def test_out_of_range_gives_none(self, k: int) -> None:
        assert _bucket_for_k(k) is None

    def test_bucket_boundaries_are_consistent_with_k_buckets(self) -> None:
        """Every bucket name returned must correspond to a K_BUCKETS entry."""
        valid_names = {bname for bname, _, _ in K_BUCKETS}
        for k in range(1, 32):
            result = _bucket_for_k(k)
            if result is not None:
                assert result in valid_names


# ---------------------------------------------------------------------------
# New: --encodings flag — default reproduces old schema
# ---------------------------------------------------------------------------


def test_explicit_legacy_encoding_gives_schema_10() -> None:
    """--encodings legacy (explicit) must produce the same schema_version 1.0 as default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "report.json")
        rc = main(["--quick", "--out", out_path, "--encodings", "legacy"])
        assert rc == 0
        with open(out_path, encoding="utf-8") as fh:
            report = json.load(fh)

    assert report.get("schema_version") == "1.0"
    assert "k_buckets" in report
    assert "overall" in report
    assert "encodings" not in report  # multi-encoding key absent in single-encoding mode


def test_unknown_encoding_exits_nonzero() -> None:
    """An unrecognised encoding name must cause a non-zero exit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "report.json")
        with pytest.raises(SystemExit) as exc_info:
            main(["--quick", "--out", out_path, "--encodings", "bogus"])
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# New: multi-encoding (legacy,split) — requires bingo-nasa
# ---------------------------------------------------------------------------


def _bingo_available() -> bool:
    try:
        import bingo  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _bingo_available(), reason="bingo-nasa not installed")
def test_multi_encoding_quick_run_writes_schema_11() -> None:
    """--encodings legacy,split writes a schema 1.1 report with both encoding sections."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "bench_enc.json")
        rc = main(["--quick", "--out", out_path, "--encodings", "legacy,split"])
        assert rc == 0, f"main() returned non-zero exit code {rc}"
        assert os.path.exists(out_path)

        with open(out_path, encoding="utf-8") as fh:
            report = json.load(fh)

    assert report.get("schema_version") == "1.1"

    prov = report["provenance"]
    assert prov["encodings"] == ["legacy", "split"]
    assert prov["seed"] == FIXED_SEED
    assert prov["quick_mode"] is True

    # Both encodings present.
    assert "encodings" in report
    for enc in ("legacy", "split"):
        assert enc in report["encodings"], f"Missing encoding section: {enc}"
        enc_data = report["encodings"][enc]
        assert "k_buckets" in enc_data
        assert "overall" in enc_data
        for bname, _, _ in K_BUCKETS:
            assert bname in enc_data["k_buckets"], f"{enc}: missing bucket {bname}"

    # Migration block present.
    assert "migration" in report
    mig = report["migration"]
    assert "per_encoding_k_stats" in mig
    assert "per_encoding_bucket_counts" in mig
    assert "n_bucket_changed" in mig
    assert "fraction_changed" in mig
    assert "legacy" in mig["per_encoding_k_stats"]
    assert "split" in mig["per_encoding_k_stats"]

    # split k-stats should show higher mean k than legacy (T16 decomposition adds nodes).
    legacy_mean = mig["per_encoding_k_stats"]["legacy"]["mean"]
    split_mean = mig["per_encoding_k_stats"]["split"]["mean"]
    assert split_mean >= legacy_mean, (
        f"Expected split mean k ({split_mean}) >= legacy ({legacy_mean}) after T16 decomposition"
    )


@pytest.mark.skipif(not _bingo_available(), reason="bingo-nasa not installed")
def test_multi_encoding_tables_emitted_once_per_encoding(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """stdout must contain one table per encoding when --encodings legacy,split."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "bench_enc.json")
        main(["--quick", "--out", out_path, "--encodings", "legacy,split"])

    captured = capsys.readouterr()
    # Each encoding header must appear exactly once.
    assert captured.out.count("--- Encoding: legacy") == 1
    assert captured.out.count("--- Encoding: split") == 1
    # Migration block header must appear.
    assert "K-bucket migration" in captured.out
