"""Unit tests for experiments.models.fallback_ledger.

Tests cover:
- No-op behaviour when disabled.
- Correct violation detection (violated_pre, violated_post).
- Sampling rate discipline.
- Full-rate O(1) event counters.
- JSON-safety of to_dict().
- Histogram key types and sentinel for conversion_failure.
"""

from __future__ import annotations

import json
import os

import pytest

# Import helpers and the ledger class.
from experiments.models.fallback_ledger import (
    FallbackLedger,
    count_nonvar,
    violates_precondition,
)
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

# ---------------------------------------------------------------------------
# Fixtures — minimal DAGs
# ---------------------------------------------------------------------------


def _make_var_only(m: int = 1) -> LabeledDAG:
    """DAG with m VAR nodes and no other nodes."""
    dag = LabeledDAG(max_nodes=10)
    for i in range(m):
        dag.add_node(NodeType.VAR, var_index=i)
    return dag


def _make_orphan_const() -> LabeledDAG:
    """DAG with one VAR and one CONST with no in-edges (violates precondition)."""
    dag = LabeledDAG(max_nodes=10)
    dag.add_node(NodeType.VAR, var_index=0)  # node 0
    dag.add_node(NodeType.CONST)  # node 1, no edges → unreachable from VAR
    return dag


def _make_connected_const() -> LabeledDAG:
    """DAG with VAR -> CONST edge (precondition satisfied)."""
    dag = LabeledDAG(max_nodes=10)
    dag.add_node(NodeType.VAR, var_index=0)  # node 0
    dag.add_node(NodeType.CONST)  # node 1
    dag.add_edge(0, 1)  # VAR feeds CONST
    return dag


def _make_op_reachable() -> LabeledDAG:
    """VAR -> ADD node (ADD reachable, precondition satisfied)."""
    dag = LabeledDAG(max_nodes=10)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.ADD)  # 1
    dag.add_edge(0, 1)
    return dag


def _make_op_orphan() -> LabeledDAG:
    """ADD node with no inputs (unreachable, violates precondition)."""
    dag = LabeledDAG(max_nodes=10)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.ADD)  # 1, no edge from VAR
    return dag


def _enabled_ledger(**env_overrides: str) -> FallbackLedger:
    """Return a FallbackLedger with ISALSR_LEDGER_ENABLED=1."""
    os.environ["ISALSR_LEDGER_ENABLED"] = "1"
    for k, v in env_overrides.items():
        os.environ[k] = v
    ledger = FallbackLedger()
    return ledger


def _disabled_ledger() -> FallbackLedger:
    os.environ["ISALSR_LEDGER_ENABLED"] = "0"
    return FallbackLedger()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestCountNonvar:
    def test_all_var(self) -> None:
        dag = _make_var_only(3)
        assert count_nonvar(dag) == 0

    def test_one_const(self) -> None:
        dag = _make_orphan_const()
        assert count_nonvar(dag) == 1

    def test_op_and_const(self) -> None:
        dag = LabeledDAG(max_nodes=10)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.CONST)
        dag.add_node(NodeType.ADD)
        assert count_nonvar(dag) == 2


class TestViolatesPrecondition:
    def test_var_only_no_violation(self) -> None:
        # VAR nodes are never "unreachable non-VAR" — they are the sources.
        assert not violates_precondition(_make_var_only())

    def test_orphan_const_violates(self) -> None:
        assert violates_precondition(_make_orphan_const())

    def test_connected_const_ok(self) -> None:
        assert not violates_precondition(_make_connected_const())

    def test_op_reachable_ok(self) -> None:
        assert not violates_precondition(_make_op_reachable())

    def test_op_orphan_violates(self) -> None:
        assert violates_precondition(_make_op_orphan())

    def test_chain_var_const_op(self) -> None:
        """VAR -> CONST -> ADD: ADD is transitively reachable."""
        dag = LabeledDAG(max_nodes=10)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.CONST)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 1)  # VAR -> CONST
        dag.add_edge(1, 2)  # CONST -> ADD  (unusual but valid for this test)
        assert not violates_precondition(dag)

    def test_multi_var_one_unreachable_op(self) -> None:
        """Two VARs, one ADD reachable from VAR-0, one ADD orphan."""
        dag = LabeledDAG(max_nodes=10)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.VAR, var_index=1)  # 1
        dag.add_node(NodeType.ADD)  # 2, connected
        dag.add_node(NodeType.ADD)  # 3, orphan
        dag.add_edge(0, 2)
        assert violates_precondition(dag)


# ---------------------------------------------------------------------------
# Disabled ledger is a strict no-op
# ---------------------------------------------------------------------------


class TestDisabledLedger:
    def test_no_counters_change(self) -> None:
        ledger = _disabled_ledger()
        dag = _make_orphan_const()
        ledger.record_pre(dag)
        ledger.record_post(dag)
        ledger.record_conversion_failure()
        ledger.record_timeout(dag)
        ledger.record_canon_raised(dag)
        ledger.record_atlas_hit(dag)
        assert ledger.n_seen == 0
        assert ledger.n_sampled == 0
        assert ledger.violated_pre == 0
        assert ledger.violated_post == 0
        assert ledger.timeout == 0
        assert ledger.conversion_failure == 0
        assert ledger.canon_raised == 0
        assert ledger.atlas_hit == 0

    def test_to_dict_reflects_disabled(self) -> None:
        ledger = _disabled_ledger()
        d = ledger.to_dict()
        assert d["enabled"] is False
        assert d["n_seen"] == 0


# ---------------------------------------------------------------------------
# record_pre
# ---------------------------------------------------------------------------


class TestRecordPre:
    def test_n_seen_increments(self) -> None:
        ledger = _enabled_ledger()
        dag = _make_var_only()
        ledger.record_pre(dag)
        assert ledger.n_seen == 1

    def test_n_sampled_increments_rate1(self) -> None:
        ledger = _enabled_ledger()
        for _ in range(5):
            ledger.record_pre(_make_var_only())
        assert ledger.n_sampled == 5

    def test_violated_pre_orphan_const(self) -> None:
        ledger = _enabled_ledger()
        ledger.record_pre(_make_orphan_const())
        assert ledger.violated_pre == 1

    def test_violated_pre_connected_const_zero(self) -> None:
        ledger = _enabled_ledger()
        ledger.record_pre(_make_connected_const())
        assert ledger.violated_pre == 0

    def test_histogram_key_is_k(self) -> None:
        ledger = _enabled_ledger()
        # orphan CONST → k=1
        ledger.record_pre(_make_orphan_const())
        assert ledger._hist_violated_pre.get(1, 0) == 1

    def test_multiple_violations_accumulate(self) -> None:
        ledger = _enabled_ledger()
        for _ in range(3):
            ledger.record_pre(_make_orphan_const())
        assert ledger.violated_pre == 3
        assert ledger._hist_violated_pre[1] == 3


# ---------------------------------------------------------------------------
# record_post
# ---------------------------------------------------------------------------


class TestRecordPost:
    def test_post_zero_after_normalization(self) -> None:
        """After _normalize_const_edges, violated_post must be 0."""
        ledger = _enabled_ledger()
        dag = _make_orphan_const()
        # Simulate adapter: record_pre (violation seen), then normalize
        ledger.record_pre(dag)
        assert ledger.violated_pre == 1
        # Normalize: add edge from node 0 (VAR) to node 1 (CONST)
        dag.add_edge(0, 1)
        ledger.record_post(dag)
        assert ledger.violated_post == 0

    def test_post_counts_remaining_violations(self) -> None:
        """If a bug leaves a node unreachable after normalization, post catches it."""
        ledger = _enabled_ledger()
        dag = _make_op_orphan()  # ADD has no edges → violates
        ledger.record_pre(dag)  # increments n_seen
        # Do NOT normalize → post should catch it
        ledger.record_post(dag)
        assert ledger.violated_post == 1

    def test_post_not_called_if_not_sampled(self) -> None:
        """With rate=2, first call is sampled (n_seen=0%2==0), second is not."""
        os.environ["ISALSR_LEDGER_SAMPLE_RATE"] = "2"
        ledger = _enabled_ledger()
        os.environ["ISALSR_LEDGER_SAMPLE_RATE"] = "1"  # restore

        dag0 = _make_orphan_const()
        dag1 = _make_orphan_const()

        # First call (n_seen=0 before increment): sampled
        ledger.record_pre(dag0)
        ledger.record_post(dag0)
        # n_seen=1 now, violated_post=1 (dag0 not normalized)

        # Second call (n_seen=1 before increment, 1%2≠0): NOT sampled
        ledger.record_pre(dag1)
        ledger.record_post(dag1)

        assert ledger.n_sampled == 1
        assert ledger.violated_post == 1  # only one sampled


# ---------------------------------------------------------------------------
# Sampling rate
# ---------------------------------------------------------------------------


class TestSamplingRate:
    @pytest.mark.parametrize("rate", [1, 2, 5, 10])
    def test_n_sampled_matches_rate(self, rate: int) -> None:
        os.environ["ISALSR_LEDGER_SAMPLE_RATE"] = str(rate)
        ledger = _enabled_ledger()
        os.environ["ISALSR_LEDGER_SAMPLE_RATE"] = "1"

        n_calls = 100
        for _ in range(n_calls):
            ledger.record_pre(_make_var_only())
        # Sampled on calls where n_seen (before increment) % rate == 0
        expected = sum(1 for i in range(n_calls) if i % rate == 0)
        assert ledger.n_sampled == expected

    def test_rate1_samples_every_call(self) -> None:
        ledger = _enabled_ledger()
        for _ in range(10):
            ledger.record_pre(_make_var_only())
        assert ledger.n_sampled == 10

    def test_deterministic_no_randomness(self) -> None:
        """Same sequence produces same sample decisions every time."""
        results = []
        for _ in range(3):
            os.environ["ISALSR_LEDGER_SAMPLE_RATE"] = "3"
            ledger = _enabled_ledger()
            os.environ["ISALSR_LEDGER_SAMPLE_RATE"] = "1"
            for _ in range(9):
                ledger.record_pre(_make_orphan_const())
            results.append(ledger.n_sampled)
        assert len(set(results)) == 1


# ---------------------------------------------------------------------------
# Full-rate O(1) events
# ---------------------------------------------------------------------------


class TestFullRateEvents:
    def test_conversion_failure(self) -> None:
        ledger = _enabled_ledger()
        ledger.record_conversion_failure()
        ledger.record_conversion_failure()
        assert ledger.conversion_failure == 2

    def test_conversion_failure_uses_sentinel_k(self) -> None:
        ledger = _enabled_ledger()
        ledger.record_conversion_failure()
        assert -1 in ledger._hist_conversion_failure

    def test_timeout(self) -> None:
        ledger = _enabled_ledger()
        dag = _make_orphan_const()
        ledger.record_timeout(dag)
        assert ledger.timeout == 1
        assert 1 in ledger._hist_timeout  # k=1

    def test_canon_raised(self) -> None:
        ledger = _enabled_ledger()
        dag = _make_op_reachable()
        ledger.record_canon_raised(dag)
        assert ledger.canon_raised == 1
        assert 1 in ledger._hist_canon_raised  # k=1 (one ADD)

    def test_atlas_hit(self) -> None:
        ledger = _enabled_ledger()
        dag = _make_connected_const()
        ledger.record_atlas_hit(dag)
        assert ledger.atlas_hit == 1
        assert 1 in ledger._hist_atlas_hit  # k=1 (one CONST)

    def test_full_rate_counted_regardless_of_sampling(self) -> None:
        """O(1) events accumulate even when rate=100 and no reachability sampling."""
        os.environ["ISALSR_LEDGER_SAMPLE_RATE"] = "100"
        ledger = _enabled_ledger()
        os.environ["ISALSR_LEDGER_SAMPLE_RATE"] = "1"

        dag = _make_orphan_const()
        for _ in range(5):
            ledger.record_timeout(dag)
            ledger.record_conversion_failure()
        assert ledger.timeout == 5
        assert ledger.conversion_failure == 5


# ---------------------------------------------------------------------------
# to_dict serialisation
# ---------------------------------------------------------------------------


class TestToDict:
    def test_json_safe(self) -> None:
        ledger = _enabled_ledger()
        ledger.record_pre(_make_orphan_const())
        ledger.record_post(_make_orphan_const())
        ledger.record_conversion_failure()
        ledger.record_timeout(_make_orphan_const())
        ledger.record_canon_raised(_make_orphan_const())
        ledger.record_atlas_hit(_make_connected_const())
        # Must not raise
        serialised = json.dumps(ledger.to_dict())
        parsed = json.loads(serialised)
        assert parsed["violated_pre"] == ledger.violated_pre

    def test_histogram_keys_are_strings(self) -> None:
        ledger = _enabled_ledger()
        ledger.record_pre(_make_orphan_const())
        d = ledger.to_dict()
        # All histogram dicts must have string keys for JSON safety.
        for key in d:
            if key.endswith("_hist"):
                for k in d[key]:
                    assert isinstance(k, str), f"{key} has non-string key {k!r}"

    def test_required_keys_present(self) -> None:
        ledger = _enabled_ledger()
        d = ledger.to_dict()
        required = {
            "enabled",
            "sample_rate",
            "n_seen",
            "n_sampled",
            "n_sampled_hist",
            "violated_pre",
            "violated_pre_hist",
            "violated_post",
            "violated_post_hist",
            "timeout",
            "timeout_hist",
            "conversion_failure",
            "conversion_failure_hist",
            "canon_raised",
            "canon_raised_hist",
            "atlas_hit",
            "atlas_hit_hist",
        }
        assert required <= set(d.keys())

    def test_disabled_ledger_to_dict(self) -> None:
        ledger = _disabled_ledger()
        d = ledger.to_dict()
        assert d["enabled"] is False
        assert d["n_seen"] == 0
        assert d["violated_pre"] == 0

    def test_conversion_failure_sentinel_in_hist(self) -> None:
        ledger = _enabled_ledger()
        ledger.record_conversion_failure()
        d = ledger.to_dict()
        # Sentinel -1 maps to string "-1" in to_dict
        assert "-1" in d["conversion_failure_hist"]

    def test_scalar_matches_hist_sum(self) -> None:
        """Sum of histogram values equals the scalar counter."""
        ledger = _enabled_ledger()
        for _ in range(3):
            ledger.record_pre(_make_orphan_const())
        for _ in range(2):
            ledger.record_timeout(_make_orphan_const())
        d = ledger.to_dict()
        assert sum(d["violated_pre_hist"].values()) == d["violated_pre"]
        assert sum(d["timeout_hist"].values()) == d["timeout"]


# ---------------------------------------------------------------------------
# n_sampled_hist — per-k denominator
# ---------------------------------------------------------------------------


class TestNSampledHist:
    def test_sum_equals_n_sampled_and_numerator_lte_denominator(self) -> None:
        """At rate=1 with a mix of violating/non-violating DAGs:
        - sum(n_sampled_hist.values()) == n_sampled
        - violated_pre_hist[k] <= n_sampled_hist[k] for every k present
        """
        ledger = _enabled_ledger()

        # k=1: one orphan CONST (violates) + one connected CONST (ok) = 2 sampled
        ledger.record_pre(_make_orphan_const())  # violating, k=1
        ledger.record_pre(_make_connected_const())  # ok,        k=1
        # k=1 from op orphan (ADD is non-VAR): violating, k=1
        ledger.record_pre(_make_op_orphan())  # violating, k=1
        # k=1 from op reachable: ok, k=1
        ledger.record_pre(_make_op_reachable())  # ok,        k=1
        # k=0: VAR-only DAG (never violates)
        ledger.record_pre(_make_var_only())  # ok,        k=0
        ledger.record_pre(_make_var_only(3))  # ok,        k=0

        d = ledger.to_dict()

        # Scalar identity
        assert sum(d["n_sampled_hist"].values()) == d["n_sampled"]

        # Numerator ≤ denominator for every k present in violated_pre_hist
        for k_str, numerator in d["violated_pre_hist"].items():
            denominator = d["n_sampled_hist"].get(k_str, 0)
            assert numerator <= denominator, (
                f"k={k_str}: violated_pre_hist={numerator} > n_sampled_hist={denominator}"
            )

        # Spot-check: k=1 was sampled 4 times, k=0 twice
        assert d["n_sampled_hist"]["1"] == 4
        assert d["n_sampled_hist"]["0"] == 2
        # k=1 violations: orphan_const + op_orphan = 2
        assert d["violated_pre_hist"].get("1", 0) == 2

    def test_denominator_tracks_sampled_subset_not_n_seen(self) -> None:
        """At rate=1000, n_sampled_hist sums to n_sampled, not n_seen."""
        os.environ["ISALSR_LEDGER_SAMPLE_RATE"] = "1000"
        ledger = _enabled_ledger()
        os.environ["ISALSR_LEDGER_SAMPLE_RATE"] = "1"

        # Feed 2000 DAGs: only calls 0 and 1000 are sampled (2 total)
        for _ in range(2000):
            ledger.record_pre(_make_orphan_const())

        assert ledger.n_seen == 2000
        assert ledger.n_sampled == 2  # calls 0 and 1000

        d = ledger.to_dict()
        total_in_hist = sum(d["n_sampled_hist"].values())
        assert total_in_hist == ledger.n_sampled  # 2, not 2000
        assert total_in_hist != ledger.n_seen

    def test_n_sampled_hist_key_in_to_dict(self) -> None:
        """n_sampled_hist appears in to_dict() with string keys."""
        ledger = _enabled_ledger()
        ledger.record_pre(_make_orphan_const())
        d = ledger.to_dict()
        assert "n_sampled_hist" in d
        for k in d["n_sampled_hist"]:
            assert isinstance(k, str), f"n_sampled_hist has non-string key {k!r}"
