"""Tests for the host-native shadow cardinality counter.

The ``isalsr`` arm shadows the candidate stream with several distinct-cardinality
sketches.  Three of them key on a fixed order over the *adapter's* output; those
measure the adapter's own renumbering as much as the host's redundancy.  The
fourth, exercised here, keys on the **host's own** structure in the host's own
node order (:mod:`isalsr.baselines.host_native`), which is the baseline the
reduction factor rho must be measured against.

The load-bearing property is refinement: host-native serialisation is sound
(equal serialisation implies equal canonical string), so the host-native
partition of the stream refines the canonical partition and its cardinality is
therefore never smaller.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from experiments.models.schemas import SearchSpaceResults
from isalsr.baselines import FixedOrder
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

HOST_NATIVE_FIELD = "shadow_distinct_host_native"
ADAPTER_ORDER_FIELDS = frozenset(
    {
        "shadow_distinct_insertion",
        "shadow_distinct_topological",
        "shadow_distinct_topological_commutative",
    }
)


def _chain_dag(n_sin: int) -> LabeledDAG:
    """Build ``sin^n_sin(x_0)`` as a LabeledDAG."""
    dag = LabeledDAG(16)
    prev = dag.add_node(NodeType.VAR, var_index=0)
    for _ in range(n_sin):
        nxt = dag.add_node(NodeType.SIN)
        dag.add_edge(prev, nxt)
        prev = nxt
    return dag


def _hll_tolerance(precision: int, n_sigma: float = 3.0) -> float:
    """Return the ``n_sigma`` relative slack of a HyperLogLog of given precision.

    HyperLogLog's asymptotic relative standard error is ``1.04 / sqrt(2**p)``;
    at ``p=16`` that is 0.41 %, so a 3-sigma one-sided margin is 1.22 %.  The
    refinement inequality is exact on the true cardinalities, so the only slack
    the assertion needs is the sketch's own estimation error.
    """
    return n_sigma * 1.04 / math.sqrt(2**precision)


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------


def test_search_space_results_has_host_native_field_defaulting_to_none() -> None:
    """Legacy run logs, which lack the field, must still construct."""
    ss = SearchSpaceResults(
        total_dags_explored=10,
        unique_canonical_dags=5,
        empirical_reduction_factor=2.0,
        max_internal_nodes_seen=3,
        theoretical_reduction_bound=6.0,
        redundancy_rate=0.5,
    )
    assert ss.shadow_distinct_host_native is None


def test_search_space_results_serialises_host_native_field() -> None:
    from dataclasses import asdict

    ss = SearchSpaceResults(
        total_dags_explored=10,
        unique_canonical_dags=5,
        empirical_reduction_factor=2.0,
        max_internal_nodes_seen=3,
        theoretical_reduction_bound=6.0,
        redundancy_rate=0.5,
        shadow_distinct_insertion=8.0,
        shadow_distinct_topological=7.0,
        shadow_distinct_topological_commutative=6.0,
        shadow_distinct_host_native=9.0,
    )
    assert asdict(ss)[HOST_NATIVE_FIELD] == pytest.approx(9.0)


# ----------------------------------------------------------------------
# Gating
# ----------------------------------------------------------------------


@pytest.mark.parametrize("module_name", ["bingo", "udfs"])
def test_host_native_sketch_off_when_shadow_hash_off(module_name: str) -> None:
    mod = pytest.importorskip(f"experiments.models.{module_name}.isalsr_runner")
    dedup = mod._CanonicalDeduplicator(shadow_hash=False)
    dedup.record_shadow(_chain_dag(1), object())
    assert dedup.shadow_counts() == {}


@pytest.mark.parametrize("module_name", ["bingo", "udfs"])
def test_host_native_absent_when_no_host_is_ever_offered(module_name: str) -> None:
    """Without a host object the counter is undefined and must not be reported."""
    mod = pytest.importorskip(f"experiments.models.{module_name}.isalsr_runner")
    dedup = mod._CanonicalDeduplicator(shadow_hash=True)
    for n_sin in range(3):
        dedup.record_shadow(_chain_dag(n_sin))
    # ``n_shadow_failures`` ships alongside the cardinalities (added 2026-08-02
    # so campaign verification reads a field instead of grepping one stderr file
    # per task); it is not one of them.
    assert set(dedup.shadow_counts()) == set(ADAPTER_ORDER_FIELDS) | {"n_shadow_failures"}


@pytest.mark.parametrize("module_name", ["bingo", "udfs"])
def test_host_native_failure_is_counted_and_never_raises(module_name: str) -> None:
    """A host the extractor cannot read must not abort the search."""
    mod = pytest.importorskip(f"experiments.models.{module_name}.isalsr_runner")
    dedup = mod._CanonicalDeduplicator(shadow_hash=True)
    dedup.record_shadow(_chain_dag(1), object())  # not an AGraph / CompGraph
    assert dedup.n_shadow_failures == 1
    assert dedup.shadow_counts()[HOST_NATIVE_FIELD] == pytest.approx(0.0)


@pytest.mark.parametrize("module_name", ["bingo", "udfs"])
def test_host_native_sketch_memory_is_constant_in_stream_length(module_name: str) -> None:
    mod = pytest.importorskip(f"experiments.models.{module_name}.isalsr_runner")
    dedup = mod._CanonicalDeduplicator(shadow_hash=True)
    before = len(dedup._shadow_host_native._registers)
    assert len(dedup._shadow[FixedOrder.TOPOLOGICAL]._registers) == before
    for _ in range(2000):
        dedup.record_shadow(_chain_dag(1), object())
    assert len(dedup._shadow_host_native._registers) == before


# ----------------------------------------------------------------------
# Refinement on a captured stream
# ----------------------------------------------------------------------


def _bingo_stream() -> list[Any]:
    """Return Bingo AGraphs: 6 expressions, each also in a row-shifted copy.

    The shifted copy prepends a non-utilised CONSTANT row.  Bingo drops dead
    rows, so the utilised structure is identical -- same canonical string -- but
    every surviving row index is shifted by one, so the host-native
    serialisation differs.  The stream therefore has 6 canonical classes and 12
    host-native classes.
    """
    from bingo.symbolic_regression.agraph.agraph import AGraph

    def _agraph(rows: list[list[int]]) -> Any:
        ag = AGraph(use_simplification=False)
        ag._command_array = np.array(rows, dtype=int)
        ag._notify_modification()
        return ag

    stream: list[Any] = []
    for n_sin in range(1, 7):
        rows = [[0, 0, 0]]
        for i in range(n_sin):
            rows.append([6, i, i])
        stream.append(_agraph(rows))
        shifted = [[1, 0, 0]] + [[row[0], row[1] + 1, row[2] + 1] for row in rows]
        stream.append(_agraph(shifted))
    return stream


def _udfs_stream() -> list[Any]:
    """Return UDFS CompGraphs: 6 expressions, each also in a key-reordered copy.

    The reordered copy holds the same nodes with the same keys and the same
    children, inserted into ``node_dict`` in a different order.  UDFS's own key
    order therefore differs while the graph -- and hence the canonical string --
    does not.
    """
    from DAG_search.comp_graph import CompGraph

    stream: list[Any] = []
    for n_sin in range(1, 7):
        node_dict: dict[int, tuple[list[int], str]] = {0: ([], "inp")}
        for i in range(1, n_sin + 1):
            node_dict[i] = ([i - 1], "sin")
        stream.append(CompGraph(1, 1, 0, node_dict=dict(node_dict)))
        reordered = dict(reversed(list(node_dict.items())))
        stream.append(CompGraph(1, 1, 0, node_dict=reordered))
    return stream


@pytest.mark.parametrize("module_name", ["bingo", "udfs"])
def test_host_native_cardinality_refines_the_canonical_partition(module_name: str) -> None:
    """``shadow_distinct_host_native >= unique_canonical_dags`` on a real stream.

    Host-native serialisation is sound -- equal serialisation implies an
    identical host graph and hence an identical canonical string -- so the
    host-native partition refines the canonical one and can never be coarser.
    The comparison allows the HyperLogLog estimation error (p=16, relative
    standard error 0.41 %) with a one-sided 3-sigma margin, i.e. 1.22 %.
    """
    mod = pytest.importorskip(f"experiments.models.{module_name}.isalsr_runner")
    if module_name == "bingo":
        pytest.importorskip("bingo")
        from experiments.models.bingo.adapter import agraph_to_labeled_dag as to_dag

        stream = _bingo_stream()
    else:
        pytest.importorskip("DAG_search.comp_graph")
        from experiments.models.udfs.adapter import compgraph_to_labeled_dag as to_dag

        stream = _udfs_stream()

    from isalsr.core.canonical import fast_canonical_string

    dedup = mod._CanonicalDeduplicator(shadow_hash=True)
    canonical_seen: set[str] = set()
    for host in stream:
        dag = to_dag(host)
        dedup.record_shadow(dag, host)
        canonical_seen.add(fast_canonical_string(dag))

    assert dedup.n_shadow_failures == 0
    unique_canonical = len(canonical_seen)
    estimate = dedup.shadow_counts()[HOST_NATIVE_FIELD]
    slack = 1.0 - _hll_tolerance(mod.SHADOW_HLL_PRECISION)
    assert estimate >= unique_canonical * slack, (
        f"host-native estimate {estimate} below the canonical count {unique_canonical}"
    )
    # The stream was built so the host-native partition is strictly finer.
    assert estimate > unique_canonical
