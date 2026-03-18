"""Unit tests for CacheEntry, CacheStats dataclasses and the dag_depth helper.

Covers:
- dag_depth: single node, linear chains, diamond, deep chain, multi-variable.
- CacheEntry: frozen immutability, equality, hashability.
- CacheStats: field accessibility.
"""

from __future__ import annotations

import dataclasses

import pytest

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.precomputed.cache_entry import CacheEntry, CacheStats, dag_depth

# ======================================================================
# Helpers
# ======================================================================


def _make_dag(max_nodes: int = 20) -> LabeledDAG:
    """Create an empty LabeledDAG with a given capacity."""
    return LabeledDAG(max_nodes)


def _make_sample_entry(**overrides: object) -> CacheEntry:
    """Create a CacheEntry with sensible defaults, allowing overrides."""
    defaults: dict[str, object] = {
        "raw_string": "Vs",
        "num_variables": 1,
        "n_nodes": 2,
        "n_internal": 1,
        "n_edges": 1,
        "n_var_nodes": 1,
        "depth": 1,
        "greedy_single": "Vs",
        "greedy_min": "Vs",
        "pruned": "Vs",
        "exhaustive": "Vs",
        "exhaustive_timed_out": False,
        "timing_greedy_single": 0.001,
        "timing_greedy_min": 0.002,
        "timing_pruned": 0.003,
        "timing_exhaustive": 0.004,
        "is_canonical": True,
        "exhaustive_eq_pruned": True,
        "greedy_single_eq_exhaustive": True,
        "greedy_min_eq_exhaustive": True,
    }
    defaults.update(overrides)
    return CacheEntry(**defaults)  # type: ignore[arg-type]


# ======================================================================
# dag_depth tests
# ======================================================================


class TestDagDepth:
    """Tests for the dag_depth function (longest-path in a DAG)."""

    def test_single_var_node(self) -> None:
        """A DAG with one VAR node has depth 0 (no edges)."""
        dag = _make_dag()
        dag.add_node(NodeType.VAR, var_index=0)
        assert dag_depth(dag) == 0

    def test_one_edge_x_to_sin(self) -> None:
        """x -> sin has one edge, so depth is 1."""
        dag = _make_dag()
        dag.add_node(NodeType.VAR, var_index=0)  # node 0: x
        dag.add_node(NodeType.SIN)  # node 1: sin
        dag.add_edge(0, 1)
        assert dag_depth(dag) == 1

    def test_chain_of_two_x_sin_exp(self) -> None:
        """x -> sin -> exp is a chain of length 2."""
        dag = _make_dag()
        dag.add_node(NodeType.VAR, var_index=0)  # node 0: x
        dag.add_node(NodeType.SIN)  # node 1: sin
        dag.add_node(NodeType.EXP)  # node 2: exp
        dag.add_edge(0, 1)
        dag.add_edge(1, 2)
        assert dag_depth(dag) == 2

    def test_diamond_dag(self) -> None:
        """Diamond: x -> sin, x -> cos, sin -> add, cos -> add. Depth = 2.

        The longest path is x -> sin -> add (or x -> cos -> add), both length 2.
        """
        dag = _make_dag()
        dag.add_node(NodeType.VAR, var_index=0)  # node 0: x
        dag.add_node(NodeType.SIN)  # node 1: sin
        dag.add_node(NodeType.COS)  # node 2: cos
        dag.add_node(NodeType.ADD)  # node 3: add
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(1, 3)
        dag.add_edge(2, 3)
        assert dag_depth(dag) == 2

    def test_deep_chain(self) -> None:
        """x -> sin -> cos -> exp -> log -> sqrt has depth 5."""
        dag = _make_dag()
        dag.add_node(NodeType.VAR, var_index=0)  # node 0
        dag.add_node(NodeType.SIN)  # node 1
        dag.add_node(NodeType.COS)  # node 2
        dag.add_node(NodeType.EXP)  # node 3
        dag.add_node(NodeType.LOG)  # node 4
        dag.add_node(NodeType.SQRT)  # node 5
        for i in range(5):
            dag.add_edge(i, i + 1)
        assert dag_depth(dag) == 5

    def test_two_variables_to_add(self) -> None:
        """x, y -> add has depth 1 (both paths are length 1)."""
        dag = _make_dag()
        dag.add_node(NodeType.VAR, var_index=0)  # node 0: x
        dag.add_node(NodeType.VAR, var_index=1)  # node 1: y
        dag.add_node(NodeType.ADD)  # node 2: add
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)
        assert dag_depth(dag) == 1

    def test_empty_dag(self) -> None:
        """An empty DAG (no nodes at all) has depth 0."""
        dag = _make_dag()
        assert dag_depth(dag) == 0

    def test_multiple_var_nodes_no_edges(self) -> None:
        """Multiple VAR nodes with no edges: depth 0."""
        dag = _make_dag()
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.VAR, var_index=2)
        assert dag_depth(dag) == 0


# ======================================================================
# CacheEntry tests
# ======================================================================


class TestCacheEntry:
    """Tests for the CacheEntry frozen dataclass."""

    def test_frozen_immutable(self) -> None:
        """Assigning to a field on a frozen dataclass raises FrozenInstanceError."""
        entry = _make_sample_entry()
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.raw_string = "modified"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Two CacheEntry instances with identical fields are equal."""
        entry1 = _make_sample_entry()
        entry2 = _make_sample_entry()
        assert entry1 == entry2

    def test_inequality_on_different_field(self) -> None:
        """Changing one field breaks equality."""
        entry1 = _make_sample_entry()
        entry2 = _make_sample_entry(raw_string="VsNVe")
        assert entry1 != entry2

    def test_hashable(self) -> None:
        """CacheEntry can be added to a set (frozen dataclass is hashable)."""
        entry1 = _make_sample_entry()
        entry2 = _make_sample_entry()
        entry3 = _make_sample_entry(raw_string="VsNVe")
        s = {entry1, entry2, entry3}
        # entry1 and entry2 are identical, so the set should have 2 elements.
        assert len(s) == 2

    def test_all_fields_present(self) -> None:
        """Verify all documented fields are accessible."""
        entry = _make_sample_entry()
        assert entry.raw_string == "Vs"
        assert entry.num_variables == 1
        assert entry.n_nodes == 2
        assert entry.n_internal == 1
        assert entry.n_edges == 1
        assert entry.n_var_nodes == 1
        assert entry.depth == 1
        assert entry.greedy_single == "Vs"
        assert entry.greedy_min == "Vs"
        assert entry.pruned == "Vs"
        assert entry.exhaustive == "Vs"
        assert entry.exhaustive_timed_out is False
        assert entry.timing_greedy_single == 0.001
        assert entry.timing_greedy_min == 0.002
        assert entry.timing_pruned == 0.003
        assert entry.timing_exhaustive == 0.004
        assert entry.is_canonical is True
        assert entry.exhaustive_eq_pruned is True
        assert entry.greedy_single_eq_exhaustive is True
        assert entry.greedy_min_eq_exhaustive is True


# ======================================================================
# CacheStats tests
# ======================================================================


class TestCacheStats:
    """Tests for the CacheStats frozen dataclass."""

    def test_all_fields_accessible(self) -> None:
        """Verify all CacheStats fields can be set and read."""
        stats = CacheStats(
            total_entries=100,
            unique_canonical_pruned=80,
            unique_canonical_exhaustive=75,
            exhaustive_timeout_count=5,
            exhaustive_eq_pruned_count=70,
            greedy_single_eq_exhaustive_count=60,
            greedy_min_eq_exhaustive_count=65,
            avg_depth=2.5,
            max_depth=7,
            avg_internal_nodes=3.2,
        )
        assert stats.total_entries == 100
        assert stats.unique_canonical_pruned == 80
        assert stats.unique_canonical_exhaustive == 75
        assert stats.exhaustive_timeout_count == 5
        assert stats.exhaustive_eq_pruned_count == 70
        assert stats.greedy_single_eq_exhaustive_count == 60
        assert stats.greedy_min_eq_exhaustive_count == 65
        assert stats.avg_depth == 2.5
        assert stats.max_depth == 7
        assert stats.avg_internal_nodes == 3.2

    def test_frozen_immutable(self) -> None:
        """CacheStats is also frozen."""
        stats = CacheStats(
            total_entries=1,
            unique_canonical_pruned=1,
            unique_canonical_exhaustive=1,
            exhaustive_timeout_count=0,
            exhaustive_eq_pruned_count=1,
            greedy_single_eq_exhaustive_count=1,
            greedy_min_eq_exhaustive_count=1,
            avg_depth=1.0,
            max_depth=1,
            avg_internal_nodes=1.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            stats.total_entries = 999  # type: ignore[misc]
