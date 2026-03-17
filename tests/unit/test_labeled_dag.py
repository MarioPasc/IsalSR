"""Unit tests for LabeledDAG.

Tests cover: node/edge operations, cycle detection, topological sort,
label-aware isomorphism, backtracking support, and expression DAGs.
These tests are foundational for the IsalSR canonical string invariant --
the DAG must correctly enforce acyclicity (DAG constraint) and support
label-aware isomorphism (needed to verify the O(k!) reduction claim).
"""

from __future__ import annotations

import pytest

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType


class TestLabeledDAGBasics:
    """Basic node/edge operations."""

    def test_empty_dag(self, empty_dag: LabeledDAG) -> None:
        assert empty_dag.node_count == 0
        assert empty_dag.edge_count == 0

    def test_add_var_node(self, empty_dag: LabeledDAG) -> None:
        node_id = empty_dag.add_node(NodeType.VAR, var_index=0)
        assert node_id == 0
        assert empty_dag.node_count == 1
        assert empty_dag.node_label(0) == NodeType.VAR
        assert empty_dag.node_data(0)["var_index"] == 0

    def test_add_multiple_nodes(self, empty_dag: LabeledDAG) -> None:
        empty_dag.add_node(NodeType.VAR, var_index=0)
        empty_dag.add_node(NodeType.VAR, var_index=1)
        empty_dag.add_node(NodeType.ADD)
        assert empty_dag.node_count == 3
        assert empty_dag.node_label(2) == NodeType.ADD

    def test_add_node_overflow(self) -> None:
        dag = LabeledDAG(max_nodes=2)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        with pytest.raises(RuntimeError, match="Maximum"):
            dag.add_node(NodeType.ADD)

    def test_const_node_data(self, empty_dag: LabeledDAG) -> None:
        empty_dag.add_node(NodeType.CONST, const_value=3.14)
        assert empty_dag.node_data(0)["const_value"] == pytest.approx(3.14)

    def test_edge_count_init_is_zero(self) -> None:
        """Regression: original IsalGraph code initialized _edge_count to 1 (B1)."""
        dag = LabeledDAG(max_nodes=5)
        assert dag.edge_count == 0


class TestLabeledDAGEdges:
    """Directed edge operations."""

    def test_add_edge_basic(self, two_var_dag: LabeledDAG) -> None:
        """Add edge x -> y (data flows from x to y)."""
        result = two_var_dag.add_edge(0, 1)
        assert result is True
        assert two_var_dag.edge_count == 1
        assert two_var_dag.has_edge(0, 1)
        assert not two_var_dag.has_edge(1, 0)  # directed

    def test_out_neighbors(self, two_var_dag: LabeledDAG) -> None:
        two_var_dag.add_edge(0, 1)
        assert two_var_dag.out_neighbors(0) == frozenset({1})
        assert two_var_dag.out_neighbors(1) == frozenset()

    def test_in_neighbors(self, two_var_dag: LabeledDAG) -> None:
        two_var_dag.add_edge(0, 1)
        assert two_var_dag.in_neighbors(1) == frozenset({0})
        assert two_var_dag.in_neighbors(0) == frozenset()

    def test_degree(self, two_var_dag: LabeledDAG) -> None:
        two_var_dag.add_edge(0, 1)
        assert two_var_dag.out_degree(0) == 1
        assert two_var_dag.in_degree(0) == 0
        assert two_var_dag.out_degree(1) == 0
        assert two_var_dag.in_degree(1) == 1

    def test_duplicate_edge_returns_false(self, two_var_dag: LabeledDAG) -> None:
        two_var_dag.add_edge(0, 1)
        result = two_var_dag.add_edge(0, 1)
        assert result is False
        assert two_var_dag.edge_count == 1

    def test_invalid_node_raises(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        with pytest.raises(IndexError):
            dag.add_edge(0, 1)  # node 1 doesn't exist

    def test_remove_edge(self, two_var_dag: LabeledDAG) -> None:
        two_var_dag.add_edge(0, 1)
        result = two_var_dag.remove_edge(0, 1)
        assert result is True
        assert two_var_dag.edge_count == 0
        assert not two_var_dag.has_edge(0, 1)

    def test_remove_nonexistent_edge(self, two_var_dag: LabeledDAG) -> None:
        result = two_var_dag.remove_edge(0, 1)
        assert result is False


class TestLabeledDAGCycleDetection:
    """Cycle detection -- the DAG constraint.

    This is critical for IsalSR: C/c instructions must silently skip
    edge insertions that would create cycles. add_edge returns False
    for cycle-creating edges.
    """

    def test_self_loop_rejected(self, single_var_dag: LabeledDAG) -> None:
        """Self-loops always create cycles in a DAG."""
        result = single_var_dag.add_edge(0, 0)
        assert result is False
        assert single_var_dag.edge_count == 0

    def test_direct_cycle_rejected(self) -> None:
        """A -> B -> A would create a cycle."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 1)  # 0 -> 1 (ok)
        result = dag.add_edge(1, 0)  # 1 -> 0 would create cycle
        assert result is False
        assert dag.edge_count == 1

    def test_indirect_cycle_rejected(self) -> None:
        """A -> B -> C -> A would create a cycle (length 3)."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.ADD)
        dag.add_node(NodeType.SIN)
        dag.add_edge(0, 1)  # 0 -> 1
        dag.add_edge(1, 2)  # 1 -> 2
        result = dag.add_edge(2, 0)  # 2 -> 0 would create cycle
        assert result is False
        assert dag.edge_count == 2

    def test_non_cycle_edge_accepted(self) -> None:
        """A -> C and B -> C: no cycle (diamond top half)."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.VAR, var_index=1)  # 1
        dag.add_node(NodeType.ADD)  # 2
        assert dag.add_edge(0, 2) is True
        assert dag.add_edge(1, 2) is True
        assert dag.edge_count == 2

    def test_has_cycle_if_added(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 1)
        assert dag.has_cycle_if_added(1, 0) is True
        assert dag.has_cycle_if_added(0, 1) is False  # already exists but not a cycle
        assert dag.has_cycle_if_added(0, 0) is True  # self-loop

    def test_v_instruction_never_creates_cycle(self) -> None:
        """V/v creates edge from existing node to NEW node -- can never cycle.

        Simulates: V+ from x creates ADD(node 1), edge x->ADD.
        New node has no outgoing edges, so path ADD->x is impossible.
        """
        dag = LabeledDAG(max_nodes=5)
        x = dag.add_node(NodeType.VAR, var_index=0)
        add = dag.add_node(NodeType.ADD)
        # New node (add) has no outgoing edges yet, so this always succeeds.
        assert dag.has_cycle_if_added(x, add) is False
        result = dag.add_edge(x, add)
        assert result is True


class TestLabeledDAGTopologicalSort:
    """Topological sort via Kahn's algorithm."""

    def test_single_node(self, single_var_dag: LabeledDAG) -> None:
        order = single_var_dag.topological_sort()
        assert order == [0]

    def test_linear_chain(self) -> None:
        """x -> sin -> exp: topological order is [x, sin, exp]."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.EXP)
        dag.add_edge(0, 1)
        dag.add_edge(1, 2)
        order = dag.topological_sort()
        assert order.index(0) < order.index(1) < order.index(2)

    def test_x_plus_y(self, x_plus_y_dag: LabeledDAG) -> None:
        """x, y -> + : both vars before ADD in topological order."""
        order = x_plus_y_dag.topological_sort()
        add_idx = order.index(2)
        assert order.index(0) < add_idx  # x before +
        assert order.index(1) < add_idx  # y before +

    def test_expression_dag(self, sin_x_mul_y_dag: LabeledDAG) -> None:
        """sin(x) * y: topological order respects all edges."""
        order = sin_x_mul_y_dag.topological_sort()
        # x(0) -> sin(2), y(1), sin(2) -> mul(3), y(1) -> mul(3)
        assert order.index(0) < order.index(2)  # x before sin
        assert order.index(2) < order.index(3)  # sin before mul
        assert order.index(1) < order.index(3)  # y before mul


class TestLabeledDAGOutputNode:
    """output_node() finds the expression root (unique non-VAR sink)."""

    def test_x_plus_y(self, x_plus_y_dag: LabeledDAG) -> None:
        assert x_plus_y_dag.output_node() == 2  # ADD node

    def test_sin_x(self, sin_x_dag: LabeledDAG) -> None:
        assert sin_x_dag.output_node() == 1  # SIN node

    def test_sin_x_mul_y(self, sin_x_mul_y_dag: LabeledDAG) -> None:
        assert sin_x_mul_y_dag.output_node() == 3  # MUL node

    def test_no_non_var_nodes_raises(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        with pytest.raises(ValueError, match="No non-VAR"):
            dag.output_node()


class TestLabeledDAGBacktracking:
    """undo_node and remove_edge for canonical search backtracking."""

    def test_undo_node_removes_last(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 1)
        dag.undo_node()
        assert dag.node_count == 1
        assert dag.edge_count == 0

    def test_undo_node_clears_edges(self) -> None:
        """undo_node removes all incident edges of the last node."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)
        dag.undo_node()
        assert dag.node_count == 2
        assert dag.edge_count == 0
        assert dag.out_degree(0) == 0
        assert dag.out_degree(1) == 0

    def test_undo_empty_is_noop(self, empty_dag: LabeledDAG) -> None:
        empty_dag.undo_node()  # should not raise
        assert empty_dag.node_count == 0


class TestLabeledDAGIsomorphism:
    """Label-aware backtracking isomorphism.

    This tests the fundamental property needed for the canonical string
    invariant: two labeled DAGs are isomorphic iff they represent the
    same expression up to internal node renumbering.
    """

    def test_identical_dags(self) -> None:
        """Two identical x+y DAGs are isomorphic."""
        dag1 = LabeledDAG(max_nodes=5)
        dag2 = LabeledDAG(max_nodes=5)
        for dag in (dag1, dag2):
            dag.add_node(NodeType.VAR, var_index=0)
            dag.add_node(NodeType.VAR, var_index=1)
            dag.add_node(NodeType.ADD)
            dag.add_edge(0, 2)
            dag.add_edge(1, 2)
        assert dag1.is_isomorphic(dag2)

    def test_different_labels_not_isomorphic(self) -> None:
        """x+y vs x*y: same structure but different labels -> not isomorphic."""
        dag1 = LabeledDAG(max_nodes=5)
        dag1.add_node(NodeType.VAR, var_index=0)
        dag1.add_node(NodeType.VAR, var_index=1)
        dag1.add_node(NodeType.ADD)
        dag1.add_edge(0, 2)
        dag1.add_edge(1, 2)

        dag2 = LabeledDAG(max_nodes=5)
        dag2.add_node(NodeType.VAR, var_index=0)
        dag2.add_node(NodeType.VAR, var_index=1)
        dag2.add_node(NodeType.MUL)
        dag2.add_edge(0, 2)
        dag2.add_edge(1, 2)

        assert not dag1.is_isomorphic(dag2)

    def test_different_node_count(self) -> None:
        dag1 = LabeledDAG(max_nodes=5)
        dag1.add_node(NodeType.VAR, var_index=0)

        dag2 = LabeledDAG(max_nodes=5)
        dag2.add_node(NodeType.VAR, var_index=0)
        dag2.add_node(NodeType.SIN)
        dag2.add_edge(0, 1)

        assert not dag1.is_isomorphic(dag2)

    def test_var_nodes_are_distinguishable(self) -> None:
        """VAR nodes are matched by var_index (x_1 maps to x_1, not x_2).

        This is crucial for IsalSR: input variables are pre-numbered and
        distinguishable, which eliminates isomorphism ambiguity over variables.
        """
        # dag1: x_1 -> ADD, x_2 -> SIN
        dag1 = LabeledDAG(max_nodes=5)
        dag1.add_node(NodeType.VAR, var_index=0)
        dag1.add_node(NodeType.VAR, var_index=1)
        dag1.add_node(NodeType.ADD)
        dag1.add_node(NodeType.SIN)
        dag1.add_edge(0, 2)
        dag1.add_edge(1, 3)

        # dag2: x_1 -> SIN, x_2 -> ADD (swapped assignments)
        dag2 = LabeledDAG(max_nodes=5)
        dag2.add_node(NodeType.VAR, var_index=0)
        dag2.add_node(NodeType.VAR, var_index=1)
        dag2.add_node(NodeType.SIN)
        dag2.add_node(NodeType.ADD)
        dag2.add_edge(0, 2)
        dag2.add_edge(1, 3)

        # NOT isomorphic because VAR nodes are fixed: x_1 goes to ADD in dag1 but SIN in dag2.
        assert not dag1.is_isomorphic(dag2)

    def test_relabeled_internal_nodes_isomorphic(self) -> None:
        """Two DAGs with internal nodes renumbered should be isomorphic.

        dag1: x(0) -> sin(1) -> mul(2), y(3) -> mul(2)  [sin=1, mul=2, y=3]
        dag2: x(0) -> mul(1), y(2) -> sin(3) -> mul(1)  [mul=1, y=2, sin=3]

        Wait -- this doesn't work because VAR nodes have fixed var_index.
        Let me construct a proper case with 2 internal nodes swapped.
        """
        # dag1: x(0), y(1), ADD(2), SIN(3), ADD feeds SIN
        dag1 = LabeledDAG(max_nodes=5)
        dag1.add_node(NodeType.VAR, var_index=0)  # 0
        dag1.add_node(NodeType.VAR, var_index=1)  # 1
        dag1.add_node(NodeType.ADD)  # 2
        dag1.add_node(NodeType.SIN)  # 3
        dag1.add_edge(0, 2)  # x -> ADD
        dag1.add_edge(1, 2)  # y -> ADD
        dag1.add_edge(2, 3)  # ADD -> SIN

        # dag2: x(0), y(1), SIN(2), ADD(3), ADD feeds SIN
        # Internal nodes have different IDs but same structure
        dag2 = LabeledDAG(max_nodes=5)
        dag2.add_node(NodeType.VAR, var_index=0)  # 0
        dag2.add_node(NodeType.VAR, var_index=1)  # 1
        dag2.add_node(NodeType.SIN)  # 2 -- was 3 in dag1
        dag2.add_node(NodeType.ADD)  # 3 -- was 2 in dag1
        dag2.add_edge(0, 3)  # x -> ADD
        dag2.add_edge(1, 3)  # y -> ADD
        dag2.add_edge(3, 2)  # ADD -> SIN

        assert dag1.is_isomorphic(dag2)

    def test_empty_dags_isomorphic(self) -> None:
        dag1 = LabeledDAG(max_nodes=0)
        dag2 = LabeledDAG(max_nodes=0)
        assert dag1.is_isomorphic(dag2)

    def test_not_isomorphic_with_non_dag(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        assert not dag.is_isomorphic("not a dag")  # type: ignore[arg-type]


class TestLabeledDAGHelpers:
    """Helper methods."""

    def test_var_nodes(self, sin_x_mul_y_dag: LabeledDAG) -> None:
        assert sin_x_mul_y_dag.var_nodes() == [0, 1]

    def test_non_var_nodes(self, sin_x_mul_y_dag: LabeledDAG) -> None:
        assert sin_x_mul_y_dag.non_var_nodes() == [2, 3]

    def test_invalid_node_raises(self, empty_dag: LabeledDAG) -> None:
        with pytest.raises(IndexError):
            empty_dag.node_label(0)

    def test_negative_node_raises(self, single_var_dag: LabeledDAG) -> None:
        with pytest.raises(IndexError):
            single_var_dag.node_label(-1)


class TestLabeledDAGRepr:
    """String representation."""

    def test_repr_empty(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        r = repr(dag)
        assert "LabeledDAG" in r
        assert "nodes=0" in r
        assert "edges=0" in r

    def test_repr_with_nodes(self, x_plus_y_dag: LabeledDAG) -> None:
        r = repr(x_plus_y_dag)
        assert "nodes=3" in r
        assert "edges=2" in r
        assert "ADD=1" in r
        assert "VAR=2" in r
