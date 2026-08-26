"""Unit tests for the round-trip property.

The round-trip property is the foundation for Phase 4 (canonical string) and
the paper's central claim: S2D(D2S(D, x_1)) ~ D for all valid labeled DAGs.

Phase 1: Short, manually inspectable strings.
Phase 2: Fixture-based DAG round-trips.
Phase 3: Expression evaluation preservation across round-trip.
"""

from __future__ import annotations

import math

import pytest

from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.string_to_dag import StringToDAG


def _roundtrip_string(string: str, num_variables: int) -> None:
    """Verify the round-trip property for a given IsalSR string.

    S2D(w) -> D1 -> D2S(D1) -> w' -> S2D(w') -> D2
    Assert D1 ~ D2 (labeled DAG isomorphism).
    """
    dag1 = StringToDAG(string, num_variables=num_variables).run()
    # Only round-trip if there are non-VAR nodes (otherwise D2S is trivial).
    if dag1.node_count <= num_variables:
        return  # VAR-only DAG, nothing to encode.
    string2 = DAGToString(dag1).run()
    dag2 = StringToDAG(string2, num_variables=num_variables).run()
    assert dag1.node_count == dag2.node_count, (
        f"Node count mismatch: {dag1.node_count} vs {dag2.node_count} "
        f"for string={string!r}, re-encoded={string2!r}"
    )
    assert dag1.edge_count == dag2.edge_count, (
        f"Edge count mismatch: {dag1.edge_count} vs {dag2.edge_count} "
        f"for string={string!r}, re-encoded={string2!r}"
    )
    assert dag1.is_isomorphic(dag2), (
        f"Round-trip failed for string={string!r}, re-encoded={string2!r}"
    )


def _roundtrip_dag(dag: LabeledDAG, num_variables: int) -> None:
    """Verify the round-trip property for a given DAG.

    D -> D2S(D) -> w -> S2D(w) -> D'
    Assert D ~ D' (labeled DAG isomorphism).
    """
    string = DAGToString(dag).run()
    dag2 = StringToDAG(string, num_variables=num_variables).run()
    assert dag.node_count == dag2.node_count
    assert dag.edge_count == dag2.edge_count
    assert dag.is_isomorphic(dag2), f"Round-trip failed: D2S produced {string!r}"


# ======================================================================
# Phase 1: Short deterministic strings (1 variable)
# ======================================================================


class TestRoundTripPhase1SingleVar:
    """Short strings with 1 variable. Manually inspectable."""

    def test_single_sin(self) -> None:
        _roundtrip_string("Vs", 1)

    def test_single_cos(self) -> None:
        _roundtrip_string("Vc", 1)

    def test_single_exp(self) -> None:
        _roundtrip_string("Ve", 1)

    def test_single_const(self) -> None:
        _roundtrip_string("Vk", 1)

    def test_single_add(self) -> None:
        """ADD with 1 input (topologically valid, semantically incomplete)."""
        _roundtrip_string("V+", 1)

    def test_chain_sin_exp(self) -> None:
        """x -> sin -> exp: VsNVe."""
        _roundtrip_string("VsNVe", 1)

    def test_two_siblings(self) -> None:
        """V+V*: both ADD and MUL from x (pointer immobility)."""
        _roundtrip_string("V+V*", 1)

    def test_chain_three(self) -> None:
        """x -> sin -> cos -> exp."""
        _roundtrip_string("VsNVcNVe", 1)


# ======================================================================
# Phase 1: Short deterministic strings (2 variables)
# ======================================================================


class TestRoundTripPhase1TwoVar:
    """Short strings with 2 variables."""

    def test_x_plus_y(self) -> None:
        """x + y via V+nnNc."""
        _roundtrip_string("V+nnNc", 2)

    def test_sin_x_mul_y(self) -> None:
        """sin(x) * y via VsNV*Nnnnc."""
        _roundtrip_string("VsNV*Nnnnc", 2)

    def test_edge_between_vars(self) -> None:
        """nc: edge y -> x (no operations, just edge)."""
        _roundtrip_string("ncV+", 2)

    def test_two_ops_from_both_vars(self) -> None:
        _roundtrip_string("V+V*", 2)

    def test_noop_and_ops(self) -> None:
        _roundtrip_string("WV+nnNc", 2)


# ======================================================================
# Phase 2: Fixture-based DAG round-trips
# ======================================================================


class TestRoundTripPhase2Fixtures:
    """Round-trip existing fixture DAGs through D2S -> S2D."""

    def test_sin_x_dag(self, sin_x_dag: LabeledDAG) -> None:
        _roundtrip_dag(sin_x_dag, 1)

    def test_x_plus_y_dag(self, x_plus_y_dag: LabeledDAG) -> None:
        _roundtrip_dag(x_plus_y_dag, 2)

    def test_sin_x_mul_y_dag(self, sin_x_mul_y_dag: LabeledDAG) -> None:
        _roundtrip_dag(sin_x_mul_y_dag, 2)


# ======================================================================
# Phase 2: Programmatic DAG round-trips
# ======================================================================


class TestRoundTripPhase2Programmatic:
    """Programmatically constructed DAGs of increasing complexity."""

    def test_x_sub_y(self) -> None:
        """x - y."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.SUB)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)
        _roundtrip_dag(dag, 2)

    def test_x_div_y(self) -> None:
        """x / y."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.DIV)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)
        _roundtrip_dag(dag, 2)

    def test_sin_cos_add(self) -> None:
        """sin(x) + cos(x): diamond shape."""
        dag = LabeledDAG(max_nodes=6)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.SIN)  # 1
        dag.add_node(NodeType.COS)  # 2
        dag.add_node(NodeType.ADD)  # 3
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(1, 3)
        dag.add_edge(2, 3)
        _roundtrip_dag(dag, 1)

    def test_three_var_add(self) -> None:
        """x + y + z: variadic ADD with 3 inputs."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.VAR, var_index=2)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 3)
        dag.add_edge(1, 3)
        dag.add_edge(2, 3)
        _roundtrip_dag(dag, 3)

    def test_nested_mul_add_const(self) -> None:
        """x * y + const: two operations + constant.

        CONST is a leaf node created by V/v, so it must be reachable from
        a VAR node via outgoing edges. We model this as:
        x -> MUL, y -> MUL, x -> CONST (V creates edge x -> CONST),
        MUL -> ADD, CONST -> ADD.
        """
        dag = LabeledDAG(max_nodes=6)
        dag.add_node(NodeType.VAR, var_index=0)  # 0: x
        dag.add_node(NodeType.VAR, var_index=1)  # 1: y
        dag.add_node(NodeType.MUL)  # 2
        dag.add_node(NodeType.CONST, const_value=2.5)  # 3
        dag.add_node(NodeType.ADD)  # 4
        dag.add_edge(0, 2)  # x -> MUL
        dag.add_edge(1, 2)  # y -> MUL
        dag.add_edge(0, 3)  # x -> CONST (reachability path)
        dag.add_edge(2, 4)  # MUL -> ADD
        dag.add_edge(3, 4)  # CONST -> ADD
        _roundtrip_dag(dag, 2)

    def test_pow_expression(self) -> None:
        """x ^ y."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.POW)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)
        _roundtrip_dag(dag, 2)


# ======================================================================
# Phase 3: Evaluation preservation across round-trip
# ======================================================================


class TestRoundTripEvaluation:
    """Numerical evaluation is preserved across D2S -> S2D round-trip.

    This verifies that the round-trip doesn't corrupt the expression
    semantics -- critical for the paper's claim that the canonical
    representation preserves expression equivalence.
    """

    def test_sin_x_evaluation(self, sin_x_dag: LabeledDAG) -> None:
        """sin(x) at x=pi/2 → 1.0, same before and after round-trip."""
        val_before = evaluate_dag(sin_x_dag, {0: math.pi / 2})
        string = DAGToString(sin_x_dag).run()
        dag2 = StringToDAG(string, num_variables=1).run()
        val_after = evaluate_dag(dag2, {0: math.pi / 2})
        assert val_before == pytest.approx(val_after)
        assert val_before == pytest.approx(1.0)

    def test_x_plus_y_evaluation(self, x_plus_y_dag: LabeledDAG) -> None:
        """x + y at x=3, y=7 → 10, preserved across round-trip."""
        val_before = evaluate_dag(x_plus_y_dag, {0: 3.0, 1: 7.0})
        string = DAGToString(x_plus_y_dag).run()
        dag2 = StringToDAG(string, num_variables=2).run()
        val_after = evaluate_dag(dag2, {0: 3.0, 1: 7.0})
        assert val_before == pytest.approx(val_after)
        assert val_before == pytest.approx(10.0)

    def test_sin_x_mul_y_evaluation(self, sin_x_mul_y_dag: LabeledDAG) -> None:
        """sin(x) * y at x=pi/2, y=5 → 5.0, preserved across round-trip."""
        val_before = evaluate_dag(sin_x_mul_y_dag, {0: math.pi / 2, 1: 5.0})
        string = DAGToString(sin_x_mul_y_dag).run()
        dag2 = StringToDAG(string, num_variables=2).run()
        val_after = evaluate_dag(dag2, {0: math.pi / 2, 1: 5.0})
        assert val_before == pytest.approx(val_after)
        assert val_before == pytest.approx(5.0)
