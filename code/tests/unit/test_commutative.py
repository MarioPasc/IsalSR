"""Unit tests for isalsr.core.commutative -- to_commutative / from_commutative.

Tests verify:
1. Structural correctness of the transformed DAGs.
2. Semantic preservation: evaluate_dag produces identical results.
3. Round-trip: from_commutative(to_commutative(D)) is isomorphic to D.
4. Edge cases: no SUB/DIV, nested SUB/DIV, multiple SUB/DIV, mixed ops.
5. POW preservation (non-commutative binary op that is NOT decomposed).
6. from_commutative pattern matching correctness and non-matching cases.
"""

from __future__ import annotations

import pytest

from isalsr.core.commutative import from_commutative, to_commutative
from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

# ======================================================================
# Helper: build common test DAGs
# ======================================================================


def _make_sub_dag() -> LabeledDAG:
    """Build DAG for x_0 - x_1 (SUB with 2 VAR inputs).

    Structure:
        node 0: VAR(x_0)
        node 1: VAR(x_1)
        node 2: SUB  with ordered_inputs [0, 1]  (x_0 - x_1)
    """
    dag = LabeledDAG(4)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.VAR, var_index=1)  # 1
    dag.add_node(NodeType.SUB)  # 2
    dag.add_edge(0, 2)  # first operand: x_0
    dag.add_edge(1, 2)  # second operand: x_1
    return dag


def _make_div_dag() -> LabeledDAG:
    """Build DAG for x_0 / x_1 (DIV with 2 VAR inputs).

    Structure:
        node 0: VAR(x_0)
        node 1: VAR(x_1)
        node 2: DIV  with ordered_inputs [0, 1]  (x_0 / x_1)
    """
    dag = LabeledDAG(4)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.VAR, var_index=1)  # 1
    dag.add_node(NodeType.DIV)  # 2
    dag.add_edge(0, 2)  # first operand: x_0
    dag.add_edge(1, 2)  # second operand: x_1
    return dag


def _make_add_dag() -> LabeledDAG:
    """Build DAG for x_0 + x_1 (ADD, no SUB/DIV)."""
    dag = LabeledDAG(4)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.VAR, var_index=1)  # 1
    dag.add_node(NodeType.ADD)  # 2
    dag.add_edge(0, 2)
    dag.add_edge(1, 2)
    return dag


def _make_pow_dag() -> LabeledDAG:
    """Build DAG for x_0 ^ x_1 (POW, non-commutative binary, not decomposed)."""
    dag = LabeledDAG(4)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.VAR, var_index=1)  # 1
    dag.add_node(NodeType.POW)  # 2
    dag.add_edge(0, 2)  # base
    dag.add_edge(1, 2)  # exponent
    return dag


def _make_nested_sub_div_dag() -> LabeledDAG:
    """Build DAG for (x_0 - x_1) / x_2.

    Structure:
        node 0: VAR(x_0)
        node 1: VAR(x_1)
        node 2: VAR(x_2)
        node 3: SUB  ordered_inputs [0, 1]
        node 4: DIV  ordered_inputs [3, 2]
    """
    dag = LabeledDAG(6)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.VAR, var_index=1)  # 1
    dag.add_node(NodeType.VAR, var_index=2)  # 2
    dag.add_node(NodeType.SUB)  # 3
    dag.add_edge(0, 3)
    dag.add_edge(1, 3)
    dag.add_node(NodeType.DIV)  # 4
    dag.add_edge(3, 4)
    dag.add_edge(2, 4)
    return dag


def _make_sin_sub_dag() -> LabeledDAG:
    """Build DAG for sin(x_0) - x_1.

    Structure:
        node 0: VAR(x_0)
        node 1: VAR(x_1)
        node 2: SIN  input [0]
        node 3: SUB  ordered_inputs [2, 1]
    """
    dag = LabeledDAG(5)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.VAR, var_index=1)  # 1
    dag.add_node(NodeType.SIN)  # 2
    dag.add_edge(0, 2)
    dag.add_node(NodeType.SUB)  # 3
    dag.add_edge(2, 3)  # first operand: sin(x_0)
    dag.add_edge(1, 3)  # second operand: x_1
    return dag


def _make_const_sub_dag() -> LabeledDAG:
    """Build DAG for x_0 - 3.14 (SUB with a CONST).

    Structure:
        node 0: VAR(x_0)
        node 1: CONST(3.14)
        node 2: SUB  ordered_inputs [0, 1]

    Note: CONST has a creation edge from node 0 for reachability.
    """
    dag = LabeledDAG(4)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.CONST, const_value=3.14)  # 1
    dag.add_edge(0, 1)  # creation edge
    dag.add_node(NodeType.SUB)  # 2
    dag.add_edge(0, 2)  # first operand: x_0
    dag.add_edge(1, 2)  # second operand: 3.14
    return dag


# ======================================================================
# to_commutative: structural tests
# ======================================================================


class TestToCommutativeStructure:
    """Tests that to_commutative produces correct DAG structure."""

    def test_sub_becomes_add_neg(self) -> None:
        """SUB(x_0, x_1) -> ADD(x_0, NEG(x_1))."""
        dag = _make_sub_dag()
        comm = to_commutative(dag)

        # Original: 3 nodes (2 VAR + 1 SUB).
        # Commutative: 4 nodes (2 VAR + 1 NEG + 1 ADD).
        assert comm.node_count == 4

        # Check labels.
        assert comm.node_label(0) == NodeType.VAR
        assert comm.node_label(1) == NodeType.VAR
        assert comm.node_label(2) == NodeType.NEG
        assert comm.node_label(3) == NodeType.ADD

        # NEG has one input: x_1 (mapped node 1).
        assert comm.in_neighbors(2) == frozenset({1})

        # ADD has two inputs: x_0 (mapped node 0) and NEG (node 2).
        assert comm.in_neighbors(3) == frozenset({0, 2})

    def test_div_becomes_mul_inv(self) -> None:
        """DIV(x_0, x_1) -> MUL(x_0, INV(x_1))."""
        dag = _make_div_dag()
        comm = to_commutative(dag)

        assert comm.node_count == 4
        assert comm.node_label(2) == NodeType.INV
        assert comm.node_label(3) == NodeType.MUL

        # INV has one input: x_1.
        assert comm.in_neighbors(2) == frozenset({1})

        # MUL has two inputs: x_0 and INV.
        assert comm.in_neighbors(3) == frozenset({0, 2})

    def test_no_sub_div_unchanged(self) -> None:
        """ADD(x_0, x_1) is unchanged (no SUB/DIV to convert)."""
        dag = _make_add_dag()
        comm = to_commutative(dag)

        assert comm.node_count == 3
        assert comm.node_label(2) == NodeType.ADD
        assert comm.in_neighbors(2) == frozenset({0, 1})

    def test_pow_preserved(self) -> None:
        """POW(x_0, x_1) is copied as-is (operand order preserved)."""
        dag = _make_pow_dag()
        comm = to_commutative(dag)

        assert comm.node_count == 3
        assert comm.node_label(2) == NodeType.POW

        # Operand order must be preserved.
        inputs = comm.ordered_inputs(2)
        assert inputs == [0, 1]

    def test_nested_sub_div(self) -> None:
        """(x_0 - x_1) / x_2 -> MUL(ADD(x_0, NEG(x_1)), INV(x_2)).

        Original: 5 nodes. SUB (1 extra) + DIV (1 extra) = 7 nodes.
        """
        dag = _make_nested_sub_div_dag()
        comm = to_commutative(dag)

        # 3 VAR + 1 NEG + 1 ADD + 1 INV + 1 MUL = 7
        assert comm.node_count == 7

        # Check that no SUB or DIV nodes exist.
        for i in range(comm.node_count):
            label = comm.node_label(i)
            assert label not in (NodeType.SUB, NodeType.DIV), (
                f"Node {i} has label {label}, but SUB/DIV should be eliminated"
            )

    def test_sin_sub_dag(self) -> None:
        """sin(x_0) - x_1 -> ADD(sin(x_0), NEG(x_1))."""
        dag = _make_sin_sub_dag()
        comm = to_commutative(dag)

        # 2 VAR + 1 SIN + 1 NEG + 1 ADD = 5
        assert comm.node_count == 5

        # SIN is preserved.
        sin_nodes = [i for i in range(comm.node_count) if comm.node_label(i) == NodeType.SIN]
        assert len(sin_nodes) == 1

    def test_const_data_preserved(self) -> None:
        """CONST node data (const_value) is preserved through conversion."""
        dag = _make_const_sub_dag()
        comm = to_commutative(dag)

        # Find the CONST node in the commutative DAG.
        const_nodes = [i for i in range(comm.node_count) if comm.node_label(i) == NodeType.CONST]
        assert len(const_nodes) == 1
        assert comm.node_data(const_nodes[0]).get("const_value") == 3.14

    def test_var_data_preserved(self) -> None:
        """VAR node data (var_index) is preserved through conversion."""
        dag = _make_sub_dag()
        comm = to_commutative(dag)

        assert comm.node_data(0).get("var_index") == 0
        assert comm.node_data(1).get("var_index") == 1


# ======================================================================
# Semantic preservation: evaluation equivalence
# ======================================================================


class TestSemanticPreservation:
    """to_commutative must preserve evaluation semantics for all inputs."""

    @pytest.mark.parametrize(
        "x0, x1",
        [(3.0, 2.0), (0.0, 5.0), (-1.5, 3.7), (100.0, 0.001), (0.0, 0.0)],
    )
    def test_sub_evaluation(self, x0: float, x1: float) -> None:
        """evaluate_dag(to_commutative(SUB_dag)) == evaluate_dag(SUB_dag)."""
        dag = _make_sub_dag()
        comm = to_commutative(dag)
        inputs = {0: x0, 1: x1}
        assert evaluate_dag(dag, inputs) == evaluate_dag(comm, inputs)

    @pytest.mark.parametrize(
        "x0, x1",
        [(6.0, 3.0), (1.0, 7.0), (-2.0, 4.0), (10.0, 0.5)],
    )
    def test_div_evaluation(self, x0: float, x1: float) -> None:
        """evaluate_dag(to_commutative(DIV_dag)) == evaluate_dag(DIV_dag)."""
        dag = _make_div_dag()
        comm = to_commutative(dag)
        inputs = {0: x0, 1: x1}
        assert evaluate_dag(dag, inputs) == evaluate_dag(comm, inputs)

    def test_div_by_near_zero(self) -> None:
        """Protected division near-zero boundary: known divergence.

        _protected_div(5.0, 1e-15) returns 1.0 (full fallback),
        but MUL(5.0, _protected_inv(1e-15)) = MUL(5.0, 1.0) = 5.0.
        This is a known limitation of independent protection functions:
        DIV protects both operands jointly, while MUL+INV protect
        the divisor independently. Semantic equivalence holds for all
        inputs where |divisor| > 1e-10.
        """
        dag = _make_div_dag()
        comm = to_commutative(dag)
        inputs = {0: 5.0, 1: 1e-15}
        # DIV returns 1.0 (joint fallback), MUL*INV returns 5.0*1.0=5.0.
        assert evaluate_dag(dag, inputs) == 1.0
        assert evaluate_dag(comm, inputs) == 5.0

    @pytest.mark.parametrize(
        "x0, x1, x2",
        [(10.0, 3.0, 2.0), (-1.0, 5.0, 0.5), (0.0, 0.0, 1.0)],
    )
    def test_nested_sub_div_evaluation(self, x0: float, x1: float, x2: float) -> None:
        """evaluate_dag(to_commutative((x0-x1)/x2)) == evaluate_dag((x0-x1)/x2)."""
        dag = _make_nested_sub_div_dag()
        comm = to_commutative(dag)
        inputs = {0: x0, 1: x1, 2: x2}
        assert evaluate_dag(dag, inputs) == evaluate_dag(comm, inputs)

    @pytest.mark.parametrize(
        "x0, x1",
        [(2.0, 3.0), (0.5, -1.0)],
    )
    def test_sin_sub_evaluation(self, x0: float, x1: float) -> None:
        """sin(x_0) - x_1 is preserved."""
        dag = _make_sin_sub_dag()
        comm = to_commutative(dag)
        inputs = {0: x0, 1: x1}
        assert evaluate_dag(dag, inputs) == evaluate_dag(comm, inputs)

    def test_pow_evaluation(self) -> None:
        """POW is not decomposed; evaluation is identical."""
        dag = _make_pow_dag()
        comm = to_commutative(dag)
        inputs = {0: 2.0, 1: 3.0}
        assert evaluate_dag(dag, inputs) == evaluate_dag(comm, inputs)

    def test_const_sub_evaluation(self) -> None:
        """x_0 - 3.14 preserves evaluation with CONST."""
        dag = _make_const_sub_dag()
        comm = to_commutative(dag)
        inputs = {0: 10.0}
        assert evaluate_dag(dag, inputs) == evaluate_dag(comm, inputs)

    def test_add_unchanged_evaluation(self) -> None:
        """ADD with no SUB/DIV evaluates identically."""
        dag = _make_add_dag()
        comm = to_commutative(dag)
        inputs = {0: 1.5, 1: 2.5}
        assert evaluate_dag(dag, inputs) == evaluate_dag(comm, inputs)


# ======================================================================
# from_commutative: structural tests
# ======================================================================


class TestFromCommutativeStructure:
    """Tests that from_commutative correctly pattern-matches and collapses."""

    def test_add_neg_becomes_sub(self) -> None:
        """ADD(x_0, NEG(x_1)) -> SUB(x_0, x_1)."""
        dag = _make_sub_dag()
        comm = to_commutative(dag)
        restored = from_commutative(comm)

        assert restored.node_count == 3
        # Should have: 2 VAR + 1 SUB
        labels = [restored.node_label(i) for i in range(restored.node_count)]
        assert labels.count(NodeType.SUB) == 1
        assert labels.count(NodeType.VAR) == 2

    def test_mul_inv_becomes_div(self) -> None:
        """MUL(x_0, INV(x_1)) -> DIV(x_0, x_1)."""
        dag = _make_div_dag()
        comm = to_commutative(dag)
        restored = from_commutative(comm)

        assert restored.node_count == 3
        labels = [restored.node_label(i) for i in range(restored.node_count)]
        assert labels.count(NodeType.DIV) == 1
        assert labels.count(NodeType.VAR) == 2

    def test_nested_roundtrip_structure(self) -> None:
        """(x0 - x1) / x2 round-trips through both conversions."""
        dag = _make_nested_sub_div_dag()
        comm = to_commutative(dag)
        restored = from_commutative(comm)

        # Should have: 3 VAR + 1 SUB + 1 DIV = 5
        assert restored.node_count == 5
        labels = [restored.node_label(i) for i in range(restored.node_count)]
        assert labels.count(NodeType.SUB) == 1
        assert labels.count(NodeType.DIV) == 1

    def test_no_collapse_when_no_pattern(self) -> None:
        """ADD(x_0, x_1) with no NEG input stays as ADD."""
        dag = _make_add_dag()
        # This is already commutative, no NEG nodes.
        restored = from_commutative(dag)

        assert restored.node_count == 3
        assert restored.node_label(2) == NodeType.ADD

    def test_neg_with_multiple_consumers_not_absorbed(self) -> None:
        """NEG feeding multiple consumers should NOT be absorbed.

        Build: ADD(x_0, NEG(x_1)) + SIN(NEG(x_1)) sharing the same NEG.
        The NEG has out_degree 2, so it should not be collapsed.
        """
        dag = LabeledDAG(6)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.VAR, var_index=1)  # 1
        dag.add_node(NodeType.NEG)  # 2: NEG(x_1)
        dag.add_edge(1, 2)
        dag.add_node(NodeType.ADD)  # 3: ADD(x_0, NEG(x_1))
        dag.add_edge(0, 3)
        dag.add_edge(2, 3)
        dag.add_node(NodeType.SIN)  # 4: SIN(NEG(x_1))
        dag.add_edge(2, 4)
        dag.add_node(NodeType.ADD)  # 5: ADD(node3, node4) -- final output
        dag.add_edge(3, 5)
        dag.add_edge(4, 5)

        restored = from_commutative(dag)

        # NEG has out_degree 2, so ADD(x_0, NEG(x_1)) should NOT become SUB.
        # All 6 nodes should be preserved.
        assert restored.node_count == 6
        labels = [restored.node_label(i) for i in range(restored.node_count)]
        assert NodeType.SUB not in labels
        assert labels.count(NodeType.NEG) == 1

    def test_variadic_add_with_3_inputs_not_collapsed(self) -> None:
        """ADD with 3 inputs (one is NEG) should NOT be collapsed to SUB.

        SUB is binary; only ADD with exactly 2 inputs can be collapsed.
        """
        dag = LabeledDAG(6)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.VAR, var_index=1)  # 1
        dag.add_node(NodeType.VAR, var_index=2)  # 2
        dag.add_node(NodeType.NEG)  # 3: NEG(x_2)
        dag.add_edge(2, 3)
        dag.add_node(NodeType.ADD)  # 4: ADD(x_0, x_1, NEG(x_2))
        dag.add_edge(0, 4)
        dag.add_edge(1, 4)
        dag.add_edge(3, 4)

        restored = from_commutative(dag)

        # ADD has 3 inputs, so no collapse.
        labels = [restored.node_label(i) for i in range(restored.node_count)]
        assert NodeType.SUB not in labels
        assert labels.count(NodeType.ADD) == 1


# ======================================================================
# Round-trip: from_commutative(to_commutative(D)) preserves semantics
# ======================================================================


class TestRoundTripSemantics:
    """Round-trip conversion must preserve evaluation semantics."""

    @pytest.mark.parametrize(
        "dag_factory, inputs",
        [
            (_make_sub_dag, {0: 5.0, 1: 3.0}),
            (_make_sub_dag, {0: -1.0, 1: -1.0}),
            (_make_div_dag, {0: 10.0, 1: 4.0}),
            (_make_div_dag, {0: 0.0, 1: 1.0}),
            (_make_nested_sub_div_dag, {0: 7.0, 1: 2.0, 2: 5.0}),
            (_make_sin_sub_dag, {0: 1.57, 1: 0.5}),
            (_make_pow_dag, {0: 2.0, 1: 3.0}),
            (_make_add_dag, {0: 1.0, 1: 2.0}),
            (_make_const_sub_dag, {0: 10.0}),
        ],
    )
    def test_round_trip_evaluation(self, dag_factory: object, inputs: dict[int, float]) -> None:
        """evaluate_dag(from_commutative(to_commutative(D))) == evaluate_dag(D)."""
        dag = dag_factory()  # type: ignore[operator]
        comm = to_commutative(dag)
        restored = from_commutative(comm)
        assert evaluate_dag(dag, inputs) == evaluate_dag(restored, inputs)


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    """Edge cases and corner cases for commutative conversion."""

    def test_single_var_dag(self) -> None:
        """DAG with only one VAR node (no ops) -- trivial case.

        Note: evaluate_dag requires an output node that is non-VAR, so we
        cannot call evaluate_dag here. We just check structural conversion.
        """
        dag = LabeledDAG(2)
        dag.add_node(NodeType.VAR, var_index=0)

        comm = to_commutative(dag)
        assert comm.node_count == 1
        assert comm.node_label(0) == NodeType.VAR

    def test_var_only_from_commutative(self) -> None:
        """from_commutative on a VAR-only DAG should be a no-op copy."""
        dag = LabeledDAG(2)
        dag.add_node(NodeType.VAR, var_index=0)

        restored = from_commutative(dag)
        assert restored.node_count == 1
        assert restored.node_label(0) == NodeType.VAR

    def test_multiple_sub_nodes(self) -> None:
        """Two independent SUB nodes: (x0 - x1) + (x2 - x3).

        Both SUB nodes should be decomposed independently.
        """
        dag = LabeledDAG(8)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.VAR, var_index=1)  # 1
        dag.add_node(NodeType.VAR, var_index=2)  # 2
        dag.add_node(NodeType.VAR, var_index=3)  # 3
        dag.add_node(NodeType.SUB)  # 4: x_0 - x_1
        dag.add_edge(0, 4)
        dag.add_edge(1, 4)
        dag.add_node(NodeType.SUB)  # 5: x_2 - x_3
        dag.add_edge(2, 5)
        dag.add_edge(3, 5)
        dag.add_node(NodeType.ADD)  # 6: (x0-x1) + (x2-x3)
        dag.add_edge(4, 6)
        dag.add_edge(5, 6)

        comm = to_commutative(dag)

        # 4 VAR + 2 NEG + 2 ADD + 1 ADD(top) = 9
        # Wait: the original ADD remains, and 2 SUB each become NEG+ADD = 4 extra nodes.
        # Original: 7 nodes. Extra: 2 (one per SUB). Total: 9.
        assert comm.node_count == 9

        # No SUB nodes should remain.
        for i in range(comm.node_count):
            assert comm.node_label(i) != NodeType.SUB

        # Evaluate for correctness.
        inputs = {0: 10.0, 1: 3.0, 2: 7.0, 3: 2.0}
        assert evaluate_dag(dag, inputs) == evaluate_dag(comm, inputs)

    def test_double_neg_not_collapsed(self) -> None:
        """ADD(NEG(x_0), NEG(x_1)): both inputs are NEG.

        With exactly 2 inputs both being NEG, _try_collapse sees 2 candidates
        and does NOT collapse (ambiguous). This is correct behavior.
        """
        dag = LabeledDAG(6)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.VAR, var_index=1)  # 1
        dag.add_node(NodeType.NEG)  # 2: NEG(x_0)
        dag.add_edge(0, 2)
        dag.add_node(NodeType.NEG)  # 3: NEG(x_1)
        dag.add_edge(1, 3)
        dag.add_node(NodeType.ADD)  # 4: ADD(NEG(x_0), NEG(x_1))
        dag.add_edge(2, 4)
        dag.add_edge(3, 4)

        restored = from_commutative(dag)

        # Both NEGs are candidates. Since there are 2, we don't collapse.
        labels = [restored.node_label(i) for i in range(restored.node_count)]
        assert NodeType.SUB not in labels
        assert labels.count(NodeType.NEG) == 2
        assert labels.count(NodeType.ADD) == 1

    def test_to_commutative_invalid_sub_arity(self) -> None:
        """SUB node with wrong number of inputs should raise ValueError."""
        dag = LabeledDAG(4)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.SUB)  # 1: only one input
        dag.add_edge(0, 1)

        with pytest.raises(ValueError, match="SUB node.*1 inputs.*expected 2"):
            to_commutative(dag)

    def test_to_commutative_invalid_div_arity(self) -> None:
        """DIV node with wrong number of inputs should raise ValueError."""
        dag = LabeledDAG(4)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.DIV)  # 1: only one input
        dag.add_edge(0, 1)

        with pytest.raises(ValueError, match="DIV node.*1 inputs.*expected 2"):
            to_commutative(dag)


# ======================================================================
# Isomorphism between original and round-tripped DAGs
# ======================================================================


class TestRoundTripIsomorphism:
    """The round-trip from_commutative(to_commutative(D)) should produce
    a DAG isomorphic to the original D (same labels, same edges, same
    operand order for binary ops).
    """

    def test_sub_roundtrip_isomorphic(self) -> None:
        dag = _make_sub_dag()
        restored = from_commutative(to_commutative(dag))
        assert dag.is_isomorphic(restored)

    def test_div_roundtrip_isomorphic(self) -> None:
        dag = _make_div_dag()
        restored = from_commutative(to_commutative(dag))
        assert dag.is_isomorphic(restored)

    def test_nested_roundtrip_isomorphic(self) -> None:
        dag = _make_nested_sub_div_dag()
        restored = from_commutative(to_commutative(dag))
        assert dag.is_isomorphic(restored)

    def test_sin_sub_roundtrip_isomorphic(self) -> None:
        dag = _make_sin_sub_dag()
        restored = from_commutative(to_commutative(dag))
        assert dag.is_isomorphic(restored)

    def test_pow_roundtrip_isomorphic(self) -> None:
        dag = _make_pow_dag()
        restored = from_commutative(to_commutative(dag))
        assert dag.is_isomorphic(restored)

    def test_add_roundtrip_isomorphic(self) -> None:
        dag = _make_add_dag()
        restored = from_commutative(to_commutative(dag))
        assert dag.is_isomorphic(restored)

    def test_const_sub_roundtrip_isomorphic(self) -> None:
        dag = _make_const_sub_dag()
        restored = from_commutative(to_commutative(dag))
        assert dag.is_isomorphic(restored)
