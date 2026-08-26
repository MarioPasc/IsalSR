"""Unit tests for the Bingo AGraph <-> LabeledDAG adapter.

Rewritten tests (test_subtraction_operand_order, test_subtraction_reversed,
test_division) assert the T16 commutative encoding: SUB -> ADD+NEG and
DIV -> MUL+INV.  All other tests are unchanged.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

bingo = pytest.importorskip("bingo")

from bingo.symbolic_regression.agraph.agraph import AGraph  # noqa: E402

from experiments.models.bingo.adapter import (  # noqa: E402
    agraph_to_labeled_dag,
    labeled_dag_to_agraph,
)
from isalsr.core.canonical import pruned_canonical_string  # noqa: E402
from isalsr.core.dag_evaluator import evaluate_dag  # noqa: E402
from isalsr.core.node_types import NodeType  # noqa: E402


def _make_agraph(cmd: list[list[int]]) -> AGraph:
    """Create an AGraph from a command array list.

    Args:
        cmd: List of [op_code, param1, param2] rows.

    Returns:
        Initialised AGraph ready for utilisation.
    """
    ag = AGraph(use_simplification=False)
    ag._command_array = np.array(cmd, dtype=int)
    ag._notify_modification()
    return ag


class TestAGraphToLabeledDAG:
    def test_addition(self) -> None:
        """x_0 + x_1 maps to two VAR nodes and one ADD node."""
        ag = _make_agraph([[0, 0, 0], [0, 1, 0], [2, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_count == 3
        assert dag.edge_count == 2
        assert dag.node_label(0) == NodeType.VAR
        assert dag.node_label(1) == NodeType.VAR
        assert dag.node_label(2) == NodeType.ADD

    def test_sin(self) -> None:
        """sin(x_0) produces a SIN node."""
        ag = _make_agraph([[0, 0, 0], [6, 0, 0]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_count == 2
        assert dag.node_label(1) == NodeType.SIN

    def test_subtraction_operand_order(self) -> None:
        """x_0 - x_1 decomposes to ADD(x_0, NEG(x_1)); operand order preserved.

        T16 rewrite: SUB is no longer emitted.  The resulting DAG has four nodes
        [VAR, VAR, NEG, ADD]; ordered_inputs on the ADD node returns [x_0, NEG].
        """
        ag = _make_agraph([[0, 0, 0], [0, 1, 0], [3, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        # Four nodes: VAR(0), VAR(1), NEG(2), ADD(3)
        assert dag.node_count == 4
        assert dag.node_label(2) == NodeType.NEG
        assert dag.node_label(3) == NodeType.ADD
        # ADD's ordered inputs: x_0 first (first add_edge), NEG second
        inputs = dag.ordered_inputs(3)
        assert inputs == [0, 2]
        # No forbidden SUB label anywhere
        labels = {dag.node_label(i) for i in range(dag.node_count)}
        assert NodeType.SUB not in labels

    def test_subtraction_reversed(self) -> None:
        """x_1 - x_0 decomposes to ADD(x_1, NEG(x_0)); NEG wraps x_0.

        T16 rewrite: SUB(param1=x_1, param2=x_0) -> ADD(x_1, NEG(x_0)).
        ordered_inputs on the ADD node must be [x_1_node, NEG_node] = [1, 2].
        """
        ag = _make_agraph([[0, 0, 0], [0, 1, 0], [3, 1, 0]])
        dag = agraph_to_labeled_dag(ag)
        # NEG wraps x_0 (node 0); ADD node is at index 3
        assert dag.node_label(2) == NodeType.NEG
        assert dag.node_label(3) == NodeType.ADD
        inputs = dag.ordered_inputs(3)
        # x_1 (node 1) is the first operand; NEG(x_0) (node 2) is second
        assert inputs == [1, 2]
        labels = {dag.node_label(i) for i in range(dag.node_count)}
        assert NodeType.SUB not in labels

    def test_different_subtraction_canonical_strings(self) -> None:
        """x-y and y-x must have different canonical strings."""
        ag1 = _make_agraph([[0, 0, 0], [0, 1, 0], [3, 0, 1]])
        ag2 = _make_agraph([[0, 0, 0], [0, 1, 0], [3, 1, 0]])
        cs1 = pruned_canonical_string(agraph_to_labeled_dag(ag1))
        cs2 = pruned_canonical_string(agraph_to_labeled_dag(ag2))
        assert cs1 != cs2

    def test_division(self) -> None:
        """x_0 / x_1 decomposes to MUL(x_0, INV(x_1)).

        T16 rewrite: DIV is no longer emitted.  The resulting DAG has four nodes
        [VAR, VAR, INV, MUL]; ordered_inputs on the MUL node returns [x_0, INV].
        """
        ag = _make_agraph([[0, 0, 0], [0, 1, 0], [5, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_count == 4
        assert dag.node_label(2) == NodeType.INV
        assert dag.node_label(3) == NodeType.MUL
        inputs = dag.ordered_inputs(3)
        assert inputs == [0, 2]
        labels = {dag.node_label(i) for i in range(dag.node_count)}
        assert NodeType.DIV not in labels

    def test_const_node(self) -> None:
        """c + x_0 produces a CONST node with a creation edge."""
        ag = _make_agraph([[0, 0, 0], [1, 0, 0], [2, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_count == 3
        assert dag.node_label(1) == NodeType.CONST
        assert dag.in_degree(1) >= 1  # CONST has creation edge

    def test_unused_rows_filtered(self) -> None:
        """Only utilized rows should be converted."""
        # Row 0: x_0, Row 1: x_1 (unused), Row 2: x_0*x_0 (unused),
        # Row 3: x_0+x_0 (unused), Row 4: sin(x_0) (output)
        ag = _make_agraph(
            [
                [0, 0, 0],
                [0, 1, 0],
                [4, 0, 1],
                [2, 0, 2],
                [6, 0, 0],
            ]
        )
        ag.get_utilized_commands()
        dag = agraph_to_labeled_dag(ag)
        cs = pruned_canonical_string(dag)
        assert cs == "Vs"  # sin(x_0)

    def test_variable_deduplication(self) -> None:
        """Multiple rows referencing same variable produce one VAR node."""
        ag = _make_agraph([[0, 0, 0], [0, 0, 0], [2, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        # Only 1 VAR node (both rows reference x_0)
        var_count = sum(1 for i in range(dag.node_count) if dag.node_label(i) == NodeType.VAR)
        assert var_count == 1

    def test_canonical_isomorphism(self) -> None:
        """Isomorphic AGraphs produce the same canonical string."""
        # x_0 + x_1
        ag1 = _make_agraph([[0, 0, 0], [0, 1, 0], [2, 0, 1]])
        # x_1 + x_0 (addition is commutative)
        ag2 = _make_agraph([[0, 1, 0], [0, 0, 0], [2, 0, 1]])
        cs1 = pruned_canonical_string(agraph_to_labeled_dag(ag1))
        cs2 = pruned_canonical_string(agraph_to_labeled_dag(ag2))
        assert cs1 == cs2

    def test_exp(self) -> None:
        """exp(x_0) produces an EXP node."""
        ag = _make_agraph([[0, 0, 0], [8, 0, 0]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_label(1) == NodeType.EXP

    def test_log(self) -> None:
        """log(x_0) produces a LOG node."""
        ag = _make_agraph([[0, 0, 0], [9, 0, 0]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_label(1) == NodeType.LOG

    def test_cos(self) -> None:
        """cos(x_0) produces a COS node."""
        ag = _make_agraph([[0, 0, 0], [7, 0, 0]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_label(1) == NodeType.COS

    def test_pow_unchanged(self) -> None:
        """POW is not decomposed; it remains a single POW node."""
        # x_0 ** x_1
        ag = _make_agraph([[0, 0, 0], [0, 1, 0], [10, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        assert dag.node_label(2) == NodeType.POW
        labels = {dag.node_label(i) for i in range(dag.node_count)}
        assert NodeType.NEG not in labels
        assert NodeType.ADD not in labels

    @pytest.mark.parametrize(
        "cmd,x0,x1,expected",
        [
            ([[0, 0, 0], [0, 1, 0], [3, 0, 1]], 5.0, 3.0, 2.0),  # x0 - x1
            ([[0, 0, 0], [0, 1, 0], [3, 1, 0]], 5.0, 3.0, -2.0),  # x1 - x0
            ([[0, 0, 0], [0, 1, 0], [5, 0, 1]], 6.0, 3.0, 2.0),  # x0 / x1
            ([[0, 0, 0], [0, 1, 0], [5, 1, 0]], 6.0, 3.0, 0.5),  # x1 / x0
        ],
    )
    def test_sub_div_operand_order_numerical(
        self, cmd: list[list[int]], x0: float, x1: float, expected: float
    ) -> None:
        """Decomposed SUB/DIV evaluates to the correct numeric value."""
        ag = _make_agraph(cmd)
        dag = agraph_to_labeled_dag(ag)
        result = evaluate_dag(dag, {0: x0, 1: x1})
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_sub_self_reference_evaluates_zero(self) -> None:
        """x - x decomposes correctly and evaluates to 0.

        Pre-T16 the undecomposed SUB had only one in-edge (duplicate rejected),
        so it evaluated as x, not 0. Decomposition fixes this.
        """
        ag = _make_agraph([[0, 0, 0], [3, 0, 0]])
        dag = agraph_to_labeled_dag(ag)
        # ADD(x, NEG(x)): two distinct nodes, two edges
        labels = [dag.node_label(i) for i in range(dag.node_count)]
        assert NodeType.NEG in labels
        assert NodeType.ADD in labels
        result = evaluate_dag(dag, {0: 7.0})
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_div_self_reference_evaluates_one(self) -> None:
        """x / x decomposes correctly and evaluates to 1.0."""
        ag = _make_agraph([[0, 0, 0], [5, 0, 0]])
        dag = agraph_to_labeled_dag(ag)
        labels = [dag.node_label(i) for i in range(dag.node_count)]
        assert NodeType.INV in labels
        assert NodeType.MUL in labels
        result = evaluate_dag(dag, {0: 7.0})
        np.testing.assert_allclose(result, 1.0, rtol=1e-12)


class TestLabeledDAGToAGraph:
    def test_roundtrip_addition(self) -> None:
        """AGraph → LabeledDAG → AGraph roundtrip for addition."""
        ag = _make_agraph([[0, 0, 0], [0, 1, 0], [2, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        ag2 = labeled_dag_to_agraph(dag)
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y1 = ag.evaluate_equation_at(x)
        y2 = ag2.evaluate_equation_at(x)
        np.testing.assert_allclose(y1, y2)

    def test_roundtrip_sin(self) -> None:
        """sin(x_0) roundtrip preserves evaluation."""
        ag = _make_agraph([[0, 0, 0], [6, 0, 0]])
        dag = agraph_to_labeled_dag(ag)
        ag2 = labeled_dag_to_agraph(dag)
        x = np.array([[0.5], [1.0]])
        y1 = ag.evaluate_equation_at(x)
        y2 = ag2.evaluate_equation_at(x)
        np.testing.assert_allclose(y1, y2, rtol=1e-10)

    def test_roundtrip_subtraction(self) -> None:
        """x_0 - x_1: AGraph -> DAG (decomposed) -> AGraph recovers SUB."""
        ag = _make_agraph([[0, 0, 0], [0, 1, 0], [3, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        ag2 = labeled_dag_to_agraph(dag)
        x = np.array([[5.0, 3.0], [10.0, 4.0]])
        y1 = ag.evaluate_equation_at(x)
        y2 = ag2.evaluate_equation_at(x)
        np.testing.assert_allclose(y1, y2, rtol=1e-12)

    def test_roundtrip_division(self) -> None:
        """x_0 / x_1: AGraph -> DAG (decomposed) -> AGraph recovers DIV."""
        ag = _make_agraph([[0, 0, 0], [0, 1, 0], [5, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        ag2 = labeled_dag_to_agraph(dag)
        x = np.array([[6.0, 3.0], [10.0, 2.0]])
        y1 = ag.evaluate_equation_at(x)
        y2 = ag2.evaluate_equation_at(x)
        np.testing.assert_allclose(y1, y2, rtol=1e-12)

    def test_roundtrip_reversed_subtraction(self) -> None:
        """x_1 - x_0 roundtrip recovers the correct command array row."""
        ag = _make_agraph([[0, 0, 0], [0, 1, 0], [3, 1, 0]])
        dag = agraph_to_labeled_dag(ag)
        ag2 = labeled_dag_to_agraph(dag)
        x = np.array([[5.0, 3.0]])
        y1 = ag.evaluate_equation_at(x)
        y2 = ag2.evaluate_equation_at(x)
        np.testing.assert_allclose(y1, y2, rtol=1e-12)
