"""Unit tests for the UDFS CompGraph <-> LabeledDAG adapter.

Rewritten tests (test_sub_l, test_sub_r_reverses_children, test_div_l,
test_div_r_reverses_children) assert the T16 commutative encoding:
sub_l/sub_r -> ADD+NEG and div_l/div_r -> MUL+INV.  All other tests
are unchanged.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Add vendored DAG_search to path
_vendor_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "experiments",
    "models",
    "udfs",
    "vendor",
)
if _vendor_dir not in sys.path:
    sys.path.insert(0, _vendor_dir)

# Skip if torch not available
torch = pytest.importorskip("torch")

from DAG_search.comp_graph import CompGraph  # noqa: E402

from experiments.models.udfs.adapter import (  # noqa: E402
    compgraph_to_labeled_dag,
    labeled_dag_to_compgraph,
)
from isalsr.core.canonical import pruned_canonical_string  # noqa: E402
from isalsr.core.dag_evaluator import evaluate_dag  # noqa: E402
from isalsr.core.node_types import NodeType  # noqa: E402


def _make_compgraph(
    m: int, n: int, k: int, node_dict: dict[int, tuple[tuple[int, ...], str]]
) -> CompGraph:
    """Create a CompGraph from dimension and node_dict parameters.

    Args:
        m: Number of input variables.
        n: Number of outputs.
        k: Number of constants.
        node_dict: Mapping of node_id -> (children, op_str).

    Returns:
        Constructed CompGraph.
    """
    return CompGraph(m, n, k, node_dict=node_dict)


class TestCompGraphToLabeledDAG:
    def test_addition(self) -> None:
        """x_0 + x_1 maps to two VAR nodes and one ADD node."""
        cg = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "+"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_count == 3  # 2 VAR + 1 ADD
        assert dag.edge_count == 2
        assert dag.node_label(0) == NodeType.VAR
        assert dag.node_label(1) == NodeType.VAR
        assert dag.node_label(2) == NodeType.ADD

    def test_sin(self) -> None:
        """sin(x_0) produces a SIN node."""
        cg = _make_compgraph(
            1,
            1,
            0,
            {
                0: ((), "="),
                1: ((0,), "sin"),
                2: ((1,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_count == 2
        assert dag.node_label(1) == NodeType.SIN

    def test_sub_l(self) -> None:
        """sub_l(x_0, x_1) = x_0 - x_1 decomposes to ADD(x_0, NEG(x_1)).

        T16 rewrite: SUB is no longer emitted.  The resulting DAG has four
        nodes [VAR, VAR, NEG, ADD].  ordered_inputs on ADD returns [x_0, NEG].
        Numerically evaluates to x_0 - x_1.
        """
        cg = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "sub_l"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_count == 4
        assert dag.node_label(2) == NodeType.NEG
        assert dag.node_label(3) == NodeType.ADD
        # ADD's first operand is x_0 (node 0); second is NEG (node 2)
        inputs = dag.ordered_inputs(3)
        assert inputs == [0, 2]
        labels = {dag.node_label(i) for i in range(dag.node_count)}
        assert NodeType.SUB not in labels
        # Numerically: 6 - 3 = 3
        result = evaluate_dag(dag, {0: 6.0, 1: 3.0})
        np.testing.assert_allclose(result, 3.0, rtol=1e-12)

    def test_sub_r_reverses_children(self) -> None:
        """sub_r(x_0, x_1) = x_1 - x_0 decomposes to ADD(x_1, NEG(x_0)).

        T16 rewrite: REVERSED_OPS swaps children before decomposition, so
        NEG wraps x_0 (node 0) and ADD has ordered_inputs [x_1, NEG] = [1, 2].
        Numerically evaluates to x_1 - x_0.
        """
        cg = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "sub_r"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_count == 4
        assert dag.node_label(2) == NodeType.NEG
        assert dag.node_label(3) == NodeType.ADD
        inputs = dag.ordered_inputs(3)
        # x_1 (node 1) is minuend; NEG(x_0) (node 2) is second operand
        assert inputs == [1, 2]
        labels = {dag.node_label(i) for i in range(dag.node_count)}
        assert NodeType.SUB not in labels
        # Numerically: 3 - 6 = -3
        result = evaluate_dag(dag, {0: 6.0, 1: 3.0})
        np.testing.assert_allclose(result, -3.0, rtol=1e-12)

    def test_div_l(self) -> None:
        """div_l(x_0, x_1) = x_0 / x_1 decomposes to MUL(x_0, INV(x_1)).

        T16 rewrite: DIV is no longer emitted.  The resulting DAG has four
        nodes [VAR, VAR, INV, MUL].  ordered_inputs on MUL returns [x_0, INV].
        Numerically evaluates to x_0 / x_1.
        """
        cg = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "div_l"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_count == 4
        assert dag.node_label(2) == NodeType.INV
        assert dag.node_label(3) == NodeType.MUL
        inputs = dag.ordered_inputs(3)
        assert inputs == [0, 2]
        labels = {dag.node_label(i) for i in range(dag.node_count)}
        assert NodeType.DIV not in labels
        # Numerically: 6 / 3 = 2
        result = evaluate_dag(dag, {0: 6.0, 1: 3.0})
        np.testing.assert_allclose(result, 2.0, rtol=1e-12)

    def test_div_r_reverses_children(self) -> None:
        """div_r(x_0, x_1) = x_1 / x_0 decomposes to MUL(x_1, INV(x_0)).

        T16 rewrite: REVERSED_OPS swaps children before decomposition, so
        INV wraps x_0 (node 0) and MUL has ordered_inputs [x_1, INV] = [1, 2].
        Numerically evaluates to x_1 / x_0.
        """
        cg = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "div_r"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_count == 4
        assert dag.node_label(2) == NodeType.INV
        assert dag.node_label(3) == NodeType.MUL
        inputs = dag.ordered_inputs(3)
        # x_1 (node 1) is numerator; INV(x_0) (node 2) is second operand
        assert inputs == [1, 2]
        labels = {dag.node_label(i) for i in range(dag.node_count)}
        assert NodeType.DIV not in labels
        # Numerically: 3 / 6 = 0.5
        result = evaluate_dag(dag, {0: 6.0, 1: 3.0})
        np.testing.assert_allclose(result, 0.5, rtol=1e-12)

    def test_identity_collapse(self) -> None:
        """Identity nodes ('=') should be collapsed."""
        cg = _make_compgraph(
            1,
            1,
            0,
            {
                0: ((), "="),
                1: ((0,), "sin"),
                2: ((1,), "="),  # output identity
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        # Only 2 nodes: VAR and SIN (identity collapsed)
        assert dag.node_count == 2

    def test_const_node(self) -> None:
        """x_0 + c_0 produces a CONST node with a creation edge."""
        cg = _make_compgraph(
            1,
            1,
            1,
            {
                0: ((), "="),  # x_0
                1: ((), "="),  # c_0
                2: ((0, 1), "+"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_count == 3
        assert dag.node_label(1) == NodeType.CONST
        # CONST should have creation edge from node 0 (x_0)
        assert dag.in_degree(1) >= 1

    def test_neg(self) -> None:
        """UDFS native neg(x_0) maps to NodeType.NEG unchanged."""
        cg = _make_compgraph(
            1,
            1,
            0,
            {
                0: ((), "="),
                1: ((0,), "neg"),
                2: ((1,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_label(1) == NodeType.NEG

    def test_inv(self) -> None:
        """UDFS native inv(x_0) maps to NodeType.INV unchanged."""
        cg = _make_compgraph(
            1,
            1,
            0,
            {
                0: ((), "="),
                1: ((0,), "inv"),
                2: ((1,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_label(1) == NodeType.INV

    def test_canonical_string_consistency(self) -> None:
        """Two isomorphic CompGraphs should produce the same canonical string."""
        # x_0 + x_1 (order 1)
        cg1 = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "+"),
                3: ((2,), "="),
            },
        )
        # x_0 + x_1 (same graph, just different build)
        cg2 = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((1, 0), "+"),
                3: ((2,), "="),
            },
        )
        dag1 = compgraph_to_labeled_dag(cg1)
        dag2 = compgraph_to_labeled_dag(cg2)
        cs1 = pruned_canonical_string(dag1)
        cs2 = pruned_canonical_string(dag2)
        # x_0+x_1 and x_1+x_0 should be isomorphic for commutative ADD
        assert cs1 == cs2

    @pytest.mark.parametrize(
        "op,x0,x1,expected",
        [
            ("sub_l", 6.0, 3.0, 3.0),  # x0 - x1
            ("sub_r", 6.0, 3.0, -3.0),  # x1 - x0
            ("div_l", 6.0, 3.0, 2.0),  # x0 / x1
            ("div_r", 6.0, 3.0, 0.5),  # x1 / x0
        ],
    )
    def test_all_four_orientations_numerical(
        self, op: str, x0: float, x1: float, expected: float
    ) -> None:
        """All four sub/div orientations evaluate to the correct numeric value."""
        cg = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), op),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        result = evaluate_dag(dag, {0: x0, 1: x1})
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_sub_l_self_reference_evaluates_zero(self) -> None:
        """sub_l(x_0, x_0) = x_0 - x_0 decomposes correctly and yields 0."""
        cg = _make_compgraph(
            1,
            1,
            0,
            {
                0: ((), "="),
                1: ((0, 0), "sub_l"),
                2: ((1,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        labels = {dag.node_label(i) for i in range(dag.node_count)}
        assert NodeType.NEG in labels
        assert NodeType.ADD in labels
        result = evaluate_dag(dag, {0: 7.0})
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_div_l_self_reference_evaluates_one(self) -> None:
        """div_l(x_0, x_0) = x_0 / x_0 decomposes correctly and yields 1."""
        cg = _make_compgraph(
            1,
            1,
            0,
            {
                0: ((), "="),
                1: ((0, 0), "div_l"),
                2: ((1,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        labels = {dag.node_label(i) for i in range(dag.node_count)}
        assert NodeType.INV in labels
        assert NodeType.MUL in labels
        result = evaluate_dag(dag, {0: 7.0})
        np.testing.assert_allclose(result, 1.0, rtol=1e-12)

    def test_udfs_native_neg_passthrough_decompose_false(self) -> None:
        """UDFS host neg remains NodeType.NEG even with decompose=False."""
        cg = _make_compgraph(
            1,
            1,
            0,
            {
                0: ((), "="),
                1: ((0,), "neg"),
                2: ((1,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg, decompose=False)
        assert dag.node_label(1) == NodeType.NEG

    def test_udfs_native_inv_passthrough_decompose_false(self) -> None:
        """UDFS host inv remains NodeType.INV even with decompose=False."""
        cg = _make_compgraph(
            1,
            1,
            0,
            {
                0: ((), "="),
                1: ((0,), "inv"),
                2: ((1,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg, decompose=False)
        assert dag.node_label(1) == NodeType.INV


class TestLabeledDAGToCompGraph:
    def test_roundtrip_simple(self) -> None:
        """CompGraph -> LabeledDAG -> CompGraph roundtrip for addition."""
        cg = _make_compgraph(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "+"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        cg2 = labeled_dag_to_compgraph(dag)
        assert cg2.inp_dim == 2
        assert cg2.outp_dim == 1
        assert cg2.n_consts == 0
        assert cg2.n_nodes() >= 3  # 2 inputs + 1 op + 1 output

    def test_roundtrip_with_const(self) -> None:
        """Round-trip with a constant node preserves n_consts."""
        cg = _make_compgraph(
            1,
            1,
            1,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "+"),
                3: ((2,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        cg2 = labeled_dag_to_compgraph(dag)
        assert cg2.n_consts == 1
