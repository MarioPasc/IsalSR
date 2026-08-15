"""Unit tests for the commutative encoding at the adapter boundary (T16).

Covers ten acceptance criteria from the T16 implementation brief:

AC-2  No SUB/DIV labels survive into decomposed LabeledDAGs.
AC-6  Operand order is numerically correct for every host and orientation.
AC-3  Self-reference: x-x -> 0, x/x -> 1 (was broken before T16).
AC-4  POW is untouched by decomposition.
AC-5  Nested/mixed expressions evaluate correctly.
AC-7  decompose=False reproduces the pre-T16 label set exactly.
AC-8  share_unary=True collapses duplicate Neg/Inv; =False keeps them separate.
AC-9  undecompose round-trips the non-shared case; raises on the shared case.
AC-1  Capacity: many Sub/Div nodes near Bingo's production stack size cause
      no RuntimeError.
AC-10 UDFS host neg/inv pass through as NodeType.NEG/INV unchanged regardless
      of the decompose flag.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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

bingo = pytest.importorskip("bingo")
torch = pytest.importorskip("torch")

from bingo.symbolic_regression.agraph.agraph import AGraph  # noqa: E402
from DAG_search.comp_graph import CompGraph  # noqa: E402

from experiments.models.bingo.adapter import agraph_to_labeled_dag  # noqa: E402
from experiments.models.commutative_encoding import (  # noqa: E402
    FORBIDDEN_ADAPTER_LABELS,
    SHARE_DECOMPOSED_UNARY,
    contains_decomposed_unary,
    emit_binary,
    extra_node_budget,
    new_unary_cache,
    undecompose,
)
from experiments.models.udfs.adapter import compgraph_to_labeled_dag  # noqa: E402
from isalsr.core.dag_evaluator import evaluate_dag  # noqa: E402
from isalsr.core.labeled_dag import LabeledDAG  # noqa: E402
from isalsr.core.node_types import NodeType  # noqa: E402

# ======================================================================
# Helpers
# ======================================================================


def _bingo(cmd: list[list[int]]) -> AGraph:
    """Build a Bingo AGraph from a command array.

    Args:
        cmd: List of [op_code, param1, param2] rows.

    Returns:
        Initialised AGraph.
    """
    ag = AGraph(use_simplification=False)
    ag._command_array = np.array(cmd, dtype=int)
    ag._notify_modification()
    return ag


def _udfs(
    m: int,
    n: int,
    k: int,
    node_dict: dict[int, tuple[tuple[int, ...], str]],
) -> CompGraph:
    """Build a UDFS CompGraph.

    Args:
        m: Input dimension.
        n: Output dimension.
        k: Number of constants.
        node_dict: Mapping of node_id -> (children, op_str).

    Returns:
        Constructed CompGraph.
    """
    return CompGraph(m, n, k, node_dict=node_dict)


def _all_labels(dag: LabeledDAG) -> set[NodeType]:
    """Return the set of all node labels in a DAG.

    Args:
        dag: The LabeledDAG to inspect.

    Returns:
        Set of distinct NodeType values present.
    """
    return {dag.node_label(i) for i in range(dag.node_count)}


# ======================================================================
# AC-2: No forbidden labels in decomposed DAGs
# ======================================================================


class TestNoForbiddenLabels:
    """AC-2: FORBIDDEN_ADAPTER_LABELS are absent from every decomposed DAG."""

    def test_forbidden_set_contains_sub_div(self) -> None:
        """FORBIDDEN_ADAPTER_LABELS covers exactly SUB and DIV."""
        assert NodeType.SUB in FORBIDDEN_ADAPTER_LABELS
        assert NodeType.DIV in FORBIDDEN_ADAPTER_LABELS
        # POW must NOT be forbidden – it has no commutative decomposition
        assert NodeType.POW not in FORBIDDEN_ADAPTER_LABELS

    @pytest.mark.parametrize(
        "cmd",
        [
            [[0, 0, 0], [0, 1, 0], [3, 0, 1]],  # SUB x0-x1
            [[0, 0, 0], [0, 1, 0], [3, 1, 0]],  # SUB x1-x0
            [[0, 0, 0], [0, 1, 0], [5, 0, 1]],  # DIV x0/x1
            [[0, 0, 0], [0, 1, 0], [5, 1, 0]],  # DIV x1/x0
            [[0, 0, 0], [3, 0, 0]],  # SUB x0-x0
            [[0, 0, 0], [5, 0, 0]],  # DIV x0/x0
            [[0, 0, 0], [0, 1, 0], [2, 0, 1], [3, 0, 2]],  # nested SUB
            [[0, 0, 0], [0, 1, 0], [2, 0, 1], [5, 0, 2]],  # nested DIV
        ],
    )
    def test_bingo_battery_no_forbidden_labels(self, cmd: list[list[int]]) -> None:
        """No decomposed Bingo DAG carries a forbidden label."""
        dag = agraph_to_labeled_dag(_bingo(cmd))
        present = _all_labels(dag)
        assert present.isdisjoint(FORBIDDEN_ADAPTER_LABELS), (
            f"Forbidden labels {present & FORBIDDEN_ADAPTER_LABELS} found in {cmd}"
        )

    @pytest.mark.parametrize(
        "op",
        ["sub_l", "sub_r", "div_l", "div_r"],
    )
    def test_udfs_battery_no_forbidden_labels(self, op: str) -> None:
        """No decomposed UDFS DAG carries a forbidden label."""
        cg = _udfs(
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
        present = _all_labels(dag)
        assert present.isdisjoint(FORBIDDEN_ADAPTER_LABELS), (
            f"Forbidden labels {present & FORBIDDEN_ADAPTER_LABELS} found for op={op}"
        )


# ======================================================================
# AC-6: Operand order is numerically correct
# ======================================================================


class TestOperandOrderBingo:
    """AC-6: Bingo SUB and DIV with both parameter orderings evaluate correctly."""

    @pytest.mark.parametrize(
        "cmd,x0,x1,expected",
        [
            ([[0, 0, 0], [0, 1, 0], [3, 0, 1]], 5.0, 3.0, 2.0),  # x0 - x1
            ([[0, 0, 0], [0, 1, 0], [3, 1, 0]], 5.0, 3.0, -2.0),  # x1 - x0
            ([[0, 0, 0], [0, 1, 0], [5, 0, 1]], 6.0, 3.0, 2.0),  # x0 / x1
            ([[0, 0, 0], [0, 1, 0], [5, 1, 0]], 6.0, 3.0, 0.5),  # x1 / x0
        ],
        ids=["sub_fwd", "sub_rev", "div_fwd", "div_rev"],
    )
    def test_numerical(self, cmd: list[list[int]], x0: float, x1: float, expected: float) -> None:
        """Decomposed DAG yields the same numeric result as the host expression."""
        dag = agraph_to_labeled_dag(_bingo(cmd))
        result = evaluate_dag(dag, {0: x0, 1: x1})
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_asymmetric_inputs_distinguish_orientations(self) -> None:
        """evaluate_dag(x0=5, x1=3) differs for x0-x1 and x1-x0."""
        dag_fwd = agraph_to_labeled_dag(_bingo([[0, 0, 0], [0, 1, 0], [3, 0, 1]]))
        dag_rev = agraph_to_labeled_dag(_bingo([[0, 0, 0], [0, 1, 0], [3, 1, 0]]))
        r_fwd = evaluate_dag(dag_fwd, {0: 5.0, 1: 3.0})
        r_rev = evaluate_dag(dag_rev, {0: 5.0, 1: 3.0})
        assert r_fwd != r_rev, "Forward and reversed SUB must produce different values"


class TestOperandOrderUDFS:
    """AC-6: All four UDFS orientations (sub_l, sub_r, div_l, div_r) evaluate correctly."""

    @pytest.mark.parametrize(
        "op,x0,x1,expected",
        [
            ("sub_l", 6.0, 3.0, 3.0),  # x0 - x1
            ("sub_r", 6.0, 3.0, -3.0),  # x1 - x0
            ("div_l", 6.0, 3.0, 2.0),  # x0 / x1
            ("div_r", 6.0, 3.0, 0.5),  # x1 / x0
        ],
        ids=["sub_l", "sub_r", "div_l", "div_r"],
    )
    def test_numerical(self, op: str, x0: float, x1: float, expected: float) -> None:
        """Decomposed DAG for each UDFS orientation yields the correct value."""
        cg = _udfs(
            2,
            1,
            0,
            {0: ((), "="), 1: ((), "="), 2: ((0, 1), op), 3: ((2,), "=")},
        )
        dag = compgraph_to_labeled_dag(cg)
        result = evaluate_dag(dag, {0: x0, 1: x1})
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_sub_l_sub_r_differ(self) -> None:
        """sub_l and sub_r produce distinct numeric results on asymmetric inputs."""
        node_dict_l: dict[int, tuple[tuple[int, ...], str]] = {
            0: ((), "="),
            1: ((), "="),
            2: ((0, 1), "sub_l"),
            3: ((2,), "="),
        }
        node_dict_r: dict[int, tuple[tuple[int, ...], str]] = {
            0: ((), "="),
            1: ((), "="),
            2: ((0, 1), "sub_r"),
            3: ((2,), "="),
        }
        r_l = evaluate_dag(compgraph_to_labeled_dag(_udfs(2, 1, 0, node_dict_l)), {0: 6.0, 1: 3.0})
        r_r = evaluate_dag(compgraph_to_labeled_dag(_udfs(2, 1, 0, node_dict_r)), {0: 6.0, 1: 3.0})
        assert r_l != r_r


# ======================================================================
# AC-3: Self-reference (x - x, x / x)
# ======================================================================


class TestSelfReference:
    """AC-3: Self-referential SUB and DIV evaluate correctly after decomposition.

    Before T16 the undecomposed SUB/DIV kept only one in-edge (duplicate
    add_edge was silently rejected), so x-x evaluated as x and x/x as x.
    """

    def test_bingo_sub_self_zero(self) -> None:
        """Bingo x - x = 0 after decomposition."""
        dag = agraph_to_labeled_dag(_bingo([[0, 0, 0], [3, 0, 0]]))
        assert NodeType.NEG in _all_labels(dag)
        assert NodeType.ADD in _all_labels(dag)
        np.testing.assert_allclose(evaluate_dag(dag, {0: 7.0}), 0.0, atol=1e-12)

    def test_bingo_div_self_one(self) -> None:
        """Bingo x / x = 1 after decomposition."""
        dag = agraph_to_labeled_dag(_bingo([[0, 0, 0], [5, 0, 0]]))
        assert NodeType.INV in _all_labels(dag)
        assert NodeType.MUL in _all_labels(dag)
        np.testing.assert_allclose(evaluate_dag(dag, {0: 7.0}), 1.0, rtol=1e-12)

    @pytest.mark.parametrize("x0", [1.0, 3.14, -2.5, 100.0])
    def test_bingo_sub_self_zero_parametrized(self, x0: float) -> None:
        """x - x = 0 for various x values."""
        dag = agraph_to_labeled_dag(_bingo([[0, 0, 0], [3, 0, 0]]))
        np.testing.assert_allclose(evaluate_dag(dag, {0: x0}), 0.0, atol=1e-12)

    def test_udfs_sub_l_self_zero(self) -> None:
        """UDFS sub_l(x, x) = 0 after decomposition."""
        cg = _udfs(1, 1, 0, {0: ((), "="), 1: ((0, 0), "sub_l"), 2: ((1,), "=")})
        dag = compgraph_to_labeled_dag(cg)
        np.testing.assert_allclose(evaluate_dag(dag, {0: 7.0}), 0.0, atol=1e-12)

    def test_udfs_div_l_self_one(self) -> None:
        """UDFS div_l(x, x) = 1 after decomposition."""
        cg = _udfs(1, 1, 0, {0: ((), "="), 1: ((0, 0), "div_l"), 2: ((1,), "=")})
        dag = compgraph_to_labeled_dag(cg)
        np.testing.assert_allclose(evaluate_dag(dag, {0: 7.0}), 1.0, rtol=1e-12)


# ======================================================================
# AC-4: POW is untouched
# ======================================================================


class TestPowUntouched:
    """AC-4: POW nodes are emitted unchanged; no NEG/INV/ADD/MUL is introduced."""

    def test_bingo_pow_label(self) -> None:
        """x_0 ** x_1 maps to a single POW node."""
        dag = agraph_to_labeled_dag(_bingo([[0, 0, 0], [0, 1, 0], [10, 0, 1]]))
        assert dag.node_label(2) == NodeType.POW
        labels = _all_labels(dag)
        assert NodeType.NEG not in labels
        assert NodeType.ADD not in labels

    def test_bingo_pow_self_reference_unchanged(self) -> None:
        """x_0 ** x_0: single in-edge as before, no decomposition applied."""
        dag = agraph_to_labeled_dag(_bingo([[0, 0, 0], [10, 0, 0]]))
        assert dag.node_label(1) == NodeType.POW
        labels = _all_labels(dag)
        assert NodeType.NEG not in labels
        assert NodeType.INV not in labels

    def test_bingo_pow_operand_order(self) -> None:
        """POW(x0, x1): ordered_inputs gives [x0, x1] = [0, 1]."""
        dag = agraph_to_labeled_dag(_bingo([[0, 0, 0], [0, 1, 0], [10, 0, 1]]))
        inputs = dag.ordered_inputs(2)
        assert inputs == [0, 1]

    def test_bingo_pow_numerical(self) -> None:
        """2 ** 3 = 8 via the decomposed adapter.

        rtol=1e-9 because the evaluator computes 2**3 via exp(3*log(2)),
        introducing ~1e-10 relative error that is not a correctness failure.
        """
        dag = agraph_to_labeled_dag(_bingo([[0, 0, 0], [0, 1, 0], [10, 0, 1]]))
        result = evaluate_dag(dag, {0: 2.0, 1: 3.0})
        np.testing.assert_allclose(result, 8.0, rtol=1e-9)


# ======================================================================
# AC-5: Nested / mixed expressions
# ======================================================================


class TestNestedMixed:
    """AC-5: Compound expressions with multiple SUB/DIV evaluate correctly."""

    def test_bingo_sub_div_nested_numerical(self) -> None:
        """(x0 - x1) / (x0 + x1) evaluates correctly via the decomposed adapter."""
        # Row 0: x0, Row 1: x1, Row 2: x0-x1, Row 3: x0+x1, Row 4: (x0-x1)/(x0+x1)
        cmd = [
            [0, 0, 0],
            [0, 1, 0],
            [3, 0, 1],  # x0 - x1 -> ADD(x0, NEG(x1))
            [2, 0, 1],  # x0 + x1
            [5, 2, 3],  # (x0-x1) / (x0+x1) -> MUL(ADD, INV(ADD_sum))
        ]
        dag = agraph_to_labeled_dag(_bingo(cmd))
        labels = _all_labels(dag)
        assert labels.isdisjoint(FORBIDDEN_ADAPTER_LABELS)
        x0, x1 = 5.0, 3.0
        result = evaluate_dag(dag, {0: x0, 1: x1})
        expected = (x0 - x1) / (x0 + x1)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_udfs_double_sub_numerical(self) -> None:
        """(x0 - x1) - x0 = -x1 evaluates correctly via two decompositions."""
        cg = _udfs(
            2,
            1,
            0,
            {
                0: ((), "="),
                1: ((), "="),
                2: ((0, 1), "sub_l"),  # x0 - x1
                3: ((2, 0), "sub_l"),  # (x0-x1) - x0 = -x1
                4: ((3,), "="),
            },
        )
        dag = compgraph_to_labeled_dag(cg)
        labels = _all_labels(dag)
        assert labels.isdisjoint(FORBIDDEN_ADAPTER_LABELS)
        result = evaluate_dag(dag, {0: 5.0, 1: 3.0})
        np.testing.assert_allclose(result, -3.0, rtol=1e-12)


# ======================================================================
# AC-7: Legacy regression path (decompose=False)
# ======================================================================


class TestLegacyRegression:
    """AC-7: decompose=False reproduces the pre-T16 label set exactly."""

    def test_bingo_legacy_sub_label(self) -> None:
        """decompose=False: SUB node is emitted (old behaviour)."""
        dag = agraph_to_labeled_dag(_bingo([[0, 0, 0], [0, 1, 0], [3, 0, 1]]), decompose=False)
        assert dag.node_label(2) == NodeType.SUB
        assert NodeType.NEG not in _all_labels(dag)
        assert NodeType.ADD not in _all_labels(dag)

    def test_bingo_legacy_sub_operand_order(self) -> None:
        """decompose=False: ordered_inputs on SUB node gives [0, 1]."""
        dag = agraph_to_labeled_dag(_bingo([[0, 0, 0], [0, 1, 0], [3, 0, 1]]), decompose=False)
        inputs = dag.ordered_inputs(2)
        assert inputs == [0, 1]

    def test_bingo_legacy_div_label(self) -> None:
        """decompose=False: DIV node is emitted (old behaviour)."""
        dag = agraph_to_labeled_dag(_bingo([[0, 0, 0], [0, 1, 0], [5, 0, 1]]), decompose=False)
        assert dag.node_label(2) == NodeType.DIV
        assert NodeType.INV not in _all_labels(dag)
        assert NodeType.MUL not in _all_labels(dag)

    def test_bingo_legacy_node_count(self) -> None:
        """decompose=False: x0-x1 produces 3 nodes, not 4."""
        dag = agraph_to_labeled_dag(_bingo([[0, 0, 0], [0, 1, 0], [3, 0, 1]]), decompose=False)
        assert dag.node_count == 3

    def test_udfs_legacy_sub_l_label(self) -> None:
        """decompose=False: sub_l maps to SUB in UDFS adapter."""
        cg = _udfs(
            2,
            1,
            0,
            {0: ((), "="), 1: ((), "="), 2: ((0, 1), "sub_l"), 3: ((2,), "=")},
        )
        dag = compgraph_to_labeled_dag(cg, decompose=False)
        assert dag.node_label(2) == NodeType.SUB

    def test_udfs_legacy_div_l_label(self) -> None:
        """decompose=False: div_l maps to DIV in UDFS adapter."""
        cg = _udfs(
            2,
            1,
            0,
            {0: ((), "="), 1: ((), "="), 2: ((0, 1), "div_l"), 3: ((2,), "=")},
        )
        dag = compgraph_to_labeled_dag(cg, decompose=False)
        assert dag.node_label(2) == NodeType.DIV


# ======================================================================
# AC-8: Sharing (share_unary=True vs False)
# ======================================================================


class TestSharing:
    """AC-8: share_unary controls whether Neg/Inv nodes are reused.

    Expression: (x0 - x2) + (x1 - x2). Both SUB nodes wrap x2 in NEG.
    With sharing: one NEG node, out_degree 2.
    Without sharing: two NEG nodes, each out_degree 1.
    Both must evaluate to the same numeric value.
    """

    # (x0-x2)+(x1-x2): 3 vars, 2 SUBs, 1 ADD
    _CMD: list[list[int]] = [
        [0, 0, 0],  # x0
        [0, 1, 0],  # x1
        [0, 2, 0],  # x2
        [3, 0, 2],  # x0 - x2
        [3, 1, 2],  # x1 - x2
        [2, 3, 4],  # (x0-x2) + (x1-x2)
    ]

    def test_share_false_two_neg_nodes(self) -> None:
        """share_unary=False produces two independent NEG nodes."""
        dag = agraph_to_labeled_dag(_bingo(self._CMD), share_unary=False)
        neg_count = sum(1 for i in range(dag.node_count) if dag.node_label(i) == NodeType.NEG)
        assert neg_count == 2

    def test_share_false_neg_out_degree_one(self) -> None:
        """share_unary=False: each NEG node has out_degree 1."""
        dag = agraph_to_labeled_dag(_bingo(self._CMD), share_unary=False)
        for i in range(dag.node_count):
            if dag.node_label(i) == NodeType.NEG:
                assert dag.out_degree(i) == 1

    def test_share_true_one_neg_node(self) -> None:
        """share_unary=True produces exactly one NEG node."""
        dag = agraph_to_labeled_dag(_bingo(self._CMD), share_unary=True)
        neg_count = sum(1 for i in range(dag.node_count) if dag.node_label(i) == NodeType.NEG)
        assert neg_count == 1

    def test_share_true_neg_out_degree_two(self) -> None:
        """share_unary=True: the single NEG node has out_degree 2."""
        dag = agraph_to_labeled_dag(_bingo(self._CMD), share_unary=True)
        neg_nodes = [i for i in range(dag.node_count) if dag.node_label(i) == NodeType.NEG]
        assert len(neg_nodes) == 1
        assert dag.out_degree(neg_nodes[0]) == 2

    @pytest.mark.parametrize("x0,x1,x2", [(5.0, 3.0, 1.0), (2.0, 7.0, 4.0)])
    def test_both_share_modes_evaluate_same(self, x0: float, x1: float, x2: float) -> None:
        """Both share modes produce numerically identical results."""
        dag_shared = agraph_to_labeled_dag(_bingo(self._CMD), share_unary=True)
        dag_indep = agraph_to_labeled_dag(_bingo(self._CMD), share_unary=False)
        r_shared = evaluate_dag(dag_shared, {0: x0, 1: x1, 2: x2})
        r_indep = evaluate_dag(dag_indep, {0: x0, 1: x1, 2: x2})
        np.testing.assert_allclose(r_shared, r_indep, rtol=1e-12)
        expected = (x0 - x2) + (x1 - x2)
        np.testing.assert_allclose(r_indep, expected, rtol=1e-12)


# ======================================================================
# AC-9: undecompose round-trips and raises on shared
# ======================================================================


class TestUndecompose:
    """AC-9: undecompose reverts ADD+NEG -> SUB and MUL+INV -> DIV."""

    def test_undecompose_sub_removes_neg(self) -> None:
        """undecompose(ADD(x0, NEG(x1))) -> DAG with SUB, no NEG."""
        dag = agraph_to_labeled_dag(_bingo([[0, 0, 0], [0, 1, 0], [3, 0, 1]]))
        assert contains_decomposed_unary(dag)
        restored = undecompose(dag)
        labels = _all_labels(restored)
        assert NodeType.SUB in labels
        assert NodeType.NEG not in labels

    def test_undecompose_div_removes_inv(self) -> None:
        """undecompose(MUL(x0, INV(x1))) -> DAG with DIV, no INV."""
        dag = agraph_to_labeled_dag(_bingo([[0, 0, 0], [0, 1, 0], [5, 0, 1]]))
        assert contains_decomposed_unary(dag)
        restored = undecompose(dag)
        labels = _all_labels(restored)
        assert NodeType.DIV in labels
        assert NodeType.INV not in labels

    def test_undecompose_no_effect_on_clean_dag(self) -> None:
        """undecompose leaves a DAG without NEG/INV unchanged (same object)."""
        dag = agraph_to_labeled_dag(_bingo([[0, 0, 0], [0, 1, 0], [2, 0, 1]]))
        assert not contains_decomposed_unary(dag)
        result = undecompose(dag)
        assert result is dag

    def test_undecompose_preserves_operand_order_sub(self) -> None:
        """undecompose preserves operand order: x0-x1 != x1-x0 after inversion."""
        dag_fwd = agraph_to_labeled_dag(_bingo([[0, 0, 0], [0, 1, 0], [3, 0, 1]]))
        dag_rev = agraph_to_labeled_dag(_bingo([[0, 0, 0], [0, 1, 0], [3, 1, 0]]))
        fwd_restored = undecompose(dag_fwd)
        rev_restored = undecompose(dag_rev)
        # After undecompose the SUB node's ordered_inputs should differ
        sub_fwd = next(
            i for i in range(fwd_restored.node_count) if fwd_restored.node_label(i) == NodeType.SUB
        )
        sub_rev = next(
            i for i in range(rev_restored.node_count) if rev_restored.node_label(i) == NodeType.SUB
        )
        assert fwd_restored.ordered_inputs(sub_fwd) != rev_restored.ordered_inputs(sub_rev)

    def test_undecompose_raises_on_shared_neg(self) -> None:
        """undecompose raises ValueError when a NEG node has out_degree > 1."""
        cmd = [
            [0, 0, 0],  # x0
            [0, 1, 0],  # x1
            [0, 2, 0],  # x2
            [3, 0, 2],  # x0 - x2
            [3, 1, 2],  # x1 - x2  (same NEG(x2) reused)
            [2, 3, 4],  # (x0-x2) + (x1-x2)
        ]
        dag = agraph_to_labeled_dag(_bingo(cmd), share_unary=True)
        # Verify sharing occurred
        neg_nodes = [i for i in range(dag.node_count) if dag.node_label(i) == NodeType.NEG]
        assert len(neg_nodes) == 1
        assert dag.out_degree(neg_nodes[0]) == 2
        with pytest.raises(ValueError, match="shared"):
            undecompose(dag)

    def test_bingo_roundtrip_through_adapter(self) -> None:
        """agraph -> labeled_dag -> agraph roundtrip evaluates identically."""
        from experiments.models.bingo.adapter import labeled_dag_to_agraph

        ag = _bingo([[0, 0, 0], [0, 1, 0], [3, 0, 1]])
        dag = agraph_to_labeled_dag(ag)
        ag2 = labeled_dag_to_agraph(dag)
        x = np.array([[5.0, 3.0], [10.0, 4.0]])
        np.testing.assert_allclose(
            ag.evaluate_equation_at(x), ag2.evaluate_equation_at(x), rtol=1e-12
        )


# ======================================================================
# AC-1: Capacity – many Sub/Div near Bingo production stack size (32)
# ======================================================================


class TestCapacity:
    """AC-1: Many decomposable ops must not cause RuntimeError on add_node."""

    def test_bingo_many_sub_no_overflow(self) -> None:
        """15 chained SUB ops on 2 vars; no RuntimeError from LabeledDAG.add_node."""
        # Build a chain: x0-x1, (prev)-x0, (prev)-x0, ...
        cmd: list[list[int]] = [[0, 0, 0], [0, 1, 0]]
        prev = 1  # last row index producing a result
        for _ in range(15):
            cmd.append([3, prev, 0])  # SUB(prev, x0)
            prev = len(cmd) - 1
        dag = agraph_to_labeled_dag(_bingo(cmd))
        # Must have completed without error; verify structure
        labels = _all_labels(dag)
        assert labels.isdisjoint(FORBIDDEN_ADAPTER_LABELS)

    def test_bingo_mixed_sub_div_no_overflow(self) -> None:
        """Alternating SUB and DIV ops near stack size 32."""
        cmd: list[list[int]] = [[0, 0, 0], [0, 1, 0]]
        prev = 0
        for i in range(14):
            op = 3 if i % 2 == 0 else 5  # alternate SUB / DIV
            cmd.append([op, prev, 1])
            prev = len(cmd) - 1
        dag = agraph_to_labeled_dag(_bingo(cmd))
        labels = _all_labels(dag)
        assert labels.isdisjoint(FORBIDDEN_ADAPTER_LABELS)

    def test_extra_node_budget_matches_actual_growth(self) -> None:
        """extra_node_budget(n, decompose=True) == n (one Neg/Inv per op)."""
        for n in [0, 1, 5, 15, 30]:
            assert extra_node_budget(n, decompose=True) == n

    def test_extra_node_budget_zero_when_disabled(self) -> None:
        """extra_node_budget returns 0 when decompose=False."""
        assert extra_node_budget(10, decompose=False) == 0


# ======================================================================
# AC-10: UDFS host neg/inv pass through unchanged
# ======================================================================


class TestUDFSNativeUnaryPassThrough:
    """AC-10: UDFS native neg/inv are mapped directly to NEG/INV, not decomposed."""

    @pytest.mark.parametrize(
        "udfs_op,expected_label",
        [("neg", NodeType.NEG), ("inv", NodeType.INV)],
    )
    def test_passthrough_default(self, udfs_op: str, expected_label: NodeType) -> None:
        """With default decompose=True, host neg/inv still map to NEG/INV."""
        cg = _udfs(1, 1, 0, {0: ((), "="), 1: ((0,), udfs_op), 2: ((1,), "=")})
        dag = compgraph_to_labeled_dag(cg)
        assert dag.node_label(1) == expected_label

    @pytest.mark.parametrize(
        "udfs_op,expected_label",
        [("neg", NodeType.NEG), ("inv", NodeType.INV)],
    )
    def test_passthrough_decompose_false(self, udfs_op: str, expected_label: NodeType) -> None:
        """With decompose=False, host neg/inv still map to NEG/INV."""
        cg = _udfs(1, 1, 0, {0: ((), "="), 1: ((0,), udfs_op), 2: ((1,), "=")})
        dag = compgraph_to_labeled_dag(cg, decompose=False)
        assert dag.node_label(1) == expected_label

    def test_neg_not_treated_as_sub_decomposition(self) -> None:
        """UDFS host neg is a pure unary pass-through: no ADD sibling is created."""
        cg = _udfs(1, 1, 0, {0: ((), "="), 1: ((0,), "neg"), 2: ((1,), "=")})
        dag = compgraph_to_labeled_dag(cg)
        assert NodeType.ADD not in _all_labels(dag)
        assert dag.node_count == 2  # VAR + NEG only

    def test_neg_numerical_correct(self) -> None:
        """UDFS host neg(x) evaluates to -x."""
        cg = _udfs(1, 1, 0, {0: ((), "="), 1: ((0,), "neg"), 2: ((1,), "=")})
        dag = compgraph_to_labeled_dag(cg)
        result = evaluate_dag(dag, {0: 5.0})
        np.testing.assert_allclose(result, -5.0, rtol=1e-12)

    def test_inv_numerical_correct(self) -> None:
        """UDFS host inv(x) evaluates to 1/x."""
        cg = _udfs(1, 1, 0, {0: ((), "="), 1: ((0,), "inv"), 2: ((1,), "=")})
        dag = compgraph_to_labeled_dag(cg)
        result = evaluate_dag(dag, {0: 4.0})
        np.testing.assert_allclose(result, 0.25, rtol=1e-12)


# ======================================================================
# Direct tests of commutative_encoding module functions
# ======================================================================


class TestEmitBinaryDirect:
    """Direct tests of emit_binary without going through an adapter."""

    def _two_var_dag(self) -> tuple[LabeledDAG, int, int]:
        """Build a two-variable skeleton DAG for direct emit_binary calls.

        Returns:
            Tuple of (dag, node_id_x0, node_id_x1).
        """
        dag = LabeledDAG(max_nodes=20)
        n0 = dag.add_node(NodeType.VAR, var_index=0)
        n1 = dag.add_node(NodeType.VAR, var_index=1)
        return dag, n0, n1

    def test_emit_sub_produces_neg_then_add(self) -> None:
        """emit_binary(SUB, a, b) -> NEG(b) at lower id, ADD at higher id."""
        dag, n0, n1 = self._two_var_dag()
        result = emit_binary(dag, NodeType.SUB, n0, n1, decompose=True)
        # Node ordering: NEG inserted first, then ADD
        assert dag.node_label(result) == NodeType.ADD
        assert dag.node_label(result - 1) == NodeType.NEG

    def test_emit_div_produces_inv_then_mul(self) -> None:
        """emit_binary(DIV, a, b) -> INV(b) at lower id, MUL at higher id."""
        dag, n0, n1 = self._two_var_dag()
        result = emit_binary(dag, NodeType.DIV, n0, n1, decompose=True)
        assert dag.node_label(result) == NodeType.MUL
        assert dag.node_label(result - 1) == NodeType.INV

    def test_emit_pow_unchanged(self) -> None:
        """emit_binary(POW, ...) emits a single POW node."""
        dag, n0, n1 = self._two_var_dag()
        result = emit_binary(dag, NodeType.POW, n0, n1, decompose=True)
        assert dag.node_label(result) == NodeType.POW

    def test_emit_add_unchanged(self) -> None:
        """emit_binary(ADD, ...) emits a single ADD node."""
        dag, n0, n1 = self._two_var_dag()
        result = emit_binary(dag, NodeType.ADD, n0, n1, decompose=True)
        assert dag.node_label(result) == NodeType.ADD

    def test_emit_binary_decompose_false_sub(self) -> None:
        """decompose=False: emit_binary(SUB, ...) emits SUB directly."""
        dag, n0, n1 = self._two_var_dag()
        result = emit_binary(dag, NodeType.SUB, n0, n1, decompose=False)
        assert dag.node_label(result) == NodeType.SUB

    def test_emit_binary_none_first_skips_edge(self) -> None:
        """first=None: no edge from first operand."""
        dag, _n0, n1 = self._two_var_dag()
        result = emit_binary(dag, NodeType.ADD, None, n1, decompose=True)
        assert dag.in_degree(result) == 1  # only n1 contributed an edge

    def test_emit_binary_none_second_no_wrapped_node(self) -> None:
        """second=None with decompose: no NEG/INV node is created."""
        dag, n0, _n1 = self._two_var_dag()
        before = dag.node_count
        result = emit_binary(dag, NodeType.SUB, n0, None, decompose=True)
        # Only one new node (ADD): NEG was not created because second is None
        assert dag.node_count == before + 1
        assert dag.node_label(result) == NodeType.ADD


class TestNewUnaryCache:
    """Tests for the new_unary_cache factory."""

    def test_none_when_sharing_disabled(self) -> None:
        """new_unary_cache(share=False) returns None."""
        assert new_unary_cache(share=False) is None

    def test_dict_when_sharing_enabled(self) -> None:
        """new_unary_cache(share=True) returns an empty dict."""
        cache = new_unary_cache(share=True)
        assert isinstance(cache, dict)
        assert len(cache) == 0

    def test_module_default_respected(self) -> None:
        """new_unary_cache(share=None) follows SHARE_DECOMPOSED_UNARY."""
        result = new_unary_cache(share=None)
        if SHARE_DECOMPOSED_UNARY:
            assert isinstance(result, dict)
        else:
            assert result is None
