"""Tests for evaluation preservation across canonical round-trip.

Targets the bug where canonical string computation reorders operands of
non-commutative binary operations (SUB, DIV, POW), causing:
    eval(D, x) != eval(S2D(canonical(D)), x)

Root cause: The evaluator uses sorted(in_neighbors) to determine operand
order for binary ops, but node IDs can change during canonical round-trip.
The canonical search picks the lexmin string, which may create nodes in a
different order (e.g., COS before SIN because 'c' < 's'), flipping operands.

Fix: Track explicit operand order (_input_order) in LabeledDAG, use it in
the evaluator, and constrain D2S/canonical to respect it for binary ops.

References:
    - Lopez-Rubio (2025). arXiv:2512.10429v2. IsalGraph.
    - This test implements the advisor's request to verify:
      eval(D, x) = eval(S2D(canonical(D)), x)
"""

from __future__ import annotations

import pytest

from isalsr.core.canonical import canonical_string
from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.string_to_dag import StringToDAG

# ======================================================================
# Helper: canonical evaluation round-trip
# ======================================================================


def _assert_canonical_eval_roundtrip(
    dag: LabeledDAG,
    inputs: dict[int, float],
    num_variables: int,
    *,
    description: str = "",
) -> None:
    """Assert that evaluation is preserved through canonical round-trip.

    Checks: eval(D, x) == eval(S2D(canonical(D)), x)

    Args:
        dag: The expression DAG.
        inputs: Mapping from var_index to scalar values.
        num_variables: Number of input variables.
        description: Human-readable description for error messages.
    """
    val_before = evaluate_dag(dag, inputs)
    canon = canonical_string(dag)
    dag2 = StringToDAG(canon, num_variables=num_variables).run()
    val_after = evaluate_dag(dag2, inputs)
    assert val_before == pytest.approx(val_after, abs=1e-8), (
        f"Canonical evaluation round-trip failed{f' ({description})' if description else ''}.\n"
        f"  Original eval: {val_before}\n"
        f"  After canonical: {val_after}\n"
        f"  Canonical string: {canon!r}\n"
        f"  Inputs: {inputs}"
    )


def _assert_d2s_eval_roundtrip(
    dag: LabeledDAG,
    inputs: dict[int, float],
    num_variables: int,
    *,
    description: str = "",
) -> None:
    """Assert that evaluation is preserved through D2S round-trip.

    Checks: eval(D, x) == eval(S2D(D2S(D)), x)

    Args:
        dag: The expression DAG.
        inputs: Mapping from var_index to scalar values.
        num_variables: Number of input variables.
        description: Human-readable description for error messages.
    """
    val_before = evaluate_dag(dag, inputs)
    string = DAGToString(dag).run()
    dag2 = StringToDAG(string, num_variables=num_variables).run()
    val_after = evaluate_dag(dag2, inputs)
    assert val_before == pytest.approx(val_after, abs=1e-8), (
        f"D2S evaluation round-trip failed{f' ({description})' if description else ''}.\n"
        f"  Original eval: {val_before}\n"
        f"  After D2S: {val_after}\n"
        f"  D2S string: {string!r}\n"
        f"  Inputs: {inputs}"
    )


# ======================================================================
# SUB: Non-commutative subtraction
# ======================================================================


class TestSubCanonicalRoundTrip:
    """Canonical round-trip for expressions involving SUB.

    SUB(a, b) = a - b, where a = first operand (lower node ID in current
    convention). The canonical search may swap SIN and COS creation order
    (since 'c' < 's'), causing the operand order to flip.
    """

    def test_sin_minus_cos(self) -> None:
        """sin(x) - cos(x): canonical should preserve evaluation.

        This is the primary reproducer. The canonical string prefers
        'c' < 's', so it creates COS before SIN, which swaps their
        node IDs and flips the subtraction.
        """
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)  # 0: x
        dag.add_node(NodeType.SIN)  # 1: sin
        dag.add_node(NodeType.COS)  # 2: cos
        dag.add_node(NodeType.SUB)  # 3: sub
        dag.add_edge(0, 1)  # x -> sin
        dag.add_edge(0, 2)  # x -> cos
        dag.add_edge(1, 3)  # sin -> sub (first operand)
        dag.add_edge(2, 3)  # cos -> sub (second operand)

        # sin(1.5) - cos(1.5) = 0.9975 - 0.0707 ≈ 0.9268
        _assert_canonical_eval_roundtrip(dag, {0: 1.5}, 1, description="sin(x) - cos(x)")

    def test_cos_minus_sin(self) -> None:
        """cos(x) - sin(x): the reverse expression, also must be preserved."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)  # 0: x
        dag.add_node(NodeType.COS)  # 1: cos
        dag.add_node(NodeType.SIN)  # 2: sin
        dag.add_node(NodeType.SUB)  # 3: sub
        dag.add_edge(0, 1)  # x -> cos
        dag.add_edge(0, 2)  # x -> sin
        dag.add_edge(1, 3)  # cos -> sub (first operand)
        dag.add_edge(2, 3)  # sin -> sub (second operand)

        # cos(1.5) - sin(1.5) ≈ -0.9268
        _assert_canonical_eval_roundtrip(dag, {0: 1.5}, 1, description="cos(x) - sin(x)")

    def test_sin_x_minus_cos_x_differ(self) -> None:
        """sin(x) - cos(x) and cos(x) - sin(x) must have different canonical strings."""
        dag1 = LabeledDAG(max_nodes=5)
        dag1.add_node(NodeType.VAR, var_index=0)
        dag1.add_node(NodeType.SIN)
        dag1.add_node(NodeType.COS)
        dag1.add_node(NodeType.SUB)
        dag1.add_edge(0, 1)
        dag1.add_edge(0, 2)
        dag1.add_edge(1, 3)  # sin first
        dag1.add_edge(2, 3)

        dag2 = LabeledDAG(max_nodes=5)
        dag2.add_node(NodeType.VAR, var_index=0)
        dag2.add_node(NodeType.COS)
        dag2.add_node(NodeType.SIN)
        dag2.add_node(NodeType.SUB)
        dag2.add_edge(0, 1)
        dag2.add_edge(0, 2)
        dag2.add_edge(1, 3)  # cos first
        dag2.add_edge(2, 3)

        c1 = canonical_string(dag1)
        c2 = canonical_string(dag2)

        # These are different expressions, so they must have different canonical strings.
        assert c1 != c2, (
            f"sin(x)-cos(x) and cos(x)-sin(x) produced the same canonical string: {c1!r}"
        )

    def test_x_minus_y(self) -> None:
        """x - y: both operands are VARs with fixed IDs (simpler case)."""
        dag = LabeledDAG(max_nodes=4)
        dag.add_node(NodeType.VAR, var_index=0)  # 0: x
        dag.add_node(NodeType.VAR, var_index=1)  # 1: y
        dag.add_node(NodeType.SUB)  # 2: sub
        dag.add_edge(0, 2)  # x -> sub (first operand)
        dag.add_edge(1, 2)  # y -> sub (second operand)

        _assert_canonical_eval_roundtrip(dag, {0: 3.0, 1: 7.0}, 2, description="x - y")

    def test_add_minus_const(self) -> None:
        """(x + y) - const: operands are non-VAR with potentially swapped IDs.

        If canonical creates CONST before ADD (label 'k' vs '+', '+' < 'k'),
        the operand order for SUB could change.

        Note: Uses const_value=1.0 (the S2D default). String encoding does
        not preserve constant values; they are optimized separately.
        """
        dag = LabeledDAG(max_nodes=6)
        dag.add_node(NodeType.VAR, var_index=0)  # 0: x
        dag.add_node(NodeType.VAR, var_index=1)  # 1: y
        dag.add_node(NodeType.ADD)  # 2: add
        dag.add_node(NodeType.CONST, const_value=1.0)  # 3: const (S2D default)
        dag.add_node(NodeType.SUB)  # 4: sub
        dag.add_edge(0, 2)  # x -> add
        dag.add_edge(1, 2)  # y -> add
        dag.add_edge(0, 3)  # x -> const (reachability edge)
        dag.add_edge(2, 4)  # add -> sub (first operand)
        dag.add_edge(3, 4)  # const -> sub (second operand)

        _assert_canonical_eval_roundtrip(dag, {0: 3.0, 1: 7.0}, 2, description="(x+y) - const")

    def test_exp_minus_log(self) -> None:
        """exp(x) - log(x): canonical may reorder ('e' < 'l' alphabetically)."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)  # 0: x
        dag.add_node(NodeType.EXP)  # 1: exp
        dag.add_node(NodeType.LOG)  # 2: log
        dag.add_node(NodeType.SUB)  # 3: sub
        dag.add_edge(0, 1)  # x -> exp
        dag.add_edge(0, 2)  # x -> log
        dag.add_edge(1, 3)  # exp -> sub (first operand)
        dag.add_edge(2, 3)  # log -> sub (second operand)

        # Use x=2.0: exp(2) - log(2) ≈ 7.389 - 0.693 ≈ 6.696
        _assert_canonical_eval_roundtrip(dag, {0: 2.0}, 1, description="exp(x) - log(x)")


# ======================================================================
# DIV: Non-commutative division
# ======================================================================


class TestDivCanonicalRoundTrip:
    """Canonical round-trip for expressions involving DIV."""

    def test_sin_div_cos(self) -> None:
        """sin(x) / cos(x) = tan(x): canonical must preserve evaluation."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)  # 0: x
        dag.add_node(NodeType.SIN)  # 1: sin
        dag.add_node(NodeType.COS)  # 2: cos
        dag.add_node(NodeType.DIV)  # 3: div
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(1, 3)  # sin / cos (first operand = sin)
        dag.add_edge(2, 3)

        # sin(1.0) / cos(1.0) = tan(1.0) ≈ 1.5574
        _assert_canonical_eval_roundtrip(dag, {0: 1.0}, 1, description="sin(x) / cos(x)")

    def test_x_div_y(self) -> None:
        """x / y with specific values."""
        dag = LabeledDAG(max_nodes=4)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.DIV)
        dag.add_edge(0, 2)  # x / y
        dag.add_edge(1, 2)

        # x=10, y=3 -> 10/3 ≈ 3.333
        _assert_canonical_eval_roundtrip(dag, {0: 10.0, 1: 3.0}, 2, description="x / y")


# ======================================================================
# POW: Non-commutative power
# ======================================================================


class TestPowCanonicalRoundTrip:
    """Canonical round-trip for expressions involving POW."""

    def test_sin_pow_cos(self) -> None:
        """sin(x) ^ cos(x): canonical must preserve evaluation."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)  # 0: x
        dag.add_node(NodeType.SIN)  # 1: sin
        dag.add_node(NodeType.COS)  # 2: cos
        dag.add_node(NodeType.POW)  # 3: pow
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(1, 3)  # sin ^ cos (base = sin)
        dag.add_edge(2, 3)

        _assert_canonical_eval_roundtrip(dag, {0: 1.0}, 1, description="sin(x) ^ cos(x)")

    def test_x_pow_y(self) -> None:
        """x ^ y."""
        dag = LabeledDAG(max_nodes=4)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.POW)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)

        # |2|^3 = 8 (protected pow uses |base|)
        _assert_canonical_eval_roundtrip(dag, {0: 2.0, 1: 3.0}, 2, description="x ^ y")


# ======================================================================
# D2S (greedy) evaluation round-trip with binary ops
# ======================================================================


class TestD2SEvalRoundTrip:
    """D2S (greedy) evaluation round-trip tests.

    These may or may not fail depending on frozenset iteration order,
    but they must be correct for the implementation to be sound.
    """

    def test_sin_minus_cos_d2s(self) -> None:
        """sin(x) - cos(x) through D2S round-trip."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.COS)
        dag.add_node(NodeType.SUB)
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(1, 3)
        dag.add_edge(2, 3)

        _assert_d2s_eval_roundtrip(dag, {0: 1.5}, 1, description="sin(x) - cos(x)")

    def test_sin_div_cos_d2s(self) -> None:
        """sin(x) / cos(x) through D2S round-trip."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.COS)
        dag.add_node(NodeType.DIV)
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(1, 3)
        dag.add_edge(2, 3)

        _assert_d2s_eval_roundtrip(dag, {0: 1.0}, 1, description="sin(x) / cos(x)")


# ======================================================================
# Complex expressions with multiple binary ops
# ======================================================================


class TestComplexBinaryOps:
    """More complex expressions combining multiple non-commutative ops."""

    def test_nested_sub(self) -> None:
        """(sin(x) + cos(x)) - exp(x): SUB with complex operands."""
        dag = LabeledDAG(max_nodes=6)
        dag.add_node(NodeType.VAR, var_index=0)  # 0: x
        dag.add_node(NodeType.SIN)  # 1: sin
        dag.add_node(NodeType.COS)  # 2: cos
        dag.add_node(NodeType.ADD)  # 3: add
        dag.add_node(NodeType.EXP)  # 4: exp
        dag.add_node(NodeType.SUB)  # 5: sub
        dag.add_edge(0, 1)  # x -> sin
        dag.add_edge(0, 2)  # x -> cos
        dag.add_edge(1, 3)  # sin -> add
        dag.add_edge(2, 3)  # cos -> add
        dag.add_edge(0, 4)  # x -> exp
        dag.add_edge(3, 5)  # add -> sub (first operand)
        dag.add_edge(4, 5)  # exp -> sub (second operand)

        _assert_canonical_eval_roundtrip(dag, {0: 0.5}, 1, description="(sin(x)+cos(x)) - exp(x)")

    def test_div_over_sub(self) -> None:
        """(x - y) / (x + y): nested non-commutative in commutative."""
        dag = LabeledDAG(max_nodes=6)
        dag.add_node(NodeType.VAR, var_index=0)  # 0: x
        dag.add_node(NodeType.VAR, var_index=1)  # 1: y
        dag.add_node(NodeType.SUB)  # 2: sub
        dag.add_node(NodeType.ADD)  # 3: add
        dag.add_node(NodeType.DIV)  # 4: div
        dag.add_edge(0, 2)  # x -> sub
        dag.add_edge(1, 2)  # y -> sub
        dag.add_edge(0, 3)  # x -> add
        dag.add_edge(1, 3)  # y -> add
        dag.add_edge(2, 4)  # sub -> div (first operand)
        dag.add_edge(3, 4)  # add -> div (second operand)

        _assert_canonical_eval_roundtrip(dag, {0: 5.0, 1: 3.0}, 2, description="(x-y) / (x+y)")


# ======================================================================
# Evaluation consistency across multiple input values
# ======================================================================


class TestMultipleInputValues:
    """Test canonical round-trip across multiple input values."""

    @pytest.mark.parametrize("x_val", [0.1, 0.5, 1.0, 1.5, 2.0, 3.14, -1.0, -0.5])
    def test_sin_minus_cos_parametric(self, x_val: float) -> None:
        """sin(x) - cos(x) for many x values."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.COS)
        dag.add_node(NodeType.SUB)
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(1, 3)
        dag.add_edge(2, 3)

        _assert_canonical_eval_roundtrip(
            dag, {0: x_val}, 1, description=f"sin({x_val}) - cos({x_val})"
        )


# ======================================================================
# Sanity: commutative ops should work (baseline)
# ======================================================================


class TestCommutativeOpsBaseline:
    """Commutative ops (ADD, MUL) should be unaffected by node ID ordering."""

    def test_sin_plus_cos_canonical(self) -> None:
        """sin(x) + cos(x): commutative, should always work."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.COS)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(1, 3)
        dag.add_edge(2, 3)

        _assert_canonical_eval_roundtrip(dag, {0: 1.5}, 1, description="sin(x) + cos(x)")

    def test_sin_times_cos_canonical(self) -> None:
        """sin(x) * cos(x): commutative, should always work."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.COS)
        dag.add_node(NodeType.MUL)
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(1, 3)
        dag.add_edge(2, 3)

        _assert_canonical_eval_roundtrip(dag, {0: 1.5}, 1, description="sin(x) * cos(x)")
