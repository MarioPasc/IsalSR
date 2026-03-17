"""Unit tests for DAG numerical evaluation.

Tests cover: basic evaluation, all operation types, protected operations,
variable-arity operations, nested expressions, and error handling.

Numerical evaluation is required for fitness computation in symbolic regression.
The evaluator must be numerically stable (protected ops) and deterministic
(sorted input order for binary ops).
"""

from __future__ import annotations

import math

import pytest

from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.errors import EvaluationError

# ======================================================================
# Helper to build simple DAGs inline
# ======================================================================


def _make_unary_dag(op: NodeType, var_value: float) -> tuple[LabeledDAG, dict[int, float]]:
    """Build op(x) DAG and return (dag, inputs)."""
    dag = LabeledDAG(max_nodes=5)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(op)
    dag.add_edge(0, 1)
    return dag, {0: var_value}


def _make_binary_dag(
    op: NodeType, x_val: float, y_val: float
) -> tuple[LabeledDAG, dict[int, float]]:
    """Build op(x, y) DAG with x(node 0) -> op, y(node 1) -> op."""
    dag = LabeledDAG(max_nodes=5)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    dag.add_node(op)
    dag.add_edge(0, 2)
    dag.add_edge(1, 2)
    return dag, {0: x_val, 1: y_val}


# ======================================================================
# Basic evaluation
# ======================================================================


class TestBasicEvaluation:
    """Simple expression evaluation."""

    def test_sin_x_at_zero(self) -> None:
        dag, inputs = _make_unary_dag(NodeType.SIN, 0.0)
        assert evaluate_dag(dag, inputs) == pytest.approx(0.0)

    def test_sin_x_at_pi_half(self) -> None:
        dag, inputs = _make_unary_dag(NodeType.SIN, math.pi / 2)
        assert evaluate_dag(dag, inputs) == pytest.approx(1.0)

    def test_x_plus_y(self) -> None:
        dag, inputs = _make_binary_dag(NodeType.ADD, 1.0, 2.0)
        assert evaluate_dag(dag, inputs) == pytest.approx(3.0)

    def test_x_times_y(self) -> None:
        dag, inputs = _make_binary_dag(NodeType.MUL, 3.0, 4.0)
        assert evaluate_dag(dag, inputs) == pytest.approx(12.0)

    def test_const_node(self) -> None:
        """Expression: x + const(3.14)."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.CONST, const_value=3.14)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)
        result = evaluate_dag(dag, {0: 1.0})
        assert result == pytest.approx(4.14)

    def test_const_default_value(self) -> None:
        """CONST with no explicit value defaults to 1.0."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.CONST)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)
        result = evaluate_dag(dag, {0: 5.0})
        assert result == pytest.approx(6.0)  # 5.0 + 1.0


# ======================================================================
# Unary operations
# ======================================================================


class TestUnaryOps:
    """All unary operation types."""

    def test_sin(self) -> None:
        dag, inputs = _make_unary_dag(NodeType.SIN, 0.0)
        assert evaluate_dag(dag, inputs) == pytest.approx(0.0)

    def test_cos(self) -> None:
        dag, inputs = _make_unary_dag(NodeType.COS, 0.0)
        assert evaluate_dag(dag, inputs) == pytest.approx(1.0)

    def test_exp(self) -> None:
        dag, inputs = _make_unary_dag(NodeType.EXP, 0.0)
        assert evaluate_dag(dag, inputs) == pytest.approx(1.0)

    def test_log(self) -> None:
        dag, inputs = _make_unary_dag(NodeType.LOG, 1.0)
        # log(|1.0| + 1e-10) ≈ log(1.0) ≈ 0.0
        assert evaluate_dag(dag, inputs) == pytest.approx(0.0, abs=1e-8)

    def test_sqrt(self) -> None:
        dag, inputs = _make_unary_dag(NodeType.SQRT, 4.0)
        assert evaluate_dag(dag, inputs) == pytest.approx(2.0)

    def test_abs_positive(self) -> None:
        dag, inputs = _make_unary_dag(NodeType.ABS, 5.0)
        assert evaluate_dag(dag, inputs) == pytest.approx(5.0)

    def test_abs_negative(self) -> None:
        dag, inputs = _make_unary_dag(NodeType.ABS, -5.0)
        assert evaluate_dag(dag, inputs) == pytest.approx(5.0)


# ======================================================================
# Binary operations
# ======================================================================


class TestBinaryOps:
    """Binary operations (SUB, DIV, POW). Input order: sorted by node ID."""

    def test_sub(self) -> None:
        """x - y where x(node 0) = 5, y(node 1) = 3 → 2."""
        dag, inputs = _make_binary_dag(NodeType.SUB, 5.0, 3.0)
        assert evaluate_dag(dag, inputs) == pytest.approx(2.0)

    def test_div(self) -> None:
        """x / y where x = 6, y = 3 → 2."""
        dag, inputs = _make_binary_dag(NodeType.DIV, 6.0, 3.0)
        assert evaluate_dag(dag, inputs) == pytest.approx(2.0)

    def test_pow(self) -> None:
        """x ^ y where x = 2, y = 3 → 8."""
        dag, inputs = _make_binary_dag(NodeType.POW, 2.0, 3.0)
        assert evaluate_dag(dag, inputs) == pytest.approx(8.0)


# ======================================================================
# Variadic operations (3+ inputs)
# ======================================================================


class TestVariadicOps:
    """ADD/MUL with 3 or more inputs."""

    def test_add_three_inputs(self) -> None:
        """x + y + z = 1 + 2 + 3 = 6."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.VAR, var_index=1)  # 1
        dag.add_node(NodeType.VAR, var_index=2)  # 2
        dag.add_node(NodeType.ADD)  # 3
        dag.add_edge(0, 3)
        dag.add_edge(1, 3)
        dag.add_edge(2, 3)
        result = evaluate_dag(dag, {0: 1.0, 1: 2.0, 2: 3.0})
        assert result == pytest.approx(6.0)

    def test_mul_three_inputs(self) -> None:
        """x * y * z = 2 * 3 * 4 = 24."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.VAR, var_index=2)
        dag.add_node(NodeType.MUL)
        dag.add_edge(0, 3)
        dag.add_edge(1, 3)
        dag.add_edge(2, 3)
        result = evaluate_dag(dag, {0: 2.0, 1: 3.0, 2: 4.0})
        assert result == pytest.approx(24.0)


# ======================================================================
# Protected operations (numerical stability)
# ======================================================================


class TestProtectedOps:
    """Numerical safety: no crashes, NaN, or Inf."""

    def test_log_zero(self) -> None:
        """log(0) should not crash (protected: log(|0| + 1e-10))."""
        dag, inputs = _make_unary_dag(NodeType.LOG, 0.0)
        result = evaluate_dag(dag, inputs)
        assert math.isfinite(result)

    def test_log_negative(self) -> None:
        """log(-5) should not crash (protected: log(|-5| + 1e-10))."""
        dag, inputs = _make_unary_dag(NodeType.LOG, -5.0)
        result = evaluate_dag(dag, inputs)
        assert math.isfinite(result)

    def test_div_by_zero(self) -> None:
        """x / 0 returns 1.0 (protected division)."""
        dag, inputs = _make_binary_dag(NodeType.DIV, 5.0, 0.0)
        result = evaluate_dag(dag, inputs)
        assert result == pytest.approx(1.0)

    def test_exp_large_positive(self) -> None:
        """exp(1000) should not overflow."""
        dag, inputs = _make_unary_dag(NodeType.EXP, 1000.0)
        result = evaluate_dag(dag, inputs)
        assert math.isfinite(result)

    def test_exp_large_negative(self) -> None:
        """exp(-1000) should not underflow to exactly 0 but stay finite."""
        dag, inputs = _make_unary_dag(NodeType.EXP, -1000.0)
        result = evaluate_dag(dag, inputs)
        assert math.isfinite(result)

    def test_sqrt_negative(self) -> None:
        """sqrt(-4) = sqrt(|-4|) = 2.0 (protected)."""
        dag, inputs = _make_unary_dag(NodeType.SQRT, -4.0)
        result = evaluate_dag(dag, inputs)
        assert result == pytest.approx(2.0)

    def test_pow_overflow(self) -> None:
        """Large power should not overflow."""
        dag, inputs = _make_binary_dag(NodeType.POW, 100.0, 100.0)
        result = evaluate_dag(dag, inputs)
        assert math.isfinite(result)


# ======================================================================
# Nested expressions
# ======================================================================


class TestNestedExpressions:
    """Compound expressions with multiple operation layers."""

    def test_sin_of_x_plus_y(self) -> None:
        """sin(x + y) at x=pi/4, y=pi/4 → sin(pi/2) = 1.0."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.VAR, var_index=1)  # 1
        dag.add_node(NodeType.ADD)  # 2
        dag.add_node(NodeType.SIN)  # 3
        dag.add_edge(0, 2)  # x -> +
        dag.add_edge(1, 2)  # y -> +
        dag.add_edge(2, 3)  # + -> sin
        result = evaluate_dag(dag, {0: math.pi / 4, 1: math.pi / 4})
        assert result == pytest.approx(1.0)

    def test_x_squared_plus_const(self) -> None:
        """x * x + 1 at x=2 → 4 + 1 = 5.

        DAG: x(0), MUL(1), CONST(2, val=1), ADD(3).
        Edges: x->MUL, x->MUL (duplicate, only one edge), x->ADD... hmm.
        Actually, x*x requires two edges from x to MUL, but duplicate edges are blocked.
        In DAG form, x*x is represented as MUL with one input x (but arity would be 1, not 2).
        This is a known limitation: x^2 should use POW instead.
        """
        # Use POW: x^2 + 1.
        dag = LabeledDAG(max_nodes=6)
        dag.add_node(NodeType.VAR, var_index=0)  # 0: x
        dag.add_node(NodeType.CONST, const_value=2.0)  # 1: const(2)
        dag.add_node(NodeType.POW)  # 2: pow
        dag.add_node(NodeType.CONST, const_value=1.0)  # 3: const(1)
        dag.add_node(NodeType.ADD)  # 4: add
        dag.add_edge(0, 2)  # x -> pow
        dag.add_edge(1, 2)  # const(2) -> pow
        dag.add_edge(2, 4)  # pow -> add
        dag.add_edge(3, 4)  # const(1) -> add
        result = evaluate_dag(dag, {0: 2.0})
        # x=2, pow(|2|, 2) = 4, add(4, 1) = 5
        assert result == pytest.approx(5.0)

    def test_sin_x_mul_y_from_fixture(self, sin_x_mul_y_dag: LabeledDAG) -> None:
        """sin(x) * y at x=pi/2, y=3 → 1.0 * 3 = 3.0."""
        result = evaluate_dag(sin_x_mul_y_dag, {0: math.pi / 2, 1: 3.0})
        assert result == pytest.approx(3.0)

    def test_x_plus_y_from_fixture(self, x_plus_y_dag: LabeledDAG) -> None:
        """x + y at x=10, y=20 → 30."""
        result = evaluate_dag(x_plus_y_dag, {0: 10.0, 1: 20.0})
        assert result == pytest.approx(30.0)


# ======================================================================
# S2D + Evaluation integration
# ======================================================================


class TestS2DPlusEvaluation:
    """End-to-end: string -> DAG -> evaluate."""

    def test_sin_x_via_s2d(self) -> None:
        """S2D('Vs', 1) then evaluate at x=pi/2 → 1.0."""
        from isalsr.core.string_to_dag import StringToDAG

        dag = StringToDAG("Vs", num_variables=1).run()
        assert evaluate_dag(dag, {0: math.pi / 2}) == pytest.approx(1.0)

    def test_x_plus_y_via_s2d(self) -> None:
        """S2D for x+y then evaluate at x=3, y=7 → 10."""
        from isalsr.core.string_to_dag import StringToDAG

        dag = StringToDAG("V+nnNc", num_variables=2).run()
        assert evaluate_dag(dag, {0: 3.0, 1: 7.0}) == pytest.approx(10.0)


# ======================================================================
# Error handling
# ======================================================================


class TestEvaluationErrors:
    """Error conditions."""

    def test_missing_variable_raises(self) -> None:
        dag, _ = _make_unary_dag(NodeType.SIN, 0.0)
        with pytest.raises(EvaluationError, match="Missing input"):
            evaluate_dag(dag, {})  # var_index=0 not provided

    def test_no_operations_dag(self) -> None:
        """DAG with only VAR nodes has no output_node → EvaluationError."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        with pytest.raises(EvaluationError):
            evaluate_dag(dag, {0: 1.0})
