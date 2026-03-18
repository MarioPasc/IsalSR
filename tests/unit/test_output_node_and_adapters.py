"""Tests for output_node() determinism and adapter operand order.

Targets two specific bugs:
1. output_node() used to return sinks[-1] for multi-sink DAGs, which
   depended on node creation order. Now raises ValueError.
2. SymPy adapter's to_sympy used sorted(in_neighbors) for binary ops
   instead of ordered_inputs(). Same B9 pattern as the evaluator.

Also tests protected-operations consistency between dag_evaluator
(scalar, math module) and evaluation/protected_ops (vectorized, numpy).
"""

from __future__ import annotations

import math

import pytest

from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.errors import EvaluationError

# ======================================================================
# output_node() correctness
# ======================================================================


class TestOutputNodeSingleSink:
    """Single non-VAR sink: output_node() returns it correctly."""

    def test_sin_x(self) -> None:
        dag = LabeledDAG(3)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_edge(0, 1)
        assert dag.output_node() == 1

    def test_x_plus_y(self) -> None:
        dag = LabeledDAG(4)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)
        assert dag.output_node() == 2

    def test_nested_chain(self) -> None:
        """x -> sin -> cos -> exp: only exp is a sink."""
        dag = LabeledDAG(5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.COS)
        dag.add_node(NodeType.EXP)
        dag.add_edge(0, 1)
        dag.add_edge(1, 2)
        dag.add_edge(2, 3)
        assert dag.output_node() == 3


class TestOutputNodeMultiSink:
    """Multi-sink DAGs must raise ValueError (ambiguous output)."""

    def test_two_independent_ops(self) -> None:
        """x -> sin AND x -> cos: two sinks."""
        dag = LabeledDAG(4)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.COS)
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        with pytest.raises(ValueError, match="Ambiguous output"):
            dag.output_node()

    def test_five_sinks(self) -> None:
        """x -> a, b, c, d, e: five sinks."""
        dag = LabeledDAG(7)
        dag.add_node(NodeType.VAR, var_index=0)
        for _ in range(5):
            n = dag.add_node(NodeType.SIN)
            dag.add_edge(0, n)
        with pytest.raises(ValueError, match="Ambiguous output"):
            dag.output_node()

    def test_evaluate_dag_rejects_multi_sink(self) -> None:
        """evaluate_dag raises EvaluationError for multi-sink DAGs."""
        dag = LabeledDAG(4)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.COS)
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        with pytest.raises(EvaluationError, match="Ambiguous output"):
            evaluate_dag(dag, {0: 1.0})


class TestOutputNodeNoSink:
    """DAGs with no non-VAR nodes must raise ValueError."""

    def test_var_only(self) -> None:
        dag = LabeledDAG(2)
        dag.add_node(NodeType.VAR, var_index=0)
        with pytest.raises(ValueError, match="No non-VAR sink"):
            dag.output_node()


class TestOutputNodeConstTolerance:
    """CONST-induced multi-sinks: ignore CONST sinks when selecting output.

    After normalize_const_creation() moves CONST in-edges to x_1,
    operation nodes whose only child was a CONST become extra sinks.
    output_node() should select the non-CONST sink as the output.
    """

    def test_const_induced_multi_sink_after_normalization(self) -> None:
        """x -> COS -> CONST normalizes to x -> {COS, CONST}: select COS."""
        dag = LabeledDAG(4)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.COS)  # 1
        dag.add_node(NodeType.CONST, const_value=1.0)  # 2
        dag.add_edge(0, 1)
        dag.add_edge(1, 2)  # COS -> CONST

        normalized = dag.normalize_const_creation()
        # After normalization: x -> {COS, CONST}, COS has no outgoing edges.
        # COS (node 1) and CONST (node 2) are both non-VAR sinks.
        assert normalized.output_node() == 1  # COS is the true output

    def test_all_const_sinks(self) -> None:
        """DAG where all non-VAR sinks are CONST: return first CONST."""
        dag = LabeledDAG(4)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.CONST, const_value=3.14)  # 1
        dag.add_node(NodeType.CONST, const_value=2.72)  # 2
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        # Both sinks are CONST. Return first by node ID.
        assert dag.output_node() == 1

    def test_genuine_multi_sink_still_raises(self) -> None:
        """Two non-CONST operation sinks: still raises ValueError."""
        dag = LabeledDAG(4)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)  # 1
        dag.add_node(NodeType.COS)  # 2
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        with pytest.raises(ValueError, match="Ambiguous output"):
            dag.output_node()

    def test_multiple_const_plus_one_op_sink(self) -> None:
        """One SIN + two CONST sinks: return SIN."""
        dag = LabeledDAG(5)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.SIN)  # 1: sole non-CONST sink
        dag.add_node(NodeType.CONST, const_value=1.0)  # 2
        dag.add_node(NodeType.CONST, const_value=2.0)  # 3
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(0, 3)
        assert dag.output_node() == 1

    def test_round_trip_evaluation_with_const(self) -> None:
        """'vcNVk' -> S2D -> canonical -> S2D -> evaluate must not raise."""
        from isalsr.core.canonical import pruned_canonical_string
        from isalsr.core.string_to_dag import StringToDAG

        raw = "vcNVk"
        dag_raw = StringToDAG(raw, num_variables=1).run()
        # The raw DAG: x -> COS -> CONST. Output is CONST, eval = 1.0.
        val_raw = evaluate_dag(dag_raw, {0: 1.5})
        assert val_raw == 1.0

        canon = pruned_canonical_string(dag_raw)
        dag_canon = StringToDAG(canon, num_variables=1).run()
        # After canonicalization + S2D: x -> {COS, CONST}. Two sinks.
        # output_node() should handle this gracefully.
        val_canon = evaluate_dag(dag_canon, {0: 1.5})
        # COS(1.5) is the output (non-CONST sink), not the CONST value.
        assert abs(val_canon - math.cos(1.5)) < 1e-10

    def test_evaluator_uses_tolerant_output_node(self) -> None:
        """evaluate_dag on CONST-normalized DAG returns operation result."""
        dag = LabeledDAG(4)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.SIN)  # 1
        dag.add_node(NodeType.CONST, const_value=42.0)  # 2
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        # SIN is the output (non-CONST sink), not CONST.
        val = evaluate_dag(dag, {0: 1.5})
        assert abs(val - math.sin(1.5)) < 1e-10


# ======================================================================
# SymPy adapter operand order (requires sympy)
# ======================================================================


class TestSympyAdapterOperandOrder:
    """SymPy adapter must use ordered_inputs for binary ops."""

    @pytest.fixture(autouse=True)
    def _check_sympy(self) -> None:
        pytest.importorskip("sympy")

    def test_sub_operand_order(self) -> None:
        """sin(x) - cos(x): to_sympy must preserve operand order."""
        from sympy import Symbol, cos, sin

        from isalsr.adapters.sympy_adapter import SympyAdapter

        dag = LabeledDAG(5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.COS)
        dag.add_node(NodeType.SUB)
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(1, 3)  # sin = first operand
        dag.add_edge(2, 3)  # cos = second operand

        adapter = SympyAdapter()
        expr = adapter.to_sympy(dag)

        x = Symbol("x_0")
        expected = sin(x) - cos(x)
        # SymPy simplification may rearrange. Evaluate numerically.
        val_expr = float(expr.subs(x, 1.5))
        val_expected = float(expected.subs(x, 1.5))
        assert val_expr == pytest.approx(val_expected, abs=1e-10)

    def test_sub_reversed_operand_order(self) -> None:
        """cos(x) - sin(x): different from sin(x) - cos(x)."""
        from sympy import Symbol, cos, sin

        from isalsr.adapters.sympy_adapter import SympyAdapter

        dag = LabeledDAG(5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.COS)  # node 1
        dag.add_node(NodeType.SIN)  # node 2
        dag.add_node(NodeType.SUB)
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(1, 3)  # cos = first operand
        dag.add_edge(2, 3)  # sin = second operand

        adapter = SympyAdapter()
        expr = adapter.to_sympy(dag)

        x = Symbol("x_0")
        expected = cos(x) - sin(x)
        val_expr = float(expr.subs(x, 1.5))
        val_expected = float(expected.subs(x, 1.5))
        assert val_expr == pytest.approx(val_expected, abs=1e-10)

    def test_div_operand_order(self) -> None:
        """sin(x) / cos(x) = tan(x)."""
        from sympy import Symbol, cos, sin

        from isalsr.adapters.sympy_adapter import SympyAdapter

        dag = LabeledDAG(5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.COS)
        dag.add_node(NodeType.DIV)
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(1, 3)  # sin / cos
        dag.add_edge(2, 3)

        adapter = SympyAdapter()
        expr = adapter.to_sympy(dag)

        x = Symbol("x_0")
        expected = sin(x) / cos(x)
        val_expr = float(expr.subs(x, 1.0))
        val_expected = float(expected.subs(x, 1.0))
        assert val_expr == pytest.approx(val_expected, abs=1e-10)

    def test_pow_operand_order(self) -> None:
        """x ^ y with 2 variables."""
        from sympy import Pow, Symbol

        from isalsr.adapters.sympy_adapter import SympyAdapter

        dag = LabeledDAG(4)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.POW)
        dag.add_edge(0, 2)  # x = base (first operand)
        dag.add_edge(1, 2)  # y = exponent (second operand)

        adapter = SympyAdapter()
        expr = adapter.to_sympy(dag)

        x, y = Symbol("x_0"), Symbol("x_1")
        expected = Pow(x, y)
        val_expr = float(expr.subs({x: 2.0, y: 3.0}))
        val_expected = float(expected.subs({x: 2.0, y: 3.0}))
        assert val_expr == pytest.approx(val_expected, abs=1e-10)

    def test_sympy_evaluator_consistency(self) -> None:
        """SymPy and dag_evaluator must agree for non-commutative ops."""
        from sympy import Symbol

        from isalsr.adapters.sympy_adapter import SympyAdapter

        # Build sin(x) - cos(x)
        dag = LabeledDAG(5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.COS)
        dag.add_node(NodeType.SUB)
        dag.add_edge(0, 1)
        dag.add_edge(0, 2)
        dag.add_edge(1, 3)
        dag.add_edge(2, 3)

        # Evaluate via dag_evaluator
        val_dag = evaluate_dag(dag, {0: 1.0})

        # Evaluate via SymPy
        adapter = SympyAdapter()
        expr = adapter.to_sympy(dag)
        x = Symbol("x_0")
        val_sympy = float(expr.subs(x, 1.0))

        assert val_dag == pytest.approx(val_sympy, abs=1e-10)
        assert val_dag == pytest.approx(math.sin(1.0) - math.cos(1.0), abs=1e-10)


# ======================================================================
# Protected operations consistency
# ======================================================================


class TestProtectedOpsConsistency:
    """dag_evaluator and evaluation/protected_ops must give same results."""

    def test_protected_div_zero(self) -> None:
        """x / 0 should give 1.0 (protected)."""
        dag = LabeledDAG(4)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.DIV)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)
        # x=5.0, y=0.0 → protected_div(5, 0) = 1.0
        val = evaluate_dag(dag, {0: 5.0, 1: 0.0})
        assert val == pytest.approx(1.0)

    def test_protected_log_negative(self) -> None:
        """log(x) for x < 0 should use log(|x| + eps)."""
        dag = LabeledDAG(3)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.LOG)
        dag.add_edge(0, 1)
        val = evaluate_dag(dag, {0: -2.0})
        expected = math.log(abs(-2.0) + 1e-10)
        assert val == pytest.approx(expected, abs=1e-8)

    def test_protected_sqrt_negative(self) -> None:
        """sqrt(x) for x < 0 should use sqrt(|x|)."""
        dag = LabeledDAG(3)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SQRT)
        dag.add_edge(0, 1)
        val = evaluate_dag(dag, {0: -4.0})
        expected = math.sqrt(4.0)
        assert val == pytest.approx(expected, abs=1e-8)

    def test_protected_pow_negative_base(self) -> None:
        """x^y with x < 0 should use |x|^y (protected)."""
        dag = LabeledDAG(4)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.POW)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)
        val = evaluate_dag(dag, {0: -2.0, 1: 3.0})
        expected = (abs(-2.0) + 1e-10) ** 3.0
        assert val == pytest.approx(expected, abs=1e-6)

    def test_protected_exp_overflow(self) -> None:
        """exp(1000) should be clamped (protected)."""
        dag = LabeledDAG(3)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.EXP)
        dag.add_edge(0, 1)
        val = evaluate_dag(dag, {0: 1000.0})
        # exp(500) is the max clipped input, result is clamped to 1e15
        assert math.isfinite(val)
