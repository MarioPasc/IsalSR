"""Integration tests for SymPy adapter.

Requires: sympy >= 1.12
"""

from __future__ import annotations

import math

sympy = __import__("pytest").importorskip("sympy")

from sympy import Symbol, cos, sin

from isalsr.adapters.sympy_adapter import SympyAdapter
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

x_sym = Symbol("x_0")
y_sym = Symbol("x_1")


class TestDAGToSymPy:
    """LabeledDAG → SymPy expression."""

    def test_sin_x(self, sin_x_dag: LabeledDAG) -> None:
        adapter = SympyAdapter()
        expr = adapter.to_sympy(sin_x_dag)
        assert expr == sin(x_sym)

    def test_x_plus_y(self, x_plus_y_dag: LabeledDAG) -> None:
        adapter = SympyAdapter()
        expr = adapter.to_sympy(x_plus_y_dag)
        assert sympy.simplify(expr - (x_sym + y_sym)) == 0

    def test_const_value(self) -> None:
        """DAG for x + 3.14 → x_0 + 3.14."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.CONST, const_value=3.14)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)
        adapter = SympyAdapter()
        expr = adapter.to_sympy(dag)
        val = float(expr.subs(x_sym, 1.0))
        assert val == __import__("pytest").approx(4.14)


class TestSymPyToDAG:
    """SymPy expression → LabeledDAG."""

    def test_sin_x_roundtrip(self) -> None:
        adapter = SympyAdapter()
        expr = sin(x_sym)
        dag = adapter.from_sympy(expr, [x_sym])
        expr2 = adapter.to_sympy(dag)
        assert sympy.simplify(expr - expr2) == 0

    def test_x_plus_y_roundtrip(self) -> None:
        adapter = SympyAdapter()
        expr = x_sym + y_sym
        dag = adapter.from_sympy(expr, [x_sym, y_sym])
        expr2 = adapter.to_sympy(dag)
        assert sympy.simplify(expr - expr2) == 0

    def test_cos_x(self) -> None:
        adapter = SympyAdapter()
        expr = cos(x_sym)
        dag = adapter.from_sympy(expr, [x_sym])
        assert dag.node_count >= 2
        expr2 = adapter.to_sympy(dag)
        assert sympy.simplify(expr - expr2) == 0


class TestSymPyNumericalConsistency:
    """DAG evaluation matches SymPy evaluation."""

    def test_sin_pi_half(self, sin_x_dag: LabeledDAG) -> None:
        from isalsr.core.dag_evaluator import evaluate_dag

        adapter = SympyAdapter()
        expr = adapter.to_sympy(sin_x_dag)
        dag_val = evaluate_dag(sin_x_dag, {0: math.pi / 2})
        sympy_val = float(expr.subs(x_sym, math.pi / 2))
        assert dag_val == __import__("pytest").approx(sympy_val)
