"""Integration tests for BFGS constant optimization.

Requires: numpy, scipy
"""

from __future__ import annotations

import numpy as np
import pytest

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.evaluation.constant_optimizer import optimize_constants


class TestConstantOptimizer:
    """BFGS constant value optimization."""

    def test_linear_cx_plus_d(self) -> None:
        """Expression c1*x + c2 on data y = 2x + 3.

        After optimization, c1 ≈ 2, c2 ≈ 3.
        """
        # DAG: x(0), c1(1), MUL(2), c2(3), ADD(4)
        # Edges: x->MUL, c1->MUL, MUL->ADD, c2->ADD
        dag = LabeledDAG(max_nodes=6)
        dag.add_node(NodeType.VAR, var_index=0)  # 0: x
        dag.add_node(NodeType.CONST, const_value=1.0)  # 1: c1
        dag.add_node(NodeType.MUL)  # 2
        dag.add_node(NodeType.CONST, const_value=1.0)  # 3: c2
        dag.add_node(NodeType.ADD)  # 4
        dag.add_edge(0, 2)  # x -> MUL
        dag.add_edge(1, 2)  # c1 -> MUL
        dag.add_edge(2, 4)  # MUL -> ADD
        dag.add_edge(3, 4)  # c2 -> ADD

        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, (100, 1))
        y = 2.0 * x.ravel() + 3.0

        optimized = optimize_constants(dag, x, y, max_iter=200)

        # Check that the optimized DAG is different from input.
        c1_opt = float(optimized.node_data(1).get("const_value", 0.0))
        c2_opt = float(optimized.node_data(3).get("const_value", 0.0))
        assert c1_opt == pytest.approx(2.0, abs=0.5)
        assert c2_opt == pytest.approx(3.0, abs=0.5)

    def test_no_constants_returns_same(self) -> None:
        """DAG with no CONST nodes returns the same object."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_edge(0, 1)

        x = np.array([[1.0], [2.0], [3.0]])
        y = np.sin(x.ravel())

        result = optimize_constants(dag, x, y)
        assert result is dag  # Same object, not a copy.

    def test_original_dag_unchanged(self) -> None:
        """Input DAG's CONST values are not modified."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.CONST, const_value=1.0)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)

        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([11.0, 12.0, 13.0])

        optimize_constants(dag, x, y)
        # Original must be unchanged.
        assert float(dag.node_data(1).get("const_value", 0.0)) == pytest.approx(1.0)
