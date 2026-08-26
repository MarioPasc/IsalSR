"""Integration tests for fitness metrics.

Requires: numpy
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.evaluation.fitness import evaluate_expression, mse, nrmse, r_squared


class TestRSquared:
    """Coefficient of determination."""

    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r_squared(y, y) == pytest.approx(1.0)

    def test_mean_prediction(self) -> None:
        """Predicting the mean gives R^2 = 0."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y, np.mean(y))
        assert r_squared(y, y_pred) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        r2 = r_squared(y, y_pred)
        assert 0.0 < r2 < 1.0

    def test_constant_target(self) -> None:
        """If all targets are identical, R^2 = 1.0 if prediction is exact."""
        y = np.array([5.0, 5.0, 5.0])
        assert r_squared(y, y) == pytest.approx(1.0)


class TestNRMSE:
    """Normalized Root Mean Square Error."""

    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert nrmse(y, y) == pytest.approx(0.0)

    def test_positive_error(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5])
        assert nrmse(y, y_pred) > 0.0


class TestMSE:
    """Mean Squared Error."""

    def test_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert mse(y, y) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        # MSE = (0 + 0 + 1) / 3 = 1/3
        assert mse(y, y_pred) == pytest.approx(1.0 / 3.0)


class TestEvaluateExpression:
    """End-to-end DAG + data → metrics."""

    def test_sin_x_perfect(self) -> None:
        """sin(x) evaluated on x, y=sin(x) → R^2=1.0."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_edge(0, 1)

        x = np.linspace(0, 2 * math.pi, 50).reshape(-1, 1)
        y = np.sin(x.ravel())
        result = evaluate_expression(dag, x, y)
        assert result["r2"] == pytest.approx(1.0, abs=1e-6)
        assert result["nrmse"] == pytest.approx(0.0, abs=1e-6)

    def test_x_plus_y(self) -> None:
        """x + y evaluated on data → correct metrics."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)

        rng = np.random.default_rng(42)
        x = rng.standard_normal((100, 2))
        y = x[:, 0] + x[:, 1]
        result = evaluate_expression(dag, x, y)
        assert result["r2"] == pytest.approx(1.0, abs=1e-6)
