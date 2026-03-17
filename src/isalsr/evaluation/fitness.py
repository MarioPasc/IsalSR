"""Fitness metrics for symbolic regression.

Primary metrics following Liu2025 (GraphDSR, Neural Networks 187:107405):
R^2 (coefficient of determination) and NRMSE (normalized root mean square error).

Dependencies: numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.labeled_dag import LabeledDAG


def r_squared(
    y_true: np.ndarray[Any, np.dtype[Any]],
    y_pred: np.ndarray[Any, np.dtype[Any]],
) -> float:
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot.

    Returns 0.0 if SS_tot is zero (constant target).
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-15:
        return 0.0 if ss_res > 1e-15 else 1.0
    return 1.0 - ss_res / ss_tot


def nrmse(
    y_true: np.ndarray[Any, np.dtype[Any]],
    y_pred: np.ndarray[Any, np.dtype[Any]],
) -> float:
    """Normalized Root Mean Square Error = RMSE / std(y_true).

    Returns 0.0 if std(y_true) is zero (constant target) and predictions match.
    """
    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    std_val = float(np.std(y_true))
    if std_val < 1e-15:
        return 0.0 if rmse_val < 1e-15 else float("inf")
    return rmse_val / std_val


def mse(
    y_true: np.ndarray[Any, np.dtype[Any]],
    y_pred: np.ndarray[Any, np.dtype[Any]],
) -> float:
    """Mean Squared Error = mean((y_true - y_pred)^2)."""
    return float(np.mean((y_true - y_pred) ** 2))


def evaluate_expression(
    dag: LabeledDAG,
    x_data: np.ndarray[Any, np.dtype[Any]],
    y_true: np.ndarray[Any, np.dtype[Any]],
) -> dict[str, float]:
    """Evaluate a DAG expression on data and return fitness metrics.

    Args:
        dag: The expression DAG to evaluate.
        x_data: Input matrix of shape (N, m) where N = samples, m = variables.
        y_true: Target vector of shape (N,).

    Returns:
        Dict with keys 'r2', 'nrmse', 'mse'.
    """
    n_samples = x_data.shape[0]
    n_vars = x_data.shape[1] if x_data.ndim > 1 else 1

    y_pred = np.zeros(n_samples)
    for i in range(n_samples):
        inputs: dict[int, float] = {}
        for j in range(n_vars):
            inputs[j] = float(x_data[i, j]) if x_data.ndim > 1 else float(x_data[i])
        try:
            y_pred[i] = evaluate_dag(dag, inputs)
        except Exception:  # noqa: BLE001
            y_pred[i] = 0.0

    return {
        "r2": r_squared(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "mse": mse(y_true, y_pred),
    }
