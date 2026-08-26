"""BFGS optimization of CONST node values.

After the expression structure (DAG topology) is determined, optimizes the
scalar values of all CONST nodes to minimize NRMSE on training data.

Mathematical reference:
    Liu2025 (GraphDSR): Two-phase approach — discrete structure search
    followed by continuous constant optimization via BFGS.
    Petersen et al. (2021, DSR): Risk-seeking policy gradient + BFGS.

Dependencies: numpy, scipy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.optimize import minimize

from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.evaluation.fitness import nrmse

log = logging.getLogger(__name__)


def optimize_constants(
    dag: LabeledDAG,
    x_data: np.ndarray[Any, np.dtype[Any]],
    y_true: np.ndarray[Any, np.dtype[Any]],
    method: str = "L-BFGS-B",
    max_iter: int = 100,
) -> LabeledDAG:
    """Optimize CONST node values via scipy.optimize.minimize.

    Args:
        dag: Expression DAG (not modified).
        x_data: Input matrix (N, m).
        y_true: Target vector (N,).
        method: Optimization method (default L-BFGS-B).
        max_iter: Maximum iterations.

    Returns:
        A NEW LabeledDAG with optimized constant values.
        The input DAG is never modified.
    """
    # Find CONST nodes.
    const_ids: list[int] = [i for i in range(dag.node_count) if dag.node_label(i) == NodeType.CONST]

    if not const_ids:
        return dag  # No constants to optimize.

    # Extract initial values.
    x0 = np.array([float(dag.node_data(c).get("const_value", 1.0)) for c in const_ids])

    n_samples = x_data.shape[0]
    n_vars = x_data.shape[1] if x_data.ndim > 1 else 1

    def objective(constants: np.ndarray[Any, np.dtype[Any]]) -> float:
        """Evaluate NRMSE with given constant values."""
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            inputs: dict[int, float] = {}
            for j in range(n_vars):
                inputs[j] = float(x_data[i, j]) if x_data.ndim > 1 else float(x_data[i])
            # Temporarily set constants in the DAG.
            for idx, c_id in enumerate(const_ids):
                dag.set_const_value(c_id, float(constants[idx]))
            try:
                y_pred[i] = evaluate_dag(dag, inputs)
            except Exception:  # noqa: BLE001
                y_pred[i] = 0.0
        return nrmse(y_true, y_pred)

    # Optimize.
    try:
        result = minimize(
            objective,
            x0,
            method=method,
            options={"maxiter": max_iter},
        )
        best_constants = result.x
    except Exception:  # noqa: BLE001
        log.warning("Constant optimization failed; using initial values.")
        best_constants = x0

    # Restore original constants in input DAG.
    for idx, c_id in enumerate(const_ids):
        dag.set_const_value(c_id, float(x0[idx]))

    # Build a new DAG with optimized constants.
    new_dag = _copy_dag_with_constants(dag, const_ids, best_constants)
    return new_dag


def _copy_dag_with_constants(
    dag: LabeledDAG,
    const_ids: list[int],
    new_values: np.ndarray[Any, np.dtype[Any]],
) -> LabeledDAG:
    """Create a copy of the DAG with updated CONST values."""
    new_dag = LabeledDAG(dag.max_nodes)

    # Copy all nodes.
    const_value_map = {c_id: float(new_values[idx]) for idx, c_id in enumerate(const_ids)}
    for i in range(dag.node_count):
        label = dag.node_label(i)
        data = dag.node_data(i)
        var_idx = data.get("var_index")
        const_val = data.get("const_value")

        if i in const_value_map:
            const_val = const_value_map[i]

        new_dag.add_node(
            label,
            var_index=int(var_idx) if var_idx is not None else None,
            const_value=float(const_val) if const_val is not None else None,
        )

    # Copy all edges.
    for i in range(dag.node_count):
        for j in dag.out_neighbors(i):
            new_dag.add_edge(i, j)

    return new_dag
