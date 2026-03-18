"""Model-agnostic metric computation.

Computes regression performance metrics from predictions and expressions.
Reuses isalsr.evaluation.fitness where possible.

Reference: docs/design/experimental_design/isalsr_experimental_design.md, Section B.3.1.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R².

    R² = 1 - SS_res / SS_tot

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        R² value. Can be negative for very poor fits.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Root Mean Squared Error.

    NRMSE = RMSE / std(y_true)

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        NRMSE value. Lower is better.
    """
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    std = float(np.std(y_true))
    if std == 0:
        return 0.0 if rmse == 0 else float("inf")
    return rmse / std


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def solution_recovered(
    expr_found: Any,
    expr_true: Any,
    variables: list[Any] | None = None,
) -> bool:
    """Check if the found expression exactly matches the ground truth.

    Uses SymPy simplification: simplify(expr_found - expr_true) == 0.

    Args:
        expr_found: Found SymPy expression.
        expr_true: Ground truth SymPy expression.
        variables: Optional list of SymPy symbols.

    Returns:
        True if expressions are symbolically equivalent.
    """
    try:
        import sympy

        diff = sympy.simplify(expr_found - expr_true)
        return diff == 0
    except Exception:  # noqa: BLE001
        log.debug("Solution recovery check failed", exc_info=True)
        return False


def jaccard_index(
    expr_found: Any,
    expr_true: Any,
) -> float:
    """Jaccard index of subexpression overlap.

    J(f_hat, f*) = |S(f_hat) ∩ S(f*)| / |S(f_hat) ∪ S(f*)|

    where S(expr) is the set of all subexpressions.

    Args:
        expr_found: Found SymPy expression.
        expr_true: Ground truth SymPy expression.

    Returns:
        Jaccard index in [0, 1]. Higher is better.
    """
    try:
        subexprs_found = _get_subexpressions(expr_found)
        subexprs_true = _get_subexpressions(expr_true)

        if not subexprs_found and not subexprs_true:
            return 1.0

        intersection = subexprs_found & subexprs_true
        union = subexprs_found | subexprs_true

        if not union:
            return 0.0

        return len(intersection) / len(union)
    except Exception:  # noqa: BLE001
        log.debug("Jaccard computation failed", exc_info=True)
        return 0.0


def _get_subexpressions(expr: Any) -> set[Any]:
    """Extract set of all subexpressions from a SymPy expression."""
    import sympy

    if not isinstance(expr, sympy.Basic):
        return set()

    subexprs = set()
    for sub in sympy.preorder_traversal(expr):
        # Normalize: convert to string for comparison
        subexprs.add(str(sympy.simplify(sub)))
    return subexprs


def model_complexity(n_nodes: int) -> int:
    """Model complexity as number of nodes in expression DAG."""
    return n_nodes
