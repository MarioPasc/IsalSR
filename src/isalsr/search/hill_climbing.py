"""Hill climbing search for symbolic regression using IsalSR strings.

Multi-restart hill climbing: start from random string, apply mutations,
canonicalize after each (MANDATORY per advisor), keep improvements.

Dependencies: numpy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from isalsr.core.canonical import canonical_string
from isalsr.core.node_types import OperationSet
from isalsr.core.string_to_dag import StringToDAG
from isalsr.evaluation.fitness import evaluate_expression
from isalsr.search.operators import insertion_mutation, point_mutation
from isalsr.search.random_search import random_isalsr_string

log = logging.getLogger(__name__)


def hill_climbing(
    x_data: np.ndarray[Any, np.dtype[Any]],
    y_true: np.ndarray[Any, np.dtype[Any]],
    num_variables: int,
    allowed_ops: OperationSet,
    n_iterations: int = 1000,
    max_tokens: int = 50,
    n_restarts: int = 10,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Multi-restart hill climbing in the canonical IsalSR space.

    Each restart: random init -> canonical -> iterate mutations, keeping
    improvements. Returns best results across all restarts.

    Args:
        x_data: Input matrix (N, m).
        y_true: Target vector (N,).
        num_variables: Number of input variables.
        allowed_ops: Allowed operations.
        n_iterations: Mutations per restart.
        max_tokens: Maximum tokens per string.
        n_restarts: Number of restarts.
        seed: Random seed.

    Returns:
        Sorted list of dicts with keys: 'string', 'r2', 'nrmse', 'mse'.
    """
    rng = np.random.default_rng(seed)
    results: list[dict[str, object]] = []

    for _restart in range(n_restarts):
        # Initialize with random canonical string.
        current = _init_canonical(num_variables, max_tokens, allowed_ops, rng)
        if current is None:
            continue
        current_metrics = _eval_string(current, num_variables, allowed_ops, x_data, y_true)
        if current_metrics is None:
            continue

        for _step in range(n_iterations):
            # Apply random mutation.
            mutation_fn = rng.choice([point_mutation, insertion_mutation])  # type: ignore[arg-type]
            try:
                mutated = mutation_fn(current, allowed_ops, rng)
                dag = StringToDAG(mutated, num_variables, allowed_ops).run()
                if dag.node_count <= num_variables:
                    continue
                canon = canonical_string(dag)  # MANDATORY canonicalization
                metrics = _eval_string(canon, num_variables, allowed_ops, x_data, y_true)
                if metrics is not None and metrics["r2"] > current_metrics["r2"]:
                    current = canon
                    current_metrics = metrics
            except Exception:  # noqa: BLE001
                continue

        results.append({"string": current, **current_metrics})

    results.sort(key=lambda d: -float(d.get("r2", -1e10)))  # type: ignore[arg-type]
    return results


def _init_canonical(
    num_variables: int,
    max_tokens: int,
    allowed_ops: OperationSet,
    rng: np.random.Generator,
) -> str | None:
    """Generate a random canonical string, retrying on failure."""
    for _ in range(100):
        raw = random_isalsr_string(num_variables, max_tokens, allowed_ops, rng)
        try:
            dag = StringToDAG(raw, num_variables, allowed_ops).run()
            if dag.node_count <= num_variables:
                continue
            return canonical_string(dag)
        except Exception:  # noqa: BLE001
            continue
    return None


def _eval_string(
    string: str,
    num_variables: int,
    allowed_ops: OperationSet,
    x_data: np.ndarray[Any, np.dtype[Any]],
    y_true: np.ndarray[Any, np.dtype[Any]],
) -> dict[str, float] | None:
    """Evaluate a canonical string and return metrics, or None on failure."""
    try:
        dag = StringToDAG(string, num_variables, allowed_ops).run()
        return evaluate_expression(dag, x_data, y_true)
    except Exception:  # noqa: BLE001
        return None
