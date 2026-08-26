"""Random search for symbolic regression using IsalSR strings.

Generates random IsalSR strings, canonicalizes each one (MANDATORY per advisor),
evaluates fitness, and returns the best results.

Dependencies: numpy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from isalsr.core.canonical import CanonicalTimeoutError, pruned_canonical_string
from isalsr.core.node_types import OperationSet
from isalsr.core.string_to_dag import StringToDAG
from isalsr.evaluation.fitness import evaluate_expression
from isalsr.search.operators import detokenize, random_token

# Default timeout (seconds) per canonicalization call.
_CANON_TIMEOUT: float = 5.0

log = logging.getLogger(__name__)


def random_isalsr_string(
    num_variables: int,
    max_tokens: int,
    allowed_ops: OperationSet,
    rng: np.random.Generator,
) -> str:
    """Generate a random valid IsalSR instruction string.

    Args:
        num_variables: Number of input variables (m).
        max_tokens: Maximum number of tokens to generate.
        allowed_ops: Allowed operation types.
        rng: NumPy random generator for reproducibility.

    Returns:
        A random IsalSR instruction string.
    """
    n_tokens = int(rng.integers(1, max_tokens + 1))
    tokens = [random_token(allowed_ops, rng) for _ in range(n_tokens)]
    return detokenize(tokens)


def random_search(
    x_data: np.ndarray[Any, np.dtype[Any]],
    y_true: np.ndarray[Any, np.dtype[Any]],
    num_variables: int,
    allowed_ops: OperationSet,
    n_iterations: int = 1000,
    max_tokens: int = 50,
    seed: int = 42,
    use_canonical: bool = True,
) -> list[dict[str, object]]:
    """Run random search in IsalSR string space.

    When ``use_canonical=True`` (default), every generated string is
    canonicalized (pruned 6-tuple variant) before evaluation, eliminating
    O(k!) duplicate representations. When ``use_canonical=False``, strings
    are evaluated as-is (baseline for WITH vs WITHOUT comparison).

    Args:
        x_data: Input matrix (N, m).
        y_true: Target vector (N,).
        num_variables: Number of input variables.
        allowed_ops: Allowed operations.
        n_iterations: Number of random strings to evaluate.
        max_tokens: Maximum tokens per string.
        seed: Random seed for reproducibility.
        use_canonical: If True, canonicalize before evaluation (default).
            If False, skip canonicalization (baseline comparison).

    Returns:
        Sorted list of dicts with keys: 'string', 'canonical', 'r2', 'nrmse', 'mse'.
    """
    rng = np.random.default_rng(seed)
    results: list[dict[str, object]] = []
    seen: set[str] = set()
    n_timeouts = 0

    for _ in range(n_iterations):
        raw = random_isalsr_string(num_variables, max_tokens, allowed_ops, rng)
        try:
            dag = StringToDAG(raw, num_variables, allowed_ops).run()
            if dag.node_count <= num_variables:
                continue  # VAR-only, skip.

            if use_canonical:
                canon = pruned_canonical_string(dag, timeout=_CANON_TIMEOUT)
                if canon in seen:
                    continue  # O(k!) deduplication.
                seen.add(canon)
                dag_eval = StringToDAG(canon, num_variables, allowed_ops).run()
            else:
                canon = raw  # No canonicalization (baseline).
                dag_eval = dag

            metrics = evaluate_expression(dag_eval, x_data, y_true)
            results.append(
                {
                    "string": raw,
                    "canonical": canon,
                    "r2": metrics["r2"],
                    "nrmse": metrics["nrmse"],
                    "mse": metrics["mse"],
                }
            )
        except CanonicalTimeoutError:
            n_timeouts += 1
            continue  # Skip DAGs that are too expensive to canonicalize.
        except Exception:  # noqa: BLE001
            continue  # Invalid string, skip.

    if n_timeouts > 0:
        log.warning(
            "Skipped %d/%d strings due to canonicalization timeout", n_timeouts, n_iterations
        )

    results.sort(key=lambda d: -float(d.get("r2", -1e10)))  # type: ignore[arg-type]
    return results
