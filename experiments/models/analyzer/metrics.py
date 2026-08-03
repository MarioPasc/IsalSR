"""Model-agnostic metric computation.

Computes regression performance metrics from predictions and expressions.
Reuses isalsr.evaluation.fitness where possible.

Reference: docs/design/experimental_design/isalsr_experimental_design.md, Section B.3.1.
"""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Wall-clock bound on any single SymPy equivalence/normalisation call.
#
# Rationale: ``sympy.simplify`` has no complexity guarantee and is
# superpolynomial on the large rational expressions a stack=32 GP host
# produces. On 2026-08-02 an unbounded call hung a Picasso probe cell for
# >20 min *after* its search had finished, past the wall limit, destroying the
# whole run -- ``run_log.json`` is written only after this step, so the run
# left no artefact at all. Because which individual a run finishes with depends
# on the search trajectory, and the dedup arms change that trajectory, such
# losses are not arm-neutral: unbounded runtime here is a bias risk in the
# paired design, not merely an availability risk.
#
# The failure is stochastic, not per-cell. Bingo stops on wall clock, so the
# generation reached -- and hence the final individual -- varies with node speed
# and load. Re-running the hung cell at the same seed/problem/config on the same
# (unbounded) code completed in 24:49 with a ~20 s tail: a different individual
# was drawn. So the pathology cannot be predicted, reproduced on demand, or
# designed around; it can only be bounded.
#
# Budget choice, from the 2026-08-02 probe: all 27 completing cells had metric
# tails of at most ~20 s, while the single pathological case exceeded 1,233 s
# without finishing. 300 s sits an order of magnitude above every case that was
# ever observed to complete and well below the one that did not, so it bounds
# the tail without changing any value that was previously computable. It is
# ~0.7 % of a 12 h production run.
SYMPY_TIMEOUT_S = 300.0


class _SymPyTimeoutError(Exception):
    """Raised when a bounded SymPy call exceeds its wall-clock budget."""


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    """Bound a block by wall-clock time using SIGALRM.

    SymPy is pure Python, so the signal is delivered between bytecodes and
    interrupts promptly. The bound is a no-op when disabled (``seconds <= 0``)
    or when not on the main thread, where ``signal.signal`` is unavailable;
    in that case the previous unbounded behaviour applies unchanged.

    Args:
        seconds: Wall-clock budget. Non-positive disables the bound.

    Yields:
        None.

    Raises:
        _SymPyTimeoutError: If the block is still running when the budget expires.
    """
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(_signum: int, _frame: Any) -> None:
        raise _SymPyTimeoutError

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def count_nonfinite_predictions(y_pred: np.ndarray) -> int:
    """Number of predictions that are NaN or +/-inf.

    A recovered expression can be well-defined on the training domain and
    undefined on part of the test domain: ``log`` of a negative argument on an
    extrapolation grid, or ``exp`` overflow where a divisor approaches zero.
    Host protected operators guard division by zero but not these.

    This is a *measured property of the run*, recorded in the RunLog so the
    failure is countable rather than inferred from a NaN metric.
    """
    return int(np.count_nonzero(~np.isfinite(np.asarray(y_pred, dtype=float))))


def _predictions_usable(y_pred: np.ndarray) -> bool:
    """Whether every prediction is finite.

    Scoring policy (T08, reviewer comment R2.7): a model that cannot produce a
    number for every point of the evaluation set is **unusable on that set**,
    and is scored as such -- ``R^2 = 0``, ``NRMSE = 1``. It is not scored on the
    subset where it happens to be defined, which would reward a model for being
    undefined exactly where it fails and would make the denominator
    problem-dependent.

    Scoring the failure rather than emitting NaN is the conservative choice: it
    counts against the arm that produced it. In the submitted campaign both
    affected cells were IsalSR runs, so this policy argues against our own
    hypothesis, and no verdict changes under it.
    """
    return bool(np.all(np.isfinite(np.asarray(y_pred, dtype=float))))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R².

    R² = 1 - SS_res / SS_tot

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        R² value. Can be negative for very poor fits. Returns 0.0 -- an
        unusable model, no better than predicting the mean -- if any
        prediction is non-finite. Never returns NaN.
    """
    if not _predictions_usable(y_pred):
        log.warning(
            "%d/%d non-finite predictions; scoring R2 = 0.0 (model unusable on this set)",
            count_nonfinite_predictions(y_pred),
            len(np.asarray(y_pred)),
        )
        return 0.0
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
        NRMSE value. Lower is better. Returns 1.0 -- the NRMSE of predicting
        the mean, the counterpart of ``R^2 = 0`` -- if any prediction is
        non-finite. Never returns NaN.
    """
    if not _predictions_usable(y_pred):
        return 1.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    std = float(np.std(y_true))
    if std == 0:
        return 0.0 if rmse == 0 else float("inf")
    return rmse / std


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error.

    Returns the variance of ``y_true`` -- the MSE of predicting the mean, the
    counterpart of ``R^2 = 0`` -- if any prediction is non-finite.
    """
    if not _predictions_usable(y_pred):
        return float(np.var(y_true))
    return float(np.mean((y_true - y_pred) ** 2))


def _strip_symbol_assumptions(expr: Any) -> Any:
    """Replace all symbols with bare versions (no assumptions).

    UDFS creates symbols with ``real=True, positive=True`` while ground truth
    uses bare symbols. SymPy treats these as distinct, so
    ``simplify(x_real - x_bare)`` does not reduce to zero. Stripping
    assumptions before comparison fixes this.
    """
    import sympy

    subs = {}
    for sym in expr.free_symbols:
        if sym.assumptions0:  # has non-default assumptions
            subs[sym] = sympy.Symbol(sym.name)
    return expr.subs(subs) if subs else expr


def solution_recovered(
    expr_found: Any,
    expr_true: Any,
    variables: list[Any] | None = None,
    timeout_s: float = SYMPY_TIMEOUT_S,
) -> bool | None:
    """Check if the found expression exactly matches the ground truth.

    Uses SymPy simplification: simplify(expr_found - expr_true) == 0.
    Symbol assumptions (real, positive, etc.) are stripped before comparison
    to avoid false negatives from assumption mismatches between SR methods.

    Three outcomes are distinguished. ``True``/``False`` mean the equivalence
    was decided; ``None`` means it could not be decided within ``timeout_s``
    and the observation must be *excluded* from the recovery rate rather than
    counted as a failure. Recording a timeout as ``False`` would be a false
    negative that lands preferentially on whichever arm finds larger
    expressions, biasing the paired comparison.

    Args:
        expr_found: Found SymPy expression.
        expr_true: Ground truth SymPy expression.
        variables: Optional list of SymPy symbols.
        timeout_s: Wall-clock budget. Non-positive disables the bound.

    Returns:
        True if symbolically equivalent, False if proved non-equivalent or the
        comparison failed, None if undetermined within the budget.
    """
    try:
        import sympy

        with _time_limit(timeout_s):
            found = _strip_symbol_assumptions(expr_found)
            true = _strip_symbol_assumptions(expr_true)
            diff = sympy.simplify(found - true)
            return bool(diff == 0)
    except _SymPyTimeoutError:
        log.warning(
            "Solution recovery undetermined: SymPy exceeded %.0f s; "
            "recording None (excluded), not False",
            timeout_s,
        )
        return None
    except Exception:  # noqa: BLE001
        log.debug("Solution recovery check failed", exc_info=True)
        return False


def jaccard_index(
    expr_found: Any,
    expr_true: Any,
    timeout_s: float = SYMPY_TIMEOUT_S,
) -> float | None:
    """Jaccard index of subexpression overlap.

    J(f_hat, f*) = |S(f_hat) ∩ S(f*)| / |S(f_hat) ∪ S(f*)|

    where S(expr) is the set of all subexpressions.

    Args:
        expr_found: Found SymPy expression.
        expr_true: Ground truth SymPy expression.
        timeout_s: Wall-clock budget. Non-positive disables the bound.

    Returns:
        Jaccard index in [0, 1], higher is better; None if the normalisation
        did not finish within the budget (excluded, not scored as 0.0).
    """
    try:
        with _time_limit(timeout_s):
            subexprs_found = _get_subexpressions(expr_found)
            subexprs_true = _get_subexpressions(expr_true)

            if not subexprs_found and not subexprs_true:
                return 1.0

            intersection = subexprs_found & subexprs_true
            union = subexprs_found | subexprs_true

            if not union:
                return 0.0

            return len(intersection) / len(union)
    except _SymPyTimeoutError:
        log.warning(
            "Jaccard undetermined: SymPy exceeded %.0f s; recording None (excluded), not 0.0",
            timeout_s,
        )
        return None
    except Exception:  # noqa: BLE001
        log.debug("Jaccard computation failed", exc_info=True)
        return 0.0


def _get_subexpressions(expr: Any) -> set[Any]:
    """Extract set of all subexpressions from a SymPy expression.

    The per-node ``simplify`` dominates this function's cost and looks like an
    obvious thing to drop -- SymPy already canonicalises ``x + x`` to ``2*x``
    on construction, so most nodes are unaffected. It is nevertheless
    **load-bearing**: replacing it with ``str``/``srepr`` changes the reported
    Jaccard index on the Pythagorean identity (0.1429 -> 0.0) and on
    cancelling rationals (0.2222 -> 0.1); ``cancel`` fixes the rational case
    but not the trigonometric ones. Measured 2026-08-02 and pinned in
    ``tests/unit/test_analyzer_metrics.py::test_simplify_is_load_bearing``.

    Bound the runtime via ``jaccard_index(timeout_s=...)`` instead of making
    the normalisation cheaper -- Jaccard is a published number.
    """
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
