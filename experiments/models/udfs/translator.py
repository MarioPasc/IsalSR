"""UDFS result translator.

Converts UDFSRawResult to the unified RunLog and TrajectoryRow schemas.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import sympy

from experiments.models.analyzer.metrics import (
    jaccard_index,
    mse,
    nrmse,
    r_squared,
    solution_recovered,
)
from experiments.models.base_runner import RawRunResult
from experiments.models.base_translator import ResultTranslator
from experiments.models.schemas import (
    BestExpression,
    RegressionResults,
    RunLog,
    RunMetadata,
    SearchSpaceResults,
    TimeResults,
    TrajectoryRow,
)
from experiments.models.udfs.runner import UDFSRawResult

log = logging.getLogger(__name__)


class UDFSTranslator(ResultTranslator):
    """Translates UDFS raw results to unified experiment schema."""

    def __init__(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray,
        ground_truth_expr: Any = None,
        ground_truth_variables: list[Any] | None = None,
    ):
        self._y_train = y_train
        self._y_test = y_test
        self._gt_expr = ground_truth_expr
        self._gt_vars = ground_truth_variables

    def to_run_log(
        self,
        raw: RawRunResult,
        metadata: RunMetadata,
    ) -> RunLog:
        """Convert UDFSRawResult to RunLog."""
        r = raw
        assert isinstance(r, UDFSRawResult)

        # Regression metrics
        r2_train = r_squared(self._y_train, r.y_pred_train)
        r2_test = r_squared(self._y_test, r.y_pred_test)
        nrmse_train = nrmse(self._y_train, r.y_pred_train)
        nrmse_test = nrmse(self._y_test, r.y_pred_test)
        mse_test = mse(self._y_test, r.y_pred_test)

        # Solution recovery
        sol_rec = False
        jac_idx = 0.0
        if r.best_sympy is not None and self._gt_expr is not None:
            sol_rec = solution_recovered(
                r.best_sympy,
                self._gt_expr,
                self._gt_vars,
            )
            jac_idx = jaccard_index(r.best_sympy, self._gt_expr)

        # Model complexity
        complexity = 0
        if r.best_sympy is not None:
            complexity = _count_sympy_nodes(r.best_sympy)

        regression = RegressionResults(
            r2_train=r2_train,
            r2_test=r2_test,
            nrmse_train=nrmse_train,
            nrmse_test=nrmse_test,
            mse_test=mse_test,
            solution_recovered=sol_rec,
            jaccard_index=jac_idx,
            model_complexity=complexity,
        )

        # Time-to-threshold: conservative upper bound from final R²
        time_to_099 = r.wall_clock_s if r2_test >= 0.99 else None
        time_to_0999 = r.wall_clock_s if r2_test >= 0.999 else None

        # Time metrics
        time_results = TimeResults(
            wall_clock_total_s=r.wall_clock_s,
            wall_clock_search_only_s=r.search_only_time_s,
            canonicalization_precomputed_s=0.0,  # no precomputed cache for UDFS
            canonicalization_runtime_s=r.canonicalization_time_s,
            cache_hit_rate=0.0,
            cache_hits=0,
            cache_misses=0,
            estimated_time_saved_s=0.0,
            time_to_r2_099_s=time_to_099,
            time_to_r2_0999_s=time_to_0999,
            evaluation_time_s=r.search_only_time_s,
            overhead_time_s=r.canonicalization_time_s,
        )

        # Search space metrics
        total = max(r.n_total_dags, 1)
        unique = max(r.n_unique_canonical, 1)
        reduction = total / unique if unique > 0 else 1.0
        redundancy = 1.0 - (unique / total) if total > 0 else 0.0

        # Estimate k (max internal nodes) for theoretical bound
        max_k = 0
        if r.best_sympy is not None:
            max_k = complexity - len(r.best_sympy.free_symbols) if complexity > 0 else 0
            max_k = max(max_k, 0)
        theoretical = float(math.factorial(min(max_k, 10))) if max_k > 0 else 1.0

        search_space = SearchSpaceResults(
            total_dags_explored=r.n_total_dags,
            unique_canonical_dags=r.n_unique_canonical,
            empirical_reduction_factor=reduction,
            max_internal_nodes_seen=max_k,
            theoretical_reduction_bound=theoretical,
            redundancy_rate=redundancy,
        )

        # Best expression with IsalSR/canonical strings
        expr_str = str(r.best_sympy) if r.best_sympy is not None else ""
        isalsr_str, canonical_str = _compute_isalsr_strings(r.best_sympy, metadata)

        best_expr = BestExpression(
            symbolic_form=expr_str,
            isalsr_string=isalsr_str,
            canonical_string=canonical_str,
            n_nodes=complexity,
            n_edges=max(complexity - 1, 0),
        )

        return RunLog(
            metadata=metadata,
            regression=regression,
            time=time_results,
            search_space=search_space,
            best_expression=best_expr,
        )

    def to_trajectory(self, raw: RawRunResult) -> list[TrajectoryRow]:
        """Extract trajectory from raw result.

        UDFS doesn't natively produce per-iteration trajectories.
        Returns a single-row trajectory with final state.
        """
        r = raw
        assert isinstance(r, UDFSRawResult)

        r2_test = r_squared(self._y_test, r.y_pred_test)
        nrmse_test = nrmse(self._y_test, r.y_pred_test)
        expr_str = str(r.best_sympy) if r.best_sympy is not None else ""
        complexity = _count_sympy_nodes(r.best_sympy) if r.best_sympy is not None else 0
        cache_rate = 0.0
        if r.n_total_dags > 0 and r.n_skipped > 0:
            cache_rate = r.n_skipped / r.n_total_dags

        return [
            TrajectoryRow(
                timestamp_s=r.wall_clock_s,
                iteration=r.total_evals,
                best_r2=r2_test,
                best_nrmse=nrmse_test,
                n_dags_explored=r.n_total_dags,
                n_unique_canonical=r.n_unique_canonical,
                current_expr=expr_str,
                current_complexity=complexity,
                cache_hit_rate_cumulative=cache_rate,
            ),
        ]

    def best_expression_sympy(self, raw: RawRunResult) -> sympy.Expr | None:
        """Extract best SymPy expression."""
        r = raw
        assert isinstance(r, UDFSRawResult)
        return r.best_sympy


def _compute_isalsr_strings(
    best_sympy: Any,
    metadata: RunMetadata,
) -> tuple[str, str]:
    """Compute IsalSR and canonical strings for the best SymPy expression.

    Only attempted for IsalSR variants. Returns ("", "") on failure or
    for baseline variants.
    """
    if metadata.representation != "isalsr" or best_sympy is None:
        return "", ""

    try:
        from isalsr.adapters.sympy_adapter import SympyAdapter
        from isalsr.core.canonical import pruned_canonical_string
        from isalsr.core.dag_to_string import DAGToString

        adapter = SympyAdapter()
        dag = adapter.from_external(best_sympy)
        converter = DAGToString(dag, initial_node=0)
        isalsr_str = converter.run()
        canonical_str = pruned_canonical_string(dag, timeout=10.0)
        return isalsr_str, canonical_str
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to compute IsalSR strings for best SymPy expr: %s", e)
        return "", ""


def _count_sympy_nodes(expr: Any) -> int:
    """Count nodes in a SymPy expression tree."""
    if expr is None:
        return 0
    try:
        return len(list(sympy.preorder_traversal(expr)))
    except Exception:  # noqa: BLE001
        return 0
