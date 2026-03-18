"""Bingo result translator.

Converts BingoRawResult to the unified RunLog and TrajectoryRow schemas.
"""

from __future__ import annotations

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
from experiments.models.bingo.runner import BingoRawResult
from experiments.models.schemas import (
    BestExpression,
    RegressionResults,
    RunLog,
    RunMetadata,
    SearchSpaceResults,
    TimeResults,
    TrajectoryRow,
)


class BingoTranslator(ResultTranslator):
    """Translates Bingo raw results to unified experiment schema."""

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
        r = raw
        assert isinstance(r, BingoRawResult)

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
            sol_rec = solution_recovered(r.best_sympy, self._gt_expr, self._gt_vars)
            jac_idx = jaccard_index(r.best_sympy, self._gt_expr)

        # Model complexity (from AGraph)
        complexity = 0
        if r.best_agraph is not None:
            try:
                complexity = r.best_agraph.get_complexity()
            except Exception:  # noqa: BLE001
                complexity = _count_sympy_nodes(r.best_sympy) if r.best_sympy else 0

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

        time_results = TimeResults(
            wall_clock_total_s=r.wall_clock_s,
            wall_clock_search_only_s=r.search_only_time_s,
            canonicalization_precomputed_s=0.0,
            canonicalization_runtime_s=r.canonicalization_time_s,
            cache_hit_rate=0.0,
            cache_hits=0,
            cache_misses=0,
            estimated_time_saved_s=0.0,
            time_to_r2_099_s=None,
            time_to_r2_0999_s=None,
            evaluation_time_s=r.search_only_time_s,
            overhead_time_s=r.canonicalization_time_s,
        )

        total = max(r.n_total_dags, 1)
        unique = max(r.n_unique_canonical, 1)
        reduction = total / unique if unique > 0 else 1.0
        redundancy = 1.0 - (unique / total) if total > 0 else 0.0

        max_k = max(complexity - 1, 0)  # approximate internal nodes
        theoretical = float(math.factorial(min(max_k, 10))) if max_k > 0 else 1.0

        search_space = SearchSpaceResults(
            total_dags_explored=r.n_total_dags,
            unique_canonical_dags=r.n_unique_canonical,
            empirical_reduction_factor=reduction,
            max_internal_nodes_seen=max_k,
            theoretical_reduction_bound=theoretical,
            redundancy_rate=redundancy,
        )

        expr_str = str(r.best_sympy) if r.best_sympy is not None else ""
        best_expr = BestExpression(
            symbolic_form=expr_str,
            isalsr_string="",
            canonical_string="",
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
        r = raw
        assert isinstance(r, BingoRawResult)

        r2_test = r_squared(self._y_test, r.y_pred_test)
        nrmse_test = nrmse(self._y_test, r.y_pred_test)
        expr_str = str(r.best_sympy) if r.best_sympy is not None else ""
        complexity = 0
        if r.best_agraph is not None:
            try:
                complexity = r.best_agraph.get_complexity()
            except Exception:  # noqa: BLE001
                pass
        cache_rate = r.n_skipped / r.n_total_dags if r.n_total_dags > 0 else 0.0

        return [
            TrajectoryRow(
                timestamp_s=r.wall_clock_s,
                iteration=r.n_generations,
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
        r = raw
        assert isinstance(r, BingoRawResult)
        return r.best_sympy


def _count_sympy_nodes(expr: Any) -> int:
    if expr is None:
        return 0
    try:
        return len(list(sympy.preorder_traversal(expr)))
    except Exception:  # noqa: BLE001
        return 0
