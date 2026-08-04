"""Bingo result translator.

Converts BingoRawResult to the unified RunLog and TrajectoryRow schemas.
"""

from __future__ import annotations

import contextlib
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import sympy

from experiments.models.analyzer.metrics import (
    count_nonfinite_predictions,
    jaccard_index,
    mse,
    nrmse,
    r_squared,
    solution_recovered,
)
from experiments.models.base_runner import RawRunResult
from experiments.models.base_translator import ResultTranslator
from experiments.models.bingo.runner import BingoRawResult, get_symbolic_form
from experiments.models.schemas import (
    BestExpression,
    RegressionResults,
    RunLog,
    RunMetadata,
    SearchSpaceResults,
    TimeResults,
    TrajectoryRow,
)

log = logging.getLogger(__name__)


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
        n_nonfinite_test = count_nonfinite_predictions(r.y_pred_test)

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
            n_nonfinite_test_predictions=n_nonfinite_test,
        )

        # Time-to-threshold: conservative upper bound from final R²
        time_to_099 = r.wall_clock_s if r2_test >= 0.99 else None
        time_to_0999 = r.wall_clock_s if r2_test >= 0.999 else None

        # Atlas / cache metrics
        total_lookups = r.atlas_hits + r.atlas_misses
        hit_rate = r.atlas_hits / total_lookups if total_lookups > 0 else 0.0
        avg_canon = r.canon_fallback_time_s / max(r.atlas_misses, 1) if r.atlas_misses > 0 else 0.0
        estimated_saved = r.atlas_hits * avg_canon

        # Cost attribution.  Search time is the wall clock minus every block the
        # dedup wrapper ran inside the same budget; overhead is the
        # representation layer's own cost, which is canonicalisation plus the
        # adapter conversion that produces the object it canonicalises.  The
        # shadow sketches are audit instrumentation, not method cost: they are
        # removed from search time but deliberately left out of the overhead and
        # reported separately, so both figures are clean.
        search_only = max(
            0.0,
            r.wall_clock_s - r.canonicalization_time_s - r.conversion_time_s - r.shadow_time_s,
        )
        overhead = r.canonicalization_time_s + r.conversion_time_s

        time_results = TimeResults(
            wall_clock_total_s=r.wall_clock_s,
            wall_clock_search_only_s=search_only,
            canonicalization_precomputed_s=r.atlas_lookup_time_s,
            canonicalization_runtime_s=r.canonicalization_time_s,
            cache_hit_rate=hit_rate,
            cache_hits=r.atlas_hits,
            cache_misses=r.atlas_misses,
            estimated_time_saved_s=estimated_saved,
            time_to_r2_099_s=time_to_099,
            time_to_r2_0999_s=time_to_0999,
            evaluation_time_s=search_only,
            overhead_time_s=overhead,
            conversion_time_s=r.conversion_time_s,
            shadow_time_s=r.shadow_time_s,
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
            penalised_in_population_mean=r.penalised_in_population_mean,
            penalised_in_population_max=r.penalised_in_population_max,
        )

        # Best expression: symbolic form + IsalSR/canonical strings
        sym_form = get_symbolic_form(r.best_agraph, r.best_sympy)
        isalsr_str, canonical_str = _compute_isalsr_strings(r.best_agraph, metadata)

        best_expr = BestExpression(
            symbolic_form=sym_form,
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

        Intermediate rows use training R² (converted from MSE fitness).
        The final row uses actual test R² for consistency with run_log.json.
        """
        r = raw
        assert isinstance(r, BingoRawResult)

        rows: list[TrajectoryRow] = []
        var_y = float(np.var(self._y_train))
        if var_y <= 0:
            var_y = 1.0

        # Intermediate snapshots (training R²)
        for snap in r.trajectory_snapshots:
            r2_train = 1.0 - snap.best_fitness / var_y if np.isfinite(snap.best_fitness) else 0.0
            dedup_rate = snap.n_skipped / snap.n_total_dags if snap.n_total_dags > 0 else 0.0
            # `n_dags_explored` must come from the SAME counter as the final row,
            # or the series is not a series.  Two different counters exist:
            #
            #   snap.n_total_dags -- candidate DAGs admitted to the dedup hook.
            #       This is rho's numerator and what the final row reports.  It
            #       is 0 on the baseline arm, which has no dedup hook at all.
            #   snap.n_evals      -- ExplicitRegression.eval_count, i.e. fitness
            #       FUNCTION INVOCATIONS.  Inflated 3.3-4.1x on the dedup arms by
            #       ScipyOptimizer/LocalOptFitnessFunction inner iterations during
            #       LM constant optimisation, so it counts a different population.
            #
            # Using n_evals on a dedup arm made the trajectory climb to ~110k and
            # then DROP to ~30k on the final row -- the same quantity measured two
            # ways.  Each arm now uses whichever counter its own final row uses:
            # the DAG counter where it exists, eval_count on the baseline (where
            # runner.py sets n_total_dags = total_evals = eval_count anyway).
            # rho itself was never affected: translator.py builds it from
            # dedup.n_total / dedup.n_unique, and no analyzer or figure code reads
            # this column.
            n_explored = snap.n_total_dags if snap.n_total_dags > 0 else snap.n_evals
            rows.append(
                TrajectoryRow(
                    timestamp_s=snap.timestamp_s,
                    iteration=snap.generation,
                    best_r2=r2_train,
                    best_nrmse=0.0,
                    n_dags_explored=n_explored,
                    n_unique_canonical=snap.n_unique_canonical,
                    current_expr="",
                    current_complexity=0,
                    cache_hit_rate_cumulative=dedup_rate,
                )
            )

        # Final row -- TRAIN metrics, like every row above it.
        #
        # This column used to switch to r2_test on the last row only, which made
        # `best_r2` two different quantities in one series and produced a
        # decrease wherever test R2 < train R2 -- 459 of Stage C's 1,260 cells,
        # 100 % of them at the final row and nowhere else (C1.10, 2026-08-04).
        # It is the same defect class as the n_dags_explored mix-up fixed above:
        # intermediate rows measuring one population, the final row another.
        #
        # Test metrics are NOT lost -- they are the authoritative copy in
        # run_log.json's `results.regression` (r2_test / nrmse_test), which is
        # what every analyzer and figure reads.  No consumer reads this column:
        # the convergence scripts read `best_r2_train` from the .npz, and
        # time_to_r2_099_s / time_to_r2_0999_s are computed in this file from
        # the snapshots.  So no reported number changes.
        r2_train_final = r_squared(self._y_train, r.y_pred_train)
        nrmse_train_final = nrmse(self._y_train, r.y_pred_train)
        sym_form = get_symbolic_form(r.best_agraph, r.best_sympy)
        complexity = 0
        if r.best_agraph is not None:
            with contextlib.suppress(Exception):
                complexity = r.best_agraph.get_complexity()
        cache_rate = r.n_skipped / r.n_total_dags if r.n_total_dags > 0 else 0.0

        rows.append(
            TrajectoryRow(
                timestamp_s=r.wall_clock_s,
                iteration=r.n_generations,
                best_r2=r2_train_final,
                best_nrmse=nrmse_train_final,
                n_dags_explored=r.n_total_dags,
                n_unique_canonical=r.n_unique_canonical,
                current_expr=sym_form,
                current_complexity=complexity,
                cache_hit_rate_cumulative=cache_rate,
            ),
        )

        return rows

    def save_convergence_log(self, raw: RawRunResult, path: Path) -> None:
        """Save dense per-generation population fitness as compressed .npz.

        Parameters
        ----------
        raw : RawRunResult
            Must be a BingoRawResult with convergence_data populated.
        path : Path
            Output file path (should end in .npz).

        Notes
        -----
        File contains:
            generations     : int32   (n_gens,)
            timestamps_s    : float64 (n_gens,)
            n_evals         : int32   (n_gens,)
            best_r2_train   : float64 (n_gens,)
            population_r2   : float64 (n_gens, pop_size)
            var_y_train     : float64 scalar

        R² = 1 - MSE / Var(y_train). Individuals with MSE=inf get R²=-inf.
        """
        r = raw
        assert isinstance(r, BingoRawResult)

        if not r.convergence_data:
            return

        var_y = float(np.var(self._y_train))
        if var_y <= 0:
            var_y = 1.0

        n_gens = len(r.convergence_data)
        pop_size = len(r.convergence_data[0][3])

        generations = np.empty(n_gens, dtype=np.int32)
        timestamps_s = np.empty(n_gens, dtype=np.float64)
        n_evals_arr = np.empty(n_gens, dtype=np.int32)
        best_r2 = np.empty(n_gens, dtype=np.float64)
        pop_r2 = np.empty((n_gens, pop_size), dtype=np.float64)

        for i, (gen, ts, ne, fitness_arr) in enumerate(r.convergence_data):
            generations[i] = gen
            timestamps_s[i] = ts
            n_evals_arr[i] = ne
            r2_arr = np.where(
                np.isfinite(fitness_arr),
                1.0 - fitness_arr / var_y,
                -np.inf,
            )
            pop_r2[i] = r2_arr
            finite_mask = np.isfinite(r2_arr)
            best_r2[i] = np.max(r2_arr[finite_mask]) if finite_mask.any() else -np.inf

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            generations=generations,
            timestamps_s=timestamps_s,
            n_evals=n_evals_arr,
            best_r2_train=best_r2,
            population_r2=pop_r2,
            var_y_train=np.float64(var_y),
        )

    def best_expression_sympy(self, raw: RawRunResult) -> sympy.Expr | None:
        r = raw
        assert isinstance(r, BingoRawResult)
        return r.best_sympy


def _compute_isalsr_strings(
    agraph: Any,
    metadata: RunMetadata,
) -> tuple[str, str]:
    """Compute IsalSR and canonical strings for the best AGraph.

    Only attempted for IsalSR variants. Returns ("", "") on failure or
    for baseline variants.
    """
    if metadata.representation != "isalsr" or agraph is None:
        return "", ""

    try:
        from experiments.models.bingo.adapter import agraph_to_labeled_dag
        from isalsr.core.canonical import pruned_canonical_string
        from isalsr.core.dag_to_string import DAGToString

        dag = agraph_to_labeled_dag(agraph)
        converter = DAGToString(dag, initial_node=0)
        isalsr_str = converter.run()
        canonical_str = pruned_canonical_string(dag, timeout=10.0)
        return isalsr_str, canonical_str
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to compute IsalSR strings for best AGraph: %s", e)
        return "", ""


def _count_sympy_nodes(expr: Any) -> int:
    if expr is None:
        return 0
    try:
        return len(list(sympy.preorder_traversal(expr)))
    except Exception:  # noqa: BLE001
        return 0
