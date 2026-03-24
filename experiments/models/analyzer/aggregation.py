"""Aggregation pipeline for experimental results.

Aggregates across seeds, problems, benchmarks, and methods.
Computes paired statistics with the full test selection logic.

Reference: docs/design/experimental_design/isalsr_experimental_design.md, Section C.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from experiments.models.analyzer.effect_sizes import (
    cohens_d_ci_bootstrap,
    cohens_d_paired,
    mean_diff_ci,
)
from experiments.models.analyzer.statistical_tests import (
    holm_bonferroni,
    paired_ttest,
    shapiro_wilk,
    wilcoxon_signed_rank,
)
from experiments.models.schemas import (
    AggregateRow,
    BenchmarkSummaryRow,
    PairedStats,
    PairedStatsMetric,
    RunLog,
)

log = logging.getLogger(__name__)

# Metrics whose raw values can be NaN / ±inf due to extrapolation failures.
# Following SRBench convention, R²-family metrics are clipped to [0, 1]
# for robust statistics: negative R² is "worse than predicting the mean,"
# and the exact magnitude of failure is uninformative.
_R2_CLIP_METRICS = frozenset({"r2_test", "r2_train"})

# Metrics that can produce NaN / ±inf from evaluation failures.
# We sanitize these with nanmean/nanstd rather than clipping.
_NAN_PRONE_METRICS = frozenset(
    {
        "r2_test",
        "r2_train",
        "nrmse_test",
        "nrmse_train",
        "mse_test",
    }
)


def _sanitize_values(values: np.ndarray, metric_name: str) -> np.ndarray:
    """Sanitize metric values for robust statistics.

    - R² metrics: clip to [0, 1] (SRBench convention).
    - All NaN-prone metrics: replace inf with NaN for nanmean/nanstd.
    """
    out = values.copy()
    if metric_name in _R2_CLIP_METRICS:
        out = np.where(np.isfinite(out), out, np.nan)
        out = np.clip(out, 0.0, 1.0)
    elif metric_name in _NAN_PRONE_METRICS:
        out = np.where(np.isfinite(out), out, np.nan)
    return out


# ======================================================================
# Metric extractors
# ======================================================================

# Maps metric names to functions that extract the value from a RunLog.
METRIC_EXTRACTORS: dict[str, Callable[[RunLog], float]] = {
    "r2_test": lambda rl: rl.regression.r2_test,
    "r2_train": lambda rl: rl.regression.r2_train,
    "nrmse_test": lambda rl: rl.regression.nrmse_test,
    "nrmse_train": lambda rl: rl.regression.nrmse_train,
    "mse_test": lambda rl: rl.regression.mse_test,
    "jaccard_index": lambda rl: rl.regression.jaccard_index,
    "model_complexity": lambda rl: float(rl.regression.model_complexity),
    "wall_clock_total_s": lambda rl: rl.time.wall_clock_total_s,
    "wall_clock_search_only_s": lambda rl: rl.time.wall_clock_search_only_s,
    "total_dags_explored": lambda rl: float(rl.search_space.total_dags_explored),
    "unique_canonical_dags": lambda rl: float(rl.search_space.unique_canonical_dags),
    "empirical_reduction_factor": lambda rl: rl.search_space.empirical_reduction_factor,
    "redundancy_rate": lambda rl: rl.search_space.redundancy_rate,
}


# ======================================================================
# Seed aggregation
# ======================================================================


def aggregate_seeds(
    run_logs: list[RunLog],
    metric_name: str,
    extractor: Callable[[RunLog], float] | None = None,
) -> AggregateRow:
    """Compute summary statistics over seeds for one metric.

    Args:
        run_logs: List of RunLog objects (one per seed).
        metric_name: Name of the metric.
        extractor: Function to extract metric value from RunLog.
            If None, uses METRIC_EXTRACTORS[metric_name].

    Returns:
        AggregateRow with mean, std, median, q25, q75, min, max.
    """
    if extractor is None:
        extractor = METRIC_EXTRACTORS[metric_name]

    raw = np.array([extractor(rl) for rl in run_logs])
    values = _sanitize_values(raw, metric_name)
    rl0 = run_logs[0]

    return AggregateRow(
        method=rl0.metadata.method,
        representation=rl0.metadata.representation,
        benchmark=rl0.metadata.benchmark,
        problem=rl0.metadata.problem,
        metric=metric_name,
        mean=float(np.nanmean(values)),
        std=float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
        median=float(np.nanmedian(values)),
        q25=float(np.nanpercentile(values, 25)),
        q75=float(np.nanpercentile(values, 75)),
        min_val=float(np.nanmin(values)),
        max_val=float(np.nanmax(values)),
    )


def aggregate_all_metrics(
    run_logs: list[RunLog],
) -> list[AggregateRow]:
    """Compute aggregates for all standard metrics."""
    rows = []
    for metric_name in METRIC_EXTRACTORS:
        rows.append(aggregate_seeds(run_logs, metric_name))
    return rows


# ======================================================================
# Paired statistical comparison
# ======================================================================


def compute_paired_stats(
    baseline_logs: list[RunLog],
    isalsr_logs: list[RunLog],
    alpha: float = 0.05,
    bootstrap_seed: int = 42,
) -> PairedStats:
    """Compute full paired statistical comparison for one problem.

    For each metric:
    1. Extract paired values (same seed → same index)
    2. Compute differences d_s = isalsr_s - baseline_s
    3. Shapiro-Wilk normality test
    4. If normal: paired t-test; else: Wilcoxon signed-rank
    5. Cohen's d with bootstrap CI
    6. Mean difference CI

    Args:
        baseline_logs: Baseline RunLogs (sorted by seed).
        isalsr_logs: IsalSR RunLogs (sorted by seed).
        alpha: Significance level.
        bootstrap_seed: Seed for bootstrap CI.

    Returns:
        PairedStats with all metrics.
    """
    assert len(baseline_logs) == len(isalsr_logs), (
        f"Mismatched seed counts: {len(baseline_logs)} vs {len(isalsr_logs)}"
    )

    rl0 = baseline_logs[0]
    paired = PairedStats(
        method=rl0.metadata.method,
        benchmark=rl0.metadata.benchmark,
        problem=rl0.metadata.problem,
    )

    for metric_name, extractor in METRIC_EXTRACTORS.items():
        raw_bl = np.array([extractor(rl) for rl in baseline_logs])
        raw_is = np.array([extractor(rl) for rl in isalsr_logs])

        # Sanitize: clip R² to [0,1], replace inf with NaN
        baseline_vals = _sanitize_values(raw_bl, metric_name)
        isalsr_vals = _sanitize_values(raw_is, metric_name)
        differences = isalsr_vals - baseline_vals

        # Drop NaN pairs for statistical tests
        valid = np.isfinite(differences)
        n_dropped = int((~valid).sum())
        if n_dropped > 0:
            log.info(
                "  %s: dropped %d/%d NaN pairs for statistical test",
                metric_name,
                n_dropped,
                len(differences),
            )
        bl_clean = baseline_vals[valid]
        is_clean = isalsr_vals[valid]
        diff_clean = differences[valid]

        if len(diff_clean) < 3:
            log.warning("  %s: <3 valid pairs, skipping statistical test", metric_name)
            sw_p = float("nan")
            stat = float("nan")
            p_raw = float("nan")
            test_used = "insufficient_data"
            d = float("nan")
            d_ci_lo = d_ci_hi = float("nan")
            mean_d = ci_lo = ci_hi = float("nan")
        else:
            # Normality test
            _sw_stat, sw_p = shapiro_wilk(diff_clean)
            normal = sw_p > alpha

            # Choose test
            if normal:
                stat, p_raw = paired_ttest(bl_clean, is_clean)
                test_used = "paired_t"
            else:
                stat, p_raw = wilcoxon_signed_rank(bl_clean, is_clean)
                test_used = "wilcoxon"

            # Effect size
            d = cohens_d_paired(diff_clean)
            d_ci_lo, d_ci_hi = cohens_d_ci_bootstrap(
                diff_clean,
                seed=bootstrap_seed,
            )
            mean_d, ci_lo, ci_hi = mean_diff_ci(diff_clean)

        paired.metrics[metric_name] = PairedStatsMetric(
            baseline_mean=float(np.nanmean(baseline_vals)),
            baseline_std=float(np.nanstd(baseline_vals, ddof=1)) if len(baseline_vals) > 1 else 0.0,
            isalsr_mean=float(np.nanmean(isalsr_vals)),
            isalsr_std=float(np.nanstd(isalsr_vals, ddof=1)) if len(isalsr_vals) > 1 else 0.0,
            mean_diff=mean_d,
            std_diff=float(np.nanstd(diff_clean, ddof=1)) if len(diff_clean) > 1 else 0.0,
            shapiro_wilk_p=sw_p,
            normality_assumed=sw_p > alpha if np.isfinite(sw_p) else False,
            test_used=test_used,
            statistic=stat,
            p_value_raw=p_raw,
            p_value_holm=None,  # set later by apply_holm_correction
            cohens_d=d,
            cohens_d_ci_lower=d_ci_lo,
            cohens_d_ci_upper=d_ci_hi,
            mean_diff_ci_lower=ci_lo,
            mean_diff_ci_upper=ci_hi,
        )

    return paired


def apply_holm_correction(
    paired_stats_list: list[PairedStats],
    alpha: float = 0.05,
) -> list[PairedStats]:
    """Apply Holm-Bonferroni correction across problems for each metric.

    Modifies p_value_holm in each PairedStatsMetric.

    Args:
        paired_stats_list: PairedStats for each problem.
        alpha: Significance level.

    Returns:
        Same list with p_value_holm updated.
    """
    if not paired_stats_list:
        return paired_stats_list

    # Get all metric names from the first entry
    metric_names = list(paired_stats_list[0].metrics.keys())

    for metric_name in metric_names:
        raw_ps = []
        indices = []
        for i, ps in enumerate(paired_stats_list):
            if metric_name in ps.metrics:
                raw_ps.append(ps.metrics[metric_name].p_value_raw)
                indices.append(i)

        if not raw_ps:
            continue

        adjusted = holm_bonferroni(raw_ps, alpha=alpha)

        for j, idx in enumerate(indices):
            paired_stats_list[idx].metrics[metric_name].p_value_holm = adjusted[j]

    return paired_stats_list


# ======================================================================
# Benchmark-level summary
# ======================================================================


def benchmark_summary(
    paired_stats_list: list[PairedStats],
    metric_name: str,
    alpha: float = 0.05,
) -> BenchmarkSummaryRow:
    """Aggregate paired statistics across problems for one metric.

    Args:
        paired_stats_list: PairedStats for each problem (Holm-corrected).
        metric_name: Which metric to summarize.
        alpha: Significance threshold for counting significant results.

    Returns:
        BenchmarkSummaryRow.
    """
    ps0 = paired_stats_list[0]
    n_problems = len(paired_stats_list)

    ds = []
    n_sig = 0
    speedups = []
    reduction_factors = []
    sol_baseline = 0.0
    sol_isalsr = 0.0

    for ps in paired_stats_list:
        m = ps.metrics.get(metric_name)
        if m is None:
            continue

        ds.append(m.cohens_d)
        p_adj = m.p_value_holm if m.p_value_holm is not None else m.p_value_raw
        if p_adj < alpha:
            n_sig += 1

        # Speedup (baseline_time / isalsr_time) — only for time metrics
        if ("time" in metric_name or "wall_clock" in metric_name) and m.isalsr_mean > 0:
            speedups.append(m.baseline_mean / m.isalsr_mean)

        # Reduction factor
        rf = ps.metrics.get("empirical_reduction_factor")
        if rf is not None:
            reduction_factors.append(rf.isalsr_mean)

    ds_arr = np.array(ds) if ds else np.array([0.0])

    return BenchmarkSummaryRow(
        method=ps0.method,
        benchmark=ps0.benchmark,
        metric=metric_name,
        n_problems=n_problems,
        n_significant=n_sig,
        mean_cohens_d=float(np.nanmean(ds_arr)),
        median_cohens_d=float(np.nanmedian(ds_arr)),
        mean_speedup=float(np.mean(speedups)) if speedups else 0.0,
        mean_reduction_factor=float(np.mean(reduction_factors)) if reduction_factors else 0.0,
        solution_rate_baseline=sol_baseline,
        solution_rate_isalsr=sol_isalsr,
    )
