"""Aggregation pipeline for experimental results.

Aggregates across seeds, problems, benchmarks, and methods.
Computes paired statistics with the full test selection logic.

Reference: docs/design/experimental_design/isalsr_experimental_design.md, Section C.
"""

from __future__ import annotations

import logging
import math
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
    CrossProblemDominanceResult,
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


def _nan_if_none(value: bool | float | None) -> float:
    """Map an undetermined metric to NaN so nan-aware statistics exclude it.

    ``solution_recovered`` and ``jaccard_index`` return None when their SymPy
    equivalence check exceeded its budget (see ``metrics.SYMPY_TIMEOUT_S``).
    Every aggregate below uses ``np.nanmean``/``nanstd``/``nanmedian``, so NaN
    is already this module's "excluded" convention -- an undetermined seed
    shrinks N instead of being counted as a failed recovery.
    """
    return float("nan") if value is None else float(value)


# Maps metric names to functions that extract the value from a RunLog.
METRIC_EXTRACTORS: dict[str, Callable[[RunLog], float]] = {
    "r2_test": lambda rl: rl.regression.r2_test,
    "r2_train": lambda rl: rl.regression.r2_train,
    "nrmse_test": lambda rl: rl.regression.nrmse_test,
    "nrmse_train": lambda rl: rl.regression.nrmse_train,
    "mse_test": lambda rl: rl.regression.mse_test,
    "jaccard_index": lambda rl: _nan_if_none(rl.regression.jaccard_index),
    "model_complexity": lambda rl: float(rl.regression.model_complexity),
    "wall_clock_total_s": lambda rl: rl.time.wall_clock_total_s,
    "wall_clock_search_only_s": lambda rl: rl.time.wall_clock_search_only_s,
    "total_dags_explored": lambda rl: float(rl.search_space.total_dags_explored),
    "unique_canonical_dags": lambda rl: float(rl.search_space.unique_canonical_dags),
    "empirical_reduction_factor": lambda rl: rl.search_space.empirical_reduction_factor,
    "redundancy_rate": lambda rl: rl.search_space.redundancy_rate,
    "solution_recovered": lambda rl: _nan_if_none(rl.regression.solution_recovered),
    # --- T19 explored-DAG structural complexity ------------------------- #
    # The primary block only.  These are sampled under a rule that is identical
    # across the three arms of a method, so an arm-versus-arm contrast on them
    # is a contrast on the search and not on the instrument.  The ``unique``
    # block is deliberately absent: it is None on the baseline arm by
    # construction, so a three-arm test on it would silently drop to two arms.
    "complexity_mean_k": lambda rl: _nan_if_none(rl.search_space.complexity_mean_k),
    "complexity_median_k": lambda rl: _nan_if_none(rl.search_space.complexity_median_k),
    "complexity_p90_k": lambda rl: _nan_if_none(rl.search_space.complexity_p90_k),
    "complexity_mean_depth": lambda rl: _nan_if_none(rl.search_space.complexity_mean_depth),
    "complexity_mean_edges": lambda rl: _nan_if_none(rl.search_space.complexity_mean_edges),
    "complexity_mean_n_op": lambda rl: _nan_if_none(rl.search_space.complexity_mean_n_op),
    "complexity_mean_shared": lambda rl: _nan_if_none(rl.search_space.complexity_mean_shared),
    "complexity_mean_nonlinear": lambda rl: _nan_if_none(rl.search_space.complexity_mean_nonlinear),
    "complexity_mean_op_entropy": lambda rl: _nan_if_none(
        rl.search_space.complexity_mean_op_entropy
    ),
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
        AggregateRow with mean, std, median, q25, q75, min, max and ``n``, the
        number of runs aggregated. The summary statistics are NaN-aware, so a
        run whose value is NaN is counted in ``n`` without entering them.
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
        n=len(run_logs),
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
        PairedStats with all metrics, ``n_seeds`` set to the number of matched
        seeds and each metric's ``n`` set to the pairs that survived NaN
        pairwise deletion for that metric.
    """
    # Match seeds by number (robust to 1-3 missing seeds per variant)
    bl_by_seed = {rl.metadata.seed: rl for rl in baseline_logs}
    is_by_seed = {rl.metadata.seed: rl for rl in isalsr_logs}
    common_seeds = sorted(set(bl_by_seed) & set(is_by_seed))

    n_bl_only = len(bl_by_seed) - len(common_seeds)
    n_is_only = len(is_by_seed) - len(common_seeds)
    if n_bl_only > 0 or n_is_only > 0:
        log.warning(
            "  Dropped %d baseline + %d isalsr unmatched seeds (keeping %d paired)",
            n_bl_only,
            n_is_only,
            len(common_seeds),
        )

    if len(common_seeds) < 3:
        raise ValueError(f"Too few paired seeds ({len(common_seeds)}) for statistical testing")

    baseline_logs = [bl_by_seed[s] for s in common_seeds]
    isalsr_logs = [is_by_seed[s] for s in common_seeds]

    rl0 = baseline_logs[0]
    paired = PairedStats(
        method=rl0.metadata.method,
        benchmark=rl0.metadata.benchmark,
        problem=rl0.metadata.problem,
        n_seeds=len(common_seeds),
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
            # The true N for this metric: pairs surviving the NaN pairwise
            # deletion above, which is <= n_seeds.
            n=int(len(diff_clean)),
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

    # Compute solution rates from paired stats (solution_recovered metric)
    sr_metrics = [
        ps.metrics["solution_recovered"]
        for ps in paired_stats_list
        if "solution_recovered" in ps.metrics
    ]
    if sr_metrics:
        sol_baseline = float(np.mean([m.baseline_mean for m in sr_metrics]))
        sol_isalsr = float(np.mean([m.isalsr_mean for m in sr_metrics]))

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


# ======================================================================
# Cross-Problem Dominance Test (CPDT)
# ======================================================================

CPDT_METRIC_ALTERNATIVES: dict[str, str] = {
    "r2_test": "greater",
    "r2_train": "greater",
    "nrmse_test": "less",
    "empirical_reduction_factor": "greater",
    "redundancy_rate": "greater",
    # T19 explored-DAG complexity: two-sided on every contrast. See
    # CPDT_COMPLEXITY_METRICS below for why the pre-registered direction is
    # deliberately not spent here.
    "complexity_mean_k": "two-sided",
    "complexity_mean_depth": "two-sided",
    "complexity_mean_nonlinear": "two-sided",
    "complexity_mean_op_entropy": "two-sided",
    "complexity_mean_shared": "two-sided",
}

# The T19 complexity metrics that enter the CPDT. Deliberately five, not the
# full descriptor set: each additional metric is another family in the
# Holm correction, and these five are the ones the hypothesis is stated
# against -- size, nesting, transcendental content, operator heterogeneity and
# subexpression reuse.
#
# **On the choice of a two-sided alternative.** The hypothesis is directional
# and was pre-registered before the campaign ran (Ezequiel Lopez-Rubio,
# 2026-08-07: the isalsr arm, and to a lesser extent the hash arm, explores
# structurally harder DAGs), so a one-sided test would be admissible and more
# powerful. It is not used, for two reasons. First, unlike R2 -- where
# one-sidedness rests on a structural argument, that removing duplicates cannot
# make the regression worse -- there is no mechanism forbidding the opposite
# outcome here. Second, a reversal would itself be a finding: it would mean
# deduplication steers the search towards SIMPLER structures, which contradicts
# the stated rationale for the representation and must be detectable rather
# than collapsed into p ~ 1. The sign of delta is reported regardless, so the
# pre-registered ordering baseline <= hash <= isalsr remains checkable without
# spending the one-sided licence on it.
CPDT_COMPLEXITY_METRICS: tuple[str, ...] = (
    "complexity_mean_k",
    "complexity_mean_depth",
    "complexity_mean_nonlinear",
    "complexity_mean_op_entropy",
    "complexity_mean_shared",
)

# Contrast policy, decided 2026-08-04 (Mario Pascual Gonzalez).
#
# Campaign C2 has three arms: baseline, hash ("Naive-Hash") and isalsr. The
# alternative hypothesis is not a property of the metric alone -- it depends on
# which pair of arms is being contrasted. Keyed by (arm_a, arm_b) with
# delta = mean(arm_b) - mean(arm_a); the value is the SciPy ``alternative``, or
# ``None`` when the contrast is reported descriptively (no p-value).
#
#   baseline -> isalsr : the submitted primary contrast. Directional for the
#       quality metrics (the claim is one-sided by pre-registration).
#   hash -> isalsr     : two-sided for quality, because neither direction is
#       pre-registered -- the naive hash is a competing representation, not a
#       null. One-sided "greater" for the redundancy metrics: a sound
#       fixed-order hash can only merge a subset of what the canonical string
#       merges, so rho_isalsr >= rho_hash holds by construction of the two maps.
#   baseline -> hash   : two-sided for quality, same reason.
#
# The redundancy metrics against ``baseline`` carry NO p-value: the baseline
# arm never merges, so rho == 1 and the empirical reduction factor == 1 by
# construction. Testing "isalsr merges more than a representation that cannot
# merge" is tautological, and a p-value there would misrepresent a definitional
# identity as an inferential finding.
#: Two-sided on all three contrasts, for the reasons on
#: :data:`CPDT_COMPLEXITY_METRICS`. Unlike the redundancy metrics, complexity
#: against ``baseline`` is **not** definitional: the baseline arm explores real
#: DAGs and its complexity distribution is a genuine measurement, not a
#: constant fixed at 1 by construction. So it carries a p-value.
_COMPLEXITY_POLICY: dict[str, str | None] = dict.fromkeys(CPDT_COMPLEXITY_METRICS, "two-sided")

CPDT_CONTRAST_POLICY: dict[tuple[str, str], dict[str, str | None]] = {
    ("baseline", "isalsr"): {
        "r2_test": "greater",
        "r2_train": "greater",
        "nrmse_test": "less",
        "empirical_reduction_factor": None,
        "redundancy_rate": None,
        **_COMPLEXITY_POLICY,
    },
    ("hash", "isalsr"): {
        "r2_test": "two-sided",
        "r2_train": "two-sided",
        "nrmse_test": "two-sided",
        "empirical_reduction_factor": "greater",
        "redundancy_rate": "greater",
        **_COMPLEXITY_POLICY,
    },
    ("baseline", "hash"): {
        "r2_test": "two-sided",
        "r2_train": "two-sided",
        "nrmse_test": "two-sided",
        "empirical_reduction_factor": None,
        "redundancy_rate": None,
        **_COMPLEXITY_POLICY,
    },
}

# Marker written into ``test_used`` for a contrast that is reported without a
# p-value because the comparison is definitional rather than inferential.
CPDT_DESCRIPTIVE_TEST = "descriptive_definitional_baseline"

# Value written into ``alternative`` for such a contrast.
CPDT_DESCRIPTIVE_ALTERNATIVE = "descriptive"

_CPDT_TIE_THRESHOLD = 1e-6


def resolve_cpdt_alternative(
    metric_name: str,
    arm_a: str = "baseline",
    arm_b: str = "isalsr",
) -> str | None:
    """Resolve the alternative hypothesis for one contrast and metric.

    Args:
        metric_name: Metric key, e.g. ``"r2_test"``.
        arm_a: Reference arm.
        arm_b: Comparison arm; delta = mean(arm_b) - mean(arm_a).

    Returns:
        ``"greater"``, ``"less"``, ``"two-sided"``, or ``None`` when the
        contrast is descriptive. Contrasts absent from the policy table fall
        back to ``CPDT_METRIC_ALTERNATIVES`` for backward compatibility.
    """
    policy = CPDT_CONTRAST_POLICY.get((arm_a, arm_b))
    if policy is None or metric_name not in policy:
        return CPDT_METRIC_ALTERNATIVES.get(metric_name, "greater")
    return policy[metric_name]


def cpdt_primary_p(result: CrossProblemDominanceResult) -> float:
    """Primary p-value of a CPDT result under its own alternative.

    Two-sided contrasts report the two-sided p; directional contrasts report
    the one-sided p. Descriptive contrasts have no primary p and return NaN.

    Args:
        result: A computed CPDT result.

    Returns:
        The p-value the contrast is judged on, or NaN if there is none.
    """
    if result.test_used == CPDT_DESCRIPTIVE_TEST:
        return float("nan")
    if result.alternative == "two-sided":
        return result.p_value_two_sided
    return result.p_value_one_sided


def apply_holm_across_contrasts(
    results: dict[str, dict[str, CrossProblemDominanceResult]],
    alpha: float = 0.05,
) -> None:
    """Holm-adjust CPDT p-values across contrasts, per metric, in place.

    The family is defined per metric as the set of contrasts that actually
    carry a finite p-value (contrast policy, decided 2026-08-04). For the
    quality metrics that is the three pairwise contrasts; for the redundancy
    metrics it is the single ``hash -> isalsr`` contrast, so Holm is the
    identity there. Descriptive entries keep ``p_value_holm = None``: they were
    never tested, so including them in the family would inflate the correction.

    Args:
        results: contrast name -> metric name -> result. Mutated in place.
        alpha: Significance level handed to the Holm procedure.
    """
    metric_names: list[str] = []
    for per_metric in results.values():
        for metric_name in per_metric:
            if metric_name not in metric_names:
                metric_names.append(metric_name)

    for metric_name in metric_names:
        family: list[tuple[str, float]] = []
        for contrast_name, per_metric in results.items():
            result = per_metric.get(metric_name)
            if result is None:
                continue
            p = cpdt_primary_p(result)
            if math.isfinite(p):
                family.append((contrast_name, p))
        if not family:
            continue
        adjusted = holm_bonferroni([p for _, p in family], alpha=alpha)
        for (contrast_name, _), p_adj in zip(family, adjusted, strict=True):
            results[contrast_name][metric_name].p_value_holm = float(p_adj)


def compute_cross_problem_dominance(
    paired_stats_list: list[PairedStats],
    metric_name: str,
    method: str,
    benchmark: str,
    alpha: float = 0.05,
    alternative: str | None = None,
    bootstrap_seed: int = 42,
    arm_a: str = "baseline",
    arm_b: str = "isalsr",
) -> CrossProblemDominanceResult:
    """Cross-problem dominance test: one paired observation per problem.

    For each problem P_i, computes delta_i = mean(arm_b) - mean(arm_a)
    across seeds, then tests H_0: E[delta] <= 0 (or >= 0 for "less") via
    Shapiro-Wilk -> one-sample t-test or Wilcoxon signed-rank. For a two-sided
    contrast the primary p-value is the two-sided one; the one-sided p-value is
    still recorded, taken in the direction of the observed mean delta (it is
    therefore descriptive, not a second test).

    Contrast policy (decided 2026-08-04): when ``alternative`` is not given it
    is resolved from ``CPDT_CONTRAST_POLICY`` for the pair ``(arm_a, arm_b)``. A
    policy entry of ``None`` marks a definitional baseline: the result is still
    produced (W/T/L, effect size, CIs, per-problem deltas) but the p-values are
    NaN and ``test_used`` is ``CPDT_DESCRIPTIVE_TEST``.

    Tie policy:
        A problem with ``|delta_i| <= 1e-6`` is a tie. Ties are snapped to
        exactly zero *before* the test, so the tested vector and the reported
        W/T/L counts are the same object; the raw deltas would otherwise let
        floating-point noise enter the test as signed evidence. Ties are then
        kept in the signed-rank test via ``zero_method="zsplit"``, which ranks
        the zeros and splits their rank sum evenly between the positive and
        negative sums. Dropping them (scipy's default ``"wilcox"``) discards
        the observations that carry the "no difference" evidence and inflates
        significance whenever the non-tied deltas lean one way. ``"zsplit"`` is
        conservative for the one-sided alternative used here. See Pratt (1959),
        JASA 54(287):655-667, on zero handling in the signed-rank test, and
        Demsar (2006), JMLR 7:1-30, on splitting ties evenly in paired
        comparisons of learners.

    Args:
        paired_stats_list: PairedStats for each problem (already computed).
        metric_name: Which metric to test.
        method: Method name (for labeling).
        benchmark: Benchmark name or "all" for pooled.
        alpha: Significance level for normality test.
        alternative: "greater", "less" or "two-sided". If None, resolved from
            the contrast policy for (arm_a, arm_b).
        bootstrap_seed: Seed for bootstrap CI.
        arm_a: Reference arm name.
        arm_b: Comparison arm name; delta = mean(arm_b) - mean(arm_a).

    Returns:
        CrossProblemDominanceResult.
    """
    from scipy import stats as sp_stats

    descriptive = False
    if alternative is None:
        resolved = resolve_cpdt_alternative(metric_name, arm_a, arm_b)
        if resolved is None:
            descriptive = True
            alternative = CPDT_DESCRIPTIVE_ALTERNATIVE
        else:
            alternative = resolved

    names: list[str] = []
    deltas: list[float] = []

    for ps in paired_stats_list:
        m = ps.metrics.get(metric_name)
        if m is None:
            continue
        if not (np.isfinite(m.isalsr_mean) and np.isfinite(m.baseline_mean)):
            continue
        names.append(ps.problem)
        deltas.append(m.isalsr_mean - m.baseline_mean)

    n = len(deltas)
    if n < 3:
        return CrossProblemDominanceResult(
            method=method,
            benchmark=benchmark,
            metric=metric_name,
            alternative=alternative,
            n_problems=n,
            n_wins=0,
            n_ties=n,
            n_losses=0,
            problem_names=names,
            problem_deltas=deltas,
            shapiro_wilk_p=float("nan"),
            normality_assumed=False,
            test_used="insufficient_data",
            statistic=float("nan"),
            p_value_one_sided=float("nan"),
            p_value_two_sided=float("nan"),
            cohens_d=float("nan"),
            cohens_d_ci_lower=float("nan"),
            cohens_d_ci_upper=float("nan"),
            mean_delta=float("nan"),
            mean_delta_ci_lower=float("nan"),
            mean_delta_ci_upper=float("nan"),
            arm_a=arm_a,
            arm_b=arm_b,
        )

    d_arr = np.array(deltas)

    # Snap sub-threshold deltas to exact zero so the tested vector and the
    # reported W/T/L counts are derived from the same numbers.
    d_test = np.where(np.abs(d_arr) <= _CPDT_TIE_THRESHOLD, 0.0, d_arr)

    n_wins = int(np.sum(d_test > 0))
    n_losses = int(np.sum(d_test < 0))
    n_ties = int(np.sum(d_test == 0))

    # Normality test
    _sw_stat, sw_p = shapiro_wilk(d_test)
    normal = sw_p > alpha

    # For a two-sided contrast the one-sided p is reported in the direction of
    # the observed mean delta; it is a description of the sample, not a second
    # pre-registered test, so it is never the value Holm corrects.
    if alternative == "two-sided":
        alt_one = "greater" if float(np.mean(d_test)) >= 0.0 else "less"
    else:
        alt_one = alternative

    # Statistical test
    if descriptive:
        # Definitional baseline: no p-value is meaningful. Everything else
        # (counts, effect size, CIs) is still estimated below.
        stat = float("nan")
        p_one = float("nan")
        p_two = float("nan")
        test_used = CPDT_DESCRIPTIVE_TEST
    elif np.all(d_test == 0):
        stat = 0.0
        p_one = 1.0
        p_two = 1.0
        test_used = "all_zeros"
    elif normal:
        test_used = "t_one_sample"
        res_one = sp_stats.ttest_1samp(d_test, 0.0, alternative=alt_one)
        res_two = sp_stats.ttest_1samp(d_test, 0.0, alternative="two-sided")
        stat = float(res_one.statistic)
        p_one = float(res_one.pvalue)
        p_two = float(res_two.pvalue)
    else:
        test_used = "wilcoxon_signed_rank"
        res_one = sp_stats.wilcoxon(d_test, alternative=alt_one, zero_method="zsplit")
        res_two = sp_stats.wilcoxon(d_test, alternative="two-sided", zero_method="zsplit")
        stat = float(res_one.statistic)
        p_one = float(res_one.pvalue)
        p_two = float(res_two.pvalue)

    # Effect sizes stay on the raw deltas: snapping moves the mean by at most
    # the tie threshold and is a test-decision rule, not an estimation rule.
    d_effect = cohens_d_paired(d_arr)
    d_ci_lo, d_ci_hi = cohens_d_ci_bootstrap(d_arr, seed=bootstrap_seed)
    mean_d, ci_lo, ci_hi = mean_diff_ci(d_arr)

    return CrossProblemDominanceResult(
        method=method,
        benchmark=benchmark,
        metric=metric_name,
        alternative=alternative,
        n_problems=n,
        n_wins=n_wins,
        n_ties=n_ties,
        n_losses=n_losses,
        problem_names=names,
        problem_deltas=deltas,
        shapiro_wilk_p=sw_p,
        normality_assumed=normal,
        test_used=test_used,
        statistic=stat,
        p_value_one_sided=p_one,
        p_value_two_sided=p_two,
        cohens_d=d_effect,
        cohens_d_ci_lower=d_ci_lo,
        cohens_d_ci_upper=d_ci_hi,
        mean_delta=mean_d,
        mean_delta_ci_lower=ci_lo,
        mean_delta_ci_upper=ci_hi,
        arm_a=arm_a,
        arm_b=arm_b,
    )
