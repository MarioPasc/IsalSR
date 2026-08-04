"""End-to-end analysis for IsalSR model experiments.

Reads all results from the output directory, computes:
- Per-problem paired stats (loads existing or recomputes)
- Benchmark summaries (aggregated across problems)
- Cross-method Friedman/Nemenyi (requires >= 2 methods)
- Reduction factor comparison across methods
- Global summary JSON

Usage:
    python -m experiments.models.analyze \
        --results-dir /path/to/results \
        --methods udfs,bingo \
        --benchmarks nguyen,feynman \
        --variants baseline,hash,isalsr
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import logging
import math  # noqa: E402 -- used in _safe_stats
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.models.analyzer.aggregation import (  # noqa: E402
    METRIC_EXTRACTORS,
    aggregate_all_metrics,
    apply_holm_correction,
    benchmark_summary,
    compute_paired_stats,
)
from experiments.models.analyzer.cross_method import (  # noqa: E402
    compare_reduction_factors,
    cross_method_friedman,
)
from experiments.models.io_utils import (  # noqa: E402
    load_all_run_logs,
    load_paired_stats,
    save_aggregate,
    save_paired_stats,
)
from experiments.models.schemas import BENCHMARK_SUMMARY_COLUMNS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# Arms analysed when ``--variants`` is not given. The two-arm default keeps
# every C1-era invocation byte-identical to its pre-three-arm behaviour.
DEFAULT_VARIANTS: tuple[str, ...] = ("baseline", "isalsr")

# Arm that never deduplicates, and therefore acts as the reference in every
# contrast and in the overhead accounting.
REFERENCE_VARIANT = "baseline"


def resolve_variants(variants: Sequence[str] | None) -> list[str]:
    """Normalise a caller-supplied arm list.

    Args:
        variants: Requested arms, or None for the two-arm default.

    Returns:
        The arms to analyse, order preserved, duplicates removed.
    """
    requested = list(DEFAULT_VARIANTS) if variants is None else list(variants)
    seen: set[str] = set()
    ordered: list[str] = []
    for variant in requested:
        if variant not in seen:
            seen.add(variant)
            ordered.append(variant)
    return ordered


# ======================================================================
# Loading helpers
# ======================================================================


def load_all_paired_stats(
    results_dir: Path,
    method: str,
    benchmark: str,
) -> list[Any]:
    """Load all paired_stats.json files for a (method, benchmark) pair."""
    bench_dir = results_dir / method / benchmark
    if not bench_dir.exists():
        log.warning("Missing directory: %s", bench_dir)
        return []

    stats = []
    for problem_dir in sorted(bench_dir.iterdir()):
        if not problem_dir.is_dir():
            continue
        ps_path = problem_dir / "paired_stats.json"
        if ps_path.exists():
            stats.append(load_paired_stats(ps_path))
    return stats


# Contrast name -> (arm_a, arm_b, per-problem file). Mirrors the triples
# written by the orchestrator for the three-arm C2 campaign; arm_a lands in
# the PairedStats "baseline" slot and arm_b in the "isalsr" slot, so
# delta = slot_isalsr - slot_baseline is already mean(arm_b) - mean(arm_a).
CPDT_CONTRASTS: tuple[tuple[str, str, str, str], ...] = (
    ("isalsr_vs_baseline", "baseline", "isalsr", "paired_stats.json"),
    ("isalsr_vs_hash", "hash", "isalsr", "paired_stats_isalsr_vs_hash.json"),
    ("hash_vs_baseline", "baseline", "hash", "paired_stats_hash_vs_baseline.json"),
)

CPDT_PRIMARY_CONTRAST = "isalsr_vs_baseline"

CPDT_CONTRAST_ARMS: dict[str, tuple[str, str]] = {
    name: (arm_a, arm_b) for name, arm_a, arm_b, _ in CPDT_CONTRASTS
}


def load_secondary_contrast_stats(
    results_dir: Path,
    method: str,
    benchmark: str,
) -> dict[str, list[Any]]:
    """Load the non-primary contrast paired stats for a (method, benchmark).

    A two-arm results root (no hash arm) yields an empty dict, which restores
    the pre-2026-08-04 single-contrast behaviour exactly.

    Args:
        results_dir: Root results directory.
        method: Method name.
        benchmark: Benchmark name.

    Returns:
        Contrast name -> list of PairedStats, one per problem. Contrasts with
        no files present are omitted.
    """
    bench_dir = results_dir / method / benchmark
    out: dict[str, list[Any]] = {}
    if not bench_dir.exists():
        return out

    for contrast_name, _arm_a, _arm_b, filename in CPDT_CONTRASTS:
        if contrast_name == CPDT_PRIMARY_CONTRAST:
            continue
        stats = []
        for problem_dir in sorted(bench_dir.iterdir()):
            if not problem_dir.is_dir():
                continue
            ps_path = problem_dir / filename
            if ps_path.exists():
                stats.append(load_paired_stats(ps_path))
        if stats:
            out[contrast_name] = stats
        else:
            log.info(
                "  No %s files under %s; skipping contrast %s",
                filename,
                bench_dir,
                contrast_name,
            )
    return out


def _recompute_contrast_paired_stats(
    bench_dir: Path,
    arm_a: str,
    arm_b: str,
    filename: str,
) -> list[Any]:
    """Load or recompute one contrast's per-problem paired stats.

    ``arm_a`` lands in the PairedStats "baseline" slot and ``arm_b`` in the
    "isalsr" slot, so the stored delta is already mean(arm_b) - mean(arm_a).

    Args:
        bench_dir: ``results_dir / method / benchmark``.
        arm_a: Reference arm directory name.
        arm_b: Comparison arm directory name.
        filename: Per-problem file holding this contrast's paired stats.

    Returns:
        One PairedStats per problem for which the contrast could be formed.
    """
    all_stats: list[Any] = []
    for problem_dir in sorted(bench_dir.iterdir()):
        if not problem_dir.is_dir():
            continue

        ps_path = problem_dir / filename
        if ps_path.exists():
            all_stats.append(load_paired_stats(ps_path))
            continue

        arm_a_dir = problem_dir / arm_a
        arm_b_dir = problem_dir / arm_b
        if not arm_a_dir.exists() or not arm_b_dir.exists():
            continue

        arm_a_logs = load_all_run_logs(arm_a_dir)
        arm_b_logs = load_all_run_logs(arm_b_dir)
        if arm_a_logs and arm_b_logs:
            try:
                log.info("  Recomputing %s for %s", filename, problem_dir.name)
                paired = compute_paired_stats(arm_a_logs, arm_b_logs)
                save_paired_stats(paired, ps_path)
                all_stats.append(paired)
            except ValueError as e:
                log.warning("  Skipping %s: %s", problem_dir.name, e)

    # Holm correction runs across problems within one contrast: the family is
    # the set of problems tested for that contrast, not the union over arms.
    if all_stats:
        apply_holm_correction(all_stats)
        for ps in all_stats:
            problem_slug = ps.problem.lower().replace("-", "_")
            save_paired_stats(ps, bench_dir / problem_slug / filename)

    return all_stats


def recompute_paired_stats_if_needed(
    results_dir: Path,
    method: str,
    benchmark: str,
    variants: Sequence[str] | None = None,
) -> list[Any]:
    """Load or recompute paired stats for all problems in a benchmark.

    If a contrast's per-problem file doesn't exist but both of its arms have
    run_logs, computes paired stats on the fly. Contrasts whose arms are not
    both in ``variants`` are not touched, so a two-arm invocation only ever
    writes ``paired_stats.json``.

    Args:
        results_dir: Root results directory.
        method: Method name.
        benchmark: Benchmark name.
        variants: Arms present in this campaign. Defaults to
            ``("baseline", "isalsr")``.

    Returns:
        The primary contrast's PairedStats, one per problem.
    """
    bench_dir = results_dir / method / benchmark
    if not bench_dir.exists():
        return []

    variant_list = resolve_variants(variants)
    primary: list[Any] = []
    for contrast_name, arm_a, arm_b, filename in CPDT_CONTRASTS:
        if arm_a not in variant_list or arm_b not in variant_list:
            continue
        stats = _recompute_contrast_paired_stats(bench_dir, arm_a, arm_b, filename)
        if contrast_name == CPDT_PRIMARY_CONTRAST:
            primary = stats

    return primary


def recompute_aggregates_if_needed(
    results_dir: Path,
    method: str,
    benchmark: str,
    variants: Sequence[str] | None = None,
) -> None:
    """Ensure aggregate.csv exists for all requested variants in all problems.

    Args:
        results_dir: Root results directory.
        method: Method name.
        benchmark: Benchmark name.
        variants: Arms to aggregate. Defaults to ``("baseline", "isalsr")``.
            An arm with no directory under a problem is skipped, not an error.
    """
    bench_dir = results_dir / method / benchmark
    if not bench_dir.exists():
        return

    variant_list = resolve_variants(variants)
    for problem_dir in sorted(bench_dir.iterdir()):
        if not problem_dir.is_dir():
            continue
        for variant in variant_list:
            variant_dir = problem_dir / variant
            agg_path = variant_dir / "aggregate.csv"
            if variant_dir.exists() and not agg_path.exists():
                logs = load_all_run_logs(variant_dir)
                if logs:
                    log.info("  Computing aggregate for %s/%s", problem_dir.name, variant)
                    agg_rows = aggregate_all_metrics(logs)
                    save_aggregate(agg_rows, agg_path)


# ======================================================================
# Benchmark summary
# ======================================================================


def compute_and_save_benchmark_summaries(
    paired_stats_list: list[Any],
    method: str,
    benchmark: str,
    output_dir: Path,
    suffix: str = "",
) -> list[dict[str, Any]]:
    """Compute benchmark summaries for all metrics and save to CSV.

    Args:
        paired_stats_list: Per-problem PairedStats for one contrast.
        method: Method name.
        benchmark: Benchmark name.
        output_dir: Directory for the CSV.
        suffix: Appended to the filename stem. Empty for the primary contrast,
            so its path is unchanged; secondary contrasts pass their contrast
            name to avoid overwriting it.

    Returns:
        One CSV row dict per metric.
    """
    if not paired_stats_list:
        return []

    rows = []
    for metric_name in METRIC_EXTRACTORS:
        row = benchmark_summary(paired_stats_list, metric_name)
        rows.append(row)

    # Compute solution rates from run logs (benchmark_summary doesn't do this)
    # We'll add them to the first row as a reference
    out_path = output_dir / f"benchmark_summary_{method}_{benchmark}{suffix}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BENCHMARK_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())

    log.info("Saved benchmark summary: %s (%d metrics)", out_path, len(rows))
    return [r.to_csv_row() for r in rows]


# ======================================================================
# Cross-method analysis
# ======================================================================

# Metrics for which a smaller value is a better result. The critical-difference
# machinery ranks larger values first, so these must be negated before ranking.
_LOWER_IS_BETTER: frozenset[str] = frozenset({"nrmse_test", "wall_clock_total_s"})


def run_cross_method(
    results_dir: Path,
    methods: list[str],
    benchmark: str,
    output_dir: Path,
    variants: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run cross-method Friedman/Nemenyi for key metrics.

    Args:
        results_dir: Root results directory.
        methods: Method names.
        benchmark: Benchmark name.
        output_dir: Directory for cross_method_{benchmark}.json.
        variants: Arms entering the (method x variant) matrix. Defaults to
            ``("baseline", "isalsr")``; a three-arm campaign gives
            ``n_methods x 3`` groups.

    Returns:
        The saved JSON payload as a dict.
    """
    variant_list = resolve_variants(variants)
    results: dict[str, Any] = {
        "benchmark": benchmark,
        "methods": methods,
        "variants": variant_list,
    }

    key_metrics = ["r2_test", "nrmse_test", "wall_clock_total_s"]
    for metric_name in key_metrics:
        extractor = METRIC_EXTRACTORS.get(metric_name)
        if extractor is None:
            continue
        try:
            result = cross_method_friedman(
                results_dir,
                methods,
                benchmark,
                extractor,
                higher_is_better=metric_name not in _LOWER_IS_BETTER,
                variants=variant_list,
            )
            results[metric_name] = result
            log.info(
                "  Friedman (%s): chi2=%.4f p=%.6f",
                metric_name,
                result.get("chi2", 0),
                result.get("p_value", 1),
            )
        except Exception as e:  # noqa: BLE001
            results[metric_name] = {"error": str(e)}
            log.warning("  Friedman (%s) failed: %s", metric_name, e)

    out_path = output_dir / f"cross_method_{benchmark}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved cross-method analysis: %s", out_path)
    return results


def run_reduction_comparison(
    results_dir: Path,
    methods: list[str],
    benchmark: str,
    output_dir: Path,
    variants: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare reduction factors across methods and deduplicating arms.

    Args:
        results_dir: Root results directory.
        methods: Method names.
        benchmark: Benchmark name.
        output_dir: Directory for reduction_comparison_{benchmark}.json.
        variants: Arms to consider. Defaults to ``("baseline", "isalsr")``.

    Returns:
        The saved JSON payload as a dict.
    """
    comparison = compare_reduction_factors(
        results_dir, methods, benchmark, variants=resolve_variants(variants)
    )

    out_path = output_dir / f"reduction_comparison_{benchmark}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    log.info("Saved reduction comparison: %s", out_path)
    return comparison


# ======================================================================
# Cross-Problem Dominance Test (CPDT)
# ======================================================================


# Value a metric takes when an arm produced no usable model. Used only by the
# conservative-substitution sensitivity check (EXECUTION-PLAN §6.4 / E3): a
# non-finite arm_b mean is read as a failure of that arm rather than dropped.
#
#   r2_train / r2_test          0.0  R^2 of the mean predictor; also the floor
#                                    the table generator clips R^2 to.
#   nrmse_test                  1.0  RMSE equal to the target's standard
#                                    deviation, i.e. the mean predictor again.
#   empirical_reduction_factor  1.0  no merging at all (rho == 1 by definition).
#   redundancy_rate             0.0  no redundancy detected.
CONSERVATIVE_FAILURE_VALUES: dict[str, float] = {
    "r2_test": 0.0,
    "r2_train": 0.0,
    "nrmse_test": 1.0,
    "empirical_reduction_factor": 1.0,
    "redundancy_rate": 0.0,
}


def _conservative_arm_b_mean(metric_name: str, arm_a_mean: float) -> float:
    """Worst-case arm_b mean substituted for a non-finite one.

    The substituted delta is clamped so that the problem can never count as a
    win for arm_b. Without the clamp a "failure" value could sit on the
    favourable side of an already-poor arm_a (an arm_a NRMSE above 1, say) and
    the sensitivity check would be anti-conservative.

    Args:
        metric_name: Metric key.
        arm_a_mean: The reference arm's finite mean for this problem.

    Returns:
        The substituted arm_b mean.
    """
    from experiments.models.analyzer.aggregation import CPDT_METRIC_ALTERNATIVES

    failure = CONSERVATIVE_FAILURE_VALUES.get(metric_name, 0.0)
    delta = failure - arm_a_mean
    if CPDT_METRIC_ALTERNATIVES.get(metric_name, "greater") == "greater":
        delta = min(delta, 0.0)
    else:
        delta = max(delta, 0.0)
    return arm_a_mean + delta


def substitute_conservative(
    paired_stats_list: list[Any],
    metric_name: str,
) -> tuple[list[Any], list[str]]:
    """Replace non-finite arm_b means with the worst-case value for one metric.

    A problem whose *reference* arm mean is also non-finite carries no
    information about either arm and is left alone, so it stays dropped under
    both readings.

    Args:
        paired_stats_list: Per-problem PairedStats for one contrast. Not
            mutated.
        metric_name: Metric to substitute.

    Returns:
        (paired stats with the substitution applied, names of the problems
        that were substituted).
    """
    substituted: list[Any] = []
    names: list[str] = []

    for ps in paired_stats_list:
        m = ps.metrics.get(metric_name)
        if m is None or math.isfinite(m.isalsr_mean) or not math.isfinite(m.baseline_mean):
            substituted.append(ps)
            continue
        new_mean = _conservative_arm_b_mean(metric_name, m.baseline_mean)
        new_metric = dataclasses.replace(
            m,
            isalsr_mean=new_mean,
            mean_diff=new_mean - m.baseline_mean,
        )
        substituted.append(dataclasses.replace(ps, metrics={**ps.metrics, metric_name: new_metric}))
        names.append(ps.problem)

    return substituted, names


def _cpdt_summary(result: Any) -> dict[str, Any]:
    """Reduce a CPDT result to the fields the sensitivity block reports.

    Args:
        result: A CrossProblemDominanceResult.

    Returns:
        N, W/T/L, the test identity and the estimation half.
    """
    return {
        "n_problems": result.n_problems,
        "n_wins": result.n_wins,
        "n_ties": result.n_ties,
        "n_losses": result.n_losses,
        "alternative": result.alternative,
        "test_used": result.test_used,
        "p_value_one_sided": result.p_value_one_sided,
        "p_value_two_sided": result.p_value_two_sided,
        "cohens_d": result.cohens_d,
        "mean_delta": result.mean_delta,
    }


def compute_conservative_sensitivity(
    inputs: dict[str, list[Any]],
    computed: dict[str, dict[str, Any]],
    method: str,
    benchmark: str,
) -> dict[str, Any]:
    """Re-run every CPDT under the conservative NaN substitution.

    The headline analysis uses pairwise deletion with the true N reported per
    metric. This block states the sensitivity of that choice: a non-finite
    arm_b mean is treated as a failure of arm_b instead of being dropped, so
    the conservative N is the pairwise N plus the number of substitutions
    (EXECUTION-PLAN §6.4, verified at E3). Both readings are reported; neither
    replaces the other.

    When no substitution applies the conservative reading is the pairwise one
    by construction, and is copied rather than recomputed.

    Args:
        inputs: Contrast name -> per-problem PairedStats.
        computed: Contrast name -> metric -> the pairwise-deletion CPDT result.
        method: Method name, for labelling the recomputed results.
        benchmark: Benchmark name, likewise.

    Returns:
        Contrast name -> metric -> {n_substituted, substituted_problems,
        pairwise_deletion, conservative}.
    """
    from experiments.models.analyzer.aggregation import (
        CPDT_METRIC_ALTERNATIVES,
        compute_cross_problem_dominance,
    )

    block: dict[str, Any] = {}
    for contrast_name, stats in inputs.items():
        arm_a, arm_b = CPDT_CONTRAST_ARMS[contrast_name]
        per_metric: dict[str, Any] = {}
        for metric_name in CPDT_METRIC_ALTERNATIVES:
            pairwise = computed.get(contrast_name, {}).get(metric_name)
            if pairwise is None:
                continue
            substituted, names = substitute_conservative(stats, metric_name)
            if names:
                conservative = compute_cross_problem_dominance(
                    substituted,
                    metric_name,
                    method=method,
                    benchmark=benchmark,
                    arm_a=arm_a,
                    arm_b=arm_b,
                )
                conservative_summary = _cpdt_summary(conservative)
            else:
                conservative_summary = copy.deepcopy(_cpdt_summary(pairwise))
            per_metric[metric_name] = {
                "n_substituted": len(names),
                "substituted_problems": names,
                "pairwise_deletion": _cpdt_summary(pairwise),
                "conservative": conservative_summary,
            }
            if names:
                log.info(
                    "  CPDT[%s] %s conservative substitution: N %d -> %d (%s)",
                    contrast_name,
                    metric_name,
                    pairwise.n_problems,
                    conservative_summary["n_problems"],
                    ", ".join(names),
                )
        block[contrast_name] = per_metric
    return block


def run_cross_problem_dominance_test(
    paired_stats_list: list[Any],
    method: str,
    benchmark: str,
    output_dir: Path,
    contrast_stats: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Run CPDT for all relevant metrics and all available contrasts.

    The primary contrast (baseline -> isalsr) stays at the top level of the
    saved JSON so existing consumers keep working; every contrast, including
    the primary one, is repeated under the ``"contrasts"`` key. Holm correction
    is applied per metric across the contrasts that carry a finite p-value
    (contrast policy, decided 2026-08-04), so a two-arm root has a family of
    one and ``p_value_holm`` equals the raw primary p-value.

    A ``"sensitivity_conservative"`` block reports every contrast and metric
    twice: once under pairwise deletion (the headline reading) and once with
    non-finite arm_b means substituted by the worst case, with the N used by
    each (EXECUTION-PLAN §6.4 / E3).

    Args:
        paired_stats_list: Per-problem PairedStats for the primary contrast.
        method: Method name.
        benchmark: Benchmark name, or "all" for the pooled analysis.
        output_dir: Directory for cross_problem_dominance_{method}_{benchmark}.json.
        contrast_stats: Optional secondary contrasts, keyed by contrast name
            (``"isalsr_vs_hash"``, ``"hash_vs_baseline"``). Absent contrasts
            are skipped.

    Returns:
        The saved JSON payload as a dict.
    """
    from experiments.models.analyzer.aggregation import (
        CPDT_METRIC_ALTERNATIVES,
        apply_holm_across_contrasts,
        compute_cross_problem_dominance,
    )

    inputs: dict[str, list[Any]] = {CPDT_PRIMARY_CONTRAST: paired_stats_list}
    for contrast_name, _arm_a, _arm_b, _fname in CPDT_CONTRASTS:
        if contrast_name == CPDT_PRIMARY_CONTRAST:
            continue
        stats = (contrast_stats or {}).get(contrast_name)
        if stats:
            inputs[contrast_name] = stats
        else:
            log.info("  CPDT: no data for contrast %s, skipping", contrast_name)

    computed: dict[str, dict[str, Any]] = {}
    errors: dict[str, dict[str, str]] = {}
    for contrast_name, stats in inputs.items():
        arm_a, arm_b = CPDT_CONTRAST_ARMS[contrast_name]
        computed[contrast_name] = {}
        errors[contrast_name] = {}
        for metric_name in CPDT_METRIC_ALTERNATIVES:
            try:
                computed[contrast_name][metric_name] = compute_cross_problem_dominance(
                    stats,
                    metric_name,
                    method=method,
                    benchmark=benchmark,
                    arm_a=arm_a,
                    arm_b=arm_b,
                )
            except Exception as e:  # noqa: BLE001
                errors[contrast_name][metric_name] = str(e)
                log.warning("  CPDT %s/%s failed: %s", contrast_name, metric_name, e)

    apply_holm_across_contrasts(computed)

    contrasts_payload: dict[str, dict[str, Any]] = {}
    for contrast_name in inputs:
        per_metric: dict[str, Any] = {}
        for metric_name in CPDT_METRIC_ALTERNATIVES:
            cpdt = computed[contrast_name].get(metric_name)
            if cpdt is None:
                per_metric[metric_name] = {"error": errors[contrast_name][metric_name]}
                continue
            per_metric[metric_name] = cpdt.to_dict()
            log.info(
                "  CPDT[%s] %s: N=%d wins=%d ties=%d losses=%d alt=%s "
                "p_1s=%.6f p_2s=%.6f p_holm=%s d=%.4f",
                contrast_name,
                metric_name,
                cpdt.n_problems,
                cpdt.n_wins,
                cpdt.n_ties,
                cpdt.n_losses,
                cpdt.alternative,
                cpdt.p_value_one_sided,
                cpdt.p_value_two_sided,
                "n/a" if cpdt.p_value_holm is None else f"{cpdt.p_value_holm:.6f}",
                cpdt.cohens_d,
            )
        contrasts_payload[contrast_name] = per_metric

    # Top level keeps the historical shape: metric -> result dict for the
    # primary contrast.
    results: dict[str, Any] = dict(contrasts_payload[CPDT_PRIMARY_CONTRAST])
    results["contrasts"] = contrasts_payload
    results["sensitivity_conservative"] = compute_conservative_sensitivity(
        inputs, computed, method, benchmark
    )

    out_path = output_dir / f"cross_problem_dominance_{method}_{benchmark}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved CPDT: %s", out_path)
    return results


# ======================================================================
# Three-axis computational overhead analysis
# ======================================================================


_K_RANGES = [(0, 5), (5, 15), (15, 32)]


def _safe_stats(values: list[float]) -> dict[str, float]:
    """Compute mean/median/std/min/max with NaN safety."""
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    arr = sorted(clean)
    n = len(arr)
    mean = sum(arr) / n
    median = arr[n // 2] if n % 2 else (arr[n // 2 - 1] + arr[n // 2]) / 2
    var = sum((x - mean) ** 2 for x in arr) / max(n - 1, 1)
    return {
        "mean": mean,
        "median": median,
        "std": var**0.5,
        "min": arr[0],
        "max": arr[-1],
        "n": n,
    }


def _overhead_for_variant(
    bench_dir: Path,
    dedup_variant: str,
    reference_variant: str,
) -> dict[str, Any]:
    """Overhead accounting for one deduplicating arm against one reference arm.

    Args:
        bench_dir: ``results_dir / method / benchmark``.
        dedup_variant: Arm whose canonicalisation cost is being measured.
        reference_variant: Arm supplying the matched-seed timing denominator.

    Returns:
        {per_problem, by_k_range, aggregate}.
    """
    per_problem: list[dict[str, Any]] = []
    # Flat lists for aggregate stats
    all_overhead_pct: list[float] = []
    all_per_dag_ms: list[float] = []
    all_search_ratio: list[float] = []
    all_total_ratio: list[float] = []
    all_rfs: list[float] = []
    # For k-range breakdown: list of (max_k, overhead_pct, per_dag_ms, rf)
    k_data: list[tuple[int, float, float, float]] = []

    for problem_dir in sorted(bench_dir.iterdir()):
        if not problem_dir.is_dir():
            continue

        isalsr_dir = problem_dir / dedup_variant
        baseline_dir = problem_dir / reference_variant
        if not isalsr_dir.exists():
            continue

        isalsr_logs = load_all_run_logs(isalsr_dir)
        baseline_logs = load_all_run_logs(baseline_dir) if baseline_dir.exists() else []

        # Index baseline by seed for matching
        bl_by_seed = {rl.metadata.seed: rl for rl in baseline_logs}

        p_overheads: list[float] = []
        p_per_dags: list[float] = []
        p_search_ratios: list[float] = []
        p_total_ratios: list[float] = []
        p_rfs: list[float] = []
        p_max_ks: list[int] = []

        for rl in isalsr_logs:
            t = rl.time
            ss = rl.search_space
            if t.wall_clock_total_s <= 0 or ss.total_dags_explored <= 0:
                continue

            overhead_pct = t.overhead_time_s / t.wall_clock_total_s * 100
            per_dag_ms = t.canonicalization_runtime_s / ss.total_dags_explored * 1000
            rf = ss.empirical_reduction_factor
            max_k = ss.max_internal_nodes_seen

            p_overheads.append(overhead_pct)
            p_per_dags.append(per_dag_ms)
            p_rfs.append(rf)
            p_max_ks.append(max_k)

            all_overhead_pct.append(overhead_pct)
            all_per_dag_ms.append(per_dag_ms)
            all_rfs.append(rf)
            k_data.append((max_k, overhead_pct, per_dag_ms, rf))

            # Search/total time ratios (need matched baseline seed)
            bl = bl_by_seed.get(rl.metadata.seed)
            if bl is not None and bl.time.wall_clock_search_only_s > 0:
                sr = bl.time.wall_clock_search_only_s / t.wall_clock_search_only_s
                tr = bl.time.wall_clock_total_s / t.wall_clock_total_s
                p_search_ratios.append(sr)
                p_total_ratios.append(tr)
                all_search_ratio.append(sr)
                all_total_ratio.append(tr)

        if p_overheads:
            per_problem.append(
                {
                    "problem": problem_dir.name,
                    "n_seeds": len(p_overheads),
                    "overhead_pct": _safe_stats(p_overheads),
                    "per_dag_canon_ms": _safe_stats(p_per_dags),
                    "search_time_ratio": _safe_stats(p_search_ratios) if p_search_ratios else None,
                    "total_time_ratio": _safe_stats(p_total_ratios) if p_total_ratios else None,
                    "reduction_factor": _safe_stats(p_rfs),
                    "max_k_seen": max(p_max_ks) if p_max_ks else 0,
                }
            )

    # K-range breakdown
    k_breakdown: list[dict[str, Any]] = []
    for lo, hi in _K_RANGES:
        subset = [(o, p, r) for mk, o, p, r in k_data if lo <= mk < hi]
        if subset:
            os_list, ps_list, rs_list = zip(*subset, strict=False)
            k_breakdown.append(
                {
                    "k_range": f"[{lo},{hi})",
                    "n_runs": len(subset),
                    "overhead_pct": _safe_stats(list(os_list)),
                    "per_dag_canon_ms": _safe_stats(list(ps_list)),
                    "reduction_factor": _safe_stats(list(rs_list)),
                }
            )

    return {
        "per_problem": per_problem,
        "by_k_range": k_breakdown,
        "aggregate": {
            "overhead_pct": _safe_stats(all_overhead_pct),
            "per_dag_canon_ms": _safe_stats(all_per_dag_ms),
            "search_time_ratio": _safe_stats(all_search_ratio),
            "total_time_ratio": _safe_stats(all_total_ratio),
            "reduction_factor": _safe_stats(all_rfs),
        },
    }


def compute_overhead_analysis(
    results_dir: Path,
    method: str,
    benchmark: str,
    output_dir: Path,
    variants: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute per-problem and aggregate computational overhead analysis.

    Reads each deduplicating arm's run_logs plus the reference arm's to compute
    overhead_pct, per_dag_canon_ms, search_time_ratio, and breakdowns by DAG
    complexity (max_k ranges).

    The top-level ``per_problem``/``by_k_range``/``aggregate`` keys describe the
    primary deduplicating arm (``isalsr`` when requested, otherwise the first
    non-baseline arm), which keeps the two-arm output shape. Every
    deduplicating arm is repeated under ``"by_variant"``.

    Args:
        results_dir: Root results directory.
        method: Method name.
        benchmark: Benchmark name.
        output_dir: Directory for computational_overhead_{method}_{benchmark}.json.
        variants: Arms present in this campaign. Defaults to
            ``("baseline", "isalsr")``.

    Returns:
        The saved JSON payload as a dict, or {} if the benchmark is absent.
    """
    bench_dir = results_dir / method / benchmark
    if not bench_dir.exists():
        return {}

    variant_list = resolve_variants(variants)
    dedup_variants = [v for v in variant_list if v != REFERENCE_VARIANT]
    if not dedup_variants:
        return {}
    primary = "isalsr" if "isalsr" in dedup_variants else dedup_variants[0]

    by_variant = {
        variant: _overhead_for_variant(bench_dir, variant, REFERENCE_VARIANT)
        for variant in dedup_variants
    }

    result: dict[str, Any] = {
        "method": method,
        "benchmark": benchmark,
        "primary_variant": primary,
        "reference_variant": REFERENCE_VARIANT,
        **by_variant[primary],
        "by_variant": by_variant,
    }

    out_path = output_dir / f"computational_overhead_{method}_{benchmark}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved computational overhead: %s", out_path)
    return result


def compute_three_axis_summary(
    method: str,
    benchmark: str,
    overhead: dict[str, Any],
    benchmark_summaries: list[dict[str, Any]],
    reduction: dict[str, Any],
    output_dir: Path,
    cpdt_results: dict[str, Any] | None = None,
    variants: Sequence[str] | None = None,
    contrast_summaries: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Combine overhead + benchmark summaries + reduction into a 3-axis summary.

    Args:
        method: Method name.
        benchmark: Benchmark name.
        overhead: Output of :func:`compute_overhead_analysis`.
        benchmark_summaries: CSV rows for the primary contrast.
        reduction: Output of :func:`compare_reduction_factors`.
        output_dir: Directory for three_axis_summary_{method}_{benchmark}.json.
        cpdt_results: Optional CPDT payload to embed.
        variants: Arms present in this campaign. Defaults to
            ``("baseline", "isalsr")``; the solution-rate block carries one
            entry per arm.
        contrast_summaries: Optional secondary-contrast CSV rows, keyed by
            contrast name. Supplies the solution rate of an arm that is not
            part of the primary contrast; without it that arm is omitted
            rather than reported as zero.

    Returns:
        The saved JSON payload as a dict.
    """
    variant_list = resolve_variants(variants)

    def _find_row(metric: str, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        source = benchmark_summaries if rows is None else rows
        return next((s for s in source if s.get("metric") == metric), {})

    # Axis 1: Search space reduction
    rf_row = _find_row("empirical_reduction_factor")
    rr_row = _find_row("redundancy_rate")
    method_reduction = reduction.get(method, {})
    search_space = {
        "mean_reduction_factor": method_reduction.get("mean_reduction_factor", 0.0),
        "mean_redundancy_rate": method_reduction.get("mean_redundancy_rate", 0.0),
        "std_redundancy_rate": method_reduction.get("std_redundancy_rate", 0.0),
        "n_observations": method_reduction.get("n_observations", 0),
        "n_significant": rf_row.get("n_significant", 0),
        "n_problems": rf_row.get("n_problems", 0),
        "mean_cohens_d_rf": rf_row.get("median_cohens_d", 0.0),
        "mean_cohens_d_rr": rr_row.get("median_cohens_d", 0.0),
        "by_variant": method_reduction.get("by_variant", {}),
    }

    # Axis 2: Regression quality
    def _quality_counts(metric: str) -> dict[str, Any]:
        row = _find_row(metric)
        n_prob = row.get("n_problems", 0)
        n_sig = row.get("n_significant", 0)
        d = row.get("mean_cohens_d", 0.0)
        # Determine direction: positive d = isalsr better for r2, negative = isalsr better for nrmse
        is_improvement_positive = "r2" in metric
        if is_improvement_positive:
            n_improved = n_sig if d > 0 else 0
            n_degraded = n_sig if d < 0 else 0
        else:
            n_improved = n_sig if d < 0 else 0
            n_degraded = n_sig if d > 0 else 0
        return {
            "n_problems": n_prob,
            "n_significant": n_sig,
            "n_improved": n_improved,
            "n_degraded": n_degraded,
            "n_neutral": n_prob - n_sig,
            "mean_cohens_d": d,
            "median_cohens_d": row.get("median_cohens_d", 0.0),
        }

    regression_quality = {
        "r2_test": _quality_counts("r2_test"),
        "r2_train": _quality_counts("r2_train"),
        "nrmse_test": _quality_counts("nrmse_test"),
    }

    # Axis 3: Computational overhead
    agg = overhead.get("aggregate", {})
    computational_overhead = {
        "mean_overhead_pct": agg.get("overhead_pct", {}).get("mean", 0.0),
        "median_overhead_pct": agg.get("overhead_pct", {}).get("median", 0.0),
        "std_overhead_pct": agg.get("overhead_pct", {}).get("std", 0.0),
        "mean_per_dag_ms": agg.get("per_dag_canon_ms", {}).get("mean", 0.0),
        "median_per_dag_ms": agg.get("per_dag_canon_ms", {}).get("median", 0.0),
        "mean_search_time_ratio": agg.get("search_time_ratio", {}).get("mean", 0.0),
        "mean_total_time_ratio": agg.get("total_time_ratio", {}).get("mean", 0.0),
        "n_runs": agg.get("overhead_pct", {}).get("n", 0),
        "by_k_range": overhead.get("by_k_range", []),
        "by_variant": {
            variant: block.get("aggregate", {})
            for variant, block in overhead.get("by_variant", {}).items()
        },
    }

    # Solution rates, one entry per arm. Each arm's rate is read from a
    # contrast that contains it: the reference arm sits in the "baseline" slot
    # of the primary contrast, every deduplicating arm in the "isalsr" slot of
    # the contrast that pairs it against the reference.
    sr_row = _find_row("solution_recovered")
    solution_rate: dict[str, float] = {}
    if REFERENCE_VARIANT in variant_list:
        solution_rate[REFERENCE_VARIANT] = sr_row.get("solution_rate_baseline", 0.0)
    for contrast_name, arm_a, arm_b, _fname in CPDT_CONTRASTS:
        if arm_a != REFERENCE_VARIANT or arm_b not in variant_list:
            continue
        rows = (
            benchmark_summaries
            if contrast_name == CPDT_PRIMARY_CONTRAST
            else (contrast_summaries or {}).get(contrast_name)
        )
        if rows:
            solution_rate[arm_b] = _find_row("solution_recovered", rows).get(
                "solution_rate_isalsr", 0.0
            )

    result: dict[str, Any] = {
        "method": method,
        "benchmark": benchmark,
        "variants": variant_list,
        "search_space": search_space,
        "regression_quality": regression_quality,
        "computational_overhead": computational_overhead,
        "solution_rate": solution_rate,
    }

    if cpdt_results:
        result["cross_problem_dominance"] = cpdt_results

    out_path = output_dir / f"three_axis_summary_{method}_{benchmark}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved three-axis summary: %s", out_path)
    return result


def build_three_axis_global(
    all_three_axis: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Combine all per-(method,benchmark) three-axis summaries into one file."""
    out_path = output_dir / "three_axis_global.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_three_axis, f, indent=2)
    log.info("Saved three-axis global summary: %s", out_path)


# ======================================================================
# Global summary
# ======================================================================


def build_global_summary(
    results_dir: Path,
    methods: list[str],
    benchmarks: list[str],
    all_benchmark_summaries: dict[str, list[dict[str, Any]]],
    all_cross_method: dict[str, dict[str, Any]],
    all_reduction: dict[str, dict[str, Any]],
    output_dir: Path,
    all_cpdt: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Build and save the global summary JSON."""
    from experiments.models.hardware_info import collect_hardware_info

    summary: dict[str, Any] = {
        "metadata": {
            "methods": methods,
            "benchmarks": benchmarks,
            "hardware": collect_hardware_info(),
        },
        "benchmark_summaries": all_benchmark_summaries,
        "cross_method": all_cross_method,
        "reduction_comparison": all_reduction,
    }

    if all_cpdt:
        summary["cross_problem_dominance"] = all_cpdt

    # Extract key highlights
    highlights: dict[str, Any] = {}
    for key, summaries in all_benchmark_summaries.items():
        r2_row = next((s for s in summaries if s.get("metric") == "r2_test"), None)
        red_row = next(
            (s for s in summaries if s.get("metric") == "empirical_reduction_factor"),
            None,
        )
        if r2_row:
            highlights[key] = {
                "r2_n_significant": r2_row.get("n_significant", 0),
                "r2_mean_cohens_d": r2_row.get("mean_cohens_d", 0),
            }
        if red_row:
            highlights.setdefault(key, {})["mean_reduction_factor"] = red_row.get(
                "mean_reduction_factor", 0
            )

    summary["highlights"] = highlights

    out_path = output_dir / "global_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Saved global summary: %s", out_path)


# ======================================================================
# Main
# ======================================================================


def run_analysis(
    results_dir: Path,
    methods: list[str],
    benchmarks: list[str],
    variants: Sequence[str] | None = None,
) -> None:
    """Run the full analysis pipeline.

    Args:
        results_dir: Root results directory.
        methods: Method names.
        benchmarks: Benchmark names.
        variants: Arms present in this campaign. Defaults to
            ``("baseline", "isalsr")``, which reproduces the two-arm pipeline
            exactly. Arm directories that are absent are skipped, never fatal.
    """
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    variant_list = resolve_variants(variants)
    log.info("Analysing arms: %s", ", ".join(variant_list))

    all_benchmark_summaries: dict[str, list[dict[str, Any]]] = {}
    all_contrast_summaries: dict[str, dict[str, list[dict[str, Any]]]] = {}
    all_cross_method: dict[str, dict[str, Any]] = {}
    all_reduction: dict[str, dict[str, Any]] = {}
    all_overhead: dict[str, dict[str, Any]] = {}
    all_three_axis: dict[str, dict[str, Any]] = {}
    all_cpdt: dict[str, dict[str, Any]] = {}
    all_paired_stats: dict[str, list[Any]] = {}
    all_contrast_stats: dict[str, dict[str, list[Any]]] = {}

    # Per-method per-benchmark analysis
    for method in methods:
        for benchmark in benchmarks:
            key = f"{method}_{benchmark}"
            log.info("=== Analyzing %s / %s ===", method, benchmark)

            # Ensure aggregates exist
            recompute_aggregates_if_needed(results_dir, method, benchmark, variants=variant_list)

            # Load or compute paired stats
            paired_stats = recompute_paired_stats_if_needed(
                results_dir,
                method,
                benchmark,
                variants=variant_list,
            )

            # Secondary contrasts (three-arm C2 roots only)
            contrast_stats = load_secondary_contrast_stats(results_dir, method, benchmark)
            if contrast_stats:
                all_contrast_stats[key] = contrast_stats
                all_contrast_summaries[key] = {
                    contrast_name: compute_and_save_benchmark_summaries(
                        stats,
                        method,
                        benchmark,
                        analysis_dir,
                        suffix=f"_{contrast_name}",
                    )
                    for contrast_name, stats in contrast_stats.items()
                }

            if paired_stats:
                all_paired_stats[key] = paired_stats
                summaries = compute_and_save_benchmark_summaries(
                    paired_stats,
                    method,
                    benchmark,
                    analysis_dir,
                )
                all_benchmark_summaries[key] = summaries
                log.info("  %d problems with paired stats", len(paired_stats))

                # Cross-Problem Dominance Test (per benchmark)
                log.info("  Computing CPDT for %s/%s...", method, benchmark)
                cpdt = run_cross_problem_dominance_test(
                    paired_stats,
                    method,
                    benchmark,
                    analysis_dir,
                    contrast_stats=contrast_stats,
                )
                all_cpdt[key] = cpdt
            else:
                log.warning("  No paired stats found for %s/%s", method, benchmark)

            # Computational overhead analysis (reads raw run_logs)
            log.info("  Computing overhead analysis...")
            overhead = compute_overhead_analysis(
                results_dir, method, benchmark, analysis_dir, variants=variant_list
            )
            all_overhead[key] = overhead

    # Cross-method analysis (per benchmark, needs >= 2 methods)
    if len(methods) >= 2:
        for benchmark in benchmarks:
            log.info("=== Cross-method analysis: %s ===", benchmark)
            try:
                cross = run_cross_method(
                    results_dir, methods, benchmark, analysis_dir, variants=variant_list
                )
                all_cross_method[benchmark] = cross
            except Exception as e:  # noqa: BLE001
                log.warning("Cross-method analysis failed for %s: %s", benchmark, e)

            try:
                reduction = run_reduction_comparison(
                    results_dir,
                    methods,
                    benchmark,
                    analysis_dir,
                    variants=variant_list,
                )
                all_reduction[benchmark] = reduction
            except Exception as e:  # noqa: BLE001
                log.warning("Reduction comparison failed for %s: %s", benchmark, e)
    else:
        log.info("Skipping cross-method analysis (need >= 2 methods, got %d)", len(methods))

    # Pooled CPDT across all benchmarks per method
    for method in methods:
        pooled_ps: list[Any] = []
        pooled_contrasts: dict[str, list[Any]] = {}
        for benchmark in benchmarks:
            key = f"{method}_{benchmark}"
            if key in all_paired_stats:
                pooled_ps.extend(all_paired_stats[key])
            # Pool each secondary contrast separately: the problem sets can
            # differ between contrasts when an arm failed on some problem.
            for contrast_name, stats in all_contrast_stats.get(key, {}).items():
                pooled_contrasts.setdefault(contrast_name, []).extend(stats)
        if pooled_ps:
            log.info("=== Pooled CPDT: %s (N=%d problems) ===", method, len(pooled_ps))
            cpdt_all = run_cross_problem_dominance_test(
                pooled_ps,
                method,
                "all",
                analysis_dir,
                contrast_stats=pooled_contrasts,
            )
            all_cpdt[f"{method}_all"] = cpdt_all

    # Three-axis summaries (per method per benchmark)
    for method in methods:
        for benchmark in benchmarks:
            key = f"{method}_{benchmark}"
            log.info("=== Three-axis summary: %s / %s ===", method, benchmark)
            overhead = all_overhead.get(key, {})
            summaries = all_benchmark_summaries.get(key, [])
            reduction = all_reduction.get(benchmark, {})
            cpdt = all_cpdt.get(key)
            three_axis = compute_three_axis_summary(
                method,
                benchmark,
                overhead,
                summaries,
                reduction,
                analysis_dir,
                cpdt_results=cpdt,
                variants=variant_list,
                contrast_summaries=all_contrast_summaries.get(key),
            )
            all_three_axis[key] = three_axis

    # Three-axis global
    build_three_axis_global(all_three_axis, analysis_dir)

    # Global summary
    log.info("=== Building global summary ===")
    build_global_summary(
        results_dir,
        methods,
        benchmarks,
        all_benchmark_summaries,
        all_cross_method,
        all_reduction,
        analysis_dir,
        all_cpdt=all_cpdt,
    )

    log.info("Analysis complete. Results in %s", analysis_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IsalSR end-to-end experiment analysis",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Base results directory (containing method/ subdirectories)",
    )
    parser.add_argument(
        "--methods",
        required=True,
        help="Comma-separated method names (e.g., 'udfs,bingo')",
    )
    parser.add_argument(
        "--benchmarks",
        required=True,
        help="Comma-separated benchmark names (e.g., 'nguyen,feynman')",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help=(
            "Comma-separated arm names (e.g., 'baseline,hash,isalsr'). "
            "Defaults to the two-arm campaign; arms with no directory on disk "
            "are skipped."
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    methods = [m.strip() for m in args.methods.split(",")]
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    run_analysis(results_dir, methods, benchmarks, variants=variants)


if __name__ == "__main__":
    main()
