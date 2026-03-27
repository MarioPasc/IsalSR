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
        --benchmarks nguyen,feynman
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math  # noqa: E402 -- used in _safe_stats
import os
import sys
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


def recompute_paired_stats_if_needed(
    results_dir: Path,
    method: str,
    benchmark: str,
) -> list[Any]:
    """Load or recompute paired stats for all problems in a benchmark.

    If paired_stats.json doesn't exist for a problem but both baseline/ and
    isalsr/ have run_logs, computes paired stats on the fly.
    """
    bench_dir = results_dir / method / benchmark
    if not bench_dir.exists():
        return []

    all_stats = []
    for problem_dir in sorted(bench_dir.iterdir()):
        if not problem_dir.is_dir():
            continue

        ps_path = problem_dir / "paired_stats.json"
        if ps_path.exists():
            all_stats.append(load_paired_stats(ps_path))
            continue

        # Try to recompute
        baseline_dir = problem_dir / "baseline"
        isalsr_dir = problem_dir / "isalsr"
        if not baseline_dir.exists() or not isalsr_dir.exists():
            continue

        baseline_logs = load_all_run_logs(baseline_dir)
        isalsr_logs = load_all_run_logs(isalsr_dir)
        if baseline_logs and isalsr_logs:
            try:
                log.info("  Recomputing paired stats for %s", problem_dir.name)
                paired = compute_paired_stats(baseline_logs, isalsr_logs)
                save_paired_stats(paired, ps_path)
                all_stats.append(paired)
            except ValueError as e:
                log.warning("  Skipping %s: %s", problem_dir.name, e)

    # Apply Holm correction
    if all_stats:
        apply_holm_correction(all_stats)
        for ps in all_stats:
            problem_slug = ps.problem.lower().replace("-", "_")
            ps_path = bench_dir / problem_slug / "paired_stats.json"
            save_paired_stats(ps, ps_path)

    return all_stats


def recompute_aggregates_if_needed(
    results_dir: Path,
    method: str,
    benchmark: str,
) -> None:
    """Ensure aggregate.csv exists for all variants in all problems."""
    bench_dir = results_dir / method / benchmark
    if not bench_dir.exists():
        return

    for problem_dir in sorted(bench_dir.iterdir()):
        if not problem_dir.is_dir():
            continue
        for variant in ["baseline", "isalsr"]:
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
) -> list[dict[str, Any]]:
    """Compute benchmark summaries for all metrics and save to CSV."""
    if not paired_stats_list:
        return []

    rows = []
    for metric_name in METRIC_EXTRACTORS:
        row = benchmark_summary(paired_stats_list, metric_name)
        rows.append(row)

    # Compute solution rates from run logs (benchmark_summary doesn't do this)
    # We'll add them to the first row as a reference
    out_path = output_dir / f"benchmark_summary_{method}_{benchmark}.csv"
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


def run_cross_method(
    results_dir: Path,
    methods: list[str],
    benchmark: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run cross-method Friedman/Nemenyi for key metrics."""
    results: dict[str, Any] = {"benchmark": benchmark, "methods": methods}

    key_metrics = ["r2_test", "nrmse_test", "wall_clock_total_s"]
    for metric_name in key_metrics:
        extractor = METRIC_EXTRACTORS.get(metric_name)
        if extractor is None:
            continue
        try:
            result = cross_method_friedman(results_dir, methods, benchmark, extractor)
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
) -> dict[str, Any]:
    """Compare reduction factors across methods."""
    comparison = compare_reduction_factors(results_dir, methods, benchmark)

    out_path = output_dir / f"reduction_comparison_{benchmark}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    log.info("Saved reduction comparison: %s", out_path)
    return comparison


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


def compute_overhead_analysis(
    results_dir: Path,
    method: str,
    benchmark: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Compute per-problem and aggregate computational overhead analysis.

    Reads isalsr + baseline run_logs to compute overhead_pct, per_dag_canon_ms,
    search_time_ratio, and breakdowns by DAG complexity (max_k ranges).
    """
    bench_dir = results_dir / method / benchmark
    if not bench_dir.exists():
        return {}

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

        isalsr_dir = problem_dir / "isalsr"
        baseline_dir = problem_dir / "baseline"
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

    result: dict[str, Any] = {
        "method": method,
        "benchmark": benchmark,
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
) -> dict[str, Any]:
    """Combine overhead + benchmark summaries + reduction into a 3-axis summary."""

    def _find_row(metric: str) -> dict[str, Any]:
        return next((s for s in benchmark_summaries if s.get("metric") == metric), {})

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
    }

    # Solution rates
    sr_row = _find_row("solution_recovered")
    solution_rate = {
        "baseline": sr_row.get("solution_rate_baseline", 0.0),
        "isalsr": sr_row.get("solution_rate_isalsr", 0.0),
    }

    result: dict[str, Any] = {
        "method": method,
        "benchmark": benchmark,
        "search_space": search_space,
        "regression_quality": regression_quality,
        "computational_overhead": computational_overhead,
        "solution_rate": solution_rate,
    }

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
) -> None:
    """Run the full analysis pipeline."""
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    all_benchmark_summaries: dict[str, list[dict[str, Any]]] = {}
    all_cross_method: dict[str, dict[str, Any]] = {}
    all_reduction: dict[str, dict[str, Any]] = {}
    all_overhead: dict[str, dict[str, Any]] = {}
    all_three_axis: dict[str, dict[str, Any]] = {}

    # Per-method per-benchmark analysis
    for method in methods:
        for benchmark in benchmarks:
            key = f"{method}_{benchmark}"
            log.info("=== Analyzing %s / %s ===", method, benchmark)

            # Ensure aggregates exist
            recompute_aggregates_if_needed(results_dir, method, benchmark)

            # Load or compute paired stats
            paired_stats = recompute_paired_stats_if_needed(
                results_dir,
                method,
                benchmark,
            )

            if paired_stats:
                summaries = compute_and_save_benchmark_summaries(
                    paired_stats,
                    method,
                    benchmark,
                    analysis_dir,
                )
                all_benchmark_summaries[key] = summaries
                log.info("  %d problems with paired stats", len(paired_stats))
            else:
                log.warning("  No paired stats found for %s/%s", method, benchmark)

            # Computational overhead analysis (reads raw run_logs)
            log.info("  Computing overhead analysis...")
            overhead = compute_overhead_analysis(results_dir, method, benchmark, analysis_dir)
            all_overhead[key] = overhead

    # Cross-method analysis (per benchmark, needs >= 2 methods)
    if len(methods) >= 2:
        for benchmark in benchmarks:
            log.info("=== Cross-method analysis: %s ===", benchmark)
            try:
                cross = run_cross_method(results_dir, methods, benchmark, analysis_dir)
                all_cross_method[benchmark] = cross
            except Exception as e:  # noqa: BLE001
                log.warning("Cross-method analysis failed for %s: %s", benchmark, e)

            try:
                reduction = run_reduction_comparison(
                    results_dir,
                    methods,
                    benchmark,
                    analysis_dir,
                )
                all_reduction[benchmark] = reduction
            except Exception as e:  # noqa: BLE001
                log.warning("Reduction comparison failed for %s: %s", benchmark, e)
    else:
        log.info("Skipping cross-method analysis (need >= 2 methods, got %d)", len(methods))

    # Three-axis summaries (per method per benchmark)
    for method in methods:
        for benchmark in benchmarks:
            key = f"{method}_{benchmark}"
            log.info("=== Three-axis summary: %s / %s ===", method, benchmark)
            overhead = all_overhead.get(key, {})
            summaries = all_benchmark_summaries.get(key, [])
            reduction = all_reduction.get(benchmark, {})
            three_axis = compute_three_axis_summary(
                method, benchmark, overhead, summaries, reduction, analysis_dir
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
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    methods = [m.strip() for m in args.methods.split(",")]
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    run_analysis(results_dir, methods, benchmarks)


if __name__ == "__main__":
    main()
