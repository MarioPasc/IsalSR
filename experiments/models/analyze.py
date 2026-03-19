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
        if baseline_logs and isalsr_logs and len(baseline_logs) == len(isalsr_logs):
            log.info("  Recomputing paired stats for %s", problem_dir.name)
            paired = compute_paired_stats(baseline_logs, isalsr_logs)
            save_paired_stats(paired, ps_path)
            all_stats.append(paired)

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
