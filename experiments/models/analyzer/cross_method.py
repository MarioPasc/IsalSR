"""Cross-method statistical analysis.

Compares IsalSR's effect across multiple SR methods using:
- Friedman test (>= 3 groups) on the (n_problems x n_groups) matrix
- Nemenyi post-hoc for pairwise comparisons
- Critical difference diagram data
- Reduction factor comparison across methods

With 2 methods x 2 variants = 4 groups, Friedman is valid (>= 3).
Each row is a problem's mean metric (averaged over seeds).

Uses existing functions from analyzer/statistical_tests.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from experiments.models.analyzer.statistical_tests import (
    critical_difference_data,
    friedman_test,
    nemenyi_posthoc,
)
from experiments.models.io_utils import load_all_run_logs

log = logging.getLogger(__name__)


def load_cross_method_results(
    results_dir: Path,
    methods: list[str],
    benchmark: str,
    metric_extractor,
) -> dict[str, dict[str, dict[str, float]]]:
    """Load per-problem mean metric values for all methods and variants.

    The mean is taken over **finite** values only. A problem with no logs, or
    with no finite value for the metric, is recorded as ``nan`` rather than
    dropped, so that problem identity is preserved and the caller can align
    the paired columns by name instead of by list position.

    Args:
        results_dir: Base results directory.
        methods: List of method names (e.g., ["udfs", "bingo"]).
        benchmark: Benchmark name (e.g., "nguyen").
        metric_extractor: Function RunLog -> float to extract metric.

    Returns:
        {method: {variant: {problem_name: mean_value}}}
    """
    results: dict[str, dict[str, dict[str, float]]] = {}

    for method in methods:
        results[method] = {}
        method_dir = results_dir / method / benchmark

        if not method_dir.exists():
            log.warning("Missing method dir: %s", method_dir)
            continue

        for variant in ["baseline", "isalsr"]:
            problem_means: dict[str, float] = {}
            for problem_dir in sorted(method_dir.iterdir()):
                if not problem_dir.is_dir():
                    continue
                variant_dir = problem_dir / variant
                if not variant_dir.exists():
                    continue

                logs = load_all_run_logs(variant_dir)
                values = [float(metric_extractor(rl)) for rl in logs]
                finite = [v for v in values if np.isfinite(v)]
                problem_means[problem_dir.name] = float(np.mean(finite)) if finite else float("nan")

            results[method][variant] = problem_means

    return results


def build_cross_method_matrix(
    results: dict[str, dict[str, dict[str, float]]],
    methods: list[str],
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Build the (n_problems x n_groups) matrix for Friedman test.

    Groups are ordered as: method1_baseline, method1_isalsr, method2_baseline, ...

    Rows are the sorted intersection of problem names across all
    (method, variant) groups, and any row still holding a non-finite value is
    dropped. The Friedman test ranks within complete blocks (Demsar 2006, JMLR
    7:1-30), so complete-case analysis over problems is the correct
    construction: a problem present for one arm and absent for the other would
    otherwise shift every subsequent row of the paired columns.

    Args:
        results: Output of load_cross_method_results.
        methods: Ordered method names.

    Returns:
        (data_matrix, group_names, problem_names, dropped_problems)
    """
    group_names: list[str] = []
    group_maps: list[dict[str, float]] = []

    for method in methods:
        for variant in ["baseline", "isalsr"]:
            key = f"{method}_{variant}"
            group_names.append(key)
            if method in results and variant in results[method]:
                group_maps.append(dict(results[method][variant]))
            else:
                raise ValueError(f"Missing data for {key}")

    all_names: set[str] = set()
    for gm in group_maps:
        all_names |= set(gm)
    common = sorted(set.intersection(*(set(gm) for gm in group_maps))) if group_maps else []

    missing = sorted(all_names - set(common))
    if missing:
        log.warning(
            "Cross-method: %d problem(s) absent from at least one group, excluded: %s",
            len(missing),
            ", ".join(missing),
        )

    kept: list[str] = []
    non_finite: list[str] = []
    rows: list[list[float]] = []
    for name in common:
        row = [gm[name] for gm in group_maps]
        if all(np.isfinite(v) for v in row):
            kept.append(name)
            rows.append(row)
        else:
            non_finite.append(name)

    if non_finite:
        log.warning(
            "Cross-method: %d problem(s) dropped for non-finite means: %s",
            len(non_finite),
            ", ".join(non_finite),
        )

    dropped = missing + non_finite
    matrix = np.array(rows, dtype=float).reshape(len(rows), len(group_names))
    return matrix, group_names, kept, dropped


def cross_method_friedman(
    results_dir: Path,
    methods: list[str],
    benchmark: str,
    metric_extractor,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Run cross-method Friedman test + Nemenyi post-hoc.

    The matrix is built by complete-case analysis over problems: only problems
    present for every (method, variant) group and finite everywhere enter the
    test, because Friedman ranks within complete blocks (Demsar 2006, JMLR
    7:1-30). Dropped problems are reported in the output.

    Args:
        results_dir: Base results directory.
        methods: Method names.
        benchmark: Benchmark name.
        metric_extractor: Function RunLog -> float.
        higher_is_better: Direction of the metric. When False the matrix is
            negated before ranking, so rank 1 is still the best group. The
            Friedman chi-square and the Nemenyi p-values are invariant under
            negation; only the rank orientation changes.

    Returns:
        Dict with: chi2, p_value, group_names, avg_ranks, cd_value,
        nemenyi_pairwise, higher_is_better, n_problems_dropped,
        dropped_problems.
    """
    results = load_cross_method_results(
        results_dir,
        methods,
        benchmark,
        metric_extractor,
    )

    data_matrix, group_names, problem_names, dropped = build_cross_method_matrix(results, methods)
    if not higher_is_better:
        data_matrix = -data_matrix
    n_problems, n_groups = data_matrix.shape

    output: dict[str, Any] = {
        "n_problems": n_problems,
        "n_groups": n_groups,
        "group_names": group_names,
        "problem_names": problem_names,
        "higher_is_better": higher_is_better,
        "n_problems_dropped": len(dropped),
        "dropped_problems": dropped,
    }

    if n_groups < 3:
        log.warning("Friedman test requires >= 3 groups, got %d", n_groups)
        output["error"] = "insufficient_groups"
        return output

    chi2, p_value = friedman_test(data_matrix)
    output["chi2"] = chi2
    output["p_value"] = p_value

    # Nemenyi post-hoc (regardless of significance, for completeness)
    nemenyi_p = nemenyi_posthoc(data_matrix)
    output["nemenyi_pairwise"] = nemenyi_p.tolist()

    # Critical difference diagram data
    cd_result = critical_difference_data(data_matrix, group_names)
    output["cd_value"] = cd_result.cd_value
    output["avg_ranks"] = cd_result.avg_ranks.tolist()
    output["cliques"] = cd_result.cliques

    return output


def compare_reduction_factors(
    results_dir: Path,
    methods: list[str],
    benchmark: str,
) -> dict[str, dict[str, float]]:
    """Compare IsalSR's redundancy reduction across methods.

    Args:
        results_dir: Base results directory.
        methods: Method names.
        benchmark: Benchmark name.

    Returns:
        {method: {mean_redundancy_rate, mean_reduction_factor, mean_skipped_pct}}
    """
    comparison: dict[str, dict[str, float]] = {}

    for method in methods:
        method_dir = results_dir / method / benchmark
        if not method_dir.exists():
            continue

        redundancy_rates = []
        reduction_factors = []

        for problem_dir in sorted(method_dir.iterdir()):
            if not problem_dir.is_dir():
                continue
            isalsr_dir = problem_dir / "isalsr"
            if not isalsr_dir.exists():
                continue

            logs = load_all_run_logs(isalsr_dir)
            for rl in logs:
                if rl.search_space.total_dags_explored > 0:
                    redundancy_rates.append(rl.search_space.redundancy_rate)
                    reduction_factors.append(rl.search_space.empirical_reduction_factor)

        if redundancy_rates:
            comparison[method] = {
                "mean_redundancy_rate": float(np.mean(redundancy_rates)),
                "mean_reduction_factor": float(np.mean(reduction_factors)),
                "std_redundancy_rate": float(np.std(redundancy_rates)),
                "n_observations": len(redundancy_rates),
            }

    return comparison
