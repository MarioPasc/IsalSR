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
from collections.abc import Sequence
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

# Arms analysed when the caller does not say otherwise. The two-arm default
# reproduces the pre-three-arm behaviour of every entry point in this module.
DEFAULT_VARIANTS: tuple[str, ...] = ("baseline", "isalsr")

# Arm that carries no deduplication by construction, hence has no reduction
# factor of its own (rho == 1).
REFERENCE_VARIANT = "baseline"


def _resolve_variants(variants: Sequence[str] | None) -> list[str]:
    """Normalise a caller-supplied variant list.

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


def load_cross_method_results(
    results_dir: Path,
    methods: list[str],
    benchmark: str,
    metric_extractor,
    variants: Sequence[str] | None = None,
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
        variants: Arms to load. Defaults to ``("baseline", "isalsr")``. An arm
            with no directory under any problem yields an empty mapping, which
            :func:`cross_method_friedman` reads as "this arm is absent".

    Returns:
        {method: {variant: {problem_name: mean_value}}}
    """
    results: dict[str, dict[str, dict[str, float]]] = {}
    variant_list = _resolve_variants(variants)

    for method in methods:
        results[method] = {}
        method_dir = results_dir / method / benchmark

        if not method_dir.exists():
            log.warning("Missing method dir: %s", method_dir)
            continue

        for variant in variant_list:
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
    variants: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Build the (n_problems x n_groups) matrix for Friedman test.

    Groups are ordered as: method1_variant1, method1_variant2, method2_variant1,
    ... so that a three-arm campaign yields ``n_methods x 3`` groups.

    Rows are the sorted intersection of problem names across all
    (method, variant) groups, and any row still holding a non-finite value is
    dropped. The Friedman test ranks within complete blocks (Demsar 2006, JMLR
    7:1-30), so complete-case analysis over problems is the correct
    construction: a problem present for one arm and absent for the other would
    otherwise shift every subsequent row of the paired columns.

    Args:
        results: Output of load_cross_method_results.
        methods: Ordered method names.
        variants: Arms, in group order. Defaults to ``("baseline", "isalsr")``.

    Returns:
        (data_matrix, group_names, problem_names, dropped_problems)
    """
    group_names: list[str] = []
    group_maps: list[dict[str, float]] = []
    variant_list = _resolve_variants(variants)

    for method in methods:
        for variant in variant_list:
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


def _present_variants(
    results: dict[str, dict[str, dict[str, float]]],
    variants: Sequence[str],
) -> list[str]:
    """Drop arms that are absent from every method in a results root.

    An arm requested on the command line but never written to disk (a two-arm
    root queried with ``--variants baseline,hash,isalsr``) would otherwise empty
    the complete-case intersection and silently reduce the matrix to zero rows.
    An arm present for *some* methods only is left in place, so that
    :func:`build_cross_method_matrix` still raises: partial data is a campaign
    defect, not a shape to accommodate.

    Args:
        results: Output of :func:`load_cross_method_results`.
        variants: Requested arms, in group order.

    Returns:
        The arms that survive, order preserved.
    """
    kept: list[str] = []
    for variant in variants:
        if any(results.get(method, {}).get(variant) for method in results):
            kept.append(variant)
        else:
            log.warning("Cross-method: arm %r absent from every method, excluded", variant)
    return kept


def cross_method_friedman(
    results_dir: Path,
    methods: list[str],
    benchmark: str,
    metric_extractor,
    higher_is_better: bool = True,
    variants: Sequence[str] | None = None,
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
        variants: Arms to compare. Defaults to ``("baseline", "isalsr")``; a
            three-arm campaign passes ``("baseline", "hash", "isalsr")`` and
            gets ``n_methods x 3`` groups.

    Returns:
        Dict with: chi2, p_value, group_names, variants, avg_ranks, cd_value,
        nemenyi_pairwise, higher_is_better, n_problems_dropped,
        dropped_problems.
    """
    variant_list = _resolve_variants(variants)
    results = load_cross_method_results(
        results_dir,
        methods,
        benchmark,
        metric_extractor,
        variants=variant_list,
    )
    variant_list = _present_variants(results, variant_list)

    data_matrix, group_names, problem_names, dropped = build_cross_method_matrix(
        results, methods, variants=variant_list
    )
    if not higher_is_better:
        data_matrix = -data_matrix
    n_problems, n_groups = data_matrix.shape

    output: dict[str, Any] = {
        "n_problems": n_problems,
        "n_groups": n_groups,
        "group_names": group_names,
        "variants": variant_list,
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


def _reduction_stats_for_variant(method_dir: Path, variant: str) -> dict[str, float] | None:
    """Aggregate the reduction metrics of one arm over every problem.

    Args:
        method_dir: ``results_dir / method / benchmark``.
        variant: Arm directory name under each problem.

    Returns:
        Mean/std reduction statistics, or None if the arm produced no run with
        a non-empty search space.
    """
    redundancy_rates: list[float] = []
    reduction_factors: list[float] = []

    for problem_dir in sorted(method_dir.iterdir()):
        if not problem_dir.is_dir():
            continue
        variant_dir = problem_dir / variant
        if not variant_dir.exists():
            continue

        for rl in load_all_run_logs(variant_dir):
            if rl.search_space.total_dags_explored > 0:
                redundancy_rates.append(rl.search_space.redundancy_rate)
                reduction_factors.append(rl.search_space.empirical_reduction_factor)

    if not redundancy_rates:
        return None
    return {
        "mean_redundancy_rate": float(np.mean(redundancy_rates)),
        "mean_reduction_factor": float(np.mean(reduction_factors)),
        "std_redundancy_rate": float(np.std(redundancy_rates)),
        "n_observations": len(redundancy_rates),
    }


def compare_reduction_factors(
    results_dir: Path,
    methods: list[str],
    benchmark: str,
    variants: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare redundancy reduction across methods, for every deduplicating arm.

    The ``baseline`` arm is skipped: it never merges, so its reduction factor is
    1 and its redundancy rate 0 by construction, and averaging that identity
    would misreport a definition as a measurement.

    The flat keys of each method entry describe the **primary** deduplicating
    arm (``isalsr`` when requested, otherwise the first non-baseline arm), which
    keeps the two-arm output shape that ``compute_three_axis_summary`` reads.
    Every deduplicating arm, including the primary one, is repeated under
    ``"by_variant"``.

    Args:
        results_dir: Base results directory.
        methods: Method names.
        benchmark: Benchmark name.
        variants: Arms to consider. Defaults to ``("baseline", "isalsr")``.

    Returns:
        {method: {mean_redundancy_rate, mean_reduction_factor,
        std_redundancy_rate, n_observations, by_variant}}
    """
    comparison: dict[str, dict[str, Any]] = {}
    dedup_variants = [v for v in _resolve_variants(variants) if v != REFERENCE_VARIANT]
    if not dedup_variants:
        return comparison
    primary = "isalsr" if "isalsr" in dedup_variants else dedup_variants[0]

    for method in methods:
        method_dir = results_dir / method / benchmark
        if not method_dir.exists():
            continue

        by_variant: dict[str, dict[str, float]] = {}
        for variant in dedup_variants:
            stats = _reduction_stats_for_variant(method_dir, variant)
            if stats is not None:
                by_variant[variant] = stats

        if primary in by_variant:
            entry: dict[str, Any] = dict(by_variant[primary])
        elif by_variant:
            entry = dict(next(iter(by_variant.values())))
        else:
            continue
        entry["by_variant"] = by_variant
        comparison[method] = entry

    return comparison
