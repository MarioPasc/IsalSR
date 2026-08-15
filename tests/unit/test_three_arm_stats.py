"""Three-arm analyzer readiness (check A8 remainder, 2026-08-04).

Campaign C2 runs three arms -- ``baseline`` (native representation),
``hash`` (Naive-Hash deduplication) and ``isalsr`` (canonical-string
deduplication). This module pins the four pieces of analyzer behaviour that a
third arm changes, each against a synthetic case with a hand-computed answer:

1. ``--variants`` reaches every hard-coded two-arm site in ``analyze.py``, and a
   two-arm root keeps working unchanged.
2. Friedman/Nemenyi runs over ``n_methods x 3`` groups with the ranking the
   construction dictates.
3. The Holm family for a tested metric has **three** members, not two
   (EXECUTION-PLAN §4.5, E2's stated pass criterion).
4. The conservative-substitution sensitivity check reports both N's, and they
   differ by exactly the number of substitutions (EXECUTION-PLAN §6.4, E3).
5. No CPDT footer cell ever renders ``nan``, which the rho contrast policy would
   otherwise produce now that the primary rho p-value is withheld by design.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.figures.models.generate_tables import (
    _cpdt_rho_cells,
    _fmt_cpdt_d,
    _fmt_cpdt_p,
    _load_paired_metrics,
    generate_table1,
    generate_table2,
    generate_table_supplementary,
)
from experiments.models.analyze import (
    CONSERVATIVE_FAILURE_VALUES,
    DEFAULT_VARIANTS,
    compute_overhead_analysis,
    recompute_aggregates_if_needed,
    recompute_paired_stats_if_needed,
    resolve_variants,
    run_analysis,
    run_cross_problem_dominance_test,
    substitute_conservative,
)
from experiments.models.analyzer.aggregation import (
    apply_holm_across_contrasts,
    compute_cross_problem_dominance,
)
from experiments.models.analyzer.cross_method import (
    build_cross_method_matrix,
    compare_reduction_factors,
    cross_method_friedman,
    load_cross_method_results,
)
from experiments.models.analyzer.statistical_tests import (
    critical_difference_data,
    friedman_test,
)
from experiments.models.io_utils import save_run_log
from experiments.models.schemas import (
    BestExpression,
    CrossProblemDominanceResult,
    PairedStats,
    PairedStatsMetric,
    RegressionResults,
    RunLog,
    RunMetadata,
    SearchSpaceResults,
    TimeResults,
)

ARMS_3 = ["baseline", "hash", "isalsr"]
ARMS_2 = ["baseline", "isalsr"]
PROBLEMS = ["Nguyen-1", "Nguyen-2", "Nguyen-3", "Nguyen-4"]
SEEDS = (1, 2, 3, 4, 5)

# Per-arm profile: (r2 offset, reduction factor, redundancy rate, wall clock).
# isalsr dominates hash dominates baseline on every axis by construction, so
# every downstream ranking and every contrast sign is known in advance.
_ARM_PROFILE: dict[str, tuple[float, float, float, float]] = {
    "baseline": (0.000, 1.00, 0.00, 12.0),
    "hash": (0.020, 1.20, 0.15, 11.0),
    "isalsr": (0.040, 1.60, 0.35, 10.0),
}


# ----------------------------------------------------------------------
# Synthetic results roots
# ----------------------------------------------------------------------


def _run_log(
    method: str,
    arm: str,
    benchmark: str,
    problem: str,
    seed: int,
    *,
    r2_test: float,
) -> RunLog:
    """Build one RunLog with the fields the analyzer reads.

    Args:
        method: Method name.
        arm: Representation arm.
        benchmark: Benchmark name.
        problem: Problem name.
        seed: Seed number.
        r2_test: Test R^2 to record; also drives train R^2 and NRMSE.

    Returns:
        A fully populated RunLog.
    """
    _, rf, redundancy, wall = _ARM_PROFILE[arm]
    canon = 0.0 if arm == "baseline" else 0.5
    return RunLog(
        metadata=RunMetadata(
            method=method,
            representation=arm,
            benchmark=benchmark,
            problem=problem,
            seed=seed,
        ),
        regression=RegressionResults(
            r2_train=min(r2_test + 0.005, 1.0),
            r2_test=r2_test,
            nrmse_train=0.1,
            nrmse_test=max(1.0 - r2_test, 0.0),
            mse_test=0.01,
            solution_recovered=(arm == "isalsr"),
            jaccard_index=0.5,
            model_complexity=5,
        ),
        time=TimeResults(
            wall_clock_total_s=wall,
            wall_clock_search_only_s=wall - 1.0,
            canonicalization_precomputed_s=0.0,
            canonicalization_runtime_s=canon,
            cache_hit_rate=0.0,
            cache_hits=0,
            cache_misses=0,
            estimated_time_saved_s=0.0,
            time_to_r2_099_s=None,
            time_to_r2_0999_s=None,
            evaluation_time_s=wall - 1.0 - canon,
            overhead_time_s=canon,
        ),
        search_space=SearchSpaceResults(
            total_dags_explored=1000,
            unique_canonical_dags=int(1000 / rf),
            empirical_reduction_factor=rf,
            max_internal_nodes_seen=6,
            theoretical_reduction_bound=720.0,
            redundancy_rate=redundancy,
        ),
        best_expression=BestExpression(
            symbolic_form="x**2",
            isalsr_string="V*",
            canonical_string="V*",
            n_nodes=3,
            n_edges=2,
        ),
    )


def _build_root(
    root: Path,
    methods: list[str],
    arms: list[str],
    benchmark: str = "benchmark",
) -> Path:
    """Write a synthetic results tree under ``root``.

    R^2 rises with the arm profile and with the problem index, so every problem
    contributes a strictly positive isalsr-minus-baseline delta and the CPDT
    direction is known.

    Args:
        root: Directory to populate.
        methods: Method names to write.
        arms: Arm directories to write per problem.
        benchmark: Benchmark directory name.

    Returns:
        ``root``, for chaining.
    """
    for method in methods:
        method_shift = 0.0 if method == "udfs" else 0.001
        for p_idx, problem in enumerate(PROBLEMS):
            slug = problem.lower().replace("-", "_")
            for arm in arms:
                offset = _ARM_PROFILE[arm][0]
                for seed in SEEDS:
                    r2 = 0.80 + offset + 0.01 * p_idx + 0.0005 * seed + method_shift
                    path = root / method / benchmark / slug / arm / f"seed_{seed}" / "run_log.json"
                    save_run_log(
                        _run_log(method, arm, benchmark, problem, seed, r2_test=r2),
                        path,
                    )
    return root


@pytest.fixture(scope="module")
def root_3arm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("three_arm")
    return _build_root(root, ["udfs", "bingo"], ARMS_3)


@pytest.fixture(scope="module")
def root_2arm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("two_arm")
    return _build_root(root, ["udfs", "bingo"], ARMS_2)


@pytest.fixture(scope="module")
def analysed_3arm(root_3arm: Path) -> Path:
    run_analysis(root_3arm, ["udfs", "bingo"], ["benchmark"], variants=ARMS_3)
    return root_3arm


@pytest.fixture(scope="module")
def analysed_2arm(root_2arm: Path) -> Path:
    run_analysis(root_2arm, ["udfs", "bingo"], ["benchmark"])
    return root_2arm


# ----------------------------------------------------------------------
# PairedStats helpers (mirroring tests/unit/test_cpdt_contrasts.py)
# ----------------------------------------------------------------------


def _metric(baseline: float, isalsr: float) -> PairedStatsMetric:
    """Build a PairedStatsMetric with the two arm means set explicitly."""
    return PairedStatsMetric(
        baseline_mean=baseline,
        baseline_std=0.0,
        isalsr_mean=isalsr,
        isalsr_std=0.0,
        mean_diff=isalsr - baseline,
        std_diff=0.0,
        shapiro_wilk_p=1.0,
        normality_assumed=True,
        test_used="paired_t",
        statistic=0.0,
        p_value_raw=1.0,
        p_value_holm=None,
        cohens_d=0.0,
        cohens_d_ci_lower=0.0,
        cohens_d_ci_upper=0.0,
        mean_diff_ci_lower=0.0,
        mean_diff_ci_upper=0.0,
        n=30,
    )


def _paired_stats(rows: list[tuple[float, float]], metric: str = "r2_test") -> list[PairedStats]:
    """One PairedStats per (baseline_mean, isalsr_mean) pair."""
    return [
        PairedStats(
            method="bingo",
            benchmark="benchmark",
            problem=f"P{i:03d}",
            metrics={metric: _metric(bl, is_)},
            n_seeds=30,
        )
        for i, (bl, is_) in enumerate(rows)
    ]


def _cpdt_result(
    metric: str,
    alternative: str,
    p_one: float,
    p_two: float,
    arm_a: str,
    arm_b: str,
) -> CrossProblemDominanceResult:
    """A CPDT result with p-values set by hand, for Holm arithmetic tests."""
    return CrossProblemDominanceResult(
        method="bingo",
        benchmark="benchmark",
        metric=metric,
        alternative=alternative,
        n_problems=10,
        n_wins=7,
        n_ties=2,
        n_losses=1,
        problem_names=[f"P{i:03d}" for i in range(10)],
        problem_deltas=[0.01] * 10,
        shapiro_wilk_p=0.5,
        normality_assumed=True,
        test_used="t_one_sample",
        statistic=2.0,
        p_value_one_sided=p_one,
        p_value_two_sided=p_two,
        cohens_d=0.5,
        cohens_d_ci_lower=0.1,
        cohens_d_ci_upper=0.9,
        mean_delta=0.01,
        mean_delta_ci_lower=0.0,
        mean_delta_ci_upper=0.02,
        arm_a=arm_a,
        arm_b=arm_b,
    )


def _holm_by_hand(p_values: list[float]) -> list[float]:
    """Holm step-down over a family of size ``len(p_values)``."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    running = 0.0
    adjusted = [0.0] * m
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


# ======================================================================
# (a) --variants plumbing
# ======================================================================


def test_resolve_variants_default_is_two_arm() -> None:
    assert resolve_variants(None) == list(DEFAULT_VARIANTS) == ["baseline", "isalsr"]


def test_resolve_variants_preserves_order_and_dedupes() -> None:
    assert resolve_variants(["isalsr", "baseline", "isalsr"]) == ["isalsr", "baseline"]


def test_aggregates_written_for_every_requested_arm(root_3arm: Path) -> None:
    recompute_aggregates_if_needed(root_3arm, "udfs", "benchmark", variants=ARMS_3)
    for arm in ARMS_3:
        path = root_3arm / "udfs" / "benchmark" / "nguyen_1" / arm / "aggregate.csv"
        assert path.exists(), arm


def test_two_arm_request_does_not_touch_the_hash_arm(tmp_path: Path) -> None:
    """A default-variants call on a three-arm root leaves hash/ untouched."""
    root = _build_root(tmp_path / "r", ["udfs"], ARMS_3)
    recompute_aggregates_if_needed(root, "udfs", "benchmark")
    assert (root / "udfs" / "benchmark" / "nguyen_1" / "isalsr" / "aggregate.csv").exists()
    assert not (root / "udfs" / "benchmark" / "nguyen_1" / "hash" / "aggregate.csv").exists()


def test_paired_stats_written_per_contrast(tmp_path: Path) -> None:
    root = _build_root(tmp_path / "r", ["udfs"], ARMS_3)
    stats = recompute_paired_stats_if_needed(root, "udfs", "benchmark", variants=ARMS_3)

    assert len(stats) == len(PROBLEMS)
    prob_dir = root / "udfs" / "benchmark" / "nguyen_1"
    for fname in (
        "paired_stats.json",
        "paired_stats_isalsr_vs_hash.json",
        "paired_stats_hash_vs_baseline.json",
    ):
        assert (prob_dir / fname).exists(), fname


def test_paired_stats_two_arm_writes_only_the_primary_file(tmp_path: Path) -> None:
    root = _build_root(tmp_path / "r", ["udfs"], ARMS_3)
    recompute_paired_stats_if_needed(root, "udfs", "benchmark")

    prob_dir = root / "udfs" / "benchmark" / "nguyen_1"
    assert (prob_dir / "paired_stats.json").exists()
    assert not (prob_dir / "paired_stats_isalsr_vs_hash.json").exists()
    assert not (prob_dir / "paired_stats_hash_vs_baseline.json").exists()


def test_overhead_reports_every_dedup_arm(root_3arm: Path, tmp_path: Path) -> None:
    out = compute_overhead_analysis(root_3arm, "udfs", "benchmark", tmp_path, variants=ARMS_3)

    assert set(out["by_variant"]) == {"hash", "isalsr"}
    assert out["primary_variant"] == "isalsr"
    # The top level still describes isalsr, as the two-arm consumers expect.
    assert out["aggregate"] == out["by_variant"]["isalsr"]["aggregate"]
    for arm in ("hash", "isalsr"):
        agg = out["by_variant"][arm]["aggregate"]
        assert agg["overhead_pct"]["n"] == len(PROBLEMS) * len(SEEDS)
        assert agg["reduction_factor"]["mean"] == pytest.approx(_ARM_PROFILE[arm][1])


def test_overhead_two_arm_shape_is_unchanged(root_2arm: Path, tmp_path: Path) -> None:
    out = compute_overhead_analysis(root_2arm, "udfs", "benchmark", tmp_path)
    assert set(out["by_variant"]) == {"isalsr"}
    assert out["aggregate"]["reduction_factor"]["mean"] == pytest.approx(1.60)
    assert {"per_problem", "by_k_range", "aggregate", "method", "benchmark"} <= set(out)


def test_run_analysis_three_arm_emits_every_artefact(analysed_3arm: Path) -> None:
    analysis = analysed_3arm / "analysis"
    expected = [
        "benchmark_summary_udfs_benchmark.csv",
        "benchmark_summary_udfs_benchmark_isalsr_vs_hash.csv",
        "benchmark_summary_udfs_benchmark_hash_vs_baseline.csv",
        "computational_overhead_udfs_benchmark.json",
        "cross_method_benchmark.json",
        "reduction_comparison_benchmark.json",
        "cross_problem_dominance_udfs_benchmark.json",
        "cross_problem_dominance_udfs_all.json",
        "three_axis_summary_udfs_benchmark.json",
        "three_axis_global.json",
        "global_summary.json",
    ]
    missing = [name for name in expected if not (analysis / name).exists()]
    assert not missing, missing


def test_run_analysis_three_arm_cpdt_has_three_contrasts(analysed_3arm: Path) -> None:
    payload = json.loads(
        (analysed_3arm / "analysis" / "cross_problem_dominance_udfs_benchmark.json").read_text()
    )
    assert set(payload["contrasts"]) == {
        "isalsr_vs_baseline",
        "isalsr_vs_hash",
        "hash_vs_baseline",
    }


def test_run_analysis_three_arm_solution_rate_covers_every_arm(analysed_3arm: Path) -> None:
    payload = json.loads(
        (analysed_3arm / "analysis" / "three_axis_summary_udfs_benchmark.json").read_text()
    )
    assert payload["variants"] == ARMS_3
    assert set(payload["solution_rate"]) == set(ARMS_3)
    # Only the isalsr arm reports recovery in the synthetic profile.
    assert payload["solution_rate"]["isalsr"] == pytest.approx(1.0)
    assert payload["solution_rate"]["baseline"] == pytest.approx(0.0)
    assert payload["solution_rate"]["hash"] == pytest.approx(0.0)


def test_run_analysis_three_arm_reduction_lists_both_dedup_arms(analysed_3arm: Path) -> None:
    payload = json.loads(
        (analysed_3arm / "analysis" / "reduction_comparison_benchmark.json").read_text()
    )
    assert set(payload["udfs"]["by_variant"]) == {"hash", "isalsr"}
    # Flat keys keep describing isalsr, so two-arm consumers are unaffected.
    assert payload["udfs"]["mean_reduction_factor"] == pytest.approx(1.60)
    assert payload["udfs"]["by_variant"]["hash"]["mean_reduction_factor"] == pytest.approx(1.20)


# ======================================================================
# (b) Friedman / Nemenyi over three arms
# ======================================================================


def _known_ranking_results() -> dict[str, dict[str, dict[str, float]]]:
    """Two methods x three arms with a fixed within-problem ordering.

    Values are chosen so that every problem ranks the six groups identically:
    udfs_isalsr > bingo_isalsr > udfs_hash > bingo_hash > udfs_baseline >
    bingo_baseline. The average ranks are therefore integers and known exactly.
    """
    values = {
        ("udfs", "baseline"): 0.95,
        ("udfs", "hash"): 0.97,
        ("udfs", "isalsr"): 0.99,
        ("bingo", "baseline"): 0.94,
        ("bingo", "hash"): 0.96,
        ("bingo", "isalsr"): 0.98,
    }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for (method, arm), base in values.items():
        # A per-problem shift common to all groups leaves the ranking intact.
        out.setdefault(method, {})[arm] = {f"P{i}": base + 0.001 * i for i in range(len(PROBLEMS))}
    return out


def test_three_arm_matrix_has_six_groups_in_declared_order() -> None:
    matrix, groups, problems, dropped = build_cross_method_matrix(
        _known_ranking_results(), ["udfs", "bingo"], variants=ARMS_3
    )
    assert groups == [
        "udfs_baseline",
        "udfs_hash",
        "udfs_isalsr",
        "bingo_baseline",
        "bingo_hash",
        "bingo_isalsr",
    ]
    assert matrix.shape == (len(PROBLEMS), 6)
    assert problems == [f"P{i}" for i in range(len(PROBLEMS))]
    assert dropped == []


def test_three_arm_nemenyi_ranking_is_the_constructed_one() -> None:
    matrix, groups, _, _ = build_cross_method_matrix(
        _known_ranking_results(), ["udfs", "bingo"], variants=ARMS_3
    )
    cd = critical_difference_data(matrix, groups)
    # Rank 1 = best. The ordering is identical in every block, so the average
    # ranks are exactly the within-block ranks.
    np.testing.assert_allclose(cd.avg_ranks, [5.0, 3.0, 1.0, 6.0, 4.0, 2.0])
    chi2, p_value = friedman_test(matrix)
    assert chi2 > 0.0
    assert 0.0 <= p_value <= 1.0


def test_two_arm_matrix_unchanged_by_default() -> None:
    results = _known_ranking_results()
    for method in results:
        del results[method]["hash"]
    matrix, groups, _, _ = build_cross_method_matrix(results, ["udfs", "bingo"])
    assert groups == ["udfs_baseline", "udfs_isalsr", "bingo_baseline", "bingo_isalsr"]
    assert matrix.shape == (len(PROBLEMS), 4)


def test_friedman_end_to_end_over_three_arms(root_3arm: Path) -> None:
    out = cross_method_friedman(
        root_3arm,
        ["udfs", "bingo"],
        "benchmark",
        lambda rl: rl.regression.r2_test,
        variants=ARMS_3,
    )
    assert out["n_groups"] == 6
    assert out["variants"] == ARMS_3
    assert out["n_problems"] == len(PROBLEMS)
    assert out["n_problems_dropped"] == 0
    # isalsr beats hash beats baseline within each method, by construction.
    ranks = dict(zip(out["group_names"], out["avg_ranks"], strict=True))
    for method in ("udfs", "bingo"):
        assert ranks[f"{method}_isalsr"] < ranks[f"{method}_hash"]
        assert ranks[f"{method}_hash"] < ranks[f"{method}_baseline"]


def test_absent_arm_is_dropped_not_fatal(root_2arm: Path) -> None:
    """Asking for hash on a two-arm root degrades to the two-arm analysis."""
    out = cross_method_friedman(
        root_2arm,
        ["udfs", "bingo"],
        "benchmark",
        lambda rl: rl.regression.r2_test,
        variants=ARMS_3,
    )
    assert out["variants"] == ARMS_2
    assert out["n_groups"] == 4
    assert out["n_problems"] == len(PROBLEMS)


def test_load_cross_method_results_three_arms(root_3arm: Path) -> None:
    results = load_cross_method_results(
        root_3arm,
        ["udfs"],
        "benchmark",
        lambda rl: rl.regression.r2_test,
        variants=ARMS_3,
    )
    assert set(results["udfs"]) == set(ARMS_3)
    for arm in ARMS_3:
        assert len(results["udfs"][arm]) == len(PROBLEMS)


def test_compare_reduction_factors_skips_the_baseline_arm(root_3arm: Path) -> None:
    out = compare_reduction_factors(root_3arm, ["udfs"], "benchmark", variants=ARMS_3)
    assert "baseline" not in out["udfs"]["by_variant"]
    assert set(out["udfs"]["by_variant"]) == {"hash", "isalsr"}


# ======================================================================
# (c) Holm across three contrasts
# ======================================================================


def _three_contrast_family(
    metric: str,
    p_primary: tuple[float, float],
    p_hash: tuple[float, float],
    p_bh: tuple[float, float],
) -> dict[str, dict[str, CrossProblemDominanceResult]]:
    return {
        "isalsr_vs_baseline": {
            metric: _cpdt_result(metric, "greater", *p_primary, "baseline", "isalsr")
        },
        "isalsr_vs_hash": {metric: _cpdt_result(metric, "two-sided", *p_hash, "hash", "isalsr")},
        "hash_vs_baseline": {metric: _cpdt_result(metric, "two-sided", *p_bh, "baseline", "hash")},
    }


def test_holm_divides_by_three_not_two() -> None:
    """A8's stated pass criterion: the family is the three pairwise contrasts.

    Raw p-values entering the family are 0.01 (one-sided, primary), 0.02
    (two-sided) and 0.04 (two-sided). Holm over three gives
    ``[3*0.01, 2*0.02, 1*0.04] = [0.03, 0.04, 0.04]`` after the running-max
    monotonicity step. Holm over two would give the primary 0.02.
    """
    results = _three_contrast_family("r2_test", (0.01, 0.02), (0.01, 0.02), (0.02, 0.04))
    apply_holm_across_contrasts(results)

    adjusted = [
        results[name]["r2_test"].p_value_holm
        for name in ("isalsr_vs_baseline", "isalsr_vs_hash", "hash_vs_baseline")
    ]
    np.testing.assert_allclose(adjusted, [0.03, 0.04, 0.04], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(adjusted, _holm_by_hand([0.01, 0.02, 0.04]))

    # The same primary p under a two-contrast family would be 0.02: the
    # difference is exactly the third multiplier.
    two = {k: v for k, v in results.items() if k != "hash_vs_baseline"}
    for per_metric in two.values():
        per_metric["r2_test"].p_value_holm = None
    apply_holm_across_contrasts(two)
    assert two["isalsr_vs_baseline"]["r2_test"].p_value_holm == pytest.approx(0.02)
    assert adjusted[0] == pytest.approx(3 * 0.01)
    assert adjusted[0] != pytest.approx(2 * 0.01)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # 3*0.01, 2*0.02, 1*0.04 -> running max -> 0.03, 0.04, 0.04
        ([0.01, 0.02, 0.04], [0.03, 0.04, 0.04]),
        # 3*0.001, 2*0.5 = 1.0 (capped), then 1*0.6 < 1.0 so it inherits 1.0
        ([0.001, 0.5, 0.6], [0.003, 1.0, 1.0]),
        # 3*0.4 = 1.2, capped at 1; every later step inherits the running max
        ([0.4, 0.4, 0.4], [1.0, 1.0, 1.0]),
        # Step-down follows the sorted order, not the contrast order:
        # 3*0.01 = 0.03, 2*0.02 = 0.04, 1*0.03 -> running max 0.04
        ([0.02, 0.01, 0.03], [0.04, 0.03, 0.04]),
    ],
)
def test_holm_three_contrast_arithmetic(raw: list[float], expected: list[float]) -> None:
    results = _three_contrast_family(
        "r2_test", (raw[0], raw[0]), (raw[1], raw[1]), (raw[2], raw[2])
    )
    apply_holm_across_contrasts(results)
    got = [
        results[name]["r2_test"].p_value_holm
        for name in ("isalsr_vs_baseline", "isalsr_vs_hash", "hash_vs_baseline")
    ]
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)


def test_rho_family_is_one_because_two_contrasts_are_descriptive() -> None:
    """rho is tested only on hash -> isalsr, so Holm is the identity there."""
    stats_by_contrast = {
        "isalsr_vs_baseline": ("baseline", "isalsr"),
        "isalsr_vs_hash": ("hash", "isalsr"),
        "hash_vs_baseline": ("baseline", "hash"),
    }
    rows = [(1.0, 1.0 + 0.05 * i) for i in range(1, 11)]
    results: dict[str, dict[str, CrossProblemDominanceResult]] = {
        name: {
            "empirical_reduction_factor": compute_cross_problem_dominance(
                _paired_stats(rows, "empirical_reduction_factor"),
                "empirical_reduction_factor",
                method="bingo",
                benchmark="benchmark",
                arm_a=arm_a,
                arm_b=arm_b,
            )
        }
        for name, (arm_a, arm_b) in stats_by_contrast.items()
    }
    apply_holm_across_contrasts(results)

    tested = results["isalsr_vs_hash"]["empirical_reduction_factor"]
    assert tested.p_value_holm == pytest.approx(tested.p_value_one_sided)
    for name in ("isalsr_vs_baseline", "hash_vs_baseline"):
        assert results[name]["empirical_reduction_factor"].p_value_holm is None


# ======================================================================
# (d) Conservative-substitution sensitivity check
# ======================================================================


@pytest.mark.parametrize(
    ("metric", "arm_a_mean", "expected"),
    [
        ("r2_test", 0.90, 0.0),
        ("r2_train", 0.50, 0.0),
        ("empirical_reduction_factor", 1.80, 1.0),
        ("redundancy_rate", 0.40, 0.0),
        ("nrmse_test", 0.20, 1.0),
        # Clamped: an arm_a already worse than the failure level must not turn
        # the substitution into a win for arm_b.
        ("nrmse_test", 2.50, 2.50),
        ("r2_test", -0.50, -0.50),
    ],
)
def test_conservative_value_is_never_favourable(
    metric: str, arm_a_mean: float, expected: float
) -> None:
    stats = _paired_stats([(arm_a_mean, float("nan"))], metric)
    out, names = substitute_conservative(stats, metric)
    assert names == ["P000"]
    assert out[0].metrics[metric].isalsr_mean == pytest.approx(expected)


def test_conservative_failure_values_cover_every_cpdt_metric() -> None:
    assert set(CONSERVATIVE_FAILURE_VALUES) == {
        "r2_test",
        "r2_train",
        "nrmse_test",
        "empirical_reduction_factor",
        "redundancy_rate",
    }


def test_substitution_leaves_finite_and_double_nan_rows_alone() -> None:
    rows = [(0.9, 0.95), (0.9, float("nan")), (float("nan"), float("nan"))]
    out, names = substitute_conservative(_paired_stats(rows), "r2_test")
    assert names == ["P001"]
    assert out[0].metrics["r2_test"].isalsr_mean == pytest.approx(0.95)
    assert math.isnan(out[2].metrics["r2_test"].isalsr_mean)


def test_substitution_does_not_mutate_the_input() -> None:
    stats = _paired_stats([(0.9, float("nan"))])
    substitute_conservative(stats, "r2_test")
    assert math.isnan(stats[0].metrics["r2_test"].isalsr_mean)


def test_sensitivity_block_reports_both_n_and_they_differ_by_one(tmp_path: Path) -> None:
    """One synthetic NaN: pairwise N drops it, conservative N keeps it."""
    rows: list[tuple[float, float]] = [(0.90, 0.90 + 0.01 * i) for i in range(1, 10)]
    rows.append((0.90, float("nan")))
    out = run_cross_problem_dominance_test(_paired_stats(rows), "bingo", "benchmark", tmp_path)

    entry = out["sensitivity_conservative"]["isalsr_vs_baseline"]["r2_test"]
    n_pairwise = entry["pairwise_deletion"]["n_problems"]
    n_conservative = entry["conservative"]["n_problems"]
    assert n_pairwise == 9
    assert n_conservative == 10
    assert n_conservative - n_pairwise == 1
    assert entry["n_substituted"] == 1
    assert entry["substituted_problems"] == ["P009"]
    # The substituted problem is a loss, so the conservative reading is weaker.
    assert entry["conservative"]["n_losses"] == entry["pairwise_deletion"]["n_losses"] + 1
    assert entry["conservative"]["mean_delta"] < entry["pairwise_deletion"]["mean_delta"]


def test_sensitivity_block_is_a_no_op_without_nan(tmp_path: Path) -> None:
    rows = [(0.90, 0.90 + 0.01 * i) for i in range(1, 11)]
    out = run_cross_problem_dominance_test(_paired_stats(rows), "bingo", "benchmark", tmp_path)
    entry = out["sensitivity_conservative"]["isalsr_vs_baseline"]["r2_test"]
    assert entry["n_substituted"] == 0
    assert entry["substituted_problems"] == []
    assert entry["conservative"] == entry["pairwise_deletion"]


def test_sensitivity_block_written_to_disk(analysed_3arm: Path) -> None:
    payload = json.loads(
        (analysed_3arm / "analysis" / "cross_problem_dominance_udfs_benchmark.json").read_text()
    )
    block = payload["sensitivity_conservative"]
    assert set(block) == {"isalsr_vs_baseline", "isalsr_vs_hash", "hash_vs_baseline"}
    for per_metric in block.values():
        for entry in per_metric.values():
            assert "pairwise_deletion" in entry
            assert "conservative" in entry
            assert "n_problems" in entry["pairwise_deletion"]
            assert "n_problems" in entry["conservative"]


# ======================================================================
# (e) Table emission: three arms, and never "nan"
# ======================================================================


def _write_cpdt(path: Path, *, with_hash: bool) -> dict[str, Any]:
    """Write a CPDT payload with the rho policy applied, and return it."""
    rows_quality = [(0.90, 0.90 + 0.005 * i) for i in range(1, 11)]
    rows_rho = [(1.0, 1.0 + 0.05 * i) for i in range(1, 11)]
    primary = [
        PairedStats(
            method="bingo",
            benchmark="benchmark",
            problem=f"P{i:03d}",
            metrics={
                "r2_test": _metric(*rows_quality[i]),
                "r2_train": _metric(*rows_quality[i]),
                "nrmse_test": _metric(rows_quality[i][1], rows_quality[i][0]),
                "empirical_reduction_factor": _metric(*rows_rho[i]),
                "redundancy_rate": _metric(*rows_rho[i]),
            },
            n_seeds=30,
        )
        for i in range(10)
    ]
    contrast_stats = {"isalsr_vs_hash": primary, "hash_vs_baseline": primary} if with_hash else None
    return run_cross_problem_dominance_test(
        primary, "bingo", "benchmark", path, contrast_stats=contrast_stats
    )


def _cpdt_footer_lines(tex: str) -> list[str]:
    """The generated CPDT summary rows, excluding the caption that names them."""
    return [line for line in tex.splitlines() if "CPDT" in line and "\\caption{" not in line]


def _problem_rows(tex: str) -> list[str]:
    """The per-problem data rows of a generated table.

    Excludes the CPDT footer, the rules and the preamble, so a column count can
    be asserted on the rows that carry problem data only.
    """
    return [
        line
        for line in tex.splitlines()
        if line.startswith("    ") and "&" in line and "CPDT" not in line
    ]


def _assert_no_nan(tex: str) -> None:
    """No typeset cell may read ``nan``.

    The caption is exempt from the substring scan because the word
    "dominance" contains it; captions carry no numbers.
    """
    for line in tex.splitlines():
        if "\\caption{" in line:
            continue
        assert "nan" not in line.lower(), line


@pytest.mark.parametrize("with_hash", [True, False])
def test_supplementary_cpdt_footer_never_prints_nan(tmp_path: Path, with_hash: bool) -> None:
    results_dir = tmp_path / "results"
    (results_dir / "analysis").mkdir(parents=True)
    _write_cpdt(results_dir / "analysis", with_hash=with_hash)

    out_dir = tmp_path / "figs"
    out_dir.mkdir()
    generate_table_supplementary(results_dir, ["bingo"], ["benchmark"], out_dir)

    tex = (out_dir / "table_supplementary_bingo.tex").read_text()
    footer = _cpdt_footer_lines(tex)
    assert footer, "the CPDT footer row must be emitted"
    for line in footer:
        assert "nan" not in line.lower(), line
    _assert_no_nan(tex)


def test_rho_footer_takes_its_p_from_the_hash_contrast(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir(parents=True)
    payload = _write_cpdt(analysis, with_hash=True)

    tested = payload["contrasts"]["isalsr_vs_hash"]["empirical_reduction_factor"]
    expected_p = tested["p_value_holm"] or tested["p_value_one_sided"]
    d_cell, p_cell = _cpdt_rho_cells(payload)

    assert p_cell == _fmt_cpdt_p(float(expected_p))
    assert d_cell == _fmt_cpdt_d(float(tested["cohens_d"]))
    assert "nan" not in p_cell.lower()


def test_rho_footer_is_descriptive_without_a_hash_contrast(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir(parents=True)
    payload = _write_cpdt(analysis, with_hash=False)

    # The primary rho contrast has no p-value by policy.
    assert math.isnan(payload["empirical_reduction_factor"]["p_value_one_sided"])
    d_cell, p_cell = _cpdt_rho_cells(payload)
    assert p_cell == "---"
    assert d_cell == _fmt_cpdt_d(float(payload["empirical_reduction_factor"]["cohens_d"]))
    assert "nan" not in (d_cell + p_cell).lower()


def test_paired_metrics_load_the_hash_arm(root_3arm: Path) -> None:
    data = _load_paired_metrics(root_3arm, "udfs", "benchmark")
    for d in data.values():
        assert set(d["hs_r2_test"]) == set(SEEDS)
        assert set(d["hs_rf"]) == set(SEEDS)
        assert all(v == pytest.approx(1.20) for v in d["hs_rf"].values())
        assert all(v == pytest.approx(1.60) for v in d["is_rf"].values())


def test_paired_metrics_two_arm_root_has_no_hash_keys(root_2arm: Path) -> None:
    data = _load_paired_metrics(root_2arm, "udfs", "benchmark")
    for d in data.values():
        assert "hs_r2_test" not in d
        assert "hs_rf" not in d


def test_table2_emits_three_arms(analysed_3arm: Path, tmp_path: Path) -> None:
    generate_table2(analysed_3arm, ["udfs"], ["benchmark"], tmp_path)
    tex = (tmp_path / "table2_r2_per_problem_udfs.tex").read_text()

    assert "HS $R^2$" in tex
    assert "{@{}l rrr r r r@{}}" in tex
    assert "\\multicolumn{3}{c}{W" in tex
    _assert_no_nan(tex)
    # Every problem row now carries three arm cells plus Delta, d, p.
    body = _problem_rows(tex)
    assert body
    for line in body:
        assert line.count("&") == 6, line


def test_table2_two_arm_root_is_unchanged(analysed_2arm: Path, tmp_path: Path) -> None:
    generate_table2(analysed_2arm, ["udfs"], ["benchmark"], tmp_path)
    tex = (tmp_path / "table2_r2_per_problem_udfs.tex").read_text()

    assert "HS $R^2$" not in tex
    assert "{@{}l rr r r r@{}}" in tex
    assert "\\multicolumn{2}{c}{W" in tex
    _assert_no_nan(tex)
    body = _problem_rows(tex)
    assert body
    for line in body:
        assert line.count("&") == 5, line


def test_table1_emits_three_arms(analysed_3arm: Path, tmp_path: Path) -> None:
    generate_table1(analysed_3arm, ["udfs"], ["benchmark"], tmp_path)
    tex = (tmp_path / "table1_three_axis_summary.tex").read_text()

    assert "$R^2$ (BL/HS/IS)" in tex
    assert "$\\rho$ (IS/HS)" in tex
    _assert_no_nan(tex)


def test_table1_two_arm_root_is_unchanged(analysed_2arm: Path, tmp_path: Path) -> None:
    generate_table1(analysed_2arm, ["udfs"], ["benchmark"], tmp_path)
    tex = (tmp_path / "table1_three_axis_summary.tex").read_text()

    assert "$R^2$ (BL/IS)" in tex
    assert "$\\rho$ &" in tex
    assert "BL/HS/IS" not in tex
    _assert_no_nan(tex)


def test_supplementary_three_arm_root_has_no_nan(analysed_3arm: Path, tmp_path: Path) -> None:
    generate_table_supplementary(analysed_3arm, ["udfs"], ["benchmark"], tmp_path)
    tex = (tmp_path / "table_supplementary_udfs.tex").read_text()
    assert _cpdt_footer_lines(tex)
    _assert_no_nan(tex)


@pytest.mark.parametrize("p", [float("nan"), float("inf")])
def test_fmt_cpdt_p_refuses_non_finite(p: float) -> None:
    assert _fmt_cpdt_p(p) == "$\\dagger$"


def test_fmt_cpdt_d_refuses_non_finite() -> None:
    assert _fmt_cpdt_d(float("nan")) == "$\\dagger$"
    assert _fmt_cpdt_d(0.5) == "$+0.50$"
