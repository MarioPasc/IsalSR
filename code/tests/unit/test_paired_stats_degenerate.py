"""Zero-variance paired differences must not be typeset as an infinite result.

Surfaced by T19. UDFS's enumeration is close to deterministic, so a structural
descriptor can come out bit-identical across every seed of a problem. The paired
t-test then divides by a zero variance: SciPy returns ``t = ±inf`` and
``p = 0.0``, and ``cohens_d_paired`` independently returns ``0.0`` because it
guards ``sd < 1e-10``. The record therefore reads "no effect whatsoever"
(``d = 0``) beside "infinitely significant" (``p = 0``), which is not a
significant result but an **undefined** one.

The correct reading is that the seed carries no information for that
(problem, metric): zero across-seed variance means the replicates are not
independent draws, so no within-problem inference is identified. Reporting an
exact sign test would be worse than reporting nothing — with 30 identical seeds
it would claim ``p = 2 × 2^-30`` from what is effectively one observation.

This is the same family of defect as T08's NaN-typeset-as-winner, and these
tests fail against the pre-fix implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.models.analyzer.aggregation import (
    apply_holm_correction,
    benchmark_summary,
    compute_paired_stats,
)
from experiments.models.schemas import (
    BestExpression,
    PairedStats,
    PairedStatsMetric,
    RegressionResults,
    RunLog,
    RunMetadata,
    SearchSpaceResults,
    TimeResults,
)

DEGENERATE = "degenerate_zero_variance"


def _run_log(seed: int, representation: str, r2_test: float, total_dags: int) -> RunLog:
    """Build a minimal RunLog.

    Parameters
    ----------
    seed
        Seed number, used to pair baseline against isalsr.
    representation
        Arm name.
    r2_test
        Test R², the metric varied to make a contrast degenerate or not.
    total_dags
        ``total_dags_explored``, a second metric so the family has >1 member.

    Returns
    -------
    RunLog
        A schema-complete run log.
    """
    return RunLog(
        metadata=RunMetadata(
            method="udfs",
            representation=representation,
            benchmark="nguyen",
            problem="Nguyen-1",
            seed=seed,
        ),
        regression=RegressionResults(
            r2_train=r2_test,
            r2_test=r2_test,
            nrmse_train=0.1,
            nrmse_test=0.2,
            mse_test=0.01,
            solution_recovered=False,
            jaccard_index=0.5,
            model_complexity=5,
        ),
        time=TimeResults(
            wall_clock_total_s=10.0,
            wall_clock_search_only_s=9.0,
            canonicalization_precomputed_s=0.0,
            canonicalization_runtime_s=1.0,
            cache_hit_rate=0.0,
            cache_hits=0,
            cache_misses=0,
            estimated_time_saved_s=0.0,
            time_to_r2_099_s=None,
            time_to_r2_0999_s=None,
            evaluation_time_s=9.0,
            overhead_time_s=1.0,
        ),
        search_space=SearchSpaceResults(
            total_dags_explored=total_dags,
            unique_canonical_dags=total_dags,
            empirical_reduction_factor=1.0,
            max_internal_nodes_seen=6,
            theoretical_reduction_bound=720.0,
            redundancy_rate=0.0,
        ),
        best_expression=BestExpression(
            symbolic_form="x_0", isalsr_string="", canonical_string="", n_nodes=2, n_edges=1
        ),
    )


def _pair(
    baseline_r2: list[float],
    isalsr_r2: list[float],
    baseline_dags: list[int] | None = None,
    isalsr_dags: list[int] | None = None,
) -> PairedStats:
    """Run ``compute_paired_stats`` over matched per-seed values."""
    n = len(baseline_r2)
    b_dags = baseline_dags or [1000 + i for i in range(n)]
    i_dags = isalsr_dags or [1100 + 2 * i for i in range(n)]
    return compute_paired_stats(
        [_run_log(i, "baseline", baseline_r2[i], b_dags[i]) for i in range(n)],
        [_run_log(i, "isalsr", isalsr_r2[i], i_dags[i]) for i in range(n)],
    )


class TestDegenerateContrastIsNotSignificant:
    """A constant non-zero difference must not report p = 0 with t = ±inf."""

    def test_constant_nonzero_difference_is_flagged_not_tested(self) -> None:
        # Every seed shows exactly the same +0.02 difference.
        ps = _pair([0.90, 0.90, 0.90, 0.90], [0.92, 0.92, 0.92, 0.92])
        m = ps.metrics["r2_test"]

        assert m.test_used == DEGENERATE
        assert not math.isfinite(m.p_value_raw), "a degenerate contrast must not carry a p-value"
        assert not math.isfinite(m.statistic)

    def test_the_old_infinite_result_is_gone(self) -> None:
        ps = _pair([0.90] * 5, [0.92] * 5)
        m = ps.metrics["r2_test"]
        # The pre-fix implementation produced exactly these two values.
        assert m.p_value_raw != 0.0
        assert m.statistic not in (float("inf"), float("-inf"))

    def test_cohens_d_is_undefined_not_zero_for_a_real_shift(self) -> None:
        # d = mean/sd = 0.02/0 is infinite, i.e. undefined. Reporting 0.0 --
        # "negligible effect" -- next to a real, perfectly consistent shift is
        # the more misleading of the two errors.
        ps = _pair([0.90] * 4, [0.92] * 4)
        m = ps.metrics["r2_test"]
        assert not math.isfinite(m.cohens_d)
        assert not math.isfinite(m.cohens_d_ci_lower)
        assert not math.isfinite(m.cohens_d_ci_upper)

    def test_descriptive_fields_are_still_reported(self) -> None:
        # The contrast is not testable, but it IS describable, and the mean
        # difference is exactly known. Losing it would be its own defect.
        ps = _pair([0.90] * 4, [0.92] * 4)
        m = ps.metrics["r2_test"]
        assert m.mean_diff == pytest.approx(0.02)
        assert m.baseline_mean == pytest.approx(0.90)
        assert m.isalsr_mean == pytest.approx(0.92)
        assert m.std_diff == pytest.approx(0.0)


class TestIdenticalArms:
    """Zero variance AND zero difference is a genuine 'no effect', not undefined."""

    def test_identical_arms_report_zero_effect_and_no_p_value(self) -> None:
        ps = _pair([0.90] * 4, [0.90] * 4)
        m = ps.metrics["r2_test"]
        assert m.test_used == DEGENERATE
        assert not math.isfinite(m.p_value_raw)
        # d = 0/0 is undefined in general, but the numerator being exactly zero
        # means the arms did not differ at all: an effect size of 0 is correct.
        assert m.cohens_d == 0.0
        assert m.mean_diff == pytest.approx(0.0)


class TestNonDegenerateIsUnchanged:
    """The ordinary path must not be disturbed by the guard."""

    @pytest.mark.parametrize("n", [3, 5, 10])
    def test_varying_differences_still_get_a_real_test(self, n: int) -> None:
        rng = np.random.default_rng(0)
        base = [0.90 + float(rng.normal(0, 0.01)) for _ in range(n)]
        isal = [b + 0.02 + float(rng.normal(0, 0.01)) for b in base]
        m = _pair(base, isal).metrics["r2_test"]

        assert m.test_used in ("paired_t", "wilcoxon")
        assert math.isfinite(m.p_value_raw)
        assert 0.0 <= m.p_value_raw <= 1.0
        assert math.isfinite(m.cohens_d)

    def test_a_degenerate_metric_does_not_contaminate_a_healthy_one(self) -> None:
        # r2_test degenerate, total_dags_explored genuinely varying.
        ps = _pair(
            [0.90] * 5,
            [0.92] * 5,
            baseline_dags=[1000, 1010, 1020, 1030, 1040],
            isalsr_dags=[1200, 1180, 1260, 1210, 1300],
        )
        assert ps.metrics["r2_test"].test_used == DEGENERATE
        healthy = ps.metrics["total_dags_explored"]
        assert healthy.test_used in ("paired_t", "wilcoxon")
        assert math.isfinite(healthy.p_value_raw)


class TestHolmExcludesUntestedContrasts:
    """A contrast with no p-value must not inflate the Holm family size."""

    @staticmethod
    def _metric(p_raw: float) -> PairedStatsMetric:
        return PairedStatsMetric(
            baseline_mean=0.9,
            baseline_std=0.01,
            isalsr_mean=0.92,
            isalsr_std=0.01,
            mean_diff=0.02,
            std_diff=0.01,
            shapiro_wilk_p=0.5,
            normality_assumed=True,
            test_used="paired_t" if math.isfinite(p_raw) else DEGENERATE,
            statistic=1.0,
            p_value_raw=p_raw,
            p_value_holm=None,
            cohens_d=0.3,
            cohens_d_ci_lower=0.1,
            cohens_d_ci_upper=0.5,
            mean_diff_ci_lower=0.01,
            mean_diff_ci_upper=0.03,
            n=5,
        )

    def _family(self, p_values: list[float]) -> list[PairedStats]:
        return [
            PairedStats(
                method="udfs",
                benchmark="nguyen",
                problem=f"P{i}",
                metrics={"r2_test": self._metric(p)},
                n_seeds=5,
            )
            for i, p in enumerate(p_values)
        ]

    def test_untestable_contrast_gets_no_corrected_p(self) -> None:
        stats = apply_holm_correction(self._family([0.01, float("nan"), 0.20]))
        assert stats[1].metrics["r2_test"].p_value_holm is None

    def test_family_size_counts_only_real_tests(self) -> None:
        # Two real tests plus one untestable contrast must be corrected as m=2,
        # not m=3. Feeding the NaN to multipletests keeps m=3 and makes every
        # other problem needlessly conservative (0.01 -> 0.03 instead of 0.02).
        with_nan = apply_holm_correction(self._family([0.01, float("nan"), 0.20]))
        without = apply_holm_correction(self._family([0.01, 0.20]))

        assert with_nan[0].metrics["r2_test"].p_value_holm == pytest.approx(
            without[0].metrics["r2_test"].p_value_holm
        )
        assert with_nan[2].metrics["r2_test"].p_value_holm == pytest.approx(
            without[1].metrics["r2_test"].p_value_holm
        )
        assert with_nan[0].metrics["r2_test"].p_value_holm == pytest.approx(0.02)

    def test_all_untestable_family_is_handled(self) -> None:
        stats = apply_holm_correction(self._family([float("nan"), float("nan")]))
        assert all(s.metrics["r2_test"].p_value_holm is None for s in stats)


class TestBenchmarkSummaryDoesNotCountDegenerateAsSignificant:
    """The user-visible consequence: `n_significant` must not absorb a p = 0.

    ``benchmark_summary`` falls back to ``p_value_raw`` when ``p_value_holm`` is
    None and counts ``p < alpha``. Before the fix a degenerate contrast supplied
    ``p_value_holm = 0.0``, so every such problem was tallied as a significant
    win -- which is how an undefined result would have reached a table.
    """

    def test_degenerate_problem_is_not_tallied(self) -> None:
        degenerate = _pair([0.90] * 5, [0.92] * 5)
        summary = benchmark_summary([degenerate], "r2_test")

        assert summary.n_problems == 1
        assert summary.n_significant == 0, "a contrast with no p-value cannot be significant"
        # nanmean over an all-NaN effect size is NaN, which is the honest
        # reading; it must not silently become 0.0.
        assert math.isnan(summary.mean_cohens_d)

    def test_a_genuinely_significant_problem_is_still_tallied(self) -> None:
        rng = np.random.default_rng(1)
        base = [0.80 + float(rng.normal(0, 0.005)) for _ in range(8)]
        isal = [b + 0.05 + float(rng.normal(0, 0.005)) for b in base]
        real = _pair(base, isal)
        apply_holm_correction([real])

        summary = benchmark_summary([real], "r2_test")
        assert summary.n_significant == 1
        assert math.isfinite(summary.mean_cohens_d)
