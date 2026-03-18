"""Unit tests for the statistical analysis pipeline."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.models.analyzer.effect_sizes import (
    cohens_d_ci_bootstrap,
    cohens_d_paired,
    mean_diff_ci,
)
from experiments.models.analyzer.statistical_tests import (
    critical_difference_data,
    friedman_test,
    holm_bonferroni,
    mcnemar_test,
    nemenyi_posthoc,
    paired_ttest,
    shapiro_wilk,
    wilcoxon_signed_rank,
)


class TestShapiroWilk:
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 30)
        w, p = shapiro_wilk(data)
        assert 0 < w <= 1
        assert p > 0.05  # should not reject normality

    def test_uniform_data(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 1, 30)
        w, p = shapiro_wilk(data)
        assert 0 < w <= 1


class TestPairedTtest:
    def test_identical_samples(self):
        a = np.ones(30)
        b = np.ones(30)
        t, p = paired_ttest(a, b)
        assert p == pytest.approx(1.0, abs=0.01)

    def test_different_samples(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 30)
        b = rng.normal(2, 1, 30)  # shifted by 2
        t, p = paired_ttest(a, b)
        assert p < 0.05


class TestWilcoxon:
    def test_identical_samples(self):
        a = np.ones(30)
        b = np.ones(30)
        w, p = wilcoxon_signed_rank(a, b)
        assert p == 1.0

    def test_different_samples(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 30)
        b = rng.normal(2, 1, 30)
        w, p = wilcoxon_signed_rank(a, b)
        assert p < 0.05


class TestMcNemar:
    def test_no_discordance(self):
        a = np.array([True, True, False, False])
        b = np.array([True, True, False, False])
        chi2, p = mcnemar_test(a, b)
        assert p == 1.0

    def test_discordance(self):
        a = np.array([True] * 20 + [False] * 10)
        b = np.array([False] * 20 + [True] * 10)
        chi2, p = mcnemar_test(a, b)
        assert chi2 > 0
        assert 0 <= p <= 1


class TestHolmBonferroni:
    def test_known_values(self):
        # Input: [0.01, 0.04, 0.03, 0.005]
        # Sorted: [0.005, 0.01, 0.03, 0.04]
        # Holm: [0.005*4, 0.01*3, 0.03*2, 0.04*1] = [0.02, 0.03, 0.06, 0.04]
        # Step-down: max(prev, current)
        p_values = [0.01, 0.04, 0.03, 0.005]
        adjusted = holm_bonferroni(p_values)
        assert len(adjusted) == 4
        assert all(a >= r for a, r in zip(adjusted, p_values))
        # The smallest p-value (0.005) should still be significant
        idx_smallest = p_values.index(0.005)
        assert adjusted[idx_smallest] < 0.05

    def test_empty(self):
        assert holm_bonferroni([]) == []

    def test_single(self):
        adjusted = holm_bonferroni([0.03])
        assert adjusted == [pytest.approx(0.03)]


class TestCohensD:
    def test_zero_difference(self):
        d = cohens_d_paired(np.zeros(30))
        assert d == 0.0

    def test_known_value(self):
        # mean=1, std=1 -> d=1
        diffs = np.ones(30)
        d = cohens_d_paired(diffs)
        # For constant differences, std = 0, so d = 0 (special case)
        # Use a spread instead
        rng = np.random.default_rng(42)
        diffs = rng.normal(1.0, 1.0, 100)
        d = cohens_d_paired(diffs)
        assert abs(d - 1.0) < 0.3  # approximately 1.0

    def test_large_effect(self):
        rng = np.random.default_rng(42)
        diffs = rng.normal(5.0, 1.0, 30)
        d = cohens_d_paired(diffs)
        assert abs(d) >= 0.8  # large effect


class TestBootstrapCI:
    def test_ci_contains_true_value(self):
        rng = np.random.default_rng(42)
        diffs = rng.normal(1.0, 1.0, 30)
        lo, hi = cohens_d_ci_bootstrap(diffs, seed=42)
        d = cohens_d_paired(diffs)
        assert lo <= d <= hi

    def test_wider_for_small_samples(self):
        rng = np.random.default_rng(42)
        diffs_small = rng.normal(1.0, 1.0, 10)
        diffs_large = rng.normal(1.0, 1.0, 100)
        lo_s, hi_s = cohens_d_ci_bootstrap(diffs_small, seed=42)
        lo_l, hi_l = cohens_d_ci_bootstrap(diffs_large, seed=42)
        width_small = hi_s - lo_s
        width_large = hi_l - lo_l
        assert width_small > width_large


class TestMeanDiffCI:
    def test_ci_contains_mean(self):
        rng = np.random.default_rng(42)
        diffs = rng.normal(2.0, 1.0, 30)
        mean, lo, hi = mean_diff_ci(diffs)
        assert lo <= mean <= hi

    def test_single_element(self):
        mean, lo, hi = mean_diff_ci(np.array([5.0]))
        assert mean == 5.0


class TestFriedman:
    def test_known_ranking(self):
        # 3 methods, 5 problems. Method 0 always best.
        rng = np.random.default_rng(42)
        data = np.column_stack(
            [
                rng.normal(10, 1, 5),  # method 0: best
                rng.normal(5, 1, 5),  # method 1: middle
                rng.normal(1, 1, 5),  # method 2: worst
            ]
        )
        chi2, p = friedman_test(data)
        assert chi2 > 0
        assert p < 0.05  # should reject null

    def test_too_few_methods_raises(self):
        with pytest.raises(ValueError, match="requires >= 3"):
            friedman_test(np.ones((5, 2)))


class TestNemenyi:
    def test_pairwise_shape(self):
        rng = np.random.default_rng(42)
        data = np.column_stack(
            [
                rng.normal(10, 1, 10),
                rng.normal(5, 1, 10),
                rng.normal(1, 1, 10),
            ]
        )
        pairwise = nemenyi_posthoc(data)
        assert pairwise.shape == (3, 3)
        # Diagonal should be 1.0 (or -1)
        for i in range(3):
            assert abs(pairwise[i, i]) <= 1.0 + 1e-10


class TestCriticalDifference:
    def test_cd_positive(self):
        rng = np.random.default_rng(42)
        data = np.column_stack(
            [
                rng.normal(10, 1, 10),
                rng.normal(5, 1, 10),
                rng.normal(1, 1, 10),
            ]
        )
        result = critical_difference_data(data, ["A", "B", "C"])
        assert result.cd_value > 0
        assert len(result.avg_ranks) == 3
