"""Unit tests for cross-method statistical analysis."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.models.analyzer.cross_method import (
    build_cross_method_matrix,
)
from experiments.models.analyzer.statistical_tests import (
    critical_difference_data,
    friedman_test,
    nemenyi_posthoc,
)


class TestBuildCrossMethodMatrix:
    def test_correct_shape(self):
        results = {
            "udfs": {
                "baseline": np.array([0.9, 0.8, 0.7]),
                "isalsr": np.array([0.95, 0.85, 0.75]),
            },
            "bingo": {
                "baseline": np.array([0.85, 0.75, 0.65]),
                "isalsr": np.array([0.90, 0.80, 0.70]),
            },
        }
        matrix, names = build_cross_method_matrix(results, ["udfs", "bingo"])
        assert matrix.shape == (3, 4)
        assert names == ["udfs_baseline", "udfs_isalsr", "bingo_baseline", "bingo_isalsr"]

    def test_column_order(self):
        results = {
            "udfs": {
                "baseline": np.array([1.0, 2.0]),
                "isalsr": np.array([3.0, 4.0]),
            },
            "bingo": {
                "baseline": np.array([5.0, 6.0]),
                "isalsr": np.array([7.0, 8.0]),
            },
        }
        matrix, _ = build_cross_method_matrix(results, ["udfs", "bingo"])
        np.testing.assert_array_equal(matrix[:, 0], [1.0, 2.0])  # udfs_baseline
        np.testing.assert_array_equal(matrix[:, 1], [3.0, 4.0])  # udfs_isalsr
        np.testing.assert_array_equal(matrix[:, 2], [5.0, 6.0])  # bingo_baseline
        np.testing.assert_array_equal(matrix[:, 3], [7.0, 8.0])  # bingo_isalsr


class TestCrossMethodFriedman:
    def test_4groups_friedman(self):
        """Friedman on 4 groups (2 methods x 2 variants)."""
        rng = np.random.default_rng(42)
        data = np.column_stack(
            [
                rng.normal(10, 1, 12),  # udfs_baseline (best)
                rng.normal(11, 1, 12),  # udfs_isalsr (better)
                rng.normal(5, 1, 12),  # bingo_baseline (worse)
                rng.normal(6, 1, 12),  # bingo_isalsr (slightly better)
            ]
        )
        chi2, p = friedman_test(data)
        assert chi2 > 0
        assert p < 0.05  # should detect difference

    def test_nemenyi_4groups(self):
        rng = np.random.default_rng(42)
        data = np.column_stack(
            [
                rng.normal(10, 1, 12),
                rng.normal(11, 1, 12),
                rng.normal(5, 1, 12),
                rng.normal(6, 1, 12),
            ]
        )
        pairwise = nemenyi_posthoc(data)
        assert pairwise.shape == (4, 4)

    def test_cd_4groups(self):
        rng = np.random.default_rng(42)
        data = np.column_stack(
            [
                rng.normal(10, 1, 12),
                rng.normal(11, 1, 12),
                rng.normal(5, 1, 12),
                rng.normal(6, 1, 12),
            ]
        )
        result = critical_difference_data(
            data,
            ["udfs_base", "udfs_isal", "bingo_base", "bingo_isal"],
        )
        assert result.cd_value > 0
        assert len(result.avg_ranks) == 4
