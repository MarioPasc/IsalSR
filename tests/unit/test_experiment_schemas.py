"""Unit tests for experiment schemas."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.models.schemas import (
    AggregateRow,
    BestExpression,
    PairedStats,
    PairedStatsMetric,
    RegressionResults,
    RunLog,
    RunMetadata,
    SearchSpaceResults,
    TimeResults,
    TrajectoryRow,
)


def _make_run_log(seed=1, r2_test=0.95):
    return RunLog(
        metadata=RunMetadata(
            method="udfs",
            representation="baseline",
            benchmark="nguyen",
            problem="Nguyen-1",
            seed=seed,
        ),
        regression=RegressionResults(
            r2_train=0.98,
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
            total_dags_explored=1000,
            unique_canonical_dags=800,
            empirical_reduction_factor=1.25,
            max_internal_nodes_seen=5,
            theoretical_reduction_bound=120.0,
            redundancy_rate=0.2,
        ),
        best_expression=BestExpression(
            symbolic_form="x**3 + x**2 + x",
            isalsr_string="V+V+V^",
            canonical_string="V+V+V^",
            n_nodes=5,
            n_edges=4,
        ),
    )


class TestRunLog:
    def test_to_dict_roundtrip(self):
        rl = _make_run_log()
        d = rl.to_dict()
        rl2 = RunLog.from_dict(d)
        assert rl2.metadata.method == "udfs"
        assert rl2.regression.r2_test == 0.95
        assert rl2.time.wall_clock_total_s == 10.0
        assert rl2.search_space.total_dags_explored == 1000

    def test_json_roundtrip(self):
        rl = _make_run_log()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            rl.save_json(path)
            rl2 = RunLog.load_json(path)
            assert rl2.regression.r2_test == rl.regression.r2_test
            assert rl2.metadata.seed == rl.metadata.seed
        finally:
            path.unlink()

    def test_json_valid_format(self):
        rl = _make_run_log()
        d = rl.to_dict()
        # Must have correct top-level structure
        assert "metadata" in d
        assert "results" in d
        assert "best_expression" in d
        assert "regression" in d["results"]
        assert "time" in d["results"]
        assert "search_space" in d["results"]


class TestTrajectoryRow:
    def test_to_csv_row(self):
        row = TrajectoryRow(
            timestamp_s=1.5,
            iteration=100,
            best_r2=0.95,
            best_nrmse=0.1,
            n_dags_explored=500,
            n_unique_canonical=400,
            current_expr="x+y",
            current_complexity=3,
            cache_hit_rate_cumulative=0.2,
        )
        csv_row = row.to_csv_row()
        assert csv_row["iteration"] == 100
        assert csv_row["current_expr"] == "x+y"
        assert float(csv_row["best_r2"]) == pytest.approx(0.95)


class TestAggregateRow:
    def test_to_csv_row(self):
        row = AggregateRow(
            method="udfs",
            representation="baseline",
            benchmark="nguyen",
            problem="Nguyen-1",
            metric="r2_test",
            mean=0.95,
            std=0.02,
            median=0.96,
            q25=0.93,
            q75=0.97,
            min_val=0.90,
            max_val=0.99,
        )
        csv_row = row.to_csv_row()
        assert csv_row["method"] == "udfs"
        assert float(csv_row["mean"]) == pytest.approx(0.95)


class TestPairedStats:
    def test_json_roundtrip(self):
        ps = PairedStats(
            method="udfs",
            benchmark="nguyen",
            problem="Nguyen-1",
            metrics={
                "r2_test": PairedStatsMetric(
                    baseline_mean=0.90,
                    baseline_std=0.05,
                    isalsr_mean=0.95,
                    isalsr_std=0.03,
                    mean_diff=0.05,
                    std_diff=0.04,
                    shapiro_wilk_p=0.6,
                    normality_assumed=True,
                    test_used="paired_t",
                    statistic=2.5,
                    p_value_raw=0.01,
                    p_value_holm=0.03,
                    cohens_d=1.25,
                    cohens_d_ci_lower=0.5,
                    cohens_d_ci_upper=2.0,
                    mean_diff_ci_lower=0.02,
                    mean_diff_ci_upper=0.08,
                ),
            },
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            ps.save_json(path)
            ps2 = PairedStats.load_json(path)
            assert ps2.metrics["r2_test"].cohens_d == 1.25
            assert ps2.metrics["r2_test"].test_used == "paired_t"
        finally:
            path.unlink()
