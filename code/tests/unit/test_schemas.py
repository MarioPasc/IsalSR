"""Tests for the recorded sample size of the two summary artefacts.

Before this change neither ``paired_stats.json`` nor ``aggregate.csv`` recorded
how many seeds a reported statistic came from, so EXECUTION-PLAN §6.2/§6.4's
"true N reported per metric" had nowhere to live and a reviewer asking "how many
seeds is this p from?" had no answer in the artefact. ``PairedStats.n_seeds``
(matched seeds), ``PairedStatsMetric.n`` (pairs surviving NaN pairwise deletion)
and the ``n`` column of ``aggregate.csv`` (runs aggregated) close that gap.

Files written by C1 carry none of the three, so every reader must treat them as
``None`` -- "not recorded" -- rather than ``0`` or an exception.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.models.analyzer.aggregation import (
    aggregate_all_metrics,
    aggregate_seeds,
    compute_paired_stats,
)
from experiments.models.io_utils import save_aggregate
from experiments.models.schemas import (
    AGGREGATE_COLUMNS,
    AggregateRow,
    BestExpression,
    PairedStats,
    PairedStatsMetric,
    RegressionResults,
    RunLog,
    RunMetadata,
    SearchSpaceResults,
    TimeResults,
)


def _run_log(
    seed: int,
    representation: str = "baseline",
    r2_test: float = 0.95,
    jaccard: float | None = 0.5,
) -> RunLog:
    """Build a minimal RunLog for the aggregation entry points."""
    return RunLog(
        metadata=RunMetadata(
            method="udfs",
            representation=representation,
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
            jaccard_index=jaccard,
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
            symbolic_form="x**3",
            isalsr_string="V^",
            canonical_string="V^",
            n_nodes=5,
            n_edges=4,
        ),
    )


def _metric(**overrides: Any) -> PairedStatsMetric:
    """Build a PairedStatsMetric with placeholder statistics."""
    base: dict[str, Any] = {
        "baseline_mean": 0.9,
        "baseline_std": 0.01,
        "isalsr_mean": 0.95,
        "isalsr_std": 0.01,
        "mean_diff": 0.05,
        "std_diff": 0.01,
        "shapiro_wilk_p": 0.5,
        "normality_assumed": True,
        "test_used": "paired_t",
        "statistic": 3.0,
        "p_value_raw": 0.02,
        "p_value_holm": None,
        "cohens_d": 1.2,
        "cohens_d_ci_lower": 0.3,
        "cohens_d_ci_upper": 2.1,
        "mean_diff_ci_lower": 0.01,
        "mean_diff_ci_upper": 0.09,
    }
    base.update(overrides)
    return PairedStatsMetric(**base)


# --------------------------------------------------------------------------- #
# PairedStats / PairedStatsMetric round trip
# --------------------------------------------------------------------------- #


def test_paired_stats_serialises_n_seeds() -> None:
    ps = PairedStats(method="udfs", benchmark="nguyen", problem="Nguyen-1", n_seeds=30)
    assert ps.to_dict()["n_seeds"] == 30


def test_paired_stats_roundtrip_preserves_the_new_fields() -> None:
    ps = PairedStats(
        method="udfs",
        benchmark="nguyen",
        problem="Nguyen-1",
        metrics={"r2_test": _metric(n=28)},
        n_seeds=30,
    )
    back = PairedStats.from_dict(ps.to_dict())
    assert back.n_seeds == 30
    assert back.metrics["r2_test"].n == 28
    assert back == ps


def test_paired_stats_roundtrip_through_json(tmp_path: Path) -> None:
    ps = PairedStats(
        method="udfs",
        benchmark="nguyen",
        problem="Nguyen-1",
        metrics={"r2_test": _metric(n=3)},
        n_seeds=3,
    )
    path = tmp_path / "paired_stats.json"
    ps.save_json(path)
    assert PairedStats.load_json(path) == ps


def test_c1_era_file_without_the_fields_still_loads() -> None:
    """A C1 artefact carries neither key; both must read back as 'not recorded'."""
    legacy = {
        "method": "udfs",
        "benchmark": "nguyen",
        "problem": "Nguyen-1",
        "metrics": {"r2_test": {k: v for k, v in _metric().to_dict().items() if k != "n"}},
    }
    assert "n_seeds" not in legacy
    assert "n" not in legacy["metrics"]["r2_test"]  # type: ignore[index]

    back = PairedStats.from_dict(legacy)
    assert back.n_seeds is None
    assert back.metrics["r2_test"].n is None
    # ``None`` and ``0`` are different claims: the first says the file predates
    # the field, the second says no observation survived NaN deletion.
    assert back.metrics["r2_test"].n != 0


def test_metric_from_dict_ignores_unknown_keys() -> None:
    d = _metric(n=5).to_dict()
    d["a_field_from_a_future_schema"] = 1
    assert PairedStatsMetric.from_dict(d).n == 5


def test_metric_defaults_to_not_recorded() -> None:
    assert _metric().n is None


# --------------------------------------------------------------------------- #
# aggregate.csv
# --------------------------------------------------------------------------- #


def test_aggregate_columns_contain_n() -> None:
    assert AGGREGATE_COLUMNS[-1] == "n"


def test_aggregate_row_csv_keys_match_the_column_list() -> None:
    row = AggregateRow(
        method="udfs",
        representation="baseline",
        benchmark="nguyen",
        problem="Nguyen-1",
        metric="r2_test",
        mean=0.9,
        std=0.01,
        median=0.9,
        q25=0.89,
        q75=0.91,
        min_val=0.88,
        max_val=0.92,
        n=30,
    )
    assert set(row.to_csv_row()) == set(AGGREGATE_COLUMNS)
    assert row.to_csv_row()["n"] == "30"


def test_aggregate_row_without_n_writes_an_empty_cell() -> None:
    row = AggregateRow(
        method="udfs",
        representation="baseline",
        benchmark="nguyen",
        problem="Nguyen-1",
        metric="r2_test",
        mean=0.9,
        std=0.0,
        median=0.9,
        q25=0.9,
        q75=0.9,
        min_val=0.9,
        max_val=0.9,
    )
    assert row.n is None
    assert row.to_csv_row()["n"] == ""


@pytest.mark.parametrize("n_runs", [1, 3, 30])
def test_aggregate_seeds_records_the_number_of_runs(n_runs: int) -> None:
    logs = [_run_log(seed=s) for s in range(n_runs)]
    assert aggregate_seeds(logs, "r2_test").n == n_runs


def test_aggregate_all_metrics_records_n_on_every_row(tmp_path: Path) -> None:
    logs = [_run_log(seed=s) for s in range(3)]
    rows = aggregate_all_metrics(logs)
    assert {r.n for r in rows} == {3}

    path = tmp_path / "aggregate.csv"
    save_aggregate(rows, path)
    with path.open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(rows)
    assert {r["n"] for r in csv_rows} == {"3"}


def test_aggregate_counts_runs_including_nan_valued_ones() -> None:
    """``n`` is the number of runs, and the statistics stay NaN-aware."""
    logs = [_run_log(seed=0, jaccard=None), _run_log(seed=1), _run_log(seed=2)]
    row = aggregate_seeds(logs, "jaccard_index")
    assert row.n == 3
    np.testing.assert_allclose(row.mean, 0.5)


# --------------------------------------------------------------------------- #
# compute_paired_stats
# --------------------------------------------------------------------------- #


def test_compute_paired_stats_records_matched_seeds() -> None:
    baseline = [_run_log(seed=s, r2_test=0.90 + 0.01 * s) for s in range(5)]
    isalsr = [_run_log(seed=s, representation="isalsr", r2_test=0.93 + 0.01 * s) for s in range(5)]
    ps = compute_paired_stats(baseline, isalsr)
    assert ps.n_seeds == 5
    assert ps.metrics["r2_test"].n == 5


def test_n_seeds_counts_only_the_intersection() -> None:
    """Unmatched seeds are dropped, and ``n_seeds`` must report what remains."""
    baseline = [_run_log(seed=s, r2_test=0.90 + 0.01 * s) for s in range(5)]
    isalsr = [
        _run_log(seed=s, representation="isalsr", r2_test=0.93 + 0.01 * s) for s in (0, 1, 2, 7)
    ]
    ps = compute_paired_stats(baseline, isalsr)
    assert ps.n_seeds == 3
    assert ps.metrics["r2_test"].n == 3


def test_per_metric_n_is_at_most_n_seeds_under_nan_deletion() -> None:
    """A metric undefined for one seed reports a smaller N than the pair count."""
    baseline = [_run_log(seed=s, r2_test=0.90 + 0.01 * s) for s in range(4)]
    isalsr = [_run_log(seed=s, representation="isalsr", r2_test=0.93 + 0.01 * s) for s in range(4)]
    # jaccard_index is extracted through _nan_if_none, so a None on either arm
    # makes that pair's difference NaN and it is deleted pairwise.
    isalsr[0] = _run_log(seed=0, representation="isalsr", r2_test=0.93, jaccard=None)

    ps = compute_paired_stats(baseline, isalsr)
    assert ps.n_seeds == 4
    assert ps.metrics["r2_test"].n == 4
    assert ps.metrics["jaccard_index"].n == 3
    assert ps.metrics["jaccard_index"].n < ps.n_seeds
