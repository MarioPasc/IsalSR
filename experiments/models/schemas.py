"""Unified data schemas for the experimental framework.

Frozen dataclasses matching the experimental design doc (Section C.7).
All schemas are model-agnostic — they define the common format that every
model runner + translator must produce.

Reference: docs/design/experimental_design/isalsr_experimental_design.md
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ======================================================================
# Run metadata
# ======================================================================


@dataclass(frozen=True)
class RunMetadata:
    """Metadata for a single experimental run."""

    method: str  # e.g., "udfs", "graphdsr"
    representation: str  # "baseline" or "isalsr"
    benchmark: str  # e.g., "nguyen", "feynman"
    problem: str  # e.g., "Nguyen-1"
    seed: int
    hardware: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunMetadata:
        return cls(**d)


# ======================================================================
# Three-axis results
# ======================================================================


@dataclass(frozen=True)
class RegressionResults:
    """Regression performance axis (Section B.3.1)."""

    r2_train: float
    r2_test: float
    nrmse_train: float
    nrmse_test: float
    mse_test: float
    solution_recovered: bool
    jaccard_index: float
    model_complexity: int  # number of nodes in expression DAG


@dataclass(frozen=True)
class TimeResults:
    """Time axis (Section B.3.2, amended by Amendment 2)."""

    wall_clock_total_s: float
    wall_clock_search_only_s: float
    canonicalization_precomputed_s: float  # sum of cached canonical timings
    canonicalization_runtime_s: float  # actual on-the-fly canonical time
    cache_hit_rate: float
    cache_hits: int
    cache_misses: int
    estimated_time_saved_s: float
    time_to_r2_099_s: float | None  # None if never reached
    time_to_r2_0999_s: float | None
    evaluation_time_s: float
    overhead_time_s: float


@dataclass(frozen=True)
class SearchSpaceResults:
    """Search space dimensionality axis (Section B.3.3)."""

    total_dags_explored: int
    unique_canonical_dags: int
    empirical_reduction_factor: float  # rho = total / unique
    max_internal_nodes_seen: int
    theoretical_reduction_bound: float  # k! for max k
    redundancy_rate: float  # 1 - unique/total


@dataclass(frozen=True)
class BestExpression:
    """Best expression found during the run."""

    symbolic_form: str  # human-readable string
    isalsr_string: str  # IsalSR instruction string (empty for baseline)
    canonical_string: str  # canonical form (empty for baseline)
    n_nodes: int
    n_edges: int


# ======================================================================
# RunLog — the main per-seed output
# ======================================================================


@dataclass(frozen=True)
class RunLog:
    """Complete per-seed result. Serializes to run_log.json."""

    metadata: RunMetadata
    regression: RegressionResults
    time: TimeResults
    search_space: SearchSpaceResults
    best_expression: BestExpression

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": asdict(self.metadata),
            "results": {
                "regression": asdict(self.regression),
                "time": asdict(self.time),
                "search_space": asdict(self.search_space),
            },
            "best_expression": asdict(self.best_expression),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunLog:
        return cls(
            metadata=RunMetadata(**d["metadata"]),
            regression=RegressionResults(**d["results"]["regression"]),
            time=TimeResults(**d["results"]["time"]),
            search_space=SearchSpaceResults(**d["results"]["search_space"]),
            best_expression=BestExpression(**d["best_expression"]),
        )

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load_json(cls, path: Path) -> RunLog:
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ======================================================================
# Trajectory (time-series during search)
# ======================================================================


@dataclass
class TrajectoryRow:
    """One row of trajectory.csv (Section C.7)."""

    timestamp_s: float
    iteration: int
    best_r2: float
    best_nrmse: float
    n_dags_explored: int
    n_unique_canonical: int
    current_expr: str
    current_complexity: int
    cache_hit_rate_cumulative: float

    COLUMNS: list[str] = field(
        default=None,  # type: ignore[assignment]
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "COLUMNS",
            [
                "timestamp_s",
                "iteration",
                "best_r2",
                "best_nrmse",
                "n_dags_explored",
                "n_unique_canonical",
                "current_expr",
                "current_complexity",
                "cache_hit_rate_cumulative",
            ],
        )

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "timestamp_s": f"{self.timestamp_s:.6f}",
            "iteration": self.iteration,
            "best_r2": f"{self.best_r2:.10f}",
            "best_nrmse": f"{self.best_nrmse:.10f}",
            "n_dags_explored": self.n_dags_explored,
            "n_unique_canonical": self.n_unique_canonical,
            "current_expr": self.current_expr,
            "current_complexity": self.current_complexity,
            "cache_hit_rate_cumulative": f"{self.cache_hit_rate_cumulative:.6f}",
        }


TRAJECTORY_COLUMNS = [
    "timestamp_s",
    "iteration",
    "best_r2",
    "best_nrmse",
    "n_dags_explored",
    "n_unique_canonical",
    "current_expr",
    "current_complexity",
    "cache_hit_rate_cumulative",
]


# ======================================================================
# Aggregate row (30-seed summary)
# ======================================================================


@dataclass(frozen=True)
class AggregateRow:
    """One row of aggregate.csv (Section C.7)."""

    method: str
    representation: str
    benchmark: str
    problem: str
    metric: str
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    min_val: float
    max_val: float

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "representation": self.representation,
            "benchmark": self.benchmark,
            "problem": self.problem,
            "metric": self.metric,
            "mean": f"{self.mean:.10f}",
            "std": f"{self.std:.10f}",
            "median": f"{self.median:.10f}",
            "q25": f"{self.q25:.10f}",
            "q75": f"{self.q75:.10f}",
            "min": f"{self.min_val:.10f}",
            "max": f"{self.max_val:.10f}",
        }


AGGREGATE_COLUMNS = [
    "method",
    "representation",
    "benchmark",
    "problem",
    "metric",
    "mean",
    "std",
    "median",
    "q25",
    "q75",
    "min",
    "max",
]


# ======================================================================
# Paired statistics
# ======================================================================


@dataclass
class PairedStatsMetric:
    """Statistical comparison for one metric between baseline and IsalSR."""

    baseline_mean: float
    baseline_std: float
    isalsr_mean: float
    isalsr_std: float
    mean_diff: float
    std_diff: float
    shapiro_wilk_p: float
    normality_assumed: bool
    test_used: str  # "paired_t" or "wilcoxon"
    statistic: float  # t or W statistic
    p_value_raw: float
    p_value_holm: float | None  # set after Holm correction
    cohens_d: float
    cohens_d_ci_lower: float
    cohens_d_ci_upper: float
    mean_diff_ci_lower: float
    mean_diff_ci_upper: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PairedStatsMetric:
        return cls(**d)


@dataclass
class PairedStats:
    """All paired statistical comparisons for one (method, problem) pair."""

    method: str
    benchmark: str
    problem: str
    metrics: dict[str, PairedStatsMetric] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "benchmark": self.benchmark,
            "problem": self.problem,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PairedStats:
        return cls(
            method=d["method"],
            benchmark=d["benchmark"],
            problem=d["problem"],
            metrics={k: PairedStatsMetric.from_dict(v) for k, v in d["metrics"].items()},
        )

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> PairedStats:
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ======================================================================
# Summary schemas
# ======================================================================


@dataclass(frozen=True)
class BenchmarkSummaryRow:
    """One row of benchmark_summary.csv."""

    method: str
    benchmark: str
    metric: str
    n_problems: int
    n_significant: int  # p < 0.05 after Holm correction
    mean_cohens_d: float
    median_cohens_d: float
    mean_speedup: float
    mean_reduction_factor: float
    solution_rate_baseline: float
    solution_rate_isalsr: float

    def to_csv_row(self) -> dict[str, Any]:
        return asdict(self)


BENCHMARK_SUMMARY_COLUMNS = [
    "method",
    "benchmark",
    "metric",
    "n_problems",
    "n_significant",
    "mean_cohens_d",
    "median_cohens_d",
    "mean_speedup",
    "mean_reduction_factor",
    "solution_rate_baseline",
    "solution_rate_isalsr",
]
