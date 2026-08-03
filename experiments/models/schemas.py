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
    representation: str  # "baseline", "isalsr" or "hash"
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
    # None = undetermined: the SymPy equivalence check exceeded its wall-clock
    # budget. Excluded from the aggregate rather than scored as a failure, so
    # the timeout cannot masquerade as "not recovered" / "no overlap".
    solution_recovered: bool | None
    jaccard_index: float | None
    model_complexity: int  # number of nodes in expression DAG
    # Number of test points where the recovered expression evaluates to NaN or
    # +/-inf. Non-zero means the expression is undefined on part of the test
    # domain (log of a negative on an extrapolation grid, exp overflow); the
    # run is then scored R2 = 0 / NRMSE = 1 rather than emitting NaN, so the
    # failure is countable instead of propagating (T08, R2.7).
    n_nonfinite_test_predictions: int = 0


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

    # Shadow distinct-cardinality estimates over the SAME candidate stream,
    # keyed by a fixed-order serialisation instead of the canonical string.
    # Populated by the "isalsr" arm when ``shadow_hash`` is enabled; ``None``
    # for arms that do not run the sketches.  HyperLogLog returns an estimate,
    # hence float rather than int.
    shadow_distinct_insertion: float | None = None
    shadow_distinct_topological: float | None = None
    shadow_distinct_topological_commutative: float | None = None

    # Distinct cardinality of the SAME stream keyed by the HOST's own structure
    # in the host's own node order (isalsr.baselines.host_native), i.e. with no
    # adapter renumbering at all.  The three fields above key on the adapter's
    # output, whose VAR/CONST/topological layout is itself a partial canonical
    # form, so they understate the redundancy IsalSR removes; this field is the
    # baseline the reduction factor rho must be measured against.  ``None`` when
    # the arm never saw a host object (the sketches were fed a bare DAG).
    shadow_distinct_host_native: float | None = None

    # Serialisation failures encountered while feeding the four sketches above.
    # Persisted because it is the ONLY evidence that those four cardinalities
    # mean what they say: ``record_shadow`` swallows extractor errors in a bare
    # ``except`` -- correct, since one bad candidate must not kill a 12 h run --
    # so a broken extractor yields plausible-but-low counts with nothing else
    # marking them as junk.  This has fired for real: a formatter once stripped
    # the ``host_native_hash`` import and every record was swallowed while the
    # counter read 0.0.  Before this field existed the only evidence lived in an
    # INFO log line, so verifying a campaign meant grepping one stderr file per
    # task and depended on log retention and on the line staying at INFO.
    # ``None`` = shadow hashing was off for this run, so the question does not apply.
    n_shadow_failures: int | None = None


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
        """Atomic write: write to temp file then rename to prevent 0-byte races."""
        import tempfile

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with open(fd, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            Path(tmp).replace(path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

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


# ======================================================================
# Cross-Problem Dominance Test (CPDT)
# ======================================================================


@dataclass
class CrossProblemDominanceResult:
    """Result of a cross-problem dominance test.

    Treats each problem as one paired observation (mean metric across seeds)
    and runs a single paired test across all N problems. Tests whether
    IsalSR systematically improves over baseline at the population level.

    Reference: Ezequiel Lopez-Rubio proposal (2026-04-28).
    """

    method: str
    benchmark: str  # specific benchmark or "all" for pooled
    metric: str
    alternative: str  # "greater" or "less"
    n_problems: int
    n_wins: int  # delta_i > tie_threshold
    n_ties: int  # |delta_i| <= tie_threshold
    n_losses: int  # delta_i < -tie_threshold
    problem_names: list[str]
    problem_deltas: list[float]
    shapiro_wilk_p: float
    normality_assumed: bool
    test_used: str  # "t_one_sample" or "wilcoxon_signed_rank"
    statistic: float
    p_value_one_sided: float
    p_value_two_sided: float
    cohens_d: float
    cohens_d_ci_lower: float
    cohens_d_ci_upper: float
    mean_delta: float
    mean_delta_ci_lower: float
    mean_delta_ci_upper: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CrossProblemDominanceResult:
        return cls(**d)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load_json(cls, path: Path) -> CrossProblemDominanceResult:
        with open(path) as f:
            return cls.from_dict(json.load(f))
