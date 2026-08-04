"""Unified data schemas for the experimental framework.

Frozen dataclasses matching the experimental design doc (Section C.7).
All schemas are model-agnostic — they define the common format that every
model runner + translator must produce.

Reference: docs/design/experimental_design/isalsr_experimental_design.md
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
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

    # SHA-256 over (X_train, y_train, X_test, y_test).  The paired design
    # asserts that all three arms of a (method, problem, seed) cell saw the
    # same data; before this field that assertion was unverifiable, and a
    # generator consuming RNG differently per arm would have made every paired
    # test compare different samples with nothing in the output saying so.
    # Check C4 is an equality over this string.  Empty = a pre-C2 log.
    data_fingerprint: str = ""

    # SHA-256 of the YAML that configured the run, by content.  Distinguishes
    # "the same config file" from "the same configuration" when six arrays are
    # submitted over several days.  Empty = a pre-C2 log.
    config_sha256: str = ""

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

    # Per-candidate work the dedup wrapper performs inside the wall-clock budget
    # but which was outside every timer until this field existed, so it was
    # silently booked as search time.  Both default to 0.0, which is also what a
    # baseline arm reports (no wrapper, no wrapper cost) and what a pre-C2
    # artifact deserialises to.
    #
    #: Adapter conversion host object -> ``LabeledDAG`` (plus the normalisation
    #: coupled to it).  Genuine method cost: the representation cannot be keyed
    #: without it, so it is part of ``overhead_time_s``.
    conversion_time_s: float = 0.0
    #: T04 shadow cardinality sketches.  Audit instrumentation, not method cost:
    #: subtracted from search time but deliberately NOT added to
    #: ``overhead_time_s``, so the overhead figure stays a statement about the
    #: representation rather than about the measurement apparatus.
    shadow_time_s: float = 0.0


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

    # Candidates the adapter refused, i.e. host expressions with no encoding in
    # the alphabet of Definition 3.2.  Persisted for the same reason as the field
    # above: the runner catches the refusal, counts it and evaluates the
    # candidate without deduplication, so a host operator outside the alphabet
    # depresses the reduction factor with nothing else recording it.  The
    # containment invariant is checked before the run starts, which makes this
    # field the empirical confirmation that the check held for every candidate
    # actually produced.  ``None`` = the arm never converted anything.
    n_conversion_failures: int | None = None

    # ------------------------------------------------------------------ #
    # T06 reachability / fallback ledger (EXECUTION-PLAN C1.9-BUG)
    # ------------------------------------------------------------------ #
    # The five paths on which a candidate is evaluated WITHOUT canonical
    # deduplication.  They are the evidence base for R1.2 and they are the
    # denominator of every honest statement about how often the representation
    # actually applies.
    #
    # These are persisted rather than logged because the population they
    # describe exists only while a search runs: a candidate that violated the
    # reachability precondition is evaluated, discarded and never seen again.
    # Unlike ``engine``, no post-hoc pass can recover them -- re-running the
    # search draws a different population.  Until 2026-08-03 only
    # ``n_conversion_failures`` reached the run log, so four of the five rates
    # were uncheckable rather than merely unchecked, and a probe walking all 69
    # keys of a run_log.json found no reachability field at all.
    #
    # ``None`` on every field = the ledger was absent (a baseline or hash arm,
    # which does not canonicalise).  Zero is a *measurement*; None is not.  The
    # distinction matters because a zero-everywhere ledger means the counters
    # are dead, not that the rates are zero, and only the sampled denominators
    # below can tell those two apart.
    ledger_enabled: bool | None = None
    ledger_sample_rate: int | None = None
    #: Candidates the ledger observed (full rate) and sampled (1 in
    #: ``ledger_sample_rate``).  ``n_ledger_sampled`` is the denominator of the
    #: two violation rates; a count with no denominator is not a rate.
    n_ledger_seen: int | None = None
    n_ledger_sampled: int | None = None
    #: Reachability precondition violated on the DAG as the adapter emitted it.
    n_violations_pre: int | None = None
    #: Still violated after ``normalize_const_creation``.  This is the residual
    #: T07's precondition hypothesis is stated against.
    n_violations_post: int | None = None
    #: Canonicalisation exceeded its wall-clock budget for one candidate.
    n_canon_timeouts: int | None = None
    #: Canonicalisation raised.  Distinct from a timeout: a raise is a defect
    #: signal, a timeout is a budget outcome.
    n_canon_raised: int | None = None
    #: Candidates answered from the precomputed atlas rather than canonicalised
    #: on the fly.  Not a fallback -- recorded here because it partitions the
    #: same stream and the five rates are otherwise not interpretable.
    n_atlas_hits: int | None = None

    # ------------------------------------------------------------------ #
    # Effective-population disclosure (Bingo only)
    # ------------------------------------------------------------------ #
    # Bingo's population-level dedup rejects a duplicate by marking it +inf and
    # setting genetic_age = 10^7 rather than removing it, so the nominal
    # population size overstates the number of individuals actually competing.
    # These two scalars summarise the per-generation count of penalised members
    # and make that shortfall auditable from the run log.  0.0 = does not apply
    # (baseline, UDFS) or no generation was ever observed.
    penalised_in_population_mean: float = 0.0
    penalised_in_population_max: float = 0.0


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
    """One row of aggregate.csv (Section C.7).

    Attributes:
        n: Number of runs aggregated into this row, or ``None`` for a row read
            from a C1-era file that predates the column. Every statistic in the
            row is NaN-aware (``nanmean``/``nanstd``/...), so a run whose value
            for this metric is NaN is counted in ``n`` but does not enter the
            statistics; ``n`` is therefore the number of seeds attempted, not
            the number of finite observations.
    """

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
    n: int | None = None

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
            "n": "" if self.n is None else str(self.n),
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
    "n",
]


# ======================================================================
# Paired statistics
# ======================================================================


@dataclass
class PairedStatsMetric:
    """Statistical comparison for one metric between baseline and IsalSR.

    Attributes:
        n: Number of paired observations that entered this metric's test, after
            pairwise deletion of NaN differences (EXECUTION-PLAN §6.4, "the true
            N reported per metric"). It is at most the enclosing
            :attr:`PairedStats.n_seeds` and may be smaller when a metric is
            undefined for some seeds. ``None`` for a record read from a C1-era
            file written before the field existed, which is distinct from ``0``
            ("no observation survived deletion").
    """

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
    n: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PairedStatsMetric:
        """Rebuild a record, tolerating fields absent from C1-era files.

        Args:
            d: Decoded JSON object. Keys the dataclass does not declare are
                ignored so a file written by a newer schema still loads; ``n``
                stays ``None`` when the file predates it.

        Returns:
            The reconstructed record.
        """
        fields = {f.name for f in dataclass_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclass
class PairedStats:
    """All paired statistical comparisons for one (method, problem) pair.

    Attributes:
        n_seeds: Number of seeds matched between the two arms, i.e. the number
            of pairs the comparison was built from before any per-metric NaN
            deletion. ``None`` for a file written before the field existed
            (C1's ``wl_subtree_unified`` tree), which is distinct from ``0``.
    """

    method: str
    benchmark: str
    problem: str
    metrics: dict[str, PairedStatsMetric] = field(default_factory=dict)
    n_seeds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "benchmark": self.benchmark,
            "problem": self.problem,
            "n_seeds": self.n_seeds,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PairedStats:
        """Rebuild from a decoded JSON object, tolerating a missing ``n_seeds``.

        Args:
            d: Decoded JSON object.

        Returns:
            The reconstructed comparison; ``n_seeds`` is ``None`` when the file
            predates the field.
        """
        return cls(
            method=d["method"],
            benchmark=d["benchmark"],
            problem=d["problem"],
            metrics={k: PairedStatsMetric.from_dict(v) for k, v in d["metrics"].items()},
            n_seeds=d.get("n_seeds"),
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
