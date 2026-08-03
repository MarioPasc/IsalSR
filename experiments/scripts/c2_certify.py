"""Machine-checked go/no-go certification of a Campaign-C2 pre-flight Stage C root.

Stage C is ``{baseline, hash, isalsr} x {UDFS, Bingo} x 70 problems x 3 seeds``
= 1,260 runs. This module walks such an output root and evaluates the seventeen
``C1.x`` criteria plus the ``C2`` and ``C4`` deliverables. Every criterion is
**computed from files on disk**; nothing is asserted.

Two properties are load-bearing and are the reason this is a script rather than
a checklist:

1. **It never raises on missing or malformed data.** A missing file, a corrupt
   JSON, a truncated CSV and an absent directory are all *recorded failures
   naming the path*, never tracebacks. A certifier that dies on the first hole
   certifies nothing, and the holes are exactly what it exists to find.
2. **It is honest about partial roots.** Run it while the array is still
   draining: every criterion reports ``observed`` against ``expected`` rather
   than silently scoping itself to what happens to be present.

The process is read-only with respect to the results root. The only files
written are the two ``--out-*`` paths.

Two deliberate deviations from the criteria as originally drafted, both
documented in the emitted JSON under ``detail["premise_note"]``:

* **C1.1** additionally requires the cell's ``run_log.json`` to exist. A cell
  that reports ``exit_code == 0`` while producing no output has not completed;
  reading the exit code alone would certify that hole as a success.
* **C1.17** ``aggregate.csv`` is written by ``aggregate_all_metrics``, which
  emits one row **per metric** (14 of them), not one row per seed. The check
  therefore asserts ``len(METRIC_EXTRACTORS)`` rows and the ``AGGREGATE_COLUMNS``
  header; the seed count of a cell is checked by C1.16 instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard

log = logging.getLogger("c2_certify")

# ====================================================================== #
# Campaign shape
# ====================================================================== #

#: The two host search methods.
METHODS: tuple[str, ...] = ("udfs", "bingo")

#: The three arms of the paired design.
ARMS: tuple[str, ...] = ("baseline", "hash", "isalsr")

#: Arms that deduplicate, i.e. that must carry a populated fallback ledger.
DEDUP_ARMS: tuple[str, ...] = ("hash", "isalsr")

#: The arm that canonicalises. Only this one produces a reduction factor.
CANONICAL_ARM: str = "isalsr"

#: Stage C seeds. Rendered on disk by ``io_utils.seed_dir`` as ``seed_{s:02d}``.
DEFAULT_SEEDS: tuple[int, ...] = (0, 101, 102)

#: The three paired contrasts the orchestrator emits per (method, problem).
CONTRAST_FILES: tuple[tuple[str, str, str], ...] = (
    ("baseline", "isalsr", "paired_stats.json"),
    ("baseline", "hash", "paired_stats_hash_vs_baseline.json"),
    ("hash", "isalsr", "paired_stats_isalsr_vs_hash.json"),
)

#: Number of rows ``aggregate_all_metrics`` writes into ``aggregate.csv``.
#: Resolved at runtime from ``METRIC_EXTRACTORS``; this is the fallback.
FALLBACK_N_AGGREGATE_ROWS: int = 14

# ====================================================================== #
# run_log.json field specification (C1.2)
# ====================================================================== #
#
# Enumerated explicitly rather than reflected off the dataclasses: the point of
# C1.2 is to detect a *schema regression*, and a spec derived from the schema
# cannot detect one. ``(json_path, allowed_types, nullable)``.

_NUM = (int, float)

RUN_LOG_FIELD_SPEC: tuple[tuple[tuple[str, ...], tuple[type, ...], bool], ...] = (
    # --- metadata ---------------------------------------------------- #
    (("metadata", "method"), (str,), False),
    (("metadata", "representation"), (str,), False),
    (("metadata", "benchmark"), (str,), False),
    (("metadata", "problem"), (str,), False),
    (("metadata", "seed"), (int,), False),
    (("metadata", "hardware"), (dict,), False),
    (("metadata", "hyperparameters"), (dict,), False),
    (("metadata", "data_fingerprint"), (str,), False),
    (("metadata", "config_sha256"), (str,), False),
    # --- results.regression ------------------------------------------- #
    (("results", "regression", "r2_train"), _NUM, False),
    (("results", "regression", "r2_test"), _NUM, False),
    (("results", "regression", "nrmse_train"), _NUM, False),
    (("results", "regression", "nrmse_test"), _NUM, False),
    (("results", "regression", "mse_test"), _NUM, False),
    (("results", "regression", "solution_recovered"), (bool,), True),
    (("results", "regression", "jaccard_index"), _NUM, True),
    (("results", "regression", "model_complexity"), (int,), False),
    (("results", "regression", "n_nonfinite_test_predictions"), (int,), False),
    # --- results.time -------------------------------------------------- #
    (("results", "time", "wall_clock_total_s"), _NUM, False),
    (("results", "time", "wall_clock_search_only_s"), _NUM, False),
    (("results", "time", "canonicalization_precomputed_s"), _NUM, False),
    (("results", "time", "canonicalization_runtime_s"), _NUM, False),
    (("results", "time", "cache_hit_rate"), _NUM, False),
    (("results", "time", "cache_hits"), (int,), False),
    (("results", "time", "cache_misses"), (int,), False),
    (("results", "time", "estimated_time_saved_s"), _NUM, False),
    (("results", "time", "time_to_r2_099_s"), _NUM, True),
    (("results", "time", "time_to_r2_0999_s"), _NUM, True),
    (("results", "time", "evaluation_time_s"), _NUM, False),
    (("results", "time", "overhead_time_s"), _NUM, False),
    # --- results.search_space (core) ------------------------------------ #
    (("results", "search_space", "total_dags_explored"), (int,), False),
    (("results", "search_space", "unique_canonical_dags"), (int,), False),
    (("results", "search_space", "empirical_reduction_factor"), _NUM, False),
    (("results", "search_space", "max_internal_nodes_seen"), (int,), False),
    (("results", "search_space", "theoretical_reduction_bound"), _NUM, False),
    (("results", "search_space", "redundancy_rate"), _NUM, False),
    # --- results.search_space (shadow sketches) -------------------------- #
    (("results", "search_space", "shadow_distinct_insertion"), _NUM, True),
    (("results", "search_space", "shadow_distinct_topological"), _NUM, True),
    (("results", "search_space", "shadow_distinct_topological_commutative"), _NUM, True),
    (("results", "search_space", "shadow_distinct_host_native"), _NUM, True),
    (("results", "search_space", "n_shadow_failures"), (int,), True),
    # --- results.search_space (the ten ledger fields) -------------------- #
    (("results", "search_space", "n_conversion_failures"), (int,), True),
    (("results", "search_space", "ledger_enabled"), (bool,), True),
    (("results", "search_space", "ledger_sample_rate"), (int,), True),
    (("results", "search_space", "n_ledger_seen"), (int,), True),
    (("results", "search_space", "n_ledger_sampled"), (int,), True),
    (("results", "search_space", "n_violations_pre"), (int,), True),
    (("results", "search_space", "n_violations_post"), (int,), True),
    (("results", "search_space", "n_canon_timeouts"), (int,), True),
    (("results", "search_space", "n_canon_raised"), (int,), True),
    (("results", "search_space", "n_atlas_hits"), (int,), True),
    # --- best_expression ------------------------------------------------- #
    (("best_expression", "symbolic_form"), (str,), False),
    (("best_expression", "isalsr_string"), (str,), False),
    (("best_expression", "canonical_string"), (str,), False),
    (("best_expression", "n_nodes"), (int,), False),
    (("best_expression", "n_edges"), (int,), False),
)

#: Fields that are typed ``str`` with a ``""`` default but whose emptiness is a
#: provenance failure, not a legitimate value (C1.2).
REQUIRED_NONEMPTY_FIELDS: tuple[tuple[str, ...], ...] = (
    ("metadata", "data_fingerprint"),
    ("metadata", "config_sha256"),
)

#: The ten ledger fields. ``None`` on all ten = the arm never asked (C1.8);
#: populated = the arm asked (C1.9). Zero is a measurement, ``None`` is not.
LEDGER_FIELDS: tuple[str, ...] = (
    "ledger_enabled",
    "ledger_sample_rate",
    "n_ledger_seen",
    "n_ledger_sampled",
    "n_violations_pre",
    "n_violations_post",
    "n_canon_timeouts",
    "n_canon_raised",
    "n_atlas_hits",
    "n_conversion_failures",
)

#: The five paths on which a candidate is evaluated WITHOUT canonical dedup.
FALLBACK_PATH_FIELDS: tuple[str, ...] = (
    "n_violations_pre",
    "n_violations_post",
    "n_canon_timeouts",
    "n_conversion_failures",
    "n_canon_raised",
)

#: Regression metrics that must be finite everywhere (C1.3).
REGRESSION_FINITE_FIELDS: tuple[str, ...] = (
    "r2_train",
    "r2_test",
    "nrmse_train",
    "nrmse_test",
    "mse_test",
)

# ====================================================================== #
# Alphabet (C1.13)
# ====================================================================== #
#
# T16: the alphabet is decomposed. There is no Sub and no Div; Pow is the only
# non-commutative operation and is permitted for Bingo only (UDFS's vendored
# search has no generic pow). A '-' or a '/' anywhere in a canonical or IsalSR
# string is a hard failure: neither character is in Sigma_SR.

FORBIDDEN_CHARS: tuple[str, ...] = ("-", "/")
FORBIDDEN_TOKENS: tuple[str, ...] = ("V-", "V/", "v-", "v/")
POW_TOKENS: tuple[str, ...] = ("V^", "v^")

#: Methods whose alphabet legitimately contains ``Pow``.
POW_ALLOWED_METHODS: tuple[str, ...] = ("bingo",)

# ====================================================================== #
# Thresholds
# ====================================================================== #

#: C1.6: fraction of isalsr runs required to show rho > 1.
MIN_FRACTION_RHO_GT_ONE: float = 0.90

#: C1.7: rho_hash <= rho_isalsr violation rate above which the check blocks.
MAX_RHO_ORDER_VIOLATION_RATE: float = 0.05

#: C1.11: production ``--mem`` recommendation = p99 MaxRSS x this.
MEM_HEADROOM_FACTOR: float = 1.5

#: C1.12: wall-clock allowance above ``--max-time``. The search budget bounds
#: the *search*; constant optimisation, the SymPy equivalence check and process
#: start-up sit outside it.
DEFAULT_WALL_SLACK_S: float = 600.0


# ====================================================================== #
# Result records
# ====================================================================== #


@dataclass
class CriterionResult:
    """Verdict for one certification criterion.

    Attributes:
        id: Criterion identifier, e.g. ``"C1.6"``.
        title: One-line human-readable statement of what was checked.
        status: ``"PASS"``, ``"FAIL"`` or ``"SKIP"``.
        expected: The value the criterion demands.
        observed: The value computed from disk.
        detail: Everything needed to act on a failure, including the named
            offending cells.
        blocking: Whether a ``FAIL`` blocks the campaign go decision.
    """

    id: str
    title: str
    status: str
    expected: Any
    observed: Any
    detail: dict[str, Any] = field(default_factory=dict)
    blocking: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of this verdict."""
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "expected": self.expected,
            "observed": self.observed,
            "blocking": self.blocking,
            "detail": self.detail,
        }


@dataclass
class Cell:
    """One ``(method, arm, benchmark, problem, seed)`` run directory.

    Every field that could fail to load is nullable and carries a sibling error
    string. Nothing here raises.
    """

    method: str
    arm: str
    benchmark: str
    problem: str
    seed: int
    directory: Path
    status: Any | None = None
    status_error: str = ""
    run_log_raw: dict[str, Any] | None = None
    run_log_error: str = ""

    @property
    def key(self) -> tuple[str, str, str, int]:
        """Return the reconcile key ``(method, arm, problem, seed)``."""
        return (self.method, self.arm, self.problem, self.seed)

    @property
    def label(self) -> str:
        """Return a compact human-readable cell name for failure lists."""
        return f"{self.method}/{self.arm}/{self.benchmark}/{self.problem}/seed_{self.seed}"

    @property
    def run_log_path(self) -> Path:
        """Return the expected ``run_log.json`` path."""
        return self.directory / "run_log.json"

    @property
    def trajectory_path(self) -> Path:
        """Return the expected ``trajectory.csv`` path."""
        return self.directory / "trajectory.csv"

    def section(self, *names: str) -> dict[str, Any]:
        """Return a nested mapping from the raw run log, or ``{}`` if absent."""
        node: Any = self.run_log_raw
        for name in names:
            if not isinstance(node, dict):
                return {}
            node = node.get(name)
        return node if isinstance(node, dict) else {}

    @property
    def regression(self) -> dict[str, Any]:
        """Return the ``results.regression`` mapping."""
        return self.section("results", "regression")

    @property
    def time(self) -> dict[str, Any]:
        """Return the ``results.time`` mapping."""
        return self.section("results", "time")

    @property
    def search_space(self) -> dict[str, Any]:
        """Return the ``results.search_space`` mapping."""
        return self.section("results", "search_space")

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the ``metadata`` mapping."""
        return self.section("metadata")

    @property
    def best_expression(self) -> dict[str, Any]:
        """Return the ``best_expression`` mapping."""
        return self.section("best_expression")


# ====================================================================== #
# Small utilities
# ====================================================================== #


def _is_number(value: Any) -> TypeGuard[float]:
    """Return whether ``value`` is a non-boolean int or float."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite(value: Any) -> TypeGuard[float]:
    """Return whether ``value`` is a finite (non-NaN, non-inf) number."""
    return _is_number(value) and math.isfinite(float(value))


def _percentile(values: list[float], q: float) -> float | None:
    """Return the nearest-rank percentile of ``values``.

    Args:
        values: Sample, need not be sorted. Empty returns ``None``.
        q: Percentile in [0, 100].

    Returns:
        The nearest-rank percentile, or ``None`` for an empty sample.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


def _summarise(values: list[float]) -> dict[str, Any]:
    """Return n / min / p50 / p90 / p99 / max for a numeric sample."""
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": round(min(values), 6),
        "p50": round(_percentile(values, 50) or 0.0, 6),
        "p90": round(_percentile(values, 90) or 0.0, 6),
        "p99": round(_percentile(values, 99) or 0.0, 6),
        "max": round(max(values), 6),
        "mean": round(sum(values) / len(values), 6),
    }


def _truncate(items: list[Any], limit: int = 50) -> dict[str, Any]:
    """Return a count plus at most ``limit`` named examples."""
    return {
        "count": len(items),
        "examples": [str(i) for i in items[:limit]],
        "truncated": max(0, len(items) - limit),
    }


def _verdict(ok: bool) -> str:
    """Map a boolean to the verdict vocabulary."""
    return "PASS" if ok else "FAIL"


def _parse_maxrss_to_gb(raw: str) -> float | None:
    """Convert an ``sacct`` MaxRSS string to gigabytes.

    ``sacct`` emits values suffixed ``K``/``M``/``G``/``T`` (kibi/mebi/gibi/tebi)
    and an empty string when the field was never sampled, which is exactly what
    ``sacct -X`` returns.

    Args:
        raw: The raw MaxRSS field.

    Returns:
        Gigabytes, or ``None`` when the field is empty or unparseable.
    """
    text = raw.strip()
    if not text:
        return None
    scale = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}
    factor = 1 / 1024**3  # bare bytes
    if text[-1].upper() in scale:
        factor = scale[text[-1].upper()]
        text = text[:-1]
    try:
        return float(text) * factor
    except ValueError:
        return None


# ====================================================================== #
# Discovery
# ====================================================================== #


def _load_registry() -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Load the orchestrator's benchmark registry.

    Returns:
        A ``(registry, error)`` pair. On import failure the registry is empty
        and ``error`` names the exception; the caller degrades to disk-derived
        expectations rather than raising.
    """
    try:
        from experiments.models.orchestrator import _BENCHMARK_REGISTRY

        return {name: list(entry[0]) for name, entry in _BENCHMARK_REGISTRY.items()}, ""
    except Exception as exc:  # noqa: BLE001 - a broken import is a finding
        return {}, f"{type(exc).__name__}: {exc}"


def _slug(problem: str) -> str:
    """Return the on-disk directory slug for a problem name."""
    return problem.lower().replace("-", "_")


def _parse_seed_dir(name: str) -> int | None:
    """Extract the seed from a ``seed_NN`` directory name, or ``None``."""
    if not name.startswith("seed_"):
        return None
    try:
        return int(name[5:])
    except ValueError:
        return None


def discover_cells(root: Path, registry: dict[str, list[dict[str, Any]]]) -> list[Cell]:
    """Walk the results root and build one :class:`Cell` per seed directory.

    Args:
        root: Campaign or smoke root.
        registry: Benchmark registry, used to map directory slugs back to
            problem names. Unknown slugs keep the slug as the problem name.

    Returns:
        Every seed directory found, in sorted order. Never raises.
    """
    slug_to_name = {
        _slug(bench["name"]): bench["name"]
        for problems in registry.values()
        for bench in problems
        if "name" in bench
    }
    cells: list[Cell] = []
    if not root.is_dir():
        return cells

    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for bench_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            for problem_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
                problem = slug_to_name.get(problem_dir.name, problem_dir.name)
                for arm_dir in sorted(p for p in problem_dir.iterdir() if p.is_dir()):
                    if arm_dir.name not in ARMS:
                        continue
                    for sd in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
                        seed = _parse_seed_dir(sd.name)
                        if seed is None:
                            continue
                        cells.append(
                            Cell(
                                method=method_dir.name,
                                arm=arm_dir.name,
                                benchmark=bench_dir.name,
                                problem=problem,
                                seed=seed,
                                directory=sd,
                            )
                        )
    return cells


def hydrate(cell: Cell) -> None:
    """Load ``status.json`` and ``run_log.json`` into a cell, recording errors.

    Args:
        cell: The cell to populate in place.
    """
    from experiments.models.status_ledger import load_status

    status_path = cell.directory / "status.json"
    if status_path.exists():
        cell.status = load_status(status_path)
        if cell.status is None:
            cell.status_error = f"unreadable: {status_path}"
    else:
        cell.status_error = f"missing: {status_path}"

    if not cell.run_log_path.exists():
        cell.run_log_error = f"missing: {cell.run_log_path}"
        return
    try:
        payload = json.loads(cell.run_log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        cell.run_log_error = f"{type(exc).__name__}: {cell.run_log_path}"
        return
    if not isinstance(payload, dict):
        cell.run_log_error = f"not a JSON object: {cell.run_log_path}"
        return
    cell.run_log_raw = payload


def build_expected_cells(
    registry: dict[str, list[dict[str, Any]]],
    observed: list[Cell],
    seeds: tuple[int, ...],
    expected_tasks: int,
) -> tuple[set[tuple[str, str, str, int]], dict[tuple[str, str, str, int], str], str]:
    """Determine the cell universe the root is certified against.

    The full campaign universe is ``methods x arms x registry problems x seeds``.
    When that cardinality equals ``--expected-tasks`` it is used verbatim, so a
    whole absent problem directory is still *named* as a gap. Otherwise the
    universe is derived from the observed ``(method, benchmark, problem, seed)``
    tuples crossed with all three arms, which is what makes the certifier usable
    on a partial root or a smoke fixture.

    Args:
        registry: Benchmark registry.
        observed: Cells found on disk.
        seeds: Campaign seeds.
        expected_tasks: The ``--expected-tasks`` argument.

    Returns:
        ``(expected_keys, key_to_benchmark, source)`` where ``source`` is
        ``"registry"`` or ``"disk"``.
    """
    canonical: set[tuple[str, str, str, int]] = set()
    canonical_bench: dict[tuple[str, str, str, int], str] = {}
    for bench_name, problems in registry.items():
        for bench in problems:
            name = bench.get("name")
            if not name:
                continue
            for method in METHODS:
                for arm in ARMS:
                    for seed in seeds:
                        key = (method, arm, name, seed)
                        canonical.add(key)
                        canonical_bench[key] = bench_name

    if canonical and len(canonical) == expected_tasks:
        return canonical, canonical_bench, "registry"

    disk: set[tuple[str, str, str, int]] = set()
    disk_bench: dict[tuple[str, str, str, int], str] = {}
    for cell in observed:
        for arm in ARMS:
            key = (cell.method, arm, cell.problem, cell.seed)
            disk.add(key)
            disk_bench[key] = cell.benchmark
    return disk, disk_bench, "disk"


# ====================================================================== #
# Criteria
# ====================================================================== #


def check_c1_1(
    by_key: dict[tuple[str, str, str, int], Cell],
    expected: set[tuple[str, str, str, int]],
) -> CriterionResult:
    """C1.1 -- every expected cell exited 0 and left a run log."""
    absent: list[str] = []
    no_status: list[str] = []
    bad_exit: list[str] = []
    not_completed: list[str] = []
    no_run_log: list[str] = []

    for key in sorted(expected):
        cell = by_key.get(key)
        name = "/".join(str(p) for p in key)
        if cell is None:
            absent.append(name)
            continue
        if cell.status is None:
            no_status.append(f"{cell.label} ({cell.status_error})")
        else:
            if cell.status.exit_code != 0:
                bad_exit.append(f"{cell.label} exit_code={cell.status.exit_code}")
            if cell.status.terminal_status != "completed":
                not_completed.append(f"{cell.label} status={cell.status.terminal_status}")
        if cell.run_log_raw is None:
            no_run_log.append(f"{cell.label} ({cell.run_log_error})")

    n_bad = len(absent) + len(no_status) + len(bad_exit) + len(not_completed) + len(no_run_log)
    n_ok = len(expected) - len({*absent, *no_status, *bad_exit, *not_completed, *no_run_log})
    return CriterionResult(
        id="C1.1",
        title="every expected cell exited 0 and produced a run log",
        status=_verdict(n_bad == 0 and len(expected) > 0),
        expected=f"{len(expected)}/{len(expected)} clean",
        observed=f"{max(0, n_ok)}/{len(expected)} clean",
        detail={
            "premise_note": (
                "run_log.json presence is folded into C1.1: a cell reporting "
                "exit_code == 0 with no output has not completed."
            ),
            "no_directory": _truncate(absent),
            "no_status_json": _truncate(no_status),
            "nonzero_exit": _truncate(bad_exit),
            "not_completed": _truncate(not_completed),
            "no_run_log": _truncate(no_run_log),
        },
    )


def _walk_spec(payload: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    """Resolve a dotted path in a run-log payload.

    Returns:
        ``(present, value)``.
    """
    node: Any = payload
    for name in path:
        if not isinstance(node, dict) or name not in node:
            return False, None
        node = node[name]
    return True, node


def check_c1_2(cells: list[Cell]) -> CriterionResult:
    """C1.2 -- every run log parses and carries every field at the right type."""
    missing: Counter[str] = Counter()
    wrong_type: Counter[str] = Counter()
    empty: Counter[str] = Counter()
    unparseable: list[str] = []
    parse_failures: list[str] = []
    n_clean = 0

    from experiments.models.io_utils import load_run_log

    for cell in cells:
        if cell.run_log_raw is None:
            unparseable.append(f"{cell.label} ({cell.run_log_error})")
            continue
        try:
            load_run_log(cell.run_log_path)
        except Exception as exc:  # noqa: BLE001 - a schema mismatch is a finding
            parse_failures.append(f"{cell.label} ({type(exc).__name__}: {exc})")

        cell_ok = True
        for path, types, nullable in RUN_LOG_FIELD_SPEC:
            dotted = ".".join(path)
            present, value = _walk_spec(cell.run_log_raw, path)
            if not present:
                missing[dotted] += 1
                cell_ok = False
                continue
            if value is None:
                if not nullable:
                    wrong_type[dotted] += 1
                    cell_ok = False
                continue
            if types == (int,) and isinstance(value, bool):
                wrong_type[dotted] += 1
                cell_ok = False
                continue
            if types is _NUM and isinstance(value, bool):
                wrong_type[dotted] += 1
                cell_ok = False
                continue
            if not isinstance(value, types):
                wrong_type[dotted] += 1
                cell_ok = False
        for path in REQUIRED_NONEMPTY_FIELDS:
            present, value = _walk_spec(cell.run_log_raw, path)
            if present and isinstance(value, str) and not value:
                empty[".".join(path)] += 1
                cell_ok = False
        if cell_ok:
            n_clean += 1

    ok = not missing and not wrong_type and not empty and not unparseable and not parse_failures
    return CriterionResult(
        id="C1.2",
        title=f"run_log.json parses; all {len(RUN_LOG_FIELD_SPEC)} fields present and typed",
        status=_verdict(ok and bool(cells)),
        expected=f"{len(cells)}/{len(cells)} complete",
        observed=f"{n_clean}/{len(cells)} complete",
        detail={
            "n_fields_checked": len(RUN_LOG_FIELD_SPEC),
            "missing_by_field": dict(missing),
            "wrong_type_by_field": dict(wrong_type),
            "empty_required_string_by_field": dict(empty),
            "unreadable_run_logs": _truncate(unparseable),
            "schema_load_failures": _truncate(parse_failures),
        },
    )


def check_c1_3(cells: list[Cell]) -> CriterionResult:
    """C1.3 -- no NaN and no inf in any regression metric."""
    offenders: list[str] = []
    nonfinite_pred: list[int] = []
    n_checked = 0
    for cell in cells:
        reg = cell.regression
        if not reg:
            continue
        n_checked += 1
        for name in REGRESSION_FINITE_FIELDS:
            value = reg.get(name)
            if not _is_finite(value):
                offenders.append(f"{cell.label} {name}={value!r}")
        raw = reg.get("n_nonfinite_test_predictions")
        if isinstance(raw, int) and not isinstance(raw, bool):
            nonfinite_pred.append(raw)

    n_nonzero = sum(1 for v in nonfinite_pred if v > 0)
    return CriterionResult(
        id="C1.3",
        title="no NaN/inf in any regression metric",
        status=_verdict(not offenders and n_checked > 0),
        expected="0 non-finite metric values",
        observed=f"{len(offenders)} non-finite across {n_checked} run logs",
        detail={
            "non_finite_metrics": _truncate(offenders),
            "n_nonfinite_test_predictions": {
                "note": (
                    "a non-zero count is legitimate: the expression is undefined "
                    "on part of the test domain and the run is scored R2=0 "
                    "rather than emitting NaN. Reported, not failed."
                ),
                "n_runs_with_nonzero": n_nonzero,
                "n_runs_reporting": len(nonfinite_pred),
                "distribution": _summarise([float(v) for v in nonfinite_pred]),
                "histogram": dict(Counter(nonfinite_pred).most_common(20)),
            },
        },
    )


def check_c1_4(root: Path, cells: list[Cell], registry: dict[str, Any]) -> CriterionResult:
    """C1.4 -- train/test shapes match what the benchmark registry generates.

    The expected shapes are obtained by calling the registry's own
    ``generate_data`` for one seed. A problem whose realised shapes differ from
    the configured ``train_size``/``test_size`` is only flagged when its bench
    dict carries **no** ``sampling`` override -- Vlad-7 (300/1200), Keijzer-6
    (50/120) and Pagie-1 (676/2500) declare theirs and are correct.
    """
    detail: dict[str, Any] = {}
    metadata_path = root / "metadata.json"
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        bench_cfg = meta.get("config", {}).get("benchmarks", {})
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        return CriterionResult(
            id="C1.4",
            title="train/test shapes match the benchmark registry",
            status="FAIL",
            expected="metadata.json with config.benchmarks",
            observed=f"{type(exc).__name__}: {metadata_path}",
            detail={"error": str(exc)},
        )

    try:
        from experiments.models.orchestrator import _generate_benchmark_data
    except Exception as exc:  # noqa: BLE001
        return CriterionResult(
            id="C1.4",
            title="train/test shapes match the benchmark registry",
            status="FAIL",
            expected="importable orchestrator._generate_benchmark_data",
            observed=f"{type(exc).__name__}: {exc}",
            detail={},
        )

    wanted = sorted({(c.benchmark, c.problem) for c in cells})
    by_name = {(bench_name, b["name"]): b for bench_name, probs in registry.items() for b in probs}
    shapes: dict[str, list[int]] = {}
    mismatches: list[str] = []
    errors: list[str] = []
    explained: list[str] = []

    for bench_name, problem in wanted:
        bench = by_name.get((bench_name, problem))
        if bench is None:
            errors.append(f"{bench_name}/{problem}: not in registry")
            continue
        cfg = bench_cfg.get(bench_name, {})
        train_size = int(cfg.get("train_size", 20))
        test_size = int(cfg.get("test_size", 100))
        seed = min((c.seed for c in cells if c.problem == problem), default=0)
        try:
            x_tr, y_tr, x_te, y_te = _generate_benchmark_data(
                bench_name, bench, train_size, test_size, seed
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{bench_name}/{problem}: {type(exc).__name__}: {exc}")
            continue
        n_tr, n_te = int(x_tr.shape[0]), int(x_te.shape[0])
        shapes[f"{bench_name}/{problem}"] = [n_tr, n_te, int(x_tr.shape[1])]
        if len(y_tr) != n_tr or len(y_te) != n_te:
            mismatches.append(f"{bench_name}/{problem}: X/y length disagreement")
            continue
        if (n_tr, n_te) != (train_size, test_size):
            note = f"{bench_name}/{problem}: {n_tr}/{n_te} vs config {train_size}/{test_size}"
            if bench.get("sampling"):
                explained.append(f"{note} [sampling={bench['sampling'].get('type', '?')}]")
            else:
                mismatches.append(f"{note} [no sampling override]")

    return CriterionResult(
        id="C1.4",
        title="train/test shapes match the benchmark registry",
        status=_verdict(not mismatches and not errors and bool(shapes)),
        expected=f"{len(wanted)} problems with reproducible shapes",
        observed=f"{len(shapes)} resolved, {len(mismatches)} unexplained, {len(errors)} errors",
        detail={
            "shapes_n_train_n_test_n_features": shapes,
            "unexplained_mismatches": _truncate(mismatches),
            "explained_by_sampling_override": _truncate(explained),
            "generation_errors": _truncate(errors),
            **detail,
        },
    )


def check_c1_5(registry: dict[str, Any], cells: list[Cell]) -> CriterionResult:
    """C1.5 -- a SymPy ground truth exists for every problem, so recovery is computable."""
    try:
        from experiments.models.orchestrator import _get_ground_truth_sympy
    except Exception as exc:  # noqa: BLE001
        return CriterionResult(
            id="C1.5",
            title="solution_recovered computable for every problem",
            status="FAIL",
            expected="importable orchestrator._get_ground_truth_sympy",
            observed=f"{type(exc).__name__}: {exc}",
            detail={},
        )

    observed_problems = {c.problem for c in cells}
    without: list[str] = []
    n_total = 0
    for bench_name, problems in registry.items():
        for bench in problems:
            n_total += 1
            try:
                expr = _get_ground_truth_sympy(bench)
            except Exception as exc:  # noqa: BLE001
                without.append(f"{bench_name}/{bench.get('name')}: {type(exc).__name__}")
                continue
            if expr is None:
                mark = " [present in root]" if bench.get("name") in observed_problems else ""
                without.append(f"{bench_name}/{bench.get('name')}: None{mark}")

    return CriterionResult(
        id="C1.5",
        title="solution_recovered computable (SymPy ground truth) for every problem",
        status=_verdict(not without and n_total > 0),
        expected=f"{n_total}/{n_total} problems with a ground truth",
        observed=f"{n_total - len(without)}/{n_total}",
        detail={"problems_without_ground_truth": _truncate(without, limit=80)},
    )


def check_c1_6(cells: list[Cell]) -> CriterionResult:
    """C1.6 -- the isalsr arm produced a live, arithmetically sane reduction factor.

    Two named diagnoses: ``rho < 1`` is arithmetically impossible (rho = total /
    unique with unique <= total), so it means a broken counter; ``rho == 1.0``
    on every run means the dedup hook is dead, not that there was no redundancy.
    """
    arm = [c for c in cells if c.arm == CANONICAL_ARM and c.search_space]
    zero_unique: list[str] = []
    rho_below_one: list[str] = []
    rho_missing: list[str] = []
    rhos: list[float] = []

    for cell in arm:
        ss = cell.search_space
        unique = ss.get("unique_canonical_dags")
        rho = ss.get("empirical_reduction_factor")
        if not isinstance(unique, int) or isinstance(unique, bool) or unique <= 0:
            zero_unique.append(f"{cell.label} unique_canonical_dags={unique!r}")
        if not _is_finite(rho):
            rho_missing.append(f"{cell.label} rho={rho!r}")
            continue
        rhos.append(float(rho))
        if float(rho) < 1.0:
            rho_below_one.append(f"{cell.label} rho={rho}")

    n_gt_one = sum(1 for r in rhos if r > 1.0)
    frac_gt_one = n_gt_one / len(rhos) if rhos else 0.0
    all_exactly_one = bool(rhos) and all(r == 1.0 for r in rhos)

    diagnoses: list[str] = []
    if rho_below_one:
        diagnoses.append(
            "BROKEN COUNTER: rho < 1 is arithmetically impossible "
            "(rho = total/unique, unique <= total)"
        )
    if all_exactly_one:
        diagnoses.append("DEAD DEDUP HOOK: rho == 1.0 on every isalsr run")

    ok = (
        bool(arm)
        and not zero_unique
        and not rho_below_one
        and not rho_missing
        and frac_gt_one >= MIN_FRACTION_RHO_GT_ONE
    )
    return CriterionResult(
        id="C1.6",
        title="isalsr: unique_canonical_dags > 0, rho >= 1 everywhere, rho > 1 on >= 90%",
        status=_verdict(ok),
        expected=(
            f"{len(arm)}/{len(arm)} with rho >= 1; >= {MIN_FRACTION_RHO_GT_ONE:.0%} with rho > 1"
        ),
        observed=(
            f"{len(arm) - len(rho_below_one) - len(rho_missing)}/{len(arm)} with rho >= 1; "
            f"{frac_gt_one:.1%} with rho > 1"
        ),
        detail={
            "diagnoses": diagnoses,
            "n_isalsr_runs": len(arm),
            "zero_or_missing_unique_canonical_dags": _truncate(zero_unique),
            "rho_below_one": _truncate(rho_below_one),
            "rho_missing_or_nonfinite": _truncate(rho_missing),
            "rho_distribution": _summarise(rhos),
        },
    )


def check_c1_7(cells: list[Cell]) -> CriterionResult:
    """C1.7 -- rho_hash <= rho_isalsr on matched (method, problem, seed) triples.

    The canonical key is coarser than the fixed-order hash key by construction,
    so the canonical arm cannot report fewer collisions. A violation means the
    two arms did not see the same candidate stream.
    """
    rho: dict[tuple[str, str, str, int], float] = {}
    for cell in cells:
        value = cell.search_space.get("empirical_reduction_factor")
        if _is_finite(value):
            rho[(cell.method, cell.arm, cell.problem, cell.seed)] = float(value)

    matched = 0
    violations: list[str] = []
    for (method, arm, problem, seed), value in sorted(rho.items()):
        if arm != "hash":
            continue
        other = rho.get((method, CANONICAL_ARM, problem, seed))
        if other is None:
            continue
        matched += 1
        if value > other:
            violations.append(
                f"{method}/{problem}/seed_{seed} rho_hash={value:.6f} > rho_isalsr={other:.6f}"
            )

    rate = len(violations) / matched if matched else 0.0
    return CriterionResult(
        id="C1.7",
        title="rho_hash <= rho_isalsr on matched (method, problem, seed)",
        status=_verdict(matched > 0 and rate <= MAX_RHO_ORDER_VIOLATION_RATE),
        expected=f"violation rate <= {MAX_RHO_ORDER_VIOLATION_RATE:.0%}",
        observed=f"{len(violations)}/{matched} = {rate:.2%}",
        detail={
            "n_matched_triples": matched,
            "violation_rate": round(rate, 6),
            "violations": _truncate(violations),
        },
    )


def check_c1_8(cells: list[Cell]) -> CriterionResult:
    """C1.8 -- the baseline arm never asked: ten ledger fields None, canon runtime 0."""
    arm = [c for c in cells if c.arm == "baseline" and c.run_log_raw is not None]
    non_null: list[str] = []
    nonzero_canon: list[str] = []
    for cell in arm:
        ss = cell.search_space
        populated = [f for f in LEDGER_FIELDS if ss.get(f) is not None]
        if populated:
            non_null.append(f"{cell.label} populated={populated}")
        runtime = cell.time.get("canonicalization_runtime_s")
        if not (_is_number(runtime) and float(runtime) == 0.0):
            nonzero_canon.append(f"{cell.label} canonicalization_runtime_s={runtime!r}")

    n_clean = len(arm) - len({*non_null, *nonzero_canon})
    return CriterionResult(
        id="C1.8",
        title="baseline: all ten ledger fields None and canonicalization_runtime_s == 0",
        status=_verdict(not non_null and not nonzero_canon and bool(arm)),
        expected=f"{len(arm)}/{len(arm)} clean baseline runs",
        observed=f"{max(0, n_clean)}/{len(arm)}",
        detail={
            "ledger_fields_populated_on_baseline": _truncate(non_null),
            "nonzero_canonicalization_runtime": _truncate(nonzero_canon),
        },
    )


def check_c1_9(cells: list[Cell]) -> CriterionResult:
    """C1.9 -- dedup arms carry a live ledger with a non-zero sampled denominator.

    ``n_ledger_sampled == 0`` is a FAILURE, not a pass with a zero rate: it is
    the difference between "asked and none occurred" and "the counters are
    dead", and a count with no denominator is not a rate.
    """
    arm = [c for c in cells if c.arm in DEDUP_ARMS and c.run_log_raw is not None]
    not_enabled: list[str] = []
    zero_sampled: list[str] = []
    missing_paths: list[str] = []
    nonfinite_paths: list[str] = []
    totals: Counter[str] = Counter()
    denominator = 0

    for cell in arm:
        ss = cell.search_space
        if ss.get("ledger_enabled") is not True:
            not_enabled.append(f"{cell.label} ledger_enabled={ss.get('ledger_enabled')!r}")
        sampled = ss.get("n_ledger_sampled")
        if not (isinstance(sampled, int) and not isinstance(sampled, bool) and sampled > 0):
            zero_sampled.append(f"{cell.label} n_ledger_sampled={sampled!r}")
        else:
            denominator += sampled
        for path in FALLBACK_PATH_FIELDS:
            value = ss.get(path)
            if value is None:
                missing_paths.append(f"{cell.label} {path}=None")
            elif not _is_finite(value):
                nonfinite_paths.append(f"{cell.label} {path}={value!r}")
            else:
                totals[path] += int(value)

    rates = {
        path: (round(totals[path] / denominator, 8) if denominator else None)
        for path in FALLBACK_PATH_FIELDS
    }
    ok = bool(arm) and not not_enabled and not zero_sampled and not missing_paths
    ok = ok and not nonfinite_paths
    n_clean = len(arm) - len({*not_enabled, *zero_sampled, *missing_paths, *nonfinite_paths})
    return CriterionResult(
        id="C1.9",
        title="dedup arms: ledger_enabled True, n_ledger_sampled > 0, five paths present",
        status=_verdict(ok),
        expected=f"{len(arm)}/{len(arm)} live ledgers",
        observed=f"{max(0, n_clean)}/{len(arm)}",
        detail={
            "n_dedup_runs": len(arm),
            "ledger_not_enabled": _truncate(not_enabled),
            "n_ledger_sampled_zero_or_missing": _truncate(zero_sampled),
            "fallback_path_None": _truncate(missing_paths),
            "fallback_path_nonfinite": _truncate(nonfinite_paths),
            "pooled_sampled_denominator": denominator,
            "pooled_counts": dict(totals),
            "fallback_rates": rates,
        },
    )


def check_c1_10(cells: list[Cell]) -> CriterionResult:
    """C1.10 -- trajectory.csv is non-empty, monotone and internally consistent."""
    missing: list[str] = []
    empty: list[str] = []
    unreadable: list[str] = []
    non_monotone: list[str] = []
    unique_exceeds: list[str] = []
    n_ok = 0
    monotone_cols = ("timestamp_s", "best_r2", "n_dags_explored")

    for cell in cells:
        path = cell.trajectory_path
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            unreadable.append(f"{path} ({type(exc).__name__})")
            continue
        if not rows:
            empty.append(str(path))
            continue

        cell_ok = True
        previous: dict[str, float] = {}
        for idx, row in enumerate(rows):
            for col in monotone_cols:
                try:
                    value = float(row[col])
                except (KeyError, TypeError, ValueError):
                    non_monotone.append(
                        f"{cell.label} row {idx}: unparseable {col}={row.get(col)!r}"
                    )
                    cell_ok = False
                    continue
                if col in previous and value < previous[col] - 1e-9:
                    non_monotone.append(
                        f"{cell.label} row {idx}: {col} {previous[col]:.6g} -> {value:.6g}"
                    )
                    cell_ok = False
                previous[col] = value
            try:
                if int(float(row["n_unique_canonical"])) > int(float(row["n_dags_explored"])):
                    unique_exceeds.append(
                        f"{cell.label} row {idx}: "
                        f"n_unique_canonical={row['n_unique_canonical']} > "
                        f"n_dags_explored={row['n_dags_explored']}"
                    )
                    cell_ok = False
            except (KeyError, TypeError, ValueError):
                unique_exceeds.append(f"{cell.label} row {idx}: unparseable counters")
                cell_ok = False
        if cell_ok:
            n_ok += 1

    ok = not missing and not empty and not unreadable and not non_monotone and not unique_exceeds
    return CriterionResult(
        id="C1.10",
        title="trajectory.csv non-empty, monotone, n_unique_canonical <= n_dags_explored",
        status=_verdict(ok and bool(cells)),
        expected=f"{len(cells)}/{len(cells)} valid trajectories",
        observed=f"{n_ok}/{len(cells)}",
        detail={
            "missing": _truncate(missing),
            "empty": _truncate(empty),
            "unreadable": _truncate(unreadable),
            "monotonicity_violations": _truncate(non_monotone),
            "unique_exceeds_explored": _truncate(unique_exceeds),
        },
    )


def check_c1_11(cells: list[Cell], sacct_csv: Path | None) -> CriterionResult:
    """C1.11 -- memory profile and the production ``--mem`` recommendation.

    ``sacct -X`` returns MaxRSS empty, so the ``--sacct-csv`` file may legitimately
    be absent; the check then falls back to ``status.json``'s ``max_rss_gb`` and
    says which source it used. Advisory: it sizes the request, it does not block.
    """
    source = "status.json:max_rss_gb"
    by_arm: dict[str, list[float]] = defaultdict(list)
    unmatched = 0
    parse_errors: list[str] = []

    if sacct_csv is not None and sacct_csv.exists():
        source = f"sacct:{sacct_csv}"
        index: dict[str, Cell] = {}
        for cell in cells:
            st = cell.status
            if st is None:
                continue
            if st.slurm_job_id and st.slurm_array_task_id:
                index[f"{st.slurm_job_id}_{st.slurm_array_task_id}"] = cell
            if st.slurm_job_id:
                index.setdefault(st.slurm_job_id, cell)
        try:
            with sacct_csv.open(newline="", encoding="utf-8") as handle:
                for raw in csv.DictReader(handle):
                    job = str(raw.get("JobID", "")).strip()
                    gb = _parse_maxrss_to_gb(str(raw.get("MaxRSS", "")))
                    if gb is None:
                        continue
                    stem = job.split(".")[0]
                    matched = index.get(stem)
                    if matched is None:
                        unmatched += 1
                        by_arm["unmatched"].append(gb)
                        continue
                    by_arm[f"{matched.method}/{matched.arm}"].append(gb)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            parse_errors.append(f"{sacct_csv}: {type(exc).__name__}: {exc}")
            source = "status.json:max_rss_gb (sacct unreadable)"

    if not any(k != "unmatched" for k in by_arm):
        by_arm = defaultdict(list)
        if sacct_csv is not None and sacct_csv.exists() and not parse_errors:
            source = "status.json:max_rss_gb (sacct joined 0 rows)"
        elif sacct_csv is None:
            source = "status.json:max_rss_gb (--sacct-csv not supplied)"
        for cell in cells:
            st = cell.status
            if st is not None and _is_finite(st.max_rss_gb):
                by_arm[f"{cell.method}/{cell.arm}"].append(float(st.max_rss_gb))

    tables = {key: _summarise(values) for key, values in sorted(by_arm.items())}
    pooled = [v for key, values in by_arm.items() if key != "unmatched" for v in values]
    p99 = _percentile(pooled, 99)
    recommendation = round(p99 * MEM_HEADROOM_FACTOR, 2) if p99 else None

    return CriterionResult(
        id="C1.11",
        title="memory profile by (method, arm) and production --mem recommendation",
        status=_verdict(bool(pooled)),
        expected="MaxRSS observations for every (method, arm)",
        observed=f"{len(pooled)} observations from {source}",
        detail={
            "source": source,
            "by_method_arm_gb": tables,
            "pooled_gb": _summarise(pooled),
            "unmatched_sacct_rows": unmatched,
            "recommended_production_mem_gb": recommendation,
            "recommendation_rule": f"p99 x {MEM_HEADROOM_FACTOR}",
            "parse_errors": parse_errors,
        },
        blocking=False,
    )


def check_c1_12(cells: list[Cell], max_time: float, slack: float) -> CriterionResult:
    """C1.12 -- no SLURM time-kill: nothing left at ``started``, wall clock bounded."""
    still_started: list[str] = []
    overruns: list[str] = []
    walls: list[float] = []
    limit = max_time + slack

    for cell in cells:
        st = cell.status
        if st is None:
            continue
        if st.terminal_status == "started":
            still_started.append(f"{cell.label} (killed from outside: OOM or wall limit)")
        if _is_finite(st.wall_clock_s):
            walls.append(float(st.wall_clock_s))
            if float(st.wall_clock_s) > limit:
                overruns.append(f"{cell.label} wall_clock_s={st.wall_clock_s:.1f} > {limit:.1f}")

    return CriterionResult(
        id="C1.12",
        title="no SLURM time-kill: terminal_status != started, wall clock within budget",
        status=_verdict(not still_started and not overruns and bool(walls)),
        expected=(
            f"0 killed; wall_clock_s <= {limit:.0f} s (max_time {max_time:.0f} + slack {slack:.0f})"
        ),
        observed=f"{len(still_started)} killed, {len(overruns)} over budget, n={len(walls)}",
        detail={
            "still_started": _truncate(still_started),
            "wall_clock_overruns": _truncate(overruns),
            "wall_clock_s_distribution": _summarise(walls),
        },
    )


def check_c1_13(cells: list[Cell]) -> CriterionResult:
    """C1.13 -- alphabet assertion over every dedup task's reported expression.

    T16 decomposed the alphabet: there is no ``Sub`` and no ``Div``, so a ``-``
    or a ``/`` in a canonical or IsalSR string has no encoding in Sigma_SR at
    all. Those remain **blocking**.

    ``Pow`` is **counted and disclosed, not blocking**, and the reason matters.
    This function reads ``run_log.json``'s ``canonical_string`` /
    ``isalsr_string``, which describe the **final best expression** -- an object
    that has been round-tripped through SymPy -- and *not* the live candidate
    stream. SymPy writes ``sqrt(x)`` as ``Pow(x, 1/2)`` and ``x/y`` as
    ``x*Pow(y, -1)``, so a UDFS run whose best expression is ``sqrt(x_0)``
    reports ``V^VkPnc`` even though the vendored ``NODE_ARITY`` table has no
    ``pow`` and the UDFS adapter has no ``POW`` mapping at all. Blocking on that
    fails the criterion for SymPy's notation rather than for anything the search
    did.

    **The candidate-stream assertion this criterion is named for is check B3**
    (``experiments/scripts/verify_alphabet_gate.py``), which hooks the
    canonicaliser itself: 65,631 live Bingo candidates, 0 forbidden labels.
    """
    arm = [c for c in cells if c.arm in DEDUP_ARMS and c.run_log_raw is not None]
    violations: list[str] = []
    no_string: list[str] = []
    pow_outside_set: list[str] = []
    n_strings = 0
    n_clean = 0

    for cell in arm:
        best = cell.best_expression
        candidates = {
            "canonical_string": best.get("canonical_string"),
            "isalsr_string": best.get("isalsr_string"),
        }
        found_any = False
        cell_ok = True
        for name, text in candidates.items():
            if not isinstance(text, str) or not text:
                continue
            found_any = True
            n_strings += 1
            hits = [t for t in FORBIDDEN_TOKENS if t in text]
            hits += [ch for ch in FORBIDDEN_CHARS if ch in text]
            if hits:
                cell_ok = False
                violations.append(
                    f"{cell.label} {name} forbidden={sorted(set(hits))} -> {text[:80]!r}"
                )
            # Counted and reported, never blocking -- see the docstring.
            if cell.method not in POW_ALLOWED_METHODS:
                pow_hits = [t for t in POW_TOKENS if t in text]
                if pow_hits:
                    pow_outside_set.append(f"{cell.label} {name} -> {text[:60]!r}")
        if not found_any:
            no_string.append(f"{cell.label} (empty canonical_string and isalsr_string)")
        elif cell_ok:
            n_clean += 1

    return CriterionResult(
        id="C1.13",
        title="alphabet: 0 forbidden labels in dedup-arm canonical/IsalSR strings",
        status=_verdict(not violations and bool(arm)),
        expected=f"0 forbidden labels over {len(arm)} dedup tasks",
        observed=f"{len(violations)} forbidden-label violations; {n_clean}/{len(arm)} tasks "
        f"clean, {len(no_string)} with no string, {len(pow_outside_set)} Pow-outside-set "
        f"(disclosed, not blocking)",
        detail={
            "forbidden_chars": list(FORBIDDEN_CHARS),
            "forbidden_tokens": list(FORBIDDEN_TOKENS),
            "pow_allowed_methods": list(POW_ALLOWED_METHODS),
            "n_strings_examined": n_strings,
            "violations": _truncate(violations),
            "tasks_with_no_string": _truncate(no_string),
            "pow_outside_operator_set": _truncate(pow_outside_set),
            "pow_note": (
                "Counted, not blocking. These strings describe the SymPy-round-tripped "
                "BEST EXPRESSION, not the candidate stream: SymPy writes sqrt(x) as "
                "Pow(x,1/2) and x/y as x*Pow(y,-1). UDFS's vendored NODE_ARITY has no "
                "pow and its adapter has no POW mapping, so the search cannot have "
                "produced one. The candidate-stream assertion is check B3."
            ),
        },
    )


def check_c1_14(cells: list[Cell]) -> CriterionResult:
    """C1.14 -- metadata.hardware.engine == "native" on every task."""
    offenders: list[str] = []
    engines: Counter[str] = Counter()
    n_checked = 0
    for cell in cells:
        if cell.run_log_raw is None:
            continue
        n_checked += 1
        hardware = cell.metadata.get("hardware")
        engine = hardware.get("engine") if isinstance(hardware, dict) else None
        engines[str(engine)] += 1
        if engine != "native":
            offenders.append(f"{cell.label} engine={engine!r}")

    return CriterionResult(
        id="C1.14",
        title='metadata.hardware.engine == "native" on every task',
        status=_verdict(not offenders and n_checked > 0),
        expected=f"{n_checked}/{n_checked} native",
        observed=f"{n_checked - len(offenders)}/{n_checked} native",
        detail={"engine_histogram": dict(engines), "non_native": _truncate(offenders)},
    )


def check_c1_15(
    root: Path,
    by_key: dict[tuple[str, str, str, int], Cell],
    expected: set[tuple[str, str, str, int]],
    expected_source: str,
    expected_tasks: int,
) -> CriterionResult:
    """C1.15 -- cell reconciliation, with every gap individually named."""
    try:
        from experiments.models.status_ledger import collect_status_ledger, reconcile
    except Exception as exc:  # noqa: BLE001
        return CriterionResult(
            id="C1.15",
            title="cell reconciliation via status_ledger.reconcile",
            status="FAIL",
            expected="importable status_ledger",
            observed=f"{type(exc).__name__}: {exc}",
            detail={},
        )

    # output_csv=None: the certifier is read-only with respect to the root.
    rows = collect_status_ledger(root, None)
    report = reconcile(rows, expected)

    # reconcile() answers "did a status record exist and terminate cleanly".
    # A cell that terminated cleanly but left no run_log.json is still a gap,
    # and it is the gap a deleted output produces, so it is named separately.
    no_run_log = sorted(
        "/".join(str(p) for p in key)
        for key in expected
        if (cell := by_key.get(key)) is not None and cell.run_log_raw is None
    )

    ok = bool(report["reconciled"]) and not no_run_log and report["n_expected"] > 0
    return CriterionResult(
        id="C1.15",
        title="cell reconciliation: expected vs observed, every gap named",
        status=_verdict(ok),
        expected=report["n_expected"],
        observed=report["n_observed"],
        detail={
            "expected_set_source": expected_source,
            "expected_tasks_argument": expected_tasks,
            "n_completed": report["n_completed"],
            "missing_no_status_record": _truncate(
                ["/".join(str(p) for p in k) for k in report["missing"]]
            ),
            "killed_still_started": _truncate(
                ["/".join(str(p) for p in k) for k in report["killed"]]
            ),
            "failed": _truncate(["/".join(str(p) for p in k) for k in report["failed"]]),
            "status_ok_but_no_run_log": _truncate(no_run_log),
        },
    )


def check_c1_16(root: Path, cells: list[Cell]) -> CriterionResult:
    """C1.16 -- all three contrasts emit a parseable file per (method, problem).

    Existence and validity only. At n = 3 the minimum two-sided Wilcoxon p is
    0.25, so **nothing** is asserted about any p-value; asserting significance
    at this sample size would be a category error.
    """
    try:
        from experiments.models.schemas import PairedStats
    except Exception as exc:  # noqa: BLE001
        return CriterionResult(
            id="C1.16",
            title="three paired-stats contrasts per (method, problem)",
            status="FAIL",
            expected="importable PairedStats",
            observed=f"{type(exc).__name__}: {exc}",
            detail={},
        )

    pairs = sorted({(c.method, c.benchmark, c.problem) for c in cells})
    seeds_with_log: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for cell in cells:
        if cell.run_log_raw is not None:
            seeds_with_log[(cell.method, cell.arm, cell.problem)].add(cell.seed)

    missing: list[str] = []
    unreadable: list[str] = []
    wrong_n_seeds: list[str] = []
    n_ok = 0
    expected_files = len(pairs) * len(CONTRAST_FILES)

    for method, benchmark, problem in pairs:
        problem_dir = root / method / benchmark / _slug(problem)
        for ref_arm, treat_arm, fname in CONTRAST_FILES:
            path = problem_dir / fname
            if not path.exists():
                missing.append(str(path))
                continue
            try:
                stats = PairedStats.load_json(path)
            except Exception as exc:  # noqa: BLE001
                unreadable.append(f"{path} ({type(exc).__name__}: {exc})")
                continue
            n_seeds = len(
                seeds_with_log[(method, ref_arm, problem)]
                & seeds_with_log[(method, treat_arm, problem)]
            )
            if n_seeds != len(DEFAULT_SEEDS):
                wrong_n_seeds.append(
                    f"{path}: n_paired_seeds={n_seeds} (expected {len(DEFAULT_SEEDS)})"
                )
                continue
            if not stats.metrics:
                unreadable.append(f"{path}: parsed but metrics dict is empty")
                continue
            n_ok += 1

    ok = not missing and not unreadable and not wrong_n_seeds and expected_files > 0
    return CriterionResult(
        id="C1.16",
        title="3 paired-stats contrasts per (method, problem), each with 3 paired seeds",
        status=_verdict(ok),
        expected=f"{expected_files} files (3 contrasts x {len(pairs)} method-problem pairs)",
        observed=f"{n_ok}/{expected_files} valid",
        detail={
            "p_value_note": (
                "existence and validity only. At n=3 the minimum two-sided "
                "Wilcoxon p is 0.25; nothing is asserted about significance."
            ),
            "missing": _truncate(missing),
            "unreadable_or_empty": _truncate(unreadable),
            "wrong_paired_seed_count": _truncate(wrong_n_seeds),
        },
    )


def check_c1_17(root: Path, cells: list[Cell]) -> CriterionResult:
    """C1.17 -- one valid ``aggregate.csv`` per (method, problem, arm).

    ``aggregate_all_metrics`` writes one row **per metric**, not per seed, so
    the row count asserted is ``len(METRIC_EXTRACTORS)``. The seed count of a
    cell is checked by C1.16.
    """
    try:
        from experiments.models.analyzer.aggregation import METRIC_EXTRACTORS
        from experiments.models.schemas import AGGREGATE_COLUMNS

        n_rows_expected = len(METRIC_EXTRACTORS)
        columns = list(AGGREGATE_COLUMNS)
    except Exception:  # noqa: BLE001
        n_rows_expected = FALLBACK_N_AGGREGATE_ROWS
        columns = []

    triples = sorted({(c.method, c.benchmark, c.problem, c.arm) for c in cells})
    missing: list[str] = []
    unreadable: list[str] = []
    wrong_rows: list[str] = []
    wrong_header: list[str] = []
    n_ok = 0

    for method, benchmark, problem, arm in triples:
        path = root / method / benchmark / _slug(problem) / arm / "aggregate.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                header = list(reader.fieldnames or [])
                rows = list(reader)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            unreadable.append(f"{path} ({type(exc).__name__})")
            continue
        if columns and header != columns:
            wrong_header.append(f"{path}: header={header}")
            continue
        if len(rows) != n_rows_expected:
            wrong_rows.append(f"{path}: {len(rows)} rows (expected {n_rows_expected})")
            continue
        n_ok += 1

    ok = not missing and not unreadable and not wrong_rows and not wrong_header and bool(triples)
    return CriterionResult(
        id="C1.17",
        title=f"aggregate.csv per (method, problem, arm), {n_rows_expected} metric rows each",
        status=_verdict(ok),
        expected=f"{len(triples)} files x {n_rows_expected} rows",
        observed=f"{n_ok}/{len(triples)} valid",
        detail={
            "premise_note": (
                "aggregate_all_metrics emits ONE ROW PER METRIC "
                f"({n_rows_expected} of them), not one row per seed. The "
                "'exactly 3 data rows' expectation is contradicted by "
                "experiments/models/analyzer/aggregation.py:aggregate_all_metrics."
            ),
            "missing": _truncate(missing),
            "unreadable": _truncate(unreadable),
            "wrong_row_count": _truncate(wrong_rows),
            "wrong_header": _truncate(wrong_header),
        },
    )


def check_c2(root: Path, expected: set[tuple[str, str, str, int]]) -> CriterionResult:
    """C2 -- ``status_ledger.csv`` exists with one row per cell and the right columns."""
    try:
        from experiments.models.status_ledger import LEDGER_COLUMNS
    except Exception as exc:  # noqa: BLE001
        return CriterionResult(
            id="C2",
            title="status_ledger.csv at the root, one row per cell",
            status="FAIL",
            expected="importable LEDGER_COLUMNS",
            observed=f"{type(exc).__name__}: {exc}",
            detail={},
        )

    path = root / "status_ledger.csv"
    if not path.exists():
        return CriterionResult(
            id="C2",
            title="status_ledger.csv at the root, one row per cell",
            status="FAIL",
            expected=f"{len(expected)} rows at {path}",
            observed="file absent",
            detail={"missing_path": str(path)},
        )
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return CriterionResult(
            id="C2",
            title="status_ledger.csv at the root, one row per cell",
            status="FAIL",
            expected=f"{len(expected)} parseable rows",
            observed=f"{type(exc).__name__}: {exc}",
            detail={"path": str(path)},
        )

    keys: Counter[tuple[str, str, str, str]] = Counter(
        (r.get("method", ""), r.get("arm", ""), r.get("problem", ""), r.get("seed", ""))
        for r in rows
    )
    duplicates = [f"{k}: {n}" for k, n in keys.items() if n > 1]
    observed_keys = {(m, a, p, int(s)) for (m, a, p, s) in keys if s.isdigit()}
    absent = sorted("/".join(str(x) for x in k) for k in expected - observed_keys)
    extra = sorted("/".join(str(x) for x in k) for k in observed_keys - expected)

    ok = header == list(LEDGER_COLUMNS) and not duplicates and not absent and not extra
    return CriterionResult(
        id="C2",
        title="status_ledger.csv at the root, one row per (method, arm, problem, seed)",
        status=_verdict(ok),
        expected=f"{len(expected)} rows, columns == LEDGER_COLUMNS",
        observed=f"{len(rows)} rows, {len(observed_keys)} distinct cells",
        detail={
            "path": str(path),
            "header_matches_LEDGER_COLUMNS": header == list(LEDGER_COLUMNS),
            "header": header,
            "duplicate_rows": _truncate(duplicates),
            "cells_absent_from_ledger": _truncate(absent),
            "unexpected_cells_in_ledger": _truncate(extra),
        },
    )


def check_c4(cells: list[Cell]) -> CriterionResult:
    """C4 -- cross-arm data identity via ``metadata.data_fingerprint``.

    Two independent assertions. First, every ``(problem, seed)`` must carry ONE
    fingerprint across all arms and methods: a disagreement means the paired
    tests compared different samples. Second, the fingerprints of distinct
    ``(problem, seed)`` pairs must be mutually distinct: a repeat means the seed
    is not reaching the generator, which would silently collapse three seeds
    into one.
    """
    fp_by_cell: dict[tuple[str, int], set[str]] = defaultdict(set)
    fp_counts: Counter[str] = Counter()
    fp_owners: dict[str, set[tuple[str, int]]] = defaultdict(set)
    missing: list[str] = []
    n_methods = len({c.method for c in cells}) or 1
    n_arms = len({c.arm for c in cells}) or 1

    for cell in cells:
        if cell.run_log_raw is None:
            continue
        fp = cell.metadata.get("data_fingerprint")
        if not isinstance(fp, str) or not fp:
            missing.append(f"{cell.label} data_fingerprint={fp!r}")
            continue
        fp_by_cell[(cell.problem, cell.seed)].add(fp)
        fp_counts[fp] += 1
        fp_owners[fp].add((cell.problem, cell.seed))

    disagreements = [
        f"{problem}/seed_{seed}: {len(fps)} distinct fingerprints across arms"
        for (problem, seed), fps in sorted(fp_by_cell.items())
        if len(fps) > 1
    ]
    collisions = [
        f"{fp[:16]}...: shared by {sorted(owners)}"
        for fp, owners in sorted(fp_owners.items())
        if len(owners) > 1
    ]
    expected_multiplicity = n_arms * n_methods
    wrong_multiplicity = [
        f"{fp[:16]}... appears {n}x (expected {expected_multiplicity}) for {sorted(fp_owners[fp])}"
        for fp, n in sorted(fp_counts.items())
        if n != expected_multiplicity
    ]

    n_expected_distinct = len(fp_by_cell)
    ok = (
        bool(fp_counts)
        and not missing
        and not disagreements
        and not collisions
        and not wrong_multiplicity
        and len(fp_counts) == n_expected_distinct
    )
    return CriterionResult(
        id="C4",
        title="cross-arm data identity: one fingerprint per (problem, seed), all distinct",
        status=_verdict(ok),
        expected=(
            f"{n_expected_distinct} distinct fingerprints, each appearing "
            f"{expected_multiplicity}x ({n_arms} arms x {n_methods} methods)"
        ),
        observed=f"{len(fp_counts)} distinct fingerprints over {sum(fp_counts.values())} run logs",
        detail={
            "n_problem_seed_pairs": n_expected_distinct,
            "expected_multiplicity": expected_multiplicity,
            "missing_fingerprint": _truncate(missing),
            "cross_arm_disagreement": _truncate(disagreements),
            "fingerprint_collisions_across_problem_seed": _truncate(collisions),
            "wrong_multiplicity": _truncate(wrong_multiplicity),
            "multiplicity_histogram": dict(Counter(fp_counts.values())),
        },
    )


# ====================================================================== #
# Reporting
# ====================================================================== #


def render_table(results: list[CriterionResult]) -> str:
    """Render the fixed-width verdict table printed to stdout."""
    lines = [
        f"{'ID':<6} {'VERDICT':<8} {'BLOCK':<6} {'OBSERVED':<44} CRITERION",
        f"{'-' * 6} {'-' * 8} {'-' * 6} {'-' * 44} {'-' * 60}",
    ]
    for r in results:
        observed = str(r.observed)
        if len(observed) > 44:
            observed = observed[:41] + "..."
        lines.append(
            f"{r.id:<6} {r.status:<8} {'yes' if r.blocking else 'no':<6} {observed:<44} {r.title}"
        )
    return "\n".join(lines)


def render_markdown(
    results: list[CriterionResult],
    root: Path,
    verdict: str,
    generated_at: str,
) -> str:
    """Render the markdown certification report."""
    lines = [
        "# Campaign C2 pre-flight -- Stage C certification",
        "",
        f"- **Verdict**: `{verdict}`",
        f"- **Root**: `{root}`",
        f"- **Generated**: {generated_at}",
        f"- **Blocking failures**: {sum(1 for r in results if r.blocking and r.status != 'PASS')}",
        "",
        "## Verdict table",
        "",
        "| ID | Verdict | Blocking | Expected | Observed | Criterion |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r.id}` | **{r.status}** | {'yes' if r.blocking else 'no'} | "
            f"{r.expected} | {r.observed} | {r.title} |"
        )
    lines += ["", "## Detail", ""]
    for r in results:
        lines += [
            f"### {r.id} -- {r.status}",
            "",
            f"{r.title}",
            "",
            "```json",
            json.dumps(r.detail, indent=2, default=str),
            "```",
            "",
        ]
    return "\n".join(lines)


# ====================================================================== #
# Entry point
# ====================================================================== #


def certify(
    root: Path,
    expected_tasks: int,
    max_time: float,
    wall_slack: float,
    seeds: tuple[int, ...],
    sacct_csv: Path | None,
) -> list[CriterionResult]:
    """Run every criterion against a results root.

    Args:
        root: Stage C output root.
        expected_tasks: Number of cells the campaign should contain.
        max_time: Per-run search budget in seconds.
        wall_slack: Allowance above ``max_time`` for setup, constant
            optimisation and the SymPy equivalence check.
        seeds: Campaign seeds.
        sacct_csv: Optional ``JobID,MaxRSS`` file for the memory profile.

    Returns:
        One :class:`CriterionResult` per criterion, in canonical order.
    """
    registry, registry_error = _load_registry()
    cells = discover_cells(root, registry)
    for cell in cells:
        hydrate(cell)
    by_key = {c.key: c for c in cells}
    expected, _bench_of, source = build_expected_cells(registry, cells, seeds, expected_tasks)

    if registry_error:
        log.error("Benchmark registry unavailable: %s", registry_error)

    return [
        check_c1_1(by_key, expected),
        check_c1_2(cells),
        check_c1_3(cells),
        check_c1_4(root, cells, registry),
        check_c1_5(registry, cells),
        check_c1_6(cells),
        check_c1_7(cells),
        check_c1_8(cells),
        check_c1_9(cells),
        check_c1_10(cells),
        check_c1_11(cells, sacct_csv),
        check_c1_12(cells, max_time, wall_slack),
        check_c1_13(cells),
        check_c1_14(cells),
        check_c1_15(root, by_key, expected, source, expected_tasks),
        check_c1_16(root, cells),
        check_c1_17(root, cells),
        check_c2(root, expected),
        check_c4(cells),
    ]


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="c2_certify",
        description="Machine-check a Campaign-C2 pre-flight Stage C output root.",
    )
    parser.add_argument("--root", type=Path, required=True, help="Stage C output root.")
    parser.add_argument("--out-json", type=Path, required=True, help="Evidence JSON path.")
    parser.add_argument("--out-md", type=Path, required=True, help="Markdown report path.")
    parser.add_argument(
        "--sacct-csv",
        type=Path,
        default=None,
        help="Optional CSV of JobID,MaxRSS rows for the C1.11 memory profile.",
    )
    parser.add_argument("--expected-tasks", type=int, default=1260, help="Expected cell count.")
    parser.add_argument("--max-time", type=float, default=900.0, help="Per-run search budget (s).")
    parser.add_argument(
        "--wall-slack",
        type=float,
        default=DEFAULT_WALL_SLACK_S,
        help="Wall-clock allowance above --max-time for non-search work (s).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated campaign seeds.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        help="Logging level for diagnostics on stderr.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the certifier and return the process exit code.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv[1:]``.

    Returns:
        ``0`` when every blocking criterion passes, ``1`` otherwise.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        seeds = tuple(int(s) for s in str(args.seeds).split(",") if s.strip())
    except ValueError:
        seeds = DEFAULT_SEEDS

    root = Path(args.root)
    generated_at = datetime.now(tz=UTC).isoformat()

    if not root.is_dir():
        results = [
            CriterionResult(
                id="C0",
                title="results root exists",
                status="FAIL",
                expected=f"directory at {root}",
                observed="absent",
                detail={"missing_path": str(root)},
            )
        ]
    else:
        results = certify(
            root=root,
            expected_tasks=int(args.expected_tasks),
            max_time=float(args.max_time),
            wall_slack=float(args.wall_slack),
            seeds=seeds,
            sacct_csv=args.sacct_csv,
        )

    n_blocking_failures = sum(1 for r in results if r.blocking and r.status != "PASS")
    verdict = "GO" if n_blocking_failures == 0 else "NO-GO"

    payload: dict[str, Any] = {r.id: r.to_dict() for r in results}
    payload["verdict"] = verdict
    payload["generated_at"] = generated_at
    payload["root"] = str(root)
    payload["expected_tasks"] = int(args.expected_tasks)
    payload["n_blocking_failures"] = n_blocking_failures

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    out_md.write_text(render_markdown(results, root, verdict, generated_at), encoding="utf-8")

    print(render_table(results))
    print()
    print(f"VERDICT: {verdict}  ({n_blocking_failures} blocking failure(s))")
    print(f"evidence: {out_json}")
    print(f"report:   {out_md}")
    return 0 if n_blocking_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
