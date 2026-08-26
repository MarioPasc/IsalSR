"""Campaign completeness and provenance integrity checks for the analyzer.

Two properties the C1 campaign did not enforce, and which its analysis was
therefore unable to state:

**Completeness (pre-flight check E6).** C1 reported 1,500 UDFS cells and 1,465
Bingo cells, a 35-cell shortfall that was neither named nor explained until a
forensic pass reconstructed it from SLURM logs months later. A count that
matches is worth nothing unless a mismatch *names* the offending cells, so
:func:`check_completeness` reconciles an expected cell grid against the run
logs actually on disk and returns the missing cells individually.

**Provenance (pre-flight check E7).** Every table in the revised manuscript must
be traceable to one campaign root, one commit and one configuration
(EXECUTION-PLAN §5.1). Pooling two roots silently produces a number that
belongs to neither, so :func:`check_provenance` refuses a root whose runs
disagree about the code or configuration that produced them.

Both checks fail closed: the caller is expected to abort unless the operator
has explicitly opted into the weaker behaviour.

Notes:
    ``git_commit`` is **not** used as a provenance key. It is ``None`` on every
    run log this project has produced, so a guard keyed on it would see a single
    value and pass vacuously on any input -- the SP-6 trap, where a
    zero-everywhere ledger means the counters are dead rather than the rates
    zero. Keys that carry one value because they carry *no* value are reported
    as non-informative instead of as agreement.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Provenance keys read from ``metadata.hardware`` that must hold exactly one
# value across the whole root. Two commits in one root is the mid-wave redeploy
# defect; two build hashes is a stale extension on some nodes.
ROOT_PROVENANCE_KEYS: tuple[str, ...] = ("git_describe", "git_dirty", "build_hash")

# Provenance keys read from ``metadata`` that must hold exactly one value per
# ``(method, benchmark)``. A config legitimately differs between suites, so the
# root-wide check would be wrong here; within one suite it must not move.
SCOPED_PROVENANCE_KEYS: tuple[str, ...] = ("config_sha256",)

RUN_LOG_NAME = "run_log.json"


class CampaignIntegrityError(RuntimeError):
    """Raised when a results root is incomplete or mixes provenance.

    Carries the two reports so a caller can persist them without re-walking the
    root, which at campaign scale is 8,400 file reads.

    Attributes:
        completeness: The cell reconciliation that triggered or accompanied the
            failure.
        provenance: The provenance scan that triggered or accompanied it.
    """

    def __init__(
        self,
        message: str,
        completeness: CompletenessReport | None = None,
        provenance: ProvenanceReport | None = None,
    ) -> None:
        super().__init__(message)
        self.completeness = completeness
        self.provenance = provenance


@dataclass(frozen=True, order=True)
class Cell:
    """One ``(method, benchmark, problem, arm, seed)`` campaign cell."""

    method: str
    benchmark: str
    problem: str
    arm: str
    seed: str

    def __str__(self) -> str:
        return f"{self.method}/{self.benchmark}/{self.problem}/{self.arm}/{self.seed}"


@dataclass
class CompletenessReport:
    """Result of reconciling an expected cell grid against what is on disk."""

    n_expected: int = 0
    n_observed: int = 0
    missing: list[Cell] = field(default_factory=list)
    unreadable: list[Cell] = field(default_factory=list)
    per_group: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """True when every expected cell is present and parses."""
        return not self.missing and not self.unreadable

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the certification artefact."""
        return {
            "n_expected": self.n_expected,
            "n_observed": self.n_observed,
            "n_missing": len(self.missing),
            "n_unreadable": len(self.unreadable),
            "complete": self.complete,
            "missing": [str(c) for c in self.missing],
            "unreadable": [str(c) for c in self.unreadable],
            "per_group": self.per_group,
        }

    def format_report(self, max_named: int = 50) -> str:
        """Render a human-readable summary that names the offending cells."""
        head = f"cell reconciliation: {self.n_observed}/{self.n_expected} present"
        if self.complete:
            return head + " -- complete"
        lines = [head]
        for label, cells in (("MISSING", self.missing), ("UNREADABLE", self.unreadable)):
            if not cells:
                continue
            lines.append(f"  {len(cells)} {label}:")
            lines.extend(f"    - {c}" for c in cells[:max_named])
            if len(cells) > max_named:
                lines.append(f"    ... and {len(cells) - max_named} more")
        return "\n".join(lines)


@dataclass
class ProvenanceReport:
    """Distinct provenance values observed across a results root."""

    n_runs: int = 0
    root_keys: dict[str, dict[str, int]] = field(default_factory=dict)
    scoped_keys: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    non_informative: list[str] = field(default_factory=list)

    @property
    def conflicts(self) -> list[str]:
        """Human-readable description of every key holding more than one value."""
        out: list[str] = []
        for key, counts in sorted(self.root_keys.items()):
            if len(counts) > 1:
                spread = ", ".join(f"{v!r} x{n}" for v, n in sorted(counts.items()))
                out.append(f"{key}: {len(counts)} distinct values across the root -- {spread}")
        for key, groups in sorted(self.scoped_keys.items()):
            for group, counts in sorted(groups.items()):
                if len(counts) > 1:
                    spread = ", ".join(f"{v!r} x{n}" for v, n in sorted(counts.items()))
                    out.append(f"{key} within {group}: {len(counts)} distinct values -- {spread}")
        return out

    @property
    def mixed(self) -> bool:
        """True when the root pools runs of differing provenance."""
        return bool(self.conflicts)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the certification artefact."""
        return {
            "n_runs": self.n_runs,
            "mixed": self.mixed,
            "conflicts": self.conflicts,
            "non_informative_keys": self.non_informative,
            "root_keys": self.root_keys,
            "scoped_keys": self.scoped_keys,
        }

    def format_report(self) -> str:
        """Render a human-readable provenance summary."""
        if not self.mixed:
            head = f"provenance: single-valued across {self.n_runs} runs"
        else:
            head = f"provenance: MIXED across {self.n_runs} runs"
        lines = [head]
        lines.extend(f"    - {c}" for c in self.conflicts)
        if self.non_informative:
            lines.append(
                "  non-informative (absent on every run, so agreement proves nothing): "
                + ", ".join(self.non_informative)
            )
        return "\n".join(lines)


def _iter_run_logs(
    results_dir: Path,
    methods: Sequence[str],
    benchmarks: Sequence[str],
) -> list[tuple[Cell, Path]]:
    """Enumerate run logs under the ``method/benchmark/problem/arm/seed`` layout.

    Args:
        results_dir: Campaign or smoke root.
        methods: Method directories to walk.
        benchmarks: Benchmark directories to walk.

    Returns:
        ``(cell, path)`` pairs, sorted, for every run log found.
    """
    found: list[tuple[Cell, Path]] = []
    for method in methods:
        for benchmark in benchmarks:
            bench_dir = results_dir / method / benchmark
            if not bench_dir.is_dir():
                continue
            for problem_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
                for arm_dir in sorted(p for p in problem_dir.iterdir() if p.is_dir()):
                    for seed_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
                        path = seed_dir / RUN_LOG_NAME
                        cell = Cell(
                            method=method,
                            benchmark=benchmark,
                            problem=problem_dir.name,
                            arm=arm_dir.name,
                            seed=seed_dir.name,
                        )
                        found.append((cell, path))
    return sorted(found)


def infer_expected_cells(
    results_dir: Path,
    methods: Sequence[str],
    benchmarks: Sequence[str],
    variants: Sequence[str],
) -> set[Cell]:
    """Infer the cell grid a complete campaign root would contain.

    The grid is the cross product of the problems and seeds observed in each
    ``(method, benchmark)`` with the arms observed anywhere in the root. This is
    deliberately not "whatever is on disk": a deleted run still has its problem,
    arm and seed attested by its siblings, so the cross product still expects it
    and the deletion is caught. Arms are pooled root-wide rather than per
    benchmark so that deleting a whole ``(benchmark, arm)`` slice is caught too.

    Args:
        results_dir: Campaign or smoke root.
        methods: Method directories to walk.
        benchmarks: Benchmark directories to walk.
        variants: Arms requested by the caller; only these are ever expected, so
            a two-arm C1 root queried for two arms yields the C1 grid.

    Returns:
        Every ``Cell`` the root is expected to hold.
    """
    observed = _iter_run_logs(results_dir, methods, benchmarks)
    requested = set(variants)

    arms_root: set[str] = {c.arm for c, _ in observed if c.arm in requested}
    problems: dict[tuple[str, str], set[str]] = defaultdict(set)
    seeds: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cell, _ in observed:
        if cell.arm not in requested:
            continue
        key = (cell.method, cell.benchmark)
        problems[key].add(cell.problem)
        seeds[key].add(cell.seed)

    expected: set[Cell] = set()
    for (method, benchmark), probs in problems.items():
        for problem in probs:
            for arm in arms_root:
                for seed in seeds[(method, benchmark)]:
                    expected.add(Cell(method, benchmark, problem, arm, seed))
    return expected


def scan_root(
    results_dir: Path,
    methods: Sequence[str],
    benchmarks: Sequence[str],
    variants: Sequence[str],
) -> tuple[CompletenessReport, ProvenanceReport]:
    """Reconcile cells and collect provenance in a single pass over the root.

    Args:
        results_dir: Campaign or smoke root.
        methods: Method directories to walk.
        benchmarks: Benchmark directories to walk.
        variants: Arms to analyse.

    Returns:
        The completeness report and the provenance report.
    """
    expected = infer_expected_cells(results_dir, methods, benchmarks, variants)
    requested = set(variants)

    comp = CompletenessReport(n_expected=len(expected))
    prov = ProvenanceReport()

    root_counts: dict[str, Counter[str]] = {k: Counter() for k in ROOT_PROVENANCE_KEYS}
    scoped_counts: dict[str, dict[str, Counter[str]]] = {
        k: defaultdict(Counter) for k in SCOPED_PROVENANCE_KEYS
    }
    present_any: dict[str, bool] = {
        k: False for k in (*ROOT_PROVENANCE_KEYS, *SCOPED_PROVENANCE_KEYS)
    }
    group_totals: dict[str, int] = defaultdict(int)

    observed_cells: set[Cell] = set()
    for cell, path in _iter_run_logs(results_dir, methods, benchmarks):
        if cell.arm not in requested:
            continue
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            comp.unreadable.append(cell)
            continue

        observed_cells.add(cell)
        prov.n_runs += 1
        group = f"{cell.method}/{cell.benchmark}"
        group_totals[group] += 1

        metadata = payload.get("metadata") or {}
        hardware = metadata.get("hardware") or {}
        for key in ROOT_PROVENANCE_KEYS:
            value = hardware.get(key)
            if value is not None:
                present_any[key] = True
            root_counts[key][repr(value)] += 1
        for key in SCOPED_PROVENANCE_KEYS:
            value = metadata.get(key)
            if value is not None:
                present_any[key] = True
            scoped_counts[key][group][repr(value)] += 1

    comp.n_observed = len(observed_cells)
    comp.missing = sorted(expected - observed_cells)
    comp.per_group = {
        group: {
            "observed": n_obs,
            "expected": sum(1 for c in expected if f"{c.method}/{c.benchmark}" == group),
        }
        for group, n_obs in sorted(group_totals.items())
    }

    # A key absent on every run holds one value only because it holds none.
    # Report that rather than counting it as agreement (the SP-6 trap).
    prov.non_informative = sorted(k for k, seen in present_any.items() if not seen)
    prov.root_keys = {
        k: dict(counts) for k, counts in root_counts.items() if k not in prov.non_informative
    }
    prov.scoped_keys = {
        k: {g: dict(c) for g, c in groups.items()}
        for k, groups in scoped_counts.items()
        if k not in prov.non_informative
    }
    return comp, prov


def enforce_integrity(
    results_dir: Path,
    methods: Sequence[str],
    benchmarks: Sequence[str],
    variants: Sequence[str],
    allow_incomplete: bool = False,
    allow_mixed_provenance: bool = False,
) -> tuple[CompletenessReport, ProvenanceReport]:
    """Check a root and refuse to analyse it when either property fails.

    Args:
        results_dir: Campaign or smoke root.
        methods: Method directories to walk.
        benchmarks: Benchmark directories to walk.
        variants: Arms to analyse.
        allow_incomplete: Downgrade a missing-cell failure to a warning. The
            named cells are logged either way.
        allow_mixed_provenance: Downgrade a provenance conflict to a warning,
            leaving the per-run values labelled in the returned report.

    Returns:
        The completeness and provenance reports.

    Raises:
        CampaignIntegrityError: If cells are missing or provenance is mixed and
            the corresponding override was not requested.
    """
    comp, prov = scan_root(results_dir, methods, benchmarks, variants)

    problems: list[str] = []
    if not comp.complete:
        (log.warning if allow_incomplete else log.error)("%s", comp.format_report())
        if not allow_incomplete:
            problems.append(
                f"{len(comp.missing)} missing and {len(comp.unreadable)} unreadable cells "
                f"(pass allow_incomplete=True to proceed)"
            )
    else:
        log.info("%s", comp.format_report())

    if prov.mixed:
        (log.warning if allow_mixed_provenance else log.error)("%s", prov.format_report())
        if not allow_mixed_provenance:
            problems.append(
                f"{len(prov.conflicts)} provenance conflict(s) "
                f"(pass allow_mixed_provenance=True to proceed)"
            )
    else:
        log.info("%s", prov.format_report())

    if problems:
        raise CampaignIntegrityError(
            "results root failed integrity checks: "
            + "; ".join(problems)
            + "\n"
            + comp.format_report()
            + "\n"
            + prov.format_report(),
            completeness=comp,
            provenance=prov,
        )
    return comp, prov
