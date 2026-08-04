"""Machine-checked go/no-go certification of the Campaign-C2 Stage D pre-flight.

Stage D is the twelve full-length cells enumerated by
:mod:`experiments.scripts.stage_d_task_spec`: three problems, three arms, two
methods, one seed, the full 43,200 s search budget. The 15-minute Stage C smoke
proves nothing about a 12-hour run -- memory growth, heap fragmentation,
dedup-set size, timeout paths and convergence are all budget-dependent -- so
Stage D is what actually sizes campaign C2's resource request.

This module evaluates the eight ``D1.x`` criteria of ``EXECUTION-PLAN`` section
4.4. It shares its vocabulary, its discovery walk and its report shape with
:mod:`experiments.scripts.c2_certify`, which it imports rather than re-implements.
The two properties that make the Stage C certifier usable hold here too:

1. **It never raises on missing or malformed data.** A missing
   ``rss_timeseries.csv``, an unreadable ``sacct`` export, an absent C1
   reference and a truncated manifest are all *recorded* findings naming the
   path, never tracebacks.
2. **It is honest about a partial root.** Every criterion reports ``observed``
   against ``expected`` over the locked 12-cell registry, so a cell that never
   ran is named rather than silently dropped from the denominator.

The process is read-only with respect to the results root except for the three
files it is asked to write: ``--out-json``, ``--out-md`` and the Stage D
pre-flight manifest at ``<root>/c2_preflight/stage_d_manifest.json`` (D1.8).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.scripts.c2_certify import (  # noqa: E402
    DEDUP_ARMS,
    RUN_LOG_FIELD_SPEC,
    Cell,
    CriterionResult,
    _is_finite,
    _load_registry,
    _parse_maxrss_to_gb,
    _percentile,
    _slug,
    _summarise,
    _truncate,
    _verdict,
    _walk_spec,
    discover_cells,
    hydrate,
    render_table,
)
from experiments.scripts.stage_d_task_spec import (  # noqa: E402
    NAN_PROBLEMS,
    STAGE_D_CELLS,
    STAGE_D_MAX_TIME_S,
    STAGE_D_SUITE,
    StageDCell,
)

log = logging.getLogger("stage_d_certify")

# ====================================================================== #
# Thresholds and locked configuration
# ====================================================================== #

#: SLURM wall limit in seconds. ``STAGE_D_WALL`` is ``0-16:00:00``.
DEFAULT_WALL_S: float = 57_600.0

#: D1.1: minimum ``(wall - elapsed) / wall``.
MIN_WALL_HEADROOM_FRAC: float = 0.10

#: D1.2(a): minimum ``(requested - peak) / requested``.
MIN_MEM_HEADROOM_FRAC: float = 0.30

#: D1.2(b): the utilisation a production request is sized for. The recommended
#: request is ``peak / 0.70``, i.e. the peak sits at 70 % of the request and the
#: remaining 30 % is the D1.2(a) headroom carried forward.
MEM_TARGET_UTILISATION: float = 0.70

#: D1.2(b) rounding granularity, in GiB. SLURM requests are integers and the
#: three Stage D classes (16 / 32 / 256 GB) are all multiples of 8, so the
#: recommendation is rounded UP to the next multiple of 8 GiB. Rounding up is
#: never optional: rounding a memory request down is an OOM.
MEM_ROUND_STEP_GB: int = 8

#: D1.6: relative tolerance on a rho DROP against C1. A drop beyond this is the
#: alarm -- T16's decomposition grew ``k`` by about 22 %, so rho was predicted to
#: RISE; a fall means the decomposition is not reaching the canonicaliser.
#: There is deliberately no upper bound: a rise is the predicted direction.
RHO_DROP_TOL: float = 0.10

#: D1.6: two-sided band on the R2 paired delta, in absolute R2 units. Stage D
#: runs ONE seed against C1's 30-seed mean, and on the hard tier the per-seed
#: baseline spread is order 0.05-0.10 (the bottleneck analysis measured a 26x to
#: 1518x variance reduction precisely because baseline seeds scatter). The band
#: is set wide so that seed noise alone does not trip it: a trip is meant to be
#: explained, not routinely absorbed.
R2_DELTA_BAND: float = 0.15

#: D1.6: tolerance for the cross-check that validates reconstructing C1's
#: absolute rho from its published per-problem deltas.
RHO_RECONSTRUCTION_TOL: float = 1e-3

#: D1.6 default reference: campaign C1's analysis directory (read-only).
DEFAULT_C1_REFERENCE: Path = Path(
    "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/"
    "model_validation/real_benchmarks/wl_subtree_unified/analysis"
)

#: D1.8: where the Stage D pre-flight manifest is written, relative to the root.
MANIFEST_RELPATH: tuple[str, str] = ("c2_preflight", "stage_d_manifest.json")

#: D1.8: the one D1 problem whose target function was corrected after C1 ran.
#: Recorded so the manifest's continuity block is sourced rather than invented;
#: it is outside Stage D's three-problem scope, which the note states.
CORRECTED_DEFINITION_PROBLEMS: tuple[str, ...] = ("I.34.27",)

# ---------------------------------------------------------------------- #
# Framing strings. These are emitted into the report so a reader never has to
# resolve a cross-reference to interpret a number.
# ---------------------------------------------------------------------- #

WALL_MARGIN_NOTE: str = (
    "The 4 h margin between the 12 h search budget and the 16 h wall exists for "
    "the post-search tail, which max_time does not bound: constant optimisation, "
    "the SymPy equivalence check and process teardown all sit outside the search "
    "timer. EXECUTION-PLAN section 11.1 (2026-08-03) records one Stage C cell "
    "that spent 7+ minutes in post-search SymPy after a correct 900 s search."
)

SACCT_TRAP_NOTE: str = (
    "sacct traps (EXECUTION-PLAN section 11.1, 2026-08-02). (1) NEVER 'sacct -X': "
    "it returns an EMPTY MaxRSS because memory is accounted on the .batch step, "
    "so the profile comes back silently blank. (2) Join on JobIDRaw, never JobID: "
    "for an array JobID reads '<array_id>_<task>' while status.json records the "
    "raw numeric id, and joining on JobID matched 42 of 1,260 rows while still "
    "reporting PASS. The producer is slurm/c2_stage_d/aggregate_worker.sh."
)

ENGINE_NOTE: str = (
    "The engine changed between C1 and C2 (pure Python -> C++ native). That "
    "changes COST, not VALUES: the canonical string is byte-identical across "
    "backends. Any difference in rho or R2 observed here is therefore NOT "
    "attributable to the engine and must be explained by the alphabet, the data "
    "or the search."
)

T16_NOTE: str = (
    "T16 decomposed the alphabet inside both adapters (SUB -> ADD+NEG, "
    "DIV -> MUL+INV), which grew k by about 22 %. More internal nodes means more "
    "labelings to collapse, so rho is expected to be >= C1 in DIRECTION. A drop "
    "is the alarm: it means the decomposition is not reaching the canonicaliser."
)

R2_BAND_NOTE: str = (
    "Both C1 and Stage D ran the same 43,200 s budget, so unlike the Stage C "
    "smoke there is no budget asymmetry to explain a difference away. A C2 value "
    "materially EXCEEDING C1 is the alarm named in EXECUTION-PLAN section 4.3: it "
    "means the dataset, the split or the metric changed. The band is therefore "
    "two-sided and the direction of every excursion is reported."
)

R2_REFERENCE_LIMIT_NOTE: str = (
    "C1's analysis directory publishes per-problem R2 only as the paired delta "
    "delta_i = mean(R2_isalsr) - mean(R2_baseline) over 30 seeds "
    "(cross_problem_dominance_*.json); absolute per-problem levels are not in any "
    "analysis artefact. The R2 neighbourhood is therefore defined on delta, which "
    "is what the reference actually contains. A change that moved both arms "
    "together would hide in delta, so Stage D's ABSOLUTE R2 per arm is reported "
    "alongside for human reading, explicitly without a machine reference."
)

RHO_RECONSTRUCTION_NOTE: str = (
    "C1's absolute rho_isalsr per problem is reconstructed as 1 + delta_rho, "
    "because the baseline arm performs no canonical dedup and reports rho = 1 "
    "exactly. The reconstruction is not assumed: it is cross-checked at run time "
    "against three_axis_summary's mean_reduction_factor, and the check is "
    "reported. If the cross-check fails the rho comparison degrades to deltas."
)

OVERHEAD_ACCOUNTING_NOTE: str = (
    "Overhead is measured under the NEW accounting introduced by this branch "
    "(F-7/F-8): overhead = canonicalisation + adapter conversion, with the T04 "
    "shadow sketches reported SEPARATELY because they are audit instrumentation, "
    "not method cost. Pre-merge code counted canonicalisation only and therefore "
    "understated wrapper cost by 1.6x-2.4x. The Bingo figure is EXPECTED to come "
    "out ABOVE the old canon-only projection of about 7.4 %. That is an "
    "ACCOUNTING CHANGE, not a regression: the same work was always being done, it "
    "was simply booked as search time. Only a missing or zero T_canon / T_eval "
    "fails this criterion."
)

MANIFEST_SCOPE_NOTE: str = (
    "This is a PRE-FLIGHT manifest, not the campaign manifest: 12 cells, 1 seed, "
    "3 problems, 6 submission splits. It is validated with strict_campaign=False, "
    "which is the mode experiments/models/manifest.py provides for exactly this "
    "case; the strict validator would reject the seed list, the arm cardinality "
    "and the 42-array split table, all of which are campaign constants Stage D "
    "does not and must not satisfy."
)

CONTINUITY_SCOPE_NOTE: str = (
    "Stage D is a 12-cell pre-flight. Its continuity exclusions are declared "
    "CONSERVATIVELY over its own three problems: excluding more problems than "
    "necessary can only weaken a continuity claim, never fabricate one. The "
    "authoritative 22-problem Bingo exclusion list belongs to the campaign "
    "manifest and to EXECUTION-PLAN section 7."
)


# ====================================================================== #
# Observations
# ====================================================================== #


@dataclass
class RssProfile:
    """Parsed ``rss_timeseries.csv`` for one Stage D cell.

    Attributes:
        path: Where the sampler was expected to write.
        present: Whether the file exists.
        error: Non-empty when the file exists but could not be used.
        n_rows: Rows successfully parsed.
        n_bad_rows: Rows skipped because a field would not parse.
        peak_vmhwm_gb: Maximum ``vmhwm_kb`` in GiB, the true high-water mark.
        p50_vmrss_gb: Median resident set size in GiB.
        p95_vmrss_gb: 95th-percentile resident set size in GiB.
        max_vmrss_gb: Maximum sampled resident set size in GiB.
    """

    path: Path
    present: bool = False
    error: str = ""
    n_rows: int = 0
    n_bad_rows: int = 0
    peak_vmhwm_gb: float | None = None
    p50_vmrss_gb: float | None = None
    p95_vmrss_gb: float | None = None
    max_vmrss_gb: float | None = None
    vmrss_gb: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary, excluding the raw sample."""
        return {
            "path": str(self.path),
            "present": self.present,
            "error": self.error,
            "n_rows": self.n_rows,
            "n_bad_rows": self.n_bad_rows,
            "peak_vmhwm_gb": _round(self.peak_vmhwm_gb),
            "p50_vmrss_gb": _round(self.p50_vmrss_gb),
            "p95_vmrss_gb": _round(self.p95_vmrss_gb),
            "max_vmrss_gb": _round(self.max_vmrss_gb),
        }


@dataclass
class Observation:
    """One Stage D registry cell joined to whatever the run left on disk.

    Attributes:
        spec: The locked registry entry. Always present: the registry is the
            expectation, so a cell that never ran still yields an observation.
        cell: The discovered run directory, or ``None`` when absent.
        rss: The sampler profile for the cell.
        sacct_max_rss_gb: MaxRSS from the ``sacct`` export, when joined.
        sacct_elapsed_s: Elapsed seconds from the ``sacct`` export, when the
            export carries an ``Elapsed`` column.
        sacct_key: The key the join matched on, for auditability.
    """

    spec: StageDCell
    cell: Cell | None
    rss: RssProfile
    sacct_max_rss_gb: float | None = None
    sacct_elapsed_s: float | None = None
    sacct_key: str = ""

    @property
    def label(self) -> str:
        """Return the compact ``method/arm/problem`` identifier."""
        return self.spec.label

    @property
    def group_key(self) -> str:
        """Return the ``method/arm`` key the memory recommendation is per."""
        return f"{self.spec.method}/{self.spec.arm}"

    def run_log(self) -> dict[str, Any] | None:
        """Return the raw run-log payload, or ``None`` if it never loaded."""
        return None if self.cell is None else self.cell.run_log_raw


def _round(value: float | None, digits: int = 4) -> float | None:
    """Return ``value`` rounded, passing ``None`` through."""
    return None if value is None else round(float(value), digits)


# ====================================================================== #
# Loading
# ====================================================================== #


def load_rss_profile(directory: Path) -> RssProfile:
    """Parse a cell's ``rss_timeseries.csv``.

    The sampler writes ``timestamp_s,vmrss_kb,vmhwm_kb``. ``vmhwm_kb`` is the
    kernel's own high-water mark and is therefore the true peak regardless of
    the sampling rate, while ``vmrss_kb`` describes the shape of the run.

    Args:
        directory: The cell's seed directory.

    Returns:
        The parsed profile. A missing or malformed file is reported in the
        returned object; nothing is raised.
    """
    path = directory / "rss_timeseries.csv"
    profile = RssProfile(path=path)
    if not path.exists():
        profile.error = f"missing: {path}"
        return profile
    profile.present = True

    hwm: list[float] = []
    rss: list[float] = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    rss.append(float(row["vmrss_kb"]) / 1024**2)
                    hwm.append(float(row["vmhwm_kb"]) / 1024**2)
                except (KeyError, TypeError, ValueError):
                    profile.n_bad_rows += 1
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        profile.error = f"{type(exc).__name__}: {path}"
        return profile

    profile.n_rows = len(rss)
    if not rss:
        profile.error = profile.error or f"no parseable rows: {path}"
        return profile
    profile.vmrss_gb = rss
    profile.peak_vmhwm_gb = max(hwm) if hwm else None
    profile.max_vmrss_gb = max(rss)
    profile.p50_vmrss_gb = _percentile(rss, 50)
    profile.p95_vmrss_gb = _percentile(rss, 95)
    return profile


def collect_observations(root: Path) -> list[Observation]:
    """Join the locked 12-cell registry to the run directories under ``root``.

    Args:
        root: Stage D results root.

    Returns:
        One :class:`Observation` per registry cell, in registry order.
    """
    registry, registry_error = _load_registry()
    if registry_error:
        log.error("Benchmark registry unavailable: %s", registry_error)

    cells = discover_cells(root, registry)
    for cell in cells:
        hydrate(cell)
    # Index on the problem SLUG: when the registry import fails, discover_cells
    # leaves the directory name in place instead of the display name.
    index = {(c.method, c.arm, _slug(c.problem), c.seed): c for c in cells}

    observations: list[Observation] = []
    for spec in STAGE_D_CELLS:
        key = (spec.method, spec.arm, spec.problem_slug, spec.seed)
        cell = index.get(key)
        directory = spec.run_dir(root) if cell is None else cell.directory
        observations.append(Observation(spec=spec, cell=cell, rss=load_rss_profile(directory)))
    return observations


def _parse_elapsed_to_s(raw: str) -> float | None:
    """Convert an ``sacct`` Elapsed field to seconds.

    ``sacct`` renders elapsed time as ``[DD-]HH:MM:SS[.mmm]``.

    Args:
        raw: The raw Elapsed field.

    Returns:
        Seconds, or ``None`` when the field is empty or unparseable.
    """
    text = raw.strip()
    if not text:
        return None
    days = 0.0
    if "-" in text:
        head, _, text = text.partition("-")
        try:
            days = float(head)
        except ValueError:
            return None
    parts = text.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    if len(values) == 2:
        values = [0.0, *values]
    return days * 86_400.0 + values[0] * 3600.0 + values[1] * 60.0 + values[2]


def join_sacct(observations: list[Observation], sacct_csv: Path | None) -> dict[str, Any]:
    """Join an ``sacct`` export onto the observations, in place.

    The join mirrors ``c2_certify.check_c1_11`` exactly: the index is keyed on
    ``f"{slurm_job_id}_{slurm_array_task_id}"`` first, with a ``setdefault`` on
    the bare ``slurm_job_id``, and every incoming ``JobID`` is reduced to its
    step-free stem. See :data:`SACCT_TRAP_NOTE` for why both halves matter.

    Args:
        observations: Observations to annotate.
        sacct_csv: Path to a ``JobID,MaxRSS[,Elapsed]`` CSV, or ``None``.

    Returns:
        A join report: source, matched and unmatched row counts, errors.
    """
    report: dict[str, Any] = {
        "source": "not supplied",
        "path": str(sacct_csv) if sacct_csv else "",
        "n_rows": 0,
        "n_matched": 0,
        "n_unmatched": 0,
        "has_elapsed_column": False,
        "errors": [],
        "trap_note": SACCT_TRAP_NOTE,
    }
    if sacct_csv is None:
        return report
    if not sacct_csv.exists():
        report["source"] = "absent"
        report["errors"].append(f"missing: {sacct_csv}")
        return report

    index: dict[str, Observation] = {}
    for obs in observations:
        status = None if obs.cell is None else obs.cell.status
        if status is None:
            continue
        if status.slurm_job_id and status.slurm_array_task_id:
            index[f"{status.slurm_job_id}_{status.slurm_array_task_id}"] = obs
        if status.slurm_job_id:
            index.setdefault(status.slurm_job_id, obs)

    report["source"] = f"sacct:{sacct_csv}"
    try:
        with sacct_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            report["has_elapsed_column"] = "Elapsed" in (reader.fieldnames or [])
            for raw in reader:
                report["n_rows"] += 1
                stem = str(raw.get("JobID", "")).strip().split(".")[0]
                matched = index.get(stem)
                if matched is None:
                    report["n_unmatched"] += 1
                    continue
                report["n_matched"] += 1
                matched.sacct_key = stem
                gb = _parse_maxrss_to_gb(str(raw.get("MaxRSS", "")))
                if gb is not None:
                    prior = matched.sacct_max_rss_gb
                    matched.sacct_max_rss_gb = gb if prior is None else max(prior, gb)
                elapsed = _parse_elapsed_to_s(str(raw.get("Elapsed", "")))
                if elapsed is not None:
                    matched.sacct_elapsed_s = elapsed
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        report["errors"].append(f"{sacct_csv}: {type(exc).__name__}: {exc}")
    return report


# ====================================================================== #
# D1.1 -- wall-clock headroom
# ====================================================================== #


def _elapsed_of(obs: Observation) -> tuple[float | None, str]:
    """Return ``(elapsed_s, source)`` for one cell.

    ``sacct`` Elapsed is preferred because it is the scheduler's own view of the
    allocation; ``status.json``'s ``wall_clock_s`` is the in-process measurement
    and excludes container start-up.

    Args:
        obs: The observation.

    Returns:
        The elapsed time in seconds and the source it came from.
    """
    if obs.sacct_elapsed_s is not None:
        return obs.sacct_elapsed_s, "sacct:Elapsed"
    status = None if obs.cell is None else obs.cell.status
    if status is not None and _is_finite(status.wall_clock_s):
        return float(status.wall_clock_s), "status.json:wall_clock_s"
    return None, "unavailable"


def check_d1_1(observations: list[Observation], wall_s: float) -> CriterionResult:
    """D1.1 -- 12/12 cells complete inside the wall with >= 10 % headroom."""
    rows: list[dict[str, Any]] = []
    absent: list[str] = []
    incomplete: list[str] = []
    no_elapsed: list[str] = []
    thin: list[str] = []
    headrooms: list[float] = []

    for obs in observations:
        if obs.cell is None:
            absent.append(f"{obs.label} (no run directory at {obs.spec.run_dir(Path('.'))})")
            continue
        status = obs.cell.status
        if status is None or status.terminal_status != "completed" or status.exit_code != 0:
            state = (
                "no status.json"
                if status is None
                else (f"status={status.terminal_status} exit={status.exit_code}")
            )
            incomplete.append(f"{obs.label} {state}")
        elapsed, source = _elapsed_of(obs)
        if elapsed is None:
            no_elapsed.append(f"{obs.label} (neither sacct Elapsed nor wall_clock_s)")
            continue
        headroom = (wall_s - elapsed) / wall_s
        headrooms.append(headroom)
        rows.append(
            {
                "cell": obs.label,
                "elapsed_s": round(elapsed, 1),
                "elapsed_source": source,
                "headroom_frac": round(headroom, 4),
                "within_headroom": headroom >= MIN_WALL_HEADROOM_FRAC,
            }
        )
        if headroom < MIN_WALL_HEADROOM_FRAC:
            thin.append(
                f"{obs.label} elapsed={elapsed:.0f}s headroom={headroom:.1%} "
                f"< {MIN_WALL_HEADROOM_FRAC:.0%}"
            )

    n_expected = len(observations)
    ok = not absent and not incomplete and not no_elapsed and not thin and n_expected > 0
    min_headroom = min(headrooms) if headrooms else None
    limit = wall_s * (1.0 - MIN_WALL_HEADROOM_FRAC)
    return CriterionResult(
        id="D1.1",
        title="12/12 cells complete within the SLURM wall with >= 10 % headroom",
        status=_verdict(ok),
        expected=(
            f"{n_expected}/{n_expected} complete; elapsed <= {limit:.0f} s of a {wall_s:.0f} s wall"
        ),
        observed=(
            f"{len(rows)}/{n_expected} timed; min headroom "
            f"{'n/a' if min_headroom is None else f'{min_headroom:.1%}'}"
        ),
        detail={
            "wall_s": wall_s,
            "search_budget_s": STAGE_D_MAX_TIME_S,
            "min_headroom_frac": _round(min_headroom),
            "headroom_limit_s": round(limit, 1),
            "margin_rationale": WALL_MARGIN_NOTE,
            "per_cell": rows,
            "no_run_directory": _truncate(absent),
            "not_completed": _truncate(incomplete),
            "elapsed_unavailable": _truncate(no_elapsed),
            "headroom_below_threshold": _truncate(thin),
        },
    )


# ====================================================================== #
# D1.2 -- memory, and the production recommendation
# ====================================================================== #


def _round_up_gb(value: float, step: int = MEM_ROUND_STEP_GB) -> int:
    """Round a memory figure UP to the next multiple of ``step`` GiB.

    Args:
        value: Raw recommendation in GiB.
        step: Granularity in GiB.

    Returns:
        The rounded request, never below ``step``.
    """
    if not math.isfinite(value) or value <= 0:
        return step
    return max(step, int(math.ceil(value / step)) * step)


def _peak_of(obs: Observation) -> tuple[float | None, str]:
    """Return the observed peak RSS in GiB and the source that won.

    The peak is the MAXIMUM of the ``sacct`` MaxRSS and the sampler's
    ``vmhwm_kb`` high-water mark. They can disagree in both directions: sacct
    polls the cgroup and can miss a short spike between polls, while the sampler
    stops when the process exits and can miss teardown.

    Args:
        obs: The observation.

    Returns:
        ``(peak_gb, source)``; ``(None, "unavailable")`` when neither exists.
    """
    candidates: list[tuple[float, str]] = []
    if obs.sacct_max_rss_gb is not None:
        candidates.append((obs.sacct_max_rss_gb, "sacct:MaxRSS"))
    if obs.rss.peak_vmhwm_gb is not None:
        candidates.append((obs.rss.peak_vmhwm_gb, "rss_timeseries:vmhwm_kb"))
    if not candidates:
        return None, "unavailable"
    peak, source = max(candidates, key=lambda pair: pair[0])
    return peak, source


def _memory_recommendation(group: str, members: list[Observation]) -> dict[str, Any]:
    """Build the production ``--mem`` recommendation for one ``(method, arm)``.

    Args:
        group: The ``method/arm`` key.
        members: Observations in that group.

    Returns:
        A row carrying the peak, its source, the sampled distribution, the
        request that was made and the request that should be made.
    """
    peaks = [(_peak_of(obs), obs.label) for obs in members]
    known = [(p, s, label) for (p, s), label in peaks if p is not None]
    requested = max(obs.spec.mem_gb for obs in members)
    pooled = [v for obs in members for v in obs.rss.vmrss_gb]

    if not known:
        return {
            "group": group,
            "requested_gb": requested,
            "peak_gb": None,
            "peak_source": "unavailable",
            "peak_cell": "",
            "vmrss_p50_gb": _round(_percentile(pooled, 50)),
            "vmrss_p95_gb": _round(_percentile(pooled, 95)),
            "headroom_frac": None,
            "recommended_gb": None,
            "margin_frac": None,
            "note": "no MaxRSS and no rss_timeseries for any cell in this group",
        }

    peak, source, cell_label = max(known, key=lambda t: t[0])
    recommended = _round_up_gb(peak / MEM_TARGET_UTILISATION)
    return {
        "group": group,
        "requested_gb": requested,
        "peak_gb": _round(peak),
        "peak_source": source,
        "peak_cell": cell_label,
        "vmrss_p50_gb": _round(_percentile(pooled, 50)),
        "vmrss_p95_gb": _round(_percentile(pooled, 95)),
        "n_rss_samples": len(pooled),
        "headroom_frac": _round((requested - peak) / requested),
        "recommended_gb": recommended,
        "margin_frac": _round((recommended - peak) / recommended),
    }


def check_d1_2(observations: list[Observation], sacct_report: dict[str, Any]) -> CriterionResult:
    """D1.2 -- peak memory inside the request with >= 30 % headroom, plus the
    production recommendation per ``(method, arm)``.

    Part (a), the headroom, is blocking. Part (b), the recommendation, is
    presented unconditionally: it is the answer to "how low can production
    request without OOM risk", and it is wanted even when (a) fails.
    """
    per_cell: list[dict[str, Any]] = []
    thin: list[str] = []
    unmeasured: list[str] = []
    missing_timeseries: list[str] = []

    for obs in observations:
        if not obs.rss.present or obs.rss.error:
            missing_timeseries.append(f"{obs.label}: {obs.rss.error or 'unusable'}")
        peak, source = _peak_of(obs)
        requested = float(obs.spec.mem_gb)
        if peak is None:
            unmeasured.append(f"{obs.label} (no sacct MaxRSS, no rss_timeseries)")
            continue
        headroom = (requested - peak) / requested
        per_cell.append(
            {
                "cell": obs.label,
                "requested_gb": obs.spec.mem_gb,
                "peak_gb": _round(peak),
                "peak_source": source,
                "sacct_max_rss_gb": _round(obs.sacct_max_rss_gb),
                "vmhwm_peak_gb": _round(obs.rss.peak_vmhwm_gb),
                "vmrss_p50_gb": _round(obs.rss.p50_vmrss_gb),
                "vmrss_p95_gb": _round(obs.rss.p95_vmrss_gb),
                "headroom_frac": _round(headroom),
                "within_headroom": headroom >= MIN_MEM_HEADROOM_FRAC,
            }
        )
        if headroom < MIN_MEM_HEADROOM_FRAC:
            thin.append(
                f"{obs.label} peak={peak:.2f} GB of {requested:.0f} GB "
                f"requested, headroom={headroom:.1%} < {MIN_MEM_HEADROOM_FRAC:.0%}"
            )

    groups: dict[str, list[Observation]] = {}
    for obs in observations:
        groups.setdefault(obs.group_key, []).append(obs)
    recommendations = [_memory_recommendation(g, m) for g, m in sorted(groups.items())]

    ok = bool(per_cell) and not thin and not unmeasured
    headrooms = [row["headroom_frac"] for row in per_cell if row["headroom_frac"] is not None]
    return CriterionResult(
        id="D1.2",
        title="peak RSS within --mem with >= 30 % headroom; production --mem recommendation",
        status=_verdict(ok),
        expected=f"headroom >= {MIN_MEM_HEADROOM_FRAC:.0%} on {len(observations)} cells",
        observed=(
            f"{len(per_cell)}/{len(observations)} measured; min headroom {min(headrooms):.1%}"
            if headrooms
            else "0 cells measured"
        ),
        detail={
            "part_a_rule": (
                f"BLOCKING: (requested - peak) / requested >= {MIN_MEM_HEADROOM_FRAC:.0%}, "
                "with peak = max(sacct MaxRSS, max vmhwm_kb in rss_timeseries.csv)."
            ),
            "part_b_rule": (
                f"ADVISORY OUTPUT: recommended_gb = ceil_to_{MEM_ROUND_STEP_GB}GB("
                f"peak / {MEM_TARGET_UTILISATION:.2f}). Sizing for "
                f"{MEM_TARGET_UTILISATION:.0%} utilisation carries the same "
                f"{MIN_MEM_HEADROOM_FRAC:.0%} headroom into production; rounding is "
                f"UP to the next multiple of {MEM_ROUND_STEP_GB} GiB because rounding "
                "a memory request down is an OOM."
            ),
            "sacct_join": sacct_report,
            "production_recommendation_by_method_arm": recommendations,
            "per_cell": per_cell,
            "headroom_below_threshold": _truncate(thin),
            "unmeasured_cells": _truncate(unmeasured),
            "missing_or_unusable_rss_timeseries": _truncate(missing_timeseries),
        },
    )


# ====================================================================== #
# D1.3 -- artefact completeness
# ====================================================================== #


def _ledger_keys(root: Path) -> tuple[set[tuple[str, str, str, int]], str]:
    """Read ``status_ledger.csv`` and return its ``(method, arm, problem, seed)`` keys.

    Args:
        root: Stage D results root.

    Returns:
        ``(keys, error)``; ``error`` is non-empty when the ledger is absent or
        unreadable, in which case ``keys`` is empty.
    """
    path = root / "status_ledger.csv"
    if not path.exists():
        return set(), f"missing: {path}"
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return set(), f"{type(exc).__name__}: {path}"
    keys: set[tuple[str, str, str, int]] = set()
    for row in rows:
        seed = str(row.get("seed", "")).strip()
        if seed.isdigit():
            keys.add(
                (
                    str(row.get("method", "")),
                    str(row.get("arm", "")),
                    str(row.get("problem", "")),
                    int(seed),
                )
            )
    return keys, ""


def _spec_violations(payload: dict[str, Any]) -> list[str]:
    """Return the ``RUN_LOG_FIELD_SPEC`` violations of one run-log payload.

    Args:
        payload: Decoded ``run_log.json``.

    Returns:
        One string per missing field or type mismatch.
    """
    problems: list[str] = []
    for path, types, nullable in RUN_LOG_FIELD_SPEC:
        dotted = ".".join(path)
        present, value = _walk_spec(payload, path)
        if not present:
            problems.append(f"missing {dotted}")
            continue
        if value is None:
            if not nullable:
                problems.append(f"null {dotted}")
            continue
        if isinstance(value, bool) and bool not in types:
            problems.append(f"bad type {dotted}=bool")
            continue
        if not isinstance(value, types):
            problems.append(f"bad type {dotted}={type(value).__name__}")
    return problems


def check_d1_3(root: Path, observations: list[Observation]) -> CriterionResult:
    """D1.3 -- run_log.json valid, trajectory.csv non-empty, ledger row present."""
    ledger, ledger_error = _ledger_keys(root)
    bad_run_log: list[str] = []
    bad_trajectory: list[str] = []
    missing_ledger: list[str] = []
    n_clean = 0

    for obs in observations:
        cell_ok = True
        payload = obs.run_log()
        if payload is None:
            reason = "no run directory" if obs.cell is None else obs.cell.run_log_error
            bad_run_log.append(f"{obs.label}: {reason}")
            cell_ok = False
        else:
            problems = _spec_violations(payload)
            if problems:
                bad_run_log.append(f"{obs.label}: {'; '.join(problems[:6])}")
                cell_ok = False

        directory = obs.spec.run_dir(root) if obs.cell is None else obs.cell.directory
        trajectory = directory / "trajectory.csv"
        if not trajectory.exists():
            bad_trajectory.append(f"{obs.label}: missing {trajectory}")
            cell_ok = False
        else:
            try:
                with trajectory.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            except (OSError, UnicodeDecodeError, csv.Error) as exc:
                bad_trajectory.append(f"{obs.label}: {type(exc).__name__}")
                rows = []
                cell_ok = False
            if not rows:
                bad_trajectory.append(f"{obs.label}: empty {trajectory}")
                cell_ok = False

        key = (obs.spec.method, obs.spec.arm, obs.spec.problem, obs.spec.seed)
        if key not in ledger:
            missing_ledger.append(f"{obs.label}/seed_{obs.spec.seed}")
            cell_ok = False
        if cell_ok:
            n_clean += 1

    ok = n_clean == len(observations) and len(observations) > 0 and not ledger_error
    return CriterionResult(
        id="D1.3",
        title="run_log.json (60-field spec), trajectory.csv and a ledger row on 12/12",
        status=_verdict(ok),
        expected=f"{len(observations)}/{len(observations)} complete artefact sets",
        observed=f"{n_clean}/{len(observations)}",
        detail={
            "n_spec_fields": len(RUN_LOG_FIELD_SPEC),
            "spec_note": (
                "The 60-field spec now includes results.time.conversion_time_s and "
                "results.time.shadow_time_s; an artefact predating the fairness "
                "audit lacks both and is reported as missing, which is the "
                "intended reading since its search time is inflated by exactly "
                "those two quantities."
            ),
            "status_ledger_error": ledger_error,
            "n_ledger_rows": len(ledger),
            "run_log_violations": _truncate(bad_run_log),
            "trajectory_violations": _truncate(bad_trajectory),
            "cells_absent_from_status_ledger": _truncate(missing_ledger),
        },
    )


# ====================================================================== #
# D1.4 -- the T08 AC-7 evidence
# ====================================================================== #


def check_d1_4(observations: list[Observation]) -> CriterionResult:
    """D1.4 -- Korns-12 and Vladislavleva-2, Bingo-isalsr: finite R2.

    These are the two cells that were NaN in the submission. If NaN recurs, the
    root cause is still live and would reproduce at 8,400-run scale, so C2 does
    not launch.
    """
    targets = [
        obs
        for obs in observations
        if obs.spec.method == "bingo"
        and obs.spec.arm == "isalsr"
        and obs.spec.problem in NAN_PROBLEMS
    ]
    rows: list[dict[str, Any]] = []
    nonfinite: list[str] = []
    unavailable: list[str] = []

    for obs in targets:
        payload = obs.run_log()
        if payload is None:
            unavailable.append(f"{obs.label}: no run log")
            continue
        regression = obs.cell.regression if obs.cell is not None else {}
        row: dict[str, Any] = {"cell": obs.label}
        for metric in ("r2_test", "r2_train"):
            value = regression.get(metric)
            finite = _is_finite(value)
            row[metric] = value if finite else repr(value)
            row[f"{metric}_finite"] = finite
            if not finite:
                nonfinite.append(f"{obs.label} {metric}={value!r}")
        rows.append(row)

    ok = bool(targets) and not nonfinite and not unavailable
    return CriterionResult(
        id="D1.4",
        title="Korns-12 and Vladislavleva-2, Bingo-isalsr: finite r2_test and r2_train",
        status=_verdict(ok),
        expected=f"{len(targets)}/{len(targets)} finite (T08 AC-7 evidence)",
        observed=f"{len(rows) - len({n.split()[0] for n in nonfinite})}/{len(targets)} finite",
        detail={
            "premise_note": (
                "These two cells were NaN in the submission. A recurrence means "
                "the T08 root cause is still live and C2 does not launch."
            ),
            "problems": list(NAN_PROBLEMS),
            "per_cell": rows,
            "non_finite": _truncate(nonfinite),
            "unavailable": _truncate(unavailable),
        },
    )


# ====================================================================== #
# D1.5 -- rho ordering
# ====================================================================== #


def _rho_of(obs: Observation) -> float | None:
    """Return the cell's empirical reduction factor, or ``None``."""
    if obs.cell is None:
        return None
    value = obs.cell.search_space.get("empirical_reduction_factor")
    return float(value) if _is_finite(value) else None


def check_d1_5(observations: list[Observation]) -> CriterionResult:
    """D1.5 -- ``rho_hash <= rho_isalsr`` on every matched (method, problem).

    The canonical key is coarser than the fixed-order hash key by construction,
    so the canonical arm cannot report fewer collisions. A violation means the
    two arms did not see the same candidate stream. Stage D runs one seed, so
    the match is on ``(method, problem)`` rather than on a triple.
    """
    rho: dict[tuple[str, str, str], float] = {}
    for obs in observations:
        value = _rho_of(obs)
        if value is not None:
            rho[(obs.spec.method, obs.spec.arm, obs.spec.problem)] = value

    rows: list[dict[str, Any]] = []
    violations: list[str] = []
    unmatched: list[str] = []
    for (method, arm, problem), value in sorted(rho.items()):
        if arm != "hash":
            continue
        other = rho.get((method, "isalsr", problem))
        if other is None:
            unmatched.append(f"{method}/{problem}: no isalsr rho to compare against")
            continue
        rows.append(
            {
                "method": method,
                "problem": problem,
                "rho_hash": round(value, 6),
                "rho_isalsr": round(other, 6),
                "ordered": value <= other,
            }
        )
        if value > other:
            violations.append(f"{method}/{problem} rho_hash={value:.6f} > rho_isalsr={other:.6f}")

    ok = bool(rows) and not violations and not unmatched
    return CriterionResult(
        id="D1.5",
        title="rho_hash <= rho_isalsr on all matched (method, problem) cells",
        status=_verdict(ok),
        expected="0 violations on every matched pair",
        observed=f"{len(violations)}/{len(rows)} violations",
        detail={
            "premise_note": (
                "The canonical key is coarser than the fixed-order hash key, so "
                "the canonical arm cannot report fewer collisions. A violation "
                "means the arms did not see the same candidate stream."
            ),
            "per_pair": rows,
            "violations": _truncate(violations),
            "unmatched": _truncate(unmatched),
        },
    )


# ====================================================================== #
# D1.6 -- the C1 neighbourhood
# ====================================================================== #


@dataclass
class C1Reference:
    """Campaign C1's published values for the Stage D problems.

    Attributes:
        path: The analysis directory the values were read from.
        available: Whether anything usable was loaded.
        errors: Every file that could not be read, named.
        rho_isalsr: ``(method, problem) -> `` reconstructed absolute
            ``rho_isalsr``.
        rho_delta: ``(method, problem) -> `` published ``delta_rho``.
        r2_delta: ``(method, problem) -> `` published ``delta_R2_test``.
        reconstruction: The cross-check that validates ``rho_isalsr``.
    """

    path: Path
    available: bool = False
    errors: list[str] = field(default_factory=list)
    rho_isalsr: dict[tuple[str, str], float] = field(default_factory=dict)
    rho_delta: dict[tuple[str, str], float] = field(default_factory=dict)
    r2_delta: dict[tuple[str, str], float] = field(default_factory=dict)
    reconstruction: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    """Read a JSON object, returning ``(payload, error)`` and never raising."""
    if not path.exists():
        return None, f"missing: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, f"{type(exc).__name__}: {path}"
    return (payload, "") if isinstance(payload, dict) else (None, f"not an object: {path}")


def _paired_deltas(block: Any) -> dict[str, float]:
    """Zip a CPDT block's ``problem_names`` and ``problem_deltas``."""
    if not isinstance(block, dict):
        return {}
    names = block.get("problem_names")
    deltas = block.get("problem_deltas")
    if not isinstance(names, list) or not isinstance(deltas, list):
        return {}
    return {str(n): float(d) for n, d in zip(names, deltas, strict=False) if _is_finite(d)}


def load_c1_reference(path: Path | None) -> C1Reference:
    """Load campaign C1's per-problem values for the Stage D comparison.

    C1's analysis directory publishes per-problem quantities only as paired
    deltas in ``cross_problem_dominance_{method}_benchmark.json``. The absolute
    ``rho_isalsr`` is recovered as ``1 + delta_rho`` because the baseline arm
    performs no canonical dedup and reports ``rho = 1`` exactly; the recovery is
    cross-checked against ``three_axis_summary``'s cohort ``mean_reduction_factor``
    and the check is reported rather than assumed.

    Args:
        path: The C1 analysis directory, or ``None`` to skip.

    Returns:
        The reference. A missing directory yields ``available=False`` with the
        reason recorded; nothing is raised.
    """
    reference = C1Reference(path=path or Path(""))
    if path is None:
        reference.errors.append("--c1-reference not supplied")
        return reference
    if not path.is_dir():
        reference.errors.append(f"not a directory: {path}")
        return reference

    for method in ("bingo", "udfs"):
        cpdt, error = _read_json(path / f"cross_problem_dominance_{method}_benchmark.json")
        if cpdt is None:
            reference.errors.append(error)
            continue
        rho_deltas = _paired_deltas(cpdt.get("empirical_reduction_factor"))
        r2_deltas = _paired_deltas(cpdt.get("r2_test"))
        for problem, delta in rho_deltas.items():
            reference.rho_delta[(method, problem)] = delta
            reference.rho_isalsr[(method, problem)] = 1.0 + delta
        for problem, delta in r2_deltas.items():
            reference.r2_delta[(method, problem)] = delta

        summary, error = _read_json(path / f"three_axis_summary_{method}_benchmark.json")
        cohort_mean = None
        if summary is None:
            reference.errors.append(error)
        else:
            value = summary.get("search_space", {}).get("mean_reduction_factor")
            cohort_mean = float(value) if _is_finite(value) else None
        rebuilt = (
            sum(1.0 + d for d in rho_deltas.values()) / len(rho_deltas) if rho_deltas else None
        )
        gap = (
            abs(rebuilt - cohort_mean) if rebuilt is not None and cohort_mean is not None else None
        )
        reference.reconstruction[method] = {
            "n_problems": len(rho_deltas),
            "mean_of_1_plus_delta": _round(rebuilt, 6),
            "published_mean_reduction_factor": _round(cohort_mean, 6),
            "abs_gap": _round(gap, 8),
            "validated": gap is not None and gap <= RHO_RECONSTRUCTION_TOL,
        }

    reference.available = bool(reference.rho_isalsr or reference.r2_delta)
    return reference


def _c2_r2_delta(
    observations: list[Observation], method: str, problem: str
) -> tuple[float | None, dict[str, Any]]:
    """Return Stage D's ``r2_test`` isalsr-minus-baseline delta for one cell pair.

    Args:
        observations: All Stage D observations.
        method: Host method.
        problem: Problem name.

    Returns:
        ``(delta, levels)``; ``delta`` is ``None`` when either arm is absent or
        non-finite, and ``levels`` always carries whatever was found.
    """
    levels: dict[str, Any] = {}
    for arm in ("baseline", "isalsr"):
        match = [
            obs
            for obs in observations
            if obs.spec.method == method and obs.spec.problem == problem and obs.spec.arm == arm
        ]
        value = None
        if match and match[0].cell is not None:
            raw = match[0].cell.regression.get("r2_test")
            value = float(raw) if _is_finite(raw) else None
        levels[f"r2_test_{arm}"] = _round(value, 6)
    baseline = levels["r2_test_baseline"]
    isalsr = levels["r2_test_isalsr"]
    if baseline is None or isalsr is None:
        return None, levels
    return isalsr - baseline, levels


def _rho_comparison(obs: Observation, reference: C1Reference) -> tuple[dict[str, Any], str | None]:
    """Compare one isalsr cell's rho against C1 and explain the difference.

    Args:
        obs: The isalsr observation.
        reference: Loaded C1 reference.

    Returns:
        ``(row, failure)``; ``failure`` is ``None`` when the cell is inside the
        band or has no reference to compare against.
    """
    key = (obs.spec.method, obs.spec.problem)
    rho_c2 = _rho_of(obs)
    rho_c1 = reference.rho_isalsr.get(key)
    row: dict[str, Any] = {
        "method": obs.spec.method,
        "problem": obs.spec.problem,
        "rho_c2": _round(rho_c2, 6),
        "rho_c1": _round(rho_c1, 6),
    }
    if rho_c2 is None or rho_c1 is None or rho_c1 <= 0:
        row["verdict"] = "no comparison"
        row["explanation"] = (
            "Stage D rho unavailable" if rho_c2 is None else "no C1 value for this cell"
        )
        return row, None

    ratio = rho_c2 / rho_c1
    floor = rho_c1 * (1.0 - RHO_DROP_TOL)
    row["ratio_c2_over_c1"] = round(ratio, 4)
    row["drop_floor"] = _round(floor, 6)
    if rho_c2 >= rho_c1:
        row["verdict"] = "PASS (rose or held)"
        row["explanation"] = (
            f"rho rose by {ratio - 1.0:+.1%}, the direction T16 predicts: the "
            "decomposition grew k by about 22 %, so there are more internal "
            "nodes to permute and more labelings to collapse."
        )
        return row, None
    if rho_c2 >= floor:
        row["verdict"] = "PASS (inside the drop tolerance)"
        row["explanation"] = (
            f"rho fell by {1.0 - ratio:.1%}, inside the {RHO_DROP_TOL:.0%} "
            "single-seed tolerance. Stage D runs one seed against C1's 30-seed "
            "mean, so a fall of this size is sampling, not signal."
        )
        return row, None
    row["verdict"] = "FAIL (drop beyond tolerance)"
    row["explanation"] = (
        f"rho fell by {1.0 - ratio:.1%}, beyond the {RHO_DROP_TOL:.0%} tolerance. "
        "This is the alarm: T16 predicts a RISE, so a drop means either the "
        "decomposition is not reaching the canonicaliser or the dedup population "
        "changed. It is NOT the engine: see engine_note."
    )
    return row, f"{obs.label} rho {rho_c2:.4f} < C1 {rho_c1:.4f} x (1 - {RHO_DROP_TOL})"


def _r2_comparison(
    observations: list[Observation], reference: C1Reference, method: str, problem: str
) -> tuple[dict[str, Any], str | None]:
    """Compare one (method, problem)'s R2 paired delta against C1.

    Args:
        observations: All Stage D observations.
        reference: Loaded C1 reference.
        method: Host method.
        problem: Problem name.

    Returns:
        ``(row, failure)``; ``failure`` is ``None`` when inside the band.
    """
    delta_c2, levels = _c2_r2_delta(observations, method, problem)
    delta_c1 = reference.r2_delta.get((method, problem))
    row: dict[str, Any] = {
        "method": method,
        "problem": problem,
        "delta_r2_test_c2": _round(delta_c2, 6),
        "delta_r2_test_c1": _round(delta_c1, 6),
        **levels,
    }
    if delta_c2 is None or delta_c1 is None:
        row["verdict"] = "no comparison"
        row["explanation"] = (
            "Stage D delta unavailable" if delta_c2 is None else "no C1 delta for this cell"
        )
        return row, None

    excess = delta_c2 - delta_c1
    row["excess"] = round(excess, 6)
    row["band"] = R2_DELTA_BAND
    if abs(excess) <= R2_DELTA_BAND:
        row["verdict"] = "PASS (inside band)"
        row["explanation"] = (
            f"the Stage D paired delta differs from C1's 30-seed mean by "
            f"{excess:+.4f}, inside the +/-{R2_DELTA_BAND:.2f} single-seed band."
        )
        return row, None
    if excess > 0:
        row["verdict"] = "FAIL (C2 materially exceeds C1)"
        row["explanation"] = (
            f"the Stage D delta EXCEEDS C1 by {excess:+.4f} at the same 12 h "
            "budget. There is no budget asymmetry to absorb this, so the suspects "
            "are the dataset, the split and the metric -- not the engine."
        )
    else:
        row["verdict"] = "FAIL (C2 materially below C1)"
        row["explanation"] = (
            f"the Stage D delta falls BELOW C1 by {excess:+.4f}. At one seed the "
            "first suspect is sampling, but the excursion is outside the band "
            "chosen to absorb exactly that, so the arm must be examined."
        )
    return row, f"{method}/{problem} excess={excess:+.4f} outside +/-{R2_DELTA_BAND}"


def check_d1_6(observations: list[Observation], reference: C1Reference) -> CriterionResult:
    """D1.6 -- the 12 h rho and R2 land in a defensible neighbourhood of C1."""
    definition = {
        "rho_rule": (
            f"BLOCKING, one-sided: rho_C2 >= rho_C1 x (1 - {RHO_DROP_TOL:.2f}). "
            "There is deliberately NO upper bound."
        ),
        "rho_rationale": T16_NOTE,
        "rho_reconstruction": RHO_RECONSTRUCTION_NOTE,
        "r2_rule": (
            f"BLOCKING, two-sided with direction flagged: "
            f"|delta_C2 - delta_C1| <= {R2_DELTA_BAND:.2f} in absolute R2 units, "
            "where delta = mean(R2_isalsr) - mean(R2_baseline) on r2_test."
        ),
        "r2_rationale": R2_BAND_NOTE,
        "r2_reference_limitation": R2_REFERENCE_LIMIT_NOTE,
        "engine_note": ENGINE_NOTE,
    }

    if not reference.available:
        return CriterionResult(
            id="D1.6",
            title="12 h rho and R2 inside a defensible neighbourhood of campaign C1",
            status="SKIP",
            expected=f"C1 analysis artefacts under {reference.path}",
            observed="reference unavailable",
            detail={
                "neighbourhood_definition": definition,
                "reference_path": str(reference.path),
                "errors": reference.errors,
                "skip_note": (
                    "Non-blocking SKIP: the comparison is unavailable, which is a "
                    "gap in the evidence, not a failure of the run."
                ),
            },
            blocking=False,
        )

    rho_rows: list[dict[str, Any]] = []
    rho_failures: list[str] = []
    for obs in observations:
        if obs.spec.arm != "isalsr":
            continue
        row, failure = _rho_comparison(obs, reference)
        rho_rows.append(row)
        if failure:
            rho_failures.append(failure)

    seen: set[tuple[str, str]] = set()
    r2_rows: list[dict[str, Any]] = []
    r2_failures: list[str] = []
    for obs in observations:
        key = (obs.spec.method, obs.spec.problem)
        if key in seen:
            continue
        seen.add(key)
        row, failure = _r2_comparison(observations, reference, *key)
        r2_rows.append(row)
        if failure:
            r2_failures.append(failure)

    n_compared = sum(1 for r in rho_rows if "ratio_c2_over_c1" in r) + sum(
        1 for r in r2_rows if "excess" in r
    )
    ok = not rho_failures and not r2_failures and n_compared > 0
    return CriterionResult(
        id="D1.6",
        title="12 h rho and R2 inside a defensible neighbourhood of campaign C1",
        status=_verdict(ok),
        expected=f"0 excursions on {len(rho_rows)} rho and {len(r2_rows)} R2 comparisons",
        observed=(
            f"{len(rho_failures)} rho excursion(s), {len(r2_failures)} R2 excursion(s), "
            f"{n_compared} comparison(s) made"
        ),
        detail={
            "neighbourhood_definition": definition,
            "reference_path": str(reference.path),
            "reference_errors": reference.errors,
            "rho_reconstruction_crosscheck": reference.reconstruction,
            "rho_comparisons": rho_rows,
            "r2_comparisons": r2_rows,
            "rho_excursions": _truncate(rho_failures),
            "r2_excursions": _truncate(r2_failures),
        },
    )


# ====================================================================== #
# D1.7 -- per-DAG timing under the new accounting
# ====================================================================== #


def check_d1_7(observations: list[Observation]) -> CriterionResult:
    """D1.7 -- ``T_canon`` and ``T_eval`` present, overhead percentage computable.

    Overhead is ``canonicalisation + conversion``; the shadow sketches are
    reported separately. Only a missing or zero ``T_canon`` (on a dedup arm) or
    ``T_eval`` (anywhere) fails this criterion -- the magnitude is disclosed, not
    graded, because the accounting changed underneath it.
    """
    rows: list[dict[str, Any]] = []
    missing_fields: list[str] = []
    zero_canon: list[str] = []
    zero_eval: list[str] = []

    for obs in observations:
        if obs.cell is None or obs.cell.run_log_raw is None:
            missing_fields.append(f"{obs.label}: no run log")
            continue
        time_block = obs.cell.time
        values: dict[str, float | None] = {}
        for name in (
            "canonicalization_runtime_s",
            "conversion_time_s",
            "evaluation_time_s",
            "shadow_time_s",
            "overhead_time_s",
        ):
            raw = time_block.get(name)
            if not _is_finite(raw):
                missing_fields.append(f"{obs.label}: {name}={raw!r}")
                values[name] = None
            else:
                values[name] = float(raw)

        canon = values["canonicalization_runtime_s"]
        conversion = values["conversion_time_s"]
        evaluation = values["evaluation_time_s"]
        shadow = values["shadow_time_s"]
        reported = values["overhead_time_s"]

        dedup = obs.spec.arm in DEDUP_ARMS
        if dedup and canon is not None and canon <= 0.0:
            zero_canon.append(f"{obs.label} canonicalization_runtime_s=0")
        if evaluation is not None and evaluation <= 0.0:
            zero_eval.append(f"{obs.label} evaluation_time_s=0")

        row: dict[str, Any] = {
            "cell": obs.label,
            "T_canon_s": _round(canon, 4),
            "T_conversion_s": _round(conversion, 4),
            "T_eval_s": _round(evaluation, 4),
            "T_shadow_s": _round(shadow, 4),
            "overhead_time_s_reported": _round(reported, 4),
        }
        if canon is not None and conversion is not None:
            computed = canon + conversion
            row["overhead_s_computed"] = _round(computed, 4)
            row["reported_matches_computed"] = reported is not None and abs(
                reported - computed
            ) <= 1e-6 * max(1.0, computed)
            if evaluation and evaluation > 0.0:
                row["overhead_pct_of_eval"] = round(100.0 * computed / evaluation, 3)
                row["canon_only_pct_of_eval"] = round(100.0 * canon / evaluation, 3)
                if shadow is not None:
                    row["shadow_pct_of_eval"] = round(100.0 * shadow / evaluation, 3)
        rows.append(row)

    by_method: dict[str, list[float]] = {}
    for row in rows:
        pct = row.get("overhead_pct_of_eval")
        if pct is not None:
            by_method.setdefault(row["cell"].split("/")[0], []).append(float(pct))

    ok = bool(rows) and not missing_fields and not zero_canon and not zero_eval
    return CriterionResult(
        id="D1.7",
        title="per-DAG T_canon and T_eval present; overhead % computable (new accounting)",
        status=_verdict(ok),
        expected=f"{len(observations)}/{len(observations)} with T_canon and T_eval > 0",
        observed=f"{len(rows)} cells timed; {len(missing_fields)} field problem(s)",
        detail={
            "accounting_note": OVERHEAD_ACCOUNTING_NOTE,
            "overhead_definition": (
                "overhead_pct_of_eval = 100 x (canonicalization_runtime_s + "
                "conversion_time_s) / evaluation_time_s. shadow_time_s is reported "
                "separately and is NOT part of overhead."
            ),
            "overhead_pct_of_eval_by_method": {
                method: _summarise(values) for method, values in sorted(by_method.items())
            },
            "per_cell": rows,
            "missing_or_nonfinite_fields": _truncate(missing_fields),
            "zero_T_canon_on_dedup_arm": _truncate(zero_canon),
            "zero_T_eval": _truncate(zero_eval),
        },
    )


# ====================================================================== #
# D1.8 -- the pre-flight MANIFEST
# ====================================================================== #


def _git(args: list[str]) -> str:
    """Run a git command in the repository and return stdout, or ``""``."""
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _build_provenance() -> tuple[Any, list[str]]:
    """Assemble the manifest's build block from the live environment.

    Returns:
        ``(BuildProvenance, notes)``. Fields that cannot be resolved are left
        empty so :func:`validate_manifest` reports them, rather than being
        filled with a plausible-looking default.
    """
    from experiments.models.manifest import BuildProvenance

    notes: list[str] = []
    commit = _git(["rev-parse", "HEAD"])
    if not commit:
        notes.append("git rev-parse HEAD failed; build.git_commit left empty")
    tag = _git(["describe", "--tags", "--exact-match"]) or _git(
        ["rev-parse", "--abbrev-ref", "HEAD"]
    )
    dirty = bool(_git(["status", "--porcelain"]))

    info: dict[str, str] = {}
    module_path = ""
    module_mtime = ""
    try:
        from isalsr.core import backends

        info = backends.build_info()
        if info.get("engine") == "cpp":
            from isalsr.core import _native  # type: ignore[attr-defined]

            module_path = str(getattr(_native, "__file__", ""))
            if module_path and Path(module_path).exists():
                module_mtime = datetime.fromtimestamp(
                    Path(module_path).stat().st_mtime, tz=UTC
                ).isoformat()
    except Exception as exc:  # noqa: BLE001 - a broken import is a finding
        notes.append(f"isalsr.core build info unavailable: {type(exc).__name__}: {exc}")

    engine = "native" if info.get("engine") == "cpp" else "python"
    return (
        BuildProvenance(
            git_commit=commit,
            git_tag=tag or "(untagged)",
            git_dirty=dirty,
            native_build_hash=info.get("build_hash", "") or "(python engine)",
            engine=engine,
            compiler=info.get("compiler", "") or "(python engine)",
            compiler_flags=info.get("ndebug", "") or "(python engine)",
            isa_level=info.get("isa_level", "") or "(python engine)",
            avx512f=str(info.get("avx512f", "")).strip() not in ("", "0", "false", "False"),
            native_module_path=module_path or "(python engine)",
            native_module_mtime=module_mtime or "(python engine)",
        ),
        notes,
    )


def _config_digests() -> tuple[list[Any], list[str]]:
    """Digest the two Stage D configuration files.

    Returns:
        ``(digests, notes)``; a config that cannot be read is named in ``notes``
        and omitted, which makes the manifest fail validation rather than
        silently claim provenance it does not have.
    """
    from experiments.models.manifest import ConfigDigest, sha256_file

    digests: list[Any] = []
    notes: list[str] = []
    for method in ("udfs", "bingo"):
        relative = f"experiments/configs/{method}_{STAGE_D_SUITE}.yaml"
        path = _REPO_ROOT / relative
        try:
            digest = sha256_file(path)
        except OSError as exc:
            notes.append(f"{relative}: {type(exc).__name__}")
            continue
        digests.append(
            ConfigDigest(method=method, suite=STAGE_D_SUITE, path=relative, sha256=digest)
        )
    return digests, notes


def _operator_sets() -> tuple[dict[str, list[str]], list[str]]:
    """Read each method's operator set from its Stage D configuration.

    Returns:
        ``({method: operators}, notes)``.
    """
    notes: list[str] = []
    sets: dict[str, list[str]] = {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - yaml is an orchestrator dep
        return sets, [f"PyYAML unavailable: {exc}"]

    for method, key in (("bingo", "operators"), ("udfs", "operator_set")):
        path = _REPO_ROOT / "experiments" / "configs" / f"{method}_{STAGE_D_SUITE}.yaml"
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            notes.append(f"{path.name}: {type(exc).__name__}")
            continue
        # The operator list lives under the section named after the METHOD
        # (``bingo:`` / ``udfs:``), not under a generic ``model:`` block.
        section = payload.get(method, {}) if isinstance(payload, dict) else {}
        values = section.get(key) if isinstance(section, dict) else None
        if isinstance(values, list) and values:
            sets[method] = [str(v) for v in values]
        else:
            notes.append(f"{path.name}: {method}.{key} absent or empty")
    return sets, notes


def _submission_splits() -> list[Any]:
    """Derive the Stage D submission splits from the locked registry.

    ``manifest.build_submission_splits`` enumerates the campaign's
    ``method x arm x suite`` arrays and sizes them from ``SUITE_SIZES``, which
    does not describe a 12-cell pre-flight: Stage D runs three problems of the
    ``hard`` suite, not ten, and UDFS runs only one of them. The splits are
    therefore derived from :data:`STAGE_D_CELLS` so that
    ``n_tasks == n_problems x len(seeds)`` holds, which is what the validator
    checks in non-strict mode.

    Returns:
        One split per ``(method, arm)``, indexed contiguously from 1.
    """
    from experiments.models.manifest import SubmissionSplit

    grouped: dict[tuple[str, str], list[StageDCell]] = {}
    for cell in STAGE_D_CELLS:
        grouped.setdefault((cell.method, cell.arm), []).append(cell)

    splits: list[Any] = []
    for index, ((method, arm), members) in enumerate(sorted(grouped.items()), start=1):
        problems = {c.problem for c in members}
        splits.append(
            SubmissionSplit(
                index=index,
                method=method,
                arm=arm,
                suite=STAGE_D_SUITE,
                n_problems=len(problems),
                n_tasks=len(members),
            )
        )
    return splits


def build_stage_d_manifest(root: Path) -> tuple[Any, list[str]]:
    """Construct the Stage D pre-flight manifest.

    Args:
        root: Stage D results root, recorded as the manifest's campaign root.

    Returns:
        ``(CampaignManifest, notes)``. ``notes`` names every field that could
        not be resolved from the environment; those fields are left empty so
        validation reports them.
    """
    from experiments.models.manifest import (
        ALPHABET_VERSION,
        SCHEMA_VERSION,
        CampaignManifest,
        OperatorSetPolicy,
    )
    from experiments.scripts.stage_d_task_spec import STAGE_D_CONSTRAINT, STAGE_D_SEED

    build, build_notes = _build_provenance()
    configs, config_notes = _config_digests()
    operators, operator_notes = _operator_sets()

    stage_d_problems = sorted({c.problem for c in STAGE_D_CELLS})
    policy = OperatorSetPolicy(
        policy="uniform_per_method",
        statement=(
            "A4b: the primitive set is uniform per method across every problem. C1 "
            "gave Bingo different sets per tier, which confounded the operator set "
            "with the problem group."
        ),
        bingo_operators=operators.get("bingo", []),
        udfs_operators=operators.get("udfs", []),
        udfs_operator_source=(
            "UDFS searches the vendored NODE_ARITY table, not the YAML: the config "
            "records the set for provenance but does not set it."
        ),
        bingo_continuity_exclusion_problems=stage_d_problems,
        corrected_definition_exclusion_problems=list(CORRECTED_DEFINITION_PROBLEMS),
        continuity_note=CONTINUITY_SCOPE_NOTE,
    )

    manifest = CampaignManifest(
        schema_version=SCHEMA_VERSION,
        campaign="c2-stage-d-preflight",
        campaign_root=str(root.resolve()),
        created_utc=datetime.now(tz=UTC).isoformat(),
        build=build,
        configs=configs,
        operator_set_policy=policy,
        arms=sorted({c.arm for c in STAGE_D_CELLS}),
        seeds=[STAGE_D_SEED],
        alphabet_version=ALPHABET_VERSION,
        node_constraint=STAGE_D_CONSTRAINT,
        submission_splits=_submission_splits(),
        notes=MANIFEST_SCOPE_NOTE,
    )
    return manifest, [*build_notes, *config_notes, *operator_notes]


def check_d1_8(root: Path, observations: list[Observation]) -> CriterionResult:
    """D1.8 -- a C2 pre-flight manifest is written for the 12 runs and validates.

    The certifier both WRITES the manifest and validates it, with
    ``strict_campaign=False``: this is a 12-cell, 1-seed, 6-split pre-flight and
    the strict validator would reject every one of those on campaign grounds.
    """
    path = root.joinpath(*MANIFEST_RELPATH)
    detail: dict[str, Any] = {
        "manifest_path": str(path),
        "scope_note": MANIFEST_SCOPE_NOTE,
        "validator": "experiments.models.manifest.validate_manifest(strict_campaign=False)",
        "n_cells_covered": len(observations),
    }

    try:
        from experiments.models.manifest import (
            ManifestValidationError,
            load_manifest,
            save_manifest,
            validate_manifest,
        )
    except Exception as exc:  # noqa: BLE001 - a broken import is a finding
        detail["import_error"] = f"{type(exc).__name__}: {exc}"
        return CriterionResult(
            id="D1.8",
            title="Stage D pre-flight MANIFEST written and validated",
            status="FAIL",
            expected="importable manifest module",
            observed=f"{type(exc).__name__}: {exc}",
            detail=detail,
        )

    try:
        manifest, notes = build_stage_d_manifest(root)
        detail["build_notes"] = notes
        save_manifest(manifest, path)
        detail["written"] = path.exists()
    except Exception as exc:  # noqa: BLE001 - a write failure is a finding
        detail["write_error"] = f"{type(exc).__name__}: {exc}"
        return CriterionResult(
            id="D1.8",
            title="Stage D pre-flight MANIFEST written and validated",
            status="FAIL",
            expected=f"manifest written to {path}",
            observed=f"{type(exc).__name__}: {exc}",
            detail=detail,
        )

    try:
        reloaded = load_manifest(path)
        validate_manifest(reloaded, strict_campaign=False)
    except ManifestValidationError as exc:
        detail["validation_problems"] = list(exc.problems)
        return CriterionResult(
            id="D1.8",
            title="Stage D pre-flight MANIFEST written and validated",
            status="FAIL",
            expected="validate_manifest(strict_campaign=False) passes",
            observed=f"{len(exc.problems)} validation problem(s)",
            detail=detail,
        )
    except Exception as exc:  # noqa: BLE001 - a truncated manifest is a finding
        detail["reload_error"] = f"{type(exc).__name__}: {exc}"
        return CriterionResult(
            id="D1.8",
            title="Stage D pre-flight MANIFEST written and validated",
            status="FAIL",
            expected="the written manifest reloads",
            observed=f"{type(exc).__name__}: {exc}",
            detail=detail,
        )

    detail["validation_problems"] = []
    detail["submission_splits"] = [s.to_dict() for s in reloaded.submission_splits]
    detail["seeds"] = list(reloaded.seeds)
    detail["arms"] = list(reloaded.arms)
    return CriterionResult(
        id="D1.8",
        title="Stage D pre-flight MANIFEST written and validated",
        status="PASS",
        expected="written and validated (strict_campaign=False)",
        observed=f"written to {path.name}, 0 validation problems",
        detail=detail,
    )


# ====================================================================== #
# Reporting
# ====================================================================== #


def render_markdown(
    results: list[CriterionResult],
    root: Path,
    verdict: str,
    generated_at: str,
) -> str:
    """Render the Stage D markdown certification report.

    Args:
        results: Criterion verdicts, in canonical order.
        root: Stage D results root.
        verdict: ``GO`` or ``NO-GO``.
        generated_at: ISO-8601 timestamp.

    Returns:
        The complete markdown document.
    """
    n_blocking = sum(1 for r in results if r.blocking and r.status != "PASS")
    lines = [
        "# Campaign C2 pre-flight -- Stage D certification (12 full-length cells)",
        "",
        f"- **Verdict**: `{verdict}`",
        f"- **Root**: `{root}`",
        f"- **Generated**: {generated_at}",
        f"- **Blocking failures**: {n_blocking}",
        f"- **Cells**: {len(STAGE_D_CELLS)} at {STAGE_D_MAX_TIME_S} s search budget",
        "",
        "## Framing",
        "",
        f"- **Wall margin (D1.1)**: {WALL_MARGIN_NOTE}",
        f"- **Memory join (D1.2)**: {SACCT_TRAP_NOTE}",
        f"- **Engine (D1.6)**: {ENGINE_NOTE}",
        f"- **T16 and rho (D1.6)**: {T16_NOTE}",
        f"- **R2 band (D1.6)**: {R2_BAND_NOTE}",
        f"- **Overhead accounting (D1.7)**: {OVERHEAD_ACCOUNTING_NOTE}",
        f"- **Manifest scope (D1.8)**: {MANIFEST_SCOPE_NOTE}",
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
    sacct_csv: Path | None,
    c1_reference: Path | None,
    wall_s: float = DEFAULT_WALL_S,
    max_time: float = float(STAGE_D_MAX_TIME_S),
) -> list[CriterionResult]:
    """Run every Stage D criterion against a results root.

    Args:
        root: Stage D output root.
        sacct_csv: Optional ``JobID,MaxRSS[,Elapsed]`` export.
        c1_reference: Optional campaign-C1 analysis directory for D1.6.
        wall_s: SLURM wall limit in seconds.
        max_time: Per-run search budget in seconds, recorded for context.

    Returns:
        One :class:`CriterionResult` per criterion, D1.1 through D1.8.
    """
    observations = collect_observations(root)
    sacct_report = join_sacct(observations, sacct_csv)
    reference = load_c1_reference(c1_reference)
    log.info(
        "Stage D: %d/%d registry cells found on disk; search budget %.0f s",
        sum(1 for o in observations if o.cell is not None),
        len(observations),
        max_time,
    )
    return [
        check_d1_1(observations, wall_s),
        check_d1_2(observations, sacct_report),
        check_d1_3(root, observations),
        check_d1_4(observations),
        check_d1_5(observations),
        check_d1_6(observations, reference),
        check_d1_7(observations),
        check_d1_8(root, observations),
    ]


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="stage_d_certify",
        description="Machine-check the Campaign-C2 Stage D full-length pre-flight.",
    )
    parser.add_argument("--root", type=Path, required=True, help="Stage D output root.")
    parser.add_argument("--out-json", type=Path, required=True, help="Evidence JSON path.")
    parser.add_argument("--out-md", type=Path, required=True, help="Markdown report path.")
    parser.add_argument(
        "--sacct-csv",
        type=Path,
        default=None,
        help="CSV of JobID,MaxRSS[,Elapsed] rows produced by aggregate_worker.sh.",
    )
    parser.add_argument(
        "--c1-reference",
        type=Path,
        default=DEFAULT_C1_REFERENCE,
        help="Campaign C1 analysis directory for the D1.6 comparison.",
    )
    parser.add_argument(
        "--wall-s",
        type=float,
        default=DEFAULT_WALL_S,
        help="SLURM wall limit in seconds (default 57600 = 16 h).",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=float(STAGE_D_MAX_TIME_S),
        help="Per-run search budget in seconds (default 43200 = 12 h).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        help="Logging level for diagnostics on stderr.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Stage D certifier and return the process exit code.

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

    root = Path(args.root)
    generated_at = datetime.now(tz=UTC).isoformat()

    if not root.is_dir():
        results = [
            CriterionResult(
                id="D0",
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
            sacct_csv=args.sacct_csv,
            c1_reference=args.c1_reference,
            wall_s=float(args.wall_s),
            max_time=float(args.max_time),
        )

    n_blocking_failures = sum(1 for r in results if r.blocking and r.status != "PASS")
    verdict = "GO" if n_blocking_failures == 0 else "NO-GO"

    payload: dict[str, Any] = {r.id: r.to_dict() for r in results}
    payload["verdict"] = verdict
    payload["generated_at"] = generated_at
    payload["root"] = str(root)
    payload["n_blocking_failures"] = n_blocking_failures
    payload["n_cells"] = len(STAGE_D_CELLS)
    payload["wall_s"] = float(args.wall_s)
    payload["max_time_s"] = float(args.max_time)

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
