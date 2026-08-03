"""Terminal-status ledger: every run leaves a record, including the ones that die.

EXECUTION-PLAN P4 and §5.5, the "anti-1,465 rule". Campaign C1 reported 1,500
UDFS cells and 1,465 Bingo cells. The 35-cell gap took a forensic pass over
SLURM logs to explain (36 OOM kills, 9 post-search SymPy hangs) and was
discovered by us rather than by a reviewer only because we went looking. At
C2's 8,400-run scale the same silence would be unexplainable.

The rule this module enforces has no third state: a cell is either a valid
``run_log.json`` **or** a ledger row naming its cause.

**Why write-ahead rather than try/except.** The dominant C1 failure was
``OUT_OF_MEMORY``: 29 of 31 IsalSR OOMs died at ``MaxRSS ~ 127.7 GB`` against a
128 GB request. An OOM arrives as ``SIGKILL``, which no Python handler observes
-- an exception-based ledger records nothing for exactly the failure mode that
caused the shortfall. So the record is written **before** the search starts,
with ``terminal_status = "started"``, and rewritten on the way out. A row still
reading ``started`` after the array has drained *is* the report: that cell was
killed from outside the process, and the ledger names it instead of leaving a
hole for someone to notice in September.

**Why one file per run rather than one appended CSV.** 1,400 concurrent array
tasks appending to a shared file interleave partial lines. Each run writes its
own ``status.json`` in its own seed directory -- no coordination, no lock, no
lost row -- and :func:`collect_status_ledger` assembles the campaign-level
``status_ledger.csv`` afterwards, when nothing is writing.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "LEDGER_COLUMNS",
    "RunStatus",
    "collect_status_ledger",
    "load_status",
    "write_status",
]

#: Filename written inside each seed directory.
STATUS_FILENAME = "status.json"

TerminalStatus = Literal["started", "completed", "failed"]


@dataclass
class RunStatus:
    """One row of the campaign status ledger.

    Mutable by design: the same object is written at ``started`` and rewritten
    at ``completed``/``failed``, so the terminal write cannot disagree with the
    provenance captured before the search began.
    """

    # --- cell identity ------------------------------------------------- #
    method: str
    arm: str
    benchmark: str
    problem: str
    seed: int

    # --- outcome ------------------------------------------------------- #
    # "started" survives only if the process was killed from outside (OOM,
    # SLURM wall limit, node failure). That is the signal, not an omission.
    terminal_status: TerminalStatus = "started"
    exit_code: int | None = None
    wall_clock_s: float | None = None
    max_rss_gb: float | None = None

    # --- provenance ----------------------------------------------------- #
    node_cpu_model: str = ""
    hostname: str = ""
    engine: str = ""
    git_commit: str = ""
    config_sha256: str = ""
    data_fingerprint: str = ""
    slurm_job_id: str = ""
    slurm_array_task_id: str = ""

    # --- diagnostics ----------------------------------------------------- #
    n_nan_metrics: int | None = None
    nan_fields: str = ""
    exception_class: str = ""
    exception_message: str = ""

    started_at: str = ""
    finished_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Return a flat CSV-safe row, dropping the nested ``extra`` mapping."""
        row = asdict(self)
        row.pop("extra", None)
        return row


#: Column order of ``status_ledger.csv``, derived from the dataclass so the two
#: cannot drift.
LEDGER_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(RunStatus) if f.name != "extra")


def write_status(status: RunStatus, seed_directory: Path) -> Path:
    """Write (or overwrite) the status record for one run.

    Written atomically via a temporary file and ``os.replace``, so a process
    killed mid-write leaves the previous complete record rather than a truncated
    one. Truncation would be the worst outcome available: it turns a readable
    ``started`` row -- which correctly reports an external kill -- into a parse
    error indistinguishable from corruption.

    Args:
        status: The record to persist.
        seed_directory: The run's own output directory.

    Returns:
        Path to the written ``status.json``.
    """
    seed_directory.mkdir(parents=True, exist_ok=True)
    target = seed_directory / STATUS_FILENAME
    tmp = seed_directory / f".{STATUS_FILENAME}.tmp"
    payload = asdict(status)
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_status(path: Path) -> RunStatus | None:
    """Load one status record, or ``None`` if it is missing or unreadable.

    Never raises. A corrupt record is itself a finding, and it must not stop the
    reconciliation that would report it.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    known = {f.name for f in fields(RunStatus)}
    try:
        return RunStatus(**{k: v for k, v in payload.items() if k in known})
    except TypeError:
        return None


def collect_status_ledger(root: Path, output_csv: Path | None = None) -> list[RunStatus]:
    """Walk a results root and assemble the campaign status ledger.

    Args:
        root: Campaign or smoke root, e.g. ``c2_smoke/``.
        output_csv: Where to write ``status_ledger.csv``. When ``None``, the
            rows are returned without writing.

    Returns:
        Every status record found, sorted by cell identity so two runs of the
        collector over the same root produce byte-identical CSVs.
    """
    rows = [
        status
        for path in sorted(root.rglob(STATUS_FILENAME))
        if (status := load_status(path)) is not None
    ]
    rows.sort(key=lambda s: (s.method, s.arm, s.benchmark, s.problem, s.seed))

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(LEDGER_COLUMNS))
            writer.writeheader()
            for status in rows:
                writer.writerow(status.to_row())
    return rows


def reconcile(
    rows: list[RunStatus],
    expected_cells: set[tuple[str, str, str, int]],
) -> dict[str, Any]:
    """Compare observed ledger rows against the cells the campaign should hold.

    This is check C1.15 and E6: a count that matches is worth nothing unless a
    mismatch **names** the offending cells. C1's shortfall was reported as a
    number, which is why explaining it needed a forensic pass.

    Args:
        rows: Status records, from :func:`collect_status_ledger`.
        expected_cells: The ``(method, arm, problem, seed)`` tuples the campaign
            is supposed to contain.

    Returns:
        Dict with ``n_expected``, ``n_observed``, ``n_completed``, and the
        explicit cell lists ``missing`` (never started), ``killed`` (started,
        no terminal record -- an OOM or wall-limit kill) and ``failed``.
    """
    observed = {(r.method, r.arm, r.problem, r.seed): r for r in rows}
    missing = sorted(expected_cells - set(observed))
    killed = sorted(k for k, r in observed.items() if r.terminal_status == "started")
    failed = sorted(k for k, r in observed.items() if r.terminal_status == "failed")
    return {
        "n_expected": len(expected_cells),
        "n_observed": len(observed),
        "n_completed": sum(1 for r in observed.values() if r.terminal_status == "completed"),
        "missing": missing,
        "killed": killed,
        "failed": failed,
        "reconciled": not missing and not killed and not failed,
    }
