"""I/O utilities for the experimental framework.

Read/write unified format files: run_log.json, trajectory.csv,
aggregate.csv, paired_stats.json.

Reference: docs/design/experimental_design/isalsr_experimental_design.md, Section C.6.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from experiments.models.schemas import (
    AGGREGATE_COLUMNS,
    TRAJECTORY_COLUMNS,
    AggregateRow,
    PairedStats,
    RunLog,
    TrajectoryRow,
)

log = logging.getLogger(__name__)


# ======================================================================
# RunLog (JSON)
# ======================================================================


def save_run_log(run_log: RunLog, path: Path) -> None:
    """Write a RunLog to run_log.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    run_log.save_json(path)
    log.debug("Saved run_log to %s", path)


def load_run_log(path: Path) -> RunLog:
    """Load a RunLog from run_log.json."""
    return RunLog.load_json(path)


def load_all_run_logs(directory: Path) -> list[RunLog]:
    """Load all run_log.json files from seed subdirectories.

    Skips corrupt or empty files with a warning (common on shared
    HPC filesystems where write-flush races can produce 0-byte files).
    """
    logs = []
    for seed_dir_path in sorted(directory.iterdir()):
        run_log_path = seed_dir_path / "run_log.json"
        if run_log_path.exists():
            try:
                logs.append(RunLog.load_json(run_log_path))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                log.warning(
                    "Skipping corrupt run_log: %s (%s: %s)",
                    run_log_path,
                    type(exc).__name__,
                    exc,
                )
    return logs


# ======================================================================
# Trajectory (CSV)
# ======================================================================


def save_trajectory(rows: list[TrajectoryRow], path: Path) -> None:
    """Write trajectory rows to trajectory.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAJECTORY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())
    log.debug("Saved trajectory (%d rows) to %s", len(rows), path)


def load_trajectory(path: Path) -> list[TrajectoryRow]:
    """Load trajectory rows from trajectory.csv."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                TrajectoryRow(
                    timestamp_s=float(r["timestamp_s"]),
                    iteration=int(r["iteration"]),
                    best_r2=float(r["best_r2"]),
                    best_nrmse=float(r["best_nrmse"]),
                    n_dags_explored=int(r["n_dags_explored"]),
                    n_unique_canonical=int(r["n_unique_canonical"]),
                    current_expr=r["current_expr"],
                    current_complexity=int(r["current_complexity"]),
                    cache_hit_rate_cumulative=float(r["cache_hit_rate_cumulative"]),
                )
            )
    return rows


# ======================================================================
# Aggregate (CSV)
# ======================================================================


def save_aggregate(rows: list[AggregateRow], path: Path) -> None:
    """Write aggregate rows to aggregate.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGGREGATE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())
    log.debug("Saved aggregate (%d rows) to %s", len(rows), path)


# ======================================================================
# Paired stats (JSON)
# ======================================================================


def save_paired_stats(stats: PairedStats, path: Path) -> None:
    """Write paired statistics to paired_stats.json."""
    stats.save_json(path)
    log.debug("Saved paired_stats to %s", path)


def load_paired_stats(path: Path) -> PairedStats:
    """Load paired statistics from paired_stats.json."""
    return PairedStats.load_json(path)


# ======================================================================
# Metadata (JSON)
# ======================================================================


def save_metadata(metadata: dict[str, Any], path: Path) -> None:
    """Write experiment metadata to metadata.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.debug("Saved metadata to %s", path)


# ======================================================================
# Directory structure (Section C.6)
# ======================================================================


def ensure_output_structure(
    base_dir: Path,
    method: str,
    benchmark: str,
    problem: str,
) -> dict[str, Path]:
    """Create the output folder tree per experimental design Section C.6.

    Returns a dict mapping logical names to directory paths.
    """
    problem_slug = problem.lower().replace("-", "_")
    problem_dir = base_dir / method / benchmark / problem_slug

    paths = {
        "problem": problem_dir,
        "baseline": problem_dir / "baseline",
        "isalsr": problem_dir / "isalsr",
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    return paths


def seed_dir(parent: Path, seed: int) -> Path:
    """Return the seed subdirectory path."""
    d = parent / f"seed_{seed:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d
