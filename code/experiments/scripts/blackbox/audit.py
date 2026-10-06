"""Completeness and provenance audit of the black-box corpus (analysis step 1).

The audit runs before any statistic is computed and decides which datasets
enter each host's analysis. PLAN D7 fixes the rule in advance: a dataset enters
a host's analysis only if all three arms completed all ten seeds there, with a
finite test R^2 and test NRMSE. Nothing else may exclude a dataset.

Four independent records are reconciled against the declared grid of
2 hosts x 20 datasets x 3 arms x 10 seeds:

* the run logs themselves (present, parseable, finite test metrics);
* each cell's ``status.json`` (terminal status and exit code);
* the campaign's run ledger ``status_ledger.csv`` (one row per executed cell);
* the SLURM task logs (the per-task environment gate and the cell tally of the
  main campaign's worker).

The provenance half reuses the main campaign's own integrity scan
(:func:`experiments.models.analyzer.completeness.scan_root`), which refuses a
root that pools more than one commit, build or configuration.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from experiments.models.analyzer.completeness import scan_root
from experiments.scripts.blackbox.config import BENCHMARK, DATASET_DIR, DATASETS, SEEDS
from experiments.scripts.review_campaign.config import ARMS, METHODS

#: Hardware/provenance fields reported verbatim from every run log.
PROVENANCE_FIELDS: tuple[str, ...] = (
    "git_hash",
    "git_dirty",
    "build_hash",
    "engine",
    "compiler",
    "isa_level",
    "python_version",
    "cpu_model",
)


class BlackboxAuditError(RuntimeError):
    """Raised when the corpus cannot be reconciled with the declared grid."""


@dataclass
class CellIssue:
    """One cell that fails a completeness criterion, with the reason."""

    method: str
    dataset: str
    arm: str
    seed: int
    reason: str


@dataclass
class AuditReport:
    """Outcome of the completeness and provenance audit."""

    n_expected: int = 0
    n_run_logs: int = 0
    n_completed: int = 0
    issues: list[CellIssue] = field(default_factory=list)
    included: dict[str, list[str]] = field(default_factory=dict)
    excluded: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    ledger: dict[str, Any] = field(default_factory=dict)
    slurm: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    e6_e7: dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """True when every declared cell completed with finite test metrics."""
        return not self.issues and self.n_completed == self.n_expected

    def to_dict(self) -> dict[str, Any]:
        """Serialise for ``audit.json``."""
        payload = asdict(self)
        payload["complete"] = self.complete
        return payload


def expected_cells() -> list[tuple[str, str, str, int]]:
    """Return the declared grid as ``(method, dataset, arm, seed)`` tuples."""
    return [
        (method, dataset, arm, seed)
        for method in METHODS
        for dataset in DATASETS
        for arm in ARMS
        for seed in SEEDS
    ]


def cell_dir(corpus: Path, method: str, dataset: str, arm: str, seed: int) -> Path:
    """Directory of one cell in the corpus layout."""
    return corpus / method / BENCHMARK / DATASET_DIR[dataset] / arm / f"seed_{seed:02d}"


def _cell_reason(directory: Path) -> str | None:
    """Return why a cell fails the completeness criteria, or None if it passes."""
    log_path = directory / "run_log.json"
    status_path = directory / "status.json"
    if not log_path.is_file():
        return "run_log.json missing"
    if not status_path.is_file():
        return "status.json missing"
    try:
        reg = json.loads(log_path.read_text(encoding="utf-8"))["results"]["regression"]
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        return f"unreadable: {type(exc).__name__}"
    if status.get("terminal_status") != "completed":
        return f"terminal_status={status.get('terminal_status')!r}"
    if int(status.get("exit_code", -1)) != 0:
        return f"exit_code={status.get('exit_code')!r}"
    for metric in ("r2_test", "nrmse_test"):
        value = reg.get(metric)
        if value is None or not math.isfinite(float(value)):
            return f"{metric} not finite ({value!r})"
    return None


def _apply_fallback(report: AuditReport) -> None:
    """PLAN D7: a dataset enters a host's analysis only if all 30 cells passed."""
    failing: dict[tuple[str, str], list[str]] = {}
    for issue in report.issues:
        failing.setdefault((issue.method, issue.dataset), []).append(
            f"{issue.arm}/seed_{issue.seed:02d}: {issue.reason}"
        )
    for method in METHODS:
        report.included[method] = [d for d in DATASETS if (method, d) not in failing]
        report.excluded[method] = {
            d: reasons for (m, d), reasons in sorted(failing.items()) if m == method
        }


def audit_ledger(corpus: Path) -> dict[str, Any]:
    """Reconcile ``status_ledger.csv`` with the declared grid.

    Args:
        corpus: Corpus root.

    Returns:
        Row count, terminal-status and exit-code tallies, and the cells present
        in one record and not the other.
    """
    path = corpus / "status_ledger.csv"
    if not path.is_file():
        return {"present": False}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_slug = {slug: name for name, slug in DATASET_DIR.items()}
    seen: set[tuple[str, str, str, int]] = set()
    for row in rows:
        dataset = by_slug.get(row["problem"].lower(), row["problem"])
        seen.add((row["method"], dataset, row["arm"], int(row["seed"])))
    declared = set(expected_cells())
    return {
        "present": True,
        "n_rows": len(rows),
        "terminal_status": dict(Counter(r["terminal_status"] for r in rows)),
        "exit_code": dict(Counter(r["exit_code"] for r in rows)),
        "n_nan_metrics_nonzero": sum(1 for r in rows if r.get("n_nan_metrics") not in {"", "0"}),
        "git_commit": sorted({r["git_commit"] for r in rows}),
        "node_cpu_model": sorted({r["node_cpu_model"] for r in rows}),
        "missing_from_ledger": sorted(map(list, declared - seen)),
        "not_declared": sorted(map(list, seen - declared)),
    }


_GATE = re.compile(r"^BBX gate: (\w+)", re.MULTILINE)
_TALLY = re.compile(r"ok=(\d+) fail=(\d+) defer=(\d+)")
_HASH = re.compile(r"build_hash[=': ]+([0-9a-f]{16})")


def audit_slurm_logs(corpus: Path) -> dict[str, Any]:
    """Tally the per-task gate verdicts and cell outcomes in the SLURM logs.

    The sweep array (``bbxw_*``) re-visits cells that a bundled task deferred;
    a cell already completed is reported as ``ok`` again there, so its tallies
    are kept apart rather than added to the campaign arrays.

    Args:
        corpus: Corpus root holding ``_slurm_logs/``.

    Returns:
        Task counts per array, gate verdicts, cell tallies and build hashes.
    """
    log_dir = corpus / "_slurm_logs"
    if not log_dir.is_dir():
        return {"present": False}
    arrays: Counter[str] = Counter()
    gates: Counter[str] = Counter()
    hashes: Counter[str] = Counter()
    tally = {"campaign": [0, 0, 0], "sweep": [0, 0, 0]}
    for path in sorted(log_dir.glob("*.out")):
        name = re.sub(r"_\d+_\d+$", "", path.stem)
        arrays[name] += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        verdicts = _GATE.findall(text)
        gates["PASS" if verdicts == ["PASS"] else repr(verdicts)] += 1
        hashes.update(_HASH.findall(text))
        bucket = "sweep" if name.startswith("bbxw_") else "campaign"
        for match in _TALLY.findall(text):
            for i, value in enumerate(match):
                tally[bucket][i] += int(value)
    return {
        "present": True,
        "n_task_logs": sum(arrays.values()),
        "tasks_per_array": dict(sorted(arrays.items())),
        "gate": dict(gates),
        "build_hashes": dict(hashes),
        "cells_ok_fail_defer": {
            k: dict(zip(("ok", "fail", "defer"), v, strict=True)) for k, v in tally.items()
        },
    }


def collect_provenance(corpus: Path) -> dict[str, Any]:
    """Distinct values of every provenance field, per host, from the run logs."""
    out: dict[str, Any] = {}
    for method in METHODS:
        values: dict[str, Counter[str]] = {f: Counter() for f in PROVENANCE_FIELDS}
        configs: Counter[str] = Counter()
        for _, dataset, arm, seed in (c for c in expected_cells() if c[0] == method):
            path = cell_dir(corpus, method, dataset, arm, seed) / "run_log.json"
            if not path.is_file():
                continue
            meta = json.loads(path.read_text(encoding="utf-8"))["metadata"]
            hardware = meta.get("hardware") or {}
            for name in PROVENANCE_FIELDS:
                values[name][str(hardware.get(name))] += 1
            configs[str(meta.get("config_sha256"))] += 1
        out[method] = {name: dict(counter) for name, counter in values.items()}
        out[method]["config_sha256"] = dict(configs)
    return out


def run_audit(corpus: Path) -> AuditReport:
    """Audit the corpus and apply the pre-declared inclusion rule.

    Args:
        corpus: Corpus root (read-only).

    Returns:
        The audit report, with the per-host inclusion lists filled in.

    Raises:
        BlackboxAuditError: If the corpus directory does not exist.
    """
    if not (corpus / METHODS[0] / BENCHMARK).is_dir():
        raise BlackboxAuditError(f"no {METHODS[0]}/{BENCHMARK} under {corpus}")
    report = AuditReport()
    for method, dataset, arm, seed in expected_cells():
        report.n_expected += 1
        directory = cell_dir(corpus, method, dataset, arm, seed)
        if (directory / "run_log.json").is_file():
            report.n_run_logs += 1
        reason = _cell_reason(directory)
        if reason is None:
            report.n_completed += 1
        else:
            report.issues.append(CellIssue(method, dataset, arm, seed, reason))
    _apply_fallback(report)
    report.ledger = audit_ledger(corpus)
    report.slurm = audit_slurm_logs(corpus)
    report.provenance = collect_provenance(corpus)
    completeness, provenance = scan_root(corpus, METHODS, [BENCHMARK], ARMS)
    report.e6_e7 = {"completeness": completeness.to_dict(), "provenance": provenance.to_dict()}
    return report
