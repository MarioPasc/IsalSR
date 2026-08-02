"""Reduce a T05 probe run to the two tables the ticket has to report.

`T05` AC-4b requires SP-1…SP-6 as a **six-row table for every Picasso probe this
ticket ran**, and SP-7's five statements established for every D2 problem on both
hosts. This turns the probe's scattered JSON into exactly those two tables, so the
work log quotes a generated artefact rather than a hand-copied one.

It also emits the per-problem run summary — ρ, unique canonical DAGs, wall clock,
engine — because the ticket's reviewers will ask what the probe actually observed,
and because `ρ = 1.0` everywhere would mean the dedup hook is dead rather than that
the rates are zero.

Usage
-----
    python slurm/t05_probe/summarise.py --root ~/execs/isalsr/t05_probe
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SP_KEYS: tuple[tuple[str, str], ...] = (
    ("SP-1_provenance", "Provenance — running the commit we think we are"),
    ("SP-2_install_freshness", "Installation freshness — the .so is the code we edited"),
    ("SP-3_engine", "Engine native, with the forced-Python negative control"),
    ("SP-4_alphabet", "Alphabet — no Sub/Div, no '-'/'/' in any canonical string"),
    ("SP-5_host", "Both hosts — UDFS and Bingo"),
    ("SP-6_fallback_counters", "T06 fallback counters live and finite"),
)

METRIC_FIELDS: tuple[str, ...] = ("r2_train", "r2_test", "nrmse_train", "nrmse_test", "mse_test")


def _load(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning ``None`` if it is missing or unparsable."""
    try:
        return json.loads(path.read_text())  # type: ignore[no-any-return]
    except (OSError, json.JSONDecodeError):
        return None


def sp_table(root: Path) -> tuple[str, bool]:
    """Build the SP-1…SP-6 six-row table across every probe cell.

    Parameters
    ----------
    root
        Probe output root.

    Returns
    -------
    tuple
        The rendered table and whether every property held on every cell.
    """
    cells = sorted(root.glob("*/sp_evidence.json"))
    neg = sorted(root.glob("*/sp_evidence_forced_python.json"))
    rows = ["| # | Property | Cells passing | Verdict |", "|---|---|---|---|"]
    all_ok = bool(cells)

    for key, label in SP_KEYS:
        present = [d for p in cells if (d := _load(p)) is not None and key in d]
        n_pass = sum(1 for d in present if d[key].get("pass"))
        if not present:
            rows.append(f"| {key.split('_')[0]} | {label} | — | **not reported** |")
            all_ok = False
            continue
        ok = n_pass == len(present)
        all_ok &= ok
        rows.append(
            f"| {key.split('_')[0]} | {label} | {n_pass}/{len(present)} | "
            f"{'**PASS**' if ok else '**FAIL**'} |"
        )

    # SP-3's negative control is a separate artefact: a probe that reports
    # `native` in both directions proves nothing and is itself a defect.
    n_neg_ok = sum(
        1
        for p in neg
        if (d := _load(p)) is not None
        and d.get("SP-3_engine", {}).get("reported_engine") == "python"
    )
    rows.append(
        f"| SP-3′ | *negative control* — forced Python actually reports `python` | "
        f"{n_neg_ok}/{len(neg)} | {'**PASS**' if neg and n_neg_ok == len(neg) else '**FAIL**'} |"
    )
    all_ok &= bool(neg) and n_neg_ok == len(neg)
    return "\n".join(rows), all_ok


def run_table(root: Path) -> tuple[str, dict[str, Any]]:
    """Summarise every `run_log.json` under the probe root.

    Parameters
    ----------
    root
        Probe output root.

    Returns
    -------
    tuple
        The rendered per-run table and a dict of aggregate statistics.
    """
    rows = [
        "| Host | Arm | Problem | R² test | ρ | unique | wall (s) | engine | NaN |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    stats: dict[str, Any] = {
        "n": 0,
        "nan_cells": 0,
        "rho_le_1": [],
        "baseline_rho_not_1": [],
        "by_host": Counter(),
    }
    rho: dict[str, list[float]] = {"bingo": [], "udfs": []}

    for path in sorted(root.rglob("run_log.json")):
        doc = _load(path)
        if doc is None:
            rows.append(f"| — | {path.parent.name} | **UNPARSABLE** | | | | | |")
            stats["nan_cells"] += 1
            continue
        meta, res = doc["metadata"], doc["results"]
        reg, ss, tm = res["regression"], res["search_space"], res["time"]
        bad = [
            f
            for f in METRIC_FIELDS
            if not isinstance(reg.get(f), (int, float))
            or math.isnan(float(reg[f]))
            or math.isinf(float(reg[f]))
        ]
        r = float(ss.get("empirical_reduction_factor") or 0.0)
        host = meta["method"]
        arm = meta["representation"]
        stats["n"] += 1
        stats["by_host"][f"{host}/{arm}"] += 1
        if bad:
            stats["nan_cells"] += 1

        # ρ is only meaningful on a deduplicating arm. The `baseline` arm is
        # un-instrumented, so ρ = 1 exactly is what it MUST report (Stage C
        # C1.8) -- counting that as a violation would bury a real one. On
        # `isalsr`, ρ < 1 is arithmetically impossible and ρ == 1 everywhere
        # means the dedup hook is dead rather than that the rate is zero (C1.6).
        if arm == "baseline":
            if r != 1.0:
                stats["baseline_rho_not_1"].append(f"{host}/{meta['problem']}={r:.4f}")
        else:
            rho.setdefault(host, []).append(r)
            if r <= 1.0:
                stats["rho_le_1"].append(f"{host}/{arm}/{meta['problem']}={r:.4f}")

        rows.append(
            f"| {host} | {arm} | {meta['problem']} | {reg['r2_test']:.4f} | {r:.4f} | "
            f"{ss.get('unique_canonical_dags')} | {tm.get('wall_clock_total_s', 0):.0f} | "
            f"{meta.get('hardware', {}).get('engine', '—')} | {','.join(bad) or '—'} |"
        )

    for host, vals in rho.items():
        if vals:
            stats[f"rho_{host}_mean"] = sum(vals) / len(vals)
            stats[f"rho_{host}_min"] = min(vals)
            stats[f"rho_{host}_max"] = max(vals)
    return "\n".join(rows), stats


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns
    -------
    int
        0 iff every SP property held and no cell carried a NaN.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser()
    sp, sp_ok = sp_table(root)
    runs, stats = run_table(root)

    parts = [
        "## SP-1…SP-6 (AC-4b)\n",
        sp,
        "\n\n## Per-run summary\n",
        runs,
        "\n\n## Aggregate\n",
        f"- runs: **{stats['n']}**  ({dict(stats['by_host'])})",
        f"- cells with NaN/inf or unparsable: **{stats['nan_cells']}**",
        f"- dedup-arm cells with ρ ≤ 1 (C1.6): **{len(stats['rho_le_1'])}** "
        f"{stats['rho_le_1'] or ''}",
        f"- baseline cells with ρ ≠ 1 (C1.8): **{len(stats['baseline_rho_not_1'])}** "
        f"{stats['baseline_rho_not_1'] or ''}",
    ]
    for host in ("bingo", "udfs"):
        if f"rho_{host}_mean" in stats:
            parts.append(
                f"- ρ {host} (dedup arms): mean **{stats[f'rho_{host}_mean']:.4f}**, "
                f"range [{stats[f'rho_{host}_min']:.4f}, {stats[f'rho_{host}_max']:.4f}]"
            )
    text = "\n".join(parts) + "\n"

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        log.info("wrote %s", args.out)
    else:
        log.info("%s", text)

    ok = (
        sp_ok
        and stats["nan_cells"] == 0
        and stats["n"] > 0
        and not stats["rho_le_1"]
        and not stats["baseline_rho_not_1"]
    )
    log.info("PROBE VERDICT: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
