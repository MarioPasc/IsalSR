#!/usr/bin/env python
"""Local six-cell pre-flight for the T19 complexity telemetry.

Runs every ``(method, arm)`` combination once, on one easy problem with a tiny
time budget, and prints the telemetry block beside the pre-T19 fields. It exists
to catch the failure that a Picasso probe would otherwise catch forty minutes
and 24 allocations later: a runner that was never wired, an import that the
formatter stripped, or a schema field that does not reach the run log.

It is a **pre-flight, not evidence**. The budget is seconds, one seed, one
problem; no number it prints means anything scientifically. Run it before every
submission of ``slurm/t19_probe/launcher.sh``.

Usage
-----
    PYTHONPATH=src:. python slurm/t19_probe/local_smoke.py <output_dir> [max_time_s]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CONFIGS = {
    "udfs": "experiments/configs/udfs_nguyen.yaml",
    "bingo": "experiments/configs/bingo_nguyen.yaml",
}
ARMS = ("baseline", "hash", "isalsr")
PROBLEM = "Nguyen-1"
#: Seed 0 only. Never 1..30 -- SP-0 forbids a probe output that could be
#: mistaken for a campaign cell, and that discipline is worth keeping locally
#: too, since local trees get rsynced.
SEED = "0"


def run_cell(method: str, config: str, arm: str, out: Path, max_time: str) -> bool:
    """Execute one ``(method, arm)`` cell.

    Parameters
    ----------
    method
        ``"udfs"`` or ``"bingo"``.
    config
        Repo-relative path to the method's YAML config.
    arm
        ``"baseline"``, ``"hash"`` or ``"isalsr"``.
    out
        Results root.
    max_time
        Per-run payload budget, in seconds.

    Returns
    -------
    bool
        Whether the cell exited zero.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.models.orchestrator",
            "--config",
            config,
            "--output-dir",
            str(out),
            "--seeds",
            SEED,
            "--problems",
            PROBLEM,
            "--variants",
            arm,
            "--max-time",
            max_time,
            "--ledger",
            "--postprocess",
            "skip",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(f"!! {method}/{arm} exited {proc.returncode}\n{proc.stderr[-2500:]}")
    return proc.returncode == 0


def main() -> int:
    """Run the six cells and report. Returns the process exit status."""
    if not 2 <= len(sys.argv) <= 3:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    max_time = sys.argv[2] if len(sys.argv) == 3 else "40"

    ok = True
    for method, config in CONFIGS.items():
        if not (REPO / config).exists():
            print(f"!! missing config: {config}")
            ok = False
            continue
        for arm in ARMS:
            ok &= run_cell(method, config, arm, out, max_time)

    cells = sorted(out.rglob("run_log.json"))
    hdr = (
        f"\n{'method':6} {'arm':9} {'mode':11} {'rate':>4} {'n':>7} {'meanK':>6} "
        f"{'depth':>6} {'uqN':>7} {'cost_s':>7} {'fail':>4} | "
        f"{'total_dags':>10} {'unique':>8} {'RF':>6} {'R2test':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for path in cells:
        d = json.loads(path.read_text())
        ss, meta = d["results"]["search_space"], d["metadata"]

        def fmt(key: str, width: int = 6, ss: dict[str, Any] = ss) -> str:
            v = ss.get(key)
            return f"{v:>{width}.2f}" if isinstance(v, (int, float)) else f"{'-':>{width}}"

        print(
            f"{meta['method']:6} {meta['representation']:9} "
            f"{str(ss.get('complexity_sampling_mode')):11} "
            f"{str(ss.get('complexity_sample_rate')):>4} "
            f"{str(ss.get('complexity_n_sampled')):>7} {fmt('complexity_mean_k')} "
            f"{fmt('complexity_mean_depth')} "
            f"{str(ss.get('complexity_unique_n_sampled')):>7} "
            f"{fmt('complexity_time_s', 7)} {str(ss.get('complexity_n_failures')):>4} | "
            f"{ss['total_dags_explored']:>10} {ss['unique_canonical_dags']:>8} "
            f"{ss['empirical_reduction_factor']:>6.3f} "
            f"{d['results']['regression']['r2_test']:>7.4f}"
        )
        if not (path.parent / "complexity.json").exists():
            print(f"   !! MISSING complexity.json for {meta['method']}/{meta['representation']}")
            ok = False

    n_side = len(list(out.rglob("complexity.json")))
    print(f"\n{len(cells)} run_log.json, {n_side} complexity.json (expect 6 and 6)")
    if len(cells) != 6 or n_side != 6:
        ok = False

    # A cell that recorded nothing is the failure this pre-flight exists for:
    # it is indistinguishable in the run log from an instrument that never fired.
    for path in cells:
        d = json.loads(path.read_text())
        if not d["results"]["search_space"].get("complexity_n_sampled"):
            m = d["metadata"]
            print(f"!! ZERO samples on {m['method']}/{m['representation']}")
            ok = False

    print("PRE-FLIGHT OK" if ok else "PRE-FLIGHT FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
