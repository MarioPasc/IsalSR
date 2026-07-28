"""Aggregate per-task outputs of the T15 three-arm normalization probe.

Each SLURM array task writes ``<results-dir>/<method>/<problem>_seed<N>/runs.json``.
This walks that tree, pools the per-run counters, and reproduces the same summary
the local single-process runs print.

Usage:
    python -m experiments.scripts.aggregate_norm_arms --results-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.scripts.measure_const_normalization_arms import format_table, summarise


def collect(results_dir: Path) -> list[dict[str, Any]]:
    """Load every runs.json under *results_dir*, flattening the per-file lists."""
    records: list[dict[str, Any]] = []
    for path in sorted(results_dir.rglob("runs.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"[warn] unreadable, skipping: {path}")
            continue
        records.extend(payload if isinstance(payload, list) else [payload])
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", default=None, help="Defaults to <results-dir>/aggregate")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    records = collect(results_dir)
    if not records:
        raise SystemExit(f"no runs.json found under {results_dir}")

    scored = [r for r in records if "arms" in r]
    errored = [r for r in records if "error" in r]
    summary = summarise(records)

    out_dir = Path(args.out) if args.out else results_dir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    table = format_table(summary)
    (out_dir / "summary.txt").write_text(table + "\n")

    print(table)
    print(f"\nruns scored: {len(scored)}   errored: {len(errored)}")
    for r in errored[:10]:
        print(f"  {r.get('method')}/{r.get('problem')}/seed{r.get('seed')}: {r['error']}")
    print(f"\nwrote {out_dir}/summary.json")

    # Guard the same way the per-task probe does: an empty or partially errored
    # campaign pools into "0 failures", which reads exactly like the clean result
    # and would be reported as one. Refuse to exit 0 on either.
    n_calls = summary["global"]["submitted"]["n_calls"]
    if n_calls == 0:
        raise SystemExit(
            f"FATAL: no DAG reached the canonicaliser across {len(records)} run(s). "
            "This is not a zero failure rate — it is no data."
        )
    if errored:
        raise SystemExit(
            f"FATAL: {len(errored)}/{len(records)} run(s) errored; the pooled counters "
            "cover only the runs that survived and must not be reported as a rate."
        )


if __name__ == "__main__":
    main()
