"""Flatten every run log of the three-arm campaign into one tidy table.

One row per (method, arm, problem, seed) cell, 12,600 rows. Everything the
manuscript reports is derived from this file downstream, so the definitions
live here and nowhere else.

Definitions
-----------
n_cand
    ``total_dags_explored`` -- candidates the arm submitted to the
    deduplication layer.
n_unique
    ``unique_canonical_dags`` -- distinct keys among them. On the native arm no
    key is computed and the two counts coincide by construction.
n_eval
    Candidates that reached the fitness evaluator: ``n_unique`` on a
    deduplicating arm, ``n_cand`` on the native arm.
rho, r
    ``n_cand / n_unique`` and ``1 - 1/rho``.
key_ms
    ``canonicalization_runtime_s / n_cand``. The naive hash arm writes its
    serialise-and-hash cost into the same field, so the column is the
    per-candidate cost of whichever key that arm uses.
eval_ms
    ``wall_clock_search_only_s / n_eval``. Search time excludes the key and the
    conversion, so this is the per-evaluation cost of everything the host does
    with a candidate it did not skip.
overhead_pct
    ``overhead_time_s / wall_clock_total_s``, where the numerator is key plus
    conversion. Zero on the native arm by construction.
r2_test, r2_train
    Clipped to [0, 1] following the SRBench convention the manuscript states.
    The raw values are kept alongside, because single cells reach large
    negative R^2 and any statistic taken on the raw column is dominated by
    them.

Usage
-----
    python -m experiments.scripts.review_campaign.extract_cells \\
        [--corpus DIR] [--analyses DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.scripts.review_campaign.config import (  # noqa: E402
    ARMS,
    METHODS,
    SUITES,
    SUITES_CORE,
    add_common_args,
)

COMPLEXITY_FIELDS = (
    "complexity_mean_k",
    "complexity_median_k",
    "complexity_p90_k",
    "complexity_max_k",
    "complexity_mean_depth",
    "complexity_mean_edges",
    "complexity_mean_shared",
    "complexity_mean_nonlinear",
    "complexity_mean_op_entropy",
    "complexity_n_sampled",
    "complexity_time_s",
)

COLUMNS = (
    "method",
    "suite",
    "in_core_50",
    "problem",
    "arm",
    "seed",
    "r2_test",
    "r2_train",
    "r2_test_raw",
    "r2_train_raw",
    "nrmse_test",
    "nrmse_train",
    "solution_recovered",
    "jaccard_index",
    "model_complexity",
    "n_nonfinite_test_predictions",
    "n_cand",
    "n_unique",
    "n_eval",
    "rho",
    "r",
    "max_k",
    "wall_s",
    "search_s",
    "canon_s",
    "conv_s",
    "overhead_s",
    "overhead_pct",
    "key_ms",
    "eval_ms",
    "n_canon_timeouts",
    "n_canon_raised",
    "n_conversion_failures",
    "n_violations_pre",
    "n_violations_post",
    "git_hash",
    "engine",
    "config_sha256",
    *COMPLEXITY_FIELDS,
)


def clip01(value: float | None) -> float | None:
    """Clip a coefficient of determination to [0, 1], as SRBench reports it."""
    if value is None:
        return None
    return min(1.0, max(0.0, value))


def row_from_log(path: Path, suite: str) -> dict[str, Any]:
    """Read one run log into a flat record.

    Args:
        path: Path to a ``run_log.json``.
        suite: Benchmark suite the problem belongs to.

    Returns:
        One record keyed by :data:`COLUMNS`.
    """
    with path.open(encoding="utf-8") as handle:
        log = json.load(handle)

    meta = log["metadata"]
    reg = log["results"]["regression"]
    tim = log["results"]["time"]
    space = log["results"]["search_space"]

    n_cand = space["total_dags_explored"]
    n_unique = space["unique_canonical_dags"]
    arm = meta["representation"]
    n_eval = n_cand if arm == "baseline" else n_unique

    wall = tim["wall_clock_total_s"]
    search = tim["wall_clock_search_only_s"]
    canon = tim["canonicalization_runtime_s"]
    overhead = tim.get("overhead_time_s") or 0.0

    record: dict[str, Any] = {
        "method": meta["method"],
        "suite": suite,
        "in_core_50": int(suite in SUITES_CORE),
        "problem": meta["problem"],
        "arm": arm,
        "seed": meta["seed"],
        "r2_test": clip01(reg["r2_test"]),
        "r2_train": clip01(reg["r2_train"]),
        "r2_test_raw": reg["r2_test"],
        "r2_train_raw": reg["r2_train"],
        "nrmse_test": reg["nrmse_test"],
        "nrmse_train": reg["nrmse_train"],
        "solution_recovered": int(bool(reg["solution_recovered"])),
        "jaccard_index": reg.get("jaccard_index"),
        "model_complexity": reg.get("model_complexity"),
        "n_nonfinite_test_predictions": reg.get("n_nonfinite_test_predictions"),
        "n_cand": n_cand,
        "n_unique": n_unique,
        "n_eval": n_eval,
        "rho": space["empirical_reduction_factor"],
        "r": space["redundancy_rate"],
        "max_k": space["max_internal_nodes_seen"],
        "wall_s": wall,
        "search_s": search,
        "canon_s": canon,
        "conv_s": tim.get("conversion_time_s") or 0.0,
        "overhead_s": overhead,
        "overhead_pct": 100.0 * overhead / wall if wall > 0 else None,
        "key_ms": 1000.0 * canon / n_cand if n_cand > 0 else None,
        "eval_ms": 1000.0 * search / n_eval if n_eval > 0 else None,
        "n_canon_timeouts": space.get("n_canon_timeouts"),
        "n_canon_raised": space.get("n_canon_raised"),
        "n_conversion_failures": space.get("n_conversion_failures"),
        "n_violations_pre": space.get("n_violations_pre"),
        "n_violations_post": space.get("n_violations_post"),
        "git_hash": meta["hardware"].get("git_describe") or meta["hardware"].get("git_hash"),
        "engine": meta["hardware"].get("engine"),
        "config_sha256": meta.get("config_sha256"),
    }
    for field in COMPLEXITY_FIELDS:
        record[field] = space.get(field)
    return record


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for suite in SUITES:
            suite_dir = args.corpus / method / suite
            if not suite_dir.is_dir():
                raise SystemExit(f"missing suite directory: {suite_dir}")
            for problem_dir in sorted(suite_dir.iterdir()):
                if not problem_dir.is_dir():
                    continue
                for arm in ARMS:
                    for seed_dir in sorted((problem_dir / arm).iterdir()):
                        log_path = seed_dir / "run_log.json"
                        if log_path.is_file():
                            rows.append(row_from_log(log_path, suite))

    out = args.analyses / "data" / "cells.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    problems = {row["problem"] for row in rows}
    print(f"{len(rows)} cells, {len(problems)} problems -> {out}")
    for method in METHODS:
        for arm in ARMS:
            n = sum(1 for r in rows if r["method"] == method and r["arm"] == arm)
            print(f"  {method:6s} {arm:9s} {n}")
    print(f"  engines: {sorted({r['engine'] for r in rows})}")
    print(f"  commits: {sorted({r['git_hash'] for r in rows})}")


if __name__ == "__main__":
    main()
