"""Derive every quantity the manuscript reports from the tidy cell table.

Reads ``data/cells.csv`` written by ``extract_cells`` and the paired-test
artefacts written by ``experiments.models.analyze``, and writes the per-problem
table, the per-method summary, the share phi, the key-cost comparison and the
k-stratified overhead. Nothing downstream reads a run log again.

Two conventions are fixed here because the manuscript depends on them.

Per-evaluation cost.
    ``eval_ms`` is read from the two deduplicating arms and never from the
    native one. On Bingo the native arm counts candidates through a different
    mechanism and reports roughly nine times as many of them, so a
    per-candidate cost taken there is not the cost of an evaluation. The two
    deduplicating arms agree closely on both hosts, which is what makes the
    quantity trustworthy.

Search-only speedup.
    ``S = T_search(native) / T_search(arm)`` is seed-matched and uses wall
    clock only, so it is free of the counter mismatch above. It is degenerate
    wherever both arms exhaust the time budget: UDFS saturates on most cells
    and S is then 1 by construction, so the summary reports the saturated and
    unsaturated cells apart.

The share phi.
    Equation (phi_rho) of the manuscript writes phi in terms of the two
    reduction factors alone,

        phi = (1/rho_ser - 1/rho) / (1 - 1/rho) = 1 - r_ser / r,

    which is scale-free and therefore does not require the two counts to come
    from streams of the same length. ``rho`` comes from the IsalSR arm and
    ``rho_ser`` from the naive hash arm of the same paired cell. The two arms
    search different trajectories; the manuscript states that confound rather
    than pretending it away.

Usage
-----
    python -m experiments.scripts.review_campaign.derive [--analyses DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.scripts.review_campaign.config import (  # noqa: E402
    ARMS,
    K_RANGES,
    METHODS,
    SUITES,
    add_common_args,
)

TEXT_COLUMNS = frozenset(
    {"method", "suite", "problem", "arm", "git_hash", "engine", "config_sha256"}
)

#: Columns aggregated over the seeds of a cell.
AGG_FIELDS = (
    "r2_test",
    "r2_train",
    "nrmse_test",
    "rho",
    "r",
    "wall_s",
    "search_s",
    "overhead_pct",
    "key_ms",
    "eval_ms",
    "max_k",
    "solution_recovered",
    "n_cand",
    "n_unique",
    "complexity_mean_k",
)

CONTRASTS = ("isalsr_vs_baseline", "isalsr_vs_hash", "hash_vs_baseline")
CPDT_METRICS = (
    "r2_test",
    "r2_train",
    "nrmse_test",
    "empirical_reduction_factor",
    "redundancy_rate",
)


def read_cells(path: Path) -> list[dict[str, Any]]:
    """Load the tidy cell table, coercing the numeric columns."""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key in TEXT_COLUMNS:
                continue
            row[key] = float(value) if value not in {"", "None"} else None
    return rows


def stats(values: list[float | None]) -> dict[str, float]:
    """Mean, standard deviation, median and range over the finite entries."""
    xs = [v for v in values if v is not None and math.isfinite(v)]
    if not xs:
        return dict.fromkeys(
            ("n", "mean", "std", "median", "p05", "p95", "min", "max"), math.nan
        ) | {"n": 0}
    ordered = sorted(xs)
    return {
        "n": len(xs),
        "mean": st.fmean(xs),
        "std": st.pstdev(xs) if len(xs) > 1 else 0.0,
        "median": st.median(xs),
        "p05": ordered[max(0, int(0.05 * (len(ordered) - 1)))],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "min": ordered[0],
        "max": ordered[-1],
    }


def build_per_problem(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate the seeds of every (method, arm, problem) cell."""
    groups: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["method"], row["problem"], row["arm"], row["suite"], int(row["in_core_50"]))
        groups[key].append(row)

    out: list[dict[str, Any]] = []
    for (method, problem, arm, suite, core), cells in sorted(groups.items()):
        record: dict[str, Any] = {
            "method": method,
            "problem": problem,
            "arm": arm,
            "suite": suite,
            "in_core_50": core,
            "n_seeds": len(cells),
        }
        for field in AGG_FIELDS:
            summary = stats([c[field] for c in cells])
            record[f"{field}_mean"] = summary["mean"]
            record[f"{field}_std"] = summary["std"]
            record[f"{field}_median"] = summary["median"]
        record["max_k_seen"] = max((c["max_k"] or 0) for c in cells)
        out.append(record)
    return out


#: A run is at the cap when it spends the whole 12-hour payload budget.
BUDGET_S = 43200.0
SATURATED_S = 43000.0


def build_speedup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[float]]:
    """Seed-matched search-only speedup of each deduplicating arm.

    S = T_search(native) / T_search(arm) on the same (method, problem, seed).
    """
    index = {(r["method"], r["problem"], r["arm"], int(r["seed"])): r for r in rows}
    out: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (method, problem, arm, seed), row in index.items():
        if arm == "baseline":
            continue
        base = index.get((method, problem, "baseline", seed))
        if base is None or not base["search_s"] or not row["search_s"]:
            continue
        out[(method, arm, problem)].append(base["search_s"] / row["search_s"])
    return out


def build_wall_ratio(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[float]]:
    """Seed-matched ratio of total wall clock, native over the given arm.

    Above 1 the deduplicating arm finished sooner. This is the quantity the
    discussion refers to when it counts the problems on which IsalSR is the
    faster of the two, and it differs from S in that it charges the key.
    """
    index = {(r["method"], r["problem"], r["arm"], int(r["seed"])): r for r in rows}
    out: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (method, problem, arm, seed), row in index.items():
        if arm == "baseline":
            continue
        base = index.get((method, problem, "baseline", seed))
        if base is None or not base["wall_s"] or not row["wall_s"]:
            continue
        out[(method, arm, problem)].append(base["wall_s"] / row["wall_s"])
    return out


def build_speedup_by_saturation(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Split S by whether both arms of a pair ran out the time budget.

    Where both arms sit at the cap the ratio is 1 whatever the method does, so
    the two strata answer different questions and are never pooled.
    """
    index = {(r["method"], r["problem"], r["arm"], int(r["seed"])): r for r in rows}
    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"saturated": [], "unsaturated": []}
    )
    for (method, problem, arm, seed), row in index.items():
        if arm == "baseline":
            continue
        base = index.get((method, problem, "baseline", seed))
        if base is None or not base["search_s"] or not row["search_s"]:
            continue
        both_at_cap = base["wall_s"] > SATURATED_S and row["wall_s"] > SATURATED_S
        key = "saturated" if both_at_cap else "unsaturated"
        buckets[(method, arm)][key].append(base["search_s"] / row["search_s"])

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, split in buckets.items():
        out[key] = {
            "n_saturated": len(split["saturated"]),
            "n_unsaturated": len(split["unsaturated"]),
            "S_saturated": stats(split["saturated"]),
            "S_unsaturated": stats(split["unsaturated"]),
        }
    return out


def build_phi(per_problem: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (method, problem), combine the two arms' reduction factors."""
    index = {(p["method"], p["problem"], p["arm"]): p for p in per_problem}
    out: list[dict[str, Any]] = []
    for method in METHODS:
        problems = sorted({p["problem"] for p in per_problem if p["method"] == method})
        for problem in problems:
            iso = index[(method, problem, "isalsr")]
            ser = index[(method, problem, "hash")]
            r, r_ser = iso["r_mean"], ser["r_mean"]
            delta_r = r - r_ser
            out.append(
                {
                    "method": method,
                    "problem": problem,
                    "suite": iso["suite"],
                    "in_core_50": iso["in_core_50"],
                    "n_cand_isalsr": iso["n_cand_mean"],
                    "n_cand_hash": ser["n_cand_mean"],
                    "rho": iso["rho_mean"],
                    "rho_ser": ser["rho_mean"],
                    "r": r,
                    "r_ser": r_ser,
                    "delta_r": delta_r,
                    "phi": delta_r / r if r > 0 else math.nan,
                    "key_ms_isalsr": iso["key_ms_mean"],
                    "key_ms_hash": ser["key_ms_mean"],
                }
            )
    return out


def build_k_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overhead and per-candidate key cost by the maximum k a run reached."""
    out: list[dict[str, Any]] = []
    for method in METHODS:
        for arm in ("hash", "isalsr"):
            for lo, hi in K_RANGES:
                subset = [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["arm"] == arm
                    and r["max_k"] is not None
                    and lo <= r["max_k"] < hi
                ]
                if not subset:
                    continue
                overhead = stats([r["overhead_pct"] for r in subset])
                key = stats([r["key_ms"] for r in subset])
                rho = stats([r["rho"] for r in subset])
                out.append(
                    {
                        "method": method,
                        "arm": arm,
                        "k_range": f"[{lo},{hi})",
                        "n_runs": len(subset),
                        "overhead_pct_mean": overhead["mean"],
                        "overhead_pct_median": overhead["median"],
                        "key_ms_mean": key["mean"],
                        "key_ms_median": key["median"],
                        "rho_mean": rho["mean"],
                    }
                )
    return out


#: Per-problem paired-statistics files, one per contrast.
PAIRED_FILES = {
    "isalsr_vs_baseline": "paired_stats.json",
    "isalsr_vs_hash": "paired_stats_isalsr_vs_hash.json",
    "hash_vs_baseline": "paired_stats_hash_vs_baseline.json",
}

#: Metrics kept from the per-problem paired statistics.
PAIRED_METRICS = (
    "r2_test",
    "r2_train",
    "nrmse_test",
    "empirical_reduction_factor",
    "redundancy_rate",
    "wall_clock_total_s",
    "solution_recovered",
)


def load_per_problem_paired(corpus: Path, suites: tuple[str, ...]) -> list[dict[str, Any]]:
    """Collect the per-problem paired statistics of all three contrasts.

    These are descriptive: the manuscript's inference is the paired test across
    problems, and the per-problem effect sizes only say how the portfolio-level
    signal distributes.
    """
    out: list[dict[str, Any]] = []
    for method in METHODS:
        for suite in suites:
            suite_dir = corpus / method / suite
            for problem_dir in sorted(suite_dir.iterdir()):
                if not problem_dir.is_dir():
                    continue
                for contrast, filename in PAIRED_FILES.items():
                    path = problem_dir / filename
                    if not path.is_file():
                        continue
                    with path.open(encoding="utf-8") as handle:
                        doc = json.load(handle)
                    for metric in PAIRED_METRICS:
                        rec = doc["metrics"].get(metric)
                        if not rec:
                            continue
                        out.append(
                            {
                                "method": method,
                                "suite": suite,
                                "problem": doc["problem"],
                                "contrast": contrast,
                                "metric": metric,
                                "n_seeds": doc["n_seeds"],
                                "mean_diff": rec["mean_diff"],
                                "cohens_d": rec["cohens_d"],
                                "d_lo": rec["cohens_d_ci_lower"],
                                "d_hi": rec["cohens_d_ci_upper"],
                                "p_raw": rec["p_value_raw"],
                                "p_holm": rec["p_value_holm"],
                                "test": rec["test_used"],
                            }
                        )
    return out


def load_cpdt(pipeline_dir: Path) -> list[dict[str, Any]]:
    """Flatten the pipeline's paired-test records for both suite sizes."""
    out: list[dict[str, Any]] = []
    for view, n_expected in (("70", 70), ("50", 50)):
        for method in METHODS:
            path = pipeline_dir / f"flat{view}" / f"cross_problem_dominance_{method}_benchmark.json"
            with path.open(encoding="utf-8") as handle:
                doc = json.load(handle)
            blocks = {"isalsr_vs_baseline": doc, **doc.get("contrasts", {})}
            for contrast in CONTRASTS:
                block = blocks.get(contrast, {})
                for metric in CPDT_METRICS:
                    record = block.get(metric)
                    if not record:
                        continue
                    if record["n_problems"] != n_expected:
                        raise SystemExit(
                            f"{path}: {contrast}/{metric} has "
                            f"{record['n_problems']} problems, expected {n_expected}"
                        )
                    out.append(
                        {
                            "suite_size": n_expected,
                            "method": method,
                            "contrast": contrast,
                            "metric": metric,
                            "test": record["test_used"],
                            "alternative": record["alternative"],
                            "n_problems": record["n_problems"],
                            "n_wins": record["n_wins"],
                            "n_ties": record["n_ties"],
                            "n_losses": record["n_losses"],
                            "cohens_d": record["cohens_d"],
                            "d_lo": record["cohens_d_ci_lower"],
                            "d_hi": record["cohens_d_ci_upper"],
                            "mean_delta": record["mean_delta"],
                            "p_one_sided": record["p_value_one_sided"],
                            "p_two_sided": record["p_value_two_sided"],
                            "p_holm": record.get("p_value_holm"),
                        }
                    )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of uniform records as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name:24s} {len(rows):5d} rows")


def build_summary(
    rows: list[dict[str, Any]],
    per_problem: list[dict[str, Any]],
    phi: list[dict[str, Any]],
    speedup: dict[tuple[str, str, str], list[float]],
    wall_ratio: dict[tuple[str, str, str], list[float]],
    by_saturation: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the method-level summary the manuscript quotes from."""
    n_problems = len({p["problem"] for p in per_problem})
    summary: dict[str, Any] = {
        "n_cells": len(rows),
        "n_problems": n_problems,
        "n_seeds": len({int(r["seed"]) for r in rows}),
    }
    for method in METHODS:
        method_pp = [p for p in per_problem if p["method"] == method]
        method_phi = [p for p in phi if p["method"] == method]
        block: dict[str, Any] = {}
        for arm in ARMS:
            arm_pp = [p for p in method_pp if p["arm"] == arm]
            arm_cells = [r for r in rows if r["method"] == method and r["arm"] == arm]
            block[arm] = {
                "n_problems": len(arm_pp),
                "n_cells": len(arm_cells),
                "rho_over_problems": stats([p["rho_mean"] for p in arm_pp]),
                "r_over_problems": stats([p["r_mean"] for p in arm_pp]),
                "rho_over_cells": stats([c["rho"] for c in arm_cells]),
                "r_over_cells": stats([c["r"] for c in arm_cells]),
                "overhead_pct_over_cells": stats([c["overhead_pct"] for c in arm_cells]),
                "overhead_pct_over_problems": stats([p["overhead_pct_mean"] for p in arm_pp]),
                "key_ms_over_cells": stats([c["key_ms"] for c in arm_cells]),
                "eval_ms_over_cells": stats([c["eval_ms"] for c in arm_cells]),
                "wall_s_over_cells": stats([c["wall_s"] for c in arm_cells]),
                "r2_test_mean_over_problems": stats([p["r2_test_mean"] for p in arm_pp])["mean"],
                "n_rho_gt_1": sum(1 for p in arm_pp if p["rho_mean"] > 1.0),
                "solution_rate": stats([p["solution_recovered_mean"] for p in arm_pp])["mean"],
                "n_canon_timeouts": sum(int(c["n_canon_timeouts"] or 0) for c in arm_cells),
                "n_canon_raised": sum(int(c["n_canon_raised"] or 0) for c in arm_cells),
                "n_conversion_failures": sum(
                    int(c["n_conversion_failures"] or 0) for c in arm_cells
                ),
                "max_k_seen": max(int(p["max_k_seen"]) for p in arm_pp),
            }
            paired = [
                s
                for (m, a, _p), values in speedup.items()
                if m == method and a == arm
                for s in values
            ]
            if paired:
                per_prob = {
                    p: st.median(values)
                    for (m, a, p), values in speedup.items()
                    if m == method and a == arm
                }
                block[arm]["S_over_cells"] = stats(paired)
                block[arm]["S_over_problems"] = stats(list(per_prob.values()))
                block[arm]["n_problems_S_ge_1"] = sum(1 for v in per_prob.values() if v >= 1.0)
                block[arm]["problems_S_ge_1"] = sorted(p for p, v in per_prob.items() if v >= 1.0)
                block[arm]["S_by_saturation"] = by_saturation.get((method, arm), {})
            block[arm]["n_cells_at_budget_cap"] = sum(
                1 for c in arm_cells if c["wall_s"] > SATURATED_S
            )
            walls = {
                p: st.median(values)
                for (m, a, p), values in wall_ratio.items()
                if m == method and a == arm
            }
            if walls:
                block[arm]["wall_ratio_over_problems"] = stats(list(walls.values()))
                block[arm]["n_problems_faster_than_native"] = sum(
                    1 for v in walls.values() if v > 1.0
                )
                block[arm]["problems_faster_than_native"] = sorted(
                    p for p, v in walls.items() if v > 1.0
                )

        block["phi"] = {
            "over_problems": stats([p["phi"] for p in method_phi]),
            "delta_r_over_problems": stats([p["delta_r"] for p in method_phi]),
            "lowest": sorted(((p["problem"], p["phi"]) for p in method_phi), key=lambda t: t[1])[
                :5
            ],
            "highest": sorted(((p["problem"], p["phi"]) for p in method_phi), key=lambda t: -t[1])[
                :5
            ],
        }
        ratios = [
            p["key_ms_isalsr"] / p["key_ms_hash"]
            for p in method_phi
            if p["key_ms_hash"] and p["key_ms_hash"] > 0
        ]
        block["key_cost"] = {
            "isalsr_ms": stats([p["key_ms_isalsr"] for p in method_phi]),
            "hash_ms": stats([p["key_ms_hash"] for p in method_phi]),
            "isalsr_over_hash": stats(ratios),
        }
        summary[method] = block
    return summary


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    data_dir = args.analyses / "data"
    rows = read_cells(data_dir / "cells.csv")
    print(f"{len(rows)} cells")

    per_problem = build_per_problem(rows)
    write_csv(data_dir / "per_problem.csv", per_problem)

    speedup = build_speedup(rows)
    write_csv(
        data_dir / "speedup.csv",
        [
            {
                "method": m,
                "arm": a,
                "problem": p,
                "n_pairs": len(v),
                "S_mean": st.fmean(v),
                "S_median": st.median(v),
            }
            for (m, a, p), v in sorted(speedup.items())
        ],
    )

    phi = build_phi(per_problem)
    write_csv(data_dir / "phi.csv", phi)
    write_csv(data_dir / "overhead_by_k.csv", build_k_strata(rows))
    write_csv(data_dir / "cpdt.csv", load_cpdt(args.analyses / "pipeline"))
    write_csv(
        data_dir / "per_problem_paired.csv",
        load_per_problem_paired(args.corpus, SUITES),
    )

    summary = build_summary(
        rows,
        per_problem,
        phi,
        speedup,
        build_wall_ratio(rows),
        build_speedup_by_saturation(rows),
    )
    out = args.analyses / "values" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    print(f"  {out.name}")

    for method in METHODS:
        block = summary[method]
        print(f"\n=== {method} ===")
        for arm in ARMS:
            arm_block = block[arm]
            rho = arm_block["rho_over_problems"]
            line = (
                f"  {arm:9s} rho {rho['mean']:.4f} +- {rho['std']:.4f} "
                f"[{rho['min']:.3f},{rho['max']:.3f}]  "
                f"r {100 * arm_block['r_over_problems']['mean']:.2f}%  "
                f"OH med {arm_block['overhead_pct_over_cells']['median']:.3f}%  "
                f"key {arm_block['key_ms_over_cells']['median']:.4f} ms  "
                f"eval {arm_block['eval_ms_over_cells']['median']:.3f} ms"
            )
            if "S_over_problems" in arm_block:
                sat = arm_block["S_by_saturation"]
                line += (
                    f"  S med {arm_block['S_over_problems']['median']:.3f} "
                    f"(>=1 on {arm_block['n_problems_S_ge_1']}/{arm_block['n_problems']})"
                )
                line += (
                    f"\n            S unsaturated {sat['S_unsaturated']['median']:.3f} "
                    f"(n={sat['n_unsaturated']})  saturated cells {sat['n_saturated']}"
                )
            print(line)
        phi_block = block["phi"]["over_problems"]
        print(
            f"  phi mean {phi_block['mean']:.4f} median {phi_block['median']:.4f} "
            f"[{phi_block['min']:.4f},{phi_block['max']:.4f}]"
        )
        cost = block["key_cost"]
        print(
            f"  key cost IsalSR {cost['isalsr_ms']['mean']:.4f} ms vs hash "
            f"{cost['hash_ms']['mean']:.4f} ms  ratio {cost['isalsr_over_hash']['mean']:.2f}x"
        )


if __name__ == "__main__":
    main()
