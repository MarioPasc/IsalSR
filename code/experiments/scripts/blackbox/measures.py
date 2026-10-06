"""Tidy cells and every derived quantity of the black-box appendix.

The per-cell definitions (rho, r, the per-candidate key cost, the per-evaluation
cost, the overhead share, R^2 clipped to [0, 1]) are those of
``review_campaign.extract_cells``; the aggregates (per-dataset means, the share
phi, the seed-matched search-only speedup S and wall-clock ratio, the
saturation split of S, the per-host summary) are those of
``review_campaign.derive``. Both are called unchanged, on this corpus and on the
main campaign's cell table, so the two campaigns are compared through one code
path.

Quantities the main campaign's summary does not carry are added here and kept
apart, under keys of their own:

* NRMSE and training R^2 per arm, and the ratio of median costs
  ``T_eval / T_canon``, which the manuscript quotes from the two medians;
* why each run stopped (``stopping``) and how many generations a Bingo run
  completed, which the reading of the wall-clock result needs (post hoc);
* the rank correlation between dataset size and S on Bingo (post hoc).
"""

from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path
from typing import Any

from scipy import stats as sp_stats

from benchmarks.datasets.srbench_blackbox import SRBENCH_BLACKBOX_DATASETS
from experiments.scripts.blackbox.audit import cell_dir
from experiments.scripts.blackbox.config import BENCHMARK, DATASET_DIR, SEEDS
from experiments.scripts.review_campaign import derive as c2
from experiments.scripts.review_campaign.config import ARMS, METHODS
from experiments.scripts.review_campaign.extract_cells import row_from_log

Row = dict[str, Any]

#: Per-cell fields kept beside the main campaign's cell table.
EXTRA_COLUMNS: tuple[str, ...] = (
    "method",
    "problem",
    "arm",
    "seed",
    "dataset",
    "termination",
    "generations",
)

DATASET_SIZE: dict[str, int] = {name: n for name, n, _ in SRBENCH_BLACKBOX_DATASETS}


def last_generation(trajectory: Path) -> int | None:
    """Generation count of a run, read from the last row of its trajectory.

    Args:
        trajectory: ``trajectory.csv`` of one cell.

    Returns:
        The ``iteration`` field of the last data row, or None if absent.
    """
    if not trajectory.is_file():
        return None
    with trajectory.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return int(float(rows[-1]["iteration"])) if rows else None


#: Key of a stop reason: (method, corpus problem directory, arm, seed).
StopKey = tuple[str, str, str, int]


def extract_cells(
    corpus: Path, included: dict[str, list[str]], reasons: dict[StopKey, str]
) -> tuple[list[Row], list[Row]]:
    """Read every admitted cell into the main campaign's tidy format.

    Args:
        corpus: Corpus root (read-only).
        included: Host -> datasets admitted by the audit.
        reasons: Stop reason of every cell, from ``stopping.scan``.

    Returns:
        The cell rows (``extract_cells.COLUMNS``) and the extra per-cell rows
        (``EXTRA_COLUMNS``), in the same order.
    """
    rows: list[Row] = []
    extra: list[Row] = []
    for method in METHODS:
        for dataset in included[method]:
            for arm in ARMS:
                for seed in SEEDS:
                    directory = cell_dir(corpus, method, dataset, arm, seed)
                    row = row_from_log(directory / "run_log.json", BENCHMARK)
                    rows.append(row)
                    generations = (
                        last_generation(directory / "trajectory.csv") if method == "bingo" else None
                    )
                    extra.append(
                        {
                            "method": method,
                            "problem": row["problem"],
                            "arm": arm,
                            "seed": seed,
                            "dataset": dataset,
                            "termination": reasons[(method, DATASET_DIR[dataset], arm, seed)],
                            "generations": generations,
                        }
                    )
    return rows, extra


def write_csv(path: Path, rows: list[Row], columns: tuple[str, ...] | None = None) -> None:
    """Write uniform records as CSV, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(columns) if columns else list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def derive_all(rows: list[Row]) -> dict[str, Any]:
    """Run the main campaign's derivations on a coerced cell table.

    Args:
        rows: Cells as returned by ``review_campaign.derive.read_cells``.

    Returns:
        ``per_problem``, ``phi``, ``speedup``, ``wall_ratio``, ``saturation``
        and the method-level ``summary``.
    """
    per_problem = c2.build_per_problem(rows)
    phi = c2.build_phi(per_problem)
    speedup = c2.build_speedup(rows)
    wall_ratio = c2.build_wall_ratio(rows)
    saturation = c2.build_speedup_by_saturation(rows)
    summary = c2.build_summary(rows, per_problem, phi, speedup, wall_ratio, saturation)
    return {
        "per_problem": per_problem,
        "phi": phi,
        "speedup": speedup,
        "wall_ratio": wall_ratio,
        "saturation": saturation,
        "summary": summary,
    }


def _median(values: list[float]) -> float:
    xs = [v for v in values if v is not None and math.isfinite(v)]
    return st.median(xs) if xs else math.nan


def governing(summary: dict[str, Any], method: str) -> dict[str, float]:
    """The quantities the comparison with the main campaign is made on.

    Definitions follow Tables 2, 4 and 5 and Section 5.4 of the manuscript:
    reduction factors are means over per-problem means, costs and overhead are
    medians over the cells of the IsalSR arm, S is the median over cells, the
    wall-clock ratio is the median over problems of the per-problem median.

    Args:
        summary: A ``build_summary`` result.
        method: Host name.

    Returns:
        Flat mapping of quantity name to value.
    """
    block = summary[method]
    iso, hsh = block["isalsr"], block["hash"]
    t_canon = iso["key_ms_over_cells"]["median"]
    t_eval = iso["eval_ms_over_cells"]["median"]
    sat = iso.get("S_by_saturation", {})
    return {
        "n_problems": float(iso["n_problems"]),
        "rho": iso["rho_over_problems"]["mean"],
        "rho_std": iso["rho_over_problems"]["std"],
        "rho_min": iso["rho_over_problems"]["min"],
        "rho_max": iso["rho_over_problems"]["max"],
        "r": iso["r_over_problems"]["mean"],
        "rho_ser": hsh["rho_over_problems"]["mean"],
        "rho_ser_std": hsh["rho_over_problems"]["std"],
        "r_ser": hsh["r_over_problems"]["mean"],
        "phi": block["phi"]["over_problems"]["mean"],
        "phi_median": block["phi"]["over_problems"]["median"],
        "phi_min": block["phi"]["over_problems"]["min"],
        "phi_max": block["phi"]["over_problems"]["max"],
        "t_canon_ms": t_canon,
        "t_ser_ms": hsh["key_ms_over_cells"]["median"],
        "t_eval_ms": t_eval,
        "t_eval_over_t_canon": t_eval / t_canon if t_canon else math.nan,
        "oh_pct": iso["overhead_pct_over_cells"]["median"],
        "oh_hash_pct": hsh["overhead_pct_over_cells"]["median"],
        "s_median": iso["S_over_cells"]["median"],
        "s_hash_median": hsh["S_over_cells"]["median"],
        "n_saturated": float(sat.get("n_saturated", 0)),
        "n_unsaturated": float(sat.get("n_unsaturated", 0)),
        "s_unsaturated_median": sat.get("S_unsaturated", {}).get("median", math.nan),
        "wall_ratio": iso["wall_ratio_over_problems"]["median"],
        "wall_ratio_hash": hsh["wall_ratio_over_problems"]["median"],
        "n_faster": float(iso["n_problems_faster_than_native"]),
        "n_faster_hash": float(hsh["n_problems_faster_than_native"]),
        "n_s_ge_1": float(iso["n_problems_S_ge_1"]),
        "r2_na": block["baseline"]["r2_test_mean_over_problems"],
        "r2_nh": hsh["r2_test_mean_over_problems"],
        "r2_is": iso["r2_test_mean_over_problems"],
    }


def quality_by_arm(per_problem: list[Row]) -> dict[str, dict[str, dict[str, float]]]:
    """Mean and median over datasets of the per-dataset quality means.

    NRMSE is not clipped, so a single run that extrapolates badly can dominate
    a mean over datasets; the median is reported beside it for that reason.

    Args:
        per_problem: ``build_per_problem`` rows.

    Returns:
        Host -> arm -> statistic name -> value.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for method in METHODS:
        out[method] = {}
        for arm in ARMS:
            rows = [p for p in per_problem if p["method"] == method and p["arm"] == arm]
            out[method][arm] = {
                "r2_test_mean": st.fmean(p["r2_test_mean"] for p in rows),
                "r2_test_median": _median([p["r2_test_mean"] for p in rows]),
                "r2_train_mean": st.fmean(p["r2_train_mean"] for p in rows),
                "nrmse_test_mean": st.fmean(p["nrmse_test_mean"] for p in rows),
                "nrmse_test_median": _median([p["nrmse_test_mean"] for p in rows]),
                "wall_h_median": _median([p["wall_s_mean"] for p in rows]) / 3600.0,
            }
    return out


def nrmse_max_cell(rows: list[Row]) -> dict[str, dict[str, Row]]:
    """The single run with the largest test NRMSE, per host and arm.

    Args:
        rows: Cell rows.

    Returns:
        Host -> arm -> ``{"problem", "seed", "nrmse_test"}`` of that run.
    """
    out: dict[str, dict[str, Row]] = {}
    for method in METHODS:
        out[method] = {}
        for arm in ARMS:
            cells = [r for r in rows if r["method"] == method and r["arm"] == arm]
            worst = max(cells, key=lambda r: r["nrmse_test"])
            out[method][arm] = {
                "problem": worst["problem"],
                "seed": int(worst["seed"]),
                "nrmse_test": worst["nrmse_test"],
            }
    return out


def termination_summary(extra: list[Row]) -> dict[str, Any]:
    """Tally the stop reasons of the admitted cells per host and arm.

    Args:
        extra: Extra per-cell rows.

    Returns:
        Per host and arm: the tally of stop reasons, and per reason the
        datasets on which at least one run stopped that way.
    """
    out: dict[str, Any] = {}
    for method in METHODS:
        out[method] = {}
        for arm in ARMS:
            cells = [e for e in extra if e["method"] == method and e["arm"] == arm]
            tally = Counter(e["termination"] for e in cells)
            datasets = {
                reason: sorted({e["dataset"] for e in cells if e["termination"] == reason})
                for reason in tally
            }
            out[method][arm] = {"tally": dict(tally), "datasets": datasets}
    return out


def generation_ratios(extra: list[Row]) -> dict[str, Any]:
    """Seed-matched ratio of generations, deduplicating arm over native, on Bingo.

    Args:
        extra: Extra per-cell rows.

    Returns:
        Per deduplicating arm: the median over datasets of the per-dataset
        median ratio, its range, and the median generation count per arm.
    """
    index = {
        (e["dataset"], e["arm"], int(e["seed"])): e["generations"]
        for e in extra
        if e["method"] == "bingo"
    }
    out: dict[str, Any] = {}
    datasets = sorted({e["dataset"] for e in extra if e["method"] == "bingo"})
    for arm in ("hash", "isalsr"):
        per_dataset = []
        for dataset in datasets:
            ratios = [
                index[(dataset, arm, s)] / index[(dataset, "baseline", s)]
                for s in SEEDS
                if index.get((dataset, arm, s)) and index.get((dataset, "baseline", s))
            ]
            if ratios:
                per_dataset.append(st.median(ratios))
        out[arm] = {
            "median_over_datasets": _median(per_dataset),
            "min": min(per_dataset),
            "max": max(per_dataset),
        }
    for arm in ARMS:
        gens = [g for (_, a, _), g in index.items() if a == arm and g]
        out[f"generations_median_{arm}"] = _median([float(g) for g in gens])
    return out


def size_vs_speedup(
    speedup: dict[tuple[str, str, str], list[float]], rows: list[Row]
) -> dict[str, Any]:
    """Rank correlation between dataset size and the median S on Bingo (post hoc).

    Args:
        speedup: ``build_speedup`` result.
        rows: Cell rows (to map run-log problem names to dataset sizes).

    Returns:
        Spearman coefficient and two-sided p per deduplicating arm, with N.
    """
    names = {r["problem"] for r in rows if r["method"] == "bingo"}
    size = {name: DATASET_SIZE[name] for name in names if name in DATASET_SIZE}
    out: dict[str, Any] = {}
    for arm in ("hash", "isalsr"):
        pairs = [
            (size[p], st.median(v))
            for (m, a, p), v in speedup.items()
            if m == "bingo" and a == arm and p in size
        ]
        res = sp_stats.spearmanr([n for n, _ in pairs], [s for _, s in pairs])
        out[arm] = {
            "spearman": float(res.statistic),
            "p_two_sided": float(res.pvalue),
            "n": len(pairs),
        }
    return out


def c2_comparator(c2_analyses: Path) -> dict[str, Any]:
    """Recompute the main campaign's governing quantities from its cell table.

    The recomputation uses the same functions as for the black-box corpus and
    is checked against the published ``values/summary.json``.

    Args:
        c2_analyses: Main campaign's analysis tree (read-only).

    Returns:
        ``governing`` per host and the published-value check.
    """
    rows = c2.read_cells(c2_analyses / "data" / "cells.csv")
    derived = derive_all(rows)
    summary = derived["summary"]
    with (c2_analyses / "values" / "summary.json").open(encoding="utf-8") as handle:
        published = json.load(handle)
    gov = {m: governing(summary, m) for m in METHODS}
    gov_published = {m: governing(published, m) for m in METHODS}
    mismatches = [
        f"{m}.{k}: {gov[m][k]!r} vs {gov_published[m][k]!r}"
        for m in METHODS
        for k in gov[m]
        if not (
            (math.isnan(gov[m][k]) and math.isnan(gov_published[m][k]))
            or math.isclose(gov[m][k], gov_published[m][k], rel_tol=1e-9, abs_tol=1e-12)
        )
    ]
    return {
        "governing": gov,
        "matches_published_summary": not mismatches,
        "mismatches": mismatches,
        "n_cells": len(rows),
    }
