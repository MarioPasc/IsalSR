"""Collect and summarize paired Bingo hard-benchmark results (baseline vs IsalSR).

For each (problem, seed) pair, loads both run_log.json files, extracts the
metrics of interest, and computes signed differences (isalsr - baseline).
Outputs a tidy CSV and a per-problem summary table to stdout.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path(
    "/media/mpascual/Sandisk2TB/research/isalsr/results/"
    "model_validation/wl_subtree_hard/models_hard/bingo/hard"
)
OUTPUT_CSV = Path(
    "/home/mpascual/research/code/IsalSR/experiments/scripts/hard_bingo_paired_data.csv"
)

PROBLEMS = [
    "i.15.10",
    "i.30.3",
    "i.37.4",
    "ii.11.27",
    "iii.17.37",
    "keijzer_6",
    "korns_12",
    "pagie_1",
    "vladislavleva_2",
    "vladislavleva_4",
]

N_SEEDS = 30


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _safe(d: dict, *keys: str, default: float = float("nan")) -> float:
    """Traverse nested dict with dot-path keys; return default on any miss."""
    cur: object = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return float(cur) if cur is not None else default


def extract_metrics(log_data: dict) -> dict[str, float | bool]:
    """Extract the scalar metrics we care about from a run_log dict."""
    r = log_data.get("results", {})
    reg = r.get("regression", {})
    time_ = r.get("time", {})
    ss = r.get("search_space", {})

    return {
        "r2_train": _safe(reg, "r2_train"),
        "r2_test": _safe(reg, "r2_test"),
        "nrmse_train": _safe(reg, "nrmse_train"),
        "nrmse_test": _safe(reg, "nrmse_test"),
        "solution_recovered": bool(reg.get("solution_recovered", False)),
        "model_complexity": _safe(reg, "model_complexity"),
        "wall_clock_total_s": _safe(time_, "wall_clock_total_s"),
        "wall_clock_search_only_s": _safe(time_, "wall_clock_search_only_s"),
        "overhead_time_s": _safe(time_, "overhead_time_s"),
        "canonicalization_runtime_s": _safe(time_, "canonicalization_runtime_s"),
        "total_dags_explored": _safe(ss, "total_dags_explored"),
        "unique_canonical_dags": _safe(ss, "unique_canonical_dags"),
        "empirical_reduction_factor": _safe(ss, "empirical_reduction_factor"),
        "max_internal_nodes_seen": _safe(ss, "max_internal_nodes_seen"),
        "redundancy_rate": _safe(ss, "redundancy_rate"),
    }


def load_run_log(path: Path) -> dict | None:
    """Load a run_log.json; return None if missing or malformed."""
    if not path.exists():
        log.warning("Missing: %s", path)
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        log.warning("Malformed JSON (%s): %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------


def collect_paired_rows() -> list[dict]:
    """Iterate (problem, seed) and build one row per pair."""
    rows: list[dict] = []

    for problem in PROBLEMS:
        prob_dir = RESULTS_ROOT / problem
        for seed_idx in range(1, N_SEEDS + 1):
            seed_str = f"seed_{seed_idx:02d}"

            bl_path = prob_dir / "baseline" / seed_str / "run_log.json"
            isr_path = prob_dir / "isalsr" / seed_str / "run_log.json"

            bl_data = load_run_log(bl_path)
            isr_data = load_run_log(isr_path)

            if bl_data is None or isr_data is None:
                log.warning("Skipping %s / %s (missing data)", problem, seed_str)
                continue

            bl = extract_metrics(bl_data)
            isr = extract_metrics(isr_data)

            row: dict = {
                "problem": problem,
                "seed": seed_idx,
                # --- IsalSR raw metrics (search-space) ---
                "reduction_factor": isr["empirical_reduction_factor"],
                "redundancy_rate": isr["redundancy_rate"],
                "max_internal_nodes_seen_isalsr": isr["max_internal_nodes_seen"],
                "total_dags_explored_isalsr": isr["total_dags_explored"],
                "unique_canonical_dags": isr["unique_canonical_dags"],
                # --- Per-variant regression metrics ---
                "r2_train_baseline": bl["r2_train"],
                "r2_train_isalsr": isr["r2_train"],
                "r2_test_baseline": bl["r2_test"],
                "r2_test_isalsr": isr["r2_test"],
                "nrmse_train_baseline": bl["nrmse_train"],
                "nrmse_train_isalsr": isr["nrmse_train"],
                "nrmse_test_baseline": bl["nrmse_test"],
                "nrmse_test_isalsr": isr["nrmse_test"],
                # --- Signed differences (isalsr - baseline; positive = IsalSR wins) ---
                "delta_r2_train": isr["r2_train"] - bl["r2_train"],
                "delta_r2_test": isr["r2_test"] - bl["r2_test"],
                "delta_nrmse_train": isr["nrmse_train"] - bl["nrmse_train"],
                "delta_nrmse_test": isr["nrmse_test"] - bl["nrmse_test"],
                # --- Model complexity ---
                "model_complexity_baseline": bl["model_complexity"],
                "model_complexity_isalsr": isr["model_complexity"],
                # --- Timing ---
                "wall_clock_search_only_s_baseline": bl["wall_clock_search_only_s"],
                "wall_clock_search_only_s_isalsr": isr["wall_clock_search_only_s"],
                "overhead_time_s": isr["overhead_time_s"],
                "canonicalization_runtime_s": isr["canonicalization_runtime_s"],
                # --- Solution recovery ---
                "solution_recovered_baseline": bl["solution_recovered"],
                "solution_recovered_isalsr": isr["solution_recovered"],
            }
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; NaN if undetermined."""
    diff = a - b
    if np.all(diff == 0) or len(diff) < 2:
        return float("nan")
    try:
        return float(stats.wilcoxon(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable per-problem summary table."""
    problems = df["problem"].unique()

    header = (
        f"{'Problem':<18} {'RF':>6} {'RedRate':>8} "
        f"{'dR2tr':>8} {'dR2te':>8} "
        f"{'p(R2tr)':>9} {'p(R2te)':>9} "
        f"{'Rec_bl':>6} {'Rec_isr':>7} "
        f"{'Overhead%':>10}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("BINGO Hard Benchmarks -- IsalSR vs Baseline Paired Summary (N=30 seeds)")
    print(sep)
    print(header)
    print(sep)

    for prob in sorted(problems):
        sub = df[df["problem"] == prob]

        rf_mean = sub["reduction_factor"].mean()
        rr_mean = sub["redundancy_rate"].mean()

        delta_r2_tr = sub["delta_r2_train"]
        delta_r2_te = sub["delta_r2_test"]

        # Wilcoxon on absolute R² values (paired)
        p_r2_tr = _wilcoxon_p(sub["r2_train_isalsr"].values, sub["r2_train_baseline"].values)
        p_r2_te = _wilcoxon_p(sub["r2_test_isalsr"].values, sub["r2_test_baseline"].values)

        rec_bl = sub["solution_recovered_baseline"].sum()
        rec_isr = sub["solution_recovered_isalsr"].sum()

        # Overhead as % of search-only time of baseline
        bl_time = sub["wall_clock_search_only_s_baseline"]
        oh_time = sub["overhead_time_s"]
        with np.errstate(divide="ignore", invalid="ignore"):
            overhead_pct = (oh_time / bl_time * 100).mean()

        print(
            f"{prob:<18} "
            f"{rf_mean:>6.3f} "
            f"{rr_mean:>8.3f} "
            f"{delta_r2_tr.mean():>+8.4f} "
            f"{delta_r2_te.mean():>+8.4f} "
            f"{p_r2_tr:>9.4f} "
            f"{p_r2_te:>9.4f} "
            f"{rec_bl:>6d} "
            f"{rec_isr:>7d} "
            f"{overhead_pct:>9.1f}%"
        )

    print(sep)

    # Global aggregates
    rf_global = df["reduction_factor"].mean()
    rr_global = df["redundancy_rate"].mean()
    p_tr_glob = _wilcoxon_p(df["r2_train_isalsr"].values, df["r2_train_baseline"].values)
    p_te_glob = _wilcoxon_p(df["r2_test_isalsr"].values, df["r2_test_baseline"].values)
    rec_bl_g = df["solution_recovered_baseline"].sum()
    rec_isr_g = df["solution_recovered_isalsr"].sum()
    oh_pct_g = (df["overhead_time_s"] / df["wall_clock_search_only_s_baseline"] * 100).mean()

    print(
        f"{'GLOBAL (all)':<18} "
        f"{rf_global:>6.3f} "
        f"{rr_global:>8.3f} "
        f"{df['delta_r2_train'].mean():>+8.4f} "
        f"{df['delta_r2_test'].mean():>+8.4f} "
        f"{p_tr_glob:>9.4f} "
        f"{p_te_glob:>9.4f} "
        f"{rec_bl_g:>6d} "
        f"{rec_isr_g:>7d} "
        f"{oh_pct_g:>9.1f}%"
    )
    print(sep)
    print(
        "Columns: RF=mean reduction factor, RedRate=mean redundancy rate, "
        "dR2tr/te=mean(isalsr-baseline) R², p=Wilcoxon p-value, "
        "Rec_bl/isr=seeds with solution recovered, Overhead%=canon/baseline_search_time"
    )
    print(sep + "\n")

    # Additional: distribution of reduction factors
    print("Reduction factor distribution (all problems × seeds):")
    print(df["reduction_factor"].describe().to_string())
    print()
    print("Redundancy rate distribution:")
    print(df["redundancy_rate"].describe().to_string())
    print()

    # Seeds where isalsr strictly dominates on R2_test
    n_win = (df["delta_r2_test"] > 0).sum()
    n_loss = (df["delta_r2_test"] < 0).sum()
    n_tie = (df["delta_r2_test"] == 0).sum()
    n_total = len(df)
    print(f"IsalSR R²_test win/loss/tie across all {n_total} pairs: {n_win}/{n_loss}/{n_tie}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("Collecting run_log.json data from: %s", RESULTS_ROOT)
    rows = collect_paired_rows()

    if not rows:
        log.error("No rows collected. Check RESULTS_ROOT path.")
        return

    df = pd.DataFrame(rows)
    log.info(
        "Collected %d paired rows (%d problems × up to 30 seeds)", len(df), df["problem"].nunique()
    )

    # Save CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved paired data to: %s", OUTPUT_CSV)

    print_summary(df)


if __name__ == "__main__":
    main()
