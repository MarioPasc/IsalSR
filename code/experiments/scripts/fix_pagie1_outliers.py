"""Diagnose and recompute Pagie-1 test R² after removing catastrophic outliers.

Identifies the outlier seeds, reports their symbolic expressions,
and recomputes Wilcoxon signed-rank test with and without outliers.
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

RESULTS_ROOT = Path(
    "/media/mpascual/Sandisk2TB/research/isalsr/results/"
    "model_validation/wl_subtree_hard/models_hard/bingo/hard/pagie_1"
)
PAIRED_CSV = Path(
    "/home/mpascual/research/code/IsalSR/experiments/scripts/hard_bingo_paired_data.csv"
)


def load_run_log(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def main() -> None:
    df = pd.read_csv(PAIRED_CSV)
    pagie = df[df["problem"] == "pagie_1"].copy()

    print("=" * 90)
    print("PAGIE-1 TEST R² OUTLIER ANALYSIS")
    print("=" * 90)

    # ── 1. Identify outlier seeds ─────────────────────────────────────────────

    print(f"\nTotal paired seeds: {len(pagie)}")
    print("\nTest R² distribution (IsalSR):")
    print(pagie["r2_test_isalsr"].describe())

    threshold = -1.0
    outliers = pagie[pagie["r2_test_isalsr"] < threshold]
    print(f"\nOutlier seeds (R² test < {threshold}):")
    for _, row in outliers.iterrows():
        print(
            f"  Seed {int(row['seed'])}: "
            f"R²_test_baseline={row['r2_test_baseline']:.6f}, "
            f"R²_test_isalsr={row['r2_test_isalsr']:.2f}, "
            f"R²_train_isalsr={row['r2_train_isalsr']:.6f}"
        )

    # Also check baseline outliers
    bl_outliers = pagie[pagie["r2_test_baseline"] < threshold]
    print(f"\nBaseline outlier seeds (R² test < {threshold}):")
    for _, row in bl_outliers.iterrows():
        print(
            f"  Seed {int(row['seed'])}: "
            f"R²_test_baseline={row['r2_test_baseline']:.2f}, "
            f"R²_train_baseline={row['r2_train_baseline']:.6f}"
        )

    # ── 2. Load symbolic expressions for outlier seeds ────────────────────────

    print("\n--- Symbolic expressions of outlier seeds ---")
    for _, row in outliers.iterrows():
        seed_idx = int(row["seed"])
        for variant in ["baseline", "isalsr"]:
            rl = load_run_log(RESULTS_ROOT / variant / f"seed_{seed_idx:02d}" / "run_log.json")
            if rl:
                expr = rl.get("best_expression", {}).get("symbolic_form", "N/A")
                r2_test = rl["results"]["regression"]["r2_test"]
                r2_train = rl["results"]["regression"]["r2_train"]
                n_nodes = rl.get("best_expression", {}).get("n_nodes", "?")
                print(
                    f"  Seed {seed_idx} ({variant}): "
                    f"R²_train={r2_train:.6f}, R²_test={r2_test:.4f}, "
                    f"n_nodes={n_nodes}"
                )
                print(f"    expr: {expr[:120]}...")

    # ── 3. Statistical tests: with and without outliers ───────────────────────

    print("\n" + "=" * 90)
    print("STATISTICAL TESTS")
    print("=" * 90)

    # WITH outliers (original)
    bl_test = pagie["r2_test_baseline"].values
    isr_test = pagie["r2_test_isalsr"].values

    try:
        stat_orig, p_orig = stats.wilcoxon(isr_test, bl_test, alternative="two-sided")
    except ValueError:
        p_orig = np.nan
        stat_orig = np.nan

    print(f"\n--- WITH outliers (n={len(pagie)}) ---")
    print(f"  Mean R² test:    baseline={np.mean(bl_test):.6f}, isalsr={np.mean(isr_test):.6f}")
    print(f"  Median R² test:  baseline={np.median(bl_test):.6f}, isalsr={np.median(isr_test):.6f}")
    print(f"  Wilcoxon (two-sided): stat={stat_orig}, p={p_orig:.6f}")

    # WITHOUT outliers
    clean = pagie[pagie["r2_test_isalsr"] >= threshold].copy()
    # Also remove baseline outliers at same seeds for paired test consistency
    outlier_seeds = set(outliers["seed"].values)
    bl_outlier_seeds = set(bl_outliers["seed"].values)
    all_outlier_seeds = outlier_seeds | bl_outlier_seeds
    clean_paired = pagie[~pagie["seed"].isin(all_outlier_seeds)].copy()

    bl_clean = clean_paired["r2_test_baseline"].values
    isr_clean = clean_paired["r2_test_isalsr"].values

    try:
        stat_clean, p_clean = stats.wilcoxon(isr_clean, bl_clean, alternative="two-sided")
    except ValueError:
        p_clean = np.nan
        stat_clean = np.nan

    print(f"\n--- WITHOUT outlier seeds {sorted(all_outlier_seeds)} (n={len(clean_paired)}) ---")
    print(f"  Mean R² test:    baseline={np.mean(bl_clean):.6f}, isalsr={np.mean(isr_clean):.6f}")
    print(
        f"  Median R² test:  baseline={np.median(bl_clean):.6f}, isalsr={np.median(isr_clean):.6f}"
    )
    print(f"  Wilcoxon (two-sided): stat={stat_clean}, p={p_clean:.6f}")

    # Cliff's delta
    n = len(bl_clean) * len(isr_clean)
    more = np.sum(isr_clean[:, None] > bl_clean[None, :])
    less = np.sum(isr_clean[:, None] < bl_clean[None, :])
    cliff_d = (more - less) / n
    print(f"  Cliff's delta: {cliff_d:+.4f}")

    # Win/loss/tie
    delta = isr_clean - bl_clean
    wins = (delta > 0).sum()
    losses = (delta < 0).sum()
    ties = (delta == 0).sum()
    print(f"  Win/loss/tie: {wins}/{losses}/{ties}")

    # ── 4. Additional robust statistics ───────────────────────────────────────

    print("\n--- Robust statistics (all seeds, trimmed/winsorized) ---")

    # Trimmed mean (10% each tail)
    from scipy.stats import trim_mean

    tm_bl = trim_mean(pagie["r2_test_baseline"].values, 0.1)
    tm_isr = trim_mean(pagie["r2_test_isalsr"].values, 0.1)
    print(f"  Trimmed mean (10%): baseline={tm_bl:.6f}, isalsr={tm_isr:.6f}")

    # Median (already robust)
    print(f"  Median: baseline={np.median(bl_test):.6f}, isalsr={np.median(isr_test):.6f}")

    # ── 5. Is Pagie-1 now sig_test? ──────────────────────────────────────────

    print("\n" + "=" * 90)
    print("CONCLUSION")
    print("=" * 90)

    alpha = 0.05
    if p_clean < alpha:
        print(f"\n  Pagie-1 IS sig_test after outlier removal (p={p_clean:.6f} < {alpha})")
    else:
        print(f"\n  Pagie-1 is NOT sig_test after outlier removal (p={p_clean:.6f} >= {alpha})")

    print("\n  sig_train: already True (from original analysis)")
    print(f"  sig_test (with outliers): p={p_orig:.6f} → {'Yes' if p_orig < alpha else 'No'}")
    print(f"  sig_test (without outliers): p={p_clean:.6f} → {'Yes' if p_clean < alpha else 'No'}")

    # Holm-Bonferroni note
    print("\n  Note: Holm-Bonferroni correction across 10 problems would require")
    print(f"  the smallest p-value to be < {alpha}/10 = {alpha / 10:.4f}.")
    print("  For Pagie-1 specifically (rank among 10 problems varies),")
    print("  corrected threshold depends on its rank in the sorted p-value list.")

    # ── 6. Train R² for completeness ─────────────────────────────────────────

    print("\n--- Train R² (for completeness) ---")
    bl_train = pagie["r2_train_baseline"].values
    isr_train = pagie["r2_train_isalsr"].values
    stat_tr, p_tr = stats.wilcoxon(isr_train, bl_train, alternative="two-sided")
    print(f"  Wilcoxon (train): stat={stat_tr}, p={p_tr:.6f}")
    print(f"  Mean train: baseline={np.mean(bl_train):.6f}, isalsr={np.mean(isr_train):.6f}")


if __name__ == "__main__":
    main()
