"""Final synthesis: Quantify the bottleneck-type hypothesis.

Hypothesis: IsalSR helps when the bottleneck is STRUCTURAL SEARCH (finding
the right operator topology), not CONSTANT OPTIMIZATION or other factors.

Quantify using:
1. Levene's test for variance reduction (the mechanism)
2. Ratio of ISR-wins at R²≥0.99 threshold (practical impact)
3. The "rescue rate": fraction of bad baseline seeds that improve with IsalSR
4. Bottleneck classification validation
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESULTS_ROOT = Path(
    "/media/mpascual/Sandisk2TB/research/isalsr/results/"
    "model_validation/wl_subtree_hard/models_hard/bingo/hard"
)
PAIRED_CSV = Path(
    "/home/mpascual/research/code/IsalSR/experiments/scripts/hard_bingo_paired_data.csv"
)

PROBLEMS_META = {
    "i.15.10": {
        "bottleneck": "structural",
        "n_nontrivial_constants": 0,
        "gt_k": 7,
        "gt_depth": 5,
        "n_vars": 3,
        "source": "feynman",
        "sig_both": True,
        "sig_train": True,
        "reason": "nested sqrt(1-ratio); const='1' trivial for LM",
    },
    "i.30.3": {
        "bottleneck": "structural",
        "n_nontrivial_constants": 0,
        "gt_k": 8,
        "gt_depth": 5,
        "n_vars": 2,
        "source": "feynman",
        "sig_both": True,
        "sig_train": True,
        "reason": "parallel sin²/sin² branches; const='2' trivial for LM",
    },
    "i.37.4": {
        "bottleneck": "structural",
        "n_nontrivial_constants": 0,
        "gt_k": 7,
        "gt_depth": 4,
        "n_vars": 3,
        "source": "feynman",
        "sig_both": True,
        "sig_train": True,
        "reason": "sqrt(product)*cos; const='2' trivial for LM",
    },
    "ii.11.27": {
        "bottleneck": "none_trivial",
        "n_nontrivial_constants": 0,
        "gt_k": 8,
        "gt_depth": 5,
        "n_vars": 4,
        "source": "feynman",
        "sig_both": False,
        "sig_train": False,
        "reason": "all 30 seeds reach R²≈1.0; no bottleneck to relieve",
    },
    "iii.17.37": {
        "bottleneck": "structural",
        "n_nontrivial_constants": 0,
        "gt_k": 8,
        "gt_depth": 5,
        "n_vars": 4,
        "source": "feynman",
        "sig_both": True,
        "sig_train": True,
        "reason": "nested sqrt(sum of squares); consts='2','4' trivial for LM",
    },
    "keijzer_6": {
        "bottleneck": "constant",
        "n_nontrivial_constants": 1,
        "gt_k": 2,
        "gt_depth": 2,
        "n_vars": 1,
        "source": "gp",
        "sig_both": False,
        "sig_train": False,
        "reason": "log(x)+γ; Euler-Mascheroni γ≈0.5772 is irrational; structure trivial",
    },
    "korns_12": {
        "bottleneck": "constant+selection",
        "n_nontrivial_constants": 4,
        "gt_k": 7,
        "gt_depth": 4,
        "n_vars": 5,
        "source": "gp",
        "sig_both": False,
        "sig_train": False,
        "reason": "9.8*x1 → 156 oscillation cycles; 4 precise real constants + 3 irrelevant vars",
    },
    "pagie_1": {
        "bottleneck": "structural",
        "n_nontrivial_constants": 0,
        "gt_k": 7,
        "gt_depth": 4,
        "n_vars": 2,
        "source": "gp",
        "sig_both": False,
        "sig_train": True,
        "reason": "sig_train but test outliers destroy test sig; x^(-4) unusual but discoverable",
    },
    "vladislavleva_2": {
        "bottleneck": "structural_depth",
        "n_nontrivial_constants": 0,
        "gt_k": 13,
        "gt_depth": 6,
        "n_vars": 1,
        "source": "gp",
        "sig_both": False,
        "sig_train": False,
        "reason": "k=13, depth=6 → k!=6.2 billion orderings; very deep heterogeneous product",
    },
    "vladislavleva_4": {
        "bottleneck": "width+constants",
        "n_nontrivial_constants": 2,
        "gt_k": 12,
        "gt_depth": 4,
        "n_vars": 5,
        "source": "gp",
        "sig_both": False,
        "sig_train": False,
        "reason": "k=12, 5×(xi-3)² requires discovering identical subexpr 5 times; only 12 seeds",
    },
}


def main() -> None:
    df = pd.read_csv(PAIRED_CSV)

    print("=" * 100)
    print("SYNTHESIS: What predicts IsalSR's advantage?")
    print("=" * 100)

    # ── 1. Variance reduction as the mechanism ────────────────────────────────

    print("\n" + "─" * 100)
    print("1. THE MECHANISM: Variance Reduction (Levene's test)")
    print("─" * 100)

    var_rows = []
    for prob in sorted(PROBLEMS_META.keys()):
        meta = PROBLEMS_META[prob]
        sub = df[df["problem"] == prob]
        bl = sub["r2_train_isalsr"].values  # intentionally: isalsr for isalsr
        bl_train = sub["r2_train_baseline"].values
        isr_train = sub["r2_train_isalsr"].values

        bl_var = np.var(bl_train)
        isr_var = np.var(isr_train)
        ratio = bl_var / max(isr_var, 1e-30)

        try:
            _, lev_p = stats.levene(bl_train, isr_train)
        except Exception:
            lev_p = np.nan

        # Rescue rate: fraction of below-median baseline seeds that improve
        median_bl = np.median(bl_train)
        bad_mask = sub["r2_train_baseline"] < median_bl
        bad_deltas = sub.loc[bad_mask, "delta_r2_train"].values
        rescue_rate = np.mean(bad_deltas > 0) if len(bad_deltas) > 0 else np.nan

        var_rows.append(
            {
                "problem": prob,
                "bottleneck": meta["bottleneck"],
                "sig_train": meta["sig_train"],
                "sig_both": meta["sig_both"],
                "var_baseline": bl_var,
                "var_isalsr": isr_var,
                "var_ratio": ratio,
                "levene_p": lev_p,
                "rescue_rate": rescue_rate,
                "n_seeds": len(sub),
            }
        )

    vdf = pd.DataFrame(var_rows)
    print(
        vdf[
            [
                "problem",
                "bottleneck",
                "sig_train",
                "var_ratio",
                "levene_p",
                "rescue_rate",
                "n_seeds",
            ]
        ].to_string(index=False)
    )

    # Spearman: var_ratio vs sig_train
    r, p = stats.pointbiserialr(vdf["sig_train"].astype(float), np.log10(vdf["var_ratio"] + 1))
    print(f"\nPoint-biserial(sig_train, log10(var_ratio)): r={r:.4f}, p={p:.4f}")

    # ── 2. Bottleneck classification ──────────────────────────────────────────

    print("\n" + "─" * 100)
    print("2. BOTTLENECK CLASSIFICATION")
    print("─" * 100)

    print(
        f"\n{'Problem':<18} {'Bottleneck':<22} {'n_consts':>8} {'k':>3} {'depth':>5} "
        f"{'sig_tr':>6} {'sig_both':>8} {'Prediction':>10} {'Match':>5}"
    )
    print("-" * 100)

    correct = 0
    total = 0
    for prob in sorted(PROBLEMS_META.keys()):
        m = PROBLEMS_META[prob]
        # Prediction: structural bottleneck → IsalSR helps (sig_train=True)
        predicted = m["bottleneck"] == "structural"
        actual = m["sig_train"]
        match = predicted == actual
        correct += int(match)
        total += 1

        print(
            f"{prob:<18} {m['bottleneck']:<22} {m['n_nontrivial_constants']:>8} "
            f"{m['gt_k']:>3} {m['gt_depth']:>5} "
            f"{str(m['sig_train']):>6} {str(m['sig_both']):>8} "
            f"{str(predicted):>10} {'✓' if match else '✗':>5}"
        )

    print(f"\nAccuracy: {correct}/{total} = {correct / total:.1%}")

    # ── 3. The Feynman-vs-GP split as a proxy ────────────────────────────────

    print("\n" + "─" * 100)
    print("3. SOURCE SPLIT (Feynman vs GP-hard) as bottleneck proxy")
    print("─" * 100)

    for prob in sorted(PROBLEMS_META.keys()):
        m = PROBLEMS_META[prob]
        print(
            f"  {prob:<18} source={m['source']:<7} bottleneck={m['bottleneck']:<22} "
            f"sig_train={m['sig_train']}"
        )

    # Fisher's exact: source × sig_train
    feynman_sig = sum(
        1 for m in PROBLEMS_META.values() if m["source"] == "feynman" and m["sig_train"]
    )
    feynman_nsig = sum(
        1 for m in PROBLEMS_META.values() if m["source"] == "feynman" and not m["sig_train"]
    )
    gp_sig = sum(1 for m in PROBLEMS_META.values() if m["source"] == "gp" and m["sig_train"])
    gp_nsig = sum(1 for m in PROBLEMS_META.values() if m["source"] == "gp" and not m["sig_train"])

    ct = np.array([[feynman_sig, feynman_nsig], [gp_sig, gp_nsig]])
    odds, p = stats.fisher_exact(ct)
    print(f"\nFisher exact (source × sig_train): OR={odds:.3f}, p={p:.4f}")
    print(f"  Feynman: {feynman_sig}/{feynman_sig + feynman_nsig} sig_train")
    print(f"  GP-hard: {gp_sig}/{gp_sig + gp_nsig} sig_train")

    # Fisher's exact: bottleneck=structural × sig_train
    struct_sig = sum(
        1 for m in PROBLEMS_META.values() if m["bottleneck"] == "structural" and m["sig_train"]
    )
    struct_nsig = sum(
        1 for m in PROBLEMS_META.values() if m["bottleneck"] == "structural" and not m["sig_train"]
    )
    other_sig = sum(
        1 for m in PROBLEMS_META.values() if m["bottleneck"] != "structural" and m["sig_train"]
    )
    other_nsig = sum(
        1 for m in PROBLEMS_META.values() if m["bottleneck"] != "structural" and not m["sig_train"]
    )

    ct2 = np.array([[struct_sig, struct_nsig], [other_sig, other_nsig]])
    odds2, p2 = stats.fisher_exact(ct2)
    print(f"\nFisher exact (structural_bottleneck × sig_train): OR={odds2:.3f}, p={p2:.4f}")
    print(f"  Structural: {struct_sig}/{struct_sig + struct_nsig} sig_train")
    print(f"  Other:      {other_sig}/{other_sig + other_nsig} sig_train")

    # ── 4. n_nontrivial_constants as quantitative predictor ──────────────────

    print("\n" + "─" * 100)
    print("4. NUMBER OF NON-TRIVIAL CONSTANTS as predictor")
    print("─" * 100)

    nconsts = np.array([PROBLEMS_META[p]["n_nontrivial_constants"] for p in sorted(PROBLEMS_META)])
    sig_tr = np.array([float(PROBLEMS_META[p]["sig_train"]) for p in sorted(PROBLEMS_META)])

    r, p = stats.pointbiserialr(sig_tr, nconsts)
    print(f"Point-biserial(sig_train, n_nontrivial_constants): r={r:.4f}, p={p:.4f}")
    print(
        "  sig_train=True problems:  n_constants = ",
        [
            PROBLEMS_META[p]["n_nontrivial_constants"]
            for p in sorted(PROBLEMS_META)
            if PROBLEMS_META[p]["sig_train"]
        ],
    )
    print(
        "  sig_train=False problems: n_constants = ",
        [
            PROBLEMS_META[p]["n_nontrivial_constants"]
            for p in sorted(PROBLEMS_META)
            if not PROBLEMS_META[p]["sig_train"]
        ],
    )

    # ── 5. The "Goldilocks zone" ─────────────────────────────────────────────

    print("\n" + "─" * 100)
    print("5. GOLDILOCKS ZONE: baseline R² vs IsalSR effect")
    print("─" * 100)

    print("\nHypothesis: IsalSR helps most when baseline R² ∈ (0.9, 1.0) — hard enough")
    print("to have variance, easy enough to be 'reachable'.\n")

    for prob in sorted(PROBLEMS_META.keys()):
        sub = df[df["problem"] == prob]
        bl_median = sub["r2_train_baseline"].median()
        bl_min = sub["r2_train_baseline"].min()
        bl_max = sub["r2_train_baseline"].max()
        bl_iqr = sub["r2_train_baseline"].quantile(0.75) - sub["r2_train_baseline"].quantile(0.25)
        sig = PROBLEMS_META[prob]["sig_train"]

        zone = (
            "GOLDILOCKS" if 0.9 < bl_median < 1.0 else ("CEILING" if bl_median >= 1.0 else "FLOOR")
        )
        # Refined: check if there's meaningful variance even at ceiling
        if zone == "CEILING" and bl_iqr > 1e-6:
            zone = "NEAR-CEILING"

        marker = "✓" if sig else "✗"
        print(
            f"  {marker} {prob:<18} R²_median={bl_median:.6f} IQR={bl_iqr:.6f} "
            f"range=[{bl_min:.6f},{bl_max:.6f}] zone={zone}"
        )

    # ── 6. Comprehensive summary table ───────────────────────────────────────

    print("\n" + "─" * 100)
    print("6. COMPREHENSIVE SUMMARY")
    print("─" * 100)

    print(
        f"\n{'Problem':<18} {'Bottleneck':<20} {'Zone':<14} {'VarRatio':>9} "
        f"{'Rescue%':>8} {'Cliff_d':>8} {'sig_tr':>6} {'sig_te':>6} "
        f"{'Reason'}"
    )
    print("-" * 130)

    for prob in sorted(PROBLEMS_META.keys()):
        m = PROBLEMS_META[prob]
        sub = df[df["problem"] == prob]

        bl_train = sub["r2_train_baseline"].values
        isr_train = sub["r2_train_isalsr"].values

        bl_var = np.var(bl_train)
        isr_var = np.var(isr_train)
        var_ratio = bl_var / max(isr_var, 1e-30)

        # Cliff's delta
        n = len(bl_train) * len(isr_train)
        more = np.sum(isr_train[:, None] > bl_train[None, :])
        less = np.sum(isr_train[:, None] < bl_train[None, :])
        cliff_d = (more - less) / max(n, 1)

        median_bl = np.median(bl_train)
        bad_deltas = sub.loc[sub["r2_train_baseline"] < median_bl, "delta_r2_train"].values
        rescue = np.mean(bad_deltas > 0) * 100 if len(bad_deltas) > 0 else 0

        bl_iqr = np.percentile(bl_train, 75) - np.percentile(bl_train, 25)
        zone = (
            "GOLDILOCKS"
            if 0.9 < median_bl < 1.0
            else ("CEILING" if bl_iqr < 1e-6 else "NEAR-CEILING" if median_bl >= 0.999 else "FLOOR")
        )

        print(
            f"  {prob:<18} {m['bottleneck']:<20} {zone:<14} {var_ratio:>9.1f} "
            f"{rescue:>7.0f}% {cliff_d:>+8.3f} "
            f"{'✓' if m['sig_train'] else '✗':>6} {'✓' if m['sig_both'] else '✗':>6} "
            f"{m['reason'][:55]}"
        )


if __name__ == "__main__":
    main()
