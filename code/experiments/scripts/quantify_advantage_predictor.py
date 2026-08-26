"""Quantify which computable variables best predict IsalSR advantage.

Goal: find 1-2 variables computable from a ground-truth expression that
predict sig_train, so we can screen SRBench problems automatically.

Candidate variables tested:
1. n_nontrivial_constants (count of non-small-integer constants)
2. k (number of internal/operator nodes in ground-truth expression)
3. k / (1 + n_nontrivial_constants)
4. Various k thresholds
5. Composite rules
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROBLEMS = {
    "i.15.10": {
        "k": 7,
        "depth": 5,
        "n_vars": 3,
        "n_nontrivial_consts": 0,
        "n_active_vars": 3,
        "sig_train": True,
        "sig_both": True,
        "bottleneck": "structural",
    },
    "i.30.3": {
        "k": 8,
        "depth": 5,
        "n_vars": 2,
        "n_nontrivial_consts": 0,
        "n_active_vars": 2,
        "sig_train": True,
        "sig_both": True,
        "bottleneck": "structural",
    },
    "i.37.4": {
        "k": 7,
        "depth": 4,
        "n_vars": 3,
        "n_nontrivial_consts": 0,
        "n_active_vars": 3,
        "sig_train": True,
        "sig_both": True,
        "bottleneck": "structural",
    },
    "ii.11.27": {
        "k": 8,
        "depth": 5,
        "n_vars": 4,
        "n_nontrivial_consts": 0,
        "n_active_vars": 4,
        "sig_train": False,
        "sig_both": False,
        "bottleneck": "none_trivial",
    },
    "iii.17.37": {
        "k": 8,
        "depth": 5,
        "n_vars": 4,
        "n_nontrivial_consts": 0,
        "n_active_vars": 4,
        "sig_train": True,
        "sig_both": True,
        "bottleneck": "structural",
    },
    "keijzer_6": {
        "k": 2,
        "depth": 2,
        "n_vars": 1,
        "n_nontrivial_consts": 1,
        "n_active_vars": 1,
        "sig_train": False,
        "sig_both": False,
        "bottleneck": "constant",
    },
    "korns_12": {
        "k": 7,
        "depth": 4,
        "n_vars": 5,
        "n_nontrivial_consts": 4,
        "n_active_vars": 2,
        "sig_train": False,
        "sig_both": False,
        "bottleneck": "constant+selection",
    },
    "pagie_1": {
        "k": 7,
        "depth": 4,
        "n_vars": 2,
        "n_nontrivial_consts": 0,
        "n_active_vars": 2,
        "sig_train": True,
        "sig_both": False,
        "bottleneck": "structural",
    },
    "vladislavleva_2": {
        "k": 13,
        "depth": 6,
        "n_vars": 1,
        "n_nontrivial_consts": 0,
        "n_active_vars": 1,
        "sig_train": False,
        "sig_both": False,
        "bottleneck": "structural_depth",
    },
    "vladislavleva_4": {
        "k": 12,
        "depth": 4,
        "n_vars": 5,
        "n_nontrivial_consts": 2,
        "n_active_vars": 5,
        "sig_train": False,
        "sig_both": False,
        "bottleneck": "width+constants",
    },
}


def build_df() -> pd.DataFrame:
    rows = []
    for name, d in PROBLEMS.items():
        row = {"problem": name}
        row.update(d)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_rule(
    df: pd.DataFrame,
    predicted: np.ndarray,
    actual: np.ndarray,
    label: str,
) -> dict:
    """Compute classification metrics for a binary rule."""
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    accuracy = (tp + tn) / len(df)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return {
        "rule": label,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def main() -> None:
    df = build_df()
    actual_train = df["sig_train"].values
    actual_both = df["sig_both"].values

    print("=" * 100)
    print("QUANTIFYING ISALSR ADVANTAGE PREDICTOR")
    print("=" * 100)

    # ── 1. Single-variable analysis ───────────────────────────────────────────

    print("\n" + "─" * 100)
    print("1. SINGLE VARIABLE ANALYSIS")
    print("─" * 100)

    print("\n--- Point-biserial correlations with sig_train ---")
    for feat in ["k", "depth", "n_vars", "n_nontrivial_consts", "n_active_vars"]:
        vals = df[feat].values.astype(float)
        r, p = stats.pointbiserialr(actual_train.astype(float), vals)
        print(f"  {feat:<25} r_pb={r:+.4f}  p={p:.4f}")

    # ── 2. Binary rule: n_nontrivial_consts = 0 ──────────────────────────────

    print("\n" + "─" * 100)
    print("2. SINGLE RULES")
    print("─" * 100)

    rules = []

    # Rule: n_consts = 0
    pred = (df["n_nontrivial_consts"] == 0).values
    rules.append(evaluate_rule(df, pred, actual_train, "n_consts=0"))

    # Rule: k in [5,10]
    pred = ((df["k"] >= 5) & (df["k"] <= 10)).values
    rules.append(evaluate_rule(df, pred, actual_train, "5≤k≤10"))

    # Rule: k in [7,8]
    pred = ((df["k"] >= 7) & (df["k"] <= 8)).values
    rules.append(evaluate_rule(df, pred, actual_train, "7≤k≤8"))

    # Rule: depth in [4,5]
    pred = ((df["depth"] >= 4) & (df["depth"] <= 5)).values
    rules.append(evaluate_rule(df, pred, actual_train, "4≤depth≤5"))

    rdf = pd.DataFrame(rules)
    print(rdf.to_string(index=False))

    # ── 3. Two-variable composite rules ───────────────────────────────────────

    print("\n" + "─" * 100)
    print("3. TWO-VARIABLE COMPOSITE RULES (target: sig_train)")
    print("─" * 100)

    composite_rules = []

    # Best candidate: n_consts=0 AND k∈[5,10]
    for k_low, k_high in [(4, 10), (5, 10), (5, 11), (6, 10), (7, 8), (7, 10)]:
        pred = ((df["n_nontrivial_consts"] == 0) & (df["k"] >= k_low) & (df["k"] <= k_high)).values
        label = f"n_consts=0 AND {k_low}≤k≤{k_high}"
        composite_rules.append(evaluate_rule(df, pred, actual_train, label))

    # n_consts=0 AND depth∈[x,y]
    for d_low, d_high in [(3, 5), (4, 5), (4, 6)]:
        pred = (
            (df["n_nontrivial_consts"] == 0) & (df["depth"] >= d_low) & (df["depth"] <= d_high)
        ).values
        label = f"n_consts=0 AND {d_low}≤depth≤{d_high}"
        composite_rules.append(evaluate_rule(df, pred, actual_train, label))

    # Continuous composite: k / (1 + n_consts)
    df["k_over_consts"] = df["k"] / (1 + df["n_nontrivial_consts"])
    for threshold in [4, 5, 6, 7]:
        pred = (df["k_over_consts"] >= threshold).values
        label = f"k/(1+n_consts) ≥ {threshold}"
        composite_rules.append(evaluate_rule(df, pred, actual_train, label))

    cdf = pd.DataFrame(composite_rules)
    print(cdf.sort_values("F1", ascending=False).to_string(index=False))

    # ── 4. Exhaustive search over (k_range × consts_threshold) ────────────────

    print("\n" + "─" * 100)
    print("4. EXHAUSTIVE SEARCH: best (k_low, k_high, max_consts) rule")
    print("─" * 100)

    best_f1 = 0
    best_rule = None
    best_metrics = None

    for k_low in range(1, 13):
        for k_high in range(k_low, 15):
            for max_consts in range(0, 5):
                pred = (
                    (df["n_nontrivial_consts"] <= max_consts)
                    & (df["k"] >= k_low)
                    & (df["k"] <= k_high)
                ).values
                m = evaluate_rule(df, pred, actual_train, "")
                if m["F1"] > best_f1:
                    best_f1 = m["F1"]
                    best_rule = f"n_consts≤{max_consts} AND {k_low}≤k≤{k_high}"
                    best_metrics = m

    if best_metrics:
        best_metrics["rule"] = best_rule
        print(f"Best rule: {best_rule}")
        print(f"  F1={best_metrics['F1']:.4f}, Acc={best_metrics['accuracy']:.4f}")
        print(
            f"  TP={best_metrics['TP']}, FP={best_metrics['FP']}, "
            f"FN={best_metrics['FN']}, TN={best_metrics['TN']}"
        )

    # Show ALL rules with F1 ≥ 0.9
    print("\nAll rules with F1 ≥ 0.9:")
    for k_low in range(1, 13):
        for k_high in range(k_low, 15):
            for max_consts in range(0, 5):
                pred = (
                    (df["n_nontrivial_consts"] <= max_consts)
                    & (df["k"] >= k_low)
                    & (df["k"] <= k_high)
                ).values
                m = evaluate_rule(df, pred, actual_train, "")
                if m["F1"] >= 0.9:
                    label = f"n_consts≤{max_consts} AND {k_low}≤k≤{k_high}"
                    print(
                        f"  {label:<35} F1={m['F1']:.3f} Acc={m['accuracy']:.2f} "
                        f"TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}"
                    )

    # ── 5. Show the misclassified cases ───────────────────────────────────────

    print("\n" + "─" * 100)
    print("5. ANALYSIS OF FALSE POSITIVES (best rule)")
    print("─" * 100)

    # Use the recommended rule: n_consts=0 AND 5≤k≤10
    pred_recommended = ((df["n_nontrivial_consts"] == 0) & (df["k"] >= 5) & (df["k"] <= 10)).values

    print("\nRecommended rule: n_consts=0 AND 5≤k≤10")
    print("\nClassification detail:")
    for i, row in df.iterrows():
        pred = pred_recommended[i]
        actual = row["sig_train"]
        match = pred == actual
        label = (
            "TP"
            if pred and actual
            else "FP"
            if pred and not actual
            else "FN"
            if not pred and actual
            else "TN"
        )
        print(
            f"  {label} {row['problem']:<18} k={row['k']:>2} n_consts={row['n_nontrivial_consts']} "
            f"→ predicted={pred}, actual_sig_train={actual} "
            f"(bottleneck={row['bottleneck']})"
        )

    # ── 6. Can we distinguish II.11.27 from winners? ──────────────────────────

    print("\n" + "─" * 100)
    print("6. THE II.11.27 PROBLEM: can any a priori variable distinguish it?")
    print("─" * 100)

    winners = df[df["sig_train"] & (df["n_nontrivial_consts"] == 0)]
    ii1127 = df[df["problem"] == "ii.11.27"]

    print("\nWinners (sig_train=True, n_consts=0):")
    print(
        winners[["problem", "k", "depth", "n_vars", "n_active_vars", "bottleneck"]].to_string(
            index=False
        )
    )
    print("\nII.11.27:")
    print(
        ii1127[["problem", "k", "depth", "n_vars", "n_active_vars", "bottleneck"]].to_string(
            index=False
        )
    )

    print("\n  II.11.27 has k=8, depth=5, n_vars=4 — indistinguishable from III.17.37")
    print("  (k=8, depth=5, n_vars=4). The difference is that II.11.27 is trivially")
    print("  solved by Bingo (R²=1.0 for ALL 30 seeds). This cannot be predicted from")
    print("  the ground-truth expression alone — it requires a quick screening run.")
    print("  Recommendation: screen with 5-seed Bingo run, exclude if median R²≥0.9999.")

    # ── 7. Fisher exact on recommended rule ───────────────────────────────────

    print("\n" + "─" * 100)
    print("7. STATISTICAL VALIDATION OF RECOMMENDED RULE")
    print("─" * 100)

    pred_bool = pred_recommended.astype(bool)
    actual_bool = actual_train.astype(bool)

    ct = np.array(
        [
            [np.sum(pred_bool & actual_bool), np.sum(pred_bool & ~actual_bool)],
            [np.sum(~pred_bool & actual_bool), np.sum(~pred_bool & ~actual_bool)],
        ]
    )
    odds, p = stats.fisher_exact(ct)
    print("\nFisher exact (n_consts=0 AND 5≤k≤10 × sig_train):")
    print(f"  OR={odds:.3f}, p={p:.4f}")
    print(f"  Contingency: {ct.tolist()}")

    # ── 8. Final recommendation ───────────────────────────────────────────────

    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    print("""
Two-variable criterion for selecting SRBench problems likely to benefit from IsalSR:

  VARIABLE 1: n_nontrivial_constants = 0
    (all constants in the ground truth are small integers: 0, 1, 2, 3, 4, ...)
    Rationale: LM discovers integer constants trivially, so the difficulty
    is purely structural (topology discovery).

  VARIABLE 2: 5 ≤ k ≤ 10
    (number of internal/operator nodes in the ground-truth expression)
    Rationale:
    - k < 5: too simple, likely trivially solvable by both variants
    - k > 10: search space too vast (k! > 3.6M), dedup effect diluted
    - k = 7-8 is the empirical sweet spot (all 4 sig_both winners)

  COMBINED RULE: n_nontrivial_constants = 0 AND 5 ≤ k ≤ 10
    Performance: F1 = 0.909, Accuracy = 90% (9/10)
    Only false positive: II.11.27 (trivially solved, R²=1.0)
    → Mitigated by 5-seed Bingo screening (exclude if median R² ≥ 0.9999)

  THREE-STEP SCREENING PIPELINE:
    1. Parse ground-truth expression → compute k and n_nontrivial_constants
    2. Filter: n_nontrivial_constants = 0 AND 5 ≤ k ≤ 10
    3. Run 5-seed Bingo screening → exclude if median R² ≥ 0.9999

  With all 3 steps: 10/10 accuracy on our 10-problem benchmark.
""")


if __name__ == "__main__":
    main()
