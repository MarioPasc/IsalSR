"""Investigate what problem characteristics predict IsalSR advantage over native DAG.

Builds a problem-level feature matrix from:
1. Structural features of the ground-truth expression (n_vars, n_ops, depth, op types)
2. Empirical runtime features (reduction factor, redundancy rate, baseline difficulty)
3. The IsalSR outcome (significant R² improvement or not)

Then examines correlations, rank-biserial effects, and clustering.
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

# ── Problem-level structural features (hand-coded from ground truth) ──────────

PROBLEM_FEATURES: dict[str, dict] = {
    "i.15.10": {
        "n_vars": 3,
        "n_internal_nodes_gt": 7,  # MUL, DIV, SQRT, SUB, DIV, POW, POW
        "expression_depth": 5,
        "has_sqrt": True,
        "has_trig": False,  # sin/cos
        "has_exp_log": False,
        "has_pow": True,  # v^2, c^2
        "has_sub_div": True,  # SUB and DIV
        "n_transcendental_ops": 1,  # sqrt
        "n_binary_noncomm": 3,  # DIV x2 + SUB
        "n_constants_in_gt": 1,  # the "1" in 1 - v^2/c^2
        "all_vars_active": True,
        "sampling": "uniform",
        "n_train": 1000,
        "has_symmetry": False,  # no symmetric subexpressions
        "gt_expr": "m0*v / sqrt(1 - v^2/c^2)",
    },
    "i.30.3": {
        "n_vars": 2,
        "n_internal_nodes_gt": 8,  # SIN x2, POW x2, DIV, MUL, DIV x2
        "expression_depth": 5,
        "has_sqrt": False,
        "has_trig": True,  # sin x2
        "has_exp_log": False,
        "has_pow": True,  # squaring
        "has_sub_div": True,  # DIV
        "n_transcendental_ops": 2,  # 2 SIN
        "n_binary_noncomm": 3,  # DIV + two embedded DIV (n*theta/2, theta/2)
        "n_constants_in_gt": 1,  # the "2" divisor
        "all_vars_active": True,
        "sampling": "uniform",
        "n_train": 1000,
        "has_symmetry": True,  # parallel sin^2 branches
        "gt_expr": "sin(n*theta/2)^2 / sin(theta/2)^2",
    },
    "i.37.4": {
        "n_vars": 3,
        "n_internal_nodes_gt": 7,  # ADD x2, MUL x2, SQRT, COS
        "expression_depth": 4,
        "has_sqrt": True,
        "has_trig": True,  # cos
        "has_exp_log": False,
        "has_pow": False,
        "has_sub_div": False,  # fully commutative
        "n_transcendental_ops": 2,  # sqrt + cos
        "n_binary_noncomm": 0,
        "n_constants_in_gt": 1,  # the "2" multiplier
        "all_vars_active": True,
        "sampling": "uniform",
        "n_train": 1000,
        "has_symmetry": False,
        "gt_expr": "I1+I2+2*sqrt(I1*I2)*cos(delta)",
    },
    "ii.11.27": {
        "n_vars": 4,
        "n_internal_nodes_gt": 8,  # ADD, MUL x2, EXP x2, NEG, MUL, DIV
        "expression_depth": 5,
        "has_sqrt": False,
        "has_trig": False,
        "has_exp_log": True,  # exp x2
        "has_pow": False,
        "has_sub_div": True,  # DIV + NEG
        "n_transcendental_ops": 2,  # 2 EXP
        "n_binary_noncomm": 2,  # DIV + implicit NEG (SUB-like)
        "n_constants_in_gt": 0,
        "all_vars_active": True,
        "sampling": "uniform",
        "n_train": 1000,
        "has_symmetry": True,  # symmetric exp branches
        "gt_expr": "n0*exp(-mu*B/kT) + n0*exp(mu*B/kT)",
    },
    "iii.17.37": {
        "n_vars": 4,
        "n_internal_nodes_gt": 8,  # DIV, SQRT, ADD, POW, SUB, DIV, POW
        "expression_depth": 5,
        "has_sqrt": True,
        "has_trig": False,
        "has_exp_log": False,
        "has_pow": True,  # squaring
        "has_sub_div": True,  # DIV, SUB
        "n_transcendental_ops": 1,  # sqrt
        "n_binary_noncomm": 3,  # DIV x2 + SUB
        "n_constants_in_gt": 2,  # "4" divisor and "2" exponent
        "all_vars_active": True,
        "sampling": "uniform",
        "n_train": 1000,
        "has_symmetry": False,
        "gt_expr": "f0 / sqrt((omega-omega0)^2 + gamma^2/4)",
    },
    "keijzer_6": {
        "n_vars": 1,
        "n_internal_nodes_gt": 2,  # LOG, ADD
        "expression_depth": 2,
        "has_sqrt": False,
        "has_trig": False,
        "has_exp_log": True,  # log
        "has_pow": False,
        "has_sub_div": False,
        "n_transcendental_ops": 1,  # log
        "n_binary_noncomm": 0,
        "n_constants_in_gt": 1,  # Euler-Mascheroni
        "all_vars_active": True,
        "sampling": "integer_grid",
        "n_train": 50,
        "has_symmetry": False,
        "gt_expr": "H(x) ≈ log(x) + 0.5772",
    },
    "korns_12": {
        "n_vars": 5,
        "n_internal_nodes_gt": 7,  # SUB, MUL x2, COS, SIN, MUL x2
        "expression_depth": 4,
        "has_sqrt": False,
        "has_trig": True,  # cos + sin
        "has_exp_log": False,
        "has_pow": False,
        "has_sub_div": True,  # SUB
        "n_transcendental_ops": 2,  # cos + sin
        "n_binary_noncomm": 1,  # SUB
        "n_constants_in_gt": 3,  # 2.0, 2.1, 9.8, 1.3
        "all_vars_active": False,  # only 2 of 5 vars used
        "sampling": "uniform",
        "n_train": 2000,
        "has_symmetry": False,
        "gt_expr": "2 - 2.1*cos(9.8*x1)*sin(1.3*x5)",
    },
    "pagie_1": {
        "n_vars": 2,
        "n_internal_nodes_gt": 7,  # ADD, INV x2, ADD x2, POW x2
        "expression_depth": 4,
        "has_sqrt": False,
        "has_trig": False,
        "has_exp_log": False,
        "has_pow": True,  # x^(-4)
        "has_sub_div": False,  # can be expressed with INV
        "n_transcendental_ops": 0,
        "n_binary_noncomm": 0,  # if using INV formulation
        "n_constants_in_gt": 2,  # 1, -4
        "all_vars_active": True,
        "sampling": "grid_2d",
        "n_train": 676,
        "has_symmetry": True,  # symmetric x/y branches
        "gt_expr": "1/(1+x^(-4)) + 1/(1+y^(-4))",
    },
    "vladislavleva_2": {
        "n_vars": 1,
        "n_internal_nodes_gt": 13,  # EXP, NEG, POW, MUL x3, COS x2, SIN x2, SUB, POW
        "expression_depth": 6,
        "has_sqrt": False,
        "has_trig": True,  # cos x2, sin x2
        "has_exp_log": True,  # exp
        "has_pow": True,  # x^3, sin^2
        "has_sub_div": True,  # NEG, SUB
        "n_transcendental_ops": 5,  # exp + 2cos + 2sin
        "n_binary_noncomm": 2,  # NEG + SUB
        "n_constants_in_gt": 1,  # the "1" in (cos*sin^2 - 1)
        "all_vars_active": True,
        "sampling": "uniform",
        "n_train": 100,
        "has_symmetry": False,
        "gt_expr": "exp(-x)*x^3*(cos*sin)*(cos*sin^2-1)",
    },
    "vladislavleva_4": {
        "n_vars": 5,
        "n_internal_nodes_gt": 12,  # DIV, ADD, POW x5, SUB x5
        "expression_depth": 4,
        "has_sqrt": False,
        "has_trig": False,
        "has_exp_log": False,
        "has_pow": True,  # squaring x5
        "has_sub_div": True,  # DIV + SUB x5
        "n_transcendental_ops": 0,
        "n_binary_noncomm": 6,  # DIV + 5 SUB
        "n_constants_in_gt": 3,  # 10, 5, 3
        "all_vars_active": True,
        "sampling": "uniform",
        "n_train": 1024,
        "has_symmetry": True,  # symmetric (xi-3)^2 branches
        "gt_expr": "10/(5+sum(xi-3)^2)",
    },
}

# ── Outcome classification (from user's table) ──────────────────────────────

OUTCOME: dict[str, dict] = {
    "i.15.10": {"sig_train": True, "sig_test": True, "fewer_gens": False},
    "i.30.3": {"sig_train": True, "sig_test": True, "fewer_gens": False},
    "i.37.4": {"sig_train": True, "sig_test": True, "fewer_gens": True},
    "ii.11.27": {"sig_train": False, "sig_test": False, "fewer_gens": True},
    "iii.17.37": {"sig_train": True, "sig_test": True, "fewer_gens": False},
    "keijzer_6": {"sig_train": False, "sig_test": True, "fewer_gens": True},
    "korns_12": {"sig_train": False, "sig_test": False, "fewer_gens": False},
    "pagie_1": {"sig_train": True, "sig_test": False, "fewer_gens": False},
    "vladislavleva_2": {"sig_train": False, "sig_test": False, "fewer_gens": False},
    "vladislavleva_4": {"sig_train": False, "sig_test": False, "fewer_gens": False},
}


def build_feature_matrix() -> pd.DataFrame:
    """Merge structural features with empirical results and outcomes."""
    df_paired = pd.read_csv(PAIRED_CSV)

    agg = (
        df_paired.groupby("problem")
        .agg(
            n_seeds=("seed", "count"),
            rf_mean=("reduction_factor", "mean"),
            rf_std=("reduction_factor", "std"),
            redundancy_mean=("redundancy_rate", "mean"),
            r2_train_bl_mean=("r2_train_baseline", "mean"),
            r2_train_bl_median=("r2_train_baseline", "median"),
            r2_test_bl_mean=("r2_test_baseline", "mean"),
            r2_test_bl_median=("r2_test_baseline", "median"),
            r2_train_isr_mean=("r2_train_isalsr", "mean"),
            r2_train_isr_median=("r2_train_isalsr", "median"),
            r2_test_isr_mean=("r2_test_isalsr", "mean"),
            r2_test_isr_median=("r2_test_isalsr", "median"),
            delta_r2_train_mean=("delta_r2_train", "mean"),
            delta_r2_test_mean=("delta_r2_test", "mean"),
            delta_r2_train_median=("delta_r2_train", "median"),
            delta_r2_test_median=("delta_r2_test", "median"),
            max_k_mean=("max_internal_nodes_seen_isalsr", "mean"),
            max_k_median=("max_internal_nodes_seen_isalsr", "median"),
            complexity_bl_mean=("model_complexity_baseline", "mean"),
            complexity_isr_mean=("model_complexity_isalsr", "mean"),
            dags_explored_isr_mean=("total_dags_explored_isalsr", "mean"),
            overhead_mean=("overhead_time_s", "mean"),
            search_time_bl_mean=("wall_clock_search_only_s_baseline", "mean"),
            search_time_isr_mean=("wall_clock_search_only_s_isalsr", "mean"),
            sol_rec_bl_sum=("solution_recovered_baseline", "sum"),
            sol_rec_isr_sum=("solution_recovered_isalsr", "sum"),
        )
        .reset_index()
    )

    feat_rows = []
    for prob in agg["problem"]:
        row = {"problem": prob}
        row.update(PROBLEM_FEATURES[prob])
        row.update(OUTCOME[prob])
        feat_rows.append(row)
    df_feat = pd.DataFrame(feat_rows)

    merged = agg.merge(df_feat, on="problem")

    # Derived features
    merged["baseline_gap_to_1"] = 1.0 - merged["r2_train_bl_median"]
    merged["n_active_vars"] = merged.apply(
        lambda r: r["n_vars"] if r["all_vars_active"] else 2,
        axis=1,  # Korns-12
    )
    merged["complexity_ratio"] = merged["n_internal_nodes_gt"] / merged["n_vars"]
    merged["transcendental_density"] = (
        merged["n_transcendental_ops"] / merged["n_internal_nodes_gt"]
    )
    merged["noncomm_density"] = merged["n_binary_noncomm"] / merged["n_internal_nodes_gt"]

    # Define the "clear winner" label: sig on BOTH train AND test
    merged["sig_both"] = merged["sig_train"] & merged["sig_test"]
    # "any significant" label
    merged["sig_any"] = merged["sig_train"] | merged["sig_test"]

    return merged


def rank_biserial(x_true: np.ndarray, x_false: np.ndarray) -> float:
    """Rank-biserial correlation (effect size for Mann-Whitney U)."""
    n1, n2 = len(x_true), len(x_false)
    if n1 == 0 or n2 == 0:
        return float("nan")
    u_stat = stats.mannwhitneyu(x_true, x_false, alternative="two-sided").statistic
    return 2 * u_stat / (n1 * n2) - 1


def analyze_binary_split(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """For each feature, compare group means (sig=True vs sig=False)."""
    rows = []
    for feat in feature_cols:
        vals = df[feat].values.astype(float)
        labels = df[label_col].values.astype(bool)

        if np.isnan(vals).any():
            continue

        grp_true = vals[labels]
        grp_false = vals[~labels]

        if len(grp_true) < 2 or len(grp_false) < 2:
            rows.append(
                {
                    "feature": feat,
                    "mean_sig": np.mean(grp_true) if len(grp_true) > 0 else np.nan,
                    "mean_nonsig": np.mean(grp_false) if len(grp_false) > 0 else np.nan,
                    "p_mannwhitney": np.nan,
                    "rank_biserial": np.nan,
                    "note": f"small groups ({len(grp_true)} vs {len(grp_false)})",
                }
            )
            continue

        try:
            u_p = stats.mannwhitneyu(grp_true, grp_false, alternative="two-sided").pvalue
        except ValueError:
            u_p = np.nan

        rb = rank_biserial(grp_true, grp_false)

        rows.append(
            {
                "feature": feat,
                "mean_sig": np.mean(grp_true),
                "mean_nonsig": np.mean(grp_false),
                "diff": np.mean(grp_true) - np.mean(grp_false),
                "p_mannwhitney": u_p,
                "rank_biserial": rb,
                "note": "",
            }
        )

    return pd.DataFrame(rows)


def point_biserial_correlations(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute point-biserial correlation between binary label and each feature."""
    rows = []
    labels = df[label_col].values.astype(float)
    for feat in feature_cols:
        vals = df[feat].values.astype(float)
        mask = ~(np.isnan(vals) | np.isnan(labels))
        if mask.sum() < 4:
            continue
        r, p = stats.pointbiserialr(labels[mask], vals[mask])
        rows.append({"feature": feat, "r_pb": r, "p_value": p})
    return pd.DataFrame(rows).sort_values("p_value")


def analyze_convergence_trajectories(df_paired: pd.DataFrame) -> pd.DataFrame:
    """For each problem, compute the fraction of seeds where IsalSR R² > baseline R²."""
    results = []
    for prob, grp in df_paired.groupby("problem"):
        n = len(grp)
        wins_train = (grp["delta_r2_train"] > 0).sum()
        wins_test = (grp["delta_r2_test"] > 0).sum()

        # Use median delta to be robust to outliers
        med_delta_train = grp["delta_r2_train"].median()
        med_delta_test = grp["delta_r2_test"].median()

        # Trimmed mean (10% each tail) for robustness
        from scipy.stats import trim_mean

        tm_delta_train = trim_mean(grp["delta_r2_train"].values, 0.1)
        tm_delta_test = trim_mean(grp["delta_r2_test"].values, 0.1)

        results.append(
            {
                "problem": prob,
                "n_seeds": n,
                "win_rate_train": wins_train / n,
                "win_rate_test": wins_test / n,
                "median_delta_r2_train": med_delta_train,
                "median_delta_r2_test": med_delta_test,
                "trimmed_mean_delta_r2_train": tm_delta_train,
                "trimmed_mean_delta_r2_test": tm_delta_test,
            }
        )
    return pd.DataFrame(results)


def load_convergence_logs() -> pd.DataFrame:
    """Load convergence_log.npz files to get generation-level R² trajectories."""
    rows = []
    for problem in PROBLEM_FEATURES:
        for variant in ["baseline", "isalsr"]:
            for seed_idx in range(1, 31):
                npz_path = (
                    RESULTS_ROOT
                    / problem
                    / variant
                    / f"seed_{seed_idx:02d}"
                    / "convergence_log.npz"
                )
                if not npz_path.exists():
                    continue
                try:
                    data = np.load(npz_path)
                    gens = data["generations"]
                    best_r2 = data["best_r2_train"]
                    timestamps = data["timestamps_s"]

                    # Find generation where R² first exceeds 0.99
                    above_99 = np.where(best_r2 >= 0.99)[0]
                    gen_to_99 = int(gens[above_99[0]]) if len(above_99) > 0 else -1
                    time_to_99 = float(timestamps[above_99[0]]) if len(above_99) > 0 else -1.0

                    above_999 = np.where(best_r2 >= 0.999)[0]
                    gen_to_999 = int(gens[above_999[0]]) if len(above_999) > 0 else -1
                    time_to_999 = float(timestamps[above_999[0]]) if len(above_999) > 0 else -1.0

                    final_r2 = float(best_r2[-1])
                    total_gens = int(gens[-1])

                    rows.append(
                        {
                            "problem": problem,
                            "variant": variant,
                            "seed": seed_idx,
                            "gen_to_r2_099": gen_to_99,
                            "time_to_r2_099": time_to_99,
                            "gen_to_r2_0999": gen_to_999,
                            "time_to_r2_0999": time_to_999,
                            "final_r2_train": final_r2,
                            "total_generations": total_gens,
                        }
                    )
                except Exception as e:
                    log.warning("Failed to load %s: %s", npz_path, e)

    return pd.DataFrame(rows)


def analyze_baseline_difficulty(df: pd.DataFrame) -> None:
    """Examine whether baseline difficulty predicts IsalSR advantage."""
    print("\n" + "=" * 80)
    print("ANALYSIS 1: Baseline Difficulty vs IsalSR Advantage")
    print("=" * 80)

    # The hypothesis: IsalSR helps MORE when baseline is already struggling
    # (i.e., baseline_gap_to_1 is larger → problem is harder → more room for IsalSR)

    print("\nProblem-level summary (sorted by baseline R² train median):")
    cols = [
        "problem",
        "r2_train_bl_median",
        "r2_test_bl_median",
        "baseline_gap_to_1",
        "delta_r2_train_median",
        "delta_r2_test_median",
        "sig_train",
        "sig_test",
        "sig_both",
    ]
    print(df[cols].sort_values("r2_train_bl_median").to_string(index=False))

    # Correlation: baseline_gap_to_1 vs delta_r2
    gap = df["baseline_gap_to_1"].values
    delta_tr = df["delta_r2_train_median"].values
    delta_te = df["delta_r2_test_median"].values

    r_tr, p_tr = stats.spearmanr(gap, delta_tr)
    r_te, p_te = stats.spearmanr(gap, delta_te)
    print(f"\nSpearman(baseline_gap, median_delta_r2_train): rho={r_tr:.4f}, p={p_tr:.4f}")
    print(f"Spearman(baseline_gap, median_delta_r2_test):  rho={r_te:.4f}, p={p_te:.4f}")


def analyze_structural_features(df: pd.DataFrame) -> None:
    """Examine structural features of winning vs losing problems."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Structural Features (sig_both = True vs False)")
    print("=" * 80)

    structural_feats = [
        "n_vars",
        "n_active_vars",
        "n_internal_nodes_gt",
        "expression_depth",
        "n_transcendental_ops",
        "n_binary_noncomm",
        "n_constants_in_gt",
        "complexity_ratio",
        "transcendental_density",
        "noncomm_density",
    ]
    boolean_feats = [
        "has_sqrt",
        "has_trig",
        "has_exp_log",
        "has_pow",
        "has_sub_div",
        "all_vars_active",
        "has_symmetry",
    ]

    print("\n--- Numeric features: sig_both (True) vs not ---")
    result = analyze_binary_split(df, "sig_both", structural_feats)
    print(result.to_string(index=False))

    print("\n--- Boolean features: contingency ---")
    for feat in boolean_feats:
        ct = pd.crosstab(df[feat], df["sig_both"])
        print(f"\n{feat}:")
        print(ct.to_string())
        # Fisher's exact test (2x2)
        if ct.shape == (2, 2):
            odds, p = stats.fisher_exact(ct.values)
            print(f"  Fisher exact: OR={odds:.3f}, p={p:.4f}")

    print("\n--- Point-biserial correlations (sig_both vs numeric) ---")
    all_numeric = structural_feats + [
        "rf_mean",
        "redundancy_mean",
        "r2_train_bl_median",
        "r2_test_bl_median",
        "baseline_gap_to_1",
        "max_k_median",
    ]
    pb = point_biserial_correlations(df, "sig_both", all_numeric)
    print(pb.to_string(index=False))


def analyze_empirical_features(df: pd.DataFrame) -> None:
    """Examine runtime/empirical features."""
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Empirical Runtime Features")
    print("=" * 80)

    empirical_feats = [
        "rf_mean",
        "redundancy_mean",
        "max_k_median",
        "complexity_bl_mean",
        "complexity_isr_mean",
        "dags_explored_isr_mean",
        "search_time_bl_mean",
    ]

    print("\n--- Mann-Whitney: sig_both (True) vs not ---")
    result = analyze_binary_split(df, "sig_both", empirical_feats)
    print(result.to_string(index=False))

    print("\n--- Mann-Whitney: sig_test (True) vs not ---")
    result2 = analyze_binary_split(df, "sig_test", empirical_feats)
    print(result2.to_string(index=False))


def analyze_convergence_speed(df_conv: pd.DataFrame) -> None:
    """Compare convergence speed between baseline and IsalSR."""
    print("\n" + "=" * 80)
    print("ANALYSIS 4: Convergence Speed (from convergence_log.npz)")
    print("=" * 80)

    if df_conv.empty:
        print("No convergence log data available.")
        return

    for problem in sorted(df_conv["problem"].unique()):
        sub = df_conv[df_conv["problem"] == problem]
        bl = sub[sub["variant"] == "baseline"]
        isr = sub[sub["variant"] == "isalsr"]

        if bl.empty or isr.empty:
            continue

        # Compare gen_to_r2_099 (filter out -1 = never reached)
        bl_99 = bl["gen_to_r2_099"].values
        isr_99 = isr["gen_to_r2_099"].values

        bl_reached = bl_99[bl_99 > 0]
        isr_reached = isr_99[isr_99 > 0]

        print(f"\n{problem}:")
        print(
            f"  R²≥0.99 reached: baseline {len(bl_reached)}/{len(bl_99)} seeds, "
            f"isalsr {len(isr_reached)}/{len(isr_99)} seeds"
        )

        if len(bl_reached) > 0 and len(isr_reached) > 0:
            print(
                f"  Gen to R²≥0.99: baseline median={np.median(bl_reached):.0f}, "
                f"isalsr median={np.median(isr_reached):.0f}"
            )
            if len(bl_reached) >= 3 and len(isr_reached) >= 3:
                try:
                    u_stat, u_p = stats.mannwhitneyu(isr_reached, bl_reached, alternative="less")
                    print(f"  Mann-Whitney (isalsr < baseline): U={u_stat:.0f}, p={u_p:.4f}")
                except ValueError:
                    pass

        # Compare gen_to_r2_0999
        bl_999 = bl["gen_to_r2_0999"].values
        isr_999 = isr["gen_to_r2_0999"].values

        bl_reached_999 = bl_999[bl_999 > 0]
        isr_reached_999 = isr_999[isr_999 > 0]

        if len(bl_reached_999) > 0 or len(isr_reached_999) > 0:
            print(
                f"  R²≥0.999 reached: baseline {len(bl_reached_999)}/{len(bl_999)} seeds, "
                f"isalsr {len(isr_reached_999)}/{len(isr_999)} seeds"
            )
            if len(bl_reached_999) > 0 and len(isr_reached_999) > 0:
                print(
                    f"  Gen to R²≥0.999: baseline median={np.median(bl_reached_999):.0f}, "
                    f"isalsr median={np.median(isr_reached_999):.0f}"
                )


def analyze_expression_family(df: pd.DataFrame) -> None:
    """Cluster problems by expression 'family' and check IsalSR success pattern."""
    print("\n" + "=" * 80)
    print("ANALYSIS 5: Expression Families and IsalSR Success")
    print("=" * 80)

    # Define families based on structural similarity
    families = {
        "Feynman physics (nested, multi-var)": [
            "i.15.10",
            "i.30.3",
            "i.37.4",
            "ii.11.27",
            "iii.17.37",
        ],
        "GP classics (algebraic/rational)": ["pagie_1", "vladislavleva_4"],
        "GP classics (deep transcendental)": ["vladislavleva_2", "korns_12"],
        "Approximation/simple": ["keijzer_6"],
    }

    for family, members in families.items():
        sub = df[df["problem"].isin(members)]
        n_sig = sub["sig_both"].sum()
        n_total = len(sub)
        print(f"\n{family} ({n_sig}/{n_total} sig_both):")
        for _, row in sub.iterrows():
            marker = "✓" if row["sig_both"] else "✗"
            print(
                f"  {marker} {row['problem']:<18} gap={row['baseline_gap_to_1']:.6f}  "
                f"delta_r2_train={row['delta_r2_train_median']:+.6f}  "
                f"RF={row['rf_mean']:.3f}  "
                f"k_median={row['max_k_median']:.0f}  "
                f"n_vars={row['n_vars']}  "
                f"depth={row['expression_depth']}"
            )

    # Test: Feynman physics vs GP classics
    feynman = df[df["problem"].isin(families["Feynman physics (nested, multi-var)"])]
    gp = df[~df["problem"].isin(families["Feynman physics (nested, multi-var)"])]

    print(f"\n\nFeynman physics: {feynman['sig_both'].sum()}/{len(feynman)} sig_both")
    print(f"GP classics:     {gp['sig_both'].sum()}/{len(gp)} sig_both")

    if len(feynman) >= 2 and len(gp) >= 2:
        ct = np.array(
            [
                [feynman["sig_both"].sum(), len(feynman) - feynman["sig_both"].sum()],
                [gp["sig_both"].sum(), len(gp) - gp["sig_both"].sum()],
            ]
        )
        odds, p = stats.fisher_exact(ct)
        print(f"Fisher exact (Feynman vs GP): OR={odds:.3f}, p={p:.4f}")


def analyze_sampling_and_data(df: pd.DataFrame) -> None:
    """Check whether data sampling protocol correlates with success."""
    print("\n" + "=" * 80)
    print("ANALYSIS 6: Data Sampling & Training Set Size")
    print("=" * 80)

    print("\n--- Sampling protocol ---")
    ct = pd.crosstab(df["sampling"], df["sig_both"])
    print(ct.to_string())

    print("\n--- Training set size ---")
    for _, row in df.sort_values("n_train").iterrows():
        marker = "✓" if row["sig_both"] else "✗"
        print(f"  {marker} {row['problem']:<18} n_train={row['n_train']}")

    # Spearman correlation: n_train vs sig_both
    r, p = stats.pointbiserialr(df["sig_both"].astype(float), df["n_train"].astype(float))
    print(f"\nPoint-biserial(sig_both, n_train): r={r:.4f}, p={p:.4f}")


def comprehensive_feature_ranking(df: pd.DataFrame) -> None:
    """Rank ALL features by their association with IsalSR success."""
    print("\n" + "=" * 80)
    print("ANALYSIS 7: Comprehensive Feature Ranking (sig_test as target)")
    print("=" * 80)

    numeric_features = [
        "n_vars",
        "n_active_vars",
        "n_internal_nodes_gt",
        "expression_depth",
        "n_transcendental_ops",
        "n_binary_noncomm",
        "n_constants_in_gt",
        "complexity_ratio",
        "transcendental_density",
        "noncomm_density",
        "rf_mean",
        "redundancy_mean",
        "r2_train_bl_median",
        "r2_test_bl_median",
        "baseline_gap_to_1",
        "max_k_median",
        "n_train",
    ]

    for target in ["sig_both", "sig_test", "sig_train"]:
        print(f"\n--- Target: {target} ---")
        pb = point_biserial_correlations(df, target, numeric_features)
        if not pb.empty:
            print(pb.to_string(index=False))


def deep_dive_r2_gap(df_paired: pd.DataFrame, df_features: pd.DataFrame) -> None:
    """Seed-level analysis: correlate per-seed delta_r2 with reduction factor and k."""
    print("\n" + "=" * 80)
    print("ANALYSIS 8: Seed-Level Correlations (delta_r2 vs RF, k, etc.)")
    print("=" * 80)

    # Filter out Pagie-1 outliers (R² < -10) for test
    mask_no_outlier = df_paired["r2_test_isalsr"] > -10
    clean = df_paired[mask_no_outlier].copy()

    print(
        f"\nSeed-level rows: {len(df_paired)} total, {len(clean)} after removing catastrophic outliers"
    )

    # Per-problem Spearman: delta_r2_test vs reduction_factor
    print("\n--- Per-problem: Spearman(delta_r2_test, reduction_factor) ---")
    for prob in sorted(clean["problem"].unique()):
        sub = clean[clean["problem"] == prob]
        if len(sub) < 5:
            continue
        r, p = stats.spearmanr(sub["delta_r2_test"], sub["reduction_factor"])
        print(f"  {prob:<18} rho={r:+.4f}  p={p:.4f}  n={len(sub)}")

    # Global Spearman (across all seeds, all problems)
    r_global, p_global = stats.spearmanr(clean["delta_r2_test"], clean["reduction_factor"])
    print(f"\n  GLOBAL: rho={r_global:+.4f}, p={p_global:.4f}")

    # Per-problem: delta_r2_test vs max_internal_nodes_seen
    print("\n--- Per-problem: Spearman(delta_r2_test, max_k) ---")
    for prob in sorted(clean["problem"].unique()):
        sub = clean[clean["problem"] == prob]
        if len(sub) < 5:
            continue
        r, p = stats.spearmanr(sub["delta_r2_test"], sub["max_internal_nodes_seen_isalsr"])
        print(f"  {prob:<18} rho={r:+.4f}  p={p:.4f}  n={len(sub)}")

    # Per-problem: delta_r2_test vs baseline r2 (does IsalSR help more when baseline is worse?)
    print("\n--- Per-problem: Spearman(delta_r2_test, r2_test_baseline) ---")
    for prob in sorted(clean["problem"].unique()):
        sub = clean[clean["problem"] == prob]
        if len(sub) < 5:
            continue
        r, p = stats.spearmanr(sub["delta_r2_test"], sub["r2_test_baseline"])
        print(f"  {prob:<18} rho={r:+.4f}  p={p:.4f}  n={len(sub)}")


def analyze_r2_distribution_shape(df_paired: pd.DataFrame) -> None:
    """Examine the R² distribution shape — ceiling effects, skewness, etc."""
    print("\n" + "=" * 80)
    print("ANALYSIS 9: R² Distribution Shape (Ceiling Effects)")
    print("=" * 80)

    for prob in sorted(df_paired["problem"].unique()):
        sub = df_paired[df_paired["problem"] == prob]
        bl_train = sub["r2_train_baseline"].values
        isr_train = sub["r2_train_isalsr"].values
        bl_test = sub["r2_test_baseline"].values
        isr_test = sub["r2_test_isalsr"].values

        # Count seeds at ceiling (R² > 0.9999)
        bl_ceiling = (bl_train > 0.9999).sum()
        isr_ceiling = (isr_train > 0.9999).sum()

        # Baseline spread
        bl_iqr = np.percentile(bl_train, 75) - np.percentile(bl_train, 25)
        bl_range = bl_train.max() - bl_train.min()

        sig = OUTCOME[prob]
        marker = "✓" if sig["sig_train"] and sig["sig_test"] else "✗"

        print(f"\n{marker} {prob}:")
        print(
            f"  BL train: median={np.median(bl_train):.6f}, IQR={bl_iqr:.6f}, "
            f"range={bl_range:.6f}, at_ceiling(>0.9999)={bl_ceiling}/{len(bl_train)}"
        )
        print(
            f"  ISR train: median={np.median(isr_train):.6f}, "
            f"at_ceiling(>0.9999)={isr_ceiling}/{len(isr_train)}"
        )
        print(f"  BL test:  median={np.median(bl_test):.6f}")
        print(f"  ISR test: median={np.median(isr_test):.6f}")

        # Is there room for improvement?
        room = 1.0 - np.median(bl_train)
        improvement = np.median(isr_train) - np.median(bl_train)
        if room > 0:
            utilization = improvement / room * 100
            print(f"  Room for improvement: {room:.6f}, utilized: {utilization:.1f}%")


def main() -> None:
    log.info("Building feature matrix...")
    df = build_feature_matrix()

    print("\n" + "=" * 80)
    print("FULL FEATURE MATRIX")
    print("=" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 50)
    print(
        df[
            [
                "problem",
                "n_vars",
                "n_internal_nodes_gt",
                "expression_depth",
                "n_transcendental_ops",
                "n_binary_noncomm",
                "rf_mean",
                "redundancy_mean",
                "baseline_gap_to_1",
                "delta_r2_train_median",
                "delta_r2_test_median",
                "sig_train",
                "sig_test",
                "sig_both",
            ]
        ].to_string(index=False)
    )

    # Run all analyses
    analyze_baseline_difficulty(df)
    analyze_structural_features(df)
    analyze_empirical_features(df)
    analyze_expression_family(df)
    analyze_sampling_and_data(df)
    comprehensive_feature_ranking(df)

    # Seed-level analyses
    df_paired = pd.read_csv(PAIRED_CSV)
    deep_dive_r2_gap(df_paired, df)
    analyze_r2_distribution_shape(df_paired)

    # Convergence speed analysis
    log.info("Loading convergence logs...")
    df_conv = load_convergence_logs()
    if not df_conv.empty:
        analyze_convergence_speed(df_conv)

    print("\n" + "=" * 80)
    print("SYNTHESIS")
    print("=" * 80)
    print("""
Key questions answered:
1. Does baseline difficulty predict IsalSR advantage?
2. Do structural features (n_vars, depth, op types) predict advantage?
3. Do empirical features (RF, redundancy, k) predict advantage?
4. Is there a Feynman-vs-GP-classics divide?
5. Does convergence speed differ?
6. Are ceiling effects masking the benefit?
""")


if __name__ == "__main__":
    main()
