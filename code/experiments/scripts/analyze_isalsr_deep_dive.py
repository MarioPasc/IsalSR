"""Deep dive: WHY does IsalSR help on some problems and not others?

Focuses on:
1. Threshold analysis: at what R² threshold does IsalSR start winning?
2. Failure-rate reduction: does IsalSR reduce left-tail failures?
3. Ground-truth reachability: is the ground truth DAG structure easier to find
   with canonical dedup?
4. The per-seed "who benefits" analysis: do bad baseline seeds benefit most?
5. Ceiling effect decomposition: among ceiling problems, what distinguishes
   significant from non-significant?
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
    "model_validation/wl_subtree_hard/models_hard/bingo/hard"
)
PAIRED_CSV = Path(
    "/home/mpascual/research/code/IsalSR/experiments/scripts/hard_bingo_paired_data.csv"
)


def load_paired() -> pd.DataFrame:
    return pd.read_csv(PAIRED_CSV)


def load_convergence_data() -> dict[str, dict[str, list[dict]]]:
    """Load convergence_log.npz into nested dict: problem -> variant -> [seed_data]."""
    problems = [
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
    result: dict[str, dict[str, list[dict]]] = {}
    for problem in problems:
        result[problem] = {"baseline": [], "isalsr": []}
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
                    result[problem][variant].append(
                        {
                            "seed": seed_idx,
                            "generations": data["generations"],
                            "best_r2_train": data["best_r2_train"],
                            "timestamps_s": data["timestamps_s"],
                        }
                    )
                except Exception:
                    pass
    return result


def threshold_analysis(df: pd.DataFrame) -> None:
    """For each problem, compute the fraction of seeds reaching R² thresholds."""
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS: Fraction of seeds reaching R² thresholds")
    print("=" * 80)

    thresholds = [0.5, 0.9, 0.95, 0.99, 0.999, 0.9999]
    problems = sorted(df["problem"].unique())

    outcome_sig = {
        "i.15.10": True,
        "i.30.3": True,
        "i.37.4": True,
        "ii.11.27": False,
        "iii.17.37": True,
        "keijzer_6": False,
        "korns_12": False,
        "pagie_1": False,
        "vladislavleva_2": False,
        "vladislavleva_4": False,
    }

    print(f"\n{'Problem':<18} {'sig':>3}", end="")
    for t in thresholds:
        print(f"  {'BL@' + str(t):>10} {'ISR@' + str(t):>10} {'Δ':>5}", end="")
    print()
    print("-" * 200)

    for prob in problems:
        sub = df[df["problem"] == prob]
        sig = "✓" if outcome_sig.get(prob, False) else "✗"
        print(f"{prob:<18} {sig:>3}", end="")

        for t in thresholds:
            bl_frac = (sub["r2_train_baseline"] >= t).mean()
            isr_frac = (sub["r2_train_isalsr"] >= t).mean()
            delta = isr_frac - bl_frac
            delta_str = f"{delta:+.2f}" if abs(delta) > 0.001 else "  ="
            print(f"  {bl_frac:>10.2f} {isr_frac:>10.2f} {delta_str:>5}", end="")
        print()

    # Same for test R²
    print(f"\n{'Problem':<18} {'sig':>3}", end="")
    for t in thresholds:
        print(f"  {'BL@' + str(t):>10} {'ISR@' + str(t):>10} {'Δ':>5}", end="")
    print("  (TEST R²)")
    print("-" * 200)

    for prob in problems:
        sub = df[df["problem"] == prob]
        sig = "✓" if outcome_sig.get(prob, False) else "✗"
        print(f"{prob:<18} {sig:>3}", end="")

        for t in thresholds:
            bl_frac = (sub["r2_test_baseline"] >= t).mean()
            isr_frac = (sub["r2_test_isalsr"] >= t).mean()
            delta = isr_frac - bl_frac
            delta_str = f"{delta:+.2f}" if abs(delta) > 0.001 else "  ="
            print(f"  {bl_frac:>10.2f} {isr_frac:>10.2f} {delta_str:>5}", end="")
        print()


def failure_rate_analysis(df: pd.DataFrame) -> None:
    """Compare 'failure rates' at different R² cutoffs."""
    print("\n" + "=" * 80)
    print("FAILURE RATE ANALYSIS: Seeds below key thresholds")
    print("=" * 80)

    cutoffs = [0.9, 0.95, 0.99]

    outcome_sig = {
        "i.15.10": True,
        "i.30.3": True,
        "i.37.4": True,
        "ii.11.27": False,
        "iii.17.37": True,
        "keijzer_6": False,
        "korns_12": False,
        "pagie_1": False,
        "vladislavleva_2": False,
        "vladislavleva_4": False,
    }

    for cutoff in cutoffs:
        print(f"\n--- R² train < {cutoff} (failure rate) ---")
        rows = []
        for prob in sorted(df["problem"].unique()):
            sub = df[df["problem"] == prob]
            bl_fail = (sub["r2_train_baseline"] < cutoff).sum()
            isr_fail = (sub["r2_train_isalsr"] < cutoff).sum()
            n = len(sub)
            rows.append(
                {
                    "problem": prob,
                    "sig": outcome_sig.get(prob, False),
                    "bl_failures": bl_fail,
                    "isr_failures": isr_fail,
                    "n": n,
                    "bl_fail_rate": bl_fail / n,
                    "isr_fail_rate": isr_fail / n,
                    "fail_reduction": (bl_fail - isr_fail) / max(bl_fail, 1),
                }
            )
        rdf = pd.DataFrame(rows)
        print(rdf.to_string(index=False))


def per_seed_benefit_analysis(df: pd.DataFrame) -> None:
    """For each problem, analyze WHICH seeds benefit from IsalSR."""
    print("\n" + "=" * 80)
    print("PER-SEED BENEFIT: Does IsalSR help the 'bad' seeds more?")
    print("=" * 80)

    outcome_sig = {
        "i.15.10": True,
        "i.30.3": True,
        "i.37.4": True,
        "ii.11.27": False,
        "iii.17.37": True,
        "keijzer_6": False,
        "korns_12": False,
        "pagie_1": False,
        "vladislavleva_2": False,
        "vladislavleva_4": False,
    }

    for prob in sorted(df["problem"].unique()):
        sub = df[df["problem"] == prob].copy()
        n = len(sub)
        sig_marker = "✓" if outcome_sig.get(prob, False) else "✗"

        # Split seeds into "good baseline" (top 50%) and "bad baseline" (bottom 50%)
        median_bl = sub["r2_train_baseline"].median()
        sub["baseline_quality"] = np.where(sub["r2_train_baseline"] >= median_bl, "good", "bad")

        good = sub[sub["baseline_quality"] == "good"]
        bad = sub[sub["baseline_quality"] == "bad"]

        delta_good = good["delta_r2_train"].values
        delta_bad = bad["delta_r2_train"].values

        # IsalSR benefit for good vs bad baseline seeds
        print(f"\n{sig_marker} {prob} (n={n}):")
        print(
            f"  Good baseline seeds (R²≥{median_bl:.6f}): "
            f"mean delta_r2={delta_good.mean():+.6f}, "
            f"wins={np.sum(delta_good > 0)}/{len(delta_good)}"
        )
        print(
            f"  Bad baseline seeds  (R²<{median_bl:.6f}):  "
            f"mean delta_r2={delta_bad.mean():+.6f}, "
            f"wins={np.sum(delta_bad > 0)}/{len(delta_bad)}"
        )

        # Is the benefit larger for bad seeds? (sign test)
        if len(delta_good) > 0 and len(delta_bad) > 0:
            try:
                u_stat, u_p = stats.mannwhitneyu(delta_bad, delta_good, alternative="greater")
                print(f"  MWU(bad_delta > good_delta): p={u_p:.4f}")
            except ValueError:
                pass


def ceiling_decomposition(df: pd.DataFrame) -> None:
    """Among problems at ceiling (R²≈1), what distinguishes significant from not?"""
    print("\n" + "=" * 80)
    print("CEILING DECOMPOSITION: Among near-perfect problems")
    print("=" * 80)

    ceiling_problems = ["i.15.10", "i.30.3", "ii.11.27", "keijzer_6"]
    outcome_sig = {"i.15.10": True, "i.30.3": True, "ii.11.27": False, "keijzer_6": False}

    for prob in ceiling_problems:
        sub = df[df["problem"] == prob]
        sig = "✓" if outcome_sig[prob] else "✗"

        bl_train = sub["r2_train_baseline"].values
        isr_train = sub["r2_train_isalsr"].values

        # Transform to log(1 - R²) to spread the ceiling
        bl_log = np.log10(1 - bl_train + 1e-16)
        isr_log = np.log10(1 - isr_train + 1e-16)

        delta_log = isr_log - bl_log  # negative = isalsr closer to 1

        # Wilcoxon on log-transformed gap
        try:
            stat, p = stats.wilcoxon(bl_log, isr_log, alternative="two-sided")
        except ValueError:
            p = np.nan

        print(f"\n{sig} {prob}:")
        print(
            f"  log10(1-R²) baseline: median={np.median(bl_log):.2f}, "
            f"IQR=[{np.percentile(bl_log, 25):.2f}, {np.percentile(bl_log, 75):.2f}]"
        )
        print(
            f"  log10(1-R²) isalsr:   median={np.median(isr_log):.2f}, "
            f"IQR=[{np.percentile(isr_log, 25):.2f}, {np.percentile(isr_log, 75):.2f}]"
        )
        print(f"  Wilcoxon on log(1-R²): p={p:.6f}")
        print(
            f"  Seeds where IsalSR is closer to 1: {np.sum(isr_train > bl_train)}/{len(bl_train)}"
        )

        # Check if all seeds are identical (truly at ceiling)
        n_identical = np.sum(bl_train == isr_train)
        print(f"  Seeds with identical R²: {n_identical}/{len(bl_train)}")

        # Look at NRMSE instead (more sensitive near ceiling)
        bl_nrmse = sub["nrmse_train_baseline"].values
        isr_nrmse = sub["nrmse_train_isalsr"].values

        print(f"  NRMSE baseline: median={np.median(bl_nrmse):.6e}")
        print(f"  NRMSE isalsr:   median={np.median(isr_nrmse):.6e}")

        try:
            stat_n, p_n = stats.wilcoxon(bl_nrmse, isr_nrmse, alternative="two-sided")
            print(f"  Wilcoxon on NRMSE: p={p_n:.6f}")
        except ValueError:
            pass


def convergence_trajectory_comparison(conv_data: dict) -> None:
    """Compare R² trajectories at matched time points."""
    print("\n" + "=" * 80)
    print("CONVERGENCE TRAJECTORY: R² at matched generation checkpoints")
    print("=" * 80)

    outcome_sig = {
        "i.15.10": True,
        "i.30.3": True,
        "i.37.4": True,
        "ii.11.27": False,
        "iii.17.37": True,
        "keijzer_6": False,
        "korns_12": False,
        "pagie_1": False,
        "vladislavleva_2": False,
        "vladislavleva_4": False,
    }

    checkpoints = [50, 100, 500, 1000, 2000, 5000, 10000]

    for prob in sorted(conv_data.keys()):
        bl_seeds = conv_data[prob]["baseline"]
        isr_seeds = conv_data[prob]["isalsr"]

        if not bl_seeds or not isr_seeds:
            continue

        sig = "✓" if outcome_sig.get(prob, False) else "✗"
        print(f"\n{sig} {prob} (baseline={len(bl_seeds)} seeds, isalsr={len(isr_seeds)} seeds):")

        for cp in checkpoints:
            bl_r2_at_cp = []
            for s in bl_seeds:
                idx = np.searchsorted(s["generations"], cp)
                if idx < len(s["best_r2_train"]):
                    bl_r2_at_cp.append(s["best_r2_train"][min(idx, len(s["best_r2_train"]) - 1)])

            isr_r2_at_cp = []
            for s in isr_seeds:
                idx = np.searchsorted(s["generations"], cp)
                if idx < len(s["best_r2_train"]):
                    isr_r2_at_cp.append(s["best_r2_train"][min(idx, len(s["best_r2_train"]) - 1)])

            if len(bl_r2_at_cp) < 3 or len(isr_r2_at_cp) < 3:
                continue

            bl_arr = np.array(bl_r2_at_cp)
            isr_arr = np.array(isr_r2_at_cp)

            try:
                _, p = stats.mannwhitneyu(isr_arr, bl_arr, alternative="greater")
            except ValueError:
                p = np.nan

            diff = np.median(isr_arr) - np.median(bl_arr)
            marker = "*" if p < 0.05 else " "
            print(
                f"  gen={cp:>6d}: BL={np.median(bl_arr):.6f} ISR={np.median(isr_arr):.6f} "
                f"Δ={diff:+.6f} p={p:.4f}{marker}"
            )


def analyze_best_expression_structure(df: pd.DataFrame) -> None:
    """Examine the structure of the best expressions found."""
    print("\n" + "=" * 80)
    print("BEST EXPRESSION STRUCTURE: Complexity and form")
    print("=" * 80)

    outcome_sig = {
        "i.15.10": True,
        "i.30.3": True,
        "i.37.4": True,
        "ii.11.27": False,
        "iii.17.37": True,
        "keijzer_6": False,
        "korns_12": False,
        "pagie_1": False,
        "vladislavleva_2": False,
        "vladislavleva_4": False,
    }

    for prob in sorted(df["problem"].unique()):
        sub = df[df["problem"] == prob]
        sig = "✓" if outcome_sig.get(prob, False) else "✗"

        bl_c = sub["model_complexity_baseline"].values
        isr_c = sub["model_complexity_isalsr"].values

        try:
            _, p = stats.wilcoxon(bl_c, isr_c, alternative="two-sided")
        except ValueError:
            p = np.nan

        print(f"\n{sig} {prob}:")
        print(
            f"  Complexity baseline: median={np.median(bl_c):.0f}, "
            f"mean={np.mean(bl_c):.1f}, IQR=[{np.percentile(bl_c, 25):.0f}, {np.percentile(bl_c, 75):.0f}]"
        )
        print(
            f"  Complexity isalsr:   median={np.median(isr_c):.0f}, "
            f"mean={np.mean(isr_c):.1f}, IQR=[{np.percentile(isr_c, 25):.0f}, {np.percentile(isr_c, 75):.0f}]"
        )
        print(f"  Wilcoxon: p={p:.4f}")

        # Does simpler = better here?
        r, p_corr = stats.spearmanr(sub["model_complexity_isalsr"], sub["r2_train_isalsr"])
        print(f"  Spearman(complexity_isr, R²_train_isr): rho={r:.4f}, p={p_corr:.4f}")


def analyze_population_diversity_proxy(conv_data: dict) -> None:
    """Use convergence speed as a proxy for population diversity.

    The hypothesis: IsalSR forces unique DAGs → higher structural diversity →
    better exploration → faster escape from local optima.

    If true, IsalSR should show:
    - More consistent convergence (lower variance in gen_to_threshold)
    - Better worst-case performance (higher min R²)
    """
    print("\n" + "=" * 80)
    print("DIVERSITY PROXY: Convergence consistency (variance reduction)")
    print("=" * 80)

    outcome_sig = {
        "i.15.10": True,
        "i.30.3": True,
        "i.37.4": True,
        "ii.11.27": False,
        "iii.17.37": True,
        "keijzer_6": False,
        "korns_12": False,
        "pagie_1": False,
        "vladislavleva_2": False,
        "vladislavleva_4": False,
    }

    for prob in sorted(conv_data.keys()):
        bl_seeds = conv_data[prob]["baseline"]
        isr_seeds = conv_data[prob]["isalsr"]

        if not bl_seeds or not isr_seeds:
            continue

        sig = "✓" if outcome_sig.get(prob, False) else "✗"

        # Final R²
        bl_final = np.array([s["best_r2_train"][-1] for s in bl_seeds])
        isr_final = np.array([s["best_r2_train"][-1] for s in isr_seeds])

        # Variance of final R²
        bl_var = np.var(bl_final)
        isr_var = np.var(isr_final)

        # Min R² (worst case)
        bl_min = np.min(bl_final)
        isr_min = np.min(isr_final)

        # Coefficient of variation (std/mean)
        bl_cv = np.std(bl_final) / max(np.mean(bl_final), 1e-10)
        isr_cv = np.std(isr_final) / max(np.mean(isr_final), 1e-10)

        # Levene's test for equal variances
        try:
            lev_stat, lev_p = stats.levene(bl_final, isr_final)
        except Exception:
            lev_p = np.nan

        print(f"\n{sig} {prob}:")
        print(
            f"  Final R² var:  BL={bl_var:.2e}, ISR={isr_var:.2e}, "
            f"ratio={bl_var / max(isr_var, 1e-20):.2f}"
        )
        print(f"  Final R² CV:   BL={bl_cv:.6f}, ISR={isr_cv:.6f}")
        print(f"  Worst seed:    BL={bl_min:.6f}, ISR={isr_min:.6f}")
        print(f"  Levene's test: p={lev_p:.4f}")


def load_best_expressions() -> pd.DataFrame:
    """Load symbolic expressions from run_log.json for structural analysis."""
    problems = [
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
    rows = []
    for problem in problems:
        for variant in ["baseline", "isalsr"]:
            for seed_idx in range(1, 31):
                path = RESULTS_ROOT / problem / variant / f"seed_{seed_idx:02d}" / "run_log.json"
                if not path.exists():
                    continue
                try:
                    with path.open() as f:
                        data = json.load(f)
                    best = data.get("best_expression", {})
                    rows.append(
                        {
                            "problem": problem,
                            "variant": variant,
                            "seed": seed_idx,
                            "n_nodes": best.get("n_nodes", np.nan),
                            "n_edges": best.get("n_edges", np.nan),
                            "symbolic_form": best.get("symbolic_form", ""),
                        }
                    )
                except Exception:
                    pass
    return pd.DataFrame(rows)


def analyze_solution_structure(expr_df: pd.DataFrame) -> None:
    """Compare structural properties of found solutions."""
    print("\n" + "=" * 80)
    print("SOLUTION STRUCTURE: n_nodes and n_edges of best expression")
    print("=" * 80)

    outcome_sig = {
        "i.15.10": True,
        "i.30.3": True,
        "i.37.4": True,
        "ii.11.27": False,
        "iii.17.37": True,
        "keijzer_6": False,
        "korns_12": False,
        "pagie_1": False,
        "vladislavleva_2": False,
        "vladislavleva_4": False,
    }

    for prob in sorted(expr_df["problem"].unique()):
        sub = expr_df[expr_df["problem"] == prob]
        bl = sub[sub["variant"] == "baseline"]
        isr = sub[sub["variant"] == "isalsr"]

        sig = "✓" if outcome_sig.get(prob, False) else "✗"

        bl_nodes = bl["n_nodes"].dropna().values
        isr_nodes = isr["n_nodes"].dropna().values
        bl_edges = bl["n_edges"].dropna().values
        isr_edges = isr["n_edges"].dropna().values

        print(f"\n{sig} {prob}:")
        if len(bl_nodes) > 0 and len(isr_nodes) > 0:
            print(
                f"  Nodes: BL median={np.median(bl_nodes):.0f} "
                f"(IQR [{np.percentile(bl_nodes, 25):.0f},{np.percentile(bl_nodes, 75):.0f}]), "
                f"ISR median={np.median(isr_nodes):.0f} "
                f"(IQR [{np.percentile(isr_nodes, 25):.0f},{np.percentile(isr_nodes, 75):.0f}])"
            )
            print(
                f"  Edges: BL median={np.median(bl_edges):.0f} "
                f"(IQR [{np.percentile(bl_edges, 25):.0f},{np.percentile(bl_edges, 75):.0f}]), "
                f"ISR median={np.median(isr_edges):.0f} "
                f"(IQR [{np.percentile(isr_edges, 25):.0f},{np.percentile(isr_edges, 75):.0f}])"
            )

            # Edge density = edges/nodes
            bl_density = bl_edges / np.maximum(bl_nodes, 1)
            isr_density = isr_edges / np.maximum(isr_nodes, 1)
            print(
                f"  Edge density: BL={np.median(bl_density):.3f}, ISR={np.median(isr_density):.3f}"
            )


def meta_analysis_effect_sizes(df: pd.DataFrame) -> None:
    """Compute standardized effect sizes (Cliff's delta) for each problem."""
    print("\n" + "=" * 80)
    print("META-ANALYSIS: Cliff's delta (non-parametric effect size)")
    print("=" * 80)

    outcome_sig = {
        "i.15.10": True,
        "i.30.3": True,
        "i.37.4": True,
        "ii.11.27": False,
        "iii.17.37": True,
        "keijzer_6": False,
        "korns_12": False,
        "pagie_1": False,
        "vladislavleva_2": False,
        "vladislavleva_4": False,
    }

    def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
        """Cliff's delta: P(X > Y) - P(X < Y)."""
        n = len(x) * len(y)
        more = np.sum(x[:, None] > y[None, :])
        less = np.sum(x[:, None] < y[None, :])
        return (more - less) / n

    print(
        f"\n{'Problem':<18} {'sig':>3} {'Cliff_d_train':>14} {'Cliff_d_test':>14} "
        f"{'interpret_train':>16} {'interpret_test':>16}"
    )
    print("-" * 90)

    def interpret_cliff(d: float) -> str:
        ad = abs(d)
        if ad < 0.147:
            return "negligible"
        elif ad < 0.33:
            return "small"
        elif ad < 0.474:
            return "medium"
        else:
            return "large"

    for prob in sorted(df["problem"].unique()):
        sub = df[df["problem"] == prob]
        sig = "✓" if outcome_sig.get(prob, False) else "✗"

        bl_train = sub["r2_train_baseline"].values
        isr_train = sub["r2_train_isalsr"].values
        bl_test = sub["r2_test_baseline"].values
        isr_test = sub["r2_test_isalsr"].values

        cd_train = cliffs_delta(isr_train, bl_train)
        cd_test = cliffs_delta(isr_test, bl_test)

        int_train = interpret_cliff(cd_train)
        int_test = interpret_cliff(cd_test)

        print(
            f"{prob:<18} {sig:>3} {cd_train:>+14.4f} {cd_test:>+14.4f} "
            f"{int_train:>16} {int_test:>16}"
        )


def analyze_interaction_baseline_x_structure(df: pd.DataFrame) -> None:
    """Test the interaction: does IsalSR benefit MORE when both
    (a) the problem is 'reachable' (R²>0.9 achievable) AND
    (b) the ground truth has specific structural properties?
    """
    print("\n" + "=" * 80)
    print("INTERACTION ANALYSIS: Reachability × Structure")
    print("=" * 80)

    # Feature from hand-coded data
    gt_features = {
        "i.15.10": {"source": "feynman", "has_sqrt": True, "gt_k": 7, "gt_depth": 5},
        "i.30.3": {"source": "feynman", "has_sqrt": False, "gt_k": 8, "gt_depth": 5},
        "i.37.4": {"source": "feynman", "has_sqrt": True, "gt_k": 7, "gt_depth": 4},
        "ii.11.27": {"source": "feynman", "has_sqrt": False, "gt_k": 8, "gt_depth": 5},
        "iii.17.37": {"source": "feynman", "has_sqrt": True, "gt_k": 8, "gt_depth": 5},
        "keijzer_6": {"source": "gp", "has_sqrt": False, "gt_k": 2, "gt_depth": 2},
        "korns_12": {"source": "gp", "has_sqrt": False, "gt_k": 7, "gt_depth": 4},
        "pagie_1": {"source": "gp", "has_sqrt": False, "gt_k": 7, "gt_depth": 4},
        "vladislavleva_2": {"source": "gp", "has_sqrt": False, "gt_k": 13, "gt_depth": 6},
        "vladislavleva_4": {"source": "gp", "has_sqrt": False, "gt_k": 12, "gt_depth": 4},
    }

    outcome_sig_both = {
        "i.15.10": True,
        "i.30.3": True,
        "i.37.4": True,
        "ii.11.27": False,
        "iii.17.37": True,
        "keijzer_6": False,
        "korns_12": False,
        "pagie_1": False,
        "vladislavleva_2": False,
        "vladislavleva_4": False,
    }

    # Define reachability: median baseline R² > 0.9
    for prob in sorted(df["problem"].unique()):
        sub = df[df["problem"] == prob]
        bl_med = sub["r2_train_baseline"].median()
        feat = gt_features[prob]
        sig = outcome_sig_both[prob]

        reachable = bl_med > 0.9
        feynman = feat["source"] == "feynman"
        has_sqrt = feat["has_sqrt"]
        gt_k = feat["gt_k"]

        print(
            f"  {prob:<18} reachable={reachable!s:<6} feynman={feynman!s:<6} "
            f"sqrt={has_sqrt!s:<6} gt_k={gt_k:>2} sig={sig}"
        )

    print("\n--- Contingency: reachable × feynman → sig_both ---")
    # Build 2x2x2 table
    for reach_val in [True, False]:
        for feyn_val in [True, False]:
            probs = [
                p
                for p in outcome_sig_both
                if (df[df["problem"] == p]["r2_train_baseline"].median() > 0.9) == reach_val
                and (gt_features[p]["source"] == "feynman") == feyn_val
            ]
            if probs:
                n_sig = sum(1 for p in probs if outcome_sig_both[p])
                print(
                    f"  reachable={reach_val}, feynman={feyn_val}: "
                    f"{n_sig}/{len(probs)} sig_both → {[p for p in probs]}"
                )

    print("\n--- Contingency: reachable × has_sqrt → sig_both ---")
    for reach_val in [True, False]:
        for sqrt_val in [True, False]:
            probs = [
                p
                for p in outcome_sig_both
                if (df[df["problem"] == p]["r2_train_baseline"].median() > 0.9) == reach_val
                and gt_features[p]["has_sqrt"] == sqrt_val
            ]
            if probs:
                n_sig = sum(1 for p in probs if outcome_sig_both[p])
                print(
                    f"  reachable={reach_val}, has_sqrt={sqrt_val}: "
                    f"{n_sig}/{len(probs)} sig_both → {[p for p in probs]}"
                )


def main() -> None:
    log.info("Loading data...")
    df = load_paired()

    threshold_analysis(df)
    failure_rate_analysis(df)
    per_seed_benefit_analysis(df)
    ceiling_decomposition(df)
    meta_analysis_effect_sizes(df)
    analyze_interaction_baseline_x_structure(df)

    log.info("Loading convergence data...")
    conv_data = load_convergence_data()
    convergence_trajectory_comparison(conv_data)
    analyze_population_diversity_proxy(conv_data)

    log.info("Loading expression data...")
    expr_df = load_best_expressions()
    if not expr_df.empty:
        analyze_solution_structure(expr_df)

    analyze_best_expression_structure(df)


if __name__ == "__main__":
    main()
