#!/usr/bin/env python
"""Comprehensive paired comparison analysis: baseline vs IsalSR.

Loads ALL production run_log.json files from model_validation/{bingo,udfs}/,
computes per-problem paired statistics, aggregate statistics by method,
and prints formatted tables with highlights.

Author: Mario Pascual Gonzalez (auto-generated analysis script)
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_ROOT = Path("/media/mpascual/Sandisk2TB/research/isalsr/results/model_validation")
METHODS = ["bingo", "udfs"]
VARIANTS = ["baseline", "isalsr"]
ALPHA = 0.05  # significance level


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_run_logs() -> list[dict[str, Any]]:
    """Walk {bingo,udfs}/<benchmark>/<problem>/<variant>/seed_XX/run_log.json."""
    records: list[dict[str, Any]] = []
    corrupt_count = 0

    for method in METHODS:
        method_dir = RESULTS_ROOT / method
        if not method_dir.exists():
            print(f"WARNING: {method_dir} does not exist, skipping.")
            continue

        for run_log_path in sorted(method_dir.rglob("run_log.json")):
            # Skip anything under a debug/ subdirectory
            if "debug" in run_log_path.parts:
                continue

            try:
                with open(run_log_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                corrupt_count += 1
                continue

            # Parse path: method/benchmark/problem/variant/seed_XX/run_log.json
            parts = run_log_path.relative_to(RESULTS_ROOT).parts
            # parts = (method, benchmark, problem, variant, seed_dir, "run_log.json")
            if len(parts) < 6:
                continue

            p_method = parts[0]
            p_benchmark = parts[1]
            p_problem = parts[2]
            p_variant = parts[3]
            p_seed_str = parts[4]  # e.g. "seed_09"

            try:
                p_seed = int(p_seed_str.replace("seed_", ""))
            except ValueError:
                continue

            rec = {
                "method": p_method,
                "benchmark": p_benchmark,
                "problem": p_problem,
                "variant": p_variant,
                "seed": p_seed,
                "path": str(run_log_path),
            }

            # Extract results
            res = data.get("results", {})
            reg = res.get("regression", {})
            time_d = res.get("time", {})
            ss = res.get("search_space", {})

            rec["r2_test"] = reg.get("r2_test")
            rec["r2_train"] = reg.get("r2_train")
            rec["nrmse_test"] = reg.get("nrmse_test")
            rec["mse_test"] = reg.get("mse_test")
            rec["solution_recovered"] = reg.get("solution_recovered", False)
            rec["jaccard_index"] = reg.get("jaccard_index")
            rec["model_complexity"] = reg.get("model_complexity")

            rec["wall_clock_total_s"] = time_d.get("wall_clock_total_s")
            rec["search_only_time_s"] = time_d.get("wall_clock_search_only_s")
            rec["canonicalization_runtime_s"] = time_d.get("canonicalization_runtime_s", 0.0)
            rec["time_to_r2_099_s"] = time_d.get("time_to_r2_099_s")
            rec["time_to_r2_0999_s"] = time_d.get("time_to_r2_0999_s")

            rec["total_dags_explored"] = ss.get("total_dags_explored")
            rec["unique_canonical_dags"] = ss.get("unique_canonical_dags")
            rec["empirical_reduction_factor"] = ss.get("empirical_reduction_factor")
            rec["redundancy_rate"] = ss.get("redundancy_rate")
            rec["max_internal_nodes_seen"] = ss.get("max_internal_nodes_seen")
            rec["theoretical_reduction_bound"] = ss.get("theoretical_reduction_bound")

            records.append(rec)

    if corrupt_count > 0:
        print(f"WARNING: {corrupt_count} corrupt/unreadable run_log.json files skipped.\n")

    return records


# ---------------------------------------------------------------------------
# Per-problem paired statistics
# ---------------------------------------------------------------------------
def compute_per_problem_stats(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Group by (method, benchmark, problem), compute paired stats.

    Returns dict keyed by (method, benchmark, problem) with statistics.
    """
    # Organize: (method, benchmark, problem, variant) -> {seed: record}
    grouped: dict[tuple[str, str, str, str], dict[int, dict]] = defaultdict(dict)
    for rec in records:
        key = (rec["method"], rec["benchmark"], rec["problem"], rec["variant"])
        grouped[key][rec["seed"]] = rec

    results: dict[tuple[str, str, str], dict[str, Any]] = {}

    # Get all unique (method, benchmark, problem) combos
    problem_keys = set()
    for m, b, p, v in grouped:
        problem_keys.add((m, b, p))

    for method, bench, problem in sorted(problem_keys):
        baseline_data = grouped.get((method, bench, problem, "baseline"), {})
        isalsr_data = grouped.get((method, bench, problem, "isalsr"), {})

        all_seeds = sorted(set(baseline_data.keys()) | set(isalsr_data.keys()))
        common_seeds = sorted(set(baseline_data.keys()) & set(isalsr_data.keys()))

        n_baseline = len(baseline_data)
        n_isalsr = len(isalsr_data)
        n_common = len(common_seeds)

        # Paired R² arrays (common seeds only)
        r2_baseline = []
        r2_isalsr = []
        for s in common_seeds:
            rb = baseline_data[s].get("r2_test")
            ri = isalsr_data[s].get("r2_test")
            if rb is not None and ri is not None and not (np.isnan(rb) or np.isnan(ri)):
                r2_baseline.append(rb)
                r2_isalsr.append(ri)

        r2_baseline = np.array(r2_baseline)
        r2_isalsr = np.array(r2_isalsr)

        # Baseline stats
        all_r2_baseline = [
            baseline_data[s].get("r2_test")
            for s in baseline_data
            if baseline_data[s].get("r2_test") is not None
        ]
        all_r2_baseline = [v for v in all_r2_baseline if not np.isnan(v)]
        mean_r2_bl = float(np.nanmean(all_r2_baseline)) if all_r2_baseline else float("nan")
        std_r2_bl = float(np.nanstd(all_r2_baseline)) if all_r2_baseline else float("nan")

        # IsalSR stats
        all_r2_isalsr = [
            isalsr_data[s].get("r2_test")
            for s in isalsr_data
            if isalsr_data[s].get("r2_test") is not None
        ]
        all_r2_isalsr = [v for v in all_r2_isalsr if not np.isnan(v)]
        mean_r2_is = float(np.nanmean(all_r2_isalsr)) if all_r2_isalsr else float("nan")
        std_r2_is = float(np.nanstd(all_r2_isalsr)) if all_r2_isalsr else float("nan")

        # IsalSR redundancy & rho
        rho_vals = [
            isalsr_data[s].get("empirical_reduction_factor")
            for s in isalsr_data
            if isalsr_data[s].get("empirical_reduction_factor") is not None
        ]
        rho_vals = [v for v in rho_vals if not np.isnan(v)]
        mean_rho = float(np.nanmean(rho_vals)) if rho_vals else float("nan")
        std_rho = float(np.nanstd(rho_vals)) if rho_vals else float("nan")

        red_vals = [
            isalsr_data[s].get("redundancy_rate")
            for s in isalsr_data
            if isalsr_data[s].get("redundancy_rate") is not None
        ]
        red_vals = [v for v in red_vals if not np.isnan(v)]
        mean_red = float(np.nanmean(red_vals)) if red_vals else float("nan")
        std_red = float(np.nanstd(red_vals)) if red_vals else float("nan")

        # Wall clock times
        wc_bl = [
            baseline_data[s].get("wall_clock_total_s")
            for s in baseline_data
            if baseline_data[s].get("wall_clock_total_s") is not None
        ]
        wc_bl = [v for v in wc_bl if not np.isnan(v)]
        mean_wc_bl = float(np.nanmean(wc_bl)) if wc_bl else float("nan")

        wc_is = [
            isalsr_data[s].get("wall_clock_total_s")
            for s in isalsr_data
            if isalsr_data[s].get("wall_clock_total_s") is not None
        ]
        wc_is = [v for v in wc_is if not np.isnan(v)]
        mean_wc_is = float(np.nanmean(wc_is)) if wc_is else float("nan")

        # Search-only time
        so_bl = [
            baseline_data[s].get("search_only_time_s")
            for s in baseline_data
            if baseline_data[s].get("search_only_time_s") is not None
        ]
        so_bl = [v for v in so_bl if not np.isnan(v)]
        mean_so_bl = float(np.nanmean(so_bl)) if so_bl else float("nan")

        so_is = [
            isalsr_data[s].get("search_only_time_s")
            for s in isalsr_data
            if isalsr_data[s].get("search_only_time_s") is not None
        ]
        so_is = [v for v in so_is if not np.isnan(v)]
        mean_so_is = float(np.nanmean(so_is)) if so_is else float("nan")

        # Canonicalization time for IsalSR
        canon_vals = [isalsr_data[s].get("canonicalization_runtime_s", 0.0) for s in isalsr_data]
        canon_vals = [v for v in canon_vals if v is not None and not np.isnan(v)]
        mean_canon = float(np.nanmean(canon_vals)) if canon_vals else float("nan")
        std_canon = float(np.nanstd(canon_vals)) if canon_vals else float("nan")

        # Canon overhead as % of wall clock
        canon_pct = (
            (mean_canon / mean_wc_is * 100.0) if (mean_wc_is and mean_wc_is > 0) else float("nan")
        )

        # Solution recovery rate
        sr_bl = [baseline_data[s].get("solution_recovered", False) for s in baseline_data]
        sr_is = [isalsr_data[s].get("solution_recovered", False) for s in isalsr_data]
        sr_rate_bl = sum(sr_bl) / len(sr_bl) if sr_bl else float("nan")
        sr_rate_is = sum(sr_is) / len(sr_is) if sr_is else float("nan")

        # Paired Wilcoxon signed-rank test on R²
        wilcoxon_stat = float("nan")
        wilcoxon_p = float("nan")
        if len(r2_baseline) >= 5:
            diff = r2_isalsr - r2_baseline
            # Remove zero differences for Wilcoxon
            nonzero_diff = diff[diff != 0]
            if len(nonzero_diff) >= 5:
                try:
                    stat, pval = stats.wilcoxon(nonzero_diff, alternative="two-sided")
                    wilcoxon_stat = float(stat)
                    wilcoxon_p = float(pval)
                except Exception:
                    pass

        # How many seeds IsalSR > baseline, baseline > IsalSR
        n_isalsr_wins = int(np.sum(r2_isalsr > r2_baseline)) if len(r2_baseline) > 0 else 0
        n_baseline_wins = int(np.sum(r2_baseline > r2_isalsr)) if len(r2_baseline) > 0 else 0
        n_ties = int(np.sum(r2_baseline == r2_isalsr)) if len(r2_baseline) > 0 else 0

        # Mean total DAGs explored
        dags_bl = [
            baseline_data[s].get("total_dags_explored")
            for s in baseline_data
            if baseline_data[s].get("total_dags_explored") is not None
        ]
        dags_bl = [v for v in dags_bl if not np.isnan(v)]
        mean_dags_bl = float(np.nanmean(dags_bl)) if dags_bl else float("nan")

        dags_is = [
            isalsr_data[s].get("total_dags_explored")
            for s in isalsr_data
            if isalsr_data[s].get("total_dags_explored") is not None
        ]
        dags_is = [v for v in dags_is if not np.isnan(v)]
        mean_dags_is = float(np.nanmean(dags_is)) if dags_is else float("nan")

        results[(method, bench, problem)] = {
            "n_baseline": n_baseline,
            "n_isalsr": n_isalsr,
            "n_common": n_common,
            "mean_r2_baseline": mean_r2_bl,
            "std_r2_baseline": std_r2_bl,
            "mean_r2_isalsr": mean_r2_is,
            "std_r2_isalsr": std_r2_is,
            "mean_rho": mean_rho,
            "std_rho": std_rho,
            "mean_redundancy": mean_red,
            "std_redundancy": std_red,
            "mean_wc_baseline": mean_wc_bl,
            "mean_wc_isalsr": mean_wc_is,
            "mean_search_baseline": mean_so_bl,
            "mean_search_isalsr": mean_so_is,
            "mean_canon_s": mean_canon,
            "std_canon_s": std_canon,
            "canon_pct_wallclock": canon_pct,
            "sr_rate_baseline": sr_rate_bl,
            "sr_rate_isalsr": sr_rate_is,
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_p": wilcoxon_p,
            "n_isalsr_wins": n_isalsr_wins,
            "n_baseline_wins": n_baseline_wins,
            "n_ties": n_ties,
            "mean_dags_baseline": mean_dags_bl,
            "mean_dags_isalsr": mean_dags_is,
        }

    return results


# ---------------------------------------------------------------------------
# Aggregate statistics by method
# ---------------------------------------------------------------------------
def compute_aggregate_stats(
    records: list[dict[str, Any]],
    per_problem: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate across all problems for each method."""
    agg: dict[str, dict[str, Any]] = {}

    for method in METHODS:
        # Collect IsalSR rho/redundancy across all problems
        rho_all = []
        red_all = []
        canon_all = []
        wc_isalsr_all = []

        for rec in records:
            if rec["method"] == method and rec["variant"] == "isalsr":
                rho = rec.get("empirical_reduction_factor")
                if rho is not None and not np.isnan(rho):
                    rho_all.append(rho)
                red = rec.get("redundancy_rate")
                if red is not None and not np.isnan(red):
                    red_all.append(red)
                canon = rec.get("canonicalization_runtime_s", 0.0)
                if canon is not None and not np.isnan(canon):
                    canon_all.append(canon)
                wc = rec.get("wall_clock_total_s")
                if wc is not None and not np.isnan(wc):
                    wc_isalsr_all.append(wc)

        # Paired Wilcoxon across ALL seeds (all problems) for this method
        # Group baseline and isalsr by (benchmark, problem, seed) for pairing
        baseline_map: dict[tuple[str, str, int], float] = {}
        isalsr_map: dict[tuple[str, str, int], float] = {}
        for rec in records:
            if rec["method"] != method:
                continue
            key = (rec["benchmark"], rec["problem"], rec["seed"])
            r2 = rec.get("r2_test")
            if r2 is None or np.isnan(r2):
                continue
            if rec["variant"] == "baseline":
                baseline_map[key] = r2
            elif rec["variant"] == "isalsr":
                isalsr_map[key] = r2

        common_keys = sorted(set(baseline_map.keys()) & set(isalsr_map.keys()))
        paired_bl = np.array([baseline_map[k] for k in common_keys])
        paired_is = np.array([isalsr_map[k] for k in common_keys])

        wilcoxon_p = float("nan")
        wilcoxon_stat = float("nan")
        if len(paired_bl) >= 5:
            diff = paired_is - paired_bl
            nonzero = diff[diff != 0]
            if len(nonzero) >= 5:
                try:
                    stat, pval = stats.wilcoxon(nonzero, alternative="two-sided")
                    wilcoxon_stat = float(stat)
                    wilcoxon_p = float(pval)
                except Exception:
                    pass

        # Count problem-level wins
        n_problems_isalsr_better = 0
        n_problems_baseline_better = 0
        n_problems_tied = 0
        for (m, b, p), st in per_problem.items():
            if m != method:
                continue
            r2_bl = st["mean_r2_baseline"]
            r2_is = st["mean_r2_isalsr"]
            if np.isnan(r2_bl) or np.isnan(r2_is):
                continue
            if r2_is > r2_bl + 1e-10:
                n_problems_isalsr_better += 1
            elif r2_bl > r2_is + 1e-10:
                n_problems_baseline_better += 1
            else:
                n_problems_tied += 1

        agg[method] = {
            "mean_rho": float(np.nanmean(rho_all)) if rho_all else float("nan"),
            "std_rho": float(np.nanstd(rho_all)) if rho_all else float("nan"),
            "median_rho": float(np.nanmedian(rho_all)) if rho_all else float("nan"),
            "mean_redundancy": float(np.nanmean(red_all)) if red_all else float("nan"),
            "std_redundancy": float(np.nanstd(red_all)) if red_all else float("nan"),
            "median_redundancy": float(np.nanmedian(red_all)) if red_all else float("nan"),
            "mean_canon_s": float(np.nanmean(canon_all)) if canon_all else float("nan"),
            "mean_wc_isalsr_s": float(np.nanmean(wc_isalsr_all)) if wc_isalsr_all else float("nan"),
            "canon_pct_total": (
                float(np.nanmean(canon_all)) / float(np.nanmean(wc_isalsr_all)) * 100.0
                if canon_all and wc_isalsr_all and np.nanmean(wc_isalsr_all) > 0
                else float("nan")
            ),
            "n_paired": len(common_keys),
            "mean_r2_baseline": float(np.nanmean(paired_bl))
            if len(paired_bl) > 0
            else float("nan"),
            "mean_r2_isalsr": float(np.nanmean(paired_is)) if len(paired_is) > 0 else float("nan"),
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_p": wilcoxon_p,
            "n_problems_isalsr_better": n_problems_isalsr_better,
            "n_problems_baseline_better": n_problems_baseline_better,
            "n_problems_tied": n_problems_tied,
            "n_rho_samples": len(rho_all),
            "n_isalsr_seed_wins": int(np.sum(paired_is > paired_bl)) if len(paired_bl) > 0 else 0,
            "n_baseline_seed_wins": int(np.sum(paired_bl > paired_is)) if len(paired_bl) > 0 else 0,
            "n_seed_ties": int(np.sum(paired_bl == paired_is)) if len(paired_bl) > 0 else 0,
        }

    return agg


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_separator(char: str = "=", width: int = 140) -> None:
    print(char * width)


def print_per_problem_table(
    method: str,
    bench: str,
    problem_stats: list[tuple[str, dict[str, Any]]],
) -> None:
    """Print a formatted table for one (method, benchmark) combination."""
    print_separator()
    print(f"  METHOD: {method.upper()}  |  BENCHMARK: {bench.upper()}")
    print_separator()

    # Header
    header = (
        f"{'Problem':<14} "
        f"{'R2_BL':>8} {'R2_IS':>8} {'dR2':>8} "
        f"{'Rho':>6} {'Red%':>6} "
        f"{'WC_BL':>8} {'WC_IS':>8} {'Canon_s':>8} {'Can%':>5} "
        f"{'SR_BL':>5} {'SR_IS':>5} "
        f"{'#BL':>4} {'#IS':>4} "
        f"{'Wilc_p':>8} {'W/L/T':>7}"
    )
    print(header)
    print("-" * len(header))

    for problem, st in problem_stats:
        dr2 = st["mean_r2_isalsr"] - st["mean_r2_baseline"]
        wlt = f"{st['n_isalsr_wins']}/{st['n_baseline_wins']}/{st['n_ties']}"

        # Format values
        r2_bl_s = f"{st['mean_r2_baseline']:.4f}" if not np.isnan(st["mean_r2_baseline"]) else "N/A"
        r2_is_s = f"{st['mean_r2_isalsr']:.4f}" if not np.isnan(st["mean_r2_isalsr"]) else "N/A"
        dr2_s = f"{dr2:+.4f}" if not np.isnan(dr2) else "N/A"
        rho_s = f"{st['mean_rho']:.2f}" if not np.isnan(st["mean_rho"]) else "N/A"
        red_s = (
            f"{st['mean_redundancy'] * 100:.1f}" if not np.isnan(st["mean_redundancy"]) else "N/A"
        )
        wc_bl_s = f"{st['mean_wc_baseline']:.1f}" if not np.isnan(st["mean_wc_baseline"]) else "N/A"
        wc_is_s = f"{st['mean_wc_isalsr']:.1f}" if not np.isnan(st["mean_wc_isalsr"]) else "N/A"
        canon_s = f"{st['mean_canon_s']:.1f}" if not np.isnan(st["mean_canon_s"]) else "N/A"
        can_pct_s = (
            f"{st['canon_pct_wallclock']:.0f}" if not np.isnan(st["canon_pct_wallclock"]) else "N/A"
        )
        sr_bl_s = f"{st['sr_rate_baseline']:.2f}" if not np.isnan(st["sr_rate_baseline"]) else "N/A"
        sr_is_s = f"{st['sr_rate_isalsr']:.2f}" if not np.isnan(st["sr_rate_isalsr"]) else "N/A"
        wilc_s = f"{st['wilcoxon_p']:.4f}" if not np.isnan(st["wilcoxon_p"]) else "N/A"

        # Highlight marker
        marker = ""
        if not np.isnan(st["wilcoxon_p"]) and st["wilcoxon_p"] < ALPHA:
            if dr2 > 0:
                marker = " <<< IsalSR sig. better"
            elif dr2 < 0:
                marker = " >>> Baseline sig. better"

        line = (
            f"{problem:<14} "
            f"{r2_bl_s:>8} {r2_is_s:>8} {dr2_s:>8} "
            f"{rho_s:>6} {red_s:>6} "
            f"{wc_bl_s:>8} {wc_is_s:>8} {canon_s:>8} {can_pct_s:>5} "
            f"{sr_bl_s:>5} {sr_is_s:>5} "
            f"{st['n_baseline']:>4} {st['n_isalsr']:>4} "
            f"{wilc_s:>8} {wlt:>7}"
            f"{marker}"
        )
        print(line)

    print()


def print_summary_table(agg: dict[str, dict[str, Any]]) -> None:
    """Print aggregate summary table."""
    print_separator("=", 100)
    print("  AGGREGATE SUMMARY ACROSS ALL PROBLEMS")
    print_separator("=", 100)

    header = (
        f"{'Method':<8} "
        f"{'MeanRho':>8} {'MedRho':>8} {'MeanRed%':>9} {'MedRed%':>9} "
        f"{'MeanCanon':>10} {'Canon%WC':>9} "
        f"{'R2_BL':>8} {'R2_IS':>8} "
        f"{'Wilc_p':>8} "
        f"{'#Pairs':>7} "
        f"{'SeedW/L/T':>10}"
    )
    print(header)
    print("-" * len(header))

    for method in METHODS:
        st = agg[method]
        wlt = f"{st['n_isalsr_seed_wins']}/{st['n_baseline_seed_wins']}/{st['n_seed_ties']}"
        line = (
            f"{method:<8} "
            f"{st['mean_rho']:.4f}  "
            f"{st['median_rho']:.4f}  "
            f"{st['mean_redundancy'] * 100:.2f}%   "
            f"{st['median_redundancy'] * 100:.2f}%   "
            f"{st['mean_canon_s']:>9.2f}s "
            f"{st['canon_pct_total']:>8.1f}% "
            f"{st['mean_r2_baseline']:.4f}  "
            f"{st['mean_r2_isalsr']:.4f}  "
            f"{'%.2e' % st['wilcoxon_p'] if not np.isnan(st['wilcoxon_p']) else 'N/A':>8} "
            f"{st['n_paired']:>7} "
            f"{wlt:>10}"
        )
        print(line)

    print()

    # Problem-level wins
    print("  PROBLEM-LEVEL WINS (by mean R2_test):")
    print("-" * 60)
    for method in METHODS:
        st = agg[method]
        total_probs = (
            st["n_problems_isalsr_better"]
            + st["n_problems_baseline_better"]
            + st["n_problems_tied"]
        )
        print(
            f"  {method.upper()}: "
            f"IsalSR better on {st['n_problems_isalsr_better']}/{total_probs}, "
            f"Baseline better on {st['n_problems_baseline_better']}/{total_probs}, "
            f"Tied: {st['n_problems_tied']}/{total_probs}"
        )
    print()


def print_highlights(
    per_problem: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    """Print highlighted problems where one variant significantly outperforms."""
    print_separator("=", 100)
    print("  HIGHLIGHTS: SIGNIFICANT DIFFERENCES (Wilcoxon p < 0.05)")
    print_separator("=", 100)

    isalsr_better = []
    baseline_better = []

    for (method, bench, problem), st in sorted(per_problem.items()):
        if np.isnan(st["wilcoxon_p"]):
            continue
        if st["wilcoxon_p"] >= ALPHA:
            continue

        dr2 = st["mean_r2_isalsr"] - st["mean_r2_baseline"]
        entry = (method, bench, problem, dr2, st["wilcoxon_p"], st)

        if dr2 > 0:
            isalsr_better.append(entry)
        elif dr2 < 0:
            baseline_better.append(entry)

    print(f"\n  IsalSR SIGNIFICANTLY BETTER ({len(isalsr_better)} problems):")
    print("-" * 100)
    if isalsr_better:
        for method, bench, problem, dr2, pval, st in sorted(isalsr_better, key=lambda x: -x[3]):
            print(
                f"  {method}/{bench}/{problem:14s} "
                f"dR2={dr2:+.6f}  p={pval:.2e}  "
                f"R2_BL={st['mean_r2_baseline']:.4f}  R2_IS={st['mean_r2_isalsr']:.4f}  "
                f"rho={st['mean_rho']:.2f}  red={st['mean_redundancy'] * 100:.1f}%"
            )
    else:
        print("  (none)")

    print(f"\n  BASELINE SIGNIFICANTLY BETTER ({len(baseline_better)} problems):")
    print("-" * 100)
    if baseline_better:
        for method, bench, problem, dr2, pval, st in sorted(baseline_better, key=lambda x: x[3]):
            print(
                f"  {method}/{bench}/{problem:14s} "
                f"dR2={dr2:+.6f}  p={pval:.2e}  "
                f"R2_BL={st['mean_r2_baseline']:.4f}  R2_IS={st['mean_r2_isalsr']:.4f}  "
                f"rho={st['mean_rho']:.2f}  canon_overhead={st['canon_pct_wallclock']:.0f}%"
            )
    else:
        print("  (none)")
    print()


def print_canonicalization_overhead(
    per_problem: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    """Print canonicalization overhead analysis."""
    print_separator("=", 100)
    print("  CANONICALIZATION OVERHEAD ANALYSIS")
    print_separator("=", 100)

    print(
        f"\n{'Method':<8} {'Benchmark':<10} {'Problem':<14} "
        f"{'Canon_s':>9} {'WC_IS_s':>9} {'Canon%':>7} {'SearchOnly':>11}"
    )
    print("-" * 80)

    for (method, bench, problem), st in sorted(per_problem.items()):
        canon = st["mean_canon_s"]
        wc = st["mean_wc_isalsr"]
        pct = st["canon_pct_wallclock"]
        so = st["mean_search_isalsr"]

        canon_s = f"{canon:.2f}" if not np.isnan(canon) else "N/A"
        wc_s = f"{wc:.2f}" if not np.isnan(wc) else "N/A"
        pct_s = f"{pct:.1f}%" if not np.isnan(pct) else "N/A"
        so_s = f"{so:.2f}" if not np.isnan(so) else "N/A"

        print(f"{method:<8} {bench:<10} {problem:<14} {canon_s:>9} {wc_s:>9} {pct_s:>7} {so_s:>11}")

    print()


def print_data_completeness(
    per_problem: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    """Print data completeness check."""
    print_separator("=", 100)
    print("  DATA COMPLETENESS CHECK")
    print_separator("=", 100)

    issues = []
    for (method, bench, problem), st in sorted(per_problem.items()):
        if st["n_baseline"] != st["n_isalsr"]:
            issues.append(
                f"  MISMATCH: {method}/{bench}/{problem}: "
                f"baseline={st['n_baseline']} seeds, isalsr={st['n_isalsr']} seeds"
            )
        elif st["n_baseline"] < 30:
            issues.append(
                f"  LOW COUNT: {method}/{bench}/{problem}: "
                f"only {st['n_baseline']} seeds (expected 30)"
            )

    if issues:
        for iss in issues:
            print(iss)
    else:
        print("  All (method, benchmark, problem) combinations have matching seed counts.")

    # Total counts
    total_bl = sum(st["n_baseline"] for st in per_problem.values())
    total_is = sum(st["n_isalsr"] for st in per_problem.values())
    total_problems = len(per_problem)
    print(
        f"\n  Total: {total_problems} problem configurations, "
        f"{total_bl} baseline runs, {total_is} IsalSR runs."
    )
    print()


def print_r2_detail_table(
    per_problem: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    """Print detailed R2 table with std for each problem."""
    print_separator("=", 130)
    print("  DETAILED R2 (mean +/- std) AND SEARCH SPACE METRICS")
    print_separator("=", 130)

    header = (
        f"{'Method':<7} {'Bench':<9} {'Problem':<14} "
        f"{'R2_BL (mean+/-std)':>22} {'R2_IS (mean+/-std)':>22} "
        f"{'rho (mean+/-std)':>18} {'red% (mean+/-std)':>20} "
        f"{'DAGs_BL':>10} {'DAGs_IS':>10}"
    )
    print(header)
    print("-" * len(header))

    for (method, bench, problem), st in sorted(per_problem.items()):
        r2_bl = (
            f"{st['mean_r2_baseline']:.4f}+/-{st['std_r2_baseline']:.4f}"
            if not np.isnan(st["mean_r2_baseline"])
            else "N/A"
        )
        r2_is = (
            f"{st['mean_r2_isalsr']:.4f}+/-{st['std_r2_isalsr']:.4f}"
            if not np.isnan(st["mean_r2_isalsr"])
            else "N/A"
        )
        rho = (
            f"{st['mean_rho']:.3f}+/-{st['std_rho']:.3f}" if not np.isnan(st["mean_rho"]) else "N/A"
        )
        red = (
            f"{st['mean_redundancy'] * 100:.2f}+/-{st['std_redundancy'] * 100:.2f}"
            if not np.isnan(st["mean_redundancy"])
            else "N/A"
        )
        dags_bl = (
            f"{st['mean_dags_baseline']:.0f}"
            if not np.isnan(st.get("mean_dags_baseline", float("nan")))
            else "N/A"
        )
        dags_is = (
            f"{st['mean_dags_isalsr']:.0f}"
            if not np.isnan(st.get("mean_dags_isalsr", float("nan")))
            else "N/A"
        )

        print(
            f"{method:<7} {bench:<9} {problem:<14} "
            f"{r2_bl:>22} {r2_is:>22} "
            f"{rho:>18} {red:>20} "
            f"{dags_bl:>10} {dags_is:>10}"
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print_separator("*", 100)
    print("  PAIRED COMPARISON ANALYSIS: BASELINE vs IsalSR")
    print(f"  Results root: {RESULTS_ROOT}")
    print(f"  Methods: {METHODS}")
    print(f"  Significance level: alpha = {ALPHA}")
    print_separator("*", 100)
    print()

    # Step 1: Load data
    print("Loading all run_log.json files...")
    records = load_all_run_logs()
    print(f"  Loaded {len(records)} total records.")

    # Count by method/variant
    from collections import Counter

    counts = Counter((r["method"], r["variant"]) for r in records)
    for (m, v), c in sorted(counts.items()):
        print(f"    {m}/{v}: {c} records")
    print()

    # Step 2: Per-problem stats
    print("Computing per-problem paired statistics...")
    per_problem = compute_per_problem_stats(records)
    print(f"  Computed stats for {len(per_problem)} (method, benchmark, problem) combos.")
    print()

    # Step 3: Data completeness
    print_data_completeness(per_problem)

    # Step 4: Per-problem tables by (method, benchmark)
    # Group problems by (method, benchmark)
    mb_groups: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    for (method, bench, problem), st in per_problem.items():
        mb_groups[(method, bench)].append((problem, st))

    for (method, bench), problem_list in sorted(mb_groups.items()):
        problem_list.sort(key=lambda x: x[0])
        print_per_problem_table(method, bench, problem_list)

    # Step 5: Detailed R2 table
    print_r2_detail_table(per_problem)

    # Step 6: Aggregate stats
    print("Computing aggregate statistics...")
    agg = compute_aggregate_stats(records, per_problem)
    print_summary_table(agg)

    # Step 7: Highlights
    print_highlights(per_problem)

    # Step 8: Canonicalization overhead
    print_canonicalization_overhead(per_problem)

    # Step 9: Additional diagnostics
    print_separator("=", 100)
    print("  ADDITIONAL DIAGNOSTICS")
    print_separator("=", 100)

    # Distribution of rho by method
    for method in METHODS:
        rho_vals = [
            r["empirical_reduction_factor"]
            for r in records
            if r["method"] == method
            and r["variant"] == "isalsr"
            and r.get("empirical_reduction_factor") is not None
            and not np.isnan(r["empirical_reduction_factor"])
        ]

        if rho_vals:
            rho_arr = np.array(rho_vals)
            print(f"\n  {method.upper()} rho distribution (N={len(rho_arr)}):")
            print(f"    Mean:   {np.mean(rho_arr):.4f}")
            print(f"    Median: {np.median(rho_arr):.4f}")
            print(f"    Std:    {np.std(rho_arr):.4f}")
            print(f"    Min:    {np.min(rho_arr):.4f}")
            print(f"    Max:    {np.max(rho_arr):.4f}")
            print(f"    P25:    {np.percentile(rho_arr, 25):.4f}")
            print(f"    P75:    {np.percentile(rho_arr, 75):.4f}")
            print(f"    P90:    {np.percentile(rho_arr, 90):.4f}")
            print(f"    P99:    {np.percentile(rho_arr, 99):.4f}")
            pct_above_1 = np.mean(rho_arr > 1.0) * 100
            print(f"    % with rho > 1.0: {pct_above_1:.1f}%")

    # Canonicalization time distribution by method
    for method in METHODS:
        canon_vals = [
            r["canonicalization_runtime_s"]
            for r in records
            if r["method"] == method
            and r["variant"] == "isalsr"
            and r.get("canonicalization_runtime_s") is not None
            and not np.isnan(r["canonicalization_runtime_s"])
        ]
        if canon_vals:
            c_arr = np.array(canon_vals)
            wc_vals = [
                r["wall_clock_total_s"]
                for r in records
                if r["method"] == method
                and r["variant"] == "isalsr"
                and r.get("wall_clock_total_s") is not None
                and not np.isnan(r["wall_clock_total_s"])
            ]
            w_arr = np.array(wc_vals)
            pct_arr = (
                c_arr[: len(w_arr)] / w_arr[: len(c_arr)] * 100 if len(w_arr) > 0 else np.array([])
            )

            print(f"\n  {method.upper()} canonicalization time (N={len(c_arr)}):")
            print(f"    Mean:   {np.mean(c_arr):.2f}s")
            print(f"    Median: {np.median(c_arr):.2f}s")
            print(f"    Std:    {np.std(c_arr):.2f}s")
            print(f"    Min:    {np.min(c_arr):.4f}s")
            print(f"    Max:    {np.max(c_arr):.2f}s")
            if len(pct_arr) > 0:
                print(f"    Mean % of WC: {np.mean(pct_arr):.1f}%")
                print(f"    Median % of WC: {np.median(pct_arr):.1f}%")
                print(f"    Max % of WC: {np.max(pct_arr):.1f}%")

    # R2 difference distribution by method
    for method in METHODS:
        baseline_map: dict[tuple[str, str, int], float] = {}
        isalsr_map: dict[tuple[str, str, int], float] = {}
        for rec in records:
            if rec["method"] != method:
                continue
            key = (rec["benchmark"], rec["problem"], rec["seed"])
            r2 = rec.get("r2_test")
            if r2 is None or np.isnan(r2):
                continue
            if rec["variant"] == "baseline":
                baseline_map[key] = r2
            elif rec["variant"] == "isalsr":
                isalsr_map[key] = r2

        common = sorted(set(baseline_map.keys()) & set(isalsr_map.keys()))
        if common:
            diffs = np.array([isalsr_map[k] - baseline_map[k] for k in common])
            print(f"\n  {method.upper()} R2 difference (IsalSR - Baseline), N={len(diffs)}:")
            print(f"    Mean:   {np.mean(diffs):+.6f}")
            print(f"    Median: {np.median(diffs):+.6f}")
            print(f"    Std:    {np.std(diffs):.6f}")
            print(f"    Min:    {np.min(diffs):+.6f}")
            print(f"    Max:    {np.max(diffs):+.6f}")
            print(f"    % IsalSR >= Baseline: {np.mean(diffs >= 0) * 100:.1f}%")
            print(f"    % IsalSR > Baseline:  {np.mean(diffs > 0) * 100:.1f}%")
            # Effect size (Cohen's d)
            if np.std(diffs) > 0:
                cohens_d = np.mean(diffs) / np.std(diffs)
                print(f"    Cohen's d: {cohens_d:+.4f}")

    print()
    print_separator("*", 100)
    print("  ANALYSIS COMPLETE")
    print_separator("*", 100)


if __name__ == "__main__":
    main()
