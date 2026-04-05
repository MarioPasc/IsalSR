"""Generate LaTeX tables for IsalSR model validation results.

Produces:
  Table 1: Unified three-axis summary + overhead (one row per method x benchmark)
  Table 2: Per-problem R² comparison (full 22-row breakdown, one per method)
  Table K: Bingo k-range overhead breakdown (for Discussion section)
  Table S: Supplementary per-problem statistics (one per method, TPAMI-ready)

Usage:
    python -m experiments.figures.models.generate_tables \
        --results-dir /path/to/results \
        --output-dir /path/to/figures
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.models.io_utils import load_all_run_logs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ======================================================================
# Data extraction helpers
# ======================================================================


def _load_paired_metrics(
    results_dir: Path,
    method: str,
    benchmark: str,
) -> dict[str, dict[str, list[float]]]:
    """Load per-seed metrics for both variants, keyed by problem.

    Returns:
        {problem: {
            "bl_r2_test": [...], "is_r2_test": [...],
            "bl_r2_train": [...], "is_r2_train": [...],
            "bl_nrmse_test": [...], "is_nrmse_test": [...],
            "bl_wall": [...], "is_wall": [...],
            "bl_search": [...], "is_search": [...],
            "bl_complexity": [...], "is_complexity": [...],
            "bl_solution": [...], "is_solution": [...],
            "is_rf": [...], "is_redundancy": [...],
            "is_overhead_pct": [...], "is_per_dag_ms": [...],
            "is_canon_ms": [...], "is_eval_ms": [...],
            "is_max_k": [...],
        }}
    """
    bench_dir = results_dir / method / benchmark
    data: dict[str, dict[str, list[float]]] = {}
    if not bench_dir.exists():
        return data

    for prob_dir in sorted(bench_dir.iterdir()):
        if not prob_dir.is_dir():
            continue
        d: dict[str, list[float]] = defaultdict(list)

        for variant, prefix in [("baseline", "bl"), ("isalsr", "is")]:
            vdir = prob_dir / variant
            if not vdir.exists():
                continue
            for rl in load_all_run_logs(vdir):
                r2t = min(max(rl.regression.r2_test, 0.0), 1.0)
                r2tr = min(max(rl.regression.r2_train, 0.0), 1.0)
                d[f"{prefix}_r2_test"].append(r2t)
                d[f"{prefix}_r2_train"].append(r2tr)
                d[f"{prefix}_nrmse_test"].append(rl.regression.nrmse_test)
                d[f"{prefix}_wall"].append(rl.time.wall_clock_total_s)
                d[f"{prefix}_search"].append(rl.time.wall_clock_search_only_s)
                d[f"{prefix}_complexity"].append(float(rl.regression.model_complexity))
                d[f"{prefix}_solution"].append(1.0 if rl.regression.solution_recovered else 0.0)

                if variant == "isalsr":
                    ss = rl.search_space
                    t = rl.time
                    d["is_rf"].append(ss.empirical_reduction_factor)
                    d["is_redundancy"].append(ss.redundancy_rate)
                    w = t.wall_clock_total_s
                    n = ss.total_dags_explored
                    if w > 0:
                        d["is_overhead_pct"].append(t.overhead_time_s / w * 100)
                    if n > 0:
                        d["is_per_dag_ms"].append(t.canonicalization_runtime_s / n * 1000)
                        d["is_eval_ms"].append(t.evaluation_time_s / n * 1000)
                    d["is_canon_ms"].append(t.canonicalization_runtime_s)
                    d["is_max_k"].append(float(ss.max_internal_nodes_seen))

        if d:
            data[prob_dir.name] = dict(d)

    return data


def _paired_test(bl: list[float], is_: list[float]) -> tuple[float, float]:
    """Run paired test (t-test or Wilcoxon) and return (cohens_d, p_value)."""
    bl_a, is_a = np.array(bl), np.array(is_)
    n = min(len(bl_a), len(is_a))
    if n < 3:
        return 0.0, 1.0
    bl_a, is_a = bl_a[:n], is_a[:n]
    diff = is_a - bl_a
    sd = np.std(diff, ddof=1)
    d = float(np.mean(diff) / sd) if sd > 1e-10 else 0.0
    try:
        _, sw_p = sp_stats.shapiro(diff)
        if sw_p > 0.05:
            _, p = sp_stats.ttest_rel(bl_a, is_a)
        else:
            res = sp_stats.wilcoxon(bl_a, is_a)
            p = res.pvalue
    except Exception:  # noqa: BLE001
        p = 1.0
    return d, float(p)


def _cohens_d_with_ci(
    bl: list[float],
    is_: list[float],
    n_boot: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute Cohen's d (paired) with bootstrap 95% CI."""
    n = min(len(bl), len(is_))
    if n < 3:
        return 0.0, 0.0, 0.0
    diff = np.array(is_[:n]) - np.array(bl[:n])
    sd = np.std(diff, ddof=1)
    d = float(np.mean(diff) / sd) if sd > 1e-10 else 0.0

    rng = np.random.default_rng(seed)
    boot_ds = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        s = np.std(sample, ddof=1)
        boot_ds[b] = np.mean(sample) / s if s > 1e-10 else 0.0
    ci_lo = float(np.percentile(boot_ds, 2.5))
    ci_hi = float(np.percentile(boot_ds, 97.5))
    return d, ci_lo, ci_hi


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        cummax = max(cummax, adj)
        adjusted[orig_idx] = min(cummax, 1.0)
    return adjusted


# ======================================================================
# Problem labels
# ======================================================================

_PROBLEM_LABELS = {
    "nguyen_1": "N-1",
    "nguyen_2": "N-2",
    "nguyen_3": "N-3",
    "nguyen_4": "N-4",
    "nguyen_5": "N-5",
    "nguyen_6": "N-6",
    "nguyen_7": "N-7",
    "nguyen_8": "N-8",
    "nguyen_9": "N-9",
    "nguyen_10": "N-10",
    "nguyen_11": "N-11",
    "nguyen_12": "N-12",
    "i.6.20a": "I.6.20a",
    "i.12.1": "I.12.1",
    "i.12.4": "I.12.4",
    "i.14.3": "I.14.3",
    "i.25.13": "I.25.13",
    "i.34.27": "I.34.27",
    "i.39.10": "I.39.10",
    "i.48.20": "I.48.20",
    "i.10.7": "I.10.7",
    "ii.3.24": "II.3.24",
}


# ======================================================================
# Table 1: Unified Three-Axis + Overhead Summary
# ======================================================================


def generate_table1(
    results_dir: Path,
    methods: list[str],
    benchmarks: list[str],
    output_dir: Path,
) -> None:
    """Unified summary table merging three-axis and overhead breakdown.

    One row per (method x benchmark). Columns cover search space reduction,
    regression quality, and computational overhead in a single view.
    """
    rows: list[str] = []

    for method in methods:
        for benchmark in benchmarks:
            data = _load_paired_metrics(results_dir, method, benchmark)
            if not data:
                continue

            # Aggregate across problems
            all_rf, all_rr, all_oh = [], [], []
            all_canon_ms, all_eval_ms = [], []
            bl_r2s, is_r2s = [], []
            n_sig = 0
            n_prob = 0
            speedups = []
            p_values = []

            for prob, d in data.items():
                if not d.get("bl_r2_test") or not d.get("is_r2_test"):
                    continue
                n_prob += 1
                bl_r2s.append(np.mean(d["bl_r2_test"]))
                is_r2s.append(np.mean(d["is_r2_test"]))
                _, p = _paired_test(d["bl_r2_test"], d["is_r2_test"])
                p_values.append(p)

                if d.get("is_rf"):
                    all_rf.extend(d["is_rf"])
                if d.get("is_redundancy"):
                    all_rr.extend(d["is_redundancy"])
                if d.get("is_overhead_pct"):
                    all_oh.extend(d["is_overhead_pct"])
                if d.get("is_per_dag_ms"):
                    all_canon_ms.extend(d["is_per_dag_ms"])
                if d.get("is_eval_ms"):
                    all_eval_ms.extend(d["is_eval_ms"])
                if d.get("bl_search") and d.get("is_search"):
                    bl_s = np.mean(d["bl_search"])
                    is_s = np.mean(d["is_search"])
                    if is_s > 0:
                        speedups.append(bl_s / is_s)

            # Holm correction for R² significance count
            adjusted = _holm_bonferroni(p_values) if p_values else []
            n_sig = sum(1 for p in adjusted if p < 0.05)

            rf_mean = np.mean(all_rf) if all_rf else 1.0
            rf_std = np.std(all_rf, ddof=1) if len(all_rf) > 1 else 0.0
            rr_mean = np.mean(all_rr) * 100 if all_rr else 0.0
            oh_mean = np.mean(all_oh) if all_oh else 0.0
            s_mean = np.mean(speedups) if speedups else 1.0
            bl_r2_mean = np.mean(bl_r2s) if bl_r2s else 0.0
            is_r2_mean = np.mean(is_r2s) if is_r2s else 0.0
            canon_mean = np.mean(all_canon_ms) if all_canon_ms else 0.0
            eval_mean = np.mean(all_eval_ms) if all_eval_ms else 0.0
            ratio = canon_mean / eval_mean if eval_mean > 0 else float("inf")
            ratio_str = f"{ratio:.1f}" if ratio < 100 else f"{ratio:.0f}"

            method_label = method.upper()
            bench_label = benchmark.capitalize()
            rows.append(
                f"    {method_label:<6} & {bench_label:<8} & {n_prob:>2} "
                f"& ${rf_mean:.2f} \\pm {rf_std:.2f}$ "
                f"& ${rr_mean:.1f}\\%$ "
                f"& ${bl_r2_mean:.3f}$ / ${is_r2_mean:.3f}$ "
                f"& {n_sig}/{n_prob} "
                f"& ${canon_mean:.3f}$ & ${eval_mean:.2f}$ "
                f"& ${ratio_str}$ "
                f"& ${oh_mean:.1f}\\%$ "
                f"& ${s_mean:.2f}$ \\\\"
            )

    tex = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Unified three-axis summary of \\IsalSR{} integration. "
        "$\\rho$: empirical reduction factor. "
        "$R^2$ sig.: number of problems with significant improvement "
        "(Holm-corrected, $\\alpha{=}0.05$). "
        "$T_{\\mathrm{canon}}$/$T_{\\mathrm{eval}}$: per-DAG "
        "canonicalization/evaluation cost (ms). "
        "OH: canonicalization overhead as percentage of total time. "
        "$S$: search-only speedup (baseline/IsalSR).}\n"
        "\\label{tab:three_axis}\n"
        "\\begin{tabular}{@{}llc cc cc rrr cc@{}}\n"
        "\\toprule\n"
        " & & & \\multicolumn{2}{c}{Search Space} "
        "& \\multicolumn{2}{c}{Regression Quality} "
        "& \\multicolumn{5}{c}{Computational Cost} \\\\\n"
        "\\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-12}\n"
        "Method & Bench. & $n$ "
        "& $\\rho$ & Red. "
        "& $R^2$ test (BL/IS) & Sig. "
        "& $T_{\\mathrm{canon}}$ & $T_{\\mathrm{eval}}$ & Ratio "
        "& OH & $S$ \\\\\n"
        "\\midrule\n"
    )
    tex += "\n".join(rows) + "\n"
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"

    out = output_dir / "table1_three_axis_summary.tex"
    out.write_text(tex)
    log.info("Saved %s", out)


# ======================================================================
# Table 2: Per-Problem R² Comparison
# ======================================================================


def generate_table2(
    results_dir: Path,
    methods: list[str],
    benchmarks: list[str],
    output_dir: Path,
) -> None:
    """Per-problem R² comparison with Cohen's d and Holm p-value."""

    for method in methods:
        rows: list[str] = []
        prev_bench = ""

        for benchmark in benchmarks:
            data = _load_paired_metrics(results_dir, method, benchmark)
            if not data:
                continue

            # Collect p-values for Holm correction within this benchmark
            prob_stats: list[tuple[str, float, float, float, float]] = []
            for prob in sorted(data.keys()):
                d = data[prob]
                if not d.get("bl_r2_test") or not d.get("is_r2_test"):
                    continue
                bl_mean = np.mean(d["bl_r2_test"])
                is_mean = np.mean(d["is_r2_test"])
                cohens_d, p_raw = _paired_test(d["bl_r2_test"], d["is_r2_test"])
                prob_stats.append((prob, bl_mean, is_mean, cohens_d, p_raw))

            # Holm correction
            raw_ps = [s[4] for s in prob_stats]
            adjusted = _holm_bonferroni(raw_ps) if raw_ps else []

            for i, (prob, bl_mean, is_mean, cohens_d, _) in enumerate(prob_stats):
                p_adj = adjusted[i] if i < len(adjusted) else 1.0
                sig = ""
                if p_adj < 0.001:
                    sig = "$^{***}$"
                elif p_adj < 0.01:
                    sig = "$^{**}$"
                elif p_adj < 0.05:
                    sig = "$^{*}$"

                label = _PROBLEM_LABELS.get(prob, prob)

                # Add midrule between benchmarks
                if benchmark != prev_bench and rows:
                    rows.append("    \\midrule")
                    prev_bench = benchmark
                elif not rows:
                    prev_bench = benchmark

                delta = is_mean - bl_mean
                delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
                rows.append(
                    f"    {label:<8} "
                    f"& ${bl_mean:.4f}$ & ${is_mean:.4f}$ "
                    f"& ${delta_str}$ "
                    f"& ${cohens_d:+.2f}$ "
                    f"& ${p_adj:.3f}${sig} \\\\"
                )

        method_label = method.upper()
        tex = (
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\small\n"
            f"\\caption{{Per-problem $R^2$ test comparison for {method_label}. "
            f"Cohen's $d$: paired effect size. "
            f"$p$: Holm-adjusted. "
            f"$^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$.}}\n"
            f"\\label{{tab:r2_per_problem_{method}}}\n"
            "\\begin{tabular}{@{}l rr r r r@{}}\n"
            "\\toprule\n"
            "Problem & BL $R^2$ & IS $R^2$ & $\\Delta$ & $d$ & $p_{\\mathrm{Holm}}$ \\\\\n"
            "\\midrule\n"
        )
        tex += "\n".join(rows) + "\n"
        tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

        out = output_dir / f"table2_r2_per_problem_{method}.tex"
        out.write_text(tex)
        log.info("Saved %s", out)


# ======================================================================
# Table K: Bingo k-Range Overhead (Discussion)
# ======================================================================


def generate_table_k_range(
    results_dir: Path,
    methods: list[str],
    benchmarks: list[str],
    output_dir: Path,
) -> None:
    """Bingo k-range computational overhead breakdown for Discussion section.

    Stratifies overhead by expression complexity (max internal nodes k)
    to show how canonicalization cost scales with DAG size.
    """
    k_data: list[tuple[int, float, float]] = []
    for benchmark in benchmarks:
        data = _load_paired_metrics(results_dir, "bingo", benchmark)
        for prob, d in data.items():
            if not d.get("is_max_k") or not d.get("is_overhead_pct"):
                continue
            for mk, oh, pd in zip(
                d.get("is_max_k", []),
                d.get("is_overhead_pct", []),
                d.get("is_per_dag_ms", []),
            ):
                if np.isfinite(mk) and np.isfinite(oh) and np.isfinite(pd):
                    k_data.append((int(mk), oh, pd))

    if not k_data:
        log.warning("No Bingo k-range data available")
        return

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Bingo canonicalization overhead stratified by expression "
        "complexity $k$ (maximum internal nodes). Overhead increases with $k$ "
        "due to the combinatorial nature of the canonical form computation, "
        "but remains bounded by the greedy WL-hash algorithm.}\n"
        "\\label{tab:k_range_overhead}\n"
        "\\begin{tabular}{@{}l rrr@{}}\n"
        "\\toprule\n"
        "$k$-range & $N$ & OH (\\%) & $T_{\\mathrm{canon}}$ (ms) \\\\\n"
        "\\midrule\n"
    )
    for lo, hi in [(0, 5), (5, 15), (15, 32)]:
        subset = [(oh, pd) for mk, oh, pd in k_data if lo <= mk < hi]
        if subset:
            ohs, pds = zip(*subset)
            tex += (
                f"    $[{lo}, {hi})$ & {len(subset)} "
                f"& ${np.mean(ohs):.1f}\\%$ "
                f"& ${np.mean(pds):.3f}$ \\\\\n"
            )
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    out = output_dir / "table_k_range_overhead.tex"
    out.write_text(tex)
    log.info("Saved %s", out)


# ======================================================================
# Table S: Supplementary Per-Problem Statistics
# ======================================================================


def _fmt_bold_underline(
    bl_val: float,
    is_val: float,
    fmt: str,
    higher_is_better: bool = True,
) -> tuple[str, str]:
    """Format two values with bold (best) and underline (worst).

    Compares formatted strings to avoid spurious bold/underline when both
    values display identically (e.g., both "0.0000" despite fp noise).

    Returns LaTeX strings for (bl_formatted, is_formatted).
    """
    bl_str = f"{bl_val:{fmt}}"
    is_str = f"{is_val:{fmt}}"

    # If they display the same, no bold/underline
    if bl_str == is_str:
        return f"${bl_str}$", f"${is_str}$"

    if higher_is_better:
        bl_better = bl_val > is_val
    else:
        bl_better = bl_val < is_val

    if bl_better:
        return f"$\\mathbf{{{bl_str}}}$", f"$\\underline{{{is_str}}}$"
    else:
        return f"$\\underline{{{bl_str}}}$", f"$\\mathbf{{{is_str}}}$"


def generate_table_supplementary(
    results_dir: Path,
    methods: list[str],
    benchmarks: list[str],
    output_dir: Path,
) -> None:
    """Comprehensive per-problem supplementary table (one per method).

    Designed for TPAMI supplementary material. Columns:
      Quality:      R² test (BL/IS), NRMSE test (BL/IS)
      Effect size:  Cohen's d [95% CI], p_Holm
      Search space: ρ (mean ± std), Redundancy rate
      Computation:  T_total BL/IS (s), Overhead %

    Bold = better of BL/IS per metric per problem.
    Underline = worse of BL/IS per metric per problem.
    """

    for method in methods:
        all_rows: list[str] = []
        prev_bench = ""

        for benchmark in benchmarks:
            data = _load_paired_metrics(results_dir, method, benchmark)
            if not data:
                continue

            # Collect all problems for Holm correction across this benchmark
            prob_stats: list[
                tuple[
                    str,  # problem name
                    dict[str, list[float]],  # raw data
                    float,  # raw p-value
                ]
            ] = []
            for prob in sorted(data.keys()):
                d = data[prob]
                if not d.get("bl_r2_test") or not d.get("is_r2_test"):
                    continue
                _, p_raw = _paired_test(d["bl_r2_test"], d["is_r2_test"])
                prob_stats.append((prob, d, p_raw))

            raw_ps = [s[2] for s in prob_stats]
            adjusted = _holm_bonferroni(raw_ps) if raw_ps else []

            for i, (prob, d, _) in enumerate(prob_stats):
                p_adj = adjusted[i] if i < len(adjusted) else 1.0

                # Midrule between benchmarks
                if benchmark != prev_bench and all_rows:
                    all_rows.append("    \\midrule")
                    prev_bench = benchmark
                elif not all_rows:
                    prev_bench = benchmark

                label = _PROBLEM_LABELS.get(prob, prob)

                # R² test
                bl_r2 = np.mean(d["bl_r2_test"])
                is_r2 = np.mean(d["is_r2_test"])
                bl_r2_f, is_r2_f = _fmt_bold_underline(bl_r2, is_r2, ".4f", higher_is_better=True)

                # NRMSE test
                bl_nrmse = np.mean(d.get("bl_nrmse_test", [0.0]))
                is_nrmse = np.mean(d.get("is_nrmse_test", [0.0]))
                bl_nrmse_f, is_nrmse_f = _fmt_bold_underline(
                    bl_nrmse, is_nrmse, ".4f", higher_is_better=False
                )

                # Cohen's d with bootstrap CI
                cd, ci_lo, ci_hi = _cohens_d_with_ci(d["bl_r2_test"], d["is_r2_test"])
                d_str = f"${cd:+.2f}$\\,[${ci_lo:+.2f}$, ${ci_hi:+.2f}$]"

                # p-value with significance stars
                sig = ""
                if p_adj < 0.001:
                    sig = "$^{***}$"
                elif p_adj < 0.01:
                    sig = "$^{**}$"
                elif p_adj < 0.05:
                    sig = "$^{*}$"
                if p_adj < 0.001:
                    p_str = f"$<$0.001{sig}"
                else:
                    p_str = f"${p_adj:.3f}${sig}"

                # Reduction factor
                rf_mean = np.mean(d.get("is_rf", [1.0]))
                rf_std = np.std(d["is_rf"], ddof=1) if len(d.get("is_rf", [])) > 1 else 0.0
                rf_str = f"${rf_mean:.2f} \\pm {rf_std:.2f}$"

                # Redundancy rate
                rr = np.mean(d.get("is_redundancy", [0.0])) * 100
                rr_str = f"${rr:.1f}\\%$"

                # Wall-clock time (bold lower = better)
                bl_t = np.mean(d.get("bl_wall", [0.0]))
                is_t = np.mean(d.get("is_wall", [0.0]))
                bl_t_f, is_t_f = _fmt_bold_underline(bl_t, is_t, ".1f", higher_is_better=False)

                # Overhead
                oh = np.mean(d.get("is_overhead_pct", [0.0]))
                oh_str = f"${oh:.1f}\\%$"

                all_rows.append(
                    f"    {label:<8} "
                    f"& {bl_r2_f} & {is_r2_f} "
                    f"& {bl_nrmse_f} & {is_nrmse_f} "
                    f"& {d_str} & {p_str} "
                    f"& {rf_str} & {rr_str} "
                    f"& {bl_t_f} & {is_t_f} & {oh_str} \\\\"
                )

        if not all_rows:
            continue

        method_label = method.upper()
        tex = (
            "\\begin{table*}[t]\n"
            "\\centering\n"
            "\\scriptsize\n"
            f"\\caption{{Per-problem comparison of native DAG representation "
            f"(BL) versus \\IsalSR{{}} canonicalization (IS) for "
            f"{method_label}. "
            f"Cohen's $d$: paired effect size with 95\\% bootstrap CI "
            f"(10,000 resamples). "
            f"$p$: Holm-adjusted ($^{{*}}p<0.05$, $^{{**}}p<0.01$, "
            f"$^{{***}}p<0.001$). "
            f"$\\rho$: empirical reduction factor. "
            f"Red.: redundancy rate. "
            f"$T$: mean wall-clock time (s). "
            f"OH: canonicalization overhead. "
            f"\\textbf{{Bold}}: better of BL/IS. "
            f"\\underline{{Underline}}: worse of BL/IS.}}\n"
            f"\\label{{tab:supplementary_{method}}}\n"
            "\\begin{tabular}{@{}l "
            "rr "  # R² test BL/IS
            "rr "  # NRMSE test BL/IS
            "l r "  # d [CI], p
            "r r "  # ρ, Red.
            "rr r"  # T_BL, T_IS, OH
            "@{}}\n"
            "\\toprule\n"
            " & \\multicolumn{2}{c}{$R^2$ test} "
            "& \\multicolumn{2}{c}{NRMSE test} "
            "& \\multicolumn{2}{c}{Effect size} "
            "& \\multicolumn{2}{c}{Search space} "
            "& \\multicolumn{3}{c}{Computation} \\\\\n"
            "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} "
            "\\cmidrule(lr){6-7} \\cmidrule(lr){8-9} "
            "\\cmidrule(lr){10-12}\n"
            "Problem "
            "& BL & IS "
            "& BL & IS "
            "& $d$\\,[95\\% CI] & $p_{\\mathrm{Holm}}$ "
            "& $\\rho$ & Red. "
            "& $T_{\\mathrm{BL}}$ (s) & $T_{\\mathrm{IS}}$ (s) & OH \\\\\n"
            "\\midrule\n"
        )
        tex += "\n".join(all_rows) + "\n"
        tex += "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"

        out = output_dir / f"table_supplementary_{method}.tex"
        out.write_text(tex)
        log.info("Saved %s", out)


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for IsalSR")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="udfs,bingo")
    parser.add_argument("--benchmarks", default="nguyen,feynman")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    methods = [m.strip() for m in args.methods.split(",")]
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating tables from %s", results_dir)
    generate_table1(results_dir, methods, benchmarks, output_dir)
    generate_table2(results_dir, methods, benchmarks, output_dir)
    generate_table_k_range(results_dir, methods, benchmarks, output_dir)
    generate_table_supplementary(results_dir, methods, benchmarks, output_dir)
    log.info("All tables saved to %s", output_dir)


if __name__ == "__main__":
    main()
