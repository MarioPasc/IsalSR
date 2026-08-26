"""Generate Forest Plot of Cohen's d for IsalSR model validation.

Per-problem effect sizes with 95% bootstrap CIs, sorted by magnitude.
One figure per method, or combined. Reveals the pattern: large positive
effects on hard problems, near-zero on easy ceiling-effect problems.

Usage:
    python -m experiments.figures.models.generate_forest_plot \
        --results-dir /path/to/results \
        --output-dir /path/to/figures
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.models.io_utils import load_all_run_logs  # noqa: E402
from experiments.plotting_styles import (  # noqa: E402
    PAUL_TOL_BRIGHT,
    apply_ieee_style,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

_PROBLEM_LABELS = {
    # Nguyen
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
    # Feynman
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
    # Hard
    "i.15.10": "I.15.10",
    "i.30.3": "I.30.3",
    "i.37.4": "I.37.4",
    "ii.11.27": "II.11.27",
    "iii.17.37": "III.17.37",
    "keijzer_6": "Keij-6",
    "korns_12": "Korns-12",
    "pagie_1": "Pagie-1",
    "vladislavleva_2": "Vlad-2",
    "vladislavleva_4": "Vlad-4",
    # Structural
    "i.16.6": "I.16.6",
    "i.29.16": "I.29.16",
    "i.50.26": "I.50.26",
    "ii.11.28": "II.11.28",
    "iii.14.14": "III.14.14",
    "keijzer_11": "Keij-11",
    "liv_14": "Liv-14",
    "r2": "R2",
    "r3": "R3",
    "vlad_7": "Vlad-7",
}

METHOD_COLORS = {
    "udfs": PAUL_TOL_BRIGHT["green"],
    "bingo": PAUL_TOL_BRIGHT["purple"],
}


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


def generate_forest_plot(
    results_dir: Path,
    output_dir: Path,
    methods: list[str],
    benchmarks: list[str],
) -> None:
    """Generate forest plot: one row per (method, problem), sorted by d."""
    apply_ieee_style()

    rows: list[tuple[str, str, str, float, float, float]] = []
    # (method, benchmark, problem_label, d, ci_lo, ci_hi)

    for method in methods:
        for benchmark in benchmarks:
            bench_dir = results_dir / method / benchmark
            if not bench_dir.exists():
                continue
            for prob_dir in sorted(bench_dir.iterdir()):
                if not prob_dir.is_dir():
                    continue
                bl_logs = (
                    load_all_run_logs(prob_dir / "baseline")
                    if (prob_dir / "baseline").exists()
                    else []
                )
                is_logs = (
                    load_all_run_logs(prob_dir / "isalsr") if (prob_dir / "isalsr").exists() else []
                )
                if len(bl_logs) < 3 or len(is_logs) < 3:
                    continue

                bl_r2 = [min(max(rl.regression.r2_test, 0.0), 1.0) for rl in bl_logs]
                is_r2 = [min(max(rl.regression.r2_test, 0.0), 1.0) for rl in is_logs]
                d, ci_lo, ci_hi = _cohens_d_with_ci(bl_r2, is_r2)

                label = _PROBLEM_LABELS.get(prob_dir.name, prob_dir.name)
                rows.append((method, benchmark, label, d, ci_lo, ci_hi))

    if not rows:
        log.warning("No data for forest plot")
        return

    # Load CPDT pooled results per method
    import json

    cpdt_summaries: list[tuple[str, float, float, float, str]] = []
    for method in methods:
        cpdt_path = results_dir / "analysis" / f"cross_problem_dominance_{method}_all.json"
        if cpdt_path.exists():
            with open(cpdt_path) as f:
                cpdt_data = json.load(f)
            r2_cpdt = cpdt_data.get("r2_test", {})
            if "error" not in r2_cpdt:
                d_val = r2_cpdt.get("cohens_d", 0.0)
                ci_lo_val = r2_cpdt.get("cohens_d_ci_lower", 0.0)
                ci_hi_val = r2_cpdt.get("cohens_d_ci_upper", 0.0)
                p_val = r2_cpdt.get("p_value_one_sided", 1.0)
                p_str = f"p={p_val:.4f}" if p_val >= 0.001 else "p<0.001"
                cpdt_summaries.append((method, d_val, ci_lo_val, ci_hi_val, p_str))

    # Sort by Cohen's d descending
    rows.sort(key=lambda r: r[3], reverse=True)

    n_per_problem = len(rows)
    n_cpdt = len(cpdt_summaries)
    n_total = n_per_problem + n_cpdt + (1 if n_cpdt > 0 else 0)
    fig_height = max(3.5, 0.28 * n_total + 1.0)
    fig, ax = plt.subplots(figsize=(3.5, fig_height))

    # Per-problem rows (top section)
    for i, (method, bench, label, d, ci_lo, ci_hi) in enumerate(rows):
        color = METHOD_COLORS.get(method, "black")
        y = n_total - 1 - i

        ax.plot([ci_lo, ci_hi], [y, y], color=color, linewidth=1.0, alpha=0.6)
        ax.plot(
            d,
            y,
            "o",
            color=color,
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.3,
            zorder=3,
        )

    # CPDT summary diamonds (bottom section, below a separator)
    if n_cpdt > 0:
        separator_y = n_total - n_per_problem - 1
        ax.axhline(y=separator_y + 0.5, color="black", linewidth=0.8, linestyle="-", alpha=0.4)

        for j, (method, d_val, ci_lo_val, ci_hi_val, p_str) in enumerate(cpdt_summaries):
            color = METHOD_COLORS.get(method, "black")
            y = separator_y - j

            ax.plot([ci_lo_val, ci_hi_val], [y, y], color=color, linewidth=1.5, alpha=0.8)
            ax.plot(
                d_val,
                y,
                "D",
                color=color,
                markersize=7,
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=4,
            )
            ax.annotate(
                p_str,
                (d_val, y),
                textcoords="offset points",
                xytext=(8, 0),
                fontsize=6,
                color=color,
                va="center",
            )

    # Zero reference line
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)

    # Cohen's d benchmarks
    for threshold, label_text in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax.axvline(x=threshold, color="0.7", linestyle=":", linewidth=0.5)
        ax.axvline(x=-threshold, color="0.7", linestyle=":", linewidth=0.5)

    # Y-axis labels
    y_labels_list: list[str] = []
    for i, (method, bench, label, d, ci_lo, ci_hi) in enumerate(rows):
        y_labels_list.append(f"{label} ({method.upper()})")
    if n_cpdt > 0:
        y_labels_list.append("")
        for method, _, _, _, _ in cpdt_summaries:
            y_labels_list.append(f"CPDT pooled ({method.upper()})")

    ax.set_yticks(np.arange(n_total))
    ax.set_yticklabels(reversed(y_labels_list), fontsize=7)
    ax.set_xlabel("Cohen's $d$ ($R^2$ test, IsalSR $-$ baseline)", fontsize=9)
    ax.set_title("Effect size per problem", fontsize=10, fontweight="bold", loc="left")
    ax.set_ylim(-0.5, n_total - 0.5)

    # Legend for methods
    for method in methods:
        ax.plot(
            [],
            [],
            "o",
            color=METHOD_COLORS.get(method, "black"),
            markersize=5,
            label=method.upper(),
        )
    ax.plot([], [], "D", color="gray", markersize=7, markeredgecolor="black", label="CPDT pooled")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out = output_dir / f"forest_plot_cohens_d.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        log.info("Saved %s", out)

    caption = (
        "Forest plot of Cohen's $d$ (paired) for $R^2$ test on each "
        "benchmark problem. Positive $d$: \\IsalSR{} improves over the "
        "native-DAG baseline. Circles: per-problem effects with 95\\% "
        "bootstrap CIs (10,000 resamples). Diamonds (bottom): Cross-Problem "
        "Dominance Test (CPDT) pooled across all 42 problems — each problem's "
        "mean $R^2$ is one paired observation. CPDT one-sided $p$-values "
        "annotated. Vertical dashed lines mark $d=0$ and Cohen's benchmarks."
    )
    (output_dir / "forest_plot_cohens_d.caption.txt").write_text(caption)
    log.info("Saved caption")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate forest plot")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="udfs,bingo")
    parser.add_argument("--benchmarks", default="nguyen,feynman")
    args = parser.parse_args()

    generate_forest_plot(
        Path(args.results_dir),
        Path(args.output_dir),
        [m.strip() for m in args.methods.split(",")],
        [b.strip() for b in args.benchmarks.split(",")],
    )


if __name__ == "__main__":
    main()
