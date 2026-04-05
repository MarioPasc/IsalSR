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

    # Sort by Cohen's d descending
    rows.sort(key=lambda r: r[3], reverse=True)

    n = len(rows)
    fig_height = max(3.5, 0.28 * n + 1.0)
    fig, ax = plt.subplots(figsize=(3.5, fig_height))

    y_positions = np.arange(n)

    for i, (method, bench, label, d, ci_lo, ci_hi) in enumerate(rows):
        color = METHOD_COLORS.get(method, "black")
        y = n - 1 - i  # top-to-bottom

        # CI horizontal line
        ax.plot([ci_lo, ci_hi], [y, y], color=color, linewidth=1.0, alpha=0.6)
        # Point estimate
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

    # Zero reference line
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)

    # Cohen's d benchmarks
    for threshold, label_text in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax.axvline(x=threshold, color="0.7", linestyle=":", linewidth=0.5)
        ax.axvline(x=-threshold, color="0.7", linestyle=":", linewidth=0.5)

    # Y-axis labels
    y_labels = []
    for i, (method, bench, label, d, ci_lo, ci_hi) in enumerate(rows):
        y_labels.append(f"{label} ({method.upper()})")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(reversed(y_labels), fontsize=7)
    ax.set_xlabel("Cohen's $d$ ($R^2$ test, IsalSR $-$ baseline)", fontsize=9)
    ax.set_title("Effect size per problem", fontsize=10, fontweight="bold", loc="left")
    ax.set_ylim(-0.5, n - 0.5)

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
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

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
        "native-DAG baseline. Horizontal bars: 95\\% bootstrap confidence "
        "intervals (10,000 resamples). Vertical dashed lines mark $d=0$ "
        "(no effect) and Cohen's benchmarks for small (0.2), medium (0.5), "
        "and large (0.8) effects. Problems where both variants achieve "
        "$R^2 = 1.0$ on all seeds yield $d = 0$ (ceiling effect); "
        "the concentrated large effects on hard problems (I.12.4, I.48.20) "
        "drive the aggregate improvement."
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
