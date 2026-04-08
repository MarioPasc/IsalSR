"""Generate aggregate diversity metrics figure across all benchmarks.

Two-panel figure (1 row x 2 columns) at IEEE single-column width:
  Left:  mean pairwise BP-GED and Levenshtein distance
  Right: effective diversity ratio (delta) and best R^2

Data is aggregated across all benchmark problems.  Per-run R^2 is
monotonised (cumulative max) before aggregation — since best_r2
represents the historical best, it should never decrease; observed
drops are snapshot-computation artifacts.

Usage:
    python -m experiments.scripts.diversity.generate_fig_aggregate_metrics \
        --input-dir /path/to/diversity \
        --output-dir /path/to/diversity
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from experiments.plotting_styles import (
    COLOR_ISALSR,
    COLOR_NATIVE,
    PLOT_SETTINGS,
    apply_ieee_style,
    save_figure,
)
from experiments.scripts.diversity.generate_fig_diversity import (
    RESULTS_DIR,
    load_all_trajectories,
    load_summary,
)

log = logging.getLogger(__name__)

BENCHMARKS: list[str] = ["I.10.7", "I.12.4", "Nguyen-1"]

SMOOTH_WINDOW: int = 31
SMOOTH_POLYORDER: int = 3


# ======================================================================
# Data loading + imputation
# ======================================================================


def _load_all_problems(
    base_dir: Path,
    benchmarks: list[str],
) -> pd.DataFrame:
    """Load trajectory + summary data from all benchmarks.

    Applies cumulative-max imputation to best_r2 per (problem, seed,
    variant) so the metric is monotonically non-decreasing.
    """
    frames: list[pd.DataFrame] = []
    for bench in benchmarks:
        bench_dir = base_dir / bench
        if not bench_dir.exists():
            log.warning("Benchmark directory not found: %s", bench_dir)
            continue

        traj = load_all_trajectories(bench_dir)
        summ = load_summary(bench_dir)

        cols = ["seed", "variant", "generation", "delta", "d_bar_bp", "d_bar_lev", "best_r2"]
        traj_cols = [c for c in cols if c in traj.columns]
        summ_cols = [c for c in cols if c in summ.columns]

        combined = pd.concat([traj[traj_cols], summ[summ_cols]])
        combined = combined.drop_duplicates(subset=["seed", "variant", "generation"], keep="last")
        combined = combined.sort_values("generation")

        # Monotonise best_r2: cumulative max per (seed, variant)
        combined["best_r2"] = combined.groupby(["seed", "variant"])["best_r2"].cummax()

        combined["problem"] = bench
        combined["uid"] = bench + "_s" + combined["seed"].astype(str)
        frames.append(combined)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ======================================================================
# Smoothing + plotting
# ======================================================================


def _smooth(values: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing. Falls back to original if too short."""
    if len(values) < SMOOTH_WINDOW:
        return values
    return savgol_filter(values, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER)


def _plot_metric(
    ax: Any,
    df: pd.DataFrame,
    metric: str,
    color: str,
    linestyle: str = "-",
    smooth: bool = True,
) -> None:
    """Plot one metric as smoothed mean +/- 95% CI band (no legend label)."""
    clean = df.dropna(subset=[metric]).sort_values("generation")
    grouped = clean.groupby("generation")[metric]
    means = grouped.mean()
    sems = grouped.sem().fillna(0)
    gens = means.index.values
    mu = means.values
    ci_lo = mu - 1.96 * sems.values
    ci_hi = mu + 1.96 * sems.values

    if smooth and len(mu) >= SMOOTH_WINDOW:
        mu = _smooth(mu)
        ci_lo = _smooth(ci_lo)
        ci_hi = _smooth(ci_hi)

    lw = float(PLOT_SETTINGS["line_width"])
    alpha = float(PLOT_SETTINGS["error_band_alpha"])

    ax.plot(gens, mu, color=color, linewidth=lw, linestyle=linestyle)
    ax.fill_between(gens, ci_lo, ci_hi, color=color, alpha=alpha)


# ======================================================================
# Main figure
# ======================================================================


def generate_figure(
    base_dir: Path,
    output_dir: Path,
    benchmarks: list[str] | None = None,
) -> None:
    """Generate the two-panel aggregate metrics figure."""
    apply_ieee_style()

    if benchmarks is None:
        benchmarks = BENCHMARKS

    df = _load_all_problems(base_dir, benchmarks)
    if df.empty:
        log.error("No data loaded. Check input directory.")
        return

    n_problems = df["problem"].nunique()
    n_seeds = df.groupby("problem")["seed"].nunique().min()
    log.info("Loaded %d rows from %d problems, ~%d seeds each", len(df), n_problems, n_seeds)

    fig_width = float(PLOT_SETTINGS["figure_width_single"])
    fig_height = fig_width * 1.0
    fig, (ax_dist, ax_ratio) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    tick_fs = int(PLOT_SETTINGS["tick_labelsize"]) - 1
    label_fs = int(PLOT_SETTINGS["axes_labelsize"]) - 1
    legend_fs = int(PLOT_SETTINGS["legend_fontsize"]) - 2

    # ------------------------------------------------------------------
    # Left panel: pairwise distances
    # ------------------------------------------------------------------
    for variant, color in [("baseline", COLOR_NATIVE), ("isalsr", COLOR_ISALSR)]:
        sub = df[df["variant"] == variant]
        _plot_metric(ax_dist, sub, "d_bar_bp", color, linestyle="-")
        _plot_metric(ax_dist, sub, "d_bar_lev", color, linestyle="--")

    ax_dist.set_xlabel("Generation $t$", fontsize=label_fs)
    ax_dist.set_ylabel(r"$\bar{d}(P_t)$", fontsize=label_fs)
    ax_dist.tick_params(labelsize=tick_fs)
    ax_dist.set_title("Pairwise distance", fontsize=label_fs)

    # Legend under left panel: color patches only
    patch_native = mpatches.Patch(color=COLOR_NATIVE, label="Native DAG")
    patch_isalsr = mpatches.Patch(color=COLOR_ISALSR, label="IsalSR")
    ax_dist.legend(
        handles=[patch_native, patch_isalsr],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=legend_fs,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
    )

    # ------------------------------------------------------------------
    # Right panel: delta + R²
    # ------------------------------------------------------------------
    for variant, color in [("baseline", COLOR_NATIVE), ("isalsr", COLOR_ISALSR)]:
        sub = df[df["variant"] == variant]
        # Exclude gen 0 for IsalSR delta: the initial random population has
        # not undergone selection yet (delta < 1.0 by construction).  Including
        # it causes Savitzky-Golay smoothing to bleed the outlier into gen 1+,
        # visually misrepresenting the invariant delta = 1.0 at all t >= 1.
        sub_delta = sub[sub["generation"] >= 1] if variant == "isalsr" else sub
        _plot_metric(ax_ratio, sub_delta, "delta", color, linestyle="-")
        _plot_metric(ax_ratio, sub, "best_r2", color, linestyle="-.")

    ax_ratio.set_xlabel("Generation $t$", fontsize=label_fs)
    ax_ratio.set_ylabel(r"$\delta(P_t)$ / $R^2$", fontsize=label_fs)
    ax_ratio.set_ylim(-0.05, 1.08)
    ax_ratio.tick_params(labelsize=tick_fs)
    ax_ratio.set_title("Diversity and fitness", fontsize=label_fs)

    # Legend under right panel: linestyle handles only (black)
    line_bp = mlines.Line2D([], [], color="black", linestyle="-", linewidth=1.2, label="BP-GED")
    line_lev = mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.2, label="Lev.")
    line_delta = mlines.Line2D(
        [], [], color="black", linestyle="-", linewidth=1.2, label=r"$\delta$"
    )
    line_r2 = mlines.Line2D([], [], color="black", linestyle="-.", linewidth=1.2, label=r"$R^2$")
    ax_ratio.legend(
        handles=[line_bp, line_delta, line_lev, line_r2],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=legend_fs,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
        handlelength=2.0,
    )

    fig.subplots_adjust(bottom=0.25, left=0.14, right=0.97, top=0.92, wspace=0.45)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = str(output_dir / "fig_aggregate_metrics")
    saved = save_figure(fig, fig_path)
    plt.close(fig)
    log.info("Saved figure: %s", saved)

    n_obs = n_problems * n_seeds
    caption = (
        f"Aggregate diversity and regression metrics across {n_problems} "
        f"benchmark problems ({', '.join(benchmarks)}), "
        f"{n_obs} independent runs per variant "
        f"({n_seeds} seeds $\\times$ {n_problems} problems). "
        "Left panel: mean pairwise BP-GED (solid) and Levenshtein (dashed) distance. "
        "Right panel: effective diversity ratio $\\delta$ (solid) and best training "
        "$R^2$ (dash-dot). Bands show 95\\% confidence intervals; curves are smoothed "
        "with a Savitzky--Golay filter (window 31, order 3). Per-run $R^2$ is "
        "monotonised (cumulative max) before aggregation."
    )
    (output_dir / "fig_aggregate_metrics.caption.txt").write_text(caption)
    log.info("Saved caption")


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate aggregate diversity metrics figure")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(RESULTS_DIR / "diversity"),
        help="Base directory containing per-benchmark subdirectories",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=",".join(BENCHMARKS),
        help="Comma-separated benchmark names",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    base_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    generate_figure(base_dir, output_dir, benchmarks)


if __name__ == "__main__":
    main()
