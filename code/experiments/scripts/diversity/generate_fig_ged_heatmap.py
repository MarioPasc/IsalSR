"""Generate BP-GED heatmap matrix figure for Conjecture X.7.

3 rows x N columns: top row = Baseline heatmaps, middle row = IsalSR
heatmaps, bottom row = BP-GED and Levenshtein time-series.
Each heatmap panel shows a 200x200 pairwise bipartite GED distance
matrix reordered by hierarchical clustering to reveal block structure.

Usage:
    python -m experiments.scripts.diversity.generate_fig_ged_heatmap \
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from experiments.plotting_styles import (
    PAUL_TOL_BRIGHT,
    PLOT_SETTINGS,
    apply_ieee_style,
    save_figure,
)
from experiments.scripts.diversity.generate_fig_diversity import (
    COLOR_BASELINE,
    COLOR_ISALSR,
    RESULTS_DIR,
    load_all_trajectories,
    load_snapshot,
    load_summary,
)

log = logging.getLogger(__name__)

DEFAULT_DISPLAY_GENS: list[int] = [0, 25, 100, 300, 500]
DEFAULT_SEED: int = 0
HEATMAP_CMAP: str = "inferno"
LINKAGE_METHOD: str = "average"
COLOR_LEV: str = PAUL_TOL_BRIGHT["cyan"]


# ======================================================================
# Clustering and reordering
# ======================================================================


def cluster_reorder_matrix(
    dist_matrix: np.ndarray,
    method: str = LINKAGE_METHOD,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder a symmetric distance matrix by hierarchical clustering.

    Args:
        dist_matrix: (N, N) symmetric distance matrix with zero diagonal.
        method: Linkage method for scipy.cluster.hierarchy.linkage.

    Returns:
        (reordered_matrix, leaf_order)
    """
    mat = dist_matrix.copy().astype(np.float64)
    np.fill_diagonal(mat, 0.0)

    # Make perfectly symmetric
    mat = (mat + mat.T) / 2.0

    # Handle NaN/inf
    finite_max = np.nanmax(mat[np.isfinite(mat)]) if np.isfinite(mat).any() else 1.0
    mat = np.where(np.isfinite(mat), mat, finite_max)

    condensed = squareform(mat, checks=False)

    # If all distances are zero, linkage still works (arbitrary order)
    z = linkage(condensed, method=method)
    order = leaves_list(z)
    reordered = mat[np.ix_(order, order)]
    return reordered, order


# ======================================================================
# Panel drawing
# ======================================================================


def draw_ged_heatmap_panel(
    ax: Any,
    dist_matrix: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str = HEATMAP_CMAP,
) -> Any:
    """Draw one reordered BP-GED heatmap panel.

    Args:
        ax: Matplotlib axes.
        dist_matrix: (N, N) already reordered distance matrix.
        vmin: Shared color scale minimum.
        vmax: Shared color scale maximum.
        cmap: Colormap name.

    Returns:
        The image mappable for colorbar creation.
    """
    im = ax.imshow(
        dist_matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="none",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return im


# ======================================================================
# Global color scale
# ======================================================================


def compute_global_vmin_vmax(
    input_dir: Path,
    seed: int,
    display_gens: list[int],
) -> tuple[float, float]:
    """Compute shared vmin/vmax across all panels.

    Args:
        input_dir: Path to diversity results.
        seed: Representative seed index.
        display_gens: Generation numbers to scan.

    Returns:
        (0.0, global_max)
    """
    global_max = 0.0
    for gen in display_gens:
        for variant in ("baseline", "isalsr"):
            try:
                snap = load_snapshot(input_dir, seed, variant, gen)
                bp_mat = snap["bp_ged_distances"]
                finite_vals = bp_mat[np.isfinite(bp_mat)]
                if len(finite_vals) > 0:
                    global_max = max(global_max, float(finite_vals.max()))
            except FileNotFoundError:
                log.warning("Missing snapshot seed=%d %s gen=%d", seed, variant, gen)
    return 0.0, max(global_max, 1.0)


# ======================================================================
# Time-series panel
# ======================================================================


def draw_distance_timeseries(ax: Any, traj_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Draw BP-GED and Levenshtein time-series with 95% CI bands.

    Combines trajectory (dense, skips snapshot gens) with summary
    (snapshot gens only) for a complete curve.

    Args:
        ax: Matplotlib axes.
        traj_df: Trajectory DataFrame (dense per-generation).
        summary_df: Summary DataFrame (snapshot generations).
    """
    # Merge for complete coverage
    cols = ["seed", "variant", "generation", "d_bar_bp", "d_bar_lev"]
    traj_cols = [c for c in cols if c in traj_df.columns]
    summ_cols = [c for c in cols if c in summary_df.columns]
    combined = pd.concat([traj_df[traj_cols], summary_df[summ_cols]])
    combined = combined.drop_duplicates(subset=["seed", "variant", "generation"], keep="last")

    line_w = float(PLOT_SETTINGS["line_width"])
    band_alpha = float(PLOT_SETTINGS["error_band_alpha"])

    for variant, color_bp, label in [
        ("baseline", COLOR_BASELINE, "Baseline"),
        ("isalsr", COLOR_ISALSR, "IsalSR"),
    ]:
        sub = combined[combined["variant"] == variant]
        grouped = sub.groupby("generation")

        # BP-GED (solid lines)
        bp_means = grouped["d_bar_bp"].mean()
        bp_sems = grouped["d_bar_bp"].sem()
        gens = bp_means.index.values
        ax.plot(
            gens,
            bp_means.values,
            color=color_bp,
            linewidth=line_w,
            label=f"{label} BP-GED",
            marker="o",
            markersize=3,
        )
        ax.fill_between(
            gens,
            bp_means.values - 1.96 * bp_sems.values,
            bp_means.values + 1.96 * bp_sems.values,
            color=color_bp,
            alpha=band_alpha,
        )

        # Levenshtein (dashed lines)
        lev_means = grouped["d_bar_lev"].mean()
        lev_sems = grouped["d_bar_lev"].sem()
        ax.plot(
            gens,
            lev_means.values,
            color=color_bp,
            linewidth=line_w,
            linestyle="--",
            label=f"{label} Levenshtein",
            marker="s",
            markersize=2,
        )
        ax.fill_between(
            gens,
            lev_means.values - 1.96 * lev_sems.values,
            lev_means.values + 1.96 * lev_sems.values,
            color=color_bp,
            alpha=band_alpha * 0.5,
        )

    ax.set_xlabel("Generation $t$", fontsize=int(PLOT_SETTINGS["axes_labelsize"]))
    ax.set_ylabel(
        r"Mean pairwise distance $\bar{d}(P_t)$",
        fontsize=int(PLOT_SETTINGS["axes_labelsize"]),
    )
    ax.tick_params(labelsize=int(PLOT_SETTINGS["tick_labelsize"]))
    ax.legend(
        fontsize=int(PLOT_SETTINGS["legend_fontsize"]) - 1,
        loc="best",
        ncol=2,
    )


# ======================================================================
# Main figure
# ======================================================================


def generate_figure(
    input_dir: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    display_gens: list[int] | None = None,
) -> None:
    """Generate the BP-GED heatmap matrix figure with distance panel.

    Args:
        input_dir: Path to diversity results directory.
        output_dir: Path for output files.
        seed: Representative seed index.
        display_gens: List of generation numbers to display.
    """
    apply_ieee_style()

    if display_gens is None:
        display_gens = DEFAULT_DISPLAY_GENS

    vmin, vmax = compute_global_vmin_vmax(input_dir, seed, display_gens)
    log.info("Global BP-GED range: [%.1f, %.1f]", vmin, vmax)

    # Load trajectory + summary data for time-series
    traj_df = load_all_trajectories(input_dir)
    summary_df = load_summary(input_dir)

    n_cols = len(display_gens)
    fig_width = float(PLOT_SETTINGS["figure_width_double"])
    fig_height = fig_width * 0.58

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Use nested GridSpec: outer splits 3 rows, inner handles heatmap cols + colorbar
    outer_gs = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[1, 1, 0.7],
        hspace=0.25,
        left=0.06,
        right=0.92,
        top=0.93,
        bottom=0.08,
    )

    # Heatmap grid for rows 0 and 1 (with colorbar column)
    heatmap_gs_top = outer_gs[0].subgridspec(
        1, n_cols + 1, width_ratios=[1] * n_cols + [0.04], wspace=0.08
    )
    heatmap_gs_mid = outer_gs[1].subgridspec(
        1, n_cols + 1, width_ratios=[1] * n_cols + [0.04], wspace=0.08
    )

    variants = [
        ("baseline", COLOR_BASELINE, heatmap_gs_top),
        ("isalsr", COLOR_ISALSR, heatmap_gs_mid),
    ]

    im = None
    for variant, _color, sub_gs in variants:
        for col_idx, gen in enumerate(display_gens):
            ax = fig.add_subplot(sub_gs[0, col_idx])
            try:
                snap = load_snapshot(input_dir, seed, variant, gen)
                bp_mat = snap["bp_ged_distances"]
                reordered, _order = cluster_reorder_matrix(bp_mat)
                im = draw_ged_heatmap_panel(ax, reordered, vmin, vmax)
            except FileNotFoundError:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=int(PLOT_SETTINGS["annotation_fontsize"]),
                )
                ax.set_xticks([])
                ax.set_yticks([])

            # Column titles on top row only
            if variant == "baseline":
                ax.set_title(
                    f"$t = {gen}$",
                    fontsize=int(PLOT_SETTINGS["tick_labelsize"]),
                )

    # Shared colorbar (spanning both heatmap rows via the top subgridspec slot)
    if im is not None:
        cbar_ax = fig.add_subplot(heatmap_gs_top[0, -1])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("BP-GED", fontsize=int(PLOT_SETTINGS["tick_labelsize"]))
        cbar.ax.tick_params(labelsize=int(PLOT_SETTINGS["annotation_fontsize"]) - 1)

    # Row labels
    fig.text(
        0.02,
        0.78,
        "Baseline",
        rotation=90,
        fontsize=int(PLOT_SETTINGS["axes_titlesize"]),
        va="center",
        ha="center",
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.48,
        "IsalSR",
        rotation=90,
        fontsize=int(PLOT_SETTINGS["axes_titlesize"]),
        va="center",
        ha="center",
        fontweight="bold",
        color=COLOR_ISALSR,
    )

    # Row 3: BP-GED + Levenshtein time-series
    ax_dist = fig.add_subplot(outer_gs[2])
    if not traj_df.empty or not summary_df.empty:
        draw_distance_timeseries(ax_dist, traj_df, summary_df)
    else:
        ax_dist.text(0.5, 0.5, "No trajectory data", transform=ax_dist.transAxes, ha="center")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = str(output_dir / "fig_ged_heatmap_matrix")
    saved = save_figure(fig, fig_path)
    plt.close(fig)
    log.info("Saved figure: %s", saved)

    caption = (
        "Pairwise bipartite graph edit distance matrices across generations "
        "for seed 0 of the I.48.20 benchmark. Top row: Bingo baseline; "
        "middle row: IsalSR. Rows and columns are reordered by hierarchical "
        "clustering (average linkage) to reveal block structure. A shared "
        "color scale is used across all panels. Bottom row: mean pairwise "
        "BP-GED (solid) and Levenshtein (dashed) distances across 20 seeds "
        "with 95\\% confidence intervals. The baseline develops large "
        "uniform blocks (duplicated individuals) that grow over time, while "
        "IsalSR preserves richer distance structure, consistent with the "
        "higher effective diversity ratio $\\eta$."
    )
    caption_path = output_dir / "fig_ged_heatmap_matrix.caption.txt"
    caption_path.write_text(caption)
    log.info("Saved caption: %s", caption_path)


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BP-GED heatmap matrix figure")
    parser.add_argument("--input-dir", type=str, default=str(RESULTS_DIR / "diversity"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--gens",
        type=str,
        default=",".join(str(g) for g in DEFAULT_DISPLAY_GENS),
        help="Comma-separated generation numbers to display",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    display_gens = sorted(int(g) for g in args.gens.split(","))

    generate_figure(input_dir, output_dir, args.seed, display_gens)


if __name__ == "__main__":
    main()
