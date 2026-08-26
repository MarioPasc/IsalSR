"""Generate PCA + KDE density landscape figure for Conjecture X.7.

3 rows x N columns: top row = Baseline PCA, middle row = IsalSR PCA,
bottom row = R^2 time-series (mean +/- 95% CI across seeds).
Each PCA panel shows KDE density contours of unique canonical classes
projected to joint PCA space, weighted by population count.

Usage:
    python -m experiments.scripts.diversity.generate_fig_pca_kde \
        --input-dir /path/to/diversity \
        --output-dir /path/to/diversity
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

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
    compute_joint_pca,
    load_all_trajectories,
    load_snapshot,
    load_summary,
)

log = logging.getLogger(__name__)

DEFAULT_DISPLAY_GENS: list[int] = [0, 25, 100, 300, 500]
DEFAULT_SEED: int = 0
STAR_COLOR: str = PAUL_TOL_BRIGHT["green"]
KDE_GRID_RESOLUTION: int = 100
KDE_CONTOUR_LEVELS: int = 8
KDE_BANDWIDTH_FACTOR: float = 0.3
POP_SIZE: int = 200


# ======================================================================
# Grouping and KDE
# ======================================================================


def group_by_canonical(
    canonical_strings: np.ndarray,
    pca_coords: np.ndarray,
    fitnesses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Group population by canonical string, returning unique representatives.

    Args:
        canonical_strings: (N,) array of canonical strings.
        pca_coords: (N, 2) PCA coordinates.
        fitnesses: (N,) MSE fitnesses.

    Returns:
        (unique_coords (U,2), weights (U,), best_idx_in_pop, n_unique)
    """
    counts = Counter(str(s) for s in canonical_strings)
    n_unique = len(counts)

    if n_unique == 0:
        return np.empty((0, 2)), np.empty(0), 0, 0

    # Pick first representative for each unique class
    seen: dict[str, int] = {}
    for i, cs in enumerate(canonical_strings):
        key = str(cs)
        if key not in seen:
            seen[key] = i

    rep_indices = list(seen.values())
    unique_coords = pca_coords[rep_indices]
    weights = np.array([counts[str(canonical_strings[i])] for i in rep_indices], dtype=np.float64)

    # Best individual (lowest finite MSE)
    finite_mask = np.isfinite(fitnesses)
    best_idx = int(np.argmin(np.where(finite_mask, fitnesses, np.inf))) if finite_mask.any() else 0

    return unique_coords, weights, best_idx, n_unique


def compute_kde_density(
    coords: np.ndarray,
    weights: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_resolution: int = KDE_GRID_RESOLUTION,
    bw_factor: float = KDE_BANDWIDTH_FACTOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute weighted KDE on 2D PCA coordinates.

    Args:
        coords: (U, 2) unique class PCA coordinates.
        weights: (U,) population counts per class.
        xlim: (xmin, xmax) for grid.
        ylim: (ymin, ymax) for grid.
        grid_resolution: Number of grid points per axis.
        bw_factor: Bandwidth scaling factor for Scott's rule.

    Returns:
        (xx, yy, density) meshgrid arrays, or None if KDE fails.
    """
    if len(coords) < 2:
        return None

    # Check if all points are effectively identical
    std = coords.std(axis=0)
    if np.allclose(std, 0, atol=1e-10):
        return None

    try:
        kde = gaussian_kde(coords.T, weights=weights, bw_method=bw_factor)
    except np.linalg.LinAlgError:
        # Singular covariance -- add tiny jitter and retry
        rng = np.random.default_rng(42)
        jittered = coords + rng.normal(0, 1e-6, coords.shape)
        try:
            kde = gaussian_kde(jittered.T, weights=weights, bw_method=bw_factor)
        except np.linalg.LinAlgError:
            return None

    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], grid_resolution),
        np.linspace(ylim[0], ylim[1], grid_resolution),
    )
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)
    return xx, yy, density


# ======================================================================
# Panel drawing
# ======================================================================


def draw_pca_kde_panel(
    ax: Any,
    pca_coords: np.ndarray,
    canonical_strings: np.ndarray,
    fitnesses: np.ndarray,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    n_total: int,
    best_r2: float,
) -> None:
    """Draw one PCA + KDE panel.

    Args:
        ax: Matplotlib axes.
        pca_coords: (N, 2) PCA coordinates for all individuals.
        canonical_strings: (N,) canonical strings.
        fitnesses: (N,) MSE fitnesses.
        color: Variant color (hex string).
        xlim: Shared x-axis limits.
        ylim: Shared y-axis limits.
        n_total: Population size (for delta computation).
        best_r2: Best R^2 at this generation for annotation.
    """
    unique_coords, weights, best_idx, n_unique = group_by_canonical(
        canonical_strings, pca_coords, fitnesses
    )

    # KDE density contours
    kde_result = compute_kde_density(unique_coords, weights, xlim, ylim)
    if kde_result is not None:
        xx, yy, density = kde_result
        cmap = LinearSegmentedColormap.from_list("custom", ["white", color])
        ax.contourf(
            xx,
            yy,
            density,
            levels=KDE_CONTOUR_LEVELS,
            cmap=cmap,
            alpha=0.7,
            zorder=1,
        )

    # Scatter unique classes
    if len(unique_coords) > 0:
        ax.scatter(
            unique_coords[:, 0],
            unique_coords[:, 1],
            c=color,
            s=PLOT_SETTINGS["scatter_size"],
            alpha=PLOT_SETTINGS["scatter_alpha"],
            edgecolors="white",
            linewidth=0.3,
            zorder=3,
            rasterized=True,
        )

    # Best individual star
    if len(pca_coords) > 0:
        ax.scatter(
            pca_coords[best_idx, 0],
            pca_coords[best_idx, 1],
            marker="*",
            s=80,
            color=STAR_COLOR,
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )

    # Annotation: delta and max R² in top-right
    delta = n_unique / n_total if n_total > 0 else 0.0
    ann_fs = int(PLOT_SETTINGS["annotation_fontsize"]) - 1
    r2_str = f"{best_r2:.2f}" if np.isfinite(best_r2) else "N/A"
    ax.text(
        0.97,
        0.97,
        f"$\\eta = {delta:.2f}$\n$R^2_{{\\max}} = {r2_str}$",
        transform=ax.transAxes,
        fontsize=ann_fs,
        va="top",
        ha="right",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


# ======================================================================
# Time-series panel
# ======================================================================


def draw_r2_timeseries(ax: Any, traj_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Draw R^2 time-series with mean and 95% CI bands.

    Combines trajectory data (dense, skips snapshot gens) with summary
    data (snapshot gens only) for a complete curve.

    Args:
        ax: Matplotlib axes.
        traj_df: Trajectory DataFrame (dense per-generation).
        summary_df: Summary DataFrame (snapshot generations).
    """
    # Merge trajectory and summary for complete coverage
    summary_subset = summary_df[["seed", "variant", "generation", "best_r2"]].copy()
    combined = pd.concat([traj_df[["seed", "variant", "generation", "best_r2"]], summary_subset])
    combined = combined.drop_duplicates(subset=["seed", "variant", "generation"], keep="last")

    for variant, color, label in [
        ("baseline", COLOR_BASELINE, "Baseline"),
        ("isalsr", COLOR_ISALSR, "IsalSR"),
    ]:
        sub = combined[combined["variant"] == variant].groupby("generation")["best_r2"]
        means = sub.mean()
        sems = sub.sem()
        gens = means.index.values
        ci_lo = means.values - 1.96 * sems.values
        ci_hi = means.values + 1.96 * sems.values

        ax.plot(
            gens,
            means.values,
            color=color,
            linewidth=float(PLOT_SETTINGS["line_width"]),
            label=label,
        )
        ax.fill_between(
            gens,
            ci_lo,
            ci_hi,
            color=color,
            alpha=float(PLOT_SETTINGS["error_band_alpha"]),
        )

    ax.set_xlabel("Generation $t$", fontsize=int(PLOT_SETTINGS["axes_labelsize"]))
    ax.set_ylabel(r"$R^2$ (train)", fontsize=int(PLOT_SETTINGS["axes_labelsize"]))
    ax.set_ylim(-0.1, 1.1)
    ax.tick_params(labelsize=int(PLOT_SETTINGS["tick_labelsize"]))
    ax.legend(fontsize=int(PLOT_SETTINGS["legend_fontsize"]), loc="lower right")


# ======================================================================
# Main figure
# ======================================================================


def generate_figure(
    input_dir: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    display_gens: list[int] | None = None,
) -> None:
    """Generate the PCA + KDE density landscape figure with R^2 panel.

    Args:
        input_dir: Path to diversity results directory.
        output_dir: Path for output files.
        seed: Representative seed index.
        display_gens: List of generation numbers to display.
    """
    apply_ieee_style()

    if display_gens is None:
        display_gens = DEFAULT_DISPLAY_GENS

    # Load trajectory data for R^2 time-series (dense per-generation curves)
    traj_df = load_all_trajectories(input_dir)
    # Load summary for per-snapshot R^2 (trajectory skips snapshot generations)
    summary_df = load_summary(input_dir)

    # Pre-compute joint PCA and collect data for all panels
    gen_data: dict[int, dict] = {}
    for gen in display_gens:
        try:
            b_coords, i_coords, _var_ratios = compute_joint_pca(input_dir, seed, gen)
            snap_b = load_snapshot(input_dir, seed, "baseline", gen)
            snap_i = load_snapshot(input_dir, seed, "isalsr", gen)
        except FileNotFoundError:
            log.warning("Missing snapshot for seed=%d gen=%d, skipping", seed, gen)
            continue

        # Shared axis limits for this generation (both variants)
        all_x = np.concatenate([b_coords[:, 0], i_coords[:, 0]])
        all_y = np.concatenate([b_coords[:, 1], i_coords[:, 1]])
        pad_x = max((all_x.max() - all_x.min()) * 0.08, 0.1)
        pad_y = max((all_y.max() - all_y.min()) * 0.08, 0.1)
        xlim = (float(all_x.min() - pad_x), float(all_x.max() + pad_x))
        ylim = (float(all_y.min() - pad_y), float(all_y.max() + pad_y))

        # Get best R^2 from summary CSV (snapshot generations)
        best_r2_b = float("-inf")
        best_r2_i = float("-inf")
        if not summary_df.empty:
            mask_b = (
                (summary_df["seed"] == seed)
                & (summary_df["variant"] == "baseline")
                & (summary_df["generation"] == gen)
            )
            mask_i = (
                (summary_df["seed"] == seed)
                & (summary_df["variant"] == "isalsr")
                & (summary_df["generation"] == gen)
            )
            if mask_b.any():
                best_r2_b = float(summary_df.loc[mask_b, "best_r2"].iloc[0])
            if mask_i.any():
                best_r2_i = float(summary_df.loc[mask_i, "best_r2"].iloc[0])

        gen_data[gen] = {
            "baseline": {
                "coords": b_coords,
                "cs": snap_b["canonical_strings"],
                "fit": snap_b["fitnesses"],
                "best_r2": best_r2_b,
                "n_total": len(b_coords),
            },
            "isalsr": {
                "coords": i_coords,
                "cs": snap_i["canonical_strings"],
                "fit": snap_i["fitnesses"],
                "best_r2": best_r2_i,
                "n_total": len(i_coords),
            },
            "xlim": xlim,
            "ylim": ylim,
        }

    available_gens = [g for g in display_gens if g in gen_data]
    if not available_gens:
        log.error("No valid generations found. Check input directory and seed.")
        return

    n_cols = len(available_gens)
    fig_width = float(PLOT_SETTINGS["figure_width_double"])
    fig_height = fig_width * 0.65

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        3,
        n_cols,
        figure=fig,
        height_ratios=[1, 1, 0.7],
        hspace=0.25,
        wspace=0.06,
        left=0.06,
        right=0.98,
        top=0.93,
        bottom=0.08,
    )

    variants = [
        ("baseline", COLOR_BASELINE),
        ("isalsr", COLOR_ISALSR),
    ]

    for col_idx, gen in enumerate(available_gens):
        data = gen_data[gen]
        for row_idx, (variant, color) in enumerate(variants):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            draw_pca_kde_panel(
                ax,
                data[variant]["coords"],
                data[variant]["cs"],
                data[variant]["fit"],
                color,
                data["xlim"],
                data["ylim"],
                data[variant]["n_total"],
                data[variant]["best_r2"],
            )
            # Column titles on top row
            if row_idx == 0:
                ax.set_title(
                    f"$t = {gen}$",
                    fontsize=int(PLOT_SETTINGS["tick_labelsize"]),
                )

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

    # Row 3: R^2 time-series spanning all columns
    ax_r2 = fig.add_subplot(gs[2, :])
    if not traj_df.empty or not summary_df.empty:
        draw_r2_timeseries(ax_r2, traj_df, summary_df)
    else:
        ax_r2.text(0.5, 0.5, "No trajectory data", transform=ax_r2.transAxes, ha="center")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = str(output_dir / "fig_pca_kde_landscape")
    saved = save_figure(fig, fig_path)
    plt.close(fig)
    log.info("Saved figure: %s", saved)

    caption = (
        "Population structure in PCA space across generations for seed 0 "
        "of the I.48.20 benchmark. Top row: Bingo baseline; middle row: "
        "IsalSR. Filled contours show weighted kernel density estimates of "
        "unique canonical classes (weight = number of isomorphic copies in "
        "the population). Green stars mark the individual with lowest MSE. "
        "Each panel annotates the effective diversity ratio "
        "$\\eta = |\\{[G] : G \\in P_t\\}| / N$ and the maximum $R^2$ "
        "at that generation. Bottom row: mean best $R^2$ across 20 seeds "
        "with 95\\% confidence intervals. The baseline population collapses "
        "to fewer unique classes as generations progress, while IsalSR "
        "maintains structural diversity through canonical deduplication "
        "without sacrificing regression quality."
    )
    caption_path = output_dir / "fig_pca_kde_landscape.caption.txt"
    caption_path.write_text(caption)
    log.info("Saved caption: %s", caption_path)


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PCA + KDE density landscape figure")
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
