"""Generate combined PCA + GED heatmap figure for diversity analysis.

4 rows grouped in 2 blocks: each block has PCA (top) + heatmap (bottom).
Block 1 = Baseline, Block 2 = IsalSR.
Horizontal colorbar spans the full figure width.

Usage:
    python -m experiments.scripts.diversity.generate_fig_combined \
        --input-dir /path/to/diversity/I.10.7 \
        --output-dir /path/to/diversity/I.10.7/figures
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde

from experiments.plotting_styles import (
    COLOR_ISALSR,
    COLOR_NATIVE,
    PAUL_TOL_BRIGHT,
    PLOT_SETTINGS,
    apply_ieee_style,
    save_figure,
)
from experiments.scripts.diversity.generate_fig_diversity import (
    RESULTS_DIR,
    compute_joint_pca,
    load_snapshot,
    load_summary,
)

log = logging.getLogger(__name__)

# ======================================================================
# Shared constants
# ======================================================================

DEFAULT_DISPLAY_GENS: list[int] = [0, 25, 100, 300, 500]
DEFAULT_SEED: int = 0

COLOR_BASELINE: str = COLOR_NATIVE
STAR_COLOR: str = PAUL_TOL_BRIGHT["green"]

KDE_GRID_RESOLUTION: int = 120
KDE_CONTOUR_LEVELS: int = 12
KDE_BANDWIDTH_FACTOR: float = 0.3
HEATMAP_CMAP: str = "inferno"
LINKAGE_METHOD: str = "average"


# ======================================================================
# Data helpers (reused from existing scripts)
# ======================================================================


def _group_by_canonical(
    canonical_strings: np.ndarray,
    pca_coords: np.ndarray,
    fitnesses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Group by canonical string. Returns (unique_coords, weights, best_idx, n_unique)."""
    counts = Counter(str(s) for s in canonical_strings)
    n_unique = len(counts)
    if n_unique == 0:
        return np.empty((0, 2)), np.empty(0), 0, 0

    seen: dict[str, int] = {}
    for i, cs in enumerate(canonical_strings):
        key = str(cs)
        if key not in seen:
            seen[key] = i

    rep_indices = list(seen.values())
    unique_coords = pca_coords[rep_indices]
    weights = np.array([counts[str(canonical_strings[i])] for i in rep_indices], dtype=np.float64)

    finite_mask = np.isfinite(fitnesses)
    best_idx = int(np.argmin(np.where(finite_mask, fitnesses, np.inf))) if finite_mask.any() else 0
    return unique_coords, weights, best_idx, n_unique


def _compute_kde(
    coords: np.ndarray,
    weights: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute weighted KDE. Returns (xx, yy, density) or None."""
    if len(coords) < 2:
        return None
    if np.allclose(coords.std(axis=0), 0, atol=1e-10):
        return None
    try:
        kde = gaussian_kde(coords.T, weights=weights, bw_method=KDE_BANDWIDTH_FACTOR)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(42)
        jittered = coords + rng.normal(0, 1e-6, coords.shape)
        try:
            kde = gaussian_kde(jittered.T, weights=weights, bw_method=KDE_BANDWIDTH_FACTOR)
        except np.linalg.LinAlgError:
            return None
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], KDE_GRID_RESOLUTION),
        np.linspace(ylim[0], ylim[1], KDE_GRID_RESOLUTION),
    )
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(positions).reshape(xx.shape)
    return xx, yy, density


def _cluster_reorder(dist_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorder distance matrix by hierarchical clustering."""
    mat = dist_matrix.copy().astype(np.float64)
    np.fill_diagonal(mat, 0.0)
    mat = (mat + mat.T) / 2.0
    finite_max = np.nanmax(mat[np.isfinite(mat)]) if np.isfinite(mat).any() else 1.0
    mat = np.where(np.isfinite(mat), mat, finite_max)
    condensed = squareform(mat, checks=False)
    z = linkage(condensed, method=LINKAGE_METHOD)
    order = leaves_list(z)
    return mat[np.ix_(order, order)], order


# ======================================================================
# Panel drawing
# ======================================================================


def _draw_pca_panel(
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
    """Draw one PCA panel with topographic KDE contours."""
    unique_coords, weights, best_idx, n_unique = _group_by_canonical(
        canonical_strings, pca_coords, fitnesses
    )

    kde_result = _compute_kde(unique_coords, weights, xlim, ylim)
    if kde_result is not None:
        xx, yy, density = kde_result

        # Filled contours: white → variant color with progressive opacity
        cmap = LinearSegmentedColormap.from_list("fill", ["white", color])
        ax.contourf(
            xx,
            yy,
            density,
            levels=KDE_CONTOUR_LEVELS,
            cmap=cmap,
            alpha=0.6,
            zorder=1,
        )
        # Contour lines: thin black for topographic readability
        ax.contour(
            xx,
            yy,
            density,
            levels=KDE_CONTOUR_LEVELS,
            colors="black",
            linewidths=0.3,
            alpha=0.5,
            zorder=2,
        )

    # Scatter unique classes
    if len(unique_coords) > 0:
        ax.scatter(
            unique_coords[:, 0],
            unique_coords[:, 1],
            c=color,
            s=int(PLOT_SETTINGS["scatter_size"]),
            alpha=float(PLOT_SETTINGS["scatter_alpha"]),
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

    # Annotation
    delta = n_unique / n_total if n_total > 0 else 0.0
    ann_fs = int(PLOT_SETTINGS["annotation_fontsize"]) - 1
    r2_str = f"{best_r2:.2f}" if np.isfinite(best_r2) else "N/A"
    ax.text(
        0.97,
        0.97,
        f"$\\delta = {delta:.2f}$\n$R^2_{{\\max}} = {r2_str}$",
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


def _draw_heatmap_panel(
    ax: Any,
    dist_matrix: np.ndarray,
    vmin: float,
    vmax: float,
) -> Any:
    """Draw one clustered BP-GED heatmap panel."""
    reordered, _order = _cluster_reorder(dist_matrix)
    im = ax.imshow(
        reordered,
        cmap=HEATMAP_CMAP,
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
# Main figure
# ======================================================================


def generate_figure(
    input_dir: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    display_gens: list[int] | None = None,
) -> None:
    """Generate the combined PCA + GED heatmap figure."""
    apply_ieee_style()

    if display_gens is None:
        display_gens = DEFAULT_DISPLAY_GENS

    summary_df = load_summary(input_dir)

    # Pre-compute all data
    gen_data: dict[int, dict] = {}
    global_ged_max = 0.0

    for gen in display_gens:
        try:
            b_coords, i_coords, _var = compute_joint_pca(input_dir, seed, gen)
            snap_b = load_snapshot(input_dir, seed, "baseline", gen)
            snap_i = load_snapshot(input_dir, seed, "isalsr", gen)
        except FileNotFoundError:
            log.warning("Missing snapshot seed=%d gen=%d, skipping", seed, gen)
            continue

        # Shared PCA axis limits
        all_x = np.concatenate([b_coords[:, 0], i_coords[:, 0]])
        all_y = np.concatenate([b_coords[:, 1], i_coords[:, 1]])
        pad_x = max((all_x.max() - all_x.min()) * 0.08, 0.1)
        pad_y = max((all_y.max() - all_y.min()) * 0.08, 0.1)
        xlim = (float(all_x.min() - pad_x), float(all_x.max() + pad_x))
        ylim = (float(all_y.min() - pad_y), float(all_y.max() + pad_y))

        # R² from summary
        best_r2_b = float("-inf")
        best_r2_i = float("-inf")
        if not summary_df.empty:
            for var in ("baseline", "isalsr"):
                mask = (
                    (summary_df["seed"] == seed)
                    & (summary_df["variant"] == var)
                    & (summary_df["generation"] == gen)
                )
                if mask.any():
                    val = float(summary_df.loc[mask, "best_r2"].iloc[0])
                    if var == "baseline":
                        best_r2_b = val
                    else:
                        best_r2_i = val

        # Track global GED max
        bp_b = snap_b["bp_ged_distances"]
        bp_i = snap_i["bp_ged_distances"]
        for m in [bp_b, bp_i]:
            fv = m[np.isfinite(m)]
            if len(fv) > 0:
                global_ged_max = max(global_ged_max, float(fv.max()))

        gen_data[gen] = {
            "baseline": {
                "coords": b_coords,
                "cs": snap_b["canonical_strings"],
                "fit": snap_b["fitnesses"],
                "bp_ged": bp_b,
                "best_r2": best_r2_b,
            },
            "isalsr": {
                "coords": i_coords,
                "cs": snap_i["canonical_strings"],
                "fit": snap_i["fitnesses"],
                "bp_ged": bp_i,
                "best_r2": best_r2_i,
            },
            "xlim": xlim,
            "ylim": ylim,
            "n_total": len(b_coords),
        }

    available_gens = [g for g in display_gens if g in gen_data]
    if not available_gens:
        log.error("No valid generations found.")
        return

    vmin, vmax = 0.0, max(global_ged_max, 1.0)
    n_cols = len(available_gens)
    fig_width = float(PLOT_SETTINGS["figure_width_double"])

    # Layout: 6 rows — PCA, Heatmap, spacer, PCA, Heatmap, colorbar
    # The spacer row (row 2) creates the visual gap between groups.
    fig = plt.figure(figsize=(fig_width, fig_width * 0.82))
    gs = GridSpec(
        6,
        n_cols,
        figure=fig,
        height_ratios=[1, 0.85, 0.12, 1, 0.85, 0.07],
        hspace=0.04,
        wspace=0.06,
        left=0.07,
        right=0.98,
        top=0.94,
        bottom=0.05,
    )

    # Row mapping: (pca_row, heatmap_row) per variant
    variants = [
        ("baseline", COLOR_BASELINE, 0, 1),  # rows 0-1
        ("isalsr", COLOR_ISALSR, 3, 4),  # rows 3-4 (skip spacer row 2)
    ]

    im = None
    for variant, color, pca_row, heatmap_row in variants:
        for col_idx, gen in enumerate(available_gens):
            data = gen_data[gen]

            # PCA panel
            ax_pca = fig.add_subplot(gs[pca_row, col_idx])
            _draw_pca_panel(
                ax_pca,
                data[variant]["coords"],
                data[variant]["cs"],
                data[variant]["fit"],
                color,
                data["xlim"],
                data["ylim"],
                data["n_total"],
                data[variant]["best_r2"],
            )
            if pca_row == 0:
                ax_pca.set_title(
                    f"$t = {gen}$",
                    fontsize=int(PLOT_SETTINGS["tick_labelsize"]),
                )

            # Heatmap panel
            ax_hm = fig.add_subplot(gs[heatmap_row, col_idx])
            im = _draw_heatmap_panel(ax_hm, data[variant]["bp_ged"], vmin, vmax)

    # Row labels (black text, no bold, professional)
    label_fs = int(PLOT_SETTINGS["axes_titlesize"])
    fig.text(
        0.02,
        0.79,
        "Native DAG",
        rotation=90,
        fontsize=label_fs,
        va="center",
        ha="center",
    )
    fig.text(
        0.02,
        0.37,
        "IsalSR",
        rotation=90,
        fontsize=label_fs,
        va="center",
        ha="center",
    )

    # Horizontal colorbar spanning full width (row 5)
    if im is not None:
        cbar_ax = fig.add_subplot(gs[5, :])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("BP-GED", fontsize=int(PLOT_SETTINGS["tick_labelsize"]))
        cbar.ax.tick_params(labelsize=int(PLOT_SETTINGS["annotation_fontsize"]) - 1)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = str(output_dir / "fig_combined_diversity")
    saved = save_figure(fig, fig_path)
    plt.close(fig)
    log.info("Saved figure: %s", saved)

    caption = (
        "Combined population diversity analysis across generations. "
        "The figure is organized in two blocks: Baseline (top) and IsalSR (bottom). "
        "Within each block, the first row shows the PCA projection of the population "
        "with kernel density contour lines (topographic style), and the second row "
        "shows the pairwise bipartite GED matrix reordered by hierarchical clustering. "
        "PCA panels annotate the effective diversity ratio $\\delta$ and best $R^2$; "
        "green stars mark the fittest individual. The horizontal colorbar (bottom) "
        "applies to all heatmap panels. "
        "The baseline develops large uniform blocks in the GED matrix (isomorphic "
        "clusters), while IsalSR maintains richer distance structure throughout."
    )
    (output_dir / "fig_combined_diversity.caption.txt").write_text(caption)
    log.info("Saved caption")


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate combined PCA + GED heatmap figure")
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
