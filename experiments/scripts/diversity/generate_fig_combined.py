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
    delta_mean: float | None = None,
    delta_std: float | None = None,
) -> None:
    """Draw one PCA panel with topographic KDE contours.

    Args:
        delta_mean: If provided, use this (aggregate) delta instead of
            computing from canonical_strings / n_total.
        delta_std: If provided, display as ± in the annotation.
    """
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
    delta = delta_mean if delta_mean is not None else (n_unique / n_total if n_total > 0 else 0.0)

    ann_fs = int(PLOT_SETTINGS["annotation_fontsize"]) - 1
    r2_str = f"{best_r2:.2f}" if np.isfinite(best_r2) else "N/A"

    if delta_std is not None and delta_std > 0.001:
        delta_str = f"$\\delta = {delta:.2f} \\pm {delta_std:.2f}$"
    else:
        delta_str = f"$\\delta = {delta:.2f}$"

    ax.text(
        0.97,
        0.97,
        f"{delta_str}\n$R^2_{{\\max}} = {r2_str}$",
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
    already_reordered: bool = False,
) -> Any:
    """Draw one clustered BP-GED heatmap panel."""
    reordered = dist_matrix if already_reordered else _cluster_reorder(dist_matrix)[0]
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


def _find_available_seeds(input_dir: Path) -> list[int]:
    """Return sorted list of seed indices that have both variants."""
    seeds = set()
    for d in input_dir.glob("seed_*"):
        if (d / "baseline").is_dir() and (d / "isalsr").is_dir():
            try:
                seeds.add(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(seeds)


def _find_median_delta_seed(summary_df: Any, seeds: list[int], last_gen: int) -> int:
    """Return the seed whose IsalSR delta at *last_gen* is closest to the median."""
    deltas: dict[int, float] = {}
    for s in seeds:
        mask = (
            (summary_df["seed"] == s)
            & (summary_df["variant"] == "isalsr")
            & (summary_df["generation"] == last_gen)
        )
        if mask.any():
            deltas[s] = float(summary_df.loc[mask, "delta"].iloc[0])
    if not deltas:
        return seeds[0]
    median_val = float(np.median(list(deltas.values())))
    return min(deltas, key=lambda s: abs(deltas[s] - median_val))


def _median_ged_matrices(matrices: list[np.ndarray]) -> np.ndarray:
    """Element-wise median of cluster-reordered GED matrices.

    Each seed's matrix is independently cluster-reordered BEFORE stacking,
    so the block structure (isomorphic clusters near the diagonal) aligns
    across seeds.  Without this, individuals at position (i,j) are
    unrelated across seeds and the median smears out the zero-distance
    blocks.
    """
    min_n = min(m.shape[0] for m in matrices)
    reordered = []
    for m in matrices:
        truncated = m[:min_n, :min_n]
        ro, _order = _cluster_reorder(truncated)
        reordered.append(ro)
    stacked = np.stack(reordered)
    return np.median(stacked, axis=0)


def _compute_pooled_pca(
    input_dir: Path,
    seeds: list[int],
    gen: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Pool WL features from ALL seeds and run one joint PCA.

    Returns:
        (b_coords, i_coords, b_cs, i_cs, b_fit, i_fit) or None.
        Coordinates are in a shared PCA space; canonical strings and
        fitnesses are concatenated across seeds.
    """
    from sklearn.decomposition import PCA

    all_feat_b: list[np.ndarray] = []
    all_feat_i: list[np.ndarray] = []
    all_cs_b: list[np.ndarray] = []
    all_cs_i: list[np.ndarray] = []
    all_fit_b: list[np.ndarray] = []
    all_fit_i: list[np.ndarray] = []

    for s in seeds:
        try:
            snap_b = load_snapshot(input_dir, s, "baseline", gen)
            snap_i = load_snapshot(input_dir, s, "isalsr", gen)
        except FileNotFoundError:
            continue
        all_feat_b.append(snap_b["wl_features"])
        all_feat_i.append(snap_i["wl_features"])
        all_cs_b.append(snap_b["canonical_strings"])
        all_cs_i.append(snap_i["canonical_strings"])
        all_fit_b.append(snap_b["fitnesses"])
        all_fit_i.append(snap_i["fitnesses"])

    if not all_feat_b:
        return None

    # Unify feature columns: pad each matrix to the global max width
    max_cols = max(
        max(f.shape[1] for f in all_feat_b),
        max(f.shape[1] for f in all_feat_i),
    )

    def _pad(f: np.ndarray) -> np.ndarray:
        if f.shape[1] < max_cols:
            return np.hstack([f, np.zeros((f.shape[0], max_cols - f.shape[1]))])
        return f

    feat_b = np.vstack([_pad(f) for f in all_feat_b])
    feat_i = np.vstack([_pad(f) for f in all_feat_i])

    combined = np.vstack([feat_b, feat_i])
    n_b = feat_b.shape[0]

    n_components = min(2, combined.shape[0], combined.shape[1])
    if n_components < 2:
        coords = np.zeros((combined.shape[0], 2))
    else:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(combined)

    b_coords = coords[:n_b]
    i_coords = coords[n_b:]

    cs_b = np.concatenate(all_cs_b)
    cs_i = np.concatenate(all_cs_i)
    fit_b = np.concatenate(all_fit_b)
    fit_i = np.concatenate(all_fit_i)

    return b_coords, i_coords, cs_b, cs_i, fit_b, fit_i


def _collect_gen_data_single(
    input_dir: Path,
    summary_df: Any,
    seed: int,
    gen: int,
) -> dict | None:
    """Collect PCA + GED + annotations for a single (seed, gen)."""
    try:
        b_coords, i_coords, _var = compute_joint_pca(input_dir, seed, gen)
        snap_b = load_snapshot(input_dir, seed, "baseline", gen)
        snap_i = load_snapshot(input_dir, seed, "isalsr", gen)
    except FileNotFoundError:
        return None

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

    return {
        "baseline": {
            "coords": b_coords,
            "cs": snap_b["canonical_strings"],
            "fit": snap_b["fitnesses"],
            "bp_ged": snap_b["bp_ged_distances"],
            "best_r2": best_r2_b,
            "n_total": len(b_coords),
        },
        "isalsr": {
            "coords": i_coords,
            "cs": snap_i["canonical_strings"],
            "fit": snap_i["fitnesses"],
            "bp_ged": snap_i["bp_ged_distances"],
            "best_r2": best_r2_i,
            "n_total": len(i_coords),
        },
    }


def _aggregate_annotations(
    summary_df: Any, seeds: list[int], gen: int
) -> dict[str, dict[str, float]]:
    """Compute mean ± std delta and mean R² across seeds for each variant."""
    result: dict[str, dict[str, float]] = {}
    for var in ("baseline", "isalsr"):
        mask = (
            (summary_df["variant"] == var)
            & (summary_df["generation"] == gen)
            & (summary_df["seed"].isin(seeds))
        )
        sub = summary_df.loc[mask]
        result[var] = {
            "delta_mean": float(sub["delta"].mean()) if len(sub) > 0 else 0.0,
            "delta_std": float(sub["delta"].std()) if len(sub) > 1 else 0.0,
            "best_r2_mean": float(sub["best_r2"].mean()) if len(sub) > 0 else float("-inf"),
        }
    return result


def generate_figure(
    input_dir: Path,
    output_dir: Path,
    seed: int | None = None,
    display_gens: list[int] | None = None,
    pool_pca: bool = True,
) -> None:
    """Generate the combined PCA + GED heatmap figure.

    Args:
        input_dir: Directory with per-seed snapshot data.
        output_dir: Directory for output figure files.
        seed: Specific seed index, or None for all-seed aggregation.
            When None (default), GED heatmaps show the element-wise
            median across all seeds, and annotations show mean +/- std.
        display_gens: Generation numbers to display as columns.
        pool_pca: If True (default) and seed is None, pool all seeds'
            individuals into one joint PCA.  If False, use the
            median-delta seed as representative for PCA.
    """
    apply_ieee_style()

    if display_gens is None:
        display_gens = DEFAULT_DISPLAY_GENS

    summary_df = load_summary(input_dir)
    aggregate_mode = seed is None

    if aggregate_mode:
        all_seeds = _find_available_seeds(input_dir)
        if not all_seeds:
            log.error("No seed directories found in %s", input_dir)
            return
        last_gen = max(display_gens)
        rep_seed = _find_median_delta_seed(summary_df, all_seeds, last_gen)
        log.info(
            "Aggregate mode: %d seeds, representative seed=%d (median delta at gen %d)",
            len(all_seeds),
            rep_seed,
            last_gen,
        )
    else:
        all_seeds = [seed]
        rep_seed = seed

    # Pre-compute all data
    gen_data: dict[int, dict] = {}
    global_ged_max = 0.0

    for gen in display_gens:
        # ---- PCA data ----
        if aggregate_mode and len(all_seeds) > 1 and pool_pca:
            # Pool WL features from ALL seeds into one joint PCA
            pca_result = _compute_pooled_pca(input_dir, all_seeds, gen)
            if pca_result is None:
                log.warning("No snapshots for gen=%d, skipping", gen)
                continue
            b_coords, i_coords, cs_b, cs_i, fit_b, fit_i = pca_result
            n_total_b = len(b_coords)
            n_total_i = len(i_coords)
        else:
            rep_data = _collect_gen_data_single(input_dir, summary_df, rep_seed, gen)
            if rep_data is None:
                log.warning("Missing snapshot seed=%d gen=%d, skipping", rep_seed, gen)
                continue
            b_coords = rep_data["baseline"]["coords"]
            i_coords = rep_data["isalsr"]["coords"]
            cs_b = rep_data["baseline"]["cs"]
            cs_i = rep_data["isalsr"]["cs"]
            fit_b = rep_data["baseline"]["fit"]
            fit_i = rep_data["isalsr"]["fit"]
            n_total_b = rep_data["baseline"]["n_total"]
            n_total_i = rep_data["isalsr"]["n_total"]

        # ---- GED data (median across seeds in aggregate mode) ----
        if aggregate_mode and len(all_seeds) > 1:
            ged_matrices: dict[str, list[np.ndarray]] = {"baseline": [], "isalsr": []}
            for s in all_seeds:
                try:
                    snap_b = load_snapshot(input_dir, s, "baseline", gen)
                    snap_i = load_snapshot(input_dir, s, "isalsr", gen)
                    ged_matrices["baseline"].append(snap_b["bp_ged_distances"])
                    ged_matrices["isalsr"].append(snap_i["bp_ged_distances"])
                except FileNotFoundError:
                    continue
            bp_b = (
                _median_ged_matrices(ged_matrices["baseline"])
                if ged_matrices["baseline"]
                else np.zeros((1, 1))
            )
            bp_i = (
                _median_ged_matrices(ged_matrices["isalsr"])
                if ged_matrices["isalsr"]
                else np.zeros((1, 1))
            )
            n_ged_seeds = len(ged_matrices["baseline"])
        else:
            snap_b_single = load_snapshot(input_dir, rep_seed, "baseline", gen)
            snap_i_single = load_snapshot(input_dir, rep_seed, "isalsr", gen)
            bp_b = snap_b_single["bp_ged_distances"]
            bp_i = snap_i_single["bp_ged_distances"]
            n_ged_seeds = 1

        for m in [bp_b, bp_i]:
            fv = m[np.isfinite(m)]
            if len(fv) > 0:
                global_ged_max = max(global_ged_max, float(fv.max()))

        # Shared PCA axis limits
        all_x = np.concatenate([b_coords[:, 0], i_coords[:, 0]])
        all_y = np.concatenate([b_coords[:, 1], i_coords[:, 1]])
        pad_x = max((all_x.max() - all_x.min()) * 0.08, 0.1)
        pad_y = max((all_y.max() - all_y.min()) * 0.08, 0.1)
        xlim = (float(all_x.min() - pad_x), float(all_x.max() + pad_x))
        ylim = (float(all_y.min() - pad_y), float(all_y.max() + pad_y))

        # Annotations: aggregate or single seed
        ann = _aggregate_annotations(summary_df, all_seeds if aggregate_mode else [seed], gen)

        gen_data[gen] = {
            "baseline": {
                "coords": b_coords,
                "cs": cs_b,
                "fit": fit_b,
                "bp_ged": bp_b,
                "best_r2": ann["baseline"]["best_r2_mean"],
                "n_total": n_total_b,
                "delta_mean": ann["baseline"]["delta_mean"],
                "delta_std": ann["baseline"]["delta_std"],
            },
            "isalsr": {
                "coords": i_coords,
                "cs": cs_i,
                "fit": fit_i,
                "bp_ged": bp_i,
                "best_r2": ann["isalsr"]["best_r2_mean"],
                "n_total": n_total_i,
                "delta_mean": ann["isalsr"]["delta_mean"],
                "delta_std": ann["isalsr"]["delta_std"],
            },
            "xlim": xlim,
            "ylim": ylim,
            "n_ged_seeds": n_ged_seeds,
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
                data[variant]["n_total"],
                data[variant]["best_r2"],
                delta_mean=data[variant].get("delta_mean"),
                delta_std=data[variant].get("delta_std"),
            )
            if pca_row == 0:
                ax_pca.set_title(
                    f"$t = {gen}$",
                    fontsize=int(PLOT_SETTINGS["tick_labelsize"]),
                )

            # Heatmap panel (aggregate GED is already cluster-reordered)
            ax_hm = fig.add_subplot(gs[heatmap_row, col_idx])
            im = _draw_heatmap_panel(
                ax_hm,
                data[variant]["bp_ged"],
                vmin,
                vmax,
                already_reordered=aggregate_mode and len(all_seeds) > 1,
            )

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
    suffix = "_all" if aggregate_mode else f"_seed{rep_seed}"
    fig_path = str(output_dir / f"fig_combined_diversity{suffix}")
    saved = save_figure(fig, fig_path)
    plt.close(fig)
    log.info("Saved figure: %s", saved)

    if aggregate_mode:
        n_seeds = len(all_seeds)
        n_ged = gen_data[available_gens[0]].get("n_ged_seeds", n_seeds)
        if pool_pca:
            pca_desc = (
                "the first row shows a joint PCA projection of all "
                f"{n_seeds} seeds pooled together (1-WL hash features)"
            )
        else:
            pca_desc = (
                "the first row shows the PCA projection from a "
                f"representative seed (seed {rep_seed}, median $\\delta$)"
            )
        caption = (
            "Combined population diversity analysis across generations, "
            f"aggregated over {n_seeds} independent seeds. "
            "The figure is organized in two blocks: Native DAG (top) and IsalSR (bottom). "
            f"Within each block, {pca_desc}, with kernel density contour lines "
            "(topographic style), and the second row shows the element-wise median "
            f"of the {n_ged} per-seed pairwise BP-GED matrices, each independently "
            "reordered by hierarchical clustering before aggregation. "
            "PCA panels annotate the mean effective diversity ratio "
            "$\\delta \\pm \\sigma$ and mean best $R^2$ across all seeds; "
            "green stars mark the fittest individual. "
            "The horizontal colorbar (bottom) applies to all heatmap panels. "
            "The baseline develops large uniform blocks in the median GED matrix "
            "(isomorphic clusters), while IsalSR maintains richer distance structure "
            "throughout."
        )
    else:
        caption = (
            "Combined population diversity analysis across generations "
            f"(seed {rep_seed}). "
            "The figure is organized in two blocks: Native DAG (top) and IsalSR (bottom). "
            "Within each block, the first row shows the PCA projection of the population "
            "with kernel density contour lines (topographic style), and the second row "
            "shows the pairwise bipartite GED matrix reordered by hierarchical clustering. "
            "PCA panels annotate the effective diversity ratio $\\delta$ and best $R^2$; "
            "green stars mark the fittest individual. The horizontal colorbar (bottom) "
            "applies to all heatmap panels. "
            "The baseline develops large uniform blocks in the GED matrix (isomorphic "
            "clusters), while IsalSR maintains richer distance structure throughout."
        )
    (output_dir / f"fig_combined_diversity{suffix}.caption.txt").write_text(caption)
    log.info("Saved caption")


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate combined PCA + GED heatmap figure")
    parser.add_argument("--input-dir", type=str, default=str(RESULTS_DIR / "diversity"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--seed",
        type=str,
        default="all",
        help="Seed index (integer) or 'all' for aggregate across all seeds (default: all)",
    )
    parser.add_argument(
        "--gens",
        type=str,
        default=",".join(str(g) for g in DEFAULT_DISPLAY_GENS),
        help="Comma-separated generation numbers to display",
    )
    parser.add_argument(
        "--no-pool-pca",
        action="store_true",
        help="In aggregate mode, use median-delta seed for PCA instead of pooling all seeds",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    display_gens = sorted(int(g) for g in args.gens.split(","))
    seed = None if args.seed.lower() == "all" else int(args.seed)

    generate_figure(input_dir, output_dir, seed, display_gens, pool_pca=not args.no_pool_pca)


if __name__ == "__main__":
    main()
