# ruff: noqa: N802, N803, N806
"""Search space analysis figure for the IsalSR arXiv paper.

Two-panel figure validating the central O(k!) search space reduction claim
via controlled permutation analysis.

Panel (a): Distinct representations per expression (log scale) vs k! reference.
Panel (b): Normalized reduction ratio (measured / theoretical k!).

Data source: Permutation analysis CSVs (perm_m*_k*.csv from Picasso or local).

Run:
    cd /home/mpascual/research/code/IsalSR && \\
    ~/.conda/envs/isalsr/bin/python experiments/scripts/generate_fig_search_space.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from glob import glob

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.plotting_styles import (  # noqa: E402
    PAUL_TOL_BRIGHT,
    PLOT_SETTINGS,
    apply_ieee_style,
    get_figure_size,
    save_figure,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =============================================================================
# Configuration
# =============================================================================

_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalsr/results/figures/arXiv_figures"

# Default data directories (Picasso results or local test data)
_PICASSO_DIR = (
    "/media/mpascual/Sandisk2TB/research/isalsr/results/"
    "arXiv_benchmarking/picasso/search_space_permutation"
)
_LOCAL_DIR = "/tmp/perm_test_data"

# Colors per num_variables (Paul Tol Bright, colorblind-safe)
M_COLORS: dict[int, str] = {
    1: PAUL_TOL_BRIGHT["blue"],  # #4477AA
    2: PAUL_TOL_BRIGHT["green"],  # #228833
}


# =============================================================================
# Data Loading
# =============================================================================


def load_permutation_data(data_dir: str) -> list[dict[str, float | int | str]]:
    """Load and concatenate all perm_m*_k*.csv files from data_dir.

    Returns:
        List of row dicts with numeric fields cast to appropriate types.
    """
    pattern = os.path.join(data_dir, "perm_m*_k*.csv")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files matching {pattern}")

    rows: list[dict[str, float | int | str]] = []
    for fpath in files:
        with open(fpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "k": int(row["k"]),
                        "m": int(row["m"]),
                        "dag_idx": int(row["dag_idx"]),
                        "source": row["source"],
                        "n_distinct_representations": int(row["n_distinct_representations"]),
                        "n_distinct_d2s": int(row["n_distinct_d2s"]),
                        "theoretical_k_factorial": int(row["theoretical_k_factorial"]),
                        "normalized_ratio": float(row["normalized_ratio"]),
                        "invariant_success_rate": float(row["invariant_success_rate"]),
                        "is_exhaustive": row["is_exhaustive"] in ("True", "1", "true"),
                        "n_perms_tested": int(row["n_perms_tested"]),
                    }
                )
    logger.info("Loaded %d rows from %d files in %s", len(rows), len(files), data_dir)
    return rows


# =============================================================================
# Panel (a): Distinct Representations per Expression
# =============================================================================


def _is_exhaustive_k(k: int) -> bool:
    """Return True if k! <= 100,000 (exhaustive enumeration threshold)."""
    return math.factorial(k) <= 100_000  # k <= 8


def plot_panel_a(
    ax: plt.Axes,
    rows: list[dict[str, float | int | str]],
) -> None:
    """Plot distinct representations per expression (log scale) with k! reference.

    Exhaustive runs (k<=8) shown in solid blue; sampled runs (k>8) shown in
    lighter cyan with a note that they are lower bounds.
    """
    # Group by k
    k_to_repr: dict[int, list[int]] = {}
    for row in rows:
        k = int(row["k"])
        n_repr = int(row["n_distinct_representations"])
        k_to_repr.setdefault(k, []).append(n_repr)

    k_values = sorted(k_to_repr.keys())
    if not k_values:
        return

    # Split into exhaustive and sampled k values
    k_exhaustive = [k for k in k_values if _is_exhaustive_k(k)]
    k_sampled = [k for k in k_values if not _is_exhaustive_k(k)]

    # k! reference line
    k_range = np.arange(min(k_values), max(k_values) + 1)
    k_factorial = np.array([math.factorial(int(kv)) for kv in k_range], dtype=float)
    ax.plot(
        k_range,
        k_factorial,
        color="0.3",
        linestyle="--",
        linewidth=PLOT_SETTINGS["line_width_thick"],
        label=r"$k!$ (theoretical)",
        zorder=5,
    )

    # Box plots — exhaustive (solid blue)
    if k_exhaustive:
        box_data_ex = [k_to_repr[k] for k in k_exhaustive]
        bp_ex = ax.boxplot(
            box_data_ex,
            positions=k_exhaustive,
            widths=0.5,
            patch_artist=True,
            showfliers=True,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
            medianprops={"color": "0.15", "linewidth": 1.2},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            boxprops={"linewidth": 0.8},
            zorder=3,
        )
        for patch in bp_ex["boxes"]:
            patch.set_facecolor(PAUL_TOL_BRIGHT["blue"])
            patch.set_alpha(0.6)
        # Legend proxy for exhaustive boxes
        bp_ex["boxes"][0].set_label("Exhaustive ($k \\leq 8$)")

    # Box plots — sampled (lighter cyan, lower bounds)
    if k_sampled:
        box_data_sm = [k_to_repr[k] for k in k_sampled]
        bp_sm = ax.boxplot(
            box_data_sm,
            positions=k_sampled,
            widths=0.5,
            patch_artist=True,
            showfliers=True,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
            medianprops={"color": "0.3", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.6, "color": "0.5"},
            capprops={"linewidth": 0.6, "color": "0.5"},
            boxprops={"linewidth": 0.6},
            zorder=3,
        )
        for patch in bp_sm["boxes"]:
            patch.set_facecolor(PAUL_TOL_BRIGHT["cyan"])
            patch.set_alpha(0.5)
        # Legend proxy for sampled boxes
        bp_sm["boxes"][0].set_label("Sampled ($k > 8$)")

    # Horizontal line at y=1 ("After canonicalization")
    ax.axhline(
        y=1,
        color=PAUL_TOL_BRIGHT["red"],
        linestyle="-",
        linewidth=PLOT_SETTINGS["line_width"],
        alpha=0.8,
        label="After canonicalization",
        zorder=4,
    )

    # Shade the reduction gap (between k! and y=1)
    ax.fill_between(
        k_range,
        np.ones_like(k_factorial),
        k_factorial,
        alpha=0.08,
        color=PAUL_TOL_BRIGHT["blue"],
        zorder=1,
    )

    ax.set_yscale("log")
    ax.set_ylim(0.5, k_factorial[-1] * 3)
    ax.set_xlabel("Internal nodes $k$")
    ax.set_ylabel("Distinct representations")
    ax.set_xticks(k_values)
    ax.legend(
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        loc="upper left",
        frameon=False,
    )

    # Panel label
    ax.text(
        -0.12,
        1.05,
        "(a)",
        transform=ax.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )


# =============================================================================
# Panel (b): Normalized Ratio
# =============================================================================


def plot_panel_b(
    ax: plt.Axes,
    rows: list[dict[str, float | int | str]],
) -> None:
    """Plot normalized ratio (n_distinct / k!) for exhaustive k values only.

    Sampled k values (k>8) are excluded from this panel because the ratio
    reflects sampling coverage, not automorphism structure. Their canonical
    invariance is noted in an annotation.
    """
    # Group by k — only exhaustive runs
    k_to_ratio: dict[int, list[float]] = {}
    n_sampled_dags = 0
    n_sampled_invariant = 0
    for row in rows:
        k = int(row["k"])
        if _is_exhaustive_k(k):
            ratio = float(row["normalized_ratio"])
            k_to_ratio.setdefault(k, []).append(ratio)
        else:
            n_sampled_dags += 1
            if float(row["invariant_success_rate"]) == 1.0:
                n_sampled_invariant += 1

    k_values = sorted(k_to_ratio.keys())
    if not k_values:
        return

    # Horizontal reference at 1.0
    ax.axhline(
        y=1.0,
        color="0.4",
        linestyle="--",
        linewidth=PLOT_SETTINGS["line_width"],
        zorder=1,
    )

    # Box plots
    box_data = [k_to_ratio[k] for k in k_values]
    bp = ax.boxplot(
        box_data,
        positions=k_values,
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        medianprops={"color": "0.15", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        boxprops={"linewidth": 0.8},
        zorder=3,
    )

    color = PAUL_TOL_BRIGHT["green"]
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points (jittered)
    rng = np.random.default_rng(42)
    for k in k_values:
        ratios = np.array(k_to_ratio[k])
        jitter = rng.uniform(-0.15, 0.15, size=len(ratios))
        ax.scatter(
            k + jitter,
            ratios,
            s=12,
            color=color,
            alpha=0.4,
            edgecolors="none",
            zorder=4,
        )

    ax.set_xlabel("Internal nodes $k$ (exhaustive)")
    ax.set_ylabel(r"Ratio $|\mathrm{distinct}| \;/\; k!$")
    ax.set_xticks(k_values)
    ax.set_ylim(-0.05, 1.15)

    # Panel label
    ax.text(
        -0.12,
        1.05,
        "(b)",
        transform=ax.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )


# =============================================================================
# Main Figure
# =============================================================================


def generate_search_space_figure(data_dir: str) -> str:
    """Generate the 2-panel search space analysis figure.

    Returns:
        Base output path (without extension).
    """
    apply_ieee_style()

    # Load data
    rows = load_permutation_data(data_dir)

    # Log summary
    k_values = sorted({int(r["k"]) for r in rows})
    n_total = len(rows)
    n_invariant = sum(1 for r in rows if float(r["invariant_success_rate"]) == 1.0)
    logger.info(
        "Data: %d DAGs across k=%s, invariant=100%%: %d/%d",
        n_total,
        k_values,
        n_invariant,
        n_total,
    )

    # Create figure
    fig_w, fig_h = get_figure_size("double", height_ratio=0.50)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    fig.subplots_adjust(wspace=0.35, left=0.09, right=0.96, top=0.90, bottom=0.16)

    # Plot panels
    plot_panel_a(ax_a, rows)
    plot_panel_b(ax_b, rows)

    # Save
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUTPUT_DIR, "fig_search_space")
    saved = save_figure(fig, out_path)
    for path in saved:
        logger.info("Saved: %s", path)
    plt.close(fig)

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate search space analysis figure")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with perm_m*_k*.csv files. Auto-detects Picasso or local.",
    )
    args = parser.parse_args()

    # Auto-detect data directory
    data_dir = args.data_dir
    if data_dir is None:
        if os.path.isdir(_PICASSO_DIR) and glob(os.path.join(_PICASSO_DIR, "perm_*.csv")):
            data_dir = _PICASSO_DIR
        elif os.path.isdir(_LOCAL_DIR) and glob(os.path.join(_LOCAL_DIR, "perm_*.csv")):
            data_dir = _LOCAL_DIR
        else:
            logger.error(
                "No data found. Run search_space_permutation_analysis.py first, "
                "or specify --data-dir."
            )
            raise SystemExit(1)

    generate_search_space_figure(data_dir)
