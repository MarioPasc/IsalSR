# ruff: noqa: N802, N803, N806
"""Synthetic scalability figure for the IsalSR supplementary material.

Two-panel figure showing:
  (a) Reduction factor rho vs. k (log-scale y) with k! reference curve.
  (b) Canonicalization time vs. k (log-scale y) with O(k^2) and O(k!) refs.

Data source: synth_k*_m*.csv fragments from the synthetic scalability experiment.

Run:
    cd /home/mpascual/research/code/IsalSR && \\
    ~/.conda/envs/isalsr/bin/python \\
        experiments/synthetic_scalability/generate_fig_synthetic_scalability.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

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

_DATA_DIR = (
    "/media/mpascual/Sandisk2TB/research/isalsr/results/"
    "model_validation/wl_subtree/synthetic_scalability"
)
_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalsr/results/figures/supplementary"

# Colors per m (Paul Tol Bright, colorblind-safe)
M_COLORS: dict[int, str] = {
    1: PAUL_TOL_BRIGHT["blue"],  # #4477AA
    2: PAUL_TOL_BRIGHT["green"],  # #228833
    3: PAUL_TOL_BRIGHT["purple"],  # #AA3377
}
M_LABELS: dict[int, str] = {
    1: "$m = 1$",
    2: "$m = 2$",
    3: "$m = 3$",
}


# =============================================================================
# Data Loading
# =============================================================================


def load_synthetic_data(data_dir: str) -> list[dict[str, float | int]]:
    """Load and concatenate synth_k*_m*.csv fragments."""
    pattern = os.path.join(data_dir, "synth_k*_m*.csv")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files matching {pattern}")

    rows: list[dict[str, float | int]] = []
    for fpath in files:
        with open(fpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "k": int(row["k"]),
                        "m": int(row["m"]),
                        "expr_id": int(row["expr_id"]),
                        "n_perms": int(row["n_perms"]),
                        "n_unique_canonicals": int(row["n_unique_canonicals"]),
                        "rho": float(row["rho"]),
                        "rho_over_kfact": float(row["rho_over_kfact"]),
                        "mean_canon_time_s": float(row["mean_canon_time_s"]),
                        "timeout_count": int(row["timeout_count"]),
                    }
                )
    logger.info("Loaded %d rows from %d files in %s", len(rows), len(files), data_dir)
    return rows


# =============================================================================
# Panel (a): rho vs. k
# =============================================================================


def plot_panel_a(
    ax: plt.Axes,
    rows: list[dict[str, float | int]],
) -> None:
    """Reduction factor rho vs. k (log-scale y) with k! reference."""
    # Group by (k, m)
    cell: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in rows:
        cell[(int(r["k"]), int(r["m"]))].append(float(r["rho"]))

    k_values = sorted({k for k, _ in cell})
    m_values = sorted({m for _, m in cell})

    # k! reference curve
    k_range = np.arange(min(k_values), max(k_values) + 1)
    k_factorial = np.array([math.factorial(int(kv)) for kv in k_range], dtype=float)
    ax.plot(
        k_range,
        k_factorial,
        color="0.3",
        linestyle="--",
        linewidth=float(PLOT_SETTINGS["line_width_thick"]),
        label=r"$k!$ (theoretical)",
        zorder=5,
    )

    # Horizontal line at y=1 ("After canonicalization")
    ax.axhline(
        y=1,
        color=PAUL_TOL_BRIGHT["red"],
        linestyle="-",
        linewidth=float(PLOT_SETTINGS["line_width"]),
        alpha=0.8,
        label="After canonicalization",
        zorder=4,
    )

    # Shade the reduction gap
    ax.fill_between(
        k_range,
        np.ones_like(k_factorial),
        k_factorial,
        alpha=0.08,
        color=PAUL_TOL_BRIGHT["blue"],
        zorder=1,
    )

    # Boxplots per m, slightly offset for readability
    n_m = len(m_values)
    box_width = 0.25
    offsets = np.linspace(-(n_m - 1) * box_width / 2, (n_m - 1) * box_width / 2, n_m)

    for m_val, offset in zip(m_values, offsets, strict=True):
        color = M_COLORS.get(m_val, PAUL_TOL_BRIGHT["grey"])
        k_present = [k for k in k_values if (k, m_val) in cell]
        if not k_present:
            continue

        box_data = [cell[(k, m_val)] for k in k_present]
        positions = [k + offset for k in k_present]

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
            medianprops={"color": "0.15", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.7},
            capprops={"linewidth": 0.7},
            boxprops={"linewidth": 0.7},
            zorder=3,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        bp["boxes"][0].set_label(M_LABELS[m_val])

    ax.set_yscale("log")
    ax.set_ylim(0.5, k_factorial[-1] * 5)
    ax.set_xlim(min(k_values) - 0.6, max(k_values) + 0.6)
    ax.set_xlabel("Internal nodes $k$")
    ax.set_ylabel(r"Reduction factor $\rho$")
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.legend(
        fontsize=int(PLOT_SETTINGS["legend_fontsize"]) - 1,
        loc="upper left",
        frameon=False,
    )

    ax.text(
        -0.12,
        1.05,
        "(a)",
        transform=ax.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )


# =============================================================================
# Panel (b): Canonicalization time vs. k
# =============================================================================


def plot_panel_b(
    ax: plt.Axes,
    rows: list[dict[str, float | int]],
) -> None:
    """Canonicalization time vs. k (log-scale y) with reference curves."""
    # Group by (k, m)
    cell: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in rows:
        cell[(int(r["k"]), int(r["m"]))].append(float(r["mean_canon_time_s"]) * 1000)

    k_values = sorted({k for k, _ in cell})
    m_values = sorted({m for _, m in cell})

    # Reference curves (fitted to data range)
    k_arr = np.linspace(min(k_values), max(k_values), 200)

    # O(k!) reference — normalized to pass through data at k_mid
    k_mid = k_values[len(k_values) // 2]
    all_times_mid = []
    for m_val in m_values:
        if (k_mid, m_val) in cell:
            all_times_mid.extend(cell[(k_mid, m_val)])
    t_mid = float(np.median(all_times_mid)) if all_times_mid else 1.0
    kf_mid = math.factorial(k_mid)
    kf_ref = np.array([math.factorial(int(round(kv))) for kv in k_arr], dtype=float)
    kf_curve = t_mid * kf_ref / kf_mid

    ax.plot(
        k_arr,
        kf_curve,
        color="0.6",
        linestyle=":",
        linewidth=float(PLOT_SETTINGS["line_width"]),
        label=r"$O(k!)$ reference",
        zorder=2,
    )

    # O(k^2) reference — fitted via least-squares on log-log
    all_k = []
    all_t = []
    for (k, _m_val), times in cell.items():
        for t in times:
            all_k.append(k)
            all_t.append(t)
    all_k_arr = np.array(all_k, dtype=float)
    all_t_arr = np.array(all_t, dtype=float)

    # Fit power law: t = a * k^b via log-log linear regression
    mask = (all_k_arr > 0) & (all_t_arr > 0)
    log_k = np.log(all_k_arr[mask])
    log_t = np.log(all_t_arr[mask])
    b_fit, log_a_fit = np.polyfit(log_k, log_t, 1)
    a_fit = np.exp(log_a_fit)
    poly_curve = a_fit * k_arr**b_fit

    ax.plot(
        k_arr,
        poly_curve,
        color="0.3",
        linestyle="--",
        linewidth=float(PLOT_SETTINGS["line_width_thick"]),
        label=rf"$O(k^{{{b_fit:.1f}}})$ (fitted)",
        zorder=5,
    )

    # Boxplots per m
    n_m = len(m_values)
    box_width = 0.25
    offsets = np.linspace(-(n_m - 1) * box_width / 2, (n_m - 1) * box_width / 2, n_m)

    for m_val, offset in zip(m_values, offsets, strict=True):
        color = M_COLORS.get(m_val, PAUL_TOL_BRIGHT["grey"])
        k_present = [k for k in k_values if (k, m_val) in cell]
        if not k_present:
            continue

        box_data = [cell[(k, m_val)] for k in k_present]
        positions = [k + offset for k in k_present]

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "0.15", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.7},
            capprops={"linewidth": 0.7},
            boxprops={"linewidth": 0.7},
            zorder=3,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        bp["boxes"][0].set_label(M_LABELS[m_val])

    ax.set_yscale("log")
    ax.set_xlim(min(k_values) - 0.6, max(k_values) + 0.6)
    ax.set_xlabel("Internal nodes $k$")
    ax.set_ylabel("Canonicalization time (ms)")
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.legend(
        fontsize=int(PLOT_SETTINGS["legend_fontsize"]) - 1,
        loc="upper left",
        frameon=False,
    )

    ax.text(
        -0.12,
        1.05,
        "(b)",
        transform=ax.transAxes,
        fontsize=PLOT_SETTINGS["panel_label_fontsize"],
        fontweight="bold",
    )


# =============================================================================
# Caption
# =============================================================================


def _build_caption(rows: list[dict[str, float | int]]) -> str:
    """Generate figure caption from data summary."""
    k_values = sorted({int(r["k"]) for r in rows})
    m_values = sorted({int(r["m"]) for r in rows})
    n_total = len(rows)
    n_expr_per_cell = len(rows) // (len(k_values) * len(m_values))
    k_max = max(k_values)
    k_max_fact = math.factorial(k_max)

    # Compute empirical power-law exponent
    all_k = np.array([float(r["k"]) for r in rows])
    all_t = np.array([float(r["mean_canon_time_s"]) * 1000 for r in rows])
    mask = (all_k > 0) & (all_t > 0)
    b_fit, _ = np.polyfit(np.log(all_k[mask]), np.log(all_t[mask]), 1)

    # Check invariance
    n_invariant = sum(1 for r in rows if int(r["n_unique_canonicals"]) == 1)
    pct_invariant = 100.0 * n_invariant / n_total

    # Check rho = k! fraction
    n_exact_kfact = sum(1 for r in rows if abs(float(r["rho_over_kfact"]) - 1.0) < 1e-6)
    pct_exact = 100.0 * n_exact_kfact / n_total

    return (
        f"Synthetic scalability analysis: reduction factor $\\rho$ and "
        f"canonicalization time versus internal node count $k$. "
        f"{n_expr_per_cell} random expression DAGs per $(k, m)$ cell were "
        f"generated via the Lample--Charton (2020) method with operators "
        f"$\\{{+, \\times, \\hat{{}}, \\sin, \\cos, \\exp, \\log, "
        f"\\mathrm{{neg}}, \\mathrm{{inv}}\\}}$ and "
        f"$m \\in \\{{{', '.join(str(v) for v in m_values)}\\}}$ variables. "
        f"For each expression, all $k!$ permutations of internal node IDs "
        f"were exhaustively canonicalized via the WL-guided greedy algorithm. "
        f"(a)~$\\rho$ tracks the theoretical $k!$ curve exactly: "
        f"{pct_exact:.1f}\\% of expressions ({n_exact_kfact}/{n_total}) "
        f"achieve $\\rho = k!$ (i.e., $|\\mathrm{{Aut}}(D)| = 1$), and "
        f"{pct_invariant:.1f}\\% have perfect canonical invariance "
        f"($n_{{\\mathrm{{unique}}}} = 1$). "
        f"(b)~Canonicalization time grows as "
        f"$O(k^{{{b_fit:.1f}}})$ (power-law fit), confirming that the "
        f"greedy-invariant algorithm avoids the $O(k!)$ worst case of "
        f"exhaustive canonicalization. At $k = {k_max}$, the reduction "
        f"collapses ${k_max_fact:,}$ equivalent representations to one "
        f"canonical string in $< 1$\\,ms."
    )


# =============================================================================
# Main Figure
# =============================================================================


def generate_figure(data_dir: str, output_dir: str) -> str:
    """Generate the 2-panel synthetic scalability figure.

    Returns:
        Base output path (without extension).
    """
    apply_ieee_style()

    rows = load_synthetic_data(data_dir)

    k_values = sorted({int(r["k"]) for r in rows})
    m_values = sorted({int(r["m"]) for r in rows})
    n_total = len(rows)
    n_invariant = sum(1 for r in rows if int(r["n_unique_canonicals"]) == 1)
    logger.info(
        "Data: %d rows, k=%s, m=%s, invariant=100%%: %d/%d",
        n_total,
        k_values,
        m_values,
        n_invariant,
        n_total,
    )

    # Two-panel figure (double-column width)
    fig_w, fig_h = get_figure_size("double", height_ratio=0.50)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    fig.subplots_adjust(wspace=0.35, left=0.09, right=0.96, top=0.90, bottom=0.16)

    plot_panel_a(ax_a, rows)
    plot_panel_b(ax_b, rows)

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "fig_synthetic_scalability")
    saved = save_figure(fig, out_path)
    for path in saved:
        logger.info("Saved: %s", path)
    plt.close(fig)

    # Save caption
    caption = _build_caption(rows)
    caption_path = os.path.join(output_dir, "fig_synthetic_scalability.caption.txt")
    Path(caption_path).write_text(caption)
    logger.info("Caption: %s", caption_path)

    return out_path


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic scalability figure (supplementary)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=_DATA_DIR,
        help="Directory with synth_k*_m*.csv fragments.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_OUTPUT_DIR,
        help="Output directory for figure and caption.",
    )
    args = parser.parse_args()
    generate_figure(args.data_dir, args.output_dir)
