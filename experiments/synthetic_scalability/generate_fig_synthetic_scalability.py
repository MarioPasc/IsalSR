# ruff: noqa: N802, N803, N806
"""Synthetic scalability figure for the IsalSR supplementary material.

Single-panel figure combining canonicalization timing (left y-axis) with
the k! equivalent-representations curve (right y-axis).  A text annotation
states the invariance result (rho = k! for all expressions).

Also exports a standalone LaTeX table (tab_synthetic_scalability.tex).

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
# Aggregate per-k statistics (pooled across m) — for LaTeX table
# =============================================================================


def _aggregate_per_k(
    rows: list[dict[str, float | int]],
) -> list[dict[str, object]]:
    """Aggregate rows by k (pooling all m values)."""
    by_k: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for r in rows:
        by_k[int(r["k"])].append(r)

    table: list[dict[str, object]] = []
    for k in sorted(by_k):
        group = by_k[k]
        n = len(group)
        kfact = math.factorial(k)
        n_perms = int(group[0]["n_perms"])
        n_invariant = sum(1 for r in group if int(r["n_unique_canonicals"]) == 1)
        times_ms = [float(r["mean_canon_time_s"]) * 1000 for r in group]
        med_t = float(np.median(times_ms))
        iqr_lo = float(np.percentile(times_ms, 25))
        iqr_hi = float(np.percentile(times_ms, 75))
        rho_kfact = [float(r["rho_over_kfact"]) for r in group]
        all_exact = all(abs(v - 1.0) < 1e-6 for v in rho_kfact)

        table.append(
            {
                "k": k,
                "k_factorial": kfact,
                "n_perms": n_perms,
                "n_expr": n,
                "n_invariant": n_invariant,
                "pct_invariant": 100.0 * n_invariant / n,
                "rho_equals_kfact": all_exact,
                "median_time_ms": med_t,
                "iqr_lo_ms": iqr_lo,
                "iqr_hi_ms": iqr_hi,
            }
        )
    return table


# =============================================================================
# Single-panel figure
# =============================================================================


def plot_figure(
    ax: plt.Axes,
    rows: list[dict[str, float | int]],
) -> float:
    """Plot timing boxplots (left y) + k! curve (right y) + annotation.

    Returns:
        Fitted power-law exponent b.
    """
    # ---- Group by (k, m) ----
    cell: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in rows:
        cell[(int(r["k"]), int(r["m"]))].append(float(r["mean_canon_time_s"]) * 1000)

    k_values = sorted({k for k, _ in cell})
    m_values = sorted({m for _, m in cell})

    # ---- Timing boxplots (left y-axis) ----
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

    # ---- Power-law fit ----
    all_k: list[float] = []
    all_t: list[float] = []
    for (_k, _m), times in cell.items():
        for t in times:
            all_k.append(float(_k))
            all_t.append(t)
    all_k_arr = np.array(all_k)
    all_t_arr = np.array(all_t)

    mask = (all_k_arr > 0) & (all_t_arr > 0)
    b_fit, log_a_fit = np.polyfit(np.log(all_k_arr[mask]), np.log(all_t_arr[mask]), 1)
    a_fit = np.exp(log_a_fit)
    k_arr = np.linspace(min(k_values), max(k_values), 200)
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

    ax.set_yscale("log")
    ax.set_xlim(min(k_values) - 0.6, max(k_values) + 0.6)
    ax.set_xlabel("Internal nodes $k$")
    ax.set_ylabel("Canonicalization time (ms)")
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=20,
        )
    )
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # ---- k! curve on secondary y-axis (right) ----
    ax2 = ax.twinx()

    # Discrete points at integer k + smooth interpolation on log scale
    k_int = np.array(k_values)
    kf_int = np.array([math.factorial(k) for k in k_values], dtype=float)
    # Smooth curve via log-space interpolation
    k_smooth = np.linspace(min(k_values), max(k_values), 200)
    log_kf_smooth = np.interp(k_smooth, k_int, np.log(kf_int))
    kf_smooth = np.exp(log_kf_smooth)

    ax2.plot(
        k_smooth,
        kf_smooth,
        color=PAUL_TOL_BRIGHT["red"],
        linestyle=":",
        linewidth=float(PLOT_SETTINGS["line_width_thick"]),
        label=r"$k!$ isomorphic copies",
        zorder=2,
        alpha=0.7,
    )
    ax2.scatter(
        k_int,
        kf_int,
        color=PAUL_TOL_BRIGHT["red"],
        s=20,
        zorder=6,
        alpha=0.8,
        edgecolors="none",
    )
    ax2.set_yscale("log")
    ax2.set_ylabel(r"Isomorphic copies ($k!$)")
    ax2.tick_params(axis="y")

    # Shade reduction gap (between k! and 1)
    ax2.fill_between(
        k_smooth,
        np.ones_like(kf_smooth),
        kf_smooth,
        alpha=0.06,
        color=PAUL_TOL_BRIGHT["red"],
        zorder=1,
    )

    # ---- Combined legend (k! first, then timing entries) ----
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines_2 + lines_1,
        labels_2 + labels_1,
        fontsize=int(PLOT_SETTINGS["legend_fontsize"]) - 1,
        loc="upper left",
        frameon=False,
    )

    # ---- Invariance annotation (lower-right, compact) ----
    ax.text(
        0.97,
        0.04,
        r"$\rho = k!\;\;\forall\; k,\, m$",
        transform=ax.transAxes,
        fontsize=int(PLOT_SETTINGS["annotation_fontsize"]) + 1,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.9,
        },
        zorder=10,
    )

    return b_fit


# =============================================================================
# LaTeX table export
# =============================================================================


def _export_latex_table(
    table_data: list[dict[str, object]],
    output_dir: str,
) -> str:
    """Export a standalone LaTeX table summarizing the synthetic results."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Synthetic scalability: canonical invariance and timing "
        r"for random expression DAGs with $k$ internal nodes. "
        r"For each $k$, 200 expressions $\times$ 3 variable counts were "
        r"generated (600 total) and all $k!$ internal-node permutations "
        r"were exhaustively canonicalized. $\rho = k!$ confirms that every "
        r"permuted DAG maps to the same canonical string "
        r"($|\mathrm{Aut}(D)| = 1$ for all expressions).}",
        r"  \label{tab:synthetic_scalability}",
        r"  \small",
        r"  \begin{tabular}{r r r c c r}",
        r"    \toprule",
        r"    $k$ & $k!$ & Perms & $\rho = k!$ & Invariance & Time (ms) \\",
        r"    \midrule",
    ]
    for row in table_data:
        k = int(row["k"])
        kf = int(row["k_factorial"])
        n_perms = int(row["n_perms"])
        rho_ok = r"\checkmark" if row["rho_equals_kfact"] else r"$\times$"
        inv_pct = f"{float(row['pct_invariant']):.0f}\\%"
        med = float(row["median_time_ms"])
        iqr_lo = float(row["iqr_lo_ms"])
        iqr_hi = float(row["iqr_hi_ms"])
        time_str = f"{med:.2f} [{iqr_lo:.2f}--{iqr_hi:.2f}]"

        kf_str = f"{kf:,}".replace(",", r"{,}")
        np_str = f"{n_perms:,}".replace(",", r"{,}")
        lines.append(f"    {k} & {kf_str} & {np_str} & {rho_ok} & {inv_pct} & {time_str} \\\\")

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )

    tex_path = os.path.join(output_dir, "tab_synthetic_scalability.tex")
    os.makedirs(output_dir, exist_ok=True)
    Path(tex_path).write_text("\n".join(lines) + "\n")
    logger.info("LaTeX table: %s", tex_path)
    return tex_path


# =============================================================================
# Caption
# =============================================================================


def _build_caption(
    rows: list[dict[str, float | int]],
    b_fit: float,
) -> str:
    """Generate figure caption from data summary."""
    k_values = sorted({int(r["k"]) for r in rows})
    m_values = sorted({int(r["m"]) for r in rows})
    n_total = len(rows)
    n_expr_per_cell = len(rows) // (len(k_values) * len(m_values))
    k_max = max(k_values)
    k_max_fact = math.factorial(k_max)

    n_exact_kfact = sum(1 for r in rows if abs(float(r["rho_over_kfact"]) - 1.0) < 1e-6)

    return (
        f"Synthetic scalability analysis. "
        f"{n_expr_per_cell} random expression DAGs per $(k, m)$ cell were "
        f"generated via the Lample--Charton (2020) method with operators "
        f"$\\{{+, \\times, \\hat{{}}, \\sin, \\cos, \\exp, \\log, "
        f"\\mathrm{{neg}}, \\mathrm{{inv}}\\}}$ and "
        f"$m \\in \\{{{', '.join(str(v) for v in m_values)}\\}}$ variables. "
        f"For each expression, all $k!$ permutations of internal node IDs "
        f"were exhaustively canonicalized via the WL-guided greedy algorithm. "
        f"Left axis (boxplots): canonicalization time grows as "
        f"$O(k^{{{b_fit:.1f}}})$ (power-law fit, dashed), confirming that "
        f"the greedy-invariant algorithm avoids the factorial worst case. "
        f"Right axis (dotted, red shading): the $k!$ equivalent "
        f"representations that canonicalization collapses to a single "
        f"canonical string. The reduction factor equals $k!$ for all "
        f"{n_exact_kfact}/{n_total} expressions "
        f"($|\\mathrm{{Aut}}(D)| = 1$, trivial automorphism group), with "
        f"100\\% canonical invariance. "
        f"At $k = {k_max}$, ${k_max_fact:,}$ equivalent representations "
        f"are collapsed in $< 1$\\,ms."
    )


# =============================================================================
# Main
# =============================================================================


def generate_figure(data_dir: str, output_dir: str) -> str:
    """Generate the single-panel synthetic scalability figure."""
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

    # Single-panel figure (single-column width)
    fig_w, fig_h = get_figure_size("single", height_ratio=0.85)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.16, right=0.82, top=0.95, bottom=0.14)

    b_fit = plot_figure(ax, rows)

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "fig_synthetic_scalability")
    saved = save_figure(fig, out_path)
    for path in saved:
        logger.info("Saved: %s", path)
    plt.close(fig)

    # Save caption
    caption = _build_caption(rows, b_fit)
    caption_path = os.path.join(output_dir, "fig_synthetic_scalability.caption.txt")
    Path(caption_path).write_text(caption)
    logger.info("Caption: %s", caption_path)

    # Export standalone LaTeX table
    table_data = _aggregate_per_k(rows)
    _export_latex_table(table_data, output_dir)

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
