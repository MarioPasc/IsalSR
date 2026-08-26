"""Reduction factor vs overhead Pareto plot, aggregated by internal node count k.

Each point is the mean +/- std of all runs sharing the same (method, k).
A staircase Pareto frontier per method shows the trade-off envelope:
for any given rho, the minimum overhead achievable.

Usage:
    python -m experiments.figures.models.generate_rf_vs_overhead \
        --results-dir /path/to/wl_subtree \
        --output-dir /path/to/figures
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.plotting_styles import (  # noqa: E402
    METHOD_COLORS,
    PLOT_SETTINGS,
    apply_ieee_style,
    save_figure,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

METHOD_MARKERS = {
    "udfs": "o",
    "bingo": "D",
}


# ======================================================================
# Data loading
# ======================================================================


def _load_isalsr_runs(results_dir: Path) -> list[dict]:
    """Load all isalsr run_log.json files, extracting key fields."""
    rows: list[dict] = []
    for rl_path in sorted(results_dir.rglob("isalsr/seed_*/run_log.json")):
        parts = rl_path.relative_to(results_dir).parts
        method = parts[0]
        try:
            with open(rl_path) as f:
                rl = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        ss = rl.get("results", {}).get("search_space", {})
        t = rl.get("results", {}).get("time", {})
        rf = ss.get("empirical_reduction_factor", 1.0)
        k = ss.get("max_internal_nodes_seen", 0)
        overhead = t.get("overhead_time_s", 0.0)
        if rf > 0 and np.isfinite(rf) and k > 0 and overhead > 0:
            rows.append({"method": method, "k": k, "rf": rf, "overhead_s": overhead})
    return rows


def _remove_iqr_outliers(values: list[float]) -> list[float]:
    """Remove outliers using 1.5x IQR rule. Returns filtered list."""
    if len(values) < 4:
        return values
    arr = np.array(values)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    filtered = arr[(arr >= lo) & (arr <= hi)]
    return filtered.tolist() if len(filtered) > 0 else values


def _aggregate_by_k(
    rows: list[dict], method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate runs by k for one method.

    Overhead outliers (1.5x IQR) are removed per bucket before
    computing mean and std to reduce the impact of wall-clock-limited runs.

    Returns (ks, rf_mean, rf_std, oh_mean, oh_std) sorted by k.
    """
    buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"rf": [], "oh": []})
    for r in rows:
        if r["method"] == method:
            buckets[r["k"]]["rf"].append(r["rf"])
            buckets[r["k"]]["oh"].append(r["overhead_s"])

    ks_sorted = sorted(buckets.keys())
    ks = np.array(ks_sorted)
    rf_mean = np.array([np.mean(buckets[k]["rf"]) for k in ks_sorted])
    rf_std = np.array([np.std(buckets[k]["rf"]) for k in ks_sorted])
    # Remove overhead outliers per k-bucket
    oh_clean = [_remove_iqr_outliers(buckets[k]["oh"]) for k in ks_sorted]
    oh_mean = np.array([np.mean(v) for v in oh_clean])
    oh_std = np.array([np.std(v) for v in oh_clean])
    return ks, rf_mean, rf_std, oh_mean, oh_std


# ======================================================================
# Pareto frontier (staircase)
# ======================================================================


def _pareto_membership(rf: np.ndarray, overhead: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points (max rf, min overhead)."""
    order = np.argsort(rf)
    n = len(rf)
    is_pareto_sorted = np.zeros(n, dtype=bool)
    oh_sorted = overhead[order]
    min_oh = float("inf")
    for i in range(n - 1, -1, -1):
        if oh_sorted[i] <= min_oh:
            is_pareto_sorted[i] = True
            min_oh = oh_sorted[i]
    # Map back to original indices
    is_pareto = np.zeros(n, dtype=bool)
    is_pareto[order] = is_pareto_sorted
    return is_pareto


def _pareto_staircase(
    rf: np.ndarray, overhead: np.ndarray, x_left: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a staircase Pareto frontier.

    Returns (xs, ys) arrays for a step-function plot.
    The staircase starts at (x_left, overhead_of_first_pareto_point)
    and proceeds right with horizontal → vertical steps.
    """
    is_pareto = _pareto_membership(rf, overhead)
    p_rf = rf[is_pareto]
    p_oh = overhead[is_pareto]

    # Sort Pareto points by rf ascending
    order = np.argsort(p_rf)
    p_rf = p_rf[order]
    p_oh = p_oh[order]

    # Build staircase path
    xs = [x_left]
    ys = [p_oh[0]]
    for i in range(len(p_rf)):
        xs.append(p_rf[i])
        ys.append(p_oh[i])
        if i + 1 < len(p_rf):
            xs.append(p_rf[i])
            ys.append(p_oh[i + 1])

    return np.array(xs), np.array(ys)


# ======================================================================
# Figure
# ======================================================================


def generate_figure(results_dir: Path, output_dir: Path) -> None:
    """Generate the aggregated RF vs overhead Pareto plot."""
    apply_ieee_style()

    rows = _load_isalsr_runs(results_dir)
    if not rows:
        log.error("No data found in %s", results_dir)
        return
    log.info("Loaded %d IsalSR runs", len(rows))

    fig_width = float(PLOT_SETTINGS["figure_width_single"])
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.85))

    tick_fs = int(PLOT_SETTINGS["tick_labelsize"])
    label_fs = int(PLOT_SETTINGS["axes_labelsize"])
    lw = float(PLOT_SETTINGS["line_width"])

    # Global x-left for staircase (slightly left of rho=1)
    x_left = 1.0

    # Track global x-max for extending staircases to the right edge
    global_rf_max = max(r["rf"] for r in rows) * 1.03

    for method in ["udfs", "bingo"]:
        ks, rf_mean, rf_std, oh_mean, oh_std = _aggregate_by_k(rows, method)
        if len(ks) == 0:
            continue

        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        is_pareto = _pareto_membership(rf_mean, oh_mean)

        # Error bars for all points
        ax.errorbar(
            rf_mean,
            oh_mean,
            xerr=rf_std,
            yerr=oh_std,
            fmt="none",
            ecolor=color,
            elinewidth=0.5,
            capsize=0,
            alpha=0.15,
            zorder=2,
        )

        # Non-Pareto points: faded
        if (~is_pareto).any():
            sc = ax.scatter(
                rf_mean[~is_pareto],
                oh_mean[~is_pareto],
                c=ks[~is_pareto],
                cmap="viridis",
                marker=marker,
                s=25,
                edgecolors=color,
                linewidths=0.4,
                alpha=0.25,
                vmin=1,
                vmax=31,
                zorder=3,
            )

        # Pareto points: full visibility
        sc = ax.scatter(
            rf_mean[is_pareto],
            oh_mean[is_pareto],
            c=ks[is_pareto],
            cmap="viridis",
            marker=marker,
            s=35,
            edgecolors=color,
            linewidths=0.8,
            vmin=1,
            vmax=31,
            zorder=4,
        )

        # Pareto staircase (extended to right edge)
        stair_x, stair_y = _pareto_staircase(rf_mean, oh_mean, x_left)
        # Extend: horizontal line from last Pareto point to right edge
        stair_x = np.append(stair_x, global_rf_max)
        stair_y = np.append(stair_y, stair_y[-1])

        # Shade dominated region (above the staircase to top of plot)
        y_top = max(r["overhead_s"] for r in rows) * 5
        ax.fill_between(
            stair_x,
            stair_y,
            y_top,
            color=color,
            alpha=0.06,
            zorder=0,
        )
        ax.plot(
            stair_x,
            stair_y,
            color=color,
            linewidth=lw,
            linestyle="-",
            alpha=0.6,
            zorder=1,
        )

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("$k$ (internal nodes)", fontsize=label_fs - 1)
    cbar.ax.tick_params(labelsize=tick_fs - 1)

    ax.set_xlabel(r"Reduction factor $\rho$", fontsize=label_fs)
    ax.set_ylabel("Canonicalization overhead (s)", fontsize=label_fs)
    ax.set_yscale("log")
    ax.tick_params(labelsize=tick_fs)

    # Legend: method shapes
    legend_handles = [
        Line2D(
            [],
            [],
            marker=METHOD_MARKERS[m],
            color=METHOD_COLORS[m],
            markerfacecolor="none",
            markeredgecolor=METHOD_COLORS[m],
            markersize=6,
            linewidth=lw,
            label=m.upper(),
        )
        for m in ["udfs", "bingo"]
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=int(PLOT_SETTINGS["legend_fontsize"]) - 1,
        loc="upper left",
        frameon=False,
    )

    fig.subplots_adjust(left=0.16, right=0.88, top=0.96, bottom=0.14)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = str(output_dir / "rf_vs_overhead")
    saved = save_figure(fig, fig_path)
    plt.close(fig)
    log.info("Saved: %s", saved)

    caption = (
        "Reduction factor $\\rho$ versus canonicalization overhead, "
        "aggregated by maximum internal node count $k$. Each point is the "
        "mean over all runs sharing the same $(k, \\text{method})$ pair; "
        "error bars show $\\pm 1$ standard deviation. Color encodes $k$ "
        "(viridis scale). Staircase lines trace the Pareto frontier per "
        "method: for any target $\\rho$, the minimum overhead achievable. "
        "UDFS (circles, green) achieves moderate $\\rho$ at low cost; "
        "Bingo (diamonds, purple) reaches higher $\\rho$ but at greater "
        "overhead due to larger DAGs explored during stochastic search."
    )
    (output_dir / "rf_vs_overhead.caption.txt").write_text(caption)


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RF vs overhead Pareto plot",
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    generate_figure(Path(args.results_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
