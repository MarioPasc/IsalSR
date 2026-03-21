# ruff: noqa: N802, N803, N806
"""Search space analysis figure for the IsalSR arXiv paper.

Two-panel figure showing the central claim: canonical string representation
eliminates redundant expression evaluations.

Panel (a): Per-bin redundancy rate by expression complexity (internal nodes k).
Panel (b): Cumulative unique discovery curve (species accumulation, population level).

Data source: Picasso HPC results (reduction_T1.csv, T_max=10).
Panel (b) recomputes locally using identical methodology (seed=42, T=10, timeout=1s).

Run:
    cd /home/mpascual/research/code/IsalSR && \\
    ~/.conda/envs/isalsr/bin/python experiments/scripts/generate_fig_search_space.py
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from collections import defaultdict

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
from isalsr.core.canonical import CanonicalTimeoutError, pruned_canonical_string  # noqa: E402
from isalsr.core.node_types import LABEL_CHAR_MAP, OperationSet  # noqa: E402
from isalsr.core.string_to_dag import StringToDAG  # noqa: E402
from isalsr.search.random_search import random_isalsr_string  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =============================================================================
# Configuration
# =============================================================================

_DATA_DIR = (
    "/media/mpascual/Sandisk2TB/research/isalsr/results/"
    "arXiv_benchmarking/picasso/search_space_analysis"
)
_CSV_FILE = os.path.join(_DATA_DIR, "reduction_T1.csv")  # T=10 (best signal)
_OUTPUT_DIR = "/media/mpascual/Sandisk2TB/research/isalsr/results/figures/arXiv_figures"

_MAX_TOKENS = 10
_SEED = 42
_N_STRINGS = 1000
_CANON_TIMEOUT = 1.0
_MIN_VALID = 20  # Minimum n_valid per bin to display

# Colors per num_variables (Paul Tol Bright, colorblind-safe)
M_COLORS: dict[int, str] = {
    1: PAUL_TOL_BRIGHT["blue"],  # #4477AA
    2: PAUL_TOL_BRIGHT["green"],  # #228833
    3: PAUL_TOL_BRIGHT["purple"],  # #AA3377
}
M_LABELS: dict[int, str] = {1: "$m=1$", 2: "$m=2$", 3: "$m=3$"}


# =============================================================================
# Data Loading (Panel a)
# =============================================================================


def load_reduction_data(csv_path: str) -> dict[int, dict[int, dict[str, float]]]:
    """Load and deduplicate reduction CSV data.

    Since all benchmarks with the same num_variables produce identical data,
    we take one representative row per (num_variables, n_internal_bin).

    Returns:
        Nested dict: m -> k -> {n_valid, n_unique, rho, ci_lower, ci_upper}.
    """
    raw: dict[tuple[int, int], dict[str, float]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row["num_variables"])
            k = int(row["n_internal_bin"])
            key = (m, k)
            if key not in raw:
                raw[key] = {
                    "n_valid": float(row["n_valid_in_bin"]),
                    "n_unique": float(row["n_unique_in_bin"]),
                    "rho": float(row["reduction_factor"]),
                    "ci_lower": float(row["ci_lower"]),
                    "ci_upper": float(row["ci_upper"]),
                }

    result: dict[int, dict[int, dict[str, float]]] = defaultdict(dict)
    for (m, k), vals in raw.items():
        result[m][k] = vals
    return dict(result)


def compute_redundancy_with_ci(
    n_valid: float, n_unique: float, ci_lower_rho: float, ci_upper_rho: float
) -> tuple[float, float, float]:
    """Convert reduction factor data to redundancy rate (%) with CI.

    Redundancy = (1 - N_unique/N_valid) * 100.
    CI is derived from bootstrap CI on rho = N_valid/N_unique.
    Since redundancy = 1 - 1/rho, and 1/rho is monotone decreasing:
      higher rho -> higher redundancy -> ci_upper_rho maps to ci_upper_redundancy.

    Returns:
        (redundancy_pct, ci_lower_pct, ci_upper_pct), all clipped to [0, 100].
    """
    if n_valid == 0:
        return (0.0, 0.0, 0.0)

    point = (1.0 - n_unique / n_valid) * 100.0

    # Convert CI: redundancy = 1 - 1/rho
    # ci_lower_rho -> lower redundancy, ci_upper_rho -> upper redundancy
    ci_lo = max(0.0, (1.0 - 1.0 / max(ci_lower_rho, 1.0)) * 100.0)
    ci_hi = min(100.0, (1.0 - 1.0 / max(ci_upper_rho, 1.0)) * 100.0)

    return (
        float(np.clip(point, 0, 100)),
        float(np.clip(ci_lo, 0, 100)),
        float(np.clip(ci_hi, 0, 100)),
    )


# =============================================================================
# Local Recomputation (Panel b)
# =============================================================================


def compute_cumulative_discovery(
    num_variables: int,
    max_tokens: int = _MAX_TOKENS,
    n_strings: int = _N_STRINGS,
    seed: int = _SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random strings and track cumulative unique canonical discovery.

    Replicates the exact Picasso methodology but retains per-string ordering
    for the species accumulation curve.

    Returns:
        (x_processed, y_unique): cumulative count arrays.
    """
    rng = np.random.default_rng(seed)
    ops = frozenset(LABEL_CHAR_MAP.values())
    allowed_ops = OperationSet(ops)

    seen: set[str] = set()
    x_list: list[int] = []
    y_list: list[int] = []
    n_valid = 0
    n_timeouts = 0

    for _ in range(n_strings):
        raw = random_isalsr_string(num_variables, max_tokens, allowed_ops, rng)
        try:
            dag = StringToDAG(raw, num_variables, allowed_ops).run()
            if dag.node_count <= num_variables:
                continue
            canon = pruned_canonical_string(dag, timeout=_CANON_TIMEOUT)
        except CanonicalTimeoutError:
            n_timeouts += 1
            continue
        except Exception:  # noqa: BLE001
            continue

        n_valid += 1
        seen.add(canon)
        x_list.append(n_valid)
        y_list.append(len(seen))

    if n_timeouts > 0:
        logger.warning("m=%d: %d canonicalization timeouts", num_variables, n_timeouts)

    logger.info(
        "m=%d: %d valid strings, %d unique canonical forms (%.1f%% unique)",
        num_variables,
        n_valid,
        len(seen),
        100.0 * len(seen) / max(n_valid, 1),
    )

    return np.array(x_list), np.array(y_list)


# =============================================================================
# Panel (a): Per-Bin Redundancy Rate
# =============================================================================


def plot_panel_a(
    ax: plt.Axes,
    data: dict[int, dict[int, dict[str, float]]],
) -> None:
    """Plot grouped bar chart of redundancy rate per k, grouped by m."""
    m_values = sorted(data.keys())

    # Determine k values to show: those where at least one m has n_valid >= MIN_VALID
    all_ks: set[int] = set()
    for m in m_values:
        for k, vals in data[m].items():
            if vals["n_valid"] >= _MIN_VALID:
                all_ks.add(k)
    k_values = sorted(all_ks)
    if not k_values:
        logger.warning("No bins with n_valid >= %d", _MIN_VALID)
        return

    # Limit to k=1..6 for clarity (higher k has near-zero redundancy)
    k_values = [k for k in k_values if k <= 6]

    n_groups = len(m_values)
    bar_width = 0.22
    x_positions = np.arange(len(k_values))

    for i, m in enumerate(m_values):
        redundancies = []
        ci_lowers = []
        ci_uppers = []

        for k in k_values:
            if k in data[m] and data[m][k]["n_valid"] >= _MIN_VALID:
                vals = data[m][k]
                red, ci_lo, ci_hi = compute_redundancy_with_ci(
                    vals["n_valid"], vals["n_unique"], vals["ci_lower"], vals["ci_upper"]
                )
                redundancies.append(red)
                ci_lowers.append(ci_lo)
                ci_uppers.append(ci_hi)
            else:
                redundancies.append(0.0)
                ci_lowers.append(0.0)
                ci_uppers.append(0.0)

        red_arr = np.array(redundancies)
        ci_lo_arr = np.array(ci_lowers)
        ci_hi_arr = np.array(ci_uppers)

        # Error bars (asymmetric)
        yerr_lo = np.clip(red_arr - ci_lo_arr, 0, None)
        yerr_hi = np.clip(ci_hi_arr - red_arr, 0, None)

        offset = (i - (n_groups - 1) / 2) * bar_width
        bars = ax.bar(
            x_positions + offset,
            red_arr,
            width=bar_width,
            color=M_COLORS[m],
            alpha=PLOT_SETTINGS["bar_alpha"],
            label=M_LABELS[m],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.errorbar(
            x_positions + offset,
            red_arr,
            yerr=[yerr_lo, yerr_hi],
            fmt="none",
            color="0.3",
            capsize=PLOT_SETTINGS["errorbar_capsize"],
            capthick=PLOT_SETTINGS["errorbar_capthick"],
            elinewidth=PLOT_SETTINGS["errorbar_linewidth"],
            zorder=4,
        )

        # Annotate sample size above bar groups (only for first m to avoid clutter)
        if i == 0:
            for j, k in enumerate(k_values):
                if k in data[m] and data[m][k]["n_valid"] >= _MIN_VALID:
                    n_val = int(data[m][k]["n_valid"])
                    max_red = max(
                        compute_redundancy_with_ci(
                            data[mv][k]["n_valid"],
                            data[mv][k]["n_unique"],
                            data[mv][k]["ci_lower"],
                            data[mv][k]["ci_upper"],
                        )[2]
                        if k in data[mv]
                        else 0.0
                        for mv in m_values
                    )
                    ax.text(
                        j,
                        max_red + 3.5,
                        f"$n$={n_val}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="0.4",
                    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{k}" for k in k_values])
    ax.set_xlabel("Internal nodes $k$")
    ax.set_ylabel("Redundancy rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        loc="upper right",
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
# Panel (b): Cumulative Unique Discovery
# =============================================================================


def plot_panel_b(
    ax: plt.Axes,
    discovery_curves: dict[int, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Plot species accumulation curves (cumulative unique vs strings processed)."""
    # Find maximum x for diagonal reference
    max_x = max(x[-1] for x, _ in discovery_curves.values() if len(x) > 0)

    # Diagonal reference (no deduplication)
    ax.plot(
        [0, max_x],
        [0, max_x],
        color=PAUL_TOL_BRIGHT["grey"],
        linestyle="--",
        linewidth=PLOT_SETTINGS["line_width"],
        label="No deduplication",
        zorder=1,
    )

    m_values = sorted(discovery_curves.keys())
    for m in m_values:
        x, y = discovery_curves[m]
        if len(x) == 0:
            continue

        pct_redundant = 100.0 * (1.0 - y[-1] / x[-1])
        label = f"{M_LABELS[m]} ({pct_redundant:.0f}% redundant)"

        ax.plot(
            x,
            y,
            color=M_COLORS[m],
            linewidth=PLOT_SETTINGS["line_width_thick"],
            label=label,
            zorder=3,
        )

        # Fill between diagonal and curve for m=1 only (strongest effect)
        if m == 1:
            ax.fill_between(
                x,
                x,  # diagonal (y=x)
                y,
                alpha=0.12,
                color=M_COLORS[m],
                zorder=2,
                label="_nolegend_",
            )
            # Label the shaded savings region
            label_idx = int(len(x) * 0.55)
            label_x = x[label_idx]
            label_y = (label_x + y[label_idx]) / 2  # midpoint of gap
            ax.text(
                label_x,
                label_y,
                "Redundant\nevaluations\nsaved",
                ha="center",
                va="center",
                fontsize=7,
                color="0.35",
                fontstyle="italic",
                rotation=40,
            )

    ax.set_xlabel("Valid strings processed")
    ax.set_ylabel("Unique canonical forms")
    ax.legend(
        fontsize=7,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        edgecolor="0.85",
    )

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


def generate_search_space_figure() -> str:
    """Generate the complete 2-panel search space analysis figure.

    Returns:
        Base output path (without extension).
    """
    apply_ieee_style()

    # --- Load data for Panel (a) ---
    logger.info("Loading reduction data from %s", _CSV_FILE)
    data = load_reduction_data(_CSV_FILE)

    # --- Compute cumulative discovery for Panel (b) ---
    logger.info("Computing cumulative discovery curves (local recomputation)...")
    discovery_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for m in sorted(data.keys()):
        logger.info("  m=%d ...", m)
        discovery_curves[m] = compute_cumulative_discovery(m)

    # --- Create figure ---
    fig_w, fig_h = get_figure_size("double", height_ratio=0.50)
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(fig_w, fig_h),
    )
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.96, top=0.90, bottom=0.15)

    # --- Plot panels ---
    plot_panel_a(ax_a, data)
    plot_panel_b(ax_b, discovery_curves)

    # --- Save ---
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUTPUT_DIR, "fig_search_space")
    saved = save_figure(fig, out_path)
    for path in saved:
        logger.info("Saved: %s", path)
    plt.close(fig)

    return out_path


if __name__ == "__main__":
    generate_search_space_figure()
