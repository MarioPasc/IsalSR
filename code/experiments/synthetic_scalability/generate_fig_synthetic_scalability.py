# ruff: noqa: N802, N803, N806
"""Synthetic scalability figure for the IsalSR supplementary material.

Single-panel figure combining canonicalization timing (left y-axis) with
the k! equivalent-representations curve (right y-axis).  A text annotation
states the invariance result: rho = k! when every ordering was enumerated,
and the observed canonical invariance when orderings were sampled instead.

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
    """Aggregate rows by k (pooling all m values).

    Timing statistics are reported in microseconds (``*_us`` keys): the raw
    ``mean_canon_time_s`` column is in seconds.
    """
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
        # Microseconds: on the compiled engine every per-k median rounds to
        # 0.01-0.02 ms, which collapses the column and hides the k trend.
        times_us = [float(r["mean_canon_time_s"]) * 1e6 for r in group]
        med_t = float(np.median(times_us))
        iqr_lo = float(np.percentile(times_us, 25))
        iqr_hi = float(np.percentile(times_us, 75))
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
                "median_time_us": med_t,
                "iqr_lo_us": iqr_lo,
                "iqr_hi_us": iqr_hi,
            }
        )
    return table


# =============================================================================
# Single-panel figure
# =============================================================================


def _rho_equals_kfact_everywhere(rows: list[dict[str, float | int]]) -> bool:
    """Whether ``rho = k!`` holds on every row.

    This is the exact predicate ``_export_latex_table`` uses to decide between
    ``\\checkmark`` and ``$\\times$`` in its ``rho = k!`` column, hoisted so the
    figure annotation and the table cannot disagree about the same data.

    Args:
        rows: Per-expression records as loaded by :func:`load_synthetic_data`.

    Returns:
        True when every row has ``rho_over_kfact`` equal to one.
    """
    return all(abs(float(r["rho_over_kfact"]) - 1.0) < 1e-6 for r in rows)


def _format_power_of_ten(value: float) -> str:
    """Render a large count as LaTeX scientific notation with two decimals."""
    exponent = int(math.floor(math.log10(value))) if value > 0 else 0
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.2f} \times 10^{{{exponent}}}"


def _invariance_badge_text(rows: list[dict[str, float | int]]) -> str:
    """Build the annotation stating what the data actually supports.

    ``rho = k!`` is only claimable when every expression enumerated all ``k!``
    orderings. When orderings are *sampled* instead, ``rho`` equals the sample
    size, so the claim is false and the standing result is the invariance itself:
    every sampled ordering of a given DAG produced one and the same canonical
    string.

    Args:
        rows: Per-expression records as loaded by :func:`load_synthetic_data`.

    Returns:
        LaTeX-ready annotation text.
    """
    if _rho_equals_kfact_everywhere(rows):
        return r"$\rho = k!\;\;\forall\; k,\, m$"

    n_invariant = sum(1 for r in rows if int(r["n_unique_canonicals"]) == 1)
    n_orderings = float(sum(int(r["n_perms"]) for r in rows))
    if n_invariant != len(rows):
        return rf"{100.0 * n_invariant / len(rows):.1f}\% canonical invariance"
    return (
        "one canonical string on all\n"
        rf"${_format_power_of_ten(n_orderings)}$ sampled orderings"
    )


def _thinned_xticks(k_values: list[int], max_labels: int = 9) -> list[int]:
    """Choose a legible subset of ``k`` values to label on the x-axis.

    Labelling all 21 ``k`` values of the scaling grid produces an unreadable run
    of digits at column width. The stride is chosen in ``k`` units (not index
    units) so the printed labels stay evenly spaced even where the grid itself
    is not.

    Args:
        k_values: Sorted ``k`` values present in the data.
        max_labels: Largest number of labels considered legible.

    Returns:
        The ``k`` values to label, always including the smallest one.
    """
    if len(k_values) <= max_labels:
        return list(k_values)
    k_min = min(k_values)
    for stride in (2, 4, 5, 10, 20, 50):
        labelled = [k for k in k_values if (k - k_min) % stride == 0]
        if len(labelled) <= max_labels:
            return labelled
    return [k_values[0], k_values[-1]]


def plot_figure(
    ax: plt.Axes,
    rows: list[dict[str, float | int]],
    powerlaw_fit: bool = False,
) -> float:
    """Plot timing boxplots (left y, microseconds) + k! curve (right y) + annotation.

    The power-law fit is always computed and returned; ``powerlaw_fit`` controls
    only whether the fitted curve and its legend entry are drawn. It defaults to
    ``False`` because on the exhaustive dataset the permutation count per
    expression is itself :math:`k!`, so any per-expression warm-up amortises as
    ``C / k!`` and biases the fitted exponent downward. The reportable exponent
    comes from a fixed-permutation-count run.

    Args:
        ax: Axes to draw on.
        rows: Per-expression records as loaded by :func:`load_synthetic_data`.
        powerlaw_fit: Draw the fitted power-law curve and its legend entry.

    Returns:
        Fitted power-law exponent b.
    """
    # ---- Group by (k, m); times in microseconds (the CSV column is seconds) ----
    cell: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in rows:
        cell[(int(r["k"]), int(r["m"]))].append(float(r["mean_canon_time_s"]) * 1e6)

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

    # ---- Power-law fit (always computed; drawn only when requested) ----
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

    if powerlaw_fit:
        k_arr = np.linspace(min(k_values), max(k_values), 200)
        poly_curve = a_fit * k_arr**b_fit

        ax.plot(
            k_arr,
            poly_curve,
            color="0.3",
            linestyle="--",
            linewidth=float(PLOT_SETTINGS["line_width_thick"]),
            # Two decimals, matching the exponent as the prose and the JSON state
            # it. One decimal printed 1.4 beside a sentence saying 1.43 -- the same
            # fit at two precisions in one document. "(fitted)" is dropped: the
            # dashed key already reads as a fit, and the parenthetical was wide
            # enough to be clipped by the legend frame.
            label=rf"$O(k^{{{b_fit:.2f}}})$ fit",
            zorder=5,
        )

    ax.set_yscale("log")
    ax.set_xlim(min(k_values) - 0.6, max(k_values) + 0.6)
    ax.set_xlabel("Internal nodes $k$")
    ax.set_ylabel(r"Canonicalization time ($\mu$s)")
    labelled_k = _thinned_xticks(k_values)
    ax.set_xticks(labelled_k)
    ax.set_xticklabels([str(k) for k in labelled_k])
    ax.set_xticks(k_values, minor=True)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=2.0)

    ax.yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=20,
        )
    )
    # A log axis whose view contains at most one power of ten shows a single
    # labelled tick, off which no value can be read. Label minor ticks there.
    # Both datasets are in this regime: 10-26 us (exhaustive, sub-decade) and
    # ~30-400 us (scaling, 1.1 decades but still only 10^2 inside the view).
    y_lo, y_hi = ax.get_ylim()
    decades_in_view = [d for d in range(-15, 16) if y_lo <= 10.0**d <= y_hi]
    if y_lo > 0 and len(decades_in_view) <= 1:
        if math.log10(y_hi / y_lo) >= 1.0:
            # Over a decade, subs 2..9 would print ~15 labels; keep 2, 3, 5.
            ax.yaxis.set_minor_locator(
                matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.3, 0.5), numticks=20)
            )
        plain = matplotlib.ticker.FormatStrFormatter("%g")
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(plain)
        ax.tick_params(axis="y", which="minor", labelsize=int(PLOT_SETTINGS["tick_labelsize"]) - 1)
    else:
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
        _invariance_badge_text(rows),
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
    exhaustive = all(bool(row["rho_equals_kfact"]) for row in table_data)
    if exhaustive:
        protocol = (
            r"All $k!$ internal-node permutations were exhaustively "
            r"canonicalized. $\rho = k!$ confirms that every permuted DAG maps "
            r"to the same canonical string "
            r"($|\mathrm{Aut}(D)| = 1$ for all expressions)."
        )
    else:
        protocol = (
            r"The Perms column gives the number of internal-node orderings "
            r"sampled per expression; $k!$ is not enumerable at these $k$, so "
            r"$\rho$ is bounded by the sample size and $\rho = k!$ does not "
            r"hold. The Invariance column is the claim under test: the fraction "
            r"of expressions whose sampled orderings all yielded one canonical "
            r"string."
        )
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Synthetic scalability: canonical invariance and timing "
        r"for random expression DAGs with $k$ internal nodes. " + protocol + "}",
        r"  \label{tab:synthetic_scalability}",
        r"  \small",
        r"  \begin{tabular}{r r r c c r}",
        r"    \toprule",
        r"    $k$ & $k!$ & Perms & $\rho = k!$ & Invariance & Time ($\mu$s) \\",
        r"    \midrule",
    ]
    for row in table_data:
        k = int(row["k"])
        kf = int(row["k_factorial"])
        n_perms = int(row["n_perms"])
        rho_ok = r"\checkmark" if row["rho_equals_kfact"] else r"$\times$"
        inv_pct = f"{float(row['pct_invariant']):.0f}\\%"
        med = float(row["median_time_us"])
        iqr_lo = float(row["iqr_lo_us"])
        iqr_hi = float(row["iqr_hi_us"])
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
    powerlaw_fit: bool = False,
) -> str:
    """Generate figure caption from data summary.

    Args:
        rows: Per-expression records as loaded by :func:`load_synthetic_data`.
        b_fit: Fitted power-law exponent.
        powerlaw_fit: Whether the figure shows the fitted curve. When ``False``
            the caption states the measured median growth instead of the fit,
            because no fitted curve is drawn to refer to.

    Returns:
        The caption text.
    """
    k_values = sorted({int(r["k"]) for r in rows})
    m_values = sorted({int(r["m"]) for r in rows})
    n_total = len(rows)
    n_expr_per_cell = len(rows) // (len(k_values) * len(m_values))
    k_max = max(k_values)
    k_max_fact = math.factorial(k_max)

    n_exact_kfact = sum(1 for r in rows if abs(float(r["rho_over_kfact"]) - 1.0) < 1e-6)
    n_invariant = sum(1 for r in rows if int(r["n_unique_canonicals"]) == 1)
    n_orderings = float(sum(int(r["n_perms"]) for r in rows))

    k_min = min(k_values)
    med_lo = float(
        np.median([float(r["mean_canon_time_s"]) * 1e6 for r in rows if int(r["k"]) == k_min])
    )
    med_hi = float(
        np.median([float(r["mean_canon_time_s"]) * 1e6 for r in rows if int(r["k"]) == k_max])
    )

    if powerlaw_fit:
        growth = (
            f"Left axis (boxplots): canonicalization time grows as "
            f"$O(k^{{{b_fit:.2f}}})$ (power-law fit, dashed), confirming that "
            f"the greedy-invariant algorithm avoids the factorial worst case. "
        )
    else:
        growth = (
            f"Left axis (boxplots): the median canonicalization time rises only "
            f"from {med_lo:.2f} to {med_hi:.2f}\\,$\\mu$s between "
            f"$k = {k_min}$ and $k = {k_max}$ "
            f"(a factor of {med_hi / med_lo:.2f}), confirming that the "
            f"greedy-invariant algorithm avoids the factorial worst case. "
        )

    if _rho_equals_kfact_everywhere(rows):
        protocol = (
            "For each expression, all $k!$ permutations of internal node IDs "
            "were exhaustively canonicalized via the WL-guided greedy algorithm. "
        )
        outcome = (
            f"The reduction factor equals $k!$ for all "
            f"{n_exact_kfact}/{n_total} expressions "
            f"($|\\mathrm{{Aut}}(D)| = 1$, trivial automorphism group), with "
            f"100\\% canonical invariance. "
            f"At $k = {k_max}$, ${k_max_fact:,}$ equivalent representations "
            f"are collapsed in $< 1$\\,ms."
        )
    else:
        # Sampled orderings: rho is the sample size, not k!, so the claim that
        # survives is the invariance -- one canonical string per expression.
        perms_per_expr = sorted({int(r["n_perms"]) for r in rows})
        perms_str = (
            f"{perms_per_expr[0]:,}".replace(",", "{,}")
            if len(perms_per_expr) == 1
            else f"{perms_per_expr[0]:,}--{perms_per_expr[-1]:,}".replace(",", "{,}")
        )
        protocol = (
            f"For each expression, ${perms_str}$ uniformly sampled permutations "
            f"of internal node IDs were canonicalized via the WL-guided greedy "
            f"algorithm; $k!$ is not enumerable at these $k$. "
        )
        outcome = (
            f"All ${_format_power_of_ten(n_orderings)}$ sampled orderings "
            f"collapsed to one canonical string per expression "
            f"({n_invariant}/{n_total} expressions, 100\\% canonical "
            f"invariance), so no automorphism was observed and the measured "
            f"reduction factor is bounded below only by the sample size. "
            f"At $k = {k_max}$ the right axis plots "
            f"${_format_power_of_ten(float(k_max_fact))}$ orderings."
        )

    return (
        f"Synthetic scalability analysis. "
        f"{n_expr_per_cell} random expression DAGs per $(k, m)$ cell were "
        f"generated via the Lample--Charton (2020) method with operators "
        f"$\\{{+, \\times, \\hat{{}}, \\sin, \\cos, \\exp, \\log, "
        f"\\mathrm{{neg}}, \\mathrm{{inv}}\\}}$ and "
        f"$m \\in \\{{{', '.join(str(v) for v in m_values)}\\}}$ variables. "
        f"{protocol}"
        f"{growth}"
        f"Right axis (dotted, red shading): the $k!$ equivalent "
        f"representations that canonicalization collapses to a single "
        f"canonical string. "
        f"{outcome}"
    )


# =============================================================================
# Main
# =============================================================================


def generate_figure(data_dir: str, output_dir: str, powerlaw_fit: bool = False) -> str:
    """Generate the single-panel synthetic scalability figure.

    Args:
        data_dir: Directory holding the ``synth_k*_m*.csv`` fragments.
        output_dir: Destination for the figure, caption and LaTeX table.
        powerlaw_fit: Overlay the fitted power law on the timing axis.

    Returns:
        The output path stem of the saved figure.
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

    # Single-panel figure (single-column width)
    fig_w, fig_h = get_figure_size("single", height_ratio=0.85)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.16, right=0.82, top=0.95, bottom=0.14)

    b_fit = plot_figure(ax, rows, powerlaw_fit=powerlaw_fit)
    logger.info("Power-law exponent b = %.3f (drawn=%s)", b_fit, powerlaw_fit)

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "fig_synthetic_scalability")
    saved = save_figure(fig, out_path)
    for path in saved:
        logger.info("Saved: %s", path)
    plt.close(fig)

    # Save caption
    caption = _build_caption(rows, b_fit, powerlaw_fit=powerlaw_fit)
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
    parser.add_argument(
        "--powerlaw-fit",
        action="store_true",
        help=(
            "Overlay the fitted power law on the timing axis. Off by default: on "
            "the exhaustive dataset the permutation count per expression is k!, so "
            "per-expression warm-up amortises as C/k! and biases the exponent down."
        ),
    )
    args = parser.parse_args()
    generate_figure(args.data_dir, args.output_dir, powerlaw_fit=args.powerlaw_fit)
