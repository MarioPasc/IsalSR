"""Experiment 6: String Compression -- Random vs Greedy D2S vs Canonical.

Measures string length before and after canonicalization for random IsalSR
strings. Compares three representations:
    |w_random|        -- the raw random string length
    |w_greedy_D2S|    -- greedy DAGToString encoding of the parsed DAG
    |w*_canonical|    -- pruned canonical string (shortest invariant encoding)

This demonstrates that canonicalization produces shorter (more compact)
representations, quantifying the compression effect of the canonical form.

Algorithm:
    1. Generate N random IsalSR strings via random_isalsr_string().
    2. For each: parse -> skip if VAR-only -> greedy D2S -> canonical.
    3. Record lengths and compression ratios.
    4. Optionally generate violin/box and histogram plots.

Output:
    CSV with columns: sample_id, n_internal, len_random, len_greedy,
    len_canonical, compression_greedy, compression_canon

Usage:
    python experiments/scripts/exp6_string_compression.py \
        --output /tmp/test.csv --n-strings 100 --plot

Author: Mario Pascual Gonzalez (mpascual@uma.es)
Date: 2026-03-19

References:
    - Lopez-Rubio (2025). arXiv:2512.10429v2. IsalGraph.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import statistics
import sys
import time

# Ensure project root is on the path for non-installed runs.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from isalsr.core.canonical import CanonicalTimeoutError, pruned_canonical_string
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.node_types import LABEL_CHAR_MAP, OperationSet
from isalsr.core.string_to_dag import StringToDAG
from isalsr.search.random_search import random_isalsr_string

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ======================================================================
# Data collection
# ======================================================================


def collect_compression_data(
    n_strings: int,
    max_tokens: int,
    num_vars: int,
    allowed_ops: OperationSet,
    timeout: float,
    seed: int,
) -> list[dict[str, int | float]]:
    """Generate random strings and measure compression from greedy D2S and canonical.

    For each random string:
        1. Parse via StringToDAG.
        2. Skip if no internal nodes (VAR-only DAG).
        3. Encode via greedy DAGToString.
        4. Encode via pruned canonical string.
        5. Record lengths and compression ratios.

    Args:
        n_strings: Number of random strings to generate.
        max_tokens: Maximum token count per random string.
        num_vars: Number of input variables (m).
        allowed_ops: Allowed operation types for string generation.
        timeout: Maximum seconds per canonicalization call.
        seed: Random seed for reproducibility.

    Returns:
        List of row dictionaries with compression measurements.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, int | float]] = []
    sample_id = 0
    n_skipped_var_only = 0
    n_skipped_parse_error = 0
    n_skipped_d2s_error = 0
    n_skipped_timeout = 0

    log_interval = max(1, n_strings // 10)

    for i in range(n_strings):
        if (i + 1) % log_interval == 0 or i == 0:
            log.info(
                "Generating string %d/%d (collected %d valid samples so far)",
                i + 1,
                n_strings,
                sample_id,
            )

        raw = random_isalsr_string(num_vars, max_tokens, allowed_ops, rng)

        # 1. Parse the random string.
        try:
            dag = StringToDAG(raw, num_vars, allowed_ops).run()
        except Exception:  # noqa: BLE001
            n_skipped_parse_error += 1
            continue

        # 2. Skip VAR-only DAGs.
        n_internal = dag.node_count - num_vars
        if n_internal <= 0:
            n_skipped_var_only += 1
            continue

        # 3. Greedy D2S encoding.
        try:
            greedy = DAGToString(dag, initial_node=0).run()
        except Exception:  # noqa: BLE001
            n_skipped_d2s_error += 1
            continue

        # 4. Canonical encoding.
        try:
            canon = pruned_canonical_string(dag, timeout=timeout)
        except CanonicalTimeoutError:
            n_skipped_timeout += 1
            continue
        except Exception:  # noqa: BLE001
            n_skipped_d2s_error += 1
            continue

        # 5. Record.
        len_random = len(raw)
        len_greedy = len(greedy)
        len_canonical = len(canon)

        # Compression ratios (< 1 means shorter than random).
        compression_greedy = len_greedy / len_random if len_random > 0 else 1.0
        compression_canon = len_canonical / len_random if len_random > 0 else 1.0

        rows.append(
            {
                "sample_id": sample_id,
                "n_internal": n_internal,
                "len_random": len_random,
                "len_greedy": len_greedy,
                "len_canonical": len_canonical,
                "compression_greedy": round(compression_greedy, 6),
                "compression_canon": round(compression_canon, 6),
            }
        )
        sample_id += 1

    log.info("")
    log.info("Collection complete: %d valid samples from %d attempts", sample_id, n_strings)
    log.info(
        "Skipped: %d VAR-only, %d parse errors, %d D2S errors, %d timeouts",
        n_skipped_var_only,
        n_skipped_parse_error,
        n_skipped_d2s_error,
        n_skipped_timeout,
    )

    return rows


def write_csv(rows: list[dict[str, int | float]], output_path: str) -> None:
    """Write compression results to CSV.

    Args:
        rows: List of row dictionaries.
        output_path: Path to the output CSV file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        "sample_id",
        "n_internal",
        "len_random",
        "len_greedy",
        "len_canonical",
        "compression_greedy",
        "compression_canon",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote %d rows to %s", len(rows), output_path)


def read_csv(csv_path: str) -> list[dict[str, int | float]]:
    """Read compression results from CSV.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of row dictionaries with parsed types.
    """
    rows: list[dict[str, int | float]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                {
                    "sample_id": int(raw["sample_id"]),
                    "n_internal": int(raw["n_internal"]),
                    "len_random": int(raw["len_random"]),
                    "len_greedy": int(raw["len_greedy"]),
                    "len_canonical": int(raw["len_canonical"]),
                    "compression_greedy": float(raw["compression_greedy"]),
                    "compression_canon": float(raw["compression_canon"]),
                }
            )
    return rows


# ======================================================================
# Summary
# ======================================================================


def print_summary(rows: list[dict[str, int | float]]) -> None:
    """Print summary statistics of compression ratios to the log.

    Reports mean, median, std of compression ratios, and the fraction
    of strings where canonical is shorter/same/longer than random.

    Args:
        rows: List of row dictionaries.
    """
    if not rows:
        log.warning("No data to summarize.")
        return

    # Extract compression ratios.
    comp_greedy = [float(r["compression_greedy"]) for r in rows]
    comp_canon = [float(r["compression_canon"]) for r in rows]
    len_random = [int(r["len_random"]) for r in rows]
    len_greedy = [int(r["len_greedy"]) for r in rows]
    len_canonical = [int(r["len_canonical"]) for r in rows]

    n = len(rows)

    log.info("")
    log.info("=" * 72)
    log.info("COMPRESSION SUMMARY (N = %d)", n)
    log.info("=" * 72)

    # -- Length statistics --
    log.info("")
    log.info("STRING LENGTHS:")
    log.info(
        "  Random:    mean=%.1f  median=%.1f  std=%.1f  min=%d  max=%d",
        statistics.mean(len_random),
        statistics.median(len_random),
        statistics.stdev(len_random) if n > 1 else 0.0,
        min(len_random),
        max(len_random),
    )
    log.info(
        "  Greedy:    mean=%.1f  median=%.1f  std=%.1f  min=%d  max=%d",
        statistics.mean(len_greedy),
        statistics.median(len_greedy),
        statistics.stdev(len_greedy) if n > 1 else 0.0,
        min(len_greedy),
        max(len_greedy),
    )
    log.info(
        "  Canonical: mean=%.1f  median=%.1f  std=%.1f  min=%d  max=%d",
        statistics.mean(len_canonical),
        statistics.median(len_canonical),
        statistics.stdev(len_canonical) if n > 1 else 0.0,
        min(len_canonical),
        max(len_canonical),
    )

    # -- Compression ratio statistics --
    log.info("")
    log.info("COMPRESSION RATIOS (len_X / len_random):")
    log.info(
        "  Greedy/Random:    mean=%.4f  median=%.4f  std=%.4f",
        statistics.mean(comp_greedy),
        statistics.median(comp_greedy),
        statistics.stdev(comp_greedy) if n > 1 else 0.0,
    )
    log.info(
        "  Canonical/Random: mean=%.4f  median=%.4f  std=%.4f",
        statistics.mean(comp_canon),
        statistics.median(comp_canon),
        statistics.stdev(comp_canon) if n > 1 else 0.0,
    )

    # -- Fraction shorter/same/longer --
    # Greedy vs Random
    g_shorter = sum(1 for g, r in zip(len_greedy, len_random, strict=True) if g < r)
    g_same = sum(1 for g, r in zip(len_greedy, len_random, strict=True) if g == r)
    g_longer = sum(1 for g, r in zip(len_greedy, len_random, strict=True) if g > r)

    # Canonical vs Random
    c_shorter = sum(1 for c, r in zip(len_canonical, len_random, strict=True) if c < r)
    c_same = sum(1 for c, r in zip(len_canonical, len_random, strict=True) if c == r)
    c_longer = sum(1 for c, r in zip(len_canonical, len_random, strict=True) if c > r)

    # Canonical vs Greedy
    cg_shorter = sum(1 for c, g in zip(len_canonical, len_greedy, strict=True) if c < g)
    cg_same = sum(1 for c, g in zip(len_canonical, len_greedy, strict=True) if c == g)
    cg_longer = sum(1 for c, g in zip(len_canonical, len_greedy, strict=True) if c > g)

    log.info("")
    log.info("PAIRWISE COMPARISONS:")
    log.info(
        "  Greedy vs Random:    shorter=%d (%.1f%%)  same=%d (%.1f%%)  longer=%d (%.1f%%)",
        g_shorter,
        100 * g_shorter / n,
        g_same,
        100 * g_same / n,
        g_longer,
        100 * g_longer / n,
    )
    log.info(
        "  Canon vs Random:     shorter=%d (%.1f%%)  same=%d (%.1f%%)  longer=%d (%.1f%%)",
        c_shorter,
        100 * c_shorter / n,
        c_same,
        100 * c_same / n,
        c_longer,
        100 * c_longer / n,
    )
    log.info(
        "  Canon vs Greedy:     shorter=%d (%.1f%%)  same=%d (%.1f%%)  longer=%d (%.1f%%)",
        cg_shorter,
        100 * cg_shorter / n,
        cg_same,
        100 * cg_same / n,
        cg_longer,
        100 * cg_longer / n,
    )
    log.info("")


# ======================================================================
# Plotting
# ======================================================================


def _assign_bin(n_internal: int) -> str:
    """Assign an n_internal value to a display bin for plotting.

    Args:
        n_internal: Number of internal nodes.

    Returns:
        Bin label string (e.g. "1-2", "3-4", "5-6", "7+").
    """
    if n_internal <= 2:
        return "1-2"
    elif n_internal <= 4:
        return "3-4"
    elif n_internal <= 6:
        return "5-6"
    else:
        return "7+"


def generate_plots(rows: list[dict[str, int | float]], output_dir: str) -> None:
    """Generate Figure 6a (violin/box) and Figure 6b (compression histogram).

    Args:
        rows: List of row dictionaries from CSV.
        output_dir: Directory to save figures.
    """
    # Lazy matplotlib import (not needed for data collection only).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.plotting_styles import (
        PAUL_TOL_BRIGHT,
        PLOT_SETTINGS,
        apply_ieee_style,
        get_figure_size,
        save_figure,
    )

    apply_ieee_style()

    if not rows:
        log.warning("No data to plot.")
        return

    # -- Organize data by bin --
    bin_order = ["1-2", "3-4", "5-6", "7+"]
    bin_data: dict[str, dict[str, list[int]]] = {
        b: {"random": [], "greedy": [], "canonical": []} for b in bin_order
    }

    for row in rows:
        b = _assign_bin(int(row["n_internal"]))
        bin_data[b]["random"].append(int(row["len_random"]))
        bin_data[b]["greedy"].append(int(row["len_greedy"]))
        bin_data[b]["canonical"].append(int(row["len_canonical"]))

    # Filter to bins with data.
    active_bins = [b for b in bin_order if bin_data[b]["random"]]

    if not active_bins:
        log.warning("No bins with data to plot.")
        return

    # ==================================================================
    # Figure 6a: Grouped Box Plot of String Lengths by n_internal Bin
    # ==================================================================
    fig_w, fig_h = get_figure_size("double", height_ratio=0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    color_random = PAUL_TOL_BRIGHT["blue"]
    color_greedy = PAUL_TOL_BRIGHT["yellow"]
    color_canon = PAUL_TOL_BRIGHT["green"]

    n_bins = len(active_bins)
    box_width = 0.22
    positions_random = []
    positions_greedy = []
    positions_canon = []

    for i in range(n_bins):
        center = i * 1.0
        positions_random.append(center - box_width - 0.02)
        positions_greedy.append(center)
        positions_canon.append(center + box_width + 0.02)

    bp_kwargs: dict[str, object] = {
        "widths": box_width,
        "patch_artist": True,
        "showfliers": True,
        "flierprops": {
            "marker": "o",
            "markersize": PLOT_SETTINGS["boxplot_flier_size"],
            "alpha": 0.5,
        },
        "medianprops": {"color": "black", "linewidth": 1.0},
        "whiskerprops": {"linewidth": PLOT_SETTINGS["boxplot_linewidth"]},
        "capprops": {"linewidth": PLOT_SETTINGS["boxplot_linewidth"]},
        "boxprops": {"linewidth": PLOT_SETTINGS["boxplot_linewidth"]},
    }

    # Random boxes.
    data_random = [bin_data[b]["random"] for b in active_bins]
    bp_r = ax.boxplot(
        data_random,
        positions=positions_random,
        **bp_kwargs,  # type: ignore[arg-type]
    )
    for patch in bp_r["boxes"]:
        patch.set_facecolor(color_random)
        patch.set_alpha(0.75)

    # Greedy boxes.
    data_greedy = [bin_data[b]["greedy"] for b in active_bins]
    bp_g = ax.boxplot(
        data_greedy,
        positions=positions_greedy,
        **bp_kwargs,  # type: ignore[arg-type]
    )
    for patch in bp_g["boxes"]:
        patch.set_facecolor(color_greedy)
        patch.set_alpha(0.75)

    # Canonical boxes.
    data_canon = [bin_data[b]["canonical"] for b in active_bins]
    bp_c = ax.boxplot(
        data_canon,
        positions=positions_canon,
        **bp_kwargs,  # type: ignore[arg-type]
    )
    for patch in bp_c["boxes"]:
        patch.set_facecolor(color_canon)
        patch.set_alpha(0.75)

    # Labels and legend.
    ax.set_xticks(list(range(n_bins)))
    ax.set_xticklabels(active_bins, fontsize=PLOT_SETTINGS["tick_labelsize"])
    ax.set_xlabel("Internal nodes ($k$)", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax.set_ylabel("String length", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax.set_title(
        "String length by representation",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )

    # Create legend handles manually.
    import matplotlib.patches as mpatches

    legend_handles = [
        mpatches.Patch(facecolor=color_random, alpha=0.75, label="Random"),
        mpatches.Patch(facecolor=color_greedy, alpha=0.75, label="Greedy D2S"),
        mpatches.Patch(facecolor=color_canon, alpha=0.75, label="Canonical"),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        loc="upper left",
    )

    fig.tight_layout()
    fig6a_path = os.path.join(output_dir, "fig6a_string_lengths_by_bin")
    saved = save_figure(fig, fig6a_path)
    log.info("Figure 6a saved: %s", saved)
    plt.close(fig)

    # ==================================================================
    # Figure 6b: Histogram of Compression Ratio (canonical / random)
    # ==================================================================
    fig_w2, fig_h2 = get_figure_size("single")
    fig2, ax2 = plt.subplots(figsize=(fig_w2, fig_h2))

    comp_canon = [float(r["compression_canon"]) for r in rows]
    comp_greedy = [float(r["compression_greedy"]) for r in rows]

    # Determine bin range for histograms.
    all_ratios = comp_canon + comp_greedy
    lo = max(0.0, min(all_ratios) - 0.05)
    hi = max(all_ratios) + 0.05
    bins = np.linspace(lo, hi, 40)

    ax2.hist(
        comp_greedy,
        bins=bins,
        alpha=0.6,
        color=PAUL_TOL_BRIGHT["yellow"],
        edgecolor="white",
        linewidth=0.5,
        label="Greedy / Random",
        density=True,
    )
    ax2.hist(
        comp_canon,
        bins=bins,
        alpha=0.6,
        color=PAUL_TOL_BRIGHT["green"],
        edgecolor="white",
        linewidth=0.5,
        label="Canonical / Random",
        density=True,
    )

    # Vertical line at x=1 (no compression).
    ax2.axvline(
        x=1.0,
        color=PAUL_TOL_BRIGHT["red"],
        linestyle="--",
        linewidth=1.2,
        label="No compression ($r = 1$)",
    )

    ax2.set_xlabel("Compression ratio", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax2.set_ylabel("Density", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax2.set_title(
        "Compression ratio distribution",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )
    ax2.legend(fontsize=PLOT_SETTINGS["legend_fontsize"])

    fig2.tight_layout()
    fig6b_path = os.path.join(output_dir, "fig6b_compression_histogram")
    saved2 = save_figure(fig2, fig6b_path)
    log.info("Figure 6b saved: %s", saved2)
    plt.close(fig2)

    # ==================================================================
    # Figure 6c: Compression ratio vs n_internal (scatter + trend)
    # ==================================================================
    fig_w3, fig_h3 = get_figure_size("single")
    fig3, ax3 = plt.subplots(figsize=(fig_w3, fig_h3))

    n_internals = [int(r["n_internal"]) for r in rows]

    ax3.scatter(
        n_internals,
        comp_greedy,
        alpha=0.3,
        s=PLOT_SETTINGS["scatter_size"],
        color=PAUL_TOL_BRIGHT["yellow"],
        edgecolors="none",
        label="Greedy / Random",
    )
    ax3.scatter(
        n_internals,
        comp_canon,
        alpha=0.3,
        s=PLOT_SETTINGS["scatter_size"],
        color=PAUL_TOL_BRIGHT["green"],
        edgecolors="none",
        label="Canonical / Random",
    )

    # Trend lines: median per n_internal.
    unique_ni = sorted(set(n_internals))
    if len(unique_ni) > 1:
        median_greedy_by_ni = []
        median_canon_by_ni = []
        for ni in unique_ni:
            g_vals = [float(r["compression_greedy"]) for r in rows if int(r["n_internal"]) == ni]
            c_vals = [float(r["compression_canon"]) for r in rows if int(r["n_internal"]) == ni]
            median_greedy_by_ni.append(statistics.median(g_vals) if g_vals else 0.0)
            median_canon_by_ni.append(statistics.median(c_vals) if c_vals else 0.0)

        ax3.plot(
            unique_ni,
            median_greedy_by_ni,
            "-o",
            color=PAUL_TOL_BRIGHT["yellow"],
            linewidth=PLOT_SETTINGS["line_width"],
            markersize=PLOT_SETTINGS["marker_size"],
            label="Greedy median",
            zorder=5,
        )
        ax3.plot(
            unique_ni,
            median_canon_by_ni,
            "-s",
            color=PAUL_TOL_BRIGHT["green"],
            linewidth=PLOT_SETTINGS["line_width"],
            markersize=PLOT_SETTINGS["marker_size"],
            label="Canonical median",
            zorder=5,
        )

    ax3.axhline(y=1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax3.set_xlabel("Internal nodes ($k$)", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax3.set_ylabel("Compression ratio", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax3.set_title(
        "Compression vs DAG complexity",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )
    ax3.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], loc="best")

    fig3.tight_layout()
    fig6c_path = os.path.join(output_dir, "fig6c_compression_vs_complexity")
    saved3 = save_figure(fig3, fig6c_path)
    log.info("Figure 6c saved: %s", saved3)
    plt.close(fig3)


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    default_output = (
        "/media/mpascual/Sandisk2TB/research/isalsr/results/"
        "arXiv_benchmarking/exp6_string_compression/compression.csv"
    )

    parser = argparse.ArgumentParser(
        description="Experiment 6: String compression -- random vs greedy D2S vs canonical.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--n-strings",
        type=int,
        default=1000,
        help="Number of random strings to generate.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Maximum number of tokens per random string.",
    )
    parser.add_argument(
        "--num-vars",
        type=int,
        default=1,
        help="Number of input variables (m).",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default="",
        help="Comma-separated label chars to use (e.g. '+,*,s,c,k'). "
        "Empty = all operations from LABEL_CHAR_MAP.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Maximum seconds per canonicalization call.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots after data collection.",
    )
    return parser.parse_args()


def _build_operation_set(ops_str: str) -> OperationSet:
    """Build an OperationSet from a comma-separated label char string.

    Args:
        ops_str: Comma-separated label characters (e.g. "+,*,s,c,k").
            Empty string means all operations.

    Returns:
        Configured OperationSet.
    """
    if not ops_str.strip():
        return OperationSet()  # All operations.

    from isalsr.core.node_types import NodeType

    chars = [c.strip() for c in ops_str.split(",") if c.strip()]
    node_types: set[NodeType] = set()
    for ch in chars:
        if ch not in LABEL_CHAR_MAP:
            log.warning("Unknown label char '%s', skipping.", ch)
            continue
        node_types.add(LABEL_CHAR_MAP[ch])

    if not node_types:
        log.warning("No valid ops parsed from '%s', using all ops.", ops_str)
        return OperationSet()

    return OperationSet(frozenset(node_types))


def main() -> None:
    """Entry point for Experiment 6."""
    args = parse_args()

    allowed_ops = _build_operation_set(args.ops)

    log.info("=" * 72)
    log.info("EXPERIMENT 6: String Compression (Random vs Greedy vs Canonical)")
    log.info("=" * 72)
    log.info("  output:     %s", args.output)
    log.info("  n_strings:  %d", args.n_strings)
    log.info("  max_tokens: %d", args.max_tokens)
    log.info("  num_vars:   %d", args.num_vars)
    log.info("  ops:        %s", allowed_ops)
    log.info("  timeout:    %.1f s", args.timeout)
    log.info("  seed:       %d", args.seed)
    log.info("  plot:       %s", args.plot)
    log.info("")

    # --- Data collection ---
    t_start = time.perf_counter()
    rows = collect_compression_data(
        n_strings=args.n_strings,
        max_tokens=args.max_tokens,
        num_vars=args.num_vars,
        allowed_ops=allowed_ops,
        timeout=args.timeout,
        seed=args.seed,
    )
    t_total = time.perf_counter() - t_start

    log.info("Data collection complete: %d valid samples in %.1f s", len(rows), t_total)

    if not rows:
        log.warning("No valid samples collected. Exiting.")
        return

    # --- Write CSV ---
    write_csv(rows, args.output)

    # --- Summary ---
    print_summary(rows)

    # --- Plotting ---
    if args.plot:
        log.info("Generating plots...")
        output_dir = os.path.dirname(args.output)
        generate_plots(rows, output_dir)
        log.info("Plotting complete.")


if __name__ == "__main__":
    main()
