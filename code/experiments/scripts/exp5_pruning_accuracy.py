"""Experiment 5: Pruning Accuracy -- Exhaustive vs Pruned Canonicalization.

Compares exhaustive vs pruned canonicalization across random DAGs, classifying
results as exact match, same-length-different-string, or length-mismatch.
This validates the 6-tuple pruning reliability claimed in the paper.

Algorithm:
    For each num_vars in {1, 2, 3} and n_internal in {min_nodes..max_nodes}:
        For each sample s in {0..samples_per_node-1}:
            1. seed = base_seed + num_vars * 100000 + n_internal * 1000 + s
            2. dag = make_random_sr_dag(num_vars, n_internal, seed)
            3. Try exhaustive: canon_ex = canonical_string(dag, timeout=timeout)
            4. Try pruned: canon_pr = pruned_canonical_string(dag, timeout=timeout)
            5. Classify: exact_match, same_length, length_mismatch, or timeout
            6. Record row

Output:
    CSV with columns: num_vars, n_internal, sample_id, canon_exhaustive,
    canon_pruned, len_exhaustive, len_pruned, match_type, len_diff,
    timed_out_exhaustive, timed_out_pruned

Plotting (--plot flag):
    Stacked bar chart of classification rates per n_internal (one subplot
    per num_vars). Summary CSV per (num_vars, n_internal).

SLURM compatibility:
    --num-vars <int> and --n-internal <int> enable single-combination dispatch
    for array job parallelism.

Usage:
    python experiments/scripts/exp5_pruning_accuracy.py \\
        --output /tmp/test.csv --max-nodes 5 --samples-per-node 5 --timeout 10 --plot

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
import sys
import time

# Ensure project root is on the path for non-installed runs.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.scripts._dag_generators import make_random_sr_dag
from isalsr.core.canonical import (
    CanonicalTimeoutError,
    canonical_string,
    pruned_canonical_string,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ======================================================================
# Data collection
# ======================================================================


def collect_accuracy_data(
    num_vars_list: list[int],
    min_nodes: int,
    max_nodes: int,
    samples_per_node: int,
    timeout: float,
    base_seed: int,
    n_internal_single: int | None,
) -> list[dict[str, object]]:
    """Collect pruning accuracy data for all configurations.

    For each (num_vars, n_internal, sample) triple, computes both the
    exhaustive and pruned canonical strings and classifies the match.

    Args:
        num_vars_list: List of num_vars values to iterate over.
        min_nodes: Minimum number of internal nodes.
        max_nodes: Maximum number of internal nodes.
        samples_per_node: Number of random DAG samples per configuration.
        timeout: Maximum seconds per canonicalization call.
        base_seed: Base random seed for reproducibility.
        n_internal_single: If not None, only run this single n_internal value
            (for SLURM array job dispatch).

    Returns:
        List of row dictionaries with accuracy results.
    """
    rows: list[dict[str, object]] = []

    if n_internal_single is not None:
        n_internal_range = [n_internal_single]
    else:
        n_internal_range = list(range(min_nodes, max_nodes + 1))

    total_configs = len(num_vars_list) * len(n_internal_range)
    config_idx = 0

    for num_vars in num_vars_list:
        for n_internal in n_internal_range:
            config_idx += 1
            log.info(
                "Config %d/%d: num_vars=%d n_internal=%d (%d samples)",
                config_idx,
                total_configs,
                num_vars,
                n_internal,
                samples_per_node,
            )

            for s in range(samples_per_node):
                seed = base_seed + num_vars * 100000 + n_internal * 1000 + s
                dag = make_random_sr_dag(num_vars, n_internal, seed)

                # --- Exhaustive ---
                timed_out_ex = False
                canon_ex = ""
                try:
                    canon_ex = canonical_string(dag, timeout=timeout)
                except CanonicalTimeoutError:
                    timed_out_ex = True

                # --- Pruned ---
                timed_out_pr = False
                canon_pr = ""
                try:
                    canon_pr = pruned_canonical_string(dag, timeout=timeout)
                except CanonicalTimeoutError:
                    timed_out_pr = True

                # --- Classify ---
                if timed_out_ex or timed_out_pr:
                    match_type = "timeout"
                    len_diff = 0
                elif canon_ex == canon_pr:
                    match_type = "exact_match"
                    len_diff = 0
                elif len(canon_ex) == len(canon_pr):
                    match_type = "same_length"
                    len_diff = 0
                else:
                    match_type = "length_mismatch"
                    len_diff = len(canon_pr) - len(canon_ex)

                len_ex = len(canon_ex) if not timed_out_ex else -1
                len_pr = len(canon_pr) if not timed_out_pr else -1

                rows.append(
                    {
                        "num_vars": num_vars,
                        "n_internal": n_internal,
                        "sample_id": s,
                        "canon_exhaustive": canon_ex if not timed_out_ex else "",
                        "canon_pruned": canon_pr if not timed_out_pr else "",
                        "len_exhaustive": len_ex,
                        "len_pruned": len_pr,
                        "match_type": match_type,
                        "len_diff": len_diff,
                        "timed_out_exhaustive": timed_out_ex,
                        "timed_out_pruned": timed_out_pr,
                    }
                )

                log.info(
                    "  num_vars=%d n_internal=%d: sample %d/%d -> %s%s%s",
                    num_vars,
                    n_internal,
                    s + 1,
                    samples_per_node,
                    match_type,
                    f" (len_diff={len_diff})" if match_type == "length_mismatch" else "",
                    " [EX_TO]" if timed_out_ex else "",
                )

    return rows


def write_csv(rows: list[dict[str, object]], output_path: str) -> None:
    """Write accuracy results to CSV.

    Args:
        rows: List of row dictionaries.
        output_path: Path to the output CSV file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        "num_vars",
        "n_internal",
        "sample_id",
        "canon_exhaustive",
        "canon_pruned",
        "len_exhaustive",
        "len_pruned",
        "match_type",
        "len_diff",
        "timed_out_exhaustive",
        "timed_out_pruned",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote %d rows to %s", len(rows), output_path)


def read_csv(csv_path: str) -> list[dict[str, object]]:
    """Read accuracy results from CSV.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of row dictionaries with parsed types.
    """
    rows: list[dict[str, object]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                {
                    "num_vars": int(raw["num_vars"]),
                    "n_internal": int(raw["n_internal"]),
                    "sample_id": int(raw["sample_id"]),
                    "canon_exhaustive": raw["canon_exhaustive"],
                    "canon_pruned": raw["canon_pruned"],
                    "len_exhaustive": int(raw["len_exhaustive"]),
                    "len_pruned": int(raw["len_pruned"]),
                    "match_type": raw["match_type"],
                    "len_diff": int(raw["len_diff"]),
                    "timed_out_exhaustive": raw["timed_out_exhaustive"] == "True",
                    "timed_out_pruned": raw["timed_out_pruned"] == "True",
                }
            )
    return rows


# ======================================================================
# Summary table
# ======================================================================


def compute_summary(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Compute per-(num_vars, n_internal) summary statistics.

    Args:
        rows: List of row dictionaries from CSV.

    Returns:
        List of summary row dictionaries.
    """
    groups: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["num_vars"]), int(row["n_internal"]))  # type: ignore[arg-type]
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for nv, ni in sorted(groups.keys()):
        group = groups[(nv, ni)]
        n_total = len(group)
        n_exact = sum(1 for r in group if r["match_type"] == "exact_match")
        n_same_len = sum(1 for r in group if r["match_type"] == "same_length")
        n_mismatch = sum(1 for r in group if r["match_type"] == "length_mismatch")
        n_timeout = sum(1 for r in group if r["match_type"] == "timeout")

        # Rates computed over non-timeout samples.
        n_valid = n_total - n_timeout
        exact_rate = n_exact / n_valid if n_valid > 0 else 0.0
        same_len_rate = n_same_len / n_valid if n_valid > 0 else 0.0
        mismatch_rate = n_mismatch / n_valid if n_valid > 0 else 0.0

        # Mean length difference among length-mismatch samples.
        mismatch_diffs = [
            int(r["len_diff"])  # type: ignore[arg-type]
            for r in group
            if r["match_type"] == "length_mismatch"
        ]
        mean_len_diff = sum(mismatch_diffs) / len(mismatch_diffs) if mismatch_diffs else 0.0

        summary.append(
            {
                "num_vars": nv,
                "n_internal": ni,
                "n_total": n_total,
                "n_exact": n_exact,
                "n_same_len": n_same_len,
                "n_length_mismatch": n_mismatch,
                "n_timeout": n_timeout,
                "exact_rate": exact_rate,
                "same_len_rate": same_len_rate,
                "mismatch_rate": mismatch_rate,
                "mean_len_diff": mean_len_diff,
            }
        )

    return summary


def write_summary_csv(summary: list[dict[str, object]], output_path: str) -> None:
    """Write summary table to CSV.

    Args:
        summary: Summary rows from compute_summary().
        output_path: Path to the output CSV file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        "num_vars",
        "n_internal",
        "n_total",
        "n_exact",
        "n_same_len",
        "n_length_mismatch",
        "n_timeout",
        "exact_rate",
        "same_len_rate",
        "mismatch_rate",
        "mean_len_diff",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    log.info("Wrote %d summary rows to %s", len(summary), output_path)


# ======================================================================
# Plotting
# ======================================================================


def generate_plots(rows: list[dict[str, object]], output_dir: str) -> None:
    """Generate stacked bar chart and summary CSV for pruning accuracy.

    Stacked bar chart:
        x-axis: n_internal
        y-axis: percentage (0-100%)
        Three categories: exact_match (green), same_length (yellow),
            length_mismatch (red). Timeouts excluded from percentages.
        One subplot per num_vars (if multiple).
        Text annotation showing overall agreement rate at top of each bar.

    Args:
        rows: List of row dictionaries from CSV.
        output_dir: Directory to save figures and summary CSV.
    """
    # Lazy matplotlib import (not needed for data collection only).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from experiments.plotting_styles import (
        PAUL_TOL_BRIGHT,
        PLOT_SETTINGS,
        apply_ieee_style,
        binomial_ci,
        get_figure_size,
        save_figure,
    )

    apply_ieee_style()

    # --- Compute summary ---
    summary = compute_summary(rows)
    summary_path = os.path.join(output_dir, "summary.csv")
    write_summary_csv(summary, summary_path)

    # --- Organize summary by num_vars ---
    by_nv: dict[int, list[dict[str, object]]] = {}
    for s_row in summary:
        nv = int(s_row["num_vars"])  # type: ignore[arg-type]
        by_nv.setdefault(nv, []).append(s_row)

    num_vars_list = sorted(by_nv.keys())
    n_panels = len(num_vars_list)

    if n_panels == 0:
        log.warning("No data to plot.")
        return

    # --- Colors ---
    color_exact = PAUL_TOL_BRIGHT["green"]
    color_same = PAUL_TOL_BRIGHT["yellow"]
    color_mismatch = PAUL_TOL_BRIGHT["red"]

    # ==================================================================
    # Figure: Stacked bar chart
    # ==================================================================
    fig_w, fig_h = get_figure_size("double", height_ratio=0.4)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(fig_w, fig_h),
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes[0]

    bar_width = 0.65
    annotation_fs = int(PLOT_SETTINGS["annotation_fontsize"])  # type: ignore[arg-type]

    for idx, nv in enumerate(num_vars_list):
        ax = axes_flat[idx]
        nv_summary = sorted(by_nv[nv], key=lambda r: int(r["n_internal"]))  # type: ignore[arg-type]

        ni_vals = [int(r["n_internal"]) for r in nv_summary]  # type: ignore[arg-type]
        x = np.arange(len(ni_vals))

        exact_rates = [float(r["exact_rate"]) * 100 for r in nv_summary]  # type: ignore[arg-type]
        same_rates = [float(r["same_len_rate"]) * 100 for r in nv_summary]  # type: ignore[arg-type]
        mismatch_rates = [float(r["mismatch_rate"]) * 100 for r in nv_summary]  # type: ignore[arg-type]

        # Stack: exact on bottom, same_length in middle, length_mismatch on top.
        ax.bar(
            x,
            exact_rates,
            bar_width,
            label="Exact match",
            color=color_exact,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.bar(
            x,
            same_rates,
            bar_width,
            bottom=exact_rates,
            label="Same length",
            color=color_same,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom_for_mismatch = [e + s for e, s in zip(exact_rates, same_rates, strict=True)]
        ax.bar(
            x,
            mismatch_rates,
            bar_width,
            bottom=bottom_for_mismatch,
            label="Length mismatch",
            color=color_mismatch,
            edgecolor="white",
            linewidth=0.5,
        )

        # Annotate agreement rate (exact + same_length) at top of each bar.
        for j, s_row in enumerate(nv_summary):
            n_valid = (
                int(s_row["n_total"])  # type: ignore[arg-type]
                - int(s_row["n_timeout"])  # type: ignore[arg-type]
            )
            if n_valid > 0:
                # Binomial CI for exact match rate.
                n_exact_count = int(s_row["n_exact"])  # type: ignore[arg-type]
                _ci_lo, _ci_hi = binomial_ci(n_exact_count, n_valid)
                bar_top = exact_rates[j] + same_rates[j] + mismatch_rates[j]
                ax.text(
                    x[j],
                    min(bar_top + 1.5, 102),
                    f"{exact_rates[j]:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=annotation_fs,
                    color="0.3",
                )

                # Report timeout count if any.
                n_to = int(s_row["n_timeout"])  # type: ignore[arg-type]
                if n_to > 0:
                    ax.text(
                        x[j],
                        -5,
                        f"TO:{n_to}",
                        ha="center",
                        va="top",
                        fontsize=annotation_fs - 1,
                        color=PAUL_TOL_BRIGHT["grey"],
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([str(ni) for ni in ni_vals])
        ax.set_xlabel("Internal nodes", fontsize=PLOT_SETTINGS["axes_labelsize"])
        if idx == 0:
            ax.set_ylabel("Percentage (%)", fontsize=PLOT_SETTINGS["axes_labelsize"])
        ax.set_ylim(0, 110)
        ax.set_title(
            f"$m = {nv}$ variable{'s' if nv > 1 else ''}",
            fontsize=PLOT_SETTINGS["axes_titlesize"],
        )

        if idx == n_panels - 1:
            ax.legend(
                fontsize=PLOT_SETTINGS["legend_fontsize"],
                loc="lower left",
            )

        # Panel label (a), (b), (c).
        panel_label = chr(ord("a") + idx)
        ax.text(
            0.02,
            0.95,
            f"({panel_label})",
            transform=ax.transAxes,
            fontsize=PLOT_SETTINGS["panel_label_fontsize"],
            fontweight="bold",
            va="top",
        )

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "fig5_pruning_accuracy")
    saved = save_figure(fig, fig_path)
    log.info("Figure saved: %s", saved)
    plt.close(fig)


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
        "arXiv_benchmarking/exp5_pruning_accuracy/accuracy.csv"
    )

    parser = argparse.ArgumentParser(
        description="Experiment 5: Pruning accuracy -- exhaustive vs pruned canonicalization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=1,
        help="Minimum number of internal nodes.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=8,
        help="Maximum number of internal nodes.",
    )
    parser.add_argument(
        "--samples-per-node",
        type=int,
        default=50,
        help="Number of random DAG samples per (num_vars, n_internal) configuration.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Maximum seconds per canonicalization call.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for DAG generation.",
    )
    parser.add_argument(
        "--num-vars",
        type=int,
        default=0,
        help="Number of input variables (0 = run all of {1, 2, 3}).",
    )
    parser.add_argument(
        "--n-internal",
        type=int,
        default=0,
        help="Single n_internal value to run (0 = run all from min to max). "
        "For SLURM array job dispatch.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots after data collection.",
    )
    return parser.parse_args()


def _print_summary(rows: list[dict[str, object]]) -> None:
    """Print a summary table of accuracy results to the log.

    Args:
        rows: List of row dictionaries.
    """
    summary = compute_summary(rows)

    log.info("")
    log.info("=" * 100)
    log.info("SUMMARY")
    log.info("=" * 100)
    header = (
        f"{'vars':>4} {'k':>3} {'n':>4}  "
        f"{'exact':>6} {'same':>6} {'mismatch':>8} {'timeout':>7}  "
        f"{'exact%':>7} {'same%':>7} {'mism%':>7} {'mean_diff':>9}"
    )
    log.info(header)
    log.info("-" * len(header))

    for s_row in summary:
        nv = int(s_row["num_vars"])  # type: ignore[arg-type]
        ni = int(s_row["n_internal"])  # type: ignore[arg-type]
        n_total = int(s_row["n_total"])  # type: ignore[arg-type]
        n_exact = int(s_row["n_exact"])  # type: ignore[arg-type]
        n_same = int(s_row["n_same_len"])  # type: ignore[arg-type]
        n_mis = int(s_row["n_length_mismatch"])  # type: ignore[arg-type]
        n_to = int(s_row["n_timeout"])  # type: ignore[arg-type]
        e_rate = float(s_row["exact_rate"]) * 100  # type: ignore[arg-type]
        s_rate = float(s_row["same_len_rate"]) * 100  # type: ignore[arg-type]
        m_rate = float(s_row["mismatch_rate"]) * 100  # type: ignore[arg-type]
        m_diff = float(s_row["mean_len_diff"])  # type: ignore[arg-type]

        log.info(
            f"{nv:>4} {ni:>3} {n_total:>4}  "
            f"{n_exact:>6} {n_same:>6} {n_mis:>8} {n_to:>7}  "
            f"{e_rate:>6.1f}% {s_rate:>6.1f}% {m_rate:>6.1f}% {m_diff:>9.2f}"
        )

    # --- Overall statistics ---
    total = len(rows)
    total_timeout = sum(1 for r in rows if r["match_type"] == "timeout")
    total_valid = total - total_timeout
    total_exact = sum(1 for r in rows if r["match_type"] == "exact_match")
    total_same = sum(1 for r in rows if r["match_type"] == "same_length")
    total_mismatch = sum(1 for r in rows if r["match_type"] == "length_mismatch")

    log.info("")
    log.info("OVERALL: %d total, %d valid (excl. timeout)", total, total_valid)
    if total_valid > 0:
        log.info(
            "  exact_match:     %d / %d = %.2f%%",
            total_exact,
            total_valid,
            total_exact / total_valid * 100,
        )
        log.info(
            "  same_length:     %d / %d = %.2f%%",
            total_same,
            total_valid,
            total_same / total_valid * 100,
        )
        log.info(
            "  length_mismatch: %d / %d = %.2f%%",
            total_mismatch,
            total_valid,
            total_mismatch / total_valid * 100,
        )
    log.info("  timeout:         %d / %d", total_timeout, total)
    log.info("")


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    """Entry point for Experiment 5."""
    args = parse_args()

    log.info("=" * 72)
    log.info("EXPERIMENT 5: Pruning Accuracy (Exhaustive vs Pruned)")
    log.info("=" * 72)
    log.info("  output:           %s", args.output)
    log.info("  min_nodes:        %d", args.min_nodes)
    log.info("  max_nodes:        %d", args.max_nodes)
    log.info("  samples_per_node: %d", args.samples_per_node)
    log.info("  timeout:          %.1f s", args.timeout)
    log.info("  base_seed:        %d", args.seed)
    log.info("  num_vars:         %s", args.num_vars if args.num_vars else "{1, 2, 3}")
    log.info("  n_internal:       %s", args.n_internal if args.n_internal else "all")
    log.info("  plot:             %s", args.plot)
    log.info("")

    # Determine num_vars list.
    num_vars_list = [1, 2, 3] if args.num_vars == 0 else [args.num_vars]

    # Determine single n_internal (for SLURM) or None (run all).
    n_internal_single: int | None = None
    if args.n_internal > 0:
        n_internal_single = args.n_internal

    # --- Data collection ---
    t_start = time.perf_counter()
    rows = collect_accuracy_data(
        num_vars_list=num_vars_list,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        samples_per_node=args.samples_per_node,
        timeout=args.timeout,
        base_seed=args.seed,
        n_internal_single=n_internal_single,
    )
    t_total = time.perf_counter() - t_start

    log.info("")
    log.info("Data collection complete: %d rows in %.1f s", len(rows), t_total)

    # --- Write CSV ---
    write_csv(rows, args.output)

    # --- Summary statistics ---
    _print_summary(rows)

    # --- Plotting ---
    if args.plot:
        log.info("")
        log.info("Generating plots...")
        output_dir = os.path.dirname(args.output)
        generate_plots(rows, output_dir)
        log.info("Plotting complete.")


if __name__ == "__main__":
    main()
