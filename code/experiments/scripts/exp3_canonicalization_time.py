"""Experiment 3: Canonicalization Time vs Number of Internal Nodes.

Measures CPU time for canonicalization (both exhaustive and pruned) as a function
of the number of internal nodes. This is a key scalability experiment for the
arXiv paper, demonstrating that 6-tuple pruning provides increasing speedup
as DAG complexity grows.

Algorithm:
    For each num_vars in {1, 2, 3} and n_internal in {min_nodes..max_nodes}:
        For each sample s in {0..samples_per_node-1}:
            1. seed = base_seed + num_vars * 10000 + n_internal * 100 + s
            2. dag = make_random_sr_dag(num_vars, n_internal, seed)
            3. Time exhaustive canonical_string(dag, timeout=timeout)
            4. Time pruned pruned_canonical_string(dag, timeout=timeout)
            5. Record: num_vars, n_internal, sample_id, times, lengths, timeout flags

Output:
    CSV with columns: num_vars, n_internal, sample_id, time_exhaustive_s,
    time_pruned_s, len_exhaustive, len_pruned, timed_out_exhaustive, timed_out_pruned

Plotting (--plot flag):
    Figure 3a: Median time vs n_internal (log scale), with IQR error bands.
    Figure 3b: Median speedup ratio (exhaustive / pruned) vs n_internal.

SLURM compatibility:
    --num-vars <int> and --n-internal <int> enable single-combination dispatch
    for array job parallelism.

Usage:
    python experiments/scripts/exp3_canonicalization_time.py \\
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


def collect_timing_data(
    num_vars_list: list[int],
    min_nodes: int,
    max_nodes: int,
    samples_per_node: int,
    timeout: float,
    base_seed: int,
    n_internal_single: int | None,
) -> list[dict[str, int | float | bool]]:
    """Collect canonicalization timing data for all configurations.

    Args:
        num_vars_list: List of num_vars values to iterate over.
        min_nodes: Minimum number of internal nodes.
        max_nodes: Maximum number of internal nodes.
        samples_per_node: Number of random DAG samples per (num_vars, n_internal).
        timeout: Maximum seconds per canonicalization call.
        base_seed: Base random seed for reproducibility.
        n_internal_single: If not None, only run this single n_internal value
            (for SLURM array job dispatch).

    Returns:
        List of row dictionaries with timing results.
    """
    rows: list[dict[str, int | float | bool]] = []

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
                seed = base_seed + num_vars * 10000 + n_internal * 100 + s
                dag = make_random_sr_dag(num_vars, n_internal, seed)

                # --- Exhaustive ---
                timed_out_ex = False
                canon_ex = ""
                t0 = time.perf_counter()
                try:
                    canon_ex = canonical_string(dag, timeout=timeout)
                except CanonicalTimeoutError:
                    timed_out_ex = True
                t_ex = time.perf_counter() - t0
                if timed_out_ex:
                    t_ex = timeout

                # --- Pruned ---
                timed_out_pr = False
                canon_pr = ""
                t0 = time.perf_counter()
                try:
                    canon_pr = pruned_canonical_string(dag, timeout=timeout)
                except CanonicalTimeoutError:
                    timed_out_pr = True
                t_pr = time.perf_counter() - t0
                if timed_out_pr:
                    t_pr = timeout

                rows.append(
                    {
                        "num_vars": num_vars,
                        "n_internal": n_internal,
                        "sample_id": s,
                        "time_exhaustive_s": t_ex,
                        "time_pruned_s": t_pr,
                        "len_exhaustive": len(canon_ex) if not timed_out_ex else -1,
                        "len_pruned": len(canon_pr) if not timed_out_pr else -1,
                        "timed_out_exhaustive": timed_out_ex,
                        "timed_out_pruned": timed_out_pr,
                    }
                )

                if (s + 1) % max(1, samples_per_node // 4) == 0 or s == samples_per_node - 1:
                    log.info(
                        "  num_vars=%d n_internal=%d: %d/%d samples (ex=%.4fs, pr=%.4fs%s%s)",
                        num_vars,
                        n_internal,
                        s + 1,
                        samples_per_node,
                        t_ex,
                        t_pr,
                        " EX_TIMEOUT" if timed_out_ex else "",
                        " PR_TIMEOUT" if timed_out_pr else "",
                    )

    return rows


def write_csv(rows: list[dict[str, int | float | bool]], output_path: str) -> None:
    """Write timing results to CSV.

    Args:
        rows: List of row dictionaries.
        output_path: Path to the output CSV file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        "num_vars",
        "n_internal",
        "sample_id",
        "time_exhaustive_s",
        "time_pruned_s",
        "len_exhaustive",
        "len_pruned",
        "timed_out_exhaustive",
        "timed_out_pruned",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote %d rows to %s", len(rows), output_path)


def read_csv(csv_path: str) -> list[dict[str, int | float | bool]]:
    """Read timing results from CSV.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of row dictionaries with parsed types.
    """
    rows: list[dict[str, int | float | bool]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                {
                    "num_vars": int(raw["num_vars"]),
                    "n_internal": int(raw["n_internal"]),
                    "sample_id": int(raw["sample_id"]),
                    "time_exhaustive_s": float(raw["time_exhaustive_s"]),
                    "time_pruned_s": float(raw["time_pruned_s"]),
                    "len_exhaustive": int(raw["len_exhaustive"]),
                    "len_pruned": int(raw["len_pruned"]),
                    "timed_out_exhaustive": raw["timed_out_exhaustive"] == "True",
                    "timed_out_pruned": raw["timed_out_pruned"] == "True",
                }
            )
    return rows


# ======================================================================
# Plotting
# ======================================================================


def generate_plots(rows: list[dict[str, int | float | bool]], output_dir: str) -> None:
    """Generate Figure 3a (time vs nodes) and Figure 3b (speedup ratio).

    Args:
        rows: List of row dictionaries from CSV.
        output_dir: Directory to save figures.
    """
    # Lazy matplotlib import (not needed for data collection only).
    import matplotlib
    import numpy as np

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

    # --- Organize data by (num_vars, n_internal) ---
    data: dict[int, dict[int, dict[str, list[float]]]] = {}
    for row in rows:
        nv = int(row["num_vars"])
        ni = int(row["n_internal"])
        if nv not in data:
            data[nv] = {}
        if ni not in data[nv]:
            data[nv][ni] = {"ex": [], "pr": []}
        data[nv][ni]["ex"].append(float(row["time_exhaustive_s"]))
        data[nv][ni]["pr"].append(float(row["time_pruned_s"]))

    num_vars_list = sorted(data.keys())
    n_panels = len(num_vars_list)

    if n_panels == 0:
        log.warning("No data to plot.")
        return

    # ==================================================================
    # Figure 3a: Time vs Nodes
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

    color_ex = PAUL_TOL_BRIGHT["blue"]
    color_pr = PAUL_TOL_BRIGHT["red"]
    marker_sz = int(PLOT_SETTINGS["marker_size"])  # type: ignore[call-overload]
    lw = float(PLOT_SETTINGS["line_width"])  # type: ignore[arg-type]
    band_alpha = float(PLOT_SETTINGS["error_band_alpha"])  # type: ignore[arg-type]

    for idx, nv in enumerate(num_vars_list):
        ax = axes_flat[idx]
        ni_vals = sorted(data[nv].keys())
        x = np.array(ni_vals)

        median_ex = np.array([np.median(data[nv][ni]["ex"]) for ni in ni_vals])
        q25_ex = np.array([np.percentile(data[nv][ni]["ex"], 25) for ni in ni_vals])
        q75_ex = np.array([np.percentile(data[nv][ni]["ex"], 75) for ni in ni_vals])

        median_pr = np.array([np.median(data[nv][ni]["pr"]) for ni in ni_vals])
        q25_pr = np.array([np.percentile(data[nv][ni]["pr"], 25) for ni in ni_vals])
        q75_pr = np.array([np.percentile(data[nv][ni]["pr"], 75) for ni in ni_vals])

        ax.plot(
            x,
            median_ex,
            "-o",
            color=color_ex,
            linewidth=lw,
            markersize=marker_sz - 1,
            label="Exhaustive",
        )
        ax.fill_between(
            x,
            q25_ex,
            q75_ex,
            alpha=band_alpha,
            color=color_ex,
        )

        ax.plot(
            x,
            median_pr,
            "--s",
            color=color_pr,
            linewidth=lw,
            markersize=marker_sz - 1,
            label="Pruned",
        )
        ax.fill_between(
            x,
            q25_pr,
            q75_pr,
            alpha=band_alpha,
            color=color_pr,
        )

        ax.set_yscale("log")
        ax.set_xlabel("Internal nodes", fontsize=PLOT_SETTINGS["axes_labelsize"])
        if idx == 0:
            ax.set_ylabel("Time (s)", fontsize=PLOT_SETTINGS["axes_labelsize"])
        ax.set_title(
            f"$m = {nv}$ variable{'s' if nv > 1 else ''}",
            fontsize=PLOT_SETTINGS["axes_titlesize"],
        )
        ax.set_xticks(x)
        ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], loc="upper left")

        # Panel label (a), (b), (c)
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
    fig3a_path = os.path.join(output_dir, "fig3a_time_vs_nodes")
    saved = save_figure(fig, fig3a_path)
    log.info("Figure 3a saved: %s", saved)
    plt.close(fig)

    # ==================================================================
    # Figure 3b: Speedup Ratio
    # ==================================================================
    fig_w2, fig_h2 = get_figure_size("single", height_ratio=0.75)
    fig2, ax2 = plt.subplots(figsize=(fig_w2, fig_h2))

    markers = ["o", "s", "^"]
    colors = [
        PAUL_TOL_BRIGHT["blue"],
        PAUL_TOL_BRIGHT["red"],
        PAUL_TOL_BRIGHT["green"],
    ]

    for idx, nv in enumerate(num_vars_list):
        ni_vals = sorted(data[nv].keys())
        x = np.array(ni_vals)

        # Median speedup: median of per-sample ratios.
        speedups = []
        for ni in ni_vals:
            ex_times = np.array(data[nv][ni]["ex"])
            pr_times = np.array(data[nv][ni]["pr"])
            # Avoid division by zero: clamp pruned time to a small positive value.
            pr_clamped = np.maximum(pr_times, 1e-9)
            ratios = ex_times / pr_clamped
            speedups.append(float(np.median(ratios)))

        ax2.plot(
            x,
            speedups,
            f"-{markers[idx % len(markers)]}",
            color=colors[idx % len(colors)],
            linewidth=lw,
            markersize=marker_sz,
            label=f"$m = {nv}$",
        )

    ax2.set_xlabel("Internal nodes", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax2.set_ylabel("Speedup (exhaustive / pruned)", fontsize=PLOT_SETTINGS["axes_labelsize"])
    ax2.set_title("Pruning speedup ratio", fontsize=PLOT_SETTINGS["axes_titlesize"])
    ax2.axhline(y=1.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax2.legend(fontsize=PLOT_SETTINGS["legend_fontsize"])

    # Only use integer ticks if we have data
    if num_vars_list:
        all_ni = sorted({ni for nv in data for ni in data[nv]})
        ax2.set_xticks(all_ni)

    fig2.tight_layout()
    fig3b_path = os.path.join(output_dir, "fig3b_speedup_ratio")
    saved2 = save_figure(fig2, fig3b_path)
    log.info("Figure 3b saved: %s", saved2)
    plt.close(fig2)


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
        "arXiv_benchmarking/exp3_canonicalization_time/timing.csv"
    )

    parser = argparse.ArgumentParser(
        description="Experiment 3: Canonicalization time vs number of internal nodes.",
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
        default=20,
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


def main() -> None:
    """Entry point for Experiment 3."""
    args = parse_args()

    log.info("=" * 72)
    log.info("EXPERIMENT 3: Canonicalization Time vs Internal Nodes")
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
    rows = collect_timing_data(
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


def _print_summary(rows: list[dict[str, int | float | bool]]) -> None:
    """Print a summary table of timing results to the log.

    Args:
        rows: List of row dictionaries.
    """
    import statistics

    # Group by (num_vars, n_internal).
    groups: dict[tuple[int, int], list[dict[str, int | float | bool]]] = {}
    for row in rows:
        key = (int(row["num_vars"]), int(row["n_internal"]))
        groups.setdefault(key, []).append(row)

    log.info("")
    log.info("=" * 90)
    log.info("SUMMARY")
    log.info("=" * 90)
    header = (
        f"{'vars':>4} {'k':>3} {'n':>4}  "
        f"{'med_ex(s)':>10} {'med_pr(s)':>10} {'speedup':>8}  "
        f"{'TO_ex':>5} {'TO_pr':>5}"
    )
    log.info(header)
    log.info("-" * len(header))

    for nv, ni in sorted(groups.keys()):
        group = groups[(nv, ni)]
        n_samples = len(group)

        ex_times = [float(r["time_exhaustive_s"]) for r in group]
        pr_times = [float(r["time_pruned_s"]) for r in group]
        to_ex = sum(1 for r in group if r["timed_out_exhaustive"])
        to_pr = sum(1 for r in group if r["timed_out_pruned"])

        med_ex = statistics.median(ex_times)
        med_pr = statistics.median(pr_times)
        speedup = med_ex / max(med_pr, 1e-9)

        log.info(
            f"{nv:>4} {ni:>3} {n_samples:>4}  "
            f"{med_ex:>10.6f} {med_pr:>10.6f} {speedup:>8.2f}x  "
            f"{to_ex:>5} {to_pr:>5}"
        )

    log.info("")


if __name__ == "__main__":
    main()
