"""Search space reduction analysis -- THE KEY EXPERIMENT for the IsalSR paper.

Measures the empirical O(k!) search space reduction achieved by canonicalization.
For each benchmark: generate N random IsalSR strings, bin by internal node count k,
count unique canonical forms per bin, and compute the reduction factor per bin.

This directly validates the paper's central claim: the canonical representation
collapses O(k!) equivalent expression DAGs into one.

Usage (basic -- backward compatible):
    python experiments/scripts/search_space_analysis.py \
        --n-strings 1000 --max-tokens 15 --seed 42 \
        --output results/search_space_analysis.csv

Usage (enhanced -- per-bin analysis with bootstrap CIs and plotting):
    python experiments/scripts/search_space_analysis.py \
        --n-strings 1000 --max-tokens-list "10,15,20,25,30" \
        --include-feynman --plot --seed 42 \
        --output results/search_space_reduction.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from collections import defaultdict
from typing import Any

import numpy as np

# Add project root to path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmarks.datasets.nguyen import NGUYEN_BENCHMARKS  # noqa: E402
from isalsr.core.canonical import CanonicalTimeoutError, pruned_canonical_string  # noqa: E402
from isalsr.core.node_types import LABEL_CHAR_MAP, OperationSet  # noqa: E402
from isalsr.core.string_to_dag import StringToDAG  # noqa: E402
from isalsr.search.random_search import random_isalsr_string  # noqa: E402

# Default timeout (seconds) per canonicalization call.
# Reduced from 5.0 to 1.0 to avoid excessive wall-clock on HPC.
# Strings that timeout are discarded — they don't contribute to reduction factor.
_CANON_TIMEOUT: float = 1.0

# Bootstrap defaults.
_N_BOOTSTRAP: int = 2000
_BOOTSTRAP_ALPHA: float = 0.05

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OUTPUT = "/media/mpascual/Sandisk2TB/research/isalsr/results/search_space_analysis.csv"


def _bootstrap_reduction_ci(
    canonical_strings: list[str],
    n_bootstrap: int = _N_BOOTSTRAP,
    alpha: float = _BOOTSTRAP_ALPHA,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Compute bootstrap 95% CI for the reduction factor of a bin.

    The reduction factor is defined as n_valid / n_unique, where n_valid is the
    number of valid strings in the bin and n_unique is the number of distinct
    canonical forms. We resample the list of canonical strings (with replacement)
    and recompute n_unique for each bootstrap replicate, then derive the CI for
    the reduction factor.

    Args:
        canonical_strings: List of canonical strings in this bin (may contain
            duplicates -- that is the whole point).
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (0.05 for 95% CI).
        rng: NumPy random generator.

    Returns:
        (reduction_factor, ci_lower, ci_upper).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_valid = len(canonical_strings)
    if n_valid == 0:
        return (1.0, 1.0, 1.0)

    n_unique = len(set(canonical_strings))
    point_estimate = n_valid / max(n_unique, 1)

    # Bootstrap: resample canonical strings, count unique per resample.
    boot_reductions = np.empty(n_bootstrap)
    indices = np.arange(n_valid)
    str_array = canonical_strings  # keep as list for indexing

    for i in range(n_bootstrap):
        boot_idx = rng.choice(indices, size=n_valid, replace=True)
        boot_sample = [str_array[j] for j in boot_idx]
        n_unique_boot = len(set(boot_sample))
        boot_reductions[i] = n_valid / max(n_unique_boot, 1)

    ci_lower = float(np.percentile(boot_reductions, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_reductions, 100 * (1 - alpha / 2)))

    return (point_estimate, ci_lower, ci_upper)


def analyze_benchmark(
    benchmark: dict[str, Any],
    n_strings: int,
    max_tokens: int,
    allowed_ops: OperationSet,
    seed: int = 42,
    n_bootstrap: int = _N_BOOTSTRAP,
) -> list[dict[str, Any]]:
    """Analyze search space reduction for a single benchmark, binned by k.

    Generates n_strings random IsalSR strings, parses and canonicalizes each,
    bins by n_internal = dag.node_count - num_variables, and computes the
    reduction factor with bootstrap CI per bin.

    Args:
        benchmark: Benchmark dict with at least 'name' and 'num_variables'.
        n_strings: Number of random strings to generate.
        max_tokens: Maximum tokens per string.
        allowed_ops: Allowed operation types.
        seed: Random seed for reproducibility.
        n_bootstrap: Number of bootstrap resamples for CI.

    Returns:
        List of per-bin result dicts with columns matching the extended CSV schema.
    """
    rng = np.random.default_rng(seed)
    nv = benchmark["num_variables"]
    bench_name = benchmark["name"]

    # Collect canonical strings per bin (k = n_internal_nodes).
    bin_canonicals: dict[int, list[str]] = defaultdict(list)
    n_timeouts = 0
    total_generated = 0

    for _ in range(n_strings):
        total_generated += 1
        raw = random_isalsr_string(nv, max_tokens, allowed_ops, rng)
        try:
            dag = StringToDAG(raw, nv, allowed_ops).run()
            if dag.node_count <= nv:
                continue
            k = dag.node_count - nv
            canon = pruned_canonical_string(dag, timeout=_CANON_TIMEOUT)
            bin_canonicals[k].append(canon)
        except CanonicalTimeoutError:
            n_timeouts += 1
            continue
        except Exception:  # noqa: BLE001
            continue

    if n_timeouts > 0:
        log.warning("%s: %d canonicalization timeouts", bench_name, n_timeouts)

    # Build per-bin rows.
    rows: list[dict[str, Any]] = []
    boot_rng = np.random.default_rng(seed + 1)

    for k in sorted(bin_canonicals.keys()):
        canon_list = bin_canonicals[k]
        n_valid = len(canon_list)
        n_unique = len(set(canon_list))
        reduction, ci_lo, ci_hi = _bootstrap_reduction_ci(
            canon_list, n_bootstrap=n_bootstrap, rng=boot_rng
        )
        theoretical = math.factorial(k)

        rows.append(
            {
                "benchmark": bench_name,
                "num_variables": nv,
                "max_tokens": max_tokens,
                "n_internal_bin": k,
                "n_generated": total_generated,
                "n_valid_in_bin": n_valid,
                "n_unique_in_bin": n_unique,
                "reduction_factor": round(reduction, 4),
                "ci_lower": round(ci_lo, 4),
                "ci_upper": round(ci_hi, 4),
                "theoretical_k_factorial": theoretical,
            }
        )

    if rows:
        total_valid = sum(r["n_valid_in_bin"] for r in rows)
        total_unique = sum(r["n_unique_in_bin"] for r in rows)
        overall_red = total_valid / max(total_unique, 1)
        log.info(
            "%s (max_tokens=%d): %d valid / %d unique = %.1fx across %d bins",
            bench_name,
            max_tokens,
            total_valid,
            total_unique,
            overall_red,
            len(rows),
        )
    else:
        log.warning("%s (max_tokens=%d): no valid strings produced", bench_name, max_tokens)

    return rows


def _plot_reduction_factors(
    all_rows: list[dict[str, Any]],
    plot_output: str,
) -> None:
    """Generate reduction factor plot: x = k (n_internal), y = reduction factor.

    Produces one line per (num_variables, max_tokens) combination, with error bars
    from bootstrap CI and a k! reference curve overlay.

    Args:
        all_rows: All per-bin result rows from all benchmarks/max_tokens.
        plot_output: Base path for figure output (no extension).
    """
    import matplotlib.pyplot as plt

    from experiments.plotting_styles import (
        PAUL_TOL_BRIGHT,
        PLOT_SETTINGS,
        apply_ieee_style,
        get_figure_size,
        save_figure,
    )

    apply_ieee_style()

    # Aggregate across benchmarks: for each (num_vars, max_tokens, k), pool data.
    # We group by (num_variables, max_tokens, n_internal_bin) and average.
    from collections import defaultdict as dd

    grouped: dict[tuple[int, int], dict[int, list[dict[str, Any]]]] = dd(lambda: dd(list))
    for row in all_rows:
        key = (row["num_variables"], row["max_tokens"])
        k = row["n_internal_bin"]
        grouped[key][k].append(row)

    fig, ax = plt.subplots(figsize=get_figure_size("double", height_ratio=0.6))

    # Color cycle: one color per num_variables, one linestyle per max_tokens.
    bright_list = list(PAUL_TOL_BRIGHT.values())
    # Collect distinct num_vars and max_tokens values.
    all_nvs = sorted({key[0] for key in grouped})
    all_mts = sorted({key[1] for key in grouped})

    nv_colors = {nv: bright_list[i % len(bright_list)] for i, nv in enumerate(all_nvs)}
    mt_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    # Plot each (nv, mt) combination.
    for key in sorted(grouped.keys()):
        nv, mt = key
        k_data = grouped[key]

        ks = sorted(k_data.keys())
        reductions = []
        ci_lowers = []
        ci_uppers = []

        for k in ks:
            rows_for_k = k_data[k]
            # Pool: weighted average by n_valid_in_bin.
            total_valid = sum(r["n_valid_in_bin"] for r in rows_for_k)
            total_unique = sum(r["n_unique_in_bin"] for r in rows_for_k)
            pooled_red = total_valid / max(total_unique, 1)
            # CI: take the average of the individual CIs (approximate).
            avg_ci_lo = float(np.mean([r["ci_lower"] for r in rows_for_k]))
            avg_ci_hi = float(np.mean([r["ci_upper"] for r in rows_for_k]))
            reductions.append(pooled_red)
            ci_lowers.append(avg_ci_lo)
            ci_uppers.append(avg_ci_hi)

        ks_arr = np.array(ks)
        red_arr = np.array(reductions)
        ci_lo_arr = np.array(ci_lowers)
        ci_hi_arr = np.array(ci_uppers)

        # Clip error bars to be non-negative (pooled point estimate may differ
        # slightly from per-benchmark bootstrap CI averages).
        yerr_lo = np.clip(red_arr - ci_lo_arr, 0, None)
        yerr_hi = np.clip(ci_hi_arr - red_arr, 0, None)

        color = nv_colors[nv]
        ls_idx = all_mts.index(mt) % len(mt_styles)
        linestyle = mt_styles[ls_idx]

        label = f"$m={nv}$, $T={mt}$"
        ax.errorbar(
            ks_arr,
            red_arr,
            yerr=[yerr_lo, yerr_hi],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=PLOT_SETTINGS["line_width"],
            marker="o",
            markersize=PLOT_SETTINGS["marker_size"],
            capsize=PLOT_SETTINGS["errorbar_capsize"],
            capthick=PLOT_SETTINGS["errorbar_capthick"],
            elinewidth=PLOT_SETTINGS["errorbar_linewidth"],
        )

    # k! reference curve.
    all_ks = sorted({row["n_internal_bin"] for row in all_rows})
    if all_ks:
        k_ref = np.arange(max(1, min(all_ks)), max(all_ks) + 1)
        k_factorial = np.array([math.factorial(int(k)) for k in k_ref], dtype=float)
        ax.plot(
            k_ref,
            k_factorial,
            color=PAUL_TOL_BRIGHT["grey"],
            linestyle="--",
            linewidth=PLOT_SETTINGS["line_width_thick"],
            label=r"$k!$ (theoretical upper bound)",
            zorder=0,
        )

    ax.set_xlabel(r"Internal nodes $k$")
    ax.set_ylabel("Reduction factor")
    ax.set_yscale("log")
    ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"], loc="upper left")
    ax.set_title("Search space reduction by internal node count")

    fig.tight_layout()
    saved = save_figure(fig, plot_output)
    for path in saved:
        log.info("Figure saved: %s", path)
    plt.close(fig)


# =========================================================================
# Extended CSV columns (in order).
# =========================================================================
_CSV_COLUMNS = [
    "benchmark",
    "num_variables",
    "max_tokens",
    "n_internal_bin",
    "n_generated",
    "n_valid_in_bin",
    "n_unique_in_bin",
    "reduction_factor",
    "ci_lower",
    "ci_upper",
    "theoretical_k_factorial",
]


def main() -> None:
    """Run search space analysis on Nguyen (and optionally Feynman) benchmarks."""
    parser = argparse.ArgumentParser(description="IsalSR search space reduction analysis")
    parser.add_argument(
        "--n-strings", type=int, default=1000, help="Number of random strings per benchmark."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=15,
        help="Maximum tokens per string (used when --max-tokens-list not set).",
    )
    parser.add_argument(
        "--max-tokens-list",
        type=str,
        default=None,
        help="Comma-separated list of max_tokens values (e.g. '10,15,20').",
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--include-feynman",
        action="store_true",
        help="Include Feynman benchmarks in addition to Nguyen.",
    )
    parser.add_argument("--plot", action="store_true", help="Generate reduction factor plot.")
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Base path for figure (no extension). "
        "Default: output directory + fig_reduction_factor.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=_N_BOOTSTRAP,
        help="Number of bootstrap resamples for CI.",
    )
    parser.add_argument(
        "--max-tokens-index",
        type=int,
        default=0,
        help="1-indexed position into max_tokens_list. "
        "When set, only process that single max_tokens value (for SLURM array dispatch). "
        "0 = process all values.",
    )
    args = parser.parse_args()

    # Determine max_tokens values.
    if args.max_tokens_list is not None:
        max_tokens_values = [int(x.strip()) for x in args.max_tokens_list.split(",")]
    else:
        max_tokens_values = [args.max_tokens]

    # SLURM array dispatch: select a single max_tokens value by index.
    if args.max_tokens_index > 0:
        idx = args.max_tokens_index - 1  # convert to 0-indexed
        if idx >= len(max_tokens_values):
            log.error(
                "max-tokens-index %d out of range (list has %d values)",
                args.max_tokens_index,
                len(max_tokens_values),
            )
            sys.exit(1)
        max_tokens_values = [max_tokens_values[idx]]
        log.info(
            "SLURM array task %d: processing max_tokens=%d",
            args.max_tokens_index,
            max_tokens_values[0],
        )

    # Build benchmark list.
    benchmarks: list[dict[str, Any]] = list(NGUYEN_BENCHMARKS)
    if args.include_feynman:
        from benchmarks.datasets.feynman import FEYNMAN_BENCHMARKS

        benchmarks.extend(FEYNMAN_BENCHMARKS)

    ops = frozenset(LABEL_CHAR_MAP.values())
    allowed_ops = OperationSet(ops)

    log.info(
        "Search space analysis: %d strings, max_tokens=%s, %d benchmarks",
        args.n_strings,
        max_tokens_values,
        len(benchmarks),
    )

    # Collect all per-bin rows.
    all_rows: list[dict[str, Any]] = []
    for mt in max_tokens_values:
        for bench in benchmarks:
            rows = analyze_benchmark(
                bench,
                args.n_strings,
                mt,
                allowed_ops,
                seed=args.seed,
                n_bootstrap=args.n_bootstrap,
            )
            all_rows.extend(rows)

    # Write CSV.
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    log.info("Results saved to %s (%d rows)", args.output, len(all_rows))

    # Summary table.
    print("\n=== Search Space Reduction Analysis (per bin) ===")
    print(
        f"{'Benchmark':<15} {'m':>3} {'T':>4} {'k':>3} "
        f"{'Valid':>7} {'Unique':>7} {'Reduction':>10} "
        f"{'CI_lo':>8} {'CI_hi':>8} {'k!':>10}"
    )
    print("-" * 85)
    for r in all_rows:
        print(
            f"{r['benchmark']:<15} {r['num_variables']:>3} {r['max_tokens']:>4} "
            f"{r['n_internal_bin']:>3} "
            f"{r['n_valid_in_bin']:>7} {r['n_unique_in_bin']:>7} "
            f"{r['reduction_factor']:>10.2f}x "
            f"{r['ci_lower']:>8.2f} {r['ci_upper']:>8.2f} "
            f"{r['theoretical_k_factorial']:>10}"
        )

    # Aggregated summary (overall per benchmark x max_tokens).
    print("\n=== Aggregated Summary ===")
    print(f"{'Benchmark':<15} {'T':>4} {'Valid':>8} {'Unique':>8} {'Reduction':>10} {'Bins':>5}")
    print("-" * 55)
    # Group by (benchmark, max_tokens).
    from itertools import groupby

    def _group_key(r: dict[str, Any]) -> tuple[str, int]:
        return (r["benchmark"], r["max_tokens"])

    for gkey, group_iter in groupby(sorted(all_rows, key=lambda r: _group_key(r)), key=_group_key):
        group = list(group_iter)
        total_valid = sum(r["n_valid_in_bin"] for r in group)
        total_unique = sum(r["n_unique_in_bin"] for r in group)
        overall_red = total_valid / max(total_unique, 1)
        print(
            f"{gkey[0]:<15} {gkey[1]:>4} {total_valid:>8} {total_unique:>8} "
            f"{overall_red:>10.1f}x {len(group):>5}"
        )

    # Plot if requested.
    if args.plot:
        plot_base = args.plot_output
        if plot_base is None:
            out_dir = os.path.dirname(args.output) or "."
            plot_base = os.path.join(out_dir, "fig_reduction_factor")
        _plot_reduction_factors(all_rows, plot_base)


if __name__ == "__main__":
    main()
