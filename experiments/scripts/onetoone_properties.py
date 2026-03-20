"""Single-pass validation of the four fundamental properties P1-P4 of IsalSR.

Validates empirically that:
    P1 (Round-trip fidelity): S2D(D2S(S2D(w))) is isomorphic to S2D(w).
    P2 (DAG acyclicity): Every DAG from S2D admits a valid topological sort.
    P3 (Canonical invariance): canonical(D) is isomorphic to D and is idempotent.
    P4 (Evaluation preservation): round-tripped DAGs evaluate identically.

These properties are the mathematical foundation of the IsalSR representation.
P5 (search space reduction) is validated separately by search_space_analysis.py.

Usage:
    python experiments/scripts/onetoone_properties.py \
        --output-dir /path/to/results --n-strings 5000 --max-tokens 20 --plot

    # Single num_vars for SLURM array dispatch:
    python experiments/scripts/onetoone_properties.py \
        --output-dir /path/to/results --num-vars 2 --n-strings 5000
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from typing import Any

import numpy as np

# Add project root to path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from isalsr.core.canonical import (  # noqa: E402
    CanonicalTimeoutError,
    pruned_canonical_string,
)
from isalsr.core.dag_evaluator import evaluate_dag  # noqa: E402
from isalsr.core.dag_to_string import DAGToString  # noqa: E402
from isalsr.core.node_types import LABEL_CHAR_MAP, OperationSet  # noqa: E402
from isalsr.core.string_to_dag import StringToDAG  # noqa: E402
from isalsr.errors import EvaluationError  # noqa: E402
from isalsr.search.random_search import random_isalsr_string  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =========================================================================
# Constants
# =========================================================================

DEFAULT_OUTPUT_DIR = (
    "/media/mpascual/Sandisk2TB/research/isalsr/results/"
    "arXiv_benchmarking/local/onetoone_properties"
)

# Evaluation test points per num_vars (diverse, covering edge cases).
EVAL_POINTS: dict[int, list[dict[int, float]]] = {
    1: [
        {0: -1.0},
        {0: 0.0},
        {0: 0.5},
        {0: 1.0},
        {0: 2.0},
    ],
    2: [
        {0: -1.0, 1: 0.5},
        {0: 0.0, 1: 1.0},
        {0: 0.5, 1: -0.5},
        {0: 1.0, 1: 2.0},
        {0: 2.0, 1: -1.0},
    ],
    3: [
        {0: 0.5, 1: 1.0, 2: 0.5},
        {0: 1.0, 1: -0.5, 2: 2.0},
        {0: -1.0, 1: 0.5, 2: 1.0},
        {0: 0.5, 1: 2.0, 2: -1.0},
        {0: 1.0, 1: 1.0, 2: 1.0},
    ],
}

# Absolute error threshold for P4 (evaluation preservation).
_P4_THRESHOLD: float = 1e-8

# Per-string CSV columns.
_PER_STRING_COLUMNS = [
    "sample_id",
    "num_vars",
    "n_internal",
    "p1_roundtrip_ok",
    "p2_acyclic",
    "p3_invariance_ok",
    "p3_idempotent",
    "p4_eval_preserved",
    "p4_max_abs_error",
    "canon_timeout",
]

# Summary CSV columns.
_SUMMARY_COLUMNS = [
    "property",
    "num_vars",
    "n_tested",
    "n_passed",
    "pass_rate",
    "ci_lower",
    "ci_upper",
]


# =========================================================================
# Core validation logic
# =========================================================================


def validate_single_string(
    string: str,
    num_vars: int,
    allowed_ops: OperationSet,
    timeout: float,
    eval_points: list[dict[int, float]],
) -> dict[str, Any] | None:
    """Validate properties P1-P4 for a single random IsalSR string.

    Args:
        string: Random IsalSR instruction string.
        num_vars: Number of input variables.
        allowed_ops: Allowed operation types.
        timeout: Canonical string timeout (seconds).
        eval_points: List of input dicts for P4 evaluation.

    Returns:
        Dict with per-string results, or None if the string produced a
        VAR-only DAG (no internal nodes to test) or failed to parse.
    """
    # Step 1: Parse string to DAG.
    try:
        dag = StringToDAG(string, num_vars, allowed_ops).run()
    except Exception:  # noqa: BLE001
        return None  # Parse error -- not a valid DAG to test.

    # Skip VAR-only DAGs (no internal nodes).
    if dag.node_count <= num_vars:
        return None

    n_internal = dag.node_count - num_vars

    # Initialize result dict.
    result: dict[str, Any] = {
        "n_internal": n_internal,
        "p1_roundtrip_ok": False,
        "p2_acyclic": False,
        "p3_invariance_ok": False,
        "p3_idempotent": False,
        "p4_eval_preserved": False,
        "p4_max_abs_error": float("nan"),
        "canon_timeout": False,
    }

    # ---- P2: DAG acyclicity ----
    try:
        order = dag.topological_sort()
        result["p2_acyclic"] = len(order) == dag.node_count
    except ValueError:
        result["p2_acyclic"] = False

    # ---- P1: Round-trip fidelity ----
    # D2S(S2D(w)) -> w', then S2D(w') -> dag_rt
    try:
        w_prime = DAGToString(dag, initial_node=0).run()
        dag_rt = StringToDAG(w_prime, num_vars, allowed_ops).run()
        result["p1_roundtrip_ok"] = dag.is_isomorphic(dag_rt)
    except Exception:  # noqa: BLE001
        result["p1_roundtrip_ok"] = False
        dag_rt = None

    # ---- P4: Evaluation preservation ----
    if dag_rt is not None:
        result["p4_max_abs_error"], result["p4_eval_preserved"] = _check_eval_preservation(
            dag, dag_rt, eval_points
        )
    else:
        result["p4_eval_preserved"] = False
        result["p4_max_abs_error"] = float("nan")

    # ---- P3: Canonical invariance ----
    try:
        canon = pruned_canonical_string(dag, timeout=timeout)
        dag_canon = StringToDAG(canon, num_vars, allowed_ops).run()
        result["p3_invariance_ok"] = dag.is_isomorphic(dag_canon)

        # Idempotence: canonical of the canonical DAG must be identical.
        canon2 = pruned_canonical_string(dag_canon, timeout=timeout)
        result["p3_idempotent"] = canon == canon2
    except CanonicalTimeoutError:
        result["canon_timeout"] = True
    except Exception:  # noqa: BLE001
        # Unexpected error during canonicalization. Mark as timeout=False
        # but P3 remains False (default).
        pass

    return result


def _check_eval_preservation(
    dag_orig: Any,
    dag_rt: Any,
    eval_points: list[dict[int, float]],
) -> tuple[float, bool]:
    """Check that original and round-tripped DAGs evaluate identically.

    If both DAGs fail to evaluate at a test point (same error behavior),
    that point is considered preserved. If only one fails, the point is
    considered a mismatch.

    Args:
        dag_orig: Original DAG from S2D(w).
        dag_rt: Round-tripped DAG from S2D(D2S(S2D(w))).
        eval_points: List of input dicts for evaluation.

    Returns:
        (max_abs_error, is_preserved) tuple.
    """
    max_err: float = 0.0
    all_preserved = True

    for inputs in eval_points:
        val_orig: float | None = None
        val_rt: float | None = None
        orig_ok = True
        rt_ok = True

        try:
            val_orig = evaluate_dag(dag_orig, inputs)
        except (EvaluationError, Exception):  # noqa: BLE001
            orig_ok = False

        try:
            val_rt = evaluate_dag(dag_rt, inputs)
        except (EvaluationError, Exception):  # noqa: BLE001
            rt_ok = False

        # Both fail: same behavior = preserved.
        if not orig_ok and not rt_ok:
            continue

        # One fails, other does not: mismatch.
        if orig_ok != rt_ok:
            all_preserved = False
            continue

        # Both succeed: compare values.
        assert val_orig is not None and val_rt is not None
        if not math.isfinite(val_orig) and not math.isfinite(val_rt):
            # Both non-finite: same behavior.
            continue
        if math.isfinite(val_orig) != math.isfinite(val_rt):
            all_preserved = False
            continue

        err = abs(val_orig - val_rt)
        max_err = max(max_err, err)
        if err >= _P4_THRESHOLD:
            all_preserved = False

    return (max_err, all_preserved)


def run_validation(
    num_vars: int,
    n_strings: int,
    max_tokens: int,
    allowed_ops: OperationSet,
    timeout: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Run the full P1-P4 validation for a given num_vars configuration.

    Generates n_strings random IsalSR strings, validates each one, and
    returns per-string results. Skips VAR-only DAGs and parse errors.

    Args:
        num_vars: Number of input variables.
        n_strings: Number of random strings to generate.
        max_tokens: Maximum tokens per string.
        allowed_ops: Allowed operation types.
        timeout: Canonical string timeout (seconds).
        seed: Random seed for reproducibility.

    Returns:
        List of per-string result dicts (only for valid non-trivial DAGs).
    """
    rng = np.random.default_rng(seed)
    eval_points = EVAL_POINTS[num_vars]
    results: list[dict[str, Any]] = []
    n_parse_errors = 0
    n_var_only = 0
    progress_step = max(1, n_strings // 10)

    log.info(
        "Starting validation: num_vars=%d, n_strings=%d, max_tokens=%d, timeout=%.1fs, seed=%d",
        num_vars,
        n_strings,
        max_tokens,
        timeout,
        seed,
    )

    for i in range(n_strings):
        if (i + 1) % progress_step == 0:
            log.info(
                "  [num_vars=%d] Progress: %d/%d (%.0f%%)",
                num_vars,
                i + 1,
                n_strings,
                100.0 * (i + 1) / n_strings,
            )

        string = random_isalsr_string(num_vars, max_tokens, allowed_ops, rng)
        result = validate_single_string(string, num_vars, allowed_ops, timeout, eval_points)

        if result is None:
            # Distinguish parse errors vs VAR-only.
            try:
                dag = StringToDAG(string, num_vars, allowed_ops).run()
                if dag.node_count <= num_vars:
                    n_var_only += 1
                else:
                    n_parse_errors += 1
            except Exception:  # noqa: BLE001
                n_parse_errors += 1
            continue

        result["sample_id"] = len(results)
        result["num_vars"] = num_vars
        results.append(result)

    n_timeouts = sum(1 for r in results if r["canon_timeout"])

    log.info(
        "  [num_vars=%d] Complete: %d valid DAGs tested, "
        "%d parse errors, %d VAR-only skipped, %d canon timeouts",
        num_vars,
        len(results),
        n_parse_errors,
        n_var_only,
        n_timeouts,
    )

    return results


def compute_summary(
    results: list[dict[str, Any]],
    num_vars: int,
) -> list[dict[str, Any]]:
    """Compute summary statistics with Clopper-Pearson 95% CIs.

    Args:
        results: Per-string result dicts from run_validation.
        num_vars: Number of input variables.

    Returns:
        List of summary rows (one per property).
    """
    from experiments.plotting_styles import binomial_ci

    summary_rows: list[dict[str, Any]] = []

    # Define property checks. For P3, exclude canon timeouts from n_tested.
    properties = [
        ("P1_roundtrip", "p1_roundtrip_ok", False),
        ("P2_acyclic", "p2_acyclic", False),
        ("P3_invariance", "p3_invariance_ok", True),
        ("P3_idempotent", "p3_idempotent", True),
        ("P4_eval_preserved", "p4_eval_preserved", False),
    ]

    for prop_name, key, exclude_timeouts in properties:
        tested = [r for r in results if not r["canon_timeout"]] if exclude_timeouts else results

        n_tested = len(tested)
        n_passed = sum(1 for r in tested if r[key])
        pass_rate = n_passed / max(n_tested, 1)
        ci_lo, ci_hi = binomial_ci(n_passed, n_tested)

        summary_rows.append(
            {
                "property": prop_name,
                "num_vars": num_vars,
                "n_tested": n_tested,
                "n_passed": n_passed,
                "pass_rate": pass_rate,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
            }
        )

    return summary_rows


# =========================================================================
# Output: CSV writing
# =========================================================================


def write_per_string_csv(results: list[dict[str, Any]], path: str) -> None:
    """Write per-string results to CSV.

    Args:
        results: Per-string result dicts.
        path: Output CSV path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PER_STRING_COLUMNS)
        writer.writeheader()
        for r in results:
            row = {col: r.get(col, "") for col in _PER_STRING_COLUMNS}
            writer.writerow(row)
    log.info("Per-string CSV saved: %s (%d rows)", path, len(results))


def write_summary_csv(summary_rows: list[dict[str, Any]], path: str) -> None:
    """Write summary statistics to CSV.

    Args:
        summary_rows: Summary dicts from compute_summary.
        path: Output CSV path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_COLUMNS)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    log.info("Summary CSV saved: %s (%d rows)", path, len(summary_rows))


# =========================================================================
# Plotting
# =========================================================================


def plot_pass_rates(summary_rows: list[dict[str, Any]], output_dir: str) -> None:
    """Plot property pass rates as a grouped bar chart.

    One group per property (P1, P2, P3-inv, P3-idem, P4), one bar per
    num_vars within each group. Error bars from Clopper-Pearson 95% CI.

    Args:
        summary_rows: Summary dicts from compute_summary.
        output_dir: Directory for output figures.
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

    figsize = get_figure_size("double", height_ratio=0.5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Organize data by property and num_vars.
    properties = [
        "P1_roundtrip",
        "P2_acyclic",
        "P3_invariance",
        "P3_idempotent",
        "P4_eval_preserved",
    ]
    prop_labels = [
        "P1\nRound-trip",
        "P2\nAcyclic",
        "P3\nInvariance",
        "P3\nIdempotent",
        "P4\nEval. pres.",
    ]
    all_nv = sorted({r["num_vars"] for r in summary_rows})

    colors_list = [
        PAUL_TOL_BRIGHT["blue"],
        PAUL_TOL_BRIGHT["red"],
        PAUL_TOL_BRIGHT["green"],
    ]

    n_props = len(properties)
    n_nv = len(all_nv)
    bar_width = PLOT_SETTINGS["bar_width"]

    x_centers = np.arange(n_props)

    for j, nv in enumerate(all_nv):
        rates = []
        ci_lo_list = []
        ci_hi_list = []
        for prop in properties:
            matching = [r for r in summary_rows if r["property"] == prop and r["num_vars"] == nv]
            if matching:
                r = matching[0]
                rates.append(r["pass_rate"])
                ci_lo_list.append(r["ci_lower"])
                ci_hi_list.append(r["ci_upper"])
            else:
                rates.append(0.0)
                ci_lo_list.append(0.0)
                ci_hi_list.append(0.0)

        rates_arr = np.array(rates)
        ci_lo_arr = np.array(ci_lo_list)
        ci_hi_arr = np.array(ci_hi_list)
        yerr_lo = np.clip(rates_arr - ci_lo_arr, 0, None)
        yerr_hi = np.clip(ci_hi_arr - rates_arr, 0, None)

        offset = (j - (n_nv - 1) / 2) * bar_width
        ax.bar(
            x_centers + offset,
            rates_arr,
            width=bar_width,
            yerr=[yerr_lo, yerr_hi],
            label=f"$m={nv}$",
            color=colors_list[j % len(colors_list)],
            alpha=PLOT_SETTINGS["bar_alpha"],
            capsize=PLOT_SETTINGS["errorbar_capsize"],
            error_kw={
                "linewidth": PLOT_SETTINGS["errorbar_linewidth"],
                "capthick": PLOT_SETTINGS["errorbar_capthick"],
            },
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(prop_labels, fontsize=PLOT_SETTINGS["tick_labelsize"])
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0.95, 1.005)
    ax.axhline(y=1.0, color=PAUL_TOL_BRIGHT["grey"], linestyle="--", linewidth=0.5, zorder=0)
    ax.legend(
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        loc="lower left",
    )
    ax.set_title("Property validation pass rates (P1-P4)")

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "fig_property_pass_rates")
    saved = save_figure(fig, fig_path)
    for p in saved:
        log.info("Figure saved: %s", p)
    plt.close(fig)


def plot_p4_error_distribution(all_results: list[dict[str, Any]], output_dir: str) -> None:
    """Plot histogram of P4 max absolute errors (log-scale x-axis).

    Args:
        all_results: All per-string result dicts across all num_vars.
        output_dir: Directory for output figures.
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

    # Collect finite P4 errors.
    errors = [
        r["p4_max_abs_error"]
        for r in all_results
        if math.isfinite(r["p4_max_abs_error"]) and r["p4_max_abs_error"] > 0
    ]

    if not errors:
        log.warning("No non-zero finite P4 errors to plot.")
        return

    figsize = get_figure_size("single")
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Log-scale bins.
    min_err = max(min(errors), 1e-18)
    max_err = max(errors)
    if min_err >= max_err:
        max_err = min_err * 10
    bins = np.logspace(np.log10(min_err), np.log10(max_err), 30)

    ax.hist(
        errors,
        bins=bins,
        color=PAUL_TOL_BRIGHT["blue"],
        alpha=PLOT_SETTINGS["bar_alpha"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(
        x=_P4_THRESHOLD,
        color=PAUL_TOL_BRIGHT["red"],
        linestyle="--",
        linewidth=PLOT_SETTINGS["line_width_thick"],
        label="Threshold ($10^{-8}$)",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Max absolute error (P4)")
    ax.set_ylabel("Count")
    ax.set_title("P4 evaluation error distribution")
    ax.legend(fontsize=PLOT_SETTINGS["legend_fontsize"])

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "fig_p4_error_distribution")
    saved = save_figure(fig, fig_path)
    for p in saved:
        log.info("Figure saved: %s", p)
    plt.close(fig)


# =========================================================================
# CLI entry point
# =========================================================================


def main() -> None:
    """Run one-to-one property validation (P1-P4) for IsalSR."""
    parser = argparse.ArgumentParser(
        description="Validate P1-P4 fundamental properties of IsalSR canonical strings."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for CSVs and figures.",
    )
    parser.add_argument(
        "--n-strings",
        type=int,
        default=5000,
        help="Number of random strings per num_vars configuration.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Maximum tokens per random string.",
    )
    parser.add_argument(
        "--num-vars",
        type=int,
        default=0,
        help="Number of input variables. 0 = run all of {1, 2, 3}. "
        "Set to a specific value for SLURM array dispatch.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Canonical string timeout in seconds.",
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
        help="Generate publication-quality figures.",
    )
    args = parser.parse_args()

    # Determine num_vars configurations to run.
    if args.num_vars == 0:
        nv_list = [1, 2, 3]
    else:
        if args.num_vars not in EVAL_POINTS:
            log.error(
                "num_vars=%d not supported (must be in %s)",
                args.num_vars,
                sorted(EVAL_POINTS.keys()),
            )
            sys.exit(1)
        nv_list = [args.num_vars]

    # All operations allowed.
    ops = frozenset(LABEL_CHAR_MAP.values())
    allowed_ops = OperationSet(ops)

    os.makedirs(args.output_dir, exist_ok=True)

    log.info(
        "IsalSR property validation: num_vars=%s, n_strings=%d, "
        "max_tokens=%d, timeout=%.1fs, seed=%d",
        nv_list,
        args.n_strings,
        args.max_tokens,
        args.timeout,
        args.seed,
    )

    all_results: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []

    for nv in nv_list:
        results = run_validation(
            num_vars=nv,
            n_strings=args.n_strings,
            max_tokens=args.max_tokens,
            allowed_ops=allowed_ops,
            timeout=args.timeout,
            seed=args.seed,
        )
        all_results.extend(results)

        # Write per-string CSV.
        csv_path = os.path.join(args.output_dir, f"results_v{nv}.csv")
        write_per_string_csv(results, csv_path)

        # Compute and accumulate summary.
        summary = compute_summary(results, nv)
        all_summary.extend(summary)

    # Write combined summary CSV.
    summary_path = os.path.join(args.output_dir, "summary.csv")
    write_summary_csv(all_summary, summary_path)

    # Print summary table.
    log.info("=" * 70)
    log.info("SUMMARY: Property Validation Results")
    log.info("=" * 70)
    log.info(
        "%-18s %4s %8s %8s %10s %10s %10s",
        "Property",
        "m",
        "Tested",
        "Passed",
        "Rate",
        "CI_lo",
        "CI_hi",
    )
    log.info("-" * 70)
    for r in all_summary:
        log.info(
            "%-18s %4d %8d %8d %10.6f %10.6f %10.6f",
            r["property"],
            r["num_vars"],
            r["n_tested"],
            r["n_passed"],
            r["pass_rate"],
            r["ci_lower"],
            r["ci_upper"],
        )

    # Report overall pass/fail.
    all_pass = all(r["pass_rate"] == 1.0 for r in all_summary)
    if all_pass:
        log.info("ALL PROPERTIES PASS at 100%% across all configurations.")
    else:
        failures = [r for r in all_summary if r["pass_rate"] < 1.0]
        for f in failures:
            log.warning(
                "FAILURE: %s (m=%d): %.4f%% pass rate (%d/%d)",
                f["property"],
                f["num_vars"],
                f["pass_rate"] * 100,
                f["n_passed"],
                f["n_tested"],
            )

    # Generate plots if requested.
    if args.plot:
        plot_pass_rates(all_summary, args.output_dir)
        plot_p4_error_distribution(all_results, args.output_dir)
        log.info("Figures saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
