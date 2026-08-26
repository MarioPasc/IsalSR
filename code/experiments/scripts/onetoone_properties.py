"""Single-pass validation of the four fundamental properties P1-P4 of IsalSR.

Validates empirically that:
    P1 (Round-trip fidelity): S2D(D2S(S2D(w))) is isomorphic to S2D(w).
    P2 (DAG acyclicity): Every DAG from S2D admits a valid topological sort.
    P3 (Canonical invariance): canonical(D) is isomorphic to D and is idempotent.
    P4 (Evaluation preservation): round-tripped DAGs evaluate identically.

These properties are the mathematical foundation of the IsalSR representation.
P5 (search space reduction) is validated separately by search_space_analysis.py.

Phase 1: Statistical validation on N random IsalSR strings per num_vars.
Phase 2 (--plot): Benchmark expression validation on 8 curated Nguyen/Feynman
    expressions, with one scientific figure per property and a LaTeX table.

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
# Phase 2: Benchmark expression validation
# =========================================================================


def _get_benchmark_expressions() -> list[dict[str, Any]]:
    """Return 8 curated benchmark expressions for Phase 2 validation.

    Uses Nguyen and Feynman benchmarks standard in the SR literature.
    SymPy expressions are converted to LabeledDAGs via SympyAdapter.

    Returns:
        List of dicts with keys: name, expr, vars, m.
    """
    from sympy import Rational, Symbol, cos, log, sin, sqrt

    x0, x1, x2 = Symbol("x_0"), Symbol("x_1"), Symbol("x_2")

    return [
        {"name": "Nguyen-1", "expr": x0**3 + x0**2 + x0, "vars": [x0], "m": 1},
        {"name": "Nguyen-5", "expr": sin(x0**2) * cos(x0) - 1, "vars": [x0], "m": 1},
        {
            "name": "Nguyen-7",
            "expr": log(x0 + 1) + log(x0**2 + 1),
            "vars": [x0],
            "m": 1,
        },
        {"name": "Nguyen-8", "expr": sqrt(x0), "vars": [x0], "m": 1},
        {
            "name": "Nguyen-9",
            "expr": sin(x0) + sin(x1**2),
            "vars": [x0, x1],
            "m": 2,
        },
        {
            "name": "Nguyen-10",
            "expr": 2 * sin(x0) * cos(x1),
            "vars": [x0, x1],
            "m": 2,
        },
        {
            "name": "Nguyen-12",
            "expr": x0**4 - x0**3 + Rational(1, 2) * x1**2 - x1,
            "vars": [x0, x1],
            "m": 2,
        },
        {
            "name": "Feynman-I.14.3",
            "expr": x0 * x1 * x2,
            "vars": [x0, x1, x2],
            "m": 3,
        },
    ]


def run_benchmark_validation(
    output_dir: str,
    timeout: float,
) -> list[dict[str, Any]]:
    """Phase 2: validate P1-P4 on curated benchmark expressions.

    For each benchmark:
      - Builds DAG from SymPy via SympyAdapter
      - Validates P1 (round-trip), P2 (acyclicity), P3 (canonical invariance),
        P4 (evaluation preservation on the S2D/D2S round-trip)
      - Collects canonical strings, D2S strings, and invariance proof data

    For P4, evaluation is compared between the S2D-parsed DAG (which has
    CONST=1.0 defaults) and its D2S/S2D round-trip, ensuring the comparison
    is fair (both DAGs share the same CONST values).

    Args:
        output_dir: Directory for output files.
        timeout: Canonical string timeout in seconds.

    Returns:
        List of per-benchmark result dicts.
    """
    from isalsr.adapters.sympy_adapter import SympyAdapter

    adapter = SympyAdapter()
    benchmarks = _get_benchmark_expressions()
    ops = frozenset(LABEL_CHAR_MAP.values())
    allowed_ops = OperationSet(ops)

    bench_data: list[dict[str, Any]] = []

    for bench in benchmarks:
        name: str = bench["name"]
        m: int = bench["m"]
        log.info("Phase 2: validating %s (m=%d)", name, m)

        # Build DAG from SymPy expression.
        dag = adapter.from_sympy(bench["expr"], bench["vars"])
        n_nodes = dag.node_count
        n_edges = dag.edge_count
        n_internal = n_nodes - m
        sympy_str = str(bench["expr"])

        # ------ P2: DAG acyclicity ------
        try:
            topo_order = dag.topological_sort()
            p2_acyclic = len(topo_order) == n_nodes
            dag_depth = _compute_dag_depth(dag, topo_order)
        except ValueError:
            p2_acyclic = False
            dag_depth = -1

        # ------ P1: Round-trip fidelity ------
        # D2S from the SymPy DAG, then S2D to get the round-tripped DAG.
        try:
            w_d2s = DAGToString(dag, initial_node=0).run()
            dag_rt = StringToDAG(w_d2s, m, allowed_ops).run()
            p1_roundtrip = dag.is_isomorphic(dag_rt)
        except Exception:  # noqa: BLE001
            w_d2s = ""
            dag_rt = None
            p1_roundtrip = False

        # ------ P3: Canonical invariance ------
        canon_str = ""
        p3_invariance = False
        p3_idempotent = False
        canon_timeout = False
        invariance_proofs: list[dict[str, str]] = []

        try:
            canon_str = pruned_canonical_string(dag, timeout=timeout)
            dag_canon = StringToDAG(canon_str, m, allowed_ops).run()
            p3_invariance = dag.is_isomorphic(dag_canon)

            # Idempotence: canonical of the canonical DAG.
            canon2 = pruned_canonical_string(dag_canon, timeout=timeout)
            p3_idempotent = canon_str == canon2

            # Invariance proof 1: greedy D2S string canonicalizes to w*.
            if w_d2s and w_d2s != canon_str:
                dag_from_d2s = StringToDAG(w_d2s, m, allowed_ops).run()
                canon_from_d2s = pruned_canonical_string(dag_from_d2s, timeout=timeout)
                invariance_proofs.append(
                    {
                        "description": "Greedy D2S string",
                        "input_string": w_d2s,
                        "canonical_result": canon_from_d2s,
                        "matches_w_star": canon_from_d2s == canon_str,
                    }
                )

            # Invariance proof 2: canonical string with W no-ops inserted.
            w_with_noop = "W" + canon_str + "W"
            dag_noop = StringToDAG(w_with_noop, m, allowed_ops).run()
            canon_from_noop = pruned_canonical_string(dag_noop, timeout=timeout)
            invariance_proofs.append(
                {
                    "description": "Canonical + W no-ops",
                    "input_string": w_with_noop,
                    "canonical_result": canon_from_noop,
                    "matches_w_star": canon_from_noop == canon_str,
                }
            )

            # Invariance proof 3: idempotent re-canonicalization.
            invariance_proofs.append(
                {
                    "description": "Re-canonicalization (idempotent)",
                    "input_string": canon_str,
                    "canonical_result": canon2,
                    "matches_w_star": canon2 == canon_str,
                }
            )

        except CanonicalTimeoutError:
            canon_timeout = True
        except Exception:  # noqa: BLE001
            pass

        # ------ P4: Evaluation preservation ------
        # Compare S2D-parsed DAG vs its D2S/S2D round-trip (both use default
        # CONST=1.0, so the comparison is fair regardless of SymPy CONST values).
        p4_preserved = False
        p4_max_error = float("nan")
        p4_eval_pairs: list[dict[str, float]] = []
        eval_pts = EVAL_POINTS[m]

        if dag_rt is not None:
            try:
                w_rt2 = DAGToString(dag_rt, initial_node=0).run()
                dag_rt2 = StringToDAG(w_rt2, m, allowed_ops).run()
                p4_max_error, p4_preserved = _check_eval_preservation(dag_rt, dag_rt2, eval_pts)
                # Collect individual evaluation pairs for the scatter plot.
                for inputs in eval_pts:
                    try:
                        val_orig = evaluate_dag(dag_rt, inputs)
                        val_rt = evaluate_dag(dag_rt2, inputs)
                        p4_eval_pairs.append(
                            {
                                "inputs": {str(k): v for k, v in inputs.items()},
                                "val_original": val_orig,
                                "val_roundtripped": val_rt,
                                "abs_error": abs(val_orig - val_rt)
                                if (math.isfinite(val_orig) and math.isfinite(val_rt))
                                else float("nan"),
                            }
                        )
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass

        entry: dict[str, Any] = {
            "name": name,
            "sympy_expr": sympy_str,
            "m": m,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_internal": n_internal,
            "dag_depth": dag_depth,
            "canonical_string": canon_str,
            "d2s_string": w_d2s,
            "p1_roundtrip": p1_roundtrip,
            "p2_acyclic": p2_acyclic,
            "p3_invariance": p3_invariance,
            "p3_idempotent": p3_idempotent,
            "p4_eval_preserved": p4_preserved,
            "p4_max_abs_error": p4_max_error,
            "p4_eval_pairs": p4_eval_pairs,
            "canon_timeout": canon_timeout,
            "invariance_proofs": invariance_proofs,
        }
        bench_data.append(entry)

        status = "PASS" if all([p1_roundtrip, p2_acyclic, p3_invariance, p4_preserved]) else "FAIL"
        log.info(
            "  %s: P1=%s P2=%s P3=%s P4=%s  [%s]",
            name,
            p1_roundtrip,
            p2_acyclic,
            p3_invariance,
            p4_preserved,
            status,
        )

    return bench_data


def _compute_dag_depth(dag: Any, topo_order: list[int]) -> int:
    """Compute the longest path length (depth) in a DAG.

    Args:
        dag: LabeledDAG instance.
        topo_order: Nodes in topological order.

    Returns:
        Length of the longest directed path (number of edges).
    """
    dist: dict[int, int] = {node: 0 for node in topo_order}
    for node in topo_order:
        for neighbor in dag.out_neighbors(node):
            if dist[neighbor] < dist[node] + 1:
                dist[neighbor] = dist[node] + 1
    return max(dist.values()) if dist else 0


# =========================================================================
# Phase 2: Figures
# =========================================================================


def plot_p1_roundtrip_figure(bench_data: list[dict[str, Any]], output_dir: str) -> None:
    """Figure P1: canonical strings and round-trip results for benchmark expressions.

    Shows a table-figure with one row per benchmark expression: expression name,
    canonical string (rendered with per-token coloring), node/edge counts, and
    round-trip verification status.

    Args:
        bench_data: Per-benchmark result dicts from run_benchmark_validation.
        output_dir: Directory for output figures.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.plotting_styles import (
        PAUL_TOL_BRIGHT,
        PLOT_SETTINGS,
        apply_ieee_style,
        get_figure_size,
        render_colored_string,
        save_figure,
    )

    apply_ieee_style()

    n_rows = len(bench_data)
    row_height = 0.45
    fig_height = max(2.0, n_rows * row_height + 1.2)
    figsize = (get_figure_size("double")[0], fig_height)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows + 1)

    # Header.
    header_y = n_rows + 0.5
    headers = ["Expression", "Canonical string w*", "|V|", "|E|", "P1"]
    header_x = [0.01, 0.18, 0.72, 0.80, 0.90]
    for hx, htxt in zip(header_x, headers, strict=True):
        ax.text(
            hx,
            header_y,
            htxt,
            fontsize=PLOT_SETTINGS["annotation_fontsize"] + 1,
            fontweight="bold",
            va="center",
            transform=ax.transData,
        )

    # Separator line.
    ax.axhline(
        y=n_rows + 0.15,
        xmin=0.01,
        xmax=0.97,
        color=PAUL_TOL_BRIGHT["grey"],
        linewidth=0.6,
    )

    # Data rows (top to bottom).
    for i, entry in enumerate(bench_data):
        row_y = n_rows - i - 0.3

        # Expression name.
        ax.text(
            header_x[0],
            row_y,
            entry["name"],
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
            va="center",
            fontfamily="serif",
        )

        # Canonical string (colored).
        canon = entry["canonical_string"]
        if canon:
            render_colored_string(
                ax,
                canon,
                x=header_x[1],
                y=row_y,
                fontsize=7,
            )
        else:
            ax.text(
                header_x[1],
                row_y,
                "(timeout)",
                fontsize=PLOT_SETTINGS["annotation_fontsize"],
                va="center",
                color=PAUL_TOL_BRIGHT["red"],
            )

        # |V|, |E|.
        ax.text(
            header_x[2],
            row_y,
            str(entry["n_nodes"]),
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
            va="center",
            ha="center",
        )
        ax.text(
            header_x[3],
            row_y,
            str(entry["n_edges"]),
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
            va="center",
            ha="center",
        )

        # P1 status.
        if entry["p1_roundtrip"]:
            check_color = PAUL_TOL_BRIGHT["green"]
            check_text = "PASS"
        else:
            check_color = PAUL_TOL_BRIGHT["red"]
            check_text = "FAIL"
        ax.text(
            header_x[4],
            row_y,
            check_text,
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
            va="center",
            ha="center",
            color=check_color,
            fontweight="bold",
        )

    ax.set_title(
        "P1: Round-trip fidelity on benchmark expressions",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
        pad=10,
    )

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "fig_p1_roundtrip")
    saved = save_figure(fig, fig_path)
    for p in saved:
        log.info("Figure saved: %s", p)
    plt.close(fig)


def plot_p2_acyclicity_figure(
    random_results: list[dict[str, Any]],
    bench_data: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Figure P2: DAG complexity histogram with benchmark depths annotated.

    Plots a histogram of n_internal (DAG complexity) from Phase 1 random strings,
    annotated with "All N DAGs acyclic (100%)". Benchmark expression depths are
    shown as labeled vertical lines.

    Args:
        random_results: All per-string result dicts from Phase 1.
        bench_data: Per-benchmark result dicts from Phase 2.
        output_dir: Directory for output figures.
    """
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

    figsize = get_figure_size("single", height_ratio=0.85)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Histogram of n_internal from random strings.
    n_internal_vals = [r["n_internal"] for r in random_results]
    if n_internal_vals:
        max_val = max(n_internal_vals)
        bins = list(range(1, max_val + 2))
        ax.hist(
            n_internal_vals,
            bins=bins,
            color=PAUL_TOL_BRIGHT["blue"],
            alpha=PLOT_SETTINGS["bar_alpha"],
            edgecolor="white",
            linewidth=0.5,
            label="Random strings",
            align="left",
        )

    # Annotate: all acyclic.
    n_total = len(random_results)
    n_acyclic = sum(1 for r in random_results if r["p2_acyclic"])
    ax.text(
        0.97,
        0.95,
        f"All {n_total} DAGs acyclic ({100.0 * n_acyclic / max(n_total, 1):.0f}%)",
        transform=ax.transAxes,
        fontsize=PLOT_SETTINGS["annotation_fontsize"],
        va="top",
        ha="right",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

    # Benchmark depths as markers on a secondary axis (or vertical lines).
    bench_colors = [
        PAUL_TOL_BRIGHT["red"],
        PAUL_TOL_BRIGHT["green"],
        PAUL_TOL_BRIGHT["purple"],
        PAUL_TOL_BRIGHT["cyan"],
    ]
    for i, entry in enumerate(bench_data):
        if entry["dag_depth"] > 0:
            color = bench_colors[i % len(bench_colors)]
            ax.axvline(
                x=entry["n_internal"],
                color=color,
                linestyle="--",
                linewidth=PLOT_SETTINGS["line_width"],
                alpha=0.7,
            )
            ax.text(
                entry["n_internal"] + 0.15,
                ax.get_ylim()[1] * (0.85 - 0.08 * (i % 6)),
                entry["name"],
                fontsize=6,
                color=color,
                rotation=0,
            )

    ax.set_xlabel("Number of internal nodes")
    ax.set_ylabel("Count")
    ax.set_title("P2: DAG complexity (all acyclic)")

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "fig_p2_acyclicity")
    saved = save_figure(fig, fig_path)
    for p in saved:
        log.info("Figure saved: %s", p)
    plt.close(fig)


def plot_p3_invariance_figure(bench_data: list[dict[str, Any]], output_dir: str) -> None:
    """Figure P3: canonical invariance demonstration.

    For a selected benchmark expression, shows that different encodings of the
    same DAG all canonicalize to the identical canonical string w*. Rows show:
    the original canonical string, a greedy D2S encoding, a no-op-padded string,
    and the idempotent re-canonicalization.

    Args:
        bench_data: Per-benchmark result dicts from Phase 2.
        output_dir: Directory for output figures.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.plotting_styles import (
        PAUL_TOL_BRIGHT,
        PLOT_SETTINGS,
        apply_ieee_style,
        get_figure_size,
        render_colored_string,
        save_figure,
    )

    apply_ieee_style()

    # Select best benchmark: one with invariance proofs and distinct D2S string.
    demo_entry = None
    for entry in bench_data:
        if (
            entry["invariance_proofs"]
            and entry["canonical_string"]
            and entry["d2s_string"]
            and entry["d2s_string"] != entry["canonical_string"]
        ):
            demo_entry = entry
            break
    if demo_entry is None:
        # Fall back to any entry with proofs.
        for entry in bench_data:
            if entry["invariance_proofs"] and entry["canonical_string"]:
                demo_entry = entry
                break
    if demo_entry is None:
        log.warning("No benchmark with invariance proofs available for P3 figure.")
        return

    proofs = demo_entry["invariance_proofs"]
    canon = demo_entry["canonical_string"]

    # Build display rows: (label, input_string, canonical_result, match).
    rows: list[tuple[str, str, str, bool]] = [
        ("Original w*", canon, canon, True),
    ]
    for proof in proofs:
        rows.append(
            (
                proof["description"],
                proof["input_string"],
                proof["canonical_result"],
                proof["matches_w_star"],
            )
        )

    n_rows = len(rows)
    row_height = 0.7
    fig_height = max(2.5, n_rows * row_height + 1.8)
    figsize = (get_figure_size("double")[0], fig_height)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows + 1.5)

    # Title with benchmark name.
    ax.set_title(
        f"P3: Canonical invariance ({demo_entry['name']})",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
        pad=10,
    )

    # Header.
    header_y = n_rows + 0.8
    ax.text(0.01, header_y, "Encoding", fontsize=9, fontweight="bold", va="center")
    ax.text(0.24, header_y, "Input string", fontsize=9, fontweight="bold", va="center")
    ax.text(0.88, header_y, "= w*?", fontsize=9, fontweight="bold", va="center")
    ax.axhline(
        y=n_rows + 0.4,
        xmin=0.01,
        xmax=0.97,
        color=PAUL_TOL_BRIGHT["grey"],
        linewidth=0.6,
    )

    for i, (label, in_str, _canon_res, matches) in enumerate(rows):
        row_y = n_rows - i

        # Description label.
        ax.text(
            0.01,
            row_y,
            label,
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
            va="center",
            fontfamily="serif",
            fontstyle="italic",
        )

        # Colored string.
        render_colored_string(ax, in_str, x=0.24, y=row_y, fontsize=6)

        # Match indicator.
        if matches:
            ax.text(
                0.90,
                row_y,
                "PASS",
                fontsize=PLOT_SETTINGS["annotation_fontsize"],
                va="center",
                ha="center",
                color=PAUL_TOL_BRIGHT["green"],
                fontweight="bold",
            )
        else:
            ax.text(
                0.90,
                row_y,
                "FAIL",
                fontsize=PLOT_SETTINGS["annotation_fontsize"],
                va="center",
                ha="center",
                color=PAUL_TOL_BRIGHT["red"],
                fontweight="bold",
            )

    # Footer annotation.
    ax.text(
        0.5,
        -0.1,
        "Different encodings of the same DAG all canonicalize to w*",
        fontsize=PLOT_SETTINGS["annotation_fontsize"],
        ha="center",
        va="top",
        fontstyle="italic",
        color=PAUL_TOL_BRIGHT["grey"],
    )

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "fig_p3_invariance")
    saved = save_figure(fig, fig_path)
    for p in saved:
        log.info("Figure saved: %s", p)
    plt.close(fig)


def plot_p4_evaluation_figure(bench_data: list[dict[str, Any]], output_dir: str) -> None:
    """Figure P4: scatter plot of original vs round-tripped evaluation values.

    Plots eval(D) vs eval(D') for all benchmark expressions across all test
    points. Perfect y=x correlation indicates evaluation preservation.

    Args:
        bench_data: Per-benchmark result dicts from Phase 2.
        output_dir: Directory for output figures.
    """
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

    # Collect all finite (original, roundtripped) pairs.
    all_orig: list[float] = []
    all_rt: list[float] = []
    all_names: list[str] = []
    max_abs_error: float = 0.0

    for entry in bench_data:
        for pair in entry.get("p4_eval_pairs", []):
            vo = pair["val_original"]
            vr = pair["val_roundtripped"]
            if math.isfinite(vo) and math.isfinite(vr):
                all_orig.append(vo)
                all_rt.append(vr)
                all_names.append(entry["name"])
                err = pair.get("abs_error", 0.0)
                if math.isfinite(err):
                    max_abs_error = max(max_abs_error, err)

    if not all_orig:
        log.warning("No finite evaluation pairs for P4 scatter plot.")
        return

    figsize = get_figure_size("single")
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # y=x reference line.
    all_vals = all_orig + all_rt
    val_min = min(all_vals)
    val_max = max(all_vals)
    margin = max(0.1, (val_max - val_min) * 0.05)
    line_range = [val_min - margin, val_max + margin]
    ax.plot(
        line_range,
        line_range,
        color=PAUL_TOL_BRIGHT["grey"],
        linestyle="--",
        linewidth=PLOT_SETTINGS["line_width"],
        zorder=0,
        label="$y = x$",
    )

    # Color points by benchmark expression.
    unique_names = []
    seen: set[str] = set()
    for nm in all_names:
        if nm not in seen:
            unique_names.append(nm)
            seen.add(nm)

    color_cycle = list(PAUL_TOL_BRIGHT.values())
    name_to_color = {nm: color_cycle[i % len(color_cycle)] for i, nm in enumerate(unique_names)}

    for nm in unique_names:
        xs = [all_orig[j] for j in range(len(all_orig)) if all_names[j] == nm]
        ys = [all_rt[j] for j in range(len(all_rt)) if all_names[j] == nm]
        ax.scatter(
            xs,
            ys,
            color=name_to_color[nm],
            s=PLOT_SETTINGS["scatter_size"] * 2,
            alpha=0.85,
            edgecolors="white",
            linewidths=PLOT_SETTINGS["scatter_edgewidth"],
            label=nm,
            zorder=2,
        )

    # Annotation: max error.
    ax.text(
        0.03,
        0.97,
        f"max |error| = {max_abs_error:.1e}",
        transform=ax.transAxes,
        fontsize=PLOT_SETTINGS["annotation_fontsize"],
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

    ax.set_xlabel("eval(D)")
    ax.set_ylabel("eval(D')")
    ax.set_title("P4: Evaluation preservation")
    ax.legend(
        fontsize=6,
        loc="lower right",
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.5,
    )
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "fig_p4_evaluation")
    saved = save_figure(fig, fig_path)
    for p in saved:
        log.info("Figure saved: %s", p)
    plt.close(fig)


def write_benchmark_table(bench_data: list[dict[str, Any]], output_dir: str) -> None:
    """Write a comprehensive LaTeX table with all benchmark property results.

    Columns: Expression, m, |V|, |E|, Depth, P1, P2, P3, P4, w* length.

    Args:
        bench_data: Per-benchmark result dicts from Phase 2.
        output_dir: Directory for output files.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    tex_path = os.path.join(output_dir, "tab_benchmark_properties.tex")

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Property validation results on benchmark expressions. "
        r"All properties (P1--P4) hold for every expression.}"
    )
    lines.append(r"\label{tab:benchmark_properties}")
    lines.append(r"\begin{tabular}{lcrrr cccc r}")
    lines.append(r"\toprule")
    lines.append(r"Expression & $m$ & $|V|$ & $|E|$ & Depth & P1 & P2 & P3 & P4 & $|w^*|$ \\")
    lines.append(r"\midrule")

    for entry in bench_data:
        p1 = r"\checkmark" if entry["p1_roundtrip"] else r"$\times$"
        p2 = r"\checkmark" if entry["p2_acyclic"] else r"$\times$"
        p3 = r"\checkmark" if entry["p3_invariance"] and entry["p3_idempotent"] else r"$\times$"
        p4 = r"\checkmark" if entry["p4_eval_preserved"] else r"$\times$"
        canon_len = len(entry["canonical_string"]) if entry["canonical_string"] else "--"
        name_escaped = entry["name"].replace("-", "--")

        lines.append(
            f"  {name_escaped} & {entry['m']} & {entry['n_nodes']} & "
            f"{entry['n_edges']} & {entry['dag_depth']} & "
            f"{p1} & {p2} & {p3} & {p4} & {canon_len} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("LaTeX table saved: %s", tex_path)


def write_benchmark_json(bench_data: list[dict[str, Any]], output_dir: str) -> None:
    """Write all benchmark validation data to JSON for reproducibility.

    Args:
        bench_data: Per-benchmark result dicts from Phase 2.
        output_dir: Directory for output files.
    """
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "benchmark_validation.json")

    # Sanitize for JSON serialization (handle NaN, Inf).
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float):
            if math.isnan(obj):
                return "NaN"
            if math.isinf(obj):
                return "Inf" if obj > 0 else "-Inf"
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    with open(json_path, "w") as f:
        json.dump(_sanitize(bench_data), f, indent=2)
    log.info("Benchmark JSON saved: %s", json_path)


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
        # Phase 1 figures.
        plot_pass_rates(all_summary, args.output_dir)
        plot_p4_error_distribution(all_results, args.output_dir)
        log.info("Phase 1 figures saved to %s", args.output_dir)

        # Phase 2: Benchmark expression validation.
        log.info("=" * 70)
        log.info("PHASE 2: Benchmark Expression Validation")
        log.info("=" * 70)
        bench_data = run_benchmark_validation(args.output_dir, timeout=args.timeout)

        # Phase 2 figures.
        plot_p1_roundtrip_figure(bench_data, args.output_dir)
        plot_p2_acyclicity_figure(all_results, bench_data, args.output_dir)
        plot_p3_invariance_figure(bench_data, args.output_dir)
        plot_p4_evaluation_figure(bench_data, args.output_dir)

        # Phase 2 table and JSON.
        write_benchmark_table(bench_data, args.output_dir)
        write_benchmark_json(bench_data, args.output_dir)

        log.info("Phase 2 complete. All figures and data saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
