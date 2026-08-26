"""Experiment 2: Levenshtein distance-1 neighborhood analysis for canonical strings.

Enumerates ALL character-level strings at Levenshtein distance 1 from a canonical
string w*, parses valid ones into DAGs, canonicalizes them, and measures:
  - How many unique canonical forms exist in the distance-1 neighborhood
  - The redundancy rate (syntactic neighbors mapping to the same canonical form)
  - Breakdown by edit operation type (deletion, substitution, insertion)

This provides an illustrative figure for the arXiv paper showing that
canonicalization collapses many syntactic neighbors into the same equivalence
class, demonstrating the O(k!) search space reduction at the local level.

Mathematical justification:
    For a canonical string w* of length L over alphabet A with |A| = a,
    the Levenshtein ball of radius 1 has at most:
        L + L*(a-1) + (L+1)*a = L + La - L + La + a = 2La + a
    strings. Many of these are invalid IsalSR strings (e.g., bare label chars
    not preceded by V/v), and many valid ones map to the same canonical form.

Usage:
    python experiments/scripts/exp2_neighborhood.py \\
        --output-dir /tmp/exp2_neighborhood
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.plotting_styles import (  # noqa: E402
    PAUL_TOL_BRIGHT,
    PLOT_SETTINGS,
    apply_ieee_style,
    get_figure_size,
    save_figure,
)
from isalsr.adapters.sympy_adapter import SympyAdapter  # noqa: E402
from isalsr.core.canonical import (  # noqa: E402
    CanonicalTimeoutError,
    pruned_canonical_string,
)
from isalsr.core.node_types import LABEL_CHAR_MAP, OperationSet  # noqa: E402
from isalsr.core.string_to_dag import StringToDAG  # noqa: E402
from isalsr.errors import InvalidTokenError  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class NeighborRecord:
    """A single neighbor string and its canonicalization result."""

    neighbor_string: str
    operation_type: str  # "deletion", "substitution", "insertion"
    canonical_form: str | None  # None if invalid/timeout
    is_valid: bool
    timed_out: bool = False


@dataclass
class NeighborhoodResults:
    """Aggregated results for the neighborhood analysis."""

    expression_str: str
    canonical_string: str
    num_vars: int
    alphabet_size: int
    string_length: int
    total_generated: int
    n_valid: int
    n_unique_canonical: int
    n_same_as_original: int
    redundancy_rate: float
    n_timed_out: int

    # Per-operation breakdown
    deletion_total: int = 0
    deletion_valid: int = 0
    deletion_unique: int = 0
    substitution_total: int = 0
    substitution_valid: int = 0
    substitution_unique: int = 0
    insertion_total: int = 0
    insertion_valid: int = 0
    insertion_unique: int = 0

    # Per-canonical-form counts (canonical -> count)
    canonical_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "expression": self.expression_str,
            "canonical_string": self.canonical_string,
            "num_vars": self.num_vars,
            "alphabet_size": self.alphabet_size,
            "string_length": self.string_length,
            "total_generated": self.total_generated,
            "n_valid": self.n_valid,
            "n_unique_canonical": self.n_unique_canonical,
            "n_same_as_original": self.n_same_as_original,
            "redundancy_rate": self.redundancy_rate,
            "n_timed_out": self.n_timed_out,
            "per_operation": {
                "deletion": {
                    "total": self.deletion_total,
                    "valid": self.deletion_valid,
                    "unique_canonical": self.deletion_unique,
                },
                "substitution": {
                    "total": self.substitution_total,
                    "valid": self.substitution_valid,
                    "unique_canonical": self.substitution_unique,
                },
                "insertion": {
                    "total": self.insertion_total,
                    "valid": self.insertion_valid,
                    "unique_canonical": self.insertion_unique,
                },
            },
            "canonical_counts": dict(
                sorted(
                    self.canonical_counts.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
            ),
        }


# ======================================================================
# Core algorithm
# ======================================================================


def build_alphabet(allowed_ops: OperationSet) -> list[str]:
    """Build the character-level alphabet for neighbor generation.

    The alphabet is the union of single-char instructions and label characters
    from the allowed operation set. This includes 'c' both as an edge instruction
    and as the COS label char -- that is correct, since we enumerate at the
    character level and let the parser determine validity.

    Args:
        allowed_ops: The set of allowed operations.

    Returns:
        Sorted list of unique characters in the alphabet.
    """
    single_chars = set("NPnpCcWVv")
    label_chars = set(allowed_ops.label_chars)
    return sorted(single_chars | label_chars)


def generate_distance_1_neighbors(
    w_star: str,
    alphabet: list[str],
) -> list[tuple[str, str]]:
    """Generate ALL strings at Levenshtein distance 1 from w_star.

    Returns (neighbor_string, operation_type) pairs for every:
      - Deletion: remove one character
      - Substitution: replace one character with another from the alphabet
      - Insertion: insert one character from the alphabet at any position

    Args:
        w_star: The canonical string to generate neighbors for.
        alphabet: The valid character alphabet.

    Returns:
        List of (neighbor_string, operation_type) pairs.
    """
    neighbors: list[tuple[str, str]] = []
    length = len(w_star)

    # Deletions: L strings
    for i in range(length):
        neighbor = w_star[:i] + w_star[i + 1 :]
        neighbors.append((neighbor, "deletion"))

    # Substitutions: L * (|A| - 1) strings
    for i in range(length):
        for c in alphabet:
            if c != w_star[i]:
                neighbor = w_star[:i] + c + w_star[i + 1 :]
                neighbors.append((neighbor, "substitution"))

    # Insertions: (L + 1) * |A| strings
    for i in range(length + 1):
        for c in alphabet:
            neighbor = w_star[:i] + c + w_star[i:]
            neighbors.append((neighbor, "insertion"))

    return neighbors


def try_parse_and_canonicalize(
    string: str,
    num_vars: int,
    allowed_ops: OperationSet,
    canon_timeout: float,
) -> tuple[str | None, bool, bool]:
    """Attempt to parse a string and compute its canonical form.

    Args:
        string: The IsalSR instruction string to parse.
        num_vars: Number of input variables.
        allowed_ops: Allowed operation set.
        canon_timeout: Timeout in seconds for canonicalization.

    Returns:
        Tuple of (canonical_form_or_None, is_valid, timed_out).
        is_valid is True if the string parses to a non-trivial DAG.
    """
    try:
        dag = StringToDAG(string, num_vars, allowed_ops).run()
    except (InvalidTokenError, ValueError, IndexError, KeyError):
        return None, False, False

    # Skip VAR-only DAGs (no internal nodes).
    var_count = len(dag.var_nodes())
    if dag.node_count <= var_count:
        return None, False, False

    # Skip DAGs with no edges (disconnected internal nodes are meaningless).
    if dag.edge_count == 0:
        return None, False, False

    try:
        canon = pruned_canonical_string(dag, timeout=canon_timeout)
        return canon, True, False
    except CanonicalTimeoutError:
        return None, True, True
    except Exception:
        return None, False, False


def analyze_neighborhood(
    expression_str: str,
    num_vars: int,
    ops_str: str,
    canon_timeout: float,
) -> NeighborhoodResults:
    """Run the full neighborhood analysis for a given expression.

    Args:
        expression_str: SymPy expression as a string (e.g., "sin(x_0) + cos(x_0)").
        num_vars: Number of input variables.
        ops_str: Comma-separated label chars for the operation set.
        canon_timeout: Timeout per canonicalization in seconds.

    Returns:
        NeighborhoodResults with all computed metrics.
    """
    import sympy

    # Parse expression.
    local_symbols = {f"x_{i}": sympy.Symbol(f"x_{i}") for i in range(num_vars)}
    # Add common sympy functions to the namespace for sympify.
    local_ns = {
        **local_symbols,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "exp": sympy.exp,
        "log": sympy.log,
        "sqrt": sympy.sqrt,
        "Abs": sympy.Abs,
    }
    expr = sympy.sympify(expression_str, locals=local_ns)
    variables = [sympy.Symbol(f"x_{i}") for i in range(num_vars)]

    log.info("Expression: %s", expr)
    log.info("Variables: %s", variables)

    # Build the operation set from label chars.
    ops_node_types = frozenset(LABEL_CHAR_MAP[ch] for ch in ops_str.split(","))
    allowed_ops = OperationSet(ops_node_types)
    log.info("Allowed ops: %s", allowed_ops)
    log.info("Label chars: %s", sorted(allowed_ops.label_chars))

    # Step 1: Build DAG from SymPy expression.
    adapter = SympyAdapter()
    dag = adapter.from_sympy(expr, variables)
    log.info(
        "DAG: %d nodes, %d edges",
        dag.node_count,
        dag.edge_count,
    )

    # Step 2: Compute canonical string.
    w_star = pruned_canonical_string(dag)
    log.info("Canonical string w*: %s (length %d)", w_star, len(w_star))

    # Step 3: Build alphabet.
    alphabet = build_alphabet(allowed_ops)
    log.info("Alphabet: %s (size %d)", "".join(alphabet), len(alphabet))

    # Step 4: Generate all distance-1 neighbors.
    neighbors = generate_distance_1_neighbors(w_star, alphabet)
    log.info("Total distance-1 neighbors: %d", len(neighbors))

    # Step 5: Parse and canonicalize each neighbor.
    records: list[NeighborRecord] = []
    valid_canonical_forms: list[str] = []
    timed_out_count = 0

    t0 = time.monotonic()
    for idx, (neighbor_str, op_type) in enumerate(neighbors):
        if (idx + 1) % 500 == 0:
            elapsed = time.monotonic() - t0
            log.info(
                "  Progress: %d / %d (%.1f%%, %.1fs elapsed)",
                idx + 1,
                len(neighbors),
                100.0 * (idx + 1) / len(neighbors),
                elapsed,
            )

        canon, is_valid, timed_out = try_parse_and_canonicalize(
            neighbor_str, num_vars, allowed_ops, canon_timeout
        )

        record = NeighborRecord(
            neighbor_string=neighbor_str,
            operation_type=op_type,
            canonical_form=canon,
            is_valid=is_valid,
            timed_out=timed_out,
        )
        records.append(record)

        if is_valid and canon is not None:
            valid_canonical_forms.append(canon)
        if timed_out:
            timed_out_count += 1

    elapsed = time.monotonic() - t0
    log.info("Neighborhood analysis complete in %.1fs", elapsed)

    # Step 6: Compute metrics.
    n_valid = sum(1 for r in records if r.is_valid and r.canonical_form is not None)
    canonical_counter = Counter(valid_canonical_forms)
    n_unique_canonical = len(canonical_counter)
    n_same_as_original = canonical_counter.get(w_star, 0)
    redundancy_rate = 1.0 - (n_unique_canonical / max(n_valid, 1))

    # Per-operation breakdown.
    op_types = ["deletion", "substitution", "insertion"]
    op_totals: dict[str, int] = {}
    op_valid: dict[str, int] = {}
    op_unique: dict[str, int] = {}

    for op in op_types:
        op_records = [r for r in records if r.operation_type == op]
        op_totals[op] = len(op_records)
        op_valid_records = [r for r in op_records if r.is_valid and r.canonical_form is not None]
        op_valid[op] = len(op_valid_records)
        op_unique[op] = len(set(r.canonical_form for r in op_valid_records))

    results = NeighborhoodResults(
        expression_str=str(expr),
        canonical_string=w_star,
        num_vars=num_vars,
        alphabet_size=len(alphabet),
        string_length=len(w_star),
        total_generated=len(neighbors),
        n_valid=n_valid,
        n_unique_canonical=n_unique_canonical,
        n_same_as_original=n_same_as_original,
        redundancy_rate=redundancy_rate,
        n_timed_out=timed_out_count,
        deletion_total=op_totals["deletion"],
        deletion_valid=op_valid["deletion"],
        deletion_unique=op_unique["deletion"],
        substitution_total=op_totals["substitution"],
        substitution_valid=op_valid["substitution"],
        substitution_unique=op_unique["substitution"],
        insertion_total=op_totals["insertion"],
        insertion_valid=op_valid["insertion"],
        insertion_unique=op_unique["insertion"],
        canonical_counts=dict(canonical_counter),
    )

    log.info("--- Results ---")
    log.info("Total generated:     %d", results.total_generated)
    log.info("Valid DAGs:          %d", results.n_valid)
    log.info("Unique canonical:    %d", results.n_unique_canonical)
    log.info("Same as original:    %d", results.n_same_as_original)
    log.info("Redundancy rate:     %.4f", results.redundancy_rate)
    log.info("Timed out:           %d", results.n_timed_out)

    return results


# ======================================================================
# Figures
# ======================================================================


def plot_neighborhood_bar(
    results: NeighborhoodResults,
    output_dir: str,
) -> None:
    """Generate bar chart: total / valid / unique / same-as-original.

    Args:
        results: The neighborhood analysis results.
        output_dir: Directory to save figures.
    """
    apply_ieee_style()

    fig, ax = plt.subplots(figsize=get_figure_size("single", height_ratio=0.85))

    categories = [
        "Total\nneighbors",
        "Valid\nDAGs",
        "Unique\ncanonical",
        "Same as\noriginal",
    ]
    values = [
        results.total_generated,
        results.n_valid,
        results.n_unique_canonical,
        results.n_same_as_original,
    ]
    colors = [
        PAUL_TOL_BRIGHT["blue"],
        PAUL_TOL_BRIGHT["cyan"],
        PAUL_TOL_BRIGHT["green"],
        PAUL_TOL_BRIGHT["yellow"],
    ]

    bars = ax.bar(categories, values, color=colors, width=0.6, edgecolor="white")

    # Add value labels on top of bars.
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(val),
            ha="center",
            va="bottom",
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
            fontweight="bold",
        )

    ax.set_ylabel("Count")
    ax.set_title(
        f"Distance-1 neighborhood of $w^*$\n"
        f"(|$w^*$| = {results.string_length}, "
        f"|$\\Sigma$| = {results.alphabet_size})",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
    )

    # Add redundancy annotation.
    ax.annotate(
        f"Redundancy: {results.redundancy_rate:.1%}",
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=PLOT_SETTINGS["annotation_fontsize"],
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=PAUL_TOL_BRIGHT["grey"],
            alpha=0.3,
        ),
    )

    ax.set_ylim(0, max(values) * 1.15)
    fig.tight_layout()

    path = os.path.join(output_dir, "fig_neighborhood_bar")
    saved = save_figure(fig, path)
    plt.close(fig)
    log.info("Saved bar chart: %s", saved)


def plot_neighborhood_by_op(
    results: NeighborhoodResults,
    output_dir: str,
) -> None:
    """Generate grouped bar chart: breakdown by operation type.

    Three groups (deletion, substitution, insertion), each with three bars:
    total, valid, unique canonical.

    Args:
        results: The neighborhood analysis results.
        output_dir: Directory to save figures.
    """
    apply_ieee_style()

    fig, ax = plt.subplots(figsize=get_figure_size("single", height_ratio=0.85))

    import numpy as np

    groups = ["Deletion", "Substitution", "Insertion"]
    total_vals = [
        results.deletion_total,
        results.substitution_total,
        results.insertion_total,
    ]
    valid_vals = [
        results.deletion_valid,
        results.substitution_valid,
        results.insertion_valid,
    ]
    unique_vals = [
        results.deletion_unique,
        results.substitution_unique,
        results.insertion_unique,
    ]

    x = np.arange(len(groups))
    bar_width = 0.22

    bars1 = ax.bar(
        x - bar_width,
        total_vals,
        bar_width,
        label="Total",
        color=PAUL_TOL_BRIGHT["blue"],
        edgecolor="white",
    )
    bars2 = ax.bar(
        x,
        valid_vals,
        bar_width,
        label="Valid DAGs",
        color=PAUL_TOL_BRIGHT["cyan"],
        edgecolor="white",
    )
    bars3 = ax.bar(
        x + bar_width,
        unique_vals,
        bar_width,
        label="Unique canonical",
        color=PAUL_TOL_BRIGHT["green"],
        edgecolor="white",
    )

    # Add value labels.
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    str(int(height)),
                    ha="center",
                    va="bottom",
                    fontsize=PLOT_SETTINGS["annotation_fontsize"] - 1,
                )

    ax.set_xlabel("Edit operation")
    ax.set_ylabel("Count")
    ax.set_title("Neighborhood by edit type")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend(
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        loc="upper left",
    )

    ax.set_ylim(0, max(total_vals) * 1.2)
    fig.tight_layout()

    path = os.path.join(output_dir, "fig_neighborhood_by_op")
    saved = save_figure(fig, path)
    plt.close(fig)
    log.info("Saved by-op chart: %s", saved)


def write_latex_table(
    results: NeighborhoodResults,
    output_dir: str,
) -> None:
    """Write LaTeX summary table.

    Writes the table directly rather than through save_latex_table to avoid
    double-escaping of LaTeX math commands in cell values.

    Args:
        results: The neighborhood analysis results.
        output_dir: Directory to save the table.
    """
    path = os.path.join(output_dir, "tab_neighborhood.tex")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Distance-1 neighborhood analysis of a canonical IsalSR string.}",
        r"\label{tab:neighborhood}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        rf"Expression & \verb|{results.expression_str}| \\",
        rf"$|w^*|$ (string length) & {results.string_length} \\",
        rf"$|\Sigma|$ (alphabet size) & {results.alphabet_size} \\",
        rf"Total neighbors & {results.total_generated} \\",
        rf"Valid DAGs & {results.n_valid} \\",
        rf"Unique canonical forms & {results.n_unique_canonical} \\",
        rf"Same as original $w^*$ & {results.n_same_as_original} \\",
        rf"Redundancy rate & {results.redundancy_rate:.4f} \\",
        rf"Timed out & {results.n_timed_out} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    log.info("Saved LaTeX table: %s", path)


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Exp2: Levenshtein distance-1 neighborhood analysis for canonical strings.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/media/mpascual/Sandisk2TB/research/isalsr/results/arXiv_benchmarking/exp2_neighborhood",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--expression",
        type=str,
        default="sin(x_0) + cos(x_0)",
        help="SymPy expression string (e.g., 'sin(x_0) + cos(x_0)').",
    )
    parser.add_argument(
        "--num-vars",
        type=int,
        default=1,
        help="Number of input variables.",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default="+,*,-,/,s,c,e,l",
        help="Comma-separated label chars for allowed operations.",
    )
    parser.add_argument(
        "--canon-timeout",
        type=float,
        default=5.0,
        help="Timeout per canonicalization in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the neighborhood analysis experiment."""
    args = parse_args()

    log.info("=" * 60)
    log.info("Exp2: Distance-1 Neighborhood Analysis")
    log.info("=" * 60)
    log.info("Expression:    %s", args.expression)
    log.info("Num vars:      %d", args.num_vars)
    log.info("Ops:           %s", args.ops)
    log.info("Canon timeout: %.1fs", args.canon_timeout)
    log.info("Output dir:    %s", args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # Run analysis.
    results = analyze_neighborhood(
        expression_str=args.expression,
        num_vars=args.num_vars,
        ops_str=args.ops,
        canon_timeout=args.canon_timeout,
    )

    # Save JSON results.
    json_path = os.path.join(args.output_dir, "neighborhood_analysis.json")
    with open(json_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    log.info("Saved JSON: %s", json_path)

    # Generate figures.
    plot_neighborhood_bar(results, args.output_dir)
    plot_neighborhood_by_op(results, args.output_dir)

    # Generate LaTeX table.
    write_latex_table(results, args.output_dir)

    log.info("=" * 60)
    log.info("Exp2 complete. All outputs in: %s", args.output_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
