# ruff: noqa: N803, N806
"""Experiment 1: Shortest-path distance between expression DAGs via canonical strings.

Illustrative experiment for the arXiv paper demonstrating that the canonical string
representation induces a meaningful metric space on expression DAGs via Levenshtein
distance. For several concrete expression pairs, we:

1. Build DAGs from SymPy expressions using ``SympyAdapter.from_sympy()``
2. Compute canonical strings using ``pruned_canonical_string()``
3. Compute Levenshtein distance with full backtrace (showing edit operations)
4. Generate output: JSON data + LaTeX table + colored-string figure

Mathematical justification:
    The Levenshtein distance on canonical strings is a metric on labeled DAGs:
    d(D1, D2) = lev(w*_D1, w*_D2) satisfies symmetry, triangle inequality,
    and identity of indiscernibles (since canonical strings are complete invariants).

Usage:
    cd /home/mpascual/research/code/IsalSR
    ~/.conda/envs/isalsr/bin/python experiments/scripts/exp1_shortest_path.py \\
        --output-dir /tmp/exp1_shortest_path
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sympy import Symbol, cos, exp, log, sin  # noqa: E402

# Ensure project root is on sys.path for imports
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
from isalsr.adapters.sympy_adapter import SympyAdapter  # noqa: E402
from isalsr.core.canonical import pruned_canonical_string  # noqa: E402

logger = logging.getLogger(__name__)

# =============================================================================
# Edit operation colors (colorblind-friendly, Paul Tol inspired)
# =============================================================================

EDIT_COLORS: dict[str, str] = {
    "match": PAUL_TOL_BRIGHT["green"],  # #228833
    "substitute": PAUL_TOL_BRIGHT["yellow"],  # #CCBB44
    "insert": PAUL_TOL_BRIGHT["blue"],  # #4477AA
    "delete": PAUL_TOL_BRIGHT["red"],  # #EE6677
}


# =============================================================================
# Levenshtein with backtrace
# =============================================================================


def levenshtein_with_backtrace(
    s: str, t: str
) -> tuple[int, list[tuple[str, int, int, str] | tuple[str, int, int, str, str]]]:
    """Compute Levenshtein distance with full edit operation backtrace.

    Uses the standard O(n*m) dynamic programming approach with the full matrix
    retained for backtrace. The backtrace walks from (len_s, len_t) to (0, 0)
    to reconstruct the optimal edit sequence.

    Args:
        s: Source string.
        t: Target string.

    Returns:
        Tuple of (distance, operations) where operations is a list of tuples:
            - ("match", i, j, char) -- characters match at positions i, j
            - ("substitute", i, j, old_char, new_char) -- substitution
            - ("insert", i, j, char) -- insert char from t at position j
            - ("delete", i, j, char) -- delete char from s at position i
        Operations are listed in order from the beginning of the strings.
    """
    n = len(s)
    m = len(t)

    # Build full DP matrix.
    dp: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # delete from s
                dp[i][j - 1] + 1,  # insert from t
                dp[i - 1][j - 1] + cost,  # match or substitute
            )

    # Backtrace from (n, m) to (0, 0).
    ops: list[tuple[str, int, int, str] | tuple[str, int, int, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s[i - 1] == t[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("match", i - 1, j - 1, s[i - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("substitute", i - 1, j - 1, s[i - 1], t[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("insert", i, j - 1, t[j - 1]))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("delete", i - 1, j, s[i - 1]))
            i -= 1
        else:
            # Should not happen with a correct DP matrix.
            raise RuntimeError(f"Backtrace stuck at i={i}, j={j}")

    ops.reverse()
    return dp[n][m], ops


# =============================================================================
# Expression pair definitions
# =============================================================================


def _build_expression_pairs() -> list[dict]:
    """Build the list of expression pairs for the experiment.

    Returns:
        List of dicts with keys: pair_id, expr_a_str, expr_b_str, expr_a, expr_b,
        variables, num_vars.
    """
    x = Symbol("x_0")
    y = Symbol("x_1")

    pairs = [
        {
            "pair_id": "A",
            "expr_a_str": r"$\sin(x)$",
            "expr_b_str": r"$x^2 + x$",
            "expr_a": sin(x),
            "expr_b": x**2 + x,
            "variables": [x],
            "num_vars": 1,
            "description": "Structurally different, moderate distance",
        },
        {
            "pair_id": "B",
            "expr_a_str": r"$\sin(x) + \cos(x)$",
            "expr_b_str": r"$\sin(x) \cdot \cos(x)$",
            "expr_a": sin(x) + cos(x),
            "expr_b": sin(x) * cos(x),
            "variables": [x],
            "num_vars": 1,
            "description": "Same operands, different combinator",
        },
        {
            "pair_id": "C",
            "expr_a_str": r"$x^2$",
            "expr_b_str": r"$x^3 + x^2 + x$",
            "expr_a": x**2,
            "expr_b": x**3 + x**2 + x,
            "variables": [x],
            "num_vars": 1,
            "description": "Containment relationship",
        },
        {
            "pair_id": "D",
            "expr_a_str": r"$\exp(x)$",
            "expr_b_str": r"$\log(x)$",
            "expr_a": exp(x),
            "expr_b": log(x),
            "variables": [x],
            "num_vars": 1,
            "description": "Same structure, different unary op",
        },
        {
            "pair_id": "E",
            "expr_a_str": r"$\sin(x) + y^2$",
            "expr_b_str": r"$\cos(x) \cdot y$",
            "expr_a": sin(x) + y**2,
            "expr_b": cos(x) * y,
            "variables": [x, y],
            "num_vars": 2,
            "description": "Mixed, 2-variable",
        },
    ]
    return pairs


# =============================================================================
# Core analysis
# =============================================================================


def analyze_pairs(
    pairs: list[dict],
) -> list[dict]:
    """Analyze each expression pair: build DAGs, compute canonical strings, Levenshtein.

    Args:
        pairs: List of pair definitions from ``_build_expression_pairs()``.

    Returns:
        List of result dicts, one per pair, with all computed data.
    """
    adapter = SympyAdapter()
    results: list[dict] = []

    for pair in pairs:
        pair_id = pair["pair_id"]
        logger.info(
            "Pair %s: %s vs %s",
            pair_id,
            pair["expr_a_str"],
            pair["expr_b_str"],
        )

        # Build DAGs.
        dag_a = adapter.from_sympy(pair["expr_a"], pair["variables"])
        dag_b = adapter.from_sympy(pair["expr_b"], pair["variables"])

        # Compute canonical strings.
        canon_a = pruned_canonical_string(dag_a)
        canon_b = pruned_canonical_string(dag_b)

        logger.info("  w*_A = %r (len=%d)", canon_a, len(canon_a))
        logger.info("  w*_B = %r (len=%d)", canon_b, len(canon_b))

        # Compute Levenshtein with backtrace.
        distance, operations = levenshtein_with_backtrace(canon_a, canon_b)
        logger.info("  d(A, B) = %d", distance)

        # Serialize operations for JSON.
        ops_serialized = []
        for op in operations:
            if op[0] == "match":
                ops_serialized.append(
                    {
                        "type": "match",
                        "i": op[1],
                        "j": op[2],
                        "char": op[3],
                    }
                )
            elif op[0] == "substitute":
                ops_serialized.append(
                    {
                        "type": "substitute",
                        "i": op[1],
                        "j": op[2],
                        "old_char": op[3],
                        "new_char": op[4],
                    }
                )
            elif op[0] == "insert":
                ops_serialized.append(
                    {
                        "type": "insert",
                        "i": op[1],
                        "j": op[2],
                        "char": op[3],
                    }
                )
            elif op[0] == "delete":
                ops_serialized.append(
                    {
                        "type": "delete",
                        "i": op[1],
                        "j": op[2],
                        "char": op[3],
                    }
                )

        result = {
            "pair_id": pair_id,
            "expr_a_str": pair["expr_a_str"],
            "expr_b_str": pair["expr_b_str"],
            "description": pair["description"],
            "num_vars": pair["num_vars"],
            "canonical_a": canon_a,
            "canonical_b": canon_b,
            "len_a": len(canon_a),
            "len_b": len(canon_b),
            "nodes_a": dag_a.node_count,
            "nodes_b": dag_b.node_count,
            "edges_a": dag_a.edge_count,
            "edges_b": dag_b.edge_count,
            "distance": distance,
            "operations": ops_serialized,
        }
        results.append(result)

    return results


# =============================================================================
# JSON output
# =============================================================================


def save_json(results: list[dict], output_dir: str) -> str:
    """Save analysis results to JSON.

    Args:
        results: List of result dicts from ``analyze_pairs()``.
        output_dir: Output directory.

    Returns:
        Path to the saved JSON file.
    """
    path = os.path.join(output_dir, "shortest_path_examples.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("JSON saved: %s", path)
    return path


# =============================================================================
# LaTeX table
# =============================================================================


def save_table(results: list[dict], output_dir: str) -> str:
    """Save a LaTeX table summarizing all pairs.

    Args:
        results: List of result dicts.
        output_dir: Output directory.

    Returns:
        Path to the saved .tex file.
    """
    rows = []
    for r in results:
        rows.append(
            {
                "Pair": r["pair_id"],
                "Expression A": r["expr_a_str"],
                "Expression B": r["expr_b_str"],
                "$|w^*_A|$": r["len_a"],
                "$|w^*_B|$": r["len_b"],
                "$d(A, B)$": r["distance"],
                "Variables": r["num_vars"],
            }
        )
    df = pd.DataFrame(rows)

    path = os.path.join(output_dir, "tab_shortest_path.tex")
    # Write directly with escape=False since columns and cells contain LaTeX math.
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    latex = df.to_latex(
        index=False,
        escape=False,
        caption="Levenshtein edit distances between canonical strings of expression pairs.",
        label="tab:shortest_path",
        position="htbp",
    )
    with open(path, "w") as f:
        f.write(latex)
    logger.info("LaTeX table saved: %s", path)
    return path


# =============================================================================
# Figure: colored canonical strings with edit operations
# =============================================================================


def _render_aligned_edit_row(
    ax: plt.Axes,
    canon_a: str,
    canon_b: str,
    operations: list[dict],
    y_center: float,
    fontsize: float,
) -> None:
    """Render an aligned pair of canonical strings with edit operation highlighting.

    For each character position in both strings, we color the background according
    to the edit operation type (match/substitute/delete/insert). The character
    itself is colored per TOKEN_COLORS (the IsalSR token color scheme).

    The layout is:
        Top row: source string (canon_a) -- deletions and matches/subs shown
        Bottom row: target string (canon_b) -- insertions and matches/subs shown

    Args:
        ax: Matplotlib axes (uses axes coordinates via ax.transAxes).
        canon_a: Canonical string of expression A.
        canon_b: Canonical string of expression B.
        operations: List of operation dicts from the backtrace.
        y_center: Vertical center position in axes coordinates [0, 1].
        fontsize: Font size for characters.
    """
    # Build character-level alignment columns.
    # Each column represents one alignment position.
    columns: list[dict] = []
    for op in operations:
        col: dict = {"type": op["type"]}
        if op["type"] == "match":
            col["top_char"] = op["char"]
            col["bot_char"] = op["char"]
        elif op["type"] == "substitute":
            col["top_char"] = op["old_char"]
            col["bot_char"] = op["new_char"]
        elif op["type"] == "delete":
            col["top_char"] = op["char"]
            col["bot_char"] = None  # gap
        elif op["type"] == "insert":
            col["top_char"] = None  # gap
            col["bot_char"] = op["char"]
        columns.append(col)

    n_cols = len(columns)
    if n_cols == 0:
        return

    # Layout parameters in axes fraction coordinates.
    char_width = 0.7 / max(n_cols, 1)  # occupy 70% of axes width
    char_width = min(char_width, 0.045)  # cap width for readability
    x_start = 0.5 - (n_cols * char_width) / 2  # center horizontally

    row_height = 0.12
    gap = 0.02
    y_top = y_center + gap / 2
    y_bot = y_center - gap / 2 - row_height

    bg_alpha = 0.25
    bg_height = row_height * 1.1

    for col_idx, col_data in enumerate(columns):
        x = x_start + col_idx * char_width
        op_type = col_data["type"]
        bg_color = EDIT_COLORS.get(op_type, "#BBBBBB")

        # Top row (source string).
        top_char = col_data.get("top_char")
        if top_char is not None:
            # Background highlight.
            bg_rect = mpatches.FancyBboxPatch(
                (x - char_width * 0.05, y_top - bg_height * 0.1),
                char_width * 0.9,
                bg_height,
                boxstyle="round,pad=0.005",
                facecolor=bg_color,
                alpha=bg_alpha,
                edgecolor="none",
                transform=ax.transAxes,
                zorder=1,
            )
            ax.add_patch(bg_rect)
            # Character text.
            ax.text(
                x + char_width * 0.4,
                y_top + row_height * 0.4,
                top_char,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontfamily="monospace",
                fontweight="bold",
                color="#333333",
                transform=ax.transAxes,
                zorder=2,
            )
        else:
            # Gap indicator for insertions.
            bg_rect = mpatches.FancyBboxPatch(
                (x - char_width * 0.05, y_top - bg_height * 0.1),
                char_width * 0.9,
                bg_height,
                boxstyle="round,pad=0.005",
                facecolor=EDIT_COLORS["insert"],
                alpha=bg_alpha * 0.5,
                edgecolor="none",
                transform=ax.transAxes,
                zorder=1,
            )
            ax.add_patch(bg_rect)
            ax.text(
                x + char_width * 0.4,
                y_top + row_height * 0.4,
                "-",
                ha="center",
                va="center",
                fontsize=fontsize * 0.8,
                fontfamily="monospace",
                color="#AAAAAA",
                transform=ax.transAxes,
                zorder=2,
            )

        # Bottom row (target string).
        bot_char = col_data.get("bot_char")
        if bot_char is not None:
            bg_rect = mpatches.FancyBboxPatch(
                (x - char_width * 0.05, y_bot - bg_height * 0.1),
                char_width * 0.9,
                bg_height,
                boxstyle="round,pad=0.005",
                facecolor=bg_color,
                alpha=bg_alpha,
                edgecolor="none",
                transform=ax.transAxes,
                zorder=1,
            )
            ax.add_patch(bg_rect)
            ax.text(
                x + char_width * 0.4,
                y_bot + row_height * 0.4,
                bot_char,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontfamily="monospace",
                fontweight="bold",
                color="#333333",
                transform=ax.transAxes,
                zorder=2,
            )
        else:
            # Gap indicator for deletions.
            bg_rect = mpatches.FancyBboxPatch(
                (x - char_width * 0.05, y_bot - bg_height * 0.1),
                char_width * 0.9,
                bg_height,
                boxstyle="round,pad=0.005",
                facecolor=EDIT_COLORS["delete"],
                alpha=bg_alpha * 0.5,
                edgecolor="none",
                transform=ax.transAxes,
                zorder=1,
            )
            ax.add_patch(bg_rect)
            ax.text(
                x + char_width * 0.4,
                y_bot + row_height * 0.4,
                "-",
                ha="center",
                va="center",
                fontsize=fontsize * 0.8,
                fontfamily="monospace",
                color="#AAAAAA",
                transform=ax.transAxes,
                zorder=2,
            )


def generate_figure(results: list[dict], output_dir: str) -> str:
    """Generate the main figure showing expression pairs with edit operations.

    For each pair, displays:
        - Expression labels (A and B with math notation)
        - Canonical strings aligned with edit operation highlighting
        - Levenshtein distance annotation

    Args:
        results: List of result dicts from ``analyze_pairs()``.
        output_dir: Output directory.

    Returns:
        Base path of saved figure (without extension).
    """
    apply_ieee_style()

    n_pairs = len(results)
    row_height = 1.4  # inches per pair row
    fig_width, _ = get_figure_size("double")
    fig_height = max(row_height * n_pairs + 1.2, 3.0)

    fig, axes = plt.subplots(
        n_pairs,
        1,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    for idx, result in enumerate(results):
        ax = axes[idx, 0]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        pair_id = result["pair_id"]
        canon_a = result["canonical_a"]
        canon_b = result["canonical_b"]
        distance = result["distance"]

        # Left column: expression labels and pair ID.
        ax.text(
            0.0,
            0.82,
            f"Pair {pair_id}",
            ha="left",
            va="center",
            fontsize=PLOT_SETTINGS["axes_titlesize"],
            fontweight="bold",
            transform=ax.transAxes,
        )

        ax.text(
            0.0,
            0.62,
            f"A: {result['expr_a_str']}",
            ha="left",
            va="center",
            fontsize=PLOT_SETTINGS["font_size"],
            transform=ax.transAxes,
        )
        ax.text(
            0.0,
            0.38,
            f"B: {result['expr_b_str']}",
            ha="left",
            va="center",
            fontsize=PLOT_SETTINGS["font_size"],
            transform=ax.transAxes,
        )

        # Right: distance badge.
        ax.text(
            0.98,
            0.82,
            f"$d = {distance}$",
            ha="right",
            va="center",
            fontsize=PLOT_SETTINGS["axes_titlesize"],
            fontweight="bold",
            color=PAUL_TOL_BRIGHT["red"],
            transform=ax.transAxes,
        )

        # Center: aligned edit operation visualization.
        _render_aligned_edit_row(
            ax,
            canon_a,
            canon_b,
            result["operations"],
            y_center=0.5,
            fontsize=PLOT_SETTINGS["annotation_fontsize"],
        )

        # Canonical string labels.
        ax.text(
            0.13,
            0.15,
            f"$w^*_A$: {canon_a!r}  (len={result['len_a']})",
            ha="left",
            va="center",
            fontsize=PLOT_SETTINGS["annotation_fontsize"] - 1,
            fontfamily="monospace",
            color="#555555",
            transform=ax.transAxes,
        )
        ax.text(
            0.13,
            0.03,
            f"$w^*_B$: {canon_b!r}  (len={result['len_b']})",
            ha="left",
            va="center",
            fontsize=PLOT_SETTINGS["annotation_fontsize"] - 1,
            fontfamily="monospace",
            color="#555555",
            transform=ax.transAxes,
        )

        # Horizontal separator (except for last row).
        if idx < n_pairs - 1:
            ax.plot(
                [0.02, 0.98],
                [-0.05, -0.05],
                color="0.7",
                linewidth=0.5,
                transform=ax.transAxes,
                clip_on=False,
            )

    # Legend for edit operation colors.
    legend_handles = [
        mpatches.Patch(facecolor=EDIT_COLORS["match"], alpha=0.6, label="Match"),
        mpatches.Patch(facecolor=EDIT_COLORS["substitute"], alpha=0.6, label="Substitute"),
        mpatches.Patch(facecolor=EDIT_COLORS["insert"], alpha=0.6, label="Insert"),
        mpatches.Patch(facecolor=EDIT_COLORS["delete"], alpha=0.6, label="Delete"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        "Levenshtein Edit Distance on Canonical Strings",
        fontsize=PLOT_SETTINGS["axes_titlesize"] + 1,
        fontweight="bold",
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    output_path = os.path.join(output_dir, "fig_shortest_path")
    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Figure saved: %s", output_path)
    return output_path


# =============================================================================
# CLI and main
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Experiment 1: Shortest-path distance between expression DAGs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/media/mpascual/Sandisk2TB/research/isalsr/results/arXiv_benchmarking/exp1_shortest_path",
        help="Directory for output files (JSON, figure, table).",
    )
    parser.add_argument(
        "--num-vars",
        type=int,
        default=1,
        help="Default number of variables (note: some pairs use 2 variables).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the shortest-path distance experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Output directory: %s", output_dir)

    # Build expression pairs.
    pairs = _build_expression_pairs()
    logger.info("Analyzing %d expression pairs", len(pairs))

    # Analyze all pairs.
    results = analyze_pairs(pairs)

    # Save JSON.
    save_json(results, output_dir)

    # Save LaTeX table.
    save_table(results, output_dir)

    # Generate figure.
    generate_figure(results, output_dir)

    # Print summary table to log.
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info("%-6s %-25s %-25s %5s %5s %5s", "Pair", "Expr A", "Expr B", "|A|", "|B|", "d")
    logger.info("-" * 60)
    for r in results:
        logger.info(
            "%-6s %-25s %-25s %5d %5d %5d",
            r["pair_id"],
            r["canonical_a"][:25],
            r["canonical_b"][:25],
            r["len_a"],
            r["len_b"],
            r["distance"],
        )
    logger.info("=" * 60)
    logger.info("Done. All outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
