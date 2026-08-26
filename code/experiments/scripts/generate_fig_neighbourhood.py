# ruff: noqa: N802, N803, N806
"""Neighbourhood figure for the IsalSR arXiv paper.

Shows that Levenshtein-distance-1 edits on canonical strings produce structurally
meaningful DAG transformations, grouped by edit type (substitution, insertion, deletion).

Base expression: sin(x_0) + cos(x_0), canonical string: VcVspv+Ppc

Layout:
    Left panel:  Base expression G_0 (DAG + heatmap + math + stats)
    Right panel: 3 rows x 3 cols of neighbour cells grouped by edit type

Run:
    cd /home/mpascual/research/code/IsalSR && \\
    ~/.conda/envs/isalsr/bin/python experiments/scripts/generate_fig_neighbourhood.py
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.scripts._figure_helpers import (
    OUTPUT_DIR,
    PANEL_COLORS,
    _compute_dag_layout,
    add_background_panel,
    apply_ieee_style,
    canonical_string,
    dag_from_string,
    dag_to_sympy_latex,
    draw_dag,
    draw_expression_cell,
    draw_math_label,
    levenshtein,
    render_token_heatmap_horizontal,
    save_figure,
    tokenize_for_display,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

BASE_CANONICAL = "VcVspv+Ppc"
NUM_VARS = 1
BASE_EXPR_LATEX = r"\sin(x_0) + \cos(x_0)"

# --- Global visual parameters ---
NODE_SIZE = 1
BASE_NODE_SIZE = 0.8
NEIGHBOUR_NODE_SIZE = 0.7
TOKEN_CELL_WIDTH = 11.0
TOKEN_CELL_HEIGHT = 13.0
MATH_FONTSIZE = 12

# Hand-picked substitution neighbours: change the combinator + at position 6 (d=1)
SUBSTITUTION_NEIGHBOURS = [
    "VcVspv*Ppc",  # sin(x_0) * cos(x_0)
    "VcVspv-Ppc",  # cos(x_0) - sin(x_0)
    "VcVspv/Ppc",  # cos(x_0) / sin(x_0)
]


def _generate_all_lev1_neighbours(
    base: str,
    num_vars: int,
    alphabet: list[str],
) -> dict[str, list[dict]]:
    """Generate all unique Lev-1 neighbours grouped by edit type.

    Returns:
        Dict with keys 'substitution', 'insertion', 'deletion'.
        Each value is a list of dicts with keys: canon, edit_desc, dag, latex.
    """

    results: dict[str, list[dict]] = {
        "substitution": [],
        "insertion": [],
        "deletion": [],
    }
    seen_canons: dict[str, set[str]] = {
        "substitution": set(),
        "insertion": set(),
        "deletion": set(),
    }

    # Substitutions: replace each character in the raw string
    for i in range(len(base)):
        for ch in "NPnpCcWV":
            if ch == base[i]:
                continue
            new_str = base[:i] + ch + base[i + 1 :]
            _try_add_neighbour(
                new_str, num_vars, "substitution", results, seen_canons, f"sub: {base[i]}->{ch}"
            )

        # For V/v at position i, also try replacing the label char
        if i > 0 and base[i - 1] in ("V", "v"):
            for label_ch in "+*-/scelr^agik":
                if label_ch == base[i]:
                    continue
                new_str = base[:i] + label_ch + base[i + 1 :]
                _try_add_neighbour(
                    new_str,
                    num_vars,
                    "substitution",
                    results,
                    seen_canons,
                    f"sub: {base[i]}->{label_ch}",
                )

    # Insertions: insert each possible character at each position
    for i in range(len(base) + 1):
        for ch in "NPnpCcW":
            new_str = base[:i] + ch + base[i:]
            _try_add_neighbour(
                new_str, num_vars, "insertion", results, seen_canons, f"ins: {ch} @{i}"
            )
        # Two-char tokens V+label
        for prefix in ("V", "v"):
            for label_ch in "+*-/scelr^agik":
                new_str = base[:i] + prefix + label_ch + base[i:]
                _try_add_neighbour(
                    new_str,
                    num_vars,
                    "insertion",
                    results,
                    seen_canons,
                    f"ins: {prefix}{label_ch} @{i}",
                )

    # Deletions: remove each character
    for i in range(len(base)):
        new_str = base[:i] + base[i + 1 :]
        _try_add_neighbour(
            new_str, num_vars, "deletion", results, seen_canons, f"del: {base[i]} @{i}"
        )

    return results


def _try_add_neighbour(
    new_str: str,
    num_vars: int,
    edit_type: str,
    results: dict[str, list[dict]],
    seen_canons: dict[str, set[str]],
    edit_desc: str,
) -> None:
    """Try to parse a candidate string and add it to results if valid and unique."""
    from isalsr.core.string_to_dag import StringToDAG

    if not new_str:
        return
    try:
        s2d = StringToDAG(new_str, num_variables=num_vars)
        dag = s2d.run()
        # Only keep non-trivial DAGs (more than just variables)
        if dag.node_count <= num_vars:
            return
        canon = canonical_string(dag)
        # Skip if same as base
        if canon == BASE_CANONICAL:
            return
        # Skip duplicates within this edit type
        if canon in seen_canons[edit_type]:
            return
        seen_canons[edit_type].add(canon)

        # Compute sympy expression
        try:
            latex = dag_to_sympy_latex(dag)
        except Exception:
            latex = "?"

        lev_dist = levenshtein(BASE_CANONICAL, canon)

        results[edit_type].append(
            {
                "canon": canon,
                "edit_desc": edit_desc,
                "dag": dag,
                "latex": latex,
                "lev_dist": lev_dist,
                "original_str": new_str,
            }
        )
    except Exception:
        pass


def _select_neighbours(
    all_neighbours: dict[str, list[dict]],
    n_per_type: int = 3,
) -> dict[str, list[dict]]:
    """Select the most interesting neighbours from each edit type.

    Selection criteria: prefer valid math expressions, small Levenshtein distance,
    and structurally varied DAGs.
    """
    selected: dict[str, list[dict]] = {}
    for edit_type, neighbours in all_neighbours.items():
        # Prefer neighbours with valid math expressions (latex != "?")
        has_expr = [n for n in neighbours if n["latex"] != "?"]
        no_expr = [n for n in neighbours if n["latex"] == "?"]
        # Sort each group by Levenshtein distance, then string length
        has_expr.sort(key=lambda x: (x["lev_dist"], len(x["canon"])))
        no_expr.sort(key=lambda x: (x["lev_dist"], len(x["canon"])))
        # Take from has_expr first, fill remaining from no_expr
        combined = has_expr + no_expr
        selected[edit_type] = combined[:n_per_type]
    return selected


def _build_substitution_entries() -> list[dict]:
    """Build neighbour entries for the hand-picked substitution canonical strings."""
    entries = []
    for canon_str in SUBSTITUTION_NEIGHBOURS:
        dag = dag_from_string(canon_str, NUM_VARS)
        lev_dist = levenshtein(BASE_CANONICAL, canon_str)
        try:
            latex = dag_to_sympy_latex(dag)
        except Exception:
            latex = "?"
        entries.append(
            {
                "canon": canon_str,
                "edit_desc": f"sub (d={lev_dist})",
                "dag": dag,
                "latex": latex,
                "lev_dist": lev_dist,
                "original_str": canon_str,
            }
        )
    return entries


def generate_neighbourhood_figure() -> str:
    """Generate the neighbourhood figure."""
    apply_ieee_style()

    logger.info("Generating all Lev-1 neighbours for %s", BASE_CANONICAL)
    alphabet = list("NPnpCcW") + [f"V{c}" for c in "+*-/scelr^agik"]
    all_neighbours = _generate_all_lev1_neighbours(BASE_CANONICAL, NUM_VARS, alphabet)

    for etype, nlist in all_neighbours.items():
        logger.info("  %s: %d unique canonical forms", etype, len(nlist))

    # Use hand-picked substitutions, auto-select insertion/deletion
    selected_auto = _select_neighbours(all_neighbours, n_per_type=3)
    selected: dict[str, list[dict]] = {
        "substitution": _build_substitution_entries(),
        "insertion": selected_auto["insertion"],
        "deletion": selected_auto["deletion"],
    }

    # Build base DAG
    base_dag = dag_from_string(BASE_CANONICAL, NUM_VARS)
    base_pos = _compute_dag_layout(base_dag)

    # -------------------------------------------------------------------------
    # Figure layout
    # -------------------------------------------------------------------------
    fig_width = 10.0
    fig_height = 9.0

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Outer grid: left (base) | right (neighbours)
    outer_gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.0, 3.0],
        wspace=0.08,
        left=0.04,
        right=0.98,
        top=0.95,
        bottom=0.03,
    )

    # Left: base expression (DAG + heatmap + math + stats)
    # Give DAG most of the vertical space so G₀ extends to the top
    left_gs = outer_gs[0, 0].subgridspec(4, 1, height_ratios=[5.0, 0.4, 0.4, 0.5], hspace=0.08)
    ax_base_dag = fig.add_subplot(left_gs[0])
    ax_base_heatmap = fig.add_subplot(left_gs[1])
    ax_base_math = fig.add_subplot(left_gs[2])
    ax_base_stats = fig.add_subplot(left_gs[3])

    # Draw base DAG
    draw_dag(ax_base_dag, base_dag, pos=base_pos, node_size=BASE_NODE_SIZE)
    ax_base_dag.set_title(r"$G_0$", fontsize=11, fontweight="bold", pad=5)

    # Draw base heatmap
    base_tokens = tokenize_for_display(BASE_CANONICAL)
    render_token_heatmap_horizontal(
        ax_base_heatmap,
        base_tokens,
        len(base_tokens) - 1,
        cell_width=TOKEN_CELL_WIDTH,
        cell_height=TOKEN_CELL_HEIGHT,
    )

    # Draw base math label
    draw_math_label(ax_base_math, BASE_EXPR_LATEX, fontsize=MATH_FONTSIZE)

    # Draw stats
    total_sub = len(all_neighbours["substitution"])
    total_ins = len(all_neighbours["insertion"])
    total_del = len(all_neighbours["deletion"])
    total_unique = total_sub + total_ins + total_del
    stats_text = (
        f"{total_unique} unique neighbours\nsub: {total_sub} / ins: {total_ins} / del: {total_del}"
    )
    ax_base_stats.text(
        0.5,
        0.5,
        stats_text,
        ha="center",
        va="center",
        fontsize=10,
        color="0.3",
        transform=ax_base_stats.transAxes,
    )
    ax_base_stats.axis("off")

    # Right: 3 rows (sub, ins, del) x 3 cols
    right_gs = outer_gs[0, 1].subgridspec(3, 1, hspace=0.22)

    edit_types = ["substitution", "insertion", "deletion"]
    edit_labels = ["Substitution", "Insertion", "Deletion"]
    panel_colors = [
        PANEL_COLORS["substitution"],
        PANEL_COLORS["insertion"],
        PANEL_COLORS["deletion"],
    ]

    all_group_axes: list[list[plt.Axes]] = []

    for row_idx, etype in enumerate(edit_types):
        # Each row: 3 columns, each column has DAG + heatmap + math
        row_gs = right_gs[row_idx].subgridspec(
            3,
            3,
            height_ratios=[2.5, 0.35, 0.35],
            hspace=0.08,
            wspace=0.15,
        )

        neighbours = selected.get(etype, [])
        row_axes = []

        for col_idx in range(3):
            ax_dag = fig.add_subplot(row_gs[0, col_idx])
            ax_hm = fig.add_subplot(row_gs[1, col_idx])
            ax_math = fig.add_subplot(row_gs[2, col_idx])

            row_axes.extend([ax_dag, ax_hm, ax_math])

            if col_idx < len(neighbours):
                nb = neighbours[col_idx]
                nb_dag = nb["dag"]

                # Title: edit description + Lev distance
                title = f"d={nb['lev_dist']}"

                draw_expression_cell(
                    ax_dag,
                    ax_hm,
                    nb_dag,
                    nb["canon"],
                    ax_math=ax_math,
                    node_size=NEIGHBOUR_NODE_SIZE,
                    cell_width=TOKEN_CELL_WIDTH,
                    cell_height=TOKEN_CELL_HEIGHT,
                    math_fontsize=MATH_FONTSIZE,
                    title=title,
                    title_fontsize=12,
                )
            else:
                ax_dag.axis("off")
                ax_hm.axis("off")
                ax_math.axis("off")

        all_group_axes.append(row_axes)

    # Add background panels for each group
    fig.canvas.draw()
    for row_idx in range(len(edit_types)):
        if all_group_axes[row_idx]:
            add_background_panel(
                fig,
                all_group_axes[row_idx],
                panel_colors[row_idx],
                label=edit_labels[row_idx],
                alpha=0.04,
                label_fontsize=9,
            )

    # Add base panel background — extend to figure top
    add_background_panel(
        fig,
        [ax_base_dag, ax_base_heatmap, ax_base_math, ax_base_stats],
        PANEL_COLORS["base"],
        label="",
        alpha=0.06,
        pad_x=0.010,
        pad_y=0.008,
        extend_top=0.97,
    )

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "fig_neighbourhood")
    saved = save_figure(fig, out_path)
    plt.close(fig)
    logger.info("Saved neighbourhood figure: %s", saved)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_neighbourhood_figure()
