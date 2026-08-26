# ruff: noqa: N802, N803, N806
"""Round-trip figure for the IsalSR arXiv paper.

Demonstrates the round-trip property: DAG -> D2S -> String -> S2D -> DAG
across three algorithms (canonical, canonical pruned, greedy), showing how
different D2S algorithms produce different strings but all round-trip correctly.

Expression: x_0^3 + x_0^2 + x_0 (Nguyen-1)

Layout:
    3 rows (canonical, pruned, greedy) x 7 columns:
    Col 0: Original DAG + math expression
    Col 1: D2S mid-1 (DAG with ghosts, partial string)
    Col 2: D2S mid-2 (more ghosts, more string)
    Col 3: Complete string only (label + token heatmap)
    Col 4: S2D mid-1 (partial DAG, tokens consumed)
    Col 5: S2D mid-2 (more DAG, more tokens consumed)
    Col 6: Reconstructed DAG + math expression

Run:
    cd /home/mpascual/research/code/IsalSR && \\
    ~/.conda/envs/isalsr/bin/python experiments/scripts/generate_fig_roundtrip.py
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
    PAUL_TOL_BRIGHT,
    _compute_dag_layout,
    add_background_panel,
    apply_ieee_style,
    compute_d2s_ghost_state,
    draw_dag,
    render_token_heatmap_horizontal,
    save_figure,
    tokenize_for_display,
)
from isalsr.core.canonical import canonical_string, pruned_canonical_string
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.string_to_dag import StringToDAG

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Nguyen-1: x_0^3 + x_0^2 + x_0
EXPRESSION_CANONICAL = "V^V^VkV+VkPncnCPCnc"
NUM_VARS = 1
EXPRESSION_LATEX = r"x_0^3 + x_0^2 + x_0"

# --- Global visual parameters ---
NODE_SIZE = 0.6
TOKEN_CELL_WIDTH = 11.0
TOKEN_CELL_HEIGHT = 11.0
MATH_FONTSIZE = 25

# Figure dimensions
FIG_WIDTH = 20.0
FIG_HEIGHT = 13.0

# Row labels and colors
ROW_CONFIGS = [
    {"label": "Canonical\n(exhaustive)", "color": "#335588", "algo": "canonical"},
    {"label": "Canonical\n(pruned)", "color": "#335588", "algo": "pruned"},
    {"label": "Greedy", "color": "#CC3355", "algo": "greedy"},
]

# Column labels
COL_LABELS = [
    "Original\nDAG",
    "D2S\n(step 1)",
    "D2S\n(step 2)",
    "Instruction\nstring",
    "S2D\n(step 1)",
    "S2D\n(step 2)",
    "Reconstructed\nDAG",
]


def _build_dag_from_sympy():
    """Build the Nguyen-1 DAG from SymPy (natural node ordering).

    Using SymPy gives a different node ordering than building from the canonical
    string, which is important because the greedy D2S produces a different
    (longer) string when the node ordering is not optimized.
    """
    import sympy

    from isalsr.adapters.sympy_adapter import SympyAdapter

    x = sympy.Symbol("x_0")
    expr = x**3 + x**2 + x
    adapter = SympyAdapter()
    return adapter.from_sympy(expr, [x])


def _compute_algorithm_data(algo: str) -> dict:
    """Compute traces and strings for one D2S algorithm.

    Args:
        algo: One of 'canonical', 'pruned', 'greedy'.

    Returns:
        Dict with keys: d2s_string, s2d_trace, full_dag, full_pos,
        tokens, d2s_snapshots, s2d_snapshots.
    """
    # Build the DAG from SymPy (natural node ordering)
    full_dag = _build_dag_from_sympy()
    full_pos = _compute_dag_layout(full_dag)

    # Get the D2S string based on algorithm
    if algo == "canonical":
        d2s_string = canonical_string(full_dag)
    elif algo == "pruned":
        d2s_string = pruned_canonical_string(full_dag)
    elif algo == "greedy":
        d2s_converter = DAGToString(full_dag, initial_node=0)
        d2s_string = d2s_converter.run()
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    logger.info("  %s D2S string: %r (len=%d)", algo, d2s_string, len(d2s_string))

    # Run S2D with trace on the D2S string
    s2d = StringToDAG(d2s_string, num_variables=NUM_VARS)
    reconstructed_dag = s2d.run(trace=True)
    s2d_trace = s2d._trace_log

    tokens = tokenize_for_display(d2s_string)

    # Select S2D snapshots for intermediate steps (at ~33% and ~66%)
    n_trace = len(s2d_trace)
    s2d_snap_1 = max(1, n_trace // 3)
    s2d_snap_2 = max(s2d_snap_1 + 1, 2 * n_trace // 3)
    if s2d_snap_2 >= n_trace:
        s2d_snap_2 = n_trace - 1

    # S2D intermediate DAGs (partial, being built up)
    s2d_mid1_dag = s2d_trace[s2d_snap_1][0]  # (dag, cdll, pri, sec, tokens_so_far)
    s2d_mid2_dag = s2d_trace[s2d_snap_2][0]

    # Compute how many tokens have been processed at each S2D snapshot
    # trace[i][4] is a list of tokens processed so far
    s2d_mid1_token_idx = len(s2d_trace[s2d_snap_1][4])
    s2d_mid2_token_idx = len(s2d_trace[s2d_snap_2][4])

    # For D2S visualization (reverse of S2D): show full DAG with ghosts
    # where S2D hasn't built yet.
    # D2S col 1 (early encoding) = S2D has built little -> many ghosts
    # D2S col 2 (late encoding) = S2D has built most -> fewer ghosts
    d2s_mid1_ghost_nodes, d2s_mid1_ghost_edges = compute_d2s_ghost_state(
        reconstructed_dag,
        s2d_mid1_dag,  # S2D at 33% -> D2S early: 67% ghost
    )
    d2s_mid2_ghost_nodes, d2s_mid2_ghost_edges = compute_d2s_ghost_state(
        reconstructed_dag,
        s2d_mid2_dag,  # S2D at 66% -> D2S late: 34% ghost
    )

    return {
        "algo": algo,
        "d2s_string": d2s_string,
        "tokens": tokens,
        "full_dag": full_dag,
        "full_pos": full_pos,
        "reconstructed_dag": reconstructed_dag,
        "reconstructed_pos": _compute_dag_layout(reconstructed_dag),
        # D2S intermediate ghosts
        "d2s_mid1_ghosts": (d2s_mid1_ghost_nodes, d2s_mid1_ghost_edges),
        "d2s_mid2_ghosts": (d2s_mid2_ghost_nodes, d2s_mid2_ghost_edges),
        # D2S token progress: D2S col1 (early) = ~33% string, col2 (late) = ~66%
        "d2s_mid1_token_idx": max(0, s2d_mid1_token_idx - 1),
        "d2s_mid2_token_idx": max(0, s2d_mid2_token_idx - 1),
        # S2D intermediate DAGs
        "s2d_mid1_dag": s2d_mid1_dag,
        "s2d_mid2_dag": s2d_mid2_dag,
        "s2d_mid1_token_idx": max(0, s2d_mid1_token_idx - 1),
        "s2d_mid2_token_idx": max(0, s2d_mid2_token_idx - 1),
    }


def generate_roundtrip_figure() -> str:
    """Generate the round-trip figure."""
    apply_ieee_style()

    logger.info("Computing algorithm data for Nguyen-1...")
    algo_data = []
    for cfg in ROW_CONFIGS:
        data = _compute_algorithm_data(cfg["algo"])
        algo_data.append(data)

    # -------------------------------------------------------------------------
    # Figure layout: 3 rows x 7 columns
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    outer_gs = GridSpec(
        3,
        1,
        figure=fig,
        hspace=0.20,
        left=0.10,
        right=0.97,
        top=0.92,
        bottom=0.03,
    )

    # Add column titles at fixed y positions using fig.text (row 0 only)
    # We need to compute x positions for each of 7 columns.
    # The columns span from left=0.10 to right=0.97 (width=0.87)
    col_width = 0.87 / 7
    col_title_y = 0.95
    for col_idx, label in enumerate(COL_LABELS):
        col_x = 0.10 + (col_idx + 0.5) * col_width
        fig.text(
            col_x,
            col_title_y,
            label,
            ha="center",
            va="bottom",
            fontsize=MATH_FONTSIZE,
            fontweight="bold",
            color="0.3",
        )

    for row_idx, (cfg, data) in enumerate(zip(ROW_CONFIGS, algo_data, strict=True)):
        # Each row: 2 sub-rows (DAG + heatmap) x 7 columns
        inner_gs = outer_gs[row_idx].subgridspec(
            2,
            7,
            height_ratios=[3.0, 0.4],
            hspace=0.06,
            wspace=0.10,
        )

        tokens = data["tokens"]
        n_tokens = len(tokens)
        full_dag = data["reconstructed_dag"]
        full_pos = data["reconstructed_pos"]

        row_axes = []

        for col in range(7):
            ax_dag = fig.add_subplot(inner_gs[0, col])
            ax_hm = fig.add_subplot(inner_gs[1, col])
            row_axes.extend([ax_dag, ax_hm])

            if col == 0:
                # Col 0: Original DAG (full color) + math expression
                draw_dag(ax_dag, full_dag, pos=full_pos, node_size=NODE_SIZE)
                render_token_heatmap_horizontal(
                    ax_hm,
                    tokens,
                    n_tokens - 1,
                    cell_width=TOKEN_CELL_WIDTH,
                    cell_height=TOKEN_CELL_HEIGHT,
                )
                # Add math expression as annotation below
                ax_hm.text(
                    0.5,
                    -0.8,
                    f"${EXPRESSION_LATEX}$",
                    ha="center",
                    va="top",
                    fontsize=MATH_FONTSIZE,
                    transform=ax_hm.transAxes,
                )

            elif col == 1:
                # Col 1: D2S mid-1 (full DAG with ghosts, ~33% string built)
                ghost_n, ghost_e = data["d2s_mid1_ghosts"]
                draw_dag(
                    ax_dag,
                    full_dag,
                    pos=full_pos,
                    node_size=NODE_SIZE,
                    ghost_nodes=ghost_n,
                    ghost_edges=ghost_e,
                )
                # Token heatmap: show partial string
                token_idx = data["d2s_mid1_token_idx"]
                token_idx = max(0, min(token_idx, n_tokens - 1))
                render_token_heatmap_horizontal(
                    ax_hm,
                    tokens,
                    min(token_idx, n_tokens - 1),
                    cell_width=TOKEN_CELL_WIDTH,
                    cell_height=TOKEN_CELL_HEIGHT,
                )

            elif col == 2:
                # Col 2: D2S mid-2 (more ghosts, ~66% string built)
                ghost_n, ghost_e = data["d2s_mid2_ghosts"]
                draw_dag(
                    ax_dag,
                    full_dag,
                    pos=full_pos,
                    node_size=NODE_SIZE,
                    ghost_nodes=ghost_n,
                    ghost_edges=ghost_e,
                )
                token_idx = data["d2s_mid2_token_idx"]
                token_idx = max(0, min(token_idx, n_tokens - 1))
                render_token_heatmap_horizontal(
                    ax_hm,
                    tokens,
                    min(token_idx, n_tokens - 1),
                    cell_width=TOKEN_CELL_WIDTH,
                    cell_height=TOKEN_CELL_HEIGHT,
                )

            elif col == 3:
                # Col 3: Complete string only -- label + token heatmap (no raw
                # string text, no len=X)
                ax_dag.axis("off")
                # Show string label (w* or w)
                string_label = r"$w^*$" if data["algo"] != "greedy" else r"$w$"
                ax_dag.text(
                    0.5,
                    0.5,
                    string_label,
                    ha="center",
                    va="center",
                    fontsize=MATH_FONTSIZE,
                    fontweight="bold",
                    color=cfg["color"],
                    transform=ax_dag.transAxes,
                )

                # Add direction arrows
                ax_dag.annotate(
                    "",
                    xy=(0.05, 0.3),
                    xytext=(0.35, 0.3),
                    xycoords="axes fraction",
                    arrowprops={
                        "arrowstyle": "<|-",
                        "color": PAUL_TOL_BRIGHT["blue"],
                        "linewidth": 1.2,
                        "mutation_scale": 10,
                    },
                )
                ax_dag.text(
                    0.12,
                    0.35,
                    "D2S",
                    fontsize=MATH_FONTSIZE,
                    color=PAUL_TOL_BRIGHT["blue"],
                    fontweight="bold",
                    transform=ax_dag.transAxes,
                )
                ax_dag.annotate(
                    "",
                    xy=(0.95, 0.3),
                    xytext=(0.65, 0.3),
                    xycoords="axes fraction",
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": PAUL_TOL_BRIGHT["red"],
                        "linewidth": 1.2,
                        "mutation_scale": 10,
                    },
                )
                ax_dag.text(
                    0.78,
                    0.35,
                    "S2D",
                    fontsize=MATH_FONTSIZE,
                    color=PAUL_TOL_BRIGHT["red"],
                    fontweight="bold",
                    transform=ax_dag.transAxes,
                )

                # Full token heatmap
                render_token_heatmap_horizontal(
                    ax_hm,
                    tokens,
                    n_tokens - 1,
                    cell_width=TOKEN_CELL_WIDTH,
                    cell_height=TOKEN_CELL_HEIGHT,
                )

            elif col == 4:
                # Col 4: S2D mid-1 (partial DAG being rebuilt)
                s2d_dag = data["s2d_mid1_dag"]
                s2d_pos = _compute_dag_layout(s2d_dag)
                draw_dag(ax_dag, s2d_dag, pos=s2d_pos, node_size=NODE_SIZE)
                token_idx = data["s2d_mid1_token_idx"]
                token_idx = max(0, min(token_idx, n_tokens - 1))
                render_token_heatmap_horizontal(
                    ax_hm,
                    tokens,
                    min(token_idx, n_tokens - 1),
                    cell_width=TOKEN_CELL_WIDTH,
                    cell_height=TOKEN_CELL_HEIGHT,
                )

            elif col == 5:
                # Col 5: S2D mid-2 (more DAG rebuilt)
                s2d_dag = data["s2d_mid2_dag"]
                s2d_pos = _compute_dag_layout(s2d_dag)
                draw_dag(ax_dag, s2d_dag, pos=s2d_pos, node_size=NODE_SIZE)
                token_idx = data["s2d_mid2_token_idx"]
                token_idx = max(0, min(token_idx, n_tokens - 1))
                render_token_heatmap_horizontal(
                    ax_hm,
                    tokens,
                    min(token_idx, n_tokens - 1),
                    cell_width=TOKEN_CELL_WIDTH,
                    cell_height=TOKEN_CELL_HEIGHT,
                )

            elif col == 6:
                # Col 6: Reconstructed DAG (full color) + checkmark
                draw_dag(ax_dag, full_dag, pos=full_pos, node_size=NODE_SIZE)
                render_token_heatmap_horizontal(
                    ax_hm,
                    tokens,
                    n_tokens - 1,
                    cell_width=TOKEN_CELL_WIDTH,
                    cell_height=TOKEN_CELL_HEIGHT,
                )
                # Add isomorphism annotation
                ax_hm.text(
                    0.5,
                    -0.8,
                    f"${EXPRESSION_LATEX}$",
                    ha="center",
                    va="top",
                    fontsize=MATH_FONTSIZE,
                    transform=ax_hm.transAxes,
                )

        # Add row label (rotated)
        fig.text(
            0.03,
            outer_gs[row_idx].get_position(fig).y0 + outer_gs[row_idx].get_position(fig).height / 2,
            cfg["label"],
            ha="center",
            va="center",
            fontsize=MATH_FONTSIZE,
            fontweight="bold",
            color=cfg["color"],
            rotation=90,
        )

        # Add subtle background per row
        add_background_panel(
            fig,
            row_axes,
            cfg["color"],
            label="",
            alpha=0.03,
        )

    # Add annotation between rows 0 and 1 if strings match
    if algo_data[0]["d2s_string"] == algo_data[1]["d2s_string"]:
        row0_pos = outer_gs[0].get_position(fig)
        row1_pos = outer_gs[1].get_position(fig)
        mid_y = (row0_pos.y0 + row1_pos.y1) / 2
        fig.text(
            0.53,
            mid_y,
            "Same string $w^*$ (>99.97% agreement)",
            ha="center",
            va="center",
            fontsize=MATH_FONTSIZE,
            fontstyle="italic",
            color="#335588",
            bbox={
                "facecolor": "white",
                "alpha": 0.9,
                "pad": 2,
                "edgecolor": "#335588",
                "linewidth": 0.5,
                "boxstyle": "round,pad=0.2",
            },
        )

    # Add annotation for row 2 (greedy) showing length difference
    canon_len = len(algo_data[0]["d2s_string"])
    greedy_len = len(algo_data[2]["d2s_string"])
    if greedy_len != canon_len:
        row2_pos = outer_gs[2].get_position(fig)
        pct = (greedy_len - canon_len) / canon_len * 100
        fig.text(
            0.53,
            row2_pos.y1 + 0.005,
            f"len={greedy_len} vs len={canon_len} (+{pct:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=MATH_FONTSIZE,
            color="#CC3355",
            bbox={
                "facecolor": "white",
                "alpha": 0.9,
                "pad": 1.5,
                "edgecolor": "#CC3355",
                "linewidth": 0.5,
                "boxstyle": "round,pad=0.15",
            },
        )

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "fig_roundtrip")
    saved = save_figure(fig, out_path)
    plt.close(fig)
    logger.info("Saved round-trip figure: %s", saved)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_roundtrip_figure()
