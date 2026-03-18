# ruff: noqa: N803, N806
"""Algorithm overview figure for the IsalSR paper.

Generates a horizontal figure showing S2D (String-to-DAG) and D2S (DAG-to-String)
execution steps for sin(x_0)*x_1 + cos(x_0). Adapted from IsalGraph's algorithm_figures.py
for directed labeled DAGs.

Layout:
    - Top row: S2D execution (5 snapshots, left to right)
    - Bottom row: D2S execution (5 snapshots, left to right)
    Each snapshot column shows: CDLL ring, instruction heatmap, DAG

Usage:
    cd /home/mpascual/research/code/IsalSR && \
    ~/.conda/envs/isalsr/bin/python experiments/scripts/generate_algorithm_overview.py \
      --output-dir /media/mpascual/Sandisk2TB/research/isalsr/results/figures
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.plotting_styles import (
    TOKEN_COLORS,
    apply_ieee_style,
    save_figure,
    tokenize_for_display,
)
from isalsr.core.canonical import canonical_string
from isalsr.core.cdll import CircularDoublyLinkedList
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.string_to_dag import StringToDAG

logger = logging.getLogger(__name__)

# =============================================================================
# Color scheme (Paul Tol colorblind-safe)
# =============================================================================

PRIMARY_COLOR = "#EE6677"  # Paul Tol red for primary pointer
SECONDARY_COLOR = "#4477AA"  # Paul Tol blue for secondary pointer
NEW_NODE_COLOR = "#CCBB44"  # Paul Tol yellow for newly inserted nodes
_DEFAULT_NODE_COLOR = "#DDDDDD"

# Node type colors for DAG visualization
_NODE_TYPE_COLORS: dict[NodeType, str] = {
    NodeType.VAR: "#4477AA",  # blue
    NodeType.ADD: "#228833",  # green
    NodeType.MUL: "#44AA66",  # light green
    NodeType.SUB: "#EE7733",  # orange
    NodeType.DIV: "#DDAA33",  # amber
    NodeType.SIN: "#AA3377",  # magenta
    NodeType.COS: "#CC6699",  # pink
    NodeType.EXP: "#6644AA",  # purple
    NodeType.LOG: "#8866BB",  # lavender
    NodeType.SQRT: "#9977CC",  # light purple
    NodeType.POW: "#AA7744",  # brown
    NodeType.ABS: "#BB88DD",  # light magenta
    NodeType.NEG: "#CC3355",  # dark rose
    NodeType.INV: "#66BBAA",  # teal
    NodeType.CONST: "#CCBB44",  # gold
}


# =============================================================================
# Node label rendering
# =============================================================================


def node_display_label(dag: LabeledDAG, node_id: int) -> str:
    """Return a display label string for a DAG node.

    Uses LaTeX math notation for operators and variable subscripts.

    Args:
        dag: The DAG containing the node.
        node_id: The node index.

    Returns:
        Display-ready string.
    """
    label = dag.node_label(node_id)
    data = dag.node_data(node_id)
    if label == NodeType.VAR:
        return f"$x_{{{data.get('var_index', '?')}}}$"
    if label == NodeType.ADD:
        return "$+$"
    if label == NodeType.MUL:
        return r"$\times$"
    if label == NodeType.SUB:
        return "$-$"
    if label == NodeType.DIV:
        return r"$\div$"
    if label == NodeType.SIN:
        return "sin"
    if label == NodeType.COS:
        return "cos"
    if label == NodeType.EXP:
        return "exp"
    if label == NodeType.LOG:
        return "log"
    if label == NodeType.SQRT:
        return r"$\sqrt{}$"
    if label == NodeType.POW:
        return r"$\wedge$"
    if label == NodeType.ABS:
        return r"$|\cdot|$"
    if label == NodeType.NEG:
        return "neg"
    if label == NodeType.INV:
        return "inv"
    if label == NodeType.CONST:
        return "$c$"
    return "?"


# =============================================================================
# CDLL extraction
# =============================================================================


def extract_cdll_order(
    cdll: CircularDoublyLinkedList,
    dag_snapshot: LabeledDAG,
    start_ptr: int,
) -> list[tuple[int, int, NodeType | None]]:
    """Extract ordered list of (cdll_idx, graph_node, label) from CDLL traversal.

    Args:
        cdll: The CDLL instance.
        dag_snapshot: The DAG snapshot for label lookups.
        start_ptr: CDLL node index to start traversal from.

    Returns:
        List of (cdll_index, graph_node_id, node_label_or_None).
    """
    if cdll.size() == 0:
        return []
    order: list[tuple[int, int, NodeType | None]] = []
    current = start_ptr
    for _ in range(cdll.size()):
        graph_node = cdll.get_value(current)
        label: NodeType | None = None
        if graph_node < dag_snapshot.node_count:
            label = dag_snapshot.node_label(graph_node)
        order.append((current, graph_node, label))
        current = cdll._next[current]
    return order


# =============================================================================
# Snapshot selection
# =============================================================================


def pick_snapshots(trace: list, n: int = 5) -> list[int]:
    """Select n evenly-spaced snapshot indices (always includes first and last).

    Args:
        trace: The trace list.
        n: Number of snapshots to pick.

    Returns:
        List of indices into the trace.
    """
    total = len(trace)
    if total <= n:
        return list(range(total))
    indices = [0]
    step = (total - 1) / (n - 1)
    for i in range(1, n - 1):
        indices.append(round(i * step))
    indices.append(total - 1)
    return indices


# =============================================================================
# CDLL ring drawing (adapted for labeled DAGs)
# =============================================================================


def draw_cdll_ring(
    ax: plt.Axes,
    cdll_order: list[tuple[int, int, NodeType | None]],
    primary_cdll_idx: int,
    secondary_cdll_idx: int,
    dag: LabeledDAG,
    *,
    new_graph_node: int | None = None,
    radius: float = 0.7,
    node_radius: float = 0.18,
) -> None:
    """Draw CDLL as a circular ring with labeled nodes and pointer arrows.

    Unlike IsalGraph's version, nodes display operation labels instead of
    plain integer IDs. Node fill colors correspond to operation type.

    Args:
        ax: Matplotlib axes.
        cdll_order: CDLL traversal from extract_cdll_order.
        primary_cdll_idx: CDLL node index of primary pointer.
        secondary_cdll_idx: CDLL node index of secondary pointer.
        dag: DAG snapshot for label lookups.
        new_graph_node: If set, highlight this graph node in yellow.
        radius: Ring radius.
        node_radius: Individual node circle radius.
    """
    n = len(cdll_order)
    if n == 0:
        ax.axis("off")
        return

    # Map cdll indices to positions in the order list
    cdll_idx_to_pos = {entry[0]: i for i, entry in enumerate(cdll_order)}
    pri_pos = cdll_idx_to_pos.get(primary_cdll_idx, 0)
    sec_pos = cdll_idx_to_pos.get(secondary_cdll_idx, 0)

    # Compute node positions (evenly spaced on circle, starting from top)
    angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]

    # Draw edges (lines between consecutive nodes)
    for i in range(n):
        j = (i + 1) % n
        ax.annotate(
            "",
            xy=positions[j],
            xytext=positions[i],
            arrowprops={
                "arrowstyle": "-",
                "color": "0.6",
                "linewidth": 0.6,
            },
        )

    # Draw nodes with operation labels
    for i, (_cdll_idx, graph_node, label) in enumerate(cdll_order):
        x, y = positions[i]

        # Determine fill color
        if new_graph_node is not None and graph_node == new_graph_node:
            color = NEW_NODE_COLOR
        elif i == pri_pos and i == sec_pos:
            color = "#AA77BB"  # Blend for overlap
        elif i == pri_pos:
            color = PRIMARY_COLOR
        elif i == sec_pos:
            color = SECONDARY_COLOR
        elif label is not None:
            color = _NODE_TYPE_COLORS.get(label, _DEFAULT_NODE_COLOR)
        else:
            color = _DEFAULT_NODE_COLOR

        circle = plt.Circle(
            (x, y),
            node_radius,
            facecolor=color,
            edgecolor="0.3",
            linewidth=0.6,
            zorder=3,
        )
        ax.add_patch(circle)

        # Show operation label inside node
        display_label = node_display_label(dag, graph_node) if graph_node < dag.node_count else "?"
        ax.text(
            x,
            y,
            display_label,
            ha="center",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color="white",
            zorder=4,
        )

    # Draw pointer arrows from outside the ring
    arrow_radius = radius + 0.30
    _draw_pointer_arrow(
        ax,
        positions[pri_pos],
        angles[pri_pos],
        arrow_radius,
        node_radius,
        PRIMARY_COLOR,
        r"$\pi$",
    )

    if sec_pos != pri_pos:
        _draw_pointer_arrow(
            ax,
            positions[sec_pos],
            angles[sec_pos],
            arrow_radius,
            node_radius,
            SECONDARY_COLOR,
            r"$\sigma$",
        )
    else:
        # Overlap: offset secondary arrow slightly
        offset_angle = angles[pri_pos] + 0.35
        _draw_pointer_arrow(
            ax,
            positions[sec_pos],
            offset_angle,
            arrow_radius,
            node_radius,
            SECONDARY_COLOR,
            r"$\sigma$",
        )

    margin = arrow_radius + 0.45
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_pointer_arrow(
    ax: plt.Axes,
    target_pos: tuple[float, float],
    angle: float,
    arrow_radius: float,
    node_radius: float,
    color: str,
    label: str,
) -> None:
    """Draw a labeled pointer arrow from outside the ring toward a node."""
    start_x = arrow_radius * np.cos(angle)
    start_y = arrow_radius * np.sin(angle)

    ax.annotate(
        "",
        xy=target_pos,
        xytext=(start_x, start_y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 1.2,
            "shrinkA": 0,
            "shrinkB": node_radius * 72,
        },
    )
    label_x = (arrow_radius + 0.22) * np.cos(angle)
    label_y = (arrow_radius + 0.22) * np.sin(angle)
    ax.text(
        label_x,
        label_y,
        label,
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color=color,
    )


# =============================================================================
# DAG drawing (topological layout for directed labeled graph)
# =============================================================================


def _compute_dag_layout(dag: LabeledDAG) -> dict[int, tuple[float, float]]:
    """Compute a layered layout for a DAG with barycentric edge-crossing reduction.

    Uses a Sugiyama-style layered approach:
    1. Assign layers by longest path from sources (VAR nodes at bottom).
    2. Order nodes within each layer by the **barycenter** (mean x-position
       of parent nodes), which minimizes edge crossings.
    3. Apply 2 top-down + 2 bottom-up ordering sweeps for refinement.

    Args:
        dag: The DAG to lay out.

    Returns:
        Dictionary mapping node ID to (x, y) position.
    """
    if dag.node_count == 0:
        return {}

    n = dag.node_count
    layer: list[int] = [0] * n
    order = dag.topological_sort()
    for node in order:
        for neighbor in dag.out_neighbors(node):
            if neighbor < n:
                layer[neighbor] = max(layer[neighbor], layer[node] + 1)

    max_layer = max(layer) if layer else 0
    layers: dict[int, list[int]] = {lv: [] for lv in range(max_layer + 1)}
    for node in range(n):
        layers[layer[node]].append(node)

    # Initial ordering: sort by node ID within each layer (stable baseline).
    for lv in layers:
        layers[lv] = sorted(layers[lv])

    # Assign initial x-positions (evenly spaced).
    pos_x: dict[int, float] = {}
    for lv, nodes in layers.items():
        for i, node in enumerate(nodes):
            pos_x[node] = float(i)

    # Barycentric ordering sweeps to minimize edge crossings.
    # 2 top-down passes (order by parent x-means) + 2 bottom-up passes.
    for _sweep in range(2):
        # Top-down: order layer lv by mean x of parents (in layer lv-1).
        for lv in range(1, max_layer + 1):
            bary: dict[int, float] = {}
            for node in layers[lv]:
                parents = [p for p in dag.in_neighbors(node) if layer[p] == lv - 1]
                if parents:
                    bary[node] = sum(pos_x[p] for p in parents) / len(parents)
                else:
                    bary[node] = pos_x[node]
            layers[lv] = sorted(layers[lv], key=lambda nd: bary[nd])
            for i, node in enumerate(layers[lv]):
                pos_x[node] = float(i)

        # Bottom-up: order layer lv by mean x of children (in layer lv+1).
        for lv in range(max_layer - 1, -1, -1):
            bary = {}
            for node in layers[lv]:
                children = [c for c in dag.out_neighbors(node) if layer[c] == lv + 1]
                if children:
                    bary[node] = sum(pos_x[c] for c in children) / len(children)
                else:
                    bary[node] = pos_x[node]
            layers[lv] = sorted(layers[lv], key=lambda nd: bary[nd])
            for i, node in enumerate(layers[lv]):
                pos_x[node] = float(i)

    # Final positions: center each layer horizontally.
    pos: dict[int, tuple[float, float]] = {}
    for lv, nodes in layers.items():
        width = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (width - 1) / 2.0) * 1.6
            y = lv * 1.4
            pos[node] = (x, y)

    return pos


def draw_dag(
    ax: plt.Axes,
    dag: LabeledDAG,
    *,
    pos: dict[int, tuple[float, float]] | None = None,
    ghost_nodes: set[int] | None = None,
    ghost_edges: set[tuple[int, int]] | None = None,
    node_size: float = 0.38,
) -> dict[int, tuple[float, float]]:
    """Draw a labeled DAG with optional ghost rendering for D2S.

    Solid nodes/edges are already-encoded parts. Ghost (dashed) nodes/edges
    show parts not yet encoded.

    Args:
        ax: Matplotlib axes.
        dag: The full DAG to draw.
        pos: Pre-computed layout positions. If None, computed automatically.
        ghost_nodes: Set of node IDs to render as ghosts (not yet encoded).
        ghost_edges: Set of (src, tgt) edges to render as ghosts.
        node_size: Radius of node circles.

    Returns:
        Layout position dict.
    """
    n = dag.node_count
    if n == 0:
        ax.axis("off")
        return {}

    if pos is None:
        pos = _compute_dag_layout(dag)

    if ghost_nodes is None:
        ghost_nodes = set()
    if ghost_edges is None:
        ghost_edges = set()

    # Collect all edges
    all_edges: list[tuple[int, int]] = []
    for src in range(n):
        for tgt in dag.out_neighbors(src):
            all_edges.append((src, tgt))

    # Draw edges first (behind nodes)
    for src, tgt in all_edges:
        if src not in pos or tgt not in pos:
            continue
        is_ghost = (src, tgt) in ghost_edges
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]

        # Compute direction for shrinking arrow endpoints
        dx = x1 - x0
        dy = y1 - y0
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 1e-6:
            continue

        # Shrink endpoints by node_size to avoid overlapping circles
        ux, uy = dx / dist, dy / dist
        sx, sy = x0 + ux * node_size, y0 + uy * node_size
        ex, ey = x1 - ux * node_size, y1 - uy * node_size

        ax.annotate(
            "",
            xy=(ex, ey),
            xytext=(sx, sy),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "0.82" if is_ghost else "0.25",
                "linewidth": 0.4 if is_ghost else 1.0,
                "linestyle": (0, (3, 3)) if is_ghost else "solid",
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 8,
            },
            zorder=1,
        )

    # Draw nodes
    for node_id in range(n):
        if node_id not in pos:
            continue
        x, y = pos[node_id]
        is_ghost = node_id in ghost_nodes
        label = dag.node_label(node_id)

        if is_ghost:
            facecolor = "white"
            edgecolor = "0.7"
            linestyle = "dashed"
            text_color = "0.5"
        else:
            facecolor = _NODE_TYPE_COLORS.get(label, _DEFAULT_NODE_COLOR)
            edgecolor = "0.3"
            linestyle = "solid"
            text_color = "white"

        circle = plt.Circle(
            (x, y),
            node_size,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.6,
            linestyle=linestyle,
            zorder=3,
        )
        ax.add_patch(circle)

        display_label = node_display_label(dag, node_id)
        ax.text(
            x,
            y,
            display_label,
            ha="center",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color=text_color,
            zorder=4,
        )

    # Set axis limits with padding.
    # Ensure a minimum extent so single-node DAGs are not oversized.
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        x_span = max(max(xs) - min(xs), 0.1)
        y_span = max(max(ys) - min(ys), 0.1)
        pad_x = 0.30 * x_span + node_size
        pad_y = 0.30 * y_span + node_size
        # Enforce minimum axis extent so single nodes do not fill the panel.
        # Must be large enough that a node_size=0.20 circle appears small.
        min_extent = 3.0
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        half_w = max((max(xs) - min(xs)) / 2 + pad_x, min_extent / 2)
        half_h = max((max(ys) - min(ys)) / 2 + pad_y, min_extent / 2)
        ax.set_xlim(cx - half_w, cx + half_w)
        ax.set_ylim(cy - half_h, cy + half_h)

    ax.set_aspect("equal")
    ax.axis("off")
    return pos


# =============================================================================
# Instruction heatmap (horizontal colored token strip)
# =============================================================================


def render_token_heatmap_horizontal(
    ax: plt.Axes,
    tokens: list[str],
    current_token_idx: int,
    *,
    cell_width: float = 0.6,
    cell_height: float = 0.5,
) -> None:
    """Render instruction tokens as a horizontal row of colored cells.

    Processed tokens have full opacity; remaining tokens are dimmed.

    Args:
        ax: Matplotlib axes.
        tokens: List of tokens (single-char or two-char).
        current_token_idx: Index of last completed token (-1 for none).
        cell_width: Width of each cell.
        cell_height: Height of each cell.
    """
    n = len(tokens)
    if n == 0:
        ax.axis("off")
        return

    for i, token in enumerate(tokens):
        color = TOKEN_COLORS.get(token, "#000000")
        completed = i <= current_token_idx
        alpha = 1.0 if completed else 0.15
        x = i * cell_width

        rect = mpatches.FancyBboxPatch(
            (x, -cell_height / 2),
            cell_width * 0.85,
            cell_height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            alpha=alpha,
            edgecolor="0.4" if completed else "0.8",
            linewidth=0.3,
        )
        ax.add_patch(rect)
        ax.text(
            x + cell_width * 0.425,
            0,
            token,
            ha="center",
            va="center",
            fontsize=5,
            fontfamily="monospace",
            fontweight="bold" if i == current_token_idx else "normal",
            color="white" if completed else "0.6",
        )

    ax.set_xlim(-cell_width * 0.2, n * cell_width + cell_width * 0.1)
    ax.set_ylim(-cell_height, cell_height)
    ax.set_aspect("auto")
    ax.axis("off")


# =============================================================================
# D2S ghost state computation
# =============================================================================


def compute_d2s_ghost_state(
    full_dag: LabeledDAG,
    output_dag: LabeledDAG,
) -> tuple[set[int], set[tuple[int, int]]]:
    """Determine which nodes and edges in the full DAG are ghosts (not yet encoded).

    The D2S output_dag contains the nodes and edges encoded so far. Nodes in
    the full_dag that have no corresponding node in the output_dag are ghosts.

    Note: This comparison uses raw node IDs, which is correct when the D2S
    operates on a canonical-derived DAG (where the i2o mapping is identity
    because S2D and D2S traverse nodes in the same order). For DAGs with
    non-identity i2o mappings, a mapping-aware version would be needed.

    Args:
        full_dag: The complete target DAG.
        output_dag: The partial DAG built by D2S so far.

    Returns:
        (ghost_nodes, ghost_edges) -- sets of IDs/edge tuples that are not yet encoded.
    """
    ghost_nodes: set[int] = set()
    for i in range(full_dag.node_count):
        if i >= output_dag.node_count:
            ghost_nodes.add(i)

    ghost_edges: set[tuple[int, int]] = set()
    for src in range(full_dag.node_count):
        for tgt in full_dag.out_neighbors(src):
            if (
                src >= output_dag.node_count
                or tgt >= output_dag.node_count
                or not output_dag.has_edge(src, tgt)
            ):
                ghost_edges.add((src, tgt))

    return ghost_nodes, ghost_edges


# =============================================================================
# Main figure generation
# =============================================================================


def generate_algorithm_overview(
    canonical_str: str,
    full_dag: LabeledDAG,
    num_variables: int,
    output_dir: str,
) -> str:
    """Generate the algorithm overview figure with S2D and D2S traces.

    Args:
        canonical_str: The canonical string for the expression.
        full_dag: The complete expression DAG.
        num_variables: Number of input variables.
        output_dir: Directory to save figures.

    Returns:
        Base path of saved figure (without extension).
    """
    # --- Generate S2D trace ---
    s2d = StringToDAG(canonical_str, num_variables=num_variables)
    s2d_dag = s2d.run(trace=True)
    s2d_trace = s2d._trace_log  # list[TraceEntry]

    # --- Generate D2S trace from the reconstructed DAG ---
    d2s = DAGToString(s2d_dag, initial_node=0)
    d2s_str = d2s.run(trace=True)
    d2s_trace = d2s._trace_log

    logger.info("Canonical string: %r (len=%d)", canonical_str, len(canonical_str))
    logger.info("D2S string: %r (len=%d)", d2s_str, len(d2s_str))
    logger.info("S2D trace: %d snapshots, D2S trace: %d snapshots", len(s2d_trace), len(d2s_trace))

    # Tokenize strings for heatmap display
    s2d_tokens = tokenize_for_display(canonical_str)
    d2s_tokens = tokenize_for_display(d2s_str)

    # Select 5 snapshots for each
    s2d_indices = pick_snapshots(s2d_trace, 5)
    d2s_indices = pick_snapshots(d2s_trace, 5)
    n_cols = max(len(s2d_indices), len(d2s_indices))

    logger.info("S2D snapshot indices: %s (of %d)", s2d_indices, len(s2d_trace))
    logger.info("D2S snapshot indices: %s (of %d)", d2s_indices, len(d2s_trace))

    # Compute fixed layouts.
    # D2S operates on s2d_dag (the DAG from S2D), NOT the original full_dag.
    # Using full_dag for ghost comparison would produce wrong results because
    # node IDs may differ (e.g., SIN=node2 in full_dag vs COS=node2 in s2d_dag).
    d2s_target_dag = s2d_dag  # The DAG that D2S actually operates on.
    d2s_target_pos = _compute_dag_layout(d2s_target_dag)
    s2d_final_dag = s2d_trace[-1][0]
    s2d_dag_pos = _compute_dag_layout(s2d_final_dag)

    # --- Create figure ---
    fig = plt.figure(figsize=(2.2 * n_cols + 0.5, 8.0))

    outer_gs = GridSpec(
        2,
        1,
        figure=fig,
        hspace=0.30,
        top=0.92,
        bottom=0.05,
        left=0.08,
        right=0.97,
    )

    # 3 sub-rows per group: CDLL ring, token heatmap, DAG
    gs_s2d = outer_gs[0].subgridspec(
        3,
        n_cols,
        height_ratios=[1.0, 0.25, 1.4],
        hspace=0.10,
        wspace=0.20,
    )
    gs_d2s = outer_gs[1].subgridspec(
        3,
        n_cols,
        height_ratios=[1.0, 0.25, 1.4],
        hspace=0.10,
        wspace=0.20,
    )

    s2d_cdll_axes = [fig.add_subplot(gs_s2d[0, c]) for c in range(n_cols)]
    s2d_instr_axes = [fig.add_subplot(gs_s2d[1, c]) for c in range(n_cols)]
    s2d_graph_axes = [fig.add_subplot(gs_s2d[2, c]) for c in range(n_cols)]
    d2s_cdll_axes = [fig.add_subplot(gs_d2s[0, c]) for c in range(n_cols)]
    d2s_instr_axes = [fig.add_subplot(gs_d2s[1, c]) for c in range(n_cols)]
    d2s_graph_axes = [fig.add_subplot(gs_d2s[2, c]) for c in range(n_cols)]

    # --- S2D snapshots ---
    for col, snap_idx in enumerate(s2d_indices):
        dag_snap, cdll_snap, pri, sec, tokens_so_far = s2d_trace[snap_idx]

        # Row 0: CDLL ring
        cdll_order = extract_cdll_order(cdll_snap, dag_snap, pri)
        # Determine if a new node was added in this step
        new_node = None
        if snap_idx > 0:
            prev_dag = s2d_trace[snap_idx - 1][0]
            if dag_snap.node_count > prev_dag.node_count:
                new_node = dag_snap.node_count - 1

        draw_cdll_ring(
            s2d_cdll_axes[col],
            cdll_order,
            pri,
            sec,
            dag_snap,
            new_graph_node=new_node,
        )

        step_label = "Init" if snap_idx == 0 else f"Step {snap_idx}"
        s2d_cdll_axes[col].set_title(step_label, fontsize=7, fontweight="bold", pad=3)

        # Row 1: Token heatmap
        # Compute how many tokens have been processed
        current_token_idx = len(tokens_so_far) - 1
        render_token_heatmap_horizontal(
            s2d_instr_axes[col],
            s2d_tokens,
            current_token_idx,
        )

        # Row 2: DAG (growing)
        draw_dag(
            s2d_graph_axes[col],
            dag_snap,
            pos={k: v for k, v in s2d_dag_pos.items() if k < dag_snap.node_count},
        )

    # --- D2S snapshots ---
    for col, snap_idx in enumerate(d2s_indices):
        output_dag_snap, cdll_snap, pri, sec, prefix = d2s_trace[snap_idx]

        # Row 0: CDLL ring
        cdll_order = extract_cdll_order(cdll_snap, output_dag_snap, pri)
        new_node = None
        if snap_idx > 0:
            prev_dag = d2s_trace[snap_idx - 1][0]
            if output_dag_snap.node_count > prev_dag.node_count:
                new_node = output_dag_snap.node_count - 1

        draw_cdll_ring(
            d2s_cdll_axes[col],
            cdll_order,
            pri,
            sec,
            output_dag_snap,
            new_graph_node=new_node,
        )

        step_label = "Init" if snap_idx == 0 else f"Step {snap_idx}"
        d2s_cdll_axes[col].set_title(step_label, fontsize=7, fontweight="bold", pad=3)

        # Row 1: Token heatmap
        d2s_tokens_so_far = tokenize_for_display(prefix)
        current_token_idx = len(d2s_tokens_so_far) - 1
        render_token_heatmap_horizontal(
            d2s_instr_axes[col],
            d2s_tokens,
            current_token_idx,
        )

        # Row 2: Target DAG with ghost rendering.
        # Use d2s_target_dag (the DAG D2S operates on), NOT full_dag,
        # because D2S builds output with d2s_target_dag's node IDs.
        ghost_nodes, ghost_edges = compute_d2s_ghost_state(d2s_target_dag, output_dag_snap)
        draw_dag(
            d2s_graph_axes[col],
            d2s_target_dag,
            pos=d2s_target_pos,
            ghost_nodes=ghost_nodes,
            ghost_edges=ghost_edges,
        )

    # Hide unused columns
    for col in range(len(s2d_indices), n_cols):
        s2d_cdll_axes[col].axis("off")
        s2d_instr_axes[col].axis("off")
        s2d_graph_axes[col].axis("off")
    for col in range(len(d2s_indices), n_cols):
        d2s_cdll_axes[col].axis("off")
        d2s_instr_axes[col].axis("off")
        d2s_graph_axes[col].axis("off")

    # --- Group boxes and labels ---
    _add_group_boxes_horizontal(
        fig,
        [s2d_cdll_axes, s2d_instr_axes, s2d_graph_axes],
        [d2s_cdll_axes, d2s_instr_axes, d2s_graph_axes],
    )

    # --- Legend at bottom ---
    legend_handles = [
        mpatches.Patch(facecolor=PRIMARY_COLOR, label=r"$\pi$ (primary)"),
        mpatches.Patch(facecolor=SECONDARY_COLOR, label=r"$\sigma$ (secondary)"),
        mpatches.Patch(facecolor=NEW_NODE_COLOR, label="New node"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=6,
        framealpha=0.8,
        bbox_to_anchor=(0.5, 0.0),
    )

    # Save
    output_path = os.path.join(output_dir, "fig_algorithm_overview")
    save_figure(fig, output_path)
    plt.close(fig)
    logger.info("Algorithm overview saved: %s", output_path)
    return output_path


def _add_group_boxes_horizontal(
    fig: plt.Figure,
    s2d_axes: list[list[plt.Axes]],
    d2s_axes: list[list[plt.Axes]],
) -> None:
    """Add grouped boxes and labels for the horizontal overview layout.

    Args:
        fig: The figure.
        s2d_axes: 3-element list of [cdll_axes_row, instr_axes_row, graph_axes_row].
        d2s_axes: 3-element list of [cdll_axes_row, instr_axes_row, graph_axes_row].
    """
    from matplotlib.transforms import Bbox

    renderer = fig.canvas.get_renderer()
    fig.draw(renderer)

    pad = 0.008

    def _group_bbox(axes_rows: list[list[plt.Axes]]) -> Bbox:
        bboxes = []
        for row in axes_rows:
            for ax in row:
                bb = ax.get_tightbbox(renderer)
                if bb is not None:
                    bboxes.append(bb.transformed(fig.transFigure.inverted()))
        if not bboxes:
            return Bbox([[0, 0], [1, 1]])
        return Bbox.union(bboxes)

    s2d_bb = _group_bbox(s2d_axes)
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (s2d_bb.x0 - pad, s2d_bb.y0 - pad),
            s2d_bb.width + 2 * pad,
            s2d_bb.height + 2 * pad,
            boxstyle="round,pad=0.005",
            facecolor="#EE6677",
            alpha=0.06,
            edgecolor="#EE6677",
            linewidth=1.0,
            transform=fig.transFigure,
            zorder=0,
        )
    )

    d2s_bb = _group_bbox(d2s_axes)
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (d2s_bb.x0 - pad, d2s_bb.y0 - pad),
            d2s_bb.width + 2 * pad,
            d2s_bb.height + 2 * pad,
            boxstyle="round,pad=0.005",
            facecolor="#4477AA",
            alpha=0.06,
            edgecolor="#4477AA",
            linewidth=1.0,
            transform=fig.transFigure,
            zorder=0,
        )
    )

    # Group titles (rotated, to the left of boxes)
    fig.text(
        s2d_bb.x0 - pad - 0.025,
        s2d_bb.y0 + s2d_bb.height / 2,
        "String-to-DAG (S2D)",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#CC3355",
        transform=fig.transFigure,
        rotation=90,
    )
    fig.text(
        d2s_bb.x0 - pad - 0.025,
        d2s_bb.y0 + d2s_bb.height / 2,
        "DAG-to-String (D2S)",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#335588",
        transform=fig.transFigure,
        rotation=90,
    )

    # Horizontal divider between the two groups
    mid_y = (s2d_bb.y0 + d2s_bb.y1) / 2
    fig.add_artist(
        plt.Line2D(
            [max(s2d_bb.x0, d2s_bb.x0) - pad, min(s2d_bb.x1, d2s_bb.x1) + pad],
            [mid_y, mid_y],
            transform=fig.transFigure,
            color="0.5",
            linewidth=0.8,
            linestyle="--",
            zorder=1,
        )
    )


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """CLI entry point for algorithm overview figure generation."""
    parser = argparse.ArgumentParser(
        description="Generate algorithm overview figure for IsalSR paper."
    )
    parser.add_argument(
        "--output-dir",
        default="/media/mpascual/Sandisk2TB/research/isalsr/results/figures",
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_ieee_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # Build expression: sin(x_0)*x_1 + cos(x_0)
    # This has k=4, uses 2 variables, 4 op types (SIN, COS, MUL, ADD),
    # depth=3, and 16-token canonical string -- rich enough to demonstrate
    # the full instruction set while remaining visually readable.
    dag = LabeledDAG(7)
    dag.add_node(NodeType.VAR, var_index=0)  # 0: x_0
    dag.add_node(NodeType.VAR, var_index=1)  # 1: x_1
    dag.add_node(NodeType.SIN)  # 2: sin(x_0)
    dag.add_edge(0, 2)
    dag.add_node(NodeType.COS)  # 3: cos(x_0)
    dag.add_edge(0, 3)
    dag.add_node(NodeType.MUL)  # 4: sin(x_0)*x_1
    dag.add_edge(2, 4)
    dag.add_edge(1, 4)
    dag.add_node(NodeType.ADD)  # 5: sin(x_0)*x_1 + cos(x_0)
    dag.add_edge(4, 5)
    dag.add_edge(3, 5)

    logger.info("Expression: sin(x_0)*x_1 + cos(x_0)")
    logger.info("DAG: k=4, 6 edges, depth=3")

    # Compute canonical string
    canon = canonical_string(dag)
    logger.info("Canonical string: %r (len=%d)", canon, len(canon))

    # Generate figure
    generate_algorithm_overview(canon, dag, num_variables=2, output_dir=args.output_dir)

    logger.info("Done. Figures saved in %s", args.output_dir)


if __name__ == "__main__":
    main()
