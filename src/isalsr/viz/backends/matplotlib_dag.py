"""Matplotlib backend for labeled-DAG drawing.

Draws a :class:`~isalsr.core.labeled_dag.LabeledDAG` as a layered diagram:
nodes as labelled discs, edges as trimmed arrows with proper arrowheads.

Arrow trimming is done in **data units** rather than via ``shrinkA``/``shrinkB``
(which are in points and silently produce headless lines). This requires
``set_aspect("equal")`` on the axes; the backend enforces it.

The drawing style matches ``experiments/scripts/generate_fig_const_normalization.py``
so all review figures read as one visual system.

Registers itself under the name ``"matplotlib"`` on import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.viz.base import DagVizBackend, Position
from isalsr.viz.registry import register_backend
from isalsr.viz.style import (
    ALERT_COLOR,
    ARROW_COLOR,
    CREATION_EDGE_COLOR,
    REACHABLE_TINT,
    color_for_node_border,
    color_for_node_face,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any

# Node-circle radius in data units.
NODE_R: float = 0.37
# Font sizes, chosen for the compiled figure size after \linewidth scaling.
FS_NODE: float = 18.0


def _node_display_label(dag: LabeledDAG, node: int) -> str:
    """Return the text label to print inside the node disc."""
    nt = dag.node_label(node)
    if nt is NodeType.VAR:
        idx = dag.node_data(node).get("var_index", node)
        return rf"$x_{{{int(idx) + 1}}}$"
    if nt is NodeType.CONST:
        return r"$c$"
    # Operator nodes: display the NodeType value (already a printable symbol).
    label_map: dict[NodeType, str] = {
        NodeType.ADD: r"$+$",
        NodeType.MUL: r"$\times$",
        NodeType.SUB: r"$-$",
        NodeType.DIV: r"$\div$",
        NodeType.SIN: r"$\sin$",
        NodeType.COS: r"$\cos$",
        NodeType.EXP: r"$\exp$",
        NodeType.LOG: r"$\log$",
        NodeType.SQRT: r"$\sqrt{\phantom{x}}$",
        NodeType.POW: r"$\hat{\ }$",
        NodeType.ABS: r"$|\cdot|$",
        NodeType.NEG: r"$-x$",
        NodeType.INV: r"$1/x$",
    }
    return label_map.get(nt, str(nt.value))


def _layered_layout(dag: LabeledDAG) -> dict[int, Position]:
    """Compute a layered layout: layer = longest path from any source node.

    Sources (in-degree 0) are placed at layer 0; each subsequent node is
    placed one layer above its highest predecessor. Within each layer nodes
    are distributed evenly on the x-axis.
    """
    n = dag.node_count
    if n == 0:
        return {}
    nodes = list(range(n))
    layer: dict[int, int] = {nd: 0 for nd in nodes}
    # Iterative relaxation until convergence (correct for any DAG topology).
    changed = True
    while changed:
        changed = False
        for nd in nodes:
            for pred in dag.in_neighbors(nd):
                cand = layer[pred] + 1
                if cand > layer[nd]:
                    layer[nd] = cand
                    changed = True

    max_l = max(layer.values(), default=0)
    positions: dict[int, Position] = {}
    for l_val in range(max_l + 1):
        nodes_l = sorted(nd for nd, la in layer.items() if la == l_val)
        m = len(nodes_l)
        for i, nd in enumerate(nodes_l):
            x = (i - (m - 1) / 2.0) * 2.5
            y = float(l_val) * 2.5
            positions[nd] = (x, y)
    return positions


def _draw_node(
    ax: Axes,
    node: int,
    pos: Position,
    dag: LabeledDAG,
    *,
    reachable: frozenset[int],
    node_colors: dict[int, str] | None,
    alert: bool,
) -> None:
    """Draw one node disc with optional reachability tint and alert ring."""
    from matplotlib.patches import Circle

    nt = dag.node_label(node)
    face = node_colors[node] if (node_colors and node in node_colors) else color_for_node_face(nt)
    border = color_for_node_border(nt)
    x, y = pos

    if alert:
        # Dashed red ring for violating (orphan CONST) nodes.
        ax.add_patch(
            Circle(
                (x, y),
                NODE_R + 0.11,
                facecolor="none",
                edgecolor=ALERT_COLOR,
                linewidth=1.5,
                linestyle=(0, (3, 2.5)),
                zorder=2,
            )
        )
    elif node in reachable and nt is not NodeType.VAR:
        # Thin green ring for reachable non-VAR nodes.
        ax.add_patch(
            Circle(
                (x, y),
                NODE_R + 0.09,
                facecolor="none",
                edgecolor=REACHABLE_TINT,
                linewidth=1.8,
                zorder=2,
            )
        )

    ax.add_patch(
        Circle(
            (x, y),
            NODE_R,
            facecolor=face,
            edgecolor=border,
            linewidth=1.6,
            zorder=3,
        )
    )
    ax.text(
        x,
        y,
        _node_display_label(dag, node),
        ha="center",
        va="center",
        fontsize=FS_NODE,
        zorder=4,
    )


def _draw_edge(
    ax: Axes,
    src_pos: Position,
    tgt_pos: Position,
    *,
    creation: bool,
) -> None:
    """Draw a directed edge as a trimmed arrow.

    Endpoints are shortened in data units (not points) so arrowheads remain
    visible regardless of axes scale.  Requires ``set_aspect("equal")``.
    """
    from matplotlib.patches import FancyArrowPatch

    (x0, y0), (x1, y1) = src_pos, tgt_pos
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-9:
        return
    ux, uy = dx / length, dy / length
    gap_src = NODE_R + 0.05
    gap_tgt = NODE_R + 0.09

    colour = CREATION_EDGE_COLOR if creation else ARROW_COLOR
    ax.add_patch(
        FancyArrowPatch(
            (x0 + ux * gap_src, y0 + uy * gap_src),
            (x1 - ux * gap_tgt, y1 - uy * gap_tgt),
            arrowstyle="-|>,head_length=7,head_width=3.4",
            mutation_scale=1.0,
            linewidth=2.1 if creation else 1.5,
            edgecolor=colour,
            facecolor=colour,
            zorder=1,
        )
    )


class MatplotlibDagBackend(DagVizBackend):
    """Matplotlib-based layered DAG drawing backend.

    Registered under the name ``"matplotlib"`` at module import time.
    """

    @property
    def name(self) -> str:
        return "matplotlib"

    def draw(
        self,
        dag: LabeledDAG,
        ax: Axes,
        *,
        node_colors: dict[int, str] | None = None,
        reachable: frozenset[int] = frozenset(),
        layout: dict[int, Position] | None = None,
    ) -> dict[int, Position]:
        """Draw ``dag`` on ``ax``.

        Parameters
        ----------
        dag:
            The DAG to draw.
        ax:
            Target axes. ``set_aspect("equal")`` is enforced; callers that
            set a different aspect ratio will see it overridden.
        node_colors:
            Optional per-node face colour overrides.
        reachable:
            Nodes satisfying the reachability condition; drawn with a green
            ring when not VAR nodes.
        layout:
            Fixed node positions (data units). Computed via
            :func:`_layered_layout` when ``None``.

        Returns
        -------
        dict[int, Position]
            The node positions used, keyed by node ID.
        """
        used_layout: dict[int, Position] = layout if layout is not None else _layered_layout(dag)

        ax.set_aspect("equal")
        ax.axis("off")

        # Draw edges first (lower z-order).
        creation_edges: set[tuple[int, int]] = set()
        # An edge is a creation edge if it leads into a CONST node that is the
        # only incoming edge and the source is a VAR.  For figure purposes we
        # colour ALL in-edges of CONST nodes in the creation colour.
        for node in range(dag.node_count):
            if dag.node_label(node) is NodeType.CONST:
                for src in dag.in_neighbors(node):
                    creation_edges.add((src, node))

        for node in range(dag.node_count):
            for tgt in dag.out_neighbors(node):
                if node in used_layout and tgt in used_layout:
                    _draw_edge(
                        ax,
                        used_layout[node],
                        used_layout[tgt],
                        creation=(node, tgt) in creation_edges,
                    )

        # Determine which CONST nodes should trigger an alert ring (in-degree 0).
        alert_nodes: set[int] = set()
        for node in range(dag.node_count):
            if dag.node_label(node) is NodeType.CONST and dag.in_degree(node) == 0:
                alert_nodes.add(node)

        # Draw nodes.
        for node in range(dag.node_count):
            if node in used_layout:
                _draw_node(
                    ax,
                    node,
                    used_layout[node],
                    dag,
                    reachable=reachable,
                    node_colors=node_colors,
                    alert=node in alert_nodes,
                )

        return used_layout


# Self-registration on import.
register_backend("matplotlib", MatplotlibDagBackend)
