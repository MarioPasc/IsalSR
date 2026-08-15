"""Matplotlib backend for labeled-DAG drawing.

Draws a :class:`~isalsr.core.labeled_dag.LabeledDAG` as a layered diagram:
nodes as labelled discs, edges as trimmed arrows with proper arrowheads.

Arrow trimming is done in **data units** rather than via ``shrinkA``/``shrinkB``
(which are in points and silently produce headless lines). This requires
``set_aspect("equal")`` on the axes; the backend enforces it.

Nodes and edges may be drawn as *ghosts* -- dashed, desaturated outlines that
mark structure belonging to a target DAG that a partially completed run has not
yet reproduced. This is what lets a trace figure hold the node positions of the
finished DAG fixed across every step while still showing what is done and what
is pending.

The drawing style matches ``experiments/scripts/generate_fig_const_normalization.py``
so all review figures read as one visual system.

Registers itself under the name ``"matplotlib"`` on import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.viz.base import DagVizBackend, Position
from isalsr.viz.label_fit import NODE_LABEL_GID
from isalsr.viz.registry import register_backend
from isalsr.viz.style import (
    ALERT_COLOR,
    ARROW_COLOR,
    CREATION_EDGE_COLOR,
    GHOST_EDGE_COLOR,
    GHOST_FACE,
    GHOST_TEXT_COLOR,
    NEW_NODE_ACCENT,
    REACHABLE_TINT,
    color_for_node_border,
    color_for_node_face,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.path import Path as MplPath
else:
    Axes = Any
    MplPath = Any

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

    Node *order* within a layer follows node ID, so this rule makes no attempt
    to reduce edge crossings; use
    :class:`~isalsr.viz.backends.graphviz_dag.GraphvizDagBackend` when
    crossings would mislead the reader.
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


def _spline_path(points: list[Position]) -> MplPath | None:
    """Build a matplotlib path from Graphviz B-spline control points.

    Graphviz emits ``1 + 3k`` points per edge: an initial point followed by
    cubic Bezier triples.  A point count outside that family means the record
    was not what this function assumes, so it returns ``None`` and the caller
    falls back to a straight arrow rather than drawing a wrong curve.

    Parameters
    ----------
    points:
        Control points in data units.

    Returns
    -------
    matplotlib.path.Path | None
        The path, or ``None`` when ``points`` is not a valid cubic sequence.
    """
    from matplotlib.path import Path

    if len(points) < 4 or (len(points) - 1) % 3 != 0:
        return None
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(points) - 1)
    return Path(points, codes)


class MatplotlibDagBackend(DagVizBackend):
    """Matplotlib-based layered DAG drawing backend.

    Registered under the name ``"matplotlib"`` at module import time.

    Attributes:
        node_r: Node-disc radius in data units.
        fs_node: Font size of the glyph printed inside each disc.
    """

    def __init__(self, node_r: float = NODE_R, fs_node: float = FS_NODE) -> None:
        """Store drawing metrics used by subsequent draw calls.

        Parameters
        ----------
        node_r:
            Node-disc radius in data units.
        fs_node:
            Font size of the glyph printed inside each disc, in points.
        """
        self.node_r = node_r
        self.fs_node = fs_node

    @property
    def name(self) -> str:
        return "matplotlib"

    def compute_layout(self, dag: LabeledDAG) -> dict[int, Position]:
        """Return node positions for ``dag`` when the caller pins none.

        Subclasses override this to swap the placement rule while keeping
        every drawing primitive of this class, so alternative layouts stay
        visually interchangeable.
        """
        return _layered_layout(dag)

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def _draw_node(
        self,
        ax: Axes,
        node: int,
        pos: Position,
        dag: LabeledDAG,
        *,
        reachable: frozenset[int],
        node_colors: dict[int, str] | None,
        alert: bool,
        ghost: bool,
        accent: bool,
        dim: float = 1.0,
    ) -> None:
        """Draw one node disc with optional accent, reachability tint or alert ring.

        ``dim`` below 1.0 renders the node spent -- present, and drawn in full,
        but no longer carrying information because it has been consumed. That is
        a different claim from ``ghost``, which says the node is not there yet.
        """
        from matplotlib.patches import Circle

        nt = dag.node_label(node)
        x, y = pos

        if ghost:
            ax.add_patch(
                Circle(
                    (x, y),
                    self.node_r,
                    facecolor=GHOST_FACE,
                    edgecolor=GHOST_EDGE_COLOR,
                    linewidth=1.0,
                    linestyle=(0, (2.6, 2.0)),
                    zorder=3,
                )
            )
            ax.text(
                x,
                y,
                _node_display_label(dag, node),
                ha="center",
                va="center",
                fontsize=self.fs_node,
                color=GHOST_TEXT_COLOR,
                zorder=4,
                gid=NODE_LABEL_GID,
            )
            return

        face = (
            node_colors[node] if (node_colors and node in node_colors) else color_for_node_face(nt)
        )
        border = color_for_node_border(nt)

        if accent:
            # Amber halo for the node created by the step being displayed.
            ax.add_patch(
                Circle(
                    (x, y),
                    self.node_r + 0.13,
                    facecolor="none",
                    edgecolor=NEW_NODE_ACCENT,
                    linewidth=2.0,
                    zorder=2,
                )
            )
        elif alert:
            # Dashed red ring for violating (orphan CONST) nodes.
            ax.add_patch(
                Circle(
                    (x, y),
                    self.node_r + 0.11,
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
                    self.node_r + 0.09,
                    facecolor="none",
                    edgecolor=REACHABLE_TINT,
                    linewidth=1.8,
                    zorder=2,
                )
            )

        ax.add_patch(
            Circle(
                (x, y),
                self.node_r,
                facecolor=face,
                edgecolor=border,
                linewidth=1.2,
                alpha=dim,
                zorder=3,
            )
        )
        ax.text(
            x,
            y,
            _node_display_label(dag, node),
            ha="center",
            va="center",
            fontsize=self.fs_node,
            alpha=dim,
            zorder=4,
            gid=NODE_LABEL_GID,
        )

    def edge_route(self, src: int, tgt: int) -> list[Position] | None:
        """Return spline control points for edge ``src -> tgt``, or ``None``.

        The base class has no router and always returns ``None``, so edges are
        drawn as straight trimmed arrows.  Subclasses backed by a layout engine
        that routes edges around nodes override this.
        """
        return None

    def _draw_edge(
        self,
        ax: Axes,
        src_pos: Position,
        tgt_pos: Position,
        *,
        creation: bool,
        ghost: bool,
        route: list[Position] | None = None,
        dim: float = 1.0,
    ) -> None:
        """Draw a directed edge, following ``route`` when one is supplied.

        Without a route the edge is a straight arrow whose endpoints are
        shortened in data units (not points) so arrowheads stay visible
        regardless of axes scale.  Requires ``set_aspect("equal")``.

        With a route the edge follows the supplied B-spline control points,
        which the layout engine has already clipped to the node boundaries and
        routed clear of intervening nodes.
        """
        from matplotlib.patches import FancyArrowPatch

        (x0, y0), (x1, y1) = src_pos, tgt_pos
        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-9:
            return
        ux, uy = dx / length, dy / length
        gap_src = self.node_r + 0.05
        gap_tgt = self.node_r + 0.09

        colour: str
        lw: float
        style: str | tuple[int, tuple[float, ...]]
        if ghost:
            colour, lw, style = GHOST_EDGE_COLOR, 0.9, (0, (2.6, 2.0))
        elif creation:
            colour, lw, style = CREATION_EDGE_COLOR, 1.6, "solid"
        else:
            colour, lw, style = ARROW_COLOR, 1.3, "solid"

        arrowstyle = "-|>,head_length=4.0,head_width=2.0"
        spline = _spline_path(route) if route else None
        if spline is not None:
            ax.add_patch(
                FancyArrowPatch(
                    path=spline,
                    arrowstyle=arrowstyle,
                    mutation_scale=1.0,
                    linewidth=lw,
                    linestyle=style,
                    edgecolor=colour,
                    facecolor="none",
                    alpha=dim,
                    zorder=1,
                )
            )
            return

        ax.add_patch(
            FancyArrowPatch(
                (x0 + ux * gap_src, y0 + uy * gap_src),
                (x1 - ux * gap_tgt, y1 - uy * gap_tgt),
                arrowstyle=arrowstyle,
                mutation_scale=1.0,
                linewidth=lw,
                linestyle=style,
                edgecolor=colour,
                facecolor=colour,
                alpha=dim,
                zorder=1,
            )
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def draw(
        self,
        dag: LabeledDAG,
        ax: Axes,
        *,
        node_colors: dict[int, str] | None = None,
        reachable: frozenset[int] = frozenset(),
        layout: dict[int, Position] | None = None,
        ghost_dag: LabeledDAG | None = None,
        accent_nodes: frozenset[int] = frozenset(),
        dim_nodes: frozenset[int] = frozenset(),
        dim_edges: frozenset[tuple[int, int]] = frozenset(),
        dim_alpha: float = 0.28,
        alert_nodes: frozenset[int] = frozenset(),
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
            :meth:`compute_layout` when ``None``.
        ghost_dag:
            A superset DAG whose nodes and edges are drawn as dashed ghosts
            wherever ``dag`` does not yet contain them.  Node IDs must agree
            with ``dag``.  ``layout`` must cover every ghost node.
        accent_nodes:
            Node IDs to surround with an amber halo, marking them as created
            by the step being displayed.
        dim_nodes:
            Node IDs already consumed by the step being displayed. They are
            drawn faded rather than dashed: the node is there, it just no
            longer holds information the algorithm has yet to move.
        dim_edges:
            Edges already consumed, faded on the same grounds.
        dim_alpha:
            Opacity applied to consumed nodes and edges.
        alert_nodes:
            Extra node IDs to ring in the alert colour, on top of the
            in-degree-0 CONST nodes the backend detects itself. Callers use it
            to flag defects only they can judge, such as a node whose in-degree
            exceeds its operator's arity.

        Returns
        -------
        dict[int, Position]
            The node positions used, keyed by node ID.
        """
        used_layout: dict[int, Position] = (
            layout if layout is not None else self.compute_layout(dag)
        )

        ax.set_aspect("equal")
        ax.axis("off")

        # ---- Ghost layer (drawn first, so real structure paints over it) ----
        if ghost_dag is not None:
            real_edges = {(s, t) for s in range(dag.node_count) for t in dag.out_neighbors(s)}
            for src in range(ghost_dag.node_count):
                for tgt in ghost_dag.out_neighbors(src):
                    if (src, tgt) in real_edges:
                        continue
                    if src in used_layout and tgt in used_layout:
                        self._draw_edge(
                            ax,
                            used_layout[src],
                            used_layout[tgt],
                            creation=False,
                            ghost=True,
                            route=self.edge_route(src, tgt),
                        )
            for node in range(dag.node_count, ghost_dag.node_count):
                if node in used_layout:
                    self._draw_node(
                        ax,
                        node,
                        used_layout[node],
                        ghost_dag,
                        reachable=frozenset(),
                        node_colors=None,
                        alert=False,
                        ghost=True,
                        accent=False,
                    )

        # ---- Real edges ----
        # Every in-edge of a CONST node is drawn in the creation-edge colour.
        creation_edges: set[tuple[int, int]] = set()
        for node in range(dag.node_count):
            if dag.node_label(node) is NodeType.CONST:
                for src in dag.in_neighbors(node):
                    creation_edges.add((src, node))

        for node in range(dag.node_count):
            for tgt in dag.out_neighbors(node):
                if node in used_layout and tgt in used_layout:
                    self._draw_edge(
                        ax,
                        used_layout[node],
                        used_layout[tgt],
                        creation=(node, tgt) in creation_edges,
                        ghost=False,
                        route=self.edge_route(node, tgt),
                        dim=dim_alpha if (node, tgt) in dim_edges else 1.0,
                    )

        # Alert ring: in-degree-0 CONST nodes, plus whatever the caller flagged.
        alerts: set[int] = set(alert_nodes)
        for node in range(dag.node_count):
            if dag.node_label(node) is NodeType.CONST and dag.in_degree(node) == 0:
                alerts.add(node)

        # ---- Real nodes ----
        for node in range(dag.node_count):
            if node in used_layout:
                self._draw_node(
                    ax,
                    node,
                    used_layout[node],
                    dag,
                    reachable=reachable,
                    node_colors=node_colors,
                    alert=node in alerts,
                    ghost=False,
                    accent=node in accent_nodes,
                    dim=dim_alpha if node in dim_nodes else 1.0,
                )

        return used_layout


# Self-registration on import.
register_backend("matplotlib", MatplotlibDagBackend)
