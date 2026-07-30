"""Draw a CDLL ring as a compact horizontal chain with pointer markers.

The CDLL state captured in a D2S trace snapshot (primary_ptr, secondary_ptr
and the list of graph-node values in traversal order) is rendered as a row of
small labelled circles connected by arrows.  Primary and secondary pointers are
marked with coloured triangles above the relevant node.

Dependency rule: only :mod:`isalsr.core` and :mod:`isalsr.viz.style`.
matplotlib is imported inside function bodies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isalsr.core.cdll import CircularDoublyLinkedList
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.viz.style import color_for_node_border, color_for_node_face

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any

# Colours for the two pointer markers.
_PRIMARY_COL: str = "#2c6fad"
_SECONDARY_COL: str = "#1b7f4b"

_NODE_R: float = 0.30
_CELL_W: float = 1.0
_FS_LABEL: float = 11.0
_FS_PTR: float = 9.0


def _node_short_label(dag: LabeledDAG, out_node: int) -> str:
    """Return a compact text label for *out_node* in the output DAG."""
    nt = dag.node_label(out_node)
    if nt is NodeType.VAR:
        idx = dag.node_data(out_node).get("var_index", 0)
        return rf"$x_{{{int(idx) + 1}}}$"
    if nt is NodeType.CONST:
        return r"$c$"
    short: dict[NodeType, str] = {
        NodeType.ADD: r"$+$",
        NodeType.MUL: r"$\times$",
        NodeType.SUB: r"$-$",
        NodeType.DIV: r"$\div$",
        NodeType.SIN: r"$\sin$",
        NodeType.COS: r"$\cos$",
        NodeType.EXP: r"$e^x$",
        NodeType.LOG: r"$\ln$",
        NodeType.SQRT: r"$\sqrt{}$",
        NodeType.POW: r"$\hat{}$",
        NodeType.ABS: r"$|\cdot|$",
        NodeType.NEG: r"$-x$",
        NodeType.INV: r"$1/x$",
    }
    return short.get(nt, str(nt.value))


def cdll_traversal(
    cdll: CircularDoublyLinkedList,
    start_ptr: int,
) -> list[tuple[int, int]]:
    """Return the CDLL in traversal order starting at *start_ptr*.

    Parameters
    ----------
    cdll:
        A :class:`~isalsr.core.cdll.CircularDoublyLinkedList` snapshot.
    start_ptr:
        The CDLL index to begin the circular traversal from.

    Returns
    -------
    list[tuple[int, int]]
        Pairs ``(cdll_index, graph_node_value)`` in circular traversal order
        beginning at *start_ptr*.
    """
    if cdll.size() == 0:
        return []
    nodes: list[tuple[int, int]] = []
    ptr = start_ptr
    while True:
        nodes.append((ptr, cdll.get_value(ptr)))
        ptr = cdll.next_node(ptr)
        if ptr == start_ptr:
            break
    return nodes


def stable_anchor(cdll: CircularDoublyLinkedList, any_ptr: int) -> int:
    """Return a traversal start that does not move as the pointers move.

    Rendering successive snapshots from the *primary pointer* makes the chain
    appear to rotate whenever that pointer advances, even though the list
    itself is unchanged.  Anchoring instead on the lowest graph-node value
    keeps the drawn order stable, so the reader sees insertions rather than
    rotations.

    The anchor is well defined for any D2S trace: the ``m`` variable nodes are
    pre-inserted before any instruction executes (Critical Invariant 7) and are
    never removed, so graph node ``0`` (that is, ``x_1``) is always present.

    Parameters
    ----------
    cdll:
        The CDLL snapshot.
    any_ptr:
        Any valid CDLL index, used to enter the ring.

    Returns
    -------
    int
        The CDLL index holding the lowest graph-node value.
    """
    entries = cdll_traversal(cdll, any_ptr)
    if not entries:
        return any_ptr
    return min(entries, key=lambda pair: pair[1])[0]


def draw_cdll(
    ax: Axes,
    cdll: CircularDoublyLinkedList,
    output_dag: LabeledDAG,
    primary_ptr: int,
    secondary_ptr: int,
    *,
    node_r: float = _NODE_R,
    cell_w: float = _CELL_W,
    fs_label: float = _FS_LABEL,
    fs_ptr: float = _FS_PTR,
) -> None:
    """Draw the CDLL state as a horizontal chain on *ax*.

    Nodes are drawn left to right in CDLL traversal order starting at the
    primary pointer.  Primary (P) and secondary (S) pointer positions are
    marked with small coloured triangles above the relevant circle.

    Parameters
    ----------
    ax:
        Target axes.  Aspect and ticks are set inside this function.
    cdll:
        The CDLL snapshot to draw.
    output_dag:
        The output DAG snapshot corresponding to *cdll*; used for node labels.
    primary_ptr:
        CDLL index of the primary pointer.
    secondary_ptr:
        CDLL index of the secondary pointer.
    node_r:
        Node circle radius in data units.
    cell_w:
        Horizontal spacing between node centres in data units.
    fs_label:
        Font size for node text labels.
    fs_ptr:
        Font size for pointer marker text.
    """
    from matplotlib.patches import Circle, FancyArrowPatch

    traversal = cdll_traversal(cdll, stable_anchor(cdll, primary_ptr))
    n = len(traversal)

    ax.set_aspect("auto")
    ax.axis("off")

    if n == 0:
        ax.text(
            0.5,
            0.5,
            "(empty)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=fs_label,
            color="#888888",
        )
        return

    y_node = 0.5
    total_w = n * cell_w
    ax.set_xlim(-cell_w * 0.4, total_w - cell_w * 0.6)
    ax.set_ylim(-0.2, 1.4)

    ptr_to_col: dict[int, int] = {idx: col for col, (idx, _) in enumerate(traversal)}

    for col, (_cdll_idx, graph_node) in enumerate(traversal):
        cx = col * cell_w
        nt = output_dag.node_label(graph_node)
        face = color_for_node_face(nt)
        border = color_for_node_border(nt)
        ax.add_patch(
            Circle(
                (cx, y_node),
                node_r,
                facecolor=face,
                edgecolor=border,
                linewidth=1.2,
                zorder=3,
            )
        )
        ax.text(
            cx,
            y_node,
            _node_short_label(output_dag, graph_node),
            ha="center",
            va="center",
            fontsize=fs_label,
            zorder=4,
        )

        # Draw arrow to next node (circular).
        next_col = (col + 1) % n
        nx = next_col * cell_w
        # For the last node in a multi-node list, draw a curved return arrow.
        if col < n - 1:
            x0 = cx + node_r + 0.05
            x1 = nx - node_r - 0.05
            ax.add_patch(
                FancyArrowPatch(
                    (x0, y_node),
                    (x1, y_node),
                    arrowstyle="-|>,head_length=5,head_width=2.5",
                    mutation_scale=1.0,
                    linewidth=0.9,
                    edgecolor="#666666",
                    facecolor="#666666",
                    zorder=2,
                )
            )

    # Pointer markers above nodes.
    pri_col = ptr_to_col.get(primary_ptr, 0)
    sec_col = ptr_to_col.get(secondary_ptr, 0)

    # When both pointers sit on the same node, draw ONE combined marker. Drawing
    # "P" unconditionally and then "P,S" on top of it renders as overlapping
    # glyphs rather than as a label.
    if secondary_ptr == primary_ptr:
        _ptr_marker(ax, pri_col * cell_w, y_node + node_r, "P,S", _PRIMARY_COL, fs_ptr)
    else:
        _ptr_marker(ax, pri_col * cell_w, y_node + node_r, "P", _PRIMARY_COL, fs_ptr)
        _ptr_marker(ax, sec_col * cell_w, y_node + node_r, "S", _SECONDARY_COL, fs_ptr)


def _ptr_marker(
    ax: Axes,
    cx: float,
    base_y: float,
    label: str,
    colour: str,
    fs: float,
) -> None:
    """Draw a small upward-pointing triangle with *label* above *(cx, base_y)*."""
    from matplotlib.patches import FancyArrowPatch

    tip_y = base_y + 0.42
    ax.add_patch(
        FancyArrowPatch(
            (cx, tip_y),
            (cx, base_y + 0.06),
            arrowstyle="-|>,head_length=5,head_width=3",
            mutation_scale=1.0,
            linewidth=0.8,
            edgecolor=colour,
            facecolor=colour,
            zorder=5,
        )
    )
    ax.text(
        cx,
        tip_y + 0.06,
        label,
        ha="center",
        va="bottom",
        fontsize=fs,
        color=colour,
        fontweight="bold",
        zorder=6,
    )
