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
from isalsr.viz.label_fit import NODE_LABEL_GID
from isalsr.viz.style import (
    NEW_NODE_ACCENT,
    PRIMARY_PTR_COLOR,
    SECONDARY_PTR_COLOR,
    color_for_node_border,
    color_for_node_face,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any

# Colours for the two pointer markers.
_PRIMARY_COL: str = PRIMARY_PTR_COLOR
_SECONDARY_COL: str = SECONDARY_PTR_COLOR

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
    n_slots: int | None = None,
    accent_nodes: frozenset[int] = frozenset(),
    primary_label: str = "P",
    secondary_label: str = "S",
    show_wraparound: bool = False,
) -> None:
    """Draw the CDLL state as a horizontal chain on *ax*.

    Nodes are drawn left to right in CDLL traversal order anchored by
    :func:`stable_anchor`, so successive snapshots show insertions rather than
    rotations.  Pointer positions are marked with small coloured triangles
    above the relevant circle.

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
    n_slots:
        Fixed slot count for the x-axis.  When several snapshots of a growing
        list are drawn side by side, passing the final list length keeps the
        circles the same physical size in every panel; without it each panel
        rescales to its own length and the nodes appear to shrink as the list
        grows.
    accent_nodes:
        Graph-node IDs to surround with an amber halo, marking them as
        created by the step being displayed.
    primary_label:
        Text of the primary-pointer marker.  Use the symbol the manuscript
        assigns to the primary pointer.
    secondary_label:
        Text of the secondary-pointer marker.
    show_wraparound:
        When ``True``, close the chain with an arc from the last node back to
        the first, so the drawing states the list's defining property: it is
        *circular*, and ``next`` from the last element reaches the first. A
        plain left-to-right chain silently reads as a finite sequence with two
        ends, which is a different data structure from the one the algorithms
        traverse.
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
    # Centre the chain inside a fixed number of slots so every panel in a
    # multi-step figure shares one cell size.
    slots = n_slots if n_slots is not None else n
    x_offset = (slots - n) * cell_w / 2.0
    ax.set_xlim(-cell_w * 0.5, slots * cell_w - cell_w * 0.5)
    # The return arc needs room below the chain.
    ax.set_ylim(-0.62 if show_wraparound else -0.2, 1.4)

    ptr_to_col: dict[int, int] = {idx: col for col, (idx, _) in enumerate(traversal)}

    for col, (_cdll_idx, graph_node) in enumerate(traversal):
        cx = x_offset + col * cell_w
        nt = output_dag.node_label(graph_node)
        face = color_for_node_face(nt)
        border = color_for_node_border(nt)
        if graph_node in accent_nodes:
            ax.add_patch(
                Circle(
                    (cx, y_node),
                    node_r + 0.09,
                    facecolor="none",
                    edgecolor=NEW_NODE_ACCENT,
                    linewidth=1.4,
                    zorder=2,
                )
            )
        ax.add_patch(
            Circle(
                (cx, y_node),
                node_r,
                facecolor=face,
                edgecolor=border,
                linewidth=1.0,
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
            gid=NODE_LABEL_GID,
        )

        # Draw arrow to next node.  The last node's successor is the first,
        # which is drawn separately as a return arc below the chain.
        next_col = (col + 1) % n
        nx = x_offset + next_col * cell_w
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

    # Return arc closing the ring: last -> first, routed below the chain.
    if show_wraparound and n > 1:
        x_last = x_offset + (n - 1) * cell_w
        x_first = x_offset
        ax.add_patch(
            FancyArrowPatch(
                (x_last, y_node - node_r - 0.04),
                (x_first, y_node - node_r - 0.04),
                # Negative rad bulges below the chain; a positive one would
                # route the arc back through the discs it is meant to skirt.
                connectionstyle="arc3,rad=-0.20",
                arrowstyle="-|>,head_length=4,head_width=2.2",
                mutation_scale=1.0,
                linewidth=0.8,
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
        _ptr_marker(
            ax,
            x_offset + pri_col * cell_w,
            y_node + node_r,
            f"{primary_label},{secondary_label}",
            _PRIMARY_COL,
            fs_ptr,
        )
    else:
        _ptr_marker(
            ax, x_offset + pri_col * cell_w, y_node + node_r, primary_label, _PRIMARY_COL, fs_ptr
        )
        _ptr_marker(
            ax,
            x_offset + sec_col * cell_w,
            y_node + node_r,
            secondary_label,
            _SECONDARY_COL,
            fs_ptr,
        )


def draw_cdll_ring(
    ax: Axes,
    cdll: CircularDoublyLinkedList,
    output_dag: LabeledDAG,
    primary_ptr: int,
    secondary_ptr: int,
    *,
    node_r: float = 0.32,
    radius: float = 1.0,
    ptr_room: float = 0.46,
    fs_label: float = _FS_LABEL,
    fs_ptr: float = _FS_PTR,
    accent_nodes: frozenset[int] = frozenset(),
    primary_label: str = "P",
    secondary_label: str = "S",
) -> None:
    """Draw the CDLL as a ring on *ax*.

    Nodes are placed on a circle in traversal order, starting at the top and
    proceeding clockwise, with one arrow per ``next`` link.  The ring closes on
    itself, which is the structure's defining property: a chain drawn with two
    free ends is a different data structure from the one the algorithms
    traverse, and the pointer arithmetic in both S2D and D2S only makes sense
    because ``next`` from the last element reaches the first.

    Traversal is anchored by :func:`stable_anchor` rather than by the primary
    pointer, so the ring does not appear to rotate between successive snapshots
    when only a pointer moved.

    Parameters
    ----------
    ax:
        Target axes.  Aspect is set to ``"equal"`` so the ring stays circular.
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
    radius:
        Ring radius in data units.  Held fixed across snapshots so the discs
        keep one physical size as the list grows.
    ptr_room:
        Radial room reserved outside the ring for the pointer markers.
    fs_label:
        Font size for node text labels.
    fs_ptr:
        Font size for pointer marker text.
    accent_nodes:
        Graph-node IDs to surround with an amber halo.
    primary_label:
        Text of the primary-pointer marker.
    secondary_label:
        Text of the secondary-pointer marker.
    """
    import math

    from matplotlib.patches import Circle, FancyArrowPatch

    traversal = cdll_traversal(cdll, stable_anchor(cdll, primary_ptr))
    n = len(traversal)

    ax.set_aspect("equal")
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

    extent = radius + node_r + ptr_room
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)

    # Start at 12 o'clock and advance clockwise, so `next` reads the way a
    # clock does rather than the way a unit circle does.
    def centre(i: int) -> tuple[float, float]:
        if n == 1:
            return (0.0, 0.0)
        theta = math.pi / 2.0 - 2.0 * math.pi * i / n
        return (radius * math.cos(theta), radius * math.sin(theta))

    positions = [centre(i) for i in range(n)]
    ptr_to_slot: dict[int, int] = {idx: i for i, (idx, _) in enumerate(traversal)}

    # ---- next-links ----
    for i in range(n if n > 1 else 0):
        j = (i + 1) % n
        x0, y0 = positions[i]
        x1, y1 = positions[j]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        ux, uy = dx / length, dy / length
        # A two-element ring has one chord shared by both links; bow them apart
        # so the reader sees two directed edges rather than one.
        rad = 0.45 if n == 2 else 0.0
        ax.add_patch(
            FancyArrowPatch(
                (x0 + ux * (node_r + 0.04), y0 + uy * (node_r + 0.04)),
                (x1 - ux * (node_r + 0.07), y1 - uy * (node_r + 0.07)),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>,head_length=3.0,head_width=1.8",
                mutation_scale=1.0,
                linewidth=0.85,
                edgecolor="#666666",
                facecolor="#666666",
                zorder=2,
            )
        )

    # ---- nodes ----
    for i, (_cdll_idx, graph_node) in enumerate(traversal):
        cx, cy = positions[i]
        nt = output_dag.node_label(graph_node)
        if graph_node in accent_nodes:
            ax.add_patch(
                Circle(
                    (cx, cy),
                    node_r + 0.09,
                    facecolor="none",
                    edgecolor=NEW_NODE_ACCENT,
                    linewidth=1.4,
                    zorder=3,
                )
            )
        ax.add_patch(
            Circle(
                (cx, cy),
                node_r,
                facecolor=color_for_node_face(nt),
                edgecolor=color_for_node_border(nt),
                linewidth=1.0,
                zorder=4,
            )
        )
        ax.text(
            cx,
            cy,
            _node_short_label(output_dag, graph_node),
            ha="center",
            va="center",
            fontsize=fs_label,
            zorder=5,
            gid=NODE_LABEL_GID,
        )

    # ---- pointer markers, placed radially outside the ring ----
    pri_slot = ptr_to_slot.get(primary_ptr, 0)
    sec_slot = ptr_to_slot.get(secondary_ptr, 0)
    if primary_ptr == secondary_ptr:
        _radial_ptr_marker(
            ax,
            positions[pri_slot],
            node_r,
            f"{primary_label},{secondary_label}",
            _PRIMARY_COL,
            fs_ptr,
        )
    else:
        _radial_ptr_marker(ax, positions[pri_slot], node_r, primary_label, _PRIMARY_COL, fs_ptr)
        _radial_ptr_marker(ax, positions[sec_slot], node_r, secondary_label, _SECONDARY_COL, fs_ptr)


def _radial_ptr_marker(
    ax: Axes,
    pos: tuple[float, float],
    node_r: float,
    label: str,
    colour: str,
    fs: float,
) -> None:
    """Draw a pointer arrow aimed inward at the ring node at *pos*."""
    import math

    from matplotlib.patches import FancyArrowPatch

    cx, cy = pos
    norm = math.hypot(cx, cy)
    ux, uy = (0.0, 1.0) if norm < 1e-9 else (cx / norm, cy / norm)

    tail = (cx + ux * (node_r + 0.40), cy + uy * (node_r + 0.40))
    head = (cx + ux * (node_r + 0.07), cy + uy * (node_r + 0.07))
    ax.add_patch(
        FancyArrowPatch(
            tail,
            head,
            arrowstyle="-|>,head_length=3.2,head_width=2.0",
            mutation_scale=1.0,
            linewidth=0.85,
            edgecolor=colour,
            facecolor=colour,
            zorder=6,
        )
    )
    ax.text(
        cx + ux * (node_r + 0.50),
        cy + uy * (node_r + 0.50),
        label,
        ha="center" if abs(ux) < 0.35 else ("left" if ux > 0 else "right"),
        va="center" if abs(uy) < 0.35 else ("bottom" if uy > 0 else "top"),
        fontsize=fs,
        color=colour,
        fontweight="bold",
        zorder=7,
    )


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

    tip_y = base_y + 0.34
    ax.add_patch(
        FancyArrowPatch(
            (cx, tip_y),
            (cx, base_y + 0.05),
            arrowstyle="-|>,head_length=4,head_width=2.4",
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
