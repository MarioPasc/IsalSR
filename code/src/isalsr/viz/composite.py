"""Two-row D2S trace figure assembler.

Draws a 2 x N grid where each column is one step of the D2S algorithm.
Row 0 shows the un-normalised DAG run (which stalls); row 1 shows the
normalised run (which completes).  The stall column in row 0 carries a red
frame and a ghost rendering of the node that could not be encoded.

When :attr:`~isalsr.viz.layout.TraceLayout.show_cdll` is ``True`` (the
default) each column contains three sub-panels: the DAG, a CDLL ring panel
showing the circular doubly-linked list state and pointer positions, and the
instruction strip.  When ``False`` pointer positions are shown via text
annotations on the DAG nodes and the CDLL row is omitted.

Dependency rule: imports :mod:`isalsr.core`, :mod:`isalsr.viz.style`,
:mod:`isalsr.viz.layout`, and :mod:`isalsr.viz.cdll_view`.
matplotlib is imported inside function bodies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isalsr.core.cdll import CircularDoublyLinkedList
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.viz.cdll_view import draw_cdll
from isalsr.viz.layout import TraceLayout
from isalsr.viz.style import (
    ALERT_COLOR,
    ARROW_COLOR,
    color_for_node_border,
    color_for_node_face,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:
    Axes = Any
    Figure = Any

# Colour for the stall-column frame.
_STALL_FRAME: str = "#b02a2a"
# Alpha for greyed-out future-token cells.
_FUTURE_ALPHA: float = 0.22
# Colour for pointer annotations (used when show_cdll=False).
_COL_PRIMARY: str = "#2c6fad"
_COL_SECONDARY: str = "#1b7f4b"

Snapshot = tuple[LabeledDAG, CircularDoublyLinkedList, int, int, str]


# ---------------------------------------------------------------------------
# Node-position helpers
# ---------------------------------------------------------------------------


def _label_for_output_node(dag: LabeledDAG, node: int) -> str:
    """Return a display label for *node* in the output DAG."""
    nt = dag.node_label(node)
    if nt is NodeType.VAR:
        idx = dag.node_data(node).get("var_index", 0)
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
        NodeType.SQRT: r"$\sqrt{\cdot}$",
        NodeType.POW: r"$\hat{\phantom{x}}$",
        NodeType.ABS: r"$|\cdot|$",
        NodeType.NEG: r"$-x$",
        NodeType.INV: r"$1/x$",
    }
    return short.get(nt, str(nt.value))


def assign_positions(
    dag: LabeledDAG,
    input_layout: dict[int, tuple[float, float]],
    input_node_types: dict[int, NodeType],
) -> dict[int, tuple[float, float]]:
    """Map output-DAG node IDs to canvas positions by matching node types.

    The mapping relies on the fact that each node type appears at most once in
    the expression used for the reachability figure (``x_1 + c``).  For each
    output-DAG node, we find the input-DAG node with the same
    :class:`~isalsr.core.node_types.NodeType` and VAR index (when applicable)
    and inherit its position.

    Parameters
    ----------
    dag:
        The output DAG snapshot.
    input_layout:
        Canvas positions keyed by input-DAG node ID.
    input_node_types:
        Mapping from input-DAG node ID to its :class:`NodeType`.

    Returns
    -------
    dict[int, tuple[float, float]]
        Canvas positions keyed by output-DAG node ID.
    """
    # Build type_to_pos: (NodeType, disambiguator) -> position.
    type_to_pos: dict[tuple[NodeType, int | None], tuple[float, float]] = {}
    for inp_node, pos in input_layout.items():
        nt = input_node_types[inp_node]
        key: tuple[NodeType, int | None] = (nt, inp_node) if nt is NodeType.VAR else (nt, None)
        type_to_pos[key] = pos

    out_layout: dict[int, tuple[float, float]] = {}
    for out_node in range(dag.node_count):
        nt = dag.node_label(out_node)
        if nt is NodeType.VAR:
            idx = dag.node_data(out_node).get("var_index", 0)
            key = (nt, int(idx))
        else:
            key = (nt, None)
        if key in type_to_pos:
            out_layout[out_node] = type_to_pos[key]
    return out_layout


# ---------------------------------------------------------------------------
# Single-column DAG panel
# ---------------------------------------------------------------------------


def draw_dag_panel(
    ax: Axes,
    snapshot: Snapshot,
    *,
    input_layout: dict[int, tuple[float, float]],
    input_node_types: dict[int, NodeType],
    ax_xlim: tuple[float, float],
    ax_ylim: tuple[float, float],
    ghost_nodes: list[tuple[NodeType, tuple[float, float]]] | None = None,
    stall: bool = False,
    show_ptr_markers: bool = True,
    layout_obj: TraceLayout | None = None,
) -> None:
    """Render one DAG column panel.

    Parameters
    ----------
    ax:
        Target axes.
    snapshot:
        A D2S trace snapshot ``(output_dag, cdll, primary_ptr, secondary_ptr,
        emitted_string)``.
    input_layout:
        Fixed positions keyed by input-DAG node ID.
    input_node_types:
        Node types keyed by input-DAG node ID.
    ax_xlim:
        Axes x limits (data units).
    ax_ylim:
        Axes y limits (data units).
    ghost_nodes:
        List of ``(NodeType, position)`` pairs for nodes that exist in the
        input DAG but could not be added to the output DAG.  Drawn as dashed
        alert circles to indicate the stall cause.
    stall:
        When ``True``, the panel carries a red background tint and annotations
        indicating the algorithm could not proceed.
    show_ptr_markers:
        When ``True``, draw P/S text annotations below pointer nodes.  Set to
        ``False`` when a CDLL panel is present in the same column, since the
        CDLL panel is the single source of pointer-position information.
    layout_obj:
        :class:`TraceLayout` for font and geometry constants.
    """
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

    lo = layout_obj or TraceLayout()
    output_dag, cdll, primary_ptr, secondary_ptr, _ = snapshot

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(*ax_xlim)
    ax.set_ylim(*ax_ylim)

    if stall:
        ax.add_patch(
            FancyBboxPatch(
                (ax_xlim[0] + 0.01, ax_ylim[0] + 0.01),
                ax_xlim[1] - ax_xlim[0] - 0.02,
                ax_ylim[1] - ax_ylim[0] - 0.02,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                transform=ax.transData,
                facecolor="#fff4f4",
                edgecolor=_STALL_FRAME,
                linewidth=2.2,
                zorder=0,
                clip_on=False,
            )
        )

    out_layout = assign_positions(output_dag, input_layout, input_node_types)

    # Identify creation edges (VAR -> CONST) for colour.
    creation_edges: set[tuple[int, int]] = set()
    for nd in range(output_dag.node_count):
        if output_dag.node_label(nd) is NodeType.CONST:
            for src in output_dag.in_neighbors(nd):
                creation_edges.add((src, nd))

    # Draw edges.
    for src in range(output_dag.node_count):
        for tgt in output_dag.out_neighbors(src):
            if src in out_layout and tgt in out_layout:
                (x0, y0), (x1, y1) = out_layout[src], out_layout[tgt]
                dx, dy = x1 - x0, y1 - y0
                length = (dx * dx + dy * dy) ** 0.5
                if length < 1e-9:
                    continue
                ux, uy = dx / length, dy / length
                gap_s = lo.node_r + 0.06
                gap_t = lo.node_r + 0.10
                col = "#1b7f4b" if (src, tgt) in creation_edges else ARROW_COLOR
                lw = 2.2 if (src, tgt) in creation_edges else 1.6
                ax.add_patch(
                    FancyArrowPatch(
                        (x0 + ux * gap_s, y0 + uy * gap_s),
                        (x1 - ux * gap_t, y1 - uy * gap_t),
                        arrowstyle="-|>,head_length=7,head_width=3.4",
                        mutation_scale=1.0,
                        linewidth=lw,
                        edgecolor=col,
                        facecolor=col,
                        zorder=1,
                    )
                )

    # Resolve which graph-node IDs the two pointers currently name.
    pri_graph = cdll.get_value(primary_ptr) if cdll.size() > 0 else -1
    sec_graph = cdll.get_value(secondary_ptr) if cdll.size() > 0 else -1

    # Draw nodes.
    for out_node in range(output_dag.node_count):
        if out_node not in out_layout:
            continue
        pos = out_layout[out_node]
        nt = output_dag.node_label(out_node)
        face = color_for_node_face(nt)
        border = color_for_node_border(nt)
        ax.add_patch(
            Circle(
                pos,
                lo.node_r,
                facecolor=face,
                edgecolor=border,
                linewidth=1.6,
                zorder=3,
            )
        )
        ax.text(
            pos[0],
            pos[1],
            _label_for_output_node(output_dag, out_node),
            ha="center",
            va="center",
            fontsize=lo.fs_node,
            zorder=4,
        )
        # Pointer markers: only when no CDLL panel is present in this column.
        is_pri = out_node == pri_graph
        is_sec = out_node == sec_graph
        if show_ptr_markers and (is_pri or is_sec):
            _draw_ptr_marker(
                ax,
                pos,
                lo.node_r,
                primary=is_pri,
                secondary=is_sec,
                same_node=(pri_graph == sec_graph),
                fs=lo.fs_annot,
            )

    # Ghost nodes (unreachable orphans).
    if ghost_nodes:
        for g_nt, g_pos in ghost_nodes:
            face = color_for_node_face(g_nt)
            border = ALERT_COLOR
            ax.add_patch(
                Circle(
                    g_pos,
                    lo.node_r,
                    facecolor=face,
                    edgecolor=border,
                    linewidth=1.6,
                    linestyle=(0, (4, 2.5)),
                    alpha=0.55,
                    zorder=3,
                )
            )
            lbl = r"$c$" if g_nt is NodeType.CONST else str(g_nt.value)
            ax.text(
                g_pos[0],
                g_pos[1],
                lbl,
                ha="center",
                va="center",
                fontsize=lo.fs_node,
                color="#888888",
                zorder=4,
            )
            # Annotation: explain why c is a ghost.
            ax.text(
                g_pos[0],
                g_pos[1] - lo.node_r - 0.30,
                r"no in-edge",
                ha="center",
                va="top",
                fontsize=lo.fs_annot * 0.88,
                color=ALERT_COLOR,
                zorder=5,
            )


def _draw_ptr_marker(
    ax: Axes,
    pos: tuple[float, float],
    node_r: float,
    *,
    primary: bool,
    secondary: bool,
    same_node: bool,
    fs: float,
) -> None:
    """Draw P / S / P,S marker below the node at *pos*.

    Used when the CDLL panel is absent (``show_cdll=False``) so the reader
    can see which nodes the two pointers currently name.
    """
    if primary and secondary and same_node:
        label, colour = "P,S", _COL_PRIMARY
    elif primary:
        label, colour = "P", _COL_PRIMARY
    else:
        label, colour = "S", _COL_SECONDARY
    ax.text(
        pos[0],
        pos[1] - node_r - 0.20,
        label,
        ha="center",
        va="top",
        fontsize=fs,
        color=colour,
        fontweight="bold",
        zorder=5,
    )


# ---------------------------------------------------------------------------
# Single-column strip panel
# ---------------------------------------------------------------------------


def draw_strip_panel(
    ax: Axes,
    emitted: str,
    *,
    final_string: str | None = None,
    stall: bool = False,
    n_slots: int | None = None,
    layout_obj: TraceLayout | None = None,
) -> None:
    """Draw the token-strip progress panel for one column.

    Parameters
    ----------
    ax:
        Target axes.
    emitted:
        Tokens emitted so far (the prefix of the complete string).
    final_string:
        The complete target string; when provided, future tokens are shown at
        low opacity so the reader can see the full trajectory.
    stall:
        When ``True``, a red '?' cell is appended to mark the halt.
    n_slots:
        Fixed axis slot count so every strip panel in the figure shares
        identical cell geometry regardless of how many tokens are emitted.
    layout_obj:
        :class:`TraceLayout` for font constants.
    """
    from matplotlib.patches import FancyBboxPatch

    from isalsr.core.string_to_dag import _tokenize

    lo = layout_obj or TraceLayout()

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Background: pink tint for stall panels; barely-visible grey when the panel
    # is empty (violated step 0: no emitted tokens, no future string).
    if stall:
        ax.set_facecolor("#fff4f4")
    elif not emitted and final_string is None:
        ax.set_facecolor("#f5f5f5")

    emitted_tokens = list(_tokenize(emitted)) if emitted else []
    future_tokens: list[str] = []
    if final_string:
        all_tokens = list(_tokenize(final_string))
        n_e = len(emitted_tokens)
        future_tokens = all_tokens[n_e:]

    stall_token = "?" if stall else None

    cell_w = 1.05
    cell_h = 1.10
    all_display = emitted_tokens + future_tokens + (["?"] if stall else [])
    n_total = max(len(all_display), 1)

    # Use a fixed axis width so every strip panel in the figure shares identical
    # cell geometry regardless of how many tokens have been emitted so far.
    display_slots = n_slots if n_slots is not None else n_total
    ax.set_xlim(-0.08, display_slots * cell_w + 0.08)
    ax.set_ylim(-0.05, cell_h + 0.05)
    ax.set_aspect("auto")

    # Draw emitted tokens (solid).
    for i, tok in enumerate(emitted_tokens):
        _strip_cell(ax, i, tok, cell_w, cell_h, lo.fs_strip, alpha=1.0)

    # Draw future tokens (faded).
    offset = len(emitted_tokens)
    for j, tok in enumerate(future_tokens):
        _strip_cell(ax, offset + j, tok, cell_w, cell_h, lo.fs_strip, alpha=_FUTURE_ALPHA)

    # Draw stall cell.
    if stall_token is not None:
        si = offset + len(future_tokens)
        ax.add_patch(
            FancyBboxPatch(
                (si * cell_w + 0.05, 0.05),
                cell_w - 0.10,
                cell_h - 0.10,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor="#fce8e8",
                edgecolor=_STALL_FRAME,
                linewidth=1.5,
                zorder=2,
            )
        )
        ax.text(
            si * cell_w + cell_w / 2,
            cell_h / 2,
            "?",
            ha="center",
            va="center",
            fontsize=lo.fs_strip * 0.85,
            color=_STALL_FRAME,
            fontweight="bold",
            family="monospace",
            zorder=3,
        )


def _strip_cell(
    ax: Axes,
    col: int,
    token: str,
    cell_w: float,
    cell_h: float,
    fs: float,
    alpha: float = 1.0,
) -> None:
    """Draw one token cell at column *col*."""
    from matplotlib.patches import FancyBboxPatch

    from isalsr.viz.style import color_for_token

    face = color_for_token(token)
    x = col * cell_w
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.05, 0.05),
            cell_w - 0.10,
            cell_h - 0.10,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=face,
            edgecolor="#333333",
            linewidth=0.6,
            alpha=alpha,
            zorder=2,
        )
    )
    ax.text(
        x + cell_w / 2,
        cell_h / 2,
        token,
        ha="center",
        va="center",
        fontsize=fs,
        color="#111111",
        rotation=0,
        family="monospace",
        alpha=alpha,
        zorder=3,
    )


# ---------------------------------------------------------------------------
# Pad column panels
# ---------------------------------------------------------------------------


def draw_pad_panel_dag(
    ax: Axes,
    *,
    ax_xlim: tuple[float, float],
    ax_ylim: tuple[float, float],
) -> None:
    """Draw a blank placeholder for pad columns that the algorithm never reached."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(*ax_xlim)
    ax.set_ylim(*ax_ylim)
    cx = (ax_xlim[0] + ax_xlim[1]) / 2.0
    cy = (ax_ylim[0] + ax_ylim[1]) / 2.0
    ax.text(
        cx,
        cy,
        "not\nreached",
        ha="center",
        va="center",
        fontsize=14.0,
        color="#cccccc",
        style="italic",
        linespacing=1.4,
    )


def draw_pad_panel_cdll(ax: Axes) -> None:
    """Draw a blank CDLL placeholder for pad columns."""
    ax.axis("off")


def draw_pad_panel_strip(ax: Axes) -> None:
    """Draw a blank strip placeholder for pad columns that the algorithm never reached."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Transparent background — no grey band that could read as a rendering artefact.


# ---------------------------------------------------------------------------
# Red stall frame overlay
# ---------------------------------------------------------------------------


def add_stall_frame(fig: Figure, ax_top: Axes, ax_bottom: Axes) -> None:
    """Draw one red rectangle spanning *ax_top* and *ax_bottom* in figure coordinates.

    The frame encloses the DAG panel (top) and the strip panel (bottom) of
    the stall column so they read as a single unit.  When a CDLL panel sits
    between them it is automatically enclosed by the same frame.
    """
    from matplotlib.patches import FancyBboxPatch

    bt = ax_top.get_position()
    bb = ax_bottom.get_position()
    x0 = min(bt.x0, bb.x0)
    y0 = bb.y0
    x1 = max(bt.x1, bb.x1)
    y1 = bt.y1
    pad = 0.005
    fig.add_artist(
        FancyBboxPatch(
            (x0 - pad, y0 - pad),
            (x1 - x0) + 2 * pad,
            (y1 - y0) + 2 * pad,
            boxstyle="round,pad=0,rounding_size=0",
            transform=fig.transFigure,
            facecolor="none",
            edgecolor=_STALL_FRAME,
            linewidth=2.8,
            clip_on=False,
            zorder=10,
        )
    )


# ---------------------------------------------------------------------------
# Row label helper
# ---------------------------------------------------------------------------


def draw_row_label(
    fig: Figure,
    ax_top: Axes,
    ax_bottom: Axes,
    text: str,
    *,
    fs: float = 12.0,
) -> None:
    """Write a rotated row label centred on the span from *ax_top* to *ax_bottom*.

    Parameters
    ----------
    fig:
        Parent figure.
    ax_top:
        The topmost axes of the row (DAG panel of the first column).
    ax_bottom:
        The bottommost axes of the row (strip panel of the first column).
    text:
        Label text.
    fs:
        Font size in source points.
    """
    bt = ax_top.get_position()
    bb = ax_bottom.get_position()
    mid_y = (bt.y1 + bb.y0) / 2.0
    x_left = bt.x0 - 0.015
    fig.text(
        x_left,
        mid_y,
        text,
        ha="right",
        va="center",
        fontsize=fs,
        rotation=90,
        transform=fig.transFigure,
    )


# ---------------------------------------------------------------------------
# Two-row trace figure assembler
# ---------------------------------------------------------------------------


def make_trace_figure(
    trace_r0: list[Snapshot],
    input_dag_r0: LabeledDAG,
    trace_r1: list[Snapshot],
    input_dag_r1: LabeledDAG,
    *,
    d2s_string: str,
    input_layout: dict[int, tuple[float, float]],
    ax_xlim: tuple[float, float] = (-2.3, 2.3),
    ax_ylim: tuple[float, float] = (-1.3, 3.5),
    layout_obj: TraceLayout | None = None,
) -> Figure:
    """Build the two-row D2S trace figure.

    Row 0 shows the violated DAG run; row 1 shows the normalised run.

    When :attr:`~isalsr.viz.layout.TraceLayout.show_cdll` is ``True`` (the
    default) each column includes a CDLL ring sub-panel between the DAG and
    strip panels, drawn via :func:`isalsr.viz.cdll_view.draw_cdll`.  Pointer
    markers on DAG nodes are suppressed in that case; the CDLL panel is the
    single source of pointer-position information.  When ``False`` the CDLL
    row is omitted and pointer markers appear on the DAG nodes instead,
    preserving the approved two-sub-row layout.

    Parameters
    ----------
    trace_r0:
        Snapshots from the violated-DAG run (partial; algorithm stalls after
        these).
    input_dag_r0:
        The violated input DAG (used to derive node-type map for positions).
    trace_r1:
        Snapshots from the normalised-DAG run (complete; 4 entries).
    input_dag_r1:
        The normalised input DAG.
    d2s_string:
        The complete D2S string produced by the normalised run; used to show
        future tokens in strip panels.
    input_layout:
        Canvas positions keyed by input-DAG node ID (same for both DAGs since
        they share the same node IDs and types).
    ax_xlim:
        DAG panel x limits in data units.
    ax_ylim:
        DAG panel y limits in data units.
    layout_obj:
        :class:`TraceLayout` controlling fonts and geometry.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure (caller must save or display it).
    """
    import matplotlib.pyplot as plt

    from isalsr.core.string_to_dag import _tokenize

    lo = layout_obj or TraceLayout()
    n_cols = lo.n_cols  # 4

    # Fixed slot count shared by every strip panel so all cells are the same size.
    # One extra slot for the stall cell in the violated row.
    n_slots = len(list(_tokenize(d2s_string))) + 1

    # Gridspec row indices: depends on whether CDLL rows are present.
    n_gs_rows = 2 * lo.n_sub_rows + 1  # 7 (with CDLL) or 5 (without)
    if lo.show_cdll:
        r0_dag, r0_cdll, r0_strip = 0, 1, 2
        r_sep = 3
        r1_dag, r1_cdll, r1_strip = 4, 5, 6
    else:
        r0_dag, r0_cdll, r0_strip = 0, -1, 1
        r_sep = 2
        r1_dag, r1_cdll, r1_strip = 3, -1, 4

    # Build input-node-type maps from the input DAGs.
    inp_types_r0: dict[int, NodeType] = {
        nd: input_dag_r0.node_label(nd) for nd in range(input_dag_r0.node_count)
    }
    inp_types_r1: dict[int, NodeType] = {
        nd: input_dag_r1.node_label(nd) for nd in range(input_dag_r1.node_count)
    }

    # Ghost node for the stall column: CONST c at its input_layout position.
    # CONST node in input_dag_r0 has node ID 1.
    _ghost_const_pos = input_layout.get(1, (1.2, 0.0))
    ghost_nodes = [(NodeType.CONST, _ghost_const_pos)]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    fig = plt.figure(figsize=lo.figsize)
    gs = fig.add_gridspec(
        n_gs_rows,
        n_cols,
        height_ratios=lo.height_ratios,
        hspace=lo.hspace,
        wspace=lo.wspace,
    )

    # ---------- Column step labels ----------
    stall_col = len(trace_r0)  # first column without a valid snapshot
    step_labels = [f"step {i}" for i in range(n_cols)]
    step_labels[stall_col] = rf"step {stall_col}: stall"
    for col in range(n_cols):
        ax_ref = fig.add_subplot(gs[r0_dag, col])
        ax_ref.set_visible(False)
        fig.text(
            ax_ref.get_position().x0 + ax_ref.get_position().width / 2,
            ax_ref.get_position().y1 + 0.01,
            step_labels[col],
            ha="center",
            va="bottom",
            fontsize=lo.fs_title,
            transform=fig.transFigure,
        )

    # ---------- Row 0 — violated DAG ----------
    dag_axes_r0: list[Axes] = []
    strip_axes_r0: list[Axes] = []
    cdll_axes_r0: list[Axes] = []

    for col in range(n_cols):
        ax_dag = fig.add_subplot(gs[r0_dag, col])
        ax_strip = fig.add_subplot(gs[r0_strip, col])
        dag_axes_r0.append(ax_dag)
        strip_axes_r0.append(ax_strip)
        if lo.show_cdll:
            ax_cdll = fig.add_subplot(gs[r0_cdll, col])
            cdll_axes_r0.append(ax_cdll)

        if col < stall_col:
            # Normal snapshot.
            snap_dag, snap_cdll, snap_pp, snap_sp, emitted = trace_r0[col]
            draw_dag_panel(
                ax_dag,
                trace_r0[col],
                input_layout=input_layout,
                input_node_types=inp_types_r0,
                ax_xlim=ax_xlim,
                ax_ylim=ax_ylim,
                show_ptr_markers=not lo.show_cdll,
                layout_obj=lo,
            )
            if lo.show_cdll:
                draw_cdll(
                    cdll_axes_r0[col],
                    snap_cdll,
                    snap_dag,
                    snap_pp,
                    snap_sp,
                    node_r=lo.cdll_node_r,
                    fs_label=lo.fs_cdll_label,
                    fs_ptr=lo.fs_cdll_ptr,
                )
            draw_strip_panel(ax_strip, emitted, n_slots=n_slots, layout_obj=lo)

        elif col == stall_col:
            # Stall column: re-use the last valid snapshot, add ghost c, red tint.
            last_dag, last_cdll, last_pp, last_sp, emitted = trace_r0[stall_col - 1]
            draw_dag_panel(
                ax_dag,
                trace_r0[stall_col - 1],
                input_layout=input_layout,
                input_node_types=inp_types_r0,
                ax_xlim=ax_xlim,
                ax_ylim=ax_ylim,
                ghost_nodes=ghost_nodes,
                stall=True,
                show_ptr_markers=not lo.show_cdll,
                layout_obj=lo,
            )
            if lo.show_cdll:
                draw_cdll(
                    cdll_axes_r0[col],
                    last_cdll,
                    last_dag,
                    last_pp,
                    last_sp,
                    node_r=lo.cdll_node_r,
                    fs_label=lo.fs_cdll_label,
                    fs_ptr=lo.fs_cdll_ptr,
                )
            draw_strip_panel(ax_strip, emitted, stall=True, n_slots=n_slots, layout_obj=lo)

        else:
            # Pad column.
            draw_pad_panel_dag(ax_dag, ax_xlim=ax_xlim, ax_ylim=ax_ylim)
            if lo.show_cdll:
                draw_pad_panel_cdll(cdll_axes_r0[col])
            draw_pad_panel_strip(ax_strip)

    # ---------- Row 1 — normalised DAG ----------
    dag_axes_r1: list[Axes] = []
    strip_axes_r1: list[Axes] = []
    cdll_axes_r1: list[Axes] = []

    for col in range(n_cols):
        ax_dag = fig.add_subplot(gs[r1_dag, col])
        ax_strip = fig.add_subplot(gs[r1_strip, col])
        dag_axes_r1.append(ax_dag)
        strip_axes_r1.append(ax_strip)
        if lo.show_cdll:
            ax_cdll = fig.add_subplot(gs[r1_cdll, col])
            cdll_axes_r1.append(ax_cdll)

        snap_dag, snap_cdll, snap_pp, snap_sp, emitted = trace_r1[col]
        draw_dag_panel(
            ax_dag,
            trace_r1[col],
            input_layout=input_layout,
            input_node_types=inp_types_r1,
            ax_xlim=ax_xlim,
            ax_ylim=ax_ylim,
            show_ptr_markers=not lo.show_cdll,
            layout_obj=lo,
        )
        if lo.show_cdll:
            draw_cdll(
                cdll_axes_r1[col],
                snap_cdll,
                snap_dag,
                snap_pp,
                snap_sp,
                node_r=lo.cdll_node_r,
                fs_label=lo.fs_cdll_label,
                fs_ptr=lo.fs_cdll_ptr,
            )
        draw_strip_panel(ax_strip, emitted, final_string=d2s_string, n_slots=n_slots, layout_obj=lo)

    # ---------- Separator row ----------
    sep_ax = fig.add_subplot(gs[r_sep, :])
    sep_ax.axis("off")

    # ---------- Row labels on left margin ----------
    # Centred on the full span from DAG top to strip bottom in each row.
    # fig.canvas.draw() finalises axes positions before get_position() is reliable.
    fig.canvas.draw()
    draw_row_label(
        fig,
        dag_axes_r0[0],
        strip_axes_r0[0],
        "violated",
        fs=lo.fs_title,
    )
    draw_row_label(
        fig,
        dag_axes_r1[0],
        strip_axes_r1[0],
        "normalised",
        fs=lo.fs_title,
    )

    # ---------- Stall frame (spans DAG top to strip bottom of stall column) ----------
    # add_stall_frame uses fig.transFigure, so fig.canvas.draw() must have run first.
    add_stall_frame(fig, dag_axes_r0[stall_col], strip_axes_r0[stall_col])

    # Suppress the unused sentinel value from the else-branch assignments.
    _ = r0_cdll, r1_cdll

    return fig
