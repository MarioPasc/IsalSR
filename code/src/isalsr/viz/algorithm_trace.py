"""Three-row execution-trace figure for the S2D and D2S algorithms.

Each column is one displayed step of a run.  Within a column the three
sub-panels show, top to bottom, the three objects the algorithms manipulate:

1. the **labeled DAG**, laid out by Graphviz ``dot`` and pinned to the same
   node positions in every column, so a node never moves once it exists;
2. the **CDLL**, as a horizontal chain carrying the two pointer markers;
3. the **instruction string**, as a token strip whose emitted prefix is solid
   and whose remainder is faded.

All three read left to right and share one cell size across columns, which is
what lets the reader compare a step against its neighbours instead of
re-reading each panel's scale.

The figure is authored at its *final* printed size: pass the width the figure
occupies in the manuscript (``\\textwidth`` for a two-column ``figure*``) and
font sizes are then true points, with no scaling factor to reason about.

Dependency rule: imports :mod:`isalsr.core` and :mod:`isalsr.viz` internals.
matplotlib is imported inside function bodies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from isalsr.core.cdll import CircularDoublyLinkedList
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.viz.bands import draw_axes_bands
from isalsr.viz.base import Position
from isalsr.viz.cdll_view import draw_cdll_ring
from isalsr.viz.instruction_view import draw_instruction_strip, tokenize_string
from isalsr.viz.label_fit import fit_node_labels
from isalsr.viz.style import (
    GHOST_EDGE_COLOR,
    NEW_NODE_ACCENT,
    PRIMARY_PTR_COLOR,
    SECONDARY_PTR_COLOR,
    color_for_node_border,
    color_for_node_face,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any

#: One trace snapshot: ``(dag, cdll, primary_ptr, secondary_ptr, emitted)``.
Snapshot = tuple[LabeledDAG, CircularDoublyLinkedList, int, int, str]

#: Information flows from the instruction string into the DAG (S2D).
STRING_TO_DAG: str = "string_to_dag"

#: Information flows from the DAG into the instruction string (D2S).
DAG_TO_STRING: str = "dag_to_string"


class TraceConsistencyError(ValueError):
    """Raised when a snapshot cannot be reconciled with the target DAG.

    Node positions are pinned from the target DAG and reused for every
    snapshot, which is only sound if snapshot node IDs denote the same nodes
    as target node IDs.  Rather than let a mismatch silently draw a node in
    the wrong place, the assembler checks the labels and raises.
    """


@dataclass(frozen=True)
class AlgorithmTraceLayout:
    """Geometry and typography for one three-row trace figure.

    Every length is in inches at final printed size, and every font size is in
    true points at that size.

    Attributes:
        fig_width: Total figure width; use the width it occupies in the
            manuscript so fonts need no mental scaling factor.
        dag_height: Height of the DAG sub-row.
        cdll_height: Height of the CDLL sub-row.
        strip_height: Height of the instruction-strip sub-row.
        title_height: Vertical room reserved for the column titles.
        legend_height: Vertical room reserved for the legend.
        margin_left: Left margin.
        margin_right: Right margin.
        margin_top: Top margin.
        margin_bottom: Bottom margin.
        wspace: Gridspec horizontal spacing, as a fraction of column width.
        hspace: Gridspec vertical spacing, as a fraction of row height.
        rankdir: Graphviz rank direction for the DAG panels.
        node_r: DAG node-disc radius in data units.
        cdll_node_r: CDLL node-disc radius in data units.
        fs_title: Column-title font size.
        fs_node: DAG node-glyph font size.
        fs_cdll_label: CDLL node-glyph font size.
        fs_cdll_ptr: Pointer-marker font size.
        fs_strip: Token-cell font size.
        fs_legend: Legend font size.
        primary_label: Symbol for the primary pointer.
        secondary_label: Symbol for the secondary pointer.
    """

    fig_width: float = 7.16
    dag_height: float = 1.05
    cdll_height: float = 1.05
    strip_height: float = 0.26
    title_height: float = 0.17
    legend_height: float = 0.20
    margin_left: float = 0.042
    margin_right: float = 0.010
    margin_top: float = 0.02
    margin_bottom: float = 0.02
    wspace: float = 0.10
    hspace: float = 0.18
    rankdir: str = "LR"
    ranksep: float = 0.75
    nodesep: float = 0.50
    node_r: float = 0.90
    cdll_node_r: float = 0.41
    cdll_radius: float = 1.15
    fs_title: float = 8.0
    fs_node: float = 11.0
    fs_cdll_label: float = 10.0
    fs_cdll_ptr: float = 7.0
    fs_strip: float = 6.0
    fs_legend: float = 6.6
    fs_row_label: float = 7.2
    primary_label: str = "$p$"
    secondary_label: str = "$q$"
    band_color: str = "#c8ccd2"
    band_alpha: float = 0.3
    consumed_alpha: float = 0.28
    row_labels: tuple[str, str, str] = ("DAG", "CDLL", r"$\Sigma_{\mathrm{SR}}^{*}$")

    @property
    def fig_height(self) -> float:
        """Total figure height in inches."""
        return (
            self.margin_top
            + self.title_height
            + self.dag_height
            + self.cdll_height
            + self.strip_height
            + self.legend_height
            + self.margin_bottom
        )

    @property
    def figsize(self) -> tuple[float, float]:
        """Matplotlib figure size in inches."""
        return (self.fig_width, self.fig_height)


# ---------------------------------------------------------------------------
# Step selection
# ---------------------------------------------------------------------------


def _steps_at_stride(n_total: int, stride: int) -> list[int]:
    """Return ``range(0, n_total, stride)`` forced to end on the final step."""
    idx = list(range(0, n_total, stride))
    span = n_total - 1
    if idx[-1] == span:
        return idx
    # Absorb the remainder into the last interval when it is smaller than half
    # a stride, otherwise give it a column of its own.
    if len(idx) > 1 and span - idx[-1] < stride / 2:
        idx[-1] = span
    else:
        idx.append(span)
    return idx


def evenly_spaced_steps(n_total: int, n_show: int) -> list[int]:
    """Return roughly ``n_show`` step indices spanning the whole trace.

    Even spacing matters for readability: a reader who sees columns labelled
    0, 2, 3, 4, 6 must first work out that the gaps are uneven before the
    columns can be compared, whereas 0, 2, 4, 6 reads as one rhythm.  The first
    and last steps are always shown, since they are the two the reader needs in
    order to see what the run started from and what it produced.

    Every stride is tried and scored on, in order, how close its column count
    lands to ``n_show`` and how uniform its gaps are.  Searching rather than
    solving keeps the rule well behaved when ``n_total - 1`` is prime, where the
    only strides that divide the span exactly are 1 and the span itself.

    The result holds one stride between every pair of consecutive steps except
    the last, which absorbs whatever remainder the stride leaves; that remainder
    is under half a stride, so no column is visibly out of rhythm.

    Parameters
    ----------
    n_total:
        Length of the trace.
    n_show:
        Desired number of displayed steps.

    Returns
    -------
    list[int]
        Strictly increasing step indices, starting at ``0`` and ending at
        ``n_total - 1``.

    Raises
    ------
    ValueError
        If ``n_total`` is not positive or ``n_show`` is below 2.
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    if n_show < 2:
        raise ValueError(f"n_show must be at least 2, got {n_show}")
    if n_total <= n_show:
        return list(range(n_total))

    def score(stride: int) -> tuple[int, int, int]:
        idx = _steps_at_stride(n_total, stride)
        gaps = [b - a for a, b in zip(idx[:-1], idx[1:], strict=True)]
        return (abs(len(idx) - n_show), max(gaps) - min(gaps), stride)

    best = min(range(1, n_total), key=score)
    return _steps_at_stride(n_total, best)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _fit_limits(
    layout: dict[int, Position],
    *,
    node_r: float,
    pad: float,
    ax_w_in: float,
    ax_h_in: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ``(xlim, ylim)`` that show ``layout`` as large as the axes allow.

    The DAG panels use ``set_aspect("equal")`` because arrow trimming is done
    in data units.  Equal aspect means the limits, not the data, decide how
    much of the panel the drawing fills, so the limits are widened on whichever
    axis is not the binding constraint.

    Parameters
    ----------
    layout:
        Node positions in data units.
    node_r:
        Node radius, added to the bounding box so discs are not clipped.
    pad:
        Extra breathing room in data units.
    ax_w_in:
        Axes width in inches.
    ax_h_in:
        Axes height in inches.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        The x and y limits.
    """
    if not layout:
        return (-1.0, 1.0), (-1.0, 1.0)
    xs = [p[0] for p in layout.values()]
    ys = [p[1] for p in layout.values()]
    margin = node_r + pad
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    ax_aspect = ax_w_in / ax_h_in
    if w / h > ax_aspect:
        h = w / ax_aspect
    else:
        w = h * ax_aspect
    return (cx - w / 2.0, cx + w / 2.0), (cy - h / 2.0, cy + h / 2.0)


def _validate_snapshot(snapshot_dag: LabeledDAG, target_dag: LabeledDAG, col: int) -> None:
    """Raise if ``snapshot_dag`` node IDs do not denote ``target_dag`` nodes."""
    if snapshot_dag.node_count > target_dag.node_count:
        raise TraceConsistencyError(
            f"column {col}: snapshot holds {snapshot_dag.node_count} nodes, "
            f"more than the target's {target_dag.node_count}"
        )
    for node in range(snapshot_dag.node_count):
        snap_label = snapshot_dag.node_label(node)
        tgt_label = target_dag.node_label(node)
        if snap_label is not tgt_label:
            raise TraceConsistencyError(
                f"column {col}: node {node} is {snap_label.name} in the snapshot "
                f"but {tgt_label.name} in the target; node IDs do not correspond, "
                "so pinned positions would place it wrongly"
            )
        if snap_label is NodeType.VAR:
            snap_idx = snapshot_dag.node_data(node).get("var_index")
            tgt_idx = target_dag.node_data(node).get("var_index")
            if snap_idx != tgt_idx:
                raise TraceConsistencyError(
                    f"column {col}: VAR node {node} has index {snap_idx} in the "
                    f"snapshot but {tgt_idx} in the target"
                )


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------


def _draw_legend(fig: Figure, lay: AlgorithmTraceLayout, *, building: bool) -> None:
    """Draw the shared legend strip along the bottom of ``fig``.

    The final entry names what the faint material means, and that differs by
    direction: when the DAG is being built the faint nodes are not there yet,
    and when it is being consumed they are there but already spent.  Spelling
    it out per figure keeps the two from being read as the same claim.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    handles: list[Any] = [
        Circle(
            (0, 0),
            1,
            facecolor=color_for_node_face(NodeType.VAR),
            edgecolor=color_for_node_border(NodeType.VAR),
            linewidth=1.0,
        ),
        Circle(
            (0, 0),
            1,
            facecolor=color_for_node_face(NodeType.ADD),
            edgecolor=color_for_node_border(NodeType.ADD),
            linewidth=1.0,
        ),
        Circle((0, 0), 1, facecolor="none", edgecolor=NEW_NODE_ACCENT, linewidth=1.8),
        Line2D([0], [0], marker="v", color=PRIMARY_PTR_COLOR, linestyle="none", markersize=4),
        Line2D([0], [0], marker="v", color=SECONDARY_PTR_COLOR, linestyle="none", markersize=4),
    ]
    labels = [
        "Variable node",
        "Operator node",
        "Added at this step",
        f"Primary pointer {lay.primary_label}",
        f"Secondary pointer {lay.secondary_label}",
    ]
    if building:
        handles.append(
            Circle(
                (0, 0),
                1,
                facecolor="none",
                edgecolor=GHOST_EDGE_COLOR,
                linewidth=1.0,
                linestyle=(0, (2.6, 2.0)),
            )
        )
        labels.append("Not yet built")
    else:
        handles.append(
            Circle(
                (0, 0),
                1,
                facecolor=color_for_node_face(NodeType.ADD),
                edgecolor=color_for_node_border(NodeType.ADD),
                linewidth=1.0,
                alpha=lay.consumed_alpha,
            )
        )
        labels.append("Already encoded")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=lay.fs_legend,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=1.4,
        borderpad=0.0,
        bbox_to_anchor=(0.5, 0.0),
    )


def _draw_row_bands(
    fig: Figure,
    rows: list[list[Any]],
    lay: AlgorithmTraceLayout,
) -> None:
    """Paint one rounded band behind each of the three sub-rows and label it.

    Thin adapter over :func:`isalsr.viz.bands.draw_axes_bands`, which both this
    figure and the expression grid share so the two read as one system.
    """
    draw_axes_bands(
        fig,
        rows,
        list(lay.row_labels),
        color=lay.band_color,
        alpha=lay.band_alpha,
        fontsize=lay.fs_row_label,
    )


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------


def make_algorithm_trace_figure(
    snapshots: list[Snapshot],
    target_dag: LabeledDAG,
    *,
    final_string: str,
    step_labels: list[str],
    direction: str,
    layout: AlgorithmTraceLayout | None = None,
) -> Figure:
    """Assemble the three-row execution-trace figure.

    Parameters
    ----------
    snapshots:
        The displayed subset of a run's trace, one entry per column, each a
        ``(dag, cdll, primary_ptr, secondary_ptr, emitted)`` tuple.
    target_dag:
        The DAG the run ends on.  Node positions are computed once from this
        DAG and pinned in every column, so a node never moves once it exists.
    final_string:
        The complete instruction string, used to draw not-yet-emitted tokens.
    step_labels:
        Column titles, one per snapshot.
    direction:
        Which way information flows, ``"string_to_dag"`` or ``"dag_to_string"``.

        Both algorithms move a fixed amount of information from one
        representation to the other, and the figure shows that conservation
        directly: at every step the **solid** material across the DAG and string
        rows is exactly the information still in play, so the source empties as
        the destination fills.

        Under ``"string_to_dag"`` the string is the source, so its consumed
        prefix fades and its unread remainder stays solid, while the DAG is the
        destination and the part not yet built is drawn as a dashed ghost.
        Under ``"dag_to_string"`` the roles swap: the DAG is the source and its
        already-encoded part fades, while the string is the destination and its
        unemitted remainder is the faint one.
    layout:
        Geometry and typography; defaults to :class:`AlgorithmTraceLayout`.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure, with a fully transparent background.

    Raises
    ------
    ValueError
        If ``step_labels`` and ``snapshots`` differ in length, or ``direction``
        is not one of the two accepted values.
    TraceConsistencyError
        If a snapshot's node IDs do not correspond to the target's.
    """
    import matplotlib.pyplot as plt

    from isalsr.viz.backends.graphviz_dag import GraphvizDagBackend

    lay = layout or AlgorithmTraceLayout()
    n_cols = len(snapshots)
    if len(step_labels) != n_cols:
        raise ValueError(f"got {len(step_labels)} step labels for {n_cols} snapshots")
    if direction not in (STRING_TO_DAG, DAG_TO_STRING):
        raise ValueError(
            f"direction must be {STRING_TO_DAG!r} or {DAG_TO_STRING!r}, got {direction!r}"
        )
    building = direction == STRING_TO_DAG
    for col, (snap_dag, _, _, _, _) in enumerate(snapshots):
        _validate_snapshot(snap_dag, target_dag, col)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    backend = GraphvizDagBackend(
        rankdir=lay.rankdir,
        node_r=lay.node_r,
        fs_node=lay.fs_node,
        align_variables=True,
        ranksep=lay.ranksep,
        nodesep=lay.nodesep,
    )
    pinned: dict[int, Position] = backend.compute_layout(target_dag)

    fig = plt.figure(figsize=lay.figsize)
    fig.patch.set_alpha(0.0)

    h = lay.fig_height
    w = lay.fig_width
    gs = fig.add_gridspec(
        3,
        n_cols,
        height_ratios=[lay.dag_height, lay.cdll_height, lay.strip_height],
        hspace=lay.hspace,
        wspace=lay.wspace,
        left=lay.margin_left,
        right=1.0 - lay.margin_right,
        top=1.0 - (lay.margin_top + lay.title_height) / h,
        bottom=(lay.margin_bottom + lay.legend_height) / h,
    )

    n_token_slots = len(tokenize_string(final_string))

    # Nodes are accented when they appeared since the *previously displayed*
    # step.  The first column has no predecessor, and its variable nodes were
    # pre-inserted before any instruction ran (Critical Invariant 7), so
    # nothing in it was "added at this step".
    prev_node_count = snapshots[0][0].node_count
    rows: list[list[Any]] = [[], [], []]
    for col, (snap_dag, snap_cdll, snap_p, snap_q, emitted) in enumerate(snapshots):
        accent = frozenset(range(prev_node_count, snap_dag.node_count))
        prev_node_count = snap_dag.node_count

        # ---- DAG ----
        # Building: draw what exists, with the rest ghosted in behind it.
        # Consuming: draw the whole input, fading the part already encoded.
        ax_dag = fig.add_subplot(gs[0, col])
        ax_dag.patch.set_alpha(0.0)
        # The variable nodes are pre-inserted before any instruction runs
        # (Critical Invariant 7) and no token ever encodes them, so they are
        # shared initial state rather than material the run consumes. Fading
        # them would claim work was done that no token accounts for, and would
        # also shade the same nodes differently in the two figures.
        done_nodes = frozenset(
            nd for nd in range(snap_dag.node_count) if snap_dag.node_label(nd) is not NodeType.VAR
        )
        done_edges = frozenset(
            (s, t) for s in range(snap_dag.node_count) for t in snap_dag.out_neighbors(s)
        )
        backend.draw(
            snap_dag if building else target_dag,
            ax_dag,
            layout=pinned,
            ghost_dag=target_dag if building else None,
            accent_nodes=accent,
            dim_nodes=frozenset() if building else done_nodes,
            dim_edges=frozenset() if building else done_edges,
            dim_alpha=lay.consumed_alpha,
        )
        ax_dag.set_title(step_labels[col], fontsize=lay.fs_title, pad=3.0)

        # ---- CDLL ----
        ax_cdll = fig.add_subplot(gs[1, col])
        ax_cdll.patch.set_alpha(0.0)
        draw_cdll_ring(
            ax_cdll,
            snap_cdll,
            snap_dag,
            snap_p,
            snap_q,
            node_r=lay.cdll_node_r,
            radius=lay.cdll_radius,
            fs_label=lay.fs_cdll_label,
            fs_ptr=lay.fs_cdll_ptr,
            accent_nodes=accent,
            primary_label=lay.primary_label,
            secondary_label=lay.secondary_label,
        )

        # ---- Instruction strip ----
        ax_strip = fig.add_subplot(gs[2, col])
        ax_strip.patch.set_alpha(0.0)
        draw_instruction_strip(
            ax_strip,
            final_string,
            emitted=emitted,
            solid_side="suffix" if building else "prefix",
            n_slots=n_token_slots,
            future_alpha=lay.consumed_alpha,
            label_rotation=0.0,
            label_fontsize=lay.fs_strip,
        )

        rows[0].append(ax_dag)
        rows[1].append(ax_cdll)
        rows[2].append(ax_strip)

    _draw_legend(fig, lay, building=building)
    # Axis positions must be final before anything is measured against them.
    fig.canvas.draw()

    # Fit the DAG limits to the panel's *measured* size. The nominal row height
    # overstates it, because the gridspec takes the inter-row gaps out of the
    # rows; fitting to the nominal value leaves the drawing smaller than the
    # panel allows.
    # ``original=True`` gives the gridspec cell.  The plain accessor returns the
    # box after equal-aspect shrinking, which is derived from whatever limits
    # were in force during the draw above -- fitting to that would chase its own
    # tail and leave the drawing far smaller than the cell allows.
    dag_box = rows[0][0].get_position(original=True)
    xlim, ylim = _fit_limits(
        pinned,
        node_r=lay.node_r,
        pad=0.12,
        ax_w_in=dag_box.width * w,
        ax_h_in=dag_box.height * h,
    )
    for ax in rows[0]:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # Discs are in data units and glyphs in points, so the requested font sizes
    # are only a ceiling; the drawn size is whatever fits the drawn disc.
    fig.canvas.draw()
    fit_node_labels(rows[0], lay.node_r, max_fontsize=lay.fs_node)
    fit_node_labels(rows[1], lay.cdll_node_r, max_fontsize=lay.fs_cdll_label)

    fig.canvas.draw()
    _draw_row_bands(fig, rows, lay)
    return fig
