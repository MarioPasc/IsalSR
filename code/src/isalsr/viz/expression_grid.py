"""Grid of expression cells: DAG, instruction string and rendered expression.

Where :mod:`isalsr.viz.algorithm_trace` shows one run unfolding over time, this
module shows a *set* of expressions side by side -- the steps of an edit path,
or the neighbours of a string under one edit. Each cell stacks the three views
that identify an expression in this work:

1. the **labeled DAG**, laid out by Graphviz ``dot`` with node-avoiding splines
   and pinned to one shared scale across the whole figure;
2. the **instruction string**, as a token strip;
3. the **expression** itself, as rendered mathematics.

Cells are grouped into rows, and the rows are banded and labelled. Two banding
schemes are supported, because the two figures this serves group differently:
an edit path wants a band per *view* (all DAGs together, all strings together),
while a neighbourhood wants a band per *row* (all substitutions together).

The figure is authored at its final printed size, so font sizes are true points.

Dependency rule: imports :mod:`isalsr.core` and :mod:`isalsr.viz` internals.
matplotlib is imported inside function bodies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.viz.backends.matplotlib_dag import MatplotlibDagBackend
from isalsr.viz.bands import BAND_ALPHA, BAND_COLOR, draw_axes_bands
from isalsr.viz.base import Position
from isalsr.viz.instruction_view import draw_instruction_strip, tokenize_string
from isalsr.viz.label_fit import fit_node_labels

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:
    Axes = Any
    Figure = Any

#: Band per view: one for the DAGs, one for the strings, one for the expressions.
BY_VIEW: str = "by_view"

#: Band per grid row: each row of cells gets its own band.
BY_ROW: str = "by_row"

#: Colour of the title above a cell flagged as an endpoint.
ENDPOINT_TITLE_COLOR: str = "#1b5e8f"

#: Colour of an ordinary cell title.
TITLE_COLOR: str = "#222222"


@dataclass(frozen=True)
class ExpressionCell:
    """One expression, shown as DAG + instruction string + rendered expression.

    Attributes:
        dag: The labeled DAG, or ``None`` when the string does not decode.
        instruction_string: The instruction string to draw as a token strip.
        title: Column title above the cell.
        math_latex: LaTeX body (no ``$``) for the expression row; empty to omit.
        emphasise_title: Draw the title in the endpoint colour.
        note: Text shown in place of the DAG when ``dag`` is ``None``.
        alert_nodes: Node IDs to ring in the alert colour, marking a defect
            such as an in-degree exceeding the operator's arity.
    """

    dag: LabeledDAG | None
    instruction_string: str
    title: str = ""
    math_latex: str = ""
    emphasise_title: bool = False
    note: str = "no valid DAG"
    alert_nodes: frozenset[int] = frozenset()


@dataclass(frozen=True)
class ExpressionRow:
    """One labelled row of the grid.

    Attributes:
        label: Rotated band label; empty leaves the band unlabelled.
        cells: The row's cells, left to right.
    """

    label: str
    cells: list[ExpressionCell]


@dataclass(frozen=True)
class ExpressionGridLayout:
    """Geometry and typography for one expression grid, in final print units.

    Attributes:
        fig_width: Total width; use the width the figure occupies in the
            manuscript so font sizes need no scaling factor.
        dag_height: Height of a cell's DAG panel.
        strip_height: Height of a cell's token-strip panel.
        math_height: Height of a cell's expression panel.
        row_gap: Vertical gap between grid rows.
        title_height: Room reserved above each row for the cell titles.
        legend_height: Room reserved at the bottom for the legend.
        margin_left: Left margin, wide enough for the rotated band labels.
        margin_right: Right margin.
        margin_top: Top margin.
        margin_bottom: Bottom margin.
        wspace: Horizontal spacing between cells, as a fraction of cell width.
        hspace: Vertical spacing inside a cell, as a fraction of panel height.
        rankdir: Graphviz rank direction for the DAG panels.
        ranksep: Graphviz rank separation, in inches.
        nodesep: Graphviz within-rank separation, in inches.
        node_r: DAG node-disc radius in data units.
        fs_title: Cell-title font size.
        fs_node: Ceiling for the DAG node-glyph size; the fitted size may be
            smaller, never larger.
        fs_strip: Token-cell font size.
        fs_math: Expression font size.
        fs_band_label: Rotated band-label font size.
        fs_legend: Legend font size.
        view_labels: Band labels used when banding by view.
        band_color: Band fill colour.
        band_alpha: Band opacity.
    """

    fig_width: float = 7.16
    dag_height: float = 1.30
    strip_height: float = 0.24
    math_height: float = 0.30
    row_gap: float = 0.16
    title_height: float = 0.17
    legend_height: float = 0.22
    margin_left: float = 0.042
    margin_right: float = 0.010
    margin_top: float = 0.02
    margin_bottom: float = 0.02
    wspace: float = 0.12
    hspace: float = 0.18
    rankdir: str = "LR"
    # Wider than the trace figure's 0.75/0.50. Graphviz reserves 2*node_r per
    # node, so ranksep is what decides whether an edge is longer than its own
    # arrowhead: at 1.25 in against a 0.95 in node the gap runs about 1.3 disc
    # diameters, which reads as an arrow rather than as two touching discs.
    # Affording it costs panel width, which is why the grid wraps onto more
    # rows rather than widening -- the figure's width is fixed by the text.
    ranksep: float = 1.25
    nodesep: float = 0.75
    node_r: float = 0.90
    fs_title: float = 8.0
    fs_node: float = 11.0
    fs_strip: float = 6.0
    fs_math: float = 7.4
    fs_band_label: float = 7.2
    fs_legend: float = 6.6
    band_color: str = BAND_COLOR
    band_alpha: float = BAND_ALPHA
    view_labels: tuple[str, str, str] = field(
        default=("DAG", r"$\Sigma^{*}_{\mathrm{SR}}$", "Expr")
    )

    @property
    def cell_height(self) -> float:
        """Height of one cell, excluding its title."""
        return self.dag_height + self.strip_height + self.math_height

    def fig_height(self, n_rows: int) -> float:
        """Total figure height for ``n_rows`` grid rows."""
        return (
            self.margin_top
            + n_rows * (self.title_height + self.cell_height)
            + (n_rows - 1) * self.row_gap
            + self.legend_height
            + self.margin_bottom
        )

    def figsize(self, n_rows: int) -> tuple[float, float]:
        """Matplotlib figure size in inches for ``n_rows`` grid rows."""
        return (self.fig_width, self.fig_height(n_rows))


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _shared_dag_scale(
    dags: list[LabeledDAG | None],
    backend: MatplotlibDagBackend,
) -> tuple[dict[int, dict[int, Position]], float, float]:
    """Lay every DAG out and return the layouts plus one common half-extent.

    Each DAG is laid out independently, but all panels are then given the same
    axis limits.  Letting every panel rescale to its own drawing would make a
    two-node DAG's discs several times larger than a six-node DAG's, so cell
    size would read as expression size -- a difference the figure does not mean
    to assert.

    Parameters
    ----------
    dags:
        The DAGs to lay out, in cell order.
    backend:
        The backend used to lay each DAG out.

    Returns
    -------
    tuple[dict[int, dict[int, Position]], float, float]
        Layout per cell index, and the common half-width and half-height.
    """
    layouts: dict[int, dict[int, Position]] = {}
    half_w = half_h = 0.0
    for idx, dag in enumerate(dags):
        if dag is None:
            continue
        pos = backend.compute_layout(dag)
        layouts[idx] = pos
        if not pos:
            continue
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        half_w = max(half_w, (max(xs) - min(xs)) / 2.0)
        half_h = max(half_h, (max(ys) - min(ys)) / 2.0)
    return layouts, half_w, half_h


def _fit_common_limits(
    half_w: float,
    half_h: float,
    *,
    node_r: float,
    pad: float,
    ax_w_in: float,
    ax_h_in: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return centred limits showing a ``half_w`` x ``half_h`` drawing as large as fits."""
    w = 2.0 * (half_w + node_r + pad)
    h = 2.0 * (half_h + node_r + pad)
    ax_aspect = ax_w_in / ax_h_in
    if w / h > ax_aspect:
        h = w / ax_aspect
    else:
        w = h * ax_aspect
    return (-w / 2.0, w / 2.0), (-h / 2.0, h / 2.0)


def _centred(layout: dict[int, Position]) -> dict[int, Position]:
    """Return ``layout`` translated so its bounding box is centred on the origin."""
    if not layout:
        return {}
    xs = [p[0] for p in layout.values()]
    ys = [p[1] for p in layout.values()]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    return {nd: (x - cx, y - cy) for nd, (x, y) in layout.items()}


def _draw_legend(fig: Figure, lay: ExpressionGridLayout, *, show_alert: bool) -> None:
    """Draw the shared legend strip along the bottom of ``fig``.

    The grid uses three colour channels a reader cannot infer -- the node-type
    palette, the green creation edge into a CONST node, and the alert ring --
    so it names them rather than leaving them to the caption.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    from isalsr.core.node_types import NodeType
    from isalsr.viz.style import (
        ALERT_COLOR,
        CREATION_EDGE_COLOR,
        color_for_node_border,
        color_for_node_face,
    )

    def disc(nt: NodeType) -> Circle:
        return Circle(
            (0, 0),
            1,
            facecolor=color_for_node_face(nt),
            edgecolor=color_for_node_border(nt),
            linewidth=1.0,
        )

    handles: list[Any] = [disc(NodeType.VAR), disc(NodeType.ADD), disc(NodeType.CONST)]
    labels = ["Variable node", "Operator node", "Constant node"]

    handles.append(Line2D([0], [0], color=CREATION_EDGE_COLOR, linewidth=1.6))
    labels.append("Creation edge")

    if show_alert:
        handles.append(
            Circle(
                (0, 0),
                1,
                facecolor="none",
                edgecolor=ALERT_COLOR,
                linewidth=1.5,
                linestyle=(0, (3, 2.5)),
            )
        )
        labels.append("In-degree exceeds arity")

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


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------


def make_expression_grid_figure(
    rows: list[ExpressionRow],
    *,
    band_mode: str = BY_VIEW,
    layout: ExpressionGridLayout | None = None,
) -> Figure:
    """Assemble a grid of expression cells.

    Parameters
    ----------
    rows:
        The grid's rows. Rows may hold different numbers of cells; shorter rows
        are left-aligned and the remaining slots stay empty.
    band_mode:
        ``BY_VIEW`` bands the DAG, string and expression panels separately,
        which suits a single row of steps. ``BY_ROW`` bands each grid row as a
        unit, which suits rows that are themselves the grouping.
    layout:
        Geometry and typography; defaults to :class:`ExpressionGridLayout`.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure, with a fully transparent background.

    Raises
    ------
    ValueError
        If ``rows`` is empty or ``band_mode`` is unknown.
    """
    import matplotlib.pyplot as plt

    from isalsr.viz.backends.graphviz_dag import GraphvizDagBackend

    if not rows:
        raise ValueError("rows must not be empty")
    if band_mode not in (BY_VIEW, BY_ROW):
        raise ValueError(f"band_mode must be {BY_VIEW!r} or {BY_ROW!r}, got {band_mode!r}")

    lay = layout or ExpressionGridLayout()
    n_rows = len(rows)
    n_cols = max(len(r.cells) for r in rows)

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

    flat = [c for r in rows for c in r.cells]
    layouts, half_w, half_h = _shared_dag_scale([c.dag for c in flat], backend)

    fig_w, fig_h = lay.figsize(n_rows)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_alpha(0.0)

    # One token-slot count for the whole figure, so cells share a cell size.
    n_slots = max((len(tokenize_string(c.instruction_string)) for c in flat), default=1)

    dag_axes: list[Axes] = []
    strip_axes: list[Axes] = []
    math_axes: list[Axes] = []
    row_axes: list[list[Axes]] = []
    per_row_views: list[tuple[list[Axes], list[Axes], list[Axes]]] = []

    unit = lay.title_height + lay.cell_height
    flat_idx = 0
    for r_i, row in enumerate(rows):
        top_in = lay.margin_top + r_i * (unit + lay.row_gap) + lay.title_height
        gs = fig.add_gridspec(
            3,
            n_cols,
            height_ratios=[lay.dag_height, lay.strip_height, lay.math_height],
            hspace=lay.hspace,
            wspace=lay.wspace,
            left=lay.margin_left,
            right=1.0 - lay.margin_right,
            top=1.0 - top_in / fig_h,
            bottom=1.0 - (top_in + lay.cell_height) / fig_h,
        )
        this_row: list[Axes] = []
        row_dags: list[Axes] = []
        row_strips: list[Axes] = []
        row_maths: list[Axes] = []
        for c_i in range(n_cols):
            ax_dag = fig.add_subplot(gs[0, c_i])
            ax_strip = fig.add_subplot(gs[1, c_i])
            ax_math = fig.add_subplot(gs[2, c_i])
            for ax in (ax_dag, ax_strip, ax_math):
                ax.patch.set_alpha(0.0)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            dag_axes.append(ax_dag)
            strip_axes.append(ax_strip)
            math_axes.append(ax_math)
            row_dags.append(ax_dag)
            row_strips.append(ax_strip)
            row_maths.append(ax_math)
            this_row.extend([ax_dag, ax_strip, ax_math])

            if c_i >= len(row.cells):
                ax_dag.axis("off")
                ax_math.axis("off")
                flat_idx += 0
                continue

            cell = row.cells[c_i]
            _draw_cell(
                fig,
                ax_dag,
                ax_strip,
                ax_math,
                cell,
                backend=backend,
                pinned=_centred(layouts.get(flat_idx, {})),
                n_slots=n_slots,
                lay=lay,
            )
            flat_idx += 1
        row_axes.append(this_row)
        per_row_views.append((row_dags, row_strips, row_maths))

    # Limits must match the panel shape, and the panel's measured size is only
    # known after a draw: the gridspec takes the inter-panel gaps out of the
    # rows, so the nominal height overstates it.
    fig.canvas.draw()
    box = dag_axes[0].get_position(original=True)
    xlim, ylim = _fit_common_limits(
        half_w,
        half_h,
        node_r=lay.node_r,
        pad=0.12,
        ax_w_in=box.width * fig_w,
        ax_h_in=box.height * fig_h,
    )
    for ax in dag_axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    fig.canvas.draw()
    fit_node_labels(dag_axes, lay.node_r, max_fontsize=lay.fs_node)

    _draw_legend(fig, lay, show_alert=any(c.alert_nodes for c in flat))

    fig.canvas.draw()
    if band_mode == BY_VIEW:
        # One band per (row, view). Banding a view across every row would draw
        # a single box spanning the gaps between rows, which is not a grouping
        # the reader can act on once the grid wraps onto more than one row.
        groups: list[list[Axes]] = []
        view_labels: list[str] = []
        for row_dags, row_strips, row_maths in per_row_views:
            groups.extend([row_dags, row_strips, row_maths])
            view_labels.extend(lay.view_labels)
        draw_axes_bands(
            fig,
            groups,
            view_labels,
            color=lay.band_color,
            alpha=lay.band_alpha,
            fontsize=lay.fs_band_label,
        )
    else:
        draw_axes_bands(
            fig,
            row_axes,
            [r.label for r in rows],
            color=lay.band_color,
            alpha=lay.band_alpha,
            fontsize=lay.fs_band_label,
        )
    return fig


def _draw_cell(
    fig: Figure,
    ax_dag: Axes,
    ax_strip: Axes,
    ax_math: Axes,
    cell: ExpressionCell,
    *,
    backend: MatplotlibDagBackend,
    pinned: dict[int, Position],
    n_slots: int,
    lay: ExpressionGridLayout,
) -> None:
    """Render one cell's three panels."""
    if cell.title:
        ax_dag.set_title(
            cell.title,
            fontsize=lay.fs_title,
            pad=3.0,
            color=ENDPOINT_TITLE_COLOR if cell.emphasise_title else TITLE_COLOR,
        )

    if cell.dag is None:
        ax_dag.axis("off")
        ax_dag.text(
            0.5,
            0.5,
            cell.note,
            ha="center",
            va="center",
            transform=ax_dag.transAxes,
            fontsize=lay.fs_math,
            color="#999999",
            style="italic",
        )
    else:
        backend.draw(cell.dag, ax_dag, layout=pinned, alert_nodes=cell.alert_nodes)

    draw_instruction_strip(
        ax_strip,
        cell.instruction_string,
        n_slots=n_slots,
        label_rotation=0.0,
        label_fontsize=lay.fs_strip,
    )

    ax_math.axis("off")
    if cell.math_latex:
        ax_math.text(
            0.5,
            0.5,
            f"${cell.math_latex}$",
            ha="center",
            va="center",
            transform=ax_math.transAxes,
            fontsize=lay.fs_math,
        )
