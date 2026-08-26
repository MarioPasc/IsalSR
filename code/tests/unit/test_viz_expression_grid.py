"""Unit tests for the expression-grid figure and the shared band helper.

The tests that matter scientifically are the ones asserting the grid cannot
misrepresent what it draws:

- every cell shares one DAG scale, so panel size never reads as expression size;
- a cell can flag nodes whose in-degree exceeds their arity, which is the
  difference between "this intermediate is well-formed" and "the adapter
  silently dropped an edge the panel drew".
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from isalsr.core.string_to_dag import StringToDAG  # noqa: E402
from isalsr.viz.bands import draw_axes_bands  # noqa: E402
from isalsr.viz.expression_grid import (  # noqa: E402
    BY_ROW,
    BY_VIEW,
    ExpressionCell,
    ExpressionGridLayout,
    ExpressionRow,
    _centred,
    _fit_common_limits,
    make_expression_grid_figure,
)
from isalsr.viz.label_fit import NODE_LABEL_GID  # noqa: E402

BASE = "VcVspv+Ppc"
OTHER = "VcVspv*Ppc"


def _cell(string: str, **kw: object) -> ExpressionCell:
    dag = StringToDAG(string, num_variables=1).run()
    return ExpressionCell(dag=dag, instruction_string=string, **kw)  # type: ignore[arg-type]


@pytest.fixture
def rows() -> list[ExpressionRow]:
    return [
        ExpressionRow(label="A", cells=[_cell(BASE), _cell(OTHER)]),
        ExpressionRow(label="B", cells=[_cell(OTHER)]),
    ]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_fit_common_limits_matches_axes_aspect() -> None:
    xlim, ylim = _fit_common_limits(2.0, 1.0, node_r=0.5, pad=0.1, ax_w_in=2.0, ax_h_in=1.0)
    assert (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) == pytest.approx(2.0)


def test_fit_common_limits_is_centred() -> None:
    xlim, ylim = _fit_common_limits(2.0, 1.0, node_r=0.5, pad=0.1, ax_w_in=1.0, ax_h_in=1.0)
    assert sum(xlim) == pytest.approx(0.0)
    assert sum(ylim) == pytest.approx(0.0)


def test_centred_translates_bounding_box_to_origin() -> None:
    out = _centred({0: (1.0, 2.0), 1: (5.0, 8.0)})
    xs = [p[0] for p in out.values()]
    ys = [p[1] for p in out.values()]
    assert (min(xs) + max(xs)) == pytest.approx(0.0)
    assert (min(ys) + max(ys)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [BY_VIEW, BY_ROW])
def test_grid_has_three_panels_per_slot(rows: list[ExpressionRow], mode: str) -> None:
    """Rows are padded to the widest row, so the axes count is rows x cols x 3."""
    fig = make_expression_grid_figure(rows, band_mode=mode)
    assert len(fig.axes) == 2 * 2 * 3


def test_grid_background_is_transparent(rows: list[ExpressionRow]) -> None:
    fig = make_expression_grid_figure(rows)
    assert fig.patch.get_alpha() == 0.0
    for ax in fig.axes:
        assert ax.patch.get_alpha() == 0.0


def test_grid_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        make_expression_grid_figure([])


def test_grid_rejects_unknown_band_mode(rows: list[ExpressionRow]) -> None:
    with pytest.raises(ValueError, match="band_mode must be"):
        make_expression_grid_figure(rows, band_mode="diagonal")


def test_grid_size_matches_layout(rows: list[ExpressionRow]) -> None:
    lay = ExpressionGridLayout()
    fig = make_expression_grid_figure(rows, layout=lay)
    assert tuple(fig.get_size_inches()) == pytest.approx(lay.figsize(len(rows)))


def test_every_dag_panel_shares_one_scale(rows: list[ExpressionRow]) -> None:
    """Panels must share limits, or cell size would read as expression size."""
    fig = make_expression_grid_figure(rows)
    fig.canvas.draw()
    dag_axes = [ax for i, ax in enumerate(fig.axes) if i % 3 == 0]
    lims = {(ax.get_xlim(), ax.get_ylim()) for ax in dag_axes}
    assert len(lims) == 1


def test_node_glyphs_share_one_fitted_size(rows: list[ExpressionRow]) -> None:
    """One size across the grid, and never larger than the requested ceiling."""
    lay = ExpressionGridLayout()
    fig = make_expression_grid_figure(rows, layout=lay)
    fig.canvas.draw()
    sizes = {t.get_fontsize() for ax in fig.axes for t in ax.texts if t.get_gid() == NODE_LABEL_GID}
    assert len(sizes) == 1
    assert sizes.pop() <= lay.fs_node


def test_cell_without_dag_draws_a_note() -> None:
    fig = make_expression_grid_figure(
        [
            ExpressionRow(
                label="A",
                cells=[ExpressionCell(dag=None, instruction_string="Vc", note="undecodable")],
            )
        ]
    )
    texts = {t.get_text() for ax in fig.axes for t in ax.texts}
    assert "undecodable" in texts


def test_alert_nodes_add_a_ring(rows: list[ExpressionRow]) -> None:
    """A flagged node must be drawn differently from an unflagged one."""
    from matplotlib.patches import Circle

    def n_circles(alert: frozenset[int]) -> int:
        dag = StringToDAG(BASE, num_variables=1).run()
        fig = make_expression_grid_figure(
            [
                ExpressionRow(
                    label="A",
                    cells=[ExpressionCell(dag=dag, instruction_string=BASE, alert_nodes=alert)],
                )
            ]
        )
        fig.canvas.draw()
        ax = fig.axes[0]
        return sum(1 for p in ax.patches if isinstance(p, Circle))

    assert n_circles(frozenset({1})) == n_circles(frozenset()) + 1


def test_by_view_bands_each_row_separately(rows: list[ExpressionRow]) -> None:
    """A view band must not span the gap between two rows of the grid."""
    from matplotlib.patches import FancyBboxPatch

    fig = make_expression_grid_figure(rows, band_mode=BY_VIEW)
    bands = [a for a in fig.artists if isinstance(a, FancyBboxPatch)]
    # Three views for each of the two rows.
    assert len(bands) == 6


def test_by_row_bands_each_row_once(rows: list[ExpressionRow]) -> None:
    from matplotlib.patches import FancyBboxPatch

    fig = make_expression_grid_figure(rows, band_mode=BY_ROW)
    bands = [a for a in fig.artists if isinstance(a, FancyBboxPatch)]
    assert len(bands) == 2


# ---------------------------------------------------------------------------
# Shared band helper
# ---------------------------------------------------------------------------


def test_bands_share_one_horizontal_extent() -> None:
    """Rotated band labels only line up if every band starts at the same x."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, axs = plt.subplots(3, 2)
    axs[0][0].set_aspect("equal")  # would shrink its own box
    fig.canvas.draw()
    draw_axes_bands(fig, [list(axs[0]), list(axs[1]), list(axs[2])], ["a", "b", "c"])
    bands = [a for a in fig.artists if isinstance(a, FancyBboxPatch)]
    assert len(bands) == 3
    xs = {(round(b.get_x(), 9), round(b.get_width(), 9)) for b in bands}
    assert len(xs) == 1
    plt.close(fig)


def test_bands_reject_label_count_mismatch() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="band groups"):
        draw_axes_bands(fig, [[ax]], ["a", "b"])
    plt.close(fig)
