"""Unit tests for the S2D/D2S trace figure and its Graphviz layout backend.

The tests that matter scientifically are the ones asserting that the figure
cannot silently misrepresent the algorithms:

- pinned positions are only used when snapshot node IDs really do denote the
  same nodes as the target's (:class:`TraceConsistencyError`);
- the Graphviz layout removes the edge crossings the built-in layered layout
  leaves behind, since a crossing lets a reader attach an operand to the wrong
  operator.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from isalsr.core.labeled_dag import LabeledDAG  # noqa: E402
from isalsr.core.node_types import NodeType  # noqa: E402
from isalsr.core.string_to_dag import StringToDAG  # noqa: E402
from isalsr.viz.algorithm_trace import (  # noqa: E402
    DAG_TO_STRING,
    STRING_TO_DAG,
    AlgorithmTraceLayout,
    Snapshot,
    TraceConsistencyError,
    _fit_limits,
    evenly_spaced_steps,
    make_algorithm_trace_figure,
)
from isalsr.viz.backends.graphviz_dag import (  # noqa: E402
    graphviz_layout,
    graphviz_layout_and_routes,
)
from isalsr.viz.backends.matplotlib_dag import _layered_layout, _spline_path  # noqa: E402
from isalsr.viz.cdll_view import draw_cdll, draw_cdll_ring  # noqa: E402, F401
from isalsr.viz.instruction_view import (  # noqa: E402
    draw_instruction_strip,
    tokenize_string,
)
from isalsr.viz.label_fit import NODE_LABEL_GID  # noqa: E402

#: Canonical string of the manuscript's running example, cos(x1) + sin(x1)*x2.
EXAMPLE = "VcVspv*pv+PpcnnC"


@pytest.fixture
def example_dag() -> LabeledDAG:
    """The running example's DAG."""
    return StringToDAG(EXAMPLE, num_variables=2).run()


@pytest.fixture
def example_trace() -> list[Snapshot]:
    """The full S2D trace of the running example."""
    s2d = StringToDAG(EXAMPLE, num_variables=2)
    s2d.run(trace=True)
    return [(d, c, p, q, "".join(str(t) for t in toks)) for d, c, p, q, toks in s2d._trace_log]


# ---------------------------------------------------------------------------
# Step selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_total", "n_show", "expected"),
    [
        (13, 5, [0, 3, 6, 9, 12]),
        (7, 5, [0, 2, 4, 6]),
        (5, 5, [0, 1, 2, 3, 4]),
        (3, 5, [0, 1, 2]),
        (11, 4, [0, 3, 6, 10]),
    ],
)
def test_evenly_spaced_steps_values(n_total: int, n_show: int, expected: list[int]) -> None:
    assert evenly_spaced_steps(n_total, n_show) == expected


@pytest.mark.parametrize("n_total", range(2, 60))
def test_evenly_spaced_steps_spans_and_is_near_uniform(n_total: int) -> None:
    """Selections start at 0, end at the last step, and hold one stride throughout.

    A span that no convenient stride divides cannot be covered by an exactly
    constant stride, so the contract is weaker than "all gaps equal": every gap
    but the last is the stride, and the last absorbs the remainder, which is
    under half a stride in either direction.
    """
    idx = evenly_spaced_steps(n_total, 5)
    assert idx[0] == 0
    assert idx[-1] == n_total - 1
    assert idx == sorted(set(idx))
    gaps = [b - a for a, b in zip(idx[:-1], idx[1:], strict=True)]
    interior = set(gaps[:-1])
    assert len(interior) <= 1, f"interior gaps {gaps} not constant for n_total={n_total}"
    stride = interior.pop() if interior else gaps[-1]
    assert 1 <= gaps[-1] < 1.5 * stride, f"final gap {gaps[-1]} vs stride {stride}"


@pytest.mark.parametrize("n_total", range(6, 60))
def test_evenly_spaced_steps_column_count_is_close_to_request(n_total: int) -> None:
    """The prime-span fallback must not collapse a long trace to two columns."""
    idx = evenly_spaced_steps(n_total, 5)
    assert abs(len(idx) - 5) <= 1, f"got {len(idx)} columns for n_total={n_total}"


@pytest.mark.parametrize(("n_total", "n_show"), [(0, 5), (-1, 5), (10, 1)])
def test_evenly_spaced_steps_rejects_degenerate_input(n_total: int, n_show: int) -> None:
    with pytest.raises(ValueError):
        evenly_spaced_steps(n_total, n_show)


# ---------------------------------------------------------------------------
# Graphviz layout
# ---------------------------------------------------------------------------


def _segments_cross(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """Return whether open segments p1p2 and p3p4 properly intersect."""

    def orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _count_crossings(dag: LabeledDAG, layout: dict[int, tuple[float, float]]) -> int:
    """Count pairs of edges whose drawn straight segments properly cross."""
    edges = [(s, t) for s in range(dag.node_count) for t in dag.out_neighbors(s)]
    total = 0
    for i, (a, b) in enumerate(edges):
        for c, d in edges[i + 1 :]:
            if len({a, b, c, d}) < 4:  # shared endpoint: not a crossing
                continue
            if _segments_cross(layout[a], layout[b], layout[c], layout[d]):
                total += 1
    return total


def test_graphviz_layout_covers_every_node(example_dag: LabeledDAG) -> None:
    pos = graphviz_layout(example_dag)
    assert set(pos) == set(range(example_dag.node_count))


def test_graphviz_layout_is_centred(example_dag: LabeledDAG) -> None:
    """Centring lets panels share one pair of axis limits."""
    pos = graphviz_layout(example_dag)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    assert (min(xs) + max(xs)) == pytest.approx(0.0, abs=1e-9)
    assert (min(ys) + max(ys)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(("rankdir", "wider"), [("LR", True), ("BT", False)])
def test_graphviz_layout_rankdir_controls_aspect(
    example_dag: LabeledDAG, rankdir: str, wider: bool
) -> None:
    """``LR`` must produce the wide-and-short drawing the figure relies on."""
    pos = graphviz_layout(example_dag, rankdir=rankdir)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    assert (w > h) is wider


def test_graphviz_layout_removes_crossings(example_dag: LabeledDAG) -> None:
    """dot's crossing reduction beats the built-in node-ID ordering.

    A crossing is not cosmetic here: it lets a reader trace an arrow into the
    wrong operator, so the figure would assert a different expression than the
    DAG encodes.
    """
    naive = _count_crossings(example_dag, _layered_layout(example_dag))
    dot = _count_crossings(example_dag, graphviz_layout(example_dag, rankdir="LR"))
    assert dot == 0
    assert dot < naive


def test_graphviz_layout_empty_dag() -> None:
    assert graphviz_layout(LabeledDAG(4)) == {}


def test_graphviz_routes_every_edge(example_dag: LabeledDAG) -> None:
    """Edges follow dot's node-avoiding splines, so every edge needs a route."""
    _pos, routes = graphviz_layout_and_routes(example_dag, rankdir="LR")
    expected = {(s, t) for s in range(example_dag.node_count) for t in example_dag.out_neighbors(s)}
    assert set(routes) == expected


def test_graphviz_routes_are_cubic_bezier_sequences(example_dag: LabeledDAG) -> None:
    """Graphviz emits 1 + 3k control points; anything else would draw wrongly."""
    _pos, routes = graphviz_layout_and_routes(example_dag, rankdir="LR")
    for key, pts in routes.items():
        assert len(pts) >= 4, f"edge {key} has too few control points"
        assert (len(pts) - 1) % 3 == 0, f"edge {key} has {len(pts)} control points"


def test_edge_routes_clear_intervening_nodes(example_dag: LabeledDAG) -> None:
    """A routed edge must not pass through a node it is not incident to.

    This is the reason for using splines at all: a straight chord between two
    node centres can cross a third node, which reads as an edge into that node
    that the DAG does not contain.
    """
    import math

    pos, routes = graphviz_layout_and_routes(example_dag, rankdir="LR")
    node_r = 0.90  # the radius the trace figure draws at
    for (src, tgt), pts in routes.items():
        for node, (nx, ny) in pos.items():
            if node in (src, tgt):
                continue
            for px, py in pts:
                assert math.hypot(px - nx, py - ny) > node_r, (
                    f"route {src}->{tgt} passes within the disc of node {node}"
                )


def test_spline_path_rejects_non_cubic_input() -> None:
    """A malformed record must fall back, not be drawn as a wrong curve."""
    assert _spline_path([(0.0, 0.0), (1.0, 1.0)]) is None
    assert _spline_path([(0.0, 0.0)] * 5) is None
    assert _spline_path([(0.0, 0.0)] * 4) is not None


# ---------------------------------------------------------------------------
# Limit fitting
# ---------------------------------------------------------------------------


def test_fit_limits_matches_axes_aspect() -> None:
    """Limits must take the axes' aspect, or equal-aspect axes would letterbox."""
    layout = {0: (-2.0, -1.0), 1: (2.0, 1.0)}
    xlim, ylim = _fit_limits(layout, node_r=0.4, pad=0.1, ax_w_in=2.0, ax_h_in=1.0)
    assert (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) == pytest.approx(2.0)


def test_fit_limits_contains_every_node_with_margin() -> None:
    layout = {0: (-2.0, -1.0), 1: (2.0, 1.0)}
    xlim, ylim = _fit_limits(layout, node_r=0.4, pad=0.1, ax_w_in=2.0, ax_h_in=1.0)
    for x, y in layout.values():
        assert xlim[0] <= x - 0.5 and x + 0.5 <= xlim[1]
        assert ylim[0] <= y - 0.5 and y + 0.5 <= ylim[1]


def test_fit_limits_empty_layout() -> None:
    assert _fit_limits({}, node_r=0.4, pad=0.1, ax_w_in=1.0, ax_h_in=1.0) == (
        (-1.0, 1.0),
        (-1.0, 1.0),
    )


# ---------------------------------------------------------------------------
# Snapshot validation
# ---------------------------------------------------------------------------


def test_trace_rejects_label_mismatch(
    example_dag: LabeledDAG, example_trace: list[Snapshot]
) -> None:
    """A relabelled target must be refused, not drawn at the pinned positions."""
    bad = StringToDAG("VsVcpv*pv+PpcnnC", num_variables=2).run()
    assert bad.node_label(2) is not example_dag.node_label(2)
    with pytest.raises(TraceConsistencyError, match="node IDs do not correspond"):
        make_algorithm_trace_figure(
            [example_trace[-1]],
            bad,
            final_string=EXAMPLE,
            step_labels=["only"],
            direction=STRING_TO_DAG,
        )


def test_trace_rejects_snapshot_larger_than_target(example_trace: list[Snapshot]) -> None:
    small = LabeledDAG(8)
    small.add_node(NodeType.VAR, {"var_index": 0})
    with pytest.raises(TraceConsistencyError, match="more than the target"):
        make_algorithm_trace_figure(
            [example_trace[-1]],
            small,
            final_string=EXAMPLE,
            step_labels=["only"],
            direction=STRING_TO_DAG,
        )


def test_trace_rejects_label_count_mismatch(
    example_dag: LabeledDAG, example_trace: list[Snapshot]
) -> None:
    with pytest.raises(ValueError, match="step labels"):
        make_algorithm_trace_figure(
            example_trace[:2],
            example_dag,
            final_string=EXAMPLE,
            step_labels=["only one"],
            direction=STRING_TO_DAG,
        )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", [STRING_TO_DAG, DAG_TO_STRING])
def test_figure_has_three_panels_per_column(
    example_dag: LabeledDAG, example_trace: list[Snapshot], direction: str
) -> None:
    idx = evenly_spaced_steps(len(example_trace), 5)
    fig = make_algorithm_trace_figure(
        [example_trace[i] for i in idx],
        example_dag,
        final_string=EXAMPLE,
        step_labels=[f"s{i}" for i in idx],
        direction=direction,
    )
    assert len(fig.axes) == 3 * len(idx)


def test_trace_rejects_unknown_direction(
    example_dag: LabeledDAG, example_trace: list[Snapshot]
) -> None:
    with pytest.raises(ValueError, match="direction must be"):
        make_algorithm_trace_figure(
            example_trace[:1],
            example_dag,
            final_string=EXAMPLE,
            step_labels=["only"],
            direction="sideways",
        )


def _solid_token_counts(fig: object) -> list[int]:
    """Return, per column, how many strip tokens are drawn at full opacity."""
    return [
        sum(1 for t in ax.texts if (t.get_alpha() or 1.0) > 0.9)
        for k, ax in enumerate(fig.axes)  # type: ignore[attr-defined]
        if k % 3 == 2
    ]


def test_string_to_dag_empties_the_string(
    example_dag: LabeledDAG, example_trace: list[Snapshot]
) -> None:
    """S2D consumes the string, so its solid token count must fall to zero.

    This is the conservation reading the figure asserts: the solid material is
    the information still in play, so the source representation empties as the
    destination fills.
    """
    idx = evenly_spaced_steps(len(example_trace), 4)
    fig = make_algorithm_trace_figure(
        [example_trace[i] for i in idx],
        example_dag,
        final_string=EXAMPLE,
        step_labels=[f"Step {i}" for i in idx],
        direction=STRING_TO_DAG,
    )
    counts = _solid_token_counts(fig)
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] == len(tokenize_string(EXAMPLE))
    assert counts[-1] == 0


def test_dag_to_string_fills_the_string(example_dag: LabeledDAG) -> None:
    """D2S produces the string, so its solid token count must rise to full."""
    from isalsr.core.dag_to_string import DAGToString

    d2s = DAGToString(example_dag, initial_node=0)
    d2s.run(trace=True)
    trace = list(d2s.trace_log)
    idx = evenly_spaced_steps(len(trace), 4)
    fig = make_algorithm_trace_figure(
        [trace[i] for i in idx],
        example_dag,
        final_string=EXAMPLE,
        step_labels=[f"Step {i}" for i in idx],
        direction=DAG_TO_STRING,
    )
    counts = _solid_token_counts(fig)
    assert counts == sorted(counts), counts
    assert counts[0] == 0
    assert counts[-1] == len(tokenize_string(EXAMPLE))


def test_figure_background_is_transparent(
    example_dag: LabeledDAG, example_trace: list[Snapshot]
) -> None:
    """The manuscript figures must carry no background tint of their own."""
    fig = make_algorithm_trace_figure(
        example_trace[:2],
        example_dag,
        final_string=EXAMPLE,
        step_labels=["a", "b"],
        direction=STRING_TO_DAG,
    )
    assert fig.patch.get_alpha() == 0.0
    for ax in fig.axes:
        assert ax.patch.get_alpha() == 0.0


def test_figure_size_matches_layout(example_dag: LabeledDAG, example_trace: list[Snapshot]) -> None:
    lay = AlgorithmTraceLayout(fig_width=7.16)
    fig = make_algorithm_trace_figure(
        example_trace[:2],
        example_dag,
        final_string=EXAMPLE,
        step_labels=["a", "b"],
        direction=STRING_TO_DAG,
        layout=lay,
    )
    assert tuple(fig.get_size_inches()) == pytest.approx(lay.figsize)


def test_node_glyphs_are_print_legible(
    example_dag: LabeledDAG, example_trace: list[Snapshot]
) -> None:
    """Fitted glyphs must not fall below what a journal figure can carry.

    The fitter guarantees glyphs *fit* their discs, which a 3 pt glyph also
    does. Legibility is the separate constraint, and it is what the figure's
    height is spent on: this test is the tripwire that stops a future height
    saving from silently shrinking the labels out of readability.
    """
    idx = evenly_spaced_steps(len(example_trace), 4)
    lay = AlgorithmTraceLayout()
    fig = make_algorithm_trace_figure(
        [example_trace[i] for i in idx],
        example_dag,
        final_string=EXAMPLE,
        step_labels=[f"Step {i}" for i in idx],
        direction=STRING_TO_DAG,
        layout=lay,
    )
    fig.canvas.draw()
    for row, floor in ((0, 7.0), (1, 6.0)):
        sizes = {
            t.get_fontsize()
            for k, ax in enumerate(fig.axes)
            if k % 3 == row
            for t in ax.texts
            if t.get_gid() == NODE_LABEL_GID
        }
        assert len(sizes) == 1, f"row {row} uses mixed glyph sizes {sorted(sizes)}"
        assert sizes.pop() >= floor


def test_figure_height_stays_within_a_page_band() -> None:
    """A two-column ``figure*`` must not take over the page it lands on."""
    assert AlgorithmTraceLayout().fig_height <= 3.0


# ---------------------------------------------------------------------------
# Sub-view options the trace figure depends on
# ---------------------------------------------------------------------------


def test_strip_fades_only_unemitted_tokens() -> None:
    import matplotlib.pyplot as plt

    _fig, ax = plt.subplots()
    draw_instruction_strip(ax, EXAMPLE, emitted="VcVsp", future_alpha=0.3)
    alphas = [t.get_alpha() for t in ax.texts]
    assert alphas[:3] == [1.0, 1.0, 1.0]
    assert set(alphas[3:]) == {0.3}
    plt.close(_fig)


def test_strip_rejects_emitted_longer_than_string() -> None:
    import matplotlib.pyplot as plt

    _fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="more than"):
        draw_instruction_strip(ax, "Vc", emitted="VcVsp")
    plt.close(_fig)


def test_strip_slots_fix_cell_geometry() -> None:
    """Panels holding different token counts must still share one cell size."""
    import matplotlib.pyplot as plt

    lims = []
    for s in ("Vc", "VcVsp"):
        _fig, ax = plt.subplots()
        draw_instruction_strip(ax, s, n_slots=12)
        lims.append(ax.get_xlim())
        plt.close(_fig)
    assert lims[0] == lims[1]


def test_cdll_slots_fix_cell_geometry(example_trace: list[Snapshot]) -> None:
    """Without fixed slots the CDLL discs shrink as the list grows."""
    import matplotlib.pyplot as plt

    lims = []
    for snap in (example_trace[0], example_trace[-1]):
        dag, cdll, p, q, _ = snap
        _fig, ax = plt.subplots()
        draw_cdll(ax, cdll, dag, p, q, n_slots=6)
        lims.append(ax.get_xlim())
        plt.close(_fig)
    assert lims[0] == lims[1]


def test_cdll_ring_places_nodes_on_a_circle(example_trace: list[Snapshot]) -> None:
    """The CDLL is circular; the drawing must not read as a chain with two ends."""
    import math

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    dag, cdll, p, q, _ = example_trace[-1]
    _fig, ax = plt.subplots()
    draw_cdll_ring(ax, cdll, dag, p, q, node_r=0.32, radius=1.0)
    # The node discs are the patches drawn at the ring radius.
    radii = [
        math.hypot(*c.center)
        for c in ax.patches
        if isinstance(c, Circle) and abs(c.get_radius() - 0.32) < 1e-9
    ]
    assert len(radii) == cdll.size()
    for r in radii:
        assert r == pytest.approx(1.0, abs=1e-9)
    plt.close(_fig)


def test_cdll_ring_draws_one_arrow_per_next_link(example_trace: list[Snapshot]) -> None:
    """A ring of n nodes has n next-links, including the one closing the loop."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    dag, cdll, p, q, _ = example_trace[-1]
    _fig, ax = plt.subplots()
    draw_cdll_ring(ax, cdll, dag, p, q)
    arrows = [a for a in ax.patches if isinstance(a, FancyArrowPatch)]
    # n next-links plus the two pointer markers.
    assert len(arrows) == cdll.size() + 2
    plt.close(_fig)


def test_cdll_ring_geometry_is_independent_of_list_length(
    example_trace: list[Snapshot],
) -> None:
    """A fixed ring radius keeps the discs one physical size across columns."""
    import matplotlib.pyplot as plt

    lims = []
    for snap in (example_trace[0], example_trace[-1]):
        dag, cdll, p, q, _ = snap
        _fig, ax = plt.subplots()
        draw_cdll_ring(ax, cdll, dag, p, q)
        lims.append((ax.get_xlim(), ax.get_ylim()))
        plt.close(_fig)
    assert lims[0] == lims[1]


def test_cdll_pointer_labels_are_configurable(example_trace: list[Snapshot]) -> None:
    """The manuscript names the pointers p and q; the figure must be able to say so."""
    import matplotlib.pyplot as plt

    dag, cdll, p, q, _ = example_trace[-1]
    _fig, ax = plt.subplots()
    draw_cdll(ax, cdll, dag, p, q, primary_label="$p$", secondary_label="$q$")
    labels = {t.get_text() for t in ax.texts}
    assert "$p$" in labels
    assert "$q$" in labels
    plt.close(_fig)
