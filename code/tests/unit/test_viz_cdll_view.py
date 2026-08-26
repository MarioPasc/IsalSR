"""Tests for the CDLL view and the public ``isalsr.viz`` surface.

Two of these pin defects found by visual inspection of the generated figure,
which the type checker and the rest of the suite could not have caught:

- the traversal anchor must not follow the primary pointer, or successive
  snapshots make the ring appear to rotate when only a pointer moved;
- coincident pointers must draw ONE combined marker, not ``"P"`` with
  ``"P,S"`` painted on top of it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from isalsr.core.dag_to_string import DAGToString
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.viz import cdll_traversal, draw_cdll, stable_anchor


def _build_host_dag() -> LabeledDAG:
    """Return ``x_1 + c`` with the constant as an orphan leaf."""
    dag = LabeledDAG(8)
    x1 = dag.add_node(NodeType.VAR, var_index=0)
    const = dag.add_node(NodeType.CONST, const_value=2.0)
    add = dag.add_node(NodeType.ADD)
    dag.add_edge(x1, add)
    dag.add_edge(const, add)
    return dag


def _normalised_snapshots() -> list[tuple[object, object, int, int, str]]:
    """Run D2S on the normalised DAG and return its trace snapshots."""
    conv = DAGToString(_build_host_dag().normalize_const_creation(), 0)
    conv.run(trace=True)
    return list(conv.trace_log)


class TestStableAnchor:
    """The traversal start must not move when the pointers move."""

    def test_anchor_is_lowest_graph_node(self) -> None:
        for _dag, cdll, pri, _sec, _emitted in _normalised_snapshots():
            anchor = stable_anchor(cdll, pri)
            values = [v for _idx, v in cdll_traversal(cdll, anchor)]
            assert values[0] == min(values)

    def test_order_is_invariant_across_snapshots(self) -> None:
        """The drawn order must change only by insertion, never by rotation."""
        orders: list[list[int]] = []
        for _dag, cdll, pri, _sec, _emitted in _normalised_snapshots():
            anchor = stable_anchor(cdll, pri)
            orders.append([v for _idx, v in cdll_traversal(cdll, anchor)])

        # Each successive order must extend the previous one as a subsequence:
        # earlier entries keep their relative order, new nodes are inserted.
        for prev, curr in zip(orders, orders[1:], strict=False):
            it = iter(curr)
            assert all(node in it for node in prev), f"{prev} is not a subsequence of {curr}"

    def test_anchor_ignores_the_pointer_it_is_given(self) -> None:
        _dag, cdll, pri, _sec, _emitted = _normalised_snapshots()[-1]
        entries = cdll_traversal(cdll, pri)
        anchors = {stable_anchor(cdll, idx) for idx, _v in entries}
        assert len(anchors) == 1, "anchor must not depend on the entry point"

    def test_empty_cdll_returns_the_given_pointer(self) -> None:
        from isalsr.core.cdll import CircularDoublyLinkedList

        empty = CircularDoublyLinkedList(4)
        assert stable_anchor(empty, 7) == 7
        assert cdll_traversal(empty, 7) == []


class TestPointerMarkers:
    """Coincident pointers draw one marker; distinct pointers draw two."""

    @staticmethod
    def _marker_labels(pri: int, sec: int) -> list[str]:
        dag, cdll, _p, _s, _e = _normalised_snapshots()[-1]
        fig, ax = plt.subplots()
        try:
            draw_cdll(ax, cdll, dag, pri, sec)
            return [t.get_text() for t in ax.texts]
        finally:
            plt.close(fig)

    def test_coincident_pointers_draw_one_combined_marker(self) -> None:
        _dag, cdll, pri, _sec, _e = _normalised_snapshots()[-1]
        labels = self._marker_labels(pri, pri)
        assert labels.count("P,S") == 1
        assert "P" not in labels, "a bare 'P' under the combined marker overlaps it"
        assert "S" not in labels

    def test_distinct_pointers_draw_two_markers(self) -> None:
        _dag, cdll, pri, sec, _e = _normalised_snapshots()[-1]
        assert pri != sec, "fixture precondition: final snapshot separates the pointers"
        labels = self._marker_labels(pri, sec)
        assert labels.count("P") == 1
        assert labels.count("S") == 1
        assert "P,S" not in labels


class TestPublicSurface:
    """Callers import from ``isalsr.viz``, not from its submodules."""

    def test_entry_points_are_exported(self) -> None:
        import isalsr.viz as viz

        for name in (
            "make_trace_figure",
            "draw_dag",
            "draw_instruction_strip",
            "draw_cdll",
            "TraceLayout",
            "DEFAULT_TRACE_LAYOUT",
        ):
            assert name in viz.__all__, f"{name} missing from isalsr.viz.__all__"
            assert hasattr(viz, name)

    def test_matplotlib_is_not_imported_at_module_scope(self) -> None:
        """The dependency rule: matplotlib enters only on a drawing call."""
        import isalsr.viz.cdll_view as mod

        src = pytest.importorskip("inspect").getsource(mod)
        header = src.split("def ", 1)[0]
        assert "import matplotlib" not in header
