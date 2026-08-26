"""Tests for the matplotlib DAG backend.

Verifies: backend registration, draw return type, layout pinning, and
the layered layout algorithm on simple structures.
"""

from __future__ import annotations

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.viz.backends.matplotlib_dag import MatplotlibDagBackend, _layered_layout

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sin_x1_dag() -> LabeledDAG:
    """Return sin(x_1): two nodes, one edge x_1 -> sin."""
    dag = LabeledDAG(8)
    x1 = dag.add_node(NodeType.VAR, var_index=0)
    sin = dag.add_node(NodeType.SIN)
    dag.add_edge(x1, sin)
    return dag


def _x1_plus_c_host() -> LabeledDAG:
    """Return x_1 + c where c has no in-edge (precondition violation)."""
    dag = LabeledDAG(8)
    x1 = dag.add_node(NodeType.VAR, var_index=0)
    c = dag.add_node(NodeType.CONST, const_value=2.0)
    add = dag.add_node(NodeType.ADD)
    dag.add_edge(x1, add)
    dag.add_edge(c, add)
    return dag


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_layered_layout_returns_entry_for_every_node() -> None:
    dag = _sin_x1_dag()
    layout = _layered_layout(dag)
    assert set(layout.keys()) == set(range(dag.node_count))


def test_layered_layout_source_at_layer_zero() -> None:
    dag = _sin_x1_dag()
    layout = _layered_layout(dag)
    # x_1 (node 0) is a source; sin (node 1) is above it.
    x, y_x1 = layout[0]
    _, y_sin = layout[1]
    assert y_sin > y_x1


def test_layered_layout_empty_dag() -> None:
    dag = LabeledDAG(4)
    assert _layered_layout(dag) == {}


# ---------------------------------------------------------------------------
# Backend draw return value
# ---------------------------------------------------------------------------


def _make_axes() -> object:
    """Return a real matplotlib Axes so we can call draw without mocking."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plt.close(fig)
    return ax


def test_draw_returns_layout_for_all_nodes() -> None:
    """draw() returns a layout dict with an entry for every node in the DAG."""
    backend = MatplotlibDagBackend()
    dag = _sin_x1_dag()
    ax = _make_axes()
    layout = backend.draw(dag, ax)
    assert set(layout.keys()) == set(range(dag.node_count))


def test_draw_respects_provided_layout() -> None:
    """When a layout is supplied, draw() must return exactly that layout."""
    backend = MatplotlibDagBackend()
    dag = _sin_x1_dag()
    fixed: dict[int, tuple[float, float]] = {0: (0.0, 0.0), 1: (0.0, 3.0)}
    ax = _make_axes()
    returned = backend.draw(dag, ax, layout=fixed)
    assert returned == fixed


def test_draw_violation_dag_does_not_raise() -> None:
    """Drawing a precondition-violating DAG (orphan CONST) must not raise."""
    backend = MatplotlibDagBackend()
    dag = _x1_plus_c_host()
    ax = _make_axes()
    layout = backend.draw(dag, ax)
    assert len(layout) == dag.node_count


def test_backend_name() -> None:
    assert MatplotlibDagBackend().name == "matplotlib"


def test_draw_reachable_set_accepted() -> None:
    """reachable parameter is accepted and does not raise."""
    backend = MatplotlibDagBackend()
    dag = _sin_x1_dag()
    ax = _make_axes()
    layout = backend.draw(dag, ax, reachable=frozenset([0, 1]))
    assert len(layout) == dag.node_count
