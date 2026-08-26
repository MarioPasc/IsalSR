"""Unit tests for the D2S trace API and the trace-figure viz modules.

Covers:
  - Default ``DAGToString.run()`` refuses a violated DAG with ``ValueError``.
  - ``run(trace=True, skip_precondition_check=True)`` stalls mid-run, raises
    ``RuntimeError``, and leaves exactly 2 snapshots in ``trace_log``.
  - ``run(trace=True)`` on the normalised DAG produces 4 snapshots and the
    correct D2S string ``"VkV+PnC"``.
  - Canonical string for the normalised DAG is ``"V+VkPnc"`` on both backends.
  - Stall snapshot has fewer nodes/edges than the final normalised snapshot.
  - Viz: ``assign_positions`` maps all output-dag nodes for each snapshot.
  - Viz: ``draw_strip_panel`` runs without error on typical inputs.
"""

from __future__ import annotations

import contextlib

import pytest

from isalsr.core.canonical import fast_canonical_string
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

# ---------------------------------------------------------------------------
# Expected constants (verified against live code; see also the generator).
# ---------------------------------------------------------------------------

_D2S_STRING: str = "VkV+PnC"
_CANONICAL_STRING: str = "V+VkPnc"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def normalised_dag() -> tuple[LabeledDAG, int]:
    """Normalised x_1 + c: CONST c has creation edge x_1 -> c."""
    dag = LabeledDAG(8)
    x1 = dag.add_node(NodeType.VAR, var_index=0)
    c = dag.add_node(NodeType.CONST)
    add = dag.add_node(NodeType.ADD)
    dag.add_edge(x1, c)  # creation edge
    dag.add_edge(x1, add)
    dag.add_edge(c, add)
    return dag, x1


@pytest.fixture()
def violated_dag() -> tuple[LabeledDAG, int]:
    """Violated x_1 + c: CONST c has in-degree 0 (no creation edge)."""
    dag = LabeledDAG(8)
    x1 = dag.add_node(NodeType.VAR, var_index=0)
    c = dag.add_node(NodeType.CONST)
    add = dag.add_node(NodeType.ADD)
    dag.add_edge(x1, add)
    dag.add_edge(c, add)
    return dag, x1


# ---------------------------------------------------------------------------
# Test: default run raises on violated DAG
# ---------------------------------------------------------------------------


def test_default_run_raises_on_violated(violated_dag: tuple[LabeledDAG, int]) -> None:
    """``run()`` without bypass must raise ``ValueError`` on a violated DAG."""
    dag, x1 = violated_dag
    conv = DAGToString(dag, initial_node=x1)
    with pytest.raises(ValueError, match="reachable"):
        conv.run()


def test_default_run_leaves_trace_log_empty_on_violated(
    violated_dag: tuple[LabeledDAG, int],
) -> None:
    """After a failed default run the trace log must be empty."""
    dag, x1 = violated_dag
    conv = DAGToString(dag, initial_node=x1)
    with contextlib.suppress(ValueError):
        conv.run(trace=True)
    assert conv.trace_log == [], "trace_log must be empty after a precondition failure"


# ---------------------------------------------------------------------------
# Test: bypass run stalls and produces partial trace
# ---------------------------------------------------------------------------


def test_bypass_raises_runtime_error(violated_dag: tuple[LabeledDAG, int]) -> None:
    """Bypass run on a violated DAG must raise ``RuntimeError`` (stall)."""
    dag, x1 = violated_dag
    conv = DAGToString(dag, initial_node=x1)
    with pytest.raises(RuntimeError, match="no valid operation"):
        conv.run(trace=True, skip_precondition_check=True)


def test_bypass_trace_has_exactly_two_snapshots(
    violated_dag: tuple[LabeledDAG, int],
) -> None:
    """Violated bypass must leave exactly 2 snapshots in ``trace_log``."""
    dag, x1 = violated_dag
    conv = DAGToString(dag, initial_node=x1)
    with contextlib.suppress(RuntimeError):
        conv.run(trace=True, skip_precondition_check=True)
    assert len(conv.trace_log) == 2


def test_bypass_snap0_is_initial_state(violated_dag: tuple[LabeledDAG, int]) -> None:
    """Snapshot 0 (before loop start) must show only x_1 and no edges."""
    dag, x1 = violated_dag
    conv = DAGToString(dag, initial_node=x1)
    with contextlib.suppress(RuntimeError):
        conv.run(trace=True, skip_precondition_check=True)
    out_dag, cdll, pp, sp, emitted = conv.trace_log[0]
    assert out_dag.node_count == 1
    assert out_dag.edge_count == 0
    assert emitted == ""
    assert cdll.get_value(pp) == x1
    assert cdll.get_value(sp) == x1


def test_bypass_snap1_has_add_node(violated_dag: tuple[LabeledDAG, int]) -> None:
    """Snapshot 1 must show x_1 and ADD (+) inserted, emitted='V+'."""
    dag, x1 = violated_dag
    conv = DAGToString(dag, initial_node=x1)
    with contextlib.suppress(RuntimeError):
        conv.run(trace=True, skip_precondition_check=True)
    out_dag, cdll, pp, sp, emitted = conv.trace_log[1]
    assert out_dag.node_count == 2
    assert out_dag.edge_count == 1
    assert emitted == "V+"
    # Primary and secondary still on x_1 (pointer does not move after V).
    assert cdll.get_value(pp) == x1
    assert cdll.get_value(sp) == x1


# ---------------------------------------------------------------------------
# Test: normal trace on normalised DAG
# ---------------------------------------------------------------------------


def test_normalised_run_d2s_string(normalised_dag: tuple[LabeledDAG, int]) -> None:
    """D2S string for normalised x_1 + c must be ``_D2S_STRING``."""
    dag, x1 = normalised_dag
    result = DAGToString(dag, initial_node=x1).run()
    assert result == _D2S_STRING


def test_normalised_trace_has_four_snapshots(
    normalised_dag: tuple[LabeledDAG, int],
) -> None:
    """Normalised trace must produce exactly 4 snapshots."""
    dag, x1 = normalised_dag
    conv = DAGToString(dag, initial_node=x1)
    conv.run(trace=True)
    assert len(conv.trace_log) == 4


def test_normalised_trace_snap0(normalised_dag: tuple[LabeledDAG, int]) -> None:
    """Snapshot 0 must show only x_1, no edges, emitted empty."""
    dag, x1 = normalised_dag
    conv = DAGToString(dag, initial_node=x1)
    conv.run(trace=True)
    out_dag, cdll, pp, sp, emitted = conv.trace_log[0]
    assert out_dag.node_count == 1
    assert out_dag.edge_count == 0
    assert emitted == ""


def test_normalised_trace_snap1_has_const(normalised_dag: tuple[LabeledDAG, int]) -> None:
    """Snapshot 1 must show x_1 and CONST c, emitted='Vk'."""
    dag, x1 = normalised_dag
    conv = DAGToString(dag, initial_node=x1)
    conv.run(trace=True)
    out_dag, cdll, pp, sp, emitted = conv.trace_log[1]
    assert out_dag.node_count == 2
    node_types = {out_dag.node_label(n) for n in range(out_dag.node_count)}
    assert NodeType.CONST in node_types
    assert emitted == "Vk"


def test_normalised_trace_snap2_has_add(normalised_dag: tuple[LabeledDAG, int]) -> None:
    """Snapshot 2 must show x_1, CONST, ADD, 2 edges, emitted='VkV+'."""
    dag, x1 = normalised_dag
    conv = DAGToString(dag, initial_node=x1)
    conv.run(trace=True)
    out_dag, cdll, pp, sp, emitted = conv.trace_log[2]
    assert out_dag.node_count == 3
    assert out_dag.edge_count == 2
    assert emitted == "VkV+"


def test_normalised_trace_snap3_final(normalised_dag: tuple[LabeledDAG, int]) -> None:
    """Snapshot 3 (post-loop) must be complete: 3 nodes, 3 edges, full string."""
    dag, x1 = normalised_dag
    conv = DAGToString(dag, initial_node=x1)
    conv.run(trace=True)
    out_dag, cdll, pp, sp, emitted = conv.trace_log[3]
    assert out_dag.node_count == 3
    assert out_dag.edge_count == 3
    assert emitted == _D2S_STRING


# ---------------------------------------------------------------------------
# Test: canonical string on both backends
# ---------------------------------------------------------------------------


def test_canonical_string_python_backend(normalised_dag: tuple[LabeledDAG, int]) -> None:
    """Canonical string on python backend must match ``_CANONICAL_STRING``."""
    dag, _ = normalised_dag
    cs = fast_canonical_string(dag, backend="python")  # type: ignore[arg-type]
    assert cs == _CANONICAL_STRING


def test_canonical_string_cpp_backend(normalised_dag: tuple[LabeledDAG, int]) -> None:
    """Canonical string on cpp backend must match ``_CANONICAL_STRING``."""
    dag, _ = normalised_dag
    cs = fast_canonical_string(dag, backend="cpp")  # type: ignore[arg-type]
    assert cs == _CANONICAL_STRING


def test_canonical_backends_agree(normalised_dag: tuple[LabeledDAG, int]) -> None:
    """Python and cpp backends must return identical canonical strings."""
    dag, _ = normalised_dag
    py = fast_canonical_string(dag, backend="python")  # type: ignore[arg-type]
    cpp = fast_canonical_string(dag, backend="cpp")  # type: ignore[arg-type]
    assert py == cpp


def test_canonical_refuses_violated_dag_python(
    violated_dag: tuple[LabeledDAG, int],
) -> None:
    """``fast_canonical_string`` must raise on a violated DAG (python)."""
    dag, _ = violated_dag
    with pytest.raises((ValueError, RuntimeError)):
        fast_canonical_string(dag, backend="python")  # type: ignore[arg-type]


def test_canonical_refuses_violated_dag_cpp(
    violated_dag: tuple[LabeledDAG, int],
) -> None:
    """``fast_canonical_string`` must raise on a violated DAG (cpp)."""
    dag, _ = violated_dag
    with pytest.raises((ValueError, RuntimeError)):
        fast_canonical_string(dag, backend="cpp")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test: trace_log is a list of 5-tuples with correct types
# ---------------------------------------------------------------------------


def test_trace_log_tuple_types(normalised_dag: tuple[LabeledDAG, int]) -> None:
    """Each trace-log entry must be a 5-tuple of the declared types."""
    from isalsr.core.cdll import CircularDoublyLinkedList

    dag, x1 = normalised_dag
    conv = DAGToString(dag, initial_node=x1)
    conv.run(trace=True)
    for i, entry in enumerate(conv.trace_log):
        assert len(entry) == 5, f"snap[{i}] has {len(entry)} elements, expected 5"
        out_dag, cdll, pp, sp, emitted = entry
        assert isinstance(out_dag, LabeledDAG), f"snap[{i}] out_dag type wrong"
        assert isinstance(cdll, CircularDoublyLinkedList), f"snap[{i}] cdll type wrong"
        assert isinstance(pp, int), f"snap[{i}] primary_ptr type wrong"
        assert isinstance(sp, int), f"snap[{i}] secondary_ptr type wrong"
        assert isinstance(emitted, str), f"snap[{i}] emitted type wrong"


# ---------------------------------------------------------------------------
# Test: viz assign_positions maps output-dag nodes correctly
# ---------------------------------------------------------------------------


def test_assign_positions_normalised_snap2(
    normalised_dag: tuple[LabeledDAG, int],
) -> None:
    """All 3 output-dag nodes in snap2 must have a position in LAYOUT."""
    from experiments.scripts.generate_fig_reachability import LAYOUT
    from isalsr.viz.composite import assign_positions

    dag, x1 = normalised_dag
    conv = DAGToString(dag, initial_node=x1)
    conv.run(trace=True)
    out_dag, *_ = conv.trace_log[2]

    inp_types = {nd: dag.node_label(nd) for nd in range(dag.node_count)}
    positions = assign_positions(out_dag, LAYOUT, inp_types)

    assert len(positions) == out_dag.node_count, (
        f"Expected {out_dag.node_count} positions, got {len(positions)}"
    )


# ---------------------------------------------------------------------------
# Test: draw_strip_panel runs without matplotlib error
# ---------------------------------------------------------------------------


def test_draw_strip_panel_emitted_only() -> None:
    """``draw_strip_panel`` must not raise on a partial string."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from isalsr.viz.composite import draw_strip_panel

    fig, ax = plt.subplots()
    draw_strip_panel(ax, "VkV+", layout_obj=None)
    plt.close(fig)


def test_draw_strip_panel_with_final_string() -> None:
    """``draw_strip_panel`` must not raise when showing future tokens."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from isalsr.viz.composite import draw_strip_panel

    fig, ax = plt.subplots()
    draw_strip_panel(ax, "Vk", final_string=_D2S_STRING, layout_obj=None)
    plt.close(fig)


def test_draw_strip_panel_stall() -> None:
    """``draw_strip_panel`` with ``stall=True`` must not raise."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from isalsr.viz.composite import draw_strip_panel

    fig, ax = plt.subplots()
    draw_strip_panel(ax, "V+", stall=True, layout_obj=None)
    plt.close(fig)
