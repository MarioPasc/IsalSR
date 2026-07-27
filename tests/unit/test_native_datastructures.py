"""Differential tests: C++ NativeCDLL / NativeLabeledDAG vs Python oracles.

Each randomised sequence applies identical operations to both a Python reference
object and the corresponding C++ object, then compares every observable field.

Coverage:
  CDLL  — 500 randomised operation sequences (seed 0..499), each 30 ops.
  DAG   — 500 randomised operation sequences (seed 0..499), each 30 ops.
  Plus explicit invariant-targeted tests for Invariants 1, 3, 8, 9.

Tested invariants (from CLAUDE.md Critical Invariants):
  1. CDLL slot indices are NOT graph node indices — get_value(slot) returns the
     graph node ID stored in that slot, not the slot index itself.
  3. add_edge(source, target) direction, and input_order_ tracks insertion order.
  8. ordered_inputs() preserves insertion order for binary ops (SUB/DIV/POW).
  9. normalize_const_creation() moves CONST edges to node 0 in sorted order.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest
from isalsr.core._native import testing as _nt  # type: ignore[import-untyped]

from isalsr.core.cdll import CircularDoublyLinkedList
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

NativeCDLL: Any = _nt.NativeCDLL
NativeLabeledDAG: Any = _nt.NativeLabeledDAG

# ---------------------------------------------------------------------------
# Fixed NodeType → integer mapping (must match labeled_dag.hpp enum class).
# ---------------------------------------------------------------------------
NODE_TYPE_INT: dict[NodeType, int] = {
    NodeType.VAR: 0,
    NodeType.ADD: 1,
    NodeType.MUL: 2,
    NodeType.SUB: 3,
    NodeType.DIV: 4,
    NodeType.SIN: 5,
    NodeType.COS: 6,
    NodeType.EXP: 7,
    NodeType.LOG: 8,
    NodeType.SQRT: 9,
    NodeType.POW: 10,
    NodeType.ABS: 11,
    NodeType.NEG: 12,
    NodeType.INV: 13,
    NodeType.CONST: 14,
}

INT_NODE_TYPE: dict[int, NodeType] = {v: k for k, v in NODE_TYPE_INT.items()}

# Ordered list for random sampling.
_ALL_LABELS: list[NodeType] = list(NodeType)

# NodeTypes whose arity requires var_index metadata.
_VAR_TYPES: frozenset[NodeType] = frozenset({NodeType.VAR})

# NodeTypes whose arity requires const_value metadata.
_CONST_TYPES: frozenset[NodeType] = frozenset({NodeType.CONST})


# ---------------------------------------------------------------------------
# Helper — traverse CDLL from `start` slot, returning the circular value sequence
# ---------------------------------------------------------------------------


def _traverse_cdll_py(cdll: CircularDoublyLinkedList, start: int) -> list[int]:
    """Traverse the circular list from `start`, returning all payloads once."""
    values = [cdll.get_value(start)]
    current = cdll.next_node(start)
    while current != start:
        values.append(cdll.get_value(current))
        current = cdll.next_node(current)
    return values


def _traverse_cdll_cpp(cdll: Any, start: int) -> list[int]:
    """Same traversal for the C++ NativeCDLL binding."""
    values = [cdll.get_value(start)]
    current = cdll.next_node(start)
    while current != start:
        values.append(cdll.get_value(current))
        current = cdll.next_node(current)
    return values


# ---------------------------------------------------------------------------
# Helper — compare DAG observable state (Python vs C++)
# ---------------------------------------------------------------------------


def _compare_dags(
    py_dag: LabeledDAG,
    cpp_dag: Any,
    *,
    check_node_data: bool = True,
) -> None:
    """Assert that py_dag and cpp_dag have identical observable state."""
    nc = py_dag.node_count
    assert cpp_dag.node_count == nc, f"node_count mismatch: py={nc}, cpp={cpp_dag.node_count}"
    assert cpp_dag.edge_count == py_dag.edge_count, (
        f"edge_count mismatch: py={py_dag.edge_count}, cpp={cpp_dag.edge_count}"
    )

    for i in range(nc):
        # Labels
        py_label = py_dag.node_label(i)
        cpp_label_int = cpp_dag.node_label(i)
        assert NODE_TYPE_INT[py_label] == cpp_label_int, (
            f"node {i}: label mismatch py={py_label}, cpp={INT_NODE_TYPE.get(cpp_label_int)}"
        )

        # Adjacency (sort both since Python returns frozenset, C++ returns sorted list)
        assert sorted(cpp_dag.out_neighbors(i)) == sorted(py_dag.out_neighbors(i)), (
            f"node {i}: out_neighbors mismatch"
        )
        assert sorted(cpp_dag.in_neighbors(i)) == sorted(py_dag.in_neighbors(i)), (
            f"node {i}: in_neighbors mismatch"
        )

        # Ordered inputs — insertion order must match exactly (Invariant 8)
        assert cpp_dag.ordered_inputs(i) == py_dag.ordered_inputs(i), (
            f"node {i}: ordered_inputs mismatch "
            f"py={py_dag.ordered_inputs(i)}, cpp={cpp_dag.ordered_inputs(i)}"
        )

        if check_node_data:
            py_data = py_dag.node_data(i)
            cpp_vi = cpp_dag.node_var_index(i)
            cpp_cv = cpp_dag.node_const_value(i)

            if "var_index" in py_data:
                assert cpp_vi == py_data["var_index"], (
                    f"node {i}: var_index mismatch py={py_data['var_index']}, cpp={cpp_vi}"
                )
            else:
                assert cpp_vi == -1, f"node {i}: expected var_index=-1, got {cpp_vi}"

            if "const_value" in py_data:
                assert not math.isnan(cpp_cv), f"node {i}: expected const_value, got NaN"
                assert math.isclose(cpp_cv, float(py_data["const_value"]), rel_tol=1e-12), (
                    f"node {i}: const_value mismatch py={py_data['const_value']}, cpp={cpp_cv}"
                )
            else:
                assert math.isnan(cpp_cv), f"node {i}: expected no const_value (NaN), got {cpp_cv}"


# ---------------------------------------------------------------------------
# CDLL differential tests — 500 random sequences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(500))
def test_cdll_differential(seed: int) -> None:
    """Randomised CDLL operations: C++ and Python must produce identical state."""
    rng = random.Random(seed)
    capacity = 20
    py_cdll = CircularDoublyLinkedList(capacity)
    cpp_cdll = NativeCDLL(capacity)

    # Track alive slot indices.  Both implementations allocate the same slot
    # indices (free-list pops in the same order), so one list suffices.
    alive: list[int] = []

    for _ in range(30):
        weights = [3, 2, 1, 1]  # insert more often than remove
        op = rng.choices(["insert", "remove", "set_value", "check"], weights=weights)[0]

        if op == "insert" and len(alive) < capacity - 2:
            value = rng.randint(0, 200)
            if not alive:
                py_slot = py_cdll.insert_after(0, value)
                cpp_slot = cpp_cdll.insert_after(0, value)
            else:
                after = rng.choice(alive)
                py_slot = py_cdll.insert_after(after, value)
                cpp_slot = cpp_cdll.insert_after(after, value)
            assert py_slot == cpp_slot, (
                f"seed={seed}: insert_after slot mismatch py={py_slot}, cpp={cpp_slot}"
            )
            alive.append(py_slot)

        elif op == "remove" and alive:
            slot = rng.choice(alive)
            alive.remove(slot)
            py_cdll.remove(slot)
            cpp_cdll.remove(slot)

        elif op == "set_value" and alive:
            slot = rng.choice(alive)
            value = rng.randint(0, 200)
            py_cdll.set_value(slot, value)
            cpp_cdll.set_value(slot, value)

        # Always verify structural consistency after every operation.
        assert py_cdll.size() == cpp_cdll.size(), (
            f"seed={seed}: size mismatch py={py_cdll.size()}, cpp={cpp_cdll.size()}"
        )
        assert len(alive) == py_cdll.size()

        if alive:
            start = alive[0]
            py_vals = _traverse_cdll_py(py_cdll, start)
            cpp_vals = _traverse_cdll_cpp(cpp_cdll, start)
            assert py_vals == cpp_vals, (
                f"seed={seed}: traversal mismatch py={py_vals}, cpp={cpp_vals}"
            )
            assert len(py_vals) == py_cdll.size()
            # next/prev must agree at every alive slot
            for s in alive:
                assert py_cdll.next_node(s) == cpp_cdll.next_node(s)
                assert py_cdll.prev_node(s) == cpp_cdll.prev_node(s)
                assert py_cdll.get_value(s) == cpp_cdll.get_value(s)


# ---------------------------------------------------------------------------
# DAG differential tests — 500 random sequences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(500))
def test_dag_differential(seed: int) -> None:
    """Randomised DAG operations: C++ and Python must produce identical state."""
    rng = random.Random(seed)
    max_nodes = 20
    py_dag = LabeledDAG(max_nodes)
    cpp_dag = NativeLabeledDAG(max_nodes)

    for _ in range(30):
        nc = py_dag.node_count
        op = rng.choices(
            ["add_node", "add_edge", "remove_edge", "undo_node"],
            weights=[4, 4, 1, 1],
        )[0]

        if op == "add_node" and nc < max_nodes - 1:
            label = rng.choice(_ALL_LABELS)
            var_idx = rng.randint(0, 4) if label in _VAR_TYPES else None
            cval = rng.uniform(-10.0, 10.0) if label in _CONST_TYPES else None

            py_result = py_dag.add_node(label, var_index=var_idx, const_value=cval)
            cpp_result = cpp_dag.add_node(
                NODE_TYPE_INT[label],
                var_idx if var_idx is not None else -1,
                cval if cval is not None else float("nan"),
            )
            assert py_result == cpp_result, (
                f"seed={seed}: add_node result mismatch py={py_result}, cpp={cpp_result}"
            )

        elif op == "add_edge" and nc >= 2:
            src = rng.randint(0, nc - 1)
            tgt = rng.randint(0, nc - 1)
            py_result = py_dag.add_edge(src, tgt)
            cpp_result = cpp_dag.add_edge(src, tgt)
            assert py_result == cpp_result, (
                f"seed={seed}: add_edge({src},{tgt}) mismatch py={py_result}, cpp={cpp_result}"
            )

        elif op == "remove_edge" and nc >= 2:
            src = rng.randint(0, nc - 1)
            tgt = rng.randint(0, nc - 1)
            py_result = py_dag.remove_edge(src, tgt)
            cpp_result = cpp_dag.remove_edge(src, tgt)
            assert py_result == cpp_result, f"seed={seed}: remove_edge({src},{tgt}) mismatch"

        elif op == "undo_node" and nc > 0:
            py_dag.undo_node()
            cpp_dag.undo_node()

        _compare_dags(py_dag, cpp_dag)


# ---------------------------------------------------------------------------
# Invariant 1: CDLL slot index ≠ graph node ID
# ---------------------------------------------------------------------------


class TestInvariant1CDLLIndexNotGraphNode:
    """CDLL slot indices are NOT graph node IDs.

    Invariant 1: pointers are CDLL slot indices; payloads are graph node IDs.
    get_value(slot) returns the graph node ID, NOT the slot index.
    """

    def test_first_slot_stores_arbitrary_graph_node(self) -> None:
        """First allocated CDLL slot is 0; its payload is the given graph node ID."""
        graph_node = 42  # arbitrary large graph node ID
        py = CircularDoublyLinkedList(10)
        cpp = NativeCDLL(10)

        py_slot = py.insert_after(0, graph_node)  # slot 0 allocated
        cpp_slot = cpp.insert_after(0, graph_node)

        assert py_slot == 0, "Python must allocate slot 0 first"
        assert cpp_slot == 0, "C++ must allocate slot 0 first"

        # The slot INDEX is 0; the PAYLOAD must be 42 (not 0).
        assert py.get_value(0) == graph_node, "get_value(slot=0) must return graph_node=42"
        assert cpp.get_value(0) == graph_node, "C++ get_value(slot=0) must return graph_node=42"

        # Explicit contradiction: slot 0 ≠ graph node 42.
        assert py_slot != graph_node
        assert cpp_slot != graph_node

    def test_slot_index_differs_from_graph_node_id(self) -> None:
        """Multiple slots: each slot index ≠ its stored graph node ID."""
        graph_ids = [100, 200, 300, 400, 500]
        py = CircularDoublyLinkedList(10)
        cpp = NativeCDLL(10)

        slots: list[int] = []
        prev = 0
        for gid in graph_ids:
            py_slot = py.insert_after(prev, gid)
            cpp_slot = cpp.insert_after(prev, gid)
            assert py_slot == cpp_slot
            assert py.get_value(py_slot) == gid
            assert cpp.get_value(cpp_slot) == gid
            # Slot index (0,1,2,…) never equals graph node ID (100,200,…)
            assert py_slot != gid
            slots.append(py_slot)
            prev = py_slot

    def test_set_value_changes_payload_not_slot_index(self) -> None:
        """set_value changes the payload (graph node ID) without changing the slot."""
        py = CircularDoublyLinkedList(5)
        cpp = NativeCDLL(5)

        slot = py.insert_after(0, 7)
        cpp.insert_after(0, 7)

        py.set_value(slot, 99)
        cpp.set_value(slot, 99)

        assert py.get_value(slot) == 99  # payload updated
        assert cpp.get_value(slot) == 99
        assert slot == 0  # slot index unchanged


# ---------------------------------------------------------------------------
# Invariant 3: add_edge direction and input_order
# ---------------------------------------------------------------------------


class TestInvariant3EdgeDirectionAndInputOrder:
    """add_edge(source, target) means source→target (source provides input to target).

    input_order[target] records insertion order of sources.
    """

    def test_edge_direction_out_in_adjacency(self) -> None:
        """add_edge(u, v): u appears in out_neighbors(u), v appears in in_neighbors(v)."""
        py = LabeledDAG(5)
        cpp = NativeLabeledDAG(5)

        x = 0
        s = 1
        for dag, is_cpp in [(py, False), (cpp, True)]:
            if is_cpp:
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 0, float("nan"))  # node 0
                dag.add_node(NODE_TYPE_INT[NodeType.SIN], -1, float("nan"))  # node 1
            else:
                dag.add_node(NodeType.VAR, var_index=0)
                dag.add_node(NodeType.SIN)

        py.add_edge(x, s)
        cpp.add_edge(x, s)

        # u → v means u is in out_neighbors(u) and v is in in_neighbors(v).
        assert s in py.out_neighbors(x)
        assert s in cpp.out_neighbors(x)
        assert x in py.in_neighbors(s)
        assert x in cpp.in_neighbors(s)
        # No reverse edge.
        assert x not in py.out_neighbors(s)
        assert x not in cpp.out_neighbors(s)

    def test_input_order_preserved_for_two_edges(self) -> None:
        """input_order records the order in which source edges were added."""
        py = LabeledDAG(5)
        cpp = NativeLabeledDAG(5)

        for dag, is_cpp in [(py, False), (cpp, True)]:
            if is_cpp:
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 0, float("nan"))  # 0
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 1, float("nan"))  # 1
                dag.add_node(NODE_TYPE_INT[NodeType.ADD], -1, float("nan"))  # 2
            else:
                dag.add_node(NodeType.VAR, var_index=0)
                dag.add_node(NodeType.VAR, var_index=1)
                dag.add_node(NodeType.ADD)

        # First edge: 0 → 2
        py.add_edge(0, 2)
        cpp.add_edge(0, 2)
        # Second edge: 1 → 2
        py.add_edge(1, 2)
        cpp.add_edge(1, 2)

        assert py.ordered_inputs(2) == [0, 1]
        assert cpp.ordered_inputs(2) == [0, 1]

    def test_remove_edge_updates_input_order(self) -> None:
        """remove_edge removes the first occurrence from input_order."""
        py = LabeledDAG(5)
        cpp = NativeLabeledDAG(5)

        for dag, is_cpp in [(py, False), (cpp, True)]:
            if is_cpp:
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 0, float("nan"))
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 1, float("nan"))
                dag.add_node(NODE_TYPE_INT[NodeType.ADD], -1, float("nan"))
            else:
                dag.add_node(NodeType.VAR, var_index=0)
                dag.add_node(NodeType.VAR, var_index=1)
                dag.add_node(NodeType.ADD)

        py.add_edge(0, 2)
        py.add_edge(1, 2)
        cpp.add_edge(0, 2)
        cpp.add_edge(1, 2)

        py.remove_edge(0, 2)
        cpp.remove_edge(0, 2)

        assert py.ordered_inputs(2) == [1]
        assert cpp.ordered_inputs(2) == [1]


# ---------------------------------------------------------------------------
# Invariant 8: operand order for binary ops
# ---------------------------------------------------------------------------


class TestInvariant8BinaryOpOperandOrder:
    """ordered_inputs() returns insertion order, NOT sorted order.

    For SUB(x, y): first add_edge(x, sub), then add_edge(y, sub).
    ordered_inputs(sub) must be [x, y], not [y, x] even if y < x.
    """

    def test_sub_first_operand_by_insertion(self) -> None:
        """x - y: ordered_inputs = [x, y] regardless of node ID order."""
        py = LabeledDAG(5)
        cpp = NativeLabeledDAG(5)

        for dag, is_cpp in [(py, False), (cpp, True)]:
            if is_cpp:
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 0, float("nan"))  # node 0 = x
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 1, float("nan"))  # node 1 = y
                dag.add_node(NODE_TYPE_INT[NodeType.SUB], -1, float("nan"))  # node 2 = sub
            else:
                dag.add_node(NodeType.VAR, var_index=0)
                dag.add_node(NodeType.VAR, var_index=1)
                dag.add_node(NodeType.SUB)

        # x is first operand (V/v edge), y is second (C/c edge).
        py.add_edge(0, 2)
        py.add_edge(1, 2)
        cpp.add_edge(0, 2)
        cpp.add_edge(1, 2)

        assert py.ordered_inputs(2) == [0, 1]
        assert cpp.ordered_inputs(2) == [0, 1]

    def test_sub_reversed_insertion_gives_reversed_operand_order(self) -> None:
        """y - x: ordered_inputs = [y, x] when y is inserted first."""
        py = LabeledDAG(5)
        cpp = NativeLabeledDAG(5)

        for dag, is_cpp in [(py, False), (cpp, True)]:
            if is_cpp:
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 0, float("nan"))  # 0 = x
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 1, float("nan"))  # 1 = y
                dag.add_node(NODE_TYPE_INT[NodeType.SUB], -1, float("nan"))  # 2 = sub
            else:
                dag.add_node(NodeType.VAR, var_index=0)
                dag.add_node(NodeType.VAR, var_index=1)
                dag.add_node(NodeType.SUB)

        # y first, then x.
        py.add_edge(1, 2)
        py.add_edge(0, 2)
        cpp.add_edge(1, 2)
        cpp.add_edge(0, 2)

        assert py.ordered_inputs(2) == [1, 0]
        assert cpp.ordered_inputs(2) == [1, 0]

    def test_ordered_inputs_not_sorted(self) -> None:
        """ordered_inputs != sorted(in_neighbors) when insertion order differs from sort order."""
        py = LabeledDAG(10)
        cpp = NativeLabeledDAG(10)

        # Build 5 VAR nodes so that we can have higher-ID sources inserted first.
        for dag, is_cpp in [(py, False), (cpp, True)]:
            if is_cpp:
                for vi in range(5):
                    dag.add_node(NODE_TYPE_INT[NodeType.VAR], vi, float("nan"))
                dag.add_node(NODE_TYPE_INT[NodeType.ADD], -1, float("nan"))  # node 5
            else:
                for vi in range(5):
                    dag.add_node(NodeType.VAR, var_index=vi)
                dag.add_node(NodeType.ADD)

        # Insert in reverse order: 4, 3, 2.
        for src in [4, 3, 2]:
            py.add_edge(src, 5)
            cpp.add_edge(src, 5)

        py_io = py.ordered_inputs(5)
        cpp_io = cpp.ordered_inputs(5)

        assert py_io == [4, 3, 2], f"py_io={py_io}"
        assert cpp_io == [4, 3, 2], f"cpp_io={cpp_io}"
        # ordered_inputs != sorted(in_neighbors)
        assert py_io != sorted(py.in_neighbors(5))
        assert cpp_io != sorted(cpp.in_neighbors(5))
        assert py_io == cpp_io


# ---------------------------------------------------------------------------
# Invariant 9: normalize_const_creation
# ---------------------------------------------------------------------------


class TestInvariant9NormalizeConstCreation:
    """normalize_const_creation() moves all CONST creation edges to node 0.

    Iterates const_nodes in sorted order (Invariant 9).
    """

    def test_const_creation_edge_moved_to_node0(self) -> None:
        """CONST edge created from node 1 (y) is moved to node 0 (x) after normalization."""
        py = LabeledDAG(10)
        cpp = NativeLabeledDAG(10)

        for dag, is_cpp in [(py, False), (cpp, True)]:
            if is_cpp:
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 0, float("nan"))  # 0 = x
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 1, float("nan"))  # 1 = y
                dag.add_node(NODE_TYPE_INT[NodeType.CONST], -1, 3.14)  # 2 = k
                dag.add_node(NODE_TYPE_INT[NodeType.ADD], -1, float("nan"))  # 3 = add
            else:
                dag.add_node(NodeType.VAR, var_index=0)
                dag.add_node(NodeType.VAR, var_index=1)
                dag.add_node(NodeType.CONST, const_value=3.14)
                dag.add_node(NodeType.ADD)

        # CONST creation edge from node 1 (y), NOT node 0 (x).
        py.add_edge(1, 2)  # y → CONST (creation from y)
        cpp.add_edge(1, 2)
        py.add_edge(0, 3)  # x → ADD
        cpp.add_edge(0, 3)
        py.add_edge(2, 3)  # CONST → ADD
        cpp.add_edge(2, 3)

        py_norm = py.normalize_const_creation()
        cpp_norm = cpp.normalize_const_creation()

        # In the normalized DAG, CONST in-neighbor must be 0 (x), not 1 (y).
        py_const_in = py_norm.in_neighbors(2)
        cpp_const_in = cpp_norm.in_neighbors(2)

        assert 0 in py_const_in, f"Expected node 0 in py in_neighbors(2), got {py_const_in}"
        assert 1 not in py_const_in
        assert 0 in cpp_const_in, f"Expected node 0 in cpp in_neighbors(2), got {cpp_const_in}"
        assert 1 not in cpp_const_in

    def test_multiple_const_nodes_normalized_sorted(self) -> None:
        """Multiple CONST nodes: creation edges added to node 0 in sorted order."""
        py = LabeledDAG(10)
        cpp = NativeLabeledDAG(10)

        for dag, is_cpp in [(py, False), (cpp, True)]:
            if is_cpp:
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 0, float("nan"))  # 0
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 1, float("nan"))  # 1
                dag.add_node(NODE_TYPE_INT[NodeType.ADD], -1, float("nan"))  # 2
                dag.add_node(NODE_TYPE_INT[NodeType.CONST], -1, 1.0)  # 3 = k1
                dag.add_node(NODE_TYPE_INT[NodeType.CONST], -1, 2.0)  # 4 = k2
            else:
                dag.add_node(NodeType.VAR, var_index=0)
                dag.add_node(NodeType.VAR, var_index=1)
                dag.add_node(NodeType.ADD)
                dag.add_node(NodeType.CONST, const_value=1.0)
                dag.add_node(NodeType.CONST, const_value=2.0)

        # Creation edges from different parents.
        py.add_edge(1, 3)  # y → CONST k1 (node 3)
        cpp.add_edge(1, 3)
        py.add_edge(1, 4)  # y → CONST k2 (node 4)
        cpp.add_edge(1, 4)
        py.add_edge(0, 2)  # x → ADD
        cpp.add_edge(0, 2)
        py.add_edge(3, 2)  # k1 → ADD
        cpp.add_edge(3, 2)
        py.add_edge(4, 2)  # k2 → ADD
        cpp.add_edge(4, 2)

        py_norm = py.normalize_const_creation()
        cpp_norm = cpp.normalize_const_creation()

        # Both CONSTs now have node 0 as their only in-neighbor.
        for c in [3, 4]:
            py_in = py_norm.in_neighbors(c)
            cpp_in = cpp_norm.in_neighbors(c)
            assert list(py_in) == [0], f"py: CONST {c} in_neighbors={py_in}"
            assert list(cpp_in) == [0], f"cpp: CONST {c} in_neighbors={cpp_in}"

        # Full state must agree.
        _compare_dags(py_norm, cpp_norm)

    def test_normalize_preserves_non_const_edges_and_order(self) -> None:
        """normalize_const_creation preserves all non-CONST edges and their order."""
        py = LabeledDAG(10)
        cpp = NativeLabeledDAG(10)

        for dag, is_cpp in [(py, False), (cpp, True)]:
            if is_cpp:
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 0, float("nan"))  # 0 x
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 1, float("nan"))  # 1 y
                dag.add_node(NODE_TYPE_INT[NodeType.CONST], -1, 7.0)  # 2 k
                dag.add_node(NODE_TYPE_INT[NodeType.SUB], -1, float("nan"))  # 3 sub
            else:
                dag.add_node(NodeType.VAR, var_index=0)
                dag.add_node(NodeType.VAR, var_index=1)
                dag.add_node(NodeType.CONST, const_value=7.0)
                dag.add_node(NodeType.SUB)

        # sub(x, k): x is first operand, k is second.
        py.add_edge(0, 3)  # x → sub (first)
        cpp.add_edge(0, 3)
        py.add_edge(1, 2)  # y → CONST k (creation edge, will be moved)
        cpp.add_edge(1, 2)
        py.add_edge(2, 3)  # k → sub (second)
        cpp.add_edge(2, 3)

        py_norm = py.normalize_const_creation()
        cpp_norm = cpp.normalize_const_creation()

        # Operand order for SUB must be preserved: [x=0, k=2].
        assert py_norm.ordered_inputs(3) == [0, 2]
        assert cpp_norm.ordered_inputs(3) == [0, 2]

        _compare_dags(py_norm, cpp_norm)

    def test_normalize_and_compare_full_state(self) -> None:
        """Normalizing both py and cpp gives identical full observable state."""
        rng = random.Random(12345)
        py = LabeledDAG(15)
        cpp = NativeLabeledDAG(15)

        # Build a random DAG with some CONST nodes.
        for _ in range(8):
            label = rng.choice(_ALL_LABELS)
            vi = rng.randint(0, 3) if label in _VAR_TYPES else None
            cval = rng.uniform(-5.0, 5.0) if label in _CONST_TYPES else None
            py.add_node(label, var_index=vi, const_value=cval)
            cpp.add_node(
                NODE_TYPE_INT[label],
                vi if vi is not None else -1,
                cval if cval is not None else float("nan"),
            )

        # Add some edges.
        nc = py.node_count
        for _ in range(12):
            s = rng.randint(0, nc - 1)
            t = rng.randint(0, nc - 1)
            py.add_edge(s, t)
            cpp.add_edge(s, t)

        # Normalize and compare.
        py_norm = py.normalize_const_creation()
        cpp_norm = cpp.normalize_const_creation()
        _compare_dags(py_norm, cpp_norm)


# ---------------------------------------------------------------------------
# Error-behaviour parity
# ---------------------------------------------------------------------------


class TestErrorBehaviourParity:
    """C++ must translate errors to the same Python exception types as the Python implementation."""

    def test_cdll_full_raises_runtime_error(self) -> None:
        """CircularDoublyLinkedList is full → RuntimeError in both implementations."""
        cap = 3
        py = CircularDoublyLinkedList(cap)
        cpp = NativeCDLL(cap)

        for i in range(cap):
            prev = 0 if i == 0 else i - 1
            py.insert_after(prev, i)
            cpp.insert_after(prev, i)

        with pytest.raises(RuntimeError):
            py.insert_after(0, 99)
        with pytest.raises(RuntimeError):
            cpp.insert_after(0, 99)

    def test_dag_invalid_node_raises_index_error(self) -> None:
        """Out-of-range node ID → IndexError in both implementations."""
        py = LabeledDAG(5)
        cpp = NativeLabeledDAG(5)

        # No nodes added yet — any node ID is invalid.
        with pytest.raises(IndexError):
            py.node_label(0)
        with pytest.raises(IndexError):
            cpp.node_label(0)

    def test_dag_full_raises_runtime_error(self) -> None:
        """max_nodes reached → RuntimeError in both implementations."""
        cap = 3
        py = LabeledDAG(cap)
        cpp = NativeLabeledDAG(cap)

        for vi in range(cap):
            py.add_node(NodeType.VAR, var_index=vi)
            cpp.add_node(NODE_TYPE_INT[NodeType.VAR], vi, float("nan"))

        with pytest.raises(RuntimeError):
            py.add_node(NodeType.VAR, var_index=cap)
        with pytest.raises(RuntimeError):
            cpp.add_node(NODE_TYPE_INT[NodeType.VAR], cap, float("nan"))

    def test_dag_cycle_silently_noop(self) -> None:
        """Cycle-creating edge is silently rejected (returns False) in both implementations."""
        py = LabeledDAG(5)
        cpp = NativeLabeledDAG(5)

        for dag, is_cpp in [(py, False), (cpp, True)]:
            if is_cpp:
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 0, float("nan"))  # 0
                dag.add_node(NODE_TYPE_INT[NodeType.SIN], -1, float("nan"))  # 1
                dag.add_node(NODE_TYPE_INT[NodeType.COS], -1, float("nan"))  # 2
            else:
                dag.add_node(NodeType.VAR, var_index=0)
                dag.add_node(NodeType.SIN)
                dag.add_node(NodeType.COS)

        # 0 → 1 → 2.
        py.add_edge(0, 1)
        py.add_edge(1, 2)
        cpp.add_edge(0, 1)
        cpp.add_edge(1, 2)

        # 2 → 0 would create 0→1→2→0: should be rejected (False, not an exception).
        py_r = py.add_edge(2, 0)
        cpp_r = cpp.add_edge(2, 0)
        assert py_r is False
        assert cpp_r is False

        # Self-loop also rejected.
        py_s = py.add_edge(1, 1)
        cpp_s = cpp.add_edge(1, 1)
        assert py_s is False
        assert cpp_s is False

    def test_dag_duplicate_edge_silently_noop(self) -> None:
        """Duplicate edge returns False without raising in both implementations."""
        py = LabeledDAG(5)
        cpp = NativeLabeledDAG(5)

        for dag, is_cpp in [(py, False), (cpp, True)]:
            if is_cpp:
                dag.add_node(NODE_TYPE_INT[NodeType.VAR], 0, float("nan"))
                dag.add_node(NODE_TYPE_INT[NodeType.SIN], -1, float("nan"))
            else:
                dag.add_node(NodeType.VAR, var_index=0)
                dag.add_node(NodeType.SIN)

        assert py.add_edge(0, 1) is True
        assert cpp.add_edge(0, 1) is True

        # Second attempt — duplicate.
        assert py.add_edge(0, 1) is False
        assert cpp.add_edge(0, 1) is False

        # Edge count must not increase.
        assert py.edge_count == 1
        assert cpp.edge_count == 1
