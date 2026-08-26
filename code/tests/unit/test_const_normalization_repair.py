"""Regression tests for the CONST creation-edge repair (T15, Critical Invariant 9).

Context
-------
``normalize_const_creation`` used to relocate *every* CONST in-edge onto node 0.
That had two defects, both confirmed on 2026-07-27 and written up in
``docs/md_files/changes/d2s_canonicalisation_failures.md``:

1. **Totality.** ``add_edge`` refuses a cycle-closing edge and returns ``False``.
   The relocation discarded that return value, so whenever a CONST could reach
   node 0 the original creation edge was deleted and the replacement refused,
   leaving the CONST with in-degree 0. D2S cannot materialise an unreachable
   node, so canonicalisation raised ``RuntimeError``. Six of 4,000 randomly
   generated DAGs hit this.
2. **Completeness.** The relocation was non-injective: two labeled DAGs that
   differ only in which node a CONST hangs off collapsed to one canonical
   string, contradicting the Complete Labeled-DAG Invariant theorem.

The repair adds ``x_i -> c`` only for CONST nodes with in-degree 0, choosing the
lowest-indexed variable that does not close a cycle, and never removes an edge.
On any DAG satisfying the Round-Trip Fidelity reachability hypothesis it is the
identity, which restores both properties.

These tests pin that behaviour for the Python reference and, when the extension
is present, for the C++ engine.
"""

from __future__ import annotations

from collections import deque

import pytest

from isalsr.core.canonical import fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.string_to_dag import StringToDAG

try:
    from isalsr.core import _native  # type: ignore[attr-defined]

    _HAS_NATIVE = True
except ImportError:  # pragma: no cover - exercised only without the extension
    _native = None  # type: ignore[assignment]
    _HAS_NATIVE = False

requires_native = pytest.mark.skipif(_HAS_NATIVE is False, reason="C++ extension not built")


# The six DAGs from T15 that used to make canonicalisation raise.
# Source strings and the generator seed are recorded in
# docs/md_files/changes/fig_d2s_failures_data.json.
T15_FAILURE_STRINGS = [
    "PpV+CWVrV*VkPVccvknvinv-vrc",
    "PVcVkNCcV-v*NNV+VsNV+VsP",
    "v/Vrv*pvkWnvcV*cvrPVkn",
    "vsPWV+vgVkWv-NViV/VcviNPvlVrWCPN",
    "cNPNVkNCVr",  # minimal counterexample: k=2, |E|=3
    "VrV+Pv/vgVkNv+v-vaNvav^Wv+pc",
]


def _reachable_from_vars(dag: LabeledDAG) -> set[int]:
    """Return every node reachable from some VAR node by directed edges."""
    queue = deque(i for i in range(dag.node_count) if dag.node_label(i) == NodeType.VAR)
    seen = set(queue)
    while queue:
        node = queue.popleft()
        for succ in dag.out_neighbors(node):
            if succ not in seen:
                seen.add(succ)
                queue.append(succ)
    return seen


def _satisfies_precondition(dag: LabeledDAG) -> bool:
    """Round-Trip Fidelity hypothesis: every non-VAR node reachable from a variable."""
    reachable = _reachable_from_vars(dag)
    return all(i in reachable for i in range(dag.node_count) if dag.node_label(i) != NodeType.VAR)


def _edge_multiset(dag: LabeledDAG) -> list[tuple[int, int]]:
    return sorted((src, tgt) for tgt in range(dag.node_count) for src in dag.in_neighbors(tgt))


def _to_native(dag: LabeledDAG):  # type: ignore[no-untyped-def]
    """Copy a Python LabeledDAG into a NativeLabeledDAG, preserving operand order."""
    label_to_int = {label: label_int for label_int, label in enumerate(NodeType)}
    native = _native.testing.NativeLabeledDAG(max(dag.node_count, 1))
    for i in range(dag.node_count):
        data = dag.node_data(i)
        native.add_node(
            label_to_int[dag.node_label(i)],
            int(data["var_index"]) if "var_index" in data else -1,
            float(data["const_value"]) if "const_value" in data else float("nan"),
        )
    for tgt in range(dag.node_count):
        for src in dag.ordered_inputs(tgt):
            native.add_edge_unchecked(src, tgt)
    return native


# ---------------------------------------------------------------------------
# Totality: the six T15 cases now canonicalise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", T15_FAILURE_STRINGS)
def test_t15_case_canonicalises(source: str) -> None:
    """Each of the six formerly-failing DAGs now yields a canonical string."""
    dag = StringToDAG(source, num_variables=2).run()
    result = fast_canonical_string(dag, timeout=10.0)
    assert isinstance(result, str)
    assert result != ""


@pytest.mark.parametrize("source", T15_FAILURE_STRINGS)
def test_t15_case_repair_preserves_every_edge(source: str) -> None:
    """The repair must not drop an edge — that was the root cause of the failure."""
    dag = StringToDAG(source, num_variables=2).run()
    repaired = dag.normalize_const_creation()
    assert _edge_multiset(repaired) == _edge_multiset(dag)
    assert repaired.edge_count == dag.edge_count


@pytest.mark.parametrize("source", T15_FAILURE_STRINGS)
def test_t15_case_precondition_survives_repair(source: str) -> None:
    """Reachability holds before the repair and must still hold after it.

    Under the old relocation this was exactly what broke: the input satisfied the
    theorem's hypothesis, and normalisation destroyed it.
    """
    dag = StringToDAG(source, num_variables=2).run()
    assert _satisfies_precondition(dag)
    assert _satisfies_precondition(dag.normalize_const_creation())


@requires_native
@pytest.mark.parametrize("source", T15_FAILURE_STRINGS)
def test_t15_case_engines_agree(source: str) -> None:
    """Python and C++ must produce the same canonical string on the six cases."""
    dag = StringToDAG(source, num_variables=2).run()
    py = fast_canonical_string(dag, timeout=10.0, backend="python")
    cpp = fast_canonical_string(dag, timeout=10.0, backend="cpp")
    assert py == cpp


# ---------------------------------------------------------------------------
# Identity on precondition-satisfying DAGs — the property the theorem rests on
# ---------------------------------------------------------------------------


def _build_const_child(const_parent: int) -> LabeledDAG:
    """sin(x1) + x2 with a CONST leaf hanging off *const_parent*."""
    dag = LabeledDAG(16)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.VAR, var_index=1)  # 1
    dag.add_node(NodeType.SIN)  # 2
    dag.add_node(NodeType.CONST, const_value=1.0)  # 3
    dag.add_node(NodeType.ADD)  # 4
    dag.add_edge(0, 2)
    dag.add_edge(const_parent, 3)
    dag.add_edge(2, 4)
    dag.add_edge(1, 4)
    return dag


def test_repair_is_identity_when_precondition_holds() -> None:
    """A CONST that already has an in-edge is left exactly where it is."""
    dag = _build_const_child(const_parent=2)
    assert _satisfies_precondition(dag)
    repaired = dag.normalize_const_creation()
    assert _edge_multiset(repaired) == _edge_multiset(dag)
    for i in range(dag.node_count):
        assert repaired.node_label(i) == dag.node_label(i)
        assert repaired.ordered_inputs(i) == dag.ordered_inputs(i)


def test_completeness_distinct_const_provenance_distinct_strings() -> None:
    """Two non-isomorphic DAGs differing only in CONST provenance must not merge.

    Under the old relocation both collapsed to ``VkVspv+NnC``, which refutes the
    Complete Labeled-DAG Invariant theorem: the DAGs are not isomorphic (variable
    anchoring forces phi(x1)=x1, and x1 has out-degree 2 in one and 1 in the other).
    """
    from_x1 = _build_const_child(const_parent=0)
    from_sin = _build_const_child(const_parent=2)

    assert not from_x1.is_isomorphic(from_sin)
    assert fast_canonical_string(from_x1) != fast_canonical_string(from_sin)


@requires_native
def test_completeness_distinct_const_provenance_native() -> None:
    """Same completeness property through the C++ engine."""
    from_x1 = _build_const_child(const_parent=0)
    from_sin = _build_const_child(const_parent=2)
    a = fast_canonical_string(from_x1, backend="cpp")
    b = fast_canonical_string(from_sin, backend="cpp")
    assert a != b


# ---------------------------------------------------------------------------
# Orphan CONST nodes still get a creation edge — Invariant 9's actual purpose
# ---------------------------------------------------------------------------


def test_orphan_const_receives_creation_edge() -> None:
    # CONTRACT CHANGED 2026-07-29: 𝒩 removed from canonicaliser.
    # The method still repairs the orphan (producer-side responsibility of adapters).
    # New division of responsibility: the *unrepaired* DAG raises at canonicaliser
    # (loud refusal); 𝒩(D) satisfies the precondition and canonicalises successfully.
    """𝒩 repairs orphan CONST; unrepaired DAG raises, 𝒩(D) canonicalises successfully."""
    dag = LabeledDAG(8)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.CONST, const_value=2.0)  # 1
    dag.add_node(NodeType.ADD)  # 2
    dag.add_edge(0, 2)
    dag.add_edge(1, 2)
    assert dag.in_degree(1) == 0
    assert not _satisfies_precondition(dag)

    repaired = dag.normalize_const_creation()
    assert repaired.in_degree(1) == 1
    assert repaired.has_edge(0, 1)
    assert _satisfies_precondition(repaired)

    # Unrepaired DAG is now refused loudly (not silently canonicalised via 𝒩).
    with pytest.raises(RuntimeError):
        fast_canonical_string(dag, timeout=5.0)

    # 𝒩(D) satisfies the precondition and canonicalises without error.
    cs = fast_canonical_string(repaired, timeout=5.0)
    assert isinstance(cs, str) and len(cs) > 0


def test_orphan_const_anchor_skips_cycle_closing_variable() -> None:
    """When x1 would close a cycle, the repair falls through to the next variable."""
    dag = LabeledDAG(8)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.VAR, var_index=1)  # 1
    dag.add_node(NodeType.CONST, const_value=3.0)  # 2
    dag.add_edge(2, 0)  # CONST reaches x1, so x1 -> CONST would be a cycle
    assert dag.in_degree(2) == 0

    repaired = dag.normalize_const_creation()
    assert not repaired.has_edge(0, 2)
    assert repaired.has_edge(1, 2)
    assert repaired.has_edge(2, 0)


@requires_native
def test_repair_engines_agree_on_structure() -> None:
    """Python and C++ repairs must produce identical edge sets and operand order."""
    cases = [
        _build_const_child(const_parent=0),
        _build_const_child(const_parent=2),
        StringToDAG("cNPNVkNCVr", num_variables=2).run(),
    ]
    for dag in cases:
        py = dag.normalize_const_creation()
        cpp = _to_native(dag).normalize_const_creation()
        assert cpp.node_count == py.node_count
        assert cpp.edge_count == py.edge_count
        for i in range(py.node_count):
            assert list(cpp.ordered_inputs(i)) == list(py.ordered_inputs(i))
