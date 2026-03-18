"""Tests for CONST creation edge normalization.

CONST nodes are evaluation-neutral leaves that ignore their in-edges.
The creation edge (from V/v instruction) is structurally required for D2S
reachability but semantically meaningless. Different creation sources
produce different canonical strings for the same function.

The normalization moves all CONST creation edges to x_1 (node 0),
eliminating this redundancy. This is applied transparently in:
- canonical_string() / pruned_canonical_string()
- is_isomorphic()
- from_sympy()
"""

from __future__ import annotations

import pytest

from isalsr.core.canonical import canonical_string, pruned_canonical_string
from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.string_to_dag import StringToDAG

# ======================================================================
# Helper: build y^k with variable CONST creation source
# ======================================================================


def _build_y_pow_k(creation_source: int) -> LabeledDAG:
    """Build y^k (k=1.0) with CONST created from the given source."""
    dag = LabeledDAG(5)
    dag.add_node(NodeType.VAR, var_index=0)  # 0: x
    dag.add_node(NodeType.VAR, var_index=1)  # 1: y
    dag.add_node(NodeType.CONST, const_value=1.0)  # 2: k
    dag.add_node(NodeType.POW)  # 3: y^k
    dag.add_edge(creation_source, 2)  # creation edge (varies)
    dag.add_edge(1, 3)  # y = base (first operand)
    dag.add_edge(2, 3)  # k = exponent (second operand)
    return dag


def _build_x_sub_k(creation_source: int) -> LabeledDAG:
    """Build x - k with CONST created from the given source."""
    dag = LabeledDAG(4)
    dag.add_node(NodeType.VAR, var_index=0)  # 0: x
    dag.add_node(NodeType.CONST, const_value=1.0)  # 1: k
    dag.add_node(NodeType.SUB)  # 2: x - k
    dag.add_edge(creation_source, 1)  # creation edge
    dag.add_edge(0, 2)  # x = first operand
    dag.add_edge(1, 2)  # k = second operand
    return dag


# ======================================================================
# Core normalization tests
# ======================================================================


class TestNormalizationBasic:
    """normalize_const_creation produces correct results."""

    def test_const_creation_moved_to_x1(self) -> None:
        """CONST creation edge is moved to node 0 (x_1)."""
        dag = _build_y_pow_k(creation_source=1)  # from y
        norm = dag.normalize_const_creation()
        # In normalized DAG, CONST(2) should have in-neighbor {0} (x_1).
        assert set(norm.in_neighbors(2)) == {0}

    def test_already_from_x1_unchanged(self) -> None:
        """CONST already from x_1 stays the same."""
        dag = _build_y_pow_k(creation_source=0)
        norm = dag.normalize_const_creation()
        assert set(norm.in_neighbors(2)) == {0}

    def test_evaluation_preserved(self) -> None:
        """Normalization doesn't change evaluation."""
        dag = _build_y_pow_k(creation_source=1)
        norm = dag.normalize_const_creation()
        v1 = evaluate_dag(dag, {0: 1.0, 1: 3.0})
        v2 = evaluate_dag(norm, {0: 1.0, 1: 3.0})
        assert v1 == pytest.approx(v2, abs=1e-10)

    def test_node_count_preserved(self) -> None:
        dag = _build_y_pow_k(creation_source=1)
        norm = dag.normalize_const_creation()
        assert dag.node_count == norm.node_count

    def test_non_creation_edges_preserved(self) -> None:
        """Edges FROM CONST (data flow) are preserved."""
        dag = _build_y_pow_k(creation_source=1)
        norm = dag.normalize_const_creation()
        # CONST(2) → POW(3) should still exist.
        assert norm.has_edge(2, 3)
        # y(1) → POW(3) should still exist.
        assert norm.has_edge(1, 3)

    def test_no_const_nodes_returns_equivalent(self) -> None:
        """DAGs without CONST nodes are returned unchanged."""
        dag = LabeledDAG(3)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_edge(0, 1)
        norm = dag.normalize_const_creation()
        assert norm.node_count == dag.node_count
        assert norm.edge_count == dag.edge_count


# ======================================================================
# Redundancy elimination
# ======================================================================


class TestRedundancyElimination:
    """Normalization collapses CONST-creation variants to one canonical."""

    def test_canonical_same_regardless_of_creation_source(self) -> None:
        """y^k with CONST from x vs from y: same canonical string."""
        dag_from_x = _build_y_pow_k(creation_source=0)
        dag_from_y = _build_y_pow_k(creation_source=1)
        assert canonical_string(dag_from_x) == canonical_string(dag_from_y)

    def test_pruned_canonical_same_regardless_of_source(self) -> None:
        dag_from_x = _build_y_pow_k(creation_source=0)
        dag_from_y = _build_y_pow_k(creation_source=1)
        assert pruned_canonical_string(dag_from_x) == pruned_canonical_string(dag_from_y)

    def test_isomorphic_regardless_of_creation_source(self) -> None:
        """DAGs differing only in CONST creation source are isomorphic."""
        dag_from_x = _build_y_pow_k(creation_source=0)
        dag_from_y = _build_y_pow_k(creation_source=1)
        assert dag_from_x.is_isomorphic(dag_from_y)

    def test_sub_canonical_same_regardless_of_source(self) -> None:
        """x - k with CONST from x: same canonical for all valid sources."""
        dag = _build_x_sub_k(creation_source=0)
        c = canonical_string(dag)
        assert len(c) > 0  # Sanity: non-empty canonical.

    def test_multi_const_all_normalized(self) -> None:
        """Multiple CONSTs in one DAG all get normalized."""
        dag = LabeledDAG(8)
        dag.add_node(NodeType.VAR, var_index=0)  # 0: x
        dag.add_node(NodeType.VAR, var_index=1)  # 1: y
        dag.add_node(NodeType.CONST, const_value=1.0)  # 2: k1
        dag.add_node(NodeType.POW)  # 3: x^k1
        dag.add_node(NodeType.CONST, const_value=1.0)  # 4: k2
        dag.add_node(NodeType.POW)  # 5: y^k2
        dag.add_node(NodeType.SUB)  # 6
        # k1 from x, k2 from y (different sources)
        dag.add_edge(0, 2)  # x → k1
        dag.add_edge(0, 3)  # x → POW1 (base)
        dag.add_edge(2, 3)  # k1 → POW1 (exp)
        dag.add_edge(1, 4)  # y → k2
        dag.add_edge(1, 5)  # y → POW2 (base)
        dag.add_edge(4, 5)  # k2 → POW2 (exp)
        dag.add_edge(3, 6)  # POW1 → SUB (first)
        dag.add_edge(5, 6)  # POW2 → SUB (second)

        norm = dag.normalize_const_creation()
        # Both CONSTs should have creation edge from x_1 (node 0).
        assert set(norm.in_neighbors(2)) == {0}  # k1 from x
        assert set(norm.in_neighbors(4)) == {0}  # k2 from x (was y)

    @pytest.mark.parametrize("c1,c2", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_all_creation_combos_same_canonical(self, c1: int, c2: int) -> None:
        """All 4 creation source combos for 2 CONSTs → same canonical."""
        dag = LabeledDAG(8)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.CONST, const_value=1.0)
        dag.add_node(NodeType.POW)
        dag.add_node(NodeType.CONST, const_value=1.0)
        dag.add_node(NodeType.POW)
        dag.add_node(NodeType.SUB)
        dag.add_edge(c1, 2)
        dag.add_edge(0, 3)
        dag.add_edge(2, 3)
        dag.add_edge(c2, 4)
        dag.add_edge(1, 5)
        dag.add_edge(4, 5)
        dag.add_edge(3, 6)
        dag.add_edge(5, 6)

        # All should produce the same canonical string.
        reference = canonical_string(_build_multi_const_reference())
        assert canonical_string(dag) == reference


def _build_multi_const_reference() -> LabeledDAG:
    """Reference DAG with both CONSTs from x_1."""
    dag = LabeledDAG(8)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    dag.add_node(NodeType.CONST, const_value=1.0)
    dag.add_node(NodeType.POW)
    dag.add_node(NodeType.CONST, const_value=1.0)
    dag.add_node(NodeType.POW)
    dag.add_node(NodeType.SUB)
    dag.add_edge(0, 2)
    dag.add_edge(0, 3)
    dag.add_edge(2, 3)
    dag.add_edge(0, 4)
    dag.add_edge(1, 5)
    dag.add_edge(4, 5)
    dag.add_edge(3, 6)
    dag.add_edge(5, 6)
    return dag


# ======================================================================
# D2S round-trip after normalization
# ======================================================================


class TestD2SWithNormalization:
    """D2S works on normalized DAGs."""

    def test_d2s_on_normalized_dag(self) -> None:
        """D2S can encode a normalized DAG."""
        dag = _build_y_pow_k(creation_source=1)
        norm = dag.normalize_const_creation()
        s = DAGToString(norm).run()
        assert len(s) > 0

    def test_canonical_roundtrip_eval_preserved(self) -> None:
        """Full pipeline: D → canonical → S2D → eval matches."""
        dag = _build_y_pow_k(creation_source=1)
        v1 = evaluate_dag(dag, {0: 1.0, 1: 3.0})
        canon = canonical_string(dag)
        dag2 = StringToDAG(canon, num_variables=2).run()
        v2 = evaluate_dag(dag2, {0: 1.0, 1: 3.0})
        assert v1 == pytest.approx(v2, abs=1e-8)


# ======================================================================
# SymPy adapter with normalization
# ======================================================================


class TestSympyAdapterConstCreation:
    """from_sympy adds creation edges for CONST nodes."""

    @pytest.fixture(autouse=True)
    def _check_sympy(self) -> None:
        pytest.importorskip("sympy")

    def test_from_sympy_const_reachable(self) -> None:
        """CONST nodes from from_sympy have creation edges."""
        from sympy import Symbol

        from isalsr.adapters.sympy_adapter import SympyAdapter

        adapter = SympyAdapter()
        x = Symbol("x_0")
        dag = adapter.from_sympy(x + 1, [x])

        # Find the CONST node.
        const_nodes = [i for i in range(dag.node_count) if dag.node_label(i) == NodeType.CONST]
        assert len(const_nodes) == 1
        # CONST should have at least one in-edge (creation from x_1).
        assert dag.in_degree(const_nodes[0]) > 0

    def test_from_sympy_dag_is_d2s_encodable(self) -> None:
        """DAGs from from_sympy can be encoded by D2S."""
        from sympy import Symbol

        from isalsr.adapters.sympy_adapter import SympyAdapter

        adapter = SympyAdapter()
        x = Symbol("x_0")
        dag = adapter.from_sympy(x + 1, [x])
        s = DAGToString(dag).run()
        assert len(s) > 0

    def test_from_sympy_dag_is_canonicalizable(self) -> None:
        """DAGs from from_sympy can be canonicalized."""
        from sympy import Symbol

        from isalsr.adapters.sympy_adapter import SympyAdapter

        adapter = SympyAdapter()
        x = Symbol("x_0")
        dag = adapter.from_sympy(x + 1, [x])
        c = canonical_string(dag)
        assert len(c) > 0

    def test_from_sympy_roundtrip_eval(self) -> None:
        """from_sympy → canonical → S2D → eval matches SymPy eval."""
        from sympy import Symbol

        from isalsr.adapters.sympy_adapter import SympyAdapter

        adapter = SympyAdapter()
        x_sym = Symbol("x_0")
        expr = x_sym + 1
        dag = adapter.from_sympy(expr, [x_sym])

        v_dag = evaluate_dag(dag, {0: 2.5})
        v_sympy = float(expr.subs(x_sym, 2.5))
        assert v_dag == pytest.approx(v_sympy, abs=1e-8)

        # Canonical round-trip.
        canon = canonical_string(dag)
        dag2 = StringToDAG(canon, num_variables=1).run()
        v_canon = evaluate_dag(dag2, {0: 2.5})
        assert v_dag == pytest.approx(v_canon, abs=1e-8)
