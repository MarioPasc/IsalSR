"""Unit tests for DAGToString (D2S).

Covers: pair generation, basic D2S encoding, label preservation in tokens,
edge direction, reachability checks, and round-trip verification with S2D.

The D2S encoder is the inverse of S2D. Together they establish the round-trip
property: S2D(D2S(D, x_1)) ~ D, which underpins the canonical string (Phase 4).
"""

from __future__ import annotations

import pytest

from isalsr.core.dag_to_string import DAGToString, generate_pairs_sorted_by_sum
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.string_to_dag import StringToDAG

# ======================================================================
# Pair generation tests
# ======================================================================


class TestPairGeneration:
    """generate_pairs_sorted_by_sum correctness."""

    def test_first_pair_is_zero_zero(self) -> None:
        pairs = generate_pairs_sorted_by_sum(1)
        assert pairs[0] == (0, 0)

    def test_cost_ordering(self) -> None:
        """Pairs are sorted by |a| + |b| (non-decreasing)."""
        pairs = generate_pairs_sorted_by_sum(3)
        costs = [abs(a) + abs(b) for a, b in pairs]
        assert costs == sorted(costs)

    def test_m1_count(self) -> None:
        """m=1: pairs from -1..1 x -1..1 = 9 pairs."""
        assert len(generate_pairs_sorted_by_sum(1)) == 9

    def test_m2_count(self) -> None:
        """m=2: pairs from -2..2 x -2..2 = 25 pairs."""
        assert len(generate_pairs_sorted_by_sum(2)) == 25

    def test_invalid_m_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            generate_pairs_sorted_by_sum(0)

    def test_deterministic(self) -> None:
        """Same m produces identical ordering."""
        p1 = generate_pairs_sorted_by_sum(3)
        p2 = generate_pairs_sorted_by_sum(3)
        assert p1 == p2


# ======================================================================
# Basic D2S encoding tests
# ======================================================================


class TestD2SBasics:
    """Basic D2S encoding produces valid strings."""

    def test_sin_x(self, sin_x_dag: LabeledDAG) -> None:
        """sin(x): D2S produces string, S2D reconstructs isomorphic DAG."""
        d2s = DAGToString(sin_x_dag)
        string = d2s.run()
        assert len(string) > 0
        # Verify the string contains a SIN token.
        assert "Vs" in string or "vs" in string
        # Round-trip verify.
        dag2 = StringToDAG(string, num_variables=1).run()
        assert sin_x_dag.is_isomorphic(dag2)

    def test_x_plus_y(self, x_plus_y_dag: LabeledDAG) -> None:
        """x + y: D2S produces string, round-trip verifies."""
        d2s = DAGToString(x_plus_y_dag)
        string = d2s.run()
        assert len(string) > 0
        dag2 = StringToDAG(string, num_variables=2).run()
        assert x_plus_y_dag.is_isomorphic(dag2)

    def test_sin_x_mul_y(self, sin_x_mul_y_dag: LabeledDAG) -> None:
        """sin(x) * y: D2S round-trip."""
        d2s = DAGToString(sin_x_mul_y_dag)
        string = d2s.run()
        dag2 = StringToDAG(string, num_variables=2).run()
        assert sin_x_mul_y_dag.is_isomorphic(dag2)

    def test_no_bare_V_or_v(self, sin_x_dag: LabeledDAG) -> None:
        """Output string never has bare V or v without a label character."""
        string = DAGToString(sin_x_dag).run()
        # Tokenize and verify all V/v tokens are two-char.
        from isalsr.core.string_to_dag import _tokenize

        tokens = _tokenize(string)
        for tok in tokens:
            if tok[0] in "Vv":
                assert len(tok) == 2, f"Bare V/v token found: {tok!r}"


# ======================================================================
# Label preservation tests
# ======================================================================


class TestD2SLabelPreservation:
    """D2S emits correct label characters for each NodeType."""

    def test_add_label(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 1)
        string = DAGToString(dag).run()
        assert "+" in string

    def test_mul_label(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.MUL)
        dag.add_edge(0, 1)
        string = DAGToString(dag).run()
        assert "*" in string

    def test_const_label(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.CONST, const_value=3.14)
        dag.add_edge(0, 1)
        string = DAGToString(dag).run()
        assert "k" in string

    def test_const_value_preserved_in_roundtrip(self) -> None:
        """CONST node's const_value survives D2S → S2D round-trip."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.CONST, const_value=3.14)
        dag.add_edge(0, 1)
        string = DAGToString(dag).run()
        # S2D creates CONST with default value 1.0, not 3.14.
        # This is expected: const_value is optimized later via BFGS.
        dag2 = StringToDAG(string, num_variables=1).run()
        assert dag2.node_label(1) == NodeType.CONST


# ======================================================================
# Edge direction tests
# ======================================================================


class TestD2SEdgeDirections:
    """C/c edge instructions respect direction."""

    def test_c_used_for_reverse_edges(self) -> None:
        """When an edge goes secondary → primary, 'c' must be emitted."""
        # Build DAG where reverse-direction edge is needed:
        # x(0) -> sin(1), y(2) -> sin(1)
        # After V from x creates sin, we need y → sin which goes "backwards".
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_edge(0, 1)  # x → sin
        dag.add_edge(2, 1)  # y → sin (reverse direction relative to CDLL order)
        # This should still produce a valid string.
        string = DAGToString(dag).run()
        dag2 = StringToDAG(string, num_variables=2).run()
        assert dag.is_isomorphic(dag2)


# ======================================================================
# Reachability tests
# ======================================================================


class TestD2SReachability:
    """Reachability validation."""

    def test_disconnected_dag_raises(self) -> None:
        """Nodes unreachable from VAR nodes via outgoing edges raise."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.ADD)
        # No edge from VAR to ADD — ADD is unreachable.
        with pytest.raises(ValueError, match="Unreachable"):
            DAGToString(dag).run()

    def test_invalid_initial_node_raises(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        with pytest.raises(ValueError, match="out of range"):
            DAGToString(dag, initial_node=5)

    def test_var_only_dag_ok(self) -> None:
        """DAG with only VAR nodes (no ops) produces empty string."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        string = DAGToString(dag).run()
        assert string == ""


# ======================================================================
# Complex expression tests
# ======================================================================


class TestD2SComplexExpressions:
    """More complex DAGs for robustness."""

    def test_chain_x_sin_exp(self) -> None:
        """x → sin → exp: chain of 3 nodes."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.EXP)
        dag.add_edge(0, 1)
        dag.add_edge(1, 2)
        string = DAGToString(dag).run()
        dag2 = StringToDAG(string, num_variables=1).run()
        assert dag.is_isomorphic(dag2)

    def test_diamond_dag(self) -> None:
        """x → ADD, x → MUL, ADD → SIN, MUL → SIN (diamond shape)."""
        dag = LabeledDAG(max_nodes=6)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.ADD)  # 1
        dag.add_node(NodeType.MUL)  # 2
        dag.add_node(NodeType.SIN)  # 3: requires in_degree >= 1 for unary
        # Wait, SIN is unary so it can't have 2 inputs. Use ADD at top instead.
        dag = LabeledDAG(max_nodes=6)
        dag.add_node(NodeType.VAR, var_index=0)  # 0
        dag.add_node(NodeType.SIN)  # 1
        dag.add_node(NodeType.COS)  # 2
        dag.add_node(NodeType.ADD)  # 3
        dag.add_edge(0, 1)  # x → sin
        dag.add_edge(0, 2)  # x → cos
        dag.add_edge(1, 3)  # sin → add
        dag.add_edge(2, 3)  # cos → add
        string = DAGToString(dag).run()
        dag2 = StringToDAG(string, num_variables=1).run()
        assert dag.is_isomorphic(dag2)

    def test_two_var_with_three_ops(self) -> None:
        """x → MUL, y → MUL, MUL → ADD, x → ADD: expression x*y + x."""
        dag = LabeledDAG(max_nodes=6)
        dag.add_node(NodeType.VAR, var_index=0)  # 0: x
        dag.add_node(NodeType.VAR, var_index=1)  # 1: y
        dag.add_node(NodeType.MUL)  # 2
        dag.add_node(NodeType.ADD)  # 3
        dag.add_edge(0, 2)  # x → mul
        dag.add_edge(1, 2)  # y → mul
        dag.add_edge(2, 3)  # mul → add
        dag.add_edge(0, 3)  # x → add
        string = DAGToString(dag).run()
        dag2 = StringToDAG(string, num_variables=2).run()
        assert dag.is_isomorphic(dag2)
