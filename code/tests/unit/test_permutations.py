"""Unit tests for isalsr.core.permutations."""

from __future__ import annotations

import itertools
import random

import pytest

from isalsr.core.canonical import pruned_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.permutations import permute_internal_nodes, random_permutations

# ---- Fixtures ----


def _build_sin_x_plus_cos_x() -> LabeledDAG:
    """sin(x) + cos(x): m=1, k=3 (SIN, COS, ADD)."""
    dag = LabeledDAG(max_nodes=4)
    dag.add_node(NodeType.VAR, var_index=0)  # 0: x
    dag.add_node(NodeType.SIN)  # 1: sin
    dag.add_node(NodeType.COS)  # 2: cos
    dag.add_node(NodeType.ADD)  # 3: add
    dag.add_edge(0, 1)  # x -> sin
    dag.add_edge(0, 2)  # x -> cos
    dag.add_edge(1, 3)  # sin -> add
    dag.add_edge(2, 3)  # cos -> add
    return dag


def _build_x_pow_y() -> LabeledDAG:
    """x^y: m=2, k=1 (POW). Tests operand order preservation."""
    dag = LabeledDAG(max_nodes=3)
    dag.add_node(NodeType.VAR, var_index=0)  # 0: x (base)
    dag.add_node(NodeType.VAR, var_index=1)  # 1: y (exponent)
    dag.add_node(NodeType.POW)  # 2: pow
    dag.add_edge(0, 2)  # x -> pow (base = first operand)
    dag.add_edge(1, 2)  # y -> pow (exponent = second operand)
    return dag


def _build_chain_dag() -> LabeledDAG:
    """sin(cos(exp(x))): m=1, k=3. Linear chain, no automorphisms."""
    dag = LabeledDAG(max_nodes=4)
    dag.add_node(NodeType.VAR, var_index=0)  # 0: x
    dag.add_node(NodeType.EXP)  # 1: exp
    dag.add_node(NodeType.COS)  # 2: cos
    dag.add_node(NodeType.SIN)  # 3: sin
    dag.add_edge(0, 1)  # x -> exp
    dag.add_edge(1, 2)  # exp -> cos
    dag.add_edge(2, 3)  # cos -> sin
    return dag


# ---- Tests ----


class TestIdentityPermutation:
    def test_identity_preserves_structure(self) -> None:
        dag = _build_sin_x_plus_cos_x()
        k = dag.node_count - len(dag.var_nodes())
        perm_dag = permute_internal_nodes(dag, list(range(k)))
        assert perm_dag.node_count == dag.node_count
        assert perm_dag.edge_count == dag.edge_count
        for i in range(dag.node_count):
            assert perm_dag.node_label(i) == dag.node_label(i)

    def test_identity_edges_match(self) -> None:
        dag = _build_chain_dag()
        k = dag.node_count - len(dag.var_nodes())
        perm_dag = permute_internal_nodes(dag, list(range(k)))
        for v in range(dag.node_count):
            assert perm_dag.in_neighbors(v) == dag.in_neighbors(v)
            assert perm_dag.out_neighbors(v) == dag.out_neighbors(v)


class TestIsomorphismHolds:
    def test_all_permutations_isomorphic_k3(self) -> None:
        dag = _build_sin_x_plus_cos_x()
        m = len(dag.var_nodes())
        k = dag.node_count - m
        for perm in itertools.permutations(range(k)):
            perm_dag = permute_internal_nodes(dag, list(perm))
            assert dag.is_isomorphic(perm_dag), f"Permutation {perm} produced non-isomorphic DAG"

    def test_all_permutations_isomorphic_chain(self) -> None:
        dag = _build_chain_dag()
        m = len(dag.var_nodes())
        k = dag.node_count - m
        for perm in itertools.permutations(range(k)):
            perm_dag = permute_internal_nodes(dag, list(perm))
            assert dag.is_isomorphic(perm_dag)


class TestLabelsPreserved:
    def test_labels_mapped_correctly(self) -> None:
        dag = _build_sin_x_plus_cos_x()
        m = len(dag.var_nodes())
        perm = [2, 0, 1]  # SIN(1)->3, COS(2)->1, ADD(3)->2
        perm_dag = permute_internal_nodes(dag, perm)

        # Node m+0=1 was SIN, mapped to m+perm[0]=m+2=3
        assert perm_dag.node_label(m + 2) == NodeType.SIN
        # Node m+1=2 was COS, mapped to m+perm[1]=m+0=1
        assert perm_dag.node_label(m + 0) == NodeType.COS
        # Node m+2=3 was ADD, mapped to m+perm[2]=m+1=2
        assert perm_dag.node_label(m + 1) == NodeType.ADD


class TestOperandOrderPreserved:
    def test_pow_operand_order(self) -> None:
        """x^y must preserve base=x, exponent=y after permutation."""
        dag = _build_x_pow_y()


        # Only permutation of 1 element: identity
        perm_dag = permute_internal_nodes(dag, [0])
        inputs = perm_dag.ordered_inputs(2)  # POW node
        assert inputs == [0, 1], f"Expected [0, 1] (x=base, y=exp), got {inputs}"

    def test_sub_operand_order(self) -> None:
        """x - y: SUB(x, y) preserves first=x, second=y."""
        dag = LabeledDAG(max_nodes=3)
        dag.add_node(NodeType.VAR, var_index=0)  # 0: x
        dag.add_node(NodeType.VAR, var_index=1)  # 1: y
        dag.add_node(NodeType.SUB)  # 2: sub
        dag.add_edge(0, 2)  # x -> sub (first operand)
        dag.add_edge(1, 2)  # y -> sub (second operand)

        perm_dag = permute_internal_nodes(dag, [0])
        inputs = perm_dag.ordered_inputs(2)
        assert inputs == [0, 1]


class TestVarNodesFixed:
    def test_variables_unchanged(self) -> None:
        dag = _build_x_pow_y()
        perm_dag = permute_internal_nodes(dag, [0])
        assert perm_dag.node_label(0) == NodeType.VAR
        assert perm_dag.node_label(1) == NodeType.VAR
        assert perm_dag.node_data(0).get("var_index") == 0
        assert perm_dag.node_data(1).get("var_index") == 1


class TestCanonicalInvariant:
    def test_all_permutations_same_canonical_k3(self) -> None:
        dag = _build_sin_x_plus_cos_x()
        m = len(dag.var_nodes())
        k = dag.node_count - m
        canon_original = pruned_canonical_string(dag, timeout=5.0)
        for perm in itertools.permutations(range(k)):
            perm_dag = permute_internal_nodes(dag, list(perm))
            canon_perm = pruned_canonical_string(perm_dag, timeout=5.0)
            assert canon_perm == canon_original, (
                f"Perm {perm}: got '{canon_perm}', expected '{canon_original}'"
            )

    def test_chain_canonical_invariant(self) -> None:
        dag = _build_chain_dag()
        m = len(dag.var_nodes())
        k = dag.node_count - m
        canon_original = pruned_canonical_string(dag, timeout=5.0)
        for perm in itertools.permutations(range(k)):
            perm_dag = permute_internal_nodes(dag, list(perm))
            canon_perm = pruned_canonical_string(perm_dag, timeout=5.0)
            assert canon_perm == canon_original


class TestInvalidPermutation:
    def test_wrong_length_raises(self) -> None:
        dag = _build_sin_x_plus_cos_x()
        with pytest.raises(ValueError, match="permutation"):
            permute_internal_nodes(dag, [0, 1])  # k=3 but perm has 2

    def test_duplicate_raises(self) -> None:
        dag = _build_sin_x_plus_cos_x()
        with pytest.raises(ValueError, match="permutation"):
            permute_internal_nodes(dag, [0, 0, 1])

    def test_out_of_range_raises(self) -> None:
        dag = _build_sin_x_plus_cos_x()
        with pytest.raises(ValueError, match="permutation"):
            permute_internal_nodes(dag, [0, 1, 5])


class TestRandomPermutations:
    def test_returns_valid_permutations(self) -> None:
        rng = random.Random(42)
        perms = random_permutations(5, 10, rng)
        assert len(perms) == 10
        for p in perms:
            assert sorted(p) == list(range(5))

    def test_deterministic_with_seed(self) -> None:
        p1 = random_permutations(5, 10, random.Random(42))
        p2 = random_permutations(5, 10, random.Random(42))
        assert p1 == p2
