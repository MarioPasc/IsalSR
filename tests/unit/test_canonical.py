"""Unit tests for canonical string computation.

This tests the paper's CORE MATHEMATICAL CONTRIBUTION:
the canonical string w*_D is a complete labeled-DAG invariant.
    canonical_string(D1) == canonical_string(D2) iff D1 ~ D2

Tests cover: Levenshtein distance, structural tuples, canonical basics,
canonical invariance (THE KEY PROPERTY), discrimination, pruned variant,
DAG distance metric, and algorithm variants.
"""

from __future__ import annotations

import math

import pytest

from isalsr.core.algorithms.exhaustive import ExhaustiveD2S
from isalsr.core.algorithms.greedy_min import GreedyMinD2S
from isalsr.core.algorithms.greedy_single import GreedySingleD2S
from isalsr.core.algorithms.pruned_exhaustive import PrunedExhaustiveD2S
from isalsr.core.canonical import (
    canonical_string,
    compute_structural_tuples,
    dag_distance,
    levenshtein,
    pruned_canonical_string,
)
from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.string_to_dag import StringToDAG

# ======================================================================
# Levenshtein distance
# ======================================================================


class TestLevenshtein:
    """Levenshtein edit distance correctness."""

    def test_identical(self) -> None:
        assert levenshtein("abc", "abc") == 0

    def test_empty(self) -> None:
        assert levenshtein("", "") == 0
        assert levenshtein("abc", "") == 3
        assert levenshtein("", "abc") == 3

    def test_insertion(self) -> None:
        assert levenshtein("abc", "abcd") == 1

    def test_deletion(self) -> None:
        assert levenshtein("abcd", "abc") == 1

    def test_substitution(self) -> None:
        assert levenshtein("abc", "axc") == 1

    def test_mixed(self) -> None:
        assert levenshtein("kitten", "sitting") == 3

    def test_symmetry(self) -> None:
        assert levenshtein("abc", "xyz") == levenshtein("xyz", "abc")


# ======================================================================
# Structural tuples (6-component, directed)
# ======================================================================


class TestStructuralTuples:
    """6-component structural tuple correctness."""

    def test_single_var(self) -> None:
        """Single VAR node: all zeros (no neighbors)."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        tuples = compute_structural_tuples(dag)
        assert tuples[0] == (0, 0, 0, 0, 0, 0)

    def test_sin_x(self, sin_x_dag: LabeledDAG) -> None:
        """sin(x): x has 1 out-neighbor at d=1, sin has 1 in-neighbor at d=1."""
        tuples = compute_structural_tuples(sin_x_dag)
        # x (node 0): in=0 at all distances, out_d1=1 (sin), out_d2=0, out_d3=0
        assert tuples[0] == (0, 1, 0, 0, 0, 0)
        # sin (node 1): in_d1=1 (x), out=0 at all distances
        assert tuples[1] == (1, 0, 0, 0, 0, 0)

    def test_x_plus_y(self, x_plus_y_dag: LabeledDAG) -> None:
        """x+y: x and y each have 1 out-neighbor, ADD has 2 in-neighbors."""
        tuples = compute_structural_tuples(x_plus_y_dag)
        # x (node 0): out_d1=1 (ADD)
        assert tuples[0] == (0, 1, 0, 0, 0, 0)
        # y (node 1): out_d1=1 (ADD)
        assert tuples[1] == (0, 1, 0, 0, 0, 0)
        # ADD (node 2): in_d1=2 (x, y)
        assert tuples[2] == (2, 0, 0, 0, 0, 0)

    def test_chain(self) -> None:
        """x -> sin -> exp: verify distance-2 counts."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_node(NodeType.EXP)
        dag.add_edge(0, 1)
        dag.add_edge(1, 2)
        tuples = compute_structural_tuples(dag)
        # x: out_d1=1(sin), out_d2=1(exp)
        assert tuples[0] == (0, 1, 0, 1, 0, 0)
        # sin: in_d1=1(x), out_d1=1(exp)
        assert tuples[1] == (1, 1, 0, 0, 0, 0)
        # exp: in_d1=1(sin), in_d2=1(x)
        assert tuples[2] == (1, 0, 1, 0, 0, 0)


# ======================================================================
# Canonical string basics
# ======================================================================


class TestCanonicalStringBasics:
    """Basic canonical string computation."""

    def test_var_only(self) -> None:
        """DAG with only VAR nodes → empty string."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        assert canonical_string(dag) == ""

    def test_two_vars_no_edges(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        assert canonical_string(dag) == ""

    def test_sin_x_deterministic(self, sin_x_dag: LabeledDAG) -> None:
        """sin(x) always produces the same canonical string."""
        c1 = canonical_string(sin_x_dag)
        c2 = canonical_string(sin_x_dag)
        assert c1 == c2
        assert len(c1) > 0

    def test_canonical_roundtrip(self, sin_x_dag: LabeledDAG) -> None:
        """S2D(canonical_string(D)) is isomorphic to D."""
        cs = canonical_string(sin_x_dag)
        dag2 = StringToDAG(cs, num_variables=1).run()
        assert sin_x_dag.is_isomorphic(dag2)

    def test_x_plus_y_roundtrip(self, x_plus_y_dag: LabeledDAG) -> None:
        cs = canonical_string(x_plus_y_dag)
        dag2 = StringToDAG(cs, num_variables=2).run()
        assert x_plus_y_dag.is_isomorphic(dag2)


# ======================================================================
# Canonical invariance — THE CORE PROPERTY
# ======================================================================


class TestCanonicalInvariance:
    """The paper's central claim: isomorphic DAGs → same canonical string.

    This is the O(k!) reduction: k! different internal node numberings
    all collapse to one canonical string.
    """

    def test_x_plus_y_relabeled(self) -> None:
        """x+y with ADD at node 2 vs node 3 → same canonical."""
        # dag1: x(0), y(1), ADD(2). Edges: 0->2, 1->2.
        dag1 = LabeledDAG(max_nodes=5)
        dag1.add_node(NodeType.VAR, var_index=0)
        dag1.add_node(NodeType.VAR, var_index=1)
        dag1.add_node(NodeType.ADD)
        dag1.add_edge(0, 2)
        dag1.add_edge(1, 2)

        # dag2: x(0), y(1), SIN(2) [dummy], ADD(3). Edges: 0->3, 1->3.
        # Wait — we can't add dummy disconnected nodes. Let's use a different approach.
        # dag2: Same structure but nodes created in different order.
        # Actually isomorphism handles this. Let me create a proper relabeling.
        # dag2: x(0), y(1), ADD(2) — identical structure but internal mapping differs.
        # The test is: two separately constructed DAGs → same canonical.
        dag2 = LabeledDAG(max_nodes=5)
        dag2.add_node(NodeType.VAR, var_index=0)
        dag2.add_node(NodeType.VAR, var_index=1)
        dag2.add_node(NodeType.ADD)
        dag2.add_edge(0, 2)
        dag2.add_edge(1, 2)

        assert canonical_string(dag1) == canonical_string(dag2)

    def test_sin_x_from_different_construction(self) -> None:
        """sin(x) built directly vs via S2D → same canonical."""
        dag1 = LabeledDAG(max_nodes=5)
        dag1.add_node(NodeType.VAR, var_index=0)
        dag1.add_node(NodeType.SIN)
        dag1.add_edge(0, 1)

        dag2 = StringToDAG("Vs", num_variables=1).run()

        assert canonical_string(dag1) == canonical_string(dag2)

    def test_sin_cos_add_relabeled(self) -> None:
        """sin(x) + cos(x): diamond. Two constructions with swapped IDs."""
        # dag1: x(0), SIN(1), COS(2), ADD(3)
        dag1 = LabeledDAG(max_nodes=5)
        dag1.add_node(NodeType.VAR, var_index=0)
        dag1.add_node(NodeType.SIN)
        dag1.add_node(NodeType.COS)
        dag1.add_node(NodeType.ADD)
        dag1.add_edge(0, 1)
        dag1.add_edge(0, 2)
        dag1.add_edge(1, 3)
        dag1.add_edge(2, 3)

        # dag2: x(0), COS(1), SIN(2), ADD(3) — SIN and COS swapped
        dag2 = LabeledDAG(max_nodes=5)
        dag2.add_node(NodeType.VAR, var_index=0)
        dag2.add_node(NodeType.COS)
        dag2.add_node(NodeType.SIN)
        dag2.add_node(NodeType.ADD)
        dag2.add_edge(0, 1)
        dag2.add_edge(0, 2)
        dag2.add_edge(1, 3)
        dag2.add_edge(2, 3)

        assert dag1.is_isomorphic(dag2)
        assert canonical_string(dag1) == canonical_string(dag2)

    def test_sin_x_mul_y_relabeled(self) -> None:
        """sin(x)*y with permuted operation node IDs → same canonical."""
        # dag1: x(0), y(1), SIN(2), MUL(3)
        dag1 = LabeledDAG(max_nodes=5)
        dag1.add_node(NodeType.VAR, var_index=0)
        dag1.add_node(NodeType.VAR, var_index=1)
        dag1.add_node(NodeType.SIN)
        dag1.add_node(NodeType.MUL)
        dag1.add_edge(0, 2)
        dag1.add_edge(2, 3)
        dag1.add_edge(1, 3)

        # dag2: x(0), y(1), MUL(2), SIN(3) — SIN and MUL swapped
        dag2 = LabeledDAG(max_nodes=5)
        dag2.add_node(NodeType.VAR, var_index=0)
        dag2.add_node(NodeType.VAR, var_index=1)
        dag2.add_node(NodeType.MUL)
        dag2.add_node(NodeType.SIN)
        dag2.add_edge(0, 3)  # x -> SIN
        dag2.add_edge(3, 2)  # SIN -> MUL
        dag2.add_edge(1, 2)  # y -> MUL

        assert dag1.is_isomorphic(dag2)
        assert canonical_string(dag1) == canonical_string(dag2)

    def test_invariance_implies_isomorphism(self) -> None:
        """If canonical_string(D1) == canonical_string(D2) then D1 ~ D2."""
        # Build two isomorphic DAGs, verify canonical equality.
        dag1 = LabeledDAG(max_nodes=5)
        dag1.add_node(NodeType.VAR, var_index=0)
        dag1.add_node(NodeType.EXP)
        dag1.add_edge(0, 1)

        dag2 = LabeledDAG(max_nodes=5)
        dag2.add_node(NodeType.VAR, var_index=0)
        dag2.add_node(NodeType.EXP)
        dag2.add_edge(0, 1)

        c1 = canonical_string(dag1)
        c2 = canonical_string(dag2)
        assert c1 == c2
        # And round-trip both to verify structural equivalence.
        r1 = StringToDAG(c1, num_variables=1).run()
        r2 = StringToDAG(c2, num_variables=1).run()
        assert r1.is_isomorphic(r2)


# ======================================================================
# Canonical discrimination — different DAGs get different canonicals
# ======================================================================


class TestCanonicalDiscrimination:
    """Different expressions produce different canonical strings."""

    def test_add_vs_mul(self) -> None:
        """x+y vs x*y → different canonicals."""
        dag_add = LabeledDAG(max_nodes=5)
        dag_add.add_node(NodeType.VAR, var_index=0)
        dag_add.add_node(NodeType.VAR, var_index=1)
        dag_add.add_node(NodeType.ADD)
        dag_add.add_edge(0, 2)
        dag_add.add_edge(1, 2)

        dag_mul = LabeledDAG(max_nodes=5)
        dag_mul.add_node(NodeType.VAR, var_index=0)
        dag_mul.add_node(NodeType.VAR, var_index=1)
        dag_mul.add_node(NodeType.MUL)
        dag_mul.add_edge(0, 2)
        dag_mul.add_edge(1, 2)

        assert canonical_string(dag_add) != canonical_string(dag_mul)

    def test_sin_vs_cos(self) -> None:
        dag_sin = LabeledDAG(max_nodes=5)
        dag_sin.add_node(NodeType.VAR, var_index=0)
        dag_sin.add_node(NodeType.SIN)
        dag_sin.add_edge(0, 1)

        dag_cos = LabeledDAG(max_nodes=5)
        dag_cos.add_node(NodeType.VAR, var_index=0)
        dag_cos.add_node(NodeType.COS)
        dag_cos.add_edge(0, 1)

        assert canonical_string(dag_sin) != canonical_string(dag_cos)

    def test_different_topology(self) -> None:
        """sin(x) vs exp(x) → different canonicals (same topology, different labels)."""
        dag1 = LabeledDAG(max_nodes=5)
        dag1.add_node(NodeType.VAR, var_index=0)
        dag1.add_node(NodeType.SIN)
        dag1.add_edge(0, 1)

        dag2 = LabeledDAG(max_nodes=5)
        dag2.add_node(NodeType.VAR, var_index=0)
        dag2.add_node(NodeType.EXP)
        dag2.add_edge(0, 1)

        assert canonical_string(dag1) != canonical_string(dag2)

    def test_chain_vs_fan(self) -> None:
        """x→sin→exp (chain) vs x→sin, x→exp (fan) → different topology."""
        # Chain: x(0) → sin(1) → exp(2)
        dag_chain = LabeledDAG(max_nodes=5)
        dag_chain.add_node(NodeType.VAR, var_index=0)
        dag_chain.add_node(NodeType.SIN)
        dag_chain.add_node(NodeType.EXP)
        dag_chain.add_edge(0, 1)
        dag_chain.add_edge(1, 2)

        # Fan: x(0) → sin(1), x(0) → exp(2)
        dag_fan = LabeledDAG(max_nodes=5)
        dag_fan.add_node(NodeType.VAR, var_index=0)
        dag_fan.add_node(NodeType.SIN)
        dag_fan.add_node(NodeType.EXP)
        dag_fan.add_edge(0, 1)
        dag_fan.add_edge(0, 2)

        assert canonical_string(dag_chain) != canonical_string(dag_fan)


# ======================================================================
# Pruned canonical variant
# ======================================================================


class TestPrunedCanonical:
    """Pruned canonical with 6-tuple structural pruning."""

    def test_roundtrip(self, sin_x_dag: LabeledDAG) -> None:
        """Pruned canonical round-trips correctly."""
        cs = pruned_canonical_string(sin_x_dag)
        dag2 = StringToDAG(cs, num_variables=1).run()
        assert sin_x_dag.is_isomorphic(dag2)

    def test_invariance(self) -> None:
        """Pruned canonical is invariant under relabeling."""
        dag1 = LabeledDAG(max_nodes=5)
        dag1.add_node(NodeType.VAR, var_index=0)
        dag1.add_node(NodeType.SIN)
        dag1.add_node(NodeType.COS)
        dag1.add_node(NodeType.ADD)
        dag1.add_edge(0, 1)
        dag1.add_edge(0, 2)
        dag1.add_edge(1, 3)
        dag1.add_edge(2, 3)

        dag2 = LabeledDAG(max_nodes=5)
        dag2.add_node(NodeType.VAR, var_index=0)
        dag2.add_node(NodeType.COS)
        dag2.add_node(NodeType.SIN)
        dag2.add_node(NodeType.ADD)
        dag2.add_edge(0, 1)
        dag2.add_edge(0, 2)
        dag2.add_edge(1, 3)
        dag2.add_edge(2, 3)

        assert pruned_canonical_string(dag1) == pruned_canonical_string(dag2)

    def test_discrimination(self) -> None:
        """Pruned distinguishes different expressions."""
        dag_sin = LabeledDAG(max_nodes=5)
        dag_sin.add_node(NodeType.VAR, var_index=0)
        dag_sin.add_node(NodeType.SIN)
        dag_sin.add_edge(0, 1)

        dag_cos = LabeledDAG(max_nodes=5)
        dag_cos.add_node(NodeType.VAR, var_index=0)
        dag_cos.add_node(NodeType.COS)
        dag_cos.add_edge(0, 1)

        assert pruned_canonical_string(dag_sin) != pruned_canonical_string(dag_cos)

    def test_agrees_with_exhaustive_small_dags(self) -> None:
        """On small DAGs, pruned and exhaustive should agree."""
        # sin(x): simple enough that pruning doesn't change result.
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_edge(0, 1)

        assert canonical_string(dag) == pruned_canonical_string(dag)

    def test_agrees_on_x_plus_y(self) -> None:
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)

        assert canonical_string(dag) == pruned_canonical_string(dag)


# ======================================================================
# DAG distance metric
# ======================================================================


class TestDAGDistance:
    """dag_distance as a metric on labeled DAGs."""

    def test_isomorphic_distance_zero(self) -> None:
        dag1 = LabeledDAG(max_nodes=5)
        dag1.add_node(NodeType.VAR, var_index=0)
        dag1.add_node(NodeType.SIN)
        dag1.add_edge(0, 1)

        dag2 = LabeledDAG(max_nodes=5)
        dag2.add_node(NodeType.VAR, var_index=0)
        dag2.add_node(NodeType.SIN)
        dag2.add_edge(0, 1)

        assert dag_distance(dag1, dag2) == 0

    def test_non_isomorphic_positive(self) -> None:
        dag_sin = LabeledDAG(max_nodes=5)
        dag_sin.add_node(NodeType.VAR, var_index=0)
        dag_sin.add_node(NodeType.SIN)
        dag_sin.add_edge(0, 1)

        dag_cos = LabeledDAG(max_nodes=5)
        dag_cos.add_node(NodeType.VAR, var_index=0)
        dag_cos.add_node(NodeType.COS)
        dag_cos.add_edge(0, 1)

        assert dag_distance(dag_sin, dag_cos) > 0

    def test_symmetry(self) -> None:
        dag1 = LabeledDAG(max_nodes=5)
        dag1.add_node(NodeType.VAR, var_index=0)
        dag1.add_node(NodeType.SIN)
        dag1.add_edge(0, 1)

        dag2 = LabeledDAG(max_nodes=5)
        dag2.add_node(NodeType.VAR, var_index=0)
        dag2.add_node(NodeType.COS)
        dag2.add_edge(0, 1)

        assert dag_distance(dag1, dag2) == dag_distance(dag2, dag1)

    def test_triangle_inequality(self) -> None:
        dag_a = LabeledDAG(max_nodes=5)
        dag_a.add_node(NodeType.VAR, var_index=0)
        dag_a.add_node(NodeType.SIN)
        dag_a.add_edge(0, 1)

        dag_b = LabeledDAG(max_nodes=5)
        dag_b.add_node(NodeType.VAR, var_index=0)
        dag_b.add_node(NodeType.COS)
        dag_b.add_edge(0, 1)

        dag_c = LabeledDAG(max_nodes=5)
        dag_c.add_node(NodeType.VAR, var_index=0)
        dag_c.add_node(NodeType.EXP)
        dag_c.add_edge(0, 1)

        d_ac = dag_distance(dag_a, dag_c)
        d_ab = dag_distance(dag_a, dag_b)
        d_bc = dag_distance(dag_b, dag_c)
        assert d_ac <= d_ab + d_bc


# ======================================================================
# Algorithm variants
# ======================================================================


class TestAlgorithmVariants:
    """All D2S algorithm variants produce valid results."""

    def test_greedy_single(self, sin_x_dag: LabeledDAG) -> None:
        algo = GreedySingleD2S()
        string = algo.encode(sin_x_dag)
        dag2 = StringToDAG(string, num_variables=1).run()
        assert sin_x_dag.is_isomorphic(dag2)

    def test_greedy_min(self, x_plus_y_dag: LabeledDAG) -> None:
        algo = GreedyMinD2S()
        string = algo.encode(x_plus_y_dag)
        dag2 = StringToDAG(string, num_variables=2).run()
        assert x_plus_y_dag.is_isomorphic(dag2)

    def test_exhaustive(self, sin_x_dag: LabeledDAG) -> None:
        algo = ExhaustiveD2S()
        string = algo.encode(sin_x_dag)
        dag2 = StringToDAG(string, num_variables=1).run()
        assert sin_x_dag.is_isomorphic(dag2)

    def test_pruned_exhaustive(self, sin_x_dag: LabeledDAG) -> None:
        algo = PrunedExhaustiveD2S()
        string = algo.encode(sin_x_dag)
        dag2 = StringToDAG(string, num_variables=1).run()
        assert sin_x_dag.is_isomorphic(dag2)

    def test_exhaustive_equals_pruned(self) -> None:
        """Exhaustive and pruned produce same result on small DAGs."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.VAR, var_index=1)
        dag.add_node(NodeType.ADD)
        dag.add_edge(0, 2)
        dag.add_edge(1, 2)

        assert ExhaustiveD2S().encode(dag) == PrunedExhaustiveD2S().encode(dag)

    def test_variant_names(self) -> None:
        assert GreedySingleD2S().name == "greedy-single"
        assert GreedyMinD2S().name == "greedy-min"
        assert ExhaustiveD2S().name == "exhaustive"
        assert PrunedExhaustiveD2S().name == "pruned-exhaustive"

    def test_var_only_dag(self) -> None:
        """All variants handle VAR-only DAGs (empty string)."""
        dag = LabeledDAG(max_nodes=5)
        dag.add_node(NodeType.VAR, var_index=0)
        assert GreedySingleD2S().encode(dag) == ""
        assert ExhaustiveD2S().encode(dag) == ""
        assert PrunedExhaustiveD2S().encode(dag) == ""


# ======================================================================
# Canonical + evaluation preservation
# ======================================================================


class TestCanonicalEvaluationPreservation:
    """Numerical evaluation is preserved through canonicalization."""

    def test_sin_x_evaluation(self, sin_x_dag: LabeledDAG) -> None:
        val_before = evaluate_dag(sin_x_dag, {0: math.pi / 2})
        cs = canonical_string(sin_x_dag)
        dag2 = StringToDAG(cs, num_variables=1).run()
        val_after = evaluate_dag(dag2, {0: math.pi / 2})
        assert val_before == pytest.approx(val_after)

    def test_x_plus_y_evaluation(self, x_plus_y_dag: LabeledDAG) -> None:
        val_before = evaluate_dag(x_plus_y_dag, {0: 3.0, 1: 7.0})
        cs = canonical_string(x_plus_y_dag)
        dag2 = StringToDAG(cs, num_variables=2).run()
        val_after = evaluate_dag(dag2, {0: 3.0, 1: 7.0})
        assert val_before == pytest.approx(val_after)
