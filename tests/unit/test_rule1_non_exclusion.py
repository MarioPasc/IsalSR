"""Unit tests for the Rule 1 non-exclusion property across all BINARY_OPS.

Rule 1 (first-operand eligibility): a binary-op candidate c with at least one
recorded in-neighbour is eligible from acting pointer u only if
ordered_inputs(c)[0] == u.  Implemented for ALL BINARY_OPS = {SUB, DIV, POW}
in canonical.py (the predicate at the V/v branch in _step and _fast_step).

Tests verify:
1.  count_rule1_exclusions returns 0 for a POW with only one in-edge.
2.  count_rule1_exclusions returns 1 for a POW with two in-edges (base+exponent).
3.  count_rule1_exclusions covers SUB and DIV (implementation covers BINARY_OPS).
4.  count_rule1_exclusions_per_op correctly attributes exclusions to each op type.
5.  fast_canonical_string succeeds on two-parent DAGs for all three op types.
6.  Round-trip fidelity: dag.is_isomorphic(S2D(fcs, m)) for all three op types.
7.  build_exclusion_dag constructs valid exclusion DAGs for POW, SUB, and DIV.
8.  A DAG with zero binary-op nodes has exclusion count 0 (total and per-op).
9.  Round-trip holds for a POW DAG decoded from a known good string.
"""

from __future__ import annotations

import random

import pytest

from experiments.scripts.validate_rule1_non_exclusion import (
    build_exclusion_dag,
    count_rule1_exclusions,
    count_rule1_exclusions_per_op,
    has_pow_node,
    satisfies_reachability,
)
from isalsr.core.canonical import fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.string_to_dag import StringToDAG

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pow_one_parent() -> LabeledDAG:
    """POW node with a single in-edge (base only, exponent not connected).

    Graph: x0 --(base)--> p
    ordered_inputs(p) = [x0]  ->  no (u, p) pair where u != x0.
    """
    dag = LabeledDAG(2)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.POW)
    dag.add_edge(0, 1)
    return dag


@pytest.fixture()
def pow_two_parents() -> LabeledDAG:
    """POW with two in-edges: base=x0, exponent=x1.

    ordered_inputs(p) = [x0, x1].
    Rule 1 excludes (x1, p): ordered_inputs(p)[0] = x0 != x1.
    """
    dag = LabeledDAG(3)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    dag.add_node(NodeType.POW)
    dag.add_edge(0, 2)  # x0 -> p  (base, first)
    dag.add_edge(1, 2)  # x1 -> p  (exponent, second)
    return dag


@pytest.fixture()
def sub_two_parents() -> LabeledDAG:
    """SUB with two in-edges: minuend=x0, subtrahend=x1."""
    dag = LabeledDAG(3)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    dag.add_node(NodeType.SUB)
    dag.add_edge(0, 2)  # x0 -> s  (minuend, first)
    dag.add_edge(1, 2)  # x1 -> s  (subtrahend, second)
    return dag


@pytest.fixture()
def div_two_parents() -> LabeledDAG:
    """DIV with two in-edges: numerator=x0, denominator=x1."""
    dag = LabeledDAG(3)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    dag.add_node(NodeType.DIV)
    dag.add_edge(0, 2)  # x0 -> d  (numerator, first)
    dag.add_edge(1, 2)  # x1 -> d  (denominator, second)
    return dag


@pytest.fixture()
def compound_pow_dag() -> LabeledDAG:
    """Larger DAG: ADD(x0, x1) as base, x2 as exponent of POW.

    ordered_inputs(p) = [a, x2].  Rule 1 excludes (x2, p).
    count_rule1_exclusions = 1.
    """
    dag = LabeledDAG(5)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    dag.add_node(NodeType.VAR, var_index=2)
    dag.add_node(NodeType.ADD)
    dag.add_node(NodeType.POW)
    dag.add_edge(0, 3)  # x0 -> a
    dag.add_edge(1, 3)  # x1 -> a
    dag.add_edge(3, 4)  # a -> p  (base, first)
    dag.add_edge(2, 4)  # x2 -> p (exponent, second)
    return dag


# ---------------------------------------------------------------------------
# Tests: count_rule1_exclusions (total)
# ---------------------------------------------------------------------------


def test_count_exclusions_single_parent_pow(pow_one_parent: LabeledDAG) -> None:
    """POW with one in-edge has no excluded pairs."""
    assert count_rule1_exclusions(pow_one_parent) == 0


def test_count_exclusions_two_parents_pow(pow_two_parents: LabeledDAG) -> None:
    """POW with two in-edges has exactly one excluded pair: (x1, p)."""
    assert count_rule1_exclusions(pow_two_parents) == 1


def test_count_exclusions_sub_two_parents(sub_two_parents: LabeledDAG) -> None:
    """SUB with two in-edges has one excluded pair — confirms SUB is covered."""
    assert count_rule1_exclusions(sub_two_parents) == 1


def test_count_exclusions_div_two_parents(div_two_parents: LabeledDAG) -> None:
    """DIV with two in-edges has one excluded pair — confirms DIV is covered."""
    assert count_rule1_exclusions(div_two_parents) == 1


def test_count_exclusions_no_binary_ops() -> None:
    """DAG with only VAR and ADD (variadic) has count 0."""
    dag = LabeledDAG(3)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    dag.add_node(NodeType.ADD)
    dag.add_edge(0, 2)
    dag.add_edge(1, 2)
    assert count_rule1_exclusions(dag) == 0


def test_count_exclusions_compound_pow(compound_pow_dag: LabeledDAG) -> None:
    """Compound POW DAG has one excluded pair: (x2, p)."""
    assert count_rule1_exclusions(compound_pow_dag) == 1


# ---------------------------------------------------------------------------
# Tests: count_rule1_exclusions_per_op (per-op breakdown)
# ---------------------------------------------------------------------------


def test_per_op_pow_only(pow_two_parents: LabeledDAG) -> None:
    """POW DAG: per_op[POW]=1, per_op[SUB]=per_op[DIV]=0."""
    per_op = count_rule1_exclusions_per_op(pow_two_parents)
    assert per_op[NodeType.POW] == 1
    assert per_op[NodeType.SUB] == 0
    assert per_op[NodeType.DIV] == 0


def test_per_op_sub_only(sub_two_parents: LabeledDAG) -> None:
    """SUB DAG: per_op[SUB]=1, per_op[POW]=per_op[DIV]=0."""
    per_op = count_rule1_exclusions_per_op(sub_two_parents)
    assert per_op[NodeType.SUB] == 1
    assert per_op[NodeType.POW] == 0
    assert per_op[NodeType.DIV] == 0


def test_per_op_div_only(div_two_parents: LabeledDAG) -> None:
    """DIV DAG: per_op[DIV]=1, per_op[POW]=per_op[SUB]=0."""
    per_op = count_rule1_exclusions_per_op(div_two_parents)
    assert per_op[NodeType.DIV] == 1
    assert per_op[NodeType.POW] == 0
    assert per_op[NodeType.SUB] == 0


def test_per_op_total_matches_sum(pow_two_parents: LabeledDAG) -> None:
    """count_rule1_exclusions equals sum of per_op values."""
    per_op = count_rule1_exclusions_per_op(pow_two_parents)
    assert count_rule1_exclusions(pow_two_parents) == sum(per_op.values())


def test_per_op_single_parent_all_zero(pow_one_parent: LabeledDAG) -> None:
    """POW with single parent: all per_op counts are 0."""
    per_op = count_rule1_exclusions_per_op(pow_one_parent)
    assert per_op[NodeType.POW] == 0
    assert per_op[NodeType.SUB] == 0
    assert per_op[NodeType.DIV] == 0


# ---------------------------------------------------------------------------
# Tests: build_exclusion_dag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_type", [NodeType.POW, NodeType.SUB, NodeType.DIV])
def test_build_exclusion_dag_has_correct_op(op_type: NodeType) -> None:
    """build_exclusion_dag creates a DAG containing the specified op type."""
    rng = random.Random(0)
    dag = build_exclusion_dag(m=2, extra_nodes=0, op_type=op_type, rng=rng)
    assert any(dag.node_label_unchecked(i) == op_type for i in range(dag.node_count))


@pytest.mark.parametrize("op_type", [NodeType.POW, NodeType.SUB, NodeType.DIV])
def test_build_exclusion_dag_exclusion_positive(op_type: NodeType) -> None:
    """build_exclusion_dag always yields count_rule1_exclusions_per_op[op] >= 1."""
    rng = random.Random(0)
    for m in [2, 3, 4]:
        dag = build_exclusion_dag(m=m, extra_nodes=0, op_type=op_type, rng=rng)
        per_op = count_rule1_exclusions_per_op(dag)
        assert per_op[op_type] >= 1, f"Expected excl>0 for {op_type.name} with m={m}"


@pytest.mark.parametrize("op_type", [NodeType.POW, NodeType.SUB, NodeType.DIV])
def test_build_exclusion_dag_reachability(op_type: NodeType) -> None:
    """build_exclusion_dag satisfies the Round-Trip Fidelity reachability precondition."""
    rng = random.Random(7)
    dag = build_exclusion_dag(m=3, extra_nodes=2, op_type=op_type, rng=rng)
    assert satisfies_reachability(dag)


# ---------------------------------------------------------------------------
# Tests: has_pow_node and satisfies_reachability helpers
# ---------------------------------------------------------------------------


def test_has_pow_node_true(pow_two_parents: LabeledDAG) -> None:
    """POW-containing DAG is detected."""
    assert has_pow_node(pow_two_parents)


def test_has_pow_node_false(sub_two_parents: LabeledDAG) -> None:
    """DAG without POW is not detected as POW-containing."""
    assert not has_pow_node(sub_two_parents)


def test_reachability_satisfied(pow_two_parents: LabeledDAG) -> None:
    """Two-parent POW DAG satisfies the reachability precondition."""
    assert satisfies_reachability(pow_two_parents)


def test_reachability_violated() -> None:
    """DAG with an unreachable POW node violates the precondition."""
    dag = LabeledDAG(3)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    dag.add_node(NodeType.POW)  # no edges: unreachable
    assert not satisfies_reachability(dag)


# ---------------------------------------------------------------------------
# Tests: fast_canonical_string succeeds (non-exclusion property)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dag_fixture, m",
    [
        ("pow_two_parents", 2),
        ("compound_pow_dag", 3),
        ("sub_two_parents", 2),
        ("div_two_parents", 2),
    ],
)
def test_fast_canonical_no_error(dag_fixture: str, m: int, request: pytest.FixtureRequest) -> None:
    """fast_canonical_string must not raise on binary-op DAGs with two parents.

    This is the core non-exclusion property: Rule 1 restricts candidates but
    must never leave the search without a valid serialisation path.
    """
    dag: LabeledDAG = request.getfixturevalue(dag_fixture)
    result = fast_canonical_string(dag, timeout=5.0, backend="python")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize("op_type", [NodeType.POW, NodeType.SUB, NodeType.DIV])
def test_fast_canonical_no_error_all_ops(op_type: NodeType) -> None:
    """fast_canonical_string succeeds on build_exclusion_dag output for each op."""
    rng = random.Random(13)
    dag = build_exclusion_dag(m=2, extra_nodes=1, op_type=op_type, rng=rng)
    result = fast_canonical_string(dag, timeout=5.0, backend="python")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests: round-trip fidelity D ~ S2D(fcs_D, m)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dag_fixture, m",
    [
        ("pow_two_parents", 2),
        ("compound_pow_dag", 3),
        ("sub_two_parents", 2),
        ("div_two_parents", 2),
    ],
)
def test_roundtrip_fidelity(dag_fixture: str, m: int, request: pytest.FixtureRequest) -> None:
    """D ~ S2D(fast_canonical_string(D), m) for binary-op DAGs with Rule 1 exclusions."""
    dag: LabeledDAG = request.getfixturevalue(dag_fixture)
    fcs = fast_canonical_string(dag, timeout=5.0, backend="python")
    decoded = StringToDAG(fcs, num_variables=m).run()
    assert dag.is_isomorphic(decoded), f"Round-trip failed for {dag_fixture}: fcs={fcs!r}"


@pytest.mark.parametrize("op_type", [NodeType.POW, NodeType.SUB, NodeType.DIV])
def test_roundtrip_all_ops_via_build(op_type: NodeType) -> None:
    """Round-trip holds for build_exclusion_dag output for each of the three op types."""
    rng = random.Random(99)
    dag = build_exclusion_dag(m=2, extra_nodes=0, op_type=op_type, rng=rng)
    fcs = fast_canonical_string(dag, timeout=5.0, backend="python")
    decoded = StringToDAG(fcs, num_variables=2).run()
    assert dag.is_isomorphic(decoded), f"Round-trip failed for {op_type.name}: fcs={fcs!r}"


def test_roundtrip_pow_from_string() -> None:
    """Round-trip holds for a POW DAG decoded from a known good string.

    'V^NNnC' with m=2 produces POW(x0, x1): base=x0, exp=x1.
    ordered_inputs(p) = [x0, x1].  Rule 1 excludes (x1, p) statically.
    """
    dag = StringToDAG("V^NNnC", num_variables=2).run()
    assert has_pow_node(dag)
    assert satisfies_reachability(dag)
    excl = count_rule1_exclusions(dag)
    assert excl > 0, f"Expected Rule 1 to exclude >= 1 pair; got {excl}"
    fcs = fast_canonical_string(dag, timeout=5.0, backend="python")
    decoded = StringToDAG(fcs, num_variables=2).run()
    assert dag.is_isomorphic(decoded), f"Round-trip failed: fcs={fcs!r}"
