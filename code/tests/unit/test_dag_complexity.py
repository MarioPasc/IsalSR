"""Unit tests for :mod:`isalsr.core.complexity`.

The descriptors are cheap enough that every one of them can be checked against an
independent oracle. Three oracles are used here:

* hand-computed :class:`DagComplexity` tuples for small DAGs drawn by hand;
* a brute-force reference implementation built only from the public
  :class:`~isalsr.core.labeled_dag.LabeledDAG` accessors, whose ``depth`` comes
  from explicit path enumeration rather than from the topological dynamic
  program under test;
* ``numpy`` and ``scipy.stats`` for the moments and the Shannon entropy.

The isomorphism-invariance tests are the scientifically load-bearing ones: a
descriptor that is not invariant under a relabelling of internal node IDs cannot
be compared across arms, because the two arms do not number their nodes the same
way.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from collections.abc import Sequence
from typing import Final

import numpy as np
import pytest
from scipy.stats import entropy as scipy_entropy

from isalsr.core.complexity import (
    DESCRIPTOR_FIELDS,
    HEADLINE_FIELDS,
    NONLINEAR_OPS,
    ComplexityAccumulator,
    DagComplexity,
    describe_dag,
    describe_dag_with_labels,
)
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.permutations import permute_internal_nodes
from isalsr.core.string_to_dag import StringToDAG

#: Operator labels used by the random-DAG generator. ``SUB``/``DIV`` are omitted
#: because the decomposed alphabet (T16) never emits them.
_OP_POOL: Final[tuple[NodeType, ...]] = (
    NodeType.ADD,
    NodeType.MUL,
    NodeType.SIN,
    NodeType.COS,
    NodeType.EXP,
    NodeType.LOG,
    NodeType.SQRT,
    NodeType.POW,
    NodeType.ABS,
    NodeType.NEG,
    NodeType.INV,
)


# ----------------------------------------------------------------------
# Oracles
# ----------------------------------------------------------------------


def _longest_path_edges(dag: LabeledDAG) -> int:
    """Return the longest directed path length, in edges, by explicit enumeration.

    Deliberately exponential: it enumerates every directed path with a depth-first
    walk instead of reusing the topological dynamic program that
    :func:`~isalsr.core.complexity.describe_dag` uses, so the two cannot share a
    bug. Only usable on small DAGs.

    Parameters
    ----------
    dag : LabeledDAG
        The DAG to measure.

    Returns
    -------
    int
        The number of edges on the longest directed path, ``0`` for an edgeless
        or empty DAG.
    """
    best = 0
    stack: list[tuple[int, int]] = [(node, 0) for node in range(dag.node_count)]
    while stack:
        node, length = stack.pop()
        if length > best:
            best = length
        for succ in dag.out_neighbors(node):
            stack.append((succ, length + 1))
    return best


def _reference_descriptor(dag: LabeledDAG) -> DagComplexity:
    """Recompute the descriptor vector from the public accessors only.

    Independent of the single-sweep implementation: every component is derived
    from a direct definition, and ``depth`` comes from :func:`_longest_path_edges`.

    Parameters
    ----------
    dag : LabeledDAG
        The DAG to describe.

    Returns
    -------
    DagComplexity
        The reference descriptor vector.
    """
    n_nodes = dag.node_count
    if n_nodes == 0:
        return DagComplexity(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)

    labels = [dag.node_label(v) for v in range(n_nodes)]
    out_degrees = [dag.out_degree(v) for v in range(n_nodes)]
    in_degrees = [dag.in_degree(v) for v in range(n_nodes)]

    n_var = sum(1 for label in labels if label is NodeType.VAR)
    n_const = sum(1 for label in labels if label is NodeType.CONST)
    n_internal = n_nodes - n_var
    n_op = n_internal - n_const

    op_counts = Counter(label for label in labels if label not in (NodeType.VAR, NodeType.CONST))
    if n_op > 0:
        total = float(n_op)
        entropy = -sum((c / total) * math.log2(c / total) for c in op_counts.values())
        entropy = 0.0 if entropy < 1e-12 else entropy
    else:
        entropy = 0.0

    return DagComplexity(
        n_nodes=n_nodes,
        n_edges=dag.edge_count,
        n_var=n_var,
        n_internal=n_internal,
        n_const=n_const,
        n_op=n_op,
        depth=_longest_path_edges(dag),
        max_in_degree=max(in_degrees),
        max_out_degree=max(out_degrees),
        n_shared=sum(1 for d in out_degrees if d >= 2),
        sharing_surplus=sum(max(0, d - 1) for d in out_degrees),
        n_nonlinear=sum(1 for label in labels if label in NONLINEAR_OPS),
        n_var_used=sum(
            1 for v, label in enumerate(labels) if label is NodeType.VAR and out_degrees[v] > 0
        ),
        n_distinct_op_labels=len(op_counts),
        op_label_entropy=entropy,
    )


def _assert_descriptors_equal(actual: DagComplexity, expected: DagComplexity) -> None:
    """Assert two descriptor vectors agree, comparing the entropy approximately.

    Parameters
    ----------
    actual : DagComplexity
        Descriptor under test.
    expected : DagComplexity
        Reference descriptor.
    """
    assert actual[:-1] == expected[:-1]
    assert actual.op_label_entropy == pytest.approx(expected.op_label_entropy, abs=1e-12)


# ----------------------------------------------------------------------
# DAG builders
# ----------------------------------------------------------------------


def _build_bare_variable() -> LabeledDAG:
    """Return the single-node DAG ``x0``."""
    dag = LabeledDAG(max_nodes=1)
    dag.add_node(NodeType.VAR, var_index=0)
    return dag


def _build_sin_x() -> LabeledDAG:
    """Return ``sin(x0)``: one VAR, one SIN, one edge."""
    dag = LabeledDAG(max_nodes=2)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.SIN)  # 1
    dag.add_edge(0, 1)
    return dag


def _build_shared_product_sum() -> LabeledDAG:
    """Return ``x0 * x1 + x0``: the smallest DAG with genuine subexpression sharing."""
    dag = LabeledDAG(max_nodes=4)
    dag.add_node(NodeType.VAR, var_index=0)  # 0: x0
    dag.add_node(NodeType.VAR, var_index=1)  # 1: x1
    dag.add_node(NodeType.MUL)  # 2
    dag.add_node(NodeType.ADD)  # 3
    dag.add_edge(0, 2)
    dag.add_edge(1, 2)
    dag.add_edge(2, 3)
    dag.add_edge(0, 3)  # x0 read a second time -> out-degree 2
    return dag


def _build_unary_chain() -> LabeledDAG:
    """Return ``sin(cos(exp(log(sqrt(x0)))))``: a depth-5 chain of five distinct ops."""
    dag = LabeledDAG(max_nodes=6)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.SQRT)  # 1
    dag.add_node(NodeType.LOG)  # 2
    dag.add_node(NodeType.EXP)  # 3
    dag.add_node(NodeType.COS)  # 4
    dag.add_node(NodeType.SIN)  # 5
    for src in range(5):
        dag.add_edge(src, src + 1)
    return dag


def _build_two_constants() -> LabeledDAG:
    """Return ``x0 * c0 + c1``: two CONST leaves, two commutative operators."""
    dag = LabeledDAG(max_nodes=5)
    dag.add_node(NodeType.VAR, var_index=0)  # 0
    dag.add_node(NodeType.CONST, const_value=2.0)  # 1
    dag.add_node(NodeType.MUL)  # 2
    dag.add_node(NodeType.CONST, const_value=3.0)  # 3
    dag.add_node(NodeType.ADD)  # 4
    dag.add_edge(0, 2)
    dag.add_edge(1, 2)
    dag.add_edge(2, 4)
    dag.add_edge(3, 4)
    return dag


def _build_shared_constant() -> LabeledDAG:
    """Return ``exp(c) + sin(c)`` with one CONST read twice, plus an unused variable."""
    dag = LabeledDAG(max_nodes=5)
    dag.add_node(NodeType.VAR, var_index=0)  # 0: never read
    dag.add_node(NodeType.CONST, const_value=1.5)  # 1
    dag.add_node(NodeType.EXP)  # 2
    dag.add_node(NodeType.SIN)  # 3
    dag.add_node(NodeType.ADD)  # 4
    dag.add_edge(1, 2)
    dag.add_edge(1, 3)
    dag.add_edge(2, 4)
    dag.add_edge(3, 4)
    return dag


def _random_dag(
    rng: random.Random,
    *,
    n_var: int,
    n_internal: int,
    p_const: float = 0.15,
    n_extra_edges: int = 2,
) -> LabeledDAG:
    """Build a random labeled DAG with all VAR nodes first.

    Internal node ``m + j`` only ever receives edges from strictly smaller node
    IDs, so acyclicity holds by construction. VAR nodes occupy ``0..m-1``, which
    is the precondition of
    :func:`~isalsr.core.permutations.permute_internal_nodes`.

    Parameters
    ----------
    rng : random.Random
        Seeded generator.
    n_var : int
        Number of VAR nodes; must be at least one.
    n_internal : int
        Number of internal (operator or CONST) nodes.
    p_const : float
        Probability that an internal node is a CONST leaf.
    n_extra_edges : int
        Number of additional edges attempted, to create subexpression sharing.

    Returns
    -------
    LabeledDAG
        The generated DAG.
    """
    dag = LabeledDAG(max_nodes=n_var + n_internal)
    for i in range(n_var):
        dag.add_node(NodeType.VAR, var_index=i)

    op_nodes: list[int] = []
    for j in range(n_internal):
        node = n_var + j
        if rng.random() < p_const:
            dag.add_node(NodeType.CONST, const_value=rng.uniform(-3.0, 3.0))
            continue
        dag.add_node(rng.choice(_OP_POOL))
        op_nodes.append(node)
        n_in = rng.randint(1, min(2, node))
        for source in rng.sample(range(node), n_in):
            dag.add_edge(source, node)

    for _ in range(n_extra_edges):
        if not op_nodes:
            break
        target = rng.choice(op_nodes)
        if target == 0:
            continue
        dag.add_edge(rng.randrange(target), target)  # refused if duplicate

    return dag


# ----------------------------------------------------------------------
# 1. Hand-computed descriptors
# ----------------------------------------------------------------------

_HAND_CASES: Final[tuple[tuple[str, LabeledDAG, DagComplexity], ...]] = (
    (
        "bare_variable",
        _build_bare_variable(),
        DagComplexity(
            n_nodes=1,
            n_edges=0,
            n_var=1,
            n_internal=0,
            n_const=0,
            n_op=0,
            depth=0,
            max_in_degree=0,
            max_out_degree=0,
            n_shared=0,
            sharing_surplus=0,
            n_nonlinear=0,
            n_var_used=0,
            n_distinct_op_labels=0,
            op_label_entropy=0.0,
        ),
    ),
    (
        "sin_x",
        _build_sin_x(),
        DagComplexity(
            n_nodes=2,
            n_edges=1,
            n_var=1,
            n_internal=1,
            n_const=0,
            n_op=1,
            depth=1,
            max_in_degree=1,
            max_out_degree=1,
            n_shared=0,
            sharing_surplus=0,
            n_nonlinear=1,
            n_var_used=1,
            n_distinct_op_labels=1,
            op_label_entropy=0.0,
        ),
    ),
    (
        "shared_product_sum",
        _build_shared_product_sum(),
        DagComplexity(
            n_nodes=4,
            n_edges=4,
            n_var=2,
            n_internal=2,
            n_const=0,
            n_op=2,
            depth=2,
            max_in_degree=2,
            max_out_degree=2,
            n_shared=1,
            sharing_surplus=1,
            n_nonlinear=0,
            n_var_used=2,
            n_distinct_op_labels=2,
            op_label_entropy=1.0,
        ),
    ),
    (
        "unary_chain",
        _build_unary_chain(),
        DagComplexity(
            n_nodes=6,
            n_edges=5,
            n_var=1,
            n_internal=5,
            n_const=0,
            n_op=5,
            depth=5,
            max_in_degree=1,
            max_out_degree=1,
            n_shared=0,
            sharing_surplus=0,
            n_nonlinear=5,
            n_var_used=1,
            n_distinct_op_labels=5,
            op_label_entropy=math.log2(5.0),
        ),
    ),
    (
        "two_constants",
        _build_two_constants(),
        DagComplexity(
            n_nodes=5,
            n_edges=4,
            n_var=1,
            n_internal=4,
            n_const=2,
            n_op=2,
            depth=2,
            max_in_degree=2,
            max_out_degree=1,
            n_shared=0,
            sharing_surplus=0,
            n_nonlinear=0,
            n_var_used=1,
            n_distinct_op_labels=2,
            op_label_entropy=1.0,
        ),
    ),
    (
        "shared_constant",
        _build_shared_constant(),
        DagComplexity(
            n_nodes=5,
            n_edges=4,
            n_var=1,
            n_internal=4,
            n_const=1,
            n_op=3,
            depth=2,
            max_in_degree=2,
            max_out_degree=2,
            n_shared=1,
            sharing_surplus=1,
            n_nonlinear=2,
            n_var_used=0,
            n_distinct_op_labels=3,
            op_label_entropy=math.log2(3.0),
        ),
    ),
)


class TestHandComputedDescriptors:
    @pytest.mark.parametrize(
        ("dag", "expected"),
        [pytest.param(dag, expected, id=name) for name, dag, expected in _HAND_CASES],
    )
    def test_full_descriptor_tuple(self, dag: LabeledDAG, expected: DagComplexity) -> None:
        _assert_descriptors_equal(describe_dag(dag), expected)

    @pytest.mark.parametrize(
        ("dag", "expected"),
        [pytest.param(dag, expected, id=name) for name, dag, expected in _HAND_CASES],
    )
    def test_reference_oracle_agrees_with_hand_values(
        self, dag: LabeledDAG, expected: DagComplexity
    ) -> None:
        # Guards the oracle itself, which the random-DAG tests below rely on.
        _assert_descriptors_equal(_reference_descriptor(dag), expected)

    def test_sharing_is_detected_not_assumed(self) -> None:
        desc = describe_dag(_build_shared_product_sum())
        assert desc.n_shared == 1
        assert desc.sharing_surplus == 1
        assert desc.max_out_degree == 2

    def test_label_counts_match_node_labels(self) -> None:
        dag = _build_two_constants()
        _, label_counts = describe_dag_with_labels(dag)
        by_type = dict(zip(NodeType, label_counts, strict=True))
        assert by_type[NodeType.VAR] == 1
        assert by_type[NodeType.CONST] == 2
        assert by_type[NodeType.MUL] == 1
        assert by_type[NodeType.ADD] == 1
        assert sum(label_counts) == dag.node_count


class TestStringDerivedDags:
    """Descriptors of DAGs produced by the S2D converter."""

    @pytest.mark.parametrize(
        ("instruction_string", "num_variables"),
        [
            ("V+", 2),
            ("VsNVc", 2),
            ("V+NnVsC", 2),
            ("V*V+Nc", 3),
            ("VkV+NVs", 2),
            ("V+nNVsVcC", 3),
        ],
    )
    def test_matches_reference_oracle(self, instruction_string: str, num_variables: int) -> None:
        dag = StringToDAG(instruction_string, num_variables).run()
        _assert_descriptors_equal(describe_dag(dag), _reference_descriptor(dag))

    def test_variable_count_matches_declared_arity(self) -> None:
        dag = StringToDAG("V+NVs", 3).run()
        assert describe_dag(dag).n_var == 3


# ----------------------------------------------------------------------
# 2-3. Depth and structural invariants on random DAGs
# ----------------------------------------------------------------------


def _random_dag_corpus(n: int, *, seed: int, max_internal: int = 8) -> list[LabeledDAG]:
    """Return *n* random DAGs small enough for exponential path enumeration.

    Parameters
    ----------
    n : int
        Number of DAGs.
    seed : int
        Seed for the generator.
    max_internal : int
        Upper bound on the number of internal nodes.

    Returns
    -------
    list of LabeledDAG
        The generated corpus.
    """
    rng = random.Random(seed)
    return [
        _random_dag(
            rng,
            n_var=rng.randint(1, 3),
            n_internal=rng.randint(1, max_internal),
            n_extra_edges=rng.randint(0, 3),
        )
        for _ in range(n)
    ]


_DEPTH_CORPUS: Final[list[LabeledDAG]] = _random_dag_corpus(50, seed=20260807)


class TestDepth:
    @pytest.mark.parametrize("index", range(len(_DEPTH_CORPUS)))
    def test_depth_equals_brute_force_longest_path(self, index: int) -> None:
        dag = _DEPTH_CORPUS[index]
        assert describe_dag(dag).depth == _longest_path_edges(dag)

    @pytest.mark.parametrize("length", [1, 2, 5, 12, 40])
    def test_depth_of_a_pure_chain_is_its_edge_count(self, length: int) -> None:
        dag = LabeledDAG(max_nodes=length + 1)
        dag.add_node(NodeType.VAR, var_index=0)
        for i in range(length):
            dag.add_node(NodeType.NEG)
            dag.add_edge(i, i + 1)
        assert describe_dag(dag).depth == length

    def test_edgeless_dag_has_zero_depth(self) -> None:
        dag = LabeledDAG(max_nodes=3)
        for i in range(3):
            dag.add_node(NodeType.VAR, var_index=i)
        assert describe_dag(dag).depth == 0


class TestStructuralInvariants:
    @pytest.mark.parametrize("index", range(len(_DEPTH_CORPUS)))
    def test_node_partition_identities(self, index: int) -> None:
        desc = describe_dag(_DEPTH_CORPUS[index])
        assert desc.n_internal == desc.n_nodes - desc.n_var
        assert desc.n_op == desc.n_internal - desc.n_const
        assert desc.n_nodes == desc.n_var + desc.n_const + desc.n_op

    @pytest.mark.parametrize("index", range(len(_DEPTH_CORPUS)))
    def test_non_negativity_and_bounds(self, index: int) -> None:
        dag = _DEPTH_CORPUS[index]
        desc = describe_dag(dag)
        assert all(value >= 0 for value in desc)
        assert desc.n_var_used <= desc.n_var
        assert desc.n_nonlinear <= desc.n_op
        assert desc.n_distinct_op_labels <= desc.n_op
        assert desc.n_shared <= desc.sharing_surplus
        assert desc.sharing_surplus <= desc.n_edges
        assert desc.depth <= desc.n_nodes - 1

    @pytest.mark.parametrize("index", range(len(_DEPTH_CORPUS)))
    def test_matches_reference_oracle(self, index: int) -> None:
        dag = _DEPTH_CORPUS[index]
        _assert_descriptors_equal(describe_dag(dag), _reference_descriptor(dag))


# ----------------------------------------------------------------------
# 4. Operator-label entropy
# ----------------------------------------------------------------------


def _dag_from_operator_multiset(labels: Sequence[NodeType]) -> LabeledDAG:
    """Build a star DAG whose internal nodes carry exactly *labels*.

    One VAR feeds every internal node, so the operator histogram is exactly the
    multiset of *labels*.

    Parameters
    ----------
    labels : Sequence[NodeType]
        Internal node labels, in order.

    Returns
    -------
    LabeledDAG
        The generated DAG.
    """
    dag = LabeledDAG(max_nodes=len(labels) + 1)
    dag.add_node(NodeType.VAR, var_index=0)
    for offset, label in enumerate(labels):
        dag.add_node(label)
        dag.add_edge(0, offset + 1)
    return dag


class TestOperatorEntropy:
    @pytest.mark.parametrize("multiplicity", [1, 2, 5, 17])
    @pytest.mark.parametrize("label", [NodeType.ADD, NodeType.SIN, NodeType.POW])
    def test_single_distinct_operator_gives_exactly_zero(
        self, label: NodeType, multiplicity: int
    ) -> None:
        desc = describe_dag(_dag_from_operator_multiset([label] * multiplicity))
        assert desc.op_label_entropy == 0.0
        assert desc.n_distinct_op_labels == 1

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            (NodeType.ADD, NodeType.MUL),
            (NodeType.SIN, NodeType.COS),
            (NodeType.LOG, NodeType.EXP),
        ],
    )
    def test_two_operators_once_each_gives_exactly_one_bit(
        self, first: NodeType, second: NodeType
    ) -> None:
        desc = describe_dag(_dag_from_operator_multiset([first, second]))
        assert desc.op_label_entropy == pytest.approx(1.0, abs=1e-15)
        assert desc.n_distinct_op_labels == 2

    def test_no_operators_gives_zero(self) -> None:
        dag = LabeledDAG(max_nodes=3)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.CONST, const_value=1.0)
        dag.add_node(NodeType.CONST, const_value=2.0)
        desc = describe_dag(dag)
        assert desc.op_label_entropy == 0.0
        assert desc.n_distinct_op_labels == 0

    @pytest.mark.parametrize("trial", range(30))
    def test_matches_scipy_entropy(self, trial: int) -> None:
        rng = random.Random(1000 + trial)
        labels = [rng.choice(_OP_POOL) for _ in range(rng.randint(2, 20))]
        desc = describe_dag(_dag_from_operator_multiset(labels))
        counts = np.array(sorted(Counter(labels).values()), dtype=float)
        assert desc.op_label_entropy == pytest.approx(
            float(scipy_entropy(counts, base=2)), abs=1e-12
        )
        assert desc.n_distinct_op_labels == len(counts)

    def test_uniform_support_is_log2_of_support_size(self) -> None:
        labels = list(_OP_POOL[:8])
        desc = describe_dag(_dag_from_operator_multiset(labels))
        assert desc.op_label_entropy == pytest.approx(3.0, abs=1e-12)


# ----------------------------------------------------------------------
# 5. Empty DAG
# ----------------------------------------------------------------------


class TestEmptyDag:
    @pytest.mark.parametrize("capacity", [0, 1, 8])
    def test_empty_dag_returns_all_zero_descriptor(self, capacity: int) -> None:
        desc = describe_dag(LabeledDAG(max_nodes=capacity))
        assert desc == DagComplexity(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)
        assert all(value == 0 for value in desc)

    def test_empty_dag_label_counts_are_all_zero(self) -> None:
        _, label_counts = describe_dag_with_labels(LabeledDAG(max_nodes=4))
        assert label_counts == [0] * len(NodeType)

    def test_empty_dag_is_accumulable(self) -> None:
        acc = ComplexityAccumulator()
        acc.observe(LabeledDAG(max_nodes=0))
        assert acc.n == 1
        assert acc.mean("n_nodes") == 0.0


# ----------------------------------------------------------------------
# 6. Isomorphism invariance
# ----------------------------------------------------------------------

_INVARIANCE_BUILDERS: Final[tuple[tuple[str, LabeledDAG], ...]] = (
    ("sin_x", _build_sin_x()),
    ("shared_product_sum", _build_shared_product_sum()),
    ("two_constants", _build_two_constants()),
    ("shared_constant", _build_shared_constant()),
    ("unary_chain", _build_unary_chain()),
)


class TestIsomorphismInvariance:
    """A descriptor must not depend on how internal nodes happen to be numbered."""

    @pytest.mark.parametrize(
        "dag", [pytest.param(dag, id=name) for name, dag in _INVARIANCE_BUILDERS]
    )
    def test_all_permutations_give_an_identical_descriptor(self, dag: LabeledDAG) -> None:
        k = dag.node_count - len(dag.var_nodes())
        expected = describe_dag(dag)
        for perm in itertools.permutations(range(k)):
            permuted = permute_internal_nodes(dag, perm)
            assert describe_dag(permuted) == expected, f"descriptor changed under {perm}"

    @pytest.mark.parametrize(
        "dag", [pytest.param(dag, id=name) for name, dag in _INVARIANCE_BUILDERS]
    )
    def test_all_permutations_give_identical_label_counts(self, dag: LabeledDAG) -> None:
        k = dag.node_count - len(dag.var_nodes())
        _, expected = describe_dag_with_labels(dag)
        for perm in itertools.permutations(range(k)):
            _, counts = describe_dag_with_labels(permute_internal_nodes(dag, perm))
            assert counts == expected, f"label histogram changed under {perm}"

    @pytest.mark.parametrize("index", range(25))
    def test_random_dags_are_invariant_under_random_relabelling(self, index: int) -> None:
        rng = random.Random(5000 + index)
        dag = _random_dag(
            rng, n_var=rng.randint(1, 3), n_internal=rng.randint(2, 9), n_extra_edges=2
        )
        k = dag.node_count - len(dag.var_nodes())
        expected = describe_dag(dag)
        for _ in range(20):
            perm = list(range(k))
            rng.shuffle(perm)
            assert describe_dag(permute_internal_nodes(dag, perm)) == expected

    def test_identity_permutation_is_a_no_op(self) -> None:
        dag = _build_shared_product_sum()
        k = dag.node_count - len(dag.var_nodes())
        assert describe_dag(permute_internal_nodes(dag, list(range(k)))) == describe_dag(dag)


# ----------------------------------------------------------------------
# 7. ComplexityAccumulator
# ----------------------------------------------------------------------


def _empirical_quantile(values: Sequence[int], q: float) -> float:
    """Return the *q*-quantile of *values* under the lower-value convention.

    Reimplements the rule directly on the raw sample: the smallest non-negative
    integer ``v`` whose cumulative count reaches ``q * len(values)``.

    Parameters
    ----------
    values : Sequence[int]
        Raw non-negative integer observations.
    q : float
        Quantile in ``[0, 1]``.

    Returns
    -------
    float
        The quantile.
    """
    counts = Counter(values)
    target = q * len(values)
    cumulative = 0
    for value in range(max(values) + 1):
        cumulative += counts.get(value, 0)
        if cumulative >= target:
            return float(value)
    return float(max(values))


def _accumulator_corpus(n: int, *, seed: int) -> tuple[ComplexityAccumulator, list[DagComplexity]]:
    """Fold *n* random DAGs into an accumulator and also return their descriptors.

    Parameters
    ----------
    n : int
        Number of DAGs to fold in.
    seed : int
        Seed for the generator.

    Returns
    -------
    tuple of (ComplexityAccumulator, list of DagComplexity)
        The accumulator and the descriptors it was fed, in order.
    """
    rng = random.Random(seed)
    acc = ComplexityAccumulator()
    descriptors: list[DagComplexity] = []
    for _ in range(n):
        dag = _random_dag(
            rng,
            n_var=rng.randint(1, 4),
            n_internal=rng.randint(1, 20),
            n_extra_edges=rng.randint(0, 4),
        )
        descriptors.append(acc.observe(dag))
    return acc, descriptors


_ACC, _DESCRIPTORS = _accumulator_corpus(600, seed=424242)


class TestAccumulatorMoments:
    @pytest.mark.parametrize("field", DESCRIPTOR_FIELDS)
    def test_mean_matches_numpy(self, field: str) -> None:
        column = np.array([getattr(d, field) for d in _DESCRIPTORS], dtype=float)
        np.testing.assert_allclose(_ACC.mean(field), np.mean(column), rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("field", DESCRIPTOR_FIELDS)
    def test_std_matches_numpy_population_std(self, field: str) -> None:
        column = np.array([getattr(d, field) for d in _DESCRIPTORS], dtype=float)
        np.testing.assert_allclose(_ACC.std(field), np.std(column, ddof=0), rtol=1e-7, atol=1e-9)

    @pytest.mark.parametrize("field", DESCRIPTOR_FIELDS)
    @pytest.mark.parametrize("largest", [True, False])
    def test_extremum_matches_python_builtins(self, field: str, largest: bool) -> None:
        column = [getattr(d, field) for d in _DESCRIPTORS]
        expected = max(column) if largest else min(column)
        assert _ACC.extremum(field, largest=largest) == pytest.approx(float(expected))

    def test_sample_count(self) -> None:
        assert _ACC.n == len(_DESCRIPTORS) == 600

    def test_empty_accumulator_reports_nan_not_zero(self) -> None:
        acc = ComplexityAccumulator()
        assert math.isnan(acc.mean("n_nodes"))
        assert math.isnan(acc.std("n_nodes"))
        assert math.isnan(acc.extremum("n_nodes", largest=True))
        assert math.isnan(acc.quantile("depth", 0.5))

    def test_std_of_a_single_observation_is_nan(self) -> None:
        acc = ComplexityAccumulator()
        acc.observe(_build_sin_x())
        assert math.isnan(acc.std("n_nodes"))
        assert acc.mean("n_nodes") == 2.0

    def test_std_of_a_constant_stream_is_zero(self) -> None:
        acc = ComplexityAccumulator()
        for _ in range(50):
            acc.observe(_build_unary_chain())
        assert acc.std("n_internal") == pytest.approx(0.0, abs=1e-12)


class TestAccumulatorQuantiles:
    @pytest.mark.parametrize("field", HEADLINE_FIELDS)
    @pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.9, 0.99, 1.0])
    def test_quantile_matches_empirical_quantile(self, field: str, q: float) -> None:
        column = [getattr(d, field) for d in _DESCRIPTORS]
        assert max(column) <= 128, "corpus must stay below the histogram cap for exactness"
        assert _ACC.quantile(field, q) == pytest.approx(_empirical_quantile(column, q))

    def test_non_headline_field_is_not_histogrammed(self) -> None:
        with pytest.raises(KeyError):
            _ACC.quantile("n_shared", 0.5)


class TestAccumulatorSerialisation:
    def test_histogram_bins_and_overflow_sum_to_n(self) -> None:
        payload = _ACC.to_dict()
        histograms = payload["histograms"]
        assert isinstance(histograms, dict)
        for field in HEADLINE_FIELDS:
            entry = histograms[field]
            assert sum(entry["bins"]) + entry["overflow"] == _ACC.n

    def test_histogram_bins_match_the_raw_counts(self) -> None:
        histograms = _ACC.to_dict()["histograms"]
        assert isinstance(histograms, dict)
        for field in HEADLINE_FIELDS:
            counts = Counter(getattr(d, field) for d in _DESCRIPTORS)
            bins = histograms[field]["bins"]
            assert all(bins[value] == count for value, count in counts.items())

    def test_label_counts_total_equals_summed_node_counts(self) -> None:
        payload = _ACC.to_dict()
        label_counts = payload["label_counts"]
        assert isinstance(label_counts, dict)
        assert sum(label_counts.values()) == sum(d.n_nodes for d in _DESCRIPTORS)

    def test_label_counts_keys_are_node_type_names(self) -> None:
        label_counts = _ACC.to_dict()["label_counts"]
        assert isinstance(label_counts, dict)
        assert set(label_counts) == {node_type.name for node_type in NodeType}

    def test_moments_cover_every_descriptor(self) -> None:
        moments = _ACC.to_dict()["moments"]
        assert isinstance(moments, dict)
        assert set(moments) == set(DESCRIPTOR_FIELDS)
        assert set(moments["depth"]) == {"mean", "std", "min", "max", "sum"}

    def test_reported_sums_are_exact(self) -> None:
        moments = _ACC.to_dict()["moments"]
        assert isinstance(moments, dict)
        for field in ("n_nodes", "n_edges", "n_internal", "n_op"):
            expected = sum(getattr(d, field) for d in _DESCRIPTORS)
            assert moments[field]["sum"] == float(expected)


class TestAccumulatorOverflow:
    """A value above the histogram cap must land in, and be reported by, the overflow bin."""

    @staticmethod
    def _deep_chain(length: int) -> LabeledDAG:
        """Return a unary chain with *length* edges."""
        dag = LabeledDAG(max_nodes=length + 1)
        dag.add_node(NodeType.VAR, var_index=0)
        for i in range(length):
            dag.add_node(NodeType.NEG)
            dag.add_edge(i, i + 1)
        return dag

    def test_overflow_is_counted_and_reported(self) -> None:
        acc = ComplexityAccumulator()
        acc.observe(_build_shared_product_sum())  # depth 2
        acc.observe(self._deep_chain(70))  # depth 70 > cap 64
        histograms = acc.to_dict()["histograms"]
        assert isinstance(histograms, dict)
        assert histograms["depth"]["cap"] == 64
        assert histograms["depth"]["overflow"] == 1
        assert sum(histograms["depth"]["bins"]) == 1
        assert sum(histograms["depth"]["bins"]) + histograms["depth"]["overflow"] == acc.n
        # n_internal = 70 <= 128, so that histogram must be exact.
        assert histograms["n_internal"]["overflow"] == 0

    def test_extremum_is_exact_despite_overflow(self) -> None:
        acc = ComplexityAccumulator()
        acc.observe(_build_shared_product_sum())
        acc.observe(self._deep_chain(70))
        assert acc.extremum("depth", largest=True) == 70.0
        assert acc.mean("depth") == pytest.approx(36.0)

    def test_quantile_inside_the_cap_is_still_exact(self) -> None:
        acc = ComplexityAccumulator()
        acc.observe(_build_shared_product_sum())
        acc.observe(self._deep_chain(70))
        assert acc.quantile("depth", 0.5) == 2.0

    def test_quantile_falling_in_the_overflow_bin_reports_the_bin_index(self) -> None:
        # The docstring of ComplexityAccumulator.quantile says an overflowing
        # value "is reported as the cap"; the implementation returns the index of
        # the overflow bin, i.e. cap + 1. The value is therefore a sentinel that
        # is one above the cap rather than the cap itself. Asserted as
        # implemented, with the discrepancy recorded here.
        acc = ComplexityAccumulator()
        acc.observe(_build_shared_product_sum())
        acc.observe(self._deep_chain(70))
        assert acc.quantile("depth", 0.9) == 65.0


class TestAccumulatorUpdate:
    def test_update_without_label_counts_leaves_the_label_histogram_empty(self) -> None:
        acc = ComplexityAccumulator()
        acc.update(describe_dag(_build_two_constants()))
        assert acc.n == 1
        assert acc.label_counts == [0] * len(NodeType)
        assert acc.mean("n_const") == 2.0

    def test_observe_returns_the_folded_descriptor(self) -> None:
        acc = ComplexityAccumulator()
        dag = _build_shared_constant()
        assert acc.observe(dag) == describe_dag(dag)

    @pytest.mark.parametrize("n_repeats", [1, 2, 10])
    def test_repeated_observation_scales_the_label_histogram(self, n_repeats: int) -> None:
        acc = ComplexityAccumulator()
        dag = _build_two_constants()
        _, single = describe_dag_with_labels(dag)
        for _ in range(n_repeats):
            acc.observe(dag)
        assert acc.label_counts == [count * n_repeats for count in single]
