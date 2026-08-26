"""T18 -- canonical-string completeness against the first-operand designation.

Ticket ``.claude/notes/review/tasks/T18-canonical-completeness-operand-order.md``.

Gate 3 of ``experiments/scripts/equivalence_gate.py`` reported five DAGs out of
10,000 for which the round-trip property

    S2D(fast_canonical_string(D)) is isomorphic to D

failed under both engines. The cause is not the canonical string and not the
C++ port: it is that ``LabeledDAG.is_isomorphic`` compared the *entire*
``_input_order`` list of every non-commutative binary node, while Sigma_SR
encodes only the **first operand** of such a node.

``V``/``v`` may create a binary op only from ``ordered_inputs[0]`` (Critical
Invariant 8 / B9) -- enforced in ``dag_to_string._find_new_out_neighbor`` and at
the four ``ordered_inputs(c)[0] == ptr_in`` sites in ``canonical``. Every
further in-edge is emitted by ``C``/``c`` in canonical-traversal order, so its
position is not recoverable from the string.

The two notions coincide on well-formed DAGs: a binary op with in-degree two --
the only shape ``dag_evaluator`` accepts -- has its second operand forced once
the first is fixed and the edge set is preserved. They diverge only on
*over-saturated* binary nodes (in-degree > 2), which no evaluator accepts.

These tests fail against the pre-fix ``_check_operand_order``.
"""

from __future__ import annotations

import itertools
import random

import pytest

from isalsr.core.canonical import fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import BINARY_OPS, UNARY_OPS, VARIADIC_OPS, NodeType
from isalsr.core.permutations import permute_internal_nodes
from isalsr.core.string_to_dag import StringToDAG

# ---------------------------------------------------------------------------
# The five gate-3 counterexamples, verbatim from
# docs/md_files/changes/t18_completeness_counterexamples.md
# ---------------------------------------------------------------------------

T18_COUNTEREXAMPLES: list[tuple[int, int, str]] = [
    (2166, 2, "pv+vgVav^VavlCNCvrV^WNvgviv*Vsv+vkV*Vsv*cncvlWvk"),
    (2256, 2, "vcvlvrCV/v^V^nCvgvcnCpWvgVaPVgV-cv-nCNv/v-C"),
    (3687, 1, "CCVlcV/nvkVkV+viVsWPCVev/NVcVlvrCVavcvrpvavl"),
    (7403, 1, "ccv/VenWViviVlv*v-PcvgVkVgV-cVapv-vgccVlNVeWVaVa"),
    (7771, 1, "vapVgV^V+ppVkViVcv^ncNCVsNCVrvgNCvgVk"),
]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _oversaturated(op: NodeType, in_edge_order: tuple[int, ...]) -> LabeledDAG:
    """Build ``op(...)`` fed by SIN, COS and SQRT of x_0, in a chosen order.

    Node layout is identical in every instance -- 0=VAR, 1=SIN, 2=COS,
    3=SQRT, 4=*op* -- so two instances differ **only** in ``_input_order[4]``.
    ``op`` receives three in-edges, one more than its arity.
    """
    dag = LabeledDAG(5)
    x = dag.add_node(NodeType.VAR, var_index=0)
    s = dag.add_node(NodeType.SIN)
    c = dag.add_node(NodeType.COS)
    r = dag.add_node(NodeType.SQRT)
    top = dag.add_node(op)
    for leaf in (s, c, r):
        assert dag.add_edge(x, leaf)
    for pos in in_edge_order:
        assert dag.add_edge((s, c, r)[pos], top)
    return dag


def _binary_two_operands(op: NodeType, swapped: bool) -> LabeledDAG:
    """Build a *well-formed* ``op(sin(x), cos(x))`` or ``op(cos(x), sin(x))``."""
    dag = LabeledDAG(4)
    x = dag.add_node(NodeType.VAR, var_index=0)
    s = dag.add_node(NodeType.SIN)
    c = dag.add_node(NodeType.COS)
    top = dag.add_node(op)
    assert dag.add_edge(x, s)
    assert dag.add_edge(x, c)
    first, second = (c, s) if swapped else (s, c)
    assert dag.add_edge(first, top)
    assert dag.add_edge(second, top)
    return dag


# ---------------------------------------------------------------------------
# 1. The regression the ticket was opened on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("corpus_index", "num_vars", "source"),
    T18_COUNTEREXAMPLES,
    ids=[str(c[0]) for c in T18_COUNTEREXAMPLES],
)
def test_gate3_counterexample_round_trips(corpus_index: int, num_vars: int, source: str) -> None:
    """S2D(fcs(D)) must be isomorphic to D for all five gate-3 failures."""
    dag = StringToDAG(source, num_vars).run()
    reconstructed = StringToDAG(fast_canonical_string(dag), num_vars).run()
    assert dag.is_isomorphic(reconstructed), (
        f"corpus index {corpus_index}: round-trip still reported non-isomorphic"
    )


@pytest.mark.parametrize(
    ("corpus_index", "num_vars", "source"),
    T18_COUNTEREXAMPLES,
    ids=[str(c[0]) for c in T18_COUNTEREXAMPLES],
)
def test_gate3_counterexample_is_a_dedup_class_of_one(
    corpus_index: int, num_vars: int, source: str
) -> None:
    """Same canonical string and isomorphic: the merge is sound, not spurious."""
    dag = StringToDAG(source, num_vars).run()
    canon = fast_canonical_string(dag)
    reconstructed = StringToDAG(canon, num_vars).run()
    assert fast_canonical_string(reconstructed) == canon
    assert dag.is_isomorphic(reconstructed)


# ---------------------------------------------------------------------------
# 2. B9 must not be weakened: the first operand still separates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", [NodeType.SUB, NodeType.DIV, NodeType.POW])
def test_well_formed_operand_swap_still_separated(op: NodeType) -> None:
    """op(sin x, cos x) and op(cos x, sin x) stay distinct (B9)."""
    a = _binary_two_operands(op, swapped=False)
    b = _binary_two_operands(op, swapped=True)
    assert not a.is_isomorphic(b)
    assert fast_canonical_string(a) != fast_canonical_string(b)


@pytest.mark.parametrize("op", [NodeType.SUB, NodeType.DIV, NodeType.POW])
def test_oversaturated_first_operand_still_separated(op: NodeType) -> None:
    """A different *first* operand separates even when the node is over-saturated."""
    a = _oversaturated(op, (0, 1, 2))  # first operand = SIN
    b = _oversaturated(op, (1, 0, 2))  # first operand = COS
    assert a.ordered_inputs(4)[0] != b.ordered_inputs(4)[0]
    assert not a.is_isomorphic(b)
    assert fast_canonical_string(a) != fast_canonical_string(b)


# ---------------------------------------------------------------------------
# 3. The relaxation itself: surplus in-edge order carries no information
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", [NodeType.SUB, NodeType.DIV, NodeType.POW])
def test_oversaturated_surplus_order_is_not_structure(op: NodeType) -> None:
    """Positions >= 1 of an over-saturated binary node must not separate.

    The canonical string cannot distinguish these two DAGs, so neither may
    ``is_isomorphic`` -- otherwise the dedup count and rho disagree with the
    invariant they are computed from.
    """
    a = _oversaturated(op, (0, 1, 2))
    b = _oversaturated(op, (0, 2, 1))
    assert a.ordered_inputs(4) != b.ordered_inputs(4)
    assert fast_canonical_string(a) == fast_canonical_string(b)
    assert a.is_isomorphic(b)


def test_binary_and_unary_over_saturation_treated_alike() -> None:
    """No unprincipled asymmetry between node kinds.

    Surplus in-edge order was always ignored on unary nodes (the evaluator
    reads ``sorted(in_neighbors)`` there). It must be ignored on binary nodes
    too, for the same reason: the node is not evaluable at that in-degree.
    """
    unary_a = _oversaturated(NodeType.COS, (0, 1, 2))
    unary_b = _oversaturated(NodeType.COS, (0, 2, 1))
    binary_a = _oversaturated(NodeType.POW, (0, 1, 2))
    binary_b = _oversaturated(NodeType.POW, (0, 2, 1))
    assert unary_a.is_isomorphic(unary_b)
    assert binary_a.is_isomorphic(binary_b) == unary_a.is_isomorphic(unary_b)


# ---------------------------------------------------------------------------
# 4. Equivalence of the two notions on well-formed DAGs
# ---------------------------------------------------------------------------


def _bijections_preserving_labels_and_edges(a: LabeledDAG, b: LabeledDAG):
    """Yield every label-, var_index- and edge-preserving bijection a -> b.

    Brute force over label classes. Reference oracle only -- keep the DAGs small.
    """
    n = a.node_count
    if n != b.node_count or a.edge_count != b.edge_count:
        return
    a_edges = {(u, w) for u in range(n) for w in a.out_neighbors(u)}
    b_edges = {(u, w) for u in range(n) for w in b.out_neighbors(u)}

    def key(g: LabeledDAG, v: int) -> tuple[str, int | None, int, int]:
        return (
            g.node_label(v).name,
            g.node_data(v).get("var_index"),  # type: ignore[return-value]
            g.in_degree(v),
            g.out_degree(v),
        )

    a_cls: dict[tuple, list[int]] = {}
    b_cls: dict[tuple, list[int]] = {}
    for v in range(n):
        a_cls.setdefault(key(a, v), []).append(v)
        b_cls.setdefault(key(b, v), []).append(v)
    if {k: len(v) for k, v in a_cls.items()} != {k: len(v) for k, v in b_cls.items()}:
        return

    keys = sorted(a_cls, key=str)
    for combo in itertools.product(*(itertools.permutations(b_cls[k]) for k in keys)):
        sigma: dict[int, int] = {}
        for k, perm in zip(keys, combo, strict=True):
            for src, dst in zip(a_cls[k], perm, strict=True):
                sigma[src] = dst
        if {(sigma[u], sigma[w]) for (u, w) in a_edges} == b_edges:
            yield sigma


def _iso_reference(a: LabeledDAG, b: LabeledDAG, *, whole_list: bool) -> bool:
    """Reference isomorphism oracle, parameterised by the operand-order rule.

    ``whole_list=True`` reproduces the pre-T18 rule (compare all of
    ``ordered_inputs`` on binary nodes); ``False`` is the current rule (compare
    position 0 only).
    """
    for sigma in _bijections_preserving_labels_and_edges(a, b):
        ok = True
        for v in range(a.node_count):
            if a.node_label(v) not in BINARY_OPS:
                continue
            oi_a = a.ordered_inputs(v)
            oi_b = b.ordered_inputs(sigma[v])
            if not oi_a:
                continue
            compare = oi_a if whole_list else oi_a[:1]
            if [sigma[x] for x in compare] != list(oi_b[: len(compare)]):
                ok = False
                break
        if ok:
            return True
    return False


def _random_small_dag(rng: random.Random, k: int, m: int) -> LabeledDAG:
    """Random DAG whose binary nodes all have in-degree <= 2.

    Edges only ever run from a lower node id to a higher one, so acyclicity is
    free (Critical Invariant 6). Operand edges are added in a deliberate order,
    which is what ``_input_order`` records (Invariant 8).
    """
    ops = sorted(UNARY_OPS | BINARY_OPS | VARIADIC_OPS, key=lambda t: t.name)
    dag = LabeledDAG(m + k)
    for i in range(m):
        dag.add_node(NodeType.VAR, var_index=i)
    for _ in range(k):
        label = rng.choice(ops)
        v = dag.add_node(label)
        arity = 1 if label in UNARY_OPS else 2
        sources = rng.sample(range(v), min(arity, v))
        rng.shuffle(sources)
        for s in sources:
            dag.add_edge(s, v)
    return dag


def test_relaxation_is_the_identity_when_binary_indegree_is_at_most_two() -> None:
    """The T18 lemma, checked against a brute-force reference oracle.

    On any DAG whose binary nodes have in-degree <= 2, comparing the first
    operand and comparing the whole ``ordered_inputs`` list define the *same*
    relation -- so the T18 relaxation cannot move any number computed on such a
    corpus, which includes everything either host adapter emits.

    Also pins ``LabeledDAG.is_isomorphic`` to the reference oracle.
    """
    rng = random.Random(20260803)
    dags = [_random_small_dag(rng, k=rng.randint(3, 6), m=rng.randint(1, 2)) for _ in range(120)]
    for d in dags:
        for v in range(d.node_count):
            if d.node_label(v) in BINARY_OPS:
                assert d.in_degree(v) <= 2

    # Distinct DAGs (mostly negative cases) plus every DAG against a permuted
    # copy of itself (guaranteed positive cases, so the test is not vacuous).
    pairs: list[tuple[LabeledDAG, LabeledDAG]] = [
        (a, b)
        for a, b in itertools.combinations(dags, 2)
        if (a.node_count, a.edge_count) == (b.node_count, b.edge_count)
    ]
    for d in dags:
        perm = list(range(d.node_count - len(d.var_nodes())))
        rng.shuffle(perm)
        pairs.append((d, permute_internal_nodes(d, perm)))

    compared = 0
    positives = 0
    for a, b in pairs:
        compared += 1
        first_only = _iso_reference(a, b, whole_list=False)
        whole_list = _iso_reference(a, b, whole_list=True)
        assert first_only == whole_list, "lemma violated: the two rules disagree"
        assert a.is_isomorphic(b) == first_only, "is_isomorphic disagrees with the reference oracle"
        positives += int(first_only)

    assert compared >= 100, f"corpus too small to be meaningful: {compared} comparable pairs"
    assert positives >= 100, f"too few isomorphic pairs to be meaningful: {positives}"


def test_reference_oracle_separates_an_over_saturated_binary_node() -> None:
    """Anti-vacuity: the two rules DO disagree once a binary node is over-saturated.

    Without this, the lemma test above could pass by never exercising the case
    the relaxation was written for.
    """
    a = _oversaturated(NodeType.POW, (0, 1, 2))
    b = _oversaturated(NodeType.POW, (0, 2, 1))
    assert _iso_reference(a, b, whole_list=False) is True
    assert _iso_reference(a, b, whole_list=True) is False
    assert a.is_isomorphic(b) is True


@pytest.mark.parametrize("op", [NodeType.SUB, NodeType.DIV, NodeType.POW])
def test_first_operand_forces_second_when_arity_is_met(op: NodeType) -> None:
    """With in-degree exactly 2, agreement at position 0 forces position 1.

    This is why restricting the check to the first operand loses no strength on
    any DAG the evaluator accepts.
    """
    for swapped in (False, True):
        dag = _binary_two_operands(op, swapped=swapped)
        reconstructed = StringToDAG(fast_canonical_string(dag), 1).run()
        assert dag.is_isomorphic(reconstructed)
        assert len(dag.ordered_inputs(3)) == 2
