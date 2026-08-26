"""Tests for the fixed-order serialisation baselines and the HLL sketch.

The scientific contract under test is that each fixed-order serialisation is

* **sound** -- equal serialisations imply equal canonical strings, hence
  isomorphic DAGs; and
* **incomplete** -- there exist isomorphic DAGs (equal canonical strings) whose
  serialisations differ.

Incompleteness is the *expected* property, not a defect: it is exactly the gap
that the IsalSR canonical string closes, so the tests assert it rather than
work around it.

The corpus is the one built by ``measure_corpus_1()`` in
``experiments/scripts/measure_fallback_ledger_corpora.py``: 5,000 random IsalSR
strings per ``num_vars`` in {1, 2, 3} at ``seed=42`` and ``max_tokens=20``, with
all operations allowed, decoded through S2D and with VAR-only DAGs dropped.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Final

import numpy as np
import pytest

from isalsr.baselines import (
    FixedOrder,
    HyperLogLog,
    SerialisationError,
    deserialise,
    fixed_order_digest,
    fixed_order_hash,
    node_order,
    serialise,
)
from isalsr.core.canonical import fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import LABEL_CHAR_MAP, NodeType, OperationSet
from isalsr.core.permutations import permute_internal_nodes
from isalsr.core.string_to_dag import StringToDAG
from isalsr.search.random_search import random_isalsr_string

# Corpus 1 parameters, copied verbatim from measure_fallback_ledger_corpora.py.
_C1_N_STRINGS: Final[int] = 5_000
_C1_MAX_TOKENS: Final[int] = 20
_C1_SEED: Final[int] = 42
_C1_NUM_VARS: Final[tuple[int, ...]] = (1, 2, 3)
_EXPECTED_CORPUS_SIZE: Final[int] = 14_841

_BACKENDS: Final[tuple[str, ...]] = ("cpp", "python")
_ORDERS: Final[tuple[FixedOrder, ...]] = tuple(FixedOrder)


# ---------------------------------------------------------------------------
# Corpus fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus() -> list[LabeledDAG]:
    """Build Corpus 1: S2D-decoded random IsalSR strings."""
    ops = OperationSet(frozenset(LABEL_CHAR_MAP.values()))
    dags: list[LabeledDAG] = []
    for num_vars in _C1_NUM_VARS:
        rng = np.random.default_rng(_C1_SEED)
        for _ in range(_C1_N_STRINGS):
            string = random_isalsr_string(num_vars, _C1_MAX_TOKENS, ops, rng)
            dag = StringToDAG(string, num_vars, ops).run()
            if dag.node_count <= num_vars:
                continue
            dags.append(dag)
    return dags


@pytest.fixture(scope="module")
def canonicals(corpus: list[LabeledDAG]) -> dict[str, list[str]]:
    """Canonical strings of the corpus under both engine backends."""
    return {b: [fast_canonical_string(d, backend=b) for d in corpus] for b in _BACKENDS}


@pytest.fixture(scope="module")
def serialisations(corpus: list[LabeledDAG]) -> dict[FixedOrder, list[str]]:
    """Serialisations of the corpus under all three fixed orders."""
    return {order: [serialise(d, order) for d in corpus] for order in _ORDERS}


def test_corpus_size(corpus: list[LabeledDAG]) -> None:
    """The corpus reproduces the documented 14,841 DAGs."""
    assert len(corpus) == _EXPECTED_CORPUS_SIZE


def test_backends_agree(canonicals: dict[str, list[str]]) -> None:
    """The C++ and Python canonicalisers agree on every corpus DAG.

    Disagreement here is the tell for a stale native build, which would
    invalidate every soundness result below.
    """
    assert canonicals["cpp"] == canonicals["python"]


# ---------------------------------------------------------------------------
# 1. Soundness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", _ORDERS, ids=lambda o: o.name)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_soundness_on_corpus(
    order: FixedOrder,
    backend: str,
    serialisations: dict[FixedOrder, list[str]],
    canonicals: dict[str, list[str]],
) -> None:
    """Equal serialisation implies equal canonical string, on every rung.

    This is the empirical face of the soundness lemma: ``enc`` is injective, so
    a collision of ``sigma`` forces a label- and operand-order-preserving
    isomorphism, which the canonical string -- a complete labeled-DAG invariant
    -- must then report as equal.
    """
    seen: dict[str, str] = {}
    violations: list[tuple[str, str, str]] = []
    for ser, canon in zip(serialisations[order], canonicals[backend], strict=True):
        if ser in seen:
            if seen[ser] != canon:
                violations.append((ser, seen[ser], canon))
        else:
            seen[ser] = canon
    assert violations == [], f"{len(violations)} soundness violations on {order.name}/{backend}"


@pytest.mark.parametrize("order", _ORDERS, ids=lambda o: o.name)
def test_soundness_implies_isomorphism(
    order: FixedOrder,
    corpus: list[LabeledDAG],
    serialisations: dict[FixedOrder, list[str]],
) -> None:
    """Colliding corpus DAGs are isomorphic under the structural predicate.

    Checked directly with ``LabeledDAG.is_isomorphic`` rather than through the
    canonical string, so the two soundness tests do not share a failure mode.
    """
    representative: dict[str, LabeledDAG] = {}
    for dag, ser in zip(corpus, serialisations[order], strict=True):
        other = representative.get(ser)
        if other is None:
            representative[ser] = dag
        else:
            assert dag.is_isomorphic(other), f"collision on non-isomorphic pair: {ser}"


# ---------------------------------------------------------------------------
# 2. Incompleteness (the expected, desired property)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", _ORDERS, ids=lambda o: o.name)
def test_incompleteness_on_corpus(
    order: FixedOrder,
    serialisations: dict[FixedOrder, list[str]],
    canonicals: dict[str, list[str]],
) -> None:
    """Some canonical classes are split by the serialisation.

    Each split class is a set of isomorphic DAGs that the fixed-order hash fails
    to merge, i.e. redundant work that the rung leaves on the table.
    """
    by_canonical: dict[str, set[str]] = {}
    for ser, canon in zip(serialisations[order], canonicals["cpp"], strict=True):
        by_canonical.setdefault(canon, set()).add(ser)
    split = [c for c, sers in by_canonical.items() if len(sers) > 1]
    assert split, f"{order.name} merged every isomorphism class -- it would be complete"


@pytest.mark.parametrize("order", _ORDERS, ids=lambda o: o.name)
def test_incompleteness_under_relabelling(order: FixedOrder) -> None:
    """A permuted copy of a DAG has the same canonical string but a different
    serialisation.

    ``permute_internal_nodes`` produces an isomorphic copy by construction, so
    any difference in serialisation is precisely a tie broken by the original
    node index -- the non-equivariance that makes the rung incomplete.

    The example is ``sin(sin(x0)) + sin(x0)``.  All three ``SIN`` nodes share the
    tie-break key ``(label, in_degree, out_degree) = ("s", 1, 1)``, so the
    topological order must fall back to the node index; a DAG whose key already
    separates every node (``sin(x0) + cos(x0)``, say) would be permuted back into
    place by the topological order and would not exhibit the gap.
    """
    dag = _build_tied_dag()
    # Swap the middle and the shallow SIN; the deep SIN and the ADD stay put.
    permuted = permute_internal_nodes(dag, [0, 2, 1, 3])

    assert fast_canonical_string(dag) == fast_canonical_string(permuted)
    assert dag.is_isomorphic(permuted)
    assert serialise(dag, order) != serialise(permuted, order)


def test_topological_order_repairs_some_relabellings() -> None:
    """The topological order is equivariant when its key separates every node.

    On ``sin(x0) + cos(x0)`` the ``(label, ...)`` component of the key already
    distinguishes SIN from COS, so a swap of the two node indices is undone and
    the topological serialisations agree -- while the insertion order, which has
    no tie-break at all, still splits the pair.  This bounds how much of the
    residual gap each rung actually closes.
    """
    dag = _build_asymmetric_dag()
    permuted = permute_internal_nodes(dag, [1, 0, 2])

    assert fast_canonical_string(dag) == fast_canonical_string(permuted)
    assert serialise(dag, FixedOrder.INSERTION) != serialise(permuted, FixedOrder.INSERTION)
    assert serialise(dag, FixedOrder.TOPOLOGICAL) == serialise(permuted, FixedOrder.TOPOLOGICAL)


def _build_asymmetric_dag() -> LabeledDAG:
    """Build ``sin(x0) + cos(x0)``: two internal unary nodes plus a sum root."""
    dag = LabeledDAG(max_nodes=8)
    x0 = dag.add_node(NodeType.VAR, var_index=0)
    s = dag.add_node(NodeType.SIN)
    c = dag.add_node(NodeType.COS)
    root = dag.add_node(NodeType.ADD)
    dag.add_edge(x0, s)
    dag.add_edge(x0, c)
    dag.add_edge(s, root)
    dag.add_edge(c, root)
    return dag


def _build_tied_dag() -> LabeledDAG:
    """Build ``sin(sin(x0)) + sin(x0)``: three SIN nodes sharing one tie key."""
    dag = LabeledDAG(max_nodes=8)
    x0 = dag.add_node(NodeType.VAR, var_index=0)
    inner = dag.add_node(NodeType.SIN)
    outer = dag.add_node(NodeType.SIN)
    shallow = dag.add_node(NodeType.SIN)
    root = dag.add_node(NodeType.ADD)
    dag.add_edge(x0, inner)
    dag.add_edge(inner, outer)
    dag.add_edge(x0, shallow)
    dag.add_edge(outer, root)
    dag.add_edge(shallow, root)
    return dag


# ---------------------------------------------------------------------------
# 3. Round trip
# ---------------------------------------------------------------------------


def test_round_trip_is_structurally_exact(corpus: list[LabeledDAG]) -> None:
    """``deserialise(serialise(D, INSERTION))`` restores D node for node."""
    for dag in corpus:
        restored = deserialise(serialise(dag, FixedOrder.INSERTION))
        assert restored.node_count == dag.node_count
        assert restored.edge_count == dag.edge_count
        for node in range(dag.node_count):
            assert restored.node_label(node) == dag.node_label(node)
            assert restored.ordered_inputs(node) == dag.ordered_inputs(node)
            if dag.node_label(node) is NodeType.VAR:
                assert restored.node_data(node)["var_index"] == dag.node_data(node)["var_index"]


def test_round_trip_is_isomorphic_and_canonical(
    corpus: list[LabeledDAG], canonicals: dict[str, list[str]]
) -> None:
    """The round trip preserves isomorphism class and canonical string."""
    for dag, canon in zip(corpus, canonicals["cpp"], strict=True):
        restored = deserialise(serialise(dag, FixedOrder.INSERTION))
        assert restored.is_isomorphic(dag)
        assert fast_canonical_string(restored) == canon


@pytest.mark.parametrize("order", _ORDERS, ids=lambda o: o.name)
def test_round_trip_all_orders_preserve_canonical(
    order: FixedOrder, corpus: list[LabeledDAG], canonicals: dict[str, list[str]]
) -> None:
    """Decoding a topological serialisation yields a relabelling of the original.

    Node identifiers are not preserved for the topological orders, but the
    isomorphism class is, so the canonical string must be unchanged.
    """
    for dag, canon in zip(corpus, canonicals["cpp"], strict=True):
        restored = deserialise(serialise(dag, order))
        assert fast_canonical_string(restored) == canon


# ---------------------------------------------------------------------------
# 4. Rung monotonicity
# ---------------------------------------------------------------------------


def test_rung_monotonicity(
    serialisations: dict[FixedOrder, list[str]], canonicals: dict[str, list[str]]
) -> None:
    """Distinct counts decrease monotonically down the ladder."""
    n_insertion = len(set(serialisations[FixedOrder.INSERTION]))
    n_topological = len(set(serialisations[FixedOrder.TOPOLOGICAL]))
    n_commutative = len(set(serialisations[FixedOrder.TOPOLOGICAL_COMMUTATIVE]))
    n_canonical = len(set(canonicals["cpp"]))

    assert n_insertion >= n_topological >= n_commutative >= n_canonical


# ---------------------------------------------------------------------------
# CONST values must not be serialised
# ---------------------------------------------------------------------------


def _const_dag(value: float) -> LabeledDAG:
    """Build ``x0 + c`` with the given constant value."""
    dag = LabeledDAG(max_nodes=4)
    x0 = dag.add_node(NodeType.VAR, var_index=0)
    const = dag.add_node(NodeType.CONST, const_value=value)
    root = dag.add_node(NodeType.ADD)
    dag.add_edge(x0, const)
    dag.add_edge(x0, root)
    dag.add_edge(const, root)
    return dag


@pytest.mark.parametrize("value", [0.0, 1.0, -3.5, 1e-12, 1e12])
def test_const_value_is_not_serialised(value: float) -> None:
    """DAGs differing only in a CONST value have equal canonical AND serialisation.

    The canonical string is over labels only, so IsalSR already merges these two
    DAGs.  If the serialisation encoded the value it would be finer than IsalSR
    for a reason unrelated to isomorphism and would confound the distinct-count
    comparison.
    """
    reference = _const_dag(2.0)
    other = _const_dag(value)
    assert fast_canonical_string(reference) == fast_canonical_string(other)
    for order in _ORDERS:
        assert serialise(reference, order) == serialise(other, order)
        assert fixed_order_digest(reference, order) == fixed_order_digest(other, order)


def test_const_value_is_lost_on_round_trip() -> None:
    """The decoder cannot recover a CONST value; it must not invent one."""
    restored = deserialise(serialise(_const_dag(7.25), FixedOrder.INSERTION))
    const_nodes = [
        i for i in range(restored.node_count) if restored.node_label(i) is NodeType.CONST
    ]
    assert len(const_nodes) == 1
    assert "const_value" not in restored.node_data(const_nodes[0])


# ---------------------------------------------------------------------------
# Operand order and the commutative local sort
# ---------------------------------------------------------------------------


def _binary_dag(label: NodeType, first: int, second: int) -> LabeledDAG:
    """Build a two-variable DAG applying *label* to (x_first, x_second)."""
    dag = LabeledDAG(max_nodes=4)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    root = dag.add_node(label)
    dag.add_edge(first, root)
    dag.add_edge(second, root)
    return dag


@pytest.mark.parametrize("label", [NodeType.POW, NodeType.SUB, NodeType.DIV])
@pytest.mark.parametrize("order", _ORDERS, ids=lambda o: o.name)
def test_operand_order_is_preserved(label: NodeType, order: FixedOrder) -> None:
    """Swapping the operands of a non-commutative binary op changes the encoding.

    Uses ``ordered_inputs`` (Critical Invariant 8).  ``sorted(in_neighbors())``
    would make both DAGs encode identically and the baseline would be lossy.
    Legacy SUB and DIV are covered even though the production alphabet is
    decomposed, because legacy corpora may still contain them.
    """
    forward = _binary_dag(label, 0, 1)
    reversed_ = _binary_dag(label, 1, 0)
    assert serialise(forward, order) != serialise(reversed_, order)
    assert fast_canonical_string(forward) != fast_canonical_string(reversed_)


@pytest.mark.parametrize("label", [NodeType.ADD, NodeType.MUL])
def test_commutative_local_sort_merges_operand_permutations(label: NodeType) -> None:
    """ADD/MUL operand permutations merge only under TOPOLOGICAL_COMMUTATIVE."""
    forward = _binary_dag(label, 0, 1)
    reversed_ = _binary_dag(label, 1, 0)

    assert fast_canonical_string(forward) == fast_canonical_string(reversed_)
    assert serialise(forward, FixedOrder.INSERTION) != serialise(reversed_, FixedOrder.INSERTION)
    assert serialise(forward, FixedOrder.TOPOLOGICAL) != serialise(
        reversed_, FixedOrder.TOPOLOGICAL
    )
    assert serialise(forward, FixedOrder.TOPOLOGICAL_COMMUTATIVE) == serialise(
        reversed_, FixedOrder.TOPOLOGICAL_COMMUTATIVE
    )


@pytest.mark.parametrize("label", [NodeType.POW, NodeType.SUB, NodeType.DIV])
def test_commutative_local_sort_does_not_touch_binary_ops(label: NodeType) -> None:
    """The local sort must not reorder the inputs of non-commutative ops."""
    forward = _binary_dag(label, 0, 1)
    reversed_ = _binary_dag(label, 1, 0)
    assert serialise(forward, FixedOrder.TOPOLOGICAL_COMMUTATIVE) != serialise(
        reversed_, FixedOrder.TOPOLOGICAL_COMMUTATIVE
    )


def test_commutative_sort_does_not_recurse() -> None:
    """The local sort changes emitted inputs only, never the node ordering."""
    dag = _build_asymmetric_dag()
    assert node_order(dag, FixedOrder.TOPOLOGICAL) == node_order(
        dag, FixedOrder.TOPOLOGICAL_COMMUTATIVE
    )


# ---------------------------------------------------------------------------
# Order properties and edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", _ORDERS, ids=lambda o: o.name)
def test_node_order_is_a_permutation(order: FixedOrder, corpus: list[LabeledDAG]) -> None:
    """Every order is total: it lists each node exactly once."""
    for dag in corpus[:500]:
        nodes = node_order(dag, order)
        assert sorted(nodes) == list(range(dag.node_count))


def test_topological_order_respects_edges(corpus: list[LabeledDAG]) -> None:
    """For every edge u -> v, u precedes v in the topological order."""
    for dag in corpus[:500]:
        position = {node: i for i, node in enumerate(node_order(dag, FixedOrder.TOPOLOGICAL))}
        for target in range(dag.node_count):
            for source in dag.ordered_inputs(target):
                assert position[source] < position[target]


@pytest.mark.parametrize("order", _ORDERS, ids=lambda o: o.name)
def test_serialise_is_deterministic(order: FixedOrder, corpus: list[LabeledDAG]) -> None:
    """Repeated calls on the same DAG return byte-identical strings."""
    for dag in corpus[:200]:
        assert serialise(dag, order) == serialise(dag, order)


@pytest.mark.parametrize("order", _ORDERS, ids=lambda o: o.name)
def test_empty_dag(order: FixedOrder) -> None:
    """A zero-node DAG serialises and decodes without special-casing."""
    dag = LabeledDAG(max_nodes=4)
    text = serialise(dag, order)
    assert text == "0|"
    assert deserialise(text).node_count == 0


@pytest.mark.parametrize("order", _ORDERS, ids=lambda o: o.name)
def test_single_variable_dag(order: FixedOrder) -> None:
    """A single VAR node round-trips and keeps its variable index."""
    dag = LabeledDAG(max_nodes=2)
    dag.add_node(NodeType.VAR, var_index=3)
    restored = deserialise(serialise(dag, order))
    assert restored.node_count == 1
    assert restored.node_label(0) is NodeType.VAR
    assert restored.node_data(0)["var_index"] == 3


@pytest.mark.parametrize("order", _ORDERS, ids=lambda o: o.name)
def test_edgeless_dag(order: FixedOrder) -> None:
    """Isolated nodes with no edges are handled."""
    dag = LabeledDAG(max_nodes=4)
    dag.add_node(NodeType.VAR, var_index=0)
    dag.add_node(NodeType.VAR, var_index=1)
    restored = deserialise(serialise(dag, order))
    assert restored.node_count == 2
    assert restored.edge_count == 0


def test_variables_are_distinguishable() -> None:
    """x_0 and x_1 must not collide."""
    a = LabeledDAG(max_nodes=2)
    a.add_node(NodeType.VAR, var_index=0)
    b = LabeledDAG(max_nodes=2)
    b.add_node(NodeType.VAR, var_index=1)
    for order in _ORDERS:
        assert serialise(a, order) != serialise(b, order)


@pytest.mark.parametrize(
    "text",
    ["", "3", "2|x0<>", "1|x0<", "1|?<>", "1|x0<5>", "abc|x0<>", "1|xz<>"],
)
def test_deserialise_rejects_malformed_input(text: str) -> None:
    """The decoder raises rather than silently building a wrong DAG."""
    with pytest.raises(SerialisationError):
        deserialise(text)


def test_unknown_order_rejected() -> None:
    """``node_order`` rejects a value that is not a FixedOrder member."""
    dag = _build_asymmetric_dag()
    with pytest.raises(SerialisationError):
        node_order(dag, "topological")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", _ORDERS, ids=lambda o: o.name)
def test_hash_matches_builtin_hash_of_serialisation(
    order: FixedOrder, corpus: list[LabeledDAG]
) -> None:
    """The live key is exactly ``hash(serialisation)``, matching the IsalSR set."""
    for dag in corpus[:200]:
        assert fixed_order_hash(dag, order) == hash(serialise(dag, order))


@pytest.mark.parametrize("order", _ORDERS, ids=lambda o: o.name)
def test_digest_is_64_bit_unsigned(order: FixedOrder, corpus: list[LabeledDAG]) -> None:
    """The stable digest fits in an unsigned 64-bit word."""
    for dag in corpus[:200]:
        digest = fixed_order_digest(dag, order)
        assert 0 <= digest < 2**64


@pytest.mark.parametrize("order", _ORDERS, ids=lambda o: o.name)
def test_digest_collision_rate_on_corpus(
    order: FixedOrder, serialisations: dict[FixedOrder, list[str]], corpus: list[LabeledDAG]
) -> None:
    """The digest introduces no collisions on the corpus."""
    distinct_serialisations = len(set(serialisations[order]))
    distinct_digests = len({fixed_order_digest(d, order) for d in corpus})
    assert distinct_digests == distinct_serialisations


def test_digest_is_stable_across_processes() -> None:
    """BLAKE2b digests are reproducible under different PYTHONHASHSEED values.

    Builtin ``hash`` is SipHash and is randomised per process, so it cannot be
    persisted; this is why a second, stable hash function exists.
    """
    snippet = (
        "from isalsr.baselines import FixedOrder, fixed_order_digest, fixed_order_hash;"
        "from isalsr.core.labeled_dag import LabeledDAG;"
        "from isalsr.core.node_types import NodeType;"
        "d = LabeledDAG(max_nodes=4);"
        "x = d.add_node(NodeType.VAR, var_index=0);"
        "s = d.add_node(NodeType.SIN);"
        "d.add_edge(x, s);"
        "print(fixed_order_digest(d, FixedOrder.INSERTION),"
        "      fixed_order_hash(d, FixedOrder.INSERTION))"
    )
    outputs: list[tuple[str, str]] = []
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        digest, live = result.stdout.split()
        outputs.append((digest, live))

    assert outputs[0][0] == outputs[1][0], "digest must be process-independent"
    assert outputs[0][1] != outputs[1][1], "builtin hash is expected to be randomised"


# ---------------------------------------------------------------------------
# HyperLogLog
# ---------------------------------------------------------------------------


def test_hll_empty_sketch_counts_zero() -> None:
    """An untouched sketch estimates zero distinct keys."""
    assert HyperLogLog(p=14).count() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("p", [3, 19, -1])
def test_hll_rejects_invalid_p(p: int) -> None:
    """Register counts outside the supported range are rejected."""
    with pytest.raises(ValueError):
        HyperLogLog(p=p)


def test_hll_register_footprint() -> None:
    """p=14 gives 16384 one-byte registers and ~0.81% relative standard error."""
    sketch = HyperLogLog(p=14)
    assert sketch.n_registers == 16_384
    np.testing.assert_allclose(sketch.relative_standard_error, 0.00813, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 10, 1_000])
def test_hll_small_cardinalities(n: int) -> None:
    """Linear counting keeps the small-range estimate close to truth."""
    sketch = HyperLogLog(p=14)
    for i in range(n):
        sketch.add(i)
    np.testing.assert_allclose(sketch.count(), float(n), rtol=0.05)


def test_hll_one_million_distinct_keys_within_two_percent() -> None:
    """On 10^6 distinct keys the estimate is within 2% of truth.

    HyperLogLog is a randomised estimator: at ``p = 14`` its relative standard
    error is ``1.04 / sqrt(16384) = 0.81 %``, so a 2 % bound on a *single*
    stream is only a 2.5-sigma assertion and fails on roughly 1 stream in 80.
    It does so for the specific stream ``range(10**6)``, which lands at
    ``+2.27 %`` (measured); over 20 independent streams the estimator's mean
    error is ``-0.20 %`` with standard deviation ``0.88 %``, i.e. unbiased and
    at its theoretical spread.

    The 2 % claim is therefore asserted where it is actually a claim about the
    estimator -- on the mean over independent replicates, whose standard error
    is ``0.81 % / sqrt(5) = 0.36 %`` -- and each individual replicate is held
    to the theoretical 4-sigma bound instead of a tuned one.
    """
    n = 1_000_000
    n_replicates = 5
    four_sigma = 4.0 * HyperLogLog(p=14).relative_standard_error

    errors: list[float] = []
    for replicate in range(n_replicates):
        sketch = HyperLogLog(p=14)
        base = replicate * 10_000_000_019
        for i in range(n):
            sketch.add(base + i)
        errors.append(sketch.count() / n - 1.0)

    assert max(abs(e) for e in errors) < four_sigma, f"replicate outside 4 sigma: {errors}"
    np.testing.assert_allclose(1.0 + float(np.mean(errors)), 1.0, rtol=0.02)


def test_hll_recovers_known_duplicate_ratio() -> None:
    """On a stream with a known duplicate ratio, n_total/count() recovers it.

    The stream has ``n_distinct`` keys each repeated ``repeats`` times, so the
    true redundancy ratio is exactly ``repeats``.
    """
    n_distinct = 200_000
    repeats = 4
    sketch = HyperLogLog(p=14)
    n_total = 0
    for _ in range(repeats):
        for i in range(n_distinct):
            sketch.add(i * 0x9E3779B1)
            n_total += 1
    np.testing.assert_allclose(n_total / sketch.count(), float(repeats), rtol=0.02)


def test_hll_is_insensitive_to_duplicates() -> None:
    """Re-adding the same keys does not change the estimate at all."""
    sketch = HyperLogLog(p=14)
    for i in range(50_000):
        sketch.add(i)
    first = sketch.count()
    for i in range(50_000):
        sketch.add(i)
    assert sketch.count() == first


def test_hll_merge_is_a_union() -> None:
    """Merging two disjoint sketches estimates the union cardinality."""
    left = HyperLogLog(p=14)
    right = HyperLogLog(p=14)
    for i in range(300_000):
        left.add(i)
    for i in range(200_000, 500_000):
        right.add(i)
    left.merge(right)
    np.testing.assert_allclose(left.count(), 500_000.0, rtol=0.02)


def test_hll_merge_rejects_mismatched_p() -> None:
    """Sketches with different register counts cannot be merged."""
    with pytest.raises(ValueError):
        HyperLogLog(p=14).merge(HyperLogLog(p=12))


def test_hll_accepts_negative_and_large_keys() -> None:
    """Signed CPython hashes and 64-bit digests are both valid keys."""
    sketch = HyperLogLog(p=14)
    keys = [-(2**62), -1, 0, 1, 2**63 - 1, 2**63]
    assert len({k & (2**64 - 1) for k in keys}) == len(keys)
    for key in keys:
        sketch.add(key)
    np.testing.assert_allclose(sketch.count(), float(len(keys)), rtol=0.1)


def test_hll_keys_are_taken_modulo_two_to_the_64() -> None:
    """Keys are reduced mod 2**64, so -1 and 2**64 - 1 are the same key.

    A documented consequence of a 64-bit sketch, not a hash collision: the two
    intended key sources (signed ``hash`` output and unsigned 64-bit digests)
    are both exactly 64 bits wide, so the reduction is lossless for them.
    """
    sketch = HyperLogLog(p=14)
    sketch.add(-1)
    first = sketch.count()
    sketch.add(2**64 - 1)
    assert sketch.count() == first


def test_hll_on_fixed_order_digests(corpus: list[LabeledDAG]) -> None:
    """The sketch approximates the exact distinct-count of the corpus digests."""
    exact: set[int] = set()
    sketch = HyperLogLog(p=14)
    for dag in corpus:
        digest = fixed_order_digest(dag, FixedOrder.INSERTION)
        exact.add(digest)
        sketch.add(digest)
    np.testing.assert_allclose(sketch.count(), float(len(exact)), rtol=0.02)
