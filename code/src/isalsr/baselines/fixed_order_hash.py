"""Fixed-order serialisation baselines for labeled DAG deduplication.

This module provides the *rungs* of the deduplication ladder against which the
IsalSR canonical string is compared.  Each rung fixes a total order on the node
set of a :class:`~isalsr.core.labeled_dag.LabeledDAG` and emits an injective
encoding of the relabelled graph.  Two DAGs collide on a rung iff their
serialisations are byte-identical.

Soundness lemma
---------------
Let ``pi_D`` assign to each labeled DAG ``D`` a bijection ``V(D) -> {0..n-1}``,
and let ``enc`` be an injective encoding of labeled DAGs on ``{0..n-1}`` with
ordered in-edges.  Then ``sigma = enc o pi`` satisfies
``sigma(D) = sigma(D') => D ~= D'``: injectivity of ``enc`` gives
``pi_D(D) = pi_D'(D')`` as labeled digraphs, so ``pi_D'^{-1} o pi_D`` is a
label- and operand-order-preserving isomorphism.

Completeness would additionally require ``pi`` to be isomorphism-equivariant.
None of the three orders below is equivariant, because every one falls back to
the original node index to break ties, and the original node index is exactly
what an isomorphism permutes.  The rungs are therefore **sound but incomplete**:
they never merge non-isomorphic DAGs, and they fail to merge isomorphic DAGs
whose numbering differs.  The residual gap to the IsalSR canonical string is
precisely the set of index-broken ties.

The three orders
----------------
``FixedOrder.INSERTION``
    Nodes in their existing ``LabeledDAG`` index order (identity permutation).

``FixedOrder.TOPOLOGICAL``
    A topological order (edge ``u -> v`` puts ``u`` before ``v``), ties broken by
    ``(label, in_degree, out_degree)`` and remaining ties by original node index.
    Deterministic and total.

``FixedOrder.TOPOLOGICAL_COMMUTATIVE``
    Identical node ordering to ``TOPOLOGICAL``; when emitting the input list of a
    ``NodeType.ADD`` or ``NodeType.MUL`` node *only*, the emitted positions are
    sorted ascending.  This is a purely local sort at one node: it does not
    recurse and it does not change the node ordering.

What is serialised, and what is not
-----------------------------------
Emitted: the ``NodeType`` label of every node, the variable index of every
``NodeType.VAR`` node (variables are pre-numbered and distinguishable), and the
in-edge list of every node in ``ordered_inputs`` order (Critical Invariant 8 --
using ``sorted(in_neighbors())`` here would silently discard the operand order of
``POW``/``SUB``/``DIV`` and make the baseline lossy).

**Not** emitted: ``const_value``.  The IsalSR canonical string is over labels
only, so it merges two DAGs that differ only in the numeric value of a ``CONST``
node.  A serialisation that encoded constant values would be finer than IsalSR
for a reason unrelated to isomorphism and would confound the comparison of
distinct-count ratios.

Restriction: this module depends only on the Python standard library and
``isalsr.core``.
"""

from __future__ import annotations

import hashlib
import heapq
from enum import Enum

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import LABEL_CHAR_MAP, NODE_TYPE_TO_LABEL, NodeType

__all__ = [
    "FixedOrder",
    "SerialisationError",
    "node_order",
    "serialise",
    "deserialise",
    "fixed_order_hash",
    "fixed_order_digest",
]

# Structural separators.  None of them occurs in ``LABEL_CHAR_MAP`` keys
# ("+*-/sceLr^agik") nor in decimal digits, which is what makes the grammar
# below unambiguously parseable and hence ``enc`` injective.
_HEADER_SEP = "|"
_RECORD_SEP = ";"
_INPUTS_OPEN = "<"
_INPUTS_CLOSE = ">"
_INPUT_SEP = ","
_VAR_PREFIX = "x"

_COMMUTATIVE_EMIT: frozenset[NodeType] = frozenset({NodeType.ADD, NodeType.MUL})


class SerialisationError(ValueError):
    """Raised when a DAG cannot be serialised or a string cannot be parsed."""


class FixedOrder(Enum):
    """The fixed node orders that define the deduplication rungs."""

    INSERTION = "insertion"
    TOPOLOGICAL = "topological"
    TOPOLOGICAL_COMMUTATIVE = "topological_commutative"


def _static_tie_key(dag: LabeledDAG, node: int) -> tuple[str, int, int, int]:
    """Return the deterministic tie-break key for *node*.

    The key is ``(label, in_degree, out_degree, node)``.  The trailing node index
    makes the key unique, so the resulting order is total; it is also the reason
    the order is not isomorphism-equivariant.

    Args:
        dag: The DAG owning *node*.
        node: Node identifier.

    Returns:
        The tie-break key.
    """
    return (
        dag.node_label(node).value,
        dag.in_degree(node),
        dag.out_degree(node),
        node,
    )


def node_order(dag: LabeledDAG, order: FixedOrder) -> list[int]:
    """Return the node identifiers of *dag* in the order induced by *order*.

    Args:
        dag: The DAG to order.
        order: Which fixed order to apply.

    Returns:
        A permutation of ``range(dag.node_count)``.

    Raises:
        SerialisationError: If *dag* contains a cycle, or *order* is unknown.
    """
    n = dag.node_count
    if order is FixedOrder.INSERTION:
        return list(range(n))
    if order in (FixedOrder.TOPOLOGICAL, FixedOrder.TOPOLOGICAL_COMMUTATIVE):
        return _topological_order(dag)
    raise SerialisationError(f"Unknown fixed order: {order!r}")


def _topological_order(dag: LabeledDAG) -> list[int]:
    """Return a deterministic topological order of *dag*.

    Kahn's algorithm driven by a min-heap on ``(label, in_degree, out_degree,
    node)``, where the degrees are those of the *full* graph, not the residual
    ones.  Because the key ends in the node index it is unique, so the heap
    never has to break a tie arbitrarily.

    Args:
        dag: The DAG to order.

    Returns:
        A topological order of ``range(dag.node_count)``.

    Raises:
        SerialisationError: If *dag* contains a cycle.
    """
    n = dag.node_count
    remaining_in = [dag.in_degree(i) for i in range(n)]
    heap: list[tuple[str, int, int, int]] = [
        _static_tie_key(dag, i) for i in range(n) if remaining_in[i] == 0
    ]
    heapq.heapify(heap)

    order: list[int] = []
    while heap:
        node = heapq.heappop(heap)[3]
        order.append(node)
        for target in sorted(dag.out_neighbors(node)):
            remaining_in[target] -= 1
            if remaining_in[target] == 0:
                heapq.heappush(heap, _static_tie_key(dag, target))

    if len(order) != n:
        raise SerialisationError(
            f"Graph contains a cycle: topological order produced {len(order)}/{n} nodes"
        )
    return order


def _node_tag(dag: LabeledDAG, node: int) -> str:
    """Return the label token emitted for *node*.

    ``VAR`` nodes emit ``x<var_index>``; every other type emits its single label
    character.  Constant values are deliberately omitted.

    Args:
        dag: The DAG owning *node*.
        node: Node identifier.

    Returns:
        The label token.

    Raises:
        SerialisationError: If a ``VAR`` node carries no ``var_index``, or the
            label has no character encoding.
    """
    label = dag.node_label(node)
    if label is NodeType.VAR:
        data = dag.node_data(node)
        if "var_index" not in data:
            raise SerialisationError(f"VAR node {node} has no var_index")
        return f"{_VAR_PREFIX}{int(data['var_index'])}"
    char = NODE_TYPE_TO_LABEL.get(label)
    if char is None:
        raise SerialisationError(f"Node {node} has label {label!r} with no character encoding")
    return char


def serialise(dag: LabeledDAG, order: FixedOrder = FixedOrder.INSERTION) -> str:
    """Serialise *dag* under the fixed node order *order*.

    The grammar is ``<n>|<record>;<record>;...`` with one record per node in
    emitted order, each record being ``<tag><i0,i1,...>`` where the ``i`` are the
    *emitted positions* of the node's inputs, in ``ordered_inputs`` order (or
    ascending, for ``ADD``/``MUL`` under
    :attr:`FixedOrder.TOPOLOGICAL_COMMUTATIVE`).

    Args:
        dag: The DAG to serialise.
        order: Which fixed order to apply.

    Returns:
        The serialisation string.

    Raises:
        SerialisationError: If *dag* cannot be encoded (cycle, unlabelled VAR,
            unknown label).
    """
    nodes = node_order(dag, order)
    position: dict[int, int] = {node: idx for idx, node in enumerate(nodes)}
    commutative_emit = order is FixedOrder.TOPOLOGICAL_COMMUTATIVE

    records: list[str] = []
    for node in nodes:
        inputs = [position[src] for src in dag.ordered_inputs(node)]
        if commutative_emit and dag.node_label(node) in _COMMUTATIVE_EMIT:
            inputs.sort()
        joined = _INPUT_SEP.join(str(i) for i in inputs)
        records.append(f"{_node_tag(dag, node)}{_INPUTS_OPEN}{joined}{_INPUTS_CLOSE}")

    return f"{len(nodes)}{_HEADER_SEP}{_RECORD_SEP.join(records)}"


def deserialise(text: str) -> LabeledDAG:
    """Reconstruct a labeled DAG from a serialisation produced by :func:`serialise`.

    The reconstruction is exact for :attr:`FixedOrder.INSERTION`: node
    identifiers, labels, variable indices and ``ordered_inputs`` lists are
    restored verbatim.  For the topological orders the result is a relabelling of
    the original, hence isomorphic to it but not identical.  ``CONST`` values are
    not recoverable because they are not emitted; reconstructed ``CONST`` nodes
    carry no ``const_value``.

    Args:
        text: A string produced by :func:`serialise`.

    Returns:
        The reconstructed DAG.

    Raises:
        SerialisationError: If *text* does not conform to the grammar.
    """
    header, sep, body = text.partition(_HEADER_SEP)
    if not sep:
        raise SerialisationError(f"Missing header separator in {text!r}")
    try:
        n = int(header)
    except ValueError as exc:
        raise SerialisationError(f"Invalid node count {header!r}") from exc

    records = body.split(_RECORD_SEP) if n > 0 else []
    if len(records) != n:
        raise SerialisationError(f"Expected {n} records, got {len(records)}")

    dag = LabeledDAG(max_nodes=max(n, 1))
    parsed_inputs: list[list[int]] = []

    for record in records:
        tag, open_sep, rest = record.partition(_INPUTS_OPEN)
        if not open_sep or not rest.endswith(_INPUTS_CLOSE):
            raise SerialisationError(f"Malformed record {record!r}")
        inner = rest[: -len(_INPUTS_CLOSE)]
        try:
            inputs = [int(tok) for tok in inner.split(_INPUT_SEP)] if inner else []
        except ValueError as exc:
            raise SerialisationError(f"Malformed input list in {record!r}") from exc
        if any(i < 0 or i >= n for i in inputs):
            raise SerialisationError(f"Input position out of range in {record!r}")
        parsed_inputs.append(inputs)

        if tag.startswith(_VAR_PREFIX):
            try:
                var_index = int(tag[len(_VAR_PREFIX) :])
            except ValueError as exc:
                raise SerialisationError(f"Malformed VAR tag {tag!r}") from exc
            dag.add_node(NodeType.VAR, var_index=var_index)
        else:
            label = LABEL_CHAR_MAP.get(tag)
            if label is None:
                raise SerialisationError(f"Unknown label token {tag!r}")
            dag.add_node(label)

    # Edges are added target-by-target so that ``_input_order`` reproduces the
    # emitted operand order exactly (Critical Invariant 8).  Acyclicity is
    # guaranteed by the source DAG, so the unchecked insertion is safe and keeps
    # the decoder O(V + E).
    for target, inputs in enumerate(parsed_inputs):
        for source in inputs:
            dag.add_edge_unchecked(source, target)

    return dag


def fixed_order_hash(dag: LabeledDAG, order: FixedOrder = FixedOrder.INSERTION) -> int:
    """Return the CPython builtin hash of the fixed-order serialisation.

    This is the **live** deduplication key.  It uses ``hash(str)`` so that its
    memory footprint and collision profile match the IsalSR deduplication set
    exactly, which stores ``hash(canonical_string)``.  Builtin ``hash`` is
    SipHash and is randomised per process by ``PYTHONHASHSEED``; the value
    therefore cannot be persisted or replayed across processes.  Use
    :func:`fixed_order_digest` for that.

    Args:
        dag: The DAG to hash.
        order: Which fixed order to apply.

    Returns:
        A 64-bit signed process-local hash.
    """
    return hash(serialise(dag, order))


def fixed_order_digest(dag: LabeledDAG, order: FixedOrder = FixedOrder.INSERTION) -> int:
    """Return a stable 64-bit BLAKE2b digest of the fixed-order serialisation.

    Unlike :func:`fixed_order_hash` this value is reproducible across processes
    and machines, which makes it the correct key for a persisted stream and for
    offline replay.

    Args:
        dag: The DAG to hash.
        order: Which fixed order to apply.

    Returns:
        An unsigned 64-bit integer digest.
    """
    payload = serialise(dag, order).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
