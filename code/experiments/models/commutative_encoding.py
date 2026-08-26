"""Producer-side commutative decomposition at the host -> LabeledDAG boundary.

The paper's instruction set (Definition 3.2, ``methodology.tex:93-119``) declares
twelve label characters ``{+, *, g, i, s, c, e, l, r, ^, a, k}``.  It has no
``-`` and no ``/``: subtraction and division are expressed through the
commutative decomposition inspired by GraphSR (Xiang et al.)::

    x - y  =  Add(x, Neg(y))
    x / y  =  Mul(x, Inv(y))

``Pow`` is the only non-commutative operation with no exact commutative
decomposition, so it keeps a dedicated instruction and is left untouched.  Under
that alphabet ``Pow`` is the sole operand-order-sensitive node, which is what
makes Definition 3.9(iv) sound as written.

This module realises that decomposition **at the adapter boundary only**, which
is the same layer as ``_normalize_const_edges`` (Critical Invariant 9).  Three
hard non-goals, from ticket T16 section 5.2:

1. The host operator sets in the YAML configs are not touched.  The host still
   searches over ``{+, -, *, /, sin, cos, exp, log}``; IsalSR is a representation
   layer at the evaluation boundary, and changing what the host searches over
   would change the search itself and break the paired design.
2. ``NodeType.SUB`` / ``NodeType.DIV`` and ``BINARY_OPS`` stay intact in
   ``isalsr.core``.  S2D must still decode legacy ``V-`` / ``V/`` strings, and the
   property corpora and the precomputed atlas contain them.  Only the *adapters*
   stop emitting them.
3. Nothing here is called from the canonicaliser.  The canonical map stays a pure
   function of the DAG it is handed (ticket T07).

Why inline rather than ``isalsr.core.commutative.to_commutative``: that function
requires every ``SUB``/``DIV`` node to already carry two ordered inputs, and a
host can emit ``x - x`` (Bingo command ``[3, r, r]``), which reaches the DAG with
a single in-edge because ``LabeledDAG.add_edge`` rejects duplicates.  Emitting the
decomposition directly also avoids a full DAG copy on a per-candidate hot path.

Restriction: imports only ``isalsr.core``.  No numpy, no host libraries.
"""

from __future__ import annotations

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

# ======================================================================
# The decomposition table
# ======================================================================

#: Maps each decomposable non-commutative binary op to the
#: ``(variadic_result, unary_wrapper)`` pair that replaces it.
#: ``POW`` is deliberately absent -- it has no exact commutative decomposition.
DECOMPOSITION: dict[NodeType, tuple[NodeType, NodeType]] = {
    NodeType.SUB: (NodeType.ADD, NodeType.NEG),
    NodeType.DIV: (NodeType.MUL, NodeType.INV),
}

#: Node types the adapters must never emit once decomposition is active.
#: Asserted by the AC-2 tests.
FORBIDDEN_ADAPTER_LABELS: frozenset[NodeType] = frozenset(DECOMPOSITION)

#: Whether a ``Neg``/``Inv`` node introduced by the decomposition is reused when
#: the same operand is wrapped more than once within one DAG.
#:
#: ``False`` -- the default -- makes the decomposition a purely local
#: node-splitting: the decomposed node set is in bijection with
#: ``host_nodes + host_sub_div_nodes``, so ``k`` grows by exactly
#: ``#Sub + #Div``.  It also keeps the adapter faithful to the host's own
#: sharing structure: the adapter does not merge the host's duplicated ``Sin``
#: or ``Sub`` nodes either, so merging only the ``Neg``/``Inv`` nodes it invents
#: would be a partial common-subexpression elimination applied inconsistently.
#:
#: ``True`` emits one ``Neg(b)`` with out-degree > 1 instead.  Both choices are
#: injective on host DAGs, so dedup soundness does not discriminate between
#: them; the discriminator is their measured effect on ``k`` and on the
#: reduction factor rho.  See ticket T16 section 5.1 and AC-4.
SHARE_DECOMPOSED_UNARY: bool = False


def new_unary_cache(share: bool | None = None) -> dict[tuple[NodeType, int], int] | None:
    """Create the per-DAG sharing cache, or ``None`` to disable sharing.

    Args:
        share: Override the module default ``SHARE_DECOMPOSED_UNARY``.  ``None``
            uses the default.

    Returns:
        An empty cache dict when sharing is on, ``None`` when it is off.  The
        result is passed as ``unary_cache`` to every :func:`emit_binary` call for
        one DAG, and must not be reused across DAGs.
    """
    enabled = SHARE_DECOMPOSED_UNARY if share is None else share
    return {} if enabled else None


# ======================================================================
# Emission
# ======================================================================


def emit_unary(
    dag: LabeledDAG,
    unary_type: NodeType,
    operand: int,
    unary_cache: dict[tuple[NodeType, int], int] | None = None,
) -> int:
    """Create ``unary_type(operand)`` and return its node id.

    Args:
        dag: The DAG under construction.
        unary_type: ``NodeType.NEG`` or ``NodeType.INV``.
        operand: Node id of the single input.
        unary_cache: Sharing memo from :func:`new_unary_cache`, or ``None``.

    Returns:
        Node id of the (possibly reused) unary node.
    """
    key = (unary_type, operand)
    if unary_cache is not None:
        hit = unary_cache.get(key)
        if hit is not None:
            return hit

    node = dag.add_node(unary_type)
    dag.add_edge(operand, node)

    if unary_cache is not None:
        unary_cache[key] = node
    return node


def emit_binary(
    dag: LabeledDAG,
    node_type: NodeType,
    first: int | None,
    second: int | None,
    *,
    decompose: bool = True,
    unary_cache: dict[tuple[NodeType, int], int] | None = None,
) -> int:
    """Create the node(s) realising ``node_type(first, second)``.

    When ``decompose`` is true and ``node_type`` is ``SUB`` or ``DIV``, emits the
    commutative form and returns the id of the resulting ``ADD`` / ``MUL`` node.
    Every other node type, ``POW`` included, is emitted unchanged.

    The unary wrapper is created *before* the variadic node so that node ids stay
    in topological order, and the first operand's edge is added *before* the
    wrapper's so that ``ordered_inputs`` reads ``[first, wrapped]``.  Operand
    order is irrelevant for ``ADD``/``MUL`` semantics but is kept deterministic so
    that the two engines and the two hosts agree byte-for-byte.

    A ``None`` operand means the host row was not utilised; the corresponding edge
    is skipped, matching the pre-decomposition adapter behaviour.

    ``first is second`` (the host's ``x - x`` / ``x / x``) is representable here
    and was not representable before: ``Neg(a)`` is a distinct node from ``a``, so
    both in-edges of the ``ADD`` are added, whereas the undecomposed ``SUB``
    silently kept a single input because ``LabeledDAG.add_edge`` rejects the
    duplicate.

    Args:
        dag: The DAG under construction.
        node_type: The host operation being emitted.
        first: Node id of the first operand, or ``None``.
        second: Node id of the second operand, or ``None``.
        decompose: Set false to reproduce the pre-T16 encoding, for A/B
            measurement and for the legacy-string regression tests.
        unary_cache: Sharing memo from :func:`new_unary_cache`, or ``None``.

    Returns:
        Node id representing the value of the operation.
    """
    spec = DECOMPOSITION.get(node_type) if decompose else None

    if spec is None:
        node = dag.add_node(node_type)
        if first is not None:
            dag.add_edge(first, node)
        if second is not None:
            # No-op when second == first; preserves the pre-T16 behaviour for
            # ADD/MUL/POW, which this ticket deliberately does not change.
            dag.add_edge(second, node)
        return node

    variadic_type, unary_type = spec

    wrapped = None if second is None else emit_unary(dag, unary_type, second, unary_cache)

    result = dag.add_node(variadic_type)
    if first is not None:
        dag.add_edge(first, result)
    if wrapped is not None:
        dag.add_edge(wrapped, result)
    return result


def contains_decomposed_unary(dag: LabeledDAG) -> bool:
    """Return true if *dag* carries any ``Neg`` or ``Inv`` node."""
    wrappers = {unary for _variadic, unary in DECOMPOSITION.values()}
    return any(dag.node_label(i) in wrappers for i in range(dag.node_count))


def undecompose(dag: LabeledDAG) -> LabeledDAG:
    """Invert the decomposition, for hosts whose alphabet has no ``Neg``/``Inv``.

    Delegates to ``isalsr.core.commutative.from_commutative``, which collapses
    ``Add(a, Neg(b))`` back to ``Sub(a, b)`` and ``Mul(a, Inv(b))`` back to
    ``Div(a, b)``.  Needed only on the LabeledDAG -> host direction for Bingo,
    whose command-array opcode table has no entry for ``Neg`` or ``Inv``.

    Args:
        dag: A decomposed DAG, as produced by the adapters.

    Returns:
        An equivalent DAG in which every collapsible pattern has become a
        ``Sub``/``Div`` node.  A DAG with no ``Neg``/``Inv`` is returned
        unchanged.

    Raises:
        ValueError: If a ``Neg``/``Inv`` node has out-degree > 1.
            ``from_commutative`` cannot absorb such a node, and duplicating it
            here would silently change the DAG the caller handed in.  This can
            only happen when the DAG was built with ``SHARE_DECOMPOSED_UNARY``
            enabled, which is a measurement setting, not the production one.
    """
    if not contains_decomposed_unary(dag):
        return dag

    wrappers = {unary for _variadic, unary in DECOMPOSITION.values()}
    shared = [
        i for i in range(dag.node_count) if dag.node_label(i) in wrappers and dag.out_degree(i) > 1
    ]
    if shared:
        raise ValueError(
            "Cannot undecompose a DAG with shared Neg/Inv nodes "
            f"(node ids {shared}); rebuild it with share_unary=False."
        )

    from isalsr.core.commutative import from_commutative

    return from_commutative(dag)


def extra_node_budget(n_decomposable: int, decompose: bool = True) -> int:
    """Upper bound on the nodes the decomposition will add.

    ``LabeledDAG`` pre-allocates ``max_nodes`` and ``add_node`` raises at the cap,
    so both adapters must widen their capacity estimate by this amount.  The bound
    is exact when sharing is off and conservative when it is on.

    Args:
        n_decomposable: Number of ``Sub`` and ``Div`` operations in the host DAG.
        decompose: Whether decomposition is active.

    Returns:
        Number of additional node slots to reserve.
    """
    return n_decomposable if decompose else 0


__all__: list[str] = [
    "DECOMPOSITION",
    "FORBIDDEN_ADAPTER_LABELS",
    "SHARE_DECOMPOSED_UNARY",
    "new_unary_cache",
    "emit_unary",
    "emit_binary",
    "contains_decomposed_unary",
    "undecompose",
    "extra_node_budget",
]
