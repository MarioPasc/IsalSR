"""Containment guard for the IsalSR encoding alphabet.

The representation is a *drop-in* layer: a host solver proposes an expression
DAG and IsalSR encodes it as an instruction string over the alphabet
:math:`\\Sigma_{SR}` of Definition 3.2, whose label set

.. math::

    \\mathcal{L} = \\{\\texttt{+},\\texttt{*},\\texttt{g},\\texttt{i},
    \\texttt{s},\\texttt{c},\\texttt{e},\\texttt{l},\\texttt{r},
    \\texttt{\\^{}},\\texttt{a},\\texttt{k}\\}

has exactly twelve entries. The invariant that makes the pairing sound is
**containment**, not equality:

    image(host operators under the adapter translation) ⊆ 𝓛

Equality is unreachable and is not the goal. 𝓛 is a *post-decomposition*
alphabet: it carries ``Neg``/``Inv`` and no ``Sub``/``Div``, whereas Bingo's
opcode table carries ``Sub``/``Div`` and has no ``Neg``/``Inv`` at all, and
UDFS carries both plus an identity operator and no ``Pow``. Three
producer-side rewrites close the gap, all of them applied at the adapter
boundary and none of them inside the canonicaliser:

* ``Sub(a, b) -> Add(a, Neg(b))`` and ``Div(a, b) -> Mul(a, Inv(b))``
  (:mod:`experiments.models.commutative_encoding`);
* identity nodes are contracted onto their child (UDFS ``'='``).

Why this module exists. Until now containment was *assumed*. A host operator
with no image raises inside the adapter, the runner catches the exception,
counts it in the fallback ledger and evaluates the candidate normally, so an
operator outside 𝓛 silently degrades deduplication for the whole run instead
of failing. One line added to a YAML operator set (``tanh``, ``square``) is
enough to trigger it, and nothing in the reported numbers would say so. The
guard turns the assumption into a precondition checked before any search
starts.

Raises:
    AlphabetCoverageError: If a configured host operator has no image in 𝓛.
"""

from __future__ import annotations

from collections.abc import Sequence

from isalsr.core.node_types import NodeType

__all__ = [
    "AlphabetCoverageError",
    "SIGMA_SR_LABELS",
    "bingo_operator_image",
    "validate_bingo_operators",
    "validate_udfs_operators",
]


class AlphabetCoverageError(ValueError):
    """A configured host operator has no image in the encoding alphabet."""


#: The twelve labels of 𝓛 (Definition 3.2). ``VAR`` is excluded: variables are
#: pre-inserted by S2D and are not created by an insertion token.
SIGMA_SR_LABELS: frozenset[NodeType] = frozenset(
    {
        NodeType.ADD,
        NodeType.MUL,
        NodeType.NEG,
        NodeType.INV,
        NodeType.SIN,
        NodeType.COS,
        NodeType.EXP,
        NodeType.LOG,
        NodeType.SQRT,
        NodeType.POW,
        NodeType.ABS,
        NodeType.CONST,
    },
)

#: Labels the core still decodes (legacy ``V-``/``V/`` strings, the property
#: corpora, the atlas) but which no adapter may emit, since they are outside 𝓛.
LEGACY_ONLY_LABELS: frozenset[NodeType] = frozenset({NodeType.SUB, NodeType.DIV})

#: Bingo operator aliases to opcodes, copied from
#: ``bingo.symbolic_regression.agraph.operator_definitions.OPERATOR_NAMES``.
#: Held statically so that validating a configuration needs no host import;
#: ``tests/unit/test_alphabet_guard.py`` asserts it still matches Bingo's own
#: table entry for entry, so drift in the dependency fails a test rather than
#: silently widening what the guard accepts.
_BINGO_ALIAS_TO_OPCODE: dict[str, int] = {
    "integer": -1,
    "load": 0,
    "x": 0,
    "constant": 1,
    "c": 1,
    "add": 2,
    "addition": 2,
    "+": 2,
    "subtract": 3,
    "subtraction": 3,
    "-": 3,
    "multiply": 4,
    "multiplication": 4,
    "*": 4,
    "divide": 5,
    "division": 5,
    "/": 5,
    "sine": 6,
    "sin": 6,
    "cosine": 7,
    "cos": 7,
    "exponential": 8,
    "exp": 8,
    "e": 8,
    "logarithm": 9,
    "log": 9,
    "power": 10,
    "pow": 10,
    "^": 10,
    "absolute value": 11,
    "||": 11,
    "|": 11,
    "square root": 12,
    "sqrt": 12,
    "safe power": 13,
    "safe pow": 13,
    "sineh": 14,
    "sinh": 14,
    "cosineh": 15,
    "cosh": 15,
    "tangent": 16,
    "tan": 16,
    "arcsin": 17,
    "asin": 17,
    "arccos": 18,
    "acos": 18,
    "arctan": 19,
    "atan": 19,
    "tangenth": 20,
    "tanh": 20,
    "square": 21,
    "sq": 21,
    "cube": 22,
    "cb": 22,
}

#: Bingo opcode to its image in 𝓛. Opcodes 3 and 5 map to a pair because the
#: decomposition emits two nodes. Terminals (variables, constants) are omitted:
#: they are always available and are not part of a configured operator set.
_BINGO_OPCODE_IMAGE: dict[int, tuple[NodeType, ...]] = {
    2: (NodeType.ADD,),
    3: (NodeType.ADD, NodeType.NEG),
    4: (NodeType.MUL,),
    5: (NodeType.MUL, NodeType.INV),
    6: (NodeType.SIN,),
    7: (NodeType.COS,),
    8: (NodeType.EXP,),
    9: (NodeType.LOG,),
    10: (NodeType.POW,),
    11: (NodeType.ABS,),
    12: (NodeType.SQRT,),
}

#: UDFS operators absorbed by a structural rewrite rather than by a label.
#: ``'='`` is the identity, contracted onto its child by the adapter.
_UDFS_ABSORBED_OPS: frozenset[str] = frozenset({"="})


def bingo_operator_image(operator: str) -> tuple[NodeType, ...]:
    """Return the labels a Bingo operator is encoded as.

    Args:
        operator: Operator name or symbol as written in a Bingo configuration,
            using any alias Bingo itself accepts.

    Returns:
        The labels of 𝓛 the operator maps to. A pair for subtraction and
        division, which the adapter decomposes into two nodes.

    Raises:
        AlphabetCoverageError: If the name is not a Bingo operator, or is one
            with no image in 𝓛.
    """
    key = operator.strip().lower()
    opcode = _BINGO_ALIAS_TO_OPCODE.get(key)
    if opcode is None:
        raise AlphabetCoverageError(
            f"{operator!r} is not a Bingo operator. "
            f"Known operators: {sorted(set(_BINGO_ALIAS_TO_OPCODE))}.",
        )
    image = _BINGO_OPCODE_IMAGE.get(opcode)
    if image is None:
        raise AlphabetCoverageError(
            f"Bingo operator {operator!r} (opcode {opcode}) has no image in the "
            f"encoding alphabet of Definition 3.2, whose labels are "
            f"{sorted(label.name for label in SIGMA_SR_LABELS)}. A candidate "
            f"containing it cannot be canonicalised: the adapter refuses it, the "
            f"run counts a conversion failure and evaluates it without "
            f"deduplication, so the reported reduction factor would silently "
            f"understate the redundancy. Either remove the operator from the "
            f"configuration or extend the alphabet, which changes the token "
            f"count of Definition 3.2 and, for a non-commutative operator, the "
            f"operand-order treatment of Rule 1 and of condition (iv) of "
            f"Definition 3.9.",
        )
    return image


def validate_bingo_operators(operators: Sequence[str]) -> dict[str, tuple[NodeType, ...]]:
    """Check that every configured Bingo operator is encodable.

    Args:
        operators: Operator names or symbols from a Bingo configuration.

    Returns:
        The image of each operator, keyed by the string as configured.

    Raises:
        AlphabetCoverageError: If any operator has no image in 𝓛. All offending
            operators are reported together, so a configuration is fixed in one
            pass rather than one error at a time.
    """
    images: dict[str, tuple[NodeType, ...]] = {}
    offenders: list[str] = []
    for operator in operators:
        try:
            images[operator] = bingo_operator_image(operator)
        except AlphabetCoverageError:
            offenders.append(operator)
    if offenders:
        # Re-raise on the first offender: its message carries the full
        # explanation, and the list names the rest.
        try:
            bingo_operator_image(offenders[0])
        except AlphabetCoverageError as exc:
            if len(offenders) > 1:
                raise AlphabetCoverageError(
                    f"{exc}\nAll offending operators: {offenders}.",
                ) from exc
            raise
    return images


def validate_udfs_operators() -> dict[str, tuple[NodeType, ...]]:
    """Check that UDFS's own operator table is fully encodable.

    UDFS takes no operator set from the experiment configuration: its search
    enumerates every entry of the vendored node table, so the set to validate is
    that table itself. The check therefore guards against drift in the vendored
    source, such as re-enabling the commented-out ``pow_l``/``pow_r`` entries.

    Returns:
        The image of each UDFS operator, keyed by the vendor's operator name.
        Absorbed operators map to the empty tuple.

    Raises:
        AlphabetCoverageError: If the vendored table contains an operator that
            is neither mapped to a label nor absorbed by a structural rewrite.
    """
    from experiments.models.udfs.adapter import UDFS_OP_TO_ISALSR
    from experiments.models.udfs.vendor.DAG_search import config as udfs_config

    decomposed = {
        NodeType.SUB: (NodeType.ADD, NodeType.NEG),
        NodeType.DIV: (NodeType.MUL, NodeType.INV),
    }
    images: dict[str, tuple[NodeType, ...]] = {}
    offenders: list[str] = []
    for operator in udfs_config.NODE_ARITY:
        if operator in _UDFS_ABSORBED_OPS:
            images[operator] = ()
            continue
        label = UDFS_OP_TO_ISALSR.get(operator)
        if label is None:
            offenders.append(operator)
            continue
        images[operator] = decomposed.get(label, (label,))
    if offenders:
        raise AlphabetCoverageError(
            f"The vendored UDFS operator table contains {offenders}, which have no "
            f"image in the encoding alphabet of Definition 3.2 and are not absorbed "
            f"by a structural rewrite. Candidates containing them cannot be "
            f"canonicalised and would be evaluated without deduplication.",
        )
    return images
