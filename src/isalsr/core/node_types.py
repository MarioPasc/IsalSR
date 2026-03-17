"""Node type registry for IsalSR.

Defines the NodeType enum and associated metadata: arity, label character mapping,
and categorization (unary, binary, variadic, leaf). The operation set is configurable
per experiment -- all possible operations are defined here, and experiments select
subsets via YAML configuration.

Mathematical justification for variable-arity ADD/MUL:
    GraphDSR (Liu2025, Neural Networks 187:107405) demonstrates that variable-arity
    operations for sum and product are essential for efficient DAG-based SR,
    enabling natural representation of expressions like x_1 + x_2 + x_3
    without nested binary trees.

Restriction: ZERO external dependencies. Only Python stdlib + enum.
"""

from __future__ import annotations

from enum import Enum


class NodeType(Enum):
    """Node types for IsalSR expression DAGs.

    Each node in an IsalSR expression DAG has exactly one NodeType.
    VAR nodes are pre-inserted as input variables. All others are created
    by V/v instructions during string execution.
    """

    VAR = "var"  # Input variable x_i. Arity = 0 (leaf).
    ADD = "+"  # Addition. Arity = variable (2+).
    MUL = "*"  # Multiplication. Arity = variable (2+).
    SUB = "-"  # Subtraction. Arity = 2 (binary).
    DIV = "/"  # Division (protected). Arity = 2 (binary).
    SIN = "s"  # Sine. Arity = 1 (unary).
    COS = "c"  # Cosine. Arity = 1 (unary).
    EXP = "e"  # Exponential. Arity = 1 (unary).
    LOG = "l"  # Logarithm (protected). Arity = 1 (unary).
    SQRT = "r"  # Square root (protected). Arity = 1 (unary).
    POW = "^"  # Power. Arity = 2 (binary).
    ABS = "a"  # Absolute value. Arity = 1 (unary).
    CONST = "k"  # Learnable constant. Arity = 0 (leaf).


# Maps single-char labels (used in V/v compound tokens) to NodeType.
# VAR is excluded because variables are pre-inserted, not created by V/v.
LABEL_CHAR_MAP: dict[str, NodeType] = {
    "+": NodeType.ADD,
    "*": NodeType.MUL,
    "-": NodeType.SUB,
    "/": NodeType.DIV,
    "s": NodeType.SIN,
    "c": NodeType.COS,
    "e": NodeType.EXP,
    "l": NodeType.LOG,
    "r": NodeType.SQRT,
    "^": NodeType.POW,
    "a": NodeType.ABS,
    "k": NodeType.CONST,
}

# Reverse map: NodeType -> label character (for D2S token emission).
NODE_TYPE_TO_LABEL: dict[NodeType, str] = {v: k for k, v in LABEL_CHAR_MAP.items()}

# Arity map: None means variable-arity (2+), 0 means leaf.
ARITY_MAP: dict[NodeType, int | None] = {
    NodeType.VAR: 0,
    NodeType.CONST: 0,
    NodeType.ADD: None,  # Variable-arity: sum of all inputs.
    NodeType.MUL: None,  # Variable-arity: product of all inputs.
    NodeType.SUB: 2,
    NodeType.DIV: 2,
    NodeType.POW: 2,
    NodeType.SIN: 1,
    NodeType.COS: 1,
    NodeType.EXP: 1,
    NodeType.LOG: 1,
    NodeType.SQRT: 1,
    NodeType.ABS: 1,
}

# Category sets for quick membership checks.
UNARY_OPS: frozenset[NodeType] = frozenset(
    {NodeType.SIN, NodeType.COS, NodeType.EXP, NodeType.LOG, NodeType.SQRT, NodeType.ABS}
)
BINARY_OPS: frozenset[NodeType] = frozenset({NodeType.SUB, NodeType.DIV, NodeType.POW})
VARIADIC_OPS: frozenset[NodeType] = frozenset({NodeType.ADD, NodeType.MUL})
LEAF_TYPES: frozenset[NodeType] = frozenset({NodeType.VAR, NodeType.CONST})
ALL_OPS: frozenset[NodeType] = UNARY_OPS | BINARY_OPS | VARIADIC_OPS

# Valid label characters (keys of LABEL_CHAR_MAP).
VALID_LABEL_CHARS: frozenset[str] = frozenset(LABEL_CHAR_MAP.keys())


class OperationSet:
    """Configurable set of allowed operations for an experiment.

    Wraps a frozenset[NodeType] with validation and label-char filtering.
    Used by S2D to reject tokens for disallowed operations, and by search
    operators to constrain mutation/crossover to valid operations.
    """

    __slots__ = ("_ops", "_label_chars")

    def __init__(self, ops: frozenset[NodeType] | None = None) -> None:
        """Initialize with a set of allowed operations.

        Args:
            ops: Allowed operation types. If None, all operations are allowed.
                 VAR and CONST are always implicitly allowed (VAR is pre-inserted,
                 CONST is a leaf type).
        """
        if ops is None:
            self._ops: frozenset[NodeType] = ALL_OPS | LEAF_TYPES
        else:
            # Always include CONST (leaf) and VAR (pre-inserted).
            self._ops = ops | {NodeType.VAR, NodeType.CONST}
        # Pre-compute allowed label chars for fast tokenizer validation.
        self._label_chars: frozenset[str] = frozenset(
            label for label, ntype in LABEL_CHAR_MAP.items() if ntype in self._ops
        )

    @property
    def ops(self) -> frozenset[NodeType]:
        """Return the set of allowed operations."""
        return self._ops

    @property
    def label_chars(self) -> frozenset[str]:
        """Return allowed label characters for V/v compound tokens."""
        return self._label_chars

    def __contains__(self, item: NodeType) -> bool:
        return item in self._ops

    def __len__(self) -> int:
        return len(self._ops)

    def __repr__(self) -> str:
        names = sorted(op.name for op in self._ops if op not in LEAF_TYPES)
        return f"OperationSet({', '.join(names)})"


__all__: list[str] = [
    "NodeType",
    "LABEL_CHAR_MAP",
    "NODE_TYPE_TO_LABEL",
    "ARITY_MAP",
    "UNARY_OPS",
    "BINARY_OPS",
    "VARIADIC_OPS",
    "LEAF_TYPES",
    "ALL_OPS",
    "VALID_LABEL_CHARS",
    "OperationSet",
]
