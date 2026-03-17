"""Shared type aliases for IsalSR.

Centralizes all type aliases used across the package. These distinguish
between CDLL internal indices and graph node indices -- conflating them
is the most common source of silent corruption.

Restriction: ZERO external dependencies. Only Python stdlib + typing.
"""

from __future__ import annotations

# Type alias for graph node indices (contiguous integers 0..N-1).
NodeId = int

# Type alias for CDLL internal node indices (allocated from free list).
CdllIndex = int

# Type alias for a single IsalSR instruction token.
# Can be single-char (N, P, n, p, C, c, W) or two-char compound (V+, Vs, vk, ...).
InstructionToken = str

# Type alias for a node label character (+, *, s, c, e, l, r, ^, a, k, -, /).
NodeLabel = str

# Type alias for a full IsalSR instruction string.
InstructionString = str

# Single-character instructions (movement, edge, no-op).
VALID_SINGLE_INSTRUCTIONS: frozenset[str] = frozenset("NnPpCcW")

# Valid label characters for V/v compound tokens.
VALID_LABEL_CHARS: frozenset[str] = frozenset("+*-/scelr^ak")

__all__: list[str] = [
    "NodeId",
    "CdllIndex",
    "InstructionToken",
    "NodeLabel",
    "InstructionString",
    "VALID_SINGLE_INSTRUCTIONS",
    "VALID_LABEL_CHARS",
]
