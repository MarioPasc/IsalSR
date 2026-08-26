"""Token-aware mutation and crossover operators for IsalSR strings.

All operators are SYNTACTIC: they produce valid token sequences but do NOT
canonicalize. Canonicalization is MANDATORY and happens at the caller level
(search algorithms) per the advisor's constraint.

Dependencies: numpy (for random number generation).
"""

from __future__ import annotations

import numpy as np

from isalsr.core.node_types import OperationSet
from isalsr.types import VALID_SINGLE_INSTRUCTIONS


def tokenize(string: str) -> list[str]:
    """Split an IsalSR string into tokens (no validation).

    Handles two-char compound tokens: V/v consume the next character.
    """
    tokens: list[str] = []
    i = 0
    while i < len(string):
        if string[i] in "Vv" and i + 1 < len(string):
            tokens.append(string[i : i + 2])
            i += 2
        else:
            tokens.append(string[i])
            i += 1
    return tokens


def detokenize(tokens: list[str]) -> str:
    """Join tokens back into an IsalSR string."""
    return "".join(tokens)


def _all_valid_tokens(allowed_ops: OperationSet) -> list[str]:
    """Build the list of all valid tokens given allowed operations."""
    tokens: list[str] = sorted(VALID_SINGLE_INSTRUCTIONS)
    for label in sorted(allowed_ops.label_chars):
        tokens.append("V" + label)
        tokens.append("v" + label)
    return tokens


def random_token(allowed_ops: OperationSet, rng: np.random.Generator) -> str:
    """Generate a single random valid IsalSR token."""
    pool = _all_valid_tokens(allowed_ops)
    return pool[rng.integers(len(pool))]


# ======================================================================
# Mutation operators
# ======================================================================


def point_mutation(string: str, allowed_ops: OperationSet, rng: np.random.Generator) -> str:
    """Replace a random token with another valid token."""
    tokens = tokenize(string)
    if not tokens:
        return string
    idx = rng.integers(len(tokens))
    tokens[idx] = random_token(allowed_ops, rng)
    return detokenize(tokens)


def insertion_mutation(string: str, allowed_ops: OperationSet, rng: np.random.Generator) -> str:
    """Insert a random valid token at a random position."""
    tokens = tokenize(string)
    pos = rng.integers(len(tokens) + 1)
    tokens.insert(pos, random_token(allowed_ops, rng))
    return detokenize(tokens)


def deletion_mutation(string: str, rng: np.random.Generator) -> str:
    """Remove a random token. Returns empty string if already empty."""
    tokens = tokenize(string)
    if not tokens:
        return string
    idx = rng.integers(len(tokens))
    tokens.pop(idx)
    return detokenize(tokens)


def subsequence_mutation(
    string: str,
    max_len: int,
    allowed_ops: OperationSet,
    rng: np.random.Generator,
) -> str:
    """Replace a random contiguous span of tokens with random tokens."""
    tokens = tokenize(string)
    if not tokens:
        return random_token(allowed_ops, rng)

    # Choose span to replace.
    start = int(rng.integers(len(tokens)))
    span_len = int(rng.integers(1, min(max_len, len(tokens) - start) + 1))
    # Generate replacement tokens.
    new_len = int(rng.integers(1, max_len + 1))
    replacement = [random_token(allowed_ops, rng) for _ in range(new_len)]
    tokens[start : start + span_len] = replacement
    return detokenize(tokens)


# ======================================================================
# Crossover operators
# ======================================================================


def one_point_crossover(parent1: str, parent2: str, rng: np.random.Generator) -> tuple[str, str]:
    """One-point crossover: split at random token positions, swap tails."""
    t1 = tokenize(parent1)
    t2 = tokenize(parent2)

    p1 = rng.integers(len(t1) + 1) if t1 else 0
    p2 = rng.integers(len(t2) + 1) if t2 else 0

    child1 = t1[:p1] + t2[p2:]
    child2 = t2[:p2] + t1[p1:]
    return detokenize(child1), detokenize(child2)


def two_point_crossover(parent1: str, parent2: str, rng: np.random.Generator) -> tuple[str, str]:
    """Two-point crossover: swap a random token substring between parents."""
    t1 = tokenize(parent1)
    t2 = tokenize(parent2)

    if len(t1) < 2 or len(t2) < 2:
        return one_point_crossover(parent1, parent2, rng)

    # Pick two sorted points in each parent.
    a1, b1 = sorted(rng.choice(len(t1) + 1, size=2, replace=False))
    a2, b2 = sorted(rng.choice(len(t2) + 1, size=2, replace=False))

    child1 = t1[:a1] + t2[a2:b2] + t1[b1:]
    child2 = t2[:a2] + t1[a1:b1] + t2[b2:]
    return detokenize(child1), detokenize(child2)
