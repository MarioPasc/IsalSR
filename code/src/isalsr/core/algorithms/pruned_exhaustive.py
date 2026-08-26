"""Pruned exhaustive D2S with 6-component structural tuple.

Preserves the canonical property while reducing branching via structural
tuple pruning. See Lopez-Rubio (2025), arXiv:2512.10429v2.

Restriction: ZERO external dependencies.
"""

from __future__ import annotations

from isalsr.core.algorithms.base import D2SAlgorithm
from isalsr.core.canonical import pruned_canonical_string
from isalsr.core.labeled_dag import LabeledDAG


class PrunedExhaustiveD2S(D2SAlgorithm):
    """Pruned exhaustive canonical string with 6-tuple pruning."""

    def encode(self, dag: LabeledDAG) -> str:
        """Encode via pruned exhaustive backtracking from x_1."""
        return pruned_canonical_string(dag)

    @property
    def name(self) -> str:
        return "pruned-exhaustive"
