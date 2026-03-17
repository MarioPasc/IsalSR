"""Exhaustive backtracking D2S (true canonical string).

Produces the true canonical string w*_D — a complete labeled-DAG invariant.
Complexity: exponential in V/v branch points. Practical for DAGs up to ~15 nodes.

Restriction: ZERO external dependencies.
"""

from __future__ import annotations

from isalsr.core.algorithms.base import D2SAlgorithm
from isalsr.core.canonical import canonical_string
from isalsr.core.labeled_dag import LabeledDAG


class ExhaustiveD2S(D2SAlgorithm):
    """Exhaustive backtracking canonical string computation."""

    def encode(self, dag: LabeledDAG) -> str:
        """Encode via exhaustive backtracking from x_1."""
        return canonical_string(dag)

    @property
    def name(self) -> str:
        return "exhaustive"
