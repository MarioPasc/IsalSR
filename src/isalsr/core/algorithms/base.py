"""Abstract base class for D2S (DAG-to-String) algorithms.

All D2S algorithms share the same contract: given a LabeledDAG, produce
an IsalSR instruction string. They differ in how they select starting
nodes and whether they explore multiple neighbor orderings.

Restriction: ZERO external dependencies. Only Python stdlib + abc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from isalsr.core.labeled_dag import LabeledDAG


class D2SAlgorithm(ABC):
    """Abstract DAG-to-String algorithm.

    Subclasses must implement ``encode()`` which converts a LabeledDAG
    into an IsalSR instruction string.
    """

    @abstractmethod
    def encode(self, dag: LabeledDAG) -> str:
        """Convert a labeled DAG to an IsalSR instruction string.

        Args:
            dag: The LabeledDAG to encode.

        Returns:
            An IsalSR instruction string.

        Raises:
            ValueError: If the DAG cannot be encoded.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this algorithm."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
