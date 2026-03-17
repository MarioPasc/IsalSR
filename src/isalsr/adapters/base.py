"""Abstract adapter interface (ABC) for external library bridges.

Follows the Bridge pattern (Gamma et al., 1994). Concrete adapters
(NetworkX, SymPy) implement this to translate between external objects
and IsalSR's LabeledDAG.

Restriction: Only Python stdlib + abc + typing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from isalsr.core.algorithms.base import D2SAlgorithm
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.string_to_dag import StringToDAG

T = TypeVar("T")


class DAGAdapter(ABC, Generic[T]):
    """Abstract bridge between external libraries and IsalSR core."""

    @abstractmethod
    def from_external(self, obj: T) -> LabeledDAG:
        """Convert an external object to a LabeledDAG."""
        ...

    @abstractmethod
    def to_external(self, dag: LabeledDAG) -> T:
        """Convert a LabeledDAG to an external object."""
        ...

    def to_isalsr_string(
        self,
        obj: T,
        *,
        algorithm: D2SAlgorithm | None = None,
    ) -> str:
        """Convert an external object to its IsalSR instruction string.

        Args:
            obj: External object.
            algorithm: D2S algorithm to use. If None, uses greedy from x_1.

        Returns:
            IsalSR instruction string.
        """
        dag = self.from_external(obj)
        if algorithm is not None:
            return algorithm.encode(dag)
        num_vars = len(dag.var_nodes())
        if dag.node_count == num_vars and dag.edge_count == 0:
            return ""
        return DAGToString(dag, initial_node=0).run()

    def from_isalsr_string(self, string: str, num_variables: int) -> T:
        """Convert an IsalSR instruction string to an external object.

        Args:
            string: IsalSR instruction string.
            num_variables: Number of input variables.
        """
        dag = StringToDAG(string, num_variables=num_variables).run()
        return self.to_external(dag)
