"""``DagVizBackend`` -- abstract drawing interface for labeled-DAG visualisation.

Concrete backends wrap one external visualisation library (currently only
matplotlib is shipped). Each backend implements a single :meth:`draw` method
that paints a ``LabeledDAG`` onto a caller-supplied matplotlib ``Axes`` and
returns the node-position mapping it used, so a figure that shares layout
across panels can pin positions for visual continuity.

Dependency rule: this module imports only :mod:`isalsr.core`. matplotlib is
imported inside method bodies of concrete backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeAlias

from isalsr.core.labeled_dag import LabeledDAG

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any

Position: TypeAlias = tuple[float, float]


class DagVizBackend(ABC):
    """Abstract base class for labeled-DAG visualisation backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend (matches the registry key)."""
        ...

    @abstractmethod
    def draw(
        self,
        dag: LabeledDAG,
        ax: Axes,
        *,
        node_colors: dict[int, str] | None = None,
        reachable: frozenset[int] = frozenset(),
        layout: dict[int, Position] | None = None,
    ) -> dict[int, Position]:
        """Draw ``dag`` on ``ax`` and return the layout used.

        Parameters
        ----------
        dag:
            The labeled DAG to draw.
        ax:
            Target matplotlib axes.
        node_colors:
            Per-node face colour override (hex or named). When ``None`` the
            backend derives colours from the node's :class:`NodeType`.
        reachable:
            Set of node IDs considered reachable from a variable. The backend
            uses this to tint or annotate nodes that satisfy the reachability
            condition.
        layout:
            If provided, the backend must use this layout instead of computing
            one. Otherwise the backend computes and returns its own layout.

        Returns
        -------
        dict[int, Position]
            Node positions used, so the caller can pin them for subsequent
            panels that must share the same layout.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
