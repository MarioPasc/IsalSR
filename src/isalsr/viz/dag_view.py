"""DAG drawing dispatch.

Thin wrapper around :func:`isalsr.viz.registry.get_backend`. Resolves a
backend name to a :class:`~isalsr.viz.base.DagVizBackend` instance and
forwards the draw call. Callers do not need to know about the registry.

Dependency rule: imports :mod:`isalsr.core` and :mod:`isalsr.viz` internals.
matplotlib is imported inside the backend's draw method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.viz.base import Position
from isalsr.viz.registry import get_backend

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


def draw_dag(
    ax: Axes,
    dag: LabeledDAG,
    *,
    backend: str = "matplotlib",
    node_colors: dict[int, str] | None = None,
    reachable: frozenset[int] = frozenset(),
    layout: dict[int, Position] | None = None,
) -> dict[int, Position]:
    """Draw ``dag`` on ``ax`` via the named backend.

    Parameters
    ----------
    ax:
        Target matplotlib axes.
    dag:
        The :class:`~isalsr.core.labeled_dag.LabeledDAG` to render.
    backend:
        Backend name as registered in the viz registry. Defaults to
        ``"matplotlib"``.
    node_colors:
        Optional per-node face colour overrides.
    reachable:
        Node IDs satisfying the reachability condition; the backend
        may tint or annotate these differently from unreachable nodes.
    layout:
        Fixed node positions. When provided the backend must use these
        rather than computing its own.

    Returns
    -------
    dict[int, Position]
        Node positions used during drawing.
    """
    b = get_backend(backend)
    return b.draw(
        dag,
        ax,
        node_colors=node_colors,
        reachable=reachable,
        layout=layout,
    )
