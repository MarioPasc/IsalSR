"""Graphviz-``dot`` layout backend for labeled-DAG drawing.

The shipped matplotlib backend places nodes with a longest-path layering that
distributes each layer evenly on the x-axis
(:func:`isalsr.viz.backends.matplotlib_dag._layered_layout`).  That rule fixes
the *rank* of every node correctly but does nothing about the *order* of nodes
inside a rank, so edges cross whenever the natural order differs from the
node-ID order.  Crossings are not cosmetic in an expression DAG: a reader who
traces an arrow through a crossing can attach an operand to the wrong operator.

This backend delegates ordering to Graphviz's ``dot``, which implements the
Sugiyama layered method with the Gansner et al. refinements -- iterative
median/transpose heuristics for crossing reduction followed by a network-simplex
solve for horizontal coordinates.

References
----------
Gansner, E. R., Koutsofios, E., North, S. C., & Vo, K.-P. (1993). A technique
for drawing directed graphs. *IEEE Transactions on Software Engineering*,
19(3), 214-230. DOI: 10.1109/32.221135.

Sugiyama, K., Tagawa, S., & Toda, M. (1981). Methods for visual understanding
of hierarchical system structures. *IEEE Transactions on Systems, Man, and
Cybernetics*, 11(2), 109-125. DOI: 10.1109/TSMC.1981.4308636.

Only *positions* come from Graphviz; the discs, glyphs and arrows are still
painted by :class:`~isalsr.viz.backends.matplotlib_dag.MatplotlibDagBackend`,
so a figure that mixes the two backends stays visually identical apart from
node placement.

Registers itself under the name ``"graphviz"`` on import.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.viz.backends.matplotlib_dag import MatplotlibDagBackend
from isalsr.viz.base import Position
from isalsr.viz.registry import register_backend

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any

#: Rank separation passed to ``dot`` (inches).
RANKSEP: float = 0.75
#: Node separation within a rank passed to ``dot`` (inches).
NODESEP: float = 0.50
#: Side of the square bounding box reserved per node (inches).
NODE_SIZE: float = 0.55
#: Data-unit distance that one Graphviz inch maps to.
UNITS_PER_INCH: float = 1.9


class GraphvizUnavailableError(RuntimeError):
    """Raised when the Graphviz ``dot`` executable cannot be located."""


def _dot_executable() -> str:
    """Return an absolute path to the ``dot`` binary.

    Conda environments install ``dot`` into ``<prefix>/bin`` but a script run
    through an absolute interpreter path (``~/.conda/envs/env/bin/python``)
    does not get that directory on ``PATH``.  Look there explicitly before
    giving up, so figure generation does not depend on the caller having
    activated the environment.

    Returns
    -------
    str
        Absolute path to ``dot``.

    Raises
    ------
    GraphvizUnavailableError
        If no ``dot`` executable is found.
    """
    found = shutil.which("dot")
    if found:
        return found
    candidate = Path(sys.executable).parent / "dot"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    raise GraphvizUnavailableError(
        "Graphviz 'dot' was not found on PATH nor in the interpreter's bin "
        "directory. Install it with: conda install -c conda-forge graphviz"
    )


def _build_dot_source(
    dag: LabeledDAG,
    *,
    rankdir: str,
    ranksep: float,
    nodesep: float,
    node_size: float,
    source_rank: frozenset[int] = frozenset(),
) -> str:
    """Serialise ``dag`` to DOT source with fixed-size, unlabelled nodes.

    Labels are omitted deliberately: glyph metrics must not influence
    placement, because the discs painted by the matplotlib backend all have
    the same radius regardless of the text inside them.

    Parameters
    ----------
    dag:
        The DAG to serialise.
    rankdir:
        Graphviz rank direction (``"BT"``, ``"TB"``, ``"LR"``, ``"RL"``).
    ranksep:
        Separation between ranks, in inches.
    nodesep:
        Separation between nodes within a rank, in inches.
    node_size:
        Width and height reserved per node, in inches.
    source_rank:
        Node IDs to force onto the minimum rank, and therefore onto a common
        line perpendicular to ``rankdir``.  Used to align the variable nodes,
        which are the algorithms' shared starting point; ``dot`` would
        otherwise place each variable at whatever rank its consumer implies,
        scattering them across the drawing.

    Returns
    -------
    str
        DOT source text.
    """
    lines = [
        "digraph D {",
        f"  rankdir={rankdir};",
        f"  ranksep={ranksep};",
        f"  nodesep={nodesep};",
        # Real splines, not straight chords: dot routes them around intervening
        # nodes and away from each other, and clips each end to the node
        # boundary.  Straight chords can pass straight through a node the edge
        # has nothing to do with, which reads as an edge that is not there.
        "  splines=true;",
        f'  node [shape=circle, fixedsize=true, width={node_size}, height={node_size}, label=""];',
    ]
    for node in range(dag.node_count):
        lines.append(f"  n{node};")
    if source_rank:
        members = " ".join(f"n{nd};" for nd in sorted(source_rank))
        lines.append(f"  {{ rank=source; {members} }}")
    for src in range(dag.node_count):
        for tgt in dag.out_neighbors(src):
            lines.append(f"  n{src} -> n{tgt};")
    lines.append("}")
    return "\n".join(lines)


def _parse_plain(
    plain: str,
) -> tuple[dict[int, Position], dict[tuple[int, int], list[Position]]]:
    """Extract node positions and edge splines from ``dot -Tplain`` output.

    The ``plain`` format emits one ``node <name> <x> <y> ...`` record per node
    and one ``edge <tail> <head> <n> <x1> <y1> ... <xn> <yn> ...`` record per
    edge, all in inches with the origin at the lower-left corner.  The ``n``
    edge points are B-spline control points in the layout ``p0, (c, c, p), ...``,
    that is one moveto followed by cubic segments, already clipped to the node
    boundaries.

    Parameters
    ----------
    plain:
        Raw stdout of ``dot -Tplain``.

    Returns
    -------
    tuple[dict[int, Position], dict[tuple[int, int], list[Position]]]
        Node positions in inches, and edge control points in inches keyed by
        ``(source, target)``.
    """
    positions: dict[int, Position] = {}
    routes: dict[tuple[int, int], list[Position]] = {}
    for line in plain.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "node" and len(parts) >= 4 and parts[1].startswith("n"):
            positions[int(parts[1][1:])] = (float(parts[2]), float(parts[3]))
        elif parts[0] == "edge" and len(parts) >= 4:
            tail, head = parts[1], parts[2]
            if not (tail.startswith("n") and head.startswith("n")):
                continue
            count = int(parts[3])
            coords = parts[4 : 4 + 2 * count]
            pts = [
                (float(coords[2 * i]), float(coords[2 * i + 1])) for i in range(len(coords) // 2)
            ]
            routes[(int(tail[1:]), int(head[1:]))] = pts
    return positions, routes


def graphviz_layout_and_routes(
    dag: LabeledDAG,
    *,
    rankdir: str = "BT",
    ranksep: float = RANKSEP,
    nodesep: float = NODESEP,
    node_size: float = NODE_SIZE,
    units_per_inch: float = UNITS_PER_INCH,
    source_rank: frozenset[int] = frozenset(),
) -> tuple[dict[int, Position], dict[tuple[int, int], list[Position]]]:
    """Lay ``dag`` out with Graphviz ``dot`` and return positions in data units.

    The returned layout is centred on the origin so that panels sharing a
    common ``xlim``/``ylim`` stay visually aligned.

    Parameters
    ----------
    dag:
        The DAG to lay out.
    rankdir:
        Graphviz rank direction.  ``"BT"`` puts sources (the variable nodes)
        at the bottom and the expression root at the top; ``"LR"`` produces a
        wide, short drawing.
    ranksep:
        Separation between ranks, in inches.
    nodesep:
        Separation between nodes within a rank, in inches.
    node_size:
        Width and height reserved per node, in inches.
    units_per_inch:
        Conversion from Graphviz inches to matplotlib data units.
    source_rank:
        Node IDs forced onto the minimum rank, hence onto a common line
        perpendicular to ``rankdir``.

    Returns
    -------
    tuple[dict[int, Position], dict[tuple[int, int], list[Position]]]
        Node positions in data units keyed by node ID, and edge spline control
        points in data units keyed by ``(source, target)``.

    Raises
    ------
    GraphvizUnavailableError
        If the ``dot`` executable cannot be located.
    RuntimeError
        If ``dot`` exits with a non-zero status.
    """
    if dag.node_count == 0:
        return {}, {}

    source = _build_dot_source(
        dag,
        rankdir=rankdir,
        ranksep=ranksep,
        nodesep=nodesep,
        node_size=node_size,
        source_rank=source_rank,
    )
    proc = subprocess.run(  # noqa: S603 - executable path is resolved, not user input
        [_dot_executable(), "-Tplain"],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"dot failed with status {proc.returncode}: {proc.stderr.strip()}")

    raw, raw_routes = _parse_plain(proc.stdout)
    if not raw:
        raise RuntimeError("dot produced no node records")

    xs = [p[0] for p in raw.values()]
    ys = [p[1] for p in raw.values()]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0

    def to_data(pt: Position) -> Position:
        return ((pt[0] - cx) * units_per_inch, (pt[1] - cy) * units_per_inch)

    positions = {nd: to_data(p) for nd, p in raw.items()}
    routes = {key: [to_data(p) for p in pts] for key, pts in raw_routes.items()}
    return positions, routes


def graphviz_layout(
    dag: LabeledDAG,
    *,
    rankdir: str = "BT",
    ranksep: float = RANKSEP,
    nodesep: float = NODESEP,
    node_size: float = NODE_SIZE,
    units_per_inch: float = UNITS_PER_INCH,
    source_rank: frozenset[int] = frozenset(),
) -> dict[int, Position]:
    """Lay ``dag`` out with ``dot`` and return node positions in data units.

    Thin wrapper over :func:`graphviz_layout_and_routes` for callers that need
    positions only; see that function for the parameter semantics.

    Returns
    -------
    dict[int, Position]
        Node positions in data units, keyed by node ID.
    """
    return graphviz_layout_and_routes(
        dag,
        rankdir=rankdir,
        ranksep=ranksep,
        nodesep=nodesep,
        node_size=node_size,
        units_per_inch=units_per_inch,
        source_rank=source_rank,
    )[0]


class GraphvizDagBackend(MatplotlibDagBackend):
    """Layered DAG backend that takes node placement from Graphviz ``dot``.

    Inherits every drawing primitive from
    :class:`~isalsr.viz.backends.matplotlib_dag.MatplotlibDagBackend` and
    overrides only the layout step, so the two backends are visually
    interchangeable.

    Registered under the name ``"graphviz"`` at module import time.
    """

    def __init__(
        self,
        rankdir: str = "BT",
        node_r: float = 0.37,
        fs_node: float = 18.0,
        align_variables: bool = False,
        ranksep: float = RANKSEP,
        nodesep: float = NODESEP,
    ) -> None:
        """Store the rank direction and drawing metrics for subsequent draw calls.

        Parameters
        ----------
        rankdir:
            Graphviz rank direction; see :func:`graphviz_layout`.
        node_r:
            Node-disc radius in data units.
        fs_node:
            Font size of the glyph printed inside each disc, in points.
        align_variables:
            When ``True``, force every VAR node onto the minimum rank so the
            variables share one line perpendicular to ``rankdir``.
        ranksep:
            Separation between ranks, in inches.  Under ``rankdir="LR"`` this is
            the horizontal gap, and so sets how far the arrows reach.
        nodesep:
            Separation within a rank, in inches.  Under ``rankdir="LR"`` this is
            the vertical gap, and so drives the drawing's height.
        """
        super().__init__(node_r=node_r, fs_node=fs_node)
        self._rankdir = rankdir
        self._align_variables = align_variables
        self._ranksep = ranksep
        self._nodesep = nodesep

    @property
    def name(self) -> str:
        return "graphviz"

    def compute_layout(self, dag: LabeledDAG) -> dict[int, Position]:
        """Return positions for ``dag`` computed by ``dot``.

        The footprint reserved per node is derived from the radius the discs
        are actually drawn at, rather than left at a fixed default.  Otherwise
        ``dot`` packs the ranks against a node box that does not match what
        gets painted, and ``ranksep``/``nodesep`` stop being the gaps the
        reader sees: enlarging the discs eats the arrows instead of pushing the
        nodes apart.
        """
        source_rank: frozenset[int] = frozenset()
        if self._align_variables:
            source_rank = frozenset(
                nd for nd in range(dag.node_count) if dag.node_label(nd) is NodeType.VAR
            )
        return graphviz_layout(
            dag,
            rankdir=self._rankdir,
            ranksep=self._ranksep,
            nodesep=self._nodesep,
            node_size=2.0 * self.node_r / UNITS_PER_INCH,
            source_rank=source_rank,
        )


# Self-registration on import.
register_backend("graphviz", GraphvizDagBackend)
