"""Visualisation subsystem for IsalSR.

**This package is the single supported implementation** for drawing IsalSR
objects. Three atomic views compose into the figures the project ships:

- the **labeled DAG** (:mod:`isalsr.viz.dag_view`), delegated to a registered
  backend via a name-keyed registry so alternative renderers can be added
  without touching callers;
- the **instruction string**, as a strip of token cells
  (:mod:`isalsr.viz.instruction_view`);
- the **CDLL**, as a chain with the primary and secondary pointer markers
  (:mod:`isalsr.viz.cdll_view`).

:func:`make_trace_figure` (:mod:`isalsr.viz.composite`) is the top-level entry
point: it assembles the three views into a 2 x N grid in which each column is
one step of a D2S run, and geometry is controlled by
:class:`~isalsr.viz.layout.TraceLayout`.

Import from ``isalsr.viz`` rather than from the submodules, so the public
surface stays the thing that is maintained.

Dependency rule: this package may import :mod:`isalsr.core`. ``matplotlib`` is
imported inside function bodies only, so it never enters the import path until a
drawing call is made, and :mod:`isalsr.core` keeps its zero-external-dependency
guarantee. Nothing in ``core``, ``adapters``, ``evaluation`` or ``search`` may
import this package.
"""

from isalsr.viz.algorithm_trace import (
    DAG_TO_STRING,
    STRING_TO_DAG,
    AlgorithmTraceLayout,
    TraceConsistencyError,
    evenly_spaced_steps,
    make_algorithm_trace_figure,
)
from isalsr.viz.bands import draw_axes_bands
from isalsr.viz.base import DagVizBackend, Position
from isalsr.viz.cdll_view import cdll_traversal, draw_cdll, draw_cdll_ring, stable_anchor
from isalsr.viz.composite import assign_positions, make_trace_figure
from isalsr.viz.dag_view import draw_dag
from isalsr.viz.expression_grid import (
    BY_ROW,
    BY_VIEW,
    ExpressionCell,
    ExpressionGridLayout,
    ExpressionRow,
    make_expression_grid_figure,
)
from isalsr.viz.instruction_view import draw_instruction_strip, tokenize_string
from isalsr.viz.layout import DEFAULT_TRACE_LAYOUT, TraceLayout
from isalsr.viz.registry import (
    VizBackendNotFoundError,
    available_backends,
    get_backend,
    register_backend,
)

__all__ = [
    "BY_ROW",
    "BY_VIEW",
    "DAG_TO_STRING",
    "DEFAULT_TRACE_LAYOUT",
    "STRING_TO_DAG",
    "AlgorithmTraceLayout",
    "ExpressionCell",
    "ExpressionGridLayout",
    "ExpressionRow",
    "draw_axes_bands",
    "make_expression_grid_figure",
    "DagVizBackend",
    "Position",
    "TraceConsistencyError",
    "TraceLayout",
    "VizBackendNotFoundError",
    "assign_positions",
    "available_backends",
    "cdll_traversal",
    "draw_cdll",
    "draw_cdll_ring",
    "draw_dag",
    "draw_instruction_strip",
    "evenly_spaced_steps",
    "get_backend",
    "make_algorithm_trace_figure",
    "make_trace_figure",
    "register_backend",
    "stable_anchor",
    "tokenize_string",
]
