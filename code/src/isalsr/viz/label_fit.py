"""Size node glyphs so they fit inside the discs that carry them.

A node disc is specified in **data units** and scales with whatever limits the
panel ends up with; the glyph inside it is specified in **points** and does not.
Nothing connects the two, so a font size that fits at one figure size spills out
of the disc at another, and the constant has to be retuned by eye every time the
layout moves.

This module ties them together after the fact: once the limits are final, it
measures the rendered disc and the rendered text and rescales the text to fit.
A text bounding box of width ``w`` and height ``h`` fits inside a circle of
diameter ``D`` exactly when its diagonal does,

.. math:: \\sqrt{w^2 + h^2} \\le D,

which is the condition used here.  Both ``w`` and ``h`` scale linearly with font
size, so one measurement pass determines the correct size without iterating.

One size is applied across every axes passed in, rather than shrinking each
label to its own maximum.  Per-label sizing would render ``+`` noticeably larger
than ``sin`` in the same drawing, which reads as a mistake rather than as a fit.

Dependency rule: matplotlib is imported inside function bodies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.text import Text
else:
    Axes = Any
    Text = Any

#: Artist group id marking a text as a node glyph eligible for fitting.
NODE_LABEL_GID: str = "isalsr-node-label"

#: Fraction of the disc diameter the text diagonal may occupy.
DEFAULT_FILL: float = 0.90


def _disc_diameter_px(ax: Axes, node_r: float) -> float:
    """Return the rendered diameter, in display units, of a disc of radius ``node_r``."""
    origin = ax.transData.transform((0.0, 0.0))
    edge = ax.transData.transform((node_r, 0.0))
    return 2.0 * float(abs(edge[0] - origin[0]))


def _node_labels(ax: Axes, gid: str) -> list[Text]:
    """Return the text artists on ``ax`` tagged as node glyphs."""
    return [t for t in ax.texts if t.get_gid() == gid]


def fit_node_labels(
    axes: Sequence[Axes],
    node_r: float,
    *,
    max_fontsize: float,
    min_fontsize: float = 2.5,
    fill: float = DEFAULT_FILL,
    gid: str = NODE_LABEL_GID,
) -> float:
    """Rescale every node glyph in ``axes`` to one size that fits its disc.

    Call this only after the axes limits are final and the figure has been
    drawn once, since the disc's rendered size depends on both.

    Parameters
    ----------
    axes:
        Axes whose node glyphs share one disc radius and should share one font
        size.  Passing every column of a row at once keeps the row uniform.
    node_r:
        Disc radius in data units, as used when the discs were drawn.
    max_fontsize:
        Upper bound on the resulting size; the requested size, when it fits.
    min_fontsize:
        Lower bound, so a pathologically small panel yields something rather
        than an invisible glyph.
    fill:
        Fraction of the disc diameter the text diagonal may occupy.
    gid:
        Artist group id identifying node glyphs.

    Returns
    -------
    float
        The font size applied, in points.  Returns ``max_fontsize`` when there
        is nothing to measure.
    """
    fig = None
    scale = math.inf
    for ax in axes:
        labels = _node_labels(ax, gid)
        if not labels:
            continue
        fig = ax.figure
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
        budget = fill * _disc_diameter_px(ax, node_r)
        for text in labels:
            bbox = text.get_window_extent(renderer=renderer)
            diagonal = math.hypot(bbox.width, bbox.height)
            if diagonal <= 0.0:
                continue
            # Width and height are both linear in font size, so the diagonal is
            # too, and this ratio is the exact factor rather than a step toward it.
            current = float(text.get_fontsize())
            scale = min(scale, (budget / diagonal) * (max_fontsize / current))

    if fig is None or not math.isfinite(scale):
        return max_fontsize

    fitted = min(max_fontsize, max_fontsize * scale)
    fitted = max(min_fontsize, fitted)
    for ax in axes:
        for text in _node_labels(ax, gid):
            text.set_fontsize(fitted)
    return fitted
