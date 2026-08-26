"""Rounded background bands that group a figure's axes into labelled rows.

A multi-panel figure whose columns each stack several different objects gives
the reader no cue about where one object stops and the next begins.  A band
supplies that grouping once for the whole row, and a rotated label in the left
margin names it, which keeps the column titles free to carry only the step or
case they identify.

All bands share one horizontal extent, taken from the gridspec cells rather
than the drawn axes: an axes with ``set_aspect("equal")`` shrinks its own box to
preserve the aspect ratio, so measuring drawn positions would give the
equal-aspect rows narrower bands than the others and the rotated labels would
not line up.

Dependency rule: matplotlib is imported inside function bodies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:
    Axes = Any
    Figure = Any

#: Default band fill colour.
BAND_COLOR: str = "#c8ccd2"

#: Default band opacity.
BAND_ALPHA: float = 0.3

#: Default colour for the rotated band labels.
BAND_LABEL_COLOR: str = "#33383f"


def draw_axes_bands(
    fig: Figure,
    groups: Sequence[Sequence[Axes]],
    labels: Sequence[str],
    *,
    color: str = BAND_COLOR,
    alpha: float = BAND_ALPHA,
    fontsize: float = 7.2,
    label_color: str = BAND_LABEL_COLOR,
    pad_x: float = 0.006,
    pad_y: float = 0.013,
    label_gap: float = 0.008,
    rounding: float = 0.012,
) -> None:
    """Paint one rounded band behind each group of axes and label it.

    The figure must already have been drawn once, so axes positions are final.

    Parameters
    ----------
    fig:
        Parent figure.
    groups:
        One sequence of axes per band, in the order the bands stack.
    labels:
        Band labels, one per group, written rotated in the left margin.
    color:
        Band fill colour.
    alpha:
        Band opacity.
    fontsize:
        Label font size in points.
    label_color:
        Label colour.
    pad_x:
        Horizontal padding around the band, in figure fractions.
    pad_y:
        Vertical padding around the band, in figure fractions.
    label_gap:
        Gap between the band's left edge and the label, in figure fractions.
    rounding:
        Corner rounding size passed to the box style.

    Raises
    ------
    ValueError
        If ``groups`` and ``labels`` differ in length.
    """
    from matplotlib.patches import FancyBboxPatch

    if len(groups) != len(labels):
        raise ValueError(f"got {len(labels)} labels for {len(groups)} band groups")

    cells = [ax.get_position(original=True) for group in groups for ax in group]
    if not cells:
        return
    x0 = min(c.x0 for c in cells) - pad_x
    x1 = max(c.x1 for c in cells) + pad_x

    for group, label in zip(groups, labels, strict=True):
        if not group:
            continue
        boxes = [ax.get_position(original=True) for ax in group]
        y0 = min(b.y0 for b in boxes) - pad_y
        y1 = max(b.y1 for b in boxes) + pad_y
        fig.add_artist(
            FancyBboxPatch(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                boxstyle=f"round,pad=0,rounding_size={rounding}",
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=-10,
                clip_on=False,
            )
        )
        if label:
            fig.text(
                x0 - label_gap,
                (y0 + y1) / 2.0,
                label,
                ha="center",
                va="center",
                rotation=90,
                fontsize=fontsize,
                color=label_color,
                transform=fig.transFigure,
            )
