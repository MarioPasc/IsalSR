"""Render an IsalSR instruction string as a horizontal token strip.

Each token occupies one coloured cell. The cell colour is determined by the
token kind: movement tokens are blue/green, edge tokens are orange/red, no-ops
are grey, and insertion tokens (``V<label>``, ``v<label>``) carry the face
colour of the :class:`~isalsr.core.node_types.NodeType` they insert, so the
strip and the DAG drawing share a mechanical colour correspondence.

Dependency rule: this module imports only :mod:`isalsr.core` and
:mod:`isalsr.viz.style`. matplotlib is imported inside function bodies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isalsr.core.string_to_dag import _tokenize
from isalsr.viz.style import color_for_token

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


def tokenize_string(instruction_string: str) -> list[str]:
    """Return the list of tokens for ``instruction_string``.

    Delegates to the core tokenizer so that the viz package never duplicates
    tokenisation logic.

    Parameters
    ----------
    instruction_string:
        A valid IsalSR instruction string such as ``"VkV+Ppc"``.

    Returns
    -------
    list[str]
        Token strings in order, e.g. ``["Vk", "V+", "P", "p", "c"]``.

    Raises
    ------
    isalsr.errors.InvalidTokenError
        If ``instruction_string`` contains an unrecognised token.
    """
    return list(_tokenize(instruction_string))


def _auto_fontsize(n_tokens: int, axis_width_inches: float) -> float:
    """Pick a label fontsize that does not overflow a token cell.

    Per-cell width in points is ``axis_width_inches * 72 / n_tokens``.
    The result is clamped to ``[5.0, 11.0]``.
    """
    if n_tokens <= 0:
        return 11.0
    per_cell_pts = axis_width_inches * 72.0 / n_tokens
    return max(5.0, min(11.0, per_cell_pts * 0.80))


def draw_instruction_strip(
    ax: Axes,
    instruction_string: str,
    *,
    cell_width: float = 1.05,
    cell_height: float = 1.1,
    show_labels: bool = True,
    label_rotation: float = 90.0,
    axis_width_inches: float | None = None,
    label_fontsize: float | None = None,
) -> list[str]:
    """Draw an instruction-string token strip on ``ax``.

    All tokens are rendered at full opacity. The function returns the token
    list so the caller can use it for assertions or annotations.

    Parameters
    ----------
    ax:
        Target matplotlib axes. The function disables all spines and ticks.
    instruction_string:
        The IsalSR instruction string to render.
    cell_width:
        Width of each token cell in axes units.
    cell_height:
        Height of each token cell in axes units.
    show_labels:
        Whether to print the token text on top of each cell.
    label_rotation:
        Text rotation in degrees (90 = vertical, readable in a narrow strip).
    axis_width_inches:
        Physical width of the axes in inches, used to auto-scale fonts.
        When ``None`` the font is not scaled.
    label_fontsize:
        Explicit fontsize override; takes precedence over ``axis_width_inches``.

    Returns
    -------
    list[str]
        The tokenised sequence, in order.
    """
    from matplotlib.patches import FancyBboxPatch

    tokens = tokenize_string(instruction_string)
    n = len(tokens)

    if label_fontsize is None:
        if axis_width_inches is not None:
            label_fontsize = _auto_fontsize(n, axis_width_inches)
        else:
            label_fontsize = 9.0

    if n == 0:
        ax.text(
            0.5,
            0.5,
            "(empty)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#888888",
            fontsize=label_fontsize,
        )
    else:
        for i, tok in enumerate(tokens):
            x = i * cell_width
            face = color_for_token(tok)
            patch = FancyBboxPatch(
                (x + 0.05, 0.05),
                cell_width - 0.10,
                cell_height - 0.10,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=face,
                edgecolor="#333333",
                lw=0.6,
            )
            ax.add_patch(patch)
            if show_labels:
                ax.text(
                    x + cell_width / 2,
                    cell_height / 2,
                    tok,
                    ha="center",
                    va="center",
                    fontsize=label_fontsize,
                    color="#111111",
                    rotation=label_rotation,
                    family="monospace",
                )

    ax.set_xlim(-0.1, n * cell_width + 0.1)
    ax.set_ylim(-0.05, cell_height + 0.05)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return tokens
