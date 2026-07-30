"""Generate the R1.3 response figure: a CONST creation edge, before and after.

Draws the labeled DAG for ``sin(x_1) + c`` in the shape a host solver delivers
it, where the constant arrives as a leaf with no incoming edge, next to its
normalisation. Every string in the figure is computed from the live
implementation rather than typed in, so the figure cannot drift from the code.

Run::

    python -m experiments.scripts.generate_fig_const_normalization --output fig.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from isalsr.core.canonical import fast_canonical_string
from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

# ----------------------------------------------------------------------------
# Palette. Muted, print-safe, and legible in greyscale.
# ----------------------------------------------------------------------------
COL_VAR = "#cfe3f7"
COL_VAR_EDGE = "#2c6fad"
COL_CONST = "#fbe3c2"
COL_CONST_EDGE = "#c17c26"
COL_OP = "#e8e8e8"
COL_OP_EDGE = "#6b6b6b"
COL_ARROW = "#4a4a4a"
COL_NEW = "#1b7f4b"
COL_ALERT = "#b02a2a"

# Sizes are chosen for the figure AFTER it is scaled to \linewidth in the
# response letter (roughly 65%), so in-figure text is deliberately large.
NODE_R = 0.37
FS_NODE = 18.0
FS_TITLE = 16.0
FS_SUBTITLE = 12.5
FS_ANNOT = 12.0
FS_FOOTER = 12.5

CONST_VALUE = 2.0
X_VALUE = 1.5

# Node id -> (x, y). Shared by both panels so the eye tracks the single change.
POS: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),  # x_1
    1: (0.0, 1.35),  # Sin
    2: (2.45, 0.0),  # Const
    3: (1.22, 2.70),  # Add
}
LABELS: dict[int, str] = {0: r"$x_1$", 1: r"$\sin$", 2: r"$c$", 3: r"$+$"}
STYLE: dict[int, tuple[str, str]] = {
    0: (COL_VAR, COL_VAR_EDGE),
    1: (COL_OP, COL_OP_EDGE),
    2: (COL_CONST, COL_CONST_EDGE),
    3: (COL_OP, COL_OP_EDGE),
}
BASE_EDGES: list[tuple[int, int]] = [(0, 1), (1, 3), (2, 3)]
NEW_EDGE: tuple[int, int] = (0, 2)


def build_host_dag() -> LabeledDAG:
    """Return sin(x_1) + c with the constant as an orphan leaf."""
    dag = LabeledDAG(16)
    x1 = dag.add_node(NodeType.VAR, var_index=0)
    sin = dag.add_node(NodeType.SIN)
    const = dag.add_node(NodeType.CONST, const_value=CONST_VALUE)
    add = dag.add_node(NodeType.ADD)
    dag.add_edge(x1, sin)
    dag.add_edge(sin, add)
    dag.add_edge(const, add)
    return dag


def _draw_node(ax: plt.Axes, node: int, *, alert: bool) -> None:
    face, edge = STYLE[node]
    x, y = POS[node]
    if alert:
        ax.add_patch(
            Circle(
                (x, y),
                NODE_R + 0.11,
                facecolor="none",
                edgecolor=COL_ALERT,
                linewidth=1.5,
                linestyle=(0, (3, 2.5)),
                zorder=2,
            )
        )
    ax.add_patch(Circle((x, y), NODE_R, facecolor=face, edgecolor=edge, linewidth=1.6, zorder=3))
    ax.text(x, y, LABELS[node], ha="center", va="center", fontsize=FS_NODE, zorder=4)


def _draw_edge(ax: plt.Axes, src: int, tgt: int, *, new: bool) -> None:
    """Draw src -> tgt, trimmed in data units so the head clears the node disc.

    Endpoints are shortened by hand rather than via ``shrinkA``/``shrinkB``,
    which are expressed in points and would need the axes scale to be known.
    The axes use an equal aspect ratio, so data units are isotropic here.
    """
    (x0, y0), (x1, y1) = POS[src], POS[tgt]
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / length, dy / length
    gap_src = NODE_R + 0.05
    gap_tgt = NODE_R + 0.09  # extra room so the head sits clear of the disc

    colour = COL_NEW if new else COL_ARROW
    ax.add_patch(
        FancyArrowPatch(
            (x0 + ux * gap_src, y0 + uy * gap_src),
            (x1 - ux * gap_tgt, y1 - uy * gap_tgt),
            arrowstyle="-|>,head_length=7,head_width=3.4",
            mutation_scale=1.0,
            linewidth=2.1 if new else 1.5,
            edgecolor=colour,
            facecolor=colour,
            zorder=1,
        )
    )


def _panel(
    ax: plt.Axes,
    *,
    title: str,
    subtitle: str,
    show_new_edge: bool,
    footer: str,
    footer_colour: str,
    footer_mono: str | None = None,
) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 3.55)
    ax.set_ylim(-1.85, 3.95)
    ax.axis("off")

    for src, tgt in BASE_EDGES:
        _draw_edge(ax, src, tgt, new=False)
    if show_new_edge:
        _draw_edge(ax, *NEW_EDGE, new=True)

    for node in POS:
        _draw_node(ax, node, alert=(node == 2 and not show_new_edge))

    ax.text(1.25, 3.68, title, ha="center", va="center", fontsize=FS_TITLE, fontweight="bold")
    ax.text(1.25, 3.22, subtitle, ha="center", va="center", fontsize=FS_SUBTITLE)

    if show_new_edge:
        ax.annotate(
            "creation edge added",
            xy=(1.22, 0.0),
            xytext=(1.22, -0.62),
            ha="center",
            va="center",
            fontsize=FS_ANNOT,
            color=COL_NEW,
            arrowprops={"arrowstyle": "-", "color": COL_NEW, "linewidth": 0.9},
        )
    else:
        ax.annotate(
            r"in-degree $0$",
            xy=(2.45, -0.42),
            xytext=(2.45, -0.86),
            ha="center",
            va="center",
            fontsize=FS_ANNOT,
            color=COL_ALERT,
            arrowprops={"arrowstyle": "-", "color": COL_ALERT, "linewidth": 0.9},
        )

    if footer_mono is None:
        ax.text(
            1.25,
            -1.55,
            footer,
            ha="center",
            va="center",
            fontsize=FS_FOOTER,
            color=footer_colour,
        )
    else:
        # Two runs, so the string keeps monospace glyphs and literal spacing:
        # mathtext would pad the '+' as a binary operator.
        ax.text(
            1.05,
            -1.55,
            footer,
            ha="right",
            va="center",
            fontsize=FS_FOOTER,
            color=footer_colour,
        )
        ax.text(
            1.18,
            -1.55,
            footer_mono,
            ha="left",
            va="center",
            fontsize=FS_FOOTER + 0.5,
            family="monospace",
            color=footer_colour,
        )


def make_figure(out_path: Path) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    before = build_host_dag()
    after = before.normalize_const_creation()

    # Everything quoted in the figure comes from the implementation.
    word = fast_canonical_string(after, backend="cpp")
    value_before = evaluate_dag(before, {0: X_VALUE})
    value_after = evaluate_dag(after, {0: X_VALUE})
    assert value_before == value_after, "normalisation must preserve evaluation"

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.2))
    _panel(
        axes[0],
        title="(a) $D$, as delivered by the host solver",
        subtitle=r"no $w \in \Sigma_{SR}^{*}$ satisfies $\mathrm{S2D}(w,1) \cong D$",
        show_new_edge=False,
        footer="no canonical string exists",
        footer_colour=COL_ALERT,
    )
    _panel(
        axes[1],
        title=r"(b) $\mathcal{N}(D)$, after normalisation",
        subtitle=r"every non-variable node is reachable from $x_1$",
        show_new_edge=True,
        footer="canonical string",
        footer_colour="black",
        footer_mono=word,
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    print(f"wrote {out_path}")
    print(f"  canonical string : {word}")
    print(f"  eval at x={X_VALUE}  : {value_before!r} (before) == {value_after!r} (after)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Destination PDF path.")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_figure(args.output)


if __name__ == "__main__":
    main()
