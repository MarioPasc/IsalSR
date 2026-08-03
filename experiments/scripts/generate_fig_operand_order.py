"""Generate the R2.1 response figure: why Definition 3.5 needs the first-operand
restriction.

Draws ``D = x_1^{x_2}`` next to the DAG decoded from the string ``NV^Nc``. The two
have the same node set and the same edge set, so the string qualifies under the
submitted reading of Definition 3.5, yet the two are not isomorphic because the
operand order of the ``Pow`` node differs. Every string, ordered input list and
numeric value in the figure is computed from the live implementation rather than
typed in, so the figure cannot drift from the code.

Run::

    python -m experiments.scripts.generate_fig_operand_order --output fig.pdf
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
from isalsr.core.string_to_dag import StringToDAG

# ----------------------------------------------------------------------------
# Palette. Matches the R1.3 figure so the letter reads as one document.
# ----------------------------------------------------------------------------
COL_VAR = "#cfe3f7"
COL_VAR_EDGE = "#2c6fad"
COL_OP = "#e8e8e8"
COL_OP_EDGE = "#6b6b6b"
COL_FIRST = "#1b7f4b"
COL_SECOND = "#4a4a4a"
COL_ALERT = "#b02a2a"

# Sizes are chosen for the figure AFTER it is scaled to \linewidth in the
# response letter (roughly 65%), so in-figure text is deliberately large.
NODE_R = 0.40
FS_NODE = 19.0
FS_TITLE = 16.0
FS_SUBTITLE = 12.5
FS_ANNOT = 12.0
FS_FOOTER = 12.5

WORD = "NV^Nc"
X1_VALUE = 2.0
X2_VALUE = 3.0

POS: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),  # x_1
    1: (2.30, 0.0),  # x_2
    2: (1.15, 1.75),  # Pow
}
LABELS: dict[int, str] = {0: r"$x_1$", 1: r"$x_2$", 2: r"$\wedge$"}
STYLE: dict[int, tuple[str, str]] = {
    0: (COL_VAR, COL_VAR_EDGE),
    1: (COL_VAR, COL_VAR_EDGE),
    2: (COL_OP, COL_OP_EDGE),
}


def build_pow_dag() -> LabeledDAG:
    """Return x_1 raised to x_2, with x_1 as the first operand."""
    dag = LabeledDAG(8)
    x1 = dag.add_node(NodeType.VAR, var_index=0)
    x2 = dag.add_node(NodeType.VAR, var_index=1)
    pow_node = dag.add_node(NodeType.POW)
    dag.add_edge(x1, pow_node)
    dag.add_edge(x2, pow_node)
    return dag


def _draw_node(ax: plt.Axes, node: int) -> None:
    face, edge = STYLE[node]
    x, y = POS[node]
    ax.add_patch(Circle((x, y), NODE_R, facecolor=face, edgecolor=edge, linewidth=1.6, zorder=3))
    ax.text(x, y, LABELS[node], ha="center", va="center", fontsize=FS_NODE, zorder=4)


def _draw_edge(ax: plt.Axes, src: int, tgt: int, *, rank: int) -> None:
    """Draw src -> tgt, trimmed in data units so the head clears the node disc.

    ``rank`` is the position of ``src`` in the ordered input list of ``tgt``;
    it selects the colour and the operand tag.
    """
    (x0, y0), (x1, y1) = POS[src], POS[tgt]
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / length, dy / length
    gap_src = NODE_R + 0.05
    gap_tgt = NODE_R + 0.09

    colour = COL_FIRST if rank == 0 else COL_SECOND
    ax.add_patch(
        FancyArrowPatch(
            (x0 + ux * gap_src, y0 + uy * gap_src),
            (x1 - ux * gap_tgt, y1 - uy * gap_tgt),
            arrowstyle="-|>,head_length=7,head_width=3.4",
            mutation_scale=1.0,
            linewidth=2.3 if rank == 0 else 1.5,
            edgecolor=colour,
            facecolor=colour,
            zorder=1,
        )
    )
    # Anchor the operand tag below the midpoint and offset it away from the
    # edge, so the two tags do not collide under the Pow node.
    mx, my = x0 + 0.40 * dx, y0 + 0.40 * dy
    ax.text(
        mx + (0.40 if src == 0 else -0.40),
        my - 0.04,
        "1st" if rank == 0 else "2nd",
        ha="center",
        va="center",
        fontsize=FS_ANNOT,
        color=colour,
        fontweight="bold" if rank == 0 else "normal",
    )


def _panel(
    ax: plt.Axes,
    dag: LabeledDAG,
    *,
    title: str,
    subtitle: str,
    footer: str,
    footer_mono: str,
    footer_colour: str,
) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 3.40)
    ax.set_ylim(-1.70, 3.35)
    ax.axis("off")

    order = dag.ordered_inputs(2)
    for rank, src in enumerate(order):
        _draw_edge(ax, src, 2, rank=rank)

    for node in POS:
        _draw_node(ax, node)

    ax.text(1.15, 3.08, title, ha="center", va="center", fontsize=FS_TITLE, fontweight="bold")
    ax.text(1.15, 2.62, subtitle, ha="center", va="center", fontsize=FS_SUBTITLE)

    ax.text(
        0.98,
        -1.40,
        footer,
        ha="right",
        va="center",
        fontsize=FS_FOOTER,
        color=footer_colour,
    )
    ax.text(
        1.11,
        -1.40,
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

    left = build_pow_dag()
    right = StringToDAG(WORD, 2).run()

    # Everything quoted in the figure comes from the implementation.
    order_left = left.ordered_inputs(2)
    order_right = right.ordered_inputs(2)
    fcs_left = fast_canonical_string(left, backend="cpp")
    fcs_right = fast_canonical_string(right, backend="cpp")
    iso = left.is_isomorphic(right)
    val_left = evaluate_dag(left, {0: X1_VALUE, 1: X2_VALUE})
    val_right = evaluate_dag(right, {0: X1_VALUE, 1: X2_VALUE})

    assert left.node_count == right.node_count, "the counterexample needs equal |V|"
    assert left.edge_count == right.edge_count, "the counterexample needs equal |E|"
    assert {(u, 2) for u in left.in_neighbors(2)} == {(u, 2) for u in right.in_neighbors(2)}, (
        "the counterexample needs the identical edge set"
    )
    assert order_left != order_right, "operand orders must differ"
    assert not iso, "the two DAGs must not be isomorphic"
    assert fcs_left != fcs_right, "the canonical strings must differ"

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.7))
    _panel(
        axes[0],
        left,
        title=r"(a) $D = x_1^{\,x_2}$",
        subtitle=r"$\sigma(\wedge) = (x_1, x_2)$",
        footer="canonical string",
        footer_mono=fcs_left,
        footer_colour="black",
    )
    _panel(
        axes[1],
        right,
        title=r"(b) $\mathrm{S2D}(w,2) = x_2^{\,x_1}$",
        subtitle=r"$\sigma(\wedge) = (x_2, x_1)$",
        footer="canonical string",
        footer_mono=fcs_right,
        footer_colour="black",
    )

    # Two runs, so the instruction string keeps monospace glyphs and a literal
    # caret: inside mathtext the '^' would typeset the next character as a
    # superscript.
    fig.text(
        0.492,
        0.965,
        r"$w = {}$",
        ha="right",
        va="center",
        fontsize=FS_SUBTITLE + 0.5,
    )
    fig.text(
        0.500,
        0.965,
        WORD,
        ha="left",
        va="center",
        fontsize=FS_SUBTITLE + 1.0,
        family="monospace",
    )
    fig.text(
        0.5,
        0.905,
        rf"places all ${left.node_count}$ nodes and all ${left.edge_count}$ edges "
        rf"of $D$, so $w$ lies in the valid string set of Definition 3.5 as written",
        ha="center",
        va="center",
        fontsize=FS_SUBTITLE + 0.5,
    )
    fig.text(
        0.5,
        0.045,
        r"(a) is not isomorphic to (b): the edge sets coincide but the operand "
        r"orders do not, and condition (iv) of Definition 3.9 separates them",
        ha="center",
        va="center",
        fontsize=FS_SUBTITLE + 0.5,
        color=COL_ALERT,
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.10, wspace=0.05)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    print(f"wrote {out_path}")
    print(f"  w                    : {WORD}")
    print(f"  |V|, |E| (a)         : {left.node_count}, {left.edge_count}")
    print(f"  |V|, |E| (b)         : {right.node_count}, {right.edge_count}")
    print(f"  ordered inputs (a)   : {order_left}")
    print(f"  ordered inputs (b)   : {order_right}")
    print(f"  canonical string (a) : {fcs_left}")
    print(f"  canonical string (b) : {fcs_right}")
    print(f"  is_isomorphic        : {iso}")
    print(f"  eval at ({X1_VALUE}, {X2_VALUE}) : {val_left!r} (a) vs {val_right!r} (b)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Destination PDF path.")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_figure(args.output)


if __name__ == "__main__":
    main()
