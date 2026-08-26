"""Regenerate the supplementary shortest-path and neighbourhood figures.

Both figures are rebuilt from live code through :mod:`isalsr.viz`, so they share
one visual system with the manuscript's S2D/D2S trace figures: Graphviz ``dot``
layout with node-avoiding splines, the :mod:`isalsr.viz.style` palette, glyphs
auto-fitted to their discs, a transparent canvas, and banded rows.

The *data* logic is imported unchanged from the original generators
(``generate_fig_shortest_path`` and ``generate_fig_neighbourhood``); only the
drawing is replaced. Nothing here recomputes an edit path or a neighbour set,
so the figures cannot disagree with the numbers those scripts produce.

Variable indexing.  The manuscript body writes variables 1-based
($x_1 \\dots x_m$) and :mod:`isalsr.viz` follows it, but
:class:`~isalsr.adapters.sympy_adapter.SympyAdapter` emits 0-based symbols.
Rendering its LaTeX unchanged would print ``x_0`` in the expression row beneath
a disc labelled ``x_1`` -- the same figure contradicting itself -- so the
symbols are shifted before the expression is typeset.

Usage
-----
    python -m experiments.scripts.generate_fig_supplementary \\
        --output-dir docs/generated/figures
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from isalsr.core.canonical import levenshtein
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.string_to_dag import StringToDAG
from isalsr.viz import (
    BY_ROW,
    BY_VIEW,
    ExpressionCell,
    ExpressionGridLayout,
    ExpressionRow,
    make_expression_grid_figure,
)

logger = logging.getLogger(__name__)

#: Maximum number of path steps shown per row of the shortest-path figure.
MAX_PATH_COLUMNS: int = 4


# ---------------------------------------------------------------------------
# Expression rendering
# ---------------------------------------------------------------------------


def _shift_variables(expr: Any) -> Any:
    """Return ``expr`` with every ``x_i`` renamed to ``x_{i+1}``.

    :class:`~isalsr.adapters.sympy_adapter.SympyAdapter` names variables from
    zero, while the manuscript body and the drawn node discs number them from
    one.  Without this the expression row would print ``x_0`` directly beneath
    a disc labelled ``x_1``.
    """
    import sympy

    subs = {
        sym: sympy.Symbol(f"x_{{{int(str(sym).split('_')[1]) + 1}}}")
        for sym in expr.free_symbols
        if str(sym).startswith("x_") and str(sym).split("_")[1].isdigit()
    }
    return expr.subs(subs) if subs else expr


def dag_to_latex(dag: LabeledDAG | None) -> str:
    """Return the DAG's expression as LaTeX, covering **every** sink.

    An intermediate on an edit path need not be a single expression: several
    of them decode to a forest.  ``to_sympy`` reports only ``output_node()``,
    so labelling such a DAG with it would describe one component and silently
    drop the others that the panel visibly draws.  Every non-variable sink is
    therefore rendered, joined by commas.

    Parameters
    ----------
    dag:
        The DAG to convert, or ``None``.

    Returns
    -------
    str
        LaTeX body without ``$`` delimiters; empty when conversion fails.
    """
    if dag is None:
        return ""
    import sympy

    from isalsr.adapters.sympy_adapter import SympyAdapter
    from isalsr.core.node_types import NodeType

    try:
        exprs = SympyAdapter().node_expressions(dag)
    except Exception:
        return ""

    sinks = [
        n
        for n in range(dag.node_count)
        if not list(dag.out_neighbors(n)) and dag.node_label(n) is not NodeType.VAR
    ]
    if not sinks:
        return ""
    parts = [str(sympy.latex(_shift_variables(exprs[n]))) for n in sinks if n in exprs]
    return r",\; ".join(parts)


def arity_violations(dag: LabeledDAG | None) -> frozenset[int]:
    """Return nodes whose in-degree exceeds their operator's arity.

    Such a node does not denote a single-valued expression: the SymPy adapter
    consumes only the first ``arity`` inputs, so the expression printed beneath
    the panel silently omits edges the panel draws. Flagging them keeps the
    figure honest about which intermediates are well-formed.

    Parameters
    ----------
    dag:
        The DAG to check, or ``None``.

    Returns
    -------
    frozenset[int]
        Offending node IDs.
    """
    if dag is None:
        return frozenset()
    from isalsr.core.node_types import ARITY_MAP, NodeType

    bad: set[int] = set()
    for node in range(dag.node_count):
        label = dag.node_label(node)
        if label in (NodeType.VAR, NodeType.CONST):
            continue
        arity = ARITY_MAP.get(label)
        if arity is not None and dag.in_degree(node) > arity:
            bad.add(node)
    return frozenset(bad)


def decode(string: str, num_vars: int) -> LabeledDAG | None:
    """Return the DAG a string decodes to, or ``None`` when it does not decode."""
    try:
        dag = StringToDAG(string, num_variables=num_vars).run()
    except Exception:
        return None
    return dag if dag.node_count > num_vars else None


# ---------------------------------------------------------------------------
# Shortest-path figure
# ---------------------------------------------------------------------------


def build_shortest_path() -> tuple[list[ExpressionRow], int]:
    """Build the rows for the Levenshtein shortest-path figure.

    Returns
    -------
    tuple[list[ExpressionRow], int]
        The single grid row, and the edit distance between the endpoints.
    """
    from experiments.scripts._figure_helpers import levenshtein_with_backtrace
    from experiments.scripts.generate_fig_shortest_path import (
        NUM_VARS,
        SOURCE_CANON,
        TARGET_CANON,
        _reconstruct_intermediates,
    )

    dist, ops = levenshtein_with_backtrace(SOURCE_CANON, TARGET_CANON)
    strings = _reconstruct_intermediates(SOURCE_CANON, TARGET_CANON, ops)
    logger.info("edit distance %d over %d intermediates", dist, len(strings))

    cells: list[ExpressionCell] = []
    for i, s in enumerate(strings):
        dag = decode(s, NUM_VARS)
        if i == 0:
            title, emph = "Source", True
        elif i == len(strings) - 1:
            title, emph = "Target", True
        else:
            title, emph = f"Step {i}", False
        cells.append(
            ExpressionCell(
                dag=dag,
                instruction_string=s,
                title=title,
                math_latex=dag_to_latex(dag),
                emphasise_title=emph,
                alert_nodes=arity_violations(dag),
            )
        )
    # Wrap onto several rows. Seven cells across one text width leaves each DAG
    # panel about 0.9 in wide, at which point the rank separation that keeps an
    # arrow visibly longer than its own head no longer fits beside the discs and
    # the edges degenerate into arrowheads. Fewer columns per row is the only
    # lever that widens the panels, since the figure's width is fixed.
    rows = [
        ExpressionRow(label="", cells=cells[i : i + MAX_PATH_COLUMNS])
        for i in range(0, len(cells), MAX_PATH_COLUMNS)
    ]
    return rows, dist


# ---------------------------------------------------------------------------
# Neighbourhood figure
# ---------------------------------------------------------------------------


def build_neighbourhood() -> tuple[list[ExpressionRow], dict[str, int]]:
    """Build the rows for the Levenshtein-1 neighbourhood figure.

    The base occupies its own row rather than the first cell of the
    substitution row, which would place it inside the substitution band and so
    assert that the base is one of its own neighbours.  It still shares the
    grid's single DAG scale; the original figure drew it in a separate panel,
    which made its discs a different size from every neighbour's and invited
    the reader to see significance in that.

    Returns
    -------
    tuple[list[ExpressionRow], dict[str, int]]
        The three grid rows, and the neighbour count per edit type.
    """
    from experiments.scripts.generate_fig_neighbourhood import (
        BASE_CANONICAL,
        NUM_VARS,
        SUBSTITUTION_NEIGHBOURS,
        _generate_all_lev1_neighbours,
        _select_neighbours,
    )

    alphabet = list("NPnpCcW") + [f"V{c}" for c in "+*-/scelr^agik"]
    all_nb = _generate_all_lev1_neighbours(BASE_CANONICAL, NUM_VARS, alphabet)
    counts = {k: len(v) for k, v in all_nb.items()}
    logger.info("neighbours: %s", counts)

    auto = _select_neighbours(all_nb, n_per_type=3)

    def cell_for(canon: str, title: str) -> ExpressionCell:
        dag = decode(canon, NUM_VARS)
        return ExpressionCell(
            dag=dag,
            instruction_string=canon,
            title=title,
            math_latex=dag_to_latex(dag),
            alert_nodes=arity_violations(dag),
        )

    base_dag = decode(BASE_CANONICAL, NUM_VARS)
    base_cell = ExpressionCell(
        dag=base_dag,
        instruction_string=BASE_CANONICAL,
        title=r"$G_0$",
        math_latex=dag_to_latex(base_dag),
        emphasise_title=True,
        alert_nodes=arity_violations(base_dag),
    )

    sub_cells = [
        cell_for(c, f"$d={levenshtein(BASE_CANONICAL, c)}$") for c in SUBSTITUTION_NEIGHBOURS
    ]
    ins_cells = [cell_for(n["canon"], f"$d={n['lev_dist']}$") for n in auto["insertion"]]
    del_cells = [cell_for(n["canon"], f"$d={n['lev_dist']}$") for n in auto["deletion"]]

    return (
        [
            ExpressionRow(label="Base", cells=[base_cell]),
            ExpressionRow(label="Substitution", cells=sub_cells),
            ExpressionRow(label="Insertion", cells=ins_cells),
            ExpressionRow(label="Deletion", cells=del_cells),
        ],
        counts,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def save(fig: Any, base: Path) -> None:
    """Write ``fig`` to ``base.pdf`` and ``base.png`` with a transparent canvas."""
    import matplotlib.pyplot as plt

    base.parent.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("pdf", 600), ("png", 400)):
        fig.savefig(
            base.with_suffix(f".{ext}"),
            dpi=dpi,
            transparent=True,
            facecolor="none",
            edgecolor="none",
        )
    plt.close(fig)
    logger.info("wrote %s.{pdf,png}", base)


def main() -> None:
    """Build and write both supplementary figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/generated/figures"),
        help="Directory to write fig_shortest_path and fig_neighbourhood into.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=7.16,
        help="Final printed width in inches (IEEE two-column \\textwidth).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sp_rows, dist = build_shortest_path()
    fig = make_expression_grid_figure(
        sp_rows,
        band_mode=BY_VIEW,
        layout=ExpressionGridLayout(fig_width=args.fig_width),
    )
    save(fig, args.output_dir / "fig_shortest_path")
    logger.info("shortest path: %d cells, d_Lev = %d", len(sp_rows[0].cells), dist)

    # Three cells per row rather than seven, so these panels are wide enough
    # for the row height to bind: unlike the path figure, this one does need a
    # taller DAG panel to keep the auto-fitted glyphs at full size.
    nb_rows, counts = build_neighbourhood()
    fig = make_expression_grid_figure(
        nb_rows,
        band_mode=BY_ROW,
        layout=ExpressionGridLayout(
            fig_width=args.fig_width,
            dag_height=1.22,
            ranksep=1.15,
            nodesep=0.70,
            row_gap=0.13,
        ),
    )
    save(fig, args.output_dir / "fig_neighbourhood")
    logger.info("neighbourhood: %s", counts)


if __name__ == "__main__":
    main()
