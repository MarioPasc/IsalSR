"""Regenerate the S2D and D2S execution-trace figures of the manuscript.

Produces ``fig_s2d`` and ``fig_d2s`` (PDF and PNG) from live code: the traces
are taken from :class:`~isalsr.core.string_to_dag.StringToDAG` and
:class:`~isalsr.core.dag_to_string.DAGToString` themselves, so the figures
cannot drift from the implementation they illustrate.

All drawing is delegated to :mod:`isalsr.viz`; this script only selects the
example, picks which steps to display, and writes the files.

Round-trip check.  The example expression is round-tripped before anything is
drawn: ``D2S(S2D(w)) == w`` must hold for the canonical string ``w``, otherwise
the two figures would illustrate a property the code does not have and the
script aborts.

Usage
-----
    python -m experiments.scripts.generate_fig_algorithms --output-dir docs/generated/figures
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from isalsr.core.dag_to_string import DAGToString
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.string_to_dag import StringToDAG
from isalsr.viz import (
    DAG_TO_STRING,
    STRING_TO_DAG,
    AlgorithmTraceLayout,
    evenly_spaced_steps,
    make_algorithm_trace_figure,
    tokenize_string,
)
from isalsr.viz.algorithm_trace import Snapshot

logger = logging.getLogger(__name__)

#: Canonical string of the running example, cos(x_1) + sin(x_1) * x_2.
CANONICAL_STRING: str = "VcVspv*pv+PpcnnC"

#: Number of input variables of the running example.
NUM_VARIABLES: int = 2

#: Number of steps displayed per figure.
N_COLUMNS: int = 4


def build_example() -> tuple[LabeledDAG, list[Snapshot], list[Snapshot], str]:
    """Run both algorithms on the example and return their traces.

    Returns
    -------
    tuple[LabeledDAG, list[Snapshot], list[Snapshot], str]
        The target DAG, the full S2D trace, the full D2S trace, and the
        string D2S emitted.

    Raises
    ------
    RuntimeError
        If the round trip ``D2S(S2D(w)) == w`` fails.
    """
    s2d = StringToDAG(CANONICAL_STRING, num_variables=NUM_VARIABLES)
    dag = s2d.run(trace=True)

    d2s = DAGToString(dag, initial_node=0)
    emitted = d2s.run(trace=True)

    if emitted != CANONICAL_STRING:
        raise RuntimeError(
            f"round trip failed: D2S(S2D({CANONICAL_STRING!r})) == {emitted!r}. "
            "The figures would illustrate a property the code does not have."
        )

    s2d_trace: list[Snapshot] = [
        (d, c, p, q, "".join(str(t) for t in toks)) for d, c, p, q, toks in s2d._trace_log
    ]
    d2s_trace: list[Snapshot] = list(d2s.trace_log)
    return dag, s2d_trace, d2s_trace, emitted


def _labels_for(indices: list[int]) -> list[str]:
    """Return column titles for ``indices``.

    Step 0 is the state before any instruction has been consumed (S2D) or
    emitted (D2S).
    """
    return [f"Step {i}" for i in indices]


def save(fig: object, base: Path) -> None:
    """Write ``fig`` to ``base.pdf`` and ``base.png`` with a transparent canvas."""
    import matplotlib.pyplot as plt

    base.parent.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("pdf", 600), ("png", 400)):
        fig.savefig(  # type: ignore[attr-defined]
            base.with_suffix(f".{ext}"),
            dpi=dpi,
            transparent=True,
            facecolor="none",
            edgecolor="none",
        )
    plt.close(fig)  # type: ignore[arg-type]
    logger.info("wrote %s.{pdf,png}", base)


def main() -> None:
    """Build and write both figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/generated/figures"),
        help="Directory to write fig_s2d and fig_d2s into.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=7.16,
        help="Final printed width in inches (IEEE two-column \\textwidth).",
    )
    parser.add_argument(
        "--rankdir",
        default="LR",
        choices=["LR", "BT", "TB", "RL"],
        help="Graphviz rank direction for the DAG panels.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dag, s2d_trace, d2s_trace, final_string = build_example()
    logger.info(
        "example: %d nodes, %d edges, |w| = %d tokens",
        dag.node_count,
        dag.edge_count,
        len(tokenize_string(final_string)),
    )

    lay = AlgorithmTraceLayout(fig_width=args.fig_width, rankdir=args.rankdir)

    s2d_idx = evenly_spaced_steps(len(s2d_trace), N_COLUMNS)
    fig = make_algorithm_trace_figure(
        [s2d_trace[i] for i in s2d_idx],
        dag,
        final_string=final_string,
        step_labels=_labels_for(s2d_idx),
        direction=STRING_TO_DAG,
        layout=lay,
    )
    save(fig, args.output_dir / "fig_s2d")

    d2s_idx = evenly_spaced_steps(len(d2s_trace), N_COLUMNS)
    fig = make_algorithm_trace_figure(
        [d2s_trace[i] for i in d2s_idx],
        dag,
        final_string=final_string,
        step_labels=_labels_for(d2s_idx),
        direction=DAG_TO_STRING,
        layout=lay,
    )
    save(fig, args.output_dir / "fig_d2s")


if __name__ == "__main__":
    main()
