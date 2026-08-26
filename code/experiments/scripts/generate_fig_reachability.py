"""Generate the R1.2 response figure for the reachability condition.

Two modes controlled by --style:

  trace (default)
      A 2-row x 4-column figure showing the D2S algorithm running
      step-by-step on two versions of x_1 + c:
        Row 0 -- violated DAG (no creation edge x_1 -> c).
                 The algorithm stalls at column 2, shown with a red frame
                 and a ghost rendering of the unreachable CONST node.
        Row 1 -- normalised DAG (creation edge added by normalize_const_creation).
                 The algorithm runs to completion in 4 steps.

  simple
      The original two-panel before/after figure kept for reference.

Every string, pointer position, and property shown in the trace figure is
computed from the live IsalSR implementation.  The script asserts these
values before drawing so the figure cannot drift from the code.

Usage::

    python -m experiments.scripts.generate_fig_reachability \\
        --output <path.pdf>            # trace style (default)
    python -m experiments.scripts.generate_fig_reachability \\
        --output <path.pdf> --style simple
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from isalsr.core.canonical import fast_canonical_string
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.viz import (
    DEFAULT_TRACE_LAYOUT,
    draw_instruction_strip,
    make_trace_figure,
)
from isalsr.viz.backends.matplotlib_dag import MatplotlibDagBackend

# ---------------------------------------------------------------------------
# Fixed layout shared by both figures: same node positions, same expression.
# Node IDs in the INPUT dag: 0 = x_1 (VAR), 1 = c (CONST), 2 = + (ADD).
# ---------------------------------------------------------------------------
LAYOUT: dict[int, tuple[float, float]] = {
    0: (-1.2, 0.0),  # x_1 -- bottom left
    1: (1.2, 0.0),  # c   -- bottom right
    2: (0.0, 2.2),  # +   -- top centre
}

# Expected D2S string and canonical string (verified below against live code).
_D2S_STRING: str = "VkV+PnC"
_CANONICAL_STRING: str = "V+VkPnc"


# ---------------------------------------------------------------------------
# DAG construction
# ---------------------------------------------------------------------------


def _build_normalised_dag() -> tuple[LabeledDAG, int]:
    """Return (dag, x1_node) for the normalised x_1 + c.

    The CONST node c carries the creation edge x_1 -> c so every
    non-variable node is reachable from x_1.
    """
    dag = LabeledDAG(8)
    x1 = dag.add_node(NodeType.VAR, var_index=0)
    c = dag.add_node(NodeType.CONST)
    add = dag.add_node(NodeType.ADD)
    dag.add_edge(x1, c)  # creation edge (x_1 -> c)
    dag.add_edge(x1, add)  # x_1 is first operand of +
    dag.add_edge(c, add)  # c is second operand of +
    return dag, x1


def _build_violated_dag() -> tuple[LabeledDAG, int]:
    """Return (dag, x1_node) for the violated x_1 + c.

    c has in-degree 0: no path from any variable reaches it.  This is the
    form a host SR solver delivers; normalisation has not been applied.
    """
    dag = LabeledDAG(8)
    x1 = dag.add_node(NodeType.VAR, var_index=0)
    c = dag.add_node(NodeType.CONST)
    add = dag.add_node(NodeType.ADD)
    dag.add_edge(x1, add)
    dag.add_edge(c, add)
    return dag, x1


# ---------------------------------------------------------------------------
# Reachability helper (used by the simple figure)
# ---------------------------------------------------------------------------


def _reachable_from_variables(dag: LabeledDAG) -> frozenset[int]:
    """Return the set of non-VAR nodes reachable from any variable via out-edges."""
    visited: set[int] = set()
    queue: list[int] = []
    for node in range(dag.node_count):
        if dag.node_label(node) is NodeType.VAR:
            queue.append(node)
    while queue:
        nd = queue.pop()
        for tgt in dag.out_neighbors(nd):
            if tgt not in visited:
                visited.add(tgt)
                queue.append(tgt)
    return frozenset(nd for nd in visited if dag.node_label(nd) is not NodeType.VAR)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _verify_all(dag_n: LabeledDAG, x1_n: int, dag_v: LabeledDAG, x1_v: int) -> str:
    """Assert all claims made in the figure, on both backends.

    Returns the D2S string of the normalised DAG (verified equal on both
    backends and matching the expected constant ``_D2S_STRING``).

    Prints every verified value for the audit trail.
    """
    # ---- D2S on normalised (Python; greedy, not canonical) ----
    d2s_n = DAGToString(dag_n, initial_node=x1_n).run()
    print(f"  D2S string (normalised): {d2s_n!r}")
    assert d2s_n == _D2S_STRING, f"D2S string mismatch: expected {_D2S_STRING!r}, got {d2s_n!r}"

    # ---- Canonical string on normalised — both backends ----
    for backend in ("python", "cpp"):
        cs = fast_canonical_string(dag_n, backend=backend)  # type: ignore[arg-type]
        print(f"  canonical ({backend}): {cs!r}")
        assert cs == _CANONICAL_STRING, (
            f"canonical mismatch on {backend}: expected {_CANONICAL_STRING!r}, got {cs!r}"
        )
    assert fast_canonical_string(dag_n, backend="python") == fast_canonical_string(
        dag_n, backend="cpp"
    ), "python and cpp backends disagree"

    # ---- Default run on violated raises ValueError ----
    for backend in ("python", "cpp"):
        refused = False
        try:
            fast_canonical_string(dag_v, backend=backend)  # type: ignore[arg-type]
        except (ValueError, RuntimeError) as exc:
            refused = True
            print(f"  [{backend}] refused violated DAG: {type(exc).__name__}")
        assert refused, f"Expected canonicalisation of violated DAG to raise on {backend}"

    conv_v = DAGToString(dag_v, initial_node=x1_v)
    try:
        conv_v.run()
        raise AssertionError("Expected ValueError from DAGToString.run() on violated DAG")
    except ValueError as exc:
        print(f"  DAGToString default refused violated: {type(exc).__name__}")

    # ---- Bypass raises RuntimeError (stall) ----
    conv_bypass = DAGToString(dag_v, initial_node=x1_v)
    try:
        conv_bypass.run(trace=True, skip_precondition_check=True)
        raise AssertionError("Expected RuntimeError from bypass run on violated DAG")
    except RuntimeError as exc:
        print(f"  bypass stall: {type(exc).__name__}: {str(exc)[:60]}")
    trace_v = conv_bypass.trace_log
    assert len(trace_v) == 2, f"Expected 2 violated snapshots, got {len(trace_v)}"
    print(f"  violated trace length: {len(trace_v)} (expected 2)")

    # ---- Normalised trace has 4 snapshots ----
    conv_n = DAGToString(dag_n, initial_node=x1_n)
    conv_n.run(trace=True)
    trace_n = conv_n.trace_log
    assert len(trace_n) == 4, f"Expected 4 normalised snapshots, got {len(trace_n)}"
    print(f"  normalised trace length: {len(trace_n)} (expected 4)")

    # Print snapshot summaries for the ledger.
    for i, (odag, cdll, pp, sp, s) in enumerate(trace_v):
        print(
            f"  violated snap[{i}]: nodes={odag.node_count}, "
            f"edges={odag.edge_count}, emitted={s!r}, "
            f"pri_graph={cdll.get_value(pp)}, sec_graph={cdll.get_value(sp)}"
        )
    for i, (odag, cdll, pp, sp, s) in enumerate(trace_n):
        print(
            f"  normalised snap[{i}]: nodes={odag.node_count}, "
            f"edges={odag.edge_count}, emitted={s!r}, "
            f"pri_graph={cdll.get_value(pp)}, sec_graph={cdll.get_value(sp)}"
        )

    return d2s_n


# ---------------------------------------------------------------------------
# Trace figure (--style trace, default)
# ---------------------------------------------------------------------------


def _collect_traces(
    dag_n: LabeledDAG,
    x1_n: int,
    dag_v: LabeledDAG,
    x1_v: int,
) -> tuple[list, list]:  # type: ignore[type-arg]
    """Collect trace snapshots for both DAGs.

    Returns
    -------
    tuple[list, list]
        ``(trace_violated, trace_normalised)``
    """
    # Violated: bypass precondition check; catch the expected stall RuntimeError.
    import contextlib

    conv_v = DAGToString(dag_v, initial_node=x1_v)
    with contextlib.suppress(RuntimeError):
        conv_v.run(trace=True, skip_precondition_check=True)
    trace_v = conv_v.trace_log

    # Normalised: standard run.
    conv_n = DAGToString(dag_n, initial_node=x1_n)
    conv_n.run(trace=True)
    trace_n = conv_n.trace_log

    return trace_v, trace_n


def make_trace_figure_script(out_path: Path) -> None:
    """Build and save the trace figure."""
    dag_n, x1_n = _build_normalised_dag()
    dag_v, x1_v = _build_violated_dag()

    print("[verify]")
    d2s_str = _verify_all(dag_n, x1_n, dag_v, x1_v)

    print("[trace collection]")
    trace_v, trace_n = _collect_traces(dag_n, x1_n, dag_v, x1_v)

    print("[figure]")
    lo = DEFAULT_TRACE_LAYOUT
    fig = make_trace_figure(
        trace_v,
        dag_v,
        trace_n,
        dag_n,
        d2s_string=d2s_str,
        input_layout=LAYOUT,
        ax_xlim=(-2.3, 2.3),
        ax_ylim=(-1.3, 3.5),
        layout_obj=lo,
    )

    # Annotation: note below the figure identifying D2S vs canonical.
    # Plain text avoids a usetex dependency.
    note_plain = f"D2S string (greedy from x₁): {d2s_str}    canonical: {_CANONICAL_STRING}"
    fig.text(
        0.5,
        0.01,
        note_plain,
        ha="center",
        va="bottom",
        fontsize=11.0,
        color="#444444",
    )

    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Simple figure (--style simple) — original two-panel
# ---------------------------------------------------------------------------


def _make_simple_figure(out_path: Path) -> None:
    """Build and write the original two-panel figure."""
    dag_n, x1_n = _build_normalised_dag()
    dag_v, x1_v = _build_violated_dag()

    word = fast_canonical_string(dag_n, backend="python")  # type: ignore[arg-type]
    print(f"  canonical string (simple): {word!r}")

    # Verify both backends agree.
    assert word == fast_canonical_string(dag_n, backend="cpp"), "backend mismatch"  # type: ignore[arg-type]

    reachable_a = _reachable_from_variables(dag_n)

    fig = plt.figure(figsize=(9.6, 5.8))
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    gs = fig.add_gridspec(2, 2, height_ratios=[4.2, 1.0], hspace=0.05, wspace=0.08)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_strip = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    backend = MatplotlibDagBackend()

    backend.draw(dag_n, ax_a, reachable=reachable_a, layout=LAYOUT)
    ax_a.set_xlim(-2.3, 2.3)
    ax_a.set_ylim(-1.3, 3.5)
    ax_a.text(
        0.5,
        1.01,
        r"(a) $\mathcal{N}(D)$: precondition satisfied",
        transform=ax_a.transAxes,
        ha="center",
        va="bottom",
        fontsize=16.0,
        fontweight="bold",
    )

    backend.draw(dag_v, ax_b, reachable=frozenset(), layout=LAYOUT)
    ax_b.set_xlim(-2.3, 2.3)
    ax_b.set_ylim(-1.3, 3.5)
    ax_b.text(
        0.5,
        1.01,
        r"(b) $D$: precondition violated",
        transform=ax_b.transAxes,
        ha="center",
        va="bottom",
        fontsize=16.0,
        fontweight="bold",
    )

    draw_instruction_strip(ax_strip, word, label_fontsize=9.5, label_rotation=0.0)

    for spine in ax_text.spines.values():
        spine.set_visible(False)
    ax_text.set_xticks([])
    ax_text.set_yticks([])
    ax_text.add_patch(
        FancyBboxPatch(
            (0.03, 0.10),
            0.94,
            0.80,
            transform=ax_text.transAxes,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor="#fce8e8",
            edgecolor="#b02a2a",
            lw=1.0,
        )
    )
    ax_text.text(
        0.5,
        0.5,
        r"no string over $\Sigma_{SR}$ encodes this DAG",
        transform=ax_text.transAxes,
        ha="center",
        va="center",
        fontsize=12.0,
        color="#b02a2a",
    )

    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Destination PDF path.")
    parser.add_argument(
        "--style",
        choices=["trace", "simple"],
        default="trace",
        help=(
            "'trace' (default): 2-row D2S step-by-step figure. "
            "'simple': original two-panel before/after figure."
        ),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.style == "trace":
        make_trace_figure_script(args.output)
    else:
        _make_simple_figure(args.output)


if __name__ == "__main__":
    main()
