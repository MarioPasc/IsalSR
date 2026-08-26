"""Figure + data for T15: the six DAGs on which canonicalisation fails.

Produces a 2x3 grid, one panel per failing DAG, and a per-algorithm failure
table. The panels highlight the confirmed mechanism: ``normalize_const_creation``
(Critical Invariant 9) relocates every CONST creation edge onto node 0 (x_1);
when x_1 already lies on a directed path back from that CONST, the relocation
closes a cycle and D2S has no legal move left.

Run:
    python -m experiments.scripts.generate_fig_d2s_failures --out <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import deque
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.scripts._figure_helpers import _compute_dag_layout, draw_dag  # noqa: E402
from isalsr.core.canonical import (  # noqa: E402
    _fast_canonical_d2s,
    canonical_string,
    fast_canonical_string,
    pruned_canonical_string,
)
from isalsr.core.labeled_dag import LabeledDAG  # noqa: E402
from isalsr.core.node_types import NodeType  # noqa: E402
from isalsr.core.string_to_dag import StringToDAG  # noqa: E402

log = logging.getLogger(__name__)

MOVES = list("NPnpCcW")
LABELS = list("+*-/scelra^gik")
NUM_VARS = 2
GEN_SEED = 31
N_SAMPLES = 4000
TIMEOUT = 10.0


def generate_corpus() -> tuple[list[tuple[str, LabeledDAG]], list[tuple[str, LabeledDAG]]]:
    """Regenerate the deterministic corpus and split it by canonicalisation outcome.

    Returns
    -------
    tuple
        ``(failures, successes)``, each a list of ``(source_string, dag)``.
    """
    rng = random.Random(GEN_SEED)
    failures: list[tuple[str, LabeledDAG]] = []
    successes: list[tuple[str, LabeledDAG]] = []
    for _ in range(N_SAMPLES):
        s = "".join(
            rng.choice(["V", "v"]) + rng.choice(LABELS)
            if rng.random() < 0.55
            else rng.choice(MOVES)
            for _ in range(rng.randint(6, 22))
        )
        try:
            dag = StringToDAG(s, num_variables=NUM_VARS).run()
        except Exception:  # noqa: BLE001 - invalid strings are simply not part of the corpus
            continue
        try:
            fast_canonical_string(dag, mode="wl_only", timeout=TIMEOUT)
            successes.append((s, dag))
        except Exception:  # noqa: BLE001 - the outcome under test
            failures.append((s, dag))
    return failures, successes


def _algorithms() -> dict[str, Any]:
    """Every canonicalisation entry point, keyed by the name used in the report."""
    return {
        "fast_canonical_string(mode='wl_only')": lambda d: fast_canonical_string(
            d, mode="wl_only", timeout=TIMEOUT
        ),
        "fast_canonical_string(mode='wl_tiebreak')": lambda d: fast_canonical_string(
            d, mode="wl_tiebreak", timeout=TIMEOUT
        ),
        "fast_canonical_string(mode='tuple_only')": lambda d: fast_canonical_string(
            d, mode="tuple_only", timeout=TIMEOUT
        ),
        "pruned_canonical_string": lambda d: pruned_canonical_string(d, timeout=TIMEOUT),
        "canonical_string (exhaustive)": lambda d: canonical_string(d, timeout=TIMEOUT),
        "_fast_canonical_d2s (no normalisation)": lambda d: _fast_canonical_d2s(
            d, timeout=TIMEOUT, mode="wl_only"
        ),
    }


def per_algorithm_table(dags: list[LabeledDAG]) -> dict[str, dict[str, Any]]:
    """Failure count per canonicalisation algorithm over the given DAGs."""
    out: dict[str, dict[str, Any]] = {}
    for name, fn in _algorithms().items():
        failed, errors = 0, []
        for d in dags:
            try:
                fn(d)
            except Exception as exc:  # noqa: BLE001 - the outcome under test
                failed += 1
                errors.append(type(exc).__name__)
        out[name] = {
            "failed": failed,
            "total": len(dags),
            "exception_types": sorted(set(errors)),
        }
    return out


def const_creation_edges(dag: LabeledDAG) -> set[tuple[int, int]]:
    """Edges that ``normalize_const_creation`` would relocate onto node 0."""
    edges: set[tuple[int, int]] = set()
    for c in range(dag.node_count):
        if dag.node_label(c) != NodeType.CONST:
            continue
        for u in dag.in_neighbors_raw(c):
            edges.add((u, c))
    return edges


def closes_cycle_after_normalisation(dag: LabeledDAG) -> list[int]:
    """CONST nodes whose relocation to node 0 would create a cycle.

    Relocating the creation edge to ``0 -> c`` closes a cycle exactly when node 0
    is already reachable from ``c`` by directed edges.
    """
    offenders = []
    for c in range(dag.node_count):
        if dag.node_label(c) != NodeType.CONST:
            continue
        seen, q = {c}, deque([c])
        while q:
            for v in dag.out_neighbors_raw(q.popleft()):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        if 0 in seen:
            offenders.append(c)
    return offenders


def var_nodes_with_in_edges(dag: LabeledDAG) -> list[int]:
    """VAR nodes carrying at least one in-edge (semantically they are sources)."""
    return [
        i
        for i in range(dag.node_count)
        if dag.node_label(i) == NodeType.VAR and dag.in_degree(i) > 0
    ]


def _layout_extent(pos: dict[int, tuple[float, float]]) -> float:
    """Largest side of the layout's bounding box (min 1.0 to avoid divide-by-zero)."""
    if not pos:
        return 1.0
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    return max(max(xs) - min(xs), max(ys) - min(ys), 1.0)


def build_figure(failures: list[tuple[str, LabeledDAG]], out_path: str) -> None:
    """Render the 2x3 grid of failing DAGs.

    ``node_size`` is in data units while each DAG's layout extent differs by an
    order of magnitude, so a fixed value makes a 3-node DAG all circles and a
    wide star all specks. Scaling the radius by the layout extent keeps the
    panels visually comparable.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.0), constrained_layout=True)
    for idx, ax in enumerate(axes.flat):
        if idx >= len(failures):
            ax.axis("off")
            continue
        src, dag = failures[idx]
        k = dag.node_count - NUM_VARS
        offenders = closes_cycle_after_normalisation(dag)
        var_in = var_nodes_with_in_edges(dag)
        ghost = {e for e in const_creation_edges(dag) if e[1] in offenders}

        pos = _compute_dag_layout(dag)
        node_size = max(0.055 * _layout_extent(pos), 0.16)

        draw_dag(ax, dag, pos=pos, ghost_edges=ghost, node_size=node_size)
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.18)

        disp = src if len(src) <= 40 else src[:37] + "..."
        ax.set_title(
            f"#{idx}   k = {k},  |E| = {dag.edge_count}\n"
            f"CONST closing a cycle: {offenders or '—'}    "
            f"VAR with in-edges: {var_in or '—'}\n"
            f"{disp}",
            fontsize=8.5,
            linespacing=1.5,
            pad=6,
            family="sans-serif",
        )

    fig.suptitle(
        "The six DAGs on which canonicalisation fails\n"
        "dashed grey = CONST creation edge that $\\mathrm{normalize\\_const\\_creation}$ "
        "relocates onto $x_1$, closing a cycle",
        fontsize=11.5,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/md_files/changes", help="output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    failures, successes = generate_corpus()
    log.info("corpus: %d failures, %d successes", len(failures), len(successes))

    fig_path = os.path.join(args.out, "fig_d2s_failures.png")
    build_figure(failures, fig_path)

    fail_dags = [d for _, d in failures]
    table = per_algorithm_table(fail_dags)

    report: dict[str, Any] = {
        "generator": {"seed": GEN_SEED, "n_samples": N_SAMPLES, "num_vars": NUM_VARS},
        "n_failures": len(failures),
        "n_successes": len(successes),
        "per_algorithm_on_failing_dags": table,
        "cases": [
            {
                "index": i,
                "source_string": s,
                "k": d.node_count - NUM_VARS,
                "n_edges": d.edge_count,
                "nodes": [(j, d.node_label(j).name) for j in range(d.node_count)],
                "edges": [(u, v) for u in range(d.node_count) for v in d.out_neighbors_raw(u)],
                "var_nodes_with_in_edges": var_nodes_with_in_edges(d),
                "const_closing_cycle_after_normalisation": closes_cycle_after_normalisation(d),
            }
            for i, (s, d) in enumerate(failures)
        ],
    }
    json_path = os.path.join(args.out, "fig_d2s_failures_data.json")
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)
    log.info("wrote %s", json_path)

    print("\nper-algorithm failures on the 6 failing DAGs")
    print(f"{'algorithm':<44} {'failed':>8} {'exceptions'}")
    print("-" * 78)
    for name, r in table.items():
        print(
            f"{name:<44} {r['failed']:>3}/{r['total']:<4} {','.join(r['exception_types']) or '-'}"
        )


if __name__ == "__main__":
    main()
