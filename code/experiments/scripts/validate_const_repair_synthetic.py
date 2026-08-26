"""Validate the CONST repair on a large synthetic DAG population (T15, AC-3).

Generates random IsalSR strings, decodes each with S2D, and scores the three
CONST-normalization policies on the resulting DAG:

  submitted  old aggressive relocation of every CONST in-edge onto node 0
  repair     new minimal repair of CONST nodes with in-degree 0  (production)
  none       no CONST handling at all

Reports, per policy: canonicalisation failures, and how many distinct canonical
strings the population collapses to. The second number is what separates the
policies scientifically — the old relocation merges non-isomorphic DAGs, so it
reports fewer distinct strings and a correspondingly inflated reduction factor.

Also checks the two claims the theorem now rests on:

  * the repair never removes an edge;
  * on any DAG satisfying the Round-Trip Fidelity reachability hypothesis, the
    repair is the identity, so 'repair' and 'none' agree byte-for-byte.

Usage:
    python -m experiments.scripts.validate_const_repair_synthetic --n 100000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import deque
from pathlib import Path
from typing import Any

from isalsr.core import _native
from isalsr.core.canonical import _py_dag_to_native
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.string_to_dag import StringToDAG

log = logging.getLogger("t15.synthetic")

ALPHABET = list("NPnpCcW") + [f"{ptr}{lbl}" for ptr in "Vv" for lbl in "+*-/scelr^agik"]
ARMS = ("submitted", "repair", "none")


def old_aggressive_normalize(dag: LabeledDAG) -> LabeledDAG:
    """Verbatim reimplementation of the pre-2026-07-27 relocation policy."""
    new = LabeledDAG(dag.max_nodes)
    const_nodes: set[int] = set()
    for i in range(dag.node_count):
        data = dag.node_data(i)
        new.add_node(
            dag.node_label(i),
            var_index=int(data["var_index"]) if "var_index" in data else None,
            const_value=float(data["const_value"]) if "const_value" in data else None,
        )
        if dag.node_label(i) == NodeType.CONST:
            const_nodes.add(i)
    for tgt in range(dag.node_count):
        if tgt in const_nodes:
            continue
        for src in dag.ordered_inputs(tgt):
            new.add_edge(src, tgt)
    for c in sorted(const_nodes):
        new.add_edge(0, c)
    return new


def satisfies_precondition(dag: LabeledDAG) -> bool:
    """Every non-VAR node reachable from some variable (Round-Trip Fidelity)."""
    queue = deque(i for i in range(dag.node_count) if dag.node_label(i) == NodeType.VAR)
    seen = set(queue)
    while queue:
        node = queue.popleft()
        for succ in dag.out_neighbors(node):
            if succ not in seen:
                seen.add(succ)
                queue.append(succ)
    return all(i in seen for i in range(dag.node_count) if dag.node_label(i) != NodeType.VAR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--num-variables", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = random.Random(args.seed)
    raw = _native.testing.fast_canonical_string_raw

    failures = dict.fromkeys(ARMS, 0)
    strings: dict[str, set[str]] = {arm: set() for arm in ARMS}
    n_decoded = 0
    n_precondition_ok = 0
    n_repair_dropped_edge = 0
    n_repair_none_disagree_when_precondition_ok = 0
    failure_examples: dict[str, list[str]] = {arm: [] for arm in ARMS}

    for i in range(args.n):
        word = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(4, 30)))
        try:
            dag = StringToDAG(word, num_variables=args.num_variables).run()
        except Exception:  # noqa: BLE001 - malformed random strings are expected
            continue
        n_decoded += 1

        precondition_ok = satisfies_precondition(dag)
        n_precondition_ok += precondition_ok

        repaired = dag.normalize_const_creation() if dag._has_const_nodes() else dag
        if repaired.edge_count < dag.edge_count:
            n_repair_dropped_edge += 1

        prepared = {
            "submitted": old_aggressive_normalize(dag) if dag._has_const_nodes() else dag,
            "repair": repaired,
            "none": dag,
        }
        results: dict[str, str | None] = {}
        for arm in ARMS:
            try:
                text = raw(_py_dag_to_native(prepared[arm]), args.timeout)
            except Exception:  # noqa: BLE001
                failures[arm] += 1
                results[arm] = None
                if len(failure_examples[arm]) < 10:
                    failure_examples[arm].append(word)
                continue
            results[arm] = text
            strings[arm].add(text)

        if precondition_ok and results["repair"] != results["none"]:
            n_repair_none_disagree_when_precondition_ok += 1

        if (i + 1) % 20_000 == 0:
            log.info(
                "%d/%d  failures submitted=%d repair=%d none=%d",
                i + 1,
                args.n,
                failures["submitted"],
                failures["repair"],
                failures["none"],
            )

    summary: dict[str, Any] = {
        "n_generated": args.n,
        "n_decoded": n_decoded,
        "n_satisfying_precondition": n_precondition_ok,
        "repair_dropped_an_edge": n_repair_dropped_edge,
        "repair_vs_none_disagree_when_precondition_holds": (
            n_repair_none_disagree_when_precondition_ok
        ),
        "arms": {
            arm: {
                "failures": failures[arm],
                "failure_rate": failures[arm] / n_decoded if n_decoded else 0.0,
                "distinct_canonical_strings": len(strings[arm]),
                "reduction_factor": (
                    (n_decoded - failures[arm]) / len(strings[arm])
                    if strings[arm]
                    else float("nan")
                ),
                "failure_examples": failure_examples[arm],
            }
            for arm in ARMS
        },
    }

    print(f"\ndecoded {n_decoded:,} DAGs of {args.n:,} generated strings")
    print(f"satisfying the reachability precondition: {n_precondition_ok:,}\n")
    print(f"{'arm':<12} {'failures':>10} {'fail %':>9} {'distinct':>10} {'rho':>8}")
    print("-" * 52)
    for arm in ARMS:
        a = summary["arms"][arm]
        print(
            f"{arm:<12} {a['failures']:>10,} {100 * a['failure_rate']:>8.4f}% "
            f"{a['distinct_canonical_strings']:>10,} {a['reduction_factor']:>8.3f}"
        )
    print(f"\nrepair dropped an edge on: {n_repair_dropped_edge} DAGs (must be 0)")
    print(
        "repair != none while precondition holds: "
        f"{n_repair_none_disagree_when_precondition_ok} DAGs (must be 0)"
    )

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
