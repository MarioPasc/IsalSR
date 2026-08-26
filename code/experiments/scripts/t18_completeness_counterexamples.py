"""Extract and characterise the canonical-string completeness counterexamples (T18).

Gate 3 of ``experiments/scripts/equivalence_gate.py`` (round-trip isomorphism)
fails on 5 of 10,000 generated DAGs. The failure is **engine-independent** --
Python and C++ produce byte-identical canonical strings and both fail the same
five -- so it is a property of the canonicaliser, not of the C++ port.

For each failing DAG this script records the pair ``(D, D')`` where
``D' = S2D(fcs(D))``, and the three facts that make it a counterexample to
*completeness* rather than a mere round-trip wobble:

    fcs(D) == fcs(D')        the two land in the SAME dedup class
    D !~ D'                  yet they are NOT isomorphic
    labels(D) == labels(D')  and it is not a trivial label mismatch

Run:
    python -m experiments.scripts.t18_completeness_counterexamples --out <path.md>
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _profile(dag: Any) -> dict[str, Any]:
    """Summarise a DAG's structure: labels, degrees and edge list."""
    from isalsr.core.node_types import NodeType  # noqa: PLC0415

    n = dag.node_count
    labels = [dag.node_label(v).name for v in range(n)]
    edges = sorted((u, v) for u in range(n) for v in dag.out_neighbors(u))
    return {
        "n_nodes": n,
        "n_edges": dag.edge_count,
        "label_multiset": dict(sorted(collections.Counter(labels).items())),
        "labels_by_node": labels,
        "edges": edges,
        "in_degree_zero_const": [
            v for v in range(n) if dag.node_label(v) == NodeType.CONST and dag.in_degree(v) == 0
        ],
        "var_as_edge_target": [
            v for v in range(n) if dag.node_label(v) == NodeType.VAR and dag.in_degree(v) > 0
        ],
    }


def _sympy_of(dag: Any, num_vars: int) -> str:
    """Render a DAG as a SymPy expression string, or a reason it cannot be."""
    try:
        from isalsr.adapters.sympy_adapter import labeled_dag_to_sympy  # noqa: PLC0415

        return str(labeled_dag_to_sympy(dag, num_vars))
    except Exception as exc:  # noqa: BLE001 -- diagnostic only
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def analyse(gate_json: Path) -> list[dict[str, Any]]:
    """Rebuild each failing DAG pair and record why it is a counterexample."""
    from isalsr.core.canonical import fast_canonical_string  # noqa: PLC0415
    from isalsr.core.string_to_dag import StringToDAG  # noqa: PLC0415

    gate = json.loads(gate_json.read_text())["gate3"]
    cases = {c["dag_idx"]: c for c in gate["mismatch_cases"] if c["kind"] == "roundtrip_engine_a"}

    out: list[dict[str, Any]] = []
    for idx, case in sorted(cases.items()):
        nv = case["num_vars"]
        d = StringToDAG(case["source_string"], num_variables=nv).run()
        s1_py = fast_canonical_string(d, backend="python")
        s1_cpp = fast_canonical_string(d, backend="cpp")
        d2 = StringToDAG(s1_py, num_variables=nv).run()
        s2_py = fast_canonical_string(d2, backend="python")

        out.append(
            {
                "corpus_index": idx,
                "num_vars": nv,
                "k": case["k"],
                "source_string": case["source_string"],
                "canonical_python": s1_py,
                "canonical_cpp": s1_cpp,
                "engines_agree": s1_py == s1_cpp,
                "canonical_of_decoded": s2_py,
                "same_dedup_class": s1_py == s2_py,
                "isomorphic": bool(d.is_isomorphic(d2)),
                "D": _profile(d),
                "D_prime": _profile(d2),
                "sympy_D": _sympy_of(d, nv),
                "sympy_D_prime": _sympy_of(d2, nv),
            }
        )
    return out


def render(rows: list[dict[str, Any]]) -> str:
    """Render the findings as a self-contained markdown report."""
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    lines = [
        "# T18 — canonical-string completeness: the five counterexamples, in full",
        "",
        f"**Commit**: `{head}`",
        "**Reproduce**: `python experiments/scripts/equivalence_gate.py --gate 3 "
        "--backend-a python --backend-b cpp --out gate3.json`, then "
        "`python -m experiments.scripts.t18_completeness_counterexamples`",
        "",
        "Each row is a pair `(D, D')` with `D' = S2D(fcs(D))` where **`fcs(D) == fcs(D')`**",
        "(same dedup class) and **`D ≇ D'`** (not isomorphic). Two distinct labeled DAGs",
        "therefore share one canonical string, which under-counts",
        "`unique_canonical_dags` and over-states ρ.",
        "",
        "## Summary",
        "",
        "| # | k | vars | engines agree | same class | isomorphic | in-deg-0 CONST "
        "| VAR as target |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['corpus_index']} | {r['k']} | {r['num_vars']} | "
            f"{r['engines_agree']} | {r['same_dedup_class']} | **{r['isomorphic']}** | "
            f"{len(r['D']['in_degree_zero_const'])} | {len(r['D']['var_as_edge_target'])} |"
        )

    lines += [
        "",
        "`engines agree = True` on every row is what rules out the C++ port: Python and",
        "C++ produce byte-identical strings and both fail. `in-deg-0 CONST = 0` on every",
        "row rules out the `is_isomorphic` precondition (CLAUDE.md invariant 9); the rows",
        "with `VAR as target = 0` additionally sit inside `𝒞₂`, where",
        "`normalize_const_creation` is equivariant.",
        "",
        "## Per-case detail",
        "",
    ]
    for r in rows:
        lines += [
            f"### Corpus index {r['corpus_index']} (k = {r['k']}, {r['num_vars']} variable(s))",
            "",
            "```",
            f"source string   : {r['source_string']}",
            f"fcs(D)  python  : {r['canonical_python']}",
            f"fcs(D)  cpp     : {r['canonical_cpp']}",
            f"fcs(D') python  : {r['canonical_of_decoded']}",
            "```",
            "",
            f"- `fcs(D) == fcs(D')`: **{r['same_dedup_class']}**",
            f"- `D ≅ D'`: **{r['isomorphic']}**",
            f"- labels equal: **{r['D']['label_multiset'] == r['D_prime']['label_multiset']}**",
            f"- `D`  : {r['D']['n_nodes']} nodes, {r['D']['n_edges']} edges, "
            f"{r['D']['label_multiset']}",
            f"- `D'` : {r['D_prime']['n_nodes']} nodes, {r['D_prime']['n_edges']} edges, "
            f"{r['D_prime']['label_multiset']}",
            "",
            f"- edges `D` : `{r['D']['edges']}`",
            f"- edges `D'`: `{r['D_prime']['edges']}`",
            f"- labels `D` : `{r['D']['labels_by_node']}`",
            f"- labels `D'`: `{r['D_prime']['labels_by_node']}`",
            "",
            f"- SymPy `D` : `{r['sympy_D']}`",
            f"- SymPy `D'`: `{r['sympy_D_prime']}`",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Extract the counterexamples and write the report."""
    ap = argparse.ArgumentParser(description="T18 completeness counterexample extractor")
    ap.add_argument("--gate-json", default="gate3.json", help="equivalence_gate --gate 3 output")
    ap.add_argument("--out", required=True, help="markdown report path")
    ap.add_argument("--json-out", default=None, help="optional raw JSON dump")
    args = ap.parse_args()

    rows = analyse(Path(args.gate_json))
    Path(args.out).write_text(render(rows))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=1, default=str))

    print(f"  {len(rows)} counterexamples -> {args.out}")
    for r in rows:
        print(
            f"    idx={r['corpus_index']:>5} k={r['k']:>3} "
            f"engines_agree={r['engines_agree']} same_class={r['same_dedup_class']} "
            f"isomorphic={r['isomorphic']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
