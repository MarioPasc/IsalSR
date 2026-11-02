"""The paper's central claims about the canonical string, each against an independent oracle.

Theorem 3.12 (TPAMI paper): for labeled DAGs satisfying the reachability
condition, ``fast_canonical_string(D1) == fast_canonical_string(D2)`` if and only
if ``D1`` and ``D2`` are isomorphic.  The existing suite pins the forward
direction on one-variable expressions and compares the two engines with each
other; this module closes the gaps a release audit found:

1. both directions of the theorem against a brute-force isomorphism oracle that
   shares no code with the canonicaliser, on 1-3 variables, including "hard"
   negative pairs with identical label and degree multisets;
2. invariance under every one of the k! internal numberings with several
   variables;
3. strings stated in the paper (Fig. 1, the README quick start), decoded and
   evaluated against ``math``;
4. a frozen corpus of canonical strings, so that a change made to *both*
   engines at once (which the C++/Python differential tests cannot see) fails;
5. the ``ISALSR_ENGINE`` switch really routes ``fast_canonical_string``.

Each test runs on every available engine.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections.abc import Iterator
from pathlib import Path

import pytest

from isalsr.core import canonical
from isalsr.core.backends import _CPP_AVAILABLE
from isalsr.core.canonical import fast_canonical_string
from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import BINARY_OPS, UNARY_OPS, NodeType
from isalsr.core.permutations import permute_internal_nodes
from isalsr.core.string_to_dag import StringToDAG

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_canonical_strings.json"

ENGINES = ["python", "cpp"] if _CPP_AVAILABLE else ["python"]

# Few labels make equal label multisets common, which is what produces the
# non-isomorphic pairs that a weak invariant would merge.
HARD_OPS = (NodeType.ADD, NodeType.MUL, NodeType.SIN, NodeType.SUB)


# ---------------------------------------------------------------------------
# Independent oracle
# ---------------------------------------------------------------------------


def _node_key(g: LabeledDAG, v: int) -> tuple[str, int | None, int, int]:
    var_index = g.node_data(v).get("var_index")
    return (g.node_label(v).name, var_index, g.in_degree(v), g.out_degree(v))  # type: ignore[return-value]


def _bijections(a: LabeledDAG, b: LabeledDAG) -> Iterator[dict[int, int]]:
    """Yield every bijection a -> b that preserves labels, variable indices and edges."""
    n = a.node_count
    if n != b.node_count or a.edge_count != b.edge_count:
        return
    a_edges = {(u, w) for u in range(n) for w in a.out_neighbors(u)}
    b_edges = {(u, w) for u in range(n) for w in b.out_neighbors(u)}
    a_cls: dict[tuple[str, int | None, int, int], list[int]] = {}
    b_cls: dict[tuple[str, int | None, int, int], list[int]] = {}
    for v in range(n):
        a_cls.setdefault(_node_key(a, v), []).append(v)
        b_cls.setdefault(_node_key(b, v), []).append(v)
    if {k: len(v) for k, v in a_cls.items()} != {k: len(v) for k, v in b_cls.items()}:
        return
    keys = sorted(a_cls, key=str)
    for combo in itertools.product(*(itertools.permutations(b_cls[k]) for k in keys)):
        sigma = {
            src: dst
            for k, perm in zip(keys, combo, strict=True)
            for src, dst in zip(a_cls[k], perm, strict=True)
        }
        if {(sigma[u], sigma[w]) for (u, w) in a_edges} == b_edges:
            yield sigma


def _isomorphic_reference(a: LabeledDAG, b: LabeledDAG) -> bool:
    """Brute-force labeled-DAG isomorphism, operand order included.

    A binary node's first operand (``ordered_inputs(v)[0]``) is what the
    alphabet encodes (Critical Invariant 8, T18), so a bijection must also map
    first operands onto first operands.  Exponential: small DAGs only.
    """
    for sigma in _bijections(a, b):
        if all(
            sigma[a.ordered_inputs(v)[0]] == b.ordered_inputs(sigma[v])[0]
            for v in range(a.node_count)
            if a.node_label(v) in BINARY_OPS and a.ordered_inputs(v)
        ):
            return True
    return False


def _random_dag(rng: random.Random, k: int, m: int, ops: tuple[NodeType, ...]) -> LabeledDAG:
    """Random expression DAG with every node reachable from a variable.

    Each internal node draws its operands from earlier nodes, so the graph is
    acyclic by construction and satisfies the reachability condition of the
    theorem.  Binary nodes receive at most two operands, in a random order (one
    when a single earlier node exists).
    """
    dag = LabeledDAG(m + k)
    for i in range(m):
        dag.add_node(NodeType.VAR, var_index=i)
    for _ in range(k):
        label = rng.choice(ops)
        v = dag.add_node(label)
        arity = 1 if label in UNARY_OPS else 2
        sources = rng.sample(range(v), min(arity, v))
        rng.shuffle(sources)
        for s in sources:
            dag.add_edge(s, v)
    return dag


def _s2d_dag(rng: random.Random, m: int, max_internal: int) -> LabeledDAG:
    """A DAG decoded by S2D from a random string, with 2..max_internal internal nodes.

    These carry what the hand-built generator does not: CONST leaves, POW and
    DIV, C/c edges into variables and between arbitrary nodes, and binary nodes
    with surplus in-edges -- the shapes S2D and the host adapters produce.
    """
    single, labels = "NPnpCcW", "+*-/scelrakgi^"
    while True:
        parts = [
            rng.choice(single) if rng.random() < 0.5 else rng.choice("Vv") + rng.choice(labels)
            for _ in range(rng.randint(3, 14))
        ]
        dag = StringToDAG("".join(parts), num_variables=m).run()
        if 2 <= dag.node_count - m <= max_internal:
            return dag


def _rebuild(dag: LabeledDAG, inputs: list[list[int]]) -> LabeledDAG | None:
    """Copy ``dag``'s nodes and wire node v from ``inputs[v]``, in that operand order.

    Returns None when the rewired edge set has a cycle (``add_edge`` refuses an
    edge only if it closes one, and any subset of an acyclic set is acyclic).
    """
    out = LabeledDAG(dag.node_count)
    for v in range(dag.node_count):
        out.add_node(dag.node_label(v), var_index=dag.node_data(v).get("var_index"))  # type: ignore[arg-type]
    for v, sources in enumerate(inputs):
        for s in sources:
            if not out.add_edge(s, v):
                return None
    return out


def _near_miss(rng: random.Random, dag: LabeledDAG) -> LabeledDAG | None:
    """A DAG with the same label, variable and degree multisets, possibly not isomorphic.

    Either a double edge swap (a->b, c->d  =>  a->d, c->b), which keeps each
    node's in- and out-degree, or the reversal of a binary node's two operands,
    which keeps the edge set and changes only the first operand the alphabet
    encodes.  Rewirings that close a cycle are rejected.  Every non-VAR node
    keeps an in-edge and the graph stays acyclic, so every node still traces
    back to a variable.  The oracle decides whether the result is isomorphic
    to ``dag``; these are the pairs a weak invariant would merge.
    """
    inputs = [list(dag.ordered_inputs(v)) for v in range(dag.node_count)]
    binary = [
        v
        for v in range(dag.node_count)
        if dag.node_label(v) in BINARY_OPS and len(set(inputs[v])) == 2
    ]
    if binary and rng.random() < 0.3:
        v = rng.choice(binary)
        inputs[v].reverse()
        return _rebuild(dag, inputs)
    edges = [(u, v) for v in range(dag.node_count) for u in inputs[v]]
    rng.shuffle(edges)
    for (a, b), (c, d) in itertools.combinations(edges, 2):
        if len({a, b, c, d}) == 4 and a not in inputs[d] and c not in inputs[b]:
            swapped = [list(x) for x in inputs]
            swapped[b][swapped[b].index(a)] = c
            swapped[d][swapped[d].index(c)] = a
            rewired = _rebuild(dag, swapped)
            if rewired is not None:
                return rewired
    return None


def _permuted_copy(rng: random.Random, dag: LabeledDAG) -> LabeledDAG:
    perm = list(range(dag.node_count - len(dag.var_nodes())))
    rng.shuffle(perm)
    return permute_internal_nodes(dag, perm)


def _theorem_corpus(rng: random.Random, kind: str, m: int) -> list[LabeledDAG]:
    """Base DAGs, two near misses of each, and a renumbered copy of each."""
    if kind == "generated":
        base = [_random_dag(rng, rng.randint(3, 6), m, HARD_OPS) for _ in range(60)]
    else:
        base = [_s2d_dag(rng, m, max_internal=6) for _ in range(60)]
    near_misses = [
        nm for d in base for nm in (_near_miss(rng, d), _near_miss(rng, d)) if nm is not None
    ]
    return base + near_misses + [_permuted_copy(rng, d) for d in base]


# ---------------------------------------------------------------------------
# 1. Theorem 3.12, both directions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("kind", ["generated", "s2d"])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_same_canonical_string_iff_isomorphic(engine: str, kind: str, m: int) -> None:
    """fcs(D1) == fcs(D2)  <=>  D1 ~ D2, decided by the brute-force oracle.

    ``generated``: few labels, so many pairs share label and degree multisets.
    ``s2d``: decoded from random strings, so CONST, POW/DIV, edges into
    variables and surplus binary in-edges occur.
    """
    rng = random.Random(1000 + 10 * m + (kind == "s2d"))
    dags = _theorem_corpus(rng, kind, m)
    strings = [fast_canonical_string(d, backend=engine) for d in dags]

    positives = hard_negatives = 0
    for (a, sa), (b, sb) in itertools.combinations(zip(dags, strings, strict=True), 2):
        if (a.node_count, a.edge_count) != (b.node_count, b.edge_count):
            assert sa != sb
            continue
        iso = _isomorphic_reference(a, b)
        assert (sa == sb) == iso, (
            f"engine={engine}: canonical strings {'agree' if sa == sb else 'differ'} "
            f"but the oracle says {'isomorphic' if iso else 'non-isomorphic'}: {sa!r} vs {sb!r}"
        )
        positives += iso
        same_multiset = sorted(_node_key(a, v) for v in range(a.node_count)) == sorted(
            _node_key(b, v) for v in range(b.node_count)
        )
        hard_negatives += same_multiset and not iso

    # Non-vacuity: the corpus must exercise both directions of the theorem.
    # Measured at these seeds: 82-100 positives, 108-255 hard negatives.
    assert positives >= 40, f"only {positives} isomorphic pairs"
    assert hard_negatives >= 50, f"only {hard_negatives} hard non-isomorphic pairs"


# ---------------------------------------------------------------------------
# 2. Every one of the k! numberings, several variables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(("m", "k", "seed"), [(2, 5, 1), (2, 6, 2), (3, 5, 3), (3, 6, 4)])
def test_all_k_factorial_numberings_collapse(engine: str, m: int, k: int, seed: int) -> None:
    """All k! relabelings of the internal nodes give one canonical string."""
    rng = random.Random(seed)
    ops = tuple(sorted(UNARY_OPS | BINARY_OPS | {NodeType.ADD, NodeType.MUL}, key=lambda t: t.name))
    dag = _random_dag(rng, k, m, ops)
    expected = fast_canonical_string(dag, backend=engine)
    seen = {
        fast_canonical_string(permute_internal_nodes(dag, list(p)), backend=engine)
        for p in itertools.permutations(range(k))
    }
    assert seen == {expected}, f"{len(seen)} distinct strings over {math.factorial(k)} numberings"


# ---------------------------------------------------------------------------
# 3. Strings stated in the paper and the README
# ---------------------------------------------------------------------------


PAPER_STRINGS = [
    # (string, m, expression it encodes) -- Fig. 1 of the paper; README quick start.
    ("VcVspv*pv+PpcnnC", 2, lambda x: math.sin(x[0]) * x[1] + math.cos(x[0])),
    ("VcVspv+Ppc", 1, lambda x: math.sin(x[0]) + math.cos(x[0])),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("string", "m", "expr"), PAPER_STRINGS, ids=[s for s, _, _ in PAPER_STRINGS]
)
def test_paper_strings_are_canonical_and_evaluate(
    engine: str, string: str, m: int, expr: object
) -> None:
    """The strings the paper calls canonical are fixed points, and mean what it says."""
    dag = StringToDAG(string, num_variables=m).run()
    assert fast_canonical_string(dag, backend=engine) == string
    for x in ([0.3, 1.7, -0.4][:m], [-1.2, 0.5, 2.0][:m]):
        got = evaluate_dag(dag, dict(enumerate(x)))
        assert math.isclose(got, expr(x), rel_tol=1e-12)  # type: ignore[operator]


def test_quick_start_numberings_collapse() -> None:
    """The README's two numberings of sin(x1) + cos(x1) share one canonical string."""
    a = StringToDAG("VsVcpv+Ppc", num_variables=1).run()
    b = StringToDAG("VcVspv+Ppc", num_variables=1).run()
    assert _isomorphic_reference(a, b)
    assert {fast_canonical_string(a, backend=e) for e in ENGINES} | {
        fast_canonical_string(b, backend=e) for e in ENGINES
    } == {"VcVspv+Ppc"}


# ---------------------------------------------------------------------------
# 4. Frozen corpus
# ---------------------------------------------------------------------------


def _golden_inputs() -> list[tuple[str, int]]:
    """The (string, m) inputs of the frozen corpus; their canonical strings are in GOLDEN_PATH."""
    rng = random.Random(20261101)
    single = "NPnpCcW"
    labels = "+*-/scelrakgi^"
    inputs = []
    for i in range(300):
        parts = [
            rng.choice(single) if rng.random() < 0.45 else rng.choice("Vv") + rng.choice(labels)
            for _ in range(rng.randint(4, 18))
        ]
        inputs.append(("".join(parts), 1 + i % 3))
    return inputs


def _write_golden() -> None:
    """Regenerate GOLDEN_PATH. Run only for a deliberate, documented change of canonical form."""
    rows = [
        {
            "input": s,
            "m": m,
            "canonical": fast_canonical_string(
                StringToDAG(s, num_variables=m).run(), backend="python"
            ),
        }
        for s, m in _golden_inputs()
    ]
    GOLDEN_PATH.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


@pytest.mark.parametrize("engine", ENGINES)
def test_canonical_strings_match_frozen_corpus(engine: str) -> None:
    """300 canonical strings (wl_only), frozen at release 1.0.0; both engines must reproduce them.

    Generated by the Python engine with ``_write_golden``.  The inputs are stored
    as strings, so the corpus does not depend on the generator above.  It
    catches a change of canonical form made to both engines at once, which the
    C++/Python differential tests cannot see.
    """
    rows = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert len(rows) == 300
    assert len({r["canonical"] for r in rows}) >= 250, "corpus too repetitive to be informative"
    wrong = [
        r
        for r in rows
        if fast_canonical_string(
            StringToDAG(r["input"], num_variables=r["m"]).run(), backend=engine
        )
        != r["canonical"]
    ]
    assert not wrong, (
        f"engine={engine}: {len(wrong)}/300 canonical strings changed, e.g. {wrong[:2]}"
    )


# ---------------------------------------------------------------------------
# 5. ISALSR_ENGINE routes fast_canonical_string
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CPP_AVAILABLE, reason="needs the C++ engine to tell the two apart")
def test_isalsr_engine_selects_the_engine_that_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ISALSR_ENGINE`` must change which engine computes the string, not only engine()."""
    native_calls = []
    real = canonical._cpp_ext.fast_canonical_string  # type: ignore[union-attr]

    def spy(*args: object) -> str:
        native_calls.append(args)
        return real(*args)  # type: ignore[no-any-return]

    monkeypatch.setattr(canonical._cpp_ext, "fast_canonical_string", spy)
    dag = StringToDAG("VcVspv*pv+PpcnnC", num_variables=2).run()

    monkeypatch.setenv("ISALSR_ENGINE", "cpp")
    via_cpp = fast_canonical_string(dag)
    assert len(native_calls) == 1, "positive control: the spy must see the C++ call"

    monkeypatch.setenv("ISALSR_ENGINE", "python")
    via_python = fast_canonical_string(dag)
    assert len(native_calls) == 1, "ISALSR_ENGINE=python still dispatched to C++"
    assert via_python == via_cpp
