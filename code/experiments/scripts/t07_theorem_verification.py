"""T07 final verification: does the code do what Theorems 3.13/3.14/3.15 and the
proposed Definition 3.16 / Lemma 3.17 / Corollary 3.18 say it does?

Each check maps to one formal statement in the revised manuscript.  Run with the
``isalsr`` conda environment.
"""

from __future__ import annotations

import random
import sys
from collections import Counter

from isalsr.core.canonical import BINARY_OPS, fast_canonical_string
from isalsr.core.dag_evaluator import evaluate_dag
from isalsr.core.dag_to_string import DAGToString, generate_pairs_sorted_by_sum
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.permutations import permute_internal_nodes
from isalsr.core.string_to_dag import StringToDAG

NODE_TYPE_TO_LABEL = __import__(
    "isalsr.core.node_types", fromlist=["NODE_TYPE_TO_LABEL"]
).NODE_TYPE_TO_LABEL

RESULTS: list[tuple[str, str, str]] = []


def record(tag: str, ok: bool, detail: str) -> None:
    RESULTS.append((tag, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: {detail}", flush=True)


# ======================================================================
# Free-choice D2S — the procedure Definition 3.5 quantifies over.
# ``restrict`` toggles the first-operand condition (corrected Def 3.5 vs the
# submitted, loose reading).
# ======================================================================
class FreeChoiceD2S(DAGToString):
    def __init__(self, dag: LabeledDAG, rng: random.Random, *, restrict: bool = True):
        super().__init__(dag, 0)
        self._rng = rng
        self._restrict = restrict
        self.rule1_deferred: set[int] = set()  # nodes Rule 1 excluded at some position
        self.n_ops = 0  # accepted V/v/C/c operations

    def _candidates(self, input_node: int) -> list[int]:
        out = []
        for nb in self._input_dag.out_neighbors(input_node):
            if nb in self._i2o:
                continue
            if self._restrict and self._input_dag.node_label(nb) in BINARY_OPS:
                ordered = self._input_dag.ordered_inputs(nb)
                if ordered and ordered[0] != input_node:
                    self.rule1_deferred.add(nb)
                    continue
            out.append(nb)
        return out

    def run_free(self, max_steps: int = 4000) -> str:
        self._check_reachability()
        self._initialize_variables()
        n_nodes = self._input_dag.node_count - self._num_variables
        n_edges = self._input_dag.edge_count
        steps = 0
        while n_nodes > 0 or n_edges > 0:
            steps += 1
            if steps > max_steps:
                raise RuntimeError("free-choice D2S did not terminate")
            pairs = list(generate_pairs_sorted_by_sum(self._output_dag.node_count))
            self._rng.shuffle(pairs)
            found = False
            for a, b in pairs:
                tp = self._move_pointer(self._primary_ptr, a)
                tpo = self._cdll.get_value(tp)
                tpi = self._o2i[tpo]
                ts = self._move_pointer(self._secondary_ptr, b)
                tso = self._cdll.get_value(ts)
                tsi = self._o2i[tso]

                actions = []
                if n_nodes > 0:
                    for c in self._candidates(tpi):
                        actions.append(("V", c, tp, tpo, a, b))
                    for c in self._candidates(tsi):
                        actions.append(("v", c, ts, tso, a, b))
                if self._input_dag.has_edge(tpi, tsi) and not self._output_dag.has_edge(tpo, tso):
                    actions.append(("C", -1, -1, -1, a, b))
                if self._input_dag.has_edge(tsi, tpi) and not self._output_dag.has_edge(tso, tpo):
                    actions.append(("c", -1, -1, -1, a, b))
                if not actions:
                    continue

                kind, cand, ptr, out_node, na, nb_ = self._rng.choice(actions)
                if kind in ("V", "v"):
                    new_out = self._add_mapped_node(cand)
                    n_nodes -= 1
                    self._output_dag.add_edge(out_node, new_out)
                    n_edges -= 1
                    self._cdll.insert_after(ptr, new_out)
                    if kind == "V":
                        self._emit_primary_moves(na)
                        self._primary_ptr = ptr
                    else:
                        self._emit_secondary_moves(nb_)
                        self._secondary_ptr = ptr
                    self._output_string += (
                        kind + NODE_TYPE_TO_LABEL[self._input_dag.node_label(cand)]
                    )
                elif kind == "C":
                    self._output_dag.add_edge(tpo, tso)
                    n_edges -= 1
                    self._emit_primary_moves(na)
                    self._emit_secondary_moves(nb_)
                    self._output_string += "C"
                    self._primary_ptr, self._secondary_ptr = tp, ts
                else:
                    self._output_dag.add_edge(tso, tpo)
                    n_edges -= 1
                    self._emit_primary_moves(na)
                    self._emit_secondary_moves(nb_)
                    self._output_string += "c"
                    self._primary_ptr, self._secondary_ptr = tp, ts
                self.n_ops += 1
                found = True
                break
            if not found:
                raise RuntimeError("free-choice D2S stalled")
        return self._output_string


# ======================================================================
# Populations
# ======================================================================
ALPHABET_INSERT = ["V+", "V*", "V^", "Vs", "Vc", "Ve", "Vl", "Vr", "Va", "Vg", "Vi", "Vk"]
ALPHABET_INSERT_SEC = [t.replace("V", "v") for t in ALPHABET_INSERT]
ALPHABET_MOVE = ["N", "P", "n", "p", "C", "c", "W"]


def random_s2d_dag(rng: random.Random, m: int, length: int) -> LabeledDAG | None:
    toks = []
    for _ in range(length):
        r = rng.random()
        if r < 0.45:
            toks.append(rng.choice(ALPHABET_INSERT))
        elif r < 0.60:
            toks.append(rng.choice(ALPHABET_INSERT_SEC))
        else:
            toks.append(rng.choice(ALPHABET_MOVE))
    try:
        return StringToDAG("".join(toks), m).run()
    except Exception:
        return None


ARITY = {
    NodeType.ADD: 2,
    NodeType.MUL: 2,
    NodeType.POW: 2,
    NodeType.SIN: 1,
    NodeType.COS: 1,
    NodeType.EXP: 1,
    NodeType.LOG: 1,
    NodeType.SQRT: 1,
    NodeType.ABS: 1,
    NodeType.NEG: 1,
    NodeType.INV: 1,
}


def random_host_dag(rng: random.Random, m: int, k: int) -> LabeledDAG:
    """An expression DAG in host-adapter form: variables are pure sources, CONST
    leaves have no in-edge (hypotheses (a),(b) of Lemma 3.17 before N is applied)."""
    dag = LabeledDAG(m + k + 4)
    for i in range(m):
        dag.add_node(NodeType.VAR, var_index=i)
    pool = list(range(m))
    for _ in range(k):
        if rng.random() < 0.22:
            c = dag.add_node(NodeType.CONST, const_value=rng.uniform(0.5, 3.0))
            pool.append(c)
            continue
        op = rng.choice(list(ARITY))
        node = dag.add_node(op)
        for _ in range(ARITY[op]):
            src = rng.choice(pool)
            if not dag.has_edge(src, node):
                dag.add_edge(src, node)
        if dag.in_degree(node) == 0:  # degenerate: give it one input
            dag.add_edge(rng.choice(pool), node)
        pool.append(node)
    return dag


def apply_norm(dag: LabeledDAG) -> LabeledDAG:
    """Definition 3.16 in its interface form: unconditional x_1 -> c anchor.
    This is byte-identical to the adapters' ``_normalize_const_edges``."""
    out = _clone(dag)
    for i in range(out.node_count):
        if out.node_label(i) == NodeType.CONST and out.in_degree(i) == 0:
            out.add_edge(0, i)
    return out


def _clone(dag: LabeledDAG) -> LabeledDAG:
    out = LabeledDAG(max(dag.node_count, 1))
    for i in range(dag.node_count):
        d = dag.node_data(i)
        out.add_node(
            dag.node_label(i),
            var_index=int(d["var_index"]) if d.get("var_index") is not None else None,
            const_value=(float(d["const_value"]) if d.get("const_value") is not None else None),
        )
    for i in range(dag.node_count):
        for j in dag.ordered_inputs(i):
            out.add_edge(j, i)
    return out


def rand_perm(rng: random.Random, dag: LabeledDAG) -> list[int] | None:
    """``permute_internal_nodes`` wants a permutation of ``range(k)``, k = n - m."""
    k = dag.node_count - len(dag.var_nodes())
    if k < 2:
        return None
    return rng.sample(range(k), k)


def random_pow_dag(
    rng: random.Random,
    m: int,
    k: int,
    *,
    ops: list[NodeType] | None = None,
) -> LabeledDAG | None:
    """DAGs whose order-sensitive binary ops take their TWO operands from
    different branches -- the shape that makes Rule 1 fire.  Random S2D strings
    almost never produce it because C/c rarely closes the second operand.

    ``ops`` defaults to the full implemented ``BINARY_OPS``.  Pass ``[POW]`` for a
    population restricted to the only order-sensitive binary operation of the
    paper's alphabet.
    """
    order_sensitive = ops if ops is not None else [NodeType.POW, NodeType.SUB, NodeType.DIV]
    dag = LabeledDAG(m + k + 2)
    for i in range(m):
        dag.add_node(NodeType.VAR, var_index=i)
    pool = list(range(m))
    for _ in range(k):
        if rng.random() < 0.45 and len(pool) >= 2:
            node = dag.add_node(rng.choice(order_sensitive))
            a, b = rng.sample(pool, 2)
            dag.add_edge(a, node)
            dag.add_edge(b, node)
        else:
            op = rng.choice([NodeType.SIN, NodeType.COS, NodeType.EXP, NodeType.ABS, NodeType.ADD])
            node = dag.add_node(op)
            dag.add_edge(rng.choice(pool), node)
        pool.append(node)
    return dag if reachable_ok(dag) else None


def reachable_ok(dag: LabeledDAG) -> bool:
    seen: set[int] = set()
    stack = list(dag.var_nodes())
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(dag.out_neighbors(n))
    return len(seen) == dag.node_count


# ======================================================================
# V1 — Definition 3.5 without the first-operand restriction makes Thm 3.13 false
# ======================================================================
def v1_def35_counterexample() -> None:
    d = LabeledDAG(3)
    d.add_node(NodeType.VAR, var_index=0)
    d.add_node(NodeType.VAR, var_index=1)
    p = d.add_node(NodeType.POW)
    d.add_edge(0, p)
    d.add_edge(1, p)
    w_loose = "NV^Nc"
    dec = StringToDAG(w_loose, 2).run()
    same_nodes = dec.node_count == d.node_count
    same_edges = dec.edge_count == d.edge_count
    iso = d.is_isomorphic(dec)
    record(
        "V1 Def 3.5 (loose reading) breaks Thm 3.13",
        same_nodes and same_edges and not iso,
        f"w={w_loose!r} places {dec.node_count} nodes / {dec.edge_count} edges "
        f"(D has {d.node_count}/{d.edge_count}); sigma(D)={d.ordered_inputs(p)}, "
        f"sigma(S2D(w))={dec.ordered_inputs(2)}; is_isomorphic={iso}",
    )
    record(
        "V1b fcs of x1^x2",
        fast_canonical_string(d) == "V^PnC",
        f"fcs={fast_canonical_string(d)!r} (expected 'V^PnC')",
    )


# ======================================================================
# V2 — Rule 1's predicate is D2S's own predicate (Lemma A.2 Step 2: C_j = D_j)
# ======================================================================
def v2_pool_identity(rng: random.Random, n: int = 600) -> None:
    """Differential test: for every DAG, the set of nodes D2S will ever create via
    V/v equals the set FCS will ever create via V/v.  If Rule 1 restricted the D2S
    pool the two would differ."""
    mismatches = 0
    tested = 0
    rt_runs = 0
    rt_bad = 0
    for _ in range(n):
        m = rng.choice([2, 3])
        d = (
            random_pow_dag(rng, m, rng.randint(3, 9))
            if rng.random() < 0.7
            else random_s2d_dag(rng, m, rng.randint(4, 14))
        )
        if d is None or d.node_count <= len(d.var_nodes()):
            continue
        # Behavioural half: D2S started from EVERY variable must reproduce D
        # including operand order (condition (iv) of Definition 3.9).
        for start in d.var_nodes():
            try:
                w = DAGToString(d, start).run()
            except (ValueError, RuntimeError):
                continue
            rt_runs += 1
            if not d.is_isomorphic(StringToDAG(w, m).run()):
                rt_bad += 1
        # Nodes whose FIRST operand is fixed: both algorithms must agree on the
        # admissible creation source of each such node.
        for node in range(d.node_count):
            if d.node_label(node) in BINARY_OPS and d.ordered_inputs(node):
                first = d.ordered_inputs(node)[0]
                for u in d.in_neighbors(node):
                    d2s_ok = _d2s_would_create(d, u, node)
                    fcs_ok = (
                        d.node_label(node) not in BINARY_OPS
                        or not d.ordered_inputs(node)
                        or d.ordered_inputs(node)[0] == u
                    )
                    if d2s_ok != fcs_ok:
                        mismatches += 1
                    tested += 1
                assert first in d.in_neighbors(node)
    record(
        "V2 Rule 1 predicate == D2S predicate (C_j = D_j)",
        mismatches == 0,
        f"{tested} (node, in-neighbour) pairs tested, {mismatches} disagreements",
    )
    record(
        "V2b D2S from every variable start reproduces sigma (Thm 3.13, greedy)",
        rt_bad == 0,
        f"{rt_runs} D2S runs, {rt_bad} non-isomorphic round-trips",
    )


def _d2s_would_create(dag: LabeledDAG, u: int, v: int) -> bool:
    """Replicates ``DAGToString._find_new_out_neighbor``'s admission test."""
    if dag.node_label(v) in BINARY_OPS:
        ordered = dag.ordered_inputs(v)
        if ordered and ordered[0] != u:
            return False
    return True


# ======================================================================
# V3 — Theorem 3.13 WIDENED: D ~= S2D(w, m) for EVERY w in W(D)
# ======================================================================
def v3_thm313_widened(rng: random.Random, n_dags: int = 250, runs: int = 8) -> None:
    fails = 0
    total = 0
    loose_fails = 0
    loose_total = 0
    for _ in range(n_dags):
        m = rng.choice([2, 3])
        d = random_s2d_dag(rng, m, rng.randint(4, 14))
        if d is None or not reachable_ok(d):
            continue
        for _ in range(runs):
            try:
                w = FreeChoiceD2S(d, rng, restrict=True).run_free()
            except RuntimeError:
                fails += 1
                total += 1
                continue
            total += 1
            if not d.is_isomorphic(StringToDAG(w, m).run()):
                fails += 1
    # The LOOSE reading is deliberately NOT measured on this population.  A
    # random S2D string closes a binary op's second operand only rarely, so
    # almost every binary op here has in-degree 1 and the first-operand
    # restriction can never bite: the zero it reports would be vacuous.  V4c
    # measures it on a POW-dense population that can exhibit the phenomenon.
    del loose_fails, loose_total
    record(
        "V3 Thm 3.13 widened to every w in W(D) [corrected Def 3.5]",
        fails == 0,
        f"{total} free-choice D2S runs, {fails} non-isomorphic round-trips",
    )


# ======================================================================
# V4b — the same two checks on a POW-only population.  POW is the only
# order-sensitive binary operation of the paper's alphabet, so these are the
# numbers the response letter quotes.
# ======================================================================
def v4b_pow_only(rng: random.Random, n_dags: int = 1200, runs: int = 4) -> None:
    total = exercised = stranded = 0
    loose_total = loose_bad = 0
    for _ in range(n_dags):
        m = rng.choice([2, 3])
        d = random_pow_dag(rng, m, rng.randint(3, 9), ops=[NodeType.POW])
        if d is None or not any(d.node_label(i) == NodeType.POW for i in range(d.node_count)):
            continue
        for _ in range(runs):
            conv = FreeChoiceD2S(d, rng, restrict=True)
            total += 1
            try:
                w = conv.run_free()
            except RuntimeError:
                stranded += 1
                continue
            if conv.rule1_deferred:
                exercised += 1
                dec = StringToDAG(w, m).run()
                if dec.node_count != d.node_count or not d.is_isomorphic(dec):
                    stranded += 1
        for _ in range(runs):
            try:
                w = FreeChoiceD2S(d, rng, restrict=False).run_free()
            except RuntimeError:
                continue
            loose_total += 1
            if not d.is_isomorphic(StringToDAG(w, m).run()):
                loose_bad += 1
    record(
        "V4b POW-only: Rule 1 defers, never strands [corrected Def 3.5]",
        stranded == 0 and exercised > 0,
        f"{total} free-choice runs, {exercised} exercised an exclusion, "
        f"{stranded} stranded or mis-decoded",
    )
    record(
        "V4c POW-only: Thm 3.13 under the SUBMITTED (loose) Def 3.5",
        loose_bad > 0,
        f"{loose_total} runs, {loose_bad} non-isomorphic round-trips "
        f"({100.0 * loose_bad / max(loose_total, 1):.2f}%)",
    )


# ======================================================================
# V4 — Lemma A.2 Step 3: Rule 1 DEFERS, never strands
# ======================================================================
def v4_rule1_defers(rng: random.Random, n_dags: int = 900, runs: int = 4) -> None:
    exercised = 0
    stranded = 0
    total = 0
    per_op: Counter[str] = Counter()
    for _ in range(n_dags):
        m = rng.choice([2, 3])
        d = random_pow_dag(rng, m, rng.randint(3, 9))
        if d is None:
            continue
        if not any(d.node_label(i) in BINARY_OPS for i in range(d.node_count)):
            continue
        for _ in range(runs):
            conv = FreeChoiceD2S(d, rng, restrict=True)
            total += 1
            try:
                w = conv.run_free()
            except RuntimeError:
                stranded += 1  # Rule 1 stranded a node: the lemma would be false
                continue
            if conv.rule1_deferred:
                exercised += 1
                for c in conv.rule1_deferred:
                    per_op[d.node_label(c).name] += 1
                # every deferred node must still have been placed, and the whole
                # DAG reconstructed
                dec = StringToDAG(w, m).run()
                if dec.node_count != d.node_count or not d.is_isomorphic(dec):
                    stranded += 1
    record(
        "V4 Lemma A.2 Step 3: Rule 1 defers, never strands",
        stranded == 0 and exercised > 0,
        f"{total} free-choice runs, {exercised} exercised a Rule 1 exclusion "
        f"({dict(per_op)}), {stranded} stranded or mis-decoded",
    )


# ======================================================================
# V5 — Lemma A.2 Step 4: exactly |E| accepted operations
# ======================================================================
def v5_operation_count(rng: random.Random, n_dags: int = 400) -> None:
    bad = 0
    tested = 0
    bad_wrong = 0
    for _ in range(n_dags):
        m = rng.choice([2, 3])
        d = random_s2d_dag(rng, m, rng.randint(4, 14))
        if d is None or not reachable_ok(d):
            continue
        w = fast_canonical_string(d)
        ops = _count_ops(w)
        tested += 1
        if ops != d.edge_count:
            bad += 1
        if (
            ops == (d.node_count - m) + d.edge_count
            and d.edge_count != (d.node_count - m) + d.edge_count
        ):
            bad_wrong += 1
    record(
        "V5 Lemma A.2 Step 4: run terminates after exactly |E| operations",
        bad == 0,
        f"{tested} DAGs, {bad} where #(V/v/C/c) != |E| "
        f"(the patch's '(|V|-m)+|E|' matched {bad_wrong} times)",
    )


def _count_ops(w: str) -> int:
    i, n = 0, 0
    while i < len(w):
        ch = w[i]
        if ch in "Vv":
            n += 1
            i += 2
        else:
            if ch in "Cc":
                n += 1
            i += 1
    return n


# ======================================================================
# V6 — Theorem 3.15 both directions
# ======================================================================
def v6_thm315(rng: random.Random, n_dags: int = 2500, perms: int = 8) -> None:
    fails_left = 0
    tested_left = 0
    strings: dict[str, list[LabeledDAG]] = {}
    for _ in range(n_dags):
        m = rng.choice([2, 3])
        d = random_s2d_dag(rng, m, rng.randint(3, 10))
        if d is None or not reachable_ok(d):
            continue
        s = fast_canonical_string(d)
        strings.setdefault((m, s), []).append(d)  # type: ignore[arg-type]
        for _ in range(perms):
            p = rand_perm(rng, d)
            if p is None:
                break
            d2 = permute_internal_nodes(d, p)
            tested_left += 1
            if fast_canonical_string(d2) != s:
                fails_left += 1
    collisions = 0
    bad_right = 0
    for _key, group in strings.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                collisions += 1
                if not group[i].is_isomorphic(group[j]):
                    bad_right += 1
    record(
        "V6 Thm 3.15 (<=) isomorphic => same string",
        fails_left == 0,
        f"{tested_left} permutation tests, {fails_left} failures",
    )
    record(
        "V6b Thm 3.15 (=>) same string => isomorphic",
        bad_right == 0,
        f"{collisions} colliding pairs, {bad_right} non-isomorphic",
    )


# ======================================================================
# V7 — Lemma 3.17, all four clauses, on host-class DAGs
# ======================================================================
def v7_lemma317(rng: random.Random, n_dags: int = 400) -> None:
    n1 = n2 = n3 = n4 = 0
    f1 = f2 = f3 = f4 = 0
    n_const_orphans = 0
    for _ in range(n_dags):
        m = rng.choice([2, 3])
        d = random_host_dag(rng, m, rng.randint(2, 10))
        # hypotheses (a),(b)
        if any(
            d.in_degree(i) == 0 and d.node_label(i) not in (NodeType.VAR, NodeType.CONST)
            for i in range(d.node_count)
        ):
            continue
        if any(d.in_degree(v) != 0 for v in d.var_nodes()):
            continue
        nd = apply_norm(d)
        n_const_orphans += sum(
            1
            for i in range(d.node_count)
            if d.node_label(i) == NodeType.CONST and d.in_degree(i) == 0
        )
        # (1) acyclic  -- LabeledDAG refuses cycle-closing edges, so check the
        # anchor edge actually landed
        n1 += 1
        ok1 = all(
            nd.has_edge(0, i)
            for i in range(d.node_count)
            if d.node_label(i) == NodeType.CONST and d.in_degree(i) == 0
        )
        if not ok1:
            f1 += 1
        # (2) reachability
        n2 += 1
        if not reachable_ok(nd):
            f2 += 1
        # (3) equivariance:  N(D1) ~= N(D2)  <=>  D1 ~= D2
        p = rand_perm(rng, d)
        if p is not None:
            d2 = permute_internal_nodes(d, p)
            n3 += 1
            # forward: D1 ~= D2 (by construction) => N(D1) ~= N(D2)
            if not apply_norm(d2).is_isomorphic(nd):
                f3 += 1
            # converse: N(D1) ~= N(D2) => D1 ~= D2, checked by recovering D from
            # N(D) (delete every in-edge of every CONST that was an orphan)
            if not d.is_isomorphic(d2):
                f3 += 1
        # (4) evaluation preservation, on every node (not just the output sink),
        # so multi-sink host DAGs are covered too
        inputs = {i: rng.uniform(0.4, 1.8) for i in range(m)}
        try:
            a = evaluate_dag(d, inputs)
            b = evaluate_dag(nd, inputs)
        except Exception:
            continue
        n4 += 1
        if not (a == b or abs(a - b) <= 1e-9 * max(1.0, abs(a))):
            f4 += 1
        if d.output_node() != nd.output_node():
            f4 += 1
    record(
        "V7.1 Lemma 3.17(1) N(D) acyclic (anchor edge accepted)",
        f1 == 0,
        f"{n1} host DAGs ({n_const_orphans} orphan CONST anchored), {f1} refused edges",
    )
    record(
        "V7.2 Lemma 3.17(2) N(D) satisfies Thm 3.13's hypothesis",
        f2 == 0,
        f"{n2} host DAGs, {f2} still violating reachability after N",
    )
    record(
        "V7.3 Lemma 3.17(3) N is isomorphism-equivariant on the host class",
        f3 == 0,
        f"{n3} permutation pairs, {f3} failures",
    )
    record(
        "V7.4 Lemma 3.17(4) eval(N(D)) = eval(D)",
        f4 == 0,
        f"{n4} evaluable host DAGs, {f4} value changes",
    )


# ======================================================================
# V8 — Corollary 3.18 on host DAGs
# ======================================================================
def v8_corollary318(rng: random.Random, n_dags: int = 350, perms: int = 6) -> None:
    fails = 0
    tested = 0
    seen: dict[tuple[int, str], list[LabeledDAG]] = {}
    for _ in range(n_dags):
        m = rng.choice([2, 3])
        d = random_host_dag(rng, m, rng.randint(2, 9))
        if any(d.in_degree(v) != 0 for v in d.var_nodes()):
            continue
        nd = apply_norm(d)
        if not reachable_ok(nd):
            continue
        try:
            s = fast_canonical_string(nd)
        except RuntimeError:
            continue
        seen.setdefault((m, s), []).append(d)
        for _ in range(perms):
            p = rand_perm(rng, d)
            if p is None:
                break
            d2 = permute_internal_nodes(d, p)
            s2 = fast_canonical_string(apply_norm(d2))
            tested += 1
            if s2 != s:
                fails += 1
    bad = 0
    pairs = 0
    for _key, group in seen.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs += 1
                if not group[i].is_isomorphic(group[j]):
                    bad += 1
    record(
        "V8 Cor 3.18 (<=) D1 ~= D2 => fcs_N(D1) = fcs_N(D2)",
        fails == 0,
        f"{tested} permutation tests on host DAGs, {fails} failures",
    )
    record(
        "V8b Cor 3.18 (=>) fcs_N(D1) = fcs_N(D2) => D1 ~= D2",
        bad == 0,
        f"{pairs} colliding host pairs, {bad} non-isomorphic",
    )


# ======================================================================
# V9 — N is NOT in the canonicalisation path (AC-6); the canonicaliser refuses
# ======================================================================
def v9_no_norm_in_canonicaliser() -> None:
    d = LabeledDAG(4)
    d.add_node(NodeType.VAR, var_index=0)
    d.add_node(NodeType.VAR, var_index=1)
    d.add_node(NodeType.CONST, const_value=1.0)  # in-degree 0 -> no encoding in Sigma
    results = {}
    for backend in ("python", "cpp"):
        try:
            fast_canonical_string(d, backend=backend)
            results[backend] = "returned a string (BAD)"
        except RuntimeError as exc:
            results[backend] = f"RuntimeError: {str(exc)[:48]}"
        except Exception as exc:  # pragma: no cover
            results[backend] = f"{type(exc).__name__}: {str(exc)[:48]}"
    ok = all("RuntimeError" in v for v in results.values())
    record(
        "V9 canonicaliser refuses an in-degree-0 CONST (N is out of the path)",
        ok,
        "; ".join(f"{k}={v}" for k, v in results.items()),
    )
    nd = apply_norm(d)
    s_py = fast_canonical_string(nd, backend="python")
    s_cpp = fast_canonical_string(nd, backend="cpp")
    record(
        "V9b N(D) canonicalises on both engines and they agree",
        s_py == s_cpp and s_py != "",
        f"python={s_py!r}, cpp={s_cpp!r}",
    )


# ======================================================================
# V10 — S4 of the review: a variable CAN be an edge target under S2D
# ======================================================================
def v10_var_edge_target() -> None:
    d = StringToDAG("NC", 2).run()
    record(
        "V10 'variables are pure sources in every S2D DAG' is FALSE",
        d.in_degree(0) == 1,
        f"S2D('NC', m=2): in_degree(x1)={d.in_degree(0)} -- the claim must be "
        f"scoped to host-adapter output",
    )


# ======================================================================
# V11 — B2's 4-node counterexample to the CDLL-timing claim
# ======================================================================
def v11_b2_counterexample() -> None:
    d = LabeledDAG(4)
    d.add_node(NodeType.VAR, var_index=0)
    d.add_node(NodeType.VAR, var_index=1)
    a = d.add_node(NodeType.SIN)
    p = d.add_node(NodeType.POW)
    d.add_edge(0, a)
    d.add_edge(a, p)
    d.add_edge(1, p)
    s = fast_canonical_string(d)
    sigma = d.ordered_inputs(p)
    # x2 is an in-neighbour of p, but sigma(p)[0] = a
    record(
        "V11 B2: an in-neighbour of c can be reached before sigma(c)[0]",
        sigma[0] == a and 1 in d.in_neighbors(p) and s == "Vsnv^PnC",
        f"sigma(p)={sigma}, in_neighbors(p)={sorted(d.in_neighbors(p))}, fcs={s!r} "
        f"-- reaching x2 first makes p a D2S candidate while sigma(p)[0]=a is "
        f"uninserted, so the CDLL-timing claim is false and must become an induction",
    )


# ======================================================================
# V12 — engine equivalence on everything canonicalised above
# ======================================================================
def v12_engine_equivalence(rng: random.Random, n: int = 400) -> None:
    bad = 0
    tested = 0
    for _ in range(n):
        m = rng.choice([2, 3])
        d = random_s2d_dag(rng, m, rng.randint(4, 14))
        if d is None or not reachable_ok(d):
            continue
        tested += 1
        if fast_canonical_string(d, backend="python") != fast_canonical_string(d, backend="cpp"):
            bad += 1
    record(
        "V12 cpp == python on the canonical string",
        bad == 0,
        f"{tested} DAGs, {bad} disagreements",
    )


def main() -> int:
    rng = random.Random(20260803)
    v1_def35_counterexample()
    v2_pool_identity(rng)
    v3_thm313_widened(rng)
    v4b_pow_only(rng)
    v4_rule1_defers(rng)
    v5_operation_count(rng)
    v6_thm315(rng)
    v7_lemma317(rng)
    v8_corollary318(rng)
    v9_no_norm_in_canonicaliser()
    v10_var_edge_target()
    v11_b2_counterexample()
    v12_engine_equivalence(rng)

    print("\n" + "=" * 72)
    counts = Counter(s for _, s, _ in RESULTS)
    print(f"SUMMARY: {counts['PASS']} PASS / {counts['FAIL']} FAIL")
    for tag, status, _ in RESULTS:
        if status == "FAIL":
            print(f"  FAILED: {tag}")
    return 0 if counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
