"""Canonical string computation for labeled DAGs.

Computes the canonical IsalSR string w*_D, which is a complete labeled-DAG invariant:
    w*_D = w*_D'  iff  D and D' are isomorphic as labeled DAGs.

Key simplification from IsalGraph (Lopez-Rubio, 2025, arXiv:2512.10429v2):
since input variables are distinguishable and x_1 is a fixed, distinguished
starting node, we run D2S from x_1 ONLY. No iteration over starting nodes.

Three canonical algorithms:
    - ``canonical_string``: Exhaustive (lexmin of shortest). O(k!). Reference only.
    - ``pruned_canonical_string``: 6-tuple pruned exhaustive. Faster but approximate.
    - ``fast_canonical_string``: **PREFERRED.** Greedy-invariant with three modes:
        - ``"wl_only"`` (DEFAULT): Sort key ``(label_char, WL_hash)``.
          Uses 1-WL subtree hash only. O(k) precomputation. 1.43x faster than
          ``"wl_tiebreak"`` on evolved Bingo DAGs. Completeness verified k=1..8.
        - ``"wl_tiebreak"``: Sort key ``(label_char, 6-tuple↓, WL_hash)``.
          Uses 6-tuple + WL hash for ties. Previous default (2026-03-26).
        - ``"tuple_only"``: Sort key ``(label_char, 6-tuple↓)``.
          Legacy mode, 6-tuple only (no WL). Slowest on large DAGs.

Theoretical justification for WL-only default (Weisfeiler & Leman, 1968;
Shervashidze et al., JMLR 2011): The 1-WL subtree hash h(v) = hash(label(v),
multiset{h(c) : c in children(v)}) captures the full rooted subtree isomorphism
type. The 6-tuple captures only depth-3 neighborhood cardinalities. Therefore
h(v) = h(w) implies tau(v) = tau(w) (WL subsumes 6-tuple), but not conversely.
WL is strictly more discriminative.

Performance optimizations (2026-03):
    - Timeout support to skip pathological DAGs.
    - Uses unchecked accessors (node_label_unchecked, node_data_unchecked,
      has_edge_unchecked, out_neighbors_raw) to eliminate bounds-check overhead
      in the backtracking hot path.
    - Uses add_edge_unchecked for V/v insertions (new node has no out-edges,
      so acyclicity is guaranteed by construction — no BFS needed).
    - generate_pairs_sorted_by_sum is cached via @lru_cache.

Restriction: ZERO external dependencies. Only Python stdlib.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections import deque
from typing import Literal

from isalsr.core.cdll import CircularDoublyLinkedList
from isalsr.core.dag_to_string import generate_pairs_sorted_by_sum
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import BINARY_OPS, NODE_TYPE_TO_LABEL, NodeType

log = logging.getLogger(__name__)

CanonicalMode = Literal["wl_only", "wl_tiebreak", "tuple_only"]


class CanonicalTimeoutError(Exception):
    """Raised when canonical string computation exceeds the time budget."""


# ======================================================================
# Public API
# ======================================================================


def canonical_string(dag: LabeledDAG, *, timeout: float | None = None) -> str:
    """Compute the canonical IsalSR string via exhaustive backtracking.

    Explores ALL valid neighbor choices at each V/v branch point.
    Starts from x_1 (node 0) only — no iteration over starting nodes.

    This is a **complete labeled-DAG invariant**:
        canonical_string(D1) == canonical_string(D2)  iff  D1 ~ D2

    WARNING: For DAGs with many internal nodes, the unpruned search can be
    extremely slow (factorial branching). Prefer ``pruned_canonical_string``
    for production use.

    Args:
        dag: The labeled DAG to canonicalize.
        timeout: Maximum wall-clock seconds. Raises CanonicalTimeoutError if exceeded.

    Returns:
        The canonical string w*_D (shortest, then lexmin).
    """
    if dag.node_count == 0:
        return ""
    num_vars = len(dag.var_nodes())
    if dag.node_count == num_vars and dag.edge_count == 0:
        return ""
    # Normalize CONST creation edges to x_1 before canonical computation.
    # This eliminates redundancy from arbitrary CONST creation sources.
    normalized = dag.normalize_const_creation() if dag._has_const_nodes() else dag
    return _canonical_d2s(normalized, pruned=False, timeout=timeout)


def pruned_canonical_string(dag: LabeledDAG, *, timeout: float | None = None) -> str:
    """Compute an approximate canonical IsalSR string with 6-tuple pruning.

    At each V/v branch point, only candidates sharing the maximum
    6-component structural tuple are explored (grouped by label to avoid
    invalid cross-label pruning). This dramatically reduces the branching
    factor while producing the correct result in the vast majority of cases.

    Mathematical justification:
        The 6-tuple tau(v) = (|in_N1|, |out_N1|, ..., |out_N3|) is an
        automorphism-invariant descriptor. Candidates with the max tuple
        within the same label group form an equivalence class under
        automorphism. In most cases (empirically >99.97%), label-aware
        pruning produces the same optimal string as exhaustive search.

    Known limitation:
        In rare cases (0.028% empirically -- 8/28890 in our benchmarks),
        the pruned canonical may be **longer** than the true canonical
        from ``canonical_string()``. This occurs when the 6-tuple's local
        neighborhood density prediction is misaligned with the global
        pointer displacement cost (the number of N/P/n/p movement
        instructions needed to reach a candidate's position in the CDLL).
        A candidate with higher local connectivity (higher tuple) may be
        farther from the current pointer, requiring more movement tokens.

        Additionally, in ~0.09% of cases, the pruned result has the same
        length as the exhaustive result but differs lexicographically
        (a different but equally short tie-breaking choice).

        The pruned result remains a **consistent labeled-DAG invariant**
        (deterministic, same output for isomorphic inputs) and is always
        a valid D2S encoding. For guaranteed optimality, use
        ``canonical_string()`` instead.

    Args:
        dag: The labeled DAG to canonicalize.
        timeout: Maximum wall-clock seconds. Raises CanonicalTimeoutError if exceeded.

    Returns:
        The pruned canonical string (consistent invariant, usually optimal).
    """
    if dag.node_count == 0:
        return ""
    num_vars = len(dag.var_nodes())
    if dag.node_count == num_vars and dag.edge_count == 0:
        return ""
    # Normalize CONST creation edges to x_1 before canonical computation.
    normalized = dag.normalize_const_creation() if dag._has_const_nodes() else dag
    return _canonical_d2s(normalized, pruned=True, timeout=timeout)


def compute_structural_tuples(
    dag: LabeledDAG,
) -> list[tuple[int, int, int, int, int, int]]:
    """Compute 6-component structural tuples for all nodes.

    For each node v, compute:
        (|in_N1(v)|, |out_N1(v)|, |in_N2(v)|, |out_N2(v)|, |in_N3(v)|, |out_N3(v)|)

    These are automorphism-invariant structural descriptors used for
    pruning at V/v branch points in the canonical search.

    Args:
        dag: The labeled DAG.

    Returns:
        List of 6-tuples indexed by node ID.
    """
    n = dag.node_count
    return [_compute_node_tuple(dag, v) for v in range(n)]


def fast_canonical_string(
    dag: LabeledDAG,
    *,
    timeout: float | None = None,
    mode: CanonicalMode = "wl_only",
    use_wl_hash: bool | None = None,
) -> str:
    """Greedy-invariant canonical string from x_0.

    At each V/v decision, candidates are sorted by an isomorphism-invariant
    key. If the best candidate is unique, it is taken greedily (no
    backtracking). Ties are resolved by backtracking over tied candidates
    only (lexmin among tied).

    This is a **complete labeled-DAG invariant**: two DAGs produce the same
    ``fast_canonical_string`` if and only if they are isomorphic under the
    SR isomorphism definition (variables fixed, internal nodes permutable).

    NOT necessarily the shortest string (unlike ``canonical_string``), but
    near-O(k²) for most practical DAGs. O(t^d × k²) only when true
    automorphisms cause ties at d recursion levels.

    Three modes control the invariant sort key used at V/v branch points:

    - ``"wl_only"`` (DEFAULT): Key = ``(label_char, WL_hash)``.
      Uses 1-WL subtree hash only. O(k) precomputation. 1.43x faster than
      ``"wl_tiebreak"`` on evolved Bingo DAGs (k=6-14). Completeness
      verified for k=1..8 (all k! permutations). Theoretical basis:
      1-WL is strictly more discriminative than the 6-tuple
      (Weisfeiler & Leman, 1968; Shervashidze et al., JMLR 2011).
    - ``"wl_tiebreak"``: Key = ``(label_char, 6-tuple↓, WL_hash)``.
      Uses 6-tuple + WL hash for tie-breaking. Previous default.
    - ``"tuple_only"``: Key = ``(label_char, 6-tuple↓)``.
      Legacy mode, no WL hash. Slowest on large DAGs due to ties.

    Args:
        dag: The labeled DAG to canonicalize.
        timeout: Maximum wall-clock seconds. Raises CanonicalTimeoutError
            if exceeded.
        mode: Invariant sort key mode. One of ``"wl_only"`` (default),
            ``"wl_tiebreak"``, or ``"tuple_only"``.
        use_wl_hash: **Deprecated.** Use ``mode`` instead.
            ``True`` maps to ``mode="wl_tiebreak"``;
            ``False`` maps to ``mode="tuple_only"``.

    Returns:
        The fast canonical string (deterministic, isomorphism-invariant).
    """
    if use_wl_hash is not None:
        warnings.warn(
            "use_wl_hash is deprecated; use mode='wl_tiebreak' or mode='tuple_only'",
            DeprecationWarning,
            stacklevel=2,
        )
        mode = "wl_tiebreak" if use_wl_hash else "tuple_only"
    if dag.node_count == 0:
        return ""
    num_vars = len(dag.var_nodes())
    if dag.node_count == num_vars and dag.edge_count == 0:
        return ""
    normalized = dag.normalize_const_creation() if dag._has_const_nodes() else dag
    return _fast_canonical_d2s(normalized, timeout=timeout, mode=mode)


def dag_distance(d1: LabeledDAG, d2: LabeledDAG) -> int:
    """Approximate labeled-DAG edit distance via Levenshtein on canonical strings.

    A metric on labeled DAGs: d(D1, D2) = levenshtein(w*_D1, w*_D2).
    Satisfies symmetry, triangle inequality, identity of indiscernibles.

    Args:
        d1: First labeled DAG.
        d2: Second labeled DAG.

    Returns:
        Levenshtein edit distance between canonical strings.
    """
    return levenshtein(canonical_string(d1), canonical_string(d2))


def levenshtein(s: str, t: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Uses O(n*m) DP with O(min(n,m)) space via single-row optimization.

    Args:
        s: First string.
        t: Second string.

    Returns:
        Minimum edits (insertions, deletions, substitutions) to transform s into t.
    """
    if len(s) < len(t):
        return levenshtein(t, s)
    if len(t) == 0:
        return len(s)

    prev_row = list(range(len(t) + 1))
    for i, sc in enumerate(s):
        curr_row = [i + 1]
        for j, tc in enumerate(t):
            insert = prev_row[j + 1] + 1
            delete = curr_row[j] + 1
            replace = prev_row[j] + (0 if sc == tc else 1)
            curr_row.append(min(insert, delete, replace))
        prev_row = curr_row

    return prev_row[-1]


# ======================================================================
# Structural tuple computation (6-component, directed)
# ======================================================================


def _compute_node_tuple(dag: LabeledDAG, node: int) -> tuple[int, int, int, int, int, int]:
    """Compute the 6-component structural tuple for a single node.

    Performs truncated BFS in both directions (in-edges and out-edges)
    up to distance 3.
    """
    in1, in2, in3 = _bfs_distance_counts(dag, node, direction="in")
    out1, out2, out3 = _bfs_distance_counts(dag, node, direction="out")
    return (in1, out1, in2, out2, in3, out3)


def _bfs_distance_counts(dag: LabeledDAG, source: int, direction: str) -> tuple[int, int, int]:
    """BFS from source following edges in one direction, count nodes at distance 1, 2, 3.

    Args:
        dag: The DAG.
        source: Starting node.
        direction: "in" (follow in_neighbors) or "out" (follow out_neighbors).

    Returns:
        Tuple (count_d1, count_d2, count_d3).
    """
    n = dag.node_count
    dist: list[int] = [-1] * n
    dist[source] = 0

    queue: deque[int] = deque([source])
    counts = [0, 0, 0]

    # Use raw accessors for BFS (no frozenset copy per neighbor lookup).
    get_neighbors = dag.in_neighbors_raw if direction == "in" else dag.out_neighbors_raw

    while queue:
        u = queue.popleft()
        d = dist[u]
        if d >= 3:
            continue
        for v in get_neighbors(u):
            if dist[v] == -1:
                dist[v] = d + 1
                if dist[v] <= 3:
                    counts[dist[v] - 1] += 1
                    queue.append(v)

    return (counts[0], counts[1], counts[2])


# ======================================================================
# Internal: CDLL traversal and instruction emission helpers
# ======================================================================


def _walk(cdll: CircularDoublyLinkedList, ptr: int, steps: int) -> int:
    """Move *ptr* through the CDLL by *steps* (positive=next, negative=prev)."""
    for _ in range(abs(steps)):
        ptr = cdll.next_node(ptr) if steps > 0 else cdll.prev_node(ptr)
    return ptr


def _primary_moves(a: int) -> str:
    """Emit N or P instructions for primary pointer displacement *a*."""
    return "N" * a if a >= 0 else "P" * (-a)


def _secondary_moves(b: int) -> str:
    """Emit n or p instructions for secondary pointer displacement *b*."""
    return "n" * b if b >= 0 else "p" * (-b)


# ======================================================================
# Internal: exhaustive/pruned canonical D2S search
# ======================================================================


def _canonical_d2s(input_dag: LabeledDAG, *, pruned: bool, timeout: float | None) -> str:
    """Find the shortest, then lex-smallest D2S string from x_1.

    Initializes m VAR nodes in the output DAG, then runs exhaustive
    backtracking over all valid neighbor choices at V/v branch points.

    Args:
        input_dag: The input labeled DAG.
        pruned: If True, use 6-tuple pruning at V/v branch points.
        timeout: Maximum wall-clock seconds, or None for no limit.
    """
    n = input_dag.node_count
    og = LabeledDAG(n)
    cdll = CircularDoublyLinkedList(n)

    # Pre-compute structural tuples for pruning (only if pruned mode).
    tuples: list[tuple[int, int, int, int, int, int]] | None = None
    if pruned:
        tuples = compute_structural_tuples(input_dag)

    # Compute timeout deadline.
    deadline: float | None = None
    if timeout is not None:
        deadline = time.monotonic() + timeout

    # Initialize: map all m VAR nodes, insert into CDLL.
    i2o: dict[int, int] = {}
    o2i: dict[int, int] = {}
    var_nodes = sorted(
        input_dag.var_nodes(),
        key=lambda v: input_dag.node_data(v).get("var_index", v),
    )

    prev_cdll: int = -1
    first_cdll: int = -1
    for inp_node in var_nodes:
        out_node = og.add_node(
            NodeType.VAR,
            var_index=int(input_dag.node_data(inp_node).get("var_index", 0)),
        )
        i2o[inp_node] = out_node
        o2i[out_node] = inp_node
        cdll_node = cdll.insert_after(prev_cdll, out_node)
        if first_cdll == -1:
            first_cdll = cdll_node
        prev_cdll = cdll_node

    num_vars = len(var_nodes)
    nleft = n - num_vars
    eleft = input_dag.edge_count

    return _step(
        input_dag,
        og,
        cdll,
        first_cdll,
        first_cdll,
        i2o,
        o2i,
        nleft,
        eleft,
        "",
        tuples,
        deadline,
    )


def _step(
    ig: LabeledDAG,
    og: LabeledDAG,
    cdll: CircularDoublyLinkedList,
    pri: int,
    sec: int,
    i2o: dict[int, int],
    o2i: dict[int, int],
    nleft: int,
    eleft: int,
    prefix: str,
    tuples: list[tuple[int, int, int, int, int, int]] | None,
    deadline: float | None,
) -> str:
    """One step of the exhaustive/pruned canonical D2S search.

    Mirrors the greedy DAGToString algorithm but branches over all valid
    neighbor choices at V/v steps. Uses in-place mutation with undo
    (backtracking) instead of deep copies for performance.

    Performance: uses unchecked accessors (no bounds checking) and
    add_edge_unchecked (no BFS cycle check) for V/v insertions where
    acyclicity is guaranteed by construction.

    At V/v branch points:
        - Collect ALL uninserted outgoing neighbors of the tentative pointer node.
        - If pruned: filter to those with max 6-component structural tuple.
        - Backtrack over all remaining candidates, keeping the best (shortest, lexmin).

    C/c edges are deterministic (no branching).
    """
    if nleft <= 0 and eleft <= 0:
        return prefix

    # Timeout check (every entry into _step).
    if deadline is not None and time.monotonic() > deadline:
        raise CanonicalTimeoutError("Canonical string computation exceeded time budget")

    pairs = generate_pairs_sorted_by_sum(og.node_count)

    for a, b in pairs:
        # ---- tentative primary position ----
        tp = _walk(cdll, pri, a)
        tp_out = cdll.get_value(tp)
        tp_in = o2i[tp_out]

        # -- V: primary has uninserted outgoing neighbor --
        if nleft > 0:
            # Use raw set access (no frozenset copy).
            cands = [n for n in ig.out_neighbors_raw(tp_in) if n not in i2o]
            # BUG FIX B9: For binary ops, V must come from the first operand.
            cands = [
                c
                for c in cands
                if ig.node_label_unchecked(c) not in BINARY_OPS
                or not ig.ordered_inputs(c)
                or ig.ordered_inputs(c)[0] == tp_in
            ]
            if cands:
                # Pruning: group by label, keep max-tuple WITHIN each group.
                # Candidates with different labels are never automorphism-
                # equivalent (automorphisms preserve labels), so cross-label
                # pruning is invalid. Only same-label candidates compete.
                if tuples is not None:
                    label_groups: dict[NodeType, list[int]] = {}
                    for c in cands:
                        label_groups.setdefault(ig.node_label(c), []).append(c)
                    pruned: list[int] = []
                    for group in label_groups.values():
                        max_tup = max(tuples[c] for c in group)
                        pruned.extend(c for c in group if tuples[c] == max_tup)
                    cands = pruned

                mov = _primary_moves(a)
                best: str | None = None
                for c in cands:
                    # Use unchecked accessors (nodes guaranteed valid).
                    label = ig.node_label_unchecked(c)
                    label_char = NODE_TYPE_TO_LABEL[label]
                    data = ig.node_data_unchecked(c)

                    # Forward: add node + edge (unchecked — new node has no
                    # out-edges, so adding an edge TO it can never create a cycle).
                    new_out = og.add_node(
                        label,
                        var_index=int(data["var_index"]) if "var_index" in data else None,
                        const_value=float(data["const_value"]) if "const_value" in data else None,
                    )
                    i2o[c] = new_out
                    o2i[new_out] = c
                    og.add_edge_unchecked(tp_out, new_out)
                    new_cdll = cdll.insert_after(tp, new_out)

                    r = _step(
                        ig,
                        og,
                        cdll,
                        tp,
                        sec,
                        i2o,
                        o2i,
                        nleft - 1,
                        eleft - 1,
                        prefix + mov + "V" + label_char,
                        tuples,
                        deadline,
                    )
                    if best is None or (len(r), r) < (len(best), best):
                        best = r

                    # Backward
                    cdll.remove(new_cdll)
                    og.remove_edge(tp_out, new_out)
                    og.undo_node()
                    del i2o[c]
                    del o2i[new_out]

                return best  # type: ignore[return-value]

        # ---- tentative secondary position ----
        ts = _walk(cdll, sec, b)
        ts_out = cdll.get_value(ts)
        ts_in = o2i[ts_out]

        # -- v: secondary has uninserted outgoing neighbor --
        if nleft > 0:
            # Use raw set access (no frozenset copy).
            cands = [n for n in ig.out_neighbors_raw(ts_in) if n not in i2o]
            # BUG FIX B9: For binary ops, v must come from the first operand.
            cands = [
                c
                for c in cands
                if ig.node_label_unchecked(c) not in BINARY_OPS
                or not ig.ordered_inputs(c)
                or ig.ordered_inputs(c)[0] == ts_in
            ]
            if cands:
                # Same label-aware pruning as for V (primary).
                if tuples is not None:
                    sec_groups: dict[NodeType, list[int]] = {}
                    for c in cands:
                        sec_groups.setdefault(ig.node_label(c), []).append(c)
                    sec_pruned: list[int] = []
                    for group in sec_groups.values():
                        max_tup = max(tuples[c] for c in group)
                        sec_pruned.extend(c for c in group if tuples[c] == max_tup)
                    cands = sec_pruned

                mov = _secondary_moves(b)
                best = None
                for c in cands:
                    label = ig.node_label_unchecked(c)
                    label_char = NODE_TYPE_TO_LABEL[label]
                    data = ig.node_data_unchecked(c)

                    # Forward (unchecked — same reasoning as V above).
                    new_out = og.add_node(
                        label,
                        var_index=int(data["var_index"]) if "var_index" in data else None,
                        const_value=float(data["const_value"]) if "const_value" in data else None,
                    )
                    i2o[c] = new_out
                    o2i[new_out] = c
                    og.add_edge_unchecked(ts_out, new_out)
                    new_cdll = cdll.insert_after(ts, new_out)

                    r = _step(
                        ig,
                        og,
                        cdll,
                        pri,
                        ts,
                        i2o,
                        o2i,
                        nleft - 1,
                        eleft - 1,
                        prefix + mov + "v" + label_char,
                        tuples,
                        deadline,
                    )
                    if best is None or (len(r), r) < (len(best), best):
                        best = r

                    # Backward
                    cdll.remove(new_cdll)
                    og.remove_edge(ts_out, new_out)
                    og.undo_node()
                    del i2o[c]
                    del o2i[new_out]

                return best  # type: ignore[return-value]

        # -- C: edge primary -> secondary in input but not output --
        # Use unchecked has_edge (nodes guaranteed valid in canonical search).
        if ig.has_edge_unchecked(tp_in, ts_in) and not og.has_edge_unchecked(tp_out, ts_out):
            og.add_edge(tp_out, ts_out)
            r = _step(
                ig,
                og,
                cdll,
                tp,
                ts,
                i2o,
                o2i,
                nleft,
                eleft - 1,
                prefix + _primary_moves(a) + _secondary_moves(b) + "C",
                tuples,
                deadline,
            )
            og.remove_edge(tp_out, ts_out)
            return r

        # -- c: edge secondary -> primary in input but not output --
        if ig.has_edge_unchecked(ts_in, tp_in) and not og.has_edge_unchecked(ts_out, tp_out):
            og.add_edge(ts_out, tp_out)
            r = _step(
                ig,
                og,
                cdll,
                tp,
                ts,
                i2o,
                o2i,
                nleft,
                eleft - 1,
                prefix + _primary_moves(a) + _secondary_moves(b) + "c",
                tuples,
                deadline,
            )
            og.remove_edge(ts_out, tp_out)
            return r

    raise RuntimeError(
        f"Canonical D2S: no valid operation found. Remaining: {nleft} nodes, {eleft} edges."
    )


# ======================================================================
# Internal: fast greedy-invariant canonical D2S
# ======================================================================


def _compute_subtree_hashes(dag: LabeledDAG) -> list[int]:
    """Compute Weisfeiler-Leman subtree hashes for all nodes.

    Bottom-up (leaves first): ``hash(node) = hash(label, sorted(child_hashes))``.
    Isomorphism-invariant by construction — isomorphic subtrees produce
    identical hashes. O(k) total via topological ordering.

    Args:
        dag: The labeled DAG.

    Returns:
        List of integer hashes indexed by node ID.
    """
    n = dag.node_count
    node_hash: list[int] = [0] * n
    out_deg = [0] * n
    for u in range(n):
        out_deg[u] = len(dag.out_neighbors_raw(u))

    # Process leaves first (nodes with no outgoing edges)
    queue: deque[int] = deque()
    for u in range(n):
        if out_deg[u] == 0:
            queue.append(u)

    processed = [False] * n
    while queue:
        u = queue.popleft()
        if processed[u]:
            continue
        processed[u] = True
        children_hashes = sorted(node_hash[v] for v in dag.out_neighbors_raw(u))
        node_hash[u] = hash((dag.node_label_unchecked(u), tuple(children_hashes)))
        for v in dag.in_neighbors_raw(u):
            out_deg[v] -= 1
            if out_deg[v] == 0 and not processed[v]:
                queue.append(v)

    return node_hash


def _invariant_candidate_key(
    node: int,
    ig: LabeledDAG,
    tuples: list[tuple[int, int, int, int, int, int]] | None,
    subtree_hashes: list[int] | None,
) -> tuple:  # type: ignore[type-arg]
    """Isomorphism-invariant sort key for a V/v candidate.

    The key shape depends on which precomputed data is available
    (controlled by ``mode`` in ``fast_canonical_string``):

    - **wl_only** (tuples=None, hashes set): ``(label_char, WL_hash)``
    - **wl_tiebreak** (both set): ``(label_char, neg_6-tuple, WL_hash)``
    - **tuple_only** (hashes=None): ``(label_char, neg_6-tuple)``

    All components are isomorphism-invariant. Within a single
    ``_fast_step`` invocation the mode is fixed, so all candidates
    share the same key arity — comparison is well-defined.
    """
    label_char = NODE_TYPE_TO_LABEL[ig.node_label_unchecked(node)]
    if tuples is not None and subtree_hashes is not None:
        # wl_tiebreak: 3-component key
        return (label_char, tuple(-x for x in tuples[node]), subtree_hashes[node])
    if tuples is not None:
        # tuple_only: 2-component key (no WL)
        return (label_char, tuple(-x for x in tuples[node]))
    # wl_only: 2-component key (no 6-tuple)
    return (label_char, subtree_hashes[node])  # type: ignore[index]


def _fast_canonical_d2s(
    input_dag: LabeledDAG,
    *,
    timeout: float | None,
    mode: CanonicalMode = "wl_only",
) -> str:
    """Find the greedy-invariant canonical D2S string from x_0.

    Same setup as ``_canonical_d2s`` but uses ``_fast_step`` which makes
    greedy choices with invariant tie-breaking instead of exhaustive search.

    Precomputation depends on ``mode``:
    - ``"wl_only"``: WL subtree hashes only (O(k)). Default.
    - ``"wl_tiebreak"``: 6-tuple (O(k²)) + WL hashes (O(k)).
    - ``"tuple_only"``: 6-tuple only (O(k²)).
    """
    n = input_dag.node_count
    og = LabeledDAG(n)
    cdll = CircularDoublyLinkedList(n)

    # Mode-dependent precomputation.
    tuples: list[tuple[int, int, int, int, int, int]] | None = None
    subtree_hashes: list[int] | None = None
    if mode == "tuple_only":
        tuples = compute_structural_tuples(input_dag)
    elif mode == "wl_tiebreak":
        tuples = compute_structural_tuples(input_dag)
        subtree_hashes = _compute_subtree_hashes(input_dag)
    else:  # wl_only (default)
        subtree_hashes = _compute_subtree_hashes(input_dag)

    deadline: float | None = None
    if timeout is not None:
        deadline = time.monotonic() + timeout

    # Initialize: map all m VAR nodes, insert into CDLL.
    i2o: dict[int, int] = {}
    o2i: dict[int, int] = {}
    var_nodes = sorted(
        input_dag.var_nodes(),
        key=lambda v: input_dag.node_data(v).get("var_index", v),
    )

    prev_cdll: int = -1
    first_cdll: int = -1
    for inp_node in var_nodes:
        out_node = og.add_node(
            NodeType.VAR,
            var_index=int(input_dag.node_data(inp_node).get("var_index", 0)),
        )
        i2o[inp_node] = out_node
        o2i[out_node] = inp_node
        cdll_node = cdll.insert_after(prev_cdll, out_node)
        if first_cdll == -1:
            first_cdll = cdll_node
        prev_cdll = cdll_node

    num_vars = len(var_nodes)
    nleft = n - num_vars
    eleft = input_dag.edge_count

    return _fast_step(
        input_dag,
        og,
        cdll,
        first_cdll,
        first_cdll,
        i2o,
        o2i,
        nleft,
        eleft,
        "",
        tuples,
        subtree_hashes,
        deadline,
    )


def _fast_step(
    ig: LabeledDAG,
    og: LabeledDAG,
    cdll: CircularDoublyLinkedList,
    pri: int,
    sec: int,
    i2o: dict[int, int],
    o2i: dict[int, int],
    nleft: int,
    eleft: int,
    prefix: str,
    tuples: list[tuple[int, int, int, int, int, int]] | None,
    subtree_hashes: list[int] | None,
    deadline: float | None,
) -> str:
    """One step of the greedy-invariant canonical D2S.

    Same structure as ``_step`` but at V/v branch points:
    1. Sort candidates by invariant key (mode-dependent sort key).
    2. Take ONLY the best-key group.
    3. If unique → greedy (no backtracking).
    4. If tied → backtrack over tied candidates only, take lexmin.

    C/c edges remain deterministic (no branching).
    Movement pairs iterate in cost order with early return on V/v (unchanged).
    """
    if nleft <= 0 and eleft <= 0:
        return prefix

    if deadline is not None and time.monotonic() > deadline:
        raise CanonicalTimeoutError("Fast canonical string computation exceeded time budget")

    pairs = generate_pairs_sorted_by_sum(og.node_count)

    for a, b in pairs:
        # ---- tentative primary position ----
        tp = _walk(cdll, pri, a)
        tp_out = cdll.get_value(tp)
        tp_in = o2i[tp_out]

        # -- V: primary has uninserted outgoing neighbor --
        if nleft > 0:
            cands = [n for n in ig.out_neighbors_raw(tp_in) if n not in i2o]
            cands = [
                c
                for c in cands
                if ig.node_label_unchecked(c) not in BINARY_OPS
                or not ig.ordered_inputs(c)
                or ig.ordered_inputs(c)[0] == tp_in
            ]
            if cands:
                # Sort by invariant key, group by best key.
                cands.sort(key=lambda c: _invariant_candidate_key(c, ig, tuples, subtree_hashes))
                best_key = _invariant_candidate_key(cands[0], ig, tuples, subtree_hashes)
                tied = [
                    c
                    for c in cands
                    if _invariant_candidate_key(c, ig, tuples, subtree_hashes) == best_key
                ]

                mov = _primary_moves(a)
                best: str | None = None
                for c in tied:
                    label = ig.node_label_unchecked(c)
                    label_char = NODE_TYPE_TO_LABEL[label]
                    data = ig.node_data_unchecked(c)

                    new_out = og.add_node(
                        label,
                        var_index=int(data["var_index"]) if "var_index" in data else None,
                        const_value=float(data["const_value"]) if "const_value" in data else None,
                    )
                    i2o[c] = new_out
                    o2i[new_out] = c
                    og.add_edge_unchecked(tp_out, new_out)
                    new_cdll = cdll.insert_after(tp, new_out)

                    r = _fast_step(
                        ig,
                        og,
                        cdll,
                        tp,
                        sec,
                        i2o,
                        o2i,
                        nleft - 1,
                        eleft - 1,
                        prefix + mov + "V" + label_char,
                        tuples,
                        subtree_hashes,
                        deadline,
                    )
                    if best is None or (len(r), r) < (len(best), best):
                        best = r

                    # Backward
                    cdll.remove(new_cdll)
                    og.remove_edge(tp_out, new_out)
                    og.undo_node()
                    del i2o[c]
                    del o2i[new_out]

                return best  # type: ignore[return-value]

        # ---- tentative secondary position ----
        ts = _walk(cdll, sec, b)
        ts_out = cdll.get_value(ts)
        ts_in = o2i[ts_out]

        # -- v: secondary has uninserted outgoing neighbor --
        if nleft > 0:
            cands = [n for n in ig.out_neighbors_raw(ts_in) if n not in i2o]
            cands = [
                c
                for c in cands
                if ig.node_label_unchecked(c) not in BINARY_OPS
                or not ig.ordered_inputs(c)
                or ig.ordered_inputs(c)[0] == ts_in
            ]
            if cands:
                cands.sort(key=lambda c: _invariant_candidate_key(c, ig, tuples, subtree_hashes))
                best_key = _invariant_candidate_key(cands[0], ig, tuples, subtree_hashes)
                tied = [
                    c
                    for c in cands
                    if _invariant_candidate_key(c, ig, tuples, subtree_hashes) == best_key
                ]

                mov = _secondary_moves(b)
                best = None
                for c in tied:
                    label = ig.node_label_unchecked(c)
                    label_char = NODE_TYPE_TO_LABEL[label]
                    data = ig.node_data_unchecked(c)

                    new_out = og.add_node(
                        label,
                        var_index=int(data["var_index"]) if "var_index" in data else None,
                        const_value=float(data["const_value"]) if "const_value" in data else None,
                    )
                    i2o[c] = new_out
                    o2i[new_out] = c
                    og.add_edge_unchecked(ts_out, new_out)
                    new_cdll = cdll.insert_after(ts, new_out)

                    r = _fast_step(
                        ig,
                        og,
                        cdll,
                        pri,
                        ts,
                        i2o,
                        o2i,
                        nleft - 1,
                        eleft - 1,
                        prefix + mov + "v" + label_char,
                        tuples,
                        subtree_hashes,
                        deadline,
                    )
                    if best is None or (len(r), r) < (len(best), best):
                        best = r

                    # Backward
                    cdll.remove(new_cdll)
                    og.remove_edge(ts_out, new_out)
                    og.undo_node()
                    del i2o[c]
                    del o2i[new_out]

                return best  # type: ignore[return-value]

        # -- C: edge primary -> secondary --
        if ig.has_edge_unchecked(tp_in, ts_in) and not og.has_edge_unchecked(tp_out, ts_out):
            og.add_edge(tp_out, ts_out)
            r = _fast_step(
                ig,
                og,
                cdll,
                tp,
                ts,
                i2o,
                o2i,
                nleft,
                eleft - 1,
                prefix + _primary_moves(a) + _secondary_moves(b) + "C",
                tuples,
                subtree_hashes,
                deadline,
            )
            og.remove_edge(tp_out, ts_out)
            return r

        # -- c: edge secondary -> primary --
        if ig.has_edge_unchecked(ts_in, tp_in) and not og.has_edge_unchecked(ts_out, tp_out):
            og.add_edge(ts_out, tp_out)
            r = _fast_step(
                ig,
                og,
                cdll,
                tp,
                ts,
                i2o,
                o2i,
                nleft,
                eleft - 1,
                prefix + _primary_moves(a) + _secondary_moves(b) + "c",
                tuples,
                subtree_hashes,
                deadline,
            )
            og.remove_edge(ts_out, tp_out)
            return r

    raise RuntimeError(
        f"Fast canonical D2S: no valid operation found. Remaining: {nleft} nodes, {eleft} edges."
    )
