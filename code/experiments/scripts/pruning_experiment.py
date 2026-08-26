"""Experiment V3: Iterative Deepening + Hybrid Pruning for Canonical Strings.

Tests provably-correct pruning strategies with increasing sophistication:
    1. EXHAUSTIVE:  No pruning (ground truth for w*_D)
    2. BB:          Branch-and-bound with loose LB (v2 control)
    3. IDS:         Iterative deepening search with length limit
    4. IDS_HYBRID:  IDS + smart candidate ordering + 1-step movement look-ahead
    5. IDS_DEDUP:   IDS_HYBRID + WL-based equivalence deduplication

Theoretical justification:

    IDS (Land & Doig, 1960; Korf, 1985):
        For target length L, prune any branch where len(prefix) + LB(remaining) > L.
        Start L at the theoretical minimum (2*k + standalone_edges).  Increment L
        until a solution is found.  The first L with a solution yields w*_D.
        Each sub-optimal iteration is fast (almost everything pruned).
        Total overhead: O(gap * branching_factor) where gap = optimal_length - LB.

    1-step look-ahead:
        After inserting candidate c, check if ANY operation is valid at (0,0)
        (zero movement cost).  If not, next step costs >= 1 movement instruction.
        Add this to the LB.  Valid because it never overestimates.

    Smart ordering:
        Order candidates by decreasing number of uninserted out-neighbors, then
        by WL color.  Candidates with more uninserted out-neighbors tend to enable
        cheaper (zero-movement) future steps, leading to shorter strings.

    WL-based dedup:
        At each branch point, candidates with the same WL color are candidates for
        automorphism equivalence.  Exploring one representative per WL-color class
        is a HEURISTIC (not guaranteed correct).  Tested empirically.

Usage:
    python experiments/scripts/pruning_experiment.py

Author: Mario Pascual Gonzalez (mpascual@uma.es)
Date: 2026-03-17

References:
    - Korf (1985). Depth-first iterative-deepening. Artificial Intelligence 27(1):97-109.
    - Land & Doig (1960). Econometrica 28(3):497-520.
    - McKay & Piperno (2014). Practical graph isomorphism, II. J. Symbolic Computation 60:94-112.
    - Lopez-Rubio (2025). arXiv:2512.10429v2. IsalGraph.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from isalsr.core.canonical import (
    _primary_moves,
    _secondary_moves,
    _walk,
)
from isalsr.core.cdll import CircularDoublyLinkedList
from isalsr.core.dag_to_string import generate_pairs_sorted_by_sum
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NODE_TYPE_TO_LABEL, NodeType

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ======================================================================
# Strategy enum
# ======================================================================


class Strategy(Enum):
    EXHAUSTIVE = "exhaustive"
    BB = "branch-and-bound"
    IDS = "IDS (plain)"
    IDS_HYBRID = "IDS + order + look-ahead"
    IDS_DEDUP = "IDS + order + LA + dedup"


# ======================================================================
# Metrics
# ======================================================================


@dataclass
class SearchMetrics:
    strategy: Strategy
    step_count: int = 0
    branch_points: int = 0
    bound_prunes: int = 0
    dedup_prunes: int = 0
    ids_iterations: int = 0
    canonical_result: str = ""
    wall_time_s: float = 0.0
    correct: bool = True


# ======================================================================
# WL coloring
# ======================================================================


def compute_wl_colors(dag: LabeledDAG, max_iterations: int = 10) -> list[int]:
    """1-Weisfeiler-Leman iterative color refinement."""
    n = dag.node_count
    color_map: dict[str, int] = {}
    colors: list[int] = []
    for v in range(n):
        label = dag.node_label(v)
        key = label.value if label is not None else "NONE"
        if key not in color_map:
            color_map[key] = len(color_map)
        colors.append(color_map[key])

    for _ in range(max_iterations):
        new_color_map: dict[tuple[int, tuple[int, ...], tuple[int, ...]], int] = {}
        new_colors: list[int] = []
        for v in range(n):
            in_c = tuple(sorted(colors[u] for u in dag.in_neighbors(v)))
            out_c = tuple(sorted(colors[u] for u in dag.out_neighbors(v)))
            sig = (colors[v], in_c, out_c)
            if sig not in new_color_map:
                new_color_map[sig] = len(new_color_map)
            new_colors.append(new_color_map[sig])
        if len(set(new_colors)) == len(set(colors)):
            break
        colors = new_colors

    return colors


# ======================================================================
# Lower bound computation
# ======================================================================


def remaining_lb(nleft: int, eleft: int) -> int:
    """Lower bound on remaining string length.

    Each node: >= 2 chars (V/v + label).  Each co-inserts 1 edge.
    Standalone edges: >= 1 char each (C/c).
    Movement: >= 0 (optimistic, never overestimates).

    Result: 2*nleft + max(0, eleft - nleft) = nleft + eleft  (when nleft <= eleft).
    """
    return 2 * nleft + max(0, eleft - nleft)


def has_zero_cost_next_op(
    input_dag: LabeledDAG,
    og: LabeledDAG,
    cdll: CircularDoublyLinkedList,
    pri: int,
    sec: int,
    i2o: dict[int, int],
    o2i: dict[int, int],
    nleft: int,
    eleft: int,
) -> bool:
    """Check if ANY valid operation exists at displacement (0,0).

    If yes, the next step costs 0 movement instructions.
    If no, the next step costs >= 1 movement instruction.
    """
    if nleft <= 0 and eleft <= 0:
        return True  # Done, no more ops needed.

    pri_out = cdll.get_value(pri)
    sec_out = cdll.get_value(sec)
    pri_in = o2i[pri_out]
    sec_in = o2i[sec_out]

    # V: primary has uninserted outgoing neighbor.
    if nleft > 0:
        for nb in input_dag.out_neighbors(pri_in):
            if nb not in i2o:
                return True

    # v: secondary has uninserted outgoing neighbor.
    if nleft > 0:
        for nb in input_dag.out_neighbors(sec_in):
            if nb not in i2o:
                return True

    # C: edge pri->sec in input but not output.
    if eleft > 0 and input_dag.has_edge(pri_in, sec_in) and not og.has_edge(pri_out, sec_out):
        return True

    # c: edge sec->pri in input but not output.
    if eleft > 0 and input_dag.has_edge(sec_in, pri_in) and not og.has_edge(sec_out, pri_out):
        return True

    return False


# ======================================================================
# Instrumented canonical search engine
# ======================================================================


def run_strategy(input_dag: LabeledDAG, strategy: Strategy) -> SearchMetrics:
    """Run canonical D2S with the given strategy, collecting metrics."""
    metrics = SearchMetrics(strategy=strategy)

    # Pre-compute signatures.
    wl_colors = compute_wl_colors(input_dag)

    n = input_dag.node_count
    og = LabeledDAG(n)
    cdll = CircularDoublyLinkedList(n)

    # Initialize VAR nodes.
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
        cn = cdll.insert_after(prev_cdll, out_node)
        if first_cdll == -1:
            first_cdll = cn
        prev_cdll = cn

    num_vars = len(var_nodes)
    nleft = n - num_vars
    eleft = input_dag.edge_count

    best_global: list[str | None] = [None]
    use_ids = strategy in (Strategy.IDS, Strategy.IDS_HYBRID, Strategy.IDS_DEDUP)
    use_bb = strategy == Strategy.BB
    use_ordering = strategy in (Strategy.IDS_HYBRID, Strategy.IDS_DEDUP)
    use_lookahead = strategy in (Strategy.IDS_HYBRID, Strategy.IDS_DEDUP)
    use_dedup = strategy == Strategy.IDS_DEDUP

    t0 = time.perf_counter()

    def _step(
        pri: int,
        sec: int,
        nleft: int,
        eleft: int,
        prefix: str,
        length_limit: int,
    ) -> str | None:
        """One step of canonical backtracking with optional pruning."""
        metrics.step_count += 1

        # IDS length check MUST be BEFORE base case: a path with movement
        # cost can reach the leaf with len(prefix) > length_limit.  Without
        # this guard, the IDS loop breaks on a suboptimal non-None result.
        if use_ids:
            total_lb = len(prefix) + remaining_lb(nleft, eleft)
            if total_lb > length_limit:
                metrics.bound_prunes += 1
                return None

        if nleft <= 0 and eleft <= 0:
            return prefix

        pairs = generate_pairs_sorted_by_sum(og.node_count)

        for a, b in pairs:
            tp = _walk(cdll, pri, a)
            tp_out = cdll.get_value(tp)
            tp_in = o2i[tp_out]

            # -- V branch point --
            if nleft > 0:
                cands = [nd for nd in input_dag.out_neighbors(tp_in) if nd not in i2o]
                if cands:
                    metrics.branch_points += 1
                    mov = _primary_moves(a)
                    mov_len = len(mov)

                    # Smart ordering: high uninserted-out-degree first, then WL color.
                    if use_ordering:
                        cands.sort(
                            key=lambda c: (
                                sum(1 for nb in input_dag.out_neighbors(c) if nb not in i2o),
                                wl_colors[c],
                            ),
                            reverse=True,
                        )

                    # Dedup: skip candidates with same WL color as one already explored.
                    seen_colors: set[int] = set()

                    best: str | None = None
                    for c in cands:
                        label = input_dag.node_label(c)
                        label_char = NODE_TYPE_TO_LABEL[label]
                        step_cost = mov_len + 2  # movement + V + label

                        # B&B pruning.
                        if use_bb:
                            lb = len(prefix) + step_cost + remaining_lb(nleft - 1, eleft - 1)
                            if best_global[0] is not None and lb > len(best_global[0]):
                                metrics.bound_prunes += 1
                                continue
                            if best is not None and lb > len(best):
                                metrics.bound_prunes += 1
                                continue

                        # IDS pruning (per-candidate).
                        if use_ids:
                            lb = len(prefix) + step_cost + remaining_lb(nleft - 1, eleft - 1)
                            if lb > length_limit:
                                metrics.bound_prunes += 1
                                continue

                        # Dedup: skip if same WL color already explored.
                        if use_dedup:
                            if wl_colors[c] in seen_colors:
                                metrics.dedup_prunes += 1
                                continue
                            seen_colors.add(wl_colors[c])

                        data = input_dag.node_data(c)

                        # Forward.
                        new_out = og.add_node(
                            label,
                            var_index=int(data["var_index"]) if "var_index" in data else None,
                            const_value=(
                                float(data["const_value"]) if "const_value" in data else None
                            ),
                        )
                        i2o[c] = new_out
                        o2i[new_out] = c
                        og.add_edge(tp_out, new_out)
                        new_cdll = cdll.insert_after(tp, new_out)

                        # Enhanced LB: 1-step look-ahead for movement cost.
                        child_limit = length_limit
                        if use_lookahead and nleft - 1 > 0:
                            if not has_zero_cost_next_op(
                                input_dag, og, cdll, tp, sec, i2o, o2i, nleft - 1, eleft - 1
                            ):
                                # Next step requires >= 1 movement.  Tighten limit.
                                child_lb = (
                                    len(prefix) + step_cost + 1 + remaining_lb(nleft - 1, eleft - 1)
                                )
                                if child_lb > length_limit:
                                    # Even with 1 extra movement, exceeds limit.  Prune.
                                    cdll.remove(new_cdll)
                                    og.remove_edge(tp_out, new_out)
                                    og.undo_node()
                                    del i2o[c]
                                    del o2i[new_out]
                                    metrics.bound_prunes += 1
                                    continue

                        r = _step(
                            tp,
                            sec,
                            nleft - 1,
                            eleft - 1,
                            prefix + mov + "V" + label_char,
                            child_limit,
                        )

                        if r is not None and (best is None or (len(r), r) < (len(best), best)):
                            best = r
                            if use_bb or use_ids:
                                if best_global[0] is None or (len(best), best) < (
                                    len(best_global[0]),
                                    best_global[0],
                                ):
                                    best_global[0] = best

                        # Backward.
                        cdll.remove(new_cdll)
                        og.remove_edge(tp_out, new_out)
                        og.undo_node()
                        del i2o[c]
                        del o2i[new_out]

                    return best

            # -- v branch point --
            ts = _walk(cdll, sec, b)
            ts_out = cdll.get_value(ts)
            ts_in = o2i[ts_out]

            if nleft > 0:
                cands = [nd for nd in input_dag.out_neighbors(ts_in) if nd not in i2o]
                if cands:
                    metrics.branch_points += 1
                    mov = _secondary_moves(b)
                    mov_len = len(mov)

                    if use_ordering:
                        cands.sort(
                            key=lambda c: (
                                sum(1 for nb in input_dag.out_neighbors(c) if nb not in i2o),
                                wl_colors[c],
                            ),
                            reverse=True,
                        )

                    seen_colors = set()
                    best = None

                    for c in cands:
                        label = input_dag.node_label(c)
                        label_char = NODE_TYPE_TO_LABEL[label]
                        step_cost = mov_len + 2

                        if use_bb:
                            lb = len(prefix) + step_cost + remaining_lb(nleft - 1, eleft - 1)
                            if best_global[0] is not None and lb > len(best_global[0]):
                                metrics.bound_prunes += 1
                                continue
                            if best is not None and lb > len(best):
                                metrics.bound_prunes += 1
                                continue

                        if use_ids:
                            lb = len(prefix) + step_cost + remaining_lb(nleft - 1, eleft - 1)
                            if lb > length_limit:
                                metrics.bound_prunes += 1
                                continue

                        if use_dedup:
                            if wl_colors[c] in seen_colors:
                                metrics.dedup_prunes += 1
                                continue
                            seen_colors.add(wl_colors[c])

                        data = input_dag.node_data(c)

                        new_out = og.add_node(
                            label,
                            var_index=int(data["var_index"]) if "var_index" in data else None,
                            const_value=(
                                float(data["const_value"]) if "const_value" in data else None
                            ),
                        )
                        i2o[c] = new_out
                        o2i[new_out] = c
                        og.add_edge(ts_out, new_out)
                        new_cdll = cdll.insert_after(ts, new_out)

                        if use_lookahead and nleft - 1 > 0:
                            if not has_zero_cost_next_op(
                                input_dag, og, cdll, pri, ts, i2o, o2i, nleft - 1, eleft - 1
                            ):
                                child_lb = (
                                    len(prefix) + step_cost + 1 + remaining_lb(nleft - 1, eleft - 1)
                                )
                                if child_lb > length_limit:
                                    cdll.remove(new_cdll)
                                    og.remove_edge(ts_out, new_out)
                                    og.undo_node()
                                    del i2o[c]
                                    del o2i[new_out]
                                    metrics.bound_prunes += 1
                                    continue

                        r = _step(
                            pri,
                            ts,
                            nleft - 1,
                            eleft - 1,
                            prefix + mov + "v" + label_char,
                            length_limit,
                        )

                        if r is not None and (best is None or (len(r), r) < (len(best), best)):
                            best = r
                            if use_bb or use_ids:
                                if best_global[0] is None or (len(best), best) < (
                                    len(best_global[0]),
                                    best_global[0],
                                ):
                                    best_global[0] = best

                        cdll.remove(new_cdll)
                        og.remove_edge(ts_out, new_out)
                        og.undo_node()
                        del i2o[c]
                        del o2i[new_out]

                    return best

            # -- C edge --
            if eleft > 0:
                if input_dag.has_edge(tp_in, ts_in) and not og.has_edge(tp_out, ts_out):
                    if og.add_edge(tp_out, ts_out):
                        mov = _primary_moves(a) + _secondary_moves(b)
                        r = _step(tp, ts, nleft, eleft - 1, prefix + mov + "C", length_limit)
                        og.remove_edge(tp_out, ts_out)
                        return r

            # -- c edge --
            if eleft > 0:
                if input_dag.has_edge(ts_in, tp_in) and not og.has_edge(ts_out, tp_out):
                    if og.add_edge(ts_out, tp_out):
                        mov = _primary_moves(a) + _secondary_moves(b)
                        r = _step(tp, ts, nleft, eleft - 1, prefix + mov + "c", length_limit)
                        og.remove_edge(ts_out, tp_out)
                        return r

        return prefix  # Fallback.

    if use_ids:
        # Iterative deepening: start at LB, increment until solution found.
        lb_init = remaining_lb(nleft, eleft)
        for L in range(lb_init, lb_init + 200):
            metrics.ids_iterations += 1
            best_global[0] = None  # Reset global best per iteration.
            metrics_step_before = metrics.step_count
            result = _step(first_cdll, first_cdll, nleft, eleft, "", L)
            if result is not None:
                metrics.canonical_result = result
                break
    else:
        result = _step(first_cdll, first_cdll, nleft, eleft, "", 10**9)
        if result is not None:
            metrics.canonical_result = result

    metrics.wall_time_s = time.perf_counter() - t0
    return metrics


# ======================================================================
# DAG generation
# ======================================================================


def make_random_sr_dag(num_vars: int, num_internal: int, seed: int) -> LabeledDAG:
    """Generate a random SR expression DAG."""
    import random

    rng = random.Random(seed)
    total = num_vars + num_internal
    dag = LabeledDAG(max_nodes=total + 2)

    for i in range(num_vars):
        dag.add_node(NodeType.VAR, var_index=i)

    unary_ops = [NodeType.SIN, NodeType.COS, NodeType.EXP, NodeType.LOG, NodeType.ABS]
    binary_ops = [NodeType.ADD, NodeType.MUL, NodeType.SUB, NodeType.DIV]

    for _ in range(num_internal):
        existing = list(range(dag.node_count))
        if rng.random() < 0.6 or len(existing) < 2:
            op = rng.choice(unary_ops)
            parent = rng.choice(existing)
            nid = dag.add_node(op)
            dag.add_edge(parent, nid)
        else:
            op = rng.choice(binary_ops)
            parents = rng.sample(existing, 2)
            nid = dag.add_node(op)
            for p in parents:
                dag.add_edge(p, nid)

    return dag


# ======================================================================
# Main experiment
# ======================================================================


def run_experiment() -> None:
    """Run the IDS + hybrid pruning experiment."""
    log.info("=" * 86)
    log.info("EXPERIMENT V3: Iterative Deepening + Hybrid Pruning for Canonical Strings")
    log.info("=" * 86)
    log.info("")

    configs = [
        (1, 3, 20),
        (1, 4, 20),
        (1, 5, 15),
        (2, 4, 15),
        (2, 5, 10),
        (2, 6, 10),
        (1, 7, 5),
        (2, 7, 5),
        (1, 8, 3),
    ]

    strategies = list(Strategy)

    agg: dict[tuple[int, int, Strategy], list[SearchMetrics]] = defaultdict(list)
    failure_counts: dict[Strategy, int] = defaultdict(int)
    total_counts: dict[Strategy, int] = defaultdict(int)
    # Track IDS gap: optimal_length - LB.
    gaps: list[int] = []

    for num_vars, num_internal, num_seeds in configs:
        log.info(f"--- Config: {num_vars} vars, {num_internal} internal, {num_seeds} seeds ---")

        for seed in range(num_seeds):
            dag = make_random_sr_dag(num_vars, num_internal, seed=seed * 1000 + num_internal)

            results: dict[Strategy, SearchMetrics] = {}
            for strat in strategies:
                m = run_strategy(dag, strat)
                results[strat] = m
                agg[(num_vars, num_internal, strat)].append(m)

            gt = results[Strategy.EXHAUSTIVE].canonical_result

            # Cross-validate against original canonical_string implementation.
            from isalsr.core.canonical import canonical_string as _canon_ref

            gt_original = _canon_ref(dag)
            if gt != gt_original:
                log.error(
                    f"  INTERNAL: exhaustive impl differs from canonical_string! "
                    f"seed={seed} vars={num_vars} k={num_internal}: "
                    f"ours='{gt}', original='{gt_original}'"
                )

            # IDS gap analysis.
            num_v = len(dag.var_nodes())
            lb_val = remaining_lb(dag.node_count - num_v, dag.edge_count)
            gaps.append(len(gt) - lb_val)

            for strat in strategies:
                if strat == Strategy.EXHAUSTIVE:
                    continue
                total_counts[strat] += 1
                if results[strat].canonical_result != gt:
                    failure_counts[strat] += 1
                    results[strat].correct = False
                    log.error(
                        f"  {strat.value} FAILED seed={seed} vars={num_vars} k={num_internal}: "
                        f"expected '{gt}', got '{results[strat].canonical_result}'"
                    )

    # ---- Correctness summary ----
    log.info("")
    log.info("=" * 86)
    log.info("CORRECTNESS (vs exhaustive w*_D)")
    log.info("=" * 86)
    for strat in strategies:
        if strat == Strategy.EXHAUSTIVE:
            continue
        total = total_counts[strat]
        fails = failure_counts[strat]
        rate = fails / total * 100 if total > 0 else 0
        tag = "PASS" if fails == 0 else "FAIL"
        log.info(f"  {strat.value:<30} {fails:>3}/{total} failures ({rate:5.1f}%)  [{tag}]")

    # ---- IDS gap analysis ----
    log.info("")
    log.info("=" * 86)
    log.info("IDS GAP ANALYSIS (optimal_length - lower_bound)")
    log.info("=" * 86)
    if gaps:
        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        min_gap = min(gaps)
        from collections import Counter

        gap_hist = Counter(gaps)
        log.info(f"  Mean gap: {avg_gap:.1f}   Min: {min_gap}   Max: {max_gap}")
        log.info(f"  Histogram: {dict(sorted(gap_hist.items()))}")
        log.info("  (Gap = 0 means IDS finds solution on first iteration)")

    # ---- Efficiency summary ----
    log.info("")
    log.info("=" * 86)
    log.info("EFFICIENCY (averaged per config)")
    log.info("=" * 86)
    log.info("")

    header = (
        f"{'Vars':>4} {'k':>3} {'Strategy':<30} "
        f"{'Steps':>8} {'BndPrune':>9} {'DedPrune':>9} "
        f"{'IDSiter':>7} {'Time(ms)':>10} {'OK%':>5}"
    )
    log.info(header)
    log.info("-" * len(header))

    for (num_vars, num_internal), _ in sorted({(nv, ni): None for nv, ni, _ in agg}.items()):
        for strat in strategies:
            key = (num_vars, num_internal, strat)
            ml = agg[key]
            if not ml:
                continue
            N = len(ml)
            avg_steps = sum(m.step_count for m in ml) / N
            avg_bp = sum(m.bound_prunes for m in ml) / N
            avg_dp = sum(m.dedup_prunes for m in ml) / N
            avg_ids = sum(m.ids_iterations for m in ml) / N
            avg_time = sum(m.wall_time_s for m in ml) / N * 1000
            pct_ok = sum(1 for m in ml if m.correct) / N * 100
            log.info(
                f"{num_vars:>4} {num_internal:>3} {strat.value:<30} "
                f"{avg_steps:>8.1f} {avg_bp:>9.1f} {avg_dp:>9.1f} "
                f"{avg_ids:>7.1f} {avg_time:>10.2f} {pct_ok:>5.1f}"
            )
        log.info("")

    # ---- Speedup summary ----
    log.info("=" * 86)
    log.info("SPEEDUP vs EXHAUSTIVE (step count ratio)")
    log.info("=" * 86)
    log.info("")

    for (num_vars, num_internal), _ in sorted({(nv, ni): None for nv, ni, _ in agg}.items()):
        ex_steps = sum(m.step_count for m in agg[(num_vars, num_internal, Strategy.EXHAUSTIVE)])
        if ex_steps == 0:
            continue
        parts = [f"vars={num_vars}, k={num_internal:<2}"]
        for strat in strategies:
            if strat == Strategy.EXHAUSTIVE:
                continue
            s_steps = sum(m.step_count for m in agg[(num_vars, num_internal, strat)])
            ratio = ex_steps / max(1, s_steps)
            correct_mark = "" if failure_counts.get(strat, 0) == 0 else "*"
            parts.append(f"{strat.value}: {ratio:.2f}x{correct_mark}")
        log.info("  " + "  |  ".join(parts))

    log.info("")
    log.info("(* = has correctness failures; speedup numbers not meaningful)")


if __name__ == "__main__":
    run_experiment()
