"""Diagnose the DAGs on which fast_canonical_string raises.

Decisive question: is the failure caused by the greedy/pruned fast path being
incomplete, or is it structural (the DAG genuinely cannot be serialised)?

If the exhaustive canonical_string succeeds where fast_canonical_string fails,
the fast path is incomplete -- a correctness bug. If both fail, the DAG violates
a precondition of D2S itself.
"""

import random
from collections import deque

from isalsr.core.canonical import (
    canonical_string,
    fast_canonical_string,
    pruned_canonical_string,
)
from isalsr.core.string_to_dag import StringToDAG

MOV = list("NPnpCcW")
LAB = list("+*-/scelra^gik")
NV = 2


def reachable_from_node0(d) -> bool:
    seen, q = {0}, deque([0])
    while q:
        for v in d.out_neighbors_raw(q.popleft()):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) == d.node_count


def reachable_from_any_var(d, nv: int) -> bool:
    """The CDLL is pre-seeded with all nv variables, so D2S can walk to any of
    them. The real precondition is that every node is reachable from the *set*
    of variables, not from node 0 alone."""
    seen = set(range(nv))
    q = deque(range(nv))
    while q:
        for v in d.out_neighbors_raw(q.popleft()):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) == d.node_count


def main() -> None:
    rng = random.Random(31)
    failures = []
    n_ok = 0
    for _ in range(4000):
        s = "".join(
            rng.choice(["V", "v"]) + rng.choice(LAB) if rng.random() < 0.55 else rng.choice(MOV)
            for _ in range(rng.randint(6, 22))
        )
        try:
            d = StringToDAG(s, num_variables=NV).run()
        except Exception:
            continue
        try:
            fast_canonical_string(d, mode="wl_only")
            n_ok += 1
        except Exception as e:
            failures.append((s, d, e))

    print(f"succeeded: {n_ok}   failed: {len(failures)}\n")
    print(
        f"{'#':>2} {'k':>3} {'edges':>5} {'reach(x1)':>9} {'reach(vars)':>11} "
        f"{'exhaustive':>26} {'pruned':>26}"
    )
    print("-" * 92)
    for i, (s, d, e) in enumerate(failures):
        r0 = reachable_from_node0(d)
        rv = reachable_from_any_var(d, NV)

        def probe(fn):
            try:
                fn(d)
                return "SUCCEEDS"
            except Exception as ex:
                return type(ex).__name__

        exh = probe(lambda x: canonical_string(x))
        prn = probe(lambda x: pruned_canonical_string(x))
        print(
            f"{i:>2} {d.node_count - NV:>3} {d.edge_count:>5} {str(r0):>9} {str(rv):>11} "
            f"{exh:>26} {prn:>26}"
        )

    if failures:
        print("\n--- detail of first failure ---")
        s, d, e = failures[0]
        print("source string :", repr(s))
        print("error         :", e)
        print("nodes         :", [(i, d.node_label(i).name) for i in range(d.node_count)])
        print(
            "edges         :", [(u, v) for u in range(d.node_count) for v in d.out_neighbors_raw(u)]
        )
        seen = set(range(NV))
        q = deque(range(NV))
        while q:
            for v in d.out_neighbors_raw(q.popleft()):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        print("reachable from vars:", sorted(seen), "of", list(range(d.node_count)))
        print("UNREACHABLE nodes  :", sorted(set(range(d.node_count)) - seen))

    # How often is each predicate right about success?
    print("\n--- predicate quality over the whole sample ---")
    rng2 = random.Random(31)
    tot = fail_r0 = fail_rv = 0
    viol_r0 = viol_rv = 0
    for _ in range(4000):
        s = "".join(
            rng2.choice(["V", "v"]) + rng2.choice(LAB) if rng2.random() < 0.55 else rng2.choice(MOV)
            for _ in range(rng2.randint(6, 22))
        )
        try:
            d = StringToDAG(s, num_variables=NV).run()
        except Exception:
            continue
        tot += 1
        r0, rv = reachable_from_node0(d), reachable_from_any_var(d, NV)
        if not r0:
            viol_r0 += 1
        if not rv:
            viol_rv += 1
        try:
            fast_canonical_string(d, mode="wl_only")
        except Exception:
            if not r0:
                fail_r0 += 1
            if not rv:
                fail_rv += 1
    print(f"total DAGs                         : {tot}")
    print(f"violate reach-from-x1              : {viol_r0} ({100 * viol_r0 / tot:.1f}%)")
    print(f"violate reach-from-any-var         : {viol_rv} ({100 * viol_rv / tot:.1f}%)")
    print(f"failures explained by !reach(x1)   : {fail_r0}")
    print(f"failures explained by !reach(vars) : {fail_rv}")


if __name__ == "__main__":
    main()
