# T18 — canonical-string completeness: operand order is not encoded

| Field | Value |
|---|---|
| Type | **Correctness defect in the central claim.** Not a port bug, not an environment bug |
| Owner | **Ezequiel** (completeness is a theorem statement) + Mario (empirical half) |
| Found by | T17 Stage B check **B4**, 2026-08-03, job 1751918; reproduced locally at `2365c82` |
| Blocks | **Stage F sign-off** (EXECUTION-PLAN §4.6). Does **not** block Stage C or Stage D |
| Status | **OPEN — mechanism identified, fix not attempted** |

---

## 1. The claim that fails

The paper's central contribution is that the canonical string is a **complete**
invariant of the labeled DAG: two DAGs share a canonical string **iff** they are
isomorphic. The "only if" direction fails.

On the deterministic 10,000-DAG corpus of
`experiments/scripts/equivalence_gate.py --gate 3`, **5 DAGs (0.05 %)** satisfy

```
    fcs(D) == fcs(S2D(fcs(D)))        both land in the SAME dedup class
    D      ≇  S2D(fcs(D))             but they are NOT isomorphic
```

That is an **unsound merge**: `unique_canonical_dags` is under-counted, so
`ρ = total / unique` is **over-stated**.

## 2. What it is *not* — three explanations ruled out

| Hypothesis | Verdict | Evidence |
|---|---|---|
| A C++ port defect | **No** | `mismatches_engine_a == mismatches_engine_b == 5`; Python and C++ produce **byte-identical** canonical strings and fail the same five. Gate 1 (54,765 comparisons) and Gate 2 (10,000) are cross-engine clean |
| A Picasso build / stale `.so` | **No** | Reproduces on the workstation at `HEAD` with `build_hash = 298fc1188bf1b051`, and on Picasso with a freshly rebuilt `.so` (gcc 13.2.0) carrying the **same** build hash |
| The `is_isomorphic` precondition (CLAUDE.md invariant 9) | **No** | **No case has an in-degree-0 CONST**, so the raise-path is never touched. Three of the five (3687, 7403, 7771) also have **no VAR as an edge target**, so they sit inside `𝒞₂` where `normalize_const_creation` is equivariant. Node counts, edge counts and label multisets are equal in every case |

It is also **distinct from T15** (`d2s_canonicalisation_failures.md`), whose 6/4,000
cases *raise* `RuntimeError` and are counted by the T06 ledger. These raise
nothing and return a well-formed string.

## 3. The mechanism — operand order is not encoded

Worked on corpus index **7771** (k = 13, 1 variable), the simplest case.

`D` and `D' = S2D(fcs(D))` are isomorphic **as unordered labeled DAGs**: the map
`D'→D` given by `2→3, 4→7, 11→9, 12→10` (identity elsewhere) carries every edge
of `D'` onto an edge of `D`, and the label multisets are identical.

They differ in **`_input_order`** on node 8, which is a **`POW`**:

| | `ordered_inputs(8)` | `raw_in(8)` |
|---|---|---|
| `D` | `[3, 7, 9, 10]` | `[3, 7, 9, 10]` |
| `D'` | `[2, 12, 11, 4]` | `[2, 4, 11, 12]` |

Under the map, `D'`'s order is `[3, 10, 9, 7]` against `D`'s `[3, 7, 9, 10]`.

`is_isomorphic` is **right** to separate them: **invariant 8** makes operand order
load-bearing for `POW`/`SUB`/`DIV`, and the evaluator reads `ordered_inputs()`.
The first two ordered inputs are what `POW` evaluates, so

```
    D  computes  POW(n3, n7)
    D' computes  POW(n3, n10)
```

**Two different functions sharing one canonical string.** This is not a
bookkeeping wobble — it is semantic.

### Why it concentrates where it does

- All five failures carry `k ∈ {13, 15, 17, 18, 19}` — the top of the corpus range.
- Node 8 in case 7771 is a **`POW` with four in-edges**. `POW` is binary; the extra
  in-edges arrive from `C`/`c` instructions. The more in-edges a non-commutative
  node accumulates, the more orderings exist that the D2S traversal must
  reproduce and the canonical string must distinguish.

**Working hypothesis (untested):** `fast_canonical_string` orders a node's inputs
by an isomorphism-invariant key, which is correct for commutative operations and
**lossy for non-commutative ones**. The string then cannot distinguish two DAGs
that differ only by a permutation of a non-commutative node's operands.

## 4. Why it did not show up before

- T01's AC-3 gate 3 reported **0 mismatches on 117,798 evolved decomposed DAGs** —
  a *different corpus*. Evolved SR candidates apparently do not reach the
  structure that triggers this; randomly constructed DAGs at high `k` do.
- **But B3 measured `max k = 37` on live Bingo candidates**, well above the
  failing range, so "unreachable in practice" is a hypothesis, not a result.

## 5. What must be established next

| # | Task | Why |
|---|---|---|
| **T18.1** | Measure the rate on the **evolved** corpus (`equivalence_gate_evolved.py`) and on the Stage C candidate streams | This is the number that decides whether the paper must disclose a rate or a bound. Everything else is secondary |
| **T18.2** | Confirm or refute the operand-order hypothesis by testing whether every failing pair differs **only** in `_input_order` on a non-commutative node | If yes, the defect is precisely localised and probably fixable |
| **T18.3** | Test whether `mode="tuple_only"` and the exhaustive `canonical_string` reference fail the same five | T15's analogous check found the failure was *not* the pruning; do not assume that carries over |
| **T18.4** | Decide: fix the encoding, or state completeness as conditional on commutativity | A conditional theorem is defensible; an overclaimed one is not |
| **T18.5** | If the encoding is fixed, **every ρ in C2 must be recomputed** — the fix changes the equivalence classes | Sequencing matters: do this **before** C2, or accept re-analysis after |

## 6. Reproduce

```bash
python experiments/scripts/equivalence_gate.py --gate 3 \
    --backend-a python --backend-b cpp --out gate3.json
python -m experiments.scripts.t18_completeness_counterexamples \
    --gate-json gate3.json --out /tmp/t18.md --json-out /tmp/t18.json
```

Full per-case detail, with every edge list, label vector and canonical string:
**`docs/md_files/changes/t18_completeness_counterexamples.md`**.

## 7. The five cases

| corpus index | k | vars | engines agree | same dedup class | isomorphic |
|---|---|---|---|---|---|
| 2166 | 19 | 2 | yes | yes | **no** |
| 2256 | 15 | 2 | yes | yes | **no** |
| 3687 | 17 | 1 | yes | yes | **no** |
| 7403 | 18 | 1 | yes | yes | **no** |
| 7771 | 13 | 1 | yes | yes | **no** |

Source strings are in the report above and in `gate3.json` under
`gate3.mismatch_cases`.
