# Canonical-string completeness: five counterexamples on the generated corpus

**Date**: 2026-08-03
**Found during**: T17 / EXECUTION-PLAN Stage B check **B4** (equivalence gate re-run on a Picasso
compute node), job `1751918`
**Reproduced**: identically on the workstation, `experiments/scripts/equivalence_gate.py --gate 3`
**Status**: **RESOLVED 2026-08-03 under ticket T18.** Not a completeness failure: the
five pairs differ only in the order of the *surplus* in-edges of an over-saturated
binary node, which Σ_SR does not encode and `dag_evaluator` refuses. The defect was
in `LabeledDAG.is_isomorphic` (it compared the whole `ordered_inputs` list rather
than the first operand). Gate 3 now reports 10,000 DAGs / 0 mismatches / 0 errors.
See `.claude/notes/review/tasks/T18-canonical-completeness-operand-order.md` §8 and
`docs/md_files/changes/t18_completeness_counterexamples.md`. The analysis below is
retained as the evidence trail; its §"two different functions" reading is superseded.

---

## 1. What was observed

Gate 3 of the equivalence harness checks **round-trip isomorphism**: for a DAG `D`,

```
    S2D( fcs(D) )  ≅  D
```

Over the gate's deterministic corpus of **10,000 randomly generated DAGs**, it fails on **5**
(0.05 %), with **0 errors** and **0 raises**.

The failure is *not* a C++/Python disagreement. Both engines report the same 5 failures and
produce byte-identical canonical strings for them:

| gate | corpus | comparisons | cross-engine mismatches | verdict |
|---|---|---|---|---|
| 1 (exhaustive `k=1..8`) | 6 DAGs × all permutations | 54,765 | **0** | PASS |
| 2 (generated) | 10,000 DAGs | 10,000 | **0** | PASS |
| 3 (round-trip) | 10,000 DAGs | 20,000 | — | **FAIL, 5 DAGs** |

So **B4's own question — "does the C++ engine agree with Python on this compiler and this CPU?" —
is answered YES.** What gate 3 exposes is a property of the canonicaliser itself, present equally
in the pure-Python implementation that produced campaign C1.

## 2. Why it matters: these are unsound merges, not cosmetic round-trip noise

The decisive test is not "is `S2D(fcs(D))` isomorphic to `D`" but "do `D` and `S2D(fcs(D))` land in
the **same dedup class**". For all five:

```
    fcs(D) == fcs(S2D(fcs(D)))       and       D  ≇  S2D(fcs(D))
```

| corpus index | `k` | vars | `fcs(D) == fcs(D')` | `D ≅ D'` | verdict |
|---|---|---|---|---|---|
| 2166 | 19 | 2 | True | **False** | unsound merge |
| 2256 | 15 | 2 | True | **False** | unsound merge |
| 3687 | 17 | 1 | True | **False** | unsound merge |
| 7403 | 18 | 1 | True | **False** | unsound merge |
| 7771 | 13 | 1 | True | **False** | unsound merge |

Two **non-isomorphic labeled DAGs share a canonical string.** That contradicts completeness of the
canonical-string invariant, which is the paper's central claim, and it biases ρ **upward**: two
distinct DAGs are counted as one canonical class, so `unique_canonical_dags` is under-counted and
`ρ = total_dags_explored / unique_canonical_dags` is over-stated.

## 3. The obvious artefact, ruled out

`is_isomorphic` assumes a precondition (CLAUDE.md invariant 9): it raises on an in-degree-0 CONST,
and `normalize_const_creation` is equivariant only on `𝒞₁ ∪ 𝒞₂` = {reachability holds} ∪ {no VAR is
an edge target}. A `False` returned outside that domain would be a measurement artefact, not a
finding. It is not the explanation:

| index | `k` | nodes/edges `D` | nodes/edges `D'` | in-deg-0 CONST | VAR as edge target | label multiset equal |
|---|---|---|---|---|---|---|
| 2166 | 19 | 21/23 | 21/23 | 0 | 1 | yes |
| 2256 | 15 | 17/19 | 17/19 | 0 | 1 | yes |
| 3687 | 17 | 18/19 | 18/19 | **0** | **0** | yes |
| 7403 | 18 | 19/20 | 19/20 | **0** | **0** | yes |
| 7771 | 13 | 14/16 | 14/16 | **0** | **0** | yes |

**No case has an in-degree-0 CONST**, so the raise-precondition is never touched. **Three of the
five (3687, 7403, 7771) also have no VAR as an edge target**, so they sit inside `𝒞₂` where
`normalize_const_creation` *is* equivariant. Those three are clean counterexamples: same node
count, same edge count, same label multiset, same canonical string, not isomorphic.

The corpus builder discards unreachable DAGs before testing (`33,222 / 43,222` discarded,
76.9 %), so the survivors are intended to satisfy the Round-Trip Fidelity reachability hypothesis.

## 4. This is a different failure class from T15

`docs/md_files/changes/d2s_canonicalisation_failures.md` (T15, 2026-07-27) documents 6 DAGs in
4,000 (0.15 %) on which `fast_canonical_string` **raises** `RuntimeError: no valid operation found`.
Those are *detected* failures — loud, countable, and already routed through the T06 fallback ledger.

The five here **raise nothing**. They return a well-formed canonical string that silently merges two
distinct DAGs. A loud failure and a silent wrong answer are not the same defect and should not be
reported as one.

## 5. What is *not* yet known

- **Rate on realistic candidates.** T01's AC-3 gate 3 reported **0 mismatches on 117,798 evolved
  decomposed DAGs** — a different corpus (evolved SR candidates, not random constructions). The
  failing `k` values here are 13–19, at the top of this corpus's range. Whether the class is
  reachable by Bingo/UDFS search at all is **open** and is the measurement that decides how much
  this matters.
- **The mechanism.** Not diagnosed. All five are large `k` with 1–2 variables.
- **Whether the greedy `wl_only` mode is implicated.** Not tested against
  `mode="tuple_only"` / the exhaustive reference. T15's analogous check found the failure was *not*
  the pruning; that must not be assumed to carry over.

## 6. Recommended handling

| | |
|---|---|
| **Stage C** | **Proceed.** The property is pre-existing, engine-independent, and present in the code that produced C1. Blocking Stage C would not fix it, and Stage C's C1.6/C1.13 give the first measurement of the class *on real candidate streams*, which is the number actually needed. |
| **Stage F / C2 sign-off** | **Blocking.** Do not sign off the campaign until the reachable-on-real-search rate is known and either bounded or disclosed. |
| **Owner** | **T07** (Ezequiel — completeness is a theorem statement, not an implementation detail) with the empirical half from T17/T01. |
| **Response letter** | If the rate on evolved candidates is non-zero, this is a disclosed limitation, not a bug to hide. A reviewer can run `--gate 3` in one command. |

## 7. Reproduce

```bash
python experiments/scripts/equivalence_gate.py --gate 3 \
    --backend-a python --backend-b cpp --out gate3.json
# then, for the soundness test that turns a round-trip failure into a merge:
#   D  = S2D(source_string);  D' = S2D(fcs(D))
#   assert fcs(D) == fcs(D')  and  not D.is_isomorphic(D')
```

The five source strings are in `gate3.json` under `gate3.mismatch_cases`.
