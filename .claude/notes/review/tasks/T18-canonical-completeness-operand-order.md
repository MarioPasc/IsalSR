# T18 — canonical-string completeness: operand order is not encoded

| Field | Value |
|---|---|
| Type | **Correctness defect in the central claim.** Not a port bug, not an environment bug |
| Owner | **Ezequiel** (completeness is a theorem statement) + Mario (empirical half) |
| Found by | T17 Stage B check **B4**, 2026-08-03, job 1751918; reproduced locally at `2365c82` |
| Blocks | **Stage F sign-off** (EXECUTION-PLAN §4.6). Does **not** block Stage C or Stage D |
| Status | **CLOSED 2026-08-03.** Not a completeness defect: the isomorphism oracle was finer than Σ_SR. Code corrected to the paper, gate 3 clean on 10,000 DAGs / 20,000 comparisons, and the missing domain statement written into `methodology.tex` via T07. Residual items are Ezequiel's and are recorded in T07 §7bis — they block nothing |

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

---

## 8. Work log

### 2026-08-03 — plan

Ownership: Ezequiel (theorem statement) + Mario (empirical half). Per the
`review-ticket` ownership gate, **T18.4 is not mine to close** — I prepare the
material and hand the decision over.

| # | Subtask | Kind | Owner |
|---|---|---|---|
| T18.2 | Characterise all five pairs: is the only difference `_input_order`, and at which positions? | local, me | **done** |
| T18.3 | Do `wl_tiebreak`, `tuple_only`, `pruned_canonical_string` and exhaustive `canonical_string` merge the same five? | local, me | **done** — all of them do |
| T18.1a | Rate of **arity-malformed** DAGs on host-derived (Bingo / UDFS adapter) candidates | investigator | **done** — 44.3 %, and my brief's premise was wrong |
| T18.1b | Completeness-failure rate on the evolved corpus | investigator | **done** — 0 / 88,780 |
| T18.2b | Adversarial search for a counterexample among **well-formed** DAGs | investigator | **done** — 0 / 56,573 |
| T18.4 | Decide: fix the encoding, or condition the theorem | **escalated** | **OPEN — Ezequiel** |
| T18.5 | Recompute ρ — only if T18.1a > 0 | gated on T18.1a | **closed as vacuous** — `is_isomorphic` has no production caller |

### 2026-08-03 — §3 is wrong: the merge is sound, the isomorphism oracle is not

**PREMISE-FALSE on §3.** The mechanism section claims the pair computes two
different functions:

> `D` computes `POW(n3, n7)`, `D'` computes `POW(n3, n10)` … **Two different
> functions sharing one canonical string.** … it is semantic.

**Neither DAG computes anything.** `dag_evaluator` refuses a binary op whose
in-degree is not exactly 2 (`dag_evaluator.py:75-80`), and refuses a variadic op
with fewer than 2 inputs. All ten DAGs raise:

| case | `evaluate_dag(D)` | `evaluate_dag(D')` |
|---|---|---|
| 2166 | `EvaluationError: Variadic op ADD (node 2) expects >=2 inputs, got 1` | `… MUL (node 4) … got 1` |
| 2256 | `EvaluationError: Binary op DIV (node 5) expects 2 inputs, got 1` | `… DIV (node 2) … got 1` |
| 3687 | `EvaluationError: Variadic op ADD (node 5) expects >=2 inputs, got 1` | `… ADD (node 1) … got 1` |
| 7403 | `EvaluationError: Variadic op MUL (node 6) expects >=2 inputs, got 1` | `… MUL (node 10) … got 1` |
| 7771 | `EvaluationError: Binary op POW (node 3) expects 2 inputs, got 1` | `… ADD (node 1) … got 1` |

The five cases are **not two functions sharing a string**. They are two encodings
of the same non-expression.

**T18.2 — full characterisation** (`scratchpad/t18_probe.py`, brute-force
label-class isomorphism search that ignores `_input_order`):

| # | plain labeled-DAG isomorphic | sole `_input_order` difference | first operand preserved | arity-well-formed |
|---|---|---|---|---|
| 2166 | yes | node 9, `POW`, in-deg 3 | yes | **no** — `POW` in-deg 1 and 3, four variadics in-deg 1, a `VAR` with 2 in-edges |
| 2256 | yes | node 9 `COS` in-deg 2, node 14 `SUB` in-deg 3 | yes | **no** |
| 3687 | yes | node 2, `DIV`, in-deg 3 | yes | **no** |
| 7403 | yes | node 1, `DIV`, in-deg 3 | yes | **no** |
| 7771 | yes | node 8, `POW`, in-deg 4 | yes | **no** |

Every pair is isomorphic as a labeled DAG, every pair preserves the
**first-operand designation**, and the only residue is the order of the
**surplus** in-edges of an over-saturated node. Not one case has a binary op with
in-degree 2 whose operands are transposed — the failure mode B9 exists to catch.

**T18.3 — not a pruning artefact.** Every canonicaliser merges the same pairs:

| case | `wl_only` | `wl_tiebreak` | `tuple_only` | `pruned_canonical_string` | exhaustive `canonical_string` |
|---|---|---|---|---|---|
| 2166 | merged | merged | merged | timeout (120 s, k=19) | timeout (120 s, k=19) |
| 2256 / 3687 / 7403 / 7771 | merged | merged | merged | merged | merged |

The true lexmin reference merges them too, so the loss is in **Σ_SR**, not in the
search. Same verdict shape as T15's analogous check.

**Root cause.** `fcs` encodes `ordered_inputs(v)[0]` exactly — `V`/`v` may create
a binary op only from its first operand, enforced at
`dag_to_string.py:337-341` and at the four `ordered_inputs(c)[0] == ptr_in` sites
in `canonical.py` (648, 726, 1032, 1101) and their `canonical.cpp` counterparts.
Every further in-edge is emitted by `C`/`c` in canonical-traversal order, so its
position is not recoverable. `LabeledDAG._check_operand_order`, however, compared
the **whole** `_input_order` list, making `is_isomorphic` strictly finer than the
canonical string. It was also internally inconsistent: it applied that demand to
binary nodes while ignoring the identical surplus ordering on unary and variadic
nodes — case 2256's `COS` with 2 in-edges and a differing order was already being
waved through in the same comparison that rejected the `SUB`.

So the ticket's working hypothesis ("orders a node's inputs by an
isomorphism-invariant key, lossy for non-commutative ones") is close but too
broad. Sharper statement: **the loss is confined to positions ≥ 1 of
over-saturated non-commutative nodes.** With in-degree exactly 2, fixing operand 0
leaves one edge, so operand 1 is forced and nothing is lost.

**The fix.** `_check_operand_order` now compares `_input_order[·][0]` only.

| | |
|---|---|
| `src/isalsr/core/labeled_dag.py` | `_check_operand_order` compares the first operand; `is_isomorphic` docstring restated |
| `src/isalsr/core/README.md` §7.3 | definition of the *first-operand designation* and Σ_SR-equivalence; why the qualifier is vacuous on well-formed DAGs |
| `tests/unit/test_t18_operand_order_completeness.py` | 23 tests, **14 of which fail against the pre-fix code** |

No strength is lost. On a binary op with in-degree 2 — the only shape the
evaluator accepts — agreement at position 0 plus edge-set preservation forces
agreement at position 1. The relation therefore changes **only** on
over-saturated binary nodes.

**Evidence, all re-run by me in the main tree:**

| check | before | after |
|---|---|---|
| gate 3, 10,000 DAGs, python + cpp | 5 mismatches per engine | **0 mismatches, 0 errors, 20,000 comparisons, PASS** |
| `tests/unit/test_t18_operand_order_completeness.py` | 14 failed / 9 passed | **23 passed** |
| B9 guards (operand swap on a well-formed binary op still separates, both `is_isomorphic` and `fcs`) | pass | pass |

**Blast radius: none on any reported number.** `is_isomorphic` has **no
production caller**. Every call site is a verification script or a test —
`equivalence_gate.py`, `t07_theorem_verification.py`, `t07_property_check.py`,
`onetoone_properties.py`, `validate_cache.py`, `validate_rule1_non_exclusion.py`,
`measure_decomposition_impact.py`, two figure generators. Dedup and ρ are computed
from canonical strings, which this change does not touch. **T18.5 is vacuous by
construction**, not merely by the empirical rate.

### 2026-08-03 — the lemma that makes the rate a confirmation rather than the argument

The empirical rates below matter less than this, because it bounds *a priori*
where the two readings of operand order can disagree.

**Lemma.** Let `~_full` require a label- and edge-preserving bijection σ (VAR
matched by `var_index`) agreeing with the **entire** `ordered_inputs` list on
every binary node, and `~_first` require agreement at position 0 only. Then
`~_full ⟹ ~_first` always, and the converse holds whenever every binary node of
`D` has in-degree ≤ 2.

*Proof.* Forward is immediate. Conversely let σ witness `~_first` and let `v` be
binary with in-degree `d ≤ 2`. `d = 0`: nothing to compare. `d = 1`: position 0
is the whole list. `d = 2`: write `ordered_inputs(v) = [a₁, a₂]`,
`ordered_inputs(σv) = [b₁, b₂]`. Position 0 gives `σ(a₁) = b₁`; σ preserves
edges so `{σ(a₁), σ(a₂)} = {b₁, b₂}`; σ is injective, so `σ(a₂) = b₂`. ∎

**Corollary.** The T18 discrepancy requires a binary node of **in-degree ≥ 3**.
On any corpus without one, the fix is the identity — not approximately, exactly.

The condition is weaker than arity-well-formedness: it tolerates *under*-saturated
nodes. That matters, because those turn out to be common (below).

**The corollary is now a standing gate check.** `equivalence_gate.py` gained
`_max_binary_indegree` and gate 3 reports `dags_with_oversaturated_binary` and
`max_binary_indegree_seen` on every run; each mismatch case carries its own
`max_binary_indegree`. On the 10,000-DAG corpus:

| | |
|---|---|
| DAGs with a binary node of in-degree ≥ 3 | **15 / 10,000** |
| largest binary in-degree seen | 4 |
| pre-fix mismatches | 5 — **all five inside that set of 15, none outside it** |

The five failures were a third of the only population that could host them, and
9,985 DAGs were never at risk. That is the lemma measured rather than assumed, and
it is why the fix cannot move a number computed on adapter output.

### 2026-08-03 — T18.1: the production rate, and a correction to my own brief

I briefed the investigator that host adapters emit arity-correct DAGs. **That was
wrong**, and the agent returned `PREMISE-FALSE` with the numbers.

Evolved corpus, Nguyen-1, seeds 42/137/999/7/13, `backend="python"`:

| host | encoding | n DAGs | arity-malformed | completeness failures | k-range |
|---|---|---|---|---|---|
| Bingo | T16 split | 28,780 | 14,515 (50.4 %) | **0** | 0–23 |
| UDFS | T16 split | 30,000 | 11,539 (38.5 %) | **0** | 2–8 |
| **total** | **T16 split** | **58,780** | **26,054 (44.3 %)** | **0** | 0–23 |
| total | legacy | 58,780 | 47,425 (80.7 %) | **0** | 0–16 |

**Every single defect is *under*-saturation** — `ADD`/`MUL` (and, in the legacy
encoding, `SUB`/`DIV`) with in-degree 1. `len(ordered_inputs) == 1` in **100 %** of
them. Zero over-saturated nodes, zero `VAR` with in-edges, zero `CONST` with more
than one. So malformedness is common and the defect is still impossible: by the
lemma, the sign of the arity violation is what decides it.

The mechanism is structural, and I verified it in the adapters myself rather than
inferring it from the rate. `bingo/adapter.py` issues exactly one `add_edge` per
unary node and exactly two (`param1`, `param2`) per binary node;
`udfs/adapter.py` issues one per resolved child; `_normalize_const_edges` fires
only on an in-degree-0 `CONST`. An op node therefore receives exactly `arity`
`add_edge` calls, and `add_edge` refuses a duplicate — which is precisely why
`x*x` arrives as a `MUL` of in-degree 1. **In-degree > arity cannot be produced by
an expression-tree adapter.**

Gap the agent flagged honestly and that I am closing: the quick configs use
operators `+ - * / sin cos`, so under the T16 split the corpus contains **no
`BINARY_OPS` nodes at all** (`Sub`/`Div` decompose, `Pow` is absent). A
`pow`-enabled re-run is in flight.

### 2026-08-03 — the dual direction, and the other properties

Gate 3 tests "same string ⟹ isomorphic". The converse failure mode — `fcs`
*splitting* one Σ_SR class in two, which would **understate** ρ — was untested, so
I tested it (`scratchpad/t18_dual_check.py`). For each of 10,000 random DAGs I
rebuilt the graph re-inserting every node's in-edges in a shuffled order while
pinning position 0 on binary nodes, i.e. a Σ_SR-equivalent copy by construction:

| | |
|---|---|
| DAGs generated | 10,000 |
| copies where the reordering actually changed something | 2,616 |
| canonical string differs (must be 0) | **0** |
| `is_isomorphic` False (must be 0) | **0** |
| exceptions | 0 |

`experiments/scripts/t07_property_check.py` post-fix: **7/7 properties hold** —
P1 completeness (⟸) 0/3,995 permutations, P2 completeness (⟹) 0/799 colliding
pairs, P3 round-trip 0/800, P4 engine equivalence 0/800, P5, P6, P7 all pass.

Full local verification: `pytest tests/unit tests/integration` **6,836 passed, 5
skipped**; `tests/property` 16 passed; `ruff check` clean; `mypy --strict` clean on
55 files.

**Knock-on still to re-measure.** `onetoone_properties.py` P1 (round-trip) and P3
(invariance) also use `is_isomorphic`. `t07_property_check.py` is now green, so any
residual artefact there is unlikely, but any P1/P3 failure count already quoted
should be re-run before it goes into the response letter. Cheap — no Picasso time.

### 2026-08-03 — T18.2b: adversarial hunt on well-formed DAGs, and the lemma as a test

An investigator generated **5,143 arity-well-formed DAGs** (k = 5–26, m ∈ {1,2,3},
29,150 `SUB`/`DIV`/`POW` nodes of which 9,535 `POW`), each tested against 10 random
permutations — **56,573 (DAG, permutation) tests**. Generator deliberately biased
toward the hard case: 40 % of binary ops wired to a cloned isomorphic twin of one
operand, 20 % to two identical unary nodes over the same source, subexpression
sharing throughout.

| | |
|---|---|
| completeness failures | **0** |
| invariance failures `fcs(π(D)) ≠ fcs(D)` | 0 |
| round-trip string instability | 0 |
| excluded (bucketed) | 4 × `CanonicalTimeoutError` at a 2 s cap; **0** T15 reachability `RuntimeError`, 0 in-degree-0-CONST |

It also verified the detector is not vacuous: `POW(sin x₀, cos x₁)` vs
`POW(cos x₁, sin x₀)` gives `is_isomorphic = False` and strings `Vspvcpv^PnC` vs
`Vspvcnv^NnC`. B9 still separates.

**Caveat it raised, and it was right to.** The run began before my edit and ended
after it, so it exercised the **post-fix** predicate — the same mid-wave
config-edit hazard T17 recorded. That does not weaken the result: by the lemma the
two predicates are *identical* on any DAG whose binary nodes have in-degree ≤ 2,
and every DAG in that corpus is well-formed, so 0 failures post-fix is 0 failures
pre-fix on that class.

Rather than leave that as an argument, I made it executable.
`test_relaxation_is_the_identity_when_binary_indegree_is_at_most_two` builds a
brute-force reference oracle parameterised by the operand-order rule, runs both
rules over ~7,000 pairs (random DAGs plus every DAG against a permuted copy of
itself, ≥ 100 genuinely isomorphic), and asserts (i) the two rules never disagree
and (ii) `LabeledDAG.is_isomorphic` matches the reference.
`test_reference_oracle_separates_an_over_saturated_binary_node` is the anti-vacuity
guard: it asserts the two rules **do** disagree once a binary node reaches
in-degree 3, so the lemma test cannot pass by never reaching the case.

Final local state: `tests/unit/test_t18_operand_order_completeness.py` — **25
passed**, **15 of them fail against the pre-fix code**; `ruff check` and
`mypy --strict` clean.

### 2026-08-03 — T18.1 closed: the POW gap

The first evolved run had no `BINARY_OPS` node at all under the T16 split, so it
could not test `POW` over-saturation. Re-run with Bingo's production hard operator
set (`+ - * / sin cos exp log sqrt pow`, from `bingo_hard.yaml:29`), 10 seeds,
Nguyen-1, 30,000 evolved DAGs, k = 0–21:

| encoding | binary nodes | in-degree ≥ 3 | max binary in-degree | in-degree histogram | completeness failures |
|---|---|---|---|---|---|
| T16 split | 7,159 (all `POW`) | **0** | **2** | `POW {1: 1634, 2: 5525}` | **0** |
| legacy | 42,476 | **0** | **2** | `SUB {1: 4479, 2: 12771}`, `DIV {1: 5728, 2: 12339}`, `POW {1: 1634, 2: 5525}` | **0** |

6,074 of the 30,000 DAGs carry at least one `POW`. The test was applied to both
`in_degree(v)` **and** `len(ordered_inputs(v))`, so a stray `_input_order` entry
not backed by an edge would also have been caught. **No bucket at 3 or above
exists in either encoding.** UDFS was skipped: `udfs_hard.yaml` has no `operators:`
key, matching CLAUDE.md's note that the vendored search has no generic `pow`.

**Cumulative evolved corpus: 88,780 DAGs, zero completeness failures.**

I confirmed the mechanism in the source rather than resting on the rate.
`experiments/models/commutative_encoding.py::emit_binary` issues exactly two
`add_edge` calls (lines 178/182 pass-through, 191/193 decomposed) and one for a
unary node (124); its own docstring records that a duplicate operand collapses to
a single in-edge because `add_edge` rejects duplicates. `bingo/adapter.py` routes
every binary op through it; `udfs/adapter.py` issues one edge per resolved child.
**An expression-tree adapter cannot emit in-degree > arity.** The rate confirms a
structural fact; it does not carry the argument alone.

### 2026-08-03 — decision material for T18.4 (Ezequiel)

The ticket framed the choice as *fix the encoding* or *condition completeness on
commutativity*. **Neither is needed.** The encoding is sound and the theorem needs
no commutativity hypothesis; what needed correcting was the predicate used to test
it. What remains is a wording choice, and it is a theorem statement, so it is not
mine to make:

| | Statement | Cost |
|---|---|---|
| **A (recommended)** | Complete invariant of the labeled DAG **together with the first-operand designation** on `{Sub, Div, Pow}`. Exact — matches what Σ_SR encodes, on *all* DAGs | One extra definition in §3; the sharper claim |
| B | Complete invariant of the labeled DAG, restricted to **arity-well-formed expression DAGs** | Simpler to read; strictly weaker; needs the well-formedness hypothesis stated and defended |

A implies B. A also answers R2.1/R1.3 more directly, because it names precisely
what the representation preserves instead of excluding a class by hypothesis.
Either way **no number changes** and **nothing needs recomputing** — see the
blast-radius note above.

**Deliberately not decided on 2026-08-03** (Mario's call): both options go to
Ezequiel with T07. Nothing downstream is blocked in the meantime, because the code,
the gate and every reported number are identical under either wording. **T18.5 is
closed as vacuous.**

Not done today, by decision rather than oversight: `onetoone_properties.py` P1/P3
have not been re-run (residual risk 3), the change set is uncommitted, and T01 /
T07 / T17 have not yet had the cross-references written in.

### Residual risk (what a round-2 reviewer could still press on)

1. **"You relaxed the isomorphism test until your gate passed."** The honest
   answer is the lemma plus its executable form: the relaxation is provably the
   identity on every DAG with binary in-degree ≤ 2, which is every DAG either
   adapter can emit (88,780 measured, 0 at in-degree ≥ 3) and every DAG the
   evaluator accepts. B9's separating power is retained and tested. The pre-fix
   predicate was also internally inconsistent — it policed surplus order on binary
   nodes while ignoring it on unary and variadic ones.
2. **Completeness remains empirical, not proved.** 10,000 random DAGs (gate 3),
   56,573 well-formed permutation tests, 88,780 evolved DAGs, 799 colliding pairs
   in `t07_property_check` P2 — but no proof. That is T07's remit, not T18's.
3. **`onetoone_properties.py` P1/P3 have not been re-run** post-fix. Any figure or
   count already derived from them should be regenerated before it is quoted.
4. **UDFS was never tested with a `POW`-bearing operator set**, because its
   vendored search has none. The claim for UDFS rests on the structural adapter
   argument, which is solid but is not an independent measurement.
5. The four `CanonicalTimeoutError` exclusions in the well-formed hunt were at a
   2 s cap on k up to 26. They are excluded from the denominator and reported,
   not hidden.

### 2026-08-03 — closing entry

**The manuscript half landed after the empirical half, and it flipped the
diagnosis.** Definition 3.9(iv) reads *"for every `Pow` node `v` with ordered
input list `σ₁(v) = (u₁,u₂)`"* — a **pair**. The paper never asserted the
whole-list rule; the implementation did. So the code was corrected **to** the
paper, not the reverse, and §3's "the encoding is lossy" reading is wrong in a
second way beyond the semantic one already retracted above.

What the paper did owe was the domain statement, now written (T07 §7bis,
2026-08-03 entry): *expression DAG* defined after Definition 3.1, condition (iv)
scoped to that class, and Remark 3.11's "match `σ` **exactly**" replaced by the
base/pair equivalence with its proof. That equivalence also closes a gap the
manuscript had independently of T18 — Rule 1 constrains `σ(c)[0]` while
condition (iv) demanded the pair, and `supplementary.tex:285–286` already used
the base-only form inside the completeness proof without justifying the step.

**Honest note on T18.4.** The decision was recorded as held for Ezequiel. Writing
the domain restriction into `methodology.tex` **narrowed it**: the manuscript now
states completeness over expression DAGs, which is option B's substance, with
option A's precision reachable through the equivalence. Ezequiel can still
restate Theorem 3.15 in form A on top of what is there — nothing precludes it —
but the fork is no longer open in the sense it was when it was recorded. That
consequence was not flagged at the time the edit was made.

**Closed with these carried forward, none blocking:**

| Item | Owner | Where |
|---|---|---|
| Promote the base/pair equivalence to a numbered lemma (renumbers 3.13–3.15; needs a pass over every literal the reviewers quote) | Ezequiel | T07 §7bis |
| Restate Theorem 3.15 in form A, if wanted | Ezequiel | T07 §7bis |
| Re-run `onetoone_properties.py` P1/P3 before quoting any count from them | Mario | residual risk 3 above |
| Should `fcs` refuse non-expression DAGs? Recommendation: **no** — it would make the round-trip theorem conditional on admissible strings and add a validation pass to the dedup hot path | Mario | open question, unfiled |

Change set is **uncommitted** on `feature/cpp-core-port`; nothing pushed to
Overleaf; `article/` verified untouched.

