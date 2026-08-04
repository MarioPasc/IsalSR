# T07 — Complete the formal foundation of Theorem 3.15

| Field | Value |
|---|---|
| Reviewer comments closed | **R2.1**, **R1.3** (and R2's B3 = "Partially") |
| Type | Theory |
| Owner | **Ezequiel** (primary — proofs, `methodology.tex`) · **Mario** (empirical verification, tests) |
| Depends on | — (can start immediately; independent of compute) |
| Blocks | T03 phase 3, T06, T13 |
| Status | **Mario's half COMPLETE.** `𝒩` removed from the canonicaliser and `is_isomorphic` in both engines; 7/7 properties hold (`t07_property_check.py`); 4,605 unit + 474 property/integration tests pass; 21 old-contract tests rewritten in place, none suppressed. **AC-5, AC-6, AC-7, AC-8 met** — Theorems 3.13/3.14/3.15 stand **exactly as submitted**. **Proofs NOT STARTED (AC-1…AC-4, Ezequiel)** — see §7bis. The **T16 alphabet fork is RESOLVED (2026-07-30, Branch B — the code aligns to the paper)**, so `𝓝 = {Pow}` and Definition 3.9(iv), Rule 1's prose and Definition 3.2 all stand as submitted: **D-2 and D-3 evaporate and §7bis.3 is now empty of work.** What remains for Ezequiel is §7bis.2 alone: the five Lemma 3.14/A.2 gaps plus a sixth, the Theorem 3.13 domain mismatch. All of it is alphabet-independent and none of it is blocked. Remaining for Mario: AC-9 (page cost → T13) and AC-10 (§8). |
| Target | 2026-08-24 |
| Proof patch | **Ezequiel's AC-1…AC-4 patch, with all four blocking repairs applied, is INTEGRATED into `reviews/internal_copy_reviewed_article/` (2026-08-03), changes in blue, review notes stripped.** B0 numbering fixed (3.13/3.14/3.15 preserved; new material at 3.16–3.18), B1 `fcs:=fcs∘𝒩` removed everywhere, B2 CDLL-timing claim replaced by a topological induction, B3 non-emptiness restated over the `𝒫ₙ` sweep, B4 Theorem 3.13 widened. Both documents compile clean. Review: `T07-appendix/ezequiel_patch_review_2026-08-02.md`. **Nothing pushed to Overleaf; `article/` untouched.** |
| R2.1 answer | **WRITTEN** in `reviews/response_to_reviewers.tex`, with `fig_operand_order.pdf`. R1.3 gains the Lemma 3.17 / Corollary 3.18 paragraph. |
| Code vs theory | **22/22 checks pass on both engines** — `experiments/scripts/t07_theorem_verification.py`, output pinned at `reviews/t07_theorem_verification_output.txt`. No code defect found; the two defects found were in the *harness* (a vacuous-zero population, and a misused `permute_internal_nodes` signature), both fixed. |

---

## T16 impact — Branch B is IMPLEMENTED, and one scope clarification is owed to the completeness theorem (added 2026-07-30)

**Branch B is no longer a decision, it is code.** Both adapters emit the paper's
alphabet; verified on all 10 production configs over ~130,000 real candidates: zero
`Sub`/`Div` nodes, zero `-`/`/` characters in any canonical string, `Pow` the only
order-sensitive binary operation present. So `𝓝 = {Pow}` is now **true of the
artefacts**, not only of the definitions. **D-2 and D-3 are closed, not merely
projected to close**: Definition 3.9(iv), Rule 1's prose and Definition 3.2 all stand
exactly as submitted, and the three-node `Sub`/`Div` counterexample to Theorem
3.15's (⇐) direction **cannot be constructed from adapter output at all**.

Empirically re-established on the decomposed population (n=5000 per host, C++
engine): round-trip `D ≅ S2D(fcs(D), m)` **100 %**; completeness under
`permute_internal_nodes` (20 permutations × 100 DAGs) **100 %**; no false merges on
UDFS.

### One scope clarification for the completeness statement — NEW, and it is pre-existing, not caused by T16

`fast_canonical_string` returns the **empty string for every `k = 0` DAG regardless
of `m`**. Two variable-only DAGs with `m = 1` and `m = 2` are non-isomorphic labeled
DAGs and receive the *same* canonical string.

This is **correct behaviour, not a defect**: the canonical string encodes the
*instruction sequence*, and the number of input variables `m` is a parameter of the
initial state consumed by S2D, not part of the string. The invariant is therefore
complete **for labeled DAGs over a fixed `m`**, which is the only regime that arises
in production (`m` is fixed per problem).

**But the theorem should say so.** If Theorem 3.15 is stated as "`fcs` is a complete
invariant for labeled DAGs" without fixing `m`, the `k = 0` case is a literal
counterexample. Recommended: quantify over `𝒟_m`, labeled DAGs with exactly `m`
pre-inserted variable nodes, and note that `fcs` is complete within each `𝒟_m`.
**This is Ezequiel's call**, flagged here rather than edited into `methodology.tex`.

Verified identical (5/5/5 occurrences) across legacy and both decomposed encodings,
so **T16 introduces nothing here** — the measurement merely surfaced it.

Full write-up: `docs/md_files/changes/t16_commutative_decomposition.md`.
| Last updated | 2026-07-30 — §7bis restructured around the T16 alphabet fork. Earlier 2026-07-29: norm-removal study landed; AC-6 answered; D-1/D-2/D-3 recorded; T16 opened. (The 0.6 % round-trip finding flagged that day was **closed** the same day: it was the recursion artefact of §7.5, not a property of the representation — round-trip is 100 % in both arms on 10,551,609 real DAGs.) |

---

## 1. Why R2.1 and R1.3 are grouped

Two reviewers, two comments, one defect: **the formal statement of Theorem 3.15 is
incomplete.** One required preprocessing step is invoked but never defined (R1.3),
and the lemma that carries the theorem's completeness direction is asserted rather
than derived (R2.1). Both land in Ezequiel-owned files (`methodology.tex`,
`supplementary.tex` Appendix A). Answering them separately would produce two
patches to the same theorem; answering them together produces one corrected
theorem statement.

This is also the **only** scientific objection in the entire review round. It is
the sole reason Reviewer 2 answered B3 *"Partially"* rather than *"Yes"*. Reviewer 2
otherwise accepts the contribution — their B2 significance statement is positive.
The rating is recoverable and this ticket is what recovers it.

---

## 2. Verbatim comments

**R2.1:**
> 1. The extension version upgrades Conjecture 2.10/2.11 (left unproven in the
> preprint) to Theorem 3.13/3.15, but the proof of the key Lemma A.2 is unexpectedly
> terse and does not formally establish that κ-minimal candidate selection always
> yields valid D2S strings. A complete proof should be provided.

**R1.3:**
> 3) The pseudocode (Table 3, Appendix C) opens with a call to
> normalize_const_creation(D), defined only as "redirect all CONST creation edges to
> x1," that appears nowhere else in the paper?

---

## 3. Established facts — the exact gaps

### 3.1 Lemma A.2 (= Lemma 3.14, `lem:fcs_valid`)

Stated at `methodology.tex:1027–1033`; proved at `supplementary.tex:119–133`. The
proof is 14 lines and its argument is "same candidate pool ⇒ same string set",
asserted rather than derived. **Four specific gaps**, all verified against source:

1. **Termination is not addressed.** `𝒲(D)` (Definition 3.5, `methodology.tex:682–688`)
   is the set of strings *producible by the D2S procedure*. Membership requires that
   the FCS run terminates having placed every node and every edge. The proof does not
   establish termination.

2. **The candidate pools are demonstrably not identical.** Rule 1 *removes*
   candidates: a `Pow` node `c` with non-empty `σ(c)` and `σ(c)[0] ≠ u` is excluded.
   This directly contradicts the proof's own sentence *"exactly the same candidate
   pool as D2S"*. The proof's central claim is false as written.

3. **Rule 1's non-exclusion argument lives elsewhere and is informal.** It sits
   inside Definition 3.8 at `methodology.tex:762–766`, not in the lemma:
   *"Rule 1 does not exclude any valid insertion ordering: by the reachability
   precondition …, the base σ(c)[0] of every Pow node c is reachable from some
   variable, so D2S will have inserted σ(c)[0] into the CDLL before c becomes a
   candidate, and some displacement in 𝒫ₙ will place the acting pointer on σ(c)[0],
   making c eligible."* This must move into the lemma and be made rigorous.

4. **Definition 3.5 has no κ.** `𝒲(D)` is defined as free choice among uninserted
   out-neighbours at each branch point. Showing that a κ-minimal choice *is* one of
   those choices is exactly the step Reviewer 2 says is missing.

5. **The reachability hypothesis is never used explicitly** in the proof, although
   it is the lemma's stated hypothesis and gap 3 depends on it.

**Dependency**: Theorem 3.15's (⇒) completeness direction rests entirely on
Lemma A.2 (`supplementary.tex:202–210`). If A.2 is not established, the completeness
half of the headline theorem is not established. That is the mechanism by which one
terse lemma downgrades B3.

### 3.2 `normalize_const_creation`

Sole occurrence in the rendered manuscript: `supplementary.tex:398–399`, the first
line of the FCS pseudocode. No definition, no justification, no complexity note
anywhere. (It also appears at `methodology.tex:830–831` inside a `\begin{comment}`
block, so it does not render — see E6.)

> ⚠️ **The rest of this subsection describes the policy as submitted. That policy
> was replaced on 2026-07-27 (T15) because it was defective. Its justification
> below — "redirecting all such edges to `x₁` is always valid" — is false. Read
> §3.3 for what the code does now. Kept here because the *submitted* results were
> produced under it, so the response letter has to describe it accurately.**

**It is not cosmetic preprocessing; it is a precondition of the invariance claim.**
Without it, two isomorphic DAGs whose `Const` nodes were created from different
sources produce *different* canonical strings — i.e. Theorem 3.15 is false without
it, and Theorem 3.15 as stated does not mention it.

The implementation's own justification (`src/isalsr/core/labeled_dag.py:591–608`):
`Const` nodes are evaluation-neutral leaves that ignore in-edges, but D2S requires
every node to be reachable from a `Var` via outgoing edges, so V/v creates a
"creation edge" pointer → `Const`. The choice of creation source is semantically
irrelevant but produces different canonical strings. Redirecting all such edges to
`x₁` is always valid because `x₁` has no incoming edges, so no cycle can arise.

Call sites: `core/canonical.py:95, 146, 231` (guarded by `dag._has_const_nodes()`),
`core/labeled_dag.py:458`. `Const` (label `k`) is in `𝒯` (Table 1,
`methodology.tex:86`) and in Σ_SR (`computational_experiments.tex:64–66`), so
`Const` nodes are in scope for the benchmark suite — this is not a corner case.

### 3.3 The normalisation scheme as finally implemented — READ BEFORE §5.2

Source: `.claude/notes/review/tasks/T15-d2s-failure-modes.md` (fix applied
2026-07-27, both engines) and T06 (measured 2026-07-28). Three things changed, and
each of them moves a claim this ticket has to make.

**(a) The submitted policy was defective, in three separate ways.**

It relocated *every* `Const` creation edge onto node 0: delete the original, add
`0 → c`. But `LabeledDAG.add_edge` **returns `False` and adds nothing** when the
edge would close a cycle (`labeled_dag.py:248`), and the return value was
discarded. So whenever node 0 was already reachable *from* the `Const`, the
original edge was deleted and the replacement silently refused, leaving the
`Const` with **in-degree 0** — unreachable, and D2S cannot materialise a node no
pointer can reach. This is the mechanism behind the 6/4,000 `RuntimeError:
no valid operation found` failures. The graph never becomes cyclic; an edge is
simply lost. Two further defects followed from the same relocation:

1. **Completeness was false as implemented.** Take `{x₁, x₂, Sin, Const, Add}`
   with `x₁→Sin`, `Sin→Add`, `x₂→Add`, and the `Const` hanging off either `x₁` or
   `Sin`. Both satisfy the reachability hypothesis. They are **not** isomorphic
   under Definition (i)–(iv), because variable anchoring forces `φ(x₁)=x₁` and
   `x₁`'s out-degree differs. The old relocation mapped both to `VkVspv+NnC`.
   That is a direct counterexample to Theorem 3.15's (⇒) direction, in the
   submitted implementation.
2. **It was not evaluation-preserving**, despite the docstring in §3.2 asserting
   `eval(D) = eval(normalize(D))`. On `x → Cos → Const` it moved the output sink
   from `Const` to `Cos`, turning `1.0` into `cos(1.5) = 0.0707`.

**(b) What the code does now.** `normalize_const_creation` adds `x_i → c`
**only for `Const` nodes with in-degree 0**, choosing the lowest-indexed variable
that does not close a cycle, and **never removes an edge**. Applied identically in
Python (`labeled_dag.py`) and C++ (`native/src/labeled_dag.cpp`) to keep T01's
engine equivalence. Complexity O(|Const|) plus the cycle checks.

The property that matters for this ticket:

> **If `D` satisfies the reachability hypothesis then no `Const` has in-degree 0,
> so the repair is the identity on exactly the class the theorem quantifies over.**

Consequences, and they are favourable:

- **The Round-Trip Fidelity hypothesis is sufficient and needs no extra clause.**
  The earlier hand-over from T15 said otherwise; that was wrong and is retracted.
  The implementation was applying the normalisation *after* the hypothesis held
  and destroying it.
- **Theorem 3.15 now holds as written** on the hypothesis class, and does **not**
  need normalisation folded into its statement for that class. This partly
  supersedes §5.2's third bullet and AC-6 — see the corrections there.
- **`Const` provenance is ordinary structure, by decision.** Two DAGs whose
  `Const` nodes hang off different parents are *different labeled DAGs* and now
  get *different* canonical strings. The old policy merged them; that merge was
  the completeness bug, not a feature.

**(c) Evidence, so the proof can cite measurements rather than assertions.**

| Check | Result |
|---|---|
| Genuine canonicalisation failures, 10⁵ random S2D DAGs, old vs new policy | **48 → 0** |
| Repair drops an edge (10⁵ DAGs) | **0** |
| `repair` vs `no-normalisation` disagreements on hypothesis-satisfying DAGs | **0** — the identity property, confirmed at 10⁵ |
| Extra equivalence classes merged by the old policy | **169**, inflating ρ from 1.040 to 1.042 |
| Real Bingo DAGs, all three policies structurally identical | 12,176,790 / 12,176,790 |
| Real UDFS DAGs, all three policies structurally identical | 234,865 / 234,865 |

Scripts: `experiments/scripts/validate_const_repair_synthetic.py`,
`measure_const_normalization_arms.py`. Tests:
`tests/unit/test_const_normalization_repair.py` (30 tests, both engines).
**No submitted number is affected by either the defect or the fix.**

**(d) One open item T15 raised for you (its AC-3′).** The reachability hypothesis
is sufficient for *success* but says nothing about *cost*: 46 / 100,000
hypothesis-satisfying DAGs at k = 24–30 exceed a 10 s budget. Production allows
60 s and sees none. If the revised Lemma A.2 makes a complexity claim alongside
the correctness one, that is the gap it has to cover.

**Connection to R1.2 (T06) — measured, no longer conditional.** Constant terminals
enter host expressions as **leaves with no in-edge**, so they have no variable
ancestor and violate the reachability precondition on arrival. Measured
2026-07-28: **85.9 %** of Bingo candidates (132,746 / 154,568) and **100 %** of
UDFS candidates (3,890 / 3,890) violate it before normalisation; **0 %** after, at
every k. On S2D-produced DAGs the rate is 0 % before and after (0 / 14,841), which
is why the property-validation experiment never saw this. The two comments
therefore explain each other, and `normalize_const_creation` has the motivation
the paper lacks: it repairs a precondition that most real candidates violate.
See §7 for the question this raises about the theorem's scope.

---

## 4. Mandatory reading

- `.claude/notes/review/source/reviewer-2.md` — §R2.1 in full, and the B2/B3 discussion
- `.claude/notes/review/source/reviewer-1.md` — §R1.3
- `.claude/notes/review/source/verified-discrepancies.md` — D9, D10, E6
- `.claude/notes/review/source/manuscript-map.md` — the numbered-environment table;
  note that A.1/A.2/A.3 restate 3.13/3.14/3.15 verbatim, so each theorem exists
  twice under two numbers
- `.claude/notes/review/tasks/T06-reachability-failure-rate.md` — the empirical half
- `.claude/notes/review/tasks/T03-gray-code-integration.md` — **may require
  re-proving this same lemma; do not produce two divergent versions**
- `src/isalsr/core/README.md`
- `CLAUDE.md` (repo root) — Critical Invariants 5, 8, 9, 10
- Source files: `article/paper/methodology.tex` (Def 3.4, 3.5, 3.8, 3.9; Thm 3.13;
  Lem 3.14; Thm 3.15) and `article/supplementary/supplementary.tex` Appendix A

---

## 5. Work specification

### 5.1 Lemma 3.14 / A.2 — full proof
Address all five gaps in §3.1 explicitly. A proof that does not visibly discharge
gap 2 (the pools are not identical) will read as evasive to a reviewer who located
that contradiction themselves. Suggested structure, to be replaced by whatever
Ezequiel judges correct:

1. Restate `𝒲(D)` and make the branch-point choice function explicit.
2. Show the FCS candidate pool at each branch point is a **non-empty subset** of
   the D2S pool (Rule 1 restricts; Rule 2 selects within).
3. Show Rule 1's restriction is *non-excluding on orderings*: for every valid D2S
   insertion ordering there is an FCS-admissible ordering producing an isomorphic
   result — this is where the reachability hypothesis is consumed, and it must be
   consumed visibly.
4. Show non-emptiness ⇒ progress ⇒ termination with every node and edge placed.
5. Conclude membership in `𝒲(D)`, then invoke Theorem 3.13.

### 5.2 `normalize_const_creation` — define it

> **Rewritten 2026-07-28.** The previous version of this subsection specified the
> *submitted* policy. Writing that definition now would document behaviour the code
> no longer has. §3.3 has the current scheme.

- A numbered definition in `methodology.tex`, placed before Definition 3.8, stating
  the operation **as implemented**: supply a creation edge `x_i → c` to each `Const`
  node **of in-degree 0**, taking the lowest-indexed variable that does not close a
  cycle; never remove an edge. Include the complexity (O(|`Const`|) plus cycle
  checks) and the guard (applied only when the DAG has `Const` nodes).
- **Do not justify it as "redirect all `Const` creation edges to `x₁`".** That is
  the old policy, it is what `methodology.tex:830` still says, and it is false as
  a validity argument: the relocation is refused exactly when it would close a
  cycle, and the submitted code discarded that refusal and orphaned the node.
  `methodology.tex:830` needs the same correction.
- **Theorem 3.15 most likely does *not* need amending, contrary to what this
  ticket previously said.** Since the repair is the identity on every DAG
  satisfying the reachability hypothesis (§3.3b), the theorem holds as stated on
  its own hypothesis class. **Your call**, and it interacts with the scope question
  in §7 — the honest alternative is to state the theorem on `normalize(D)`, which
  is what the implementation actually canonicalises.
- Evaluation-preservation is now **true and worth a remark**: `eval(D) =
  eval(normalize(D))` holds because the repair only adds in-edges to `Const` nodes,
  which ignore them, and never changes out-degrees. Note that this claim was
  *false* under the submitted policy (§3.3a-2), so if the remark is added it should
  not be phrased as though it always held.

### 5.3 Structural cleanup that these fixes enable
Appendix A's proofs restate 3.13/3.14/3.15 verbatim, so every theorem exists twice
under two numbers — which is why R2 had to write "Theorem 3.13/3.15" and "Lemma A.2"
in the same sentence, and is adjacent to the broken cross-reference in R2.4.
Consider whether the restatements can be replaced by references. Coordinate with
T11 (cross-references) and T13 (page budget) — this is a page-saving opportunity in
a revision that must not grow.

### 5.4 Empirical verification (Mario)

> **Rewritten 2026-07-28.** The fourth bullet asked for a test whose premise is now
> false. Under the current scheme two DAGs whose `Const` nodes hang off different
> parents are **not isomorphic** and are *supposed* to receive different canonical
> strings (§3.3b). Building "two isomorphic DAGs with `Const` created from
> different sources" is not possible under Definition (i)–(iv), and the old
> policy's habit of merging them was the completeness bug, not the feature the
> response letter should advertise.

Every claim the proof makes that is checkable in code must be checked:

- Rule 1's non-exclusion property, tested on DAGs containing `Pow`. **Still open.**
- Termination, on the full 14,841-DAG corpus and on evolved DAGs. **Largely done**:
  0 canonicalisation failures on 14,841 (T06), on 10⁵ synthetic S2D DAGs (T15
  AC-3), on 12,176,790 real Bingo DAGs and on 234,865 real UDFS DAGs (T15 AC-4).
  What is *not* covered is termination as a theorem — see §3.3d on the k = 24–30
  timeouts.
- `eval(D) = eval(normalize(D))`, property-based test. **Now true and testable**;
  it was false under the submitted policy, and
  `tests/unit/test_output_node_and_adapters.py` previously enshrined the wrong
  behaviour. Two tests there were rewritten to assert preservation.
- **Replaces the old fourth bullet.** The concrete evidence that the step is a
  precondition rather than a convenience is now the T06 measurement, which is
  stronger: **85.9 % of Bingo and 100 % of UDFS candidates violate the reachability
  precondition on arrival, and 0 % do after normalisation.** That is what belongs
  in the response to R1.3. Instrumentation:
  `experiments/models/fallback_ledger.py`; tests `tests/unit/test_fallback_ledger.py`
  (42 tests).
- **New, and worth having**: the completeness counterexample against the *old*
  policy is already pinned as a regression test in
  `tests/unit/test_const_normalization_repair.py` (30 tests, both engines). It
  demonstrates that the submitted implementation merged two non-isomorphic DAGs,
  which is the honest framing for R2.1's (⇒) direction.

---

## 6. Acceptance criteria

- **AC-0.** §7 Work log filled in as the work proceeds.
- **AC-1.** Lemma 3.14/A.2 has a complete proof addressing all five §3.1 gaps by name.
- **AC-2.** The claim "exactly the same candidate pool as D2S" no longer appears, or
  is replaced by a correct statement about pool inclusion.
- **AC-3.** The reachability hypothesis is *used* in the proof, at an identified step.
- **AC-4.** Termination is established, not assumed.
- **AC-5.** `normalize_const_creation` is a numbered definition with justification,
  validity argument, and complexity.
- **AC-6.** Theorem 3.15's statement accounts for the normalisation.
  *(**Answered 2026-07-29, provisionally, by removing `𝒩` from the canonicaliser
  rather than by amending anything.** The two-arm study agrees on 130,311,281 real
  DAGs with per-DAG agreement 1.000000 and identical `n_unique`/ρ, and eliminates
  D-1 by construction. With `𝒩` out of the canonicalisation path `fcs` is a pure
  function of `D`, so Theorems 3.13/3.14/3.15 stand **exactly as submitted** — no
  restatement, no extra lemma, no scoping remark. Two earlier routes are closed:
  restating on `𝒩(D)` is **refuted** by the D-1 counterexample, and the
  adapter-image lemma is unnecessary if `𝒩` is not in the pipeline at all.
  **HOLD LIFTED 2026-07-29** — the v2 campaign closed every caveat: 30/30 tasks,
  per-DAG agreement **1.000000** on **10,551,609** real DAGs, identical `n_unique`
  and ρ in both arms, **0** equivariance failures in 1,681,544 samples, and
  round-trip **100 % in both arms under one comparator**. ρ = **1.7931** against
  the paper's independent **1.793**. The code change is cleared to proceed.)*
- **AC-7.** All §5.4 tests written and passing, including the pre/post-normalisation
  counterexample.
- **AC-8.** Consistent with T03's insertion point and with T06's statement of the
  precondition. *(T06 used the hypothesis exactly as stated in `methodology.tex:976`
  — "every non-variable node of D is reachable from some variable via directed
  paths" — measured on `D` as delivered by each source and again on `normalize(D)`.
  **T06's AC-7 is blocked on your confirmation of this statement**; if the wording
  changes, T06's rates are unaffected but its phrasing must follow.)*
- **AC-9.** Page cost reported to T13.
- **AC-10.** §8 filled.

---

## 7. Work log

### 2026-07-28 — Inbound from T06 and T15. No proof work done; this is Ezequiel's.

Two tickets landed material that changes what this one has to prove. Nothing here
is a proof, and §8 is deliberately untouched.

**From T15 (2026-07-27): the normalisation scheme changed.** Full detail in §3.3.
Short version: the submitted policy relocated every `Const` creation edge to node 0,
`add_edge` refuses a cycle-closing edge and returns `False`, and the return value was
discarded — so the `Const` lost its only in-edge and became unreachable. The
replacement adds `x_i → c` only for in-degree-0 `Const` nodes and never removes an
edge. **The consequence for you is favourable**: the repair is the identity on
every DAG satisfying the reachability hypothesis, so the hypothesis is sufficient,
the theorem holds as written on its class, and no extra clause is needed. An
earlier T15 hand-over said the opposite and claimed the normalisation could make
the graph cyclic; both statements were wrong and are retracted in §3.3.

Two defects of the submitted policy are worth stating in the response letter rather
than waiting to be found: it made **Theorem 3.15's (⇒) direction false** (concrete
counterexample in §3.3a-1, now a regression test), and it was **not
evaluation-preserving** despite a docstring claiming it was (§3.3a-2).

**From T06 (2026-07-28): the precondition is violated by most real input.**
Measured with new instrumentation, both engines, k-stratified:

| Population | N | Violated on arrival | After normalisation |
|---|---|---|---|
| Random S2D strings (the 14,841 corpus) | 14,841 | 0.0 % | 0.0 % |
| Synthetic random DAGs | 49,980 | 0.0 % | 0.0 % |
| Bingo search output | 154,568 | **85.9 %** | **0.0 %** |
| UDFS search output | 3,890 | **100.0 %** | **0.0 %** |

On Bingo the rate rises monotonically with k: 0 % at k = 0, 27.3 % at k = 1,
100 % for every k ≥ 24 — the profile predicted by "violated iff the expression
contains at least one constant terminal".

**The substantive question for Ezequiel, and it is the one thing T06 could not
resolve.** Verbatim from T06's close:

> AC-7 is the one criterion I cannot close. It requires T07's precondition
> statement; T07 is NOT STARTED and is Ezequiel's. I used T15's settled statement
> and left it open rather than claiming an agreement that hasn't happened. The
> substantive question for him: 85.9–100 % of real DAGs satisfy the hypothesis only
> after normalisation, so round-trip fidelity as implemented is a property of
> `normalize(D)`, not `D`.

Stated as a gap between quantifier and domain: Theorem 3.13 quantifies over DAGs
whose non-variable nodes are all reachable from some variable. T15 established that
the repair is the identity on exactly that class, which is what makes the canonical
string a complete invariant there. T06 then established that **85.9–100 % of the
DAGs the implementation actually canonicalises are outside that class when they
arrive** — a host constant is a leaf, and a leaf has no ancestor. They enter the
class only after `normalize_const_creation` runs. So the object the theorem
describes and the object the code canonicalises coincide only post-normalisation.

Three ways to close it, in ascending order of how much they change the paper:

1. **Add a lemma** that `normalize_const_creation` maps the host-adapter image into
   the hypothesis class, then apply the theorem unchanged. Smallest edit; keeps
   Theorem 3.13's statement intact and makes the pipeline argument explicit.
2. **State round-trip fidelity on `normalize(D)`** rather than `D`. Most faithful
   to the implementation; costs a restatement of 3.13, 3.14 and 3.15.
3. **Leave the theorem and scope it explicitly**, noting in the text that the
   canonicaliser's first step establishes the hypothesis for inputs that do not
   satisfy it. Cheapest in pages; weakest as a formal statement.

No recommendation is offered here — it is a decision about what the paper claims,
and §5.2's third bullet has been softened accordingly rather than pre-empting it.

**Also inherited, small and editorial.** `methodology.tex:830` still reads
`// redirect all Const creation edges to x_1`, which describes the old policy.
It should become "supply a creation edge to Const nodes that have none".

**What was corrected in this ticket, and why.** §3.2's validity argument
("redirecting all such edges to `x₁` is always valid because `x₁` has no incoming
edges") is false — that is precisely the step that failed — so §3.2 now carries a
warning banner and §3.3 states the current scheme. §5.2 was rewritten because it
instructed you to define the *old* policy. §5.4's fourth bullet was replaced
because it asked for a test whose premise no longer holds: two DAGs with `Const`
nodes on different parents are not isomorphic under the current scheme, so they
*should* get different canonical strings, and constructing the requested pair is
not possible. Nothing in §6's acceptance criteria was relaxed; AC-6 may simply turn
out to be discharged by argument rather than by an amendment.

### 2026-07-29 — Mario takes `normalize_const_creation`; a second non-equivariance defect found

**Scope for this session, agreed by email.** Ezequiel (2026-07-28, 17:26 and
later): *"haz tú lo de normalize_const_creation … Me voy a esperar a que Mario
actualice normalize_const_creation en el manuscrito"*. So AC-5, AC-6, AC-7 and
the R1.3 half of §8 are Mario's; AC-1…AC-4 (the Lemma 3.14/A.2 proof) stay
Ezequiel's and **no proof work was done here**.

**Plan.** (a) settle the AC-6 scope question; (b) numbered definition of
`normalize_const_creation` in `methodology.tex`; (c) correct
`methodology.tex:830` and the rendered twin in `supplementary.tex`; (d) close
§5.4's still-open Rule 1 non-exclusion test; (e) a figure motivating the step;
(f) fill §8.3's R1.3 response. Manuscript root confirmed as the live Overleaf
checkout `…/journal/69c1637a28a81fea2badda9a/` (remote
`git.overleaf.com/69c1637a28a81fea2badda9a`, pulled clean).
**`double_blind/paper/methodology.tex` is a byte-identical copy, not a symlink —
every manuscript edit must be mirrored.**

**Decisions taken by Mario, 2026-07-29.** AC-6 → restate on `𝒩(D)`
(subsequently refuted, see below). Numbering → insert the new definition before
Def 3.8 **and** unnumber Remark 3.12 ("Tightened automorphism bound") so the
count balances and **Theorem 3.15 keeps its number**, leaving Ezequiel's
in-flight work on 3.15 undisturbed. Figure → response letter + supplementary,
not the main paper (T13 page budget). Overleaf → pull, edit, commit, push.

**Three manuscript defects found, all in R2.1's area.** Full write-up:
`.claude/notes/review/tasks/T07-appendix/const_normalization_equivariance.md`.
In all three the **code is correct and the paper is wrong**, so no reported
number moves.

| # | Defect | Owner |
|---|---|---|
| D-1 | `𝒩` is not isomorphism-equivariant | Mario |
| D-2 | Def 3.9(iv) constrains operand order for `Pow` only, but `Sub`/`Div` are non-commutative and are in every production operator set — **(⇐) of Thm 3.15 false as stated** | **Ezequiel** |
| D-3 | Rule 1 prose says "`Pow` node"; implementation applies it to all of `BINARY_OPS = {Sub, Div, Pow}` | **Ezequiel** |

**D-2 is the most serious and needs no exotic input.** Three-node DAGs with
identical edge sets: `Sub` with `σ=(x₁,x₂)` gives `V-PnC`, with `σ=(x₂,x₁)`
gives `pv-nC`; likewise `Div` (`V/PnC` / `pv/nC`). For `Sub` and `Div` the
identity bijection satisfies Definition 3.9 (i)–(iii) and (iv) is vacuous
because the node is not `Pow` — so `D₁ ≅ D₂` per the definition while
`fcs_{D₁} ≠ fcs_{D₂}`. The code is right (`x₁−x₂ ≠ x₂−x₁`); condition (iv) must
range over `{Sub, Div, Pow}`. Note this would be correct under
`OperationSet.commutative()`, which is **not** the alphabet the experiments ran
(`experiments/configs/bingo_*.yaml` all list `-` and `/`).

**D-1 — `𝒩` is not isomorphism-equivariant.**

There exist `D₁ ≅ D₂` with `𝒩(D₁) ≇ 𝒩(D₂)`, hence `fcs_{D₁} ≠ fcs_{D₂}`.
`D₂ = permute_internal_nodes(D₁, [1,0,2,3])`, so the two are isomorphic by
construction. Both normalisations satisfy reachability. Strings:
`VkpvknvsncNVsNpppC` vs `pvkpvknvsncPVsNppC`.

Cause: `for c in sorted(const_nodes)` is a **node-index-ordered** iteration, and
node indices are exactly what isomorphism permutes; anchoring one orphan `Const`
creates paths that make another orphan's preferred anchor cycle-closing. Needs
≥2 orphan `Const` nodes **and** a `Var` with in-edges **and** a `Const ⇝ Var`
path.

Confirmed on the **C++ production engine** in all three modes, on the Python
engine, on `pruned_canonical_string`, and on `canonical_string` — the exhaustive
lexmin reference that prunes nothing. So it is neither a pruning artefact nor a
port artefact, and D2S is innocent: it is handed two structurally different
graphs and correctly returns two different strings. This is a **distinct defect
from the one T15 fixed**, and it is a property of the *repaired* policy.

**Consequences, and they bound it tightly.**

- **No submitted number is affected and no published claim is false.** Theorem
  3.15 as submitted is stated on DAGs satisfying reachability; the
  counterexample has orphan `Const` nodes and lies outside that hypothesis.
- Condition "a `Var` with in-edges" is **impossible by construction** on
  host-adapter output (`bingo/adapter.py:143-159`, `udfs/adapter.py:104-147`),
  so no candidate in any reported experiment can trigger it.
- **It does refute the AC-6 option chosen this morning.** Restating on `𝒩(D)`
  extends the domain to exactly the DAGs with orphan `Const` nodes, and the
  counterexample then sits *inside* the new hypothesis while violating the
  conclusion. Recorded so it is not re-proposed.
- `is_isomorphic` inherits the defect (it applies `𝒩` per Critical Invariant 9)
  and returns `False` on a pair built by the repo's own `permute_internal_nodes`.
  Latent; unreachable in production for the same adapter reason.

**The safe class**, and it is where Ezequiel's instinct is provably right.
`𝒩` is equivariant on `𝒞 = 𝒞₁ ∪ 𝒞₂` where `𝒞₁` = DAGs satisfying reachability
(`𝒩 = id` there) and `𝒞₂` = DAGs in which no `Var` node is an edge target. On
`𝒞₂` no orphan `Const` can reach a variable, so `x₁` never closes a cycle and
every orphan anchors to `x₁`: *"lo mejor es que x₁ sea el padre de todos ellos"*
is not a convention on that class, it is forced.

**On abandoning `𝒩` altogether (asked, answered).** Dropping it from
`canonical.py` is measurably a no-op on production: both adapters already repair
orphan `Const` nodes upstream (T06 `violated_post = 0`), and T15's `none` arm was
structurally identical to `repair` on 12,176,790 Bingo and 234,865 UDFS DAGs with
0 disagreements. Without it the canonicaliser **raises** on an orphan `Const`
rather than silently returning a wrong string — verified — which is the correct
failure mode. What it gives up is the safety net for non-adapter producers
(`from_sympy`, the precomputed atlas). **Reachability itself cannot be
abandoned**: a `Const` with in-degree 0 has no in-neighbour for a pointer to sit
on, so no `V`/`v` token can create it and D2S cannot emit that node at all.

**Σ_SR extension raised and assessed (2026-07-29).** Proposal: add an
instruction that creates a `Const` node *without* a creation edge, removing the
need for `𝒩` entirely. **Assessment: right design, wrong revision.** It would
make D-1 vanish by construction and weaken the reachability precondition to a
condition expression DAGs satisfy automatically — but it changes Σ_SR
(Definition 3.2, Table 1), forces re-proof of 3.13/3.14/3.15 against a changed
alphabet on top of the Lemma A.2 rewrite, introduces a *new* canonicalisation
sub-problem (the edgeless token's position in the string is unconstrained, which
is fresh string multiplicity), requires re-verifying the O(k!) claim, and buys
nothing measurable — on adapter output all constants already encode identically,
so it is a bijective re-encoding and ρ is unchanged. **Recommendation: two
sentences in Future Work**, draft in §6.4 of the appendix doc. This also
pre-empts a reviewer proposing it.

**Rule 1 non-exclusion (§5.4 bullet 1) — returned, partial, sent back.**
13,394 `Pow` DAGs, **0** Rule-1-attributable failures, round-trip 13,391/13,391
(100 %); 19 tests written and re-run green by me. **Not accepted**: only 24 of
13,394 DAGs (0.18 %) actually exercised a Rule 1 exclusion, which cannot carry a
universal non-exclusion claim. Redo requested with a generator targeting ≥ 2,000
exercising DAGs, `backend="cpp"` explicit (the 3 observed timeouts were
Python-backend backtracking cost), and `roundtrip_rate` replaced by two raw
integers. Iteration 1 of the 2-round budget.

**Open at the time of writing**: the measured equivariance-failure rate across
four populations (delegated, running); the Rule 1 redo; the three-arm
abandonment experiment (§7.3 of the appendix doc, designed, not launched); and
the AC-6 decision itself, now that the `𝒩(D)` restatement is refuted.

### 2026-07-29 — State of the algorithm: what IsalSR provably does and does not do

Written because the ticket had accumulated defects faster than it had
accumulated structure. **The distinction that organises all of them: only one
of the three is the algorithm misbehaving; the other two are the paper
mis-describing an algorithm that is doing the right thing.**

The pipeline as it actually runs:

```
host DAG (Bingo / UDFS)
   -> adapter          + _normalize_const_edges   <- the repair really happens HERE
   -> normalize_const_creation                    <- redundant second application
   -> D2S / FCS (greedy on the 1-WL sort key)
   -> canonical string
```

| Property | Status | Evidence / caveat |
|---|---|---|
| **Dedup soundness** — two different expressions are never merged | ✅ holds | All 6 bypass paths audited in code (T06). A failed candidate is evaluated and never enters `canonical_seen`. Only theoretical hole: a 64-bit hash collision, < 3×10⁻⁶ |
| **Round-trip fidelity** on DAGs satisfying reachability | ✅ holds | 0 failures on the 14,841 corpus, 10⁵ synthetic, 12,176,790 Bingo, 234,865 UDFS |
| **Completeness (⇒)** — same string ⇒ isomorphic | ✅ holds now | Was **false as submitted**; the pre-T15 policy merged non-isomorphic DAGs |
| **Completeness (⇐)** — isomorphic ⇒ same string | ⚠️ **two holes** | D-1 and D-2 below |
| **Evaluation preservation** `eval(D) = eval(𝒩(D))` | ✅ holds now | Was **false as submitted** (`x→Cos→Const` turned 1.0 into 0.0707) |
| **Reachability established before canonicalisation** | ✅ holds | 85.9 % (Bingo) / 100 % (UDFS) violated on arrival → **0 %** after. Established by the **adapter**, not by `𝒩` and not by the object the theorem quantifies over |
| **Termination** | ⚠️ empirical only | 0 failures on 12.4 M DAGs at the 60 s budget, but not proven. 46/100,000 synthetic DAGs at k = 24–30 exceed 10 s. T15's AC-3′, Ezequiel's |
| **Rule 1 excludes no valid ordering** | ✅ holds, now non-vacuously | Per op: POW 13,261 DAGs / 1,507 exercising; SUB 13,226 / 1,507; DIV 13,253 / 1,503. **0** failures, round-trip 100 % in all three. 39 tests |
| **`𝒩` is isomorphism-equivariant** | ❌ **fails**, but only outside `𝒞` | See the measurement below |
| **O(k!) reduction claim** | ✅ untouched | ρ identical across all three normalisation policies on 12.4 M real DAGs |

**The two holes in (⇐), and they are different in kind.**

- **D-1 is a genuine algorithm defect.** `𝒩` processes orphan `Const` nodes in
  node-index order, which is exactly what isomorphism permutes. Needs ≥ 2 orphan
  `Const` nodes **and** a `Var` with an in-edge **and** a `Const ⇝ Var` path —
  and no adapter ever points an edge into a variable, so it cannot occur in any
  reported experiment, and it lies outside Theorem 3.15's hypothesis anyway.
- **D-2 is the paper describing the wrong object.** The algorithm satisfies the
  property we meant; Definition 3.9(iv) states a weaker one. See T16 — the root
  cause is that the implemented alphabet is larger than the declared one.

**Equivariance measured, two independent seeds** (`experiments/scripts/
validate_const_equivariance.py`, K = 8 permutations per DAG via
`permute_internal_nodes`, C++ engine):

| Population | N DAGs | perm. tests | failures | in safe class `𝒞` |
|---|---|---|---|---|
| Random S2D, m = 2 | 400–500 | 3,200–4,000 | **0** | 100 % |
| Random S2D, m = 3 | 400–500 | 3,192–3,984 | **0** | 100 % |
| Const-free synthetic | 20 | 160 | 0 — **vacuous**, generator emits no `Const` | 100 % |
| **Bingo adapter output** | **15,530** | **123,240** | **0** | **100 % in `𝒞₂`** |
| Adversarial (built to violate `𝒞`) | 400–2,000 | 2,352–11,408 | **18.1 – 20.1 %** | **0 %** |

Cross-tabulation, both seeds: **failures inside `𝒞` = 0; every failure is
outside `𝒞`.** Bingo's `frac_in_c2 = 1.0000` — every candidate has all variables
as pure sources, which is the structural reason the defect is unreachable in
production. UDFS is 100 % in `𝒞₂` by the same adapter argument
(`udfs/adapter.py:108-151` adds edges only into operator nodes); stated as a code
argument, not measured.

**Honest limit, recorded rather than buried**: `𝒞` is **sufficient, not tight**.
There are DAGs outside `𝒞` that do not fail — the interference only fires when
the two `Const` chains target the two lowest-indexed variables. So `𝒞` is a
usable theorem hypothesis but it does **not** characterise the failure set, and
must not be presented as if it did.

Full suite after both new test files: **4,541 passed, 5 skipped** (4,478 + 24
equivariance + 39 Rule 1), `ruff` and `mypy --strict` clean. All re-run by me.

**How (⇐) gets repaired without losing anything else.** Two independent moves,
neither of which costs another property:

1. **Remove `𝒩` from the canonicalisation path.** `fcs` becomes a pure function
   of the DAG and (⇐) follows from the isomorphism-invariance of the sort key κ.
   Round-trip is unaffected (stated on reachability-satisfying DAGs, where `𝒩`
   was already the identity); reachability is still established by the adapters;
   evaluation preservation becomes trivial; soundness and (⇒) are untouched; and
   `is_isomorphic` becomes a true isomorphism test. Being validated by the
   two-arm study (T07-appendix §7.3).
2. **Resolve the alphabet mismatch** — T16.

### 2026-07-29 — The norm-removal study: setup, numbers, and the AC-6 answer

**Decision taken by Mario, 2026-07-29 (provisional, pending caveat 2 below):
remove `𝒩` from the canonicalisation path.** This recovers D-1 by construction
and discharges AC-6 **without amending any theorem** — Theorems 3.13/3.14/3.15
stand exactly as submitted, because `fcs` becomes a pure function of `D` and the
object the theorem describes finally coincides with the object the code
canonicalises. Ezequiel's in-flight work on 3.15 is unaffected under this branch,
which is what the R1.3 `\changeref` already promised him.

#### Experimental setup, so it is reproducible

| Item | Value |
|---|---|
| Arms | `keep` = production (`𝒩` applied inside canonicalisation) · `drop` = `_native.testing.fast_canonical_string_raw` (no `𝒩`) |
| Engine | **C++** (`backend="cpp"`) throughout, on Picasso |
| Adapters | repair `Const` upstream in **both** arms — this is production reality |
| Equivariance oracle | `permute_internal_nodes`, K = 8 permutations per sampled DAG, 1-in-50 sampling |
| Canonicalisation budget | 60 s (live populations), 10 s (synthetic) |
| Jobs | `1679357` synthetic (20 tasks) · `1679404` bingo (15) · `1679413` udfs (15) · `1679358` adversarial (1) |
| Outcome | **50/50 COMPLETED, 0 FAILED, 0 OOM, 51/51 result files** |
| Scripts | `experiments/scripts/t07_norm_removal_{study,aggregate,figures}.py`; `slurm/t07_norm_removal_launch.sh` + `slurm/workers/t07_norm_removal_slurm.sh` |
| Results | `picasso:~/execs/isalsr/t07_norm_removal/` (self-contained), `summary.json` |
| Problems | 5 × 3 seeds: Nguyen-5, I.6.20a, Pagie-1, Keijzer-11, R1 |

#### The numbers (re-read by Mario from `summary.json`, not taken from the agent)

| Population | DAGs / arm | per-DAG agreement | `n_unique` keep = drop | ρ equal | equivariance fails keep / drop |
|---|---|---|---|---|---|
| **bingo** | **93,910,005** | **1.000000** | 325,486 = 325,486 | yes | **0 / 0** (14,809,528 samples) |
| **udfs** | **36,401,276** | **1.000000** | 111,553 = 111,553 | yes | **0 / 0** (11,657,120 samples) |
| synthetic | 500,000 | **1.000000** | 491,490 vs 491,491 | yes | 0 / 0 (79,864 samples) |
| adversarial | 12 | n/a (`n_both_ok` = 0) | — | — | **53 / 0** (96 samples) |

> 🔴 **RETRACTED 2026-07-29 (same day): the bingo and udfs rows above are
> contaminated and must not be quoted.** The as-run code's `_round_trip_keep`
> calls `_score_keep`, which re-imports `isalsr.core.canonical.
> fast_canonical_string` *at call time*. During a live search that name is
> monkey-patched to the `TwoArmRecorder`, and **no re-entrancy guard exists in
> either version**. So every round-trip check on a bingo/udfs DAG re-entered the
> recorder and scored the S2D-decoded string as though it were a fresh adapter
> DAG. `n_total`, `n_unique`, ρ, the equivariance sample counts and the
> round-trip rates for **bingo and udfs** are therefore all inflated by recursive
> re-entry, and the population is not "adapter output" — an unknown fraction are
> S2D re-decodes. Verified by diffing the Picasso as-run copy against local
> (`asrun_study.py:371`).
>
> **This also supersedes caveat 1 below**: ρ ≈ 288 was attributed to pooling.
> Pooling is real but recursion inflation is at least as large, and the two
> cannot be separated from the existing output.
>
> **Synthetic and adversarial are unaffected and stand**: neither monkey-patches
> `fast_canonical_string`, so neither can recurse. Those are the rows the AC-6
> conclusion now rests on.
>
> **FIXED and re-running (2026-07-29, later).** `keep_fn` threaded through both
> round-trip helpers, plus a re-entrancy guard on `TwoArmRecorder.__call__` and
> an `n_reentrant_calls` field so the property is provable from the output.
> Measured `n_reentrant_calls = 0`, so the threading is what fixed it and the
> guard is a backstop. Corrected local numbers, both arms, C++ engine:
>
> | Problem | DAGs | Bingo's own counter | study n_total / n_unique | ρ | agreement | round-trip |
> |---|---|---|---|---|---|---|
> | Nguyen-5 | 10,520 | 10520 / 5936 | 10,520 / 5,936 | **1.772** | 1.000000 | 10,520/10,520 |
> | I.6.20a | 351,200 | 351200 / 197739 | 351,200 / 197,739 | **1.776** | 1.000000 | **351,200/351,200** |
>
> The study's counters now reproduce **Bingo's own independent dedup counters
> exactly**, and ρ lands on the production 1.793. Under recursion they could not
> have matched — that cross-check is what confirms the fix.
> **Caveats 2 and 3 are both closed**: round-trip uses `fast_canonical_string` in
> both arms and is 100 % on 351,200 real DAGs each, so the 0.6 % failure rate was
> the same artefact and not a property of the representation.
> **v2 CAMPAIGN COMPLETE — 30/30 tasks, 0 FAILED, 0 OOM.** Jobs `1679689` /
> `1679698`, results in `~/execs/isalsr/t07_norm_removal_v2`, 51/51 files.
>
> | | bingo keep | bingo drop | udfs keep | udfs drop |
> |---|---|---|---|---|
> | `n_total` | 10,286,517 | 10,286,517 | 265,092 | 265,092 |
> | `n_unique` | **5,736,798** | **5,736,798** | **141,009** | **141,009** |
> | ρ | **1.7931** | **1.7931** | 1.8800 | 1.8800 |
> | raised / timeout | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
> | equivariance fails | **0** / 1,639,064 | **0** / 1,639,064 | **0** / 42,480 | **0** / 42,480 |
> | round-trip | **100 %** | **100 %** | **100 %** | **100 %** |
> | comparator | `fast_canonical_string` (both arms) | | | |
> | per-DAG agreement | **1.000000** | | **1.000000** | |
>
> **ρ = 1.7931 against the paper's independently produced 1.793** — four
> significant figures from a different code path, where the contaminated run gave
> 288.5. That is the external cross-check the first campaign lacked.
>
> **All three caveats are closed.** Removing `𝒩` from the canonicaliser is a
> measured no-op on 10,551,609 real DAGs; round-trip holds at 100 % in both arms
> under one comparator; and the 0.6 % failure rate was the recursion artefact,
> not a property of the representation.
>
> One precision: the 21 carried-over `synthetic`/`adversarial` files predate the
> fix. They cannot have recursed (no monkey-patch) and their agreement and
> equivariance figures stand, but their **drop-arm** round-trip rows still use
> `structural_key` and remain comparator artefacts. No claim depends on them.
>
> **No submitted TPAMI number, and no T15 or T06 number, is affected** — verified,
> not assumed: re-entry requires an installed monkey-patch, and
> `grep -rn` over `experiments/models/` finds none in production, while T15's
> script patches but never round-trips and T06's ledger patches nothing.

**On the clean populations (500,012 DAGs) the arms agree on every one.** Neither
raised nor timed out on any real DAG in the contaminated runs either, which is
still meaningful for the raise/timeout axis. On the adversarial population `keep` gives **53 silent
wrong answers in 96 samples (55.2 %)** and refuses nothing; `drop` gives 0 and
refuses all 12 loudly (`keep` silent-wrong 7, `drop` loud-refusal 12). That pair
of columns is the whole argument for removal in one line: keeping `𝒩` means being
quietly wrong on inputs outside `𝒞`; dropping it means refusing loudly on inputs
that genuinely have no encoding in Σ_SR.

Synthetic is the only non-adversarial place the arms differ in *behaviour*: at
k ≥ 21, `keep` records 359 **timeouts** where `drop` records 358 **raises**, same
k-distribution. `𝒩` makes an orphan-`Const` DAG canonicalisable in principle but
expensive, so it burns the budget; without `𝒩` it is refused immediately. `drop`
ends with one *more* success (499,642 vs 499,641) and an identical ρ.

#### Three caveats that must travel with these numbers

1. **The ρ values here are NOT the paper's reduction factor.** ρ ≈ 288 (bingo),
   326–343 (udfs) against the paper's 1.793. `n_total` is summed over 15 tasks
   while `n_unique` is the **union** of their distinct sets, and independent seeds
   rediscover the same structures, so the union barely grows. A pooled-union
   artefact; the JSON honestly names the field `rho_lower_bound`. It does not
   affect the conclusion (both arms get the identical value) but **must never be
   quoted as a reduction factor.**
2. **The round-trip axis is not yet validly compared.** `keep` used
   `round_trip_comparator = fast_canonical_string`, `drop` used `structural_key`.
   The synthetic `drop` figure `round_trip_rate = 0.0187` is a **comparator
   artefact, not a failure**. Round-trip fidelity under `drop` is **unverified**.
   **The AC-6 code change is held until this is closed** — re-measurement with a
   common comparator is in flight.
3. **New finding, not caused by `𝒩`, potentially more important than anything
   else here.** In the `keep` arm — the production configuration — round-trip
   succeeds on only **99.39 %** of real DAGs: 93,333,870 / 93,910,005 (bingo) and
   36,179,317 / 36,401,276 (udfs), i.e. ~576,000 and ~222,000 failures.
   Synthetic `keep` is a clean 1.0000, and the rate is ~0.6 % in *both* arms, so
   `𝒩` is not the cause. **Round-trip fidelity is Theorem 3.13**, so a 0.6 %
   failure rate on real search output is a finding in its own right. Candidate
   explanation: the `Sub`/`Div` alphabet mismatch of T16 — 61.1 % of Bingo
   candidates carry those labels and the paper's `𝓛` does not contain them. Note
   61.1 % containing vs 0.6 % failing rules out a simple "contains Sub/Div ⇒
   fails" rule; the finer condition is under investigation. **Flagged, not
   concluded.**

Full write-up, mechanism, safe class and options:
`.claude/notes/review/tasks/T07-appendix/const_normalization_equivariance.md` §7.3.

#### Operational note for whoever reruns this

`--export` on SLURM separates **variables** with commas, so a comma-separated
value is silently truncated at the first comma and the remainder is parsed as
malformed export entries. Job `1679359` lost 4 of its 5 problems that way and
failed every task with `PROBLEM_IDX >= 1` in under 3 seconds. The problem list now
uses `|` as separator in both launcher and worker. The dry-run output shows this
plainly — read it.

### 2026-07-29 — `𝒩` REMOVED from the canonicaliser. Mario's half of T07 is closed.

**The change.** CONST normalisation is no longer applied during canonicalisation
or isomorphism testing. Removed from `canonical.py` (3 sites), `labeled_dag.py`
`is_isomorphic`, and `native/src/canonical.cpp`. `LabeledDAG.
normalize_const_creation` **still exists, unchanged** — it is now purely a
*producer-side* repair (the host adapters call their own `_normalize_const_edges`);
the canonicaliser simply never invokes it. `output_node`'s docstring was also
corrected: it still described the pre-T15 relocation policy.

**Verification — property table, `experiments/scripts/t07_property_check.py`**
(one script, <1 min, both engines, re-run after any canonicaliser change):

| | Property | Result |
|---|---|---|
| P7 | S2D output satisfies the precondition by construction | 800/800 |
| P1 | Completeness (⇐): isomorphic ⇒ same string | **0 failures / 3,995 permutations** |
| P4 | Engine equivalence, cpp == python | 0 disagreements / 800 |
| P2 | Completeness (⇒): same string ⇒ isomorphic | 0 failures / **799 colliding pairs** |
| P3 | Round-trip `D ≅ S2D(fcs(D), m)` | 0 failures / 800 |
| P5 | `eval(D) = eval(𝒩(D))` | 0 failures / 8 CONST-bearing DAGs |
| P6 | `𝒩` absent from the canonicaliser | orphan refused on both engines; `𝒩(D)` canonicalises on both |

P2 and P5 were **vacuous on the first run** (0 colliding pairs, 0 evaluable
CONST DAGs) and were fixed before being reported: P2 now seeds the pool with
deliberate isomorphic copies, P5 builds well-formed CONST DAGs explicitly
including the `x → Cos → Const` shape the pre-T15 policy broke.

**Test suite**: **4,605 unit passed / 5 skipped**, **474 property+integration
passed**, ruff and `mypy --strict` clean. 21 tests encoding the old contract were
**rewritten in place, none deleted, skipped or xfail-ed** — verified by grepping
the three files for suppressions (0 hits) and by the test count rising 4,584 →
4,605. The non-equivariance of `𝒩` is **preserved as a method-level regression
pin**, so the knowledge survives and nobody reintroduces `𝒩` into the
canonicalisation path.

**Behaviour change, stated plainly.** A DAG with an in-degree-0 `Const` is now
**refused loudly** by the canonicaliser instead of being silently repaired into a
different graph. That is correct: such a DAG has no encoding in Σ_SR at all.
Producers establish the precondition; the canonicaliser assumes it.

**Environment trap found and documented in `CLAUDE.md`.** `pip install -e .
--no-build-isolation` fails silently (`BackendUnavailable: scikit_build_core`)
and leaves the stale `.so` loaded; `pip … | tail` reports `tail`'s exit status,
not pip's. I reported two "successful" rebuilds that had not happened. It was
caught only because the D-1 check ran against **both** backends and they
disagreed — `cpp` reproducing the old strings while `python` raised. Use
`--force-reinstall --no-deps` and always verify the `.so` mtime.

### 2026-08-02 — Ezequiel's proof patch reviewed. Not integrated; four blocking items.

Ezequiel sent a Claude-generated patch for AC-1…AC-4 (`results/T07_Ezequiel/`:
`methodology_T07.tex`, `supplementary_T07.tex`, `response_to_reviewers_T07.tex`,
plus two PDFs) and asked whether it is coherent with the rest of the revision
before he integrates it into Overleaf. Read-only `git fetch` confirms he has
**not** pushed it: the only new remote commit, `8807b45` (2026-07-31), has an
empty diff against `79157c4`.

**Verdict: do not integrate as-is.** The five-step proof skeleton is right and it
discharges gap 2 visibly — the false *"exactly the same candidate pool as D2S"*
sentence is named and replaced by the inclusion `𝒞_j ⊆ 𝒟_j`, which is the core of
what R2.1 asked for. Four items block, four should be fixed. Full write-up with
counterexamples and proposed repairs:
`T07-appendix/ezequiel_patch_review_2026-08-02.md`.

| # | Item | Kind |
|---|---|---|
| B0 | Inserting the new definition before Def 3.8 renumbers everything after it — his own PDF has Thm **3.14**/Lem **3.15**/Thm **3.16** where the reviewers wrote 3.13/3.14/3.15. 8 literal `3.13` + 9 literal `3.15` in the response letter, no `\ref`s | **mechanical, high blast radius** |
| B1 | Patch defines `fcs_D := fcs_{𝒩(D)}` in the Fast Canonical String definition, Thm 3.15/A.3, and keeps `D ← 𝒩(D)` in Table 3 | **coherence** |
| B2 | Step 3's claim *"`σ(c)[0]` is already in the CDLL by the time `c` first becomes a candidate"* is false | **proof defect** |
| B3 | *"`𝒞_j ≠ ∅` whenever an out-neighbour of the tentative position is uninserted"* is false; Step 4 cites a statement Step 3 never proved | **proof defect** |
| B4 | The sixth gap (Thm 3.13 stated on `D = S2D(w,m)` and concluding about `D2S(D,x₁)`) is untouched, so Lemma A.2's final inference is unlicensed | **proof gap** |

**B1 is the one that matters most for coordination, and it is a missed hand-over,
not a disagreement.** AC-6 was discharged on 2026-07-29 by *removing* `𝒩` from the
canonicaliser, and restating on `𝒩(D)` was refuted the same day by D-1. The patch
re-proposes exactly the refuted route. Re-verified today, both engines:
`fast_canonical_string` raises `RuntimeError: no valid operation found` on an
in-degree-0 `Const`; `canonical.py` mentions `normalize_const_creation` only in the
three comments explaining why it is not called. The patch would therefore ship a
manuscript that describes code we do not have — the exact failure mode Ezequiel
made his own deciding argument in T16.

It also contradicts the R1.3 answer sitting in the same letter unchanged
(`:493–497`: `𝒩` acts "at the interface between a host solver and the
representation"; "Table 3 … now states the precondition explicitly in place of the
undefined call"), and the patch's own R1.3 `\changeref` still promises that
replacement while `supplementary_T07.tex` keeps the call.

**B2, minimal counterexample, verified in code.** `V = {x₁, x₂, Sin(a), Pow(p)}`,
`E = {x₁→a, a→p, x₂→p}`, `σ(p) = (a, x₂)`, `fcs = Vsnv^PnC`. A free-choice run may
put the pointer on `x₂` first: `p ∈ 𝒟_j` there while `σ(p)[0] = a` is uninserted.
This is verbatim the informal argument at `methodology.tex:761–766` that §3.1 gap 3
required to be moved into the lemma **and made rigorous**; it was moved, not
repaired. Fix is an induction on a topological order: Rule 1 **defers** `c`, never
strands it.

**B4 is nearly free to fix**: Appendix A's existing proof of 3.13 never uses
`D = S2D(w,m)` nor the greedy choice, so it already establishes *"`D ≅ S2D(w,m)`
for every `w ∈ 𝒲(D)`"* over labeled DAGs satisfying reachability. Restate the
theorem to match its own proof and Lemma A.2's last line goes through.

**Also checked, and no change needed** — recorded so it is not re-raised: the
`k = 0` scope worry at the head of this ticket is not a live defect. `fcs` does
return `''` for both `m = 1` and `m = 2`, but Theorem 3.15 quantifies both DAGs
over one shared `m` and Definition 3.9(iii) forces equal `m`, so the pair is not a
counterexample.

**Ezequiel's own three criticisms are correct** (verbosity; justifying theorems
with experiments; the appendix talking about a previous version of a proof) —
with one asymmetry worth telling him: both the test evidence and the "the old
sentence was false" admission belong in the **response letter**, which should keep
them. It is the **article** that must not narrate its own review history.

**GAP 2 IS REVERSED — the ticket has been wrong about this since it was
written.** Found while drafting the manuscript edits. **D2S already applies
Rule 1's exact predicate**: `dag_to_string._find_new_out_neighbor:338–341` is the
same test as `canonical.py:647–649`/`:725–727`, same `BINARY_OPS` scope (bug fix
B9 / Critical Invariant 8). So Rule 1 does **not** restrict the D2S pool — §3.1
gap 2's *"Rule 1 removes candidates … the proof's central claim is false as
written"* is itself wrong, and so is Ezequiel's Step 2 and the R2.1 letter
paragraph conceding it. The submitted sentence *"exactly the same candidate pool
as D2S"* is **true of the algorithms**.

The real defect is one level down: **Definition 3.5 and the D2S pseudocode
describe the procedure without its first-operand restriction**, so `𝒲(D)` is too
large and **Theorem 3.13 is false under that reading**. Verified: for
`D = x₁^x₂`, `w = NV^Nc` places every node and every edge of `D` (hence
`w ∈ 𝒲(D)` as Definition 3.5 is written) yet `S2D(w,2) = x₂^x₁` and
`is_isomorphic → False`. Stating the restriction in Definition 3.5 makes the pool
identity derivable **and** is what makes the widening of Theorem 3.13 to
`w ∈ 𝒲(D)` sound — without it the widening cannot be done at all.

**Proposal for where `𝒩` belongs, sent back with the review (§5 of the appendix
doc).** Ezequiel is right that it must be formalised somewhere — it does run,
before canonicalisation. The decisive fact, found while answering him: **the
operation that ran in every reported experiment is not
`LabeledDAG.normalize_const_creation`** but the adapters' `_normalize_const_edges`
(`bingo/adapter.py:212–216`, `udfs/adapter.py:216–224`, byte-identical), which
anchors every in-degree-0 `Const` to node 0 unconditionally — no least-index
search, no acyclicity test. `grep -rn normalize_const_creation experiments/models/`
returns **nothing**; the general routine is used only by measurement scripts.

Three consequences: the interface form makes **no node-index-ordered decision**, so
it is isomorphism-equivariant by inspection and **D-1 does not touch it**; the two
forms coincide on the interface class, so **no reported number depends on which one
the paper defines**; and the code's soundness silently rests on "no `Var` is an edge
target" — `_normalize_const_edges` discards `add_edge`'s return value, the same
pattern that caused the T15 orphaning bug.

Proposed: **Definition (`𝒩` adds `x₁→c` to every in-degree-0 `Const`) + Lemma (on
host DAGs where every in-degree-0 node is `Var`/`Const` and no `Var` is an edge
target: `𝒩(D)` acyclic; satisfies Thm 3.13's hypothesis; `𝒩` equivariant with
`𝒩(D₁) ≅ 𝒩(D₂) ⟺ D₁ ≅ D₂`; evaluation-preserving) + Corollary
(`fcs_{𝒩(D₁)} = fcs_{𝒩(D₂)} ⟺ D₁ ≅ D₂` on host output)**, all placed **after
Theorem 3.15** so nothing renumbers. Theorems 3.13/3.14/3.15 stay exactly as
submitted and `fcs` stays a pure function of `D`. The corollary is the statement the
deduplication experiments actually rely on and which the paper has never made — it
is currently justified only empirically (0 disagreements / 123,240 permutation tests
on 15,530 Bingo DAGs, `frac_in_c2 = 1.0000`). Proof sketches for all four lemma
clauses are in §5.2 of the appendix doc; each is 3–4 lines.

**Small code-hardening item this raises (Mario's, not blocking the paper):** assert
`add_edge`'s return value in both adapters' `_normalize_const_edges`, so the
dependency on hypothesis (b) fails loudly instead of silently.

**Edited `.tex` files produced for Ezequiel to review** (his originals preserved
under `results/T07_Ezequiel/original_ezequiel/`). Our changes are marked in
**red** over his blue; `[MPG 2026-08-02 — …]` brackets explain each removal and
must be stripped before integration. All three documents compile clean with no
undefined references, and the rebuilt PDF confirms the numbering is restored:
Theorem **3.13** / Lemma **3.14** / Theorem **3.15** as the reviewers cite them,
with the new Definition **3.16**, Lemma **3.17** and Corollary **3.18** after
Theorem 3.15. Per-file ledger: `results/T07_Ezequiel/CHANGES_MPG.md`; patches
`diff_MPG_*.patch`; PDFs `*_MPG.pdf`.

Status of the ticket is unchanged: Mario's half stays complete, AC-1…AC-4 stay
open with Ezequiel. Nothing was pushed to Overleaf and no file under the Overleaf
checkout was edited. A Spanish draft email covering all issues plus the proposal
is §6 of the appendix doc.

### 2026-08-03 — Patch integrated into the reviewed copy; R2.1 written; every theorem re-verified against code

**AC-1…AC-4 are discharged in text, AC-9/AC-10 remain.** Ezequiel's five-step
skeleton, with the four blocking repairs of the 2026-08-02 review applied, is now
in `reviews/internal_copy_reviewed_article/`, changes in **blue**, all `[MPG …]`
review notes and all red markup stripped. Both documents compile clean (0 errors,
0 undefined references). **Numbering confirmed from the rebuilt PDF: Theorem
3.13 / Lemma 3.14 / Theorem 3.15 exactly as the reviewers cite them**, with the
new Definition 3.16, Lemma 3.17 and Corollary 3.18 after Theorem 3.15. Paper
still 13 pages, supplementary 12.

**Independent verification of every formal statement against both engines.**
`experiments/scripts/t07_theorem_verification.py` (deterministic, seed 20260803,
output pinned at `reviews/t07_theorem_verification_output.txt`): **22/22 PASS**.
It implements a *free-choice D2S sampler* — the procedure Definition 3.5
quantifies over — which is what makes the widened Theorem 3.13 testable rather
than merely asserted.

| Check | Result |
|---|---|
| Def 3.5 loose reading breaks Thm 3.13 | `NV^Nc` on `x₁^x₂`: 3 nodes / 2 edges placed, decodes to `x₂^x₁`, `is_isomorphic=False` |
| Thm 3.13 widened, corrected Def 3.5 | **2,000** free-choice runs, **0** non-isomorphic |
| Thm 3.13 under the **submitted** Def 3.5 (POW-dense population) | **3,538 / 4,548 = 77.79 %** non-isomorphic — the defect is typical, not a curiosity |
| Lemma A.2 Step 3, Rule 1 defers (POW-only) | **4,548** runs, **4,081** exercised an exclusion, **0** stranded |
| Lemma A.2 Step 4, `|E|` accepted operations | 400 DAGs, 0 mismatches; the patch's `(|V|−m)+|E|` matched **0** times |
| Thm 3.15 (⇐)/(⇒) | 18,280 permutations 0 fails; 629 colliding pairs 0 non-isomorphic |
| Lemma 3.17 (i)–(iv) | 400 host DAGs; acyclicity, reachability, equivariance, evaluation all 0 failures |
| Corollary 3.18 | 2,100 permutation tests on host DAGs, 0 failures |
| `𝒩` out of the canonicaliser | orphan CONST refused on **both** engines; `𝒩(D)` canonicalises, engines agree |
| Engine equivalence | 400 DAGs, 0 disagreements |

**One harness defect worth recording, because it is the failure mode this ticket
has hit twice.** The first version measured the loose Def 3.5 reading on random
S2D DAGs and reported a rate that swung 0.5 % → 0.1 % → 0.0 % across seeds. Cause:
a random S2D string rarely closes a binary op's second operand, so almost every
binary op there has in-degree 1 and the first-operand restriction *cannot bite*.
The zero was **vacuous**, exactly like T07's earlier `Const`-free generator and
P2/P5's first run. The check was moved to a POW-dense generator that can exhibit
the phenomenon, and the vacuous variant deleted with a comment saying why.
**No rate from the random-S2D population may be quoted for this property.**

**Response letter written** (`reviews/response_to_reviewers.tex`): R2.1 in full,
plus the R1.3 addendum naming Lemma 3.17 / Corollary 3.18. New figure
`fig_operand_order.pdf`, generated from live code by
`experiments/scripts/generate_fig_operand_order.py`, which asserts equal node
sets, equal edge sets, differing operand orders, non-isomorphism and differing
canonical strings before drawing. Letter compiles: 17 pages, 0 Overfull, 0
unresolved references, 0 warnings; `rcomment` blocks verified untouched.

**Nothing pushed to Overleaf**, and no file under `article/` was edited — only
`reviews/` and `reviews/internal_copy_reviewed_article/`.

### 2026-08-03 — inbound from T18: the domain of condition (iv), and the base/pair equivalence

T18 found five DAGs on which the implementation's isomorphism predicate and the
canonical string disagreed. The diagnosis lands here, because the defect was in
the **statement of Definition 3.9(iv)**, not in the code's fidelity to it.

**What the manuscript already had right.** Condition (iv) is written *"for every
`Pow` node `v` with ordered input list `σ₁(v) = (u₁,u₂)`"* — a **pair**. So the
paper never asserted the whole-list rule for arbitrary in-degree. The
implementation did, and that is what over-separated. The code has been corrected
to the paper, not the reverse.

**What was genuinely missing, and is now stated.** Three gaps, all pre-existing
and none caused by T16 — the same species as the `𝒟_m` clarification at §T16
impact above:

1. **Definition 3.1 does not force in-degree to match arity.** It admits any
   acyclic `E`, so a `Pow` node may have three in-neighbours and
   Eq. (operand_order) has no pair to read. Every operand-order statement in the
   paper is silently conditional on `|σ(v)| = 2`.
2. **Condition (iv)'s domain was never named.** Outside expression DAGs it is
   either vacuous or, on the reading of the Remark below it, *stronger than the
   invariant of Theorem 3.15* — which is exactly the five-DAG discrepancy.
3. **Rule 1 and condition (iv) speak about different objects.** Rule 1
   (Definition 3.8, `methodology.tex:777`) excludes `c` when `σ(c)[0] ≠ u` —
   the base alone. Condition (iv) demands the whole pair. The completeness proof
   (`supplementary.tex:285–286`) already *uses* the base-only form. The step from
   one to the other was never justified; it was the implicit hinge of the (⇒)
   direction that R2.1 is about.

**The bridging lemma**, now written into the Remark rather than added as a new
numbered environment: on expression DAGs, `σ₂(φ(v)) = (φ(u₁), φ(u₂))` is
**equivalent** to `σ₂(φ(v))[0] = φ(σ₁(v)[0])`. One direction is immediate; for
the other, condition (i) sends `N⁻(v)` onto `N⁻(φ(v))`, so
`{φ(u₁),φ(u₂)} = {σ₂(φ(v))[0], σ₂(φ(v))[1]}`, and injectivity of `φ` leaves the
exponent no other image. Gap 3 closes: Rule 1 constrains precisely as much as
condition (iv) requires, no more and no less.

Why the restriction is load-bearing and not cosmetic: `Σ_SR` writes the base
explicitly (a `Vℓ`/`vℓ` token creates the node with its first in-edge) and every
later in-edge with a `C`/`c` token at a position the canonical traversal picks.
Read over graphs with surplus in-edges, condition (iv) separates DAGs that no
string can distinguish — strictly finer than Theorem 3.15's invariant.

**Edits made** (annotated copy only; `article/` verified untouched, nothing
pushed):

| Location | Change |
|---|---|
| `methodology.tex`, paragraph after Definition 3.1 | defines *expression DAG*; states that evaluation is defined there and nowhere else, and that host output always qualifies |
| `methodology.tex`, Definition 3.9(iv) | one blue clause naming the class the condition is stated on |
| `methodology.tex`, Remark 3.11 (*Necessity of condition (iv)*) | "match `σ` **exactly**" replaced by the base/pair equivalence with its proof, plus the `Σ_SR`-encoding reason the restriction matters |

**No new numbered environment**, deliberately: `remark` shares the `theorem`
counter, so inserting one would renumber 3.13/3.14/3.15 — which the reviewers
quote as literals. Verified from the rebuilt PDF: Definitions 3.1–3.9, Remarks
3.10–3.12, Theorem 3.13, Lemma 3.14, Theorem 3.15, Definition 3.16, Lemma 3.17,
Corollary 3.18 — **identical to before**. Both documents compile with 0 errors
and 0 undefined references; 0 `\color{red}` remain.

**No consequence for the letter.** The R2.1 answer's claim that condition (iv)
"compares ordered input lists" stays true, and `supplementary.tex:286`'s use of
the base-only form is now *exact* rather than a weakening. No number changes
anywhere; `is_isomorphic` has no production caller.

**Left for Ezequiel.** Whether to promote the equivalence from a remark to a
numbered lemma (it would renumber, so it needs a deliberate pass over every
literal the reviewers quote), and the T18.4 wording fork recorded in
`T18-canonical-completeness-operand-order.md` §8.

---

## 7bis. Hand-over to Ezequiel — what is left, and what is now settled

> **Read §7bis.0 first.** Half of what is left below is blocked on one decision
> (T16) and half is not. The list is organised by that split, because working the
> blocked half before the decision means writing text that may have to be
> deleted.

### 7bis.0 The fork that organises everything below

The manuscript and the production runs disagree about **which binary operations
are non-commutative**, and several of the remaining items read differently
depending on how that is resolved.

| | Non-commutative binary ops | Where this holds today |
|---|---|---|
| **As declared** (Def 3.2, `𝓛 = {+, *, g, i, s, c, e, l, r, ^, a, k}`) | `BINARY_OPS = {Pow}` | the manuscript, `methodology.tex:93-135` |
| **As implemented and as run** | `BINARY_OPS = {Pow, Sub, Div}` | `node_types.py`, both adapters, **every** production YAML |

`Sub` and `Div` are not a corner case in the runs: **61.1 %** of production-
configured Bingo candidates contain at least one (3,054 / 5,000 AGraphs, T16 §2),
and they occur about as often as `Add` and `Mul`. `OperationSet.commutative()`,
the factory that would realise the declared alphabet, is used by **no**
production config.

T16 offers two ways to close this, and **the choice is yours** because it is your
alphabet and your theorems:

- **Branch A — align the paper to the code** (T16 §4a). `𝓛` gains `-` and `/`;
  12 labels/31 tokens becomes 14/35; the commutative encoding is demoted to an
  available variant the reported runs did not use. Manuscript-only, **no number
  moves, no re-execution.**
- **Branch B — align the code to the paper** (T16 §4b). The adapters decompose
  `a − b → Add(a, Neg(b))` and `a / b → Mul(a, Inv(b))`. Every claim becomes true
  exactly as written, and **every number is recomputed**, because `x − y` becomes
  two nodes instead of one and `k` shifts for 61.1 % of candidates.

> ### RESOLVED 2026-07-30 — **Branch B.** Ezequiel decided; T16 AC-1 is closed.
>
> > *"Debes relanzar los experimentos para que todos usen el alfabeto definido en
> > el artículo. Es importante que esté alineada la teoría con el código. […]
> > Habrá que publicar el código fuente cuando acepten el artículo. Lo siguiente
> > que va a hacer todo el mundo es pasarle el artículo y el código a un LLM, el
> > cual se dará cuenta en menos de un segundo de cualquier incoherencia entre
> > ambos."*
>
> **What this means for your half of T07, and it is the best available outcome:**
> the declared alphabet becomes the alphabet that runs, so `𝓝 = {Pow}` throughout,
> **Definition 3.9(iv), Rule 1's prose, Definition 3.2 and the "31 tokens" count
> all stand exactly as submitted.** D-2 and D-3 evaporate. §7bis.3 below is
> retained as the record of what Branch A would have cost, and for the
> counterexamples, which stay useful as tests.
>
> The cost lands on Mario: the adapters must decompose, and the IsalSR arm
> re-runs on `D1 + D2`. Implementation spec, invariance analysis and validation
> gates are in **T16 §5–§8**. It still **gates T02 Wave 1**.

**Nothing about `normalize_const_creation`, `Const`, or the R1.3 answer depends
on this fork.** `Const` (`k`) is in `𝓛` under both branches, so AC-5, AC-6, AC-7,
AC-8, the numbered definition, the R1.3 response text and its figure stand
unchanged whichever way you decide.

---

### 7bis.1 Settled by measurement — nothing here needs re-deriving

- **AC-5 / R1.3.** The definition of `normalize_const_creation` is final. Text,
  properties N1–N5, the plain-language justification, both historical defects and
  the evidence table are in
  `T07-appendix/const_normalization_equivariance.md` **§4bis** (definition) and
  **§5** (paper prose, jargon-free). Mario carries these into `methodology.tex`.
- **AC-6 — discharged without amending any theorem.** `𝒩` is out of the
  canonicalisation path, so `fcs` is a pure function of `D` and the object the
  theorem describes is the object the code canonicalises. **Theorems 3.13, 3.14
  and 3.15 keep their statements exactly as submitted.** Two alternatives were
  explored and are closed: restating on `𝒩(D)` is **refuted** by the D-1
  counterexample (§1.6), and the adapter-image lemma is unnecessary once `𝒩` is
  not in the pipeline.
- **AC-7.** Rule 1 non-exclusion is carried **per op**, so it backs the proof
  under either branch of the fork: `Pow`, `Sub` and `Div` each tested separately,
  **0 failures**, 100 % round-trip. Numbers in the table in §7bis.2. Plus the
  property table above.
- **AC-8.** T06's precondition statement is unchanged and remains correct.

### 7bis.2 Yours, and **independent of the alphabet fork** — start now

None of the five items below changes wording depending on T16. They are the
whole of AC-1…AC-4 except for Rule 1's *named scope*, and they are not blocked on
Mario or on compute.

1. **Gap 1 — termination (AC-4).** Establish that the FCS run terminates having
   placed every node and every edge, rather than assuming it. The argument runs
   through "candidate pool non-empty at every branch point ⇒ progress ⇒
   termination". Its only contact with the fork is that non-emptiness is what
   Rule 1 could destroy; write it over "every non-commutative binary operation"
   and it holds verbatim under both branches. **No text differs between A and B.**
2. **Gap 2 — the pools are not identical (AC-2).** The proof's central sentence,
   *"exactly the same candidate pool as D2S"*, is false: Rule 1 **removes**
   candidates. This is true under both branches — Rule 1 restricts even when it
   filters `Pow` alone — so the sentence must go either way, replaced by a
   statement about pool **inclusion** (Rule 1 restricts, Rule 2 selects within).
   The reviewer located this contradiction themselves; a proof that does not
   visibly discharge it will read as evasive.
3. **Gap 4 — `κ`-minimal choice ∈ `𝒲(D)` (AC-1).** Definition 3.5 defines `𝒲(D)`
   by *free* choice among uninserted out-neighbours at each branch point, and has
   no `κ`. Showing that a `κ`-minimal choice *is* one of those free choices is
   exactly the step R2.1 says is missing. Purely about the choice function;
   labels do not enter. **No text differs between A and B.**
4. **Gap 5 — consume the reachability hypothesis (AC-3).** It is the lemma's
   stated hypothesis and the submitted proof never invokes it. It is now consumed
   **twice**: once inside Rule 1's non-exclusion argument, and once because the
   canonicaliser no longer repairs a violation but refuses it. The second use
   makes the hypothesis genuinely load-bearing in a way the submitted text did
   not need, which strengthens the proof rather than complicating it.
   **No text differs between A and B.**
5. **The sixth gap — a domain mismatch inside the chain, found 2026-07-29 and not
   previously recorded.** Lemma 3.14 applies Theorem 3.13 to an arbitrary labeled
   DAG `D`, but **Theorem 3.13 is stated only for `D = S2D(w, m)`**, i.e. DAGs in
   the image of S2D (`methodology.tex:974-975`). The lemma's *"therefore
   `D ≅ S2D(fcs_D, m)`"* does not follow for a `D` that is not assumed to be an
   S2D image. Either widen 3.13's quantifier or add the hypothesis to 3.14. This
   is a quantifier-domain question with no label content whatsoever.
   **No text differs between A and B.** A round-2 reviewer checking the chain
   will find it.

**How to write these without waiting for T16.** State Rule 1 over a set
`𝓝 ⊆ 𝓛` of non-commutative binary operations, prove non-exclusion for an
arbitrary member of `𝓝`, and instantiate `𝓝` in exactly one place at the end:
`𝓝 = {Pow}` under Branch B, `𝓝 = {Sub, Div, Pow}` under Branch A. Then the
T16 decision costs you a one-line edit instead of a rewrite.

**The empirical backing for Rule 1 non-exclusion already covers both branches**,
so you are not waiting on a measurement either (AC-7, closed by Mario):

| Op | DAGs tested | DAGs where Rule 1 actually excluded a candidate | Failures | Round-trip |
|---|---|---|---|---|
| `Pow` | 13,261 | 1,507 | **0** | 100 % |
| `Sub` | 13,226 | 1,507 | **0** | 100 % |
| `Div` | 13,253 | 1,503 | **0** | 100 % |

39 tests, C++ backend, broken out per op precisely so that `Pow`'s density cannot
mask a gap in the other two.

### 7bis.3 Was decided by the alphabet fork — **now resolved: no edits required**

**Branch B was chosen (§7bis.0), so every row below lands in the "no edit"
column. Nothing in this subsection is work for you any more.** It is kept for
three reasons: it records what Branch A would have cost, the counterexamples
remain useful as regression tests, and the last row still carries one live
obligation for T09.

Both items are the *same defect seen twice*: the manuscript names `Pow` where the
running code meant `{Pow, Sub, Div}`. Branch B removes `Sub` and `Div` from the
representation, so the manuscript becomes correct without being touched.

| # | Item | Location | **Branch A** (`𝓝 = {Pow, Sub, Div}`) | **Branch B** (`𝓝 = {Pow}`) |
|---|---|---|---|---|
| D-2 | Definition 3.9(iv) constrains operand order for `Pow` alone | `methodology.tex:920-929`, Remark `:955-957` | **Edit required.** Widen (iv) to `{Sub, Div, Pow}`; delete or restrict to `{Add, Mul}` the Remark's claim that all other binaries are commutative | **No edit.** (iv) is already correct, and the Remark is already correct |
| D-3 | Rule 1's prose scope says "`Pow` node" | `methodology.tex:752-760` | **Edit required.** Scope becomes `BINARY_OPS`; note the Table 3 caption ("binary non-commutative node") **already** matches the code, so the manuscript is internally inconsistent today | **No edit** to the prose. Optional one-sentence remark that the implementation applies Rule 1 to a superset, because S2D can still decode legacy `V-`/`V/` strings (T16 §5.2 keeps the core constant wide on purpose) |
| — | Def 3.2 / Table 1 / the token count | `methodology.tex:93-119` | **Edit required.** `𝓛` gains `-` and `/`: 12 labels → 14, **31 tokens → 35**. The commutative-encoding paragraph (`:121-135`) must be demoted from a property of the runs to an available variant | **No edit.** 12 labels / 31 tokens stands, and the commutative-encoding paragraph becomes true of the runs for the first time |

**Why D-2 is the more serious of the two, under Branch A.** It needs no exotic
input. Three-node DAGs with identical edge sets: `Sub` with `σ=(x₁,x₂)` gives
`V-PnC`, with `σ=(x₂,x₁)` gives `pv-nC`; likewise `Div` (`V/PnC` / `pv/nC`). For
`Sub` and `Div` the identity bijection satisfies Definition 3.9 (i)–(iii), and
(iv) is vacuous because the node is not `Pow` — so `D₁ ≅ D₂` per the definition
while `fcs_{D₁} ≠ fcs_{D₂}`. That falsifies the **(⇐)** direction of Theorem 3.15
*as stated*. **The code is right** (`x₁ − x₂ ≠ x₂ − x₁`); the definition is too
coarse. Under Branch B the counterexample cannot be constructed from adapter
output at all, because `Sub` and `Div` never reach the representation.

**No reported number is wrong under either branch.** The implementation is
self-consistent and *stricter* than the declared definition: it distinguishes
DAGs the definition would merge, which is the safe direction. What is wrong today
is the description.

### 7bis.4 The T16 decision — **taken 2026-07-30. Recorded for context only.**

Branch B. Nothing here is outstanding for you; the consequences below are now
Mario's, tracked in T16. Two of them were what made the decision urgent:

- **It gates T02 Wave 1.** Under Branch B the adapters change, so `k` shifts for
  61.1 % of candidates and every k-stratified number moves: `ρ` and the reduction
  factor, T06's violation-rate profile, T02's overhead-by-k, the bottleneck-type
  analysis, the search-space-reduction figures. Wave 1 must not launch before
  this is settled, or it runs the wrong code.
- **The direction of the `ρ` change is not predictable a priori.** More nodes
  means more structural variety (pushes `ρ` down) but also more opportunity for
  isomorphic rediscovery (pushes `ρ` up). It has to be measured, not argued.

**One correction to what this section previously said.** It called protected
`Inv` "the main scientific risk of Branch B", on the grounds that `a / b` and
`Mul(a, Inv(b))` differ near zero. That mis-classified the risk: IsalSR's
evaluator is never in the fitness path — the runners cache `canon_hash → fitness`
from the host, and `grep` over `experiments/models/` finds no use of
`evaluate_dag`. The protected regime therefore matters for *validating* the
decomposition and for dedup soundness, not for any reported number. See T16 §7.1.

**The reviewer-optics trade, for the record.** Branch A would have answered R2.3
with "we mis-described our alphabet" and demoted the "no operand-order tracking
is required" selling point. Branch B keeps every claim true as written, at the
cost of re-running the IsalSR arm on `D1 + D2`. Ezequiel's deciding argument was
neither of these: the source must be published on acceptance, and a paper/code
divergence will be found immediately by any reader with an LLM.

### 7bis.5 Editorial, inherited from T15 — Mario's, listed so you can see them

`methodology.tex:830` (inside `\begin{comment}`, so unrendered) and its
**rendered twin** in `supplementary.tex` near `:398` still read
`// redirect all Const creation edges to x_1`. That describes neither the current
policy nor the submitted one correctly, and the canonicaliser performs no
normalisation at all — so the pseudocode's first line is replaced by an explicit
statement of the precondition it assumes. Alphabet-independent.

---

## 8. Proposed answer

### 8.1 Before / after

| Item | Submitted | Revised | Source |
|---|---|---|---|
| Proof of Lemma A.2, length | 14 lines | 5 labelled steps, ~95 lines | AC-1 |
| Gap 1 — termination | not addressed | Step 4, measure argument, **`|E|`** accepted operations | AC-4 |
| Gap 2 — pools claimed identical | asserted | **derived** `𝒞_j = 𝒟_j`, Step 2 — the ticket had this backwards; Rule 1 restates D2S's own predicate | AC-2 |
| Gap 3 — Rule 1 non-exclusion | informal, in Def 3.8 | Step 3, topological induction; the false CDLL-timing claim removed from Def 3.8 too | AC-1 |
| Gap 4 — κ-minimal choice ∈ 𝒲(D) | not shown | Step 5, from `𝒞_j = 𝒟_j` | AC-1 |
| Gap 5 — reachability hypothesis used | never invoked | Step 3, base case **and** closure of the induction | AC-3 |
| Gap 6 — Thm 3.13's domain | `D = S2D(w,m)`, one string | widened to every labeled DAG satisfying reachability and every `w ∈ 𝒲(D)`; **proof unchanged** | new |
| Def 3.5's first-operand restriction | **absent — Thm 3.13 false** | stated in Def 3.5 and Table 2 | new |
| `normalize_const_creation` defined | **nowhere** | Definition 3.16, with complexity `O(|C| m (|V|+|E|))` | AC-5 |
| Theorem 3.15 mentions normalisation | no | **no, and correctly so** — `𝒩` is out of the canonicalisation path | AC-6 |
| Interface argument for host DAGs | **never made** | Lemma 3.17 + Corollary 3.18, after Thm 3.15 | new |
| Tests covering the above | — | 22/22 in `t07_theorem_verification.py`, both engines | AC-7 |

### 8.2 Changes made to the manuscript

Applied to `reviews/internal_copy_reviewed_article/`, changes in blue. **Not**
pushed to Overleaf and **not** applied under `article/`.

| File | Change |
|---|---|
| `paper/methodology.tex` | Def 3.5 gains the first-operand restriction and the `NV^Nc` counterexample; Def 3.8 gains the precondition sentence and loses the false timing claim; Rule 1's prose replaced by the deferral argument; Thm 3.13 widened; Thm 3.15 **untouched**; new §3.6 "The host-solver interface" with Definition 3.16, Lemma 3.17, Corollary 3.18 |
| `supplementary/supplementary.tex` | Thm A.1 statement mirrors 3.13 and its proof opens on "a D2S run"; part (iv) cites the restriction; Lemma A.2 proof replaced by the five steps; Thm A.3 **untouched**; Lemma 3.17 restated with its four-clause proof; Table 2's two insertion guards gain `σ(v)=ε or σ(v)[0]=u`; Table 3's first line becomes the Precondition |
| `reviews/response_to_reviewers.tex` | R2.1 written; R1.3 gains the Lemma 3.17 / Corollary 3.18 paragraph; Table 1 rows for R1.3 and R2.1; `\fcs`, `rlemma`, `rcoro` added to the preamble |
| `reviews/fig_operand_order.pdf` | new figure for R2.1, generated from live code |

### 8.3 Response text — WRITTEN

Both are in `reviews/response_to_reviewers.tex`. Note that the R2.1 structure
sketched here was **wrong on gap 2** and was not followed: the letter does not
concede that "exactly the same candidate pool as D2S" is false, because it is
true of the two algorithms. The concession the letter makes instead is the
correct one, that Definition 3.5 described D2S without its first-operand
restriction and Theorem 3.13 is false under that reading.

R2.1 runs: concede; name Definition 3.5 as the root defect and show it with
Figure 3; note the restriction is already used by Theorem 3.13's own proof at
part (iv), so no proof changes; derive `𝒞_j = 𝒟_j`; widen Theorem 3.13; walk the
five steps as prose; report the free-choice verification; volunteer the
termination-versus-cost limitation; state that no reported number changes;
cross-link R1.3 and R2.4.

### 8.4 Residual risk

> Candidates: the new proof being judged terse again — mitigate by discharging each
> gap in a separately labelled step; the amended Theorem 3.15 statement reading as a
> weakening of the headline claim (it is a *correction*, and should be framed as
> making explicit what the implementation always did); page cost in a paper R2 wants
> shorter.
