# T07 — Complete the formal foundation of Theorem 3.15

| Field | Value |
|---|---|
| Reviewer comments closed | **R2.1**, **R1.3** (and R2's B3 = "Partially") |
| Type | Theory |
| Owner | **Ezequiel** (primary — proofs, `methodology.tex`) · **Mario** (empirical verification, tests) |
| Depends on | — (can start immediately; independent of compute) |
| Blocks | T03 phase 3, T06, T13 |
| Status | **NOT STARTED** (proofs, Ezequiel) · **Inbound material ready**: T15 changed the normalisation scheme on 2026-07-27 and T06 measured its effect on 2026-07-28. **§3.2 and §5.4 below described the pre-fix semantics and have been corrected — read §3.3 before writing anything.** |
| Target | 2026-08-24 |
| Last updated | 2026-07-28 — §3.3 added, §5.2/§5.4 corrected, T06/T15 hand-over recorded in §7 |

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
- **AC-6.** Theorem 3.15's statement accounts for the normalisation. *(Amended
  2026-07-28: may be dischargeable **by argument** rather than by an amendment.
  Since the repair is now the identity on the hypothesis class (§3.3b), the theorem
  holds as written there; what still needs settling is the scope question in §7 —
  the implementation canonicalises `normalize(D)`, and 85.9–100 % of real inputs
  enter the hypothesis class only through that step.)*
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

---

## 8. Proposed answer

### 8.1 Before / after

| Item | Submitted | Revised | Source |
|---|---|---|---|
| Proof of Lemma A.2, length | 14 lines | | |
| Gap 1 — termination | not addressed | | AC-4 |
| Gap 2 — pools claimed identical (false) | asserted | | AC-2 |
| Gap 3 — Rule 1 non-exclusion | informal, in Def 3.8 | | AC-1 |
| Gap 4 — κ-minimal choice ∈ 𝒲(D) | not shown | | AC-1 |
| Gap 5 — reachability hypothesis used | never invoked | | AC-3 |
| `normalize_const_creation` defined | **nowhere** | | AC-5 |
| Its complexity stated | no | | AC-5 |
| Theorem 3.15 mentions normalisation | no | | AC-6 |
| Counterexample without normalisation | not given | | AC-7 |
| Tests covering the above | — | | AC-7 |
| Theorem restatements in Appendix A | verbatim duplicates | | AC-9 / T13 |

### 8.2 Changes made to the manuscript

| File | Lines (revised) | Change |
|---|---|---|
| `article/paper/methodology.tex` | | |
| `article/supplementary/supplementary.tex` | | |

### 8.3 Draft response text

```latex
%% --- R2.1 ---
\begin{response}
%% Structure that works here:
%%  1. Accept the criticism directly. The reviewer is right, and gap 2 is worse
%%     than they said -- the proof's own central sentence is false, because Rule 1
%%     does restrict the pool. Saying this ourselves is worth more than being
%%     told it in round 2.
%%  2. Walk the five gaps and say where each is now discharged, with line numbers.
%%  3. Note that the reachability hypothesis is now consumed at an identified
%%     step, since it was previously stated and unused.
%%  4. Note the added tests, so the proof has an executable counterpart.
\changeref{}
\end{response}

%% --- R1.3 ---
\begin{response}
%% Structure that works here:
%%  1. Confirm the omission -- the step appeared once, in pseudocode, undefined.
%%  2. State plainly that it is a precondition of Theorem 3.15, not preprocessing:
%%     without it the theorem is false for DAGs with Const nodes. Give the
%%     counterexample from AC-7.
%%  3. Point at the new numbered definition and the amended theorem statement.
%%  4. If T06 confirms the constant-subexpression mechanism, connect the two
%%     comments: this step is also what makes R1.2's precondition hold in practice.
\changeref{}
\end{response}
```

### 8.4 Residual risk

> Candidates: the new proof being judged terse again — mitigate by discharging each
> gap in a separately labelled step; the amended Theorem 3.15 statement reading as a
> weakening of the headline claim (it is a *correction*, and should be framed as
> making explicit what the implementation always did); page cost in a paper R2 wants
> shorter.
