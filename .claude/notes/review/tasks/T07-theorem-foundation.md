# T07 — Complete the formal foundation of Theorem 3.15

| Field | Value |
|---|---|
| Reviewer comments closed | **R2.1**, **R1.3** (and R2's B3 = "Partially") |
| Type | Theory |
| Owner | **Ezequiel** (primary — proofs, `methodology.tex`) · **Mario** (empirical verification, tests) |
| Depends on | — (can start immediately; independent of compute) |
| Blocks | T03 phase 3, T06, T13 |
| Status | NOT STARTED |
| Target | 2026-08-24 |

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

**Connection to R1.2 (T06)**: constant-only subexpressions have no variable ancestor
and therefore *violate* the reachability precondition; normalisation is the repair.
If T06's measurement confirms this, the two tickets explain each other and the
normalisation step gains the motivation the paper currently lacks. Coordinate.

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
- A numbered definition in `methodology.tex`, placed before Definition 3.8, with:
  the operation, why the choice of creation source is semantically irrelevant, why
  `x₁` is always a valid target (no incoming edges ⇒ no cycle), the complexity
  (O(number of `Const` nodes)), and the guard condition (applied only when the DAG
  has `Const` nodes).
- **Amend the statement of Theorem 3.15** so the normalisation is part of the
  hypothesis or part of the canonical-string definition. As it stands the theorem is
  false for DAGs with `Const` nodes created from different sources, and stating it
  correctly is a stronger position than adding a remark.
- A short lemma or remark establishing evaluation-preservation:
  `eval(D) = eval(normalize(D))`.

### 5.3 Structural cleanup that these fixes enable
Appendix A's proofs restate 3.13/3.14/3.15 verbatim, so every theorem exists twice
under two numbers — which is why R2 had to write "Theorem 3.13/3.15" and "Lemma A.2"
in the same sentence, and is adjacent to the broken cross-reference in R2.4.
Consider whether the restatements can be replaced by references. Coordinate with
T11 (cross-references) and T13 (page budget) — this is a page-saving opportunity in
a revision that must not grow.

### 5.4 Empirical verification (Mario)
Every claim the proof makes that is checkable in code must be checked:
- Rule 1's non-exclusion property, tested on DAGs containing `Pow`.
- Termination, on the full 14,841-DAG corpus and on evolved DAGs.
- `eval(D) = eval(normalize(D))`, property-based test.
- Failure of Theorem 3.15 **without** normalisation: construct two isomorphic DAGs
  with `Const` nodes created from different sources and show their canonical strings
  differ pre-normalisation and agree post-normalisation. This single test is the
  concrete evidence that the step is a precondition rather than a convenience, and
  it belongs in the response to R1.3.

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
- **AC-7.** All §5.4 tests written and passing, including the pre/post-normalisation
  counterexample.
- **AC-8.** Consistent with T03's insertion point and with T06's statement of the
  precondition.
- **AC-9.** Page cost reported to T13.
- **AC-10.** §8 filled.

---

## 7. Work log

_(empty — to be filled by the implementing agent)_

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
