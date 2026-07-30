# T04 — Naive fixed-order-serialisation hash dedup baseline

| Field | Value |
|---|---|
| Reviewer comments closed | **R1.4** (and materially improves B4 for all three reviewers) |
| Type | New experiment — full comparator |
| Owner | **Mario** (+ Claude Code) |
| Depends on | T01 (engine), T02 (campaign infrastructure and headline numbers) |
| Blocks | T09 (tables), T13 (page budget) |
| Status | NOT STARTED |
| Target | implementation 2026-08-17 · results 2026-09-01 |

---

## T16 impact — the hash must serialise **decomposed** DAGs (added 2026-07-30)

T16 moved the adapters to the paper's alphabet: `Sub` and `Div` are rewritten to
`Add(a, Neg(b))` and `Mul(a, Inv(b))` at the host→`LabeledDAG` boundary, leaving
`Pow` as the only non-commutative operation.

**Requirement: this arm must hash the same object IsalSR canonicalises.** If the
naive serialisation is built over undecomposed DAGs while the IsalSR arm
canonicalises decomposed ones, the comparison answering R1.4 is run on two different
representations and is meaningless.

**This is satisfied by construction, and there is nothing to do — provided you build
on the adapters.** The decomposition lives *inside* `agraph_to_labeled_dag` and
`compgraph_to_labeled_dag`, so every consumer inherits it. Do not construct
`LabeledDAG`s by any other route for this arm. Verified 2026-07-30: no T04 code
exists yet, so this is recorded as a constraint on the implementation rather than a
check already performed.

Note also that `k` grows ~22 % under decomposition, so any fixed-width or
capacity-bounded serialisation must be sized against the **new** `k` distribution
(Bingo mean 5.47 → 6.72, p95 11 → 15).

---

## 1. Why this is its own ticket

R1.4 is the heaviest single request in the round and the one most likely to decide
round 2. It is not grouped with anything: it is a self-contained new comparator,
and unlike R1.2 (a measurement on existing machinery) or R3.1 (more problems on
existing machinery), it introduces a *method* into the evaluation that the paper
currently does not have.

**Decision taken 2026-07-27**: this baseline is expected to appear in the final
article as a **full comparison**, not as a footnote — reviewers will look for it.
It must simultaneously be made clear that it does not have IsalSR's properties.

**Verbatim comment:**

> 4) There is no comparison against naive hash-based deduplication on a fixed-order
> DAG serialization. This is the obvious baseline and its absence makes it hard to
> assess how much of the benefit requires 1-WL machinery versus a much simpler
> approach.

---

## 2. The scientific shape of the answer

The comparison is **not** a horse race that IsalSR wins. It is a decomposition, and
saying so plainly is the strongest available response.

A fixed-order serialisation hash is **sound but incomplete**: it never merges
non-isomorphic DAGs, but it fails to merge isomorphic DAGs that differ in node
numbering — which is exactly the redundancy IsalSR targets. So the observed
reduction factor decomposes:

```
ρ_total  =  ρ_exact   ×   ρ_iso
            ^^^^^^^^      ^^^^^
            caught by     caught ONLY by an
            a fixed-      isomorphism-complete
            order hash    invariant
```

The paper's job is to report both factors, on real search trajectories, per method.
That answers R1.4 exactly as asked ("how much of the benefit requires 1-WL
machinery versus a much simpler approach") and it is a *better* result than a win,
because it quantifies the contribution instead of asserting it.

**What the existing evidence predicts.** Two pieces already in the submission
constrain the answer and should be cited in §8:
- `supplementary.tex:689–693`: UDFS duplicates "arise only from the commutative
  symmetries of ADD and MUL", i.e. from operand permutation — which a fixed-order
  hash does **not** catch. Predicts ρ_exact ≈ 1 for UDFS.
- `supplementary.tex:782–786`: on the 5,400 synthetic DAGs every one has trivial
  automorphism group and ρ = k! exactly — a regime where a fixed-order hash catches
  nothing beyond byte-identical repeats.

Bingo is the interesting case: stochastic GP re-generates byte-identical individuals
(the B12 note in `CLAUDE.md` records that VarAnd produces unmodified `parent.copy()`
offspring ~36 % of the time), so ρ_exact should be materially above 1 there. That
is the honest concession, and it should be stated before a reviewer extracts it.

**Reviewer 1 has already pre-framed the cost axis in their own B2 statement**:
1-WL canonicalisation is *"a meaningful middle ground between the O(k!) exhaustive
search and hash-based approaches that offer no correctness guarantee"*. They expect
the hash baseline to lose on completeness. The open questions they actually want
answered are **by how much** and **at what cost**.

---

## 3. Mandatory reading

- `.claude/notes/review/tasks/EXECUTION-PLAN.md` — Wave 3; the certification gate
  applies to this wave too
- `.claude/notes/review/source/reviewer-1.md` — §R1.4 and the full B2 statement
- `.claude/notes/review/source/verified-discrepancies.md` — context on ρ (E8)
- `.claude/notes/review/source/codebase-pointers.md` — `sympy_adapter` is flagged as
  the likely place to build the fixed-order serialisation;
  `model_validation/diversity/dedup_smoke/` may already hold a related smoke test
- `CLAUDE.md` (repo root) — B12 VarAnd clone detection; the UDFS `processes: 1`
  constraint; dedup uses `set[int]` not `set[str]`
- `src/isalsr/core/README.md`
- `.claude/notes/review/tasks/T02-cpp-reexecution-campaign.md` — protocol to match
- `docs/md_files/design/experimental_design/isalsr_experimental_design.md`

---

## 4. Steel-man the baseline

A weak implementation of the comparator is worse than none: R1 will read it as
evasion. Requirements:

- Implement **at least three** fixed orders and report the best-performing one as
  *the* baseline: (i) node-insertion order as the host solver produced it;
  (ii) topological order with ties broken by `(label, in-degree, out-degree)`;
  (iii) DFS pre-order from `x_1` with children sorted by `(label, subtree size)`.
- Serialise labels **and** operand order (Critical Invariant 8) so the baseline is
  semantically sound, not accidentally lossy.
- Hash with the same 64-bit function the IsalSR dedup set uses, so the memory and
  collision analysis is shared and the comparison is not confounded by hashing.
- Order (iii) in particular starts to approximate a canonical form; if it performs
  close to IsalSR, that is a genuine finding and must be reported as one, not
  buried. Note in §7 where the boundary between "naive fixed order" and "canonical
  form" actually falls — this is a legitimately interesting contribution.

---

## 5. Work specification

Two measurement modes, both required (decision 2026-07-27, "1+2").

### 5.1 Mode 1 — offline replay (mechanism decomposition)
Replay the stored DAG streams from the T02 campaign through all three fixed-order
hashers and through IsalSR canonicalisation, on identical input sequences. Produces
`ρ_exact`, `ρ_iso`, and `ρ_total` per (method, problem, seed), plus per-DAG cost for
each scheme. This is the controlled comparison: identical inputs, zero search
confound.

### 5.2 Mode 2 — live third arm (end-to-end comparison)
A third variant, `hash`, run as a full solver campaign alongside `baseline` and
`isalsr`. **Decided 2026-07-27: full live arm on the complete suite** —
2 methods × ≈70 problems × 30 seeds = **4,200 runs**, same protocol as T02, full
12 h budget. This is Wave 3 in `EXECUTION-PLAN.md`; it launches after the C++ wave,
never in competition with it.

This is what makes it "a full comparison" in the article: R², NRMSE, solution
recovery, wall-clock and `S` for the hash variant on the same footing as the other
two arms, across every problem the other arms cover.

**Run Mode 1 before Wave 3 launches.** The replay costs no queue time and yields the
ρ_exact / ρ_iso decomposition — the number R1.4 actually asked for. If the replay
shows the hash catches almost nothing, that is worth knowing before committing
50,400 core-hours, and it changes how the live arm is framed rather than whether it
runs.

### 5.3 Statistical treatment
Three arms means the paired-test structure changes. Use CPDT pairwise
(`isalsr` vs `baseline`, `hash` vs `baseline`, `isalsr` vs `hash`) with Holm
correction across the three contrasts, plus a Friedman/Nemenyi over the three arms
per method for the critical-difference figure. Do **not** silently reuse the
two-arm machinery.

### 5.4 Completeness demonstration
Construct and report a small explicit family of isomorphic DAG pairs that the hash
baseline separates and IsalSR merges. One worked example in the paper is worth more
than a paragraph of assertion, and it makes the soundness/completeness distinction
concrete for a reader.

---

## 6. Acceptance criteria

- **AC-0.** §7 Work log filled in as the work proceeds.
- **AC-1.** Three fixed-order serialisations implemented, unit-tested, and shown to
  be sound (never merges non-isomorphic DAGs) on the 14,841-DAG corpus.
- **AC-2.** Mode 1 replay complete; `ρ_exact`, `ρ_iso`, `ρ_total` reported per
  method, per problem, with dispersion, and stratified by k.
- **AC-3.** Mode 2 campaign complete (Wave 3, ≈4,200 runs across the full suite) or
  every missing run accounted for. Mode 1 was run and read **before** Wave 3 launched.
- **AC-4.** Per-DAG cost of hash vs canonicalisation measured on Picasso hardware
  under the C++ engine, with the resulting `S` for all three arms.
- **AC-5.** Three-arm statistical comparison done correctly (§5.3), including the
  critical-difference diagram.
- **AC-6.** Worked isomorphic-pair example produced (§5.4).
- **AC-7.** The paper text states explicitly that the hash baseline is **sound but
  incomplete** and does not provide a labeled-DAG isomorphism invariant.
- **AC-8.** If the hash baseline turns out to be competitive on any axis, that is
  reported without softening. Record it in §7 first.
- **AC-9.** §8 filled.

---

## 7. Work log

_(empty — to be filled by the implementing agent)_

---

## 8. Proposed answer

### 8.1 Before / after

| Quantity | Submitted | Revised | Source |
|---|---|---|---|
| Hash-dedup baseline present | **absent** | present, full comparator | §5.2 |
| Best fixed order (of three tested) | — | | §4 |
| ρ_exact (hash-catchable), UDFS | not reported | | Mode 1 |
| ρ_exact (hash-catchable), Bingo | not reported | | Mode 1 |
| ρ_iso (WL-only), UDFS | not reported | | Mode 1 |
| ρ_iso (WL-only), Bingo | not reported | | Mode 1 |
| ρ_total, UDFS | 1.56 | | |
| ρ_total, Bingo | 1.83 | | |
| Fraction of duplicates requiring 1-WL, UDFS | not reported | | **the number R1.4 asked for** |
| Fraction of duplicates requiring 1-WL, Bingo | not reported | | **the number R1.4 asked for** |
| Per-DAG cost, hash (ms) | — | | |
| Per-DAG cost, IsalSR canon (ms) | 0.817 (Bingo, Python) | | T01/T02 |
| `S`, hash arm, Bingo | — | | Mode 2 |
| `S`, IsalSR arm, Bingo | 0.93 | | T02 |
| R² test, hash vs baseline | — | | CPDT |
| R² test, IsalSR vs hash | — | | CPDT |
| Merges non-isomorphic DAGs? | — | hash: no · IsalSR: no | AC-1 |
| Merges isomorphic renumbered DAGs? | — | hash: **no** · IsalSR: yes | §5.4 |

### 8.2 Changes made to the manuscript

| File | Lines (revised) | Change |
|---|---|---|
| | | |

### 8.3 Draft response text

```latex
%% --- R1.4 ---
\begin{response}
%% Structure that works here:
%%  1. Accept the request without hedging; the baseline is now in the paper.
%%  2. Give the decomposition rho_total = rho_exact x rho_iso and the two numbers
%%     per method. This is literally what the reviewer asked for.
%%  3. Concede the Bingo case explicitly if rho_exact is large there -- the
%%     reviewer will find it otherwise, and conceding it costs nothing given the
%%     rho_iso factor is the paper's actual claim.
%%  4. State soundness vs completeness once, precisely, and point at the worked
%%     example.
%%  5. Give the cost comparison; the hash is cheaper and we say so.
\changeref{}
\end{response}
```

### 8.4 Residual risk

> Candidates: a reviewer arguing our fixed orders were chosen to lose (mitigated by
> §4's three orders and by reporting the best); order (iii) approaching canonicality
> and blurring the contribution; whether the three-arm correction was applied
> correctly; whether ρ_iso is large enough on UDFS to carry the claim given UDFS's
> duplicates are commutative-symmetry duplicates.
