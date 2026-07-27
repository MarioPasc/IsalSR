# T03 — Gray-code integration: design-space analysis, implementation, ablation

| Field | Value |
|---|---|
| Reviewer comments closed | none (internal decision 2026-07-27) |
| Type | Theory + implementation + experiment — **SECONDARY. Reserves no queue capacity.** |
| Owner | **Ezequiel** (design authority, theory) + **Mario** (implementation, experiments) |
| Depends on | T01 · must not delay or compete with T02 |
| Blocks | T13 (page budget) · T10 only if the ablation runs *and* changes `S` |
| Status | NOT STARTED |
| Target | design analysis 2026-08-10 · implementation 2026-08-24 · **ablation go/no-go 2026-08-31** |

---

## 0. Scope — re-scoped 2026-07-27

The priority of this revision is the **C++ re-execution (T02)**, which is the
headline result. This ticket is explicitly secondary and is subordinate to it in
three ways:

1. **It reserves no queue capacity.** The Gray ablation is Wave 4 in
   `EXECUTION-PLAN.md` — pure spillover, launching only if Waves 1–3 are complete
   and the queue is free.
2. **Go/no-go date: 2026-08-31.** A 12 h campaign launched after that cannot finish,
   be analysed, and reach the 2026-09-10 number freeze. If the date passes without a
   launch, this ticket closes as *design + implementation + theory*, with the
   ablation reported as characterised future work.
3. **Promotion to headline is a late, evidence-gated option, not the plan.** Gray
   replaces the C++ results as the headline **only if** the ablation completes before
   the freeze *and* clears the §5 Phase 5 promotion rule. Otherwise the C++ numbers
   stand and Gray is reported as an ablation.

What is **not** descoped: the design analysis (§2), the implementation, and — if the
chosen insertion point demands it — the proofs. Those cost no queue time, are on
Ezequiel's and the implementation track rather than the campaign track, and are
worth completing regardless of whether the ablation runs. Closing this ticket with a
correct, proved, tested Gray encoding and no ablation is a **successful** outcome.

---

## 1. Why this ticket exists

Not a reviewer request. Added by internal decision to bring López-Rubio's
variable-length Gray code for the natural numbers into IsalSR, mirroring the
integration proposed for IsalHG.

**Reference**: E. López-Rubio, *A Variable-Length Gray Code for the Natural
Numbers*, arXiv:2607.16088 (2026). PDF at
`/home/mpascual/research/code/IsalHG/docs/references/variable_length_gray_code.pdf`.
Its §5.3 names IsalHG and proposes exactly this integration.

**The structural motivation.** Σ_SR encodes pointer displacement in **unary**: to
move a pointer δ slots you emit |δ| tokens (`N`/`P` for primary, `n`/`p` for
secondary). String length therefore scales with *layout distance* rather than with
structure — this is the sole origin of the `n` factor in the length bound and of
the `|a| + |b|` cost model in the D2S spiral order (Critical Invariant 5). A
variable-length Gray code replaces Θ(|δ|) tokens with ≈ `1 + ⌊log₂(|δ|+1)⌋` bits,
and has the property that adjacent magnitudes differ in a single bit.

**The insertion point is deliberately left open.** It is the substance of this
ticket, not an input to it.

---

## 2. Design space — the agent must analyse this, not assume it

Three candidate insertion points are known. There may be others; finding a better
one is in scope.

### (a) Post-canonicalisation transcoding `T`
Apply `T` to `w*_D` after canonicalisation, as IsalHG's
`docs/article/theoretical/stability_reformulations.md` §4 formulates it. `T` is
injective on encoder outputs (movement blocks are delimited by V/C tokens, and the
per-pointer sub-run rendering order is deterministic), so
`T(w*_{D₁}) = T(w*_{D₂}) ⟺ w*_{D₁} = w*_{D₂} ⟺ D₁ ≅ D₂`, and Theorem 3.15 transfers
with a one-paragraph injectivity argument. **No theorem is re-proven.**

> **Consequence that must be stated plainly in the analysis**: under (a) the
> equivalence classes are *identical*, so **ρ is provably unchanged** and the
> deduplication axis cannot improve. Canonicalisation time also cannot improve —
> `T` is strictly extra work after the fact. The only quantities that move are
> string length, dedup-key memory, and the Appendix F Levenshtein/metric substrate.
> Any claim that (a) "produces better results" on ρ or on `S` is false by
> construction, and an agent that reports one has made an error.

### (b) Alphabet-level redesign of Σ_SR
Movement instructions are replaced by coded displacement tokens `D(pointer, δ)`
inside the alphabet itself. The canonical string changes. The D2S spiral order
resorts from `|a| + |b|` to `bits(a) + bits(b)`, which changes which candidate is
selected at each branch point, which changes `𝒲(D)`, which changes `w*_D`.
Theorems 3.13 and 3.15 and Lemma 3.14 must all be re-derived. This is the only
variant under which ρ, canonicalisation cost, or `S` can move, and therefore the
only variant under which "ship it as the main algorithm" is a meaningful sentence.

> **Re-proof cost and re-execution cost are explicitly declared non-constraints for
> this ticket** (decision, Mario, 2026-07-27). Ezequiel takes the proof obligation.
> The analysis should nonetheless *state* what the obligation is, because T13 has
> to find page budget for it and T07 is already re-writing the proof of Lemma 3.14
> for R2.1 — the two must not diverge.

### (c) Cost-model-only variant
Keep the unary alphabet, but change only the D2S candidate ordering to sort by
coded length rather than by `|a| + |b|`. Changes `w*_D` and so requires re-proof of
the invariance direction, but leaves S2D and the alphabet untouched. A middle
point worth costing.

### Evaluation matrix the analysis must fill

| Axis | (a) transcoding | (b) alphabet | (c) cost model | notes |
|---|---|---|---|---|
| Theorems requiring re-proof | none | 3.13, 3.14, 3.15 | 3.14, 3.15 | |
| ρ / dedup classes | identical by construction | may change | may change | |
| Canonicalisation time | strictly worse (+T) | ? | ? | |
| `\|w*\|` scaling | O(k log n) | O(k log n) | O(k n) | |
| Appendix F metric substrate | new | new | unchanged | avalanche caveat below |
| Interaction with T07 (R2.1 proof) | none | direct | direct | |
| Page budget required (T13) | small | large | medium | |

**Known caveat to carry over from IsalHG** (`stability_reformulations.md` §4 and
§7.4): a Gray-coded bitstring is the bit-level form of a magnitude-weighted
substitution cost and inflates avalanche cost by a `log n` token-length factor when
used as a *metric* substrate. Separately, arithmetic coding must **never** be the
metric substrate (one changed symbol rescales every subsequent coding interval).
IsalSR's Appendix F is exactly a metric-space section, so this caveat is live here,
not theoretical.

---

## 3. Mandatory reading

**Gray code and the IsalHG precedent** (`/home/mpascual/research/code/IsalHG/`)
- `docs/references/variable_length_gray_code.pdf` — **the primary source**, §5.3
- `docs/article/theoretical/stability_reformulations.md` — §4, §5, §7.4, §7.5
- `docs/article/DEVELOPMENT/T-M4/OPEN/T-M4a.md` — the sibling ticket
- `docs/article/DEVELOPMENT/SESSIONS.md` — S6 execution plan

**IsalSR specification**
- `.claude/notes/review/tasks/EXECUTION-PLAN.md` — Wave 4 and the go/no-go rule
- `CLAUDE.md` (repo root) — Critical Invariants 4, 5, 8; the instruction-set table
- `src/isalsr/core/README.md`
- `.claude/notes/review/tasks/T01-cpp-core-port.md` — the engine this lands on
- `.claude/notes/review/tasks/T07-theorem-foundation.md` — **coordinate; do not
  produce two divergent proofs of Lemma 3.14**

**Review context**
- `.claude/notes/review/source/reviewer-2.md` — R2.1, and the reason B3 is "Partially"
- `.claude/notes/review/source/manuscript-map.md` — Def 3.4, 3.5, 3.8; Thm 3.13/3.15
- `.claude/notes/review/source/README.md` — the 12-page constraint

---

## 4. Non-goals

- Arithmetic coding as a *metric* substrate. Ruled out upstream; see §2 caveat.
- Any change to how UDFS or Bingo search.
- Sibling-repository work. IsalHG's own Gray ticket (T-M4a) is theirs.

---

## 5. Work specification

**Phase 1 — analysis (Ezequiel + Mario, deliverable before any code).**
A design document at `docs/md_files/design/gray_code_integration.md` filling the §2
matrix, naming the recommended insertion point, and stating the exact proof
obligation it creates. Must include the honest statement about (a) not being able
to move ρ. Must be reviewed by Ezequiel before Phase 2 starts.

**Phase 2 — implementation.** On the C++ core from T01, behind a switch so the
non-Gray path remains runnable for the ablation. Python reference implementation
first, C++ second, byte-equivalence between them (same gate discipline as T01 §5.3).

**Phase 3 — theory.** Whatever Phase 1 identified. If the chosen point requires
re-proving 3.13/3.14/3.15, those proofs are Ezequiel's and must be written against
the *revised* Lemma 3.14 that T07 produces, not the submitted one.

**Phase 4 — ablation. Conditional; go/no-go 2026-08-31.** Wave 4 in
`EXECUTION-PLAN.md`: paired campaign, Gray vs non-Gray, same protocol as T02, full
12 h budget on all ≈70 problems (4,200 runs). **Launches only if Waves 1–3 are
complete and the queue is free.** Report per axis: ρ, R² train/test, `|w*|`
distribution, canonicalisation time, `S`, and the Appendix F metric quantities.
Statistical treatment identical to the main campaign (CPDT primary).

If the go/no-go date passes without a launch, skip to Phase 5 and record the
outcome as *not measured* — do not launch a truncated campaign to have something to
report. A partial ablation is worse than none: it invites the reviewer to ask what
the missing problems would have shown.

**Phase 5 — promotion decision.** Gray replaces the C++ results as the headline
**only if** the ablation completed before the freeze **and** Gray wins or ties on
every reported axis and strictly wins on at least one, with the comparison stated as
a paired test. A win on `|w*|` alone with a loss on `S` is not a promotion; it is an
ablation reported in the supplementary. If the ablation did not run, the C++ results
stand as the headline and Gray ships as design + implementation + theory. Record the
decision and its evidence in §7.

---

## 6. Acceptance criteria

- **AC-0.** §7 Work log filled in as the work proceeds.
- **AC-1.** Phase 1 design document exists, fills the §2 matrix, names a
  recommendation with justification, and has Ezequiel's sign-off recorded in §7.
- **AC-2.** Implementation passes the same byte-equivalence discipline as T01 §5.3
  against its own Python reference; round-trip and canonical-invariance tests pass
  on k = 1..8 exhaustively.
- **AC-3.** If the chosen insertion point changes `w*_D`, Theorems 3.13/3.15 and
  Lemma 3.14 are re-proved and the proofs are consistent with T07's rewrite.
- **AC-4.** Either the ablation campaign is complete with a per-axis paired
  comparison (CPDT, effect sizes, CIs), **or** the 2026-08-31 go/no-go passed and
  that is recorded in §7 with the queue state that caused it. Both satisfy AC-4.
- **AC-5.** Promotion decision taken against the §5 Phase 5 rule, with evidence,
  recorded in §7. "Not promoted; C++ results remain the headline" is the expected
  outcome and a fully acceptable one. A negative result is equally reportable.
- **AC-6.** T13 informed of the page budget the chosen option needs.
- **AC-7.** §8 filled.

---

## 7. Work log

_(empty — to be filled by the implementing agent)_

---

## 8. Proposed answer

> No reviewer asked for this. §8 exists because the revision must *disclose* it
> coherently: a reviewer who reads a new encoding in a revision will ask why. The
> draft text below is for the response-letter cover paragraph and, if promoted, for
> a methodology subsection.

### 8.1 Before / after

| Quantity | Unary Σ_SR (submitted) | Gray-coded (revised) | Paired test | Verdict |
|---|---|---|---|---|
| Insertion point chosen | n/a | (a)/(b)/(c) | — | Phase 1 |
| Theorems re-proved | none | | — | |
| ρ, UDFS | | | CPDT | |
| ρ, Bingo | | | CPDT | |
| Mean `\|w*\|`, evolved DAGs | | | Wilcoxon | |
| Canonicalisation cost (ms/DAG) | | | Wilcoxon | |
| `S`, Bingo | | | CPDT | |
| `S`, UDFS | | | CPDT | |
| R² test, UDFS | | | CPDT | |
| R² test, Bingo | | | CPDT | |
| Appendix F: Lev-1 neighbourhood size | | | — | |
| Appendix F: shortest-path behaviour | | | — | |
| **Promoted to main algorithm?** | — | | — | §5 Phase 5 |

### 8.2 Changes made to the manuscript

| File | Lines (revised) | Change |
|---|---|---|
| | | |

### 8.3 Draft response text

```latex
%% cover-paragraph fragment: disclosure of the Gray-coded encoding
```

### 8.4 Residual risk

> Candidates: a reviewer arguing that adding a new encoding exceeds the scope of a
> revision; the avalanche caveat undermining the Appendix F claims; whether the
> re-proved theorems are now *more* terse than the ones R2.1 already objected to;
> whether a promotion on string length alone is over-claimed.
