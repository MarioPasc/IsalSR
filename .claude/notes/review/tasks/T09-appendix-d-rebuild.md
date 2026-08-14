# T09 — Appendix D rebuild and numerical consistency

| Field | Value |
|---|---|
| Reviewer comments closed | **R2.5**, **R2.6**, **R2.3** (and E1, E2, E8) |
| Type | Bookkeeping + reproducibility |
| Owner | **Mario** (+ Claude Code) |
| Depends on | T02 (authoritative campaign), T05 (added problems), T08 (cell counts) |
| Blocks | T13 |
| Status | ✅ **AC-0…AC-8 ALL MET, 2026-08-14, nothing gated.** C2 landed (12,600/12,600, `engine=native` on every cell), so T09 became the **audit** of a numbers pass written by another lane rather than its author — and the audit was not vacuous. 🔴 **Shipped Appendix D.1 documented 42 of 70** while §4.2 promised it listed every problem: R2.5's own defect surviving into the revision. Now **70/70**, and the set-diff is automated (`verify.py`, symmetric difference empty against both per-problem tables). 🔴 **The letter's R2.6 `\changeref` was false in two of three clauses** — the appendix *derived* the run count and D.3 carried no completeness statement; both now true. 🔴 **The k-strata caption claimed "overhead increases with $k$"** — false for both keys (15.2/16.9/12.2 and 16.8/19.6/15.4). 🔴 **Keijzer-6's printed expression was not its target** (42 % error at $x=1$); all 70 expressions audited against the campaign's own $y$ and the audit **persisted** — 36 bitwise-exact, 33 within 3.7 × 10⁻⁹, Keijzer-6 alone genuine — **no campaign number moves**. 🔴 **That audit then falsified a number I had already written into the letter** ("68 exact"); retracted and corrected the same hour, per a pre-commitment made before it ran. AC-4's run count recomputed by me from the 12,600-row ledger, not a formula. AC-7 is a mechanism, not a list: **120 anchored checks**, ambiguity now fails (`'22'` used to pass on `\cite{randall2022bingo}`). Suite **7,810 passed, 5 skipped**; supplementary 18 pp, letter 35 pp, both 0 errors / 0 undefined / 0 overfull. ⚠ **Two live defects handed on, neither mine**: `response_to_reviewers.tex:1904–1912` (T12) tells reviewers the abstract is provisional when it is final — false in three clauses, in a paragraph about candour; and the **double-blind supplementary still documents 42 of 70** (unowned; board says T14). |
| Target | provenance 2026-08-17 · final tables 2026-09-08 |

---

## T16 impact — R2.3 is answered here; the disclosure framing is **CLOSED** (2026-08-04)

> ⚠ **This section is superseded in two places. Read §7's 2026-08-04 entry before
> using anything below.**
>
> 1. **The "OPEN DECISION" below is closed.** It was closed in the response letter,
>    not in this ticket. `response_to_reviewers.tex:1020–1160` is complete R2.3 prose
>    and `:1132–1142` discloses the divergence outright — *"the candidates reaching
>    the canonical map carried fourteen labels rather than the twelve of
>    $\mathcal{L}$ … every result reported in the revision comes from experiments
>    re-executed under $\Sigma_{SR}$."* It also carries an argument this section did
>    not anticipate: the implemented encoding was strictly **finer**, so it separated
>    expressions Σ_SR identifies and never the converse — **no candidate was ever
>    treated as a duplicate of an expression it does not equal.** The defect is one of
>    coverage, not soundness.
> 2. 🔴 **The number table below must NOT be "carried into the continuity table" as
>    its heading instructs.** Those are *randomly generated* graphs. Its own source
>    (`t16_commutative_decomposition.md:98–103`) states: production ρ is **1.7931 /
>    1.880**, not the 1.2960 / 2.1505 tabulated, and the measurement *"establishes
>    direction and invariance, not magnitude."* Use it for **sign and invariance
>    only**; every magnitude comes from C2. Secondary: `61.1 %` is recorded but only
>    reproducible as **59.40 %**, and it is Bingo-only (UDFS is 52.00 %).

**R2.3 is delegated to T16 for its substance and returns here for its wording.**

The technical position, which is now true of the code: **Σ_SR and the host operator
set are different objects.** The host searches over `{+, −, ×, ÷, sin, cos, exp,
log}`; Σ_SR has twelve labels and no `-` and no `/`, because `−` and `÷` enter the
representation through the commutative decomposition `x − y = Add(x, Neg(y))` and
`x / y = Mul(x, Inv(y))`. `Pow` is the only non-commutative operation, which is what
makes Definition 3.9(iv) sound as written. Nothing in Definition 3.2 needs to change.

**OPEN DECISION — how much of the history to disclose. Not settled; do not assume
either way when drafting.** The reviewer asked specifically about the relationship
between Σ_SR and the host operator set, so *some* answer about that relationship is
unavoidable; what is open is whether the answer also states that the submitted
implementation diverged from Definition 3.2 and was corrected. Two considerations
that belong to whoever decides, recorded so the decision is made on facts:

1. **The numbers move visibly.** `k`, canonical string length, canonicalisation cost
   and ρ all change between the submitted tables and the revision (§ below), and
   this ticket's own continuity table exists to explain movement. A reader comparing
   versions will see it.
2. **Ezequiel's stated rationale for fixing the code at all** (T16 §4) was that
   TPAMI acceptance obliges publishing the source, and the first thing a reader does
   is hand the paper and the code to an LLM, which finds a paper/code divergence
   immediately. That argument bears on disclosure as much as on the fix.

**Numbers this ticket must carry into the continuity table** (measured, n=5000,
seed 42, C++ engine):

| Quantity | Bingo | UDFS |
|---|---|---|
| mean `k` | 5.47 → 6.72 (+22.9 %) | 3.27 → 3.99 (+22.0 %) |
| mean canonical length | 21.2 → 26.9 (+27 %) | 11.1 → 13.5 (+22 %) |
| canonicalisation cost | +24.6 % | +10.8 % |
| ρ (random-population probe) | 1.2960 → 1.2960 (exactly invariant) | 2.1505 → 2.1805 (+1.4 %) |

**R², NRMSE and solution recovery do not move** — fitness is computed by the host on
the host's own representation. Production ρ magnitudes come from Wave 1, not from
the probe above.

Full write-up: `docs/md_files/changes/t16_commutative_decomposition.md`.

---

## 1. Why these are grouped

Five defects, one root cause, stated in `verified-discrepancies.md`:

> **Appendix D.1–D.3 was written for the 22-problem arXiv configuration and never
> updated when the suite grew to 50 problems.**

R2.5 (Feynman counts), R2.6 (run count), E1 (k-stratified overhead), E2 (phantom
overhead range) and E8 (ρ range) are all downstream of that single failure to
propagate. R2.3 (Σ_SR vs host operator set) joins them because its fix lives in the
same two places — the Appendix D.2 configuration block and the Section IV.2
inclusion criterion — and because the operator set is one more thing Appendix D
documents incorrectly for the post-22-problem tiers.

Fixing them one at a time would produce five patches to the same appendix.
Rebuilding the appendix from the campaign manifest fixes them structurally and
removes the failure mode.

**Reviewer 2 read the appendices line by line and cross-checked them against the
embedded preprint. Assume they will do so again.** Every number in the revised
appendix must be regenerated from data, not retyped.

---

## 2. Verbatim comments

**R2.5:**
> 5. Section 4.2 refers to a "20-problem subset of AI Feynman," but Table 5 (in the
> Appendix) lists only 10 equations, while Tables 6–7 (in the Appendix) contain 24.
> These counts should be consistent and Table 5 should list all problems used.

**R2.6:**
> 6. Appendix D.2 reports "2 × 2 × (12 + 10) × 30 = 2,640 total runs," but the paper
> uses a 50 problem suite, so the correct count is 6,000. The same error recurs in
> Appendix D.3. Please reconcile and confirm all 50 problems were run with 30 seeds.

**R2.3:**
> 3. Section 4.2 defines Σ_SR as including Pow and √, but Appendix D.2 specifies the
> host operator set as {+, −, ×, ÷, sin, cos, exp, log}, which excludes both.
> Benchmark problems such as Nguyen-8 (√x) and Nguyen-11 (xʸ) require them. Please
> clarify this discrepancy.

---

## 3. Established facts

### 3.1 True composition of the submitted suite (counted from the per-problem tables)

| Family | n | IDs |
|---|---|---|
| Nguyen | 12 | N-1 … N-12 |
| AI Feynman | 24 | I.6.20a, I.10.7, I.12.1, I.12.4, I.13.12, I.14.3, I.15.10, I.16.6, I.25.13, I.29.16, I.30.3, I.34.27, I.37.4, I.39.10, I.44.4, I.48.20, I.50.26, II.3.24, II.11.3, II.11.27, II.11.28, III.10.19, III.14.14, III.17.37 |
| Vladislavleva | 3 | Vlad-2, Vlad-4, Vlad-7 |
| Livermore | 3 | Liv-4, Liv-14, Liv-19 |
| R (Koza rational) | 3 | R1, R2, R3 |
| Pagie | 2 | Pagie-1, Pagie-2 |
| Keijzer | 2 | Keij-6, Keij-11 |
| Korns | 1 | Korns-12 |
| **Total** | **50** | |

The true split is **12 + 24 + 14**, not the "32 core (12 Nguyen + 20 Feynman) + 18
extension" described at `computational_experiments.tex:78–91`.

**The reproducibility consequence is the most substantive item here.**
`computational_experiments.tex:52–54` and `:92–95` both promise that *"The benchmark
tables in Appendix D.1 list every problem with its expression, input dimensionality,
sampling protocol, and source citation."* They do not. Appendix D.1 documents 22.
**28 of the 50 problems have no expression, no variable range and no sampling
protocol anywhere in the submission.** R2 asked for Table 5 to list all problems
used; the honest scope is larger than they realised.

### 3.2 Run count
`supplementary.tex:560–561` and `:574` both state 2,640, which is
`2 × 2 × (12 + 10) × 30` — the 22-problem arXiv suite.
`2 × 2 × 50 × 30 = 6,000`. R2's arithmetic is correct. Note the main text's cell
counts do not reach 6,000 either (T08, E4).

### 3.3 Σ_SR versus the host operator set — **four** sets, not two
The manuscript contains four different operator sets and never says they are
different objects:

| Source | Set | Role |
|---|---|---|
| `computational_experiments.tex:63–67` | Σ_SR = {+, ×, Neg, Inv, sin, cos, exp, log, √, \|·\|, Pow, Const} | **encoding alphabet**, used as the benchmark *inclusion criterion* |
| `supplementary.tex:557–559` | {+, −, ×, ÷, sin, cos, exp, log} | **host solvers' search primitives** |
| `methodology.tex:965–967` | experiments "exclude Pow" (k_∧ = 0) | isomorphism-condition simplification |
| `supplementary.tex:747–748` | {+, ×, ∧, sin, cos, exp, log, neg, inv} | the *synthetic* benchmark operator set |

R2 read the first two as the same object, which is a reasonable reading of the text
as written. **Additionally, the second is factually wrong for the post-22-problem
tiers**: `CLAUDE.md` records that the hard configs extend the operator set with
`sqrt` (and `pow` for Bingo only — UDFS's vendored search has no generic `pow`),
because Pagie-1, I.15.10, I.37.4 and III.17.37 are otherwise structurally
unsolvable. Verify against the actual YAML configs and report the per-tier operator
set as it was really run. This is not only an ambiguity; part of it is an error.

**On Nguyen-8 and Nguyen-11 specifically**: N-8 = √x on [0,4]; N-11 = xʸ on [0,1]².
Both ranges are non-negative, so both are expressible from {exp, log, ×}
(√x = exp(½ log x), xʸ = exp(y log x)). Both solve empirically to R² = 1.0000 under
both methods. **The results stand; only the description is incomplete.** Say exactly
this — it is a complete answer and it costs nothing.

### 3.4 E1 — k-stratified Bingo overhead disagrees between main text and appendix

| Source | k < 5 | 5 ≤ k < 15 | 15 ≤ k < 32 |
|---|---|---|---|
| `results.tex:177–179` | 38.5 % | **45.9 %** | **41.6 %** |
| `supplementary.tex:720–722` (Table 8) | 38.5 % | **47.0 %** | **49.9 %** |

Two of three buckets disagree and the *shapes* differ: the appendix is monotone
increasing, the main text is non-monotone. The main text cites Table 8 explicitly as
its source. R2 checked every other number in these tables; this pair survived by
chance.

### 3.5 E2 — a range that exists nowhere
`supplementary.tex:734` asserts *"35.5–56.0 % total overhead reported for Bingo in
the main text"*. The main text reports a 39.2 % median (`results.tex:176`, Table 2).
No such range appears anywhere in the main paper.

### 3.6 E8 — ρ range understated
`discussion.tex:10–11` states *"The observed ρ values, 1.45–1.96 across the
50-problem suite"*. The per-problem tables give UDFS ρ ∈ [1.11, 1.98] and Bingo
ρ ∈ [1.57, 1.96]; the union is **[1.11, 1.98]**. The supplementary states the
per-method ranges correctly (`:599`, `:620–621`); only the discussion is wrong.

---

## 4. Mandatory reading

- `.claude/notes/review/source/reviewer-2.md` — §R2.3, §R2.5, §R2.6
- `.claude/notes/review/source/verified-discrepancies.md` — D1, D2, D7, E1, E2, E8,
  and the Aggregate view at the end (the root-cause statement)
- `.claude/notes/review/source/manuscript-map.md` — Appendix D.1–D.5 structure,
  Tables 4–9, and the hardcoded cross-reference inventory
- `.claude/notes/review/tasks/T02-cpp-reexecution-campaign.md` — §5.3 provenance discipline
- `.claude/notes/review/tasks/T05-benchmark-extension.md` — added problems need the same treatment
- `.claude/notes/review/tasks/T08-nan-and-paired-test-integrity.md` — cell counts
- `docs/md_files/design/experimental_design/data_benchmarking_design.md`
- `docs/md_files/changes/{hard_problem_selection_rationale,candidate_problem_screening,roundoff_problem_selection}.md`
  — these already contain the expressions, ranges and sampling protocols for the 28
  undocumented problems; the information exists, it just never reached the appendix
- `benchmarks/datasets/{nguyen,feynman,hard,cherrypicked,roundoff}.py` — the
  authoritative definitions

---

## 5. Work specification

### 5.1 Generate, do not retype
Build a script that emits the Appendix D.1 benchmark tables **directly** from
`benchmarks/datasets/*.py` and the campaign `MANIFEST.json`: problem ID, expression,
dimensionality, variable ranges, train/test sizes, sampling protocol, source
citation, and the tier it belongs to. Commit the script. Every future suite change
then propagates automatically, which is the structural fix for the root cause.

### 5.2 Documentation completeness
Every problem in the final suite (50 + T05's additions) documented. Zero undocumented
problems. Verify by diffing the generated table's ID list against the per-problem
results tables' ID list — they must be identical sets.

### 5.3 Reconcile the suite description
Rewrite `computational_experiments.tex:78–91` to describe the true composition and
the true tier structure. Delete "20-problem subset" and "32 core + 18 extension".

### 5.4 Run count
Recompute from the manifest, not from a formula. State the count, the factorisation,
and — as R2 explicitly asked — confirm that all problems were run with 30 seeds, or
enumerate the exceptions (T08).

### 5.5 Operator sets
Add a short passage distinguishing the four objects in §3.3. State the per-tier host
operator set as actually configured, including the `sqrt`/`pow` extensions. Add the
N-8 / N-11 explanation from §3.3. Reconcile `methodology.tex:965–967`'s "exclude
Pow" with whatever the configs really did.

### 5.6 Single-source every cross-document number
E1, E2 and E8 all exist because a number was typed in two places. For every number
that appears in both the main text and the supplementary, generate both occurrences
from the same analysis artefact, or state one and reference the other. Produce a
**numerical audit list**: every numeric claim in `results.tex`, `discussion.tex`,
`computational_experiments.tex` and `supplementary.tex`, with its source artefact
and a pass/fail check. This list is also the round-2 insurance policy.

---

## 6. Acceptance criteria

- **AC-0.** §7 Work log filled in as the work proceeds.
- **AC-1.** Appendix D.1 table generator committed and reproducible from the repo.
- **AC-2.** 100 % of suite problems documented with expression, dimensionality,
  range, sampling protocol and citation. Verified by set-diff against the results tables.
- **AC-3.** Suite composition described correctly in Section IV.2; the "20-problem
  subset" and "32 + 18" framings are gone.
- **AC-4.** Run count recomputed from the manifest and consistent everywhere it appears.
- **AC-5.** The four operator sets distinguished; per-tier host operator set stated
  as actually run; N-8/N-11 explained; `methodology.tex:965–967` reconciled.
- **AC-6.** E1, E2, E8 resolved; the k-stratified table appears once and is
  referenced, not duplicated.
- **AC-7.** Numerical audit list produced and fully passing.
- **AC-8.** §8 filled.

---

## 7. Work log

### 2026-08-04 — session opened against a partial gate; plan and scope decision

**Gate.** `EXECUTION-PLAN.md` §10.4 lists T09 under *"Do not spawn — manuscript-side,
consumes C2's output"*. That is true of the numbers and false of the structure, and
the two halves separate cleanly:

| Half | Status | Reason |
|---|---|---|
| Problem **definitions** (AC-1, AC-2 generation, AC-3, AC-5) | **not gated** | T05 froze D2 at 20 problems, registry resolves **70/70**, draw pinned at commit `0e4a573`. C2 changes results, never definitions. |
| Result **numbers** (AC-4, AC-6 values, AC-7 pass, AC-8) | **gated on C2** | T02 `NOT STARTED`; Stage D in pre-flight; results target 2026-09-03. |

**Mario's scope decision (this session): structural half + manuscript prose.** Build
the generator and the audit harness, and carry the prose corrections into the
manuscript now, accepting a second editing pass when C2's numbers land. Numbers are
not to be placeholdered — T09 exists *because* numbers were typed rather than
generated, and a placeholder row is the same failure mode.

**Mario's second decision: the T16 disclosure framing stays OPEN pending a currency
re-check.** The ticket's §T16-impact block dates from 2026-07-30 and
`response_to_reviewers.tex` has grown to 92.6 kB since; the framing is not to be
chosen from a stale premise. Investigation 1 below establishes the current state
first.

**Finding before any delegation — AC-5 is larger than §3.3 states, and §5.5 is stale.**
Dumping the operator set from all 14 campaign configs directly:

| Host | Operators | n | Uniform across the 7 suites? |
|---|---|---|---|
| Bingo | `+ − × ÷ sin cos exp log sqrt pow` | 10 | yes |
| UDFS | `+ − × ÷ sin cos exp log sqrt neg inv` | 11 | yes |

Three consequences:

1. `supplementary.tex:557–559` states a single 8-operator set for everything. It is
   wrong three ways, not one: it omits `sqrt` (**both** hosts), omits `pow` (Bingo),
   and asserts a **global** set where the configuration has always had a **per-method**
   one. R2.3 caught the first; the third is unraised.
2. **The fifth operator set §8.4 listed as a residual risk exists and is now found.**
   The manuscript's four objects are really five: Σ_SR, *Bingo's* host set, *UDFS's*
   host set, the `methodology.tex:965–967` "exclude Pow" simplification, and the
   synthetic-benchmark set. UDFS has no generic `pow` — its vendored search cannot
   express it — so the two host sets are not reconcilable into one.
3. **§5.5's instruction to "state the per-tier host operator set" is stale for the
   revision.** A4b (`EXECUTION-PLAN.md` §11.2, 2026-08-03) made the set uniform across
   tiers and C2 re-runs every affected cell under it. Per-tier variation is a property
   of the **submitted** campaign only, and belongs in the response letter as history,
   not in the revised appendix as configuration. This makes the R2.3 answer simpler:
   one set per host, stated once.

**Plan.**

| # | Kind | Deliverable | Acceptance check |
|---|---|---|---|
| 1 | investigate | Currency audit: what `response_to_reviewers.tex` already says about R2.3/R2.5/R2.6 and the alphabet; whether §T16-impact's numbers are still current; what the code does today | A per-comment status table + a yes/no on each §T16-impact number |
| 2 | investigate | Ground truth of the **submitted** suite: exact problem-ID list per appendix table, and the line ranges for AC-3/AC-5/AC-6 edits | ID lists that reproduce or refute §3.1's 12+24+14 |
| 3 | implement | `experiments/scripts/generate_appendix_d_tables.py` — emits D.1 from `benchmarks/datasets/*.py` for all 70 problems (AC-1, AC-2) | Regenerates 70/70 rows with expression, dim, ranges, sizes, protocol, citation, tier |
| 4 | implement | Numerical-audit harness (AC-7): every numeric claim in the 4 `.tex` files bound to a source artefact with a pass/fail check | Inventory complete; runs and reports; passes deferred to C2 |
| 5 | **me** | Manuscript prose: AC-3 suite composition, AC-5 operator sets, AC-6 structural single-sourcing | Compiles; "20-problem subset" and "32 + 18" gone |

1 and 2 are read-only and run concurrently. 3 and 4 follow, serialized against the
Stage D agent's lane (`slurm/`, `experiments/models/`, `experiments/configs/`,
`src/isalsr/` are **not** mine this session).

### 2026-08-04 — §T16-impact is stale in both directions; the disclosure decision was already taken

**The "OPEN DECISION" at the head of this ticket is closed, and it was closed in the
response letter, not here.** `response_to_reviewers.tex:1020–1160` carries complete
R2.3 prose, and `:1132–1142` discloses in full:

> *"the candidates reaching the canonical map carried fourteen labels rather than the
> twelve of $\mathcal{L}$ … Both routines now apply the decomposition, and every
> result reported in the revision comes from experiments re-executed under
> $\Sigma_{SR}$ as Definition~3.2 defines it."*

The letter also makes an argument this ticket did not anticipate and which is
stronger than either framing offered: the implemented encoding was strictly
**finer** than the defined one, so it separated expressions Σ_SR identifies and never
the converse — **no candidate was ever treated as a duplicate of an expression it
does not equal.** The defect is therefore one of *coverage* (the experiments did not
exercise the alphabet Section 3 is stated over), not of *soundness*. Nothing further
is open. R2.2 (`:981–1017`) is also complete and defers to R2.3 by design.

🔴 **But §T16-impact's number table must not be used as this ticket instructs.** It is
headed *"Numbers this ticket must carry into the continuity table"*. Its own source,
`docs/md_files/changes/t16_commutative_decomposition.md:98–103`, says the opposite:

> *"These are randomly generated graphs, not evolved live-search populations. ρ here
> (Bingo 1.2960, UDFS 2.1505) is **not** production ρ (1.7931, 1.880) … This
> measurement establishes direction and invariance, not magnitude. Magnitudes come
> from Wave 1."*

Carrying 1.2960 / 2.1505 into a continuity table as production deltas would put a
wrong number in front of the one reviewer who checked every number in round 1. The
`k` and canonical-length deltas carry the same caveat. **Corrective rule for whoever
writes the continuity table: §T16-impact establishes sign and invariance only; every
magnitude comes from C2.** Secondary: the `61.1 %` Sub/Div incidence is recorded at
`:27` but is only *reproducible* as **59.40 %** (`:95–96`), and it is Bingo-only —
UDFS is 52.00 % (`:151–153`).

**Code state re-verified today** (all four): `decompose: bool = True` in both adapters
(`bingo/adapter.py:71`, `udfs/adapter.py:92`); `SHARE_DECOMPOSED_UNARY = False`
(`commutative_encoding.py:76`); `NodeType.SUB`/`DIV` retained (`node_types.py:39–40`);
no caller outside the two adapters.

🔴 **R2.6 — this ticket's own comment — is an empty `\todoblock` at
`response_to_reviewers.tex:1246`.** R2.5 (`:1176–1232`) is complete prose but says
nothing about the alphabet. R2.4 (`:1172`) is also an empty skeleton (T11's, flagged
in passing). So T09 owes the letter one response block, gated only on the run count.

### 2026-08-04 — submitted-suite ground truth, verified in the sources

Confirmed by direct read, line numbers corrected against this ticket's §3 (three had
drifted):

| Claim | Verdict |
|---|---|
| Appendix D.1 documents 22 of 50 | **CONFIRMED.** `supplementary.tex:455–536`; `tab:nguyen` 12 rows (`:487–511`), `tab:feynman` 10 rows (`:513–536`). |
| True composition is 12 + 24 + 14 = 50 | **CONFIRMED** from the union of `table_supplementary_{udfs,bingo}.tex` (50 rows each, identical ID sets *and* identical order). |
| "core (32) = 12 Nguyen + 20-problem subset of AI Feynman" | **CONFIRMED WRONG**, `computational_experiments.tex:79–81`. Drawn by family the true line is 36 + 14, not 32 + 18. |
| E1 — main text vs appendix disagree | **CONFIRMED.** `results.tex:178` said 38.5/45.9/41.6; `tab:k_range_overhead:720–722` says 38.5/47.0/49.9. **New detail that decides the fix:** the appendix is internally consistent — its own prose (`:729–732`) reads the table as monotone. Only the main text is wrong. |
| E2 — "35.5–56.0 %" exists nowhere | **CONFIRMED.** `grep "35\.5"` over all seven `paper/*.tex`: **0 hits**. |
| E8 — ρ range | **CONFIRMED.** `discussion.tex:10` said 1.45–1.96; the supplementary's own prose already says [1.11, 1.98] (`:599`) and [1.57, 1.96] (`:620`). The discussion contradicted its own appendix. |

**Also note for T08/T05**: `tab:feynman` lists `I.34.27`, the problem later found to be
byte-identical to `I.12.1` and corrected on 2026-08-04. It is one of the 10 problems
Appendix D.1 *did* document, and it was documented wrongly.

### 2026-08-04 — AC-6 structural half applied (E1, E2, E8)

Three surgical edits, **6 insertions / 8 deletions**, in the live Overleaf checkout.
**Not pushed.** All three are permanent under C2: after them the main text and the
discussion restate no appendix number at all, so the numbers pass cannot reintroduce
the defect.

| # | File | Change |
|---|---|---|
| E1 | `article/paper/results.tex:176–179` | Deleted the three restated k-stratified percentages; the sentence now reads *"the median overhead is $39.2\%$; Table~8 in the appendices reports the $k$-stratified breakdown."* The table is stated once and referenced. |
| E2 | `article/supplementary/supplementary.tex:733–734` | Deleted the phantom *"$35.5$--$56.0\%$ total overhead reported for Bingo in the main text"* and the false attribution with it; the sentence's argument is preserved. |
| E8 | `article/paper/discussion.tex:10–11` | Deleted the incorrect $1.45$--$1.96$ range and the duplicated per-method means; ranges are now referenced to the appendices, which compute them. |

**Deliberately not changed yet, and why.** `discussion.tex` still says "the suite"
where the count will become 70, and `supplementary.tex` still says 50/2,640. Changing
one occurrence of a count without changing all of them is precisely the failure mode
this ticket exists to remove. Counts move in one pass, driven by the AC-7 audit list,
once C2 fixes them.

**Cross-reference note for T11**: `results.tex:176` still hardcodes "Table~8 in the
appendices" because the supplementary is a separate document and `\ref` cannot
resolve across it. That is R2.4's defect class and belongs to T11, not here.

**Verified after editing** (scratch copy, live checkout untouched, **not pushed**):
`main.tex` compiles exit 0, **0 errors, 12 pages** — still exactly at the hard limit,
so T13's page budget is unaffected. `supplementary.tex` compiles exit 0, 0 errors,
10 pages. Both page counts unchanged from the submission.

### 2026-08-04 — AC-7 scoped, and the deletion strategy does not generalise

Measured the audit surface directly. **2,386 numeric literals** across the six files:

| File | prose | table cells |
|---|---|---|
| `computational_experiments.tex` | 71 | 0 |
| `results.tex` | 126 | 82 |
| `discussion.tex` | 59 | 0 |
| `supplementary.tex` | 313 | 307 |
| `table_supplementary_{udfs,bingo}.tex` | 0 | 1,428 |
| **total** | **569** | **1,817** |

**43 literals appear in more than one of the four narrative files.** Some are regex
artefacts (`13.12`, `16.6`, `48.20` are Feynman *problem IDs*, not measurements), and
the audit tool must discriminate — but the real ones are exactly the E1/E2/E8 class:
`0.28`/`0.82` (canonicalisation times), `1.56`/`1.83` (ρ means), `1.07` (S), `240`
(training-set size), `50`, `22`, `30`.

🔴 **My own E8 edit exposed the limit of deleting duplicates.** Removing the ρ means
from `discussion.tex:10–11` did not remove them from the document: `:81` restates
*"The higher $\rho$ in Bingo than in UDFS ($1.83$ vs.\ $1.56$)"*, and `:22` restates
`0.28`/`0.82` from `results.tex`. **Deleting those too would be wrong** — unlike E1
and E2, those sentences carry arguments that need the comparison, and stripping them
would trade a consistency defect for a vaguer paper, which R1 and R3 would both
object to.

**Design decision (mine) — AC-7 becomes a mechanism, not a list.** §5.6 offers two
options: *"generate both occurrences from the same analysis artefact, or state one and
reference the other."* Deletion implements the second and does not scale past E1/E2.
The first is what the remaining ~40 need:

1. **`numbers.tex`** — a generated macro file (`\newcommand{\rhoMeanBingo}{1.83}`, …)
   emitted by the analysis pipeline and `\input` by **both** documents. A number is
   then defined once and used freely, so prose stays quantitative and "typed in two
   places" becomes structurally impossible rather than merely audited.
2. **`numerical_audit.py`** — extracts every numeric literal, reports which are still
   hardcoded rather than macro-referenced, and binds each remaining one to its source
   artefact with a pass/fail check.

This is strictly stronger than the ticket's "numerical audit list", and it is the
round-2 insurance policy §5.6 asks for. The macro file cannot be *populated* until C2
lands; it can be built, wired and tested now against the submitted numbers.

### 2026-08-04 — AC-1 and AC-2 met; the generator immediately found a defect nobody had raised

`experiments/scripts/generate_appendix_d_tables.py` + `tests/unit/test_appendix_d_generator.py`,
emitting `docs/generated/appendix_d/{appendix_d_tables.tex,appendix_d_benchmarks.json}`.
**Re-verified by me in the main tree**, not taken on the agent's word:

| Check | Result |
|---|---|
| Rows emitted | **70/70**; tiers 12/10/10/10/8/6/14 |
| Nine mandatory fields non-empty | **70/70, zero gaps** |
| Tests | **51 passed** |
| `ruff` / `mypy --strict` | clean (55 source files) |
| Determinism | re-run byte-identical |
| Citation resolution | 69/70 against `references.bib`; the 70th needed a new entry |

Citation tally cross-checks **exactly** against the independent manuscript audit:
`udrescu2020` × 30 (= 24 submitted Feynman + 6 D2 remainder), `uy2011` × 12,
`strogatz1994` × 14, `petersen2021` × 6 (= Liv-4/14/19 + R1/R2/R3),
`vladislavleva2009` × 3, `pagie1997` × 2, `keijzer2003` × 2, `korns2011` × 1.
Two independent routes to 70 agreeing is the check that matters.

🔴 **PREMISE-FALSE returned, and it was right — I had the train/test sizes wrong.**
My brief asserted sizes come from the `generate_data` defaults. They do not:
`orchestrator.py:644–645` reads `train_size`/`test_size` from each config's
`benchmarks:` block, with the `generate_data` defaults never reached. **Nguyen is
240/1000, not 20/100; Feynman is 1000/250, not 160/40.** Had this gone unchallenged,
the rebuilt appendix would have documented sizes the campaign never ran — the exact
failure the ticket exists to remove, reintroduced by the fix. Recorded as the clearest
argument in this ticket for generating over retyping.

🔴 **New unraised defect, found *by* generating: `computational_experiments.tex:94–95`
stated training sizes are $\{20, 240, 1000\}$. No problem in the suite uses 20.** The
realised set is {50, 100, 240, 300, 676, 1000, 1024, 2000} train and {100, 120, 221,
250, 1000, 1200, 2000, 2500, 5000} test. `20` is the orchestrator's *fallback* default
(`bench_cfg.get("train_size", 20)`), i.e. a value that appears only when a config omits
the key — which none does. Same class as R2.5, and neither R2 nor this ticket had it.

**Independent confirmation of the C4 fix**: the generated table renders `I.34.27` as
$\frac{x_0x_1}{2\pi}$ against `I.12.1`'s $x_0x_1$, so the restored $1/(2\pi)$ is live
in the definitions and the two problems are no longer byte-identical.

**Filed, deliberately not fixed — Stage D is being submitted and these touch benchmark
definitions:**

| Issue | Location |
|---|---|
| LHS leaked into the expression string (`"Ef = q1 * r / …"`, `"flux = Pwr / …"`) | `benchmarks/datasets/feynman.py`, `I.12.4` and `II.3.24` |
| English annotation inside an expression (`"[written cos(y)*cos(x)/sin(y)]"`) | `benchmarks/datasets/strogatz.py`, `Strogatz-shearflow1` |
| Problem id `test_4` is non-descriptive and would print verbatim in a TPAMI appendix | `benchmarks/datasets/feynman_remainder.py` |
| Power notation inconsistent (`^` vs `**`) across modules | all dataset modules |

The first three do **not** reach the appendix — the generator renders from
`sympy_expression` (58/70) and parses only Nguyen's strings — so these are repo
hygiene, not manuscript defects. **`test_4` must get a display name before the
appendix ships**, and an ID rename is unsafe while the campaign is in flight.

### 2026-08-04 — AC-3 and AC-5 applied to the manuscript

All counts below are read out of `appendix_d_benchmarks.json`, none hand-tallied.
(I first wrote "eight families" from memory and the generated tally corrected it to
**nine** — a small demonstration of the same point.)

| AC | File | Change |
|---|---|---|
| AC-3 | `computational_experiments.tex:48–55` | "assembled in three stages: a $32$-problem core … an $18$-problem extension" → a $70$-problem suite with no post-hoc filtering; the Appendix D.1 promise now also names sampling domain and train/test sizes, and states the tables are generated from the executed definitions. |
| AC-3 | `computational_experiments.tex:78–95` | Composition rewritten over **nine** families — Nguyen 12, AI Feynman 30, ODE-Strogatz 14, Vladislavleva 3, DSO-Livermore 3, Koza rational 3, Pagie 2, Keijzer 2, Korns 1 = 70. "20-problem subset" and "32 core + 18 extension" are gone. Protocol tally (53 uniform / 14 fixed / 3 grid) and the true size ranges replace the false $\{20, 240, 1000\}$. |
| AC-5 | `supplementary.tex:557+` | New **Operator sets** paragraph. States Bingo's 10 and UDFS's 11 separately, notes UDFS has no generic power operator, states each is uniform across the suite, and distinguishes Σ_SR as a property of the *representation* from the hosts' *search* primitives — with the composition identities that explain why neither set contains the other. |
| — | `references.bib` | Added `strogatz1994` (Strogatz, *Nonlinear Dynamics and Chaos*, Addison-Wesley 1994). Required: the 14 ODE-Strogatz problems had no resolvable citation. |

**Deliberately left wrong**: the `2 \times 2 \times (12+10) \times 30 = 2{,}640` run
count at `supplementary.tex:560–561` and `:574`. AC-4 requires it be recomputed **from
the manifest, not from a formula**, and the C2 manifest does not exist. I split the
sentence so the operator-set claim no longer depends on it, but I did not substitute a
design figure — writing 8,400 now would be exactly the retyped-formula error R2.6
caught, with a different number.

**Verification, re-run by me after every edit** (scratch copy; live checkout **not
pushed**): `main.tex` compiles exit 0, **0 errors, 0 undefined references, 12 pages**;
`supplementary.tex` exit 0, 0 errors, 10 pages. Both page counts **unchanged from the
submission** despite the added operator-set paragraph, so T13's 12-page constraint is
not disturbed. The project ships a pre-built `main.bbl`, so `references.bib` alone was
not enough — `bibtex` was re-run and the regenerated `main.bbl` installed.

**For T05/Karl**: I cited `strogatz1994` as the primary origin. The ODE-Strogatz
*benchmark set* is distributed via SRBench (`lacava2021`, already in the bib); if the
D2 justification prose wants the distribution provenance too, cite both there.

### 2026-08-04 — AC-7 harness built, and it immediately caught a defect I had just introduced

`experiments/scripts/numerical_audit.py` + `tests/unit/test_numerical_audit.py`,
emitting `docs/generated/audit/numerical_audit.{json,md}`. **Re-verified by me**:
**53 tests pass**, `ruff` and `mypy --strict` clean, regeneration deterministic.

| Quantity | Value |
|---|---|
| Literals inventoried across the six files | **2,425** |
| `measurement` / `problem_id_fragment` / `structural_count` / `cross_reference` / `year` / `typography` | 2,101 / 135 / 61 / 76 / 0 / 52 |
| Measurements bound to an artefact | 14 |
| **Measurements UNBOUND** | **2,087** |
| Cross-file duplicate groups (all six files) | 126 |
| Cross-file duplicate groups (four narrative files) | **34** |

It independently rediscovered all five duplicates I had found by hand (`0.28`, `0.82`,
`1.56`, `1.83`, `1.07`), which is the check that it is not vacuous. Three lexer bugs
were caught **red before fixing**, the most instructive being that a leading-dot
alternative split `I.48.20` into `.48` + `.20`, destroying the problem-id token — the
same over-counting my own throwaway regex had committed.

🔴 **It caught a defect in my own AC-3 edit, within the hour.** My rewrite set
`computational_experiments.tex:48` to *"The $70$-problem suite"* while `:9` still read
*"$50$ benchmark problems drawn from **eight** published suites"* — against the
**nine** families I had just written at `:79`. **Two contradictory suite sizes and two
contradictory family counts in one file, introduced by the fix for contradictory
counts.** This is the ticket's own thesis demonstrated against the ticket's own author,
and it is the strongest evidence in this session that AC-7 had to be a mechanism
rather than a list: a human re-reading would plausibly have missed it, and R2 would
not have.

**Repaired**: `:9` now reads *"$70$ benchmark problems drawn from nine published
suites"*. Re-verified — the only surviving `50` in a suite-size sense across all four
narrative files is `$N = 50$` at `:166` and `:237`.

**Deliberately left, and registered as the head of the numbers-pass queue.** Those two
are the CPDT sample size in the statistical-testing subsection, not a description of
the suite: `N` is the number of problems entering the paired test, which **T08 owns**
(true `N` per metric) and which C2 fixes. `:237` additionally ties an accuracy claim
(*"accurate within $10^{-3}$"*) to `N`, so it cannot be retyped without recomputation.
The document is therefore coherent in the state the whole manuscript is in —
**descriptions updated, results pending** — rather than half-migrated.

**The UNBOUND count is the honest measure of remaining exposure.** 2,087 of 2,101
measurements have no source artefact, because C2's artefacts do not exist. That number
is the work AC-7 has left, and it is now countable instead of guessed.

**Two corrections to the harness's own report**, checked by me:
- It flags `240` as a cross-file duplicate; both occurrences are in `supplementary.tex`
  (`:463`, `:491`). My brief was wrong to list it as cross-file. Correctly surfaced
  under `within_file_repeats`.
- `year` has zero members because every four-digit year in these files sits inside a
  `\cite{}` key and classifies as `cross_reference`. Correct, not a gap.

**Final verification after every edit** (scratch copy, live checkout **not pushed**):
final-pass `main.tex` **0 errors, 0 undefined references, 12 pages**;
`supplementary.tex` **0 errors, 10 pages**. Both unchanged from the submission.

### 2026-08-06 — the annotated supplementary brought level with the response letter

**The letter promised supplementary content that did not exist, and every results
float still carried superseded C1 numbers.** Worked only in
`reviews/internal_copy_reviewed_article/supplementary/`; `article/` untouched
(`git status` clean at start and unchanged under `article/` at end).

Written into Appendix D (blue): the 70-problem, nine-source composition with the
44/130 ground-truth-track count; the **pre-registered selection rule** for the 20
added problems, which R3.1 promises and which was nowhere in the document (pool
`|E| = 92`, seed `2 547 107 438` derived from the sorted eligible identifiers, no
stratification, nothing removed post-draw); the three-arm campaign shape and its
multiplicity consequence (Holm over three contrasts, Friedman/Nemenyi over three
arms).

**New Appendix E, `The naive hash comparator`** — R1.4's `\changeref` names
`Appendix~E` twice, and `paper/results.tex:187` already pointed there for the
per-problem breakdown, so the letter and the parallel agent agreed on the letter
E before this appendix existed. Contains σ and the key computed from the *host's
own* stored structure, Lemma E.1 (soundness) with proof, an explicit
incompleteness counter-example on `sin(x₀)+cos(x₁)` with φ = (2 3), the Δ/φ
definitions and the ρ-form identity, the one-stream requirement, and the
HyperLogLog protocol at m = 2¹⁶ (0.41 % s.e.).

🔴 **Appendix lettering shifted: E→F, F→G, G→H.** This breaks three references in
`paper/discussion.tex` (`:35` E.2, `:57` E.1, `:138` E.1) which must become F.2,
F.1, F.1. **Not fixed here — `paper/` belongs to the parallel agent.** It also
makes the letter's R2.4 location "Appendices F.2 and G" stale (now G.2 and H).

| Check | Result |
|---|---|
| Labels before → after | 39 → 48; **0 lost**, 9 added (all `sec:supp_hash*`, `lem:hash_sound`, `tab:supp_hash_phi`) |
| Compile | 0 errors, 0 undefined refs, 0 undefined citations |
| Overfull hboxes | **0** — improved on HEAD, which had 1 (10.32 pt, `tab:synthetic_scalability`) |
| `color{red}` / `[MPG` | 0 / 0 (pending scaffolding uses a named colour `pendingred`) |
| Pages | 13 → 14 |
| Placeholders | 28 `\pendingnum` + 6 `\pendingblock` = 34, all ledgered |
| Supplementary→main cross-refs | 19 walked, **19 pass** — all are content references (Definition 3.x, Theorem 3.15) or number-free descriptions, and all sit in Appendix A, which was not edited |

**Deliberately not changed**: the run count stays a placeholder, not `8,400`. AC-4's
rule is honoured — the realised count comes from the C2 manifest. The seed count is
*also* a placeholder, because the letter says 20 and the live campaign says 30 (see
§8.2).

**Left alone as campaign-independent**, after checking each: `tab:supp_shortest_path`,
`fig:shortest_path`, `tab:supp_neighbourhood`, `fig:neighbourhood`. These are
Levenshtein distances and neighbourhood counts computed from the canonicaliser on
fixed example expressions; none uses Sub or Div, so T16's decomposition does not
move them, and none derives from a benchmark run. The synthetic ρ = k! and 100 %
invariance rows were kept as real values for the same reason — they are properties
of the construction — while only that table's **timing column** was placeholdered,
since the engine was re-timed.

### 2026-08-14 — the gate opened, and the numbers landed from a different lane

**C2 completed on 2026-08-14** — 12,600/12,600 cells, certifier **GO** (19/19),
commit `2dd56fd` / tag `campaign/c2`, engine `native` on every cell. Everything
T09 recorded as **⧗C2** is therefore unblocked, but not by this ticket: repo
commit `3c5e91f` built `experiments/scripts/review_campaign/` (an eight-step
pipeline ending in `verify.py`) and manuscript commit `1f9a77d` carried its
outputs into all five documents.

**This changes what T09 is.** Its structural half is done and its numbers half
was executed elsewhere, so the remaining work is **audit, not authorship**:
T09 owns cross-document numerical consistency (AC-4, AC-6, AC-7), and a numbers
pass performed by another lane is exactly the event AC-7 exists to check. Two
facts already establish that the audit is not vacuous:

- 🔴 **The letter now makes three claims about the appendix that T09 must verify
  are true in the appendix.** `response_to_reviewers.tex` R2.6 is **complete
  prose** (no longer the empty `\todoblock` recorded above) and its `\changeref`
  asserts *"Appendix D.2 and D.3, the run count restated as
  $2 \times 3 \times 70 \times 30 = 12{,}600$ and sourced from the campaign
  ledger; Appendix D.3, the completeness statement and the per-arm cell counts"*.
  A claim typed in one document about another is this ticket's own failure mode.
- 🔴 **`docs/generated/appendix_d/appendix_d_tables.tex` is `\input` by nothing.**
  `supplementary.tex` has exactly four `\input`s and none is the generated
  appendix. The supplementary does now discuss the 14 ODE-Strogatz problems in
  prose (`:643–815`), so either D.1 was re-typed by hand — reintroducing the root
  cause — or it does not cover 70/70. AC-2 is met in `docs/generated/`; whether it
  is met *in the shipped document* is open.

**Plan.**

| # | Kind | Deliverable | Acceptance check |
|---|---|---|---|
| 1 | investigate | Currency audit of the numbers pass against the four open ACs: every stated run count and its provenance; whether shipped Appendix D.1 covers 70/70 and whether its rows are generated; E1/E2/E8's post-pass values and single-sourcing; truth of the R2.6 `\changeref`'s three claims | Per-AC status table with file:line evidence; yes/no per letter claim |
| 2 | investigate | State of the two audit harnesses: does `review_campaign/verify.py` run and pass today; does `numerical_audit.py` still run against the current documents and what is its UNBOUND count now; overlap or complement | Both exit codes and headline counts, plus the division of labour in one paragraph |
| 3 | **me** | Close whatever 1 and 2 expose; `test_4` display name; §8 filled with real numbers | Both documents compile; `verify.py` exits 0 |

1 and 2 are read-only and run concurrently.

### 2026-08-14 — AC-4 met: the run count recomputed from the ledger, by me

AC-4 says "recomputed from the manifest". **There is no `MANIFEST.json`** — T02's AC-3
is open on exactly that. `…/c2_3arm/status_ledger.csv` is the better artefact for this
purpose anyway: a manifest states intent, the ledger states what executed. Recomputed
directly from its 12,600 rows, nothing taken from any agent or document:

| Quantity | Ledger |
|---|---|
| Rows (= runs) | **12,600**, and $2 \times 3 \times 70 \times 30 = 12{,}600$ |
| `terminal_status` | `completed` on 12,600 / 12,600 |
| `exit_code` | `0` on 12,600 / 12,600 |
| Cells per (host, arm) | **2,100**, on all six blocks — exactly balanced |
| (host, arm, problem) groups | **420**, each with exactly **30** distinct seeds (1…30) |
| `engine` | `native` on **12,600 / 12,600** — the C++ engine on every cell |
| `git_commit` | **1** distinct (`2dd56fd7…`) |
| `config_sha256` | **14** = 2 hosts × 7 suites |
| Cells with any NaN metric | **0** |
| Suite sizes | nguyen 12, feynman 10, hard 10, cherrypicked 10, roundoff 8, feynman\_remainder 6, strogatz 14 = **70** |

R2.6 asked two things. Both are now answerable from data: the count is 12,600, and
**every problem ran at all 30 seeds with no exceptions** — unlike the submitted
campaign, where 35 Bingo cells were missing (T08).

### 2026-08-14 — three live defects found, all verified by me before acting

🔴 **1. Shipped Appendix D.1 documents 42 of 70 problems.** This is R2.5's own defect
surviving into the revision with different numbers. `supplementary.tex` §D.1 holds four
hand-typed tables — `tab:nguyen` (12), `tab:feynman` (10), `tab:strogatz` (14),
`tab:feynman_ext` (6). Set-diffed against `appendix_d_benchmarks.json`, the **28
undocumented problems are exactly the `hard` + `cherrypicked` + `roundoff` tiers**.
Meanwhile `computational_experiments.tex:128–130` promises *"The benchmark tables in
Appendix D.1 list every problem with its expression, input dimensionality, sampling
protocol, and source citation."* **The promise is false as shipped.** AC-2 is met in
`docs/generated/` and **not** in the document. The generated 70-problem file is
`\input` by nothing.

🔴 **2. The response letter's R2.6 `\changeref` was false in two of three clauses.**
It asserts (a) D.2 *and* D.3 restate the count, (b) "sourced from the campaign ledger",
(c) D.3 carries a completeness statement and per-arm cell counts. Measured: (a)
**PARTIAL** — D.2 only; (b) **FALSE** — `:928–930` read *"Each of the two solvers
**therefore** runs three arms…"*, which is design arithmetic, the very thing R2.6
condemns; (c) **FALSE** — D.3 (`:946–973`) had no completeness statement and no cell
counts. A claim typed in one document about another, in the answer to the comment about
numbers typed rather than computed.

🔴 **3. The k-range table's caption contradicted its own table and the prose below it.**
Caption: *"Overhead increases with $k$ for both keys."* Data
(`analyses/data/overhead_by_k.csv`): naive hash $15.2 \to 16.9 \to 12.2$, IsalSR
$16.8 \to 19.6 \to 15.4$. **Neither is increasing.** Three lines below, the body prose
correctly says *"the largest value falls in the middle one"*. E1's exact class,
re-created in the caption after E1's body text was fixed.

**Repaired by me, in the live Overleaf checkout, not pushed:**

| File | Change |
|---|---|
| `supplementary.tex` §D.2 | Run count no longer derived: *"The total we report, $12{,}600$ runs, is read from the campaign's run ledger, which carries one row per executed run, and not computed from that design"*, with a forward reference to D.3 |
| `supplementary.tex` §D.3 | New `\added` paragraph: the ledger, its 12,600 rows, exit status zero on all of them, the balanced grid (2,100 per host-arm, 420 groups × 30 seeds), one revision and one engine, 14 digests. Makes the letter's `\changeref` true |
| `supplementary.tex` caption `tab:k_range_overhead` | *"The per-DAG key cost rises with $k$ for both keys; the integrated overhead does not follow it, and peaks in the middle bucket."* |

**Verified after editing** (scratch build; live checkout untouched by the compile):
**0 errors, 0 undefined references or citations, 0 overfull hboxes, 17 pages** — and a
pre-edit baseline compiled from `git show HEAD:` is **also 17 pages**, so the three
edits cost **zero pages**.

### 2026-08-14 — `test_4` resolves by disclosure, not by renaming

§8.4's residual-risk item *"`test_4` must get a display name before the appendix ships"*
is **closed, and the ticket's proposed fix was the wrong one.** `test_4` is the
**canonical AI Feynman identifier** (`feynman_remainder.py:7–8, :16`), the name the
database itself uses for that bonus equation and the name the pre-registered draw record
carries. Renaming it in the appendix would break traceability to the source database and
to the draw. The supplementary already does the right thing at the `tab:feynman_ext`
caption: *"\textsc{test\_4} is one of the database's $20$ bonus equations."* Keep the
identifier; the disclosure is the display name.

### 2026-08-14 — AC-7 passes, and "passing" means less than it sounds

Both harnesses re-run by me. `review_campaign/verify.py` **exits 0**: 16/16 derived-value
recomputations against `values/summary.json`, 33/33 literals, placeholders paper 0 /
supplementary 12 / letter 0. `numerical_audit.py` **exits 0**: 1,370 literals inventoried
(measurement 1,161, cross-reference 107, structural 48, problem-id 29, typography 25),
66 bound, **1,143 measurements unbound**, 55 cross-file duplicate groups.

🔴 **But `verify.py`'s literal test is substring-presence over a per-group concatenated
blob, so it is near-vacuous for short literals.** Its own output reports
`ok paper '22'` and `ok paper '13'`; measured, `"22"` occurs 24 times and `"13"` 16 times
in that blob, so both checks are satisfied by any unrelated digit pair. The two harnesses
are also **disjoint** — overlap 0 literals — so nothing is cross-validated: A is a census
that asserts nothing, B is a spot-check that enumerates nothing.

**Measured coverage gaps, neither harness reaching them:** the table *body* files
(`table_supplementary_{udfs,bingo}_body.tex` 1,145 numeric tokens each,
`tab_supp_phi_per_problem.tex` 644, plus the paper's four headline tables) — **4,110
tokens including every headline table**; the response letter's `\changeref` blocks (17
blocks, 81 numeric tokens); prose structural counts such as `N = 70`; table
cross-reference numbers; and the whole double-blind mirror. The mirror carries **no**
measurement drift at present — numeric-token streams are identical on 17 of 18 mirrored
files, the single delta being a citation year dropped by anonymisation.

### 2026-08-14 — AC-2 met **in the shipped document**: 70/70, and generating found another wrong number

The 28 undocumented problems now have two generated tables in Appendix D.1. Set-diffed
by me against `appendix_d_benchmarks.json`, expanding the `\input` bodies:
**suite 70, documented 70, missing 0.**

Presentation decisions, both mine:

- **Grouped by source, not by internal tier.** The tiers are `hard`, `cherrypicked` and
  `roundoff`; **`cherrypicked` must never appear in a TPAMI appendix** — it invites the
  reading that problems were chosen to favour the method. The main text already frames
  these 28 as *"a $28$-problem extension targeting regimes where structural search
  dominates"* (`computational_experiments.tex:125`), so the disclosure exists and the
  tables adopt that framing. Split 14 AI Feynman / 14 from six benchmark families,
  which is the split the appendix prose already draws. A test now hard-fails on any
  tier name reaching emitted LaTeX.
- **Both are `table*`.** The 4-column single-column form overflowed by 52.98 pt — the
  shipped `tab:feynman` fits only because its expressions are hand-abbreviated, which is
  the transcription this ticket removes.

🔴 **A second wrong number, found by generating rather than by reading — Keijzer-6.**
The emitted row read $\log(x_0) + 0.577215664901533$. That is **not the function the
data comes from**: `hard.py:235` sets `target_fn = harmonic`, which computes
$H(x)=\sum_{i=1}^{x} 1/i$ exactly (`:60–62`), while `:230–231` carry the asymptotic
approximation, labelled as such in the source comments. At the smallest training point
$x = 1$ the two differ by **42 %** ($H(1) = 1$ against $\gamma = 0.5772$). An appendix
promising each problem's expression cannot print that one.

**A whole-suite audit followed, and it is the check AC-2 should always have had.**
Every one of the 70 `sympy_expression`s was lambdified and compared against the
campaign's actual $y$ on 64 points:

| Verdict | n | Max relative error |
|---|---|---|
| `exact` (bitwise) | **36** | 0.0 |
| `floating_point` | **33** | ≤ **3.693 × 10⁻⁹** |
| `approximation` | **1** — Keijzer-6 | **4.228 × 10⁻¹** |

> ⚠ **These are the corrected figures.** The first pass of this audit reported
> "68 exact, 1 rounding, 1 approximation" and **that was wrong** — see the
> retraction entry below. The counts above come from the persisted
> `data_agreement` block, read by me out of the file.

So exactly one of 70 was wrong, and it is now the only one carrying a display override.
The table prints $\sum_{i=1}^{x_0} 1/i$ and the caption discloses that recovery is
assessed against $\log x_0 + \gamma$, *"which no expression over the operator set
represents"* — a limitation of the benchmark that the submitted version never stated.
**No campaign number moves**: the runs were always against the harmonic data; only the
`solution_recovered` matching used the asymptotic form, and that is unchanged.

Three pre-existing string defects (`I.12.4`, `II.3.24` LHS leaks; `Strogatz-shearflow1`'s
`[written …]` annotation) were re-checked and reach **no** emitted LaTeX — the renderer
prefers `sympy_expression`, which is numerically exact for all three. They survive only
in the JSON's `expression` field, filed as a trap for a future consumer.

**Verified after every edit** (scratch build): **0 errors, 0 undefined references or
citations, 0 overfull hboxes, 0 `color{red}`, 18 pages.** The supplementary grew 17 → 18;
the three earlier edits cost zero pages and the two full-width tables cost one.
`supplementary.bbl` needed a `bibtex` re-run for `keijzer2003`, `korns2011`, `pagie1997`
and `vladislavleva2009`, none of which was in the shipped `.bbl`.

**Decision (mine): AC-7 is not met by an exit code.** `verify.py` is being hardened to
anchor every literal to a `file:line` with a required same-line anchor, to fail on an
ambiguous match rather than pass, to include the table bodies in its corpus, and to
assert the structural counts (12,600 / 2,100 / 70 / 30) from `summary.json`. Those four
counts are precisely what R2.6 complained about.

### 2026-08-14 — R2.5's second request answered, and I committed the ticket's own defect doing it

**A concurrent session held `reviews/response_to_reviewers.tex` for most of this session**
(T10). It released the lane, and **declined to paste my drafted paragraph** on the
grounds that it had not re-derived the Keijzer-6 audit, the 28-of-50 count or the
70-of-70 set-diff itself. That was the right call and it is the standing rule working:
whoever owns the evidence writes the claim.

**R2.5 answered the counting question and was silent on the reviewer's second request.**
The comment ends *"and Table 5 should list all problems used"*; the block's `\changeref`
promised *"suite size once; the per-problem result tables name the set they cover"*,
which is consistency, not coverage. That was the honest limit when written — the
appendix documented 42 of 70. Two paragraphs added: the 22-of-50 coverage gap against
Section 4.2's promise, the 70-of-70 fix, and the Keijzer-6 discrepancy with the
70-problem numeric audit behind it. `\changeref` extended. Letter compiles **0 errors,
0 undefined, 0 overfull, 35 pages**.

🔴 **And then the other session's parting warning caught me.** It flagged that
`verify.py`'s coverage is a whitelist and that any new quoted literal must be registered
when written. My new paragraph quotes **`42\%`** and **`1.6 \times 10^{-8}`** — both from
a subagent's one-off scan that existed **in no file**. I had put two numbers in front of
the reviewers with no artefact behind them.

**This is the exact defect T09 exists to remove, committed by T09's author inside T09's
own answer, an hour after writing that a claim must be generated rather than typed.**
It is the third time this ticket's thesis has been demonstrated against its own author
(the earlier two: the $\{20,240,1000\}$ training sizes, and the 50-vs-70 contradiction
AC-7 caught within the hour). Corrective action: the audit is being persisted as a
`data_agreement` block in `appendix_d_benchmarks.json`, with a validator that hard-fails
the generator if any benchmark's recorded expression drifts from the data it was
generated from and has no declared display override. **Pre-committed: if the persisted
split does not reproduce 68 exact / 1 rounding / 1 approximation, the sentence is
retracted, not defended.** *(It did not. The commitment was honoured the same hour —
see the retraction entry below.)*

**Two claims I verified rather than inherited**, both of which the letter and my new D.3
paragraph assert: the 14 configuration digests are a **perfect bijection** with the 14
(host, suite) pairs — each pair has exactly one digest and each digest belongs to
exactly one pair — and every one of the 420 host-arm-problem groups carries the same 30
seeds. Neither is "approximately" true; both are exact.

**One number I nearly got wrong in this very ticket.** I first recorded the main paper as
"12 pages, untouched". The annotated build is **18**; 12 is the `[final]` count, which I
have not measured and which belongs to T13. Corrected in §8.1 to say so.

### 2026-08-14 — AC-2's verification clause automated, and a false paragraph found in the letter

**AC-2 says "Verified by set-diff against the results tables."** Until today that diff was
done by hand. `verify.py` now performs it: it extracts Appendix D.1's region, splices the
two `\input` bodies, reads first-column ids from all six tabulars and diffs three sets
against the generated inventory.

```
  ok   Appendix D.1                            70 of 70 ids
  ok   table_supplementary_udfs_body.tex       70 of 70 ids
  ok   table_supplementary_bingo_body.tex      70 of 70 ids
```

Symmetric difference empty on all three, no duplicates. **117 → 120 checks, exit 0.**
Two id-shape normalisations are declared rather than inferred (`tab:nguyen` prints bare
`1`…`12`; `tab:strogatz` omits the `Strogatz-` prefix), and a test pins the dict to
exactly those two so a shape change fails loudly. The agent's own new tests caught a real
extractor bug mid-build — the first tabular chunk carries the column spec together with
the header row, so `Nguyen-begin{tabular}{@{}ll@{}}` was being read as an id — fixed by
stripping the environment preamble, not by loosening the check.

🔴 **Found while cross-checking R2.6's neighbourhood: `response_to_reviewers.tex:1904–1912`
is false in three clauses, and it is a paragraph about candour.** It reads:

> *"One thing is deliberately not yet final, and we would rather say so than let it be
> discovered. The effect figures the abstract quotes … are those of the submitted
> campaign. The campaign that replaces them is running, and the abstract will carry its
> figures … before the manuscript is resubmitted."*

Measured against `main.tex:90`: the abstract carries **$38.1\%$ / $43.7\%$** reduction,
**$70$** problems, Cohen's *d* **$2.54$ / $7.05$**, and names *"a naive hash of a
fixed-order DAG serialization"*. The submitted campaign had **two** arms and **50**
problems and could not have produced any of those. The abstract already carries C2's
figures, and `verify.py` binds them. So: the figures are not the submitted campaign's,
the replacing campaign is not running, and the promised update has already happened.

**Left for T12, whose block R2.8 is — not silently.** Shipping it would tell three
reviewers that the abstract is provisional when it is final, inside the one paragraph
that stakes its credibility on volunteering a weakness. Surfaced to Mario in the close
report rather than fixed here.

**Also verified while there**, since R2.6's block asserts it: *"the analysis pipeline
refuses to produce a table from an incomplete root instead of reporting whatever it
finds."* True — `analyze.py:1213` raises `CampaignIntegrityError`, `:1490–1492` logs
*"Analysis refused"* and exits 2.

### 2026-08-14 — the pre-commitment fired: I retracted my own number before submission

**I put "sixty-eight agree exactly, one differs by $1.6 \times 10^{-8}$" into the
response letter. It was wrong.** The persisted audit measures **36 exact / 33
floating-point / 1 approximation**.

**How the wrong number arose.** The first pass was a throwaway scan that tested
`max_rel > 1e-8` and printed nothing for anything below it. Sixty-eight rows printed
nothing, and *"printed nothing"* was read as *"agrees exactly"*. The scan never
distinguished bitwise zero from tiny non-zero, so it could not have supported the claim
it was used for. Only **36** problems agree bitwise; 33 more differ by a non-zero amount
far below any tolerance (median ≈ 1e-16, max **3.693 × 10⁻⁹**). The quoted `1.6e-8` was
also seed-dependent — it appears at seed 7 and not at seed 42 — and it was never "the
one that differs", only the largest of thirty-three.

**Corrected in the letter**, which now reads *"sixty-nine agree to within floating-point
rounding, thirty-six of them bitwise and the remainder to at most $3.7 \times 10^{-9}$,
and Keijzer-6 is the only genuine discrepancy."* Every figure is read from
`data_agreement` in `appendix_d_benchmarks.json`. Letter recompiled: **0 errors, 0
undefined, 0 overfull, 35 pages**.

**Counts are stable, so the correction is safe.** The agent swept seeds × sample sizes:
36/33/1 at seed 7 (64, 256, 1024 points) and at seed 42 (64, 256, 1024). Declared
parameters: `DATA_AGREEMENT_SEED = 42`, `DATA_AGREEMENT_MAX_POINTS = 256`,
`FLOATING_POINT_TOLERANCE = 1e-6` — eight orders below the only real discrepancy and two
above the worst rounding, so no measurement sits near the threshold. Whole generator runs
in 0.80 s.

**Three things worth keeping from this.**

1. **The artefact caught its author.** Persisting the audit was proposed to satisfy a
   process rule; within the hour it falsified a claim already written into the letter.
   That is the entire argument of this ticket, demonstrated for the fourth time today
   and the first time against a number that had already reached the reviewers' document.
2. **The pre-commitment did the work.** The brief said *"if the persisted audit does not
   reproduce 68/1/1, tell me immediately and do not adjust a threshold to make it."*
   The agent reported the discrepancy in its first line and changed no threshold. Had the
   instruction been "confirm 68/1/1", a tolerance of `1e-8` would have reproduced it and
   nobody would have known.
3. **The validator is the durable part, not the numbers.** Any future benchmark whose
   recorded expression drifts from the data it was generated from now aborts the
   generator with the id, the error and the point count, unless it carries a declared
   display override. A wrong formula can no longer reach the appendix silently.

### 2026-08-14 — new phase: the 12 remaining placeholders are a compute job, not an edit

Mario asked whether the supplementary's remaining `\pendingnum`/`\pendingblock`
placeholders are still relevant and to re-execute them on the C++ engine. **They are,
and they are the last unfilled numbers in either document.** All twelve are in the
synthetic permutation study (`sec:supp_scalability_synthetic`):

| # | What | Submitted value |
|---|---|---|
| 1 | `fig_synthetic_scalability.pdf` (`\pendingblock`, `:1444`) | figure |
| 2 | power-law exponent of per-permutation time in $k$ (`:1478`) | $O(k^{0.7})$ |
| 3 | mean per-permutation time at $k = 9$ (`:1483`) | "under $1$ ms" |
| 4–12 | the nine per-$k$ cells of `tab:synthetic_scalability` (`:1506–1514`), median [IQR] ms | $0.04\,[0.03\text{--}0.05]$ … $0.27\,[0.23\text{--}0.31]$ |

**Two independent reasons they cannot simply be restored**, the second of which the
pending ledger records and which is the more serious:

1. The submitted timings are **Python-engine** timings; every other number in the
   revision is from the compiled engine.
2. 🔴 **The archived Picasso outputs do not implement the protocol the text states.**
   `PENDING_LEDGER_supplementary.md:7–15`: the run on disk used **30 DAGs per cell** and
   **stopped sampling at 50,000 permutations for $k = 9$**, against the **200 DAGs and
   exhaustive $k!$** the text describes. So the text has been describing an experiment
   that was not run, independently of which engine ran it.

**Engine equivalence established before any measurement.** Local
`build_hash = 298fc1188bf1b051`, `engine_name = cpp`, isa `x86-64-v3`, gcc 12.2.0; the
`.so` is dated 2026-07-31 and the newest C++ source 2026-07-30, so the build is current
with its sources. A campaign `run_log.json` records
`metadata.hardware.build_hash = 298fc1188bf1b051`. **Identical.** The re-timing will
therefore be on the same engine build that produced every reported campaign number, and
**no rebuild may be attempted** — a rebuild changes the hash and breaks that equivalence.

🔴 **A third defect, found by checking the stated protocol's own arithmetic.**
`:1438–1439` claims the protocol spans $\sum_{k=1}^{9} 600 \cdot k! \approx 2.2 \times
10^{8}$ canonicalization calls. The true sum is
$600 \times 409{,}113 = 245{,}467{,}800 = \mathbf{2.45 \times 10^{8}}$. The stated
$2.2 \times 10^8$ is exactly the $k = 9$ term alone ($600 \times 9! = 2.18 \times 10^8$)
— a partial sum written as the total. The $5\,400$ expression count is correct
($9 \times 3 \times 200$). Neither reviewer caught this; it is checkable in one line.

The protocol to reproduce, as the text states it: $k \in \{1..9\}$, $m \in \{1,2,3\}$,
200 Lample–Charton DAGs per cell over $\{+, \times, \wedge, \sin, \cos, \exp, \log,
\mathrm{neg}, \mathrm{inv}\}$, all $k!$ permutations, `fast_canonical_string` in
`wl_only` mode, 120 s per-permutation timeout. The claim *"no permutation timed out"*
must be **re-measured**, not carried over. $\rho = k!$ and 100 % invariance are
properties of the construction and are not being re-measured.

### 2026-08-14 — sized, gated and submitted: job 2001009 on sr004

**Sizing, measured not estimated** (all nine $k$ timed directly, nothing extrapolated):
per-permutation cost on the compiled engine is **0.0070–0.0101 ms**, flat-to-slightly-rising
over $k = 5 \to 9$ (+44 %). Pure canonicalisation 2,445 s; wall 4,711 s (**78.5 min**)
single-core on this workstation, of which $k = 9$ is **89.6 %**. The submitted Python
figures were 0.04–0.27 ms, so the engine change alone moves these numbers by roughly
**4–27×**.

🔴 **The sizing agent found a silent-fallback hazard, and it is real.**
`backends.py:42` sets `DEFAULT_BACKEND` from `_CPP_AVAILABLE`, and `canonical.py:62–68`
swallows the `ImportError`. If `_native` fails to import, the study runs the **pure-Python**
canonicaliser with no error and no warning, and the only trace is the metadata block.
Given that the entire point of this job is to report *engine* timings, that failure mode
would be invisible and fatal. The worker therefore **hard-fails** unless both
`engine() == "cpp"` and `build_hash == 298fc1188bf1b051`.

**LOCAL was the agent's verdict on wall-clock and it is the wrong criterion here.**
This workstation is a 13th-gen Intel i7-13700KF; **all 12,600 C2 cells ran on AMD EPYC
7H12**, and `supplementary.tex` §D.3 now states outright that CPUs were pinned *because*
wall-clock is a reported quantity. The desktop part is roughly twice as fast, and the
table's own caption invites comparison with the campaign's key costs (0.038 / 0.076 ms,
measured on EPYC). Publishing desktop timings beside EPYC timings, in the same document
that just explained why it pins CPUs, is exactly the inconsistency this ticket exists to
prevent. **Picasso, `--constraint=sr`.**

**Shape: one job, one task, one core, 27 cells in sequence.** Not an array, for three
reasons: concurrent cells would measure memory-bandwidth contention as well as the
algorithm; a single core matches every C2 cell's `--cpus-per-task=1`; and at ~2–3 h on
EPYC it clears SCBI's two-hour floor as a single submission with no array-placement cost.
Output is 28 small files, so `$LOCALSCRATCH` is not required — stated in the worker rather
than left implicit.

**Gates cleared, in order:** local smoke exit 0 with `engine=cpp` and a complete 15-column
fragment; study code **byte-identical** local vs Picasso (`md5sum` on all three files), so
only the two new scripts were shipped and the remote tree was not disturbed; remote engine
`cpp` / `298fc1188bf1b051` / `x86-64-v3`; live quota read (HOME 23.6 GB/0.28 TB, 13.9k
files; FSCRATCH **226.7k/250k files — 91 % of the soft file quota**, worth knowing though
this job writes 28 files); `sbatch --test-only` → "1 processors on nodes **sr004**".

**Submitted: job `2001009`**, `slurm/synthetic_retime/{launcher,worker}.sh`,
`--constraint=sr --cpus-per-task=1 --mem=8G --time=0-08:00:00`. Started immediately on
**sr004**, no queue wait.

🔴 **A fourth defect, fixed while waiting.** `:1438–1439` claimed the protocol spans
$\sum_{k=1}^{9} 600 \cdot k! \approx 2.2 \times 10^{8}$ calls. The sum is
**245,467,800 = 2.45 × 10⁸**; the stated figure is the $k = 9$ term alone. Corrected to
state the exact count.

### 2026-08-14 — I briefed a duplicate generator into existence and caught it before it shipped

While the job ran I briefed the values extractor to emit the results table for
`tab:synthetic_scalability` — nine rows, `\checkmark` when $\rho = k!$, invariance as a
measured percentage, `median [q1--q3]` at two decimals, `362{,}880` separators — and
argued in the brief that generating the $\rho$ and invariance columns turns two
construction *assertions* into two measured *confirmations*.

🔴 **All of it already exists.** `generate_fig_synthetic_scalability.py:329–378`,
`_export_latex_table`, emits precisely those columns from precisely those fragments, with
`\checkmark`/`$\times$` on `rho_equals_kfact`, `f"{med:.2f} [{iqr_lo:.2f}--{iqr_hi:.2f}]"`,
and `f"{kf:,}".replace(",", r"{,}")`. Had the brief run to completion, the same nine rows
would have had **two implementations that can drift** — the defect this entire ticket
exists to eliminate, introduced by the ticket's author, in the ticket's own workstream,
for the second time today.

Retracted mid-flight. The extractor now **imports** that function rather than
reimplementing it, and its scope narrows to the five quantities that genuinely do not
exist anywhere: the power-law exponent with standard error and $R^2$, the mean ms at
$k = 9$, `total_timeouts`, the $m$-dependence check, and the provenance block.

**The generalisable lesson, which is not the one I would have guessed.** Every earlier
instance today was *retyping a number*. This one was *rebuilding a generator* — the same
failure a level up, and invisible to every check the ticket has built, because two
generators that agree today produce no audit finding at all. `verify.py` would have passed.
The only defence is reading the repo before briefing, and the brief that caused it was
written with the file open two screens away.

**Standing rule for the rest of this revision**: before commissioning any emitter, grep for
one. The cost of the check is a minute; the cost of the miss is a silent divergence that
surfaces after publication.

### 2026-08-14 — the k = 1..9 window was hiding the result, and Mario caught it

**Two manuscript claims did not survive the re-run, and a third was about to be
published as a null that is not true.**

**1. The power law collapses at k ≤ 9 — because k ≤ 9 is the wrong window.**
On the compiled engine the log–log fit over k = 1..9 gives an exponent of
**0.0131 ± 0.0216, R² = 0.0007, p = 0.54** — indistinguishable from zero, against the
manuscript's $O(k^{0.7})$. I reproduced the agent's slope to four decimals independently.
Reporting "$O(k^{0.013})$" would have dressed a null result as a power law.

🔴 **Mario asked whether higher $k$ would show the pattern. It does, and the answer
changes the finding entirely.** Probed locally at 2,000 sampled permutations:

| $k$ | µs/permutation |
|---|---|
| 1–9 | 7.6 → 10.3 (**flat**) |
| 12 | 14.0 |
| 20 | 40.6 |
| 32 | 81.2 |

**7.9× from $k = 9$ to $k = 32$**, a log–log slope of **≈1.6–1.8**. The flat region is a
fixed per-call overhead floor of ~7–10 µs that masks the true scaling; the compiled
engine is simply too fast for $k \le 9$ to be informative. So the honest result is not
"no growth" but **sub-quadratic growth, consistent with the near-$O(k^2)$ bound the
paper already claims for `fast_canonical_string`, and nowhere near factorial** — a
better and more defensible statement than either the submitted $O(k^{0.7})$ or the null
I was an hour from reporting.

**This is the session's most consequential catch and it was the user's, not mine.** The
protocol was faithfully reproduced, every gate passed, the numbers were correct — and
the conclusion would have been wrong, because the *design* stopped where the C++ engine
stops being measurable. Fidelity to a stated protocol is not the same as fitness for the
question.

**2. "No observable dependence on $m$" is false.** Medians rise with $m$ at 8 of 9 $k$;
pooled Kruskal–Wallis **p = 2.4 × 10⁻²³**, per-$k$ significant at 8/9 after Holm. The
mechanism is structural, not noise: at fixed $k$ a larger $m$ adds VAR nodes and so
enlarges the DAG. Magnitude is modest (~13 % from $m=1$ to $m=3$ at $k=9$).

**3. Two-decimal milliseconds collapse all nine cells to `0.01 [0.01--0.01]`.** The
submitted 0.04–0.27 ms range was a *Python-engine* artefact; at 7–10 µs the unit no
longer resolves anything.

**Decisions taken with Mario** (all three as recommended): report the fit as measured
with its SE and R² rather than as a power law over the flat window; **microseconds** in
the table; state the $m$-dependence with its mechanism.

**Second job submitted rather than cancelling the first.** `2001009` (k = 1..9,
exhaustive) is **not** replaceable by sampling — the ρ = k! claim is that *every one* of
the k! orderings collapses, and 600·11! = 2.4 × 10¹⁰ is where exhaustive enumeration dies
of combinatorics, not where the canonicaliser does. The timing question needs no
enumeration, so `2001113` covers **k ∈ {10,12,14,16,18,20,24,28,32} × m ∈ {1,2,3}** at
**20,000 sampled permutations** per expression, ~2.2 h on EPYC — clearing SCBI's two-hour
floor as a single submission. The grid tops out at **k = 32 deliberately: it is the top
of the campaign's own k-stratified overhead table**, so the synthetic curve and the
empirical one now meet at the same k instead of describing disjoint ranges.

🔴 **Reporting constraint that must not be lost between here and the table**: rows with
$k \ge 10$ are **sampled**, so ρ is the count of distinct canonical strings among 20,000
sampled orderings, not among $k!$. Invariance still means something there; "$\rho = k!$"
does not. Any table mixing the two must mark which rows are which.

### 2026-08-14 — the extension was confounded; Mario caught that too, and the fix is measured

**Mario's objection**: a scaling curve needs one condition with only $k$ varying, and my
two-job design did not have one. He was right, and the confound is larger than I expected.

**The measurement that settles it.** Same $k = 9$, same 15 DAGs, same engine, same mode —
only the number of permutations averaged per expression varies:

| permutations / expression | µs per permutation | vs exhaustive |
|---|---|---|
| 24 | **16.55** | **+66 %** |
| 1,000 | 10.92 | +9.4 % |
| 5,000 | 11.31 | +13.3 % |
| 20,000 | 10.65 | +6.8 % |
| 362,880 (exhaustive) | **9.98** | — |

A per-expression warm-up cost amortises as $\approx C/P$, so **a small sample reads high**.

🔴 **This invalidates the exhaustive run as a source for the scaling curve, and it
invalidates my "fixed overhead floor" explanation of the flat region.** Under exhaustive
enumeration $P = k!$, so $k = 1$ averages **one cold call** and $k = 9$ averages 362,880
warm ones. The low-$k$ points are inflated by a factor that shrinks as $k$ grows — which
manufactures a flat region out of nothing. The flatness I was an hour from publishing as
a property of the algorithm is, at least in part, an artefact of the measurement design.

**And my proposed fix was itself confounded**: pairing exhaustive $k \le 9$ against
sampled $k \ge 10$ would have compared inflated low-$k$ points with clean high-$k$ ones,
biasing the very slope being reported. Job `2001113` was cancelled at 8 min for that
reason, not because it failed.

**The clean design, job `2001366`.** One job, one core, one node, one engine, one mode:
$P = 20{,}000$ permutations per expression at **every** $k$, $k \in \{8,\dots,36\}$ (21
values) $\times$ $m \in \{1,2,3\}$, `n_expr = 100`. The residual $+6.8\,\%$ bias is common
to every $k$, so it shifts the intercept and not the slope, which is the reported
quantity. **$k = 8$ is a forced floor, not a choice**: a cell can only supply $P$ distinct
orderings when $k! \ge P$, and $8! = 40{,}320$ is the first factorial above 20,000.

**Division of labour between the two jobs, and it is not interchangeable.**
`2001009` (exhaustive, $k = 1..9$) proves $\rho = k!$ — the claim that *every one* of the
$k!$ orderings collapses, which sampling cannot establish. `2001366` measures the cost
curve. **The exhaustive job's timing column must not feed the scaling fit**, and the
scaling job's $\rho$ is a count over sampled orderings, not $k!$. Each answers what only
it can answer.

**Also worth recording**: `build_info()` reports gcc **12.2.0** locally and **13.2.0** on
the compute node while the `build_hash` is identical on both, so the hash covers the
source and not the toolchain. Every published number comes from Picasso, as the campaign's
did, so the comparison is sound — but "matching build hash" is a weaker guarantee than it
sounds and should not be quoted as binary identity.

### 2026-08-14 — measured for T13: the main paper is 6 pages over, and promotion will not fix it

Surfaced by the T12 session and **verified by me rather than relayed**, because I had
already recorded once in this ticket that I had *not* measured the `[final]` count and
should not state it.

Injected `final` into `\usepackage[commandnameprefix=ifneeded]{changes}` in a scratch copy
of `article/paper` and compiled:

| Build | Pages |
|---|---|
| annotated (as the tree stands) | 18 |
| **`final` (markup stripped)** | **18** |
| `final`, minus the CRediT and competing-interest blocks | 18 |
| `final`, also minus all three bios and photos | 17 |
| TPAMI limit, *including* references, biographies and photos | **12** |

*(Rows 1–2 measured by me; rows 3–4 by the T12 session, whose decomposition is the more
useful one — it isolates the markup at 0 pages, CRediT at 0, and the mandatory bios and
photos at 1, leaving **6 pages in the body**. An earlier ticket recorded 15 pages and
understated by three.)*

**Stripping the revision markup removes colour, not content, so there is no page dividend
waiting in the promotion step.** The three `IEEEbiography` blocks are live and uncommented,
so those 18 pages already carry the bios and photos the limit counts. The main file is
**6 pages over**, and the overage is structural rather than presentational.

This is **T13's**, not T09's, and it is recorded here only because T09 measured it. Two
interactions worth stating for whoever takes it:

- It **dwarfs anything an editorial pass can recover.** T12 is running a de-AI and
  filler-removal pass over `article/paper` aimed at shortening; six pages is not a prose
  problem.
- It **collides with the C7 decision.** The revision keeps the supplementary separate as
  digital-library material, against R2's request to fold it into the main paper. That
  decision is now doubly forced — folding 18 pages of supplementary into a main file
  already 6 over is arithmetically impossible — but it also means the 12-page fix has to
  come entirely out of the main text, which is where R1 and R3 asked for *more*
  explanation, not less.

**T09's own contribution to the overage is nil**: everything this ticket added went into
the supplementary (17 → 18 pages), and the only main-text edits it made were **deletions**
(E1's restated percentages, E8's ρ range).

### 2026-08-14 — exhaustive run landed: 27/27, and three of four claims are settled

`2001009` **COMPLETED**, 2 h 40 m 57 s on sr004, **27/27 fragments**, every cell
exhaustive (`200 expressions x 362880 perms` at $k = 9$). Copied to
`c2_3arm/synthetic_retime/data/exhaustive/` with its SLURM logs. **Certified cell
count re-counted after the copy: 12,600, unchanged** — the census skips anything not
matching `<method>/<bench>/<problem>/<arm>/seed_NN/`, as predicted, and now verified
rather than assumed.

**Measured, 600 expressions per $k$, EPYC 7H12, C++ engine:**

| $k$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| median µs | 12.06 | 12.35 | 12.10 | 12.37 | 13.83 | 15.11 | 16.85 | 18.43 | 20.46 |

Invariance **100 %** and $\rho = k!$ on all nine, 600 expressions each.

| Claim | Verdict |
|---|---|
| "no permutation timed out" | ✅ **holds** — 0 timeouts across 245,467,800 calls |
| "$\sum 600 \cdot k! \approx 2.2 \times 10^8$" | 🔴 **wrong** — realised **245,467,800**, exactly the correction I applied |
| "$\rho = k!$, 100 % invariance" | ✅ **holds** on 5,400/5,400 |
| "no observable dependence on $m$" | 🔴 **false** — Kruskal–Wallis significant after Holm at **9 of 9** $k$ (the reduced probe found 8/9) |

**The exponent moved between the probe and the real run, which is itself instructive.**
Reduced local (n=20, i7): 0.0131, R² 0.0007, p 0.54 — a null. Real (n=200, EPYC):
**0.2111, SE 0.0040, R² 0.3375** — small but unambiguously non-zero. So the "no
detectable growth" reading I brought to Mario was an artefact of the *probe's* noise,
not of the engine. **Neither figure may be reported as the scaling exponent**: both come
from the exhaustive design where $P = k!$ varies with $k$, which biases the slope
*downward* by inflating the low-$k$ points. `2001366` supplies the reportable one.

The exhaustive table's timing column stays — it honestly reports "mean per-permutation
time when enumerating all $k!$ orderings", which is what was measured. It simply must not
be read as a scaling series, and the appendix will say so.

**Written**: `c2_3arm/synthetic_retime/README.md`, carrying the two-experiment split, the
$P$-inflation measurement that forbids pooling them, the schema, the regeneration
commands, and the three defects above. Nobody reading that directory can pool the two by
accident without ignoring its first table.

### 2026-08-14 — µs conversion and the fit removal, decided by looking at the render

**Units → µs.** In ms at two decimals all nine cells collapse to `0.01`/`0.02` and the
$k$-trend disappears. Converted in the single table emitter and in the figure axis; the
JSON gained `*_us` keys alongside the untouched `*_ms` ones, because a key named `_ms`
holding µs is the trap this ticket exists to remove. All nine regenerated rows reproduce
the campaign values exactly, 25 → 28 tests, `ruff` clean,
`run_synthetic_scalability.py` md5 unchanged throughout (job `2001366` was running
against a byte-identical deployed copy).

🔴 **The `$O(k^{0.2})$` fit is gated off, and I decided that by rendering the figure
rather than by reading the statistic.** On the chart the dashed line passes *above* the
$k=1$ boxes and *below* the $k=9$ boxes, cutting across the data — which is what
$R^2 = 0.337$ looks like drawn. Two independent reasons not to ship it: the fit is weak,
and its exponent is confounded by the varying-$P$ design and therefore not reportable at
all. **Gated, not deleted** (`--powerlaw-fit`, default off): the fixed-$P$ run is exactly
the dataset that will want it. The fit is still *computed* and logged, so this is a
plotting change and not a measurement change.

**The figure is stronger without it.** Its argument was never the regression line: the
right axis climbs from 1 to 362,880 isomorphic copies while the boxes move 12.06 → 20.46
µs. The regenerated caption now states that directly — *"the median canonicalization time
rises only from 12.06 to 20.46 µs between $k = 1$ and $k = 9$ (a factor of 1.70)"* — which
is a measured claim where the fit was an inferred one.

**And the figure now corroborates the $m$ correction instead of contradicting it.** At
every $k$ the $m=3$ boxes sit above $m=2$ above $m=1$. The manuscript's "no observable
dependence on $m$" was contradicted by its own figure; stating the dependence makes text
and figure agree.

**Two residual items for the final figure pass**, neither blocking:
- The log $y$-axis carries a single labelled tick (`10¹`) across a 10–30 µs band, so a
  reader cannot read values off it. Needs minor ticks before it ships.
- The generator's caption still ends *"collapsed in $<1$ ms"* — true at 20.46 µs but two
  orders loose, and pre-existing. The manuscript's own caption is hand-written and gets
  the measured figure, so this only affects the advisory `caption.txt`.

### 2026-08-14 — the clean scaling run lands, and it vindicates the redesign

`2001366` **COMPLETED**, 4 h 25 m on sr031, **63/63 fragments**, 6,300 rows. Copied to
`data/scaling/`. Two design checks passed exactly:

- `n_perms` is **20,000 on every one of the 6,300 rows** — the fixed-$P$ condition held
  with no exceptions, so the confound that invalidated the exhaustive fit is gone.
- `n_unique_canonicals` is **1 on every row** — 126 million sampled permutations, every
  one collapsing to a single canonical string, at $k$ up to **36**. The submitted study
  verified invariance only to $k = 9$.

| $k$ | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 36 |
|---|---|---|---|---|---|---|---|---|
| median µs | 24.54 | 31.46 | 46.46 | 69.52 | 94.19 | 117.94 | 142.56 | 171.83 |

**Clean power-law exponent = 1.4301 (SE 0.0061), $R^2 = 0.8970$**, $n = 6{,}300$,
0 timeouts. Reproduced independently by hand and by `extract_synthetic_values`.

**The three estimates, and why only one is reportable:**

| Source | Exponent | $R^2$ | Status |
|---|---|---|---|
| Submitted manuscript (Python engine) | 0.7 | — | superseded |
| Exhaustive $k \le 9$, this campaign | 0.211 | 0.34 | **confounded** — $P = k!$ varies with $k$ |
| Reduced local probe | 0.013 | 0.0007 | noise; the "null" I nearly published |
| **Fixed-$P$, $k = 8..36$** | **1.430** | **0.897** | **reportable** |

**This is a materially better result than the submitted one.** $O(k^{1.43})$ is
sub-quadratic and sits inside the near-$O(k^2)$ bound the paper already claims for
`fast_canonical_string`, with $R^2 = 0.90$ on 6,300 individual expressions — a real power
law, not a line drawn through noise. The submitted $O(k^{0.7})$ understated the growth and
rested on a design that could not measure it.

**Every step of the redesign is now justified by data rather than by argument**: the
fixed-$P$ condition (the 66 % inflation measurement), the extension past $k = 9$ (the
7.0× rise from 24.54 to 171.83 µs that $k \le 9$ cannot see), and the refusal to fit the
exhaustive series (0.211 against 1.430 — a factor of 6.8).

### 2026-08-14 — σ collision resolved, and the gate caught me with it

The T12 session found `\sigma` naming two objects. Verified in the sources rather than
taken on report, and **it is worse inside my own file than they described**:
`\sigma(v)` is the *ordered input list* of a node (`methodology.tex:33–36`, carrying
Rule 1, condition (iv) and the `Pow` operand order) and appears **23 times** in
Appendices A–C; `\sigma(D)` is a *serialization* and appears **17 times** in Appendix E.
Same letter, two functions, one file. Structurally R2.2, which the reviewers raised about
`{g,i}` versus `{−,/}`.

The serialization sense yielded, since the node sense is theorem-bearing and older:
**17 occurrences renamed to `\mathrm{ser}`**, bounded by the `\label{sec:supp_hash}`
anchor so the 23 node-sense uses could not be touched. Zero `\sigma` remain after that
anchor; 23 remain before it, as intended.

🔴 **`verify.py` then failed — on my rename.** `\rho_\sigma = 1.0000` was an anchored
literal; renaming it left the check matching nowhere. **This is the anchor-repointing case
I had warned T12 about an hour earlier, landing on me.** Repointed the literal with a
comment recording why, rather than reverting the rename: **121/121 pass**.

A second consequence needed fixing too: `U_{\mathrm{ser}}` is wider than `U_\sigma`, so
the $\Delta$/$\phi$ display equation overflowed the column by 4.998 pt. `\qquad` → `\quad`
restores **0 overfull**.

Regenerated `tab_supp_phi_per_problem.tex` from the updated generator rather than
hand-editing the header — the diff is header-only across three tables and **no number
moved**. Final: **0 errors, 0 undefined, 0 overfull, 18 pages, verify 121/121.**

### 2026-08-14 — all twelve placeholders filled; and I offered a cut on a false premise

**Every placeholder in the package is gone.** `paper 0 · supplementary 0 · letter 0`,
scaffolding block deleted per its own instruction, **0 errors, 0 undefined references,
0 overfull hboxes, verify 128/128**, supplementary at 19 pages.

What landed in the appendix: the exhaustive subsection reports invariance, $\rho = k!$
and the 12.06 → 20.46 µs span and **fits nothing**, stating why and quoting the inflation
measurement so a reader can check the reasoning; a new subsection
`sec:supp_scalability_sweep` carries the fixed-sample result ($O(k^{1.43})$,
$R^2 = 0.90$, 6,300 expressions, $1.26 \times 10^8$ calls, 24.54 → 171.83 µs) with
`fig:synthetic_sweep`; the $m$ dependence is stated with its magnitude and mechanism.

🔴 **Two more shared-name defects, found by wiring the figure.**
Both generator invocations write `fig_synthetic_scalability.pdf`, so copying the sweep
into the supplementary would have silently overwritten the exhaustive figure — which is
why the sweep never travelled and the subsection carrying the headline exponent shipped
without its picture. Installed as `fig_synthetic_sweep.pdf`. And the legend rendered
`{b_fit:.1f}` → `$O(k^{1.4})$` beside prose saying `1.43`: **one fit at two precisions in
one document**, E1 in miniature. Both now `.2f`.

**Today's root cause, stated once**: `\sigma` naming two functions, `\rho_\sigma` and
`\rho_{\mathrm{ser}}` naming one quantity, two generator runs writing one path, one
exponent at two precisions. Every defect this evening was **something sharing a name that
should not**, and none of them was a wrong measurement.

🔴 **I offered to cut `fig:synthetic_sweep` to recover a page, and the premise was wrong.**
The 12-page ceiling is on the **main manuscript file only** — `source/README.md` states it
as *"12 pages for the main file, including references, biographies and photos"* — and the
board's C7 decision keeps the supplementary separate as digital-library material, with R1
and R3 both endorsing that and R3 accepting it *as is*. **The supplementary has no page
budget**, so the page I offered did not need recovering, and the cut would have removed
the only picture of the exponent the manuscript now leads with. Corrected by the T12
session; recorded because the error is mine.

**An argument for T14's C7 answer, which is stronger than the drafted one.** R2's C7 asks
for the supplementary to be folded into the main paper *within the strict limit*.
Measured now: main **17** pages against a ceiling of **12**, supplementary **19**. Merging
would require a **36-page document to fit in 12**. At submission the pair was 12 + 10 and
the request was merely unreasonable; it is now arithmetically impossible — and the growth
is **entirely material R2 themselves asked for**, R2.5's benchmark tables plus this
figure. Declining C7 in the reviewer's own terms is more courteous and more convincing
than declining it on the page limit alone.

---

## 8. Proposed answer

### 8.1 Before / after

**Updated 2026-08-14. No cell is gated any more.** C2 completed, and every value below
was re-measured by me after the campaign landed — from `status_ledger.csv`,
`analyses/values/summary.json`, `analyses/data/overhead_by_k.csv` and the compiled
documents, never from a subagent's report or from a markdown note.

| Quantity | Submitted | Revised | Source |
|---|---|---|---|
| Suite description, Section IV.2 | "32 core (12 Nguyen + 20 Feynman) + 18 extension" | 70 problems, nine published families, no post-hoc filtering | AC-3 |
| True composition, submitted suite | 12 + 24 + 14 = 50 | (unchanged; stated correctly for the first time) | AC-3 |
| True composition, revised suite | — | 12 Nguyen + 30 Feynman + 14 Strogatz + 14 other = 70 | AC-3 |
| Feynman problems in the suite | 24 (text said 20) | 30 (24 + 6 D2 remainder) | AC-2 |
| Problems documented in Appendix D.1 | 22 of 50 | **70 of 70** | AC-2 |
| Problems undocumented | **28** | **0** | AC-2 |
| Fields documented per problem | expression, dim, protocol, citation (for 22) | 9 fields, all non-empty, for 70 | AC-2 |
| Appendix tables generated from data | no | **yes** — 51 tests, deterministic, `ruff`/`mypy` clean | AC-1 |
| ODE-Strogatz citation | absent from `references.bib` | `strogatz1994` added | AC-2 |
| Training sizes stated | $\{20, 240, 1000\}$ — **20 is used by no problem** | {50, 100, 240, 300, 676, 1000, 1024, 2000} | AC-3 |
| Total runs, stated | 2,640 | **12,600**, and stated once | AC-4 |
| Run count provenance | design formula `2×2×(12+10)×30` | the campaign **run ledger**, one row per executed run | AC-4 |
| Completeness statement | absent — 35 Bingo cells were missing and unmentioned | present in D.3: 12,600/12,600 exit 0, no undefined metric | AC-4 |
| Cells per host and arm | never stated | **2,100** on all six blocks, exactly balanced | AC-4 |
| Seeds per problem | claimed 30, not true of Bingo | **30** on all **420** host-arm-problem groups, verified | AC-4 |
| Engine per cell | Python, mixed CPU pool | **`native` (C++) on 12,600/12,600**, one revision, 14 config digests | AC-4 |
| Recurrence in Appendix D.3 | present | resolved; D.3 now sources the count rather than restating it | AC-4 |
| Operator sets in the manuscript | 4, undifferentiated | **5**, distinguished — the two hosts differ from each other | AC-5 |
| Host set stated | one shared $\{+,-,\times,\div,\sin,\cos,\exp,\log\}$ | per method: Bingo 10, UDFS 11 (no generic `pow`) | AC-5 |
| Host set actually used | varied by tier (hard tier added sqrt/pow) | uniform across the suite per method (A4b) | AC-5 |
| Σ_SR vs host set distinguished | no | yes, with the composition identities | AC-5 |
| N-8 / N-11 explained | no | yes — already in the letter's R2.3 block | AC-5 |
| k-overhead, 5 ≤ k < 15 (main / appendix) | 45.9 % / 47.0 % — disagree | main text no longer restates; Table 8 referenced once | AC-6 |
| k-overhead, 15 ≤ k < 32 (main / appendix) | 41.6 % / 49.9 % — disagree | as above | AC-6 |
| "35.5–56.0 %" range in main text | asserted; **0 hits** in `paper/*.tex` | removed, with its false attribution | AC-6 |
| ρ range in discussion | 1.45–1.96 | not restated; referenced to the appendices | AC-6 |
| ρ per-problem range | stated 1.45–1.96; true union [1.11, 1.98] | UDFS **[1.11, 2.12]** (mean 1.66), Bingo **[1.19, 1.85]** (mean 1.79) | AC-6 |
| k-strata caption | "overhead increases with $k$" — **false for both keys** | per-DAG cost rises; overhead peaks in the middle bucket | AC-6 |
| Bingo overhead | 39.2 % median | **16.3 %** mean / 16.29 % median (the C++ port's payoff) | AC-6 |
| Cross-file duplicated measurements | ≈40, unaudited | **41 groups enumerated**; 11 coupled and asserted to agree; residue of 31 named | AC-7 |
| Numeric checks that run in CI | **0** | **117+**, each anchored to a `file:line` with a same-line anchor | AC-7 |
| Literal check strength | n/a | ambiguity is now a **failure**: `'22'` matched 24 places and passed silently | AC-7 |
| Expressions checked against the data | never | **70/70**, persisted as `data_agreement`: **36 bitwise-exact, 33 within 3.7 × 10⁻⁹, 1 approximation** (Keijzer-6, 42 %, fixed). A validator now hard-fails the generator on any undeclared drift | AC-2 |
| Repository test suite | — | **7,810 passed, 5 skipped**; the three harnesses contribute 208 | AC-1/7 |
| Supplementary page count | 10 (submitted) → 17 (before this session) | **18** — the two D.1 tables cost one page | T13 |
| Main paper | 12 pp | **18 pp in the annotated build**, untouched this session — the `[final]` count that the 12-page limit governs is **T13's to establish, and I have not measured it** | T13 |

### 8.2 Changes made to the manuscript

Applied 2026-08-04 in the live Overleaf checkout, **not pushed**. Both documents
compile exit 0, 0 errors, 0 undefined references, at 12 and 10 pages — unchanged.

| File | Lines (revised) | Change |
|---|---|---|
| `article/paper/computational_experiments.tex` | 48–55 | "assembled in three stages: a $32$-problem core … an $18$-problem extension" → 70-problem suite, no post-hoc filtering; the Appendix D.1 promise extended to sampling domain and train/test sizes, and stated to be generated from the executed definitions (AC-3) |
| `article/paper/computational_experiments.tex` | 78–95 | Composition rewritten over nine families totalling 70; protocol tally 53/14/3; true size ranges replace $\{20, 240, 1000\}$ (AC-3) |
| `article/supplementary/supplementary.tex` | 557–575 | New **Operator sets** paragraph: Bingo's 10 and UDFS's 11 stated separately, uniformity across the suite, and Σ_SR distinguished from the hosts' search primitives via the composition identities (AC-5) |
| `article/paper/results.tex` | 176–177 | Three restated $k$-stratified percentages deleted; Table 8 referenced, not duplicated (AC-6 / E1) |
| `article/supplementary/supplementary.tex` | 733–734 | Phantom "$35.5$–$56.0\%$ … reported in the main text" and its false attribution deleted (AC-6 / E2) |
| `article/paper/discussion.tex` | 10–11 | Incorrect $1.45$–$1.96$ ρ range and duplicated per-method means deleted; ranges referenced to the appendices (AC-6 / E8) |
| `article/paper/references.bib` | +`strogatz1994` | New entry; the 14 ODE-Strogatz problems had no resolvable citation |
| `article/paper/main.bbl` | regenerated | The project ships a pre-built `.bbl`; `references.bib` alone leaves the citation undefined |

**Applied 2026-08-14, second pass** (live Overleaf checkout, **not pushed**). Line
numbers are omitted deliberately: a concurrent session was editing this tree throughout,
and this ticket's own AC-7 finding is that line-numbered references to a live checkout
are perishable. Anchors are content.

| File | Change |
|---|---|
| `article/supplementary/supplementary.tex` §D.2 | Run count no longer derived. *"Each of the two solvers **therefore** runs three arms … for a total of $12{,}600$ runs"* → *"The total we report, $12{,}600$ runs, is read from the campaign's run ledger, which carries one row per executed run, and not computed from that design"*, with a forward reference to D.3 (AC-4) |
| `article/supplementary/supplementary.tex` §D.3 | New `\added` paragraph: the ledger, 12,600 rows, exit status zero on every one, no undefined metric, the balanced grid (2,100 per host-arm; 420 groups × 30 seeds), one revision and one engine, 14 configuration digests. **This is what makes the letter's R2.6 `\changeref` true** (AC-4) |
| `article/supplementary/supplementary.tex` caption `tab:k_range_overhead` | *"Overhead increases with $k$ for both keys"* → *"The per-DAG key cost rises with $k$ for both keys; the integrated overhead does not follow it, and peaks in the middle bucket"* (AC-6) |
| `article/supplementary/supplementary.tex` §D.1 | New introductory paragraph and two `table*` floats, `tab:feynman_struct` and `tab:gp_struct`, `\input` from generated bodies; the `tab:gp_struct` caption discloses Keijzer-6's target and recovery criterion (AC-2) |
| `article/supplementary/tab_supp_bench_struct_{feynman,other}.tex` | **New**, generated; 14 data rows each |
| `article/supplementary/supplementary.bbl` | Regenerated by `bibtex`: `keijzer2003`, `korns2011`, `pagie1997`, `vladislavleva2009` were absent from the shipped `.bbl` |
| `reviews/response_to_reviewers.tex` R2.5 | Two new paragraphs answering the reviewer's **second** request, which the block did not address: the 22-of-50 coverage gap, the 70-of-70 fix, and the Keijzer-6 discrepancy with the 70-problem numeric audit behind it. `\changeref` extended |

**Verified after every edit, by me**: `supplementary.tex` **0 errors, 0 undefined
references or citations, 0 overfull hboxes, 0 `color{red}`, 18 pages**;
`response_to_reviewers.tex` **0 errors, 0 undefined, 0 overfull, 35 pages**. Repository
suite **7,810 passed, 5 skipped**. `review_campaign.verify` **exit 0, all 120 checks
pass**. `ruff` clean on all six files this session touched.

**Nothing is left deliberately wrong.** The `2{,}640` figure now survives in exactly two
places, both correct: the verbatim reviewer quotation in the letter, and the "submitted"
column of the continuity table.

### 8.2b New deliverables in the repository

| Path | Role |
|---|---|
| `experiments/scripts/generate_appendix_d_tables.py` | Emits Appendix D.1 for all 70 problems from `benchmarks/datasets/*.py`; LaTeX + JSON from one pass. The structural fix for the root cause. |
| `tests/unit/test_appendix_d_generator.py` | **88 tests**; hard-fails on an empty mandatory field, on a tier name reaching emitted LaTeX, on a >4-significant-figure float in an expression cell, and on Keijzer-6 rendering as a logarithm |
| `docs/generated/appendix_d/appendix_d_tables.tex` | Generated appendix, 7 tables |
| `docs/generated/appendix_d/appendix_d_benchmarks.json` | Machine-readable inventory; both harnesses consume it |
| `docs/generated/appendix_d/tab_supp_bench_struct_{feynman,other}.tex` | **New.** The two supplementary-ready bodies for the 28 previously undocumented problems, plus an emitted caption note for any display override |
| `experiments/scripts/review_campaign/verify.py` | **The enforcing harness.** 117+ checks: 16 derived-value recomputations, 62 literals each anchored to a `file:line` with a same-line anchor and **failing on an ambiguous match**, 30 coupled-quantity locations in 11 groups, 3 table row counts, 3 placeholder counts |
| `tests/unit/test_review_campaign_verify.py` | **New**, 21 tests. All 21 error against the pre-change module, and one demonstrates the old vacuity directly: `'22' in "…\cite{randall2022bingo}…"` passed |
| `experiments/scripts/numerical_audit.py` | **The census.** 1,370 literals inventoried and classified; `macro_suffix` now bijective base-26 so all 1,224 proposed macro names are letters-only and unique — `\rhoX26` was not a parseable control word |
| `tests/unit/test_numerical_audit.py` | **63 tests.** Its cross-file-duplicate canary is re-pinned to C2's seven ρ and cost literals, with a docstring recording that it is a canary whose values move whenever the reported campaign is re-executed |

### 8.3 Draft response text

> **Status, 2026-08-14 — all three blocks are written and this ticket owes the letter
> nothing.** Re-verified today by direct read of `reviews/response_to_reviewers.tex`:
>
> | Block | State |
> |---|---|
> | **R2.3** | complete; carries the decomposition, the 12-label Σ_SR, `sqrt`/`pow` and a `tab:operator_sets` table |
> | **R2.5** | complete; **and extended by me today** with the coverage answer (22 of 50 → 70 of 70) and the Keijzer-6 discrepancy, which the block did not previously address |
> | **R2.6** | 🔴 was an empty `\todoblock`; **now complete prose**, written by the numbers lane. It concedes the stale figure, declines to certify the submitted campaign as complete (35 Bingo cells were missing), and states the revision's 12,600 |
>
> 🔴 **The one thing that had to be fixed was R2.6's `\changeref`, which promised things
> the appendix did not contain** — see §7, 2026-08-14. Two of its three clauses were
> false. The appendix now delivers all three, so the letter is true as written.
>
> The 2026-08-04 skeleton below is retained only as a record of intent. **Do not paste
> it over the existing prose.**
>
> ---
>
> *Superseded record from 2026-08-04:*
>
> | Block | State | Lines |
> |---|---|---|
> | **R2.3** | complete prose; already carries the decomposition, the 12-label Σ_SR, `sqrt`/`pow` and a `tab:operator_sets` table | 1020–1160 |
> | **R2.5** | complete prose; **silent on the alphabet**, which is correct — R2.5 is a counting comment | 1176–1232 |
> | **R2.6** | 🔴 **empty `\todoblock`** | 1235–1247 |
>
> So the only block this ticket still owes is **R2.6**, and it is gated on one number:
> the run count from the C2 manifest. The skeleton below stands for R2.6 and is
> retained for R2.3/R2.5 only as a record of intent — **do not overwrite the existing
> prose with it.**
>
> **Two corrections R2.5's existing text will need when C2 lands**, both from this
> session: the "28 undocumented problems" figure becomes **0 of 70**, and the promise
> in Section IV.2 is now true because Appendix D.1 is generated rather than
> maintained. A third, unraised item is available to volunteer if the letter wants it:
> the manuscript stated training sizes $\{20, 240, 1000\}$ and **no problem uses 20**.

```latex
%% --- R2.5 ---
\begin{response}
%% Structure that works here:
%%  1. Confirm all three counts and give the true composition (12 + 24 + 14).
%%  2. Volunteer the larger problem behind the one the reviewer found: 28 of the
%%     50 problems had no documented expression or sampling protocol, despite two
%%     sentences in Section IV.2 promising exactly that. Appendix D.1 now
%%     documents every problem and is generated from the benchmark definitions.
%%  3. Name the root cause once -- the appendix was written for the 22-problem
%%     configuration and not propagated -- because it also explains R2.6, and
%%     because a stated root cause is more convincing than five separate fixes.
\changeref{}
\end{response}

%% --- R2.6 ---
\begin{response}
%%  1. Confirm the reviewer's arithmetic; give the corrected count from the
%%     campaign manifest.
%%  2. Answer the explicit question ("confirm all 50 problems were run with 30
%%     seeds") with the ledger, including the exceptions from T08.
\changeref{}
\end{response}

%% --- R2.3 ---
\begin{response}
%%  1. Confirm the discrepancy and explain that these are two different objects:
%%     Sigma_SR is the encoding alphabet used as an inclusion criterion, the D.2
%%     set is the host solvers' search primitives. The manuscript never said so.
%%  2. Concede the part that is an outright error, not an ambiguity: the D.2 set
%%     did not reflect the sqrt/pow extensions actually configured for some tiers.
%%  3. Answer N-8 and N-11 directly -- both expressible from {exp, log, x} on
%%     non-negative domains, both solved to R^2 = 1.0000. The results stand; the
%%     description did not.
\changeref{}
\end{response}
```

### 8.4 Residual risk

Updated 2026-08-04. Two of the three original candidates have resolved; four new ones
replace them.

**Resolved.**

- *"A fifth operator set surfacing somewhere unchecked"* — **it existed and is now
  found.** The two hosts have *different* sets (Bingo 10, UDFS 11); the manuscript
  asserted one shared set of 8. Stated explicitly in the revised
  `supplementary.tex`, so it can no longer surface as a surprise.
- *"The generated appendix growing past the page budget"* — the generated tables are
  169 lines across 7 tables. Both documents still compile at **12 and 10 pages**,
  unchanged. Still to confirm with T13 once the tables are *inserted* into the
  supplementary rather than living in `docs/generated/`.

**Resolved 2026-08-14.**

- *"The `2,640` → manifest run count is still unwritten."* — **written**, and from the
  ledger rather than a manifest, because no `MANIFEST.json` exists (T02 AC-3). The
  ledger is the better artefact: a manifest states intent, the ledger states what ran.
- *"`test_4` would print verbatim in a TPAMI appendix; it needs a display name."* —
  **the proposed fix was wrong.** `test_4` is the AI Feynman database's own identifier
  and the one the pre-registered draw records; renaming it would break traceability.
  The caption already discloses it. Keep the id.
- *"The audit harness's UNBOUND count is the real measure of exposure."* — superseded.
  UNBOUND counts what a *census* cannot label; what matters is what an *assertion*
  cannot catch. That is now 117 anchored checks against an enumerated residue of 31
  coupled groups, which is a bounded and stated quantity rather than a guess.

**Open — what a round-2 reviewer could still object to.**

- 🔴 **`verify.py` is a whitelist, not a guarantee.** It asserts what it was told to
  assert. The concurrent lane found eleven R1.1 numbers outside `build_checks`, **one
  of which was wrong** — a "median ratio" that was really a ratio of medians, 16 % away
  from the number printed beside it. That is the shape of the remaining risk: not a
  number that drifted, but a number nobody registered. Any new quoted literal must be
  registered when it is written, not audited afterwards.
- 🔴 **31 of the 41 cross-file coupled groups are still unasserted.** Fourteen have one
  anchored location but no coupling; seventeen are untouched (`0.08, 0.266, 1.00, 1.45,
  11, 120, 14, 17.6, 2.2e8, 200, 32, 50, 5000, 60, 600, 66, 92`). Each is a number
  typed in two documents with nothing checking they agree — E1's exact mechanism.
- **The double-blind mirror is checked by neither harness.** Measured today it carries
  no drift: numeric-token streams are identical on 17 of 18 mirrored files, the sole
  delta being a citation year dropped by anonymisation. That is a measurement, not a
  guarantee, and it predates today's supplementary edits — **the mirror has not been
  re-synced since**, so it is now stale by two tables, a bibliography and four edits.
- **`*.bbl` is gitignored.** Four new citations mean the supplementary's bibliography
  must be rebuilt wherever it is compiled. Overleaf runs `bibtex` itself, so this
  should be invisible — but if any part of the workflow relies on an uploaded `.bbl`,
  those four come out undefined and nothing in the repository would catch it.
- **Appendix D.1 gained two tables, so every supplementary table number after it shifts
  by two.** `results.tex` currently hardcodes none, which is the state that must hold.
  T11's note that the k-range table is "Table 10" was already off by one before this
  session (it was 11); it is now **13**.
- **Three `expression` strings in the JSON inventory still carry an LHS or a prose
  annotation** (`I.12.4`, `II.3.24`, `Strogatz-shearflow1`). They reach no emitted
  LaTeX today because the renderer prefers `sympy_expression`, which is numerically
  exact for all three — but they are a trap for any future consumer that prints that
  field.
- **A reader comparing versions will see `k`, ρ and the cost axis move.** The letter
  discloses why (R2.3). §T16's random-graph figures must not be quoted as production
  magnitudes; every magnitude comes from C2.
