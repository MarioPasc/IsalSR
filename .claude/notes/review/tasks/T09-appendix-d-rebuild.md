# T09 — Appendix D rebuild and numerical consistency

| Field | Value |
|---|---|
| Reviewer comments closed | **R2.5**, **R2.6**, **R2.3** (and E1, E2, E8) |
| Type | Bookkeeping + reproducibility |
| Owner | **Mario** (+ Claude Code) |
| Depends on | T02 (authoritative campaign), T05 (added problems), T08 (cell counts) |
| Blocks | T13 |
| Status | **STRUCTURAL HALF COMPLETE; NUMBERS GATED ON C2.** AC-1, AC-2, AC-3, AC-5 met and re-verified in the main tree; AC-0 filled as work proceeded; AC-6 met structurally (E1, E2, E8 removed, both documents compile at 12/10 pages). **AC-4, AC-6's values, AC-7's passing run and AC-8 are gated on the C2 manifest and cannot be met by this ticket yet.** Appendix D.1 now generated from the benchmark definitions for **70/70** problems, nine mandatory fields each, 51 tests, deterministic. 🔴 **The §T16 "OPEN DECISION" is closed** — the response letter already discloses in full (`response_to_reviewers.tex:1132–1142`). 🔴 **§T16's number table must not be used as instructed**: those are random-graph measurements, production ρ is 1.7931/1.880 not 1.2960/2.1505. 🔴 **R2.6 is still an empty `\todoblock`** (`:1246`), needing only the run count. 🔴 **New unraised defect**: the manuscript claimed training sizes $\{20, 240, 1000\}$; **no problem uses 20**. |
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

---

## 8. Proposed answer

### 8.1 Before / after

Cells marked **⧗C2** are gated on the campaign manifest and are deliberately empty —
not overlooked. Everything else is measured and re-verified.

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
| Total runs, stated | 2,640 | ⧗C2 — recompute from manifest, not formula | AC-4 |
| Total runs, actual | 6,000 (for the submitted campaign) | ⧗C2 | AC-4 |
| Recurrence in Appendix D.3 | present | ⧗C2 | AC-4 |
| Operator sets in the manuscript | 4, undifferentiated | **5**, distinguished — the two hosts differ from each other | AC-5 |
| Host set stated | one shared $\{+,-,\times,\div,\sin,\cos,\exp,\log\}$ | per method: Bingo 10, UDFS 11 (no generic `pow`) | AC-5 |
| Host set actually used | varied by tier (hard tier added sqrt/pow) | uniform across the suite per method (A4b) | AC-5 |
| Σ_SR vs host set distinguished | no | yes, with the composition identities | AC-5 |
| N-8 / N-11 explained | no | yes — already in the letter's R2.3 block | AC-5 |
| k-overhead, 5 ≤ k < 15 (main / appendix) | 45.9 % / 47.0 % — disagree | main text no longer restates; Table 8 referenced once | AC-6 |
| k-overhead, 15 ≤ k < 32 (main / appendix) | 41.6 % / 49.9 % — disagree | as above | AC-6 |
| "35.5–56.0 %" range in main text | asserted; **0 hits** in `paper/*.tex` | removed, with its false attribution | AC-6 |
| ρ range in discussion | 1.45–1.96 | not restated; referenced to the appendices | AC-6 |
| ρ range, true union | [1.11, 1.98] | ⧗C2 | AC-6 |
| Cross-file duplicated measurements | ≈40, unaudited | ⧗C2 — harness built, inventory pending | AC-7 |
| Main paper page count | 12 | 12 (unchanged) | T13 |

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

**Not yet applied, and why**: the `2{,}640` run count (`supplementary.tex:560–561`,
`:574`) stays as submitted. AC-4 requires it be recomputed from the manifest; writing
a design figure now would repeat R2.6's error with a different number.

### 8.2b New deliverables in the repository

| Path | Role |
|---|---|
| `experiments/scripts/generate_appendix_d_tables.py` | Emits Appendix D.1 for all 70 problems from `benchmarks/datasets/*.py`; LaTeX + JSON from one pass. The structural fix for the root cause. |
| `tests/unit/test_appendix_d_generator.py` | 51 tests; hard-fails on any empty mandatory field, and 6 parametrised LaTeX-defect cases prove the validator is not vacuous |
| `docs/generated/appendix_d/appendix_d_tables.tex` | Generated appendix, 7 tables |
| `docs/generated/appendix_d/appendix_d_benchmarks.json` | Machine-readable inventory; the audit harness consumes it |
| `experiments/scripts/numerical_audit.py` | AC-7 harness (in progress) |

### 8.3 Draft response text

> **Status correction (2026-08-04).** This section assumes three response blocks need
> drafting. **Two are already written.** Verified in
> `reviews/response_to_reviewers.tex`:
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

**Open.**

- **The `2,640` → manifest run count is still unwritten.** It is the single most
  explicitly-checked number in R2.6, and it stays wrong in the manuscript until C2
  produces a manifest. If the campaign slips, this is the item that slips with it.
- **`test_4` would print verbatim in a TPAMI appendix.** It needs a display name.
  The ID cannot be renamed while the campaign is in flight, so this must be a
  presentation-layer mapping in the generator, applied before the appendix ships.
- **The audit harness's UNBOUND count is the real measure of exposure.** Every
  measurement it cannot bind to a source artefact is a number a reviewer could
  question and we could not answer from data. That count is the round-2 risk, and
  it is not knowable until the harness runs against C2's artefacts.
- **A reader comparing versions will see `k`, ρ and the cost axis move.** The letter
  discloses why (R2.3), but the continuity table must quote **production** magnitudes
  from C2 and not §T16's random-graph figures. A wrong magnitude there would be
  found by the one reviewer who checked every number in round 1.
