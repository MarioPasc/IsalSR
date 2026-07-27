# T09 — Appendix D rebuild and numerical consistency

| Field | Value |
|---|---|
| Reviewer comments closed | **R2.5**, **R2.6**, **R2.3** (and E1, E2, E8) |
| Type | Bookkeeping + reproducibility |
| Owner | **Mario** (+ Claude Code) |
| Depends on | T02 (authoritative campaign), T05 (added problems), T08 (cell counts) |
| Blocks | T13 |
| Status | NOT STARTED |
| Target | provenance 2026-08-17 · final tables 2026-09-08 |

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

_(empty — to be filled by the implementing agent)_

---

## 8. Proposed answer

### 8.1 Before / after

| Quantity | Submitted | Revised | Source |
|---|---|---|---|
| Suite description, Section IV.2 | "32 core (12 Nguyen + 20 Feynman) + 18 extension" | | AC-3 |
| True composition | 12 + 24 + 14 | | |
| Table 5 (`tab:feynman`) rows | 10 | | AC-2 |
| Feynman problems actually run | 24 | | |
| Problems documented in Appendix D.1 | 22 of 50 | | AC-2 |
| Problems undocumented | **28** | 0 | AC-2 |
| Total runs, stated | 2,640 | | AC-4 |
| Total runs, actual | 6,000 | | AC-4 |
| Recurrence in Appendix D.3 | present | | AC-4 |
| Operator sets in the manuscript | 4, undifferentiated | | AC-5 |
| Host set stated for all tiers | {+,−,×,÷,sin,cos,exp,log} | | AC-5 |
| Host set actually used (hard tier) | + sqrt, + pow (Bingo) | | AC-5 |
| N-8 / N-11 explained | no | | AC-5 |
| k-overhead, 5 ≤ k < 15 (main / appendix) | 45.9 % / 47.0 % | | AC-6 |
| k-overhead, 15 ≤ k < 32 (main / appendix) | 41.6 % / 49.9 % | | AC-6 |
| "35.5–56.0 %" range in main text | asserted, does not exist | | AC-6 |
| ρ range in discussion | 1.45–1.96 | | AC-6 |
| ρ range, true union | [1.11, 1.98] | | AC-6 |
| Appendix tables generated from data | no | yes | AC-1 |

### 8.2 Changes made to the manuscript

| File | Lines (revised) | Change |
|---|---|---|
| | | |

### 8.3 Draft response text

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

> Candidates: a fifth operator set surfacing somewhere unchecked; the generated
> appendix growing past the page budget (coordinate with T13 — a compact
> multi-column table may be needed); R2 re-checking the numerical audit list and
> finding an entry the script did not cover.
