# T10 — Claim calibration in the Discussion and Conclusion

| Field | Value |
|---|---|
| Reviewer comments closed | **R1.1** (and E3, the "near-linear" overstatement) |
| Type | Framing / honesty of claims |
| Owner | **Mario** (`discussion.tex`, `results.tex`) · **Ezequiel** (`conclusion.tex`) |
| Depends on | T02 (the new `S`), T01 (§8.1 projection), T03 (if Gray changes cost) |
| Blocks | T12, T13 |
| Status | **COMPLETE 2026-08-14.** All nine acceptance criteria met, every number personally re-derived. The seven `\pendingnum{}` placeholders are filled from campaign C2 and the letter now carries none. **Answer A is confirmed by the data and the concession got harder, not softer**: Bingo's `S` fell 0.93 → 0.72. AC-1 stands as a negative result (§4's break-even formula is refuted by the code's `S`), and a second premise failed on the data — there is **no ρ range in which Bingo's `S ≥ 1`**, because ρ is nearly constant on that host (corr(ρ, `S`) = +0.04); the letter says so and names run length (`r = +0.41`) as what actually predicts `S`. Four defects fixed beyond the original scope: the `T_eval/T_canon` median mislabel (4 sites), a false CD separation claim in `results.tex`, p-values without effect sizes in `conclusion.tex`, and a 3× understatement of the Bingo wall-clock cost in the abstract. **Three edits need Ezequiel's and Karl's sign-off — see §9.4.** |
| Target | 2026-09-10 |

---

## T16 impact — the cost side of `S` moves against us (added 2026-07-30)

T16 aligned the adapters to the paper's alphabet, so `Sub`/`Div` are now emitted as
`Add`+`Neg` / `Mul`+`Inv`. **Canonicalisation gets more expensive**, because the DAGs
are bigger: `k` +22.9 % (Bingo) / +22.0 % (UDFS), and per-DAG canonicalisation cost
**+24.6 %** (Bingo) / **+10.8 %** (UDFS) on a random-population probe.

**Why this matters here specifically.** R1.1's complaint is about `S = T_BL / T_IS`
being framed as "approximately neutral" at 0.93. The alphabet correction pushes the
IsalSR arm's canonicalisation cost **up**, so the honest expectation is that `S` does
not improve on this account and may worsen. **Do not write the calibration text
against the old cost numbers, and do not assume the C++ port's speedup and the
alphabet's cost increase cancel** — they are independent and both land in Wave 1.
Wait for the measured `S`.

The offsetting consideration, which is real but must not be oversold: on Bingo the
reduction factor was **exactly** invariant under the change (3,858 distinct strings
in both encodings), and on UDFS it *rose* 1.4 % because host-native `neg`/`inv` now
unify with decomposed ones. So the representation got strictly better at its job
while costing more per DAG. That is a trade to state plainly, not a win to claim.

Full write-up: `docs/md_files/changes/t16_commutative_decomposition.md`.

---

## 1. Why R1.1 and E3 are grouped

Same defect, same two files, same fix. In both cases **the Results and Introduction
state the claim correctly and the Discussion and Conclusion overstate it.** The
reviewer's complaint in R1.1 is precisely that mismatch, and E3 is the identical
mismatch on a different quantity that no reviewer happened to name.

| Claim | Stated correctly in | Overstated in |
|---|---|---|
| Bingo wall-clock effect | `results.tex:180–185` (conditional, per-problem) | `discussion.tex:66–68` ("approximately neutral") |
| Canonicalisation complexity | `introduction.tex:51–52`, `related_work.tex:81–82` (near-O(k²)) | `conclusion.tex:9–11`, `discussion.tex:97` ("near-linear") |

Fixing them together means one pass over the late sections with one rule: **no
section may claim more than the Results section supports.**

E3 carries an extra hazard. **Reviewer 1's own B2 significance statement uses
"near-O(k^2)"** — they have already written the correct complexity back to us. A
"near-linear" claim surviving into the revision reads as inconsistent with what the
reviewer themselves recorded, on a point they clearly understood.

---

## 2. Verbatim comment

> 1) The bingo search-only speedup is s = 0.93, a net loss under a fixed wall-clock
> budget, yet the paper characterizes the overhead as "approximately neutral." This
> needs a more honest framing. The claim holds only on the subset of problems where
> rho is large enough to compensate.

---

## 3. Established facts

### 3.1 R1.1
- `S = 0.93` for Bingo, `S = 1.07` for UDFS — `results.tex:57–58`, Table 2.
- The objectionable phrasing, `discussion.tex:66–68`:
  > When evaluation cost is comparable to canonicalisation cost, as in Bingo where
  > both are sub-millisecond, the median overhead is 39 % and the wall-clock effect
  > is **approximately neutral** after the search-time savings offset the
  > canonicalisation cost.
- The Results section is **already** properly conditional, `results.tex:180–185`:
  > on the 30 problems where Bingo exceeds ρ = 1.85 the search-only speedup recovers
  > to S ≥ 0.95, and on three problems — N-8, II.11.27, and Keijzer-11 — it exceeds
  > unity.
- Per-problem corroboration, `table_supplementary_bingo.tex`: only **4 of 50** rows
  have `T_IS < T_BL` (I.6.20a, II.11.27, Keij-11, N-8). The other 46 are slower under
  IsalSR. Highest overheads are Nguyen rows: N-12 61.3 %, N-5 60.5 %, N-4 59.6 %,
  Keij-6 60.3 %, N-3 58.2 %, N-7 56.8 %.
- The **abstract is accurate** (`main.tex:81`, "at a median of 39 % in the
  sub-millisecond evaluation regime"). The overstatement is localised to the discussion.

### 3.2 E3 — three coexisting growth claims
`methodology.tex:885–887` derives O(k) insertion steps × O(k) candidates = **O(k²)**.
`supplementary.tex:736, 793` fits **O(k^0.7)** *per permutation* — a per-call cost,
not the whole-DAG canonicalisation cost. `discussion.tex:22–24` infers "near-linear
behaviour" from median per-DAG times. Three different growth claims coexist with no
stated relationship between them. Fixing E3 means stating that relationship once,
not just deleting the word "near-linear".

---

## 4. Two possible answers — prepare both, choose on T02's data

**Answer A — the concession (if `S` stays below 1 for Bingo).**
Accept the criticism, delete "approximately neutral", and replace it with the
regime characterisation the Results section already contains: IsalSR's wall-clock
benefit is realised when `T_eval / T_canon` is large (UDFS: ≈ 64:1, `S = 1.07`) and
is a net cost when the two are comparable (Bingo: ≈ 1:3.3, `S = 0.93`); the
crossover is at ρ ≈ 1.85 given the measured cost ratio. **State the break-even
condition as a formula**, not as a list of problems. A reader can then predict
whether IsalSR pays on their own solver, which is more useful than the original
sentence and is a genuine contribution rather than a retreat.

**Answer B — the fix (if the C++ engine pushes Bingo's `S` above 1).**
Report the new `S`, state that the previous figure reflected a Python
implementation, and *still* give the break-even formula — because the regime
characterisation is what makes the result generalise, and because dropping it would
look like the concession was avoided rather than earned.

**In both cases the break-even analysis is the deliverable.** Do not write Answer B
without it.

Derive it explicitly. With `ρ` the reduction factor, `T_e` the mean evaluation cost
and `T_c` the mean canonicalisation cost, the search-only speedup is

```
S  =  ρ / (1 + ρ · T_c / T_e)          ⇒     S > 1  ⟺  ρ > 1 + ρ·T_c/T_e
```

Verify this against the definition of `S` actually used in the analysis code
(`models/analyzer/`) before publishing it — do not reproduce the formula above on
trust. Then report the measured `T_c/T_e` per method and the implied break-even ρ.

---

## 5. Mandatory reading

- `.claude/notes/review/source/reviewer-1.md` — §R1.1 and the **full B2 statement**
- `.claude/notes/review/source/verified-discrepancies.md` — D11, E3
- `.claude/notes/review/tasks/T01-cpp-core-port.md` — §8.1, the projected `S`
- `.claude/notes/review/tasks/T02-cpp-reexecution-campaign.md` — §8.1, the measured `S`
- `.claude/notes/review/tasks/T03-gray-code-integration.md` — if promoted, cost changes again
- `docs/md_files/changes/bottleneck_type_analysis.md` — the existing regime analysis;
  reuse its framing rather than inventing a second one
- Source: `article/paper/{results,discussion,conclusion}.tex`, `main.tex:81`

---

## 6. Work specification

1. Verify the `S` definition in the analysis code and derive the break-even
   condition from it.
2. Compute `T_c/T_e` and the break-even ρ per method under the C++ engine.
3. Rewrite `discussion.tex:66–68`. Choose Answer A or B on the data; keep the
   break-even statement either way.
4. Sweep for every other unconditional wall-clock claim in `discussion.tex` and
   `conclusion.tex`.
5. Fix E3: replace "near-linear" at `conclusion.tex:9–11` and `discussion.tex:97`
   with the derived near-O(k²); reconcile `discussion.tex:22–24`; state once how the
   per-permutation O(k^0.7) fit relates to the whole-DAG O(k²) bound.
6. Add the per-problem count (`x of 50` rows with `T_IS < T_BL`) to the discussion.
   It is the single most direct answer to "the claim holds only on the subset of
   problems where rho is large enough" and it is currently only inferable from the
   supplementary tables.

---

## 7. Acceptance criteria

- **AC-0.** §8 Work log filled in as the work proceeds.
- **AC-1.** `S` definition verified against code; break-even condition derived and
  checked, not assumed.
- **AC-2.** "Approximately neutral" is gone.
- **AC-3.** Break-even condition stated in the paper as a condition on ρ and
  `T_c/T_e`, with the measured values per method.
- **AC-4.** The count of problems where IsalSR is wall-clock faster appears in the
  main text.
- **AC-5.** No unconditional wall-clock claim survives in `discussion.tex` or
  `conclusion.tex`.
- **AC-6.** "Near-linear" is gone from both locations; the relationship between
  O(k^0.7), the observed medians, and the O(k²) bound is stated once.
- **AC-7.** The revised claims are checked against the abstract and the introduction
  for consistency — those two are currently correct and must stay correct.
- **AC-8.** §9 filled.

---

## 8. Work log

### 2026-08-14 — reopened on T02's completion; plan for the closing pass

T02's campaign C2 completed on 2026-08-14 (12,600 cells, certifier GO 19/19), and
the numbers were carried into all five documents by commit `1f9a77d`. **T10's
blocker is gone and its seven `\pendingnum{}` placeholders are already filled** —
`grep` returns one occurrence in the letter and it is the macro definition itself.
What is left is therefore not writing but *verification*: the ticket's own standing
rule is that a number nobody re-derived is a number nobody checked.

Plan for this pass:

1. Run `experiments/scripts/review_campaign/verify.py` in the main tree. It asserts
   each quoted literal against `analyses/values/summary.json` and exits non-zero on
   drift. **Done — 49 literals, 16 derived-value gates, all pass; 0 placeholders in
   the paper, 0 in the letter, 12 in the supplementary (the synthetic permutation
   study, not T10's).**
2. Verify the **eleven** R1.1/discussion numbers `verify.py` does *not* cover
   (the two `T_eval/T_canon` medians, the UDFS suite-level `S`, the ρ–`S`
   correlation, the two ρ intervals, the 66-of-70 count, the run-length
   correlation, the UDFS 29-of-70 count, the CD mean ranks, the 1.45× slowdown,
   the Korns-12 triple). Delegated read-only, with the *median-of-ratios versus
   ratio-of-medians* distinction called out explicitly as a fail condition.
3. Re-sweep AC-5 and AC-7 against the current text rather than against the
   2026-08-06 state, since the whole Practical Impact subsection was rewritten
   after that entry was written.
4. Judge over- **and under**-claiming in both directions, and close.

Two facts established before delegating, both of which change what R1.1 may say:

- **The baseline arm was re-run.** The 2026-07-27 decision (README) was that the
  native arm would *not* be re-executed on the original 50 problems. The campaign
  as executed is 2 methods × 3 arms × 70 problems × 30 seeds = 12,600 cells, and
  `verify.py` confirms 2,100 cells per (method, arm) — i.e. all 70 problems in
  every arm. The letter's claim that the comparison has "a baseline re-run on the
  same hardware in the same campaign rather than inherited from an earlier one" is
  therefore true, and the residual wall-clock confound that `EXECUTION-PLAN.md` §5
  reserved a disclosure for **no longer exists on this contrast**. Do not carry
  that disclosure into R1.1.
- **The Bingo number moved against us**, from `S = 0.93` to `S = 0.72`. The C++
  engine's speedup did not cancel T16's alphabet cost; §15's warning was correct in
  direction. R1.1 states this outright and names it a worse ratio than the one the
  reviewer objected to. That is the right call and it is not to be softened.

### 2026-08-14 — R1.2's measurement was already in the campaign; `reachability/` built from it

Asked to make the T06 numbers self-contained. Investigating where to copy them
from turned up something better: **every deduplicating cell of C2 already writes a
`fallback_ledger.json` carrying exactly the quantities R1.2 asks for**, and nobody
had pooled them. Mario's rule — *"everything we can re-derive from the C2 campaign
rather than re-computing is worth doing"* — then settles the whole question.

`…/results/review/c2_3arm/reachability/` now holds three files, all campaign-derived:
`README.md`, `reachability.json`, `reachability_cells.csv` (8,400 rows). Generator:
`experiments/scripts/review_campaign/reachability.py`, exits non-zero on a failed check.

| Quantity | T06 probe (what the letter quotes) | C2-derived |
|---|---|---|
| Candidates | 154,568 Bingo / 3,890 UDFS | **17,270,162,980** |
| Violate on arrival | 85.88 % / 100 % | **74.72 %** — 74.15 % Bingo, **100.0000 %** UDFS |
| Violate after `𝒩` | 0 | **0** |
| Bypass events (4 paths) | 0, bound 1.64e-5 | **0**, bound **1.74e-10** |

**Six validation checks, all 8,400/8,400** — and the first two are the point, not
ceremony: `instrumentation_enabled` and `population_non_empty`. This is SP-6, the
failure that cost a 1,260-run wave: a disabled counter and a counter that saw
nothing both report zero. Without those two checks, C4's four zeros would be
indistinguishable from "nothing was measured". The other four are `readable`,
`full_census` (`sample_rate = 1`, `n_seen == n_sampled`, so these are census rates
and not estimates), `histograms_sum_to_counters`, and
`violations_never_exceed_samples`.

**An independent cross-check on what is being counted.** Ledger candidates ÷
campaign evaluations reproduces `ρ` through separate code: `udfs/hash` **1.0000**
against `ρ_σ = 1.0000` — exact, to the unit, over 146,745,434 candidates, which it
must be because the naive hash provably removes nothing on UDFS — then 1.6837,
1.7604, 1.8201 against 1.6637, 1.7270, 1.7850.

**Cleaned on instruction.** The first cut of this folder also carried the T06/T15
local probes (`norm_arms/`, `t06_ledger/`). Mario: *"only numbers derived from the
campaign"*. Removed, after preserving the Picasso-recovered UDFS half — which
existed in no local copy — at
`/media/mpascual/Sandisk2TB/research/isalsr/results/t15_norm_arms/udfs/`. Nothing
was lost; the folder no longer invites a comparison between populations five orders
of magnitude apart.

**What this unblocks and what it changes.**

- The provenance objection to writing the Appendix D block is **gone**. It can be
  sourced from a committed, regenerable artefact instead of ticket prose.
- **The letter's R1.2 figures are now the weaker evidence** and should be replaced
  wholesale, not patched. Whoever writes that block should also drop four claims
  the extraction found unsupportable: the bounds are **Wilson**, not "one-sided
  Poisson" as the letter says; `atlas_hit = 0` is zero **by configuration**, the
  atlas being disabled, not by a live path declining to fire; T06's fourth
  population (49,980 synthetic DAGs at 0 %) is **vacuous**, its generator emitting
  no `Const` so it cannot exhibit the violation; and the `ρ = 1.80 → 3.91`
  timeout-bias pair quoted in the letter **has no artefact at all**, the surviving
  one giving 1.7759 → 2.6595, which T06 itself records must not be quoted.
- `74.72 %` is a pooled census dominated by Bingo's long runs, not a per-problem
  mean. `reachability_cells.csv` carries the distribution.

Certified tree untouched — `find -newermt` confirms nothing outside `reachability/`
was written. Artefact byte-reproducible on re-run; ruff clean; all 17 figures in
its README re-checked programmatically against the JSON.

### 2026-08-14 — audit findings fixed on Mario's instruction; two items blocked on the supplementary lane

Mario: *"Fix everything you have found. Don't worry about the hardcoded appendix
references, I will reconcile them in a different ticket."* Fixed accordingly, in
files across four tickets' lanes. **Recorded here rather than in each ticket
because a fix nobody can find is a fix nobody can audit** — T06, T07, T11, T12 and
T14 each need to be told.

| # | Was | Now | File |
|---|---|---|---|
| 1 | *"count timed-out DAGs as unique, which is conservative"* — the letter itself concedes at `:529` that the implementation does the opposite | Corrected: a timed-out candidate is evaluated **without** its string entering the dedup set, so the bias on ρ is **upward**, not conservative; and no timeout occurred on any of the 12,176,790 Bingo or 234,865 UDFS candidates, so nothing depends on it | `discussion.tex:166` |
| 2 | *"The effective number of paired seeds … is reported per problem in the appendix tables"* — the tables assert a uniform 30 | Measured and stated: **exactly 30 on every problem, host and contrast**. Verified myself: 12,600 cells, **zero** non-finite `r2_test`, all 420 (method, problem, arm) groups at 30. R2.7's NaN failure does not recur | `computational_experiments.tex:248` |
| 3 | The `\changeref` promised a Section 6 future-work remark on `𝒩` that did not exist | **Written**, rather than deleting the promise: the creation edge is an artefact of Σ_SR, and an insertion that creates a node without an edge would remove `𝒩` and weaken the reachability condition, at the cost of a larger string space | `discussion.tex`, Limitations |
| 4 | Letter claimed the wall-clock winner count "moves into the main text", then gave 13 (Bingo) **and 29 (UDFS)**; only 13 was there | UDFS's 29 added, and **registered in `verify.py`** (121 checks now) so it cannot drift | `discussion.tex:78`, `verify.py` |
| 5 | R2.8/R3.2: *"it now reads ``on both methods''"* | The clause was not repaired, it was **deleted** — the campaign replaced the quantity it reported. Both answers now say so | letter `:1831`, `:2195` |
| 6 | R2.8: *"The campaign that replaces them is running … we have not written provisional values"* | Replaced with what is true: every abstract figure is the re-executed campaign's, produced by the pipeline that produces the tables, with a check asserting each against it | letter `:1912` |
| 7 | R2.4: *"The main document has three tables"* | Submitted had three, the revision has five, and the FCS pseudocode is in neither | letter `:1526` |
| 8 | R1.3: claimed **pseudocode** for `𝒩`; `\changeref` claimed it sits "immediately before the D2S insertion rules" | Pseudocode claim dropped (Appendix C holds only S2D, D2S, FastCanonical); position corrected to §3.6 | letter `:561`, `:821` |
| 9 | R1.4 `\changeref`: tree-hashing precedent "in Sec. 2.4" | §2.3 — 2.4 is *Benchmarking in SR* | letter `:1002` |
| 10 | R2.7 `\changeref`: "Tables 6 and 7 … per-problem effective seed count"; "Korns-12 a tie at $R^2 = 0.0000$ under both variants" | Table numbers dropped for a description; Korns-12 restated for **three** arms as 0.000 / 0.016 / 0.000 | letter `:1810` |
| 11 | R2.5: present-tense "Tables 6--7 **are** per-problem result tables" | Past tense, plus a note that both have been renumbered by the appendix additions | letter `:1583` |
| 12 | R3.1 `\changeref`: "new table" in App. D.1 | Two tables, documenting all seventy | letter `:2177` |
| 13 | Summary-of-changes R2.7 row pointed at "Tables 6--7" | App. D | letter `:260` |
| 14 | **`previously_published_statement`** described *the journal manuscript* with the submitted campaign's numbers: 50 problems, 8 sources, ρ = 1.56/1.83, d = 2.38, p = 2.7×10⁻²², overhead 39 %, and a top-gain problem list from the old campaign | Rebuilt on C2: 70 problems, 9 sources, three arms, ρ = 1.66/1.79, d = 2.54/7.05 at p < 10⁻¹², R² p with effect sizes, overhead 0.04 %/16.1 % **and the 1.45× Bingo cost**. Top-gain list **recomputed** from `per_problem.csv` (I.30.3, Strogatz-lv2, I.12.4, III.17.37, II.34.29a), not transcribed. Supplementary page count 13 → 18 | `previously_published_statement/main.tex` |

**Verification.** `verify.py` **121/121**, ruff clean. Paper 18 pp, `main_anonymous`
18 pp, letter 36 pp, previously-published statement — all 0 errors, 0 overfull,
0 undefined references. Three double-blind paper mirrors byte-identical;
`double_blind/paper/computational_experiments.tex` carries its anonymisation delta
and received the same edit separately.

**Blocked on the supplementary lane** (held by the T09 session for a compute job
filling its twelve remaining `\pendingnum`):

1. **The Appendix D subsection for R1.2.** Letter `:539` still says the
   violation-rate measurements "are collected in Appendix~D"; they are collected
   nowhere. **This sentence must not ship until the appendix exists**, and it is
   the last false claim left in the letter.
2. `supplementary.tex:983–984`, the twin of fix 1 above.
3. `double_blind/supplementary/supplementary_anonymous.tex`, stale by 59 lines and
   both `bench_struct` tables — the double-blind tree documents **42 of 70**
   problems where the live tree documents 70 of 70 (**T14**).

**A provenance gap Mario must rule on.** T06 is closed and its numbers are real,
but they live at `picasso:~/execs/isalsr/t15_norm_arms/udfs/` and in per-run
`run_log.json`, **not** in a committed artefact — there is no `reachability.json`
answering to `appendix_d_benchmarks.json`. Everything else in the revision now
stands on a generated file that `verify.py` can assert against. If the Appendix D
block is written from the ticket's recorded measurement, it will be the one
headline block in the revision that is not reproducible from the repository.
Reporting it is still strictly better than R1.2's status quo, in which the
quantity is never reported at all — but the decision to pull T06's JSON down and
commit it is Mario's, and it should be taken before the appendix is written, not
after.

### 2026-08-14 — letter audit: three broken cross-references fixed in `discussion.tex`; six defects filed out of lane

Asked after T10 closed: *what is left to write in the letter, and is anything in
it stale against the article?* Two independent read-only audits — one over all 18
`\changeref` usages, one over the letter's prose assertions — converged on the
same two defects, which is the agreement that made them worth acting on.

**In lane, fixed.** The revision inserted the naive-hash comparator as a **new
Appendix E**, so everything after it shifted a letter: scalability moved E → **F**.
`discussion.tex` hard-codes its appendix pointers rather than using `\ref`, and
all three were left behind:

| Line | Was | Should be | Target |
|---|---|---|---|
| 36 | Appendix E.2 | **F.2** | `sec:supp_scalability_synthetic` — the 5,400-DAG study (9 k × 3 m × 200 = 5,400, checked) |
| 58 | Appendix E.1 | **F.1** | `sec:supp_scalability_empirical` — per-`k` over the 70 problems |
| 159 | Appendix E.1 | **F.2** | the `O(k^{0.7})` fit is the *synthetic* study; this one had **both** parts wrong |

All three now point at the naive-hash appendix in the shipped PDF. This is R2.4's
own defect — a cross-reference to something that is not there — recurring inside
the revision that answers it. Fixed, double-blind mirror re-synced, paper rebuilt
(18 pp, 0/0/0), `verify.py` 120/120.

**The root cause is not fixed and cannot be fixed here.** ~20 hardcoded
`Appendix~X` pointers remain across `methodology.tex`, `computational_experiments.tex`
and `results.tex`; every one of them re-breaks if another `\section` is inserted
into the supplementary. The paper and the supplementary are separate documents, so
`\ref` does not resolve across them without `xr`. **T11 owns cross-document
consistency and should carry a standing check**, not a one-time sweep.

**Out of lane, filed with evidence, not absorbed:**

| # | Defect | Owner |
|---|---|---|
| 1 | **Letter `:539` and its `\changeref` `:541` tell R1 the reachability violation-rate measurements "are collected in Appendix~D".** They are in no `article/` file. Nine independent search patterns — `violation rate`, `violation-rate`, `85.88`, `154{,}568`, `3{,}890`, `12{,}176`, `234{,}865`, `reachability failure`, `fail the precondition` — return **zero hits** across all paper `.tex` and the supplementary. A promise to a reviewer, about that reviewer's own comment, that the manuscript does not keep | **T06** |
| 2 | Letter `:1905–1912` (R2.8) says the abstract's figures "are those of the submitted campaign" and "the campaign that replaces them is running". C2 finished 2026-08-14 and the abstract carries its figures under `verify.py`. Three false clauses in a paragraph opening *"we would rather say so than let it be discovered"* | **T12** |
| 3 | Letter `:1831` (R2.8) and `:2195` (R3.2) both say the abstract "now reads ``on both methods''". That phrase is absent — now **and at HEAD before my edits**, so not a regression of mine. The abstract says "under both hosts" | **T12** |
| 4 | Letter `:561` claims the revision gives `normalize_const_creation` **pseudocode**. Appendix C holds exactly three algorithm tables — S2D, D2S, FastCanonical. Definition 3.16 is prose | **T07** |
| 5 | Letter `:1525` says "the main document has three tables"; `main.aux` lists **five**. Letter `:257–260` and `:1582` point at supplementary "Tables 6--7" for per-problem results; those are now the new benchmark-inventory tables, and the per-problem tables are 10–11 | **T11** |
| 6 | Letter `:1836` says the previously-published statement carries the corrected headline figures. It still reports the 50-problem campaign (ρ = 1.56/1.83, d = 2.38, p = 2.7×10⁻²², overhead 39 %) | **T11** |
| 7 | **`double_blind/supplementary/supplementary_anonymous.tex` is stale**: 1,670 lines / 4 `\input`s against the live 1,729 / 6, missing both `bench_struct` tables — so the double-blind tree still documents 42 of 70 problems while the live tree documents 70 of 70. Under double-blind that is the tree reviewers read, and **no harness checks it**: `verify.py`'s `GROUPS` is `{paper, supplementary, letter}` | **T14** |

**One imprecision in R1.1 itself, left as written and recorded here.** The letter
says the count of wall-clock winners "moves into the main text", then gives
13 of 70 for Bingo *and* 29 of 70 for UDFS. Only the Bingo count is in
`discussion.tex`; `29` appears nowhere in `article/paper/*.tex`. The sentence is
defensible — the count the reviewer asked about is Bingo's, and that one did move
— but a reviewer who greps for 29 will not find it. Either add the UDFS count to
`discussion.tex` or attribute it to the letter. **Flagged for T14's final pass
rather than changed unilaterally**, because it trades against T13's page budget.

### 2026-08-14 — every R1.1 number re-derived; one real defect, three overclaims removed

**The pipeline verifier passes, and passing it was not sufficient.**
`experiments/scripts/review_campaign/verify.py` asserts 117 checks and all pass,
but its coverage is the set of literals someone thought to register. Eleven
R1.1-bearing numbers sat outside it. Recomputed from
`analyses/data/*.csv` (script kept at `scratchpad/t10_recompute.py`):

| # | Printed | Recomputed | Verdict |
|---|---|---|---|
| 1 | Bingo "median per-DAG ratio `T_eval/T_canon` of 20" | median-of-ratios **23.21**; ratio-of-medians **20.08** | **FAIL — mislabelled** |
| 2 | UDFS "median ratio exceeds 5,000" | 6851.1 / 5761.5 | passes as an inequality, same mislabel |
| 3 | UDFS suite-level `S = 1.00` | 1.0005 over cells | ✅ |
| 4 | Bingo corr(ρ, `S`) `= +0.04` | Pearson +0.0435, N = 70 | ✅ |
| 5 | 22 with `S ≥ 1` over ρ ∈ [1.77, 1.84]; 48 over [1.19, 1.85] | 22 / 48; [1.7727, 1.8377] / [1.1889, 1.8530] | ✅ |
| 6 | \|ρ − 1.79\| ≤ 0.09 on 66 of 70 | 66 | ✅ |
| 7 | `r = +0.41` vs log native wall clock | Pearson +0.4066 | ✅ |
| 8 | UDFS 29 of 70 faster end to end | 29 | ✅ |
| 9 | CD mean ranks 1.51 / 2.10 / 2.39 | baseline 1.5143, hash 2.100, isalsr 2.3857 | ✅ |
| 10 | Bingo median seed-matched slowdown 1.45× | 1.4462 | ✅ |
| 11 | Korns-12 R²: 0.000 / 0.016 / 0.000 | 0.000000 / 0.015980 / 0.000000 | ✅ |

**Item 1 is the defect and it is now fixed.** "A median per-DAG ratio of 20" is
the *ratio of the two medians* (1.48 / 0.074), not the *median of the per-cell
ratios*, which is 23.21 — 16 % higher. The letter stated it as an equality, so
the two readings are not interchangeable there. Relabelled to "a ratio of median
per-DAG costs" at all four sites (`letter` R1.1 ×2, `discussion.tex:68`,
`results.tex:296`) rather than swapping in 23, because 20 is reproducible by a
reader dividing Table 2's two printed medians and 23 is reproducible from
nothing in the paper. The relabel also does not flatter us: 20 is the *smaller*
ratio. The same mislabel in the provenance ledger (`values.py:406, 513`,
`"median ratio of…"`) is corrected.

**Two overclaims removed that no reviewer raised.**

1. `results.tex:149` said *"On UDFS the three separate"* on the R² critical-
   difference axis. They do not. Mean ranks are 2.25 (native), 2.393 (naive
   hash), 1.357 (IsalSR); the native and naive-hash arms differ by **0.143**
   against a critical difference of **0.396**, and `critical_difference.json`
   puts them in one clique. Only IsalSR separates. Rewritten to say exactly
   that — which is the *stronger* claim for IsalSR as well as the true one.
2. `conclusion.tex:23–26` reported the R² paired test as two p-values and called
   them "what settles the effect of deduplication on regression quality", with
   no effect size. Against Table 2's Bingo row — 0.976 / 0.975 / 0.976, mean
   difference **+0.0006**, `d = +0.08` — a bare `p = 7.5×10⁻⁴` reads as a quality
   claim the data does not carry. Effect sizes added on Mario's decision, and
   the sentence now separates the two hosts: improvement where the budget binds,
   unchanged where the host saturates. **`conclusion.tex` is Ezequiel's file; this
   is the second T10 edit in it and both need flagging to him.**

**The abstract understated the Bingo cost by roughly 3×.** It reported the
canonicalisation overhead (16.1 %) and no wall-clock verdict, so a reader of the
abstract alone infers a 16 % cost where `results.tex:311` measures **1.45×**.
That is the R1.1 defect surviving in the most-read part of the paper. On Mario's
decision the closing sentence now carries one clause of cost —
*"…and $16.1\%$ on Bingo, where the deduplicated search is $1.45\times$ slower end
to end"* — and the abstract was tightened from **343 to 297 words** by cutting two
explanatory relative clauses and four redundant phrases. No advantage was
removed. `main.tex` is Ezequiel's and the abstract pass is **T12/Karl's** — this
is a substantive edit in both their lanes and must not be silently inherited.

**A √2 error survives in a generated artefact.** `analyses/values/critical_difference.json`
carries `cd_value = 0.5603` on **every** per-host axis. The Demšar-correct value
for k = 3, N = 70 is `q₀.₀₅(3,∞)/√2 · √(3·4/(6·70)) = 0.3962`, and
0.5603 = 0.3962·√2 — i.e. the raw studentized range, the exact bug that
`statistical_tests.py:233` was fixed for and that
`tests/unit/test_nemenyi_critical_difference.py` pins. **No printed number is
affected**: the paper's 0.40 and 0.90 are correct (I recomputed both against
scipy), nothing in `review_campaign/` writes or reads that file, and
`cd_diagram.py` goes through `critdd` directly. Every clique conclusion in the
manuscript holds under either threshold. But the file is a provenance trap for
anyone who reads it as the source of the figure. **Filed, not absorbed — it
belongs to whoever owns the CD figure.**

**Not a defect, checked and cleared.** `discussion.tex:110` says IsalSR
"preserves regression quality" on cheap-evaluation workloads while the paired
test returns a significant `+0.08` on Bingo. That is the right word, not an
under-claim: the means agree to three decimals and `d = 0.08` is negligible.
Calling it an improvement would be the overclaim.

**Verification.** `verify.py` 117/117 after the edits (two of its anchors were
tied to abstract phrasing I rewrote; the anchors were re-pointed, the checks were
not weakened). `pytest tests/unit -k "statistical or analyzer or aggregation or
nemenyi or nan_integrity"` → 100 passed. `ruff` clean. All four documents build
at 0 errors, 0 overfull boxes, 0 undefined references: paper 18 pp,
`main_anonymous` 18 pp, letter, and the three double-blind mirrors
(`discussion`, `results`, `conclusion`) restored to byte-identical after my edits
broke them.

### 2026-08-06 — `S` verified against the code; §4's break-even formula does not survive it

**AC-1, and it came out negative.** The formula proposed in §4,

```
S  =  ρ / (1 + ρ · T_c / T_e)
```

is **not** what the code computes, and it must not be published. What the code
computes, and what the manuscript defines, is a ratio of two *measured* times:

| Where | Statement |
|---|---|
| `computational_experiments.tex:186–195` | `T_search = T_total − T_canon`; baseline has `T_canon = 0`; `S = T_search^BL / T_search^IS` |
| `experiments/models/bingo/translator.py:116–124` | `search_only = wall_clock − canonicalization_time − conversion_time − shadow_time` |
| `experiments/models/udfs/translator.py:116–124` | identical |
| `experiments/models/analyzer/aggregation.py:397–398` | `speedups.append(m.baseline_mean / m.isalsr_mean)`, guarded to time metrics |

So `S` is an empirical ratio of realised search times, not a model. Two
consequences, both of which changed what got written:

1. **Under an exhausted, equal wall-clock budget `S > 1` is forced.** If both arms
   run for the same `T`, then `T_search^BL = T` and `T_search^IS = T − T_canon`, so
   `S = T/(T − T_canon) ≥ 1` identically. Bingo's `S = 0.93` is therefore only
   possible because Bingo terminates on convergence rather than on the budget
   (`evolve_until_convergence(max_time=…)`), so the two arms stop at different wall
   clocks. The supplementary per-problem table corroborates this: Bingo's `T_BL`
   and `T_IS` differ per row and sit far below 43,200 s on many problems, while
   every UDFS row is pinned at the budget.
2. **The submitted discussion sentence was wrong in a second way**, beyond the
   adjective. "The wall-clock effect is approximately neutral *after the
   search-time savings offset the canonicalisation cost*" describes an end-to-end
   quantity, but `S` has the canonicalisation cost **already removed** from the
   IsalSR arm before the ratio is taken. `S < 1` therefore says something sharper
   than "the overhead was not recovered": with the overhead discounted, the
   deduplicated search still needed more time. That is the reading the letter now
   uses, and it is the strongest honest form of the concession.

Any correct break-even model would need an extra assumption about the termination
rule (budget-exhausted versus convergence-terminated), and would give a different
expression in each case. **§4's derivation and §6 step 1–2 are therefore
retired**; AC-3 is re-scoped from "state the formula" to "state the condition in
terms of the two measured quantities `ρ` and `T_eval/T_canon`, and report the
measured crossover", which is what the letter and the discussion now do. This is
exactly the risk §9.4 anticipated.

### 2026-08-06 — R1.1 written with visible placeholders; E3 closed

**R1.1.** Written into `reviews/response_to_reviewers.tex`, replacing
`\todoblock{R1.1}`. A macro was added to the letter preamble next to `\todoblock`:

```latex
\newcommand{\pendingnum}[1]{{\color{todored}\textsf{[PENDING: #1]}}}
```

Seven placeholders, all campaign-dependent, all rendered red:

| # | Awaits |
|---|---|
| 1 | median per-DAG `T_eval/T_canon`, Bingo |
| 2 | median canonicalisation overhead as % of total run time, Bingo |
| 3 | search-only speedup `S`, Bingo, suite-level value for Table 2 |
| 4 | median per-DAG `T_eval/T_canon`, UDFS |
| 5 | search-only speedup `S`, UDFS, suite-level value for Table 2 |
| 6 | the ρ range over which Bingo's `S ≥ 1`, **or** an explicit statement that no such range occurs |
| 7 | count of suite problems where Bingo's IsalSR arm finishes in less total wall-clock time, with the suite size |

Placeholder 6 is deliberately phrased so it cannot be filled with a
direction-assuming number. The answer contains **no table**: a table of pending
cells is worse than none.

Two values are quoted outright because they are facts about the **submitted**
manuscript rather than about the campaign, and both are traced to §3.1: `S = 0.93`
(which the reviewer quoted) and `4 of 50` Bingo problems with `T_IS < T_BL`. Both
stay true whichever way the re-execution moves.

The answer also discloses, unprompted, that the submitted numbers came from a
Python canonicaliser and from the pre-T16 encoding, that the compiled engine
lowers the per-canonicalisation cost while the commutative normal form raises it,
and that we do not predict the net effect. It volunteers the protocol limitation
(equal wall-clock budgets; an evaluation-count budget would report a different
number) without claiming which direction that would move.

**E3 — complete, no campaign number needed.** Three occurrences in the annotated
copy:

| Location | Was | Now |
|---|---|---|
| `conclusion.tex:11` | "in near-linear time" | "in near-$O(k^2)$ time", blue |
| `discussion.tex:97` | "runs in near-linear time" | "runs in near-$O(k^2)$ time", blue |
| `discussion.tex:21–26` | two median timings "indicate near-linear behavior" | the medians now evidence that the greedy path is taken almost always; growth attributed to the near-$O(k^2)$ analysis of `Section~\ref{sec:canonical}`, blue |

The relation is stated once, at `discussion.tex:100–107`: the `O(k^0.7)` power law
is a log--log fit over `k ≤ 9` on synthetic DAGs, so it characterises the
benchmarked range and not the asymptotics, and it sits below the `O(k^2)` bound
because backtracking seldom fires at those sizes.

`related_work.tex:70` also contains "near-linear", but it describes **1-WL
hashing's ability to distinguish graphs**, not our canonicalisation cost. It is
correct and was left alone. `introduction.tex:82` and `related_work.tex:81` were
already correct.

**AC-5 sweep.** The one surviving unconditional wall-clock claim was
`discussion.tex:72–74`, "at neutral wall-clock cost". It is gone. `conclusion.tex`
makes no wall-clock claim.

**Verification.** Letter: two `pdflatex` passes, exit 0 both, `Overfull` 0,
unresolved references 0. Annotated paper: three passes, `^!` 0, undefined
references 0, `color{red}` 0, review notes 0; Theorems 3.13/3.14/3.15 keep their
numbers. `article/` untouched.

**One warning is not ours.** The letter's log carries
`LaTeX Warning: Float too large for page by 3.64752pt` on the summary-of-changes
table, which a parallel agent was editing during this session. `git show HEAD` of
the letter compiles with zero warnings, and nothing in the R1.1 block or the
preamble macro enters that float. Owner of that table needs to shorten a cell.

---

## 9. Proposed answer

### 9.1 Before / after

| Quantity | Submitted | Revised | Source |
|---|---|---|---|
| `S`, Bingo | **0.93** | **0.72** — moved *against* us | C2, verified |
| `S`, UDFS | 1.07 | **1.96** unsaturated / **1.00** suite median (1,727 of 2,100 cells at the cap) | C2, verified |
| Discussion characterisation | "approximately neutral" | **removed**; net cost stated outright, regime given as a joint condition on ρ and `T_eval/T_canon` | AC-2 ✅ |
| Break-even condition stated | no | **yes, as a condition on the two measured ratios** — not as §4's formula, which the code refutes (§8) | AC-3 ✅, re-scoped |
| `T_e / T_c`, Bingo | 1 : 3.3 | **20 : 1** (ratio of medians; median-of-ratios 23.2) | C2, verified |
| `T_e / T_c`, UDFS | 64 : 1 | **> 5,000 : 1** (5,762 ratio-of-medians; 6,851 median-of-ratios) | C2, verified |
| Break-even ρ, Bingo | ≈ 1.85 (implicit) | **no such range exists** — ρ spans [1.19, 1.85] on the losers and [1.77, 1.84] on the winners; corr(ρ, `S`) = **+0.04** | C2, verified |
| What *does* predict `S` on Bingo | — | **run length**, `r = +0.41` against log native wall clock | C2, verified |
| Problems with `T_IS < T_BL`, Bingo | 4 / 50 (supplementary only) | **13 / 70, in the main text** | AC-4 ✅ |
| Problems with `T_IS < T_BL`, UDFS | not reported | **29 / 70** (letter) | C2, verified |
| Median canon overhead, Bingo | 39.2 % | **16.1 %** | C2, verified |
| Median canon overhead, UDFS | — | **0.04 %** | C2, verified |
| Abstract's cost statement | overhead only, no wall-clock verdict | **overhead + the 1.45× Bingo slowdown**, one clause; abstract 343 → 297 words | AC-7 ✅ |
| Conclusion's R² claim | two p-values, no effect size | **p with Cohen's `d`**, and the two hosts separated | new, 2026-08-14 |
| `results.tex:149` R² CD claim | "on UDFS the three separate" — **false** | only IsalSR separates; native and naive hash are one clique (0.143 < CD 0.396) | new, 2026-08-14 |
| `T_eval/T_canon` label | "median ratio" | "ratio of median costs" at all four sites | new, 2026-08-14 |
| Complexity in conclusion | "near-linear" | near-$O(k^2)$ | AC-6 ✅ |
| Complexity in discussion (×2) | "near-linear" | near-$O(k^2)$ | AC-6 ✅ |
| Complexity in introduction | near-O(k²) ✓ | unchanged | AC-7 ✅ |
| Complexity in related work | near-quadratic ✓ | unchanged | AC-7 ✅ |
| Relationship of O(k^0.7) to O(k²) | unstated | stated once, `discussion.tex:100–107` | AC-6 ✅ |
| Answer chosen | — | **A**, and structured so it stays true if T02 delivers B | §4 |

Answer A was chosen not because the data favoured it but because the data did not
exist yet. **The campaign has now settled it: Answer A is correct, and by a wider
margin than the submitted numbers implied.** Bingo's `S` fell from 0.93 to 0.72;
the C++ engine's speedup did not offset T16's alphabet cost, exactly as §15
warned. The letter states this outright and calls 0.72 "a worse ratio than the
0.93 the reviewer objected to". That sentence stays.

**One structural claim in §4 also failed and had to be withdrawn from the answer.**
§4 and the reviewer both assumed the winning problems are "the subset where ρ is
large enough". They are not. On Bingo ρ is nearly constant — within 0.09 of 1.79
on 66 of 70 problems — so it cannot discriminate, and corr(ρ, `S`) = +0.04. The
22 problems with `S ≥ 1` span ρ ∈ [1.77, 1.84], *inside* the [1.19, 1.85] the
losers span. R1.1 therefore concedes the reviewer's premise is wrong in our
favour on the mechanism and against us on the outcome, and names what does
predict `S`: run length, `r = +0.41`. This is the honest form of the answer and
it is more useful to a practitioner than the conditional the reviewer proposed.

### 9.2 Changes made to the manuscript

Applied under `article/` (the `changes`-package tree is the single marked-up
source; there is no separate annotated copy). **Not** pushed to Overleaf.

| File | Lines | Change |
|---|---|---|
| `paper/discussion.tex` | 19–34 | growth-rate inference from two median timings withdrawn; medians reattributed to backtracking frequency; growth pointed at `Section~\ref{sec:canonical}` |
| `paper/discussion.tex` | 62–120 | "approximately neutral" removed; net wall-clock cost stated with `S = 0.72`, 13 of 70 and the 1.45× slowdown; sign of `S − 1` given as the joint condition on ρ and `T_eval/T_canon`; the ρ-conditional explicitly withdrawn on the data; "at neutral wall-clock cost" replaced |
| `paper/discussion.tex` | 68 | **2026-08-14** — "the median `T_eval/T_canon`" → "the ratio of median costs" |
| `paper/discussion.tex` | 154–165 | "near-linear" → near-$O(k^2)$; the `O(k^{0.7})` fit qualified as a range description, with why it sits below the bound |
| `paper/conclusion.tex` | 11 | "near-linear" → near-$O(k^2)$ |
| `paper/conclusion.tex` | 23–26 | **2026-08-14** — R² p-values given their Cohen's `d`; the two hosts separated (improves where the budget binds, unchanged where the host saturates) |
| `paper/results.tex` | 149–152 | **2026-08-14** — "on UDFS the three separate" corrected: only IsalSR separates; native and naive hash are one clique |
| `paper/results.tex` | 296–298 | **2026-08-14** — same `T_eval/T_canon` relabel |
| `paper/main.tex` | 90 (abstract) | **2026-08-14** — the 1.45× Bingo slowdown added as one clause; two explanatory relative clauses and four redundant phrases cut; 343 → 297 words |
| `reviews/response_to_reviewers.tex` | preamble, R1.1 | `\pendingnum` macro added; R1.1 answer written, then filled from C2; **2026-08-14** the `T_eval/T_canon` relabel ×2 |
| `double_blind/paper/{discussion,results,conclusion}.tex` | — | re-synced byte-identical after the above |
| `double_blind/paper/main_anonymous.tex` | 79 | abstract delta mirrored |
| `experiments/scripts/review_campaign/values.py` | 406, 513 | provenance note relabelled to "ratio of median…" |
| `experiments/scripts/review_campaign/verify.py` | 458, 468 | two abstract anchors re-pointed at the rewritten sentence (checks unchanged) |

### 9.3 Response text — WRITTEN

In `reviews/response_to_reviewers.tex`, replacing `\todoblock{R1.1}`. Nine
paragraphs of prose, no display objects, no table, seven `\pendingnum{}`
placeholders.

The sketch above was followed except on one point. It planned to "state the
break-even condition as a formula"; the formula is refuted by the code (§8), so
the letter states the condition in terms of the two quantities that are actually
measured, `ρ` and `T_eval/T_canon`, and defers the crossover to a placeholder.

R1.1 runs: concede at `S = 0.93` and remove the phrase; say what `S` actually
measures and show that the submitted sentence described a quantity `S` does not
report; locate the defect as the mismatch between our own results and discussion
sections; give the regime as a joint condition on `ρ` and `T_eval/T_canon`;
disclose that the numbers are being re-measured, why, and that the engine change
and the alphabet correction push the cost in opposite directions; give the count
of problems where IsalSR wins end to end, with the submitted 4 of 50 stated
outright; volunteer the equal-wall-clock-budget limitation; volunteer E3
unprompted, including that R1's own significance statement already used
near-$O(k^2)$; `\changeref` to Section 6 and Section 7.

The answer deliberately contains **no** "no reported number changes" sentence.
Numbers do change here, and claiming otherwise would be the same failure mode the
comment is about.

### 9.4 Residual risk

- **Closed.** The seven placeholders are filled; `grep` finds one `\pendingnum` in
  the letter and it is the macro definition. The two hard-coded submitted values
  survive and are correctly attributed: `0.93` as "the $0.93$ the reviewer
  objected to", `4 of 50` as "In the submitted manuscript that count was four of
  the fifty problems". `discussion.tex` no longer carries any submitted-campaign
  number — the whole Practical Impact subsection was rebuilt on C2.
- **Anticipated and realised**: "the break-even formula not matching the code's
  `S` definition". It did not match. Handled in §8 (2026-08-06).
- **The one a round-2 reviewer can still press.** `S` is measured under equal
  wall-clock budgets, and the letter volunteers this. A reviewer may reply that an
  evaluation-count budget is the fairer accounting for a method whose whole claim
  is that it removes evaluations, and that under it Bingo's number would improve.
  We do not report that number and do not know it. The letter's position — we
  report seconds because seconds are what a practitioner spends, and we make no
  claim under the other accounting — is defensible but it is a refusal, not an
  answer. **If queue time allows before the freeze, the cheapest possible answer
  is an offline replay: `ρ` is already measured, so `S` under an evaluation budget
  is bounded below by `ρ/(1 + ρ·T_c/T_e)` with the measured `T_c/T_e = 1/20`,
  giving ≈ 1.66 on Bingo.** That is a *model*, not a measurement, and §8 is
  explicit that models of `S` must not be published as if measured — so it would
  have to be labelled as an upper-bound argument or run for real.
- **Three edits sit in files this ticket does not own.** `conclusion.tex` (×2: the
  near-$O(k^2)$ complexity claim and the R² effect sizes) and `main.tex` (the
  abstract's cost clause and the 343 → 297 word trim) are **Ezequiel's**, and the
  abstract pass is **T12/Karl's**. All three were made because leaving them would
  have left the letter's own statements untrue or the abstract understating a cost
  by 3×. They need explicit sign-off, not silent inheritance.
- **`analyses/values/critical_difference.json` carries a √2-too-wide `cd_value`**
  (0.5603 where Demšar gives 0.3962). Nothing printed depends on it and every
  clique conclusion survives either threshold, but it is a provenance trap and it
  is not T10's to fix. Filed for the CD-figure owner.
- **`verify.py`'s coverage is a whitelist, not a guarantee.** Eleven R1.1 numbers
  were outside it and one of them was wrong. Any future ticket that adds a number
  to the manuscript must add it to `build_checks`, or the next audit finds it the
  hard way.
