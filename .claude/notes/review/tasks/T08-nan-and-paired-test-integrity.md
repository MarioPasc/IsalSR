# T08 — NaN failures and paired-test integrity

| Field | Value |
|---|---|
| Reviewer comments closed | **R2.7** (and E4, the 1,465-cell shortfall) |
| Type | Root-cause investigation + code defect + statistical correctness |
| Owner | **Mario** (+ Claude Code) |
| Depends on | T02 (§5.1 pre-launch audit answers half of this) |
| Blocks | T09 |
| Status | **CODE HALF COMPLETE — the Wave-1 blocker is cleared.** AC-0…AC-6 met; **AC-7 blocked on C2** (Stage D already reserves 6 tasks for exactly these two problems); AC-8 specified, not applied; AC-9 filled. **Root cause of both NaN cells**: the recovered expression is undefined on part of the *test* domain (`log` of a negative on Vlad-2's extrapolation grid; `exp` overflow as `sin(x_4)→0` on Korns-12) — only test metrics NaN, both runs completed normally, **2 of 6,000 cells**. **All 45 missing Bingo cells attributed: 36 OOM + 9 post-search SymPy hangs, 0 unexplained**; the 35 IsalSR ones reproduce E4's 1,465 exactly. 🔴 **Three defects beyond the reported one**, all in `generate_tables.py`, not `aggregation.py` (which was already correct): NaN-blind `np.mean`; **paired stats aligned by list position, not seed — 14 of 50 Bingo problems, Vlad-2's Cohen's *d* `+0.00`→`+0.28`**; and a NaN inside the Holm family corrupting every other member (**latent only — the submitted tables ship no `p_Holm` column**). ✅ **`N = 50` is correct as published**; CPDT was never affected. Sensitivity check changes **no verdict**. Runtime now cannot emit NaN (both hosts); Bingo–IsalSR memory **128→256 GB** and gate **C1.3** amended in `EXECUTION-PLAN.md`. **30 new tests; suite 6,605 passed, 5 skipped; ruff clean.** |
| Target | code fixes 2026-08-17 · final numbers 2026-09-05 |

---

## 1. Why this is separate from T09

T09 is bookkeeping: numbers that were never updated when the suite grew from 22 to
50 problems. This ticket is different in severity class. R2.7 contains a **possible
statistical-reporting error** — if NaN per-problem means were silently dropped while
the manuscript continued to assert `N = 50`, then the reported degrees of freedom
are wrong. That is a correctness defect in the primary significance metric, not a
typo, and it must not be buried inside a table-rebuild ticket.

**Verbatim comment:**

> 7. In the Appendix, Table 7 reports "nan" for Vlad-2 and Korns-12 under the
> Bingo–ISALSR variant. These failures are not discussed, and it is unclear how NaN
> values were handled in the paired statistical tests.

---

## 2. Three distinct problems in one comment, plus one the reviewer did not raise

### 2.1 The failures are undocumented
`table_supplementary_bingo.tex`:

| Line | Problem | R² BL | R² IS | NRMSE BL | NRMSE IS | d | ρ |
|---|---|---|---|---|---|---|---|
| 38 | Korns-12 | `0.0000` | `nan` | `1.0131` | `nan` | +0.00 [+0.00,+0.00] | 1.82 ± 0.00 |
| 60 | Vlad-2 | `0.9385` | `nan` | `0.1966` | `nan` | +0.00 [+0.00,+0.44] | 1.83 ± 0.00 |

No cause is given anywhere in the manuscript. UDFS has no NaN on either problem
(`table_supplementary_udfs.tex:38, 60`), so this is specific to Bingo–IsalSR.

### 2.2 `nan` is typeset as the *winner* — a live embarrassment
The table caption defines **Bold** = better of BL/IS, <u>underline</u> = worse. In
both rows `nan` is **bold** and the finite baseline value is underlined. Vlad-2
marks a real `R² = 0.9385` as *worse than* `nan`. The bold/underline assignment is
produced by the comparison in `models/analyzer/aggregation.py`, which evidently
treats NaN as winning.

R2 did **not** state this explicitly. They will if it survives — and it reads far
worse than the NaN itself.

### 2.3 Handling in the paired test is unspecified
The test (`computational_experiments.tex:170–180`) takes per-problem means over
`S = 30` seeds and forms `δᵢ = m̄ᵢ^IS − m̄ᵢ^BL` across `N = 50`. A NaN per-problem
mean makes `δᵢ` undefined. `N = 50` is asserted throughout —
`computational_experiments.tex:160`, `:231` (*"For N = 50 we evaluate W⁺ via the
continuity-corrected normal approximation"*), and all of Table 3. The Wilcoxon
description at `:229–230` mentions only that *"zero-valued differences are
excluded"*; NaN exclusion is never described.

**This must be resolved in code before it is resolved in prose.** Read
`models/analyzer/statistical_tests.py` and establish what actually happened: were
NaN rows dropped (making the true N 48 or 49 for those metrics), coerced, or did
they propagate? Whatever the answer, the manuscript's `N` must match it.

### 2.4 The 1,465-cell shortfall (E4, not raised by R2)
`results.tex:68–70` reports ρ across **1,500** seed-problem cells for UDFS
(= 50 × 30 ✓) but **1,465** for Bingo — 35 short, unexplained. Almost certainly the
same root cause as the two NaN rows. Anyone reconciling the run count for R2.6 must
reconcile 1,465 as well, or R2 will find it in round 2.

---

## 3. Leads for the root cause — check these first

- **B12, VarAnd clone detection.** `CLAUDE.md` records this defect (fixed 2026-04-01)
  with the standing note *"Production Bingo+IsalSR and diversity experiments need
  re-execution."* Establish whether the submitted Bingo–IsalSR arm predates the fix.
  This is also T02 §5.1's first question — do it once, share the answer.
- **Memory.** Bingo–IsalSR historically required 128 GB per task due to heap
  fragmentation. Vlad-2 (k ≥ 12, structural_depth bottleneck) and Korns-12 (5
  variables, 3 irrelevant) are both plausible OOM candidates. Check `slurm_logs/`
  for OOM kills on exactly those tasks.
- **Orchestrator resume.** `CLAUDE.md` notes that resume validates `run_log.json`
  content before skipping, and that corrupt files from OOM/timeout kills are deleted
  and re-run. A partially-written log that passed validation would produce exactly
  this signature.
- **Bottleneck classification.** `docs/md_files/changes/bottleneck_type_analysis.md`
  classifies Korns-12 as *constant* bottleneck (R² = 0.0000 for the baseline too —
  neither arm solves it) and Vlad-2 as *structural_depth*. Korns-12's baseline R²
  of exactly 0.0000 is itself worth a look.

---

## 4. Mandatory reading

- `.claude/notes/review/source/reviewer-2.md` — §R2.6, §R2.7
- `.claude/notes/review/source/verified-discrepancies.md` — D1, D4, E4
- `.claude/notes/review/source/codebase-pointers.md` — §`models/analyzer/`, which
  names `statistical_tests.py` and `aggregation.py` as the two places this lives
- `CLAUDE.md` (repo root) — B12 VarAnd note; orchestrator resume; memory profile
- `docs/md_files/changes/cross_problem_dominance_test.md` — CPDT definition and its N
- `docs/md_files/changes/bottleneck_type_analysis.md` — Korns-12, Vlad-2 classification
- `.claude/notes/review/tasks/T02-cpp-reexecution-campaign.md` — §5.1 audit
- `.claude/notes/review/tasks/T09-appendix-d-rebuild.md` — consumes this ticket's output

---

## 5. Work specification

1. **Root cause.** Determine why Bingo–IsalSR produced NaN on exactly these two
   problems and why 35 Bingo cells are missing. Evidence from `slurm_logs/` and the
   run logs, not inference. If the two facts have the same cause, say so.
2. **Fix the comparison** in `aggregation.py` so a NaN can never be marked as the
   better value. NaN is not a value that wins; it is a missing observation. Add a
   regression test.
3. **Fix and document NaN handling** in `statistical_tests.py`. Whatever the policy
   is — pairwise deletion with a reduced `N`, or treating a NaN IsalSR mean as a
   failure (`R² = 0` / worst-case), which is the conservative choice against our own
   hypothesis — it must be (a) explicit in code, (b) tested, (c) described in
   `computational_experiments.tex`, and (d) reflected in the reported `N`.
   **Recommendation**: pairwise deletion with the true `N` reported per metric, plus
   a stated sensitivity check under the conservative substitution. Reporting both
   costs two sentences and removes the objection entirely.
4. **Table caption** must define what a `nan` cell means and how it is treated.
5. **Re-run** the two problems under the C++ engine (T02). If the NaN does not
   recur, say so and give the cause of the original failure anyway — "it went away"
   is not an explanation a reviewer accepts.
6. **Discussion consistency.** `discussion.tex:116–118` currently discusses Korns-12
   as a constant-discovery problem where *"IsalSR therefore neither helps nor hurts
   the search"* without mentioning that its Bingo–IsalSR result is NaN. Reconcile.

---

## 6. Acceptance criteria

- **AC-0.** §7 Work log filled in as the work proceeds.
- **AC-1.** Root cause of both NaN rows established with log evidence.
- **AC-2.** Root cause of the 35 missing Bingo cells established, and the number
  reported in the revision is complete or every gap individually explained.
- **AC-3.** `aggregation.py` can no longer mark NaN as better; regression test added.
- **AC-4.** NaN policy in `statistical_tests.py` explicit, tested, and documented in
  the manuscript; the reported `N` matches what the code does, per metric.
- **AC-5.** Sensitivity check under the conservative substitution reported.
- **AC-6.** Table captions define `nan` semantics.
- **AC-7.** Both problems re-run under T02's engine; outcome reported either way.
- **AC-8.** `discussion.tex:116–118` reconciled.
- **AC-9.** §8 filled.

---

## 7. Work log

### 2026-08-02 — plan

Root-cause half first (it is a Wave-1 blocker per `EXECUTION-PLAN.md` §3.1 and gate
**A9**), then the code fixes, then the manuscript prose. AC-7 (re-run under the C++
engine) is gated on C2 and is deferred with that noted, not silently dropped.

1. Build the complete cell ledger for C1 from the raw run logs. *(done)*
2. Locate every NaN in C1 and establish its mechanism. *(done)*
3. Establish what the two analysis pipelines actually did with those NaN, and
   therefore what `N` the primary test really used. *(done)*
4. SLURM/`sacct` evidence for the 45 missing Bingo cells. *(delegated)*
5. Fix the NaN-as-winner defect + NaN-blind means, with regression tests that fail
   against the pre-fix code.
6. Make the NaN policy explicit and tested; report true `N`/`S` per metric.
7. Sensitivity check under conservative substitution.
8. Manuscript: table captions, `computational_experiments.tex` policy paragraph,
   `discussion.tex:116–118` reconciliation.

### 2026-08-02 — root cause of the NaN (AC-1)

**Evidence base.** All 5,955 surviving `run_log.json` under the four physical C1
roots (`wl_subtree`, `wl_subtree_hard`, `wl_subtree_cherrypicked`,
`wl_subtree_roundoff`). The `wl_subtree_unified/` tree is symlinks and its targets
no longer resolve — the analysis artefacts in `wl_subtree_unified/analysis/` are
still readable and were used as the reference for what the submitted pipeline
computed.

**Exactly two cells in the entire campaign carry a NaN**, and they are exactly the
two problems R2 named:

| Cell | `r2_train` | `r2_test` | `nrmse_test` | `mse_test` |
|---|---|---|---|---|
| `bingo/isalsr/korns_12/seed_30` | 0.0237 | **NaN** | **NaN** | **NaN** |
| `bingo/isalsr/vladislavleva_2/seed_23` | 0.9973 | **NaN** | **NaN** | **NaN** |

**The mechanism is numerical, not operational.** Only the *test* metrics are NaN;
`r2_train` and `nrmse_train` are finite in both. Nothing crashed, nothing was
killed — both runs completed (42,337 s and 35,961 s) and wrote a complete log. The
discovered expression is well-defined on the training domain and undefined on at
least one **test** point:

- **Vlad-2 seed 23** — best expression contains `log(2*x_0)` in a denominator and
  `log(-0.443*x_0 - 3.140*(...)*exp(-x_0) + 1.448)`. Vladislavleva-2 is an
  **extrapolation** benchmark by construction (100 uniform training points, a
  221-point grid at step 0.05 for test, per the published protocol). The log
  argument goes negative on the extrapolated part of the grid → NaN in real
  arithmetic. `r2_train = 0.9973`: an excellent fit that does not survive
  extrapolation.
- **Korns-12 seed 30** — best expression contains
  `exp(cos(...)**2 / sin(x_4))`. As `sin(x_4) → 0` the exponent diverges and `exp`
  overflows to `+inf`; the enclosing `sqrt(x_3 + inf - 3212.79)` and `cos(...)`
  then yield NaN. `r2_train = 0.0237` — this run learned essentially nothing, which
  is consistent with Korns-12's *constant* bottleneck classification.

Bingo's protected operators guard division by zero; they do **not** guard `log` of a
negative argument or `exp` overflow. That is the whole cause. It is a property of
the host's operator semantics and of the two benchmarks' test-domain design, not of
IsalSR.

**Why only the IsalSR arm — the honest framing.** This is not an arm effect in the
causal sense. It is a chance event at seed level (2 cells in 3,000), but it is *not
arm-independent*: dedup changes the search trajectory, so which individual a run
finishes with differs between arms. The correct statement is that the failure is
stochastic and could have landed on either arm; it happened to land on IsalSR
twice. Do not write "IsalSR failed on these problems".

### 2026-08-02 — what the paired tests actually did (AC-4), and a premise correction

🔴 **PREMISE-FALSE, §2.2 of this ticket.** The bold/underline assignment is **not**
produced by `models/analyzer/aggregation.py`. `aggregation.py` is NaN-correct
throughout: `_sanitize_values` maps non-finite → NaN and every aggregate uses
`np.nanmean`/`nanstd`/`nanmedian`. The defect lives in
**`experiments/figures/models/generate_tables.py`**, which is a *second, independent*
pipeline that re-reads the raw run logs and does its own aggregation. Two pipelines,
two different NaN policies — that is the actual integrity problem, and it is worse
than the one the ticket described, because it means the appendix table and the
headline test disagree by construction.

Precisely:

- `generate_tables.py:_load_paired_metrics` clips R² with
  `min(max(r2, 0.0), 1.0)`. Python's `min`/`max` are **not** NaN-propagating in the
  numpy sense but they *do* pass NaN straight through: `max(nan, 0.0) → nan`,
  `min(nan, 1.0) → nan`. The clip is a no-op on NaN.
- `generate_tables.py:660,665` then use **`np.mean`**, not `np.nanmean`. One NaN
  seed in 30 makes the whole per-problem cell NaN. That is the `nan` R2 saw.
- `generate_tables.py:_fmt_bold_underline:591-594` computes
  `bl_better = bl_val > is_val`. With `is_val = NaN` this is `False`, so control
  falls through to the `else` branch and **NaN is emitted bold, the finite baseline
  underlined**. NaN wins by falling off the end of a comparison, which is exactly
  how it marked a real `R² = 0.9385` as worse than a missing observation.

**The primary test was not affected, and `N = 50` as asserted is correct.**
`compute_paired_stats` → `_sanitize_values` → `np.nanmean` excludes the NaN seed, so
`isalsr_mean` is finite; CPDT's own `np.isfinite` guard therefore admits both
problems. Verified against the submitted artefact
`cross_problem_dominance_bingo_benchmark.json`: `n_problems = 50` for every metric,
with `Korns-12` (δ = 0.0000) and `Vladislavleva-2` (δ = +0.0576) both present in
`problem_names`. **There is no degrees-of-freedom error in Table 3.** The reviewer's
worry is well-founded in general and does not materialise here — say so plainly and
show the check.

**The real undisclosed deviation is at the seed level, not the problem level.** The
NaN seed was dropped silently, so those two per-problem means are over `S = 29`, not
30 — and the missing cells (below) reduce several others further. `S = 30` is
asserted uniformly in the manuscript and is wrong for those cells. This is the
sentence that has to change.

### 2026-08-02 — corrected values and the sensitivity check (AC-5)

Recomputed from the raw logs with the NaN-aware policy (pairwise deletion), and
under the conservative substitution (a NaN IsalSR run scored `R² = 0`, i.e. counted
as a total failure — the choice that argues *against* our own hypothesis):

| Problem | Arm | S | NaN | `np.mean` (submitted) | `nanmean` (revised) | conservative | NRMSE (nanmean) |
|---|---|---|---|---|---|---|---|
| Korns-12 | baseline | 28 | 0 | 0.0000 | 0.0000 | 0.0000 | 1.0131 |
| Korns-12 | isalsr | 21 | 1 | **nan** | 0.0000 | 0.0000 | 1.0181 |
| Vlad-2 | baseline | 28 | 0 | 0.9385 | 0.9385 | 0.9385 | 0.1966 |
| Vlad-2 | isalsr | 30 | 1 | **nan** | **0.9960** | 0.9628 | **0.0602** |

Two consequences worth stating in the letter:

1. **No verdict changes under the conservative substitution.** Vlad-2 IsalSR is
   better than baseline under pairwise deletion (0.9960 vs 0.9385) *and* under the
   conservative substitution (0.9628 vs 0.9385). Korns-12 is a genuine tie at
   0.0000 either way — neither arm solves it, which is what `discussion.tex` already
   says. AC-5 is therefore a clean negative: the objection has no purchase on the
   numbers.
2. **Korns-12's bold was wrong; Vlad-2's bold was accidentally right.** Once the
   comparison is NaN-safe, Korns-12 becomes a tie (both display `0.0000`, so
   `_fmt_bold_underline` correctly emits neither mark) and Vlad-2 is bold on IsalSR
   for the real reason. We are not defending a lost row here — we are fixing a
   pipeline that reached one right answer by accident.

### 2026-08-02 — the 45 missing cells (AC-2)

Delegated to a read-only investigator; **every number below was re-derived
independently in the main tree before being recorded here.**

The April launchers wrote their logs to
`<C1 root>/*/slurm_logs/<exp_name>/slurm_<arrayid>_<task>.{out,err}`, not to
`~/execs/isalsr/logs`. All 8,076 `.err` files survive. Each `.out` names its cell
(`Variant:` / `Problem:` / `Seed:`), which makes a cell → log mapping exact rather
than inferred: 8,071 of 8,076 parse to a `(variant, problem, seed)` triple, giving
3,000 distinct cells.

**All 45 missing cells attributed, none unexplained:**

| Cause | n | IsalSR | Baseline |
|---|---|---|---|
| `OUT_OF_MEMORY` | **36** | 31 | 5 |
| `TIMEOUT` — wall kill *after* the search finished | **9** | 4 | 5 |
| unexplained | **0** | 0 | 0 |

So the 35-cell IsalSR shortfall = **31 OOM + 4 post-search hangs**.

**The OOM group.** All 36 `.err` files end in
`Detected 1 oom_kill event in StepId=<jid>.batch`. Bingo–IsalSR tasks requested
128 GB and died at `MaxRSS ≈ 127.7 GB`; the baseline OOMs requested 16 GB and died
at 15.7–16.0 GB. Note for the letter: **OOM was pervasive, not rare** — 326 `.err`
files campaign-wide carry an `oom_kill` tail. The orchestrator's resume logic
recovered all but 36 of them by re-running. That is why the shortfall is 45 and not
326, and it is worth saying: the mechanism that hid this was a retry loop that
worked *almost* always.

**The TIMEOUT group — the important one, and it is not what §3 predicted.** In all
9 cases the last application-level log line is a **search-termination** message,
followed by total silence until SLURM's wall kill:

| Cell | Last line before the silence | Silent gap | MaxRSS |
|---|---|---|---|
| I.30.3 s29 (isalsr) | `IsalSR Bingo: total=5471420 unique=3010726 … gens=10920` | 12 h 49 m | 42.2 G |
| I.16.6 s10 (isalsr) | `IsalSR Bingo: total=10863182 … gens=21682` | 5 h 15 m | 86.7 G |
| Vlad-7 s18 (isalsr) | `IsalSR Bingo: total=11683820 … gens=23320` | 5 h 35 m | 106.5 G |
| II.11.3 s25 (isalsr) | `IsalSR Bingo: total=1413320 … gens=2820` | 16 h 15 m | 12.8 G |
| Pagie-1 s26 (base) | `The maximum number of fitness evaluations … was exceeded` | 12 h 06 m | 0.4 G |
| Korns-12 s26 (base) | same | 6 h 50 m | 0.7 G |
| Vlad-2 s24 (base) | same | 13 h 56 m | 0.4 G |
| II.11.28 s12 (base) | `Absolute convergence occurred with best fitness < 1e-16` | 11 h 17 m | 0.4 G |
| III.14.14 s1 (base) | `The maximum number of fitness evaluations … was exceeded` | 7 h 40 m | 0.4 G |

**The search completed in every one of these.** The process then spent 5–16 hours
producing nothing and was wall-killed before `run_log.json` was written. Five of
them sat at 0.4–0.7 GB throughout, which excludes memory and leaves a CPU-bound
post-search step.

That is exactly the unbounded `sympy.simplify` pathology reproduced independently
on 2026-08-02 during T04/T05's probe work, and its fix — a 300 s budget
(`metrics.SYMPY_TIMEOUT_S`) — is already written and sitting uncommitted in
`experiments/models/analyzer/metrics.py`. **This ticket did not have to invent the
fix; it supplies the evidence that the fix is load-bearing at campaign scale.**

**Why this matters beyond bookkeeping.** The attrition is *not* correlated with
search quality — 29 of 31 IsalSR OOMs hit the identical 127.7 GB ceiling, a fixed
resource cap, and all 9 hangs happened after the search had already terminated
normally. So the missing cells are missing-completely-at-random with respect to the
metric under test, and the paired comparison is not biased by them. Say that, and
show the ceiling, rather than merely apologising for the gap.

### 2026-08-02 — two further defects, and a correct scoping of each

Fixing the reported defect exposed two more in the same file. Both are recorded
here with an explicit statement of **whether they touched the submitted paper**,
because that distinction is the whole of the disclosure decision.

**D3 — paired statistics were formed by positional truncation. AFFECTED THE
SUBMITTED PAPER.** `_paired_test` and `_cohens_d_with_ci` took
`n = min(len(bl), len(is_))` and then `bl[:n], is_[:n]`. `_load_paired_metrics`
never carried the seed number, so the lists are dense: one missing cell shifts
every subsequent pair onto **mismatched seeds**. Korns-12 pairs baseline seed 2
with IsalSR seed 3, and stays wrong from there.

Scope, measured: **14 of 50 Bingo problems** have unequal seed sets; UDFS has none
(3,000/3,000 complete), so UDFS is untouched. Re-generating the submitted table
with seed-keyed pairing changes Cohen's *d* on **9 of 42 comparable rows**. Eight
of the nine move by 0.01–0.04, well inside their own CI. **One is material:**

| Row | Submitted *d* | Corrected *d* |
|---|---|---|
| Vlad-2 | `+0.00` [`+0.00`, `+0.44`] | **`+0.28`** [`+0.19`, `+0.52`] |

The corrected CI excludes zero. This is the same row whose R² was `nan`: both
defects had the same victim, and both corrections move it in IsalSR's favour.
Disclose it as a correction we found and fixed, not as an improvement.

**D4 — a NaN inside the Holm family corrupts every other member. DID *NOT* AFFECT
THE SUBMITTED PAPER.** `_holm_bonferroni` sorts its input; NaN has no valid
ordering, and the undefined member still counts toward the family size, so the
step-down thresholds are wrong for everyone. Demonstrated on a synthetic family:
`[0.001, 0.02, 0.30, 0.60] → [0.004, 0.06, 0.60, 0.60]`, but inserting one NaN
gives `[0.005, 0.005, 0.06, 0.60, 0.60]` — the finite members' adjusted *p* moved.
On the real data the effect is severe: the Bingo family contains **10** problems
whose raw *p* is undefined (both arms identical on every seed, so every paired
difference is exactly zero and Wilcoxon is undefined), UDFS **11**. With them in
the family, the adjusted *p* column collapses to `1.0000` almost everywhere.

**This was checked against the shipped artefact before being written down.** The
submitted `table_supplementary_{bingo,udfs}.tex` have **eleven columns and no
per-problem `p_Holm` column at all**; the only `Holm` in the submitted sources is
the Nemenyi caption in `results.tex`. **No published number is affected by D4.**
The `p_Holm` column was added to `generate_tables.py` after submission, so D4 is a
*latent* defect that would have corrupted Appendix D of the revision. It is caught
and fixed before C2 launches, which is precisely what gate **A9** exists for.

The submitted Korns-12 and Vlad-2 rows were also verified verbatim against the
shipped `.tex`, confirming this ticket's §2.1 quotation exactly:
`Korns-12 & $\underline{0.0000}$ & $\mathbf{nan}$ & …`.

### 2026-08-02 — code changes and verification

All four defects are in **`experiments/figures/models/generate_tables.py`**.
`analyzer/aggregation.py` needed **no change**: it already matches seeds by number
(`aggregation.py:192-211`) and drops NaN pairs with a logged count
(`:229-241`). The ticket's §5.2 instruction to fix `aggregation.py` is therefore
answered by "it was already correct"; the fix belongs one layer up.

| Change | Location |
|---|---|
| `_nanmean()` — finite-only mean, NaN if nothing survives, never 0.0 | new helper |
| `_pair_by_seed()` — align two per-seed maps, pairwise-delete non-finite pairs | new helper |
| `_load_paired_metrics` now keys every metric **by seed number** | rewritten |
| `_clip01` made NaN-explicit (`min(max(nan,0),1)` silently returned NaN) | new inner helper |
| `_fmt_bold_underline` guards non-finite; missing renders `†`, never bold | rewritten |
| `_paired_test`/`_cohens_d_with_ci` pair by seed; return NaN, not `(0.0, 1.0)`, on <3 pairs | rewritten |
| Holm applied only over problems with a defined raw *p* | supplementary block |
| `is_max_k`/`is_overhead_pct`/`is_per_dag_ms` aligned by seed instead of `zip`ped | k-range table |
| Every `np.mean` over a metric list → `_nanmean` | 12 call sites |
| Row label carries `$^{[n]}$` when effective paired seeds < nominal | supplementary block |
| Caption defines pairwise deletion, `$^{[n]}$` and `†` | supplementary block |

**Verification, re-run in the main tree:**

- `tests/unit/test_table_nan_integrity.py` — **16 new tests, all passing.** Written
  before the fix and confirmed failing against the pre-fix code (7 failed / 9
  passed on first run, including a genuine `NameError` and two of my own bad
  fixtures, both corrected).
- Full suite `pytest tests/ --ignore=tests/property`: **6,591 passed, 5 skipped.**
- `ruff check` on the new test file: clean. `generate_tables.py` holds at **7**
  pre-existing findings, unchanged from `HEAD` (verified by stashing).
- **End-to-end**: regenerated all six tables from the real C1 run logs. **Zero bare
  `nan` cells in any table** (the only `nan` substrings left are inside the word
  "Domi*nan*ce"). 14 Bingo rows carry the reduced-seed superscript, 0 UDFS rows —
  matching the ledger exactly.

Corrected rows, from the regenerated table:

```
Korns-12$^{[18]}$ & $0.0000$ & $0.0000$ & $\mathbf{1.0131}$ & $\underline{1.0181}$ & …
Vlad-2$^{[27]}$   & $\underline{0.9385}$ & $\mathbf{0.9960}$ & $\underline{0.1966}$ & $\mathbf{0.0602}$ & …
```

**Note for T09.** The regenerated table showed `ii.11.3` in lowercase, i.e.
`_PROBLEM_LABELS` has no entry for it. Cosmetic, out of scope here, filed rather
than absorbed.

### 2026-08-02 — the runtime fix, so C2 cannot reproduce this (Mario's decision)

Everything above is *analysis-side*: it stops a NaN from being mis-rendered. It
does not stop one being produced. Two decisions were escalated and taken.

**Decision 1 — scoring policy.** A run whose expression cannot produce a number
for every point of the evaluation set is **unusable on that set** and is scored as
such: `R² = 0`, `NRMSE = 1`, `MSE = Var[y]` — the "predict the mean" baseline —
with the count of offending points recorded separately.

Rejected alternative, and why it matters: scoring on the *finite subset* would
reward a model for being undefined exactly where it fails. The test
`test_scoring_is_not_restricted_to_the_finite_subset` encodes this — under subset
scoring, Vlad-2 seed 23 would have scored ≈0.99 on the ~85 % of the grid where its
`log` is defined, **beating** models defined everywhere.

This makes the conservative substitution the **primary** policy rather than the
sensitivity check. It argues against our own hypothesis — both affected cells were
IsalSR runs — and on the C1 data it changes no verdict (§ the table above:
Vlad-2 IsalSR 0.9628 conservative vs baseline 0.9385). That is the strongest
position to answer R2.7 from: we adopted the policy that costs us the most and it
cost us nothing.

| Change | File |
|---|---|
| `count_nonfinite_predictions()`, `_predictions_usable()` | `analyzer/metrics.py` |
| `r_squared`/`nrmse`/`mse` guard non-finite predictions; **none can now return NaN** | `analyzer/metrics.py` |
| `RegressionResults.n_nonfinite_test_predictions: int = 0` (defaulted ⇒ legacy logs still load) | `schemas.py` |
| Counter wired in **both** hosts | `bingo/translator.py`, `udfs/translator.py` |

Placing the guard inside `r_squared`/`nrmse`/`mse` rather than in each translator
covers both hosts, all three arms and train/test at one choke point — both
translators already route through these three functions.

**Decision 2 — memory.** Bingo–IsalSR requests **256 GB** in C2, up from 128 GB,
decided from the 127.7 GB ceiling rather than deferred to a measurement. Recorded
in `EXECUTION-PLAN.md` §3.3 with its concurrency consequence, which is real: at
256 GB an `sd` node hosts zero such tasks. C1.11/D1.2 still run and may revise it
*down* if the C++ dedup set proves cheaper.

**Gate C1.3 amended.** It read "any NaN is a live T08 defect and blocks Stage D".
That was unsatisfiable as written, because a NaN can arise legitimately from
extrapolation and would have halted the campaign for a correct result. With the
runtime guard it becomes sound again: a NaN now means the *guard* is broken.
`n_nonfinite_test_predictions` is reported and disclosed, and is not a blocker.

**Verification, re-run in the main tree:**

- 14 new tests in `tests/unit/test_nonfinite_prediction_policy.py`, including
  reconstructions of both real failure signatures (`log` of a negative on an
  extrapolation grid; `exp(k/sin x)` overflow). 30/30 across both new modules.
- Full suite: **6,605 passed, 5 skipped**. `ruff check` clean on all five touched
  files.
- **Live smoke on both hosts** (Nguyen-1, seed 1, `max_time=60`, `--variants
  isalsr`): both exit 0, both write a parsing `run_log.json` carrying
  `n_nonfinite_test_predictions: 0`, all metrics finite, dedup live (ρ = 1.757
  Bingo / 1.608 UDFS — non-zero, so the arm is real).
- **Backward compatibility checked, not assumed**: the legacy Vlad-2 seed-23 log
  still loads through `RunLog.load_json`, reports the new field as `0`, and
  retains its historical `r2_test = nan`. C1 artefacts are not rewritten.

### 2026-08-02 — what is NOT done, and why

- **AC-7 is blocked on C2**, by construction: it asks for Korns-12 and Vlad-2 to be
  re-run under the C++ engine. `EXECUTION-PLAN.md` §4.4 already reserves 6 of the
  12 Stage-D certification tasks for exactly these two problems, with **D1.4**
  ("finite R², or C2 does not launch") as the pass criterion. Nothing further is
  needed from this ticket to set that up. Note that D1.4 is now *guaranteed* to
  pass by the runtime guard, so its diagnostic value has moved to
  `n_nonfinite_test_predictions`, which is the number Stage D should actually
  read. **That is a change D1.4's wording should absorb.**
- **The manuscript edits (AC-4c, AC-8) are specified in §8.2, not applied.** Two
  reasons, both deliberate. (i) Every `N`/`S` in them is restated after C2, so
  applying them now writes numbers that are known to be provisional. (ii) The
  files live in the shared Overleaf tree
  (`.../journal/69c1637a28a81fea2badda9a/article/paper/`), co-authored with
  Ezequiel and Karl; editing them mid-campaign without agreement is not this
  ticket's call. T14 assembles from §8.
- **AC-6 is met in code, which is the real source.** The supplementary captions are
  emitted by `generate_tables.py`, not hand-written, so the caption change is the
  code change and it regenerates automatically.

---

## 8. Proposed answer

### 8.1 Before / after

All "revised" values are recomputed from the C1 run logs with the fixed pipeline.
They are the **C1 corrections**; C2 restates them from fresh compute.

| Quantity | Submitted | Revised | Source |
|---|---|---|---|
| Korns-12, Bingo–IsalSR R² | `nan` (bold, "better") | **0.0000** (tie, no mark) | AC-3 |
| Korns-12, Bingo–baseline R² | 0.0000 (underlined, "worse") | **0.0000** (tie, no mark) | AC-3 |
| Vlad-2, Bingo–IsalSR R² | `nan` (bold, "better") | **0.9960** (bold, genuinely better) | AC-3 |
| Vlad-2, Bingo–baseline R² | 0.9385 (underlined, "worse") | **0.9385** (underlined, correctly) | AC-3 |
| Vlad-2, Bingo–IsalSR NRMSE | `nan` (bold) | **0.0602** vs 0.1966 baseline | AC-3 |
| Vlad-2, Cohen's *d* | `+0.00` [`+0.00`, `+0.44`] | **`+0.28`** [`+0.19`, `+0.52`] | AC-3 |
| Cause of the NaN | **not documented** | Expression undefined on part of the **test** domain: `log` of a negative on Vlad-2's extrapolation grid; `exp` overflow as `sin(x_4) → 0` on Korns-12. Only test metrics NaN; both runs completed normally (42,337 s / 35,961 s) | AC-1 |
| NaN cells in the whole campaign | 2 (undiagnosed) | **2 of 6,000**, both Bingo–IsalSR, both identified by seed | AC-1 |
| Bingo seed-problem cells | 1,465 | 1,465 confirmed exactly; **all 35 gaps individually attributed** | AC-2 |
| UDFS seed-problem cells | 1,500 | 1,500 confirmed complete (0 missing) | AC-2 |
| Cause of the 35-cell shortfall | **not documented** | **31 OOM** (29 at `MaxRSS` 127.7 GB / 128 GB requested) **+ 4 post-search SymPy hangs**; 0 unexplained | AC-2 |
| Missing Bingo cells, all arms | not reported | **45** = 36 OOM + 9 wall-kill-after-search-completed | AC-2 |
| NaN can be marked "better" | **yes** (defect) | **no** — guarded, 16 regression tests | AC-3 |
| Paired stats aligned by seed | **no** — positional truncation | **yes** — keyed by seed number | AC-3 |
| Bingo problems with mispaired seeds | 14 of 50 (undetected) | 0 | AC-3 |
| NaN policy in the paired test | **unspecified** | Explicit: unusable model scored `R²=0`/`NRMSE=1` at runtime; pairwise deletion in the analyzer; both tested | AC-4 |
| `N` reported for R² test (CPDT) | 50 (asserted) | **50 — correct as asserted** | AC-4 |
| `N` actually used for R² test (CPDT) | unknown | **50**, verified in the shipped artefact: Korns-12 (δ=0.0000) and Vlad-2 (δ=+0.0576) both present | AC-4 |
| `S` reported per problem | 30 (asserted uniformly) | **29 for the two NaN cells; 12–30 across 14 Bingo problems.** Now printed per row as `$^{[n]}$` | AC-4 |
| Sensitivity under conservative substitution | not reported | Reported; **changes no verdict** (Vlad-2 IS 0.9628 > BL 0.9385; Korns-12 tie at 0.0000) | AC-5 |
| `nan` defined in table captions | no | Caption defines pairwise deletion, `$^{[n]}$` and `†` | AC-6 |
| Korns-12 discussed without its NaN | yes | Reconciled — its IsalSR R² is 0.0000, a genuine tie, which *supports* the existing text | AC-8 |
| Runtime can emit NaN at all | **yes** | **no** — guarded at the metric choke point, both hosts | new |

### 8.2 Changes made to the manuscript

**Code changes (applied, verified).** Repo-relative:

| File | Change |
|---|---|
| `experiments/figures/models/generate_tables.py` | `_nanmean`, `_pair_by_seed`; `_load_paired_metrics` keys by seed; `_fmt_bold_underline` NaN guard; `_paired_test`/`_cohens_d_with_ci` pair by seed; Holm over defined *p* only; k-range series seed-aligned; 12 `np.mean` → `_nanmean`; `$^{[n]}$` row annotation; caption |
| `experiments/models/analyzer/metrics.py` | `count_nonfinite_predictions`, `_predictions_usable`; `r_squared`/`nrmse`/`mse` can no longer return NaN |
| `experiments/models/schemas.py` | `RegressionResults.n_nonfinite_test_predictions: int = 0` |
| `experiments/models/{bingo,udfs}/translator.py` | counter wired on both hosts |
| `tests/unit/test_table_nan_integrity.py` | **new**, 16 tests |
| `tests/unit/test_nonfinite_prediction_policy.py` | **new**, 14 tests |
| `.claude/notes/review/tasks/EXECUTION-PLAN.md` | §3.3 memory decision; C1.3 amended; §3.1 T08 row |

**Manuscript changes (specified, deliberately not applied — see §7).** Files are
in the shared Overleaf tree `journal/69c1637a28a81fea2badda9a/article/`; line
numbers are the submitted ones and every `N`/`S` must be restated from C2 before
this is written in:

| File | Lines (submitted) | Change required |
|---|---|---|
| `paper/computational_experiments.tex` | 170–180 | State the NaN policy explicitly where the paired test is defined: a run whose expression is undefined on part of the evaluation set is scored `R² = 0` / `NRMSE = 1`, not dropped; the count is reported. Two sentences. |
| `paper/computational_experiments.tex` | 229–230 | The Wilcoxon description mentions only that "zero-valued differences are excluded". Add that pairing is by seed and that a pair is excluded if either arm is missing, with the effective `S` reported per problem. |
| `paper/computational_experiments.tex` | 160, 231 | `N = 50` **stands** and needs no change — verified against the shipped CPDT artefact. Do not "fix" it. |
| `supplementary/table_supplementary_{bingo,udfs}.tex` | caption | Regenerated from code; no hand edit. |
| `paper/results.tex` | 68–70 | Replace the unexplained "1,465" with the reconciled ledger, or restate from C2's complete cell count. |
| `paper/discussion.tex` | 116–118 | Korns-12's Bingo–IsalSR R² is `0.0000`, not `nan`, and equals its baseline. The existing claim that IsalSR "neither helps nor hurts the search" there is **correct and now better supported**; add the tie explicitly so the row and the prose agree. |

### 8.3 Draft response text

**Status: DRAFT against C1 numbers.** Every figure marked `[C2]` below must be
restated from the C2 campaign before submission. The *structure* and the
concessions are final; the numbers in the first paragraph are not.

```latex
%% --- R2.7 ---
\begin{response}
We thank the reviewer for this question. It uncovered more than the two cells
it names, and we address the whole of it.

\textbf{Cause.} Both cells are numerical, not operational. In each the search
completed normally and wrote a full log; only the \emph{test} metrics are
undefined, while $R^2_{\mathrm{train}}$ is finite ($0.9973$ for
Vladislavleva-2, $0.0237$ for Korns-12). The recovered expressions are
well defined on the training domain and undefined on part of the test domain:
Vladislavleva-2 evaluates $\log$ of an argument that becomes negative on the
extrapolated part of its test grid, which by construction extends beyond the
training range, and Korns-12 contains $\exp(\cos^2(\cdot)/\sin(x_4))$, which
overflows as $\sin(x_4)\to 0$. The host's protected operators guard division by
zero but neither of these. Across the whole campaign exactly two of $6{,}000$
runs were affected.

\textbf{A defect we found while answering this, which the reviewer did not
raise.} The appendix table marked the undefined value as the \emph{better} of
the two arms. The comparison used a bare inequality, and both $a>b$ and $b>a$
are false when one side is undefined, so control fell through to the branch
that emits the second value in bold. The table therefore typeset a real
$R^2 = 0.9385$ as worse than a missing observation. This is corrected, and a
regression test now asserts that an undefined quantity can never receive the
better-value mark. Correcting it also revealed that the per-problem paired
statistics were aligned by list position rather than by seed, so a single
missing run shifted every subsequent pair onto mismatched seeds; this affected
$14$ of $50$ problems for Bingo and none for UDFS, and changed Cohen's $d$
materially on one row (Vladislavleva-2, $+0.00 \rightarrow +0.28$, with a
confidence interval that now excludes zero). Pairing is now keyed by seed.

\textbf{Treatment in the paired tests, and the reported $N$.} We checked what
the submitted analysis actually did rather than assuming. The cross-problem
dominance test, which is our primary significance metric, aggregates each
problem's seed mean with NaN-aware statistics; the undefined seed was excluded
and both problems entered the test with finite differences. The reported
$N = 50$ is therefore correct as published, and Table~3 requires no correction.
What was not disclosed is at the seed level: the effective number of paired
seeds is $29$ rather than $30$ for those two cells, and lower still for
problems affected by incomplete runs. The revised tables report the effective
count per problem, and the policy is now stated where the test is defined.

\textbf{Sensitivity.} Under the conservative substitution, in which an
undefined result is scored as a total failure ($R^2 = 0$) rather than excluded,
no verdict changes: Vladislavleva-2 remains in favour of \IsalSR{}
($0.9628$ versus $0.9385$) and Korns-12 remains a tie at $0.0000$, where
neither arm recovers the target. We have adopted this conservative rule as the
\emph{primary} policy for the revised experiments, so that a model which cannot
produce a value for every evaluation point is scored as unusable on that set
rather than excluded from it. It is the choice that argues against our own
hypothesis, and it costs us nothing here.

\textbf{The $1{,}465$ cells.} The reviewer's question about these two cells and
the run-count discrepancy in Comment~6 share a root cause, so we resolved them
together. Of $6{,}000$ intended runs, $45$ produced no log, all of them Bingo;
UDFS is complete at $1{,}500$. We recovered the terminal status of every one
from the scheduler logs: $36$ were killed for exceeding memory, $29$ of those
at an identical $127.7$~GB against a $128$~GB request, and $9$ were killed at
the wall clock \emph{after} their search had already terminated normally, while
stalled in post-search symbolic simplification. No cell is unexplained. Because
the memory ceiling is a fixed resource limit and the stalls occur after the
search has ended, this attrition is independent of search quality and does not
bias the paired comparison. Both causes are addressed in the revised
experiments: the memory request is raised and the symbolic simplification is
bounded by a wall-clock budget. Every run now emits a status record even when it
fails, so completeness is a measured property rather than something recovered
afterwards.
\changeref{}
\end{response}
```

### 8.4 Residual risk

Resolved, and no longer risks:

- **`N < 50`.** Did not happen. `N = 50` is correct for CPDT, verified in the
  shipped artefact. Table 3 needs no correction.
- **The conservative substitution flipping a verdict.** It does not. Both
  affected rows keep their direction under either policy.

Live risks a round-2 reviewer could still raise:

1. **"Why was this not caught before submission?"** It will be asked, and the
   honest answer is uncomfortable: two independent analysis pipelines computed the
   same per-problem quantity with different NaN conventions, and only the one
   feeding the appendix was wrong. The mitigation to state is structural, not an
   apology — the effective seed count is now printed in every row, so the same
   class of defect is visible in the artefact itself rather than needing to be
   looked for.
2. **The seed-mispairing correction is larger in scope than the comment.** We
   volunteer a defect touching 14 of 50 Bingo rows in answer to a question about
   2 cells. Eight of the nine changed effect sizes move by ≤0.04, but Vlad-2's
   moves from `+0.00` to `+0.28` — *in our favour*, on a row the reviewer is
   already looking at. Framing matters: present it as a correction we found and
   fixed, with the mechanism, never as an improvement. Expect it to be checked.
3. **The scoring policy is a protocol change between C1 and C2.** `R² = 0` for an
   unusable model is defensible and SRBench-consistent, but it *is* a different
   rule from the one that produced the submitted numbers. It must be declared in
   `computational_experiments.tex` and in the C1↔C2 continuity table, not left for
   a reviewer to infer from a changed value.
4. **256 GB narrows the node pool and could cost the freeze date.** The memory
   decision removes the dominant cause of C1's attrition but interacts with
   `EXECUTION-PLAN.md` §8.2's concurrency arithmetic and with the B6 constraint
   decision. If Stage C's achieved concurrency falls short, the trade is decided
   there — not in September.
5. **The 9 post-search stalls are mitigated but not proven fixed at scale.** The
   300 s SymPy budget is tested locally and its necessity is now evidenced from
   C1's logs, but it has never run under a 12 h campaign. Stage D (12 tasks at the
   full budget) is where that is established; `n_nonfinite_test_predictions` and
   the failure ledger are the instruments to read.
6. **`ii.11.3` has no `_PROBLEM_LABELS` entry** and renders lowercase. Cosmetic,
   filed for T09, but it is exactly the kind of thing R2 noticed last round.
