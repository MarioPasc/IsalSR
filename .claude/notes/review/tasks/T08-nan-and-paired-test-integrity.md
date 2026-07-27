# T08 — NaN failures and paired-test integrity

| Field | Value |
|---|---|
| Reviewer comments closed | **R2.7** (and E4, the 1,465-cell shortfall) |
| Type | Root-cause investigation + code defect + statistical correctness |
| Owner | **Mario** (+ Claude Code) |
| Depends on | T02 (§5.1 pre-launch audit answers half of this) |
| Blocks | T09 |
| Status | NOT STARTED |
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

_(empty — to be filled by the implementing agent)_

---

## 8. Proposed answer

### 8.1 Before / after

| Quantity | Submitted | Revised | Source |
|---|---|---|---|
| Korns-12, Bingo–IsalSR R² | `nan` (bold, "better") | | |
| Korns-12, Bingo–baseline R² | 0.0000 (underlined, "worse") | | |
| Vlad-2, Bingo–IsalSR R² | `nan` (bold, "better") | | |
| Vlad-2, Bingo–baseline R² | 0.9385 (underlined, "worse") | | |
| Cause of the NaN | **not documented** | | AC-1 |
| Bingo seed-problem cells | 1,465 | | AC-2 |
| UDFS seed-problem cells | 1,500 | | |
| Cause of the 35-cell shortfall | **not documented** | | AC-2 |
| NaN can be marked "better" | **yes** (defect) | no | AC-3 |
| NaN policy in the paired test | **unspecified** | | AC-4 |
| `N` reported for R² test | 50 (asserted) | | AC-4 |
| `N` actually used for R² test | unknown | | AC-4 |
| Sensitivity under conservative substitution | not reported | | AC-5 |
| `nan` defined in table captions | no | | AC-6 |
| Korns-12 discussed without its NaN | yes | | AC-8 |

### 8.2 Changes made to the manuscript

| File | Lines (revised) | Change |
|---|---|---|
| | | |

### 8.3 Draft response text

```latex
%% --- R2.7 ---
\begin{response}
%% Structure that works here:
%%  1. Give the cause. Not "these were rare failures" -- the actual mechanism,
%%     with what the logs showed.
%%  2. Volunteer the defect the reviewer did not name: the table typeset nan as
%%     the better value, marking a real R^2 = 0.9385 as worse than a missing
%%     observation. Fixed, with a regression test. Conceding this unprompted is
%%     worth more than waiting for round 2 to raise it.
%%  3. State the NaN policy in the paired test precisely, give the true N per
%%     metric, and give the sensitivity check. This is the half of the comment
%%     that is actually about statistical validity.
%%  4. Close the 1,465-cell gap in the same breath -- the reviewer will connect
%%     it to R2.6 whether or not we do.
\changeref{}
\end{response}
```

### 8.4 Residual risk

> Candidates: the true `N` turning out to be < 50 for some metric, which weakens
> Table 3 slightly — report it anyway, the alternative is a misreported test; a
> reviewer asking why the NaN was not caught before submission; whether the
> conservative substitution changes any significance verdict (if it does, that is
> the headline of this ticket and must be surfaced in §7 immediately).
