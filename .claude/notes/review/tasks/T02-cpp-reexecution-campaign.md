# T02 — Full re-execution on the C++ engine + continuity table

| Field | Value |
|---|---|
| Reviewer comments closed | none directly (produces the evidence for **R1.1**, **R2.6**, **R2.7**) |
| Type | Computational experiment — **PRIORITY. Wave 1. Launches before everything else.** |
| Owner | **Mario** (+ Claude Code) |
| Depends on | T01 (equivalence gate must pass first) · `EXECUTION-PLAN.md` §2 certification gate |
| Blocks | T04, T08, T09, T10 |
| Status | NOT STARTED |
| Target | launch 2026-08-11 · results 2026-09-01 · **hard freeze 2026-09-10** |

---

## 1. Why this ticket exists and why it is separate from T01

T01 proves the C++ engine computes the same canonical strings. This ticket spends
the compute. They are split because the gate between them is a genuine go/no-go:
~72,000 core-hours should not be committed until byte-exactness is signed off and
the projected `S` (T01 AC-6) shows the campaign will actually answer R1.1.

**Decision taken 2026-07-27**: the article reports **only** the C++ numbers and
treats the engine as an implementation detail. A Python↔C++ continuity table is
produced **for the response letter only** — Reviewer 2 checked every number in the
submitted version and will notice that they all moved; the continuity attachment is
what stops that from reading as instability.

---

## 2. Mandatory reading

- `.claude/notes/review/tasks/EXECUTION-PLAN.md` — **read first**; it is
  authoritative on waves, the certification gate, and the no-early-stopping decision
- `.claude/notes/review/source/README.md`
- `.claude/notes/review/source/reviewer-1.md` — R1.1
- `.claude/notes/review/source/reviewer-2.md` — R2.6, R2.7
- `.claude/notes/review/source/verified-discrepancies.md` — D1, D4, D11, E1, E4, E8
- `.claude/notes/review/source/codebase-pointers.md` — results-directory inventory
- `.claude/notes/review/tasks/T01-cpp-core-port.md` — §7 work log and §8.1 numbers
- `CLAUDE.md` (repo root) — Operational requirements, especially the B12 VarAnd
  clone-detection fix and the UDFS `processes: 1` constraint
- `docs/md_files/design/experimental_design/isalsr_experimental_design.md`
- `docs/md_files/design/experimental_design/data_benchmarking_design.md`
- `docs/md_files/changes/cross_problem_dominance_test.md`
- The `picasso-sbatch` skill — **before writing or editing any SLURM script**

---

## 3. Established facts

- The campaign is 2 methods × 2 variants × 50 problems × 30 seeds = **6,000 runs**
  at 43,200 s (12 h) each. The submitted supplementary states 2,640, which is the
  22-problem arXiv configuration and is wrong (R2.6 / D1).
- The submitted main text reports 1,500 seed-problem cells for UDFS (= 50 × 30 ✓)
  but **1,465 for Bingo** — 35 short, unexplained (E4). Almost certainly the same
  root cause as the two NaN rows (D4). This campaign must produce a complete
  6,000-run ledger or explain every missing cell.
- `CLAUDE.md` records a known defect: **B12, VarAnd clone detection**, fixed
  2026-04-01, with the standing note *"Production Bingo+IsalSR and diversity
  experiments need re-execution."* Establish before launch whether the submitted
  Bingo+IsalSR numbers predate that fix. If they do, that is independently
  sufficient reason for this campaign and is a strong, honest line in the response
  letter.
- UDFS is **budget-saturated**: 36 of 50 problems report `T ≈ 43,200 s` for both
  variants. A faster canonicaliser does not shorten a UDFS run; it buys more
  evaluations inside the same wall-clock.
- UDFS uses `multiprocessing.get_context('spawn')`. Spawned workers re-import
  modules and bypass the monkey-patch. All production configs use `processes: 1`.
  **This constraint carries over to the native engine** — verify the native
  extension is loaded inside whatever process actually calls `evaluate_cgraph`.
- Several sibling result directories exist (`wl_subtree`, `wl_subtree_unified`,
  `wl_subtree_hard`, `wl_subtree_cherrypicked`, `wl_subtree_roundoff`) and the
  manuscript never records which campaign produced which table. This campaign must
  not repeat that.

---

## 4. Non-goals

- Do not change any search hyperparameter, operator set, seed, sampling protocol,
  or statistical procedure. The **only** intended change is the canonicalisation
  engine. Anything else is a confound.
- Do not add problems here (that is T05) or variants here (T04, T03).

---

## 5. Work specification

### 5.1 Pre-launch audit
Establish, and record in §7, the exact provenance of the submitted numbers: which
results directory produced Table 2, Table 3, Table 6 and Table 7 of the submission,
and whether the Bingo+IsalSR arm predates the B12 fix. This audit is a prerequisite
for T08 and T09 as well; do it once, here, properly.

### 5.2 Campaign definition — **see `EXECUTION-PLAN.md`, which is authoritative**

This is **Wave 1**, the priority campaign and the headline result of the revision.
It launches first; every other wave is scheduled around it.

Identical to the submitted protocol except for the engine: S50 × 30 seeds ×
{UDFS, Bingo} × {baseline, IsalSR}, `max_time = 43,200 s`, 1 core per run, 8–16 GB
(Bingo+IsalSR historically needed 128 GB — re-check under the native engine; the
C++ dedup set should reduce it substantially).

Three things are settled elsewhere and must not be re-litigated here:
- **The `baseline` arm is NOT re-run on S50** (`EXECUTION-PLAN.md` §5). Its March
  numbers stand and are paired against the new IsalSR arm. Wave 1 is therefore
  `isalsr` only: 2 arrays, 3,000 runs. The residual wall-clock confound is
  characterised via D1–D3 and mitigated by pinning `--constraint` to the March
  baseline's predominant node type; both are mandatory, not optional.
- **Wave 2** (the T05 extension) launches later into **the same campaign root and
  MANIFEST**, and is where `baseline` does run. Splitting the launch does not split
  the provenance provided the root, the engine build and the configs are identical.
  Record the split.
- **Blocking prerequisites** are enumerated in `EXECUTION-PLAN.md` §2b: T01, T06's
  instrumentation half, T08's root-cause half, a frozen MANIFEST schema, and checks
  P1–P3. Anything measured *during* a run must exist in the code before launch.

### 5.2b Certification gate — nothing launches before it passes

**Do not submit an array unless you are 100% sure the code is correct.** The eight
conditions G1–G8 in `EXECUTION-PLAN.md` §2 are the gate: equivalence, clean suite,
a *parsed* smoke log, non-zero dedup on the IsalSR arm, confirmation that the
**native** engine loaded on a compute node (not a silent Python fallback),
`sbatch --test-only`, one real single task completed and validated, and a correct
`MANIFEST.json`.

G7 — the single real task — is not optional and not parallelisable with the full
launch. A subtly wrong array is worse than a failing one: a failing array is caught
in minutes, a wrong one is caught during analysis in September.

### 5.3 Provenance discipline
One campaign root, `…/results/model_validation/real_benchmarks/cpp_v1/`, with a
`MANIFEST.json` recording: git commit of `IsalSR`, native-extension build hash,
compiler and flags, node CPU model per run, engine (`native`/`python`), config YAML
hash, and the seed. **Every table in the revised manuscript must be traceable to
exactly one campaign root.** This closes the root cause behind D1/D4/E1/E4.

### 5.4 No early stopping — settled 2026-07-27

A saturation-based early stop was considered and **rejected**. Every arm runs the
full 43,200 s budget. Reasoning is recorded in `EXECUTION-PLAN.md` §4 so it is not
re-proposed; in one line: wall-clock `T` is a *reported* quantity that produces
`S`, and a stop rule firing asymmetrically across the two arms of a pair would make
`S` a measurement of the stop rule rather than of the method — silently
manufacturing the R1.1 result we are trying to earn.

The protocol's own convergence criterion (`evolve_until_convergence`) continues to
terminate runs exactly as before. That is the protocol, not early stopping.

### 5.5 Analysis regeneration
Re-run the full pipeline on the new campaign root and regenerate every artefact:
`benchmark_summary_*`, `computational_overhead_*`, `cross_method_*`,
`reduction_comparison_*`, `three_axis_*`, `cross_problem_dominance_*`,
`global_summary.json`, plus the per-problem LaTeX tables and all four figures.

### 5.6 Continuity table (response letter only)
A per-axis mapping from the submitted Python numbers to the new C++ numbers, at
method granularity and for the headline per-problem rows the reviewers named. It
must let a reviewer confirm that ρ and R² are statistically unchanged (the
representation did not change) while the cost axis moved (the engine did). If ρ or
R² *did* move materially, that is a finding: investigate it before writing it up.

---

## 6. Acceptance criteria

- **AC-0.** §7 Work log filled in as the work proceeds (see README standing rules).
- **AC-1.** Pre-launch audit (§5.1) recorded in §7, including a definitive answer on
  the B12 fix and the submitted Bingo+IsalSR arm.
- **AC-1b.** Certification gate G1–G8 (`EXECUTION-PLAN.md` §2) passed with evidence
  recorded in §7, **before** any array was submitted. G7's single-task log is quoted.
- **AC-2.** Wave 1 complete, or every missing run individually accounted for with a
  cause. **The 1,465-cell shortfall must not recur unexplained.**
- **AC-3.** `MANIFEST.json` present and complete; one campaign root shared with
  Wave 2; every revised table traceable to it.
- **AC-4.** D1–D3 (`EXECUTION-PLAN.md` §5) run and reported; node-type pinning
  applied at launch; the residual confound stated as a bounded limitation in the
  paper and in §8.4, not omitted.
- **AC-5.** Full analysis pipeline regenerated; all artefacts present.
- **AC-6.** Continuity table (§5.6) produced as a standalone artefact ready for the
  response-letter appendix.
- **AC-7.** ρ and R² compared Python vs C++ with a statistical test, not eyeballed.
  If they differ significantly, the cause is identified before the ticket closes.
- **AC-8.** §8 filled.

---

## 7. Work log

_(empty — to be filled by the implementing agent)_

---

## 8. Proposed answer

> This ticket produces evidence rather than a direct response. §8.1 and §8.3 feed
> T10 (R1.1), T08 (R2.7) and T09 (R2.6). §8.5 is the response-letter attachment.

### 8.1 Before / after — headline axes

| Quantity | Submitted (Python engine) | Revised (C++ engine) | Δ | Test / source |
|---|---|---|---|---|
| Total runs | 2,640 *(stated)* / 6,000 *(actual)* | | | |
| Bingo seed-problem cells | 1,465 | | | |
| UDFS seed-problem cells | 1,500 | | | |
| ρ, UDFS | 1.56 ± 0.24 | | | |
| ρ, Bingo | 1.83 ± 0.09 | | | |
| Median canon overhead, Bingo | 39.2 % | | | |
| Median canon overhead, UDFS | 0.6 % | | | |
| Search-only speedup `S`, Bingo | **0.93** | | | |
| Search-only speedup `S`, UDFS | 1.07 | | | |
| Problems with `T_IS < T_BL`, Bingo | 4 / 50 | | | |
| CPDT R² test p, UDFS | 0.00018 | | | |
| CPDT R² test p, Bingo | 0.0013 | | | |
| Problems with NaN, Bingo–IsalSR | 2 (Vlad-2, Korns-12) | | | T08 |

### 8.2 Changes made to the manuscript

| File | Lines (revised) | Change |
|---|---|---|
| | | |

### 8.3 Draft response text

_(no direct comment; the engine change is described once in the response-letter
cover paragraph and in the C7/Summary-of-changes note. Draft that paragraph here.)_

```latex
%% cover-paragraph fragment: the engine change
```

### 8.4 Residual risk

> Candidates: reviewers reading a wholesale change of numbers as a different paper;
> heterogeneous Picasso CPUs making wall-clock comparisons noisy; whether the
> engine change should be disclosed in the manuscript at all given it is presented
> as an implementation detail (recommendation: one sentence in Appendix D.3 stating
> the core is a C++ implementation, so the claim is not misleading by omission).

### 8.5 Continuity attachment

> Table to be appended to `response_to_reviewers.tex`. One row per quantity the
> reviewers cited by number, so R2 can walk their own list.
