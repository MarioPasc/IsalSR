# T02 — Full re-execution on the C++ engine + continuity table

| Field | Value |
|---|---|
| Reviewer comments closed | none directly (produces the evidence for **R1.1**, **R2.6**, **R2.7**) |
| Type | Computational experiment — **owns the pre-flight certification suite for Campaign C2** |
| Owner | **Mario** (+ Claude Code) |
| Depends on | T01 (equivalence gate) · T04 + T05 (both gate the launch) · `EXECUTION-PLAN.md` §4 |
| Blocks | T04, T08, T09, T10 |
| Status | NOT STARTED |
| Target | pre-flight complete ≈2026-08-18 · **C2 launch on Mario's sign-off only** · results 2026-09-03 · **hard freeze 2026-09-10** |

---

## ⛔ Amendment 2026-07-31 — read before doing anything on this ticket

**This ticket no longer launches "Wave 1". There are no waves.**
`EXECUTION-PLAN.md` was rewritten on 2026-07-31 and is authoritative. The campaign is
now **Campaign C2**, a single gated launch:

```
{ baseline , hash , isalsr } × { UDFS , Bingo } × ( D1 ∪ D2 ) × 20 seeds
    = 8,400 runs, 6 arrays × 1,400 tasks, 100,800 core-hours, one campaign root
```

Three things changed that this ticket assumed otherwise:

| Was | Is |
|---|---|
| Wave 1 = `isalsr` arm only, 3,000 runs, launches first | **All three arms launch together.** Nothing submits until T04 *and* T05 land |
| The `baseline` arm is **not** re-run on S50 | **It is re-run.** The old decision is superseded (`EXECUTION-PLAN.md` §9.2); D1–D3 confound characterisation and `--constraint` pinning as a *mitigation* are deleted with it |
| 30 seeds | **20 seeds.** Rationale and disclosure obligation: `EXECUTION-PLAN.md` §6.3 and the boxed note in §0.4 |

**What this ticket now owns:** the pre-flight certification suite,
`EXECUTION-PLAN.md` §4 Stages **B, C, D** — the micro-jobs, the 420-task 15-minute
smoke, and the 12-task full-length certification. Stage A is desk work shared with
T04/T05/T08; Stage E is the analysis dry-run; Stage F is Mario's sign-off.

**Inherited from T06 (2026-07-31):** check **B9**, the fallback-counter
re-verification. T06 is closed and is not a launch blocker; it supplies only the pass
threshold. Executing the probe — both hosts, frozen commit, production sampling rate,
overhead re-measured under the C++ engine and the decomposed alphabet — is this
ticket's, because it is certification, not measurement. Re-checked at scale as C1.9.

### 🚫 This ticket does not submit the campaign

**`EXECUTION-PLAN.md` §4.0 SP-0 is binding.** No agent working this ticket submits
C2 or anything resembling it. Everything submitted here is a **probe**: `max_time
≤ 1,800 s`, ≤ 60 tasks, seed 0 only, output to `~/execs/isalsr/t02_*/`, never the
campaign root. The one exception is Stage D's 12 full-length certification tasks,
which are explicitly authorised by §4.4 and are **12 tasks, not 1,400** — submit them
only after Stage C has passed in full.

**Before trusting any Picasso result from this ticket, establish SP-1…SP-6**
(`EXECUTION-PLAN.md` §4.0): provenance, installation freshness (the stale-`.so`
trap), engine **with a negative control**, alphabet on the real candidate stream,
**both hosts**, and live reachability/fallback counters. Report them as a six-row
table in every work-log entry — an entry without it is not evidence.

**SP-7 for this ticket:** the MANIFEST validator passes on a probe run, and per-DAG
`T_canon` and `T_eval` are present and non-zero (check P2).

---

## T16 impact — C2 must run the corrected alphabet (added 2026-07-30)

**This is now a launch blocker, and it is invisible in the logs if you get it wrong.**

T16 found that the adapters emitted `Sub` and `Div` as primitive node types, while
the paper's Σ_SR (Definition 3.2) has twelve labels and **no `-` and no `/`** —
subtraction and division are supposed to enter through `x − y = Add(x, Neg(y))` and
`x / y = Mul(x, Inv(y))`, leaving `Pow` as the only non-commutative operation.
**61.1 % of production candidates carried the wrong labels.** The fix
(`experiments/models/commutative_encoding.py`, applied inline inside both adapters)
is implemented and validated.

**Why it does not change this ticket's budget.** C2 already re-runs IsalSR on all of
`D1 ∪ D2`, so the corrected alphabet adds **zero** runs. What changes is *which code*
C2 executes. The IsalSR arm has **two independent reasons** its submitted numbers are
void — the C++ engine and the alphabet. The `baseline` arm is untouched *by the
alphabet* (it never invokes the adapter); it is re-run in C2 for a different reason,
namely to remove the cross-campaign hardware confound (`EXECUTION-PLAN.md` §2 item 3).

**What moves, and therefore what the continuity table (§5) must explain.** `k`
(+22.9 % Bingo, +22.0 % UDFS), canonical string length (+27 % / +22 %),
canonicalisation cost (+24.6 % / +10.8 %), ρ, and every k-stratified table. Fitness
and everything derived from it — R², NRMSE, solution recovery — do **not** move,
because fitness is computed by the host on the host's own representation and the
runners cache `canon_hash → fitness` without ever calling `evaluate_dag`.

**Gate G9** in `EXECUTION-PLAN.md` §2 is the check: run
`experiments/scripts/verify_alphabet_gate.py` against the G7 single task's real
candidate stream and confirm zero `Sub`/`Div` nodes, zero `-`/`/` characters in any
canonical string, and `Pow` as the only order-sensitive binary operation. It passed
locally on all 10 production configs (~130,000 DAGs). Do not accept unit tests as
evidence here — the production path runs through the orchestrator, the runner, the
monkey-patched evaluation hook and the deduplicator before a DAG reaches the
canonicaliser.

Full write-up: `docs/md_files/changes/t16_commutative_decomposition.md`.

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
  authoritative on the campaign shape, the SP-0…SP-7 Picasso discipline (§4.0), the
  six-stage pre-flight gate (§4), and the no-early-stopping decision (§5.4)
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

_These describe **C1, the submitted campaign**. For C2's shape see §5.2._

- **C1** was 2 methods × 2 variants × 50 problems × 30 seeds = **6,000 runs**
  at 43,200 s (12 h) each. The submitted supplementary states 2,640, which is the
  22-problem arXiv configuration and is wrong (R2.6 / D1).
  **C2 is 3 arms × 2 methods × ≈70 problems × 20 seeds = 8,400 runs.**
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

This is **Campaign C2**, a single gated launch of all six arrays:

```
{ baseline , hash , isalsr } × { UDFS , Bingo } × ( D1 ∪ D2 ) × 20 seeds
    = 8,400 runs, 6 arrays × 1,400 tasks
```

`max_time = 43,200 s`, 1 core per run. **Memory is sized from measurement, not from
history**: Stage C1.11 (`MaxRSS` at 15 min) and Stage D1.2 (`MaxRSS` at 12 h with
≥30 % headroom) produce the production `--mem` per `(method, arm)`. Bingo+IsalSR
historically needed 128 GB from heap fragmentation; the C++ dedup set should reduce
that substantially — **measure it, do not assume it, and do not carry 128 GB forward
by default either.**

Four things are settled in `EXECUTION-PLAN.md` and must not be re-litigated here:

- **All three arms run**, including `baseline` on D1. The "baseline is not re-run"
  decision is superseded (§9.2). What it bought — 36,000 core-hours — is spent to
  remove the cross-campaign, cross-hardware, five-months-apart confound in `S`.
- **One gated launch.** T04 (the `hash` arm) and T05 (D2) both gate it. There is no
  partial launch and no split submission.
- **20 seeds**, with the top-up conditions in the boxed note of §0.4. Every table
  says 20 seeds until all four conditions hold.
- **Blocking prerequisites** are `EXECUTION-PLAN.md` §3: T01, T04, T05, T06's
  instrumentation half, T08's code half, a frozen MANIFEST schema, and checks P1–P5.
  Anything measured *during* a run must exist in the code before launch.

**The `S` caveat, so it is not re-derived optimistically.** T01 AC-6 established that
`T_search = wall_clock − canon_time_total`, hence `dS/dT_canon = 0` **exactly**: no
engine can move `S`. C2 is not justified by `S` and this ticket must not claim it is.
`EXECUTION-PLAN.md` §2 states the six things C2 actually buys.

### 5.2b Certification gate — nothing launches before it passes

**Do not submit an array unless you are 100 % sure the code is correct.** The gate is
`EXECUTION-PLAN.md` §4, Stages **A → B → C → D → E → F**, cheapest first, each
gating the next. This ticket owns Stages B, C and D:

| Stage | What | Cost |
|---|---|---|
| **B** | 8 Picasso micro-jobs (< 5 min each): environment probe, **engine probe with negative control**, alphabet gate on the frozen commit, **equivalence gate re-run on a compute node**, node census, constraint decision, `sbatch --test-only` on all six arrays, resume/corruption behaviour | ≈2 core-h |
| **C** | **420-task, 15-minute full-coverage smoke** — every problem × arm × method. 15 blocking criteria, plus the dedup-off equivalence control (C3), cross-arm data-fingerprint identity (C4) and the comparison against C1 (C5) | ≈105 core-h |
| **D** | **12-task full-length certification** at the real 43,200 s budget: trace problem × 3 arms × 2 methods, plus Korns-12 and Vlad-2 on Bingo (the T08 NaN cells). Plus the detailed single-problem trace (D2) and the T04 Mode 1 replay (D3) | 144 core-h |

Stage D is not optional and not parallelisable with the launch. A subtly wrong array
is worse than a failing one: a failing array is caught in minutes, a wrong one is
caught during analysis in September.

**A 15-minute smoke proves nothing about a 12-hour run.** Memory growth, heap
fragmentation, dedup-set size, timeout paths and convergence behaviour are all
budget-dependent. That is why Stage D exists and why it runs the full budget.

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
- **AC-1b.** **Pre-flight Stages A–E** (`EXECUTION-PLAN.md` §4) passed with evidence
  recorded in §7 and in `EXECUTION-PLAN.md` §11.2's sign-off table, **before** any campaign array was
  submitted. SP-1…SP-6 reported as a six-row table for every Picasso probe. Stage D's
  12 certification logs are quoted. Stage F is signed by Mario.
- **AC-2.** C2 complete (8,400 runs), or every missing run individually accounted for
  with a cause in the status ledger. **The 1,465-cell shortfall must not recur
  unexplained.** If capacity forces truncation, whole `(method, problem, seed)`
  triples are dropped across all three arms — never individual runs
  (`EXECUTION-PLAN.md` §5.5).
- **AC-3.** `MANIFEST.json` present and complete; one campaign root shared with
  all six arrays; every revised table traceable to it.
- **AC-4.** ~~D1–D3 confound characterisation~~ — **superseded** by the full
  three-arm re-launch (`EXECUTION-PLAN.md` §9.2). Replaced by: the node census (B5)
  run and reported; the constraint decision (B6) taken with its evidence; CPU model
  recorded per run; **arm balance across node types reported as a measured covariate**
  in the analysis, whether or not pinning was possible.
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
