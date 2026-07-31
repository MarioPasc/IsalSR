# Execution plan — Campaign C2, the unified three-arm re-launch

Single source of truth for **what gets launched, in what order, and under what
gate**. Referenced by T02, T03, T04, T05, T06, T08. If a ticket and this file
disagree about a launch, **this file wins**; update the ticket.

| Field | Value |
|---|---|
| Campaign id | **C2** (the submitted campaign is C1 = `wl_subtree_unified`) |
| Shape | `{baseline, hash, isalsr} × {UDFS, Bingo} × (D1 ∪ D2) × 20 seeds` |
| Runs | **8,400** (6 SLURM arrays × 1,400 tasks) |
| Budget per run | `max_time = 43,200 s` (12 h), 1 core |
| Core-hours | **100,800** committed |
| Launch model | **one gated launch, all six arrays** — nothing submits until every blocker and every pre-flight gate in §4 has passed |
| Status | **NOT SUBMITTED.** Blockers open (§3). Pre-flight suite not started (§4) |
| Number freeze | 2026-09-10 |

---

## 0. Decision log

Read this before proposing any change to the campaign shape. These decisions are
settled; re-opening one requires a stated reason and an entry here.

### 0.1 — 2026-07-27 (Mario)

- **Early stopping is abandoned.** Full 12 h budget on every run, every arm. §5.4.
- **Gray (T03) is secondary** and reserves no queue capacity. §8.3.
- **Nothing launches as an array until the certification gate passes.**

### 0.2 — 2026-07-30 (T16, the alphabet correction)

Every IsalSR number in the submitted paper was produced under the **wrong
alphabet**. The paper's Σ_SR (Definition 3.2) has twelve labels and **no `-` and no
`/`**: subtraction and division enter through `x − y = Add(x, Neg(y))` and
`x / y = Mul(x, Inv(y))`, leaving `Pow` as the only non-commutative operation. The
adapters emitted `Sub` and `Div` as primitive node types anyway, and **61.1 % of
production candidates contain them.** The code is now aligned to the paper
(`experiments/models/commutative_encoding.py`, applied inline inside both adapters).

What moves: `k` (+22.9 % Bingo, +22.0 % UDFS), canonical string length, ρ, the
reduction factor, canonicalisation cost, and every k-stratified table. What does
**not** move: fitness and everything derived from it (R², NRMSE, solution recovery),
because fitness is computed by the host on the host's own representation and the
runners cache `canon_hash → fitness` without ever calling `evaluate_dag`.

### 0.3 — 2026-07-30 (T01 AC-6): the C++ port cannot move `S`

`T_search` is **derived** as `wall_clock − canon_time_total`. Therefore
`dS/dT_canon = 0` **exactly** and Bingo's `S = 0.93` is invariant to engine speed.
There is no break-even `T_canon`; the literal go/no-go fails against a 1,000× engine.

**This finding survives the C2 redesign unchanged.** C2 is not justified by `S`, and
any text that implies it is must be corrected. What the port does deliver is Bingo
canonicalisation overhead **39.2 % → ≈7.4 %** on the decomposed alphabet (the
alphabet itself costs ≈1.3 pp of that), and the wall-clock penalty. §2 states the
actual justification for the 100,800 core-hours.

### 0.4 — 2026-07-31 (Mario): **full three-arm re-launch, one campaign**

Supersedes the Wave 1/2/3 sequencing and the "baseline is not re-run" decision
(§9.1, §9.2). The campaign is now a **single unified launch**:

```
    { baseline , hash , isalsr }  ×  { UDFS , Bingo }  ×  ( D1 ∪ D2 )  ×  20 seeds
```

Rationale, in Mario's words: *"we will get clean and comparable numbers, and we will
make sure that everything is absolutely comparable."* Every number in the revised
paper then comes from one commit, one build, one node pool, one alphabet, one
protocol, one campaign root.

Four sub-decisions taken with it:

| # | Decision | Consequence |
|---|---|---|
| a | **20 seeds**, not 30 | cuts the cost from 151,200 to 100,800 core-hours. §6.3 states the statistical cost and the disclosure obligation |
| b | **One gated launch, all six arrays** | nothing submits until T04 *and* T05 have landed. Maximum comparability, at the price of inheriting the later of the two slip dates |
| c | **Gray (T03) stays spillover** | not in the committed budget; go/no-go 2026-08-31. §8.3 |
| d | **T04 Mode 1 replay runs on the pre-flight certification streams** | the ρ_exact/ρ_iso number arrives *before* the 100,800 core-hours are committed, which is the entire point of running Mode 1 first. §4.5 (D3) |

> **On the 20 seeds — read this before anyone "optimises" it back to 30.**
>
> **We launch at 20 seeds. Full stop.** The 10 remaining seeds are a *possible later
> top-up*, not part of the plan, not part of the budget, and not something the
> analysis is allowed to depend on.
>
> The reasoning is a priority ordering, not a statistical one: **the first obligation
> is that results land.** A complete, analysed, three-arm campaign at 20 seeds before
> the 2026-09-10 freeze is worth more than a 30-seed campaign that is 80 % done on
> 2026-09-08 — the latter is not a weaker paper, it is *no paper*, because a paired
> design with unbalanced completion cannot be analysed at all (§5.5). §8.2's
> arithmetic is unforgiving: even at 20 seeds we need ≈200–300 concurrent cores.
>
> **The top-up, if it happens.** Seeds 21–30 are purely additive under the resume
> logic (a completed `(method, arm, problem, seed)` is skipped, so re-launching the
> same six arrays with an extended seed range costs only the new cells). It ranks
> **first** in the spillover order, above Gray (§8.4). Conditions, all of them:
>
> 1. All 8,400 runs of the 20-seed campaign are **complete and analysed**, with the
>    cell ledger reconciled (§5.5) — not merely "mostly finished".
> 2. There is enough wall clock left that the top-up can itself complete and be
>    re-analysed before the freeze — 36,000 core-hours for D1, 14,400 for D2.
> 3. The top-up runs the **same** `campaign/c2` tag, build and node pool. A top-up on
>    a different commit is not a top-up; it is a second campaign, and it would
>    reintroduce exactly the confound C2 exists to remove (§2, item 3).
> 4. If only part of the top-up lands, **we report 20 seeds and discard the partial
>    extra cells.** Reporting 30 seeds for some problems and 20 for others makes `S`
>    problem-dependent and is worse than not running them.
>
> Until all four hold, every table, every test and every sentence in the paper says
> **20 seeds**. Write it that way from the start; do not leave a placeholder that
> assumes 30.

---

## 1. Notation and campaign definition

| Symbol | Meaning | n |
|---|---|---|
| `D1` | the suite as submitted (nguyen 12 + feynman 10 + hard 10 + cherrypicked 10 + roundoff 8) | 50 |
| `D2` | Feynman remainder + ODE-Strogatz (T05) | ≈ 20 |
| `D` | `D1 ∪ D2` | ≈ 70 |
| `C1` | the submitted campaign, `wl_subtree_unified` | reference only |
| `C2` | this campaign | — |

**Arms** (`--variants`): `baseline`, `hash`, `isalsr`. `gray` is spillover only.
**Methods**: UDFS, Bingo. **Seeds**: 1…20.

**Array topology.** One SLURM array task = one `(problem, seed)` pair for a fixed
`(method, arm)` — this is what `slurm/workers/models_experiment_slurm.sh` already
does. So C2 is **6 arrays × (70 × 20) = 6 × 1,400 tasks**. See pre-flight check
**A12**: 1,400 exceeds the default `MaxArraySize` on many SLURM builds and the
arrays may need chunking.

**Campaign root** (one, and only one):

```
Picasso : /mnt/home/users/tic_163_uma/mpascual/fscratch/results/isalsr/c2_3arm/
Local   : /media/mpascual/Sandisk2TB/research/isalsr/results/model_validation/real_benchmarks/c2_3arm/
```

with `MANIFEST.json` at the root (schema frozen per T02 §5.3, extended per **A6**).
**Every table and every figure in the revised manuscript must be traceable to this
one root.** That closes the root cause behind discrepancies D1/D4/E1/E4.

Pre-flight output goes to sibling roots that are **never** merged into the campaign
root: `c2_smoke/`, `c2_cert/`, `c2_trace/`.

---

## 2. Why C2 costs 100,800 core-hours — the justification, stated once

If this paragraph cannot be written, the campaign should not run. T01 AC-6 already
established that the headline quantity R1.1 complains about (`S`) **cannot** move
(§0.3), so "the C++ port makes it faster" is not the justification and must not be
written as one.

C2 buys six things, none of which is `S`:

1. **The three-arm comparison R1.4 demands.** The hash baseline must run under the
   same protocol, same problems, same seeds, same hardware as the other two arms.
   There is no way to obtain that from C1, which has no hash arm at all.
2. **The corrected alphabet.** Every IsalSR number in C1 is void (§0.2). `k`,
   canonical string length, ρ, canonicalisation cost and every k-stratified table
   must be regenerated. This alone forces the `isalsr` arm.
3. **Elimination of the node-heterogeneity confound.** Under the superseded plan the
   `baseline` arm was reused from March and `S` absorbed any node difference between
   two campaigns run five months apart on a mixed Xeon/EPYC pool. Re-running the
   baseline removes the confound entirely rather than bounding it. R1 can now be
   answered with "both arms, same pool, same window, CPU model recorded per run"
   instead of a limitations paragraph.
4. **The R3.1 coverage extension (D2).** Twenty problems with no prior data.
5. **A defensible `N`.** C1 reports 1,500 UDFS cells and **1,465** Bingo cells — 35
   missing, unexplained (T08 §2.4) — and two NaN cells typeset as *winners*. C2 is
   the opportunity to make cell completeness a measured, enforced property (§5.5)
   rather than something discovered by a reviewer.
6. **The overhead figure.** Bingo canonicalisation overhead 39.2 % → ≈7.4 %, measured
   on Picasso hardware under the native engine, for all three arms.

**What C2 does not buy: a better `S`.** State that in the response letter before a
reviewer states it for us.

---

## 3. What must land before anything is submitted

The dividing line: **anything measured *during* a run must be in the code before
launch; anything computed *after* can land later.** Getting this wrong means
re-running 8,400 jobs to recover a counter.

### 3.1 Blocking tickets

| Ticket | What exactly is required | Status 2026-07-31 |
|---|---|---|
| **T01** | The engine. Equivalence gate passed, **and re-passed on a Picasso compute node** (pre-flight **B4**) | AC-0..4, 6..9 met. **AC-5 open** — the Picasso benchmark has never been submitted; `slurm/t01_close/` is written and syntax-checked only. AC-3 gate 3 + AC-8 closed on the workstation: 117,798 evolved decomposed DAGs, 0 mismatches, commit `98fd57a` |
| **T16** | Adapter decomposition in `commutative_encoding.py` + both adapters | **DONE 2026-07-30.** Re-assert on the frozen commit at **B3** |
| **T04** | The three fixed-order serialisations, the `hash` runner for both methods, the hash-arm counters, and the stream persistence Mode 1 needs. **AC-1 soundness proven on the 14,841-DAG corpus** | NOT STARTED. **Gates the whole launch** (§0.4b) |
| **T05** | D2 problem definitions, sympy ground truth, unit tests, configs, and the **pre-registered, committed** selection rule (T05 AC-3) | NOT STARTED. **Gates the whole launch** (§0.4b) |
| **T06** | The **instrumentation half only** (§4.1 counters for the five fallback paths). Not the analysis, not the write-up | done — re-verify it survives the C2 code at **C1.9** |
| **T08** | The **root-cause half** (§5.1) plus any *runtime* fix it implies. The analyzer-side fixes (NaN-as-winner, NaN policy) are needed for the pre-flight analysis dry-run **E3**, so in practice all of T08's code half must land | root-cause done; code fixes to confirm at **A9** |
| **T02** | §5.3 MANIFEST schema, **frozen**, extended for three arms (**A6**) | to freeze |

**T03 (Gray), T07 (proofs), T09–T13 (manuscript) do not block.** T07 has one
coupling: T06's definition of a precondition violation must match T07's statement.
Agree the definition, then instrument.

### 3.2 Engineering checks that are nobody's ticket

| # | Check | If it fails |
|---|---|---|
| **P1** | **Per-candidate stream persistence.** T04 Mode 1 replays a DAG/canonical-hash stream. C1 persisted only aggregate counts, so replay of the submitted campaign was never possible. Decide the format and the sampling rate (full persistence is millions of entries per run) and land it before launch | Mode 1 can only ever replay C2 post-hoc, and the ρ_exact/ρ_iso go/no-go on the hash arm is lost |
| **P2** | **Cost fields survive the C++ port.** `T_canon` and `T_eval` per DAG feed T10's break-even analysis and R1.1's answer | Restore before launch |
| **P3** | **Data fingerprinting.** Every run must record `sha256` of `(X_train, y_train, X_test, y_test)` | Without it, **C5** cannot prove the three arms saw identical data, and the paired design is unverifiable |
| **P4** | **Terminal-status record.** Every run must write a status record **even when it fails** (§5.5) | T08's 35-cell shortfall recurs at 8,400-run scale and is again unexplainable |
| **P5** | **Allocation sizing.** §8.2 — 100,800 core-hours against the freeze needs ≈200–300 concurrent cores. **Policy half MEASURED 2026-07-31 and it passes**: QOS `long_uma` allows `MaxWall = 7 days` and **`cpu = 9000` concurrent cores per user**, with `MaxJobsPU`/`MaxSubmitJobsPU` unset. 300 cores is 3.3 % of the entitlement | The binding constraint is therefore **contention, not policy** — which cannot be read from `sacctmgr` and must be measured. Time the 420-task smoke (Stage C) end to end and divide: that gives the *achieved* concurrency under real queue pressure, and it is the number §8.2 needs |
| **P6** | 🔴 **Quota headroom (see A13).** FSCRATCH is at 248.4k/250.0k files; HOME is over its space quota with 6 days of grace | C2 hits the hard file quota mid-campaign, and every running task keeps burning wallclock while all its writes fail. **Fix before Stage C, not before launch** |

---

## 4. Pre-flight certification suite

> **Do not submit an array to Picasso unless you are 100 % sure the code is
> correct.** A 1,400-task array failing identically costs 1,400 allocations and a day
> of queue time; a *subtly wrong* array costs the deadline, because the error is
> found during analysis in September.

Six stages, cheapest first. **Each stage gates the next.** A failure means fix, then
re-run that stage from the top — not "note it and continue".

Total pre-flight compute: **≈250 core-hours, 0.25 % of the campaign.**

Procedure and commands: `.claude/skills/review-ticket/references/picasso-loop.md`.
SLURM directives: invoke the **`picasso-sbatch`** skill; it is the authority.
Every check writes an evidence artefact into `c2_preflight/` and a row in §10.2.
**"I checked it" is not evidence. A parsed file is.**

### 4.0 Standing Picasso discipline — SP-0…SP-7

> **This subsection is binding on every ticket (T02, T03, T04, T05, T06) and on every
> agent working one.** T02/T04/T05/T06 reference it rather than restating it. If a
> ticket appears to authorise something wider than SP-0, the ticket is stale and this
> section wins.

#### SP-0 — Nobody except Mario submits the campaign

No ticket, and no agent working a ticket, ever submits C2 or anything resembling it.
C2 is submitted **once**, by Mario, after Stage F sign-off (§4.6).

Everything a ticket submits to Picasso is a **probe**, and probes are hard-capped:

| Limit | Value |
|---|---|
| `max_time` per task | **≤ 1,800 s (30 min)** |
| Tasks per submission | **≤ 60** |
| Seeds | seed **0** only (never 1…20 — a probe output must never be mistakable for a campaign cell) |
| Output root | `~/execs/isalsr/<ticket>_<purpose>/` — **never** the campaign root, never `c2_smoke/`/`c2_cert/` unless the ticket owns that stage |
| Wall-clock cost | a probe that would cost more than ≈30 core-hours needs Mario's approval first |

A probe answers *"does this work on Picasso?"*. It does **not** produce a number for
the paper. Any number a probe produces is provisional until C2 reproduces it.

#### SP-1…SP-6 — the standing property probe

**Before trusting any Picasso result — yours or anyone's — establish all six.** These
are not paranoia; each one has already burned this project at least once.

| # | Property | How to establish it | Why it exists |
|---|---|---|---|
| **SP-1** | **Provenance.** You are running the commit you think you are | Record `git rev-parse HEAD`, `git describe --tags --always --dirty`, and working-tree cleanliness *from the compute node*. Compare against the commit you synced | A `-dirty` tag or a hash mismatch invalidates the probe silently. Nothing downstream is worth reading |
| **SP-2** | **Installation freshness.** The *installed package* is the code you edited | Print `isalsr.__file__`, then `stat -c "%y" $(python -c "from isalsr.core import _native; print(_native.__file__)")`. Assert the `.so` mtime post-dates the last commit touching `src/isalsr/core/*.cpp`. Rebuild only with `pip install -e . --force-reinstall --no-deps` — **never** `--no-build-isolation` (it aborts with `BackendUnavailable` and the **stale `.so` keeps loading**). Never read pip's status through a pipe (`pip ... \| tail` reports `tail`'s exit code) | Python resolves from the repo, the extension from site-packages. **A C++ edit can appear to have no effect while the Python half of the same change works.** A repo-local `find` will not reveal the stale build. This is how the 2026-07-29 CONST-normalisation removal was caught mid-verification |
| **SP-3** | **Engine, with a negative control.** The C++ canonicaliser is genuinely live | Assert `_ENGINE == native` **and** that `fast_canonical_string` dispatches to C++. Then re-run the same probe with the Python path forced and assert it reports `python` | **A probe that reports `native` in both directions proves nothing** and is itself a defect. A silent pure-Python fallback looks exactly like success |
| **SP-4** | **Alphabet.** The paper's Σ_SR is what is actually being canonicalised | On the probe's **own candidate stream** — not in unit tests — count `NodeType.SUB`, `NodeType.DIV`, and `-`/`/` in every canonical string. All four must be **0**. `POW` may appear only where the operator set permits. Harness: `slurm/alphabet_gate/` (~90 s) | 61.1 % of C1's candidates carried the wrong labels and **it is invisible in the logs**. It surfaces only in September, during analysis, after the compute is spent |
| **SP-5** | **Both hosts.** UDFS **and** Bingo, every time | Every probe exercises both. A result on one host is not a result | The two hosts have different adapters, different dedup hooks and different failure modes: UDFS monkey-patches `evaluate_cgraph` at module level and its `spawn` workers bypass the patch (safe only because `processes: 1`); Bingo's `VarAnd` produces `parent.copy()` offspring with `fit_set=True` ~36 % of the time (B12). A fix verified on one host is unverified on the other |
| **SP-6** | **Reachability and fallback counters are live.** The T06 ledger is counting, not dead | The five paths (pre-normalisation violation, post-normalisation violation, 60 s timeout, conversion failure, canonicalisation raised) are present and **finite** in the probe output, at the production sampling rate | **A zero-everywhere ledger means the counters are dead, not that the rates are zero.** These counters cannot be recovered post-hoc: the population only exists while a search runs |

**Report all six in every Picasso work-log entry**, as a fixed six-row table. An entry
without it is not evidence and the check does not count as passed.

#### SP-7 — the ticket's own contribution assertion

Each ticket adds **one falsifiable statement** its probe must establish about the
thing that ticket contributes, and states it in its own amendment block:

| Ticket | SP-7 |
|---|---|
| **T02** | The MANIFEST validator passes on a probe run, and per-DAG `T_canon` and `T_eval` are present and non-zero (P2) |
| **T04** | The `TOPOLOGICAL` fixed-order hash runs **inside the live search** on both hosts, and the three-rung shadow counters run inside the live `isalsr` search on both hosts **without an OOM and at constant memory** (AC-10); the candidate stream persists at the chosen rate and replays; on identical replayed input `ρ_hash ≤ ρ_isalsr` **without exception**; equal hash ⇒ equal canonical string (soundness). *Rescoped 2026-07-31: one order ships as the arm, three ship as offline serialisations — T04 §4.* |
| **T05** | Every D2 problem loads on Picasso with the expected train/test shapes, carries a `sympy_expression` so `solution_recovered` is computable, and runs 30 min on both hosts without crashing, under the declared operator set |
| **T06** | *(No probe of its own — T06 is closed.)* T06 **supplies the pass threshold** for check **B9**, which T02 executes. T06 reopens only if B9 fails |

---

### 4.1 Stage A — desk checks (no queue time)

| # | Check | Pass criterion | Evidence |
|---|---|---|---|
| **A1** | **Freeze the commit.** Annotated tag `campaign/c2` on the exact commit C2 will run. Working tree clean; tag pushed | tag resolves; `git status` empty | `git show campaign/c2` |
| **A2** | `pytest tests/ -v`, `ruff check src/ tests/ experiments/`, `mypy --strict src/isalsr/` on `campaign/c2` | all green, zero skips in the core suite | raw command output, not a claim |
| **A3** | **Backend parity.** Rebuild per `CLAUDE.md` (`pip install -e . --force-reinstall --no-deps`; **never** `--no-build-isolation`). Verify the `.so` mtime is newer than the last C++ edit **at the site-packages path**, not in the repo tree. Then run every core-semantics check against **both** `backend="python"` and `backend="cpp"` | byte-identical canonical strings; `.so` mtime post-dates the last C++ commit | `stat -c "%y" $(python -c "from isalsr.core import _native; print(_native.__file__)")` + parity report |
| **A4** | **Config equivalence.** Dump the resolved hyperparameters for all 6 `(method, suite)` configs. The three arms must differ **only** in the `--variants` flag; nothing arm-specific may live in a YAML | a diff table showing operator set, pop size, stack size, cx/mut rates, LM settings and `max_time` identical across arms for a given `(method, problem)` | `c2_preflight/config_diff.md` |
| **A4b** | **Operator-set policy — decide and record.** C1 used *different* operator sets per tier (`hard`/`cherrypicked` add `sqrt`, Bingo adds `pow`). **Recommendation: freeze D1's per-tier sets exactly as submitted** so continuity with C1 holds, and adopt the hard-tier set for D2, disclosed in Appendix D.2. The invariant that actually matters: **for a fixed `(method, problem)` the operator set is identical across all three arms** | the invariant holds for 70/70 problems; the policy is written into the MANIFEST | `c2_preflight/operator_sets.csv` |
| **A5** | **Seed declaration.** Seeds 1…20, and confirm they are the *same integers* C1 used, so the continuity table (§7) can restrict C1 to the same 20 seeds and compare like-for-like. Seed 0 is reserved for smoke and must never appear in the campaign | recorded in MANIFEST; `0 ∉ seeds` | MANIFEST |
| **A6** | **MANIFEST schema frozen** and extended for C2: git commit + tag, native build hash, compiler + flags, config sha256 per `(method, suite)`, operator-set policy, arm list, seed list, alphabet version (`decomposed`), engine, node-constraint string, submission splits. Plus a validator that **fails** on any missing field | validator exits non-zero on a deliberately truncated MANIFEST | `experiments/models/manifest.py` + its test |
| **A7** | **RunLog schema accepts three arms.** `RunMetadata.representation` currently documents `"baseline" or "isalsr"`. Extend to `"hash"`; extend `hardware` to carry `cpu_model`, `hostname`, `slurm_job_id`, `slurm_array_task_id`, `mem_requested_gb`, `max_rss_gb`, `engine`, `git_commit`, `build_hash`, `config_sha256`, `data_fingerprint` | round-trip `to_dict`/`from_dict` test passes for all three arms | `tests/unit/test_schemas.py` |
| **A8** | **Analyzer three-arm readiness.** `analyze.py` accepts `--variants baseline,hash,isalsr`; pairwise CPDT with Holm across **three** contrasts and Friedman/Nemenyi over three arms are implemented and unit-tested on synthetic data | a synthetic case with a known answer reproduces it; the Holm correction divides by 3, not 2 | `tests/unit/test_three_arm_stats.py` |
| **A9** | **T08 code half landed.** NaN can never be marked better in `aggregation.py` (regression test); NaN policy in `statistical_tests.py` explicit, tested, and the reported `N` matches what the code does, per metric | both regression tests green | test output |
| **A10** | **Failure ledger implemented (P4).** A run that raises, OOMs or is time-killed still leaves a status record | kill a local run with `SIGKILL` mid-search; a status row still exists | `c2_preflight/ledger_demo.csv` |
| **A11** | **Hash-collision bound, stated not hoped.** Both the IsalSR dedup set and the T04 hash arm use 64-bit keys. Birthday bound `n²/2⁶⁵`: at `n = 10⁷` entries per run this is `2.7 × 10⁻⁶`; across 5,600 dedup-bearing runs the expected number of collisions is `≈1.5 × 10⁻²`. Record the arithmetic and the observed max entries per run from Stage C | a written bound in the MANIFEST notes, and a measured `max(n)` from Stage C that does not invalidate it | `c2_preflight/collision_bound.md` |
| **A12** | ~~SLURM limits~~ — **MEASURED 2026-07-31, PASSES.** `MaxArraySize = 4096` (so 1,400-task arrays are fine, **no chunking needed**), `MaxJobCount = 15000`, `MaxSubmitJobsPU` unset. Re-check on the frozen commit's submission day only if it has been weeks | 1,400 < 4,096 ✓ | `scontrol show config` |
| **A13** | 🔴 **Storage and file-count projection — this is now a live blocker, not a formality.** Measured 2026-07-31: FSCRATCH is at **248.4k files against a 250.0k soft quota** (400k hard) and HOME is **0.56 TB against a 0.28 TB soft quota with 6 days of grace left**. C2 writes ≥5 files per run × 8,400 runs ≈ **42,000 files**, plus 420 smoke runs. **The account does not currently have room for it.** Required: free FSCRATCH file headroom (archive or delete old campaigns), bring HOME back under quota before the grace expires, and — per the ≥15,000-file rule — either consolidate per-run output into one archive or **mail `soporte@scbi.uma.es` before the first array** | `quota` shows ≥60,000 files of FSCRATCH headroom and HOME under its soft quota, **before** Stage C | `c2_preflight/storage_projection.md` + a `quota` capture |

### 4.2 Stage B — Picasso micro-jobs (each < 5 min, 1 task)

These exist because **everything that only appears on a compute node appears there
and nowhere else**: module differences, dataset paths, permissions, memory profile,
environment activation, compiler-dependent floating point.

| # | Check | Pass criterion |
|---|---|---|
| **B1** | **Environment probe.** One task printing: hostname, `lscpu` model, `isalsr.__file__`, the native module path **and mtime**, `_ENGINE`, `git describe --tags`, numpy/scipy/bingo versions, `free -g`, `$TMPDIR`, and a resolvability check on every D1∪D2 dataset path. Reuse `slurm/smoke_cpp/` | `_ENGINE == native`; tag `== campaign/c2`; 70/70 dataset paths resolve |
| **B2** | **C++ capability probe with a negative control.** Assert `fast_canonical_string` dispatches to C++. **Then re-run the same probe with the Python path forced** — the variable is **`ISALSR_ENGINE=python`** (there is no `ISALSR_FORCE_PYTHON`) — and assert it reports `python`. 🔴 **Assert on observed dispatch, not on a reported string.** Until 2026-07-31 `fast_canonical_string` read `backends.DEFAULT_BACKEND` and bypassed the override, so `engine()` and `build_info()` both said `python` while C++ ran; this probe would have **passed while proving nothing**. Fixed in `canonical.py:349` (→ `_backends.engine()`), found by T04. Verify by counting calls into `_cpp_ext.fast_canonical_string`, not by printing the engine name | the probe reports `native` **and observably calls C++** in run 1, and reports `python` **and observably does not** in run 2. **A probe that says `native` in both proves nothing** — and one that says `python` while running C++ is worse |
| **B2b** | ✅ **RESOLVED 2026-07-31 by T04 — no action, recorded so it is not re-opened.** The B2 defect meant any both-backends sweep driven by `ISALSR_ENGINE` was running **C++ twice**, which would have voided T01's AC-3 gate 3 (117,798 evolved decomposed DAGs). Checked: `experiments/scripts/equivalence_gate_evolved.py:11–12` and `experiments/models/equivalence_probe.py:13–14` both select the backend by **explicit `backend=` kwarg**, and `ISALSR_ENGINE` appears in **no** gate harness. **T01's gate stands; it does not re-run.** Exposure was limited to probes not yet written | verified by grep over `equivalence_gate*.py`, `equivalence_probe.py`, `slurm/t01_close/`, `slurm/smoke_cpp/` |
| **B3** | **Alphabet gate on the frozen commit.** `bash slurm/alphabet_gate/launcher.sh` (~90 s). Precedent: job 1692451, 2026-07-30, 5,551/5,551/225/461 DAGs, zero forbidden labels | 0 `NodeType.SUB`, 0 `NodeType.DIV`, 0 `-`, 0 `/` in any canonical string; `POW` present only where the operator set permits |
| **B4** | **Equivalence gate re-run on a compute node.** T01 G1 passed on the workstation only. Re-run at reduced scale on Picasso: exhaustive `k = 1..8` + ≥5,000 evolved **decomposed** DAGs, byte-exact C++ vs Python | 0 mismatches, 0 errors, `self_comparison == false`. A workstation pass does **not** certify a different compiler, libstdc++ or CPU |
| **B5** | **Node-pool census.** A 20-task, 1-minute array recording `lscpu` model + a fixed single-core canonicalisation microbenchmark per task | the empirical distribution of node types reachable from our QOS, with a per-model speed factor |
| **B6** | **Node-constraint decision.** *Measured 2026-07-31: every node carries its family as a **feature**, so the choice is finer than `intel`/`amd`* — `sd` (52 c, 182 GB, Intel, **avx512**), `sr` (128 c, 439 GB, AMD), `bc` (256 c, 683 GB, AMD, **avx512**), `bl` (128 c, 1855 GB, AMD). Choose one with enough capacity for the concurrency §8.2 needs. If none has it, **do not pin** — mandate CPU-model recording per run (A7) and report arm balance across node types as a measured covariate | either a pinned constraint, or a written argument plus a balance-reporting plan. Never "we assume it was fine" |
| **B6b** | 🔴 **AVX-512 portability of the C++ engine.** The login node is a Xeon Gold 6230R — an `sd` machine **with AVX-512** — but `sr` and `bl`, the bulk of the CPU cluster, **do not have it**. An extension built on the login node with `-march=native` emits AVX-512 and dies with **SIGILL** the moment it lands on `sr`/`bl`. Check the build flags in `pyproject.toml`/`CMakeLists.txt`, then run the same import probe on an `sd` node and on an `sr` node | the native module imports and canonicalises correctly on **both** an `avx512` node and a non-`avx512` node. If it does not, either rebuild with a portable baseline (`-march=x86-64-v2`) or pin `--constraint=avx512` — and note that pinning to `avx512` restricts the pool to `sd`+`bc`, which interacts with B6 |
| **B7** | **`sbatch --test-only` on all six arrays**, with the real `--array`, `--mem`, `--time` and `--constraint` | exit 0 on all six; the reported task count is exactly 1,400 per array (or the A12 chunking equivalent) |
| **B8** | **Resume and idempotency.** Run one task; re-submit it; then corrupt its `run_log.json` and re-submit again | second run **skips**; the corrupted run is **detected, deleted and re-run**. Both behaviours observed, not assumed |
| **B9** | **T06 counter re-verification** (owned by T02, threshold set by T06). One ≤30 min probe on **both hosts** on the frozen commit: the five fallback counters present and finite **at the production sampling rate**, and the instrumentation overhead **re-measured under the C++ engine and the decomposed alphabet**. Both changed underneath T06's original measurement — an overhead that was negligible as a *fraction* of a Python canonicaliser costing ~24× more per DAG may not be negligible now | counters live and finite on both hosts; overhead below the threshold T06 supplies. **A zero-everywhere ledger means the counters are dead, not that the rates are zero** — at N = 10,000 a short run samples almost nothing (the 2026-07-28 60 s smoke drew 5 DAGs), so design the probe so a live counter is distinguishable from a dead one. **If the overhead is now material**, the counters come out of C2 and T06 reopens for a separate subsampled characterisation run: a violation *rate* does not need the full campaign, a paired *timing* does |

### 4.3 Stage C — the 15-minute full-coverage smoke (**420 tasks, ≈105 core-hours**)

This is the coverage test: **every problem × every arm × every method, at least
once, on real Picasso hardware.**

**Configuration.** `max_time = 900 s`, **seed 0** (deliberately outside the campaign
seed set so a smoke output can never contaminate C2), all ≈70 problems × 3 arms ×
2 methods = **420 tasks**. Output root: `c2_smoke/`. Resources: as production, so
the memory profile measured here is the profile that sizes production.

Every criterion below is **blocking**. A single violation stops the stage.

| # | Criterion | Threshold |
|---|---|---|
| **C1.1** | Every task exits 0 | 420 / 420 |
| **C1.2** | Every `run_log.json` exists, parses, and validates against the extended RunLog schema — **every field present, correct type**: `r2_train`, `r2_test`, `nrmse_train`, `nrmse_test`, `mse_test`, `solution_recovered`, `jaccard_index`, `model_complexity`; `wall_clock_total_s`, `wall_clock_search_only_s`, `canonicalization_precomputed_s`, `canonicalization_runtime_s`, `cache_hit_rate`, `cache_hits`, `cache_misses`, `estimated_time_saved_s`, `time_to_r2_099_s`, `time_to_r2_0999_s`, `evaluation_time_s`, `overhead_time_s`; `total_dags_explored`, `unique_canonical_dags`, `empirical_reduction_factor`, `max_internal_nodes_seen`, `theoretical_reduction_bound`, `redundancy_rate`; `symbolic_form`, `isalsr_string`, `canonical_string`, `n_nodes`, `n_edges` | 420 / 420 |
| **C1.3** | **No NaN and no inf** in any regression metric | 420 / 420. Any NaN is a live T08 defect and blocks Stage D |
| **C1.4** | Every dataset loaded with the expected train/test shapes, asserted against the benchmark registry (Vlad-7 is 300/1200, Keijzer-6 is 50/120, Pagie-1 is 676/2500 — these are not typos, do not "fix" them) | 70 / 70 problems |
| **C1.5** | `solution_recovered` is **computable** for every problem, i.e. a `sympy_expression` ground truth exists — the known gap for the D2 additions (T05 AC-4) | 70 / 70 |
| **C1.6** | `isalsr` arm: `unique_canonical_dags > 0` and `empirical_reduction_factor ≥ 1` on **140/140**; `ρ > 1` on **≥ 90 %**. `ρ < 1` is arithmetically impossible and means a counter is broken. A ρ of exactly 1.0 everywhere means the dedup hook is dead and the entire arm is a null result | see cells |
| **C1.7** | **Hash-arm sanity:** `ρ_hash ≤ ρ_isalsr` for the same `(method, problem)`. This is **guaranteed** on identical input streams (a fixed-order hash is sound but incomplete) and **strongly expected** live. Report every violation | 140/140 expected; investigate if violations exceed 5 % |
| **C1.8** | `baseline` arm: dedup counters absent or zero and `canonicalization_runtime_s == 0`. Proves the baseline really is un-instrumented and is not silently paying canonicalisation cost | 140 / 140 |
| **C1.9** | **T06 fallback counters** present and finite on every `isalsr` task, covering all five paths (pre-normalisation violation, post-normalisation violation, 60 s timeout, conversion failure, canonicalisation raised). Report the five rates. This is check **B9** re-run at scale: B9 establishes the counters are alive and affordable on 2 probes, C1.9 confirms it holds across all 70 problems | 140/140 present; the five rates reported; overhead consistent with B9's measurement |
| **C1.10** | `trajectory.csv` non-empty; `timestamp_s` monotone non-decreasing; `best_r2` monotone non-decreasing; `n_dags_explored` monotone non-decreasing; `n_unique_canonical ≤ n_dags_explored` | 420 / 420 |
| **C1.11** | **Memory profile.** `sacct` `MaxRSS` per task, tabulated by `(method, arm)`. Bingo+IsalSR historically needed 128 GB from heap fragmentation; the C++ dedup set should reduce that materially — **measure it, do not assume it**. Size production `--mem` at p99 + 50 % headroom | a table, and a production `--mem` per `(method, arm)` derived from it |
| **C1.12** | **`max_time` is honoured.** Every task terminates at ≈900 s or earlier by convergence; **none** is killed by the SLURM wall limit. A SLURM kill without a `max_time` stop means `max_time` is not reaching `evolve_until_convergence` — the known Bingo defect in `CLAUDE.md` | 0 SLURM time-kills |
| **C1.13** | **Alphabet assertion on the real candidate stream** of every `isalsr` and `hash` task, not only in unit tests: 0 forbidden labels | 280 / 280 |
| **C1.14** | **Engine assertion**: every task records `engine == native` | 420 / 420 |
| **C1.15** | **Cell completeness reconciliation**: expected 420, observed 420, with a machine-checked comparison. This is the mechanism that must prevent a recurrence of C1's unexplained 35-cell shortfall | exact match, or every gap individually named |

#### C2 — the failure ledger (deliverable, not a check)

The smoke must emit `c2_smoke/status_ledger.csv`, one row per
`(method, arm, problem, seed)`:

```
method, arm, problem, seed, exit_code, terminal_status, wall_clock_s, max_rss_gb,
node_cpu_model, hostname, engine, git_commit, config_sha256, data_fingerprint,
n_nan_metrics, nan_fields, exception_class, exception_message, slurm_job_id,
slurm_array_task_id
```

**The production campaign must write the identical ledger.** This is how T08's
"35 missing cells, cause unknown" becomes impossible: every cell is either present
with data or present in the ledger with a named cause.

#### C3 — dedup-off equivalence control (**BLOCKING, and the most valuable check here**)

For 6 tasks (2 methods × 3 problems), run the `isalsr` runner with **dedup forced
off** and compare against the `baseline` arm at the same seed.

**Pass:** identical best expression, identical `r2_train`, identical
`total_dags_explored`. That isolates *"the wrapper itself perturbs the search"*
(RNG consumption, evaluation ordering, object identity) from *"dedup changes the
search"*, which is the effect we claim.

**If it fails, every paired comparison in the paper is confounded by an unintended
side effect of the wrapper.** If exact reproduction is not achievable for a
principled reason, the plan must state the residual difference and **bound** it
before Stage D — not wave at it.

#### C4 — cross-arm data identity (**BLOCKING**)

For every `(problem, seed)`, the `data_fingerprint` (P3) must be **identical across
all three arms and both methods**. If the data generator consumes RNG differently
per method or per arm, the paired test compares different data and the whole design
is void.

**Pass:** `70 × 1` distinct fingerprints at the smoke's single seed, each appearing
exactly 6 times.

#### C5 — comparison against the submitted campaign C1 (**reasoned sign-off, not a threshold**)

Reference:
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/model_validation/real_benchmarks/wl_subtree_unified/analysis/`
(`benchmark_summary_{bingo,udfs}_benchmark.csv`, `three_axis_summary_*`,
`cross_problem_dominance_*`, `computational_overhead_*`).

**Write the expectations down before looking at the numbers.** Restricted to D1:

| Quantity | Expectation | What a violation means |
|---|---|---|
| R² at 900 s vs C1 at 43,200 s | smoke `≤` C1, per problem | a smoke R² materially **exceeding** the published 12 h value means the dataset, the split or the metric changed |
| ρ (isalsr), decomposed vs C1 | smoke ρ **≥** C1 ρ in direction — `k` grew ≈22 %, so there are more internal nodes to permute | a *drop* means either the decomposition is not reaching the canonicaliser or the dedup population changed unexpectedly |
| Korns-12 / Vlad-2, Bingo–isalsr | **finite**, not NaN | the T08 root cause is still live and Stage D must not proceed |
| Cell count | 420/420 present | the C1 shortfall mechanism is still present |
| Baseline R², D1 | within seed noise of C1's baseline at comparable budget | the baseline path changed when it should not have — the baseline never invokes the adapter |
| `POW` presence | only in configs whose operator set includes it | operator-set drift (A4b) |

**Deliverable:** `c2_preflight/smoke_vs_C1.md` — a table with every anomaly either
explained or escalated. An unexplained anomaly blocks Stage D.

### 4.4 Stage D — full-length certification (**12 tasks, 144 core-hours**)

The 15-minute smoke proves nothing about a 12-hour run: memory growth, heap
fragmentation, dedup-set size, timeout paths and convergence behaviour are all
budget-dependent. Stage D is the successor to the old G7 single task and it is
**not optional and not parallelisable with the full launch**.

**Composition — 12 tasks at the full 43,200 s budget:**

| Group | Cells | Why these |
|---|---|---|
| Trace problem | 1 problem × 3 arms × 2 methods = 6 | a structural-bottleneck problem where IsalSR is predicted to help (recommend **Pagie-1** or **I.15.10**); this is the run that produces the detailed trace |
| NaN problems | Korns-12 + Vlad-2 × 3 arms × Bingo only = 6 | **the T08 AC-7 evidence.** These are the two cells that were NaN in the submission |

**Pass criteria:**

| # | Criterion |
|---|---|
| **D1.1** | 12/12 complete within the SLURM wall limit with ≥10 % headroom |
| **D1.2** | `MaxRSS` within the requested `--mem` with **≥30 % headroom**, at 12 h — this, not the smoke, is what sizes production memory |
| **D1.3** | Full `run_log.json` + `trajectory.csv` + ledger row valid on all 12 |
| **D1.4** | Korns-12 and Vlad-2, Bingo–isalsr: **finite R²**. If NaN recurs, C2 does not launch — the root cause is still live and would reproduce at 8,400-run scale |
| **D1.5** | `ρ_hash ≤ ρ_isalsr` on all matched cells |
| **D1.6** | For the D1 cells that existed in C1, the 12 h ρ and R² land in a defensible neighbourhood of C1's values, with the T16 `k` shift accounted for. Differences are explained, not noted |
| **D1.7** | Per-DAG `T_canon` and `T_eval` present (P2) and the overhead percentage computable; sanity-check it against the projected Bingo 39.2 % → ≈7.4 % |
| **D1.8** | MANIFEST written correctly and completely for all 12; the A6 validator passes on it |

#### D2 — the detailed single-problem trace (**explicit deliverable**)

On **one** `(method, problem, seed)` — recommend Bingo × the trace problem × seed 1 —
enable full instrumentation and persist:

1. `c2_trace/candidates.jsonl` (or parquet) — per candidate: `k`, node-label
   multiset, canonical string **and** its hash, the three fixed-order hashes,
   `T_canon`, `T_eval`, fallback path taken, whether it was a dedup hit.
2. `c2_trace/canon_cost_hist.json` — canonicalisation-cost histogram stratified by
   `k`, feeding T10.
3. `c2_trace/fallback_ledger.md` — the five T06 rates with **at least one worked
   example of each residual post-normalisation violation**.
4. `c2_trace/spot_check.json` — 20 candidates drawn at random from the stream,
   re-canonicalised independently in **pure Python**, matched **byte-exact** against
   the C++ output recorded during the run. This is an end-to-end check that the
   engine used *in production* is the engine the gate certified.
5. `c2_trace/stream_size.md` — measured bytes per run at the chosen sampling rate,
   multiplied out to 8,400 runs and checked against A13.

**Pass:** all five artefacts produced; the 20/20 spot check clean; the projected
stream volume inside the storage budget.

#### D3 — T04 Mode 1 replay on the certification streams

Replay `c2_trace/candidates.jsonl` (plus the 12 certification runs' streams) through
all three fixed-order hashers and through IsalSR canonicalisation on **identical
input sequences**. This is the controlled comparison: same inputs, zero search
confound.

**Produces:** `ρ_exact`, `ρ_iso`, `ρ_total` per method, stratified by `k`.

**Two hard correctness checks that only Mode 1 can make:**

- **Hash soundness (T04 AC-1).** Any two DAGs sharing a fixed-order hash must share
  a canonical string. A violation is an unsound merge and kills the arm.
- **IsalSR soundness.** Any two DAGs sharing a canonical string must satisfy
  `is_isomorphic`. Spot-check on the largest equivalence classes in the stream.

**Decision recorded either way.** If `ρ_exact ≈ 1.00` for both methods, the live
hash arm is expected to be a null result — *which is itself the answer to R1.4*, and
that answer is now in hand for ≈0 core-hours. The arm still runs (§0.4), but the
framing in the paper changes and §10.1 must record that we knew.

### 4.5 Stage E — analysis dry-run on the pre-flight data

The analysis pipeline has **never** been run on three arms. Discovering that in
September is the single most expensive failure mode left.

| # | Check | Pass criterion |
|---|---|---|
| **E1** | Full analyzer end-to-end on `c2_smoke/` (420 runs, 3 arms, 1 seed) | every artefact in T02 §5.5 produced without exception, with 3 arms present: `benchmark_summary_*`, `computational_overhead_*`, `cross_method_*`, `reduction_comparison_*`, `three_axis_*`, `cross_problem_dominance_*`, `global_summary.json` |
| **E2** | Three-arm statistics: pairwise CPDT (`isalsr` vs `baseline`, `hash` vs `baseline`, `isalsr` vs `hash`) with **Holm across three contrasts**, plus Friedman/Nemenyi over the three arms per method | outputs exist; `N` reported per metric; the synthetic test from A8 confirms the correction divides by 3 |
| **E3** | **NaN policy, adversarially tested.** Inject a synthetic NaN into a copy of the smoke root | (a) the NaN is **never** bold/marked better; (b) the reported `N` drops by exactly 1 for that metric; (c) the conservative-substitution sensitivity check runs and reports |
| **E4** | LaTeX table generation on 3-arm data | tables emit with three arms and **compile** |
| **E5** | Figure generation: forest plot with CPDT diamonds, critical-difference diagram over 3 arms | figures produced, axes labelled, no silent 2-arm fallback |
| **E6** | **Cell-count reconciliation is enforced by the analyzer.** Delete one run from a copy of the smoke root | the analyzer reports 419/420 and **names the missing cell**. Silent tolerance of missing cells is the C1 defect and must be made impossible |
| **E7** | The analyzer refuses to mix campaign roots: point it at `c2_smoke/` and `wl_subtree_unified/` together | it errors, or it labels provenance per row. It must not silently pool |

### 4.6 Stage F — go/no-go sign-off

A single meeting/commit that records:

1. §10.2 fully filled: every check A1–E7 with its evidence artefact, date and result.
2. **Explicit acknowledgement of the T01 AC-6 finding** (§0.3) — that `S` cannot move
   — and the §2 statement of what C2 actually buys. The earlier Wave-1 HOLD is
   superseded by C2 only if §2 is accepted as written.
3. The **allocation answer** (P5): the concurrency the account can actually sustain,
   and therefore the projected completion date against the 2026-09-10 freeze.
4. The production `--mem` and `--time` per `(method, arm)`, derived from C1.11 and
   D1.2 measurements, not from history.
5. The node-constraint decision (B6), stated with its evidence.
6. Signed by Mario. **No agent submits C2.**

---

## 5. Protocol invariants — these must not drift between arms

Violating any of these silently voids the paired design. They are checked in §4 and
must be re-checked in the production data during analysis.

### 5.1 One commit, one build
All 8,400 runs execute the `campaign/c2` tag and the same native build hash.
Recorded per run (A7), asserted during analysis.

### 5.2 Identical data per `(problem, seed)` across all arms and methods
Enforced by the `data_fingerprint` (P3, C4).

### 5.3 Identical operator set per `(method, problem)` across arms
Enforced by A4b.

### 5.4 No early stopping
Considered 2026-07-27 and **rejected**. Recorded here so it is not re-proposed.

Wall-clock `T` is a **reported** quantity: it produces `S`, the overhead percentages
and the cost column of Table 2. A stop rule firing at different times on different
arms of the same `(problem, seed)` — and IsalSR saturating earlier is precisely our
hypothesis — would make `S` a measurement of the stop rule rather than of the
method. ρ is equally exposed: `ρ = evaluations / unique canonical strings`, and
truncating a run truncates both terms non-proportionally.

**Every arm runs the full 43,200 s budget.** The protocol's own convergence criterion
(exact solution recovery, in `evolve_until_convergence`) continues to terminate runs
as it always did; that is the protocol, not early stopping.

### 5.5 Completeness discipline — the anti-1,465 rule
Every run either produces a valid `run_log.json` **or** a ledger row naming its
cause. There is no third state.

If the campaign must be truncated for capacity, **drop whole `(method, problem,
seed)` triples across all three arms** — never individual runs. A partially
completed triple is worse than a missing one: it silently unbalances the paired
test. Any truncation is recorded in §10.1 and disclosed in the paper.

---

## 6. Statistical treatment

### 6.1 Three arms changes the machinery
Pairwise CPDT (`isalsr` vs `baseline`, `hash` vs `baseline`, `isalsr` vs `hash`)
with **Holm correction across the three contrasts**, plus Friedman/Nemenyi over the
three arms per method for the critical-difference diagram. **Do not silently reuse
the two-arm machinery** (T04 §5.3). Verified at A8/E2.

### 6.2 CPDT remains primary
CPDT treats each problem as one paired observation: `δᵢ = mean_seeds(m^A) −
mean_seeds(m^B)` over `N` problems, Shapiro-Wilk → paired t or Wilcoxon. It is the
primary significance metric for R² and the reduction factor. Per-problem
Holm-corrected tests are supplementary detail.

**N moves from 50 to ≈70.** Report CPDT at **both** `N = 50` and `N ≈ 70`, per method,
per metric (T05 AC-6). If the extension weakens the result, report that — it is the
honest outcome and far cheaper than being caught. The selection rule for D2 is
pre-registered and outcome-blind (T05 §5); cite the commit hash.

### 6.3 The 20-seed decision — its cost, and the disclosure obligation

C1 used 30 seeds. C2 uses 20 (§0.4a).

- **CPDT is essentially unaffected.** It pools over problems, not seeds. Going from
  50 to ≈70 problems gains far more than dropping 30→20 seeds loses; the per-problem
  mean `δᵢ` merely gains standard error `∝ 1/√20` instead of `1/√30`, a factor 1.22.
- **Per-problem paired tests lose power.** A Wilcoxon signed-rank at `S = 20` has a
  minimum attainable two-sided p of `2/2²⁰ = 2⁻¹⁹ ≈ 1.9 × 10⁻⁶`, versus
  `2⁻²⁹ ≈ 1.9 × 10⁻⁹` at `S = 30`. At fixed effect size the non-centrality parameter
  scales as `√S`, so it shrinks by `√(30/20) = 1.22`. These are the *supplementary*
  tables, which is why this is acceptable — but it must be stated, not hidden.
- **Reviewer exposure, named.** R1 explicitly endorsed the protocol: *"50 problems,
  30 seeds, Demsar-style paired inference."* Reducing seeds while a reviewer praised
  the seed count is visible and will be noticed. **The response letter must state the
  change, the reason (a three-arm campaign at 30 seeds is 151,200 core-hours) and the
  power arithmetic above, in the same paragraph that announces `N = 70`.** Trading
  seeds for problems strengthens the primary metric and weakens a supplementary one;
  say so in exactly those terms.
- **Mitigation, free:** seeds 1…20 must be the *same integers* C1 used (A5), so the
  continuity table can restrict C1 to those 20 seeds and compare like-for-like.
- **Restoration path:** if spillover capacity exists, topping up D1 to 30 seeds is
  `3 × 2 × 50 × 10 = 3,000` runs (36,000 core-hours) and is purely additive under
  the resume logic. It ranks **above** Gray in the spillover order (§8.3).

### 6.4 NaN policy
Per T08: pairwise deletion with the **true `N` reported per metric**, plus a stated
sensitivity check under the conservative substitution (a NaN IsalSR mean treated as a
failure). Both are reported. NaN can never be marked as the better value. Verified
at A9/E3.

---

## 7. Comparison against the submitted campaign (C1)

C1 is **reference only**. No C1 number enters a C2 table. Its two uses:

1. **The continuity table** (T02 §5.6, response-letter appendix). A per-axis mapping
   from the submitted Python/old-alphabet numbers to C2, at method granularity and
   for the headline per-problem rows the reviewers named. It must let a reviewer
   confirm that **R² is statistically unchanged** (the representation did not change
   the host's search) while **ρ, `k` and the cost axis moved** (the alphabet and the
   engine did). Restrict C1 to seeds 1…20 for the comparison (§6.3).
   If R² *did* move materially, that is a finding: investigate before writing it up.
2. **Pre-flight sanity** (C5, D1.6) — catching a broken pipeline by checking C2's
   numbers against a known-good reference before spending 100,800 core-hours.

**Reference path:**
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/model_validation/real_benchmarks/wl_subtree_unified/`

Known C1 headline values, for the sanity comparisons:

| Quantity | UDFS | Bingo |
|---|---|---|
| ρ | 1.56 ± 0.24 | 1.83 ± 0.09 |
| CPDT R² test, one-sided p (N = 42) | 0.00018 | 0.0013 |
| CPDT R² test Cohen's d | 0.303 | 0.034 |
| Canonicalisation overhead | 0.6 % | 51.0 % (39.2 % under the later measurement) |
| Seed-problem cells | 1,500 | **1,465** (35 unexplained) |

---

## 8. Budget

### 8.1 Committed

| Block | Arms × methods × problems × seeds | Runs | Core-hours |
|---|---|---|---|
| Pre-flight Stage B | micro-jobs | ~30 | ≈2 |
| Pre-flight Stage C | 3 × 2 × 70 × 1, 900 s | 420 | ≈105 |
| Pre-flight Stage D | 12 × 43,200 s | 12 | 144 |
| **Pre-flight total** | | | **≈250** |
| **C2 campaign** | 3 × 2 × 70 × 20 | **8,400** | **100,800** |

### 8.2 The concurrency arithmetic — answer this before T04/T05 finish (P5)

T04 and T05 both target 2026-08-17. A gated launch (§0.4b) therefore lands ≈2026-08-20
after pre-flight. Analysis, tables, figures and the response letter need at least a
week, so the campaign must **complete** by ≈2026-09-03.

```
    2026-08-20 → 2026-09-03  =  14 days  =  336 h
    100,800 core-hours / 336 h  ≈  300 concurrent cores, with zero queue loss.
```

Relaxing completion to the freeze itself (2026-09-10, 504 h) still needs **≈200
cores**. At 100 sustainable cores C2 takes 42 days and **misses the freeze
regardless of how correct the code is.**

**This is the single largest risk in the plan and it is not a code risk.** Confirm
the group allocation and QOS limits **now**, not at launch.

### 8.3 Trade order, if capacity is short — in this order and no other

1. **Accelerate T04/T05** to land by ≈2026-08-10. Every day earlier is ≈7,200
   core-hours of headroom. This is the cheapest lever and it costs no science.
2. **D2 → Strogatz only** (14 problems instead of ≈20): 8,400 → 7,680 runs,
   −8,640 core-hours. Keeps the "SRBench ground-truth track is now covered" claim;
   costs the Feynman-remainder half of the R3.1 answer.
3. **Drop the `hash` arm to Mode 1 only.** −33,600 core-hours. Mode 1 (D3) already
   answers R1.4's literal question (`ρ_exact` vs `ρ_iso`); the live arm makes it "a
   full comparison". **Only if D3 showed `ρ_exact ≈ 1`** — otherwise this trades away
   the answer to the heaviest comment in the round.
4. **Seeds 20 → 15.** −25,200 core-hours. Costs supplementary-table power; §6.3's
   disclosure paragraph gets worse.
5. **Never trade away the `isalsr` arm on D1.** It is the corrected-alphabet
   headline and the whole reason C2 exists.

### 8.4 Spillover, in priority order — reserves nothing

| Priority | Block | Runs | Core-hours |
|---|---|---|---|
| 1 | **Restore D1 to 30 seeds** (all three arms) | 3,000 | 36,000 |
| 2 | Restore D2 to 30 seeds | 1,200 | 14,400 |
| 3 | **Gray ablation (T03)**, 2 methods × 70 × 20 | 2,800 | 33,600 |

**Gray go/no-go: 2026-08-31.** A 12 h campaign launched after that cannot finish, be
analysed and reach the freeze. If the date passes, T03 ships as design +
implementation + theory with the ablation declared as characterised future work.
That is an acceptable outcome, not a failure.

---

## 9. Superseded decisions — recorded so they are not re-litigated

### 9.1 The Wave 1/2/3/4 sequencing (2026-07-27 → superseded 2026-07-31)
C2 is a single gated launch (§0.4b). The old waves map as: Wave 1 + Wave 2 → the
`isalsr` and `baseline` arms of C2; Wave 3 → the `hash` arm of C2; Wave 4 → spillover
(§8.4, unchanged).

### 9.2 "The `baseline` arm is not re-run on S50" (2026-07-27 → superseded 2026-07-31)
The old argument was sound on its own terms — the baseline never invokes the adapter,
so neither the C++ port nor the T16 alphabet correction can touch it — but it bought
36,000 core-hours at the cost of a cross-campaign, cross-hardware, five-month-apart
comparison for `S`. §0.4 spends that money to remove the confound outright.

**What this deletes:** the D1–D3 confound-characterisation checks and the
`--constraint`-pinning *mitigation*. **What survives:** the node census itself, as
pre-flight B5/B6, because heterogeneity *within* C2 across six separately-scheduled
arrays is a live risk even in a single campaign.

### 9.3 The Wave-1 HOLD (2026-07-30 → resolved by §2)
Wave 1 was held because AC-6 showed the C++ port cannot move `S`, making 36,000
core-hours for the overhead figure alone questionable. C2 changes the question: the
spend now buys the three-arm comparison, the corrected alphabet, the confound
removal, the D2 coverage and a defensible `N` (§2). **The AC-6 finding itself is
unchanged and must not be re-derived optimistically.**

---

## 10. Agent dispatch order

**One agent per ticket.** Each agent owns its ticket end to end, via the
`review-ticket` skill, and writes its own §7 work log. Do not fragment a ticket across
agents and do not merge two tickets into one agent.

The only thing that needs deciding is **order**, and it is set by two facts: T04 and T05
both gate the launch (§0.4b), and T02 cannot start its pre-flight until they exist.

### 10.0 Before any agent — check P5, the allocation

§8.2: even at 20 seeds C2 needs ≈200–300 concurrent cores. At 100 sustainable cores it
takes 42 days and misses the freeze *regardless of how correct the code is*. This is the
only item that can invalidate the whole plan, it costs an email, and every agent-day
spent before it is answered is spent at risk. **Not an agent task.**

### 10.1 Round 1 — three agents, in parallel

| Agent | Ticket | Why it is in this round | Files it owns |
|---|---|---|---|
| **T04** | Naive-hash comparator | **Launch gate.** The longest single piece of new code in the revision: three serialisations + soundness on the 14,841-DAG corpus, a `hash` runner for both hosts, stream persistence (P1), cost fields (P2). Start it first because it finishes last | new hash module, `experiments/models/{bingo,udfs}/`, `tests/unit/` |
| **T05** | Benchmark extension | **Launch gate.** Independent of T04 — different files entirely. Its first half (the 120-equation criterion-(ii) classification and the **pre-registered, committed** selection rule) is docs-only and must precede its own implementation half | `docs/md_files/changes/`, `benchmarks/datasets/`, `experiments/configs/`, `slurm/` |
| **T01** | C++ port, AC-5 only | Small and nearly done: submit the already-written `slurm/t01_close/` benchmark. Closes the last open AC on the engine and doubles as the **first live rehearsal of SP-1…SP-6** — better to find a broken engine probe here than in T02's Stage B | `slurm/t01_close/`, T01 §7 |

Disjoint file lanes, safe concurrently. **This round is the critical path**: every day it
slips is ≈7,200 core-hours of headroom (§8.2).

**One shared dependency to resolve in the first hour, not later:** both T04 and T05 write
through the extended `RunLog`/MANIFEST schema (checks **A6, A7, P3, P4** — `representation`
accepting `"hash"`, the provenance fields, `data_fingerprint`, the status-ledger writer).
Whichever agent starts first lands that schema change; the other rebases onto it.
Left implicit, both agents retrofit their runners later and the ledger arrives too late
to be uniform. **Say this explicitly in both agents' briefs.**

### 10.2 Round 2 — one agent

| Agent | Ticket | Scope |
|---|---|---|
| **T08** | NaN and paired-test integrity | The code half: NaN can never be marked better (`aggregation.py`, regression test); explicit, tested NaN policy with the true `N` per metric (`statistical_tests.py`); and the three-arm statistics (**A8**: CPDT pairwise with Holm over *three* contrasts, Friedman/Nemenyi over three arms) |

Can overlap the tail of Round 1 — `experiments/models/analyzer/` is a lane neither T04
nor T05 touches. It must land before T02 reaches Stage E, not before Stage A.

### 10.3 Round 3 — one agent, after T04 and T05 have landed

| Agent | Ticket | Scope |
|---|---|---|
| **T02** | Pre-flight and campaign | Stages **A → B → C → D → E**, then hand to Mario for Stage F. Owns the `campaign/c2` tag, the 8 micro-jobs (incl. **B9**, inherited from T06), the 420-task smoke, the 12-task certification with the detailed trace, and the analysis dry-run |

**T02 is days of work, not weeks.** Stages A and B are a day; Stage C is one 420-task
array (≈105 core-hours, wall-clock hours not days) plus its analysis; Stage D is 12 tasks
at 12 h, so one overnight; Stage E is a day. **If T02 is taking weeks, something is
wrong** — the likely cause is a stage failing repeatedly, which is a finding to escalate,
not a schedule to absorb. The only thing in this plan that legitimately takes weeks is
C2's own execution, and no agent waits on that.

**Interleave:** T02's Stage D3 is T04's Mode 1 replay. Hand the trace stream back to the
T04 agent (via `SendMessage`, so it keeps its context) rather than making T02 learn T04's
code. Its result is read **before** launch — that is the whole point of running Mode 1
first (§0.4d).

### 10.4 Do not spawn

- **T06** — closed. Its only obligation is one threshold Mario states in 15 minutes
  (T06 AC-10); the probe itself is T02's check B9. Its `k`-stratification refresh waits
  for C2 results.
- **T03 (Gray)** — spillover, reserves nothing, go/no-go 2026-08-31 (§8.4).
- **T09–T14** — manuscript-side. They consume C2's output and cannot start usefully
  before it exists.

### 10.5 What every ticket agent must be told

Put these in the brief, every time — they are the things an agent will otherwise get
wrong at cost:

1. **`EXECUTION-PLAN.md` §4.0 SP-0**: you do **not** submit C2 or anything resembling it.
   Probes only: `max_time ≤ 1,800 s`, ≤ 60 tasks, **seed 0**, output to
   `~/execs/isalsr/t<NN>_*/`.
2. **SP-1…SP-6 before trusting any Picasso result**, reported as a six-row table in the
   work log. An entry without it is not evidence.
3. **Your SP-7** — the one falsifiable statement your ticket must establish (§4.0 table).
4. **Load the `picasso-sbatch` skill before writing or editing any SLURM script**, and
   work its *CPU array jobs* silent-failure checklist. The array-size and file-count
   support thresholds, `$LOCALSCRATCH`, and the FSCRATCH purge policy all bite at C2's
   scale.
5. **20 seeds, not 30** (§0.4a and the boxed note). Write every artefact that way from
   the start; do not leave a placeholder assuming 30.
6. **The alphabet is decomposed** — no `Sub`, no `Div`, `Pow` the only non-commutative
   operation — and `k` is ~22 % larger than any pre-T16 number you will find in the repo.

---

## 11. Ledgers

### 11.1 Decisions and anomalies during the campaign

| Date | Item | Decision / finding | Recorded by |
|---|---|---|---|
| | | | |

### 11.2 Pre-flight sign-off

| Stage | # | Check | Date | Result | Evidence artefact |
|---|---|---|---|---|---|
| A | A1 | Commit frozen (`campaign/c2`) | | | |
| A | A2 | pytest / ruff / mypy clean | | | |
| A | A3 | Backend parity, `.so` freshness | | | |
| A | A4 | Config equivalence across arms | | | |
| A | A4b | Operator-set policy decided | | | |
| A | A5 | Seed set declared (1…20, ⊂ C1) | | | |
| A | A6 | MANIFEST schema frozen + validator | | | |
| A | A7 | RunLog accepts three arms | | | |
| A | A8 | Analyzer three-arm readiness | | | |
| A | A9 | T08 code half landed | | | |
| A | A10 | Failure ledger implemented | | | |
| A | A11 | Hash-collision bound stated | | | |
| A | A12 | SLURM array/job limits | | | |
| A | A13 | Storage projection | | | |
| B | B1 | Environment probe | | | |
| B | B2 | C++ capability probe + negative control | | | |
| B | B3 | Alphabet gate on frozen commit | | | |
| B | B4 | Equivalence gate on a compute node | | | |
| B | B5 | Node-pool census | | | |
| B | B6 | Node-constraint decision | | | |
| B | B6b | AVX-512 portability of the C++ engine | | | |
| B | B7 | `sbatch --test-only`, all six arrays | | | |
| B | B8 | Resume / idempotency | | | |
| B | B9 | T06 counter re-verification (threshold from T06) | | | |
| C | C1.1–C1.15 | 420-task smoke, all criteria | | | |
| C | C2 | Failure ledger emitted | | | |
| C | C3 | Dedup-off equivalence control | | | |
| C | C4 | Cross-arm data identity | | | |
| C | C5 | Comparison against C1 | | | |
| D | D1.1–D1.8 | 12-task full-length certification | | | |
| D | D2 | Detailed single-problem trace | | | |
| D | D3 | T04 Mode 1 replay + soundness | | | |
| E | E1–E7 | Analysis dry-run on 3 arms | | | |
| F | — | Sign-off (Mario) | | | |

### 11.3 Launch ledger

One row per array. Fill at submission.

| Array | Method | Arm | Suite | Tasks | Job ID | Submitted | Completed | Failed | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | UDFS | baseline | D1∪D2 | 1,400 | | | | | |
| 2 | UDFS | hash | D1∪D2 | 1,400 | | | | | |
| 3 | UDFS | isalsr | D1∪D2 | 1,400 | | | | | |
| 4 | Bingo | baseline | D1∪D2 | 1,400 | | | | | |
| 5 | Bingo | hash | D1∪D2 | 1,400 | | | | | |
| 6 | Bingo | isalsr | D1∪D2 | 1,400 | | | | | |
