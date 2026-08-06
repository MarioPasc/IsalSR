# Execution plan — Campaign C2, the unified three-arm re-launch

Single source of truth for **what gets launched, in what order, and under what
gate**. Referenced by T02, T03, T04, T05, T06, T08. If a ticket and this file
disagree about a launch, **this file wins**; update the ticket.

| Field | Value |
|---|---|
| Campaign id | **C2** (the submitted campaign is C1 = `wl_subtree_unified`) |
| Shape | `{baseline, hash, isalsr} × {UDFS, Bingo} × (D1 ∪ D2) × 30 seeds` (was 20; §0.4a superseded 2026-08-05) |
| Runs | **8,400** (6 SLURM arrays × 1,400 tasks) |
| Budget per run | `max_time = 43,200 s` (12 h), 1 core |
| Core-hours | **100,800** committed |
| Launch model | **one gated launch, all six arrays** — nothing submits until every blocker and every pre-flight gate in §4 has passed |
| Status | **NOT SUBMITTED.** Pre-flight A–E complete: A–C signed, **Stage D GO** (§4.4, 13/13 cells, 8/8 criteria) and **Stage E GO** (§4.5, 7/7 checks, 2026-08-05, `stage_e_certification.json`). Remaining before launch: D3, one clean Stage C wave (**v5**) on the final config, HOME quota, Stage F sign-off, `campaign/c2` tag. ⚠ **Stage E must be re-run on v5** and must then pass **without** `--allow-mixed-provenance`; v4 needed it because 161 of its cells recorded `a455d6c-dirty` |
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
    { baseline , hash , isalsr }  ×  { UDFS , Bingo }  ×  ( D1 ∪ D2 )  ×  30 seeds
```

Rationale, in Mario's words: *"we will get clean and comparable numbers, and we will
make sure that everything is absolutely comparable."* Every number in the revised
paper then comes from one commit, one build, one node pool, one alphabet, one
protocol, one campaign root.

Four sub-decisions taken with it:

| # | Decision | Consequence |
|---|---|---|
| a | ~~**20 seeds**, not 30~~ **SUPERSEDED 2026-08-05: 30 seeds.** The 151,200 figure assumed 12 h for both hosts; Bingo measures ~5.15 h, so 30 seeds is ≈108,000 core-h and ≈63 h of wall clock. §6.3's disclosure obligation is withdrawn |
| b | **One gated launch, all six arrays** | nothing submits until T04 *and* T05 have landed. Maximum comparability, at the price of inheriting the later of the two slip dates |
| c | **Gray (T03) stays spillover** | not in the committed budget; go/no-go 2026-08-31. §8.3 |
| d | **T04 Mode 1 replay runs on the pre-flight certification streams** | the ρ_exact/ρ_iso number arrives *before* the 100,800 core-hours are committed, which is the entire point of running Mode 1 first. §4.5 (D3) |

> ### ✅ SUPERSEDED 2026-08-05 (Mario): **the campaign launches at 30 seeds.**
>
> This box argued from one premise — *"§8.2's arithmetic is unforgiving: even at 20
> seeds we need ≈200–300 concurrent cores"* — and that premise is now measured and
> false. Two findings reversed it (§11.1, 2026-08-05):
>
> - **Cost.** §8.1 assumed 12 h for both hosts. UDFS does saturate 12.00 h, but
>   Bingo stops on `max_evals` at ~5.15 h, so 30 seeds is **≈108,000 core-hours**,
>   not the 151,200 sub-decision (a) priced.
> - **Throughput.** Apportioning the array throttle by work instead of uniformly
>   (1.9×, free) puts 12,600 runs at **≈63 h**, or ≈54 h after a Day-1 rebalance —
>   against the 2026-09-03 target, not the freeze.
>
> **The priority ordering below is not overturned; it is satisfied.** "The first
> obligation is that results land" argued against a 30-seed campaign that is 80 %
> done on 2026-09-08. At 63 h that campaign is not 80 % done, it is finished with
> weeks to spare, so the ordering now *chooses* 30 seeds rather than tolerating it.
>
> **What this deletes.** The four top-up conditions below are moot — seeds 21–30
> are in the campaign, not a spillover — and §8.4's priority-1 and priority-2
> spillover entries are absorbed. **§6.3's disclosure obligation about reduced
> supplementary-table power goes away entirely**, and C1's seed count is restored,
> which also makes §7's C1-vs-C2 continuity comparison like-for-like.
>
> **The one thing that did not get easier: completeness.** §5.5 binds harder at
> 12,600 cells than at 8,400, and condition 4 below states the reason correctly —
> reporting 30 seeds for some problems and 20 for others makes `S`
> problem-dependent. That is now the *campaign's* obligation, not a top-up's.
>
> ⚠ **Everything the paper says must now say 30 seeds**, and the inverse of the
> old instruction applies: do not leave a placeholder that assumes 20.
>
> All 14 campaign configs carry `n_seeds: 30`, locked by
> `test_budget_uniformity.py::test_n_seeds_is_uniform_across_every_campaign_config`
> and by `test_c2_slot_plan.py::test_campaign_seed_count_is_declared_by_every_config`.
> `bingo_hard_trace.yaml` deliberately stays at 20 — it is the Stage D trace config,
> not a campaign config, and its seed is passed explicitly (`c2_stage_d/worker.sh:246`).
>
> ---
>
> **The original decision follows, retained because its reasoning was right for
> what was known on 2026-07-31.**
>
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
**Methods**: UDFS, Bingo. **Seeds**: 1…30 (§0.4a superseded 2026-08-05).

**Array topology — corrected 2026-08-03 (§11.1, T17 §2.2).** One SLURM array task
= one `(problem, seed)` pair for a fixed `(method, arm)`. The earlier statement
"6 arrays × 1,400 tasks" did not match the configs: **every config file declares
exactly one benchmark suite**, and the launcher maps one config to one array. The
real shape is

```
7 suites × 3 arms × 2 methods = 42 arrays          (8,400 runs either way)
```

Per-array size is `suite_size × n_seeds`: nguyen 240, feynman 200, hard 200,
cherrypicked 200, roundoff 160, feynman_remainder 120, strogatz 280 — all far
inside `MaxArraySize = 4096` (**A12**, measured), so **no chunking is needed**
and the earlier chunking caveat is withdrawn. Stage C uses the identical
topology at 3 seeds (42 arrays, 18–42 tasks each, 1,260 total): certifying a
topology you will not launch certifies nothing.

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
| **T04** | The three fixed-order serialisations, the `hash` runner for both methods, the hash-arm counters, and the stream persistence Mode 1 needs. **AC-1 soundness proven on the 14,841-DAG corpus** | **DONE except the C2 arm itself (2026-08-02).** Probe complete: 28/28 cells, **336/336 SP checks**, SP-3 negative control genuine, **AC-10 PASS** (shadow RSS −0.24 %/−0.14 %, no OOM). Provisional: **UDFS ρ_hash = 1.0000 — zero duplicates — vs IsalSR 1.396–2.243 ⇒ 100 % needs 1-WL; Bingo 4.3 %** (the AC-8 concession). Formal spec + proofs + all numbers: `T04-appendix/naive_hash_baseline.md`. **C2-READY** — readiness assessed and documented in `T04-appendix/c2_launch_readiness.md`; nothing left implies further implementation. `shadow_distinct_host_native` shipped untested in `a24d73c` (after probe commit `a4206b8`) and was **closed 2026-08-02**: 6 tests, both hosts, incl. a `n_shadow_failures` tripwire; unit suite 6,146 passed, 0 failed. ⚠ **C2 confirmation step (not a blocker):** enable that counter in the `slurm/t05_probe/` submission so sketch + extractors are seen running together on Picasso once, and assert `n_shadow_failures == 0` with a non-null `shadow_distinct_host_native` at Stage C — it is the only same-stream measure free of the adapter-renumbering bias, which the probe showed inflates the naive baseline from 0 % to 94.6 % on UDFS. Open ACs (3, 4, 5, and 2's dispersion/k-strata) are **blocked on C2**; AC-7 on `/review-answer` |
| **T05** | D2 problem definitions, sympy ground truth, unit tests, configs, and the **pre-registered, committed** selection rule (T05 AC-3) | **DONE except the Picasso probe (2026-08-02).** `D2 = 20`: all 14 ODE-Strogatz + 6 AI Feynman drawn by the pre-registered rule. Registry resolves **70/70**; `solution_recovered` computable on **70/70**. Selection rule committed **before** the draw — rule `d95e7d9`, draw `0e4a573`. Configs at 30 seeds (were 20; §0.4a superseded 2026-08-05), hard-tier operator set. Local smoke green on **both hosts**. Probe harness `slurm/t05_probe/` written, **not submitted** (T04's probe in flight). ⚠ Two knock-ons for this plan: **C1.5 was failing on five *D1* problems** and is now fixed, and **five D1 definitions were corrected**, so §7's C1↔C2 continuity table must **exclude** `I.39.10, I.12.4, II.3.24, II.11.27, III.17.37` — see `docs/md_files/changes/feynman_definition_corrections.md` |
| **T06** | The **instrumentation half only** (§4.1 counters for the five fallback paths). Not the analysis, not the write-up | done — re-verify it survives the C2 code at **C1.9** |
| **T08** | The **root-cause half** (§5.1) plus any *runtime* fix it implies. The analyzer-side fixes (NaN-as-winner, NaN policy) are needed for the pre-flight analysis dry-run **E3**, so in practice all of T08's code half must land | **CODE HALF DONE 2026-08-02.** Root cause established for both NaN cells (expression undefined on the *test* domain — `log` of a negative on Vlad-2's extrapolation grid, `exp` overflow on Korns-12; only test metrics NaN, both runs completed normally) and for **all 45** missing cells (**36 OOM + 9 post-search SymPy hangs, 0 unexplained**). Four defects fixed in `generate_tables.py` + a runtime scoring policy in `metrics.py`; 30 new tests, full suite **6,605 passed**. ⚠ **Two carries into this plan: the 256 GB memory decision (§3.3) and the amended C1.3.** Re-confirm at **A9** |
| **T02** | §5.3 MANIFEST schema, **frozen**, extended for three arms (**A6**) | to freeze |

**T03 (Gray), T07 (proofs), T09–T13 (manuscript) do not block.** T07 has one
coupling: T06's definition of a precondition violation must match T07's statement.
Agree the definition, then instrument.

### 3.3 Memory sizing — decided 2026-08-02 (Mario), from T08's forensics

> ✅ **SUPERSEDED 2026-08-05 by the measurement this section itself provides for.**
> Stage D's full-length `bingo_isalsr` cells peak at **`MaxRSS` 1.05–1.16 GB**, and
> the requirement is bounded *above* rather than extrapolated: `canonical_seen:
> set[int]` is the only unbounded container, a candidate cannot enter it without
> being scored, and Bingo stops on `max_evals = 100M` — so `n_unique ≤ 100M` **on
> any problem, Vladislavleva-4 included**, giving a worst case of **9.4 GB** at the
> measured 81.5 B/entry. **The request is now 32 GB** (`c2_slot_plan.MEM_GB`), 3.4×
> that ceiling. C1's ceiling was pymalloc arena fragmentation plus `set[str]`, both
> since removed. The reasoning below is retained because it is why 256 GB was
> right to assume *before* the measurement existed. See §11.1, 2026-08-05.

**Bingo–IsalSR requests 256 GB in C2, not 128 GB. Decided now, not deferred to a
measurement.**

The evidence, from T08's reconstruction of C1's SLURM logs: **36 of the 45 missing
cells are `OUT_OF_MEMORY`**, and 29 of the 31 IsalSR ones died at
`MaxRSS ≈ 127.7 GB` against a 128 GB request — a hard ceiling, hit repeatedly, on
Vladislavleva-4 (18 cells) and Korns-12 (9 cells). Campaign-wide, **326** `.err`
files carry an `oom_kill` tail; the orchestrator's resume logic re-ran and
recovered all but 36. C1's 1,465-vs-1,500 shortfall is what that ceiling looks
like after the retry loop has hidden most of it.

C2 is larger than C1 (≈70 problems, three arms), so the same ceiling would recur.

| | C1 | C2 |
|---|---|---|
| Bingo–IsalSR `--mem` | 128 GB | **256 GB** |
| Observed `MaxRSS` at failure | 127.7 GB | — |

**Consequences that must be carried, not discovered:**

- **Concurrency.** At 256 GB per task the pool narrows: `sd` (182 GB) cannot host
  one at all, `sr` (439 GB) hosts 1, `bc` (683 GB) 2, `bl` (1855 GB) 7. §8.2 needs
  ≈200–300 concurrent cores; re-derive the achieved concurrency from Stage C
  (§4.3) under this request **before** Stage F signs off, and interact this with
  the **B6** node-constraint decision — it is now partly made for us.
- **This does not replace C1.11/D1.2.** Both still run and still size the *other*
  five `(method, arm)` combinations from measurement. 256 GB is a floor for
  Bingo–IsalSR set from evidence, not a substitute for measuring; if D1.2 shows
  12 h `MaxRSS` comfortably under 128 GB **under the C++ dedup set**, the request
  may be revised *down* before launch, with the measurement recorded.
- **The failure ledger (P4) is what makes this checkable.** Under P4 an OOM leaves
  a status row instead of silence, so a recurrence is counted at Stage C rather
  than inferred from a cell shortfall in September.

**Where it lands.** C1's configs are historical and are **not** edited:
`slurm/{hard,cherrypicked,roundoff,models}_config.yaml` all carry
`mem_gb: 128` for the Bingo IsalSR group. When C2's configs are cut, that value
becomes `256` for Bingo–IsalSR only; the baseline and hash arms keep their
measured values.

### 3.2 Engineering checks that are nobody's ticket

| # | Check | If it fails |
|---|---|---|
| **P1** | **Per-candidate stream persistence.** T04 Mode 1 replays a DAG/canonical-hash stream. C1 persisted only aggregate counts, so replay of the submitted campaign was never possible. Decide the format and the sampling rate (full persistence is millions of entries per run) and land it before launch | Mode 1 can only ever replay C2 post-hoc, and the ρ_exact/ρ_iso go/no-go on the hash arm is lost |
| **P2** | **Cost fields survive the C++ port.** `T_canon` and `T_eval` per DAG feed T10's break-even analysis and R1.1's answer | Restore before launch |
| **P3** | ✅ **DONE 2026-08-03.** `experiments/models/provenance.py`: `data_fingerprint()` (SHA-256 over name+shape+IEEE-754 bytes of the four arrays, cast to float64 and made contiguous so the *sample* is certified, not the container) and `config_sha256()`. Both recorded on `RunMetadata`. C4 rehearsed locally: identical across 3/3 arms on both hosts | Without it, **C4** cannot prove the three arms saw identical data, and the paired design is unverifiable |
| **P4** | ✅ **DONE 2026-08-03.** `experiments/models/status_ledger.py`: **write-ahead** `status.json` per run (`terminal_status="started"` before the search, rewritten after) — the only design that survives the dominant C1 failure, an OOM `SIGKILL`, which no Python handler observes. Atomic `os.replace`; one file per seed dir, not a shared append, because 1,400 concurrent tasks interleave partial lines. `collect_status_ledger()` emits `status_ledger.csv` deterministically; `reconcile()` **names** missing/killed/failed cells (C1.15, E6). Orchestrator records and continues, then returns exit 1 if any cell failed | T08's 35-cell shortfall recurs at 8,400-run scale and is again unexplainable |
| **P5** | **Allocation sizing.** §8.2 — 100,800 core-hours against the freeze needs ≈200–300 concurrent cores. **Policy half MEASURED 2026-07-31 and it passes**: QOS `long_uma` allows `MaxWall = 7 days` and **`cpu = 9000` concurrent cores per user**, with `MaxJobsPU`/`MaxSubmitJobsPU` unset. 300 cores is 3.3 % of the entitlement | The binding constraint is therefore **contention, not policy** — which cannot be read from `sacctmgr` and must be measured. Time the 1,260-task smoke (Stage C) end to end and divide: that gives the *achieved* concurrency under real queue pressure, and it is the number §8.2 needs |
| **P6** | 🔴 **Quota headroom (see A13).** Re-read 2026-08-02: FSCRATCH **222.8k / 250.0k** files (was 248.4k on 07-31 — something was cleaned up), HOME **0.43 TB / 0.28 TB soft** with **4 days of grace left** (was 6). So: ≈27k files of FSCRATCH headroom against C2's ≈42,000, and a HOME grace clock that expires ≈2026-08-06 | Two separate failures. **HOME**: when grace expires, writes to HOME are blocked — that is days away and independent of C2. **FSCRATCH**: C2 hits the hard file quota mid-campaign and every running task keeps burning wallclock while all its writes fail. **Re-read live on the day** (`ssh picasso 'quota'`); these numbers move. Fix HOME this week; fix FSCRATCH before Stage C |

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
| Seeds | seed **0** only (never 1…30 — a probe output must never be mistakable for a campaign cell). **Widened from 1…20 on 2026-08-05**: the campaign now spans 1…30, so a probe at seed 25 would collide where it previously would not. Stage C's 0/101/102 remain disjoint |
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
| **A4b** | **Operator-set policy — DECIDED 2026-08-03: uniform per method, across every problem.** C1 gave Bingo *different* sets per tier (`nguyen`/`feynman` omit `sqrt` and `pow`; `hard`/`cherrypicked`/`roundoff` include them), which **confounds the operator set with the problem group** and makes any per-group difference in results unattributable. All seven `bingo_*.yaml` now carry `["+","-","*","/","sin","cos","exp","log","sqrt","pow"]` on D1 and D2 alike. UDFS takes **no** operator set from the YAML — `to_dag_regressor_kwargs()` never forwards the field and the search enumerates the vendored `NODE_ARITY` table — so its set is already uniform and not ours to change; the eight `udfs_*.yaml` lists are documentation only and now all record that fixed table. **This supersedes the earlier recommendation to freeze D1's per-tier sets.** Its cost is *not* compute (§0.4b re-runs the baseline on `D1 ∪ D2` regardless) but **continuity**: the 22 D1 problems whose Bingo set changed are no longer like-for-like against C1 and must be **excluded from §7's continuity table**, alongside the five T05 already excludes. Two invariants to check, not one: **(i)** for a fixed `(method, problem)` the operator set is identical across all three arms; **(ii)** every configured operator has an image in `𝓛` (Definition 3.2) — now *enforced* at config construction by `experiments/models/alphabet_guard.py` rather than assumed, because an operator outside `𝓛` is refused by the adapter, counted, and the candidate then evaluated **undeduplicated**, depressing ρ for a whole run with nothing in the reported numbers saying so | (i) holds for 70/70 problems; (ii) a deliberately bad config raises `AlphabetCoverageError`; the policy **and the continuity exclusion** are written into the MANIFEST | `c2_preflight/operator_sets.csv` + `pytest tests/unit/test_alphabet_guard.py` |
| **A5** | **Seed declaration.** Seeds 1…30 (§0.4a superseded 2026-08-05). C2 now spans C1's **full** seed set, so §7's continuity table needs no restriction to a matching subset — confirm the integers coincide rather than that a subset does. Seed 0 is reserved for smoke and must never appear in the campaign; 101/102 likewise (Stage C). Also assert all 14 configs declare `n_seeds: 30`, since `orchestrator.py:641` falls back to it whenever `--seeds` is absent | recorded in MANIFEST; `0 ∉ seeds`; `n_seeds == 30` on 14/14 | MANIFEST |
| **A6** | **MANIFEST schema frozen** and extended for C2: git commit + tag, native build hash, compiler + flags, config sha256 per `(method, suite)`, operator-set policy, arm list, seed list, alphabet version (`decomposed`), engine, node-constraint string, submission splits. Plus a validator that **fails** on any missing field | validator exits non-zero on a deliberately truncated MANIFEST | `experiments/models/manifest.py` + its test |
| **A7-BUG** | ✅ **CLOSED 2026-08-03.** `collect_hardware_info()` now returns `engine` (normalised to `native`/`python`, read from **actual dispatch** via `backends.engine()` so the B2 defect cannot recur), plus `build_hash`, `isa_level`, `avx512f`, `compiler`, `native_module_path/_mtime`, `cpu_model`, `hostname`, `git_describe`, `git_dirty`, the four SLURM ids and `mem_requested_gb`. Verified live on both hosts: `engine=native`, `build_hash=298fc1188bf1b051`, `isa_level=x86-64-v3`, `avx512f=0`. Ten pre-C2 fields retained verbatim so C1 artefacts still parse. Write-up: `docs/md_files/changes/c2_run_provenance.md` | a probe `run_log.json` shows `metadata.hardware.engine == "native"` — **done**, 3/3 arms × 2 hosts |
| **A7** | **RunLog schema accepts three arms.** `RunMetadata.representation` currently documents `"baseline" or "isalsr"`. Extend to `"hash"`; extend `hardware` to carry `cpu_model`, `hostname`, `slurm_job_id`, `slurm_array_task_id`, `mem_requested_gb`, `max_rss_gb`, `engine`, `git_commit`, `build_hash`, `config_sha256`, `data_fingerprint` | round-trip `to_dict`/`from_dict` test passes for all three arms | `tests/unit/test_schemas.py` |
| **A8** | **Analyzer three-arm readiness.** `analyze.py` accepts `--variants baseline,hash,isalsr`; pairwise CPDT with Holm across **three** contrasts and Friedman/Nemenyi over three arms are implemented and unit-tested on synthetic data | a synthetic case with a known answer reproduces it; the Holm correction divides by 3, not 2 | `tests/unit/test_three_arm_stats.py` |
| **A9** | **T08 code half landed.** NaN can never be marked better in `aggregation.py` (regression test); NaN policy in `statistical_tests.py` explicit, tested, and the reported `N` matches what the code does, per metric | both regression tests green | test output |
| **A10** | **Failure ledger implemented (P4).** A run that raises, OOMs or is time-killed still leaves a status record | kill a local run with `SIGKILL` mid-search; a status row still exists | `c2_preflight/ledger_demo.csv` |
| **A11** | **Hash-collision bound, stated not hoped.** Both the IsalSR dedup set and the T04 hash arm use 64-bit keys. Birthday bound `n²/2⁶⁵`: at `n = 10⁷` entries per run this is `2.7 × 10⁻⁶`; across 5,600 dedup-bearing runs the expected number of collisions is `≈1.5 × 10⁻²`. Record the arithmetic and the observed max entries per run from Stage C | a written bound in the MANIFEST notes, and a measured `max(n)` from Stage C that does not invalidate it | `c2_preflight/collision_bound.md` |
| **A12** | ~~SLURM limits~~ — **MEASURED 2026-07-31, PASSES.** `MaxArraySize = 4096` (so 1,400-task arrays are fine, **no chunking needed**), `MaxJobCount = 15000`, `MaxSubmitJobsPU` unset. Re-check on the frozen commit's submission day only if it has been weeks | 1,400 < 4,096 ✓ | `scontrol show config` |
| **A13** | 🔴 **Storage and file-count projection — this is now a live blocker, not a formality.** Measured 2026-07-31: FSCRATCH is at **248.4k files against a 250.0k soft quota** (400k hard) and HOME is **0.56 TB against a 0.28 TB soft quota with 6 days of grace left**. C2 writes ≥5 files per run × 8,400 runs ≈ **42,000 files**, plus ≈1,260 smoke runs (≈7,600 files). **The account does not currently have room for it.** Required: free FSCRATCH file headroom (archive or delete old campaigns), bring HOME back under quota before the grace expires, and — per the ≥15,000-file rule — either consolidate per-run output into one archive or **mail `soporte@scbi.uma.es` before the first array** | `quota` shows ≥60,000 files of FSCRATCH headroom and HOME under its soft quota, **before** Stage C | `c2_preflight/storage_projection.md` + a `quota` capture |

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
| **B6b-PRE** | ✅ **MEASURED 2026-08-02 by T04 — the build does not work out of the box on Picasso. Read before Stage B.** Three facts, all verified on the login node: **(1) Picasso's system compiler is `g++ (SUSE) 7.5.0`, which cannot compile the portability flag itself** — `cc1plus: error: bad value ('x86-64-v3') for '-march=' switch`, because `x86-64-v3` was introduced in GCC 11. `pip install -e . --force-reinstall --no-deps` **fails outright**. Fix: `module load gcc/11.1.0` (available; the module list tops out there) and export `CXX`/`CC` before building. **(2) The rebuilt `.so` imports and runs with the gcc module UNLOADED**, so no runtime `module load` is needed in workers — verified by importing in a shell without it. **(3) `build_info()` then reports `isa_level = x86-64-v3`, `avx512f = 0`** — AVX2-only, hence portable across `sd`/`sr`/`bc`/`bl`, which is the outcome B6b wants. **(4) The extension installed on Picasso was dated 2026-07-28 while the last C++ commit was 2026-07-30 — SP-2 FAILED.** Anyone launching C2 without an explicit rebuild would have run a **stale engine** against current Python and never seen an error | `module load gcc/11.1.0` present in the build step; `.so` mtime post-dates the last `src/isalsr/core/native/**` commit; `isa_level=x86-64-v3`, `avx512f=0` |
| **B6b** | 🔴 **AVX-512 portability of the C++ engine.** The login node is a Xeon Gold 6230R — an `sd` machine **with AVX-512** — but `sr` and `bl`, the bulk of the CPU cluster, **do not have it**. An extension built on the login node with `-march=native` emits AVX-512 and dies with **SIGILL** the moment it lands on `sr`/`bl`. Check the build flags in `pyproject.toml`/`CMakeLists.txt`, then run the same import probe on an `sd` node and on an `sr` node | the native module imports and canonicalises correctly on **both** an `avx512` node and a non-`avx512` node. If it does not, either rebuild with a portable baseline (`-march=x86-64-v2`) or pin `--constraint=avx512` — and note that pinning to `avx512` restricts the pool to `sd`+`bc`, which interacts with B6 |
| **B7** | **`sbatch --test-only` on all six arrays**, with the real `--array`, `--mem`, `--time` and `--constraint` | exit 0 on all six; the reported task count is exactly 1,400 per array (or the A12 chunking equivalent) |
| **B8** | **Resume and idempotency.** Run one task; re-submit it; then corrupt its `run_log.json` and re-submit again | second run **skips**; the corrupted run is **detected, deleted and re-run**. Both behaviours observed, not assumed |
| **B9** | **T06 counter re-verification** (owned by T02, threshold set by T06). One ≤30 min probe on **both hosts** on the frozen commit: the five fallback counters present and finite **at the production sampling rate**, and the instrumentation overhead **re-measured under the C++ engine and the decomposed alphabet**. Both changed underneath T06's original measurement — an overhead that was negligible as a *fraction* of a Python canonicaliser costing ~24× more per DAG may not be negligible now | counters live and finite on both hosts; overhead below the threshold T06 supplies. **A zero-everywhere ledger means the counters are dead, not that the rates are zero** — at N = 10,000 a short run samples almost nothing (the 2026-07-28 60 s smoke drew 5 DAGs), so design the probe so a live counter is distinguishable from a dead one. **If the overhead is now material**, the counters come out of C2 and T06 reopens for a separate subsampled characterisation run: a violation *rate* does not need the full campaign, a paired *timing* does |

### 4.3 Stage C — the 15-minute full-coverage smoke (**1,260 tasks, ≈315 core-hours**)

> **Amended 2026-08-03: three seeds, not one.** `compute_paired_stats` requires
> **three matched seeds** (`aggregation.py:207`), so at one seed none of the three
> paired contrasts is ever constructed and the smoke certifies every artefact
> *except* the ones the paired design rests on. Two seeds does not work either —
> the threshold is `< 3`. Three is also exactly `scipy.stats.shapiro`'s minimum,
> so the Shapiro-Wilk → t/Wilcoxon branch is exercised at its lower boundary.
> The stage's cost rises from ≈105 to ≈315 core-hours, **0.3 % of the campaign.**
> It buys the code path only: at `n = 3` the minimum attainable two-sided
> Wilcoxon `p` is 0.25, so no number from this stage can mean anything. Ticket:
> `T17-c2-submission-certification.md` §0.1.

This is the coverage test: **every problem × every arm × every method, at least
once, on real Picasso hardware.**

**Configuration.** `max_time = 900 s`, **seeds 0, 101, 102** (all deliberately outside the campaign
seed set **and** the 21…30 top-up range, so a smoke output can never contaminate
C2), all ≈70 problems × 3 arms × 2 methods × 3 seeds = **1,260 tasks**. Output
root: `c2_smoke/`. Resources: as production, so the memory profile measured here
is the profile that sizes production.

Every criterion below is **blocking**. A single violation stops the stage.

| # | Criterion | Threshold |
|---|---|---|
| **C1.1** | Every task exits 0 | 1,260 / 1,260 |
| **C1.2** | Every `run_log.json` exists, parses, and validates against the extended RunLog schema — **every field present, correct type**: `r2_train`, `r2_test`, `nrmse_train`, `nrmse_test`, `mse_test`, `solution_recovered`, `jaccard_index`, `model_complexity`; `wall_clock_total_s`, `wall_clock_search_only_s`, `canonicalization_precomputed_s`, `canonicalization_runtime_s`, `cache_hit_rate`, `cache_hits`, `cache_misses`, `estimated_time_saved_s`, `time_to_r2_099_s`, `time_to_r2_0999_s`, `evaluation_time_s`, `overhead_time_s`; `total_dags_explored`, `unique_canonical_dags`, `empirical_reduction_factor`, `max_internal_nodes_seen`, `theoretical_reduction_bound`, `redundancy_rate`; `symbolic_form`, `isalsr_string`, `canonical_string`, `n_nodes`, `n_edges`; plus `data_fingerprint`, `config_sha256` and the ten ledger fields | 1,260 / 1,260 |
| **C1.3** | **No NaN and no inf** in any regression metric. **Amended 2026-08-02 by T08:** this is now *enforced by the runtime*, not merely hoped for — a run whose expression is undefined on part of the evaluation set is scored `R² = 0` / `NRMSE = 1` and records `regression.n_nonfinite_test_predictions > 0`. A NaN metric is therefore once again an unambiguous defect signal. **Additional criterion:** report the distribution of `n_nonfinite_test_predictions` across all 1,260 tasks; a non-zero count is a legitimate scientific outcome (extrapolation failure), **not** a blocker, but it must be counted and disclosed | 1,260 / 1,260 NaN-free. Any NaN now means the guard itself is broken and blocks Stage D |
| **C1.4** | Every dataset loaded with the expected train/test shapes, asserted against the benchmark registry (Vlad-7 is 300/1200, Keijzer-6 is 50/120, Pagie-1 is 676/2500 — these are not typos, do not "fix" them) | 70 / 70 problems |
| **C1.5** | `solution_recovered` is **computable** for every problem, i.e. a `sympy_expression` ground truth exists — the known gap for the D2 additions (T05 AC-4) | 70 / 70 |
| **C1.6** | `isalsr` arm: `unique_canonical_dags > 0` and `empirical_reduction_factor ≥ 1` on **140/140**; `ρ > 1` on **≥ 90 %**. `ρ < 1` is arithmetically impossible and means a counter is broken. A ρ of exactly 1.0 everywhere means the dedup hook is dead and the entire arm is a null result | see cells |
| **C1.7** | **Hash-arm sanity:** `ρ_hash ≤ ρ_isalsr` for the same `(method, problem)`. This is **guaranteed** on identical input streams (a fixed-order hash is sound but incomplete) and **strongly expected** live. Report every violation | 140/140 expected; investigate if violations exceed 5 % |
| **C1.8** | `baseline` arm: dedup counters absent or zero and `canonicalization_runtime_s == 0`. Proves the baseline really is un-instrumented and is not silently paying canonicalisation cost | 140 / 140 |
| **C1.9-BUG** | ✅ **CLOSED 2026-08-03.** `SearchSpaceResults` now carries all five paths (`n_violations_pre/_post`, `n_canon_timeouts`, `n_conversion_failures`, `n_canon_raised`), the `n_atlas_hits` partition **and the denominators that make them rates** (`ledger_enabled`, `ledger_sample_rate`, `n_ledger_seen`, `n_ledger_sampled`), via `FallbackLedger.to_search_space_fields()`; per-k histograms go to a sibling `fallback_ledger.json`. `None` = arm never asked, `0` = asked and none occurred — the distinction SP-6 needs. 🔴 **Closing it exposed a launch-blocking defect: `ISALSR_LEDGER_ENABLED` defaults to `"0"` and NO worker/launcher/config sets it**, so C2 as configured would have recorded five zeros on all 8,400 runs. Now an auditable launch parameter (`--ledger`, `--ledger-sample-rate`) plus a loud warning; the keep/drop decision remains B9's and Mario's (T06 AC-10). Verified live, UDFS: `n_seen=476, violated_pre=476, violated_post=0` | a probe `run_log.json` shows the five rates present and finite — **done**; **the launcher must pass `--ledger`** |
| **C1.9** | **T06 fallback counters** present and finite on every `isalsr` task, covering all five paths (pre-normalisation violation, post-normalisation violation, 60 s timeout, conversion failure, canonicalisation raised). Report the five rates. This is check **B9** re-run at scale: B9 establishes the counters are alive and affordable on 2 probes, C1.9 confirms it holds across all 70 problems | 140/140 present; the five rates reported; overhead consistent with B9's measurement |
| **C1.10** | `trajectory.csv` non-empty; `timestamp_s` monotone non-decreasing; `best_r2` monotone non-decreasing; `n_dags_explored` monotone non-decreasing; `n_unique_canonical ≤ n_dags_explored` | 1,260 / 1,260 |
| **C1.11** | **Memory profile.** `MaxRSS` per task, tabulated by `(method, arm)`. Bingo+IsalSR historically needed 128 GB from heap fragmentation; the C++ dedup set should reduce that materially — **measure it, do not assume it**. Size production `--mem` at p99 + 50 % headroom. 🔴 **`sacct -X` returns an EMPTY `MaxRSS`** (verified on Picasso 2026-07-31): memory is accounted on the **`.batch` step**, not the allocation. Use `sacct -j <ID> -n -P -o JobID,MaxRSS \| awk -F'\|' '$1 ~ /\.batch$/'`. A profile built with `-X` comes back **silently blank** — a table of empty cells, no error | a populated table with 1,260 non-empty `MaxRSS` values, and a production `--mem` per `(method, arm)` derived from it |
| **C1.12** | **`max_time` is honoured.** Every task terminates at ≈900 s or earlier by convergence; **none** is killed by the SLURM wall limit. A SLURM kill without a `max_time` stop means `max_time` is not reaching `evolve_until_convergence` — the known Bingo defect in `CLAUDE.md` | 0 SLURM time-kills |
| **C1.13** | **Alphabet assertion on the real candidate stream** of every `isalsr` and `hash` task, not only in unit tests: 0 forbidden labels | 280 / 280 |
| **C1.14** | **Engine assertion**: every task records `engine == native` | 1,260 / 1,260 |
| **C1.15** | **Cell completeness reconciliation**: expected 1,260, observed 1,260, with a machine-checked comparison. This is the mechanism that must prevent a recurrence of C1's unexplained 35-cell shortfall | exact match, or every gap individually named |
| **C1.16** | **Paired-stats path constructed — the reason for three seeds.** All three contrasts emit a file per problem: `paired_stats.json`, `paired_stats_hash_vs_baseline.json`, `paired_stats_isalsr_vs_hash.json`; each parses and reports `n_seeds == 3`; the across-problem Holm correction runs and writes back. **Assert existence and validity, never a p-value** — at `n = 3` the minimum two-sided Wilcoxon `p` is 0.25 | 3 × 70 × 2 = 420 files |
| **C1.17** | `aggregate.csv` per `(method, problem, arm)`, exactly 3 rows each | 420 files |

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

**Pass:** `70 × 3` = **210** distinct fingerprints, each appearing exactly 6 times.
Also assert the 210 are mutually distinct: two seeds yielding identical data means
the seed is not reaching the generator and the design has no replication at all.

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
| Cell count | 1,260/1,260 present | the C1 shortfall mechanism is still present |
| Baseline R², D1 | within seed noise of C1's baseline at comparable budget | the baseline path changed when it should not have — the baseline never invokes the adapter |
| `POW` presence | Bingo: reachable on **every** problem, since the set is now uniform (A4b); UDFS: **never**, its table has no `pow` | operator-set drift (A4b) |

**Deliverable:** `c2_preflight/smoke_vs_C1.md` — a table with every anomaly either
explained or escalated. An unexplained anomaly blocks Stage D.

### 4.4 Stage D — full-length certification (**13 tasks, 156 core-hours**)

The 15-minute smoke proves nothing about a 12-hour run: memory growth, heap
fragmentation, dedup-set size, timeout paths and convergence behaviour are all
budget-dependent. Stage D is the successor to the old G7 single task and it is
**not optional and not parallelisable with the full launch**.

**Composition — 13 tasks at the full 43,200 s budget.** Amended 2026-08-04
(audit.md §7.3): the twelve certification cells run with the shadow sketches
**off**, and a thirteenth cell carries the D2 trace with them **on**. The split
exists because the sketches cost **17.6 %** of Bingo's wall clock, paid by the
`isalsr` arm alone: certifying under that penalty would confound D1.6's ρ
comparison against a C1 that never paid it — which is precisely the question
C5 §3.5 hands to D1.6. The trace cell runs at **seed 102**, because it repeats
cell 10's `(method, suite, problem, arm)` and at a shared seed the orchestrator
would write both runs into one directory.


| Group | Cells | Why these |
|---|---|---|
| Trace problem | **Pagie-1** × 3 arms × 2 methods = 6 | the structural-bottleneck problem where IsalSR is predicted to help (2026-04-19 analysis). Locked by Mario, audit.md §7 row 2 |
| NaN problems | Korns-12 + Vlad-2 × 3 arms × Bingo only = 6 | **the T08 AC-7 evidence.** These are the two cells that were NaN in the submission |
| D2 trace cell | Bingo × Pagie-1 × isalsr × seed 102 = 1 | shadow sketches **on**, `bingo_hard_trace.yaml`. **Not a certification cell** — excluded from D1.1–D1.8 by `STAGE_D_CERTIFICATION_CELLS`, so its ρ and R² never enter a table |

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

> ✅ **EXECUTED 2026-08-05 on `c2_smoke_v4/`: GO, 7/7, 182 s.** Harness
> `experiments/scripts/stage_e_certify.py`; runbook `slurm/c2_stage_e/RUNBOOK.md`;
> write-up `docs/md_files/changes/stage_e_design.md`. Per-check evidence in
> §11.2; the four defects it found are in §11.1 (2026-08-05).
>
> **It runs locally, not on Picasso** — every failure needed an analyzer fix and
> a deploy is a config edit (defect 10), `analyze.py`/`generate_tables`/the figure
> suite are the local step in September anyway, E3/E6/E7 mutate copies of the
> root, and E4 needs `pdflatex`. Cost is a soft probe: the analyzer is 49 s on
> 1,260 runs ⇒ **≈5.5 min at 8,400**, which must not be confused with the
> orchestrator's `--postprocess only` aggregation (1 h 35 m ⇒ ≈11 h).
>
> 🔴 **Owed: re-run on the v5 root.** E1 passed on v4 only with
> `--allow-mixed-provenance`, because the guard rediscovered v4's own dirty split
> (`a455d6c` ×1,099 vs `a455d6c-dirty` ×161). On v5 it must pass **without** that
> flag; if it does not, the wave was not clean and `campaign/c2` must not be cut.

The analysis pipeline has **never** been run on three arms. Discovering that in
September is the single most expensive failure mode left.

| # | Check | Pass criterion |
|---|---|---|
| **E1** | Full analyzer end-to-end on `c2_smoke/` (1,260 runs, 3 arms, 3 seeds — paired stats now exist, so E2's three contrasts have real input) | every artefact in T02 §5.5 produced without exception, with 3 arms present: `benchmark_summary_*`, `computational_overhead_*`, `cross_method_*`, `reduction_comparison_*`, `three_axis_*`, `cross_problem_dominance_*`, `global_summary.json` |
| **E2** | Three-arm statistics: pairwise CPDT (`isalsr` vs `baseline`, `hash` vs `baseline`, `isalsr` vs `hash`) with **Holm across three contrasts**, plus Friedman/Nemenyi over the three arms per method | outputs exist; `N` reported per metric; the synthetic test from A8 confirms the correction divides by 3 |
| **E3** | **NaN policy, adversarially tested.** Inject a synthetic NaN into a copy of the smoke root | (a) the NaN is **never** bold/marked better; (b) the reported `N` drops by exactly 1 for that metric; (c) the conservative-substitution sensitivity check runs and reports |
| **E4** | LaTeX table generation on 3-arm data | tables emit with three arms and **compile** |
| **E5** | Figure generation: forest plot with CPDT diamonds, critical-difference diagram over 3 arms | figures produced, axes labelled, no silent 2-arm fallback |
| **E6** | **Cell-count reconciliation is enforced by the analyzer.** Delete one run from a copy of the smoke root | the analyzer reports 1,259/1,260 and **names the missing cell**. Silent tolerance of missing cells is the C1 defect and must be made impossible |
| **E7** | The analyzer refuses to mix campaign roots: point it at `c2_smoke/` and `wl_subtree_unified/` together | it errors, or it labels provenance per row. It must not silently pool |

### 4.6 Stage F — go/no-go sign-off

A single meeting/commit that records:

1. §11.2 fully filled: every check A1–E7 with its evidence artefact, date and result.
2. **Explicit acknowledgement of the T01 AC-6 finding** (§0.3) — that `S` cannot move
   — and the §2 statement of what C2 actually buys. The earlier Wave-1 HOLD is
   superseded by C2 only if §2 is accepted as written.
3. The **achieved concurrency** measured from Stage C (§8.2), and therefore the
   projected completion date against the 2026-09-10 freeze — plus, if it falls short,
   which §8.3 trade is being taken, decided **now** rather than in week three.
4. The production `--mem` and `--time` per `(method, arm)`, derived from C1.11 and
   D1.2 measurements, not from history.
5. The node-constraint decision (B6) and the AVX-512 portability result (B6b), each
   stated with its evidence.
6. **Quota headroom** (A13 / P6) confirmed sufficient for ≈42,000 files, re-read live
   on the day, not from an earlier capture.
7. Signed by Mario. **No agent submits C2.**

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
Enforced by A4b, which since 2026-08-03 requires the stronger property that the
set is identical across *problems* too, one per method. A4b also carries the
containment check: every configured operator must be encodable in `𝓛`.

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
the two-arm machinery** (T04 §5.3). Verified at A8/E2 — **closed 2026-08-04**.

**Contrast policy, decided 2026-08-04 (Mario) — pre-registered directions only.**
Every contrast now persists its own effect size, p-value and `p_value_holm`:

| Contrast (δ = treatment − reference) | R²_train, R²_test | NRMSE_test | ρ, redundancy |
|---|---|---|---|
| baseline → isalsr | one-sided (as submitted) | one-sided | **descriptive, no p** |
| hash → isalsr | **two-sided** | **two-sided** | one-sided |
| baseline → hash | **two-sided** | **two-sided** | **descriptive, no p** |

Two-sided wherever no directional hypothesis was ever registered; one-sided for
ρ only on `hash → isalsr`, where the direction is guaranteed by construction (a
sound fixed-order hash cannot merge more than the canonical string).

**ρ is not tested against the baseline.** `bingo/runner.py:428` sets
`n_unique_canonical = total_evals`, so the baseline's ρ is **1.0 by
construction**; a paired test against it asks only "does any duplicate exist?"
and returns p ≈ 0 tautologically. The submitted table's ρ p-values
(2.7 × 10⁻²² UDFS) are withdrawn from the revision on those grounds; ρ against
the baseline is reported as mean ± std, and the inferential ρ claim moves to
`isalsr` vs `hash`, which is the comparison R1.4 actually asked for. Holm's
family per metric is therefore **3 for R²/NRMSE and 1 for ρ**.

### 6.1a Tie handling in the CPDT — corrected 2026-08-04
The CPDT called `scipy.stats.wilcoxon` with its default `zero_method="wilcox"`,
which **discards** tied problems — while the project's own W/T/L rationale
treats a tie as evidence of no difference. Dropping ties inflates significance
whenever the non-tied δ lean one way. Two changes, both conservative:

1. `|δ| ≤ 1e-6` is snapped to exactly 0 **before** the test, so the tested
   vector and the reported W/T/L are the same partition. Previously the
   threshold governed only the display and floating-point noise entered the
   test as signed evidence.
2. `zero_method="zsplit"` (Pratt 1959, JASA 54:655–667; Demšar 2006, JMLR
   7:1–30) splits the zeros' rank sum evenly between the positive and negative
   sums.

Recomputed on C1's archived δ (N = 50): **no significance claim flips at
α = 0.05**; Bingo's R² rows move from *** to ** (4.4 × 10⁻⁴ → 6.1 × 10⁻³).
**Any C1 CPDT p-value quoted in the revision must be the corrected one.**
Effect sizes stay on the raw δ — snapping is a decision rule, not an estimator.

### 6.2 CPDT remains primary
CPDT treats each problem as one paired observation: `δᵢ = mean_seeds(m^A) −
mean_seeds(m^B)` over `N` problems, Shapiro-Wilk → paired t or Wilcoxon. It is the
primary significance metric for R² and the reduction factor. Per-problem
Holm-corrected tests are supplementary detail.

**N moves from 50 to ≈70.** Report CPDT at **both** `N = 50` and `N ≈ 70`, per method,
per metric (T05 AC-6). If the extension weakens the result, report that — it is the
honest outcome and far cheaper than being caught. The selection rule for D2 is
pre-registered and outcome-blind (T05 §5); cite the commit hash.

### 6.3 ~~The 20-seed decision~~ — WITHDRAWN 2026-08-05, and it improves the letter

> ✅ **C2 runs 30 seeds (§0.4a, superseded).** Everything below described the cost
> of a reduction that is no longer being made. **The disclosure obligation is
> withdrawn: there is nothing to disclose.** Three consequences worth stating
> positively, because they are all improvements:
>
> - **The named reviewer exposure disappears.** R1 endorsed *"50 problems, 30
>   seeds, Demsar-style paired inference"*. The bullet below required the response
>   letter to explain why we had cut a seed count the reviewer praised. We now
>   deliver 30 seeds **and** N = 70 problems, so the letter says the protocol was
>   extended in both directions. **Do not write the disclosure paragraph.**
> - **The supplementary tables keep their power.** Wilcoxon at `S = 30` reaches a
>   minimum two-sided `p = 2⁻²⁹ ≈ 1.9 × 10⁻⁹` rather than `2⁻¹⁹ ≈ 1.9 × 10⁻⁶`, and
>   the 1.22× non-centrality loss below never happens.
> - **The continuity comparison gets easier, not harder.** The "mitigation" bullet
>   asked that seeds 1…20 be the same integers C1 used so §7 could restrict C1 to
>   a matching subset. At 30 seeds **no restriction is needed** — C2 spans C1's
>   full seed set. A5 is updated accordingly.
>
> The restoration-path bullet and §8.4's priority-1/2 spillover entries are
> absorbed into the campaign. Retained below for the record.

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
   engine did). **No seed restriction is needed** (§6.3, withdrawn 2026-08-05): C2 spans 1…30, the same set C1 used, so the comparison is like-for-like on seeds without subsetting.
   If R² *did* move materially, that is a finding: investigate before writing it up.
   **THREE exclusions, all mandatory, all because the compared object changed and
   not the method.**

   **(1) Six D1 problems whose definitions were corrected** — five by T05 (§3):
   `I.39.10, I.12.4, II.3.24, II.11.27, III.17.37`, plus **`I.34.27`, corrected
   2026-08-04**: it was implemented as `hbar * omega`, which dropped the `1/(2π)`
   of the AI Feynman definition and made it byte-for-byte identical to `I.12.1`.
   C1 therefore ran a *different function* under this name, so its C1 row is not
   comparable. (Note that `I.12.1` itself is **unchanged** and stays in the
   continuity table.) Applies to **both** methods.

   **(2) For Bingo only**, the 22 D1 problems C1 configured without `sqrt` and
   `pow`, whose operator set A4b made uniform: Bingo now searches a strictly
   larger primitive set there, so a C1↔C2 difference on those rows confounds the
   alphabet, the engine **and** the search space.

   **(3) For UDFS only — added 2026-08-05, and it corrects a sentence that stood
   here until F-20 was closed.** This paragraph used to read *"UDFS is unaffected
   by the second exclusion, its set having never been ours to set."* That was true
   of the **operator set** and is now false of the **search-space bound**: F-20
   found `udfs_feynman.yaml` at `n_calc_nodes: 7` against every other suite's 5,
   and it was levelled **down to 5**. `n_calc_nodes` caps the number of
   intermediate nodes UDFS may enumerate (`dag_search.py:594`), i.e. the size of
   representable expressions — so **the 10 `feynman` problems ran C1 with a
   strictly larger reachable space than they will run C2**, and their UDFS rows
   are not like-for-like. Direction of the effect is known and adverse-to-us:
   C2's UDFS may recover *fewer* Feynman expressions than C1 did, for a reason
   that has nothing to do with IsalSR. **Exclude the 10 `feynman` problems from
   the UDFS half of the continuity table**, and if a reviewer asks why UDFS looks
   weaker on Feynman than in the submission, this is the answer — give it before
   they ask.

   State all three exclusions in the letter's continuity appendix rather than
   dropping the rows silently.
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
| Pre-flight Stage C | 3 × 2 × 70 × 3, 900 s | 1,260 | ≈315 |
| Pre-flight Stage D | 12 × 43,200 s | 12 | 144 |
| **Pre-flight total** | | | **≈460** |
| **C2 campaign** | 3 × 2 × 70 × **30** | **12,600** | **≈108,000** |

> **Updated 2026-08-05.** Both columns moved, in opposite directions. Seeds went
> 20 → 30 (§0.4a superseded), but the 100,800 figure assumed **12 h for both
> hosts** — UDFS does saturate 12.00 h, Bingo stops on `max_evals` at ~5.15 h. So
> 12,600 runs cost **≈108,000** core-hours (UDFS 75,600 + Bingo ≈32,400), not the
> 151,200 that §0.4a's sub-decision (a) priced 30 seeds at. At 2,016 apportioned
> slots that is **≈63 h**, or ≈54 h after a Day-1 rebalance (§11.1, 2026-08-05).
> Stage D is 13 cells, not 12 (the D2 trace cell).

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

**The policy half is measured and it passes (2026-07-31).** `sacctmgr` reports that
`tic_163_uma`/`mpascual` hold `short`, `medium_uma` and `long_uma`; `long_uma` allows
`MaxWall = 7 days` and **`cpu = 9000` concurrent cores per user**, with `MaxJobsPU` and
`MaxSubmitJobsPU` unset. The cluster has 334 CPU nodes and `MaxJobCount = 15000`.
**300 cores is 3.3 % of the entitlement — we are nowhere near a policy ceiling.**

**The binding constraint is therefore contention, and `sacctmgr` cannot tell us that.**
It has to be measured:

> **Stage C is the measurement.** Time the 1,260-task smoke end to end and divide:
> `1,260 tasks × 0.25 h ÷ wall-clock hours = achieved concurrent cores` under real queue
> pressure, at exactly C2's `--mem`/`--constraint`/throttle. If the smoke sustains ≥300,
> C2 fits the 2026-09-03 target. If it sustains 100, invoke §8.3 **before** launching,
> not after discovering it in week three.

Record the achieved figure in §11.1. Set the array throttle (`%K`) from it — an
unthrottled 1,400-task array is antisocial and invites manual intervention from support.

> ### 🔴 Measure the throttle, not just the wall clock (added 2026-08-04)
>
> **The formula above measures `min(cluster contention, your own `%K` ceiling)`, and
> it does not tell you which one bound.** The first Stage C wave ran at `%8` — 42
> arrays × 8 = a **336-task ceiling we imposed** — and returned 245 cores. That was
> read as contention and very nearly triggered a §8.3 trade; it was 73 % fill of our
> own cap, against a `cpu = 9000` entitlement with thousands of cores idle.
>
> Re-running the identical wave at `%24` gave **476 cores** and a 39 min 44 s span,
> putting C2 at **8.8 days**. The rule that follows:
>
> **Never read an achieved-concurrency figure without stating the throttle it was
> measured under, and never invoke §8.3 until the figure has been re-measured at a
> materially higher `%K`.** Every trade on that list costs science; raising `%K`
> costs a shell variable.

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
| ~~1~~ | ~~Restore D1 to 30 seeds~~ — **ABSORBED into the campaign 2026-08-05** | — | — |
| ~~2~~ | ~~Restore D2 to 30 seeds~~ — **ABSORBED into the campaign 2026-08-05** | — | — |
| **1** | **Gray ablation (T03)**, 2 methods × 70 × 30 | 4,200 | ≈43,400 |

> Priorities 1 and 2 no longer exist: C2 launches at 30 seeds, so there is
> nothing to restore. Gray moves to the top of the spillover order and grows with
> the seed count (2 × 70 × 30 = 4,200 runs; ≈43,400 core-h at the measured
> per-host runtimes rather than a flat 12 h).

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

§8.2 (as written, now superseded — see §11.1 2026-08-05: the throttle allocation, not contention, was the binding constraint; 30 seeds lands at ≈63 h): even at 20 seeds C2 needs ≈200–300 concurrent cores. At 100 sustainable cores it
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
| **T02** | Pre-flight and campaign | Stages **A → B → C → D → E**, then hand to Mario for Stage F. Owns the `campaign/c2` tag, the 8 micro-jobs (incl. **B9**, inherited from T06), the 1,260-task smoke, the 12-task certification with the detailed trace, and the analysis dry-run |

**T02 is days of work, not weeks.** Stages A and B are a day; Stage C is one 1,260-task
wave (≈315 core-hours, wall-clock hours not days) plus its analysis; Stage D is 12 tasks
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
5. **30 seeds** (§0.4a, superseded 2026-08-05 — the boxed note's cost premise was measured false). Write every artefact that way from
   the start; do not leave a placeholder assuming 30.
6. **The alphabet is decomposed** — no `Sub`, no `Div`, `Pow` the only non-commutative
   operation — and `k` is ~22 % larger than any pre-T16 number you will find in the repo.

---

## 11. Ledgers

### 11.1 Decisions and anomalies during the campaign

| Date | Item | Decision / finding | Recorded by |
|---|---|---|---|
| 2026-08-06 | 🔴 **D3 found the k=0 completeness defect** | **LAUNCH BLOCKER, found before any campaign core-hours were committed — which is exactly what §0.4d says D3 is for.** The Mode 1 replay returned exit 2 on **37 IsalSR soundness violations**, all between DAGs with **zero internal nodes**. Cause: Σ_SR encodes only the instructions that *build* internal nodes and the m variables are pre-inserted (invariant 7), so **every k=0 DAG canonicalises to `""`** — for every m and every choice of output variable. `""` is the faithful encoding of the *initial state*; what it cannot do is name an output, because a `LabeledDAG` carries no output marker (for k≥1 the output is the unique sink; at k=0 there are m sinks). **Verified on the real Bingo adapter: `f(x)=X_0` and `f(x)=X_1` share the key `""`.** Both runners acted on it — `bingo/isalsr_runner.py:670` transferred the cached fitness, `udfs/isalsr_runner.py:480` skipped evaluation outright. **Pre-existing, not from T18, so C1 carries it too.** Fix is a *domain restriction*, not a patch: at k=0 the relabeling group is 0!=1, so there is no redundancy to collapse and ρ is undefined on an object with no structure. k=0 candidates are scored, never deduplicated, never cached, excluded from ρ; **completeness is claimed for k≥1**. Incidence **0.0593 %** of the Bingo Pagie-1 stream (71/119,795 sampled) and **4.6 %** on a short synthetic 2-var run. UDFS cannot reach k=0 (`n_calc_nodes = 5` on all 7 campaign configs) but the guard is symmetric, because an arm-specific dedup rule is the asymmetry a paired design cannot tolerate. `n_nonstructural` persisted per run. **D3 now exits 0**: hash soundness 0 unsound merges on all three fixed orders (the arm's veto passes), replay fidelity 0/119,795. Commit `b223a08` | Claude, `stage_d_mode1_replay` + adapter probe |
| 2026-08-06 | 🔴 **v5 ledger job died in 2 s** | **Stage C v5 was 1,260/1,260 COMPLETED with single provenance and still produced no `status_ledger.csv` and no verdict.** `aggregate_worker.sh` asserted `C2_CONFIG_LIST` at **file scope**, but the launcher does not export it to the dependent ledger job — correctly, since a full-root walk needs no config — so under `set -u` the `:?` form aborted **before the role was selected**. One of the 2026-08-05 aggregation-split changes T17-HANDOFF §3.3 flagged as never having executed on Picasso; v5 is what it was for. Each variable is now asserted inside the role that consumes it, the array role still failing loudly on an empty list. Re-run on v5's data: ledger in 25 s, **18/19 criteria PASS**, the one failure being C1.2 because v5 predates `n_nonstructural`. Commit `b786d20`, 4 regression tests verified non-vacuous against `HEAD~1` | Claude, job 1783830 |
| 2026-08-06 | 🔴 **ρ was a ratio over two populations** | **The k=0 guard's own accounting was wrong, and Stage C v5b passed 19/19 with ρ still inflated.** `n_total` is incremented when a candidate is first *seen*, several steps before the DAG exists to be classified, so the guard excluded k=0 from `n_unique` but left it in `n_total` — making `ρ = n_total / n_unique` a ratio over two different populations. **Measured on the full 1,260-run v5b wave, and the error flatters IsalSR:** bingo/hash **2.0045 → 1.7872 (+12.15 %)**, bingo/isalsr **2.1102 → 1.8534 (+13.86 %)**, udfs/* **unchanged** (cannot reach k=0 — the first independent confirmation of `n_calc_nodes = 5` on real data). **It also corrects the incidence figure**: the Stage D trace put k=0 at 0.0593 %, but that is a 1-in-100 subsample of one *hard* 2-variable problem; over the whole wave it is **12.2 %** of bingo/isalsr's stream and **18.2 %** on Nguyen-3, because easy 1-variable problems keep many tiny individuals alive. **A subsampled trace of one hard problem was not a safe basis for that estimate — the 0.0593 % quoted in `b223a08` is superseded.** `total_dags_explored` now counts structural candidates only; candidates seen stays recoverable as `total_dags_explored + n_nonstructural`. Verified end to end: a run reporting `n_total_dags = 5150` pre-guard now reports **4888 + 262 = 5150** exactly. Accounting moved into `record_nonstructural()` so it is unit-testable rather than two increments that must be kept in step. Commit `040beab`; wave **v5c** re-run on it | Claude, full-wave scan |
| 2026-08-06 | C1.7 moved 0/420 → 8/420 | ⚠ **Watch, not blocking.** v5b showed 8 matched triples with `ρ_hash > ρ_isalsr`, all Bingo, against v5's 0. C1.7 tolerates ≤5 % and observed 1.90 %, so it PASSES its stated criterion. **Causation is not established and must not be asserted**: Bingo is not seed-reproducible (the accepted C3 finding — three identical `baseline --seeds 0` runs gave 155,449 / 41,023 / 41,049 candidates), so the two arms are measured on *different* candidate streams and wave-to-wave variation is expected. The by-construction guarantee (§6.1: a sound fixed-order hash cannot merge more than the canonical string) holds **per stream**, and D3 — the controlled comparison, same inputs, zero search confound — confirms it: ρ_iso 1.0254 ≥ ρ_exact 1.0251. **Re-check on v5c and at 30 seeds**, since ρ against `hash` is one of the *inferential* contrasts | Claude, `stage_c_certification.json` |
| 2026-08-06 | Stage C v5, provenance | ✅ **The `--allow-mixed-provenance` obligation is dischargeable.** v5 ran 1,260/1,260 at a **single** `git_describe` (`115bf89`), `git_dirty: false`, **one** `build_hash` (`298fc118…`) — against v4's `a455d6c` ×1,099 / `a455d6c-dirty` ×161 split. Superseded by v5b on `b786d20`, which carries the k=0 guard and the ledger fix | Claude, run-log scan |
| 2026-07-31 | A12, SLURM limits | **PASS.** `MaxArraySize = 4096` ≥ 1,401 — 1,400-task arrays need no chunking. `MaxJobCount = 15000`, `MaxSubmitJobsPU` unset | Claude, `scontrol show config` |
| 2026-07-31 | P5, allocation policy | **PASS.** QOS `long_uma`: `MaxWall = 7 days`, **`cpu = 9000` per user**; `short`/`medium_uma` also held. 300 cores = 3.3 % of entitlement. **Binding constraint is contention, not policy** — measure it from Stage C (§8.2) | Claude, `sacctmgr` |
| 2026-07-31 | B6b, AVX-512 | **NEW EXPOSURE.** Login node is `sd`/Intel **with avx512**; `sr` and `bl` (180 of 334 CPU nodes) **without**. A `-march=native` build of the C++ engine would SIGILL on `sr`/`bl` as a *fraction* of tasks — looking like flaky nodes, not a build bug. Folded into T01 AC-5 | Claude, `sinfo` |
| 2026-08-02 | C1.11, `sacct -X` | **DEFECT IN THE PLAN AS WRITTEN, fixed.** `sacct -X` returns an **empty** `MaxRSS`; memory is accounted on the `.batch` step. The Stage C memory profile would have come back silently blank — no error, just empty cells — and Stage C1.11's only purpose is to size production `--mem` | Claude, verified on job 1679902 |
| 2026-08-02 | Queue state | No IsalSR jobs queued or running. Last IsalSR activity was the G9 alphabet gate, 2026-07-30 (1692435 FAILED 20 s, 1692445 FAILED 14 s, **1692451 COMPLETED 1 m 27 s** — the recorded pass). Cluster load: 3,950 jobs / 43 users / 16,531 CPUs, 503 PD / 3,447 R | Claude, `squeue`/`sacct` |
| 2026-08-03 | A7-BUG, C1.9-BUG, P3, P4 | **ALL FOUR CLOSED.** The during-run half of §3's dividing line is now complete. 6,247 unit tests pass; ruff and `mypy --strict` clean; 3-arm local runs green on **both** hosts. Write-up `docs/md_files/changes/c2_run_provenance.md`; certification ticket `T17-c2-submission-certification.md` | Claude |
| 2026-08-03 | 🔴 **T06 ledger off by default** | **LAUNCH BLOCKER, found while closing C1.9-BUG.** `ISALSR_LEDGER_ENABLED` defaults to `"0"` and is set in **no** worker, launcher or config — only in `measure_ledger_overhead.py` and unit tests. C2 as configured would have written five reachability rates of zero on all 8,400 runs, unrecoverably, and it would have read as "no fallbacks occurred". Exactly SP-6's trap. Mitigated by `--ledger`/`--ledger-sample-rate` flags + a warning; **the launcher must pass `--ledger`**, and B9/T06 AC-10 still own the keep-or-drop decision | Claude, grep + live run |
| 2026-08-03 | 🔴 **1-seed runs exited non-zero** | **STAGE C WOULD HAVE FAILED C1.1 ON ALL 420 TASKS.** `compute_paired_stats` raises below 3 matched seeds; Stage C ran **one** seed by design. Every task would have raised *after a complete, correct run*, written no `status_ledger.csv`, and exited 1. Fixed: the contrast is skipped with a logged reason when fewer than 3 seeds are matched. The guard stays — SP-0 probes elsewhere legitimately run one seed | Claude, local 3-arm run |
| 2026-08-03 | **§2.2 array topology — Option A** | **DECIDED.** C2 ships as **42 arrays**, one per `(method, arm, suite)`, not 6 merged ones. Every config declares exactly one suite and the launcher maps one entry to one array, so 42 is what the configs already describe; Option B would have required two new merged configs and two new `config_sha256`. Option A changes **no configuration content**, so it cannot perturb the A4b operator-set invariant, and a 42-task array fails ≈33× more cheaply than a 1,400-task one. Suite sizes × 3 seeds: nguyen 36, feynman 30, hard 30, cherrypicked 30, roundoff 24, feynman_remainder 18, strogatz 42 → **210 per `(method, arm)`, 1,260 total**. §1 and §11.3 updated to match | Claude, T17 §2.2 |
| 2026-08-03 | **§2.1 `--ledger` in the worker** | **CLOSED.** `slurm/c2_smoke/worker.sh` passes `--ledger` on every task. Without it all 1,260 (and later all 8,400) runs record five reachability rates of zero, which reads as "no fallbacks occurred" and means "nothing was counted" — SP-6's trap verbatim, and unrecoverable because the population exists only while a search runs. C1.9 asserts `ledger_enabled == true` **and `n_ledger_sampled > 0`** on 840/840 dedup-arm tasks | Claude |
| 2026-08-03 | 🔴 **Per-task post-processing is a race** | **THIRD LAUNCH BLOCKER, found while building the Stage C harness.** `orchestrator.py:631-698` runs after every cell: it writes `aggregate.csv` (3 concurrent tasks share the path), the three paired contrasts (which need arms living in *other* SLURM arrays), and `collect_status_ledger`, **a full recursive walk of the whole output tree followed by a write to one shared CSV — executed 1,260 times concurrently on GPFS, and 8,400 times in the campaign.** Torn artefacts and an arm-ordering dependence, on top of a filesystem hammer. Fixed with `--postprocess {auto,skip,only}`: array tasks run `skip`, and one dependent `afterany` job runs `only` over the whole root once the arrays drain | Claude, `orchestrator.py:631` |
| 2026-08-03 | **Stage C memory request deviates from §3.3** | **DECIDED, recorded rather than silently applied.** §3.3 sets Bingo-IsalSR to 256 GB in C2, from C1 runs that hit `MaxRSS ≈ 127.7 GB` after *hours* of evolution. A 900 s run cannot approach it, and holding 256 GB × 210 tasks for 15 minutes would turn §8.2's achieved-concurrency figure into a measurement of fat-node availability rather than of core contention. Stage C requests **UDFS 16 GB, Bingo 32 GB, Bingo-IsalSR 48 GB** — >4× any plausible 900 s peak — and the production `--constraint`. C1.11's product is the *measured* `MaxRSS`, which the request does not affect, and the plan itself designates **D1.2 at the full 12 h budget** as what sizes production memory. **Consequence to carry:** the Stage C concurrency figure applies to the five non-Bingo-IsalSR arm-arrays; Bingo-IsalSR's production concurrency stays bounded by §3.3's node arithmetic (`sd` 0, `sr` 1, `bc` 2, `bl` 7 per node) | Claude, T17 AC-3 |
| 2026-08-03 | **Stage C does not pin a node family** | **DECIDED.** `--constraint=cpu`. The engine is `isa_level = x86-64-v3`, `avx512f = 0`, hence portable across `sd`/`sr`/`bc`/`bl` (B6b), and **Stage C produces no number that enters a table**, so the wall-clock-homogeneity argument for pinning does not apply to it. Every run records its own `cpu_model` (A7), so B5's node census arrives as a by-product and the arm balance across node types is reportable. The pinning decision for **C2 itself** stays open at B6 | Claude |
| 2026-08-03 | 🔴 **A13 fails for the campaign, passes for Stage C** | **MEASURED live 17:31 UTC.** FSCRATCH **222.8k / 250.0k soft / 400.0k hard** files → **27.2k of headroom against the ticket's ≥60,000 criterion**. Stage C needs ≈7.9k inodes and **fits** (230.7k < 250k soft). Campaign C2 needs ≈45k and **does not** (275.7k > 250k soft, still < hard). Separately, HOME is **0.43 TB against a 0.28 TB soft quota with 2 days of grace**, and **436 GB of the 0.43 TB is `~/execs/vena`** — a different project entirely. Both must be cleared before C2; neither blocks Stage C, whose logs are ≈20 MB / 2.6k files on HOME. Projection: `c2_preflight/storage_projection.md` | Claude, `quota` |
| 2026-08-03 | **A4/A5: 10 configs still declare `n_seeds: 30`** | **RECORDED, not blocking Stage C.** The five D1 suites × both methods still carry `n_seeds: 30`; the D2 suites correctly carry 20. §0.4a fixes the campaign at 20. Stage C is unaffected — its seeds are passed explicitly (`--seeds 0,101,102`) and its task counts derive from that flag, never from `n_seeds`. **Must be corrected before C2 is submitted.** A4 otherwise passes: no arm block overrides a host-search hyperparameter, and the top-level `isalsr:` block holds only canonicaliser settings | Claude, `c2_preflight/config_diff.md` |
| 2026-08-03 | **B6b-PRE re-measured: `gcc/11.1.0` still present** | The module list on `picasso3` today offers `gcc/{11.1.0, 12.2.0, 13.1.0, 13.2.0, 14.3.0, 15.2.0}`. Built with **`gcc/13.2.0`**; `pip install -e . --force-reinstall --no-deps` exits 0, the `.so` mtime is **2026-08-03 19:27** (post-dating the last `src/isalsr/core/native/**` commit, 2026-07-30), `build_hash = 298fc1188bf1b051` **identical to the local gcc 12.2.0 build**, `isa_level = x86-64-v3`, `avx512f = 0`, and it imports with every module purged. ⚠ **Trap re-confirmed:** piping `module load` into anything (`module load gcc/13.2.0 \| tail`) runs it in a subshell and the `PATH` change is lost — the build then silently used the system g++ 7.5.0 | Claude |
| 2026-08-05 | ✅ **T07 / B4 gate 3 RECONCILED — the 5 counterexamples are GONE, re-measured on `3d5a79c`** | **The 2026-08-03 row below reads "blocking for Stage F"; the code says otherwise, so it was re-measured rather than argued about.** Re-ran the full B4 harness (`experiments/scripts/equivalence_gate.py --gate all --backend-a python --backend-b cpp`) on the current working commit: **gate 1** 33 DAGs / **54,765** comparisons, 0 cross-mismatch, 0 invariance-mismatch; **gate 2** 10,000 DAGs / 10,000 comparisons, 0 mismatches (43,222 generated, 76.9 % discarded as unreachable); **gate 3** 10,000 DAGs / **20,000** comparisons, **0 mismatches on BOTH engines**, 0 errors. Verdict **PASS**, 3.45 s. **The check is not vacuous:** the gate-3 corpus contains **15 DAGs with an over-saturated binary node** (max binary in-degree 4), which is exactly the class CLAUDE.md names as the standing control — the failure mode is representable in the corpus and does not occur. **Mechanism, so this is an explanation and not a coincidence: T18 (2026-08-03) narrowed `is_isomorphic` to compare `ordered_inputs(v)[0]` only.** Σ_SR encodes the first operand and nothing further — every surplus in-edge is emitted by `C`/`c` in canonical-traversal order, so its position carries no information the canonical string could recover. The old whole-list comparison was therefore **strictly finer than the canonical string**, and the "5 unsound merges" were **5 spurious failures of the checker**, not of the representation. On a binary node of in-degree exactly 2 — the only case `dag_evaluator` accepts — agreement at position 0 forces agreement at position 1, so no strength is lost. Locked by `tests/unit/test_t18_operand_order_completeness.py` (25 tests, incl. `test_oversaturated_first_operand_still_separated`); 808 round-trip/isomorphism/completeness tests pass repo-wide. **SP-3 discharged with a negative control:** the harness probes the C++ path explicitly (*"C++ canonicaliser is live"*), and forcing `--backend-b python` makes it report `self_comparison=true` and **FAIL** at the top level — so a PASS means both engines genuinely ran. ⚠ **What is NOT closed:** this is the *generated* corpus. The rate on **evolved** candidates remains open (T01 measured 0/117,798 on a different corpus; B3 shows real Bingo candidates reach k = 37) — Stage D's dedup arms report `n_canon_raised = 0` and `n_canon_timeouts = 0` across 11.4–14.1 M sampled candidates, which is corroborating but is a *raise* counter, not an isomorphism check. **T07's remaining Stage-F obligation is Ezequiel's proof work (§7bis.2), not this gate** | Claude, gate report `T07-appendix/gate_all_3d5a79c_2026-08-05.json` |
| 2026-08-03 | 🟡 **Canonical-string completeness: 5 counterexamples** (SUPERSEDED — see the 2026-08-05 row above; retained for the record) | **FOUND AT B4, ESCALATED, does not block Stage C.** Gate 3 (round-trip isomorphism) fails on **5 of 10,000** generated DAGs (0.05 %), `k ∈ {13,15,17,18,19}`, **identically on both engines** — so B4's own question (does C++ agree with Python on this compiler and CPU?) is answered **yes**: gate 1 54,765 comparisons / 0 mismatches, gate 2 10,000 / 0. The failure is a property of the canonicaliser, present equally in the Python code that produced C1. **It is not cosmetic:** for all five, `fcs(D) == fcs(S2D(fcs(D)))` **and** `D ≇ S2D(fcs(D))` — two non-isomorphic labeled DAGs share a canonical string, an **unsound merge**, which biases ρ *upward*. The obvious artefact is ruled out: **no case has an in-degree-0 CONST**, and three of the five also have **no VAR as an edge target**, so they sit inside `𝒞₂` where `normalize_const_creation` is equivariant; node counts, edge counts and label multisets are equal. **Distinct from T15**, whose 6/4,000 *raise* and are counted by the T06 ledger — these raise nothing and return a well-formed string. Open: the rate on *evolved* candidates (T01 measured 0/117,798, a different corpus — but B3 shows real Bingo candidates reach **k = 37**). Owner **T07**; blocking for **Stage F**, not Stage C. Write-up: `docs/md_files/changes/canonical_completeness_counterexamples.md` | Claude, job 1751918 + local repro |
| 2026-08-03 | **Bingo `trajectory.csv` counted the wrong population** | **FIXED.** On the dedup arms, `n_dags_explored` rows 1…n−1 carried `ExplicitRegression.eval_count` — fitness-function *invocations*, inflated **3.3–4.1×** by ScipyOptimizer/LocalOptFitnessFunction inner iterations during LM constant optimisation — while the final row carried `dedup.n_total` (candidate DAGs). The series climbed to ~110k then **dropped to ~30k**: the same quantity measured two ways. **ρ was never affected** — it is built from `dedup.n_total / dedup.n_unique` in the same loop, and no analyzer or figure code reads this column (0 grep hits). The correct per-snapshot value already existed unused. Fixed in `experiments/models/bingo/translator.py`: each arm now uses whichever counter its own final row uses (`n_total_dags` where it exists; `n_evals` on the baseline, where `n_total_dags` is 0 by construction and the final row uses `eval_count` anyway). Verified monotone on all three arms. Without this, **C1.10 would have failed on all 420 Bingo dedup-arm tasks** | Claude |
| 2026-08-03 | **Baseline ρ is definitional, not measured** | Recorded so no one reads it as evidence. `bingo/runner.py:428` sets `n_unique_canonical = total_evals`, so the baseline arm's `empirical_reduction_factor` is **1.000 by construction** — it has no dedup hook and nothing to measure. Intermediate baseline rows also write `n_unique_canonical = 0` and jump on the last row. Neither is a defect; both must be stated rather than compared against the dedup arms as if they were measurements | Claude |
| 2026-08-03 | **SP-1 could never have passed, by construction** | The deployment path is `rsync`, and the historical sync command **excluded `.git`**. Picasso's `.git` therefore described a months-old commit while the working tree was current, so `git describe --tags --always --dirty` on a compute node reported a stale hash with `-dirty` **permanently** — the check that exists to catch a stale deployment instead cried wolf on every run. Observed live: remote `b34cded-dirty` against local `d02106f`. Fixed by `slurm/c2_smoke/deploy.sh`, which refuses to deploy a dirty local tree, syncs `.git` with everything else, and then **verifies from the remote side** that HEAD matches and the tree is clean before rebuilding | Claude, job 1751916/1751917 |
| 2026-08-03 | 🔴 **`sbatch --export` cannot carry a comma in a value** | **CAUGHT IN THE FIRST WAVE, FIXED.** `--export` is comma-separated, so `C2_SEEDS=0,101,102` delivered **`C2_SEEDS=0`** to every worker. Each array then had `n_problems` valid indices instead of `n_problems × 3`: **265 of 1,260 tasks died** with `index N out of range`, while every task *below* the cut produced a correct-looking **seed-0 cell**. An array that is 1/3 right and 2/3 wrong is far worse than one that fails outright — and the 42-task probe **could not** catch it, because index 1 is valid under both readings. Fixed by shipping the list colon-separated and translating in the worker, which now also **asserts** it decoded to more than one seed rather than trusting the transport. Verified at the exact failing indices before relaunch (feynman index 11 → `I.25.13 101`, index 30 → `I.48.20 102`) | Claude, wave 1 |
| 2026-08-03 | **B9 answered: the T06 counters are essentially free** | **PASS, both hosts, under the C++ engine and the decomposed alphabet.** UDFS overhead **0.04 %** (`n_ledger_sampled = 2,397`), Bingo **0.22 %** (`n_ledger_sampled = 277,553`), counters live in both. A single 900 s Bingo cell sampled **711,419** candidates with all five paths finite (`violated_pre = 495,589`, `violated_post = 0`, `timeout = 0`, `conversion_failure = 0`, `canon_raised = 0`) and per-k histograms out to k = 41. **The keep/drop decision (T06 AC-10, Mario's) can be taken on measured evidence: keep.** The earlier "counters dead" report was a defect in the probe's own JSON path, not in the ledger | Claude, jobs 1751997/1751998 |
| 2026-08-03 | **Stage C memory sized from measurement** | Peak `MaxRSS` across the 42-task probe: **0.53 GB**, flat from a 13 s run to a 15 min one. Requests cut from 16/32/48 GB to **8/16/16 GB** (still 15–30× the observed peak). Over-requesting is not free: SLURM cannot pack tasks it believes need 48 GB, which throttles exactly the achieved-concurrency figure §8.2 must measure here. **First reading: ≈336 concurrent tasks**, at or above §8.2's 300-core target | Claude |
| 2026-08-03 | **Post-search SymPy tail is real and unbounded by `max_time`** | One probe cell (`bingo/hash/roundoff`) finished its **search** correctly at 900 s — Bingo logged *"Preemtively stopping because maximum time would be exceeded"*, so `max_time` **is** honoured (C1.12) — then spent **7+ further minutes** in post-search SymPy before being cancelled. This is T08's documented class (9 such hangs in C1). The 40-min SLURM wall bounds it, but the campaign must expect a tail well past the nominal budget and must not read it as a hung search | Claude, job 1751963 |
| 2026-08-03 | 🔴 **P5/§8.2 MEASURED: achieved concurrency ≈ 245 cores — C2 misses 2026-09-03, meets the freeze** | **This is the number Stage C exists to produce, and it is a schedule finding, not a formality.** The 1,260-task wave ran **1,260/1,260 COMPLETED, zero failures**, spanning **20:28:54 → 21:46:02 = 77.1 min** and delivering **315 core-hours**, i.e. **315 / 1.285 h ≈ 245 concurrent cores** under real queue pressure. Task durations cluster tightly at 15:13–15:17 (900 s payload + ~15 s overhead), so the figure is not distorted by a long tail. **Consequences, using §8.2's own arithmetic:** at 245 sustained cores C2's 100,800 core-hours takes **411 h ≈ 17.1 days**. Against the 2026-09-03 completion target (14 days) that is **≈3 days short**; against the 2026-09-10 freeze (504 h) it **fits, with ≈19 % headroom**. §8.3's trade order should therefore be considered **now**, not in week three — the cheapest lever (item 1, accelerating T04/T05) is already spent, so the live options are D2→Strogatz-only or the seed reduction. **Two caveats that must travel with the number:** it was measured at **8/16 GB per task with a `%8`-per-array throttle** (a 336-task ceiling, so contention cost ≈27 % of the ceiling), and **Bingo-IsalSR at §3.3's 256 GB will be far more constrained** — `sd` cannot host one at all. The 245 applies to the five non-Bingo-IsalSR arm-arrays | Claude, jobs 1752689–1753133 |
| 2026-08-03 | **C1.11 memory profile measured** | Peak `MaxRSS` over all **1,260** tasks: **0.67 GB**, against 8 GB (UDFS) / 16 GB (Bingo) requested — 12–24× headroom, and 380× below §3.3's 256 GB Bingo-IsalSR figure. At 900 s the dedup set never approaches the C1 ceiling. **This does not resize production**: §4.4 D1.2, at the full 12 h budget, is what sizes C2's `--mem`, and the C1 OOMs occurred after *hours* of evolution | Claude |
| 2026-08-03 | ⚠ **The aggregation step is not cheap — size its wall limit for C2** | Stage C's single `--postprocess only` job took **≈7 min per config**, ≈1 h 40 m for all 14, over 1,260 runs. It re-reads every `run_log.json` for every `(benchmark, problem, arm)` and every contrast, so the cost scales with the campaign, not with the number of configs: at C2's **8,400** runs the same job is ≈6.7× larger, i.e. **≈11 hours**. Stage C ran it with a 2 h limit, which fitted only just. **C2's aggregation job needs ≥24 h and should be split per method**, or the campaign ends with 8,400 valid run logs and no `aggregate.csv`, no paired stats and no ledger — recoverable, but only by re-running the step and noticing it was truncated | Claude, job 1753134 |
| 2026-08-03 | ⚠ **Never edit a config while an array is reading it** | Recorded because it was nearly done. The 10 stale `n_seeds: 30` configs are a known pre-C2 fix, and the obvious moment to fix them is while waiting for a wave to drain — which is exactly wrong. `config_sha256` is computed **per run, from the file on disk**, and every task records it. Editing and syncing a config mid-wave gives some cells one hash and some another, silently, inside a campaign whose whole premise (§5.1) is one commit and one configuration. **Config edits happen between waves, never during one.** | Claude |
| 2026-08-03 | 🔴 **C3 has no mechanism yet** | The dedup-off equivalence control needs the `isalsr` runner run with dedup **forced off**, and **no such switch exists** in either host runner (`experiments/models/{udfs,bingo}/isalsr_runner.py`). C3 is blocking for **Stage D**, not for Stage C's 1,260-task array, and runs as its own 6-task job. Implement before Stage D | Claude, grep |
| 2026-08-04 | ✅ **B6 CLOSED: pinned to `sr`, and it made the wave FASTER, not slower — 592 cores, C2 in 7.1 days** | **The expected cost of pinning did not materialise.** Stage C v3 ran the identical 1,260-task wave with `--constraint=sr` and the corrected `I.34.27`: **1,260/1,260 COMPLETED, 1,260/1,260 placed on `sr`**, spanning **14:16:45 → 14:48:40 = 31 min 55 s**. Achieved concurrency = 315 core-h ÷ 0.532 h = **592 cores**, *above* the 476 measured on the unpinned `cpu` pool. The reason is packing, not luck: `sr` nodes carry **128 cores** against `sd`'s 52, so a 1-core task fits far more readily and the scheduler does not have to straddle families. **C2's 100,800 core-hours therefore take ≈170 h ≈ 7.1 days**, better than both earlier projections. The accepted trade (Mario: "we may no longer get quick access to running nodes, but that's OK") was real but did not bite at this scale — it may still bite for Bingo–IsalSR if §3.3's 256 GB request survives D1.2, since `sr` is 450 GB/node and hosts one such task per node. **Do not generalise this to a claim that pinning is free**; it is free *here*, on 1-core 900 s tasks, and D1.2 must re-check it at 12 h | Claude, jobs from 1760708 |
| 2026-08-04 | ✅ **CONCURRENCY RE-MEASURED AT `%24`: 476 cores. C2 takes 8.8 days. §8.3 is NOT needed** | **This closes the schedule question and supersedes the 245-core projection below.** The Stage C wave was re-run identically except for the throttle (`C2_THROTTLE=24`, ceiling 42 × 24 = 1,008) into `c2_smoke_v2/`: **1,260/1,260 COMPLETED, zero failures**, spanning **11:32:49 → 12:12:33 = 39 min 44 s**. Achieved concurrency = 315 core-h ÷ 0.662 h = **476 cores**, against 245 at `%8` — a **1.94×** gain from a single shell variable, with a peak of **909 tasks running at once**. Some contention is real (476 < the 1,008 ceiling, and 1.94× < the 3× the throttle rose), but the throttle was clearly the **dominant** constraint. **Consequences:** C2's 100,800 core-hours now take **211.8 h ≈ 8.8 days**, which meets the 2026-09-03 completion target with **≈37 % headroom** and the 2026-09-10 freeze with far more. **No §8.3 trade is required** — not D2→Strogatz-only, not dropping the hash arm, not cutting seeds, and not the 12 h→8 h budget. The peak was 90 % of the `%24` ceiling, so more is available if ever needed. Caveat unchanged: Bingo–IsalSR at §3.3's 256 GB stays node-bound (`sd` 0, `sr` 1, `bc` 2, `bl` 7 per node) regardless of throttle | Claude, jobs 1758681–1759668 |
| 2026-08-04 | 🔴 **The 245-core figure is OUR OWN THROTTLE, not cluster contention — §8.3's trade is probably unnecessary** | **CORRECTION to the 2026-08-03 entry below, and the most consequential finding of the day.** Stage C ran with `C2_THROTTLE=8` (`launcher.sh:46`), i.e. 42 arrays × 8 = a **336-task concurrency ceiling that we imposed**. The measured 245 is **73 % fill of our own cap**, not a measurement of what the cluster would give. Live state 2026-08-04: `sinfo` reports **7,612 idle CPUs** (20,940 alloc / 38,204 total) and QOS `long_uma` allows **`cpu=9000` per user** — so the entitlement is ~37× the throttle we set. **Consequence:** the "C2 takes 17.1 days and misses 2026-09-03" conclusion does not follow from the data; it is an artefact of the throttle. **Re-measure achieved concurrency at a materially higher `%K` before invoking any §8.3 trade** — every trade on that list costs science (D2 coverage, the hash arm, or seeds) and this one costs a shell variable. Caveat that survives: Bingo–IsalSR at §3.3's 256 GB stays node-bound (`sd` 0, `sr` 1, `bc` 2, `bl` 7 per node) regardless of throttle | Claude, `launcher.sh:46` + `sinfo`/`sacctmgr` |
| 2026-08-04 | **12 h → 8 h budget: REJECTED on evidence** | Mario asked whether the per-run budget should drop to 8 h to fit the schedule. **Answer: no.** Replaying C1's **5,955** trajectories truncated at 28,800 s (zero compute) shows the two hosts respond very differently. UDFS is budget-insensitive: W/T/L 35/14/1 → 35/13/2, sign-test p 5.4×10⁻¹⁰ → 5.1×10⁻⁹. **Bingo erodes badly**: 22/23/5 → **19/23/8**, p 7.6×10⁻⁴ → **2.6×10⁻²**, and mean δᵢ falls **+0.00087 → +0.00027 (−69 %)**. At 6 h it inverts (16/23/11, p = 0.22, δ̄ < 0). Three problems flip win→loss. This is the 2026-04-19 bottleneck mechanism showing up directly — IsalSR pays an early exploration cost and overtakes in gen 500–1000, so truncating the budget truncates our own effect. **The asymmetry that decides it:** the seed cut (30→20) costs *precision*; a budget cut costs *effect size*, and Bingo's is already d = 0.034. Caveats travelling with the number: `best_r2` here is train R² from `trajectory.csv`, not the test-R² headline, and this is C1's alphabet/engine — the mechanism transfers, the exact p-values do not | Claude, C1 trajectory replay |
| 2026-08-04 | **R2.3 needs no change to the campaign shape** | Asked whether the host-operator-set correction (R2.3 / A4b) obliges us to add D1+D2 for the baseline arm. **It is already there:** §0.4 (2026-07-31) supersedes §9.2 and fixes the shape at `{baseline, hash, isalsr} × {UDFS, Bingo} × (D1 ∪ D2) × 20 seeds` = 8,400 runs. R2.3's sentence *"the runs whose operator set changed were repeated under it"* is discharged by that full re-launch. The only knock-on is §7's continuity table, whose 22-problem Bingo exclusion is already recorded | Claude |
| 2026-08-04 | 🔴 **C1.10 FAILED 452/1,260 — `trajectory.csv` reported two different quantities** | **FIXED.** `best_r2` carried **train** R² on every intermediate row and **test** R² on the final row alone, so the series decreased whenever `r2_test < r2_train`. A scan of all 1,260 trajectories put **100 % of the violations at the final row and nowhere else**, both hosts (UDFS 241, Bingo 218). **Same defect class as the `n_dags_explored` mix-up fixed 2026-08-03** — intermediate rows measuring one population, the final row another — which is why it survived: that fix corrected the counter column and not this one. Both translators now report train metrics on the final row; test metrics stay authoritative in `run_log.json`'s `results.regression`. **No reported number changes**: the convergence scripts read `best_r2_train` from the `.npz` and `time_to_r2_*` is computed from the snapshots, so this column has no analysis consumer (0 grep hits). 5 new tests, verified to fail 4/5 against the pre-fix translators | Claude, job 1758604 |
| 2026-08-04 | 🔴 **C4 FAILED 203/210 — but the property it protects PASSES** | **Criterion corrected; one real finding escalated.** The shortfall decomposes exactly: **−4** Pagie-1 and Keijzer-6 (deterministic grids, 3 seeds → 1 fingerprint each; the **only** 2 of 70 that are seed-invariant, and the published protocol) and **−3** `I.12.1` ≡ `I.34.27`. `cross_arm_disagreement = 0` on **1,260/1,260**, so all three arms and both methods saw identical data — **the paired design is not void.** 🔴 **`I.12.1` (`mu*N_s`) and `I.34.27` (`hbar*omega`) are the same problem**: both reduce to `sympy x_0*x_1` on `[1,5]²` and generate byte-identical data. Our `feynman.py` dropped the `1/(2π)` the AI Feynman catalogue specifies. It is in **D1**, so it has inflated `N` since C1 — CPDT counts one observation twice. **Mario's decision:** restore the constant (recommended; I.34.27 then joins §7's continuity exclusions) or drop a problem and report `N = 69`. Write-up: `docs/md_files/changes/c4_fingerprint_findings.md` | Claude, job 1758604 |
| 2026-08-04 | ⚠ **C1.11 said PASS while sizing production memory from 3 % of the campaign** | **FIXED.** `certify.sh` emitted `sacct`'s `JobID` (`<array_id>_<task>`), while the certifier joins on `status.json`'s `slurm_job_id` (= `$SLURM_JOB_ID`, the **per-task** id). Those coincide only for **task 1 of each array**, so the join matched **42 of 1,260** rows, binned 1,218 as `unmatched` — **and still reported PASS**. Exactly the C1.11 trap the plan already warns about (`sacct -X` returning blank), in a second disguise: not blank, just silently sparse. Now emits `JobIDRaw` | Claude |
| 2026-08-04 | **The 10 stale `n_seeds: 30` configs are now 20** | Closed the A4 finding of 2026-08-03. All **14** configs verified at `n_seeds: 20`, and the **A4b operator-set invariant re-checked after the edit**: 1 distinct set per method, non-empty (`bingo` 10 operators, `udfs` 11). Done **between waves**, never during one — the certifier job reads no config, which is what makes the moment safe | Claude |
| 2026-08-03 | **Stage C moves to 3 seeds** (§4.3 amended) | **DECIDED (Mario).** Seeds **0, 101, 102** — disjoint from campaign seeds 1…20 *and* from the 21…30 top-up range, so SP-0's "never mistakable for a campaign cell" intent is preserved. Reason: at one seed the three paired contrasts are **never constructed**, so the smoke would certify everything except the machinery the paired design rests on, and the first three-arm run of that code would be the campaign itself — §4.5's most expensive failure mode. **Two seeds does not work**: the threshold is `< 3`. Three is also `scipy.stats.shapiro`'s minimum, exercising the Shapiro→t/Wilcoxon branch at its boundary. Cost 105 → **315 core-hours (0.3 % of C2)**. Buys the code path only — at `n = 3` the minimum two-sided Wilcoxon `p` is 0.25, so **no number from Stage C means anything** | Claude, `aggregation.py:207` |

| 2026-08-04 | 🔴 **Fairness audit of the whole experimental setup** | **16 findings; 5 defect clusters fixed.** Commissioned to check whether the setup over-favours the competitor arms or IsalSR. Core design verified sound: identical configs/budgets/seeds/data across arms, host-enforced budget windows with no hidden pre-computation, an honestly-naive hash competitor, arm-independent metric scoring. Defects were in *accounting*, and cut both ways — pro-IsalSR: CPDT tie-dropping (§6.1a), the tautological ρ p-value (§6.1), overhead understated 1.57×/2.43× by excluding adapter conversion; pro-competitor: `T_search` inflated by untimed instrumentation (so C1's Bingo `S = 0.93` partly measures wrapper bookkeeping), duplicates denied the LM restarts the baseline gets, and Bingo's cross-arm `total_dags_explored` flattering the baseline 3–4× (`eval_count` vs candidates — **never report it cross-arm**). Full log: `audit.md` | Claude |
| 2026-08-04 | **Cost attribution corrected** | `conversion_time_s` and `shadow_time_s` are now timed per candidate and reported. `wall_clock_search_only_s = wall − canon − conversion − shadow`; `overhead_time_s = canon + conversion` (shadow excluded — it is audit instrumentation, not method cost). ⚠ **Two consequences to carry:** D1.7's "Bingo ≈7.4 %" expectation was canon-only and the honest figure is higher — not a regression; and **T01 AC-6's `S` value was derived under the old formula**. The mechanism (`dS/dT_canon = 0`) survives; the number moves upward for `isalsr` | Claude |
| 2026-08-04 | **Shadow sketches OFF campaign-wide** | **DECIDED (Mario).** Measured on the v4 wave: Bingo `isalsr` **17.6 %** of wall clock — *more than the 14.8 % method overhead it instruments* — against UDFS's 0.034 %, paid by one arm inside a fixed budget. Since C1's trajectory replay shows Bingo's paired effect eroding 69 % from 12 h to 8 h, that is our own effect being cut. `shadow_hash: false` in all 14 configs, **in the method block** — `create_runner` passes `config.get(method, {})`, so a key under `isalsr:` is silently ignored and the sketches would have run on all 8,400 runs unrecoverably (the `ISALSR_LEDGER_ENABLED` trap again). Fixed-order numbers now come from D2/D3 only | Claude, audit.md §7.3 |
| 2026-08-04 | **"Steel-man" second hash number WITHDRAWN** | **Mario's challenge, accepted.** The audit had asked for the adapter-order ρ_hash to be reported beside the live host-native one. That does not survive scrutiny: the adapter's renumbering is *part of IsalSR's own preprocessing*, so the adapter-order rung is not an independent competitor but a hybrid nobody could build without our adapter, and two published ρ_hash numbers invite "which is the baseline?". **Replacement, and it is stronger:** state which representation the naive hash keys on, and answer the "UDFS ρ_hash = 1.0000 looks rigged" objection *mechanistically* — systematic enumeration emits structurally distinct `node_dict`s, so a representation-order hash has nothing to merge; the redundancy is purely isomorphic. The adapter-order rung stays computable from D3 and is held in reserve for the letter | Mario + Claude |
| 2026-08-04 | **A6 / A8 / A13 / B8 closed** | MANIFEST + strict validator (46 tests); three-arm analyzer with Holm-by-3 asserted (52 tests); FSCRATCH **155.4k / 250k = 94.6k headroom** after archiving the superseded `c2_smoke`/`c2_smoke_v2` roots (only 15.9k of the drop attributable — the earlier 248.6k reading was likely stale); resume/corruption proven on Picasso. ⚠ **HOME remains over quota** (0.34/0.28 TB) — Mario's lane, deliberately deferred, and mitigated for the campaign by writing all wave/Stage-D logs to FSCRATCH | Claude |
| 2026-08-04 | **Stage C v3 GO; C4 PASS** | 1,260/1,260 on `sr`, span 31 m 55 s. C4 multiplicity histogram **{6: 204, 18: 2}** — 204 fingerprints at 3 arms × 2 methods, the two declared deterministic grids at 18; `cross_arm_disagreement = 0`, `duplicate_problems = 0` (the `I.34.27` 1/(2π) restoration holds), `seed_collapse = 0`. **The last Stage C blocker closed** | Claude, job 1761777 |
| 2026-08-04 | **Audit branch merged; C5 signed** | `6c3798f` merges 20 commits into `cpp-core-port` (zero conflicts); A2 green **on the merged commit** (6,759 tests, ruff, mypy-strict); C5 signed by Mario with §3.5's ρ deviation handed to D1.6 | Claude |
| 2026-08-04 | ⚠ **A deploy IS a config edit — never during a wave** | Generalises the 2026-08-03 config rule. A mid-wave redeploy split provenance across two HEADs inside a certification wave; caught ≈10 min in, wave cancelled, root wiped, relaunched on a stable commit. Separately, 161 of v4's 1,260 cells recorded `a455d6c-dirty` from an in-place `sed` on the *deployed* tree — same lesson, second form. **The deployed tree is read-only while any array is running** | Claude |
| 2026-08-04 | **Two launcher bugs only `--test-only` could find** | (1) python resolved via a workstation `~/.conda` path absent on Picasso; (2) `GROUPS=` is a **reserved bash array** — assignment returns an error status and `set -e` killed the launcher before its first `echo`, exiting 1 with **zero output**. Both fixed. Neither is reachable locally: there is no `sbatch` on the workstation. **Run `--test-only` on the cluster for every new launcher** | Claude |
| 2026-08-05 | **Stage D submitted — 13 cells** | Jobs **1769422** (udfs 3 × 16 GB), **1769423** (bingo_std 6 × 32 GB), **1769424** (bingo_isalsr 4 × 256 GB), certifier **1769425** (`afterany`, dependency verified non-null). **13/13 RUNNING within 20 s**, including all three 256 GB cells — `sr`'s 128-core/450 GB nodes absorbed them without queueing. Deployed `00635ae` from a temporary clean clone, because `deploy.sh` (rightly) refuses a dirty tree and the tree carried another agent's in-flight work; SP-1 and SP-2 both verified from the remote side. ⚠ **`git pull` is impossible on Picasso** — no outbound SSH to GitHub; `deploy.sh` is the only path | Claude |
| 2026-08-05 | 🔴 **A clean Stage C wave on `00635ae` is still owed** | v4 certified the merged commit but under the **pre-shadow-off configs**, and 161 of its cells are `-dirty`. §5.1 requires the campaign to launch on a commit **and configuration** a Stage C wave has certified, so one more 1,260-task wave (~35 min at `%24`/`sr` + aggregation) runs on `00635ae` **before the `campaign/c2` tag**. Stage D is unaffected: its cells record their own clean provenance | Claude |
| 2026-08-05 | ✅ **Stage E COMPLETE — 7/7, and it found four defects, three of them invisible to an exit code** | **VERDICT GO**, 182 s end to end on `c2_smoke_v4`, run **locally**: every failure needed an analyzer fix, and defect 10 forbids deploying while Stage D's arrays run. **E4 🔴 the D2 extension breaks LaTeX compilation.** `generate_tables.py` typeset identifiers raw, and every T05 D2 name carries an underscore (`strogatz_vdp1`, `liv_19`, `pagie_2`, `feynman_remainder`); a bare `_` outside math mode aborts `pdflatex`. **18 rows per table across 4 tables — exactly the rows the coverage extension added.** The generator exited 0 and the tables did not compile. Three emission sites fixed via a shared `_latex_escape`. **E5 🔴 the hash arm vanished from every CD diagram.** `generate_critical_difference.py` iterated a **hardcoded** `["baseline","isalsr"]` at both loaders with no `--variants` plumbing, so the three-arm root gave **4 groups** where 2 methods × 3 arms gives 6 — the arm R1.4 asks about, absent from every critical-difference figure, silently. `cross_method.py` had been extended for three arms; the figure generator had not. Now 6 groups on all 70 problems. **E6/E7 🔴 neither check existed**: `reconcile()` lived in `status_ledger.py` and `analyze.py` never called it, and there was no provenance check of any kind. New `analyzer/completeness.py` fails closed (exit **2**, overrides `--allow-incomplete` / `--allow-mixed-provenance`), names missing cells rather than counting them, and infers the expected grid as a cross product so a deletion cannot define itself away. Note the ledger alone would **not** have caught E6: deleting a `run_log.json` leaves `status.json` intact. **Two carries.** ⚠ **`git_commit` is `None` on all 1,260 runs** — A7 lists it, `collect_hardware_info()` never fills it, so a guard keyed on it would pass **vacuously on every campaign**; the SP-6 trap again. The guard keys on `git_describe`/`git_dirty`/`build_hash` and reports absent keys as *non-informative* rather than as agreement. ⚠ **The guard independently rediscovered v4's dirty split** (`a455d6c` ×1,099 vs `a455d6c-dirty` ×161, `config_sha256` clean at 1 per method×suite), so **the owed clean Stage C wave is now enforced by code**: E1 needs `--allow-mixed-provenance` on v4, and on the v5 root it must pass **without** it or the `campaign/c2` tag must not be cut. Cost note: the analyzer is 49 s / 1,260 runs ⇒ **≈5.5 min at 8,400** — not to be confused with the orchestrator's `--postprocess only` aggregation (1 h 35 m ⇒ ≈11 h). Tests `tests/unit` **6,929 passed**; the 2 failures in the tree are `test_appendix_d_generator.py`, the T09 agent's **untracked** work, verified zero import coupling to any file touched here and left alone; ruff `src/`+`tests/` clean and `experiments/` **24 → 22** on the modified files (zero introduced); `mypy --strict` clean. Write-up: `docs/md_files/changes/stage_e_design.md`; runbook `slurm/c2_stage_e/RUNBOOK.md` | Claude |
| 2026-08-05 | 🔴 **F-19 FIXED — three suites ran at a 10× tighter search budget** | **`bingo_{roundoff,strogatz,feynman_remainder}.yaml` omitted `max_evals` and silently inherited `BingoConfig.max_evals = 10_000_000` (`config.py:29`) against the other four suites' 100M** — a **10× budget disparity on 28 of 70 problems**, undocumented, and pooled into the N=70 CPDT. It matters because **`max_evals` is the BINDING budget for Bingo**: all six completed Stage D cells stopped on it at 100M ± 0.07 % while `max_time = 43,200 s` **never fired on any cell**. Fixed in all three configs with the rationale inline; verified through the **loader**, not the YAML text: `max_evals`, `max_time`, `population_size`, `stack_size` are now single-valued across all 7 suites. Locked by `tests/unit/test_budget_uniformity.py` (17 tests), which asserts each suite *declares* the key (never inherits it) and that the declared value exceeds the default it replaces — the same defence shape as the `shadow_hash` and `ISALSR_LEDGER_ENABLED` traps. ⚠ **Carry:** C1's `roundoff`/`strogatz`/`feynman_remainder` results were very likely produced under the 10M cap, so **their δ are not comparable with the other four suites' and must be checked before any pooled CPDT is re-quoted**. Local edits only; **nothing deployed** — Stage D's cells were running | Claude |
| 2026-08-05 | ✅ **F-20 CLOSED — UDFS `n_calc_nodes` levelled to 5, and it adds a third §7 continuity exclusion** | **Decision taken: level DOWN, feynman 7 → 5.** Verified **through the loader**, not the YAML text: `n_calc_nodes {5: 7}`, and `max_orders {200000: 7}`, `max_time {43200: 7}`, `processes {1: 7}` alongside it — UDFS's search-capacity knobs are now single-valued across all seven suites. Locked by `test_budget_uniformity.py::test_udfs_budget_is_uniform_across_suites`, parametrised over all three keys, so this cannot silently un-level. **Two reasons for levelling down rather than up, both recorded inline in the config:** (i) the original rationale (*"Feynman problems have up to 3 variables; more calc nodes needed"*) is **inverted against the portfolio** — the 5-variable suites (`hard`, `cherrypicked`, `roundoff`) already ran at 5, so the suite with the *fewest* variables carried the *largest* cap; (ii) **UDFS saturates its 12 h budget on 100 % of runs** (C1, n=600, mean = median = max = 12.00 h), so raising the cap enlarges the enumeration space without enlarging the budget and UDFS would cover a *smaller* fraction of it — the change would likely have degraded results portfolio-wide. ⚠ **Carry, and it is a new §7 exclusion nobody had:** `n_calc_nodes` caps the number of intermediate nodes UDFS may enumerate (`dag_search.py:594`), so the **10 `feynman` problems ran C1 with a strictly larger reachable space than they will run C2** and their UDFS rows are **not like-for-like**. §7 previously asserted *"UDFS is unaffected by the second exclusion, its set having never been ours to set"* — true of the operator set, **false of the search-space bound** once F-20 was closed. §7 now carries **three** exclusions, not two, and the direction is adverse to us: C2's UDFS may recover fewer Feynman expressions than C1 did, for a reason unrelated to IsalSR. Say it in the letter before a reviewer asks | Mario + Claude |
| 2026-08-05 | **Measured runtimes: UDFS saturates its budget on 100 % of runs, Bingo never does** | **The two hosts are budget-bound by different things, and this governs every capacity estimate.** C1, n=600 UDFS runs: **mean 12.00 h, median 12.00 h, max 12.00 h** — UDFS has **no `max_evals`** (its knobs are `max_time` and `max_orders: 200000`) and exhausts the full 12 h every time. Bingo isalsr, n=564: **mean 5.15 h, median 4.04 h**, max 11.76 h, corroborated by Stage D's 8 completed cells (mean 4.57 h, max 6.87 h). **Consequence for §8.1's budget**, which assumed 12 h for both: at measured runtimes C2 costs **≈71,400 core-hours at 20 seeds** (UDFS 50,400 + Bingo 21,000), not 100,800 — and **≈107,100 at 30 seeds**, i.e. 7.5 d at the measured 592 cores or 9.4 d at 476. **30 seeds is therefore affordable against the 2026-09-10 freeze**, reversing §0.4a's cost premise. ⚠ **Two consequences that must travel with this:** (1) because UDFS is *time*-budgeted, **node speed changes what UDFS explores**, so a mixed node pool alters UDFS's science and not merely its clock — this is now a second, independent reason for B6's `sr` pin; (2) any 8 h budget is refused twice over — it would cut a third of UDFS's search on 100 % of runs, and §11.1 (2026-08-04) already rejected 12 h→8 h on Bingo's 69 % effect erosion | Claude |
| 2026-08-05 | 🔴 **Stage D: `max_evals` binds, `max_time` never does — the arms stop on different rules at wall clocks differing 4.6×** | **OBSERVED FROM THE STAGE E LANE, handed to the Stage D lane; not acted on (SP-0).** Both trace problems now have all three Bingo arms complete, and none of the six cells used the 12 h budget. **Four facts, certain:** (1) the **baseline hits `max_evals = 100M` on both problems** — `total_dags_explored` 100,070,885 (Pagie-1) and 100,054,500 (Vlad-2), just over the cap `bingo_hard.yaml:38`; (2) **no cell reaches `max_time = 43,200 s`** — wall clocks are 3,593 / 10,909 / 12,501 / 16,628 / 18,790 / 20,179 s; (3) **no cell recovers the solution** (`solution_recovered = False` ×6), so none stopped on that; (4) within one `(problem, seed)` the arms differ **up to 4.6×** in wall clock (Vlad-2: baseline 3,593 s vs isalsr 16,628 s). **This is §5.4 in its own words** — "a stop rule firing at different times on different arms of the same `(problem, seed)` would make `S` a measurement of the stop rule rather than of the method" — and `S`, the overhead percentages and Table 2's cost column are all computed from wall clock. **Direction of bias is AGAINST IsalSR**: the baseline finishes 4.6× sooner only because it exhausts its eval cap sooner, so IsalSR reads as far slower on the cost axis and **D1.7's overhead figure computed from these cells would be wrong**. ✅ **RESOLVED 2026-08-05, and it CORRECTS this row's original claim.** I first read `total_dags_explored = 12.95M` on the Vlad-2 isalsr arm and concluded the dedup arms had *not* reached the cap, inferring two different stop rules. **That was wrong**: on the dedup arms that field is `dedup.n_total` (candidate DAGs), a different counter from Bingo's `eval_count`. The `.err` logs are decisive — `c2d_bingo_isalsr_1769424_3.err`: *"The maximum number of fitness evaluations (100000000) was exceeded. Total fitness evals: 100035803"*, against the baseline's 100,054,500. **All six completed cells stop on the SAME rule, `max_evals`, at 100M ± 0.07 %.** So there is no §5.4 stop-rule confound: the arms are already at an equal *fitness-evaluation* budget, and the 4.6× wall-clock spread measures cost per evaluation, not a differing stop rule. What survives unchanged: `max_time = 43,200 s` **never binds**, so the 12 h budget the design commits to is fictional for Bingo. Note the counters are not commensurable across arms (baseline `total_dags_explored` = `eval_count`, inflated 3.3–4.1× by LM inner iterations; dedup arms = candidate DAGs — §11.1 2026-08-03, "never report it cross-arm"), so the 100M cap is consumed in different units per arm | Claude (Stage E lane) |
| 2026-08-05 | ✅ **Bingo-IsalSR memory: 256 GB → 32 GB, on a HARD BOUND rather than an extrapolation** | **§3.3's own revision clause invoked** ("if D1.2 shows 12 h MaxRSS comfortably under 128 GB … the request may be revised down before launch, with the measurement recorded"). Stage D's three completed `bingo_isalsr` cells, full length and stopped on `max_evals`: **`MaxRSS` 1.05–1.16 GB** at 6.3–7.2 M unique canonical hashes; the baseline cells, which hold no dedup set at all, sit at 0.39–0.42 GB. **The reason this is not an extrapolation from three problems** — and therefore covers **Vladislavleva-4, which Stage D does not run** and which was 18 of C1's 27 OOM cells: `canonical_seen: set[int]` is the *only* unbounded container in the arm (`_parent_ids` is bounded by `population_size`, the T06 histograms by the k range); a candidate cannot enter it without being scored; and Bingo stops on `max_evals = 100M` on **all six** completed cells at ±0.07 % while `max_time` never fires. So `n_unique ≤ 100M` **on any problem**. Measured cost of a `set[int]` under the production allocator (`PYTHONMALLOC=malloc`): **81.5 B/entry**, flat 1 M→32 M, with a **1.16× transient at each table resize** (at 24 M: `VmRSS` 2.086 GB vs `VmHWM` 2.414 GB). Worst case = 100M × 81.5 B × 1.16 + 0.42 GB = **9.4 GB**; the independent structural model (2²⁸ × 16 B table + 100M × 32 B PyLongs + the old table still mapped) gives 9.40 GB. **32 GB is 3.4× the ceiling and 28× the observed.** C1's 127.7 GB is explained and gone: pymalloc arena fragmentation over 10k+ generations (fixed at `worker.sh:59`, whose comment already said so) plus `set[str]` at ~150 B/entry. **What this unlocks:** at 256 GB, `sr`'s 450 GB nodes hosted **one** such task each — a hard ceiling of 154 concurrent out of 1,400, sterilising ~115 of every node's 128 cores, and separately capped at **190 by the QOS's own `mem=50000000M`**, a limit not previously recorded. That was the campaign's real capacity ceiling. Gone. Locked by `tests/unit/test_c2_slot_plan.py::test_bingo_isalsr_memory_covers_the_hard_ceiling` | Claude, Stage D + `mem_ceiling.py` |
| 2026-08-05 | 🔴 **The uniform `%K` was the dominant schedule loss — 1.92×, invisible to every Stage C measurement** | **Reconstructed the concurrency envelope of both `%24` waves from `sacct` (zero new compute).** They are a spike, not a plateau: peak **934 of the 1,008 ceiling (93 %)**, held for **2.7 % of the span**, decaying to a mean of 67 cores over the final quarter. So the scheduler *granted* ~934 concurrent 1-core `sr` slots within minutes and **contention was never the binding constraint**. The loss is structural: the 42 arrays carry a **5.4× work spread** (`udfs:*:strogatz` 280×12.0 h against `bingo:*:feynman_remainder` 120×5.15 h), and under a uniform `%K` the small arrays drain early and hand their slots back to nobody — the survivors stay capped at `K`. Minimising `max_i N_i·T_i/K_i` subject to `Σ K_i ≤ C` gives `K_i ∝ N_i·T_i`, i.e. every array finishing together at the floor `ΣN_iT_i/C`. **At 20 seeds and the same 1,008 slots: 140.0 h → 72.7 h.** Stage C could not show this because it has **1.25 tasks per slot** and is ramp and drain end to end; C2 has 8.3. Sensitivity: planning at `T_bingo = 8 h` beats uniform by 1.65× even if the truth is 4 h, and **never loses anywhere in [4, 12] h** — asserted, not assumed, in `tests/unit/test_c2_slot_plan.py`. Landed in `experiments/scripts/c2_slot_plan.py` (49 tests); `C2_UNIFORM_THROTTLE` restores the old behaviour for A/B | Claude |
| 2026-08-05 | 🔴 **Every concurrency figure in this ledger was measured under a QOS the campaign cannot use** | **245 / 476 / 592 all came from `short`** (`MaxWall = 2 h`), at priority **118,933**. C2's tasks need 13 h and land in `medium_uma` at **28,873** — and 118,933 − 28,873 = 90,060 is *exactly* the QOS term (`PriorityWeightQOS = 100000`, `short` 10000/10000 → 100,000 points against `medium_uma` 1000/10000 → 10,000); every other factor is identical. Measured live with `sbatch --test-only`: a **1 h 59 m** request started **immediately**, a **2 h 01 m** request was estimated **three hours out**, and 13 h / 16 h / 23 h / 2 d 23 h / 6 d 23 h all got that same three-hours-out estimate. **Two consequences.** (1) The 592-core figure is an optimistic upper bound for C2 and must not be quoted without this caveat. (2) **Right-sizing `--time` below 13 h buys nothing** — there is no backfill cliff above 2 h — so the wall should be *generous* against the post-search SymPy tail rather than tight. Set to **16 h**, and kept under 3 days because above that the job drops to `long_uma` and loses a further 5,000 points. ⚠ **Fairshare is unmodelled and will erode**: `PriorityWeightFairShare = 50000` is the largest weight, `sshare` gives FairShare 0.1577 at EffectvUsage **4.97× our within-account share**, and C2 adds 72,030 core-h against a 14-day decay half-life. Front-load UDFS; watch achieved concurrency in the first six hours | Claude, `sacct`/`sacctmgr`/`--test-only` |
| 2026-08-05 | ✅ **The aggregation is ≈2 minutes, not ≈11 hours — the extrapolation was in the wrong variable** | The 2026-08-03 row projects the `--postprocess only` job to **≈11 h at 8,400 runs**, scaling the measured 1 h 34 m by 6.7× *in runs*. **Measured: it does not scale with runs at all.** Reading every one of the 1,260 `run_log.json` with the full 3-contrast amplification takes **0.36 s**, and `collect_status_ledger` over the whole root **0.28 s** — 0.6 s of a 5,651 s job. The real unit of work is `(problem × contrast × metric)`, and that count is **identical at Stage C and C2** (70 × 2 × 3 × 14); only the sample size inside each bootstrap grows, 3 seeds → 20, which costs **2.6 %**. 95 % of the wall was `cohens_d_ci_bootstrap` — a **Python loop over 10,000 resamples** of an array of length 3–20, so pure interpreter dispatch (0.1066 s at n=3, 0.1094 s at n=20). **Vectorised bit-identically** (`Generator.choice(a, size=n, replace=True)` delegates to `integers`, and one blocked draw consumes the PCG64 stream in the same order): 51× at n=20, 128× at n=3, 36 tests including a 200-input random sweep and a verbatim copy of the original loop as the oracle. Separately, `orchestrator.py:555` rebuilt `status_ledger.csv` from a **full recursive walk of the whole root, once per config** — 14 identical walks writing one shared path, and a race the moment they run concurrently. Hoisted behind `--no-status-ledger` into one dependent `--postprocess ledger` job, which lets the 14 configs run as an **array**. **All fourteen now take 19 s locally against ~590 s, and all 841 artefacts are BYTE-IDENTICAL to the old path's on the real 1,260-run corpus.** Consequence: `AGG_WALL` drops to **1 h 59 m**, which is under `short`'s MaxWall and therefore worth 20× the QOS priority — the one place in C2 where trimming a wall buys anything. ⚠ **Carry:** `compute_paired_stats` is **not** bit-reproducible between Picasso and the workstation (different scipy/BLAS; 4 of 841 files differed on borderline inputs). Do not diff Picasso-written artefacts against locally regenerated ones and read it as a code change — it cost an hour here | Claude |
| 2026-08-05 | 🔴 **Campaign SLURM logs defaulted to HOME, which cannot hold them** | T17-HANDOFF §3.3 states the campaign logs are *"already redirected to FSCRATCH"*. **They are not** — neither in the code (`c2_smoke/launcher.sh:32` and `c2_stage_d/launcher.sh:30` both default to `$HOME/execs/…`) nor on disk (no FSCRATCH log directory existed, and Stage D duly wrote to HOME). The redirection was an env-var override nobody applied: the `ISALSR_LEDGER_ENABLED` / `shadow_hash` trap a third time. HOME's file quota is **35.0k soft with 13.5k used**, of which `~/execs` alone is 10.7k; C2 writes two log files per task — **16,800 at 20 seeds (86 % of the quota), 25,200 at 30 (over it)**. Now a launcher **default**, not a habit. With logs on FSCRATCH the binding constraint moves: 166.4k / 250.0k soft leaves 83.6k, against ≈45k results + 16.8k logs at 20 seeds (**fits, 21.8k spare**) and ≈67k + 25.2k at 30 (**exceeds soft by 8.6k**, hard is 400k). Archiving the superseded roots recovers little (`c2_smoke_v3`/`v4` are 7,932 files each), so **30 seeds needs the ≥15,000-file support request T17-HANDOFF already lists as outstanding** | Claude, `quota` + launcher source |
| 2026-08-05 | **Launcher unified: one code path, two profiles; Stage C's memory deviation withdrawn** | `C2_PROFILE={smoke,campaign}` selects a coherent *group* of defaults (seeds, payload budget, wall, root, slot budget) instead of eight independent env vars that each silently default to the smoke value. Everything structural — topology, slot apportionment, memory, aggregation shape — is now shared, so the owed clean Stage C wave certifies **the campaign's launcher** rather than a cousin of it (§1: "certifying a topology you will not launch certifies nothing"). ✅ The Stage C memory deviation recorded 2026-08-03 is **withdrawn**: it existed only because §3.3's 256 GB across 210 concurrent smoke tasks would have measured fat-node availability rather than core contention, and at 32 GB there is nothing to deviate from. Verified: `sbatch --test-only` **42/42 accepted** at the new production shape (8,400 tasks, 2,016 slots, throttles %46–%80, 32 GB, 16 h, `--constraint=sr`); full unit suite **7,031 passed** with zero regressions (the 2 failures are T09's untracked `test_appendix_d_generator.py`); ruff and `mypy --strict` clean. **Nothing deployed** — Stage D's arrays are reading `00635ae` (defect 10); this goes out with the owed wave. Revised projection: **36 h at 20 seeds, 54 h at 30**, against 140 h and 210 h. Write-up: `T17-appendix/capacity_optimisation_worklog.md` | Claude |
| 2026-08-05 | ✅ **C2 moves to 30 seeds (Mario) — 12,600 runs, ≈63 h planned / ≈54 h after a Day-1 rebalance** | **§0.4a's 20-seed decision is superseded.** It rested on a cost premise the measured runtimes reversed (§11.1 2026-08-05): 30 seeds is ≈108,000 core-hours, and the apportioned throttle turns that into a **63 h makespan at 2,016 slots**, comfortably inside the 2026-09-03 target. **§6.3's disclosure paragraph about reduced supplementary-table power is no longer needed**, and C1's seed count is restored. ⚠ **Correction to carry:** the "54 h" first quoted was the *oracle* figure — the makespan when the allocation is weighted with the true `T_bingo`. The shipped plan weights at `T_bingo = 8 h` on purpose (F-19 raised three suites' `max_evals` tenfold; the 20 D2 problems have no runtime data), which costs 63.0 h rather than 54.1 because UDFS is left slightly under-slotted while the Bingo arrays drain 22 h early. `c2_slot_plan --table` now prints **both**, labelled — the allocation basis and the expectation — because quoting the pessimistic one as a forecast pads the schedule invisibly and quoting the oracle one as a plan does the reverse. **The 9 h are recoverable in flight:** `scontrol update JobId=<id> ArrayTaskThrottle=<n>` re-apportions a *running* array (verified on Picasso's SLURM 25.05.1), touches no config and no deployed file — so it is **not** defect 10 — and `c2_slot_plan --bingo-hours 5.15 --rebalance job_ids.txt` emits the 42 lines and returns the campaign to **54.1 h at 1,998 mean cores**. Launch weighted high, read the first day's Bingo wall clocks, rebalance. 🔴 **`n_seeds` in all 14 configs moved 20 → 30 at the same time**: `orchestrator.py:641` reads it as a **live fallback** whenever `--seeds` is absent, so a manual re-run, a resume or an analysis script would silently have got 20 — the `ISALSR_LEDGER_ENABLED` / `shadow_hash` shape a fourth time. Verified through the YAML loader, not the file text; locked by `test_campaign_seed_count_is_declared_by_every_config`. `config_sha256` changes for all 14, which is why it was done between waves; the owed clean Stage C wave certifies them. **No SLURM wall was raised**: 63 h is a *campaign makespan*, not a task duration, and one task is one run — 16 h stays correct, and raising it would only let a hung task burn 54 h of an allocation instead of 16 | Mario + Claude |
| 2026-08-05 | 🔴 **30 seeds is BLOCKED on FSCRATCH inodes until ~13.4k are freed — and the launcher now refuses instead of finding out at hour 40** | Measured on `c2_smoke_v4`: 7,932 inodes for 1,260 runs, of which 843 are the fixed per-`(problem, contrast)` artefacts ⇒ **5.63 inodes/run + 843 + 2 SLURM log files per task**. At 30 seeds that is **96,981 against 83,600 free** (166.4k / 250.0k soft) — **short by 13,381**. At 20 seeds it fits with 18.7k spare. Two measured remedies, together +29.7k, both Mario's call because both touch his data: `conda clean -a -y` (~21,800 inodes of package cache) and tarring the superseded `c2_smoke_v3` (~7,900; v1/v2 are already `.tar.gz`). After both, 16.4k spare. The 400k *hard* limit means the campaign would probably survive on GPFS grace — a gamble on a grace period nobody has measured, for a fix that is one command. **New guard `check_inode_budget()`** runs before the first `sbatch` on both `submit` and `--test-only`, projects from the measured coefficients against the live quota, and refuses with the exact shortfall and the two commands. This is P6's stated failure mode — *"C2 hits the hard file quota mid-campaign and every running task keeps burning wallclock while all its writes fail"* — turned from an hour-40 discovery into a two-second refusal. 🔴 **The guard's own first version reported "would submit" for the 30-seed request**: Picasso's `quota` separates the space and file halves with a literal `║`, so positional fields straddled the divider and yielded `used = 0` — a check passing while measuring nothing, the same shape as *both* C1.11 defects. Rewritten to split on the divider and to **fail closed** on an implausible parse; verified on the cluster against the live quota with a negative control that feeds it garbage and confirms it still refuses | Claude |
| 2026-08-05 | **Plan reconciled to 30 seeds — eight live contradictions, two of them binding rules** | **Raised by Mario: the configs said 30 while the plan still said "We launch at 20 seeds. Full stop."** Changing the configs without the decision record is how the next agent reverts them, so the plan is now consistent end to end. §0.4a's boxed note carries a superseding block and the original is retained beneath it; §6.3 is **withdrawn** rather than edited. **Two were live rules, not prose:** 🔴 **SP-0's probe cap said "seed 0 only (never 1…20)"** — at 30 seeds a probe at seed 25 would have read as legal and collided with a campaign cell; widened to 1…30 (Stage C's 0/101/102 remain disjoint, checked). 🔴 **§11.2's A5 row still read PASS** against a seed set that no longer exists — **reopened**, because a sign-off that certifies the wrong thing is worse than an open one. Also fixed: §7's "restrict C1 to seeds 1…20" (no restriction is needed now — C2 spans C1's full set, so the continuity comparison is *easier*), §10.5's instruction to every ticket agent, A5's pre-flight text, §8.1's budget (8,400/100,800 → **12,600/≈108,000**, both columns moving in opposite directions), §8.4 (spillover priorities 1 and 2 **absorbed**; Gray promoted and rescaled to 4,200 runs) and §11.3's launch ledger (per-array `%K`, verified row-by-row against the generator — **two of the seven I first typed were wrong**, taken from the 20-seed run). ✅ **The best consequence is for the letter**: R1 endorsed *"50 problems, 30 seeds"*, and §6.3 had obliged us to explain why we cut a seed count the reviewer praised. We now deliver 30 seeds **and** N = 70. **Do not write that disclosure paragraph.** | Mario + Claude |
| 2026-08-05 | 🔴 **`bingo_hard_trace.yaml` silently broke its own one-key invariant** | **Also raised by Mario, as "worth confirming".** It was not a missed edit *or* a deliberate exception — it was a contract violation with nothing enforcing the contract. The trace config's header states it is *"`bingo_hard.yaml` with ONE key changed: `shadow_hash: true`"* and its body adds *"if that file changes, this one must change with it"*. Moving `bingo_hard.yaml` to `n_seeds: 30` left the trace at 20, so it differed by **two** keys and nothing failed — `test_budget_uniformity.py` deliberately excludes trace and probe configs from `CAMPAIGN_SUITES`, so no test covered it. It matters because the split exists precisely so the certification cells and the traced cell differ **visibly and only** in `shadow_hash` (audit.md §7.3); a second drifting key makes `config_sha256` stop meaning "this is the shadow variant". Set to 30 — inert there, since `c2_stage_d/worker.sh:246` passes `--seeds 102` explicitly — and the invariant is now **enforced**, not merely documented, by `test_c2_slot_plan.py::test_trace_config_differs_from_its_parent_by_one_key`, which flattens both YAMLs and asserts the difference is exactly `{bingo.shadow_hash}` | Mario + Claude |

| 2026-08-05 | ✅ **STAGE D COMPLETE — GO, 8/8, and it resolves the C5 §3.5 ρ handoff** | **13/13 cells COMPLETED, zero failures**, on `00635ae` with clean provenance (one `build_hash`, two `config_sha256` = 8 certification cells + the trace, exactly as the shadow split intends). UDFS landed at **12:00:12** — saturating `max_time` to the second, as C1's n=600 predicted. **D1.4 is the one that mattered and it passes: Korns-12 and Vlad-2 Bingo-isalsr both finite** (R²_test −0.0217 and 0.9940). The C1 NaN does not recur, so the root cause is dead rather than dormant. **D1.6 answers what C5 §3.5 handed forward.** Stage C's 900 s wave put Bingo ρ **1.1–1.7 % below C1**, and Mario signed C5 with that deviation deferred to D1.6. At 12 h against 12 h the gap collapses: ratios `ρ_C2/ρ_C1` = **0.9973** (Pagie-1), **0.9977** (Vlad-2), **1.0013** (Korns-12), and UDFS Pagie-1 **rose to 1.0656**. Zero excursions against a one-sided 10 % floor. **The Stage C shortfall was the budget gap, exactly as hypothesised** — not a canonicaliser regression. The ρ reconstruction (`1 + δ_ρ`, since the baseline arm reports ρ = 1 by construction) was **cross-checked, not assumed**: it reproduces the published `mean_reduction_factor` to `abs_gap` 0.0005 (bingo) and 0.0 (udfs) over 50 problems. R²: 0 excursions on 4 comparisons, largest 0.115 against a 0.15 band. **D1.2 independently corroborates the 32 GB decision.** Its own rule — `ceil_to_8GB(peak / 0.70)` — recommends **8 GB for every group including `bingo/isalsr`** (peak 1.193 GB). The shipped 32 GB is **4× the certifier's recommendation and 27× the observed peak**, so the reduction from 256 GB is conservative under two independent derivations. **D1.7** overhead under the corrected canon+conversion accounting: Bingo mean **7.83 %** of eval (p50 8.7 %, max 17.7 %), UDFS **0.027 %** — above the old canon-only ≈7.4 % projection, which §11.1 2026-08-04 already flagged as an accounting change rather than a regression. 🔴 **One reporting defect worth carrying: the Picasso certifier printed `GO`, `n_blocking_failures: 0` while D1.6 — whose own rules say BLOCKING, twice — had status `SKIP` and `blocking: false`.** Its C1 reference is `/media/.../Sandisk2TB/...`, a workstation path no compute node can reach, so it could never have evaluated. The report does say SKIP rather than claiming a pass, but the headline hides it: a reader stopping at the verdict banks an unevaluated blocking criterion. **Either ship the C1 analysis directory to FSCRATCH before Stage D, or run the certifier where C1 lives.** Same family as the two C1.11 defects and the inode guard: a check that degrades to advisory when its input is missing | Claude, `stage_d_certification_with_c1.json` |

### 11.2 Pre-flight sign-off

| Stage | # | Check | Date | Result | Evidence artefact |
|---|---|---|---|---|---|
| A | A1 | Commit frozen (`campaign/c2`) | 2026-08-03 | **OPEN, deliberately.** The tag must sit on the commit C2 will run, and the code is not final (C3 unimplemented, `n_seeds` stale in 10 configs). Stage C runs on recorded commits instead (`53a1c1c` → `5f282cc`) | — |
| A | A2 | pytest / ruff / mypy clean | 2026-08-03 | **PASS with one scoped caveat.** `pytest tests/` **6,818 passed, 5 skipped**; `mypy --strict src/isalsr/` clean (55 files); `ruff` clean on `src/` and `tests/`. `experiments/models/` carries **444** violations (N806 `X_train`-style names, E501) — **identical at HEAD**, so entirely pre-existing and none introduced | raw output |
| A | A3 | Backend parity, `.so` freshness | 2026-08-03 | **PASS.** 2,000 random DAGs, **0 backend mismatches, 0 errors**; `.so` mtime post-dates the last native commit at the site-packages path; `isa_level=x86-64-v3`, `avx512f=0` | local + `verify_build.py` |
| A | A4 | Config equivalence across arms | 2026-08-03 | **PASS + finding.** No arm block overrides a host-search hyperparameter; the top-level `isalsr:` block holds only canonicaliser settings. 🔴 **10 configs still declare `n_seeds: 30`** — fix before C2 | `c2_preflight/config_diff.md` |
| A | A4b | Operator-set policy decided (uniform per method) | 2026-08-03 | **Decided.** Configs updated; containment guard landed. Still to do at sign-off: dump `operator_sets.csv`, record the policy **and the 22-problem continuity exclusion** in the MANIFEST | `c2_preflight/operator_sets.csv` |
| A | A5 | Seed set declared (1…30, = C1's set) | 🔴 **REOPENED 2026-08-05** | The 2026-08-03 **PASS** certified `1…20` and no longer describes the campaign (§0.4a superseded). Must be re-signed against **1…30**: `0 ∉ seeds`, Stage C's 0/101/102 still disjoint, all 14 configs declaring `n_seeds: 30`, and the seed integers coinciding with C1's so §7 needs no subsetting. Re-run before the `campaign/c2` tag | `c2_preflight/seed_declaration.md` |
| A | A6 | MANIFEST schema frozen + validator | 2026-08-04 | ✅ **CLOSED.** `experiments/models/manifest.py` + strict validator: nothing defaults, `from_dict` rejects any missing top-level or nested field, and the CLI exits 1 on a truncated manifest (the graded criterion). Records commit+tag, build hash, compiler+flags, per-`(method, suite)` `config_sha256`, the A4b policy **with the 22-problem Bingo continuity exclusion enumerated**, three arms, seeds 1–20 (seed 0 rejected), `decomposed`, engine, `sr`, 42 arrays / 8,400 tasks. 46 tests | `tests/unit/test_manifest.py` |
| A | A7 | RunLog accepts three arms | 2026-08-04 | **PASS.** Three arms round-trip; the provenance block is A7-BUG's. Extended since by the fairness audit: `conversion_time_s`, `shadow_time_s` (cost attribution) and `penalised_in_population_{mean,max}`, all legacy-tolerant. **Certified field count 56 → 60** | `tests/unit/test_schemas.py`, `test_cost_attribution.py` |
| A | A8 | Analyzer three-arm readiness | 2026-08-04 | ✅ **CLOSED.** `analyze.py` takes `--variants baseline,hash,isalsr` end to end; Friedman/Nemenyi over three arms; pairwise CPDT over the three contrasts with **Holm dividing by 3, asserted directly** (`3 × 0.01`, explicitly not `2 × 0.01`); conservative-substitution sensitivity check; three-arm table emission. Contrast policy and ρ handling: audit.md §6.1/§7.3 | `tests/unit/test_three_arm_stats.py`, `test_cpdt_contrasts.py` |
| A | A9 | T08 code half landed | 2026-08-04 | **PASS**, and hardened by the audit: NaN can never be marked better (regression-tested), pairwise deletion reports the true per-metric `N`, and the conservative-substitution check now exists (it did not when T08 closed — audit F-14) | `test_table_nan_integrity.py` |
| A | A10 | Failure ledger implemented | 2026-08-03 | **PASS** (P4). Write-ahead `status.json` survives an OOM `SIGKILL`; `collect_status_ledger` emits `status_ledger.csv`; `reconcile()` names missing cells. Exercised at scale: Stage C 1,260 rows / 1,260 distinct cells | `c2_smoke*/status_ledger.csv` |
| A | A11 | Hash-collision bound stated | 2026-08-03 | **PASS, pending one input.** `n²/2⁶⁵ = 2.7×10⁻⁶` per run at `n = 10⁷`; `≈1.5×10⁻²` expected across 5,600 dedup-bearing runs. The bound is **quadratic in n** and was stated at an *assumed* `n`; re-evaluate against Stage C's measured `max(total_dags_explored)` | `c2_preflight/collision_bound.md` |
| A | A12 | SLURM array/job limits | 2026-07-31 | **PASS** — `MaxArraySize=4096` ≥ 1,401; no chunking | `scontrol show config` |
| A | A13 | 🔴 Storage **and file-count** headroom (see P6) | 2026-08-03 | 🔴 **Stage C PASS, campaign FAIL.** FSCRATCH 222.8k/250.0k soft: Stage C needs ≈7.9k and fits; C2 needs ≈45k and does not (**27.2k headroom vs the ≥60k criterion**). HOME 0.43/0.28 TB, **2 days grace**, of which **436 GB is `~/execs/vena`** (a different project) | `c2_preflight/storage_projection.md`, `quota_capture.txt` |
| B | B1 | Environment probe | 2026-08-03 | **PASS.** **70/70** dataset paths resolve with the declared shapes and **70/70** carry a SymPy ground truth, so C1.5's precondition is met *before* Stage C | `stage_b/*/b1_environment.json` |
| B | B2 | C++ capability probe + negative control | 2026-08-03 | **PASS, genuinely two-sided.** Run 1 `engine=cpp`, `cpp_invoked=True`; run 2 under `ISALSR_ENGINE=python`, `engine=python`, `cpp_invoked=False`. Asserted on **observed dispatch** (spy on `_cpp_ext.fast_canonical_string`), both hosts | `stage_b/*/b2_sp_probe_{cpp,python}.json` |
| B | B3 | Alphabet gate on frozen commit | 2026-08-03 | **PASS.** 65,631 live Bingo candidates: 0 `SUB`, 0 `DIV`, 0 `-`, 0 `/`; `POW` the only binary; NEG 70,622 / INV 47,821 present ⇒ the T16 decomposition **is** reaching the canonicaliser. **Max k = 37.** ⚠ Under a *bounded* budget Bingo solves Nguyen-1 before any candidate reaches the canonicaliser (`DAGs observed: 0`), so B3's Bingo problems must be structurally hard — Pagie-1 and I.29.16 | `stage_b/*/b3_alphabet_gate_*.json` |
| B | B4 | Equivalence gate on a compute node | 2026-08-03, **gate 3 re-measured 2026-08-05** | **PASS, now on all three gates.** 2026-08-03 on Picasso (gcc 13.2.0, `self_comparison=false`): gate 1 **54,765** comparisons / 0 mismatches, gate 2 10,000 / 0 errors, ~~gate 3 fails 5/10,000 identically on both engines~~. ✅ **Gate 3 re-run on `3d5a79c`: 10,000 DAGs / 20,000 comparisons / 0 mismatches on BOTH engines**, and the corpus contains **15 over-saturated binary nodes** so the check is not vacuous. The five were **spurious failures of the checker**, not of the representation: T18 narrowed `is_isomorphic` to `ordered_inputs(v)[0]`, which is all Σ_SR encodes, so the old whole-list comparison was strictly finer than the canonical string. §11.1 2026-08-05 has the full reconciliation. ⚠ Re-measured **locally**, not on a compute node — the 2026-08-03 Picasso run remains the cross-engine authority; re-run gate 3 on Picasso with the owed Stage C wave to close this at the same provenance | `stage_b/b4/b4_equivalence_gate.json`; `T07-appendix/gate_all_3d5a79c_2026-08-05.json` |
| B | B5 | Node-pool census | | **Arrives as a by-product of Stage C**: every run records its own `cpu_model` (A7), so the reachable node distribution and the arm balance are computed from the 1,260 run logs rather than from a separate array | |
| B | B6 | Node-constraint decision | 2026-08-04 | ✅ **CLOSED: pin `--constraint=sr`** (AMD EPYC 7H12). Two independent reasons, either sufficient: **(i)** data generation is not bit-reproducible across families (~1 ULP), which split 35/210 cells in the unpinned v2 wave and fails C4; **(ii)** wall clock is a *reported* quantity and `sd` 2.1 GHz vs `sr` 2.6 GHz makes a mixed pool partly a measurement of the scheduler. `sr` is the largest pool (154 × 128 = **19,712 cores**, >2× the `cpu=9000` entitlement) and matches the engine's `x86-64-v3`/`avx512f=0` build. **Verified: 1,260/1,260 v3 tasks placed on `sr`, and throughput rose to 592 cores** | v3 wave, `sacct NodeList` |
| B | B6b | AVX-512 portability of the C++ engine | 2026-08-03 | **PASS.** Built with `gcc/13.2.0`; `build_hash = 298fc1188bf1b051` **identical to the local gcc 12.2.0 build**; `isa_level=x86-64-v3`, `avx512f=0`; imports with every module purged ⇒ portable across `sd`/`sr`/`bc`/`bl` | `verify_build.py` output |
| B | B7 | `sbatch --test-only`, **all 42 arrays** | 2026-08-03 | **PASS.** exit 0 on 42/42, task counts exactly 210 per `(method, arm)`, 1,260 total | launcher `--test-only` |
| B | B8 | Resume / idempotency | 2026-08-04 | ✅ **CLOSED, both halves, locally and on Picasso.** Jobs **1762279** (fresh, 3,951 B) → **1762282** ("already exists", byte-identical, skipped) → truncated to 1,975 B → **1762284** ("Corrupt run_log detected, re-running" → valid 3,951 B). Two cases beyond the ask: zero-byte file, and valid JSON that is not a RunLog | `tests/integration/test_resume_corruption.py` (6) |
| B | B9 | T06 counter re-verification (threshold from T06) | in flight | Paired 240 s runs with and without `--ledger`, both hosts. **C1.9 already confirms liveness on real Picasso cells** (`ledger_enabled=true`, `n_ledger_sampled > 0`, 3/3) | `stage_b/*/b9_overhead.json` |
| C | — | **The 1,260-task wave itself** | 2026-08-03 | **1,260/1,260 COMPLETED, 0 failed**, 20:28:54 → 21:46:02 (77.1 min). Achieved concurrency ≈245 cores — **but see the 2026-08-04 correction in §11.1: that is 73 % of our own `%8` throttle, not a cluster limit** | jobs 1752689–1753133 |
| C | C1.1–C1.9 | exit codes, schema, NaN, shapes, ground truth, ρ, hash sanity, baseline purity, ledger liveness | 2026-08-04 | **ALL PASS.** C1.1 1260/1260; C1.2 all 56 fields; C1.3 **0 non-finite**; C1.4 70/70; C1.5 70/70; C1.6 420/420 with ρ≥1 and **100 % ρ>1**; C1.7 **0/420 violations**; C1.8 420/420; C1.9 840/840 counters live | `stage_c_certification.json`, job 1758634 |
| C | C1.10 | trajectory non-empty, monotone | 2026-08-04 | 🔴 FAILED 808/1,260 in v1 → **FIXED and RE-CERTIFIED `1260/1260 PASS` in v2.** 100 % of v1 violations sat at the final row: `best_r2` carried train R² throughout and test R² on the last row. Both translators corrected; 5 regression tests, verified to fail 4/5 pre-fix | jobs 1758604 → 1759722 |
| C | C1.11 | memory profile → production `--mem` | 2026-08-04 | ⚠ **v1 PASSED on 42 of 1,260 rows — a `JobID` vs `JobIDRaw` join bug, fixed.** v2 reports **1,260 observations**. Peak `MaxRSS` **0.67 GB**. **Does not size production — D1.2 at 12 h does** | `stage_c_maxrss.csv` |
| C | C1.12–C1.17 | wall limit, alphabet, engine, reconciliation, paired stats, aggregate | 2026-08-04 | **ALL PASS.** C1.12 **0 SLURM time-kills**; C1.13 **0 forbidden labels**; C1.14 1260/1260 `native`; C1.15 1,260 reconciled, 0 unnamed; C1.16 **420/420** contrasts valid at `n_seeds == 3`; C1.17 420/420 × 14 metric rows | `stage_c_certification.json` |
| C | C2 | Failure ledger emitted | 2026-08-04 | **PASS.** `status_ledger.csv` at the root: **1,260 rows, 1,260 distinct cells** | `c2_smoke/status_ledger.csv` |
| C | C3 | Dedup-off equivalence control | 2026-08-03 | **UDFS PASS outright** (313/313 identical candidate-stream prefix; `r2_train` bit-identical). **Bingo: criterion untestable — the baseline does not reproduce itself** (155,449 / 41,023 / 41,049 candidates at one seed). Residual **bounded**: wrapper perturbation 3 inner evals vs baseline self-noise 12. Accepted by Mario; **must be stated in the paper** | T17 §6 |
| C | **ALL** | **Stage C re-certified on `c2_smoke_v3`** | 2026-08-04 | ✅ **VERDICT: GO — 19/19 criteria PASS, 0 blocking failures.** Wave 1,260/1,260 COMPLETED, **1,260/1,260 on `sr`**, span 31 m 55 s (592 cores). C1.10 **1260/1260**; C4 **PASS** with `cross_arm_disagreement=0`, `duplicate_problems_blocking=0`, `seed_collapse_blocking=0`, `wrong_multiplicity=0`, multiplicity histogram `{6: 204, 18: 2}` — the signature of a correct campaign. Aggregation completed in 1 h 35 under the new 6 h wall | `c2_smoke_v3/c2_preflight/stage_c_certification.{json,md}`, job 1761777 |
| C | C4 | Cross-arm data identity | 2026-08-04 | ✅ **RESOLVED — see the row above.** Was the one remaining blocker; two independent findings, both now fixed.| **(a)** `I.12.1` ≡ `I.34.27`: both reduce to `x_0*x_1` on `[1,5]²` and generate byte-identical data, a genuine **D1** duplicate inflating `N` since C1. **(b)** The data is **not bit-reproducible across CPU families**: v2 showed 35/210 pairs split, **all 35 partitioning exactly by node family, 0 exceptions**, measured at **≤1 ULP (5.55×10⁻¹⁷)**. Not a config race (`config_sha256` identical), not non-determinism (8 draws → 1 fingerprint). **Recommendation: pin the node family — it fixes C4, removes the wall-clock confound, and closes B6.** Pagie-1/Keijzer-6 seed-invariance is expected and now a declared exemption | `c4_fingerprint_findings.md`, probe jobs 1760654/1760655 |
| C | C5 | Comparison against C1 | 2026-08-04 | ✅ **SIGNED (Mario).** Every row MEETS except **§3.5**: Bingo ρ at 900 s sits **1.1–1.7 % below** C1 where §2 expected a rise. The dangerous cause is falsified directly — SP-4 finds **0** SUB/DIV in 292 canonical strings — leaving the 48× budget gap, **handed to D1.6** (12 h vs 12 h). §2 untouched | `c2_preflight/smoke_vs_C1.md` |
| D | D1.1–D1.8 | 12-task full-length certification | 2026-08-05 | ✅ **GO — 13/13 cells COMPLETED, 8/8 criteria PASS, 0 blocking failures.** D1.1 min headroom 25.0 % of the 16 h wall; D1.2 min headroom 96.2 %; D1.3 12/12 artefact sets; **D1.4 2/2 finite — the C1 NaN does not recur**; D1.5 0/4 ρ violations; **D1.6 0 ρ and 0 R² excursions over 8 comparisons**; D1.7 12/12 with T_canon and T_eval > 0; D1.8 manifest validated. ⚠ D1.6 came back **SKIP** on Picasso — its C1 reference is a workstation path the compute node cannot see — and was re-run locally to a PASS. Evidence: `c2_stage_d/c2_preflight/stage_d_certification{,_with_c1}.{json,md}` | `stage_d_certification_with_c1.md` |
| D | D2 | Detailed single-problem trace | 2026-08-05 | **RUNNING as cell 13.** Bingo × Pagie-1 × isalsr at **seed 102** under `bingo_hard_trace.yaml` (`shadow_hash: true`), sampling 1 candidate in 100 (571.7 B/record measured). Split from cell 10 so the certification cells stay clean — audit.md §7.3 | `c2_trace/` |
| D | D3 | T04 Mode 1 replay + soundness | 2026-08-06 | ✅ **PASS, and it earned the whole stage.** First run returned **exit 2** on 37 IsalSR soundness violations and thereby found the **k=0 completeness defect** (§11.1, 2026-08-06) *before* any campaign core-hours were committed — §0.4d's stated purpose, discharged. After the structural-scope fix: **`ok=true`**, hash soundness **0 unsound merges across all three fixed orders** (the veto that could have killed the hash arm), replay fidelity **0 digest and 0 canonical mismatches over 119,795 records**, IsalSR soundness **376 pairs / 0 failures** with the **71** k=0 exclusions *declared in the report* rather than silent. ρ ladder orders correctly: ρ_total 1.0237 ≤ ρ_exact 1.0237–1.0251 ≤ ρ_iso 1.0254. ⚠ **These absolute values are NOT comparable to Stage D's ~1.83** — the trace is a deterministic 1-in-100 subsample, which destroys most duplicate pairs; the script says so itself and only the *ordering* is meaningful here | `c2_stage_d/c2_preflight/stage_d3_mode1_replay.{json,md}` |
| E | E1–E7 | Analysis dry-run on 3 arms | 2026-08-05 | ✅ **VERDICT: GO — 7/7 checks PASS**, 182 s end to end on `c2_smoke_v4` (1,260 runs, 3 arms × 3 seeds). **E1** 103 artefacts, all 6 families, all 3 arms, 49 s ⇒ ≈5.5 min at C2's 8,400. **E2** 3 contrasts × 5 metrics × 16 files; audit §6.1 policy holds on real data — ρ `descriptive` vs baseline, two-sided where unregistered, ρ inferential only on `isalsr_vs_hash` with a Holm family of 1; Friedman over 6 groups. **E3** injected NaN never bolded, paired N 3→2, table discloses `[2]`, conservative substitution present. **E4** 6 tables emitted, **6 compiled**, zero `nan` cells. **E5** all 10 CD diagrams at 6 groups. **E6** refused with exit 2, named `udfs/nguyen/nguyen_1/baseline/seed_00`, 395/396 reconciled. **E7** refused with exit 2, named `build_hash`/`config_sha256`/`git_describe`. **Four defects found and fixed — three invisible to an exit code** (§11.1, 2026-08-05). ⚠ **Re-run on v5 and it must pass E7 without `--allow-mixed-provenance`, or the tag must not be cut** | `c2_stage_e/artefacts/stage_e_certification.{json,md}`, `docs/md_files/changes/stage_e_design.md`, `slurm/c2_stage_e/RUNBOOK.md` |
| F | — | Sign-off (Mario) | | | |

### 11.3 Launch ledger

**42 arrays**, one per `(method, arm, suite)` — see §1 and §11.1 (2026-08-03).
Task counts are `suite_size × 30` for C2 and `suite_size × 3` for Stage C.
Fill at submission; the launcher writes the job ids to
`<logs>/job_ids.txt` in submission order, which is the order below.

**Updated 2026-08-05 for 30 seeds.** Throttles are no longer uniform: they are
apportioned by each array's work (`c2_slot_plan.py`), which is a 1.9x makespan
gain at the same total slots. The `%K` column below is the campaign's, at the
default 2,016-slot budget.

| Suite | Problems | C2 tasks/array | C2 `%K` | Stage C tasks/array |
|---|---|---|---|---|
| nguyen | 12 | 360 | 69 / 46 | 36 |
| feynman | 10 | 300 | 58 / 38 | 30 |
| hard | 10 | 300 | 58 / 38 | 30 |
| cherrypicked | 10 | 300 | 58 / 38 | 30 |
| roundoff | 8 | 240 | 46 / 31 | 24 |
| feynman_remainder | 6 | 180 | 35 / 23 | 18 |
| strogatz | 14 | 420 | 80 / 54 | 42 |
| **per (method, arm)** | **70** | **2,100** | | **210** |
| **total (× 6)** | | **12,600** | **2,016** | **1,260** |

`%K` is given as **UDFS / Bingo** — the two differ because UDFS runs 12.00 h per
task and Bingo ~5.15 h, so UDFS needs proportionally more concurrent slots to
finish alongside it. Regenerate with
`python -m experiments.scripts.c2_slot_plan --seeds 1-30 --table`.

| # | Method | Arm | Suite | Tasks | Job ID | Submitted | Completed | Failed | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1–7 | UDFS | baseline | nguyen … strogatz | 240/200/200/200/160/120/280 | | | | | |
| 8–14 | UDFS | hash | nguyen … strogatz | 240/200/200/200/160/120/280 | | | | | |
| 15–21 | UDFS | isalsr | nguyen … strogatz | 240/200/200/200/160/120/280 | | | | | |
| 22–28 | Bingo | baseline | nguyen … strogatz | 240/200/200/200/160/120/280 | | | | | |
| 29–35 | Bingo | hash | nguyen … strogatz | 240/200/200/200/160/120/280 | | | | | |
| 36–42 | Bingo | isalsr | nguyen … strogatz | 240/200/200/200/160/120/280 | | | | | |
| 43 | — | — | aggregation (`--postprocess only`, `afterany`) | 1 | | | | | |
