# Work log — C2 capacity and schedule optimisation

**Session**: 2026-08-05, Stage E complete, Stage D still running (5 of 13 cells).
**Brief**: `capacity_optimisation_brief.md`.
**Submitted to Picasso**: nothing that executes. All live work was `sbatch
--test-only` (which allocates nothing) plus read-only `sacct`/`sinfo`/`sacctmgr`/
`quota` queries. Everything else was computed locally from artefacts already on
disk. **SP-0 held.**

---

## 0. Standing property table (SP-1…SP-6)

No code was executed on a compute node this session, so the SP-1…SP-6 probe does
not apply in its usual form. The row that *is* load-bearing is SP-1, because the
`--test-only` sweep named the deployed `worker.sh` and the deployed tree is the
one Stage D is running against.

| # | Property | State this session |
|---|---|---|
| SP-1 | Provenance | Deployed tree `00635ae`, `git describe = 00635ae`, working tree **clean** (read from the login node). Local tree `3d5a79c-dirty` — **not deployed, and must not be**: defect 10, Stage D's arrays are reading the deployed tree |
| SP-2 | Install freshness | n/a — nothing was built or executed |
| SP-3 | Engine + negative control | n/a — no search ran |
| SP-4 | Alphabet | n/a — no candidate stream produced |
| SP-5 | Both hosts | Both hosts covered *analytically* (UDFS and Bingo arrays both in the `--test-only` sweep and both in the makespan model). No host was executed |
| SP-6 | Live counters | n/a — no ledger population existed |

**Nothing in this log is a scientific number.** Every figure is a scheduler
property, a resource measurement, or a model over measurements already in
§11.1.

---

## 1. Headline

Five findings, ranked by size. **(1) and (2) together are worth ~3.7× on the
makespan and cost no science whatsoever.** (3) is a correction that removes a
day of planned wall clock. (4) and (5) are traps that would have bitten during
the campaign.

| # | Finding | Effect |
|---|---|---|
| 1 | **The uniform `%K` throttle is the dominant schedule loss, not contention.** 42 arrays carry unequal work but get equal slots, so the long arrays set the makespan while the short ones idle their share | **1.92×** at every throttle level. At `%24`, 140 h → 72.7 h |
| 2 | **Bingo-IsalSR's 256 GB request caps the arm at ~154 concurrent tasks and sterilises 115 of every node's 128 cores.** Stage D measures peak RSS at **1.14 GB** | Arm makespan 46.8 h → 4.7 h at 32 GB. The single largest structural ceiling, and it is now measured away |
| 3 | **The aggregation job does not cost ≈11 h — it costs ≈1 h 45 m, and ≈2 min if one loop is vectorised.** Its cost scales with *problems*, not runs, and 95 % of it is a Python bootstrap loop that is **n-independent** and **bit-identically vectorisable** | ≥24 h wall recommendation is 14× over-provisioned; the ledger's 6.7×-in-runs extrapolation is wrong |
| 4 | 🔴 **Every concurrency figure in §11.1 (245 / 476 / 592) was measured under the `short` QOS, which the campaign cannot use.** Measured today: raising the requested wall from 1 h 59 m to 2 h 01 m moves the scheduler's own start estimate from *now* to *+3 h* | The 592-core figure is not transferable to C2. Not a lever — a caveat that must travel with the schedule |
| 5 | 🔴 **All campaign SLURM logs default to HOME, which has 21.5k inode headroom.** C2 writes 16,800 log files at 20 seeds and **25,200 at 30** | 30 seeds is **blocked** by the HOME quota until the three superseded smoke-wave log trees (8,979 files) are archived. T17-HANDOFF §3.3 states logs are "already redirected to FSCRATCH"; they are not, in the code or on disk |

---

## 2. Finding 1 — the throttle **allocation**, not the throttle level

### 2.1 What the Stage C waves actually show

Reconstructed the exact concurrency envelope from `sacct` `Start`/`End` for both
`%24` waves. No new compute; the data was already on the cluster.

| | v2 (`%24`, `cpu`) | v3 (`%24`, `sr`) |
|---|---|---|
| tasks | 1,260 | 1,260 |
| span | 39.7 min | 31.9 min |
| core-hours | 247.4 | 247.4 |
| achieved concurrency | 374 | **465** |
| **peak** concurrency | 951 | 934 |
| ceiling (42 × 24) | 1,008 | 1,008 |
| peak as % of ceiling | 94 % | 93 % |
| **% of span at ≥90 % of ceiling** | **2.3 %** | **2.7 %** |
| mean concurrency, final 25 % of span | 62 | 67 |
| placement | 1,112 `sr` / 148 `sd` | **1,260 / 1,260 `sr`** (P-7 ✅) |

> My reconstruction gives 374/465 against the ledger's 476/592. The difference is
> the denominator: the ledger divides 315 nominal core-hours (1,260 × 0.25 h) by
> the span, I divide the 247.4 core-hours `sacct` actually records (many tasks
> finished under 900 s). Same wave; mine is the delivered figure. The relative
> conclusions are unaffected.

**The envelope is a spike, not a plateau.** The wave touches 93 % of its own
ceiling — so the scheduler *granted* ~934 concurrent 1-core `sr` slots within
minutes, and contention is not what bound it — but it holds that level for 2.7 %
of the span and decays 14× by the final quarter.

### 2.2 Why, and why it gets worse in C2

Two causes, and the second is the one that matters at campaign scale.

1. **Stage C has 1.25 tasks per slot.** 1,260 tasks against 1,008 slots is barely
   more than one task-duration of work. Ramp-up and drain are most of the span by
   construction. *This alone makes Stage C unfit to measure sustained
   concurrency at `%24` or above* — see §6.
2. **The arrays carry unequal work but get equal slots.** Array *i* under
   throttle *Kᵢ* finishes at ≈ *Nᵢ·Tᵢ / Kᵢ*. With a uniform *K* the makespan is
   set by the largest *Nᵢ·Tᵢ* while every smaller array returns its slots early
   and they are not reallocated — the other arrays are still capped at *K*.

At C2 scale (8.3 tasks per slot) cause 1 disappears and cause 2 becomes the
whole story.

### 2.3 The model, and the fix

`makespan.py` (scratchpad). Inputs are measured: UDFS **12.00 h/run** (C1 n=600,
mean = median = max, saturates `max_time`), Bingo **5.15 h/run** (C1 isalsr
n=564; Stage D's 8 completed cells give 4.57 h). 20 seeds ⇒ 8,400 runs,
**72,030 core-hours**.

| uniform | slots | makespan | mean cores | work-proportional | mean cores | gain |
|---|---|---|---|---|---|---|
| `%8` | 336 | 420.0 h (17.5 d) | 172 | 224.0 h (9.3 d) | 322 | 1.88× |
| `%24` | 1,008 | **140.0 h (5.8 d)** | 514 | **72.7 h (3.0 d)** | 990 | **1.92×** |
| `%48` | 2,016 | 70.0 h (2.9 d) | 1,029 | **36.0 h (1.5 d)** | 1,998 | 1.94× |
| `%96` | 4,032 | 35.0 h (1.5 d) | 2,058 | 17.9 h (0.75 d) | 4,014 | 1.95× |

At 30 seeds (108,045 core-h) the same table reads 210.0 h → **109.1 h** at
`%24`, and 105.0 h → **54.1 h** at `%48`.

Where the uniform throttle loses it, at `%24`:

| array | N | T | K uniform | finishes | K proportional | finishes |
|---|---|---|---|---|---|---|
| `udfs:*:strogatz` (×3) | 280 | 12.00 h | 24 | **140.0 h** | 47 | 71.5 h |
| `udfs:*:nguyen` (×3) | 240 | 12.00 h | 24 | 120.0 h | 40 | 72.0 h |
| `bingo:*:feynman_remainder` (×3) | 120 | 5.15 h | 24 | **25.8 h** | 9 | 68.7 h |

Three arrays finish in 26 h and then contribute nothing while three others run
for another 114 h. **The fix is `Kᵢ ∝ Nᵢ·Tᵢ`, one function in the launcher, same
total slots, no configuration content changed, no science touched.**

### 2.4 Sensitivity — it does not need `T_bingo` to be right

`T_bingo` is the uncertain input (F-19 raised three suites' `max_evals` 10×, and
20 D2 problems have no runtime data at all). Allocating with an assumed
`T_bingo = 8 h` and evaluating at the truth:

| true `T_bingo` | uniform `%24` | proportional (planned at 8 h) | proportional (oracle) | planned vs uniform |
|---|---|---|---|---|
| 4.00 h | 140.0 h | 84.7 h | 68.6 h | 1.65× |
| 5.15 h | 140.0 h | 84.7 h | 72.7 h | 1.65× |
| 8.00 h | 140.0 h | 84.7 h | 84.7 h | 1.65× |
| 10.00 h | 140.0 h | 105.3 h | 93.3 h | 1.33× |
| 12.00 h | 140.0 h | 126.3 h | 101.8 h | 1.11× |

**It never loses to uniform, anywhere in the plausible range**, and it wins
1.65× at the measured value. Plan at `T_bingo = 8 h`.

### 2.5 Recommendation

**Σ Kᵢ = 2,016 (mean `%48`), allocated proportional to `Nᵢ·Tᵢ` with
`T_udfs = 12 h`, `T_bingo = 8 h`.** Modelled makespan **36 h at 20 seeds, 54 h at
30**. This is 22 % of the `cpu = 9000` entitlement and 15 % of `sr`'s usable
13,312 cores. `sbatch --test-only` accepts `%24`, `%48`, `%96` and `%280` (§5).

The allocation change is unconditional — it is a 1.9× win at *any* grant level.
The slot raise from 1,008 to 2,016 is a bet on the grant, and §4 is why that bet
is less safe than §11.1 currently implies.

---

## 3. Finding 2 — Bingo-IsalSR memory, and the ceiling it imposes

### 3.1 The measurement (P-4)

`sacct` on the `.batch` step (never `-X`; `JobIDRaw`), plus the Stage D workers'
own RSS sampler:

> ⚠ **Written mid-stage. Superseded by §14, which has the final 13/13 figures.**
> The conclusion does not move — the peak rose from 1.105 GB to **1.193 GB**, and
> D1.2's own production rule ended up recommending **8 GB**, against the 32 GB
> shipped.

| group | completed cells | `MaxRSS` range | worst |
|---|---|---|---|
| `bingo_std` (32 GB req.) | 5 | 0.41 – 1.14 GB | 1.144 GB |
| `bingo_isalsr` (**256 GB** req.) | 3 | **1.05 – 1.11 GB** | 1.105 GB |

The workers' `max_rss_gb` field agrees: `{0.449, 0.463, 0.526, 1.073, 1.087,
1.103, 1.104, 1.14}`. Cells ran 3.0–6.9 h and stopped on `max_evals = 100M`, so
these are full-length dedup sets, not truncated ones.

**D1.2's rule — `max(MaxRSS, VmHWM) + 30 %` — gives 1.49 GB. The request is
256 GB: 172× the requirement.**

The 100× gap against C1's 127.7 GB OOMs is explained, not anomalous: the dedup
sets moved from `set[str]` to `set[int]` (~150 B → ~28 B per entry, CLAUDE.md),
and 12.95M candidates at ~30 % unique lands at a few hundred MB plus interpreter
and population — which is what 1.1 GB is.

### 3.2 What the request costs

| `--mem` | tasks/node (450 GB) | node-bound | QOS-mem-bound (`mem=50000000M`) | ceiling | batches over 1,400 | arm makespan | our own cores sterilised per node |
|---|---|---|---|---|---|---|---|
| **256 G** | **1** | 154 | 190 | **154** | 9.09 | **46.8 h** | **115 of 128** |
| 128 G | 3 | 462 | 381 | 381 | 3.67 | 18.9 h | 121 |
| 64 G | 7 | 1,078 | 762 | 762 | 1.84 | 9.5 h | 121 |
| **32 G** | **14** | 2,156 | 1,525 | **1,525** | 0.92 | **4.7 h** | 114 |

Two ceilings the plan has not stated:

- **`sr` hosts one 256 GB task per node, so 154 concurrent Bingo-IsalSR tasks
  consume all 154 `sr` nodes — 19,712 cores of hardware to run 154 tasks.** Live
  today: 128 of 154 `sr` nodes have ≥256 GB free, 137 have ≥32 GB free.
- **QOS `long_uma`/`medium_uma` carry `MaxTRESPU mem=50000000M` (47.7 TiB).** At
  256 GB that is a hard cap of **190 concurrent Bingo-IsalSR tasks** regardless
  of nodes, throttle or entitlement. This limit is not mentioned anywhere in the
  plan and it binds before the `cpu = 9000` limit does.

### 3.3 Recommendation

**256 GB → 32 GB**, which is 28× the measured peak and 21× D1.2's rule. Memory
then stops being a constraint on any axis.

**Three caveats, and the third is why this is a recommendation and not a
decision:**

1. Stage D's `bingo_isalsr` cells are Pagie-1, Korns-12 and Vladislavleva-2.
   C1's OOMs were **Vladislavleva-4 (18 cells)** and Korns-12 (9). **Vlad-4 —
   two thirds of the OOM population — is not in Stage D.** A 10× headroom over
   Vlad-2 still leaves 32 GB with 3× margin, but it is extrapolation.
   → ✅ **Retired by §10.1.** The `max_evals = 100M` bound makes `n_unique ≤ 100M`
   on *any* problem, so Vlad-4 is covered without being run.
2. One `bingo_isalsr` cell is still RUNNING at 8:32; the dedup set grows
   monotonically, so its final `MaxRSS` can only go up.
   → ✅ **Resolved.** It finished at 10:03:55 with `MaxRSS` **1.193 GB** — the
   stage's peak, and the prediction held: longest cell, largest set, largest RSS.
3. §3.3's 256 GB was **Mario's decision of 2026-08-02, taken deliberately from
   evidence and explicitly not deferred to a measurement.** §3.3 does provide for
   revision — *"if D1.2 shows 12 h MaxRSS comfortably under 128 GB … the request
   may be revised down before launch, with the measurement recorded."* It does.
   **This is that measurement; the revision is Mario's call, not mine.**

A middle option that costs almost nothing: **64 GB** — 56× the measured peak,
ceiling 762 concurrent, arm makespan 9.5 h. It removes the ceiling while
conceding the Vlad-4 extrapolation.

---

## 4. Finding 4 — the QOS cliff, and why §11.1's concurrency figures do not transfer

Measured with `sbatch --test-only` (allocates nothing), 10-task array, 8 GB,
`--constraint=sr`, 2026-08-05 18:09 local:

| requested `--time` | scheduler's estimated start |
|---|---|
| **0-01:59:00** | **2026-08-05T18:09:54 — immediately** |
| 0-02:01:00 | 2026-08-05T21:10:54 (**+3 h**) |
| 0-13:00:00 | 2026-08-05T21:10:54 |
| 0-16:00:00 | 2026-08-05T21:10:54 |
| 0-23:00:00 | 2026-08-05T21:10:54 |
| 2-23:00:00 | 2026-08-05T21:10:55 |
| 3-01:00:00 | 2026-08-05T21:12:55 |
| 6-23:00:00 | 2026-08-05T21:12:55 |

**Two minutes of extra requested wall costs three hours of queue.** That is the
`short` → `medium_uma` QOS transition made visible. Confirmed from the priority
ledger:

| jobs | wall | QOS | priority |
|---|---|---|---|
| Stage C v3 (`c2s_*`) | 40 min | **`short`** | **118,933** |
| Stage D (`c2d_*`) | 16 h | `medium_uma` | **28,873** |

`PriorityWeightQOS = 100000` and the highest QOS priority on the cluster is
`short`'s 10000, so `short` earns 100,000 points and `medium_uma`'s 1000 earns
10,000. The difference 118,933 − 28,873 = **90,060** is that 90,000 to within
rounding — every other factor is identical.

**Three consequences.**

1. **Every achieved-concurrency figure in §11.1 — 245, 476, 592 — was measured
   at priority 118,933 under a QOS whose `MaxWall` is 2 h.** C2's tasks need 13 h
   and will run at 28,873. Those figures are optimistic upper bounds for C2, and
   §11.1 currently reads as though they are transferable. **They are not.**
2. **Right-sizing `--time` below 13 h buys nothing.** The only cliff is at 2 h and
   the campaign cannot reach it; 13 h, 16 h, 23 h and 7 days all get the same
   estimate. So the classic "shorten the wall for backfill" lever is
   **empirically dead here** — and that is good news, because it means the
   SymPy-tail risk (§11.1 2026-08-03: a cell that finished its search correctly
   and then spent 7+ further minutes in post-search SymPy) should be bought off
   with a **generous** wall, not a tight one. **Recommend 16 h**, as Stage D used.
   Never trade `--time` against §5.5 completeness; there is nothing to win.
3. `medium_uma` is entered automatically at ≤3 days and is the best QOS the
   campaign can reach. A wall above 3 days would drop to `long_uma` (priority
   500) and lose a further 5,000 points. **Keep every wall under 3 days.**

**Fairshare, which nobody has modelled.** `PriorityWeightFairShare = 50000` is
the largest weight, with a 14-day decay half-life. `sshare` today:
`FairShare = 0.157691`, `EffectvUsage = 0.075309` against `NormShares = 0.015152`
— we are already at **4.97× our within-account share**, contributing ~7,885 of
the 28,873. C2 adds 72,030 core-hours in a few days. **Achieved concurrency will
decline over the campaign as fairshare erodes**, by up to 27 % of priority if it
goes to zero. Stage C's 32-minute waves cannot see this at all. Two mitigations,
both free: front-load the UDFS arrays (they are the long pole anyway), and do
not spend fairshare on pre-flight re-runs that are not owed.

---

## 5. Finding 3 — the aggregation job is not the problem the plan thinks it is

§11.1 (2026-08-03) records the Stage C `--postprocess only` job at 1 h 34 m over
1,260 runs and extrapolates: *"the cost scales with the campaign … at C2's 8,400
runs the same job is ≈6.7× larger, i.e. ≈11 hours"*, hence ≥24 h wall and a
split per method. **Measured locally on the 1,260-run `c2_smoke_v4` root:**

| step | measured |
|---|---|
| `collect_status_ledger` over the whole root | **0.28 s** |
| reading all 1,260 `run_log.json`, with the 3-contrast re-read amplification | **0.36 s** |
| `postprocess_output_root`, real, per problem | **4.15 – 4.37 s** |

I/O is 0.6 s of a job that took 5,651 s. **The cost is not I/O and does not scale
with runs.** The unit of work is `(problem × contrast × metric)`, and that count
is **identical** at Stage C and C2: 70 problems × 2 methods × 3 contrasts × 14
metrics. Only the sample size inside each bootstrap grows, 3 seeds → 20.

`cohens_d_ci_bootstrap` (`analyzer/effect_sizes.py:39`) is a **Python `for` loop
over 10,000 resamples**, each calling `rng.choice` + `np.std` + `np.mean` on an
array of length 3–20. Timed:

| n seeds | loop | vectorised | speed-up | bit-identical? |
|---|---|---|---|---|
| 3 | 0.1066 s | 0.0008 s | 128× | **yes** |
| 5 | 0.1083 s | 0.0009 s | 126× | **yes** |
| 10 | 0.1085 s | 0.0014 s | 78× | **yes** |
| 15 | 0.1091 s | 0.0018 s | 62× | **yes** |
| **20** | **0.1094 s** | 0.0021 s | **51×** | **yes** |
| 30 | 0.1093 s | 0.0029 s | 38× | **yes** |

**Going from 3 seeds to 20 costs 2.6 %, not 6.7×.** The cost is pure Python
dispatch overhead. `compute_paired_stats` makes 14 bootstrap calls per contrast,
42 per problem, 5,880 for the campaign — 95 % of the aggregation's wall clock.

**Corrected projection.** 140 (method, problem) × 3 × 14 × 0.109 s = **640 s
locally at 20 seeds**. The Picasso/workstation ratio from the Stage C job is
5,651 / 588 = 9.6×, so **C2's aggregation is ≈1 h 45 m, not ≈11 h.**

Two independent improvements, both free:

- **Vectorise the bootstrap.** `rng.integers(0, n, size=(n_boot, n))` reproduces
  the loop's PCG64 stream exactly — `Generator.choice(a, size=n, replace=True)`
  with no `p` delegates to `integers`, and a single block draw consumes the
  stream in the same order. **Verified bit-identical at every n above.** ≈1 h 45 m
  → **≈2 min**, and the job then fits inside the 2 h `short` QOS at priority
  118,933 (§4). ⚠ This edits analysis code, so it needs a regression test over
  the real corpus and a Stage E re-run (182 s, local) before adoption.
- **Hoist `collect_status_ledger` out of the per-config loop.**
  `orchestrator.py:555` walks the *entire* root and writes one shared
  `status_ledger.csv` — **once per config, so 14 times**. Hoisting it makes the
  14 configs mutually disjoint (each writes only under its own
  `method/benchmark/`) and therefore safe to run as a 14-task array with one
  dependent walk. Without the hoist, parallelising the loop is a race on
  `status_ledger.csv` — the same class of defect as the per-task
  post-processing race of 2026-08-03.

**Recommendation:** hoist the ledger walk and split the loop into a 14-task array
+ one dependent walk. Treat the bootstrap vectorisation as a separate, tested
change. Either way, **drop the ≥24 h wall to 6 h** — but do not shrink it to 2 h
chasing the `short` QOS until the vectorisation is in, because a truncated
aggregation is exactly the failure §11.1 already recorded once.

---

## 6. Finding 5 — HOME inode quota blocks the 30-seed option

Live, 2026-08-05:

| filesystem | space | files | soft | hard | grace |
|---|---|---|---|---|---|
| HOME | 23.38 GB / 0.28 TB | **13.5k** | **35.0k** | 150.0k | none |
| FSCRATCH | 0.44 TB / 1.40 TB | **166.4k** | **250.0k** | 400.0k | none |

The HOME over-quota crisis of 2026-08-02 (0.43 TB, grace expiring) is **resolved**
— 23.38 GB now. FSCRATCH has **83.6k inode headroom** against C2's ≈45k (20
seeds) / ≈67k (30). Both A13 criteria pass on bytes.

**The inode side does not.** Every launcher writes SLURM logs to HOME:
`c2_smoke/launcher.sh:32` defaults `C2_LOGS_DIR` to
`…/mpascual/execs/isalsr/c2_smoke/logs`, and `c2_stage_d/launcher.sh:30` defaults
`D_LOGS_DIR` to `${HOME}/execs/isalsr/c2_stage_d/logs`. **There is no FSCRATCH
log directory on the cluster at all** (checked). T17-HANDOFF §3.3 states
*"Campaign logs already redirected to FSCRATCH"* — that redirection exists only
as an env-var override at launch time, and Stage D did not use it.

`~/execs` holds **10,709 files**, 79 % of HOME's 13.5k:

| tree | files |
|---|---|
| `execs/isalsr/c2_smoke/` (superseded) | 3,933 |
| `execs/isalsr/c2_smoke_v3/` (superseded) | 2,523 |
| `execs/isalsr/c2_smoke_v2/` (superseded) | 2,523 |
| everything else | 1,730 |

C2 writes 2 log files per task:

| | log files | HOME after, as-is | after archiving the 8,979 superseded |
|---|---|---|---|
| 20 seeds | 16,800 | 30.3k / 35.0k — fits at 86 % | 21.3k — comfortable |
| **30 seeds** | **25,200** | **38.7k / 35.0k — EXCEEDS** | 29.7k — fits |

**§11.1 (2026-08-05) concluded 30 seeds is affordable on core-hours. It is — but
it is currently blocked by the HOME inode quota.** The fix is free: archive the
three superseded wave log trees (they are pre-flight, superseded, and their
results are already archived), *or* point `C2_LOGS_DIR` at FSCRATCH in the
launcher default rather than relying on an override.

⚠ This is the `ISALSR_LEDGER_ENABLED` / `shadow_hash` shape a third time: **a
default that silently does the wrong thing unless someone remembers to override
it.** It should be a launcher default with an assertion, not a habit.

---

## 7. Probes: what was run, what was not, and why

| # | Probe | Status | Result |
|---|---|---|---|
| **P-2** | `sbatch --test-only` on all 42 arrays at production shape (UDFS 240–280 tasks / 16 G / 13 h; Bingo 32 G / 14 h; Bingo-IsalSR **256 G** / 14 h; all `--constraint=sr`, `%24`) | ✅ **run on the cluster** | **42/42 accepted, rc = 0.** Includes 280-task arrays at 256 GB. Run on Picasso, per defect 15 — there is no `sbatch` on the workstation |
| **P-4** | Memory re-sizing from D1.2 | ✅ | §3. `MaxRSS` 1.14 GB peak; `.batch` step only, `JobIDRaw` |
| **P-5** | Quota, inodes | ✅ | §6. FSCRATCH passes; **HOME blocks 30 seeds** |
| **P-6** | Node census + live QOS | ✅ | `sr` 154 nodes / 19,712 cores, **6,400 drained-or-reserved (32 %)**, usable 13,312, idle 5,481. `long_uma`/`medium_uma` `MaxTRESPU cpu=9000, mem=50000000M`; `MaxTRES` per job `cpu=40000, node=500`; `MaxJobsPU`/`MaxSubmitJobsPU` unset; `MaxArraySize` 4096. **No account-level `GrpTRES` on `tic_163_uma`** |
| **P-7** | Placement | ✅ (retrospective) | v3: **1,260/1,260 on `sr`**. v2 (unpinned): 1,112 `sr` / 148 `sd` — the C4 failure, visible |
| **P-8** | Aggregation cost at scale | ✅ (locally, and decomposed) | §5. Stronger than the planned timing run: it found *why* |
| — | Throttle acceptance | ✅ | `%24`, `%48`, `%96`, `%280` all accepted by `--test-only` |
| **P-1** | Throttle sweep: re-run Stage C at `%48` and `%96` | ❌ **not run, and should not be** | Two reasons, below |
| **P-3** | Parallel-wave feasibility, 3 × ≤60-task probes | ❌ **not run — the question is answered analytically** | Below |

### 7.1 Why P-1 as written should not be run

**It violates SP-0.** Stage C is 1,260 tasks and ≈315 core-hours, against SP-0's
≤60 tasks and ≈30 core-hours. The brief lists it as the highest-value probe while
§3.1 of the same brief forbids it.

**And it cannot answer the question anyway.** Stage C has 1,260 tasks of 900 s.
At `%48` the ceiling is 2,016 — larger than the whole task list. Every task
starts at once, the span collapses to one task duration plus ramp, and the
measurement degenerates into a ramp-rate measurement. The measured envelope
(§2.1) already shows this beginning at `%24`: 93 % of ceiling touched, held for
2.7 % of the span. **To measure sustained concurrency at ceiling *C* you need
≫ *C* tasks of the production duration**, which is the campaign.

What the retrospective analysis gives instead, for free: the scheduler granted
934 concurrent 1-core `sr` slots within minutes at `%24`. Combined with `%48`/
`%96`/`%280` being accepted by `--test-only`, an entitlement of 9,000 cores and
5,481 idle `sr` cores, **`%48` is well supported.** What is *not* supported is
assuming the grant carries over at the campaign's 4.1×-lower QOS priority (§4).

### 7.2 Why the three-parallel-wave design is strictly dominated

The brief asks whether three seed-block waves can run concurrently. They can —
but **three waves at `%24` and one wave at `%72` are the same request to SLURM.**
The scheduler sees 42 arrays either way; wave structure is bookkeeping, not a
scheduling primitive. So the wave split buys exactly the concurrency a higher
`%K` buys, while adding three campaign roots, three submission windows, three
chances at defect 10 (a deploy or config edit landing mid-wave), and a seed-block
bookkeeping burden on §5.5's completeness accounting.

**Recommend against it. Raise `%K` and fix the allocation instead** — same
concurrency, one shell variable, one root, one submission.

The brief's own constraint — all blocks must pin the same family, because UDFS is
time-budgeted and node speed changes what it explores — is correct and is
independently reinforced by §4: node speed is not the only cross-block hazard,
QOS and fairshare drift over a multi-day campaign too.

---

## 8. Recommendations, ranked

Nothing here trades science. Nothing here touches the 12 h budget, the hash arm,
D2 coverage, the per-run instrumentation or the `sr` pin.

| # | Change | Where | Effect | Risk |
|---|---|---|---|---|
| 1 | **Throttle proportional to `Nᵢ·Tᵢ`** (`T_udfs = 12 h`, `T_bingo = 8 h`), total unchanged | launcher, one function | **1.65–1.92×** makespan | none — no config content changes |
| 2 | **Raise the total to Σ Kᵢ ≈ 2,016** (mean `%48`) | `C2_THROTTLE` | a further ≈2× | grant risk; 22 % of entitlement, `--test-only` accepts |
| 3 | **Bingo-IsalSR `--mem` 256 G → 32 G** (or 64 G) | slurm config | removes a 154-task ceiling and a 47.7 TiB QOS-memory ceiling; arm 46.8 h → 4.7 h | **Mario's decision** (§3.3). Vlad-4 not in Stage D |
| 4 | **Archive the 8,979 superseded smoke log files; default `C2_LOGS_DIR` to FSCRATCH with an assertion** | launcher | unblocks **30 seeds** | none |
| 5 | **Hoist `collect_status_ledger` out of the per-config loop; run the 14 configs as an array** | `orchestrator.py:555`, `aggregate_worker.sh` | ≈1 h 45 m → ≈10 min; also removes a latent shared-write race | needs a test |
| 6 | **Aggregation wall ≥24 h → 6 h** | launcher | frees nothing directly; stops over-reserving | none |
| 7 | **Keep `--time` generous (16 h) and under 3 days** | launcher | protects §5.5 against the SymPy tail at **zero** scheduling cost (§4) | none |
| 8 | **Submit UDFS arrays first** | launcher order | long pole first; also spends fairshare while it is highest | none |
| 9 | **Vectorise `cohens_d_ci_bootstrap`** | `effect_sizes.py:39` | 51× on the aggregation, bit-identical | needs a regression test on the real corpus + a Stage E re-run |

**Combined effect of 1 + 2 + 3, at 20 seeds: modelled makespan ≈ 36 h against
the uniform-`%24` baseline of 140 h.** At 30 seeds, ≈ 54 h. Both are inside the
2026-09-03 target with room the plan has not been counting on — but see §4 before
treating any of it as banked.

---

## 9. Carries and open items

- ⚠ **The `≈71,400 core-hours at 20 seeds` figure is a lower bound.** It is built
  from C1's Bingo mean of 5.15 h, which (a) *includes* the 8 `roundoff` problems
  that F-19 has now shown ran at a **10× smaller `max_evals`**, and (b) *excludes*
  the 20 D2 problems entirely, which have no runtime data at all. 28 of 70
  problems (40 %) will run longer in C2 than in C1. **Budget ≈80,000 core-hours
  and re-derive `T_bingo` from Stage D's certification when it lands.** The
  sensitivity table (§2.4) is why this does not invalidate the allocation fix.
- 🔴 **`--test-only` models only task 1 of an array.** It confirmed acceptance and
  the QOS cliff; it cannot confirm that 2,016 concurrent slots will be granted.
  The only honest measurement of that is the campaign's own first hours —
  **watch the achieved concurrency in the first 6 h and be ready to lower `%K`**,
  which is safe to do mid-campaign (it does not touch the deployed tree or any
  config, so it is not defect 10).
- 🔴 **Fairshare erosion is unmodelled** and will reduce concurrency over the
  campaign. `sshare` at launch and at the midpoint would quantify it, free.
- Stage D was still running at 8:32 when this log was written; **5 cells
  outstanding** (3 UDFS, 1 `bingo_std`, 1 `bingo_isalsr`) and the certifier
  (1769425) PENDING on `afterany`. The UDFS cells will settle the `--time`
  recommendation empirically — they saturate `max_time = 12 h` and their SLURM
  `Elapsed` is the SymPy tail made visible. **Re-read them before the tag.**
- `sr` currently has **6,400 of 19,712 cores drained or reserved (32 %)**. That
  is a third of the pinned pool unavailable, and it is not in any capacity figure
  the plan carries. Re-check on the day (P-6).
- **Nothing was deployed.** The local tree is `3d5a79c-dirty`; the deployed tree
  is `00635ae`, clean, and Stage D is reading it. Every change recommended here
  lands **between** waves (defect 10).

---

## 10. IMPLEMENTED — 2026-08-05, local only, nothing deployed

Mario asked for the tested proposals to be landed. All nine are in, plus one
that only became available once the aggregation got cheap. **Nothing was
deployed**: Stage D's arrays are still reading the deployed tree at `00635ae`
(defect 10). Every change below goes out with the owed clean Stage C wave.

### 10.1 The memory question, settled — it is a hard bound, not an extrapolation

This was the item flagged as "recommendation, not decision" in §3.3, because
Stage D covers Pagie-1, Korns-12 and Vladislavleva-2 but **not** Vladislavleva-4,
which was 18 of C1's 27 OOM cells. That gap is now closed, by bounding rather
than by extrapolating.

**The argument.**

1. `IsalSRDeduplicator.canonical_seen: set[int]` (`bingo/isalsr_runner.py:236`) is
   the **only** unbounded container in the arm. Verified by inspection:
   `_parent_ids` is bounded by `population_size = 500` and the T06 per-k
   histograms by the k range.
2. A candidate cannot enter that set without being scored, and Bingo stops on
   `max_evals = 100M` — observed on **all six** completed Stage D cells at
   100M ± 0.07 %, while `max_time = 43,200 s` never fired. Therefore
   `n_unique ≤ n_total ≤ 100,000,000`, **on any problem, Vlad-4 included**.
3. Measured cost of a `set[int]` under the production allocator
   (`PYTHONMALLOC=malloc`, `worker.sh:59`): **81.5 bytes/entry**, flat from 1 M
   to 32 M entries, with a **1.16×** transient at each power-of-two table resize
   (visible directly: at 24 M entries `VmRSS` 2.086 GB against `VmHWM` 2.414 GB).
4. Hence **worst-case peak RSS = 100M × 81.5 B × 1.16 + 0.42 GB ≈ 9.4 GB.** The
   independent structural model (2²⁸ slots × 16 B table + 100M × 32 B PyLongs +
   the old table still mapped during the resize) gives 9.40 GB — the two agree.

**Measured, for contrast:**

| Stage D group | dedup set | `MaxRSS` |
|---|---|---|
| `bingo_isalsr` (3 completed, stopped on `max_evals`) | 6.3–7.2 M unique | **1.05–1.16 GB** |
| `bingo_std` baseline (no set at all) | — | 0.39–0.42 GB |

The observed peak sits **14× below** the ceiling, because ρ ≈ 1.83 and LM inner
iterations consume ~8 evals per candidate.

**Why C1 hit 127.7 GB and this does not.** Two causes, both already removed:
`worker.sh:55-59` switched to `PYTHONMALLOC=malloc` with exactly this rationale
in its comment — *"pymalloc fragments the heap over 10k+ generations (256 KB
arenas pinned by surviving objects) … this is what keeps Bingo-IsalSR off the OOM
ceiling"* — and the dedup set moved from `set[str]` (~150 B/entry) to `set[int]`.
The 100× drop is explained, not lucky.

**Landed: 256 GB → 32 GB**, i.e. 3.4× the hard ceiling and 28× the observed peak.
Even 16 GB would clear the ceiling by 1.7×. This is the revision §3.3 explicitly
provided for, taken on the measurement it asked for. An OOM would still be
*named* rather than silent (P4 writes the status record ahead of the search).

### 10.2 What changed

| File | Change |
|---|---|
| `experiments/scripts/c2_slot_plan.py` | **NEW.** Single source of truth for the 42 arrays: task counts from the registry, work-proportional throttle apportionment, `MEM_GB`, `WALL`, submission order. Every number carries its derivation |
| `tests/unit/test_c2_slot_plan.py` | **NEW, 49 tests.** Including `test_proportional_never_loses_to_uniform` across 3/20/30 seeds × `%8/%24/%48/%96`, the `T_bingo` sensitivity sweep, and `test_bingo_isalsr_memory_covers_the_hard_ceiling` |
| `slurm/c2_smoke/launcher.sh` | `C2_PROFILE={smoke,campaign}`; plan-driven submission (one Python call, not 42); logs default to **FSCRATCH**; aggregation split; `C2_UNIFORM_THROTTLE` escape hatch |
| `slurm/c2_smoke/aggregate_worker.sh` | Two roles: one config per array task, or the single ledger+certification job |
| `experiments/models/orchestrator.py` | `postprocess_output_root(write_ledger=…)`; new `write_status_ledger()`; `--no-status-ledger`; `--postprocess ledger` (needs no config) |
| `experiments/models/analyzer/effect_sizes.py` | `cohens_d_ci_bootstrap` vectorised, **bit-identically** |
| `tests/unit/test_effect_sizes_bootstrap.py` | **NEW, 36 tests.** Bit-identity against a verbatim copy of the original loop across n, seeds, `n_boot`, CI level, seven degenerate inputs and a 200-input random sweep |

**One profile, two configurations.** Stage C and C2 now differ only in seeds,
payload budget, wall and output root. Everything structural — topology,
apportionment, memory, aggregation shape — is shared, so the owed clean Stage C
wave certifies the campaign's launcher rather than a cousin of it (§1:
"certifying a topology you will not launch certifies nothing"). A side effect
worth recording: **the Stage C memory deviation of 2026-08-03 is withdrawn.** It
existed only because 256 GB × 210 concurrent smoke tasks would have measured
fat-node availability; at 32 GB there is nothing to deviate from.

### 10.3 The tenth change, which only became possible after the ninth

With the bootstrap vectorised, all fourteen configs post-process in **19 s**
locally against ~590 s before. That drops the aggregation under the 2 h `short`
MaxWall — so `AGG_WALL` is now **1 h 59 m**, not 6 h, which buys QOS priority
**118,933 instead of 28,873** and, per §4's measurement, an *immediate* start
instead of a three-hour queue. Margin is >200× per array task. This is the only
place in the campaign where trimming a wall buys anything.

### 10.4 Verification

| Check | Result |
|---|---|
| `tests/unit/test_c2_slot_plan.py` | **49 passed** |
| `tests/unit/test_effect_sizes_bootstrap.py` | **36 passed** |
| Full unit suite | **7,031 passed**, 5 skipped, 2 failed — the 2 are `test_appendix_d_generator.py`, T09's untracked work, pre-existing and recorded in the Stage E entry. **Zero regressions** |
| `ruff` on the six touched files | clean |
| `mypy --strict src/isalsr/` | clean, 55 files |
| `bash -n` on both shell files | clean |
| **Old vs new aggregation path, same machine, real 1,260-run corpus** | **all 841 artefacts BYTE-IDENTICAL** (`aggregate.csv`, `paired_stats*.json`, `status_ledger.csv`) |
| `sbatch --test-only`, 42 arrays at the **new** shape | **42/42 accepted** — 8,400 tasks, 2,016 slots, throttles %46–%80, 32 GB, 16 h, `--constraint=sr` |
| Launcher `--dry-run`, both profiles | smoke 1,260 tasks / 1,008 slots; campaign 8,400 / 2,016 |

**A false alarm worth recording, because it nearly became a wrong conclusion.**
The first byte-comparison reported 4 of 841 `paired_stats` files differing. The
cause was **not** the bootstrap: those four artefacts had been written *on
Picasso* by job 1766718 and I was regenerating them *locally*, and
`compute_paired_stats` is not bit-reproducible across the two scipy/BLAS builds
on borderline inputs. Isolating it — computing both contrasts in-process, once
with the original loop monkeypatched back in — returned **IDENTICAL on all
four**, and re-running the A/B on one machine gave 841/841. ⚠ **Carry: do not
diff Picasso-written analysis artefacts against locally regenerated ones and read
a difference as a code change.**

### 10.5 Corrected: the quota picture after moving logs to FSCRATCH

§6 said the HOME inode quota blocks 30 seeds. With logs defaulting to FSCRATCH
that is no longer the binding constraint, and the arithmetic moves:

| | results | logs | total | FSCRATCH after (soft 250.0k) |
|---|---|---|---|---|
| now | — | — | — | 166.4k |
| **20 seeds** | ≈45k | 16.8k | 61.8k | **228.2k — fits**, 21.8k spare |
| **30 seeds** | ≈67k | 25.2k | 92.2k | **258.6k — exceeds soft** by 8.6k (hard is 400k) |

Archiving the superseded roots recovers little on FSCRATCH (`c2_smoke_v3` and
`c2_smoke_v4` are 7,932 files each), so **30 seeds needs the ≥15,000-file support
request that T17-HANDOFF §3.3 already lists as outstanding.** HOME is now
incidental: `~/execs` holds 10.7k files of a 35.0k soft quota and the campaign no
longer adds to it.

### 10.6 Revised projection

| | before | after |
|---|---|---|
| Makespan, 20 seeds | 140.0 h (5.8 d) at uniform `%24` | **36.0 h (1.5 d)** |
| Makespan, 30 seeds | 210.0 h (8.8 d) | **54.1 h (2.3 d)** |
| Bingo-IsalSR concurrency ceiling | 154 tasks (node-bound), 190 (QOS mem) | none binding |
| Aggregation | ≈11 h claimed / ≈1 h 45 m real | **≈2 min**, and at 20× the QOS priority |
| C2 SLURM logs | HOME, 86 % of quota at 20 seeds, over it at 30 | FSCRATCH |

⚠ These assume the scheduler grants 2,016 slots. It granted 934 under the
`short` QOS; C2 runs at 4.1× lower priority (§4) and fairshare erodes as the
campaign burns. **Watch achieved concurrency in the first six hours.** Lowering
`C2_SLOT_BUDGET` mid-campaign is safe — it touches no config and no deployed
file, so it is not defect 10.

---

## 12. 30 SEEDS — 2026-08-05 (Mario)

Campaign moved from 20 to 30 seeds. §0.4a fixed it at 20 on a cost premise the
measured runtimes reversed; §6.3's disclosure paragraph about reduced
supplementary-table power now goes away entirely, and C1's seed count is
restored.

### 12.1 A correction to §10.6 that changes the headline number

**I quoted 54 h. The plan as shipped delivers 63 h.** The 54 h came from the
*oracle* row of the §2.4 sensitivity table — the makespan when the allocation is
weighted with the true `T_bingo`. The shipped plan weights with `T_bingo = 8 h`
on purpose, because F-19 raised three suites' `max_evals` tenfold and the 20 D2
problems have no runtime data at all. Scoring that pessimistically-weighted plan
at the measured 5.15 h gives **63.0 h**, not 54.1: UDFS is left slightly
under-slotted and the Bingo arrays drain 22 h early, idling 806 slots.

Both numbers are now printed, separately and labelled, by
`c2_slot_plan --table` — the allocation basis and the expectation. Quoting the
pessimistic figure as a forecast is how a schedule acquires invisible padding;
quoting the oracle figure as a plan is how it acquires invisible optimism.

### 12.2 The 9 hours are recoverable in flight, and that is the point

`scontrol update JobId=<id> ArrayTaskThrottle=<n>` re-apportions a **running**
array (verified available on Picasso's SLURM 25.05.1). It touches no config and
no file in the deployed tree, so it is **not** defect 10 and is safe mid-campaign.

`c2_slot_plan --bingo-hours <h> --rebalance <job_ids.txt>` emits the 42 lines.
Fed the measured 5.15 h it returns the plan to **54.1 h at 1,998 mean cores** —
the oracle figure, recovered. It refuses if the job-id count does not match the
array count rather than applying a shifted mapping.

**So the conservative weighting is not a 9-hour tax; it is a Day-1 decision
deferred until Bingo's real cost under F-19 is observable.** Launch weighted at
8 h, read the first day's Bingo wall clocks, rebalance.

### 12.3 🔴 30 seeds is BLOCKED on inodes until ~13.4k are freed

| | results | logs | need | FSCRATCH free | verdict |
|---|---|---|---|---|---|
| 20 seeds | 48,135 | 16,800 | 64,935 | 83,600 | fits, 18.7k spare |
| **30 seeds** | 71,781 | 25,200 | **96,981** | 83,600 | **short by 13,381** |

Coefficients measured on `c2_smoke_v4`: 7,932 inodes for 1,260 runs, of which
843 are the fixed per-`(problem, contrast)` artefacts that do not scale with
seeds ⇒ **5.63 inodes/run + 843 + 2 log files/task**.

Two measured remedies, together +29.7k — **Mario's call, both touch his data**:

```bash
conda clean -a -y                                   # ~21,800 inodes (package cache)
cd $FSCRATCH/results/isalsr && \
  tar czf c2_smoke_v3.tar.gz --remove-files c2_smoke_v3   # ~7,900 (v1/v2 already .tar.gz)
```

After both: 113.3k free against 96,981 needed — **16.4k spare.**

This is a *soft* quota (hard is 400k), so the campaign would probably survive on
GPFS grace. That is a gamble on a grace period nobody has measured, for a fix
that takes one command.

### 12.4 The launcher now refuses rather than discovering this at hour 40

P6's failure mode verbatim: *"C2 hits the hard file quota mid-campaign and every
running task keeps burning wallclock while all its writes fail."* At 30 seeds
that is discovered ~40 h into a 63 h campaign. `check_inode_budget()` runs before
the first `sbatch` on both `submit` and `--test-only`, projects from the measured
coefficients against the live quota, and refuses with the exact shortfall and the
two commands above.

🔴 **Its first version reported "would submit" for the 30-seed request.**
Picasso's `quota` separates the space and file halves with a literal `║`, so the
positional fields I used straddled the divider and yielded `used = 0`. A check
that passes while measuring nothing — the same shape as both C1.11 defects
(`sacct -X` returning a blank `MaxRSS`, then `JobID` matching 42 of 1,260 rows
and still reporting PASS). Rewritten to split on the divider, and to **fail
closed**: an implausible parse (`used <= 0`, `soft <= used`, no row) refuses
rather than waving through. Verified on the cluster against the live quota, with
a negative control that feeds it garbage and confirms it still refuses.

### 12.5 `n_seeds` in the configs — the fourth instance of the same trap

`orchestrator.py:641` reads `n_seeds` from the config **whenever `--seeds` is
absent**. The launcher always passes it, but a manual re-run of a failed cell, a
resume, or an analysis script does not — and would silently have got 20. All 14
configs moved to `n_seeds: 30`, verified **through the YAML loader** rather than
the file text, and locked by
`test_c2_slot_plan.py::test_campaign_seed_count_is_declared_by_every_config`.
Done on the **local** tree only: Stage D's running tasks read the *deployed*
tree, so this is not "editing a config while an array reads it".

⚠ `config_sha256` changes for all 14. That is expected and is why it happens
between waves; the owed clean Stage C wave certifies the final configs.

### 12.6 What did NOT change, and why

**No SLURM wall was raised to 54 h or 63 h.** Those are *campaign makespans* —
how long 42 arrays take to drain — and SLURM has no campaign-duration parameter.
`--time` bounds **one task**, and one task is **one run**: UDFS 12.00 h plus the
SymPy tail, Bingo ≤ 11.76 h observed. **16 h is correct and raising it would be
actively harmful** — a hung task would burn 54 h of an allocation instead of 16
before the ledger caught it. The three walls, all verified sufficient at 30 seeds:

| job | wall | basis |
|---|---|---|
| the 42 arrays | 16 h/task | UDFS 12 h + tail; unchanged by seed count |
| aggregation array | 1 h 59 m/task | one config; n=30 bootstrap ≈ 40 s. >150× margin, and inside `short` |
| ledger + certify | 1 h 59 m | full-root walk, 0.28 s at 1,260 runs ⇒ ~2.8 s at 12,600 |

Seed count changes the number of tasks (8,400 → 12,600), not the length of any
one of them. Also unchanged: the slot budget (2,016), the `sr` pin, the 12 h
search budget, the memory table.

### 12.7 Executed: v3 archived, logs merged — 30 seeds now fits

**`conda clean` was wrong advice and I had put it in the guard's error text.**
Two measurements killed it: `conda clean -a --dry-run` reports *nothing* to remove
(the cache is already clean), and **16,060 of the 18,300 files under
`conda_pkgs` are hardlinked into the envs** (`nlink ≥ 2`), so deleting a pkgs
entry frees a directory entry and not an inode. The guard now says so explicitly,
to stop the next person spending an afternoon on it.

**What was actually done.** `c2_smoke_v3` archived to `c2_smoke_v3.tar.gz` —
tarred, the archive verified to hold all 7,932 entries and to read cleanly,
*then* the source removed. FSCRATCH **166.4k → 158.5k**, headroom **83.6k → 91.6k**.

**And that alone is still not enough**, which is why logs are now merged:

| | results | logs | need | free | |
|---|---|---|---|---|---|
| as-is, split | 71,781 | 25,200 | 96,981 | 83,600 | short 13,381 |
| v3 archived, split | 71,781 | 25,200 | 96,981 | 91,531 | **short 5,450** |
| v3 archived, **merged** | 71,781 | **12,600** | **84,381** | 91,531 | **fits, +7,150** |
| v3+v4 archived, split | 71,781 | 25,200 | 96,981 | 99,462 | fits, +2,481 — but v5 eats it |

Split logs do not fit even with **both** superseded smoke roots archived once the
owed v5 wave is accounted for. `C2_MERGE_LOGS=1` (default, both profiles) omits
`--error` so SLURM folds stderr into stdout: 25,200 → 12,600 files. Nothing is
lost — `c2_certify` reads `run_log.json`, `status.json` and `sacct`, never the
SLURM logs (grepped), and `PYTHONUNBUFFERED=1` keeps the two streams interleaving
line-cleanly. Verified by stubbing `sbatch` on `PATH`: `--error` is emitted only
when `C2_MERGE_LOGS=0`.

**Live on Picasso, after both changes:** `need 84381 (results 71781 + logs 12600,
merged) against 91600 free` → **SUBMIT**; the split-logs control at the same
moment → **REFUSED**. Both `--test-only` sweeps accept **42/42** at 12,600 tasks.

⚠ **Sequencing for v4.** The owed clean Stage C wave writes a v5 root (~7,932
inodes), leaving ~7,150 − 7,932 < 0. **Archive `c2_smoke_v4` the same way once v5
has passed**, which restores ~7,930 and returns the margin to ~7,150. The guard
enforces this at submit time; it does not need remembering.

### 12.8 Two contradictions Mario caught, both real

1. **`bingo_hard_trace.yaml` at `n_seeds: 20`.** Not a missed edit and not a safe
   exception: the file's own header declares it to be `bingo_hard.yaml` *"with ONE
   key changed: `shadow_hash: true`"*, so moving the parent to 30 made the
   difference **two keys** — with nothing failing, because
   `test_budget_uniformity.py` deliberately excludes trace configs from
   `CAMPAIGN_SUITES`. That matters: the split exists so the certification and
   traced cells differ *visibly and only* in `shadow_hash` (audit.md §7.3). Set to
   30 (inert — `worker.sh:246` passes `--seeds 102`) and the invariant is now
   **enforced** by `test_trace_config_differs_from_its_parent_by_one_key`, which
   flattens both YAMLs and asserts the diff is exactly `{bingo.shadow_hash}`.
2. **§0.4a still said "We launch at 20 seeds. Full stop."** Eight live
   contradictions fixed, two of them binding rules rather than prose: **SP-0's
   probe cap said "never 1…20"**, so a probe at seed 25 would have looked legal
   and collided with a campaign cell; and **§11.2's A5 row still read PASS**
   against a seed set that no longer exists, now **reopened**. Details in
   EXECUTION-PLAN §11.1, 2026-08-05. Row-by-row checksum of §11.3's new per-array
   `%K` column against the generator caught **two of seven values I had typed from
   the 20-seed run**.

### 12.9 Verification of the 30-seed switch

| Check | Result |
|---|---|
| `tests/unit/test_c2_slot_plan.py` | **52 passed** (3 new: seed-count declaration, 30-seed plan, expected-vs-planned separation) |
| `tests/unit/test_budget_uniformity.py` | 17 passed — the F-19 lock still holds after the config edit |
| Full unit suite | **7,034 passed**, 5 skipped, 2 failed (T09's untracked `test_appendix_d_generator.py`, pre-existing) |
| `ruff` / `mypy --strict` / `bash -n` | clean |
| 14 configs at `n_seeds: 30` | verified through the loader, single-valued |
| Campaign dry-run | 42 arrays, **12,600 tasks**, 2,016 slots, throttles %23–%80 |
| Inode guard, live on Picasso | 20 seeds → submit; **30 seeds → REFUSED** with the shortfall; garbage input → REFUSED |
| `--rebalance` at 5.15 h | 63.2 h → **54.1 h**; refuses a mismatched job-id count |

---

## 13. F-20 and T07 reconciled — 2026-08-05, while Stage D drains

Both §11.1 rows read 🔴 while the code said otherwise. Reconciled by measurement,
not by editing prose.

### 13.1 F-20 — closed, and it produced a §7 exclusion nobody had

The decision had already been taken in the working tree (uncommitted, not mine):
`udfs_feynman.yaml` **7 → 5**, with the rationale written inline. Verified
**through the loader**: `n_calc_nodes {5: 7}`, and alongside it
`max_orders {200000: 7}`, `max_time {43200: 7}`, `processes {1: 7}`. Locked by
`test_udfs_budget_is_uniform_across_suites`, parametrised over all three keys.

Levelling *down* is the right direction and the config says why: the original
rationale ("Feynman has up to 3 variables; more calc nodes needed") is **inverted
against the portfolio** — the 5-variable suites already ran at 5, so the suite
with the *fewest* variables carried the *largest* cap; and UDFS saturates its 12 h
budget on 100 % of runs, so raising the cap enlarges the enumeration space without
enlarging the budget.

🔴 **The carry nobody had propagated.** §7 read *"UDFS is unaffected by the second
exclusion, its set having never been ours to set."* True of the **operator set**,
**false of the search-space bound** once F-20 closed. `n_calc_nodes` caps the
intermediate nodes UDFS may enumerate (`dag_search.py:594`), so **the 10 `feynman`
problems ran C1 with a strictly larger reachable space than they will run C2**.
§7 now carries **three** exclusions, not two. Direction is adverse to us — C2's
UDFS may recover fewer Feynman expressions than C1 did, for a reason unrelated to
IsalSR — so it belongs in the letter *before* a reviewer finds it.

### 13.2 T07 / gate 3 — the 5 counterexamples are gone, measured

Re-ran the full B4 harness on `3d5a79c`, python vs cpp:

| gate | corpus | comparisons | mismatches |
|---|---|---|---|
| 1 (exhaustive perm) | 33 DAGs, k ≤ 8 | **54,765** | 0 cross, 0 invariance |
| 2 (corpus) | 10,000 DAGs | 10,000 | 0 |
| **3 (round-trip)** | **10,000 DAGs** | **20,000** | **0 on both engines** |

**PASS**, 3.45 s. **The check is not vacuous**: the gate-3 corpus contains **15
DAGs with an over-saturated binary node** (max in-degree 4) — precisely the class
CLAUDE.md names as the standing control.

**Mechanism, so this is an explanation and not luck.** T18 narrowed
`is_isomorphic` to compare `ordered_inputs(v)[0]` only. That is all Σ_SR encodes —
surplus in-edges are emitted by `C`/`c` in canonical-traversal order, so their
positions carry no information the string could recover. The old whole-list
comparison was therefore **strictly finer than the canonical string**, and the
"5 unsound merges" were **5 false positives of the checker**. At in-degree exactly
2 — the only case `dag_evaluator` accepts — agreement at position 0 forces
agreement at position 1, so nothing is lost.

**SP-3 discharged with a negative control**: the harness reports *"C++
canonicaliser is live"*, and forcing `--backend-b python` makes it declare
`self_comparison=true` and **FAIL** at the top level. A PASS therefore means both
engines genuinely ran.

⚠ **Two things this does not close.** (a) It is the *generated* corpus; the rate on
**evolved** candidates is still open (T01 saw 0/117,798 on a different corpus;
B3 shows real Bingo candidates reach k = 37). Stage D's dedup arms report
`n_canon_raised = 0` and `n_canon_timeouts = 0` over 11.4–14.1 M sampled
candidates — corroborating, but that is a *raise* counter, not an isomorphism
check. (b) It was measured **locally**; the 2026-08-03 Picasso run stays the
cross-engine authority. **Re-run gate 3 on Picasso alongside the owed Stage C
wave** to close it at the same provenance.

**T07 itself is not discharged** — §7bis.2 (Ezequiel's five Lemma 3.14/A.2 gaps
plus the Theorem 3.13 domain mismatch) is untouched by this and remains the
Stage-F obligation.

### 13.3 Documents updated

| Where | Change |
|---|---|
| §11.1 | F-20 row 🔴 → ✅ with the decision, the lock and the new §7 carry; new T07 reconciliation row; the 2026-08-03 counterexample row demoted to 🟡 and retained verbatim |
| §11.2 | **B4 sign-off row** updated — it still recorded "gate 3 fails 5/10,000"; now records the re-measurement, with the local-vs-Picasso provenance caveat stated |
| §7 | **two exclusions → three**, and the false "UDFS is unaffected" sentence replaced |
| `T07-theorem-foundation.md` | closure note at the head, pointing at the evidence and stating what remains |
| `T07-appendix/gate_all_3d5a79c_2026-08-05.json` | the gate report, persisted out of scratch |

---

## 14. STAGE D FINAL — 13/13, `GO`, 8/8 (2026-08-05 21:38 UTC)

**These supersede every in-flight Stage D figure above** (§1 row 2, §3.1, §3.3,
§7 P-4, §9). Where a number moved, the direction is stated.

### 14.1 Verdict

13/13 cells `COMPLETED`, zero failures, on `00635ae` — one `build_hash`, two
`config_sha256` (8 certification cells + the trace), exactly as the shadow split
intends. UDFS landed at **12:00:12**, saturating `max_time` to the second.

| | verdict | observed |
|---|---|---|
| D1.1 | PASS | 12/12; min wall headroom **25.0 %** of 16 h |
| D1.2 | PASS | 12/12; min headroom **96.2 %** |
| D1.3 | PASS | 12/12 artefact sets |
| **D1.4** | **PASS** | **2/2 finite — the C1 NaN does not recur** |
| D1.5 | PASS | **0/4** ρ violations |
| **D1.6** | **PASS** | **0 ρ, 0 R² excursions over 8 comparisons** |
| D1.7 | PASS | 12/12 with `T_canon`, `T_eval` > 0 |
| D1.8 | PASS | manifest validated, 0 problems |

### 14.2 Memory — the 32 GB decision, now doubly corroborated

| group | requested | peak | D1.2's own recommendation |
|---|---|---|---|
| bingo/baseline | 32 G | 0.524 G | 8 G |
| bingo/hash | 32 G | 1.231 G | 8 G |
| **bingo/isalsr** | **256 G** | **1.193 G** | **8 G** |
| udfs/{baseline,hash,isalsr} | 16 G | 0.524–0.586 G | 8 G |

D1.2's rule is `ceil_to_8GB(peak / 0.70)` and it lands on **8 GB for every
group**. The shipped **32 GB is 4× the certifier's recommendation and 27× the
observed peak** — and independently 3.4× the `max_evals`-bounded 9.4 GB ceiling
of §10.1. Two derivations that share no assumptions, same conclusion.

### 14.3 D1.6 — the C5 §3.5 handoff, answered

Stage C's 900 s wave put Bingo ρ **1.1–1.7 % below C1**, and C5 was signed with
that deviation deferred here. At 12 h against 12 h it collapses:

| method | problem | ρ_C2 | ρ_C1 | ratio |
|---|---|---|---|---|
| udfs | Pagie-1 | 1.8554 | 1.7412 | **1.0656** (rose) |
| bingo | Pagie-1 | 1.8289 | 1.8338 | 0.9973 |
| bingo | Korns-12 | 1.8238 | 1.8214 | 1.0013 |
| bingo | Vlad-2 | 1.8277 | 1.8320 | 0.9977 |

Zero excursions against a one-sided 10 % floor. **The Stage C shortfall was the
budget gap, not a canonicaliser regression.** The ρ reconstruction (`1 + δ_ρ`,
since the baseline arm reports ρ = 1 by construction) was **cross-checked, not
assumed** — it reproduces the published `mean_reduction_factor` to `abs_gap`
0.0005 (bingo) and 0.0 (udfs) over 50 problems. R²: 0 excursions on 4
comparisons, largest 0.115 against a 0.15 band.

### 14.4 Runtimes — the `T_bingo = 8 h` planning weight holds

Ten Bingo cells: 1.00, 3.03, 3.47, 4.62, 5.22, 5.61, 6.77, 6.87, 10.05, 10.82 h
⇒ **mean 5.75 h, max 10.82 h**. C1's n=564 gave 5.15 h, so Stage D runs slightly
hotter — expected, these are three *hard* problems. The weight sits above the
mean and inside the 4–12 h band `test_sensitivity_to_the_assumed_bingo_runtime`
asserts, so the apportionment is unchanged. Both Korns-12 arms ran ~10 h; it is a
genuinely slow problem, not one bad cell.

**Overhead (D1.7, canon + conversion):** Bingo mean **7.83 %** of eval (p50 8.7,
max 17.7), UDFS **0.027 %**. Above the old canon-only ≈7.4 % projection, which is
the accounting correction §11.1 2026-08-04 already flagged, not a regression.

### 14.5 Two things to carry

🔴 **The Picasso certifier reported `GO` / `n_blocking_failures: 0` with D1.6 at
`SKIP`.** D1.6's own rules say BLOCKING twice, but its C1 reference is a
`/media/.../Sandisk2TB/` path no compute node can reach, so it degraded to
advisory and the headline banked an unevaluated blocking criterion. The report is
honest — it says SKIP, not PASS — but a reader stopping at the verdict is misled.
Re-ran locally to get the real answer. **Before the next Stage D: ship the C1
analysis directory to FSCRATCH, or run the certifier where C1 lives.** Same
family as the two C1.11 defects and the first version of my own inode guard.

⚠ **Korns-12 hash returned R²_test = −4.015**, against baseline −0.014 and
isalsr −0.022. Finite, so D1.4 is unaffected (it names the *isalsr* arm). But
R²_train is 0.0232 — that arm fit ~2 % of train variance then landed 5× worse
than the mean on test, while baseline and isalsr both produced near-mean
predictors. At one seed it is an anecdote; **at 30 seeds it would dominate the
hash arm's mean on that problem**, and §6.4's policy covers NaN, not
finite-but-wild. Direction is *away* from IsalSR, which is the direction to be
most careful claiming. Check the hash-vs-baseline contrast on Korns-12 once real
seeds exist.

### 14.6 Artefacts

Stage D pulled to
`/media/mpascual/Sandisk2TB/research/isalsr/results/model_validation/real_benchmarks/c2_stage_d/`
(13 run logs, 13 trajectories, 13 RSS series, 263 MB), with
`c2_preflight/stage_d_certification{,_with_c1}.{json,md}` — the second being the
D1.6-complete re-run.

---

## 11. Artefacts

| Path | What |
|---|---|
| `scratchpad/makespan.py` | The makespan model, sensitivity and memory-ceiling tables (§2, §3.2) |
| `scratchpad/agg_probe.py` | Aggregation I/O decomposition (§5) |
| `scratchpad/agg_probe2.py` | Real `postprocess_output_root` timing per config (§5) |
| `scratchpad/boot_probe.py` | Bootstrap scaling + bit-identity of the vectorised form (§5) |
| `scratchpad/p2_testonly.sh`, `p2b.sh` | The `--test-only` sweep, run on Picasso (§7) |
| `scratchpad/wave_v2.psv`, `wave_v3.psv` | Raw `sacct` task timelines for the two `%24` waves (§2.1) |
| `scratchpad/mem_ceiling.py` | The `set[int]` measurement and the `max_evals`-bounded worst case (§10.1) |
| `scratchpad/verify_split.sh` | Byte-comparison of the aggregation paths on the real corpus (§10.4) |
| `scratchpad/isolate.py` | In-process loop-vs-vectorised diff that cleared the bootstrap (§10.4) |
| `scratchpad/p3_plan_testonly.sh` | `--test-only` at the new production shape, 42/42 (§10.4) |

Scratchpad root:
`/tmp/claude-1000/-home-mpascual-research-code-IsalSR/e632c995-8fc8-4171-aa37-d3fe5ffb2656/scratchpad/`
