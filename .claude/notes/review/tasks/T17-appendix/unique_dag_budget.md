# The unique-DAG budget proposal — investigation

**Date:** 2026-08-05
**Branch:** `feature/cpp-core-port`, HEAD `8fe5018`; Stage D deployed commit `00635ae`
**Status:** investigation only. No config edited, nothing submitted to Picasso, nothing
committed. Stage D untouched (7 cells still RUNNING at the time of writing).
**Scope:** resolve what terminates the Stage D Bingo cells; test hypotheses (a)–(e);
quantify the post-hoc route; scope the refactor; recommend.
**Addendum folded in (2026-08-05):** the corrected throughput break-even (§4f) and the
adapter-conversion accounting proposal (§4g).

---

## 0. Summary of what changed as a result of this investigation

Four premises that the brief and `audit.md` §F-17 treat as established are wrong or
incomplete. They are corrected here, with evidence.

| # | Premise as stated | Finding |
|---|---|---|
| 1 | "What stopped the dedup arms is UNRESOLVED" | **RESOLVED.** All six completed Bingo cells stopped on the *same* rule — Bingo status 3, `max_fitness_evaluations` — at 100 M ± 0.07 %. |
| 2 | F-17: "the dedup arms … keep running long after the baseline has spent its 100 M" | **REFUTED on these cells.** The dedup arms spent their 100 M too. The counter that reads 11–13 M is `dedup.n_total` (candidates), a different quantity from Bingo's internal fitness-invocation counter. |
| 3 | F-11 / brief: "LM inflation 3.3–4.1×" | **REFUTED.** Measured 13.8–16.3× on Stage D. The brief's hypothesis-(e) arithmetic inherits the error and is void. |
| 4 | "the baseline's candidate count is unrecorded" | **PARTLY REFUTED.** It is exactly recoverable *post hoc* as `501 × generations` (AgeFitnessEA). What is genuinely unmeasured is only the baseline's **canonically distinct** count. |

And two new findings that neither the brief nor the audit contains:

> **F-19 (new, and it is a launch blocker).** The seven production Bingo configs run
> under **two different evaluation caps**. Four (`nguyen`, `feynman`, `hard`,
> `cherrypicked`) plus `hard_trace` set `max_evals: 100000000`. **Three
> (`feynman_remainder`, `roundoff`, `strogatz`) set no `max_evals` at all** and therefore
> inherit `BingoConfig.max_evals = 10_000_000` (`experiments/models/bingo/config.py:29`;
> `from_dict` at `config.py:58-64` filters to known fields and falls back to the
> dataclass default). Those three suites run Bingo at a **10× tighter** budget than the
> other four. At the hard tier's measured throughput that is ~615 k–725 k candidates
> instead of ~6.2 M–7.3 M. Nothing in `EXECUTION-PLAN.md`, `audit.md` or any design
> document records this as a decision; it is an omission. It is invisible under the
> current framing precisely because `max_evals` was never recognised as a stop rule
> (F-17). Evidence:
>
> ```
> bingo_cherrypicked.yaml       max_evals=100000000  max_time=43200
> bingo_feynman_remainder.yaml  max_evals=<absent>   max_time=43200   -> 10_000_000
> bingo_feynman.yaml            max_evals=100000000  max_time=43200
> bingo_hard_trace.yaml         max_evals=100000000  max_time=43200
> bingo_hard.yaml               max_evals=100000000  max_time=43200
> bingo_nguyen.yaml             max_evals=100000000  max_time=43200
> bingo_roundoff.yaml           max_evals=<absent>   max_time=43200   -> 10_000_000
> bingo_strogatz.yaml           max_evals=<absent>   max_time=43200   -> 10_000_000
> ```
>
> This makes option (i) below **mandatory rather than optional**: whichever protocol is
> chosen, three of seven Bingo suites are currently running a different one. It also
> means C1's pooled CPDT (N = 42, and N = 50 with the extensions) mixes suites run at
> two budgets.

> **F-18 (new).** At an identical fitness-invocation budget, on identical hardware, the
> Bingo `isalsr` arm's per-evaluated-individual cost is **1.50× (Pagie-1) and 4.03×
> (Vladislavleva-2)** the baseline's, measured on `wall_clock_search_only_s` — which by
> construction already excludes canonicalisation, conversion and shadow. Consequently
> **73 % (Pagie-1) and 81 % (Vladislavleva-2) of IsalSR's wall-clock excess is not booked
> as overhead anywhere.** The reported `overhead_time_s / wall_clock_total_s` of 11.3 % /
> 15.0 % is a true statement about canon + conversion and a badly incomplete statement
> about the cost of the arm. This runs **against the competitor**, i.e. in IsalSR's favour,
> and it is the first thing a reviewer computes from Table 2.

---

## 1. Work log

Commands in order. Outputs are quoted in §2–§4. Local python is
`/home/mpascual/.conda/envs/isalsr/bin/python`; remote access is read-only `ssh picasso`
(`picasso3.scbi.uma.es`), consistent with SP-0.

| # | What | Command / file |
|---|---|---|
| 1 | Index the ledger, audit, handoff, configs, Bingo runners | `grep -n '^#\{1,4\} '` over `EXECUTION-PLAN.md`, `audit.md`; `cat T17-HANDOFF.md`, `bingo_hard.yaml`, `bingo_hard_trace.yaml` |
| 2 | Locate the stop logic | `grep -n 'max_time\|max_evals\|evolve_until_convergence' experiments/models/bingo/runner.py` → lines 384–389; `experiments/models/bingo/config.py:27–29` |
| 3 | Read Bingo's exit criteria | `sed -n '294,420p' $CONDA/site-packages/bingo/evolutionary_optimizers/evolutionary_optimizer.py`; `checkpoint_controller.py` |
| 4 | Locate Stage D logs (the brief's path returned nothing because logs live under `logs/`) | `ssh picasso 'find ~/fscratch/execs_logs/isalsr/c2_stage_d -maxdepth 3'` → `/mnt/home/users/tic_163_uma/mpascual/fscratch/execs_logs/isalsr/c2_stage_d/logs/` |
| 5 | **Map every Bingo exit message to its cell** | `ssh picasso` loop over `c2d_bingo_*.err` + `c2d_bingo_*.out` (§3) |
| 6 | Job states | `ssh picasso 'sacct -j 1769422,1769423,1769424 -X'` |
| 7 | Pull run logs + trajectories (12 files, 1.0 MB) | `rsync -az --include='run_log.json' --include='trajectory.csv' picasso:…/c2_stage_d/ $SCRATCH/stage_d/` |
| 8 | Pull Stage C v4 run logs (1,260 files) for the UDFS half | `rsync … picasso:…/c2_smoke_v4/ $SCRATCH/smoke_v4/` |
| 9 | Analysis A–G (candidate rate, LM inflation, per-invocation cost, equal-budget truncation, break-even, measurement cost) | local python over the pulled artefacts; all tables in §2 |
| 10 | Cost-attribution provenance | `sed -n '178,228p' audit.md` (F-7), `sed -n '399,440p' audit.md` (§4.1); `grep -n 'overhead_time_s' experiments/models/*/translator.py` |

Nothing was written to the repository except this file.

---

## 2. Results

### 2.1 Stage D state (2026-08-05, `sacct -j 1769422,1769423,1769424 -X`)

```
1769423_1 c2d_bingo_std     COMPLETED  03:02:04    bingo/baseline/Pagie-1/101
1769423_2 c2d_bingo_std     COMPLETED  05:36:32    bingo/hash/Pagie-1/101
1769423_3 c2d_bingo_std     RUNNING    06:03:31    bingo/baseline/Korns-12/101
1769423_4 c2d_bingo_std     RUNNING    06:03:31    bingo/hash/Korns-12/101
1769423_5 c2d_bingo_std     COMPLETED  01:00:07    bingo/baseline/Vladislavleva-2/101
1769423_6 c2d_bingo_std     COMPLETED  03:28:41    bingo/hash/Vladislavleva-2/101
1769424_1 c2d_bingo_isalsr  COMPLETED  05:13:31    bingo/isalsr/Pagie-1/101
1769424_2 c2d_bingo_isalsr  RUNNING    06:03:31    bingo/isalsr/Korns-12/101
1769424_3 c2d_bingo_isalsr  COMPLETED  04:37:33    bingo/isalsr/Vladislavleva-2/101
1769424_4 c2d_bingo_isalsr  RUNNING    06:03:31    bingo/isalsr/Pagie-1/102  (D2 trace)
1769422_{1,2,3} c2d_udfs    RUNNING    06:03:31
```

All six completed cells ran on **AMD EPYC 7H12 / 128 core / 503.7 GB** nodes
(`sr004, sr014, sr056, sr097, sr118`), same OS, same `git_describe=00635ae`,
`git_dirty=False`, `engine=native`, `build_hash=298fc118…`, `isa_level=x86-64-v3`,
`compiler=gcc 13.2.0`. Hardware is not a confound for anything below; **node identity
is not controlled** (six cells, five nodes), which is the one confound I cannot exclude
for F-18.

### 2.2 The primary table, with every counter named

`run_log.json`, `results.{time,search_space,regression}`; "evals (log)" from the
SLURM `.err` exit message (§3).

| cell | wall_s | search_only_s | canon_s | conv_s | `total_dags_explored` | `unique_canonical_dags` | ρ | evals (log) | R²_test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pagie_1/baseline | 10,909.5 | 10,909.5 | 0.0 | 0.0 | 100,070,885 | 100,070,885 | 1.000 | 100,070,885 | 0.7488 |
| pagie_1/hash | 20,179.0 | 18,535.5 | 481.8 | 1,161.7 | 11,488,430 | 6,434,660 | 1.785 | 100,007,430 | 0.5760 |
| pagie_1/isalsr | 18,789.9 | 16,670.7 | 906.3 | 1,212.9 | 11,453,360 | 6,262,371 | 1.829 | 100,008,393 | 0.7882 |
| vlad_2/baseline | 3,593.4 | 3,593.4 | 0.0 | 0.0 | 100,054,500 | 100,054,500 | 1.000 | 100,054,500 | 0.9985 |
| vlad_2/hash | 12,501.0 | 10,742.4 | 515.3 | 1,243.3 | 12,740,930 | 7,209,039 | 1.767 | 100,003,489 | 0.9972 |
| vlad_2/isalsr | 16,627.7 | 14,127.9 | 1,083.4 | 1,416.4 | 12,951,350 | 7,085,963 | 1.828 | 100,035,803 | 0.9940 |

`shadow_time_s = 0.0` and `shadow_hash = False` on all six (config as intended).
`n_canon_timeouts = n_canon_raised = n_conversion_failures = n_atlas_hits = 0` on all
four dedup cells. `solution_recovered = False` on all six.

**The three columns are three different quantities.** On the baseline,
`total_dags_explored = unique_canonical_dags = evals(log)` — all three are Bingo's
fitness-invocation counter (`bingo/runner.py:436-437`,
`n_total_dags=total_evals, n_unique_canonical=total_evals  # baseline: all unique`).
On the dedup arms `total_dags_explored = dedup.n_total` and
`unique_canonical_dags = dedup.n_unique` are candidate counts
(`bingo/isalsr_runner.py:951-952`), while `evals(log)` is the invocation counter.
This is F-11 restated with numbers.

### 2.3 The baseline's candidate count is exactly recoverable

`build_bingo_pipeline` builds an **`AgeFitnessEA`** (`bingo/runner.py:244-251`), which
per generation emits `population_size` offspring plus one age-zero random injection:
**501 new candidates/generation** at `population_size = 500`.

Measured on the four dedup arms, `total_dags_explored / last_iteration`:

| cell | last iteration | `n_total` | per generation |
|---|---:|---:|---:|
| pagie_1/hash | 22,930 | 11,488,430 | **501.02** |
| pagie_1/isalsr | 22,860 | 11,453,360 | **501.02** |
| vlad_2/hash | 25,430 | 12,740,930 | **501.02** |
| vlad_2/isalsr | 25,850 | 12,951,350 | **501.02** |

Four independent cells agree to 5 significant figures. The identity
`candidates = 501 × generations` therefore holds for the baseline too, and its candidate
count is:

| cell | generations | candidates = 501 × G |
|---|---:|---:|
| pagie_1/baseline | 12,300 | **6,162,300** |
| vlad_2/baseline | 14,480 | **7,254,480** |

(`generations` is the last trajectory snapshot at `snapshot_frequency = 10`, so these are
lower bounds accurate to <0.1 %.)

### 2.4 LM inflation, measured — and the brief's 3.3–4.1× is wrong

`evals / individuals-evaluated`:

| cell | evals | individuals evaluated | **LM invocations per individual** |
|---|---:|---:|---:|
| pagie_1/baseline | 100,070,885 | 6,162,300 (=501×G) | **16.24** |
| pagie_1/hash | 100,007,430 | 6,434,660 (=`n_unique`) | **15.54** |
| pagie_1/isalsr | 100,008,393 | 6,262,371 | **15.97** |
| vlad_2/baseline | 100,054,500 | 7,254,480 | **13.79** |
| vlad_2/hash | 100,003,489 | 7,209,039 | **13.87** |
| vlad_2/isalsr | 100,035,803 | 7,085,963 | **14.12** |

Within a problem the three arms agree to ≤4.5 %. This is a strong internal consistency
check: two independent routes to "individuals evaluated" (501×G for the baseline,
`dedup.n_unique` for the dedup arms) land on the same LM factor. `audit.md:259-260`'s
**3.3–4.1×** is therefore not the factor operating in the hard tier and should be
retracted or re-scoped to whatever configuration produced it (UNVERIFIED which).

**Consequence.** The three arms are, as executed, at an **approximately equal
evaluation budget in every sense that matters**: equal fitness invocations (100 M
± 0.07 %) *and* near-equal numbers of individuals put through the local optimiser
(6.16–6.43 M on Pagie-1; 7.09–7.25 M on Vladislavleva-2). What differs is that the
baseline spent those evaluations on a stream containing isomorphic duplicates, while
IsalSR spent them on 6.26 M / 7.09 M *canonically distinct* structures having generated
11.45 M / 12.95 M candidates. That is the paper's claim, and the campaign is already
measuring it — accidentally, and not under the protocol the manuscript describes.

### 2.5 F-18 — the unbooked per-evaluation slowdown

Per fitness invocation, `wall_clock_search_only_s / evals`:

| problem | baseline | hash | isalsr | hash/base | isalsr/base |
|---|---:|---:|---:|---:|---:|
| pagie_1 | 109.0 µs | 185.3 µs | 166.7 µs | 1.70× | **1.53×** |
| vlad_2 | 35.9 µs | 107.4 µs | 141.2 µs | 2.99× | **3.93×** |

Per **evaluated individual** (the quantity the break-even algebra needs),
`search_only / individuals`:

| problem | e_B (baseline) | e_I (isalsr) | e_I/e_B |
|---|---:|---:|---:|
| pagie_1 | 1,770 µs | 2,662 µs | **1.504** |
| vlad_2 | 495 µs | 1,994 µs | **4.028** |

`wall_clock_search_only_s` is defined as `wall − canon − conversion − shadow`
(`bingo/translator.py:118`), so this excess is *not* canonicalisation, *not* conversion
and *not* the shadow sketches. It is unattributed.

How much of IsalSR's wall-clock excess the reported overhead captures:

| problem | excess vs baseline | `overhead_time_s` | share booked | share **unbooked** |
|---|---:|---:|---:|---:|
| pagie_1 | 7,880.4 s | 2,119.2 s | 26.9 % | **73.1 %** |
| vlad_2 | 13,034.3 s | 2,499.8 s | 19.2 % | **80.8 %** |

**Leading hypotheses, neither confirmed:**

- **H1 — memory/allocator pressure.** The `isalsr` cell holds a ~7.1 M-entry
  `set[int]` (plus the atlas); this is why §3.3 sizes it at 256 GB against the
  baseline's 32 GB. The effect is far larger on Vladislavleva-2 (100 training points,
  arrays that fit in L1 for the baseline) than on Pagie-1 (676 points). *That
  ordering is the signature of cache/TLB pressure*, and it is the reason I rank H1
  first.
- **H2 — structural selection.** Deduplication prevents the population collapsing onto
  a few cheap structures, so the surviving individuals are systematically more
  expensive to evaluate. `model_complexity` is 31–32 on all six cells (capped by
  `stack_size = 32`), so this is not visible in the recorded fields.
- **H3 — node co-tenancy.** Six cells, five distinct nodes, memory-bandwidth
  contention uncontrolled. Cannot be excluded from these artefacts.

Distinguishing H1/H2/H3 needs a controlled probe (same node, dedup on/off, RSS traced).
`rss_timeseries.csv` exists per cell on Picasso and would test H1 directly; I did not
run that analysis here.

### 2.6 Equal-budget truncation of the six trajectories (task 3)

`best_r2` in `trajectory.csv` is **train** R² (verified: `pagie_1/baseline` last row
0.9898630732 = `results.regression.r2_train` 0.98986307316649). Running maximum taken.

**(1) Equal unique-canonical-DAG budget — hash vs isalsr only.**
The baseline cannot appear: its `n_unique_canonical` is `eval_count` by construction.

*Pagie-1* (common budget 6,262,371):

| unique DAGs | hash R² | hash t (s) | isalsr R² | isalsr t (s) | Δ(i−h) |
|---:|---:|---:|---:|---:|---:|
| 626,237 | 0.975415 | 994 | 0.964385 | 1,329 | −0.011030 |
| 1,565,592 | 0.991404 | 3,314 | 0.991568 | 3,379 | +0.000164 |
| 3,131,185 | 0.992511 | 7,775 | 0.992604 | 7,758 | +0.000093 |
| 4,696,778 | 0.992558 | 13,156 | 0.995287 | 12,619 | +0.002729 |
| 6,262,371 | 0.992558 | 19,414 | **0.998244** | 18,790 | **+0.005686** |

*Vladislavleva-2* (common budget 7,085,963):

| unique DAGs | hash R² | hash t (s) | isalsr R² | isalsr t (s) | Δ(i−h) |
|---:|---:|---:|---:|---:|---:|
| 708,596 | 0.997803 | 717 | 0.989252 | 939 | −0.008550 |
| 1,771,490 | 0.998138 | 2,088 | 0.992662 | 2,651 | −0.005476 |
| 3,542,981 | 0.998167 | 4,780 | 0.993750 | 6,302 | −0.004417 |
| 5,314,472 | 0.998239 | 7,997 | 0.994545 | 10,877 | −0.003695 |
| 7,085,963 | **0.998348** | 12,192 | 0.995752 | 16,628 | **−0.002596** |

IsalSR beats naive hash at equal unique-DAG budget on Pagie-1 at every level from 25 %
onward, and **loses to it at every level on Vladislavleva-2**, in R² *and* in time.
n = 1 seed per cell; this is an anecdote, not a result.

**(2) Equal candidates-generated budget — all three arms** (baseline via 501×G; this
needs no new code).

*Pagie-1* (common 6,162,300):

| candidates | baseline R² (t s) | hash R² (t s) | isalsr R² (t s) |
|---:|---:|---:|---:|
| 616,230 | 0.934274 (578) | 0.954660 (525) | 0.939352 (627) |
| 3,081,150 | 0.976657 (4,877) | 0.992511 (3,730) | 0.992207 (3,791) |
| 6,162,300 | 0.989863 (10,909) | 0.992511 (8,859) | **0.995236** (8,472) |

*Vladislavleva-2* (common 7,254,480):

| candidates | baseline R² (t s) | hash R² (t s) | isalsr R² (t s) |
|---:|---:|---:|---:|
| 725,448 | 0.989339 (242) | 0.994099 (383) | 0.982110 (502) |
| 3,627,240 | 0.998918 (1,644) | 0.998158 (2,489) | 0.992662 (3,045) |
| 7,254,480 | **0.998962** (3,593) | 0.998172 (5,756) | 0.993750 (7,368) |

**(3) Equal wall clock — all three arms**, truncated at the shortest arm's total:

*Pagie-1*, at 10,909 s: baseline 0.989863, hash 0.992558, **isalsr 0.995236**.
*Vladislavleva-2*, at 3,593 s: **baseline 0.998962**, hash 0.998163, isalsr 0.992662.

**The two trace problems disagree on every axis.** Pagie-1 is the structural-bottleneck
problem where the 2026-04-19 analysis predicts IsalSR helps, and it does, on all three
axes. Vladislavleva-2 is `bottleneck = structural_depth (k ≥ 12)`, predicted *not* to
benefit, and IsalSR loses on all three axes. That is consistent with the existing
bottleneck-type theory, which is reassuring — but it means the choice of budget does not
rescue Vladislavleva-2 and no budget will.

### 2.7 What cannot be computed without the baseline measurement

Exactly one thing, and everything downstream of it:

> **D_base — the number of canonically distinct DAGs among the baseline's candidate
> stream** (equivalently ρ_base = 501·G / D_base).

Consequences of not having it:

1. "IsalSR explores k× more distinct structures than the baseline" is **not computable
   at all** — for any budget definition. The baseline's ρ = 1.000 is definitional
   (`bingo/runner.py:437`), which T17-HANDOFF decision 3 already concedes for the
   inferential ρ test but which *also* silently blocks this claim.
2. A budget "counted in canonically-distinct DAGs" cannot be **enforced** on the
   baseline arm without it, so Mario's proposal is not implementable on the baseline
   without this measurement. This is the load-bearing dependency.
3. The break-even in §4f cannot be evaluated without assuming ρ_base = ρ_isalsr, an
   assumption I show below is (i) not neutral and (ii) probably conservative in
   IsalSR's favour, but unquantified.

Everything else — candidate counts, LM factors, per-invocation costs, all three
truncation analyses in §2.6 except the baseline's unique-axis row — is available **post
hoc from the artefacts already on disk**.

### 2.8 Cost of measuring D_base

Per-candidate representation cost on the isalsr arm,
`(canon + conversion) / total_dags_explored`: **185.0 µs** (Pagie-1), **193.0 µs**
(Vladislavleva-2). Applying it to the baseline's candidate stream:

| problem | C_base | added time | % of baseline wall | % of a full 12 h run |
|---|---:|---:|---:|---:|
| pagie_1 | 6,162,300 | 1,140 s | 10.5 % | 2.6 % |
| pagie_1 @12 h | 24,401,793 | 4,515 s | — | 10.5 % |
| vlad_2 | 7,254,480 | 1,400 s | 39.0 % | 3.2 % |
| vlad_2 @12 h | 87,213,652 | 16,834 s | — | **39.0 %** |

Full-rate canonical accounting on the baseline costs **10.5–39.0 %** of that arm's wall
clock — not the ≈17.6 % the shadow-sketch decision was taken against, and on
Vladislavleva-2 substantially worse. That number matters for §4d.

---

## 3. The stop-reason finding (task 1) — FACT, with log evidence

**All six completed Bingo cells terminated on Bingo's exit status 3,
`max_fitness_evaluations`.** The stop rule is *identical* across all three arms.

Mechanism (`$CONDA/site-packages/bingo/evolutionary_optimizers/evolutionary_optimizer.py`):
`_check_exit_criteria` tests, in order, `_convergence` (0) → `_stagnation` (1, inert:
`stagnation_generations` is not passed, and `_stagnation` returns `False` on `None`) →
`_hit_max_evals` (3) → `_hit_time_limit` (4) → `_not_enough_time_for_another_checkpoint`
(5). `_hit_max_evals` compares `self.get_fitness_evaluation_count()`, i.e.
`Evaluation.eval_count` → `fitness_function.eval_count`, against `cfg.max_evals`.

Log evidence — `~/fscratch/execs_logs/isalsr/c2_stage_d/logs/*.err`, message text from
`_make_optim_result(status=3)`:

| `.err` file | cell (from the matching `.out`) | message |
|---|---|---|
| `c2d_bingo_std_1769423_1.err` | bingo/**baseline**/Pagie-1/101 | "The maximum number of fitness evaluations (100000000) was exceeded. Total fitness evals: **100070885**" |
| `c2d_bingo_std_1769423_2.err` | bingo/**hash**/Pagie-1/101 | "… Total fitness evals: **100007430**" |
| `c2d_bingo_std_1769423_5.err` | bingo/**baseline**/Vladislavleva-2/101 | "… Total fitness evals: **100054500**" |
| `c2d_bingo_std_1769423_6.err` | bingo/**hash**/Vladislavleva-2/101 | "… Total fitness evals: **100003489**" |
| `c2d_bingo_isalsr_1769424_1.err` | bingo/**isalsr**/Pagie-1/101 | "… Total fitness evals: **100008393**" |
| `c2d_bingo_isalsr_1769424_3.err` | bingo/**isalsr**/Vladislavleva-2/101 | "… Total fitness evals: **100035803**" |

No cell logged status 0 (convergence), 1 (stagnation), 2 (max generations), 4 (max time)
or 5 (pre-emptive checkpoint stop). Status 5 was the plausible alternative and is ruled
out arithmetically as well as empirically: it fires when
`remaining_time / gen_speed / check_freq < 0.25`, which at `check_freq = 10` and
~23,000 s remaining would need >10,000 s per generation.

**The confound is therefore not "two different stop rules".** Within these six cells it
is a *single* stop rule, expressed in a currency (fitness invocations) that the campaign
does not report, reached at wall-clock times differing by up to 4.63×. F-17's mechanism
paragraph — "their evaluation counter advances more slowly and they keep running long
after the baseline has spent its 100 M" — is **wrong for these cells** and should be
corrected in `audit.md`.

**But F-17's conclusion survives, by a different route.** On C1 (`audit.md` F-17
table) **52/264** Bingo `isalsr` cells ran ≥95 % of 12 h — those hit `max_time`
(status 4) while their paired baselines hit `max_evals` (status 3). So across the
campaign the binding rule *does* vary by cell and by arm; it simply did not vary in
these six. That heterogeneity is the real defect: it is not that `max_evals` binds, it
is that **which of `max_evals` and `max_time` binds is a per-cell accident**, so neither
"equal time" nor "equal evaluations" is true campaign-wide.

---

## 4. Verdicts

### (a) "There is a much cheaper fix for the immediate bug" — **CONFIRMED, with a caveat that changes the recommendation**

*Mechanically confirmed.* `max_evals` reaches `evolve_until_convergence` at exactly two
sites: `bingo/runner.py:387` and `bingo/isalsr_runner.py:849`, both
`max_fitness_evaluations=cfg.max_evals`. Default `BingoConfig.max_evals = 10_000_000`
(`bingo/config.py:29`). Five production configs carry `max_evals: 100000000`
(`bingo_{nguyen,feynman,hard,cherrypicked}.yaml`, `bingo_hard_trace.yaml`); **three
carry none and silently inherit 10 M — see F-19, §0**; two debug configs carry 5000.
**No UDFS config has `max_evals` at all** — `udfs_hard.yaml:27` has
only `max_time: 43200`, which is why UDFS shows 300/300 paired cells at exactly 43,200 s.

*Is it needed?* No. It is a safety cap with no scientific role. Raising it to e.g. 10¹²
makes `max_time` the sole binding rule on Bingo, matching UDFS and matching the
manuscript's stated 12 h budget and `EXECUTION-PLAN.md` §5.4's *"Every arm runs the full
43,200 s budget."* One-line change in five YAML files.

*The caveat, and it is not small.* Under equal wall clock the trace-problem evidence
does **not** uniformly favour IsalSR:

| problem | R² at equal wall clock (10,909 s / 3,593 s) | winner |
|---|---|---|
| pagie_1 | base 0.98986, hash 0.99256, **isalsr 0.99524** | IsalSR |
| vlad_2 | **base 0.99896**, hash 0.99816, isalsr 0.99266 | baseline |

Extrapolating each arm's measured throughput to a genuine 43,200 s:

| problem | baseline candidates @12 h | isalsr candidates @12 h | isalsr **unique** @12 h (measured rate) |
|---|---:|---:|---:|
| pagie_1 | 24.40 M | 26.33 M | 14.40 M |
| vlad_2 | **87.21 M** | 33.65 M | 18.41 M |

So option (A) restores the protocol the paper claims, at a cost of ~9 extra core-hours
per affected Bingo baseline cell, and it hands the reviewer a Vladislavleva-2 row in
which the baseline generated 2.59× more candidates than IsalSR. **Do it anyway** — the
alternative is publishing a protocol description that is false as executed — but do it
with eyes open, and do it *together* with the D_base measurement (§4c/§4d), because
without D_base you cannot answer "and how many of those 87 M were distinct?".

### (b) "The proposal as literally stated is unfair in the opposite direction" — **CONFIRMED**

If the budget is "N, counted in each arm's own currency", the baseline's currency *is*
its raw candidate count (it has no other; ρ = 1.000 definitionally). At N = 7 M the
baseline stops after 7 M generated candidates, of which ≈7/ρ_base M are distinct; IsalSR
stops after 7 M *distinct*, having generated ρ·7 M ≈ 12.8 M. IsalSR receives ≈ρ_base ≈
1.8× more search. The fix the brief proposes — canonicalise the baseline **for
accounting only, never for dedup** — is the only formulation that makes the budget
equal, and it is exactly the D_base measurement.

*One addition.* Enforcing a distinct-count budget on the baseline requires the canonical
key **online, at full rate, inside the run**. That crosses `EXECUTION-PLAN.md` §3's
dividing line ("anything measured *during* a run must be in the code before launch") and
costs the baseline arm 10.5–39.0 % of its wall clock (§2.8). If instead the budget is
enforced in the *invocation* currency and D_base is merely *recorded*, the same
measurement is needed but no arm's stop rule changes — a materially cheaper design.

### (c) "Most of this is computable post hoc" — **CONFIRMED, and more so than the brief claims**

Three of the four quantities are already on disk:

| quantity | availability |
|---|---|
| candidates generated, all three arms | **available post hoc** — `501 × generations` from `trajectory.csv` (§2.3). The brief assumed this needed new code; it does not. |
| fitness invocations, all three arms | available (`.err` exit message; and `total_dags_explored` on the baseline) |
| distinct structures, dedup arms | available (`n_unique_canonical`, real) |
| **distinct structures, baseline** | **NOT AVAILABLE.** `bingo/runner.py:437` sets it to `total_evals`. The single blocking gap. |

Quality-at-equal-unique-budget for hash vs isalsr is fully computable today (§2.6(1)).
Quality-at-equal-candidate-budget for all three arms is fully computable today
(§2.6(2)) and is a *legitimate, publishable* equal-budget analysis that nobody has run.
The brief's claim that the baseline measurement is "the ONLY forward-looking code change
strictly required" is **correct**.

### (d) "Measuring the baseline's unique count is affordable if subsampled" — **PARTIAL: cost CONFIRMED, subsampling REFUTED**

*Cost, confirmed and re-measured.* Full-rate canonical accounting on the baseline costs
**10.5 % (Pagie-1) / 39.0 % (Vladislavleva-2)** of that arm's wall clock (§2.8) — worse
than the brief's ≈17.6 % anchor, because the shadow figure was measured on the *isalsr*
arm whose per-candidate budget is spread over ~16 LM invocations, whereas the baseline's
candidates are cheaper per unit. The brief's structural argument is nevertheless right:
**under a count budget this cost does not steal search**, it only inflates measured
time, and `wall_clock_search_only_s` already isolates it.

*Subsampling, refuted.* A 1-in-100 subsample does **not** estimate a distinct-element
count. Estimating the number of distinct values from a sample is a known-hard problem
with a matching lower bound: any estimator based on a fraction *q* of the data has ratio
error Ω(√(1/q)) on some input (Charikar, Chaudhuri, Motwani, Narasayya, *Towards
estimating the number of distinct values of an attribute*, PODS 2000). At q = 0.01 that
is a ~10× error bar on D_base — useless for a quantity whose break-even threshold
(§4e) sits between 1.7 and 4.7. Subsampling works for *rates* (µs per candidate, which
is what the D2 tracer samples at 1-in-100) and fails for *cardinalities*.

The correct instrument is the one already built: HyperLogLog at p = 16 over the **full**
stream (±0.41 %). The 64 KB sketch is free; the cost is the canonical key, which cannot
be subsampled away. So: full-rate canonicalisation on the baseline, keyed into an HLL,
dedup **off** — precisely the shadow-sketch machinery pointed at the baseline arm.

### (e) "The answer may not favour IsalSR" — **PARTIAL: the arithmetic is REFUTED, the caution is CONFIRMED on one of two problems**

*The arithmetic is void.* It used LM inflation 3.3–4.1×, giving "24–30 M baseline
candidates" on Vladislavleva-2. Measured inflation is **13.79×** and the baseline
generated **7,254,480** candidates, not 24–30 M (§2.4). The conclusion drawn from that
arithmetic — that the baseline might cover 13–16 M distinct against IsalSR's 7.09 M *at
the current budget* — is wrong by roughly 4×. **At the current equal-invocation budget
IsalSR strictly wins the coverage comparison**, and the coordinator's addendum
correcting me on this point is right: under a distinct-count budget IsalSR gets N
genuinely distinct structures while the baseline's nominal N contains only N/ρ_base
distinct ones. Coverage is not where the risk lies.

*The caution survives on the throughput axis, and only on one problem.* At **equal wall
clock** and assuming ρ_base = ρ_isalsr = 1.83:

| problem | baseline distinct @12 h | isalsr distinct @12 h | ratio |
|---|---:|---:|---:|
| pagie_1 | 13.33 M | 14.40 M | 1.08 in IsalSR's favour |
| vlad_2 | 47.66 M | 18.41 M | **2.59 against IsalSR** |

Equivalently, the break-even baseline redundancy above which IsalSR explores more
distinct structures per second is **ρ\* = (C_base/T_base)·(T_isalsr/U_isalsr)**:

| problem | ρ\* | measured ρ_isalsr | survives iff |
|---|---:|---:|---|
| pagie_1 | **1.695** | 1.829 | ρ_base > 1.70 — *plausible* |
| vlad_2 | **4.737** | 1.828 | ρ_base > 4.74 — *unlikely* |

There is a principled reason to expect ρ_base > ρ_isalsr: the dedup arm removes
duplicates from its population, so its offspring stream is less redundant *by
construction*, and the baseline's population accumulates isomorphic copies through
drift. B12's ~36 % verbatim-clone rate alone puts a floor of ρ ≥ 1/(1−0.36) = 1.56 on
both arms. But nothing in the artefacts bounds ρ_base above 4.74. **This is the strongest
argument for measuring D_base: it converts a 2.6× exposure into a number.**

### (f) [Addendum 1] The corrected throughput break-even — **the coordinator's inequality is CORRECT as a special case, and its special case does not hold**

*Derivation.* Fix a common budget of N canonically distinct DAGs. Write ρ, ρ_base for
the redundancy of the isalsr and baseline candidate streams; c = t_canon + t_conv per
candidate on the isalsr arm; e_I, e_B for the per-evaluated-individual cost on the two
arms. IsalSR canonicalises ρN candidates and evaluates N of them; the baseline
generates ρ_base·N candidates and evaluates all of them:

  T_I = ρN·c + N·e_I  T_B = ρ_base·N·e_B

IsalSR wins on time iff ρ·c + e_I < ρ_base·e_B, i.e.

  **c / e_B < (ρ_base − e_I/e_B) / ρ**  … (★)

Setting e_I = e_B and ρ_base = ρ collapses (★) to **c/e < 1 − 1/ρ**, the coordinator's
form. At ρ = 1.83 the threshold is **0.4536**, matching the quoted 0.454. So the
derivation is right; what fails is its premise e_I = e_B.

*Evaluated on Stage D (ρ_base = ρ assumed; c, e from §2.5):*

| problem | c (µs) | e_I | e_B | e_I/e_B | LHS c/e_B | RHS (★) | naive RHS | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| pagie_1 | 185.0 | 2,662 | 1,770 | 1.504 | 0.1045 | **0.1777** | 0.4533 | **WIN** |
| vlad_2 | 193.1 | 1,994 | 495 | 4.028 | 0.3901 | **−1.2037** | 0.4530 | **LOSE** |

The naive form predicts a win on *both* problems and is contradicted by the observed
wall clock on Vladislavleva-2 (isalsr 16,628 s vs baseline 3,593 s at equal invocations).
The corrected form (★) reproduces the observation. On Vladislavleva-2 the RHS is
**negative**: e_I/e_B = 4.03 already exceeds ρ = 1.83, so *even with free
canonicalisation and free conversion IsalSR would still lose on time*. Canonicalisation
cost is not the operative mechanism there; F-18 is.

*Per-DAG costs, re-derived rather than taken from CLAUDE.md.* The CLAUDE.md production
table (Bingo t_canon ≈ 0.46 ms, t_eval ≈ 0.14 ms, ratio ≈ 3.3) predates the T01 C++
core port. Measured on `00635ae` with `engine=native`:

| host | source | ρ | t_canon / candidate | t_conv / candidate | t_shadow / candidate |
|---|---|---:|---:|---:|---:|
| Bingo isalsr | Stage D 12 h | 1.828 | **79–84 µs** | 106–109 µs | 0 (off) |
| Bingo hash | Stage D 12 h | 1.767–1.785 | **40–42 µs** | 101–108 µs | 0 (off) |
| Bingo isalsr | Stage C v4 900 s (n=210) | 1.795 | 68.6 µs | 99.9 µs | 192.9 µs |
| UDFS isalsr | Stage C v4 900 s (n=210) | 1.541 | 133.2 µs | 128.5 µs | 211.1 µs |
| UDFS hash | Stage C v4 900 s (n=210) | **1.000** | 55.1 µs | 162.1 µs | 0 |

Canonicalisation is **5.5× cheaper than CLAUDE.md's 0.46 ms**. That number should be
updated; it is quoted in the project brief and is now misleading.

*UDFS side of the prediction.* UDFS Stage D cells are all still RUNNING, so this is from
Stage C v4 (900 s) and is **provisional**. Median UDFS `isalsr`: 1,402 candidates / 954
unique in 900 s → ≈**943 ms per evaluated individual**, against t_canon+t_conv ≈ 262 µs
per candidate. LHS = c/e_B ≈ 2.8 × 10⁻⁴ against a threshold of 0.351 at ρ = 1.541 —
**three orders of magnitude inside the win region.** The addendum's UDFS prediction is
CONFIRMED in sign and margin, provisionally.

*Verdict on the addendum's overall prediction.* "IsalSR wins quality-at-equal-unique-DAGs
on both hosts, wins time on UDFS, loses time on Bingo":

- *wins time on UDFS* — **CONFIRMED (provisional, Stage C only).**
- *loses time on Bingo* — **CONFIRMED on Vladislavleva-2, REFUTED on Pagie-1** (IsalSR
  wins time on Pagie-1: (★) LHS 0.1045 < RHS 0.1777, and the observed equal-wall-clock
  R² also favours it). The Bingo outcome is problem-dependent, tracking the
  bottleneck-type classification rather than the host.
- *wins quality-at-equal-unique-DAGs on both hosts* — **REFUTED as stated on Bingo.**
  On Vladislavleva-2 IsalSR loses to *naive hash* at every budget level (§2.6(1)). Two
  problems, one seed; but the prediction is not free-standing.

Mario's framing point — significance on any one of the three axes is worth reporting —
is sound, and §2.6(1) on Pagie-1 (Δ R² = +0.0057 at the full common budget, isalsr over
hash, in less time) is exactly the shape of evidence it describes. It needs seeds.

### (g) [Addendum 2] Subtracting adapter-conversion time — **REFUTE. I agree with the coordinator, on stronger grounds than stated.**

**Objection (ii) — confirmed with exact citations.**

- `audit.md` **F-7** ("Untimed wrapper work booked as 'search' (both directions) —
  FIXED", lines 178–201) states: *"the reported **overhead % understates** the
  representation layer's true cost (conversion is genuinely part of the method) —
  *favors isalsr*"*, and prescribes `overhead_time_s = canon + conversion`, with shadow
  excluded *"— it is audit instrumentation, not method cost"*.
- `audit.md` **§4.1** (lines 399–438) records the fix as LANDED, with the measured
  consequence: *"the previously reported overhead % was understated **1.57× (UDFS) /
  2.43× (Bingo)**"*.
- `EXECUTION-PLAN.md` §11.1, ledger row dated **2026-08-04** ("Cost attribution
  corrected") lists *"overhead understated 1.57×/2.43× by excluding adapter conversion"*
  among the **pro-IsalSR** accounting defects, alongside CPDT tie-dropping and the
  tautological ρ p-value.
- Implementation: `experiments/models/{bingo,udfs}/translator.py:120`
  `overhead = r.canonicalization_time_s + r.conversion_time_s`;
  `schemas.py:109-110` — *"without it, so it is part of `overhead_time_s`"*.
  `c2_certify.py` `RUN_LOG_FIELD_SPEC` grew 56 → 60 fields for this.

The proposal is therefore a **literal revert of a fairness fix made one day ago and
classified at the time as removing a pro-IsalSR bias**, with the diff in our own history
and the reasoning in a document a reviewer could be shown. Objection (ii): **CONFIRMED.**

**Objection (i) — confirmed, and the "add it to the competitors" half is worse than
stated.** `conversion_time_s` on the Stage D baseline cells is **exactly 0.0** — not
small, zero — because the baseline never constructs a `LabeledDAG`
(`bingo/runner.py:95` default `0.0`, never assigned). Adding conversion to the baseline
would bill it for work it provably does not perform. The `hash` arm *does* pay
conversion (1,161.7 s / 1,243.3 s), because it converts before hashing; that is real and
correctly charged.

**Would it change headline numbers, and in which direction?**

| cell | overhead % (canon+conv, as reported) | overhead % (canon only) | inflation from the fix |
|---|---:|---:|---:|
| pagie_1/isalsr | 11.28 % | 4.82 % | 2.34× |
| vlad_2/isalsr | 15.03 % | 6.52 % | 2.31× |
| pagie_1/hash | 8.14 % | 2.39 % | **3.41×** |
| vlad_2/hash | 14.07 % | 4.12 % | **3.41×** |
| UDFS isalsr (Stage C) | 0.041 % | 0.021 % | 1.95× |

Excluding conversion would roughly **halve** Bingo's reported overhead and cut UDFS's by
about half. It is materially favourable to IsalSR. It also produces an
argument-defeating asymmetry: it flatters the **naive-hash competitor by 3.41×** against
IsalSR's 2.31–2.34×, because the hash arm's key is cheaper while its conversion is
identical — so the manoeuvre would *shrink the measured gap between IsalSR and the
straw-man it is meant to beat*. It damages the paper on its own terms.

`S = T_search^baseline / T_search^isalsr` (`analyze.py:856-857`) uses
`wall_clock_search_only_s`, which subtracts conversion from *both* numerator and
denominator; re-excluding conversion from `overhead_time_s` alone would leave `S`
unchanged but break the identity
`wall = search_only + overhead + shadow`, silently.

**My recommendation, independent of the coordinator's.** Refute the subtraction; adopt
the decomposition — which is the coordinator's alternative (iii), and I reach it
independently. Concretely:

1. Keep `overhead_time_s = canon + conversion` as the headline. Do not touch it.
2. Report the **three-way decomposition** `t_canon : t_conv : t_shadow` per candidate in
   the supplement — the fields already exist and are already populated
   (79/106/0 µs and 84/109/0 µs on the Stage D isalsr cells; 133/128/211 µs on UDFS
   Stage C).
3. Argue the integration point in prose: conversion is the cost of bolting the
   representation onto a host that stores DAGs in its own format
   (`AGraph.command_array`, UDFS `node_dict`); a native host would pay `t_canon` only.
   State it as a limitation with a number attached — "the intrinsic representation cost
   is 79 µs of the 185 µs charged" — not as a subtraction.
4. **New, and more urgent than either:** F-18 says the headline overhead is incomplete
   in the *opposite* direction by 3–4× the amount under discussion. Arguing about
   whether to remove 106 µs of conversion while 73–81 % of the arm's wall-clock excess
   is unattributed is optimising the wrong term. Resolve F-18 first.

---

## 5. Refactor scope

`EXECUTION-PLAN.md` §3: *"anything measured during a run must be in the code before
launch; anything computed after can land later."* Options are cumulative: (ii) ⊃ (i),
(iii) ⊃ (ii).

### (i) Minimal fix — make `max_time` the sole binding rule

| File | Change |
|---|---|
| `experiments/configs/bingo_nguyen.yaml` | `max_evals: 100000000` → `1000000000000` |
| `experiments/configs/bingo_feynman.yaml` | idem |
| `experiments/configs/bingo_hard.yaml` | idem |
| `experiments/configs/bingo_cherrypicked.yaml` | idem |
| `experiments/configs/bingo_hard_trace.yaml` | idem (must move in lockstep — it is byte-identical but for `shadow_hash`) |
| `experiments/configs/bingo_roundoff.yaml` | **F-19 — no `max_evals` key; inherits 10 M.** Must be set explicitly, not left to the default. |
| `experiments/configs/bingo_strogatz.yaml` | idem (F-19) |
| `experiments/configs/bingo_feynman_remainder.yaml` | idem (F-19) |
| `experiments/models/bingo/config.py:29` | consider raising the *default* too, so an omitted key cannot silently re-create F-19 |
| `tests/unit/` | new test asserting all production Bingo configs declare `max_evals` explicitly and agree — the same shape as `test_shadow_hash_config.py`, which exists precisely because a silently-ignored key bit us once already |
| `docs/…/computational_experiments.tex`, `EXECUTION-PLAN.md` §5.4 | protocol statement now true as executed |
| `audit.md` F-17 | correct the mechanism paragraph per §3 |

**Metrics whose MEANING changes:** none. Values change: every Bingo baseline cell
lengthens (1.0–3.5 h → 12 h), `wall_clock_total_s`, `S`, `overhead %`, Table 2's cost
column and all trajectory endpoints move. `ρ`, R², NRMSE, `solution_recovered` keep
their meaning and change value.
**Re-certification:** `config_sha256` moves ⇒ §5.1's one-commit/one-configuration rule
requires a fresh **Stage C** wave (1,260 tasks, ~35 min) — which is *already owed* on
`00635ae` (ledger 2026-08-05). **Stage D must be re-run** (13 cells, 156 core-h): D1.1
(12 h wall-limit headroom), D1.2 (`MaxRSS` at 12 h), D1.6 (ρ/R² vs C1) and D1.7
(overhead) are all budget-dependent and none of the current cells were produced under
the new rule. Cost: ~1.5 days of Stage C+D, dominated by Stage D's 12 h wall.

### (ii) (i) + measure D_base (record-only, no stop-rule change)

Adds, on top of (i):

| File | Change |
|---|---|
| `experiments/models/bingo/runner.py` | Replace the baseline `Evaluation` with a counting/accounting subclass: convert each candidate, compute the canonical key, feed an HLL; **never** deduplicate. Set `n_total_dags = 501×G` (or a direct counter) and `n_unique_canonical = HLL.cardinality()`. This is where line 437's `n_unique_canonical=total_evals` dies. |
| `experiments/models/udfs/runner.py` | Same for UDFS |
| `experiments/models/{bingo,udfs}/runner.py` (raw result) | New fields for accounting time and the HLL estimate |
| `experiments/models/schemas.py` | `SearchSpaceResults`: `baseline_distinct_estimated: bool`, `accounting_time_s`; `TimeResults`: `accounting_time_s` |
| `experiments/models/{bingo,udfs}/translator.py` | `wall_clock_search_only_s = wall − canon − conversion − shadow − accounting`; **`accounting_time_s` must NOT enter `overhead_time_s`** (same rule as shadow: instrumentation, not method) |
| `experiments/models/bingo/isalsr_runner.py` (`_serial_eval` snapshot path) | Emit real `n_unique_canonical` on the baseline trajectory too |
| `experiments/scripts/c2_certify.py` | `RUN_LOG_FIELD_SPEC` 60 → ~63 |
| `experiments/models/analyzer/aggregation.py` | `METRIC_EXTRACTORS` — ρ becomes a measured quantity on all three arms; F-3's "descriptive vs inferential" split (T17-HANDOFF decision 3) **can be lifted** |
| `experiments/models/analyze.py`, `analyzer/cross_method.py`, `figures/models/generate_tables.py` | three-arm ρ with a real baseline (this is already A8's open work) |
| tests | new suite mirroring `test_cost_attribution.py`; HLL accuracy test; a test that the baseline arm's fitness stream is bit-identical with accounting on/off (the C3 "dedup-off equivalence" pattern) |

**Metrics whose MEANING changes:**
- `empirical_reduction_factor` on the **baseline** — from a definitional 1.000 to a
  measurement. **This retires T17-HANDOFF decision 3 and audit F-3.** It is the single
  highest-value change on this list.
- `unique_canonical_dags` on the baseline — from "= eval_count" to "distinct canonical
  DAGs". A cross-campaign comparison against C1 on this field becomes invalid.
- `wall_clock_search_only_s` — a fourth subtrahend. §4.1's warning applies again.

**Re-certification:** everything in (i), plus **Stage A** (unit/ruff/mypy), **Stage B**
(the SP probes; the equivalence gate must show the accounting path does not perturb the
baseline's search — an SP-3-style negative control), **Stage C** (field count 60→63
invalidates every pre-change artefact for C1.2), **Stage D** (D1.2 memory: the baseline
now carries an HLL and a converter; D1.7 gains a term). Cost: ~3–5 days engineering
+ ~1.5 days certification.

### (iii) Full budget redesign — stop on distinct-DAG count

Everything in (ii), plus:

| File | Change |
|---|---|
| `experiments/models/bingo/{runner,isalsr_runner}.py` | A custom termination path. `evolve_until_convergence` has no distinct-count criterion; the only clean route is to raise a sentinel from inside `_serial_eval` when the counter crosses N and catch it around the `evolve_until_convergence` call — i.e. **an early stop, which §5.4 considered and rejected in 2026-07-27**. |
| `experiments/models/udfs/{runner,isalsr_runner}.py` | Same, inside the monkey-patched `evaluate_cgraph` |
| all 14 configs | `max_unique_dags: N` replaces/joins `max_time`; **N must be chosen**, and the choice is a free parameter a reviewer will ask about |
| `slurm/*_config.yaml`, every launcher | wall-clock requests can no longer be derived from `max_time`; runs become variable-length and a SLURM limit must still cap them, reintroducing a second stop rule |
| `experiments/models/analyzer/*`, `analyze.py`, all figure generators | wall clock moves from *controlled variable* to *outcome*; `S`, overhead %, Table 2's cost column and the CD diagrams all change interpretation |
| manuscript §computational_experiments, Table 1, Table 2, Table S, forest plot | rewritten |

**Metrics whose MEANING changes:** `wall_clock_total_s` (input → outcome); `S`
(a ratio at fixed budget → a ratio at fixed coverage); `overhead %` (share of a fixed
budget → share of a variable one); every quality metric (at fixed time → at fixed
coverage). Effectively **every headline number in the paper changes meaning.**

**Re-certification:** all of Stages A–E, plus a new pre-flight for the N choice, plus
the §5.4 rejection has to be formally reversed in the plan. **And a residual stop-rule
heterogeneity remains**: cells that cannot reach N inside the SLURM wall stop on time
instead, which is the same defect in a new currency.

---

## 6. Conclusions and recommendation

### What is true

1. All six completed Stage D Bingo cells stopped on **one** rule, `max_evals`, at
   100 M ± 0.07 % fitness invocations (§3). The arms are, on these cells, at an equal
   *evaluation* budget already.
2. Campaign-wide the binding rule is heterogeneous (C1: 52/264 isalsr cells hit
   `max_time` instead), so neither "equal time" nor "equal evaluations" describes C1.
   That, not `max_evals` per se, is the defect.
2b. **F-19**: three of seven production Bingo suites carry no `max_evals` and run at a
   **10× tighter** inherited default. The heterogeneity is therefore not only per-cell
   but per-*suite*, and the pooled CPDT mixes the two. This alone makes the config fix
   mandatory before C2 launches, independent of which protocol is chosen.
3. The baseline's candidate count is recoverable post hoc (501×G). Its **canonically
   distinct** count is not, and that single gap blocks the headline coverage claim,
   blocks any distinct-count budget, and blocks the ρ inferential test against the
   baseline.
4. Measuring it costs 10.5–39.0 % of the baseline arm's wall clock at full rate, and
   **cannot be subsampled** (Charikar et al., PODS 2000).
5. The two trace problems disagree on every budget axis, in the direction the
   2026-04-19 bottleneck-type analysis predicts. Pagie-1 (structural) favours IsalSR
   under equal time, equal candidates and equal unique-DAGs. Vladislavleva-2
   (structural_depth) favours the baseline under all three, and favours *naive hash*
   over IsalSR under the unique-DAG budget.
6. **F-18**: 73–81 % of IsalSR's wall-clock excess over the baseline at equal
   invocations is unattributed — it is neither canon, nor conversion, nor shadow. On
   Vladislavleva-2 it is large enough (e_I/e_B = 4.03 > ρ = 1.83) that IsalSR would lose
   on time even with a free canonicaliser.

### What I would do, and why

**Take option (ii): the minimal fix plus the baseline-distinct measurement, recorded but
not made a stop rule. Do not take option (iii) before the freeze.**

Reasons, in order of weight:

1. **Option (iii) buys nothing that (ii) does not.** Once D_base is recorded, every
   equal-distinct-coverage analysis Mario wants is computable *post hoc* by truncating
   trajectories — exactly as §2.6 does today for hash vs isalsr. Enforcing the budget
   online changes what every headline number means, reopens §5.4's rejected early-stop
   decision, makes run length unpredictable against SLURM walls, and reintroduces
   stop-rule heterogeneity through the back door. The scientific content of the
   three-way tension (unique DAGs / time / quality) is fully available from recorded
   counters and a truncation script.
2. **Option (i) alone is not enough and is mildly dangerous on its own.** It makes the
   protocol description true, but it hands a reviewer a Vladislavleva-2 row where the
   baseline generates 2.59× more candidates than IsalSR, with no way to answer "how many
   were distinct?". (i) without (ii) converts a hidden problem into a visible unanswerable
   one.
3. **The D_base measurement is the highest-leverage change available.** It simultaneously
   closes audit F-3, retires T17-HANDOFF decision 3, makes the headline coverage claim
   computable for the first time, and resolves the 1.70/4.74 break-even uncertainty
   that currently makes the whole comparison unfalsifiable. One measurement, four
   problems closed.
4. **F-18 must be resolved before either.** It is a bigger number than everything under
   discussion and it is the first thing a reviewer computes. `rss_timeseries.csv` is
   already on Picasso for all six cells; the H1 test is a plotting exercise, not a new
   run.

### Timeline judgement

Freeze **2026-09-10**; today **2026-08-05**; 36 days. C2 itself ≈**8.8 days** at 476
cores, and it cannot start until Stage D lands and a clean Stage C wave runs on the
launch commit (already owed).

| Path | Engineering | Certification | C2 | Slack to freeze |
|---|---:|---:|---:|---:|
| (i) only | ~0.5 d | Stage C 0.5 d + **Stage D 1.0 d** | 8.8 d | ≈25 d |
| **(ii)** | **3–5 d** | Stage A/B 1 d + C 0.5 d + **D 1.0 d** | 8.8 d | **≈19–21 d** |
| (iii) | 10–15 d | A–E ≥3 d + D 1.0 d | 8.8 d | ≈3–8 d, **and analysis/writing not yet begun** |

(ii) fits with ~3 weeks of slack for analysis, figures, the response letter and the
manuscript edits — which is roughly what T09–T14 need. (iii) consumes the slack
entirely and leaves no margin for a single failed wave; against a hard freeze on a TPAMI
revision that is not a defensible risk.

**Sequencing.** Stage D must be allowed to finish (7 cells running; the trace cell is
the D2/D3 input and cannot be re-run cheaply). Do the F-18 forensics *now*, from the RSS
series and the completed artefacts — it is read-only and blocks nothing. Land (i)+(ii)
as one commit after Stage D completes, then one Stage C wave, then Stage D again, then
C2.

### The plain statement, as requested

The proposal does not harm IsalSR's measured advantage in the way the brief feared —
under a distinct-count budget IsalSR gets strictly more real structural coverage, and
that part of my earlier analysis was wrong. But two things found here **do** cut against
the paper as currently framed, and neither depends on which budget is chosen:

- On Vladislavleva-2, at equal wall clock, equal candidates *and* equal unique DAGs,
  IsalSR is beaten by the baseline and by the naive-hash arm alike. One seed, one
  problem, and consistent with the bottleneck-type theory — but it will be in Table 2
  and it will be read.
- IsalSR's arm costs 1.5–4.6× the baseline's wall clock at equal evaluations, while the
  paper reports 11–15 % overhead. Both numbers are true; together they are indefensible
  without F-18 resolved and disclosed.

Reporting these now is worth more than having a reviewer derive them from Table 2.

---

## 7. Unverified / open

| Item | Status |
|---|---|
| Provenance of `audit.md`'s 3.3–4.1× LM figure | **UNVERIFIED** — refuted for the hard tier, but I did not find which run produced it |
| UDFS 12 h per-DAG costs and ρ | **UNVERIFIED** — all three UDFS Stage D cells still RUNNING; §4f's UDFS numbers are Stage C 900 s |
| Cause of F-18 (H1 memory / H2 structural / H3 co-tenancy) | **UNRESOLVED** — `rss_timeseries.csv` exists per cell and would test H1 |
| ρ_base on any problem | **UNMEASURED** — the whole point of §4c/§4d |
| ~~Whether `bingo_roundoff.yaml` carries `max_evals`~~ | **CLOSED — it does not.** Promoted to F-19 (§0): three suites inherit the 10 M default |
| Whether C1 results for the `roundoff` / `strogatz` / `feynman_remainder` suites were produced under the 10 M cap | **UNVERIFIED** — check before the pooled CPDT is re-quoted; if so, those problems' δ values are not comparable to the other suites' |
| Korns-12 (all 3 arms) and the D2 trace cell | in flight; conclusions here rest on 2 problems × 1 seed |
| Whether ρ_base > ρ_isalsr, and by how much | argued mechanically (dedup lowers the isalsr stream's redundancy); **not quantified** |

---

# 8. LOCKED DECISIONS — Mario, 2026-08-05

This section is the decision record. Sections 0–7 are the investigation that
produced it; where they disagree with this section, **this section wins**.

## 8.1 The question the campaign answers

> Given the **same number of DAGs to explore** for each host (UDFS, Bingo), and
> changing **only the deduplication approach** (none → naive-hash → IsalSR),
> which allocates that budget most efficiently — in ρ, in convergence speed, in
> test R²/NRMSE?

The treatment is the *allocation policy*, not the search algorithm. Everything
below follows from holding the budget fixed and letting the policy decide how it
is spent.

## 8.2 The decisions

| # | Decision | Status |
|---|---|---|
| **D1** | **The budget is a maximum number of DAGs *evaluated*, counted post-deduplication in each arm.** Baseline: every generated DAG counts (it evaluates all of them). Naive-hash: only hash-distinct DAGs count. IsalSR: only canonically distinct DAGs count. | **LOCKED** |
| **D2** | **`max_time` is not the budget.** | **LOCKED — rejected** |
| **D3** | **The baseline is *not* budgeted on its canonically-distinct count.** | **LOCKED — rejected** |
| **D4** | **`D_base` (the baseline's canonically-distinct count) is nevertheless *measured and reported*.** | **LOCKED — required for reporting, not for budgeting** |
| **D5** | **Conversion time stays inside `overhead_time_s`.** | **LOCKED — unchanged (audit F-7)** |

### D1 — why post-dedup DAGs evaluated is the right currency

The fitness evaluation is the expensive, method-neutral unit of work, and under
D1 **every arm performs the same number of them.** The baseline spends some of
its budget re-evaluating structures it has already seen; IsalSR spends every one
on a structure it has never seen. Converting the same expenditure into more
distinct coverage *is* the method's value proposition, so the design measures
exactly the claimed effect.

It also removes the two confounds §4 established:

- **Wall clock is not neutral.** §4 measured 27,829 / 9,309 / 7,078 evaluations
  per second (baseline / hash / isalsr, Vlad-2) — monotone in dedup strength.
  The baseline is fast *because* re-evaluating duplicates is cheap. Budgeting on
  time therefore pays the baseline for its own waste.
- **`eval_count` is not neutral either.** It is LM-inflated 13.8–16.2× and the
  inflation varies by problem and arm, so it is only an approximate proxy for
  individuals evaluated (see §8.3).

### D2 — why not `max_time`

More wall clock lets the no-overhead arms evaluate more candidates, and the more
candidates drawn, the more non-isomorphic structures are covered *by chance*.
Time budgets therefore dilute the treatment effect in the competitors' favour.
Independently, §3 established that `max_time = 43,200 s` **never binds** on
Bingo — the 12 h budget is fictional today.

### D3 — why the baseline is not budgeted on `D_base`

Running the baseline until it *covers* N canonically-distinct structures means
letting it generate ρ_base × N candidates, granting it proportionally more wall
clock and more chances to stumble on a solution. That converts the experiment
into **"time to cover N distinct structures"** — an interesting quantity (§8.4),
but a *time* comparison, and one that dilutes the performance question D1 is
built to answer. Budgeting on N-distinct-for-everyone equalises the wrong thing.

> This supersedes the orchestrator's earlier proposal (hypothesis (b), §4b) that
> a fair budget requires counting canonically-distinct DAGs on the baseline.
> That proposal is **withdrawn as a budgeting rule** and retained only as D4.

### D4 — measure `D_base` anyway (orchestrator's dissent, accepted)

D3 removes `D_base` from the *budget*. It does not remove it from the *report*.
Without it we cannot state:

1. how many of the baseline's evaluations were spent on isomorphic repeats —
   i.e. the size of the waste IsalSR eliminates, which is the headline claim;
2. the "unique DAGs evaluated vs time, coloured by isomorphic duplicates
   evaluated" figure (§8.4), whose baseline series is undefined without it;
3. ρ_base, currently **unmeasured** on every problem (§7).

Measurement is **counting, never deduplicating** — the canonicaliser runs for
accounting only and must not alter the baseline's search. Per §4d, **subsampling
is refused**: distinct-count estimation from a q-fraction sample carries
Ω(√(1/q)) error (Charikar et al., PODS 2000). Full-stream HyperLogLog is
required, at a measured 10.5–39 % wall-clock cost — affordable precisely because
D1 budgets on DAG count, so the cost inflates measured time (already separately
accounted) without stealing search.

## 8.3 The landed number that changes the plan

**The current `max_evals = 100M` configuration already implements D1, to within
≈3 %.** Individuals evaluated, derived as `501 × generations` (AgeFitnessEA;
identity verified against `total_dags_explored` to within 500 on every dedup
arm):

| problem | arm | individuals evaluated | of which distinct | LM evals/individual |
|---|---|---:|---:|---:|
| vladislavleva_2 | baseline | 7,254,480 | unmeasured (incl. duplicates) | 13.79 |
| vladislavleva_2 | hash | 7,209,039 | 7,209,039 hash-distinct | 13.88 |
| vladislavleva_2 | isalsr | **7,085,963** | **7,085,963 canonically distinct** | 14.12 |
| pagie_1 | baseline | 6,162,300 | unmeasured (incl. duplicates) | 16.24 |
| pagie_1 | hash | 6,434,660 | 6,434,660 hash-distinct | 15.55 |
| pagie_1 | isalsr | **6,262,371** | **6,262,371 canonically distinct** | 15.97 |

Because LM inflation per individual is near-constant *within a problem*, a fixed
`eval_count` budget is implicitly a fixed individuals-evaluated budget. **The
Stage D cells already ran the design D1 specifies.**

Consequence: D1 is a **tightening**, not a redesign. The budget must become
explicit and exact — counted on individuals passed to the fitness function,
post-dedup — rather than implicit through an LM-inflated proxy whose inflation
varies 13.79–16.24 across problems and arms. The existing Stage D data remains
interpretable under D1 with that ±3 % caveat stated.

## 8.4 Post-hoc derivability — "time to evaluate X distinct DAGs"

**Yes for the dedup arms, today, with no new runs.** `trajectory.csv` carries
`timestamp_s, iteration, best_r2, best_nrmse, n_dags_explored,
n_unique_canonical, cache_hit_rate_cumulative` per snapshot, and on the hash and
isalsr arms `n_unique_canonical` is a genuine measurement. Truncating each
trajectory at a common unique count yields time-to-N-distinct and quality-at-N-
distinct directly.

**No for the baseline**, where `n_unique_canonical = n_dags_explored` by
construction (`bingo/runner.py:428`). It yields time-to-N-*evaluated*, duplicates
included. The baseline series of the requested figure needs D4.

**Requested figure** — unique DAGs evaluated (x) vs wall-clock time (y), one
series per arm, coloured by isomorphic duplicates evaluated (zero for IsalSR by
construction; `total − D_base` for the baseline; `hash_distinct − D_hash` for
naive-hash). Two of three series are computable now; all three once D4 lands.

## 8.5 Changes to make

| # | Change | Where | Blocking? |
|---|---|---|---|
| **C-1** | **Fix F-19.** Add `max_evals` to `bingo_{roundoff,strogatz,feynman_remainder}.yaml`. They silently inherit `config.py:29`'s 10 M default — a **10× tighter budget on 28 of 70 problems**, undocumented, pooled into the N=70 CPDT | `experiments/configs/` | **YES — launch blocker** |
| **C-2** | Make the budget explicit: count individuals passed to the fitness function post-dedup, not LM-inflated `eval_count` | both hosts' runners | YES — measured *during* a run (§3 dividing line) |
| **C-3** | Implement D4: full-stream HLL canonical-distinct counting on the baseline arm, counting only, never deduplicating | `bingo/runner.py`, `udfs/runner.py` | YES — during-run |
| **C-4** | Record `n_individuals_evaluated` and `D_base` on `RunLog` + `TrajectoryRow`; certified field count 60 → 62 | `schemas.py` | YES — during-run |
| **C-5** | Decide and document whether `max_time` is retained as a *safety* wall only, given it never binds | configs + §5.4 | Before Stage F |
| **C-6** | Check whether C1's `roundoff`/`strogatz`/`feynman_remainder` results ran under the 10 M cap; if so their δ are not comparable to the other suites' | C1 archive | Before re-quoting the pooled CPDT |
| **C-7** | Re-certify: C-2/C-3/C-4 are during-run changes, so Stage C **and** Stage D must be re-run. Stage E is unaffected (it certifies the analysis pipeline, not the protocol) — but re-run it on the new root | pre-flight | Before `campaign/c2` |

## 8.6 What this does not change

- **`overhead_time_s = canon + conversion`** stands (D5, audit F-7).
- **The 11–15 % overhead figure is correct** — §4 confirmed `evaluation_time_s ==
  wall_clock_search_only_s` exactly on all six cells, so nothing escapes the
  books. The 1.7–4.6× wall-clock gap is *not* an accounting defect; it is dedup
  steering the search toward costlier, structurally novel individuals. Both must
  be reported, and the gap must not be presented as overhead.
- **Vladislavleva-2 still shows IsalSR losing** to both competitors on all three
  budget axes at n=1 seed. Nothing in D1–D5 changes that; it is a result, not an
  artefact, and it goes in Table 2.

## 8.7 Landed 2026-08-05 — F-19 closed, F-20 open, runtimes measured

**F-19 — FIXED.** `max_evals: 100000000` added to `bingo_roundoff.yaml`,
`bingo_strogatz.yaml`, `bingo_feynman_remainder.yaml`, with the rationale inline.
Verified through the **loader**, not the YAML text: `max_evals`, `max_time`,
`population_size`, `stack_size` are now single-valued across all seven suites.
Locked by `tests/unit/test_budget_uniformity.py` (17 tests) — each suite must
*declare* the key rather than inherit it, and the declared value must exceed the
default it replaces, so a future "fix" cannot silently write 10M back in.
Local edits only; nothing deployed (Stage D was running).
⚠ Carry: C1's results for those three suites were very likely produced under the
10M cap, so their δ are not comparable with the other four suites' — check
before any pooled CPDT is re-quoted (C-6).

**F-20 — OPEN, Mario's decision.** `udfs_feynman.yaml` sets `n_calc_nodes: 7`
against 5 in every other suite. `n_calc_nodes` bounds the size of the expressions
UDFS enumerates, so a per-suite value confounds the suite with the reachable
search space — the A4b defect, in UDFS. Deliberately unfixed: 7→5 shrinks UDFS's
reach on 10 Feynman problems and may lose recoveries; 5→7 inflates enumeration
cost portfolio-wide. The regression test is left **failing** as a live flag
rather than passing on a guessed direction.

**Runtimes — the two hosts are bound by different things.**

| host | n | mean | median | max | bound by |
|---|---:|---:|---:|---:|---|
| UDFS (C1) | 600 | **12.00 h** | 12.00 h | 12.00 h | `max_time` — saturates on **100 %** of runs; no `max_evals` exists |
| Bingo isalsr (C1) | 564 | 5.15 h | 4.04 h | 11.76 h | `max_evals` |
| Bingo (Stage D) | 8 | 4.57 h | — | 6.87 h | `max_evals` |

**Cost at measured runtimes**, superseding §8.1's 12-h-for-both assumption:

| seeds | runs | UDFS | Bingo | total core-h | @592 cores | @476 cores |
|---|---:|---:|---:|---:|---:|---:|
| 20 | 8,400 | 50,400 | 21,000 | **71,400** | 5.0 d | 6.3 d |
| 30 | 12,600 | 75,600 | 31,500 | **107,100** | 7.5 d | 9.4 d |

**30 seeds is affordable** against the 2026-09-10 freeze, reversing §0.4a's cost
premise. Recommended shape: **launch 1–20 first, then 21–30 as an additive
second wave** — purely additive under the resume logic, and if anything fails
you still hold a complete, analysable 20-seed campaign (§0.4a's own priority
ordering: "the first obligation is that results land").

**Two consequences that must travel with these numbers:**

1. Because UDFS is *time*-budgeted, **node speed changes what UDFS explores**.
   This is a third, independent reason for B6's `sr` pin, and it **forbids
   running different seed blocks on different node families** — the blocks would
   not be exchangeable, contaminating UDFS's R²/ρ/recovery, not merely its clock.
2. Any 8 h budget is refused twice: it would cut a third of UDFS's search on
   100 % of runs, and §11.1 (2026-08-04) already rejected 12 h→8 h on Bingo's
   69 % effect erosion.
