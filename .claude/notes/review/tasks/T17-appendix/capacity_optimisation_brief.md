# Briefing pack — C2 resource and schedule optimisation

**Written**: 2026-08-05, for a fresh conversation whose job is to optimise the
allocation, throttle, wave structure and node pinning of campaign C2.
**Status of C2**: NOT submitted. Stage D running. SP-0 holds — **nobody except
Mario submits C2**, and the optimising agent submits **nothing** but probes.

---

## 0. The one-paragraph problem

C2 is `{baseline, hash, isalsr} × {UDFS, Bingo} × 70 problems × N seeds`, as 42
SLURM arrays (one per `(method, arm, suite)`), 1 core per task, on Picasso. At
20 seeds that is 8,400 runs; at 30 seeds, 12,600. The question is how to run it
fastest without changing a single number it produces.

---

## 1. Read these, in this order

| # | Path | Read for |
|---|---|---|
| 1 | `.claude/notes/review/tasks/EXECUTION-PLAN.md` **§11.1** | The decision/anomaly ledger. Denser and more current than anything else. Read it *first* and read it *all* |
| 2 | same, **§1** | Array topology: 42 arrays, why not 6, per-array task counts |
| 3 | same, **§3.3** | Memory sizing: Bingo-IsalSR 256 GB and the node arithmetic it forces |
| 4 | same, **§4.0 (SP-0…SP-7)** | Standing Picasso discipline. **Binding on you** |
| 5 | same, **§4.3** | Stage C, the 1,260-task smoke — the harness any optimisation is measured with |
| 6 | same, **§5.1–5.5** | Protocol invariants. §5.4 (no early stopping) and §5.5 (completeness) constrain what you may trade |
| 7 | same, **§8** | Budget and the concurrency arithmetic, including the §8.3 trade order |
| 8 | same, **§11.3** | Launch ledger: the exact 42-array shape |
| 9 | `.claude/notes/review/tasks/T17-HANDOFF.md` | **Defects 1–16, do-not-reintroduce.** Several are pure ops traps that will cost you a day each |
| 10 | `.claude/notes/review/tasks/T17-appendix/unique_dag_budget.md` **§8** | The locked budget decisions (D1–D5) and F-19/F-20. Determines run *duration*, so it determines capacity |
| 11 | `.claude/notes/review/tasks/audit.md` **§7.3** | Shadow sketches: why they are off, and their 17.6 % cost |
| 12 | `docs/md_files/changes/stage_d_design.md` | Stage D harness, RSS sampling, memory recommendation method |
| 13 | `docs/md_files/changes/stage_e_design.md` | Stage E; also the analysis-side cost figures (analyzer ≈5.5 min at 8,400 runs vs aggregation ≈11 h) |
| 14 | `slurm/c2_smoke/{launcher.sh,worker.sh,deploy.sh}` | The launcher actually used. `C2_THROTTLE`, `C2_SEEDS`, `C2_LOGS_DIR` live here |
| 15 | `slurm/c2_stage_d/RUNBOOK.md` | The submission procedure that worked |
| 16 | `slurm/c2_tag_procedure.md` | The tag procedure (not yet executed) |
| 17 | `.claude/CLAUDE.md` + `CLAUDE.md` | Project invariants, Bingo operational requirements |
| 18 | skill **`picasso-sbatch`** | The authority on SLURM directives for this cluster |

---

## 2. Measured facts you must not re-derive (and must not contradict)

| Fact | Value | Source |
|---|---|---|
| Achieved concurrency, `%8` throttle, `cpu` pool | 245 cores | §11.1 2026-08-03 |
| Achieved concurrency, `%24` throttle, `cpu` pool | **476 cores**, peak 909 tasks | §11.1 2026-08-04 |
| Achieved concurrency, `%24`, pinned `--constraint=sr` | **592 cores** — pinning made it *faster* | §11.1 2026-08-04 |
| QOS `long_uma` entitlement | `MaxWall = 7 days`, **`cpu = 9000` per user**, `MaxJobsPU` unset | §11.1 2026-07-31 |
| `MaxArraySize` | 4096 — no chunking needed | A12 |
| `sr` pool | 154 nodes × 128 cores = **19,712 cores**, 439–450 GB/node | B6 |
| Other families | `sd` 52 c/182 GB (Intel, avx512, 2.1 GHz), `bc` 256 c/683 GB, `bl` 128 c/1855 GB | B6 |
| **UDFS run duration** | **12.00 h mean = median = max, n=600.** Saturates `max_time` on 100 % of runs | C1 archive, 2026-08-05 |
| **Bingo run duration** | mean 5.15 h, median 4.04 h, max 11.76 h (n=564); Stage D mean 4.57 h | C1 + Stage D |
| C2 cost at measured runtimes | **≈71,400 core-h at 20 seeds; ≈107,100 at 30** | 2026-08-05 |
| Aggregation job (`--postprocess only`) | 1 h 35 m over 1,260 runs ⇒ **≈11 h at 8,400**; needs ≥24 h wall, split per method | §11.1 2026-08-03 |
| Stage C wave duration | 1,260 tasks in **31 m 55 s** on `sr` at `%24` | §11.1 2026-08-04 |

---

## 3. The hard constraints — optimise inside these, never through them

1. **SP-0.** You submit **nothing** but probes: ≤1,800 s, ≤60 tasks, seed 0 only,
   output under `~/execs/isalsr/<purpose>/`, never a campaign root.
2. **Pin `--constraint=sr`** (B6, closed). Two independent reasons, and a third
   found 2026-08-05:
   - data generation is not bit-reproducible across families (~1 ULP), which
     split 35/210 cells in the unpinned v2 wave and **fails C4**;
   - wall clock is a *reported* quantity; `sd` 2.1 GHz vs `sr` 2.6 GHz makes a
     mixed pool partly a measurement of the scheduler;
   - **UDFS is time-budgeted**, so a faster node explores *more* in its 12 h.
     Node speed therefore changes UDFS's **science**, not just its clock.
     **This forbids running different seed blocks on different families.**
3. **One commit, one configuration** (§5.1). A deploy IS a config edit; never
   deploy or edit a config while any array is running (defect 10, §11.1
   2026-08-03/04).
4. **Never drop individual runs.** If capacity forces truncation, drop whole
   `(method, problem, seed)` triples across all three arms (§5.5).
5. **No early stopping, no budget cut.** §5.4; and 12 h→8 h was rejected on
   evidence (§11.1 2026-08-04, Bingo effect erodes 69 %).
6. **`--postprocess skip` on array tasks**, one dependent `afterany` job with
   `--postprocess only`. Per-task post-processing is a GPFS hammer and a race
   (§11.1 2026-08-03).

---

## 4. Where the headroom actually is

Ranked by expected gain per unit of risk. **(1) is nearly free and is the big one.**

| # | Lever | Evidence | Notes |
|---|---|---|---|
| 1 | **Raise the throttle.** `%8 → %24` gave 245 → 476 cores (1.94×) on the same wave | §11.1 2026-08-04 | Peak was 909 tasks against a 1,008 ceiling — ~90 % fill, so the cap was still binding. `%48` or `%64` is untested and the entitlement is **9,000 cores**, ~15× what `%24` achieved |
| 2 | **Parallel seed-block waves** | untested | See §5 |
| 3 | **Right-size `--mem`.** Stage C measured peak `MaxRSS` 0.67 GB against 8/16 GB requested | §11.1 2026-08-04 | Over-requesting throttles packing directly. **But D1.2 at 12 h is what sizes production**, not the 900 s figure, and Bingo-IsalSR's 256 GB (§3.3) is node-bound: `sr` hosts **1 per node** |
| 4 | **Split the aggregation per method** | §11.1 2026-08-03 | ≈11 h at 8,400 runs; ≥24 h wall or split |
| 5 | **Order the arrays so UDFS starts first** | 2026-08-05 | UDFS is always 12 h; Bingo averages 5 h. Starting the long pole first shortens the makespan |

**The Bingo-IsalSR 256 GB request is the real capacity ceiling**, not the core
count: `sd` hosts 0, `sr` 1, `bc` 2, `bl` 7 per node. Re-deriving it from D1.2's
measurement (`max(sacct MaxRSS, VmHWM) + 30 %`) is probably the single largest
structural win available, and Stage D was built to produce exactly that number.

---

## 5. The three-parallel-wave question — answer and constraint

**Splitting seeds into blocks (1–10, 11–20, 21–30) and running the blocks as
parallel waves is sound**, provided each block contains **all arms and both
methods** for its seeds. C4 (cross-arm data identity) is checked *within* a
`(problem, seed)`, so a whole-block wave keeps every arm of a given seed on the
same hardware and C4 still passes.

**They must all pin the same family.** Running distinct blocks on distinct
families is refused: UDFS is time-budgeted, so a faster family explores more
within the 12 h, which makes seed blocks **non-exchangeable** and contaminates
UDFS's R², ρ and recovery rate — not merely its timing. (For Bingo, where
`max_evals` binds, node speed changes only the clock; UDFS is the killer.)

Capacity is not the obstacle: three concurrent waves at `%24` ≈ 1,776 cores,
against a 9,000-core entitlement and an `sr` pool of 19,712 cores. The binding
question is whether the scheduler grants it, which is measurable.

**If hardware-effect quantification is wanted**, run the campaign on `sr` and
add a small dedicated cross-family probe as a covariate study. Do not obtain it
by scattering the campaign.

---

## 6. Probes to run on Picasso (all SP-0 compliant)

Each answers one question with a parsed artefact. **"I checked it" is not
evidence.**

| # | Probe | Command shape | Pass criterion |
|---|---|---|---|
| **P-1** | **Throttle sweep — the highest-value probe.** Re-run the Stage C 1,260-task wave at `%48` and `%96`, changing nothing else | `C2_THROTTLE=48 bash slurm/c2_smoke/launcher.sh` | Achieved concurrency = core-hours ÷ span. Compare against 476 (`%24`, `cpu`) and 592 (`%24`, `sr`). Report the peak concurrent task count too — if it sits near the ceiling, the throttle is still binding |
| **P-2** | **`sbatch --test-only` on all 42 arrays** with the real `--array`, `--mem`, `--time`, `--constraint` | launcher `--test-only` | exit 0 on 42/42, task counts exactly `suite_size × n_seeds`. **Run this on the cluster — there is no `sbatch` on the workstation** (defect 15) |
| **P-3** | **Parallel-wave feasibility.** Submit three seed-block waves of a *reduced* suite simultaneously, all `--constraint=sr` | 3 × ≤60-task probes | All three progress concurrently; measure aggregate achieved concurrency vs a single wave. Confirms the scheduler does not serialise them |
| **P-4** | **Memory re-sizing from D1.2.** Read Stage D's per-`(method, arm)` recommendation | `stage_d_certification.json` | `max(sacct MaxRSS, VmHWM) + 30 %`. **`sacct` memory: NEVER `-X`** (blank MaxRSS) — filter the `.batch` step and emit **`JobIDRaw`**, not `JobID` (§11.1 2026-08-04) |
| **P-5** | **Quota, live on the day.** FSCRATCH **inodes**, not just bytes | `ssh picasso 'quota'` | ≥60,000 inode headroom before launch; C2 writes ≈45k (≈67k at 30 seeds). HOME is separately over quota — Mario's lane |
| **P-6** | **Node-pool census under load** | `sinfo -o '%n %c %m %f %T'`, `sacctmgr show qos` | Confirm `sr` capacity and the live `cpu=9000` entitlement on the day |
| **P-7** | **Placement verification** on any probe | `sacct -j <id> -o NodeList` | 100 % of tasks on `sr`. A single stray node breaks C4 |
| **P-8** | **Aggregation cost** at scale | time `--postprocess only` on a large root | Extrapolate to 8,400/12,600 runs; size the wall at ≥24 h or split per method |

**Reporting.** Every Picasso work-log entry carries the fixed **SP-1…SP-6**
six-row table (provenance, install freshness, engine + negative control,
alphabet, both hosts, live counters). An entry without it is not evidence.

---

## 7. Traps that have each cost this project a day

From `T17-HANDOFF.md` §4 — read the full list, these are the ops-relevant ones:

- `sbatch --export` is comma-separated: **a value containing a comma is
  truncated**. `C2_SEEDS=0,101,102` delivered `C2_SEEDS=0` and killed 265 of
  1,260 tasks while the rest looked fine. Ship colon-separated, translate in the
  worker, and **assert** the decode.
- **`GROUPS` is a reserved bash array.** Assigning to it returns an error status;
  under `set -e` the launcher dies before its first `echo`, exiting 1 with no
  output.
- **`~/.conda` does not exist on Picasso** — the env is under `fscratch/conda_envs`.
- **`git pull` cannot work on Picasso** — no outbound SSH to GitHub. `deploy.sh`
  (rsync **including `.git`**) is the only path.
- **`deploy.sh` refuses a dirty tree, and it is right.** If the tree carries
  another agent's work, deploy from a temporary clean clone of the branch tip.
- **Piping `module load` runs it in a subshell** and the `PATH` change is lost —
  the build then silently uses the system g++ 7.5.0, which cannot compile
  `-march=x86-64-v3`.
- Never read pip's status through a pipe (`pip … | tail` reports `tail`'s code).

---

## 8. What is NOT open for optimisation

Cutting any of these buys time by spending science, and each was already decided:

- the 12 h budget (§5.4; 8 h rejected on measured effect erosion);
- the hash arm (it is what R1.4 asks for);
- D2 coverage (R3.1);
- per-run instrumentation — the T06 ledger costs 0.04 %/0.22 % and stays;
- the `sr` pin (§3 above, three independent reasons).

If the schedule still does not close after §4's levers, the trade order is
§8.3's, and **it is Mario's decision, not the optimiser's.**
