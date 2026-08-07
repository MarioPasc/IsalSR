# `slurm/c2_campaign/` — submitting Campaign C2

The final review experiments: **42 arrays, 12,600 runs, ≈80,000 core-hours**, submitted
as **3,474 SLURM tasks**.

> ### 🔴 One task is a CHUNK of cells, not a cell (2026-08-07)
>
> SCBI asked us to group short jobs so every submitted task runs for at least two
> hours — the scheduler spends longer placing a short job than the job spends
> running, and 12,600 placements saturate the queue for everyone. They were right:
> 51.5 % of the 23,058 job records this account produced since 2026-07-01 ran under
> two minutes, and ~43 % of the planned campaign tasks would have.
>
> Each array task now runs a contiguous block of cells **in sequence**, on
> `$LOCALSCRATCH`, under a deadline. Cells, memory, constraint, seeds and the
> paired design are **unchanged**; only the packaging is different.
>
> | | before | after |
> |---|---|---|
> | SLURM tasks | 12,600 | **3,474** |
> | shortest expected task | 0.1 h | **20.0 h** |
> | wall per task | 16 h | 25–47 h |
> | makespan | 48.0 h | **49.2 h** (+2.5 %) |
>
> Full rationale, measurements and the reply to SCBI: **`CAMPAIGN_BRIEF.md`**.

This directory is the campaign's front door. It deliberately contains **no
worker, no `#SBATCH` header and no topology of its own.**

---

## 1. Why there is no `worker.sh` here

The campaign and the Stage C smoke are **one launcher under two profiles**
(`EXECUTION-PLAN` §10.2). They differ only in seeds, payload budget, wall and
output root; topology, throttle apportionment, memory, log layout and the
aggregation shape are shared. That is not an accident of implementation — it is
the property that makes the pre-flight mean anything:

> "Certifying a topology you will not launch certifies nothing." (§1)

A self-contained copy of `launcher.sh` and `worker.sh` in this directory would
be **uncertified code at the moment 108,000 core-hours are committed**, because
the Stage C wave exercises `slurm/c2_smoke/`. So `launch.sh` below is a gate
plus a delegation, and every byte that actually runs on a compute node lives in
`slurm/c2_smoke/`, where the smoke wave proved it.

The `picasso-sbatch` skill states the same rule as its hard rule 14: *if the
project already has a working worker for this software, extend it rather than
writing a fresh one* — environment fixes accumulate in workers and are not
discoverable from the application code.

| File | Role |
|---|---|
| `stage_f_preflight.sh` | The go/no-go gate. Ten checks, fails closed, submits nothing |
| `launch.sh` | Runs the gate, then delegates to `c2_smoke/launcher.sh` with `C2_PROFILE=campaign` |
| *(worker, aggregation, slot plan)* | **`slurm/c2_smoke/` and `experiments/scripts/c2_slot_plan.py` — referenced, never copied** |

---

## 2. The sequence

**The full runbook, with the reasoning behind each step, is `CAMPAIGN_BRIEF.md` §7.**
The short form:

```bash
# 0. From the workstation, repo root, on the commit to be tagged.
#    If the tree carries another agent's uncommitted work, deploy from a clean
#    clone of the branch tip -- deploy.sh refuses a dirty tree and it is right
#    (defect 14).  Never commit someone else's files.
python -m pytest tests/unit -q
bash slurm/c2_smoke/mock_chunk_test.sh          # 26/26, real payload, ~10 min

# 1. Deploy.  The only path: Picasso has no outbound SSH (defect 13).
bash slurm/c2_smoke/deploy.sh          # rsync incl. .git, verify SP-1, rebuild, verify SP-2

# 2. 🔴 Prove the copy-back ON THE DEPLOYED TREE.  20 minutes.  Do not skip it:
#    it is the only step that can catch localscratch dropping results, and a
#    per-cell check cannot -- see CAMPAIGN_BRIEF.md §5.
ssh picasso 'cd $REPO && bash slurm/c2_smoke/chunk_smoke.sh'
ssh picasso 'cd $REPO && python slurm/c2_smoke/chunk_smoke_verify.py \
             $FSCRATCH/results/isalsr/c2_chunk_smoke'
#    expect: SMOKE OK -- localscratch loses nothing

# 3. The gate. Submits nothing; exits non-zero on any failure.
bash slurm/c2_campaign/stage_f_preflight.sh

# 4. Preview, then submit.  PACED -- see the box below.
bash slurm/c2_campaign/launch.sh --dry-run
ssh picasso 'cd $REPO && bash slurm/c2_campaign/submit_paced.sh --dry-run'
ssh picasso 'cd $REPO && bash slurm/c2_campaign/submit_paced.sh'
```

> ### 🔴 Submit with `submit_paced.sh`, not `launch.sh`
>
> `launch.sh` submits all 42 arrays in a tight loop. On 2026-08-06 that got
> `Slurm temporarily unable to accept job / Resource temporarily unavailable`
> after **29 of 42**, aborting *before* it wrote `job_ids.txt` or submitted the
> aggregation — leaving 29 untracked arrays on the cluster. Re-submitting the
> remaining 13 with a 20 s gap succeeded with **zero** refusals, so the binding
> constraint is slurmctld RPC **rate**, not `MaxJobCount`.
>
> `submit_paced.sh` paces at 20 s (`C2_SLEEP` to raise it) and is **idempotent**:
> it skips arrays that already exist, so re-running after an abort completes the
> set instead of duplicating it. It submits the aggregation only once all 42
> exist. `launch.sh` remains the gate-plus-delegation entry point and is correct
> for a smoke-sized wave.
>
> **Your first act after any submission error is `squeue`** — assume jobs exist
> until proven otherwise.

`launch.sh` refuses to submit unless `stage_f_preflight.sh` passes, so step 1 is
a preview of the gate rather than a separate obligation. `submit_paced.sh` does
**not** run the gate: run it yourself, or use it only to finish a partial
submission the gate already cleared.

### 2a. Verify the payload before trusting the submission

`sbatch --test-only` validates a **resource request** and never runs the worker.
On 2026-08-06 it reported 42/42 accepted while every task was about to abort on
its seed spec. Gate **G12** now decodes the campaign seed spec through the
worker's own arithmetic, but the strongest check is cheap and takes four
minutes:

```bash
# 42 real campaign-profile tasks, short payload, throwaway root.
# C2_MAX_BUNDLE=1 is REQUIRED since chunking: --one-task submits array 1-1 per
# array, but a task now runs its whole BUNDLE, and bingo:*:nguyen's bundle is
# 164 cells -- at 240 s each that is an 11-hour "four-minute" probe.
C2_PROFILE=campaign C2_MAX_TIME=240 C2_MAX_BUNDLE=1 \
C2_RESULTS_DIR=$FSCRATCH/results/isalsr/c2_probe \
C2_LOGS_DIR=$FSCRATCH/execs/isalsr/c2_probe/logs \
  bash slurm/c2_smoke/launcher.sh --one-task
# expect: 42/42 COMPLETED, 42 run_log.json, three arms x 14
```

This probe exercises the decode and the payload. It does **not** exercise the
multi-cell chunk loop or the deadline, because `C2_MAX_BUNDLE=1` reduces every
task to one cell. (Localscratch staging still runs — it is governed by
`C2_USE_LOCALSCRATCH`, not by the bundle.) `chunk_smoke.sh` in step 2 is what
covers the chunk loop, the deadline and the copy-back together.

---

## 3. What the campaign requests

Derived by `experiments/scripts/c2_slot_plan.py`, unit-tested in
`tests/unit/test_c2_slot_plan.py`. Regenerate with
`python -m experiments.scripts.c2_slot_plan --seeds 1-30 --table`.

| | value | why |
|---|---|---|
| Arrays | 42 = 7 suites × 3 arms × 2 methods | one config declares one suite |
| Cells | 12,600 = 70 problems × 30 seeds × 6 | §0.4a as superseded 2026-08-05 |
| **SLURM tasks** | **3,474** | SCBI's 2 h floor; chunk sized per array from the C1-measured per-suite cell duration |
| Bundle | 2–164 cells per task | `bingo:*:nguyen` is 0.15 h per cell and needs 164 to clear 2 h; `udfs:*:hard` is 12 h and needs 2 |
| Throttle | apportioned, 2,016 slots | work-proportional; **1.9× makespan gain** over uniform `%K`, free. Capped at the *post-chunking* task count, which is what keeps chunking makespan-neutral |
| `--mem` | 32 G Bingo, 16 G UDFS | unchanged: cells run as separate processes, so peak RSS is still per-cell. 27× observed peak, 3.4× the `max_evals`-bounded 9.4 GB ceiling |
| `--time` | 25–47 h **per task** | `B × (12 h payload + teardown)` where that fits; otherwise `1.4 × expected chunk + one cell in reserve`. Longest is 47 h, a day inside `medium_uma`'s 72 h |
| Deadline | `wall − 12.5 h` | a cell only STARTS if its full budget still fits, so a `TIMEOUT` is impossible by construction. The **first** cell is exempt — without that the array livelocks |
| `--constraint` | `sr` | C4 needs one CPU family (data is not bit-reproducible across families, ~1 ULP); wall clock is a *reported* quantity |
| Output | `$LOCALSCRATCH`, copied back per cell | SCBI request + manual §4.9. `sr` nodes carry 800 GB; the campaign is megabytes |
| Logs | FSCRATCH, **merged** | 3,474 files not 25,200 |
| Makespan | ≈49 h | +2.5 % against the unchunked plan's 48.0 h |

**The wall is per task, not per campaign.** SLURM has no campaign-duration
parameter, and raising `--time` to the makespan would let a hung task burn 49 h
of an allocation before the ledger caught it. The deadline is what bounds a
chunked task from the inside.

---

## 4. The Day-1 rebalance

The plan is weighted with `T_bingo = 8 h` on purpose — F-19 raised three suites'
`max_evals` tenfold and the 20 D2 problems have no runtime data at all. Scored
at the measured 5.15 h that pessimism costs 9 h of makespan, and it is
**recoverable in flight**:

```bash
python -m experiments.scripts.c2_slot_plan --bingo-hours <measured> \
       --rebalance <logs>/job_ids.txt        # emits 42 scontrol lines
```

`scontrol update JobId=<id> ArrayTaskThrottle=<n>` re-apportions a **running**
array. It touches no config and no file in the deployed tree, so it is **not**
defect 10 and is safe mid-campaign. Read the first day's Bingo wall clocks, then
rebalance.

---

## 5. Rollback

```bash
scancel <each id in job_ids.txt>            # or: scancel --name=c2s_*
```

The orchestrator's resume logic makes a re-launch additive: a completed
`(method, arm, problem, seed)` is skipped, so re-running the same arrays costs
only the missing cells. **Do not delete the campaign root to "start clean"** —
the status ledger is what makes cell completeness provable (§5.5), and a
partially completed triple is worse than a missing one because it silently
unbalances the paired test.

---

## 6. Standing rules this directory does not override

- **SP-0** — C2 is submitted once, by Mario, after Stage F sign-off.
- **Defect 10** — a deploy IS a config edit. Never deploy while an array runs.
- **Defect 13** — `git pull` cannot work on Picasso (no outbound SSH). `deploy.sh`
  is the only path.
- **Defect 15** — there is no `sbatch` on the workstation. `--test-only` runs on
  the cluster.
- **§5.5** — if capacity forces truncation, drop whole `(method, problem, seed)`
  triples across all three arms, never individual runs.
