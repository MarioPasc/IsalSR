# `slurm/c2_campaign/` — submitting Campaign C2

The final review experiments: **42 arrays, 12,600 runs, ≈108,000 core-hours.**

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

```bash
# 0. From the workstation, repo root, on the commit to be tagged.
bash slurm/c2_smoke/deploy.sh          # rsync incl. .git, verify SP-1, rebuild, verify SP-2

# 1. The gate. Submits nothing; exits non-zero on any failure.
bash slurm/c2_campaign/stage_f_preflight.sh

# 2. Dry run, then the cluster-side accept test, then the campaign.
bash slurm/c2_campaign/launch.sh --dry-run
bash slurm/c2_campaign/launch.sh --test-only     # runs ON Picasso (defect 15)
bash slurm/c2_campaign/launch.sh                 # 42 arrays, 12,600 tasks
```

`launch.sh` refuses to submit unless `stage_f_preflight.sh` passes, so step 1 is
a preview of the gate rather than a separate obligation.

---

## 3. What the campaign requests

Derived by `experiments/scripts/c2_slot_plan.py`, unit-tested in
`tests/unit/test_c2_slot_plan.py`. Regenerate with
`python -m experiments.scripts.c2_slot_plan --seeds 1-30 --table`.

| | value | why |
|---|---|---|
| Arrays | 42 = 7 suites × 3 arms × 2 methods | one config declares one suite |
| Tasks | 12,600 = 70 problems × 30 seeds × 6 | §0.4a as superseded 2026-08-05 |
| Throttle | apportioned, 2,016 slots | work-proportional; **1.9× makespan gain** over uniform `%K`, free |
| `--mem` | 32 G Bingo, 16 G UDFS | 4× D1.2's recommendation, 27× observed peak, 3.4× the `max_evals`-bounded 9.4 GB ceiling |
| `--time` | 16 h **per task** | UDFS saturates 12.00 h + SymPy tail; Bingo ≤ 11.76 h observed |
| `--constraint` | `sr` | C4 needs one CPU family (data is not bit-reproducible across families, ~1 ULP); wall clock is a *reported* quantity |
| Logs | FSCRATCH, **merged** | 12,600 files not 25,200 — load-bearing for the inode budget |
| Makespan | ≈63 h planned, ≈54 h after a Day-1 rebalance | see §4 |

**16 h is per task, not per campaign.** SLURM has no campaign-duration
parameter, and raising `--time` to the makespan would let a hung task burn 54 h
of an allocation instead of 16 before the ledger caught it.

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
