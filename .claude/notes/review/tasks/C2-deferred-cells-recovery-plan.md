# C2 — deferred cells: evidence, mechanism, and a recovery plan

**Status: DRAFT for review. Nothing in here has been executed.**
Written 2026-08-09 13:30 CEST, while campaign C2 is still running (83 % complete).

| | |
|---|---|
| Campaign | C2, commit `2dd56fd`, tag `campaign/c2`, submitted 2026-08-07 17:52 CEST |
| Campaign root | `$FSCRATCH/results/isalsr/c2_3arm` (Picasso) |
| Logs | `$FSCRATCH/execs/isalsr/c2_3arm/logs` |
| Problem | ~512 cells deferred and climbing; projected **≈1,100–1,200** at end of the main pass (≈9 % of 12,600) |
| Secondary problem | 1 cell lost to OOM: `bingo/cherrypicked/vlad_7/isalsr/seed_25` |
| Core finding | The deferral is **by design and benign per task**, but the sweep pass as configured **cannot clear it in one iteration** |

---

## 1. What "deferred" means, and why it is not an error

Since the SCBI chunking change (2026-08-07) one SLURM array task runs a **bundle**
of cells in sequence, under an internal deadline. `worker.sh` starts a new cell
only if the cell's *full* payload budget still fits inside the task's remaining
wall clock:

- `slurm/c2_smoke/worker.sh:416` — the deadline
- `slurm/c2_smoke/worker.sh:441` — the first cell is always exempt (without it the array livelocks)
- `slurm/c2_smoke/worker.sh:451` — unstarted cells are counted as `N_DEFERRED`
- `slurm/c2_smoke/worker.sh:494` — the `Cells: X ok, Y failed, Z deferred (of B)` summary line
- `slurm/c2_smoke/worker.sh:496` — the `DEFERRED:` list

This is what makes a `TIMEOUT` impossible by construction, and it is correct.
**A task that defers exits `COMPLETED`.** That is the crux: nothing in `sacct`
reveals a deferred cell, so any monitoring keyed on SLURM state reports a
perfectly healthy campaign while ~9 % of it is unwritten.

The deferred cells are meant to be picked up by the **sweep arrays**
(`c2w_*`), which are `afterany` on the main arrays.

> ⚠ `submit_paced.sh` does **not** submit the sweeps — `launcher.sh:557-592` does.
> For this campaign they were submitted as a separate paced pass
> (`slurm/c2_campaign/submit_sweeps.sh`, 42 arrays, ids in
> `<logs>/sweep_job_ids.txt`), and the aggregation was repointed onto all 84
> arrays. See EXECUTION-PLAN §11.1, 2026-08-07.

---

## 2. Evidence

All commands are read-only and safe to run while the campaign is live.

### 2.1 Count the deferred cells

```bash
ssh picasso 'L=$FSCRATCH/execs/isalsr/c2_3arm/logs
grep -h "^Cells: *[0-9]* ok," $L/*.out \
  | sed -n "s/.*, \([0-9]*\) deferred.*/\1/p" | awk "{s+=\$1} END {print s}"'
```

Reading at 83 % complete: **512**, still climbing as tasks finish.

### 2.2 Deferral is concentrated, not uniform

```bash
ssh picasso 'L=$FSCRATCH/execs/isalsr/c2_3arm/logs
for f in $L/c2s_*.out; do
  d=$(sed -n "s/^Cells: *[0-9]* ok, *[0-9]* failed, *\([0-9]*\) deferred.*/\1/p" "$f")
  [ -n "$d" ] && [ "$d" -gt 0 ] && echo "$(basename $f | sed "s/_[0-9]*_[0-9]*\.out//") $d"
done | awk "{s[\$1]+=\$2; n[\$1]++} END {for (k in s) print k, s[k], n[k]}" | sort -k2 -rn'
```

Reading at 83 %:

| Array | Bundle | Deferred | over N finished tasks |
|---|---:|---:|---:|
| `c2s_uh_feynman` | 27 | 127 | 6 |
| `c2s_ui_feynman` | 27 | 119 | 6 |
| `c2s_ub_feynman` | 27 | 105 | 5 |
| `c2s_bb_feynman` | 123 | 50 | 1 |
| `c2s_bb_nguyen` | 164 | 40 | 1 |
| `c2s_uh_feynman_remainder` | 4 | 28 | 28 |
| `c2s_ub_feynman_remainder` | 4 | 23 | 23 |
| `c2s_ui_feynman_remainder` | 4 | 20 | 20 |

**Every affected array is a large-bundle array.** The `B=2` arrays
(`strogatz`, `hard`, `cherrypicked`) defer nothing: 2 cells × 12 h = 24 h fits
inside their 25 h wall.

Only 17 of 36 `udfs:*:feynman` tasks had finished at this reading, so that block
roughly doubles. `feynman_remainder` defers exactly 1 cell per task, over
45 tasks/array × 3 arms = 135.

### 2.3 One real cell failure (separate issue)

```bash
ssh picasso 'grep -h "^FAILED:" $FSCRATCH/execs/isalsr/c2_3arm/logs/*.out | sort -u'
# FAILED:   Vlad-7/seed=25(rc=137)
```

`1840152_59` (`c2s_bi_cherrypicked`), OOM-killed after 11 h 23 m. **`MaxRSS` on
the `.batch` step was 1,084,696 K ≈ 1.03 GB against a 32 GB request** — matching
Stage D's measured 1.05–1.16 GB peak — so this is a fast allocation spike between
accounting samples, *not* the gradual `canonical_seen` growth that §3.3 sized
32 GB for. The chunk loop contained it: `2 ok, 1 failed, 0 deferred`, and
`Copy-back: 14 file(s) verified`.

---

## 3. Root cause

`experiments/scripts/c2_slot_plan.py` sizes each array's bundle and wall from a
**single point estimate per (method, suite)**:

- `c2_slot_plan.py:190` — the `CELL_HOURS` table
- `c2_slot_plan.py:246` — `MIN_TASK_HOURS = 2.0` (SCBI's floor)
- `c2_slot_plan.py:254` — `DEADLINE_SAFETY = 1.4`, calibrated so the *simulated*
  first-pass spill stays small

Measured against the realised campaign (n = 10,488 cells; script:
`scratchpad/measure_cell_hours.py`, adapt the root path):

| method:suite | n | median h | **p90 h** | assumed | median/assumed |
|---|---:|---:|---:|---:|---:|
| udfs:feynman | 461 | 0.18 | **12.00** | 0.90 | 0.2× |
| udfs:feynman_remainder | 269 | 12.00 | 12.00 | 6.00 | **2.0×** |
| udfs:nguyen | 956 | 12.00 | 12.00 | 11.50 | 1.0× |
| udfs:hard / cherrypicked / roundoff | ~2,000 | 12.00 | 12.00 | 12.00 | 1.0× |
| udfs:strogatz | 1,069 | 12.00 | 12.00 | 11.00 | 1.1× |
| bingo:feynman | 739 | 0.00 | 0.05 | 0.20 | 0.0× |
| bingo:nguyen | 918 | 0.03 | 0.43 | 0.15 | 0.2× |
| bingo:cherrypicked | 874 | 3.68 | 6.45 | 7.50 | 0.5× |
| bingo:hard | 822 | 3.15 | 7.08 | 5.30 | 0.6× |
| bingo:roundoff | 683 | 2.02 | 5.55 | 5.00 | 0.4× |
| bingo:feynman_remainder | 430 | 0.08 | 4.16 | 2.00 | 0.0× |
| bingo:strogatz | 1,260 | 0.09 | 1.01 | 2.00 | 0.0× |

> 🔴 **Correction to an earlier verbal claim.** I initially said `udfs:feynman`
> "saturates 12 h per cell and the planner assumed 1.2 h, so the estimate was 10×
> wrong". That is **not** what the data shows. The **median** is 0.18 h — the
> assumption of 0.9 h is if anything *generous* at the median. The problem is the
> **tail**: p90 is 12.00 h, a ~67× spread inside one suite.

So the real root cause is:

1. **The bundle is sized on a central estimate of a bimodal distribution.**
   `udfs:feynman` is a mixture of "solved in minutes" and "saturates the 12 h
   budget". `B=27` is right for the median and catastrophic whenever ~3 cells in
   a bundle land in the tail: 3 × 12 h = 36 h > the 34.5 h start-deadline, and
   the remaining ~22 cells defer. The `c2_slot_plan.py` comment above
   `DEADLINE_SAFETY` already notes the distributions are "heavily right-skewed";
   a 1.4× multiplier on a point estimate does not cover a 67× spread.
2. **One genuine point-estimate error**: `udfs:feynman_remainder` assumed 6.0 h,
   realised median 12.00 h. `B=4` then needs 48 h against a wall that fits 3
   cells, so it defers exactly 1 per task — which matches the observation.
3. `bingo:*` assumptions are all *over*-estimates, so those bundles are
   conservative; `bb_feynman`/`bb_nguyen` deferrals come from the same tail
   effect at very large bundles (B=123/164), not from an under-estimate.

**Memory is not the cause of any deferral.** The only memory event in the entire
campaign is the single Vlad-7 OOM, and even that showed 1.03 GB steady state.

---

## 4. Why the existing sweep pass will not finish the job

The sweep arrays were submitted with **the same bundle, wall and deadline** as
the main pass — deliberately, so the sweep runs an identical protocol. Resume
makes them additive, but each pass only advances a task by however many cells fit
inside its deadline.

For `udfs:*:feynman` (25 cells/task, 47 h wall, 34.5 h start-deadline, tail cells
at 12 h): a task starts cells at t≈0, 12, 24 h and stops → **≈3 cells per pass**.

```
pass 1 (main)   :  3 done, 22 deferred
pass 2 (sweep)  :  6 done, 19 deferred
...
pass 9          : 25 done,  0 deferred
```

**≈9 passes × ~36 h ≈ 12 days** to clear `udfs:*:feynman` by repetition. That is
inside the 2026-09-10 freeze but wasteful, and it requires 8 manual re-launches.

The `B=4` arrays (`feynman_remainder`, 1 deferred each) clear in one extra pass.
The `B=2` arrays never deferred and their sweeps will exit in seconds.

---

## 5. Plan

### 5.0 Do nothing until the main pass and the existing sweeps have drained

Resume is additive and safe, but launching a second topology over a live root
while the sweeps are pending risks two tasks running the same cell concurrently
and racing on the per-cell copy-back. **Wait for `c2w_*`, then act.**

Watch with:

```bash
bash slurm/c2_campaign/health.sh     # cell_fail / deferred / sweep_bad are the fields
```

### 5.1 Confirm additivity (already verified, re-state for the record)

Re-running is safe and accumulative:

- `slurm/c2_smoke/worker.sh:382` `stage_in()` copies each cell's **existing
  durable results** from the campaign root into localscratch *before* the
  orchestrator runs, "so the orchestrator's OWN resume check applies".
- The orchestrator validates `run_log.json` **content**, not merely existence,
  before skipping (`.claude/CLAUDE.md`, "Orchestrator resume"), so the OOM cell's
  partial output is treated as corrupt and re-run rather than skipped.
- `stage_out()` (`worker.sh:390`) merges back with `cp -a` — no deletion.

⇒ Repeated passes converge on one complete tree. **Do not delete the campaign
root to "start clean"** (§5.5): a partially completed triple is worse than a
missing one.

### 5.2 Recommended fix — one recovery pass at B=1/B=2, not eight at B=27

Re-launch **only the affected arrays** with a bundle small enough that the
deadline cannot bite, sized from the **p90**, not the median:

| Array group | Now | Recovery | Wall needed | Expected passes |
|---|---:|---:|---:|---:|
| `udfs:*:feynman` | B=27 | **B=2** | 25 h | 1 |
| `udfs:*:feynman_remainder` | B=4 | **B=2** | 25 h | 1 |
| `bingo:*:feynman`, `bingo:*:nguyen` | B=123/164 | **B=16** | ~25 h | 1 |

Mechanism: the launcher already exposes `C2_MAX_BUNDLE` as a ceiling
(`slurm/c2_campaign/CAMPAIGN_BRIEF.md` §8, "escape hatches"), and
`C2_PROBLEMS=<a:b:c>` restricts the problem set (**colon-separated** — `--export`
splits on commas).

Sketch, to be finalised after the sweeps drain:

```bash
# Per affected (method, arm, suite), campaign profile, SAME results root.
C2_PROFILE=campaign C2_MAX_BUNDLE=2 \
C2_RESULTS_DIR=$FSCRATCH/results/isalsr/c2_3arm \
C2_LOGS_DIR=$FSCRATCH/execs/isalsr/c2_3arm/logs_recovery \
  bash slurm/c2_smoke/launcher.sh
```

⚠ Points to settle before running this:

1. **`launcher.sh` submits all 42 arrays.** It has no "only these arrays" flag.
   Either add one, or drive `submit_paced.sh`-style per-array `sbatch` calls from
   a recovery script (`submit_sweeps.sh` is the closest existing template — it
   already iterates the plan row by row and can be filtered).
2. **Do not deploy to change this** while anything is running (defect 10). The
   recovery must use env vars and sbatch arguments, not edits to the deployed
   tree. If a code change is unavoidable, the campaign must be fully drained
   first, and the provenance split must be recorded.
3. **`C2_MIN_JOBID`** applies to any `submit_paced.sh`-based recovery, for the
   same job-name collision reason (EXECUTION-PLAN §11.1, 2026-08-07).
4. Logs to a **separate** `logs_recovery` dir, so `health.sh`'s `deferred=` count
   over `logs/*.out` keeps describing the main pass rather than mixing passes.

### 5.3 The single OOM cell

`bingo/cherrypicked/vlad_7/isalsr/seed_25`. The sweep retries it automatically at
the identical 32 GB. Bingo is not seed-reproducible (the accepted C3 finding), so
the retry is a genuine second draw, not a replay.

- If it succeeds → nothing to do.
- If it OOMs again → re-run that **one** cell at `--mem=64G`. Note in the ledger
  that this cell ran under a different memory request; memory is not a reported
  quantity (wall clock is), so this does not corrupt any measurement, but it must
  be disclosed.
- If it fails a third time → §5.5 applies: drop the whole `(bingo, Vlad-7, 25)`
  triple across **all three arms** rather than leave the paired design unbalanced.

### 5.4 Only then: aggregation and statistics

The existing aggregation (`1840324`) and ledger (`1840325`) are `afterany` on the
84 main+sweep arrays and will fire **before** any recovery pass exists. Their
output will therefore describe an incomplete tree.

Plan: let them run (they are harmless and their ledger names the gaps), then
**re-run aggregation + ledger after the recovery pass**:

```bash
# aggregation array, 42 tasks, one per config
sbatch --job-name=c2s_aggregate --array=1-42 ... slurm/c2_smoke/aggregate_worker.sh
# then the single full-root walk + certifier
sbatch --job-name=c2s_ledger ... --export=...,C2_LEDGER_ONLY=1,C2_EXPECTED_TASKS=12600 ...
```

🔴 `C2_EXPECTED_TASKS` must be **12600** (cells), not the task count.
`launcher.sh:625` passed the task count and made the certifier fall through to
the self-referential "disk" universe; `submit_paced.sh:191` gets it right. Fixed
in `cda276a`, but the deployed tree still carries the old value — **pass it
explicitly**.

---

## 6. Files to read to understand the mechanism

| File | Why |
|---|---|
| `slurm/c2_smoke/worker.sh` | The chunk loop. Lines 382 (`stage_in`), 390 (`stage_out`), 416–451 (the deadline and deferral), 494–496 (the summary the evidence above parses) |
| `experiments/scripts/c2_slot_plan.py` | Bundle/wall/throttle sizing. `CELL_HOURS` (190), `MIN_TASK_HOURS` (246), `DEADLINE_SAFETY` (254) |
| `experiments/scripts/c2_task_spec.py` | `decode_chunk(problems, seeds, bundle, task_index)` — maps an array index to its cells. Use it to name exactly which cells a task held |
| `slurm/c2_smoke/launcher.sh` | Submits the 42 arrays **and** the `c2w_*` sweeps (557–592) and the aggregation/ledger chain |
| `slurm/c2_campaign/submit_paced.sh` | What actually submitted C2. No sweep block; `C2_MIN_JOBID` guard |
| `slurm/c2_campaign/submit_sweeps.sh` | The sweep pass, and the closest template for a filtered recovery launcher |
| `slurm/c2_campaign/CAMPAIGN_BRIEF.md` | §7 runbook, §8 escape hatches (`C2_MAX_BUNDLE`, `C2_PROBLEMS`, `C2_SLOT_BUDGET`) |
| `slurm/c2_campaign/health.sh` | The probe; `cell_fail` / `deferred` / `sweep_bad` fields exist because of this investigation |
| `.claude/notes/review/tasks/EXECUTION-PLAN.md` | §5.5 completeness rule, §11.1 anomaly ledger, §11.3 launch ledger with all 84 job ids |

---

## 7. Acceptance criteria for "the campaign is complete"

1. `find $FSCRATCH/results/isalsr/c2_3arm -name run_log.json | wc -l` = **12600**.
2. Deferred total across **all** log dirs (main + sweep + recovery) = **0**.
3. `grep -h "^FAILED:" .../*.out` returns nothing, or every remaining failure is
   documented in EXECUTION-PLAN §11.1 with a §5.5 decision.
4. `c2_certify` run with `--expected-tasks 12600` reports **GO**,
   `n_blocking_failures = 0`, and `C1.15 expected_set_source: "registry"` with
   `expected == observed == 12600`.
5. `status_ledger.csv` reconciles with no unexplained cell.
6. Provenance is single-valued: every `run_log.json` reports
   `git_describe: campaign/c2`. **A recovery pass must not redeploy**, or this
   breaks and C2's whole rationale (§2, item 3) goes with it.

---

## 8. Open questions for Mario

1. **Is a bimodal-aware planner worth building**, or is a "small bundle for the
   recovery pass" good enough? A principled fix sizes the bundle from the p90 (or
   from `min(B such that B × p90 ≤ deadline)`), which would have given
   `udfs:feynman` B=2 from the start at the cost of more SLURM tasks — pushing
   back against SCBI's 2 h floor, since a fast `udfs:feynman` cell is 0.18 h.
   **This is a genuine tension: SCBI's floor wants big bundles, the tail wants
   small ones.** Worth stating in the reply to them if we change the shape.
2. **Should the recovery pass keep `--constraint=sr`?** Yes, in my view: C4 needs
   one CPU family and wall clock is a reported quantity. Confirm.
3. Do you want the `CELL_HOURS` table updated in-repo from this campaign's
   measurements (a good input for any future campaign), or left as the C1-derived
   values with the measurements recorded only here?
