# C2 — deferred cells: evidence, mechanism, and a recovery plan

**Status: §1–§8 written as a draft 2026-08-09 13:30 CEST at 83 % complete.
§9–§12 added 2026-08-09 15:30 CEST: the mechanism is now BUILT and TESTED on
Picasso, and §5.2's prescription is corrected. Nothing has been run against the
campaign root — the recovery pass itself still waits for §10 step 1.**

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

> 🔴 **This table is wrong in one place and loose in another; both are corrected
> in §9.** `B=2` at a **25 h** wall gives a start-deadline of `25 − 12.5 = 12.5 h`,
> and the worst-case elapsed time at which cell 2 begins is *also* 12.5 h. The
> worker defers on `elapsed >= cutoff` (`worker.sh:450`), so the second cell
> defers on the boundary — the prescription reintroduces, at the margin, exactly
> the failure it was written to remove. The implemented rule sizes the wall as
> `(B−1)·allowance + CELL_RESERVE` and then adds one hour (`floor(x)+1`, not
> `ceil(x)`) precisely to make the inequality **strict**, which yields **B=3 at a
> 38 h wall** (deadline 25.5 h against a worst case of 25.0 h). The `B=16` figure
> for Bingo was a guess rather than a derivation; the derived values are B=3
> (`safe`) or B=38/63 (`p90`).

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

> **Answers, 2026-08-09, from the implementation and its tests — see §11.**
> **Q1: built, but scoped to the recovery pass only.** `c2_slot_plan.py` gained a
> `--recovery` mode; `build_plan` is byte-for-byte unchanged and the main pass is
> untouched. Rewriting `bundle_size` itself was rejected *for now* because
> `submit_sweeps.sh` reads the planner from the **deployed** tree, so a change
> there alters the shape of any re-run while the campaign is live. The tension
> you named is real and is now quantified rather than asserted (§11.3).
> **Q2: kept.** `C2_CONSTRAINT` defaults to `sr`; only the throwaway tests used
> `cpu`. **Q3: neither — a *second* table.** `CELL_HOURS` (central, sizes for
> throughput) is untouched; `P90_CELL_HOURS` (tail, sizes for the deadline) is
> new. Collapsing them would lose the very property that caused this: for
> `udfs:feynman` the right central estimate and the right tail estimate differ by
> 67×, and the bundle needs both.

---

## 9. What was built

Three pieces, all **additive**. Nothing in the deployed tree was touched, no tag
was moved, and `build_plan` still emits the same 42 rows it emitted at launch
(pinned by `test_main_plan_is_unaffected_by_the_recovery_additions`).

| File | Role |
|---|---|
| `experiments/scripts/c2_missing_cells.py` | **NEW.** Enumerates the cells a results root does not contain, by comparing the registry universe against `run_log.json` on disk. `--summary`, `--list`, `--selectors`, `--strict`. Exit 0 = complete, 2 = gaps, 1 = error |
| `experiments/scripts/c2_slot_plan.py` | **EXTENDED.** `P90_CELL_HOURS`, `recovery_allowance_h`, `recovery_bundle`, `recovery_wall_hours`, `defers_nothing`, `build_recovery_plan`, `match_selector`, and a `--recovery` CLI mode |
| `slurm/c2_campaign/submit_recovery.sh` | **NEW.** Paced, idempotent, filtered submitter with the `c2r_` job-name prefix |
| `tests/unit/test_c2_recovery_plan.py` | **NEW.** 65 unit tests |

### 9.1 The sizing rule, and why it is the one that ends the deferral

A chunked task starts cell *i* only if `elapsed < wall − CELL_RESERVE_H`. Charge
every cell an allowance `a` and size the wall as

```
wall  =  floor( (B−1)·a  +  CELL_RESERVE_H )  +  1          # hours
cutoff = wall − CELL_RESERVE_H                              # what the worker gets
```

then the last cell of the chunk begins at worst at `(B−1)·a`, which is **strictly**
below `cutoff`. `floor(x)+1` rather than `ceil(x)` is deliberate: the test is
`elapsed >= cutoff`, so equality defers.

Two modes:

| Mode | Allowance `a` | Guarantee | `udfs:*` | `bingo:nguyen` | `bingo:feynman` |
|---|---|---|---:|---:|---:|
| **`safe`** (default) | `CELL_RESERVE_H` = 12.5 h | **Distribution-free.** Every cell is charged the full payload cap, which `max_time: 43200` makes a hard bound | B=3, 38 h | B=3, 38 h | B=3, 38 h |
| `p90` | `min(12.5, p90 + 0.5)` | Probabilistic: holds unless several cells of one chunk land above p90 | B=3, 38 h | B=38, 47 h | B=63, 47 h |

`safe` is the default because the 2026-08-09 measurement is exactly the argument
against betting on a distribution: `udfs:feynman` spans 67× *inside one suite*.
`p90` exists for the case where the placement count matters more than the
guarantee — it cuts the twelve worst arrays from 1,140 tasks to 525.

### 9.2 Provenance: two trees, and only numbers cross between them

Acceptance criterion 6 requires every `run_log.json` to report
`git_describe: campaign/c2`, and §5.2's point 2 forbids a redeploy. But the
`--recovery` mode did not exist when the campaign was deployed. `submit_recovery.sh`
therefore splits the two roles explicitly:

* `C2_TOOLS_DIR` — a **separate** checkout, which computes the plan;
* `ISALSR_REPO_DIR` — the **deployed** tree, which supplies `worker.sh`, the
  configs and the payload, and therefore the tag.

Only `bundle`, `wall`, `cutoff` and `n_tasks` cross that boundary, as `--export`
values. The script `cmp`s `c2_task_spec.py` and every selected config between the
two trees and refuses to submit if they differ — `c2_task_spec.py` owns the
partition, so a divergence there would hand a task cells the plan never sized it
for. Verified: the deployed tree is `2dd56fd` / `campaign/c2`, and the recovery
checkout matches it byte-for-byte on all 23 configs and on `c2_task_spec.py`.

> 🔴 **A recovery pass must not use `python -m` (found 2026-08-09).**
> `experiments` is a **namespace package**, and the conda env's editable install
> registers a meta-path finder that contributes the **deployed** checkout to
> `experiments.__path__` *ahead of everything `PYTHONPATH` adds*. Measured:
> ```
> experiments.__path__ = ['/mnt2/.../repos/IsalSR/experiments',
>                         '/mnt2/.../repos/IsalSR_recovery/experiments', ...]
> ```
> So `python -m experiments.scripts.c2_slot_plan` from the recovery checkout
> loads the **deployed** planner and dies with `unrecognized arguments:
> --recovery`. The submitter therefore invokes the planner **by absolute file
> path**, and `c2_missing_cells.py` imports **nothing** from `c2_slot_plan`
> (its three axes and `match_selector` are re-declared locally, with a unit test
> pinning them to the planner's definitions so they cannot drift). This is the
> same family as defect 10: two checkouts, one import namespace.

---

## 10. Workflow

Ordered, copy-pasteable. **Steps 1–2 are the gate; do not start at step 3.**

### Step 0 — one-time: put the recovery tools on Picasso

The deployed tree must not change (§5.2, point 2), so the tools live beside it.

```bash
# from the workstation, in the IsalSR repo
rsync -az --delete \
  --exclude '.git' --exclude '.claude' --exclude 'docs' --exclude 'reviews' \
  --exclude 'article' --exclude 'results' --exclude 'scratchpad' \
  --exclude '__pycache__' --exclude '*.egg-info' --exclude '.hypothesis' \
  --exclude 'build' --exclude '.mypy_cache' --exclude '.pytest_cache' \
  --exclude '.ruff_cache' --exclude '*.so' \
  ./ picasso:/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR_recovery/
```

⚠ **Do NOT run `slurm/c2_smoke/deploy.sh`** — it targets the live tree.

### Step 1 — wait for the campaign AND the sweeps to drain

```bash
ssh picasso 'bash $FSCRATCH/repos/IsalSR/slurm/c2_campaign/health.sh'
ssh picasso '/usr/bin/squeue -u $USER -h -o "%j" | grep -cE "^c2[sw]_"'   # must print 0
```

`submit_recovery.sh` enforces this itself and refuses while anything `c2s_*` or
`c2w_*` is queued. The reason is §5.0: two passes over one cell race on the
per-cell copy-back.

### Step 2 — find out what is actually missing

```bash
ssh picasso
F=/mnt/home/users/tic_163_uma/mpascual/fscratch
PY=$F/conda_envs/isalsr/bin/python
D=$F/repos/IsalSR_recovery

$PY $D/experiments/scripts/c2_missing_cells.py \
    --results-dir $F/results/isalsr/c2_3arm --seeds 1-30 --strict --summary
```

Exit status 0 means the tree is complete and **there is nothing to do — stop
here**. Exit 2 means there are gaps.

### Step 3 — scope the recovery pass to exactly those arrays

```bash
SEL=$($PY $D/experiments/scripts/c2_missing_cells.py \
        --results-dir $F/results/isalsr/c2_3arm --seeds 1-30 --strict --selectors)
echo "$SEL"
```

### Step 4 — dry run, then submit

```bash
bash $D/slurm/c2_campaign/submit_recovery.sh --only "$SEL" --dry-run
bash $D/slurm/c2_campaign/submit_recovery.sh --only "$SEL"
```

Defaults: `safe` mode, the **campaign** results root, `--constraint=sr`,
`logs_recovery/`, `c2r_` job names, 20 s pacing. Job ids land in
`$FSCRATCH/execs/isalsr/c2_3arm/logs_recovery/recovery_job_ids.txt`.

Add `--mode p90` if the placement count matters more than the guarantee, or
`--with-aggregation` to chain step 6 automatically.

The script is idempotent by job name: if it aborts part-way, **re-run it** and it
submits only what is missing.

### Step 5 — verify the pass closed the gap

```bash
L=$F/execs/isalsr/c2_3arm/logs_recovery
grep -h "^Cells: *[0-9]* ok," $L/*.out \
  | sed -n 's/.*, \([0-9]*\) deferred.*/\1/p' | awk '{s+=$1} END {print s+0}'   # expect 0
grep -h "^FAILED:" $L/*.out | sort -u                                          # expect empty

$PY $D/experiments/scripts/c2_missing_cells.py \
    --results-dir $F/results/isalsr/c2_3arm --seeds 1-30 --strict --summary
find $F/results/isalsr/c2_3arm -name run_log.json | wc -l                      # expect 12600
```

If cells are still missing, re-run steps 3–4. The pass is additive (§5.1, §11.1)
so repetition converges; in `safe` mode a second pass should not be needed.

### Step 6 — re-run aggregation and the status ledger

The launch-time aggregation (`1840324`) and ledger (`1840325`) fire *before* any
recovery pass exists and describe an incomplete tree. Re-run them:

```bash
R=$F/repos/IsalSR                       # DEPLOYED tree, for provenance
LOGS=$F/execs/isalsr/c2_3arm/logs_recovery
CONFIGS=$(for m in udfs bingo; do for s in nguyen feynman hard cherrypicked \
          roundoff feynman_remainder strogatz; do \
          printf '%s/experiments/configs/%s_%s.yaml ' "$R" "$m" "$s"; done; done)

AGG=$(sbatch --parsable --job-name=c2r_aggregate --array=1-42 --time=0-01:59:00 \
  --ntasks=1 --cpus-per-task=2 --mem=16G --constraint=sr --account=tic_163_uma \
  --output=$LOGS/c2r_aggregate_%A_%a.out --error=$LOGS/c2r_aggregate_%A_%a.err \
  --export="ALL,ISALSR_REPO_DIR=$R,C2_RESULTS_DIR=$F/results/isalsr/c2_3arm,C2_CONFIG_LIST=${CONFIGS% }" \
  $R/slurm/c2_smoke/aggregate_worker.sh | tail -1 | tr -cd '0-9')

sbatch --job-name=c2r_ledger --time=0-02:00:00 \
  --ntasks=1 --cpus-per-task=1 --mem=16G --constraint=sr --account=tic_163_uma \
  --dependency="afterany:$AGG" \
  --output=$LOGS/c2r_ledger_%j.out --error=$LOGS/c2r_ledger_%j.err \
  --export="ALL,ISALSR_REPO_DIR=$R,C2_RESULTS_DIR=$F/results/isalsr/c2_3arm,C2_LEDGER_ONLY=1,C2_EXPECTED_TASKS=12600,C2_MAX_TIME=43200" \
  $R/slurm/c2_smoke/aggregate_worker.sh
```

🔴 `C2_EXPECTED_TASKS=12600` is the **cell** count, not the task count. The
deployed `launcher.sh:625` passes the task count, which makes the certifier fall
through to the self-referential "disk" universe (fixed in `cda276a`, but the
deployed tree still carries the old value). `submit_recovery.sh --with-aggregation`
does all of step 6 with this value already set.

Then run the §7 acceptance checks.

---

## 11. Validation

Everything below was executed on 2026-08-09 **while the campaign was live**
(966 tasks running, 44 pending, 10,531 → 10,547 `run_log.json` during the
session). Nothing was written to `$FSCRATCH/results/isalsr/c2_3arm` or
`.../execs/isalsr/c2_3arm`; the deployed tree was read only and `git status`
on it is still clean at `2dd56fd` / `campaign/c2`.

### 11.1 Controlled, on the workstation

| Check | Result |
|---|---|
| `pytest tests/unit -q` | **7,688 passed, 1 failed, 5 skipped** — baseline was 7,625/1/5, so +63 new and the same single pre-existing failure (`test_numerical_audit`, a manuscript audit, untouched by this work) |
| `pytest tests/unit/test_c2_recovery_plan.py -q` | **65 passed** |
| `ruff check` / `ruff format --check` | clean on all three changed/added Python files |
| `bash -n` | clean on `submit_recovery.sh`, `submit_sweeps.sh`, `submit_paced.sh` |
| **Main plan regression** | `c2_slot_plan --seeds 1-30 --tsv` **byte-identical** to the pre-edit output (`git stash` A/B). This is the check that matters most while the campaign runs |

The unit tests assert the property directly rather than the outputs: for every
`(method, suite, mode)` triple, `defers_nothing(B, a, wall)` holds and `B+1`
breaks it — i.e. the bundle is the *largest* that still clears the deadline. They
also assert that `decode_chunk` partitions the array exactly once at B ∈ {1, 2, 3,
7, 27, 63, 123, 164} and that the cell **set** is identical at B=27 and B=3, which
is what makes re-running at a different bundle sound.

### 11.2 On Picasso — throwaway root, `--constraint=cpu`, 60 s payload

Results root `$FSCRATCH/results/isalsr/c2_recovery_test`, logs
`$FSCRATCH/execs/isalsr/c2_recovery_test/logs`, both **deleted afterwards**
(161 files). Payload `udfs_feynman.yaml`, problem `I.6.20a`, `C2_MAX_TIME=60`.
13 tasks total. The worker was the **deployed** one throughout.

**The deferral, reproduced and then removed** — same six cells, same payload:

| Job | Shape | Result |
|---|---|---|
| `1867715` `c2t_old` | B=6, cutoff **150 s** (under-provisioned, mimics the campaign) | `Cells: 3 ok, 0 failed, **3 deferred** (of 6)`; `DEFERRED: I.6.20a/seed=4 seed=5 seed=6` |
| `1867723` `c2t_new` | B=3, cutoff **360 s** = `(B−1)·(payload+teardown)`, the §9.1 rule scaled | task 1 `3 ok, 0 failed, **0 deferred**`; task 2 `3 ok, 0 failed, **0 deferred**` |

**Additivity and resume — the key test.** Same root, two passes:

| Job | Pass | Elapsed | Evidence |
|---|---|---|---|
| `1867731` `c2t_add1` | seeds 1–3, B=3 | 3 m 33 s | 3 cells written |
| `1868137_1` `c2t_add2` | seeds 1–3 again | **27 s** | `Skipping I.6.20a seed=1 variant=baseline (already exists)` ×3 |
| `1868137_2` `c2t_add2` | seeds 4–6 | 3 m 56 s | 3 new cells written |

`run_log.json` mtimes for seeds 1–3 are **bit-identical before and after**
(`1786277415.457…`, `1786277482.333…`, `1786277547.626…`), so the completed cells
were not re-executed, not rewritten, and not touched — while the missing ones
ran. All six present afterwards. **§5.1's additivity claim is now measured, not
argued.**

**`submit_recovery.sh` end to end** — `1868143` `c2r_ub_feynman`, 7 tasks,
`udfs:baseline:feynman` at 2 seeds (20 cells, B=3):

* all 7 tasks `COMPLETED` (1 m 56 s – 4 m 00 s);
* `6 × "3 ok, 0 failed, 0 deferred"` and `1 × "2 ok, 0 failed, 0 deferred"` —
  **20/20 cells, zero deferred**;
* `find … -name run_log.json | wc -l` = **20**;
* the worker logged `SP-1 tag: campaign/c2`, `SP-1 commit: 2dd56fd…` — **the
  provenance is the campaign's even though the plan came from a different
  checkout**, which is the whole point of §9.2.

**Fail-closed guards, each exercised:**

| Guard | Behaviour |
|---|---|
| Live campaign | `FATAL: 1007 c2s_*/c2w_* job(s) are still queued or running.` — refuses (§5.0) |
| No `--only` | Refuses, and prints the `c2_missing_cells.py` command that produces it |
| `--dependency afterany-123` | `FATAL: … is not after{any,ok,notok,corr}:<ids>` — refuses **before** submitting anything |
| Malformed job id | `_clean_job_id` is `submit_paced.sh`'s (`tail -n 1 | sed`), verified against the Lua wrapper's ANSI banner: it returned `1868143` correctly where a naive `tr -dc 0-9` over the whole output returned `3101867715` |
| Tools/worker divergence | `cmp` on `c2_task_spec.py` + all 23 configs — identical, so the submission proceeded |

**Job-name collision** — the trap §5.2 point 3 warns about:

```
c2s_/c2w_ names in sacct today : 1379
c2r_ names in sacct today      :    0
queued jobs matching ^c2r_     :    0
```

`c2r_` is disjoint from both live prefixes, so the idempotence query cannot be
masked by the campaign and cannot mask it. Recovery logs also go to a separate
`logs_recovery/`, so `health.sh`'s `deferred=` count over `logs/*.out` keeps
describing the main pass.

### 11.3 Measured numbers this work produced

`measure_cell_hours.py` re-run over the live root (n = 10,533 cells with a
recorded `wall_clock_total_s`) confirms §3's table and adds the max:

| method:suite | n | p50 | **p90** | p99 | max |
|---|---:|---:|---:|---:|---:|
| `udfs:*` (all seven suites) | 4,783 | 12.00 | 12.00 | 12.00 | 12.00 |
| `udfs:feynman` | 462 | **0.18** | **12.00** | 12.00 | 12.00 |
| `bingo:feynman` | 740 | 0.00 | 0.05 | 3.92 | 6.69 |
| `bingo:nguyen` | 919 | 0.03 | 0.43 | 2.52 | 3.04 |
| `bingo:strogatz` | 1,260 | 0.09 | 1.01 | 2.47 | 3.30 |
| `bingo:feynman_remainder` | 435 | 0.08 | 4.21 | 6.62 | 8.64 |
| `bingo:roundoff` | 686 | 2.05 | 5.55 | 7.21 | 8.51 |
| `bingo:cherrypicked` | 880 | 3.69 | 6.47 | 8.42 | 10.18 |
| `bingo:hard` | 830 | 3.29 | 7.18 | 10.83 | 11.76 |

The census gives the gap **directly**, which nothing before this could do. At
14:00 CEST, with the campaign still running:

```
TOTAL   expected 12600   present 10540   missing 2060
```

Of that, `udfs:*:feynman` alone accounts for 438 and `udfs:*:feynman_remainder`
for 271 — the two array groups §2.2 identified. This number includes cells still
in flight, so it is an upper bound on the true deferral; the deferred counter
over the campaign log dir read **531** at the same moment.

**The SCBI trade-off, quantified rather than asserted.** For the twelve arrays
§2.2 names:

| Mode | SLURM tasks | Shortest expected task | SCBI 2 h floor |
|---|---:|---:|---|
| `safe` | 1,140 | 0.45 h (`bingo:*:nguyen`) | violated by the 6 Bingo arrays |
| `p90` | 525 | 5.70 h | cleared by every array |

`format_recovery_notes` prints exactly this, per array, so the choice is never
silent. The honest framing for SCBI is that **a recovery task is short because
most of its cells are already complete** — `1868137_1` above ran three cells in
27 s — and no bundle size changes that; sizing up to clear the floor only
reintroduces the deadline.

---

## 12. Final conclusions

**What the fix is.** One recovery pass at a bundle small enough that the worker's
own start-deadline cannot bite, submitted only over the arrays that still have
gaps, from a separate tools checkout, against the unchanged deployed worker, under
a job-name prefix that cannot collide with anything live. In `safe` mode the
no-deferral property is **distribution-free**: every cell is charged the full 12 h
payload cap, so a chunk of three cannot fail to start its third cell no matter how
the per-cell distribution behaves. This replaces the ≈9 sweep passes and ≈12 days
§4 estimated with **one pass**.

**What it does NOT fix.**

1. **It does not make the main planner tail-aware.** `build_plan` is unchanged, so
   a future campaign submitted the same way will defer the same way. The fix for
   that is to move `bundle_size` onto `P90_CELL_HOURS`; it was deliberately not
   done here because `submit_sweeps.sh` reads the planner from the **deployed**
   tree, and changing it mid-flight is defect 10 by another route. **Do this
   before the next campaign, not during this one.**
2. **It does not recover a cell whose payload genuinely fails.** The single OOM
   (`bingo/cherrypicked/vlad_7/isalsr/seed_25`) is a §5.3 matter; a recovery pass
   retries it at the same 32 GB and nothing more.
3. **It does not shorten `udfs`.** Three cells at 12 h is a 36 h task. The pass is
   fast in *placements*, not in wall clock.
4. **It cannot distinguish "deferred" from "still running"** while the campaign is
   live — hence the step-1 gate. The 2,060 figure above is an upper bound.

**Residual risks.**

* **Namespace shadowing (§9.2).** Any *future* tool added to the recovery checkout
  that imports a symbol the deployed tree's copy does not define will fail at run
  time. The failure is loud, and `test_missing_cells_imports_nothing_from_the_planner`
  guards the one case that exists — but the trap is structural.
* **`p90` mode is a bet.** At B=63 a `bingo:feynman` chunk holds ~6 cells above
  p90 in expectation, and the observed max is 6.69 h. It clears the 34.5 h
  deadline comfortably at the mean, but it is not a proof. Prefer `safe` unless
  the placement count is the binding constraint.
* **The `safe` margin is 30 minutes.** B=3 at a 38 h wall gives a 25.5 h deadline
  against a 25.0 h worst case. That half-hour is the teardown budget for two
  cells; the largest post-search tail on record is 7 minutes (§11.1, 2026-08-03),
  so the margin is ~4× the observed worst — but it is not unbounded. If teardown
  ever grows, use `--bundle 2` (deadline 13.5 h against a 12.5 h worst case).
* **The recovery pass runs 1,140 tasks at `safe`**, which is a real load on the
  scheduler and a real conversation with SCBI. §11.3 has the numbers for it.

**Standing warnings, unchanged.** §5.5 still governs: every run either produces a
valid `run_log.json` or a ledger row naming its cause, and any truncation drops
whole `(method, problem, seed)` triples across **all three arms**. And **do not
delete the campaign root to "start clean"** — resume is additive precisely so that
you never have to, and a partially completed triple is worse than a missing one.
