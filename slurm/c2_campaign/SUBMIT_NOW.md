# Campaign C2 — the launch record

> ## ✅ EXECUTED 2026-08-07 17:52 CEST
>
> C2 is **submitted and running** on commit `2dd56fd`, tag `campaign/c2`.
> This file is now a record of what was done, not a set of instructions.
> The living runbook is `CAMPAIGN_BRIEF.md` §7; the ledgers are
> `EXECUTION-PLAN` §11.1 (anomalies) and §11.3 (job ids).
>
> 🔴 **Do not run `deploy.sh` while the arrays run** (defect 10). Five defects
> were found during this launch and their fixes are deliberately **not**
> deployed to this campaign — deploying them mid-flight would split provenance
> across two HEADs, which is exactly what cost v4 161 of its 1,260 cells.

---

## 1. What was submitted

| | |
|---|---|
| Commit | `2dd56fd`, tag `campaign/c2` (annotated, pushed) |
| Native build | `build_hash 298fc1188bf1b051`, `engine=cpp` — unchanged by T19 |
| Main arrays | **42**, 3,474 tasks, **12,600 cells** |
| Sweep arrays | **42**, `afterany` on the mains |
| Aggregation | `1840324`, 42 tasks, `afterany` on all **84** arrays |
| Ledger + certifier | `1840325`, `afterany` on `1840324`, `C2_EXPECTED_TASKS=12600` |
| Node pool | `sr`, `--mem` 16 G UDFS / 32 G Bingo, 1 core |
| Campaign root | `$FSCRATCH/results/isalsr/c2_3arm` (was empty at G8) |
| Submission | `submit_paced.sh` at 20 s — **zero refusals** |

Job ids: `EXECUTION-PLAN` §11.3, and on the cluster in
`$FSCRATCH/execs/isalsr/c2_3arm/logs/{job_ids,sweep_job_ids,job_ids_aggregation}.txt`.

## 2. The gates it passed

**Stage C v6**, re-run on this HEAD because T19 and the SCBI chunking changed
`src/`, `experiments/` and `slurm/c2_smoke/` — G11 compares those byte-for-byte,
so the v5e certification on `2ff0050` was void.

- 1,260/1,260 cells, **19/19 criteria PASS**, 0 blocking failures
- single provenance `campaign/c2-13-g2dd56fd`, `engine_histogram: {native: 1260}`
- C1.13 alphabet clean (no `-`, `/`, `V-`, `V/`) — T16 decomposition holding
- C1.15 `expected_set_source: registry`, expected = observed = 1,260
- C4 210 problem-seed pairs, multiplicity 6, 0 missing fingerprints

**Chunk smoke** on the deployed tree: 24/24 tasks COMPLETED,
`SMOKE OK -- localscratch loses nothing`, `complexity.json` (T19) in every cell.

**Local pre-flight**: unit suite 7,625 passed / 1 failed / 5 skipped (the one
failure is a manuscript audit against an external Overleaf checkout — see
§11.1); `mock_chunk_test.sh` **26/26** including the SIGTERM copy-back.

**Stage F**: **13 PASS / 0 FAIL**. Inode projection 75,255 against 86,600 free.

## 3. Two overrides that the runbooks do not mention

Both are recorded in full in `EXECUTION-PLAN` §11.1. They are repeated here
because anyone re-reading this file to re-launch will hit them again.

### `C2_MIN_JOBID` is **mandatory** whenever Stage C ran the same day

`submit_paced.sh` skips arrays whose **job name** already appears in
`sacct -S today`, scoped by `C2_MIN_JOBID` (default **0**). The smoke and
campaign profiles build the *same* 42 job names. Since §4 requires Stage C to be
re-run on the commit being submitted, the default makes the campaign a silent
no-op: 42 SKIPs, zero arrays submitted, aggregation attached to the **smoke**
wave, exit 0.

```bash
MIN=$(( $(sacct -S today -n -P -X -o JobID | cut -d_ -f1 | sort -n | tail -1) + 1 ))
C2_MIN_JOBID=$MIN bash slurm/c2_campaign/submit_paced.sh --dry-run
# the dry run MUST report "already present (job id >= $MIN): 0"
```

### The sweep arrays must be submitted separately

`submit_paced.sh` has no sweep block; `launcher.sh` does. Without them, cells
that a task's deadline refused to start are never recovered, and the aggregation
runs against a tree missing them. After the mains are up, submit the 42 `c2w_*`
arrays `afterany` on the mains, then repoint the aggregation:

```bash
scontrol update JobId=<agg> Dependency=afterany:<42 mains>:<42 sweeps>
scontrol show job <agg> | tr ' ' '\n' | grep '^Dependency='   # expect 84 entries
```

## 4. Watching it

```bash
ssh picasso 'squeue'                      # NOT squeue -u (the Lua wrapper rejects it)
ssh picasso 'sacct -X -S today -n -P -o State -j $(paste -sd, <logs>/job_ids.txt) | sort | uniq -c'
ssh picasso 'find $FSCRATCH/results/isalsr/c2_3arm -name run_log.json | wc -l'
ssh picasso 'grep -l "\[FATAL\] copy-back" <logs>/*.out | wc -l'   # must stay 0
```

`[FATAL] copy-back` means a task's results are still on a compute node and it
deliberately did not delete them. Recover before the node's scratch is reclaimed.

**Day 1 rebalance.** The plan is weighted at `T_bingo = 8 h` on purpose. Once
real Bingo wall clocks exist:

```bash
python -m experiments.scripts.c2_slot_plan --bingo-hours <measured> \
       --rebalance <logs>/job_ids.txt        # emits 42 scontrol lines
```

`scontrol update JobId=<id> ArrayTaskThrottle=<n>` re-apportions a **running**
array; it touches no config and no deployed file, so it is not defect 10.

## 5. Rollback

```bash
ssh picasso 'scancel --name=c2s_*; scancel --name=c2w_*'
```

Resume is additive — a completed `(method, arm, problem, seed)` is skipped.
**Do not delete the campaign root to "start clean"**: the status ledger is what
makes cell completeness provable (§5.5), and a partially completed triple is
worse than a missing one because it silently unbalances the paired test.
