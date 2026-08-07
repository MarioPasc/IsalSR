# Campaign C2 — the prepared submission

**Prepared 2026-08-06. Nothing here has been executed.**

> ## 🔴 STALE AS OF 2026-08-07 — the certified commit has moved
>
> **T19 landed after this file was written**, adding explored-DAG structural
> telemetry (`P7` in EXECUTION-PLAN §3.2). It had to land before launch, for the
> same reason as the T06 ledger: it is measured *during* a run, so no post-hoc
> pass can recover it, and missing it would mean re-running 12,600 jobs.
>
> **Consequence: `2ff0050` is no longer the commit to submit.** The table below
> still names it, and step 1 still re-cuts the tag onto it. Both are now wrong.
> Before submission:
>
> 1. Re-cut `campaign/c2` onto the new branch tip, not `2ff0050`.
> 2. **Re-run Stage C** on that commit — moving the tag invalidates the prior
>    certification (§4, `slurm/c2_tag_procedure.md`), which is exactly why v5e
>    was re-run last time.
> 3. Re-run Stage E and the Stage F gate on the same commit.
>
> **What is NOT invalidated**, by construction: T19 is pure Python and touches
> neither `canonical.py` nor `src/isalsr/core/native/`, so the native
> `build_hash 298fc1188bf1b051`, the engine equivalence gate and D3 hash
> soundness all stand. The re-certification is the cheap half.
>
> Evidence that the telemetry works on Picasso: probe array `1814948`,
> 24 cells, `slurm/t19_probe/verify.py`. Write-up:
> `.claude/notes/review/tasks/T19-dag-complexity-telemetry.md`.
>
> ⚠ There is also a second, **uncommitted** workstream in this tree at the time
> of writing (SCBI task chunking, touching `c2_task_spec.py`, `c2_slot_plan.py`
> and `slurm/c2_smoke/*`). Reconcile it before re-cutting the tag; do not
> certify a tree that is half-way through someone else's change.

Everything below is verified ready except the tag move, which is step 1.

| | state |
|---|---|
| Certified commit | **`2ff0050`** — Stage C **v5e GO, 19/19, 0 SKIPs**, 1,260/1,260, single provenance |
| Deployed on Picasso | `2ff0050`, clean, `engine=cpp`, `build_hash 298fc1188bf1b051` |
| Campaign payload | **proven** — 42-task campaign-profile probe, 42/42 COMPLETED, 3 arms × 14 |
| Stage E | 7/7, no `--allow-mixed-provenance` (on v5d; analyzer unchanged since) |
| D3 | hash soundness 0 unsound merges; fidelity 0/119,795 |
| Inode headroom | 84,381 needed against ~100,000 free |
| Campaign root | empty |
| 🔴 `campaign/c2` tag | **points at `24f83a0`, whose `worker.sh` cannot run — must move** |

---

## 1. Move the tag onto the certified commit

The tag was cut before the seed defect was found. Procedure §4: moving it
invalidates the prior certification, which is why v5e was re-run — v5e certifies
`2ff0050`, so the tag belongs there.

```bash
cd ~/research/code/IsalSR
git fetch origin

# Delete the stale tag, locally and remotely.
git tag -d campaign/c2
git push origin :refs/tags/campaign/c2

# Re-cut on the certified commit.
git tag -a campaign/c2 2ff0050 -m "Campaign C2: three-arm re-execution on the native engine.

Arms:      baseline, hash, isalsr
Seeds:     1..30
Alphabet:  decomposed (T16)
Engine:    native
Node pool: sr
Cohort:    70 problems (D1 50 + D2 20), 42 arrays, 12,600 runs
Scope:     canonical completeness claimed for k >= 1 (D3, 2026-08-06)
Certified: Stage C v5e on this commit (19/19 GO, 1,260/1,260, single
           provenance); Stage E 7/7 with no provenance override; D3 hash
           soundness 0 unsound merges; 42-task campaign payload probe 42/42."

git rev-parse campaign/c2^{commit}    # must print 2ff0050...
git push origin campaign/c2
```

> ⚠ If the working tree is dirty with another agent's work, that is fine for
> tagging (a tag references a commit, not the tree) but **not** for
> `deploy.sh`, which refuses a dirty tree and is right to. Deploy from a clean
> clone of the branch tip instead. Never commit someone else's files.

## 2. Redeploy so the remote reports the tag

Every run records `git describe`; without this they record `…-2-g2ff0050`
instead of `campaign/c2`, and §5.1 wants the tag.

```bash
bash slurm/c2_smoke/deploy.sh     # from a clean checkout of 2ff0050
# expect: SP-1 OK: remote is exactly campaign/c2, clean
```

## 3. Run the gate — must be 12/12

```bash
bash slurm/c2_campaign/stage_f_preflight.sh
# expect: VERDICT: GO -- every blocking check passed
```

G12 is new and specifically covers the defect that killed the first submission.

## 4. Submit — **paced**

```bash
ssh picasso 'cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR && \
             bash slurm/c2_campaign/submit_paced.sh --dry-run'

ssh picasso 'cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR && \
             bash slurm/c2_campaign/submit_paced.sh'
```

Takes ~15 minutes of wall clock for the 42 arrays (20 s apart), then submits the
aggregation array and the ledger job.

**Do not use `launch.sh` for this.** It submits in a tight loop, which is what
got `Resource temporarily unavailable` after 29 of 42 arrays and left them
untracked. `submit_paced.sh` is idempotent — if it is refused, re-run it and it
completes the set; raise `C2_SLEEP=45` if it is refused twice.

## 5. Immediately after

```bash
# 1. Confirm 42 arrays and 12,600 tasks, and that tasks are NOT failing fast.
ssh picasso 'squeue'
ssh picasso "sacct -S today -n -P -X -o JobID,State | sort | uniq -c | head"

# 2. Within ~10 min, confirm run logs are appearing.  Zero run logs with many
#    COMPLETED/FAILED tasks is the signature of the 2026-08-06 failure.
ssh picasso 'find $FSCRATCH/results/isalsr/c2_3arm -name run_log.json | wc -l'

# 3. Record the 42 job ids in EXECUTION-PLAN §11.3 (submit_paced.sh writes them
#    to <logs>/job_ids.txt in ascending order).
```

## 6. Day 1 — rebalance

The plan is deliberately weighted at `T_bingo = 8 h`. Once real Bingo wall
clocks exist, recover the ~9 h that pessimism costs:

```bash
python -m experiments.scripts.c2_slot_plan \
       --bingo-hours <measured> --rebalance <logs>/job_ids.txt
```

`scontrol update JobId=<id> ArrayTaskThrottle=<n>` re-apportions a **running**
array; it touches no config and no deployed file, so it is not defect 10.

## 7. Standing rules

- **Do not deploy while the arrays run** (defect 10) — a mid-wave redeploy
  splits provenance across two HEADs.
- Lowering `C2_SLOT_BUDGET` mid-campaign is safe; it is not defect 10.
- If capacity forces truncation, drop whole `(method, problem, seed)` triples
  across all three arms, never individual runs (§5.5).
- SCBI have been mailed about the 12,600-task footprint (Mario, 2026-08-06).
  If they ask for a smaller resident set, the chained-batch design is assessed
  in EXECUTION-PLAN §11.1 and converts safely mid-campaign, because the resume
  logic skips completed cells.
