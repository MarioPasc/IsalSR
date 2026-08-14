# C2 close-out — handoff prompt for a successor agent

Paste everything inside the fence below as a `/loop` argument, i.e.

```
/loop <paste the block>
```

It is written to be self-contained: a fresh agent with no memory of this session
should be able to finish the campaign from it.

---

```
You are taking over the final stage of IsalSR campaign C2 on the Picasso HPC
cluster (UMA). Loop: check state, act only when a gate opens, otherwise
re-schedule and stay quiet. Repo: /home/mpascual/research/code/IsalSR
Python: ~/.conda/envs/isalsr/bin/python

════════════════════ WHAT C2 IS ════════════════════
The TPAMI-revision campaign: {baseline, hash, isalsr} x {UDFS, Bingo} x 70
problems x 30 seeds = 12,600 cells, ~80,000 core-hours, submitted 2026-08-07 on
commit 2dd56fd / tag campaign/c2. The three arms differ ONLY in deduplication:
baseline none, hash a fixed-order serialisation, isalsr the canonical string
(1-WL, C++ native engine). Fitness must therefore be near-identical across arms;
a systematic R2 gap between arms would be a BUG, not a result.

Main pass + sweeps are DONE. A recovery pass is finishing the last deferred
cells. Nothing else is outstanding.

════════════════════ PATHS ════════════════════
Picasso campaign root : $FSCRATCH/results/isalsr/c2_3arm        <- THE SCIENCE
Picasso logs          : $FSCRATCH/execs/isalsr/c2_3arm/logs{,_recovery}
Deployed tree         : $FSCRATCH/repos/IsalSR        (2dd56fd, PROVENANCE)
Tools checkout        : $FSCRATCH/repos/IsalSR_recovery   (recovery scripts)
Local backup          : /media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/review/c2_3arm
where $FSCRATCH = /mnt/home/users/tic_163_uma/mpascual/fscratch
Layout: <method>/<suite>/<problem_slug>/<arm>/seed_NN/run_log.json
(slug is lowercase, e.g. Vlad-7 -> vlad_7; seeds zero-padded: seed_01)

════════════════════ JOBS TO WATCH ════════════════════
Recovery arrays (7), job ids:
  1960414 c2r_ub_feynman     1960514 c2r_uh_feynman   1960614 c2r_ui_feynman
  1960714 c2r_bb_feynman     1960719 c2r_bh_feynman   1960724 c2r_bi_nguyen
  1960734 c2r_bi_feynman
Also at $FSCRATCH/execs/isalsr/c2_3arm/logs_recovery/recovery_job_ids.txt

Health probe (in repo): bash slurm/c2_campaign/health.sh
Per-check commands:
  ssh picasso 'sacct -X -n -P -o State -j <ids> | sort | uniq -c'
  ssh picasso 'find $FSCRATCH/results/isalsr/c2_3arm -name run_log.json | wc -l'
  ssh picasso 'grep -h "^FAILED:" $FSCRATCH/execs/isalsr/c2_3arm/logs_recovery/*.out | wc -l'
  ssh picasso 'grep -l "\[FATAL\] copy-back" .../logs_recovery/*.out | wc -l'
NOTE: use `squeue`, never `squeue -u` (the Lua wrapper rejects it).

Baseline as of 2026-08-13 09:37: 12,378/12,600 cells, 221 COMPLETED /
104 RUNNING, 0 failures, 0 stranded copy-backs, 0 deferrals.

════════════════════ DO THIS WHEN THE RECOVERY DRAINS ════════════════════
STEP 1 - CENSUS (a real gate; do not skip)
  ssh picasso 'D=$FSCRATCH/repos/IsalSR_recovery; PY=$FSCRATCH/conda_envs/isalsr/bin/python; cd $D && \
    PYTHONPATH=$D/src:$D $PY $D/experiments/scripts/c2_missing_cells.py \
    --results-dir $FSCRATCH/results/isalsr/c2_3arm --seeds 1-30 --strict --summary'
  MUST report 12600 expected / 12600 present / 0 missing.
  If gaps remain: re-scope with --selectors and re-run submit_recovery.sh
  (below) before doing ANYTHING else. Do not copy or aggregate a partial tree.

STEP 2 - INCREMENTAL COPY (then VERIFY, do not trust exit 0)
  source ~/.bash_aliases
  parallel-ssh-copy picasso:$FSCRATCH/results/isalsr/c2_3arm \
    /media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/review/c2_3arm 8
  It copies CONTENTS into the destination, so the dst must end in /c2_3arm.
  It uses --ignore-existing, so only new cells transfer (cheap).
  VERIFY BOTH SIDES - file count, run_log.json count, and SUM OF REGULAR-FILE
  BYTES (find -type f -printf '%s\n' | awk '{s+=$1} END {print s}').
  The byte check is the one that matters: --partial --inplace --ignore-existing
  would silently keep and then skip a truncated file. Compare file bytes, not
  `du -sb` (directory inode sizes differ between GPFS and ext4 and are noise).

STEP 3 and STEP 4 RUN IN PARALLEL.

STEP 3 - ANALYSIS AGENT on the complete corpus
  Spawn a read-only subagent against the LOCAL copy ONLY (never Picasso).
  It must write a brief to a file; you read ONLY the brief, never the raw data.
  Tell it: rho by arm, CPDT, solution recovery, arm balance, complexity
  telemetry (T19), canonicalisation overhead. REQUIRE robust/clipped statistics
  (see TRAPS). Forbid it from modifying anything.

STEP 4 - RE-RUN AGGREGATION + LEDGER on the full corpus
  Per section 10 of .claude/notes/review/tasks/C2-deferred-cells-recovery-plan.md.
  Submit c2r_aggregate (42-task array, one per config) then c2r_ledger
  (--dependency=afterany on it), with ISALSR_REPO_DIR = the DEPLOYED tree.
  🔴 PASS C2_EXPECTED_TASKS=12600 EXPLICITLY. See TRAPS.
  Expect the certifier to report GO, n_blocking_failures=0, C1.15
  expected_set_source="registry", expected == observed == 12600.

STEP 5 - CLOSE OUT
  Copy the new root-level artifacts back (aggregate.csv, paired stats,
  status_ledger.csv, c2_preflight/). Reconcile the analysis brief against the
  certifier. Update EXECUTION-PLAN sections 11.1/11.3 and the recovery plan with
  final numbers. Then stop the loop and report.

════════════════════ HARD SAFETY RULES ════════════════════
1. NEVER modify $FSCRATCH/repos/IsalSR (the deployed tree) and NEVER run
   slurm/c2_smoke/deploy.sh while any job runs. Compute nodes read it; changing
   it splits provenance across two HEADs ("defect 10"; it once cost 161 cells).
   All cells must report git_describe = "campaign/c2".
2. NEVER hand-write into the campaign root. Only jobs write there.
3. The local copy is the ONLY backup of ~80,000 core-hours. Never delete it.
4. Do not scancel anything without asking the user first.
5. fscratch: SPACE is fine (0.47/1.40 TB). INODES are the constraint
   (~231k/250k soft). Do NOT delete build_gedlib/ - it belongs to another
   project and the user has said so explicitly. conda_pkgs/ (~23k inodes) is
   the safe reclaim if ever needed.

════════════════════ TRAPS ALREADY PAID FOR ════════════════════
These cost real debugging. Do not rediscover them.

A. C2_EXPECTED_TASKS is a CELL count (12600), not a task count. The deployed
   launcher.sh:625 still passes the task count; the fix is committed locally but
   deliberately NOT deployed. If you omit it, c2_certify falls back to a
   self-referential "disk" universe and reports GO on an unverified tree.
B. c2_missing_cells.py --strict was INVERTED (checked top-level
   method/variant/problem/seed; the real schema nests them under metadata and
   the arm is "representation"). Fixed in commit and synced to
   IsalSR_recovery ONLY. If you use another checkout, verify --strict agrees
   with plain presence before trusting it.
C. Job-name collision: the smoke and campaign profiles build the SAME c2s_*
   names. submit_paced.sh dedups by name over `sacct -S today` scoped by
   C2_MIN_JOBID (default 0), so a same-day Stage C makes it a silent no-op.
   Recovery uses the c2r_ prefix and is unaffected.
D. R2 OUTLIER CONTAMINATION. udfs/strogatz/strogatz_lv1/seed_24/isalsr has
   r2_test = -423.31 and single-handedly shifts arm means by -0.216; the same
   effect runs the OTHER way on bingo/hard (one cell at -319.34). r2_train is
   identical across arms. ALWAYS use median/clipped/robust statistics. Raw means
   on this corpus are misleading in both directions.
E. cache_hit_rate / cache_hits / cache_misses are identically ZERO on every
   cell - a dead field. Do not report it.
F. Bingo total_dags_explored is NOT comparable baseline-vs-dedup (counter
   semantics differ, ~10.8x apparent gap). rho and hash-vs-isalsr ARE fine.
G. udfs on feynman_remainder saturates its 12h cap with ~0% solution recovery
   on all 6 problems (median r2_test 0.55-0.63). Known, not a defect - but it
   means the D2 Feynman extension yields little UDFS signal. Bingo handles them
   (r2 = 1.0) yet recovers almost nothing symbolically (0.6-4.6%).
H. bingo/nguyen/nguyen_8/isalsr was an entirely empty block (all 30 seeds
   deferred). Recovery covers it. Confirm it is populated at census.

════════════════════ REFERENCE DOCS ════════════════════
.claude/notes/review/tasks/C2-deferred-cells-recovery-plan.md   (sections 9-12: what was
   built, the workflow, validation, conclusions)
.claude/notes/review/tasks/C2-results-verification-brief.md     (mid-campaign verification)
.claude/notes/review/tasks/EXECUTION-PLAN.md                    (section 5.5 completeness rule,
   section 11.1 anomaly ledger, section 11.3 launch ledger with all job ids)
slurm/c2_campaign/README.md, CAMPAIGN_BRIEF.md, SUBMIT_NOW.md

════════════════════ LOOP BEHAVIOUR ════════════════════
Arm ONE persistent Monitor on the recovery job ids that fires on drain or on any
FAILED/OUT_OF_MEMORY/TIMEOUT/NODE_FAIL state, plus a ~1h fallback ScheduleWakeup
(sessions restart often and kill monitors; re-arm after a restart, but never arm
two monitors for the same gate or a duplicate drain event can trigger a
concurrent double copy). While nothing is actionable, report one short line
(cells, states, failures) and reschedule. Report failures as plainly as
successes; a false all-clear here costs a re-run of 12,600 jobs.
```
