# Stage D submission runbook

**Nothing in this file has been executed.** Stage D is submitted only after
Mario merges `feature/experiment-fairness-audit` into `cpp-core-port` and gives
an explicit go (audit.md §7 row 1). SP-0 stands: no agent submits C2 or
anything resembling it.

Read `EXECUTION-PLAN.md` §4.0 (SP-0…SP-7) before step 1. Every step below that
touches Picasso ends by recording the six-row SP-1…SP-6 table in the work log;
an entry without it is not evidence.

---

## Step 0 — Preconditions

Do not start until all four hold.

| # | Precondition | How to check |
|---|---|---|
| 0.1 | The audit branch is merged into `cpp-core-port` | `git log --oneline cpp-core-port \| head`, and `git branch --merged cpp-core-port \| grep experiment-fairness-audit` |
| 0.2 | Mario has said go, in writing, in this session | not inferable from any tool output |
| 0.3 | The A2 gate is green **on the merged commit**, not on either parent | re-run the gate; a green result on the pre-merge commit does not transfer |
| 0.4 | Local tree clean, and it is the merged commit | `git status --porcelain` empty; `git rev-parse HEAD` matches 0.1 |

Also re-read the live quota — from the day, not from an earlier capture
(EXECUTION-PLAN §4.6 item 6):

```bash
ssh picasso 'quota'
```

Stage D writes 12 cells: ≈120 files against the FSCRATCH headroom last measured
at 94.6k inodes. The D2 trace is the only large artefact; its projected size is
in `c2_trace/stream_size.md` once step 5 completes.

---

## Step 1 — Deploy the merged commit

`deploy.sh` refuses a dirty local tree, rsyncs **including `.git`**, then
verifies from the remote side that HEAD matches and the tree is clean before
rebuilding. That remote-side verification is the point: the historical sync
excluded `.git`, so SP-1 reported a stale hash with `-dirty` permanently and
could never have passed (§11.1, 2026-08-03).

```bash
cd <repo root on the workstation>
bash slurm/c2_smoke/deploy.sh
```

Then confirm SP-1 and SP-2 **from the remote**, not from the local tree:

```bash
ssh picasso 'cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR \
  && git rev-parse HEAD && git status --porcelain | wc -l'

ssh picasso 'eval "$(conda shell.bash hook)" && conda activate isalsr \
  && python -c "import isalsr; from isalsr.core import _native, backends; \
     import os, datetime; print(isalsr.__file__); print(_native.__file__); \
     print(datetime.datetime.fromtimestamp(os.path.getmtime(_native.__file__))); \
     print(backends.engine(), backends.build_info())"'
```

**SP-2 assertion:** the `.so` mtime must post-date the last commit touching
`src/isalsr/core/**`. The editable install puts the `.so` under
`site-packages/isalsr/core/`, **not** in the repo tree, so a repo-local `find`
will not reveal a stale build. Rebuild only with
`pip install -e . --force-reinstall --no-deps` — never `--no-build-isolation`,
which aborts with `BackendUnavailable` while the stale `.so` keeps loading, and
never read pip's status through a pipe.

---

## Step 2 — Re-run Stage C on the merged commit

**This step is not optional and it is not a formality.** `c2_smoke_v3` was run
from `cpp-core-port` before this branch merged, so its run logs carry **58 of
the 60** spec fields — `conversion_time_s` and `shadow_time_s` are absent
(`c2_preflight/smoke_vs_C1.md` §3.7). C1.2 on the merged commit expects 60.
Stage D's D1.7 also depends on the corrected accounting, which pre-merge code
understates 1.6–2.4×.

Write to a **new** root so `c2_smoke_v3` is preserved as the C5 evidence:

```bash
ssh picasso
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR

export C2_RESULTS_DIR=/mnt/home/users/tic_163_uma/mpascual/fscratch/results/isalsr/c2_smoke_v4
export C2_LOGS_DIR=$HOME/execs/isalsr/c2_smoke_v4/logs

bash slurm/c2_smoke/launcher.sh --dry-run       # inspect the 42 arrays first
bash slurm/c2_smoke/launcher.sh                 # 1,260 tasks, ~33 min at %8/sr
```

Monitor per step 6. When the arrays drain, the dependent aggregation job runs
`--postprocess only` and then `c2_certify`. Read the verdict:

```bash
ssh picasso 'python -c "import json,sys; \
  d=json.load(open(\"'"$C2_RESULTS_DIR"'/c2_preflight/stage_c_certification.json\")); \
  print(d[\"verdict\"], d[\"n_blocking_failures\"]); \
  [print(k, v[\"status\"]) for k,v in d.items() if isinstance(v,dict) and v.get(\"status\")!=\"PASS\"]"'
```

**Gate: proceed only on `GO`.** Confirm specifically that C1.2 now reports all
60 fields present — that is the field this re-run exists to establish.

Then obtain Mario's signature on `c2_preflight/smoke_vs_C1.md` (C5). Note that
§3 was drafted against `c2_smoke_v3`; if the v4 numbers move materially, §3 is
re-drafted before signature. §2 is frozen either way.

---

## Step 3 — `sbatch --test-only` on all three Stage D groups

Validates the resource request against the live partition without queueing
anything. Catches an unsatisfiable `--mem`/`--constraint` pair — and 256 GB on
`sr` is exactly the combination worth checking, since `sd` cannot host one at
all.

```bash
ssh picasso
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR
bash slurm/c2_stage_d/launcher.sh --dry-run     # prints the sbatch lines
bash slurm/c2_stage_d/launcher.sh --test-only   # SLURM validates, submits nothing
```

Expect three arrays: `udfs` 1-3 @16G, `bingo_std` 1-6 @32G,
`bingo_isalsr` 1-3 @256G, all `--constraint=sr`, wall `0-16:00:00`.
Cross-check against the registry, which is the only source of truth:

```bash
python -m experiments.scripts.stage_d_task_spec --list
```

---

## Step 4 — One short probe of the Stage D worker

Validates the two things Stage D adds that Stage C never exercised: the RSS
sampler and the D2 trace flag. Runs under SP-0 caps — `max_time ≤ 1800 s`,
seed 0, `~/execs` output — enforced by the worker itself, which refuses a
larger budget or a campaign root in probe mode.

Probe the **trace cell** (`bingo_isalsr` index 1), because it is the only cell
whose extra persistence needs proving:

```bash
ssh picasso
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR

sbatch --job-name=c2d_probe --account=tic_163_uma \
       --array=1-1 --time=0-00:40:00 --cpus-per-task=1 \
       --mem=32G --constraint=sr \
       --output=$HOME/execs/isalsr/c2d_probe/probe_%A_%a.out \
       --error=$HOME/execs/isalsr/c2d_probe/probe_%A_%a.err \
       --export="ALL,ISALSR_REPO_DIR=$PWD,D_GROUP=bingo_isalsr,\
D_RESULTS_DIR=$HOME/execs/isalsr/c2d_probe,D_PROBE_MAX_TIME=900,D_RSS_INTERVAL=15" \
       slurm/c2_stage_d/worker.sh
```

`D_RSS_INTERVAL=15` rather than 60 so a 900 s probe still yields ~60 rows.
Production uses 60.

Four things must hold in the probe output:

| # | Check | Where |
|---|---|---|
| 4.1 | The worker echoed `[WARN] SP-0 PROBE MODE` and `Cell 10/12` | `.out` header |
| 4.2 | `rss_timeseries.csv` exists, header `timestamp_s,vmrss_kb,vmhwm_kb`, ≥30 rows, `vmhwm_kb` monotone non-decreasing | `<seed dir>/rss_timeseries.csv` |
| 4.3 | `[D2] detailed trace ENABLED` and all five artefacts present | `<seed dir>/c2_trace/` |
| 4.4 | The spot check is 20/20 byte-exact and reports two distinct engines | `c2_trace/spot_check.json` |

Also confirm the trace did **not** fire anywhere else: probe `bingo_std`
index 1 the same way and assert no `c2_trace/` directory appears.

---

## Step 5 — Submit Stage D

Only after steps 0–4 are all green.

```bash
ssh picasso
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR
bash slurm/c2_stage_d/launcher.sh
```

Submits the three arrays plus one `afterany` aggregation + certification job,
and writes `job_ids.txt` and `certify_job_id.txt` under the logs dir. The
launcher verifies the dependency actually stuck (`Dependency=(null)` means
SLURM dropped it and the job would run against an empty tree) and cancels the
job rather than letting it produce a report on nothing.

`afterany`, not `afterok`, is deliberate: a cell that OOMs must still be
certified, because observing that is D1.2's entire purpose.

Expect ≈12–16 h wall. The 256 GB group is the schedule risk: `sr` hosts one
such task per node.

---

## Step 6 — Monitoring

```bash
squeue -u $USER                      # bare, no format string
squeue -u $USER -t PENDING -o "%.18i %.9P %.20j %.8T %.10M %R"
```

Memory and elapsed, once tasks finish. **Both traps below have already cost
this project a silent wrong answer** (§11.1, 2026-08-02):

```bash
# CORRECT: JobIDRaw, and the .batch step
sacct -j "$(paste -sd, ~/execs/isalsr/c2_stage_d/logs/job_ids.txt)" \
      -n -P -o JobIDRaw,MaxRSS,Elapsed,State \
  | awk -F'|' '$1 ~ /\.batch$/'
```

- **Never `sacct -X`.** It returns an **empty** `MaxRSS` — memory is accounted
  on the `.batch` step. The profile comes back silently blank: no error, just
  empty cells, and D1.2 is the whole reason Stage D runs.
- **Never join on `JobID`.** For an array it reads `<array_id>_<task>`, while
  `status.json` records the raw numeric id. Joining on `JobID` matched 42 of
  1,260 rows and still reported PASS.

Watch the live RSS curve without waiting for `sacct`:

```bash
tail -5 <results>/bingo/hard/pagie_1/isalsr/seed_101/rss_timeseries.csv
```

`vmhwm_kb` is monotone, so the last row is the true peak so far regardless of
sampling rate.

---

## Step 7 — Read the certification, then Stage F

```bash
cat <results>/c2_preflight/stage_d_certification.md
python -c "import json; d=json.load(open('<results>/c2_preflight/stage_d_certification.json')); \
  print(d['verdict'], d['n_blocking_failures'])"
```

Then, in order:

1. **D1.2's production `--mem` recommendation** per (method, arm) is the number
   Stage F item 4 requires. It is derived from measurement — the max of sacct
   `MaxRSS` and the `rss_timeseries` `VmHWM`, plus ≥30 % headroom — not from
   history. If Bingo–IsalSR's 12 h peak comes in comfortably under 128 GB, §3.3
   permits revising the 256 GB request **down**, with the measurement recorded.
2. **D1.6** settles the ρ direction question that C5 §3.5 handed forward. It is
   12 h against 12 h; C5's comparison was 900 s against 12 h and could not.
3. **D1.7** will report a Bingo overhead **above** the old canon-only ≈7.4 %
   projection. That is the accounting change (conversion is now counted, shadow
   is separated), **not** a regression. Read it as such.
4. **D3** (`stage_d_mode1_replay.py`) on the trace stream. A hash-soundness
   violation — two DAGs sharing a fixed-order digest but not a canonical string
   — is an unsound merge and kills the arm. It exits non-zero by design.
5. **C5 signature** on `c2_preflight/smoke_vs_C1.md`, if not already obtained
   at step 2.
6. **Stage F** (EXECUTION-PLAN §4.6): §11.2 filled A1–E7, the achieved
   concurrency and projected completion date, the production `--mem`/`--time`,
   the B6/B6b node decisions, live quota, and Mario's signature.

**No agent submits C2.**
