# RUNBOOK: SRBench black-box campaign (R3.2)

1,200 cells = 20 datasets x 10 seeds x 3 arms (baseline / hash / isalsr) x 2 hosts
(UDFS, Bingo). The hosts, arms, 12 h budget and canonicaliser are exactly C2's.
The datasets are SRBench v2.0's black-box track, real-world only, with d ≤ 5:
`benchmarks/datasets/data/srbench_blackbox/PROVENANCE.md`.

Every command below is meant to be copied and pasted. `W$` runs on the workstation, from
`<repo>/code` on the merged, **committed** branch. `P$` runs on the Picasso **login** node.

```
export FSCRATCH=/mnt/home/users/tic_163_uma/mpascual/fscratch   # exported: the smoke check reads it from Python
REPO=$FSCRATCH/repos/IsalSR        # the whole repository, with .git; the project is $REPO/code
ENV=$FSCRATCH/conda_envs/isalsr    # conda env; editable install of $REPO/code + the C2-built engine
RES=$FSCRATCH/results/isalsr/srbench_blackbox
```

## 0. Why the deployment looks like this (read once)

- C2's Picasso tree and env were deleted after the main campaign (found 5 Oct 2026), so this
  campaign **rebuilds both the way C2 was built** and proves the rebuild equivalent (T01b log):
  - `create_env.sh` creates `$ENV` with C2's interpreter build (`python=3.11.15=h17756b0_1`,
    the build every C2 `run_log` names) and the pip versions of `env_requirements.txt`, each
    tagged with its provenance: on record for C2, forced by PyPI release history, or (where C2's
    version cannot be recovered) a disclosed choice.
  - `deploy.sh` rsyncs the repository **with `.git`** to `$REPO`, verifies SP-1 from the remote
    side, and builds the extension with C2's recipe (`slurm/c2_smoke/deploy.sh`: GCC 13.2.0
    module, `pip install -e . --force-reinstall --no-deps` with build isolation, then
    `verify_build.py`). `build_hash` must be `298fc1188bf1b051`, C2's.
- One tree serves everything: the env's editable install maps `isalsr`, `experiments` and
  `benchmarks` to `$REPO/code` (pyproject `wheel.packages`), and the engine `.so` lives in the
  env's site-packages. The `pyboot/` import shim of T01's two-tree design is **not used**; it
  stays in the repository only as a tool for testing code in isolated git worktrees, and the
  gate fails if it is active.
- **P1** (arm identity): every file that can decide what an arm computes (all of `isalsr`
  except `viz`, `CMakeLists.txt`, `pyproject.toml`, all of `experiments/models`, the cell
  decoder, C2's worker and the two C2 configs the black-box configs copy) is byte-identical to
  tag `campaign/c2` (`2dd56fd`), apart from six listed differences, each with the reason it
  cannot change an arm (`code_identity.py`, `EXPECTED`). Checked on the workstation by
  `deploy.sh` (recorded in `BBX_DEPLOY.json`) and again on the deployed `.git` by `preflight.sh`.
- Each task runs `slurm/blackbox/worker.sh`. It runs the gate (`check_env.py`: three module
  paths, engine and build hash, deployed SHA-256s, the P1 verdict, C2's python and every pinned
  version, all printed into the task log), then `exec`s **C2's own worker**
  `slurm/c2_smoke/worker.sh` (byte-identical to tag `campaign/c2`). That worker provides local
  scratch, per-cell copy-back, the deadline rule, `--ledger --postprocess skip` and
  `PYTHONMALLOC=malloc`, and records SP-1 (HEAD) per task.

## 1. Environment (once) and deploy (workstation)

The env is created once, on the login node, from the deployed tree; it is never modified
afterwards (`create_env.sh` refuses an existing env). First deploy without building, then
create the env, then build:
```bash
W$ python -m pytest tests/unit -q -x               # must pass on the merged branch
W$ git status --porcelain                          # must be empty (deploy.sh refuses otherwise)
W$ bash slurm/blackbox/deploy.sh --no-build        # rsync + SP-1 + P1 + BBX_DEPLOY.json
W$ ssh picasso "bash -l /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR/code/slurm/blackbox/create_env.sh"   # once
W$ bash slurm/blackbox/deploy.sh                   # same, plus the C2-recipe build + verify_build.py
```
Every later deploy is the last line only. Never deploy while an array of this campaign runs
(C2 defect 10: a deploy is a code edit).

## 2. Pre-flight (login node; submits nothing)

```bash
P$ cd $REPO/code && bash slurm/blackbox/preflight.sh
```
It must end with `PRE-FLIGHT PASS`. It checks:
- **P1.** HEAD equals the deployed commit, the tree is clean, and `code_identity.py` passes on
  the deployed `.git`.
- **P2.** `isalsr.__file__` lies under `$REPO/code/src/isalsr`; `experiments.models.orchestrator`
  and `benchmarks.datasets.srbench_blackbox` under `$REPO/code`; `_native` in `$ENV`'s
  site-packages; engine `cpp` with `build_hash == 298fc1188bf1b051`; the deployed files match
  `BBX_DEPLOY.json`; python is C2's build; every pin of `env_requirements.txt` (and torch) holds.
- **P3.** The 20 datasets load (SHA-256 vs manifest) and split for seeds 1-10.
- **P4.** `launcher.sh --dry-run` covers 1,200 cells.
- **P5.** `verify_build.py`; the cpp-vs-python differential tests
  (`test_native_{build,canonical,s2d,datastructures}.py`, `test_equivalence_gate.py`) and the
  black-box protocol tests; C2's equivalence gate `experiments/scripts/equivalence_gate.py
  --gate all --backend-a python --backend-b cpp` on its full corpus (the Stage-B4 invocation).
  Reports go to `$REPO/code/build/bbx_preflight/`.

If P1 fails, **do not edit the deployed tree**: fix the cause locally, commit, redeploy.

## 3. Smoke array (6 cells, about 30 min including queue)

```bash
P$ cd $REPO/code && bash slurm/blackbox/launcher.sh --smoke
```
This submits 6 arrays x 1 task: one per (host, arm), on `banana` (n = 5300, the largest) at
seed 1, with `--max-time 120` and a 30 min wall. Results go to
`$FSCRATCH/results/isalsr/srbench_blackbox_smoke`.

### Smoke acceptance (all must hold)
```bash
P$ S=$FSCRATCH/results/isalsr/srbench_blackbox_smoke
P$ sacct -X -n -P -o JobName,State,ExitCode,Elapsed,MaxRSS -j $(paste -sd, $FSCRATCH/execs/isalsr/srbench_blackbox_smoke/logs/job_ids.txt)
P$ grep -h "^BBX " $FSCRATCH/execs/isalsr/srbench_blackbox_smoke/logs/bbx_*_smoke_*.out | sort | uniq -c
P$ conda activate isalsr && python - <<'PY'
import json, math, glob, os
S = os.path.expandvars("$FSCRATCH/results/isalsr/srbench_blackbox_smoke")
logs = sorted(glob.glob(f"{S}/*/srbench_blackbox/banana/*/seed_01/run_log.json"))
assert len(logs) == 6, len(logs)
for f in logs:
    d = os.path.dirname(f); r = json.load(open(f))["results"]; arm = f.split("/")[-3]
    reg, ss, t = r["regression"], r["search_space"], r["time"]
    ok = math.isfinite(reg["r2_test"]) and math.isfinite(reg["nrmse_test"])
    ok &= os.path.isfile(f"{d}/complexity.json")
    ok &= json.load(open(f"{d}/status.json"))["terminal_status"] == "completed"
    if arm != "baseline":
        ok &= ss["empirical_reduction_factor"] is not None and t["canonicalization_runtime_s"] > 0
        ok &= (ss["n_ledger_seen"] or 0) > 0
    print(f.split("/")[-6], arm, round(reg["r2_test"], 4), round(reg["nrmse_test"], 4), "PASS" if ok else "FAIL")
PY
P$ ls $S/bbx_provenance.json
```
Pass criteria:
- 6 rows `COMPLETED 0:0`.
- Every task log shows the same `BBX ..._file` paths (all under `$REPO/code`, `_native` under `$ENV`),
  `build_hash=298fc1188bf1b051` and `BBX gate: PASS`.
- 6 `PASS` lines.
- The provenance stamp exists.

The local smoke passed 18/18 on these criteria (485_analcatdata_vehicle, banana and titanic x 3 arms x 2 hosts,
seed 1, 60 s). A local end-to-end run of `worker.sh` on a scratch copy of the deploy layout passed the gate, ran
one cell, deferred one by the deadline rule and verified its copy-back. See the T01 log §5 and §3.9.

## 4. Full submission

```bash
P$ cd $REPO/code && bash slurm/blackbox/launcher.sh --dry-run     # inspect: 1,200 cells
P$ cd $REPO/code && bash slurm/blackbox/launcher.sh               # submits; writes job_ids.txt (+ sweep_job_ids.txt)
```
The launcher runs the gate on the login node first and writes `$RES/bbx_provenance.json`, which holds
the deploy record, the resolved module paths, the engine build info and the library versions. Packaging is set by
`slurm/blackbox/packaging.tsv`; the derivation is in that file's header:

| array | datasets | cells | B | tasks | `--time` | mem | throttle | est. cell (h) |
|---|---|---|---|---|---|---|---|---|
| `bbx_ub_all` | all 20 | 200 | 1 | 200 | 16:00:00 | 16G | %200 | 12.0 (always: never meets stop_thresh) |
| `bbx_uh_all` | all 20 | 200 | 1 | 200 | 16:00:00 | 16G | %200 | 12.0 |
| `bbx_ui_all` | all 20 | 200 | 1 | 200 | 16:00:00 | 16G | %200 | 12.0 |
| `bbx_bb_short` | 12 with est. < 2 h | 120 | 3 | 40 | 21:00:00 | 32G | %40 | 1.0-1.95, so 3.1-5.9 per task |
| `bbx_bb_long` | 1029, 1030, 529, 556, 557, 690, banana, titanic | 80 | 1 | 80 | 16:00:00 | 32G | %80 | 2.0-11.4 |
| `bbx_bh_all` | all 20 | 200 | 1 | 200 | 16:00:00 | 32G | %200 | 3.3-12 |
| `bbx_bi_all` | all 20 | 200 | 1 | 200 | 16:00:00 | 32G | %200 | 4.3-12 |
| **total** | | **1,200** | | **1,120** | | | | |

There is one sweep array (`bbxw_bb_short`, afterany on the 7 arrays) for the only row with B > 1.
Expected makespan once running is about 12.5 h, set by the 12 h UDFS cells. Every task is
allowed to run concurrently (1,120 at most, against C2's 2,016 slots), so queue wait dominates.
Expected compute is about 9,900 core-h: UDFS 600 x 12.1 h plus Bingo about 2,600. The upper bound,
with every cell at 12 h, is 14,400 core-h.

If `sbatch` refuses part-way ("Resource temporarily unavailable"), run `squeue -u $USER` **first**.
The launcher exits at the first refusal, so the arrays already submitted are listed above its error.
To submit only the rest, copy `packaging.tsv` to `/tmp/rest.tsv` and delete the rows already submitted. Then run
`BBX_PACKAGING=/tmp/rest.tsv BBX_EXPECTED_CELLS=<their cells> bash slurm/blackbox/launcher.sh`.
Sweep arrays from a partial submission depend only on the arrays submitted in that same call.

## 5. Health probe (one line, any time)

```bash
P$ squeue -u $USER -h -o '%j %T' | awk '$1 ~ /^bbx/' | sort | uniq -c; for m in udfs bingo; do for a in baseline hash isalsr; do printf '%s/%s %s/200\n' $m $a $(ls -d $RES/$m/srbench_blackbox/*/$a/seed_*/run_log.json 2>/dev/null | wc -l); done; done; grep -l '"terminal_status": "failed"' $RES/*/srbench_blackbox/*/*/seed_*/status.json 2>/dev/null | wc -l
```
It prints running and pending counts per array, completed run logs per (host, arm) out of 200, and the
number of failed cells, which must stay 0.

## 6. Completion and rsync back (workstation)

When `squeue` shows no `bbx*` job and the health probe reads 200/200 six times:
```bash
P$ cd /tmp && conda activate isalsr && \
   python -m experiments.models.orchestrator --postprocess ledger --output-dir $RES   # status_ledger.csv
W$ mkdir -p /media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/review/srbench_blackbox
W$ rsync -az --info=progress2 picasso:$RES/ /media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/review/srbench_blackbox/
W$ rsync -az picasso:$FSCRATCH/execs/isalsr/srbench_blackbox/logs/ /media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/review/srbench_blackbox/_slurm_logs/
```
The aggregation (`--postprocess only` per config) and the appendix tables belong to T05. They run locally on the
synced tree. PLAN D7's fallback applies at the 14 Oct cut-off: a dataset enters a host's analysis only if all
3 arms x 10 seeds completed.
