# RUNBOOK: SRBench black-box campaign (R3.2)

1,200 cells = 20 datasets x 10 seeds x 3 arms (baseline / hash / isalsr) x 2 hosts
(UDFS, Bingo). The hosts, arms, 12 h budget and canonicaliser are exactly C2's.
The datasets are SRBench v2.0's black-box track, real-world only, with d ≤ 5:
`benchmarks/datasets/data/srbench_blackbox/PROVENANCE.md`.

Every command below is meant to be copied and pasted. `W$` runs on the workstation, from
`<repo>/code` on the merged, **committed** branch. `P$` runs on the Picasso **login** node.

```
export FSCRATCH=/mnt/home/users/tic_163_uma/mpascual/fscratch   # exported: the smoke check reads it from Python
C2=$FSCRATCH/repos/IsalSR          # C2 tree; its editable install provides isalsr + engine. NEVER modified.
BBX=$FSCRATCH/repos/IsalSR_bbx     # this campaign's experiments/ benchmarks/ slurm/ (no src/)
RES=$FSCRATCH/results/isalsr/srbench_blackbox
```

## 0. Why the deployment looks like this (read once)

- The conda env `isalsr` on Picasso has an **editable** install of the C2 tree
  (`pip install -e .` in `$C2`, `slurm/c2_smoke/deploy.sh`). Through `pyproject`
  `wheel.packages = ["src/isalsr", "experiments", "benchmarks"]`, that install
  redirects **all three** packages to `$C2`, ahead of `PYTHONPATH` and the cwd. A
  separate tree run "from its cwd" would therefore execute C2's orchestrator,
  which has no `srbench_blackbox` suite.
- `slurm/blackbox/pyboot/sitecustomize.py` (on `PYTHONPATH`, with
  `ISALSR_BBX_ROOT=$BBX`) pins `experiments` and `benchmarks` to `$BBX`. It
  leaves `isalsr` and the C++ engine on the C2 install. No rebuild, no
  reinstall: the canonicaliser is C2's by construction, and the gate checks it.
- Each task runs `slurm/blackbox/worker.sh`. It runs the gate (`check_env.py`:
  three module paths, build hash, deployed SHA-256s, printed into the task log),
  then `exec`s **C2's own worker** `slurm/c2_smoke/worker.sh` (byte-identical to
  tag `campaign/c2`). That worker provides local scratch, per-cell copy-back,
  the deadline rule, `--ledger --postprocess skip` and `PYTHONMALLOC=malloc`.
- The C2 tree's `src/isalsr` must be byte-identical to tag `campaign/c2`
  (`2dd56fd`). Pre-flight P1 checks this against the working tree, so an
  uncommitted edit fails it too.

## 1. Deploy (workstation)

```bash
W$ python -m pytest tests/unit -q -x        # must pass on the merged branch
W$ git status --porcelain                   # must be empty (deploy.sh refuses otherwise)
W$ bash slurm/blackbox/deploy.sh            # rsync experiments/ benchmarks/ slurm/ -> $BBX, writes BBX_DEPLOY.json
```

## 2. Pre-flight (login node; submits nothing)

```bash
P$ cd $BBX && bash slurm/blackbox/preflight.sh
```
It must end with `PRE-FLIGHT PASS`. It checks:
- **P1.** `$C2/src/isalsr` is identical to `campaign/c2`. It also prints C2 HEAD and status.
- **P2.** `isalsr.__file__` lies under `$C2/src/isalsr`. `experiments.models.orchestrator` and
  `benchmarks.datasets.srbench_blackbox` lie under `$BBX`. The engine is `cpp` with
  `build_hash == 298fc1188bf1b051`. The deployed files match `BBX_DEPLOY.json`.
- **P3.** The 20 datasets load (SHA-256 vs manifest) and split for seeds 1-10.
- **P4.** `launcher.sh --dry-run` covers 1,200 cells.

Spot-check by hand, and paste the output into the campaign log:
```bash
P$ cd /tmp && conda activate isalsr && \
   PYTHONPATH=$BBX/slurm/blackbox/pyboot ISALSR_BBX_ROOT=$BBX \
   python -c "import isalsr, experiments.models.orchestrator as o, benchmarks.datasets.srbench_blackbox as b; \
              from isalsr.core import backends; print(isalsr.__file__); print(o.__file__); print(b.__file__); \
              print(backends.build_info()['build_hash'])"
```
Expected output: `$C2/src/isalsr/__init__.py`, then `$BBX/experiments/models/orchestrator.py`, then
`$BBX/benchmarks/datasets/srbench_blackbox.py`, then `298fc1188bf1b051`.

If P1 fails because `$C2` is not at the tag, **do not touch `$C2`**. Record
`git -C $C2 log -1 --oneline`, `git -C $C2 status --short` and
`git -C $C2 diff --stat 2dd56fd -- src/isalsr`, and stop: the campaign must not run
on a different canonicaliser.

## 3. Smoke array (6 cells, about 30 min including queue)

```bash
P$ cd $BBX && bash slurm/blackbox/launcher.sh --smoke
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
- Every task log shows the same three `BBX ..._file` paths (C2 / bbx / bbx), `build_hash=298fc1188bf1b051` and
  `BBX gate: PASS`.
- 6 `PASS` lines.
- The provenance stamp exists.

The local smoke passed 18/18 on these criteria (485_analcatdata_vehicle, banana and titanic x 3 arms x 2 hosts,
seed 1, 60 s). A local end-to-end run of `worker.sh` on a scratch copy of the deploy layout passed the gate, ran
one cell, deferred one by the deadline rule and verified its copy-back. See the T01 log §5 and §3.9.

## 4. Full submission

```bash
P$ cd $BBX && bash slurm/blackbox/launcher.sh --dry-run     # inspect: 1,200 cells
P$ cd $BBX && bash slurm/blackbox/launcher.sh               # submits; writes job_ids.txt (+ sweep_job_ids.txt)
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
P$ cd /tmp && conda activate isalsr && PYTHONPATH=$BBX/slurm/blackbox/pyboot ISALSR_BBX_ROOT=$BBX \
   python -m experiments.models.orchestrator --postprocess ledger --output-dir $RES   # status_ledger.csv
W$ mkdir -p /media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/review/srbench_blackbox
W$ rsync -az --info=progress2 picasso:$RES/ /media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/review/srbench_blackbox/
W$ rsync -az picasso:$FSCRATCH/execs/isalsr/srbench_blackbox/logs/ /media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/review/srbench_blackbox/_slurm_logs/
```
The aggregation (`--postprocess only` per config) and the appendix tables belong to T05. They run locally on the
synced tree. PLAN D7's fallback applies at the 14 Oct cut-off: a dataset enters a host's analysis only if all
3 arms x 10 seeds completed.
