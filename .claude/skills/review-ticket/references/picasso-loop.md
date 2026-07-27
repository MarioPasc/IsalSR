# Picasso loop — commands, paths, failure signatures

Detail for §5 of `SKILL.md`. Read this when a ticket needs cluster compute
(T02, T03 Phase 4, T04, T05).

**The `picasso-sbatch` skill is the authority on SLURM directives.** Invoke it
before writing or editing any launcher or worker. Everything below is about the
*loop around* those scripts — smoke, sync, staged submission, monitoring — not
about their contents.

---

## Facts

| Item | Value |
|---|---|
| SSH alias | `picasso` → `picasso3.scbi.uma.es`, user `mpascual` |
| Repo path on cluster | `/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR` |
| Datasets | `/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/` |
| Logs | `~/execs/isalsr/logs/` |
| Local conda env | `~/.conda/envs/isalsr/bin/python` |
| Local results root | `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/` |
| Job profile | **CPU-only.** 1 core/run, 8–16 GB (Bingo+IsalSR historically 128 GB), 15–17 h wallclock, `max_time = 43,200 s` |
| Containers | Singularity `.sif` only — no Docker |

Never request `--gres` for IsalSR work. If a script in this repo does, it is a bug.

Existing launchers to match in shape (`slurm/`): `models_launch.sh`,
`hard_launch.sh`, `cherrypicked_launch.sh`, `roundoff_launch.sh`. All expose
`--dry-run` and `--experiment <group>`.

---

## Stage 1 — local smoke (hard gate, ≤ 10 min)

```bash
~/.conda/envs/isalsr/bin/python -m experiments.models.orchestrator \
    --config experiments/configs/<cfg>.yaml \
    --seeds 1 --problems Nguyen-1
```

Use a smoke config with `max_time: 120`. Never smoke against a production config —
you will wait 12 hours.

**Validate the output, not the exit code:**

```bash
python - <<'PY'
import json, sys, pathlib
p = sorted(pathlib.Path("<results_dir>").rglob("run_log.json"))[-1]
d = json.loads(p.read_text())
need = ["problem", "seed", "variant", "method", "metrics"]
missing = [k for k in need if k not in d]
print("PATH", p); print("MISSING", missing)
print("dedup", d.get("metrics", {}).get("n_duplicates_eliminated"))
PY
```

A `run_log.json` that exists but is truncated is the exact failure mode the
orchestrator's resume validation was hardened against (`CLAUDE.md`, Operational
requirements). Parse it.

For an IsalSR variant, `n_duplicates_eliminated` must be **non-zero**. Zero means
the dedup hook is not firing and the whole campaign would be a null result. For
UDFS specifically, confirm `processes: 1` in the config — spawned workers re-import
modules and bypass the monkey-patch.

---

## Stage 2 — sync

```bash
rsync -avz --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.egg-info' \
  --exclude 'results' --exclude '.hypothesis' --exclude 'build' --exclude '.ruff_cache' \
  ./ picasso:/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR/
```

`--delete` matters: a stale module left behind will shadow the new one and the
failure looks like a logic bug.

Verify the import actually works on the cluster before submitting anything:

```bash
ssh picasso 'cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR && \
  source ~/.bashrc && conda activate isalsr && \
  python -c "import isalsr, isalsr.core.canonical as c; print(isalsr.__file__); \
             print(c.fast_canonical_string.__module__)"'
```

**If the ticket involves the C++ core (T01/T02/T03)**, also confirm which engine
loaded — a silent fallback to pure Python would make the whole campaign
meaningless:

```bash
ssh picasso '... python -c "from isalsr.core import canonical as c; print(getattr(c, \"_ENGINE\", \"unknown\"))"'
```

For long transfers use `tmux` on the local side.

---

## Stage 3 — staged submission

```bash
# 3a. syntax + resource validation, does not queue
ssh picasso 'cd <repo>/slurm && sbatch --test-only workers/<worker>.sh'

# 3b. ONE real task — the cluster smoke
ssh picasso 'cd <repo>/slurm && sbatch --array=1-1 workers/<worker>.sh'

# 3c. only after 3b is verified green
ssh picasso 'cd <repo>/slurm && sbatch --array=1-<N> workers/<worker>.sh'
```

Stage 3b is the one people skip and the one that pays. It catches: module-load
differences between login and compute nodes, missing dataset paths, permissions,
memory profile, and env activation — none of which `--test-only` sees.

Before 3c, sanity-check the arithmetic: `N` tasks × wallclock × cores. If the
product exceeds ~5,000 core-hours and the ticket does not already authorise it,
escalate to the human first (`SKILL.md` §7).

---

## Stage 4 — monitor for early errors

```bash
ssh picasso 'squeue -u mpascual -o "%.10i %.9P %.24j %.2t %.10M %.6D %R"'
ssh picasso 'ls -t ~/execs/isalsr/logs/*.err | head -5 | xargs tail -n 30'
ssh picasso 'sacct -u mpascual --starttime now-2hours \
   --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,ReqMem'
```

Run these from a **backgrounded** `until` loop or via `Monitor`. Never poll in the
foreground.

```bash
# background, fires once the condition is met
until ! ssh picasso 'squeue -h -u mpascual -j <JOBID>' | grep -q .; do sleep 300; done
```

### Failure signatures — escalate immediately on any of these

| Signature | Meaning | Action |
|---|---|---|
| `ModuleNotFoundError` | sync incomplete or env not activated | cancel array, fix, resync |
| `FileNotFoundError` on a dataset | path differs on cluster | cancel, fix path |
| `oom-kill` / `MaxRSS ≈ ReqMem` | under-requested memory | cancel, raise `--mem` |
| Task exits `< 1 min` | config or import failure, not a fast run | cancel, read `.err` |
| `CANCELLED` without a `scancel` | wallclock or scheduler limit | cancel, re-check limits |
| `State=FAILED` on ≥ 3 of the first 10 tasks | systematic, not sporadic | cancel the whole array |
| `n_duplicates_eliminated == 0` on an IsalSR arm | dedup hook not firing | cancel — the arm is a null result |

Cancel with `ssh picasso 'scancel <JOBID>'`. A 300-task array failing identically
costs 300 allocations and a day of queue time; killing at task 3 costs nothing.

---

## Stage 5 — retrieve and verify

```bash
rsync -avz picasso:/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR/results/<campaign>/ \
  /media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/<campaign>/
```

Then, before any analysis:

1. **Count the cells.** Expected = problems × seeds × methods × variants. A
   shortfall must be explained per-run, not averaged over. This is exactly the
   defect Reviewer 2 will catch — the submitted paper reported 1,465 Bingo cells
   where 1,500 were expected and never explained the gap (T08).
2. **Check `MANIFEST.json` is complete** (T02 §5.3): git commit, build hash,
   compiler flags, node CPU per run, engine, config hash, seed.
3. **Check for NaN** in every metric column before it reaches a table. NaN is a
   missing observation, never a winning value (T08 AC-3).
4. **Verify one campaign root per table.** The submitted version had five sibling
   result directories and no record of which produced which table; that ambiguity is
   the root cause behind four separate reviewer comments.

---

## Resume after a partial campaign

The orchestrator validates `run_log.json` *content* before skipping, and deletes
corrupt logs from OOM/timeout kills so they re-run. So a plain re-submit of the same
array resumes correctly — but confirm the count afterwards rather than assuming it.
