# T19 probe — how to run it

Answers one question: **does the explored-DAG structural telemetry populate
correctly, on Picasso, for every `(method, arm)`, without disturbing anything
the campaign already records?** It produces no number for the paper.

SP-0 caps (EXECUTION-PLAN §4.0) are enforced by the launcher, which refuses to
submit otherwise: 24 tasks (cap 60), `max_time = 900 s` (cap 1800), seeds 0 and
101 only (never 1…30), output under `~/execs/isalsr/t19_probe/` (never a
campaign root).

---

## The four steps, in order

### 1. Local pre-flight — seconds, catches most of what would waste a wave

```bash
cd ~/research/code/IsalSR
PYTHONPATH=src:. python slurm/t19_probe/local_smoke.py /tmp/t19_smoke 40
# expect: 6 run_log.json, 6 complexity.json, PRE-FLIGHT OK
```

Runs all six `(method, arm)` cells on Nguyen-1 with a 40 s budget and prints the
telemetry beside the pre-T19 fields. It exists to catch, in under five minutes,
the failures a Picasso probe would otherwise surface forty minutes and 24
allocations later — a runner that was never wired, an import the formatter
stripped as unused, a schema field that does not reach the run log. **A cell
reporting zero samples is the specific failure it hunts**, because in the run
log that is indistinguishable from an instrument that never fired.

### 2. Deploy

`deploy.sh` refuses a dirty tree and is right to. If another workstream has
uncommitted files, deploy from a clean worktree at your own commit and exclude
theirs:

```bash
git worktree add --detach /tmp/t19-clean HEAD

rsync -az --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
      --exclude 'build/' --exclude '.pytest_cache' --exclude 'docs/generated' \
      /tmp/t19-clean/ \
      picasso:/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR/
```

**Deploy into the campaign checkout, not a sibling directory.** The editable
install is scikit-build-core's, and its finder pins `isalsr.__path__[0]` to
whichever tree it was installed from — so a new module in a side-by-side
checkout is invisible however you set `PYTHONPATH`, and the run dies with
`ModuleNotFoundError: No module named 'isalsr.core.complexity'`.

Then assert the deployed code is the code you wrote:

```bash
ssh picasso 'cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR && \
  eval "$(conda shell.bash hook)" && conda activate isalsr && \
  export PYTHONPATH=$PWD/src:$PWD && python -c "
import hashlib, isalsr.core.complexity as cx
from isalsr.core import backends
print(cx.__file__); print(backends.engine(), backends.build_info()[\"build_hash\"])
print(hashlib.sha256(open(cx.__file__,\"rb\").read()).hexdigest()[:16])"'

sha256sum src/isalsr/core/complexity.py | cut -c1-16   # must match
```

The SHA-256 is the load-bearing provenance record, not `git rev-parse HEAD`:
the sources are rsynced on top of a checkout whose `HEAD` does not contain them,
so the commit names code that is not running. The worker prints both, plus the
dirty-path count. **A probe may do this; C2 may not** — the campaign must deploy
from a clean checkout at a tag.

### 3. Submit — dry-run, then test-only, then for real

```bash
ssh picasso 'cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR && \
             bash slurm/t19_probe/launcher.sh --dry-run'
ssh picasso '... && bash slurm/t19_probe/launcher.sh --test-only'   # must accept 24 tasks
ssh picasso '... && bash slurm/t19_probe/launcher.sh --one'         # optional: 1 task first
ssh picasso '... && bash slurm/t19_probe/launcher.sh'               # the 24-task array
```

`--test-only` catches an unsatisfiable request in one second; a live submission
with a bad constraint just sits PENDING, which is indistinguishable from queue
pressure. Resources: 1 core, 16 G, 40 min wallclock, `--constraint=sr` — pinned
because `complexity_time_s` is a reported quantity and an unpinned Intel/AMD
pool would turn it into a measurement of the scheduler.

Expect ~35 minutes: two waves of 12 at a 900 s payload.

```bash
ssh picasso "sacct -j <JOBID> -X -n -P -o JobID,State | awk -F'|' '{print \$2}' | sort | uniq -c"
```

### 4. Fetch and verify — the gate

```bash
rsync -az picasso:/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/t19_probe/results/ /tmp/t19probe/
PYTHONPATH=src:. python slurm/t19_probe/verify.py /tmp/t19probe
# expect: VERDICT: GO -- 14/14 gates passed   (exit 0)
```

Half the fourteen gates check the **pre-T19** fields — a probe proving the new
block works while the campaign's existing record regressed would be worth
nothing. The three that carry the most weight:

| gate | what it protects |
|---|---|
| **G4 / G5** | every pre-existing field still present and typed, 86 fields × 24 cells against the frozen `RUN_LOG_FIELD_SPEC` |
| **G9** | the sampling rule is **identical across the three arms of a method** — without this, an arm-versus-arm contrast measures the instrument, not the search |
| **G14** | SP-4 on the probe's own candidate stream: 0 `SUB`, 0 `DIV` over every sampled node, i.e. the decomposed alphabet is really what was measured |

G12 reports overhead only over **budget-bound** cells. A run that converges in
three seconds still pays the fixed generation-0 population sample, so its
cost/wall ratio measures how fast Nguyen-1 is solved rather than the campaign
regime.

---

## Last run

Array `1814948`, 2026-08-07 — 24/24 COMPLETED, **14/14 GO**, engine `cpp`,
`build_hash 298fc1188bf1b051` unchanged. Overhead: UDFS 0.001–0.003 %, Bingo
0.69 % on its longest cell. Full write-up, including the descriptive signal and
its caveats: `.claude/notes/review/tasks/T19-dag-complexity-telemetry.md`.
