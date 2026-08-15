# Rebuilding the IsalSR environment on Picasso

**Date**: 2026-07-28 · **Found while**: running the T15 UDFS probe · **Affects**: T02 (Wave 1), T03–T05

Four defects blocked every attempt to run an IsalSR experiment on Picasso. Each was
found by a short single-task smoke, and each is invisible from the workstation. This
note records the verified recipe so the T02 campaign does not rediscover them.

---

## 1. What was wrong

| # | Symptom | Cause |
|---|---|---|
| 1 | `cc1plus: error: bad value ('x86-64-v3') for '-march=' switch` | `CMakeLists.txt:85` pins `-march=x86-64-v3`; Picasso's default `/usr/bin/g++` is **GCC 7.5.0**, and that arch value landed in GCC 11 |
| 2 | `module load` appears to succeed but `g++` stays at 7.5.0 | `module` is defined only in a **login shell**. `ssh picasso '…'` silently no-ops it |
| 3 | `ModuleNotFoundError: numpy`, then `statsmodels`, then `pkg_resources` | The env held **25 packages** — a `[dev,native]` env built for the T01 port. `dependencies = []` in `pyproject.toml` by design (core is stdlib-only), so `pip install -e .` does not repair it |
| 4 | A task that never ran reported `n_calls: 0, n_failures: 0` and exited **0** | The probe caught per-run exceptions, wrote a summary, and returned success; SLURM marked it `COMPLETED` |

Defect 4 is the dangerous one. Aggregated over an array it reads as "0 failures on
real data" — indistinguishable from the clean result the probe exists to establish.
`measure_const_normalization_arms.py` and `aggregate_norm_arms.py` now exit non-zero
when pooled `n_calls == 0` or any run errored.

---

## 2. The recipe

Every command runs in a **login shell** (`bash -lc`), on the login node, from
`$FSCRATCH`. GCC 13.2.0's libstdc++ is GLIBCXX_3.4.32, below the 3.4.34 the conda env
and the system both provide, so the built `.so` loads at runtime without the module.

```bash
ssh picasso 'bash -lc "
module load gcc/13.2.0
eval \$(conda shell.bash hook)
conda activate isalsr
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR

# 1. C++ extension. --no-build-isolation works here because the env already has
#    scikit-build-core and nanobind; without them, drop the flag so pip fetches them.
export CC=\$(command -v gcc) CXX=\$(command -v g++) CMAKE_BUILD_PARALLEL_LEVEL=4
rm -rf build
pip install -e . --no-build-isolation

# 2. Runtime dependencies. Install by name, NOT via -e \".[bench]\" — the extras form
#    rebuilds isalsr, and if the gcc module is not loaded that rebuild fails and pip
#    rolls the whole transaction back.
pip install numpy scipy sympy networkx h5py matplotlib pandas pyyaml \
            scikit-learn stopit zss tqdm bingo-nasa statsmodels scikit-posthocs

# 3. torch is REQUIRED: experiments/models/udfs/vendor/DAG_search/comp_graph.py:12
#    imports it at module scope. IsalSR is CPU-only, so take the CPU wheel (~200 MB
#    against ~2.5 GB for the CUDA build).
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. setuptools 81 removed pkg_resources, which the vendored UDFS search imports.
pip install \"setuptools<81\"
"'
```

Verify before submitting anything:

```bash
ssh picasso 'bash -lc "
eval \$(conda shell.bash hook); conda activate isalsr
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR
python -c \"
from isalsr.core import _native
import experiments.scripts.measure_const_normalization_arms
import pkg_resources, torch, statsmodels
assert hasattr(_native.testing, 'fast_canonical_string_raw')
print('preflight OK')
\""'
```

`critdd` is needed only by `experiments/figures/models/generate_critical_difference.py`
and is not on the runner path.

---

## 3. Picasso quirks that cost time

- **`squeue` is a Lua wrapper** that rejects standard flags: `-u`, `--noheader`, and
  `%.9P`-style format strings all error. A wait loop built on `squeue` falls through
  silently and reports a running job as finished. Use `sacct` for anything scripted;
  use `squeue --user=<name>` for a human-readable listing.
- **`sbatch --parsable` prepends ANSI codes and a multi-line warning**, so the job ID
  must be taken from the *last* line before stripping non-digits. The T15 launcher's
  `_clean_job_id` already does this.
- **UDFS overshoots its own `max_time` by roughly 12×** because it checks the clock
  only between order-enumeration stages: a 300 s budget ran 1 h 00 m wall. Size
  wallclock requests against the overshoot, not the budget.

---

## 4. Sequence that catches these

Each defect above surfaced at a different stage, and none would have surfaced from a
dry run alone:

| Stage | Catches |
|---|---|
| `sbatch --test-only` | unsatisfiable resource requests |
| **one real task** | every defect in §1 — imports, compiler, silent-success |
| the array | nothing new, if the single task was read properly |

Four single-task smokes were needed here, at 11 s, 11 s, 17 s and 1 h. The 17 s one
exited `COMPLETED`; only reading its `summary.json` showed it had canonicalised
nothing.

---

## 5. Related

- [`T15-d2s-failure-modes.md`](../../../.claude/notes/review/tasks/T15-d2s-failure-modes.md) — the ticket this was found under
- [`EXECUTION-PLAN.md`](../../../.claude/notes/review/tasks/EXECUTION-PLAN.md) §2 — the G1–G8 certification gate this belongs in
- `slurm/t15_norm_arms_launch.sh`, `slurm/workers/t15_norm_arms_slurm.sh`
