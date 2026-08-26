# C++ Build Guide — isalsr.core._native

## Overview

`isalsr.core._native` is a nanobind extension module containing performance-critical
routines from `isalsr.core`.  It is an optional drop-in: if the `.so` is absent,
every call falls back to the pure-Python implementation transparently.

Build system: **scikit-build-core ≥ 0.9** + **CMake ≥ 3.18** + **ninja**.
Language standard: **C++17**.  Minimum Python: **3.11**.

---

## Local workstation build

**Requirements**: gcc ≥ 10, cmake ≥ 3.18, ninja.

```bash
conda activate isalsr
# One-time: install build-time deps (nanobind, ninja)
pip install nanobind ninja

# Build and install in editable mode
pip install -e ".[dev,native]" -v

# Verify
python -c "from isalsr.core import backends; print(backends.engine()); print(backends.build_info())"
```

Expected output: `cpp` followed by a dict whose `isa_level` key shows `x86-64-v3`
on the i7-13700KF (AVX2 + FMA, no AVX-512).

### Local override: -march=native

Only for profiling on the workstation.  Do NOT submit the resulting `.so` to SLURM.

```bash
CMAKE_ARGS="-DISALSR_NATIVE_MARCH=ON" pip install -e ".[dev,native]" -v
```

### Sanitizer build (development)

```bash
CMAKE_ARGS="-DISALSR_ENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug" \
    pip install -e ".[dev,native]" -v
```

### Rebuilding after C++ edits

Editable installs only re-run CMake when `pyproject.toml` or `CMakeLists.txt`
change.  After editing `native/src/*.cpp` or `native/include/**/*.hpp`:

```bash
pip install -e ".[dev,native]" -v   # re-triggers CMake build
```

---

## Picasso (SLURM + Singularity) build

Picasso's login-node default is **gcc 7.5.0** (too old for C++17).
The **build** requires loading a C++17-capable toolchain, but the **installed
`.so` is statically linked** against libstdc++ and libgcc, so no `module load`
is needed at run time inside SLURM tasks.

### One-time environment setup on the Picasso login node

Verified end to end on 2026-07-27.  `envs_dirs[0]` is already
`fscratch/conda_envs`, so a plain `-n isalsr` lands in the right place.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n isalsr python=3.11 -y          # match the workstation (3.11.15)
conda activate isalsr

module load gcc/13.2.0                          # login default is 7.5.0 — too old for C++17

cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR
pip install nanobind ninja scikit-build-core    # cache the build deps while there is internet
pip install -e ".[dev]"
```

The build deps must be installed **on the login node**: compute nodes have no
outbound internet, so a compute-node build only works with
`--no-build-isolation` against an already-populated environment.

Verify immediately after:

```bash
python -c "from isalsr.core import backends; import json; \
           print(backends.engine()); print(json.dumps(backends.build_info(), sort_keys=True))"
```

Expect `cpp` and `isa_level: x86-64-v3`.  The build targets the portable v3 ISA
level (AVX2 + FMA) because 178 of 333 Picasso CPU nodes lack AVX-512; one binary
covers all node classes and yields a single deterministic build hash across the
whole campaign.  **Measured 2026-07-27: `build_hash` is `298fc1188bf1b051` on
both the workstation (gcc 12.2.0) and Picasso (gcc 13.2.0)** — identical, which
is the property that keeps engine timings comparable across machines.

### Compute-node smoke

```bash
bash slurm/smoke_cpp/launcher.sh --test-only   # validate the request, no queue
bash slurm/smoke_cpp/launcher.sh               # one real task
```

Builds on the compute node with `--no-build-isolation`, asserts the native
engine loaded, then runs the equivalence gate and the benchmark there.

### SLURM tasks

No module-load lines are required in worker scripts at run time — the `.so` is
statically linked against libstdc++/libgcc and runs on any node.

```bash
conda activate isalsr
python -m experiments.models.orchestrator ...
```

> **Do not prepend `$REPO_DIR/src` to `PYTHONPATH`.**  The generic Picasso
> worker template does this, and here it is actively harmful: the `.so` is
> installed into `site-packages/isalsr/core/`, not into the source tree, so a
> `src`-first path makes `import isalsr` resolve to the sources, silently fall
> back to pure Python, and produce a run that measures nothing.  Every campaign
> worker must instead assert the engine before doing any work:
>
> ```python
> from isalsr.core import backends
> assert backends.engine() == "cpp", "native engine not loaded"
> ```

---

## ISA selection rationale

| Flag | Value | Reason |
|---|---|---|
| Default march | `x86-64-v3` | 178/333 Picasso nodes lack AVX-512 |
| `ISALSR_NATIVE_MARCH=ON` | `native` | Local profiling only; not for SLURM |
| AVX-512 | NOT targeted | Would fail on ~53% of the cluster |

---

## Extension layout

```
src/isalsr/core/native/
    include/isalsr/
        fnv.hpp              FNV-1a 64-bit hash (header-only)
        cdll.hpp             Circular doubly linked list
        labeled_dag.hpp      Labeled DAG, ordered in-edges
        node_types.hpp       Token grammar / label mapping
        string_to_dag.hpp    S2D
        wl.hpp               1-WL subtree hash
        canonical.hpp        fast_canonical_string (wl_only)
    src/
        bindings.cpp         NB_MODULE entry point
        probe.cpp            engine_name / build_info / fnv1a64
        cdll.cpp
        labeled_dag.cpp
        node_types.cpp
        string_to_dag.cpp
        wl.cpp
        canonical.cpp
CMakeLists.txt               Build definition (repo root)
```

The `.so` is installed into `isalsr/core/` by CMake's `install()` rule so
that `import isalsr.core._native` resolves in both editable and regular installs.

**Why the directory is `native/` and not `_native/`.** The built extension is
`isalsr.core._native`. A sibling directory of that exact name could be picked up
as an implicit namespace package. In practice CPython's `FileFinder` tries
extension-module loaders before the namespace fallback, so the `.so` would still
win — but the margin is a single import-machinery detail, and the two names are
free to differ, so they do.

The C++ sources travel with the checkout (so a `git pull` on a cluster is enough
to rebuild) but are excluded from wheels via `wheel.exclude` in
`pyproject.toml`; only the compiled `.so` ships.

**Verifying the layout after any move**, all four must hold:

```bash
python -c "from isalsr.core import _native; print(_native.__file__)"   # a .so, not a directory
python -c "from isalsr.core import backends; print(backends.engine())" # cpp
ISALSR_ENGINE=python python -c "from isalsr.core import backends; print(backends.engine())"  # python
find "$(python -c 'import isalsr,os;print(os.path.dirname(isalsr.__file__))')" -name '*.cpp' | head  # empty
```

---

## Engine switch

```python
from isalsr.core import backends

# Which engine is active?
backends.engine()           # "cpp" or "python"

# Full build metadata
backends.build_info()       # dict with compiler, isa_level, build_hash, ...

# Force Python engine for one session (no recompile needed)
# ISALSR_ENGINE=python python -m experiments.models.orchestrator ...
```

The env variable `ISALSR_ENGINE` accepts `cpp` or `python` and wins over the
compiled-in default.  Requesting `cpp` when the extension is absent raises
`RuntimeError` immediately rather than failing silently later.
