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

### One-time build on Picasso login node

```bash
module load gcc/13.2.0
module load cmake/3.31.4

conda activate isalsr
pip install nanobind ninja    # if not already present

pip install -e ".[dev,native]" -v
```

Verify immediately after:

```bash
python -c "from isalsr.core import backends; print(backends.engine()); print(backends.build_info())"
```

`isa_level` should be `x86-64-v3`.  The build targets the portable v3 ISA
level (AVX2 + FMA) because 178 of 333 Picasso nodes lack AVX-512; one binary
covers all node classes and produces a single deterministic build hash across
the entire 3,000-run campaign.

### SLURM tasks

No module-load lines are required in worker scripts.  The statically-linked
`.so` runs on any node regardless of which system gcc is loaded:

```bash
# Worker preamble — no gcc module needed at runtime
conda activate isalsr
python -m experiments.models.orchestrator ...
```

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
