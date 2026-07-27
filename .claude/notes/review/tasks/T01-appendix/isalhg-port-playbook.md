# IsalHG → IsalSR C++ Porting Playbook

Primary source evidence: IsalHG repo at `/home/mpascual/research/code/IsalHG`.
All file:line locators below are IsalHG unless prefixed `IsalSR:`.

---

## 1. Build System

### Backend swap (REQUIRED — IsalSR currently uses setuptools)

IsalSR `pyproject.toml` lines 2–3 must be replaced with:

```toml
[build-system]
requires = ["scikit-build-core>=0.9", "nanobind>=2.0"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
minimum-version = "build-system.requires"
cmake.version = ">=3.18"
ninja.version = ">=1.10"
build-dir = "build/{wheel_tag}"
wheel.packages = ["src/isalsr"]

[tool.scikit-build.cmake.define]
CMAKE_BUILD_TYPE = "Release"
```

Source: `IsalHG/pyproject.toml` lines 1–21.

### CMakeLists.txt shape (verbatim flags from `IsalHG/CMakeLists.txt`)

```cmake
cmake_minimum_required(VERSION 3.18)
project(isalsr_native LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -DNDEBUG -fno-plt -funroll-loops)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _IPO_OK OUTPUT _IPO_ERR)
    if(_IPO_OK)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
endif()

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core
    NB_STATIC
    src/isalsr/core/_native/bindings.cpp
    src/isalsr/core/_native/src/cdll.cpp
    src/isalsr/core/_native/src/node_types.cpp      # ← token.cpp analogue
    src/isalsr/core/_native/src/labeled_dag.cpp     # ← sparse_hypergraph.cpp analogue
    src/isalsr/core/_native/src/dag_to_string.cpp   # ← h2s.cpp analogue
    src/isalsr/core/_native/src/string_to_dag.cpp   # ← s2h.cpp analogue
    src/isalsr/core/_native/src/wl.cpp              # reuse pattern verbatim
    src/isalsr/core/_native/src/canonical.cpp       # reuse pattern verbatim
    src/isalsr/core/_native/src/thread_pool.cpp     # copy verbatim
)

find_package(Threads REQUIRED)
target_link_libraries(_core PRIVATE Threads::Threads)
target_include_directories(_core PRIVATE src/isalsr/core/_native/include)

install(TARGETS _core LIBRARY DESTINATION isalsr/core)
```

Source: `IsalHG/CMakeLists.txt` lines 1–79.

### Where the .so lands

`install(TARGETS _core LIBRARY DESTINATION isalsr/core)` places `_core.so`
at `src/isalsr/core/_core.so`. scikit-build-core's editable redirect mode
(`.pth` → `src/`) makes `import isalsr.core._core` find it without copying.
C++ source edits require `pip install -e ".[dev]"` re-run (incremental CMake,
typically seconds after first build). Source: `IsalHG/pyproject.toml` lines 16–17,
`IsalHG/docs/engineering/DEVELOPMENT.md` lines 43–47.

### PGO (opt-in, GCC two-stage)

```bash
# Stage 1 — instrument
CMAKE_ARGS="-DISALSR_PGO_GENERATE=ON" pip install -e ".[dev]" --no-build-isolation
# Stage 2 — training run (replace with IsalSR sweep over representative DAGs)
python scratchpad/cpp_pgo_train.py
# Stage 3 — profile-use build
CMAKE_ARGS="-DISALSR_PGO_USE=ON" pip install -e ".[dev]" --no-build-isolation --force-reinstall
```

Profile data in `build/pgo-data/`. PGO default OFF. Measured gain in IsalHG:
−1 to −3% on small designs, noise on large designs (parallel ceiling dominates).
Worth copying the infra; do not prioritise the two-stage run until after the
parallel seed loop is in place (R3 dwarfs PGO). Source: `IsalHG/CMakeLists.txt`
lines 34–52, `IsalHG/docs/engineering/CPP_OPTIMIZATION_LOG.md` rounds 7 and 8+PGO.

### `-march=native` risk on Picasso

The CMakeLists.txt comment (line 14) says "acceptable for a research repo where
the dev workstation and CI share a CPU family." This is FALSE for IsalSR: local
workstation is Intel i7-13620H; Picasso A100 nodes are AMD EPYC 7H12. A binary
built with `-march=native` on the Intel host will use AVX-512 or Tiger-Lake
intrinsics absent on EPYC (or vice versa) and will either SIGILL or produce
wrong code. Mitigation: add a CMake option `ISALSR_NATIVE_MARCH` (default ON
for dev, OFF for SLURM builds) and guard the flag behind it.

---

## 2. Engine Switch

IsalHG does NOT have a graceful ImportError fallback — `canonical.py` line 69
does a hard `from isalhg.core._core import canonical_string as _core_canonical_string`.
If the `.so` is absent this raises `ImportError` at module import.

**IsalSR must add the fallback explicitly.** Required pattern for
`src/isalsr/core/canonical.py` (module level):

```python
# backends.py (unchanged from IsalHG pattern)
DEFAULT_BACKEND: Backend = "cpp"

# canonical.py — add at top, replacing hard import
try:
    from isalsr.core._core import canonical_string as _core_canonical_string
    _HAS_NATIVE = True
except ImportError:
    _core_canonical_string = None  # type: ignore[assignment]
    _HAS_NATIVE = False

# Override default when native is absent
if not _HAS_NATIVE:
    import isalsr.core.backends as _backends
    _backends.DEFAULT_BACKEND = "python"
```

Force an engine for benchmarking (caller side — no module reload needed):

```python
from isalsr.core.canonical import canonical_string
canonical_string(dag)                         # uses DEFAULT_BACKEND
canonical_string(dag, backend="cpp")          # force native
canonical_string(dag, backend="python")       # force pure-Python
```

Source: `IsalHG/src/isalhg/core/backends.py` (full file, 71 lines),
`IsalHG/src/isalhg/core/canonical.py` lines 65–201.

### Dispatch registry pattern (copy verbatim)

```python
_CANONICAL_STRING_BACKENDS: dict[str, object] = {
    "python": _python_canonical_string,
    "cpp": _cpp_canonical_string,
}

def canonical_string(dag, *, backend=None, ...):
    impl = resolve(backend, _CANONICAL_STRING_BACKENDS)
    return impl(dag, ...)
```

`resolve()` in `backends.py` handles `None` → `DEFAULT_BACKEND` and raises
`ValueError` on unknown names. Source: `IsalHG/src/isalhg/core/backends.py`
lines 45–70.

---

## 3. Binding Surface

### Pattern: Python object → C++ view copy at FFI boundary (never zero-copy)

```cpp
// In bindings.cpp, for every entry point:
isalhg::SHG H = shg_from_python(py_H);   // copies under GIL
nb::gil_scoped_release release;           // release before compute
result = canonical_string_compute(H, ...); // pure C++ from here
```

Source: `IsalHG/src/isalhg/core/_native/bindings.cpp` lines 104–138, 211–220.

### Per-TU binding surface table

| TU (IsalHG) | IsalSR analogue | Python→C++ input | C++→Python output | Copy/View | Persistent C++ obj? |
|---|---|---|---|---|---|
| `cdll.cpp` | `cdll.cpp` | int capacity | `SlotIdx` (int32) | copy | No — per-call Cdll |
| `token.cpp` | `node_types.cpp` | none (pure C++) | `std::string` via serialize | N/A | No |
| `sparse_hypergraph.cpp` | `labeled_dag.cpp` | `nb::handle py_H` attrs | None (internal) | **full copy** | No |
| `h2s.cpp` | `dag_to_string.cpp` | C++ SHG/DAG ref | `std::string` or `nb::list` | N/A | No |
| `s2h.cpp` | `string_to_dag.cpp` | `std::string` | `nb::tuple(labels, edges)` | copy out | No |
| `structural_tuples.cpp` | (adapt for DAG — see §8) | C++ DAG ref | `std::vector<NodeId>` | copy out | No |
| `wl.cpp` | `wl.cpp` | C++ DAG ref | `std::vector<int64_t>` | copy out | No |
| `canonical.cpp` | `canonical.cpp` | C++ DAG ref | `std::string` | N/A | No |
| `thread_pool.cpp` | `thread_pool.cpp` | none | none | N/A | **YES** — process singleton |

All complex types that cross the Python↔C++ boundary are **copied**, not viewed.
No buffer protocol, no `nb::ndarray`, no shared memory. The only persistent C++
object is the `canonical_thread_pool()` process-level singleton (static storage,
sized to `hardware_concurrency()`). Source: `IsalHG/src/isalhg/core/_native/bindings.cpp`
lines 1–313, `IsalHG/src/isalhg/core/_native/src/canonical.cpp` lines 100–145.

### Exception translation idiom (copy verbatim, rename namespace)

```cpp
struct PyExcCache { PyObject* IsalHGError; ... };
PyExcCache g_exc{};

void init_exception_cache() {
    nb::module_ errors_mod = nb::module_::import_("isalhg.errors");
    ...
}
nb::register_exception_translator(
    [](const std::exception_ptr& p, void*) { translate_exception(p); });
```

Source: `IsalHG/src/isalhg/core/_native/bindings.cpp` lines 40–101.

### `nb::gil_scoped_release` discipline

- Acquire GIL to call `shg_from_python()` (reads Python object attributes).
- Release GIL around `canonical_string_compute()` / `string_to_hypergraph_compute()`.
- Thread pool workers are spawned AFTER GIL is released; they touch only C++ objects.
Source: `IsalHG/src/isalhg/core/_native/bindings.cpp` lines 211–225, 244–251.

---

## 4. Data Structure Choices (C++ side)

### LabeledDAG C++ view (adapt from SHG)

IsalHG SHG uses SoA layout; transferable pattern:

```cpp
struct LabeledDAG {
    int32_t  n_nodes;                              // node count
    int32_t  n_edges;                              // directed edge count
    std::vector<int32_t>  node_types;              // NodeType enum → int32_t per node
    std::vector<std::vector<int32_t>> in_neighbors;  // ordered by insertion (invariant 3/8)
    std::vector<std::vector<int32_t>> out_neighbors; // sorted or insertion-ordered
    // Derived (built by finalise()):
    std::vector<uint64_t> wl_colours;             // 1-WL per node
};
using NodeId = int32_t;
```

Key differences from SHG:
- Node type label (NodeType enum) on nodes, NOT on edges.
- `in_neighbors` must preserve **insertion order** (critical for binary ops — invariant 8).
- No `edge_members` or `vertex_edges` — edges are explicit pairs.
- No `primal_adj` — graph IS the DAG.
- No `eta_cache` — no hyperedges to sum over.

Source for SHG layout: `IsalHG/src/isalhg/core/_native/src/sparse_hypergraph.cpp` lines 1–103,
`IsalHG/src/isalhg/core/_native/bindings.cpp` lines 104–138.

### CDLL (copy verbatim from IsalHG)

```cpp
struct CdllNode { int32_t next, prev; int32_t data; };  // data = NodeId (graph node index)
struct Cdll {
    std::vector<CdllNode> slots_;  // pre-allocated capacity
    std::vector<int32_t>  free_;   // descending stack; pop_back() → slot 0 first
    int32_t capacity_, size_;
};
using SlotIdx = int32_t;
```

CDLL is NOT the graph. `slots_[ptr].data` = graph node index. `ptr` is a CDLL slot
index. **Do not conflate** (IsalSR invariant 1). Source: `IsalHG/src/isalhg/core/_native/src/cdll.cpp`.

### Token struct (adapt from IsalHG token.hpp)

IsalHG token carries `edge_label`, `i`, `j`, `n_labels`, `labels[MAX_NEW]`.
IsalSR token carries `node_type` (the single-char label V+/V*/Vs/...), pointer
displacement, `new_node_type`. Exact fields differ; the `std::array<..., MAX_NEW>`
pattern (Round 4 — eliminates heap in inner loop) is directly applicable.

### WL hash type

`uint64_t` per node, FNV-1a 64-bit mixing. Cross-process stable (no ASLR dependency).
Source: `IsalHG/src/isalhg/core/_native/src/wl.cpp` lines 35–94,
`IsalHG/src/isalhg/core/_native/include/isalhg/fnv.hpp` (FNV domain constants).

---

## 5. What FAILED (do not repeat)

Source: `IsalHG/docs/engineering/CPP_OPTIMIZATION_LOG.md` "Negative results" section
(lines 248–289) and round-specific notes.

| # | Attempted optimisation | Outcome | Root cause |
|---|---|---|---|
| 1 | Per-frame slot-displacement cache (2.5 kB `std::array` on stack per recursion frame) | −5 to −15% regression on greedy_min | L1d cache thrash under 16-thread parallel load; math showed net-zero theoretical gain |
| 2 | Stack-allocated `best_prefix` + `tmp_move_block` (`std::array<Token, 32>`) | +2% greedy_single, regression on parallel greedy_min | Extra stack pressure per worker thread; heap allocator amortises well across pool |
| 3 | Arena-pooled sub-completion vectors keyed by recursion depth (depth×2+slot indexing) | Reverted | `std::vector<std::vector<Token>>` invalidates references on grow; `std::deque` fix costs more in indirection than saved in allocation |
| 4 | Flat 1-D eta cache (stride instead of `vector<vector<int32_t>>`) | ~3% regression on STS13/Doily | Eta comparison reached only after cascade short-circuit; extra key arithmetic is on the hot path even when never compared |
| 5 | CPU pinning to P-cores via `taskset -c 0-11` | Doily greedy_min 80ms → 110ms | With 15 seeds and persistent pool, capping at 12 cores queues seeds; OS scheduler outperforms manual pinning on hybrid-core CPU |

---

## 6. What WON

Source: `IsalHG/docs/engineering/CPP_OPTIMIZATION_LOG.md` rounds 1–11.

| Round | Optimisation | Measured gain |
|---|---|---|
| Phase 0 | C++ port (baseline vs pure Python) | 101–180× (single-thread, pre-Round 3) |
| R1 | V-branch prefix shortcut (eliminate per-permutation prefix copy) | 0.5–2.2% |
| R2 | LTO (`CMAKE_INTERPROCEDURAL_OPTIMIZATION`) + `-funroll-loops -fno-plt` | 2–5% |
| **R3** | **Parallel seed loop (`std::async` + `nb::gil_scoped_release`)** | **−73–89% on greedy_min; 3.8–9.4× vs R0** |
| R4 | `VCandidate` → `std::array<..., MAX_NEW>` (heap-free inner loop) | 0–2%; enables R5 |
| R5 | Persistent thread pool (replace `std::async` per-call with reused pool) | −6–12% on small designs |
| R6 | Callback-based perm enumeration (eliminate `vector<vector>` build-up) | 0–4% cleanup |
| R7 | PGO (two-stage GCC profile-use) | −1–3% (noise on large designs) |
| R8 | Running counters in `EncoderState` (replace O(n) scan with field read) | −1–10% |
| R8+PGO | PGO regeneration against R8 source | additional −1–3.4% |
| R9 | PI nbrdeg seed selector (cheaper O(n·d̄) vs O(n²·depth) BFS) | −12–74% on non-vertex-transitive inputs (15.2% of random inputs) |
| R10 | Incidence-restricted candidate scan + drop per-cost sort + k_disp cap | ~2× on tie-complete canonical |
| R11 | Inverted displacement enumeration (per-edge rather than blind tuples) | 7–9× on sparse large-n |

**Priority for IsalSR**: implement R3 (parallel seed loop + GIL release) first —
it is the single largest win and all other rounds are <10% incremental on top.

---

## 7. Benchmark Protocol

Source: `IsalHG/docs/engineering/CPP_SPEEDUP.md` (full file),
`IsalHG/docs/engineering/CPP_OPTIMIZATION_LOG.md` rounds 8 and 9.

| Item | Specification |
|---|---|
| Reps | median of **4** reps |
| Per-rep policy | **best-of-9** wall-clock runs |
| Warmup | `--warmup 3` (3 cold runs discarded before timing) |
| Driver script | `scratchpad/cpp_vs_levi.py` (or IsalSR equivalent) |
| Raw artefacts | JSON files in `scratchpad/bench/`, one per rep: e.g. `nbrdeg_pgo_rep[1-4].json` |
| Thermal discipline | 6 s sleep between runs; 30 s cooldown after rebuild; baseline and optimised measured in **same** thermal state |
| Correctness gate | Byte-equal canonical strings between C++ and Python on every fixture (`--- ALL EQ True ---` smoke check) |
| Result table columns | Design \| C++ greedy_min (ms) \| greedy_single (ms) \| Speedup vs Python \| [competitor backends] |
| PGO training driver | `scratchpad/cpp_pgo_train.py` — training rep counts scaled per design size |

**Minimum benchmark set for IsalSR**: pick 4–6 representative DAGs spanning k=3 to k=15
internal nodes (analogous to Fano/STS9/STS13/Doily size gradient). Run each under
`backend="python"` and `backend="cpp"` with 30 seeds each; report best-of-9 / median-of-4.

---

## 8. Hypergraph-Specific — Discard List

The following IsalHG design elements exist solely because of hyperedges. Do NOT port to IsalSR.

| IsalHG element | File | Reason not applicable to labeled DAGs |
|---|---|---|
| `SHG::edge_members` (sorted `vector<NodeId>` per edge) | `sparse_hypergraph.cpp` | Hyperedges are sets; DAG edges are directed pairs |
| `SHG::vertex_edges` (inverse index: node → incident edges) | `sparse_hypergraph.cpp` | Replace with `in_neighbors` / `out_neighbors` per node |
| `SHG::primal_adj` (clique expansion of hyperedges) | `sparse_hypergraph.cpp` | DAG adjacency is explicit; no reconstruction needed |
| `SHG::eta_cache` (sum of xi shell counts per edge) | `sparse_hypergraph.cpp` lines 83–100 | No hyperedges to aggregate over |
| `SHG::edge_labels` / `n_edge_labels` | `sparse_hypergraph.cpp` | IsalSR labels are on NODES (NodeType), not edges |
| `required_k_compute()` / `k = max_arity` | `canonical.cpp` lines 18–25 | IsalSR VM has exactly 2 pointers (primary, secondary); k is fixed |
| Variable-arity V-perm enumeration (`enumerate_label_perms_cb`) | `h2s.cpp` | In IsalSR's V token, exactly one new node is created; no multi-node permutation |
| `xi_counts` / `xi_labelled_counts` (BFS shell counts) | `structural_tuples.cpp`, `sparse_hypergraph.cpp` | Hypergraph-specific seed selector metric; for DAGs use in/out-degree + WL |
| `structural_depth` parameter (xi BFS depth) | `canonical.cpp` line 53 | Depth-3 BFS is only needed for xi-based seed selection |
| `SHG::max_arity` | `sparse_hypergraph.cpp` | Fixed at 2 in IsalSR (binary ops); arity selection logic absent |
| `SHG::n_edge_labels` / edge label vocabulary | `sparse_hypergraph.cpp`, `bindings.cpp` | No edge vocabulary in IsalSR |

**Transferable from SHG (adapt, not discard):**
- SoA memory layout with `int32_t` indices → apply to LabeledDAG C++ view
- `finalise()` pattern (build derived structures after populating raw data) → build `wl_colours` and ordered adjacency lists
- BFS connectivity check in `canonical.cpp` `is_connected()` → DAG has a unique root (x_0 has no in-neighbors); connectivity check is different for DAGs with multiple variables

**WL hash for DAGs (wl.cpp adaptation):**
- Replace "sorted member colours" with ordered in-neighbor colours (insertion order matters — invariant 8)
- Replace edge-label hash component with node-type hash component
- Otherwise FNV-1a mixing structure is identical

---

## Appendix: TU sizes and port priority

| IsalHG TU | LoC | Port status | Priority |
|---|---:|---|---|
| `h2s.cpp` | 1016 | Adapt (DAG traversal, no arity enumeration) | High — core algorithm |
| `s2h.cpp` | 289 | Adapt (IsalSR 2-char token grammar) | High |
| `structural_tuples.cpp` | 182 | Adapt (DAG degree-based seed selector) | Medium |
| `canonical.cpp` | 158 | Near-verbatim (multi-seed dispatch + thread pool) | High |
| `token.cpp` | 125 | Adapt (IsalSR token alphabet) | High |
| `sparse_hypergraph.cpp` | 103 | Redesign as `labeled_dag.cpp` | High |
| `wl.cpp` | 94 | Adapt (directed edges, node-type labels) | Medium |
| `cdll.cpp` | 68 | **Copy verbatim** | Low |
| `thread_pool.cpp` | 12 | **Copy verbatim** | Low (but do early — enables R3) |
