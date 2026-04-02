# IsalSR Precomputed Canonical String Cache — Design & Implementation Task

**Purpose**: Precompute and persist canonical string representations for all four D2S algorithms, eliminating redundant canonicalization across seeds, SR methods, and experiment reruns.

**Author**: Mario Pascual González  
**Date**: March 2026

---

## 1. Motivation

The canonicalization step (especially `exhaustive` and `pruned-exhaustive`) is the single most CPU-intensive operation in IsalSR. Its output is a **pure function** of the DAG structure — it depends on no random seed, no training data, and no SR method. Therefore, for a given `(num_variables, operator_set, max_internal_nodes)` configuration, the mapping:

```
DAG → { algorithm_name: canonical_string }
```

is deterministic and reusable. Precomputing this mapping yields:

1. **Cross-seed reuse**: The same DAG encountered in seed 3 and seed 17 is canonicalized only once.
2. **Cross-method reuse**: UDFS and GraphDSR encountering the same DAG share the lookup.
3. **Algorithm cross-validation**: Storing all four outputs lets us verify offline that `exhaustive == pruned-exhaustive` (correctness) and that greedy variants agree when they should.
4. **Timing decoupling**: Canonicalization wall-time is measured during precomputation, not during the SR experiment — the experiment measures *pure search performance*.

---

## 2. Precomputation Strategy

### 2.1. Phase 1 — Systematic Enumeration (seed-independent)

For each benchmark configuration `(m, Ops, k_max)` where `m` = number of input variables, `Ops` = allowed operator set, `k_max` = maximum internal nodes:

1. Enumerate all valid **DAG skeletons** with `k ∈ {1, ..., k_max}` intermediary nodes (UDFS-style: topological numbering, random predecessor sampling is replaced by exhaustive enumeration for small k).
2. For each skeleton, enumerate all **operator labelings** (assign each intermediary node a label from `Ops`, respecting arity constraints).
3. For each resulting labeled DAG `D`:
   a. Run `StringToDAG` → `DAGToString` round-trip to obtain the raw IsalSR string.
   b. Compute all four D2S algorithms on `D`.
   c. Record the timing of each algorithm.
   d. Compute a **DAG hash** (defined below) for indexing.
4. Deduplicate: if two labeled DAGs produce the same exhaustive canonical string, they are isomorphic — keep only one representative.

**Scalability note**: For `m=2`, `|Ops|=8`, `k=5`, the number of DAG frames is on the order of 10⁶–10⁷. This is feasible (minutes to hours). For `k=6+`, the space grows rapidly — use sampling (Phase 2).

### 2.2. Phase 2 — Random Sampling Supplement (seed-dependent)

For configurations where systematic enumeration is infeasible (`k > k_max`):

1. For each random seed `s ∈ {1, ..., 30}`:
   a. Generate `N_sample` random valid IsalSR strings (using `random_isalsr_string()` from `isalsr.search.operators`).
   b. Decode each string via `StringToDAG` → obtain the DAG.
   c. Compute all four D2S algorithms + timing.
   d. Insert into the cache (no deduplication needed — the cache handles it via hash keys).
2. The cache grows monotonically across seeds. After all 30 seeds, it contains the union of all DAGs encountered.

### 2.3. Phase 3 — Lazy Runtime Extension

During actual SR experiments, any DAG **not found** in the precomputed cache is canonicalized on-the-fly, and the result is **appended** to the cache file. This ensures the cache is complete after the first full experiment run.

---

## 3. DAG Hashing

We need a fast, collision-resistant hash to index DAGs in the cache. The hash must be **representation-independent** (two isomorphic DAGs must hash to the same value) OR **string-based** (hash the raw IsalSR string before canonicalization).

**Chosen approach**: Use the **exhaustive canonical string** itself as the primary key. This is the only truly isomorphism-invariant identifier. However, since we are *precomputing* it, we need a secondary key for lookup *before* we know the canonical string.

**Two-level indexing**:

```
Level 1 (fast lookup): raw_isalsr_string → cache entry
Level 2 (canonical index): exhaustive_canonical_string → list of raw strings that map to it
```

This means: during an experiment, the SR method produces a raw string `w`. We look up `w` in the Level 1 index. If found → return the precomputed canonical forms. If not found → canonicalize, store, and return.

For DAGs not generated from strings (e.g., constructed programmatically), we use the `greedy-single` output as the Level 1 key (it is the fastest D2S and deterministic given the node ordering).

---

## 4. File Format

### 4.1. Choice of Format

| Format | Pros | Cons | Verdict |
|--------|------|------|---------|
| JSON | Human-readable, easy to inspect | Slow for >10⁵ entries, large on disk | Small benchmarks only (Nguyen) |
| HDF5 (.h5) | Fast random access, compression, large-scale | Requires `h5py`, opaque | **Primary format** for Feynman/SRBench |
| NPZ | NumPy-native, fast | Bad for variable-length strings | Not suitable |
| SQLite | Fast keyed lookup, ACID, no external deps | Overkill for batch writes | Good alternative to HDF5 |

**Decision**: Use **HDF5** as the primary storage format (via `h5py`), with a JSON **sidecar** for metadata. For small benchmarks (Nguyen, ≤12 problems), also export a human-readable JSON mirror.

### 4.2. HDF5 Internal Structure

```
cache_{benchmark}_{m}vars_{ops_hash}.h5
│
├── attrs/                              # HDF5 root attributes (metadata)
│   ├── "num_variables": int            # m
│   ├── "operator_set": str             # JSON-encoded list, e.g. '["+",...,"log"]'
│   ├── "max_internal_nodes": int       # k_max for systematic enumeration
│   ├── "creation_timestamp": str       # ISO 8601
│   ├── "isalsr_version": str           # e.g. "0.1.0"
│   ├── "git_hash": str                 # Git commit hash of IsalSR
│   └── "phase": str                    # "systematic", "sampled", or "mixed"
│
├── strings/                            # Group: all D2S outputs
│   ├── raw            [N]  vlen-str    # Raw IsalSR string (Level 1 key)
│   ├── greedy_single  [N]  vlen-str    # GreedySingleD2S output
│   ├── greedy_min     [N]  vlen-str    # GreedyMinD2S output
│   ├── pruned         [N]  vlen-str    # PrunedExhaustiveD2S output
│   ├── exhaustive     [N]  vlen-str    # ExhaustiveD2S output (true canonical)
│   └── is_canonical   [N]  bool        # True if raw == exhaustive (already canonical)
│
├── dag_properties/                     # Group: structural properties of each DAG
│   ├── n_nodes        [N]  int32       # Total number of nodes
│   ├── n_internal     [N]  int32       # Number of internal (non-VAR) nodes = k
│   ├── n_edges        [N]  int32       # Number of edges
│   ├── n_var_nodes    [N]  int32       # Should always == m
│   ├── depth          [N]  int32       # Longest path length in the DAG
│   └── structural_tuple [N, 6] int32   # 6-component tuple of the start node (x_1)
│
├── timings/                            # Group: wall-clock time for each algorithm (seconds)
│   ├── greedy_single  [N]  float64
│   ├── greedy_min     [N]  float64
│   ├── pruned         [N]  float64
│   └── exhaustive     [N]  float64
│
├── correctness/                        # Group: cross-validation flags
│   ├── exhaustive_eq_pruned  [N] bool  # True if exhaustive == pruned (should always be True)
│   ├── greedy_single_eq_exhaustive [N] bool  # True if greedy-single matches canonical
│   └── greedy_min_eq_exhaustive    [N] bool  # True if greedy-min matches canonical
│
└── canonical_index/                    # Group: reverse index (canonical → raw strings)
    ├── unique_canonical    [U]  vlen-str  # U unique canonical strings
    ├── canonical_to_first  [U]  int64     # Index into strings/ of the first raw entry
    └── multiplicity        [U]  int32     # How many raw strings map to this canonical
```

Where `N` = total entries in the cache, `U` = number of unique canonical strings (U ≤ N).

### 4.3. JSON Sidecar

```json
{
  "cache_file": "cache_nguyen_1var_8ops.h5",
  "num_variables": 1,
  "operator_set": ["+", "*", "-", "/", "sin", "cos", "exp", "log"],
  "max_internal_nodes": 5,
  "total_entries": 1234567,
  "unique_canonical_entries": 205678,
  "empirical_mean_reduction_factor": 6.0,
  "phase": "systematic",
  "creation_timestamp": "2026-03-18T14:30:00Z",
  "isalsr_version": "0.1.0",
  "git_hash": "abc123def",
  "generation_wall_time_s": 3421.5,
  "algorithms_stored": ["greedy_single", "greedy_min", "pruned", "exhaustive"],
  "correctness_summary": {
    "exhaustive_eq_pruned_rate": 1.0,
    "greedy_single_eq_exhaustive_rate": 0.87,
    "greedy_min_eq_exhaustive_rate": 0.94
  }
}
```

---

## 5. File Naming and Folder Structure

```
precomputed/
├── README.md                               # This document (abridged)
├── generate_cache.py                       # Main generation script
├── cache_lookup.py                         # Lookup API module
│
├── nguyen/                                 # One folder per benchmark
│   ├── nguyen_1var.h5                      # m=1, all Nguyen 1-var problems
│   ├── nguyen_1var.json                    # Sidecar metadata
│   ├── nguyen_2var.h5                      # m=2, all Nguyen 2-var problems
│   ├── nguyen_2var.json
│   └── nguyen_mirror.json                  # Human-readable mirror (small enough)
│
├── feynman/
│   ├── feynman_{m}var.h5                   # Grouped by number of variables
│   ├── feynman_{m}var.json
│   └── ...
│
└── srbench/
    ├── srbench_{m}var.h5
    ├── srbench_{m}var.json
    └── ...
```

The grouping by `m` (number of variables) is because `m` determines the initial CDLL state and the number of distinguished start nodes. DAGs with different `m` are structurally incompatible and cannot share cache entries.

---

## 6. Python API: `CacheManager`

### 6.1. Public Interface

```python
class CacheManager:
    """Manages the precomputed canonical string cache.

    Provides O(1) lookup of canonical strings for known DAGs,
    with transparent fallback to on-the-fly computation and
    cache extension for unknown DAGs.

    Args:
        cache_dir: Path to the precomputed/ directory.
        num_variables: Number of input variables (m).
        operator_set: List of operator labels.
        read_only: If True, do not extend the cache on misses.
    """

    def __init__(
        self,
        cache_dir: Path,
        num_variables: int,
        operator_set: list[str],
        read_only: bool = False,
    ) -> None: ...

    def lookup(self, raw_string: str) -> CacheEntry | None:
        """Look up a raw IsalSR string in the cache.

        Returns CacheEntry with all four D2S outputs and timings,
        or None if the string is not in the cache.
        """
        ...

    def lookup_or_compute(
        self, dag: LabeledDAG, raw_string: str | None = None
    ) -> CacheEntry:
        """Look up or compute canonical strings for a DAG.

        If the DAG (identified by its raw string) is in the cache,
        return the cached entry. Otherwise, compute all four D2S
        algorithms, store the result in the cache (if not read_only),
        and return the entry.
        """
        ...

    def flush(self) -> None:
        """Write any pending cache entries to disk."""
        ...

    @property
    def stats(self) -> CacheStats:
        """Return cache hit/miss statistics."""
        ...
```

### 6.2. Data Classes

```python
@dataclass(frozen=True)
class CacheEntry:
    """A single precomputed cache entry."""
    raw_string: str
    greedy_single: str
    greedy_min: str
    pruned: str
    exhaustive: str              # True canonical
    n_nodes: int
    n_internal: int
    n_edges: int
    depth: int
    structural_tuple: tuple[int, int, int, int, int, int]
    timing_greedy_single_s: float
    timing_greedy_min_s: float
    timing_pruned_s: float
    timing_exhaustive_s: float
    exhaustive_eq_pruned: bool
    greedy_single_eq_exhaustive: bool
    greedy_min_eq_exhaustive: bool


@dataclass
class CacheStats:
    """Cache hit/miss statistics for a session."""
    total_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    on_the_fly_computations: int = 0
    time_saved_estimate_s: float = 0.0   # Sum of exhaustive timings for hits

    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.cache_hits / self.total_lookups
```

---

## 7. Generation Script: `generate_cache.py`

### 7.1. CLI Interface

```bash
# Systematic enumeration for Nguyen (1 variable, k_max=5)
python precomputed/generate_cache.py \
    --benchmark nguyen \
    --num-variables 1 \
    --operators "+,*,-,/,sin,cos,exp,log" \
    --max-internal-nodes 5 \
    --mode systematic \
    --output-dir precomputed/nguyen/

# Random sampling supplement (larger DAGs)
python precomputed/generate_cache.py \
    --benchmark nguyen \
    --num-variables 1 \
    --operators "+,*,-,/,sin,cos,exp,log" \
    --max-internal-nodes 8 \
    --mode sampled \
    --n-samples 500000 \
    --seed 42 \
    --output-dir precomputed/nguyen/

# Extend existing cache with runtime misses
python precomputed/generate_cache.py \
    --mode extend \
    --cache-file precomputed/nguyen/nguyen_1var.h5 \
    --input-strings runtime_misses.txt
```

### 7.2. Algorithm (pseudocode)

```
function generate_systematic(m, Ops, k_max):
    entries = []
    for k in 1..k_max:
        for skeleton in enumerate_dag_skeletons(m, k):
            for labeling in enumerate_operator_labelings(skeleton, Ops):
                dag = build_labeled_dag(skeleton, labeling, m)
                raw = GreedySingleD2S().encode(dag)   # fast, for keying

                entry = CacheEntry(
                    raw_string = raw,
                    greedy_single = timed(GreedySingleD2S().encode, dag),
                    greedy_min    = timed(GreedyMinD2S().encode, dag),
                    pruned        = timed(PrunedExhaustiveD2S().encode, dag),
                    exhaustive    = timed(ExhaustiveD2S().encode, dag),
                    ...dag_properties(dag)...,
                )
                entries.append(entry)

    deduplicate by exhaustive canonical string
    write_h5(entries)
    write_json_sidecar(entries)


function generate_sampled(m, Ops, max_tokens, n_samples, seed):
    rng = np.random.default_rng(seed)
    entries = []
    for i in 1..n_samples:
        raw = random_isalsr_string(m, max_tokens, Ops, rng)
        try:
            dag = StringToDAG(raw, m).run()
        except (InvalidTokenError, InvalidDAGError):
            continue   # skip invalid strings

        entry = compute_cache_entry(dag, raw)
        entries.append(entry)

    deduplicate
    merge with existing cache (if any)
    write_h5(entries)
```

---

## 8. Implementation Task Specification (for Agent)

### 8.1. Files to Create

| File | Description | Priority |
|------|-------------|----------|
| `precomputed/cache_entry.py` | `CacheEntry` and `CacheStats` dataclasses | P0 |
| `precomputed/cache_manager.py` | `CacheManager` class (lookup, compute, flush) | P0 |
| `precomputed/generate_cache.py` | CLI generation script | P0 |
| `precomputed/enumerate_dags.py` | Systematic DAG skeleton + labeling enumeration | P0 |
| `precomputed/__init__.py` | Package init | P0 |
| `tests/unit/test_cache_manager.py` | Unit tests | P1 |
| `tests/unit/test_enumerate_dags.py` | Enumeration correctness tests | P1 |
| `tests/integration/test_cache_h5.py` | HDF5 read/write round-trip | P1 |

### 8.2. Dependencies

- `h5py` (add to `pyproject.toml` under `[project.optional-dependencies] experiments`)
- All core IsalSR modules (already implemented: `StringToDAG`, `DAGToString`, `canonical_string`, `pruned_canonical_string`, all four D2S algorithms, `LabeledDAG`, `NodeType`)

### 8.3. Testing Criteria

1. **Round-trip**: For every entry in the cache, `StringToDAG(entry.raw_string).run()` produces a DAG whose `ExhaustiveD2S().encode()` matches `entry.exhaustive`.
2. **Exhaustive == Pruned**: `entry.exhaustive_eq_pruned` is `True` for 100% of entries.
3. **Canonical invariance**: Two different raw strings mapping to the same exhaustive canonical string must produce isomorphic DAGs (verify via `LabeledDAG.is_isomorphic()`).
4. **Lookup correctness**: `cache.lookup(raw)` returns the same result as computing from scratch.
5. **Cache extension**: After a miss + compute, subsequent lookup returns the computed entry.
6. **Determinism**: Running `generate_cache.py` twice with the same arguments produces identical HDF5 files (byte-identical after sorting).

### 8.4. Performance Targets

| Configuration | Expected N (entries) | Expected wall-time | Expected file size |
|---------------|---------------------|--------------------|--------------------|
| Nguyen 1-var, k≤5, 8 ops | ~10⁵–10⁶ | 5–30 min | 50–500 MB |
| Nguyen 2-var, k≤4, 8 ops | ~10⁵–10⁶ | 5–30 min | 50–500 MB |
| Feynman, sampled, 500k samples | 500k | 10–60 min | 100–300 MB |

### 8.5. Important Implementation Notes

1. **The raw string is NOT unique** — two different execution orders of `StringToDAG` can produce the same DAG from different raw strings. The `exhaustive` canonical string IS unique per isomorphism class. Use `exhaustive` as the deduplication key.

2. **Timing must use `time.perf_counter_ns()`** — nanosecond resolution, not `time.time()`. Each algorithm is timed individually.

3. **The cache must be thread-safe for writes** — if using multiprocessing for generation, accumulate entries in memory and write in a single batch. HDF5 does not support concurrent writers.

4. **Graceful handling of canonicalization timeouts** — for large DAGs (k > 10), `exhaustive` may be infeasible. Set a timeout (e.g., 60s). If exhaustive times out, store `pruned` as the best available canonical and set a flag `exhaustive_timed_out: True`.

5. **Operator set hashing** — the cache file name includes a hash of the sorted operator set to avoid collisions. Use `hashlib.md5(json.dumps(sorted(ops)).encode()).hexdigest()[:8]`.

6. **Variable-length strings in HDF5** — use `h5py.special_dtype(vlen=str)` for all string datasets. This is critical for performance with variable-length IsalSR strings.

---

## 9. Usage During Experiments

During an SR experiment, the `CacheManager` integrates as follows:

```python
# In the experiment runner:
cache = CacheManager(
    cache_dir=Path("precomputed/nguyen"),
    num_variables=problem.num_variables,
    operator_set=config.operator_set,
)

# Inside the search loop (e.g., UDFS or random search):
for dag in search_method.generate_candidates():
    raw_string = GreedySingleD2S().encode(dag)
    entry = cache.lookup_or_compute(dag, raw_string)

    # Use entry.pruned (or entry.exhaustive) as the canonical form
    canonical = entry.pruned

    if canonical in seen_canonicals:
        n_duplicates += 1
        continue  # Skip isomorphic duplicate

    seen_canonicals.add(canonical)
    fitness = evaluate(dag, x_train, y_train)
    # ... store results ...

cache.flush()  # Persist any runtime extensions
log.info("Cache stats: %s", cache.stats)
```

The `cache.stats.time_saved_estimate_s` field accumulates the sum of `timing_exhaustive_s` for all cache hits — this is the estimated CPU time saved by precomputation.
