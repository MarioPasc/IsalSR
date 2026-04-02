# Amendments to IsalSR Experimental Design Document

**Reason**: Introduction of a precomputed canonical string cache that decouples canonicalization cost from SR experiment runtime.

---

## Amendment 1: New Section B.2.5 — Precomputation Phase

**Insert after Section B.2 (Independent Variables), before B.3 (Dependent Variables).**

### B.2.5. Precomputation Phase (Phase 0)

Before any SR experiment is executed, a **precomputed canonical string cache** is generated for each benchmark configuration `(m, Ops, k_max)`. This cache stores the output of all four D2S algorithms (greedy-single, greedy-min, pruned-exhaustive, exhaustive) for every DAG encountered during the experiments. The precomputation is performed once and reused across all seeds, SR methods, and reruns.

**Rationale**: The canonicalization step is a pure function of the DAG — it is independent of the random seed, training data, and SR method. Precomputing it eliminates redundant computation and, critically, **decouples the canonicalization overhead from the SR search time**, allowing cleaner measurement of both.

**Cache generation modes**:

| Mode | Scope | Seed-dependent? | Used for |
|------|-------|----------------|----------|
| Systematic enumeration | All DAGs with k ≤ k_max internal nodes | No | UDFS-style methods, small benchmarks |
| Random sampling | N_sample random IsalSR strings per seed | Yes (per seed) | Stochastic methods, large benchmarks |
| Lazy extension | DAGs not found in cache during runtime | Yes (per run) | Completeness guarantee |

**Storage format**: HDF5 with JSON sidecar (see companion document: `precomputed_cache_design.md`). Grouped by `(benchmark, num_variables)`, with datasets for raw strings, all four D2S outputs, DAG structural properties, per-algorithm timings, and cross-validation flags.

**Cache lookup during experiments**: The `CacheManager` provides O(1) lookup by raw IsalSR string. On a cache hit, all canonical forms and timings are returned without recomputation. On a miss, the DAG is canonicalized on-the-fly and the result is appended to the cache.

---

## Amendment 2: Modification to Section B.3.2 — Time Metrics

**Replace the existing Time metrics table with the following expanded version.**

#### B.3.2. Time

| Metric | Formula | Units | Interpretation |
|--------|---------|-------|----------------|
| **Wall-clock time (search only)** $T_{\text{search}}$ | Total elapsed minus canonicalization (cache-served) | seconds | Pure search performance |
| **Wall-clock time (total)** $T_{\text{total}}$ | Full elapsed including any cache misses | seconds | End-to-end time |
| **Time to threshold** $T_\tau$ | First time $t$ s.t. $R^2(t) \geq \tau$ | seconds | Time to reach $R^2 = 0.99$ |
| **Canonicalization overhead (precomputed)** $T_{\text{canon}}^{\text{pre}}$ | Sum of `timing_exhaustive_s` from cache for all DAGs explored | seconds | What canonicalization *would have cost* without cache |
| **Canonicalization overhead (runtime)** $T_{\text{canon}}^{\text{rt}}$ | Actual time spent on cache misses | seconds | Residual canonicalization cost |
| **Cache hit rate** | $\text{hits} / (\text{hits} + \text{misses})$ | $[0, 1]$ | Cache effectiveness |
| **Estimated time saved** | $T_{\text{canon}}^{\text{pre}} - T_{\text{canon}}^{\text{rt}}$ | seconds | CPU time saved by precomputation |
| **Speedup** | $S = T_{\text{search}}^{\text{baseline}} / T_{\text{search}}^{\text{IsalSR}}$ | dimensionless | >1 means IsalSR is faster |

**Key distinction**: With the precomputed cache, the **primary time metric becomes $T_{\text{search}}$** (search-only time), which excludes canonicalization served from cache. This is the fairest comparison because:

- The baseline method does not perform canonicalization at all.
- IsalSR's canonicalization is a one-time offline cost, amortized across all experiments.
- $T_{\text{canon}}^{\text{pre}}$ is reported separately so reviewers can assess the full cost.

However, we also report $T_{\text{total}}$ for transparency, and $T_{\text{canon}}^{\text{pre}}$ (the hypothetical cost without cache) to show what the overhead *would* be in a non-cached deployment.

---

## Amendment 3: Modification to Section B.4 — Experimental Protocol

**Replace step 2b with the following.**

2b. **For each random seed $s \in \{1, \ldots, 30\}$**:
   - **Load the precomputed cache** for the current `(benchmark, m, Ops)` configuration.
   - Run $M$(Baseline) on $(X_{\text{train}}, y_{\text{train}})$ → record all metrics.
   - Run $M$(IsalSR) on $(X_{\text{train}}, y_{\text{train}})$ **with cache lookup** → record all metrics. The `CacheManager` handles canonical string retrieval transparently.
   - Evaluate best expression on $(X_{\text{test}}, y_{\text{test}})$.
   - **Flush the cache** (persist any runtime extensions).
   - **Log cache statistics** (hit rate, misses, estimated time saved).

**Insert new step 0 before step 1.**

0. **Precompute the canonical string cache** (Phase 0):
   a. For each benchmark configuration `(m, Ops, k_max)`, run `generate_cache.py` in systematic mode.
   b. Optionally, run in sampled mode with seeds {1, ..., 30} to pre-warm the cache for stochastic methods.
   c. Validate the cache: verify `exhaustive == pruned` for 100% of entries, run round-trip tests.
   d. Record the precomputation wall-time and store in `metadata.json`.

---

## Amendment 4: Modification to Section C.6 — Output Folder Structure

**Add the `precomputed/` subtree to the folder structure.**

```
results/
├── precomputed/                            # ← NEW: Precomputed cache
│   ├── README.md
│   ├── generate_cache.py
│   ├── cache_lookup.py
│   ├── nguyen/
│   │   ├── nguyen_1var.h5
│   │   ├── nguyen_1var.json
│   │   ├── nguyen_2var.h5
│   │   └── nguyen_2var.json
│   ├── feynman/
│   │   ├── feynman_{m}var.h5
│   │   └── feynman_{m}var.json
│   └── srbench/
│       ├── srbench_{m}var.h5
│       └── srbench_{m}var.json
│
├── config.yaml
├── metadata.json                           # ← MODIFIED: add precomputation metadata
│   ...existing structure unchanged...
```

---

## Amendment 5: Modification to Section C.7 — `run_log.json` Schema

**Add the following fields to the `results.time` object in `run_log.json`.**

```json
{
  "results": {
    "time": {
      "wall_clock_total_s": 142.5,
      "wall_clock_search_only_s": 130.2,
      "canonicalization_precomputed_s": 12.3,
      "canonicalization_runtime_s": 0.0,
      "cache_hit_rate": 1.0,
      "cache_hits": 250000,
      "cache_misses": 0,
      "estimated_time_saved_s": 12.3,
      "time_to_r2_099_s": 45.2,
      "time_to_r2_0999_s": 98.7,
      "evaluation_time_s": 110.4,
      "overhead_time_s": 19.8
    }
  }
}
```

**Field definitions**:

| Field | Description |
|-------|-------------|
| `wall_clock_total_s` | Full elapsed time including cache misses |
| `wall_clock_search_only_s` | Elapsed time minus all canonicalization (cache-served + runtime) |
| `canonicalization_precomputed_s` | Sum of `timing_exhaustive_s` from cache entries for all explored DAGs |
| `canonicalization_runtime_s` | Actual wall-clock spent on cache misses (on-the-fly canonicalization) |
| `cache_hit_rate` | Fraction of lookups served from cache |
| `cache_hits` | Number of cache hits |
| `cache_misses` | Number of cache misses |
| `estimated_time_saved_s` | `canonicalization_precomputed_s - canonicalization_runtime_s` |

---

## Amendment 6: Modification to Section C.7 — `trajectory.csv` Schema

**Add one column.**

| Column | Type | Description |
|--------|------|-------------|
| ... | ... | ...(existing columns unchanged)... |
| `cache_hit_rate_cumulative` | float | Cumulative cache hit rate up to this iteration |

---

## Amendment 7: Addition to Section C.5 — Summary Statistics

**Add to the summary statistics table.**

| Statistic | Description |
|-----------|-------------|
| ...(existing rows unchanged)... |
| $T_{\text{canon}}^{\text{pre}}$ | Precomputed canonicalization time (what it would have cost) |
| Cache hit rate | Fraction of canonical lookups served from cache |
| Precomputation wall-time | One-time cost to generate the cache (reported in `metadata.json`) |

---

## Amendment 8: Addition to Section D — Summary Table

**Add one row to the summary table.**

| Metric | Axis | Per-run storage | Statistical test | Effect size | Goal direction |
|--------|------|----------------|-----------------|-------------|----------------|
| ...(existing rows unchanged)... |
| Cache hit rate | Infrastructure | `run_log.json` | Descriptive only | — | Higher = better coverage |
| Precomputed canon. time | Time | `run_log.json` | Descriptive only | — | Understand offline cost |

---

## Amendment 9: Addition to Section E — Recommended Figures

**Add one figure.**

8. **Cache effectiveness plot**: Cache hit rate vs. problem complexity (k), showing that the systematic enumeration covers most of the search space for small DAGs and that the hit rate remains high even for stochastic methods.

---

## Amendment 10: New Section B.7 — Reporting the Precomputation Cost

In the paper, the precomputation cost must be reported transparently. We propose:

- **Table in the paper**: "Precomputation cost per benchmark configuration", with columns: benchmark, m, |Ops|, k_max, N_entries, N_unique, generation time (hh:mm:ss), file size (MB).
- **Discussion paragraph**: Argue that the precomputation is a one-time offline cost, amortized across all experiments and future work. Report the amortized cost per experiment run (= precomputation time / total number of runs).
- **Fair comparison note**: The primary time comparison uses $T_{\text{search}}$ (search-only), but $T_{\text{total}}$ (including cache misses) is also reported. The baseline method's time includes no canonicalization overhead, so the comparison via $T_{\text{search}}$ is the fairest.
