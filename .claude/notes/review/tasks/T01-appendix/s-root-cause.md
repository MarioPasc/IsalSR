# S Root-Cause Analysis: T_canon Anatomy and Why S = 0.93 for Bingo

Investigation date: 2026-07-27
Primary sources: bingo/isalsr_runner.py, udfs/isalsr_runner.py, bingo/translator.py,
udfs/translator.py, bingo/config.py, experiments/models/analyze.py, 300 run_log.json files

---

## 1. Timer Anatomy

### Bingo (experiments/models/bingo/isalsr_runner.py)

**BEFORE T_canon timer — charged to T_search:**
- `dag = agraph_to_labeled_dag(indv)` — line 291 — runs on every individual before t0
- GC/malloc_trim every 5,000 individuals — lines 286-287 — before t0 is set

**T_canon STARTS:** line 301 `t0 = time.perf_counter()`

**INSIDE T_canon:**
- Atlas lookup (if atlas is not None) — lines 305-312
- `fast_canonical_string(dag, timeout=...)` — lines 319-333
- `hash(canonical)` — line 344 (executed BEFORE the stop at line 346)

**T_canon STOPS:** line 346 `self.dedup.canon_time_total += time.perf_counter() - t0`

**AFTER T_canon timer — charged to T_search (production path, enforce_dedup=False):**
- Set lookup: `if canon_hash in self.dedup.canonical_seen` — line 388
- Set insert: `self.dedup.canonical_seen.add(canon_hash)` — line 395
- Fitness evaluation: `self.fitness_function(indv)` — line 398

Population-level ops (_rebuild_population_set lines 172-188; population_canonicals lines 348-382)
are NOT active in production: `enforce_population_dedup=False` by default (config.py:36).

**T_search derived at run end:** line 519 `search_only = wall_clock - dedup.canon_time_total`
Run wall-clock starts at line 461 `t0 = time.perf_counter()` and stops at line 487.

---

### UDFS (experiments/models/udfs/isalsr_runner.py)

**BEFORE T_canon timer — charged to T_search** (in `wrap_evaluate_cgraph`):
- `labeled_dag = compgraph_to_labeled_dag(cgraph)` — line 137
- `_resolve_canonical_hash(labeled_dag)` called at line 153 (timer starts inside)

**T_canon STARTS** (inside `_resolve_canonical_hash`): line 95 `t0 = time.perf_counter()`

**INSIDE T_canon:**
- Atlas lookup (if atlas is not None) — lines 99-107
- `fast_canonical_string(labeled_dag, timeout=...)` — line 115

**T_canon STOPS:** line 126 `self.canon_time_total += time.perf_counter() - t0`

**AFTER T_canon timer — charged to T_search** (back in `wrap_evaluate_cgraph`):
- `hash(canonical)` — line 127 (returned by `_resolve_canonical_hash` AFTER timer stop)
- Set lookup: `if canon_hash in self.canonical_seen` — line 170
- Set insert: `self.canonical_seen.add(canon_hash)` — line 177
- Fitness evaluation: `self._original_evaluate(...)` — line 179

**T_search derived at run end:** line 277 `search_only = wall_clock - dedup.canon_time_total`

---

## 2. Inside / Outside Table

| Operation                        | Bingo: in T_canon?                         | UDFS: in T_canon?                          |
|----------------------------------|--------------------------------------------|--------------------------------------------|
| DAG conversion (adapter)         | OUTSIDE — line 291, before t0 at line 301  | OUTSIDE — line 137, before t0 at line 95   |
| fast_canonical_string            | INSIDE — lines 319-333                     | INSIDE — line 115                          |
| hash(canonical)                  | INSIDE — line 344, before stop at line 346  | OUTSIDE — line 127, after stop at line 126  |
| set lookup (canonical_seen)      | OUTSIDE — line 388                         | OUTSIDE — line 170                         |
| set insert (canonical_seen)      | OUTSIDE — line 395                         | OUTSIDE — line 177                         |
| population_canonicals ops        | OUTSIDE — inactive (enforce_dedup=False)   | N/A                                        |
| GC / malloc_trim                 | OUTSIDE — line 287, before t0 per call     | N/A                                        |
| Fitness evaluation               | OUTSIDE — line 398                         | OUTSIDE — line 179                         |

Critical corollary: `overhead_time_s = r.canonicalization_time_s` in both translators
(bingo/translator.py:118, udfs/translator.py:118). The paper's reported "overhead" equals
T_canon only. DAG conversion, set ops, and GC — all unique to IsalSR arm — are invisible
in the overhead figure but inflate T_search_IsalSR.

---

## 3. How S is Produced

T_search is DERIVED, not measured by a separate timer:
  Bingo line 519: search_only = wall_clock - dedup.canon_time_total
  UDFS line 277:  search_only = wall_clock - dedup.canon_time_total

For the baseline arm, canon_time_total = 0 by construction, so T_search_baseline = T_total_baseline.

S is computed per matched seed:
  sr = bl.time.wall_clock_search_only_s / t.wall_clock_search_only_s   [analyze.py:402]
then averaged across seeds per problem, then globally.

Any operation between the run's outer t0 (line 461) and wall_clock (line 487) that is NOT
accumulated into canon_time_total is automatically charged to T_search, including OS scheduling
jitter, Python GIL overhead, and all IsalSR bookkeeping outside the per-individual timer.

---

## 4. Termination Regime

Sample: 150 Bingo-baseline + 150 Bingo-IsalSR run_log.json files from
/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/model_validation/
real_benchmarks/wl_subtree/bingo (22 problems, ≤7 seeds each).

| Arm      | n   | Ceiling hits (T≥43140s) | Converged | T_total mean | T_total max | T_search mean |
|----------|-----|-------------------------|-----------|--------------|-------------|---------------|
| baseline | 150 | 0                       | 150       | 2,478 s      | 14,090 s    | ≈2,478 s      |
| isalsr   | 150 | 0                       | 150       | 7,561 s      | 42,338 s    | 5,214 s       |

0 of 300 sampled runs hit the 43,200 s ceiling. Both arms converge in every sampled run.
Ceiling effects do not distort S. The per-seed ratio S=0.93 (reported) is consistent with
easy problems (fast convergence both arms, ratio ≈ 1.0) dominating the mean of ratios while
hard problems (large absolute gap) dominate the aggregate mean difference.

---

## 5. Verdict: H1 / H2 / H3

### H1 — Bookkeeping is outside T_canon (charged to T_search)

Evidence FOR (confirmed from code):
- `agraph_to_labeled_dag` (Bingo line 291) and `compgraph_to_labeled_dag` (UDFS line 137) run
  on every individual before T_canon starts. Set lookup (Bingo:388, UDFS:170) and set insert
  (Bingo:395, UDFS:177) run after T_canon stops. GC/malloc_trim (Bingo:287) runs before t0.
- These operations are present in IsalSR but absent from baseline, directly inflating T_search_IsalSR.
- The overhead_time_s metric does NOT capture these costs (translator.py:118).

Evidence AGAINST:
- Per-call cost of `agraph_to_labeled_dag` is unknown from static reading. At ~0.05 ms/call on
  10M individuals, total ~500 s — plausible but unverified.

Note: A C++ port of fast_canonical_string alone does NOT fix H1. DAG conversion and set ops
remain Python and remain outside T_canon regardless of how fast canonicalization runs.

### H2 — Memory and cache pressure

Evidence FOR:
- canonical_seen (set[int]) grows to millions of entries (~280 MB at 10M × 28 B). Hash lookups
  on a structure exceeding L3 cache (~30 MB on Xeon Gold 6230R) cause random-access cache misses.

Evidence AGAINST:
- Switch from set[str] to set[int] already addressed OOM. Residual cache pressure is real but
  indistinguishable from H3 without per-generation profiling data.

### H3 — Deduplication changes search trajectory

Evidence FOR:
- T_search_IsalSR aggregate mean = 5,214 s vs baseline = 2,478 s (2.1× from aggregate means).
  T_total gap is 3× (7,561 vs 2,478 s).
- Duplicates receive fitness=inf, genetic_age=10,000,000 (lines 390-392, constant line 53),
  guaranteeing immediate removal by AgeFitnessEA Pareto dominance. Effective population per
  generation is reduced, potentially requiring more generations to satisfy convergence threshold.

Evidence AGAINST:
- Cannot confirm direction of causality from static reading alone. The 2.1× T_search gap is also
  consistent with H1 if `agraph_to_labeled_dag` costs ~0.27 ms/call (plausible but unverified).
- Easy problems contribute near-zero trajectory effect; the gap may be driven by a subset
  of harder problems.

### Ranking

H3 >= H1 >> H2

H1 is CONFIRMED from code (operations outside T_canon are real, present in both runners).
H3 is SUPPORTED by timing data (2.1× T_search gap too large to plausibly be pure bookkeeping
unless agraph_to_labeled_dag is expensive).
H2 is plausible but not separable from H3.

What the evidence cannot determine: the quantitative split between H1 and H3 requires
per-generation profiling with separate timers for (a) DAG conversion, (b) set ops, (c) all
other EA overhead. This data is not in the existing run logs.

H1 and H3 are not mutually exclusive — both contribute to S < 1.

---

## 6. Consequence for C++ Port of fast_canonical_string

T_search = T_total - T_canon is a mathematical identity (DERIVED not measured).
A C++ port of fast_canonical_string reduces T_canon by D per run and T_total by the same D
(same number of canonical calls, each faster):

  T_search_new = (T_total - D) - (T_canon - D) = T_total - T_canon = T_search_old

S is invariant to canonicalization speed. This follows from the accounting definition and
is independent of which hypothesis dominates. 0/300 sampled runs hit the ceiling, so
there is no ceiling-exploitation benefit from faster canonicalization either.

What a C++ port WOULD do:
- Reduce overhead_pct (T_canon/T_total): smaller T_canon, smaller fraction.
- Reduce T_total: faster runs, more cluster throughput.
- NOT affect S.

To improve S, one of two interventions is required:
1. If H1 dominates: port the full dedup pipeline (DAG conversion + canonicalization + set ops)
   as a single C extension and time the entire block as T_canon. This moves H1 costs from
   T_search into T_canon. This is a timer-boundary fix — does not change actual runtime
   but changes what S measures.
2. If H3 dominates: change dedup strategy (e.g., delay dedup to generation N>0, use
   age-gated dedup, compensate effective population size). This would require new experiments.

Neither intervention can be quantified from static reading alone.
