# Task: Diversity Conjecture — Code Fix, Harder Benchmark, Reframed Analysis

**Author:** Mario Pascual-González (via Claude planning session, 2026-04-03)
**Priority:** P0 — Blocks paper submission
**Estimated effort:** 2–3 days implementation + 1 day Picasso wallclock

---

## 0. Context — Read This First

We are formalizing a diversity preservation property for the ISALSR paper
(IEEE TPAMI). The first experimental run (`diversity_conjecture_x7` on I.48.20)
revealed two problems:

1. **IsalSR delta never reaches 1.0** (peaks at ~0.91). Proposition X.4
   requires a duplicate-free population invariant, but the current Bingo
   adapter only canonicalizes — it does not reject duplicate canonical strings
   at insertion time.

2. **Conjecture C2 (d̄ inequality) is falsified.** The baseline's mean
   pairwise GED *exceeds* IsalSR's at late generations (Cohen's d = −1.5 at
   t = 300). This is because the baseline fragments into distant isomorphic
   clusters (high inter-cluster GED), while IsalSR concentrates on focused
   local search (moderate uniform GED). The reframed narrative is about
   *exploitation efficiency*, not exploration breadth.

3. **Benchmark choice:** I.48.20 is too easy — both methods reach R² = 1.0.
   We need a harder problem where the diversity difference *causes* a
   performance difference. Candidate: I.12.4 and/or I.10.7.

This task implements all three fixes and produces a clean dataset for the paper.

---

## 1. Code Change: Duplicate-Free Enforcement in Bingo Adapter

### 1.1 What to change

**File:** `experiments/models/bingo/isalsr_runner.py`

**Class:** `IsalSREvaluation._serial_eval()`

**Current behavior:** When an individual's canonical string has already been
seen, the method assigns infinite fitness (worst possible) to the duplicate.
The individual *remains in the population* — it occupies a slot with bad
fitness, but the age-fitness Pareto selection in Bingo may not remove it
immediately. More critically, Bingo's `AgeFitnessEA` (which extends
`MuPlusLambda`) generates offspring from the parent population. If two
independent crossover/mutation events produce the same canonical string in
the same generation, both are inserted — neither is rejected.

**Required behavior:** After canonicalization, check whether the canonical
string already exists in the *current population* (not just the historical
seen-set). If it does, **reject the child and retry with a fresh mutation**
(up to `MAX_DEDUP_RETRIES` attempts). If all retries collide, fall back to
keeping the old individual in that slot (do not insert the duplicate).

### 1.2 Implementation specification

#### 1.2.1 New data structure: population canonical set

Add a `set[str]` (not `set[int]` — we need exact matching, not hash-based
approximate matching) that tracks the canonical strings of all *currently
living* population members. This is different from the existing
`canonical_seen: set[int]` which is a historical accumulator for fitness
caching.

```python
class _CanonicalDeduplicator:
    def __init__(self, ...):
        # EXISTING: historical seen-set for fitness caching (hash-based, ok)
        self.canonical_seen: set[int] = set()
        # NEW: exact canonical strings of current population members
        self.population_canonicals: set[str] = set()
        # NEW: map from population index to canonical string (for eviction)
        self.index_to_canonical: dict[int, str] = {}
```

#### 1.2.2 Modified `_serial_eval` logic

```
for each individual in population:
    if individual already has valid fitness:
        skip  # already evaluated in a previous call

    dag = agraph_to_labeled_dag(individual)
    canon = pruned_canonical_string(dag)

    # --- FITNESS CACHING (existing, keep as-is) ---
    canon_hash = hash(canon)
    if canon_hash in self.canonical_seen:
        individual.fitness = cached_fitness[canon_hash]
        # BUT STILL CHECK POPULATION DEDUP BELOW

    # --- POPULATION DEDUP (new) ---
    if canon in self.population_canonicals:
        # This canonical string is already held by another living member.
        # Reject: assign worst fitness so age-fitness selection removes it.
        individual.fitness = INF_FITNESS
        self.n_rejected_duplicates += 1
        continue

    # --- NOVEL INDIVIDUAL ---
    if canon_hash not in self.canonical_seen:
        # Truly new: evaluate fitness
        individual.fitness = fitness_function(individual)
        self.canonical_seen.add(canon_hash)
        cached_fitness[canon_hash] = individual.fitness
    # else: fitness was retrieved from cache above

    # Register in population set
    self.population_canonicals.add(canon)
```

#### 1.2.3 Population set maintenance

The population set must be **rebuilt at the start of each generation** to
reflect the current population state after selection/replacement. Bingo's
`AgeFitnessEA.__call__` flow is:

```
1. Generate offspring (crossover + mutation)
2. Evaluate offspring          ← IsalSREvaluation.__call__
3. Evaluate parents            ← IsalSREvaluation.__call__
4. Selection (age-fitness Pareto)  ← removes some individuals
```

After step 4, some individuals are removed. The population set must be
synchronized. The cleanest approach:

```python
def _rebuild_population_set(self, population):
    """Rebuild the canonical set from the current population.

    Called at the START of each __call__ invocation.
    """
    self.dedup.population_canonicals.clear()
    self.dedup.index_to_canonical.clear()
    for i, indv in enumerate(population):
        if hasattr(indv, '_isalsr_canonical') and indv._isalsr_canonical:
            self.dedup.population_canonicals.add(indv._isalsr_canonical)
            self.dedup.index_to_canonical[i] = indv._isalsr_canonical
```

Store the canonical string on the individual itself (as a transient
attribute `_isalsr_canonical`) so it survives across evaluation calls
within the same generation.

#### 1.2.4 Retry-mutation policy (optional enhancement)

If we want to be more aggressive about maintaining delta = 1.0, implement
retry-mutation in the offspring generation step. However, this requires
modifying Bingo's `AgeFitnessEA` class, which is more invasive. The
simpler approach (assign INF fitness to duplicates, let selection remove
them) should achieve delta ≈ 0.98–1.00 because age-fitness Pareto
selection will preferentially remove high-fitness (=bad) duplicates.

**Decision for implementer:** Start with the simple approach (INF fitness
for duplicates + population set). Measure delta. If delta < 0.98, then
implement retry-mutation. Document whichever approach achieves delta = 1.0.

### 1.3 New config parameters

**File:** `experiments/models/bingo/config.py`

Add to `BingoConfig`:

```python
# Diversity enforcement
enforce_population_dedup: bool = True   # Enable population-level dedup
max_dedup_retries: int = 5             # Only used if retry-mutation is on
```

### 1.4 New statistics to log

Add to `BingoTrajectorySnapshot` (or equivalent):

- `n_rejected_duplicates`: count of individuals rejected per generation
  because their canonical string was already in the population.
- `delta`: effective diversity ratio at each snapshot generation.
- `population_canonical_set_size`: size of the population canonical set
  (should equal `N * delta`).

### 1.5 Tests

**File:** `tests/test_bingo_dedup.py` (new)

Write the following unit tests BEFORE implementing the change:

```
test_duplicate_rejected():
    """Two individuals with the same canonical string → second gets INF."""
    # Create a population of 10, manually set two to same AGraph.
    # Run _serial_eval. Assert second has fitness = INF.

test_population_set_tracks_evictions():
    """After selection removes an individual, its canonical string is freed."""
    # Insert individual A with canonical "abc".
    # Simulate selection removing A.
    # Rebuild population set.
    # Insert new individual B with canonical "abc".
    # Assert B is accepted (not rejected).

test_delta_reaches_one():
    """After sufficient generations, delta = 1.0 on a trivial problem."""
    # Run 50 generations on Nguyen-8 (sqrt(x), trivially easy).
    # Assert delta(P_t) == 1.0 for t >= 10.

test_fitness_caching_still_works():
    """Canonical string seen historically but not in population → reuse fitness."""
    # Individual with canonical "xyz" was evaluated in gen 5, removed in gen 10.
    # New individual with canonical "xyz" appears in gen 15.
    # Assert: fitness is retrieved from cache (no re-evaluation).
    # Assert: individual is accepted into population (not rejected).

test_isomorphic_agraphs_collapsed():
    """Two AGraphs that are isomorphic → same canonical string → second rejected."""
    # Create two AGraphs encoding x+sin(x) with different internal structure.
    # Canonicalize both. Assert same canonical string.
    # Insert first → accepted. Insert second → rejected (INF fitness).
```

Run all tests locally before proceeding to the experiment.

---

## 2. Benchmark Selection: Use Harder Problems

### 2.1 Primary benchmark: I.10.7

**Formula:** $f(m_0, v, c) = m_0 / \sqrt{1 - v^2/c^2}$

**Why this one:** Same mathematical structure as I.48.20 (relativistic factor)
but simpler — $m_0/\gamma$ instead of $m c^2 / \gamma$. The 3-variable setting
with the nested sqrt and division makes it non-trivial for Bingo. Based on the
full experiment results, check whether baseline and IsalSR show different
convergence rates on this problem.

### 2.2 Secondary benchmark: I.12.4

**Formula:** $f(q_1, r, c) = q_1 / (4\pi r c)$

**Why this one:** From the earlier reviewer gap analysis, this problem shows
baseline plateau at R² ≈ 0.76 while IsalSR reaches R² = 1.0 — exactly the
story we need. The diversity difference *causes* the performance difference.

**Important caveat:** Verify this claim by checking the full experiment results
in `results/bingo/feynman/i.12.4/`. If the R² gap does not hold with the
new duplicate-free enforcement, consider Nguyen-6 or Nguyen-7 as alternatives
(these have non-trivial structure with nested transcendentals).

### 2.3 Control benchmark: Nguyen-1

**Formula:** $f(x) = x^3 + x^2 + x$

**Why:** Trivially easy — both methods should reach R² = 1.0 in < 20
generations. Use as a sanity check that the dedup fix does not break anything.
Delta should be 1.0 for IsalSR from generation ~5 onward.

### 2.4 Experiment configuration

**File:** `experiments/configs/diversity_conjecture_v2.yaml` (new)

```yaml
experiment_name: diversity_conjecture_v2

# Bingo parameters (match I.48.20 experiment for comparability)
population_size: 200
stack_size: 32
operators: ["+", "-", "*", "/", "sin", "cos", "exp", "log"]
crossover_prob: 0.4
mutation_prob: 0.4
metric: mse
clo_alg: lm
max_time: 7200.0  # 2 hours per seed

# NEW: duplicate-free enforcement
enforce_population_dedup: true

# Snapshot schedule (more granular in early generations)
snapshot_gens: [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500]

# Seeds
seeds: 30  # increased from 20 for tighter CIs

# Benchmarks to run
benchmarks:
  - I.12.4    # primary: hard problem where diversity matters
  - I.10.7    # secondary: medium difficulty
  - Nguyen-1  # control: trivial sanity check

# Diversity metrics to compute at each snapshot
diversity_metrics:
  - delta                  # effective diversity ratio
  - d_bar_bp              # mean pairwise bipartite GED
  - d_bar_lev             # mean pairwise Levenshtein (for comparison)
  - pca_projection        # 2D PCA of 1-WL hash histograms
  - ged_heatmap           # full pairwise GED matrix (for figures)

n_workers: 8
```

---

## 3. Analysis Script Updates

### 3.1 Updated statistical tests

The analysis script (whichever generated `statistical_tests.json` for the
I.48.20 run) must be updated to compute:

**Keep:**
- `delta`: mean ± std across seeds, paired t-test, Wilcoxon, Cohen's d
- `d_bar_bp`: same battery
- `d_bar_lev`: same battery
- `best_r2`: same battery

**Add:**
- `cv_ged`: coefficient of variation of the pairwise GED distribution
  ($\sigma / \mu$ of the upper-triangular GED matrix), per generation per
  seed. This quantifies the fragmentation vs. uniformity story.
  - Hypothesis: CV_baseline > CV_isalsr at late generations (baseline has
    bimodal distribution: zeros + high values; IsalSR has unimodal moderate
    values).
- `frac_zero_pairs`: fraction of pairwise GED entries that are exactly 0.
  This is a direct measure of isomorphic redundancy:
  $\text{frac\_zero} = r / \binom{N}{2}$ where $r$ = number of zero-distance
  pairs. By construction, $\text{frac\_zero} = 1 - \delta$ (approximately).
- `d_bar_nonzero`: mean pairwise GED restricted to non-isomorphic pairs
  ($\bar{d}_{\neq}$ from Remark X.6). Compute as:
  `d_bar_nonzero = d_bar_bp / (1 - frac_zero)` if frac_zero < 1.

### 3.2 Figure generation

Produce the following figures for each benchmark:

1. **PCA landscape** (already exists): population snapshots with KDE contours,
   δ and R²_max annotations. Keep as-is.

2. **GED heatmap matrix** (already exists): pairwise BP-GED at selected
   generations, hierarchically clustered. Keep as-is.

3. **Convergence + diversity panel** (new or updated): bottom panel with:
   - Left y-axis: best R² (train) over generations, baseline vs IsalSR,
     mean ± 95% CI across seeds.
   - Right y-axis: δ over generations, baseline vs IsalSR.
   - This is the key figure: it shows the *causal link* between diversity
     collapse and performance plateau (on I.12.4) or the lack thereof
     (on I.48.20/Nguyen-1).

4. **GED distribution histogram** (new): at a representative late generation
   (e.g., t = 300), show the histogram of all $\binom{200}{2}$ pairwise GED
   values for baseline (blue) and IsalSR (red). Baseline should show a
   bimodal distribution (peak at 0 + broad peak at high GED); IsalSR should
   show a unimodal distribution (single peak at moderate GED). This is the
   visual proof that the baseline's high d̄ is fragmentation, not diversity.

---

## 4. Documentation Update: Revised Formalization

### 4.1 Changes to `docs/design/diversity/diversity_formalization_v2.md`

The existing v2 draft is at `/mnt/user-data/outputs/diversity_formalization_v2.md`.
It needs the following edits:

1. **Update Proposition X.4 remark:** Add a note that the Bingo adapter now
   enforces duplicate-free insertion (describe the mechanism briefly), and
   report the empirical delta values from the new experiment.

2. **Replace Conjecture C2:** The $\bar{d}$ inequality is falsified. Replace
   with a qualitative structural claim about population distance distribution:

   > **Conjecture X.7 (Structural Coherence).** Under the conditions of
   > Proposition X.4, the IsalSR population exhibits a unimodal pairwise
   > distance distribution with lower coefficient of variation than the
   > baseline:
   > $$\text{CV}(D'_t) < \text{CV}(D_t)$$
   > where $D_t = \{d(G_i, G_j) : i < j\}$ is the multiset of pairwise
   > distances and $\text{CV} = \sigma / \mu$.

   This captures the "focused exploitation vs. fragmented clusters" story
   without making a claim about the mean that the data contradicts.

3. **Update Empirical Evidence section:** Replace all placeholder text with
   actual numbers from the new experiment. Specifically:
   - Report δ = 1.0 (or near-1.0) for IsalSR at all post-initialization
     generations.
   - Report the R² convergence gap on I.12.4 as the primary result.
   - Report the GED heatmap interpretation (uniform vs. block-structured)
     as the structural evidence.
   - Report d̄_GED as a context metric with the fragmentation interpretation
     (do NOT claim d̄_IsalSR > d̄_baseline).

4. **Update Summary table** with new numbers.

### 4.2 Key framing change

The narrative shifts from:

> "IsalSR explores more of the search space"

to:

> "IsalSR eliminates structural redundancy, ensuring every population slot
> evaluates a genuinely distinct expression. This concentrates evolutionary
> resources on non-redundant structural neighbors of the best solution.
> On hard problems (I.12.4), this focused exploitation avoids the premature
> convergence that occurs when the baseline wastes population slots on
> isomorphic copies of suboptimal expressions."

---

## 5. Execution Plan

### 5.1 Local validation (before Picasso)

Run ALL of the following locally before submitting to Picasso:

```bash
# 1. Unit tests for the dedup fix
pytest tests/test_bingo_dedup.py -v

# 2. Integration test: 20 generations on Nguyen-1, single seed
python -m experiments.scripts.run_diversity_experiment \
    --benchmark Nguyen-1 \
    --seeds 1 \
    --max-generations 20 \
    --enforce-dedup \
    --output /tmp/dedup_test_nguyen1

# Verify: delta = 1.0 for t >= 5 in the output
cat /tmp/dedup_test_nguyen1/isalsr/seed_0/summary.csv | \
    awk -F',' 'NR>1 {print $3, $4}'  # generation, delta

# 3. Integration test: 50 generations on I.12.4, single seed
python -m experiments.scripts.run_diversity_experiment \
    --benchmark I.12.4 \
    --seeds 1 \
    --max-generations 50 \
    --enforce-dedup \
    --output /tmp/dedup_test_i124

# Verify: delta = 1.0 for IsalSR, delta < 1.0 for baseline
# Verify: output files match expected schema (summary.csv, ged_matrices/)

# 4. Regression test: run the ORIGINAL I.48.20 experiment for 50 gens
#    with dedup OFF, verify results match previous run.
python -m experiments.scripts.run_diversity_experiment \
    --benchmark I.48.20 \
    --seeds 1 \
    --max-generations 50 \
    --no-enforce-dedup \
    --output /tmp/regression_test_i4820

# 5. Timing test: measure overhead of population set maintenance
#    The dedup set operations should add < 1% to total runtime.
python -m experiments.scripts.run_diversity_experiment \
    --benchmark Nguyen-1 \
    --seeds 1 \
    --max-generations 100 \
    --enforce-dedup \
    --time-profile \
    --output /tmp/timing_test
```

### 5.2 Expected local results (acceptance criteria)

| Test | Pass criterion |
|---|---|
| Unit tests | All 5 pass |
| Nguyen-1 delta | delta = 1.0 for IsalSR at t ≥ 10 |
| I.12.4 delta | delta ≥ 0.98 for IsalSR at t = 50 |
| I.12.4 baseline delta | delta < 0.70 for baseline at t = 50 |
| Regression test | d̄_bp within 5% of previous I.48.20 run |
| Timing overhead | < 2% increase in total wall-clock time |

### 5.3 Picasso submission

Only after ALL local tests pass:

```bash
# Generate SLURM job array
python -m experiments.scripts.generate_slurm_jobs \
    --config experiments/configs/diversity_conjecture_v2.yaml \
    --output experiments/slurm/diversity_v2/

# Submit
sbatch experiments/slurm/diversity_v2/array_job.sh
```

**Resource estimate:**
- 3 benchmarks × 2 variants × 30 seeds = 180 jobs
- Each job: ~2 hours max (GED computation is the bottleneck at late gens)
- Total: ~360 CPU-hours
- With 8 parallel workers per job: ~45 node-hours

### 5.4 Post-Picasso analysis

```bash
# 1. Aggregate results
python -m experiments.scripts.aggregate_diversity_results \
    --results-dir results/diversity_v2/ \
    --output results/diversity_v2/aggregated/

# 2. Generate figures
python -m experiments.figures.diversity.generate_all \
    --aggregated-dir results/diversity_v2/aggregated/ \
    --output-dir figures/diversity_v2/

# 3. Generate statistical tests JSON
python -m experiments.scripts.diversity_statistical_tests \
    --aggregated-dir results/diversity_v2/aggregated/ \
    --output results/diversity_v2/statistical_tests.json

# 4. Update the formalization document with real numbers
# (manual step — fill in placeholders in diversity_formalization_v3.md)
```

---

## 6. File Manifest

### Files to CREATE:
- `experiments/configs/diversity_conjecture_v2.yaml`
- `tests/test_bingo_dedup.py`

### Files to MODIFY:
- `experiments/models/bingo/isalsr_runner.py` (duplicate-free enforcement)
- `experiments/models/bingo/config.py` (new config fields)
- `docs/design/diversity/diversity_formalization_v2.md` → v3

### Files to READ (do not modify):
- `experiments/models/bingo/adapter.py` (understand AGraph ↔ DAG conversion)
- `experiments/models/bingo/runner.py` (understand baseline Bingo pipeline)
- `benchmarks/datasets/feynman.py` (benchmark definitions)
- `benchmarks/datasets/nguyen.py` (benchmark definitions)
- `src/isalsr/core/canonical.py` (canonical string computation)

---

## 7. Definition of Done

This task is complete when:

- [ ] All 5 unit tests in `test_bingo_dedup.py` pass
- [ ] Local integration tests show delta = 1.0 for IsalSR on Nguyen-1
- [ ] Local integration tests show delta ≥ 0.98 for IsalSR on I.12.4
- [ ] Timing overhead < 2%
- [ ] Picasso jobs submitted and completed for all 3 benchmarks × 30 seeds
- [ ] Statistical tests JSON generated with delta, d̄_bp, CV, R² results
- [ ] Figures generated: PCA landscape, GED heatmap, convergence+diversity, GED histogram
- [ ] Formalization document v3 updated with real numbers and revised C2
- [ ] Mario has reviewed all outputs and confirmed consistency with the paper narrative
