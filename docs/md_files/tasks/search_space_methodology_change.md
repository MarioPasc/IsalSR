# Methodology Change: Search Space Reduction Experiment

**Date**: 2026-03-21
**Authors**: Mario Pascual Gonzalez, with AI assistance
**Status**: Implemented, pending Picasso execution

---

## 1. Problem Statement

The original search space experiment (`experiments/scripts/search_space_analysis.py`)
generated 1000 random IsalSR strings, parsed them into DAGs, canonicalized each,
and measured how many distinct canonical forms appeared per complexity bin (internal
node count k). This approach was fundamentally flawed for validating the paper's
central O(k!) claim:

**Why random sampling fails at this task:**

- The IsalSR string space is astronomically larger than the number of structurally
  unique DAGs. At k=8, there are 40,320 equivalent representations per expression,
  but the total string space contains billions of possible strings. The probability
  that two independently sampled random strings produce isomorphic DAGs is vanishingly
  small.
- Measured results: ~89% redundancy at k=1 (trivial expressions), ~36% at k=2, and
  effectively 0% at k>=4. Overall ~20% redundancy. This dramatically underestimates
  the true reduction potential.
- The experiment measures a **collision rate** (how often random strings happen to
  produce the same canonical form), not the **equivalence class size** (how many
  distinct representations exist per expression). These are fundamentally different
  quantities. The paper claims the equivalence class size is O(k!); the collision
  rate depends on sampling density and says almost nothing about this.

**Analogy**: Imagine proving that a hash function has 2^256 possible outputs by
randomly generating 1000 inputs and counting collisions. You'd find zero collisions,
which says nothing about the output space size. You need a different experiment design.

---

## 2. The New Approach: Controlled Permutation Analysis

### 2.1 Core Idea

Instead of randomly sampling and hoping for collisions, we **deliberately construct**
all equivalent representations of each expression and verify that canonicalization
collapses them all.

A labeled DAG D = (V, E, l, delta) with m variable nodes (IDs 0..m-1) and k internal
nodes (IDs m..m+k-1) admits k! distinct **node-numbering schemes**: permutations of
the internal node IDs that produce structurally distinct labeled DAGs, all encoding
the same mathematical expression.

The canonical string, if it is a complete isomorphism invariant, must map ALL k!
numbering schemes to the same string.

### 2.2 Algorithm

For each expression DAG D with k internal nodes:

1. **Compute canonical string** of D once: w* = pruned_canonical_string(D)
2. **Generate permutations**: all k! permutations if k <= 8 (k! <= 40,320); else
   sample 100,000 random permutations.
3. **For each permutation** pi of {0, 1, ..., k-1}:
   - Build D_pi = permute_internal_nodes(D, pi): creates an isomorphic copy where
     internal node m+i is moved to position m+pi(i), preserving all edges, labels,
     and operand order.
   - Compute a **structural fingerprint**: the tuple of (node_label, ordered_inputs)
     for each node in new-ID order. This is the adjacency representation of D_pi.
   - Compute the **greedy D2S string**: DAGToString(D_pi, initial_node=0).run()
   - Collect both in sets of distinct values.
4. **Verify canonical invariance** on a subset of 100 permutations:
   - Compute pruned_canonical_string(D_pi) and check it equals w*.
5. **Record metrics**.

### 2.3 What We Measure

Three metrics, each capturing a different facet of the reduction:

**Metric 1: n_distinct_representations (structural fingerprint count)**

This is the PRIMARY metric. The structural fingerprint encodes the full adjacency
structure of the labeled DAG, including node IDs, labels, and operand order:

```python
fingerprint = tuple(
    (dag.node_label(v).value, tuple(dag.ordered_inputs(v)))
    for v in range(dag.node_count)
)
```

Two permuted DAGs D_pi1 and D_pi2 produce the same fingerprint if and only if
pi1 and pi2 differ by an automorphism of D. Therefore:

    n_distinct_representations = k! / |Aut(D)|

where |Aut(D)| is the size of the automorphism group of D.

**Why this is NOT an "invented" metric:**
- This is the standard adjacency representation of a labeled graph.
- The count k!/|Aut(D)| follows from the Orbit-Stabilizer theorem in group theory.
- Each distinct fingerprint corresponds to a distinct labeled DAG that any
  node-ID-aware algorithm treats as a different object.
- The canonical string collapses all k!/|Aut(D)| distinct objects to ONE string.
- For "generic" DAGs (no symmetry, |Aut(D)|=1), this equals exactly k!.

**Metric 2: n_distinct_d2s (greedy D2S string count)**

A SECONDARY metric providing a conservative lower bound. The greedy D2S algorithm
makes structurally-determined choices (minimum displacement cost) that are largely
insensitive to node IDs. Many permutations produce the same greedy D2S string
because the algorithm's decisions depend on topology, not on which integer labels
the nodes carry.

Empirically: n_distinct_d2s is 2-8% of k!, far below the true equivalence class
size. This metric is reported for completeness but should NOT be used to validate
the O(k!) claim.

**Metric 3: invariant_success_rate (canonical invariance verification)**

The VERIFICATION metric. For each tested permutation, we check whether the
canonical string of D_pi equals w*. A complete isomorphism invariant must achieve
100% success rate. Any failure would indicate a bug in the canonical string
algorithm.

### 2.4 Why the Structural Fingerprint Is the Right Primary Metric

The paper's claim is (Introduction, paragraph 2):

> "For a DAG with k internal nodes, there exist Theta(k!) distinct
> node-numbering schemes that all encode the same expression."

The structural fingerprint directly counts these distinct node-numbering schemes
(modulo automorphisms). It is:

1. **Mathematically exact** for exhaustive permutation runs (k <= 8): the count
   equals k!/|Aut(D)| by the Orbit-Stabilizer theorem.
2. **A lower bound** for sampled runs (k > 8): if we observe N distinct
   fingerprints in S samples, the true count is at least N and at most k!.
3. **Algorithm-independent**: it does not depend on any specific encoding algorithm
   (D2S, S2D, or canonical). It measures a property of the DAG itself.
4. **Honest about automorphisms**: DAGs with non-trivial symmetry groups naturally
   produce n_distinct < k!, and this is correctly attributed to the
   Orbit-Stabilizer theorem rather than presented as a limitation.

The greedy D2S metric would be the wrong choice because it conflates two effects:
the node-numbering equivalence (which we want to measure) and the greedy
algorithm's insensitivity to node IDs (which is an artifact of the D2S
implementation, not a property of the search space).

---

## 3. Results (Local Validation, k=1..8)

| k | k!     | DAGs tested | Distinct repr (median) | Ratio median | Invariant rate |
|---|--------|-------------|------------------------|--------------|----------------|
| 1 | 1      | 5           | 1                      | 1.000        | 100%           |
| 2 | 2      | 10          | 2                      | 1.000        | 100%           |
| 3 | 6      | 10          | 6                      | 1.000        | 100%           |
| 4 | 24     | 10          | 24                     | 1.000        | 100%           |
| 5 | 120    | 10          | 120                    | 1.000        | 100%           |
| 6 | 720    | 10          | 720                    | 1.000        | 100%           |
| 7 | 5,040  | 5           | 5,040                  | 1.000        | 100%           |
| 8 | 40,320 | 5           | 40,320 (range: 20,160-40,320) | 0.900 | 100%       |

64 out of 65 DAGs achieve the full k! distinct representations. The one exception
at k=8 has |Aut(D)|=2 (a binary commutative operator with two identical subtrees),
giving n_distinct = k!/2 = 20,160.

Canonical invariance: **100% across all 65 DAGs and all tested permutations.**

---

## 4. Relationship to Journal Experiments (UDFS, Bingo)

The arXiv and journal experiments answer **different but complementary questions**:

### 4.1 What Each Experiment Measures

| Property | Permutation Analysis (arXiv) | Integration Experiments (Journal) |
|----------|------------------------------|-----------------------------------|
| **Question** | "How many equivalent representations EXIST per expression?" | "How many redundant evaluations does canonicalization SKIP in practice?" |
| **Scope** | Static structural analysis | During actual SR search (evolution/enumeration) |
| **Denominator** | k! permutations (all or sampled) | Total DAGs evaluated by SR algorithm |
| **Numerator** | Distinct structural fingerprints | Distinct canonical strings seen |
| **Algorithm bias** | None (all permutations equally weighted) | Yes (depends on which DAGs the search generates) |
| **What it proves** | Theoretical O(k!) bound is tight | Practical speedup from deduplication |

### 4.2 Why Both Are Needed

The permutation analysis answers: "CAN canonicalization provide O(k!) reduction?"
The answer is yes — each expression has k!/|Aut(D)| distinct representations, and
canonical invariance holds for 100% of tested permutations.

The integration experiments answer: "DOES canonicalization provide speedup in
practice?" This depends on the search algorithm:
- **Bingo** (stochastic GP): 41.6% redundancy in smoke tests — mutation/crossover
  frequently rediscover equivalent structures.
- **UDFS** (systematic enumeration): 6.15% redundancy — structured traversal
  encounters fewer redundancies than stochastic search.

The gap between theoretical potential (k!) and practical measurement (6-42%)
exists because GP/UDFS do not uniformly explore all k! numberings of each
expression. They have structural biases. But the THEORETICAL BOUND is what
makes the approach valuable: as search algorithms become more exploratory
(larger populations, more generations, broader mutation), they approach the
k! limit.

### 4.3 Should the Permutation Approach Be the "Pivotal Stone" of the Journal?

**No — but it should be a foundational validation section.**

The journal's main contribution is showing that IsalSR canonicalization
**integrates with and accelerates existing SR methods** (UDFS, Bingo). The
permutation analysis validates the theoretical claim but does not measure
actual speedup. The journal should include:

1. **Section: Theoretical Validation** — Permutation analysis proving the O(k!)
   bound is tight and the canonical invariant holds (same as arXiv experiment).
2. **Section: UDFS Integration** — Practical deduplication during systematic
   enumeration. Measures actual evaluations saved, wall-clock speedup.
3. **Section: Bingo Integration** — Practical deduplication during genetic
   programming. Measures redundancy rate, convergence speed, solution quality.

The permutation analysis provides the **theoretical foundation** that explains
WHY the integration experiments show improvement. Without it, the integration
results are empirical observations without a theoretical guarantee. With it, the
reader understands: "the canonical string eliminates up to k! equivalent
representations, and in practice UDFS/Bingo exploit X% of that potential."

### 4.4 What the Journal Experiments Already Do Correctly

The journal runners (`IsalSRUDFSRunner`, `IsalSRBingoRunner`) correctly:
- Measure `n_total`, `n_unique`, `n_skipped` during actual search
- Compute `empirical_reduction_factor = n_total / n_unique`
- Track `redundancy_rate = 1 - n_unique/n_total`
- Compare against `theoretical_reduction_bound = k!`
- Use hash-based deduplication (`set[int]`) for memory efficiency
- Apply paired statistical tests (Wilcoxon/t-test, Holm-Bonferroni)

These measurements are the right approach for the journal. The permutation
experiment should be added as a SEPARATE validation section, not as a
replacement for the integration experiments.

---

## 5. Implementation Summary

### Files Created (arXiv permutation experiment)

| File | Purpose |
|------|---------|
| `src/isalsr/core/permutations.py` | `permute_internal_nodes()` + `random_permutations()` |
| `tests/unit/test_permutations.py` | 15 unit tests (isomorphism, operand order, canonical invariance) |
| `experiments/scripts/search_space_permutation_analysis.py` | Main experiment script (CLI, CSV output) |
| `slurm/workers/search_space_permutation_slurm.sh` | SLURM array worker (1 task per k, k=1..15) |
| `experiments/scripts/generate_fig_search_space.py` | 2-panel publication figure |

### Files Modified

| File | Change |
|------|--------|
| `slurm/config.yaml` | Added `search_space_permutation` experiment section |
| `slurm/launch.sh` | Registered new experiment in EXPERIMENTS array |
| `.claude/CLAUDE.md` | Added `permutations.py` to key modules + experiment docs |

### How to Run

```bash
# Local quick test (k=3, 5 DAGs)
python experiments/scripts/search_space_permutation_analysis.py \
    --k-value 3 --n-dags 5 --num-vars 1 --output /tmp/test.csv

# Submit to Picasso (k=1..15, 100 DAGs each, 15 array tasks)
bash slurm/launch.sh --experiment search_space_permutation

# Generate figure (after data collection)
python experiments/scripts/generate_fig_search_space.py --data-dir <results_dir>
```

---

## 6. Lessons Learned

1. **The right experiment for a theoretical claim is a direct construction, not
   random sampling.** Random sampling measures collision rates, not equivalence
   class sizes. When the claim is about a structural property (k! equivalent
   representations), the experiment must directly construct those equivalences.

2. **The greedy D2S is not a good proxy for the number of distinct
   representations.** It's a deterministic algorithm whose choices depend on
   topology, not node IDs. The structural fingerprint (adjacency representation)
   is the correct proxy because it directly encodes what distinguishes two
   different node numberings.

3. **Automorphisms are a feature, not a bug.** DAGs with symmetry produce fewer
   than k! distinct representations, and the Orbit-Stabilizer theorem predicts
   exactly how many. The normalized ratio n_distinct/k! measures the "asymmetry"
   of the DAG. Generic DAGs achieve ratio 1.0; symmetric DAGs fall below. The
   canonical string handles both cases correctly.

4. **The arXiv and journal experiments serve different purposes.** The permutation
   analysis validates the theoretical bound; the integration experiments measure
   practical impact. Both are needed for a complete scientific story.
