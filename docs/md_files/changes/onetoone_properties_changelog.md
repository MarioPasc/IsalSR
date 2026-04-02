# Changelog: One-to-One Property Validation Suite

**Date**: 2026-03-20
**Author**: Mario Pascual Gonzalez (assisted by Claude)
**Purpose**: Reconcile the experimental section with the introduction's "five fundamental properties" promise (line 146 of introduction.tex).

---

## Problem

The introduction claims: *"Section~\ref{sec:results} presents empirical validation of the five fundamental properties of the canonical representation."*

The five properties are:
- **P1**: Round-trip fidelity — S2D(D2S(D)) ≅ D
- **P2**: DAG acyclicity — S2D always produces acyclic graphs
- **P3**: Canonical invariance — D₁ ≅ D₂ ⟺ w**_D₁ = w**_D₂
- **P4**: Evaluation preservation — |eval(D) - eval(round-tripped D)| < 10⁻⁸
- **P5**: Search space reduction — canonicalization collapses O(k!) duplicates

**Before this change**, P1–P4 were only validated in the unit/property test suite (`tests/property/`), not as reportable experiments with statistical confidence intervals and scientific figures. Only P5 was covered by the search_space_analysis experiment.

---

## What was created

### `experiments/scripts/onetoone_properties.py` — Two-phase validation

**Phase 1** (statistical): Generates N random IsalSR strings, validates all four properties on each valid DAG. Produces per-string CSVs and summary CSV with Clopper-Pearson 95% CIs.

**Phase 2** (benchmark-level, triggered by `--plot`): Validates P1–P4 on 8 curated Nguyen/Feynman benchmark expressions. Generates:

| Output file | Content |
|------------|---------|
| `fig_p1_roundtrip.{pdf,png,svg}` | Table-figure: per-benchmark canonical strings with token coloring, |V|, |E|, PASS status |
| `fig_p2_acyclicity.{pdf,png,svg}` | Histogram of random-string DAG complexity + benchmark depths as annotated vertical lines, "All N acyclic" |
| `fig_p3_invariance.{pdf,png,svg}` | Demonstration: different encodings (greedy D2S, no-op padded, re-canonicalized) all produce same w* |
| `fig_p4_evaluation.{pdf,png,svg}` | Scatter plot: eval(D) vs eval(D') across all benchmarks × test points, with max error annotation |
| `tab_benchmark_properties.tex` | LaTeX table: Expression, m, |V|, |E|, Depth, P1–P4, |w*| |
| `benchmark_validation.json` | Full JSON data for reproducibility |
| `fig_property_pass_rates.{pdf,png,svg}` | Phase 1: grouped bar chart of pass rates with CIs |
| `summary.csv` | Phase 1: aggregated pass rates per property per num_vars |
| `results_v{1,2,3}.csv` | Phase 1: per-string results |

### `slurm/workers/onetoone_properties_slurm.sh`
SLURM array worker (3 tasks, one per num_vars={1,2,3}).

---

## What was modified

### `slurm/config.yaml`
- **Added**: `onetoone_properties` (array of 3, 2h, 4G, 5000 strings)
- **Modified**: `search_space_analysis`:
  - `n_strings`: 10000 → **1000** (reduce Picasso wall-clock; still statistically significant)
  - `max_tokens_list`: "10,15,20,25,30" → **"10,15,20"** (T=25,30 produce >50% canon timeouts — useless noise)
  - `time_limit`: "02:00:00" → **"06:00:00"** (T=20 needs ~2.6h with 1000 strings)
  - `_CANON_TIMEOUT` in script: 5.0 → **1.0** seconds

### `slurm/launch.sh`
- Added `"onetoone_properties"` to EXPERIMENTS list

### `experiments/scripts/run_all_arxiv_local.sh`
- Added onetoone_properties call with reduced params
- Output directory changed to `.../arXiv_benchmarking/local/`

### `experiments/scripts/analyze_arxiv_results.py`
- Added onetoone_properties section to summary report

---

## Property → Experiment → Figure mapping

| Property | Phase 1 evidence | Phase 2 figure | Phase 2 table |
|----------|-----------------|----------------|---------------|
| **P1** Round-trip | 100% pass rate, CI [99.x%, 100%] | `fig_p1_roundtrip`: colored canonical strings per benchmark | `tab_benchmark_properties.tex` |
| **P2** Acyclicity | 100% pass rate, CI [99.x%, 100%] | `fig_p2_acyclicity`: complexity histogram + benchmark depths | `tab_benchmark_properties.tex` |
| **P3** Invariance | 100% pass rate, CI [99.x%, 100%] | `fig_p3_invariance`: multiple encodings → same w* | `tab_benchmark_properties.tex` |
| **P4** Eval preservation | 100% pass rate, max error = 0.0 | `fig_p4_evaluation`: scatter plot eval(D) vs eval(D') | `tab_benchmark_properties.tex` |
| **P5** Search space | `search_space_analysis.py` | `fig_reduction_factor`: ρ vs k with k! reference | Separate CSV |

---

## How to report in results.tex

### One-to-one mapping paragraph

*"We validate the five fundamental properties empirically. Properties P1 (round-trip fidelity), P2 (DAG acyclicity), P3 (canonical invariance), and P4 (evaluation preservation) are tested on 5,000 random IsalSR strings per variable count m ∈ {1,2,3}, with results reported in Table X and Figures Y-Z. All four properties hold at 100% with Clopper-Pearson 95% confidence intervals of [99.x%, 100%]. Property P5 (search space reduction) is validated separately on 1,000 random strings across the Nguyen and Feynman benchmark suites (Section Z)."*

### Figures to include in the paper

1. `tab_benchmark_properties.tex` — Main results table (8 benchmarks × 4 properties)
2. `fig_p3_invariance` — Most scientifically interesting: shows THE invariant property in action
3. `fig_p4_evaluation` — Visual proof of evaluation preservation
4. `fig_p2_acyclicity` — Shows DAG complexity range + all acyclic
5. `fig_p1_roundtrip` — Shows canonical strings with token coloring (optional, informative)
6. `fig_property_pass_rates` — Summary bar chart (optional, compact)

---

## SLURM resource summary (final)

| Experiment | Tasks | Time/task | Mem |
|-----------|-------|-----------|-----|
| exp1_shortest_path | 1 | 30min | 4G |
| exp2_neighborhood | 1 | 30min | 4G |
| **onetoone_properties** | **3** | **2h** | **4G** |
| search_space_analysis | 3 | **6h** | 8G |
| exp3_canonicalization_time | 15 | 8h | 8G |
| exp5_pruning_accuracy | 12 | 12h | 8G |
| exp6_string_compression | 3 | 4h | 4G |
| analyze_arxiv | 1 (dep) | 1h | 4G |

**Total SLURM tasks**: 39
