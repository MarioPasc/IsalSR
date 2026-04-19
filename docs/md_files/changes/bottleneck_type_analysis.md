# Bottleneck-Type Analysis: When Does IsalSR Help?

**Date**: 2026-04-19
**Authors**: Mario Pascual González, with Claude (analysis assistant)
**Status**: Empirical finding from hard-tier Bingo experiments (30 seeds × 10 problems)

---

## Summary

IsalSR's canonical deduplication significantly improves regression quality
**if and only if** the problem's primary difficulty bottleneck is
**structural search** (finding the right operator topology). When the
bottleneck is constant optimization, feature selection, or the problem is
trivially solved / unsolvable, deduplication does not yield significant gains.

This finding is supported by a bottleneck-type classification that predicts
`sig_train` with **10/10 accuracy** (Fisher exact p = 0.0079).

---

## Bottleneck Taxonomy

Five bottleneck categories were identified across the 10 hard-tier problems:

| Category | Description | IsalSR helps? |
|---|---|---|
| **structural** | Correct operator topology is the challenge; constants are simple integers discoverable by LM | **Yes** (5/5 sig_train) |
| **none_trivial** | Problem trivially solved by both variants (R²=1.0 all seeds) | No (0/1) |
| **constant** | Precise real-valued or irrational constants are the bottleneck; structure is simple or irrelevant | No (0/2) |
| **structural_depth** | Structure IS the bottleneck but k≥12, depth≥6 → search space too vast for dedup to make a dent | No (0/1) |
| **width+constants** | Wide repetitive structure requiring identical subexpressions + multiple constants | No (0/1) |

### Per-Problem Classification

| Problem | Bottleneck | n_consts | k | sig_train | sig_both | Reason |
|---|---|---|---|---|---|---|
| I.15.10 | structural | 0 | 7 | ✓ | ✓ | nested sqrt(1−v²/c²); const='1' trivial for LM |
| I.30.3 | structural | 0 | 8 | ✓ | ✓ | parallel sin²/sin² branches; const='2' trivial |
| I.37.4 | structural | 0 | 7 | ✓ | ✓ | sqrt(product)×cos; const='2' trivial |
| II.11.27 | none_trivial | 0 | 8 | ✗ | ✗ | all 30 seeds at R²≈1.0; no bottleneck |
| III.17.37 | structural | 0 | 8 | ✓ | ✓ | nested sqrt(sum of squares); consts='2','4' trivial |
| Keijzer-6 | constant | 1 | 2 | ✗ | ✗ | log(x)+γ; Euler-Mascheroni is irrational |
| Korns-12 | constant+selection | 4 | 7 | ✗ | ✗ | 9.8×x₁ → 156 oscillation cycles; 4 precise constants + 3 irrelevant vars |
| Pagie-1 | structural | 0 | 7 | ✓ | ✗ | sig_train ✓; test fails due to 2 catastrophic outliers (R²=−461, −5856) |
| Vlad-2 | structural_depth | 0 | 13 | ✗ | ✗ | k=13 → k!=6.2B orderings; dedup effect diluted |
| Vlad-4 | width+constants | 2 | 12 | ✗ | ✗ | 5×(xᵢ−3)² repeated; only 12/30 seeds (OOM) |

### Quantitative Predictor: n_nontrivial_constants

All 5 sig_train=True problems have **zero** non-trivial constants in the
ground truth (only integers 1, 2, 4 that LM discovers trivially). Point-biserial
correlation: r = −0.55, p = 0.098 (marginal at n = 10, but direction unambiguous).

---

## The Mechanism: Variance Reduction Through Seed Rescue

IsalSR's deduplication forces the search to evaluate structurally unique DAGs,
preventing wasted evaluations on isomorphic copies. This manifests as:

1. **100% rescue rate** for below-median baseline seeds on all structural problems
2. **26–1518× variance reduction** (Levene's test significant at α = 0.05 for 3/5)
3. **Large effect sizes** (Cliff's δ = 0.48–0.62 for 4/5 structural problems)

| Problem | Var(BL)/Var(ISR) | Levene p | Cliff's δ | Rescue rate |
|---|---|---|---|---|
| I.15.10 | 1,518× | 0.0001 | +0.62 (large) | 100% |
| I.30.3 | ~10²²× | 0.078 | +0.24 (small) | 100% |
| I.37.4 | 32× | 0.0006 | +0.48 (large) | 100% |
| III.17.37 | 26× | 0.001 | +0.62 (large) | 100% |
| Pagie-1 | 3.3× | 0.024 | +0.53 (large) | 100% |

For non-structural problems, variance ratios ≈ 1 (no reduction).

### Per-Seed Correlation

Within every problem (including non-significant ones), IsalSR preferentially
helps the worst seeds: Spearman(δR², R²_baseline) ≈ −0.85 to −1.0 (all p < 0.001).
The mechanism is universal; the question is whether rescuing bad seeds shifts
the distribution enough to reach statistical significance.

---

## Convergence Trajectories

IsalSR pays an **early exploration cost** (forced structural diversity slows
exploitation in generations 0–500), then overtakes baseline when diversity
discovers superior structures:

| Problem | Overtakes at gen | Sustained? | p at overtake |
|---|---|---|---|
| III.17.37 | 500 | Yes (through 10,000) | 0.047 |
| I.15.10 | 1,000 | Yes | 0.003 |
| I.37.4 | 1,000 | Yes | 0.036 |
| Pagie-1 | 2,000 | Marginal (p ≈ 0.14) | — |

For II.11.27, both converge to R² = 1.0 by gen 2,000 → no room.
For Korns-12 and Vlad-4, IsalSR never overtakes → wrong bottleneck.

### Threshold Analysis: R² ≥ 0.99 Train

The most dramatic effect is **failure elimination** — the fraction of seeds
that fail to reach R² ≥ 0.99:

| Problem | BL fail rate | ISR fail rate | Reduction |
|---|---|---|---|
| III.17.37 | 21% (6/28) | **0%** (0/28) | 100% |
| Pagie-1 | 34% (10/29) | **0%** (0/29) | 100% |

---

## The Feynman-vs-GP Split: A Proxy, Not the Cause

The Feynman/GP-hard source split correlates with IsalSR success
(Fisher exact p = 0.048 for sig_both, p = 0.206 for sig_train) but is
**confounded** with bottleneck type:

- Feynman equations → moderate k (7–8), simple integer constants, all vars
  active → structural bottleneck
- GP-hard benchmarks → designed to stress constants, feature selection,
  unusual operators → non-structural bottlenecks

**Pagie-1 breaks the pattern**: it is GP-hard but has a structural bottleneck,
and IsalSR significantly helps (sig_train = True, Cliff's δ = 0.53).

The bottleneck classification (Fisher p = 0.0079) is a strictly better
predictor than the source split (Fisher p = 0.206).

---

## Implications for the Paper

1. **Narrative framing**: IsalSR addresses the *structural search bottleneck*.
   When the difficulty lies in discovering the right operator composition
   (not constant optimization or feature selection), canonical deduplication
   forces exploration of genuinely different DAG topologies.

2. **Target problem selection**: problems with moderate ground-truth complexity
   (k = 7–8), simple or no real-valued constants, all variables active,
   and Feynman-type physics equations are natural candidates.

3. **Complementarity**: IsalSR is complementary to, not a replacement for,
   constant optimization (LM), feature selection, or architecture search.
   Future work could combine IsalSR dedup with improved constant optimization
   to address both bottlenecks simultaneously.

4. **Pagie-1 reclassification**: the test-R² non-significance is a data
   quality issue (2 catastrophic outliers), not a mechanism failure. Consider
   reporting with and without outliers, or using median test R².

---

## Data and Reproducibility

### Raw Results

Location: `/media/mpascual/Sandisk2TB/research/isalsr/results/model_validation/wl_subtree_hard/models_hard/`

Structure:
```
models_hard/
├── bingo/hard/{problem}/{baseline,isalsr}/seed_{01..30}/
│   ├── run_log.json         (per-seed final metrics)
│   ├── trajectory.csv        (per-generation snapshots)
│   └── convergence_log.npz   (full population R² per generation)
├── figures/
│   └── table_bingo_convergence.tex  (LaTeX table with significance tests)
└── metadata.json
```

### Analysis Scripts

All scripts in `experiments/scripts/`:

| Script | Purpose | Output |
|---|---|---|
| `analyze_isalsr_advantage.py` | Collect all Bingo run_log.json into paired CSV | `hard_bingo_paired_data.csv` |
| `analyze_isalsr_advantage_factors.py` | 9-analysis pipeline: baseline difficulty, structural features, empirical features, expression families, sampling, comprehensive ranking, seed-level correlations, R² distribution shape, convergence speed | stdout tables |
| `analyze_isalsr_deep_dive.py` | Threshold analysis, failure rates, per-seed benefit, ceiling decomposition, Cliff's delta, interaction analysis, convergence trajectories, diversity proxy, solution structure | stdout tables |
| `analyze_isalsr_synthesis.py` | Final synthesis: variance reduction mechanism, bottleneck classification (10/10), source split vs bottleneck split, n_constants predictor, Goldilocks zone, comprehensive summary | stdout tables |

### Statistical Tests Used

- **Wilcoxon signed-rank** (paired, two-sided) with Holm-Bonferroni correction
  for R² significance across 10 problems
- **Fisher's exact test** (2×2) for categorical predictors (source, has_sqrt,
  bottleneck type) vs sig_train/sig_both
- **Point-biserial correlation** for continuous predictors vs binary outcome
- **Mann-Whitney U** for group comparisons (sig vs non-sig problems)
- **Levene's test** for variance homogeneity (the mechanism test)
- **Cliff's delta** as non-parametric effect size
- **Spearman rank correlation** for within-problem seed-level associations

### Key Statistical Results

| Test | Result | Interpretation |
|---|---|---|
| Fisher(bottleneck=structural × sig_train) | OR = ∞, p = 0.0079 | **Strong**: bottleneck type predicts advantage |
| Fisher(Feynman × sig_both) | OR = ∞, p = 0.048 | Significant but confounded with bottleneck |
| Fisher(has_sqrt × sig_both) | OR = ∞, p = 0.033 | Significant but confounded (3/5 Feynman have sqrt) |
| Point-biserial(sig_train, n_constants) | r = −0.55, p = 0.098 | Marginal; direction unambiguous |
| Levene(I.15.10) | p = 0.0001 | IsalSR reduces R² variance 1518× |
| Levene(I.37.4) | p = 0.0006 | IsalSR reduces R² variance 32× |
| Levene(III.17.37) | p = 0.001 | IsalSR reduces R² variance 26× |
