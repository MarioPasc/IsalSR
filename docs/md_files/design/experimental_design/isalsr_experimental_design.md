# IsalSR Experimental Design: Three-Axis Research Report

**Authors**: Mario Pascual González, Ezequiel López-Rubio  
**Affiliation**: University of Málaga  
**Date**: March 2026  
**Target venue**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

---

## 0. Executive Summary

This document formalizes the experimental framework for evaluating IsalSR's central claim: **an isomorphism-invariant DAG representation reduces the symbolic regression search space by O(k!) for k internal nodes**, and this reduction accelerates existing published SR methods when their internal DAG representation is replaced by IsalSR canonical strings. The document is structured in three axes:

- **(A) Literature Review**: Identification and classification of existing SR methods that use DAG or graph-based representations, together with their available implementations.
- **(B) Experimental Design**: Formalization of metrics (regression performance, wall-clock time, search space dimensionality) and the validation protocol.
- **(C) Statistical Analysis**: Design of output data structures, folder hierarchy, and the downstream statistical tests (p-values, Cohen's d, 95% CI).

---

## A. Literature Review: DAG-Based Symbolic Regression Methods

### A.1. Taxonomy of Candidate Baselines

Ezequiel's email specifies the experimental design: take already-published SR proposals, replace **only** their DAG representation with IsalSR canonical strings, and compare. This requires identifying methods whose internal representation is a DAG (or tree convertible to DAG) and whose codebase is modular enough to permit such a swap.

We classify candidate baselines by their **search paradigm** and **representation type**:

| # | Method | Paradigm | Representation | Invariance? | Venue | Code |
|---|--------|----------|---------------|-------------|-------|------|
| 1 | **UDFS** (Kahlmeyer et al., 2024) | Systematic (unbiased enumeration) | Expression DAGs (skeletons + labelings) | No (enumerates all labelings) | IJCAI 2024 | `github.com/kahlmeyer94/DAG_search` |
| 2 | **GraphDSR** (Liu et al., 2025) | Deep RL + GNN | DAG with adjacency matrix | No (categorical sampling per node) | Neural Networks 187:107405 | Referenced in IsalSR README |
| 3 | **GSR** (Xiang et al., 2025) | Hybrid neural-guided MCTS | Expression Graphs (EGs) via TRS + DAG | Yes (permutation-invariant via TRS) | NeurIPS 2025 | `openreview.net/forum?id=JYB6wFcbky` |
| 4 | **SymRegg** (de França & Kronberger, 2025) | Perturbation + e-graph | E-graphs (DAG + equivalence classes) | Partial (equivalence saturation) | arXiv:2511.01009 | Haskell-based (`eggp` tool) |
| 5 | **PySR / SymbolicRegression.jl** (Cranmer, 2020–2025) | Regularized evolution + SA | Expression trees (not DAG) | No | JOSS / arXiv:2305.01582 | `github.com/MilesCranmer/PySR` |
| 6 | **DSO / uDSR** (Petersen et al., 2021; Landajuela et al., 2022) | Risk-seeking policy gradient + GP | Token sequences (pre-order traversal) | No | ICLR 2021, NeurIPS 2022 | `github.com/dso-org/deep-symbolic-optimization` |
| 7 | **Operon** (Burlacu et al., 2020) | GP with Levenberg-Marquardt | Expression trees | No | GECCO 2020 | `github.com/heal-research/operon` |
| 8 | **PSE / PSRN** (Ruan et al., 2026) | Parallel GPU enumeration | Layered network (implicit DAG) | No | Nature Computational Science 6(1) | `github.com/intell-sci-comput/PSE` |

### A.2. Primary Baselines (DAG-native, swappable representation)

The most suitable baselines for Ezequiel's experimental protocol are those that **already use an explicit DAG internally**, because the representation swap is cleanest. We identify **three tiers**:

**Tier 1 — Direct DAG swap (ideal baselines)**:

1. **UDFS** (Kahlmeyer et al., 2024): The cleanest candidate. UDFS enumerates DAG skeletons (unlabeled DAGs) and then exhaustively labels operator nodes. IsalSR canonical strings would collapse the O(k!) equivalent labelings into a single canonical representative *before* evaluation, directly reducing the number of DAG frames to evaluate. The comparison is:
   - **Baseline**: UDFS enumerates N_baseline DAG frames.
   - **IsalSR**: Canonicalize each skeleton → deduplicate → enumerate N_isalsr ≤ N_baseline / k! unique canonical frames.
   - **Metric**: N_baseline / N_isalsr ≈ k! (empirical reduction factor).

2. **GraphDSR** (Liu et al., 2025): Uses a GNN to sample DAGs incrementally (node types from categorical distributions + adjacency matrix). IsalSR can replace the adjacency-matrix representation with canonical strings, canonicalizing after each sampling step. The GNN would then operate on a reduced (canonical) state space.

**Tier 2 — Adaptable with moderate effort**:

3. **GSR** (Xiang et al., 2025): Already addresses permutation invariance via Expression Graphs and a TRS (Term-Rewriting System). This is the most scientifically interesting comparator because GSR's invariance mechanism (TRS-based) differs fundamentally from IsalSR's (graph-isomorphism-based canonical strings). The comparison tests *which invariance mechanism is more efficient*.

4. **SymRegg** (de França & Kronberger, 2025): Uses e-graphs to store equivalence classes. IsalSR canonical strings could replace the e-graph's equivalence detection: instead of maintaining an e-graph and running equality saturation, one canonicalizes each expression and uses string equality.

**Tier 3 — Tree-based (require tree→DAG adapter)**:

5. **PySR**, **Operon**, **DSO**: These use expression trees. An adapter that converts trees to DAGs (via common-subexpression elimination) and then canonicalizes with IsalSR would be needed. This is feasible but introduces confounding factors (the tree→DAG conversion itself changes the representation). These are useful as **secondary comparators** but not primary baselines.

### A.3. Benchmark Datasets

Following standard practice in SR literature (La Cava et al., 2021; de França et al., 2024):

| Benchmark | # Problems | # Variables | Source |
|-----------|-----------|-------------|--------|
| **Nguyen** (Uy et al., 2011) | 12 | 1–2 | Standard GP benchmark |
| **Feynman** (Udrescu & Tegmark, 2020) | 120 | 1–9 | Physics equations from Feynman Lectures |
| **SRBench** (La Cava et al., 2021; updated 2025) | 252+ | 1–100+ | PMLB real-world + synthetic |

The Nguyen benchmark is small but well-understood; Feynman provides physically meaningful ground truths of varying complexity; SRBench provides the community-standard large-scale evaluation.

### A.4. Key References

1. Kahlmeyer, P., Giesen, J., Habeck, M., & Voigt, H. (2024). *Scaling Up Unbiased Search-based Symbolic Regression*. IJCAI-24, 4264–4272.
2. Liu, Y., et al. (2025). *Mathematical expression exploration with graph representation and generative graph neural network*. Neural Networks, 187, 107405.
3. Xiang, Z., Ashen, K., Qian, X., & Qian, X. (2025). *Graph-based Symbolic Regression with Invariance and Constraint Encoding*. NeurIPS 2025.
4. de França, F. O., & Kronberger, G. (2025). *Equality Graph Assisted Symbolic Regression*. arXiv:2511.01009.
5. Cranmer, M. (2023). *Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl*. arXiv:2305.01582.
6. Petersen, B. K., et al. (2021). *Deep Symbolic Regression: Recovering mathematical expressions from data via risk-seeking policy gradients*. ICLR 2021.
7. Landajuela, M., et al. (2022). *A Unified Framework for Deep Symbolic Regression*. NeurIPS 2022.
8. Burlacu, B., Kronberger, G., & Kommenda, M. (2020). *Operon C++: An Efficient Genetic Programming Framework for Symbolic Regression*. GECCO 2020.
9. La Cava, W., et al. (2021). *Contemporary Symbolic Regression Methods and their Relative Performance*. NeurIPS Datasets & Benchmarks.
10. Ruan, K., et al. (2026). *Fast and efficient symbolic expression discovery through parallelized symbolic enumeration*. Nature Computational Science, 6(1).
11. Udrescu, S.-M., & Tegmark, M. (2020). *AI Feynman: A physics-inspired method for symbolic regression*. Science Advances, 6(16).
12. López-Rubio, E. (2025). *IsalGraph*. arXiv:2512.10429v2.
13. de França, F. O., et al. (2024). *SRBench++: Principled Benchmarking of Symbolic Regression With Domain-Expert Interpretation*. IEEE Trans. Evol. Comput.
14. Imai Aldeia, G. S., et al. (2025). *Call for Action: towards the next generation of symbolic regression benchmark*. GECCO 2025 Workshop.
15. Kronberger, G., et al. (2024). *Reducing redundancy in GP search with equality saturation*. GECCO 2024.

---

## B. Experimental Design

### B.1. Central Hypothesis

**H₀ (null)**: Replacing the DAG representation of a published SR method with IsalSR canonical strings does not significantly change (a) regression performance, (b) wall-clock time, or (c) the number of unique DAGs explored.

**H₁ (alternative)**: The IsalSR representation reduces the number of unique DAGs explored by a factor approaching O(k!), reduces wall-clock time to reach a given fitness threshold, and maintains or improves regression performance.

### B.2. Independent Variables (Factors)

| Factor | Levels | Description |
|--------|--------|-------------|
| **Representation** (primary) | {Baseline, IsalSR} | The internal DAG representation used by the SR method |
| **Method** (blocking) | {UDFS, GraphDSR, GSR, SymRegg, ...} | The published SR method being modified |
| **Benchmark** | {Nguyen, Feynman, SRBench-subset} | The problem set |
| **Problem** (nested in Benchmark) | Individual problems | Each target expression |
| **Random seed** | {1, 2, ..., 30} | For stochastic methods; 30 runs per (method, representation, problem) |

The design is a **paired comparison**: for each (method, problem, seed) triple, we run both the Baseline and IsalSR variants, measuring the dependent variables below.

### B.3. Dependent Variables (Metrics)

We define three axes of metrics, matching Ezequiel's specification:

#### B.3.1. Regression Performance

Let $\hat{y} = \hat{f}(X)$ be the model prediction and $y$ the ground truth.

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| **R² (coefficient of determination)** | $R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$ | $(-\infty, 1]$ | Higher is better |
| **NRMSE (normalized RMSE)** | $\text{NRMSE} = \frac{\sqrt{\frac{1}{N}\sum_i (y_i - \hat{y}_i)^2}}{\sigma_y}$ | $[0, \infty)$ | Lower is better |
| **Solution rate** | $\text{SR} = \frac{|\{p : \hat{f}_p \equiv f_p^*\}|}{|\mathcal{P}|}$ | $[0, 1]$ | Exact symbolic recovery (via SymPy simplification) |
| **Jaccard index** (Kahlmeyer et al., 2024) | $J(\hat{f}, f^*) = \frac{|S(\hat{f}) \cap S(f^*)|}{|S(\hat{f}) \cup S(f^*)|}$ | $[0, 1]$ | Soft recovery: overlap of subexpression sets |
| **Model complexity** | Number of nodes in the expression DAG | $\mathbb{Z}^+$ | Lower is simpler |

#### B.3.2. Time

| Metric | Formula | Units | Interpretation |
|--------|---------|-------|----------------|
| **Wall-clock time** $T_{\text{wall}}$ | `time.perf_counter()` end − start | seconds | Total elapsed time |
| **Time to threshold** $T_\tau$ | First time $t$ such that $R^2(t) \geq \tau$ | seconds | Time to reach $R^2 = 0.99$ (or 0.999) |
| **Canonicalization overhead** $T_{\text{canon}}$ | Cumulative time in `canonical_string()` | seconds | IsalSR-specific overhead |
| **Speedup** | $S = T_{\text{wall}}^{\text{baseline}} / T_{\text{wall}}^{\text{IsalSR}}$ | dimensionless | >1 means IsalSR is faster |

#### B.3.3. Search Space Dimensionality

| Metric | Formula | Units | Interpretation |
|--------|---------|-------|----------------|
| **Total DAGs explored** $N_{\text{total}}$ | Counter incremented at each DAG evaluation | integer | Raw search effort |
| **Unique canonical DAGs** $N_{\text{unique}}$ | Size of the set of canonical strings seen | integer | Effective search effort after deduplication |
| **Empirical reduction factor** | $\rho = N_{\text{total}} / N_{\text{unique}}$ | dimensionless | Measures redundancy eliminated; should approach k! |
| **Theoretical reduction bound** | $k!$ where $k$ = # internal nodes | dimensionless | Upper bound from IsalSR theory |
| **Redundancy rate** | $r = 1 - N_{\text{unique}} / N_{\text{total}}$ | $[0, 1)$ | Fraction of DAGs that are isomorphic duplicates |

### B.4. Experimental Protocol

For each baseline method $M \in \{M_1, \ldots, M_B\}$:

1. **Implement the adapter**: Create `IsalSR-M` by replacing the DAG representation in method $M$ with IsalSR canonical strings. The search algorithm, operator set, and all hyperparameters remain identical.

2. **For each benchmark problem $p \in \mathcal{P}$**:
   a. Generate training data $(X_{\text{train}}, y_{\text{train}})$ and test data $(X_{\text{test}}, y_{\text{test}})$ with fixed random seed.
   b. **For each random seed $s \in \{1, \ldots, 30\}$**:
      - Run $M$(Baseline) on $(X_{\text{train}}, y_{\text{train}})$ → record all metrics.
      - Run $M$(IsalSR) on $(X_{\text{train}}, y_{\text{train}})$ → record all metrics.
      - Evaluate best expression on $(X_{\text{test}}, y_{\text{test}})$.

3. **Compute paired differences** for each metric: $\delta_{p,s} = \text{metric}^{\text{IsalSR}}_{p,s} - \text{metric}^{\text{Baseline}}_{p,s}$.

### B.5. Computational Budget

Following SRBench conventions, each run is given a maximum wall-clock time budget (e.g., 1 hour per problem). The budget is the **same** for both Baseline and IsalSR variants, ensuring a fair comparison.

### B.6. Confound Control

| Threat | Mitigation |
|--------|------------|
| Hardware variability | Same machine, same CPU, no parallelism during timing |
| Random seed dependence | 30 seeds per configuration (standard in SRBench) |
| Hyperparameter advantage | Identical hyperparameters for both variants |
| Implementation bias | Minimal code changes — only the representation layer |
| Dataset memorization | Train/test split with held-out test set |

---

## C. Statistical Analysis

### C.1. Statistical Framework

For each metric $m$, each (method $M$, problem $p$) pair yields 30 paired observations:

$$\{(x_{s}^{\text{Baseline}}, x_{s}^{\text{IsalSR}})\}_{s=1}^{30}$$

We compute paired differences $d_s = x_s^{\text{IsalSR}} - x_s^{\text{Baseline}}$ and test:

$$H_0: \mu_d = 0 \quad \text{vs} \quad H_1: \mu_d \neq 0$$

### C.2. Test Selection

| Condition | Test | Justification |
|-----------|------|---------------|
| $d_s$ approximately normal (Shapiro-Wilk $p > 0.05$) | **Paired t-test** | Parametric, maximum power |
| $d_s$ non-normal | **Wilcoxon signed-rank test** | Non-parametric alternative |
| Multiple problems (family-wise) | **Holm-Bonferroni correction** | Controls FWER at $\alpha = 0.05$ |
| Aggregated across all problems | **Friedman test** + Nemenyi post-hoc | Non-parametric multi-comparison |

### C.3. Effect Size: Cohen's d (Paired)

For paired data:

$$d_{\text{paired}} = \frac{\bar{d}}{s_d}$$

where $\bar{d} = \frac{1}{n}\sum_{s=1}^{n} d_s$ and $s_d = \sqrt{\frac{1}{n-1}\sum_{s=1}^{n}(d_s - \bar{d})^2}$.

Interpretation (Cohen, 1988): $|d| < 0.2$ negligible, $0.2 \leq |d| < 0.5$ small, $0.5 \leq |d| < 0.8$ medium, $|d| \geq 0.8$ large.

### C.4. 95% Confidence Interval

For the mean difference:

$$\text{CI}_{95\%}(\mu_d) = \bar{d} \pm t_{0.025, n-1} \cdot \frac{s_d}{\sqrt{n}}$$

where $t_{0.025, n-1}$ is the critical value of the t-distribution with $n-1$ degrees of freedom.

For Cohen's d, we use the non-central t-distribution (Hedges & Olkin, 1985) or the bootstrap method (10,000 resamples) for the CI:

$$\text{CI}_{95\%}(d) = [d_{\text{boot}, 2.5\%}, \; d_{\text{boot}, 97.5\%}]$$

### C.5. Summary Statistics to Report (Per Metric, Per Method, Per Problem)

| Statistic | Description |
|-----------|-------------|
| $\bar{x}^{\text{Baseline}}$, $\bar{x}^{\text{IsalSR}}$ | Mean of each variant |
| $s^{\text{Baseline}}$, $s^{\text{IsalSR}}$ | Standard deviation of each variant |
| $\bar{d}$ | Mean paired difference |
| $s_d$ | Std of paired differences |
| $t$-statistic or $W$-statistic | Test statistic |
| $p$-value (raw) | Unadjusted p-value |
| $p$-value (adjusted) | Holm-Bonferroni adjusted |
| $d_{\text{paired}}$ | Cohen's d |
| $\text{CI}_{95\%}(\mu_d)$ | 95% CI for mean difference |
| $\text{CI}_{95\%}(d)$ | 95% CI for Cohen's d |
| Shapiro-Wilk $p$ | Normality test on $d_s$ |

---

## C.6. Output Experiment Folder Structure

```
results/
├── config.yaml                          # Global experiment configuration
├── metadata.json                        # Git hash, timestamp, hardware specs
│
├── {method}/                            # e.g., udfs/, graphdsr/, gsr/, ...
│   ├── {benchmark}/                     # e.g., nguyen/, feynman/, srbench/
│   │   ├── {problem}/                   # e.g., nguyen_01/, I_6_20a/, ...
│   │   │   ├── baseline/
│   │   │   │   ├── seed_{s}/            # e.g., seed_01/, seed_02/, ..., seed_30/
│   │   │   │   │   ├── run_log.json     # Per-run raw data (see C.7)
│   │   │   │   │   ├── trajectory.csv   # Time-series of fitness during search
│   │   │   │   │   └── best_expr.json   # Best expression found
│   │   │   │   └── aggregate.csv        # 30-seed summary for this (method, problem, baseline)
│   │   │   │
│   │   │   ├── isalsr/
│   │   │   │   ├── seed_{s}/
│   │   │   │   │   ├── run_log.json
│   │   │   │   │   ├── trajectory.csv
│   │   │   │   │   └── best_expr.json
│   │   │   │   └── aggregate.csv
│   │   │   │
│   │   │   └── paired_stats.json        # Statistical test results for this problem
│   │   │
│   │   └── benchmark_summary.csv        # Aggregated across all problems in the benchmark
│   │
│   └── method_summary.csv               # Aggregated across all benchmarks for this method
│
├── global_summary.csv                   # Grand summary across all methods
├── statistical_tests/
│   ├── friedman_test.json               # Friedman test across all methods
│   ├── nemenyi_posthoc.json             # Nemenyi post-hoc test
│   └── critical_difference_plot.svg     # Critical difference diagram
│
└── figures/
    ├── r2_boxplots/                     # One boxplot per method × benchmark
    ├── speedup_heatmaps/                # Speedup heatmaps
    ├── reduction_factor_vs_k/           # ρ vs k scatter plots
    └── pareto_fronts/                   # Accuracy vs complexity Pareto fronts
```

### C.7. File Schemas

#### `run_log.json` — Per-seed raw data

```json
{
  "method": "udfs",
  "representation": "isalsr",
  "benchmark": "nguyen",
  "problem": "nguyen_01",
  "seed": 1,
  "hardware": {
    "cpu": "Intel Xeon Gold 6226R",
    "ram_gb": 128,
    "python_version": "3.11.9"
  },
  "hyperparameters": {
    "max_intermediary_nodes": 5,
    "max_dag_skeletons": 200000,
    "operator_set": ["+", "*", "-", "/", "sin", "cos", "exp", "log"]
  },
  "results": {
    "regression": {
      "r2_train": 0.9998,
      "r2_test": 0.9995,
      "nrmse_train": 0.0141,
      "nrmse_test": 0.0224,
      "mse_train": 1.23e-05,
      "mse_test": 2.45e-05,
      "solution_recovered": true,
      "jaccard_index": 1.0,
      "model_complexity": 7
    },
    "time": {
      "wall_clock_s": 142.5,
      "time_to_r2_099_s": 45.2,
      "time_to_r2_0999_s": 98.7,
      "canonicalization_time_s": 12.3,
      "evaluation_time_s": 110.4,
      "overhead_time_s": 19.8
    },
    "search_space": {
      "total_dags_explored": 1500000,
      "unique_canonical_dags": 250000,
      "empirical_reduction_factor": 6.0,
      "max_internal_nodes_seen": 5,
      "theoretical_reduction_bound": 120,
      "redundancy_rate": 0.8333
    }
  },
  "best_expression": {
    "symbolic": "x**3 + x**2 + x",
    "isalsr_string": "V+NnncVs...",
    "canonical_string": "...",
    "n_nodes": 7,
    "n_edges": 6
  }
}
```

#### `trajectory.csv` — Time-series during search

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_s` | float | Elapsed seconds since run start |
| `iteration` | int | Search iteration number |
| `best_r2` | float | Best R² found so far |
| `best_nrmse` | float | Best NRMSE found so far |
| `n_dags_explored` | int | Cumulative DAGs explored |
| `n_unique_canonical` | int | Cumulative unique canonical strings |
| `current_expr` | str | Current best expression (symbolic) |
| `current_complexity` | int | Current best expression complexity |

#### `aggregate.csv` — 30-seed summary per (method, problem, representation)

| Column | Type | Description |
|--------|------|-------------|
| `method` | str | Method name |
| `representation` | str | `baseline` or `isalsr` |
| `benchmark` | str | Benchmark name |
| `problem` | str | Problem ID |
| `metric` | str | Metric name |
| `mean` | float | Mean over 30 seeds |
| `std` | float | Standard deviation |
| `median` | float | Median |
| `q25` | float | 25th percentile |
| `q75` | float | 75th percentile |
| `min` | float | Minimum |
| `max` | float | Maximum |

#### `paired_stats.json` — Statistical test results per problem

```json
{
  "method": "udfs",
  "benchmark": "nguyen",
  "problem": "nguyen_01",
  "n_seeds": 30,
  "metrics": {
    "r2_test": {
      "baseline_mean": 0.9982,
      "baseline_std": 0.0015,
      "isalsr_mean": 0.9991,
      "isalsr_std": 0.0008,
      "mean_diff": 0.0009,
      "std_diff": 0.0012,
      "shapiro_wilk_p": 0.32,
      "normality_assumed": true,
      "test_used": "paired_t_test",
      "t_statistic": 4.11,
      "p_value_raw": 0.00031,
      "p_value_holm": 0.0037,
      "cohens_d": 0.75,
      "cohens_d_ci_lower": 0.21,
      "cohens_d_ci_upper": 1.28,
      "mean_diff_ci_lower": 0.00045,
      "mean_diff_ci_upper": 0.00135
    },
    "wall_clock_s": { "..." : "..." },
    "total_dags_explored": { "..." : "..." },
    "empirical_reduction_factor": { "..." : "..." }
  }
}
```

#### `benchmark_summary.csv` — Aggregated across problems

| Column | Type | Description |
|--------|------|-------------|
| `method` | str | Method name |
| `benchmark` | str | Benchmark name |
| `metric` | str | Metric name |
| `n_problems` | int | Number of problems |
| `n_significant` | int | Problems where p_holm < 0.05 |
| `mean_cohens_d` | float | Average Cohen's d across problems |
| `median_cohens_d` | float | Median Cohen's d |
| `mean_speedup` | float | Average speedup (for time metrics) |
| `mean_reduction_factor` | float | Average ρ (for search space metrics) |
| `solution_rate_baseline` | float | Fraction of problems solved (baseline) |
| `solution_rate_isalsr` | float | Fraction of problems solved (IsalSR) |

---

## C.8. Recommended Python Libraries for Statistical Analysis

| Library | Purpose | Version |
|---------|---------|---------|
| `scipy.stats` | Paired t-test, Wilcoxon, Shapiro-Wilk | ≥1.11 |
| `statsmodels` | Holm-Bonferroni correction, Friedman test | ≥0.14 |
| `scikit-posthocs` | Nemenyi post-hoc test | ≥0.9 |
| `numpy` + `bootstrapped` | Bootstrap CI for Cohen's d | — |
| `matplotlib` / `seaborn` | Visualization | — |
| `pandas` | Data wrangling | — |

### C.9. Statistical Analysis Pipeline (pseudocode)

```
for each method M:
  for each benchmark B:
    for each problem p in B:
      load 30 baseline values and 30 IsalSR values
      compute d_s = isalsr_s - baseline_s for s = 1..30
      test normality: shapiro_wilk(d_s)
      if normal:
        paired t-test → (t, p_raw)
      else:
        wilcoxon signed-rank → (W, p_raw)
      Cohen's d = mean(d_s) / std(d_s)
      CI_mean = mean(d_s) ± t_crit * std(d_s)/sqrt(30)
      CI_d = bootstrap_ci(d_s, n_boot=10000)
      store in paired_stats.json

    collect all p_raw across problems in B
    apply Holm-Bonferroni correction → p_adjusted
    update paired_stats.json with p_adjusted

  aggregate across benchmarks → method_summary.csv

Friedman test across all methods → friedman_test.json
Nemenyi post-hoc → nemenyi_posthoc.json
Generate critical difference diagram
```

---

## D. Summary Table: Metrics to Store and Their Statistical Treatment

| Metric | Axis | Per-run storage | Statistical test | Effect size | Goal direction |
|--------|------|----------------|-----------------|-------------|----------------|
| R² (test) | Regression | `run_log.json` | Paired t / Wilcoxon | Cohen's d + 95% CI | Higher = better |
| NRMSE (test) | Regression | `run_log.json` | Paired t / Wilcoxon | Cohen's d + 95% CI | Lower = better |
| Solution rate | Regression | `run_log.json` | McNemar's test | Odds ratio | Higher = better |
| Jaccard index | Regression | `run_log.json` | Paired t / Wilcoxon | Cohen's d + 95% CI | Higher = better |
| Model complexity | Regression | `run_log.json` | Paired t / Wilcoxon | Cohen's d + 95% CI | Lower = better |
| Wall-clock time | Time | `run_log.json` | Paired t / Wilcoxon | Cohen's d + 95% CI | Lower = better |
| Time to R²≥0.99 | Time | `run_log.json` | Paired t / Wilcoxon | Cohen's d + 95% CI | Lower = better |
| Canonicalization overhead | Time | `run_log.json` | Descriptive only | — | Understand cost |
| Total DAGs explored | Search space | `run_log.json` | Paired t / Wilcoxon | Cohen's d + 95% CI | Lower = better |
| Unique canonical DAGs | Search space | `run_log.json` | Paired t / Wilcoxon | Cohen's d + 95% CI | Lower = better |
| Reduction factor ρ | Search space | `run_log.json` | One-sample t vs k! | — | Close to k! |
| Redundancy rate | Search space | `run_log.json` | Paired t / Wilcoxon | Cohen's d + 95% CI | Higher = better |
| R²(t) trajectory | Time-series | `trajectory.csv` | AUC comparison | — | Higher AUC = better |

---

## E. Recommended Figures for the Paper

1. **Critical difference diagram** (Nemenyi post-hoc): ranking Baseline vs IsalSR across all problems.
2. **Boxplots of R² (test)**: Baseline vs IsalSR, grouped by benchmark.
3. **Speedup heatmap**: methods × problems, colored by $S = T_{\text{baseline}} / T_{\text{IsalSR}}$.
4. **ρ vs k scatter plot**: empirical reduction factor versus number of internal nodes, with theoretical k! curve overlaid.
5. **Convergence curves**: R²(t) averaged over seeds, Baseline vs IsalSR, for selected problems.
6. **Pareto front**: accuracy (R²) vs complexity (# nodes), showing both variants.
7. **Forest plot**: Cohen's d with 95% CI per problem, sorted by effect size.
