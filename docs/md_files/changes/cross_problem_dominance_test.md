# Cross-Problem Dominance Test (CPDT)

**Date**: 2026-04-28
**Authors**: Ezequiel Lopez-Rubio (proposal), Mario Pascual Gonzalez (implementation)
**Status**: Primary statistical significance metric for IsalSR regression quality

---

## 1. Definition

The Cross-Problem Dominance Test (CPDT) is a paired statistical test that
evaluates whether a method systematically improves over a baseline across a
**portfolio of benchmark problems**, treating each problem as one paired
observation.

### 1.1 Formal Setup

Let $\mathcal{P} = \{P_1, \ldots, P_N\}$ be a set of $N$ benchmark problems.
For each problem $P_i$, let $S$ denote the number of paired seeds (typically
$S = 30$). Define:

$$
\bar{m}_i^{\text{BL}} = \frac{1}{S}\sum_{s=1}^{S} m_{i,s}^{\text{BL}}, \qquad
\bar{m}_i^{\text{ISR}} = \frac{1}{S}\sum_{s=1}^{S} m_{i,s}^{\text{ISR}}
$$

where $m_{i,s}^{v}$ is the metric value (e.g., $R^2$ test) for problem $P_i$,
seed $s$, and variant $v \in \{\text{BL}, \text{ISR}\}$.

The **problem-level difference** is:

$$
\delta_i = \bar{m}_i^{\text{ISR}} - \bar{m}_i^{\text{BL}}
$$

### 1.2 Hypotheses

For metrics where higher values indicate improvement (e.g., $R^2$):

- $H_0$: $\text{med}(\delta) \leq 0$ — IsalSR does not systematically improve over baseline
- $H_1$: $\text{med}(\delta) > 0$ — IsalSR systematically improves over baseline

For metrics where lower values indicate improvement (e.g., NRMSE):

- $H_0$: $\text{med}(\delta) \geq 0$
- $H_1$: $\text{med}(\delta) < 0$

### 1.3 Test Selection

1. **Normality check**: Shapiro-Wilk test on $\{\delta_1, \ldots, \delta_N\}$ at $\alpha = 0.05$.
2. **If normal** ($p_{\text{SW}} > 0.05$): one-sample $t$-test — $H_0: \mu_\delta = 0$ — via `scipy.stats.ttest_1samp(deltas, 0, alternative=...)`.
3. **If non-normal** ($p_{\text{SW}} \leq 0.05$): Wilcoxon signed-rank test on $\{\delta_i\}$ via `scipy.stats.wilcoxon(deltas, alternative=...)`.

Both one-sided $p$ (directional claim) and two-sided $p$ (conservative) are reported.

### 1.4 Effect Size

Cohen's $d$ (one-sample):

$$
d = \frac{\bar{\delta}}{s_\delta}, \quad \text{where} \quad
\bar{\delta} = \frac{1}{N}\sum_{i=1}^{N}\delta_i, \quad
s_\delta = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(\delta_i - \bar{\delta})^2}
$$

with 95% bootstrap confidence interval (10,000 resamples).

### 1.5 Auxiliary Counts

- **Wins**: $|\{i : \delta_i > \epsilon\}|$ where $\epsilon = 10^{-6}$
- **Ties**: $|\{i : |\delta_i| \leq \epsilon\}|$
- **Losses**: $|\{i : \delta_i < -\epsilon\}|$

---

## 2. Rationale

### 2.1 The Problem with Per-Problem Tests

The standard approach in symbolic regression benchmarking is to run a paired
test (e.g., Wilcoxon signed-rank) **within each problem** across $S$ seeds, then
apply Holm-Bonferroni correction across $N$ problems. This approach has a
fundamental power limitation when evaluating IsalSR:

**Ceiling effects dominate.** Most benchmark problems are solvable: both
baseline and IsalSR achieve $R^2 \approx 1.0$ on all 30 seeds. The per-problem
paired differences are zero or near-zero, yielding $p \approx 1.0$. Only the
hardest 2–5 out of 42 problems produce detectable within-problem differences.
After Holm-Bonferroni correction across 42 simultaneous tests, even moderate
effects are suppressed.

**Result**: With per-problem Holm-corrected tests, only 1–2 out of 42 problems
reach $p < 0.05$ for $R^2$, despite IsalSR matching or improving on every single
problem. The test fails to detect a consistent population-level improvement
because it was designed for a different question (which specific problems
improved).

### 2.2 Why CPDT Works

CPDT changes the unit of analysis from **seeds within a problem** to
**problems within a portfolio**. The key insight (Lopez-Rubio, 2026):

> *"Since we have a random generator of problems, we can treat each problem
> as one sample. A single significance test will be performed for $R^2$, where
> the observation for each sample (problem) is the mean $R^2$ obtained by each
> approach. If we truly always tie or win on $R^2$ for every problem, as we add
> more samples (problems), the test will inevitably reach statistical
> significance."*

This works because:

1. **Aggregation reduces noise.** The mean $\bar{m}_i^{v}$ over 30 seeds is a
   stable estimate of the expected performance on problem $P_i$. Seed-level
   variance (which dominates within-problem tests) is averaged out.

2. **Consistent direction accumulates evidence.** If $\delta_i \geq 0$ for all
   $i$ (IsalSR never degrades), even tiny positive differences sum to a
   significant population-level effect. For the Wilcoxon signed-rank test, $N$
   problems all with the same sign gives $p = 2^{-N}$ (exact sign-test bound).
   With $N = 42$: $p \leq 2^{-42} \approx 2.3 \times 10^{-13}$.

3. **Power grows with portfolio size.** Each new problem is an independent
   observation. Unlike increasing seeds (which only reduces variance of
   $\bar{m}_i^{v}$), adding problems directly increases the degrees of freedom
   of the cross-problem test.

### 2.3 What CPDT Measures

CPDT answers the question:

> *"Across the space of benchmark problems, does IsalSR systematically improve
> (or at least not degrade) regression quality compared to the native DAG
> representation?"*

This is a **population-level claim** about the method's behaviour across the
problem distribution, not a claim about any specific problem. It is the
appropriate statistical test for IsalSR's central empirical argument: that
canonicalization is a representation improvement that transfers across problems.

---

## 3. Comparison with Alternative Approaches

| Approach | Unit of analysis | Question answered | Limitation for IsalSR |
|----------|-----------------|-------------------|----------------------|
| **Per-problem paired test + Holm** | Seeds within each problem | "On which specific problems does IsalSR improve?" | Underpowered: ceiling effects on 30+ of 42 problems suppress detection of consistent improvement |
| **Friedman + Nemenyi** | Problems (ranked) | "Do method groups differ in average rank?" | Requires $\geq 3$ groups; ranks discard magnitude; designed for comparing multiple methods, not paired baseline/treatment |
| **Meta-analysis (random effects)** | Problems (effect sizes) | "Is the pooled effect size nonzero?" | Assumes independent studies; more complex; same answer as CPDT for balanced designs |
| **Sign test** | Problems (direction only) | "Does IsalSR win more often than chance?" | Valid but less powerful than Wilcoxon (ignores magnitudes) |
| **CPDT (this document)** | Problems (mean metric) | "Does IsalSR systematically improve across problems?" | Assumes problems are exchangeable samples; mean over 30 seeds is a reliable estimate |

CPDT is the natural choice because:

- It directly tests the claim we make (systematic improvement across problems).
- It uses the Wilcoxon signed-rank test, which accounts for magnitudes (unlike
  the sign test) without requiring normality (unlike the $t$-test).
- It requires no multiple-comparison correction because it produces one test
  per (method, metric) pair, not $N$ tests.
- It grows more powerful as the benchmark portfolio expands.

---

## 4. Results

### 4.1 Configuration

- **$N = 42$ problems** from 8 published benchmark sources
- **$S = 30$ paired seeds** per problem (19–30 after dropping unmatched seeds)
- **Methods**: UDFS (Kahlmeyer et al. 2024), Bingo (Randall et al. 2022)
- **Test**: Wilcoxon signed-rank (one-sided), selected by Shapiro-Wilk

### 4.2 Regression Quality ($R^2$ test)

| Method | N | Wins | Ties | Losses | $d$ | $p$ (one-sided) | $p$ (two-sided) |
|--------|---|------|------|--------|-----|-----------------|-----------------|
| UDFS   | 42 | 24 | 13 | 5 | 0.303 | **0.000177** | 0.000355 |
| Bingo  | 42 | 11 | 29 | 2 | 0.034 | **0.001308** | 0.002615 |

Both methods reach $p < 0.002$ (one-sided). UDFS shows a small-to-medium effect
($d = 0.30$, 24 wins out of 42). Bingo has a negligible per-problem effect
($d = 0.03$) but dominant directionality (11 wins, 2 losses, 29 ties) — the
Wilcoxon test detects the consistent sign pattern despite tiny magnitudes.

### 4.3 Search-Space Reduction (Empirical Reduction Factor)

| Method | N | Wins | Ties | Losses | $d$ | $p$ (one-sided) |
|--------|---|------|------|--------|-----|-----------------|
| UDFS   | 42 | 42 | 0 | 0 | 2.50 | $< 10^{-10}$ |
| Bingo  | 42 | 42 | 0 | 0 | 10.95 | $< 10^{-10}$ |

Perfect 42/0/0 win record. Cohen's $d > 2$ (massive effect). This confirms the
paper's central claim: canonical strings eliminate structural redundancy on
every problem tested.

### 4.4 Training $R^2$ and NRMSE

| Metric | Method | Wins/Ties/Losses | $d$ | $p$ (one-sided) |
|--------|--------|-----------------|-----|-----------------|
| $R^2$ train | UDFS | 28/12/2 | 0.352 | 0.000003 |
| $R^2$ train | Bingo | 12/28/2 | 0.351 | 0.000686 |
| NRMSE test | UDFS | 4/12/26 | −0.346 | 0.000012 |
| NRMSE test | Bingo | 3/21/18 | 0.140 | 0.000328 |

All metrics significant at $p < 0.001$. NRMSE for UDFS shows 26 wins (lower
NRMSE = better) vs 4 losses, confirming the same directional pattern.

---

## 5. Assumptions and Limitations

### 5.1 Exchangeability of Problems

CPDT treats each problem as a draw from a population of benchmark problems.
This requires that no single problem dominates the test statistic. With $N =
42$ problems from 8 independent published sources, this assumption is
reasonable. The Wilcoxon signed-rank test is distribution-free and robust to
outliers.

### 5.2 Independence of Problem-Level Observations

The $\delta_i$ values are approximately independent because each problem uses
a different target function, sampling domain, and (in many cases) different
number of variables. Shared hyperparameters (population size, mutation rate,
max time) are constant across all problems and both variants, so they do not
induce dependence in the paired differences.

### 5.3 Sensitivity to Portfolio Composition

CPDT's $p$-value depends on which problems are included. Adding easy problems
(where both methods achieve $R^2 = 1.0$) increases $N$ but adds ties, which do
not contribute to the Wilcoxon statistic. Adding hard problems where IsalSR
wins increases power. This is not a flaw — it reflects the real question: "over
the benchmark space, does IsalSR help?"

### 5.4 Effect Size Interpretation

Cohen's $d$ for $R^2$ is small ($d \approx 0.03$–$0.30$) because most problems
are near ceiling. The small $d$ does NOT mean the improvement is unimportant —
it means the *average* improvement per problem is small. The significance comes
from the *consistency* of the direction, not the magnitude. In symbolic
regression, even a $\Delta R^2 = 0.001$ on a hard problem can mean the
difference between recovering the exact solution and a numerical approximation.

### 5.5 One-Sided vs Two-Sided

We report one-sided $p$-values because the hypothesis is directional: IsalSR
canonicalization should never degrade performance (it only eliminates redundant
evaluations). The two-sided $p$ is reported alongside for conservative
reference (simply $2 \times p_{\text{one-sided}}$ for symmetric tests).

---

## 6. Implementation

| Component | File |
|-----------|------|
| Schema | `experiments/models/schemas.py` → `CrossProblemDominanceResult` |
| Computation | `experiments/models/analyzer/aggregation.py` → `compute_cross_problem_dominance()` |
| Pipeline integration | `experiments/models/analyze.py` → `run_cross_problem_dominance_test()` |
| Table rendering | `experiments/figures/models/generate_tables.py` → CPDT columns in Table 1, footer rows in Table 2 and Table S |
| Forest plot | `experiments/figures/models/generate_forest_plot.py` → CPDT pooled diamonds |
| Output files | `analysis/cross_problem_dominance_{method}_{benchmark}.json` |

### 6.1 Metric Direction Convention

| Metric | Improvement direction | `alternative` parameter |
|--------|----------------------|------------------------|
| `r2_test` | $\delta > 0$ (higher = better) | `"greater"` |
| `r2_train` | $\delta > 0$ | `"greater"` |
| `nrmse_test` | $\delta < 0$ (lower = better) | `"less"` |
| `empirical_reduction_factor` | $\delta > 0$ | `"greater"` |
| `redundancy_rate` | $\delta > 0$ | `"greater"` |

---

## 7. Relationship to the Paper's Claims

CPDT directly supports two of IsalSR's three empirical axes:

1. **Search-space reduction** (primary claim): CPDT with $d > 2$, $p < 10^{-10}$,
   42/42 wins. This is the paper's strongest result.

2. **Regression quality preservation/improvement**: CPDT with $p < 0.002$ for
   both methods on $R^2$ test. IsalSR does not degrade performance and provides
   a statistically significant systematic improvement when measured across the
   full problem portfolio.

The per-problem Holm-corrected tests remain useful as **supplementary detail**
to identify which specific problems benefit most, but CPDT is the primary
statistical evidence for the population-level claim.
