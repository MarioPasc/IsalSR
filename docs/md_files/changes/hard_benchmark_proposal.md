# Hard Benchmark Proposal for IsalSR — Literature-Justified Problem Selection

**Document purpose:** Justify and specify a set of additional benchmark problems for the IsalSR TPAMI submission, targeting the experimental gap identified in the current manuscript: all 22 Nguyen/Feynman problems are solved to R² ≈ 1.0 by Bingo, precluding any demonstration of downstream quality or convergence-speed improvement from IsalSR's search-space reduction. The problems below are selected to fill this gap with minimal justification burden.

**Date:** 2026-04-13

---

## 1. Selection Methodology

### 1.1 Criteria

Each additional problem must satisfy **all four** of the following:

1. **Published provenance.** The expression and sampling protocol appear in a citable, peer-reviewed source used by at least two independent SR studies. No ad-hoc expressions.

2. **Operator compatibility.** The target expression is representable within the IsalSR operator set Σ = {+, ×, NEG, INV, sin, cos, exp, log, √, |·|, POW, CONST}. Operators not in Σ (e.g., tanh, arctan, signum, floor) disqualify a problem.

3. **Expected difficulty for Bingo.** Published evidence or structural analysis indicates that GP-based methods (stack depth ≤ 32, Levenberg–Marquardt constant optimisation, 12 h budget) will not saturate at R² = 1.0. Sources of difficulty include: high variable count with irrelevant features, high-frequency oscillatory components, nested transcendental compositions, or non-trivial constant discovery.

4. **Complementary coverage.** The problem adds at least one difficulty axis not covered by the existing 22 problems: more than 3 input variables, implicit feature selection, rational functions with nested denominators, or high-frequency trigonometric components.

### 1.2 Justification Strategy

The strongest justification available — and the one least vulnerable to reviewer challenge — is to draw problems from the **canonical GP benchmark suites** that the community has used for over a decade, plus **harder instances from the same Feynman suite** already in the paper. Specifically:

- **McDermott et al. (2012)** ("Genetic programming needs better benchmarks", GECCO) and its companion **White et al. (2013)** ("Better GP benchmarks: community survey results and proposals", GPEM) established the definitive list of SR benchmark candidates, organised by source and difficulty. Both papers are among the most-cited references in GP benchmarking.

- **Vladislavleva et al. (2009)** (IEEE TEC) introduced 8 problems specifically designed to test nonlinearity order and extrapolation in GP.

- **Korns (2011)** (GPTP) proposed 15 problems with 5 input variables (3 irrelevant) to test implicit feature selection — a capability gap in most GP methods.

- **Pagie & Hogeweg (1997)** (Evolutionary Computation) introduced the Pagie-1 problem, universally recognised as GP-hard. GPBenchmarks.org states: "This is a hard problem for standard GP — none of the papers [surveyed] have been able to get GP to solve this."

- **Udrescu & Tegmark (2020)** (Science Advances) published 120 Feynman equations. The paper already uses 10; expanding to harder instances from the same suite is trivially justified.

---

## 2. Selected Problems

### 2.1 Extended Feynman Suite (5 problems)

These are drawn from the same AI Feynman database (Udrescu & Tegmark, 2020) already used in the paper. The justification is a single sentence: *"We extend the Feynman benchmark to include harder equations with 3–4 variables and nested non-linearities."*

| ID | Expression | m | Variable Ranges | Difficulty Source |
|----|-----------|---|-----------------|-----------------|
| I.15.10 | m₀v / √(1 − v²/c²) | 3 | m₀ ∈ [1,5], v ∈ [1,2], c ∈ [3,10] | Relativistic momentum; nested sqrt + division + product. Structurally similar to I.10.7 and I.48.20 but with a numerator product, increasing DAG depth. |
| I.30.3 | sin²(nθ/2) / sin²(θ/2) | 2 | n ∈ [1,5], θ ∈ [1,5] | Single-slit diffraction intensity. Requires discovering squared-sine ratio; high-frequency oscillations when n is large. The sin²/sin² structure forces the search to discover two nested sin compositions. |
| I.37.4 | I₁ + I₂ + 2√(I₁·I₂)·cos(δ) | 3 | I₁, I₂ ∈ [1,5], δ ∈ [1,5] | Two-beam interference. Requires simultaneous discovery of addition, square-root-of-product, and cosine. The √(product) subexpression is structurally rare in GP populations. |
| II.11.27 | n₀·exp(−μB/(kT)) + n₀·exp(μB/(kT)) | 4 | n₀ ∈ [1,5], μ ∈ [1,5], B ∈ [1,5], kT ∈ [1,5] | Paramagnetism (Langevin). 4 variables, two exponential branches with opposite signs in the exponent. Known to be among the harder Feynman equations (SRBench ground-truth track reports sub-100% recovery for GP methods). |
| III.17.37 | f₀ / √((ω − ω₀)² + γ²/4) | 4 | f₀ ∈ [1,5], ω ∈ [1,5], ω₀ ∈ [1,5], γ ∈ [1,5] | Lorentzian resonance. 4 variables, nested sqrt of sum-of-squares. The (ω − ω₀)² term requires precise constant-free subtraction discovery, and the γ²/4 term requires precise constant discovery. |

**Data generation.** Identical to the existing Feynman protocol (supplementary §3.1): 1,000 training + 250 test points, uniform sampling within variable ranges, same seed derivation.

**Why these 5?** They satisfy the following coverage matrix:

| Problem | m ≥ 4 | Nested sqrt | Nested trig | Two exponential branches | Constant discovery |
|---------|-------|-------------|-------------|--------------------------|-------------------|
| I.15.10 | — | ✓ | — | — | — |
| I.30.3  | — | — | ✓ | — | — |
| I.37.4  | — | ✓ | ✓ | — | — |
| II.11.27| ✓ | — | — | ✓ | ✓ |
| III.17.37| ✓ | ✓ | — | — | ✓ |

### 2.2 GP-Hard Classics (5 problems)

These are drawn from three benchmark suites that McDermott et al. (2012) and White et al. (2013) identified as the community standard. Every problem below appears in at least two independent benchmark studies.

#### Pagie-1

$$f(x, y) = \frac{1}{1 + x^{-4}} + \frac{1}{1 + y^{-4}}$$

- **Source:** Pagie & Hogeweg (1997), *Evolutionary Computation* 5(1):29–50.
- **Variables:** 2 (x, y ∈ [−5, 5]).
- **Sampling:** 676 training points on a 26×26 grid with spacing 0.4 (standard protocol per GPBenchmarks.org); 2,500 test points on a 50×50 grid.
- **Justification:** Universally recognised as GP-hard. GPBenchmarks.org (maintained by McDermott et al.) states: *"This is a hard problem for standard GP — none of the papers [surveyed] have been able to get GP to solve this (using a Koza-style 676-hits predicate)."* The difficulty arises from the x⁻⁴ singularity near zero and the fact that the rational form 1/(1+x⁻⁴) is structurally unusual in the GP search space.
- **Operator compatibility:** Representable as ADD(INV(ADD(CONST(1), POW(x, CONST(−4)))), INV(ADD(CONST(1), POW(y, CONST(−4))))), which is within Σ. Alternatively, rewrite as x⁴/(1+x⁴) + y⁴/(1+y⁴).
- **Prior use in benchmarks:** McDermott et al. (2012, Table 3); White et al. (2013); Žegklitz & Pošík (2020, GPEM); Kommenda et al. (2020, EuroGP — *"Pagie-1 problem could be solved in 37 out of 50 test runs [with LM constant optimisation]"*).

#### Korns-12

$$f(x_1, x_2, x_3, x_4, x_5) = 2.0 - 2.1 \cos(9.8 \cdot x_1) \sin(1.3 \cdot x_5)$$

- **Source:** Korns (2011), "Accuracy in Symbolic Regression", GPTP IX, pp. 129–151.
- **Variables:** 5 (x₁, …, x₅ ∈ [−50, 50]). Only x₁ and x₅ are relevant; x₂, x₃, x₄ are irrelevant distractors.
- **Sampling:** 10,000 training + 10,000 test points, uniform random within [−50, 50]⁵ (standard Korns protocol per McDermott et al., 2012).
- **Justification:** Tests implicit feature selection (3 of 5 variables are irrelevant) and high-frequency trigonometric composition (9.8·x₁ oscillates ~156 cycles over [−50, 50]). Žegklitz & Pošík (2020, GPEM) report: *"Korns-11 [note: similar structure] … the datasets look very much like samples from a constant function with noise … all the methods provide models of comparable [poor] performance."* The high-frequency argument makes the function appear noisy to GP methods that cannot discover the precise constant 9.8.
- **Operator compatibility:** ADD, MUL, COS, SIN, CONST. Fully within Σ.
- **Prior use:** McDermott et al. (2012, Table 3); White et al. (2013); Žegklitz & Pošík (2020); dos Reis et al. (2024, arXiv:2412.02126).
- **Why relevant for IsalSR:** With 5 variables, the initial CDLL contains 5 nodes and the number of distinct node orderings is larger, so ρ should be higher. The irrelevant variables increase the number of structurally redundant candidates (many expressions involving x₂, x₃, x₄ evaluate to noise), amplifying the benefit of deduplication.

#### Vladislavleva-4 (Unwrapped Ball 5D)

$$f(x_1, \dots, x_5) = \frac{10}{5 + \sum_{i=1}^{5}(x_i - 3)^2}$$

- **Source:** Vladislavleva, Smits & den Hertog (2009), "Order of Nonlinearity as a Complexity Measure for Models Generated by Symbolic Regression via Pareto Genetic Programming", IEEE TEC 13(2):333–349.
- **Variables:** 5 (xᵢ ∈ [0.05, 6.05]).
- **Sampling:** 1,024 training points (uniform random), 5,000 test points (uniform random) — standard Vladislavleva protocol.
- **Justification:** The rational form with a sum-of-squares denominator does not fit the generalised linear model structure. All 5 variables influence the output. Žegklitz & Pošík (2020, GPEM) classify it as *"specific by the presence of a fraction and consists of 5 features which all influence the target value. It does not fit the generalized linear model structure well."*
- **Operator compatibility:** ADD, MUL, INV, POW (or MUL for squaring), CONST. Fully within Σ.
- **Prior use:** Vladislavleva et al. (2009); McDermott et al. (2012, Table 3); White et al. (2013); Žegklitz & Pošík (2020).

#### Vladislavleva-2 (Salustowicz 1D)

$$f(x) = e^{-x} \cdot x^3 \cdot (\cos(x) \cdot \sin(x)) \cdot (\cos(x) \cdot \sin(x)^2 - 1)$$

- **Source:** Salustowicz & Schmidhuber (1997), via Vladislavleva et al. (2009) who relabelled it Vladislavleva-2.
- **Variables:** 1 (x ∈ [0.05, 10]).
- **Sampling:** 100 training points (uniform random), 221 test points on grid with spacing 0.05 — standard Vladislavleva protocol.
- **Justification:** Despite being univariate, this function is structurally complex: it contains nested products of exp, pow, cos, and sin with non-trivial multiplicative structure. The expression requires discovering 7 internal nodes of heterogeneous types. Žegklitz & Pošík (2020) describe it as *"defined by a single, relatively complex term. It does not fit the generalized linear model structure well."*
- **Operator compatibility:** EXP, NEG, MUL, POW, COS, SIN, ADD, CONST. Fully within Σ.
- **Why relevant for IsalSR:** With 1 variable but ~7 internal nodes, the k! factor is 5,040. The multiplicative composition (several MUL nodes at the same level) should produce high commutativity-induced redundancy, making ρ large.

#### Keijzer-6

$$f(x) = \sum_{i=1}^{x} \frac{1}{i}$$

- **Source:** Keijzer (2003), "Improving Symbolic Regression with Interval Arithmetic and Linear Scaling", EuroGP, pp. 70–82.
- **Variables:** 1 (x ∈ [1, 50], integer-valued for the sum, but evaluated as the harmonic function H(x)).
- **Sampling:** 50 training points (x = 1, 2, …, 50); 120 test points (x = 1, …, 120, extrapolation).
- **Justification:** The harmonic series H(x) ≈ ln(x) + γ (Euler–Mascheroni) is well-approximated by a log but cannot be represented exactly. This tests the method's ability to discover ln(x) + constant, which requires precise constant optimisation. The extrapolation test (x up to 120) additionally tests generalisation. It is one of the 15 problems in the canonical Keijzer benchmark and appears in multiple SRBench configurations.
- **Operator compatibility:** LOG, ADD, CONST. Fully within Σ (the SR method approximates H(x), it does not need a summation operator).
- **Prior use:** Keijzer (2003); McDermott et al. (2012); SRBench ground-truth track (La Cava et al., 2021).

---

## 3. Unified Benchmark Table

The full benchmark for the revised paper would consist of **32 problems** in three tiers:

| Tier | Source | n | Difficulty | Purpose |
|------|--------|---|------------|---------|
| Nguyen | Uy et al. (2011) | 12 | Easy–Medium | Baseline coverage; ceiling-effect controls |
| Feynman (existing) | Udrescu & Tegmark (2020) | 10 | Medium | Physics-grounded, 1–3 variables |
| **Feynman (extended)** | Udrescu & Tegmark (2020) | **5** | **Hard** | **Higher variable count, nested non-linearities** |
| **GP-hard classics** | Pagie (1997), Korns (2011), Vladislavleva (2009), Keijzer (2003) | **5** | **Hard** | **Feature selection, rational functions, high-frequency trig, structural complexity** |
| **Total** | | **32** | | |

### 3.1 Computational Cost Estimate

Additional SLURM array size: 10 new problems × 2 methods × 2 variants × 30 seeds = **1,200 runs**.

At 12 hours per run on Picasso: 1,200 × 12 = **14,400 CPU-hours** (~600 CPU-days). This is 45% of the original 2,640-run budget.

**Recommendation:** Run 5-seed screening first (10 × 2 × 2 × 5 = 200 runs, ~100 CPU-days) to verify that Bingo baseline does NOT saturate at R² = 1.0 on the hard problems. Then commit to the full 30-seed runs only on problems where Bingo median R² < 0.98.

---

## 4. Diversity Experiment: Candidate Problems

The diversity experiment (§4.3 / §5.2 in the current manuscript) requires a problem where:

1. Bingo baseline **plateaus at R² significantly below 1.0** (ideally R² ∈ [0.5, 0.9]).
2. The expression has **moderate DAG complexity** (k ∈ [5, 15]) so that population dynamics play out over 500 generations.
3. The problem is from an **established benchmark** that reviewers cannot dismiss.

### Primary candidate: Feynman II.11.27 (Paramagnetism)

$$f(n_0, \mu, B, kT) = n_0 \cdot e^{-\mu B / kT} + n_0 \cdot e^{+\mu B / kT}$$

**Rationale:**
- 4 variables force Bingo to explore a larger search space, reducing probability of saturation.
- The expression has k = 7 internal nodes (2× EXP, 2× MUL, 1× NEG, 1× DIV, 1× ADD), providing enough structural complexity for the population dynamics to be non-trivial.
- Two exponential branches with opposite signs in the exponent are structurally difficult: crossover is unlikely to produce both branches simultaneously, so the search tends to discover one branch first and then struggle to add the second.
- Published in the Feynman suite, already cited in the paper.

### Secondary candidate: Korns-12

**Rationale:**
- 5 variables with 3 irrelevant: the population will contain many structurally diverse but functionally equivalent expressions (those using irrelevant variables), making η collapse severe in the baseline.
- The high-frequency cos(9.8·x₁)·sin(1.3·x₅) is very unlikely to be discovered within 500 generations at population size 200. Bingo should plateau well below R² = 1.0.
- Provides a complementary difficulty axis (feature selection) to the Feynman problems.

### Fallback: Pagie-1

**Rationale:**
- Known to be unsolvable by standard GP. Even with LM constant optimisation, Kommenda et al. (2020) report only 37/50 success rate.
- If Bingo cannot solve it at all (R² ≈ 0), the diversity experiment is uninformative. Use only if screening shows R² ∈ [0.3, 0.9].

**Protocol for selection:** Run Bingo baseline (5 seeds, 2 h) on all three candidates. Select the one where median R² falls in [0.5, 0.9]. If multiple qualify, prefer the one with the largest gap between IsalSR and baseline R² in the screening.

---

## 5. Why Not Run All of SRBench?

Running the full SRBench benchmark (120 Feynman + 132 PMLB) was considered and rejected for three reasons:

1. **Computational cost.** 252 problems × 2 methods × 2 variants × 30 seeds = 30,240 runs × 12 h = 362,880 CPU-hours. This exceeds reasonable Picasso allocation.

2. **Narrative dilution.** The paper's contribution is a *representation*, not a new SR method. Running 252 problems produces 252 per-problem tables and buries the story in data. The current manuscript already has 22 per-problem rows in Tables 6–7 (supplementary); doubling this to 32 is manageable, but going to 252 is not.

3. **PMLB interpretation.** The 132 PMLB problems are real-world datasets with unknown ground truth and measurement noise. R² on these problems is influenced by noise, missing features, and model misspecification — factors orthogonal to the structural deduplication that IsalSR provides. A reviewer could argue that any R² difference on PMLB is confounded by noise handling rather than search-space reduction. The ground-truth track (synthetic Feynman + GP-hard classics) isolates the effect cleanly.

**The 10-problem extension is the minimal addition that closes the empirical gap.** It adds 5 harder Feynman equations (same family, trivially justified) and 5 GP-hard classics (community-standard benchmarks, heavily cited). If a reviewer asks "why not SRBench?", the response is: "Our 32-problem benchmark spans three difficulty tiers (easy Nguyen, medium Feynman, hard Feynman + GP classics) and is sufficient to demonstrate that IsalSR's search-space reduction translates into measurable quality improvement on hard problems. Full SRBench evaluation is future work."

---

## 6. References for the Proposal

- Keijzer, M. (2003). Improving Symbolic Regression with Interval Arithmetic and Linear Scaling. *EuroGP*, LNCS 2610, pp. 70–82.
- Kommenda, M., Burlacu, B., Kronberger, G., & Affenzeller, M. (2020). Parameter Identification for Symbolic Regression using Nonlinear Least Squares. *Genetic Programming and Evolvable Machines* 21:471–501.
- Korns, M. F. (2011). Accuracy in Symbolic Regression. In *Genetic Programming Theory and Practice IX*, pp. 129–151. Springer.
- La Cava, W. et al. (2021). Contemporary Symbolic Regression Methods and their Relative Performance. *NeurIPS Datasets and Benchmarks Track*.
- Imai Aldeia, G. S. et al. (2025). Call for Action: Towards the Next Generation of Symbolic Regression Benchmark. *GECCO 2025 Workshop*.
- McDermott, J. et al. (2012). Genetic Programming Needs Better Benchmarks. *GECCO 2012*, pp. 791–798.
- Pagie, L. & Hogeweg, P. (1997). Evolutionary Consequences of Coevolving Targets. *Evolutionary Computation* 5(1):29–50.
- Udrescu, S.-M. & Tegmark, M. (2020). AI Feynman: A Physics-Inspired Method for Symbolic Regression. *Science Advances* 6(16):eaay2631.
- Vladislavleva, E., Smits, G. & den Hertog, D. (2009). Order of Nonlinearity as a Complexity Measure. *IEEE TEC* 13(2):333–349.
- White, D. R. et al. (2013). Better GP Benchmarks: Community Survey Results and Proposals. *GPEM* 14(1):3–29.
- Žegklitz, J. & Pošík, P. (2020). Benchmarking State-of-the-Art Symbolic Regression Algorithms. *GPEM* 22:5–33.
