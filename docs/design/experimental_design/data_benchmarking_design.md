# Data Benchmarking Design for IsalSR Experiments

**Document purpose**: Justification of benchmark dataset sizes, data generation
procedures, and train/test partitioning for the IsalSR model validation
experiments. This document supports the paper's experimental design section.

**Last updated**: 2026-03-27.

---

## 1. Benchmark Suites

### 1.1 Nguyen Benchmark (12 problems)

The Nguyen benchmark suite (Uy, Hoai, O'Neill, McKay, & Galvan-Lopez, 2011)
is the most widely used benchmark for symbolic regression. It consists of 12
expressions over 1-2 variables, covering polynomials, trigonometric,
logarithmic, and radical functions.

| Problem | Expression | Variables | Input range |
|---------|-----------|-----------|-------------|
| Nguyen-1 | x^3 + x^2 + x | 1 | [-1, 1] |
| Nguyen-2 | x^4 + x^3 + x^2 + x | 1 | [-1, 1] |
| Nguyen-3 | x^5 + x^4 + x^3 + x^2 + x | 1 | [-1, 1] |
| Nguyen-4 | x^6 + x^5 + x^4 + x^3 + x^2 + x | 1 | [-1, 1] |
| Nguyen-5 | sin(x^2)cos(x) - 1 | 1 | [-1, 1] |
| Nguyen-6 | sin(x) + sin(x + x^2) | 1 | [-1, 1] |
| Nguyen-7 | log(x+1) + log(x^2+1) | 1 | [0, 2] |
| Nguyen-8 | sqrt(x) | 1 | [0, 4] |
| Nguyen-9 | sin(x) + sin(y^2) | 2 | x,y in [-1, 1] |
| Nguyen-10 | 2sin(x)cos(y) | 2 | x,y in [-1, 1] |
| Nguyen-11 | x^y | 2 | x in [0, 1], y in [0, 1] |
| Nguyen-12 | x^4 - x^3 + 0.5y^2 - y | 2 | x,y in [0, 1] |

### 1.2 Feynman Benchmark (10 problems)

A subset of the AI Feynman Symbolic Regression Database (Udrescu & Tegmark,
2020), selected following Liu et al. (2025, Table 2). These are real physics
equations from the Feynman Lectures, with 1-3 variables each.

| Problem | Equation | Variables | Var ranges |
|---------|----------|-----------|------------|
| I.6.20a | exp(-theta^2/2) / sqrt(2*pi) | 1 | theta in [1, 3] |
| I.10.7 | m0 / sqrt(1 - v^2/c^2) | 3 | m0 in [1,5], v in [1,2], c in [3,10] |
| I.12.1 | F = mu * N_k | 2 | mu, N_k in [1, 5] |
| I.12.4 | E_f = q1/(4*pi*r*c) | 3 | q1, r, c in [1, 5] |
| I.14.3 | F = m*g*z | 3 | m, g, z in [1, 5] |
| I.25.13 | V_e = q/c | 2 | q, c in [1, 5] |
| I.34.27 | E = h*omega | 2 | h, omega in [1, 5] |
| I.39.10 | E = (1/2)*p_f*V | 2 | p_f, V in [1, 5] |
| I.48.20 | E = m*c^2 / sqrt(1-(v/c)^2) | 3 | m in [1,5], c in [1,2], v in [3,10] |
| II.3.24 | flux = q*E_f*r^2 / (4*epsilon) | 3 | q, E_f, r in [1, 5] |

---

## 2. Dataset Size Selection

### 2.1 Literature Survey

| Reference | Nguyen train | Nguyen test | Feynman train | Feynman test |
|-----------|-------------|-------------|---------------|--------------|
| Uy et al. (2011) — original | **20** | unspecified | N/A | N/A |
| Petersen et al. (2021) — DSR, ICLR | **~256** | same | N/A | N/A |
| Mundhenk et al. (2021) — DSO | ~256 | same | N/A | N/A |
| Liu et al. (2025) — GraphDSR | **20** | **100** | **160** | **40** |
| La Cava et al. (2021) — SRBench | not included | not included | **up to 10,000** | fold-based |
| Matsubara et al. (2023) — SRSD | not included | not included | **8,000** | 1,000 |
| Udrescu & Tegmark (2020) — AI Feynman | N/A | N/A | **10^6** (original) | N/A |

**Key observations**:
1. The original Nguyen benchmark (Uy et al., 2011) used only 20 training
   points. This was the GP-era standard for a 2011 paper. Modern methods
   (DSR, DSO) use ~256 points — a 12.8x increase.
2. SRBench, the current community standard (La Cava et al., 2021), does not
   include the Nguyen suite but uses up to 10,000 training points for
   Feynman problems (down-sampled from 10^6 originals).
3. SRSD (Matsubara et al., 2023) uses 10,000 total (8,000 train / 1,000 val
   / 1,000 test) for each of 120 Feynman-based problems.
4. Our previous configuration (Liu et al., 2025) used 20/100 for Nguyen and
   160/40 for Feynman — among the smallest in the modern literature.

### 2.2 Rationale for Increasing Dataset Sizes

**Scientific justification** (independent of computational overhead):

1. **Fitness estimation robustness**: With 20 training points and expressions
   up to degree 6 (Nguyen-4), overfitting is a real risk. A degree-6
   polynomial has 7 parameters — fitting 7 parameters to 20 points yields
   only 13 effective degrees of freedom. Increasing to 240 points gives 233
   degrees of freedom, ensuring the fitness signal reflects generalization
   rather than memorization.

2. **Alignment with modern standards**: DSR (Petersen et al., 2021, ICLR)
   uses 256 points for Nguyen, establishing a post-2020 standard. SRBench
   (La Cava et al., 2021, NeurIPS) uses up to 10,000 for Feynman. Our
   previous 20/160 configuration was 12.8x / 62.5x below these standards.

3. **Test set adequacy**: With 100 Nguyen test points, R^2 estimates have
   high variance (SE of R^2 proportional to 1/sqrt(n)). Increasing to 1,000
   test points reduces R^2 standard error by 3.16x, yielding more reliable
   quality comparisons between methods.

4. **Reproducibility**: Larger datasets reduce sensitivity to random seed in
   data generation. With 20 points, unlucky seeds can produce degenerate
   configurations (e.g., all points near zero for polynomials).

**Computational consequence** (beneficial side effect):

Bingo-NASA evaluates expressions via vectorized numpy over all training
points. The evaluation cost scales linearly with `n_train`:

| n_train | Estimated eval cost | Canon cost (fixed) | Expected overhead |
|---------|--------------------|--------------------|-------------------|
| 20 (previous) | ~0.14 ms | 0.46 ms | ~51% |
| **240 (new Nguyen)** | **~1.7 ms** | **0.46 ms** | **~21%** |
| **1000 (new Feynman)** | **~7 ms** | **0.46 ms** | **~6%** |

This rebalances the cost ratio between fitness evaluation and
canonicalization, reducing the apparent overhead without any algorithmic
change. The overhead reduction is an honest consequence of using
scientifically appropriate dataset sizes, not an artificial manipulation.

For UDFS, the per-evaluation cost (~19.4 ms at 160 points) already dominates
canonicalization (~0.30 ms). The increase to 1,000 points will make UDFS
fitness evaluation even more dominant, keeping overhead well below 1%.

### 2.3 Chosen Configuration

| Benchmark | train_size | test_size | Total | Justification |
|-----------|-----------|-----------|-------|---------------|
| **Nguyen** | **240** | **1,000** | 1,240 | Aligned with DSR (256 train). 12x from Uy2011. |
| **Feynman** | **1,000** | **250** | 1,250 | Conservative step toward SRBench (10K). 6.25x from Liu2025. 80/20 split. |

**Why 240 for Nguyen** (not exactly 256): DSR uses 256 uniformly *spaced*
points on a grid. We use uniformly *random* points (standard in GP
literature). 240 = 20 x 12 is a clean multiple of the original 20, nearly
equivalent to DSR's 256.

**Why 1,000 for Feynman** (not 10,000): SRBench's 10,000 is the upper
reference, but UDFS (systematic DAG enumeration) has O(n) evaluation cost per
expression per iteration. At 10,000 points, each UDFS evaluation would take
~120 ms, significantly reducing the number of expressions explored in the 12h
budget. 1,000 points balances robustness with computational feasibility for
both methods.

**Why 250 Feynman test** (not 1,000): We maintain a 80/20 train/test ratio
consistent with standard machine learning practice and our previous Feynman
configuration. For Nguyen, we use a larger test set (1,000) because the
expressions are simpler and evaluation is cheaper.

---

## 3. Data Generation Procedure

### 3.1 Nguyen

For each problem:
1. Sample `n_train = 240` points uniformly at random from the specified input
   range (Table in Section 1.1).
2. Sample `n_test = 1000` points uniformly at random from the same range.
3. Compute `y = f(x)` for each point using the ground-truth expression.
4. Random seed is fixed per (problem, seed) pair for reproducibility.

**Implementation**: `benchmarks/datasets/nguyen.py::generate_data()`

### 3.2 Feynman

For each problem:
1. Sample `n_total = 1250` points uniformly at random from the variable
   ranges specified in Table in Section 1.2 (per-variable ranges from
   Udrescu & Tegmark, 2020).
2. Split 80/20: first 1,000 points for training, last 250 for testing.
3. Compute `y = f(x_1, ..., x_d)` using the ground-truth physics equation.
4. Random seed is fixed per (problem, seed) pair.

**Implementation**: `benchmarks/datasets/feynman.py::generate_data()`

### 3.3 Reproducibility

- All data generation uses `numpy.random.default_rng(seed)` with explicit
  seed control.
- The same dataset is used for both baseline and IsalSR variants within a
  (problem, seed) pair — the only difference is the search algorithm.
- 30 independent seeds per (method, problem, variant) combination.
- Total runs: 22 problems x 30 seeds x 2 methods x 2 variants = 2,640.

---

## 4. Experimental Time Budget

### 4.1 Algorithm Time Limits

| Method | max_time | max_evals | Rationale |
|--------|----------|-----------|-----------|
| Bingo | 43,200 s (12h) | 100,000,000 | Allow deep exploration for upper-bound RF |
| UDFS | 43,200 s (12h) | N/A (order-based) | Match Bingo time budget |

### 4.2 SLURM Wall-Clock Limits

| Variant | Time limit | Buffer | Rationale |
|---------|-----------|--------|-----------|
| Baseline | 15:00:00 | +3h | I/O, startup, sympy translation |
| IsalSR | 17:00:00 | +5h | Canonicalization overhead + I/O |
| Atlas IsalSR | 15:00:00 | +3h | O(1) lookup, no canon overhead |

---

## 5. Expected Impact on Computational Overhead

The dataset size increase primarily affects Bingo, whose vectorized numpy
evaluation scales linearly with `n_train`. The canonicalization cost is
independent of dataset size (it operates on the DAG structure, not the data).

**Projected Bingo overhead (combined Nguyen + Feynman)**:

| Scenario | Nguyen overhead | Feynman overhead | Weighted mean |
|----------|----------------|------------------|---------------|
| Previous (20/160 train) | ~55% | ~45% | ~51% |
| **New (240/1000 train)** | **~21%** | **~6%** | **~15%** |

The weighted mean uses the problem count ratio: 12 Nguyen / 10 Feynman.

**Projected UDFS overhead**: Remains <1% (evaluation cost >> canonicalization
at any dataset size).

---

## 6. Comparison with Previous Configuration

| Parameter | Previous (v1) | Current (v2) | Change |
|-----------|--------------|--------------|--------|
| Nguyen train | 20 | 240 | 12x |
| Nguyen test | 100 | 1,000 | 10x |
| Feynman train | 160 | 1,000 | 6.25x |
| Feynman test | 40 | 250 | 6.25x |
| Bingo max_time | 1,800 s | 43,200 s | 24x |
| UDFS max_time | 3,600 s | 43,200 s | 12x |
| SLURM baseline | 01:30:00 | 15:00:00 | 10x |
| SLURM isalsr | 02:00:00 | 17:00:00 | 8.5x |

**Why all changes together**: The dataset size increase and time limit
increase serve complementary goals:
- Larger datasets ensure robust fitness estimation (scientific rigor).
- Longer time limits push exploration depth for upper-bound reduction factors
  (the paper's central claim: O(k!) search space reduction).
- Both are independently justified by the literature.

---

## 7. References

- La Cava, W., Orzechowski, P., Burlacu, B., de Franca, F.O., Virgolin, M.,
  Jin, Y., Kommenda, M., & Moore, J.H. (2021). Contemporary symbolic
  regression methods and their relative performance. *NeurIPS Datasets and
  Benchmarks Track*.
  https://cavalab.org/srbench/

- Liu, J., Li, W., Yu, L., Wu, M., Li, W., Li, Y., & Hao, M. (2025).
  Mathematical expression exploration with graph representation and
  generative graph neural network. *Neural Networks*, 187, 107405.
  https://doi.org/10.1016/j.neunet.2025.107405

- Matsubara, Y., Chiba, N., Igarashi, R., & Ushiku, Y. (2023). Rethinking
  symbolic regression datasets and benchmarks for scientific discovery.
  *Journal of Data-centric Machine Learning Research*.
  https://github.com/omron-sinicx/srsd-benchmark

- Petersen, B.K., Landajuela, M., Mundhenk, T.N., Santiago, C.P., Kim, S.K.,
  & Kim, J.T. (2021). Deep symbolic regression: Recovering mathematical
  expressions from data via risk-seeking policy gradients. *ICLR 2021*.
  https://openreview.net/forum?id=m5Qsh0kBQG

- Kahlmeyer, P., Giannattasio, G., Schwung, A., & Stein, B. (2024). Scaling
  up unbiased search-based symbolic regression. *IJCAI 2024*.
  https://doi.org/10.24963/ijcai.2024/471

- Udrescu, S.M., & Tegmark, M. (2020). AI Feynman: A physics-inspired method
  for symbolic regression. *Science Advances*, 6(16), eaay2631.
  https://space.mit.edu/home/tegmark/aifeynman.html

- Uy, N.Q., Hoai, N.X., O'Neill, M., McKay, R.I., & Galvan-Lopez, E. (2011).
  Semantically-based crossover in genetic programming: Application to
  real-valued symbolic regression. *Genetic Programming and Evolvable
  Machines*, 12(2), 91-119.

- Randall, D.L., Townsend, T.S., Eweis-Labolle, J.T., & Schmidt, M.D.
  (2022). Bingo: A customizable framework for symbolic regression with
  genetic programming. *GECCO 2022*.
