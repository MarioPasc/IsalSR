# Candidate Problem Screening for IsalSR Experiments

**Date**: 2026-04-20
**Authors**: Mario Pascual González, with Claude (analysis assistant)
**Status**: Screening complete; candidates identified for future experiments
**Depends on**: `docs/md_files/changes/bottleneck_type_analysis.md`

---

## Screening Criterion

From the bottleneck-type analysis (2026-04-19), IsalSR's canonical deduplication
helps **if and only if** the problem's bottleneck is **structural search** — finding
the right operator topology. Two computable variables predict this:

1. **`n_nontrivial_constants = 0`**: all constants in the ground truth are small
   integers {0, 1, 2, 3, 4} that LM discovers trivially.
2. **`k >= 5`**: sufficient structural complexity (operator nodes) for dedup to
   matter. Higher k → more k! isomorphic copies → more potential benefit.

**There is no upper bound on k.** Higher k gives MORE potential advantage
(more isomorphic copies to deduplicate). The k=13 failure of Vlad-2 was due
to structural_depth confounds, not k itself.

### Convention: Integer Exponents as Trivial Constants

POW(x, n) nodes have integer exponents as CONST leaves. We classify all integer
exponents ≤ 10 as "effectively trivial" because LM converges to nearby integers
easily. This is consistent with the project's existing classification (e.g.,
Pagie-1 with x⁻⁴ has n_nontrivial = 0; Nguyen-4 with x⁶ has n_nontrivial = 0).
Non-integer exponents (e.g., x^1.5) or very large integer exponents would be
nontrivial.

Nontrivial constants include: π, e, γ (Euler-Mascheroni), 0.5772, 9.8, 1.3,
2.1, 0.5 (as a multiplicative coefficient), and any integer > 10.

---

## Suites Scanned

| Suite | Source | Total problems | Candidates | New candidates |
|---|---|---|---|---|
| **AI Feynman** | Udrescu & Tegmark (2020), Science Advances | ~100 | 18 | **15** |
| **SRBench ground-truth** | La Cava et al. (2021), NeurIPS | 29 | 12 | 2 (Keijzer-4, Keijzer-11) |
| **Vladislavleva** | Vladislavleva et al. (2009), IEEE TEC | 8 | 2 | **1** (Vlad-7) |
| **Korns** | Korns (2011), GPTP IX | 15 | 1 | **1** (Korns-9) |
| **Keijzer (full)** | Keijzer (2003), EuroGP | 15 | 3 | 1 (Keijzer-12) |
| **Pagie** | Pagie & Hogeweg (1997) | 2 | 2 | **1** (Pagie-2) |
| **R (rational)** | SRBench/DSO, Koza tradition | 3 | 3 | **3** (R1, R2, R3) |
| **Jin** | Jin et al. (2019) | 6 | 0 | 0 |
| **DSO-Livermore** | Mundhenk et al. (2021), ICLR | 23 | ~8 | **5** |
| **Koza** | Koza (1992) | 3 | 3 | 0 (duplicates) |

**Total genuinely new candidates: ~29** (after deduplication across suites).

---

## Already in Our Suite (32 Problems)

For reference, these problems are already in our benchmark and are NOT candidates:

| Tier | Problems | Status |
|---|---|---|
| Nguyen (12) | Nguyen 1–12 | Trivially solved by Bingo (R² ≈ 1.0) |
| Feynman medium (10) | I.6.20a, I.12.1, I.14.3, I.25.13, I.34.27, I.39.10, I.12.4, II.3.24, I.10.7, I.48.20 | Trivially solved |
| Hard Feynman (5) | I.15.10, I.30.3, I.37.4, II.11.27, III.17.37 | Tested; 4/5 structural winners |
| Hard GP (5) | Pagie-1, Korns-12, Vlad-2, Vlad-4, Keijzer-6 | Tested; Pagie-1 structural winner |

---

## New Candidates by Suite

### A. AI Feynman (15 new candidates)

Source: Udrescu & Tegmark (2020). Equations verified against the AI Feynman
dataset (space.mit.edu/home/tegmark/aifeynman.html) and florianBachinger/
FeynmanEquations-Python-JDIQ.

**Disqualified Feynman equations**: I.26.2 (arcsin), I.30.5 (arcsin),
II.35.21 (tanh). ~56 equations fail n_nontrivial > 0 (mostly due to π in
4πε₀, h/2π, etc.). ~10 fail k < 5.

| ID | Expression | n_vars | k | Structural profile | Priority |
|---|---|---|---|---|---|
| **I.29.16** | sqrt(x₁²+x₂²−2x₁x₂cos(θ₁−θ₂)) | 4 | 11 | cos inside sqrt-of-sum; law of cosines | **HIGH** |
| **I.50.26** | x₁(cos(ωt)+α·cos²(ωt)) | 4 | 8 | Repeated cos subtree; nonlinear oscillation | **HIGH** |
| **I.16.6** | (u+v)/(1+uv/c²) | 3 | 6 | Relativistic velocity addition; unique topology | **HIGH** |
| **II.11.28** | 1+nα/(1−nα/3) | 2 | 6 | Clausius-Mossotti; nested rational | **HIGH** |
| **III.10.19** | mom·sqrt(Bx²+By²+Bz²) | 4 | 7 | L2-norm × scalar; 3D vector structure | **MEDIUM** |
| **II.11.3** | qEf/(m(ω₀²−ω²)) | 5 | 6 | Resonance denominator (ω₀²−ω²) | **MEDIUM** |
| **III.14.14** | I₀(exp(qV/kT)−1) | 5 | 6 | Shockley diode; exp-minus-1 branch | **MEDIUM** |
| **I.18.4** | (m₁r₁+m₂r₂)/(m₁+m₂) | 4 | 5 | Center of mass; clean k=5 benchmark | **MEDIUM** |
| **I.12.11** | q(Ef+Bv·sin(θ)) | 5 | 5 | Lorentz force scalar; k=5 with sin | **MEDIUM** |
| **I.44.4** | nkbT·ln(V₂/V₁) | 5 | 5 | Isothermal work; only candidate with log | **MEDIUM** |
| **I.13.12** | Gm₁m₂(1/r₂−1/r₁) | 5 | 6 | Gravitational PE difference; inv nodes | **MEDIUM** |
| **II.11.20** | nρpd²Ef/(3kbT) | 5 | 6 | Langevin polarizability; rational | **LOW** |
| **II.11.17** | n₀(1+pdEfcos(θ)/(kbT)) | 6 | 7 | Boltzmann density; 6 vars + cos | **LOW** |
| **I.40.1** | n₀exp(−mgx/(kbT)) | 6 | 7 | Barometric formula; 6 vars + exp | **LOW** |
| **I.9.18** | Gm₁m₂/((x₂−x₁)²+(y₂−y₁)²+(z₂−z₁)²) | 9 | 11 | Newton gravity 3D; 9 vars (Bingo only) | **LOW** |

**Lorentz-gamma family** (omitted — structurally redundant with I.10.7, I.15.10, I.15.3t):
I.15.1 (k=7, 3v), I.48.2 (k=8, 3v), I.34.14 (k=9, 3v), I.15.3x (k=8, 4v).
Select at most 1 if Lorentz representation is needed. Best candidate: I.34.14
(k=9, most complex numerator 1+v/c).

**Excluded**: I.11.19 (3D dot product, k=5, trivially solved); II.36.38 (8 vars,
feature selection bottleneck); II.35.18 (sech-like, constant-adjacent bottleneck).

### B. Classic GP Suites

#### Vladislavleva (8 problems, 1 new candidate)

| ID | Expression | n_vars | k | n_nontrivial | Passes? | Notes |
|---|---|---|---|---|---|---|
| Vlad-1 | exp(−(x₁−1)²)/(1.2+(x₂−2.5)²) | 2 | 6 | 2 (1.2, 2.5) | No | |
| Vlad-2 | exp(−x)x³cos(x)sin(x)(cos(x)sin²(x)−1) | 1 | 13 | 0 | Already in suite | |
| Vlad-3 | Vlad-2 × (x₂−5) | 2 | 15 | 1 (constant 5) | No | |
| Vlad-4 | 10/(5+Σ(xᵢ−3)²) | 5 | 12 | 2 (10, 5) | Already in suite | |
| Vlad-5 | 30(x₁−1)(x₃−1)/((x₁−10)x₂²) | 3 | 8 | 2 (30, 10) | No | |
| Vlad-6 | 6sin(x₁)cos(x₂) | 2 | 3 | 1 (6) | No | k < 5 |
| **Vlad-7** | **(x₁−3)(x₂−3)+2sin(x₁−4)(x₂−4)** | **2** | **9** | **0** | **Yes** | **NEW** |
| Vlad-8 | ((x₁−3)⁴+(x₂−3)³−(x₂−3))/((x₂−2)⁴+10) | 2 | 11 | 1 (10) | No | |

**Vlad-7** is the standout: bivariate, k=9, mixed product + sin structure, all
constants {2, 3, 4} are trivial. Published in Vladislavleva et al. (2009).

#### Korns (15 problems, 1 new candidate)

| ID | Expression | n_nontrivial | Passes? |
|---|---|---|---|
| Korns-1 | 1.57+24.3x₄ | 2 | No |
| Korns-2 | 0.23+14.2(x₄+x₂)/(3x₅) | 2 | No |
| Korns-3 | 4.9(x₄−x₁+x₂/x₅)/(3x₅)−5.41 | 2 | No |
| Korns-4 | 0.13sin(x₃)−2.3 | 2 | No |
| Korns-5 | 3+2.13ln(|x₅|) | 1 | No |
| Korns-6 | 1.3+0.13sqrt(|x₁|) | 2 | No |
| Korns-7 | 213.809(1−exp(−0.547238x₁)) | 2 | No |
| Korns-8 | 6.87+11sqrt(|7.23x₁x₄x₅|) | 3 | No |
| **Korns-9** | **sqrt(\|x₁\|)/ln(\|x₂\|)·exp(x₃)/x₄²** | **0** | **Yes (k≈9)** |
| Korns-10 | 0.81+24.3(2x₂+3x₃²)/(4x₄³+5x₅⁴) | 2 | No |
| Korns-11 | 6.87+11cos(7.23x₁³) | 3 | No |
| Korns-12 | 2−2.1cos(9.8x₁)sin(1.3x₅) | 3 | Already in suite |
| Korns-13 | 32−3tan(...)... | — | Disqualified (tan) |
| Korns-14 | (contains tan) | — | Disqualified (tan) |
| Korns-15 | (contains tan) | — | Disqualified (tan) |

**Korns-9** passes but has a caveat: 5 variables with only 4 active (x₅ irrelevant),
requiring feature selection. Our bottleneck analysis suggests feature selection
bottleneck → IsalSR may not help. However, Korns-9 has only 1 irrelevant variable
(vs. 3 in Korns-12), so the feature selection difficulty is lower. The structural
complexity (k≈9) with heterogeneous operators (sqrt, log, exp, div, pow) makes
this a borderline candidate worth screening.

#### Keijzer (15 problems, 2 new candidates)

| ID | Expression | n_vars | k | n_nontrivial | Passes? |
|---|---|---|---|---|---|
| Keijzer-1/2/3 | 0.3x·sin(2πx) | 1 | 3 | 2 (0.3, π) | No |
| **Keijzer-4** | **x³exp(−x)cos(x)sin(x)(cos(x)sin²(x)−1)** | **1** | **14** | **0** | **Yes** |
| Keijzer-5 | 30x₁x₃/((x₁−10)x₂²) | 3 | 5 | 1 (30) | No |
| Keijzer-6 | H(x) ≈ log(x)+γ | 1 | 2 | 1 (γ) | Already in suite |
| Keijzer-7 | log(x) | 1 | 1 | 0 | No (k=1) |
| Keijzer-8 | sqrt(x) | 1 | 1 | 0 | No (k=1) |
| Keijzer-9 | arcsinh(x) = log(x+sqrt(x²+1)) | 1 | 4 | 0 | No (k=4) |
| Keijzer-10 | x^y | 2 | 1 | 0 | No (k=1) |
| **Keijzer-11** | **xy+sin((x−1)(y−1))** | **2** | **6** | **0** | **Yes** |
| Keijzer-12 | x₁⁴−x₁³+x₂²/2−x₂ | 2 | 9 | 0 | Yes (= Nguyen-12, already tested) |
| Keijzer-13 | 6sin(x₁)cos(x₂) | 2 | 3 | 1 (6) | No |
| Keijzer-14 | 8/(2+x₁²+x₂²) | 2 | 5 | 1 (8) | No |
| Keijzer-15 | x₁³/5+x₂³/2−x₂−x₁ | 2 | 9 | 1 (5) | No |

**Keijzer-4 is structurally identical to Vlad-2** (same expression, different domain:
Keijzer uses [0,10], Vlad-2 uses (0.05,10)). Since Vlad-2 (k=13) is already in
our suite and classified as structural_depth bottleneck, Keijzer-4 is not a
genuinely new structure. However, Keijzer-4's broader domain [0,10] includes x=0
where exp(−x)·x³=0, which may make the problem easier (providing a boundary anchor).
Running Keijzer-4 could test whether Vlad-2's failure was due to domain effects
vs. structural depth.

**Keijzer-11** (k=6, 2 vars) is bivariate with nested trig — a genuinely new
structure not represented in our suite.

#### Pagie (2 problems, 1 new candidate)

| ID | Expression | n_vars | k | n_nontrivial | Passes? |
|---|---|---|---|---|---|
| Pagie-1 | 1/(1+x⁻⁴)+1/(1+y⁻⁴) | 2 | 7 | 0 | Already in suite |
| **Pagie-2** | **1/(1+x⁻⁴)+1/(1+y⁻⁴)+1/(1+z⁻⁴)** | **3** | **10** | **0** | **Yes** |

**Pagie-2** is the 3D extension of Pagie-1. Higher k (10 vs 7) → more dedup
potential. However, it has no standardized sampling protocol (not in DSO or
SRBench). Would need to define a protocol (e.g., 10×10×10 grid on [−5,5]³,
skip zero). Risk: adding a third variable may push the search space beyond
GP feasibility within our 12h budget.

#### R (Rational) Problems (3 new candidates)

Source: DSO benchmarks (Petersen et al., 2021), attributed to Koza tradition.
Domain: 20 evenly-spaced points in [−1, 1].

| ID | Expression | n_vars | k | n_nontrivial | Notes |
|---|---|---|---|---|---|
| **R1** | **(x+1)³/(x²−x+1)** | **1** | **7** | **0** | Rational polynomial, cubic/quadratic |
| **R2** | **(x⁵−3x³+1)/(x²+1)** | **1** | **9** | **0** | Rational polynomial, quintic/quadratic |
| **R3** | **(x⁶+x⁵)/(x⁴+x³+x²+x+1)** | **1** | **11** | **0** | Rational polynomial, sextic/quartic |

R1-R3 are structurally rich rational functions with exclusively small-integer
constants. R3 (k=11) is particularly interesting — it tests whether dedup helps
at high k for rational polynomials. These are pure structural-search problems
(no trig, no transcendentals), making them clean test cases for the bottleneck
hypothesis.

**Caution**: All are univariate, domain [−1,1]. May be trivially solved by Bingo
with stack=32 and LM. 5-seed screening advised.

#### Jin (6 problems, 0 candidates)

All Jin problems either have nontrivial constants or k < 5. **No candidates.**

#### DSO-Livermore Extended (23 problems, 5 new candidates)

Source: Mundhenk et al. (2021), ICLR; DSO benchmarks.csv (Petersen et al., 2021).
Note: this is a DIFFERENT numbering than SRBench's 6-problem Livermore subset.

| ID | Expression | n_vars | k | Notes |
|---|---|---|---|---|
| **Liv-4** | **ln(x+1)+ln(x²+1)+ln(x)** | **1** | **8** | Three-log sum (differs from SRBench Livermore-4) |
| **Liv-14** | **x₁³+x₁²+x₁+sin(x₁)+sin(x₂²)** | **2** | **8** | Polynomial + trig hybrid |
| **Liv-19** | **ln(x²+x)+ln(x³+x)** | **1** | **9** | Nested log-of-polynomial |
| **Liv-9** | **x⁹+x⁸+...+x (9 terms)** | **1** | **16** | Very high k; polynomial degree 9 |
| **Liv-22** | **x⁸+x⁷+...+x (8 terms)** | **1** | **14** | Very high k; polynomial degree 8 |

Livermore-9 and Livermore-22 are interesting for testing the k > 10 regime but
may be trivially solved (univariate polynomials). Livermore-14 is the most
promising: bivariate, mixed polynomial + trig, k=8.

---

## Priority Rankings

### Tier 1: Highest Priority (new structure, k ∈ [6,11], structural bottleneck likely)

These problems offer genuinely new structural challenges not represented in our suite:

| # | Problem | k | n_vars | Why |
|---|---|---|---|---|
| 1 | **I.29.16** (law of cosines) | 11 | 4 | cos inside sqrt-of-sum; unique topology; k=11 tests high-k regime |
| 2 | **I.50.26** (nonlinear oscillation) | 8 | 4 | Repeated cos subtree; tests shared-structure discovery |
| 3 | **Vlad-7** | 9 | 2 | Mixed product + sin; k=9; published benchmark with protocol |
| 4 | **R2** (rational quintic) | 9 | 1 | Pure structural; rational polynomial; clean test case |
| 5 | **I.16.6** (relativistic velocity) | 6 | 3 | Nested rational; unique topology (no Lorentz-gamma overlap) |
| 6 | **II.11.28** (Clausius-Mossotti) | 6 | 2 | Nested rational; tests algebraic structure discovery |

### Tier 2: Good Candidates (worth screening)

| # | Problem | k | n_vars | Why | Risk |
|---|---|---|---|---|---|
| 7 | **R3** (rational sextic) | 11 | 1 | Tests k=11 for rational functions | May be trivially solved (univariate) |
| 8 | **R1** (rational cubic) | 7 | 1 | Clean rational benchmark | May be trivially solved |
| 9 | **III.14.14** (Shockley diode) | 6 | 5 | exp-minus-1 branch; 5 vars | Feature selection possible |
| 10 | **II.11.3** (driven oscillator) | 6 | 5 | Resonance denominator; 5 vars | Near-singularity issues |
| 11 | **Keijzer-11** | 6 | 2 | Bivariate trig; published protocol | Moderate k, may be too easy |
| 12 | **Pagie-2** | 10 | 3 | 3D Pagie; k=10 | No standard protocol |
| 13 | **Liv-14** (poly+trig hybrid) | 8 | 2 | Mixed operators | Less well-known benchmark |

### Tier 3: Lower Priority (high risk of trivial solution or bottleneck mismatch)

| # | Problem | Risk |
|---|---|---|
| I.18.4, I.12.11, I.44.4 (k=5 Feynman) | May be trivially solved — k=5 is at the lower boundary |
| I.13.12, II.11.20 (k=6, 5 vars Feynman) | Feature selection may dominate |
| II.11.17, I.40.1 (k=7, 6 vars Feynman) | Too many variables; GP budget may be insufficient |
| I.9.18 (k=11, 9 vars) | 9 variables; UDFS infeasible; Bingo asymmetric |
| Korns-9 (k=9, 5 vars, 1 irrelevant) | Feature selection confound |
| Liv-9, Liv-22 (k=14-16, univariate poly) | Likely trivially solved despite high k |
| Koza-2, Koza-3 | Duplicates of Nguyen polynomial structure |

---

## Recommended Experimental Plan

### Phase 1: 5-Seed Bingo Screening (8 problems)

Screen Tier 1 and top Tier 2 candidates to determine which are neither trivially
solved nor unsolvable within the 12h Bingo budget:

1. I.29.16 (law of cosines, k=11, 4 vars)
2. I.50.26 (nonlinear oscillation, k=8, 4 vars)
3. Vlad-7 (product + sin, k=9, 2 vars)
4. R2 (rational quintic, k=9, 1 var)
5. I.16.6 (relativistic velocity, k=6, 3 vars)
6. II.11.28 (Clausius-Mossotti, k=6, 2 vars)
7. R3 (rational sextic, k=11, 1 var)
8. III.14.14 (Shockley diode, k=6, 5 vars)

**Accept criterion**: Median baseline R² ∈ [0.3, 0.999] (not trivially solved,
not unsolvable). Problems with median R² > 0.999 are excluded as "none_trivial"
bottleneck.

### Phase 2: Full 30-Seed Paired Comparison

Run accepted problems through the full Bingo baseline vs. IsalSR comparison
(30 seeds, Wilcoxon signed-rank, Holm-Bonferroni correction).

### Expected Outcomes

Based on the bottleneck-type theory:
- **Structural-bottleneck problems** (n_nontrivial=0, moderate k, all vars active)
  should show IsalSR advantage
- **Problems with k > 10** (I.29.16, R3) will directly test whether IsalSR's
  benefit INCREASES with k (contradicting the k≤10 upper bound from 10-problem sample)
- **5-variable problems** (III.14.14, II.11.3) will test whether IsalSR works
  at higher dimensionality than our current winners (max 4 vars)

---

## Final Selection: `models_cherrypicked` Suite (10 problems)

**Date**: 2026-04-20. Implementation: `benchmarks/datasets/cherrypicked.py`.

From the priority rankings above, 10 problems were selected for the
`models_cherrypicked` benchmark suite. Selection balances k-range [6–11],
n_vars [1–5], structural diversity (rational, trig, mixed), and source
diversity (Feynman, Vladislavleva, DSO/Koza, Keijzer, Livermore).

| # | Problem | k | n_vars | Source | Why selected |
|---|---|---|---|---|---|
| 1 | I.29.16 | 11 | 4 | Feynman | cos inside sqrt-of-sum; tests high-k regime |
| 2 | I.50.26 | 8 | 4 | Feynman | Repeated cos subtree; shared-structure discovery |
| 3 | I.16.6 | 6 | 3 | Feynman | Nested rational; unique topology |
| 4 | II.11.28 | 6 | 2 | Feynman | Clausius-Mossotti; nested rational |
| 5 | III.14.14 | 6 | 5 | Feynman | exp-minus-1; tests 5-var structural search |
| 6 | Vlad-7 | 9 | 2 | Vladislavleva 2009 | Mixed product + sin; published protocol |
| 7 | R2 | 9 | 1 | DSO/Koza | Pure rational quintic; clean test case |
| 8 | R3 | 11 | 1 | DSO/Koza | Rational sextic; high-k rational polynomial |
| 9 | Keijzer-11 | 6 | 2 | McDermott 2012 | Bivariate trig; new structure |
| 10 | Liv-14 | 8 | 2 | DSO-Livermore | Polynomial + trig hybrid |

**Excluded from Tier 1**: R1 was replaced by III.14.14 to include a 5-variable
Feynman problem and test higher dimensionality.

**Excluded from Tier 2**: Pagie-2 (no standard protocol), II.11.3 (singularity
risk), Korns-9 (feature selection confound), Keijzer-4 (duplicate of Vlad-2).

All 10 problems use uniform sampling. Only Vlad-7 overrides train/test sizes
(300/1200 per Vladislavleva 2009 protocol). No new operators required beyond
the `bingo_hard.yaml` set.

---

## Cross-Suite Duplicates

Detected duplicates across benchmark suites (same ground-truth expression):

| Expression | Appears as |
|---|---|
| x⁴+x³+x²+x | Nguyen-2, Koza-1 |
| sin(x²)cos(x)−1 | Nguyen-5, Livermore-5 (SRBench) |
| x⁴−x³+x₂²/2−x₂ | Nguyen-12, Keijzer-12 |
| 8/(2+x²+y²) | Keijzer-14, Livermore-14 (SRBench) |
| exp(−x)x³cos(x)sin(x)(cos(x)sin²(x)−1) | Keijzer-4, Vlad-2 |
| x^y | Nguyen-11, Keijzer-3, Keijzer-10 |

---

## Disqualified Operators Across All Suites

| Operator | Problems requiring it |
|---|---|
| tan | Korns-13, Korns-14, Korns-15 |
| arcsin | Feynman I.26.2, I.30.5 |
| tanh | Feynman II.35.21 |

All other scanned problems use only operators in IsalSR's Σ = {+, ×, NEG, INV,
sin, cos, exp, log, √, \|·\|, POW, CONST}.

---

## The π Problem: Why Most Feynman Equations Fail

Of ~100 Feynman equations, ~56 fail the n_nontrivial criterion due to the
presence of π (in 4πε₀, h/(2π), ℏ=h/(2π), etc.). This is a fundamental property
of physics equations: most involve physical constants that are transcendental
numbers. IsalSR's canonical dedup cannot help when the bottleneck is discovering
these precise constants.

However, many of these equations are structurally simple (k < 5) even ignoring
the constant issue. The ~15 candidates that pass are exactly those with complex
operator topology AND exclusively integer-valued physical constants (relativistic
mechanics, thermodynamics, electrostatics).

---

## Files and Scripts

| File | Role |
|---|---|
| `docs/md_files/changes/bottleneck_type_analysis.md` | Screening criterion derivation |
| `experiments/scripts/quantify_advantage_predictor.py` | Exhaustive search for optimal criterion |
| `benchmarks/datasets/srbench.py` | SRBench metadata (29 problem names) |
| `benchmarks/datasets/nguyen.py` | Nguyen suite expressions |
| `benchmarks/datasets/feynman.py` | Feynman medium-tier expressions |
| `benchmarks/datasets/hard.py` | Hard-tier expressions (10 problems) |

### External Sources

| Source | URL / Reference |
|---|---|
| AI Feynman dataset | space.mit.edu/home/tegmark/aifeynman.html |
| DSO benchmarks.csv | github.com/dso-org/deep-symbolic-optimization |
| SRBench ground-truth | github.com/cavalab/srbench |
| McDermott et al. (2012) | "GP needs better benchmarks", GECCO. Appendix A: full problem table |
| Vladislavleva et al. (2009) | IEEE TEC 13(2):333–349 |
| Korns (2011) | "Accuracy in Symbolic Regression", GPTP IX, Springer |
| Keijzer (2003) | EuroGP, LNCS 2610:67–82 |
| Mundhenk et al. (2021) | "Symbolic Regression via Neural-Guided GP", ICLR (Livermore extended) |
