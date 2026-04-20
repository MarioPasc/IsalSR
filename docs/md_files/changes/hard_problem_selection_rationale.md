# Hard-Problem Selection Rationale

**Date**: 2026-04-13 (original selection), 2026-04-20 (this memory document)
**Primary source**: `docs/md_files/changes/hard_benchmark_proposal.md`

---

## Why We Needed Hard Problems

Bingo solves all 22 existing problems (12 Nguyen + 10 Feynman) to R² ≈ 1.0
across 30 seeds, preventing demonstration of downstream quality / convergence-speed
gains from IsalSR's search-space reduction. A harder tier was needed to create
measurable separation between baseline and IsalSR.

---

## Selection Criteria (all four required)

1. **Published provenance** — expression and sampling protocol from a peer-reviewed
   source used by ≥2 independent SR studies. No ad-hoc expressions.
2. **Operator compatibility** — representable within IsalSR's Σ = {+, ×, NEG, INV,
   sin, cos, exp, log, √, |·|, POW, CONST}. Problems requiring tanh, arctan,
   signum, floor are disqualified.
3. **Expected difficulty for GP** — published evidence or structural analysis
   indicating GP-based methods (stack ≤ 32, LM, 12h budget) will NOT saturate
   at R² = 1.0. Based on difficulty characterisations from Žegklitz & Pošík (2020),
   McDermott et al. (2012), and SRBench ground-truth track recovery rates.
4. **Complementary coverage** — must add a difficulty axis not in the existing 22:
   m > 3 vars, implicit feature selection, nested denominators, or high-frequency trig.

---

## Literature Sources

| Source | Role |
|---|---|
| McDermott et al. (2012), GECCO — "GP needs better benchmarks" | Community-standard problem lists |
| White et al. (2013), GPEM — "Better GP benchmarks" | Companion study with recommended protocols |
| La Cava et al. (2021), NeurIPS — SRBench | Ground-truth track; referenced for II.11.27 ("sub-100% recovery") and Keijzer-6 |
| Žegklitz & Pošík (2020), GPEM — "Benchmarking state-of-the-art SR algorithms" | Empirical difficulty characterisation for GP-hard classics |
| Vladislavleva et al. (2009), IEEE TEC | Source for Vlad-2 and Vlad-4 |
| Korns (2011), GPTP IX | Source for Korns-12 (feature selection) |
| Pagie & Hogeweg (1997), Evol. Comp. | Source for Pagie-1 (universally GP-hard) |
| Udrescu & Tegmark (2020), Science Advances | Feynman suite extended to harder instances |
| Keijzer (2003), EuroGP LNCS 2610 | Source for Keijzer-6 (harmonic number) |

---

## SRBench Role in Selection

SRBench (La Cava et al., 2021) was **NOT** used as a quantitative R² threshold
table for selecting problems. It was referenced in two specific ways:

1. **As evidence of difficulty**: II.11.27 — "SRBench ground-truth track reports
   sub-100% recovery for GP methods." Keijzer-6 — "appears in multiple SRBench
   configurations."
2. **As justification for NOT using the full suite** (Section 5 of the proposal):
   the full SRBench (252 problems) was rejected due to computational cost
   (362,880 CPU-hours), narrative dilution, and the fact that PMLB's 132
   real-world datasets have unknown ground truth / measurement noise that
   confounds the structural deduplication signal.

---

## Per-Problem Rationale

### Extended Feynman (5 problems, uniform 1000 train / 250 test)

| Problem | Expression | Why selected |
|---|---|---|
| I.15.10 | m₀v/√(1−v²/c²) | 3 vars; nested sqrt+div+product; extends I.10.7/I.48.20 with numerator complexity |
| I.30.3 | sin²(nθ/2)/sin²(θ/2) | Squared-sine ratio; high-frequency oscillations when n large |
| I.37.4 | I₁+I₂+2√(I₁I₂)cos(δ) | 3 vars; sqrt-of-product×cos; √(product) structurally rare in GP |
| II.11.27 | n₀e^(−μB/kT)+n₀e^(μB/kT) | 4 vars; dual exp branches; SRBench sub-100% recovery |
| III.17.37 | f₀/√((ω−ω₀)²+γ²/4) | 4 vars; nested sqrt of sum-of-squares; constant discovery |

### GP-Hard Classics (5 problems, per-problem sampling)

| Problem | Expression | Why selected |
|---|---|---|
| Pagie-1 | 1/(1+x⁻⁴)+1/(1+y⁻⁴) | Universally GP-hard (GPBenchmarks.org: "none solved this"); x⁻⁴ singularity |
| Korns-12 | 2−2.1cos(9.8x₁)sin(1.3x₅) | Feature selection (3/5 vars irrelevant); high-freq trig (156 cycles) |
| Vlad-4 | 10/(5+Σ(xᵢ−3)²) | 5 vars all relevant; rational with wide sum-of-squares denominator |
| Vlad-2 | e⁻ˣx³(cos·sin)(cos·sin²−1) | Univariate but k=13 heterogeneous; deep multiplicative composition |
| Keijzer-6 | H(x)≈log(x)+γ | Harmonic number; irrational constant; extrapolation; SRBench standard |

---

## Coverage Matrix

| Difficulty axis | Problems covering it |
|---|---|
| m ≥ 4 input variables | II.11.27, III.17.37, Korns-12, Vlad-4 |
| Nested sqrt | I.15.10, I.37.4, III.17.37 |
| Nested trig | I.30.3, I.37.4 |
| Dual exponential branches | II.11.27 |
| Irrational constant discovery | Keijzer-6 (γ), Korns-12 (9.8, 1.3) |
| Implicit feature selection | Korns-12 |
| Negative-exponent rational | Pagie-1 |
| Wide denominator | Vlad-4 |
| Deep heterogeneous product | Vlad-2 |

---

## Post-Hoc Findings (2026-04-19)

After running the full experiments, the bottleneck-type analysis
(`docs/md_files/changes/bottleneck_type_analysis.md`) found that IsalSR's advantage
correlates perfectly with **structural bottleneck** problems — those where the
difficulty is finding the right operator topology, not optimizing constants or
selecting features. This was not known at selection time; the 10 problems were
chosen for complementary difficulty coverage, not to test the bottleneck hypothesis.

---

## Files

| File | Role |
|---|---|
| `docs/md_files/changes/hard_benchmark_proposal.md` | Original proposal (211 lines, full detail) |
| `benchmarks/datasets/hard.py` | Implementation of 10 benchmarks |
| `experiments/configs/bingo_hard.yaml` | Bingo experiment config |
| `experiments/configs/udfs_hard.yaml` | UDFS experiment config |
| `docs/md_files/changes/bottleneck_type_analysis.md` | Post-hoc bottleneck analysis |
| `benchmarks/datasets/srbench.py` | SRBench metadata loader |
