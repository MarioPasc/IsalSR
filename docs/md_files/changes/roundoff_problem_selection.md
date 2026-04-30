# Roundoff Problem Selection: 42 → 50

**Date**: 2026-04-30
**Author**: Mario Pascual Gonzalez
**Status**: Ready for execution

## Motivation

The CPDT (Cross-Problem Dominance Test) pools all problems into a single
paired test. With N=42 problems, the results are already statistically
significant (UDFS R² test p=0.00018, Bingo p=0.0013). Increasing to N=50
provides a rounder number for presentation and marginally increases power.

More importantly, these 8 problems fill gaps in our benchmark portfolio:
- **Under-represented suites**: DSO-Livermore (only Liv-14 previously),
  R-rational (R2/R3 but not R1), Pagie (only Pagie-1).
- **Structural diversity**: log-heavy expressions (Liv-4, Liv-19, I.44.4),
  L2-norm structure (III.10.19), resonance denominator (II.11.3).
- **k-range coverage**: adds k=5 boundary test (I.44.4) and k=10 (Pagie-2).

## Selection Criteria

All 8 problems pass the structural-bottleneck screening criterion:
1. `n_nontrivial_constants = 0` (all constants are small integers)
2. `k >= 5` (sufficient structural complexity)

No new operators required — all use the existing Σ_SR alphabet.

## Selected Problems

| # | Problem | k | n_vars | Source | Expression | Sampling |
|---|---|---|---|---|---|---|
| 1 | III.10.19 | 7 | 4 | AI Feynman | mom·√(Bx²+By²+Bz²) | uniform 1000/250 |
| 2 | II.11.3 | 6 | 5 | AI Feynman | qEf/(m(ω₀²−ω²)) | uniform 1000/250 |
| 3 | I.13.12 | 6 | 5 | AI Feynman | G·m₁·m₂·(1/r₂−1/r₁) | uniform 1000/250 |
| 4 | I.44.4 | 5 | 5 | AI Feynman | n·kB·T·ln(V₂/V₁) | uniform 1000/250 |
| 5 | R1 | 7 | 1 | DSO/Koza | (x+1)³/(x²−x+1) | uniform 1000/250 |
| 6 | Pagie-2 | 10 | 3 | Pagie & Hogeweg | 1/(1+x⁻⁴)+1/(1+y⁻⁴)+1/(1+z⁻⁴) | uniform 1000/250 |
| 7 | Liv-4 | 8 | 1 | DSO-Livermore | ln(x+1)+ln(x²+1)+ln(x) | uniform 1000/250 |
| 8 | Liv-19 | 9 | 1 | DSO-Livermore | ln(x²+x)+ln(x³+x) | uniform 1000/250 |

### Per-Problem Rationale

**III.10.19** (magnetic moment × L2-norm, k=7, 4 vars): L2-norm of a 3D
vector multiplied by a scalar. Introduces sqrt-of-sum-of-squares structure
not present elsewhere in the suite. All variables in [1, 5].

**II.11.3** (driven oscillator, k=6, 5 vars): Resonance denominator
(ω₀²−ω²). Variable ranges ensure ω₀ ∈ [3,5] and ω ∈ [1,2], so the
denominator is always positive (min = 9−4 = 5). Tests rational structure
with 5 variables.

**I.13.12** (gravitational PE difference, k=6, 5 vars): G·m₁·m₂·(1/r₂−1/r₁).
Tests INV (1/x) node usage in a multivariate context. All variables in [1, 5].

**I.44.4** (isothermal work, k=5, 5 vars): n·kB·T·ln(V₂/V₁). The ONLY
Feynman candidate with a log operation. At k=5 (boundary of our screening
criterion), this serves as a boundary test: if IsalSR helps even at k=5,
the claim is strengthened; if not, CPDT counts it as a tie (δ≈0).

**R1** (rational cubic, k=7, 1 var): (x+1)³/(x²−x+1). Completes the
R-rational family (R2 at k=9 and R3 at k=11 are already in cherrypicked).
Denominator (x−1/2)²+3/4 ≥ 3/4 > 0 always. Domain [−1, 1].

**Pagie-2** (3D Pagie, k=10, 3 vars): 3D extension of Pagie-1 (already in
hard tier at k=7). Higher k provides more dedup potential. Uses the
numerically stable form x⁴/(x⁴+1) to avoid overflow at x=0. Domain
[−5, 5]³ matching Pagie-1.

**Liv-4** (three-log sum, k=8, 1 var): ln(x+1)+ln(x²+1)+ln(x). Three
independent log terms with polynomial arguments. Domain [0.1, 10].
Under-represents DSO-Livermore suite in our portfolio.

**Liv-19** (log-of-polynomial, k=9, 1 var): ln(x²+x)+ln(x³+x). Nested
log-of-polynomial structure. Domain [0.1, 10] (x > 0 required for both
arguments). Tests whether GP can discover log-of-factored-polynomial.

## Domain Safety Verification

| Problem | Potential singularity | Mitigation |
|---|---|---|
| II.11.3 | ω₀² = ω² (resonance) | ω₀ ∈ [3,5], ω ∈ [1,2] → min gap = 5 |
| I.13.12 | r₁ = 0 or r₂ = 0 | All vars ∈ [1, 5] |
| I.44.4 | V₁ = 0 (ln singularity) | All vars ∈ [1, 5] |
| R1 | x² − x + 1 = 0 | Min = 3/4 > 0 ∀x |
| Pagie-2 | x = 0 (x⁻⁴ overflow) | Uses x⁴/(x⁴+1) form |
| Liv-4 | x = 0 (ln(x)) | x ∈ [0.1, 10] |
| Liv-19 | x = 0 (ln(x²+x)) | x ∈ [0.1, 10] |

## Execution

Same infrastructure as hard/cherrypicked: 30 seeds, 2 methods (UDFS + Bingo),
2 variants (baseline + isalsr), max_time=43200 (12h).

**Total SLURM tasks**: 8 problems × 30 seeds × 4 groups = 960 tasks.

### Picasso Commands

```bash
# Preview (dry run)
bash slurm/roundoff_launch.sh --dry-run

# Submit all 4 groups (phased: UDFS → Bingo → Analysis)
bash slurm/roundoff_launch.sh

# Submit single group
bash slurm/roundoff_launch.sh --experiment udfs_roundoff_baseline

# Analysis only (after all experiments complete)
bash slurm/roundoff_launch.sh --analyze-only
```

### Post-Execution

After completion, merge results into the unified directory:
```bash
python experiments/scripts/merge_results.py \
    --source /mnt/home/users/tic_163_uma/mpascual/execs/isalsr/models_roundoff \
    --target /path/to/wl_subtree_unified \
    --flatten benchmark
```

Then re-run the analysis pipeline with N=50:
```bash
python -m experiments.models.analyze \
    --results-dir /path/to/wl_subtree_unified \
    --methods udfs,bingo \
    --benchmarks benchmark
```

## Files

| File | Role |
|---|---|
| `benchmarks/datasets/roundoff.py` | 8 problem definitions |
| `experiments/configs/udfs_roundoff.yaml` | UDFS config |
| `experiments/configs/bingo_roundoff.yaml` | Bingo config |
| `slurm/roundoff_config.yaml` | SLURM resource config (Picasso paths) |
| `slurm/roundoff_launch.sh` | Phased SLURM launcher |
| `tests/unit/test_roundoff_benchmarks.py` | 48 unit tests |
| `experiments/models/orchestrator.py` | Updated: roundoff registered |
| `slurm/workers/models_experiment_slurm.sh` | Updated: roundoff dispatch |

## References

- Udrescu & Tegmark (2020). AI Feynman. Science Advances 6(16).
- Petersen et al. (2021). DSO. NeurIPS. (R1 attribution)
- Pagie & Hogeweg (1997). Evolutionary consequences of coevolving targets.
  Evolutionary Computation 5(4):401–418. (Pagie-2)
- Mundhenk et al. (2021). Symbolic Regression via Neural-Guided GP. ICLR.
  (Liv-4, Liv-19)
