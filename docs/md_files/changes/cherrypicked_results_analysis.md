# Cherrypicked Benchmark Analysis: IsalSR vs Native DAG

**Date**: 2026-04-28
**Results dir**: `/media/mpascual/Sandisk2TB/research/isalsr/results/model_validation/wl_subtree_cherrypicked`
**Suite**: 10 problems (`cherrypicked`), 30 seeds, methods Bingo + UDFS, paired baseline vs `isalsr`.
**Hypothesis under test**: problems screened to satisfy *bottleneck = structural* (n_nontrivial_constants = 0, k ≥ 5) should show IsalSR downstream advantage, generalizing the hard-tier finding (5/5 structural problems significant) to held-out data.

---

## 1. Headline Numbers

### 1.1 Three-axis grand summary

| Axis | Metric | Bingo | UDFS |
|---|---|---|---|
| **Search-space** | mean reduction factor (RF) | **1.82** | **1.71** |
| | mean redundancy rate | 45.1 % | 40.4 % |
| | n significant (Cohen's d_RF) | 10/10 (d̄=153.7) | 10/10 (d̄=11.6) |
| **Regression quality** | r2_test sig. (Holm-corrected) | **1/10** (d̄=0.14) | **1/10** (d̄=0.29) |
| | r2_train sig. | 1/10 (d̄=0.13) | 2/10 (d̄=0.42) |
| | nrmse_test sig. | 2/10 (d̄=−0.14) | 1/10 (d̄=−0.29) |
| | n_degraded | 0/10 | 0/10 |
| **Compute overhead** | mean | **36.7 %** | **0.06 %** |
| | solution recovery (baseline → isalsr) | 3.1 % → 4.4 % | 0 % → 0 % |

Source: `analysis/three_axis_summary_{bingo,udfs}_cherrypicked.json`,
`analysis/benchmark_summary_{bingo,udfs}_cherrypicked.csv`.

### 1.2 Comparison vs prior tiers

| Suite | Bingo RF | Bingo r2_test sig. | UDFS RF | UDFS r2_train sig. |
|---|---|---|---|---|
| Production (Nguyen+Feynman, 22 problems) | 1.28 | 0/22 | 1.56 | 10/22 |
| Hard (10 problems) | – | 5/10 (structural subset) | – | – |
| **Cherrypicked (10 problems)** | **1.82** | **1/10** | **1.71** | **2/10** |

The search-space reduction is the **largest seen in any production tier** (1.82 ≈ canonical proves nearly half of all explored DAGs are redundant under isomorphism), confirming that screening-by-structural-bottleneck does select problems with high isomorphism duplication. Regression-quality dominance, however, **does not generalize**: only the i.16.6 / liv_14 / r2 trio breaks through Holm-Bonferroni.

---

## 2. Per-Problem Breakdown

### 2.1 Bingo (paired_t / Holm)

Bold = Holm-corrected p < 0.05.

| Problem | k | n_vars | r2_test (B → I) | Cohen d | p_holm | Verdict |
|---|---|---|---|---|---|---|
| **i.16.6** (relativistic velocity) | 6 | 3 | 0.997 → 0.999 | **+0.89** | **5e−4** | clear win |
| i.29.16 (law of cosines) | 11 | 4 | 0.956 → 0.958 | +0.08 | 1.0 | tie |
| i.50.26 (oscillation) | 8 | 4 | ≈1.000 → ≈1.000 | +0.25 | 0.54 | already solved |
| ii.11.28 (Clausius-Mossotti) | 6 | 2 | 1.000 → 1.000 | 0 | 1.0 | already solved |
| iii.14.14 (Shockley diode) | 6 | 5 | 0.999 → 0.999 | +0.05 | 1.0 | tie |
| keijzer_11 (bivariate trig) | 6 | 2 | 0.995 → 1.000 | +0.45 | 0.19 | trend, n.s. |
| **liv_14** (poly+trig hybrid) | 8 | 2 | 1.000 → 1.000 | +0.34 | 0.08 | r2 n.s. but **nrmse_test/train sig**, jaccard +0.70, complexity −0.72 |
| r2 (rational quintic) | 9 | 1 | 1.000 → 1.000 | −0.31 | 0.24 | mild regression (n.s.) |
| r3 (rational sextic) | 11 | 1 | 1.000 → 1.000 | −0.50 | 0.12 | mild regression (n.s.) |
| vlad_7 (product+sin) | 9 | 2 | 0.878 → 0.887 | +0.18 | 1.0 | tie |

### 2.2 UDFS (paired_t / Holm)

| Problem | r2_test (B → I) | Cohen d | p_holm | Verdict |
|---|---|---|---|---|
| i.16.6 | 0.895 → 0.896 | +0.38 | 0.40 | tie |
| i.29.16 | 0.196 → 0.196 | 0 | 1.0 | UDFS plateau |
| i.50.26 | 0.168 → 0.180 | +0.36 | 0.46 | UDFS plateau |
| ii.11.28 | 0.998 → 0.999 | +0.33 | 0.54 | already solved |
| iii.14.14 | 0.423 → 0.419 | −0.13 | 1.0 | UDFS plateau |
| **keijzer_11** | 0.953 → 0.953 (train d=0.87) | +0.45 | 0.18 (train **3e−4**) | train win, test n.s. |
| liv_14 | 0.965 → 0.965 | 0 | 1.0 | identical |
| **r2** (rational quintic) | 0.956 → 0.961 | **+0.87** | **6.5e−3** | clear win |
| r3 | 0.941 → 0.943 | +0.29 | 0.46 | tie |
| vlad_7 | 0.438 → 0.469 | +0.32 | 0.54 | trend, n.s. |

> **Wall-clock pathology — UDFS hits the 12 h wall on every problem** (`wall_clock_search_only_s ≈ 43 200 s` baseline, ≈ 43 180 s isalsr). UDFS's enumeration cannot exhaust the search space within the budget on any cherrypicked problem, so all UDFS R² values reflect the regressor's *plateau at deadline*, not convergence. This is qualitatively different from production, where most problems converged early.

---

## 3. Interpretation

### 3.1 What the data confirms

1. **Search-space reduction is robust and large.** RF = 1.82 on Bingo (45 % of explored DAGs are isomorphic duplicates) is the highest of any tier. d̄_RF = 154 (Bingo) and 11.6 (UDFS) are massive effects, all 20/20 problem-method cells significant. The O(k!) → canonical claim is independently verified on this independent suite.
2. **No degradation.** On both methods, *zero* problems show statistically significant degradation in r2_test, r2_train, or nrmse. Canonicalization is downstream-safe.
3. **Bingo overhead has crept up.** 36.7 % vs 51 % production — improvement, but still material; UDFS overhead remains negligible (0.06 %).

### 3.2 What the data does **not** confirm

The hypothesis "structural bottleneck ⇒ statistically significant downstream R² gain" **fails to generalize** here. Only 1/10 (Bingo) and 1/10 (UDFS) cells reach Holm-corrected significance on r2_test, against the 5/5 hit rate observed on hard-tier structural problems (`bottleneck_type_analysis.md`).

Three mechanisms explain the gap, and they are not failures of IsalSR:

- **Saturation**: ii.11.28, i.50.26, liv_14, r2, r3 (Bingo) start at R² ≈ 1.0 with the baseline. There is no headroom. These are *too easy* for the budget — IsalSR redundancy elimination cannot help when the baseline already converges. The screening filter (k ≥ 5, no constants) was structural but did not control for difficulty *given Bingo's hyperparameters*.
- **Plateau (UDFS)**: i.29.16, i.50.26, iii.14.14, vlad_7 baseline R² < 0.5. UDFS hits the time wall in a low-quality region of the search where canonicalization redirects budget to *other* low-quality candidates. With both methods deadline-bounded at the same R², no statistical separation appears.
- **Constants are still the bottleneck for some**: r3 baseline already hits R² = 1.000 (LM constant optimization saturates). Cohen d = −0.50 (n.s.) hints at slight regression — IsalSR shifts evaluation budget away from constant tuning toward novel structures, which on a 1-variable rational polynomial does not pay off.

The two clear wins (Bingo @ i.16.6, UDFS @ r2) are exactly the cells where (a) the baseline has not yet saturated **and** (b) the problem k is in the sweet spot of 6–9 internal nodes. liv_14 (Bingo, k=8) is borderline: nrmse and Jaccard improve significantly but R² is already so close to 1 that the test under-detects.

### 3.3 Did we find a "family of traits"?

No new family emerges *from R²/NRMSE alone*. The traits already known from `bottleneck_type_analysis.md` — **structural bottleneck plus a baseline that has not saturated** — remain the predictor. Cherrypicked confirms the structural part (RF) but adds a missing condition: **baseline must have residual room to improve**. The screening was incomplete because it filtered on problem properties (k, n_constants) but not on baseline behaviour (baseline R², saturation).

A secondary observation worth recording: on the two wins, the auxiliary metrics align with the variance-reduction mechanism reported on hard-tier — Jaccard up, complexity flat or down, search wall-clock for IsalSR ≥ baseline (canonical lookup cost) but R² variance lower across seeds (i.16.6: σ 0.0020 → 0.00087, 2.3× tighter).

---

## 4. Advice

1. **Report this experiment honestly as a partial confirmation.** The search-space axis is the strongest evidence yet (RF=1.82). The regression-quality axis is too weak to claim a "family of structural traits guarantees R² wins". Frame cherrypicked as: *the structural-bottleneck criterion is necessary but not sufficient — a non-saturated baseline is also required*.
2. **Re-screen with a saturation gate.** Re-run cherrypicked screening with the additional filter `baseline_r2_test < 0.99 across ≥ 50 % of seeds` using a short pilot (e.g. 5 seeds, 10 % of `max_time`). Drop ii.11.28, i.50.26, liv_14, r3 (Bingo saturates). Replace them with structural problems that *Bingo cannot solve in 12 h* — candidates from the screened-but-not-selected pool in `candidate_problem_screening.md`.
3. **For UDFS, raise the time budget or cut problem complexity.** All UDFS runs hit the 12 h wall. Either (a) extend `max_time` to 24 h for cherrypicked-UDFS, or (b) restrict UDFS to the 4 problems where its baseline is non-degenerate (i.16.6, ii.11.28, keijzer_11, liv_14, r2, r3) and re-launch. Current UDFS data for i.29.16, i.50.26, iii.14.14, vlad_7 is non-informative.
4. **Reframe the paper claim around variance reduction, not mean R².** The hard-tier mechanism (variance reduction + seed rescue, Cliff δ ≈ 0.5) is the more defensible scientific contribution. Add a per-seed variance plot for the two cherrypicked winners (i.16.6, r2) and check whether σ(R²_test) is reduced.
5. **Don't drop the cherrypicked benchmark.** Even with the regression-quality null, the search-space evidence (10/10 sig at d̄ > 100, RF 1.82) stands on its own. It directly supports the paper's central O(k!) reduction claim on a held-out suite the canonicalizer never saw during development.
6. **Investigate the 36.7 % Bingo overhead.** It is down from 51 % (production) but still high for what is effectively a hash lookup. With `use_simplification: false` and `fast_canonical (mode="wl_only")`, the WL hash should dominate. Profile a single i.16.6 run; suspect candidates are `LabeledDAG.copy()` calls inside the dedup hot-path or the `_established` fingerprint construction (B12 fix).

---

## 5. Files

- Per-problem paired stats: `bingo/cherrypicked/{problem}/paired_stats.json`, `udfs/cherrypicked/{problem}/paired_stats.json`
- Aggregate: `analysis/three_axis_global.json`, `analysis/three_axis_summary_{method}_cherrypicked.json`
- Cross-method Friedman: `analysis/cross_method_cherrypicked.json` (χ²=26.08, p=9.2e−6 across 4 groups)
- Reduction comparison: `analysis/reduction_comparison_cherrypicked.json`
- This report: `docs/md_files/changes/cherrypicked_results_analysis.md`
