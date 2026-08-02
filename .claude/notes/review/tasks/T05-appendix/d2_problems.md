# T05 Appendix A — the twenty D2 problems

Companion to `T05-benchmark-extension.md`. This is the reference sheet for the
problems the R3.1 extension adds; the reasoning behind *which* problems is in
`docs/md_files/changes/r31_extension_selection.md`, and the machine-readable
version is `docs/md_files/changes/r31_extension_selection_draw.json`.

`D2 = 20`, so `D = D1 ∪ D2 = 70` and CPDT's `N` moves from 50 to 70.

---

## A.1 ODE-Strogatz — 14 problems, taken whole

The 14 datasets of La Cava, Danai & Spector (2016), built from seven two-state
dynamical systems in Strogatz's *Nonlinear Dynamics and Chaos*. Each system
contributes two problems, one per state derivative. Together with SRBench's 116
AI Feynman datasets they are **exactly** SRBench's ground-truth track (130), and
until now the paper covered none of them.

**Selection: none.** All 14, no filter, no cap. That is what licenses the claim
that the non-Feynman half of SRBench's ground-truth track is covered in full
rather than sampled.

**Data.** Published simulation output, not generated at run time: 400 rows per
problem, four trajectories of 100 samples from initial conditions inside a stable
basin. Taken **byte-verbatim from PMLB** (MIT) rather than from the upstream
`lacava/ode-strogatz` repository (GPL-3.0, incompatible with this repo's MIT
licence); the two agree to `7.1 × 10⁻¹⁵` on all 14 datasets, so nothing is lost by
the substitution. sha256 per file: `benchmarks/datasets/data/strogatz/PROVENANCE.md`.

**Split.** 300 train / 100 test — SRBench's 75/25 — with the seed selecting the
permutation. Ranges below are the **empirical** extent of the published samples,
not a sampling domain; there is nothing to sample.

| Problem | Target | `x` range | `y` range |
|---|---|---|---|
| Strogatz-bacres1 | `20 − x − x·y/(1 + 0.5x²)` | [3.49, 16.33] | [9.79, 54.45] |
| Strogatz-bacres2 | `10 − x·y/(1 + 0.5x²)` | [3.49, 16.33] | [9.79, 54.45] |
| Strogatz-barmag1 | `0.5·sin(x − y) − sin(x)` | [4.12, 5.99] | [0.20, 6.65] |
| Strogatz-barmag2 | `0.5·sin(y − x) − sin(y)` | [4.12, 5.99] | [0.20, 6.65] |
| Strogatz-glider1 | `−0.05x² − sin(y)` | [0.16, 5.60] | [−1.32, 28.77] |
| Strogatz-glider2 | `x − cos(y)/x` | [0.16, 5.60] | [−1.32, 28.77] |
| Strogatz-lv1 | `3x − 2xy − x²` | [0.00, 8.00] | [0.00, 3.00] |
| Strogatz-lv2 | `2y − xy − y²` | [0.00, 8.00] | [0.00, 3.00] |
| Strogatz-predprey1 | `x·(4 − x − y/(1 + x))` | [0.01, 6.58] | [2.23, 11.65] |
| Strogatz-predprey2 | `y·(x/(1 + x) − 0.075y)` | [0.01, 6.58] | [2.23, 11.65] |
| Strogatz-shearflow1 | `cot(y)·cos(x)` | [−4.35, 3.60] | [−2.75, 2.15] |
| Strogatz-shearflow2 | `(cos²(y) + 0.1·sin²(y))·sin(x)` | [−4.35, 3.60] | [−2.75, 2.15] |
| Strogatz-vdp1 | `10·(y − ⅓(x³ − x))` | [−1.20, 1.94] | [−0.20, 0.93] |
| Strogatz-vdp2 | `−x/10` | [−1.20, 1.94] | [−0.20, 0.93] |

Every equation was taken from **two independent sources that agree exactly**: the
`outstr` literals in the upstream MATLAB/Simulink generator `simulate_ode.m`, and
each PMLB `metadata.yaml`. They were not transcribed from a paper's prose.

**Criterion (ii).** All 14 pass. `shearflow1` is the only one needing an argument:
Σ_SR has no `cot`, but `cot(y) = cos(y)·sin(y)⁻¹` and `Inv` is one of Σ_SR's twelve
labels, so the target is representable. The module stores it in that form. The
other thirteen use only `+`, `−`, `×`, `÷`, `sin`, `cos` and integer powers.

**Two properties worth stating before a reviewer does.**

*Leakage.* A random 75/25 split of trajectory data puts temporally adjacent,
nearly identical points on both sides. That is SRBench's protocol and we follow
it, but it inflates absolute R² on these 14. It cannot bias the paired contrast —
all three arms see the identical split — so it affects reported levels, not the
comparison. Appendix D.1 should say so once.

*Saturation.* At 400 samples these are small problems, and a 45 s local smoke
already reached R² ≈ 0.98. If they saturate, they contribute `δᵢ ≈ 0` and
**weaken** CPDT rather than strengthen it. §5.4 of the selection rule pre-commits
us to reporting that outcome if it happens.

---

## A.2 AI Feynman remainder — 6 problems, drawn

Drawn by the pre-registered rule: uniformly from the **92** equations of the
AI Feynman database that satisfy criterion (ii) and were not already among the
suite's 24. Seed `2547107438`, derived as `sha256(sorted eligible ids)[:16] mod 2³²`
— a function of the pool, not a chosen parameter. Rule committed at `d95e7d9`,
draw at `0e4a573`, in that order.

| Problem | Target | `n` | Ranges |
|---|---|---|---|
| I.12.2 | `F = q1·q2·r/(4π·ε·r³)` | 4 | [1,5]⁴ |
| II.34.29a | `mom = q·h/(4π·m)` | 3 | [1,5]³ |
| II.34.29b | `E = g·mom·B·Jz/(h/2π)` | 5 | [1,5]⁵ |
| III.19.51 | `E = −m·q⁴/(2(4πε)²(h/2π)²)·(1/n²)` | 5 | [1,5]⁵ |
| III.4.32 | `n = 1/(exp((h/2π)·ω/(kb·T)) − 1)` | 4 | [1,5]⁴ |
| test_4 (bonus) | `v = √(2/m·(E − U − L²/(2m·r²)))` | 5 | m,U,L,r ∈ [1,3]; E ∈ [8,12] |

Sampling is uniform i.i.d., 1000 train / 250 test — identical to every other
Feynman-derived tier in the suite.

`I.12.2` is kept in its **published** form, with `r` in the numerator and `r³` in
the denominator, rather than simplified to `1/r²`. The published form is what the
database distributes and what its sampling ranges were chosen for.

**Domain safety**, asserted in tests rather than assumed: every variable is ≥ 1, so
no denominator vanishes; `III.4.32`'s exponent argument is bounded below by
`1/(50π) ≈ 6.4 × 10⁻³ > 0`, so `exp(·) − 1 > 0`; and `test_4`'s radicand is bounded
below by `8 − 3 − 9/2 = 0.5 > 0` at the worst corner, so the square root is real
everywhere.

---

## A.3 What the extension does not claim

The revised suite covers **44 of SRBench's 130** ground-truth datasets — 30 AI
Feynman equations and all 14 ODE-Strogatz. It does **not** cover the track in
full, and the response letter must not say it does. The claim that survives
scrutiny is narrower and still worth making: *the component that was wholly
absent is now wholly present.*

The 122-dataset black-box track stays out of scope, and the reason is structural
rather than budgetary: those datasets have no ground-truth expression, so
criterion (i) has no expression whose provenance can be checked, criterion (iii)
has no structure whose difficulty can be assessed, and `solution_recovered` — a
reported metric — is undefined. A reviewer may fairly reply that R² and NRMSE
remain computable there; the honest concession is that the exclusion rests on
*some* of our metrics being undefined, not all of them.
