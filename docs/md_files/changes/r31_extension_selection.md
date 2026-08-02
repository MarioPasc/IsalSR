# Pre-registered selection rule for the R3.1 benchmark extension (D2)

**Status:** pre-registration. Written and committed **before** the rule was executed
and before any D2 run existed.
**Date:** 2026-08-02
**Ticket:** T05 (`.claude/notes/review/tasks/T05-benchmark-extension.md`), AC-3
**Campaign:** C2, problem set `D2` (`EXECUTION-PLAN.md` §1)
**Executed by:** `experiments/scripts/r31_draw_extension.py`

---

## 1. Why this document exists

The paper's primary significance metric is the Cross-Problem Dominance Test (CPDT),
which treats each problem as one paired observation. When every per-problem
difference `δᵢ` is non-negative, the CPDT p-value **decreases monotonically with `N`**.
Raising `N` from 50 to about 70 therefore strengthens the headline statistic *by
construction*, independently of whether the added problems say anything new.

That is a real benefit and the paper will claim it. It is only defensible if the
problem list was fixed in advance and chosen without reference to expected outcome.
This document fixes it. The mechanism is git history, not assertion: the rule and the
script that executes it are committed with **no output**, and the drawn list lands in
a separate, later commit, so a third party can verify the ordering (§6).

The hazard is not hypothetical for this project. The suite already contains a tier
named `cherrypicked`, selected *deliberately* for predicted IsalSR advantage using the
bottleneck-type analysis, and disclosed as such
(`docs/md_files/changes/candidate_problem_screening.md`). A reviewer who reads that
document will look at this extension for the same behaviour. It must not be there.

---

## 2. Scope

`D2 = STROGATZ ∪ FEYNMAN_REMAINDER`, target size ≈ 20 problems.

Cost, which is what fixes the size: each added problem costs
`3 arms × 2 methods × 20 seeds = 120 runs` at a 12 h budget = **1,440 core-hours**.
D2's committed share of C2 is ≈28,800 core-hours (`EXECUTION-PLAN.md` §1, §8.2),
which is 20 problems. The cap is arithmetic, not judgement.

---

## 3. The Strogatz half — no selection at all

```
STROGATZ := every ODE-Strogatz dataset in PMLB
```

That is **all 14**, with no filter, no cap and no tie-break: `bacres1`, `bacres2`,
`barmag1`, `barmag2`, `glider1`, `glider2`, `lv1`, `lv2`, `predprey1`, `predprey2`,
`shearflow1`, `shearflow2`, `vdp1`, `vdp2`.

Taking the whole set is what licenses the claim the response letter wants to make:
SRBench's ground-truth track is exactly `116 feynman_* + 14 strogatz_*` = 130
datasets (counted from `cavalab/srbench/docs/csv/groundtruth.csv`), so covering all
14 Strogatz problems means the paper covers the non-Feynman half of that track in
full rather than sampling it. A subset would forfeit the claim and reintroduce a
selection question for no saving.

All 14 satisfy the paper's four inclusion criteria. Criterion (ii) is the only one
needing an argument: `shearflow1` is published as `cot(y)·cos(x)`, and Σ_SR has no
`cot` — but `cot(y) = cos(y)·sin(y)⁻¹` and `Inv` is one of Σ_SR's twelve labels, so
the target is representable. The other thirteen use only `+`, `−`, `×`, `÷`, `sin`,
`cos` and integer powers.

Data provenance, licence and the equations themselves:
`benchmarks/datasets/data/strogatz/PROVENANCE.md`.

---

## 4. The Feynman half — the rule

### 4.1 Eligible pool

```
E := { e ∈ AIFEYNMAN_120 : representable_syntactic(e) }  \  IN_SUITE_IDS
```

- `AIFEYNMAN_120` — the complete AI Feynman database, 100 base + 20 bonus equations,
  as `benchmarks/datasets/feynman_catalogue.py`.
- `representable_syntactic(e)` — **criterion (ii) under the paper's own reading**:
  every function symbol written in the published formula is in Σ_SR. Determined by
  walking the sympy parse tree, not by string matching.
- `IN_SUITE_IDS` — the 24 AI Feynman ids already used by the 50-problem suite,
  matched **by id**, so that the pool cannot depend on the separate question of
  whether each of those 24 is faithfully transcribed (five are not; filed against
  T09, §5.3 below).

Criterion (ii) is applied in its **syntactic** form, which is the stricter of the two
defensible readings (§5.2). Using the more permissive semantic reading would *enlarge*
the pool, so this choice cannot be an attempt to admit favourable problems.

### 4.2 The draw

```
K   := 6
ids := sorted(E)                                    # lexicographic, deterministic
seed:= int(sha256("|".join(ids)).hexdigest()[:16], 16) mod 2**32
FEYNMAN_REMAINDER := sorted( numpy.random.default_rng(seed).permutation(ids)[:K] )
```

The seed is **derived from the eligible pool itself**. It is not a free parameter:
there is no knob to vary and therefore nothing to fish over. Given the four inclusion
criteria and the existing suite, `E` is determined, and given `E` the seed and the
draw are determined. Any third party re-running
`experiments/scripts/r31_draw_extension.py` on this commit must obtain the same six
ids or the pre-registration has failed.

`K = 6` because `14 + 6 = 20`, the budgeted size (§2).

### 4.3 Explicit non-goals

These are the things the rule deliberately does **not** do, and why:

| Not done | Why not |
|---|---|
| Stratify by arity, by node count `k`, or by operator class | Every candidate stratifying variable is correlated with the structural-bottleneck axis along which IsalSR's advantage was previously characterised (`docs/md_files/changes/bottleneck_type_analysis.md`: advantage iff `n_nontrivial_constants = 0` and `k ≥ 5`). Stratifying on it *is* selection, however neutrally it is described. |
| Screen for predicted IsalSR advantage | The opposite of what this extension is for. That is what the `cherrypicked` tier is, and it is disclosed as such. |
| Order lexicographically instead of drawing | AI Feynman numbering tracks the lecture volume — Volume I mechanics, Volume III quantum — so a lexicographic prefix is biased toward polynomial targets. |
| Exclude problems expected to be easy, saturated, or unsolvable | Outcome reasoning. Saturated problems contribute `δᵢ = 0` to CPDT, which weakens rather than strengthens the headline; excluding them would be selection in our own favour. |
| Re-draw if the sample looks unbalanced | The draw is executed once. "Unbalanced" is only definable against an outcome expectation. |

---

## 5. What the extension does *not* license us to say

### 5.1 The seed reduction travels with the `N` increase

C2 runs at **20 seeds**, not the 30 the submitted campaign used
(`EXECUTION-PLAN.md` §0.4a, §6.3). Reviewer R1 explicitly endorsed *"50 problems, 30
seeds, Demsar-style paired inference"*, so the reduction is visible and will be
noticed. The response letter must present both changes **in the same paragraph**, as
one deliberate trade: more problems, fewer seeds per problem, primary metric
strengthened, supplementary per-problem tests weakened, both reported. Announcing the
`N` increase in one place and the seed reduction in another would be the kind of
presentation this document exists to prevent.

The arithmetic, stated rather than gestured at: CPDT pools over problems, so the
per-problem mean `δᵢ` gains standard error by a factor `√(30/20) = 1.22` while `N`
rises by a factor 1.4. Per-problem Wilcoxon tests lose power by the same 1.22 in
non-centrality, and their minimum attainable two-sided p rises from `2⁻²⁹ ≈ 1.9×10⁻⁹`
to `2⁻¹⁹ ≈ 1.9×10⁻⁶`.

### 5.2 Criterion (ii) is a small filter, and we say so

Measured over all 120 equations, criterion (ii) excludes **4** under the syntactic
reading and **3** under the semantic one — not "a substantial fraction". The four:

| Equation | Formula | Blocking operator |
|---|---|---|
| I.26.2 | `theta1 = arcsin(n*sin(theta2))` | `arcsin` |
| I.30.5 | `theta = arcsin(lambd/(n*d))` | `arcsin` |
| bonus (PMLB `feynman_test_10`) | `theta1 = arccos((cos(theta2)-v/c)/(1-v/c*cos(theta2)))` | `arccos` |
| II.35.21 | `M = n_rho*mom*tanh(mom*B/(kb*T))` | `tanh` (syntactic only) |

`tanh` is the one that separates the readings: `tanh(u) = (eᵘ − e⁻ᵘ)(eᵘ + e⁻ᵘ)⁻¹`
uses only `Exp`, `Neg`, `Add`, `Mul` and `Inv`, all of which are in Σ_SR, so `tanh`
is semantically representable and the paper's own example list for criterion (ii)
(`tanh`, `arctan`, `sgn`) is meant syntactically. The same argument makes `tan` and
`cot` representable, which is what admits Strogatz `shearflow1` (§3). Only the
inverse trigonometric functions are excluded under both readings.

**Consequence for the response to R3.1: the choice of 24 of 120 cannot be attributed
to criterion (ii).** A reviewer can falsify that claim from PMLB in ten minutes. The
binding constraints are criterion (iv) — complementary coverage, which caps redundant
physics-formula coverage — and compute: all 120 AI Feynman equations plus the 250+
SRBench problems would be ≈45,840 runs, ≈7.6× the submitted campaign.

### 5.3 Five of the existing 24 are mislabelled

Established while building the catalogue, filed against T09, and **not** acted on
here because changing a target function would break C1↔C2 continuity on that problem:

| Suite id | Target as implemented | AI Feynman database | Nature |
|---|---|---|---|
| I.39.10 | `0.5·p_r·V` | `I.39.1`: `3/2·pr·V` | wrong coefficient and wrong id |
| I.12.4 | `q1/(4π·r·c)` | `q1/(4π·ε·r²)` | different function (`1/r` vs `1/r²`) |
| II.3.24 | `p·r/(4π)` | `Pwr/(4π·r²)` | different function (`r` vs `1/r²`) |
| II.11.27 | `n0·e^(−μB/kT) + n0·e^(μB/kT)` | Clausius–Mossotti polarisation | different equation |
| III.17.37 | `f0/√((ω−ω0)² + γ²/4)` | `β(1 + α·cos θ)` | different equation |

They remain valid regression problems and every arm regresses the same target on the
same data, so no reported comparison is affected. What is affected is Appendix D.1,
which documents each problem by expression and citation. This is T09's to resolve.

### 5.4 Reporting obligations, fixed in advance

Whatever the extension does to the headline:

1. CPDT is reported at **both** `N = 50` and `N ≈ 70`, per method, per metric, at 20
   seeds throughout.
2. A **per-tier** breakdown is reported, so a reader can see whether the extension
   moved the result or merely diluted it.
3. If the extension **weakens** the result, that is what gets written. It is the
   honest outcome and it is far cheaper than being caught. This clause is
   pre-registered precisely so that it is not a decision taken after seeing the
   numbers.

---

## 6. Commit protocol (this is what AC-3 checks)

| Commit | Contains | Must **not** contain |
|---|---|---|
| **1** | this document, `experiments/scripts/r31_draw_extension.py`, `benchmarks/datasets/feynman_catalogue.py`, `benchmarks/datasets/strogatz.py` and their tests | the drawn ids, in any file |
| **2** | `docs/md_files/changes/r31_extension_selection_draw.json` — the script's output, and only that | — |

Both hashes are recorded in T05 §8. Nothing is submitted to Picasso between them.

---

## 7. Escape hatch

If D2 cannot be made ready, the trade is **scope, not schedule**
(`EXECUTION-PLAN.md` §8.3 item 2): drop to **Strogatz only, 14 problems**,
`N = 64`. That preserves the "SRBench ground-truth track is now covered" claim and
costs the Feynman-remainder half of the R3.1 answer. Delaying the launch is not on
the list; every day of slip is ≈7,200 core-hours of headroom the campaign does not
get back.

Taking the hatch is recorded in T05 §8 if it happens.
