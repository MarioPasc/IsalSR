# C4 — what 203 distinct fingerprints instead of 210 actually means

**Date**: 2026-08-04
**Found by**: Stage C certification, job 1758604 (`c2_certify.py`, verdict NO-GO)
**Status**: one blocking finding needs **Mario's decision**; one is expected and needs **disclosure**

---

## 1. The headline

C4 asserted "210 distinct fingerprints (70 problems × 3 seeds), each appearing
exactly 6 times". The Stage C root produced **203**. The shortfall decomposes
exactly, with nothing left over:

```
210  expected
 −4  Pagie-1 and Keijzer-6: 3 seeds collapse to 1 fingerprint each  (2 problems × 2 lost)
 −3  I.12.1 and I.34.27 are the same problem: 3 seed-pairs merge    (1 pair × 3 seeds)
────
203  observed                                                        ✓
```

**The property C4 exists to protect passes on 1,260/1,260 cells.**
`cross_arm_disagreement = 0`: for every `(problem, seed)`, all three arms and
both methods saw byte-identical data. The paired design is **not** void. What
failed is the auxiliary "all mutually distinct" assertion, which conflated two
unrelated situations.

---

## 2. Finding A (blocking) — `I.12.1` and `I.34.27` are the same problem

Both live in the **D1 `feynman` suite**, i.e. this has been true since C1.

| | `I.12.1` | `I.34.27` |
|---|---|---|
| Physical law | `F = mu * N_s` (friction) | `E = hbar * omega` (photon energy) |
| `expression` | `mu * N_s` | `hbar * omega` |
| **`sympy_expression`** | **`x_0*x_1`** | **`x_0*x_1`** |
| `num_variables` | 2 | 2 |
| `var_ranges` | `[(1.0, 5.0), (1.0, 5.0)]` | `[(1.0, 5.0), (1.0, 5.0)]` |

Same function, same domain, same sampling. At any given seed the generated
`X_train`, `y_train`, `X_test`, `y_test` are **byte-identical** — verified
directly, not inferred from the hash.

### Why it happened

The AI Feynman catalogue (`feynman_catalogue.py`) defines I.34.27 as
`(h/(2*pi))*omega` with `h` and `omega` both free on `[1,5]`. Our
implementation in `benchmarks/datasets/feynman.py` folded the `1/(2π)` into the
symbol name (`hbar * omega`) and **dropped the constant**. With the constant
present the data would differ; without it, the two problems coincide.

### Why it matters

CPDT treats **each problem as one paired observation** and runs a sign or
Wilcoxon test over `N` problems. Two of the `N` are the same problem, with
identical data, so their `δᵢ` are not independent draws. The reported `N` is
overstated by one — `N = 42` in C1, and `N = 50 / 70` as C2 is planned.

The effect on any single p-value is small. The exposure is not: R1 explicitly
endorsed the protocol, and a duplicated benchmark is the kind of thing a
reviewer checks directly.

### The decision — Mario's, not an agent's

| Option | Effect | Cost |
|---|---|---|
| **A. Restore the `1/(2π)`** in I.34.27, matching the cited AI Feynman definition | The two problems become distinct; `N` stays 70 | Changes a **D1 problem definition**, so I.34.27 joins the §7 continuity-table exclusion list alongside the five T05 already excludes. Adds a `CONST` node, nudging the problem's bottleneck type toward "constant" |
| **B. Drop one of the two**, report `N = 69` (and `N = 49` for the D1-only CPDT) | Honest and simple; no definition changes | Loses one problem; every `N` in the paper and the continuity table changes |

**Recommendation: A.** It restores fidelity to the source we cite, keeps `N`,
and the continuity-exclusion machinery for corrected D1 definitions already
exists and is already being used for five other problems. `1/(2π)` is a leading
multiplicative constant, which both hosts fit trivially, so it should not
materially change difficulty.

Either way the change is **pre-launch**: the data generator runs during every
run, so this is on the "during-run" side of §3's dividing line.

---

## 3. Finding B (expected) — Pagie-1 and Keijzer-6 do not vary with the seed

Measured across all 70 problems at seeds 0 and 101: **exactly 2 are
seed-invariant**, and both are deterministic by construction.

| Problem | Sampling | Why the seed cannot matter |
|---|---|---|
| **Pagie-1** | `grid_2d_skip_zero` | 26 × 26 deterministic grid, origin skipped (676 train / 2500 test) |
| **Keijzer-6** | `integer_grid` | fixed integer grid, 50 train / 120 test, an extrapolation benchmark |

This is the **published protocol** for both, and `CLAUDE.md` says explicitly of
these shapes: *"these are not typos, do not 'fix' them."* It is not a seeding
defect and there is nothing to repair.

**But it must be disclosed, because it changes what a seed means.** For these
two problems the campaign's 20 seeds replicate the **search's RNG only**, not
the data. Every other problem replicates both. A per-problem variance figure for
Pagie-1 is therefore a measure of search stochasticity alone, and should not be
read as sampling variability. Pagie-1 is one of the five structural-bottleneck
problems where IsalSR shows its largest variance reduction, so this belongs in
the text rather than in a footnote discovered later.

---

## 4. What changed in the code

`experiments/scripts/c2_certify.py` — `check_c4` now separates three properties
that the single "all distinct" assertion had conflated:

1. **Cross-arm identity** (blocking) — one fingerprint per `(problem, seed)`
   across all arms and methods. *The* property C4 exists for. Currently 0
   violations.
2. **Cross-problem distinctness** (blocking) — two different problems must not
   share data. This is what now names Finding A instead of burying it.
3. **Cross-seed distinctness** (blocking, with a declared exemption) — seeds
   within a problem must differ, except for `SEED_INVARIANT_PROBLEMS =
   {Pagie-1, Keijzer-6}`, whose exemption is stated in the source with its
   justification and reported in the evidence JSON rather than silently applied.

The multiplicity histogram now excludes exempt fingerprints, so a
deterministic-grid problem legitimately appearing 18× is no longer reported as
wrong.

**No number in any table changes as a result of this file.** Finding A's fix,
if taken, does change I.34.27's numbers — that is the point of taking it before
launch rather than after.
