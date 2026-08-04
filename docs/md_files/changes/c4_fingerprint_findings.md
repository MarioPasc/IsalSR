# C4 — what 203 distinct fingerprints instead of 210 actually means

**Date**: 2026-08-04
**Found by**: Stage C certification, job 1758604 (`c2_certify.py`, verdict NO-GO)
**Status**: ✅ **BOTH DECISIONS TAKEN, IMPLEMENTED, AND VERIFIED ON PICASSO.**

> **Stage C re-certified GO on `c2_smoke_v3` (job 1761777): 19/19 criteria pass,
> 0 blocking failures.** C4 now reports `cross_arm_disagreement 0`,
> `duplicate_problems_blocking 0`, `seed_collapse_blocking 0`,
> `wrong_multiplicity 0`, with a multiplicity histogram of **`{6: 204, 18: 2}`**
> — 204 fingerprints at the correct multiplicity of 6 (3 arms × 2 methods) and
> the two declared deterministic grids at 18 (× 3 seeds). 1,260/1,260 tasks were
> placed on `sr`.

> ### Resolution
>
> **Finding A → option A.** The `1/(2π)` is restored in `I.34.27`
> (`benchmarks/datasets/feynman.py`), matching the AI Feynman database. The two
> problems are now distinct; `N` stays at 70. **`I.34.27` joins §7's
> continuity-table exclusions** — C1 ran a different function under that name.
> Regression tests: `tests/unit/test_benchmark_suite_distinctness.py` (5 tests,
> verified to fail 3/5 against the old definition).
>
> **Finding C → pin the node family.** `slurm/c2_smoke/launcher.sh` now defaults
> to `--constraint=sr` (AMD EPYC 7H12; 154 nodes × 128 c = **19,712 cores**, the
> largest pool and >2× the `cpu=9000` entitlement). This closes **B6**. The
> accepted cost is slower queue access than an unpinned `cpu` request —
> comparability over turnaround.
>
> **Finding B** stays as documented: a disclosure obligation, not a defect.

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

## 3b. Finding C (blocking, found in the v2 re-run) — the data is not bit-reproducible across CPU families

The Stage C **re-run** (`c2_smoke_v2/`, throttle `%24`) failed C4 differently:
**238** distinct fingerprints over 210 `(problem, seed)` pairs, i.e. **35 pairs
carrying more than one fingerprint** — where the first wave had **zero**.

### It is not a seeding bug, a config race, or non-determinism

| Ruled out | Evidence |
|---|---|
| Config race | `config_sha256` **identical** across every split (e.g. `1485f9da6def` on all three `I.12.4` arms) |
| Non-deterministic generator | 8 repeated draws of the same `(problem, seed)` in one process → **1** fingerprint |
| Runner mutating the arrays | `data_fingerprint` is computed in `orchestrator.py:697`, **before any arm touches the data** |

### It is CPU architecture, and the correlation is perfect

**All 35 splits partition *exactly* by node family — 0 exceptions.** The odd cell
is the one that landed on a different family:

```
I.12.4/seed_0   2a798997845a  bingo{baseline,hash,isalsr} + udfs{hash,isalsr}   → sr (AMD)
                cf001e447874  udfs/baseline                                     → sd (Intel)
```

Minority cells by arm: `udfs/baseline` 33, `bingo/baseline` 5, `udfs/hash` 2 —
i.e. it is whichever arm happened to be scheduled on the minority architecture,
not a property of the arm.

The first wave showed 0 disagreements only because `%8` kept it on a narrower
slice of the pool; `%24` spread it across `sd` and `sr` and exposed this.

### The magnitude, measured rather than assumed

`slurm/c2_smoke/arch_probe.sh` ran the identical generator pinned to `sd005`
(Intel Xeon Gold 6230R) and `sr004` (AMD EPYC 7H12), same numpy 2.4.6:

| Problem | Fingerprint | max \|Δy\| | max ULP |
|---|---|---|---|
| I.12.4 | DIFFER | **5.551×10⁻¹⁷** | **1** |
| I.6.20a, Nguyen-1, Nguyen-2, Liv-14, Vladislavleva-2 | DIFFER | 0 on the sampled head | 0 |
| I.14.3, Nguyen-5 | MATCH | 0 | 0 |

**The difference is one unit in the last place** — libm/SIMD rounding in the
transcendental evaluation, at 5.6×10⁻¹⁷ against a machine epsilon of 2.2×10⁻¹⁶.
Its effect on R², NRMSE or any reported metric is **nil**. Its effect on a
*byte-exact* fingerprint is total, which is why C4 fails.

### The decision — Mario's

| Option | Effect | Cost |
|---|---|---|
| **1. Pin the node family for C2** (`--constraint=sr`) | C4 passes exactly; the paired arms provably see identical data | Narrows the pool; interacts with §3.3's 256 GB Bingo–IsalSR request (`sr` = 439 GB/node → 1 such task per node) |
| **2. Relax the fingerprint** to a tolerance hash (e.g. 12 significant digits) | Keeps the whole pool and the measured 476-core concurrency | C4 stops certifying byte-exact identity, which was the entire point of P3 |

**Recommendation: option 1, and it closes B6 as a side effect.** Pinning is
independently required by the plan's own reasoning: **wall clock is a reported
quantity** (`S`, the overhead percentages, Table 2's cost column), and Intel `sd`
at 2.1 GHz versus AMD `sr` at 2.6 GHz turns any timing comparison across a mixed
pool into a measurement of the scheduler. The `picasso-sbatch` skill states the
same rule. One constraint flag therefore fixes C4 **and** removes the timing
confound **and** resolves the open B6 decision.

Do **not** take option 2 on the grounds that 1 ULP is harmless. It is harmless
*to the metrics*, and the honest way to say so is to pin the pool and keep the
strict check, not to weaken the check until it stops reporting.

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
