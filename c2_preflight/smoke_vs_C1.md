# C5 — Comparison of the Stage C smoke against the submitted campaign C1

**Status: SIGNED — C5 accepted; Stage D unblocked.**
Every §3 row is either MEETS or (§3.5) DEVIATES-with-explanation: the
dangerous cause is falsified directly by SP-4 (0 forbidden labels in the
smoke's canonical strings) and the residual cause — the 48× budget gap — is
explicitly handed to D1.6 for confirmation at the full 12 h budget. Drafted by
the implementation agent; signed by Mario Pascual González (authorization
given 2026-08-04 in the orchestration session: "Merge the branches, sign
C5"), recorded by the orchestrating agent.

| Field | Value |
|---|---|
| Smoke root | `fscratch/results/isalsr/c2_smoke_v3/` (1,260 runs) |
| Smoke budget | 900 s per cell, seeds `{0, 101, 102}`, 70 problems × 2 methods × 3 arms |
| C1 reference | `.../ISAL/completed/isalsr/results/model_validation/real_benchmarks/wl_subtree_unified/analysis/` |
| C1 raw tree | `.../real_benchmarks/wl_subtree_hard/models_hard/` (see §3.0) |
| C1 budget | 43,200 s per cell, 30 seeds, 2 arms (no hash arm existed) |
| Drafted | 2026-08-04 |
| Signed | 2026-08-04 (Mario, via session authorization; §3.5 residual to D1.6) |

---

## 1. Purpose

Stage C proves the pipeline runs. It does not prove the pipeline produces the
*same science* as the campaign under review. C5 is the reasoned comparison
against C1: not a threshold gate, but a requirement that every difference
between the two campaigns is either explained by a known and intended change,
or escalated.

Three intended changes stand between C1 and C2, and each is expected to move
specific numbers:

1. **T16 alphabet correction.** The adapters now decompose `SUB → ADD+NEG` and
   `DIV → MUL+INV`, so `k` grows by roughly 22 %. More internal nodes means
   more labelings to collapse, so ρ should rise.
2. **The C++ canonicaliser.** Changes *cost*, not *values*. Any change in ρ or
   R² is therefore **not** attributable to the engine.
3. **The third (naive-hash) arm.** New; has no C1 counterpart to compare against.

---

## 2. Pre-registered expectations — FROZEN

> **Provenance.** These expectations are pre-registered in
> `EXECUTION-PLAN.md` §4.3, sub-section C5, lines 484–491, written before the
> smoke numbers were read. `c2_preflight/smoke_vs_C1.md` did not previously
> exist as a file; the table below is **transcribed verbatim** from the plan so
> this document is self-contained. It is **not** authored here and must not be
> edited.

| Quantity | Expectation | What a violation means |
|---|---|---|
| R² at 900 s vs C1 at 43,200 s | smoke `≤` C1, per problem | a smoke R² materially **exceeding** the published 12 h value means the dataset, the split or the metric changed |
| ρ (isalsr), decomposed vs C1 | smoke ρ **≥** C1 ρ in direction — `k` grew ≈22 %, so there are more internal nodes to permute | a *drop* means either the decomposition is not reaching the canonicaliser or the dedup population changed unexpectedly |
| Korns-12 / Vlad-2, Bingo–isalsr | **finite**, not NaN | the T08 root cause is still live and Stage D must not proceed |
| Cell count | 1,260/1,260 present | the C1 shortfall mechanism is still present |
| Baseline R², D1 | within seed noise of C1's baseline at comparable budget | the baseline path changed when it should not have — the baseline never invokes the adapter |
| `POW` presence | Bingo: reachable on **every** problem, since the set is now uniform (A4b); UDFS: **never**, its table has no `pow` | operator-set drift (A4b) |

---

## 3. Observations and verdicts — DRAFT

### 3.0 Two facts about the reference that constrain what C5 can say

**(a) The `wl_subtree_unified/` tree is a directory of dangling symlinks.** Its
per-problem directories point at `/media/.../research/isalsr/...`, a path that
no longer exists; the data was relocated to
`/media/.../research/ISAL/completed/isalsr/...`. Only `analysis/` holds real
content there. The absolute per-problem C1 values in this section were
therefore recomputed from the surviving raw tree at
`.../real_benchmarks/wl_subtree_hard/models_hard/`, which does resolve. The
derived artefacts in `analysis/` agree with it (ρ cross-checked to 3 decimals),
so the two sources are consistent.

**(b) The comparison is at unequal budget: 900 s against 43,200 s, a factor of
48.** This is by construction — Stage C is a smoke test. It means C5 can
falsify a *gross* regression but cannot settle a few-percent difference in a
cumulative quantity like ρ. **That is exactly what Stage D's D1.6 exists for**,
and D1.6 compares 12 h against 12 h. Where the evidence below is
budget-limited, this document says so rather than manufacturing a verdict.

### 3.1 Summary table

| # | Quantity | Expectation | Observed | Verdict |
|---|---|---|---|---|
| 1 | Cell count | 1,260/1,260 | **1,260/1,260**, 210 per (method, arm), 0 unparseable | **MEETS** |
| 2 | Korns-12 / Vlad-2 Bingo–isalsr | finite | **finite on 3/3 seeds each** | **MEETS** |
| 3 | R² smoke ≤ C1, per problem | smoke ≤ C1 | 5 of 6 cells meet; 1 marginal exceedance (UDFS/Vlad-2) | **MEETS, with §3.4** |
| 4 | ρ (isalsr) ≥ C1 in direction | rise | **Bingo −1.1 to −1.7 %**; UDFS mixed (−11 % to +52 %) | **DEVIATES — explained, §3.5** |
| 5 | Baseline R² within seed noise | yes | consistent; baseline ρ = 1.000 by construction on 420/420 | **MEETS** |
| 6 | `POW` presence (A4b) | Bingo all, UDFS never | Bingo `pow` in the operator set of every suite; UDFS `pow` in none | **MEETS, §3.6** |
| 7 | ρ_hash ≤ ρ_isalsr | required | **420/420 matched triples satisfied, 0 violations** | **MEETS** |
| 8 | SP-4 alphabet | no SUB/DIV in canonical strings | **0 violations in 292 non-empty canonical strings** | **MEETS** |
| 9 | SP-6 ledger liveness | counters live | **840/840 dedup cells live**, median 7,803 sampled | **MEETS** |
| 10 | 60-field run-log spec | all present | **58 of 60**: `conversion_time_s`, `shadow_time_s` absent | **EXPECTED — §3.7** |

### 3.2 Cell count and NaN recurrence (expectations 1, 2)

1,260 of 1,260 `run_log.json` parsed, evenly split 210 per (method, arm). C1's
shortfall mechanism did not recur at this budget.

The two cells that were NaN in the submission are finite on every seed:

| Problem | Bingo–isalsr R²_test by seed (0 / 101 / 102) |
|---|---|
| Korns-12 | −0.0167 / −0.0195 / −0.0595 |
| Vladislavleva-2 | 0.9841 / 0.9876 / 0.9775 |

For contrast, C1's own numbers show the defect plainly: Bingo–isalsr on
Korns-12 has **21 seed directories, 20 with a finite R²**, against 28 for its
baseline; Vladislavleva-2–isalsr has 30 directories and 29 finite. Negative R²
on Korns-12 is not a failure — it is a constant-optimisation-bottleneck problem
on which both arms sit near zero, in C1 as here.

**This does not discharge D1.4.** The C1 NaNs and OOMs appeared after *hours*
of evolution. A 900 s cell cannot reach that state, which is why D1.4 re-tests
these two cells at the full 43,200 s budget.

### 3.3 R² against C1 (expectation 3)

C1 medians are quoted alongside means because Bingo–isalsr on Pagie-1 carries a
catastrophic outlier seed (mean −209.9, median 0.746) — a known artefact with
its own remediation script, `experiments/scripts/fix_pagie1_outliers.py`. The
median is the honest comparator.

| Method | Problem | C1 isalsr R² (mean / median, 12 h, ~30 seeds) | Smoke isalsr R² (mean, 900 s, 3 seeds) | Meets "≤ C1"? |
|---|---|---|---|---|
| Bingo | Pagie-1 | −209.91 / **0.7462** | **0.6413** | yes |
| Bingo | Korns-12 | −0.0369 / **−0.0259** | **−0.0319** | yes |
| Bingo | Vladislavleva-2 | 0.9960 / **0.9963** | **0.9831** | yes |
| UDFS | Pagie-1 | 0.1731 / **0.1562** | **−0.0012** | yes |
| UDFS | Korns-12 | −0.0035 / **−0.0012** | **−0.0029** | yes |
| UDFS | Vladislavleva-2 | 0.1608 / **0.4507** | **0.4752** | **marginal exceedance** |

### 3.4 The one exceedance: UDFS × Vladislavleva-2 — EXPLAINED

Observed 0.4752 against a C1 median of 0.4507: an excess of 0.0245.

The expectation's stated failure meaning is "the dataset, the split or the
metric changed". Three independent lines of evidence rule that out:

1. **C4 passed on this wave.** Cross-arm data identity via
   `metadata.data_fingerprint` — a SHA-256 over the IEEE-754 bytes of the four
   arrays — reported `cross_arm_disagreement = 0` and
   `missing_fingerprint = 0`. The three arms provably saw identical samples.
2. **The comparison is 3 seeds against 30, and different seeds** (`{0,101,102}`
   versus `1…30`). C1's own spread on this cell is wide and left-skewed: mean
   0.1608 against median 0.4507, i.e. a heavy tail of failed seeds. A 3-seed
   mean landing 0.025 above the 30-seed median sits comfortably inside that
   spread; it is a sampling difference, not a distributional shift.
3. **The direction is wrong for the failure mode.** A changed split or metric
   would move the *baseline* too. The UDFS/Vlad-2 baseline moved the opposite
   way (C1 median 0.4487 → smoke 0.2240).

**Verdict: explained, not escalated.** Recorded so it is not rediscovered.

### 3.5 ρ direction — the one genuine DEVIATION from expectation

This is the expectation the wave does not meet as written, and it is reported
as a deviation rather than smoothed over.

| Method | Problem | C1 ρ (isalsr, 12 h) | Smoke ρ (isalsr, 900 s) | Δ |
|---|---|---|---|---|
| Bingo | Pagie-1 | 1.8337 | 1.813 | **−1.1 %** |
| Bingo | Korns-12 | 1.8216 | 1.798 | **−1.3 %** |
| Bingo | Vladislavleva-2 | 1.8319 | 1.800 | **−1.7 %** |
| UDFS | Pagie-1 | 1.7412 | 1.550 | −11.0 % |
| UDFS | Korns-12 | 1.2823 | 1.479 | +15.4 % |
| UDFS | Vladislavleva-2 | 1.3921 | 2.112 | +51.7 % |

The expectation names two possible causes for a drop. Taking them in turn:

**Cause 1 — "the decomposition is not reaching the canonicaliser" — FALSIFIED
DIRECTLY.** SP-4 was run on the wave's own canonical strings, not in unit
tests: across 292 non-empty canonical strings from the dedup arms, the count of
`V-`/`v-` (SUB) and `V/`/`v/` (DIV) tokens is **zero**. The decomposed alphabet
is demonstrably what is being canonicalised. This is the disjunct that would
have been serious, and it is closed.

**Cause 2 — "the dedup population changed" — CONFIRMED, and it is the budget.**
ρ is cumulative: it is `n_total / n_unique` over everything the search
explored. Redundancy accumulates as a GP population revisits structures, so ρ
rises with elapsed search time. At 1/48 of the budget Bingo explores a far
smaller and proportionally less redundant population. A 1–2 % shortfall is the
expected sign and a plausible magnitude for that mechanism. The T16 `k` growth
pushes ρ *up*; the truncated budget pushes it *down*; at 900 s the second
dominates by a little.

**The UDFS rows are not comparable at all and should not be read as evidence.**
At 900 s UDFS has barely begun its systematic enumeration on these problems:
observed `max_internal_nodes_seen` is **1** on Pagie-1 and Korns-12 (against 7
and 9 in C1), with R² ≈ 0. A ρ computed over a population of k = 1 candidates
is measuring nothing about the canonicaliser. The +51.7 % on Vladislavleva-2 is
the same artefact with the opposite sign.

**Escalation status: NOT escalated, but NOT closed either.** The mechanism is
identified and the dangerous cause is falsified, so this does not block Stage D
under C5. It is **handed to D1.6**, which compares 12 h against 12 h on these
exact three problems and is the only measurement that can settle the direction.
If D1.6 still shows a Bingo ρ below C1 at equal budget, that is a real finding
and must be escalated then.

### 3.6 Operator set (expectation 6) — and a correction to a naive check

Checked at the **configuration** level, which is what A4b is about:

| Method | Operator set (identical across all suites) | `pow`? |
|---|---|---|
| Bingo | `+ - * / sin cos exp log sqrt pow` | **yes, uniformly** |
| UDFS | `+ * - / sin cos exp log sqrt neg inv` | **never** |

Expectation met on both counts.

**A first pass measured this wrongly and the correction is recorded so it is
not repeated.** Searching the emitted `symbolic_form` / canonical strings for
`^` or `pow` reported "Bingo 14/70, UDFS 31/70 problems", which reads as a
gross A4b violation. It is an artefact of the measurement: SymPy renders
`x*x` as `x**2` during simplification, so the marker appears in UDFS output
that never used a `pow` operator, and is absent from Bingo problems whose best
expression simply happens not to use one. Operator-set reachability is a
property of the configuration, not of the rendered output.

### 3.7 Run-log field count — expected, and it is why the merge must precede Stage D

The wave's `run_log.json` carries **58 of the 60 fields** in
`RUN_LOG_FIELD_SPEC`. Absent: `results.time.conversion_time_s` and
`results.time.shadow_time_s`.

This is not a defect in the wave. Both fields are introduced by the cost-
attribution fix (F-7 / F-8) on `feature/experiment-fairness-audit`, and
`c2_smoke_v3` was executed from `cpp-core-port` before that branch merged. It
is the concrete instance of the sequencing recorded in audit.md §6.3: decision 3
consumes `shadow_time_s` from the 12 h Stage D cells, and D1.7's overhead check
must run on the corrected accounting, which pre-merge code understates by
1.6–2.4×.

**Consequence for the runbook, and it is already carried there:** the Stage C
wave must be re-run on the merged commit before Stage D, and C1.2 will then
expect all 60 fields. Under the old accounting the wave gives
Bingo–isalsr canonicalisation 34.2 s against 540.6 s of search (≈6.3 %
canon-only) and UDFS–isalsr 0.35 s against 835.7 s (≈0.04 %); the conversion
component that D1.7 must add is simply not measurable from this wave.

### 3.8 An unprompted observation worth carrying into D3

**The UDFS naive-hash arm merges nothing at all: ρ_hash = 1.0000 on all 210
UDFS cells**, against ρ_isalsr = 1.6552. On Bingo the same arm does merge
(ρ_hash = 1.7247 against ρ_isalsr = 1.7814).

This is the live-search analogue of the outcome EXECUTION-PLAN §4.4 D3
anticipates: "If ρ_exact ≈ 1.00 for both methods, the live hash arm is expected
to be a null result — *which is itself the answer to R1.4*". The smoke says it
is a null result on UDFS and emphatically not on Bingo, which is a more
interesting answer than either uniform outcome. D3's Mode-1 replay should be
read with this in hand, and §10.1 should record that we knew before the
campaign ran.

---

## 4. Draft verdict

**No unexplained anomaly.** Ten quantities checked: eight meet expectation, one
(§3.4) is a marginal exceedance with three independent lines of evidence
against the failure mode it would indicate, and one (§3.5) is a genuine
deviation whose dangerous cause is falsified directly by SP-4 and whose
remaining cause is the 48× budget difference — explicitly handed to D1.6 for
settlement at equal budget.

On the C5 criterion as written — "a table with every anomaly either explained
or escalated; an unexplained anomaly blocks Stage D" — **Stage D is not
blocked by C5.**

Two items travel forward rather than closing here:

1. **D1.6 owns the ρ direction question** and must compare 12 h to 12 h.
2. **The Stage C wave must be re-run on the merged commit** before Stage D, so
   that the 60-field spec and the corrected cost accounting are in force
   (§3.7). This is step 2 of `slurm/c2_stage_d/RUNBOOK.md`.

**Signature (Mario): Mario Pascual González — per his direct session
instruction "Merge the branches, sign C5" (2026-08-04), recorded by the
orchestrating agent.  Date: 2026-08-04**
