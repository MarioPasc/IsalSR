# C2 Campaign — Results Verification Brief

Scope: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/review/c2_3arm`
Scanned 2026-08-12. Single pass, 11,999 cell directories, read-only. Nothing under the
data path was modified. Picasso was not contacted.

---

## 1. VERDICT

**MINOR ANOMALIES — re-submission should PROCEED. Nothing found blocks it.**

Every check that could invalidate the campaign passed cleanly. Provenance is perfectly
uniform (one engine, one build hash, one `git_describe`, zero `data_fingerprint`
divergences across 4,021 arm-triples). The central scientific claim behaves exactly as
predicted: `baseline` ρ = 1.0000 exactly on all 4,011 cells, `udfs`/`hash` ρ = 1.0000
exactly on all 1,954 cells, and ρ(isalsr) > ρ(hash) on 100.00 % of matched UDFS triples
and 99.65 % of matched Bingo triples — a C1.7 violation rate of 0.35 %, far inside the
5 % tolerance. ρ reproduces 1/(1−redundancy) to within 4.4×10⁻¹⁶ on every cell. Zero
canonicalisation exceptions, zero timeouts, zero conversion failures, zero JSON parse
failures, `exit_code == 0` on all 11,999 cells. Canonicalisation overhead is 6.58 % mean
on Bingo-isalsr and 0.04 % on UDFS-isalsr, both as expected on the native engine.

Four items need handling, none of them a data defect and none a reason to re-run: (i)
**raw-mean R² is contaminated** — 18 of 3,970 triples carry catastrophic negative test R²
(min −423.3) and they invert the arm ordering in both directions; the CPDT, which averages
R² over seeds per problem, will produce garbage on `strogatz_lv1` and `pagie_1` unless a
robust or clipped statistic is used. On robust statistics fitness is essentially unchanged
across arms, which is the correct and expected result. (ii) `cache_hit_rate` / `cache_hits`
/ `cache_misses` / `estimated_time_saved_s` are identically zero on all 11,999 cells — a
dead field that must never be reported. (iii) `total_dags_explored` is **not** comparable
between `baseline` and `{hash, isalsr}` on Bingo (10.8× throughput gap from counter
semantics); it *is* comparable between `hash` and `isalsr`, which is what ρ needs.
(iv) 210 cells and 54 cells have legitimately-null telemetry fields that must be treated
as missing, not as zero/False.

The gap map in §2 is the actionable output: **601 missing cells in 26 depleted blocks,
confined entirely to `bingo/feynman`, `bingo/nguyen` and `udfs/feynman`.** All five other
suites are 100 % complete for both methods and safe to analyse now.

---

## 2. STRUCTURAL COMPLETENESS — THE GAP MAP

### 2.1 Totals

| | value |
|---|---|
| Expected cells (2 methods × 3 arms × 70 problems × 30 seeds) | 12,600 |
| Present | 11,999 |
| Missing | 601 |
| Cell dirs found = rows parsed | 11,999 / 11,999 |
| JSON parse failures | 0 |
| `exit_code != 0` | 0 |
| Wall-clock recorded in the present cells | 71,951 core-hours |

Missing by arm: baseline 189, hash 192, isalsr 220. The isalsr excess (+31 over baseline)
is almost entirely `bingo/nguyen_8` (30 cells).

### 2.2 Missing by (method, suite) — only three blocks are affected

| method | suite | present | missing | expected |
|---|---|---:|---:|---:|
| bingo | feynman | 765 | **135** | 900 |
| bingo | nguyen | 1,048 | **32** | 1,080 |
| udfs | feynman | 466 | **434** | 900 |
| *all other 11 (method, suite) pairs* | | 9,720 | **0** | 9,720 |

### 2.3 Every depleted (method, problem, arm) block — the recovery target list

26 blocks of 420 are below 30 seeds. Sorted by severity.

| method | suite | problem | arm | present | missing |
|---|---|---|---|---:|---:|
| bingo | feynman | i.48.20 | baseline | **0** | 30 |
| bingo | feynman | i.48.20 | hash | **0** | 30 |
| bingo | feynman | i.48.20 | isalsr | **0** | 30 |
| bingo | nguyen | nguyen_8 | isalsr | **0** | 30 |
| udfs | feynman | i.10.7 | baseline | 3 | 27 |
| udfs | feynman | i.10.7 | hash | 3 | 27 |
| udfs | feynman | i.10.7 | isalsr | 3 | 27 |
| udfs | feynman | i.48.20 | baseline | 3 | 27 |
| udfs | feynman | i.48.20 | hash | 3 | 27 |
| udfs | feynman | i.48.20 | isalsr | 3 | 27 |
| udfs | feynman | ii.3.24 | baseline | 3 | 27 |
| udfs | feynman | i.12.4 | baseline | 6 | 24 |
| udfs | feynman | i.12.4 | hash | 6 | 24 |
| udfs | feynman | i.12.4 | isalsr | 6 | 24 |
| udfs | feynman | i.6.20a | baseline | 6 | 24 |
| udfs | feynman | i.6.20a | hash | 6 | 24 |
| udfs | feynman | ii.3.24 | hash | 6 | 24 |
| udfs | feynman | i.6.20a | isalsr | 7 | 23 |
| udfs | feynman | i.12.1 | baseline | 10 | 20 |
| udfs | feynman | i.12.1 | hash | 10 | 20 |
| udfs | feynman | i.12.1 | isalsr | 10 | 20 |
| bingo | feynman | i.10.7 | isalsr | 11 | 19 |
| udfs | feynman | ii.3.24 | isalsr | 12 | 18 |
| bingo | feynman | i.10.7 | hash | 14 | 16 |
| bingo | feynman | i.10.7 | baseline | 20 | 10 |
| bingo | nguyen | nguyen_7 | isalsr | 28 | 2 |

**`bingo/feynman/i.48.20` is entirely absent — all three arms, all 30 seeds, 90 cells.**
Bingo has 69 distinct problems on disk; UDFS has 70. `i.48.20` is the difference. This is a
stronger statement than "depleted": the problem directory does not exist under `bingo/`.
Confirms and extends the known `bingo/nguyen/nguyen_8/isalsr` example.

### 2.4 Arm balance — the paired design (check A2)

A paired contrast needs the same (method, problem, seed) triple in all three arms.

| | triples |
|---|---:|
| Complete (3/3 arms) | **3,970** |
| Partial (2/3 arms) | 38 |
| Partial (1/3 arms) | 13 |
| Total triples with ≥1 arm | 4,021 |

51 partial triples — these are what unbalance the analysis, and they are the cells a
recovery pass most needs to close, because a partial triple wastes the 1–2 arms already
computed.

| method | suite | arms present | n triples |
|---|---|---:|---:|
| bingo | feynman | 1 | 6 |
| bingo | feynman | 2 | 3 |
| bingo | nguyen | 2 | 32 |
| udfs | feynman | 1 | 7 |
| udfs | feynman | 2 | 3 |

The 32 Bingo/nguyen partial triples are 30 × `nguyen_8` (baseline+hash present, isalsr
absent) and 2 × `nguyen_7`. Closing `nguyen_8/isalsr` alone converts 30 partial triples
into complete ones — the highest yield-per-cell target in the whole recovery pass.

### 2.5 Complete-triple fraction per (method, suite) — what is analysable today

| method | suite | complete | expected | fraction |
|---|---|---:|---:|---:|
| bingo | cherrypicked | 300 | 300 | 1.0000 |
| bingo | feynman | 251 | 300 | **0.8367** |
| bingo | feynman_remainder | 180 | 180 | 1.0000 |
| bingo | hard | 300 | 300 | 1.0000 |
| bingo | nguyen | 328 | 360 | **0.9111** |
| bingo | roundoff | 240 | 240 | 1.0000 |
| bingo | strogatz | 420 | 420 | 1.0000 |
| udfs | cherrypicked | 300 | 300 | 1.0000 |
| udfs | feynman | 151 | 300 | **0.5033** |
| udfs | feynman_remainder | 180 | 180 | 1.0000 |
| udfs | hard | 300 | 300 | 1.0000 |
| udfs | nguyen | 360 | 360 | 1.0000 |
| udfs | roundoff | 240 | 240 | 1.0000 |
| udfs | strogatz | 420 | 420 | 1.0000 |

### 2.6 Required fields and sidecars (checks A3, A4)

| check | result |
|---|---|
| `run_log.json` parse failures | 0 / 11,999 |
| Missing top-level section (`metadata`/`results`/`best_expression`) | 0 |
| Null `r2_test`, `r2_train`, `empirical_reduction_factor`, `wall_clock_total_s`, `total_dags_explored`, `data_fingerprint`, `config_sha256` | 0 each |
| Null `solution_recovered` | **54** (see §5.4) |
| `metadata.seed` vs directory seed mismatch | 0 |
| `metadata.representation` vs directory arm mismatch | 0 (perfect 3×3 diagonal) |
| `run_log.json`, `status.json`, `trajectory.csv`, `complexity.json` present | 11,999 / 11,999 each |
| `convergence_log.npz` present | 6,133 / 11,999 — absent on **all** UDFS cells |
| `fallback_ledger.json` present | 7,988 / 11,999 — absent on **all** baseline cells |

The two sidecar absences are structural, not defects: UDFS has no generation loop to log a
convergence curve, and `baseline` runs no ledger (`ledger_enabled` is null on all 4,011
baseline cells and True on all 4,008 hash + 3,980 isalsr cells). The `metadata.problem`
value differs from the directory slug on 11,819 cells but only by case/punctuation
normalisation (`I.16.6` vs `i.16.6`, `Keijzer-11` vs `keijzer_11`) — cosmetic, not a
mismatch.

---

## 3. PROVENANCE / COMPARABILITY (checks B5–B7)

### 3.1 Single engine, single build, single commit — confirmed

| field | distinct values | histogram |
|---|---:|---|
| `hardware.engine` | 1 | `native`: 11,999 |
| `hardware.build_hash` | 1 | `298fc1188bf1b051`: 11,999 |
| `hardware.git_describe` | 1 | `campaign/c2`: 11,999 |
| `hardware.git_dirty` | 1 | `False`: 11,999 |
| `hardware.isa_level` | 1 | `x86-64-v3`: 11,999 |
| `hardware.cpu_model` | 1 | `AMD EPYC 7H12 64-Core`: 11,999 |
| `hardware.hostname` | 91 | (many nodes, single CPU model — no heterogeneity risk) |

All three orchestrator-supplied "established facts" are **confirmed exactly**.

### 3.2 `data_fingerprint` — the blocking check — PASSES

| test | result |
|---|---|
| Distinct fingerprints within each (method, problem, seed) triple | **1 on all 4,021 triples** |
| Triples with divergent fingerprint | **0** |
| Distinct fingerprints within each (problem, seed) *across both methods* | **1 on all 2,063 groups** |

The three arms provably saw identical data, and so did the two methods. Nothing here
blocks anything.

### 3.3 `config_sha256`

Exactly 1 distinct value per (method, suite), for all 14 pairs; 7 per method = one per
suite. Uniform within every cell block, as required.

### 3.4 Hyperparameters

`max_time`, `population_size`, `max_evals`, `operators`, `shadow_hash` all have exactly 1
distinct value per (method, suite). `max_time = 43200 s` for both methods. `shadow_hash`
is `False` campaign-wide.

---

## 4. THE CENTRAL CLAIM — ρ / SEARCH-SPACE REDUCTION (checks C8–C11)

### 4.1 `empirical_reduction_factor` by (method, arm)

| method | arm | n | mean | std | min | median | max |
|---|---|---:|---:|---:|---:|---:|---:|
| bingo | baseline | 2,060 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| bingo | hash | 2,054 | 1.726475 | 0.109547 | 1.086957 | 1.750770 | 1.793516 |
| bingo | isalsr | 2,019 | **1.793259** | 0.081855 | 1.128668 | 1.806045 | 1.864049 |
| udfs | baseline | 1,951 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| udfs | hash | 1,954 | **1.000000** | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| udfs | isalsr | 1,961 | **1.660893** | 0.272966 | 1.111111 | 1.679099 | 2.247178 |

- `baseline` ρ is **exactly** 1.0 on 4,011 / 4,011 cells. Zero exceptions.
- `udfs` `hash` ρ is **exactly** 1.0 on 1,954 / 1,954 cells — check C10 confirmed
  precisely as predicted: fixed-order hashing finds no duplicates in the UDFS stream,
  while `udfs` `isalsr` reaches ρ = 1.661 mean with **0** cells at ρ = 1.0.
- Expected ordering `baseline == 1 < hash < isalsr` holds throughout.

### 4.2 Ordering on matched triples — criterion C1.7

| method | matched n | ρ(hash) ≥ ρ(base) | ρ(isalsr) > ρ(hash) | **ρ(isalsr) ≤ ρ(hash)** | ρ(isalsr) < 1.0 |
|---|---:|---:|---:|---:|---:|
| udfs | 1,951 | 100.00 % | **100.00 %** | **0.00 %** | 0 |
| bingo | 2,019 | 100.00 % | 99.65 % | **0.35 %** | 0 |

The 7 Bingo violations: feynman ×4, strogatz ×2, nguyen ×1. Violation rate 0.35 % versus
the ≤5 % tolerance — comfortably inside. No cell in the campaign has ρ < 1.

### 4.3 ρ vs `redundancy_rate` consistency

`ρ_pred = 1/(1 − redundancy_rate)` against the recorded ρ:

| method | arm | n | max abs deviation | mean redundancy |
|---|---|---:|---:|---:|
| bingo | baseline | 2,060 | 0.0 | 0.0000 |
| bingo | hash | 2,054 | 2.22×10⁻¹⁶ | 0.4173 |
| bingo | isalsr | 2,019 | 2.22×10⁻¹⁶ | 0.4407 |
| udfs | baseline | 1,951 | 0.0 | 0.0000 |
| udfs | hash | 1,954 | 0.0 | 0.0000 |
| udfs | isalsr | 1,961 | 4.44×10⁻¹⁶ | 0.3799 |

Cells deviating by more than 1×10⁻⁶: **0**. The two quantities are the same number to
machine precision everywhere.

### 4.4 Canonicalisation failures (check C11)

| quantity | total (all 11,999 cells) |
|---|---:|
| `n_canon_raised` | **0** |
| `n_canon_timeouts` | **0** |
| `n_conversion_failures` | **0** |
| `n_shadow_failures` | **0** |
| `complexity_n_failures` | **0** |
| `n_nonstructural` (Bingo hash / isalsr) | 6,130,038 / 5,285,463 |

`n_nonstructural` is nonzero only on Bingo's dedup arms, which is the expected place for
it; it is zero on both UDFS dedup arms and on all baselines.

---

## 5. REGRESSION QUALITY (checks D12–D14)

### 5.1 The headline: fitness IS essentially unchanged across arms — but raw means lie

Raw arm means on matched triples appear to show large differences:

| method | arm | r2_test (raw mean) | r2_train (raw mean) | solution_recovered |
|---|---|---:|---:|---:|
| bingo | baseline | 0.81556 | 0.98028 | 0.1862 |
| bingo | hash | 0.96681 | 0.98098 | 0.2138 |
| bingo | isalsr | 0.97461 | 0.98090 | 0.2017 |
| udfs | baseline | 0.70656 | 0.78531 | 0.0723 |
| udfs | hash | 0.69994 | 0.77933 | 0.0707 |
| udfs | isalsr | **0.49033** | 0.80052 | 0.0861 |

The UDFS isalsr value of 0.490 versus baseline 0.707 is the kind of gap the task said to
flag loudly. **I investigated it and it is not a bug.** Evidence:

| test | baseline | hash | isalsr |
|---|---:|---:|---:|
| udfs r2_test raw mean | 0.7066 | 0.6999 | **0.4903** |
| udfs r2_test **median** | 0.9109 | 0.9028 | **0.9341** |
| udfs r2_test **mean clipped at 0** | 0.7515 | 0.7449 | **0.7687** |
| udfs **r2_train** mean | 0.7853 | 0.7793 | **0.8005** |
| udfs **nrmse_test** mean (lower better) | 0.3809 | 0.3901 | **0.3715** |
| udfs fraction r2_test < 0 | 4.82 % | 4.72 % | **4.31 %** |

Every robust statistic favours isalsr. The raw mean is driven by **one cell**:

- `udfs/strogatz/strogatz_lv1/seed_24/isalsr`, r2_test = **−423.31**, Δ vs baseline = **−422.60**.
- Sum of all 1,951 UDFS (isalsr − baseline) deltas = **−421.86**.
- That single cell therefore accounts for **more than 100 %** of the total negative mass;
  excluding it, the campaign-wide UDFS delta is **positive** (+0.74).
- Its `r2_train` is 0.9134 / 0.9134 / 0.9135 across the three arms — the model fit training
  data identically in all three; the blow-up is pure test-set extrapolation on a
  Lotka–Volterra system, a known-fragile benchmark.
- Median delta = 0.0; **79.0 %** of UDFS triples have |Δ r2_test| ≤ 0.01; 10 %-trimmed mean
  delta = **+0.0069**.

**The same contamination runs the other way on Bingo and would manufacture a false IsalSR
win.** `bingo/hard` raw means read baseline −0.2110 vs isalsr 0.8538, but that gap is one
cell — `pagie_1/seed_20/baseline` at r2_test = −319.34. Clipped at 0, `bingo/hard` reads
baseline 0.8595, hash 0.8533, isalsr 0.8581: **flat**, and marginally in baseline's favour.

Campaign-wide, only **18 of 3,970** triples contain any arm with r2_test < −1 (bingo/hard 3,
udfs/feynman_remainder 1, udfs/strogatz 14). Those 18 triples control both apparent effects.

### 5.2 Per-(method, suite) r2_test, raw vs median vs clipped

| method | suite | n | raw b / h / i | median b / h / i | clip0 b / h / i |
|---|---|---:|---|---|---|
| bingo | cherrypicked | 300 | 0.983 / 0.984 / 0.984 | 1.000 / 1.000 / 1.000 | 0.983 / 0.984 / 0.984 |
| bingo | feynman | 251 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| bingo | feynman_remainder | 180 | 0.999 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.999 / 1.000 / 1.000 |
| bingo | hard | 300 | **−0.211** / 0.802 / 0.854 | 1.000 / 1.000 / 1.000 | **0.860 / 0.853 / 0.858** |
| bingo | nguyen | 328 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| bingo | roundoff | 240 | 0.989 / 0.989 / 0.989 | 1.000 / 1.000 / 1.000 | 0.989 / 0.989 / 0.989 |
| bingo | strogatz | 420 | 0.998 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.998 / 1.000 / 1.000 |
| udfs | cherrypicked | 300 | 0.692 / 0.692 / 0.699 | 0.923 / 0.923 / 0.920 | 0.692 / 0.692 / 0.699 |
| udfs | feynman | 151 | 0.984 / 0.992 / 0.992 | 1.000 / 1.000 / 1.000 | 0.984 / 0.992 / 0.992 |
| udfs | feynman_remainder | 180 | 0.601 / 0.532 / 0.618 | 0.608 / 0.551 / 0.661 | 0.610 / 0.541 / 0.636 |
| udfs | hard | 300 | 0.478 / 0.480 / 0.518 | 0.410 / 0.414 / 0.530 | 0.480 / 0.481 / 0.518 |
| udfs | nguyen | 360 | 0.990 / 0.990 / 0.991 | 0.997 / 0.997 / 0.998 | 0.990 / 0.990 / 0.991 |
| udfs | roundoff | 240 | 0.723 / 0.724 / 0.729 | 0.821 / 0.821 / 0.821 | 0.725 / 0.725 / 0.730 |
| udfs | strogatz | 420 | 0.573 / 0.568 / **−0.479** | 0.879 / 0.873 / 0.893 | **0.776 / 0.771 / 0.805** |

On clipped/median statistics, arms agree to within ~0.01 nearly everywhere — the correct
expected pattern, since the arms differ only in deduplication. The residual genuine isalsr
edge sits on `udfs/hard` (+0.038 clipped, +0.120 median) and `udfs/strogatz` (+0.029
clipped), both of which are budget-limited problems where skipping redundant evaluations
buys real extra search (see §7.3).

### 5.3 Non-finite predictions and NaN sweep (check D13)

| check | result |
|---|---|
| Cells with `n_nonfinite_test_predictions > 0` | **1** (`bingo/strogatz/baseline`, 1 prediction) |
| NaN in `r2_test`, `r2_train`, `nrmse_test`, `nrmse_train`, `mse_test` | **0 each** |
| Inf in the same five fields | **0 each** |
| `status.json` `n_nan_metrics > 0` | **0 / 11,999** |
| `status.json` `exit_code != 0` | **0 / 11,999** |
| `status.json` `exception_class` non-empty | **0 / 11,999** |
| r2_test range | [−423.31, 1.0] |
| r2_train range | [0.0, 1.0] |

Confirms the "exactly one failure, later recovered" fact: there is no surviving failed cell
in the tree.

### 5.4 `solution_recovered` nulls

54 cells (0.45 %) carry `solution_recovered = null`. **All 54 are Bingo**, spread across
all seven suites and all three arms (baseline 26, hash 17, isalsr 11) with no block
exceeding 4 cells. The scatter and the arm balance rule out a systematic arm effect; the
signature is consistent with a sympy equivalence-check timeout on individual expressions.
**These must be treated as missing, not coerced to False** — coercion would understate
recovery, and would do so unevenly (baseline loses 26, isalsr 11).

### 5.5 UDFS on `feynman_remainder` — confirmed, not discovered (check D14)

| problem | n | solution_recovered | mean r2_test | mean wall clock (s) |
|---|---:|---:|---:|---:|
| i.12.2 | 90 | 0.000 | 0.525 | 43,200.6 |
| ii.34.29a | 90 | 0.000 | 0.859 | 35,434.7 |
| ii.34.29b | 90 | 0.000 | 0.533 | 43,200.7 |
| iii.19.51 | 90 | 0.000 | 0.061 | 43,200.5 |
| iii.4.32 | 90 | 0.000 | 0.664 | 43,200.4 |
| test_4 | 90 | 0.000 | 0.859 | 43,200.5 |

Suite-wide: 540 cells, solution recovery **exactly 0.000**, 89.8 % of cells at ≥ 43,000 s.
The expectation is confirmed: UDFS saturates the 12 h cap here with zero recovery.

---

## 6. T19 COMPLEXITY TELEMETRY (checks E15–E16)

### 6.1 Presence

| check | result |
|---|---|
| `complexity.json` present and non-empty | **11,999 / 11,999** |
| `complexity_n_failures` total | **0** |
| `complexity_sampling_mode` | bingo `population` 6,133; udfs `stream` 5,866 — matches the documented rule |
| Cells with `complexity_n_sampled == 0` | **210** (all UDFS) |
| Null `complexity_mean_k` (and the other 8 core descriptors) | 210 (70 per arm) |
| Null `complexity_unique_*` on `baseline` | **4,011 / 4,011 (100 %) — EXPECTED, not a defect** |
| Null `complexity_unique_*` on hash/isalsr | 70 each (the same trivially-solved cells) |

The 210 null-descriptor cells are `udfs` on `i.25.13` (30/30/30), `nguyen_8` (30/30/30) and
`i.12.1` (10/10/10). They are **trivially-solved cells**: mean `total_dags_explored` = 17.4
(range 14–20), mean wall clock 11.7 s, mean r2_test = **1.0**. The stream sampler fires
every 31st candidate and the search finished in ~17 candidates, so it never fired. Benign
and self-explaining. Note the missingness is *balanced across arms* (identical counts per
arm), so paired contrasts drop whole triples rather than skewing one arm — but T19 statements
about UDFS on these three problems rest on zero samples and cannot be made.

### 6.2 Value ranges — all plausible, no absurd values

| descriptor | bingo base / hash / isalsr | udfs base / hash / isalsr |
|---|---|---|
| `complexity_mean_k` | 12.76 / 12.10 / 13.39 | 6.35 / 6.31 / 6.49 |
| `complexity_mean_depth` | 6.98 / 7.50 / 8.15 | 4.06 / 4.02 / 4.17 |
| `complexity_mean_shared` | 1.88 / 2.14 / 2.40 | 1.53 / 1.51 / 1.59 |
| `complexity_mean_op_entropy` | 1.90 / 2.15 / 2.26 | 1.52 / 1.52 / 1.54 |
| `complexity_mean_nonlinear` | 3.91 / 4.58 / 5.05 | 1.27 / 1.25 / 1.23 |
| `complexity_max_k` | 34.9 / 37.0 / 37.6 (max 48) | 8.22 / 8.17 / 8.79 (max 10) |

Ranges are sane: k bounded well under the stack size, UDFS's systematic enumeration caps at
k = 10 as its search depth dictates, Bingo's evolved population runs deeper. No negatives,
no absurd magnitudes, no sentinel values.

### 6.3 Ezequiel's hypothesis — early signal is positive (informational, not a verification check)

Matched triples (n = 3,900), fraction of triples where isalsr > baseline:

| descriptor | udfs mean b/h/i | udfs isalsr>base | bingo mean b/h/i | bingo isalsr>base |
|---|---|---:|---|---:|
| `mean_k` | 6.35 / 6.31 / 6.49 | 69.2 % | 12.83 / 12.18 / 13.39 | 51.7 % |
| `mean_depth` | 4.06 / 4.02 / 4.17 | 64.1 % | 7.00 / 7.53 / 8.15 | **89.0 %** |
| `mean_op_entropy` | 1.52 / 1.52 / 1.54 | 62.0 % | 1.90 / 2.16 / 2.26 | **93.3 %** |
| `mean_shared` | 1.53 / 1.51 / 1.59 | 72.4 % | 1.88 / 2.15 / 2.40 | **91.5 %** |

The predicted ordering baseline < hash < isalsr holds on depth, entropy and sharing for
Bingo, and the isalsr arm leads on all four descriptors for both methods. The telemetry is
recording what T19 was built to record. Formal testing belongs to the analysis stage.

---

## 7. COST (checks F17–F18)

### 7.1 Wall clock by (method, suite)

| method | suite | n | mean (s) | median (s) | max (s) | frac ≥ 43,000 s |
|---|---|---:|---:|---:|---:|---:|
| bingo | cherrypicked | 900 | 12,850 | 13,305 | 36,658 | 0.00 |
| bingo | feynman | 765 | 1,072 | 11 | 37,042 | 0.00 |
| bingo | feynman_remainder | 540 | 7,129 | 761 | 31,091 | 0.00 |
| bingo | hard | 900 | 13,143 | 12,865 | 42,337 | 0.00 |
| bingo | nguyen | 1,048 | 913 | 100 | 16,462 | 0.00 |
| bingo | roundoff | 720 | 9,805 | 9,138 | 30,618 | 0.00 |
| bingo | strogatz | 1,260 | 1,149 | 323 | 11,897 | 0.00 |
| udfs | cherrypicked | 900 | 43,200 | 43,200 | 43,202 | **1.00** |
| udfs | feynman | 466 | 6,452 | 656 | 43,202 | 0.10 |
| udfs | feynman_remainder | 540 | 41,906 | 43,200 | 43,202 | **0.90** |
| udfs | hard | 900 | 43,200 | 43,200 | 43,204 | **1.00** |
| udfs | nguyen | 1,080 | 35,784 | 43,200 | 43,201 | 0.80 |
| udfs | roundoff | 720 | 43,201 | 43,200 | 43,203 | **1.00** |
| udfs | strogatz | 1,260 | 38,372 | 43,200 | 43,201 | 0.80 |

Expected pattern confirmed exactly: UDFS pins the 43,200 s cap on unsolved problems, Bingo
never reaches it (max 42,337 s) because it stops on `max_evals` or convergence. No cell
exceeds the cap by more than 3.7 s (max 43,203.7 s), so no runaway job. Peak RSS 9.42 GB
(one Bingo baseline cell); mean 0.53 GB — no memory pressure.

### 7.2 Canonicalisation overhead as a fraction of wall clock

| method | arm | n | mean canon (s) | mean % | median % | p95 % | max % |
|---|---|---:|---:|---:|---:|---:|---:|
| bingo | baseline | 2,060 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 |
| bingo | hash | 2,054 | 159.52 | 3.805 | 3.811 | 5.979 | 6.826 |
| bingo | isalsr | 2,019 | 330.30 | **6.583** | 6.349 | 9.631 | 17.177 |
| udfs | baseline | 1,951 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 |
| udfs | hash | 1,954 | 2.98 | 0.008 | 0.006 | 0.017 | 0.040 |
| udfs | isalsr | 1,961 | 12.48 | **0.041** | 0.021 | 0.082 | 0.803 |

Bingo isalsr overhead is 6.58 % mean — **single-digit percent on the native engine, as
expected**, and a large improvement on the 51 % recorded by the pre-native campaign. Worst
suites are `nguyen` (8.58 %) and `strogatz` (8.40 %), i.e. the short runs where the fixed
cost amortises over less work; still under 10 %. UDFS overhead is negligible at 0.04 %.

### 7.3 `total_dags_explored` — comparable between hash and isalsr, NOT against baseline

| method | arm | mean | median | max |
|---|---|---:|---:|---:|
| bingo | baseline | 41,738,327 | 16,615,865 | 100,145,738 |
| bingo | hash | 3,914,052 | 1,172,840 | 18,322,070 |
| bingo | isalsr | 3,883,082 | 1,182,860 | 17,755,940 |
| udfs | baseline | 85,801 | 47,634 | 804,993 |
| udfs | hash | 72,762 | 48,658 | 592,342 |
| udfs | isalsr | 113,247 | 75,545 | 1,016,217 |

Bingo baseline records a median 8.41× more DAGs than its hash arm and reaches the
`max_evals = 1×10⁸` cap on 33.5 % of cells, while no hash/isalsr cell exceeds 1.83×10⁷.
Throughput reads 11,662 DAG/s on baseline vs 1,082 DAG/s on hash — a 10.8× gap that
**cannot** be produced by a 3.8 % canonicalisation cost. This is a counter-semantics
difference (baseline's counter tracks the host's raw fitness-evaluation stream; the dedup
arms count individuals passing the dedup hook), not a real slowdown.

Consequences, and they are narrow:
- ρ = total/unique is computed **within** an arm, so every ρ in §4 is internally valid.
- `hash` and `isalsr` share the same instrumentation — their `total_dags_explored` median
  ratio is exactly **1.0000** on Bingo — so the hash-vs-isalsr ρ contrast, which is the
  C1.7 criterion and the paper's claim, is apples-to-apples.
- **Any statement comparing raw `total_dags_explored` or DAG-throughput between `baseline`
  and a dedup arm is invalid** and must not be made.
- UDFS is unaffected: baseline and hash throughput are identical (1.47 vs 1.47 DAG/s) and
  isalsr's 2.36 DAG/s is a genuine effect — dedup frees budget, letting isalsr explore
  1.39× more DAGs inside the same 12 h cap. That is the mechanism behind the `udfs/hard`
  and `udfs/strogatz` quality edge in §5.2.

---

## 8. ANOMALIES, RANKED BY SEVERITY

**A1 — Raw-mean R² is unusable on 4 problems; it will corrupt the CPDT. (High, analysis-stage)**
18 of 3,970 triples carry r2_test < −1, min −423.31. They invert the arm ordering in both
directions: they manufacture a −0.216 UDFS isalsr *deficit* (one cell,
`udfs/strogatz/strogatz_lv1/seed_24/isalsr`, contributes −422.60 of the −421.86 total) and a
+1.065 Bingo/hard isalsr *surplus* (one cell, `bingo/hard/pagie_1/seed_20/baseline`, at
−319.34). The CPDT averages R² over seeds per problem before testing, so a single such seed
propagates straight into that problem's δ. **Fix at analysis time** — clip R² at 0 (SRBench
convention) or use a median/trimmed statistic, and state the choice. Not a data defect;
`r2_train` for the offending cells is normal and near-identical across arms (0.9134 /
0.9134 / 0.9135), confirming the models fit and only test-set extrapolation exploded.

**A2 — `total_dags_explored` is not cross-arm comparable against `baseline` on Bingo. (Medium, reporting-stage)**
Median ratio 8.41×, throughput ratio 10.8×, incompatible with the measured 3.8 % overhead.
Counter semantics differ. ρ is unaffected (computed within-arm); hash-vs-isalsr is
unaffected (ratio exactly 1.0000). Do not report baseline-vs-isalsr DAG counts or
throughput. Detail in §7.3.

**A3 — `cache_hit_rate`, `cache_hits`, `cache_misses`, `estimated_time_saved_s` are identically zero on all 11,999 cells. (Medium, reporting-stage)**
Including every `isalsr` cell, where `redundancy_rate` is 0.44. The fields are simply not
populated in C2. Reporting "0 % cache hit rate" would be actively misleading — the dedup
effect is carried by `redundancy_rate` and `empirical_reduction_factor`, which are correct.
Suppress these four fields from all outputs.

**A4 — 54 null `solution_recovered`, all on Bingo. (Low)**
0.45 % of cells, scattered over all suites and arms (baseline 26 / hash 17 / isalsr 11), no
block above 4. Consistent with sympy equivalence-check timeouts. Must be handled as missing;
coercing to False would understate recovery and would do so unevenly across arms.

**A5 — 210 UDFS cells have no complexity telemetry. (Low, expected)**
`udfs` on `i.25.13`, `nguyen_8`, `i.12.1`. `complexity_n_sampled = 0` because the search
finished in ~17 candidates (r2_test = 1.0, ~12 s) before the every-31st stream sampler
fired. Balanced across arms, so paired contrasts drop whole triples cleanly. T19 simply has
no data for UDFS on those three problems.

**A6 — `bingo/feynman/i.48.20` does not exist on disk at all. (Informational — recovery input)**
Not merely depleted: the problem directory is absent, so Bingo has 69 problems to UDFS's 70.
90 cells. Worth confirming the recovery pass's problem list is built from the union of 70 and
not from `ls bingo/feynman/`, or this block will be silently skipped again.

**Non-anomalies, checked and cleared:** absent `convergence_log.npz` on all UDFS cells (no
generation loop); absent `fallback_ledger.json` on all baseline cells (`ledger_enabled` null
on baseline, True on all 7,988 dedup cells); null `complexity_unique_*` on baseline (no
cache — documented expected); `metadata.problem` case differences vs directory slugs.

---

## 9. WHAT I COULD NOT CONCLUDE

**Safe to read now — 100 % complete triples for both methods:**
`cherrypicked` (600), `feynman_remainder` (360), `hard` (600), `roundoff` (480),
`strogatz` (840). 2,880 complete triples, no imbalance. Any per-suite statistic on these is
trustworthy today, subject to the R² robustness caveat in A1.

**Read with care:** `nguyen` at 95.6 % complete (688/720). Bingo's `nguyen_8` has 30
baseline+hash pairs with no isalsr partner and `nguyen_7` has 2 more. Any paired Bingo/nguyen
statistic silently drops `nguyen_8` entirely, which removes a whole problem from a
per-problem test — the CPDT's N would fall from the nominal count without that being visible
in the output. Do not compute Bingo/nguyen CPDT until recovery.

**Not trustworthy yet:** `feynman`, both methods. UDFS is at 50.3 % complete triples with
six problems depleted to 3–12 seeds; per-problem seed means over 3 seeds have standard
errors several times those over 30, and the CPDT weights every problem equally regardless of
how many seeds backed its δ, so `i.10.7`, `i.48.20`, `ii.3.24`, `i.12.4`, `i.6.20a` and
`i.12.1` would each enter with the same weight as a fully-populated problem. Bingo/feynman is
at 83.7 % and additionally missing `i.48.20` entirely. **Any pooled cross-problem statistic
(CPDT, Friedman/Nemenyi, the N=70 headline) is premature** — it would be computed over a
problem set whose membership depends on which cells happened to finish.

**Checks I could not run:**
- Cross-arm agreement of the *best expression found* — I compared metrics, not
  `best_expression.canonical_string`. Verifying that the arms converge to equivalent
  expressions where R² ties would be a stronger bug check than the metric comparison, and it
  is not covered here.
- `trajectory.csv` and `convergence_log.npz` contents were sampled only (9 Bingo cells) to
  resolve the §7.3 counter question. Convergence-curve integrity across 11,999 cells is
  unverified.
- `fallback_ledger.json` contents were not inspected; only presence and the
  `ledger_enabled` flag were checked.
- Whether the 7 Bingo ρ(isalsr) ≤ ρ(hash) violations share a root cause — 7 cells across 3
  suites, below the noise floor for the C1.7 criterion, so I did not open them.

---

## 10. BOTTOM LINE FOR THE RECOVERY PASS

Proceed. 601 cells across 26 blocks, listed in §2.3. Highest yield first:

1. `bingo/nguyen/nguyen_8/isalsr` — 30 cells, converts 30 partial triples to complete.
2. `bingo/feynman/i.48.20` all three arms — 90 cells, restores a problem missing from Bingo entirely (build the problem list from the 70-problem union, not from disk).
3. `udfs/feynman` six problems × 3 arms — 434 cells, the bulk; lifts UDFS/feynman from 50.3 % to 100 % complete triples and unblocks every pooled statistic.
4. `bingo/feynman/i.10.7` all three arms (45 cells) and `bingo/nguyen/nguyen_7/isalsr` (2 cells).

No provenance, integrity or correctness finding stands in the way. The re-submitted cells
must carry `engine=native`, `build_hash=298fc1188bf1b051`, `git_describe=campaign/c2` to
remain poolable with the 11,999 already on disk.
