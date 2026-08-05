# Fairness and methodology audit — C2 experimental setup

**Date**: 2026-08-04
**Branch**: `feature/experiment-fairness-audit` (worktree `.claude/worktrees/fairness-audit`, base `b60798e`)
**Auditor**: Claude (independent of the T17/T02 certification agent on `feature/cpp-core-port`)
**Mandate (Mario)**: check the whole experimental setup — metrics, timings, tests,
compared models — for anything that (a) over-favours the competitor arms
(`baseline`, `hash`) against `isalsr`, (b) unfairly favours `isalsr` in a way a
reviewer could attack, (c) is a methodological fallacy or code bug. Fix what
needs fixing, minimally, under the review premise. This audit does not
re-litigate settled decisions in `EXECUTION-PLAN.md` §0 unless a fairness defect
forces it.

---

## 1. Scope and method

Audited: the C2 campaign as specified in `EXECUTION-PLAN.md` (read in full) and
`T17-HANDOFF.md`, and the code implementing it at `b60798e`. Four parallel
read-only investigations plus direct checks by the auditor:

| Lane | What was inspected |
|---|---|
| A — timing/budget | what `max_time` bounds per host×arm; every cost-field formula; where ledger/shadow/conversion time lands; `estimated_time_saved_s` |
| B — hash arm | live serialisation choice; adapter-renumbering bias; `shadow_distinct_host_native`; dedup-benefit symmetry hash↔isalsr; per-host key rules |
| C — statistics | CPDT direction and tie handling; three-arm readiness (A8); NaN policy; W/T/L and bold logic; ρ aggregation; effect sizes; Friedman/CD machinery |
| D — dedup semantics | duplicate-hit path per host×arm; constants vs structure in the key; counter-population parity; B12 clones; collision handling; `dedup_enabled` |
| Direct | all 22 configs; `slurm/c2_smoke/{worker,launcher}.sh`; manuscript claims (`computational_experiments.tex`, `results.tex`); Bingo/UDFS host-native key code |

Direction convention: **favors-baseline/hash** = competitor advantage (Mario's
question); **favors-isalsr** = a bias a reviewer could call unfair in our favour
(must be fixed or disclosed).

---

## 2. Verified sound — checked, no change needed

1. **Config parity across arms.** All 14 production configs: `max_time: 43200`,
   `n_seeds: 20`, identical host hyperparameters per method across suites
   (Bingo pop 500 / stack 32 / uniform 10-operator set per A4b; UDFS
   `processes: 1` everywhere, so the spawn-worker patch bypass stays dormant).
   The `isalsr:` block carries only canonicaliser settings; the arm is selected
   exclusively by `--variants`.
2. **Harness parity.** `slurm/c2_smoke/worker.sh:161-163`: one invocation shape
   for all arms; `--ledger` passed to every task (inert on baseline); one shared
   `MAX_TIME`. `launcher.sh`: `--time`/`--constraint` shared; only `--mem`
   differs (scheduling, not compute speed).
3. **Budget window honesty.** `max_time` is host-enforced (UDFS
   `dag_search.py:1930/:1223`; Bingo `evolve_until_convergence`) and encloses
   *all* wrapper work — no dedup-arm computation happens outside the budget, no
   warm caches, no pre-computed canonicalisation. Post-search SymPy/predict is
   outside the reported wall clock in all four runner classes (symmetric).
4. **Baseline purity.** Baseline pays zero canonicalisation
   (`canonicalization_runtime_s` hardcoded 0; C1.8 certified 420/420); atlas is
   dead in C2 (no `--atlas-dir`), so `cache_*`, `canonicalization_precomputed_s`
   and `estimated_time_saved_s` are identically 0 — no hidden benefit anywhere.
5. **`estimated_time_saved_s`** formula uses canonicalisation time only (no LM
   term) — no savings over-estimate is possible; dead in C2 anyway.
6. **NaN policy (T08 halves a and b).** NaN can never be bolded/marked better
   (`generate_tables.py:664-675` + `test_table_nan_integrity.py`); pairwise
   deletion with the true per-metric `n` (`aggregation.py:236-247,299`). Runtime
   scoring (R²=0 / NRMSE=1 on non-finite test predictions, `metrics.py:102-179`)
   is arm-independent — both translators call the same functions; variant never
   enters.
7. **Effect sizes.** Paired d_z = mean(δ)/sd(δ) on the δ vector, percentile
   bootstrap, fixed seed — correct construction.
8. **UDFS dedup interception** is total in the C2 configuration (sequential
   branch resolves the patched module global; `processes: 1` in all 8 configs);
   hash arm uses the same interception point.
9. **Hash arm is the honest naive competitor.** The live arm keys on the
   **host-native** representation (`HASH_ARM_KEY_MODE = "host_native"`,
   `udfs/isalsr_runner.py:61`, `bingo/isalsr_runner.py:73`) — Bingo's own
   `command_array` row order, UDFS's own `node_dict` order — *not* on the
   adapter's renumbered output. The steel-manned adapter-order rung
   (ρ_hash 1.38 vs 1.02 on UDFS) is recorded on the same stream via the shadow
   counters. Dedup benefit machinery is one shared code path for hash and isalsr
   arms; only `representation_string` differs.
10. **Per-host key rules are one principle, not an inconsistency.** Bingo's key
    drops dead rows and masks unread `param2`; UDFS's key keeps unused
    terminals. Both follow the soundness constraint *the hash key must not
    distinguish what the adapter/canonical map identifies, nor identify what it
    distinguishes* (dead rows never reach the adapter; unused UDFS terminals
    become real VAR/CONST nodes, `udfs/isalsr_runner.py:83-108` docstring). This
    is what guarantees `ρ_hash ≤ ρ_isalsr` (C1.7, D1.5). Needs one disclosure
    sentence in the paper, no code change.
11. **One-sided CPDT is openly declared** in the manuscript ("The reported
    one-sided p-value is computed in the direction of …",
    `computational_experiments.tex:303-306`), directions are consistent across
    metrics (`CPDT_METRIC_ALTERNATIVES`), and the D2 extension's outcome-blind,
    pre-registered selection rule is stated with its commit provenance.
12. **C3/C4 controls** (wrapper perturbation; cross-arm data identity) exist and
    are driven by T17; C4's node-family finding is already resolved by the B6
    `--constraint=sr` pin.

---

## 3. Findings

Severity: 🔴 must fix before C2 analysis · 🟠 must disclose / route · 🟢 note.

### F-1 🔴 CPDT Wilcoxon silently drops ties (favors **isalsr**) — FIXED
`aggregation.py:532-533` called `scipy.stats.wilcoxon` with the default
`zero_method="wilcox"`, which removes zero differences from the test — while
the project's own W/T/L rationale treats ties as evidence of no difference.
With many ties and positively-leaning non-ties this inflates significance
(measured demo: p = 0.0289 vs 0.1055 with tie-splitting at 60/70 ties). A
reviewer replicating the CPDT with default scipy would get *our* number, but
one who reads Pratt (1959, JASA 54:655–667) or Demšar (2006, JMLR 7:1–30,
ties split evenly) would object. **Fix**: `zero_method="zsplit"` (zeros' ranks
split between the positive and negative sums) — the conservative choice.

**Impact quantified on C1's own archived δ's** (N = 50,
`wl_subtree_unified/analysis/cross_problem_dominance_*_all.json`, recomputed
2026-08-04 with the exact new pipeline — snap at 1e-6, Shapiro on the snapped
vector, zsplit):

| Method | Metric | W/T/L | old p₁ₛ | corrected p₁ₛ | verdict |
|---|---|---|---|---|---|
| UDFS | R²_test | 28/16/6 | 5.87×10⁻⁵ | 6.58×10⁻⁵ | unchanged (***) |
| UDFS | R²_train | 33/15/2 | 2.93×10⁻⁷ | 1.39×10⁻⁷ | unchanged (***) |
| UDFS | NRMSE_test | 5/15/30 | 2.69×10⁻⁶ | 3.42×10⁻⁶ | unchanged (***) |
| Bingo | R²_test | 15/32/3 | 4.42×10⁻⁴ | **6.06×10⁻³** | *** → ** (13.7×) |
| Bingo | R²_train | 16/31/3 | 3.14×10⁻⁴ | 3.47×10⁻³ | *** → ** |
| Bingo | NRMSE_test | 5/21/24 | 9.91×10⁻⁵ | 3.14×10⁻⁴ | unchanged (***) |

**No significance claim flips at α = 0.05.** Bingo's R² rows lose one star.
If any C1 CPDT p-value is quoted in the revision (continuity table, response
letter), it must be the corrected one, with the policy change stated. C2's
analysis uses the corrected policy from the start.

### F-2 🔴 Tie threshold applied to the display, not the test — FIXED
`_CPDT_TIE_THRESHOLD = 1e-6` governed only the displayed W/T/L
(`aggregation.py:509-511`); the test consumed raw floating-point δ's, so a
"tie" in the table could enter the test as a win or loss decided by fp noise
(demo: p 0.0039 ↔ 0.241 depending on noise sign). **Fix**: snap |δ| ≤ 1e-6 to
0 before testing; W/T/L and the tested vector are now provably the same
partition. Effect sizes stay on raw δ (estimation, not decision). Same
`zsplit` + all-zeros guard applied to the per-problem supplementary test in
`generate_tables.py:_paired_test`, whose `except → p=1.0` also became
`p=NaN` (its own <3-seeds branch already documents why 1.0 is wrong).

### F-3 🔴 ρ "paired test" vs the baseline is tautological (favors **isalsr** presentation) — ROUTE to A8 + paper
Baseline `empirical_reduction_factor` is **1.0 by construction**
(`bingo/runner.py:428`: `n_unique_canonical = total_evals`; §11.1 2026-08-03
already records this). The CPDT on ρ/redundancy therefore tests
"ρ_isalsr > 1", which is arithmetically guaranteed whenever any duplicate
exists — the reported p = 2.7×10⁻²² (UDFS) is a one-sample triviality dressed
as a comparison. **Required**: in the three-arm rework (A8), the inferential ρ
contrast is **isalsr vs hash** (both measured); ρ vs baseline is reported
descriptively (mean ± std), without a p-value. The manuscript's
Table `cpdt_summary` ρ rows must change accordingly.

### F-4 🔴 CD-diagram ranks inverted for lower-is-better metrics — FIXED
`compute_critical_difference` ranks higher values as better
(`statistical_tests.py:216`); `run_cross_method` (`analyze.py:207`) passes
`nrmse_test` and `wall_clock_total_s` unnegated → the published
critical-difference figure orients those axes backwards (Friedman χ² is
rank-symmetric, so only ranks/cliques are affected — but ranks are what the
figure shows). **Fix**: `higher_is_better` parameter, negation for the two
lower-is-better metrics, recorded in the output JSON.

### F-5 🟠 Cross-method Friedman: non-finite means and column misalignment — FIXED
`load_cross_method_results` (`cross_method.py:71-72`) used `np.mean` (NaN
poisons a problem silently) and skipped problems missing for one variant,
which can misalign the paired columns. **Fix**: finite-only means, alignment
on problem names (sorted intersection), complete-case row drop with the
dropped problems named in the output (Friedman requires complete blocks,
Demšar 2006). Plus `ranks` dtype float (`statistical_tests.py:214` truncated
.5 average ranks on integer matrices).

### F-6 🟠 SymPy failure accounting asymmetric — FIXED
`solution_recovered` returned **False** and `jaccard_index` **0.0** on generic
SymPy exceptions, while both return **None (excluded)** on timeout — and the
docstrings' own argument (failures land preferentially on the arm with larger
expressions) applies equally to exceptions. **Fix**: exceptions → None
(undetermined), logged at warning; consistent with the timeout policy.

### F-7 🔴 Untimed wrapper work booked as "search" (both directions) — FIXED
Adapter conversion (host→LabeledDAG), the T06 ledger records and the T04
shadow sketches all run **inside** the wall clock but **outside**
`canon_time_total` (`udfs/isalsr_runner.py:334,355,371`;
`bingo/isalsr_runner.py:502,517,533` — the shadow comment says "sits outside
every timer"). Since `T_search = wall − canon` and
`OH% = canon/wall` (`translator.py:104-121`, `analyze.py:384`):
- the reported **overhead % understates** the representation layer's true
  cost (conversion is genuinely part of the method) — *favors isalsr*;
- the reported **T_search for dedup arms is inflated** by instrumentation, so
  `S = T_search^bl / T_search^IS` is biased **against isalsr** — part of C1's
  Bingo S = 0.93 is literally untimed wrapper work in the denominator.
**Fix**: per-candidate timers for conversion and shadow;
new RunLog fields `conversion_time_s`, `shadow_time_s` (additive, tolerant
loading for legacy artefacts);
`wall_clock_search_only_s = wall − canon − conversion − shadow`;
`overhead_time_s = canon + conversion` (shadow excluded from both — it is
audit instrumentation, not method cost, and is now visible on its own).
**Consequence to carry**: T01 AC-6's "S cannot move" statement was derived
under the old formula; with the corrected derivation S will shift (upward for
isalsr). The AC-6 *mechanism* (dS/dT_canon = 0) survives; the number does not.
The manuscript's `T_total = T_search + T_canon` decomposition
(`computational_experiments.tex:185-195`) must name the conversion term.

### F-8 🟠 Shadow sketches: isalsr-arm-only cost inside a fixed budget — MEASURE, then Mario decides
`shadow_hash` defaults ON **only** for the isalsr arm
(`udfs/isalsr_runner.py:496`, `bingo/isalsr_runner.py:721`; no config or SLURM
file overrides it): 4 serialisations + 2 topological sorts + 4 HLL updates per
candidate, pure Python, previously unmeasured in time (T04 AC-10 measured RSS
only). Every second spent there is search budget the baseline and hash arms
keep — an anti-isalsr bias of unknown size, on top of the F-7 misattribution.
The sketches are *worth keeping* (they carry the competitor's steel-man number
and the F-13 clone decomposition), but the cost must be known.
**Action**: `shadow_time_s` (F-7) makes it measurable per run.
**Measured (local smoke scale, 1 seed/host, small k)**: per-candidate shadow
42.9 µs (UDFS) / 53.1 µs (Bingo) — *more than canonicalisation itself*
(28.7 / 18.7 µs); shadow share of wall clock 0.55 % (UDFS) / 2.12 % (Bingo).
Smoke-scale k is small; the ratios will not transfer to production directly.
**Decision rule proposed**: keep shadow ON through Stage D; read
`shadow_time_s / wall_clock_total_s` from the 12 h D1 cells (now recorded per
run); if the Bingo share stays above ≈1 %, either accept with explicit
disclosure (the sketches buy the competitor's steel-man number and the F-12
clone decomposition) or restrict sketches to the Stage-D trace cell and run
C2's isalsr arm with `shadow_hash: false`. **Mario's call at Stage F.**

### F-9 🟢 T06 ledger cost asymmetry — DISCLOSE
The fallback ledger runs only on dedup arms, inside their budget: UDFS 0.04 %,
Bingo 0.22 % (B9, jobs 1751997/8). Direction anti-isalsr, magnitude bounded
and measured. One sentence in the paper's integration paragraph suffices.

### F-10 🟠 Duplicates are culled, not cache-served — manuscript wording + disclosure
In the C2 code path there is **no fitness cache**: a detected duplicate is
assigned `np.inf` fitness (UDFS `(zeros, inf)`, `udfs/isalsr_runner.py:390-396`;
Bingo additionally `genetic_age = 10,000,000`, `bingo/isalsr_runner.py:622-628`),
so selection eliminates it — equivalent to rejecting duplicate offspring. The
`fitness_cache` exists only behind `enforce_population_dedup`, which no C2
config sets. Consequences:
1. The manuscript says a duplicate's "evaluation is skipped" and the mechanism
   is "transparent … no hyperparameters, search operators, or termination
   criteria are modified" (`computational_experiments.tex:33-45`) — literally
   true, but a reviewer reading the runner will see the worst-fitness
   assignment and call "transparent" an overstatement. **Add one sentence**:
   *"A detected duplicate is assigned the worst possible fitness (and, under
   Bingo's age-fitness selection, a maximal age), so selection removes it from
   the population — the standard duplicate-rejection policy; no cached fitness
   is reused."*
2. Denied LM restarts: the baseline re-optimises constants every time it
   re-encounters a structure; the dedup arms optimise each structure once.
   This is **conservative against isalsr** on constant-bottlenecked problems
   (consistent with Keijzer-6/Korns-12 showing no gain) and should be stated
   as such — it strengthens, not weakens, the claim.
3. `EXECUTION-PLAN.md` §0.2's rationale ("the runners cache canon_hash →
   fitness without ever calling evaluate_dag") describes the
   `enforce_population_dedup` branch, not the C2 path — the *conclusion*
   (per-candidate fitness values are alphabet-independent; nothing needs
   re-deriving since C2 re-runs everything) stands, but the stated mechanism
   is wrong and should be corrected in the decision log.
4. `n_penalised_per_gen` (computed, previously discarded) is now persisted
   (mean/max) so the effective-population impact is reportable.

### F-11 🟠 Bingo cross-arm `total_dags_explored` is apples-to-oranges (favors **baseline**) — DO NOT REPORT cross-arm
Bingo baseline counts fitness *invocations* (`eval_count`, LM-inflated
3.3–4.1×, `bingo/runner.py:427`); dedup arms count candidate DAGs
(`dedup.n_total`). The 2026-08-03/04 fixes corrected trajectory rows only;
`run_log.search_space.total_dags_explored` is still paired across arms via
`METRIC_EXTRACTORS` (`aggregation.py:100`). Any cross-arm "DAGs explored" or
throughput comparison flatters the baseline ~3–4×. ρ is intra-arm and
unaffected; hash-vs-isalsr is clean (same counter). **Not removed from
`METRIC_EXTRACTORS` here** — that would break the Stage C certifier's
14-metric-row count mid-flight on the other branch. **Required**: A8 rework
marks the metric non-comparable across Bingo arms (skip or annotate its
paired stats); the paper must not cite cross-arm evaluation counts for Bingo.

### F-12 🟠 Bingo ρ includes verbatim clones the host would not have re-evaluated — DECOMPOSE in reporting
Stock Bingo skips `fit_set=True` clones (~36 % of offspring); the dedup arms
force all offspring through the hook (B12) and count clone hits as duplicates.
So part of Bingo's "r = 45 % of evaluations eliminated" is elimination the
baseline gets for free from clone-skipping — a reviewer-attackable
over-credit in the *phrasing* "evaluations eliminated". The instrumentation
already records the fix: `shadow_distinct_host_native` counts distinct
host-native genotypes, so `n_total − distinct_host_native` ≈ verbatim-copy
duplicates and the remainder are genuine structural rediscoveries (IsalSR's
contribution beyond the host's own skip). **Required**: report Bingo's ρ
decomposed into verbatim-copy vs structural duplicates (data exists in C2 by
construction); phrase r as "candidate evaluations avoided by structural
deduplication, of which X pp are verbatim copies Bingo's clone-skipping
already avoids". UDFS is symmetric (same hook both arms) and unaffected.
This is also a second independent reason to keep the shadow sketches (F-8).

### F-13 🟠 A8 three-arm machinery confirmed absent — blocker already known, requirements added
`analyze.py:108,148`, `cross_method.py:60,98`, `generate_tables.py:83` all
hardcode `["baseline","isalsr"]`; zero `hash` references in the analyzer or
figures; no three-arm test exists. Already the plan's open A8. This audit adds
to A8's requirements: (i) Holm across exactly 3 contrasts; (ii) F-3's ρ
handling (isalsr-vs-hash inferential, vs-baseline descriptive); (iii) F-11's
non-comparability annotation; (iv) the F-1/F-2 tie policy applies to all three
contrasts.

### F-14 🟢 Conservative-substitution sensitivity check does not exist
T08's §6.4 policy promises it; nothing implements it (no code, no test).
Stage E's E3 gate would catch this. Post-C1.3 runtime scoring makes NaN means
nearly impossible, so the check is a formality — but E3 asserts it runs.
**Route**: implement with the A8/Stage-E work, not here (avoids colliding with
the T08 lane).

### F-15 🟢 Suite enrichment and the pooled CPDT — disclosed; add a sensitivity row
The 28-problem extension is hypothesis-enriched ("targeting regimes where
structural search dominates" — criterion (iii)), openly declared in the
manuscript, with no post-hoc filtering, and the D2 extension is
outcome-blind and pre-registered. The residual exposure is only that the
pooled one-sided CPDT over an enriched suite can be read as an unconditional
claim. **Recommendation** (manuscript lane): report per-tier CPDT and a
leave-tier-out sensitivity (pooled minus the enriched tier) in the
supplement — the per-benchmark machinery already computes it, and C2's root
keeps suites separate by construction. No code change.

### F-16 🟢 Minor notes
- `hash(str)` dedup keys are SipHash-salted per process — fine within a run;
  would silently break any future cross-process atlas mixing (atlas is None in
  C2). Note for T04 posterity.
- `canonicalization_precomputed_s ⊂ canonicalization_runtime_s` (subset, not
  disjoint) — any consumer summing the two double-counts; both are 0 in C2.
- `_paired_test` Cohen's d returns 0.0 when sd ≤ 1e-10 even if the mean is
  non-zero (uniform shift reads as no effect) — conservative, rare, left as
  is, recorded here.
- Hash-arm clone-bypass (B12) is inherited by the hash arm through subclassing
  but had no parametrised test — covered by the new cost-attribution tests
  only incidentally; a one-line parametrisation of
  `test_dedup_clone_bypass.py` remains nice-to-have.
- `analyze.py:102-105` reuses cached `paired_stats.json` without revalidation
  — harmless for a fresh C2 root; do not run the analyzer twice into the same
  root across code versions.

---

## 4. Changes made on this branch

### 4.1 Cost attribution (F-7 / F-8 / F-10.4) — LANDED

Files: `experiments/models/schemas.py` (`TimeResults` + `conversion_time_s`,
`shadow_time_s`; `SearchSpaceResults` + `penalised_in_population_{mean,max}`);
`experiments/models/{udfs,bingo}/isalsr_runner.py` (per-candidate
`perf_counter` accumulation around the adapter conversion and inside
`record_shadow` — after the disabled-guard, so the field is exactly 0.0 when
shadow is off); `{udfs,bingo}/runner.py` (raw-result fields);
`{udfs,bingo}/translator.py` (new formulas); `experiments/scripts/c2_certify.py`
(`RUN_LOG_FIELD_SPEC` 56 → **60** fields); tests.

New formulas, both hosts:
`wall_clock_search_only_s = max(0, wall − canon − conversion − shadow)`;
`overhead_time_s = canon + conversion`; shadow excluded from overhead
(instrumentation, reported on its own).

Tests: new `tests/unit/test_cost_attribution.py` (19 tests, shown red → green);
targeted suites 121 passed; **full `tests/unit`: 6,422 passed, 5 skipped**;
`ruff` clean on touched files; `mypy src/isalsr/` clean.

Measured consequence (smoke scale): the previously reported overhead % was
understated **1.57× (UDFS) / 2.43× (Bingo)**; search-only time was inflated by
0.55 % / 2.12 % of wall clock. See F-8 for the shadow decision rule.

**Operational consequences to carry at merge time:**
1. The certified RunLog field list grows 56 → 60. Stage C artefacts produced
   before this branch (smoke v1–v3) will fail C1.2 on the four new keys —
   after merging, either re-run the Stage C wave (≈32–40 min at `%24`/`sr`) or
   mark the four fields non-blocking for pre-merge roots. Under the
   one-commit rule (§5.1) the campaign relaunches from the merged commit
   anyway, so a fresh Stage C certification is the clean path.
2. `wall_clock_search_only_s` and `overhead_time_s` change *meaning*. D1.7's
   sanity expectation ("Bingo ≈7.4 % under the C++ engine") was canon-only;
   with conversion included the honest number will be higher. Do not read
   that as a regression.
3. T01 AC-6's "S cannot move" number was derived under the old formula (see
   F-7). Mechanism survives; the value of S will shift upward for isalsr.
4. Merge before launch or not at all — never mid-campaign (config_sha256 /
   one-commit discipline).

### 4.2 Statistics fixes (F-1/F-2/F-4/F-5/F-6) — LANDED

| Fix | Files | Evidence |
|---|---|---|
| F-1/F-2 tie policy | `analyzer/aggregation.py` (`d_test` snap; W/T/L, Shapiro, all-zeros guard, t and Wilcoxon all on `d_test`; `zero_method="zsplit"` on both Wilcoxon calls; effect sizes stay on raw δ; tie-policy docstring with Pratt 1959 / Demšar 2006); `figures/models/generate_tables.py` `_paired_test` (all-zero guard → (0.0, 1.0); zsplit; except → p = NaN) | regression demo in tests: N=70 with 60 noise-ties: old p 0.00535 → new 0.20217 |
| F-4/F-5 Friedman/CD | `analyzer/cross_method.py` (name-keyed loading, finite-only means, sorted-intersection alignment, complete-case row drop with dropped problems named in output; `higher_is_better` param, negation, recorded in JSON); `analyze.py` (`_LOWER_IS_BETTER = {nrmse_test, wall_clock_total_s}`); `analyzer/statistical_tests.py` (float ranks) | synthetic tests: rank order flips with direction while χ² identical; fractional ranks on int input; dropped problems named |
| F-6 SymPy symmetry | `analyzer/metrics.py` (generic exception → None for both `solution_recovered` and `jaccard_index`, warning + exc_info) | old test `test_error_is_false_not_none` shown red, renamed/inverted |

New `tests/unit/test_stats_fairness_fixes.py` (20 tests); two existing tests
migrated (they asserted the old, defective behaviour). Full `tests/unit`:
6,422 passed, 5 skipped, 0 failed (via the §4.3 shim). `ruff` clean in all
edited regions; `mypy src/isalsr/` untouched and clean.

Residual note from the implementer: `jaccard_index` scores a non-`sympy.Basic`
input 0.0 through its normal path (the isinstance guard, not the exception
path) while `solution_recovered` now returns None for the same input — a
smaller instance of the same asymmetry, left unchanged (out of the minimal
brief), recorded here for the A8/Stage-E owner.

### 4.3 ⚠ Worktree test-evidence hazard (discovered during 4.1)

The editable install's `ScikitBuildRedirectingFinder` maps the top-level
`experiments` and `benchmarks` packages to the **main checkout**
(`/home/mpascual/research/code/IsalSR`), so `pytest` run from *any worktree*
silently imports and tests the main repo's code, not the worktree's. All test
evidence in this audit was produced through a shim that repoints those two
packages (scratchpad `run_worktree_pytest.py`). **Any agent testing from a
worktree in this repo must do the same** — this belongs in
`docs/tasks/lessons.md` at merge time.

An intermittent failure cluster observed while the two implementers ran
concurrently (17 → 1 → 0 failures across successive full-suite runs) did not
reproduce once the tree was stable: two consecutive shimmed full-suite runs by
the auditor returned **6,422 passed, 5 skipped, 0 failed — twice, identical
tallies**. Cause attributed to testing during concurrent edits, plus
subprocess-spawning tests resolving the main repo (the meta-path redirect
cannot be shimmed for child processes). Post-merge CI on the main checkout is
unaffected by construction.

## 5. Verdict

**The C2 design's core fairness properties hold.** Identical configs, budgets,
seeds and data across arms; a host-enforced budget window that encloses all
wrapper work with no hidden pre-computation; an honestly-naive hash competitor
with its steel-man recorded on the same stream; arm-independent metric scoring;
a correctly-constructed paired effect size; and openly declared one-sided
testing and suite enrichment. Nothing in the setup gives the competitors a
*designed* advantage, and nothing hides a designed advantage for IsalSR.

**What was wrong was accounting, not design**, and it cut both ways:

- *Pro-IsalSR* (reviewer-attackable): the CPDT dropped ties (F-1) and let fp
  noise decide "ties" (F-2) — both fixed; corrected C1 p-values move but **no
  significance claim flips** (F-1 table). The ρ p-value against the
  definitional baseline is tautological (F-3 — reporting change routed to A8).
  The reported overhead % understated the wrapper's cost 1.57×/2.43× at smoke
  scale (F-7 — fixed).
- *Pro-competitor* (Mario's question): the isalsr arm alone pays untimed
  shadow-sketch cost inside its fixed budget — 0.55 %/2.12 % of wall at smoke
  scale, now timed per run (F-8, decision at Stage F); the ledger adds a
  bounded 0.04 %/0.22 % (F-9, disclose); duplicates are denied the LM restarts
  the baseline enjoys (F-10, disclose as conservative); `T_search` for dedup
  arms absorbed instrumentation, biasing S downward against IsalSR (F-7 —
  fixed); Bingo's cross-arm evaluation counts flatter the baseline 3–4×
  (F-11, do not report cross-arm).

**Decisions this audit leaves to Mario / the ticket owners:**

1. **Merge sequencing**: this branch must merge before the `campaign/c2` tag is
   cut (never mid-campaign); the RunLog field list grows 56 → 60, so Stage C
   re-certifies on the merged commit (≈33 min at `%24`/`sr`).
2. **Shadow sketches in C2's isalsr arm**: keep-with-disclosure vs
   Stage-D-trace-only — read `shadow_time_s` from D1's 12 h cells first (F-8).
3. **A8 rework requirements** (F-3, F-11, F-13, F-14): ρ inferential contrast
   is isalsr-vs-hash; ρ-vs-baseline descriptive; `total_dags_explored`
   non-comparable across Bingo arms; tie policy applies to all three
   contrasts; conservative-substitution check still to be implemented.
4. **Manuscript sentences** (review-answer lane): duplicate-culling mechanism
   (F-10 wording provided); the conversion term in the cost decomposition and
   the corrected S derivation (F-7); Bingo ρ decomposed into verbatim-copy vs
   structural duplicates via `shadow_distinct_host_native` (F-12); hash-key
   per-host rule disclosure (§2.10); per-tier CPDT sensitivity row (F-15);
   corrected C1 CPDT values wherever quoted (F-1).
5. **Decision-log corrections**: §0.2's "runners cache canon_hash → fitness"
   mechanism claim (F-10.3); T01 AC-6's S value shifts under the corrected
   derivation (F-7) — the mechanism finding survives, the number does not.

---

## 6. Decisions taken (Mario, 2026-08-04, via question dialogue)

*(Rewritten 2026-08-04 after an implementer agent accidentally reverted the
uncommitted first version with `git checkout`; content restored from the
session record.)*

| # | Question | Decision |
|---|---|---|
| 1 | CPDT sidedness for the three contrasts | **Pre-registered directions only.** isalsr-vs-baseline stays one-sided (continuity with the submission); isalsr-vs-hash one-sided *only* for ρ/redundancy (direction guaranteed by construction); everything else two-sided |
| 2 | ρ inference vs a definitional baseline (F-3) | **Descriptive vs baseline** (mean ± std, no p); the inferential ρ test is isalsr-vs-hash. The submitted table's ρ p-values disappear from the revision; the response letter explains why |
| 3 | Shadow sketches in C2's isalsr arm (F-8) | **Decide after Stage D**: keep ON through Stage D, read `shadow_time_s` from the 12 h cells, then keep-with-disclosure vs trace-only |
| 4 | Merge timing | **Before the `campaign/c2` tag**, once T17 closes C4; re-run the Stage C wave on the merged commit before the tag is cut |

### 6.1 Per-contrast CPDT — IMPLEMENTED (commit `9b51cd2`)

Mario additionally asked to persist per-contrast effect size and p-value for
IsalSR-vs-{Baseline, Naive-Hash} and Baseline-vs-Naive-Hash. Landed:
`CPDT_CONTRAST_POLICY` (+ `resolve_cpdt_alternative`, `cpdt_primary_p`,
`apply_holm_across_contrasts`) in `analyzer/aggregation.py`;
`arm_a`/`arm_b`/`p_value_holm` on `CrossProblemDominanceResult`
(legacy-tolerant `from_dict`); `analyze.py` loads
`paired_stats_hash_vs_baseline.json` / `paired_stats_isalsr_vs_hash.json` per
problem and writes a `"contrasts"` block into
`cross_problem_dominance_{method}_{benchmark}.json` while keeping the legacy
top-level shape. 35 new tests (`test_cpdt_contrasts.py`); full `tests/unit`
**6,457 passed, 5 skipped, 0 failed**; two-arm C1-era roots byte-identical to
the legacy behaviour (regression tests). Two recorded caveats:
- **Primary-contrast ρ p-values are now NaN by policy** (decision 2), so
  `generate_tables.py:858-867` would render `$nan$` in Table 2's CPDT footer
  until the three-arm table work reads the ρ test from the
  `isalsr_vs_hash` contrast — assigned to the A8-remainder agent.
- For two-sided contrasts the stored `statistic` is SciPy's statistic in the
  sign-of-mean direction; the two-sided p is unaffected.

### 6.2 C4 supervision — CLOSED (job 1761777)

Verdict read from `c2_smoke_v3/c2_preflight/stage_c_certification.json`
(2026-08-04 16:23): **overall GO; C4 PASS** — multiplicity histogram
**{6: 204, 18: 2}** (204 fingerprints at 3 arms × 2 methods; Pagie-1 and
Keijzer-6, the two declared deterministic grids, at 18 = 3 seeds × 6),
`missing_fingerprint = 0`, `cross_arm_disagreement = 0`,
`duplicate_problems_blocking = 0` (the `I.34.27` 1/(2π) restoration holds),
`seed_collapse_blocking = 0`. T17 agent's report concurs: wave 1,260/1,260
COMPLETED, all on `sr`, span 31 m 55 s; C1.10 1260/1260; C1.11 peak MaxRSS
0.67 GB; aggregation 1 h 35 m inside its 6 h wall. **The sole remaining
Stage C blocker from T17-HANDOFF §4.1 is closed.**

### 6.3 Sequencing correction (follows from decisions 3 + 4)

**This branch must merge before *Stage D*, not merely before the tag.**
Decision 3 consumes `shadow_time_s` from the 12 h D1 cells — a field that
exists only on this branch — and D1.7's overhead sanity check must run on the
corrected accounting (pre-merge code understates wrapper cost 1.6–2.4×).
Order: remaining blockers land → Mario reviews and merges → one Stage C
re-cert wave on the merged commit (≈33 min at `%24`/`sr`) → Stage D.

### 6.4 T17 orchestration hand-over (Mario, 2026-08-04)

This session now orchestrates the remaining T17/pre-launch items. A dedicated
implementation agent (Opus) is working, in order: **A6** (manifest.py +
validator), **A13** (FSCRATCH inodes: tar-then-delete of the superseded
`c2_smoke`/`c2_smoke_v2` roots only — `v3`, `~/execs/vena` and HOME are
hands-off; ≈16k of the ≈28k shortfall), **B8** corruption half (local
integration tests + SP-0-capped single-task Picasso probe), **A8 remainder**
(analyze.py `--variants` plumbing, 3-arm Friedman/Nemenyi, conservative-
substitution check, minimal 3-arm table emission — gated on commit `9b51cd2`
being present, so it builds on the contrast machinery instead of duplicating
it), and the **`campaign/c2` tag procedure** (prepare only; SP-0 — Mario
tags at Stage F). A8 hand-off note: the pairwise-CPDT half of A8 is
discharged by §6.1; what remains is the list above plus E1–E7.

### 6.5 Blocker closure — VERIFIED (2026-08-04, 8 commits `6a8b60b..63d0771`)

Independently re-verified by the orchestrator (not taken from the agent's
report): full `tests/unit` **6,555 passed, 5 skipped, 0 failed**; manifest +
resume-corruption suites 52 passed; three-arm/Holm/footer/conservative subset
60 passed; working tree clean; **no `campaign/*` tag exists** (procedure
documented in `slurm/c2_tag_procedure.md` only); live `quota` re-read:
FSCRATCH **155.4k/250k = 94.6k headroom** (≥60k criterion PASSES; only
15.9k of the drop attributable to the smoke-root archival — the earlier
248.6k reading was likely stale), superseded `c2_smoke`/`c2_smoke_v2` roots
tarred (7,932 members each, verified) then removed, `c2_smoke_v3` intact.

| Item | State |
|---|---|
| A6 manifest + strict validator | CLOSED (46 tests; truncated manifest → exit 1) |
| A13 FSCRATCH inodes | **PASSES**; HOME *space* quota still over (0.34/0.28 TB, 2 days grace) — Mario's lane; ≥15k-file mail to `soporte@scbi.uma.es` still outstanding |
| B8 resume + corruption | CLOSED — local integration tests (6) + Picasso probe jobs 1762279/1762282/1762284: fresh → skip → corrupt-detect-delete-rerun, all observed |
| A8 remainder | CLOSED — `--variants` end-to-end, 3-arm Friedman/Nemenyi, conservative-substitution check, 3-arm tables incl. the ρ-footer fix (no `$nan$`); Holm-by-3 asserted (52 tests) |
| `campaign/c2` tag | Procedure only; not cut (SP-0) |

**Needs Mario's confirmation (none blocking):**
1. Conservative-substitution failure levels chosen by the implementer:
   R² → 0, NRMSE → 1, ρ → 1, redundancy → 0, plus a clamp so substitution can
   never favour the dedup arm. Consistent with T08's runtime scoring policy
   (R² = 0 / NRMSE = 1); endorse, confirm.
2. ρ's Holm family is 1 (its other two contrasts are descriptive by decision
   2) — statistically correct, stated so the tables are read right.
3. Table 1/S still print the primary contrast's raw one-sided R² p rather
   than `p_value_holm` — narrative choice for the review-answer lane.
4. `c2_smoke/worker.sh`'s ≥2-seed guard is unusable for SP-0 probes (seed 0
   only); a dedicated probe worker exists, but the friction will recur.

---

## 7. Stage D configuration lock (Mario, 2026-08-04, via question dialogue)

| # | Decision | Value |
|---|---|---|
| 1 | Merge & submission authority | **Mario merges** `feature/experiment-fairness-audit` → `cpp-core-port` himself, after review. The implementation agent builds everything on the audit branch now; **submission of the Stage C re-cert wave and Stage D is gated on the merge landing** and happens only on an explicit go |
| 2 | Trace problem | **Pagie-1** (6 cells: × 3 arms × 2 methods). D2 detailed trace = **Bingo × Pagie-1 × seed 101**, run as extra persistence inside that cell |
| 3 | Stage D config | Defaults accepted: `max_time 43,200 s`, wall **16 h**, `--constraint=sr`, **seed 101** all 12 cells, shadow ON + `--ledger` ON, mem Bingo-IsalSR **256 GB** / Bingo other **32 GB** / UDFS **16 GB**. **Addendum (Mario): measure memory properly so production can request less without OOM risk** — per-task RSS time series (sampler) in addition to `sacct` MaxRSS (`JobIDRaw`, `.batch` step); D1.2 output = recommended production `--mem` per (method, arm) at peak + ≥30 % headroom |
| 4 | Headline tables | **Holm-adjusted p (`p_value_holm`) in main Tables 1/S; raw one-sided in the supplement.** On C1's numbers Bingo R² ≈ 1.8×10⁻² after Holm — still < 0.05 |
| — | Conservative-substitution levels | Confirmed as implemented (R²→0, NRMSE→1, ρ→1, redundancy→0, clamp) — no objection raised |

Cell enumeration (12): Pagie-1 × {baseline, hash, isalsr} × {UDFS, Bingo} (6) +
{Korns-12, Vlad-2} × {baseline, hash, isalsr} × Bingo only (6). Memory groups:
UDFS 3 × 16 GB, Bingo non-isalsr 6 × 32 GB, Bingo isalsr 3 × 256 GB.
C5 (`smoke_vs_C1.md` §3) is drafted by the agent from `c2_smoke_v3` against
the frozen §2 expectations and **signed by Mario** before Stage D submits.

### 7.1 Stage D implementation — COMPLETE, nothing submitted (2026-08-04)

All seven deliverables built and committed on this branch. **No Picasso
submission and no deploy happened**; SP-0 holds and the post-merge sequence is
`slurm/c2_stage_d/RUNBOOK.md`, unexecuted.

| # | Deliverable | Where | Evidence |
|---|---|---|---|
| 1 | Stage D harness | `slurm/c2_stage_d/{launcher,worker,aggregate_worker}.sh`, `experiments/scripts/stage_d_task_spec.py` | 35 registry tests; `bash -n` clean |
| 2 | D1 certifier | `experiments/scripts/stage_d_certify.py` | 66 tests; 154 with `test_c2_certify`+`test_manifest` |
| 3 | D2 trace + D3 replay | `experiments/models/stage_d_trace.py`, `experiments/scripts/stage_d_mode1_replay.py` | 85 tests (46/33/6) |
| 4 | C5 draft | `c2_preflight/smoke_vs_C1.md` | 10 quantities against 1,260 run logs; **DRAFT, awaiting signature** |
| 5 | Headline tables → Holm | `experiments/figures/models/generate_tables.py` | 24 tests; 112 across the four table suites |
| 6 | Runbook | `slurm/c2_stage_d/RUNBOOK.md` | 8 steps, unexecuted |
| 7 | Design record | `docs/md_files/changes/stage_d_design.md` | this table |

**Decisions taken inside the lock** (none contradict §7): registry is the sole
cell source, consumed by worker (`D_KEY='value'` eval) and certifier (Python);
three arrays because `--mem` is per job; RSS period 60 s, sound because `VmHWM`
is monotone — validated locally, peak 190.2 MB retained while `VmRSS` fell to
10.4 MB; D1.2 peak = `max(sacct MaxRSS, timeseries VmHWM)`, recommendation
`ceil_to_8GB(peak / 0.70)`; D1.6 ρ one-sided at ≥ 0.90 × C1, R² two-sided at
|Δδ| ≤ 0.15; D2 sampling rate 100, from 571.7 B/record measured live.

**Three findings Mario should read before merging:**

1. **C5 §3.5 — a real deviation.** Bingo ρ at 900 s sits **1.1–1.7 % below**
   C1, where §2 expects a rise. The dangerous cause is falsified directly:
   SP-4 finds **0** SUB/DIV tokens in 292 canonical strings, so the
   decomposition *is* reaching the canonicaliser. The remaining cause is the
   48× budget gap (ρ is cumulative). Handed to D1.6, which is 12 h vs 12 h.
2. **`c2_smoke_v3` carries 58 of 60 spec fields** — `conversion_time_s` and
   `shadow_time_s` do not exist pre-merge. This is the concrete instance of
   §6.3 and is why RUNBOOK step 2 re-runs Stage C on the merged commit.
3. **`wl_subtree_unified/` is a directory of dangling symlinks.** Only
   `analysis/` resolves; absolute per-problem values were recomputed from
   `wl_subtree_hard/models_hard/`. The certifier's D1.6 reconstructs C1 ρ from
   the CPDT deltas and cross-checks it against `three_axis_summary` (gap
   5.1×10⁻⁴ bingo, 0.0 udfs).

**Unprompted observation.** `ρ_hash = 1.0000` on **all 210** UDFS smoke cells
against `ρ_isalsr = 1.6552`, while Bingo's hash arm does merge (1.7247 vs
1.7814). That is §4.4 D3's anticipated null result appearing on one host only —
a more informative answer to R1.4 than either uniform outcome, and §10.1 should
record that we knew before the campaign ran.

**Verification.** Full `tests/unit` re-run after every landing; ruff clean on
all touched files. The 26 failures seen in one intermediate run were a mid-edit
snapshot (`NameError: StageDTracer` before its import landed) and clear on
re-run — not a regression.

### 7.2 Execution log (2026-08-04, runbook driven by the orchestrator on Mario's direct instruction)

Mario's instruction: *"Merge the branches, sign C5. Ignore the HOME space for
now, and execute C re-cert + Stage D submission."* The build agent declined to
act on relayed consent (correctly, from its vantage) and surfaced that HOME's
grace is **32 h** — shorter than Stage D's horizon. Mitigation without touching
HOME: all wave/Stage-D logs redirected to FSCRATCH via the launchers' existing
`C2_LOGS_DIR`/`D_LOGS_DIR` overrides; only the SP-0 probe writes to
`~/execs` (the worker's own guard requires it — see below).

| Step | Result |
|---|---|
| Merge | `6c3798f` (parents `e7e03c9` + `1856a97`, zero conflicts); C5 signed `a001810`, signature line completed `a470da2` |
| A2 gate on merged commit | 6,759 passed / 5 skipped / 0 failed; ruff clean; `mypy --strict` clean |
| Deploy #1 | `a470da2` — SP-1 OK (remote exact, clean, `.git` synced), SP-2 OK (gcc 13.2.0, `build_hash 298fc118…`, `engine=cpp`) |
| Wave #1 (aborted) | 🔴 **Self-inflicted provenance split**: a mid-wave redeploy (the launcher fix below) would have left cells recording two different HEADs. Cancelled all 42 arrays + aggregation ≈10 min in, wiped `c2_smoke_v4`, relaunched on the stable commit. **Lesson (generalises the config rule): never deploy while a wave is running — a deploy IS a config edit** |
| `--test-only` findings | Two real launcher bugs the local phase could not catch: (1) python resolved via the workstation `~/.conda` path that does not exist on Picasso — fixed `a455d6c`; (2) `GROUPS=` — a **reserved bash array**; assignment returns error status and `set -e` killed the script silently — fixed `9b351e7`. After both: exit 0, 12/12 cells enumerate per the §7 lock, TRACE flag on Bingo×Pagie-1×isalsr |
| Probe #1 (1765639) | FAILED **by design**: the worker's own SP-0 guard rejects probe output outside `~/execs/isalsr/` (my FSCRATCH redirect violated the convention the worker enforces). Guard kept; probe resubmitted per RUNBOOK verbatim |
| Wave #2 (live) | 42 arrays on `a455d6c`, root `c2_smoke_v4`, logs on FSCRATCH, `%24`, aggregation+certifier **1766718** (`afterany`) |
| Probe #2 | **1766802** (bingo_isalsr worker, 900 s payload, RSS interval 15 s, `~/execs/isalsr/c2d_probe`) |
| Monitor | armed on 1766718 (re-cert verdict), 1766802 (probe), and wave failure counts |

Pending at the time of writing: re-cert verdict (gates Stage D submission),
probe artefact check (`rss_timeseries.csv` + trace flag), post-wave deploy of
`9b351e7`, then the three Stage D groups + `c2d_certify`.

### 7.3 Shadow sketches — what they are, measured cost, and the decision (Mario, 2026-08-04)

**Definition.** In the `isalsr` arm only, per candidate DAG, the runner computes
four extra serialisations and feeds each into a HyperLogLog sketch (p = 16,
64 KB, ±0.41 %): three **fixed-order hashes** (insertion / topological /
topological-commutative) over the *adapter's renumbered* DAG, and one
**host-native** hash (Bingo `command_array` row order, UDFS `node_dict` order).
They never affect the search — nothing is deduplicated by them; they only count
distinct values beside the real canonical dedup. Fields:
`shadow_distinct_{insertion,topological,topological_commutative,host_native}`.

**Measured cost** (v4 wave, 900 s, 210 cells per arm, median share of wall
clock, from the `shadow_time_s` field added by F-7):

| Host | shadow % of wall | method overhead % (canon + conversion) |
|---|---|---|
| Bingo `isalsr` | **17.6 %** | 14.8 % |
| UDFS `isalsr` | 0.034 % | 0.042 % |

On Bingo the instrumentation costs **more than the method it instruments**, and
it is paid by one arm inside a fixed budget. The C1 trajectory replay
(§11.1, 2026-08-04) showed Bingo's paired effect eroding **69 %** from 12 h to
8 h; a 17.6 % cut aimed at the `isalsr` arm alone is that mechanism pointed at
our own effect, whose Bingo `d` is already 0.034.

**Decisions.**

| # | Question | Decision |
|---|---|---|
| 1 | Stage D | **13 cells**: the 12 locked cells run shadow **OFF** (so D1.6/D1.7 compare to C1 without a budget penalty C1 never paid), plus one extra Bingo × Pagie-1 × `isalsr` cell with trace **and** shadow ON, which produces the D2 stream, the D3 replay input and a 12 h shadow-cost figure for the record. Cost +12 core-hours |
| 2 | C2 production | **Shadow OFF on both hosts.** Uniform, simplest to state, both `isalsr` arms run at full budget |
| 3 | Source of the fixed-order numbers | **Stage D trace + D3 Mode-1 replay only.** Cross-arm approximation and dedicated side-runs both declined |

**Revision of F-8 / §2.9 — the "steel-man" obligation is withdrawn (Mario's
challenge, accepted).** The audit recommended reporting the adapter-order
fixed-order ρ alongside the live host-native ρ, on the reasoning that the live
number could read as a strawman. That reasoning does not survive scrutiny: the
adapter's renumbering is *part of IsalSR's own preprocessing*, so the
adapter-order rung is not an independent naive competitor but a hybrid that
cannot be built without our adapter. R1.4 asks what a practitioner obtains
**without** IsalSR, and that is the committed T04 design — hash the
representation the host already stores. Publishing two ρ_hash numbers invites
"which one is the baseline?" and dilutes the single clean contrast.

What replaces it, and it is stronger: **state which representation the naive
hash keys on and why that is the naive one**, and answer the "UDFS ρ_hash =
1.0000 looks rigged" objection *mechanistically* — UDFS's systematic
enumeration emits structurally distinct `node_dict`s, so a representation-order
hash has nothing to merge; its redundancy is purely isomorphic and therefore
invisible without an isomorphism-invariant key. That is the paper's thesis
demonstrated, not a caveat. The adapter-order rung remains computable offline
from D3 on the traced stream and is **held in reserve** for the response letter
if a reviewer asks whether a cleverer ordering was tried — never as a second
baseline in the tables.

**Consequence carried, not hidden.** With shadow off everywhere and only
trace + D3 as the source, the F-12 verbatim-clone decomposition narrows from a
per-problem table to **one worked example (the traced Bingo × Pagie-1 cell)
plus the mechanism** — VarAnd emits `parent.copy()` offspring at
`(1 − P_cx)(1 − P_mut)` ≈ 36 %, already established by B12. Bingo's redundancy
claim must therefore be phrased as "candidate evaluations avoided by structural
deduplication, of which a measured share on the traced problem are verbatim
copies the host's own clone-skipping already avoids", rather than as a
portfolio-wide decomposition. This is a reporting limitation, not a defect.

**As implemented (2026-08-04).** The decision is a *configuration* change, not a
code change — the runner's default is untouched, so a bare runner outside the
campaign behaves as before.

| Piece | Where |
|---|---|
| `shadow_hash: false` in all **14** production configs | in the **method** block (`bingo:` / `udfs:`), **not** `isalsr:` — `create_runner` hands the runner `config.get(method, {})`, so a key under `isalsr:` would be **silently ignored** and shadow would have stayed on for all 8,400 runs. Same shape as the `ISALSR_LEDGER_ENABLED` trap; caught by reading the call site before editing |
| `experiments/configs/bingo_hard_trace.yaml` | the trace cell's config: byte-identical to `bingo_hard.yaml` except `shadow_hash: true`, so the difference lives in `config_sha256` instead of a flag |
| Registry now **13 cells** | 12 certification + the traced one. `STAGE_D_CERTIFICATION_CELLS` is what D1.1–D1.8 iterate; the trace cell is excluded structurally, not by convention |
| Trace cell runs at **seed 102** | it repeats cell 10's `(method, suite, problem, arm)`, so at seed 101 the orchestrator would write both runs into one directory and the second would overwrite the first. 102 is outside campaign seeds 1–20 and the 21–30 top-up |
| Tests | `tests/unit/test_shadow_hash_config.py` (35) locks all 14 configs, the block the runner actually reads, and the one-key trace diff; the Stage D lock tests were updated to the new 13/12 registry |

**Sequencing consequence — the config change must precede the final re-cert.**
Turning shadow off is a *configuration* change (`shadow_hash: false` in the
`isalsr:` block), so it moves `config_sha256`. Under §5.1's one-commit,
one-configuration rule the order is: land the config change → run the clean
Stage C wave (v5, which was needed anyway: 161 of v4's 1,260 cells recorded
`a455d6c-dirty` from the mid-wave `sed`) → read its verdict → submit Stage D's
13 cells. Stage D must not run under a configuration no Stage C wave has
certified.
