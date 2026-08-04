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
