# T17 — C2 submission certification: prove the campaign is correct before spending 100,800 core-hours

## 0.1 ⚠ Amendment to `EXECUTION-PLAN.md` §4.3 — three seeds, not one

**The plan specifies Stage C at one seed. This ticket runs three: 0, 101, 102.**
Record the change in §11.1 before submitting; §4.3 has been amended to match.

**Why one seed was wrong.** `compute_paired_stats`
(`experiments/models/analyzer/aggregation.py:207`) requires **three matched
seeds**:

```python
if len(common_seeds) < 3:
    raise ValueError(f"Too few paired seeds ({len(common_seeds)}) for statistical testing")
```

At one seed, none of the three paired contrasts is ever constructed. The smoke
would then certify every artefact **except** the ones the paired design actually
rests on — `paired_stats.json`,
`paired_stats_hash_vs_baseline.json`, `paired_stats_isalsr_vs_hash.json` — and
the first time that code ran on three arms would be on the real campaign in
September. That is the single most expensive failure mode left (§4.5).

**Note that two seeds does not work.** The threshold is `< 3`, not `< 2`. Three
is the minimum that computes, and it is also exactly the minimum sample
`scipy.stats.shapiro` accepts — so the Shapiro-Wilk → *t*-test / Wilcoxon branch
selection is exercised **at its lower boundary**, which is where it is most
likely to break.

**What three seeds buys, and what it does not.** It buys the *code path*: three
contrasts constructed, Holm applied across problems, `aggregate.csv` written per
variant, and Stage E (§4.5 E1/E2) fed real paired input instead of an empty
directory. It buys **no statistics whatsoever**. A Wilcoxon signed-rank at
`n = 3` has `2³ = 8` sign configurations, so its minimum attainable two-sided
`p` is `2/8 = 0.25` — it *cannot* reach significance, by construction. Any
number this stage emits is a smoke signal about plumbing. Do not read it, do not
quote it, do not put it in a table.

**Why 0, 101, 102.** SP-0's constraint is that a smoke output must never be
mistakable for, or mergeable into, a campaign cell. That intent is preserved
exactly: all three are outside the campaign seeds 1…20 **and** outside the
21…30 top-up range reserved as spillover priority 1 (§8.4), which seeds 21/22
would have collided with. Verified: `parse_seeds("0,101,102")` resolves, and
`seed_dir` yields `seed_00`, `seed_101`, `seed_102` — visually unmistakable as
non-campaign directories.

**The ≥3-seed guard in the orchestrator stays.** It is no longer triggered in
Stage C, but SP-0 probes elsewhere legitimately run a single seed and must not
abort after a complete run.

---

| Field | Value |
|---|---|
| Type | **Certification gate.** No new science. Its only product is a go/no-go with evidence |
| Owner | **Mario** (+ Claude Code) |
| Implements | `EXECUTION-PLAN.md` **Stage C** (§4.3), plus the Stage A/B checks that gate it (§4.1, §4.2) |
| Depends on | T01, T04, T05, T08 (code halves — all landed), and the four provenance closures of 2026-08-03 |
| Blocks | **Campaign C2.** Stage F sign-off cannot be written without this ticket's artefacts |
| Shape | `{baseline, hash, isalsr} × {UDFS, Bingo} × 70 problems × 3 seeds` = **1,260 tasks**, `max_time = 900 s` |
| Seeds | **0, 101, 102** — three, and disjoint from both the campaign set (1…20) and the reserved top-up set (21…30). §0.1 explains why three and not one |
| Cost | ≈315 core-hours — **0.3 % of the campaign it certifies** |
| Status | **IN PROGRESS (2026-08-03).** Three launch blockers closed (§2.1 `--ledger`, §2.2 topology → 42 arrays, and a newly-found post-processing race). Stage A done bar A1/A6; Stage B green on B1/B2/B3/B4/B6b/B7, B9 in flight. Harness built and `--test-only` clean on 42/42. **Escalated, not blocking Stage C:** five canonical-string completeness counterexamples (T07). **Blocking C2, not Stage C:** FSCRATCH file headroom and the HOME quota grace |

---

## 0. What this ticket is, and what it is not

**It is the coverage test.** Every problem × every arm × every method, at least
once, on real Picasso hardware, producing every artefact the analysis will later
consume. It answers one question: *if we submit 8,400 runs tomorrow, will they
produce the measurements the paper needs, correctly?*

**It is not a source of numbers.** No quantity this ticket produces enters a
table. A 900 s run is not a 43,200 s run. Anything that looks like a result here
is a sanity signal, nothing more.

**It is not Stage D.** The 15-minute smoke proves nothing about a 12-hour run:
memory growth, heap fragmentation, dedup-set size and convergence are all
budget-dependent. Stage D (12 tasks at full budget) follows this ticket and is
separately blocking.

> ### The standing rule that overrides everything below
>
> **`EXECUTION-PLAN.md` §4.0 SP-0: nobody except Mario submits C2.** This ticket
> submits **Stage C**, which is a named stage of the pre-flight suite, not a
> probe — SP-0's carve-out is explicit: an output root under `c2_smoke/` is
> permitted "unless the ticket owns that stage", and **this ticket owns Stage C**.
>
> What still binds, without exception:
> - **Seeds 0, 101, 102 only.** All three deliberately outside the campaign seed
>   set (1…20) *and* the reserved 21…30 top-up range, so a smoke output can never
>   be mistaken for, or merged into, a C2 cell. See §0.1.
> - **`max_time = 900 s`**, never the production budget.
> - **Output root `c2_smoke/`**, never the campaign root.
> - A failing stage means **fix, then re-run the stage from the top** — never
>   "note it and continue".

---

## 1. Read these before doing anything

Grouped by the question each answers. Do not skip §1.1: two of the four items
there were written on 2026-08-03 and change what the artefacts contain.

### 1.1 The contract — what a correct run must produce

| File | Why |
|---|---|
| `.claude/notes/review/tasks/EXECUTION-PLAN.md` | **Authoritative.** §3 (blockers, and the during/after dividing line), §4.0 (SP-0…SP-7), §4.3 (Stage C, every criterion), §5 (protocol invariants). If this ticket and the plan disagree, **the plan wins** |
| `docs/md_files/changes/c2_run_provenance.md` | The four closures of 2026-08-03 (A7-BUG, C1.9-BUG, P3, P4) and **the two launch-blocking defects found while closing them**. Read §3 in full — one of them silently disables the R1.2 evidence base |
| `experiments/models/schemas.py` | Every field C1.2 enumerates. `RunMetadata` (now with `data_fingerprint`, `config_sha256`), `SearchSpaceResults` (now with the ten ledger fields), `RegressionResults.n_nonfinite_test_predictions` |
| `experiments/models/status_ledger.py` | The P4 write-ahead protocol and `reconcile()`, which is how C1.15 names cells instead of counting them |

### 1.2 The alphabet and the operator sets — SP-4, A4b, C1.13

| File | Why |
|---|---|
| `experiments/models/alphabet_guard.py` | The containment guard. `validate_bingo_operators()` / `validate_udfs_operators()` — every configured operator must have an image in 𝓛 |
| `experiments/models/commutative_encoding.py` | `SUB → ADD+NEG`, `DIV → MUL+INV`, applied **inline inside both adapters**. `k` is ≈22 % larger than any pre-T16 number in the repo |
| `docs/md_files/changes/t16_commutative_decomposition.md` | Why 61.1 % of C1's candidates carried forbidden labels, and why it was invisible in the logs |
| `experiments/configs/bingo_*.yaml` (7 files) | A4b: **uniform operator set per method across every problem**. All seven must carry `["+","-","*","/","sin","cos","exp","log","sqrt","pow"]` |
| `experiments/configs/udfs_*.yaml` (7 files) | UDFS takes **no** operator set from the YAML — `to_dag_regressor_kwargs()` never forwards it. These lists are documentation of the vendored `NODE_ARITY` table only |

### 1.3 The engine — SP-2, SP-3, B2, B6b, C1.14

| File | Why |
|---|---|
| `CLAUDE.md` § "Rebuilding the C++ extension" | **Read before trusting any `backend="cpp"` result.** `--no-build-isolation` SILENTLY FAILS and the stale `.so` keeps loading. Never read pip's status through a pipe |
| `src/isalsr/core/backends.py` | `engine()` honours `ISALSR_ENGINE`; `build_info()` returns `build_hash`/`isa_level`/`avx512f` |
| `experiments/models/hardware_info.py` | `_engine_info()` reads **actual dispatch**, not the compiled-in default — the B2 defect, guarded |
| `slurm/smoke_cpp/`, `slurm/alphabet_gate/` | Existing harnesses to reuse for B1/B3. Alphabet gate precedent: job 1692451, 2026-07-30, 5,551 DAGs, zero forbidden labels |

### 1.4 The hosts and their failure modes — SP-5

| File | Why |
|---|---|
| `experiments/models/orchestrator.py` | `create_runner` (three arms), `_configure_ledger`, the write-ahead status block, the ≥3-seed guard on paired stats |
| `experiments/models/bingo/isalsr_runner.py` | B12: `VarAnd` produces `parent.copy()` offspring with `fit_set=True` ~36 % of the time; `_established` is what catches them |
| `experiments/models/udfs/isalsr_runner.py` | Monkey-patches `evaluate_cgraph` at **module level**; `spawn` workers bypass the patch. Safe only because every config sets `processes: 1` — **verify this still holds** |
| `experiments/models/fallback_ledger.py` | The five paths, and the `ISALSR_LEDGER_ENABLED` default that is the subject of §2.1 |

### 1.5 Submitting to Picasso

**Load the `picasso-sbatch` skill before writing or editing any SLURM script.**
It is the authority on directives, and its *CPU array jobs* silent-failure
checklist is what this ticket's scale needs: array-size limits, the ≥15,000-file
support threshold, `$LOCALSCRATCH`, and the FSCRATCH purge policy.

| File | Why |
|---|---|
| `.claude/skills/review-ticket/references/picasso-loop.md` | The submit/monitor/collect loop |
| `slurm/workers/models_experiment_slurm.sh` | The existing worker. Task id = `problem_index × n_seeds + seed_index + 1` |
| `slurm/{hard,cherrypicked,roundoff}_config.yaml` | C1 resource profiles. **Historical — do not edit them.** C2's are new files |
| `slurm/t04_probe/`, `slurm/t05_probe/` | Recent probe harnesses; `slurm/t04_probe/sp_probe.py` is the SP-1…SP-6 reporter to reuse |

---

## 2. Blockers this ticket must resolve before submitting anything

These were found on 2026-08-03 while closing the provenance gaps. Each would
have silently degraded or invalidated the campaign.

### 2.1 🔴 The launcher must pass `--ledger`

`ISALSR_LEDGER_ENABLED` defaults to `"0"`. A repo-wide grep finds it set in
`measure_ledger_overhead.py` and in unit tests — **in no worker, no launcher, no
config.**

Launch C2 without it and all 8,400 runs record five reachability rates of zero.
That reads as *"no fallbacks occurred"*; it means *"nothing was counted"*; and
the difference is **unrecoverable**, because the population exists only while a
search runs. This is SP-6's trap verbatim.

**Required:** the worker passes `--ledger`, and C1.9 asserts
`ledger_enabled == true` on 280/280 dedup-arm tasks. **Do not** assert the rates
are zero — assert the counters are *alive*, i.e. `n_ledger_sampled > 0`.

**Still open, and not this ticket's to decide:** check **B9** reserves the right
to remove the counters from C2 if their overhead is material under the C++ engine
and the decomposed alphabet — both changed underneath T06's original
measurement. Measure the overhead at B9; the keep/drop call is Mario's (T06
AC-10). If they are dropped, C1.9 is struck and T06 reopens for a separate
subsampled run — a violation *rate* does not need the full campaign.

### 2.2 ⚠ The array topology in the plan does not match the configs

`EXECUTION-PLAN.md` §1 states **6 arrays × 1,400 tasks**. But every config file
declares **exactly one** benchmark suite, and the launcher maps one experiment
entry to one `(method, arm, config)` array. The real shape is:

```
7 suites × 3 arms × 2 methods = 42 arrays        (8,400 runs either way)
```

Two ways to reconcile, and the **smoke must use whichever production will use** —
certifying a topology you will not launch certifies nothing:

| Option | Consequence |
|---|---|
| **A — 42 arrays**, one per `(method, arm, suite)` | No new configs. Per-suite resource sizing already exists. Smoke arrays are 18–42 tasks each (suite size × 3 seeds; largest is Strogatz, 14 × 3 = 42), comfortably inside every limit. §11.3's launch ledger grows from 6 rows to 42 |
| **B — 2 merged configs** (one per method, all 7 `benchmarks:` sections; the orchestrator already loops over them) → 6 arrays as planned | Matches the plan and A12's arithmetic as written. Costs two new config files and their `config_sha256` |

**Recommendation: A.** It changes no configuration content, so it cannot perturb
the A4b operator-set invariant, and smaller arrays fail more cheaply. Record the
decision in §11.1 either way, and update §1 and §11.3 of the plan to match.

### 2.3 Still open, and not blocking *this* ticket

Both are post-hoc computations over run logs, so the smoke can produce its
artefacts without them. Both block the **campaign**.

| Item | State |
|---|---|
| **A6** — MANIFEST schema + validator | `experiments/models/manifest.py` **does not exist**. Needed before submission, consumed at Stage F |
| **A8** — three-arm analyzer | `analyze.py` hardcodes `["baseline", "isalsr"]` in several places (≈lines 108–115, 148, 360–366). Pairwise CPDT with **Holm across three contrasts** and Friedman/Nemenyi over three arms are not implemented. Blocks **Stage E**, not Stage C |

---

## 3. Acceptance criteria

### AC-1 — Desk checks (Stage A, no queue time)

| # | Check | Pass criterion | Evidence |
|---|---|---|---|
| **A1** | Annotated tag `campaign/c2` on the exact commit; tree clean; tag pushed | tag resolves, `git status` empty | `git show campaign/c2` |
| **A2** | `pytest tests/`, `ruff check src/ tests/ experiments/models/`, `mypy --strict src/isalsr/` | all green. Baseline 2026-08-03: **6,247 passed, 5 skipped**, ruff and mypy clean | raw output, not a claim |
| **A3** | Backend parity. Rebuild per `CLAUDE.md`; `.so` mtime newer than the last C++ edit **at the site-packages path** | byte-identical canonical strings on both backends | `stat -c "%y" $(python -c "from isalsr.core import _native; print(_native.__file__)")` |
| **A4** | Resolved hyperparameters for all `(method, suite)` configs; the three arms differ **only** in `--variants` | a diff table; nothing arm-specific in any YAML | `c2_preflight/config_diff.md` |
| **A4b** | **Operator sets.** (i) identical across arms for every `(method, problem)`; (ii) every configured operator has an image in 𝓛 | (i) 70/70; (ii) a deliberately bad config raises `AlphabetCoverageError` | `c2_preflight/operator_sets.csv` + `pytest tests/unit/test_alphabet_guard.py` |
| **A5** | Seeds declared 1…20 and confirmed to be the same integers C1 used; `0 ∉ seeds` | recorded in MANIFEST | MANIFEST |
| **A13** | 🔴 **Quota, re-read live on the day.** ≈42,000 files for C2 + ≈1,260×6 ≈ 7,600 for the smoke (a `status.json` now joins the per-run set, and three seeds triple the count over the original estimate) | ≥60,000 files FSCRATCH headroom; HOME under soft quota | `ssh picasso 'quota'` capture |

### AC-2 — Micro-jobs (Stage B, each < 5 min)

| # | Check | Pass criterion |
|---|---|---|
| **B1** | Environment probe: hostname, `lscpu`, `isalsr.__file__`, native path **and mtime**, `_ENGINE`, `git describe`, versions, `free -g`, and a resolvability check on all 70 dataset paths | `engine == native`; tag `== campaign/c2`; **70/70** paths resolve |
| **B2** | C++ capability probe **with a negative control**: re-run under `ISALSR_ENGINE=python` (there is no `ISALSR_FORCE_PYTHON`). Assert on **observed dispatch**, by counting calls into `_cpp_ext.fast_canonical_string`, not on a printed name | `native` **and observably calls C++** in run 1; `python` **and observably does not** in run 2. **A probe that says `native` in both proves nothing** |
| **B3** | Alphabet gate on the frozen commit (`bash slurm/alphabet_gate/launcher.sh`, ~90 s) | 0 `NodeType.SUB`, 0 `NodeType.DIV`, 0 `-`, 0 `/` in any canonical string |
| **B4** | Equivalence gate re-run **on a compute node**: exhaustive k=1..8 + ≥5,000 evolved decomposed DAGs | 0 mismatches, `self_comparison == false`. A workstation pass does not certify a different compiler or CPU |
| **B5/B6** | Node census (20-task, 1-min array) → node-constraint decision | either a pinned constraint or a written argument + a balance-reporting plan |
| **B6b** | **AVX-512 portability.** Build with `module load gcc/11.1.0` (the system g++ 7.5.0 **cannot compile `-march=x86-64-v3`** and `pip install` fails outright). Import the `.so` on an `sd` node **and** an `sr` node | imports and canonicalises on both; `isa_level=x86-64-v3`, `avx512f=0` |
| **B7** | `sbatch --test-only` on **every** array, with real `--array`, `--mem`, `--time`, `--constraint` | exit 0 on all; task counts exactly as intended (see §2.2) |
| **B8** | Resume/idempotency: run one task; re-submit; then corrupt its `run_log.json` and re-submit | second run **skips**; corrupted run is **detected, deleted, re-run**. Both observed |
| **B9** | T06 counter overhead **re-measured under the C++ engine and the decomposed alphabet**, both hosts | counters live and finite; overhead below T06's threshold. **Design the probe so a live counter is distinguishable from a dead one** — a 60 s smoke once drew 5 DAGs |

### AC-3 — The 1,260-task certification array (Stage C)

**Configuration:** `max_time = 900 s`, **seeds 0, 101, 102** (`--seeds 0,101,102`),
70 problems × 3 arms × 2 methods, output root `c2_smoke/`, **production `--mem`
and `--constraint`** (the memory profile measured here is what sizes
production), worker passes `--ledger`.

Every criterion is **blocking**. One violation stops the stage.

| # | Criterion | Threshold |
|---|---|---|
| **C1.1** | Every task exits 0 | 1,260/1,260 |
| **C1.2** | Every `run_log.json` exists, parses, and validates against the extended schema — **every field present, correct type** (the full list is in `EXECUTION-PLAN.md` §4.3, plus `data_fingerprint`, `config_sha256`, and the ten ledger fields) | 1,260/1,260 |
| **C1.3** | **No NaN, no inf** in any regression metric. Since T08 this is *runtime-enforced*: an expression undefined on part of the evaluation set scores `R²=0`/`NRMSE=1` and records `n_nonfinite_test_predictions > 0`. **A NaN now means the guard itself is broken.** Additionally: report the distribution of `n_nonfinite_test_predictions` — non-zero is a legitimate scientific outcome (extrapolation failure), not a blocker, but it must be counted and disclosed | 1,260/1,260 NaN-free |
| **C1.4** | Train/test shapes asserted against the registry. **Vlad-7 is 300/1200, Keijzer-6 is 50/120, Pagie-1 is 676/2500 — these are not typos, do not "fix" them** | 70/70 |
| **C1.5** | `solution_recovered` computable for every problem (a `sympy_expression` exists) | 70/70 |
| **C1.6** | `isalsr` arm: `unique_canonical_dags > 0` and `ρ ≥ 1` on 420/420; `ρ > 1` on ≥90 %. **`ρ < 1` is arithmetically impossible and means a counter is broken; `ρ = 1.0` everywhere means the dedup hook is dead and the arm is a null result** | see cells |
| **C1.7** | `ρ_hash ≤ ρ_isalsr` for matched `(method, problem, seed)`. Guaranteed on identical streams, strongly expected live. Local rehearsal (UDFS, Nguyen-1): 1.0000 ≤ 1.6081 ✓ | 420/420 expected; investigate if violations exceed 5 % |
| **C1.8** | `baseline` arm: ledger fields `None`, `canonicalization_runtime_s == 0`. Proves the baseline is genuinely un-instrumented | 420/420 |
| **C1.9** | **T06 counters live** on every dedup-arm task: `ledger_enabled == true` **and `n_ledger_sampled > 0`**, all five paths present and finite. Report the five rates. See §2.1 | 840/840 |
| **C1.10** | `trajectory.csv` non-empty; `timestamp_s`, `best_r2`, `n_dags_explored` monotone non-decreasing; `n_unique_canonical ≤ n_dags_explored` | 1,260/1,260 |
| **C1.11** | **Memory profile.** `MaxRSS` per task by `(method, arm)`; size production `--mem` at p99 + 50 %. 🔴 **`sacct -X` returns an EMPTY `MaxRSS`** — memory is accounted on the `.batch` step. Use `sacct -j <ID> -n -P -o JobID,MaxRSS \| awk -F'\|' '$1 ~ /\.batch$/'`. A profile built with `-X` comes back **silently blank** | 1,260 non-empty values |
| **C1.12** | **`max_time` honoured.** Every task ends at ≈900 s or earlier by convergence; **none** killed by the SLURM wall limit | 0 SLURM time-kills |
| **C1.13** | **Alphabet assertion on the real candidate stream** of every `isalsr` and `hash` task, not only in unit tests: 0 forbidden labels. `POW` only where the operator set permits (Bingo: every problem, since A4b made the set uniform; UDFS: **never**) | 840/840 |
| **C1.14** | `metadata.hardware.engine == "native"` on every task | 1,260/1,260 |
| **C1.15** | **Cell reconciliation** via `status_ledger.reconcile()`: expected 1,260, observed 1,260, machine-checked, **every gap individually named** | exact match, or named |
| **C1.16** | **Paired-stats path constructed — the reason for three seeds (§0.1).** All **three** contrasts emit a file for every problem: `paired_stats.json` (isalsr vs baseline), `paired_stats_hash_vs_baseline.json`, `paired_stats_isalsr_vs_hash.json`. Each parses, reports `n_seeds == 3`, and carries every metric. The across-problem Holm correction runs and writes back. **Assert the files exist and validate; assert nothing about their p-values** — at `n = 3` the minimum attainable two-sided Wilcoxon `p` is 0.25. ⚠ **Amended 2026-08-03:** `PairedStats` had fields `[method, benchmark, problem, metrics]` and **no seed count at all**, so this criterion was unverifiable as written — and, more seriously, §6.2/§6.4's requirement to report *"the true `N` per metric"* was unrepresentable in the artefact a reviewer would be shown. Fixed by adding `n_seeds` (matched seeds) and a per-metric `n` (paired observations surviving NaN deletion). Both default-tolerant on load, so C1's files still parse | 3 × 70 × 2 = **420 files** |
| **C1.17** | `aggregate.csv` written per `(method, problem, arm)`. ⚠ **Amended 2026-08-03: "exactly 3 rows" was wrong.** `aggregate_all_metrics` emits **one row per metric**, aggregated *over* seeds — 14 rows, not 3. The criterion that carries the intended meaning is: the file exists, has one row per metric, and every row's new `n` column equals **3** | 420 files, 14 rows each, `n == 3` |

### AC-4 — Stage C deliverables

| # | Deliverable | Content |
|---|---|---|
| **C2** | `c2_smoke/status_ledger.csv` | One row per `(method, arm, problem, seed)`, 23 columns. Emitted by `collect_status_ledger()`. **The production campaign must write the identical ledger** — this is how "35 missing cells, cause unknown" becomes impossible |
| **C3** | **Dedup-off equivalence control (BLOCKING, the most valuable check here)** | 6 tasks (2 methods × 3 problems): run the `isalsr` runner with **dedup forced off**, compare against `baseline` at the same seed. **Pass:** identical best expression, `r2_train`, `total_dags_explored`. This isolates *"the wrapper perturbs the search"* (RNG consumption, evaluation ordering, object identity) from *"dedup changes the search"*, which is the effect we claim. **If it fails, every paired comparison in the paper is confounded.** If exact reproduction is unachievable for a principled reason, **bound** the residual before Stage D |
| **C4** | **Cross-arm data identity (BLOCKING)** | `data_fingerprint` identical across all three arms and both methods for every `(problem, seed)`: **210 distinct fingerprints (70 problems × 3 seeds), each appearing exactly 6 times.** Three seeds strengthen this materially over one: a generator that consumed RNG per arm could coincidentally agree on a single draw, but not across three. Also assert the **70 × 3 fingerprints are mutually distinct** — two seeds yielding the same data would mean the seed is not reaching the generator. Rehearsed locally: 3/3 arms identical on both hosts |
| **C5** | `c2_preflight/smoke_vs_C1.md` | Reasoned sign-off against C1. **Write the expectations down before looking at the numbers.** Reference: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/model_validation/real_benchmarks/wl_subtree_unified/analysis/`. Expect: smoke R² ≤ C1's 12 h value per problem; ρ **≥** C1's in direction (k grew ≈22 %, so **a drop means the decomposition is not reaching the canonicaliser**); Korns-12 and Vlad-2 Bingo–isalsr **finite**; baseline R² within seed noise. Every anomaly explained or escalated; an unexplained one blocks Stage D |

### AC-5 — Work log

`EXECUTION-PLAN.md` §11.2 filled for every check A1–C5 with date, result and
evidence artefact. **"I checked it" is not evidence. A parsed file is.**

Plus the **SP-1…SP-6 six-row table**, mandatory in every Picasso work-log entry.
An entry without it is not evidence and the check does not count as passed:

| # | Property | Established by |
|---|---|---|
| SP-1 | Provenance — running the commit you think you are | `git rev-parse HEAD`, `git describe --tags --always --dirty`, tree cleanliness, **from the compute node** |
| SP-2 | Installation freshness — the installed package is the code you edited | `isalsr.__file__` + `.so` mtime at the **site-packages** path |
| SP-3 | Engine, **with a negative control** | B2, both directions |
| SP-4 | Alphabet — on the probe's **own candidate stream**, not in unit tests | C1.13 |
| SP-5 | **Both hosts**, every time | every check run on UDFS *and* Bingo |
| SP-6 | Fallback counters **live**, not dead | C1.9 — `n_ledger_sampled > 0`, not merely present |

**SP-7 (this ticket's own falsifiable statement):** *every one of the 1,260 cells
produces a schema-valid `run_log.json` and a reconciled ledger row, on both
hosts, under the decomposed alphabet and the native engine, with the T06
counters demonstrably alive — or the failing cells are individually named.*

---

## 4. Method

1. **Load the `picasso-sbatch` skill.** Write the launcher/worker pair with it;
   work its CPU-array silent-failure checklist. Do not hand-roll directives.
2. Resolve §2.2 (array topology) and §2.1 (`--ledger`) **first** — both change
   what the worker looks like.
3. Stage A → B → C, **each gating the next**. A failure means fix and re-run the
   stage from the top.
4. Submit Stage C as one wave, throttled (`%K`). An unthrottled array at this
   scale is antisocial and invites manual intervention from support.
5. **Time the array end to end.** `1,260 tasks × 0.25 h ÷ wall-clock hours` is
   the *achieved concurrent cores* under real queue pressure, at C2's exact
   `--mem`/`--constraint`. This is the number §8.2 needs and the only way to get
   it: the policy ceiling (`cpu = 9000`) is not binding, **contention is**, and
   `sacctmgr` cannot report contention. At three seeds this estimate is three
   times better resolved than at one, and closer to campaign scale. If it
   sustains ≥300, C2 fits the 2026-09-03 target. If it sustains 100, **invoke
   §8.3 before launching**, not in week three.
6. Write the work log, then hand to Mario for Stage D and Stage F.

---

## 5. Failure protocol

| If | Then |
|---|---|
| A single C1.x criterion fails | Stop. Fix. Re-run **the whole stage**. Partial re-runs certify nothing |
| C3 (dedup-off control) fails | **Do not proceed to Stage D.** Every paired comparison in the paper is confounded by a wrapper side effect. Either fix, or state and bound the residual difference in writing |
| C4 (data identity) fails | Stop the campaign design review. The paired test is comparing different data and no amount of compute repairs it |
| ρ = 1.0 on every isalsr cell | The dedup hook is dead. This is a null result, not a small one |
| The five ledger rates are all 0 **with `ledger_enabled == false`** | §2.1. The counters are dead. Fix the launcher, re-run |
| A NaN appears in any regression metric | The T08 runtime guard is broken. Blocks Stage D |
| Cell count ≠ 1,260 | Name every gap from `reconcile()`. An unnamed gap is the C1 defect recurring |
| Any paired-stats file missing, or `n_seeds ≠ 3` (C1.16) | The three-arm contrast machinery does not work. Fixing it in September, on campaign data, is the failure mode §0.1 exists to prevent |
| The 210 fingerprints are not mutually distinct (C4) | The seed is not reaching the data generator. Every "independent" seed is running the same sample and the paired design has no replication at all |

---

## 6. Work log

### 2026-08-03 — blockers resolved, Stage A and Stage B run, Stage C harness built

**Evidence root:** `/media/mpascual/Sandisk2TB/research/isalsr/c2_preflight/`
**Harness:** `slurm/c2_smoke/{deploy,launcher,worker,aggregate_worker,stage_b_launcher,stage_b_worker,stage_b4_worker}.sh`,
`experiments/scripts/{c2_task_spec,c2_certify,c2_stage_a_evidence}.py`

#### The three §2 blockers

| # | Resolution |
|---|---|
| §2.1 `--ledger` | **Closed.** `worker.sh` passes `--ledger` on every task. C1.9 asserts `ledger_enabled == true` **and** `n_ledger_sampled > 0` — a present-but-zero counter is a failure, not a rate of zero. |
| §2.2 topology | **Option A, 42 arrays.** Verified: 210 tasks per `(method, arm)`, **1,260 total**, largest array 42 (strogatz). Changes no configuration content, so the A4b invariant cannot be perturbed. |
| **§2.3 (new)** | 🔴 **Per-task post-processing was a race.** `orchestrator.py:631-698` ran after every cell: `aggregate.csv` written by 3 concurrent tasks, paired contrasts requiring arms in *other* arrays, and `collect_status_ledger` — a full recursive tree walk plus a shared-CSV write — executed **1,260 times concurrently on GPFS** (8,400 in the campaign). Fixed with `--postprocess {auto,skip,only}`; the certifier independently confirmed that `paired_stats_isalsr_vs_hash.json` is **never emitted** without the `only` pass, so C1.16 would have failed campaign-wide. |

#### Stage A

| # | Result | Evidence |
|---|---|---|
| A1 | **OPEN — deliberately not tagged.** `campaign/c2` must sit on the commit C2 will run; the code is not final (C3 unimplemented, `n_seeds` stale). Stage C runs on a recorded commit instead. | — |
| A2 | pytest / ruff / mypy — see below | raw output |
| A3 | **PASS.** `.so` mtime 2026-07-31 > last C++ commit 2026-07-30; `isa_level=x86-64-v3`, `avx512f=0`; **2,000 random DAGs, 0 backend mismatches, 0 errors** | local |
| A4 | **PASS with one recorded finding.** No arm block overrides a host-search hyperparameter; the top-level `isalsr:` block holds only canonicaliser settings. 🔴 **10 configs still declare `n_seeds: 30`** (the five D1 suites × both methods) against §0.4a's 20. Does not affect Stage C (seeds are explicit); **must be fixed before C2** | `config_diff.md` |
| A4b | **PASS.** (i) operator set identical across arms for 70/70 problems *and* uniform across problems per method; (ii) containment holds, and a deliberately bad operator raises `AlphabetCoverageError` | `operator_sets.csv` |
| A5 | **PASS.** Seeds 0/101/102 disjoint from campaign 1…20 and top-up 21…30; render as `seed_00`, `seed_101`, `seed_102` | `seed_declaration.md` |
| A11 | **PASS.** Birthday bound stated: `10¹⁴/2⁶⁵ = 2.7×10⁻⁶` per run, `1.5×10⁻²` expected over 5,600 dedup runs. Re-evaluate against the measured `max(total_dags_explored)` from Stage C | `collision_bound.md` |
| A13 | 🔴 **Stage C PASS, campaign FAIL.** FSCRATCH **222.8k / 250.0k soft**: Stage C needs ≈7.9k and fits; C2 needs ≈45k and does not (27.2k headroom vs the ≥60k criterion). HOME **0.43 TB / 0.28 TB, 2 days grace**, of which **436 GB is `~/execs/vena`** — a different project | `storage_projection.md`, `quota_capture.txt` |

#### Stage B (jobs 1751916 udfs, 1751917 bingo, 1751918 gate)

| # | Result |
|---|---|
| B1 | **PASS. 70/70 problems resolve with the declared shapes, 70/70 carry a SymPy ground truth** — C1.5's precondition is met before Stage C, not discovered during it |
| B2 | **PASS, and genuinely two-sided.** Run 1 `engine=cpp, cpp_invoked=True`; run 2 under `ISALSR_ENGINE=python` `engine=python, cpp_invoked=False`. Asserted on **observed dispatch** (a spy on `_cpp_ext.fast_canonical_string`), not on a printed name |
| B3 | **PASS.** 65,631 real Bingo candidates: 0 `SUB`, 0 `DIV`, 0 `-`, 0 `/`; `POW` the only binary; NEG 70,622 and INV 47,821 present, so the T16 decomposition **is** reaching the canonicaliser. **Max k observed = 37** |
| B4 | **PASS on its own question** — gate 1 54,765 comparisons / 0 cross-engine mismatches, gate 2 10,000 / 0, on gcc 13.2.0 and a Picasso CPU. 🔴 **Gate 3 (round-trip) fails on 5/10,000, identically on both engines** — see below |
| B6b | **PASS.** Built on Picasso with `gcc/13.2.0`; `build_hash = 298fc1188bf1b051`, **identical to the local gcc 12.2.0 build**; `isa_level=x86-64-v3`, `avx512f=0`, imports with every module purged → portable across `sd`/`sr`/`bc`/`bl` |
| B7 | **PASS. `sbatch --test-only` exit 0 on all 42 arrays**, task counts exactly as intended |
| B9 | in flight |

#### C3 — dedup-off equivalence control (the most valuable check here)

A `nodedup` arm was added to both hosts: the wrapper stays installed and keeps
canonicalising and counting, but **suppression is disabled**, so every candidate the
baseline would evaluate is still evaluated. That isolates *"the wrapper perturbs the
search"* from *"dedup changes the search"*, which is the effect we claim.

| host | best expression | `r2_train` | `total_dags_explored` | verdict |
|---|---|---|---|---|
| **UDFS** | identical | **identical bit-for-bit** | 766 vs 771 (+0.65 %) | **PASS** |
| **Bingo** | differs (algebraically identical function) | identical (1.0) | 41,061 vs 70,640 | **criterion untestable — see below** |

**UDFS passes outright.** An outermost recorder of every `evaluate_cgraph` host-native
hash gives a **313/313 identical candidate-stream prefix** between `baseline` and
`nodedup`. The 766-vs-771 gap is wall-clock truncation, not path divergence — a
baseline-vs-baseline *repeat* only matches to 311. The wrapper does not perturb the
UDFS search.

**Bingo cannot satisfy an exact-equality criterion, because the baseline does not
reproduce itself.** Verified independently in the main tree — three identical
`--variants baseline --seeds 0` invocations:

| run | `total_dags_explored` | best expression |
|---|---|---|
| 1 | **155,449** | `x_0*((x_0 + 0.5)**2 - 0.2…)` |
| 2 | 41,023 | `x_0**3 + x_0**2 + x_0` |
| 3 | 41,049 | `x_0**3 + x_0**2 + x_0` |

A 3.8× spread in candidates explored and two different symbolic forms, at one seed.
Two mechanisms, and they compound: **(i)** the protocol budgets by **wall clock**, not
by generations (§5.4), so how far the search gets is machine-state dependent by
construction; **(ii)** floating-point and iteration-count jitter inside the LM constant
optimiser — at generation 0, with RNG-state digests bit-identical and 500 outer fitness
calls in both, the inner evaluation counts still differ.

**The residual is bounded, which is what §4.3 C3 requires when exact reproduction is
unachievable for a principled reason:** the wrapper's generation-0 perturbation is
**3 inner evaluations**, while baseline-vs-baseline self-noise is **12**. The wrapper's
effect is smaller than the noise floor of the thing it is being compared against.

**Consequence to carry into the paper, not to discover in review:** any claim of
per-seed determinism for Bingo is false. The paired design survives — both arms carry
the same noise and CPDT pools over problems, not seeds — but the response letter must
say so rather than implying seed-level reproducibility.

#### A third counter inconsistency, same root as the trajectory bug

`bingo/runner.py:427` sets the **baseline** arm's `n_total_dags = total_evals`
(`ExplicitRegression.eval_count`, which includes LM inner calls), while the wrapper arms
report `dedup.n_total` (individuals entering `_serial_eval`). The two differ by the same
**3.3–4.1×** LM inflation factor as the trajectory defect. ρ is unaffected — it is
computed within an arm from a matching numerator and denominator — but **a cross-arm
comparison of `total_dags_explored` for Bingo compares two different quantities.** Do not
put baseline and dedup-arm "DAGs explored" in the same column without reconciling them.

#### Two findings that outlive this ticket

1. 🔴 **Canonical-string completeness — 5 counterexamples.** `fcs(D) == fcs(S2D(fcs(D)))` **and** `D ≇ S2D(fcs(D))` on 5 of 10,000 generated DAGs (`k ∈ {13,15,17,18,19}`): two non-isomorphic labeled DAGs sharing a canonical string, i.e. an **unsound merge**, which biases ρ upward. Engine-independent, so present in the Python code that produced C1. The precondition artefact is ruled out — no case has an in-degree-0 CONST, and three of the five also have no VAR as an edge target. **Distinct from T15**, whose failures *raise*. Owner **T07**; blocking for Stage F, not Stage C. Write-up: `docs/md_files/changes/canonical_completeness_counterexamples.md`
2. **Bingo `trajectory.csv` counted the wrong population — fixed.** `n_dags_explored` mixed fitness-function invocations (LM-inflated 3.3–4.1×) with candidate DAGs, so the series climbed to ~110k then dropped to ~30k. **ρ was never affected** (built from `dedup.n_total`/`dedup.n_unique`; no analyzer or figure reads the column). Without the fix **C1.10 would have failed on all 420 Bingo dedup-arm tasks**.

#### Ticket criteria corrected against the code

| # | Correction |
|---|---|
| C1.16 | `PairedStats` had **no seed count at all**, so "reports `n_seeds == 3`" was unverifiable — and §6.2/§6.4's "true `N` per metric" was unrepresentable in the artefact a reviewer sees. Added `n_seeds` and a per-metric `n` (post-NaN-deletion). Legacy C1 files load with `None` — verified on a real one (`Korns-12`) |
| C1.17 | "exactly 3 rows" was wrong: `aggregate_all_metrics` emits **one row per metric**. C1's own files are 14 rows / 12 columns, so this was never true. Criterion is now 14 rows with `n == 3` |
| C1.1 | `status.json` survives a deleted `run_log.json`, so an exit-code-only check would certify that hole as success. C1.1 now also requires the run log to exist |

#### SP-1…SP-6

| # | Property | Established by |
|---|---|---|
| SP-1 | Provenance | **ESTABLISHED — by `deploy.sh`, not by `sp_probe.py`.** Two separate defects were in the way. **(a)** Deployment is `rsync` and the sync excluded `.git`, so Picasso reported a months-old hash with `-dirty` **permanently** (observed: remote `b34cded-dirty` vs local `d02106f`). Fixed by `slurm/c2_smoke/deploy.sh`, which refuses a dirty local tree, syncs `.git`, and **verifies remote HEAD and cleanliness from the remote side**: `Remote HEAD: 711a44e… dirty_files=0`. **(b)** `slurm/t04_probe/sp_probe.py`'s SP-1 compares the node's files against a **frozen provenance manifest generated for T04's probe commit `a4206b8`**, so it reports five "mismatches" that are simply this ticket's own changes; its own `node_git_head_ignored` field shows the node at `53a1c1c`, the commit actually deployed, with `synced_tree_clean: true`. The manifest must be regenerated per commit (`slurm/t04_probe/make_provenance.py`) before that probe's SP-1 means anything — filed, not silently ignored. The remote-side HEAD comparison is the stronger check and is what SP-1 rests on here |
| SP-2 | Installation freshness | **PASS.** `.so` at the site-packages path, mtime 2026-08-03 19:27 > last native commit 2026-07-30; asserted in the deploy script, not eyeballed |
| SP-3 | Engine, negative control | **PASS.** B2, both directions, on observed dispatch |
| SP-4 | Alphabet, own candidate stream | **PASS.** B3: 65,631 candidates, 0 forbidden labels |
| SP-5 | Both hosts | **PASS.** Every Stage B check ran on UDFS *and* Bingo; Stage C covers both across all 42 arrays |
| SP-6 | Counters live, not dead | B9 in flight; C1.9 asserts `n_ledger_sampled > 0`, not mere presence |

| Date | Stage | Check | Result | Evidence |
|---|---|---|---|---|
| 2026-08-03 | A | A3, A4, A4b, A5, A11, A13 | see table above | `c2_preflight/` |
| 2026-08-03 | B | B1, B2, B3, B4, B6b, B7 | PASS (B4 on cross-engine; gate 3 escalated) | `c2_preflight/stage_b/`, jobs 1751916–18 |
