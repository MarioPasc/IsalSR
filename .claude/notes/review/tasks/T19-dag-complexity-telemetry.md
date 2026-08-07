# T19 — Explored-DAG complexity telemetry

**Opened** 2026-08-07 (Mario, relaying a proposal from Ezequiel Lopez-Rubio).
**Status** in progress.
**Branch** `feature/cpp-core-port`.

---

## 1. The proposal, and what it commits us to

> *"It would be good to save the complexity of the explored DAGs per
> (method, arm, problem, seed), since we have a theory that IsalSR — and, to a
> lesser extent, Naive-Hash — explores harder, more complex DAGs due to its
> unique-DAG exploration."*

The deliverable is a per-cell block of structural statistics over the DAGs a run
actually explores, aggregated afterwards into, for each `(method, arm)` pair, a
mean with a 95 % CI, a p-value against the other two arms of the **same** method,
and an effect size. The paper gains a subsection that tests the hypothesis
instead of asserting it.

The word *complexity* is not a measurable quantity, so §3 replaces it with an
explicit descriptor vector and says what each descriptor is a proxy for.

### 1.1 The mechanism the measurement must be able to see

Worth stating, because it determines what has to be recorded and at what
cadence. Three distinct routes could make the `isalsr` arm's explored DAGs more
complex than `baseline`'s, and they are not equivalent:

1. **Budget conversion.** Every run is time-budgeted (`max_time = 43,200 s`).
   Dedup answers a repeat from `fitness_cache` instead of re-running constant
   optimisation, so the same wall clock buys more generations. Later generations
   carry structurally larger individuals (GP bloat), so the run-level mean rises
   *without the per-generation distribution changing at all*.
2. **Population enforcement.** Bingo's `isalsr` arm penalises a duplicate that is
   already in the population (`_enforce_dedup`, `penalised_in_population_*`).
   That is a genuine change to the search: the population is pushed off
   structures it already holds. This route *does* change the per-generation
   distribution.
3. **Cache-vs-recompute divergence.** A cached fitness is not always the fitness
   a recomputation would return — constant optimisation is a local method seeded
   from each individual's own constants, so two isomorphic AGraphs can score
   differently. The arms' trajectories therefore genuinely diverge rather than
   merely running at different speeds.

Routes 1 and 2 are distinguishable **only** if complexity is recorded against
generation, not just as one run-level number. That is why §4 records a
trajectory as well as a run-level aggregate.

---

## 2. What already exists (survey, 2026-08-07)

Established by three parallel read-only sweeps over the runners, the schema
layer and the core DAG API. Everything below is quoted with `file:line` because
the instrumentation points depend on it.

### 2.1 Nothing structural is recorded today

The existing per-run counters are all *cardinalities*, never *shapes*:
`n_total_dags`, `n_unique_canonical`, `n_skipped`, `n_nonstructural`,
`empirical_reduction_factor`, `redundancy_rate` (`experiments/models/schemas.py`
`SearchSpaceResults`, :118-244). Not one of them says anything about how large,
how deep, or how nonlinear the explored DAGs were. `k` appears in the paper only
for the **final** returned model, never for the explored population.

`experiments/models/stage_d_trace.py` does persist a per-candidate stream
(`candidates.jsonl`, `DEFAULT_SAMPLE_RATE = 100`, :116-118), but it is off in
every campaign run, is wired only into the `isalsr`/`hash` runners, and was built
for T04 Mode 1 replay rather than for aggregate statistics.

**Conclusion: the data does not exist and cannot be recovered post-hoc.** The
population it describes only exists while a search is running. This must land
before C2, exactly as EXECUTION-PLAN §3 requires ("anything measured *during* a
run must be in the code before launch").

### 2.2 The asymmetry that constrains the whole design

| method | arm | per-candidate hook | `LabeledDAG` built? |
|---|---|---|---|
| Bingo | baseline | **none** — `_TrajectoryEvaluation.__call__` is per-*generation* (`bingo/runner.py:159`) | **no** |
| Bingo | hash | `IsalSREvaluation._serial_eval` (`bingo/isalsr_runner.py:528`) | yes, `:560` |
| Bingo | isalsr | same, `:528` | yes, `:560` |
| UDFS | baseline | `_TrajectoryTracker...wrapped` (`udfs/runner.py:96-116`) | **no** |
| UDFS | hash | `_CanonicalDeduplicator...wrapped` (`udfs/isalsr_runner.py:423`) | yes, `:436` |
| UDFS | isalsr | same, `:423` | yes, `:436` |

**The baseline arm never builds a `LabeledDAG`.** This is the single fact that
shapes the design, and it rules out the obvious implementation.

### 2.3 Cost budget

| quantity | value | source |
|---|---|---|
| canonicalisation, Bingo | 0.82 ms/DAG median | `docs/generated/audit/numerical_audit.md:767` |
| canonicalisation, UDFS | 0.28 ms/DAG median | same |
| fitness eval, Bingo | ≈ 0.14 ms | CLAUDE.md, production table |
| fitness eval, UDFS | ≈ 19.4 ms | same |
| Bingo candidates per run | up to `max_evals = 100 M`, ≈ 5.15 h wall | EXECUTION-PLAN §11.1 |
| UDFS candidates per run | ≈ 2.2 M, 12.00 h wall | derived from the above |

A `LabeledDAG` conversion plus a structural pass costs ≈ 150–250 µs in Python.
Against Bingo's 0.14 ms fitness evaluation that is **more than 100 % overhead**
if run on every candidate.

---

## 3. Design decisions

### D1 — Reject the naive implementation

Converting every candidate in every arm is rejected. On the Bingo `baseline` arm
it would more than double per-candidate cost. Because runs are **time**-budgeted,
that cost is not merely an overhead to be subtracted afterwards — it removes
evaluations from the search, changing R², solution recovery and `n_total_dags`
for the arm that is supposed to be the control. It would also destroy the
overhead figure C2 exists to produce (EXECUTION-PLAN §2, item 6).

Subtracting the time afterwards, the way `T_search = wall_clock − canon_time_total`
does, cannot repair this: you can subtract seconds from a report, not evaluations
from a search.

### D2 — Sample, at a rate that is identical across arms

Structural descriptors are extracted from a deterministic sub-sample of the
explored stream. Comparability requires only that the **sampling rule be
identical across the three arms of a method**, which it is.

Statistical adequacy: at the rates in D3 each run contributes 10⁴–10⁶ sampled
DAGs, so the standard error of a per-run mean is two to three orders of magnitude
below the between-seed dispersion that the paired tests actually operate on. The
sample is not the limiting source of uncertainty; the seed is.

### D3 — Two sampling modes, one per method, dictated by §2.2

- **Bingo → `population`.** At each trajectory snapshot (`gen % snapshot_freq == 0`,
  default every 10 generations) the whole population is described. The insertion
  point is the *same* line in both classes — `bingo/runner.py:180` and
  `bingo/isalsr_runner.py:485` are structurally identical — so all three arms
  sample the same object at the same cadence. This is the only option that gives
  the Bingo baseline a hook at all, and it is the natural estimator: it measures
  the structural regime the search is operating in at generation *g*.
  Cost ≈ 500 individuals × 250 µs every 10 generations ≈ 0.4 % of wall clock.
- **UDFS → `stream`.** Every 127th candidate, at the per-candidate hook that all
  three arms already have. 127 is prime, so the phase cannot alias with the
  population size (500), the stack size (32) or the snapshot frequency (1000).
  Cost ≈ 17 k samples/run ≈ 0.01 % of wall clock.

`complexity_sampling_mode` is recorded per run so no reader has to guess. The
statistical comparison is *within* a method, so the mode is constant across every
contrast that is actually computed.

### D4 — Measure on the decomposed `LabeledDAG` in all three arms

Descriptors are extracted from the post-T16 decomposed `LabeledDAG` produced by
the same adapter in every arm, never from the host's native representation. A
`command_array` or `CompGraph` carries the *host's* alphabet, in which `Sub` and
`Div` are primitive; `k` would then differ between arms by construction rather
than by search behaviour. Where the arm already holds a converted DAG
(`hash`, `isalsr`), that object is reused — it is the same object a fresh
conversion would produce, so reuse changes cost and not the measurement.

### D5 — Do not touch `canonical.py` or the native engine

The WL sweep in `_compute_subtree_hashes` (`canonical.py:836-874`) already
computes out-degrees and touches every edge, so descriptors could ride along at
near-zero marginal cost. **Rejected.** It would help only the `isalsr` arm — the
one arm that needs the help least — and it would change the C++ build hash,
invalidating the engine equivalence gate, D3 hash soundness and the Stage C
certification of `2ff0050`. Under D2 the Python pass is already inside the noise.

The consequence of confining the change to Python is stated as a caveat in §7:
the `campaign/c2` tag must still move and Stage C must still be re-run, but the
**engine** certification (build hash, equivalence gate, D3) is untouched.

### D6 — Flat scalars in `SearchSpaceResults`, distributions in a sidecar

The run-level summary lands as flat scalar fields on `SearchSpaceResults`
(`schemas.py:244`), because the analyzer's `METRIC_EXTRACTORS`
(`analyzer/aggregation.py:91-105`) is a hardcoded dict of scalar lambdas — a
scalar field is usable by the existing aggregation, paired-stats and CPDT
machinery the day it lands. Full histograms go to a sibling `complexity.json` in
the seed directory, so the distributional tests are possible without inflating
either the schema or `RUN_LOG_FIELD_SPEC`.

Every new field carries a default: `RunLog.from_dict` uses bare `**d[...]`
(`schemas.py:285-292`), so an older reader would otherwise crash on a newer file.
`tests/unit/test_c2_certify.py:638` requires each new field to be declared in
`RUN_LOG_FIELD_SPEC` (`experiments/scripts/c2_certify.py:90`); that is a feature
and the entries are added deliberately.

---

## 4. The descriptor set

Thirteen descriptors, all obtained from **one** fused `O(|V| + |E|)` sweep — one
`topological_sort()` (`labeled_dag.py:364`, Kahn) followed by a single loop that
does the longest-path DP, the degree extrema and the label histogram together.
Accessors are the unchecked/raw variants (`node_label_unchecked` :177,
`in_neighbors_raw` :126, `out_neighbors_raw` :113) which return the live set with
no bounds check and no `frozenset` copy.

| # | descriptor | proxy for | anchor |
|---|---|---|---|
| 1 | `n_nodes` \|V\| | raw size | — |
| 2 | `n_edges` \|E\| | raw size | — |
| 3 | `n_var` | variables actually wired in | feature selection |
| 4 | `n_internal` = \|V\| − `n_var` | **the paper's `k`** | matches `CacheEntry` (`precomputed/cache_entry.py:54`) |
| 5 | `n_const` | free-parameter count | constant-optimisation load |
| 6 | `n_op` = `n_internal` − `n_const` | operator count | model size, SRBench (La Cava et al., NeurIPS D&B 2021) |
| 7 | `depth` | compositional nesting | Koza (1992) tree depth |
| 8 | `max_in_degree` | widest operator | arity saturation |
| 9 | `max_out_degree` | heaviest reuse | — |
| 10 | `n_shared` = #{v : outdeg ≥ 2} | **subexpression reuse** | the DAG-vs-tree quantity; GraphSR / GraphDSR |
| 11 | `sharing_surplus` = Σ max(0, outdeg−1) | edges saved vs tree unfolding | as above |
| 12 | `n_nonlinear` | transcendental / protected content | Vladislavleva et al., IEEE TEC 13(2) 2009 (order of nonlinearity); Kommenda et al., EUROCAST 2015 |
| 13 | `op_label_entropy` | operator-mix heterogeneity | Shannon (1948) |

`n_nonlinear` counts `{SIN, COS, EXP, LOG, SQRT, POW, ABS, INV, DIV}`. It is a
deliberately cheap surrogate for order-of-nonlinearity, which needs a Chebyshev
approximation per node and has no place in a hot loop. `DIV` is in the set for
legacy safety only — the decomposed alphabet must not emit it, and SP-4 checks
that.

**Headline descriptor: `n_internal` (`k`).** It is what the paper already
stratifies by, so the new subsection connects to existing tables rather than
introducing an orthogonal axis.

### 4.1 Aggregation, at constant memory

Per run, and separately for the `all` and `unique` streams:

- exact integer `sum` and `sum of squares` (Python `int`, so exact — no Welford
  needed and no floating-point drift over 10⁶ updates), `min`, `max`, `count`
  for all 13;
- an exact integer histogram, capped with an overflow bin, for the four headline
  descriptors (`n_internal`, `depth`, `n_edges`, `n_op`) — which yields exact
  medians and quantiles and makes distribution-level tests (KS, Mann-Whitney)
  possible from the sidecar alone;
- a 15-slot `NodeType` histogram, which answers "which operators does each arm
  explore" for free.

Footprint ≈ 3 KB per accumulator. Nothing is retained per candidate.

### 4.2 The `unique` stream

`hash` and `isalsr` maintain a dedup cache, so a second accumulator fed only on a
cache **miss** costs nothing extra and answers the other reading of the
hypothesis — whether the *distinct* structures visited are more complex, as
opposed to the stream as a whole. `baseline` has no cache, so these fields are
`None` there. That asymmetry is intrinsic, is documented, and is why the
`unique` block is **secondary**: the headline claim rests on the `all` stream,
which is defined identically in all three arms.

---

## 5. Changes

### 5.1 New modules

| File | Role |
|---|---|
| `src/isalsr/core/complexity.py` | `DagComplexity` (15 descriptors), `describe_dag`, `describe_dag_with_labels`, `ComplexityAccumulator`. Stdlib only, per the `isalsr.core` dependency rule |
| `experiments/models/complexity_telemetry.py` | `ComplexityTelemetry`: the sampler, the timer, the failure counter, and the two output surfaces (`scalars()` for the run log, `sidecar()` for `complexity.json`) |
| `slurm/t19_probe/{launcher,worker}.sh`, `tasks.txt`, `verify.py` | The SP-0-capped Picasso probe and its 14-gate verifier |
| `tests/unit/test_dag_complexity.py`, `tests/unit/test_complexity_telemetry.py` | 491 tests |

### 5.2 Instrumentation points

| Arm | File:line (at time of writing) | Call |
|---|---|---|
| Bingo baseline | `bingo/runner.py` `_TrajectoryEvaluation.__call__` | `_sample_population_complexity(...)` at gen 0 and at each `gen % snapshot_freq == 0` |
| Bingo hash/isalsr | `bingo/isalsr_runner.py` `IsalSREvaluation.__call__` | the **same helper, same position** |
| Bingo hash/isalsr | `bingo/isalsr_runner.py` `_serial_eval` | secondary: `observe_unique(dag)` on a sampled dedup miss, placed after the key is final and before every branch, so no `continue` can bypass it |
| UDFS baseline | `udfs/runner.py` `_TrajectoryTracker...wrapped` | `observe_converted(cgraph, compgraph_to_labeled_dag)` on the 1-in-31 grid |
| UDFS hash/isalsr | `udfs/isalsr_runner.py` `_CanonicalDeduplicator...wrapped` | sampling decided at the top (same candidate index as baseline), `observe(dag, unique=not is_duplicate)` once the duplicate verdict is known |

`_sample_population_complexity` lives in `bingo/runner.py` and is imported by
`isalsr_runner.py`, so the two arms cannot drift apart by editing one of them.

### 5.3 Persistence

Follows the existing `last_shadow` / `last_ledger` pattern exactly
(`orchestrator.py`): the runner exposes `last_complexity`, the orchestrator
splats `scalars()` into `SearchSpaceResults` via `dataclasses.replace` and writes
`sidecar()` to `complexity.json`. **No change to `RawRunResult` or to either
translator** — the pattern exists precisely so a new per-run block costs nothing
in the host-specific layer.

25 new fields on `SearchSpaceResults`, all defaulted, all declared in
`RUN_LOG_FIELD_SPEC`.

### 5.4 Analysis

Nine descriptors added to `METRIC_EXTRACTORS`, which is all it takes to reach
per-problem aggregation, the three paired contrasts and the CPDT — every one of
them iterates that dict. Five enter `CPDT_METRIC_ALTERNATIVES` and the contrast
policy, two-sided on all three arm pairs (§ rationale in the source).

The `unique` block is deliberately **not** an extractor: it is `None` on the
baseline arm, so a three-arm test on it would silently degrade to a two-arm test.

---

## 6. Verification

### 6.1 Local

| Check | Result |
|---|---|
| `tests/unit/test_dag_complexity.py` + `test_complexity_telemetry.py` | **491 passed** |
| `ruff check` on all five T19 files | clean |
| `mypy --strict src/isalsr/core/complexity.py` | clean |
| `tests/unit/` full suite | 3 pre-existing failures, none in T19 (see §6.3) |
| Six-cell end-to-end smoke, all `(method, arm)` | telemetry populated on 6/6, sidecar written 6/6, pre-existing fields and R² unchanged |
| Descriptor cost | `describe_dag` **6.8 µs**, `accumulator.observe` **8.9 µs**, against `fast_canonical_string` (native) **11.1 µs** and `agraph_to_labeled_dag` **23.8 µs** |

Isomorphism invariance is the test that matters most: the adapters number nodes
in host order, so a descriptor that moved under relabelling would make every
cross-arm comparison meaningless. Asserted as an identical tuple over **all** k!
permutations of five hand-built DAGs plus 25 random DAGs × 20 relabellings.

### 6.2 Picasso probe — array `1814948`, **14/14 GO**

24 tasks (2 problems × 2 seeds × 3 arms × 2 methods), `max_time = 900 s`, seeds
0 and 101, `--constraint=sr` (AMD EPYC 7H12), output under
`~/execs/isalsr/t19_probe/`. **24/24 COMPLETED, 0 failed.**

| gate | verdict | evidence |
|---|---|---|
| G1 cell count | PASS | 24/24 `run_log.json` |
| G2 full factorial | PASS | every `(method, arm, problem, seed)` present |
| G3 sidecar written | PASS | 24/24 `complexity.json` |
| **G4 pre-T19 fields intact** | **PASS** | no pre-existing field lost or nulled |
| **G5 frozen field spec** | **PASS** | 86 fields × 24 cells, types and nullability |
| G6 telemetry fired | PASS | every cell sampled > 0 DAGs |
| G7 descriptors finite | PASS | 9 descriptors × 24 cells |
| G8 sampling mode | PASS | bingo = population, udfs = stream |
| **G9 identical rule across arms** | **PASS** | bingo = 25, udfs = 31, no raggedness |
| G10 unique block placement | PASS | on hash + isalsr, `None` on baseline |
| G11 zero describe failures | PASS | 0 across all cells |
| G12 overhead | PASS | see §6.2a |
| G13 schema coverage | PASS | 25 fields, both directions |
| **G14 SP-4 alphabet** | **PASS** | 0 `SUB`, 0 `DIV` over every sampled node |

G4, G5 and G9 are the ones that matter. G4/G5 say the campaign's existing
record is undisturbed — a probe proving the new block works while the old
fields regressed would be worth nothing. G9 says the three arms sampled under
the same rule, which is what makes an arm-versus-arm contrast a contrast on the
search rather than on the instrument. G14 confirms the decomposed alphabet on
the probe's own candidate stream rather than in a unit test.

SP-1..SP-3 from the compute node: engine `cpp`, **`build_hash
298fc1188bf1b051` unchanged** (D5 held — no C++ was touched), `.so` mtime
2026-08-06, `complexity.py` SHA-256 `7bc4829b30b6a5ef` matching the local file
byte for byte.

### 6.2a Overhead — the honest number is higher than the local smoke suggested

⚠ **Correcting an earlier figure in this document.** The 40 s local smoke gave
≈0.2 %; the probe does not support that as the campaign figure.

Measured `complexity_time_s / wall_clock_total_s`:

| cells | overhead |
|---|---|
| UDFS (all 12) | **0.001 – 0.003 %** |
| Bingo, longest cell (1,025 generations) | **0.69 %** |
| Bingo, shortest cells (~100 generations) | **1.87 – 1.96 %** |

The spread is entirely an amortisation artefact, not a scaling problem: the
generation-0 population sample is a fixed 500-DAG cost, so a run that converges
in three seconds charges it against almost no wall clock. The ratio is
asymptotically flat, because both the sample count and the wall clock grow
linearly in generations — which is why the longest cell sits at 0.69 %.

**Projection for the campaign regime.** Bingo stops on `max_evals = 100M` at
~500 evaluations per generation, i.e. ~200,000 generations, giving ~8,000
sampling events × 500 individuals × 32.7 µs ≈ **131 s against ~18,500 s, or
≈0.7 %** — matching the longest observed cell. Max absolute cost anywhere in
the probe: **2.75 s**.

**Why 0.7 % is acceptable even against a 7.4 % canonicalisation overhead:** the
cost is applied under an identical rule in all three arms, so it cancels in
every arm-versus-arm contrast, and `complexity_time_s` is recorded per run so
any absolute overhead figure can be corrected exactly rather than estimated. If
it must be reduced, raising `ISALSR_COMPLEXITY_GEN_FREQ` to 50 halves it
without any code change; generation 0 is sampled unconditionally, so no run can
be left with `n = 0`.

### 6.2b Descriptive signal — **not evidence**

A 900 s probe on two easy problems with two seeds proves nothing scientific,
and the verifier prints that caveat next to the table. It is recorded only
because a *reversed* ordering would have indicated the instrument was wrong.

Bingo, mean `k` over the sampled populations:

| problem | seed | baseline | hash | isalsr |
|---|---|---|---|---|
| Nguyen-1 | 0 | 6.53 | 8.42 | 8.94 |
| Nguyen-1 | 101 | 6.51 | 7.89 | 9.71 |
| Nguyen-10 | 0 | 9.26 | 10.47 | 14.79 |
| Nguyen-10 | 101 | 5.80 | 10.15 | 10.94 |

The pre-registered ordering `baseline ≤ hash ≤ isalsr` holds in **4/4** Bingo
cells, and on all five headline descriptors simultaneously (k, depth,
n_nonlinear, n_shared, entropy) — not only on k.

UDFS shows a much weaker effect (baseline ≈ hash, isalsr marginally above),
which is exactly what T04 predicts: UDFS's `ρ_hash = 1.0000`, so the hash arm
merges nothing and its candidate stream is the baseline's. The one visible
difference is that UDFS-isalsr sampled roughly twice as many candidates
(390–425 vs 167–242) in the same budget — the budget-conversion route of §1.1,
directly observable.

⚠ Do not quote any of these numbers. C2 replaces them.

### 6.3 Concurrency note — another agent is working in this tree

Discovered while running the suite: `experiments/scripts/c2_slot_plan.py`,
`experiments/scripts/c2_task_spec.py` and `slurm/c2_smoke/{launcher,worker}.sh`
carry a second agent's **uncommitted** work implementing SCBI-requested task
chunking. Consequences, all handled:

- The 7 failures in `test_c2_slot_plan.py` and `test_orchestrator_flags.py` are
  theirs, not T19's — confirmed by re-running against a stashed tree.
- Nothing of theirs is staged in any T19 commit.
- `slurm/t19_probe/worker.sh` decodes from a flat `tasks.txt` instead of
  `c2_task_spec.py`, so the probe does not depend on code that is mid-edit.
- The Picasso deployment was refreshed **excluding those four paths**, so their
  deployed copy survives.
- Because the T19 sources were rsynced on top of a checkout whose `HEAD` does
  not contain them, `git rev-parse HEAD` is **not** valid provenance for this
  probe. The worker therefore also records the SHA-256 of the two T19 modules
  and the dirty-path count. A probe may do this; **C2 may not** — it must deploy
  from a clean checkout at a tag (EXECUTION-PLAN §4).

---

### 6.4 The statistics the paper section needs — verified end to end

A three-arm, two-problem, three-seed UDFS tree was built locally and run through
`--postprocess only`. All nine descriptors appear in `aggregate.csv` with
mean/std/median/quartiles/n, and all three arm-pair contrast files
(`paired_stats.json`, `paired_stats_isalsr_vs_hash.json`,
`paired_stats_hash_vs_baseline.json`) carry the full per-metric record:
Shapiro-Wilk, test selection, statistic, raw and Holm-corrected p, Cohen's d
with CI, and the mean-difference CI.

The CPDT — the project's primary significance metric — was then exercised on
`complexity_mean_k` over a synthetic 70-problem set with a true effect of +0.4:

| quantity | value |
|---|---|
| N problems | 70 |
| mean Δ (isalsr − baseline) | **+0.509** |
| 95 % CI | [+0.370, +0.648] |
| p (two-sided) | **3.49 × 10⁻¹⁰** |
| Cohen's d | **0.875** [0.622, 1.192] |
| W/T/L | 54 / 0 / 16 |
| test selected | `t_one_sample` (Shapiro-Wilk p = 0.18) |
| alternative | `two-sided` — the T19 policy, applied correctly |

That is exactly the "mean ± 95 % CI, p-value and effect size against the other
arms of the same method" the request asked for, and it needed no new statistics
code.

### 6.4a 🔴 Finding — zero across-seed variance makes the *per-problem* t-test degenerate

Surfaced by the same run, and it needs a decision before the supplementary
tables are built.

UDFS's enumeration is close to deterministic, so on Nguyen-1 the complexity
descriptors came out **bit-identical across seeds 0, 101 and 102**. The paired
t-test then divides by `std_diff = 0`:

| metric | `std_diff` | statistic | `p_raw` |
|---|---|---|---|
| `r2_test` | 0.0019 | 9.04 | 0.012 |
| `empirical_reduction_factor` | 0.0053 | 196.2 | 2.6 × 10⁻⁵ |
| `complexity_mean_k` | **0.0000** | **−inf** | **0.0** |
| `complexity_mean_depth` | **0.0000** | **+inf** | **0.0** |
| `complexity_mean_op_entropy` | **0.0000** | **−inf** | **0.0** |

`p = 0.0` with `t = ±inf` is not a significant result, it is an undefined one,
and it would be indefensible in a supplementary table — the same family of
defect as T08's NaN-typeset-as-winner.

**This is pre-existing analyzer behaviour on zero-variance input, not something
T19 introduced** — but T19 makes it *likely* rather than hypothetical, because
the complexity descriptors are the metrics most prone to zero seed variance.

Mitigating facts, and why this is not a blocker:

- **The CPDT is unaffected.** It treats each problem as one observation, so its
  variance comes from the 70 problems, not from seeds. §6.4 confirms it behaves
  correctly. House policy already makes CPDT the primary test.
- The degenerate values appear only in the per-problem supplementary detail.

**Recommendation (not implemented here — it touches shared analyzer code that a
second workstream is currently in):** in `compute_paired_stats`, when
`std_diff == 0`, record `test_used = "degenerate_zero_variance"` and emit
`p_value_raw = None` rather than `0.0`, exactly as the ledger fields distinguish
"not measured" from "measured zero". Then either report the contrast
descriptively or fall back to a sign test.

## 6.5 Stated limitations — read before writing the paper section

Four, none of them fatal, all of which a reviewer could raise.

1. **Bingo's population sample includes penalised duplicates.** In the `isalsr`
   arm, `_enforce_dedup` rejects an in-population duplicate by marking it `+inf`
   and `genetic_age = 10^7` rather than removing it, so it is still a population
   member when the sample is taken. The bias is expected to be small — a
   penalised individual is by construction a *duplicate of something already in
   the population*, hence drawn from the same structural distribution — but it
   is not zero, and it applies to one arm only. **The control already exists**:
   `penalised_in_population_mean` and `_max` are recorded per run, so the
   analysis can condition on them. Do this before claiming the effect.
2. **The two estimands are not the same across methods.** Bingo measures the
   *population at generation g*; UDFS measures the *proposed candidate stream*.
   Both are legitimate, `complexity_sampling_mode` records which, and every
   contrast the analysis computes is within a method — but the two methods'
   numbers must never be pooled or directly compared.
3. **A candidate whose conversion or canonicalisation fails is dropped from the
   sample**, because there is no DAG to describe. Rates are ~0 and are measured
   independently by the T06 ledger, but `complexity_n_sampled` is the honest
   denominator and should be quoted rather than the candidate count.
4. **Route 1 and route 2 of §1.1 are not separated by the run-level mean alone.**
   A higher mean can mean "the arm reached later, bloatier generations" rather
   than "the arm's population is structurally different at matched generation".
   Bingo's per-generation sampling makes the distinction testable — compare arms
   at matched `g`, not only at run level — but the run-level scalar on its own
   does **not** establish the mechanism, and the paper must not claim it does.

---

## 7. Consequence for the C2 launch

**This change moves the certified commit.** `slurm/c2_campaign/SUBMIT_NOW.md`
records Stage C v5e GO on `2ff0050`; any commit on top of it invalidates that
certification, so before submission the `campaign/c2` tag must be re-cut and
Stage C re-run (procedure: `slurm/c2_tag_procedure.md`, EXECUTION-PLAN §4).

What is **not** invalidated, by construction of D5: the native build hash
`298fc1188bf1b051`, the engine equivalence gate, and D3 hash soundness — the
change is pure Python and touches neither `canonical.py` nor
`src/isalsr/core/native/`.

---

## 8. Commit log

*(appended per commit)*
