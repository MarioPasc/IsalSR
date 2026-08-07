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

*(filled in as they land; see §8 for the running commit log)*

---

## 6. Verification

*(filled in)*

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
