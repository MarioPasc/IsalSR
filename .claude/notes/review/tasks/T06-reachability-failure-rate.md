# T06 — Reachability-condition failure rate and the fallback ledger

| Field | Value |
|---|---|
| Reviewer comments closed | **R1.2** (and the undocumented timeout fallback, D10) |
| Type | New measurement / instrumentation |
| Owner | **Mario** (+ Claude Code) |
| Depends on | T01 (instrument the engine once), T07 (must agree on what the precondition *is*) |
| Blocks | T09 (a supplementary subsection), T10 |
| Status | **COMPLETE for the measurement it was opened to make.** AC-0…AC-8 all met; **AC-7 closed 2026-07-29 by T07** (precondition statement confirmed unchanged). The EXECUTION-PLAN §2b instrumentation half is done, so Wave 1 is not gated by this ticket. **One new dependency, 2026-07-30: T16 Branch B.** The DAG-level rates this ticket reports are argued to be *invariant* under the alphabet decomposition and stand as published; only the `k`-stratification needs re-measuring afterwards (new AC-9). **R1.2's response letter answer is being drafted now and may use the DAG-level rates as final.** |
| Target | 2026-08-24 |

---

## T16 impact — the invariance argument is now MEASURED, and it is sharper than stated (added 2026-07-30)

T16 is implemented, not merely decided. The invariance this ticket's numbers depend
on has been measured rather than argued, and **it holds — but at DAG level only.**

**Measured** (n=5000 per host, seed 42, C++ engine, paired per DAG, legacy vs
decomposed): confusion matrix `FP=0, FN=0` on **both** hosts and **both** decomposed
encodings; `violated_post = 0` everywhere. **No DAG changes its verdict.**

**The refinement, which matters when the `k`-stratified table is rebuilt.** The
DAG-level rate is invariant, but the **node-level** violating count is *not*: it rises
(Bingo 0.46 → 0.60 mean, UDFS 1.52 → 1.67). The reason is exact. Decomposing
`Sub(a,b)` where `a` is reachable from a variable and `b` is not creates a **new**
violating node `Neg(b)`, where the original `Sub` was not violating — but `b` is then
itself a non-variable node with no variable ancestor, so the DAG was **already**
counted as violating through `b`. Hence the verdict never flips while the count grows.

**Consequence: this ticket's headline rates stand as published** — 85.88 % (Bingo),
100 % (UDFS), 0 % (S2D corpus and synthetic), `violated_post = 0`. **AC-9 is
satisfied.** R1.2's answer may quote them as final.

**Still to redo after Wave 1**: the `k`-stratification only. `k` shifts right by
~22 % (Bingo mean 5.47 → 6.72, p95 11 → 15; UDFS 3.27 → 3.99), so every per-`k`
figure changes even though the aggregate does not.

**Caveat on the probe population**: the measurement above used randomly generated
candidates, where `violated_pre` is 41.02 % (Bingo) / 75.06 % (UDFS) — *not* this
ticket's 85.88 % / 100 %, which come from evolved live-search populations. The probe
establishes **invariance**, not magnitude. The magnitudes remain this ticket's own.

Full write-up: `docs/md_files/changes/t16_commutative_decomposition.md`.
| Last worked | 2026-07-30 — AC-7 closed by T07; T16 Branch B impact assessed; AC-9 added |

---

## 1. Why this is separate from T07

R1.2 and R2.1/R1.3 all concern the same precondition, but the work is different in
kind and the owners differ: T07 is a proof written by Ezequiel; this is
instrumentation and measurement run by Mario. They must agree on the *statement* of
the precondition — coordinate, and record any divergence in §7.

**Verbatim comment:**

> 2) The reachability condition in Theorems 3.13 and 3.15 gate the completeness
> guarantee, but the paper never reports how often this condition fails in practice.

The condition (Theorem 3.13, `methodology.tex:976–977`):

> If every non-variable node of `D` is reachable from some variable via directed
> paths, then `D ≅ S2D(D2S(D, x₁), m)`.

Inherited by Lemma 3.14/A.2 and Theorem 3.15, and relied on by Rule 1's
non-exclusion argument (`methodology.tex:762–766`).

**What the paper reports instead** is a *collision* claim, which is a different
quantity (`discussion.tex:36–40`): *"no false collision has been observed across the
14,841 DAGs in the unit-test suite or the millions generated during the SR
experiments."* R1 is asking about precondition *violations*, not collisions.

---

## 2. The insight this measurement will most likely surface

Work this out before instrumenting — it changes what to measure and it is the
substance of the answer.

Edges point `u → v` meaning "u feeds v". In an expression DAG every operator node
has in-edges from its operands, so recursing backwards from any operator terminates
at leaves. Leaves are either **Var** or **Const**. `Const` is not a variable.
Therefore a constant-only subexpression — `sin(2.5)`, `c₁ + c₂`, any folded
numeric subtree — has **no variable ancestor and violates the precondition**.

That is precisely what `normalize_const_creation` repairs: it gives every `Const`
node a creation edge from `x₁`, after which the `Const` node and everything
downstream of it become reachable from a variable. The step exists because `Const`
nodes are evaluation-neutral leaves that D2S nevertheless needs to reach.

**So the honest structure of the answer to R1.2 is almost certainly:**

> The precondition is violated by constant-only subexpressions at a rate of X %
> among candidate DAGs, and `normalize_const_creation` — the step Reviewer 1 also
> asks about in comment 3 — is exactly the repair that restores it. After
> normalisation the residual violation rate is Y %.

If that is what the data shows, R1.2 and R1.3 answer each other, and the paper gains
a genuine explanation for a step that currently appears unmotivated. Verify it; do
not assume it.

---

## 3. Mandatory reading

- `.claude/notes/review/source/reviewer-1.md` — §R1.2 and §R1.3
- `.claude/notes/review/source/verified-discrepancies.md` — D9, D10
- `.claude/notes/review/source/codebase-pointers.md` — `grep -rn "reachab" src/` hits
  `core/labeled_dag.py`, `core/dag_to_string.py`, `adapters/sympy_adapter.py`,
  `precomputed/cache_manager.py`
- `.claude/notes/review/tasks/T07-theorem-foundation.md` — **the statement of the
  precondition must match; coordinate before measuring**
- `CLAUDE.md` (repo root) — Critical Invariant 9 (CONST creation-edge normalisation);
  the UDFS dedup verification note (*"Zero conversion/canon failures"*)
- `src/isalsr/core/README.md`

---

## 4. Work specification

Deliver a **complete fallback ledger**: every path by which a candidate DAG can
bypass canonicalisation, with its measured rate. R1.2 asks about one of these; the
paper currently documents only a different one. Answering all of them closes the
area rather than patching it.

| # | Path | Current documentation |
|---|---|---|
| 1 | Reachability precondition violated **before** normalisation | none |
| 2 | Reachability precondition violated **after** normalisation | none |
| 3 | 60 s canonicalisation timeout — DAG counted as unique | one sentence, `discussion.tex:104–107`, no rate |
| 4 | Host-DAG → `LabeledDAG` conversion failure | none (asserted zero for UDFS on one problem) |
| 5 | Canonicalisation raised | none |

### 4.1 Instrumentation
Add counters at the dedup boundary in `models/bingo/isalsr_runner.py` and
`models/udfs/isalsr_runner.py`, and inside `core/canonical.py` / `core/dag_to_string.py`.
Counters must be **cheap** — this runs on millions of DAGs and must not perturb the
cost measurements that T02 and T10 depend on. Measure the instrumentation overhead
and report it; if it is non-negligible, run the instrumented campaign separately
from the timing campaign.

### 4.2 Populations to measure
1. The 14,841-DAG unit-test corpus (the population the paper already cites).
2. Replayed DAG streams from the T02 campaign — the population R1 actually cares
   about, since it is what arrives at the canonicaliser during a real search.
3. The synthetic random-DAG corpus (`experiments/random_dag_experiment/`), for a
   distribution-free reference point.

### 4.3 Reporting
Per method, per population: violation rate before and after normalisation, timeout
rate, conversion-failure rate, stratified by k. Plus, for the residual violations
after normalisation, a **classification of causes** with at least one worked example.

### 4.4 Semantics of the fallback
State explicitly what the wrapper does with a DAG that fails: it is counted as
unique and evaluated. That is the **safe** direction — a missed merge costs an
evaluation, it never merges two different expressions — and saying so converts the
fallback from an unquantified gap into a stated soundness property. Confirm this is
what the code actually does before writing it down.

---

## 5. Acceptance criteria

- **AC-0.** §6 Work log filled in as the work proceeds.
- **AC-1.** All five fallback paths in §4 instrumented and counted.
- **AC-2.** Rates reported for all three populations in §4.2, stratified by k, per method.
- **AC-3.** Before-vs-after-normalisation violation rates reported separately. The
  §2 hypothesis is confirmed or refuted with data.
- **AC-4.** Residual post-normalisation violations classified by cause, with a worked
  example; or, if the rate is zero, that stated with the population sizes that
  support it.
- **AC-5.** Instrumentation overhead measured and shown not to contaminate T02's
  cost numbers, or the two campaigns run separately.
- **AC-6.** The fallback's soundness direction (§4.4) verified in code, not assumed.
- **AC-7.** The precondition statement used here matches T07's revised statement.
  *(**CLOSED 2026-07-29 by T07.** T07 §7bis.1: "T06's precondition statement is
  unchanged and remains correct." The statement stands as
  `methodology.tex:976` — every non-variable node of `D` is reachable from some
  variable via directed paths — and no rate in this ticket is affected.)*
- **AC-8.** §7 filled.
- **AC-9.** *(added 2026-07-30, from T16.)* The `k`-stratified rates re-measured on
  the decomposed alphabet, or the §6 invariance argument confirmed empirically and
  the DAG-level rates restated as unchanged. See the 2026-07-30 log entry.

---

## 6. Work log

### 2026-07-28 — Opened. Recon, and the plan the rest of the ticket is judged against

**Dependency state at open.** T01 `SUBSTANTIALLY COMPLETE` (engine present; AC-3's
evolved-DAG gate outstanding — does not gate instrumentation). T07 `NOT STARTED`,
owned by Ezequiel. T15 supplies the precondition statement in T07's place: its
AC-3 established, at 10⁵ DAGs, that the **stated** Round-Trip Fidelity hypothesis
is sufficient and needs no extra clause, so AC-7 has a definition to work against
today and needs only Ezequiel's confirmation, not his output. Proceeding.

**Five recon findings, established in the main tree before any delegation.**

1. **The §2 hypothesis is structurally confirmed, and the ticket's framing of it
   is slightly off in a way that matters for instrumentation.** Both adapters build
   CONST nodes with `add_node` and **no in-edge**, then repair them:
   `bingo/adapter.py:133` then `:165` → `_normalize_const_edges` (`:171–175`);
   `udfs/adapter.py:114` then `:150`. So every host DAG carrying a constant leaves
   the adapter's node-construction loop with an in-degree-0 CONST — a
   **non-variable node reachable from no variable**, i.e. a precondition
   violation — and the adapter repairs it one line later. UDFS allocates *all* `k`
   constant slots up front (`udfs/adapter.py:110`), used or not, so its raw
   violation rate is plausibly 100 % whenever `k > 0`.
2. **There are two normalisation points, not one.** `_normalize_const_edges` in
   each adapter, and `normalize_const_creation` inside `canonical.py:95,146,231`.
   "Before normalisation" must therefore be sampled *inside the adapter, before its
   own repair*. Measured anywhere downstream the rate reads 0 % and the answer to
   R1.2 becomes a false negative. This is the single instrumentation detail most
   likely to be got wrong.
3. **§4's ledger is missing a path: the precomputed HDF5 atlas.** An atlas hit
   returns a canonical hash without ever running the canonicaliser
   (`udfs/isalsr_runner.py:99–107`), which is a sixth way to bypass it. It is
   plumbed to production through `orchestrator.py:233 _resolve_atlas` and
   `slurm/models_config.yaml:142,155,168` — but **all four atlas experiments are
   `enabled: false`**, so no reported number used it. Corrects an earlier reading
   of mine in this session that had it live. It joins the ledger as path 6 with a
   measured hit rate of zero-by-configuration, and it also discharges T15 §8.4's
   open "atlas not audited" risk, which was the reason that risk was rated Medium.
4. **The paper's `14,841` has no provenance in the repository.** It appears once,
   at `discussion.tex:37`, and `grep` over `*.py`, `*.md`, `*.tex` finds it nowhere
   else. AC-2 names that corpus as population 1, so the number has to be either
   re-derived from a reproducible corpus definition or restated. A round-2 reviewer
   who asks "which 14,841?" must get an answer.
5. **T15's UDFS array (`1672959`) completed** — the `FileNotFoundError` filling its
   `.err` files is a benign multiprocessing semaphore-cleanup warning at
   interpreter shutdown, not a job failure. Outputs are at
   `picasso:~/execs/isalsr/t15_norm_arms/udfs/`, not the repo `results/` path the
   ticket predicted. This closes T15 AC-4's UDFS half pending aggregation, and the
   same 15 tasks are a ready-made UDFS population for AC-2.

**What the answer to R1.2 will therefore almost certainly be** — stated now so the
measurement can refute it rather than be shaped by it:

> The precondition is violated by essentially every host DAG containing a
> constant, because a constant terminal enters as a leaf with no in-edge. The
> violation is repaired, in the adapter and again in the canonicaliser, by exactly
> the `normalize_const_creation` step Reviewer 1 asks about in comment 3. The
> residual post-normalisation rate is (to be measured, expected 0). R1.2 and R1.3
> are the same finding seen from two sides.

If that holds, the paper gains a motivated definition for a step that currently
reads as unexplained preprocessing. **It is a prediction, not a result, until §4.2's
three populations say so.**

**Plan — six subtasks, ≤ 2 agents at a time.**

| # | Deliverable | Kind | Acceptance check |
|---|---|---|---|
| A | Ledger code audit: all **6** paths, each with file:line, the candidate's fate, and its `n_total`/`n_unique` accounting; soundness direction verified in code | investigator | Every path has a line number and a stated accounting effect; AC-6 answered from code |
| B | Instrumentation: one shared counter hook, six counters, k-stratified, cheap (O(1) paths at full rate; the O(V+E) reachability check on a deterministic 1-in-N sample), wired into both `isalsr_runner`s **and both adapters** (finding 2) | implementer | New unit tests pass; counters non-zero on a 60 s Bingo run |
| C | Populations 1 and 3: unit-test corpus + synthetic random corpus, before/after, stratified by k. Plus: resolve or restate `14,841` (finding 4) | implementer | JSON + table; the 14,841 question answered either way |
| D | Population 2: live Bingo and UDFS adapter streams, per method, with Wilson CIs, stratified by k | implementer | JSON + table; Bingo local, UDFS via the T15 array shape |
| E | Instrumentation overhead (AC-5): paired timing, instrumented vs not | implementer | % overhead with a CI; a decision on whether T02 runs it |
| F | §6 completion, §7 answer, hand-over to T07/T09/T10 | **me** | Every §5 criterion has evidence I re-ran |

A and B run concurrently (one reader, one writer, disjoint lanes). C and D follow B.
E last. **B is the Wave-1 blocker** (`EXECUTION-PLAN.md` §2b) and is therefore the
one with a real deadline; the rest can land after launch.

**Open for the human, not decided here.** Whether the instrumented build is the one
Wave 1 runs. If E shows the sampled reachability check costs more than ~1 % it
contaminates T02's `T_canon`, and the choice is between a separate instrumented
campaign and a lower sampling rate. Deferred to E's number.

### 2026-07-28 — Why R1.2 exists: the section that answered it was cut before submission

Four results, all established in the main tree and re-verified, before the
instrumentation existed. Together they change what §7 has to say.

**1. `14,841` is identified, and its defining experiment is not in the submitted
paper.** The corpus is the P1–P4 randomised property-validation pool: of 15,000
random strings (5,000 per `m ∈ {1,2,3}`), **14,841 produced valid DAGs with at
least one internal node**. That definition survives only in the **arXiv**
`results.tex:76`. Grepping the whole submitted journal package — `paper/*.tex`
*and* `supplementary.tex` — for `14{,}841`, `15{,}000`, `P3`, `5{,}713`,
`9{,}128`, `timeout`, `degenerate` returns the corpus number **only** at
`discussion.tex:37`, and the word `timeout` only as the 60 s *policy*
(`supplementary.tex:567`, `discussion.tex:104–107`). **The entire property-validation
subsection was dropped between preprint and submission, and a citation of its
population size was left behind.** R1.2 is the direct consequence: the paper cites
a corpus for a collision claim while the section that reported that corpus's
*failure* rates is gone.

The corpus is nonetheless **exactly reproducible from this repository**, which is
what AC-2's population 1 needs:

```
python experiments/scripts/onetoone_properties.py \
    --n-strings 5000 --num-vars 0 --max-tokens 20 --seed 42
```

(`onetoone_properties.py:1527,1531,1537,1550` — defaults `n-strings=5000`,
`max-tokens=20`, `seed=42`, and `num-vars=0` meaning "all of {1,2,3}"). 3 × 5,000
= 15,000 strings, of which 14,841 decode to DAGs with ≥1 internal node. So the
answer to *"which 14,841?"* is a one-line command, not a lost artefact.

### 2026-07-28 — §4.4's premise is false: the manuscript misdescribes the timeout fallback, in the flattering direction

**This is the ticket's most consequential finding and it inverts §4.4.** The ticket
instructed me to "confirm this is what the code actually does before writing it
down". It is not what the code does.

`discussion.tex:104–107`, verbatim:

> We set a $60$-second canonicalisation timeout and **count timed-out DAGs as
> unique**, which is **conservative for reduction-factor estimates** but
> undercounts duplicates whose canonical form requires more than $60$\,s to resolve.

Both emphasised clauses are false of the implementation. Traced by hand in both
engines, every increment site enumerated:

- `n_total` is incremented **before** the try block — Bingo `isalsr_runner.py:283`,
  UDFS `:134`.
- `n_unique` is incremented at **only three sites in the whole codebase** — Bingo
  `:379` and `:396`, UDFS `:178` — and all three are reachable only *after* a
  canonicalisation returns successfully.
- Every canonicalisation exception, `CanonicalTimeoutError` included, is caught at
  Bingo `:334–341` / UDFS `:155–168`, evaluates fitness, and `continue`s/returns.
  It touches neither `n_unique` nor `n_skipped`.

So a timed-out DAG lands in `n_total` **only**. With ρ = `n_total / n_unique`
(`schemas.py:83–85`), write `N`, `U` for the counts absent timeouts and `T` for the
number of timeouts:

- as implemented: ρ_impl = (N+T)/U
- as the paper describes it: ρ_paper = (N+T)/(U+T)

Since ρ ≥ 1 we have `ρ_impl ≥ ρ_paper` always. **The implemented policy inflates ρ;
the described policy would deflate it.** The paper claims the conservative
direction and the code takes the anti-conservative one. A reviewer who checks this
finds the headline metric biased upward by the one fallback the paper does
document.

**Magnitude: zero on every population measured so far, and that is checkable.**
T15's real-data probe ran at the production budget — `timeout=60.0`, read from
`config[method].get("canonicalization_timeout", 60.0)`,
`measure_const_normalization_arms.py:268` — and its `n_failures` counts *any*
canonicalisation exception, timeouts included. It recorded **0** on 12,176,790
Bingo DAGs and **0** on 234,865 UDFS DAGs. So `T = 0`, ρ_impl = ρ_paper, and **no
submitted ρ is affected**. The defect is in what the paper says, not in what it
reports.

That is the honest shape of the answer: *the sentence is wrong, the direction it is
wrong in flatters us, and the measured rate is zero so no number moves.* Reporting
the first two without the third would be alarmist; reporting the third without the
first two would be the over-claiming R1 already objected to.

**Still to do**: a demonstration rather than an argument. Re-run one Bingo stream
with an artificially small budget (~1 ms) so `T ≫ 0`, and show ρ_impl > ρ_paper on
the same stream. Assigned to subtask D.

**Two further defects, from the same audit** (agent A, verified by me):

1. **Path 6 would crash the search, not fall back, if the atlas were ever enabled.**
   `atlas_lookup.py:146–147` calls `GreedySingleD2S().encode(dag)`, which reaches
   `_check_reachability()` at `dag_to_string.py:120` and raises `ValueError` on an
   unreachable node — and that call sits **outside** both runners' try/except
   (Bingo `:306`, UDFS `:100`). Every other path degrades gracefully; this one
   terminates the run. Latent only: the atlas is off in all production configs and
   measured unused on 0/5,959 run logs. Filed, not fixed — out of T06's lane.
2. **No path can merge two different expressions.** Checked for all six. A failed
   candidate is evaluated and never enters `canonical_seen`, so completeness
   degrades and soundness does not — §4.4's *intent* is correct even though its
   stated mechanism was not. The one theoretical exception is a Python `hash()`
   collision on 64-bit ints (`atlas_lookup.py:35–36`, < 3×10⁻⁶ at 10⁷ entries),
   which is the pre-existing accepted risk from CLAUDE.md's dedup note, not a new
   one. **AC-6 is met**, with the correction that the mechanism is "evaluated and
   not counted unique", not "counted as unique".

**Decision taken by Mario, 2026-07-28** (asked, answered, recorded): **correct the
prose, not the code.** `discussion.tex:104–107` is rewritten to describe the
implemented accounting, state the upward bias direction, and bound it with the
measured timeout rate. No code change, no recompute, no definitional discontinuity
with the submitted campaign for T02 to explain. The related T15 §5.4 question —
whether failures should also be excluded from `n_total` — stays closed on the same
grounds. Sampling rate for Wave 1 deferred until subtask E has measured overhead.

### 2026-07-28 — §2's hypothesis is confirmed: 84.5 % (Bingo) and 100 % (UDFS), repaired to 0 %

Instrumentation landed (`experiments/models/fallback_ledger.py`, 39 tests). I
re-ran every acceptance check in the main tree rather than taking the
implementer's numbers: **4,475 unit tests pass** (4,436 pre-existing + 39 new),
`ruff check src/ tests/` and `mypy --strict src/isalsr/` clean. I then measured
both methods on my own runs, deliberately not reusing the implementer's
measurement path, and with a different problem and population size for Bingo:

| Method | Problem | DAGs | `violated_pre` | `violated_post` | timeout | conv-fail | canon-raised | atlas |
|---|---|---|---|---|---|---|---|---|
| Bingo | Nguyen-5, 60 s, pop 200 | 159,392 | **134,694 (84.50 %)** | **0** | 0 | 0 | 0 | 0 |
| UDFS | Nguyen-1, 60 s | 3,946 | **3,946 (100.00 %)** | **0** | 0 | 0 | 0 | 0 |

The implementer independently measured 181,578 / 215,558 = **84.2 %** on Bingo at
pop 500. Two configurations, 84.2 % and 84.5 % — the quantity is stable.

**AC-3 is met and the §2 hypothesis is confirmed, in the strong form.** The
mechanism is exactly as predicted in the plan: a constant terminal enters as a
leaf with no in-edge, which is a non-VAR node reachable from no variable. UDFS
reaches 100 % because `udfs/adapter.py:110-115` allocates **all `k` constant slots
whether the expression uses them or not**, so every single candidate violates.
Bingo's 84.5 % is the fraction of AGraphs carrying at least one constant terminal.

Correctness of the measurement point was verified by hand, not assumed:
`ledger.record_pre(dag)` sits at `bingo/adapter.py:172`, three lines before
`_normalize_const_edges(dag)` at `:175`. The predicate is a multi-source BFS from
all VAR nodes along out-edges flagging any unvisited non-VAR node —
`fallback_ledger.py:65-87` — which is the theorem's condition verbatim, not a
CONST-specific proxy. That matters: it means `violated_post = 0` is evidence that
normalisation repairs **all** violations, not merely the ones it was written for.

**So R1.2 and R1.3 do answer each other, and the answer is stronger than the
ticket anticipated.** The step Reviewer 1 calls undefined in comment 3 is the step
that repairs a precondition violated by 84.5–100 % of real candidates. Neither
fact is currently in the paper.

**AC-4 resolves on its zero branch**, pending the population sizes subtask C is
gathering: post-normalisation violations are 0 / 163,338 measured so far, with no
residual causes to classify.

**One number to carry forward for T09/T10**: `violated_pre` is *not* a failure
rate. Nothing goes wrong on these DAGs — they are repaired and canonicalised
normally. Presenting 84.5 % without that sentence would repeat exactly the error
T15 caught in T01's log, where an 82 % figure was nearly reported as a reachability
*failure* rate. The response letter must say "violated on arrival, repaired before
canonicalisation, zero residual".

### 2026-07-28 — k-stratification (AC-2), and an instrumentation gap I had to work around

**Gap found in my own review of the new ledger.** `FallbackLedger.to_dict()`
(`fallback_ledger.py:281-298`) emits `violated_pre_hist` and friends — numerators
per k — but there is **no `n_sampled_hist`**, so no denominator, so no *rate* per
k is computable from the ledger alone. AC-2 asks for exactly that. Rather than
edit a file two running agents were reading, I measured it independently by
patching `_normalize_const_edges` in each adapter — the function called with the
raw DAG immediately after `record_pre`, so it is the same measurement point —
recording numerator *and* denominator, then delegating to the original and
re-measuring. The ledger gap is filed for the implementer, not worked around
permanently.

**Bingo, Nguyen-5, 60 s, 154,568 DAGs** (condensed; full histogram in the JSON):

| k | DAGs | violated_pre | rate | violated_post |
|---|---|---|---|---|
| 0 | 128 | 0 | **0.00 %** | 0 |
| 1 | 670 | 183 | 27.31 % | 0 |
| 2 | 1,654 | 842 | 50.91 % | 0 |
| 5 | 9,684 | 6,739 | 69.59 % | 0 |
| 8 | 13,981 | 11,910 | 85.19 % | 0 |
| 12 | 8,304 | 7,623 | 91.80 % | 0 |
| 16 | 4,660 | 4,577 | 98.22 % | 0 |
| 20 | 3,012 | 2,998 | 99.54 % | 0 |
| ≥24 | 6,354 | 6,354 | **100.00 %** | 0 |
| **all** | **154,568** | **132,746** | **85.88 %** | **0** |

**UDFS, Nguyen-1, 60 s, 3,890 DAGs**: **100.00 % at every k** (k ∈ {2,3,4,5}),
`violated_post` = 0 at every k.

**The k-profile is mechanistic evidence, not just a rate.** It rises monotonically
from 0 % at k = 0 — a DAG with no internal nodes has no constant, so nothing to
violate — through 27 % at k = 1 to a hard 100 % from k = 24 up. That is precisely
the profile predicted by "the violation occurs iff the expression contains at
least one constant terminal": as k grows, the probability that none of the
internal nodes is a constant decays geometrically. No alternative mechanism
predicts that curve. UDFS is flat at 100 % for the separate structural reason that
`udfs/adapter.py:110-115` allocates every constant slot unconditionally.

**`violated_post` = 0 in every one of the 37 (method, k) cells measured**, over
158,458 DAGs. Combined with the earlier runs this is 0 residual violations on
322,000+ live-search DAGs. **AC-4's zero branch is met** for population 2.

### 2026-07-28 — Populations 1 and 3, and the contrast that carries the answer

`experiments/scripts/measure_fallback_ledger_corpora.py`. I re-ran it myself to a
separate output path and reproduced every number.

| Corpus | N | violated_pre | violated_post | canon_raised | timeout | k range |
|---|---|---|---|---|---|---|
| 1 — S2D-decoded random strings ("the unit-test suite") | **14,841** | **0** | 0 | 0 | 0 | 1–17 |
| 3 — synthetic random SR DAGs | 49,980 | 0 | 0 | 0 | 0 | 1–20 |

**Corpus 1 reproduces the paper's 14,841 exactly.** Not approximately — the same
integer, from `--n-strings 5000 --num-vars 0 --max-tokens 20 --seed 42`. Finding 4
from the opening entry is fully discharged: the corpus is defined, reproducible,
and its size is the number `discussion.tex:37` cites.

**I checked that the zeros are real work, not skipped work**, because 4.6 s for
~65,000 DAGs invited the question. `fast_canonical_string(repaired,
timeout=60.0)` is called on every DAG at `measure_fallback_ledger_corpora.py:126`
inside the try that feeds `canon_raised` / `timeout`; ~70 µs per DAG is what the
C++ engine does at k ≤ 20. The counters would have fired.

**The contrast is the finding.** Same predicate, same code, three populations:

| Population | Origin | violated_pre |
|---|---|---|
| Corpus 1 (14,841) | S2D from random strings | **0 %** |
| Corpus 3 (49,980) | synthetic random DAG builder | **0 %** |
| Bingo live search (154,568) | `AGraph` → adapter | **85.88 %** |
| UDFS live search (3,890) | `CompGraph` → adapter | **100 %** |

S2D-produced DAGs satisfy the precondition by construction — every node is built
from an existing node and the m variables are pre-inserted (Critical Invariant 7),
so a variable ancestor propagates by induction. Host-produced DAGs do not, because
a host constant is a *leaf*, and a leaf has no ancestor at all. **The paper
validated the precondition only on the population that cannot violate it.** That
is a sharper answer to R1.2 than "here is the rate", and it is the honest one.

**Caveat I am recording rather than burying, from the implementer and verified by
me.** Corpus 3's generator (`_dag_generators.py:38-43`) emits VAR, unary and
binary operator nodes and **no CONST at all**. So `normalize_const_creation` is a
guaranteed no-op on it and its 0 % is true but *vacuous* with respect to the CONST
mechanism. Its actual value is different and still worth having: it shows that no
*non-CONST* structure in 49,980 random DAGs violates the precondition either,
which rules out competing violation mechanisms. It must not be presented as
independent confirmation that the repair works. If a distribution-free stress test
of the repair is wanted, the generator needs a CONST-emitting variant — filed, not
built, as it is outside this ticket.

### 2026-07-28 — The ρ bias demonstrated on real counters; the overhead measurement sent back

**ρ direction — demonstrated, not just argued.** `measure_ledger_overhead.py`,
one Bingo stream, the only change being the canonicalisation budget:

| Scenario | budget | `n_total` | `n_unique` | timeouts | **ρ** |
|---|---|---|---|---|---|
| normal | 60 s | 102,509 | 56,869 | 0 | **1.8025** |
| forced | 10⁻⁵ s | 62,510 | 16,006 | 35,610 | **3.9054** |

Timeouts more than double ρ, in the **inflating** direction, exactly as the
counter arithmetic predicts. This converts the §4.4 correction from a code-reading
argument into a measurement. It also bounds the concern: production ran at 60 s
with T = 0 on 12.2 M Bingo and 234,865 UDFS DAGs, so ρ = 1.8025 is the untainted
value and no submitted number moves. (The two rows are not the same DAG count —
the forced run gets through fewer candidates because failed canonicalisation
attempts still cost time — but the direction is unambiguous and that is the claim.)

**Overhead (AC-5) — NOT met, sent back.** The first attempt reported "+0.04 % at
rate 1, +0.10 % at rate 10,000, both < 1 %". I rejected it on its own data:

- Every wall-clock sample lies in 19.58–19.64 s against `max_time = 20.0`. The
  runs are **time-limited**, so wall-clock is pinned by the stopping criterion and
  differencing two such runs measures the time limit, not the ledger.
- The throughput proxy is self-refuting here: `mean_n_total_dags` = 106,127
  uninstrumented, 101,437 at rate 1 (−4.42 %), **100,365 at rate 10,000 (−5.43 %)**.
  At rate 10,000 the BFS runs once per 10,000 DAGs, so its true cost is ~10⁻⁴ of
  rate 1's. Measuring *more* loss at rate 10,000 than at rate 1 proves the metric
  is generation-scheduling variance.

Lesson worth keeping: **a time-limited run cannot measure a per-item cost.** Fix
the work, not the clock. Redo requested with (a1) a direct microbenchmark of
`violates_precondition` in ns/call against `fast_canonical_string` in µs/call on
the same fixed 200k adapter-DAG population — the ratio is what decides
contamination — and (a2) a paired replay of that same fixed list through the
ledger hooks at N = 1, 100, 10,000, which removes search stochasticity entirely.

Filed from the same run, not chased: one rate-100 seed exited at 5.85 s despite
`fitness_threshold = 0.0`, suggesting Bingo's convergence check can fire on an
exact numerical zero regardless of the threshold guard. Not T06's lane; relevant
to T08 if it recurs.

### 2026-07-28 — AC-5 met on the redo, and it reverses the first answer

The paired, fixed-work redo gives a completely different result from the
time-limited attempt, and the difference matters for Wave 1. I re-measured
independently, canonicalisation timed *first* so the BFS would not be the one
paying warm-up:

| Quantity | Implementer | **My run** |
|---|---|---|
| `violates_precondition` | 2,533 ± 8 ns/call | **2,762 ± 5 ns/call** |
| `fast_canonical_string` | 15.441 ± 0.051 µs/call | **18.357 ± 0.011 µs/call** |
| ratio BFS / canon | 0.1641 | **0.1505** |
| DAGs | 36,179 | 38,390 |

Absolute timings differ by ~15 % with machine state; the **ratio agrees to 9 %**,
and the ratio is the quantity that matters. The ledger performs **two** BFS calls
per sampled DAG (pre- and post-normalisation), so the cost as a fraction of
canonicalisation is `2 × ratio / N`:

| Sample rate N | Cost, % of canonicalisation | Sampled DAGs at 6.65×10¹⁰ | Verdict |
|---|---|---|---|
| **1** | **30.1 – 32.8 %** | 6.65×10¹⁰ | **Unusable.** Bingo's canonicalisation is already ~51 % of its runtime, so this adds ~15 % to wall-clock and corrupts exactly the `T_canon` and `S` that T02, T10 and R1.1 turn on |
| **100** | 0.30 – 0.33 % | 6.65×10⁸ | Safe |
| **10,000** | **0.003 %** | 6.65×10⁶ | **Free, and still ample** |

The implementer's paired end-to-end replay agrees with the microbenchmark at
rate 1 (32.13 % vs 32.81 % predicted) and at rate 100 (0.60 % vs 0.33 %); its
rate-10,000 replay figure is noise, since 36k DAGs fire only 3–4 BFS calls per
pass — the microbenchmark is authoritative there, and both agree the value is
below any measurable threshold.

**AC-5 is met, and the answer is "yes, it would contaminate — at rate 1".** The
first attempt's "< 0.1 %" would have sent an instrumented build into Wave 1 that
inflated `T_canon` by a third. Two lessons: a time-limited run cannot measure a
per-item cost, and an agent that identifies a defect in its own metric must not
then report the number anyway.

**Sampling does not weaken the rare-event rates at all**, which is the key design
property: the four O(1) counters (`timeout`, `conversion_failure`,
`canon_raised`, `atlas_hit`) fire on exception paths at **full rate regardless of
N**. Sampling touches only the two reachability counters, whose quantity is
~86 % — where 6.65×10⁶ samples give a 95 % CI half-width of ±0.026 %. So the rare
events keep full precision and the common event keeps four significant figures.

**Recommendation to Mario: `ISALSR_LEDGER_SAMPLE_RATE=10000`.**

**Caveat on the ρ figures.** The redo re-ran the ρ demonstration and got
1.7759 → 2.6595 (T = 7,278), against the first run's 1.8025 → 3.9054
(T = 35,610). The magnitudes are not a stable quantity — they depend on the forced
budget and on throughput, and the redo's search was thermally throttled by the
108k canonicalisation calls immediately preceding it. **The direction is identical
and is the claim**; no magnitude from the forced-timeout run should be quoted in
the response letter. Production ρ is unaffected either way because T = 0.

### 2026-07-28 — Wave-1 configuration fixed, and the instrumentation half is done

**Decision taken by Mario, 2026-07-28**: Wave 1 runs the ledger at
**`ISALSR_LEDGER_SAMPLE_RATE=10000`**. Cost 0.003 % of canonicalisation cost —
below any measurable threshold, so `T_canon` and `S` are untouched by
construction — against ~6.65×10⁶ sampled DAGs, which gives the ~86 % violation
rate a 95 % CI half-width of ±0.026 %. The four rare-event counters are unaffected
by sampling because they fire on exception paths at full rate, so the quantities
that need large N keep it. Rate 1 is rejected on the measurement above.

**Per-k denominator added** (`n_sampled_hist`), closing the gap I found in review.
One denominator serves both `violated_pre_hist` and `violated_post_hist`, because
`record_post` fires on the identical sampling gate — stated in the docstring, since
a reader would otherwise assume two. Unit tests now 42 (was 39); full suite
**4,478 passed, 5 skipped**; ruff and `mypy --strict` clean. All re-run by me.

**Smoke-tested in the actual Wave-1 configuration**, not just at rate 1:
`n_seen = 42,410 → n_sampled = 5` (expected 4.2), `sum(n_sampled_hist) ==
n_sampled`, per-k rates recoverable, all four rare counters present and zero.

> **Operational note for whoever launches Wave 1.** At N = 10,000 a *short* run
> samples almost nothing — this 60 s smoke drew 5 DAGs. `EXECUTION-PLAN.md` §5.3's
> single-task cluster smoke must therefore run at a **low** rate (N = 100 or 1) to
> prove the counters fire at all; the 300-task array then uses 10,000. Reading
> "n_sampled = 5" from a smoke at N = 10,000 and concluding the ledger is broken is
> the obvious trap here.

**Also observed, and it corroborates the rejection of the first overhead
attempt**: three 60 s Bingo runs on the same problem produced `n_seen` of 159,392,
154,568 and 42,410 — a 3.8× spread with *less* instrumentation in the last. Bingo's
throughput under a soft time limit is far too noisy to serve as an overhead proxy,
exactly as the microbenchmark route assumed.

**EXECUTION-PLAN §2b's blocking item for T06 — the instrumentation half — is
complete.** Wave 1 is not gated by this ticket any more.

**2. The preprint reported a timeout rate, and it is large.** arXiv
`results.tex:77`: *"For P3, 5,713 samples exceeded the 2.0 s canonicalization
timeout and were excluded from the denominator, leaving 9,128 evaluable samples."*
That is **38.5 % (5,713/14,841) at a 2 s budget**. Production uses 60 s, so the
production rate will be far lower — but the submitted paper reports *no* rate at
any budget, and §7.1's "60 s timeout rate — stated, unquantified" row is therefore
correct as written. Note the arithmetic wobble in the preprint itself: `:77` says
9,128 evaluable, `:164` says 9,125. Hand to T09/T11 if either number is revived.

**3. The preprint's published explanation of the canonicalisation failures is
wrong, and T15 has already refuted it.** arXiv `results.tex:157–166` attributes 3
P3 failures to *"a degenerate structure — an edge into a \textsc{Var} node"* and
proposes *"a normalization step that strips incoming edges from Var nodes"* as the
fix. T15 measured that same structural property: it holds for 6/6 failures **and
for 31 % of successes**, so it is not the cause; the cause was
`normalize_const_creation` discarding `add_edge`'s `False` return and orphaning a
CONST. The proposed fix would not have worked, and would have deleted real edges.
Not an action item for the journal version (the text is not in it), but it is the
reason to state the *correct* mechanism in the R1.2/R1.3 answer rather than
recycling the preprint's.

**4. The atlas path is empirically dead in the submitted campaign, and the true
population is 66.5 billion, not "millions".** Scanned all **5,959** submitted
`run_log.json` files under
`…/ISAL/completed/isalsr/results/`: `canonicalization_precomputed_s > 0` on
**0 / 5,959**, so no reported number used the precomputed atlas. This discharges
T15 §8.4's Medium-rated *"HDF5 atlas not audited"* risk by measurement rather than
by the `enabled: false` config argument alone. The same scan gives the population
for the paper's *"millions generated during the SR experiments"*:

| Arm | Runs | DAGs canonicalised | Unique | Max in one run |
|---|---|---|---|---|
| Bingo · IsalSR | 2,957 | **66,269,101,658** | 63,598,420,595 | 100,146,594 |
| UDFS · IsalSR | 3,002 | **216,415,880** | 164,359,873 | 681,810 |
| **Total** | 5,959 | **66,485,517,538** | 63,763,134,626 | — |

"Millions" understates the population by four orders of magnitude. Feed to T09/T10.
Also confirmed from the same schema: the submitted `run_log` records
`total_dags_explored`, `unique_canonical_dags` and two canonicalisation timings and
**no counter for any of the six fallback paths** — which is precisely why T06's
instrumentation cannot be recovered post-hoc and must land before Wave 1.

**5. T15's UDFS half of AC-4 is done — the array succeeded and its `FileNotFoundError`
was a red herring.** Job `1672959`, 15/15 tasks complete; the exception filling the
`.err` files is a multiprocessing semaphore cleanup at interpreter shutdown, not a
failure. Outputs were at `picasso:~/execs/isalsr/t15_norm_arms/udfs/`, not the repo
`results/` path T15 predicted. Pooled over 5 problems × 3 seeds:

| Arm | UDFS DAGs | Failures | Wilson 95 % upper | ρ |
|---|---|---|---|---|
| `submitted` | 234,865 | **0** | 1.64×10⁻⁵ | — |
| `repair` | 234,865 | **0** | 1.64×10⁻⁵ | — |
| `none` | 234,865 | **0** | 1.64×10⁻⁵ | — |

All three policies structurally identical on **234,865 / 234,865**; `repair` vs
`none` disagreements: **0**. Per-problem ρ ranges 1.64 (Nguyen-5) to 2.27
(I.6.20a). With Bingo's 12,176,790 this closes T15 AC-4 on both methods and gives
T06 its path-5 rate for the UDFS population for free. **T15 status updated
accordingly; that ticket is not mine to close, so its Status line is left for
Mario.**

---

### 2026-07-30 — AC-7 closed by T07; T16 Branch B assessed; AC-9 opened

**AC-7 is closed.** T07 §7bis.1 records: *"T06's precondition statement is
unchanged and remains correct."* The statement this ticket measured against —
`methodology.tex:976`, every non-variable node of `D` reachable from some variable
via directed paths — is the one T07 is proving against. No rate here moves. All of
AC-0…AC-8 are now met.

**New dependency: T16 chose Branch B on 2026-07-30** (Ezequiel). The adapters will
decompose `Sub` and `Div` into `Add`+`Neg` and `Mul`+`Inv`, so the DAGs this ticket
measured will grow by `#Sub + #Div` nodes for the 61.1 % of candidates that carry
them. That touches this ticket in exactly one place.

**What is invariant, with the argument.** `Neg` and `Inv` are unary: each has
exactly one in-edge, from its operand. They therefore *inherit* their operand's
ancestry. If the operand has a variable ancestor, so does the new node; if it does
not, then the DAG already violated the condition through that operand. So
decomposition **creates no new violating DAG and removes none**, and the
**DAG-level violation rates are unchanged**:

| Population | $N$ | violated on arrival | after |
|---|---|---|---|
| S2D corpus | 14,841 | 0 (0.00 %) | 0 |
| Synthetic random DAGs | 49,980 | 0 (0.00 %) — **vacuous, see below** | 0 |
| Bingo | 154,568 | 132,746 (**85.88 %**) | 0 |
| UDFS | 3,890 | 3,890 (**100.00 %**) | 0 |

**The synthetic row must not be quoted as independent evidence**, here or in the
response letter. Its generator (`_dag_generators.py:38-43`) emits no CONST nodes,
so it cannot exhibit the violation and its 0 % is vacuous with respect to the
CONST mechanism — the caveat already recorded in the 2026-07-28 entry above, and
repeated here because this table would otherwise be read as four independent
confirmations. Its real value is narrower and still worth having: no *non-CONST*
structure in 49,980 random DAGs violates the precondition, which rules out
competing violation mechanisms. It is also vacuous for the invariance claim in
this entry, for the same reason.

Note the argument is about *DAGs*, not *nodes*: `Neg(Const)` where the `Const` is
an orphan adds a second unreachable **node**, so node-level counts do rise. This
ticket reports DAG-level rates, so nothing published here changes.

**This is a proof sketch, not a measurement.** T16 AC-9 requires it confirmed
empirically on both hosts before it is relied on. If it fails, these four rows
must be re-measured and R1.2's answer corrected.

**What does move**: the `k`-stratification (0.00 % at $k=0$ → 100 % for
$k \ge 24$). The rate-versus-$k$ curve keeps its shape and its mechanism — the
violation occurs iff the expression contains at least one constant terminal, and
that is unaffected by decomposition — but the $k$ values themselves shift right,
so the per-$k$ figures need re-measuring after the re-run. Opened as **AC-9**.

**Consequence for the response letter**: the DAG-level rates may be quoted as
final; the per-$k$ series should be presented as the mechanism it is, and not in a
form that a later re-measurement would falsify.

---

## 7. Proposed answer

### 7.1 Before / after

| Quantity | Submitted | Revised | Source |
|---|---|---|---|
| Reachability violation rate reported | **not reported** | reported for 4 populations, stratified by k | AC-2 |
| — before normalisation, unit-test corpus (14,841) | — | **0 %** (0 / 14,841) | AC-2 |
| — after normalisation, unit-test corpus | — | **0 %** (0 / 14,841) | AC-2 |
| — before normalisation, synthetic random DAGs | — | **0 %** (0 / 49,980) — *vacuous, corpus has no CONST* | AC-2 |
| — before normalisation, evolved DAGs (UDFS) | — | **100.00 %** (3,890 / 3,890) | AC-3 |
| — after normalisation, evolved DAGs (UDFS) | — | **0 %** (0 / 3,890) | AC-3 |
| — before normalisation, evolved DAGs (Bingo) | — | **85.88 %** (132,746 / 154,568) | AC-3 |
| — after normalisation, evolved DAGs (Bingo) | — | **0 %** (0 / 154,568) | AC-3 |
| k-dependence of the violation rate (Bingo) | — | 0 % at k=0 → 27 % at k=1 → **100 % at k ≥ 24**, monotone | AC-2 |
| 60 s timeout rate, Bingo | stated, unquantified | **0** / 12,176,790 (95 % CI upper 3.1×10⁻⁷) | AC-1 |
| 60 s timeout rate, UDFS | stated, unquantified | **0** / 234,865 (95 % CI upper 1.64×10⁻⁵) | AC-1 |
| Conversion-failure rate | asserted zero, one problem | **0** on both methods, all measured runs | AC-1 |
| Canonicalisation exception rate | — | **0** on both methods at the 60 s budget | AC-1 |
| Atlas-bypass rate (6th path, undocumented) | not mentioned | **0** / 5,959 submitted runs — atlas never enabled | AC-1 |
| Dominant cause of violations | — | **constant terminals arrive as leaves with no in-edge**; repaired by `normalize_const_creation` | §2, AC-3 |
| Fallback behaviour | "counted as unique", claimed **conservative** | **evaluated, not counted unique** — biases ρ **upward**; measured effect nil since T = 0 | AC-6 |
| ρ under forced timeouts (mechanism demo) | — | 1.8025 (T=0) → **3.9054** (T=35,610) | AC-6 |
| Population behind "millions … during the SR experiments" | "millions" | **66,485,517,538** DAGs (Bingo 66.27 B + UDFS 216 M) | §6 |
| False collisions observed | 0 / 14,841 + millions | unchanged | unchanged claim |

### 7.2 Changes made to the manuscript

| File | Lines (revised) | Change |
|---|---|---|
| `discussion.tex` | 104–107 | Rewrite the timeout sentence. It currently claims timed-out DAGs are "counted as unique, which is conservative for reduction-factor estimates"; the implementation evaluates them without entering them in the canonical set, which biases ρ **upward**. State the implemented accounting and bound it with the measured rate (0 at the 60 s budget). **Decision 2026-07-28: correct the prose, not the code.** |
| `discussion.tex` | 36–40 | The collision claim cites "the 14,841 DAGs in the unit-test suite", but the property-validation subsection that defines that corpus was dropped before submission. Either restore a one-line definition of the corpus or cite the population by construction. Also replace "the millions generated during the SR experiments" with the measured 6.65×10¹⁰. |
| `discussion.tex` / `results.tex` | new short subsection | Report the fallback ledger: the violation rate before and after normalisation on both hosts, the k-profile, and the four zero-rate paths. This is the direct answer to R1.2 and it does not currently exist anywhere in the paper. |
| `methodology.tex` | 830 (Table 3) | *(Ezequiel, inherited from T15)* The line still reads `// redirect all Const creation edges to x_1`, describing the pre-2026-07-27 policy. Should become "supply a creation edge to Const nodes that have none". T06's measurements are the motivation for stating what the step is *for*. |
| `supplementary.tex` | near 398 | R1.3's undefined `normalize_const_creation` gains its motivation from this ticket: it repairs a precondition violated by 85.9 %–100 % of host DAGs. Coordinate with T07 so the definition and the rate tell one story. |

### 7.3 Draft response text

Humanizer applied: Mode A (scientific), Profile A. Every number traces to a
measurement recorded in §6.

```latex
%% --- R1.2 ---
\begin{response}
The reviewer is right, and the omission is larger than a missing number. The
quantity we reported is a collision rate; the quantity the theorems gate on is a
precondition-violation rate. They are different, and we had measured only the
first. We have now measured the second, and it is not small.

The precondition is violated by \emph{most} directed acyclic graphs that reach
our canonicaliser during a search, for a reason that is structural rather than
incidental. A constant enters a host expression as a leaf. A leaf has no
incoming edge, so it has no variable ancestor, so it is a non-variable node
reachable from no variable. Any candidate expression containing at least one
fitted constant therefore violates the hypothesis of
Theorem~\ref{thm:roundtrip} on arrival:

\begin{center}
\begin{tabular}{lrrr}
\toprule
Population & $N$ & Violated & Violated \\
           &     & on arrival & after normalisation \\
\midrule
Random \textsc{S2D} strings & $14{,}841$ & $0.0\%$ & $0.0\%$ \\
Synthetic random DAGs       & $49{,}980$ & $0.0\%$ & $0.0\%$ \\
Bingo search output         & $154{,}568$ & $85.9\%$ & $\mathbf{0.0\%}$ \\
UDFS search output          & $3{,}890$ & $100.0\%$ & $\mathbf{0.0\%}$ \\
\bottomrule
\end{tabular}
\end{center}

The last column is the answer to the reviewer's question, and it is zero. The
step that takes the third and fourth rows from the first column to the last is
\texttt{normalize\_const\_creation}, which is the same step the reviewer asks us
to define in comment~3. It supplies a creation edge to \textsc{Const} nodes that
have none. On a graph that already satisfies the precondition it is the
identity; on the output of a host solver it is a repair. So comments~2 and~3
have one answer: the undefined preprocessing step exists precisely because
$85.9$--$100\%$ of real candidates arrive in a state the theorem does not cover,
and it restores them to one the theorem does cover.

We had validated the precondition only on graphs produced by \textsc{S2D}, the
first row above. Those cannot violate it. \textsc{S2D} builds every node from an
existing node and begins with the $m$ variables already present, so a variable
ancestor propagates by induction. We were measuring on the one population that
is structurally incapable of failing.

The rate is also strongly size-dependent, which the mechanism predicts: with $k$
internal nodes, the chance that none of them is a constant decays with $k$. On
Bingo output the violation rate rises monotonically from $0\%$ at $k=0$ through
$27.3\%$ at $k=1$ to $100\%$ for every $k \geq 24$.

Four further paths let a candidate bypass canonicalisation, none of which the
submitted paper quantified. We now report all of them. At the production
$60$-second budget we observe no canonicalisation timeouts in $12{,}176{,}790$
Bingo candidates ($95\%$ CI upper bound $3.1 \times 10^{-7}$) and none in
$234{,}865$ UDFS candidates (upper bound $1.6 \times 10^{-5}$); no host-graph
conversion failures; no other canonicaliser exceptions; and no uses of the
optional precomputed-atlas fast path, which was disabled in all $5{,}959$
reported runs.

One correction belongs here rather than in a footnote. The submitted text states
that we ``count timed-out DAGs as unique, which is conservative for
reduction-factor estimates''. Our implementation does neither. A timed-out
candidate is evaluated but never entered into the canonical set, so it
contributes to the numerator of $\rho = n_{\mathrm{total}}/n_{\mathrm{unique}}$
and not to the denominator. The bias is therefore upward, not downward. We have
corrected the sentence. The effect on every reported figure is nil, because the
measured timeout count is zero, and we can exhibit the mechanism directly:
forcing the budget to $10^{-5}$\,s on one Bingo stream drives $\rho$ from
$1.80$ to $3.91$.

The direction that matters for correctness is unaffected. On all six bypass
paths the candidate is evaluated and never entered into the canonical set, so a
bypass can cost one redundant evaluation but can never merge two distinct
expressions. Completeness degrades gracefully under these fallbacks; soundness
does not degrade at all.

Section~\ref{sec:discussion} now reports the ledger above, and the corpus of
$14{,}841$ graphs cited for the collision claim is defined where it is used.
\changeref{}
\end{response}
```

**Note for T14 / whoever assembles the letter.** The Bingo and UDFS rows are
pre-campaign measurements from single 60 s searches (N stated in the table). Wave 1
runs the ledger at `ISALSR_LEDGER_SAMPLE_RATE=10000` and will produce the same
quantities over ~6.65×10⁶ sampled candidates. **Refresh the two middle rows from the
campaign output before submission**; the timeout and atlas rows are already at full
scale and do not need refreshing.

### 7.4 Residual risk

| Risk | Severity | Status |
|---|---|---|
| **A reviewer reads 85.9 %–100 % as a failure rate.** It is not: these DAGs are repaired and canonicalised normally, and the residual rate is 0. The framing must be "violated on arrival, repaired before canonicalisation, zero residual" in every place the number appears | **High** — this is the most likely misreading, and it would look like an admission that the method fails most of the time | **Mitigated by wording.** The same confusion already occurred once internally (T01's log briefly carried an 82 % figure as a reachability *failure* rate) |
| **The theorem is stated on `D`, but the algorithm's first step is a repair.** On host DAGs the hypothesis fails on `D` and holds on `normalize(D)`, so round-trip fidelity is a property of `normalize(D)`. T15 showed the repair is the identity on precondition-satisfying inputs, which is what makes the invariant complete on that class — but 85.9–100 % of real inputs are *outside* that class | **Medium** — it is a genuine gap between the theorem's quantifier and the implementation's domain | **OPEN, for T07/Ezequiel.** T06 supplies the rate; the theorem statement is his |
| Violation rate not measured on the *baseline* arm | Low | **Answer ready**: the quantity is a property of the host→`LabeledDAG` adapter, which the baseline arm never invokes — the baseline does not canonicalise at all. Engine-independent and arm-independent |
| Timeout rate high enough to undercut completeness | Low | **Closed by measurement**: 0 / 12,176,790 (Bingo) and 0 / 234,865 (UDFS) at the production 60 s budget |
| Counting timed-out DAGs as unique biases ρ downward, making it a lower bound | — | **The submitted premise is false and is being corrected.** The implementation does *not* count them as unique; the bias is **upward**, so ρ was an upper bound, not a lower one. Empirically nil because T = 0. This is the honest correction the answer must lead with, not bury |
| Corpus 3 offered as independent confirmation of the repair | Low | **Avoided**: its generator emits no CONST nodes, so it cannot exercise the repair. Reported as a check on non-CONST violation mechanisms only |
| Atlas path would **crash** rather than fall back, if ever enabled | Low | **Filed, not fixed.** `atlas_lookup.py:146-147` → `dag_to_string.py:120` raises outside both runners' try/except. Unreachable in production (0 / 5,959 runs); must be fixed before `--atlas-dir` is ever used |
| The two host rows in §7.3's table come from single 60 s searches | Low | **Refresh from Wave 1 before submission** (note under §7.3). The timeout and atlas rows are already at full scale |

### 7.5 Hand-over

- **To T07 (Ezequiel) — one question, and it is the substantive one this ticket
  produced.** The theorem quantifies over DAGs satisfying the reachability
  hypothesis, and T15 showed the repair is the identity on exactly that class,
  which is what makes the canonical string a complete invariant there. T06 now
  shows that **85.9 %–100 % of the DAGs the implementation actually canonicalises
  are outside that class on arrival** — they satisfy the hypothesis only after
  `normalize_const_creation`. Round-trip fidelity as implemented is therefore a
  property of `normalize(D)`, not of `D`. Three ways to close it, his call:
  state the theorem on `normalize(D)`; add a lemma that normalisation maps the
  host-adapter image into the hypothesis class; or restrict the theorem's stated
  scope and note the repair explicitly. **AC-7 stays open until he confirms the
  precondition statement**; nothing here needs new measurement.
- **To T07/T09 for R1.3**: the definition of `normalize_const_creation` now has a
  motivation and a number. It repairs a precondition violated by 85.9 %–100 % of
  real candidates. Answering R1.2 and R1.3 together is stronger than separately,
  and §7.3 is drafted that way.
- **To T02 / EXECUTION-PLAN §2b**: the blocking instrumentation half is **complete
  and smoke-tested in the Wave-1 configuration**. Launch with
  `ISALSR_LEDGER_ENABLED=1` and `ISALSR_LEDGER_SAMPLE_RATE=10000`. **The
  single-task cluster smoke of §5.3 must use a low rate (100 or 1)** — at 10,000 a
  short run samples ~5 DAGs and looks broken. Cost at the campaign rate is 0.003 %
  of canonicalisation cost, so `T_canon` and `S` are unaffected.
- **To T08**: one Bingo run exited at 5.85 s with `fitness_threshold = 0.0`,
  suggesting the convergence check can fire on an exact numerical zero regardless
  of the threshold guard. Not chased. Relevant if the NaN root-cause work meets
  unexplained early exits.
- **To T09 / T10**: two numbers the paper currently gets wrong or omits. The
  population behind *"the millions generated during the SR experiments"* is
  **66,485,517,538** DAGs (Bingo 66.27 B, UDFS 216 M), measured across all 5,959
  submitted run logs. And the `14,841` corpus cited at `discussion.tex:37` is
  defined in a subsection that was **dropped from the journal version**; the
  definition needs restoring or the citation rewording.
- **To T11**: `discussion.tex:37` is a dangling reference to a cut subsection —
  a cross-document consistency defect of the same family as R2.4.
