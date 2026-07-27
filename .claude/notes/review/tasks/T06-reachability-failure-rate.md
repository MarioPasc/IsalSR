# T06 — Reachability-condition failure rate and the fallback ledger

| Field | Value |
|---|---|
| Reviewer comments closed | **R1.2** (and the undocumented timeout fallback, D10) |
| Type | New measurement / instrumentation |
| Owner | **Mario** (+ Claude Code) |
| Depends on | T01 (instrument the engine once), T07 (must agree on what the precondition *is*) |
| Blocks | T09 (a supplementary subsection), T10 |
| Status | NOT STARTED |
| Target | 2026-08-24 |

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
- **AC-8.** §7 filled.

---

## 6. Work log

_(empty — to be filled by the implementing agent)_

---

## 7. Proposed answer

### 7.1 Before / after

| Quantity | Submitted | Revised | Source |
|---|---|---|---|
| Reachability violation rate reported | **not reported** | | AC-2 |
| — before normalisation, unit-test corpus (14,841) | — | | |
| — after normalisation, unit-test corpus | — | | |
| — before normalisation, evolved DAGs (UDFS) | — | | |
| — after normalisation, evolved DAGs (UDFS) | — | | |
| — before normalisation, evolved DAGs (Bingo) | — | | |
| — after normalisation, evolved DAGs (Bingo) | — | | |
| 60 s timeout rate, UDFS | stated, unquantified | | |
| 60 s timeout rate, Bingo | stated, unquantified | | |
| Conversion-failure rate | asserted zero, one problem | | |
| Canonicalisation exception rate | — | | |
| Dominant cause of violations | — | | §2 |
| Fallback behaviour | "counted as unique" (timeout only) | | AC-6 |
| False collisions observed | 0 / 14,841 + millions | | unchanged claim |

### 7.2 Changes made to the manuscript

| File | Lines (revised) | Change |
|---|---|---|
| | | |

### 7.3 Draft response text

```latex
%% --- R1.2 ---
\begin{response}
%% Structure that works here:
%%  1. Accept that the paper reported a collision rate where a violation rate was
%%     needed -- these are different quantities and the reviewer is right.
%%  2. Give the rates, before and after normalisation, on the population that
%%     matters (DAGs arriving at the canonicaliser during real searches).
%%  3. If section 2's hypothesis holds, connect it to R1.3 explicitly: the
%%     normalisation step the reviewer asks about in comment 3 is the repair that
%%     makes the precondition hold. Answering the two comments together is
%%     stronger than answering them apart.
%%  4. Give the full fallback ledger, including the 60 s timeout rate that the
%%     paper mentioned but never quantified.
%%  5. State the soundness direction: a failed DAG is evaluated, never merged
%%     wrongly. Completeness degrades gracefully; soundness does not degrade.
\changeref{}
\end{response}
```

### 7.4 Residual risk

> Candidates: a reviewer asking why the violation rate was not measured on the
> *baseline* arm too (it is engine-independent, but say so); whether the timeout
> rate is high enough to undercut the completeness claim; whether counting timed-out
> DAGs as unique biases ρ downward — it does, conservatively, and that is worth
> stating since it makes ρ a lower bound.
