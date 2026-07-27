# T15 — D2S failure modes: when canonicalisation raises, and how often on real data

| Field | Value |
|---|---|
| Reviewer comments closed | none directly (feeds **R1.2** via T06, **R2.1/R1.3** via T07) |
| Type | Investigation + theory |
| Owner | **Mario** (+ Claude Code); theory half needs **Ezequiel** |
| Depends on | T01 (C++ engine, for the real-data half) |
| Blocks | T06 (its violation-rate definition), T07 (the theorem's hypotheses) |
| Status | NOT STARTED |
| Target | 2026-08-17 (before Wave 1, because the real-data half needs campaign instrumentation) |
| Opened | 2026-07-27, by Mario, from a T01 finding |

---

## 1. Why this ticket exists

During T01's equivalence testing, `fast_canonical_string` raised
`RuntimeError: Fast canonical D2S: no valid operation found` on **6 of 4,000**
randomly generated DAGs (0.15 %). Two facts made this worth its own ticket rather
than a footnote.

**It is not the pruning.** The obvious hypothesis — that the greedy/WL/6-tuple
pruning discards the branch that would have completed — is **refuted**. All three
canonicalisers fail on exactly the same 6 DAGs:

| Algorithm | Pruning | Result on the 6 |
|---|---|---|
| `fast_canonical_string(mode="wl_only")` | greedy + WL, backtrack on ties | `RuntimeError` |
| `pruned_canonical_string` | 6-tuple pruning + backtracking | `RuntimeError` |
| `canonical_string` | **none** — exhaustive true lexmin | `RuntimeError` |

The exhaustive reference explores every branch and still finds no completion. The
defect is in D2S itself, not in anything layered on top of it.

**The failing DAGs satisfy the theorem's stated precondition.** `methodology.tex:976`,
Theorem (Round-Trip Fidelity), requires that *every non-variable node of D is
reachable from some variable via directed paths*. Measured over the same 4,000
DAGs:

| Predicate | Violated |
|---|---|
| reachable from **x₁ alone** | 3,295 / 4,000 (82.4 %) |
| **reachable from any variable** (the theorem's actual condition) | **0 / 4,000 (0.0 %)** |
| canonicalisation raises | 6 / 4,000 (0.15 %) |

All 6 failures satisfy the precondition. **The stated hypothesis is therefore not
sufficient** — either the theorem needs an additional hypothesis, or the
implementation diverges from the algorithm the proof describes. Either way it
touches a claim the paper makes.

> Recorded so it is not repeated: an earlier T01 note reported a "76–82 %
> reachability violation rate". That figure came from a filter requiring
> reachability from **x₁ only**, which is not the condition in the theorem. The
> true violation rate on S2D-produced DAGs is **0 %**. Reporting 82 % as a
> reachability-violation rate in the response letter would have been wrong, and
> would have invited a reviewer to ask why a method with an 80 % precondition
> failure rate works at all.

---

## 2. Mandatory reading

- `.claude/notes/review/tasks/T01-cpp-core-port.md` §7 — the entries dated
  2026-07-27 on equivalence and on the failure path
- `src/isalsr/core/canonical.py` — `_fast_step`, and the `RuntimeError` raise site
  (the "no valid operation found" path)
- `src/isalsr/core/labeled_dag.py` — `normalize_const_creation`
- `methodology.tex:970–985` (Theorem Round-Trip Fidelity) and `:1029`, `:1060`
  (the two results that invoke the reachability condition)
- Repo-root `CLAUDE.md`, Critical Invariants 6, 7 and 9

Reproduction script (already written, not committed):
`scratchpad/diagnose_failures.py` — regenerates the 6 failures deterministically
from `random.Random(31)`, `num_variables=2`.

---

## 3. Established facts

- Seed `random.Random(31)`, 4,000 generated strings, `num_variables=2` reproduces
  exactly 6 failures. They are stable and cheap to regenerate.
- Failure #0: source string `'PpV+CWVrV*VkPVccvknvinv-vrc'`.
  Nodes: `0=VAR 1=VAR 2=ADD 3=SQRT 4=MUL 5=CONST 6=COS 7=CONST 8=INV 9=SUB 10=SQRT`.
  Edges: `(0,6) (1,0) (1,2) (1,3) (1,4) (1,5) (1,7) (7,8) (8,0) (8,9) (8,10)`.
  Error: *no valid operation found. Remaining: 4 nodes, 4 edges.*
- Both engines (Python and C++) fail identically, with the same exception type and
  message — so this is not a port artefact. Verified over 4,000 DAGs in T01.
- The production runners already set `canonicalization_timeout = 60.0 s`
  (`bingo/config.py:32`, `udfs/config.py:29`), so this failure mode is distinct
  from a timeout.

### Root cause — CONFIRMED 2026-07-27, not a hypothesis

`normalize_const_creation` (Critical Invariant 9) relocates every CONST creation
edge onto node 0. That closes a cycle exactly when node 0 is already reachable
from the CONST by directed edges, and a cyclic graph leaves D2S with no legal
instruction.

Decisive evidence — the only difference between these two rows is the
normalisation step:

| Entry point | Failed |
|---|---|
| `fast_canonical_string(mode="wl_only")` (applies normalisation) | **6 / 6** |
| `_fast_canonical_d2s` (normalisation bypassed) | **0 / 6** |

Discriminating statistics over the same 4,000 DAGs:

| Property | Failures | Successes |
|---|---|---|
| A VAR node carries in-edges | **6 / 6 (100 %)** | 1,240 / 3,994 (31.0 %) |
| VAR in-edges **and** ≥ 1 CONST | **6 / 6 (100 %)** | 498 / 3,994 (12.5 %) |

Neither structural property is sufficient alone — a third of *successful* DAGs
have VAR in-edges. The sufficient condition is the cycle test; stating it
precisely remains AC-3.

The enabling precondition is that a **VAR node has in-edges**. A variable is
semantically a source, but `C`/`c` may direct an edge into one and S2D permits it:
Critical Invariant 6 forbids cycles, not in-edges on leaves.

**Full write-up, figure and per-algorithm table**:
`docs/md_files/changes/d2s_canonicalisation_failures.md`.

---

## 4. Non-goals

- Do **not** "fix" the failure by catching the exception and skipping the DAG.
  Silently discarding candidates changes ρ and the reduction factor.
- Do **not** change the canonical algorithm to make the symptom disappear before
  the mechanism is understood. T01 established that the engines agree; keep it so.
- Do **not** rewrite the theorem. Ezequiel owns `methodology.tex`; this ticket
  supplies the counterexample and the empirical rate, not the proof.

---

## 5. Work specification

### 5.1 Characterise the 6 (the "peek")
For each failing DAG produce: the source string, the node/label list, the edge
list, a rendered figure, which nodes remained unplaced, and the pointer state at
the point of failure. Identify what they have in common. Specifically test:

- Do all 6 have an in-edge into a VAR node? What is the in-degree distribution of
  VAR nodes across the failing versus succeeding populations?
- Do all 6 contain a CONST node whose normalisation would close a cycle? Instrument
  `normalize_const_creation` to report when relocating a creation edge would create
  one.
- Does the failure survive if `normalize_const_creation` is skipped? That is the
  single cleanest discriminator for the leading hypothesis.
- Is the DAG still failing after removing in-edges into VAR nodes?

### 5.2 Find the true precondition
State the weakest condition that is actually sufficient for D2S to terminate with
every node placed. Verify it by generating ≥10⁵ DAGs, partitioning by the candidate
condition, and confirming zero failures inside the satisfying class and a failure
in every violating one. A condition that merely correlates is not an answer.

### 5.3 Measure it on real data — this is the half that matters for the paper
The 0.15 % figure is on *randomly generated* DAGs, which are not what the search
produces. Measure the rate on the population that actually reaches the
canonicaliser:

- Bingo `AGraph` → `LabeledDAG` via `experiments/models/bingo/adapter.py`
- UDFS `CompGraph` → `LabeledDAG` via `experiments/models/udfs/adapter.py`

Run short searches on a spread of the 50-problem suite and count, per method:
DAGs canonicalised, failures, and the failure rate with a binomial CI. **If the
rate is zero on real data, that is the headline** and it should be stated as such
— it means the failure mode is an artefact of uniform random generation and does
not affect any reported number. If it is non-zero, every affected run must be
identified, because those candidates were silently dropped from the dedup stream.

Note the dependency: this needs the same instrumentation T06 requires, so the two
should share one hook rather than each adding their own.

### 5.4 Consequence for the reported results
Determine whether any submitted number is affected. Specifically: when
canonicalisation raises inside a production run, what does the runner do — skip
the candidate, count it as unique, or abort? Read the exception path in both
`isalsr_runner.py` files and state which. This decides whether ρ is biased.

---

## 6. Acceptance criteria

- **AC-0.** §7 Work log filled as the work proceeds.
- **AC-1.** The 6 DAGs characterised, with figures and a stated common structural
  feature — or evidence that they have none in common.
- **AC-2.** The leading hypothesis (CONST normalisation closing a cycle) is
  confirmed or refuted with a decisive test, not an argument.
- **AC-3.** A candidate sufficient precondition stated and validated on ≥10⁵ DAGs
  with zero failures inside the satisfying class.
- **AC-4.** Failure rate measured on **real** Bingo and UDFS DAGs, per method, with
  counts and a binomial CI. An honest zero is a valid and welcome result.
- **AC-5.** The exception path in both production runners documented, and a
  statement of whether ρ or the reduction factor is biased by it.
- **AC-6.** A regression test in `tests/unit/` that pins each of the 6 DAGs and its
  current behaviour, so any future change to D2S or to `normalize_const_creation`
  is caught.
- **AC-7.** §8 filled, including an explicit hand-over note to T07 (theorem
  hypotheses) and T06 (violation-rate definition).

---

## 7. Work log

_(empty — to be filled by the implementing agent)_

---

## 8. Proposed answer

### 8.1 Findings

| Question | Answer | Evidence |
|---|---|---|
| Is the failure caused by pruning? | **No** — exhaustive `canonical_string` fails identically | T01, 2026-07-27 |
| Do the failures violate the stated reachability condition? | **No** — all 6 satisfy it | T01, 2026-07-27 |
| True precondition | | AC-3 |
| Failure rate, random DAGs | 0.15 % (6 / 4,000) | T01 |
| Failure rate, real Bingo DAGs | | AC-4 |
| Failure rate, real UDFS DAGs | | AC-4 |
| Is any reported number biased? | | AC-5 |

### 8.2 Changes made to the repository

| Path | Change |
|---|---|
| | |

### 8.3 Hand-over

- **To T07** (Ezequiel): whether the Round-Trip Fidelity theorem needs an
  additional hypothesis beyond reachability-from-some-variable.
- **To T06**: the correct definition of a precondition violation, and the
  distinction between *violating the precondition* and *canonicalisation failing* —
  they are not the same event and the rates differ.

### 8.4 Residual risk

