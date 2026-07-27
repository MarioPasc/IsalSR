# T15 — D2S failure modes: when canonicalisation raises, and how often on real data

| Field | Value |
|---|---|
| Reviewer comments closed | none directly (feeds **R1.2** via T06, **R2.1/R1.3** via T07) |
| Type | Investigation + theory |
| Owner | **Mario** (+ Claude Code); theory half needs **Ezequiel** |
| Depends on | T01 (C++ engine, for the real-data half) |
| Blocks | T06 (its violation-rate definition), T07 (the theorem's hypotheses) |
| Status | **IN PROGRESS** — AC-1, AC-2, AC-5 met (root cause confirmed). **Open: AC-3 (sufficient precondition), AC-4 (rate on real data), AC-6 (regression test), AC-7.** |
| Target | 2026-08-17 (before Wave 1, because the real-data half needs campaign instrumentation) |
| Opened | 2026-07-27, by Mario, from a T01 finding |
| Last worked | 2026-07-27 — root cause confirmed, figure + write-up produced |

> **Start here if you are picking this up fresh.** §3 has the confirmed root
> cause, §2 lists every file you need with line numbers, and §7 records what was
> already tried. The two questions still open are: *what is the weakest sufficient
> precondition* (§5.2) and *does this ever happen on real search output* (§5.3).
> The second one decides whether any published number is affected — do that first.

---

## 1. Why this ticket exists

During T01's equivalence testing, `fast_canonical_string` raised
`RuntimeError: Fast canonical D2S: no valid operation found` on **6 of 4,000**
randomly generated DAGs (0.15 %). Two facts made this worth its own ticket rather
than a footnote.

**It is not the pruning.** The obvious hypothesis — that the greedy/WL/6-tuple
pruning discards the branch that would have completed — is **refuted**. Every
canonicalisation entry point fails on exactly the same 6 DAGs (`timeout = 10 s`):

| Algorithm | Pruning mechanism | Failed | Exception |
|---|---|---|---|
| `fast_canonical_string(mode="wl_only")` ← **production** | greedy on 1-WL, backtrack on ties | **6 / 6** | `RuntimeError` |
| `fast_canonical_string(mode="wl_tiebreak")` | 1-WL + 6-tuple tiebreak | **6 / 6** | `RuntimeError` |
| `fast_canonical_string(mode="tuple_only")` | 6-tuple only | **6 / 6** | `RuntimeError` |
| `pruned_canonical_string` | 6-tuple + backtracking | **6 / 6** | `RuntimeError` |
| `canonical_string` (exhaustive) | **none** — true lexmin | **6 / 6** | `RuntimeError` |
| `_fast_canonical_d2s` — **normalisation bypassed** | greedy on 1-WL | **0 / 6** | — |

The exhaustive reference explores every branch and still finds no completion, so
no pruning rule is discarding a valid one. The last row isolates the cause: it
differs from the first only by `normalize_const_creation`.

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

### Read first — the digest
| Path | Why |
|---|---|
| `docs/md_files/changes/d2s_canonicalisation_failures.md` | **The full write-up.** Root cause, the per-algorithm table, the mechanism, the ρ-bias finding. Everything below is supporting detail. |
| `docs/md_files/changes/fig_d2s_failures.png` | The 2×3 figure of all six DAGs |
| `docs/md_files/changes/fig_d2s_failures_data.json` | Machine-readable: every node, edge, source string and cycle-closing CONST per case |

### Code that produces the evidence
| Path | Why |
|---|---|
| `experiments/scripts/generate_fig_d2s_failures.py` | Regenerates the figure, the JSON and the per-algorithm table. Read `closes_cycle_after_normalisation()` — it is the cycle test the precondition should be built on. |
| `experiments/scripts/diagnose_d2s_failures.py` | Minimal deterministic reproduction of the 6 |

### Code under investigation
| Path | Why |
|---|---|
| `src/isalsr/core/canonical.py:~1163` | The `RuntimeError("no valid operation found")` raise site inside `_fast_step` |
| `src/isalsr/core/canonical.py:~231` | Where `normalize_const_creation()` is applied, guarded by `_has_const_nodes()` (also at `:95`, `:146`) |
| `src/isalsr/core/canonical.py:840,871–877` | The tie-handling: candidates sorted by invariant key, tied group explored recursively, lexmin taken. **Do not add a node-ID tiebreak** — node IDs are what isomorphism permutes. |
| `src/isalsr/core/labeled_dag.py:~641` | `normalize_const_creation`, the `for c in sorted(const_nodes): new.add_edge(0, c)` loop — the line that closes the cycle |
| `src/isalsr/core/string_to_dag.py` | Why a VAR node can acquire in-edges at all (`C`/`c` semantics) |

### Production wiring — what actually ran in the experiments
| Path | Why |
|---|---|
| `experiments/models/bingo/config.py:33`, `experiments/models/udfs/config.py:30` | `use_fast_canonical: bool = True` |
| `experiments/models/bingo/isalsr_runner.py:321–326` | The branch selecting `fast_canonical_string`; **no `mode=` is passed**, so `"wl_only"` applies |
| `experiments/models/udfs/isalsr_runner.py:113–115` | Same for UDFS |
| `experiments/models/bingo/isalsr_runner.py:335–341` | **The exception path.** Fitness is evaluated, then `continue` — the candidate never enters `canonical_seen` |
| `experiments/models/udfs/isalsr_runner.py:120–123` | UDFS's exception path: `return None` |
| `experiments/models/bingo/adapter.py`, `experiments/models/udfs/adapter.py` | `AGraph`/`CompGraph` → `LabeledDAG`. This is the conversion whose output §5.3 must sample. |

### Specification and theory
| Path | Why |
|---|---|
| `methodology.tex:970–985` | Theorem (Round-Trip Fidelity) — the reachability condition, stated verbatim in §6 of the write-up |
| `methodology.tex:1029`, `:1060` | The two results that invoke the condition |
| `methodology.tex:762–763` | The POW-operand clause of the same precondition |
| Repo-root `CLAUDE.md` | Critical Invariants **6** (cycle no-op on `C`/`c`), **7** (variables pre-inserted), **9** (`normalize_const_creation`), **10** (label-aware pruning) |
| `src/isalsr/core/README.md` | Full instruction semantics |

Manuscript root:
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/69c1637a28a81fea2badda9a/article/paper/`

### Related tickets
| Path | Why |
|---|---|
| `.claude/notes/review/tasks/T01-cpp-core-port.md` §7 | Where this was found; the 2026-07-27 entries on equivalence and the failure path |
| `.claude/notes/review/tasks/T06-reachability-failure-rate.md` | R1.2 wants this rate on the evolved distribution; share one instrumentation hook |
| `.claude/notes/review/tasks/T07-theorem-foundation.md` | Owns the theorem; receives the counterexample |
| `.claude/notes/review/tasks/EXECUTION-PLAN.md` §2b | Why instrumentation must land **before** Wave 1 |

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

### The six cases

`k` = internal nodes; `num_variables = 2` throughout. Full node/edge lists in
`fig_d2s_failures_data.json`.

| # | k | \|E\| | CONST closing a cycle | VAR with in-edges | S2D source |
|---|---|---|---|---|---|
| 0 | 9 | 11 | node 7 | node 0 | `PpV+CWVrV*VkPVccvknvinv-vrc` |
| 1 | 8 | 9 | node 3 | node 0 | `PVcVkNCcV-v*NNV+VsNV+VsP` |
| 2 | 8 | 9 | node 5 | node 0 | `v/Vrv*pvkWnvcV*cvrPVkn` |
| 3 | 11 | 12 | node 5 | node 0 | `vsPWV+vgVkWv-NViV/VcviNPvlVrWCPN` |
| **4** | **2** | **3** | node 2 | node 0 | `cNPNVkNCVr` |
| 5 | 11 | 12 | node 6 | node 0 | `VrV+Pv/vgVkNv+v-vaNvav^Wv+pc` |

**Case #4 is the minimal counterexample** — 2 internal nodes, 3 edges. Use it for
the proof and for the regression test; the others add nothing but size.

### Which canonicaliser generated every reported number

**`fast_canonical_string(mode="wl_only")`** — greedy, guided by the 1-WL subtree
hash. Verified through the whole chain: `use_fast_canonical` defaults to `True`
and is `true` in every production YAML; the runners call
`fast_canonical_string(dag, timeout=…)` **without** a `mode=` argument, so the
default `"wl_only"` applies. This matches the manuscript abstract's "greedy
algorithm guided by 1-Weisfeiler–Leman (1-WL) subtree hashing".

> **Terminology, because it is easy to conflate.** The **1-WL hash is a sort key**
> — candidates are ordered by `(label_char, WL_hash)`, taken greedily when the best
> is unique, and tied candidates are explored with lexmin taken. The **6-tuple is a
> separate pruning rule** (Critical Invariant 10), used only by
> `pruned_canonical_string` and by modes `wl_tiebreak` / `tuple_only`. Production
> `wl_only` **does not use the 6-tuple at all**. "1-WL pruning" merges two
> different mechanisms; prefer "greedy on the 1-WL sort key".

### What production does when canonicalisation raises — AC-5, resolved

Both runners catch the exception (`# noqa: BLE001`) and continue, differently:

| Method | Behaviour | Location |
|---|---|---|
| Bingo | fitness evaluated, then `continue` — candidate **never added** to `canonical_seen` | `bingo/isalsr_runner.py:335–341` |
| UDFS | `return None`, caller falls through to plain evaluation | `udfs/isalsr_runner.py:120–123` |

In both, the candidate is counted in `n_total` but never in `n_unique`. Since
**ρ = n_total / n_unique**, every failure **inflates ρ** — that is, biases it in
the direction that flatters the method. At 0.15 % the magnitude is negligible, but
the direction is why §5.3 must measure the real rate rather than assume it.

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

### 5.1 Characterise the 6 (the "peek") — **DONE 2026-07-27**
Figure, JSON and per-algorithm table produced; root cause confirmed. See §3 and
`docs/md_files/changes/d2s_canonicalisation_failures.md`.

One sub-item was **not** done and is still worth having: the *pointer state and the
set of unplaced nodes* at the moment of failure. The error message reports only a
count ("Remaining: 4 nodes, 4 edges"). Instrumenting `_fast_step` to name the
unplaced nodes would make the mechanism visible directly rather than inferred from
the cycle test.

### 5.2 Find the true precondition — **OPEN, AC-3**
State the weakest condition that is actually sufficient for D2S to terminate with
every node placed.

Starting candidate, from the confirmed mechanism:

> the reachability condition of Theorem (Round-Trip Fidelity) **and** no CONST node
> can reach node 0 by directed edges

`closes_cycle_after_normalisation()` in `generate_fig_d2s_failures.py` already
implements the second clause. Validate by generating ≥10⁵ DAGs, partitioning by the
candidate condition, and confirming **zero** failures inside the satisfying class
and a failure in every violating one. A condition that merely correlates is not an
answer — note that "a VAR has in-edges" holds for 31 % of *successful* DAGs, so it
is necessary-ish but nowhere near sufficient.

Also worth settling: is the right fix to the *condition* or to the *algorithm*?
`normalize_const_creation` anchors every CONST to node 0 unconditionally. Anchoring
instead to any node that does not close a cycle would remove the failure mode
entirely — but that changes Critical Invariant 9 and needs Ezequiel's sign-off,
because the canonical strings it produces would differ from the submitted ones.

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

### 5.4 Consequence for the reported results — **DONE 2026-07-27 (AC-5)**
Both runners catch and continue; the candidate lands in `n_total` but not
`n_unique`, so each failure inflates ρ. Detail in §3. What remains is the
*magnitude*, which depends entirely on §5.3's real-data rate.

Decide, and record the decision: should a canonicalisation failure be excluded
from `n_total` as well, so ρ is computed only over candidates it could actually
classify? That removes the upward bias but changes ρ's definition, so it needs to
be agreed rather than applied.

---

## 6. Acceptance criteria

- **AC-0.** §7 Work log filled as the work proceeds. — *partially filled, keep going*
- **AC-1. ✅ MET.** The 6 DAGs characterised, with figure and a stated common
  structural feature. `docs/md_files/changes/{d2s_canonicalisation_failures.md,
  fig_d2s_failures.png, fig_d2s_failures_data.json}`.
- **AC-2. ✅ MET.** The CONST-normalisation hypothesis is **confirmed** by a
  decisive test: bypassing `normalize_const_creation` takes the failure count from
  6/6 to 0/6, with nothing else changed.
- **AC-3. ⬜ OPEN.** A candidate sufficient precondition stated and validated on
  ≥10⁵ DAGs with zero failures inside the satisfying class. Starting candidate in
  §5.2.
- **AC-4. ⬜ OPEN — the one that matters for the paper.** Failure rate measured on
  **real** Bingo and UDFS DAGs, per method, with counts and a binomial CI. An
  honest zero is a valid and welcome result.
- **AC-5. ✅ MET.** Exception path documented for both runners; ρ is inflated by
  each failure (candidate counted in `n_total`, never in `n_unique`). §3.
- **AC-6. ⬜ OPEN.** A regression test in `tests/unit/` pinning each of the 6 DAGs
  and its current behaviour, so any future change to D2S or to
  `normalize_const_creation` is caught. Build it from case #4 plus the other five;
  the source strings and the generator seed are in §3 and in the JSON.
- **AC-7. ⬜ OPEN.** §8 filled, including hand-over notes to T07 and T06.

---

## 7. Work log

### 2026-07-27 — Opened from T01, root cause confirmed the same day

**How it surfaced.** While measuring the C++ port's equivalence (T01), a probe over
4,000 generated DAGs hit `RuntimeError: Fast canonical D2S: no valid operation
found` on 6 of them. Both engines failed identically — same exception type *and*
message — so it was never a port artefact; the C++ reproduces the Python behaviour
exactly, including on the failure path.

**What was tried, in order.**

1. *Is it pruning?* Ran all six canonicalisation entry points on the 6 DAGs.
   All five real canonicalisers fail, including `canonical_string`, which does no
   pruning at all and computes true lexmin. **Refuted.** If pruning were dropping
   the completing branch, the exhaustive search would find it.
2. *Is it the reachability condition?* Read the theorem verbatim
   (`methodology.tex:976`) and measured both predicates. All 6 failures **satisfy**
   the stated condition. **Refuted**, and it means the hypothesis is not sufficient.
3. *Structural features.* "A VAR node has in-edges" holds for 6/6 failures but also
   31 % of successes; adding "≥1 CONST" gives 6/6 vs 12.5 %. Suggestive, not
   sufficient — recorded so nobody mistakes correlation for the answer.
4. *The decisive test.* Called `_fast_canonical_d2s` directly, bypassing
   `normalize_const_creation`. **0/6 failures.** Nothing else differs. **Confirmed.**

**Dead end worth recording.** The first framing of this — carried in the T01 log
for several hours — was that these were "reachability-condition violations" at a
rate of 76–82 %. That number came from a filter requiring reachability from **x₁
alone**, which is not what the theorem says. The real violation rate under the
stated condition is **0 %**. Two lessons: D2S starts with *all m* variables
pre-inserted in the CDLL (Critical Invariant 7), so reachability from x₁
specifically was never required; and the ~80 % figure would have been actively
misleading if it had reached the response letter as an answer to R1.2.

**Incidental fix.** Building the figure exposed a latent bug in the shared helper
`node_display_label` (`experiments/scripts/generate_algorithm_overview.py`): it
returned `r"$\sqrt{}$"`, and an empty group is a matplotlib mathtext parse error,
so *any* figure containing a SQRT node crashed. Fixed to `r"$\sqrt{\ }$"`.

**Deliverables produced.** `docs/md_files/changes/d2s_canonicalisation_failures.md`
(write-up), `fig_d2s_failures.png`/`.pdf` (2×3 figure),
`fig_d2s_failures_data.json` (machine-readable cases),
`experiments/scripts/generate_fig_d2s_failures.py` (regenerates all three),
`experiments/scripts/diagnose_d2s_failures.py` (minimal reproduction).

**What I did not do, and why.** No fix was applied. The obvious one — anchoring
CONST creation edges to a node that does not close a cycle — changes Critical
Invariant 9 and therefore changes canonical strings relative to the submitted run.
That is Ezequiel's call, not an implementation detail, and it should not be taken
before §5.3 establishes whether the failure ever occurs on real search output.

**Where to start next.** §5.3 / AC-4. If the rate on real Bingo and UDFS DAGs is
zero, the whole thing is an artefact of uniform random generation, no published
number is affected, and the ticket closes as a characterised non-issue plus a
regression test. If it is non-zero, ρ is biased upward on the affected runs and
that has to be quantified before T02's numbers are reported.

---

## 8. Proposed answer

### 8.1 Findings

| Question | Answer | Evidence |
|---|---|---|
| Is the failure caused by pruning? | **No** — the exhaustive `canonical_string`, which prunes nothing, fails identically on all 6 | §1 table |
| Do the failures violate the stated reachability condition? | **No** — all 6 satisfy it; the condition is not sufficient | §6 of the write-up |
| What causes it? | **`normalize_const_creation` closes a cycle.** Bypassing it: 6/6 → 0/6 | §3 |
| Which canonicaliser ran in the experiments? | `fast_canonical_string(mode="wl_only")` — greedy on the 1-WL sort key, no 6-tuple | §3 |
| Failure rate, random DAGs | **0.15 %** (6 / 4,000) | §3 |
| Failure rate, real Bingo DAGs | | AC-4 |
| Failure rate, real UDFS DAGs | | AC-4 |
| True sufficient precondition | | AC-3 |
| Is any reported number biased? | **ρ is inflated** — failures count in `n_total`, never in `n_unique`. Magnitude depends on AC-4 | §3 |

### 8.2 Changes made to the repository

| Path | Change |
|---|---|
| `docs/md_files/changes/d2s_canonicalisation_failures.md` | New. Full write-up: root cause, per-algorithm table, mechanism, ρ bias |
| `docs/md_files/changes/fig_d2s_failures.{png,pdf}` | New. 2×3 figure of the six failing DAGs |
| `docs/md_files/changes/fig_d2s_failures_data.json` | New. Machine-readable cases |
| `experiments/scripts/generate_fig_d2s_failures.py` | New. Regenerates figure, JSON and the per-algorithm table |
| `experiments/scripts/diagnose_d2s_failures.py` | New. Minimal deterministic reproduction |
| `experiments/scripts/generate_algorithm_overview.py` | Fix: `node_display_label` returned `$\sqrt{}$`, a mathtext parse error that crashed any figure containing a SQRT node |

Commits on `feature/cpp-core-port`: `145d496`, `2d3231f`, `11a45e0`.

### 8.3 Hand-over

- **To T07** (Ezequiel): the Round-Trip Fidelity theorem's hypothesis is not
  sufficient. Six explicit counterexamples satisfy it and still fail, the smallest
  with k=2 and 3 edges (case #4, source `cNPNVkNCVr`). The theorem constrains the
  *input* DAG, but `normalize_const_creation` is applied **after** that check and
  can make the graph cyclic. Either the hypothesis gains a clause — candidate: *no
  CONST node reaches node 0 by directed edges* — or the normalisation is changed to
  pick a non-cycle-closing anchor. The second changes Critical Invariant 9 and with
  it the canonical strings, so it is a decision, not a fix.
- **To T06**: *violating the precondition* and *canonicalisation failing* are
  different events with different rates, and R1.2 asks for the second. On synthetic
  DAGs the precondition is violated 0 % of the time while canonicalisation fails
  0.15 % of the time. Do not report one as the other. T06 and T15 §5.3 need the
  same instrumentation — share one hook rather than adding two.
- **To T02 / EXECUTION-PLAN §2b**: if AC-4 finds a non-zero rate on real data, the
  counter must be in the code **before** Wave 1 launches, because the population it
  measures exists only while the campaign runs.

### 8.4 Residual risk

