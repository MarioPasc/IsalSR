# T15 — D2S failure modes: when canonicalisation raises, and how often on real data

| Field | Value |
|---|---|
| Reviewer comments closed | none directly (feeds **R1.2** via T06, **R2.1/R1.3** via T07) |
| Type | Investigation + theory + fix |
| Owner | **Mario** (+ Claude Code); theory half needs **Ezequiel** |
| Depends on | T01 (C++ engine, for the real-data half) |
| Blocks | T06 (its violation-rate definition), T07 (the theorem's hypotheses) |
| Status | **IN PROGRESS — everything Mario owns is met and re-verified against HEAD; the close decision is pending Mario.** AC-0…AC-7 met. The minimal `Const`-creation repair is shipped and confirmed in **both** engines (`labeled_dag.py:701–715`, `labeled_dag.cpp:339–375`), and the canonicaliser and `is_isomorphic` do **not** apply it in either engine (`canonical.py:192–201`/`:252–261`/`:388–397`, `canonical.cpp:409–418`, `labeled_dag.py:472–481`). AC-4's UDFS half closed 2026-07-28 from T06 (array `1672959`, 234,865 DAGs, 0 failures). **Open: AC-3′, which is Ezequiel's** (a cost/termination clause beside a correctness theorem, not a correctness gap). **Recommendation: close as DONE and move AC-3′ to T07 §7bis.2** — evidence and the counter-argument in §7's 2026-08-06 entry. **Not closed here; that is Mario's decision.** |
| Target | 2026-08-17 (before Wave 1, because the real-data half needs campaign instrumentation) |
| Opened | 2026-07-27, by Mario, from a T01 finding |
| Last worked | 2026-08-06 — re-verified against HEAD; three statements in §3, §5.2 and §7 found stale and annotated; close recommendation written |

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

### Root cause — CONFIRMED 2026-07-27, then CORRECTED the same day

> **The mechanism first recorded here was wrong in its final step, and the
> correction matters for T07.** The original text said the normalised graph
> becomes *cyclic*. It does not. `LabeledDAG.add_edge` **returns `False` and adds
> nothing** when an edge would close a cycle (`labeled_dag.py:248`), and
> `normalize_const_creation` discarded that return value. So the CONST's original
> creation edge was deleted during the copy and the replacement `0 -> c` was
> silently *refused*, leaving the CONST with **in-degree 0**. D2S cannot
> materialise a node no pointer can reach, hence "no valid operation found".
> Verified on all six: exactly one edge lost each, `topological_sort` succeeds
> every time (so never cyclic), and every one of them has ≥1 non-VAR node
> unreachable after normalisation against 0 before it.
>
> **Consequence: the theorem's hypothesis is sufficient and needs no new clause.**
> The implementation applies `normalize_const_creation` *after* the hypothesis
> holds and destroys it. The correct amendment is "the precondition must hold on
> `normalize(D)`, not on `D`" — or, as taken below, make the normalisation
> incapable of destroying it.

`normalize_const_creation` (Critical Invariant 9) relocated every CONST creation
edge onto node 0. That relocation is refused exactly when node 0 is already
reachable from the CONST by directed edges, and the CONST is then orphaned.

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

- **AC-0. ✅ MET.** §7 work log filled, including the correction to the mechanism
  first recorded and the two further defects found while fixing it.
- **AC-1. ✅ MET.** The 6 DAGs characterised, with figure and a stated common
  structural feature. `docs/md_files/changes/{d2s_canonicalisation_failures.md,
  fig_d2s_failures.png, fig_d2s_failures_data.json}`.
- **AC-2. ✅ MET.** The CONST-normalisation hypothesis is **confirmed** by a
  decisive test: bypassing `normalize_const_creation` takes the failure count from
  6/6 to 0/6, with nothing else changed.
- **AC-3. ✅ MET.** Validated on 10⁵ random S2D DAGs, all satisfying the
  reachability hypothesis: **zero** genuine canonicalisation failures under the
  repaired policy, against 48 under the old one. The 46 residual exceptions are
  `CanonicalTimeoutError` at k = 24–30 against the probe's 10 s budget, not
  correctness failures. Guards: the repair dropped an edge on 0 DAGs, and agreed
  byte-for-byte with no-normalisation on 0 disagreements.
  `experiments/scripts/validate_const_repair_synthetic.py`.
  **The stated hypothesis is therefore sufficient** — see AC-3′ below for the one
  clause that still needs Ezequiel.
- **AC-4. ✅ MET — both methods.** Bingo: 5 problems × 3 seeds × 120 s,
  **12,176,790 DAGs, 0 failures**, Wilson 95 % CI upper bound 3.1×10⁻⁷, all three
  policies structurally identical on 12,176,790 / 12,176,790.
  **UDFS (closed 2026-07-28 from T06):** array `1672959`, 15/15 tasks,
  **234,865 DAGs, 0 failures in all three arms**, Wilson 95 % upper 1.64×10⁻⁵,
  all three policies structurally identical on 234,865 / 234,865, `repair` vs
  `none` disagreements **0**. Per-problem ρ 1.64 (Nguyen-5) – 2.27 (I.6.20a).
  Outputs at `picasso:~/execs/isalsr/t15_norm_arms/udfs/` — **not** the repo
  `results/` path this ticket predicted, which is why they read as missing. The
  `FileNotFoundError` filling the `.err` files is a multiprocessing semaphore
  cleanup at interpreter shutdown, not a job failure.
- **AC-5. ✅ MET.** Exception path documented for both runners; ρ is inflated by
  each failure (candidate counted in `n_total`, never in `n_unique`). §3.
  Magnitude on real Bingo data: **exactly zero**, since there are no failures.
- **AC-6. ✅ MET.** `tests/unit/test_const_normalization_repair.py`, 30 tests,
  both engines: each of the 6 DAGs canonicalises, the repair preserves every
  edge, the precondition survives it, and the completeness counterexample is
  pinned. Twelve tests that encoded the old contract were rewritten in place.
- **AC-7. ✅ MET.** §8 filled below, with hand-over to T07, T06 and T02.

**AC-3′ — NEW, OPEN, for Ezequiel.** The reachability hypothesis is sufficient
for *success*, but not for *termination within a budget*: 46 / 100,000
precondition-satisfying DAGs (k = 24–30) exceed 10 s. Production allows 60 s and
sees none of this, but the theorem says nothing about cost. If T07 wants a
complexity claim alongside the correctness one, that is the gap.

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

### 2026-07-27 (later) — mechanism corrected, fix applied to both engines

**The recorded mechanism was wrong in its last step.** See the box in §3. The
normalised graph is never cyclic; `add_edge` refuses the cycle-closing edge and
returns `False`, which `normalize_const_creation` ignored, so the CONST lost its
only in-edge. This reframes the hand-over to T07 completely: the Round-Trip
Fidelity hypothesis is *sufficient*, and the implementation was violating it
after the fact. Ezequiel does not need to weaken or extend the theorem.

**Two further defects surfaced while fixing it**, neither previously recorded.

1. *Completeness was false as stated.* Take `{x1, x2, SIN, CONST, ADD}` with
   `x1->SIN`, `SIN->ADD`, `x2->ADD`, and the CONST hanging off either `x1` or
   `SIN`. Both satisfy the reachability hypothesis; they are **not** isomorphic
   under Definition (i)-(iv) (variable anchoring forces φ(x1)=x1, and x1's
   out-degree differs). The old relocation gave both `VkVspv+NnC`. That is a
   direct counterexample to Theorem (Fast Canonical String is a Complete
   Labeled-DAG Invariant), ⟹ direction.
2. *The relocation was not evaluation-preserving*, despite its docstring claiming
   `eval(D) == eval(normalize(D))`. On `x -> COS -> CONST` it moved the output
   sink from CONST to COS: `1.0` became `cos(1.5) = 0.0707`. Canonicalisation
   silently changed the function the DAG represents. The old
   `test_round_trip_evaluation_with_const` enshrined this as correct.

**The fix.** `normalize_const_creation` now adds `x_i -> c` **only for CONST nodes
with in-degree 0**, taking the lowest-indexed variable that does not close a
cycle, and never removes an edge. Applied identically to Python
(`labeled_dag.py`) and C++ (`native/src/labeled_dag.cpp`), per the decision to
keep the engines equivalent (T01). Header/binding docs updated.

Why this specific policy, and not a cycle-safe version of the old one: if `D`
satisfies the reachability hypothesis then no CONST has in-degree 0, so the
repair is **the identity on exactly the class the theorem quantifies over**.
Completeness therefore holds unqualified, and the theorem needs no normal-form
caveat. A cycle-safe relocation would fix totality but leave completeness broken.

**New entry point.** `_native.testing.fast_canonical_string_raw` skips
normalization, so all three policies can be scored through one shared canonical
core. Testing submodule only, not production API.

**Verification.**

| Check | Result |
|---|---|
| Six T15 cases canonicalise | 6/6, both engines, strings agree |
| Repair drops an edge on the six | 0/6 (was 1 edge lost each) |
| Precondition survives the repair | 6/6 (was 0/6) |
| Full unit suite | 4,436 passed, 5 skipped |
| Property + integration | 474 passed |
| ruff `src/ tests/`, mypy strict | clean |

**Ten tests encoded the old contract and were rewritten**, not deleted, each
stating the new contract and why it changed: 8 in `test_const_normalization.py`
(class `TestRedundancyElimination` → `TestProvenanceIsStructural`, assertions
inverted), 2 in `test_native_datastructures.py` (plus a new orphan-CONST
cross-engine case), 2 in `test_output_node_and_adapters.py` (the round-trip one
now asserts the value is *preserved*). New file
`tests/unit/test_const_normalization_repair.py`, 30 tests.

**AC-4, Bingo half — done, and the answer is zero.** Structural argument first:
neither adapter ever makes a VAR an edge target (`bingo/adapter.py:143-159`,
`udfs/adapter.py:104-147`), so `in_degree(x1) = 0` always, no CONST can reach
x1, and the old policy's relocation could never be refused. The trigger is
**impossible by construction**, not merely rare. Measured on one 120 s Nguyen-5
Bingo search: **501,500 DAGs, 0 failures in all three arms, and all three
policies produced structurally identical DAGs on 501,500/501,500** — same
`n_unique` (278,186), same ρ (1.803). So the submitted numbers are unaffected by
the policy change, and were unaffected by the defect.

**UDFS half deferred to Picasso.** UDFS checks its own `max_time` only between
order-enumeration stages, so a 20 s budget ran past 900 s locally. Submitted as
an array job instead: `slurm/t15_norm_arms_launch.sh`.

**AC-3 — validated at 10⁵, and it separates the policies.**
`experiments/scripts/validate_const_repair_synthetic.py`, 100,000 random S2D
DAGs, **all 100,000 satisfying the reachability hypothesis**, 10 s budget:

| Arm | Failures | Genuine (`no valid operation`) | Distinct strings | ρ |
|---|---|---|---|---|
| `submitted` (old) | 94 | **48** | 95,899 | 1.042 |
| `repair` (new) | 46 | **0** | 96,068 | 1.040 |
| `none` | 46 | **0** | 96,068 | 1.040 |

Guards: the repair dropped an edge on **0** DAGs, and `repair` disagreed with
`none` on **0** DAGs whose precondition holds — the identity property the
completeness argument needs, confirmed empirically at 10⁵.

**All 46 residual failures are `CanonicalTimeoutError`, not correctness
failures.** They are large DAGs (k = 24–30) exceeding the probe's 10 s budget;
production allows 60 s. Both surviving arms time out on exactly the same DAGs.
So on this population the fix takes genuine canonicalisation failures from 48 to
**zero**, and the reachability hypothesis *is* sufficient for success —
the ticket's original AC-3 suspicion that it was insufficient was an artefact of
the normalisation bug, not a property of the theorem.

The distinct-string columns are the completeness gap made quantitative: the old
policy merged **169** extra equivalence classes it should have kept apart,
inflating ρ from 1.040 to 1.042 in the flattering direction.

**AC-4, Bingo half — complete, N = 12.2 million.** 5 problems (one per tier)
× 3 seeds × 120 s, `experiments/scripts/measure_const_normalization_arms.py`:

| Arm | DAGs | Failures | 95% CI upper | ρ |
|---|---|---|---|---|
| `submitted` | 12,176,790 | **0** | < 3.1×10⁻⁷ | 1.793 |
| `repair` | 12,176,790 | **0** | < 3.1×10⁻⁷ | 1.793 |
| `none` | 12,176,790 | **0** | < 3.1×10⁻⁷ | 1.793 |

**All three policies produced structurally identical DAGs on 12,176,790 /
12,176,790** — not merely equal canonical strings, the same graphs. So on real
Bingo output the policy is provably irrelevant, exactly as the adapter argument
predicts, and **no submitted Bingo number is affected by the defect or by the
fix**. Results: `…/results/t15_norm_arms/{synthetic_100k.json,bingo_local/}`.

**Still open.**
- ~~UDFS half of AC-4~~ — **CLOSED 2026-07-28 from T06.** Array `1672959`
  completed; 234,865 DAGs, 0 failures, all three arms structurally identical on
  100 %. The adapter argument predicted zero and zero is what it measured.
- ~~The precomputed HDF5 atlas fast-path was not audited~~ — **CLOSED 2026-07-28
  from T06, by measurement.** All **5,959** submitted `run_log.json` files were
  scanned: `canonicalization_precomputed_s > 0` on **0 / 5,959**. No reported
  number used the atlas, so no atlas entry can encode the old merge into a
  published result. (The config-level argument — all four `*_atlas` experiments
  are `enabled: false` in `slurm/models_config.yaml` — agrees, but the run-log
  scan is the evidence that settles it.) The atlas remains a live *code* path and
  is carried in T06's ledger as path 6; if it is ever enabled, its entries must be
  regenerated under the repaired policy first.
- **`methodology.tex` Table 3 line 830** still describes the old relocation
  (`// redirect all Const creation edges to x_1`). Ezequiel's call; the theorem
  itself now holds as stated, which is the cheaper outcome.

### 2026-08-06 — re-verified against HEAD; three statements above are stale; close recommendation

The ticket's claims were checked line by line against the shipped code. **The
substance holds.** Where the ticket and the code disagree, the code is newer and
the code wins: T15's last dated entry is 2026-07-27, with an embedded update on
2026-07-28, while `canonical.cpp` last moved 2026-07-29, `canonical.py`
2026-07-31 and `labeled_dag.py` 2026-08-04. Everything below follows the code.

**What is confirmed in HEAD, both engines.**

| Claim | Where it lives now |
|---|---|
| The repair adds `x_i → c` **only** for `Const` nodes of in-degree 0, taking the lowest-indexed variable that does not close a cycle, and removes no edge | `src/isalsr/core/labeled_dag.py:701–715`; `native/src/labeled_dag.cpp:339–375` |
| The canonicaliser does **not** apply the repair, at any of its three entry points | `src/isalsr/core/canonical.py:192–201`, `:252–261`, `:388–397`; `native/src/canonical.cpp:409–418` |
| `is_isomorphic` does not apply it either | `src/isalsr/core/labeled_dag.py:472–481` |
| Production DAGs are repaired producer-side instead | `experiments/models/bingo/adapter.py`, `experiments/models/udfs/adapter.py`, `_normalize_const_edges` |
| Both engines refuse an unencodable input rather than silently repairing it | `canonical.py:1199`, `canonical.cpp:313` |

One mechanism detail is worth stating precisely, because it is described loosely
in several places. There is **no dedicated precondition check**. An in-degree-0
`Const` is refused because the D2S sweep runs out of legal operations, which
surfaces as the generic exhaustion error. The effect is what the ticket claims;
the mechanism is not a guard.

**Three statements above are stale and must not be quoted.**

1. **§3, the 6/6 failure table**, is framed as "`fast_canonical_string` (applies
   normalisation) → 6/6 fail". The canonicaliser has not applied normalisation
   since 2026-07-29. The table is a historical record of the diagnosis, not a
   description of current behaviour.
2. **§5.2** says the repair "anchors every `Const` to node 0 unconditionally".
   That is false of `normalize_const_creation` and has been since the fix. It is
   still true of the **adapters'** `_normalize_const_edges`, which is a different
   function with a different precondition; the two were conflated.
3. **§7's `fast_canonical_string_raw` distinction** treated "raw" as the variant
   that skips normalisation. Production skips it too, so the distinction no longer
   separates anything.

By the same token, **AC-2's and AC-3's numbers are archival**: they were measured
against a canonicaliser that applied normalisation, and that code path no longer
exists. They remain valid as evidence for the decision they justified. They cannot
be reproduced from HEAD, and nothing reader-facing should cite them as current.

**A code-hygiene defect, recorded but not fixed here.** `canonical.py:190–191`
still carries a comment saying normalisation is applied before canonical
computation, directly above the line that says it is not. Cosmetic, contradictory,
and a trap for the next reader. Left for whoever next edits that file.

**Per-item sign-off.** Read this as the review surface; nothing below needs
re-deriving.

| AC | Owner | State | Evidence, re-checked against HEAD |
|---|---|---|---|
| AC-0 | Mario | ✅ met | §7, four dated entries including this one |
| AC-1 | Mario | ✅ met | the six DAGs characterised with figure and common structural feature, `docs/md_files/changes/d2s_canonicalisation_failures.md` |
| AC-2 | Mario | ✅ met, **numbers archival** | bypassing the old normalisation took 6/6 failures to 0/6; measured against a code path that no longer exists, so do not cite it as current behaviour |
| AC-3 | Mario | ✅ met, **numbers archival** | 10⁵ random S2D DAGs, zero genuine failures under the repaired policy against 48 under the old one; 0 edges dropped, 0 disagreements against no-normalisation |
| AC-4 | Mario | ✅ met, both hosts | Bingo 12,176,790 DAGs / 0 failures, Wilson 95 % upper 3.1×10⁻⁷; UDFS 234,865 DAGs / 0 failures in all three arms, Wilson 95 % upper 1.64×10⁻⁵ |
| AC-5 | Mario | ✅ met | the refusal path is documented; note the mechanism is D2S exhaustion, not a dedicated precondition guard (`canonical.py:1199`, `canonical.cpp:313`) |
| AC-6 | Mario | ✅ met | `tests/unit/test_const_normalization_repair.py`, parametrised over both engines; twelve old-contract tests rewritten in place, none suppressed |
| AC-7 | Mario | ✅ met | §8 filled, with hand-over to T07, T06 and T02 |
| **AC-3′** | **Ezequiel** | ⬜ **open** | termination within a budget: 46 of 100,000 precondition-satisfying DAGs at k = 24–30 exceed the probe's 10 s; production allows 60 s and sees none |
| Status / close | **Mario** | ⬜ **pending** | recommendation below; not decided here |

**Recommendation on the close decision — Mario's call, not this entry's.**

**Recommended: close T15 as DONE and move AC-3′ to T07 §7bis.2.** The evidence:

- The ticket's own deliverables are all discharged. The root cause is confirmed
  and corrected in `§7`, the six cases are characterised, the fix is shipped in
  both engines, and the real-data half is measured on both hosts
  (12,176,790 Bingo DAGs and 234,865 UDFS DAGs, zero failures in every arm).
- Its outputs have already been consumed downstream and have shipped. T06 took the
  violation-rate definition, T07 took the hypothesis, and both R1.2 and R1.3 are
  written into `reviews/response_to_reviewers.tex` on the strength of them.
- AC-3′ is not a correctness gap. It asks for a cost clause beside a correctness
  theorem: the reachability hypothesis is sufficient for success but says nothing
  about time, and 46 of 100,000 synthetic DAGs at k = 24–30 exceed a ten-second
  probe budget. That is a statement about a theorem, and the theorems are T07's.
- The reviewer-facing obligation AC-3′ creates is **already discharged**. R2.1
  volunteers the termination-versus-cost limitation in the letter, with the 46 of
  100,000 figure and the 10,286,517 Bingo and 265,092 UDFS candidates that meet the
  sixty-second production budget without a single failure. Keeping T15 open no
  longer protects anything the reviewers will see.
- The counter-argument, which is why this is a recommendation and not a decision:
  moving an open item across tickets is how items get lost, and T07's §7bis is
  already carrying six things for Ezequiel. If the preference is to keep the owner
  and the item together, leave T15 open at **IN PROGRESS** and close it when
  Ezequiel signs off; the cost of that is only that a ticket with no Mario-side
  work sits open for a few weeks.

Whichever way it goes, the three stale statements above should be struck or
annotated before anyone reads this ticket cold.

---

## 8. Proposed answer

### 8.1 Findings

| Question | Answer | Evidence |
|---|---|---|
| Is the failure caused by pruning? | **No** — the exhaustive `canonical_string`, which prunes nothing, fails identically on all 6 | §1 table |
| Do the failures violate the stated reachability condition? | **Not on input.** All 6 satisfy it; `normalize_const_creation` then destroyed it | §3 |
| What causes it? | **`normalize_const_creation` dropped an edge.** `add_edge` refuses a cycle-closing edge and returns `False`; the return value was discarded, so the CONST lost its only in-edge and became unreachable. The graph is never cyclic | §3 box, §7 |
| Which canonicaliser ran in the experiments? | `fast_canonical_string(mode="wl_only")` — greedy on the 1-WL sort key, no 6-tuple | §3 |
| Failure rate, random DAGs, old policy | **0.15 %** (6 / 4,000); at 10⁵, 48 genuine failures | §3, AC-3 |
| Failure rate, random DAGs, repaired policy | **0** genuine failures / 100,000 | AC-3 |
| Failure rate, real Bingo DAGs | **0 / 12,176,790** (95 % CI upper 3.1×10⁻⁷) | AC-4 |
| Failure rate, real UDFS DAGs | **0 / 234,865** (95 % CI upper 1.64×10⁻⁵), all three policies identical on 100 % | AC-4 |
| True sufficient precondition | The **stated** reachability hypothesis, once normalisation can no longer destroy it. No extra clause needed | AC-3 |
| Was completeness actually true? | **No, under the old policy.** Two non-isomorphic DAGs differing only in CONST provenance both gave `VkVspv+NnC`, refuting Theorem (Complete Labeled-DAG Invariant), ⟹ direction. Fixed | §7 |
| Was normalisation evaluation-preserving? | **No, under the old policy**, despite its docstring saying so. On `x→COS→CONST` it moved the output sink and turned 1.0 into cos(1.5) | §7 |
| Is any reported number biased? | **No.** ρ inflation requires a failure, and there are none on real data. All three policies are structurally identical on 100 % of 12.2 M real Bingo DAGs, so ρ = 1.793 either way | AC-4 |
| Does the fix change any submitted number? | **No** — the repair is the identity on adapter output, verified 12,176,790 / 12,176,790 | AC-4 |

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

**The fix and its measurement (2026-07-27, later):**

| Path | Change |
|---|---|
| `src/isalsr/core/labeled_dag.py` | `normalize_const_creation` → minimal repair: add `x_i -> c` only for CONST with in-degree 0, lowest-indexed variable that does not close a cycle; never removes an edge |
| `src/isalsr/core/native/src/labeled_dag.cpp` | Same, kept byte-equivalent with the Python reference (T01) |
| `src/isalsr/core/native/include/isalsr/labeled_dag.hpp` | Invariant-9 doc comment rewritten |
| `src/isalsr/core/native/src/bindings.cpp` | New `testing.fast_canonical_string_raw` — canonical core with no CONST handling, so policies can be compared on one implementation. Not production API |
| `src/isalsr/core/native/src/canonical.cpp` | Entry-point comment: the repair is the identity on precondition-satisfying input |
| `tests/unit/test_const_normalization_repair.py` | **New.** 30 tests, both engines: the 6 cases, edge preservation, precondition survival, completeness counterexample, orphan-CONST repair, cycle-avoiding anchor |
| `tests/unit/test_const_normalization.py` | 8 tests rewritten; `TestRedundancyElimination` → `TestProvenanceIsStructural`, assertions inverted |
| `tests/unit/test_native_datastructures.py` | 2 tests rewritten + 1 new orphan-CONST cross-engine case |
| `tests/unit/test_output_node_and_adapters.py` | 2 tests rewritten; the round-trip one now asserts the value is *preserved* |
| `experiments/scripts/validate_const_repair_synthetic.py` | **New.** 3-arm validation on 10⁵ synthetic DAGs (AC-3) |
| `experiments/scripts/measure_const_normalization_arms.py` | **New.** 3-arm probe on real search output, paired on one DAG stream (AC-4) |
| `experiments/scripts/aggregate_norm_arms.py` | **New.** Pools the Picasso array's per-task outputs |
| `slurm/t15_norm_arms_launch.sh`, `slurm/workers/t15_norm_arms_slurm.sh` | **New.** Picasso array for the UDFS half of AC-4 |
| `.claude/CLAUDE.md` | Critical Invariant 9 rewritten |

Verification: 4,436 unit + 474 property/integration tests pass; `ruff check
src/ tests/` and `mypy --strict src/isalsr/` clean.

### 8.3 Hand-over

- **To T07** (Ezequiel) — **good news, and one correction to the earlier
  hand-over.** The Round-Trip Fidelity hypothesis **is** sufficient and needs no
  extra clause. The earlier draft of this section said otherwise, and said the
  normalisation "can make the graph cyclic"; both were wrong. What happened is
  that `normalize_const_creation` was applied *after* the hypothesis held and
  destroyed it, by dropping an edge `add_edge` had refused. With the repair, the
  normalisation is provably the identity on every DAG satisfying the hypothesis,
  so the theorem holds as written.
  Two things do need your decision:
  1. **Theorem (Complete Labeled-DAG Invariant) was false as previously
     implemented.** Counterexample: `{x1, x2, SIN, CONST, ADD}` with `x1→SIN`,
     `SIN→ADD`, `x2→ADD`, and the CONST hanging off `x1` in one and `SIN` in the
     other. Both satisfy the hypothesis, they are not isomorphic under Definition
     (i)–(iv), and both gave `VkVspv+NnC`. The repair fixes this, so the theorem
     is now true — but it was not true of the submitted implementation, and the
     proof in Appendix A should say why the normalisation step is harmless.
  2. **`methodology.tex` Table 3, line 830** still reads
     `// redirect all Const creation edges to x_1`. That line now describes
     something the code no longer does. It should become "supply a creation edge
     to Const nodes that have none".
- **To T06**: *violating the precondition* and *canonicalisation failing* are
  different events with different rates, and R1.2 asks for the second. On
  synthetic DAGs the precondition is violated 0 % of the time; canonicalisation
  failed 0.048 % under the old policy and **0 %** under the repaired one. Do not
  report one as the other. T06 and T15 §5.3 need the same instrumentation —
  `experiments/scripts/measure_const_normalization_arms.py` already provides the
  hook (it monkey-patches `isalsr.core.canonical.fast_canonical_string`), so
  reuse it rather than adding a second.
- **To T02 / EXECUTION-PLAN §2b**: the counter is no longer urgent for Wave 1.
  AC-4 found zero failures on 12.2 M real Bingo DAGs and the trigger is
  impossible by construction on adapter output, so there is nothing for a
  campaign-time counter to catch. Keep the probe available, but it is not a
  blocker.

### 8.4 Residual risk

| Risk | Severity | Status |
|---|---|---|
| **Precomputed HDF5 atlas not audited.** If `src/isalsr/precomputed/` atlases were built over a population containing non-x₁ CONST provenance, their entries encode the old merge and are now inconsistent with the online path | Medium → **none for published numbers** | **CLOSED 2026-07-28** (T06). `canonicalization_precomputed_s > 0` on **0 / 5,959** submitted run logs: no reported number used the atlas. Still live as a code path — regenerate before ever enabling `--atlas-dir` |
| UDFS half of AC-4 unmeasured | Low — the adapter argument predicts zero, as it did for Bingo | **CLOSED 2026-07-28** (T06). 0 / 234,865, all arms identical on 100 % |
| `methodology.tex:830` describes the old policy | Low — editorial | **OPEN**, Ezequiel |
| CONST-provenance variants no longer deduplicate together | Low — they are genuinely different labeled DAGs; zero effect on any measured population, where provenance is always x₁ | **ACCEPTED**, by decision 2026-07-27 |
| Timeouts at k ≥ 24 under a 10 s budget | Low — production uses 60 s and saw none | **ACCEPTED**; tracked as AC-3′ |

