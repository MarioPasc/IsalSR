# T01 — C++ core port + numerical-equivalence gate

| Field | Value |
|---|---|
| Reviewer comments closed | none directly (enabler for **R1.1**, T10) |
| Type | Implementation |
| Owner | **Mario** (+ Claude Code) |
| Depends on | — |
| Blocks | T02, T03, T06 |
| Status | **SUBSTANTIALLY COMPLETE, AND NOW THE CRITICAL-PATH BLOCKER.** AC-0,1,2,4,7 met; **AC-3 partial** (evolved-DAG gate outstanding); **AC-5 and AC-6 must be redone on the decomposed alphabet** (T16); **AC-8 new** |
| Target | 2026-08-10 |

---

## 0. What remains — read this first (added 2026-07-30)

**T01 is the highest-value next action on the whole revision.** Every other Wave-1
blocker has cleared:

| Wave-1 blocker (`EXECUTION-PLAN.md` §2b) | State |
|---|---|
| **T01** | **OUTSTANDING — this ticket** |
| T06 instrumentation half | done |
| T08 root-cause half | done |
| T16 corrected alphabet | **done 2026-07-30**, gate G9 passed on Picasso (job 1692451) |
| T02 §5.3 MANIFEST schema | to freeze |
| P1–P3 engineering checks | to confirm |

The critical path is `T01 → T02 → T09 → T13 → T12 → T14`, and T02 is a multi-week
Picasso campaign. **Anything that delays T01 delays submission.** Launch is targeted
at ≈2026-08-10 against a 2026-09-10 number freeze.

Three concrete items, in dependency order:

1. **AC-3 gate 3** — the ≥100,000 evolved-DAG replay, byte-exact, both engines.
   This is the one genuinely unfinished piece of the port itself.
2. **AC-8 (new)** — every equivalence gate must also run on **decomposed** DAGs.
   The stored trajectories the replay draws on were produced under the *old*
   alphabet, so gate 3 as written certifies the engine on a population Wave 1 will
   never canonicalise. See §5.3 gate 6.
3. **AC-5 / AC-6 redone** — the benchmark and the projected `S` were measured on
   legacy-alphabet DAGs. T16 raises `k` by ~22 % and per-DAG canonicalisation cost by
   **+24.6 %** (Bingo) / **+10.8 %** (UDFS), so both numbers are stale, and the
   direction is unfavourable. AC-6 is the go/no-go for committing the campaign.

---

## 1. Why this ticket exists and why it is not grouped

This is **not** a reviewer request. It was added by internal decision on 2026-07-27
for one scientific reason:

> **R1.1 cannot be answered by framing.** The Bingo search-only speedup is
> `S = 0.93` — a net loss under a fixed wall-clock budget. No amount of careful
> wording turns a loss into a gain. The only honest way to close that axis is to
> make canonicalisation cheap enough that the eliminated evaluations actually pay
> for it.

The arithmetic. Bingo's fitness evaluation costs ≈ 0.14 ms/DAG; canonicalisation
costs 0.817 ms/DAG (`results.tex:57–58`), a 3.3 : 1 cost ratio against us. With
ρ = 1.83, IsalSR removes ≈ 45 % of evaluations, but pays ≈ 5.8× the saved cost to
do it — hence `S = 0.93` and a 39 % median overhead. A C++ core that brings
canonicalisation to the same order as the evaluation itself flips the sign of that
term. The port is therefore the highest-leverage item in the revision, and it is
kept separate from the campaign that consumes it (T02) so that correctness can be
signed off before ~72,000 core-hours are committed to it.

**Precedent exists.** IsalHG already carries a complete nanobind C++ core with
exactly the module decomposition IsalSR needs. Port, do not invent.

| IsalHG module | IsalSR counterpart |
|---|---|
| `_native/src/cdll.cpp` | `core/cdll.py` |
| `_native/src/s2h.cpp` | `core/string_to_dag.py` |
| `_native/src/h2s.cpp` | `core/dag_to_string.py` |
| `_native/src/wl.cpp` | 1-WL subtree hash inside `core/canonical.py` |
| `_native/src/canonical.cpp` | `core/canonical.py` |
| `_native/src/structural_tuples.cpp` | 6-tuple pruning in `core/canonical.py` |
| `_native/src/token.cpp` | `core/node_types.py` tokenisation |
| `_native/src/sparse_hypergraph.cpp` | `core/labeled_dag.py` |

---

## 2. Mandatory reading

Before writing any code:

**Review context**
- `.claude/notes/review/source/README.md`
- `.claude/notes/review/source/reviewer-1.md` — §R1.1 in particular
- `.claude/notes/review/source/codebase-pointers.md`
- `.claude/notes/review/source/verified-discrepancies.md` — D11, E3

**IsalSR specification**
- `CLAUDE.md` (repo root) — the Critical Invariants section is binding on the port
- `src/isalsr/core/README.md`
- `docs/md_files/DEVELOPMENT.md`

**IsalHG port precedent** (`/home/mpascual/research/code/IsalHG/`)
- `docs/engineering/CODE_DESIGN.md`
- `docs/engineering/DEVELOPMENT.md` — build flow, PGO two-stage flow
- `docs/engineering/ALGORITHMS.md`
- `docs/engineering/CPP_OPTIMIZATION_LOG.md` — what was tried and what failed
- `docs/engineering/CPP_SPEEDUP.md` — benchmarking methodology to copy
- `CMakeLists.txt`, `pyproject.toml`

---

## 3. Established facts

- `isalsr.core` currently has **zero external dependencies** (stdlib only). This
  layering rule is enforced repo-wide and the port must preserve the *Python-facing*
  contract: `import isalsr.core.canonical` must keep working with no build step,
  falling back to the pure-Python path when the extension is absent.
- The canonical algorithm in production is `fast_canonical_string(mode="wl_only")`.
  `pruned_canonical_string` and `canonical_string` are legacy/reference.
- Completeness of `wl_only` was verified exhaustively for k = 1..8 (all k!
  permutations) and statistically for k = 10..15. 890 tests pass. **These are the
  regression oracle for the port.**
- The 14,841-DAG unit-test corpus cited at `discussion.tex:38` lives in `tests/`.
- Eleven Critical Invariants in `CLAUDE.md` govern correctness. Invariants 1, 3, 5,
  8, 9 and 10 are the ones a naive port breaks:
  - (1) CDLL indices ≠ graph node indices
  - (3) `add_edge(source, target)` direction and `_input_order`
  - (5) spiral displacement sorted by `|a|+|b|`, not `a+b`
  - (8) operand order for binary ops via `ordered_inputs()`
  - (9) `normalize_const_creation` applied where guarded by `_has_const_nodes()`
  - (10) label-aware 6-tuple pruning (partition **by label** before max-τ)

---

## 4. Non-goals

- Do **not** port `isalsr.evaluation`, `isalsr.search`, `isalsr.adapters`, or
  anything under `experiments/`. Only `src/isalsr/core/`.
- Do **not** change any algorithm. This is a re-implementation, not a redesign.
  Algorithmic change belongs to T03.
- Do **not** port the competitor solvers. UDFS and Bingo stay as they are.

---

## 5. Work specification

### 5.1 Build system
Mirror IsalHG: `nanobind` + CMake, C++17, Release with `-O3 -march=native
-DNDEBUG`, LTO. Extension built as `isalsr.core._native`. `pyproject.toml` gains an
optional `[native]` extra. **Graceful degradation is mandatory**: every
`isalsr.core` module attempts the native import and falls back to the existing pure
Python on `ImportError`, selected by a single module-level switch so a run can be
forced onto either engine for benchmarking.

### 5.2 Port order
Bottom-up, each with its own equivalence test before the next starts:
`cdll` → `labeled_dag` → `node_types`/tokenisation → `string_to_dag` (S2D) →
`dag_to_string` (D2S) → `wl` hash → `structural_tuples` → `canonical`.

### 5.3 The equivalence gate
The port is not accepted on "tests pass". It is accepted on **bit-exact canonical
string identity** against the Python engine:

1. **Exhaustive**, k = 1..8: for every DAG in the enumeration corpus and every one
   of the k! internal-node permutations, `cpp_canonical(D) == py_canonical(D)` as
   byte strings. Reuse `core/permutations.py`.
2. **Corpus**, the full 14,841-DAG unit-test suite: byte-identical on all.
3. **Evolved**, ≥ 100,000 DAGs replayed from stored Bingo and UDFS trajectories in
   `…/results/model_validation/real_benchmarks/wl_subtree_unified/`: byte-identical.
4. **Round-trip**: `S2D(D2S(D, x_1))` ≅ `D` on all of the above, under both engines.
5. Any single mismatch blocks the ticket. Do not "explain" a mismatch — fix it.
6. **Decomposed alphabet** (added 2026-07-30, T16). Gates 1–4 must **also** pass on
   DAGs carrying the paper's alphabet — `Neg`/`Inv` present, `Sub`/`Div` absent,
   `Pow` the only order-sensitive binary operation.

   **Why this is not redundant with gates 1–4.** Gate 3 replays DAGs from
   `…/wl_subtree_unified/`, a campaign produced *before* T16, so every replayed DAG
   is legacy-encoded. Passing gate 3 alone would certify the C++ engine on a label
   distribution Wave 1 will never canonicalise. The exhaustive and corpus gates are
   alphabet-agnostic in principle — `Neg`/`Inv` are ordinary labels the canonicaliser
   already handles — but "in principle" is exactly what this gate exists to refuse.

   **Cheapest sound construction**: replay the same stored trajectories through the
   *current* adapters, which decompose inline, and compare the two engines on the
   resulting DAGs. `experiments/scripts/measure_decomposition_impact.py` already
   builds all three encodings (`legacy`, `split`, `shared`) paired per DAG and can be
   reused rather than rewritten. Cover `split` — the production setting — and confirm
   the `k` distribution of the replayed population matches the shifted one
   (Bingo mean 5.47 → 6.72, p95 11 → 15), not the legacy one.

### 5.4 Benchmark
Copy IsalHG's `CPP_SPEEDUP.md` methodology: best-of-9, median-of-4-reps, 3 warmup
reps, driver script committed, raw JSON committed. Report per-DAG canonicalisation
time stratified by k over the same k-buckets the paper already uses
(k < 5, 5 ≤ k < 15, 15 ≤ k < 32), for both engines, on the workstation **and** on a
Picasso compute node — the paper's cost numbers are Picasso numbers and that is the
comparison that will be reported.

### 5.5 Determinism
The extension must be deterministic and platform-stable for the hash values that
feed the dedup set. If FNV/WL hashing is 64-bit, confirm the same seeds and the same
mixing constants as Python. Pin them in a test.

---

## 6. Acceptance criteria

- **AC-0 (mandatory, applies to every ticket).** §7 Work log is filled in as the
  work proceeds: decisions taken, problems found, dead ends, anything that surprised
  you, and any disagreement with this specification. A ticket whose work log is
  empty is not complete regardless of the criteria below.
- **AC-1.** Native extension builds from a clean checkout on the workstation and on
  a Picasso compute node; build instructions in `docs/md_files/technical_report/` or
  a new `docs/engineering/CPP_BUILD.md`.
- **AC-2.** Pure-Python fallback verified: the full test suite passes with the
  extension absent, and passes with it present.
- **AC-3.** All **six** equivalence gates in §5.3 pass with **zero** mismatches.
  Evidence: a committed report with counts, not a claim.
  **Status 2026-07-30: PARTIAL.** Gates 1, 2 and 4 pass. **Gate 3 (≥100,000 evolved
  DAGs) is outstanding** and gate 6 is new. This is the single unfinished piece of
  the port itself and the first thing to do.
- **AC-4.** `python -m pytest tests/ -v`, `ruff check`, and `mypy --strict` all clean.
- **AC-5.** Benchmark table produced: per-DAG canonicalisation time, Python vs C++,
  by k-bucket, on Picasso hardware, with the speedup factor and its dispersion.
  **REOPENED 2026-07-30 (T16).** The measured table used legacy-alphabet DAGs.
  Decomposition raises `k` ~22 % on both hosts, which **moves DAGs across the paper's
  own k-buckets** (`k < 5`, `5 ≤ k < 15`, `15 ≤ k < 32`) — Bingo p95 goes 11 → 15, so
  mass shifts out of the middle bucket. Re-measure on decomposed DAGs and report both,
  so the continuity table in T02 §5 can attribute movement to the engine and to the
  alphabet separately rather than confounding them.
- **AC-6.** A projected `S` for Bingo and UDFS computed from the measured C++
  canonicalisation cost and the *existing* ρ values, stated as a projection. This is
  the go/no-go signal for T02: if the projection does not move Bingo's `S` above 1.0,
  say so in §7 and escalate before committing the campaign.
  **REOPENED 2026-07-30 (T16), and the correction pushes the wrong way.** The
  projection must use the **decomposed** canonicalisation cost, which is **+24.6 %**
  (Bingo) / **+10.8 %** (UDFS) above legacy, because that is what Wave 1 will actually
  pay. Do not net this against the C++ speedup by assumption — the two are independent
  and both land in Wave 1. Use the ρ values that go with the decomposed alphabet
  (Bingo exactly invariant; UDFS +1.4 %), not the submitted ones.
  **This is the go/no-go and it is now harder to clear. If it fails, escalate before
  committing 36,000 core-hours — an honest negative here is worth more than a
  campaign that confirms it slowly.**
- **AC-7.** §8 filled.
- **AC-8 (new, 2026-07-30, T16).** The equivalence gate covers the **decomposed**
  alphabet (§5.3 gate 6): `Neg`/`Inv` present, `Sub`/`Div` absent, `Pow` the only
  order-sensitive binary operation, byte-exact between engines. Evidence: counts from
  a replay through the current adapters, plus the `k` distribution of the replayed
  population showing the shift.
  Rationale: T16 changed what Wave 1 canonicalises. An engine certified only on the
  pre-T16 label distribution is certified on a population that will never occur.
- **AC-9 (new, 2026-07-30).** The two literal canonical strings printed in the
  manuscript are re-derived under the final engine **and** the final alphabet:
  `VcVspv*pv+PpcnnC` (`methodology.tex:256`, a figure caption explicitly labelled
  "the canonical string") and `VgnV*C` (`methodology.tex:272`, inside a
  `\begin{comment}` block, so not typeset). The first is the one that matters.
  These were already flagged in §7 as the manuscript-number carve-out; T16 makes the
  re-derivation mandatory rather than precautionary, since a string containing `-` or
  `/` is now *by construction* unreachable from the adapters.

---

## 7. Work log

> Append entries as `### YYYY-MM-DD — <topic>`. Record what you decided and why,
> what broke, and what you could not resolve.

### 2026-07-27 — Ticket opened, plan recorded before any code

**Gating check.** `Depends on: —`. Nothing blocks T01. T01 blocks T02/T03/T06 and
sits at the head of the critical path (`README.md` §Dependency spine), so schedule
slip here is schedule slip on the submission.

**Sizing.** `src/isalsr/core/` is 3,557 LoC of Python across 16 files
(`canonical.py` 1035, `labeled_dag.py` 683, `dag_to_string.py` 377,
`string_to_dag.py` 304, `commutative.py` 284, `dag_evaluator.py` 215,
`node_types.py` 213, `cdll.py` 137, `permutations.py` 125, `algorithms/` 174).
IsalHG's `_native` is 2,047 LoC of C++ across 9 translation units
(`h2s.cpp` 1016, `s2h.cpp` 289, `structural_tuples.cpp` 182, `canonical.cpp` 158,
`token.cpp` 125, `sparse_hypergraph.cpp` 103, `wl.cpp` 94, `cdll.cpp` 68,
`thread_pool.cpp` 12). Comparable scale; the ticket's "port, do not invent"
premise is credible on size grounds.

**Plan — phases, gated.**

| Phase | Deliverable | Gate to next |
|---|---|---|
| P0 | Recon: IsalHG porting playbook + IsalSR port surface inventory + premise checks on the §5.3 corpora | Design decisions answered by Mario |
| P1 | Build skeleton: CMake + nanobind + `[native]` extra + `_ENGINE` switch + CI-less local build doc | Extension imports; fallback verified |
| P2 | `cdll` → `labeled_dag` → `node_types`/tokenisation | Per-module equivalence test |
| P3 | `string_to_dag` (S2D) → `dag_to_string` (D2S) | Round-trip gate §5.3(4) |
| P4 | `wl` hash → `structural_tuples` → `canonical` | Determinism pin (§5.5) |
| P5 | Equivalence gates §5.3(1)–(4), all four | AC-3, zero mismatches |
| P6 | Benchmark (workstation + Picasso node), projected `S` | AC-5, AC-6 → go/no-go for T02 |

**Delegation policy.** P0 is two read-only investigators in parallel. P1–P4 are one
implementer at a time in the main tree (no worktrees: the ticket's own §4.3 hazard
applies — `pip install -e` resolves to the main checkout and a worktree's tests
would silently exercise main's code, and a worktree cannot see the built
extension). P5 and P6 verification are re-run by the orchestrator, never trusted
from an agent's closing claim.

### 2026-07-27 — Design decisions taken (Mario)

| Question | Decision | Consequence |
|---|---|---|
| Port scope | **Canonicaliser hot path only** — the ticket's §5.2 list verbatim (8 TUs): `cdll`, `labeled_dag`, `node_types` tokenisation, `string_to_dag`, `dag_to_string`, `wl`, `structural_tuples`, `canonical`. | `commutative.py`, `dag_evaluator.py`, `permutations.py` and `algorithms/` stay pure Python. Keeps `dag_evaluator`'s float semantics out of the equivalence surface entirely, which is the right call — it is the one module whose bit-exactness would move R² numbers. |
| Compiler flags | **`-O3 -march=x86-64-v3 -DNDEBUG -flto`**, one portable binary. Not `-march=native`. | One `build_hash` for the whole campaign; `S` is not confounded by ISA. Costs an estimated 2–5 % against native, which is noise against the 3.3 : 1 gap being closed. Removes the first residual risk named in §8.4. |
| Python-engine defects found during the port | **Fix in both engines.** Do not gate the port on reproducing a defect. | Deviates from the entry above, which is superseded. One carve-out held open: a fix that changes a number already printed in the submitted manuscript is escalated before it ships, because that is an undisclosed data change rather than an engineering detail. |
| AC-6 go/no-go | **Escalate with numbers; do not self-decide.** Report projected `S` for both methods, the break-even `T_canon`, and `dS/dT_canon`. | Matches the standing rule that a result changing what the paper claims goes to a human. |

*(Supersedes the closing paragraph of the previous entry: the C++ engine is no
longer required to be bug-for-bug identical to Python; it is required to be
byte-identical to the **fixed** Python engine, with both fixed together.)*

### 2026-07-27 — Build-environment reconnaissance (orchestrator, not delegated)

**Workstation.** gcc 12.2.0 (Debian), cmake 3.25.1, ninja **absent**, Python
3.11.15, `nanobind` **absent** from the `isalsr` env (`pybind11` 3.0.2 is present
but is not what IsalHG uses). CPU: i7-13700KF — AVX2 + FMA, **no AVX-512**. So
`x86-64-v3` is also the ceiling locally; `v4` was never an option.

**Picasso — three findings, one of them blocking.**

| # | Finding | Consequence |
|---|---|---|
| E-1 | **`fscratch/repos/IsalSR` does not exist, and there is no `isalsr` conda env.** Present: `IsalHG`, `slim-diff`, `VENA*`; envs `isalhg`, `isalhg-hypercot`, `vena`, `vena-v100`. | The March campaign's remote environment is **gone**. AC-1 ("builds from a clean checkout on a Picasso compute node") requires standing the env up from scratch. This is unscheduled work sitting on the critical path and must be done before Wave 1, not discovered at G5. |
| E-2 | Default login toolchain is **gcc 7.5.0** (SUSE) — too old for the C++17 the port needs. Modules available up to `gcc/15.2.0`; `cmake` 3.28.3 default plus modules to 3.31.4. | Build and *runtime* must both `module load` the same gcc, or the extension must be linked `-static-libstdc++ -static-libgcc`. The latter is preferable: it removes a runtime module dependency from 3,000 SLURM tasks and one class of "works on the login node, dies on the compute node" failure. Pin this at G5. |
| E-3 | The CPU pool is **four distinct classes**, and only two advertise AVX-512: `sd,intel,avx512` (123 nodes, 52c), `sr,amd` (154 nodes, 128c), `bc,amd,avx512` (32 nodes, 256c), `bl,amd` (24 nodes, 128c). Partitions are now `cpu_partition` / `gpu_partition`. | **Independently confirms the `x86-64-v3` decision**: 178 of 333 CPU nodes have no AVX-512, so a `v4` or `native`-on-`bc` build would not run pool-wide. Also flags for `EXECUTION-PLAN.md` §5/P3: the partition and feature names differ from those recorded for the March campaign, so the "pin `--constraint` to the March node type" mitigation may not be executable as written. That is T02's problem, but it is discovered here. |

Not chased further: mapping feature tags to CPU model strings (`scontrol` does not
expose `CPUModel` on this cluster). D2 is answerable from the March run logs
locally, since `hardware_info.py` records CPU per run — left to T02.

### 2026-07-27 — IsalHG playbook received, verified, and corrected on one point

Playbook at `T01-appendix/isalhg-port-playbook.md` (398 lines). Every load-bearing
claim re-checked by me directly against IsalHG sources rather than taken on trust:

| Claim | Verified at |
|---|---|
| `scikit-build-core>=0.9` + `nanobind>=2.0`, backend `scikit_build_core.build` | `IsalHG/pyproject.toml:1–6` |
| C++17, `-O3 -march=native -DNDEBUG -fno-plt -funroll-loops`, IPO via `check_ipo_supported` | `IsalHG/CMakeLists.txt:4,17,21–24` |
| **No graceful fallback exists in IsalHG** — hard `from isalhg.core._core import …` | `IsalHG/canonical.py:69` |
| `backends.py` with `DEFAULT_BACKEND` + `resolve()` dispatch registry | `IsalHG/src/isalhg/core/backends.py` |

Reported IsalHG gains: **101–180× single-thread** for the bare C++ port (phase 0),
rising to 499–1,494× versus pure Python once parallelism is added; 5 optimisations
tried and reverted (L1d thrash from a per-frame displacement cache, stack pressure
from `std::array` prefixes, arena-pooled vectors, flat η stride, manual CPU pinning).

#### Correction: R3 (parallel seed loop) is worth **zero** to this ticket — and is a hazard

The playbook's headline recommendation is "implement R3 first — it is the single
largest win (−73 to −89 % wall-clock)". **That advice does not transfer, and
following it would be a defect rather than an optimisation.**

- `EXECUTION-PLAN.md` §1 fixes the campaign budget at **1 core per run**; `CLAUDE.md`
  independently records `processes: 1` in all four production YAMLs and `cpus: 1` in
  the SLURM configs. Intra-call parallelism is therefore unavailable in exactly the
  configuration that produces the reported numbers.
- Worse than unavailable: IsalHG sizes its pool from `hardware_concurrency()`, which
  on a Picasso node reports 52–256 rather than the 1 core SLURM allocated. Spawning
  that many workers inside a 1-core cgroup oversubscribes it and would make
  canonicalisation *slower*, silently, on the cluster only — the single most
  expensive failure mode available to this ticket, since it would not reproduce on
  the workstation.
- IsalHG's own negative result #5 (manual CPU pinning: 80 ms → 110 ms) is the same
  phenomenon observed from the other direction.

**Consequence, now binding on the port:** no threading of any kind in
`isalsr.core._native`. `thread_pool.cpp` is *not* copied, despite the playbook
listing it as "copy verbatim, do early". The speedup must come entirely from
single-thread work — which is where IsalHG's 101–180× phase-0 figure lives anyway,
and that alone is far more than AC-6 needs.

#### Provisional AC-6 arithmetic (not a result — a sanity check on feasibility)

Bingo canonicalisation is 0.817 ms/DAG against a 0.14 ms/DAG evaluation, i.e. 3.3 : 1
against us. A single-thread speedup of only **20×** puts canonicalisation at
≈0.041 ms/DAG and inverts the ratio to ≈1 : 3.4 in our favour; IsalHG's phase-0 range
implies considerably more. The feasibility of AC-6 is therefore not in doubt — but
the number reported must be measured on a Picasso node under §5.4's protocol, never
projected from IsalHG's figures, which were obtained on different algorithms and
different inputs.

Also noted: the playbook states the workstation CPU as an i7-13620H. It is an
**i7-13700KF** (measured). Immaterial to any decision; recorded so the benchmark
section does not inherit the wrong hardware string.

### 2026-07-27 — Port-surface inventory: three of the ticket's four §5.3 premises do not hold

Inventory at `T01-appendix/isalsr-port-surface.md`. The agent returned P1 REFUTED,
P2 PARTIAL, P3 REFUTED, P4 CONFIRMED. I re-ran the decisive checks myself rather
than accepting them; the results below are mine, not the agent's.

#### BLOCKER — canonical strings are not reproducible across Python sessions

`canonical.py:702` computes the 1-WL subtree hash as

```python
node_hash[u] = hash((dag.node_label_unchecked(u), tuple(children_hashes)))
```

`NodeType` is an `Enum`, and CPython's `Enum.__hash__` delegates to
`hash(self._name_)` — a **string** hash, and therefore salted by `PYTHONHASHSEED`.
Measured directly: `hash(NodeType.SIN)` is `-596725465393948695` at seed 0,
`+9050745085082211361` at seed 42, `+2878440809595582671` at seed 1337, exactly
equal to `hash("SIN")` in each case.

That salt propagates all the way to the output. Experiment
(`scratchpad/hashseed_probe.py`, 120 randomly generated DAGs, generator seeded
independently of `PYTHONHASHSEED`):

| Quantity | seed 0 | seed 42 | seed 1337 |
|---|---|---|---|
| Canonical strings differing from seed 0 | — | **12 / 120 (10 %)** | **13 / 120 (11 %)** |
| Distinct canonical strings | 120 | 120 | 120 |
| Permuted copies canonicalising equal | 40 / 40 | 40 / 40 | 40 / 40 |

Example (index 6): `'V*V+V/VkVkpvicNNV/VgpppC'` at seed 0 versus
`'V*V+V/VkVkpvicNV/VgpppC'` at seed 42 — different lengths, so not a formatting
artefact.

**Consequences, in order of severity.**

1. **§5.3's acceptance gate is untestable as written.** "Byte-exact canonical string
   identity against the Python engine" has no fixed referent when the Python engine
   returns a different string per session. Nothing in the port can fix this; the
   Python side must be made deterministic first.
2. **The scientific claim survives; the reproducibility claim does not.** The WL hash
   is a sort key over candidates, and the label component is isomorphism-invariant,
   so within any single session the function is still a valid invariant — the
   40/40 permutation result confirms it holds under each seed independently. What
   changes across sessions is *which representative* of each isomorphism class is
   chosen. ρ and the reduction factor are counts of distinct classes and so are
   expected to be unaffected.
3. **Honest limitation of the evidence above.** The "partition identical" check is
   weaker than it looks: all 120 DAGs landed in distinct classes, so partition
   equality was satisfied vacuously. The load-bearing evidence for point 2 is the
   invariance check, and that rests on 40 permuted pairs per seed, not a proof.
   Before ρ is asserted to be seed-independent in the response letter, this needs a
   proper run over the k=1..8 exhaustive corpus under ≥2 seeds.

**Fix** (authorised 2026-07-27: repair both engines rather than reproduce the
defect): replace `hash()` at `canonical.py:702` with a fixed-constant FNV-1a
64-bit over the label ordinal and the child hashes. This is the same primitive the
C++ side needs for §5.5, and the build skeleton already exposes `fnv1a64` for
exactly this purpose, so both engines can be pinned against one shared test vector.
Note that the fix *will* change roughly 10 % of canonical strings relative to
whatever the March campaign produced — harmless for counts, but it is the reason
the §5.3 gate must be defined against the *fixed* Python engine, not against
stored March artefacts.

#### P3 REFUTED — there are no replayable DAG streams, and this is not only T01's problem

Two independent failures. The path in the ticket
(`…/research/isalsr/results/model_validation/…`) does not exist; the real root is
`…/research/ISAL/completed/isalsr/…`. More seriously, `wl_subtree_unified/`
contains **zero** `run_log.json` files, and the 2,640 run logs under `wl_subtree/`
persist only aggregates — `total_dags_explored` and `unique_canonical_dags` as
integers. No DAG structures and no canonical strings were ever written.

- **For T01**: §5.3 gate 3 ("≥100,000 DAGs replayed from stored trajectories") cannot
  be run. It needs replacing, and the natural substitute is to generate the evolved
  distribution live — run Bingo and UDFS briefly under both engines and compare
  canonical strings in-process — which tests the same thing without needing history.
- **For the campaign**: this is the answer to `EXECUTION-PLAN.md` §2b check **P1**,
  and the answer is that it fails. The plan already states the consequence: the DAG
  or canonical-hash stream must be logged **before Wave 1**, or T04 Mode 1 can only
  ever replay Wave 3's own runs and loses the `isalsr`-arm decomposition that
  answers R1.4. Escalated rather than absorbed — it is outside T01's scope.

#### P1 REFUTED, P2 PARTIAL — the cited test corpora do not match the manuscript

- "14,841" appears nowhere in `tests/` or in any source file; the only occurrence in
  the whole repo is inside an agent definition. No stored DAG artefact exists; tests
  build DAGs dynamically from 8 seed expressions, and the completeness tests exercise
  **41,217** permutation instances across k=1..8.
- Current fast-canonical test count is **94** collected (86 unit + 8 property), not
  890. Large-expression coverage is k=10 and k=11 only, not k=10..15.
- Both figures are cited to `discussion.tex`, which brings up the next item.

#### The submitted manuscript is not in this repository

`results.tex`, `discussion.tex` and `response_to_reviewers.tex` do not exist here.
The only LaTeX present is `docs/md_files/technical_report/`. Every ticket-cited
locator of the form `results.tex:57–58` or `discussion.tex:38` is therefore
unverifiable from this checkout, and §8.1's "as submitted" column cannot be
populated without the manuscript sources. Raised with Mario, who supplied the
location: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/69c1637a28a81fea2badda9a/`
(with `article/` and `reviews/` beneath it). Extraction under way.

### 2026-07-27 — Decisions on the four premise failures (Mario)

| Question | Decision |
|---|---|
| Scope of the hash fix | **FNV-1a 64-bit in both engines, pinned by a shared test vector.** Not a `PYTHONHASHSEED=0` pin, which cannot make C++ match a salted Python string hash and fails silently on any unpinned invocation. |
| Replacement for §5.3 gate 3 | **Live dual-engine comparison during short searches**: canonicalise every DAG twice in-process under Bingo and UDFS and assert equality, target ≥100,000 DAGs. Tests the evolved distribution without needing history, and emits the stream check P1 wants. |
| `EXECUTION-PLAN.md` check P1 (hash-stream logging) | **Folded into T01**, declared as deliberate scope creep. The instrumentation sits in the same dedup hook the port touches and the gate-3 replacement already emits the stream; doing it separately means a second pass over the same code and leaves Wave 1 blocked meanwhile. |

### 2026-07-27 — Build skeleton landed and independently verified

Delivered: `CMakeLists.txt`, `native/{src,include}`, `src/isalsr/core/backends.py`,
`tests/unit/test_native_build.py`, `docs/engineering/CPP_BUILD.md`, and a
`pyproject.toml` switched to `scikit-build-core` + `nanobind`. No algorithm ported.

I re-ran every acceptance check myself rather than accepting the agent's summary:

| Check | Result |
|---|---|
| `backends.engine()` | `cpp` |
| `build_info()` | `isa_level=x86-64-v3`, `avx2=1`, `fma=1`, **`avx512f=0`**, `compiler=gcc 12.2.0`, `cplusplus=201703`, `ndebug=1`, `build_hash=298fc1188bf1b051` |
| `ISALSR_ENGINE=python` | `python` — fallback switch works |
| `_native.fnv1a64` vs pure-Python reference | agrees on empty input, ASCII, high bytes, the full 0..255 range, and 1 kB |
| `pytest tests/ -q` | **1528 passed, 1 skipped** (was 1495; +33 new) |
| `mypy src/isalsr/` | clean, 40 files |
| `ruff check src/` | clean |

`build_info()` reporting `avx512f=0` on a machine that *has* no AVX-512 is the
switch behaving correctly, and it is the field G5 should assert on a compute node —
it distinguishes "native engine loaded" from "silent pure-Python fallback" without
running anything.

**Two loose ends recorded rather than hidden.**

1. **The `.so` installs to site-packages, not into the source tree** —
   `…/envs/isalsr/lib/python3.11/site-packages/isalsr/core/_native.cpython-311-x86_64-linux-gnu.so`.
   Consequence for the campaign: `rsync ./ picasso:…` **will not carry the
   extension**. It must be built on Picasso as part of environment setup, which
   compounds finding E-1 (no `isalsr` env exists there at all). The SLURM worker
   must therefore verify `backends.engine() == "cpp"` and fail loudly if not —
   otherwise 3,000 tasks would silently run the pure-Python engine and produce a
   campaign that measures nothing.
2. **Six pre-existing `ruff` errors in `tests/`** — `test_bingo_adapter.py` (F841),
   `test_statistical_analysis.py` (B905), `test_udfs_adapter.py` (E402 ×4). Not
   introduced by this work (those files are unmodified per `git status`), but AC-4
   requires `ruff check` clean across `src/` **and** `tests/`, so they are T01's to
   clear before close.

### 2026-07-27 — Determinism fix landed and independently verified

`canonical.py` now computes the 1-WL subtree hash with a fixed-constant FNV-1a
64-bit over the label's *value string* and the child hashes, little-endian, offset
basis `0xcbf29ce484222325`, prime `0x100000001b3`. Hashing the label value rather
than an ordinal table means the enum can be reordered without changing a single
canonical string, and the C++ side reproduces it in five lines.

Verified by me, re-running the probe with a fourth seed the implementing agent
never tried (99991):

| Seed | Canonical strings differing from seed 0 | Distinct strings | Permutation invariance |
|---|---|---|---|
| 0 | — | 120 | 40 / 0 |
| 42 | **0 / 120** (was 12) | 120 | 40 / 0 |
| 1337 | **0 / 120** (was 13) | 120 | 40 / 0 |
| 99991 | **0 / 120** | 120 | 40 / 0 |

`pytest`: **1630 passed, 1 skipped** (+102 new). `mypy` and `ruff check src/`
clean. No existing test encoded a canonical string that changed, so no expected
literals were rewritten — which is itself informative: the suite never pinned a
literal canonical string, so it could not have caught this defect.

Test vector `tests/data/wl_hash_vectors.json` is now the shared oracle both engines
are pinned to. First three entries: `("var", []) → 7567199770864868670`,
`("+", []) → 12638127826927718602`, `("*", []) → 12638128926439346813`.

### 2026-07-27 — **PREMISE-FALSE: §1's rationale for this ticket does not survive contact with the manuscript**

Manuscript located and read directly (not via agent summary):
`…/69c1637a28a81fea2badda9a/article/paper/`.

#### The cost figures in §1 are stale by one campaign

Table `tab:three_axis`, `results.tex:57–58`, is the submitted source of truth:

| Method | ρ | Red. | T_canon | T_eval | OH | S |
|---|---|---|---|---|---|---|
| UDFS | 1.56 ± 0.24 | 34.2 % | 0.296 ms | **≈519 ms** | 0.05 % | 1.07 |
| Bingo | 1.83 ± 0.09 | 45.2 % | 0.817 ms | **1.29 ms** | 39.2 % | 0.93 |

§1 of this ticket asserts "Bingo's fitness evaluation costs ≈ 0.14 ms/DAG …
a 3.3 : 1 cost ratio against us". The manuscript prints **1.29 ms**, so the true
ratio is **0.817 / 1.29 = 0.63 : 1** — canonicalisation is *already cheaper than
evaluation*, not 3.3× more expensive. The ticket's follow-on arithmetic ("pays
≈5.8× the saved cost") therefore has no basis. Same for UDFS: the ticket says
1 : 64, the manuscript says T_eval/T_canon **> 1,500** (`results.tex:191`). Both
ticket figures trace to `CLAUDE.md`'s older 22-problem campaign and were superseded
by the 50-problem run actually submitted.

#### The deeper problem: `S` is *defined* to be insensitive to canonicalisation cost

`computational_experiments.tex:115–127`:

> T_total = T_search + T_canon  … The baseline variant has T_canon = 0 by
> definition, so T_search = T_total. … The search-only speedup
> S = T_search^baseline / T_search^IsalSR **isolates the effect of deduplication on
> pure search time.**

`S` subtracts `T_canon` from the IsalSR side before dividing. Making
canonicalisation faster reduces `T_total` and reduces `OH`, but leaves
`T_search^IsalSR` **unchanged by construction** — so it leaves `S` unchanged.

§1 claims "A C++ core that brings canonicalisation to the same order as the
evaluation itself flips the sign of that term." It cannot. The term it would flip
is not in `S`. And `S = 0.93` is not caused by canonicalisation being expensive —
it says that, *even after canonicalisation is excluded*, Bingo–IsalSR needs more
search time than the baseline. That is a statement about the search trajectory
under deduplication, not about engine speed.

#### What the port does and does not buy

| Quantity | Effect of the C++ port | Why |
|---|---|---|
| `OH` (39.2 % Bingo) | **Large improvement** — this is the real prize | OH = T_canon/T_total, directly proportional to canonicalisation cost |
| Wall-clock, and the Nemenyi wall-clock gap at `results.tex:193–196` | **Large improvement** | Bingo's IsalSR arm currently exceeds the critical difference; a 20× cheaper canonicaliser plausibly collapses it |
| `S` (0.93 Bingo) | **No first-order effect** | `T_canon` is subtracted out before the ratio is formed |
| ρ, reduction factor | **No effect** | counts of canonical classes, engine-independent |

So T01 remains clearly worth doing — but its deliverable against R1.1 is
"canonicalisation overhead falls from 39.2 % to ≈2 % and the wall-clock penalty
disappears", **not** "S rises above 1.0". AC-6 as written asks for a projected `S`
and says to escalate if it does not exceed 1.0; under the manuscript's own
definition that escalation is guaranteed regardless of how fast the C++ is.
Escalated to Mario rather than quietly reinterpreting the acceptance criterion.

### 2026-07-27 — Port phase P2 complete (CDLL + LabeledDAG)

`native/src/{cdll,labeled_dag}.cpp` with headers, bound for differential testing as
`_native.testing.{NativeCDLL,NativeLabeledDAG}`. Verified by me:
**2,648 passed, 1 skipped** (+1,018 differential tests: 500 randomised CDLL operation
sequences and 500 randomised DAG build sequences, each 30 ops, plus 20 targeted
invariant and error-path tests). `engine()`=`cpp`, `isa_level=x86-64-v3`, mypy and
ruff clean. A grep confirms `native/` contains **no** `std::async`, `std::thread`,
`thread_pool` or `hardware_concurrency` — the no-threading rule held.

Carried forward: the implementer used `std::set<int32_t>` where Python uses
`set[int]`, so topological-sort tie order can differ between engines. It avoided
asserting exact topo-sort equality for that reason, which is the right call. Not a
problem for the canonical path — `_wl_subtree_hashes` sorts `children_hashes`
before mixing, and candidate ties are resolved by lexmin backtracking rather than
by iteration order — but it is an assumption to **verify at the P5 equivalence
gate**, not to carry on trust.

### 2026-07-27 — Port phase P3 complete (token grammar + S2D)

`native/src/{node_types,string_to_dag}.cpp`. Verified: **4,766 passed, 1 skipped**
(+2,118 differential tests over 2,000 generated strings, 1–5 variables, 0–25
tokens, plus 118 explicit edge cases). `native/` still threading-free.

I audited the test rather than the summary: `tests/unit/test_native_s2d.py:147`
compares `ordered_inputs()` in **insertion order** per node, not merely sorted
neighbour sets. That distinction is the whole point of invariant 8 — a test
comparing sorted neighbours would pass against a port that had silently swapped
operand order for SUB/DIV/POW, and every downstream number would be wrong.

**Scope reduction (deviates from ticket §5.2, deliberately).** `canonical.py:50`
imports only `generate_pairs_sorted_by_sum` from `dag_to_string.py`; the
`DAGToString` class is used solely by `adapters/base.py`,
`algorithms/greedy_{single,min}.py` and `search/population.py`, none of which run
in the production campaign. §5.2 lists `dag_to_string` in the port order, but under
the hot-path-only scope decision it buys nothing: 377 lines of Python that would
need porting *and* equivalence-testing for zero effect on `T_canon`. Only the
displacement-pair generator (invariant 5) is being ported. Recorded here rather
than silently skipped.

### 2026-07-27 — `S` root cause: the invariance is exact algebra, and H3 dominates

Investigation at `T01-appendix/s-root-cause.md`. I verified the load-bearing claims
against the code myself.

#### `T_search` is derived, so `S` is *exactly* invariant to canonicalisation speed

Both runners compute it identically —
`search_only = wall_clock - dedup.canon_time_total`
(`bingo/isalsr_runner.py:519`, `udfs/isalsr_runner.py:277`) — and the translator
reports `overhead_time_s = r.canonicalization_time_s`
(`bingo/translator.py:118`).

Speed canonicalisation by a factor *f*: `wall_clock` falls by
`T_canon·(1 − 1/f)` and the measured `T_canon` falls to `T_canon/f`, so

```
T_search_new = (wall_clock − T_canon·(1 − 1/f)) − T_canon/f = wall_clock − T_canon
```

— unchanged for every *f*. This is not "approximately insensitive"; it is an
identity. **No C++ engine can move `S` by making `fast_canonical_string` faster.**

#### What is charged where

| Operation | Bingo | UDFS |
|---|---|---|
| Atlas lookup | inside `T_canon` | inside `T_canon` |
| `fast_canonical_string` | inside `T_canon` | inside `T_canon` |
| `hash(canonical)` | **inside** | **outside** |
| DAG conversion (`agraph_to_labeled_dag` / `compgraph_to_labeled_dag`) | **outside** → `T_search` | **outside** → `T_search` |
| dedup set lookup / insert | **outside** → `T_search` | **outside** → `T_search` |

The published 39.2 % overhead therefore *understates* IsalSR's true cost: the
conversion and set work is real IsalSR-only expense that the overhead metric does
not count and that `T_search` silently absorbs.

#### Termination regime — nothing hits the ceiling

Across 300 sampled `wl_subtree` Bingo runs (150 per arm), **0 hit the 43,200 s
ceiling in either arm**. Baseline `T_total` mean 2,478 s; IsalSR `T_total` mean
7,561 s, of which `T_search` 5,214 s. Every run converges early, so `S` is a
genuine time-to-converge comparison, not a budget artefact.

#### Verdict: H3 ≥ H1 ≫ H2

`T_search` is 2.1× larger for IsalSR (5,214 s vs 2,478 s) — a gap of ≈2,736 s that
canonicalisation is already excluded from. H1 is confirmed from the code: DAG
conversion and set operations are outside the timer and inflate `T_search`. But an
order-of-magnitude check bounds it: `T_canon` ≈ 2,347 s at 0.817 ms/DAG implies
≈2.9 M canonicalisations, so even a conversion costing 0.3 ms/DAG accounts for only
≈860 s — roughly a third of the gap. **The remainder is H3: deduplication genuinely
changes Bingo's search trajectory and it converges more slowly in search time.**
That is a property of the method, not of the implementation, and it cannot be
engineered away. The split cannot be pinned down more precisely without
per-generation profiling.

#### Consequence for the revision

Porting the *bookkeeping* (conversion + dedup set) to C++ would genuinely reduce
`T_search^IsalSR` and so genuinely improve `S` — that work is IsalSR-only, so unlike
a faster evaluator it does not cancel between arms. But the bound above says it
recovers at most about a third of the gap, and `S = 0.93` would not reach 1.0 on
that alone. AC-6 will therefore report `S` as unchanged-by-construction plus a
bounded estimate of what bookkeeping removal could recover, and escalate as §6
specifies. **The honest answer to R1.1 is that Bingo's `S < 1` is mostly real.**

### 2026-07-27 — Equivalence harness reported a false PASS; caught on verification

`experiments/scripts/equivalence_gate.py` was delivered reporting
`engine_b: "cpp"`, `self_comparison: false` and **0 mismatches** across all three
gates. None of it was a cross-engine comparison. Both sides ran the Python
canonicaliser.

Verified in the main tree:

```
grep -c backend src/isalsr/core/canonical.py          -> 0
dir(_native)          -> ['build_info','engine_name','fnv1a64','testing']
dir(_native.testing)  -> ['NativeCDLL','NativeLabeledDAG','NativeStringToDAG','tokenize']
```

No `backend` parameter exists on the canonicaliser and no native canonicalisation
symbol exists at all, so `backend="cpp"` could not route anywhere native.

**Root cause.** The harness used `backends.engine()` as its capability probe. That
returns `"cpp"` whenever the **extension imports**, which says nothing about
whether `fast_canonical_string` has a C++ implementation. Conflating "the `.so`
loaded" with "this function is ported" produced a PASS on a test that exercised
nothing.

**Why this one matters more than the others.** This file is the evidence for AC-3,
and `EXECUTION-PLAN.md` §2 gate G1 gates a 36,000 core-hour campaign on it. A
harness that cannot distinguish "verified equivalent" from "compared Python to
itself" is worse than no harness, because it converts an unexamined assumption into
a documented result. Returned for one iteration with instructions to probe the
capability rather than the extension, and to make a self-comparison report
`pass: false` with a stated reason — a self-comparison must never be reportable as
a passing equivalence gate.

**Second finding, kept rather than discarded.** The harness discards ~76 % of
randomly generated DAGs (23.9 % yield) because they contain nodes unreachable from
node 0. The filter itself is correct — I checked line 374 and it uses the
structural criterion (reachability from node 0 via out-edges), not the circular
"the round-trip failed". But the *rate* is evidence for **T06 / R1.2**, which asks
for exactly this reachability-condition failure rate, so it is being promoted to a
reported field (`dags_generated`, `dags_discarded_unreachable`, `discard_rate`)
rather than left in an agent's prose. Note the population differs from T06's: this
is uniformly random synthetic DAGs, whereas R1.2 asks about DAGs arriving at the
canonicaliser during real searches. The two rates are not interchangeable and T06
still needs its own instrumentation.

### 2026-07-27 — First independent speedup measurement: 11.9×, zero mismatches

Taken by me in the main tree, not reported by an agent. The native canonicaliser
now exists — `_native` exposes `fast_canonical_string`, `wl_node_hash` and
`CanonicalTimeoutError` — so a genuine cross-engine comparison is finally possible.

Corpus: 400 randomly generated DAGs surviving a reachability filter, k ∈ [3, 18],
median k = 8, which spans the range the campaign actually exercises.

| Engine | Cost (workstation) | Mismatches |
|---|---|---|
| Python | 0.1705 ms/DAG | — |
| C++ | **0.0143 ms/DAG** | **0 / 400** |

**Speedup 11.9×, byte-identical output.**

#### Projected effect on the published numbers

`OH = T_canon / T_total` and `T_search` is untouched by the engine, so with
`T_canon → T_canon / 11.9` and Bingo's published `OH = 39.2 %`:

```
T_search = 0.608·T_total ,  T_canon = 0.392·T_total
T_canon' = 0.392/11.9 = 0.0329·T_total
T_total' = 0.608 + 0.0329 = 0.641·T_total
OH'      = 0.0329 / 0.641 = 5.1 %
```

| Quantity | As submitted | Projected on C++ |
|---|---|---|
| Bingo `T_canon` | 0.817 ms/DAG | ≈0.069 ms/DAG |
| Bingo overhead | 39.2 % | **≈5.1 %** |
| Bingo total wall-clock | — | **≈36 % lower** |
| UDFS `T_canon` | 0.296 ms/DAG | ≈0.025 ms/DAG |
| UDFS overhead | 0.05 % | ≈0.004 % |
| `S` (both) | 1.07 / 0.93 | **unchanged** — exact algebra, see the root-cause entry |

That is the headline the revision can defend: the overhead R1.1 objects to falls by
roughly 8×, and Bingo's wall-clock penalty — which currently exceeds the Nemenyi
critical difference at `results.tex:193–196` — plausibly collapses inside it.

#### Four caveats, stated because the number will be quoted

1. **Workstation, not Picasso.** Measured on an i7-13700KF. AC-5 requires the
   comparison on a Picasso compute node, and that is where the paper's costs come
   from. The ratio is the transferable quantity, not the absolute.
2. **Synthetic corpus, not the evolved distribution.** Randomly generated DAGs are
   not what Bingo's search produces; the speedup may vary with k, and the k-bucket
   breakdown AC-5 requires is not yet measured.
3. **11.9× is far below IsalHG's 101–180×,** and that is expected rather than
   disappointing: the algorithm still performs the same backtracking, and each call
   now pays a fixed FFI cost to copy the Python `LabeledDAG` into C++. At
   0.0143 ms/DAG that copy is plausibly a large share of what remains, which makes
   it the obvious next optimisation — though 11.9× already clears what the ticket
   needs.
4. **Measured mid-flight** while the P4 workstream was still running. Re-measure on
   the final tree before anything is quoted.

### 2026-07-27 — P4 complete: the canonicaliser is ported, and AC-4 is met

`native/src/{wl,canonical}.cpp`; `canonical.py` gained a keyword-only `backend`
parameter dispatching through `backends.resolve()`. Only `mode="wl_only"` is
native — `wl_tiebreak`, `tuple_only` and the legacy exhaustive canonicalisers
remain Python, which keeps the 6-tuple machinery out of C++ entirely.

Final tree, verified by me:

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4,856 passed, 5 skipped** (1,495 at session start) |
| `mypy src/isalsr/` | clean, 40 files |
| `ruff check src/ tests/` | **clean** — AC-4 now fully met |
| WL hash vector conformance | 57/57 |
| `native/` threading audit | clean of `std::async`, `std::thread`, `thread_pool`, `hardware_concurrency` |

**AC-4 closed.** The six pre-existing `ruff` errors were mine to clear and are now
fixed: `zip(..., strict=True)` in `test_statistical_analysis.py:99` (which also
strengthens the assertion, both sequences being length 4); the unused binding
dropped in `test_bingo_adapter.py:103` while keeping the call as a smoke check;
and the four `E402`s in `test_udfs_adapter.py` covered by a per-file ignore that
follows the convention already established for `tests/integration/*.py` — the file
uses the identical `pytest.importorskip` + `sys.path` pattern, so restructuring
working test code to satisfy a linter would have been the wrong fix.

#### Equivalence holds on the failure path too

My own check, 4,000 generated DAGs, both engines:

| Outcome | Count |
|---|---|
| Succeeded in both, canonical strings identical | 3,994 |
| Raised in both, **identical exception type and message** | 6 |
| Diverged | **0** |

Reproducing failures identically matters as much as reproducing successes: an
engine that raised a different exception type would abort campaign runs
differently, and 3,000 runs would diverge in a way no equivalence test on
successful cases would catch.

#### A distinction T06 needs — and a correction to my own earlier number

**Correction.** An earlier version of this entry reported a "~76–82 % reachability
violation rate". That was wrong. The figure came from my own filter, which
required every node to be reachable from **x₁ alone**. The condition the paper
actually states (`methodology.tex:976`, Theorem Round-Trip Fidelity) is that every
**non-variable** node be reachable from **some variable** — and D2S starts with all
*m* variables already in the CDLL, so reachability from x₁ specifically was never
the requirement.

Re-measured over the same 4,000 DAGs:

| Predicate | Violated |
|---|---|
| reachable from x₁ only (my incorrect filter) | 3,295 / 4,000 (82.4 %) |
| **reachable from any variable — the real condition** | **0 / 4,000 (0.0 %)** |
| canonicalisation actually raises | 6 / 4,000 (0.15 %) |

So there is **no ~80 % violation rate**, and none should be reported. Quoting it
against R1.2 would have invited the obvious question of how a method with an 80 %
precondition failure rate works at all.

**The real finding is sharper.** All 6 failing DAGs **satisfy** the theorem's
stated precondition and still fail, so the hypothesis as written is not
sufficient. And the failure is not a pruning artefact: the exhaustive
`canonical_string`, which does no pruning and computes true lexmin, fails on
exactly the same 6.

Opened as **T15** (`T15-d2s-failure-modes.md`) on Mario's instruction, covering
both the characterisation of the 6 and the failure rate on *real* Bingo/UDFS DAGs
rather than uniformly random ones. Hands over to T07 (theorem hypotheses) and T06
(the definition of a violation, which is not the same event as a failure).

### 2026-07-27 — AC-5 benchmark driver delivered; corpus representativeness queried

`experiments/scripts/bench_canonical.py` implements the §5.4 protocol
(3 warmups, best-of-9 per rep, median of 4 reps, engines alternated in the same
thermal state, fixed seed, JSON provenance). Quick-mode result:

| Bucket | Python (ms/DAG) | C++ (ms/DAG) | Speedup |
|---|---|---|---|
| k < 5 | 0.0243 | 0.0030 | 7.98× |
| 5 ≤ k < 15 | 0.1171 | 0.0123 | 9.52× |
| 15 ≤ k < 32 | 0.7677 | 0.0684 | 11.23× |
| overall | 0.1171 | 0.0123 | **9.52×** |

Speedup rising with k is the expected shape: the per-call FFI copy of the Python
`LabeledDAG` into C++ is a fixed cost, so it dominates at small k and amortises at
large k. It also means the *aggregate* speedup depends on the k-distribution of
whatever corpus is used — which is exactly why the next point matters.

**Concern raised on verification, not accepted as delivered.** The driver builds
its corpus with `make_random_sr_dag(num_vars=1, …)` (line 185), chosen because a
single variable guarantees structural reachability and so gives a 0 % discard rate.
That is a reasonable way to dodge the ~80 % discard problem, but it benchmarks a
distribution the campaign does not run on: the 50-problem suite spans 1–5
variables, and multi-variable DAGs have a different branching structure at the
root. Re-measuring independently across `num_vars ∈ {1, 2, 3, 5}` with a
reachability filter, stratified by the same k-buckets, before any number from this
driver is quoted. My own earlier spot measurement (`num_variables=2`, k ∈ [3,18],
median k = 8) gave 11.9×, against this driver's 9.52× overall — consistent in
magnitude, but the gap is corpus composition and needs pinning down rather than
averaging over.

Projected Bingo overhead from the driver's 9.52×: **6.3 %** (against 5.1 % from my
11.9× measurement). Either way the conclusion holds — 39.2 % falls to single
digits — but the figure quoted in the response letter must come from the Picasso
run required by AC-5, not from the workstation.

### 2026-07-27 — Move executed, Picasso environment stood up, compute-node smoke green

**Move.** `native/` → `src/isalsr/core/native/` (Mario's instruction). The
directory is `native`, not `_native`, so it cannot collide with the extension
module name at all — the namespace-package question simply does not arise.
`wheel.exclude = ["isalsr/core/native/**"]` keeps the C++ out of wheels while the
sources travel with the checkout, which is what let Picasso build straight from a
`git pull`. All four post-move gates pass, plus `mypy`/`ruff` clean and the
equivalence gate still green.

**Correction to finding E-1.** My earlier reconnaissance reported the Picasso repo
absent. It exists at the documented path and was in sync at `bff3654`. The conda
half of E-1 was correct: there was no `isalsr` env, and creating it was real work.

**Environment (AC-1).** `conda create -n isalsr python=3.11` (3.11.15, matching the
workstation), `module load gcc/13.2.0`, build deps cached on the login node
because compute nodes have no outbound internet. Then `pip install -e ".[dev]"`.

**`build_hash` is `298fc1188bf1b051` on both machines** — workstation gcc 12.2.0
and Picasso gcc 13.2.0 — which is exactly the property the `x86-64-v3` decision
was taken for. One build hash across heterogeneous hardware means engine timings
stay comparable and the MANIFEST has a single value to record.

**Compute-node smoke — job 1659650, `COMPLETED 0:0`, 43 s, node `sd051`,
Intel Xeon Gold 6230R.** That is one of the two CPU models the March campaign
ran on, so these numbers are directly comparable to the published ones.

| Stage | Result |
|---|---|
| Build on the compute node (`--no-build-isolation`) | OK |
| Engine identity | `cpp`; `.so` path correct; `isa_level=x86-64-v3`; `avx512f=0` |
| Equivalence gate | **PASS** — `engine_a=python`, `engine_b=cpp`, `self_comparison=false`, 0 mismatches on all three gates |
| Benchmark | see AC-5 table below |

The worker deliberately does **not** prepend `$REPO_DIR/src` to `PYTHONPATH`,
against the generic Picasso worker template. Here that idiom is harmful: the
`.so` installs into site-packages, so a `src`-first path resolves `isalsr` to the
source tree, silently falls back to pure Python, and yields a run that measures
nothing. Every campaign worker must instead assert `backends.engine() == "cpp"`
before doing any work. Documented in `CPP_BUILD.md`.

### 2026-07-27 — AC-5 measured on Picasso hardware

Job 1659650, Xeon Gold 6230R, protocol 3 warmup + 4 reps × best-of-9, engines
alternated in the same thermal state.

| Bucket | n | Python (ms/DAG) | C++ (ms/DAG) | Speedup |
|---|---|---|---|---|
| k < 5 | 50 | 0.0485 | 0.0063 | 7.68× |
| 5 ≤ k < 15 | 50 | 0.2417 | 0.0242 | 10.00× |
| 15 ≤ k < 32 | 50 | 1.7134 | 0.1265 | **13.54×** |
| overall | 150 | 0.2417 | 0.0242 | **10.00×** |

Speedup rises with k because the per-call FFI copy of the Python `LabeledDAG` is
a fixed cost that amortises. The campaign's own k-distribution therefore decides
the aggregate figure, which is why the buckets are reported rather than a single
number.

### 2026-07-27 — AC-2 failed on first test, and the failure was worth having

`ISALSR_ENGINE=python` had been verified, but AC-2 asks for something stricter:
the suite must pass with the extension **absent**. Removing the `.so` broke
`pytest` at **collection** with 4 errors — three native test modules import
`_native` at module scope, and `test_canonical_determinism.py` did too.

A `pytestmark = skipif(...)` guard was not sufficient: those modules dereference
the extension inside `@pytest.mark.parametrize` decorators, which execute during
collection regardless of any skip mark. The working fix is `pytest.importorskip`
at module top for the three modules that are meaningless without the extension,
and a per-test `requires_native` mark for `test_canonical_determinism.py` —
whose Python-side vector tests are the *oracle* and must keep running when the
extension is gone.

One further failure surfaced: `test_cpp_backend_raises_when_unavailable` asserted
`TypeError`. That was correct when `fast_canonical_string` had no `backend`
parameter, but the dispatcher now raises `RuntimeError`. Asserting the narrower
exception made the stricter, better behaviour look like a regression. Widened to
`(TypeError, RuntimeError)` with the reasoning recorded in the docstring, since
the property being defended is "never falls back silently", not the exception
class.

| AC-2 condition | Result |
|---|---|
| Suite with extension **absent** | **1,639 passed, 22 skipped** |
| Suite with extension **present** | **4,878 passed, 5 skipped** |
| `ruff` / `mypy`, both states | clean |

This is precisely the defect AC-2 exists to catch: without it, anyone cloning the
repository without building would find the test suite broken at collection.

#### Deferred: move `native/` under `src/isalsr/core/` (Mario, 2026-07-27) — DONE, see above

C++ sources currently live at repo-root `native/`. They are to be moved under
`src/isalsr/core/` for the final integration — **not now**, while agents hold write
locks on `native/` and `CMakeLists.txt`.

The reason the sources were placed at the root in the first place is a name
collision, and it governs how the move must be done: a directory
`src/isalsr/core/_native/` sits next to the extension `isalsr.core._native`.
CPython's `FileFinder` tries extension-module loaders before falling back to
implicit namespace packages, so the `.so` should still win — but the margin is one
import-machinery detail, and the current build installs the `.so` into
site-packages rather than the source tree, so the two only ever meet in an editable
install. Either give the source directory a distinct name (`_native_src/`) or
accept the collision and prove it works.

Post-move verification, all four required before the move is called done:

1. `python -c "from isalsr.core import _native; print(_native.__file__)"` resolves to
   the `.so`, **not** to a directory.
2. `backends.engine()` still returns `cpp`, and `build_info()` still reports
   `isa_level=x86-64-v3`.
3. `ISALSR_ENGINE=python` still forces the fallback.
4. Full suite still green, and the C++ sources are still excluded from the wheel.

#### Two further findings for other tickets

- **`discussion.tex:37` makes an unsupported numerical claim.** Verbatim: "no false
  collision has been observed across the $14{,}841$ DAGs in the unit-test suite or
  the millions generated during the SR experiments". No corpus of that size exists
  in the repository, and "14841" appears in no source or test file. Verdict (a) —
  a reviewer can ask for it and we cannot produce it. Belongs to **T09**.
- **Two literal canonical strings are printed in the manuscript**:
  `VcVspv*pv+PpcnnC` at `methodology.tex:256` (figure caption, explicitly "the
  canonical string") and `VgnV*C` at `methodology.tex:272` (inside a `\begin{comment}`
  block, so not typeset). Both must be re-derived under the fixed engine before
  submission — this is the manuscript-number carve-out flagged when the fix-both-
  engines decision was taken. `methodology.tex:256` is the one that matters.


### 2026-07-30 — T16 lands; T01 becomes the critical-path blocker; AC-5/AC-6 reopened, AC-8/AC-9 added

**Status change, and it is not a change in this ticket's code.** T16 (the Σ_SR
alphabet correction) completed and passed gate G9 on a Picasso compute node
(job 1692451, native C++ engine, 4/4 production configs). With T06's instrumentation
half and T08's root-cause half already done, **T01 is now the only substantive
`EXECUTION-PLAN.md` §2b blocker left before Wave 1.** §0 records this.

**Three consequences for this ticket, none of which were foreseeable when it was
written.**

1. **The equivalence gate now has a population problem (new AC-8, §5.3 gate 6).**
   Gate 3 replays ≥100,000 DAGs from `…/wl_subtree_unified/`, a campaign produced
   *before* T16. Every one of those DAGs is legacy-encoded — it carries `Sub` and
   `Div` nodes that the adapters can no longer emit. Passing gate 3 as written would
   certify the C++ engine against a label distribution **Wave 1 will never
   canonicalise**. The fix is cheap (replay the same trajectories through the current
   adapters, which decompose inline) but it has to be done deliberately, because
   nothing about it fails loudly: the gate would report a clean pass on the wrong
   corpus.

2. **AC-5's benchmark is stale in a way that matters structurally, not just
   numerically.** Decomposition raises `k` by ~22 % on both hosts, which moves DAGs
   **across the paper's own k-buckets** — Bingo p95 goes 11 → 15, so mass leaves the
   `5 ≤ k < 15` bucket. A speedup table stratified by k is therefore comparing
   different populations per bucket before and after. Report both encodings so T02's
   continuity table can separate the engine effect from the alphabet effect instead
   of confounding them.

3. **AC-6's go/no-go got harder, and this should be said plainly rather than
   discovered in September.** AC-6 asks whether the projected `S` clears 1.0 for
   Bingo. T16 raises per-DAG canonicalisation cost by **+24.6 %** (Bingo) / **+10.8 %**
   (UDFS), and canonicalisation cost is precisely the denominator that has to shrink
   for `S` to rise. The C++ speedup and the alphabet cost increase are **independent**
   and both land in Wave 1; netting them by assumption is exactly the reasoning error
   that would commit 36,000 core-hours to a result we could have predicted.
   The offsetting term is real but small and must not be oversold: ρ was **exactly**
   invariant on Bingo (3,858 distinct strings in both encodings) and rose 1.4 % on
   UDFS. So the representation got strictly better at its job while costing more per
   DAG. **If the recomputed projection does not clear 1.0, escalate before launching
   — R1.1's complaint is about `S`, and an honest negative is worth more than a
   campaign that confirms it slowly.**

**AC-9 added** for the two literal canonical strings in `methodology.tex`. These were
already noted in the 2026-07-27 entry as a manuscript-number carve-out; T16 promotes
them from precautionary to mandatory, because a canonical string containing `-` or
`/` is now unreachable from the adapters by construction. `methodology.tex:256` is
typeset and is the one that matters; `:272` sits inside a `\begin{comment}` block.

**Not changed by T16**: AC-1, AC-2, AC-4 and the determinism work (§5.5) are all
alphabet-independent — `Neg` and `Inv` are ordinary labels the canonicaliser already
handled, and no C++ source was touched. The full suite is green at
**5,258 passed, 5 skipped**, ruff and mypy clean, on commit `582c779`.

---

## 8. Proposed answer

> Fill only when §6 is fully met. T14 pastes this into
> `reviews/response_to_reviewers.tex`. This ticket has no reviewer comment of its
> own; its output feeds T10 (R1.1) and the continuity attachment in T02. Write
> §8.1 and §8.2 anyway — T02 and T10 consume them directly.

### 8.1 Before / after

Measured on Picasso (job 1659650, Intel Xeon Gold 6230R — one of the two CPU
models the submitted campaign used). Protocol: 3 warmup + 4 reps × best-of-9,
engines alternated in the same thermal state.

| Quantity | Python engine (as submitted) | C++ engine (revised) | Source |
|---|---|---|---|
| Canonicalisation cost, Bingo (median, ms/DAG) | 0.817 | ≈0.082 (projected at 10.0×) | `results.tex:58`; AC-5 |
| Canonicalisation cost, UDFS (median, ms/DAG) | 0.296 | ≈0.030 (projected at 10.0×) | `results.tex:57`; AC-5 |
| Benchmark cost, k < 5 (ms/DAG) | 0.0485 | **0.0063** (7.68×) | AC-5 |
| Benchmark cost, 5 ≤ k < 15 (ms/DAG) | 0.2417 | **0.0242** (10.00×) | AC-5 |
| Benchmark cost, 15 ≤ k < 32 (ms/DAG) | 1.7134 | **0.1265** (13.54×) | AC-5 |
| Canon : eval cost ratio, Bingo | **0.63 : 1** (0.817 / 1.29) | ≈0.06 : 1 | `results.tex:58` |
| Canon : eval cost ratio, UDFS | **1 : >1,500** | ≈1 : 17,000 | `results.tex:191` |
| **Bingo overhead** | **39.2 %** | **≈6.1 % (projected)** | `results.tex:58`; AC-6 |
| UDFS overhead | 0.05 % | ≈0.005 % (projected) | `results.tex:57`; AC-6 |
| Projected Bingo `S` at ρ = 1.83 | 0.93 (measured) | **0.93 — unchanged by construction** | AC-6 |
| Projected UDFS `S` at ρ = 1.56 | 1.07 (measured) | **1.07 — unchanged by construction** | AC-6 |
| Canonical strings byte-identical to Python | — | **yes, 0 mismatches** | AC-3 |
| `build_hash`, workstation vs Picasso | — | identical (`298fc1188bf1b051`) | AC-1 |

> **The ticket's §1 cost figures were stale and are corrected above.** §1 states
> Bingo's evaluation at ≈0.14 ms/DAG and a "3.3 : 1 ratio against us"; the
> manuscript prints `T_eval = 1.29 ms`, so canonicalisation was **already cheaper
> than evaluation** (0.63 : 1). The 0.14 ms figure comes from `CLAUDE.md`'s earlier
> 22-problem campaign, superseded by the 50-problem run that was submitted.

> **`S` cannot move, and this is arithmetic rather than a measurement.** Both
> runners compute `search_only = wall_clock − dedup.canon_time_total`
> (`bingo/isalsr_runner.py:519`, `udfs/isalsr_runner.py:277`), matching
> `computational_experiments.tex:115–127`. For any speedup *f*,
> `(wall_clock − T_canon(1−1/f)) − T_canon/f = wall_clock − T_canon`. `S` is
> therefore invariant to canonicalisation speed by construction. Per AC-6 this is
> reported rather than worked around, and escalated.

### 8.2 Changes made to the repository

| Path | Change |
|---|---|
| `CMakeLists.txt` | New. nanobind target `_native`, C++17, `-O3 -march=x86-64-v3 -DNDEBUG -fno-plt -funroll-loops`, LTO, `-static-libstdc++ -static-libgcc`; `ISALSR_NATIVE_MARCH` and sanitizer options |
| `pyproject.toml` | Build backend → `scikit-build-core`; `wheel.exclude` for C++ sources; ruff per-file ignores for `importorskip` modules |
| `src/isalsr/core/native/` | New. 8 translation units + 7 headers: cdll, labeled_dag, node_types, string_to_dag, wl, canonical, probe, bindings |
| `src/isalsr/core/backends.py` | New. `engine()`, `build_info()`, `resolve()`, `DEFAULT_BACKEND`, `ISALSR_ENGINE` override |
| `src/isalsr/core/canonical.py` | `_wl_node_hash` (fixed-constant FNV-1a) replaces seed-salted `hash()`; keyword-only `backend` dispatch |
| `experiments/scripts/equivalence_gate.py` | New. Three gates + JSON report; a self-comparison can never report a pass |
| `experiments/scripts/bench_canonical.py` | New. k-stratified benchmark, §5.4 protocol, JSON provenance |
| `tests/data/wl_hash_vectors.json` | New. Shared hash oracle pinning both engines |
| `tests/unit/test_native_*.py`, `test_canonical_determinism.py`, `test_equivalence_gate.py`, `test_bench_canonical.py` | New. +3,383 tests |
| `slurm/smoke_cpp/{launcher,worker}.sh` | New. Compute-node build + engine assertion + gates + benchmark |
| `docs/engineering/CPP_BUILD.md` | New. Workstation and Picasso build, ISA rationale, the `PYTHONPATH` hazard |

Branch `feature/cpp-core-port`, commits `929a5d9`, `b34cded`.

### 8.3 Draft response text

_(no direct reviewer comment; T10 §8.3 consumes these numbers for R1.1)_

For T10's benefit, the defensible claim is **not** that `S` improves. It is:

> Canonicalisation was re-implemented in C++ behind an unchanged Python API,
> reducing its cost by a factor of 10.0 on the same hardware class used for the
> reported experiments (13.5× on the largest expressions). Bingo's
> canonicalisation overhead falls from 39.2 % to approximately 6 %, and the
> canonicalisation-to-evaluation cost ratio from 0.63 : 1 to about 0.06 : 1.
> Canonical strings are byte-identical between the two implementations.

### 8.4 Residual risk

1. **`S` is unchanged, and R1.1 asked about `S`.** The overhead answer is strong,
   but a reviewer may observe that the specific quantity they questioned did not
   move. The honest position — established here and not to be softened — is that
   `S = 0.93` is *not* caused by canonicalisation cost: with canonicalisation
   already excluded, Bingo–IsalSR needs ≈2.1× more search time than the baseline
   (`T_search` 5,214 s vs 2,478 s over 300 sampled runs). Deduplication changes
   the search trajectory. At most about a third of that gap is bookkeeping the
   port could remove; the rest is a property of the method.
2. **`-march=native` is not used**, so the original concern is void; but
   `x86-64-v3` still assumes AVX2, which holds on all 333 Picasso CPU nodes and
   is asserted in `build_info()`.
3. **Equivalence is verified on synthetic corpora only.** The §5.3 gate over
   ≥100,000 *evolved* DAGs has not been run — the stored trajectories persist
   only aggregate counts, so the replacement (live dual-engine comparison during
   short searches) is implemented in design but not yet executed. **This is the
   main outstanding item on AC-3.**
4. **The determinism fix changes canonical strings** relative to the submitted
   run for roughly 10 % of DAGs, choosing a different representative of the same
   isomorphism class. Class *counts* — hence ρ and the reduction factor — are
   unaffected. But `methodology.tex:256` prints a literal canonical string
   (`VcVspv*pv+PpcnnC`) in a figure caption, and it must be re-derived under the
   fixed engine before submission.
5. **The k-distribution decides the aggregate speedup**, which ranges 7.68×–13.54×
   across buckets. Quoting "10×" without the stratification would be imprecise.
6. **ρ's seed-independence is argued, not proved.** The invariance evidence is 40
   permuted pairs per seed plus the exhaustive k=1..8 gate; a full exhaustive run
   under two hash seeds would settle it properly.
