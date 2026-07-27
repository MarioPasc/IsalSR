# T01 — C++ core port + numerical-equivalence gate

| Field | Value |
|---|---|
| Reviewer comments closed | none directly (enabler for **R1.1**, T10) |
| Type | Implementation |
| Owner | **Mario** (+ Claude Code) |
| Depends on | — |
| Blocks | T02, T03, T06 |
| Status | NOT STARTED |
| Target | 2026-08-10 |

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
- **AC-3.** All four equivalence gates in §5.3 pass with **zero** mismatches.
  Evidence: a committed report with counts, not a claim.
- **AC-4.** `python -m pytest tests/ -v`, `ruff check`, and `mypy --strict` all clean.
- **AC-5.** Benchmark table produced: per-DAG canonicalisation time, Python vs C++,
  by k-bucket, on Picasso hardware, with the speedup factor and its dispersion.
- **AC-6.** A projected `S` for Bingo and UDFS computed from the measured C++
  canonicalisation cost and the *existing* ρ values, stated as a projection. This is
  the go/no-go signal for T02: if the projection does not move Bingo's `S` above 1.0,
  say so in §7 and escalate before committing the campaign.
- **AC-7.** §8 filled.

---

## 7. Work log

> Append entries as `### YYYY-MM-DD — <topic>`. Record what you decided and why,
> what broke, and what you could not resolve.

_(empty — to be filled by the implementing agent)_

---

## 8. Proposed answer

> Fill only when §6 is fully met. T14 pastes this into
> `reviews/response_to_reviewers.tex`. This ticket has no reviewer comment of its
> own; its output feeds T10 (R1.1) and the continuity attachment in T02. Write
> §8.1 and §8.2 anyway — T02 and T10 consume them directly.

### 8.1 Before / after

| Quantity | Python engine (as submitted) | C++ engine (revised) | Source |
|---|---|---|---|
| Canonicalisation cost, Bingo (mean, ms/DAG) | 0.817 | | |
| Canonicalisation cost, UDFS (mean, ms/DAG) | 0.296 | | |
| Cost, k < 5 (ms/DAG) | | | |
| Cost, 5 ≤ k < 15 (ms/DAG) | | | |
| Cost, 15 ≤ k < 32 (ms/DAG) | | | |
| Canon : eval cost ratio, Bingo | 3.3 : 1 | | |
| Canon : eval cost ratio, UDFS | 1 : 64 | | |
| Projected Bingo `S` at ρ = 1.83 | 0.93 (measured) | | AC-6 |
| Projected UDFS `S` at ρ = 1.56 | 1.07 (measured) | | AC-6 |
| Canonical strings byte-identical to Python | — | | AC-3 |

### 8.2 Changes made to the repository

| Path | Change |
|---|---|
| | |

### 8.3 Draft response text

_(no direct reviewer comment; leave empty and cross-reference T10 §8.3)_

### 8.4 Residual risk

> What could a round-2 reviewer still object to? Candidates to address here:
> `-march=native` reproducibility across heterogeneous Picasso nodes; whether the
> engine change makes the revision's numbers non-comparable to the reviewed ones
> (this is why T02 produces a continuity table); whether byte-exact equivalence was
> verified on the *evolved* DAG distribution and not only on synthetic corpora.
