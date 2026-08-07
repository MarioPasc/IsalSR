# CLAUDE.md -- IsalSR

## Project Identity

**IsalSR**: Instruction Set and Language for Symbolic Regression.
Authors: Ezequiel Lopez-Rubio (supervisor), Mario Pascual Gonzalez (PhD student).
University of Malaga. Extends IsalGraph (topology-only graphs) to **labeled DAGs**
for symbolic regression with isomorphism-invariant string representations.

> For full mathematical foundation, architecture, and adapter design:
> read `src/isalsr/core/README.md`.

---

## Scientific Mindset

- Approach every task as a world-class symbolic regression scientist: think step by step,
  reason, and justify decisions with literature references and mathematical rigor.
- Do NOT please the user. If something won't work, has theoretical flaws, or is
  scientifically incorrect -- say so. We are doing serious research.
- Be proactive and creative. If a task sparks a connection to another concept,
  report it to the user if it could enhance the research.
- When generating plans for local agents, ensure: (1) the agent has access to
  local code and will know the implementation; (2) provide theoretical background
  so the agent can validate; (3) deliver testable results from code being
  implemented; (4) respect strict folder and code organization for maintainability.
- Prioritize correctness over speed. Every algorithm must be mathematically justified.

---

## Environment

- **Conda env**: `isalsr` (activate: `conda activate isalsr`)
- **Python**: `~/.conda/envs/isalsr/bin/python`

| Command | Purpose |
|---------|---------|
| `python -m pytest tests/unit/ -v` | Unit tests (fast, no external deps) |
| `python -m pytest tests/integration/ -v` | Integration tests (networkx, sympy, scipy) |
| `python -m pytest tests/property/ -v` | Property-based tests (hypothesis) |
| `python -m pytest tests/ -v --cov=isalsr` | Full suite with coverage |
| `python -m ruff check --fix src/ tests/` | Lint + autofix |
| `python -m ruff format src/ tests/` | Format |
| `python -m mypy src/isalsr/` | Type checking (strict) |
| `python -m pip install -e ".[dev]"` | Install in dev mode |
| `python -m pip install -e ".[experiments]"` | Install with experiment deps (bingo, statsmodels, etc.) |
| `python -m experiments.models.orchestrator --config <yaml> --seeds 1 --problems Nguyen-1` | Run experiment |

---

## Architecture Overview

### Instruction Set (Alphabet Sigma_SR)

**Two-tier encoding**: Single-char movement + two-char labeled node insertion.
**Configurable operation set**: All ops in registry; experiments select subsets via YAML.

| Token | Type | Semantics |
|-------|------|-----------|
| `N/P` | Movement | Move primary pointer next/prev in CDLL |
| `n/p` | Movement | Move secondary pointer next/prev in CDLL |
| `C`   | Edge | Directed edge primary→secondary (DAG cycle check; no-op if cycle) |
| `c`   | Edge | Directed edge secondary→primary (DAG cycle check; no-op if cycle) |
| `W`   | No-op | Skip |
| `V+`  | Insert | New ADD node (variable-arity 2+) + edge primary→new |
| `V*`  | Insert | New MUL node (variable-arity 2+) + edge primary→new |
| `V-`  | Insert | New SUB node (binary) + edge primary→new |
| `V/`  | Insert | New DIV node (binary, protected) + edge primary→new |
| `Vs`  | Insert | New SIN node (unary) + edge primary→new |
| `Vc`  | Insert | New COS node (unary) + edge primary→new |
| `Ve`  | Insert | New EXP node (unary) + edge primary→new |
| `Vl`  | Insert | New LOG node (unary, protected) + edge primary→new |
| `Vr`  | Insert | New SQRT node (unary, protected) + edge primary→new |
| `V^`  | Insert | New POW node (binary) + edge primary→new |
| `Va`  | Insert | New ABS node (unary) + edge primary→new |
| `Vg`  | Insert | New NEG node (unary: -x) + edge primary→new |
| `Vi`  | Insert | New INV node (unary: 1/x, protected) + edge primary→new |
| `Vk`  | Insert | New CONST node (leaf) + edge primary→new |
| `v[label]` | Insert | Same as V-variants but from secondary pointer |

**Tokenization**: V/v consume the next character as a label; all others are single-char tokens.
Bare 'c' = edge instruction; 'c' after V/v = COS label. Tokenizer disambiguates by context.

**Commutative encoding**: NEG and INV enable elimination of non-commutative binary ops:
SUB(x,y) = ADD(x, NEG(y)), DIV(x,y) = MUL(x, INV(y)). Use `OperationSet.commutative()`
for a fully commutative alphabet (no SUB, DIV; optionally include POW).
Inspired by GraphSR (Xiang et al.).

**The adapters decompose (T16, 2026-07-30). This is now mandatory, not optional.**
The paper's Σ_SR has **12 labels and no `-` or `/`**: `Pow` is the only
non-commutative operation. `experiments/models/commutative_encoding.py` applies
`SUB → ADD+NEG` and `DIV → MUL+INV` **inline inside both adapters**, so every
consumer of `agraph_to_labeled_dag` / `compgraph_to_labeled_dag` inherits it.
Three hard non-goals: do **not** edit the YAML host operator sets (that changes the
host's search and breaks the paired design); do **not** remove `NodeType.SUB`/`DIV`
or narrow `BINARY_OPS` (S2D must still decode legacy `V-`/`V/` strings); do **not**
decompose inside the canonicaliser (`fcs` stays a pure function of the DAG).
`decompose=False` reproduces the pre-T16 encoding for A/B work only.
Sharing of emitted NEG/INV is **off** (`SHARE_DECOMPOSED_UNARY = False`) — measured
benefit was 0.6 % on k and 0.05 % on ρ, not worth breaking `undecompose()`.
Full write-up: `docs/md_files/changes/t16_commutative_decomposition.md`.
**Every IsalSR number produced before this date used the wrong alphabet.**

### Initial State

For m input variables x_1, ..., x_m:
- **DAG**: m nodes with labels VAR, no edges
- **CDLL**: [x_1, x_2, ..., x_m] in natural order
- **Pointers**: both on CDLL node for x_1
- Variables are pre-numbered and distinguishable (no isomorphism ambiguity)

### Core Data Structures

- **LabeledDAG**: Directed graph with node labels, dual adjacency (in + out), cycle detection.
  Nodes are contiguous integer IDs. Labels are `NodeType` enum values.
- **CDLL**: Array-backed circular doubly linked list. Reused from IsalGraph verbatim.
  Nodes have internal indices (from free list) and store graph node indices as `_data` payloads.
- **Two pointers** (primary, secondary): These are CDLL node indices, NOT graph
  node indices. Use `cdll.get_value(ptr)` to get the graph node.

### Dependency Layering

```
experiments/ benchmarks/  -> anything (torch, matplotlib, ...)
isalsr.search             -> numpy
isalsr.evaluation         -> numpy, scipy
isalsr.adapters           -> optional: networkx, sympy
isalsr.core               -> ZERO external deps (stdlib only)
```

### Key Modules

```
src/isalsr/core/cdll.py              CircularDoublyLinkedList
src/isalsr/core/labeled_dag.py       LabeledDAG (directed, labeled, cycle detection)
src/isalsr/core/node_types.py        NodeType enum, arity registry, label mapping
src/isalsr/core/string_to_dag.py     StringToDAG converter (S2D)
src/isalsr/core/dag_to_string.py     DAGToString converter (D2S)
src/isalsr/core/canonical.py         Canonical string (fast_canonical preferred; greedy-invariant from x_0)
src/isalsr/core/dag_evaluator.py     Evaluate DAG numerically (topological sort)
src/isalsr/core/commutative.py       SUB/DIV <-> ADD+NEG/MUL+INV conversion
src/isalsr/core/complexity.py        Structural descriptors of a DAG + streaming accumulator (T19)
src/isalsr/core/permutations.py      Permute internal node IDs (isomorphic copies)
src/isalsr/core/algorithms/          D2S algorithm variants
src/isalsr/viz/                      DAG + instruction-string + CDLL drawing (canonical)
src/isalsr/adapters/                 NetworkX, SymPy bridges
src/isalsr/evaluation/               Fitness metrics, constant optimization
src/isalsr/search/                   String mutation/crossover, search algorithms
```

### Experimental Framework (`experiments/models/`)

Three-layer architecture for paired comparison experiments (baseline vs IsalSR):

```
experiments/models/
    base_runner.py                   ModelRunner ABC (fit → RawRunResult)
    base_translator.py               ResultTranslator ABC (RawRunResult → RunLog)
    schemas.py                       Unified schemas (RunLog, TrajectoryRow, PairedStats)
    io_utils.py                      I/O helpers (JSON/CSV, folder structure)
    complexity_telemetry.py          Sampled structural telemetry over explored DAGs (T19)
    hardware_info.py                 CPU/RAM/Python version capture
    orchestrator.py                  CLI entry point: iterates (method, problem, seed, variant)
    analyzer/
        statistical_tests.py         Paired t/Wilcoxon, Holm-Bonferroni, Friedman+Nemenyi, McNemar
        effect_sizes.py              Cohen's d + bootstrap CI
        aggregation.py               Seed aggregation, paired stats pipeline
        metrics.py                   R², NRMSE, solution recovery, Jaccard index
        cross_method.py              Cross-method Friedman/Nemenyi on (method × variant) matrix
    udfs/                            UDFS integration (vendored, MIT)
        adapter.py                   CompGraph ↔ LabeledDAG (handles sub_l/sub_r/div_l/div_r)
        runner.py                    Baseline: DAGRegressor wrapper
        isalsr_runner.py             IsalSR: monkey-patches evaluate_cgraph()
        vendor/DAG_search/           Vendored source (unmodified)
    bingo/                           Bingo-NASA integration (pip: bingo-nasa, Apache 2.0)
        adapter.py                   AGraph command_array ↔ LabeledDAG
        runner.py                    Baseline: manual pipeline (matches SymbolicRegressor)
        isalsr_runner.py             IsalSR: subclasses Evaluation._serial_eval()
```

**Adding new models**: Implement adapter, config, runner(s), translator. Register in
`orchestrator.py` factories (`create_runner`, `create_translator`). The analyzer is
model-agnostic — it consumes unified RunLog/TrajectoryRow schemas.

**Rebuilding the C++ extension (read before trusting any `backend="cpp"` result)**:

```bash
python -m pip install -e . --force-reinstall --no-deps    # correct
python -m pip install -e . --no-build-isolation           # SILENTLY FAILS
```

- `--no-build-isolation` requires `scikit_build_core` in the env. It is **not**
  installed, so pip aborts with `BackendUnavailable: Cannot import
  'scikit_build_core.build'` and **the stale `.so` keeps being loaded**.
- Never read the exit status through a pipe: `pip ... | tail` reports `tail`'s
  status, not pip's. Use `set -o pipefail` or `${PIPESTATUS[0]}`.
- **Verify the rebuild actually happened**, every time:
  ```bash
  stat -c "%y" $(python -c "from isalsr.core import _native; print(_native.__file__)")
  ```
  The editable install is scikit-build-core's (`_editable_skbc_isalsr.pth`), and
  it places the `.so` under `site-packages/isalsr/core/`, **not** in the repo
  tree — so a repo-local `find` will not reveal a stale build.
- Python sources resolve from the repo while the extension resolves from
  site-packages, so a C++ edit can appear to have no effect while the Python
  half of the same change works. **Run any core-semantics check against BOTH
  backends**; disagreement is the tell. This is how the 2026-07-29 removal of
  CONST normalisation was caught mid-verification.

**Explored-DAG structural telemetry (T19, 2026-08-07)**

Every C2 run records the distribution of structural descriptors — size, depth,
subexpression sharing, transcendental content, operator-mix entropy — over the
DAGs it actually explores, per `(method, arm, problem, seed)`. It tests
Ezequiel's hypothesis that the `isalsr` arm, and less strongly the `hash` arm,
explores structurally harder DAGs. **No post-hoc pass can recover this**: the
population being described exists only while a search runs.

- **Sampled, at a rule identical across the three arms of a method**, so the
  residual instrumentation cost is common to all three and cancels in every
  arm-versus-arm contrast. Bingo describes the whole population every 25
  generations (its baseline has **no** per-candidate hook — `__call__` is
  per-generation); UDFS every 31st candidate. `complexity_sampling_mode` records
  which. **Measured overhead (probe 1814948): UDFS 0.001–0.003 %; Bingo 0.69 %
  on its longest cell, projecting to ≈0.7 % at campaign scale.** Short converged
  runs show up to 1.96 % purely because the fixed generation-0 sample is
  amortised over few generations — the ratio is asymptotically flat. Raise
  `ISALSR_COMPLEXITY_GEN_FREQ` to halve it if ever needed; generation 0 is
  always sampled, so no run can end with `n = 0`.
- **Never measure on the host's native representation.** `command_array` and
  `CompGraph` carry the host's alphabet, in which `Sub`/`Div` are primitive, so
  `k` would differ between arms by construction rather than by search behaviour.
  All three arms are measured on the post-T16 decomposed `LabeledDAG`.
- **Do not "optimise" this into `canonical.py`.** The WL sweep already computes
  out-degrees and could carry descriptors free, but that helps only the arm that
  needs help least and changes the C++ build hash, invalidating the engine
  equivalence gate and D3.
- **Enabled by default** (`ISALSR_COMPLEXITY=0` is the kill switch). The T06
  ledger defaulted off, was set in no config, and cost a 1,260-run wave that
  recorded five rates of zero — which reads as "no fallbacks" and means "nothing
  was counted" (SP-6).
- Flat scalars land on `SearchSpaceResults`; exact histograms and `NodeType`
  counts go to `complexity.json` beside the run log. Nine descriptors are in
  `METRIC_EXTRACTORS`, five in the CPDT (two-sided on every contrast).
- The `complexity_unique_*` block is `None` on `baseline`, which holds no cache.
  **No headline claim may rest on it.**

Write-up: `.claude/notes/review/tasks/T19-dag-complexity-telemetry.md`.

**Operational requirements**:
- Bingo runners **must** pass `max_time=cfg.max_time` to `evolve_until_convergence()`.
  Without it, evolution runs until `max_evals` (10M), far exceeding SLURM time limits.
  Bingo checks `max_time` every `convergence_check_frequency` generations (overshoot bounded).
- Deduplicators use `set[int]` (hash-based) instead of `set[str]` for `canonical_seen`.
  Reduces per-entry memory from ~150 bytes to ~28 bytes, preventing OOM on long runs.
  Collision probability < 3×10⁻⁶ for 10M entries (birthday bound n²/2⁶⁵).
- Orchestrator resume: validates `run_log.json` content (not just existence) before skipping.
  Corrupt files from OOM/timeout kills are deleted and re-run on next launch.
- **VarAnd clone detection (B12, fix 2026-04-01)**: Bingo's `VarAnd` creates offspring via
  `parent.copy()` when crossover doesn't fire (P=0.6). `AGraph.copy()` preserves
  `fit_set=True`, so `_serial_eval`'s `not indv.fit_set` guard skipped ~36% of offspring.
  Fix: exploit MuPlusLambda's call structure — `__call__` detects the parent evaluation
  (all fit_set=True, _call_count>0) and records `_parent_ids`.  In `_serial_eval`,
  any individual NOT in `_parent_ids` is forced through dedup regardless of fit_set.
  Achieves δ_finite=1.000 at ALL generations (verified 3 seeds × 50 gens).
  Tests: `tests/integration/test_dedup_clone_bypass.py` (8 tests).
  **Production Bingo+IsalSR and diversity experiments need re-execution.**
- **UDFS dedup verified correct (2026-04-01)**: UDFS monkey-patches `evaluate_cgraph` at
  module level. Empirically verified on Nguyen-1 (11,375 calls, 100% intercepted,
  23.6% redundancy, 20/20 canonical spot-checks correct). Zero conversion/canon failures.
  **Latent bug**: UDFS uses `multiprocessing.get_context('spawn')` — spawned workers
  import modules fresh and bypass the patch. All production configs use `processes: 1`
  (confirmed in all 4 YAML configs + SLURM `cpus: 1`), so production results are valid.
  If `processes > 1` is ever needed, the patch must be applied inside worker init.
  Verification script: `experiments/scripts/verify_udfs_dedup.py`.

---

## Critical Invariants

### Invariants (violating these causes silent corruption)

1. **CDLL indices != graph node indices.** Pointers are CDLL node indices.
   To get graph node: `cdll.get_value(pointer)`. NEVER conflate them.
2. **`insert_after(cdll_node, graph_node)`** -- first arg is CDLL index, second is payload.
3. **`LabeledDAG.add_edge(source, target)`** -- both args are graph node indices.
   Edge semantics: source provides input to target (data flows source → target).
   **`_input_order`** tracks insertion order per node (critical for binary ops).
4. **Pointer immobility on V/v.** The pointer does NOT advance after V/v insertion.
5. **`generate_pairs_sorted_by_sum`** must sort by `|a|+|b|` (total displacement cost),
   not `a+b` (algebraic sum). The number of movement instructions emitted is `|a|+|b|`.
6. **DAG cycle check on C/c.** Before adding edge u→v via C/c, check if path v→u exists.
   If yes, the instruction is silently skipped (no-op). V/v never creates cycles.
7. **Variables are pre-inserted.** The m input variables exist before any instructions
   execute. They are NOT created by V/v instructions. They have fixed, known labels.
8. **Operand order for binary ops (B9).** For SUB, DIV, POW: the first `add_edge` call
   sets the first operand. V/v creates the first edge; C/c creates the second.
   Evaluator uses `ordered_inputs()`, NOT `sorted(in_neighbors())`.
   D2S/canonical only create binary ops via V/v from the FIRST operand.
   **What Σ_SR encodes is `ordered_inputs(v)[0]` and nothing further (T18,
   2026-08-03).** Surplus in-edges are emitted by C/c in canonical-traversal
   order, so their positions are not recoverable from the string.
   `is_isomorphic` therefore compares position 0 only. This loses nothing where
   it matters: with in-degree ≤ 2, agreement at position 0 plus edge-set
   preservation forces the rest, so the two readings can differ only at a binary
   node of **in-degree ≥ 3** — which `dag_evaluator` refuses outright and no host
   adapter can emit (each op node gets exactly `arity` `add_edge` calls, and
   duplicates are refused, so in-degree ≤ arity always). Do not restore the
   whole-list comparison: it made `is_isomorphic` strictly finer than the
   canonical string and produced five spurious gate-3 round-trip failures.
   Standing check: gate 3's `dags_with_oversaturated_binary`.
   Write-up: `src/isalsr/core/README.md` §7.3.
9. **CONST creation edge repair (revised 2026-07-27, T15).** CONST nodes ignore in-edges
   (evaluation-neutral) but need a "creation edge" for D2S reachability.
   `normalize_const_creation()` adds `x_i -> c` **only for CONST nodes with in-degree 0**,
   taking the lowest-indexed variable that does not close a cycle. It never removes an
   edge.
   **It is a PRODUCER-side step (updated 2026-07-29, T07).** The host adapters call
   their own specialisation (`_normalize_const_edges`, anchoring unconditionally to
   node 0 — sound because adapter output never makes a VAR an edge target). The
   canonicaliser and `is_isomorphic` do **not** apply it: they assume the precondition
   and raise on an in-degree-0 CONST, which has no encoding in Σ_SR at all. This keeps
   `fcs` a pure function of `D`. Do not reintroduce the call into `canonical.py`,
   `is_isomorphic` or `canonical.cpp` — `normalize_const_creation` is **not
   isomorphism-equivariant** in general (it iterates CONST nodes in node-index order,
   which is exactly what isomorphism permutes); it is equivariant only on
   `𝒞₁ ∪ 𝒞₂` = {reachability holds} ∪ {no VAR is an edge target}.
   **It is the identity on any DAG satisfying the Round-Trip Fidelity reachability
   hypothesis** — that is what makes the canonical string a *complete* labeled-DAG
   invariant. CONST provenance is ordinary structure: two DAGs whose CONST nodes hang
   off different parents are different labeled DAGs and get different canonical strings.
   The former behaviour (relocate *all* CONST in-edges onto node 0) dropped edges when
   `add_edge` refused them as cycle-closing, merged non-isomorphic DAGs, and was not
   evaluation-preserving. Full definition, properties N1–N5 and the five measurements
   (E1–E5) that justify the step: `src/isalsr/core/README.md` §6. See also
   `docs/md_files/changes/d2s_canonicalisation_failures.md`.
10. **Label-aware pruning (B13).** The 6-tuple pruning must partition candidates BY LABEL
    before taking max-τ. Cross-label pruning is invalid (automorphisms preserve labels).
    Implemented in canonical.py for both V (primary) and v (secondary) sections.
11. **Bingo dedup must check ALL offspring (B12).** `AGraph.copy()` preserves `fit_set=True`.
    VarAnd creates unmodified copies when crossover/mutation don't fire (~36% of offspring).
    `IsalSREvaluation._serial_eval` must NOT rely solely on `not indv.fit_set` — use the
    `_established` dict (id → command_array fingerprint) to detect new individuals.

### Edge Direction Convention

- Edge u→v means "u provides input to v" (data flows from u to v).
- For `sin(x)`: edge x→sin. For `x+y`: edges x→+, y→+.
- V/v creates edge from pointer's node to new node (existing → new).
- C creates edge from primary→secondary. c creates edge secondary→primary.

---

## Code Organization Rules

### Dependency Rules (strictly enforced)

- `isalsr.core`: ZERO external deps. Only Python stdlib + typing.
- `isalsr.adapters`: optional deps. Each adapter imports its library independently.
- `isalsr.evaluation`: numpy, scipy. Fitness metrics and constant optimization.
- `isalsr.search`: numpy. String-level search operators and algorithms.
- `isalsr.viz`: may import `isalsr.core`. **matplotlib is imported inside function
  bodies only**, never at module scope. Must NOT be imported by `core`, `adapters`,
  `evaluation` or `search`.
- `experiments/`, `benchmarks/`: may use anything (torch, matplotlib, pandas, etc.)

### Visualization -- `src/isalsr/viz/` is the ONLY supported implementation

All drawing of DAGs, instruction strings and the CDLL lives in `src/isalsr/viz/`.
Import from the package root (`from isalsr.viz import ...`), never from its
submodules -- the public surface in `__init__.py` is the thing that is maintained.

| Entry point | Draws |
|---|---|
| `make_trace_figure` | **Top-level entry.** 2 x N grid; one column per D2S step |
| `draw_dag` | Labeled DAG, dispatched through the backend registry |
| `draw_instruction_strip` | Instruction string as token cells |
| `draw_cdll` | CDLL chain with primary/secondary pointer markers |
| `TraceLayout` | All geometry and font sizes (`show_cdll` toggles the CDLL sub-row) |

- **Do not add drawing code to `experiments/scripts/`.** Scripts that predate this
  package and carry their own implementation (e.g. `generate_algorithm_overview.py`
  and its local `draw_cdll_ring`) are **deprecated**. Leave them as they are; do not
  extend them and do not port new figures onto them.
- Size in-figure fonts for the **final** rendered size. A figure scaled to
  `\linewidth` in the response letter (single-column a4, 2.3 cm margins) is ~6.46 in,
  so a 13.6 in figure renders at ~0.47x and a 9.5 pt annotation lands near 4.5 pt.
- Anchor CDLL traversal with `stable_anchor`, not the primary pointer: starting from
  a moving pointer makes the ring appear to rotate between steps when only the
  pointer moved.

### Coding Conventions

- Full type annotations on ALL function signatures.
- Google-style docstrings on all public functions and classes.
- `__slots__` on performance-critical data structures (CDLL, LabeledDAG).
- No `print()` for diagnostics -- use `logging` or raise exceptions.
- All files under `src/isalsr/` must pass `ruff check` and `mypy --strict`.

### Sibling Project Reference

IsalGraph (topology-only graphs) is at `/home/mpascual/research/code/IsalGraph`.
The CDLL implementation is reused verbatim. The G2S/S2G algorithms are adapted.
All internal imports must use package paths: `from isalsr.core.labeled_dag import LabeledDAG`.

---

## Mathematical Foundation (brief)

**Round-trip property**: For any valid IsalSR string w,
`S2D(w)` is isomorphic to `S2D(D2S(S2D(w), x_1))` as labeled DAGs.

**Canonical string algorithms** (in `src/isalsr/core/canonical.py`):

| Function | Complexity | Use case |
|----------|-----------|----------|
| `fast_canonical_string` | Near-O(k²) | **PREFERRED.** Greedy-invariant, 3 modes. Default: `mode="wl_only"`. |
| `pruned_canonical_string` | O(k! pruned) | Legacy. Exhaustive backtracking with 6-tuple pruning. |
| `canonical_string` | O(k!) | Reference. True lexmin exhaustive (slow). |

**`fast_canonical_string` (preferred, Ezequiel's insight 2026-03-25)**:
At each V/v decision point, candidates are sorted by isomorphism-invariant key.
If the best candidate is unique, it is taken greedily (no backtracking). Ties
are resolved by backtracking over tied candidates only (lexmin among tied).

Three modes via `mode` parameter:

| Mode | Sort key | Default? |
|------|----------|----------|
| `"wl_only"` | `(label_char, WL_hash)` | **YES** |
| `"wl_tiebreak"` | `(label_char, 6-tuple↓, WL_hash)` | No (previous default) |
| `"tuple_only"` | `(label_char, 6-tuple↓)` | No (legacy) |

**WL-only is the default (since 2026-03-27)** because:
1. 1-WL subtree hash is strictly more discriminative than the 6-tuple
   (Weisfeiler & Leman 1968; Shervashidze et al., JMLR 2011).
   WL captures full subtree structure; 6-tuple only depth-3 neighborhood counts.
2. 1.43x mean speedup on evolved Bingo DAGs (k=6-14), range 1.09-1.73x.
3. Completeness verified exhaustively k=1..8 (all k! permutations, up to 40,320)
   and statistically k=10-15 (100 random permutations). 890 tests pass.

The `use_wl_hash` parameter is deprecated; use `mode` instead.

**Search space reduction**: For k internal nodes, O(k!) equivalent labelings
collapse to one canonical string. This is the paper's central contribution.

**DAG distance**: `Levenshtein(w*_D, w*_D')` approximates labeled DAG edit distance.

Full details: `src/isalsr/core/README.md`

---

## Scientific Development Protocol

### 1. Evidence-Grounded Changes
- Every non-trivial decision must cite: a paper, a mathematical justification, or empirical data.
- "I think this is better" is not valid. "This reduces variance because [formula/reference]" is.
- When proposing architectural or methodological changes, state the expected effect and why.
- If no evidence exists, flag it explicitly as a hypothesis and propose a way to test it.

### 2. Research Workflow: Plan -> Test -> Analyze -> Fix
**Planning phase:**
- Break the task into checkable items in `docs/tasks/todo.md`.
- For each item, annotate: objective, success metric, and relevant references.
- Proactively flag: "Based on [paper/method], we could also try X -- want me to include it?"
- Write specs before code. Ambiguity in spec = ambiguity in results.

**Testing phase:**
- Define quantitative success criteria before running anything.
- Log all hyperparameters, seeds, and environment details (reproducibility is non-negotiable).
- Use controlled comparisons: change one variable at a time unless explicitly doing ablations.
- When I report a bug, don't start by trying to fix it. Instead, start by writing a test that
  reproduces the bug. Then, have subagents try to fix the bug and prove it with a passing test.

**Analysis phase:**
- Be proactive: if results reveal an anomaly or improvement opportunity, report it with evidence.
- Propose fixes or enhancements with: (a) what you found, (b) why it matters, (c) what to do.
- Always compute and report: mean, std, confidence intervals or statistical tests where applicable.
- Distinguish between statistically significant and practically significant differences.
- If a metric degrades, investigate root cause before proposing a fix.

**Fixing phase:**
- Fixes must reference what the analysis revealed. No blind patches.
- After fixing, re-run the relevant test to confirm the fix and check for regressions.
- Update `docs/tasks/lessons.md` with the failure mode and the corrective pattern.

### 3. Interdisciplinary Rigor (CS x AI x Mathematics)
- Code changes: justify with computational complexity, memory, or convergence arguments.
- Model changes: justify with loss landscape, gradient dynamics, or information-theoretic reasoning.
- Mathematical changes: justify with graph theory, combinatorics, or formal language theory.
- When in doubt about a mathematical claim, flag it -- do not assume.

### 4. Proactive Scientific Agent Behavior
- During planning and analysis: if you identify a method, paper, or trick that could improve
  the current approach, **propose it immediately** with a one-line rationale.
- Suggest ablations or controls the user may not have considered.
- If a result contradicts expectations, form a hypothesis and propose a diagnostic experiment.
- Never silently ignore warnings, NaNs, or unexpected distributions -- investigate and report.

### 5. Code & Experiment Standards
- All functions: typed, documented (docstring, no usage examples), brief inline comments.
- Prefer libraries over custom implementations. Cite the library and version.
- Logging over print. Use `logging` module with appropriate levels.
- Atomic functions, low cyclomatic complexity, OOP with dataclasses where appropriate.
- Experiment configs: use YAML/JSON, never hardcode hyperparameters in scripts.
- Random seeds must be set and logged. Results must be reproducible.

### 6. Communication Standards
- When reporting results: tables > prose. Include units, dataset split, and N.
- When proposing changes: state the current state, the proposed change, and the expected delta.
- When uncertain: quantify uncertainty. "This might work" -> "This has ~X% chance based on [reasoning]."
- Use LaTeX notation for any mathematical expression in documentation or comments.

### 7. Verification & Self-Correction
- Never mark a task done without quantitative evidence it works.
- After any correction from the user: update `docs/tasks/lessons.md` with the pattern.
- Challenge your own proposals before presenting them. Ask: "What could go wrong?"
- If a subagent is used, verify its output -- trust but verify.

---

## Custom Agents and Skills

### Agents (`.claude/agents/`) -- Use via Agent tool

| Agent | Model | When to use |
|-------|-------|-------------|
| `proposal-guard` | Sonnet | After implementing modules, writing experiments, or any significant code change. Validates alignment with advisor's hypothesis. **MANDATORY after new module completion.** |
| `test-runner` | Haiku | After any code edit for fast pytest + ruff + mypy feedback. |
| `implementation-scientist` | Opus | For complex core modules (LabeledDAG, S2D, D2S, canonical). Implements with mathematical rigor and runs tests. |

### Skills (`.claude/commands/`) -- Invoke with /command

| Skill | Purpose |
|-------|---------|
| `/test-and-verify` | Full pipeline: pytest + ruff + mypy + hypothesis alignment check |

### Skills (`.claude/skills/`) -- TPAMI revision workflow

| Skill | Purpose |
|-------|---------|
| `review-ticket` | Drive one ticket in `.claude/notes/review/tasks/` to completion: plan, delegate, submit and monitor Picasso jobs, verify, write the work log. |
| `review-answer` | Turn a completed ticket into the reviewer-facing answer in `reviews/response_to_reviewers.tex`: audit the ticket's numbers for retractions, interview the user about figures/tables, write to a fixed narrative spine and style contract, verify, push to Overleaf. |

### Advisor's Non-Negotiable Constraints

These constraints MUST be enforced at all times. The `proposal-guard` agent checks them:

1. **We do NOT invent a new SR method.** We provide an invariant representation.
2. **Every string modification must be followed by canonicalization.**
3. **The O(k!) search space reduction is the paper's main claim.**
4. **We do NOT pretend to be SR experts.** We are graph theory / combinatorics experts.
5. **CPDT is the primary statistical significance metric for R² and reduction factor.**
   Per-problem Holm-corrected tests are reported as supplementary detail; the narrative
   and all headline claims must reference the Cross-Problem Dominance Test (pooled N=42).

---

## Key References

- Lopez-Rubio (2025). arXiv:2512.10429v2. IsalGraph preprint.
- Liu et al. (2025). Neural Networks 187:107405. GraphDSR. `docs/bibliography/`
- Xiang et al. GraphSR. Texas A&M / Brookhaven. `docs/bibliography/GraphSR.png`
- Petersen et al. (2021). DSR. NeurIPS.
- You et al. (2018). GraphRNN. ICML.
- Fey & Lenssen (2019). PyTorch Geometric. ICLR Workshop.
- Kahlmeyer et al. (2024). UDFS. IJCAI. DOI:10.24963/ijcai.2024/471.
- Randall et al. (2022). Bingo. GECCO. NASA open-source.

## Detailed Specifications

- @src/isalsr/core/README.md -- Full math, architecture, instruction semantics
- @docs/DEVELOPMENT.md -- Development workflow, testing, experiment commands
- @docs/ISALSR_AGENT_CONTEXT.md -- Full agent context document
- @docs/design/experimental_design/isalsr_experimental_design.md -- Three-axis comparison framework
- @docs/design/experimental_design/experimental_design_amendments.md -- Cache integration amendments
- @docs/design/experimental_design/data_benchmarking_design.md -- Dataset sizes, train/test splits, literature justification
- @docs/md_files/changes/t16_commutative_decomposition.md -- Adapter-level SUB/DIV decomposition, the NEG/INV sharing experiment, and its validation (2026-07-30)
- @docs/md_files/changes/bottleneck_type_analysis.md -- Bottleneck-type analysis: when does IsalSR help? (2026-04-19)
- @docs/md_files/changes/hard_problem_selection_rationale.md -- Why we chose the 10 hard problems (SRBench, McDermott, screening)
- @docs/md_files/changes/candidate_problem_screening.md -- Screening 8 SR benchmark suites for IsalSR-compatible candidates (2026-04-20)
- @docs/md_files/changes/roundoff_problem_selection.md -- 8 problems to round from 42 to N=50 (2026-04-30)
- Save every output in `/media/mpascual/Sandisk2TB/research/isalsr`

## arXiv Search Space Experiment: Controlled Permutation Analysis

**Purpose**: Directly validate the O(k!) search space reduction claim.
Instead of random sampling (which finds ~20% collisions), this experiment
DELIBERATELY constructs all k! isomorphic copies of each expression DAG
by permuting internal node IDs, then verifies canonical invariance.

**Key files**:
- `src/isalsr/core/permutations.py` — `permute_internal_nodes(dag, perm)`: creates isomorphic DAG copy
- `experiments/scripts/search_space_permutation_analysis.py` — main experiment script
- `slurm/workers/search_space_permutation_slurm.sh` — SLURM worker (array: 1 task per k)
- `experiments/scripts/generate_fig_search_space.py` — 2-panel figure (log-scale k! + normalized ratio)

**Metrics**:
- `n_distinct_representations`: structural fingerprint count = k!/|Aut(D)| (exact)
- `n_distinct_d2s`: greedy D2S string count (conservative lower bound)
- `invariant_success_rate`: canonical invariance verification (should be 100%)

**Results** (local test, k=1..8): n_distinct_representations = k! for 64/65 DAGs (one k=8
DAG has |Aut(D)|=2, giving ratio=0.5). Invariant success rate = 100% across all DAGs.

**Launch**: `bash slurm/launch.sh --experiment search_space_permutation`

## Preliminary Experimental Findings (Smoke Tests, 2026-03-18)

| Method | Redundancy Rate | Reduction Factor | Justification |
|--------|----------------|------------------|---------------|
| UDFS   | 6.15%          | 1.07 (k=3)      | Systematic enumeration: few cross-skeleton isomorphisms |
| Bingo  | **41.6%**      | **1.71**         | Stochastic GP: mutation/crossover rediscovers same structures |

Full report: `/media/mpascual/Sandisk2TB/research/isalsr/results/experimental_framework_report_2026-03-18.md`

## Production Results (fast_canonical + WL, Picasso, 2026-03-26)

**2,640 SLURM tasks: 22 problems × 30 seeds × 2 methods × 2 variants.**
Hardware: Intel Xeon Gold 6230R / AMD EPYC 7H12 (Picasso HPC).

### Three-Axis Summary

| Axis | UDFS | Bingo | Status |
|------|------|-------|--------|
| Search Space Reduction | RF=1.56, 34% redundancy | RF=1.28, 22% redundancy | **CHECKED** (100% significant) |
| Regression Quality | Improved (10/22 R² train sig.) | Preserved (0 sig. diff) | **CHECKED** |
| Computational Overhead | **0.6%** (negligible) | **51.0%** (significant) | UDFS CHECKED, Bingo OPEN |

### Computational Overhead (Production)

| Metric | Bingo (659 runs) | UDFS (660 runs) |
|--------|------------------|-----------------|
| Per-DAG canon cost | 0.46 ms (mean) | 0.30 ms (mean) |
| Overhead % | 51.0% (mean) | 0.6% (mean) |
| Fitness eval cost | ~0.14 ms | ~19.4 ms |
| Canon/eval ratio | ~3.3:1 | ~1:64 |

Bingo overhead by k-range: k<5: 44%, k=5-14: 49%, k=15-31: 56%.

### Analysis Pipeline

Run: `python -m experiments.models.analyze --results-dir <path> --methods udfs,bingo --benchmarks benchmark`

Outputs in `analysis/`:
- `benchmark_summary_{method}_benchmark.csv` — per-metric aggregates
- `cross_method_benchmark.json` — Friedman/Nemenyi across methods
- `reduction_comparison_benchmark.json` — RF comparison
- `computational_overhead_{method}_benchmark.json` — overhead analysis by problem and k-range
- `three_axis_summary_{method}_benchmark.json` — executive summary per method
- `three_axis_global.json` — grand summary for LaTeX tables
- `cross_problem_dominance_{method}_benchmark.json` — **CPDT (primary stat. sig.)**
- `global_summary.json` — combined output (includes CPDT)

**Unified results dir**: `/media/mpascual/Sandisk2TB/research/isalsr/results/model_validation/real_benchmarks/wl_subtree_unified`
(42 problems flattened into single `benchmark/` dir; created by `experiments/scripts/merge_results.py --flatten benchmark`)

**Legacy multi-tier dir**: `wl_subtree_full/` (preserves nguyen/feynman/hard/cherrypicked subdirs)

### Cross-Problem Dominance Test (CPDT) — Primary Statistical Significance (2026-04-28)

**Origin**: Ezequiel Lopez-Rubio (PI) proposal.

**Problem**: Per-problem paired tests (30 seeds, Holm-corrected) are underpowered for R²
when most problems saturate near 1.0. Only 1-2/42 problems reach significance individually.

**Solution**: Treat each problem as ONE paired observation. For problem P_i, compute
δ_i = mean(R²_IsalSR) − mean(R²_baseline) across 30 seeds. Run one-sided paired test on
{δ_1, …, δ_N}. Shapiro-Wilk → t-test or Wilcoxon.

**Key property**: As N grows, if IsalSR consistently matches or beats baseline (δ_i ≥ 0),
the p-value decreases monotonically. For N=42 pooled problems with all δ_i ≥ 0, the sign
test alone gives p ≈ 2^{-42} ≈ 2.3×10^{-13}.

**Results (N=42 problems, unified benchmark)**:

| Metric | UDFS | Bingo |
|--------|------|-------|
| R² test p (one-sided) | **0.00018*** | **0.0013** |
| R² test Cohen's d | 0.303 | 0.034 |
| R² test W/T/L | 24/13/5 | 11/29/2 |
| R² train p | 0.000003 | 0.00069 |
| Reduction factor p | ≈ 0 | ≈ 0 |

**CPDT is now the PRIMARY statistical significance metric for R² and reduction factor.**
All tables and the forest plot use CPDT-derived p-values and Cohen's d.

**Implementation**:
- Schema: `CrossProblemDominanceResult` in `experiments/models/schemas.py`
- Function: `compute_cross_problem_dominance()` in `experiments/models/analyzer/aggregation.py`
- Pipeline: integrated in `experiments/models/analyze.py` (per-benchmark + pooled)
- Figures: `experiments/figures/models/generate_tables.py` uses CPDT in Table 1 header,
  CPDT footer rows in Table 2 and Table S; `generate_forest_plot.py` adds CPDT diamonds

## Hard-Tier Benchmark Suite (added 2026-04-13)

**Why**: Bingo solves all 22 existing problems (Nguyen + Feynman) to R² ≈ 1.0,
preventing demonstration of downstream quality / convergence-speed gains from
IsalSR's search-space reduction. The hard tier closes this empirical gap.

**Source**: `docs/md_files/changes/hard_benchmark_proposal.md`.

### 32-problem, 3-tier benchmark

| Tier | Source file | n | Difficulty | Suite key |
|------|-------------|---|------------|-----------|
| Nguyen | `benchmarks/datasets/nguyen.py` | 12 | Easy–Medium | `nguyen` |
| Feynman | `benchmarks/datasets/feynman.py` | 10 | Medium | `feynman` |
| **Hard** | `benchmarks/datasets/hard.py` | **10** | **Hard** | `hard` |

### Hard suite (10 problems)

**Extended Feynman (uniform sampling, 1000 train + 250 test)**:
I.15.10, I.30.3, I.37.4, II.11.27, III.17.37.

**GP-hard classics (per-problem sampling)**:
- Pagie-1 (676 train / 2500 test, 26×26 grid, skip-zero)
- Korns-12 (2000 train / 2000 test, **subsampled** from canonical 10k/10k —
  preserves 5-variable feature-selection difficulty inside UDFS single-process budget)
- Vladislavleva-4 (1024/5000 uniform)
- Vladislavleva-2 (100 uniform train / 221 grid test, step 0.05)
- Keijzer-6 (50 train integers / 120 test integers, **extrapolation**)

### Operator-set extension

The hard configs (`experiments/configs/{udfs,bingo}_hard.yaml`) extend the
production operator set with `sqrt` (and `pow` for Bingo only — UDFS's
vendored search has no generic `pow`). Pagie-1, I.15.10, I.37.4, III.17.37
are otherwise structurally unsolvable. All other hyperparameters
(pop=500, stack=32, cx=0.4, mut=0.4, LM, max_time=43200) match production.

### Sampling protocol dispatch

`hard.generate_data` shares the Feynman signature
`(bench, n_samples, train_ratio, seed)`; per-problem sampling protocols
are encoded in each bench's `sampling` dict (`type`: `uniform | grid_2d_skip_zero |
grid_1d_train_uniform_test_grid | integer_grid`). The orchestrator's
`_get_ground_truth_sympy` was extended to read pre-built `sympy_expression`
keys, unlocking solution_recovery for all 10 hard problems uniformly
(including 4-/5-variable cases).

### SLURM launchers

| Command | Effect |
|---------|--------|
| `bash slurm/hard_launch.sh` | UDFS + Bingo (baseline + isalsr) on 10 hard problems. 4 arrays × 300 tasks = 1200 runs. |
| `bash slurm/hard_launch.sh --dry-run` | Preview sbatch commands. |
| `bash slurm/hard_launch.sh --experiment udfs_hard_baseline` | Single group. |
| `bash slurm/diversity_hard_launch.sh --experiment all` | Diversity on 3 candidates (II.11.27, Korns-12, Pagie-1) as 3 separate 60-task arrays. |
| `bash slurm/diversity_hard_launch.sh --experiment diversity_paramagnetism` | Single diversity benchmark. |

Resources (per task): UDFS baseline/isalsr 8G, Bingo baseline 16G, Bingo
isalsr 128G (heap fragmentation). Time: 15h baseline, 17h isalsr.

Diversity per task: 8h, 8 CPUs, 16G; pop=200, stack=32, max_time=7200, dedup
enforced for isalsr variant. Snapshots match diversity v2.

### Tests

- `tests/unit/test_hard_benchmarks.py` — 51 tests, target_fn correctness,
  sampling shapes, NaN-free outputs, sympy ground truth.
- `tests/integration/test_hard_smoke.py` — 12 tests, Bingo + UDFS on
  II.11.27 / Pagie-1 / Keijzer-6 (one per sampling type), max_time=30s.

### Diversity candidate priority

1. **II.11.27** (paramagnetism, 4 vars, two opposite-sign exp branches) — primary.
2. **Korns-12** (5 vars, 3 irrelevant, high-frequency trig) — secondary.
3. **Pagie-1** — fallback only if screening shows R² ∈ [0.3, 0.9].

## Bottleneck-Type Analysis (2026-04-19)

**Source**: `docs/md_files/changes/bottleneck_type_analysis.md`

IsalSR's advantage is predicted by the problem's **bottleneck type**, not by
source (Feynman vs GP) or individual structural features.

**Core finding**: bottleneck=structural × sig_train → Fisher exact p = 0.0079,
10/10 classification accuracy. IsalSR helps **if and only if** the primary
difficulty is structural search (finding the right operator topology), not
constant optimization, feature selection, or trivial/unsolvable problems.

| Bottleneck | Problems | sig_train |
|---|---|---|
| structural (k=7–8, integer consts) | I.15.10, I.30.3, I.37.4, III.17.37, Pagie-1 | **5/5** |
| none_trivial (R²=1.0 all seeds) | II.11.27 | 0/1 |
| constant (precise real/irrational) | Keijzer-6, Korns-12 | 0/2 |
| structural_depth (k≥12) | Vlad-2 | 0/1 |
| width+constants | Vlad-4 | 0/1 |

**Mechanism**: variance reduction through seed rescue. For structural problems,
IsalSR reduces R² variance by 26–1518× (Levene p < 0.001) and rescues 100%
of below-median baseline seeds. Cliff's δ = 0.48–0.62 (large effect).

**Convergence**: IsalSR pays an early exploration cost (gen 0–500), then
overtakes baseline (gen 500–1000) for structural problems. For non-structural
problems, the cost is never recovered.

**Analysis scripts**: `experiments/scripts/analyze_isalsr_{advantage,advantage_factors,deep_dive,synthesis}.py`
**Paired data**: `experiments/scripts/hard_bingo_paired_data.csv`

## Cherrypicked Benchmark Suite (added 2026-04-20)

**Why**: The hard-tier bottleneck analysis (2026-04-19) found IsalSR helps
**if and only if** the bottleneck is structural search (n_nontrivial_constants=0,
k≥5). To validate this hypothesis on independent data, we screened 8 published
SR benchmark suites (~200 problems) and selected 10 new problems predicted to
show IsalSR advantage. This suite intentionally "cherry-picks" structurally
favorable problems.

**Source**: `docs/md_files/changes/candidate_problem_screening.md` (2026-04-20)

### 50-problem, 5-tier benchmark

| Tier | Source file | n | Difficulty | Suite key |
|------|-------------|---|------------|-----------|
| Nguyen | `benchmarks/datasets/nguyen.py` | 12 | Easy–Medium | `nguyen` |
| Feynman | `benchmarks/datasets/feynman.py` | 10 | Medium | `feynman` |
| Hard | `benchmarks/datasets/hard.py` | 10 | Hard | `hard` |
| Cherrypicked | `benchmarks/datasets/cherrypicked.py` | 10 | Predicted IsalSR advantage | `cherrypicked` |
| **Roundoff** | `benchmarks/datasets/roundoff.py` | **8** | **Portfolio completion (N=50)** | `roundoff` |

### Cherrypicked suite (10 problems)

| Problem | k | n_vars | Source | Sampling |
|---|---|---|---|---|
| I.29.16 (law of cosines) | 11 | 4 | Feynman | uniform 1000/250 |
| I.50.26 (nonlinear oscillation) | 8 | 4 | Feynman | uniform 1000/250 |
| I.16.6 (relativistic velocity) | 6 | 3 | Feynman | uniform 1000/250 |
| II.11.28 (Clausius-Mossotti) | 6 | 2 | Feynman | uniform 1000/250 |
| III.14.14 (Shockley diode) | 6 | 5 | Feynman | uniform 1000/250 |
| Vlad-7 (product + sin) | 9 | 2 | Vladislavleva 2009 | uniform **300/1200** |
| R2 (rational quintic) | 9 | 1 | DSO/Koza | uniform 1000/250 |
| R3 (rational sextic) | 11 | 1 | DSO/Koza | uniform 1000/250 |
| Keijzer-11 (bivariate trig) | 6 | 2 | McDermott 2012 | uniform 1000/250 |
| Liv-14 (poly+trig hybrid) | 8 | 2 | DSO-Livermore | uniform 1000/250 |

k-range: [6, 11]. n_vars range: [1, 5]. All uniform sampling. Vlad-7 uses
published protocol (300 train / 1200 test); all others use 1000/250.

### Execution

Same as `models_hard`: Bingo/UDFS, 30 seeds, same hyperparameters. Only the
problem set and output folder (`models_cherrypicked`) change.

### Key files

| File | Role |
|---|---|
| `benchmarks/datasets/cherrypicked.py` | 10 problem definitions |
| `experiments/configs/bingo_cherrypicked.yaml` | Bingo config |
| `experiments/configs/udfs_cherrypicked.yaml` | UDFS config |
| `slurm/cherrypicked_config.yaml` | SLURM resource config |
| `slurm/cherrypicked_launch.sh` | Phased launcher (UDFS → Bingo → Analysis) |
| `tests/unit/test_cherrypicked_benchmarks.py` | 48 unit tests |

### SLURM launchers

| Command | Effect |
|---------|--------|
| `bash slurm/cherrypicked_launch.sh` | All 4 groups (1200 tasks total) |
| `bash slurm/cherrypicked_launch.sh --dry-run` | Preview sbatch commands |
| `bash slurm/cherrypicked_launch.sh --experiment udfs_cherrypicked_baseline` | Single group |

Resources per task: identical to `models_hard`.

## Roundoff Benchmark Suite (added 2026-04-30)

**Why**: Bring the total benchmark cohort from 42 to N=50. Fills gaps in
DSO-Livermore (only Liv-14), R-rational (R2/R3 but not R1), Pagie (only
Pagie-1), and adds log-heavy and L2-norm structures.

**Source**: `docs/md_files/changes/roundoff_problem_selection.md` (2026-04-30)

### Roundoff suite (8 problems)

| Problem | k | n_vars | Source | Sampling |
|---|---|---|---|---|
| III.10.19 (magnetic moment × L2-norm) | 7 | 4 | AI Feynman | uniform 1000/250 |
| II.11.3 (driven oscillator) | 6 | 5 | AI Feynman | uniform 1000/250 |
| I.13.12 (gravitational PE) | 6 | 5 | AI Feynman | uniform 1000/250 |
| I.44.4 (isothermal work) | 5 | 5 | AI Feynman | uniform 1000/250 |
| R1 (rational cubic) | 7 | 1 | DSO/Koza | uniform 1000/250 |
| Pagie-2 (3D Pagie) | 10 | 3 | Pagie & Hogeweg | uniform 1000/250 |
| Liv-4 (three-log sum) | 8 | 1 | DSO-Livermore | uniform 1000/250 |
| Liv-19 (log-of-polynomial) | 9 | 1 | DSO-Livermore | uniform 1000/250 |

k-range: [5, 10]. n_vars range: [1, 5]. All uniform sampling, 1000/250.

### Key files

| File | Role |
|---|---|
| `benchmarks/datasets/roundoff.py` | 8 problem definitions |
| `experiments/configs/bingo_roundoff.yaml` | Bingo config |
| `experiments/configs/udfs_roundoff.yaml` | UDFS config |
| `slurm/roundoff_config.yaml` | SLURM resource config |
| `slurm/roundoff_launch.sh` | Phased launcher (UDFS → Bingo → Analysis) |
| `tests/unit/test_roundoff_benchmarks.py` | 48 unit tests |

### SLURM launchers

| Command | Effect |
|---------|--------|
| `bash slurm/roundoff_launch.sh` | All 4 groups (960 tasks total) |
| `bash slurm/roundoff_launch.sh --dry-run` | Preview sbatch commands |
| `bash slurm/roundoff_launch.sh --experiment udfs_roundoff_baseline` | Single group |

Resources per task: identical to `models_hard`/`models_cherrypicked`.
