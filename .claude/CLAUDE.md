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
src/isalsr/core/canonical.py         Canonical string (from x_1, 6-tuple pruning)
src/isalsr/core/dag_evaluator.py     Evaluate DAG numerically (topological sort)
src/isalsr/core/commutative.py       SUB/DIV <-> ADD+NEG/MUL+INV conversion
src/isalsr/core/permutations.py      Permute internal node IDs (isomorphic copies)
src/isalsr/core/algorithms/          D2S algorithm variants
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

**Operational requirements**:
- Bingo runners **must** pass `max_time=cfg.max_time` to `evolve_until_convergence()`.
  Without it, evolution runs until `max_evals` (10M), far exceeding SLURM time limits.
  Bingo checks `max_time` every `convergence_check_frequency` generations (overshoot bounded).
- Deduplicators use `set[int]` (hash-based) instead of `set[str]` for `canonical_seen`.
  Reduces per-entry memory from ~150 bytes to ~28 bytes, preventing OOM on long runs.
  Collision probability < 3×10⁻⁶ for 10M entries (birthday bound n²/2⁶⁵).
- Orchestrator resume: validates `run_log.json` content (not just existence) before skipping.
  Corrupt files from OOM/timeout kills are deleted and re-run on next launch.

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
9. **CONST creation edge normalization.** CONST nodes ignore in-edges (evaluation-neutral)
   but need a "creation edge" for D2S reachability. `normalize_const_creation()` moves
   all CONST creation edges to x_1 (node 0). Applied in canonical, is_isomorphic, from_sympy.
10. **Label-aware pruning (B13).** The 6-tuple pruning must partition candidates BY LABEL
    before taking max-τ. Cross-label pruning is invalid (automorphisms preserve labels).
    Implemented in canonical.py for both V (primary) and v (secondary) sections.

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
- `experiments/`, `benchmarks/`: may use anything (torch, matplotlib, pandas, etc.)

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

**Canonical string**: `w*_D = lexmin{ w in argmin |D2S(D, x_1)| }`.
Computed from x_1 only (fixed, distinguished start node).
Uses 6-component structural tuple for backtracking candidate pruning:
`(|in_N1(v)|, |out_N1(v)|, |in_N2(v)|, |out_N2(v)|, |in_N3(v)|, |out_N3(v)|)`

This is a **complete labeled-DAG invariant**: `w*_D = w*_D'` iff `D ~ D'`.

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

### Advisor's Non-Negotiable Constraints

These constraints MUST be enforced at all times. The `proposal-guard` agent checks them:

1. **We do NOT invent a new SR method.** We provide an invariant representation.
2. **Every string modification must be followed by canonicalization.**
3. **The O(k!) search space reduction is the paper's main claim.**
4. **We do NOT pretend to be SR experts.** We are graph theory / combinatorics experts.

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
