# Codebase and results pointers

Where the implementation and the result artefacts live, for anyone who has to re-measure or re-run something in response to R1.2, R1.4 or R3.1.

**Repo**: `/home/mpascual/research/code/IsalSR`
**Conda env**: `isalsr` — run everything as `conda run -n isalsr python …`
**Results root**: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/`
(Note: `article/CLAUDE.md` records this as `/media/mpascual/Sandisk2TB/research/isalsr/results/`, which does not exist. The path above is the live one.)
**Paper**: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/69c1637a28a81fea2badda9a` ; The reviewer's comments answer is located in `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/69c1637a28a81fea2badda9a/reviews`, while the main article is in `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/69c1637a28a81fea2badda9a/article/{paper,supplementary}`
---

## Core implementation — `src/isalsr/`

| Path | What it holds |
|---|---|
| `core/canonical.py` | **Fast canonical string.** The `wl_only` mode named in `supplementary.tex:753` lives here. Calls `normalize_const_creation` at lines **95, 146, 231** (guarded by `dag._has_const_nodes()`). |
| `core/labeled_dag.py` | `LabeledDAG`. **`normalize_const_creation()` defined at line 591**, docstring `:592–608` (see `verified-discrepancies.md` D9 — this is R1.3's missing definition). Also `:458` call site, and `:396` a CONST-tolerance note. Contains reachability logic. |
| `core/dag_to_string.py` | D2S encoder. Contains reachability logic — **primary place to instrument for R1.2**. |
| `core/string_to_dag.py` | S2D decoder. |
| `core/cdll.py` | Circular doubly-linked list, primary/secondary pointers. Shared design with IsalGraph. |
| `core/commutative.py` | SUB→ADD∘NEG, DIV→MUL∘INV decomposition. |
| `core/permutations.py` | Internal-node permutation enumeration — the machinery behind the $k!$ synthetic study (Appendix E.2). |
| `core/node_types.py` | Operation-type enum and the `NODE_TYPE_TO_LABEL` table cited in the proof of Theorem A.1 (`supplementary.tex:85–92`). |
| `core/dag_evaluator.py` | Expression evaluation. |
| `evaluation/protected_ops.py` | The $\dagger$-marked protected implementations of Table 1 (Inv, Exp, Log, Sqrt, Pow). |
| `evaluation/constant_optimizer.py` | Constant fitting. |
| `adapters/{sympy,networkx}_adapter.py` | Conversions to/from SymPy and NetworkX. Both contain reachability logic. **`sympy_adapter` is the likely place to build a fixed-order serialization for the R1.4 hash baseline.** |
| `precomputed/` | DAG enumeration + cache (`cache_manager.py`, `enumerate_dags.py`, `atlas_lookup.py`). |
| `search/` | Standalone search (`random_search.py`, `hill_climbing.py`, `operators.py`) — not the host solvers. |

`grep -rn "reachab" src/` hits: `core/labeled_dag.py`, `core/dag_to_string.py`, `adapters/sympy_adapter.py`, `precomputed/cache_manager.py`.

---

## Host-solver integration — `experiments/models/`

The drop-in wrappers described in `computational_experiments.tex:32–43` ("≈200 lines of Python for Bingo, ≈100 for UDFS", `discussion.tex:92`).

| Path | Role |
|---|---|
| `models/bingo/runner.py` (14.3K) | Baseline Bingo run. |
| `models/bingo/isalsr_runner.py` (22.7K) | **IsalSR-augmented Bingo** — the canonical-string hash set at the evaluation boundary. |
| `models/bingo/translator.py` (10.7K) | Bingo stack program ↔ `LabeledDAG`. |
| `models/bingo/adapter.py`, `config.py`, `vendor/` | Bingo config and vendored solver. |
| `models/udfs/runner.py` (6.8K) | Baseline UDFS run. |
| `models/udfs/isalsr_runner.py` (10.5K) | **IsalSR-augmented UDFS.** |
| `models/udfs/translator.py` (8.9K) | UDFS computation graph ↔ `LabeledDAG`. |
| `models/analyzer/` | `aggregation.py`, `statistical_tests.py`, `effect_sizes.py`, `metrics.py`, `cross_method.py` |

`models/analyzer/statistical_tests.py` implements the Demšar-style paired test of Section IV.4 (Shapiro–Wilk screen → $t$ or Wilcoxon, one-sided). **This is where the NaN-handling question from R2.7 gets answered** — check how NaN per-problem means enter or leave the $\delta_i$ vector, and whether $N$ stays 50.

`models/analyzer/aggregation.py` builds the per-problem tables, so the bold/underline assignment that marks `nan` as "better" (R2.7, point 2 in `verified-discrepancies.md` D4) originates here.

---

## Experiment drivers — `experiments/`

| Path | Role |
|---|---|
| `orchestrator.py` (18.8K) | Campaign orchestration. |
| `base_runner.py`, `base_translator.py`, `schemas.py`, `io_utils.py` | Shared run scaffolding and result schemas. |
| `hardware_info.py` | Records the hardware reported in Appendix D.3. |
| `analyze.py` (27.6K) | Top-level analysis entry point. |
| `synthetic_scalability/` | The $5{,}400$-DAG / $k!$-permutation study of Appendix E.2 (Table 9, Fig. 2). |
| `random_dag_experiment/` | Random-DAG generation (Lample–Charton grow procedure, `supplementary.tex:746–750`). |
| `scripts/exp1_shortest_path.py` … `exp6_string_compression.py` | The six arXiv-era intrinsic-property experiments. **Not part of the TPAMI manuscript** — but R3's B2 credits the paper with them (see `reviewer-3.md`). |
| `scripts/analyze_isalsr_*.py` | Deep-dive analyses (advantage factors, synthesis, convergence, hard benchmarks). |
| `scripts/generate_fig_neighbourhood.py`, `generate_algorithm_overview.py`, `figures/` | Figure generation. |
| `scripts/fix_pagie1_outliers.py` | Pagie-1 outlier handling — relevant context: Pagie-1 is the one UDFS problem with a descriptive $d = -0.92$ against IsalSR (`results.tex:154`). |

### Configs — `experiments/configs/`

```
bingo_nguyen.yaml   bingo_feynman.yaml   bingo_hard.yaml   bingo_cherrypicked.yaml   bingo_roundoff.yaml
udfs_nguyen.yaml    udfs_feynman.yaml    udfs_hard.yaml    udfs_cherrypicked.yaml    udfs_roundoff.yaml
nguyen.yaml         feynman.yaml         srbench.yaml      diversity_conjecture_v2.yaml
debug_*.yaml
```

**`srbench.yaml` already exists** and is directly relevant to R3.1:

```yaml
experiment: {name: srbench, seed: 42, n_runs: 10}
data:   {source: "srbench"}
search: {operations: ["+","*","-","/","s","c","e","l","r","^","a"], max_tokens: 100,
         n_iterations: 500000, population_size: 2000}
evaluation: {metric: r_squared, constant_optimization: true, bfgs_max_iter: 200}
```

Two things to note before citing it: `n_runs: 10` (not the 30 seeds used elsewhere), and its operator set includes `r` (sqrt) and `^` (pow) — i.e. it does **not** match the $\{+,-,\times,\div,\sin,\cos,\exp,\log\}$ host set of Appendix D.2. Whether it has ever been run is unverified; no `srbench` results directory exists.

---

## Results — `…/isalsr/results/`

```
results/
├── arXiv_benchmarking/
│   ├── local/{exp1_shortest_path, exp2_neighborhood, exp3_canonicalization_time,
│   │          exp4_search_space, exp5_pruning_accuracy, exp6_string_compression,
│   │          onetoone_properties}
│   └── picasso/{analyze_arxiv, exp1_shortest_path, exp2_neighborhood,
│                exp3_canonicalization_time, onetoone_properties,
│                search_space_analysis, search_space_permutation}
├── model_validation/
│   ├── real_benchmarks/          ← the TPAMI campaign
│   │   ├── wl_subtree/           bingo/, udfs/, analysis/, figures/, slurm_logs/
│   │   ├── wl_subtree_unified/   bingo/, udfs/, analysis/, figures/     ← see below
│   │   ├── wl_subtree_cherrypicked/
│   │   ├── wl_subtree_roundoff/
│   │   └── wl_subtree_hard/models_hard/
│   └── diversity/{diversity, diversity_hard, dedup_smoke, old_diversity}
├── bingo/nguyen/nguyen_1/
├── udfs/
├── cache_validation/{validation, figures}
└── metadata.json
```

**`real_benchmarks/wl_subtree_unified/analysis/` is the most likely source of the manuscript's Tables 2, 3, 6 and 7** — it is the only directory carrying a `three_axis_*` and `cross_problem_dominance_*` set:

```
benchmark_summary_{bingo,udfs}_benchmark.csv
computational_overhead_{bingo,udfs}_benchmark.json
cross_method_benchmark.json
cross_problem_dominance_{bingo,udfs}_{all,benchmark}.json
global_summary.json           (83.7K)
reduction_comparison_benchmark.json
three_axis_global.json        (42.0K)
three_axis_summary_{bingo,udfs}_benchmark.json
```

**Verify this before relying on it.** Several sibling campaigns (`wl_subtree`, `wl_subtree_roundoff`, `wl_subtree_cherrypicked`) carry their own `analysis/` directories, and the manuscript never records which campaign produced which table. Establishing that mapping is a prerequisite for answering D1 (run count), D4 (NaN), E1 (k-stratified overhead mismatch) and E4 (1,465 cells) — each of those asks "which numbers came from where".

`model_validation/diversity/dedup_smoke/` is worth a look for R1.4: the name suggests a deduplication smoke test already exists.

---

## Tests — `tests/`

`tests/{unit,property,integration}/` + `conftest.py`.

The **14,841 DAGs** cited in `discussion.tex:38` ("no false collision has been observed across the 14,841 DAGs in the unit-test suite") come from here. Property-based tests are the natural home for a reachability-violation counter (R1.2).

---

## HPC

Runs execute on **Picasso** (SCBI, UMA), CPU-only, 1 core + 8–16 GB per run, 15–17 h wallclock, SLURM (`supplementary.tex:571–583`). SLURM scripts: `slurm/` at repo root; captured logs under each campaign's `slurm_logs/`.

Before writing or editing any SLURM script, read the **`picasso-sbatch`** skill — it is the source of truth for partitions, GPU/CPU selection flags and wallclock limits, and the values recorded elsewhere have gone stale before. Use `sbatch --test-only` before any real submission. For a fast pre-flight, the **`test-picasso-loginexa`** skill validates a smoke run on the V100 login node with no queue.

Scale reference for R3.1: the current campaign is 6,000 runs at a 12 h budget each. Extending to all 120 AI Feynman equations plus ~250 SRBench problems would be ≈45,840 runs, about 7.6× the present campaign. UDFS is already budget-saturated — 36 of 50 problems report $T \approx 43{,}200$ s for both variants in `table_supplementary_udfs.tex`, so every added problem costs close to the full budget.

---

## Docs inside the repo

| Path | Content |
|---|---|
| `docs/md_files/technical_report/` | Detailed implementation and experimental design. |
| `docs/md_files/design/` | Design docs (3-axis framework, amendments, cache design). |
| `docs/md_files/changes/`, `docs/md_files/tasks/`, `docs/md_files/bibliography/` | Change log, task notes, bibliography. |
| `docs/{js,css,data,images}/` | Companion website source — the anonymised mirror is at `https://little-manifold.github.io/isalsr-anon/`, which is still the URL printed in the **non-anonymous** `computational_experiments.tex:2–4` (see `verified-discrepancies.md` E7). |
| `CLAUDE.md` (38.5K, repo root) | Repo-level instructions. |
