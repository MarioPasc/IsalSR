# Codebase and results pointers

Where the implementation and the result artefacts live, for anyone who has to re-measure or re-run something in response to R1.2, R1.4 or R3.1.

**Repo**: `/home/mpascual/research/code/IsalSR`
**Conda env**: `isalsr` — run everything as `conda run -n isalsr python …`, or `~/.conda/envs/isalsr/bin/python -m …` (the module form matters: the scripts are packages, not files).
**Results root**: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/`
(Note: `article/CLAUDE.md` records this as `/media/mpascual/Sandisk2TB/research/isalsr/results/`, which does not exist. The path above is the live one.)
**Paper**: `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/69c1637a28a81fea2badda9a` ; The reviewer's comments answer is located in `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/69c1637a28a81fea2badda9a/reviews`, while the main article is in `/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/69c1637a28a81fea2badda9a/article/{paper,supplementary}`

> **If you are here to touch a number in the manuscript, start at
> §"The three-arm campaign" below.** Every campaign-derived figure in the
> revised paper, supplementary and letter comes from one corpus through one
> pipeline, and one command re-derives and re-checks all of them. The older
> sections of this file describe the code that *produces* results; that section
> describes where the reported results actually came from.

---

## The three-arm campaign, its analysis, and the manuscript

Added 2026-08-14, when the campaign's numbers were carried into the manuscript.
This section supersedes the table-provenance guesswork further down: the
question "which campaign produced which table" now has one answer.

### 1. The corpus

`…/results/review/c2_3arm/` — 12,600 cells, ~80,000 core-hours, commit
`2dd56fd` / tag `campaign/c2`, engine `native` on every cell, integrity-verified
against Picasso. Read its `README.md` first; it carries the layout and five
standing caveats.

```
<method>/<suite>/<problem_slug>/<arm>/seed_NN/{run_log.json, trajectory.csv,
                                              complexity.json, status.json,
                                              fallback_ledger.json,
                                              convergence_log.npz}
<method>/<suite>/<problem_slug>/<arm>/aggregate.csv          420 files, 23 metrics each
<method>/<suite>/<problem_slug>/paired_stats*.json           420 files, 3 contrasts
status_ledger.csv                                            12,600 rows
c2_preflight/stage_c_certification.{md,json}                 19 criteria, verdict GO
```

`method ∈ {udfs, bingo}`, `arm ∈ {baseline, hash, isalsr}`, seven suites
(`nguyen feynman hard cherrypicked roundoff feynman_remainder strogatz`) = 70
problems × 30 seeds.

**Do not write into this tree.** Its digests are certified. The pipeline reads
the 420 `aggregate.csv` and 420 `paired_stats*.json` it already carries and never
recomputes them; the scratch views live under `$HOME/c2_work`.

### 2. The pipeline — `experiments/scripts/review_campaign/`

Code in the repo, outputs beside the data. Read the module README first. One
command rebuilds everything and re-checks it:

```bash
bash experiments/scripts/review_campaign/run_all.sh [CORPUS] [ANALYSES]
```

| Module | Produces | Read it when |
|---|---|---|
| `config.py` | Corpus/analyses paths, suite order, arm labels, `K_RANGES` | You need the canonical suite list or the 50-problem core |
| `extract_cells.py` | `data/cells.csv`, one row per cell, 12,600 rows | **Any new per-cell quantity starts here.** The definitions of `n_eval`, `key_ms`, `eval_ms`, `overhead_pct` and the R² clipping live in its docstring |
| `derive.py` | `per_problem.csv`, `phi.csv`, `speedup.csv`, `overhead_by_k.csv`, `cpdt.csv`, `per_problem_paired.csv`, `values/summary.json` | You want an aggregate. `summary.json` is what the prose quotes from |
| `values.py` | `values/pending_values.{json,md}` — one row per placeholder of the two `PENDING_LEDGER_*.md`, with the LaTeX literal, the raw value and its provenance | You are refilling a placeholder, or checking where a printed number came from |
| `tables.py` | Every `tabular` the three documents `\input` | A table needs a column added or a width fixed |
| `figures.py` | `reduction_factor_distribution.pdf`, `rf_vs_overhead.pdf` | Those two figures |
| `cd_diagram.py` | `cd_2d_udfs.pdf`, `cd_2d_bingo.pdf`, `cd_legend.pdf` | The critical-difference figure. Wraps `experiments.figures.models.generate_critical_difference`, strips the per-panel legends, and draws one shared legend |
| `verify.py` | Pass/fail on every quoted literal against the derived value, plus the placeholder count per document | **Run this after any edit to a number.** Exits non-zero on drift |

### 3. The outputs — `…/c2_3arm/analyses/`

21 MB, generated only. `analyses/README.md` states every definition and the
headline table.

```
data/      cells.csv per_problem.csv per_problem_paired.csv phi.csv
           speedup.csv overhead_by_k.csv cpdt.csv
values/    summary.json  pending_values.{json,md}
tables/    tab_three_axis  tab_cpdt_summary  tab_phi_by_host  tab_key_cost
           tab_supp_phi_per_problem  tab_supp_k_range
           table_supplementary_{udfs,bingo}_body
           tab_letter_suite_breakdown  tab_letter_continuity
figures/   cd_2d_{udfs,bingo}  cd_legend  reduction_factor_distribution
           rf_vs_overhead
pipeline/  flat70/  flat50/  by_suite/     ← raw `experiments.models.analyze` output
```

`pipeline/flat70` and `flat50` are the pooled (N=70) and pre-revision (N=50)
views; `by_suite` is per-suite, which is where the R3.1 breakdown comes from.

### 4. Where each artefact lands in the manuscript

| Artefact | Consumed by |
|---|---|
| `tab_three_axis` | `paper/results.tex` Table 2 |
| `tab_cpdt_summary` | `paper/results.tex` Table 3 |
| `tab_phi_by_host` | `paper/results.tex` Table 4 |
| `tab_key_cost` | `paper/results.tex` Table 5 |
| `cd_2d_{udfs,bingo}` + `cd_legend` | `paper/results.tex` Fig. 3, as two `\subfloat`s under one shared legend in a `figure*` |
| `reduction_factor_distribution` | `paper/results.tex` Fig. 4 |
| `tab_supp_phi_per_problem` | `supplementary.tex` Table 10 (`tab:supp_hash_phi`) |
| `tab_supp_k_range` | `supplementary.tex` Table 11 (`tab:k_range_overhead`) |
| `table_supplementary_{udfs,bingo}_body` | wrapped by `supplementary/table_supplementary_{udfs,bingo}.tex` |
| `rf_vs_overhead` | `supplementary.tex` Fig. `fig:empirical_scalability` |
| `tab_letter_suite_breakdown` | `reviews/response_to_reviewers.tex`, R3.1 |
| `tab_letter_continuity` | `reviews/response_to_reviewers.tex`, continuity appendix (T02 AC-6) |

**Prose that quotes campaign numbers**, and therefore needs re-checking whenever
the corpus changes: `paper/main.tex` (abstract), `paper/results.tex`,
`paper/computational_experiments.tex` §4.1 and §4.4, `paper/methodology.tex`
(after `eq:phi_rho`), `paper/discussion.tex`, `paper/conclusion.tex`;
`supplementary.tex` §D.2–D.3, §sec:supp_hash*, §sec:supp_scalability_empirical;
`reviews/response_to_reviewers.tex` cover letter, R1.1, R1.4, R2.6, R2.7, R3.1.

**The double-blind tree mirrors these.** Nine content files are byte-identical
to their `article/` counterparts and three carry localised anonymisation deltas
(`main_anonymous.tex`, `computational_experiments.tex`,
`supplementary_anonymous.tex`). Re-sync by recovering each delta as a patch from
the last synced commit and re-applying it to the edited file — do not retype the
rules. `double_blind/paper/{introduction,related_work}.tex` still name
IsalGraph; that predates this work and is unresolved.

### 5. Five things that will bite you

1. **ρ against the native arm has no p-value and must not be given one.** ρ ≡ 1
   there by construction. `analyzer/aggregation.py:CPDT_CONTRAST_POLICY` marks
   the contrast `descriptive_definitional_baseline` and emits `NaN`. The
   inferential contrast for ρ is IsalSR against the naive hash.
2. **Per-evaluation cost must never be read from the Bingo native arm.** That
   arm counts candidates through a different mechanism and reports ~9× as many
   (`c2_3arm/README.md` §5). The two deduplicating arms agree on both hosts.
3. **`S` is degenerate wherever both arms exhaust the budget.** 1,727 of the
   2,100 UDFS IsalSR cells sit at the 12 h cap, where `S = 1` whatever the
   method does. Report the strata apart.
4. **The same-stream φ does not exist.** Shadow sketches ran off on all 12,600
   cells, a deliberate decision recorded in the configs themselves and in
   `audit.md` §7.3. φ is a cross-arm estimate through φ = 1 − r_σ/r. Any text
   promising a within-stream measurement or HyperLogLog estimates is stale.
5. **`--allow-mixed-provenance` is required when analysing a flattened view**,
   because flattening pools seven per-suite configuration digests. Commit, build
   hash and engine are uniform across all 12,600 cells, which is the guarantee
   that matters.

### 6. A correctness fix that travelled with this work

`experiments/models/analyzer/statistical_tests.py:critical_difference_data`
computed the Nemenyi critical difference from the raw studentized range. Demšar
(2006 §3.2.2) defines it from the studentized range **divided by √2**, and the
divided values reproduce his Table 5 to three decimals for every k from 2 to 10.
The threshold was 41 % too wide, so the diagram declared differences
indistinguishable that the test separates — including in the submitted figure.
Fixed; pinned by `tests/unit/test_nemenyi_critical_difference.py` (11 tests).

`experiments/figures/models/generate_critical_difference.py` also carried a
four-entry cycle list against six plotted groups, so once the hash arm existed
half the markers took another arm's colour. The list is now built from the
groups actually plotted, and `generate_cd_2d` takes `out_stem`,
`treatment_labels` and `mark_override`.

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
| `adapters/{sympy,networkx}_adapter.py` | Conversions to/from SymPy and NetworkX. Both contain reachability logic. (An earlier note proposed `sympy_adapter` as the place to build R1.4's fixed-order serialization. **That would have been wrong**: the key must read the *host's* stored structure, and any intermediate representation that renumbers nodes has already done part of the canonicalization. The arm keys on Bingo's `command_array` row order and UDFS's `node_dict` order instead — see `experiments/models/{bingo,udfs}/isalsr_runner.py` and T04.) |
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

`models/analyzer/statistical_tests.py` implements the Demšar-style paired test of Section IV.4 (Shapiro–Wilk screen → $t$ or Wilcoxon, one-sided), and `critical_difference_data` the Nemenyi threshold. R2.7's NaN question was answered here: non-finite entries are ignored when forming a problem's seed mean, so $N$ stayed 50 in the submitted analysis. **The √2 bug in the critical difference lived in this file** — see §6 of the three-arm section above.

`models/analyzer/aggregation.py` builds the per-problem tables, so the bold/underline assignment that marked `nan` as "better" (R2.7, point 2 in `verified-discrepancies.md` D4) originated here; it is fixed, and `CPDT_CONTRAST_POLICY` in the same file is what decides which contrasts get a p-value at all.

`models/analyze.py` is the campaign-level driver: `CPDT_CONTRAST_POLICY`-aware, three-arm, with `--variants`, `--allow-incomplete` and `--allow-mixed-provenance`. It writes into `<results-dir>/analysis/`, which is why the review pipeline copies that directory out rather than pointing the manuscript at it.

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
| `scripts/review_campaign/` | **The revision's analysis pipeline.** See §"The three-arm campaign" above. |
| `figures/models/generate_tables.py` | Three-arm table generator (Table 1, per-problem tables, k-range). Predates `review_campaign/tables.py`; still useful as a cross-check, but its supplementary table is two-arm and its problem labels come from directory names, so it mislabels the lowercase D2 suites. |
| `figures/models/generate_critical_difference.py` | The `critdd` wrapper the CD figure is drawn through. Now takes `out_stem`, `treatment_labels` and `mark_override`, and builds its colour cycle from the plotted groups. |
| `scripts/c2_certify.py` | The 19-criterion campaign certifier. Two traps: pass `--expected-tasks` (a **cell** count) and `--seeds`, or its smoke-calibrated defaults make it certify an unverified tree or fail a complete one. It also resolves from the deployed tree, so a patched copy must be loaded by file path. |
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
├── review/
│   └── c2_3arm/                  ← the campaign the revision reports. See §"The
│       ├── udfs/  bingo/            three-arm campaign" above; it supersedes
│       ├── analyses/                everything in this subsection
│       ├── c2_preflight/
│       ├── status_ledger.csv
│       └── README.md
├── bingo/nguyen/nguyen_1/
├── udfs/
├── cache_validation/{validation, figures}
└── metadata.json
```

> **Superseded 2026-08-14.** Everything below in this subsection describes the
> *submitted* campaign. The revision reports `review/c2_3arm` exclusively, and
> the provenance question the paragraph below could not answer is now answered
> by `c2_3arm/analyses/` and its README. Keep this text for reading the
> submitted numbers; do not source a revised table from it.

**`real_benchmarks/wl_subtree_unified/analysis/` is the most likely source of the *submitted* manuscript's Tables 2, 3, 6 and 7** — it is the only directory carrying a `three_axis_*` and `cross_problem_dominance_*` set:

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

Tests that pin a *reported* statistic, and so must not be relaxed:

| File | Pins |
|---|---|
| `unit/test_nemenyi_critical_difference.py` | The Nemenyi constant against Demšar (2006) Table 5 for k = 2…10, the 1/√N scaling, and the 3-group vs 6-group thresholds the manuscript quotes (0.396 and 0.901) |
| `unit/test_three_arm_stats.py` | Three-arm paired-statistics plumbing |
| `unit/test_table_nan_integrity.py` | An undefined value can never carry the "better" mark (R2.7) |
| `unit/test_stats_fairness_fixes.py` | Pairing by seed number rather than by position |

Fast loop for anything in `experiments/models/analyzer/`:
`python -m pytest tests/unit -q -k "statistical or analyzer or cross_method or aggregation"` (73 tests, ~3 s).

---

## HPC

Runs execute on **Picasso** (SCBI, UMA), CPU-only, 1 core + 8–16 GB per run, 15–17 h wallclock, SLURM (`supplementary.tex:571–583`). SLURM scripts: `slurm/` at repo root; captured logs under each campaign's `slurm_logs/`.

Before writing or editing any SLURM script, read the **`picasso-sbatch`** skill — it is the source of truth for partitions, GPU/CPU selection flags and wallclock limits, and the values recorded elsewhere have gone stale before. Use `sbatch --test-only` before any real submission. For a fast pre-flight, the **`test-picasso-loginexa`** skill validates a smoke run on the V100 login node with no queue.

Scale reference for R3.1, updated after the revision campaign: it ran **12,600 runs** at a 12 h budget each (2 hosts × 3 arms × 70 problems × 30 seeds), ≈80,000 core-hours, median wall 18,981 s, peak RSS 9.42 GB against a 32 GB request. The submitted campaign was 6,000 runs at two arms. Extending to all 120 AI Feynman equations plus ~250 SRBench problems at three arms would be ≈66,600 runs, about 5.3× the campaign just run. UDFS remains budget-saturated — 1,727 of its 2,100 IsalSR cells sit at the cap — so every added problem costs close to the full budget.

The 128 GB Bingo–IsalSR request is history: the compiled deduplication set stores 64-bit hashes and the whole campaign fitted a 32 GB request. Size a future one from measurement, not from that number.

---

## The manuscript tree

`…/article/journal/69c1637a28a81fea2badda9a`, a git repo of its own (branch
`master`, remote is Overleaf). Five compiled documents.

| Path | Builds | Notes |
|---|---|---|
| `article/paper/main.tex` | 18 pp | `\input`s introduction, related_work, methodology, computational_experiments, results, discussion, conclusion, and the four `tab_*.tex` |
| `article/supplementary/supplementary.tex` | 17 pp | `\input`s `table_supplementary_{udfs,bingo}.tex`, which in turn `\input` the `_body` files |
| `reviews/response_to_reviewers.tex` | 35 pp | Self-contained bibliography; `\input`s the two `tab_letter_*.tex` |
| `double_blind/paper/main_anonymous.tex` | 18 pp | Mirror |
| `double_blind/supplementary/supplementary_anonymous.tex` | 17 pp | Mirror; a full copy of the supplementary carrying four anonymisation deltas, not a wrapper |

Build any of them with two `pdflatex -interaction=nonstopmode` passes from its
own directory. There is no bibtex step: `main.bbl` and the letter's
`thebibliography` are checked in.

**Marker conventions.**

- `\added{...}` — the `changes` package, loaded as
  `\usepackage[commandnameprefix=ifneeded]{changes}`. Blue in the annotated
  build; `[final]` strips every mark and yields the submission PDF. There is one
  tree, not two: the marked-up manuscript *is* the source.
- `\pendingnum{}` / `\pendingblock{}` — red `[PENDING …]` scaffolding for values
  awaiting a campaign. **Not** a revision mark, and it must survive `[final]`.
  Every occurrence is recorded in `reviews/PENDING_LEDGER_{paper,supplementary}.md`
  with the literal it replaced. The paper's are all resolved and its scaffolding
  block is deleted from `main.tex`; the supplementary keeps 12, all in the
  synthetic permutation study, so its preamble block stays.
- `\pendingnum` is invoked in *text* mode and its argument carries no `$…$`.
  Putting it inside `$…$` closes the outer math mode and breaks the build.

**Standing gates.** `grep -c "color{red}" *.tex` returns 0 (the pending colour is
named `pendingred`); Definition 3.2 and Theorems 3.13/3.14/3.15 keep their
numbers; the double-blind tree carries no author name, affiliation or real URL.

**Table widths are tight.** Four of the new floats needed `\setlength{\tabcolsep}`
between 1.2 pt and 4 pt to fit a column; the generators emit that line themselves,
so re-running `tables.py` preserves the fit. Check `Overfull \hbox` in the log
after any column change — the target is zero across all five documents.

---

## Docs inside the repo

| Path | Content |
|---|---|
| `docs/md_files/technical_report/` | Detailed implementation and experimental design. |
| `docs/md_files/design/` | Design docs (3-axis framework, amendments, cache design). |
| `docs/md_files/changes/`, `docs/md_files/tasks/`, `docs/md_files/bibliography/` | Change log, task notes, bibliography. |
| `docs/{js,css,data,images}/` | Companion website source — the anonymised mirror is at `https://little-manifold.github.io/isalsr-anon/`, which is still the URL printed in the **non-anonymous** `computational_experiments.tex:2–4` (see `verified-discrepancies.md` E7). |
| `CLAUDE.md` (38.5K, repo root) | Repo-level instructions. |
