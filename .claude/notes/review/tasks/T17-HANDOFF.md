# T17 handoff — Stage C submitted and green; what's left

**Session date**: 2026-08-03
**Branch**: `feature/cpp-core-port`
**Last commit at handoff**: see `git log --oneline -1` (session ended around `2365c82`+)
**Ticket**: `.claude/notes/review/tasks/T17-c2-submission-certification.md`

---

## 0. Read these first, in this order

| # | File | Why |
|---|---|---|
| 1 | `.claude/notes/review/tasks/EXECUTION-PLAN.md` **§11.1** | The decision/anomaly ledger. Everything this session found is recorded there with evidence. **Start here** — it is denser and more current than this file |
| 2 | `.claude/notes/review/tasks/T17-c2-submission-certification.md` **§6 Work log** | Per-check results, the SP-1…SP-6 table, and the three ticket criteria corrected against the code |
| 3 | `.claude/notes/review/tasks/T18-canonical-completeness-operand-order.md` | **New ticket.** The completeness defect, its mechanism, and what must be established next |
| 4 | `docs/md_files/changes/t18_completeness_counterexamples.md` | Full per-case detail for the five counterexamples |
| 5 | `docs/md_files/changes/canonical_completeness_counterexamples.md` | The earlier write-up; T18 supersedes its analysis but this has the ruled-out hypotheses |
| 6 | `EXECUTION-PLAN.md` §4.3 / §4.4 / §4.5 | Stage C criteria, Stage D, Stage E — what comes next |

Do **not** re-read the whole plan. §11.1 plus T17 §6 is enough to resume.

---

## 1. What was done

**Stage C — the 1,260-task certification array — was submitted and completed
1,260/1,260 with zero failures.** Output root:

```
picasso:/mnt/home/users/tic_163_uma/mpascual/fscratch/results/isalsr/c2_smoke/
```

### Harness built (all new, all committed)

| Path | Role |
|---|---|
| `slurm/c2_smoke/deploy.sh` | Sync **including `.git`** + verify remote HEAD/cleanliness from the remote side + rebuild with `gcc/13.2.0`. **Use this, not a bare rsync** — SP-1 is unsatisfiable otherwise |
| `slurm/c2_smoke/launcher.sh` | 42 arrays (`--dry-run`, `--test-only`, `--one-task`, `--only m:arm:suite`) + dependent aggregation job |
| `slurm/c2_smoke/worker.sh` | One array task = one `(problem, seed)`; passes `--ledger --postprocess skip` |
| `slurm/c2_smoke/aggregate_worker.sh` | `--postprocess only` over all 14 configs, then the certifier |
| `slurm/c2_smoke/stage_b_{launcher,worker}.sh`, `stage_b4_worker.sh`, `stage_b_probe.py` | B1/B2/B3/B4/B9 |
| `slurm/c2_smoke/verify_build.py` | SP-2 assertion (in a file, not a heredoc — see §3) |
| `experiments/scripts/c2_task_spec.py` | Array-index → `(problem, seed)`. The **only** place the registry lives |
| `experiments/scripts/c2_certify.py` | All 19 criteria (C1.1–C1.17, C2, C4). Exit 0 only if every blocking one passes |
| `experiments/scripts/c2_stage_a_evidence.py` | A4/A4b/A5/A11 artefacts |
| `experiments/scripts/t18_completeness_counterexamples.py` | T18 extractor |

### Code changes

- `orchestrator.py`: `--postprocess {auto,skip,only}` + `postprocess_output_root()`; `nodedup` arm.
- `schemas.py` / `analyzer/aggregation.py`: `PairedStats.n_seeds`, per-metric `n`, `aggregate.csv` `n` column. Legacy C1 files load with `None` (verified on a real `Korns-12` artefact).
- `bingo/translator.py`: trajectory `n_dags_explored` source fix.
- `{udfs,bingo}/isalsr_runner.py`: `dedup_enabled` switch for C3.

---

## 2. Results that matter

| Quantity | Value | Consequence |
|---|---|---|
| **Stage C outcome** | **1,260/1,260 COMPLETED, 0 failed** | C1.1 satisfied |
| **Achieved concurrency (§8.2)** | **≈245 cores** (315 core-h ÷ 77.1 min) | C2 = 411 h ≈ 17.1 d → **misses 2026-09-03 by ~3 d, fits the 2026-09-10 freeze**. §8.3 trade decision is **due now** |
| **Peak MaxRSS over 1,260** | **0.67 GB** | Requests cut 48 GB → 16 GB. **Does not resize production** — D1.2 at 12 h does |
| **T06 counter overhead (B9)** | UDFS **0.04 %**, Bingo **0.22 %**, both live | **Keep the counters** (T06 AC-10 answerable on evidence) |
| **Alphabet (B3)** | 65,631 live Bingo candidates, **0 forbidden labels**, max k = **37** | T16 decomposition confirmed reaching the canonicaliser |
| **Datasets (B1)** | **70/70** resolve, **70/70** SymPy ground truth | C1.5 precondition met |
| **Aggregation cost** | ~7 min/config, ~1 h 40 for 14 over 1,260 runs | **C2's aggregation job needs ≥24 h, not 2 h** |

---

## 3. Defects found and fixed — do not reintroduce these

1. **`sbatch --export` cannot carry a comma in a value.** `C2_SEEDS=0,101,102` arrived as `C2_SEEDS=0`. 265/1,260 tasks died; the rest silently produced **correct-looking seed-0 cells**. Ship colon-separated; the worker asserts its decode. *The 42-task probe could not catch this — index 1 is valid under both readings.*
2. **Per-task post-processing was a race.** `aggregate.csv`, the paired contrasts and a full-tree ledger walk ran after every cell → 1,260 concurrent GPFS writers. Hence `--postprocess skip|only`.
3. **Bingo `trajectory.csv`** mixed LM-inflated fitness calls (rows 1..n−1) with candidate DAGs (last row) → a 3.6× drop. **ρ was never affected.**
4. **B9 reported live counters as dead** — `run_log.json` nests under `results` and names the cost block `time`, not `computational_cost`.
5. **B3 could pass over an empty candidate stream** — stale scratch + resume logic → `DAGs observed: 0` with `no_forbidden_labels` vacuously true.
6. **SP-1 was structurally unsatisfiable** — rsync excluded `.git`.
7. **C1.13 read the SymPy-rendered best expression, not the candidate stream** — SymPy writes `sqrt(x)` as `Pow(x,1/2)` and `x/y` as `x·Pow(y,−1)`. Now: `-`/`/` blocking, `Pow` disclosed.
8. **Never edit a config while an array reads it** — `config_sha256` is per-run from disk.
9. **Never pipe `module load`** — subshell, PATH change lost, silently builds with system g++ 7.5.0.

---

## 4. Open, in priority order

### 4.1 Immediate
- **Read the Stage C certification verdict**: `c2_smoke/c2_preflight/stage_c_certification.{json,md}`, job **1753134** (was still running at handoff). If it timed out at its 2 h wall, re-run the certifier alone — the postprocess artefacts persist.
- **Fill `c2_preflight/smoke_vs_C1.md` §3** (C5). §2's expectations are **pre-registered and frozen**; do not edit them.
- **Fill `EXECUTION-PLAN.md` §11.2** Stage C rows and §11.3's launch ledger.

### 4.2 T18 — the completeness defect
See the ticket. **T18.1 (rate on evolved candidates) is the one that decides how much this matters.**

### 4.3 Blocking C2, outside T17's scope
| Item | State |
|---|---|
| **A6** MANIFEST schema + validator | `experiments/models/manifest.py` **does not exist** |
| **A8** three-arm analyzer | `analyze.py` hardcodes `["baseline","isalsr"]` (~lines 108–115, 148, 360–366). Pairwise CPDT with Holm over **three** contrasts not implemented. **Blocks Stage E** |
| **A1** `campaign/c2` tag | Deliberately not created — must sit on the *final* commit |
| **`n_seeds: 30`** in 10 configs | The five D1 suites × both methods. §0.4a fixes C2 at 20 |
| **A13 quota** | 🔴 FSCRATCH **27.2k** file headroom vs the **≥60k** criterion; C2 needs ≈45k. HOME over quota, grace expiring, **436 GB of it `~/execs/vena`** (different project) |
| **B8** resume/idempotency | Only half exercised; the deliberate-corruption half is outstanding |
| **B5/B6** node census / constraint | Census arrives free from Stage C's per-run `cpu_model`. **The C2 pinning decision is still open** |

### 4.4 Stage D (after C5 signs off)
12 tasks at the full 43,200 s budget, per §4.4. **D1.2 is what sizes production `--mem`** — not Stage C.

---

## 5. Two things that are Mario's / Ezequiel's call, not an agent's

1. **T18** — whether to fix the encoding or state completeness as conditional on commutativity. If fixed, **every ρ in C2 must be recomputed**, so the sequencing decision comes first.
2. **§8.3 trade** — at 245 cores C2 misses the 2026-09-03 target. The cheapest lever (accelerate T04/T05) is already spent.

**Accepted and needing no further work** (Mario, 2026-08-03): Bingo is not seed-reproducible — three identical `baseline --seeds 0` runs gave 155,449 / 41,023 / 41,049 candidates and two different expressions. C3 is therefore bounded rather than exact for Bingo (wrapper perturbation 3 inner evals < baseline self-noise 12); **UDFS passes C3 outright**. To be noted in the paper, not fixed.

---

## 6. Operational notes

```bash
# deploy (ALWAYS this, never a bare rsync)
bash slurm/c2_smoke/deploy.sh

# Stage C
bash slurm/c2_smoke/launcher.sh --dry-run | --test-only | --one-task | (bare)

# monitoring — squeue -u is REJECTED by Picasso's Lua wrapper; use bare squeue
ssh picasso 'sacct -j $(paste -sd, ~/execs/isalsr/c2_smoke/logs/job_ids.txt) -X -n -P -o State | sort | uniq -c'
# memory: NEVER -X (returns empty MaxRSS); filter the .batch step
ssh picasso "sacct -j <ID> -n -P -o JobID,MaxRSS | awk -F'|' '\$1 ~ /\.batch\$/'"
```

- Conda env on Picasso: `/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/isalsr`
- Logs: `~/execs/isalsr/c2_smoke/logs/`, job ids in `job_ids.txt`
- **SP-0 still binds**: nobody except Mario submits C2 itself.
