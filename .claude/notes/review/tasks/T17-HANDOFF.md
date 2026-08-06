# C2 handoff — pre-flight A–E are GO; what is left

**Rewritten**: 2026-08-05 (supersedes the 2026-08-03 version entirely)
**Branch**: `feature/cpp-core-port`, pushed. Deployed commit: `b786d20`
**Campaign state**: pre-flight **A–E complete** — A–C signed, **Stage D GO**
(13/13 cells, 8/8 criteria), **Stage E GO** (7/7 checks, 2026-08-05) and
**D3 GO** (2026-08-06); Stage F approved by Mario 2026-08-06;
C2 itself **not submitted** (SP-0: Mario only)

---

## 2026-08-06 — what changed today, read this first

**Stage F was approved, and two blocking defects were found on the way to it.**
Both were found by checks that exist precisely to find them, and both are fixed.

1. **D3 found the k=0 completeness defect** (`b223a08`). Every zero-internal-node
   DAG canonicalises to `""`, so `f(x)=X_0` and `f(x)=X_1` shared a dedup key;
   Bingo transferred the cached fitness between them and UDFS skipped the second
   outright. **Pre-existing — C1 carries it too.** Fixed as a *domain
   restriction*: at k=0 the relabeling group is 0!=1, so there is nothing to
   collapse and ρ is undefined. k=0 candidates are scored, never deduplicated,
   excluded from ρ; `n_nonstructural` records how many. **Completeness is now
   claimed for k≥1** — that sentence must reach the paper.
2. **The v5 ledger job died in 2 s** (`b786d20`). `aggregate_worker.sh` asserted
   `C2_CONFIG_LIST` at file scope; the ledger job is not given it. A
   1,260/1,260 clean wave produced no ledger and no verdict.
3. **The k=0 guard's own ρ accounting was wrong** (`040beab`), and **v5b passed
   19/19 with ρ still inflated** — which is the lesson worth keeping: a green
   certification is not proof that a *new* quantity is right, because no
   criterion was watching it yet. `n_total` counts a candidate when it is first
   seen, before the DAG exists to classify, so k=0 was excluded from `n_unique`
   but not from `n_total`. Measured over the whole wave, **bingo/hash +12.15 %,
   bingo/isalsr +13.86 %** — in the direction that flatters us. UDFS unaffected.
   **The 0.0593 % incidence quoted on 2026-08-06 is superseded: it is 12.2 %
   campaign-wide and 18.2 % on Nguyen-3.** The trace it came from is a 1-in-100
   subsample of a single hard problem.

**Sequence executed today**: deploy `115bf89` → Stage C **v5** (1,260/1,260,
single provenance, 18/19 — C1.2 fails only because v5 predates
`n_nonstructural`) → k=0 and ledger fixes → archive `c2_smoke_v4` and
`c2_smoke_v5` → deploy `b786d20` → **v5b** (1,260/1,260, **19/19 GO**, 0 SKIPs)
→ ρ-accounting defect found *in* v5b's own data → archive `c2_smoke_v5b` →
deploy `040beab` → Stage C **v5c**, the wave the tag will rest on.

Every archive was tarred, its entry count verified against the source (7,932
each), and only then removed.

**New**: `slurm/c2_campaign/` — a Stage F gate (10 checks, fails closed) and a
launch wrapper delegating to the certified `c2_smoke/launcher.sh` under
`C2_PROFILE=campaign`. **No worker or topology is copied**; what the smoke wave
certifies is what launches.

---

## 0. Read these first, in this order

| # | File | Why |
|---|---|---|
| 1 | `EXECUTION-PLAN.md` **§11.1** | The decision/anomaly ledger. Denser and more current than this file |
| 2 | `audit.md` | The fairness audit: 16 findings, what was fixed, and **§6–§7 the locked decisions** (contrast policy, shadow off, Stage D config). Everything downstream depends on §6.1 and §7.3 |
| 3 | `EXECUTION-PLAN.md` §4.4 / §4.5 / §6.1 / §6.1a | Stage D as it now runs, Stage E, and the corrected statistics |
| 4 | `c2_preflight/smoke_vs_C1.md` | C5, signed. §3.5 is the one live scientific question |
| 5 | `T18-canonical-completeness-operand-order.md` | The completeness defect. Unchanged by any of this |

---

## 1. What is running right now

**Stage D, 13 cells, submitted 2026-08-05 on `00635ae`.**

| Job | Group | Cells | Mem | State at submission |
|---|---|---|---|---|
| **1769422** | `udfs` | 3 | 16 GB | RUNNING |
| **1769423** | `bingo_std` | 6 | 32 GB | RUNNING |
| **1769424** | `bingo_isalsr` | 4 | 256 GB | RUNNING |
| **1769425** | `c2d_certify` | 1 | 16 GB | PENDING (`afterany`, dependency verified non-null) |

13/13 started within 20 s, including all three 256 GB cells. Payload 12 h under
a 16 h wall, `--constraint=sr`, seed 101 (trace cell 102).

```bash
ssh picasso 'sacct -j 1769422,1769423,1769424,1769425 -X -n -P -o JobID,State'
# memory: NEVER -X (blank MaxRSS); filter the .batch step, and emit JobIDRaw
ssh picasso "sacct -j <ID> -n -P -o JobIDRaw,MaxRSS | awk -F'|' '\$1 ~ /\.batch\$/'"
```

Logs: `fscratch/execs_logs/isalsr/c2_stage_d/logs` — **on FSCRATCH, not HOME**,
because HOME is over quota with grace expiring. Results:
`fscratch/results/isalsr/c2_stage_d/`.

---

## 2. The decisions that now govern everything (do not re-litigate)

1. **Shadow sketches are OFF for C2, both hosts** (`audit.md` §7.3). They cost
   **17.6 %** of Bingo's wall clock — more than the method overhead they
   instrument — paid by the `isalsr` arm alone inside a fixed budget. The
   fixed-order/steel-man numbers come from **D2 + D3 only**.
2. **`shadow_hash` lives in the METHOD block** (`bingo:` / `udfs:`), never
   under `isalsr:`. `create_runner` passes `config.get(method, {})`; a key in
   the wrong block is silently ignored. Locked by
   `tests/unit/test_shadow_hash_config.py`.
3. **ρ is descriptive against the baseline, inferential only against `hash`**
   (§6.1). The baseline's ρ is 1.0 *by construction*, so the submitted table's
   ρ p-values are withdrawn.
4. **CPDT ties are snapped and split** (`zsplit`, §6.1a). Any C1 p-value quoted
   in the revision must be the **corrected** one.
5. **Main tables print `p_value_holm`**; raw one-sided goes to the supplement.
6. **The second, adapter-order hash number is NOT reported** — see §11.1
   (2026-08-04). One naive baseline, defended mechanistically.

---

## 3. Open work, in priority order

### 3.1 ✅ Stage E LANDED 2026-08-05 — `GO`, 7/7 checks, 182 s
Ran **locally** on `c2_smoke_v4/` (1,260 runs, 3 arms × 3 seeds). Harness
`experiments/scripts/stage_e_certify.py`; write-up
`docs/md_files/changes/stage_e_design.md`; runbook `slurm/c2_stage_e/RUNBOOK.md`.

**Four defects found, three of which every generator exited 0 on:**

| # | Defect | Fix |
|---|---|---|
| **E4** | The D2 extension **broke LaTeX compilation** — identifiers typeset raw, and every T05 D2 name carries an underscore (`strogatz_vdp1`, `liv_19`, `pagie_2`, `feynman_remainder`). **18 rows per table × 4 tables**: exactly the rows the coverage extension added | `_latex_escape` at three emission sites |
| **E5** | The **hash arm vanished from every CD diagram** — `generate_critical_difference.py` iterated a hardcoded `["baseline","isalsr"]`, giving 4 groups where 2 methods × 3 arms gives 6. `cross_method.py` had been extended for three arms; the figure generator had not | `--variants` threaded through both loaders, all four generators and both CLIs |
| **E6/E7** | **Neither check existed.** `reconcile()` sat unused in `status_ledger.py`; there was no provenance check at all | `analyzer/completeness.py`, failing closed with exit 2 |

**Two carries.** `git_commit` is `None` on all 1,260 runs, so a guard keyed on it
would pass **vacuously on every campaign** (the SP-6 trap) — the guard keys on
`git_describe`/`git_dirty`/`build_hash` and reports absent keys as
*non-informative*, never as agreement. And the guard **independently rediscovered
v4's dirty split**, so the owed clean Stage C wave is now enforced by code.

🔴 **Owed: re-run Stage E on v5.** It passed on v4 only with
`--allow-mixed-provenance`. On v5 it must pass **without** it, or the
`campaign/c2` tag must not be cut.

### 3.2 ✅ Stage D LANDED 2026-08-05 — `GO`, 13/13 cells, 8/8 criteria

Full numbers: `T17-appendix/capacity_optimisation_worklog.md` **§14**; ledger row
in EXECUTION-PLAN §11.1, 2026-08-05.

- **D1.4 is the one that mattered and it passes.** Korns-12 and Vlad-2 on
  Bingo-isalsr both return finite R² (−0.0217, 0.9940). The C1 NaN does not
  recur.
- **D1.6 answered C5 §3.5.** ρ ratios C2/C1 = 0.9973 / 0.9977 / 1.0013, UDFS rose
  to 1.0656, zero excursions. **The Stage C shortfall was the budget gap**, not a
  canonicaliser regression. Reconstruction cross-checked to `abs_gap` ≤ 0.0005.
- **D1.2 sized production memory.** Peak `bingo/isalsr` **1.193 GB**; D1.2's own
  rule recommends **8 GB**; shipped **32 GB** (4× that, 27× the peak). §3.3's
  256 GB is superseded — see §11.1 and `c2_slot_plan.MEM_GB`.
- **D1.7** Bingo overhead **7.83 %** of eval, UDFS 0.027 % — above the old ≈7.4 %
  canon-only projection, which is the accounting correction, not a regression.
- 🔴 **Trap for the next Stage D:** the Picasso certifier reported `GO` /
  `n_blocking_failures: 0` while **D1.6 sat at `SKIP`** — its `--c1-reference` is
  a workstation path no compute node can reach. Ship the C1 analysis dir to
  FSCRATCH, or run `stage_d_certify.py` locally. It was re-run locally to get the
  real verdict.
- ⚠ **Korns-12 hash R²_test = −4.015** (baseline −0.014, isalsr −0.022). Finite,
  so D1.4 is unaffected, but at 30 seeds it would dominate that problem's hash
  mean. §6.4 covers NaN, not finite-but-wild. Watch it.
- Still open: **D3** (`stage_d_mode1_replay.py`) on the D2 trace stream.

### 3.3 The remaining sequence — for the agent picking this up

**Nothing pushed on 2026-08-05 has executed a single task on Picasso.** The
throttle apportionment, 32 GB, merged logs, FSCRATCH log paths, the aggregation
array split, the vectorised bootstrap and `n_seeds: 30` are all uncertified. That
is what v5 is for, and it is the gate in front of everything else.

| # | Step | Notes |
|---|---|---|
| 1 | **Deploy `2838a12`** | `slurm/c2_smoke/deploy.sh`. Refuses a dirty tree — the tree is clean and pushed. **Stage D is finished, so defect 10 no longer blocks.** Verify HEAD + cleanliness *from the remote side*, which deploy.sh does |
| 2 | **Stage C v5** | `bash slurm/c2_smoke/launcher.sh` (profile `smoke`, the default). 1,260 tasks, 1,008 apportioned slots, ~35 min, ≈315 core-h, writes to `c2_smoke/`. **Run `--test-only` first** — there is no `sbatch` on the workstation (defect 15) |
| 3 | **v5 must pass without `--allow-mixed-provenance`** | That is the whole point: v4 had 161 `-dirty` cells. If it needs the flag, the deploy was wrong |
| 4 | **Archive `c2_smoke_v4`** | `tar czf c2_smoke_v4.tar.gz c2_smoke_v4` → verify entry count → *then* `rm -rf`. Frees ~7,930 inodes. v3 already archived this way. **The inode guard will refuse the campaign until this is done** |
| 5 | **Re-sign A5 at 1…30** | §11.2's A5 row is 🔴 **REOPENED** — its PASS certified 1…20. Assert `0 ∉ seeds`, 0/101/102 disjoint, `n_seeds == 30` on 14/14 configs |
| 6 | **`campaign/c2` tag** | **Mario's.** Procedure in `slurm/c2_tag_procedure.md`, deliberately not cut |
| 7 | **Stage F → submit C2** | **Mario's, SP-0.** `C2_PROFILE=campaign`, 12,600 tasks, ≈108,000 core-h |

**Do not do 6 or 7.** Stop after 5 and report.

| Other item | State |
|---|---|
| **HOME quota** | ✅ resolved — 23.38 GB / 0.28 TB, and campaign logs now default to FSCRATCH *in the launcher*, not by override |
| **FSCRATCH inodes** | 158.5k / 250.0k after archiving v3. 30 seeds needs 84,381 against 91,600 free — fits, but v5 (+7.9k) then v4's archive (−7.9k) is the sequence. Guard enforces |
| **≥15,000-file support mail** | still outstanding (`soporte@scbi.uma.es`) |
| **Gate 3 on Picasso** | ⚠ re-measured **locally** on `3d5a79c` (0/10,000, both engines). The Picasso re-run at the same provenance is owed — cheap, fold it into v5 |

---

## 4. Defects found the hard way — do not reintroduce

Additions to the 2026-08-03 list (1–9 there still stand: `--export` commas,
per-task post-processing races, live-counter JSON paths, `module load` in a
pipe, …):

10. **A deploy IS a config edit.** Never deploy while an array is running: a
    mid-wave redeploy splits provenance across two HEADs, and an in-place edit
    of the deployed tree marks every subsequent cell `-dirty` (161 of v4's
    1,260).
11. **`GROUPS` is a reserved bash array.** Assigning to it returns an error
    status; under `set -e` the launcher died before its first `echo`, exiting 1
    with no output at all.
12. **`~/.conda` does not exist on Picasso** (the env is under `fscratch/conda_envs`).
    Resolve python the way `c2_smoke/launcher.sh` does.
13. **`git pull` cannot work on Picasso** — no outbound SSH to GitHub. `deploy.sh`
    (rsync **including `.git`**, remote-side SP-1/SP-2) is the only path.
14. **`deploy.sh` refuses a dirty tree, and it is right.** If the tree carries
    another agent's work, deploy from a temporary clean clone of the branch tip.
    Never relax SP-1 and never commit someone else's files.
15. **Run `--test-only` on the cluster for every new launcher.** There is no
    `sbatch` on the workstation, so launcher bugs are unreachable locally.
16. **Worktree pytest silently tests the MAIN repo.** The editable install maps
    `experiments`/`benchmarks` to the main checkout; use the repointing shim, or
    test from the main checkout.

---

## 5. Still Mario's call, not an agent's

1. **T18** — fix the encoding or state completeness as conditional. If fixed,
   **every ρ must be recomputed**, so the sequencing decision comes first.
2. **HOME quota** (§3.3).
3. **Stage F sign-off and the `campaign/c2` tag.** SP-0 is unchanged: **nobody
   except Mario submits C2.**
4. Four confirmations parked in `audit.md` §6.5 (conservative-substitution
   levels, ρ's Holm family of 1, raw-vs-Holm in Table 1/S, the worker's
   ≥2-seed guard blocking SP-0 probes).

**Accepted, needing no further work** (Mario, 2026-08-03): Bingo is not
seed-reproducible — three identical `baseline --seeds 0` runs gave
155,449 / 41,023 / 41,049 candidates. C3 is therefore bounded rather than exact
for Bingo (wrapper perturbation 3 inner evals < baseline self-noise 12); **UDFS
passes C3 outright**. To be stated in the paper, not fixed.
