# C2 handoff — Stage D is running; what is left

**Rewritten**: 2026-08-05 (supersedes the 2026-08-03 version entirely)
**Branch**: `feature/cpp-core-port`, pushed. Deployed commit: `00635ae`
**Campaign state**: pre-flight A–C complete and signed; **Stage D running**;
Stage E ready to start in parallel; C2 itself **not submitted** (SP-0: Mario only)

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

### 3.1 Stage E — startable now, in parallel with D
Input is `c2_smoke_v4/` (1,260 runs, 3 arms × 3 seeds, all 60 fields, 420 valid
paired-stat files). A8 is closed, so E1/E2/E4/E5 have real three-arm input.
**Read `audit.md` §6.1 and §7.3 before touching the analyzer.** E3 and E6 are
the adversarial checks (inject a NaN; delete a run) and are the point of the
stage. Nothing in Stage E writes to a campaign root.

### 3.2 When Stage D lands
- Read `stage_d_certification.{json,md}` (D1.1–D1.8).
- **D1.6 owns C5 §3.5**: is Bingo's ρ shortfall the budget gap, at 12 h vs 12 h?
- **D1.2 is what sizes production `--mem`**, from `max(sacct MaxRSS, VmHWM)` + 30 %.
  Stage C's 0.67 GB peak is a 900 s number and sizes nothing.
- **D1.7's overhead is now canon + conversion**, so expect it *above* the old
  canon-only ≈7.4 % projection. That is the correction, not a regression.
- Then **D3** (`stage_d_mode1_replay.py`) on the D2 stream.

### 3.3 Before the `campaign/c2` tag
| Item | State |
|---|---|
| **One clean Stage C wave on the final config** | 🔴 owed. v4 ran the pre-shadow-off configs and 161 of its cells are `-dirty`. ~35 min at `%24`/`sr` + aggregation |
| **HOME quota** | 🔴 0.34 / 0.28 TB, grace expiring. Mario's lane; deferred by his instruction. Campaign logs already redirected to FSCRATCH |
| **`campaign/c2` tag** | Procedure written (`slurm/c2_tag_procedure.md`), tag deliberately **not cut** — SP-0 |
| **Stage F** | Mario's sign-off |
| **≥15,000-file support mail** | still outstanding (`soporte@scbi.uma.es`) |

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
