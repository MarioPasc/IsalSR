# Stage E runbook — the analysis dry-run (E1–E7)

**Stage E does not submit to Picasso.** It runs on the workstation. The
directory lives under `slurm/` only so the six pre-flight stages sit together;
there is no launcher and no worker, and nothing here consumes an allocation.

Why local, in one line: every Stage E failure needs an **analyzer code fix**,
and a deploy is a config edit that must never happen while an array is running
(`T17-HANDOFF.md` defect 10). Full reasoning: `docs/md_files/changes/stage_e_design.md` §1.

---

## 0. Preconditions

| # | Precondition | How to check |
|---|---|---|
| 1 | A Stage C root exists on Picasso with all 60 fields and 420 paired-stat files | `ssh picasso 'find <root> -name run_log.json \| wc -l'` → 1,260 |
| 2 | `pdflatex` is on `PATH` (E4 compiles every emitted table) | `which pdflatex latexmk` |
| 3 | ≥3 GB free where the work dir lives (one 572 MB working copy + two ~180 MB fixtures) | `df -h <work-dir>` |
| 4 | The `isalsr` env resolves | `~/.conda/envs/isalsr/bin/python -c "import isalsr"` |

Stage E reads the smoke root and never writes to it. Fixtures are copies.

---

## 1. Mirror the Stage C root

```bash
BASE=/media/mpascual/Sandisk2TB/research/isalsr/results/model_validation/real_benchmarks
rsync -az --info=stats2 \
  picasso:/mnt/home/users/tic_163_uma/mpascual/fscratch/results/isalsr/<smoke_root> \
  $BASE/
```

Verify before proceeding — a partial mirror makes E6 report failures that are
the transfer's, not the analyzer's:

```bash
find $BASE/<smoke_root> -name run_log.json         | wc -l   # expect 1260
find $BASE/<smoke_root> -name 'paired_stats*.json' | wc -l   # expect  420
```

---

## 2. Run the certifier

```bash
cd ~/research/code/IsalSR
~/.conda/envs/isalsr/bin/python -m experiments.scripts.stage_e_certify \
  --smoke-root $BASE/<smoke_root> \
  --work-dir   $BASE/c2_stage_e
```

Roughly 3 minutes end to end (49 s of it the analyzer). Useful flags:

| Flag | Effect |
|---|---|
| `--reuse-main` | Skip rebuilding the 572 MB working copy — use when iterating |
| `--only E3,E6` | Run a subset while fixing one check |
| `--benchmarks`, `--methods`, `--variants` | Override the defaults |

Exit 0 iff every requested check passes.

---

## 3. Read the verdict

```
$BASE/c2_stage_e/artefacts/stage_e_certification.md     # the table
$BASE/c2_stage_e/artefacts/stage_e_certification.json   # per-check evidence
$BASE/c2_stage_e/artefacts/logs/                        # every subprocess log
```

A check that raises is recorded as a **failure**, never skipped.

---

## 4. The one expected override, and when it must disappear

E1 runs with `--allow-mixed-provenance` because `c2_smoke_v4` pools two commits
(`a455d6c` ×1,099 and `a455d6c-dirty` ×161 — the mid-wave `sed`, §11.1
2026-08-04). The guard rediscovers this from the data alone.

**On the v5 root this override must not be needed.** After the clean Stage C
wave on `00635ae`, re-run Stage E and confirm the analyzer accepts the root
with **no** provenance flag:

```bash
~/.conda/envs/isalsr/bin/python -m experiments.models.analyze \
  --results-dir $BASE/<v5_root> \
  --methods udfs,bingo \
  --benchmarks nguyen,feynman,hard,cherrypicked,roundoff,feynman_remainder,strogatz \
  --variants baseline,hash,isalsr
# exit 0 required. Exit 2 means the wave was not clean and the
# campaign/c2 tag must not be cut (EXECUTION-PLAN §5.1).
```

---

## 5. What Stage E does *not* certify

- **No number here means anything.** Stage C ran at 3 seeds, where the minimum
  attainable two-sided Wilcoxon p is 0.25. Stage E certifies code paths, not
  results.
- **Cost.** The analyzer at 49 s / 1,260 runs projects to ≈5.5 min at 8,400.
  That is *not* the orchestrator's `--postprocess only` aggregation, which took
  1 h 35 m on the same input and projects to ≈11 h; size that job separately.
- **D2/D3.** The fixed-order hash numbers come from the Stage D trace and the
  Mode 1 replay, not from here.
