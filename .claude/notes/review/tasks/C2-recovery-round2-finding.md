# C2 recovery — round 2 will be needed (Bingo). Interim finding, 2026-08-13 12:00

## Status at time of writing

| Quantity | Value |
|---|---|
| Cells | 12,379 / 12,600 |
| Recovery tasks | 221 COMPLETED / 104 RUNNING |
| Recovery failures / deferrals / stranded copy-backs | 0 / 0 / 0 |

## UDFS will close; Bingo will not

All 101 running UDFS tasks are on **cell 3/3** (bundle B=3, `safe` mode). They
started their last cell ~09:11, each saturating the 12 h payload cap (trap G),
landing ~21:10 against a walltime ending ~23:10. ~2 h margin. **101 cells expected
in one burst around 21:10.**

The three running Bingo tasks hold 117 unwritten cells against ~20 h of walltime:

| Task | Cell | Remaining | Rate so far | Time needed |
|---|---|---|---|---|
| `c2r_bb_feynman_1960714_5` | 28/60 | 32 | 0.97 h/cell | ~31 h |
| `c2r_bh_feynman_1960719_5` | 19/60 | 41 | 1.30 h/cell | ~53 h |
| `c2r_bi_feynman_1960734_5` | 16/60 | 44 | 1.41 h/cell | ~62 h |

All three sit on **I.10.7**, and the sampled one is 5.6 h into that cell,
converged to fitness 8.6e-12 yet still running to the 12 h cap — so the tail is
slower than the historical average, not faster. The start-deadline stops a task
*starting* any cell whose full payload no longer fits, so these tasks stop
starting cells ~8 h from now. **Expect ~100 Bingo cells deferred**; the STEP 1
census will fail and a second recovery round is mandatory.

## Root cause: `--mode p90` on Bingo, and a length-biased tail

`submit_recovery.sh` (header, lines 30–32) offers two sizings:

- `safe` — charges every cell the 12 h **cap**; B=3 at a 38 h wall; deferral
  impossible by construction.
- `p90` — charges the measured p90; **B=63 on `bingo:feynman`**; ~4× fewer SLURM
  placements, "at the cost of a merely probabilistic guarantee".

UDFS got `safe` (B=3, holding). Bingo got `p90` (B=60/63, failing). Back-solving
`wall = (B-1)*allowance + CELL_RESERVE` gives an allowance of ~11 min/cell against
an observed 0.97–1.41 h/cell.

**Why the p90 estimate was the wrong statistic.** The start-deadline defers
exactly those cells that did not fit. The deferred set is therefore a
**length-biased sample** of the cell-duration distribution — conditioned on being
long. A p90 taken from the *unbiased* population understates it systematically.
The script's own header anticipates this ("any bundle sized on a central estimate
is a bet, and this pass is not the place to bet"); the bet was taken anyway for
Bingo only.

## RESOLVED — cancelled and resubmitted, 2026-08-13 ~12:30 (user-authorised)

The user authorised cancelling the three Bingo tasks and resubmitting with correct
parameters, which clears hard safety rule 4.

**Precondition checked first: copy-back is PER-CELL, not at task exit.** Proof:
`bingo/feynman/i.10.7` held exactly seeds 1–27 (`baseline`, task on cell 28),
1–18 (`hash`, cell 19), 1–15 (`isalsr`, cell 16). Every completed cell was already
on GPFS, so cancelling forfeited only the three in-flight cells. Had copy-back been
at task exit, cancelling would have destroyed ~60 finished cells.

Actions:

1. `scancel 1960714 1960719 1960734` — Bingo recovery only. UDFS arrays
   (1960414/1960514/1960614) deliberately left running; they are on cell 3/3 with
   ~2 h of margin.
2. Resubmitted `safe` mode: `--only 'bingo:baseline:feynman,bingo:hash:feynman,
   bingo:isalsr:feynman'`, B=3, 100 tasks/array, 32 G, 38 h wall.
   **New job ids: 1986376 (bb), 1986377 (bh), 1986385 (bi).**
   `recovery_job_ids.txt` is appended, not overwritten — it now holds all 10 ids.

**🔴 Trap C fired and was caught only by `--dry-run`.** The first attempt printed
`SKIP (already submitted today)` for all three arrays: the name dedup at
`submit_recovery.sh:245` matches `c2r_*` names over `sacct -S today` scoped by
`C2_MIN_JOBID` (default **0**), so the jobs just cancelled masked their own
replacements. Without the dry-run this would have been a **silent no-op**. Fix:
`C2_MIN_JOBID=1960800`. Always dry-run a resubmission that reuses a job name.

Two things verified after submission:

- **Resume-skip works; present cells are not rewritten.** 12 tasks completed within
  a minute, and `i.10.7` seeds 01/05/10 kept their original mtimes
  (2026-08-07/09/11). This is why planning the full 300-cell suite scope is safe:
  only the ~120 genuinely-missing cells actually run.
- Consequently the earlier **0.97–1.41 h/cell figure was an average over a mix of
  skipped and executed cells**, so the true cost of a missing cell is higher than
  stated above — which strengthens, not weakens, the case for cancelling.

## Original action plan (superseded by the above)

1. STEP 1 census will report a shortfall. That is expected, not a new defect.
2. Re-scope with `c2_missing_cells.py --selectors` and resubmit **in `safe` mode**
   (default; do **not** pass `--mode p90`). ~100 cells at B=3 → ~34 tasks, 38 h wall.
3. Do **not** pre-submit before drain: `submit_recovery.sh` plans from
   currently-missing cells, which would include cells the running tasks are about
   to write — risking two jobs writing the same cell directory.
   *(Superseded: cancelling first removed the race, since the cancelled tasks can
   no longer write.)*
4. Only then STEP 2 (copy) onward. Do not copy or aggregate a partial tree.

## Note on the health probe

`slurm/c2_campaign/health.sh` reads `job_ids.txt` and `logs/` (main pass) only —
never `recovery_job_ids.txt` or `logs_recovery/`. It therefore reports a false
`ALERT ... queue empty` while 104 recovery tasks run, and its
`deferred=992 cell_fail=1 bad=1` are historical main-pass numbers. More seriously
it would stay **silent through a recovery failure** — the exact blindness its own
header warns about. Unfixed; the loop's monitor covers the gate instead.
