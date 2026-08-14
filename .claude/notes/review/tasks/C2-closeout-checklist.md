# C2 close-out checklist (user-instructed, 2026-08-14)

> ## ✅ CLOSED 2026-08-14 — all steps complete
>
> - **Census**: 12,600 expected / 12,600 present / **0 missing** (`--strict`).
> - **Aggregation**: 14/14 configs postprocessed → 420 `aggregate.csv`,
>   420 `paired_stats*.json`, `status_ledger.csv` (12,600 rows).
> - **Certification**: **GO**, 0 blocking failures, 19/19 criteria
>   (job 1993088, after fixing C1.16 — see §13.3 of the recovery plan).
> - **Execution health**: `completed`/`exit_code 0`/`native` on 12,600/12,600;
>   0 NaN-metric cells, 0 exceptions; grid balanced at 2,100 per (method, arm);
>   nothing time-killed (max wall 43,223 s vs the 12 h cap + teardown).
> - **Backup**: identical on both sides — 65,944 files, 38,338,981,466 B;
>   manifest digest `17d6f23e…`, run-log content digest `0594504a…`,
>   stats digest `a46658ab…`. `convergence_log.npz` is path+size only.
> - **README**: written at the results root of the backup.
> - **Docs**: EXECUTION-PLAN §11.1/§11.3 and the recovery plan §13 updated.

Original plan below, retained for the next recovery.

Gate: last cell was `bingo/feynman/i.10.7/isalsr/seed_18` (task `1986385_86`).

## 0. Gate — STEP 1 census (do not skip)

- [ ] `ssh picasso 'D=$FSCRATCH/repos/IsalSR_recovery; PY=$FSCRATCH/conda_envs/isalsr/bin/python; cd $D && PYTHONPATH=$D/src:$D $PY $D/experiments/scripts/c2_missing_cells.py --results-dir $FSCRATCH/results/isalsr/c2_3arm --seeds 1-30 --strict --summary'`
- [ ] MUST report **12600 expected / 12600 present / 0 missing**.
- [ ] If gaps remain: re-scope with `--selectors` and resubmit **`safe` mode**
      (`C2_MIN_JOBID` above the newest job id — trap C) before anything else.
      Do **not** copy or aggregate a partial tree.

## 1. Aggregation + ledger over the WHOLE corpus (all 12,600 cells)

User instruction: statistics must run over **every cell**, not only the recovered
ones.

- [ ] Submit `c2r_aggregate` (42-task array, one per config) then `c2r_ledger`
      (`--dependency=afterany`), with `ISALSR_REPO_DIR` = the **DEPLOYED** tree
      (`$FSCRATCH/repos/IsalSR`). Per §10 of `C2-deferred-cells-recovery-plan.md`;
      `submit_recovery.sh --with-aggregation` chains both.
- [ ] 🔴 **Pass `C2_EXPECTED_TASKS=12600` explicitly** (trap A). It is a CELL
      count. The deployed `launcher.sh:625` passes the TASK count; omitting it
      makes `c2_certify` fall back to a self-referential "disk" universe and
      report GO on an unverified tree.
- [ ] Expect: certifier **GO**, `n_blocking_failures=0`, C1.15
      `expected_set_source="registry"`, expected == observed == 12600.

## 2. Copy back to the local backup (merge, then VERIFY)

Destination (must end in `/c2_3arm`):
`/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/review/c2_3arm`

- [ ] `source ~/.bash_aliases`
- [ ] `parallel-ssh-copy picasso:$FSCRATCH/results/isalsr/c2_3arm <dst> 8`
      Copies CONTENTS into dst; `--ignore-existing` so only new cells transfer.
- [ ] Re-run after aggregation so the root-level artifacts come too:
      `aggregate.csv`, paired stats, `status_ledger.csv`, `c2_preflight/`.
- [ ] **VERIFY BOTH SIDES — do not trust exit 0**: file count, `run_log.json`
      count, and **sum of regular-file bytes**
      (`find -type f -printf '%s\n' | awk '{s+=$1} END {print s}'`).
      The byte check is the one that matters: `--partial --inplace
      --ignore-existing` would silently keep and then skip a truncated file.
      Compare file bytes, **not** `du -sb` (GPFS vs ext4 inode sizes are noise).

## 3. Results-organisation README (user-requested)

- [ ] Write a markdown file at the **results-level folder**
      (`<dst>/README.md`) explaining the layout:
      `<method>/<suite>/<problem_slug>/<arm>/seed_NN/run_log.json`, the three
      arms (baseline / hash / isalsr differ ONLY in deduplication), the 70
      problems x 30 seeds x 2 methods x 3 arms = 12,600 cells, the root-level
      analysis artifacts, and `complexity.json` (T19 telemetry).
- [ ] Carry the standing caveats into it: R2 outlier contamination (use
      median/clipped/robust statistics — trap D), `cache_hit_rate` is a dead
      all-zero field (trap E), Bingo `total_dags_explored` is not comparable
      baseline-vs-dedup (trap F).

## 4. Docs

- [ ] `EXECUTION-PLAN.md` §11.1 (anomaly ledger) and §11.3 (launch ledger) with
      final numbers and the new job ids **1986376 / 1986377 / 1986385**.
- [ ] `C2-deferred-cells-recovery-plan.md` — final outcome.
- [ ] Fold in `C2-recovery-round2-finding.md` (the `p90` length-bias defect, the
      cancel/resubmit, trap C).

## 5. Commit and push

- [ ] Commit under the user's name (`MarioPasc`, mario.pg02@gmail.com — the repo
      default; do **not** add Co-Authored-By, disabled in settings).
- [ ] Conventional-commit subject, <=72 chars, body explaining *why*.
- [ ] Branch `feature/cpp-core-port`. Push.

## Standing safety rules

1. NEVER modify `$FSCRATCH/repos/IsalSR` (deployed tree) and never run
   `deploy.sh` while jobs run. All cells must report `git_describe = campaign/c2`.
2. NEVER hand-write into the campaign root. Only jobs write there.
3. The local copy is the ONLY backup of ~80,000 core-hours. Never delete it.
4. Do not `scancel` without asking the user.
5. fscratch INODES are the constraint (~225k/250k). Do NOT delete `build_gedlib/`.
