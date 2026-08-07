#!/usr/bin/env bash
# =============================================================================
# Campaign C2 pre-flight -- Stage C launcher (EXECUTION-PLAN.md §4.3, T17)
# =============================================================================
# The 15-minute full-coverage smoke: every problem x every arm x every method,
# once, on real Picasso hardware, producing every artefact the analysis will
# later consume.
#
#   {baseline, hash, isalsr} x {udfs, bingo} x 70 problems x 3 seeds = 1,260
#
# Topology (T17 §2.2, Option A -- decided 2026-08-03).  One array per
# (method, arm, suite) = 2 x 3 x 7 = 42 arrays.  Chosen over two merged configs
# because it changes no configuration content and therefore cannot perturb the
# A4b operator-set invariant, and because smaller arrays fail more cheaply.
# Largest array is strogatz, 14 problems x 3 seeds = 42 tasks -- far inside
# MaxArraySize (4096) and the 1,000-task courtesy threshold.
#
# Usage:
#   bash slurm/c2_smoke/launcher.sh --dry-run          # print, do not submit
#   bash slurm/c2_smoke/launcher.sh --one-task         # 2 tasks: udfs+bingo isalsr, 1 problem
#   bash slurm/c2_smoke/launcher.sh --test-only        # sbatch --test-only on all 42 (B7)
#   bash slurm/c2_smoke/launcher.sh                    # the full 1,260-task wave
#   bash slurm/c2_smoke/launcher.sh --only udfs:isalsr:nguyen
#   C2_PROFILE=campaign bash slurm/c2_smoke/launcher.sh --dry-run   # C2 itself
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Profile ---------------------------------------------------------------
# One launcher, two profiles.  Stage C exists to certify the code path C2 will
# use, and "certifying a topology you will not launch certifies nothing" (§1) --
# so the smoke and the campaign differ ONLY in seeds, payload budget, wall and
# output root.  Everything structural (topology, slot apportionment, memory,
# aggregation shape) is shared, and the owed clean Stage C wave therefore
# certifies the campaign's launcher and not a cousin of it.
#
# Selecting a profile sets a coherent GROUP of defaults.  That is deliberate:
# eight independent env vars that each silently default to the smoke value is
# the ISALSR_LEDGER_ENABLED / shadow_hash trap a third time.
C2_PROFILE="${C2_PROFILE:-smoke}"

case "${C2_PROFILE}" in
  smoke)
    DEF_SEEDS="0,101,102"      # T17 §0.1: outside campaign 1..20 AND the 21..30 top-up range
    DEF_MAX_TIME=900           # payload budget, seconds.  Never the production 43,200
    DEF_WALL="0-00:40:00"      # >2x the payload, so a SLURM kill means a real defect (C1.12)
    DEF_ROOT_NAME="c2_smoke"
    DEF_SLOT_BUDGET=1008       # same total as the certified %24 wave, apportioned
    DEF_AGG_WALL="0-01:59:00"
    # The smoke's wall is B x (payload + teardown), which is exact because a
    # smoke cell cannot exceed its 900 s budget.  At the campaign's bundles (up
    # to 164) that wall would be 68 h; capped at 4 it is 1 h 50 m, i.e. still
    # inside `short` (MaxWall 2 h, priority 118,933 against medium_uma's 28,873).
    # The smoke still exercises the chunk loop, the deadline and the staging --
    # it just does not need a 164-cell chunk to do so.
    DEF_MAX_BUNDLE=4
    ;;
  campaign)
    # 30 seeds, not 20 (Mario, 2026-08-05).  §0.4a fixed the campaign at 20 on a
    # cost premise that the measured runtimes reversed: UDFS 12.00 h and Bingo
    # ~5.15 h put 30 seeds at ~107,100 core-hours, and the apportioned throttle
    # puts that at a 54 h makespan.  30 seeds restores the seed count C1 used, so
    # §6.3's disclosure paragraph about reduced supplementary-table power goes
    # away entirely.
    #
    # ⚠ `--seeds` is always passed explicitly, but `orchestrator.py:641` falls
    # back to the config's own `n_seeds` when it is not -- so all 14 configs were
    # moved to `n_seeds: 30` at the same time.  Leaving them at 20 would have
    # been the ISALSR_LEDGER_ENABLED trap once more: a default that is silently
    # wrong on any path that does not override it.
    DEF_SEEDS="1-30"
    DEF_MAX_TIME=43200         # §5.4: 12 h.  8 h was REJECTED on measured effect erosion
    DEF_WALL=""                # per-method, from the plan (16 h; see c2_slot_plan.WALL)
    DEF_ROOT_NAME="c2_3arm"    # §1: the one and only campaign root
    DEF_SLOT_BUDGET=2016
    DEF_AGG_WALL="0-01:59:00"
    DEF_MAX_BUNDLE=0           # no ceiling: c2_slot_plan's own bounds govern
    ;;
  *)
    echo "FATAL: C2_PROFILE must be 'smoke' or 'campaign', got '${C2_PROFILE}'" >&2
    exit 1 ;;
esac

# ---- Configuration ---------------------------------------------------------
export ISALSR_REPO_DIR="${ISALSR_REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR}"
FSCRATCH="${ISALSR_FSCRATCH:-/mnt/home/users/tic_163_uma/mpascual/fscratch}"
RESULTS_ROOT="${C2_RESULTS_DIR:-${FSCRATCH}/results/isalsr/${DEF_ROOT_NAME}}"

# 🔴 Logs go to FSCRATCH, not HOME.  Measured 2026-08-05: HOME's file quota is
# 35.0k soft with 13.5k used, and ~/execs alone holds 10.7k of that.  C2 writes
# two log files per task -- 16,800 at 20 seeds, 25,200 at 30 -- so the 30-seed
# campaign EXCEEDS the HOME soft quota outright and the 20-seed one lands at
# 86 % of it.  FSCRATCH has 83.6k inodes of headroom.  T17-HANDOFF §3.3 claimed
# the logs were "already redirected to FSCRATCH"; they were not, in the code or
# on disk -- the redirection existed only as an env-var override nobody applied,
# and Stage D duly wrote to HOME.  It is a default now, not a habit.
LOGS_DIR="${C2_LOGS_DIR:-${FSCRATCH}/execs/isalsr/${DEF_ROOT_NAME}/logs}"
ACCOUNT="${C2_ACCOUNT:-tic_163_uma}"

# One log file per task, not two.  SLURM folds stderr into stdout when --error is
# omitted, halving the campaign's log inodes: 25,200 -> 12,600 at 30 seeds.
#
# This is not cosmetic -- it is load-bearing for the 30-seed campaign.  Measured
# 2026-08-05, with FSCRATCH at 91.5k inodes of headroom after archiving
# c2_smoke_v3: 30 seeds needs 71,781 result inodes, so split logs come to 96,981
# and do NOT fit even with BOTH superseded smoke roots archived; merged logs come
# to 84,381 and fit with ~7k to spare.
#
# Nothing is lost.  No code reads the .err stream: c2_certify reads run_log.json,
# status.json and sacct, never the SLURM logs (grepped).  The worker sets
# PYTHONUNBUFFERED=1 and bash's echo is unbuffered, so the two streams interleave
# line-cleanly rather than tearing.  Diagnosis by grep is unaffected -- the
# `.err` content the ledger cites (e.g. Bingo's "maximum number of fitness
# evaluations ... exceeded") simply lands in the .out file.
#
# Applied to BOTH profiles on purpose: the smoke must certify the log layout the
# campaign uses, not a cousin of it.
MERGE_LOGS="${C2_MERGE_LOGS:-1}"

SEEDS="${C2_SEEDS_SPEC:-${DEF_SEEDS}}"
# 🔴 sbatch --export is COMMA-separated, so a comma inside a VALUE is parsed as
# the start of the next variable. Exporting C2_SEEDS=0,101,102 delivered
# C2_SEEDS=0 to the worker -- one seed instead of three -- so every array had
# n_problems valid indices instead of n_problems*3 and every task above that
# died with "index N out of range". Worse, the tasks that DID run produced
# correct-looking seed-0 cells, so the array was 1/3 right and 2/3 failed rather
# than failing outright. Ship the list colon-separated; the worker translates.
SEEDS_EXPORT="${SEEDS//,/:}"
MAX_TIME="${C2_MAX_TIME:-${DEF_MAX_TIME}}"
WALL_OVERRIDE="${C2_WALL:-${DEF_WALL}}"
# Per-cell teardown allowance used by the smoke's no-spill wall.  Generous on
# purpose: §11.1 (2026-08-03) records a cell that finished its search correctly
# and then spent 7+ further minutes in SymPy.
SMOKE_TEARDOWN_S="${C2_TEARDOWN_S:-600}"

# ---- Slot budget -----------------------------------------------------------
# The throttle is now apportioned per array in proportion to that array's WORK
# (tasks x per-run hours) by experiments/scripts/c2_slot_plan.py, instead of one
# uniform %K for all 42.
#
# 🔴 The uniform %K was the dominant schedule loss, and it was invisible in the
# Stage C measurements.  The 42 arrays carry a 5.4x work spread
# (udfs:*:strogatz 280 x 12.0 h against bingo:*:feynman_remainder 120 x ~5 h),
# so under a uniform %K the small arrays drain early and hand their slots back
# to nobody -- the survivors are still capped at %K.  At 20 seeds and 1,008
# total slots the uniform split finishes in 140.0 h and the apportioned one in
# 72.7 h: 1.92x, for the same slots and no configuration change at all.  Stage C
# could not show this because it has 1.25 tasks per slot and is ramp-up and
# drain end to end; C2 has 8.3.  See T17-appendix/capacity_optimisation_worklog.md §2.
SLOT_BUDGET="${C2_SLOT_BUDGET:-${DEF_SLOT_BUDGET}}"
# Escape hatch: C2_UNIFORM_THROTTLE=24 restores the pre-2026-08-05 behaviour for
# A/B work.  It is strictly worse; it exists so the claim above stays falsifiable.
UNIFORM_THROTTLE="${C2_UNIFORM_THROTTLE:-}"

# Aggregation wall.  2 h was NOT enough and cost a verdict: job 1753134 spent
# ~1h40 on the 14-config `--postprocess only` loop over 1,260 runs and was
# killed at its wall BEFORE the certifier emitted anything.  The artefacts
# persisted, so `certify.sh` recovered it -- but the stage read as failed for a
# day.
#
# That job is now a 14-TASK ARRAY plus one dependent ledger job, so the wall
# sizes ONE config, not fourteen.  Measured 2026-08-05: the cost is ~4.2 s per
# problem and is set by (problem x contrast x metric), a count that is IDENTICAL
# at Stage C and C2 -- the earlier "x6.7 in runs => ~11 h" extrapolation does not
# hold, because going from 3 seeds to 20 costs 2.6 %, not 570 %.  Real I/O over
# the whole 1,260-run root is 0.6 s; 95 % of the wall WAS the 10,000-iteration
# Python bootstrap loop in cohens_d_ci_bootstrap, now vectorised bit-identically
# (tests/unit/test_effect_sizes_bootstrap.py).  All fourteen configs together
# take 19 s locally against ~590 s before, and the 841 artefacts they write are
# byte-identical to the old path's.
#
# 1 h 59 m, not 6 h, and the two minutes matter: `short` has MaxWall = 2 h and
# QOS priority 10000 against medium_uma's 1000, which under
# PriorityWeightQOS = 100000 is 118,933 against 28,873.  Measured with
# `sbatch --test-only` 2026-08-05: a 1:59 request started IMMEDIATELY, a 2:01
# request was estimated three hours out, and 13 h / 16 h / 23 h / 7 d all got
# that same three-hours-out estimate.  This is the ONLY place in the campaign
# where trimming the wall buys anything at all.
#
# Margin: one array task is one config -- 1.4 s at Stage C's 3 seeds, and the
# 2026-08-03 incident that motivated the 6 h wall was the OLD serial loop taking
# 1 h 40 m for all fourteen.  That work is now 28x smaller and parallel.  A wall
# kill stays recoverable: the artefacts persist and certify.sh reads them off
# disk, which is exactly how job 1753134 was salvaged.
AGG_WALL="${C2_AGG_WALL:-${DEF_AGG_WALL}}"

# Node family: NOT pinned.  The engine is x86-64-v3 / avx512f=0, hence portable
# across sd/sr/bc/bl (B6b), and Stage C produces no number that enters a table,
# so the wall-clock-homogeneity argument for pinning does not apply here.  Every
# run records its own cpu_model (A7), so B5/B6 get the node census as a
# by-product and the arm balance is reportable.
# 🔴 PINNED to the AMD `sr` family (2026-08-04). Was `cpu` (any node), which is
# what exposed the C4 failure: data generation is NOT bit-reproducible across
# CPU families, so a cell on an Intel `sd` node and its paired cell on an AMD
# `sr` node produced data differing by ~1 ULP (5.55e-17) and therefore different
# `data_fingerprint`s. 35 of 210 (problem, seed) pairs split, and all 35
# partitioned EXACTLY by node family. Pinning removes it at the source.
#
# Two independent reasons, either of which would suffice:
#   1. C4 -- the paired arms must provably see identical data.
#   2. Wall clock is a REPORTED quantity (S, the overhead percentages, Table 2's
#      cost column). Intel `sd` at 2.1 GHz against AMD `sr` at 2.6 GHz turns any
#      timing comparison across a mixed pool into a measurement of the
#      scheduler. This is also the `picasso-sbatch` skill's standing rule and it
#      closes the open B6 decision.
#
# Why `sr` and not another family (measured 2026-08-04):
#   sr  154 nodes x 128 c = 19,712 cores, 450 GB/node, AMD, no avx512  <- chosen
#   sd  123 x  52 =  6,396 cores, 187 GB, Intel, avx512
#   bc   32 x 256 =  8,192 cores, 700 GB, AMD, avx512
#   bl   24 x 128 =  3,072 cores, 1855 GB, AMD, no avx512
# `sr` is the largest pool by a factor of 2.4 and is itself more than double the
# QOS entitlement (cpu=9000), so the pin does not bind the entitlement. And the
# engine is built `isa_level=x86-64-v3, avx512f=0`, i.e. exactly the ISA `sr`
# provides -- B6b already certified it imports and runs there.
#
# ✅ At §3.3's original 256 GB, `sr`'s 450 GB/node hosted exactly ONE
# Bingo-IsalSR task per node -- a hard ceiling of 154 concurrent tasks out of
# 1,400, sterilising ~115 of each node's 128 cores, and separately capped at 190
# by the QOS's own `mem=50000000M` per-user limit.  That was the campaign's real
# capacity ceiling and it is gone at 32 GB (14 per node, no QOS bind).
#
# ACCEPTED COST (Mario, 2026-08-04): a pinned pool queues slower than `cpu`.
# That is the intended trade -- comparability over turnaround.
CONSTRAINT="${C2_CONSTRAINT:-sr}"

# ---- Local scratch ---------------------------------------------------------
# On by default (SCBI mail, 2026-08-07; manual §4.9).  The worker writes the
# payload's whole output tree to the compute node's own disk and copies back
# only the finished artefacts of each cell.  `sr` nodes carry 800 GB of
# localscratch against a campaign footprint measured in megabytes.
#
# C2_USE_LOCALSCRATCH=0 writes straight to FSCRATCH.  It exists so the staging
# path stays falsifiable, not because it is ever the better choice.
USE_LOCALSCRATCH="${C2_USE_LOCALSCRATCH:-1}"

# Topology, task counts, per-array throttle, memory and wall now come from
# experiments/scripts/c2_slot_plan.py -- ONE python call for all 42 arrays,
# instead of 42 separate `c2_task_spec --count` invocations, and one place where
# the resource table can be unit-tested (tests/unit/test_c2_slot_plan.py, 49
# tests).  Memory in particular is no longer a bash case statement: see
# c2_slot_plan.MEM_GB for the derivation of Bingo-IsalSR's 32 GB.
#
# ✅ The Stage C memory DEVIATION recorded on 2026-08-03 is WITHDRAWN.  It
# existed because §3.3's 256 GB, held across 210 concurrent smoke tasks, would
# have turned the §8.2 achieved-concurrency figure into a measurement of
# fat-node availability.  With Bingo-IsalSR at 32 GB there is nothing to deviate
# from, so the smoke now requests exactly what the campaign requests -- which is
# what §5.1 wanted all along.

# ---- Argument parsing ------------------------------------------------------
MODE="submit"
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   MODE="dry" ;;
        --test-only) MODE="test" ;;
        --one-task)  MODE="one" ;;
        --only)      ONLY="$2"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done

# --dry-run is meant to be runnable from the workstation, where the Picasso
# paths do not exist; only the modes that actually submit need the directories.
if [[ "${MODE}" != "dry" ]]; then
    mkdir -p "${LOGS_DIR}" "${RESULTS_ROOT}"
fi

# ---- Picasso's Lua sbatch wrapper prepends ANSI + a warning banner to
# ---- --parsable output.  Take the LAST line first: a line-wise sed leaves the
# ---- banner's newlines in place and the guard then fires AFTER the job was
# ---- already submitted, leaving an untracked job on the cluster.
_clean_job_id() {
    tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}
submit() {
    local raw id
    raw=$(sbatch --parsable "$@") || { echo "sbatch failed" >&2; return 1; }
    id=$(_clean_job_id "${raw}")
    [[ "${id}" =~ ^[0-9]+$ ]] || { echo "FATAL: unparsable job id: ${raw@Q}" >&2; return 1; }
    echo "${id}"
}

PY="$(conda run -n isalsr which python 2>/dev/null || echo python3)"
REPO_LOCAL="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# The plan is computed from the LOCAL checkout, which deploy.sh has already
# proven identical to the deployed one.  One call, 42 rows.
PLAN_ARGS=(--config-dir "${REPO_LOCAL}/experiments/configs" --seeds "${SEEDS}" --tsv)
MAX_BUNDLE="${C2_MAX_BUNDLE:-${DEF_MAX_BUNDLE}}"
[[ "${MAX_BUNDLE}" -gt 0 ]] && PLAN_ARGS+=(--max-bundle "${MAX_BUNDLE}")
if [[ -n "${UNIFORM_THROTTLE}" ]]; then
    PLAN_ARGS+=(--uniform "${UNIFORM_THROTTLE}")
else
    PLAN_ARGS+=(--budget "${SLOT_BUDGET}")
fi
PLAN="$(PYTHONPATH="${REPO_LOCAL}/src:${REPO_LOCAL}" \
        "${PY}" -m experiments.scripts.c2_slot_plan "${PLAN_ARGS[@]}")" \
    || { echo "FATAL: c2_slot_plan failed; nothing submitted" >&2; exit 1; }
[[ "$(wc -l <<<"${PLAN}")" -eq 42 ]] \
    || { echo "FATAL: plan has $(wc -l <<<"${PLAN}") rows, expected 42" >&2; exit 1; }
# Ten columns since chunking (2026-08-07).  A plan emitted by an older checkout
# has seven, and `read` would silently leave BUNDLE and CUTOFF_S empty -- which
# means bundle 1 and NO deadline, i.e. the campaign runs unchunked and the
# TIMEOUT guard is off.  Refuse instead.
[[ "$(awk -F'\t' 'NR==1{print NF}' <<<"${PLAN}")" -eq 10 ]] \
    || { echo "FATAL: plan has $(awk -F'\t' 'NR==1{print NF}' <<<"${PLAN}") columns, expected 10" >&2; exit 1; }

# ---- A13 preflight: inodes, before anything is submitted --------------------
# P6's failure mode, in its own words: "C2 hits the hard file quota mid-campaign
# and every running task keeps burning wallclock while all its writes fail."
# That is discovered at 70 % of a 54-hour campaign, and it is unrecoverable in
# the sense that matters -- the compute is spent.  This turns it into a two-
# second refusal at submit time.
#
# Coefficients are MEASURED on c2_smoke_v4 (1,260 runs, 7,932 inodes, of which
# 843 are the fixed per-(problem, contrast) artefacts that do NOT scale with
# seeds): 5.63 inodes per run, plus 843, plus two SLURM log files per task.
# FAILS CLOSED.  The first version of this function parsed the wrong columns and
# reported "would submit" for a request that overruns the quota by 13,333 inodes
# -- a check that passes while measuring nothing, which is the exact shape of the
# two C1.11 defects (`sacct -X` returning a blank MaxRSS, then `JobID` matching
# 42 of 1,260 rows and still reporting PASS).  `quota` on Picasso separates the
# space and file halves with a literal U+2551 box character, so positional $6/$7
# straddle the divider.  Split on the divider, then sanity-check the result and
# REFUSE if it is not plausible.
# 🔴 Two different counts since chunking: RESULT inodes scale with CELLS (5.63
# per run, unchanged by grouping), LOG inodes scale with TASKS.  Passing the task
# count for both would under-count results by 3.6x and wave through a submission
# that overruns the quota.
check_inode_budget() {
    local n_cells="$1" n_tasks="$2" half used soft avail need_results need_logs need
    half="$(quota 2>/dev/null | awk -F'║' '/fscratc/ {print $2}')"

    _to_int() {  # "166.4k" | "250.0k" | "400000" -> integer
        awk '{ v=$0; sub(/[kKmM]$/,"",v);
               if ($0 ~ /[kK]$/) v*=1000; else if ($0 ~ /[mM]$/) v*=1000000;
               printf "%d", v }' <<<"$1"
    }
    used=$(_to_int "$(awk '{print $1}' <<<"${half}")")
    soft=$(_to_int "$(awk '{print $2}' <<<"${half}")")

    # Fail closed on an implausible parse rather than waving the campaign through.
    if [[ -z "${half}" ]] || (( used <= 0 || soft <= 0 || soft <= used )); then
        echo "FATAL: could not parse the FSCRATCH file quota (used='${used}' soft='${soft}')." >&2
        echo "       Refusing to submit blind.  Check \`quota\` by hand, or set" >&2
        echo "       C2_SKIP_INODE_CHECK=1 once you have confirmed the headroom." >&2
        return 1
    fi

    avail=$(( soft - used ))
    need_results=$(( n_cells * 563 / 100 + 843 ))
    need_logs=$(( MERGE_LOGS == 1 ? n_tasks : n_tasks * 2 ))
    need=$(( need_results + need_logs ))

    printf '  inode preflight: need %d (results %d + logs %d, %s) against %d free (%d / %d soft)\n' \
           "${need}" "${need_results}" "${need_logs}" \
           "$([[ ${MERGE_LOGS} == 1 ]] && echo merged || echo split)" \
           "${avail}" "${used}" "${soft}"
    if (( need > avail )); then
        cat >&2 <<EOF
FATAL: projected inode use exceeds the FSCRATCH soft quota by $(( need - avail )).
       Nothing submitted.  Without this check the campaign would run for tens of
       hours and then have every write fail while the tasks kept burning
       wallclock (EXECUTION-PLAN P6).

       What actually frees inodes here (measured 2026-08-05):
         C2_MERGE_LOGS=1 (the default)   halves the log count; already applied
                                         if the line above says "merged"
         tar czf c2_smoke_v4.tar.gz c2_smoke_v4 && rm -rf c2_smoke_v4
                                         ~7,930, but ONLY after the owed clean
                                         Stage C wave has passed on its
                                         replacement -- v3 is already archived

       What does NOT: \`conda clean\`.  Its dry run reports nothing to remove, and
       16,060 of the 18,300 files under conda_pkgs are hardlinked into the envs
       (nlink >= 2), so deleting the pkgs entry frees a directory entry and not
       an inode.  Do not spend time on it.

       Override with C2_SKIP_INODE_CHECK=1 only if the 400k hard limit is known
       to cover the shortfall AND the grace period is understood.
EOF
        return 1
    fi
    return 0
}

# Runs on submit AND on --test-only.  --test-only is executed on the cluster
# (defect 15: there is no sbatch on the workstation), so it is exactly where the
# quota is readable and where you want to see the projection before committing.
if [[ ( "${MODE}" == "submit" || "${MODE}" == "test" ) && "${C2_SKIP_INODE_CHECK:-0}" != "1" ]]; then
    check_inode_budget "$(awk -F'\t' '{s+=$10} END {print s}' <<<"${PLAN}")" \
                       "$(awk -F'\t' '{s+=$4}  END {print s}' <<<"${PLAN}")" \
        || { [[ "${MODE}" == "test" ]] && echo "  (--test-only: continuing anyway)" || exit 1; }
fi

JOB_IDS=()
CONFIG_LIST=""
N_TASKS_TOTAL=0
N_CELLS_TOTAL=0
N_ARRAYS=0
SLOTS_TOTAL=0

echo "C2 launcher -- profile '${C2_PROFILE}'"
echo "  repo:       ${ISALSR_REPO_DIR}"
echo "  results:    ${RESULTS_ROOT}"
echo "  logs:       ${LOGS_DIR}"
echo "  seeds:      ${SEEDS}   max_time: ${MAX_TIME}s"
echo "  constraint: ${CONSTRAINT}"
if [[ -n "${UNIFORM_THROTTLE}" ]]; then
    echo "  throttle:   UNIFORM %${UNIFORM_THROTTLE} (A/B escape hatch -- strictly worse)"
else
    echo "  throttle:   apportioned, ${SLOT_BUDGET} slots over 42 arrays"
fi
echo "  mode:       ${MODE}${ONLY:+   filter: ${ONLY}}"
echo ""

while IFS=$'\t' read -r METHOD ARM SUITE N_TASKS THROTTLE MEM PLAN_WALL \
                        BUNDLE CUTOFF_S N_CELLS; do
      [[ -n "${METHOD}" ]] || continue
      KEY="${METHOD}:${ARM}:${SUITE}"
      [[ -n "${ONLY}" && "${KEY}" != "${ONLY}" ]] && continue

      CONFIG="${ISALSR_REPO_DIR}/experiments/configs/${METHOD}_${SUITE}.yaml"
      LOCAL_CONFIG="${REPO_LOCAL}/experiments/configs/${METHOD}_${SUITE}.yaml"
      [[ -f "${LOCAL_CONFIG}" ]] || { echo "FATAL: missing config ${LOCAL_CONFIG}" >&2; exit 1; }

      # 🔴 The wall and the worker's deadline must move TOGETHER.  A wall shorter
      # than the deadline assumes lets the worker start a cell it cannot finish,
      # and SLURM then kills it -- reintroducing exactly the TIMEOUT the deadline
      # exists to make impossible.  So the two are always derived from the same
      # arithmetic, never set independently.
      #
      # The campaign takes the plan's wall, which is sized against the C1-measured
      # per-suite cell duration.  The SMOKE cannot: its cells are capped at
      # MAX_TIME = 900 s, not at the campaign's 43,200 s, so the plan's wall would
      # be ~50x too long while the plan's BUNDLE is sized for 12 h cells.  It
      # derives its own NO-SPILL wall instead -- B x (payload + teardown) -- which
      # is exact, because a smoke cell provably cannot exceed its payload budget.
      #
      # This is what keeps §10.2's promise real: the smoke exercises the same
      # bundles, the same deadline arithmetic and the same worker loop as the
      # campaign, on a payload 48x shorter.
      if [[ -n "${WALL_OVERRIDE}" ]]; then
          WALL_S=$(( BUNDLE * (MAX_TIME + SMOKE_TEARDOWN_S) + SMOKE_TEARDOWN_S ))
          MIN_WALL_S=$(awk -F'[-:]' '{print (($1*24)+$2)*3600 + $3*60 + $4}' <<<"${WALL_OVERRIDE}")
          (( WALL_S < MIN_WALL_S )) && WALL_S=${MIN_WALL_S}
          WALL=$(printf '%d-%02d:%02d:00' $((WALL_S / 86400)) \
                 $((WALL_S % 86400 / 3600)) $((WALL_S % 3600 / 60)))
          CUTOFF_S=$(( WALL_S - MAX_TIME - SMOKE_TEARDOWN_S ))
          (( CUTOFF_S < 1 )) && CUTOFF_S=1
      else
          WALL="${PLAN_WALL}"
      fi
      JOB_NAME="c2s_${METHOD:0:1}${ARM:0:1}_${SUITE}"

      case "${MODE}" in
          one)  ARRAY_SPEC="1-1" ;;
          *)    ARRAY_SPEC="1-${N_TASKS}%${THROTTLE}" ;;
      esac

      SB_ARGS=(
          --array="${ARRAY_SPEC}"
          --job-name="${JOB_NAME}"
          --time="${WALL}"
          --ntasks=1 --cpus-per-task=1
          --mem="${MEM}G"
          --constraint="${CONSTRAINT}"
          --account="${ACCOUNT}"
          --output="${LOGS_DIR}/${JOB_NAME}_%A_%a.out"
      )
      # Omitting --error is what makes SLURM merge stderr into stdout.
      [[ "${MERGE_LOGS}" == "1" ]] \
          || SB_ARGS+=(--error="${LOGS_DIR}/${JOB_NAME}_%A_%a.err")
      SB_ARGS+=(
          --export="ALL,ISALSR_REPO_DIR=${ISALSR_REPO_DIR},C2_METHOD=${METHOD},C2_ARM=${ARM},C2_SUITE=${SUITE},C2_CONFIG=${CONFIG},C2_SEEDS=${SEEDS_EXPORT},C2_MAX_TIME=${MAX_TIME},C2_RESULTS_DIR=${RESULTS_ROOT},C2_BUNDLE=${BUNDLE},C2_START_CUTOFF_S=${CUTOFF_S},C2_USE_LOCALSCRATCH=${USE_LOCALSCRATCH}"
          "${SCRIPT_DIR}/worker.sh"
      )

      N_ARRAYS=$((N_ARRAYS + 1))
      N_TASKS_TOTAL=$((N_TASKS_TOTAL + N_TASKS))
      N_CELLS_TOTAL=$((N_CELLS_TOTAL + N_CELLS))
      SLOTS_TOTAL=$((SLOTS_TOTAL + THROTTLE))
      CONFIG_LIST="${CONFIG_LIST} ${CONFIG}"

      case "${MODE}" in
        dry)
            printf '  %-32s %4d cells /%3d = %4d tasks  %%%-3d  %3dG  %-11s cut %6ds\n' \
                   "${KEY}" "${N_CELLS}" "${BUNDLE}" "${N_TASKS}" "${THROTTLE}" \
                   "${MEM}" "${WALL}" "${CUTOFF_S}"
            ;;
        test)
            if OUT=$(sbatch --test-only "${SB_ARGS[@]}" 2>&1); then
                printf '  %-32s %4d tasks  %%%-3d  OK\n' "${KEY}" "${N_TASKS}" "${THROTTLE}"
            else
                printf '  %-32s %4d tasks  FAILED: %s\n' "${KEY}" "${N_TASKS}" "${OUT}" ; exit 1
            fi
            ;;
        one|submit)
            ID=$(submit "${SB_ARGS[@]}") || exit 1
            JOB_IDS+=("${ID}")
            printf '  %-32s %4s tasks (B=%d)  %%%-3d  %3dG  job %s\n' "${KEY}" \
                   "$([[ ${MODE} == one ]] && echo 1 || echo "${N_TASKS}")" \
                   "${BUNDLE}" "${THROTTLE}" "${MEM}" "${ID}"
            ;;
      esac
done <<< "${PLAN}"

echo ""
echo "Arrays: ${N_ARRAYS}   tasks: ${N_TASKS_TOTAL}   cells: ${N_CELLS_TOTAL}   slots: ${SLOTS_TOTAL}"

# ---- Aggregation: a 14-task array, then one ledger job ---------------------
# afterany, not afterok: if some arrays fail, a status ledger that NAMES the
# missing cells is exactly what this stage exists to produce (C1.15, §5.5).
#
# Was ONE job looping over 14 configs.  Two reasons it is now an array:
#
#   1. Each `--postprocess only` call is scoped to its own config, and a config
#      declares exactly one (method, suite) -- so the aggregate.csv and
#      paired_stats.json paths of any two configs are DISJOINT.  Fourteen
#      concurrent tasks cannot collide.
#   2. Except on one file.  orchestrator.py:555 rebuilt the campaign
#      status_ledger.csv from a FULL RECURSIVE WALK of the whole root, once per
#      config -- fourteen identical walks writing one shared path.  Serially
#      that is waste; concurrently it is a race, the same class as the per-task
#      post-processing race of 2026-08-03.  The walk is now hoisted out
#      (`--postprocess only --no-status-ledger`) into the single dependent
#      ledger job below, which runs it exactly once.
#
# Cost, measured 2026-08-05 on the 1,260-run root: ~4.2 s per problem, set by
# (problem x contrast x metric) and NOT by the number of runs -- 3 seeds to 20
# costs 2.6 %.  The old "~11 h at 8,400 runs" projection extrapolated in the
# wrong variable.
if [[ "${MODE}" == "submit" && ${#JOB_IDS[@]} -gt 0 ]]; then
    DEP=$(IFS=:; echo "${JOB_IDS[*]}")
    N_CONFIGS=$(wc -w <<<"${CONFIG_LIST}")

    # ---- Sweep pass: the deadline's counterpart --------------------------
    # A chunked task refuses to START a cell whose full payload budget no longer
    # fits in the wall, so a chunk that drew an unlucky run of full-budget cells
    # leaves the tail unrun.  That is a deliberate trade -- it buys a campaign in
    # which a SLURM TIMEOUT is impossible -- but the tail still has to run.
    #
    # The sweep is the SAME 42 arrays, resubmitted with the SAME partition, held
    # behind `afterany`.  It costs nothing when nothing spilled: the
    # orchestrator's resume logic reads each cell's run_log.json and skips it, so
    # a task whose cells all completed exits in seconds.  Simulated against the
    # measured per-suite distributions, the sweep carries 5-20 real cells out of
    # 12,600.
    #
    # Submitted at launch, not by hand, for the reason §11.3 gives about the
    # aggregation: a recovery step that depends on someone remembering is a
    # recovery step that does not happen.
    if [[ "${C2_SWEEP:-1}" == "1" ]]; then
        SWEEP_IDS=()
        while IFS=$'\t' read -r METHOD ARM SUITE N_TASKS THROTTLE MEM PLAN_WALL \
                                BUNDLE CUTOFF_S N_CELLS; do
            [[ -n "${METHOD}" ]] || continue
            [[ "${BUNDLE}" -gt 1 ]] || continue     # B=1 cannot spill: nothing to sweep
            CONFIG="${ISALSR_REPO_DIR}/experiments/configs/${METHOD}_${SUITE}.yaml"
            WALL="${WALL_OVERRIDE:-${PLAN_WALL}}"
            SWEEP_NAME="c2w_${METHOD:0:1}${ARM:0:1}_${SUITE}"
            SID=$(submit \
                --array="1-${N_TASKS}%${THROTTLE}" \
                --job-name="${SWEEP_NAME}" \
                --time="${WALL}" \
                --ntasks=1 --cpus-per-task=1 \
                --mem="${MEM}G" \
                --constraint="${CONSTRAINT}" \
                --account="${ACCOUNT}" \
                --dependency="afterany:${DEP}" \
                --output="${LOGS_DIR}/${SWEEP_NAME}_%A_%a.out" \
                --export="ALL,ISALSR_REPO_DIR=${ISALSR_REPO_DIR},C2_METHOD=${METHOD},C2_ARM=${ARM},C2_SUITE=${SUITE},C2_CONFIG=${CONFIG},C2_SEEDS=${SEEDS_EXPORT},C2_MAX_TIME=${MAX_TIME},C2_RESULTS_DIR=${RESULTS_ROOT},C2_BUNDLE=${BUNDLE},C2_START_CUTOFF_S=${CUTOFF_S},C2_USE_LOCALSCRATCH=${USE_LOCALSCRATCH}" \
                "${SCRIPT_DIR}/worker.sh") || exit 1
            if scontrol show job "${SID}" | grep -q 'Dependency=(null)'; then
                echo "FATAL: sweep dependency dropped -- cancelling ${SID}" >&2
                scancel "${SID}"; exit 1
            fi
            SWEEP_IDS+=("${SID}")
        done <<< "${PLAN}"
        if [[ ${#SWEEP_IDS[@]} -gt 0 ]]; then
            echo "Sweep arrays: ${#SWEEP_IDS[@]} (afterany on the ${#JOB_IDS[@]} main arrays)"
            # The aggregation must wait for the SWEEP, not just the main pass, or
            # it aggregates a tree that is still missing its deferred cells.
            DEP="${DEP}:$(IFS=:; echo "${SWEEP_IDS[*]}")"
            printf '%s\n' "${SWEEP_IDS[@]}" > "${LOGS_DIR}/sweep_job_ids.txt"
        fi
    fi

    AGG_ID=$(submit \
        --job-name=c2s_aggregate \
        --array="1-${N_CONFIGS}" \
        --time="${AGG_WALL}" \
        --ntasks=1 --cpus-per-task=2 --mem=16G \
        --constraint="${CONSTRAINT}" \
        --account="${ACCOUNT}" \
        --dependency="afterany:${DEP}" \
        --output="${LOGS_DIR}/c2s_aggregate_%A_%a.out" \
        --error="${LOGS_DIR}/c2s_aggregate_%A_%a.err" \
        --export="ALL,ISALSR_REPO_DIR=${ISALSR_REPO_DIR},C2_RESULTS_DIR=${RESULTS_ROOT},C2_CONFIG_LIST=${CONFIG_LIST# }" \
        "${SCRIPT_DIR}/aggregate_worker.sh") || exit 1

    # sbatch ACCEPTS a malformed dependency and records Dependency=(null): the
    # job would then start immediately, against partial input.
    if scontrol show job "${AGG_ID}" | grep -q 'Dependency=(null)'; then
        echo "FATAL: aggregation dependency dropped -- cancelling ${AGG_ID}" >&2
        scancel "${AGG_ID}"; exit 1
    fi
    echo "Aggregation array ${AGG_ID} (${N_CONFIGS} tasks, afterany on ${#JOB_IDS[@]} arrays)"

    # The single full-root walk.  afterany on the aggregation array, so a ledger
    # naming the gaps is still produced when some configs failed.
    #
    # 🔴 C2_EXPECTED_TASKS is a CELL count, not a task count.  `c2_certify`
    # documents the argument as "Number of cells the campaign should contain",
    # and `build_expected_cells` uses the strict registry universe ONLY when
    # `len(canonical) == expected_tasks`.  Before chunking the two were equal and
    # this was harmless; since 2026-08-07 they are not (Stage C: 1,260 cells in
    # 453 tasks), and passing the task count made the certifier fall through to
    # the "disk" universe -- completeness measured against what was FOUND rather
    # than what SHOULD exist, under which a whole absent problem directory is
    # invisible.  Stage C v6 duly reported GO with `expected_set_source: disk`.
    # `submit_paced.sh` always got this right; only this path was wrong.
    LEDGER_ID=$(submit \
        --job-name=c2s_ledger \
        --time="0-02:00:00" \
        --ntasks=1 --cpus-per-task=1 --mem=16G \
        --constraint="${CONSTRAINT}" \
        --account="${ACCOUNT}" \
        --dependency="afterany:${AGG_ID}" \
        --output="${LOGS_DIR}/c2s_ledger_%j.out" \
        --error="${LOGS_DIR}/c2s_ledger_%j.err" \
        --export="ALL,ISALSR_REPO_DIR=${ISALSR_REPO_DIR},C2_RESULTS_DIR=${RESULTS_ROOT},C2_LEDGER_ONLY=1,C2_EXPECTED_TASKS=${N_CELLS_TOTAL},C2_MAX_TIME=${MAX_TIME}" \
        "${SCRIPT_DIR}/aggregate_worker.sh") || exit 1

    if scontrol show job "${LEDGER_ID}" | grep -q 'Dependency=(null)'; then
        echo "FATAL: ledger dependency dropped -- cancelling ${LEDGER_ID}" >&2
        scancel "${LEDGER_ID}"; exit 1
    fi
    echo "Status-ledger job ${LEDGER_ID} (afterany on ${AGG_ID})"
fi

if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    echo ""
    echo "Monitor:  ssh picasso 'squeue'"
    echo "States:   ssh picasso 'sacct -j ${JOB_IDS[0]} -X -n -P -o JobID,State | cut -d\| -f2 | sort | uniq -c'"
    echo "Memory:   ssh picasso \"sacct -j <ID> -n -P -o JobID,MaxRSS | awk -F'|' '\\\$1 ~ /\\.batch\$/'\"   # NOT -X"
    echo "Logs:     ${LOGS_DIR}"
    printf '%s\n' "${JOB_IDS[@]}" > "${LOGS_DIR}/job_ids.txt"
    echo "Job ids:  ${LOGS_DIR}/job_ids.txt"
fi
