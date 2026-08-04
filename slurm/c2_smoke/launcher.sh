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
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configuration ---------------------------------------------------------
export ISALSR_REPO_DIR="${ISALSR_REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR}"
RESULTS_ROOT="${C2_RESULTS_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/results/isalsr/c2_smoke}"
LOGS_DIR="${C2_LOGS_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/c2_smoke/logs}"
ACCOUNT="tic_163_uma"

SEEDS="0,101,102"          # T17 §0.1: outside campaign 1..20 AND the 21..30 top-up range
# 🔴 sbatch --export is COMMA-separated, so a comma inside a VALUE is parsed as
# the start of the next variable. Exporting C2_SEEDS=0,101,102 delivered
# C2_SEEDS=0 to the worker -- one seed instead of three -- so every array had
# n_problems valid indices instead of n_problems*3 and every task above that
# died with "index N out of range". Worse, the tasks that DID run produced
# correct-looking seed-0 cells, so the array was 1/3 right and 2/3 failed rather
# than failing outright. Ship the list colon-separated; the worker translates.
SEEDS_EXPORT="${SEEDS//,/:}"
MAX_TIME=900               # payload budget, seconds.  Never the production 43,200
WALL="0-00:40:00"          # SLURM limit: >2x the payload, so a SLURM kill means a real defect (C1.12)
THROTTLE="${C2_THROTTLE:-8}"   # per array; 42 arrays x 8 = up to 336 concurrent (§8.2 target ~300)

# 🔴 The 336-task ceiling above is OURS, not the cluster's.  The achieved 245
# cores measured on 2026-08-03 is 73 % of it, and the QOS entitlement is
# cpu=9000 with thousands of cores routinely idle -- so "C2 takes 17.1 days"
# was an artefact of this variable, not a contention measurement.  Raise it
# (C2_THROTTLE=24 -> 1,008) before trading away D2 coverage, the hash arm or
# seeds under §8.3.  See EXECUTION-PLAN §11.1, 2026-08-04.

# Aggregation wall.  2 h was NOT enough and cost a verdict: job 1753134 spent
# ~1h40 on the 14-config `--postprocess only` loop over 1,260 runs and was
# killed at its wall BEFORE the certifier emitted anything.  The artefacts
# persisted, so `certify.sh` recovered it -- but the stage read as failed for a
# day.  The cost scales with RUNS, not configs, so C2's 8,400 runs need >=24 h.
AGG_WALL="${C2_AGG_WALL:-0-06:00:00}"

# Node family: NOT pinned.  The engine is x86-64-v3 / avx512f=0, hence portable
# across sd/sr/bc/bl (B6b), and Stage C produces no number that enters a table,
# so the wall-clock-homogeneity argument for pinning does not apply here.  Every
# run records its own cpu_model (A7), so B5/B6 get the node census as a
# by-product and the arm balance is reportable.
CONSTRAINT="${C2_CONSTRAINT:-cpu}"

METHODS=(udfs bingo)
ARMS=(baseline hash isalsr)
SUITES=(nguyen feynman hard cherrypicked roundoff feynman_remainder strogatz)

# Memory per (method, arm).  DEVIATION FROM PRODUCTION, RECORDED DELIBERATELY:
# EXECUTION-PLAN §3.3 sets Bingo-IsalSR to 256 GB in C2, a figure derived from
# C1 runs that hit MaxRSS ~127.7 GB after HOURS of evolution.  A 900 s run
# cannot approach it, and holding 256 GB x 210 tasks for 15 minutes would make
# the §8.2 achieved-concurrency figure a measurement of fat-node availability
# rather than of core contention.  C1.11's product is the MEASURED MaxRSS, which
# the request does not affect, and the plan itself designates Stage D (D1.2, at
# the full 12 h budget) as what sizes production memory.  Values below are
# >4x any plausible 900 s peak and are re-checked against the one-task probe
# before the wave goes out.
# Overridable, because the first 42-task probe measured a peak MaxRSS of 343 MB
# across every arm -- 50-140x below these requests. Over-requesting is not free:
# SLURM cannot pack tasks, which throttles exactly the achieved-concurrency
# figure §8.2 needs from this stage. Set from measurement, not from history.
mem_for() {
    case "$1:$2" in
        bingo:isalsr)  echo "${C2_MEM_BINGO_ISALSR:-16}" ;;
        bingo:*)       echo "${C2_MEM_BINGO:-16}" ;;
        udfs:*)        echo "${C2_MEM_UDFS:-8}" ;;
    esac
}

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

task_count() {   # config -> number of array tasks (n_problems * n_seeds)
    "${PY}" -m experiments.scripts.c2_task_spec --config "$1" --seeds "${SEEDS}" --count
}

JOB_IDS=()
CONFIG_LIST=""
N_TASKS_TOTAL=0
N_ARRAYS=0

echo "Stage C -- C2 pre-flight certification smoke"
echo "  repo:       ${ISALSR_REPO_DIR}"
echo "  results:    ${RESULTS_ROOT}"
echo "  logs:       ${LOGS_DIR}"
echo "  seeds:      ${SEEDS}   max_time: ${MAX_TIME}s   wall: ${WALL}"
echo "  constraint: ${CONSTRAINT}   throttle: %${THROTTLE} per array"
echo "  mode:       ${MODE}${ONLY:+   filter: ${ONLY}}"
echo ""

for METHOD in "${METHODS[@]}"; do
  for ARM in "${ARMS[@]}"; do
    for SUITE in "${SUITES[@]}"; do
      KEY="${METHOD}:${ARM}:${SUITE}"
      [[ -n "${ONLY}" && "${KEY}" != "${ONLY}" ]] && continue

      CONFIG="${ISALSR_REPO_DIR}/experiments/configs/${METHOD}_${SUITE}.yaml"
      LOCAL_CONFIG="$(cd "${SCRIPT_DIR}/../.." && pwd)/experiments/configs/${METHOD}_${SUITE}.yaml"
      [[ -f "${LOCAL_CONFIG}" ]] || { echo "FATAL: missing config ${LOCAL_CONFIG}" >&2; exit 1; }

      N_TASKS="$(task_count "${LOCAL_CONFIG}")"
      [[ "${N_TASKS}" =~ ^[0-9]+$ && "${N_TASKS}" -gt 0 ]] \
          || { echo "FATAL: bad task count '${N_TASKS}' for ${KEY}" >&2; exit 1; }
      MEM="$(mem_for "${METHOD}" "${ARM}")"
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
          --error="${LOGS_DIR}/${JOB_NAME}_%A_%a.err"
          --export="ALL,ISALSR_REPO_DIR=${ISALSR_REPO_DIR},C2_METHOD=${METHOD},C2_ARM=${ARM},C2_SUITE=${SUITE},C2_CONFIG=${CONFIG},C2_SEEDS=${SEEDS_EXPORT},C2_MAX_TIME=${MAX_TIME},C2_RESULTS_DIR=${RESULTS_ROOT}"
          "${SCRIPT_DIR}/worker.sh"
      )

      N_ARRAYS=$((N_ARRAYS + 1))
      N_TASKS_TOTAL=$((N_TASKS_TOTAL + N_TASKS))
      CONFIG_LIST="${CONFIG_LIST} ${CONFIG}"

      case "${MODE}" in
        dry)
            printf '  %-32s %3d tasks  %3dG  %s\n' "${KEY}" "${N_TASKS}" "${MEM}" "${ARRAY_SPEC}"
            ;;
        test)
            if OUT=$(sbatch --test-only "${SB_ARGS[@]}" 2>&1); then
                printf '  %-32s %3d tasks  OK\n' "${KEY}" "${N_TASKS}"
            else
                printf '  %-32s %3d tasks  FAILED: %s\n' "${KEY}" "${N_TASKS}" "${OUT}" ; exit 1
            fi
            ;;
        one|submit)
            ID=$(submit "${SB_ARGS[@]}") || exit 1
            JOB_IDS+=("${ID}")
            printf '  %-32s %3s tasks  %3dG  job %s\n' "${KEY}" \
                   "$([[ ${MODE} == one ]] && echo 1 || echo "${N_TASKS}")" "${MEM}" "${ID}"
            ;;
      esac
    done
  done
done

echo ""
echo "Arrays: ${N_ARRAYS}   tasks: ${N_TASKS_TOTAL}"

# ---- Aggregation job -------------------------------------------------------
# afterany, not afterok: if some arrays fail, a status ledger that NAMES the
# missing cells is exactly what Stage C exists to produce (C1.15, §5.5).
if [[ "${MODE}" == "submit" && ${#JOB_IDS[@]} -gt 0 ]]; then
    DEP=$(IFS=:; echo "${JOB_IDS[*]}")
    AGG_ID=$(submit \
        --job-name=c2s_aggregate \
        --time="${AGG_WALL}" \
        --ntasks=1 --cpus-per-task=2 --mem=32G \
        --constraint="${CONSTRAINT}" \
        --account="${ACCOUNT}" \
        --dependency="afterany:${DEP}" \
        --output="${LOGS_DIR}/c2s_aggregate_%j.out" \
        --error="${LOGS_DIR}/c2s_aggregate_%j.err" \
        --export="ALL,ISALSR_REPO_DIR=${ISALSR_REPO_DIR},C2_RESULTS_DIR=${RESULTS_ROOT},C2_CONFIG_LIST=${CONFIG_LIST# }" \
        "${SCRIPT_DIR}/aggregate_worker.sh") || exit 1

    # sbatch ACCEPTS a malformed dependency and records Dependency=(null): the
    # job would then start immediately, against partial input.
    if scontrol show job "${AGG_ID}" | grep -q 'Dependency=(null)'; then
        echo "FATAL: aggregation dependency dropped -- cancelling ${AGG_ID}" >&2
        scancel "${AGG_ID}"; exit 1
    fi
    echo "Aggregation job ${AGG_ID} (afterany on ${#JOB_IDS[@]} arrays)"
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
