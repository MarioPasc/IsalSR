#!/usr/bin/env bash
# =============================================================================
# T19 complexity-telemetry PROBE -- launcher
# =============================================================================
#   bash slurm/t19_probe/launcher.sh --dry-run     # print the sbatch command
#   bash slurm/t19_probe/launcher.sh --test-only   # sbatch --test-only, no queue
#   bash slurm/t19_probe/launcher.sh --one         # ONE task (cluster smoke)
#   bash slurm/t19_probe/launcher.sh               # the full 24-task probe
#
# Answers one question: does the T19 structural telemetry populate correctly,
# on Picasso, for every (method, arm) combination, WITHOUT disturbing anything
# the campaign already records?  It produces no number for the paper.
#
# SP-0 (EXECUTION-PLAN §4.0) is binding and this script enforces it:
#   * 24 tasks        (cap 60)
#   * max_time 900 s  (cap 1800)
#   * seeds 0 and 101 (never 1..30)
#   * output under ~/execs/isalsr/t19_probe/, never the campaign root
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurable ----------------------------------------------------------
export ISALSR_REPO_DIR="${ISALSR_REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR}"
export T19_TASKS="${ISALSR_REPO_DIR}/slurm/t19_probe/tasks.txt"
export T19_OUT="/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/t19_probe"
export T19_RESULTS_DIR="${T19_OUT}/results"
export T19_MAX_TIME="900"          # seconds; SP-0 cap is 1800
LOGS_DIR="${T19_OUT}/logs"

# Resources.  A 900 s payload cannot approach the campaign's memory profile
# (Stage D measured 1.05-1.16 GB peak on FULL-LENGTH bingo_isalsr cells), so a
# uniform 16 G is ~14x the measured ceiling and keeps every task schedulable on
# one node family.
MEM_GB=16
CPUS=1
WALLCLOCK="0-00:40:00"             # payload 900 s + startup, teardown, margin
THROTTLE=12
# Pinned to the campaign's node pool.  Wall clock is a reported quantity here
# (complexity_time_s), and the sd/sr families differ enough in single-core
# speed that an unpinned pool would turn that number into a measurement of the
# scheduler.
CONSTRAINT="sr"
ACCOUNT="tic_163_uma"

MODE="submit"
case "${1:-}" in
    --dry-run)  MODE="dry" ;;
    --test-only) MODE="test" ;;
    --one)      MODE="one" ;;
    "")         ;;
    *)          echo "unknown option: $1" >&2; exit 2 ;;
esac

# ---- Pre-flight ------------------------------------------------------------
N_TASKS=$(grep -vc -e '^[[:space:]]*#' -e '^[[:space:]]*$' "${SCRIPT_DIR}/tasks.txt" || true)
if [[ "${N_TASKS}" -ne 24 ]]; then
    echo "[FATAL] tasks.txt has ${N_TASKS} tasks; expected 24." >&2
    exit 1
fi
if [[ "${N_TASKS}" -gt 60 ]]; then
    echo "[FATAL] ${N_TASKS} tasks exceeds the SP-0 cap of 60." >&2
    exit 1
fi
if [[ "${T19_MAX_TIME}" -gt 1800 ]]; then
    echo "[FATAL] max_time ${T19_MAX_TIME}s exceeds the SP-0 cap of 1800s." >&2
    exit 1
fi
# A probe must never write into the campaign root, whatever else goes wrong.
case "${T19_RESULTS_DIR}" in
    *c2_3arm*|*c2_smoke*|*c2_cert*|*c2_trace*)
        echo "[FATAL] results dir '${T19_RESULTS_DIR}' collides with a campaign root." >&2
        exit 1 ;;
esac
# SP-0: assert no campaign seed appears in the task list.
if awk '!/^[[:space:]]*#/ && NF {if ($4 >= 1 && $4 <= 30) exit 1}' "${SCRIPT_DIR}/tasks.txt"; then
    :
else
    echo "[FATAL] tasks.txt contains a seed in the campaign range 1..30 (SP-0)." >&2
    exit 1
fi

mkdir -p "${LOGS_DIR}" "${T19_RESULTS_DIR}"

ARRAY_SPEC="1-${N_TASKS}%${THROTTLE}"
[[ "${MODE}" == "one" ]] && ARRAY_SPEC="1"

# Picasso's Lua sbatch wrapper prepends ANSI codes and a multi-line warning to
# --parsable output, so the id must be taken from the LAST line before any
# non-digit stripping.  A line-by-line sed leaves the warning's newlines in and
# the guard then fires AFTER the job was already submitted.
_clean_job_id() {
    tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}

SBATCH_ARGS=(
    --parsable
    --job-name=t19probe
    --array="${ARRAY_SPEC}"
    --time="${WALLCLOCK}"
    --ntasks=1
    --cpus-per-task="${CPUS}"
    --mem="${MEM_GB}G"
    --constraint="${CONSTRAINT}"
    --account="${ACCOUNT}"
    --output="${LOGS_DIR}/t19_%A_%a.out"
    --error="${LOGS_DIR}/t19_%A_%a.err"
    --export="ALL,ISALSR_REPO_DIR=${ISALSR_REPO_DIR},T19_TASKS=${T19_TASKS},T19_RESULTS_DIR=${T19_RESULTS_DIR},T19_MAX_TIME=${T19_MAX_TIME}"
    "${SCRIPT_DIR}/worker.sh"
)

echo "T19 complexity-telemetry probe"
echo "  tasks:      ${N_TASKS} (array ${ARRAY_SPEC})"
echo "  payload:    max_time=${T19_MAX_TIME}s"
echo "  resources:  ${CPUS} cpu, ${MEM_GB}G, ${WALLCLOCK}, constraint=${CONSTRAINT}"
echo "  results:    ${T19_RESULTS_DIR}"
echo "  logs:       ${LOGS_DIR}"
echo ""

case "${MODE}" in
    dry)
        echo "[DRY-RUN] sbatch ${SBATCH_ARGS[*]}"
        exit 0 ;;
    test)
        sbatch --test-only "${SBATCH_ARGS[@]:1}"
        exit 0 ;;
esac

RAW="$(sbatch "${SBATCH_ARGS[@]}")" || { echo "sbatch failed" >&2; exit 1; }
JOB_ID="$(_clean_job_id "${RAW}")"
if [[ ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "[FATAL] unparsable job id: ${RAW@Q}" >&2
    echo "        A job may nonetheless have been submitted -- run 'squeue' now." >&2
    exit 1
fi

echo "${JOB_ID}" > "${LOGS_DIR}/job_id.txt"
echo "Submitted array ${JOB_ID} (${N_TASKS} tasks)"
echo ""
echo "Monitor:  ssh picasso 'squeue'"
echo "States:   ssh picasso \"sacct -j ${JOB_ID} -X -n -P -o JobID,State | awk -F'|' '{print \\\$2}' | sort | uniq -c\""
echo "Verify:   python slurm/t19_probe/verify.py <local copy of ${T19_RESULTS_DIR}>"
