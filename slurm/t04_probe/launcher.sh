#!/usr/bin/env bash
# =============================================================================
# T04 hash-arm Picasso PROBE launcher
# =============================================================================
# Submits two arrays -- one per host -- covering all three arms plus the AC-10
# shadow ON/OFF pair.  28 tasks total.
#
#   bash slurm/t04_probe/launcher.sh --dry-run     # print sbatch commands
#   bash slurm/t04_probe/launcher.sh --test-only   # sbatch --test-only, no queue
#   bash slurm/t04_probe/launcher.sh --one         # ONE task per host (cluster smoke)
#   bash slurm/t04_probe/launcher.sh               # the full 28-task probe
#
# SP-0 (EXECUTION-PLAN.md §4.0) is binding and this script enforces it:
#   * max_time 1500 s   (cap 1800)
#   * 28 tasks          (cap 60)
#   * seed 0 only       (never 1..20 -- a probe must never look like a C2 cell)
#   * output under ~/execs/isalsr/t04_probe/, never the campaign root
# This is a probe.  It answers "does this work on Picasso?".  It produces NO
# number for the paper.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurable ----------------------------------------------------------
export CONDA_ENV_NAME="isalsr"
export REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR"
export T04_TASKS="${REPO_DIR}/slurm/t04_probe/tasks.txt"
export T04_OUT="/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/t04_probe"
export T04_MAX_TIME="1500"          # seconds; SP-0 cap is 1800
LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/t04_probe/logs"
ACCOUNT="tic_163_uma"
THROTTLE=8

mkdir -p "${LOGS_DIR}" "${T04_OUT}"

MODE="run"
case "${1:-}" in
    --dry-run)   MODE="dry" ;;
    --test-only) MODE="test" ;;
    --one)       MODE="one" ;;
    "")          MODE="run" ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
esac

# ---- Job-id capture --------------------------------------------------------
# Picasso's Lua sbatch wrapper prepends ANSI codes and a multi-line warning to
# --parsable output.  Take the LAST line BEFORE stripping: a line-by-line sed
# leaves the warning's newlines in place, and a guard that then rejects the
# result fires *after* the job was submitted, leaving an untracked job running.
_clean_job_id() {
    tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}

submit() {
    local raw id
    raw=$(sbatch --parsable "$@") || { echo "sbatch failed" >&2; return 1; }
    id=$(_clean_job_id "${raw}")
    [[ "${id}" =~ ^[0-9]+$ ]] || {
        echo "FATAL: unparsable job id: ${raw@Q}" >&2
        echo "       run 'squeue -u \$USER' NOW -- assume the job exists." >&2
        return 1
    }
    echo "${id}"
}

# ---- Per-host submission ---------------------------------------------------
# --mem differs by host, so the two hosts go as separate arrays.  Bingo+IsalSR
# historically needed 128 G from heap fragmentation; the C++ dedup set should
# cut that sharply, but AC-10 requires it MEASURED, not assumed -- so request
# generously here and read MaxRSS from sacct afterwards.
submit_host() {
    local host="$1" first="$2" last="$3" mem="$4"
    local range="${first}-${last}"
    [[ "${MODE}" == "one" ]] && range="${first}-${first}"

    local -a args=(
        --array="${range}%${THROTTLE}"
        --job-name="t04probe-${host}"
        --mem="${mem}"
        --account="${ACCOUNT}"
        --output="${LOGS_DIR}/t04probe-${host}_%A_%a.out"
        --error="${LOGS_DIR}/t04probe-${host}_%A_%a.err"
        --export=ALL,CONDA_ENV_NAME="${CONDA_ENV_NAME}",REPO_DIR="${REPO_DIR}",T04_TASKS="${T04_TASKS}",T04_OUT="${T04_OUT}",T04_MAX_TIME="${T04_MAX_TIME}"
        "${SCRIPT_DIR}/worker.sh"
    )

    case "${MODE}" in
        dry)
            echo "[DRY-RUN] sbatch ${args[*]}"
            ;;
        test)
            echo "--- sbatch --test-only (${host}) ---"
            sbatch --test-only "${args[@]}"
            ;;
        *)
            local id
            id=$(submit "${args[@]}") || exit 1
            echo "Submitted ${host}: job ${id}  (tasks ${range})"
            echo "  logs: ${LOGS_DIR}/t04probe-${host}_${id}_*.out"
            ;;
    esac
}

echo "T04 PROBE -- mode=${MODE}, max_time=${T04_MAX_TIME}s, seed 0, out=${T04_OUT}"
echo ""

# tasks.txt rows 1-14 are bingo, 15-28 are udfs (comments/blanks stripped by the worker)
submit_host bingo  1 14 "48G"
submit_host udfs  15 28 "16G"

echo ""
echo "Monitor:  squeue -u \$USER -o '%.10i %.9P %.20j %.2t %.10M %R'"
echo "Errors:   tail -n 40 ${LOGS_DIR}/*.err"
echo "MaxRSS:   sacct -j <JOBID> -X -o JobID,State,Elapsed,MaxRSS,NodeList"
