#!/usr/bin/env bash
# =============================================================================
# T05 D2 Picasso PROBE launcher
# =============================================================================
# Usage:
#   bash slurm/t05_probe/launcher.sh --dry-run     # print sbatch commands
#   bash slurm/t05_probe/launcher.sh --test-only   # sbatch --test-only, no queue
#   bash slurm/t05_probe/launcher.sh --one         # ONE real task per host
#   bash slurm/t05_probe/launcher.sh               # the 40-task probe
#
# 🚫 DO NOT SUBMIT WHILE ANOTHER PROBE ARRAY IS IN FLIGHT.  SP-0 discipline is
#    per-probe, and two probes with overlapping log directories is exactly the
#    confusion it exists to prevent.  Check first:  ssh picasso 'squeue'
#
# 🚫 THIS IS NOT THE CAMPAIGN.  EXECUTION-PLAN.md §4.0 SP-0: max_time <= 1800 s,
#    <= 60 tasks, seed 0 only, output under ~/execs/isalsr/.  C2 is submitted
#    once, by Mario, after Stage F sign-off.  Nothing here produces a number for
#    the paper.
#
# Run --dry-run, then --test-only, then --one, then the array.  Do not skip the
# middle steps; they are what catch the errors that only appear on a compute
# node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurable ----------------------------------------------------------
export CONDA_ENV_NAME="isalsr"
export REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR"
export T05_TASKS="${REPO_DIR}/slurm/t05_probe/tasks.txt"
export T05_OUT="/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/t05_probe"
export T05_MAX_TIME="1500"          # seconds; SP-0 cap is 1800
LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/logs"
ACCOUNT="tic_163_uma"
THROTTLE=10

mkdir -p "${LOGS_DIR}" "${T05_OUT}"

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
        echo "       run 'squeue' NOW -- assume the job exists." >&2
        return 1
    }
    echo "${id}"
}

# ---- Per-host submission ---------------------------------------------------
# --mem differs by host, so the two hosts go as separate arrays.  Bingo+IsalSR
# historically needed 128 G from heap fragmentation; the C++ dedup set should
# cut that sharply.  These figures mirror the T04 probe, which is the closest
# measured precedent; read MaxRSS from sacct afterwards and resize, remembering
# that `sacct -X` returns an EMPTY MaxRSS -- memory is on the .batch step.
submit_host() {
    local host="$1" first="$2" last="$3" mem="$4"
    local range="${first}-${last}"
    [[ "${MODE}" == "one" ]] && range="${first}-${first}"

    local -a args=(
        --array="${range}%${THROTTLE}"
        --job-name="t05probe-${host}"
        --mem="${mem}"
        --account="${ACCOUNT}"
        --output="${LOGS_DIR}/t05probe-${host}_%A_%a.out"
        --error="${LOGS_DIR}/t05probe-${host}_%A_%a.err"
        --export=ALL,CONDA_ENV_NAME="${CONDA_ENV_NAME}",REPO_DIR="${REPO_DIR}",T05_TASKS="${T05_TASKS}",T05_OUT="${T05_OUT}",T05_MAX_TIME="${T05_MAX_TIME}"
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
            echo "  logs: ${LOGS_DIR}/t05probe-${host}_${id}_*.out"
            ;;
    esac
}

echo "T05 D2 PROBE -- mode=${MODE}, max_time=${T05_MAX_TIME}s, seed 0, out=${T05_OUT}"
echo "20 D2 problems x 2 hosts, isalsr arm only.  40 tasks, SP-0 cap is 60."
echo ""

# tasks.txt rows 1-20 are bingo, 21-40 are udfs (comments/blanks stripped by the
# worker before indexing).  Regenerate with make_tasks.py if D2 ever changes --
# and if it does, re-check these boundaries.
submit_host bingo  1 20 "48G"
submit_host udfs  21 40 "16G"

echo ""
echo "After the array:"
echo "  ssh picasso 'sacct -j <ID> -X -n -P -o JobID,State,Elapsed,NodeList'"
echo "  ssh picasso 'sacct -j <ID> -n -P -o JobID,MaxRSS | awk -F\"|\" '\\''\$1 ~ /\\.batch\$/'\\'''"
echo "  python slurm/t05_probe/check_d2.py --verify-runs ${T05_OUT} --out sp7_all.json"
