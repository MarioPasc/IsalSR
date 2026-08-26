#!/usr/bin/env bash
# Submit the C++ extension smoke to Picasso (T01 AC-1 / certification gates G5, G8).
#
# Usage:
#   bash slurm/smoke_cpp/launcher.sh --dry-run    # print the sbatch command
#   bash slurm/smoke_cpp/launcher.sh --test-only  # sbatch --test-only, no queue
#   bash slurm/smoke_cpp/launcher.sh              # submit one task
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="isalsr"
export REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR"
export GCC_MODULE="gcc/13.2.0"
export LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/logs"
mkdir -p "${LOGS_DIR}"

MODE="submit"
case "${1:-}" in
    --dry-run)   MODE="dry" ;;
    --test-only) MODE="test" ;;
    "")          ;;
    *)           echo "unknown argument: $1" >&2; exit 2 ;;
esac

SBATCH_ARGS=(
    --parsable
    --output="${LOGS_DIR}/smoke_cpp_%j.out"
    --error="${LOGS_DIR}/smoke_cpp_%j.err"
    --export="ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},LOGS_DIR=${LOGS_DIR},GCC_MODULE=${GCC_MODULE}"
    "${SCRIPT_DIR}/worker.sh"
)

if [[ "${MODE}" == "dry" ]]; then
    echo "[DRY-RUN] sbatch ${SBATCH_ARGS[*]}"
    exit 0
fi

if [[ "${MODE}" == "test" ]]; then
    sbatch --test-only "${SBATCH_ARGS[@]}"
    exit $?
fi

# Picasso's Lua sbatch wrapper prepends ANSI codes and a multi-line warning to
# --parsable output.  Take the LAST line before stripping: a line-by-line sed
# leaves the warning's newlines intact, and a guard that then rejects the value
# fires only after the job is already queued -- leaving it untracked.
_clean_job_id() {
    tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}

RAW=$(sbatch "${SBATCH_ARGS[@]}")
JOB_ID=$(_clean_job_id "${RAW}")
if [[ ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "FATAL: unparsable job id from: ${RAW@Q}" >&2
    echo "A job may still have been submitted -- run: squeue -u \$USER" >&2
    exit 1
fi

echo "Submitted job ${JOB_ID}"
echo "Monitor : squeue -j ${JOB_ID}"
echo "Stdout  : ${LOGS_DIR}/smoke_cpp_${JOB_ID}.out"
echo "Stderr  : ${LOGS_DIR}/smoke_cpp_${JOB_ID}.err"
echo "Reports : ${LOGS_DIR}/{equivalence,bench}_${JOB_ID}.json"
