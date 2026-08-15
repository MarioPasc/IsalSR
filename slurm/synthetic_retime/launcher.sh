#!/usr/bin/env bash
# Submit the synthetic permutation-scalability re-timing to Picasso.
#
# Re-measures the twelve values still standing as placeholders in the
# supplementary's synthetic-scalability appendix, on the compiled engine and on
# the CPU family the C2 campaign ran on.
#
# Usage:
#   bash slurm/synthetic_retime/launcher.sh --dry-run
#   bash slurm/synthetic_retime/launcher.sh --test-only
#   bash slurm/synthetic_retime/launcher.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurable ----------------------------------------------------------
export CONDA_ENV_NAME="isalsr"
export REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR"
export RESULTS_DIR="/mnt/home/users/tic_163_uma/mpascual/results/isalsr/synthetic_retime"
export LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/logs"

# Protocol, exactly as the supplementary states it. Do not "tune" these: the
# archived run that these values replace used 30 expressions per cell and capped
# k=9 at 50,000 permutations, against the 200 and exhaustive k! the text
# describes, and that mismatch is half the reason for re-running at all.
export N_EXPR=200          # expressions per (k, m) cell
export MAX_PERMS=50000000  # >= 9! = 362,880, so every cell is exhaustive
export PERM_TIMEOUT=120.0  # seconds, per permutation
export GLOBAL_SEED=42

mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}" 2>/dev/null || true

MODE="submit"
case "${1:-}" in
    --dry-run)   MODE="dry" ;;
    --test-only) MODE="test" ;;
    "")          ;;
    *)           echo "unknown flag: $1" >&2; exit 2 ;;
esac

# Picasso's Lua sbatch wrapper prepends ANSI codes and a multi-line warning to
# --parsable output, so the id must be taken from the LAST line before any
# character stripping. A line-by-line sed leaves the newlines in place and the
# guard then fires *after* the job was already queued -- an untracked job on the
# cluster, which is worse than no guard.
_clean_job_id() {
    tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}

SBATCH_ARGS=(
    --parsable
    --export="ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},RESULTS_DIR=${RESULTS_DIR},N_EXPR=${N_EXPR},MAX_PERMS=${MAX_PERMS},PERM_TIMEOUT=${PERM_TIMEOUT},GLOBAL_SEED=${GLOBAL_SEED}"
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

RAW=$(sbatch "${SBATCH_ARGS[@]}") || { echo "sbatch failed" >&2; exit 1; }
JOB_ID=$(_clean_job_id "${RAW}")
[[ "${JOB_ID}" =~ ^[0-9]+$ ]] || {
    echo "FATAL: unparsable job id: ${RAW@Q}" >&2
    echo "A job may nonetheless have been queued -- run 'squeue' before resubmitting." >&2
    exit 1
}

echo "Submitted job ${JOB_ID}"
echo "Monitor:  ssh picasso 'squeue'"
echo "Log:      ${LOGS_DIR}/synth_retime_${JOB_ID}.out"
echo "Results:  ${RESULTS_DIR}  (27 synth_k*_m*.csv fragments + metadata)"
