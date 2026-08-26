#!/usr/bin/env bash
# Submit the T16 alphabet certification gate (G9) to Picasso.
#
# Usage:
#   bash slurm/alphabet_gate/launcher.sh              # submit
#   bash slurm/alphabet_gate/launcher.sh --dry-run    # print the sbatch command
#   bash slurm/alphabet_gate/launcher.sh --test-only  # sbatch --test-only probe
#
# CPU-only job: no --gres, no GPU constraint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurable ----------------------------------------------------------
export REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR"
export CONDA_PREFIX_ABS="/mnt/home/users/tic_163_uma/mpascual/fscratch/conda_envs/isalsr"
export WORK_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/g9_alphabet_gate"
LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/logs"

# Both hosts, with and without `pow` in the operator set. bingo_hard is the one
# that proves Pow SURVIVES; the others prove nothing order-sensitive appears.
export GATE_CONFIGS="bingo_nguyen bingo_hard udfs_nguyen udfs_hard"
export GATE_PAIRS="bingo_nguyen:Nguyen-1 bingo_hard:Pagie-1 udfs_nguyen:Nguyen-1 udfs_hard:Keijzer-6"
export GATE_MAX_TIME="15"

mkdir -p "${LOGS_DIR}"

MODE="submit"
case "${1:-}" in
    --dry-run)   MODE="dry" ;;
    --test-only) MODE="test" ;;
esac

# Picasso's Lua sbatch wrapper prepends ANSI codes and a multi-line warning to
# --parsable output. Take the LAST line first, then strip.
_clean_job_id() {
    tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}

SBATCH_ARGS=(
    --job-name=isalsr-g9
    --time=0-00:20:00
    --ntasks=1
    --cpus-per-task=4
    --mem=16G
    --constraint=cpu
    --account=tic_163_uma
    --output="${LOGS_DIR}/g9_%j.out"
    --error="${LOGS_DIR}/g9_%j.err"
    --export="ALL,REPO_DIR=${REPO_DIR},CONDA_PREFIX_ABS=${CONDA_PREFIX_ABS},WORK_DIR=${WORK_DIR},GATE_CONFIGS=${GATE_CONFIGS},GATE_PAIRS=${GATE_PAIRS},GATE_MAX_TIME=${GATE_MAX_TIME}"
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

RAW=$(sbatch --parsable "${SBATCH_ARGS[@]}")
JOB_ID=$(_clean_job_id "${RAW}")
if [[ ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "FATAL: unparsable job id from sbatch: ${RAW@Q}" >&2
    echo "A job MAY have been submitted -- run: squeue -u \$USER" >&2
    exit 1
fi

echo "Submitted G9 gate job ${JOB_ID}"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Log:      ${LOGS_DIR}/g9_${JOB_ID}.out"
