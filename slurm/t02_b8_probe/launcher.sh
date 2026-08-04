#!/usr/bin/env bash
# =============================================================================
# B8 probe launcher -- submits ONE task (EXECUTION-PLAN §4.2 B8, SP-0 capped)
# =============================================================================
# Usage:  bash launcher.sh <phase-label>
#
# Submits a single 1-task job that runs one Nguyen-1 / seed 0 / baseline cell.
# Called three times by drive.sh to observe run -> skip -> delete+re-run.
#
# SP-0 caps honoured here and re-asserted in worker.sh:
#   tasks per submission  1   (cap 60)
#   max_time              300 s (cap 1800)
#   seeds                 0 only
#   output root           ~/execs/isalsr/t02_b8_probe/   (never a campaign root)
# =============================================================================
set -euo pipefail

PHASE="${1:?usage: launcher.sh <phase-label>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ISALSR_REPO_DIR="${ISALSR_REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR}"
PROBE_ROOT="${B8_PROBE_ROOT:-/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/t02_b8_probe}"
RESULTS_DIR="${PROBE_ROOT}/results"
LOGS_DIR="${PROBE_ROOT}/logs"
CONFIG="${ISALSR_REPO_DIR}/experiments/configs/bingo_nguyen.yaml"
PROBLEM="Nguyen-1"
MAX_TIME=300
WALL="0-00:20:00"
MEM=16
CONSTRAINT="${B8_CONSTRAINT:-sr}"
ACCOUNT="tic_163_uma"

mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}"

JOB_NAME="b8probe_${PHASE}"

sbatch --parsable \
    --array=1-1 \
    --job-name="${JOB_NAME}" \
    --time="${WALL}" \
    --ntasks=1 --cpus-per-task=1 \
    --mem="${MEM}G" \
    --constraint="${CONSTRAINT}" \
    --account="${ACCOUNT}" \
    --output="${LOGS_DIR}/${JOB_NAME}_%A_%a.out" \
    --error="${LOGS_DIR}/${JOB_NAME}_%A_%a.err" \
    --export="ALL,ISALSR_REPO_DIR=${ISALSR_REPO_DIR},B8_CONFIG=${CONFIG},B8_PROBLEM=${PROBLEM},B8_MAX_TIME=${MAX_TIME},B8_RESULTS_DIR=${RESULTS_DIR},B8_PHASE=${PHASE}" \
    "${SCRIPT_DIR}/worker.sh"
