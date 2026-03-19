#!/usr/bin/env bash
# =============================================================================
# SLURM debug worker: sequential test of all (method, benchmark) combinations
# =============================================================================
# Runs ONE seed, ONE problem per (method, benchmark, variant) to verify
# the full pipeline works before launching thousands of array tasks.
#
# NOT an array job — runs everything sequentially in a single task.
#
# Usage (via models_launch.sh):
#   bash slurm/models_launch.sh --experiment models_debug
#
# Or directly:
#   sbatch slurm/workers/models_debug_slurm.sh
#
set -euo pipefail

echo "=== Models Debug: Sequential Pipeline Test ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

# Load MPI module (required by bingo-nasa via mpi4py)
module load openmpi 2>/dev/null || module load mpi 2>/dev/null || echo "WARN: No MPI module found"
echo "MPI library check:"
ldconfig -p 2>/dev/null | grep libmpi || echo "  libmpi not in ldconfig"
ls /usr/lib64/libmpi* 2>/dev/null || ls /usr/lib/libmpi* 2>/dev/null || echo "  libmpi not in /usr/lib"

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"

REPO_DIR="${ISALSR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"

RESULTS_DIR="${MODELS_RESULTS_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/models_debug}"
mkdir -p "$RESULTS_DIR"

echo "Repo:    ${REPO_DIR}"
echo "Results: ${RESULTS_DIR}"
echo ""

# Test matrix: 4 combinations (method x benchmark), seed=1, first problem only
# Each runs both baseline + isalsr variants
CONFIGS=(
    "udfs:nguyen:experiments/configs/udfs_nguyen.yaml:Nguyen-1"
    "bingo:nguyen:experiments/configs/bingo_nguyen.yaml:Nguyen-1"
    "udfs:feynman:experiments/configs/udfs_feynman.yaml:I.6.20a"
    "bingo:feynman:experiments/configs/bingo_feynman.yaml:I.6.20a"
)

SEED=1
FAILED=0

for entry in "${CONFIGS[@]}"; do
    IFS=':' read -r method benchmark config problem <<< "$entry"

    for variant in baseline isalsr; do
        echo "=============================================="
        echo "Testing: ${method} / ${benchmark} / ${variant}"
        echo "  Problem: ${problem}, Seed: ${SEED}"
        echo "  Config:  ${config}"
        echo "=============================================="

        if $PYTHON -m experiments.models.orchestrator \
            --config "${config}" \
            --output-dir "${RESULTS_DIR}" \
            --seeds "${SEED}" \
            --problems "${problem}" \
            --variants "${variant}"; then
            echo "  -> OK"
        else
            echo "  -> FAILED (exit code $?)"
            FAILED=$((FAILED + 1))
        fi
        echo ""
    done
done

echo "=============================================="
echo "Debug Summary"
echo "=============================================="
echo "Total tests: $((${#CONFIGS[@]} * 2))"
echo "Failed:      ${FAILED}"
echo ""

# Show results tree
echo "Results tree:"
find "${RESULTS_DIR}" -name "run_log.json" -type f | sort
echo ""

if [[ $FAILED -gt 0 ]]; then
    echo "SOME TESTS FAILED. Check output above."
    exit 1
else
    echo "ALL TESTS PASSED."
fi

echo ""
echo "=== Debug complete: $(date) ==="
