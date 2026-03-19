#!/usr/bin/env bash
# =============================================================================
# SLURM debug worker: fast sequential test of all (method, benchmark) combos
# =============================================================================
# Uses lightweight debug configs (60s max per problem, small populations).
# Runs ONE seed, ONE problem per (method, benchmark, variant) = 8 total runs.
# Expected total runtime: ~10-15 minutes.
#
# Usage:
#   bash slurm/models_launch.sh --experiment models_debug
#
set -euo pipefail

echo "=== Models Debug: Sequential Pipeline Test ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo ""

# ---- MPI setup ----
# bingo-nasa requires mpi4py which needs libmpi.so
# Try Picasso system modules first, then fall back to conda-bundled MPI.
echo "--- MPI Setup ---"
MPI_LOADED=false

# Try Picasso openmpi modules (most compatible with gcc-based conda envs)
for mod in openmpi_gcc/5.0.2_gcc7.5.0 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/4.1.5_gcc9.5.0_2024; do
    if module load "$mod" 2>/dev/null; then
        echo "  Loaded module: $mod"
        MPI_LOADED=true
        break
    fi
done

if [[ "$MPI_LOADED" == "false" ]]; then
    echo "  WARN: No system MPI module loaded."
    echo "  Trying conda-bundled MPI..."
fi

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

# Ensure conda lib is in LD_LIBRARY_PATH (for conda-installed openmpi)
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)/envs/isalsr}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Verify MPI is loadable
PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"
echo "  Testing mpi4py import..."
if $PYTHON -c "from mpi4py import MPI; print(f'  mpi4py OK: {MPI.Get_library_version().split(chr(10))[0]}')" 2>/dev/null; then
    echo "  MPI: READY"
else
    echo "  ERROR: mpi4py cannot load MPI library."
    echo "  Available in conda lib:"
    ls "${CONDA_PREFIX}/lib"/libmpi* 2>/dev/null || echo "    (none)"
    echo ""
    echo "  Fix options:"
    echo "    1. conda install -c conda-forge openmpi mpi4py"
    echo "    2. Load a Picasso module: module load openmpi_gcc/5.0.2_gcc7.5.0"
    echo ""
    echo "  Continuing anyway (UDFS tests will work, Bingo tests will fail)..."
fi
echo ""

# ---- Setup ----
REPO_DIR="${ISALSR_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_DIR"

RESULTS_DIR="${MODELS_RESULTS_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/models/debug}"
mkdir -p "$RESULTS_DIR"

echo "Repo:    ${REPO_DIR}"
echo "Results: ${RESULTS_DIR}"
echo ""

# ---- Test matrix ----
# Uses debug configs: max_time=60s, small populations, 1 seed
# Each entry: method:benchmark:config:problem
CONFIGS=(
    "udfs:nguyen:experiments/configs/debug_udfs_nguyen.yaml:Nguyen-1"
    "udfs:feynman:experiments/configs/debug_udfs_feynman.yaml:I.6.20a"
    "bingo:nguyen:experiments/configs/debug_bingo_nguyen.yaml:Nguyen-1"
    "bingo:feynman:experiments/configs/debug_bingo_feynman.yaml:I.6.20a"
)

SEED=1
PASSED=0
FAILED=0
SKIPPED=0

for entry in "${CONFIGS[@]}"; do
    IFS=':' read -r method benchmark config problem <<< "$entry"

    for variant in baseline isalsr; do
        echo "=============================================="
        echo "Testing: ${method} / ${benchmark} / ${variant}"
        echo "  Problem: ${problem}, Seed: ${SEED}"
        echo "  Config:  ${config}"
        echo "----------------------------------------------"

        START_TIME=$(date +%s)

        if $PYTHON -m experiments.models.orchestrator \
            --config "${config}" \
            --output-dir "${RESULTS_DIR}" \
            --seeds "${SEED}" \
            --problems "${problem}" \
            --variants "${variant}" 2>&1; then
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "  -> PASSED (${ELAPSED}s)"
            PASSED=$((PASSED + 1))
        else
            EXIT_CODE=$?
            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))
            echo "  -> FAILED (exit code ${EXIT_CODE}, ${ELAPSED}s)"
            FAILED=$((FAILED + 1))
        fi
        echo ""
    done
done

echo "=============================================="
echo "Debug Summary"
echo "=============================================="
echo "Passed:  ${PASSED} / $((PASSED + FAILED + SKIPPED))"
echo "Failed:  ${FAILED}"
echo "Skipped: ${SKIPPED}"
echo ""

# Show results tree
echo "Results:"
find "${RESULTS_DIR}" -name "run_log.json" -type f | sort | while read -r f; do
    echo "  $f"
done
echo ""

if [[ $FAILED -gt 0 ]]; then
    echo "SOME TESTS FAILED. Review output above."
    exit 1
else
    echo "ALL TESTS PASSED."
fi

echo ""
echo "=== Debug complete: $(date) ==="
