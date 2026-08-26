#!/usr/bin/env bash
# =============================================================================
# SLURM compute worker: model-based experiment analysis
# =============================================================================
# Runs AFTER all experiment array jobs complete (via --dependency=afterok).
# Produces benchmark summaries, cross-method Friedman/Nemenyi, and global
# summary JSON.
#
# Environment variables (exported by models_launch.sh):
#   ISALSR_REPO_DIR   - Path to IsalSR repository
#   MODELS_RESULTS_DIR - Base results directory
#   MODELS_METHODS     - Comma-separated method names
#   MODELS_BENCHMARKS  - Comma-separated benchmark names
#
set -euo pipefail

echo "=== Models Analysis: SLURM Worker ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

# Load MPI module (required by bingo-nasa via mpi4py on Picasso)
for mod in openmpi_gcc/5.0.2_gcc7.5.0 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/4.1.5_gcc9.5.0_2024; do
    module load "$mod" 2>/dev/null && break
done

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

# Ensure conda lib is in LD_LIBRARY_PATH (for conda-installed openmpi)
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)/envs/isalsr}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
cd "$REPO_DIR"

PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"
RESULTS_DIR="${MODELS_RESULTS_DIR:?ERROR: MODELS_RESULTS_DIR not set}"
METHODS="${MODELS_METHODS:-udfs,bingo}"
BENCHMARKS="${MODELS_BENCHMARKS:-nguyen,feynman}"

echo "Results: ${RESULTS_DIR}"
echo "Methods: ${METHODS}"
echo "Benchmarks: ${BENCHMARKS}"
echo ""

$PYTHON -m experiments.models.analyze \
    --results-dir "${RESULTS_DIR}" \
    --methods "${METHODS}" \
    --benchmarks "${BENCHMARKS}"

echo ""
echo "=== Analysis complete: $(date) ==="
