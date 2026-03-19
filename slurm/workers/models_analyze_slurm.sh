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
module load openmpi 2>/dev/null || module load mpi 2>/dev/null || true

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

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
