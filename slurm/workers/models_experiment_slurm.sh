#!/usr/bin/env bash
# =============================================================================
# SLURM compute worker: model-based experiment (ARRAY JOB)
# =============================================================================
# Each array task runs ONE (problem, seed) pair for a given
# (method, benchmark, variant).
#
# Environment variables (exported by models_launch.sh):
#   ISALSR_REPO_DIR          - Path to IsalSR repository
#   MODELS_METHOD            - SR method name (e.g., "udfs", "bingo")
#   MODELS_BENCHMARK         - Benchmark name (e.g., "nguyen", "feynman")
#   MODELS_VARIANT           - Variant ("baseline" or "isalsr")
#   MODELS_EXPERIMENT_CONFIG - Path to YAML config file
#   MODELS_N_SEEDS           - Number of seeds (for task ID decoding)
#   MODELS_RESULTS_DIR       - Base results directory
#
# Task ID encoding: task_id = problem_index * n_seeds + seed_index + 1
# (both 0-indexed, task_id is 1-indexed)
#
set -euo pipefail

echo "=== Models Experiment: SLURM Array Task ==="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Array ID:  ${SLURM_ARRAY_TASK_ID:-1}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "Method:    ${MODELS_METHOD}"
echo "Benchmark: ${MODELS_BENCHMARK}"
echo "Variant:   ${MODELS_VARIANT}"

# Load MPI 5.0.9 module (required by bingo-nasa via mpi4py on Picasso)
# Error "Please use mpi 5.0.9" occurs if wrong version loaded.
for mod in openmpi_gcc/5.0.9_gcc7 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/5.0.9_gcc14; do
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

# Decode array task ID -> (problem_index, seed)
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
N_SEEDS="${MODELS_N_SEEDS:-30}"

PROBLEM_IDX=$(( (TASK_ID - 1) / N_SEEDS ))
SEED=$(( (TASK_ID - 1) % N_SEEDS + 1 ))

# Resolve problem name from benchmark definitions
PROBLEM_NAME=$($PYTHON -c "
import sys
sys.path.insert(0, '.')
benchmark = '${MODELS_BENCHMARK}'
if benchmark == 'nguyen':
    from benchmarks.datasets.nguyen import NGUYEN_BENCHMARKS as B
elif benchmark == 'feynman':
    from benchmarks.datasets.feynman import FEYNMAN_BENCHMARKS as B
else:
    print(f'ERROR: Unknown benchmark: {benchmark}', file=sys.stderr)
    sys.exit(1)
idx = ${PROBLEM_IDX}
if idx >= len(B):
    print(f'ERROR: Problem index {idx} >= {len(B)}', file=sys.stderr)
    sys.exit(1)
print(B[idx]['name'])
")

echo "Problem:   ${PROBLEM_NAME} (index ${PROBLEM_IDX})"
echo "Seed:      ${SEED}"
echo ""

CONFIG="${MODELS_EXPERIMENT_CONFIG:?ERROR: MODELS_EXPERIMENT_CONFIG not set}"
RESULTS_DIR="${MODELS_RESULTS_DIR:?ERROR: MODELS_RESULTS_DIR not set}"

$PYTHON -m experiments.models.orchestrator \
    --config "${CONFIG}" \
    --output-dir "${RESULTS_DIR}" \
    --seeds "${SEED}" \
    --problems "${PROBLEM_NAME}" \
    --variants "${MODELS_VARIANT}"

echo ""
echo "=== Task ${TASK_ID} (${PROBLEM_NAME}, seed=${SEED}, ${MODELS_VARIANT}) complete: $(date) ==="
