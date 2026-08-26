#!/usr/bin/env bash
# =============================================================================
# SLURM compute worker: T15 three-arm CONST-normalization probe (ARRAY JOB)
# =============================================================================
# Each array task runs ONE (problem, seed) search for a given method and scores
# three CONST-normalization policies on the identical DAG stream:
#
#   submitted  old aggressive relocation of every CONST in-edge onto node 0
#   repair     new minimal repair of CONST nodes with in-degree 0  (production)
#   none       no CONST handling at all
#
# Exists because UDFS checks its own max_time only between order-enumeration
# stages and so overruns a short budget by orders of magnitude; the probe is not
# locally practical for UDFS. Bingo is fast enough to run on the workstation.
#
# Environment variables (exported by t15_norm_arms_launch.sh):
#   ISALSR_REPO_DIR    - Path to IsalSR repository
#   T15_METHOD         - SR method name ("udfs" or "bingo")
#   T15_MAX_TIME       - Per-run search budget in seconds
#   T15_N_SEEDS        - Number of seeds (for task ID decoding)
#   T15_RESULTS_DIR    - Base results directory
#
# Task ID encoding: task_id = problem_index * n_seeds + seed_index + 1
# (both 0-indexed, task_id is 1-indexed)
#
set -euo pipefail

echo "=== T15 normalization arms: SLURM Array Task ==="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Array ID:  ${SLURM_ARRAY_TASK_ID:-1}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "Method:    ${T15_METHOD}"

# Load MPI 5.0.9 module (required by bingo-nasa via mpi4py on Picasso).
for mod in openmpi_gcc/5.0.9_gcc7 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/5.0.9_gcc14; do
    module load "$mod" 2>/dev/null && break
done

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)/envs/isalsr}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Bypass CPython's pymalloc arena allocator (see models_experiment_slurm.sh).
export PYTHONMALLOC=malloc

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
cd "$REPO_DIR"

PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"

# The probe reaches the canonical core through the C++ extension's testing
# submodule. Without it there is nothing to measure, so fail loudly and early
# rather than silently producing an empty result.
$PYTHON -c "
from isalsr.core import _native
assert hasattr(_native.testing, 'fast_canonical_string_raw'), (
    'C++ extension predates the T15 raw entry point; rebuild it on Picasso'
)
print('[check] native extension OK')
"

# One problem per benchmark tier — must match DEFAULT_PROBLEMS in
# experiments/scripts/measure_const_normalization_arms.py.
PROBLEMS=(
    "nguyen:Nguyen-5"
    "feynman:I.6.20a"
    "hard:Pagie-1"
    "cherrypicked:Keijzer-11"
    "roundoff:R1"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
N_SEEDS="${T15_N_SEEDS:-3}"

PROBLEM_IDX=$(( (TASK_ID - 1) / N_SEEDS ))
SEED=$(( (TASK_ID - 1) % N_SEEDS + 1 ))

if (( PROBLEM_IDX >= ${#PROBLEMS[@]} )); then
    echo "[FATAL] problem index ${PROBLEM_IDX} >= ${#PROBLEMS[@]}" >&2
    exit 1
fi

PROBLEM_SPEC="${PROBLEMS[$PROBLEM_IDX]}"
RESULTS_DIR="${T15_RESULTS_DIR:?ERROR: T15_RESULTS_DIR not set}"
OUT_DIR="${RESULTS_DIR}/${T15_METHOD}/${PROBLEM_SPEC//:/_}_seed${SEED}"

echo "Problem:   ${PROBLEM_SPEC} (index ${PROBLEM_IDX})"
echo "Seed:      ${SEED}"
echo "Max time:  ${T15_MAX_TIME}s"
echo "Output:    ${OUT_DIR}"
echo ""

$PYTHON -m experiments.scripts.measure_const_normalization_arms \
    --methods "${T15_METHOD}" \
    --seeds "${SEED}" \
    --problems "${PROBLEM_SPEC}" \
    --max-time "${T15_MAX_TIME}" \
    --workers 1 \
    --out "${OUT_DIR}"

echo ""
echo "=== Task ${TASK_ID} (${PROBLEM_SPEC}, seed=${SEED}) complete: $(date) ==="
