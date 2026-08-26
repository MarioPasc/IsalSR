#!/usr/bin/env bash
# =============================================================================
# SLURM compute worker: T07 norm-removal study (ARRAY JOB)
# =============================================================================
#
# Each task runs ONE canonicalisation-population slice and writes a
# results.json to the appropriate output subdirectory.
#
# Population routing (via T07_POPULATION):
#   synthetic   one array task = one chunk of N_PER_TASK random strings
#   adversarial single task    = the fixture + variants
#   bingo       one array task = one (problem, seed) live Bingo run
#   udfs        one array task = one (problem, seed) live UDFS run
#
# Environment variables exported by t07_norm_removal_launch.sh:
#   T07_POPULATION          synthetic | adversarial | bingo | udfs
#   T07_REPO_DIR            Repository root on Picasso
#   T07_RESULTS_ROOT        Base results directory
#   T07_N_SEEDS             Number of seeds (for task-ID decoding)
#   T07_N_SYNTHETIC_PER_TASK  Strings per synthetic task
#   T07_BINGO_PROBLEMS      comma-separated suite:problem list (bingo)
#   T07_BINGO_MAX_TIME      Search budget in seconds (bingo)
#   T07_UDFS_PROBLEMS       comma-separated suite:problem list (udfs)
#   T07_UDFS_MAX_TIME       Search budget in seconds (udfs)
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Job header
# ---------------------------------------------------------------------------
echo "=== T07 norm-removal study: SLURM Worker ==="
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node:       $(hostname)"
echo "Start:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Git commit: $(git -C "${T07_REPO_DIR:-.}" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Population: ${T07_POPULATION:-?}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Load OpenMPI (needed by bingo-nasa / mpi4py on Picasso)
for mod in openmpi_gcc/5.0.9_gcc7 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/5.0.9_gcc14; do
    module load "$mod" 2>/dev/null && break
done

# Conda activation: probe the standard install locations in order
_CONDA_FOUND=false
for _candidate in \
    "${HOME}/miniconda/3" \
    "${HOME}/miniconda3" \
    "${HOME}/Miniconda3" \
    "${HOME}/anaconda3" \
    "${HOME}/Anaconda3" \
    "${HOME}/miniforge" \
    "${HOME}/mambaforge"
do
    if [[ -f "${_candidate}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1090
        source "${_candidate}/etc/profile.d/conda.sh"
        conda activate isalsr
        _CONDA_FOUND=true
        break
    fi
done

if ! ${_CONDA_FOUND}; then
    # Fall back to eval-based activation if none of the paths matched
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate isalsr 2>/dev/null || true
fi

CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base 2>/dev/null)/envs/isalsr}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONMALLOC=malloc       # bypass CPython arena allocator to avoid OOM
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ---------------------------------------------------------------------------
# Repo and PYTHONPATH (load-bearing: editable install may resolve from cache)
# ---------------------------------------------------------------------------
REPO_DIR="${T07_REPO_DIR:?ERROR: T07_REPO_DIR not set}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"

# Resolve the interpreter from the ACTIVATED env. The previous fallback to a
# bare `python3` was a footgun: both conda-activation branches above end in
# `|| true`, so a failed activation would silently hand the run to the system
# interpreter, which has no isalsr installed. Fail loudly here instead.
PYTHON="$(command -v python || true)"
if [[ -z "${PYTHON}" || "${PYTHON}" != *"isalsr"* ]]; then
    echo "[FATAL] conda env 'isalsr' is not active (python resolved to '${PYTHON:-<none>}')." >&2
    echo "[FATAL] Refusing to run against the wrong interpreter." >&2
    exit 1
fi
echo "[check] interpreter: ${PYTHON}"

# Sanity-check: the C++ testing entry point must be present
"${PYTHON}" -c "
from isalsr.core import _native
assert hasattr(_native.testing, 'fast_canonical_string_raw'), (
    'C++ extension predates the T07 raw entry point — rebuild on Picasso'
)
print('[check] _native.testing.fast_canonical_string_raw  OK')
"

# ---------------------------------------------------------------------------
# Task-index decoding
# ---------------------------------------------------------------------------
TASK_IDX="${SLURM_ARRAY_TASK_ID:-0}"
N_SEEDS="${T07_N_SEEDS:-3}"
RESULTS_ROOT="${T07_RESULTS_ROOT:?ERROR: T07_RESULTS_ROOT not set}"
POPULATION="${T07_POPULATION:?ERROR: T07_POPULATION not set}"

case "${POPULATION}" in
# -----------------------------------------------------------------------
# synthetic
# -----------------------------------------------------------------------
synthetic)
    N_PER_TASK="${T07_N_SYNTHETIC_PER_TASK:-100000}"
    OUT_DIR="${RESULTS_ROOT}/synthetic/task${TASK_IDX}"
    mkdir -p "${OUT_DIR}"
    echo "Task ${TASK_IDX}: synthetic  n=${N_PER_TASK}  out=${OUT_DIR}"
    "${PYTHON}" -m experiments.scripts.t07_norm_removal_study \
        --population synthetic \
        --n "${N_PER_TASK}" \
        --seed 31 \
        --task-id "${TASK_IDX}" \
        --timeout 10.0 \
        --sample-rate 50 \
        --equivariance-k 8 \
        --out "${OUT_DIR}"
    ;;

# -----------------------------------------------------------------------
# adversarial (single task only — SLURM_ARRAY_TASK_ID unused)
# -----------------------------------------------------------------------
adversarial)
    OUT_DIR="${RESULTS_ROOT}/adversarial"
    mkdir -p "${OUT_DIR}"
    echo "Task: adversarial  out=${OUT_DIR}"
    "${PYTHON}" -m experiments.scripts.t07_norm_removal_study \
        --population adversarial \
        --seed 31 \
        --timeout 10.0 \
        --sample-rate 1 \
        --equivariance-k 8 \
        --out "${OUT_DIR}"
    ;;

# -----------------------------------------------------------------------
# bingo / udfs (shared task-index decoding logic)
# -----------------------------------------------------------------------
bingo|udfs)
    if [[ "${POPULATION}" == "bingo" ]]; then
        PROBLEMS_CSV="${T07_BINGO_PROBLEMS:?ERROR: T07_BINGO_PROBLEMS not set}"
        MAX_TIME="${T07_BINGO_MAX_TIME:-21600}"
    else
        PROBLEMS_CSV="${T07_UDFS_PROBLEMS:?ERROR: T07_UDFS_PROBLEMS not set}"
        MAX_TIME="${T07_UDFS_MAX_TIME:-21600}"
    fi

    # Split the problem list into an array.
    # Separator is '|', NOT ','. SLURM's --export uses commas to separate
    # VARIABLES, so a comma-separated value is silently truncated at the first
    # comma and the remainder is parsed as (malformed) extra export entries.
    # That is exactly how job 1679359 lost 4 of its 5 problems and failed
    # every task with PROBLEM_IDX >= 1 in under 3 seconds.
    IFS='|' read -r -a PROBLEMS <<< "${PROBLEMS_CSV}"
    N_PROBLEMS="${#PROBLEMS[@]}"

    PROBLEM_IDX=$(( TASK_IDX / N_SEEDS ))
    SEED=$(( TASK_IDX % N_SEEDS + 1 ))

    if (( PROBLEM_IDX >= N_PROBLEMS )); then
        echo "[FATAL] problem index ${PROBLEM_IDX} >= ${N_PROBLEMS}" >&2
        exit 1
    fi

    PROBLEM_SPEC="${PROBLEMS[$PROBLEM_IDX]}"
    SUITE="${PROBLEM_SPEC%%:*}"
    PROBLEM="${PROBLEM_SPEC##*:}"

    OUT_DIR="${RESULTS_ROOT}/${POPULATION}/${SUITE}_${PROBLEM}_seed${SEED}"
    mkdir -p "${OUT_DIR}"
    echo "Task ${TASK_IDX}: ${POPULATION}  ${SUITE}/${PROBLEM}  seed=${SEED}  out=${OUT_DIR}"

    "${PYTHON}" -m experiments.scripts.t07_norm_removal_study \
        --population "${POPULATION}" \
        --method "${POPULATION}" \
        --suite "${SUITE}" \
        --problem "${PROBLEM}" \
        --seed "${SEED}" \
        --max-time "${MAX_TIME}" \
        --timeout 60.0 \
        --sample-rate 50 \
        --equivariance-k 8 \
        --out "${OUT_DIR}"
    ;;

*)
    echo "[FATAL] Unknown population: ${POPULATION}" >&2
    exit 1
    ;;
esac

echo ""
echo "=== Task ${TASK_IDX} (${POPULATION}) complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
