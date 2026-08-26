#!/usr/bin/env bash
# =============================================================================
# B8 probe worker -- resume and idempotency on Picasso (EXECUTION-PLAN §4.2 B8)
# =============================================================================
# One array task = one (problem, seed) cell, run three times across three
# separate submissions to observe: run -> skip -> corrupt -> delete+re-run.
#
# Modelled on slurm/c2_smoke/worker.sh, with two deliberate differences, both
# required by the SP-0 probe caps:
#
#   * SEED is fixed to 0.  c2_smoke/worker.sh refuses a single-seed list (it
#     asserts >= 2 to catch the --export comma-truncation bug), but SP-0 allows
#     probes seed 0 ONLY, so the array decode is replaced by a fixed cell.
#   * Output goes to ~/execs/isalsr/t02_b8_probe/, never a campaign root.
#
# This probe answers "does resume work on Picasso?".  It produces no number for
# the paper.
#
# Environment variables (exported by launcher.sh):
#   ISALSR_REPO_DIR   - repo checkout on Picasso
#   B8_CONFIG         - absolute path to the YAML config
#   B8_PROBLEM        - problem name
#   B8_MAX_TIME       - payload budget in seconds (<= 1800 per SP-0)
#   B8_RESULTS_DIR    - probe output root
#   B8_PHASE          - free-text label for the log header
# =============================================================================
set -euo pipefail

START_TIME=$(date +%s)

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
CONFIG="${B8_CONFIG:?ERROR: B8_CONFIG not set}"
PROBLEM_NAME="${B8_PROBLEM:?ERROR: B8_PROBLEM not set}"
MAX_TIME="${B8_MAX_TIME:?ERROR: B8_MAX_TIME not set}"
RESULTS_DIR="${B8_RESULTS_DIR:?ERROR: B8_RESULTS_DIR not set}"
PHASE="${B8_PHASE:-unlabelled}"

# SP-0: probes run seed 0 and nothing else, so a probe cell can never be
# mistaken for a campaign cell.
SEED=0
METHOD=bingo
ARM=baseline

# SP-0 guard, asserted rather than trusted: a probe that quietly grew a
# production budget is the failure this cap exists to prevent.
if (( MAX_TIME > 1800 )); then
    echo "[FATAL] B8_MAX_TIME=${MAX_TIME} exceeds the SP-0 cap of 1800 s." >&2
    exit 1
fi
case "${RESULTS_DIR}" in
    *"/execs/isalsr/t02_b8_probe"*) : ;;
    *) echo "[FATAL] B8_RESULTS_DIR must live under ~/execs/isalsr/t02_b8_probe/" >&2
       exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Environment.  Copied from slurm/c2_smoke/worker.sh; every line was added
# because something failed without it.
# ---------------------------------------------------------------------------

# bingo-nasa imports mpi4py, whose ABI probe dlopen()s libmpi at IMPORT time.
for mod in openmpi_gcc/5.0.9_gcc7 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/5.0.9_gcc14; do
    module load "$mod" 2>/dev/null && break
done

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)/envs/isalsr}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONMALLOC=malloc

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="$(command -v python3)"

RUN_LOG="${RESULTS_DIR}/bingo/nguyen/nguyen_1/${ARM}/seed_$(printf '%02d' ${SEED})/run_log.json"

echo "=========================================="
echo "B8 probe | phase ${PHASE} | job ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)  ($(lscpu | sed -n 's/^Model name: *//p' | head -1))"
echo "Start:       $(date)"
echo "Problem:     ${PROBLEM_NAME}  seed ${SEED}  arm ${ARM}"
echo "Results:     ${RESULTS_DIR}"
echo "SP-1 commit: $(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo n/a)"
if [[ -f "${RUN_LOG}" ]]; then
    echo "PRE  run_log: EXISTS, $(stat -c %s "${RUN_LOG}") bytes"
    "${PYTHON}" -c "import json,sys; json.load(open('${RUN_LOG}')); print('PRE  run_log: parses as JSON')" \
        2>/dev/null || echo "PRE  run_log: DOES NOT PARSE (corrupt)"
else
    echo "PRE  run_log: ABSENT"
fi
echo "=========================================="
echo ""

set +e
"${PYTHON}" -m experiments.models.orchestrator \
    --config "${CONFIG}" \
    --output-dir "${RESULTS_DIR}" \
    --seeds "${SEED}" \
    --problems "${PROBLEM_NAME}" \
    --variants "${ARM}" \
    --max-time "${MAX_TIME}" \
    --postprocess skip
RC=$?
set -e

echo ""
if [[ -f "${RUN_LOG}" ]]; then
    echo "POST run_log: $(stat -c %s "${RUN_LOG}") bytes"
    "${PYTHON}" -c "import json; json.load(open('${RUN_LOG}')); print('POST run_log: parses as JSON')"
else
    echo "POST run_log: ABSENT"
fi

END_TIME=$(date +%s)
echo "Duration:  $(( (END_TIME - START_TIME) / 60 ))m $(( (END_TIME - START_TIME) % 60 ))s"
echo "=== B8 probe phase ${PHASE} rc=${RC} ==="
exit ${RC}
