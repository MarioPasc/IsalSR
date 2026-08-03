#!/usr/bin/env bash
# =============================================================================
# Campaign C2 pre-flight -- Stage C worker (EXECUTION-PLAN.md §4.3, ticket T17)
# =============================================================================
# One array task = one (problem, seed) pair for a fixed (method, arm, suite).
#
# NO #SBATCH directives here on purpose: the launcher supplies every resource
# flag on the sbatch command line, because the 42 arrays differ in size, memory
# and job name.  (Same pattern as slurm/t04_probe and slurm/t05_probe.)
#
# Environment variables (exported by launcher.sh):
#   ISALSR_REPO_DIR   - repo checkout on Picasso
#   C2_METHOD         - "udfs" | "bingo"
#   C2_ARM            - "baseline" | "hash" | "isalsr"
#   C2_SUITE          - benchmark suite key (nguyen, ..., strogatz)
#   C2_CONFIG         - absolute path to the YAML config
#   C2_SEEDS          - comma-separated seed list, e.g. "0,101,102"
#   C2_MAX_TIME       - per-run payload budget in seconds (900 for Stage C)
#   C2_RESULTS_DIR    - output root (c2_smoke/)
#
# Task id -> (problem, seed) decoding is delegated to
# experiments/scripts/c2_task_spec.py so the benchmark registry lives in exactly
# one place.  The previous bash worker hard-coded a five-suite table and would
# have silently refused strogatz and feynman_remainder -- 20 of the 70 problems.
# =============================================================================
set -euo pipefail

START_TIME=$(date +%s)

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
METHOD="${C2_METHOD:?ERROR: C2_METHOD not set}"
ARM="${C2_ARM:?ERROR: C2_ARM not set}"
SUITE="${C2_SUITE:?ERROR: C2_SUITE not set}"
CONFIG="${C2_CONFIG:?ERROR: C2_CONFIG not set}"
SEEDS="${C2_SEEDS:?ERROR: C2_SEEDS not set}"
MAX_TIME="${C2_MAX_TIME:?ERROR: C2_MAX_TIME not set}"
RESULTS_DIR="${C2_RESULTS_DIR:?ERROR: C2_RESULTS_DIR not set}"

# ---------------------------------------------------------------------------
# Environment.  Extended from slurm/workers/models_experiment_slurm.sh -- every
# line below was added because something failed without it; do not prune.
# ---------------------------------------------------------------------------

# bingo-nasa imports mpi4py, whose ABI probe dlopen()s libmpi at IMPORT time.
# A single-process job that never touches MPI still dies in ~13 s without this.
for mod in openmpi_gcc/5.0.9_gcc7 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/5.0.9_gcc14; do
    module load "$mod" 2>/dev/null && break
done

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)/envs/isalsr}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Bypass CPython's pymalloc arena allocator.  pymalloc fragments the heap over
# 10k+ generations (256 KB arenas pinned by surviving objects); glibc malloc
# mmaps large allocations and malloc_trim(0) reclaims them.  This is what keeps
# Bingo-IsalSR off the OOM ceiling.
export PYTHONMALLOC=malloc

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# One core per task: keep BLAS from oversubscribing what we did not reserve.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="$(command -v python3)"

# ---------------------------------------------------------------------------
# Decode the array index.  Echo the decoded tuple: a silently wrong decode
# produces a complete, plausible, WRONG result set.
# ---------------------------------------------------------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

# The launcher ships the seed list COLON-separated because sbatch --export is
# comma-separated and would otherwise truncate "0,101,102" to "0". Accept either
# form, then assert the count matches what the array was sized for -- a silently
# short seed list is the exact failure that made 2/3 of the first wave die with
# "index out of range" while the other 1/3 produced correct-looking cells.
SEEDS="${SEEDS//:/,}"
N_SEEDS_DECODED=$(awk -F, '{print NF}' <<<"${SEEDS}")
if [[ "${N_SEEDS_DECODED}" -lt 2 ]]; then
    echo "[FATAL] C2_SEEDS decoded to ${N_SEEDS_DECODED} seed(s): '${SEEDS}'." >&2
    echo "        Stage C requires three. Check the launcher's --export quoting." >&2
    exit 1
fi

SPEC="$("${PYTHON}" -m experiments.scripts.c2_task_spec \
            --config "${CONFIG}" --seeds "${SEEDS}" --index "${TASK_ID}")"
PROBLEM_NAME="$(awk '{print $1}' <<<"${SPEC}")"
SEED="$(awk '{print $2}' <<<"${SPEC}")"
[[ -n "${PROBLEM_NAME}" && -n "${SEED}" ]] \
    || { echo "[FATAL] empty decode for task ${TASK_ID}: '${SPEC}'" >&2; exit 1; }

# ---------------------------------------------------------------------------
# SP-1..SP-3 provenance header.  Recorded per task, from the COMPUTE NODE, so a
# stale checkout or a stale .so is visible in the log of every run rather than
# inferred afterwards.
# ---------------------------------------------------------------------------
echo "=========================================="
echo "C2 Stage C | job ${SLURM_JOB_ID:-local} task ${TASK_ID}"
echo "Node:        $(hostname)  ($(lscpu | sed -n 's/^Model name: *//p' | head -1))"
echo "Start:       $(date)"
echo "Method/Arm:  ${METHOD} / ${ARM}"
echo "Suite:       ${SUITE}"
echo "Problem:     ${PROBLEM_NAME}"
echo "Seed:        ${SEED}"
echo "Config:      ${CONFIG}"
echo "Results:     ${RESULTS_DIR}"
# SP-1 from the compute node: report HEAD, which is the thing that matters and
# is a pure read.  Deliberately NOT `--dirty`: rsync rewrites mtimes, so git's
# index looks stale and `describe --dirty` appends "-dirty" on content that is
# byte-identical to HEAD.  Clearing that would need `git update-index --refresh`,
# which WRITES .git/index -- and 1,260 concurrent array tasks writing one shared
# index is a far worse problem than the cosmetic suffix.  Tree cleanliness is
# asserted once, from the login node, by deploy.sh before submission.
echo "SP-1 commit: $(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo n/a)"
echo "SP-1 tag:    $(git -C "${REPO_DIR}" describe --tags --always 2>/dev/null || echo n/a)"
"${PYTHON}" - <<'PROV'
import datetime, os, sys
try:
    import isalsr
    from isalsr.core import _native, backends
    print(f"SP-2 pkg:    {isalsr.__file__}")
    print(f"SP-2 native: {_native.__file__}")
    print(f"SP-2 mtime:  {datetime.datetime.fromtimestamp(os.path.getmtime(_native.__file__))}")
    print(f"SP-3 engine: {backends.engine()}")
    print(f"SP-3 build:  {backends.build_info()}")
except Exception as exc:  # a broken engine must be loud, not inferred later
    print(f"[FATAL] engine probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
PROV
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# Payload.
#
#   --ledger          T17 §2.1.  ISALSR_LEDGER_ENABLED defaults to "0" and is
#                     set in no config; without this flag all 1,260 runs record
#                     five reachability rates of zero, which reads as "no
#                     fallbacks occurred" and means "nothing was counted".
#                     Unrecoverable after the fact (SP-6).
#   --postprocess skip
#                     Aggregates, the three paired contrasts and the status
#                     ledger are a campaign-level step.  Left on, 1,260 tasks
#                     would each walk the whole output tree and write the same
#                     three files concurrently.  aggregate_worker.sh does it
#                     once, afterwards.
#   --max-time 900    Stage C budget.  Never the production 43,200 s.
# ---------------------------------------------------------------------------
set +e
"${PYTHON}" -m experiments.models.orchestrator \
    --config "${CONFIG}" \
    --output-dir "${RESULTS_DIR}" \
    --seeds "${SEED}" \
    --problems "${PROBLEM_NAME}" \
    --variants "${ARM}" \
    --max-time "${MAX_TIME}" \
    --ledger \
    --postprocess skip
RC=$?
set -e

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "=== Task ${TASK_ID} (${METHOD}/${ARM}/${SUITE}/${PROBLEM_NAME}/seed=${SEED}) rc=${RC} ==="
exit ${RC}
