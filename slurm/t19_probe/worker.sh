#!/usr/bin/env bash
# =============================================================================
# T19 complexity-telemetry PROBE -- worker
# =============================================================================
# One array task = one (method, arm, problem, seed) cell, read from tasks.txt.
#
# NO #SBATCH directives here on purpose: the launcher supplies every resource
# flag on the sbatch command line.  Same pattern as slurm/c2_smoke,
# slurm/t04_probe and slurm/t05_probe.
#
# Derived from slurm/c2_smoke/worker.sh @ 2ff0050, which is the worker proven on
# this cluster.  Every environment line below was added there because something
# failed without it -- do not prune them.  The one deliberate divergence is the
# task decode: c2_smoke delegates to experiments/scripts/c2_task_spec.py, which
# is under active modification for the SCBI chunking request, so this probe
# reads a flat tasks.txt instead and depends on none of it.
#
# Environment variables (exported by launcher.sh):
#   ISALSR_REPO_DIR   - repo checkout on Picasso
#   T19_TASKS         - absolute path to tasks.txt
#   T19_RESULTS_DIR   - output root (NEVER the campaign root)
#   T19_MAX_TIME      - per-run payload budget in seconds (900; SP-0 cap 1800)
# =============================================================================
set -euo pipefail

START_TIME=$(date +%s)

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
TASKS="${T19_TASKS:?ERROR: T19_TASKS not set}"
RESULTS_DIR="${T19_RESULTS_DIR:?ERROR: T19_RESULTS_DIR not set}"
MAX_TIME="${T19_MAX_TIME:?ERROR: T19_MAX_TIME not set}"

# ---------------------------------------------------------------------------
# Environment.  Inherited verbatim from the C2 worker.
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

# Bypass CPython's pymalloc arena allocator: it fragments the heap over 10k+
# generations, and glibc malloc + malloc_trim(0) is what keeps Bingo-IsalSR off
# the OOM ceiling.
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
SPEC="$(grep -v '^[[:space:]]*#' "${TASKS}" | grep -v '^[[:space:]]*$' \
        | sed -n "${TASK_ID}p")"
[[ -n "${SPEC}" ]] || { echo "[FATAL] no task at index ${TASK_ID} in ${TASKS}" >&2; exit 1; }

METHOD="$(awk '{print $1}' <<<"${SPEC}")"
ARM="$(awk '{print $2}' <<<"${SPEC}")"
PROBLEM_NAME="$(awk '{print $3}' <<<"${SPEC}")"
SEED="$(awk '{print $4}' <<<"${SPEC}")"
[[ -n "${METHOD}" && -n "${ARM}" && -n "${PROBLEM_NAME}" && -n "${SEED}" ]] \
    || { echo "[FATAL] short decode for task ${TASK_ID}: '${SPEC}'" >&2; exit 1; }

# SP-0: a probe must never write a seed the campaign owns.
if [[ "${SEED}" -ge 1 && "${SEED}" -le 30 ]]; then
    echo "[FATAL] seed ${SEED} is inside the campaign range 1..30 (SP-0)." >&2
    exit 1
fi

CONFIG="${REPO_DIR}/experiments/configs/${METHOD}_nguyen.yaml"
[[ -f "${CONFIG}" ]] || { echo "[FATAL] missing config ${CONFIG}" >&2; exit 1; }

# ---------------------------------------------------------------------------
# SP-1..SP-3 provenance header, recorded per task FROM THE COMPUTE NODE, so a
# stale checkout or a stale .so is visible in the log of every run rather than
# inferred afterwards.
#
# Deliberately NOT `describe --dirty`: rsync rewrites mtimes, so git's index
# looks stale and would append "-dirty" to content byte-identical to HEAD.
# Tree cleanliness is asserted once, from the login node, by the launcher.
# ---------------------------------------------------------------------------
echo "=========================================="
echo "T19 complexity probe | job ${SLURM_JOB_ID:-local} task ${TASK_ID}"
echo "Node:        $(hostname)  ($(lscpu | sed -n 's/^Model name: *//p' | head -1))"
echo "Start:       $(date)"
echo "Method/Arm:  ${METHOD} / ${ARM}"
echo "Problem:     ${PROBLEM_NAME}"
echo "Seed:        ${SEED}"
echo "Config:      ${CONFIG}"
echo "Results:     ${RESULTS_DIR}"
echo "SP-1 commit: $(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo n/a)"
echo "SP-1 tag:    $(git -C "${REPO_DIR}" describe --tags --always 2>/dev/null || echo n/a)"
# 🔴 The commit above is NOT sufficient provenance for this probe.  The T19
# sources are rsynced on top of the deployed checkout without a commit of their
# own, and a second agent's uncommitted work is present in the same tree, so
# `rev-parse HEAD` names a commit that does NOT contain the code being tested.
# Content hashes of the files under test are therefore the load-bearing
# provenance record; compare them against the local tree before believing any
# result. (A probe is allowed this; the campaign is not -- C2 must deploy from
# a clean checkout at a tag, per EXECUTION-PLAN §4.)
echo "SP-1 dirty:  $(git -C "${REPO_DIR}" status --porcelain 2>/dev/null | wc -l) modified path(s)"
for f in src/isalsr/core/complexity.py experiments/models/complexity_telemetry.py; do
    echo "SP-1 sha256: $(sha256sum "${REPO_DIR}/${f}" 2>/dev/null | cut -c1-16)  ${f}"
done
"${PYTHON}" - <<'PROV'
import datetime, os, sys
try:
    import isalsr
    from isalsr.core import _native, backends
    from isalsr.core import complexity as _cx
    print(f"SP-2 pkg:    {isalsr.__file__}")
    print(f"SP-2 native: {_native.__file__}")
    print(f"SP-2 mtime:  {datetime.datetime.fromtimestamp(os.path.getmtime(_native.__file__))}")
    print(f"SP-3 engine: {backends.engine()}")
    print(f"SP-3 build:  {backends.build_info()}")
    # T19: the telemetry module must be importable on the COMPUTE node, and it
    # must be the one carrying the descriptor set this probe exists to check.
    print(f"T19 module:  {_cx.__file__}")
    print(f"T19 fields:  {len(_cx.DESCRIPTOR_FIELDS)} -> {','.join(_cx.DESCRIPTOR_FIELDS)}")
except Exception as exc:  # a broken engine must be loud, not inferred later
    print(f"[FATAL] engine probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
PROV
echo "T19 env:     ISALSR_COMPLEXITY=${ISALSR_COMPLEXITY:-<unset, defaults on>}"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# Payload.  --ledger and --postprocess skip mirror the C2 worker: the ledger
# defaults off and would otherwise record five reachability rates of zero (SP-6),
# and post-processing is a campaign-level step that 24 concurrent tasks must not
# each perform.
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
echo "=== Task ${TASK_ID} (${METHOD}/${ARM}/${PROBLEM_NAME}/seed=${SEED}) rc=${RC} ==="
exit ${RC}
