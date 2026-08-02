#!/usr/bin/env bash
# =============================================================================
# T05 D2 Picasso PROBE worker (ARRAY JOB)
# =============================================================================
# This is a PROBE, not a campaign.  EXECUTION-PLAN.md §4.0 SP-0 is binding:
#   max_time <= 1800 s, <= 60 tasks, seed 0 only, output under ~/execs/isalsr/.
# It answers "do the D2 problems work on Picasso?".  It produces NO number for
# the paper; anything it measures is provisional until C2 reproduces it.
#
# Each array task runs ONE (method, problem) cell on the `isalsr` arm at seed 0,
# reading its parameters from tasks.txt line $SLURM_ARRAY_TASK_ID.
#
# DERIVED FROM slurm/t04_probe/worker.sh, deliberately and almost verbatim.
# That worker carries environment fixes that are not discoverable from the
# application code and each of which was added because something failed on this
# cluster: the MPI 5.0.9 module load, the conda LD_LIBRARY_PATH, PYTHONMALLOC.
# A freshly written worker would be missing all three.  Keep them in sync.
#
# CPU-only: no --gres, no GPU constraint.  The native extension is
# single-threaded by design, so one core is correct.
#
# Pinned to --constraint=intel (sd nodes, Xeon Gold 6230R) to match the T04
# probe and T01's AC-5 table, so per-DAG cost numbers stay comparable.  The
# extension is built -march=x86-64-v3 (AVX2, portable across sd/sr/bc/bl), so
# this pin is for timing comparability, NOT to dodge the AVX-512 SIGILL trap.
#SBATCH -J isalsr-t05-probe
#SBATCH --time=0-00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=intel
#SBATCH --account=tic_163_uma

set -euo pipefail

START_TIME=$(date +%s)
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

# ---------------------------------------------------------------------------
# Decode the task tuple.  Echoed unconditionally: a silently wrong decode
# yields a complete, plausible, WRONG result set.
# ---------------------------------------------------------------------------
LINE=$(grep -v '^#' "${T05_TASKS}" | grep -v '^[[:space:]]*$' | sed -n "${TASK_ID}p")
[[ -z "${LINE}" ]] && { echo "[FATAL] no task at row ${TASK_ID} of ${T05_TASKS}"; exit 1; }
read -r METHOD VARIANT PROBLEM CONFIG SUITE <<< "${LINE}"
for v in METHOD VARIANT PROBLEM CONFIG SUITE; do
    [[ -z "${!v}" ]] && { echo "[FATAL] empty ${v} in row ${TASK_ID}: '${LINE}'"; exit 1; }
done

echo "=========================================="
echo "T05 D2 PROBE (not a campaign run)"
echo "Job:          ${SLURM_JOB_ID:-local}  task ${TASK_ID}"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "Decoded:      method=${METHOD} variant=${VARIANT} problem=${PROBLEM}"
echo "              config=${CONFIG} suite=${SUITE} seed=0"
echo "=========================================="

# ---------------------------------------------------------------------------
# ENVIRONMENT -- mirrors slurm/t04_probe/worker.sh; see the note in the header.
# ---------------------------------------------------------------------------
module_loaded=0
for m in miniconda/3 miniconda3 Miniconda3 anaconda3 Anaconda3; do
    if module avail 2>&1 | grep -qiE "(^|/)${m}([[:space:]]|/|$)"; then
        module load "$m" && module_loaded=1 && break
    fi
done
[ "$module_loaded" -eq 0 ] && echo "[env] No conda module; assuming conda in PATH."

# MPI 5.0.9 -- REQUIRED by bingo-nasa via mpi4py.  mpi4py's ABI-probing import
# hook dlopen()s libmpi at IMPORT time, so its absence kills the task in ~13 s,
# long before any search starts.  A wrong major version yields "Please use mpi
# 5.0.9".  Mirrors slurm/workers/models_experiment_slurm.sh:31-35.
for mod in openmpi_gcc/5.0.9_gcc7 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/5.0.9_gcc14; do
    module load "$mod" 2>/dev/null && { echo "[env] loaded $mod"; break; }
done

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

# conda-installed openmpi lives here; without it mpi4py finds the module's libmpi
# but not its conda-side dependencies.
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)/envs/${CONDA_ENV_NAME}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Bypass CPython's pymalloc arena allocator in favour of glibc malloc: pymalloc
# fragments the heap over 10k+ generations and is what puts Bingo+IsalSR on the
# OOM killer.  Production setting; keep it here so the memory profile transfers.
export PYTHONMALLOC=malloc

PY="$(command -v python)"
OUT_DIR="${T05_OUT}/${METHOD}_${VARIANT}_${SUITE}_${PROBLEM}"
mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# SP-1..SP-6 -- the standing property probe, reused verbatim from T04.  Runs
# BEFORE the search, so a failure costs seconds rather than 25 minutes.
# ---------------------------------------------------------------------------
echo "--- SP-1..SP-6 ---"
"${PY}" "${REPO_DIR}/slurm/t04_probe/sp_probe.py" \
    --out "${OUT_DIR}/sp_evidence.json" \
    --method "${METHOD}" || { echo "[FATAL] SP probe failed"; exit 1; }

# SP-3 negative control: force the Python engine and assert it is ACTUALLY used.
# A probe that reports `native` in both directions proves nothing.
echo "--- SP-3 negative control ---"
ISALSR_ENGINE=python "${PY}" "${REPO_DIR}/slurm/t04_probe/sp_probe.py" \
    --out "${OUT_DIR}/sp_evidence_forced_python.json" \
    --method "${METHOD}" --expect-engine python \
    || { echo "[FATAL] SP-3 negative control failed -- engine override not honoured"; exit 1; }

# ---------------------------------------------------------------------------
# SP-7.1, 7.2, 7.4 -- D2's own contribution assertion, on a COMPUTE node.
# The dataset paths are what this establishes that a local run cannot: the
# Strogatz files are vendored in the repo tree and must resolve after rsync.
# ---------------------------------------------------------------------------
echo "--- SP-7 (pre-run) ---"
"${PY}" "${REPO_DIR}/slurm/t05_probe/check_d2.py" \
    --pre --out "${OUT_DIR}/sp7_pre.json" \
    || { echo "[FATAL] SP-7 pre-run checks failed"; exit 1; }

# ---------------------------------------------------------------------------
# THE PROBE RUN
# ---------------------------------------------------------------------------
echo "--- search (max_time=${T05_MAX_TIME}s, seed 0) ---"
"${PY}" -m experiments.models.orchestrator \
    --config "${CONFIG}" \
    --seeds 1 \
    --problems "${PROBLEM}" \
    --variants "${VARIANT}" \
    --output-dir "${OUT_DIR}" \
    --max-time "${T05_MAX_TIME}"

# ---------------------------------------------------------------------------
# SP-7.3, 7.5 -- the run_log parses, validates, and carries no NaN or inf.
# ---------------------------------------------------------------------------
echo "--- SP-7 (post-run) ---"
"${PY}" "${REPO_DIR}/slurm/t05_probe/check_d2.py" \
    --verify-runs "${OUT_DIR}" --out "${OUT_DIR}/sp7_post.json" \
    || { echo "[FATAL] SP-7 post-run checks failed"; exit 1; }

# ---------------------------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------------------------
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
echo "Output:    ${OUT_DIR}"
