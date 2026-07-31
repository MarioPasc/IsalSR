#!/usr/bin/env bash
# =============================================================================
# T04 hash-arm Picasso PROBE worker (ARRAY JOB)
# =============================================================================
# This is a PROBE, not a campaign.  EXECUTION-PLAN.md §4.0 SP-0 is binding:
#   max_time <= 1800 s, <= 60 tasks, seed 0 only, output under ~/execs/isalsr/.
# It answers "does the hash arm work on Picasso?".  It produces NO number for
# the paper; anything it measures is provisional until C2 reproduces it.
#
# Each array task runs ONE (method, variant, problem) cell at seed 0, reading
# its parameters from tasks.txt line $SLURM_ARRAY_TASK_ID.
#
# Establishes SP-1..SP-6 per task (provenance, install freshness, engine +
# negative control, alphabet, both hosts, fallback counters) and writes the
# evidence as JSON next to the run output.
#
# CPU-only: no --gres, no GPU constraint.  The native extension is
# single-threaded by design, so one core is correct.
#
# Pinned to --constraint=intel (sd nodes, Xeon Gold 6230R) so the per-DAG cost
# numbers are comparable with T01's AC-5 table, which was measured there.
# The extension is built -march=x86-64-v3 (CMakeLists.txt:23), which is AVX2
# and portable across sd/sr/bc/bl -- so this pin is for timing comparability,
# NOT to dodge the AVX-512 SIGILL trap (pre-flight B6b).
#SBATCH -J isalsr-t04-probe
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
LINE=$(grep -v '^#' "${T04_TASKS}" | grep -v '^[[:space:]]*$' | sed -n "${TASK_ID}p")
[[ -z "${LINE}" ]] && { echo "[FATAL] no task at row ${TASK_ID} of ${T04_TASKS}"; exit 1; }
read -r METHOD VARIANT PROBLEM CONFIG SHADOW <<< "${LINE}"
for v in METHOD VARIANT PROBLEM CONFIG SHADOW; do
    [[ -z "${!v}" ]] && { echo "[FATAL] empty ${v} in row ${TASK_ID}: '${LINE}'"; exit 1; }
done

echo "=========================================="
echo "T04 PROBE (not a campaign run)"
echo "Job:          ${SLURM_JOB_ID:-local}  task ${TASK_ID}"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "Decoded:      method=${METHOD} variant=${VARIANT} problem=${PROBLEM}"
echo "              config=${CONFIG} shadow=${SHADOW} seed=0"
echo "=========================================="

# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------
module_loaded=0
for m in miniconda/3 miniconda3 Miniconda3 anaconda3 Anaconda3; do
    if module avail 2>&1 | grep -qiE "(^|/)${m}([[:space:]]|/|$)"; then
        module load "$m" && module_loaded=1 && break
    fi
done
[ "$module_loaded" -eq 0 ] && echo "[env] No conda module; assuming conda in PATH."

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY="$(command -v python)"
OUT_DIR="${T04_OUT}/${METHOD}_${VARIANT}_${PROBLEM}_shadow-${SHADOW}"
mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# SP-1..SP-6 -- the standing property probe.  Runs BEFORE the search, so a
# failure costs seconds rather than 25 minutes.  "I checked it" is not
# evidence; this writes a parsed artefact.
# ---------------------------------------------------------------------------
echo "--- SP-1..SP-6 ---"
"${PY}" "${REPO_DIR}/slurm/t04_probe/sp_probe.py" \
    --out "${OUT_DIR}/sp_evidence.json" \
    --method "${METHOD}" || { echo "[FATAL] SP probe failed"; exit 1; }

# SP-3 negative control: force the Python engine and assert it is ACTUALLY used.
# canonical.py:349 was fixed 2026-07-31 to honour ISALSR_ENGINE; before that the
# override was reported but ignored, so this control passed while proving nothing.
echo "--- SP-3 negative control ---"
ISALSR_ENGINE=python "${PY}" "${REPO_DIR}/slurm/t04_probe/sp_probe.py" \
    --out "${OUT_DIR}/sp_evidence_forced_python.json" \
    --method "${METHOD}" --expect-engine python \
    || { echo "[FATAL] SP-3 negative control failed -- engine override not honoured"; exit 1; }

# ---------------------------------------------------------------------------
# THE PROBE RUN
# ---------------------------------------------------------------------------
SHADOW_FLAG=""
[[ "${SHADOW}" == "off" ]] && SHADOW_FLAG="--no-shadow-hash"

echo "--- search (max_time=${T04_MAX_TIME}s, seed 0) ---"
"${PY}" -m experiments.models.orchestrator \
    --config "${CONFIG}" \
    --seeds 1 \
    --problems "${PROBLEM}" \
    --variants "${VARIANT}" \
    --output-dir "${OUT_DIR}" \
    --max-time "${T04_MAX_TIME}" \
    ${SHADOW_FLAG}

# ---------------------------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------------------------
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
echo "Output:    ${OUT_DIR}"
