#!/usr/bin/env bash
# =============================================================================
# Campaign C2 pre-flight -- Stage D aggregation + certification (single job)
# =============================================================================
# Runs once, after all three Stage D arrays reach a terminal state (afterany).
#
# Three steps, in order:
#   1. Collect sacct memory accounting for the 12 cells.
#   2. Run the orchestrator's post-processing over the whole root, ONCE.
#   3. Run the D1 certifier.
#
# Environment (exported by launcher.sh):
#   ISALSR_REPO_DIR, D_RESULTS_DIR, D_LOGS_DIR, D_JOB_IDS (colon-separated)
# =============================================================================
set -uo pipefail

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
RESULTS_DIR="${D_RESULTS_DIR:?ERROR: D_RESULTS_DIR not set}"
LOGS_DIR="${D_LOGS_DIR:?ERROR: D_LOGS_DIR not set}"
JOB_IDS="${D_JOB_IDS:-}"

for mod in openmpi_gcc/5.0.9_gcc7 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/5.0.9_gcc14; do
    module load "$mod" 2>/dev/null && break
done
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)/envs/isalsr}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="$(command -v python3)"

PREFLIGHT_DIR="${RESULTS_DIR}/c2_preflight"
mkdir -p "${PREFLIGHT_DIR}"
SACCT_CSV="${LOGS_DIR}/stage_d_maxrss.csv"

# ---------------------------------------------------------------------------
# 1. sacct.  Two traps are baked into this one line (§11.1, 2026-08-02):
#
#    * NEVER `-X`.  `sacct -X` returns an EMPTY MaxRSS -- memory is accounted
#      on the `.batch` step, not the allocation.  The profile would come back
#      silently blank: no error, just empty cells.
#    * `JobIDRaw`, never `JobID`.  For an array, `JobID` reads
#      `<array_id>_<task>` while `status.json` records the raw numeric id;
#      joining on `JobID` matched 42 of 1,260 rows and still reported PASS.
# ---------------------------------------------------------------------------
if [[ -n "${JOB_IDS}" ]]; then
    JOB_CSV="${JOB_IDS//:/,}"
    echo "JobID,MaxRSS" > "${SACCT_CSV}"
    sacct -j "${JOB_CSV}" -n -P -o JobIDRaw,MaxRSS \
        | awk -F'|' '$1 ~ /\.batch$/ && $2 != "" {print $1","$2}' \
        >> "${SACCT_CSV}"
    echo "[1/3] sacct rows: $(( $(wc -l < "${SACCT_CSV}") - 1 )) -> ${SACCT_CSV}"
else
    echo "[1/3] no D_JOB_IDS supplied; certifier will fall back to status.json"
    SACCT_CSV=""
fi

# ---------------------------------------------------------------------------
# 2. Post-processing, ONCE over the whole root.  The array tasks all ran
#    --postprocess skip; left on, each would walk the entire tree and write the
#    same shared files concurrently.
# ---------------------------------------------------------------------------
FAILED=0
for METHOD in udfs bingo; do
    CFG="${REPO_DIR}/experiments/configs/${METHOD}_hard.yaml"
    [[ -f "${CFG}" ]] || { echo "[WARN] missing ${CFG}"; continue; }
    echo "[2/3] postprocess: ${CFG}"
    "${PYTHON}" -m experiments.models.orchestrator \
        --config "${CFG}" \
        --output-dir "${RESULTS_DIR}" \
        --postprocess only || { echo "[WARN] postprocess rc=$? for ${METHOD}"; FAILED=$((FAILED + 1)); }
done

# ---------------------------------------------------------------------------
# 3. Certify.
# ---------------------------------------------------------------------------
CERT_ARGS=(
    --root "${RESULTS_DIR}"
    --out-json "${PREFLIGHT_DIR}/stage_d_certification.json"
    --out-md "${PREFLIGHT_DIR}/stage_d_certification.md"
    --log-level INFO
)
[[ -n "${SACCT_CSV}" ]] && CERT_ARGS+=(--sacct-csv "${SACCT_CSV}")
[[ -n "${D_C1_REFERENCE:-}" ]] && CERT_ARGS+=(--c1-reference "${D_C1_REFERENCE}")

echo "[3/3] certify"
"${PYTHON}" -m experiments.scripts.stage_d_certify "${CERT_ARGS[@]}"
CERT_RC=$?

echo "=== Stage D aggregation done: postprocess failures=${FAILED}, certify rc=${CERT_RC} ==="
exit $(( FAILED > 0 ? 1 : CERT_RC ))
