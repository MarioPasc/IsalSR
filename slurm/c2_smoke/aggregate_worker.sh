#!/usr/bin/env bash
# =============================================================================
# Campaign C2 pre-flight -- Stage C aggregation job (runs once, after the arrays)
# =============================================================================
# Rebuilds, over the whole output root and from files already on disk:
#   * aggregate.csv per (method, benchmark, problem, arm)          -> C1.17
#   * paired_stats.json / _hash_vs_baseline / _isalsr_vs_hash      -> C1.16
#   * the across-problem Holm correction
#   * status_ledger.csv                                            -> C2
#
# Why this is a separate job and not part of each task: the three arms live in
# three different SLURM arrays, so no single task can see all of them, and the
# three seeds of one arm are three concurrent tasks writing one aggregate.csv.
# Submitted with --dependency=afterany so it also runs when some arrays failed
# -- a partial ledger naming the gaps is worth more than no ledger.
#
# Environment (exported by launcher.sh):
#   ISALSR_REPO_DIR, C2_RESULTS_DIR, C2_CONFIG_LIST (space-separated paths)
# =============================================================================
set -euo pipefail

START_TIME=$(date +%s)

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
RESULTS_DIR="${C2_RESULTS_DIR:?ERROR: C2_RESULTS_DIR not set}"
CONFIG_LIST="${C2_CONFIG_LIST:?ERROR: C2_CONFIG_LIST not set}"

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
PYTHON="${CONDA_PREFIX}/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="$(command -v python3)"

echo "=========================================="
echo "C2 Stage C aggregation | job ${SLURM_JOB_ID:-local}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "Root:    ${RESULTS_DIR}"
echo "Commit:  $(git -C "${REPO_DIR}" describe --tags --always --dirty 2>/dev/null || echo n/a)"
echo "=========================================="

FAILED=0
for CFG in ${CONFIG_LIST}; do
    echo ""
    echo "--- postprocess: $(basename "${CFG}") ---"
    if ! "${PYTHON}" -m experiments.models.orchestrator \
            --config "${CFG}" \
            --output-dir "${RESULTS_DIR}" \
            --postprocess only; then
        echo "[WARN] postprocess failed for ${CFG}"
        FAILED=$((FAILED + 1))
    fi
done

# Certification: every C1.x criterion, computed from the files on disk.  Exits
# non-zero on any blocking failure, so the job's own state carries the verdict.
CERT_DIR="${RESULTS_DIR}/c2_preflight"
mkdir -p "${CERT_DIR}"
echo ""
echo "--- Stage C certification ---"
set +e
"${PYTHON}" -m experiments.scripts.c2_certify \
    --root "${RESULTS_DIR}" \
    --out-json "${CERT_DIR}/stage_c_certification.json" \
    --out-md "${CERT_DIR}/stage_c_certification.md" \
    --expected-tasks 1260 \
    --max-time 900
CERT_RC=$?
set -e

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "Postprocess failures: ${FAILED}"
echo "Certification exit:   ${CERT_RC}"
echo "Report: ${CERT_DIR}/stage_c_certification.md"
exit $(( FAILED > 0 ? 1 : CERT_RC ))
