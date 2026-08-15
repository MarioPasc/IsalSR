#!/usr/bin/env bash
# =============================================================================
# Campaign C2 pre-flight -- Stage C certification ONLY (no post-processing)
# =============================================================================
# Runs experiments.scripts.c2_certify over a Stage C root whose post-processing
# artefacts are already on disk, and writes the go/no-go verdict.
#
# Why this exists as a separate worker from aggregate_worker.sh:
#   Job 1753134 ran the 14-config `--postprocess only` loop (~1h40 over 1,260
#   runs) and then TIMED OUT at its 2 h wall *before* the certifier produced a
#   verdict.  The post-processing artefacts persist (420 aggregate.csv, 420
#   paired-stats files), so re-running the whole aggregation would repeat 1h40
#   of GPFS-heavy work to reach the same starting point.  This worker skips
#   straight to the certifier.
#
#   Derived from aggregate_worker.sh -- the environment block is copied
#   verbatim, NOT rewritten.  The mpi4py module probe and PYTHONMALLOC are
#   load-bearing (mpi4py dlopen()s libmpi at *import*, killing the job in ~13 s
#   without them).
#
# Environment (exported by certify.sh):
#   ISALSR_REPO_DIR, C2_RESULTS_DIR
#   C2_SACCT_CSV     optional -- JobID,MaxRSS rows for the C1.11 memory profile
#   C2_EXPECTED      optional -- expected cell count (default 1260)
#   C2_MAX_TIME      optional -- per-run search budget in seconds (default 900)
#   C2_CERT_SEEDS    optional -- COLON-separated Stage C seeds (default 0:101:102).
#                    Colon, not comma: sbatch --export is comma-separated, so a
#                    comma in a value silently truncates it to the first seed.
# =============================================================================
set -euo pipefail

START_TIME=$(date +%s)

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
RESULTS_DIR="${C2_RESULTS_DIR:?ERROR: C2_RESULTS_DIR not set}"
SACCT_CSV="${C2_SACCT_CSV:-}"
EXPECTED="${C2_EXPECTED:-1260}"
MAX_TIME="${C2_MAX_TIME:-900}"
CERT_SEEDS_RAW="${C2_CERT_SEEDS:-0:101:102}"
CERT_SEEDS="${CERT_SEEDS_RAW//:/,}"

# Assert the decode rather than trusting the transport.  If a comma ever
# survives into --export, this arrives as a single seed and the certifier would
# reconcile a 1,260-cell root against a 420-cell expectation -- a wrong verdict,
# not a missing one.  Fail loudly instead.
N_SEEDS_DECODED=$(awk -F, '{print NF}' <<<"${CERT_SEEDS}")
if [[ "${N_SEEDS_DECODED}" -lt 3 ]]; then
    echo "[FATAL] C2_CERT_SEEDS decoded to ${N_SEEDS_DECODED} seed(s): '${CERT_SEEDS}'." >&2
    echo "        Expected >=3.  sbatch --export ate a comma; ship the list colon-separated." >&2
    exit 1
fi

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
echo "C2 Stage C certification | job ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "Root:      ${RESULTS_DIR}"
echo "Expected:  ${EXPECTED} cells, max_time=${MAX_TIME}s, seeds=${CERT_SEEDS}"
echo "sacct CSV: ${SACCT_CSV:-<none -- C1.11 will report no memory profile>}"
echo "Commit:    $(git -C "${REPO_DIR}" describe --tags --always --dirty 2>/dev/null || echo n/a)"
echo "=========================================="

CERT_DIR="${RESULTS_DIR}/c2_preflight"
mkdir -p "${CERT_DIR}"

CERT_ARGS=(
    --root "${RESULTS_DIR}"
    --out-json "${CERT_DIR}/stage_c_certification.json"
    --out-md   "${CERT_DIR}/stage_c_certification.md"
    --expected-tasks "${EXPECTED}"
    --max-time "${MAX_TIME}"
    --seeds "${CERT_SEEDS}"
    --log-level INFO
)
# Only pass --sacct-csv when the file really exists: C1.11 reporting "no memory
# profile" is an honest partial result, whereas pointing the certifier at a
# missing path is an argument error that produces no verdict at all.
if [[ -n "${SACCT_CSV}" && -s "${SACCT_CSV}" ]]; then
    CERT_ARGS+=(--sacct-csv "${SACCT_CSV}")
else
    echo "[warn] no sacct CSV at '${SACCT_CSV}' -- C1.11 will be unpopulated"
fi

set +e
"${PYTHON}" -m experiments.scripts.c2_certify "${CERT_ARGS[@]}"
CERT_RC=$?
set -e

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
echo "Certification exit: ${CERT_RC}"
echo "Report: ${CERT_DIR}/stage_c_certification.md"
exit ${CERT_RC}
