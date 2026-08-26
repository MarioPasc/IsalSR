#!/usr/bin/env bash
#SBATCH -J synth-retime
#SBATCH --time=0-08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --constraint=sr
#SBATCH --account=tic_163_uma
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/logs/synth_retime_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/logs/synth_retime_%j.err

# Re-time the synthetic permutation-scalability study on the compiled engine.
#
# Why this is one sequential single-core job and not an array:
#   * Per-permutation canonicalization time is a REPORTED quantity (supplementary
#     Table `tab:synthetic_scalability` and the power-law fit beside it). Running
#     27 cells concurrently on one node would measure memory-bandwidth contention
#     as well as the algorithm, so the cells run one after another on one core --
#     the same shape as every cell of the C2 campaign (`--cpus-per-task=1`).
#   * `--constraint=sr` pins to the AMD EPYC 7H12 family. All 12,600 C2 cells ran
#     on that model, and the supplementary states outright that CPUs were pinned
#     because wall-clock is reported. A desktop part would be roughly twice as
#     fast and the number would not be comparable with the campaign's key costs.
#   * The whole protocol is ~78 min single-core on a fast desktop, so on EPYC it
#     runs ~2-3 h: comfortably over SCBI's two-hour floor as a single submission,
#     with no array-placement cost to the scheduler at all.
#
# Output is 27 fragment CSVs plus one metadata JSON -- a handful of small files,
# not the thousands that make $LOCALSCRATCH mandatory, so it writes straight to
# $RESULTS_DIR under $HOME. Stated rather than left implicit.

set -euo pipefail

START_TIME=$(date +%s)

echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "Git commit:   $(git -C "${REPO_DIR:-.}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "=========================================="

# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------
module_loaded=0
for m in miniconda/3 miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
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

echo "[cpu] $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"

# ---------------------------------------------------------------------------
# ENGINE GATE -- the single most important check in this script.
#
# `isalsr.core.backends` falls back to the pure-Python canonicalizer silently
# when the compiled extension fails to import: DEFAULT_BACKEND is chosen from
# _CPP_AVAILABLE and the ImportError is swallowed. A Python run would complete
# with no error and produce timings several times larger than the campaign's,
# which is precisely the number this job exists to measure. Fail loudly instead.
# ---------------------------------------------------------------------------
python - <<'PYGATE'
import sys
from isalsr.core.backends import engine, DEFAULT_BACKEND
from isalsr.core import _native

eng = engine()
info = _native.build_info()
expected = "298fc1188bf1b051"   # the build every one of the 12,600 C2 cells recorded
print(f"[engine] engine={eng} default={DEFAULT_BACKEND} build_hash={info['build_hash']} "
      f"isa={info['isa_level']} compiler={info['compiler']}")
if eng != "cpp":
    sys.exit(f"FATAL: engine is {eng!r}, not 'cpp'. Refusing to report Python timings.")
if info["build_hash"] != expected:
    sys.exit(f"FATAL: build_hash {info['build_hash']} != campaign build {expected}. "
             "These timings would not be comparable with the reported campaign.")
print("[engine] gate passed")
PYGATE

mkdir -p "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# PAYLOAD -- 27 (k, m) cells, strictly sequential.
#
# One invocation per cell, because the runner writes the per-cell fragment
# `synth_k{K}_m{M}.csv` that the figure script globs only when it is given
# exactly one k and one m. A single invocation over all cells would write one
# combined CSV that the figure script does not read.
# ---------------------------------------------------------------------------
for K in 1 2 3 4 5 6 7 8 9; do
    for M in 1 2 3; do
        FRAG="${RESULTS_DIR}/synth_k${K}_m${M}.csv"
        if [[ -s "${FRAG}" ]]; then
            echo "[skip] k=${K} m=${M} already present"
            continue
        fi
        echo "[cell] k=${K} m=${M} start $(date +%H:%M:%S)"
        python -m experiments.synthetic_scalability.run_synthetic_scalability \
            --output-dir "${RESULTS_DIR}" \
            --n-expr "${N_EXPR}" \
            --k-values "${K}" \
            --m-values "${M}" \
            --max-perms "${MAX_PERMS}" \
            --timeout "${PERM_TIMEOUT}" \
            --seed "${GLOBAL_SEED}"
        echo "[cell] k=${K} m=${M} done  $(date +%H:%M:%S)"
    done
done

# ---------------------------------------------------------------------------
# COMPLETENESS -- 27 fragments or the run is not usable.
# ---------------------------------------------------------------------------
N_FRAG=$(find "${RESULTS_DIR}" -maxdepth 1 -name 'synth_k*_m*.csv' | wc -l)
echo "[check] fragments written: ${N_FRAG} / 27"
[[ "${N_FRAG}" -eq 27 ]] || { echo "FATAL: incomplete fragment set" >&2; exit 1; }

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
