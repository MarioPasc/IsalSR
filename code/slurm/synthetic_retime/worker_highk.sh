#!/usr/bin/env bash
#SBATCH -J synth-highk
#SBATCH --time=0-08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --constraint=sr
#SBATCH --account=tic_163_uma
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/logs/synth_highk_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/logs/synth_highk_%j.err

# High-k extension of the synthetic scalability study.
#
# WHY THIS EXISTS, and why it is a second job rather than a wider first one.
#
# The k = 1..9 run (job 2001009) enumerates all k! permutations exhaustively.
# That is what proves the rho = k! collapse, and it cannot be sampled: the claim
# is that EVERY one of the k! orderings maps to one canonical string. It is also
# why k stops at 9 -- 600 * 11! is 2.4e10 canonicalizations, and 600 * 12! is
# 2.9e11, so exhaustive enumeration dies of combinatorics well before the
# canonicalizer does.
#
# But the TIMING question does not need exhaustive enumeration. Per-permutation
# cost is estimated just as well from a fixed sample, so k can go far higher for
# almost no compute. That matters, because measured locally the cost is FLAT
# across k = 1..9 (7.6 -> 10.3 us) and then rises sharply: 14.0 us at k = 12,
# 40.6 at k = 20, 81.2 at k = 32. The flat region is a fixed per-call overhead
# floor masking the true scaling, and a study that stopped at k = 9 would have
# concluded there is no growth. There is: roughly O(k^1.6-1.8) once k clears the
# floor -- sub-quadratic, consistent with the near-O(k^2) bound, nowhere near
# factorial.
#
# The top of the grid, k = 32, is deliberate: it is the top of the campaign's
# own k-stratified overhead table, so the synthetic curve and the empirical one
# meet at the same k rather than describing disjoint ranges.
#
# CONSEQUENCE FOR REPORTING, which must not be lost between here and the table:
# these rows are SAMPLED. rho is the number of distinct canonical strings among
# MAX_PERMS sampled orderings, not among k!. Invariance still means something
# (every sampled ordering agreed); "rho = k!" does not. Any table mixing these
# rows with the exhaustive ones has to mark which is which.

set -euo pipefail

START_TIME=$(date +%s)

echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "Git commit:   $(git -C "${REPO_DIR:-.}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "K values:     ${K_VALUES}"
echo "Sampling:     ${MAX_PERMS} permutations per expression"
echo "=========================================="

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

# Engine gate -- identical to the k=1..9 worker. A silent fall-back to the
# Python canonicalizer would produce timings several times larger with no error,
# and timing is the entire output of this job.
python - <<'PYGATE'
import sys
from isalsr.core.backends import engine, DEFAULT_BACKEND
from isalsr.core import _native

eng = engine()
info = _native.build_info()
expected = "298fc1188bf1b051"
print(f"[engine] engine={eng} default={DEFAULT_BACKEND} build_hash={info['build_hash']} "
      f"isa={info['isa_level']} compiler={info['compiler']}")
if eng != "cpp":
    sys.exit(f"FATAL: engine is {eng!r}, not 'cpp'. Refusing to report Python timings.")
if info["build_hash"] != expected:
    sys.exit(f"FATAL: build_hash {info['build_hash']} != campaign build {expected}.")
print("[engine] gate passed")
PYGATE

mkdir -p "${RESULTS_DIR}"

# K_VALUES arrives colon-separated: --export splits on commas, so a comma inside
# a value is truncated and its tail is parsed as a junk variable name. Nothing
# errors; the job simply runs a shorter grid than intended.
IFS=':' read -r -a KS <<< "${K_VALUES}"
echo "[grid] ${#KS[@]} k values x 3 m values = $(( ${#KS[@]} * 3 )) cells"

for K in "${KS[@]}"; do
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

EXPECTED=$(( ${#KS[@]} * 3 ))
N_FRAG=$(find "${RESULTS_DIR}" -maxdepth 1 -name 'synth_k*_m*.csv' | wc -l)
echo "[check] fragments in ${RESULTS_DIR}: ${N_FRAG} (this job expected ${EXPECTED})"
[[ "${N_FRAG}" -ge "${EXPECTED}" ]] || { echo "FATAL: incomplete fragment set" >&2; exit 1; }

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished:  $(date)"
echo "Duration:  $((ELAPSED / 3600))h $(((ELAPSED / 60) % 60))m $((ELAPSED % 60))s"
