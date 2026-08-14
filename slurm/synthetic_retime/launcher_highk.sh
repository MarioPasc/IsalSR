#!/usr/bin/env bash
# Submit the high-k extension of the synthetic scalability study to Picasso.
#
# Complements the k = 1..9 exhaustive run: that one proves rho = k! and cannot be
# sampled; this one measures how canonicalization cost scales once k clears the
# fixed-overhead floor, which needs a sample and not an enumeration.
#
# Usage:
#   bash slurm/synthetic_retime/launcher_highk.sh --dry-run
#   bash slurm/synthetic_retime/launcher_highk.sh --test-only
#   bash slurm/synthetic_retime/launcher_highk.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_NAME="isalsr"
export REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR"
export RESULTS_DIR="/mnt/home/users/tic_163_uma/mpascual/results/isalsr/synthetic_highk"
export LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/logs"

# Colon-separated on purpose: sbatch --export splits on commas, so a comma here
# would be truncated to "8" and the rest parsed as junk variable names, with no
# error and a silently shorter grid. The worker translates back.
#
# The grid starts at k = 8 and not lower, and that bound is forced by the design
# rather than chosen. Every cell must sample the SAME number of permutations, and
# a cell can only supply MAX_PERMS distinct orderings if k! >= MAX_PERMS. At
# MAX_PERMS = 20,000 that is true from 8! = 40,320 upward and false below it, so
# k = 7 cannot be measured under the same condition as k = 8 at this P. Lowering
# P to reach k = 7 would trade the whole grid's accuracy for one point.
export K_VALUES="8:9:10:11:12:13:14:15:16:17:18:19:20:22:24:26:28:30:32:34:36"

# 20,000 sampled permutations per expression, IDENTICAL AT EVERY k. This is the
# controlled variable of the whole experiment and the reason this job exists.
#
# Measured at k = 9, m = 2, on the same 15 DAGs, varying only the permutation
# count: 24 perms -> 16.55 us, 1,000 -> 10.92, 5,000 -> 11.31, 20,000 -> 10.65,
# exhaustive 362,880 -> 9.98. The per-permutation mean is inflated by a
# per-expression warm-up cost that amortises as ~C/P, so a small sample reads
# HIGH -- by 66% at P = 24 and still 6.8% at P = 20,000.
#
# The consequence is that the exhaustive k = 1..9 run CANNOT supply a scaling
# curve: there P = k!, so k = 1 averages one cold call and k = 9 averages 362,880
# warm ones, and the low-k points are inflated by a factor that shrinks as k
# grows. That manufactures a flat region out of nothing. Holding P fixed leaves a
# residual bias too, but it is common to every k and so shifts the intercept
# rather than the slope, which is the quantity being reported.
export MAX_PERMS=20000

# 100 expressions x 20,000 permutations = 2,000,000 timed calls per cell, which
# is ample for a mean; the projected wall is ~2.75 h on EPYC, clearing SCBI's
# two-hour floor with margin under the 8 h request.
export N_EXPR=100
export PERM_TIMEOUT=120.0
export GLOBAL_SEED=42

mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}" 2>/dev/null || true

MODE="submit"
case "${1:-}" in
    --dry-run)   MODE="dry" ;;
    --test-only) MODE="test" ;;
    "")          ;;
    *)           echo "unknown flag: $1" >&2; exit 2 ;;
esac

_clean_job_id() {
    tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}

SBATCH_ARGS=(
    --parsable
    --export="ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_DIR=${REPO_DIR},RESULTS_DIR=${RESULTS_DIR},K_VALUES=${K_VALUES},MAX_PERMS=${MAX_PERMS},N_EXPR=${N_EXPR},PERM_TIMEOUT=${PERM_TIMEOUT},GLOBAL_SEED=${GLOBAL_SEED}"
    "${SCRIPT_DIR}/worker_highk.sh"
)

if [[ "${MODE}" == "dry" ]]; then
    echo "[DRY-RUN] sbatch ${SBATCH_ARGS[*]}"
    exit 0
fi

if [[ "${MODE}" == "test" ]]; then
    sbatch --test-only "${SBATCH_ARGS[@]}"
    exit $?
fi

RAW=$(sbatch "${SBATCH_ARGS[@]}") || { echo "sbatch failed" >&2; exit 1; }
JOB_ID=$(_clean_job_id "${RAW}")
[[ "${JOB_ID}" =~ ^[0-9]+$ ]] || {
    echo "FATAL: unparsable job id: ${RAW@Q}" >&2
    echo "A job may nonetheless have been queued -- run 'squeue' before resubmitting." >&2
    exit 1
}

echo "Submitted job ${JOB_ID}"
echo "Grid:     k in {${K_VALUES//:/, }}, m in {1,2,3} = 27 cells, ${MAX_PERMS} sampled perms/expr"
echo "Log:      ${LOGS_DIR}/synth_highk_${JOB_ID}.out"
echo "Results:  ${RESULTS_DIR}"
