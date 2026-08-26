#!/usr/bin/env bash
# =============================================================================
# T15 three-arm CONST-normalization probe — Picasso launcher
# =============================================================================
#
# Measures how three CONST-normalization policies behave on the DAG population
# that real SR search actually produces:
#
#   submitted  old aggressive relocation of every CONST in-edge onto node 0
#   repair     new minimal repair of CONST nodes with in-degree 0  (production)
#   none       no CONST handling at all
#
# All three are scored on the SAME DAG at the moment the search hands it to the
# canonicaliser, so the comparison is exactly paired.
#
# UDFS is the reason this runs on the cluster. It checks its own max_time only
# between order-enumeration stages, so a short budget overruns by orders of
# magnitude and the probe is not practical on the workstation. Bingo finishes
# locally in minutes and is included here only for cluster-side confirmation.
#
# Usage:
#   bash slurm/t15_norm_arms_launch.sh                  # submit udfs + bingo
#   bash slurm/t15_norm_arms_launch.sh --dry-run        # print sbatch commands
#   bash slurm/t15_norm_arms_launch.sh --method udfs    # single method
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/workers/t15_norm_arms_slurm.sh"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR"
RESULTS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/t15_norm_arms"
LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/logs/t15_norm_arms"
ACCOUNT="tic_163_uma"
CONSTRAINT="cpu"

N_PROBLEMS=5          # one per benchmark tier; see PROBLEMS in the worker
N_SEEDS=3
MAX_CONCURRENT=8

# Per-run search budget. UDFS overshoots this substantially because it only
# checks the clock between enumeration stages, hence the generous wall time.
MAX_TIME=600

# method:time_limit:cpus:mem_gb
# UDFS production tasks use 8G/15h; the probe is shorter but the overshoot is
# unbounded in principle, so the wall time stays generous and the memory is
# doubled to absorb the three canonical-string sets held per run.
METHOD_SPECS=(
    "udfs:08:00:00:1:16"
    "bingo:04:00:00:1:32"
)

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
DRY_RUN=false
SINGLE_METHOD=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --method)  SINGLE_METHOD="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--dry-run] [--method udfs|bingo]"
            exit 1
            ;;
    esac
done

# Created on Picasso; skipped on a dry run so the command can be previewed from
# the workstation, where these paths do not exist.
if ! ${DRY_RUN}; then
    mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}"
fi

# ---------------------------------------------------------------------------
# Job-ID capture
#
# Picasso's Lua sbatch wrapper prepends ANSI codes and a multi-line warning to
# --parsable output. Take the LAST line before stripping non-digits: a naive
# sed runs line-by-line and returns a multi-line "id", and the guard would then
# fire AFTER the job was already submitted, leaving it untracked on the cluster.
# ---------------------------------------------------------------------------
_clean_job_id() {
    tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}

submit() {
    local raw id
    raw=$(sbatch --parsable "$@") || { echo "sbatch failed" >&2; return 1; }
    id=$(_clean_job_id "${raw}")
    [[ "${id}" =~ ^[0-9]+$ ]] || { echo "FATAL: unparsable job id: ${raw@Q}" >&2; return 1; }
    echo "${id}"
}

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
N_TASKS=$(( N_PROBLEMS * N_SEEDS ))
SUBMITTED=()

for spec in "${METHOD_SPECS[@]}"; do
    IFS=':' read -r method th tm ts cpus mem_gb <<< "${spec}"
    time_limit="${th}:${tm}:${ts}"

    if [[ -n "${SINGLE_METHOD}" && "${method}" != "${SINGLE_METHOD}" ]]; then
        continue
    fi

    SBATCH_ARGS=(
        --job-name="t15_arms_${method}"
        --array="1-${N_TASKS}%${MAX_CONCURRENT}"
        --time="${time_limit}"
        --ntasks=1
        --cpus-per-task="${cpus}"
        --mem="${mem_gb}G"
        --constraint="${CONSTRAINT}"
        --account="${ACCOUNT}"
        --output="${LOGS_DIR}/t15_arms_${method}_%A_%a.out"
        --error="${LOGS_DIR}/t15_arms_${method}_%A_%a.err"
        --export="ALL,ISALSR_REPO_DIR=${REPO_DIR},T15_METHOD=${method},T15_MAX_TIME=${MAX_TIME},T15_N_SEEDS=${N_SEEDS},T15_RESULTS_DIR=${RESULTS_DIR}"
        "${WORKER_SCRIPT}"
    )

    if ${DRY_RUN}; then
        echo "[DRY-RUN] sbatch --parsable ${SBATCH_ARGS[*]}"
        continue
    fi

    JOB_ID=$(submit "${SBATCH_ARGS[@]}") || exit 1
    SUBMITTED+=("${method}=${JOB_ID}")
    echo "Submitted ${method}: job ${JOB_ID} (${N_TASKS} tasks, throttle ${MAX_CONCURRENT})"
done

if ${DRY_RUN}; then
    echo ""
    echo "Dry run only — nothing submitted."
    exit 0
fi

echo ""
echo "Submitted: ${SUBMITTED[*]:-none}"
echo "Monitor:   squeue -u \$USER"
echo "Logs:      ${LOGS_DIR}/"
echo "Results:   ${RESULTS_DIR}/<method>/<suite>_<problem>_seed<N>/summary.json"
echo ""
echo "Aggregate when finished:"
echo "  python -m experiments.scripts.aggregate_norm_arms --results-dir ${RESULTS_DIR}"
