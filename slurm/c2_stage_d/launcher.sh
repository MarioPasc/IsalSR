#!/usr/bin/env bash
# =============================================================================
# Campaign C2 pre-flight -- Stage D launcher (EXECUTION-PLAN.md §4.4, audit.md §7)
# =============================================================================
# Submits the 12 full-length certification cells as THREE arrays -- one per
# memory group -- plus one dependent aggregation + certification job.
#
#   group          cells  --mem   why
#   -------------  -----  ------  -------------------------------------------
#   udfs               3   16 GB  UDFS holds no large dedup set
#   bingo_std          6   32 GB  baseline holds none, hash holds 64-bit digests
#   bingo_isalsr       3  256 GB  §3.3's evidence floor: 29 C1 cells died at
#                                 MaxRSS ~127.7 GB against a 128 GB request
#
# Three arrays rather than one, because sbatch --mem is per JOB.  A single
# 12-task array would have to request 256 GB for all twelve, which on `sr`
# (439 GB/node) would serialise the whole stage onto one task per node.
#
# SP-0: this script SUBMITS.  It must not be run until Mario has merged the
# audit branch into cpp-core-port and explicitly said go.  See RUNBOOK.md.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_LOCAL="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- Configuration ---------------------------------------------------------
ISALSR_REPO_DIR="${ISALSR_REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR}"
RESULTS_ROOT="${D_RESULTS_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/results/isalsr/c2_stage_d}"
LOGS_DIR="${D_LOGS_DIR:-${HOME}/execs/isalsr/c2_stage_d/logs}"
ACCOUNT="${D_ACCOUNT:-tic_163_uma}"

# Locked by audit.md §7 row 3.  Read from the registry so the shell and the
# Python side cannot drift.
CONSTRAINT="${D_CONSTRAINT:-sr}"
WALL="${D_WALL:-0-16:00:00}"
CPUS="${D_CPUS:-1}"
AGG_WALL="${D_AGG_WALL:-0-08:00:00}"
RSS_INTERVAL="${D_RSS_INTERVAL:-60}"

# Resolve python the way c2_smoke/launcher.sh does: the workstation-style
# ~/.conda path does not exist on Picasso (fscratch/conda_envs does).
PY="${ISALSR_PYTHON:-$(conda run -n isalsr which python 2>/dev/null || echo python3)}"

MODE="submit"
ONLY_GROUP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   MODE="dry" ;;
        --test-only) MODE="test" ;;
        --only)      ONLY_GROUP="$2"; shift ;;
        -h|--help)
            sed -n '2,25p' "${BASH_SOURCE[0]}"
            echo "Usage: $0 [--dry-run|--test-only] [--only GROUP]"
            exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
    shift
done

mkdir -p "${LOGS_DIR}"

# Registry queries run against the LOCAL checkout when previewing and against
# the deployed one when submitting; both must be the same commit, which
# deploy.sh asserts.
spec() { PYTHONPATH="${REPO_LOCAL}/src:${REPO_LOCAL}" "${PY}" \
             -m experiments.scripts.stage_d_task_spec "$@"; }

GROUPS=$(spec --groups)

echo "=========================================================="
echo "C2 Stage D launcher"
echo "  mode:        ${MODE}"
echo "  repo:        ${ISALSR_REPO_DIR}"
echo "  results:     ${RESULTS_ROOT}"
echo "  logs:        ${LOGS_DIR}"
echo "  constraint:  ${CONSTRAINT}"
echo "  wall:        ${WALL}"
echo "=========================================================="
spec --list
echo "=========================================================="

# Strip ANSI and take the last line: some sbatch wrappers on this cluster
# decorate stdout, and a decorated job id poisons every downstream dependency.
_clean_job_id() { sed -e 's/\x1b\[[0-9;]*m//g' | tr -d '[:space:]' | tail -c 16; }

JOB_IDS=()

for GROUP in ${GROUPS}; do
    if [[ -n "${ONLY_GROUP}" && "${GROUP}" != "${ONLY_GROUP}" ]]; then
        continue
    fi

    N_TASKS=$(spec --group "${GROUP}" --count)
    MEM_GB=$(spec --group "${GROUP}" --mem-gb)
    JOB_NAME="c2d_${GROUP}"

    # No throttle: 12 tasks total is far below any queue limit, and Stage D's
    # whole point is that all 12 run at the full budget concurrently.
    ARRAY_SPEC="1-${N_TASKS}"

    SBATCH_ARGS=(
        --job-name="${JOB_NAME}"
        --account="${ACCOUNT}"
        --array="${ARRAY_SPEC}"
        --time="${WALL}"
        --cpus-per-task="${CPUS}"
        --mem="${MEM_GB}G"
        --constraint="${CONSTRAINT}"
        --output="${LOGS_DIR}/${JOB_NAME}_%A_%a.out"
        --error="${LOGS_DIR}/${JOB_NAME}_%A_%a.err"
        --export="ALL,ISALSR_REPO_DIR=${ISALSR_REPO_DIR},D_GROUP=${GROUP},D_RESULTS_DIR=${RESULTS_ROOT},D_RSS_INTERVAL=${RSS_INTERVAL}"
    )

    echo ""
    echo "--- group ${GROUP}: ${N_TASKS} tasks, ${MEM_GB} GB, array ${ARRAY_SPEC}"

    case "${MODE}" in
        dry)
            echo "sbatch ${SBATCH_ARGS[*]} ${ISALSR_REPO_DIR}/slurm/c2_stage_d/worker.sh"
            ;;
        test)
            sbatch --test-only "${SBATCH_ARGS[@]}" \
                   "${ISALSR_REPO_DIR}/slurm/c2_stage_d/worker.sh"
            ;;
        submit)
            JID=$(sbatch --parsable "${SBATCH_ARGS[@]}" \
                         "${ISALSR_REPO_DIR}/slurm/c2_stage_d/worker.sh" | _clean_job_id)
            [[ -n "${JID}" ]] || { echo "[FATAL] empty job id for ${GROUP}" >&2; exit 1; }
            echo "    submitted: ${JID}"
            JOB_IDS+=("${JID}")
            ;;
    esac
done

if [[ "${MODE}" != "submit" ]]; then
    echo ""
    echo "Mode '${MODE}': nothing submitted."
    exit 0
fi

# ---------------------------------------------------------------------------
# Dependent aggregation + certification.  afterany, not afterok: a cell that
# OOMs must still be certified -- D1.2's whole purpose is to observe that.
# ---------------------------------------------------------------------------
DEP=$(IFS=:; echo "${JOB_IDS[*]}")
AGG_ARGS=(
    --job-name="c2d_certify"
    --account="${ACCOUNT}"
    --time="${AGG_WALL}"
    --cpus-per-task=2
    --mem=16G
    --constraint="${CONSTRAINT}"
    --dependency="afterany:${DEP}"
    --output="${LOGS_DIR}/c2d_certify_%j.out"
    --error="${LOGS_DIR}/c2d_certify_%j.err"
    --export="ALL,ISALSR_REPO_DIR=${ISALSR_REPO_DIR},D_RESULTS_DIR=${RESULTS_ROOT},D_LOGS_DIR=${LOGS_DIR},D_JOB_IDS=${DEP}"
)
AGG_JID=$(sbatch --parsable "${AGG_ARGS[@]}" \
                 "${ISALSR_REPO_DIR}/slurm/c2_stage_d/aggregate_worker.sh" | _clean_job_id)

# A dependency that SLURM could not parse is silently dropped and the job runs
# immediately -- against an empty results tree.  Verify it stuck.
sleep 2
if scontrol show job "${AGG_JID}" 2>/dev/null | grep -q 'Dependency=(null)'; then
    echo "[FATAL] aggregation job ${AGG_JID} lost its dependency; cancelling." >&2
    scancel "${AGG_JID}" || true
    exit 1
fi

printf '%s\n' "${JOB_IDS[@]}" > "${LOGS_DIR}/job_ids.txt"
echo "${AGG_JID}" > "${LOGS_DIR}/certify_job_id.txt"

echo ""
echo "=========================================================="
echo "Submitted ${#JOB_IDS[@]} arrays (12 cells): ${JOB_IDS[*]}"
echo "Aggregation + certification: ${AGG_JID} (afterany)"
echo "Job ids:  ${LOGS_DIR}/job_ids.txt"
echo "=========================================================="
