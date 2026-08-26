#!/usr/bin/env bash
# =============================================================================
# Campaign C2 pre-flight -- Stage B launcher (EXECUTION-PLAN.md §4.2)
# =============================================================================
# Three micro-jobs, all CPU-only:
#
#   udfs   -- B1, B2 (+ negative control), B3, B9   on the UDFS host
#   bingo  -- B1, B2 (+ negative control), B3, B9   on the Bingo host   (SP-5)
#   b4     -- the equivalence gate re-run on a COMPUTE node.  T01 G1 passed on
#             the workstation only, which certifies neither this compiler
#             (gcc 13.2.0 here vs 12.2.0 locally) nor this CPU.
#
# Stage B gates Stage C.  A failure means fix, then re-run the stage from the
# top -- never "note it and continue".
#
# Usage:
#   bash slurm/c2_smoke/stage_b_launcher.sh --dry-run
#   bash slurm/c2_smoke/stage_b_launcher.sh --test-only
#   bash slurm/c2_smoke/stage_b_launcher.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ISALSR_REPO_DIR="${ISALSR_REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR}"
EVIDENCE_DIR="${C2_EVIDENCE_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/results/isalsr/c2_preflight/stage_b}"
LOGS_DIR="${C2_LOGS_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/c2_smoke/logs}"
ACCOUNT="tic_163_uma"
CONSTRAINT="${C2_CONSTRAINT:-cpu}"

MODE="submit"
[[ "${1:-}" == "--dry-run" ]] && MODE="dry"
[[ "${1:-}" == "--test-only" ]] && MODE="test"

# --dry-run must work from the workstation, where the Picasso paths do not exist.
if [[ "${MODE}" != "dry" ]]; then
    mkdir -p "${LOGS_DIR}" "${EVIDENCE_DIR}"
fi

_clean_job_id() { tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'; }
submit() {
    local raw id
    raw=$(sbatch --parsable "$@") || { echo "sbatch failed" >&2; return 1; }
    id=$(_clean_job_id "${raw}")
    [[ "${id}" =~ ^[0-9]+$ ]] || { echo "FATAL: unparsable job id: ${raw@Q}" >&2; return 1; }
    echo "${id}"
}

go() {   # go <name> <time> <mem_g> <script> <export-string>
    local name="$1" wall="$2" mem="$3" script="$4" exports="$5"
    local args=(
        --job-name="${name}" --time="${wall}"
        --ntasks=1 --cpus-per-task=1 --mem="${mem}G"
        --constraint="${CONSTRAINT}" --account="${ACCOUNT}"
        --output="${LOGS_DIR}/${name}_%j.out"
        --error="${LOGS_DIR}/${name}_%j.err"
        --export="${exports}"
        "${script}"
    )
    case "${MODE}" in
        dry)  echo "  [DRY] sbatch ${args[*]}" ;;
        test) sbatch --test-only "${args[@]}" >/dev/null 2>&1 \
                  && echo "  ${name}: --test-only OK" \
                  || { echo "  ${name}: --test-only FAILED"; exit 1; } ;;
        *)    local id; id=$(submit "${args[@]}") || exit 1
              echo "  ${name}: job ${id}"; echo "${id}" >> "${LOGS_DIR}/stage_b_job_ids.txt" ;;
    esac
}

echo "Stage B -- C2 pre-flight micro-jobs (mode: ${MODE})"
echo "  repo:     ${ISALSR_REPO_DIR}"
echo "  evidence: ${EVIDENCE_DIR}"
echo "  logs:     ${LOGS_DIR}"
echo ""
[[ "${MODE}" == "submit" ]] && : > "${LOGS_DIR}/stage_b_job_ids.txt"

COMMON="ALL,ISALSR_REPO_DIR=${ISALSR_REPO_DIR},C2_EVIDENCE_DIR=${EVIDENCE_DIR}"

# B1/B2/B3/B9 per host.  B9 runs two 240 s searches, B3 two short gates; 30 min
# is comfortable headroom and keeps a hang distinguishable from slow progress.
go c2b_udfs  0-00:30:00 16 "${SCRIPT_DIR}/stage_b_worker.sh" "${COMMON},C2_METHOD=udfs"
go c2b_bingo 0-00:30:00 32 "${SCRIPT_DIR}/stage_b_worker.sh" "${COMMON},C2_METHOD=bingo"

# B4: exhaustive k=1..8 plus the evolved decomposed corpus, byte-exact C++ vs
# Python.  Host-independent, so one job.
go c2b_gate4 0-02:00:00 16 "${SCRIPT_DIR}/stage_b4_worker.sh" "${COMMON}"

echo ""
if [[ "${MODE}" == "submit" ]]; then
    echo "Monitor:  ssh picasso 'squeue'"
    echo "Evidence: ${EVIDENCE_DIR}"
    echo "Job ids:  ${LOGS_DIR}/stage_b_job_ids.txt"
fi
