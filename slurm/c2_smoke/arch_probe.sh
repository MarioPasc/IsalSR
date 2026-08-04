#!/usr/bin/env bash
# =============================================================================
# C4 cross-architecture reproducibility probe -- one task per node family
# =============================================================================
# Stage C v2 failed C4 with 35 of 210 (problem, seed) pairs carrying more than
# one data_fingerprint, and all 35 partition EXACTLY by CPU family. This runs
# the same generator on an Intel `sd` node and an AMD `sr` node so the
# difference can be quantified in ULP rather than inferred from a hash.
#
# Runs on the login node. Two 5-minute jobs, then a local diff.
#
# Usage:  bash slurm/c2_smoke/arch_probe.sh [--dry-run]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ISALSR_REPO_DIR="${ISALSR_REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR}"
OUT_DIR="${C4_PROBE_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/c4_arch_probe}"
LOGS_DIR="${OUT_DIR}/logs"
ACCOUNT="tic_163_uma"
mkdir -p "${LOGS_DIR}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

_clean_job_id() { tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'; }
submit() {
    local raw id
    raw=$(sbatch --parsable "$@") || { echo "sbatch failed" >&2; return 1; }
    id=$(_clean_job_id "${raw}")
    [[ "${id}" =~ ^[0-9]+$ ]] || { echo "FATAL: unparsable job id: ${raw@Q}" >&2; return 1; }
    echo "${id}"
}

IDS=()
for FAMILY in sd sr; do
    ARGS=(
        --job-name="c4probe_${FAMILY}"
        --time=0-00:10:00
        --ntasks=1 --cpus-per-task=2 --mem=8G
        --constraint="${FAMILY}"
        --account="${ACCOUNT}"
        --output="${LOGS_DIR}/c4probe_${FAMILY}_%j.out"
        --error="${LOGS_DIR}/c4probe_${FAMILY}_%j.err"
        --export="ALL,ISALSR_REPO_DIR=${ISALSR_REPO_DIR},C4_OUT=${OUT_DIR}/${FAMILY}.json"
        "${SCRIPT_DIR}/arch_probe_worker.sh"
    )
    if ${DRY_RUN}; then
        echo "[DRY-RUN] sbatch --parsable ${ARGS[*]}"
        sbatch --test-only "${ARGS[@]}"
        continue
    fi
    ID=$(submit "${ARGS[@]}") || exit 1
    IDS+=("${ID}")
    echo "submitted ${FAMILY}: job ${ID}"
done

${DRY_RUN} && exit 0

echo ""
echo "When both finish:"
echo "  python -m experiments.scripts.c4_arch_reproducibility --compare ${OUT_DIR}/sd.json ${OUT_DIR}/sr.json"
printf '%s\n' "${IDS[@]}" > "${LOGS_DIR}/job_ids.txt"
