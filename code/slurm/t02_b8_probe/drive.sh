#!/usr/bin/env bash
# =============================================================================
# B8 probe driver -- runs the three phases and reports what was observed.
# =============================================================================
# Usage (on the Picasso login node):
#   bash slurm/t02_b8_probe/drive.sh <phase>
#     skip     : resubmit an intact cell, expect SKIP
#     corrupt  : truncate the run_log mid-JSON (no submission)
#     rerun    : resubmit the corrupt cell, expect DETECT + DELETE + RE-RUN
#
# Kept as three explicit invocations rather than one script so each submission
# is a genuine, separate resubmission -- which is the behaviour B8 asks to see.
# =============================================================================
set -uo pipefail

PHASE="${1:?usage: drive.sh skip|corrupt|rerun}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE_ROOT="${B8_PROBE_ROOT:-/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/t02_b8_probe}"
RUN_LOG="${PROBE_ROOT}/results/bingo/nguyen/nguyen_1/baseline/seed_00/run_log.json"
LOGS_DIR="${PROBE_ROOT}/logs"

wait_for_drain() {
    for _ in $(seq 1 120); do
        if squeue 2>/dev/null | grep -q "you have submitted 0 jobs"; then return 0; fi
        sleep 15
    done
    echo "[WARN] queue did not drain within 30 min" >&2
    return 1
}

case "${PHASE}" in
    corrupt)
        SIZE=$(stat -c %s "${RUN_LOG}")
        head -c $((SIZE / 2)) "${RUN_LOG}" > "${RUN_LOG}.trunc" && mv "${RUN_LOG}.trunc" "${RUN_LOG}"
        echo "corrupted ${RUN_LOG}: ${SIZE} -> $(stat -c %s "${RUN_LOG}") bytes"
        echo "tail: $(tail -c 40 "${RUN_LOG}")"
        ;;
    skip|rerun)
        LABEL="phase$([[ ${PHASE} == skip ]] && echo 2_skip || echo 3_rerun)"
        bash "${SCRIPT_DIR}/launcher.sh" "${LABEL}" >/dev/null 2>&1
        wait_for_drain
        NEWEST_ERR=$(ls -t "${LOGS_DIR}"/b8probe_${LABEL}_*.err 2>/dev/null | head -1)
        NEWEST_OUT="${NEWEST_ERR%.err}.out"
        echo "=== ${LABEL} : $(basename "${NEWEST_OUT}") ==="
        cat "${NEWEST_OUT}"
        echo "--- orchestrator decision ---"
        grep -Ei "skipping|corrupt run_log|running nguyen" "${NEWEST_ERR}" || echo "(no decision line found)"
        ;;
    *)
        echo "unknown phase ${PHASE}" >&2; exit 1 ;;
esac
