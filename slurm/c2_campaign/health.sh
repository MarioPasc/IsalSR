#!/usr/bin/env bash
# =============================================================================
# Campaign C2 -- one-line health probe
# =============================================================================
# Emits ONE line, prefixed OK / WARN / ALERT, so it can drive a monitor loop.
#
# Coverage rule: it must speak on FAILURE, not only on progress. A probe that
# greps for the happy path stays silent through a crashloop, and silence then
# looks identical to "still running" -- which is how the 2026-08-06 submission
# looked healthy while all 12,600 tasks were dying on startup.
#
#   bash slurm/c2_campaign/health.sh
# =============================================================================
set -uo pipefail

REMOTE="${ISALSR_REMOTE:-picasso}"

ssh "${REMOTE}" "bash -s" <<'REMOTE_EOF'
F=/mnt/home/users/tic_163_uma/mpascual/fscratch
L=$F/execs/isalsr/c2_3arm/logs
R=$F/results/isalsr/c2_3arm

IDS=$(paste -sd, "$L/job_ids.txt" 2>/dev/null)
[ -n "$IDS" ] || { echo "ALERT job_ids.txt unreadable at $L"; exit 0; }
SW=$(paste -sd, "$L/sweep_job_ids.txt" 2>/dev/null)

ST=$(sacct -X -n -P -o State -j "$IDS" 2>/dev/null)
RUN=$(grep -c '^RUNNING'   <<<"$ST")
PEND=$(grep -c '^PENDING'  <<<"$ST")
DONE=$(grep -c '^COMPLETED' <<<"$ST")
BAD=$(grep -cE '^(FAILED|OUT_OF_MEMORY|TIMEOUT|NODE_FAIL|BOOT_FAIL)' <<<"$ST")
CANC=$(grep -c '^CANCELLED' <<<"$ST")

CELLS=$(find "$R" -name run_log.json 2>/dev/null | wc -l)
FATAL=$(grep -l '\[FATAL\] copy-back' "$L"/*.out 2>/dev/null | wc -l)
# quota's fscratch row: name space quota limit grace <sep> files quota limit grace
INODES=$(quota 2>/dev/null | awk '/^fscratc/ {print $7}')

LVL=OK
MSG=""
[ "${BAD:-0}"   -gt 0 ] && { LVL=WARN;  MSG="$MSG ${BAD} task(s) in a failure state;"; }
[ "${FATAL:-0}" -gt 0 ] && { LVL=ALERT; MSG="$MSG ${FATAL} log(s) with [FATAL] copy-back -- results stranded on a compute node;"; }
# Nothing left to run, yet the tree is short of 12,600 cells.
[ "${RUN:-0}" -eq 0 ] && [ "${PEND:-0}" -eq 0 ] && [ "${CELLS:-0}" -lt 12600 ] \
    && { LVL=ALERT; MSG="$MSG queue empty but only ${CELLS}/12600 cells;"; }
[ "${CELLS:-0}" -ge 12600 ] && MSG="$MSG ALL CELLS PRESENT;"

echo "$LVL cells=${CELLS}/12600 run=${RUN} pend=${PEND} done=${DONE} bad=${BAD} canc=${CANC} fatal_copyback=${FATAL} inodes=${INODES}/250.0k |${MSG}"
REMOTE_EOF
