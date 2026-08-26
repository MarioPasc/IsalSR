#!/usr/bin/env bash
# =============================================================================
# C2 chunked-topology smoke on Picasso -- does localscratch lose anything?
# =============================================================================
#   bash slurm/c2_smoke/chunk_smoke.sh --dry-run   # print the 12 arrays
#   bash slurm/c2_smoke/chunk_smoke.sh             # submit (run ON Picasso)
#
# Shape: 2 problems x 2 seeds x 3 arms x 2 methods = 24 cells, run TWICE:
#
#   wave A  C2_USE_LOCALSCRATCH=1   ->  <root>/staged     the campaign's path
#   wave B  C2_USE_LOCALSCRATCH=0   ->  <root>/direct     the reference
#
# The two waves are identical in every other respect, so `chunk_smoke_verify.py`
# can diff the trees FILE BY FILE. That diff is the point of the whole script.
#
# 🔴 Why a diff and not a checklist.  The 2026-08-07 mock counted `run_log.json`
# and passed, while `metadata.json` -- which the orchestrator writes at the ROOT
# of the output tree, outside every cell directory -- was being left on the node
# and lost. `c2_certify.py` reads that file and criterion C1.4 fails without it,
# so the campaign would have run for two days and then failed certification on
# evidence that no longer existed. Every artefact the check looked for came back;
# that is exactly why the check passed. A whole-tree diff cannot be fooled that
# way, because it does not know what it is looking for.
#
# This runs the REAL slurm/c2_smoke/worker.sh under the REAL chunking and
# staging. It is not a model of the campaign path, it is that path on a 40 s
# payload -- "certifying a topology you will not launch certifies nothing" (§1).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${ISALSR_REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR}"
FSCRATCH="${ISALSR_FSCRATCH:-/mnt/home/users/tic_163_uma/mpascual/fscratch}"
ROOT="${C2_PROBE_ROOT:-${FSCRATCH}/results/isalsr/c2_chunk_smoke}"
LOGS="${C2_PROBE_LOGS:-${FSCRATCH}/execs/isalsr/c2_chunk_smoke/logs}"
ACCOUNT="${C2_ACCOUNT:-tic_163_uma}"
CONSTRAINT="${C2_CONSTRAINT:-sr}"

# Seeds 0 and 101, never 1..30: SP-0 forbids a probe output that could be
# mistaken for a campaign cell, and probe trees do get rsynced. Two seeds is
# also the minimum the worker's own guard accepts.
SEEDS="0,101"
PROBLEMS="Nguyen-1,Nguyen-2"
MAX_TIME="${C2_PROBE_MAX_TIME:-40}"     # per-CELL payload budget, seconds
BUNDLE=2                                # 4 cells / 2 = 2 tasks per array
TEARDOWN=600                            # matches the launcher's SMOKE_TEARDOWN_S
WALL_S=$(( BUNDLE * (MAX_TIME + TEARDOWN) + TEARDOWN ))
CUTOFF_S=$(( WALL_S - MAX_TIME - TEARDOWN ))
WALL=$(printf '0-%02d:%02d:00' $((WALL_S / 3600)) $((WALL_S % 3600 / 60)))

# 8 GB, not the campaign's 16/32: a 40 s payload allocates nothing, and this
# probe measures file movement, not memory. Under `short` (MaxWall 2 h) so it
# starts immediately -- the wall above is 21 minutes.
MEM=8

MODE="submit"
[[ "${1:-}" == "--dry-run" ]] && MODE="dry"

echo "C2 chunked-topology smoke"
echo "  repo:      ${REPO}"
echo "  root:      ${ROOT}/{staged,direct}"
echo "  logs:      ${LOGS}"
echo "  shape:     ${PROBLEMS} x seeds ${SEEDS} x {baseline,hash,isalsr} x {udfs,bingo}"
echo "  bundle:    ${BUNDLE} cells/task   wall ${WALL}   cutoff ${CUTOFF_S}s   payload ${MAX_TIME}s"
echo ""

if [[ "${MODE}" == "submit" ]]; then
    command -v sbatch >/dev/null 2>&1 || {
        echo "FATAL: no sbatch here. Run this ON Picasso." >&2; exit 1; }
    # Start clean: the verifier compares file COUNTS between the two waves, and
    # a leftover tree from an earlier run would make both waves look complete.
    rm -rf "${ROOT}" "${LOGS}"
    mkdir -p "${ROOT}/staged" "${ROOT}/direct" "${LOGS}"
fi

_clean_job_id() { tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'; }

# 🔴 DERIVE the task count; never hard-code it.
#
# The first version of this script wrote `N_TASKS=2` because 2 problems x 2 seeds
# / bundle 2 is obviously 2. It was obviously 2 and it was wrong: `--export` had
# truncated the problem list to one entry, the array really held 1 task, and all
# twelve task-2's died with "index 2 out of range [1, 1]". A hard-coded count
# cannot disagree with the decode, so it cannot catch a decode that changed --
# and the decode is the thing most likely to change.
#
# Asking `c2_task_spec --count` uses the SAME arithmetic the worker will use, on
# the SAME arguments, so a mismatch is impossible by construction rather than by
# inspection.
PY="${FSCRATCH}/conda_envs/isalsr/bin/python"
[[ -x "${PY}" ]] || PY="$(command -v python3)"
task_count() {  # <config>
    PYTHONPATH="${REPO}/src:${REPO}" "${PY}" -m experiments.scripts.c2_task_spec \
        --config "$1" --seeds "${SEEDS}" --problems "${PROBLEMS}" \
        --bundle "${BUNDLE}" --count
}

EXPECTED_TASKS=2      # 2 problems x 2 seeds / bundle 2
IDS=()
for WAVE in staged direct; do
    USE_LOCAL=$([[ "${WAVE}" == "staged" ]] && echo 1 || echo 0)
    for METHOD in udfs bingo; do
        for ARM in baseline hash isalsr; do
            NAME="c2cs_${WAVE:0:1}${METHOD:0:1}${ARM:0:1}"
            CONFIG="${REPO}/experiments/configs/${METHOD}_nguyen.yaml"
            N_TASKS="$(task_count "${CONFIG}")" || {
                echo "FATAL: could not size ${NAME}" >&2; exit 1; }
            if [[ "${N_TASKS}" != "${EXPECTED_TASKS}" ]]; then
                echo "FATAL: ${NAME} decodes to ${N_TASKS} task(s), expected ${EXPECTED_TASKS}." >&2
                echo "       The problem or seed list is not reaching the decode intact." >&2
                exit 1
            fi

            if [[ "${MODE}" == "dry" ]]; then
                printf '  %-14s %-6s %-8s %-6s  %d tasks x %d cells  localscratch=%d\n' \
                       "${NAME}" "${WAVE}" "${ARM}" "${METHOD}" "${N_TASKS}" "${BUNDLE}" "${USE_LOCAL}"
                continue
            fi

            # 🔴 C2_SEEDS is shipped COLON-separated: sbatch --export is
            # comma-separated, so a comma in a VALUE starts the next variable and
            # silently truncates the list. The worker translates back.
            RAW=$(sbatch --parsable \
                --array="1-${N_TASKS}" \
                --job-name="${NAME}" \
                --time="${WALL}" \
                --ntasks=1 --cpus-per-task=1 --mem="${MEM}G" \
                --constraint="${CONSTRAINT}" \
                --account="${ACCOUNT}" \
                --output="${LOGS}/${NAME}_%A_%a.out" \
                --export="ALL,ISALSR_REPO_DIR=${REPO},C2_METHOD=${METHOD},C2_ARM=${ARM},C2_SUITE=nguyen,C2_CONFIG=${CONFIG},C2_SEEDS=${SEEDS//,/:},C2_PROBLEMS=${PROBLEMS//,/:},C2_MAX_TIME=${MAX_TIME},C2_RESULTS_DIR=${ROOT}/${WAVE},C2_BUNDLE=${BUNDLE},C2_START_CUTOFF_S=${CUTOFF_S},C2_USE_LOCALSCRATCH=${USE_LOCAL}" \
                "${SCRIPT_DIR}/worker.sh" 2>&1)
            ID=$(_clean_job_id "${RAW}")
            [[ "${ID}" =~ ^[0-9]+$ ]] || {
                echo "FATAL: unparsable job id for ${NAME}: $(tail -n1 <<<"${RAW}")" >&2; exit 1; }
            IDS+=("${ID}")
            printf '  %-14s %-6s %-8s %-6s  %d tasks x %d cells  job %s\n' \
                   "${NAME}" "${WAVE}" "${ARM}" "${METHOD}" "${N_TASKS}" "${BUNDLE}" "${ID}"
        done
    done
done

[[ "${MODE}" == "dry" ]] && { echo ""; echo "[DRY-RUN] nothing submitted."; exit 0; }

printf '%s\n' "${IDS[@]}" > "${LOGS}/job_ids.txt"
echo ""
echo "Submitted ${#IDS[@]} arrays, $(( ${#IDS[@]} * N_TASKS )) tasks, 48 cells."
echo ""
echo "When they finish:"
echo "  python slurm/c2_smoke/chunk_smoke_verify.py ${ROOT}"
