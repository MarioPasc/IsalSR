#!/usr/bin/env bash
# =============================================================================
# Campaign C2 -- paced, idempotent submission of the 42 arrays + aggregation
# =============================================================================
# Why this exists instead of `launch.sh` calling the launcher once.
#
# On 2026-08-06 the launcher submitted 29 of 42 arrays in a tight loop and then
# got
#     sbatch: error: Slurm temporarily unable to accept job, sleeping and retrying
#     sbatch: error: Batch job submission failed: Resource temporarily unavailable
# and aborted -- before writing job_ids.txt and before submitting the
# aggregation, leaving 29 untracked arrays on the cluster.  Re-submitting the
# remaining 13 with a 20 s gap between them succeeded with no refusals at all.
#
# So the binding constraint was submission RATE (slurmctld RPC pressure), not
# MaxJobCount: the same 12,600 records were accepted once they arrived slowly.
# This script paces them, and -- because a partial submission is now a known
# failure mode rather than a surprise -- it is idempotent: it skips any array
# that already exists, so re-running after an abort completes the set instead of
# duplicating it.
#
#   bash slurm/c2_campaign/submit_paced.sh --dry-run   # print, submit nothing
#   bash slurm/c2_campaign/submit_paced.sh             # gate, then submit
#   C2_SLEEP=30 bash slurm/c2_campaign/submit_paced.sh # slower, if refused again
#
# RUN IT ON PICASSO, from the deployed tree: it needs sbatch, and the worker
# path recorded by sbatch must be the deployed worker (defect 15, SP-1).
#
# It does NOT run the Stage F gate -- `launch.sh` does that.  Use launch.sh
# unless you are recovering a partial submission.
# =============================================================================
set -uo pipefail

REPO="${ISALSR_REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR}"
FSCRATCH="${ISALSR_FSCRATCH:-/mnt/home/users/tic_163_uma/mpascual/fscratch}"
RESULTS_ROOT="${C2_RESULTS_DIR:-${FSCRATCH}/results/isalsr/c2_3arm}"
LOGS_DIR="${C2_LOGS_DIR:-${FSCRATCH}/execs/isalsr/c2_3arm/logs}"
ACCOUNT="${C2_ACCOUNT:-tic_163_uma}"
CONSTRAINT="${C2_CONSTRAINT:-sr}"
SEEDS="${C2_SEEDS_SPEC:-1-30}"
MAX_TIME="${C2_MAX_TIME:-43200}"
SLOT_BUDGET="${C2_SLOT_BUDGET:-2016}"
AGG_WALL="${C2_AGG_WALL:-0-01:59:00}"
SLEEP_BETWEEN="${C2_SLEEP:-20}"

# Arrays belonging to THIS campaign, by job id.  Filtering by name alone is
# wrong: the smoke and campaign profiles build the SAME job names (both come
# from the one launcher), so "every c2s_ job today" also matches the Stage C
# waves and would report all 42 as already submitted.
MIN_JOBID="${C2_MIN_JOBID:-0}"

DRY=false
[[ "${1:-}" == "--dry-run" ]] && DRY=true

command -v sbatch >/dev/null 2>&1 || {
    echo "FATAL: no sbatch here. Run this ON Picasso, from ${REPO}." >&2; exit 1; }

$DRY || mkdir -p "${LOGS_DIR}" "${RESULTS_ROOT}"

_clean_job_id() {  # the Lua wrapper prepends ANSI + a banner to --parsable
    tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}

PY="${FSCRATCH}/conda_envs/isalsr/bin/python"
[[ -x "${PY}" ]] || PY="$(command -v python3)"

PLAN="$(cd "${REPO}" && PYTHONPATH="${REPO}/src:${REPO}" "${PY}" \
        -m experiments.scripts.c2_slot_plan --config-dir "${REPO}/experiments/configs" \
        --seeds "${SEEDS}" --budget "${SLOT_BUDGET}" --tsv)" \
    || { echo "FATAL: slot plan failed; nothing submitted" >&2; exit 1; }
[[ "$(wc -l <<<"${PLAN}")" -eq 42 ]] \
    || { echo "FATAL: plan has $(wc -l <<<"${PLAN}") rows, expected 42" >&2; exit 1; }

EXISTING="$(sacct -S today -n -P -X -o JobID,JobName 2>/dev/null \
    | awk -F'|' -v m="${MIN_JOBID}" \
        '{split($1,a,"_"); if (a[1]+0 >= m && $2 ~ /^c2s_/ && $2 !~ /aggregate|ledger/) print $2}' \
    | sort -u)"
N_EXIST=$(grep -c . <<<"${EXISTING}" || true)

echo "Campaign C2 -- paced submission"
echo "  repo:    ${REPO}"
echo "  results: ${RESULTS_ROOT}"
echo "  seeds:   ${SEEDS}   max_time: ${MAX_TIME}s   pacing: ${SLEEP_BETWEEN}s"
echo "  already present (job id >= ${MIN_JOBID}): ${N_EXIST}"
echo ""

declare -a IDS=()
N_NEW=0

while IFS=$'\t' read -r METHOD ARM SUITE N_TASKS THROTTLE MEM WALL; do
    [[ -n "${METHOD}" ]] || continue
    JOB_NAME="c2s_${METHOD:0:1}${ARM:0:1}_${SUITE}"

    if grep -qx "${JOB_NAME}" <<<"${EXISTING}"; then
        printf '  %-34s SKIP (already submitted)\n' "${JOB_NAME}"
        continue
    fi
    if ${DRY}; then
        printf '  %-34s %4d tasks  %%%-3d  %3dG  %s\n' \
               "${JOB_NAME}" "${N_TASKS}" "${THROTTLE}" "${MEM}" "${WALL}"
        continue
    fi

    RAW=$(sbatch --parsable \
        --array="1-${N_TASKS}%${THROTTLE}" \
        --job-name="${JOB_NAME}" \
        --time="${WALL}" \
        --ntasks=1 --cpus-per-task=1 \
        --mem="${MEM}G" \
        --constraint="${CONSTRAINT}" \
        --account="${ACCOUNT}" \
        --output="${LOGS_DIR}/${JOB_NAME}_%A_%a.out" \
        --export="ALL,ISALSR_REPO_DIR=${REPO},C2_METHOD=${METHOD},C2_ARM=${ARM},C2_SUITE=${SUITE},C2_CONFIG=${REPO}/experiments/configs/${METHOD}_${SUITE}.yaml,C2_SEEDS=${SEEDS},C2_MAX_TIME=${MAX_TIME},C2_RESULTS_DIR=${RESULTS_ROOT}" \
        "${REPO}/slurm/c2_smoke/worker.sh" 2>&1)
    ID=$(_clean_job_id "${RAW}")

    if [[ ! "${ID}" =~ ^[0-9]+$ ]]; then
        printf '  %-34s FAILED: %s\n' "${JOB_NAME}" "$(tail -n 1 <<<"${RAW}")"
        echo ""
        echo "Submitted ${N_NEW} array(s) before this refusal."
        echo "RE-RUN THIS SCRIPT to submit the rest -- it skips what already exists."
        echo "If it refuses again, raise the pacing: C2_SLEEP=45 bash $0"
        [[ ${#IDS[@]} -gt 0 ]] && printf '%s\n' "${IDS[@]}" >> "${LOGS_DIR}/job_ids.txt"
        exit 1
    fi

    IDS+=("${ID}")
    N_NEW=$((N_NEW + 1))
    printf '  %-34s %4d tasks  %%%-3d  %3dG  job %s\n' \
           "${JOB_NAME}" "${N_TASKS}" "${THROTTLE}" "${MEM}" "${ID}"
    sleep "${SLEEP_BETWEEN}"
done <<< "${PLAN}"

${DRY} && { echo ""; echo "[DRY-RUN] nothing submitted."; exit 0; }

# ---- All 42 must exist before the aggregation can depend on them -----------
mapfile -t ALL_IDS < <(sacct -S today -n -P -X -o JobID,JobName 2>/dev/null \
    | awk -F'|' -v m="${MIN_JOBID}" \
        '{split($1,a,"_"); if (a[1]+0 >= m && $2 ~ /^c2s_/ && $2 !~ /aggregate|ledger/) print a[1]"|"$2}' \
    | sort -u -t'|' -k2,2 | cut -d'|' -f1 | sort -n | uniq)

echo ""
echo "Arrays now present: ${#ALL_IDS[@]} (new this run: ${N_NEW})"
if [[ ${#ALL_IDS[@]} -ne 42 ]]; then
    echo "Not all 42 arrays are present -- aggregation NOT submitted." >&2
    echo "Re-run this script to finish, then it will submit the aggregation." >&2
    exit 1
fi

CONFIG_LIST="$(awk -F'\t' -v r="${REPO}" \
    '{printf "%s/experiments/configs/%s_%s.yaml ", r, $1, $3}' <<<"${PLAN}")"
CONFIG_LIST="${CONFIG_LIST% }"
DEP=$(IFS=:; echo "${ALL_IDS[*]}")

AGG_RAW=$(sbatch --parsable \
    --job-name=c2s_aggregate --array="1-42" --time="${AGG_WALL}" \
    --ntasks=1 --cpus-per-task=2 --mem=16G \
    --constraint="${CONSTRAINT}" --account="${ACCOUNT}" \
    --dependency="afterany:${DEP}" \
    --output="${LOGS_DIR}/c2s_aggregate_%A_%a.out" \
    --error="${LOGS_DIR}/c2s_aggregate_%A_%a.err" \
    --export="ALL,ISALSR_REPO_DIR=${REPO},C2_RESULTS_DIR=${RESULTS_ROOT},C2_CONFIG_LIST=${CONFIG_LIST}" \
    "${REPO}/slurm/c2_smoke/aggregate_worker.sh" 2>&1)
AGG_ID=$(_clean_job_id "${AGG_RAW}")
[[ "${AGG_ID}" =~ ^[0-9]+$ ]] || { echo "FATAL: aggregation not submitted: ${AGG_RAW}" >&2; exit 1; }
# sbatch ACCEPTS a malformed dependency and records Dependency=(null): the job
# would then start immediately, against partial input.
scontrol show job "${AGG_ID}" | grep -q 'Dependency=(null)' \
    && { echo "FATAL: aggregation dependency dropped -- cancelling" >&2; scancel "${AGG_ID}"; exit 1; }
echo "Aggregation array ${AGG_ID} (42 tasks, afterany on 42 arrays)"

LED_RAW=$(sbatch --parsable \
    --job-name=c2s_ledger --time="0-02:00:00" \
    --ntasks=1 --cpus-per-task=1 --mem=16G \
    --constraint="${CONSTRAINT}" --account="${ACCOUNT}" \
    --dependency="afterany:${AGG_ID}" \
    --output="${LOGS_DIR}/c2s_ledger_%j.out" \
    --error="${LOGS_DIR}/c2s_ledger_%j.err" \
    --export="ALL,ISALSR_REPO_DIR=${REPO},C2_RESULTS_DIR=${RESULTS_ROOT},C2_LEDGER_ONLY=1,C2_EXPECTED_TASKS=12600,C2_MAX_TIME=${MAX_TIME}" \
    "${REPO}/slurm/c2_smoke/aggregate_worker.sh" 2>&1)
LED_ID=$(_clean_job_id "${LED_RAW}")
[[ "${LED_ID}" =~ ^[0-9]+$ ]] || { echo "FATAL: ledger not submitted: ${LED_RAW}" >&2; exit 1; }
scontrol show job "${LED_ID}" | grep -q 'Dependency=(null)' \
    && { echo "FATAL: ledger dependency dropped -- cancelling" >&2; scancel "${LED_ID}"; exit 1; }
echo "Status-ledger job ${LED_ID} (afterany on ${AGG_ID})"

printf '%s\n' "${ALL_IDS[@]}" > "${LOGS_DIR}/job_ids.txt"
printf '%s\n%s\n' "${AGG_ID}" "${LED_ID}" > "${LOGS_DIR}/job_ids_aggregation.txt"
echo ""
echo "Job ids: ${LOGS_DIR}/job_ids.txt (42 rows, ascending)"
