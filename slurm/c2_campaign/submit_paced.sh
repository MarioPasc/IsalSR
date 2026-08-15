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
# 🔴 TWO THINGS THIS SCRIPT DOES NOT DO (both found 2026-08-07):
#
#   1. It does not choose `C2_MIN_JOBID` for you, and the default of 0 is unsafe
#      after a same-day Stage C -- see the fail-closed guard below.
#
#   2. It submits NO SWEEP ARRAYS.  `launcher.sh` submits a `c2w_*` sweep per
#      array (`afterany` on the mains) to pick up cells a task's deadline
#      refused to start, and extends the aggregation dependency onto them.
#      Without that step, those cells (5-20 of 12,600 by the launcher's own
#      simulation) are never recovered and the aggregation runs against a tree
#      missing them -- and it breaks the commitment made to SCBI in
#      CAMPAIGN_BRIEF.md section 6.  Run `submit_sweeps.sh` after this script.
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
# SCBI request 2026-08-07: stage output on the compute node, copy back results.
USE_LOCALSCRATCH="${C2_USE_LOCALSCRATCH:-1}"
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
# Ten columns since chunking (2026-08-07).  A seven-column plan from an older
# checkout would leave BUNDLE and the deadline EMPTY in `while read`, and the
# worker would then run one cell per task with no TIMEOUT guard -- i.e. silently
# submit the 12,600-task shape SCBI asked us to stop submitting.
[[ "$(awk -F'\t' 'NR==1{print NF}' <<<"${PLAN}")" -eq 10 ]] \
    || { echo "FATAL: plan has $(awk -F'\t' 'NR==1{print NF}' <<<"${PLAN}") columns, expected 10" >&2; exit 1; }

EXISTING="$(sacct -S today -n -P -X -o JobID,JobName 2>/dev/null \
    | awk -F'|' -v m="${MIN_JOBID}" \
        '{split($1,a,"_"); if (a[1]+0 >= m && $2 ~ /^c2s_/ && $2 !~ /aggregate|ledger/) print $2}' \
    | sort -u)"
N_EXIST=$(grep -c . <<<"${EXISTING}" || true)

# 🔴 FAIL CLOSED when the floor is unset and a same-day Stage C could mask the
# whole campaign (found 2026-08-07, before it cost a submission).
#
# This script skips arrays by JOB NAME, and the smoke and campaign profiles
# build the SAME names -- both from the one launcher,
# `c2s_${METHOD:0:1}${ARM:0:1}_${SUITE}`.  The runbook REQUIRES Stage C on the
# commit being submitted, which in practice is the same day.  So with the
# default MIN_JOBID=0 this script would: print SKIP forty-two times, submit ZERO
# arrays, find `ALL_IDS` = the 42 STAGE C ids, satisfy `-eq 42`, submit the
# aggregation `afterany` on the SMOKE wave, write the smoke ids to job_ids.txt,
# and exit 0.  A silent no-op that reads as a successful submission -- the same
# family as 2026-08-06, where `--test-only` said 42/42 while every task was
# about to abort.
#
# Refusing is right rather than auto-deriving a floor: on a genuine RECOVERY run
# the already-submitted campaign arrays must still be skipped, and only the
# operator knows which of the two situations this is.
if [[ "${MIN_JOBID}" == "0" && "${N_EXIST}" -gt 0 ]]; then
    cat >&2 <<EOF
FATAL: ${N_EXIST} c2s_* array(s) already exist today and C2_MIN_JOBID is unset.

  Stage C builds the SAME 42 job names as the campaign, so without a floor this
  run would skip all of them, submit nothing, and attach the aggregation to the
  Stage C wave -- while exiting 0.

  If those arrays are a Stage C wave, set the floor above them:

      MIN=\$(( \$(sacct -S today -n -P -X -o JobID | cut -d_ -f1 | sort -n | tail -1) + 1 ))
      C2_MIN_JOBID=\$MIN bash \$0

  If they ARE this campaign and you are completing a partial submission, pass
  the campaign's own floor (the id of its first array), not 0.
EOF
    exit 1
fi

echo "Campaign C2 -- paced submission"
echo "  repo:    ${REPO}"
echo "  results: ${RESULTS_ROOT}"
echo "  seeds:   ${SEEDS}   max_time: ${MAX_TIME}s   pacing: ${SLEEP_BETWEEN}s"
echo "  already present (job id >= ${MIN_JOBID}): ${N_EXIST}"
echo ""

declare -a IDS=()
N_NEW=0

N_CELLS_TOTAL=0
while IFS=$'\t' read -r METHOD ARM SUITE N_TASKS THROTTLE MEM WALL \
                        BUNDLE CUTOFF_S N_CELLS; do
    [[ -n "${METHOD}" ]] || continue
    JOB_NAME="c2s_${METHOD:0:1}${ARM:0:1}_${SUITE}"
    N_CELLS_TOTAL=$((N_CELLS_TOTAL + N_CELLS))

    if grep -qx "${JOB_NAME}" <<<"${EXISTING}"; then
        printf '  %-34s SKIP (already submitted)\n' "${JOB_NAME}"
        continue
    fi
    if ${DRY}; then
        printf '  %-34s %4d cells /%3d = %4d tasks  %%%-3d  %3dG  %s\n' \
               "${JOB_NAME}" "${N_CELLS}" "${BUNDLE}" "${N_TASKS}" \
               "${THROTTLE}" "${MEM}" "${WALL}"
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
        --export="ALL,ISALSR_REPO_DIR=${REPO},C2_METHOD=${METHOD},C2_ARM=${ARM},C2_SUITE=${SUITE},C2_CONFIG=${REPO}/experiments/configs/${METHOD}_${SUITE}.yaml,C2_SEEDS=${SEEDS},C2_MAX_TIME=${MAX_TIME},C2_RESULTS_DIR=${RESULTS_ROOT},C2_BUNDLE=${BUNDLE},C2_START_CUTOFF_S=${CUTOFF_S},C2_USE_LOCALSCRATCH=${USE_LOCALSCRATCH}" \
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
    printf '  %-34s %4d tasks (B=%d)  %%%-3d  %3dG  job %s\n' \
           "${JOB_NAME}" "${N_TASKS}" "${BUNDLE}" "${THROTTLE}" "${MEM}" "${ID}"
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
    --export="ALL,ISALSR_REPO_DIR=${REPO},C2_RESULTS_DIR=${RESULTS_ROOT},C2_LEDGER_ONLY=1,C2_EXPECTED_TASKS=${N_CELLS_TOTAL},C2_MAX_TIME=${MAX_TIME}" \
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
