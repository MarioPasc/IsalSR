#!/usr/bin/env bash
# =============================================================================
# Campaign C2 -- submit the sweep arrays `submit_paced.sh` omits, and repoint
# the aggregation to wait for them.
# =============================================================================
# `launcher.sh:557-592` submits, for every array with BUNDLE > 1, a sweep array
# `c2w_*` with `--dependency=afterany:<main arrays>`, then EXTENDS the
# aggregation dependency to include the sweeps:
#
#     "The aggregation must wait for the SWEEP, not just the main pass, or it
#      aggregates a tree that is still missing its deferred cells."
#
# `submit_paced.sh` has no sweep block at all, so a campaign submitted through
# it never recovers cells that a task's deadline refused to start (est. 5-20 of
# 12,600), and breaks the commitment made to SCBI in CAMPAIGN_BRIEF.md section 6.
#
# A sweep task whose cells are all complete exits in seconds -- the orchestrator
# resume logic skips a completed (method, arm, problem, seed). Submitting the
# sweeps minutes after the mains is equivalent to submitting them together,
# because they are `afterany` on the mains either way.
#
# Paced at 20 s, same reason as submit_paced.sh: the binding constraint on
# 2026-08-06 was slurmctld RPC rate.
#
#   bash submit_sweeps.sh --dry-run
#   bash submit_sweeps.sh
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
USE_LOCALSCRATCH="${C2_USE_LOCALSCRATCH:-1}"
SLEEP_BETWEEN="${C2_SLEEP:-20}"
AGG_JOB="${C2_AGG_JOB:-}"

DRY=false
[[ "${1:-}" == "--dry-run" ]] && DRY=true

command -v sbatch >/dev/null 2>&1 || {
    echo "FATAL: no sbatch here. Run this ON Picasso." >&2; exit 1; }

JOB_IDS_FILE="${LOGS_DIR}/job_ids.txt"
[[ -s "${JOB_IDS_FILE}" ]] || { echo "FATAL: ${JOB_IDS_FILE} missing" >&2; exit 1; }
mapfile -t MAIN_IDS < "${JOB_IDS_FILE}"
[[ ${#MAIN_IDS[@]} -eq 42 ]] || {
    echo "FATAL: expected 42 main arrays, found ${#MAIN_IDS[@]}" >&2; exit 1; }
DEP="$(IFS=:; echo "${MAIN_IDS[*]}")"

PY="${FSCRATCH}/conda_envs/isalsr/bin/python"
[[ -x "${PY}" ]] || PY="$(command -v python3)"

PLAN="$(cd "${REPO}" && PYTHONPATH="${REPO}/src:${REPO}" "${PY}" \
        -m experiments.scripts.c2_slot_plan --config-dir "${REPO}/experiments/configs" \
        --seeds "${SEEDS}" --budget "${SLOT_BUDGET}" --tsv)" \
    || { echo "FATAL: slot plan failed; nothing submitted" >&2; exit 1; }
[[ "$(wc -l <<<"${PLAN}")" -eq 42 ]] \
    || { echo "FATAL: plan has $(wc -l <<<"${PLAN}") rows, expected 42" >&2; exit 1; }
[[ "$(awk -F'\t' 'NR==1{print NF}' <<<"${PLAN}")" -eq 10 ]] \
    || { echo "FATAL: plan is not 10 columns" >&2; exit 1; }

# Idempotence: skip sweeps that already exist, scoped by job id like
# submit_paced.sh, so a Stage C `c2w_*` from earlier today cannot mask one.
MIN_JOBID="${C2_MIN_JOBID:-0}"
EXISTING="$(sacct -S today -n -P -X -o JobID,JobName 2>/dev/null \
    | awk -F'|' -v m="${MIN_JOBID}" \
        '{split($1,a,"_"); if (a[1]+0 >= m && $2 ~ /^c2w_/) print $2}' \
    | sort -u)"

echo "Campaign C2 -- sweep arrays"
echo "  results: ${RESULTS_ROOT}"
echo "  depends: afterany on ${#MAIN_IDS[@]} main arrays (${MAIN_IDS[0]}..${MAIN_IDS[-1]})"
echo "  already present (job id >= ${MIN_JOBID}): $(grep -c . <<<"${EXISTING}" || true)"
echo ""

declare -a SWEEP_IDS=()
_clean_job_id() { tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'; }

while IFS=$'\t' read -r METHOD ARM SUITE N_TASKS THROTTLE MEM WALL \
                        BUNDLE CUTOFF_S N_CELLS; do
    [[ -n "${METHOD}" ]] || continue
    [[ "${BUNDLE}" -gt 1 ]] || continue      # B=1 cannot spill: nothing to sweep
    SWEEP_NAME="c2w_${METHOD:0:1}${ARM:0:1}_${SUITE}"

    if grep -qx "${SWEEP_NAME}" <<<"${EXISTING}"; then
        printf '  %-34s SKIP (already submitted)\n' "${SWEEP_NAME}"
        continue
    fi
    if ${DRY}; then
        printf '  %-34s %4d tasks (B=%d)  %%%-3d  %3dG  %s\n' \
               "${SWEEP_NAME}" "${N_TASKS}" "${BUNDLE}" "${THROTTLE}" "${MEM}" "${WALL}"
        continue
    fi

    RAW=$(sbatch --parsable \
        --array="1-${N_TASKS}%${THROTTLE}" \
        --job-name="${SWEEP_NAME}" \
        --time="${WALL}" \
        --ntasks=1 --cpus-per-task=1 \
        --mem="${MEM}G" \
        --constraint="${CONSTRAINT}" \
        --account="${ACCOUNT}" \
        --dependency="afterany:${DEP}" \
        --output="${LOGS_DIR}/${SWEEP_NAME}_%A_%a.out" \
        --export="ALL,ISALSR_REPO_DIR=${REPO},C2_METHOD=${METHOD},C2_ARM=${ARM},C2_SUITE=${SUITE},C2_CONFIG=${REPO}/experiments/configs/${METHOD}_${SUITE}.yaml,C2_SEEDS=${SEEDS},C2_MAX_TIME=${MAX_TIME},C2_RESULTS_DIR=${RESULTS_ROOT},C2_BUNDLE=${BUNDLE},C2_START_CUTOFF_S=${CUTOFF_S},C2_USE_LOCALSCRATCH=${USE_LOCALSCRATCH}" \
        "${REPO}/slurm/c2_smoke/worker.sh" 2>&1)
    SID=$(_clean_job_id "${RAW}")

    if [[ ! "${SID}" =~ ^[0-9]+$ ]]; then
        printf '  %-34s FAILED: %s\n' "${SWEEP_NAME}" "$(tail -n 1 <<<"${RAW}")"
        echo "RE-RUN this script to submit the rest -- it skips what already exists." >&2
        [[ ${#SWEEP_IDS[@]} -gt 0 ]] && printf '%s\n' "${SWEEP_IDS[@]}" >> "${LOGS_DIR}/sweep_job_ids.txt"
        exit 1
    fi

    # sbatch ACCEPTS a malformed dependency and records Dependency=(null); the
    # sweep would then start immediately, against a tree still being written.
    if scontrol show job "${SID}" | grep -q 'Dependency=(null)'; then
        echo "FATAL: sweep dependency dropped -- cancelling ${SID}" >&2
        scancel "${SID}"; exit 1
    fi

    SWEEP_IDS+=("${SID}")
    printf '  %-34s %4d tasks (B=%d)  %%%-3d  %3dG  job %s\n' \
           "${SWEEP_NAME}" "${N_TASKS}" "${BUNDLE}" "${THROTTLE}" "${MEM}" "${SID}"
    sleep "${SLEEP_BETWEEN}"
done <<< "${PLAN}"

${DRY} && { echo ""; echo "[DRY-RUN] nothing submitted."; exit 0; }

[[ ${#SWEEP_IDS[@]} -gt 0 ]] || { echo "No sweeps submitted."; exit 0; }
printf '%s\n' "${SWEEP_IDS[@]}" > "${LOGS_DIR}/sweep_job_ids.txt"
echo ""
echo "Sweep arrays: ${#SWEEP_IDS[@]} -> ${LOGS_DIR}/sweep_job_ids.txt"

# ---- Repoint the aggregation so it waits for the sweeps too -----------------
if [[ -n "${AGG_JOB}" ]]; then
    SWEEP_DEP="$(IFS=:; echo "${SWEEP_IDS[*]}")"
    echo ""
    echo "Repointing aggregation ${AGG_JOB} to wait for the sweeps..."
    scontrol update JobId="${AGG_JOB}" Dependency="afterany:${DEP}:${SWEEP_DEP}" \
        || { echo "FATAL: could not update ${AGG_JOB}" >&2; exit 1; }
    scontrol show job "${AGG_JOB}" | grep -q 'Dependency=(null)' \
        && { echo "FATAL: aggregation dependency is now null" >&2; exit 1; }
    scontrol show job "${AGG_JOB}" | tr ' ' '\n' | grep '^Dependency=' | head -1
fi
