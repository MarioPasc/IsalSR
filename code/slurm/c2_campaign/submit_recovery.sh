#!/usr/bin/env bash
# =============================================================================
# Campaign C2 -- RECOVERY pass over the cells the start-deadline deferred
# =============================================================================
# Why this exists, and why it is not `submit_sweeps.sh` run twice.
#
# A chunked task starts a cell only if the cell's FULL payload budget still fits
# in its remaining wall (`worker.sh:450`).  That is what makes a SLURM TIMEOUT
# impossible by construction, and it is correct.  Its cost is that a chunk which
# draws an unlucky run of full-budget cells leaves its tail unstarted, exits
# COMPLETED, and is therefore invisible to `sacct`.  The 2026-08-09 audit found
# ~512 such cells at 83 % of the campaign, projecting to ~1,100-1,200.
#
# The sweep arrays re-run the SAME bundle, wall and deadline, so each pass only
# advances a task by however many cells fit inside that same deadline.  For
# `udfs:*:feynman` (B=27, 34.5 h deadline, tail cells at 12 h) that is ~3 cells
# per pass, i.e. ~9 passes and ~12 days to clear the array by repetition.
#
# This script clears it in ONE pass instead, by shrinking the bundle until the
# deadline provably cannot bite:
#
#     wall = (B-1) * allowance + CELL_RESERVE,   deadline = wall - CELL_RESERVE
#     => the last cell of the chunk starts at worst (B-1) * allowance < deadline
#
# with `allowance = 12.5 h` (the 12 h payload cap plus the teardown allowance)
# in the default `safe` mode.  Charging every cell the CAP is what removes the
# distributional assumption: the 2026-08-09 measurement showed `udfs:feynman`
# spanning 67x inside one suite (median 0.18 h, p90 12.00 h), so any bundle
# sized on a central estimate is a bet, and this pass is not the place to bet.
# `safe` gives B=3 at a 38 h wall for every array.  `--mode p90` charges the
# measured p90 instead, which gives B=63 on `bingo:feynman` and ~4x fewer SLURM
# placements, at the cost of a merely probabilistic guarantee.
#
# 🔴 PROVENANCE.  Acceptance criterion 6 requires every `run_log.json` to report
# `git_describe: campaign/c2`.  So this script NEVER redeploys and never runs the
# worker from its own checkout.  It splits the two trees explicitly:
#
#   C2_TOOLS_DIR  -- where the PLAN is computed (this checkout; needs the
#                    `--recovery` mode added after the campaign was deployed)
#   ISALSR_REPO_DIR -- what the WORKER and the payload run from (the DEPLOYED
#                    tree, untouched, still at the campaign's tag)
#
# Only numbers cross that boundary: bundle, wall, deadline, task count.  The
# script refuses to submit if the two trees disagree about any file whose
# semantics the two halves must share (the configs, and `c2_task_spec.py`, which
# owns the partition).
#
# 🔴 JOB NAMES.  `c2s_*` (main) and `c2w_*` (sweep) are built by the SAME
# expression in launcher.sh, submit_paced.sh and submit_sweeps.sh, which is how a
# Stage C smoke could mask a whole campaign (§11.1, 2026-08-07).  This pass uses
# a THIRD prefix, `c2r_`, so it can neither mask nor be masked by either.
#
#   bash submit_recovery.sh --only 'udfs:*:feynman' --dry-run
#   bash submit_recovery.sh --only "$(python -m experiments.scripts.c2_missing_cells \
#            --results-dir $FSCRATCH/results/isalsr/c2_3arm --selectors)"
# =============================================================================
set -uo pipefail

usage() {
    cat <<'EOF'
usage: submit_recovery.sh --only <selectors> [options]

  --only SEL          REQUIRED. Comma-separated method:arm:suite selectors,
                      '*' wildcards a field, e.g. 'udfs:*:feynman,bingo:*:nguyen'.
                      Get it from c2_missing_cells.py --selectors.
  --mode safe|p90     Bundle sizing (default safe). See the header.
  --bundle N          Force the chunk size. Refused if it could still defer.
  --results-dir DIR   Results root (default: the campaign root).
  --logs-dir DIR      Log dir (default: <campaign logs>/../logs_recovery).
  --budget N          Total concurrent array slots (default 2016).
  --dependency SPEC   e.g. afterany:123:456. Validated, and the submission is
                      cancelled if SLURM records Dependency=(null).
  --with-aggregation  Also chain the aggregation array + status ledger
                      afterany on the recovery arrays, with the CELL count
                      (12600) as C2_EXPECTED_TASKS.
  --allow-live-campaign
                      Submit even though c2s_*/c2w_* jobs are still queued.
                      Only for a throwaway results root: two passes over the
                      same cell race on the per-cell copy-back.
  --dry-run           Print what would be submitted, submit nothing.
  -h, --help          This.
EOF
}

# ---- defaults ---------------------------------------------------------------
FSCRATCH="${ISALSR_FSCRATCH:-/mnt/home/users/tic_163_uma/mpascual/fscratch}"
# The DEPLOYED tree: what the worker runs, and what fixes the provenance tag.
REPO="${ISALSR_REPO_DIR:-${FSCRATCH}/repos/IsalSR}"
# THIS checkout: where the recovery-aware planner lives.  Resolved from the
# script's own location so a copy of the repo anywhere works without an env var.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="${C2_TOOLS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

RESULTS_ROOT="${C2_RESULTS_DIR:-${FSCRATCH}/results/isalsr/c2_3arm}"
LOGS_DIR="${C2_LOGS_DIR:-${FSCRATCH}/execs/isalsr/c2_3arm/logs_recovery}"
ACCOUNT="${C2_ACCOUNT:-tic_163_uma}"
CONSTRAINT="${C2_CONSTRAINT:-sr}"
SEEDS="${C2_SEEDS_SPEC:-1-30}"
MAX_TIME="${C2_MAX_TIME:-43200}"
SLOT_BUDGET="${C2_SLOT_BUDGET:-2016}"
USE_LOCALSCRATCH="${C2_USE_LOCALSCRATCH:-1}"
SLEEP_BETWEEN="${C2_SLEEP:-20}"
# Total CELLS in the campaign.  Passed to the certifier so it uses the registry
# universe rather than falling through to the self-referential "disk" one --
# launcher.sh:625 passed the TASK count and that is exactly what happened.
EXPECTED_CELLS="${C2_EXPECTED_TASKS:-12600}"
MIN_JOBID="${C2_MIN_JOBID:-0}"

ONLY="${C2_RECOVERY_ONLY:-}"
MODE="${C2_RECOVERY_MODE:-safe}"
BUNDLE="${C2_RECOVERY_BUNDLE:-}"
DEPENDENCY=""
WITH_AGG=false
ALLOW_LIVE=false
DRY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)        ONLY="${2:?--only needs a value}"; shift 2 ;;
        --mode)        MODE="${2:?--mode needs a value}"; shift 2 ;;
        --bundle)      BUNDLE="${2:?--bundle needs a value}"; shift 2 ;;
        --results-dir) RESULTS_ROOT="${2:?--results-dir needs a value}"; shift 2 ;;
        --logs-dir)    LOGS_DIR="${2:?--logs-dir needs a value}"; shift 2 ;;
        --budget)      SLOT_BUDGET="${2:?--budget needs a value}"; shift 2 ;;
        --dependency)  DEPENDENCY="${2:?--dependency needs a value}"; shift 2 ;;
        --with-aggregation) WITH_AGG=true; shift ;;
        --allow-live-campaign) ALLOW_LIVE=true; shift ;;
        --dry-run)     DRY=true; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "FATAL: unknown argument '$1'" >&2; usage >&2; exit 1 ;;
    esac
done

[[ -n "${ONLY}" ]] || {
    echo "FATAL: --only is required.  A recovery pass must name the arrays it" >&2
    echo "       covers; resubmitting all 42 is what this script exists to avoid." >&2
    echo "       Derive it:  python <tools>/experiments/scripts/c2_missing_cells.py \\" >&2
    echo "                     --results-dir ${RESULTS_ROOT} --selectors" >&2
    exit 1; }

command -v sbatch >/dev/null 2>&1 || {
    echo "FATAL: no sbatch here. Run this ON Picasso." >&2; exit 1; }

# ---- fail closed on a malformed dependency BEFORE anything is submitted -----
# sbatch ACCEPTS a syntactically odd dependency and records Dependency=(null);
# the array then starts immediately, against a tree still being written.  Two
# guards: the string must look like a dependency, and every submitted job is
# re-read to confirm SLURM kept it.
if [[ -n "${DEPENDENCY}" ]]; then
    if [[ ! "${DEPENDENCY}" =~ ^after(any|ok|notok|corr):[0-9][0-9:_]*$ ]]; then
        echo "FATAL: --dependency '${DEPENDENCY}' is not after{any,ok,notok,corr}:<ids>" >&2
        exit 1
    fi
fi

# ---- §5.0: do not overlay a second topology on a live campaign --------------
# Resume is additive, but two tasks running the SAME cell concurrently race on
# the per-cell copy-back.  The main and sweep passes own the campaign root until
# they drain, so refuse by default.
LIVE=$(/usr/bin/squeue -u "${USER}" -h -o "%j" 2>/dev/null \
        | grep -c -E '^c2[sw]_' || true)
if [[ "${LIVE}" -gt 0 ]]; then
    if ${ALLOW_LIVE}; then
        echo "⚠ ${LIVE} c2s_/c2w_ job(s) still queued; --allow-live-campaign given."
        echo "  This is only safe against a THROWAWAY results root."
        echo "  results: ${RESULTS_ROOT}"
    else
        cat >&2 <<EOF
FATAL: ${LIVE} c2s_*/c2w_* job(s) are still queued or running.

  A recovery pass over the same root as a live pass can put two tasks on the
  same cell, and they then race on the per-cell copy-back.  Wait for the main
  arrays AND the sweeps to drain (bash slurm/c2_campaign/health.sh), then re-run.

  --allow-live-campaign overrides this, and is meant only for a test against a
  throwaway results root.
EOF
        exit 1
    fi
fi

# ---- the two trees must agree on the things they share ----------------------
[[ -d "${REPO}" ]]  || { echo "FATAL: worker tree ${REPO} not found" >&2; exit 1; }
[[ -d "${TOOLS}" ]] || { echo "FATAL: tools tree ${TOOLS} not found" >&2; exit 1; }
[[ -x "${REPO}/slurm/c2_smoke/worker.sh" || -f "${REPO}/slurm/c2_smoke/worker.sh" ]] \
    || { echo "FATAL: ${REPO}/slurm/c2_smoke/worker.sh missing" >&2; exit 1; }

# `c2_task_spec.py` owns the partition.  The plan is computed with the TOOLS
# copy and decoded at run time with the WORKER copy; if they differ, a task can
# be handed cells the plan never sized it for.
if ! cmp -s "${TOOLS}/experiments/scripts/c2_task_spec.py" \
            "${REPO}/experiments/scripts/c2_task_spec.py"; then
    echo "FATAL: c2_task_spec.py differs between the tools tree and the deployed" >&2
    echo "       tree.  The plan and the decode would disagree about which cells" >&2
    echo "       a task owns.  Refusing." >&2
    exit 1
fi

PY="${FSCRATCH}/conda_envs/isalsr/bin/python"
[[ -x "${PY}" ]] || PY="$(command -v python3)"

PLAN_ARGS=(--config-dir "${TOOLS}/experiments/configs" --seeds "${SEEDS}"
           --budget "${SLOT_BUDGET}" --recovery --recovery-mode "${MODE}"
           --only "${ONLY}")
[[ -n "${BUNDLE}" ]] && PLAN_ARGS+=(--bundle "${BUNDLE}")

# 🔴 Invoked BY FILE PATH, not `python -m`.  Measured on Picasso 2026-08-09:
# `experiments` is a NAMESPACE package, and the conda env's editable install
# registers a meta-path finder that contributes the DEPLOYED checkout to
# `experiments.__path__` AHEAD of everything PYTHONPATH adds.  So
# `python -m experiments.scripts.c2_slot_plan` from this checkout loads the
# DEPLOYED planner -- which predates `--recovery` and dies with "unrecognized
# arguments".  Running the file directly makes __main__ come from this tree.
#
# Its imports (`c2_task_spec`, `experiments.models.orchestrator`) still resolve
# to the deployed tree, and that is the RIGHT outcome, not a compromise: the plan
# is then sized against the same registry and the same partition function the
# worker will decode with.  The `cmp` guards above are what make that safe.
PLAN="$("${PY}" "${TOOLS}/experiments/scripts/c2_slot_plan.py" "${PLAN_ARGS[@]}" --tsv)" \
    || { echo "FATAL: recovery plan failed; nothing submitted." >&2
         echo "       If it said 'unrecognized arguments: --recovery', the tools" >&2
         echo "       tree ${TOOLS} predates the recovery mode." >&2
         exit 1; }
[[ -n "${PLAN}" ]] || { echo "FATAL: recovery plan is empty" >&2; exit 1; }
[[ "$(awk -F'\t' 'NR==1{print NF}' <<<"${PLAN}")" -eq 10 ]] \
    || { echo "FATAL: plan is not 10 columns -- stale checkout?" >&2; exit 1; }

# The configs are the OTHER shared artefact: the plan sizes an array from the
# tools copy, the worker expands the suite from the deployed copy.
while IFS=$'\t' read -r METHOD _ SUITE _ _ _ _ _ _ _; do
    [[ -n "${METHOD}" ]] || continue
    CFG="experiments/configs/${METHOD}_${SUITE}.yaml"
    cmp -s "${TOOLS}/${CFG}" "${REPO}/${CFG}" || {
        echo "FATAL: ${CFG} differs between the tools tree and the deployed tree." >&2
        exit 1; }
done <<< "${PLAN}"

${DRY} || mkdir -p "${LOGS_DIR}" "${RESULTS_ROOT}"

# ---- idempotence ------------------------------------------------------------
# By job name, like submit_paced.sh, but the `c2r_` prefix cannot collide with
# the campaign or the smoke, so the default floor of 0 is safe here (it is NOT
# in submit_paced.sh, where c2s_ is built by both profiles).  What it does catch
# is a re-run after a partial submission, which is the behaviour we want.
EXISTING="$(sacct -S today -n -P -X -o JobID,JobName 2>/dev/null \
    | awk -F'|' -v m="${MIN_JOBID}" \
        '{split($1,a,"_"); if (a[1]+0 >= m && $2 ~ /^c2r_/ && $2 !~ /aggregate|ledger/) print $2}' \
    | sort -u)"

echo "Campaign C2 -- RECOVERY pass"
echo "  worker tree (provenance) : ${REPO}"
echo "  tools tree  (plan only)  : ${TOOLS}"
echo "  results                  : ${RESULTS_ROOT}"
echo "  logs                     : ${LOGS_DIR}"
echo "  arrays                   : ${ONLY}"
echo "  sizing                   : mode=${MODE}${BUNDLE:+ bundle=${BUNDLE}} seeds=${SEEDS}"
echo "  already present (id >= ${MIN_JOBID}): $(grep -c . <<<"${EXISTING}" || true)"
echo ""

declare -a IDS=()
_clean_job_id() { tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'; }

while IFS=$'\t' read -r METHOD ARM SUITE N_TASKS THROTTLE MEM WALL \
                        BUNDLE_R CUTOFF_S N_CELLS; do
    [[ -n "${METHOD}" ]] || continue
    JOB_NAME="c2r_${METHOD:0:1}${ARM:0:1}_${SUITE}"

    if grep -qx "${JOB_NAME}" <<<"${EXISTING}"; then
        printf '  %-34s SKIP (already submitted today)\n' "${JOB_NAME}"
        continue
    fi
    if ${DRY}; then
        printf '  %-34s %4d cells /%2d = %4d tasks  %%%-4d %3dG  %s  cutoff %ss\n' \
               "${JOB_NAME}" "${N_CELLS}" "${BUNDLE_R}" "${N_TASKS}" \
               "${THROTTLE}" "${MEM}" "${WALL}" "${CUTOFF_S}"
        continue
    fi

    SBATCH_ARGS=(--parsable
        --array="1-${N_TASKS}%${THROTTLE}"
        --job-name="${JOB_NAME}"
        --time="${WALL}"
        --ntasks=1 --cpus-per-task=1
        --mem="${MEM}G"
        --constraint="${CONSTRAINT}"
        --account="${ACCOUNT}"
        --output="${LOGS_DIR}/${JOB_NAME}_%A_%a.out"
        --export="ALL,ISALSR_REPO_DIR=${REPO},C2_METHOD=${METHOD},C2_ARM=${ARM},C2_SUITE=${SUITE},C2_CONFIG=${REPO}/experiments/configs/${METHOD}_${SUITE}.yaml,C2_SEEDS=${SEEDS},C2_MAX_TIME=${MAX_TIME},C2_RESULTS_DIR=${RESULTS_ROOT},C2_BUNDLE=${BUNDLE_R},C2_START_CUTOFF_S=${CUTOFF_S},C2_USE_LOCALSCRATCH=${USE_LOCALSCRATCH}")
    [[ -n "${DEPENDENCY}" ]] && SBATCH_ARGS+=(--dependency="${DEPENDENCY}")

    RAW=$(sbatch "${SBATCH_ARGS[@]}" "${REPO}/slurm/c2_smoke/worker.sh" 2>&1)
    ID=$(_clean_job_id "${RAW}")

    if [[ ! "${ID}" =~ ^[0-9]+$ ]]; then
        printf '  %-34s FAILED: %s\n' "${JOB_NAME}" "$(tail -n 1 <<<"${RAW}")"
        echo "RE-RUN this script to submit the rest -- it skips what already exists." >&2
        [[ ${#IDS[@]} -gt 0 ]] && printf '%s\n' "${IDS[@]}" >> "${LOGS_DIR}/recovery_job_ids.txt"
        exit 1
    fi
    if [[ -n "${DEPENDENCY}" ]] && scontrol show job "${ID}" | grep -q 'Dependency=(null)'; then
        echo "FATAL: dependency dropped on ${ID} -- cancelling it and stopping" >&2
        scancel "${ID}"
        [[ ${#IDS[@]} -gt 0 ]] && printf '%s\n' "${IDS[@]}" >> "${LOGS_DIR}/recovery_job_ids.txt"
        exit 1
    fi

    IDS+=("${ID}")
    printf '  %-34s %4d cells /%2d = %4d tasks  %%%-4d %3dG  %s  job %s\n' \
           "${JOB_NAME}" "${N_CELLS}" "${BUNDLE_R}" "${N_TASKS}" \
           "${THROTTLE}" "${MEM}" "${WALL}" "${ID}"
    sleep "${SLEEP_BETWEEN}"
done <<< "${PLAN}"

${DRY} && { echo ""; echo "[DRY-RUN] nothing submitted."; exit 0; }

[[ ${#IDS[@]} -gt 0 ]] || { echo "No recovery arrays submitted."; exit 0; }
printf '%s\n' "${IDS[@]}" >> "${LOGS_DIR}/recovery_job_ids.txt"
echo ""
echo "Recovery arrays: ${#IDS[@]} -> ${LOGS_DIR}/recovery_job_ids.txt"

# ---- optional: re-run aggregation + ledger over the completed tree ----------
${WITH_AGG} || exit 0

DEP="$(IFS=:; echo "${IDS[*]}")"
CONFIG_LIST=""
for m in udfs bingo; do
    for s in nguyen feynman hard cherrypicked roundoff feynman_remainder strogatz; do
        CONFIG_LIST="${CONFIG_LIST}${REPO}/experiments/configs/${m}_${s}.yaml "
    done
done
CONFIG_LIST="${CONFIG_LIST% }"

AGG_RAW=$(sbatch --parsable \
    --job-name=c2r_aggregate --array="1-42" --time="0-01:59:00" \
    --ntasks=1 --cpus-per-task=2 --mem=16G \
    --constraint="${CONSTRAINT}" --account="${ACCOUNT}" \
    --dependency="afterany:${DEP}" \
    --output="${LOGS_DIR}/c2r_aggregate_%A_%a.out" \
    --error="${LOGS_DIR}/c2r_aggregate_%A_%a.err" \
    --export="ALL,ISALSR_REPO_DIR=${REPO},C2_RESULTS_DIR=${RESULTS_ROOT},C2_CONFIG_LIST=${CONFIG_LIST}" \
    "${REPO}/slurm/c2_smoke/aggregate_worker.sh" 2>&1)
AGG_ID=$(_clean_job_id "${AGG_RAW}")
[[ "${AGG_ID}" =~ ^[0-9]+$ ]] || { echo "FATAL: aggregation not submitted: ${AGG_RAW}" >&2; exit 1; }
scontrol show job "${AGG_ID}" | grep -q 'Dependency=(null)' \
    && { echo "FATAL: aggregation dependency dropped -- cancelling" >&2; scancel "${AGG_ID}"; exit 1; }
echo "Aggregation array ${AGG_ID} (42 tasks, afterany on ${#IDS[@]} recovery arrays)"

LED_RAW=$(sbatch --parsable \
    --job-name=c2r_ledger --time="0-02:00:00" \
    --ntasks=1 --cpus-per-task=1 --mem=16G \
    --constraint="${CONSTRAINT}" --account="${ACCOUNT}" \
    --dependency="afterany:${AGG_ID}" \
    --output="${LOGS_DIR}/c2r_ledger_%j.out" \
    --error="${LOGS_DIR}/c2r_ledger_%j.err" \
    --export="ALL,ISALSR_REPO_DIR=${REPO},C2_RESULTS_DIR=${RESULTS_ROOT},C2_LEDGER_ONLY=1,C2_EXPECTED_TASKS=${EXPECTED_CELLS},C2_MAX_TIME=${MAX_TIME}" \
    "${REPO}/slurm/c2_smoke/aggregate_worker.sh" 2>&1)
LED_ID=$(_clean_job_id "${LED_RAW}")
[[ "${LED_ID}" =~ ^[0-9]+$ ]] || { echo "FATAL: ledger not submitted: ${LED_RAW}" >&2; exit 1; }
scontrol show job "${LED_ID}" | grep -q 'Dependency=(null)' \
    && { echo "FATAL: ledger dependency dropped -- cancelling" >&2; scancel "${LED_ID}"; exit 1; }
echo "Status-ledger job ${LED_ID} (afterany on ${AGG_ID}, C2_EXPECTED_TASKS=${EXPECTED_CELLS})"
printf '%s\n%s\n' "${AGG_ID}" "${LED_ID}" >> "${LOGS_DIR}/recovery_job_ids_aggregation.txt"
