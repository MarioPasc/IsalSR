#!/usr/bin/env bash
# =============================================================================
# Campaign C2 pre-flight -- submit the Stage C certifier on its own
# =============================================================================
# Use when the Stage C arrays have drained and `--postprocess only` has already
# written aggregate.csv / the three paired contrasts / status_ledger.csv, but no
# verdict exists -- e.g. job 1753134, which TIMED OUT at its 2 h wall after the
# ~1h40 aggregation loop and never reached the certifier.
#
# Runs ON THE PICASSO LOGIN NODE.
#
# Usage:
#   bash slurm/c2_smoke/certify.sh              # build sacct CSV + submit
#   bash slurm/c2_smoke/certify.sh --dry-run    # print the sbatch command only
#   bash slurm/c2_smoke/certify.sh --no-sacct   # skip the memory profile (C1.11)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ISALSR_REPO_DIR="${ISALSR_REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR}"
RESULTS_ROOT="${C2_RESULTS_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/results/isalsr/c2_smoke}"
LOGS_DIR="${C2_LOGS_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/c2_smoke/logs}"
ACCOUNT="tic_163_uma"
# Deliberately NOT pinned, unlike launcher.sh. The certifier only reads files
# already on disk: it generates no data and produces no timed quantity, so
# neither of the two reasons for pinning the wave applies to it. Leaving it on
# any CPU node just gets the verdict back sooner.
CONSTRAINT="${C2_CONSTRAINT:-cpu}"

# The 2 h wall is what killed 1753134.  The certifier alone is far cheaper than
# the aggregation it followed, but it still walks 1,260 run logs plus 420
# paired-stats files, so give it room rather than re-discovering the limit.
WALL="${C2_CERT_WALL:-0-04:00:00}"

EXPECTED="${C2_EXPECTED:-1260}"
MAX_TIME="${C2_MAX_TIME:-900}"
CERT_SEEDS="${C2_CERT_SEEDS:-0,101,102}"

# 🔴 sbatch --export is COMMA-separated, so a comma inside a VALUE starts the
# next variable.  Exporting C2_CERT_SEEDS=0,101,102 delivers C2_CERT_SEEDS=0 and
# two junk assignments -- the same defect that cost 265 of the 1,260 Stage C
# tasks (EXECUTION-PLAN §11.1, 2026-08-03).  Here it would be quieter and worse:
# the certifier would build its expected-cell set from ONE seed, so C1.15/C2/C4
# would reconcile 420 cells against a 1,260-cell root and the verdict would be
# meaningless rather than absent.  Ship colon-separated; the worker translates
# and ASSERTS the decode rather than trusting the transport.
CERT_SEEDS_EXPORT="${CERT_SEEDS//,/:}"

DRY_RUN=false
USE_SACCT=true
for arg in "$@"; do
    case "${arg}" in
        --dry-run)  DRY_RUN=true ;;
        --no-sacct) USE_SACCT=false ;;
        *) echo "unknown argument: ${arg}" >&2; exit 1 ;;
    esac
done

mkdir -p "${LOGS_DIR}"

[[ -d "${RESULTS_ROOT}" ]] || { echo "FATAL: results root not found: ${RESULTS_ROOT}" >&2; exit 1; }

# --- C1.11 memory profile ----------------------------------------------------
# 🔴 sacct -X returns an EMPTY MaxRSS: memory is accounted on the .batch STEP,
# not on the allocation.  A profile built with -X comes back silently blank --
# a table of empty cells and no error -- which is the worst possible outcome
# when its only purpose is to size production --mem.  Hence: no -X, and filter
# to .batch.
SACCT_CSV="${LOGS_DIR}/stage_c_maxrss.csv"
if ${USE_SACCT}; then
    JOB_IDS_FILE="${LOGS_DIR}/job_ids.txt"
    if [[ -s "${JOB_IDS_FILE}" ]]; then
        echo "Building C1.11 memory profile from $(wc -l < "${JOB_IDS_FILE}") arrays..."
        # 🔴 Emit JobIDRaw, NOT JobID.  The certifier joins sacct rows onto cells
        # through status.json's `slurm_job_id`, which is $SLURM_JOB_ID -- the
        # per-task id, unique to each array element.  sacct's JobID column is
        # instead "<array_id>_<task>", which equals $SLURM_JOB_ID only for task 1
        # of each array.  Emitting JobID therefore joined exactly 42 of 1,260
        # rows (one per array) and silently binned the other 1,218 as
        # "unmatched", so C1.11 -- whose only job is to size production --mem --
        # reported a profile built from 3 % of the campaign and still said PASS.
        # JobIDRaw is the per-task numeric id and matches status.json exactly.
        {
            echo "JobID,MaxRSS"
            sacct -j "$(paste -sd, "${JOB_IDS_FILE}")" -n -P -o JobIDRaw,MaxRSS \
                | awk -F'|' '$1 ~ /\.batch$/ && $2 != "" {print $1","$2}'
        } > "${SACCT_CSV}"
        echo "  ${SACCT_CSV}: $(( $(wc -l < "${SACCT_CSV}") - 1 )) rows"
    else
        echo "[warn] ${JOB_IDS_FILE} missing -- skipping the memory profile"
        SACCT_CSV=""
    fi
else
    SACCT_CSV=""
fi

# --- submit ------------------------------------------------------------------
# Picasso's Lua sbatch wrapper prepends ANSI + a multi-line warning to
# --parsable output, so take the LAST line before stripping non-digits.  A
# line-wise sed leaves the banner's newlines in place and the "id" comes back
# multi-line -- and the guard then fires AFTER the job was already submitted.
_clean_job_id() { tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'; }
submit() {
    local raw id
    raw=$(sbatch --parsable "$@") || { echo "sbatch failed" >&2; return 1; }
    id=$(_clean_job_id "${raw}")
    [[ "${id}" =~ ^[0-9]+$ ]] || { echo "FATAL: unparsable job id: ${raw@Q}" >&2; return 1; }
    echo "${id}"
}

SBATCH_ARGS=(
    --job-name=c2s_certify
    --time="${WALL}"
    --ntasks=1 --cpus-per-task=2 --mem=16G
    --constraint="${CONSTRAINT}"
    --account="${ACCOUNT}"
    --output="${LOGS_DIR}/c2s_certify_%j.out"
    --error="${LOGS_DIR}/c2s_certify_%j.err"
    --export="ALL,ISALSR_REPO_DIR=${ISALSR_REPO_DIR},C2_RESULTS_DIR=${RESULTS_ROOT},C2_SACCT_CSV=${SACCT_CSV},C2_EXPECTED=${EXPECTED},C2_MAX_TIME=${MAX_TIME},C2_CERT_SEEDS=${CERT_SEEDS_EXPORT}"
    "${SCRIPT_DIR}/certify_worker.sh"
)

if ${DRY_RUN}; then
    echo "[DRY-RUN] sbatch --parsable ${SBATCH_ARGS[*]}"
    sbatch --test-only "${SBATCH_ARGS[@]}"
    exit 0
fi

JOB_ID=$(submit "${SBATCH_ARGS[@]}") || exit 1
echo "Submitted certification job ${JOB_ID}"
echo "Monitor:  ssh picasso 'squeue'"
echo "Log:      ${LOGS_DIR}/c2s_certify_${JOB_ID}.out"
echo "Verdict:  ${RESULTS_ROOT}/c2_preflight/stage_c_certification.md"
