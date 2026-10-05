#!/usr/bin/env bash
# =============================================================================
# SRBench black-box campaign (R3.2) -- launcher.  Run on the Picasso login node.
# =============================================================================
#   20 datasets x 10 seeds x 3 arms x 2 hosts = 1,200 cells
#
# Topology: one array per row of packaging.tsv.  A row is a (method, arm) pair,
# optionally restricted to a subset of datasets, with its own bundle (cells per
# task), wall and throttle; packaging.tsv documents how each number was derived.
# Each array task runs a contiguous chunk of (dataset, seed) cells through
# worker.sh -> slurm/c2_smoke/worker.sh, i.e. C2's certified worker, under C2's
# deadline rule: a cell starts only if its full 12 h budget still fits in the
# wall, so a SLURM TIMEOUT is impossible by construction and every cell gets an
# identical budget.  Cells a chunk could not start are picked up by a sweep
# array (same partition, afterany), exactly as in C2.
#
# Usage (from $FSCRATCH/repos/IsalSR_bbx on the login node):
#   bash slurm/blackbox/launcher.sh --dry-run     # print every sbatch command
#   bash slurm/blackbox/launcher.sh --smoke       # 6 cells: banana seed 1, 6 arrays x 1 task, 120 s
#   bash slurm/blackbox/launcher.sh               # the 1,200-cell campaign
# --dry-run also works on the workstation (BBX_ROOT=<repo>/code, BBX_PYTHON=...).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FSCRATCH="${ISALSR_FSCRATCH:-/mnt/home/users/tic_163_uma/mpascual/fscratch}"
BBX_ROOT="${BBX_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
C2_TREE="${BBX_C2_TREE:-${FSCRATCH}/repos/IsalSR}"
ACCOUNT="${BBX_ACCOUNT:-tic_163_uma}"
CONSTRAINT="${BBX_CONSTRAINT:-sr}"        # AMD EPYC, the C2 node class (timing + data bit-identity)
SEEDS="1-10"                              # PLAN D7: SRBench's 10 trials; seed = split + host RNG
# Partial re-submission after an sbatch refusal: point BBX_PACKAGING at a copy
# of packaging.tsv holding only the rows still to submit, and set
# BBX_EXPECTED_CELLS to their cell count.  Both default to the full campaign.
EXPECTED_CELLS="${BBX_EXPECTED_CELLS:-1200}"
PACKAGING="${BBX_PACKAGING:-${SCRIPT_DIR}/packaging.tsv}"
TEARDOWN_S=1800                           # C2 CELL_RESERVE_H - PAYLOAD_CAP_H = 0.5 h

MODE="submit"
case "${1:-}" in
    --dry-run) MODE="dry" ;;
    --smoke)   MODE="smoke" ;;
    "")        ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
esac

if [[ "${MODE}" == "smoke" ]]; then
    MAX_TIME=120
    ROOT_NAME="srbench_blackbox_smoke"
else
    MAX_TIME=43200                        # PLAN D7: 12 h, as C2
    ROOT_NAME="srbench_blackbox"
fi
RESULTS_ROOT="${BBX_RESULTS_DIR:-${FSCRATCH}/results/isalsr/${ROOT_NAME}}"
LOGS_DIR="${BBX_LOGS_DIR:-${FSCRATCH}/execs/isalsr/${ROOT_NAME}/logs}"

# ---- Python with the black-box import shim ----------------------------------
if [[ -z "${BBX_PYTHON:-}" ]]; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate isalsr 2>/dev/null || true
    BBX_PYTHON="${CONDA_PREFIX:-}/bin/python"
fi
export ISALSR_BBX_ROOT="${BBX_ROOT}"
export PYTHONPATH="${BBX_ROOT}/slurm/blackbox/pyboot${PYTHONPATH:+:${PYTHONPATH}}"
py() { (cd "${BBX_ROOT}" && "${BBX_PYTHON}" "$@"); }

config_for() { echo "${BBX_ROOT}/experiments/configs/blackbox/$1_srbench_blackbox.yaml"; }

# Picasso's Lua sbatch wrapper prepends ANSI + a banner to --parsable output (C2).
_clean_job_id() { tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'; }
submit() {
    local raw id
    raw=$(sbatch --parsable "$@") || { echo "sbatch failed" >&2; return 1; }
    id=$(_clean_job_id "${raw}")
    [[ "${id}" =~ ^[0-9]+$ ]] || { echo "FATAL: unparsable job id: ${raw@Q}" >&2; return 1; }
    echo "${id}"
}
wall_to_s() { awk -F'[-:]' '{print (($1*24)+$2)*3600 + $3*60 + $4}' <<<"$1"; }

# ---- Gate + provenance stamp (not in --dry-run: the cluster paths must exist) --
if [[ "${MODE}" != "dry" ]]; then
    mkdir -p "${LOGS_DIR}" "${RESULTS_ROOT}"
    py "${BBX_ROOT}/slurm/blackbox/check_env.py" --c2-tree "${C2_TREE}" --bbx-root "${BBX_ROOT}" \
        --write-stamp "${RESULTS_ROOT}/bbx_provenance.json" \
        || { echo "FATAL: BBX gate failed on the login node; nothing submitted" >&2; exit 1; }
fi

echo "BBX launcher -- mode ${MODE}"
echo "  bbx root:   ${BBX_ROOT}"
echo "  C2 tree:    ${C2_TREE}"
echo "  results:    ${RESULTS_ROOT}"
echo "  logs:       ${LOGS_DIR}"
echo "  seeds:      ${SEEDS}   max_time: ${MAX_TIME}s   constraint: ${CONSTRAINT}"
echo ""

JOB_IDS=()
SWEEP_ROWS=()
declare -A SMOKE_SEEN=()
N_CELLS_TOTAL=0
N_TASKS_TOTAL=0
N_ARRAYS=0

while IFS=$'\t' read -r METHOD ARM PROBLEMS BUNDLE WALL MEM THROTTLE TAG; do
    [[ -z "${METHOD}" || "${METHOD}" == \#* ]] && continue
    CONFIG="$(config_for "${METHOD}")"
    [[ -f "${CONFIG}" ]] || { echo "FATAL: missing ${CONFIG}" >&2; exit 1; }

    if [[ "${MODE}" == "smoke" ]]; then
        # One cell per (method, arm): the largest dataset, seed 1.  The C2
        # worker refuses a single-seed spec, so seeds stay "1-2" and the array
        # is cut to task 1 = (banana, seed 1) at bundle 1.
        [[ -n "${SMOKE_SEEN[${METHOD}:${ARM}]:-}" ]] && continue
        SMOKE_SEEN[${METHOD}:${ARM}]=1
        SEED_SPEC="1-2"; PROBLEMS="banana"; BUNDLE=1; WALL="0-00:30:00"; THROTTLE=1
        TAG="smoke"
    else
        SEED_SPEC="${SEEDS}"
    fi

    SPEC_ARGS=(--config "${CONFIG}" --seeds "${SEED_SPEC}" --count)
    [[ "${PROBLEMS}" != "all" ]] && SPEC_ARGS+=(--problems "${PROBLEMS//:/,}")
    N_CELLS=$(py -m experiments.scripts.c2_task_spec "${SPEC_ARGS[@]}" --bundle 1)
    N_TASKS=$(py -m experiments.scripts.c2_task_spec "${SPEC_ARGS[@]}" --bundle "${BUNDLE}")
    if [[ "${MODE}" == "smoke" ]]; then N_CELLS=1; N_TASKS=1; fi

    WALL_S=$(wall_to_s "${WALL}")
    CUTOFF_S=$(( WALL_S - MAX_TIME - TEARDOWN_S ))
    (( CUTOFF_S < 1 )) && CUTOFF_S=1
    JOB_NAME="bbx_${METHOD:0:1}${ARM:0:1}_${TAG}"
    PROBLEMS_EXPORT=""
    [[ "${PROBLEMS}" != "all" ]] && PROBLEMS_EXPORT="${PROBLEMS//,/:}"
    EXPORTS="ALL,ISALSR_REPO_DIR=${BBX_ROOT},BBX_C2_TREE=${C2_TREE},C2_METHOD=${METHOD},C2_ARM=${ARM},C2_SUITE=srbench_blackbox,C2_CONFIG=${CONFIG},C2_SEEDS=${SEED_SPEC//,/:},C2_MAX_TIME=${MAX_TIME},C2_RESULTS_DIR=${RESULTS_ROOT},C2_BUNDLE=${BUNDLE},C2_START_CUTOFF_S=${CUTOFF_S},C2_USE_LOCALSCRATCH=1,C2_PROBLEMS=${PROBLEMS_EXPORT}"
    SB_ARGS=(
        --array="1-${N_TASKS}%${THROTTLE}"
        --job-name="${JOB_NAME}"
        --time="${WALL}"
        --ntasks=1 --cpus-per-task=1
        --mem="${MEM}G"
        --constraint="${CONSTRAINT}"
        --account="${ACCOUNT}"
        --output="${LOGS_DIR}/${JOB_NAME}_%A_%a.out"
        --export="${EXPORTS}"
        "${SCRIPT_DIR}/worker.sh"
    )

    N_ARRAYS=$((N_ARRAYS + 1))
    N_CELLS_TOTAL=$((N_CELLS_TOTAL + N_CELLS))
    N_TASKS_TOTAL=$((N_TASKS_TOTAL + N_TASKS))
    case "${MODE}" in
        dry)
            printf '  %-18s %-8s %4d cells / B=%-2d = %3d tasks  %%%-4d %3sG  wall %s  cutoff %ss\n' \
                "${JOB_NAME}" "${ARM}" "${N_CELLS}" "${BUNDLE}" "${N_TASKS}" "${THROTTLE}" \
                "${MEM}" "${WALL}" "${CUTOFF_S}"
            printf '    sbatch'; printf ' %q' "${SB_ARGS[@]}"; printf '\n'
            ;;
        smoke|submit)
            ID=$(submit "${SB_ARGS[@]}") || exit 1
            JOB_IDS+=("${ID}")
            printf '  %-18s %4d tasks (B=%d) %%%-4d %3sG  job %s\n' \
                "${JOB_NAME}" "${N_TASKS}" "${BUNDLE}" "${THROTTLE}" "${MEM}" "${ID}"
            (( BUNDLE > 1 )) && SWEEP_ROWS+=("${JOB_NAME}|${N_TASKS}|${THROTTLE}|${WALL}|${MEM}|${EXPORTS}")
            ;;
    esac
done < "${PACKAGING}"

echo ""
echo "Arrays: ${N_ARRAYS}   tasks: ${N_TASKS_TOTAL}   cells: ${N_CELLS_TOTAL}"
if [[ "${MODE}" != "smoke" && "${N_CELLS_TOTAL}" -ne "${EXPECTED_CELLS}" ]]; then
    echo "FATAL: packaging covers ${N_CELLS_TOTAL} cells, expected ${EXPECTED_CELLS}" >&2
    [[ "${MODE}" == "submit" ]] && echo "       (arrays already submitted: ${JOB_IDS[*]:-none}; scancel them)" >&2
    exit 1
fi

# ---- Sweep arrays: the deadline's counterpart (C2 launcher.sh §sweep) -------
# Only rows with B > 1 can defer a cell.  Same partition (same n_cells, n_tasks),
# held behind afterany on every main array; a task whose cells all completed
# exits in seconds because the orchestrator skips cells with a valid run_log.
if [[ "${MODE}" == "submit" && ${#SWEEP_ROWS[@]} -gt 0 ]]; then
    DEP=$(IFS=:; echo "${JOB_IDS[*]}")
    SWEEP_IDS=()
    for row in "${SWEEP_ROWS[@]}"; do
        IFS='|' read -r NAME N_TASKS THROTTLE WALL MEM EXPORTS <<<"${row}"
        SID=$(submit --array="1-${N_TASKS}%${THROTTLE}" --job-name="${NAME/bbx_/bbxw_}" \
              --time="${WALL}" --ntasks=1 --cpus-per-task=1 --mem="${MEM}G" \
              --constraint="${CONSTRAINT}" --account="${ACCOUNT}" \
              --dependency="afterany:${DEP}" \
              --output="${LOGS_DIR}/${NAME/bbx_/bbxw_}_%A_%a.out" \
              --export="${EXPORTS}" "${SCRIPT_DIR}/worker.sh") || exit 1
        if scontrol show job "${SID}" | grep -q 'Dependency=(null)'; then
            echo "FATAL: sweep dependency dropped -- cancelling ${SID}" >&2; scancel "${SID}"; exit 1
        fi
        SWEEP_IDS+=("${SID}")
    done
    echo "Sweep arrays: ${#SWEEP_IDS[@]} (afterany on ${#JOB_IDS[@]} main arrays)"
    printf '%s\n' "${SWEEP_IDS[@]}" > "${LOGS_DIR}/sweep_job_ids.txt"
fi
if [[ "${MODE}" != "dry" ]]; then
    printf '%s\n' "${JOB_IDS[@]}" > "${LOGS_DIR}/job_ids.txt"
    echo "Job ids -> ${LOGS_DIR}/job_ids.txt"
fi
