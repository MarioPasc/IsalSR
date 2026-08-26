#!/usr/bin/env bash
# =============================================================================
# T07 norm-removal study — Picasso launcher
# =============================================================================
#
# Two-arm study: keep (production normalize_const_creation inside C++ canon)
# vs drop (fast_canonical_string_raw, no normalization).
#
# Populations
#   synthetic   Random S2D DAGs.   CPU-only, 20 tasks, 1.5h, 8 CPUs, 16G.
#   adversarial Constructed DAGs.  CPU-only, 1 task,   0.5h, 4 CPUs,  8G.
#   bingo       Live Bingo search. CPU-only, 15 tasks, 2h,   8 CPUs, 64G.
#   udfs        Live UDFS search.  CPU-only, 15 tasks, 2h,   8 CPUs, 32G.
#
# Usage:
#   bash slurm/t07_norm_removal_launch.sh               # all populations
#   bash slurm/t07_norm_removal_launch.sh --dry-run     # print sbatch, no submit
#   bash slurm/t07_norm_removal_launch.sh --population synthetic
#   bash slurm/t07_norm_removal_launch.sh --population bingo
#   bash slurm/t07_norm_removal_launch.sh --population udfs
#   bash slurm/t07_norm_removal_launch.sh --population adversarial
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/workers/t07_norm_removal_slurm.sh"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR"
RESULTS_ROOT="${T07_RESULTS_ROOT:-/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/t07_norm_removal}"
LOGS_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/isalsr/logs/t07_norm_removal"
ACCOUNT="tic_163_uma"
CONSTRAINT="cpu"
MAX_CONCURRENT=8

# Synthetic: one array job; each task processes N_SYNTHETIC_PER_TASK strings.
# Sizing measured, not guessed: the local smoke ran 2,000 DAGs in 96.6 s
# (48.3 ms/DAG, both arms + 1-in-50 equivariance at K=8), so 25,000/task is
# ~20 min locally and ~35 min on a slower Picasso core, against 1.5 h.
# 20 x 25,000 = 500,000 DAGs, both arms. That is far more than the claim needs:
# a zero-failure observation over ~10^4 equivariance tests already gives a
# Wilson 95% upper bound near 3x10^-4, and T15 separately covered 12.4 M real
# DAGs. Sized for a fast, reliable answer rather than maximum N.
N_SYNTHETIC_TASKS=20
N_SYNTHETIC_PER_TASK=25000

# Live search: one task per (problem, seed) pair
BINGO_PROBLEMS=(
    "nguyen:Nguyen-5"
    "feynman:I.6.20a"
    "hard:Pagie-1"
    "cherrypicked:Keijzer-11"
    "roundoff:R1"
)
UDFS_PROBLEMS=(
    "nguyen:Nguyen-5"
    "feynman:I.6.20a"
    "hard:Pagie-1"
    "cherrypicked:Keijzer-11"
    "roundoff:R1"
)
N_SEEDS=3
# 30 min of Bingo already yields ~10^6 candidate DAGs — ample to detect any
# arm disagreement, since a single disagreeing DAG falsifies the claim.
BINGO_MAX_TIME=1800
# UDFS checks its own max_time ONLY between order-enumeration stages, so it
# overshoots badly: T15 measured a 20 s budget running 900 s (45x). A 10 min
# budget against a 2 h wallclock leaves a 12x overshoot margin.
UDFS_MAX_TIME=600

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
DRY_RUN=false
SINGLE_POPULATION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=true; shift ;;
        --population)   SINGLE_POPULATION="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--dry-run] [--population synthetic|adversarial|bingo|udfs]" >&2
            exit 1 ;;
    esac
done

# Create directories only when submitting (they do not exist on the workstation)
if ! ${DRY_RUN}; then
    mkdir -p "${LOGS_DIR}" "${RESULTS_ROOT}"
fi

# ---------------------------------------------------------------------------
# Job-ID capture helpers
#
# Picasso's Lua sbatch wrapper prepends ANSI codes and a multi-line warning to
# --parsable output.  Take the LAST line before stripping non-digits so that the
# guard fires on the right string and is not confused by intermediate lines.
# ---------------------------------------------------------------------------
_clean_job_id() {
    tail -n 1 <<<"$1" | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[^0-9]//g'
}

submit() {
    local raw id
    raw=$(sbatch --parsable "$@") || return 1
    id=$(_clean_job_id "${raw}")
    [[ "${id}" =~ ^[0-9]+$ ]] || {
        echo "FATAL: unparsable job id: ${raw@Q}" >&2
        return 1
    }
    echo "${id}"
}

SUBMITTED=()

# ---------------------------------------------------------------------------
# Helper: submit one array job
# ---------------------------------------------------------------------------
_submit_array() {
    local pop="$1"
    local n_tasks="$2"
    local time_limit="$3"
    local cpus="$4"
    local mem_gb="$5"
    shift 5
    # remaining args passed as --export key=value pairs
    local extra_export="$*"

    local export_str="ALL"
    export_str+=",T07_POPULATION=${pop}"
    export_str+=",T07_REPO_DIR=${REPO_DIR}"
    export_str+=",T07_RESULTS_ROOT=${RESULTS_ROOT}"
    export_str+=",T07_N_SEEDS=${N_SEEDS}"
    if [[ -n "${extra_export}" ]]; then
        export_str+=",${extra_export}"
    fi

    local log_tag="t07_${pop}"
    # Arrays are 0-indexed; worker decodes via sed -n "$((TASK_IDX+1))p"
    local array_spec="0-$((n_tasks - 1))%${MAX_CONCURRENT}"

    local sbatch_args=(
        --job-name="${log_tag}"
        --array="${array_spec}"
        --time="${time_limit}"
        --ntasks=1
        --cpus-per-task="${cpus}"
        --mem="${mem_gb}G"
        --constraint="${CONSTRAINT}"
        --account="${ACCOUNT}"
        --output="${LOGS_DIR}/${log_tag}_%A_%a.out"
        --error="${LOGS_DIR}/${log_tag}_%A_%a.err"
        --export="${export_str}"
        "${WORKER_SCRIPT}"
    )

    if ${DRY_RUN}; then
        echo "[DRY-RUN] sbatch --parsable ${sbatch_args[*]}"
        return 0
    fi

    local jid
    jid=$(submit "${sbatch_args[@]}") || exit 1
    SUBMITTED+=("${pop}=${jid}")
    echo "Submitted ${pop}: job ${jid} (${n_tasks} tasks, throttle ${MAX_CONCURRENT})"
}

# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------
_run_synthetic() {
    _submit_array "synthetic" \
        "${N_SYNTHETIC_TASKS}" \
        "0-01:30:00" 8 16 \
        "T07_N_SYNTHETIC_PER_TASK=${N_SYNTHETIC_PER_TASK}"
}

# ---------------------------------------------------------------------------
# Adversarial (single task — small population)
# ---------------------------------------------------------------------------
_run_adversarial() {
    local export_str="ALL"
    export_str+=",T07_POPULATION=adversarial"
    export_str+=",T07_REPO_DIR=${REPO_DIR}"
    export_str+=",T07_RESULTS_ROOT=${RESULTS_ROOT}"
    export_str+=",T07_N_SEEDS=${N_SEEDS}"

    local sbatch_args=(
        --job-name="t07_adversarial"
        --time="0-00:30:00"
        --ntasks=1
        --cpus-per-task=4
        --mem=8G
        --constraint="${CONSTRAINT}"
        --account="${ACCOUNT}"
        --output="${LOGS_DIR}/t07_adversarial_%j.out"
        --error="${LOGS_DIR}/t07_adversarial_%j.err"
        --export="${export_str}"
        "${WORKER_SCRIPT}"
    )

    if ${DRY_RUN}; then
        echo "[DRY-RUN] sbatch --parsable ${sbatch_args[*]}"
        return 0
    fi

    local jid
    jid=$(submit "${sbatch_args[@]}") || exit 1
    SUBMITTED+=("adversarial=${jid}")
    echo "Submitted adversarial: job ${jid} (1 task)"
}

# ---------------------------------------------------------------------------
# Bingo
# ---------------------------------------------------------------------------
_run_bingo() {
    local n_tasks=$(( ${#BINGO_PROBLEMS[@]} * N_SEEDS ))
    local problems_csv
    problems_csv=$(printf "%s|" "${BINGO_PROBLEMS[@]}")
    problems_csv="${problems_csv%|}"  # strip trailing separator

    # 64G, not 16G. CLAUDE.md records that production Bingo+IsalSR tasks need
    # 128G because of heap fragmentation, and this study is strictly heavier
    # than production: it holds a distinct-string set for BOTH arms over a 6 h
    # stream. 16G would have been OOM-killed. 64G is the compromise against
    # queue latency; raise to 128G if any task reports oom-kill.
    _submit_array "bingo" \
        "${n_tasks}" \
        "0-02:00:00" 8 64 \
        "T07_BINGO_PROBLEMS=${problems_csv},T07_BINGO_MAX_TIME=${BINGO_MAX_TIME}"
}

# ---------------------------------------------------------------------------
# UDFS
# ---------------------------------------------------------------------------
_run_udfs() {
    local n_tasks=$(( ${#UDFS_PROBLEMS[@]} * N_SEEDS ))
    local problems_csv
    problems_csv=$(printf "%s|" "${UDFS_PROBLEMS[@]}")
    problems_csv="${problems_csv%|}"

    _submit_array "udfs" \
        "${n_tasks}" \
        "0-02:00:00" 8 32 \
        "T07_UDFS_PROBLEMS=${problems_csv},T07_UDFS_MAX_TIME=${UDFS_MAX_TIME}"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${SINGLE_POPULATION}" in
    "")
        _run_synthetic
        _run_adversarial
        _run_bingo
        _run_udfs
        ;;
    synthetic)   _run_synthetic ;;
    adversarial) _run_adversarial ;;
    bingo)       _run_bingo ;;
    udfs)        _run_udfs ;;
    *)
        echo "Unknown population: ${SINGLE_POPULATION}" >&2
        exit 1 ;;
esac

if ${DRY_RUN}; then
    echo ""
    echo "Dry run only — nothing submitted."
    exit 0
fi

echo ""
echo "Submitted: ${SUBMITTED[*]:-none}"
echo "Monitor:   squeue -u \$USER"
echo "Logs:      ${LOGS_DIR}/"
echo "Results:   ${RESULTS_ROOT}/<population>/"
echo ""
echo "Aggregate when finished:"
echo "  python -m experiments.scripts.t07_norm_removal_aggregate \\"
echo "      --results-dir ${RESULTS_ROOT} \\"
echo "      --out ${RESULTS_ROOT}/summary.json"
