#!/usr/bin/env bash
# =============================================================================
# IsalSR Model-Based Experiment Launcher for Picasso HPC
# =============================================================================
#
# Submits SLURM array jobs for all (method, benchmark, variant) combinations
# defined in models_config.yaml, then submits an analysis job that depends
# on all experiments completing.
#
# Usage:
#   bash slurm/models_launch.sh                                  # Submit all
#   bash slurm/models_launch.sh --dry-run                        # Print commands only
#   bash slurm/models_launch.sh --experiment udfs_nguyen_baseline # Single group
#   bash slurm/models_launch.sh --analyze-only                   # Only analysis
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/models_config.yaml"
WORKER_SCRIPT="${SCRIPT_DIR}/workers/models_experiment_slurm.sh"
ANALYZE_SCRIPT="${SCRIPT_DIR}/workers/models_analyze_slurm.sh"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DRY_RUN=false
SINGLE_EXP=""
ANALYZE_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --experiment)
            SINGLE_EXP="$2"
            shift 2
            ;;
        --analyze-only)
            ANALYZE_ONLY=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--dry-run] [--experiment NAME] [--analyze-only]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Parse config.yaml using Python
# ---------------------------------------------------------------------------
parse_config() {
    "$PYTHON" -c "
import yaml, json, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
json.dump(cfg, sys.stdout)
"
}

CONFIG_JSON=$(parse_config)

REPO_DIR=$(echo "$CONFIG_JSON" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['repo_dir'])")
RESULTS_DIR=$(echo "$CONFIG_JSON" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['results_dir'])")
CONDA_ENV=$(echo "$CONFIG_JSON" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['conda_env'])")
CONSTRAINT=$(echo "$CONFIG_JSON" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['constraint'])")
ACCOUNT=$(echo "$CONFIG_JSON" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['account'])")

echo "=============================================="
echo "IsalSR Model-Based Experiment Launcher"
echo "=============================================="
echo "Repo:       ${REPO_DIR}"
echo "Results:    ${RESULTS_DIR}"
echo "Conda env:  ${CONDA_ENV}"
echo "Constraint: ${CONSTRAINT}"
echo "Account:    ${ACCOUNT}"
echo "Dry run:    ${DRY_RUN}"
echo ""

# ---------------------------------------------------------------------------
# Helper: get experiment config field
# ---------------------------------------------------------------------------
get_exp_field() {
    local exp_name="$1"
    local field="$2"
    echo "$CONFIG_JSON" | "$PYTHON" -c "
import json, sys
cfg = json.load(sys.stdin)
exp = cfg['experiments'].get('${exp_name}', {})
print(exp.get('${field}', ''))
"
}

# ---------------------------------------------------------------------------
# Submit a single experiment array job
# ---------------------------------------------------------------------------
# Returns: job ID on stdout. All status messages go to stderr.
submit_experiment() {
    local exp_name="$1"

    local enabled
    enabled=$(get_exp_field "$exp_name" "enabled")
    if [[ "$enabled" != "True" ]]; then
        echo "[SKIP] ${exp_name}: disabled in config" >&2
        return 0
    fi

    local method benchmark variant config_file n_seeds n_problems time_limit cpus mem_gb
    method=$(get_exp_field "$exp_name" "method")
    benchmark=$(get_exp_field "$exp_name" "benchmark")
    variant=$(get_exp_field "$exp_name" "variant")
    config_file=$(get_exp_field "$exp_name" "config")
    n_seeds=$(get_exp_field "$exp_name" "n_seeds")
    n_problems=$(get_exp_field "$exp_name" "n_problems")
    time_limit=$(get_exp_field "$exp_name" "time_limit")
    cpus=$(get_exp_field "$exp_name" "cpus")
    mem_gb=$(get_exp_field "$exp_name" "mem_gb")

    local array_size=$((n_seeds * n_problems))
    local out_dir="${RESULTS_DIR}/slurm_logs/${exp_name}"

    echo "[${exp_name}]" >&2
    echo "  Method:    ${method}" >&2
    echo "  Benchmark: ${benchmark}" >&2
    echo "  Variant:   ${variant}" >&2
    echo "  Array:     1-${array_size} (${n_problems} problems x ${n_seeds} seeds)" >&2
    echo "  Time:      ${time_limit}" >&2
    echo "  CPUs:      ${cpus}" >&2
    echo "  Mem:       ${mem_gb}G" >&2
    echo "  Out:       ${out_dir}" >&2

    local sbatch_cmd="sbatch \
        --job-name=isalsr_${exp_name} \
        --output=${out_dir}/slurm_%A_%a.out \
        --error=${out_dir}/slurm_%A_%a.err \
        --time=${time_limit} \
        --cpus-per-task=${cpus} \
        --mem=${mem_gb}G \
        --constraint=${CONSTRAINT} \
        --account=${ACCOUNT} \
        --chdir=${REPO_DIR} \
        --array=1-${array_size} \
        --export=ALL,ISALSR_REPO_DIR=${REPO_DIR},MODELS_METHOD=${method},MODELS_BENCHMARK=${benchmark},MODELS_VARIANT=${variant},MODELS_EXPERIMENT_CONFIG=${REPO_DIR}/${config_file},MODELS_N_SEEDS=${n_seeds},MODELS_RESULTS_DIR=${RESULTS_DIR} \
        ${WORKER_SCRIPT}"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would execute:" >&2
        echo "    ${sbatch_cmd}" >&2
        echo "" >&2
        echo "12345"
        return 0
    fi

    # Create output directory
    mkdir -p "${out_dir}"

    echo "  Submitting..." >&2
    local sbatch_output
    sbatch_output=$(eval "${sbatch_cmd}" 2>&1)
    echo "  ${sbatch_output}" >&2
    echo "" >&2

    # Extract job ID (Picasso wraps sbatch, output varies)
    local job_id
    job_id=$(echo "$sbatch_output" | grep -oP 'job\s+\K[0-9]+' | head -1)
    if [[ -z "$job_id" ]]; then
        job_id=$(echo "$sbatch_output" | grep -oP '[0-9]+' | head -1)
    fi
    echo "$job_id"
}

# ---------------------------------------------------------------------------
# Submit analysis job with dependencies
# ---------------------------------------------------------------------------
submit_analysis() {
    local dep_flag="$1"

    local time_limit cpus mem_gb
    time_limit=$(get_exp_field "models_analyze" "time_limit")
    cpus=$(get_exp_field "models_analyze" "cpus")
    mem_gb=$(get_exp_field "models_analyze" "mem_gb")
    local methods benchmarks
    methods=$(get_exp_field "models_analyze" "methods")
    benchmarks=$(get_exp_field "models_analyze" "benchmarks")

    local out_dir="${RESULTS_DIR}/slurm_logs/models_analyze"

    echo "[models_analyze]" >&2
    echo "  Methods:    ${methods}" >&2
    echo "  Benchmarks: ${benchmarks}" >&2
    echo "  Time:       ${time_limit}" >&2
    echo "  CPUs:       ${cpus}" >&2
    echo "  Mem:        ${mem_gb}G" >&2
    if [[ -n "$dep_flag" ]]; then
        echo "  Dependency: ${dep_flag}" >&2
    fi
    echo "  Out:        ${out_dir}" >&2

    # Export variables containing commas via shell env (--export=ALL inherits them).
    # Cannot use --export=VAR=val,VAR2=val2 because commas in values are ambiguous.
    export ISALSR_REPO_DIR="${REPO_DIR}"
    export MODELS_RESULTS_DIR="${RESULTS_DIR}"
    export MODELS_METHODS="${methods}"
    export MODELS_BENCHMARKS="${benchmarks}"

    local sbatch_cmd="sbatch \
        --job-name=isalsr_models_analyze \
        --output=${out_dir}/slurm_%j.out \
        --error=${out_dir}/slurm_%j.err \
        --time=${time_limit} \
        --cpus-per-task=${cpus} \
        --mem=${mem_gb}G \
        --constraint=${CONSTRAINT} \
        --account=${ACCOUNT} \
        --chdir=${REPO_DIR} \
        --export=ALL \
        ${dep_flag} \
        ${ANALYZE_SCRIPT}"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would execute:" >&2
        echo "    ${sbatch_cmd}" >&2
        echo "" >&2
        return 0
    fi

    mkdir -p "${out_dir}"

    echo "  Submitting..." >&2
    local sbatch_output
    sbatch_output=$(eval "${sbatch_cmd}" 2>&1)
    echo "  ${sbatch_output}" >&2
    echo "" >&2
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
EXPERIMENT_GROUPS=(
    udfs_nguyen_baseline
    udfs_nguyen_isalsr
    bingo_nguyen_baseline
    bingo_nguyen_isalsr
    udfs_feynman_baseline
    udfs_feynman_isalsr
    bingo_feynman_baseline
    bingo_feynman_isalsr
)

if [[ "$ANALYZE_ONLY" == "true" ]]; then
    echo "=== Submitting analysis job only ==="
    submit_analysis ""
elif [[ -n "$SINGLE_EXP" ]]; then
    if [[ "$SINGLE_EXP" == "models_analyze" ]]; then
        submit_analysis ""
    else
        submit_experiment "$SINGLE_EXP"
    fi
else
    echo "=== Submitting all experiment groups + analysis ==="
    echo ""

    ALL_JOB_IDS=()
    for exp in "${EXPERIMENT_GROUPS[@]}"; do
        job_id=$(submit_experiment "$exp")
        if [[ -n "$job_id" && "$job_id" != "0" ]]; then
            ALL_JOB_IDS+=("$job_id")
        fi
    done

    # Submit analysis with afterok dependency on ALL experiment jobs
    if [[ ${#ALL_JOB_IDS[@]} -gt 0 ]]; then
        DEP_STRING=$(IFS=:; echo "${ALL_JOB_IDS[*]}")
        echo "=== Submitting analysis job ==="
        echo "  Depends on ${#ALL_JOB_IDS[@]} experiment jobs: ${DEP_STRING}"
        echo ""
        submit_analysis "--dependency=afterok:${DEP_STRING}"
    else
        echo ""
        echo "[WARN] No experiment jobs submitted. Skipping analysis."
    fi
fi

echo "Done."
