#!/usr/bin/env bash
# =============================================================================
# IsalSR Benchmark Launcher for Picasso HPC
# =============================================================================
#
# Master executor that reads slurm/config.yaml and dispatches SLURM jobs
# for each enabled experiment.
#
# Usage:
#   bash slurm/launch.sh                                    # Submit all
#   bash slurm/launch.sh --dry-run                          # Print commands only
#   bash slurm/launch.sh --experiment random_search_canon   # Single experiment
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config.yaml"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

# Resolve Python from the active conda env (works regardless of install path)
PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DRY_RUN=false
SINGLE_EXP=""
CACHE_MODE=false

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
        --cache)
            CACHE_MODE=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
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
echo "IsalSR Benchmark Launcher"
echo "=============================================="
echo "Repo:       ${REPO_DIR}"
echo "Results:    ${RESULTS_DIR}"
echo "Conda env:  ${CONDA_ENV}"
echo "Constraint: ${CONSTRAINT}"
echo "Account:    ${ACCOUNT}"
echo "Dry run:    ${DRY_RUN}"
echo ""

# ---------------------------------------------------------------------------
# Get experiment config as JSON
# ---------------------------------------------------------------------------
get_exp_config() {
    local exp_name="$1"
    echo "$CONFIG_JSON" | "$PYTHON" -c "
import json, sys
cfg = json.load(sys.stdin)
exp = cfg['experiments'].get('${exp_name}', {})
json.dump(exp, sys.stdout)
"
}

# ---------------------------------------------------------------------------
# Submit a single experiment job
# ---------------------------------------------------------------------------
submit_experiment() {
    local exp_name="$1"
    local exp_config
    exp_config=$(get_exp_config "$exp_name")

    local enabled
    enabled=$(echo "$exp_config" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin).get('enabled', False))")
    if [[ "$enabled" != "True" ]]; then
        echo "[SKIP] ${exp_name}: disabled in config"
        return
    fi

    local time_limit cpus mem_gb
    time_limit=$(echo "$exp_config" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['time_limit'])")
    cpus=$(echo "$exp_config" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['cpus'])")
    mem_gb=$(echo "$exp_config" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin)['mem_gb'])")

    # Check if this experiment uses array jobs (has n_runs or n_shards).
    local n_runs n_shards
    n_runs=$(echo "$exp_config" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin).get('n_runs', 0))")
    n_shards=$(echo "$exp_config" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin).get('n_shards', 0))")

    local out_dir="${RESULTS_DIR}/${exp_name}"

    # Resolve worker script. Cache generation/merge experiments share
    # a single worker script with CACHE_EXPERIMENT_NAME exported.
    local worker_script
    if [[ "$exp_name" == generate_cache_* ]]; then
        worker_script="${SCRIPT_DIR}/workers/generate_cache_slurm.sh"
    elif [[ "$exp_name" == merge_cache_* ]]; then
        worker_script="${SCRIPT_DIR}/workers/merge_cache_slurm.sh"
    else
        worker_script="${SCRIPT_DIR}/workers/${exp_name}_slurm.sh"
    fi

    if [[ ! -f "$worker_script" ]]; then
        echo "[ERROR] Worker script not found: ${worker_script}"
        return 1
    fi

    # Create output directory (skip on dry-run since Picasso paths may not exist locally)
    if [[ "$DRY_RUN" == "false" ]]; then
        mkdir -p "${out_dir}"
    fi

    # Build sbatch command. Use --array for multi-run or multi-shard experiments.
    local array_size=0
    if [[ "$n_runs" -gt 0 ]]; then
        array_size=$n_runs
    elif [[ "$n_shards" -gt 0 ]]; then
        array_size=$n_shards
    fi

    local array_flag=""
    local output_pattern error_pattern
    if [[ "$array_size" -gt 0 ]]; then
        array_flag="--array=1-${array_size}"
        output_pattern="${out_dir}/slurm_%A_%a.out"
        error_pattern="${out_dir}/slurm_%A_%a.err"
    else
        output_pattern="${out_dir}/slurm_%j.out"
        error_pattern="${out_dir}/slurm_%j.err"
    fi

    # Extra exports for cache generation/merge experiments.
    local extra_exports="ISALSR_REPO_DIR=${REPO_DIR}"
    if [[ "$exp_name" == generate_cache_* || "$exp_name" == merge_cache_* ]]; then
        local num_vars
        num_vars=$(echo "$exp_config" | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin).get('num_variables', 1))")
        extra_exports="${extra_exports},CACHE_EXPERIMENT_NAME=${exp_name},CACHE_NUM_VARS=${num_vars}"
    fi

    # Optional dependency flag (set externally for merge-after-generate chains).
    # MUST come BEFORE the worker script in the sbatch command.
    local dep_flag="${DEPENDENCY_FLAG:-}"

    local sbatch_cmd="sbatch \
        --job-name=isalsr_${exp_name} \
        --output=${output_pattern} \
        --error=${error_pattern} \
        --time=${time_limit} \
        --cpus-per-task=${cpus} \
        --mem=${mem_gb}G \
        --constraint=${CONSTRAINT} \
        --account=${ACCOUNT} \
        --chdir=${REPO_DIR} \
        --export=ALL,${extra_exports} \
        ${array_flag} \
        ${dep_flag} \
        ${worker_script}"

    # All status info goes to stderr so that $(submit_experiment ...) captures
    # only the job ID on stdout.
    echo "[${exp_name}]" >&2
    echo "  Time:  ${time_limit}" >&2
    echo "  CPUs:  ${cpus}" >&2
    echo "  Mem:   ${mem_gb}G" >&2
    if [[ "$array_size" -gt 0 ]]; then
        echo "  Array: 1-${array_size} (${array_size} parallel tasks)" >&2
    fi
    echo "  Out:   ${out_dir}" >&2
    if [[ -n "$dep_flag" ]]; then
        echo "  Dependency: ${dep_flag}" >&2
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would execute:" >&2
        echo "    ${sbatch_cmd}" >&2
        echo "" >&2
        # Return a fake job ID for dry-run chaining.
        echo "12345"
        return 0
    else
        echo "  Submitting..." >&2
        local sbatch_output
        sbatch_output=$(eval "${sbatch_cmd}" 2>&1)
        echo "  ${sbatch_output}" >&2
        echo "" >&2
        # Extract numeric job ID from sbatch output.
        # Picasso wraps sbatch in a Lua script, so the output may be:
        #   "Submitted batch job 12345"
        #   "Submitted batch job 12345 on cluster picasso"
        # We extract the FIRST sequence of digits after "job".
        local job_id
        job_id=$(echo "$sbatch_output" | grep -oP 'job\s+\K[0-9]+' | head -1)
        if [[ -z "$job_id" ]]; then
            # Fallback: extract any number from the output.
            job_id=$(echo "$sbatch_output" | grep -oP '[0-9]+' | head -1)
        fi
        echo "$job_id"
        return 0
    fi
}

# ---------------------------------------------------------------------------
# Cache pipeline: generate shards then merge with --dependency=afterok
# ---------------------------------------------------------------------------
submit_cache_pipeline() {
    local gen_name="$1"     # e.g., "generate_cache_nguyen_1var"
    local merge_name="$2"   # e.g., "merge_cache_nguyen_1var"

    echo "================================================"
    echo "Cache pipeline: ${gen_name} -> ${merge_name}"
    echo "================================================"

    # Submit the generate (array) job and capture its job ID.
    local gen_job_id
    gen_job_id=$(submit_experiment "$gen_name")

    if [[ -z "$gen_job_id" || "$gen_job_id" == "0" ]]; then
        echo "[ERROR] Failed to capture job ID for ${gen_name}"
        return 1
    fi

    echo "  Generate job ID: ${gen_job_id}"

    # Submit the merge job with afterok dependency on the generate array.
    # --dependency=afterok:JOB_ID waits for ALL tasks in the array.
    DEPENDENCY_FLAG="--dependency=afterok:${gen_job_id}" submit_experiment "$merge_name"
    echo "  Merge depends on generate job ${gen_job_id} (afterok)"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
EXPERIMENTS=(
    "search_space_analysis"
    "random_search_canon"
    "random_search_nocanon"
    "hill_climbing_canon"
    "hill_climbing_nocanon"
    "gp_canon"
    "gp_nocanon"
    "aggregate_results"
)

# Cache generation pipelines (generate -> merge with afterok dependency).
CACHE_PIPELINES=(
    "generate_cache_nguyen_1var:merge_cache_nguyen_1var"
    "generate_cache_nguyen_2var:merge_cache_nguyen_2var"
    "generate_cache_feynman_1var:merge_cache_feynman_1var"
    "generate_cache_feynman_2var:merge_cache_feynman_2var"
    "generate_cache_feynman_3var:merge_cache_feynman_3var"
)

if [[ -n "$SINGLE_EXP" ]]; then
    submit_experiment "$SINGLE_EXP"
elif [[ "$CACHE_MODE" == "true" ]]; then
    # Submit all cache pipelines (generate -> afterok -> merge).
    echo "=== Submitting ALL cache generation pipelines ==="
    echo ""
    for pipeline in "${CACHE_PIPELINES[@]}"; do
        gen_name="${pipeline%%:*}"
        merge_name="${pipeline##*:}"
        submit_cache_pipeline "$gen_name" "$merge_name"
    done
else
    for exp in "${EXPERIMENTS[@]}"; do
        submit_experiment "$exp"
    done
fi

echo "Done."
