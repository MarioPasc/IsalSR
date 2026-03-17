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

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DRY_RUN=false
SINGLE_EXP=""

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
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Parse config.yaml using Python
# ---------------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
fi

parse_config() {
    python3 -c "
import yaml, json, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
json.dump(cfg, sys.stdout)
"
}

CONFIG_JSON=$(parse_config)

REPO_DIR=$(echo "$CONFIG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['repo_dir'])")
RESULTS_DIR=$(echo "$CONFIG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['results_dir'])")
CONDA_ENV=$(echo "$CONFIG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['conda_env'])")
CONSTRAINT=$(echo "$CONFIG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['constraint'])")
ACCOUNT=$(echo "$CONFIG_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['account'])")

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
    echo "$CONFIG_JSON" | python3 -c "
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
    enabled=$(echo "$exp_config" | python3 -c "import json,sys; print(json.load(sys.stdin).get('enabled', False))")
    if [[ "$enabled" != "True" ]]; then
        echo "[SKIP] ${exp_name}: disabled in config"
        return
    fi

    local time_limit cpus mem_gb
    time_limit=$(echo "$exp_config" | python3 -c "import json,sys; print(json.load(sys.stdin)['time_limit'])")
    cpus=$(echo "$exp_config" | python3 -c "import json,sys; print(json.load(sys.stdin)['cpus'])")
    mem_gb=$(echo "$exp_config" | python3 -c "import json,sys; print(json.load(sys.stdin)['mem_gb'])")

    local out_dir="${RESULTS_DIR}/${exp_name}"
    local worker_script="${SCRIPT_DIR}/workers/${exp_name}_slurm.sh"

    if [[ ! -f "$worker_script" ]]; then
        echo "[ERROR] Worker script not found: ${worker_script}"
        return 1
    fi

    # Create output directory (skip on dry-run since Picasso paths may not exist locally)
    if [[ "$DRY_RUN" == "false" ]]; then
        mkdir -p "${out_dir}"
    fi

    local sbatch_cmd="sbatch \
        --job-name=isalsr_${exp_name} \
        --output=${out_dir}/slurm_%j.out \
        --error=${out_dir}/slurm_%j.err \
        --time=${time_limit} \
        --cpus-per-task=${cpus} \
        --mem=${mem_gb}G \
        --constraint=${CONSTRAINT} \
        --account=${ACCOUNT} \
        --chdir=${REPO_DIR} \
        --export=ALL,ISALSR_REPO_DIR=${REPO_DIR} \
        ${worker_script}"

    echo "[${exp_name}]"
    echo "  Time:  ${time_limit}"
    echo "  CPUs:  ${cpus}"
    echo "  Mem:   ${mem_gb}G"
    echo "  Out:   ${out_dir}"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would execute: ${sbatch_cmd}"
    else
        echo "  Submitting..."
        eval "${sbatch_cmd}"
    fi
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

if [[ -n "$SINGLE_EXP" ]]; then
    submit_experiment "$SINGLE_EXP"
else
    for exp in "${EXPERIMENTS[@]}"; do
        submit_experiment "$exp"
    done
fi

echo "Done."
