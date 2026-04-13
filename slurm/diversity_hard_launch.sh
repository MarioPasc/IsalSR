#!/usr/bin/env bash
# =============================================================================
# Diversity-Hard Launcher for Picasso HPC
# =============================================================================
# Submits ONE 60-task array per selected experiment block (60 = 2 variants
# * 30 seeds). Each block targets a single hard benchmark — they run as
# independent SLURM arrays so users can launch any subset.
#
# Usage:
#   bash slurm/diversity_hard_launch.sh --experiment all
#   bash slurm/diversity_hard_launch.sh --experiment diversity_paramagnetism
#   bash slurm/diversity_hard_launch.sh --experiment diversity_korns12
#   bash slurm/diversity_hard_launch.sh --experiment diversity_pagie1
#   bash slurm/diversity_hard_launch.sh --experiment all --dry-run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${SCRIPT_DIR}/diversity_hard_config.yaml"
WORKER_SCRIPT="${SCRIPT_DIR}/workers/diversity_hard_slurm.sh"

DRY_RUN=false
EXPERIMENT="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --experiment {all|diversity_paramagnetism|diversity_korns12|diversity_pagie1} [--dry-run]"
            exit 1
            ;;
    esac
done

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true
PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"

# Parse global config
PICASSO_REPO=$($PYTHON -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['repo_dir'])")
RESULTS_DIR=$($PYTHON  -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['results_dir'])")
CONSTRAINT=$($PYTHON   -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['constraint'])")
ACCOUNT=$($PYTHON      -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['account'])")

# Parse shared config
SHARED=$($PYTHON -c "import yaml,json; json.dump(yaml.safe_load(open('${CONFIG}'))['shared'], __import__('sys').stdout)")
N_SEEDS=$(    echo "$SHARED" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['n_seeds'])")
TIME_LIMIT=$( echo "$SHARED" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['time_limit'])")
CPUS=$(       echo "$SHARED" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['cpus'])")
MEM_GB=$(     echo "$SHARED" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['mem_gb'])")
N_TASKS=$(( 2 * N_SEEDS ))

ALL_EXPERIMENTS=$($PYTHON -c "
import yaml, json, sys
cfg = yaml.safe_load(open('${CONFIG}'))
print(' '.join(name for name, blk in cfg['experiments'].items() if blk.get('enabled', True)))
")

echo "============================================================"
echo " Diversity-Hard Launcher"
echo "============================================================"
echo " Config:      $CONFIG"
echo " Repo (HPC):  $PICASSO_REPO"
echo " Results:     $RESULTS_DIR"
echo " Seeds:       $N_SEEDS  (-> $N_TASKS tasks/experiment)"
echo " Per task:    $TIME_LIMIT, $CPUS CPUs, ${MEM_GB}G RAM"
echo " Selection:   $EXPERIMENT"
echo " Dry-run:     $DRY_RUN"
echo " Available:   $ALL_EXPERIMENTS"
echo "============================================================"
echo ""

submit_one() {
    local exp_name="$1"

    local enabled benchmark
    enabled=$($PYTHON -c "
import yaml
cfg = yaml.safe_load(open('${CONFIG}'))
exp = cfg['experiments'].get('${exp_name}', {})
print(exp.get('enabled', False))
")
    if [[ "$enabled" != "True" ]]; then
        echo "[SKIP] ${exp_name}: disabled or missing in config" >&2
        return 0
    fi

    benchmark=$($PYTHON -c "
import yaml
cfg = yaml.safe_load(open('${CONFIG}'))
print(cfg['experiments']['${exp_name}']['benchmark'])
")

    local exp_results="${RESULTS_DIR}/${exp_name}"
    local slurm_out="${exp_results}/slurm_logs"

    echo "[${exp_name}]"
    echo "  Benchmark:   ${benchmark}"
    echo "  Array:       1-${N_TASKS} (2 variants * ${N_SEEDS} seeds)"
    echo "  Time:        ${TIME_LIMIT}, CPUs: ${CPUS}, Mem: ${MEM_GB}G"
    echo "  Out:         ${exp_results}"

    local sbatch_cmd="sbatch \
        --job-name=div_hard_${exp_name} \
        --constraint=${CONSTRAINT} \
        --account=${ACCOUNT} \
        --time=${TIME_LIMIT} \
        --cpus-per-task=${CPUS} \
        --mem=${MEM_GB}G \
        --array=1-${N_TASKS}%20 \
        --output=${slurm_out}/exp_%A_%a.out \
        --error=${slurm_out}/exp_%A_%a.err \
        --chdir=${PICASSO_REPO} \
        --export=ALL,ISALSR_REPO_DIR=${PICASSO_REPO},DIVERSITY_HARD_CONFIG=${PICASSO_REPO}/slurm/diversity_hard_config.yaml,DIVERSITY_HARD_BENCHMARK=${benchmark},DIVERSITY_HARD_RESULTS_DIR=${exp_results} \
        ${WORKER_SCRIPT}"

    if $DRY_RUN; then
        echo "  [DRY-RUN] Would execute:"
        echo "    $sbatch_cmd"
        echo ""
        return 0
    fi

    mkdir -p "${slurm_out}"
    local job_id
    job_id=$(eval "$sbatch_cmd" | awk '{print $NF}')
    echo "  Submitted: $job_id"
    echo ""
}

if [[ "$EXPERIMENT" == "all" ]]; then
    for exp in $ALL_EXPERIMENTS; do
        submit_one "$exp"
    done
else
    submit_one "$EXPERIMENT"
fi

echo "Done."
