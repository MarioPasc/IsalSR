#!/usr/bin/env bash
# =============================================================================
# Diversity Conjecture v2 — SLURM Launcher for Picasso HPC
# =============================================================================
# Submits array job for 3 benchmarks × 2 variants × 30 seeds = 180 tasks.
#
# Each task runs ONE (benchmark, seed, variant) triple.
# See slurm/workers/diversity_v2_slurm.sh for task ID decoding.
#
# Usage:
#   bash slurm/diversity_v2_launch.sh               # submit all
#   bash slurm/diversity_v2_launch.sh --dry-run      # print sbatch commands only
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Arguments ----
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---- Resolve config ----
CONFIG="${REPO_DIR}/slurm/diversity_v2_config.yaml"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

# Activate conda for Python YAML parsing
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true
PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"

# ---- Parse global config ----
PICASSO_REPO=$($PYTHON -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['repo_dir'])")
RESULTS_DIR=$($PYTHON -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['results_dir'])")
CONSTRAINT=$($PYTHON -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['constraint'])")
ACCOUNT=$($PYTHON -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['account'])")

# ---- Parse experiment config ----
EXP=$($PYTHON -c "import yaml,json; json.dump(yaml.safe_load(open('${CONFIG}'))['experiment'], __import__('sys').stdout)")

N_SEEDS=$(      echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['n_seeds'])")
N_BENCHMARKS=$( echo "$EXP" | $PYTHON -c "import json,sys; print(len(json.load(sys.stdin)['benchmarks']))")
TIME_LIMIT=$(   echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['time_limit'])")
CPUS=$(         echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['cpus'])")
MEM_GB=$(       echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['mem_gb'])")
AN_TIME=$(      echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['analysis']['time_limit'])")
AN_CPUS=$(      echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['analysis']['cpus'])")
AN_MEM=$(       echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['analysis']['mem_gb'])")
BENCHMARKS=$(   echo "$EXP" | $PYTHON -c "import json,sys; print(', '.join(json.load(sys.stdin)['benchmarks']))")

N_TASKS=$(( N_BENCHMARKS * 2 * N_SEEDS ))

SLURM_OUT="${RESULTS_DIR}/slurm_logs"

echo "============================================================"
echo " Diversity Conjecture v2 — Picasso SLURM Launcher"
echo "============================================================"
echo " Config:      $CONFIG"
echo " Repo (HPC):  $PICASSO_REPO"
echo " Results:     $RESULTS_DIR"
echo " Benchmarks:  $BENCHMARKS"
echo " Seeds:       $N_SEEDS"
echo " Array tasks: $N_TASKS (${N_BENCHMARKS} benchmarks × 2 variants × ${N_SEEDS} seeds)"
echo " Per task:    $TIME_LIMIT, $CPUS CPUs, ${MEM_GB}G RAM"
echo " Dry-run:     $DRY_RUN"
echo "============================================================"
echo ""

# ---- Submit experiment array job ----
EXPERIMENT_CMD="sbatch \
    --job-name=div_v2_exp \
    --constraint=$CONSTRAINT \
    --account=$ACCOUNT \
    --time=$TIME_LIMIT \
    --cpus-per-task=$CPUS \
    --mem=${MEM_GB}G \
    --array=1-${N_TASKS}%20 \
    --output=${SLURM_OUT}/exp_%A_%a.out \
    --error=${SLURM_OUT}/exp_%A_%a.err \
    --export=ISALSR_REPO_DIR=$PICASSO_REPO,DIVERSITY_V2_CONFIG=${PICASSO_REPO}/slurm/diversity_v2_config.yaml \
    ${PICASSO_REPO}/slurm/workers/diversity_v2_slurm.sh"

echo "Experiment array ($N_TASKS tasks, throttle=20)"
echo "  $EXPERIMENT_CMD"
echo ""

if $DRY_RUN; then
    EXP_JOB_ID=12345
    echo "  [DRY-RUN] Would submit. Fake job ID: $EXP_JOB_ID"
else
    mkdir -p "$SLURM_OUT" 2>/dev/null || true
    EXP_JOB_ID=$(eval "$EXPERIMENT_CMD" | awk '{print $NF}')
    echo "  Submitted job: $EXP_JOB_ID"
fi

echo ""
echo "============================================================"
echo " Summary:"
echo "   Job:    $EXP_JOB_ID ($N_TASKS array tasks)"
echo "   Monitor: squeue -u \$USER"
echo "   Logs:    $SLURM_OUT/"
echo "   Results: $RESULTS_DIR/{I.12.4,I.10.7,Nguyen-1}/"
echo "============================================================"
