#!/usr/bin/env bash
# =============================================================================
# Diversity Conjecture X.7 — SLURM Launcher for Picasso HPC
# =============================================================================
# Submits:
#   Phase 1: Array job (2 * N_SEEDS tasks) for baseline + isalsr evolution
#   Phase 2: Single analysis job (figure generation) depends on Phase 1
#
# Usage:
#   bash slurm/diversity_launch.sh               # submit all
#   bash slurm/diversity_launch.sh --dry-run      # print sbatch commands only
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
CONFIG="${REPO_DIR}/slurm/diversity_config.yaml"
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
TIME_LIMIT=$(   echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['time_limit'])")
CPUS=$(         echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['cpus'])")
MEM_GB=$(       echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['mem_gb'])")
AN_TIME=$(      echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['analysis']['time_limit'])")
AN_CPUS=$(      echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['analysis']['cpus'])")
AN_MEM=$(       echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['analysis']['mem_gb'])")

N_TASKS=$(( 2 * N_SEEDS ))

SLURM_OUT="${RESULTS_DIR}/slurm_logs"

echo "============================================================"
echo " Diversity Conjecture X.7 — Picasso SLURM Launcher"
echo "============================================================"
echo " Config:      $CONFIG"
echo " Repo (HPC):  $PICASSO_REPO"
echo " Results:     $RESULTS_DIR"
echo " Seeds:       $N_SEEDS"
echo " Array tasks: $N_TASKS (${N_SEEDS} baseline + ${N_SEEDS} isalsr)"
echo " Per task:    $TIME_LIMIT, $CPUS CPUs, ${MEM_GB}G RAM"
echo " Analysis:    $AN_TIME, $AN_CPUS CPUs, ${AN_MEM}G RAM"
echo " Dry-run:     $DRY_RUN"
echo "============================================================"
echo ""

# ---- Submit experiment array job ----
EXPERIMENT_CMD="sbatch \
    --job-name=diversity_exp \
    --constraint=$CONSTRAINT \
    --account=$ACCOUNT \
    --time=$TIME_LIMIT \
    --cpus-per-task=$CPUS \
    --mem=${MEM_GB}G \
    --array=1-${N_TASKS}%10 \
    --output=${SLURM_OUT}/exp_%A_%a.out \
    --error=${SLURM_OUT}/exp_%A_%a.err \
    --export=ISALSR_REPO_DIR=$PICASSO_REPO,DIVERSITY_N_SEEDS=$N_SEEDS,DIVERSITY_RESULTS_DIR=$RESULTS_DIR,DIVERSITY_CONFIG=${PICASSO_REPO}/slurm/diversity_config.yaml \
    ${PICASSO_REPO}/slurm/workers/diversity_experiment_slurm.sh"

echo "Phase 1: Experiment array ($N_TASKS tasks, throttle=10)"
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

# ---- Submit analysis job (depends on experiment) ----
ANALYZE_CMD="sbatch \
    --job-name=diversity_analyze \
    --constraint=$CONSTRAINT \
    --account=$ACCOUNT \
    --time=$AN_TIME \
    --cpus-per-task=$AN_CPUS \
    --mem=${AN_MEM}G \
    --dependency=afterany:${EXP_JOB_ID} \
    --output=${SLURM_OUT}/analyze_%j.out \
    --error=${SLURM_OUT}/analyze_%j.err \
    --export=ISALSR_REPO_DIR=$PICASSO_REPO,DIVERSITY_RESULTS_DIR=$RESULTS_DIR,DIVERSITY_CONFIG=${PICASSO_REPO}/slurm/diversity_config.yaml \
    ${PICASSO_REPO}/slurm/workers/diversity_analyze_slurm.sh"

echo "Phase 2: Analysis (depends on job $EXP_JOB_ID)"
echo "  $ANALYZE_CMD"
echo ""

if $DRY_RUN; then
    echo "  [DRY-RUN] Would submit."
else
    AN_JOB_ID=$(eval "$ANALYZE_CMD" | awk '{print $NF}')
    echo "  Submitted job: $AN_JOB_ID"
fi

echo ""
echo "============================================================"
echo " Summary:"
echo "   Phase 1 (experiment): job $EXP_JOB_ID ($N_TASKS array tasks)"
if ! $DRY_RUN; then
    echo "   Phase 2 (analysis):  job $AN_JOB_ID (depends on Phase 1)"
fi
echo ""
echo "   Monitor: squeue -u \$USER"
echo "   Logs:    $SLURM_OUT/"
echo "============================================================"
