#!/usr/bin/env bash
# =============================================================================
# SLURM compute worker: Diversity Conjecture X.7 (ARRAY JOB)
# =============================================================================
# Each array task runs ONE (seed, variant) pair.
# Task layout: tasks 1..N_SEEDS = baseline, N_SEEDS+1..2*N_SEEDS = isalsr.
#
# Environment variables (exported by diversity_launch.sh):
#   ISALSR_REPO_DIR          - Path to IsalSR repository
#   DIVERSITY_N_SEEDS        - Number of seeds (for task ID decoding)
#   DIVERSITY_RESULTS_DIR    - Results directory
#   DIVERSITY_CONFIG         - Path to diversity_config.yaml
#
set -euo pipefail

echo "=== Diversity Experiment: SLURM Array Task ==="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Array ID:  ${SLURM_ARRAY_TASK_ID:-1}"
echo "Node:      $(hostname)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK:-1}"
echo "Start:     $(date)"
echo ""

# Load MPI module (required by bingo-nasa via mpi4py)
for mod in openmpi_gcc/5.0.9_gcc7 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/5.0.9_gcc14; do
    module load "$mod" 2>/dev/null && break
done

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

# Ensure conda lib is in LD_LIBRARY_PATH (for conda-installed openmpi)
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)/envs/isalsr}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
cd "$REPO_DIR"

PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"

# Read config
CONFIG="${DIVERSITY_CONFIG:-${REPO_DIR}/slurm/diversity_config.yaml}"
N_SEEDS="${DIVERSITY_N_SEEDS:-$($PYTHON -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['experiment']['n_seeds'])")}"
RESULTS_DIR="${DIVERSITY_RESULTS_DIR:-$($PYTHON -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['results_dir'])")}"

# Read experiment parameters from config
EXP=$($PYTHON -c "import yaml,json; json.dump(yaml.safe_load(open('${CONFIG}'))['experiment'], __import__('sys').stdout)")

POP_SIZE=$(     echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('population_size', 200))")
STACK_SIZE=$(   echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('stack_size', 16))")
MAX_TIME=$(     echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('max_time', 3600.0))")
SNAPSHOT_GENS=$(echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('snapshot_gens', '0,5,10,20,30,50,70,100,120,150'))")
TRAJ_FREQ=$(    echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('trajectory_freq', 1))")
GED_MODE=$(     echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('ged_mode', 'exact'))")
GED_TIMEOUT=$(  echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('ged_timeout', 10.0))")

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
N_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

echo "Config: N=$POP_SIZE, stack=$STACK_SIZE, max_time=$MAX_TIME"
echo "        GED mode=$GED_MODE, timeout=${GED_TIMEOUT}s/pair, workers=$N_WORKERS"
echo "        snapshots=$SNAPSHOT_GENS, trajectory_freq=$TRAJ_FREQ"
echo "        seeds=$N_SEEDS, task_id=$TASK_ID"
echo "Output: $RESULTS_DIR"
echo ""

$PYTHON -m experiments.scripts.diversity.diversity_experiment \
    --task-id "$TASK_ID" \
    --seeds "$N_SEEDS" \
    --pop-size "$POP_SIZE" \
    --stack-size "$STACK_SIZE" \
    --max-time "$MAX_TIME" \
    --snapshot-gens "$SNAPSHOT_GENS" \
    --trajectory-freq "$TRAJ_FREQ" \
    --ged-mode "$GED_MODE" \
    --ged-timeout "$GED_TIMEOUT" \
    --n-workers "$N_WORKERS" \
    --output-dir "$RESULTS_DIR"

echo ""
echo "=== Task ${TASK_ID} complete: $(date) ==="
