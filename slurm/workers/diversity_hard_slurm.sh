#!/usr/bin/env bash
# =============================================================================
# SLURM compute worker: Diversity Conjecture — Hard-Tier (ARRAY JOB)
# =============================================================================
# Each array task runs ONE (seed, variant) pair for a SINGLE benchmark.
# Unlike diversity_v2, each launched experiment runs only ONE benchmark
# (per the user's "separately" requirement); the benchmark name is exported
# by the launcher via DIVERSITY_HARD_BENCHMARK.
#
# Task layout (S = n_seeds, default 30):
#   tasks  1..S        -> baseline seeds 0..S-1
#   tasks  S+1..2S     -> isalsr   seeds 0..S-1
# Total = 2 * S = 60 (default).
#
# Environment variables (exported by launcher):
#   ISALSR_REPO_DIR             - Path to IsalSR repository
#   DIVERSITY_HARD_CONFIG       - Path to diversity_hard_config.yaml
#   DIVERSITY_HARD_BENCHMARK    - Single benchmark name (e.g., "Pagie-1")
#   DIVERSITY_HARD_RESULTS_DIR  - Per-experiment results directory
#
set -euo pipefail

echo "=== Diversity-Hard Experiment: SLURM Array Task ==="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Array ID:  ${SLURM_ARRAY_TASK_ID:-1}"
echo "Node:      $(hostname)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK:-1}"
echo "Start:     $(date)"
echo ""

for mod in openmpi_gcc/5.0.9_gcc7 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/5.0.9_gcc14; do
    module load "$mod" 2>/dev/null && break
done

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)/envs/isalsr}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONMALLOC=malloc

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
cd "$REPO_DIR"

PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"

CONFIG="${DIVERSITY_HARD_CONFIG:?ERROR: DIVERSITY_HARD_CONFIG not set}"
BENCHMARK="${DIVERSITY_HARD_BENCHMARK:?ERROR: DIVERSITY_HARD_BENCHMARK not set}"
RESULTS_DIR="${DIVERSITY_HARD_RESULTS_DIR:?ERROR: DIVERSITY_HARD_RESULTS_DIR not set}"

# Read shared config
SHARED=$($PYTHON -c "import yaml,json,sys; json.dump(yaml.safe_load(open('${CONFIG}'))['shared'], sys.stdout)")

N_SEEDS=$(       echo "$SHARED" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['n_seeds'])")
POP_SIZE=$(      echo "$SHARED" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('population_size', 200))")
STACK_SIZE=$(    echo "$SHARED" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('stack_size', 32))")
MAX_TIME=$(      echo "$SHARED" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('max_time', 7200.0))")
SNAPSHOT_GENS=$( echo "$SHARED" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('snapshot_gens', '0,10,25,50,100,150,200,300,400,500'))")
TRAJ_FREQ=$(     echo "$SHARED" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('trajectory_freq', 1))")
ENFORCE_DEDUP=$( echo "$SHARED" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('enforce_dedup', False))")

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
N_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

# Decode task ID -> (seed, variant)
read -r SEED VARIANT < <($PYTHON -c "
n_seeds = ${N_SEEDS}
task_id = ${TASK_ID} - 1  # 0-based
if task_id < n_seeds:
    variant = 'baseline'
    seed = task_id
else:
    variant = 'isalsr'
    seed = task_id - n_seeds
print(seed, variant)
")

echo "Config: benchmark=$BENCHMARK, variant=$VARIANT, seed=$SEED"
echo "        N=$POP_SIZE, stack=$STACK_SIZE, max_time=$MAX_TIME"
echo "        enforce_dedup=$ENFORCE_DEDUP, workers=$N_WORKERS"
echo "        snapshots=$SNAPSHOT_GENS, trajectory_freq=$TRAJ_FREQ"
echo "Output: $RESULTS_DIR"
echo ""

CMD=(
    "$PYTHON" -m experiments.scripts.diversity.diversity_experiment
    --benchmark "$BENCHMARK"
    --seeds "$N_SEEDS"
    --single-seed "$SEED"
    --variant "$VARIANT"
    --pop-size "$POP_SIZE"
    --stack-size "$STACK_SIZE"
    --max-time "$MAX_TIME"
    --snapshot-gens "$SNAPSHOT_GENS"
    --trajectory-freq "$TRAJ_FREQ"
    --n-workers "$N_WORKERS"
    --output-dir "$RESULTS_DIR"
)

if [[ "$VARIANT" == "isalsr" && "$ENFORCE_DEDUP" == "True" ]]; then
    CMD+=("--enforce-dedup")
fi

"${CMD[@]}"

echo ""
echo "=== Task ${TASK_ID} (${BENCHMARK}/${VARIANT}/seed=${SEED}) complete: $(date) ==="
