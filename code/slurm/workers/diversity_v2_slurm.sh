#!/usr/bin/env bash
# =============================================================================
# SLURM compute worker: Diversity Conjecture v2 (ARRAY JOB)
# =============================================================================
# Each array task runs ONE (benchmark, seed, variant) triple.
#
# Task layout for B benchmarks, S seeds:
#   For benchmark index b (0-based):
#     tasks [b*2S+1 .. b*2S+S]       = baseline seeds 0..S-1
#     tasks [b*2S+S+1 .. b*2S+2S]    = isalsr seeds 0..S-1
#
# Example: 3 benchmarks, 30 seeds → 180 tasks
#   Tasks   1-30:  I.12.4  baseline  seeds 0-29
#   Tasks  31-60:  I.12.4  isalsr    seeds 0-29
#   Tasks  61-90:  I.10.7  baseline  seeds 0-29
#   Tasks 91-120:  I.10.7  isalsr    seeds 0-29
#   Tasks 121-150: Nguyen-1 baseline seeds 0-29
#   Tasks 151-180: Nguyen-1 isalsr   seeds 0-29
#
# Differences from v1:
#   - Supports multiple benchmarks (decoded from task ID)
#   - Passes --enforce-dedup for isalsr variant
#
# Environment variables (exported by launch script):
#   ISALSR_REPO_DIR          - Path to IsalSR repository
#   DIVERSITY_V2_CONFIG      - Path to diversity_v2_config.yaml
#
set -euo pipefail

echo "=== Diversity v2 Experiment: SLURM Array Task ==="
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

CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)/envs/isalsr}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Force malloc allocator (bypasses pymalloc arenas) to prevent OOM
export PYTHONMALLOC=malloc

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
cd "$REPO_DIR"

PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"

# Read config
CONFIG="${DIVERSITY_V2_CONFIG:-${REPO_DIR}/slurm/diversity_v2_config.yaml}"
RESULTS_DIR=$($PYTHON -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['results_dir'])")

# Read experiment parameters
EXP=$($PYTHON -c "import yaml,json; json.dump(yaml.safe_load(open('${CONFIG}'))['experiment'], __import__('sys').stdout)")

N_SEEDS=$(     echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['n_seeds'])")
POP_SIZE=$(    echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('population_size', 200))")
STACK_SIZE=$(  echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('stack_size', 32))")
MAX_TIME=$(    echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('max_time', 7200.0))")
SNAPSHOT_GENS=$(echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('snapshot_gens', '0,10,25,50,100,150,200,300,400,500'))")
TRAJ_FREQ=$(   echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('trajectory_freq', 1))")
ENFORCE_DEDUP=$(echo "$EXP" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('enforce_dedup', False))")

# Read benchmark list
BENCHMARKS=$($PYTHON -c "
import yaml, json, sys
cfg = yaml.safe_load(open('${CONFIG}'))
benchmarks = cfg['experiment']['benchmarks']
json.dump(benchmarks, sys.stdout)
")
N_BENCHMARKS=$($PYTHON -c "import json; print(len(json.loads('${BENCHMARKS}')))")

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
N_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

# Decode task ID → (benchmark_idx, seed, variant)
read -r BENCHMARK SEED VARIANT < <($PYTHON -c "
import json, sys
benchmarks = json.loads('${BENCHMARKS}')
n_seeds = ${N_SEEDS}
task_id = ${TASK_ID} - 1  # 0-based

tasks_per_benchmark = 2 * n_seeds
benchmark_idx = task_id // tasks_per_benchmark
within = task_id % tasks_per_benchmark

if within < n_seeds:
    variant = 'baseline'
    seed = within
else:
    variant = 'isalsr'
    seed = within - n_seeds

print(benchmarks[benchmark_idx], seed, variant)
")

# Per-benchmark results subdirectory
BENCH_RESULTS="${RESULTS_DIR}/${BENCHMARK}"

echo "Config: benchmark=$BENCHMARK, variant=$VARIANT, seed=$SEED"
echo "        N=$POP_SIZE, stack=$STACK_SIZE, max_time=$MAX_TIME"
echo "        enforce_dedup=$ENFORCE_DEDUP, workers=$N_WORKERS"
echo "        snapshots=$SNAPSHOT_GENS, trajectory_freq=$TRAJ_FREQ"
echo "Output: $BENCH_RESULTS"
echo ""

# Build command
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
    --output-dir "$BENCH_RESULTS"
)

# Add --enforce-dedup only for isalsr variant
if [[ "$VARIANT" == "isalsr" && "$ENFORCE_DEDUP" == "True" ]]; then
    CMD+=("--enforce-dedup")
fi

"${CMD[@]}"

echo ""
echo "=== Task ${TASK_ID} (${BENCHMARK}/${VARIANT}/seed=${SEED}) complete: $(date) ==="
