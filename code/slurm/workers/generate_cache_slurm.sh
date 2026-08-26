#!/usr/bin/env bash
# SLURM compute worker: generate_cache (ARRAY JOB)
# Each array task generates one shard of the precomputed cache.
# Submit with: sbatch --array=1-20 generate_cache_slurm.sh
set -euo pipefail

echo "=== generate_cache: SLURM Array Task ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-1}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
CONFIG="${REPO_DIR}/slurm/config.yaml"
cd "$REPO_DIR"

# Parse config — extract experiment-specific parameters.
# The calling sbatch should export CACHE_EXPERIMENT_NAME (e.g., "generate_cache_nguyen_1var").
EXPERIMENT="${CACHE_EXPERIMENT_NAME:?ERROR: CACHE_EXPERIMENT_NAME not set}"

RESULTS_DIR=$($PYTHON -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['results_dir'])")
BENCH_CFG=$($PYTHON -c "import yaml,json; json.dump(yaml.safe_load(open('${CONFIG}'))['experiments']['${EXPERIMENT}'], __import__('sys').stdout)")

SEED_BASE=$(echo "$BENCH_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('seed_base', 42))")
NUM_VARS=$(echo "$BENCH_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['num_variables'])")
N_STRINGS=$(echo "$BENCH_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['n_strings_per_task'])")
MAX_TOKENS=$(echo "$BENCH_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('max_tokens', 30))")
TIMEOUT=$(echo "$BENCH_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('exhaustive_timeout', 60))")
OPS=$(echo "$BENCH_CFG" | $PYTHON -c "import json,sys; ops=json.load(sys.stdin).get('ops', None); print(ops if ops else '')")

OUT_DIR="${RESULTS_DIR}/precomputed_cache/${EXPERIMENT}"
mkdir -p "$OUT_DIR"

# Array task ID = shard ID (1-indexed).
RUN_ID="${SLURM_ARRAY_TASK_ID:-1}"
SEED=$((SEED_BASE + RUN_ID - 1))

echo "Experiment: ${EXPERIMENT}"
echo "Num variables: ${NUM_VARS}"
echo "Strings/task: ${N_STRINGS}"
echo "Max tokens: ${MAX_TOKENS}"
echo "Exhaustive timeout: ${TIMEOUT}s"
echo "Shard ${RUN_ID} (seed=${SEED})"
echo "Output: ${OUT_DIR}"

OPS_FLAG=""
if [[ -n "$OPS" ]]; then
    OPS_FLAG="--ops ${OPS}"
fi

python -m isalsr.precomputed.generate_cache \
    --mode sampled \
    --num-variables "$NUM_VARS" \
    --n-strings "$N_STRINGS" \
    --max-tokens "$MAX_TOKENS" \
    --seed "$SEED" \
    --run-id "$RUN_ID" \
    --exhaustive-timeout "$TIMEOUT" \
    $OPS_FLAG \
    --output "${OUT_DIR}/cache_shard_${RUN_ID}.h5"

echo ""
echo "=== Shard ${RUN_ID} complete: $(date) ==="
