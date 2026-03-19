#!/usr/bin/env bash
# SLURM compute worker: Canonicalization Time vs Nodes (Experiment 3)
# Array job: SLURM_ARRAY_TASK_ID = n_internal value
set -euo pipefail

echo "=== Experiment 3: Canonicalization Time - SLURM Worker ==="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Array ID:  ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true
PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
CONFIG="${REPO_DIR}/slurm/config.yaml"
cd "$REPO_DIR"

RESULTS_DIR=$($PYTHON -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['results_dir'])")
EXP_CFG=$($PYTHON -c "import yaml,json; json.dump(yaml.safe_load(open('${CONFIG}'))['experiments']['exp3_canonicalization_time'], __import__('sys').stdout)")

SAMPLES=$( echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('samples_per_node', 200))")
TIMEOUT=$( echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('timeout', 120))")
SEED=$(    echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")

N_INTERNAL=${SLURM_ARRAY_TASK_ID:-1}
OUT_DIR="${RESULTS_DIR}/exp3_canonicalization_time"
mkdir -p "$OUT_DIR"

echo "Config: n_internal=$N_INTERNAL, samples=$SAMPLES, timeout=$TIMEOUT, seed=$SEED"
echo "Output: $OUT_DIR"

# Run for all num_vars values (1, 2, 3) at this n_internal
for NVARS in 1 2 3; do
    echo "--- num_vars=$NVARS, n_internal=$N_INTERNAL ---"
    python experiments/scripts/exp3_canonicalization_time.py \
        --output "${OUT_DIR}/timing_v${NVARS}_k${N_INTERNAL}.csv" \
        --num-vars "$NVARS" \
        --n-internal "$N_INTERNAL" \
        --samples-per-node "$SAMPLES" \
        --timeout "$TIMEOUT" \
        --seed "$SEED"
done

echo "Finished: $(date)"
