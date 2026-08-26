#!/usr/bin/env bash
# SLURM compute worker: One-to-One Property Validation (P1-P4)
# Array job: SLURM_ARRAY_TASK_ID = num_vars (1, 2, or 3).
set -euo pipefail

echo "=== One-to-One Properties (P1-P4) - SLURM Worker ==="
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
EXP_CFG=$($PYTHON -c "import yaml,json; json.dump(yaml.safe_load(open('${CONFIG}'))['experiments']['onetoone_properties'], __import__('sys').stdout)")

N_STRINGS=$( echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('n_strings', 5000))")
MAX_TOKENS=$(echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('max_tokens', 20))")
TIMEOUT=$(   echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('timeout', 2.0))")
SEED=$(      echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")

# Array task = num_vars
NVARS=${SLURM_ARRAY_TASK_ID:-1}
OUT_DIR="${RESULTS_DIR}/onetoone_properties"
mkdir -p "$OUT_DIR"

echo "Config: n_strings=$N_STRINGS, max_tokens=$MAX_TOKENS, timeout=$TIMEOUT, seed=$SEED"
echo "Task: num_vars=$NVARS"
echo "Output: $OUT_DIR"

python experiments/scripts/onetoone_properties.py \
    --output-dir "$OUT_DIR" \
    --n-strings "$N_STRINGS" \
    --max-tokens "$MAX_TOKENS" \
    --num-vars "$NVARS" \
    --timeout "$TIMEOUT" \
    --seed "$SEED" \
    --plot

echo "Finished: $(date)"
