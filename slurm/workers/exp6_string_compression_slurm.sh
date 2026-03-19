#!/usr/bin/env bash
# SLURM compute worker: String Compression Analysis (Experiment 6)
set -euo pipefail

echo "=== Experiment 6: String Compression - SLURM Worker ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true
PYTHON="$(conda run -n isalsr which python 2>/dev/null || echo python3)"

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
CONFIG="${REPO_DIR}/slurm/config.yaml"
cd "$REPO_DIR"

RESULTS_DIR=$($PYTHON -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['results_dir'])")
EXP_CFG=$($PYTHON -c "import yaml,json; json.dump(yaml.safe_load(open('${CONFIG}'))['experiments']['exp6_string_compression'], __import__('sys').stdout)")

N_STRINGS=$( echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('n_strings', 10000))")
MAX_TOKENS=$(echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('max_tokens', 30))")
SEED=$(      echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")

OUT_DIR="${RESULTS_DIR}/exp6_string_compression"
mkdir -p "$OUT_DIR"

echo "Config: n_strings=$N_STRINGS, max_tokens=$MAX_TOKENS, seed=$SEED"
echo "Output: $OUT_DIR"

# Run for num_vars 1, 2, 3
for NVARS in 1 2 3; do
    echo "--- num_vars=$NVARS ---"
    python experiments/scripts/exp6_string_compression.py \
        --output "${OUT_DIR}/compression_v${NVARS}.csv" \
        --n-strings "$N_STRINGS" \
        --max-tokens "$MAX_TOKENS" \
        --num-vars "$NVARS" \
        --seed "$SEED" \
        --plot
done

echo "Finished: $(date)"
