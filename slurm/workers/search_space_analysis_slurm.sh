#!/usr/bin/env bash
# SLURM compute worker: Search Space Reduction Analysis (THE KEY EXPERIMENT)
set -euo pipefail

echo "=== Search Space Analysis: SLURM Worker ==="
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
BENCH_CFG=$($PYTHON -c "import yaml,json; json.dump(yaml.safe_load(open('${CONFIG}'))['experiments']['search_space_analysis'], __import__('sys').stdout)")

N_STRINGS=$(echo "$BENCH_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('n_strings', 10000))")
MAX_TOKENS=$(echo "$BENCH_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('max_tokens', 30))")
SEED=$(echo "$BENCH_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")
OUT_DIR="${RESULTS_DIR}/search_space_analysis"

mkdir -p "$OUT_DIR"
echo "Config: n_strings=$N_STRINGS, max_tokens=$MAX_TOKENS, seed=$SEED"
echo "Output: $OUT_DIR"

python experiments/scripts/search_space_analysis.py \
    --n-strings "$N_STRINGS" \
    --max-tokens "$MAX_TOKENS" \
    --seed "$SEED" \
    --output "${OUT_DIR}/search_space_analysis.csv"

echo "Finished: $(date)"
