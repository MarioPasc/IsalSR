#!/usr/bin/env bash
# SLURM compute worker: Search Space Reduction Analysis (Experiment 4, enhanced)
# Array job: SLURM_ARRAY_TASK_ID = index into max_tokens_list (1-indexed).
set -euo pipefail

echo "=== Search Space Analysis: SLURM Worker ==="
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
BENCH_CFG=$($PYTHON -c "import yaml,json; json.dump(yaml.safe_load(open('${CONFIG}'))['experiments']['search_space_analysis'], __import__('sys').stdout)")

N_STRINGS=$(     echo "$BENCH_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('n_strings', 10000))")
MAX_TOKENS_LIST=$(echo "$BENCH_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('max_tokens_list', '15,20,25,30'))")
INCLUDE_FEYNMAN=$(echo "$BENCH_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('include_feynman', True))")
SEED=$(           echo "$BENCH_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")

# Array task index (1-indexed) selects which max_tokens value to process.
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

OUT_DIR="${RESULTS_DIR}/search_space_analysis"
mkdir -p "$OUT_DIR"

echo "Config: n_strings=$N_STRINGS, max_tokens_list=$MAX_TOKENS_LIST, seed=$SEED, include_feynman=$INCLUDE_FEYNMAN"
echo "Task: index=$TASK_ID"
echo "Output: $OUT_DIR"

FEYNMAN_FLAG=""
if [[ "$INCLUDE_FEYNMAN" == "True" ]]; then
    FEYNMAN_FLAG="--include-feynman"
fi

python experiments/scripts/search_space_analysis.py \
    --n-strings "$N_STRINGS" \
    --max-tokens-list "$MAX_TOKENS_LIST" \
    --max-tokens-index "$TASK_ID" \
    --seed "$SEED" \
    --output "${OUT_DIR}/reduction_T${TASK_ID}.csv" \
    $FEYNMAN_FLAG

echo "Finished: $(date)"
