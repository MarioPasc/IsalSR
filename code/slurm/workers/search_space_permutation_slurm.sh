#!/usr/bin/env bash
# SLURM compute worker: Search Space Permutation Analysis
# Array job: SLURM_ARRAY_TASK_ID indexes into (k, m) grid.
#   task_id = (k-1) * n_vars + var_idx   (0-indexed internally, 1-indexed for SLURM)
#   Example with max_k=15, num_vars_list="1,2,3":
#     task  1 → k=1, m=1    task  2 → k=1, m=2    task  3 → k=1, m=3
#     task  4 → k=2, m=1    task  5 → k=2, m=2    task  6 → k=2, m=3
#     ...
#     task 43 → k=15, m=1   task 44 → k=15, m=2   task 45 → k=15, m=3
#
# Validates the O(k!) search space reduction claim.
set -euo pipefail

echo "=== Search Space Permutation Analysis - SLURM Worker ==="
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
EXP_CFG=$($PYTHON -c "import yaml,json; json.dump(yaml.safe_load(open('${CONFIG}'))['experiments']['search_space_permutation'], __import__('sys').stdout)")

N_DAGS=$(         echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('n_dags', 100))")
N_PERMS_SAMPLE=$( echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('n_perms_sample', 100000))")
N_CANON_VERIFY=$( echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('n_canon_verify', 100))")
CANON_TIMEOUT=$(  echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('canon_timeout', 5.0))")
SEED=$(           echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('seed', 42))")
NUM_VARS_LIST=$(  echo "$EXP_CFG" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('num_vars_list', '1,2'))")

# Parse num_vars_list into array
IFS=',' read -ra NVARS_ARR <<< "$NUM_VARS_LIST"
N_VARS_COUNT=${#NVARS_ARR[@]}

# Decode SLURM_ARRAY_TASK_ID into (k, m)
# task_id is 1-indexed; internally we use 0-indexed
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
TASK_IDX=$((TASK_ID - 1))
K_VALUE=$(( TASK_IDX / N_VARS_COUNT + 1 ))
VAR_IDX=$(( TASK_IDX % N_VARS_COUNT ))
NVARS=${NVARS_ARR[$VAR_IDX]}

OUT_DIR="${RESULTS_DIR}/search_space_permutation"
mkdir -p "$OUT_DIR"

echo "Config: k=$K_VALUE, m=$NVARS (task $TASK_ID of $((15 * N_VARS_COUNT)))"
echo "        n_dags=$N_DAGS, n_perms_sample=$N_PERMS_SAMPLE"
echo "        n_canon_verify=$N_CANON_VERIFY, canon_timeout=$CANON_TIMEOUT, seed=$SEED"
echo "Output: $OUT_DIR"

python experiments/scripts/search_space_permutation_analysis.py \
    --output "${OUT_DIR}/perm_m${NVARS}_k${K_VALUE}.csv" \
    --k-value "$K_VALUE" \
    --n-dags "$N_DAGS" \
    --n-perms-sample "$N_PERMS_SAMPLE" \
    --n-canon-verify "$N_CANON_VERIFY" \
    --num-vars "$NVARS" \
    --seed "$SEED" \
    --canon-timeout "$CANON_TIMEOUT"

echo "Finished: $(date)"
