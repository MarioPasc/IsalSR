#!/usr/bin/env bash
# SLURM compute worker: merge_cache (SINGLE TASK)
# Merges all cache shards from an array job into one HDF5 file.
# Run AFTER the generate_cache array job completes.
set -euo pipefail

echo "=== merge_cache: Single Task ==="
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

# Merge each experiment's shards.
EXPERIMENT="${CACHE_EXPERIMENT_NAME:?ERROR: CACHE_EXPERIMENT_NAME not set}"
NUM_VARS="${CACHE_NUM_VARS:?ERROR: CACHE_NUM_VARS not set}"

SHARD_DIR="${RESULTS_DIR}/precomputed_cache/${EXPERIMENT}"
OUTPUT="${SHARD_DIR}/cache_merged.h5"

echo "Experiment: ${EXPERIMENT}"
echo "Shard dir:  ${SHARD_DIR}"
echo "Output:     ${OUTPUT}"

python -m isalsr.precomputed.generate_cache \
    --mode merge \
    --num-variables "$NUM_VARS" \
    --input-dir "$SHARD_DIR" \
    --output "$OUTPUT"

echo ""
echo "=== Merge complete: $(date) ==="
