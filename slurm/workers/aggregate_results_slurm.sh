#!/usr/bin/env bash
# SLURM compute worker: Aggregate all results
# Runs AFTER all search experiments complete. Merges per-run CSVs,
# computes summary statistics, generates comparison tables.
set -euo pipefail

echo "=== Aggregate Results: SLURM Worker ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true

REPO_DIR="${ISALSR_REPO_DIR:?ERROR: ISALSR_REPO_DIR not set}"
CONFIG="${REPO_DIR}/slurm/config.yaml"
cd "$REPO_DIR"

RESULTS_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['results_dir'])")
AGG_DIR="${RESULTS_DIR}/aggregate"
mkdir -p "$AGG_DIR"

echo "Results dir: $RESULTS_DIR"
echo "Aggregate dir: $AGG_DIR"

# Merge per-run CSVs for each experiment
for EXP_DIR in random_search_canon random_search_nocanon \
               hill_climbing_canon hill_climbing_nocanon \
               gp_canon gp_nocanon; do
    EXP_PATH="${RESULTS_DIR}/${EXP_DIR}"
    MERGED="${EXP_PATH}/all_runs.csv"
    if [[ -d "$EXP_PATH" ]]; then
        echo "Merging: ${EXP_DIR}"
        # Merge all run_*.csv files, keeping header from first file only
        head -1 "$(ls "${EXP_PATH}"/run_*.csv 2>/dev/null | head -1)" > "$MERGED" 2>/dev/null || continue
        for f in "${EXP_PATH}"/run_*.csv; do
            tail -n +2 "$f" >> "$MERGED"
        done
        echo "  -> ${MERGED} ($(wc -l < "$MERGED") lines)"
    fi
done

# Run analysis script
python experiments/scripts/analyze_results.py \
    --results-dir "$RESULTS_DIR" \
    --output "${AGG_DIR}/summary.csv"

echo ""
echo "=== Aggregation complete: $(date) ==="
