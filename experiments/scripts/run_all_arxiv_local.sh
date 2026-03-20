#!/usr/bin/env bash
# =============================================================================
# Local test runner for all arXiv experiments (reduced parameters)
# =============================================================================
#
# Runs all 6 experiments with small parameters for quick local validation.
# Production runs use SLURM via slurm/launch.sh with full parameters.
#
# Usage:
#   bash experiments/scripts/run_all_arxiv_local.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_DIR"

PYTHON="${HOME}/.conda/envs/isalsr/bin/python"
OUT="/media/mpascual/Sandisk2TB/research/isalsr/results/arXiv_benchmarking/local"

echo "=============================================="
echo "IsalSR arXiv Experiments - Local Test Runner"
echo "=============================================="
echo "Repo:    ${REPO_DIR}"
echo "Output:  ${OUT}"
echo "Python:  ${PYTHON}"
echo ""

# Ensure output directory exists
mkdir -p "$OUT"

# --- Experiment 1: Shortest Path (illustrative, fast) ---
echo "=== Experiment 1: Shortest Path ==="
$PYTHON experiments/scripts/exp1_shortest_path.py \
    --output-dir "$OUT/exp1_shortest_path"
echo ""

# --- Experiment 2: Neighborhood (illustrative, fast) ---
echo "=== Experiment 2: Distance-1 Neighborhood ==="
$PYTHON experiments/scripts/exp2_neighborhood.py \
    --output-dir "$OUT/exp2_neighborhood"
echo ""

# --- One-to-One Property Validation P1-P4 (reduced: 200 strings) ---
echo "=== One-to-One Properties (P1-P4) ==="
$PYTHON experiments/scripts/onetoone_properties.py \
    --output-dir "$OUT/onetoone_properties" \
    --n-strings 200 \
    --max-tokens 15 \
    --timeout 2 \
    --plot
echo ""

# --- Experiment 3: Canonicalization Time (reduced: max 6 nodes, 5 samples) ---
echo "=== Experiment 3: Canonicalization Time ==="
$PYTHON experiments/scripts/exp3_canonicalization_time.py \
    --output "$OUT/exp3_canonicalization_time/timing.csv" \
    --max-nodes 6 \
    --samples-per-node 5 \
    --timeout 15 \
    --plot
echo ""

# --- Experiment 4: Search Space Reduction (reduced: 50 strings, single token length) ---
echo "=== Experiment 4: Search Space Reduction ==="
$PYTHON experiments/scripts/search_space_analysis.py \
    --n-strings 50 \
    --max-tokens-list "10" \
    --n-bootstrap 100 \
    --output "$OUT/exp4_search_space/reduction.csv" \
    --plot
echo ""

# --- Experiment 5: Pruning Accuracy (reduced: max 6 nodes, 10 samples) ---
echo "=== Experiment 5: Pruning Accuracy ==="
$PYTHON experiments/scripts/exp5_pruning_accuracy.py \
    --output "$OUT/exp5_pruning_accuracy/accuracy.csv" \
    --max-nodes 6 \
    --samples-per-node 10 \
    --timeout 15 \
    --plot
echo ""

# --- Experiment 6: String Compression (reduced: 100 strings, short timeout) ---
echo "=== Experiment 6: String Compression ==="
$PYTHON experiments/scripts/exp6_string_compression.py \
    --output "$OUT/exp6_string_compression/compression.csv" \
    --n-strings 100 \
    --max-tokens 15 \
    --timeout 2 \
    --plot
echo ""

# --- Analysis: Aggregate all results ---
echo "=== Aggregating Results ==="
$PYTHON experiments/scripts/analyze_arxiv_results.py \
    --results-dir "$OUT"
echo ""

echo "=============================================="
echo "All experiments complete!"
echo "Results: ${OUT}"
echo "=============================================="
