#!/usr/bin/env bash
# Local run of the permutation analysis with reduced parameters.
# Produces minimum viable results for the arXiv paper.
# Expected runtime: ~1.5 hours on a single core.
set -euo pipefail

PYTHON="${HOME}/.conda/envs/isalsr/bin/python"
OUT_DIR="/media/mpascual/Sandisk2TB/research/isalsr/results/arXiv_benchmarking/picasso/search_space_permutation"
mkdir -p "$OUT_DIR"

cd "$(dirname "$0")/../.."

echo "=== Local Permutation Analysis ==="
echo "Output: $OUT_DIR"
echo "Start:  $(date)"
echo ""

# k=1..8: exhaustive (all k! perms), 50 DAGs per (k,m)
# k=9..12: sampled (50K perms), 30 DAGs per (k,m)
# m=1,2 only (m=3 deferred to Picasso)

for m in 1 2; do
  for k in $(seq 1 12); do
    OUTFILE="${OUT_DIR}/perm_m${m}_k${k}.csv"
    if [ -f "$OUTFILE" ]; then
      echo "[SKIP] $OUTFILE already exists"
      continue
    fi

    if [ "$k" -le 8 ]; then
      N_DAGS=50
      N_PERMS=100000   # will be exhaustive since k!<=40320
    else
      N_DAGS=30
      N_PERMS=50000
    fi

    echo "[$(date +%H:%M:%S)] k=$k m=$m n_dags=$N_DAGS n_perms=$N_PERMS ..."
    $PYTHON experiments/scripts/search_space_permutation_analysis.py \
        --k-value "$k" \
        --num-vars "$m" \
        --n-dags "$N_DAGS" \
        --n-perms-sample "$N_PERMS" \
        --n-canon-verify 50 \
        --canon-timeout 5.0 \
        --seed 42 \
        --output "$OUTFILE" \
      2>&1 | grep -E "OVERALL|Invariant|Mean norm|Generated|WARNING" || true
    echo ""
  done
done

echo "=== Finished: $(date) ==="
echo "Results in: $OUT_DIR"
echo ""
echo "Generate figure with:"
echo "  $PYTHON experiments/scripts/generate_fig_search_space.py --data-dir $OUT_DIR"
