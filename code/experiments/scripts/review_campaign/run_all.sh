#!/usr/bin/env bash
# Rebuild the whole analysis tree of the revision campaign from the corpus.
#
#   bash experiments/scripts/review_campaign/run_all.sh [CORPUS] [ANALYSES]
#
# CORPUS defaults to the local backup of the campaign and ANALYSES to the
# `analyses/` tree beside it. Nothing here writes into the corpus: the paired
# statistics and aggregates it already carries are read, never recomputed, and
# the two flattened views are symlink trees under a scratch root.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${ISALSR_PYTHON:-$HOME/.conda/envs/isalsr/bin/python}"

CORPUS="${1:-/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/review/c2_3arm}"
ANALYSES="${2:-${CORPUS}/analyses}"
SCRATCH="${ISALSR_SCRATCH:-$HOME/c2_work}"

cd "$REPO"
mkdir -p "$SCRATCH" "$ANALYSES"/{data,values,tables,figures,pipeline}

# ---------------------------------------------------------------------------
# 1. Flattened views. The manuscript reports one 70-problem suite; the seven
#    per-suite directories are the campaign's own layout. The 50-problem view
#    is the suite as it stood before the revision extended it.
# ---------------------------------------------------------------------------
"$PYTHON" - "$CORPUS" "$SCRATCH" <<'PY'
import shutil, sys
from pathlib import Path
from experiments.scripts.review_campaign.config import METHODS, SUITES, SUITES_CORE

corpus, scratch = Path(sys.argv[1]), Path(sys.argv[2])
for name, suites in (("flat70", SUITES), ("flat50", SUITES_CORE)):
    root = scratch / name
    if root.exists():
        shutil.rmtree(root)
    n = 0
    for method in METHODS:
        for suite in suites:
            for problem in sorted((corpus / method / suite).iterdir()):
                if not problem.is_dir():
                    continue
                dest = root / method / "benchmark" / problem.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(problem)
                n += 1
    print(f"{name}: {n // len(METHODS)} problems x {len(METHODS)} methods")
PY

# ---------------------------------------------------------------------------
# 2. Paired statistics and the paired test across problems, on both views and
#    per suite. `--allow-mixed-provenance` is required and correct: flattening
#    pools the seven per-suite configurations, which differ in the benchmark
#    block and, on three suites, in the host operator set. Commit, build hash
#    and engine are uniform across all 12,600 cells.
# ---------------------------------------------------------------------------
for VIEW in flat70 flat50; do
  "$PYTHON" -m experiments.models.analyze \
    --results-dir "$SCRATCH/$VIEW" --methods udfs,bingo --benchmarks benchmark \
    --variants baseline,hash,isalsr --allow-mixed-provenance
  rm -rf "$ANALYSES/pipeline/$VIEW"
  cp -r "$SCRATCH/$VIEW/analysis" "$ANALYSES/pipeline/$VIEW"
done

"$PYTHON" -m experiments.models.analyze \
  --results-dir "$CORPUS" --methods udfs,bingo \
  --benchmarks nguyen,feynman,hard,cherrypicked,roundoff,feynman_remainder,strogatz \
  --variants baseline,hash,isalsr
rm -rf "$ANALYSES/pipeline/by_suite"
cp -r "$CORPUS/analysis" "$ANALYSES/pipeline/by_suite"
rm -rf "$CORPUS/analysis"

# ---------------------------------------------------------------------------
# 3. Derived data, resolved placeholder values, tables, figures.
# ---------------------------------------------------------------------------
"$PYTHON" -m experiments.scripts.review_campaign.extract_cells \
  --corpus "$CORPUS" --analyses "$ANALYSES"
"$PYTHON" -m experiments.scripts.review_campaign.derive \
  --corpus "$CORPUS" --analyses "$ANALYSES"
"$PYTHON" -m experiments.scripts.review_campaign.values --analyses "$ANALYSES"
"$PYTHON" -m experiments.scripts.review_campaign.tables --analyses "$ANALYSES"
"$PYTHON" -m experiments.scripts.review_campaign.figures --analyses "$ANALYSES"

"$PYTHON" -m experiments.scripts.review_campaign.cd_diagram \
  --results-dir "$SCRATCH/flat70" --analyses "$ANALYSES"

# ---------------------------------------------------------------------------
# 4. Every number the manuscript quotes, checked against what was just derived.
#    Exits non-zero on the first drift, so a stale literal cannot survive a
#    rebuild unnoticed.
# ---------------------------------------------------------------------------
"$PYTHON" -m experiments.scripts.review_campaign.verify --analyses "$ANALYSES"

echo "analysis tree rebuilt in $ANALYSES"
