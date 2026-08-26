#!/usr/bin/env bash
# Close T01 on Picasso hardware: AC-1 (rebuild), AC-3 gate 3, AC-8, AC-5.
#
# Distinct from slurm/smoke_cpp/, which certified AC-1 and §5.3 gates 1/2/4 at
# commit b34cded.  This worker adds the two things T16 made necessary:
#
#   * the EVOLVED-DAG equivalence gate (§5.3 gate 3), which the stored
#     trajectories could never supply because they persist only aggregate
#     counts -- so the DAGs are generated live from real short searches;
#   * the DECOMPOSED alphabet (§5.3 gate 6 / AC-8).  The adapters decompose by
#     default since T16, so a live search yields the paper's 12-label alphabet
#     without any special corpus construction;
#   * the AC-5 benchmark re-measured under BOTH encodings, paired on the same
#     host individuals, because decomposition moves DAGs across the paper's own
#     k-buckets and the engine effect must not be confounded with it.
#
# Pinned to --constraint=intel deliberately.  The published cost numbers and
# the earlier AC-5 table (job 1659650) both come from an Intel Xeon Gold 6230R
# (sd node).  Benchmarking on an AMD node would make the two tables
# incomparable, which is the entire purpose of measuring here rather than on
# the workstation.
#
# CPU-only, one core, matching production.  No --gres, no GPU constraint.
# The extension is single-threaded by design (no threading is permitted in
# isalsr.core._native), so extra cores would buy nothing.
#SBATCH -J isalsr-t01-close
#SBATCH --time=0-03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --constraint=intel
#SBATCH --account=tic_163_uma

set -euo pipefail

START_TIME=$(date +%s)
TAG="${SLURM_JOB_ID:-local}"

echo "=========================================="
echo "Job:          ${TAG}"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "Git commit:   $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "CPU:          $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
echo "=========================================="

# ---------------------------------------------------------------- environment
module_loaded=0
for m in miniconda/3 miniconda3 Miniconda3 anaconda3 Anaconda3; do
    if module avail 2>&1 | grep -qiE "(^|/)${m}([[:space:]]|/|$)"; then
        module load "$m" && module_loaded=1 && break
    fi
done
[ "$module_loaded" -eq 0 ] && echo "[env] no conda module; assuming conda on PATH"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

# A C++17 compiler is needed to BUILD.  It is deliberately NOT needed at run
# time: the extension is linked -static-libstdc++ -static-libgcc so that
# campaign tasks carry no module dependency.
module load "${GCC_MODULE}"
echo "[env] gcc: $(gcc --version | head -1)"
echo "[env] python: $(python --version)"

cd "${REPO_DIR}"

# NOTE: PYTHONPATH deliberately does NOT prepend ${REPO_DIR}/src.
# The compiled .so installs into site-packages/isalsr/core/, NOT into the
# source tree.  Putting src/ first would make `import isalsr` resolve to the
# source tree, which has no extension -- the engine would silently fall back to
# pure Python and every number below would measure nothing.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ------------------------------------------------- 1. build on a compute node
echo ""
echo "--- [1/5] building extension on the compute node (AC-1) ---"
pip install -e ".[dev]" --no-build-isolation -q
echo "build OK"

# --------------------------------------------- 2. engine identity (gate G5/G8)
echo ""
echo "--- [2/5] engine identity ---"
python - <<'PY'
import json, sys
from isalsr.core import backends, _native

info = backends.build_info()
print("engine     :", backends.engine())
print("so path    :", _native.__file__)
print("build_info :", json.dumps(info, sort_keys=True))

if backends.engine() != "cpp":
    sys.exit("FATAL: native engine did not load -- this run would measure pure Python")
if not _native.__file__.endswith(".so"):
    sys.exit(f"FATAL: _native resolved to {_native.__file__}, not an extension module")
if info.get("isa_level") != "x86-64-v3":
    sys.exit(f"FATAL: unexpected ISA level {info.get('isa_level')!r}")
print("engine identity OK")
PY

# ------------------------------- 3. §5.3 gates 1/2/4 regression on this node
echo ""
echo "--- [3/5] synthetic equivalence gates 1/2/4 (regression) ---"
python -m experiments.scripts.equivalence_gate --gate all --quick \
    --out "${LOGS_DIR}/t01_equivalence_${TAG}.json"

# ------------------- 4. §5.3 gate 3 + gate 6: EVOLVED, DECOMPOSED (AC-3, AC-8)
echo ""
echo "--- [4/5] evolved-DAG equivalence gate, decomposed alphabet (AC-3, AC-8) ---"
python -m experiments.scripts.equivalence_gate_evolved \
    --out "${LOGS_DIR}/t01_evolved_${TAG}.json"

# ------------- 5. AC-5: paired legacy-vs-split benchmark on Picasso hardware
echo ""
echo "--- [5/5] canonicalisation benchmark, both encodings, FULL (AC-5) ---"
python -m experiments.scripts.bench_canonical \
    --encodings legacy,split \
    --out "${LOGS_DIR}/t01_bench_${TAG}.json"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished: $(date)"
echo "Duration: $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "Reports:  ${LOGS_DIR}/t01_{equivalence,evolved,bench}_${TAG}.json"
