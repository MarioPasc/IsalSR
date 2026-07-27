#!/usr/bin/env bash
# Compute-node smoke for the isalsr.core._native C++ extension (T01 AC-1).
#
# Proves four things that only a compute node can prove:
#   1. the extension BUILDS there (--no-build-isolation, offline);
#   2. it LOADS there, and is genuinely the C++ engine, not a silent fallback;
#   3. canonical strings are byte-identical to the Python engine on that node;
#   4. the measured speedup on Picasso hardware, which is the number the paper
#      must quote (the workstation figure is not comparable).
#
# CPU-only. No --gres, no GPU constraint. One core, matching production.
#SBATCH -J isalsr-smoke-cpp
#SBATCH --time=0-00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --constraint=cpu
#SBATCH --account=tic_163_uma

set -euo pipefail

START_TIME=$(date +%s)

echo "=========================================="
echo "Job:          ${SLURM_JOB_ID:-local}"
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
# time: the extension is linked -static-libstdc++ -static-libgcc precisely so
# that campaign tasks do not depend on a module being loaded.
module load "${GCC_MODULE}"
echo "[env] gcc: $(gcc --version | head -1)"
echo "[env] python: $(python --version)"

cd "${REPO_DIR}"

# NOTE: PYTHONPATH deliberately does NOT prepend ${REPO_DIR}/src.
# The compiled .so installs into site-packages/isalsr/core/, NOT into the
# source tree.  Putting src/ first would make `import isalsr` resolve to the
# source tree, which has no extension -- the engine would silently fall back
# to pure Python and this smoke would "pass" while proving nothing.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ------------------------------------------------- 1. build on a compute node
echo ""
echo "--- [1/4] building extension on the compute node ---"
pip install -e ".[dev]" --no-build-isolation -q
echo "build OK"

# --------------------------------------------- 2. engine identity (gate G5/G8)
echo ""
echo "--- [2/4] engine identity ---"
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

# ------------------------------------------------- 3. equivalence on this node
echo ""
echo "--- [3/4] cross-engine equivalence on this node ---"
python -m experiments.scripts.equivalence_gate --gate all --quick \
    --out "${LOGS_DIR}/equivalence_${SLURM_JOB_ID:-local}.json"

# ------------------------------------------ 4. speedup on Picasso hardware
echo ""
echo "--- [4/4] canonicalisation benchmark on Picasso hardware ---"
python -m experiments.scripts.bench_canonical --quick \
    --out "${LOGS_DIR}/bench_${SLURM_JOB_ID:-local}.json"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "Finished: $(date)"
echo "Duration: $((ELAPSED / 60))m $((ELAPSED % 60))s"
