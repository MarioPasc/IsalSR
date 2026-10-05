#!/usr/bin/env bash
# =============================================================================
# Create the Picasso conda env `isalsr` for the SRBench black-box campaign (R3.2).
# Run ONCE on the Picasso LOGIN node (it needs the network), before deploy.sh.
# =============================================================================
# C2's env ($FSCRATCH/conda_envs/isalsr) was deleted after the campaign.  This
# script rebuilds it with C2's interpreter build and, for every pip package,
# C2's version where it can be established; env_requirements.txt documents
# the provenance of each pin.  The `isalsr` package itself is NOT installed
# here: deploy.sh builds it from the deployed tree with C2's recipe.
#
# Choices that differ from a plain `conda create` + `pip install`:
#   * python is pinned to the exact `defaults` build C2 ran (h17756b0_1);
#   * torch comes from the CPU wheel index: the UDFS vendor imports it at
#     module level but no arm ever calls it (get_gradient has no caller), and
#     the CUDA build would add GBs and thousands of files for nothing;
#   * the conda package cache goes to a temporary directory on the login node
#     and pip neither caches nor byte-compiles: the fscratch FILE quota
#     (250k soft) is close to full, and neither cache changes what runs.
#
#   bash slurm/blackbox/create_env.sh            # refuses to touch an existing env
# =============================================================================
set -euo pipefail

FSCRATCH="${ISALSR_FSCRATCH:-/mnt/home/users/tic_163_uma/mpascual/fscratch}"
ENV_PREFIX="${BBX_ENV_PREFIX:-${FSCRATCH}/conda_envs/isalsr}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQS="${HERE}/env_requirements.txt"
PYTHON_SPEC="python=3.11.15=h17756b0_1"   # C2: "3.11.15 (main, Jun 11 2026, 15:20:16) [GCC 14.3.0]"
TORCH_SPEC="torch==2.12.0+cpu"            # see env_requirements.txt header ([amb] rule)
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
FREEZE_OUT="${BBX_FREEZE_OUT:-${ENV_PREFIX}/bbx_pip_freeze.txt}"

[[ -e "${ENV_PREFIX}" ]] && { echo "FATAL: ${ENV_PREFIX} exists; this script never modifies an env" >&2; exit 1; }
[[ -f "${REQS}" ]] || { echo "FATAL: ${REQS} missing" >&2; exit 1; }

source "$(conda info --base)/etc/profile.d/conda.sh"
PKGS_TMP="$(mktemp -d /tmp/isalsr_conda_pkgs.XXXXXX)"
trap 'rm -rf "${PKGS_TMP}"' EXIT
export CONDA_PKGS_DIRS="${PKGS_TMP}"
export PIP_NO_CACHE_DIR=1

echo "== conda create ${PYTHON_SPEC} -> ${ENV_PREFIX}"
conda create -y -p "${ENV_PREFIX}" --override-channels -c defaults "${PYTHON_SPEC}"
conda activate "${ENV_PREFIX}"
PY="${ENV_PREFIX}/bin/python"
"${PY}" -VV

echo "== pip install -r ${REQS}"
"${PY}" -m pip install --no-compile -r "${REQS}"
echo "== pip install ${TORCH_SPEC} (CPU index, deps pinned above)"
"${PY}" -m pip install --no-compile --no-deps --index-url "${TORCH_INDEX}" "${TORCH_SPEC}"

echo "== pip check"
"${PY}" -m pip check
"${PY}" -m pip freeze --all > "${FREEZE_OUT}"
echo "pip freeze -> ${FREEZE_OUT}"
# bingo imports mpi4py, whose import dlopen()s libmpi: same module list as C2's worker.
for mod in openmpi_gcc/5.0.9_gcc7 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/5.0.9_gcc14; do
    module load "$mod" 2>/dev/null && break
done
"${PY}" - <<'PY'
import importlib, sys
for m in ("numpy", "scipy", "sympy", "sklearn", "pandas", "bingo", "mpi4py", "torch", "stopit", "yaml"):
    try:
        mod = importlib.import_module(m)
        print(f"  import {m:8s} ok  {getattr(mod, '__version__', '')}")
    except Exception as exc:  # report every failure, then fail
        print(f"  import {m:8s} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
PY
echo "Env ready: ${ENV_PREFIX}  ($(find "${ENV_PREFIX}" | wc -l) files)"
