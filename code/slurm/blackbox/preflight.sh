#!/usr/bin/env bash
# =============================================================================
# SRBench black-box campaign (R3.2) -- pre-flight.  Picasso LOGIN node, after
# create_env.sh and deploy.sh.  Submits nothing.  Fails closed: exit 1 on the
# first failed check.
# =============================================================================
#   P1  the deployed repository is the deployed commit, clean, and its
#       arm-deciding files equal tag campaign/c2 (code_identity.py, re-run here
#       on the deployed .git; deploy.sh ran it on the workstation)
#   P2  gate (check_env.py): modules -> deployed tree, engine cpp with build
#       hash 298fc1188bf1b051 from site-packages, deployed SHA-256s, C2's python
#       and every pinned library version
#   P3  the 20 vendored datasets load (SHA-256 vs manifest) and split, seed 1-10
#   P4  the packaging covers exactly 1,200 cells (launcher --dry-run)
#   P5  engine equivalence on this build: verify_build.py (C2 SP-2), the
#       cpp-vs-python differential tests, C2's equivalence gate (all three
#       gates, full corpus), and the black-box protocol tests
#
#   cd $FSCRATCH/repos/IsalSR/code && bash slurm/blackbox/preflight.sh
# =============================================================================
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(git -C "${CODE_ROOT}" rev-parse --show-toplevel)"
OUT_DIR="${BBX_PREFLIGHT_DIR:-${CODE_ROOT}/build/bbx_preflight}"   # gitignored

fail() { echo "[FAIL] $*" >&2; exit 1; }
ok()   { echo "[ OK ] $*"; }

echo "BBX pre-flight: repo=${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

for mod in openmpi_gcc/5.0.9_gcc7 openmpi_gcc/5.0.9_gcc15 openmpi_gcc/5.0.9_gcc14; do
    module load "$mod" 2>/dev/null && break
done
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true
PYTHON="${CONDA_PREFIX:?conda env isalsr not active}/bin/python"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# ---- P1: the deployed repository ----------------------------------------------
HEAD="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
DEPLOYED="$("${PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1]))['git_head'])" \
            "${CODE_ROOT}/slurm/blackbox/BBX_DEPLOY.json")" || fail "P1: no BBX_DEPLOY.json; run deploy.sh"
echo "       HEAD ${HEAD}  ($(git -C "${REPO_ROOT}" describe --tags --always))  deployed ${DEPLOYED}"
[[ "${HEAD}" == "${DEPLOYED}" ]] || fail "P1: HEAD != deployed commit"
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] \
    || fail "P1: deployed tree differs from HEAD: $(git -C "${REPO_ROOT}" status --short | head -5)"
"${PYTHON}" "${CODE_ROOT}/slurm/blackbox/code_identity.py" --repo "${REPO_ROOT}" \
    --json "${OUT_DIR}/code_identity.json" || fail "P1: arm-deciding files differ from campaign/c2"
ok "P1 deployed tree = ${HEAD:0:7}, clean, arm-deciding files = campaign/c2"

# ---- P2: gate -------------------------------------------------------------------
cd /tmp   # nothing may resolve from the working directory
"${PYTHON}" "${CODE_ROOT}/slurm/blackbox/check_env.py" --code-root "${CODE_ROOT}" \
    || fail "P2: import/engine/hash/version gate"
ok "P2 modules + engine + deployed hashes + versions"

# ---- P3: data -----------------------------------------------------------------
"${PYTHON}" - <<'PY' || fail "P3: data"
from benchmarks.datasets import srbench_blackbox as bbx
from experiments.models.provenance import data_fingerprint
n = 0
for b in bbx.SRBENCH_BLACKBOX_BENCHMARKS:
    for seed in range(1, 11):
        arrays = bbx.generate_data(b, seed=seed)
        n += 1
    print(f"       {b['name']:30s} n_train={arrays[0].shape[0]:5d} n_test={arrays[2].shape[0]:5d} "
          f"fp(seed10)={data_fingerprint(*arrays)[:12]}")
assert n == 200, n
PY
ok "P3 20 datasets x 10 seeds load (sha256 vs manifest) and split"

# ---- P4: packaging -------------------------------------------------------------
cd "${CODE_ROOT}"
OUT="$(BBX_PYTHON="${PYTHON}" bash slurm/blackbox/launcher.sh --dry-run)" \
    || { echo "${OUT}"; fail "P4: launcher --dry-run"; }
grep -E '^Arrays:' <<<"${OUT}"
grep -qE 'cells: 1200$' <<<"${OUT}" || fail "P4: dry run does not cover 1,200 cells"
ok "P4 packaging covers 1,200 cells"

# ---- P5: engine equivalence on this build ----------------------------------------
cd "${CODE_ROOT}"
"${PYTHON}" slurm/c2_smoke/verify_build.py || fail "P5: verify_build.py (SP-2)"
"${PYTHON}" -m pytest -q -p no:cacheprovider -o addopts="" \
    tests/unit/test_native_build.py tests/unit/test_native_canonical.py \
    tests/unit/test_native_s2d.py tests/unit/test_native_datastructures.py \
    tests/unit/test_equivalence_gate.py \
    tests/unit/test_srbench_blackbox.py tests/unit/test_srbench_blackbox_configs.py \
    > "${OUT_DIR}/pytest_engine.log" 2>&1 \
    || { tail -30 "${OUT_DIR}/pytest_engine.log"; fail "P5: engine/protocol tests"; }
tail -1 "${OUT_DIR}/pytest_engine.log"
"${PYTHON}" experiments/scripts/equivalence_gate.py --gate all --backend-a python --backend-b cpp \
    --out "${OUT_DIR}/equivalence_gate.json" > "${OUT_DIR}/equivalence_gate.log" 2>&1 \
    || { tail -30 "${OUT_DIR}/equivalence_gate.log"; fail "P5: equivalence gate"; }
"${PYTHON}" - "${OUT_DIR}/equivalence_gate.json" <<'PY' || fail "P5: equivalence gate report"
import json, sys
r = json.load(open(sys.argv[1]))
gates = {g: r[g]["pass"] for g in ("gate1", "gate2", "gate3")}
print(f"       equivalence gate: pass={r['pass']} self_comparison={r['self_comparison']} {gates}")
assert r["pass"] is True and r["self_comparison"] is False and all(gates.values())
PY
ok "P5 build verified; cpp == python on the differential tests and the full equivalence gate"

echo "PRE-FLIGHT PASS"
