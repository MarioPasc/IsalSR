#!/usr/bin/env bash
# =============================================================================
# SRBench black-box campaign (R3.2) -- pre-flight.  Picasso LOGIN node, after
# deploy.sh.  Submits nothing.  Fails closed: exit 1 on the first failed check.
# =============================================================================
#   P1  the C2 tree the editable install points to is the C2 code, unmodified
#       (src/isalsr identical to tag campaign/c2, no uncommitted change under src/)
#   P2  imports: isalsr -> C2 tree; experiments + benchmarks -> bbx tree; engine
#       cpp with build hash 298fc1188bf1b051; deployed files hash as recorded
#   P3  the 20 vendored datasets load (SHA-256 vs manifest) and split, seed 1-10
#   P4  the packaging covers exactly 1,200 cells (launcher --dry-run)
#
#   cd $FSCRATCH/repos/IsalSR_bbx && bash slurm/blackbox/preflight.sh
# =============================================================================
set -euo pipefail

FSCRATCH="${ISALSR_FSCRATCH:-/mnt/home/users/tic_163_uma/mpascual/fscratch}"
BBX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C2_TREE="${BBX_C2_TREE:-${FSCRATCH}/repos/IsalSR}"
C2_TAG_COMMIT="2dd56fd76a9ace327c2fd949688a5c3b677c1bfe"   # tag campaign/c2

fail() { echo "[FAIL] $*" >&2; exit 1; }
ok()   { echo "[ OK ] $*"; }

echo "BBX pre-flight: bbx=${BBX_ROOT}  C2=${C2_TREE}"

# ---- P1: the C2 tree -------------------------------------------------------
[[ -d "${C2_TREE}/.git" ]] || fail "P1: ${C2_TREE} is not a git checkout"
C2_HEAD="$(git -C "${C2_TREE}" rev-parse HEAD)"
echo "       C2 HEAD     ${C2_HEAD}  ($(git -C "${C2_TREE}" describe --tags --always))"
echo "       C2 status   $(git -C "${C2_TREE}" status --porcelain | wc -l) path(s) differ from HEAD"
git -C "${C2_TREE}" cat-file -e "${C2_TAG_COMMIT}^{commit}" 2>/dev/null \
    || fail "P1: commit ${C2_TAG_COMMIT} (campaign/c2) is not in ${C2_TREE}/.git"
# The working tree's src/isalsr must equal the tag's, byte for byte, whatever
# HEAD is: that is what `isalsr` will import.  (diff against a commit compares
# the WORKING TREE, so uncommitted edits are caught too.)
git -C "${C2_TREE}" diff --quiet "${C2_TAG_COMMIT}" -- src/isalsr \
    || fail "P1: ${C2_TREE}/src/isalsr differs from campaign/c2: $(git -C "${C2_TREE}" diff --stat "${C2_TAG_COMMIT}" -- src/isalsr | tail -1)"
[[ -z "$(git -C "${C2_TREE}" status --porcelain --untracked-files=all -- src/isalsr)" ]] \
    || fail "P1: untracked or modified files under ${C2_TREE}/src/isalsr"
ok "P1 C2 tree: src/isalsr identical to campaign/c2 (${C2_TAG_COMMIT:0:7})"

# ---- P2: imports, engine, deployed hashes ----------------------------------
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate isalsr 2>/dev/null || true
PYTHON="${CONDA_PREFIX:?conda env isalsr not active}/bin/python"
export ISALSR_BBX_ROOT="${BBX_ROOT}"
export PYTHONPATH="${BBX_ROOT}/slurm/blackbox/pyboot${PYTHONPATH:+:${PYTHONPATH}}"
cd /tmp   # nothing may resolve from the working directory
"${PYTHON}" "${BBX_ROOT}/slurm/blackbox/check_env.py" --c2-tree "${C2_TREE}" --bbx-root "${BBX_ROOT}" \
    || fail "P2: import/engine/hash gate"
ok "P2 imports + engine + deployed hashes"

# ---- P3: data --------------------------------------------------------------
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

# ---- P4: packaging ---------------------------------------------------------
cd "${BBX_ROOT}"
OUT="$(bash slurm/blackbox/launcher.sh --dry-run)" || { echo "${OUT}"; fail "P4: launcher --dry-run"; }
grep -E '^Arrays:' <<<"${OUT}"
grep -qE 'cells: 1200$' <<<"${OUT}" || fail "P4: dry run does not cover 1,200 cells"
ok "P4 packaging covers 1,200 cells"

echo "PRE-FLIGHT PASS"
