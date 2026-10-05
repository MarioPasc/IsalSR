#!/usr/bin/env bash
# =============================================================================
# Deploy the repository to Picasso and build the C++ engine with C2's recipe
# (SRBench black-box campaign, R3.2).  Run from the workstation.
# =============================================================================
# Layout (T01b).  C2's tree and env were deleted from Picasso after the
# campaign, so the black-box campaign rebuilds both the way C2 was built:
#
#   $FSCRATCH/repos/IsalSR        the whole repository, WITH .git (SP-1 checks
#                                 HEAD on the cluster); the project is under code/
#   $FSCRATCH/conda_envs/isalsr   created once by create_env.sh; holds an editable
#                                 install of $REPO/code, so isalsr, experiments and
#                                 benchmarks all resolve from the deployed tree and
#                                 the engine .so from the env's site-packages
#
# Steps, each fail-closed:
#   1. local tree clean (the record must name a real commit);
#   2. P1 on the workstation: every arm-deciding file is byte-identical to tag
#      campaign/c2 or a listed, justified exception (code_identity.py);
#   3. rsync the repository (with .git; without .claude/, caches, build dirs);
#   4. SP-1 from the remote side: remote HEAD == local HEAD, remote tree clean;
#   5. deploy record -> $REPO/code/slurm/blackbox/BBX_DEPLOY.json (gitignored):
#      commit, P1 report, SHA-256 of the files check_env.py re-verifies per task;
#   6. (unless --no-build) C2's build recipe, slurm/c2_smoke/deploy.sh: GCC 13.2.0
#      module (system g++ 7.5 cannot compile -march=x86-64-v3), `rm -rf build`,
#      `pip install -e . --force-reinstall --no-deps` WITH build isolation, pip's
#      status read without a pipe, then verify_build.py with every module purged.
#      Only addition: build_constraints.txt as PIP_CONSTRAINT (the backend
#      versions C2's own unpinned build resolved).
#
#   bash slurm/blackbox/deploy.sh              # from <repo>/code
#   bash slurm/blackbox/deploy.sh --no-build   # steps 1-5 only
# =============================================================================
set -euo pipefail

REMOTE="${ISALSR_REMOTE:-picasso}"
FSCRATCH_REMOTE="${ISALSR_FSCRATCH:-/mnt/home/users/tic_163_uma/mpascual/fscratch}"
REPO_REMOTE="${BBX_REPO_REMOTE:-${FSCRATCH_REMOTE}/repos/IsalSR}"
ENV_REMOTE="${BBX_ENV_REMOTE:-${FSCRATCH_REMOTE}/conda_envs/isalsr}"
GCC_MODULE="${ISALSR_GCC_MODULE:-gcc/13.2.0}"   # C2: compiler "gcc 13.2.0" in every run_log
DO_BUILD=true
[[ "${1:-}" == "--no-build" ]] && DO_BUILD=false

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="$(git -C "${CODE_DIR}" rev-parse --show-toplevel)"
cd "${CODE_DIR}"

# --- 1. clean tree ------------------------------------------------------------
if [[ -n "$(git status --porcelain)" ]]; then
    echo "FATAL: local tree is dirty; commit first so BBX_DEPLOY.json names a real commit:" >&2
    git status --short >&2
    exit 1
fi
LOCAL_HEAD="$(git rev-parse HEAD)"
echo "Local  HEAD: ${LOCAL_HEAD}  ($(git describe --tags --always --dirty))"

# --- 2. P1 on the workstation -------------------------------------------------
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
python3 slurm/blackbox/code_identity.py --repo "${REPO_DIR}" --json "${WORK}/identity.json" \
    || { echo "FATAL: P1 -- arm-deciding files differ from campaign/c2" >&2; exit 1; }

# --- 3. rsync -------------------------------------------------------------------
echo "Syncing ${REPO_DIR} -> ${REMOTE}:${REPO_REMOTE} (with .git) ..."
ssh "${REMOTE}" "mkdir -p ${REPO_REMOTE}"
# Excluded paths are all gitignored, so the remote tree is still clean for git.
# `.claude/` holds per-machine state, agent worktrees and private notes.
rsync -az --delete \
    --exclude '/.claude/' --exclude '__pycache__' --exclude '*.egg-info' \
    --exclude '/build/' --exclude '/code/build/' --exclude '.hypothesis' \
    --exclude '.mypy_cache' --exclude '.pytest_cache' --exclude '.ruff_cache' \
    --exclude '.coverage' --exclude '/code/slurm/blackbox/BBX_DEPLOY.json' \
    "${REPO_DIR}/" "${REMOTE}:${REPO_REMOTE}/"

# --- 4. SP-1 from the remote side -------------------------------------------------
REMOTE_STATE="$(ssh "${REMOTE}" "cd ${REPO_REMOTE} && printf '%s|%s' \"\$(git rev-parse HEAD)\" \"\$(git status --porcelain | wc -l)\"")"
REMOTE_HEAD="${REMOTE_STATE%%|*}"
REMOTE_DIRTY="${REMOTE_STATE##*|}"
echo "Remote HEAD: ${REMOTE_HEAD}  dirty_files=${REMOTE_DIRTY}"
[[ "${REMOTE_HEAD}" == "${LOCAL_HEAD}" ]] || { echo "FATAL: SP-1 -- remote HEAD != local HEAD" >&2; exit 1; }
[[ "${REMOTE_DIRTY}" == "0" ]] || { echo "FATAL: SP-1 -- remote tree dirty after sync" >&2; exit 1; }
echo "SP-1 OK: remote is exactly ${LOCAL_HEAD:0:7}, clean."

# --- 5. deploy record ---------------------------------------------------------------
python3 - "${WORK}/identity.json" "${WORK}/BBX_DEPLOY.json" "${REPO_REMOTE}" <<'PY'
import datetime, hashlib, json, pathlib, subprocess, sys

identity = json.loads(pathlib.Path(sys.argv[1]).read_text())
# Files whose content decides what a black-box cell computes or how it is run,
# relative to code/.  The arm-deciding C2 code is covered by P1 (code_identity).
files = sorted(
    [
        "experiments/models/orchestrator.py",
        "benchmarks/datasets/srbench_blackbox.py",
        "benchmarks/datasets/data/srbench_blackbox/manifest.csv",
        "experiments/configs/blackbox/udfs_srbench_blackbox.yaml",
        "experiments/configs/blackbox/bingo_srbench_blackbox.yaml",
        "experiments/scripts/c2_task_spec.py",
        "slurm/blackbox/worker.sh",
        "slurm/blackbox/check_env.py",
        "slurm/blackbox/env_requirements.txt",
        "slurm/c2_smoke/worker.sh",
    ]
    + [str(p) for p in pathlib.Path("experiments/models/udfs").glob("*.py")]
    + [str(p) for p in pathlib.Path("experiments/models/bingo").glob("*.py")]
    + [str(p) for p in pathlib.Path("benchmarks/datasets/data/srbench_blackbox").glob("*.tsv.gz")]
)
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
record = {
    "git_head": git("rev-parse", "HEAD"),
    "git_describe": git("describe", "--tags", "--always", "--dirty"),
    "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "deployed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "repo_remote": sys.argv[3],
    "code_identity": identity,
    "sha256": {f: hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest() for f in files},
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(record, indent=2, sort_keys=True))
print(f"deploy record: {record['git_head']} ({record['git_describe']}), {len(files)} files hashed, "
      f"P1 {'PASS' if identity['pass'] else 'FAIL'}")
PY
rsync -az "${WORK}/BBX_DEPLOY.json" "${REMOTE}:${REPO_REMOTE}/code/slurm/blackbox/BBX_DEPLOY.json"
ssh "${REMOTE}" "cd ${REPO_REMOTE} && test -z \"\$(git status --porcelain)\"" \
    || { echo "FATAL: SP-1 -- the deploy record dirtied the remote tree (is it gitignored?)" >&2; exit 1; }

${DO_BUILD} || { echo "Skipping build (--no-build)."; exit 0; }

# --- 6. build: C2's recipe (slurm/c2_smoke/deploy.sh) -------------------------------
# Never pipe `module load` (subshell: the PATH change is lost) and never read
# pip's status through a pipe.
echo "Building the native extension with ${GCC_MODULE} into ${ENV_REMOTE} ..."
ssh "${REMOTE}" "bash -l -c '
set -e
cd ${REPO_REMOTE}/code
module load ${GCC_MODULE}
export CXX=\$(which g++) CC=\$(which gcc)
echo \"  compiler: \$(\$CXX --version | head -1)\"
rm -rf build
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_REMOTE}
export PIP_CONSTRAINT=${REPO_REMOTE}/code/slurm/blackbox/build_constraints.txt PIP_NO_CACHE_DIR=1
mkdir -p build
LOG=${REPO_REMOTE}/code/build/bbx_pip_install.log
set +e
python -m pip install -v -e . --force-reinstall --no-deps > \$LOG 2>&1
RC=\$?
set -e
echo \"  PIP_EXIT=\$RC  (log: \$LOG)\"
[ \$RC -eq 0 ] || { tail -25 \$LOG; exit \$RC; }
grep -hoE \"(nanobind|scikit_build_core|ninja)-[0-9][^ ]*(whl|tar.gz)\" \$LOG | sort -u | sed \"s/^/  backend: /\"
module purge 2>/dev/null || true
python slurm/c2_smoke/verify_build.py
'"
echo "Deploy + build complete.  Next: RUNBOOK step 2 (preflight.sh on the login node)."
