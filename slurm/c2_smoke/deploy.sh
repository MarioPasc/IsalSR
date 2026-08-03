#!/usr/bin/env bash
# =============================================================================
# Deploy the working tree to Picasso and re-establish SP-1 / SP-2
# =============================================================================
# Run from the workstation, from the repo root.
#
# Why this exists.  We deploy by rsync, not `git pull`, and the historical sync
# command excluded `.git`.  The consequence is that Picasso's `.git` describes
# whatever commit was there months ago while the working tree is today's code,
# so `git describe --tags --always --dirty` on a compute node reports a stale
# hash with `-dirty` appended -- **permanently**.  SP-1 ("you are running the
# commit you think you are") can then never pass, and the check that exists to
# catch a stale deployment instead cries wolf on every run until it is ignored.
#
# So: commit first, sync `.git` with everything else, and then VERIFY from the
# remote side rather than assuming.
#
#   bash slurm/c2_smoke/deploy.sh              # sync + verify + rebuild
#   bash slurm/c2_smoke/deploy.sh --no-build   # sync + verify only
# =============================================================================
set -euo pipefail

REMOTE="${ISALSR_REMOTE:-picasso}"
REPO_REMOTE="${ISALSR_REPO_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR}"
GCC_MODULE="${ISALSR_GCC_MODULE:-gcc/13.2.0}"
DO_BUILD=true
[[ "${1:-}" == "--no-build" ]] && DO_BUILD=false

# --- SP-1 precondition: a dirty local tree makes remote provenance meaningless.
if [[ -n "$(git status --porcelain)" ]]; then
    echo "FATAL: local tree is dirty. Commit before deploying, or SP-1 cannot pass:" >&2
    git status --short >&2
    exit 1
fi
LOCAL_HEAD="$(git rev-parse HEAD)"
LOCAL_DESC="$(git describe --tags --always --dirty)"
echo "Local  HEAD: ${LOCAL_HEAD}  (${LOCAL_DESC})"

echo "Syncing to ${REMOTE}:${REPO_REMOTE} ..."
rsync -az --delete \
    --exclude '__pycache__' --exclude '*.egg-info' --exclude 'results' \
    --exclude '.hypothesis' --exclude 'build' --exclude '.mypy_cache' \
    --exclude '.pytest_cache' --exclude '.ruff_cache' \
    ./ "${REMOTE}:${REPO_REMOTE}/"
echo "Sync done (including .git, so remote provenance is real)."

# --- SP-1, verified from the remote side, not assumed.
REMOTE_STATE="$(ssh "${REMOTE}" "cd ${REPO_REMOTE} && \
    printf '%s|%s|%s' \"\$(git rev-parse HEAD)\" \
                      \"\$(git describe --tags --always --dirty)\" \
                      \"\$(git status --porcelain | wc -l)\"")"
REMOTE_HEAD="${REMOTE_STATE%%|*}"
REMOTE_REST="${REMOTE_STATE#*|}"
REMOTE_DESC="${REMOTE_REST%%|*}"
REMOTE_DIRTY="${REMOTE_REST##*|}"
echo "Remote HEAD: ${REMOTE_HEAD}  (${REMOTE_DESC})  dirty_files=${REMOTE_DIRTY}"

[[ "${REMOTE_HEAD}" == "${LOCAL_HEAD}" ]] \
    || { echo "FATAL: SP-1 -- remote HEAD != local HEAD" >&2; exit 1; }
[[ "${REMOTE_DIRTY}" == "0" ]] \
    || { echo "FATAL: SP-1 -- remote tree is dirty after sync (${REMOTE_DIRTY} files)" >&2; exit 1; }
echo "SP-1 OK: remote is exactly ${LOCAL_DESC}, clean."

${DO_BUILD} || { echo "Skipping rebuild (--no-build)."; exit 0; }

# --- SP-2: rebuild, then verify the artefact rather than trusting pip.
# The system g++ is 7.5.0 and CANNOT compile -march=x86-64-v3 (a GCC 11+ flag),
# so a gcc module is mandatory.  Never pipe `module load` -- it runs in a
# subshell and the PATH change is silently lost, which is how an earlier attempt
# built against the system compiler while appearing to succeed.  Never read
# pip's status through a pipe either.
echo "Rebuilding native extension with ${GCC_MODULE} ..."
ssh "${REMOTE}" "bash -l -c '
set -e
cd ${REPO_REMOTE}
module load ${GCC_MODULE}
export CXX=\$(which g++) CC=\$(which gcc)
echo \"  compiler: \$(\$CXX --version | head -1)\"
rm -rf build
eval \"\$(conda shell.bash hook)\"
conda activate isalsr
python -m pip install -e . --force-reinstall --no-deps > /tmp/isalsr_build.log 2>&1
RC=\$?
echo \"  PIP_EXIT=\$RC\"
[ \$RC -eq 0 ] || { tail -25 /tmp/isalsr_build.log; exit \$RC; }
module purge 2>/dev/null || true
python - <<PY
import os, datetime, subprocess
from isalsr.core import _native, backends
so = _native.__file__
mt = datetime.datetime.fromtimestamp(os.path.getmtime(so))
last = subprocess.run(["git","log","-1","--format=%ct","--","src/isalsr/core/native"],
                      capture_output=True, text=True).stdout.strip()
print("  so:        ", so)
print("  so_mtime:  ", mt)
print("  engine:    ", backends.engine())
print("  build:     ", backends.build_info())
assert "site-packages" in so, "SP-2: .so is not in site-packages"
if last:
    assert os.path.getmtime(so) > float(last), "SP-2: .so is OLDER than the last C++ commit"
print("  SP-2 OK: artefact post-dates the last native commit and imports with modules purged.")
PY
'"
echo "Deploy complete."
