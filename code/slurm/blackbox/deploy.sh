#!/usr/bin/env bash
# =============================================================================
# Deploy the SRBench black-box tree to Picasso (R3.2).  Run from the workstation.
# =============================================================================
# What goes where, and why:
#
#   $FSCRATCH/repos/IsalSR       C2 tree.  NOT touched.  Its editable install
#                                provides `isalsr` and the C++ engine (build hash
#                                298fc1188bf1b051), so the canonicaliser is C2's
#                                by construction: nothing is rebuilt or reinstalled.
#   $FSCRATCH/repos/IsalSR_bbx   this deploy: code/{experiments,benchmarks,slurm}
#                                only.  `src/` is deliberately NOT shipped, so no
#                                copy of `isalsr` exists in this tree to be
#                                imported by mistake.
#
# The editable install also redirects `experiments` and `benchmarks` to the C2
# tree (pyproject wheel.packages), so jobs run with slurm/blackbox/pyboot on
# PYTHONPATH and ISALSR_BBX_ROOT pointing here; see pyboot/sitecustomize.py.
#
# A deploy record, BBX_DEPLOY.json, is written into the deployed tree: the local
# commit and the SHA-256 of every file that decides what an arm computes.
# check_env.py re-verifies those hashes on the cluster, before every task.
#
#   bash slurm/blackbox/deploy.sh             # from <repo>/code, clean tree only
# =============================================================================
set -euo pipefail

REMOTE="${ISALSR_REMOTE:-picasso}"
BBX_REMOTE="${BBX_ROOT_REMOTE:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/IsalSR_bbx}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${CODE_DIR}"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "FATAL: local tree is dirty; commit first so BBX_DEPLOY.json names a real commit:" >&2
    git status --short >&2
    exit 1
fi

RECORD="$(mktemp)"
trap 'rm -f "${RECORD}"' EXIT
python3 - "${RECORD}" <<'PY'
import hashlib, json, pathlib, subprocess, sys, datetime

files = sorted(
    [
        "experiments/models/orchestrator.py",
        "benchmarks/datasets/srbench_blackbox.py",
        "benchmarks/datasets/data/srbench_blackbox/manifest.csv",
        "experiments/configs/blackbox/udfs_srbench_blackbox.yaml",
        "experiments/configs/blackbox/bingo_srbench_blackbox.yaml",
        "experiments/scripts/c2_task_spec.py",
        "slurm/blackbox/pyboot/sitecustomize.py",
        "slurm/blackbox/worker.sh",
        "slurm/c2_smoke/worker.sh",
    ]
    + [str(p) for p in pathlib.Path("experiments/models/udfs").glob("*.py")]
    + [str(p) for p in pathlib.Path("experiments/models/bingo").glob("*.py")]
)
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
record = {
    "git_head": git("rev-parse", "HEAD"),
    "git_describe": git("describe", "--tags", "--always", "--dirty"),
    "deployed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "sha256": {f: hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest() for f in files},
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(record, indent=2, sort_keys=True))
print(f"deploy record: {record['git_head']} ({record['git_describe']}), {len(files)} files hashed")
PY

echo "Syncing code/{experiments,benchmarks,slurm} -> ${REMOTE}:${BBX_REMOTE}"
ssh "${REMOTE}" "mkdir -p ${BBX_REMOTE}"
rsync -az --delete \
    --exclude '__pycache__' --exclude '.mypy_cache' --exclude '.ruff_cache' \
    experiments benchmarks slurm "${REMOTE}:${BBX_REMOTE}/"
rsync -az "${RECORD}" "${REMOTE}:${BBX_REMOTE}/BBX_DEPLOY.json"
ssh "${REMOTE}" "test ! -e ${BBX_REMOTE}/src" \
    || { echo "FATAL: ${BBX_REMOTE}/src exists; remove it (isalsr must come from the C2 tree)" >&2; exit 1; }
echo "Deploy complete.  Next: RUNBOOK step 2 (preflight.sh on the login node)."
