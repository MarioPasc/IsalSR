#!/usr/bin/env bash
#
# Regenerate the `codeocean` branch from `main`.
#
#   bash tools/make_codeocean_branch.sh [source-ref] [target-branch]
#
# Why this exists
# ---------------
# Code Ocean's git import only takes four directory names — code, data,
# environment, metadata — and files their *contents* into the matching capsule
# directories. Anything outside them is neither mounted during a Reproducible
# Run nor selectable as the master script, so a repository whose sources sit at
# its root produces a capsule that cannot see its own code.
#
# The fix is to put the whole project under `code/`. Doing that on `main` would
# move pyproject.toml out of the repository root and break `pip install
# git+...`, the README paths and the docs site. So `main` keeps the normal
# layout and this script mechanically derives the capsule layout onto a separate
# branch:
#
#     main                        codeocean
#       pyproject.toml     ->       code/pyproject.toml
#       src/               ->       code/src/
#       tests/             ->       code/tests/
#       run                ->       code/run
#       environment/       ->       environment/     (unchanged)
#       metadata/          ->       metadata/        (unchanged)
#
# The branch is derived, never edited by hand: every change belongs on `main`,
# then this script is re-run. It is regenerated from scratch each time, so there
# is no merge to resolve.
#
set -euo pipefail

SOURCE_REF="${1:-main}"
TARGET_BRANCH="${2:-codeocean}"

# Directories Code Ocean recognises and that therefore stay at the root.
# `code` is listed so a re-run is idempotent if one ever exists on the source.
KEEP_AT_ROOT=(environment metadata data code)

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

if ! git rev-parse --verify --quiet "${SOURCE_REF}" >/dev/null; then
    echo "FATAL: no such ref: ${SOURCE_REF}" >&2
    exit 1
fi

SOURCE_SHA="$(git rev-parse "${SOURCE_REF}")"
STAGE="$(mktemp -d)"
INDEX="$(mktemp -u)"
cleanup() { rm -rf "${STAGE}" "${INDEX}"; }
trap cleanup EXIT

echo "== deriving ${TARGET_BRANCH} from ${SOURCE_REF} (${SOURCE_SHA:0:8}) =="

# Tracked files only — exactly what the git import would deliver.
git archive --format=tar "${SOURCE_REF}" | tar -x -C "${STAGE}"

mkdir -p "${STAGE}/code"
shopt -s dotglob nullglob
for entry in "${STAGE}"/*; do
    base="$(basename "${entry}")"
    skip=""
    for keep in "${KEEP_AT_ROOT[@]}"; do
        [ "${base}" = "${keep}" ] && skip=1 && break
    done
    [ -n "${skip}" ] && continue
    mv "${entry}" "${STAGE}/code/"
done
shopt -u dotglob nullglob

# The master script must be executable in the capsule.
[ -f "${STAGE}/code/run" ] && chmod +x "${STAGE}/code/run"

# Sanity: the run script and the project marker must be siblings, or the capsule
# will start and immediately fail to locate the project.
if [ ! -f "${STAGE}/code/run" ] || [ ! -f "${STAGE}/code/pyproject.toml" ]; then
    echo "FATAL: code/run and code/pyproject.toml must both exist after the move." >&2
    ls -la "${STAGE}/code" >&2
    exit 1
fi
if [ ! -f "${STAGE}/environment/Dockerfile" ] || [ ! -f "${STAGE}/metadata/metadata.yml" ]; then
    echo "FATAL: environment/Dockerfile and metadata/metadata.yml must stay at the root." >&2
    exit 1
fi

# Build the commit with a throwaway index so the working tree is untouched.
# --force: .gitignore now sits at code/.gitignore and its patterns must not
# drop files that were tracked on the source ref.
export GIT_INDEX_FILE="${INDEX}"
GIT_WORK_TREE="${STAGE}" git add -A --force .
TREE="$(GIT_WORK_TREE="${STAGE}" git write-tree)"
unset GIT_INDEX_FILE

MESSAGE="chore(codeocean): regenerate capsule layout from ${SOURCE_REF} ${SOURCE_SHA:0:8}

Derived by tools/make_codeocean_branch.sh. Do not edit this branch by hand:
make the change on ${SOURCE_REF} and re-run the script."

if git rev-parse --verify --quiet "${TARGET_BRANCH}" >/dev/null; then
    PARENT="$(git rev-parse "${TARGET_BRANCH}")"
    if [ "$(git rev-parse "${PARENT}^{tree}")" = "${TREE}" ]; then
        echo "   ${TARGET_BRANCH} is already up to date with ${SOURCE_REF}."
        exit 0
    fi
    COMMIT="$(git commit-tree "${TREE}" -p "${PARENT}" -m "${MESSAGE}")"
else
    COMMIT="$(git commit-tree "${TREE}" -m "${MESSAGE}")"
fi

git update-ref "refs/heads/${TARGET_BRANCH}" "${COMMIT}"

echo "   ${TARGET_BRANCH} -> ${COMMIT:0:8}"
echo
echo "capsule root:"
git ls-tree --name-only "${TARGET_BRANCH}" | sed 's/^/     /'
echo
echo "Push it with:"
echo "     git push -f origin ${TARGET_BRANCH}"
