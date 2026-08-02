"""Write ``.provenance.json`` stamping the local tree's commit and file hashes.

Run on the workstation immediately before rsyncing to Picasso.  The node's own
``.git`` is not synced -- rsync excludes it, and rewriting a remote checkout's
git state is destructive -- so a commit id read on the node is stale and proves
nothing.  This stamp travels with the files and lets SP-1 verify that the bytes
on the compute node are the bytes that were committed.

Refuses to write a stamp for a dirty tree: a probe run against uncommitted work
is unreproducible, which is precisely what SP-1 exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

# Sources whose bytes must match for a T04 probe result to be trustworthy.
TRACKED_GLOBS = (
    "src/isalsr/core/canonical.py",
    "src/isalsr/core/backends.py",
    "src/isalsr/baselines/*.py",
    "experiments/models/orchestrator.py",
    "experiments/models/schemas.py",
    "experiments/models/bingo/isalsr_runner.py",
    "experiments/models/udfs/isalsr_runner.py",
    "experiments/models/bingo/adapter.py",
    "experiments/models/udfs/adapter.py",
    "experiments/models/commutative_encoding.py",
    "slurm/t04_probe/*.py",
    "slurm/t04_probe/*.sh",
    "slurm/t04_probe/tasks.txt",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def main() -> int:
    repo = Path(__file__).resolve().parents[2]

    file_sha: dict[str, str] = {}
    for pattern in TRACKED_GLOBS:
        for path in sorted(repo.glob(pattern)):
            if path.is_file():
                rel = path.relative_to(repo).as_posix()
                file_sha[rel] = hashlib.sha256(path.read_bytes()).hexdigest()

    # Cleanliness is judged over the files a probe's result actually depends on,
    # not the whole tree.  Ticket markdown and notes are routinely edited in
    # parallel sessions and have no bearing on reproducibility; refusing on them
    # would either block the probe or tempt someone to commit another session's
    # half-written work.  A dirty *source* file still refuses, which is the case
    # SP-1 cares about.
    relevant = _git(repo, "status", "--porcelain", "--", *sorted(file_sha))
    if relevant:
        print("REFUSING: probe-relevant sources are dirty. Commit first.", file=sys.stderr)
        for line in relevant.splitlines()[:20]:
            print(f"  {line}", file=sys.stderr)
        return 1

    whole_tree = _git(repo, "status", "--porcelain")
    other_dirty = [ln for ln in whole_tree.splitlines() if ln.strip()]
    if other_dirty:
        print(f"note: {len(other_dirty)} file(s) dirty outside the probe's dependency set:")
        for line in other_dirty[:10]:
            print(f"  {line}")

    stamp = {
        "head": _git(repo, "rev-parse", "HEAD"),
        "describe": _git(repo, "describe", "--tags", "--always", "--dirty"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "tree_clean": True,  # over the probe's dependency set; see above
        "dirty_outside_dependency_set": other_dirty[:20],
        "file_sha256": file_sha,
    }
    out = repo / ".provenance.json"
    out.write_text(json.dumps(stamp, indent=2, sort_keys=True))
    print(f"wrote {out}")
    print(f"  head     {stamp['head']}")
    print(f"  describe {stamp['describe']}")
    print(f"  files    {len(file_sha)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
