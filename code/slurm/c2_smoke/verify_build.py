"""SP-2 verification: the installed extension is the code we just built.

Called by ``deploy.sh`` on the remote host **with every module unloaded**, so it
doubles as the "does it import on a bare compute node?" check.

Lives in a file rather than a heredoc on purpose: nesting a Python heredoc
inside a single-quoted ``ssh '...'`` command strips the Python string quotes and
the whole block dies with a ``SyntaxError`` *after* pip has already reported
success — which looks exactly like a build failure and is not one.

Exits non-zero if the artefact is stale, is not in site-packages, or the engine
does not come up native.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys


def main() -> int:
    """Print the build provenance and assert SP-2."""
    from isalsr.core import _native, backends

    so = _native.__file__
    mtime = os.path.getmtime(so)
    print("  so:       ", so)
    print("  so_mtime: ", datetime.datetime.fromtimestamp(mtime))
    print("  engine:   ", backends.engine())
    print("  build:    ", dict(backends.build_info()))

    problems: list[str] = []

    # An editable install puts the artefact in site-packages, not the repo tree,
    # so a repo-local `find` will never reveal a stale one.
    if "site-packages" not in so:
        problems.append("SP-2: .so is not under site-packages")

    last = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", "src/isalsr/core/native"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if last and mtime <= float(last):
        problems.append(f"SP-2: .so mtime {mtime} is not newer than the last native commit {last}")

    if backends.engine() != "cpp":
        problems.append(f"SP-3: engine is {backends.engine()!r}, expected 'cpp'")

    for p in problems:
        print(f"[FATAL] {p}", file=sys.stderr)
    if problems:
        return 1
    print("  SP-2 OK: artefact post-dates the last native commit, imports with modules purged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
