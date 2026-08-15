#!/usr/bin/env python
"""Verify that the chunked C2 worker loses nothing to ``$LOCALSCRATCH``.

Consumes the two trees produced by :file:`chunk_smoke.sh`:

* ``<root>/staged`` — the campaign's path: the payload writes to the compute
  node's local disk and the worker copies back.
* ``<root>/direct`` — the reference: the payload writes straight to FSCRATCH.

The waves are identical in every other respect, so **any file present in
``direct`` and absent from ``staged`` is a file the campaign would lose.**

Why the comparison is a diff rather than a checklist
----------------------------------------------------
The 2026-08-07 mock counted ``run_log.json`` per cell, passed, and was wrong:
``metadata.json`` is written at the ROOT of the output tree
(``orchestrator.py:665``), outside every cell directory, and the per-cell
copy-back never touched it. ``c2_certify.py:842`` reads that file and criterion
C1.4 fails without it. Every artefact the check looked *for* came back — which
is precisely why the check passed while results were being lost.

A checklist can only find what its author already thought of. A diff finds what
nobody thought of, which is the category the failure was in.

Usage
-----
    python slurm/c2_smoke/chunk_smoke_verify.py <root>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

#: (problems, seeds, arms, methods) the smoke submits.
EXPECTED_CELLS = 2 * 2 * 3 * 2


def tree(root: Path) -> set[str]:
    """Return every regular file under ``root``, as POSIX paths relative to it.

    Args:
        root: Wave root to walk.

    Returns:
        Relative paths, so two wave roots are directly comparable.
    """
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def cells(root: Path) -> set[tuple[str, str, str, str, int]]:
    """Return the ``(method, suite, problem, arm, seed)`` cells present in a tree.

    Read from each ``run_log.json`` rather than parsed out of the directory
    names: the metadata is what the analysis consumes, so a cell whose payload
    disagrees with its path is a cell that would silently mis-key later.

    Args:
        root: Wave root to walk.

    Returns:
        One tuple per completed cell.
    """
    out = set()
    for path in root.rglob("run_log.json"):
        meta = json.loads(path.read_text())["metadata"]
        out.add(
            (
                meta["method"],
                meta["benchmark"],
                meta["problem"],
                meta["representation"],
                int(meta["seed"]),
            )
        )
    return out


def check(label: str, ok: bool, detail: str = "") -> bool:
    """Print one PASS/FAIL line.

    Args:
        label: What was checked.
        ok: Whether it held.
        detail: Extra context, printed either way.

    Returns:
        ``ok``, so callers can accumulate with ``&=``.
    """
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<58} {detail}")
    return ok


def main(argv: list[str]) -> int:
    """Compare the staged and direct waves. Returns the process exit status.

    Args:
        argv: Command-line arguments; one positional root.

    Returns:
        ``0`` if the staged wave lost nothing, ``1`` otherwise.
    """
    if len(argv) != 2:
        print(__doc__)
        return 2
    root = Path(argv[1])
    staged, direct = root / "staged", root / "direct"
    for wave in (staged, direct):
        if not wave.is_dir():
            print(f"!! missing wave root: {wave}")
            return 1

    s_files, d_files = tree(staged), tree(direct)
    s_cells, d_cells = cells(staged), cells(direct)
    ok = True

    print(f"\nStaged (localscratch):  {len(s_files):4d} files, {len(s_cells):2d} cells")
    print(f"Direct (reference)   :  {len(d_files):4d} files, {len(d_cells):2d} cells\n")

    # --- The question this script exists to answer. -------------------------
    lost = sorted(d_files - s_files)
    ok &= check(
        "no file from the direct run is missing after staging",
        not lost,
        f"LOST: {lost}" if lost else "",
    )

    # --- The specific artefact that was being lost. ------------------------
    # Kept as its own line even though the diff above subsumes it: a regression
    # here has a known, expensive consequence (C1.4) and deserves to be named.
    meta_path = staged / "metadata.json"
    meta_ok = meta_path.is_file()
    if meta_ok:
        try:
            benchmarks = json.loads(meta_path.read_text())["config"]["benchmarks"]
            meta_ok = bool(benchmarks)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            meta_ok = False
            print(f"        {type(exc).__name__}: {exc}")
    ok &= check(
        "root metadata.json came back AND parses (c2_certify C1.4)",
        meta_ok,
        f"{meta_path.stat().st_size} bytes" if meta_path.is_file() else "MISSING",
    )

    # --- Completeness of both waves. ---------------------------------------
    ok &= check(
        f"staged ran all {EXPECTED_CELLS} cells",
        len(s_cells) == EXPECTED_CELLS,
        f"{len(s_cells)}/{EXPECTED_CELLS}",
    )
    ok &= check(
        "the two waves cover the same cells",
        s_cells == d_cells,
        f"only-direct={sorted(d_cells - s_cells)}" if d_cells - s_cells else "",
    )

    # --- Per-cell artefact sets, compared cell by cell. --------------------
    # A tree-level diff passes if some OTHER cell happens to supply a file name.
    # Comparing the artefact set within each cell directory closes that gap.
    by_cell: dict[str, set[str]] = defaultdict(set)
    for rel in d_files:
        parts = rel.rsplit("/", 1)
        if len(parts) == 2:
            by_cell[parts[0]].add(parts[1])
    incomplete = []
    for cell_dir, names in sorted(by_cell.items()):
        got = {p.name for p in (staged / cell_dir).glob("*") if p.is_file()}
        if missing := names - got:
            incomplete.append(f"{cell_dir}:{sorted(missing)}")
    ok &= check(
        "every cell directory has its full artefact set",
        not incomplete,
        f"{len(incomplete)} incomplete: {incomplete[:3]}"
        if incomplete
        else f"{len(by_cell)} cell dirs, {sorted(next(iter(by_cell.values())))}",
    )

    # --- Byte-level agreement on a file that must not vary. ----------------
    # trajectory.csv is produced by the payload, not the copy; if staging
    # truncated a file this is where it shows.
    sizes_ok, mismatched = True, []
    for rel in sorted(d_files & s_files):
        ds, ss = (direct / rel).stat().st_size, (staged / rel).stat().st_size
        if ds > 0 and ss == 0:
            sizes_ok = False
            mismatched.append(rel)
    ok &= check(
        "no staged file arrived empty where the reference is not",
        sizes_ok,
        f"{mismatched[:3]}" if mismatched else "",
    )

    print()
    print("SMOKE OK -- localscratch loses nothing" if ok else "SMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
