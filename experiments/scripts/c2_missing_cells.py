"""Enumerate the C2 cells a results root does not yet contain.

Why this exists
---------------
A campaign task that hits its start-deadline **defers** its remaining cells and
exits ``COMPLETED`` (``slurm/c2_smoke/worker.sh:450``). Nothing in ``sacct``
reveals a deferred cell, so SLURM state cannot answer "what is still missing".
The 2026-08-09 audit found ~512 such cells while every job record read healthy.

The only sound way to answer the question is to compare the **expected universe**
against what is on disk:

* the universe is ``methods x arms x suites x problems x seeds``, with the
  problem list resolved through ``c2_task_spec.load_problem_names`` — the *same*
  call ``decode_chunk`` makes, so the two agree by construction rather than by
  convention;
* presence is ``<root>/<method>/<suite>/<slug>/<arm>/seed_NN/run_log.json``, the
  layout ``experiments/models/io_utils.py`` creates and ``worker.sh``'s
  ``cell_relpath`` stages.

This is a **read-only** tool. It is safe to run against a live campaign root, and
the counts it reports are a lower bound while tasks are still writing.

Presence versus validity
------------------------
By default a cell counts as present when its ``run_log.json`` exists. ``--strict``
additionally parses it and requires the fields the orchestrator's own resume
check looks at, which is what makes an OOM-truncated file count as *missing*
rather than as done. ``--strict`` is the mode to use before declaring the
campaign complete; the default is the mode to use while it runs, because it does
not open 12,600 files on GPFS.

Usage
-----
``--summary`` (default) prints one row per ``(method, arm, suite)``.
``--list`` prints ``method arm suite problem seed`` for every missing cell.
``--selectors`` prints the comma-separated ``method:arm:suite`` list that
``slurm/c2_campaign/submit_recovery.sh --only`` consumes, which is how a recovery
pass is scoped to exactly the arrays that need it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.models.orchestrator import parse_seeds  # noqa: E402
from experiments.scripts.c2_task_spec import (  # noqa: E402
    TaskSpecError,
    load_problem_names,
    problem_slug,
)

#: The campaign universe's three axes.
#:
#: 🔴 Duplicated from ``c2_slot_plan`` rather than imported, and that is
#: deliberate. ``experiments`` is a **namespace package**, and the conda
#: environment carries an editable install whose meta-path finder contributes the
#: DEPLOYED checkout to ``experiments.__path__`` *ahead* of anything
#: ``PYTHONPATH`` adds (measured on Picasso, 2026-08-09). So a module that exists
#: in both trees always resolves to the deployed one, and importing a symbol that
#: only the newer checkout defines raises ``ImportError`` at run time. This tool
#: must run from a recovery checkout against a campaign whose deployed tree is
#: frozen at its tag, so it carries no cross-module dependency that the deployed
#: tree could shadow. ``tests/unit/test_c2_recovery_plan.py`` pins these to
#: ``c2_slot_plan``'s definitions so the copies cannot drift.
METHODS: tuple[str, ...] = ("udfs", "bingo")
ARMS: tuple[str, ...] = ("baseline", "hash", "isalsr")
SUITES: tuple[str, ...] = (
    "nguyen",
    "feynman",
    "hard",
    "cherrypicked",
    "roundoff",
    "feynman_remainder",
    "strogatz",
)


def match_selector(key: str, patterns: list[str]) -> bool:
    """Return whether ``method:arm:suite`` matches any ``*``-globbed pattern.

    Args:
        key: A ``method:arm:suite`` array identifier.
        patterns: Selectors such as ``udfs:*:feynman``.

    Returns:
        ``True`` if any pattern matches every field.
    """
    fields = key.split(":")
    for pattern in patterns:
        parts = pattern.split(":")
        if len(parts) == len(fields) and all(
            p == "*" or p == f for p, f in zip(parts, fields, strict=True)
        ):
            return True
    return False


#: Fields a ``run_log.json`` must carry before ``--strict`` calls the cell done.
#:
#: Deliberately short. The certifier (``c2_certify.py``) owns the full 40-field
#: specification; duplicating it here would make two definitions of "valid" that
#: can drift. What this needs is only the property the *resume* logic needs: the
#: file parses and names the run it claims to be.
#:
#: 🔴 These live under ``metadata``, NOT at the top level, and the arm field is
#: ``representation``, not ``variant`` (measured against the C2 tree, 2026-08-12).
#: The original spelling — top-level ``("method", "variant", "problem", "seed")``
#: — matched NOTHING, so ``--strict`` reported every one of the 11,999 written
#: cells as missing and the census returned 0 present / 12,600 missing. That is
#: worse than a crash: the plan's workflow feeds this census straight into
#: ``submit_recovery.sh --only``, so following the documented procedure would
#: have scoped the recovery pass to the ENTIRE campaign. The resume logic would
#: still have skipped completed cells, so no data was ever at risk, but the plan
#: and every count derived from it would have been nonsense.
#:
#: A ``--strict`` predicate that can never be satisfied is the inverse of the
#: SP-6 pattern this project keeps hitting: instead of silently passing, it
#: silently condemns. Both come from asserting a shape nobody measured.
STRICT_REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "method",
    "representation",
    "problem",
    "seed",
)

#: Backwards-compatible alias. Retained so existing imports keep resolving.
STRICT_REQUIRED_FIELDS: tuple[str, ...] = STRICT_REQUIRED_METADATA_FIELDS


class MissingCellsError(Exception):
    """Raised when the expected universe cannot be built."""


@dataclass(frozen=True)
class Cell:
    """One ``(method, arm, suite, problem, seed)`` unit of campaign work."""

    method: str
    arm: str
    suite: str
    problem: str
    seed: int

    @property
    def array_key(self) -> str:
        """Return the ``method:arm:suite`` array this cell belongs to."""
        return f"{self.method}:{self.arm}:{self.suite}"

    def relpath(self) -> str:
        """Return the cell's directory relative to the results root.

        Mirrors ``worker.sh``'s ``cell_relpath``, which is itself
        ``io_utils.ensure_output_structure``.

        Returns:
            ``<method>/<suite>/<slug>/<arm>/seed_NN``.
        """
        return os.path.join(
            self.method, self.suite, problem_slug(self.problem), self.arm, f"seed_{self.seed:02d}"
        )


def expected_cells(
    config_dir: str,
    seeds: list[int],
    methods: tuple[str, ...] = METHODS,
    arms: tuple[str, ...] = ARMS,
    suites: tuple[str, ...] = SUITES,
) -> list[Cell]:
    """Build the full campaign universe from the configs on disk.

    Args:
        config_dir: Directory holding ``{method}_{suite}.yaml``.
        seeds: Campaign seeds.
        methods: Methods to enumerate.
        arms: Arms to enumerate.
        suites: Suites to enumerate.

    Returns:
        Every expected cell, in ``(method, arm, suite, problem, seed)`` order.

    Raises:
        MissingCellsError: If a config is missing or declares no problems.
    """
    if not seeds:
        raise MissingCellsError("need at least one seed")

    problems: dict[tuple[str, str], list[str]] = {}
    for method in methods:
        for suite in suites:
            path = os.path.join(config_dir, f"{method}_{suite}.yaml")
            if not os.path.isfile(path):
                raise MissingCellsError(f"missing config: {path}")
            try:
                names = load_problem_names(path)
            except TaskSpecError as exc:
                raise MissingCellsError(str(exc)) from exc
            if not names:
                raise MissingCellsError(f"{path}: no problems in suite {suite}")
            problems[(method, suite)] = names

    return [
        Cell(method, arm, suite, problem, seed)
        for method in methods
        for arm in arms
        for suite in suites
        for problem in problems[(method, suite)]
        for seed in seeds
    ]


def _run_log_is_valid(path: str) -> bool:
    """Return whether a ``run_log.json`` parses and names its run.

    Args:
        path: Absolute path to the file.

    Returns:
        ``True`` when the JSON loads to a mapping whose ``metadata`` block
        carries every field of :data:`STRICT_REQUIRED_METADATA_FIELDS`.
    """
    try:
        with open(path) as handle:
            blob: Any = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(blob, dict):
        return False
    metadata = blob.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return all(field in metadata for field in STRICT_REQUIRED_METADATA_FIELDS)


def missing_cells(root: str, cells: list[Cell], strict: bool = False) -> list[Cell]:
    """Return the subset of ``cells`` with no usable ``run_log.json`` under ``root``.

    Args:
        root: Results root.
        cells: The expected universe.
        strict: Also parse each present file and require
            :data:`STRICT_REQUIRED_FIELDS`.

    Returns:
        The missing cells, in the order given.
    """
    out: list[Cell] = []
    for cell in cells:
        path = os.path.join(root, cell.relpath(), "run_log.json")
        if strict:
            if not _run_log_is_valid(path):
                out.append(cell)
        elif not os.path.isfile(path):
            out.append(cell)
    return out


def summarise(cells: list[Cell], missing: list[Cell]) -> str:
    """Render a per-array expected/present/missing table.

    Args:
        cells: The expected universe.
        missing: The missing subset.

    Returns:
        A human-readable table with a total row.
    """
    order: list[str] = []
    total: dict[str, int] = {}
    gaps: dict[str, int] = {}
    for cell in cells:
        key = cell.array_key
        if key not in total:
            order.append(key)
            total[key] = 0
            gaps[key] = 0
        total[key] += 1
    for cell in missing:
        gaps[cell.array_key] += 1

    lines = [f"{'array':34s} {'expected':>9s} {'present':>8s} {'missing':>8s}"]
    for key in order:
        lines.append(f"{key:34s} {total[key]:9d} {total[key] - gaps[key]:8d} {gaps[key]:8d}")
    lines += [
        "",
        f"{'TOTAL':34s} {len(cells):9d} {len(cells) - len(missing):8d} {len(missing):8d}",
    ]
    return "\n".join(lines)


def selectors(missing: list[Cell]) -> str:
    """Return the ``--only`` string covering every array with a gap.

    Args:
        missing: The missing cells.

    Returns:
        A comma-separated ``method:arm:suite`` list, or the empty string when
        nothing is missing. Emitting nothing rather than a placeholder is
        deliberate: ``submit_recovery.sh`` refuses an empty ``--only``, so a
        complete tree cannot accidentally trigger a full 42-array resubmission.
    """
    seen: list[str] = []
    for cell in missing:
        if cell.array_key not in seen:
            seen.append(cell.array_key)
    return ",".join(seen)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured argument parser.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Enumerate the C2 cells a results root does not yet contain"
    )
    parser.add_argument("--results-dir", required=True, help="Campaign results root")
    parser.add_argument(
        "--config-dir",
        default=os.path.join(here, "..", "configs"),
        help="Directory holding {method}_{suite}.yaml",
    )
    parser.add_argument("--seeds", default="1-30", help="Seed spec, e.g. '1-30'")
    parser.add_argument(
        "--only",
        default=None,
        help="Restrict the universe to these method:arm:suite selectors ('*' "
        "wildcards a field). Without it the whole 12,600-cell universe is walked.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Parse each run_log.json and require its identity fields, so a "
        "truncated file counts as missing. Slower; use before declaring "
        "completeness.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--summary", action="store_true", help="Per-array expected/present/missing (default)"
    )
    group.add_argument(
        "--list", action="store_true", help="One 'method arm suite problem seed' row per gap"
    )
    group.add_argument(
        "--selectors",
        action="store_true",
        help="Comma-separated method:arm:suite list for submit_recovery.sh --only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print the requested view of the campaign's gaps.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` when nothing is missing, ``2`` when something is, ``1`` on error.
        The three-way exit is what lets a monitor loop distinguish "complete"
        from "incomplete" without parsing the output.
    """
    args = build_parser().parse_args(argv)
    try:
        cells = expected_cells(args.config_dir, parse_seeds(args.seeds))
        if args.only:
            patterns = [s.strip() for s in args.only.split(",") if s.strip()]
            for pattern in patterns:
                if len(pattern.split(":")) != 3:
                    raise MissingCellsError(f"selector {pattern!r} is not method:arm:suite")
            cells = [c for c in cells if match_selector(c.array_key, patterns)]
            if not cells:
                raise MissingCellsError(f"selectors {patterns} match no array")
        gaps = missing_cells(args.results_dir, cells, strict=args.strict)
    except (MissingCellsError, TaskSpecError, ValueError, OSError) as exc:
        print(f"c2_missing_cells: {exc}", file=sys.stderr)
        return 1

    if args.list:
        payload = "\n".join(f"{c.method}\t{c.arm}\t{c.suite}\t{c.problem}\t{c.seed}" for c in gaps)
    elif args.selectors:
        payload = selectors(gaps)
    else:
        payload = summarise(cells, gaps)

    if payload:
        print(payload)
    return 2 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
