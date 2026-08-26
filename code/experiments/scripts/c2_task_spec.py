"""Decode a SLURM array index into the ``(problem, seed)`` cells it must run.

Campaign C2 submits one array task per *chunk* of ``(problem, seed)`` cells of a
single-suite config. The decode used to live in the bash worker, which carried
its own hard-coded benchmark-to-module table and was missing two of the seven
registered suites. This helper resolves the problem list through the
orchestrator's own registry instead, so the shell never needs to know which
suites exist.

The ordering is deterministic and stable: problems in registry order (after the
config's suite is expanded), seeds in the order given on the command line, with
the seed varying fastest.

Chunking (added 2026-08-07, SCBI request)
-----------------------------------------
One array task used to be one cell. Picasso's administrators asked us to group
short jobs so that **every submitted task runs for at least two hours**, because
the scheduler spends more time placing a job than a short job spends running,
and 12,600 such placements saturate the queue for every user of the cluster.

``decode_chunk`` therefore maps a 1-based array index onto a *contiguous block*
of cells. The partition is **even** — block sizes differ by at most one — so no
task gets a ragged short remainder, which would reintroduce exactly the
sub-two-hour tasks the grouping exists to remove.

It is also a **pure function of** ``(n_cells, n_tasks, index)`` and of nothing
else. That is load-bearing: a sweep pass over the cells a first pass could not
finish re-derives the same partition from the same inputs, so a cell can never
be silently reassigned between passes.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.models.orchestrator import get_benchmarks, parse_seeds  # noqa: E402


class TaskSpecError(Exception):
    """Raised when a config or an array index cannot be decoded."""


def load_problem_names(config_path: str, problem_filter: str | None = None) -> list[str]:
    """Return the problem names of the config's single benchmark suite.

    Args:
        config_path: Path to the experiment YAML.
        problem_filter: Optional comma-separated problem filter, passed through
            to the orchestrator's ``get_benchmarks``.

    Returns:
        Problem names in registry order.

    Raises:
        TaskSpecError: If the config declares zero or more than one suite. All
            seven C2 configs declare exactly one, and a multi-suite config would
            make the array index ambiguous.
    """
    with open(config_path) as handle:
        config: dict[str, Any] = yaml.safe_load(handle)

    suites = list((config.get("benchmarks") or {}).keys())
    if len(suites) != 1:
        raise TaskSpecError(
            f"{config_path}: expected exactly one benchmark suite, found {len(suites)}: {suites}"
        )

    return [bench["name"] for bench in get_benchmarks(suites[0], problem_filter)]


def decode_index(problems: list[str], seeds: list[int], index: int) -> tuple[str, int]:
    """Map a 1-based array index onto a ``(problem, seed)`` pair.

    Args:
        problems: Problem names, in the order the array was sized against.
        seeds: Seeds, in the order given on the command line.
        index: 1-based SLURM array task index.

    Returns:
        The problem name and seed for that task.

    Raises:
        TaskSpecError: If ``index`` is outside ``[1, len(problems) * len(seeds)]``.
    """
    total = len(problems) * len(seeds)
    if not problems or not seeds:
        raise TaskSpecError("empty problem or seed list")
    if index < 1 or index > total:
        raise TaskSpecError(f"index {index} out of range [1, {total}]")

    zero_based = index - 1
    return problems[zero_based // len(seeds)], seeds[zero_based % len(seeds)]


def problem_slug(problem: str) -> str:
    """Return the on-disk directory name the orchestrator gives a problem.

    The rule mirrors ``experiments.models.io_utils.ensure_output_structure``,
    which is the code that actually creates the directory. The worker needs it to
    stage a single cell between local scratch and FSCRATCH, and a bash
    reimplementation would be a silent-divergence trap -- a wrong slug does not
    fail, it copies nothing, and the cell is then lost with the node's scratch.
    ``tests/unit/test_c2_task_spec_chunking.py`` pins the two together.

    Args:
        problem: Problem name as it appears in the benchmark registry.

    Returns:
        The directory name, e.g. ``"nguyen_1"`` for ``"Nguyen-1"``.
    """
    return problem.lower().replace("-", "_")


def n_tasks_for(n_cells: int, bundle: int) -> int:
    """Return how many array tasks a chunked array carries.

    Args:
        n_cells: Total ``(problem, seed)`` cells in the array.
        bundle: Requested cells per task.

    Returns:
        ``ceil(n_cells / bundle)``, at least one.

    Raises:
        TaskSpecError: If either argument is not positive.
    """
    if n_cells < 1:
        raise TaskSpecError(f"n_cells must be positive, got {n_cells}")
    if bundle < 1:
        raise TaskSpecError(f"bundle must be positive, got {bundle}")
    return max(1, -(-n_cells // bundle))


def chunk_bounds(n_cells: int, n_tasks: int, index: int) -> tuple[int, int]:
    """Return the half-open cell range ``[lo, hi)`` owned by a 1-based task index.

    The partition is even: with ``q, r = divmod(n_cells, n_tasks)`` the first
    ``r`` tasks take ``q + 1`` cells and the rest take ``q``. Splitting evenly
    rather than taking fixed-size blocks with a short tail is what keeps the
    *last* task of every array above the two-hour floor.

    Args:
        n_cells: Total cells in the array.
        n_tasks: Number of array tasks the cells are spread over.
        index: 1-based SLURM array task index.

    Returns:
        The ``(lo, hi)`` half-open range of 0-based cell offsets.

    Raises:
        TaskSpecError: If the index is outside ``[1, n_tasks]`` or the counts are
            not positive.
    """
    if n_cells < 1 or n_tasks < 1:
        raise TaskSpecError(f"n_cells={n_cells} and n_tasks={n_tasks} must both be positive")
    if n_tasks > n_cells:
        raise TaskSpecError(f"n_tasks {n_tasks} exceeds n_cells {n_cells}")
    if index < 1 or index > n_tasks:
        raise TaskSpecError(f"index {index} out of range [1, {n_tasks}]")

    quotient, remainder = divmod(n_cells, n_tasks)
    zero_based = index - 1
    lo = zero_based * quotient + min(zero_based, remainder)
    hi = lo + quotient + (1 if zero_based < remainder else 0)
    return lo, hi


def decode_chunk(
    problems: list[str], seeds: list[int], bundle: int, index: int
) -> list[tuple[str, int]]:
    """Map a 1-based array index onto the block of cells that task must run.

    Args:
        problems: Problem names, in the order the array was sized against.
        seeds: Seeds, in the order given on the command line.
        bundle: Requested cells per task; the realised block size may differ by
            one because the partition is evened out.
        index: 1-based SLURM array task index.

    Returns:
        The ``(problem, seed)`` pairs for that task, in decode order.

    Raises:
        TaskSpecError: If the lists are empty or the index is out of range.
    """
    if not problems or not seeds:
        raise TaskSpecError("empty problem or seed list")

    n_cells = len(problems) * len(seeds)
    n_tasks = n_tasks_for(n_cells, bundle)
    lo, hi = chunk_bounds(n_cells, n_tasks, index)
    return [decode_index(problems, seeds, offset + 1) for offset in range(lo, hi)]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Decode a SLURM array index into (problem, seed) for a C2 config",
    )
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config")
    parser.add_argument(
        "--seeds",
        required=True,
        help="Seed specification, e.g. '0,101,102' or '1-30'. Order is preserved.",
    )
    parser.add_argument(
        "--problems",
        default=None,
        help="Optional problem filter, e.g. 'Nguyen-1,Nguyen-2' or 'all'.",
    )
    parser.add_argument(
        "--bundle",
        type=int,
        default=1,
        help="Cells per array task (SCBI job-grouping request, 2026-08-07). With "
        "--index, print every cell of that task, one 'problem seed' pair per "
        "line. With --count, print the number of TASKS rather than of cells.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="1-based array task index to decode")
    group.add_argument(
        "--count",
        action="store_true",
        help="Print the array task count instead of decoding (n_problems * "
        "n_seeds cells, divided into chunks of --bundle)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print either the task count or the decoded ``<problem> <seed>`` pair.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` on success, ``1`` if the config or index could not be decoded.
    """
    args = build_parser().parse_args(argv)
    try:
        problems = load_problem_names(args.config, args.problems)
        seeds = parse_seeds(args.seeds)
        if args.bundle < 1:
            raise TaskSpecError(f"--bundle must be positive, got {args.bundle}")
        if args.count:
            payload = str(n_tasks_for(len(problems) * len(seeds), args.bundle))
        else:
            cells = decode_chunk(problems, seeds, args.bundle, args.index)
            payload = "\n".join(
                f"{problem} {seed} {problem_slug(problem)}" for problem, seed in cells
            )
    except (TaskSpecError, ValueError, OSError) as exc:
        # stderr only: the caller substitutes stdout into a command line, so a
        # partial or empty token must never be printed on the failure path.
        print(f"c2_task_spec: {exc}", file=sys.stderr)
        return 1

    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
