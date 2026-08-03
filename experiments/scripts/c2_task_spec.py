"""Decode a SLURM array index into the ``(problem, seed)`` cell it must run.

Campaign C2 submits one array task per ``(problem, seed)`` of a single-suite
config. The decode used to live in the bash worker, which carried its own
hard-coded benchmark-to-module table and was missing two of the seven registered
suites. This helper resolves the problem list through the orchestrator's own
registry instead, so the shell never needs to know which suites exist.

The ordering is deterministic and stable: problems in registry order (after the
config's suite is expanded), seeds in the order given on the command line, with
the seed varying fastest.
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="1-based array task index to decode")
    group.add_argument(
        "--count",
        action="store_true",
        help="Print the total task count (n_problems * n_seeds) instead of decoding",
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
        if args.count:
            payload = str(len(problems) * len(seeds))
        else:
            problem, seed = decode_index(problems, seeds, args.index)
            payload = f"{problem} {seed}"
    except (TaskSpecError, ValueError, OSError) as exc:
        # stderr only: the caller substitutes stdout into a command line, so a
        # partial or empty token must never be printed on the failure path.
        print(f"c2_task_spec: {exc}", file=sys.stderr)
        return 1

    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
