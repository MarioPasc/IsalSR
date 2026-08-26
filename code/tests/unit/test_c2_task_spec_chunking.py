"""Unit tests for the C2 array-chunking decode (SCBI job-grouping request).

Picasso's administrators asked us to group short jobs so that every submitted
SLURM task runs for at least two hours. One array task therefore runs a
*contiguous block* of ``(problem, seed)`` cells instead of a single cell.

The properties tested here are the ones whose failure is silent:

* the blocks **partition** the cell set -- a gap loses a cell from the paired
  design, an overlap runs it twice and the second run overwrites the first;
* the partition is a **pure function** of ``(n_cells, n_tasks, index)``, because
  the sweep pass re-derives it and must agree cell for cell;
* blocks are **even**, because a ragged short remainder reintroduces exactly the
  sub-two-hour tasks the grouping exists to remove;
* ``--bundle 1`` is **identical** to the pre-chunking decode, so the certified
  single-cell path is still reachable;
* the problem slug matches ``io_utils`` exactly, because the worker uses it to
  move a cell between local scratch and FSCRATCH and a wrong slug copies
  nothing rather than failing.
"""

from __future__ import annotations

import itertools
import tempfile
from pathlib import Path

import pytest

from experiments.models.io_utils import ensure_output_structure, seed_dir
from experiments.scripts.c2_task_spec import (
    TaskSpecError,
    chunk_bounds,
    decode_chunk,
    decode_index,
    n_tasks_for,
    problem_slug,
)

PROBLEMS = [f"P-{i}" for i in range(7)]
SEEDS = list(range(1, 31))


# ---------------------------------------------------------------------------
# n_tasks_for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_cells", "bundle", "expected"),
    [(360, 1, 360), (360, 2, 180), (360, 164, 3), (300, 27, 12), (7, 3, 3), (5, 9, 1)],
)
def test_n_tasks_for_is_ceiling_division(n_cells: int, bundle: int, expected: int) -> None:
    assert n_tasks_for(n_cells, bundle) == expected


def test_n_tasks_never_exceeds_cells() -> None:
    """A task with no cells would sit in the queue doing nothing and still count."""
    for n_cells in range(1, 60):
        for bundle in range(1, 80):
            assert 1 <= n_tasks_for(n_cells, bundle) <= n_cells


@pytest.mark.parametrize(("n_cells", "bundle"), [(0, 4), (-1, 4), (10, 0), (10, -3)])
def test_n_tasks_for_rejects_non_positive(n_cells: int, bundle: int) -> None:
    with pytest.raises(TaskSpecError):
        n_tasks_for(n_cells, bundle)


# ---------------------------------------------------------------------------
# chunk_bounds -- the partition properties
# ---------------------------------------------------------------------------


def test_chunks_partition_the_cell_set_exactly() -> None:
    """Every cell belongs to exactly one task, for every shape the plan can emit."""
    for n_cells in range(1, 130):
        for bundle in range(1, 40):
            n_tasks = n_tasks_for(n_cells, bundle)
            covered: list[int] = []
            for index in range(1, n_tasks + 1):
                lo, hi = chunk_bounds(n_cells, n_tasks, index)
                covered.extend(range(lo, hi))
            assert covered == list(range(n_cells)), (n_cells, bundle)


def test_chunk_sizes_differ_by_at_most_one() -> None:
    """Evenness is what keeps the LAST task of an array above the 2 h floor."""
    for n_cells in range(1, 130):
        for bundle in range(1, 40):
            n_tasks = n_tasks_for(n_cells, bundle)
            sizes = [
                hi - lo
                for lo, hi in (chunk_bounds(n_cells, n_tasks, i) for i in range(1, n_tasks + 1))
            ]
            assert max(sizes) - min(sizes) <= 1, (n_cells, bundle, sizes)


def test_no_chunk_exceeds_the_requested_bundle() -> None:
    """Evening out must never round UP past the size the wall was sized for."""
    for n_cells in range(1, 130):
        for bundle in range(1, 40):
            n_tasks = n_tasks_for(n_cells, bundle)
            sizes = [
                hi - lo
                for lo, hi in (chunk_bounds(n_cells, n_tasks, i) for i in range(1, n_tasks + 1))
            ]
            assert max(sizes) <= bundle, (n_cells, bundle, sizes)


def test_chunk_bounds_is_deterministic() -> None:
    """The sweep pass re-derives the partition; it must agree cell for cell."""
    first = [chunk_bounds(1000, 37, i) for i in range(1, 38)]
    second = [chunk_bounds(1000, 37, i) for i in range(1, 38)]
    assert first == second


@pytest.mark.parametrize("index", [0, -1, 38])
def test_chunk_bounds_rejects_out_of_range_index(index: int) -> None:
    with pytest.raises(TaskSpecError):
        chunk_bounds(1000, 37, index)


def test_chunk_bounds_rejects_more_tasks_than_cells() -> None:
    with pytest.raises(TaskSpecError):
        chunk_bounds(5, 9, 1)


# ---------------------------------------------------------------------------
# decode_chunk
# ---------------------------------------------------------------------------


def test_bundle_one_reproduces_the_legacy_decode() -> None:
    """The certified single-cell path must remain reachable byte for byte."""
    total = len(PROBLEMS) * len(SEEDS)
    for index in range(1, total + 1):
        assert decode_chunk(PROBLEMS, SEEDS, 1, index) == [decode_index(PROBLEMS, SEEDS, index)]


def test_decode_chunk_covers_every_cell_once() -> None:
    n_cells = len(PROBLEMS) * len(SEEDS)
    for bundle in (2, 3, 7, 27, 164):
        n_tasks = n_tasks_for(n_cells, bundle)
        seen = list(
            itertools.chain.from_iterable(
                decode_chunk(PROBLEMS, SEEDS, bundle, i) for i in range(1, n_tasks + 1)
            )
        )
        assert len(seen) == n_cells
        assert len(set(seen)) == n_cells
        assert set(seen) == {(p, s) for p in PROBLEMS for s in SEEDS}


def test_decode_chunk_keeps_the_seed_varying_fastest() -> None:
    """Order is part of the contract: it is what makes the partition contiguous."""
    cells = decode_chunk(PROBLEMS, SEEDS, 5, 1)
    assert cells == [("P-0", 1), ("P-0", 2), ("P-0", 3), ("P-0", 4), ("P-0", 5)]


def test_decode_chunk_rejects_empty_inputs() -> None:
    with pytest.raises(TaskSpecError):
        decode_chunk([], SEEDS, 4, 1)
    with pytest.raises(TaskSpecError):
        decode_chunk(PROBLEMS, [], 4, 1)


# ---------------------------------------------------------------------------
# problem_slug -- pinned against the code that creates the directory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "problem",
    ["Nguyen-1", "I.15.10", "III.14.14", "Vlad-7", "Keijzer-11", "II.11.27", "Korns-12"],
)
def test_problem_slug_matches_the_directory_io_utils_creates(problem: str) -> None:
    """A wrong slug does not raise -- it copies nothing and the cell is lost."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        paths = ensure_output_structure(base, "bingo", "hard", problem, variants=["hash"])
        actual = seed_dir(paths["hash"], 7).relative_to(base).as_posix()
        assert actual == f"bingo/hard/{problem_slug(problem)}/hash/seed_07"
