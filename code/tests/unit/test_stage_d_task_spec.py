"""Tests for the Stage D 12-cell registry.

The registry is the single source of truth for Stage D: the SLURM worker
resolves its array index through it and the certifier enumerates expected cells
from it. A silent change here produces a complete, plausible, wrong
certification, so the locked configuration of ``audit.md`` §7 is asserted
literally rather than recomputed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.scripts.stage_d_task_spec import (
    ARMS,
    NAN_PROBLEMS,
    STAGE_D_CELLS,
    STAGE_D_CERTIFICATION_CELLS,
    STAGE_D_CONSTRAINT,
    STAGE_D_GROUPS,
    STAGE_D_MAX_TIME_S,
    STAGE_D_SEED,
    STAGE_D_SUITE,
    STAGE_D_TRACE_SEED,
    STAGE_D_WALL,
    TRACE_CELL,
    TRACE_PROBLEM,
    StageDSpecError,
    _emit_shell,
    cells_for_group,
    decode_index,
    group_mem_gb,
    main,
    trace_cell,
)


class TestLockedConfiguration:
    """The values audit.md §7 fixes. Changing one changes what Stage D means."""

    def test_thirteen_cells_twelve_of_them_certification(self) -> None:
        # Twelve certification cells (audit.md §7 row 3) plus the D2 trace cell
        # (§7.3), which repeats cell 10 under `shadow_hash: true` and is
        # excluded from certification by construction.
        assert len(STAGE_D_CELLS) == 13
        assert len(STAGE_D_CERTIFICATION_CELLS) == 12
        assert all(not c.trace for c in STAGE_D_CERTIFICATION_CELLS)

    def test_budget_and_wall(self) -> None:
        assert STAGE_D_MAX_TIME_S == 43_200
        assert STAGE_D_WALL == "0-16:00:00"
        assert STAGE_D_CONSTRAINT == "sr"

    def test_seed_101_on_every_certification_cell_102_on_the_trace_cell(self) -> None:
        assert STAGE_D_SEED == 101
        assert STAGE_D_TRACE_SEED == 102
        assert {c.seed for c in STAGE_D_CERTIFICATION_CELLS} == {101}
        assert trace_cell().seed == 102
        # Both lie outside the campaign seeds (1..20) and the 21..30 top-up.
        assert {c.seed for c in STAGE_D_CELLS}.isdisjoint(set(range(1, 31)))

    def test_group_sizes_and_memory(self) -> None:
        expected = {"udfs": (3, 16), "bingo_std": (6, 32), "bingo_isalsr": (4, 256)}
        for group, (n, mem) in expected.items():
            assert len(cells_for_group(group)) == n, group
            assert group_mem_gb(group) == mem, group

    def test_problem_coverage(self) -> None:
        """Pagie-1 on both hosts x 3 arms; the two NaN problems on Bingo only."""
        udfs = {(c.problem, c.arm) for c in STAGE_D_CELLS if c.method == "udfs"}
        assert udfs == {(TRACE_PROBLEM, a) for a in ARMS}

        bingo = {(c.problem, c.arm) for c in STAGE_D_CELLS if c.method == "bingo"}
        assert bingo == {(p, a) for p in (TRACE_PROBLEM, *NAN_PROBLEMS) for a in ARMS}

    def test_all_cells_share_one_suite(self) -> None:
        assert {c.suite for c in STAGE_D_CELLS} == {STAGE_D_SUITE}

    def test_indices_are_dense_and_unique(self) -> None:
        assert [c.index for c in STAGE_D_CELLS] == list(range(1, 14))

    def test_group_indices_are_dense_within_each_group(self) -> None:
        for group in STAGE_D_GROUPS:
            cells = cells_for_group(group)
            assert [c.group_index for c in cells] == list(range(1, len(cells) + 1))


class TestTraceCell:
    """D2 persists from exactly one cell; more than one would double-count."""

    def test_exactly_one_trace_cell(self) -> None:
        assert sum(1 for c in STAGE_D_CELLS if c.trace) == 1

    def test_trace_cell_is_bingo_pagie1_isalsr(self) -> None:
        cell = trace_cell()
        assert (cell.method, cell.problem, cell.arm) == TRACE_CELL
        assert (cell.method, cell.problem, cell.arm) == ("bingo", "Pagie-1", "isalsr")
        # Seed 102, not 101: the trace cell repeats cell 10's coordinates, so at
        # a shared seed the orchestrator would write both into one directory.
        assert cell.seed == 102

    def test_no_other_cell_carries_the_flag(self) -> None:
        for cell in STAGE_D_CELLS:
            if cell is not trace_cell():
                assert cell.trace is False, cell.label


class TestIndexDecoding:
    """The worker's only contract with the registry."""

    @pytest.mark.parametrize("group", STAGE_D_GROUPS)
    def test_every_valid_index_decodes(self, group: str) -> None:
        cells = cells_for_group(group)
        for i, expected in enumerate(cells, start=1):
            assert decode_index(group, i) == expected

    @pytest.mark.parametrize("group", STAGE_D_GROUPS)
    def test_index_zero_and_overflow_raise(self, group: str) -> None:
        n = len(cells_for_group(group))
        with pytest.raises(StageDSpecError, match="out of range"):
            decode_index(group, 0)
        with pytest.raises(StageDSpecError, match="out of range"):
            decode_index(group, n + 1)

    def test_unknown_group_raises(self) -> None:
        with pytest.raises(StageDSpecError, match="unknown group"):
            decode_index("bingo_isarlsr", 1)

    def test_decoding_covers_the_registry_exactly_once(self) -> None:
        seen = [
            decode_index(g, i)
            for g in STAGE_D_GROUPS
            for i in range(1, len(cells_for_group(g)) + 1)
        ]
        assert sorted(c.index for c in seen) == list(range(1, 14))


class TestShellEmission:
    """A key=value contract, so adding a field cannot silently shift a decode."""

    def test_emits_key_equals_quoted_value(self) -> None:
        text = _emit_shell(trace_cell())
        lines = text.splitlines()
        assert all(line.startswith("D_") and "='" in line for line in lines)

    def test_carries_every_field_the_worker_dereferences(self) -> None:
        text = _emit_shell(trace_cell())
        for key in (
            "D_METHOD",
            "D_ARM",
            "D_PROBLEM",
            "D_PROBLEM_SLUG",
            "D_SEED",
            "D_SUITE",
            "D_TRACE",
            "D_MEM_GB",
            "D_CONFIG_NAME",
            "D_MAX_TIME",
            "D_INDEX",
            "D_GROUP",
            "D_GROUP_INDEX",
        ):
            assert f"{key}='" in text, key

    def test_trace_flag_is_shell_truthy_only_on_the_trace_cell(self) -> None:
        assert "D_TRACE='1'" in _emit_shell(trace_cell())
        other = decode_index("bingo_std", 1)
        assert "D_TRACE='0'" in _emit_shell(other)

    def test_values_containing_punctuation_are_quoted(self) -> None:
        """Problem names carry '-' and '.'; unquoted they would split or glob."""
        text = _emit_shell(decode_index("bingo_isalsr", 3))
        assert "D_PROBLEM='Vladislavleva-2'" in text


class TestPathConventions:
    """Must match experiments.models.io_utils, the producer of the layout."""

    def test_problem_slug_matches_io_utils(self) -> None:
        for cell in STAGE_D_CELLS:
            assert cell.problem_slug == cell.problem.lower().replace("-", "_")

    def test_expected_slugs(self) -> None:
        slugs = {c.problem_slug for c in STAGE_D_CELLS}
        assert slugs == {"pagie_1", "korns_12", "vladislavleva_2"}

    def test_run_dir_layout(self) -> None:
        cell = trace_cell()
        assert cell.run_dir(Path("/r")) == Path("/r/bingo/hard/pagie_1/isalsr/seed_102")
        clean = next(
            c
            for c in STAGE_D_CERTIFICATION_CELLS
            if (c.method, c.problem, c.arm) == ("bingo", TRACE_PROBLEM, "isalsr")
        )
        assert clean.run_dir(Path("/r")) == Path("/r/bingo/hard/pagie_1/isalsr/seed_101")
        # The whole point: the traced run cannot overwrite the certified one.
        assert clean.run_dir(Path("/r")) != cell.run_dir(Path("/r"))

    def test_config_name_matches_launcher_convention(self) -> None:
        for cell in STAGE_D_CERTIFICATION_CELLS:
            assert cell.config_name == f"{cell.method}_{cell.suite}.yaml"

    def test_trace_cell_takes_the_trace_config(self) -> None:
        assert trace_cell().config_name == "bingo_hard_trace.yaml"

    def test_configs_exist_in_the_repo(self) -> None:
        root = Path(__file__).resolve().parents[2]
        for name in {c.config_name for c in STAGE_D_CELLS}:
            assert (root / "experiments" / "configs" / name).is_file(), name


class TestCli:
    """The worker and launcher shell out to this; exit codes are load-bearing."""

    def test_count_per_group(self, capsys: pytest.CaptureFixture[str]) -> None:
        for group, n in (("udfs", 3), ("bingo_std", 6), ("bingo_isalsr", 4)):
            assert main(["--group", group, "--count"]) == 0
            assert capsys.readouterr().out.strip() == str(n)

    def test_mem_gb_per_group(self, capsys: pytest.CaptureFixture[str]) -> None:
        for group, mem in (("udfs", 16), ("bingo_std", 32), ("bingo_isalsr", 256)):
            assert main(["--group", group, "--mem-gb"]) == 0
            assert capsys.readouterr().out.strip() == str(mem)

    def test_groups_listing(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--groups"]) == 0
        assert capsys.readouterr().out.split() == list(STAGE_D_GROUPS)

    def test_index_emits_shell(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--group", "bingo_isalsr", "--index", "1"]) == 0
        out = capsys.readouterr().out
        assert "D_PROBLEM='Pagie-1'" in out
        # Index 1 of this group is the CERTIFICATION cell: trace off, seed 101.
        assert "D_TRACE='0'" in out
        assert "D_SEED='101'" in out

        assert main(["--group", "bingo_isalsr", "--index", "4"]) == 0
        traced = capsys.readouterr().out
        assert "D_TRACE='1'" in traced
        assert "D_SEED='102'" in traced
        assert "D_CONFIG_NAME='bingo_hard_trace.yaml'" in traced

    def test_bad_index_exits_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--group", "udfs", "--index", "99"]) == 1
        assert "ERROR" in capsys.readouterr().err

    def test_index_without_group_exits_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--index", "1"]) == 1
        assert "ERROR" in capsys.readouterr().err

    def test_json_registry_round_trips(self, capsys: pytest.CaptureFixture[str]) -> None:
        import json

        assert main(["--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload) == 13
        assert sum(1 for row in payload if row["trace"]) == 1
