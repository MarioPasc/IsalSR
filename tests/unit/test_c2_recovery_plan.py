"""Unit tests for the tail-aware recovery sizing and the missing-cell census.

The property under test is the one the recovery pass exists to establish:

    a chunk of ``B`` cells, each charged an allowance ``a``, under a wall ``W``,
    reaches its LAST cell -- i.e. ``worker.sh``'s start-deadline
    ``W - CELL_RESERVE_H`` strictly exceeds the worst-case elapsed time
    ``(B - 1) * a`` at which that cell begins.

Everything else here guards the two ways that property can be lost in practice:
a partition that stops covering every cell, and a plan that silently changes the
main pass.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from experiments.scripts import c2_missing_cells as mc
from experiments.scripts import c2_slot_plan as sp
from experiments.scripts.c2_task_spec import decode_chunk, n_tasks_for

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments",
    "configs",
)


# ---------------------------------------------------------------------------
# The no-deferral inequality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", sp.METHODS)
@pytest.mark.parametrize("suite", sp.SUITES)
@pytest.mark.parametrize("mode", sp.RECOVERY_MODES)
def test_recovery_sizing_defers_nothing(method: str, suite: str, mode: str) -> None:
    """Every (method, suite, mode) triple must clear the deadline inequality."""
    allowance = sp.recovery_allowance_h(method, suite, mode)
    bundle = sp.recovery_bundle(allowance, n_cells=10_000)
    wall = sp.recovery_wall_hours(bundle, allowance)
    assert wall <= sp.MAX_WALL_H
    assert wall >= sp.MIN_WALL_H
    assert sp.defers_nothing(bundle, allowance, wall)


def test_safe_mode_charges_the_full_payload_cap() -> None:
    """`safe` must be distribution-free: the allowance is the cap, not the p90."""
    for method in sp.METHODS:
        for suite in sp.SUITES:
            assert sp.recovery_allowance_h(method, suite, "safe") == sp.CELL_RESERVE_H
    # Which pins the bundle at 3 everywhere: 2 x 12.5 = 25 h < 25.5 h deadline.
    assert sp.recovery_bundle(sp.CELL_RESERVE_H, 10_000) == 3
    assert sp.recovery_wall_hours(3, sp.CELL_RESERVE_H) == 38


def test_p90_mode_is_never_looser_than_the_cap() -> None:
    """The p90 allowance is clipped by the payload reserve, never above it."""
    for (method, suite), p90 in sp.P90_CELL_HOURS.items():
        allowance = sp.recovery_allowance_h(method, suite, "p90")
        assert allowance <= sp.CELL_RESERVE_H
        assert allowance == min(sp.CELL_RESERVE_H, p90 + sp.RECOVERY_TEARDOWN_H)


def test_p90_mode_buys_bigger_bundles_where_the_tail_is_short() -> None:
    """The whole point of `p90`: fewer placements on the fast Bingo suites."""
    safe = sp.recovery_bundle(sp.recovery_allowance_h("bingo", "feynman", "safe"), 10_000)
    p90 = sp.recovery_bundle(sp.recovery_allowance_h("bingo", "feynman", "p90"), 10_000)
    assert p90 > safe
    # ... and no bigger bundle on the suites that saturate the 12 h cap.
    assert sp.recovery_bundle(
        sp.recovery_allowance_h("udfs", "feynman", "p90"), 10_000
    ) == sp.recovery_bundle(sp.recovery_allowance_h("udfs", "feynman", "safe"), 10_000)


def test_recovery_bundle_is_the_largest_that_still_clears_the_deadline() -> None:
    """One more cell must break the inequality, or the bundle is not maximal."""
    for allowance in (0.55, 0.93, 4.71, 6.05, 7.68, sp.CELL_RESERVE_H):
        bundle = sp.recovery_bundle(allowance, 10_000)
        assert sp.defers_nothing(bundle, allowance, sp.recovery_wall_hours(bundle, allowance))
        bigger = bundle + 1
        assert not sp.defers_nothing(bigger, allowance, sp.recovery_wall_hours(bigger, allowance))


def test_recovery_bundle_never_exceeds_the_array() -> None:
    """A chunk larger than the array would make the last tasks empty."""
    assert sp.recovery_bundle(0.05, n_cells=2) == 2
    assert sp.recovery_bundle(sp.CELL_RESERVE_H, n_cells=1) == 1


def test_forced_bundle_that_could_defer_is_refused_not_warned() -> None:
    """An operator override is validated, because a warning would be ignored."""
    with pytest.raises(sp.SlotPlanError, match="would defer"):
        sp.recovery_bundle(sp.CELL_RESERVE_H, n_cells=1000, forced=8)
    assert sp.recovery_bundle(sp.CELL_RESERVE_H, n_cells=1000, forced=2) == 2


def test_wall_makes_the_deadline_strict_not_merely_equal() -> None:
    """`worker.sh` defers on `elapsed >= cutoff`, so equality must not occur."""
    # 34.5 / 11.5 = 3 exactly, i.e. the pathological case for ceil().
    allowance = 11.5
    bundle = sp.recovery_bundle(allowance, 10_000)
    wall = sp.recovery_wall_hours(bundle, allowance)
    assert wall - sp.CELL_RESERVE_H > (bundle - 1) * allowance


def test_recovery_allowance_rejects_unknown_mode() -> None:
    with pytest.raises(sp.SlotPlanError, match="unknown recovery mode"):
        sp.recovery_allowance_h("udfs", "feynman", "median")


# ---------------------------------------------------------------------------
# Coverage: a different bundle must still partition the array exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bundle", [1, 2, 3, 7, 27, 63, 123, 164])
def test_every_cell_is_owned_by_exactly_one_task_at_any_bundle(bundle: int) -> None:
    """Re-running at a smaller bundle must not orphan or duplicate a cell."""
    problems = [f"P{i}" for i in range(12)]
    seeds = list(range(1, 31))
    n_cells = len(problems) * len(seeds)
    n_tasks = n_tasks_for(n_cells, bundle)

    seen: list[tuple[str, int]] = []
    for index in range(1, n_tasks + 1):
        seen.extend(decode_chunk(problems, seeds, bundle, index))

    assert len(seen) == n_cells
    assert len(set(seen)) == n_cells
    assert set(seen) == {(p, s) for p in problems for s in seeds}


def test_partition_at_recovery_bundle_covers_the_main_pass_partition() -> None:
    """The recovery pass and the main pass must agree on the cell SET."""
    problems = [f"P{i}" for i in range(10)]
    seeds = list(range(1, 31))
    main = {
        cell
        for i in range(1, n_tasks_for(300, 27) + 1)
        for cell in decode_chunk(problems, seeds, 27, i)
    }
    recovery = {
        cell
        for i in range(1, n_tasks_for(300, 3) + 1)
        for cell in decode_chunk(problems, seeds, 3, i)
    }
    assert main == recovery


# ---------------------------------------------------------------------------
# build_recovery_plan
# ---------------------------------------------------------------------------


def test_build_recovery_plan_selects_only_what_was_asked() -> None:
    plan = sp.build_recovery_plan(
        CONFIG_DIR, list(range(1, 31)), only=["udfs:*:feynman", "bingo:baseline:nguyen"]
    )
    keys = sorted(p.key for p in plan)
    assert keys == [
        "bingo:baseline:nguyen",
        "udfs:baseline:feynman",
        "udfs:hash:feynman",
        "udfs:isalsr:feynman",
    ]


def test_build_recovery_plan_rejects_a_malformed_selector() -> None:
    with pytest.raises(sp.SlotPlanError, match="method:arm:suite"):
        sp.build_recovery_plan(CONFIG_DIR, [1, 2], only=["udfs:feynman"])


def test_build_recovery_plan_requires_a_selector() -> None:
    with pytest.raises(sp.SlotPlanError, match="--only is required"):
        sp.build_recovery_plan(CONFIG_DIR, [1, 2], only=[])


def test_build_recovery_plan_rejects_a_selector_matching_nothing() -> None:
    with pytest.raises(sp.SlotPlanError, match="match none"):
        sp.build_recovery_plan(CONFIG_DIR, [1, 2], only=["udfs:*:no_such_suite"])


@pytest.mark.parametrize("mode", sp.RECOVERY_MODES)
def test_built_recovery_plan_defers_nothing_and_covers_every_cell(mode: str) -> None:
    plan = sp.build_recovery_plan(
        CONFIG_DIR, list(range(1, 31)), only=["udfs:*:*", "bingo:*:*"], mode=mode
    )
    assert len(plan) == len(sp.METHODS) * len(sp.ARMS) * len(sp.SUITES)
    assert sum(p.n_cells for p in plan) == 12_600
    for p in plan:
        allowance = sp.recovery_allowance_h(p.method, p.suite, mode)
        assert sp.defers_nothing(p.bundle, allowance, p.wall_h)
        assert p.n_tasks * p.bundle >= p.n_cells
        assert p.throttle >= 1
        assert p.throttle <= p.n_tasks
        assert p.mem_gb == sp.MEM_GB[(p.method, p.arm)]


def test_recovery_plan_memory_matches_the_main_pass() -> None:
    """Memory is not the variable being changed (the one C2 OOM was a spike)."""
    main = {p.key: p.mem_gb for p in sp.build_plan(CONFIG_DIR, list(range(1, 31)))}
    rec = {
        p.key: p.mem_gb
        for p in sp.build_recovery_plan(CONFIG_DIR, list(range(1, 31)), only=["*:*:*"])
    }
    assert rec == main


def test_recovery_tsv_has_the_ten_columns_the_shell_reads() -> None:
    plan = sp.build_recovery_plan(CONFIG_DIR, list(range(1, 31)), only=["udfs:*:feynman"])
    rows = sp.format_plan_tsv(plan).splitlines()
    assert len(rows) == 3
    for row in rows:
        assert len(row.split("\t")) == 10


def test_recovery_tsv_cutoff_matches_the_wall_it_ships_with() -> None:
    """A deadline larger than `wall - CELL_RESERVE` reintroduces TIMEOUT."""
    plan = sp.build_recovery_plan(CONFIG_DIR, list(range(1, 31)), only=["*:*:*"])
    for row in sp.format_plan_tsv(plan).splitlines():
        fields = row.split("\t")
        wall_h = int(fields[6].split("-")[0]) * 24 + int(fields[6].split("-")[1].split(":")[0])
        assert int(fields[8]) == int((wall_h - sp.CELL_RESERVE_H) * 3600)


def test_match_selector_wildcards_each_field() -> None:
    assert sp.match_selector("udfs:hash:feynman", ["udfs:*:feynman"])
    assert sp.match_selector("udfs:hash:feynman", ["*:hash:*"])
    assert sp.match_selector("udfs:hash:feynman", ["*:*:*"])
    assert not sp.match_selector("udfs:hash:feynman", ["bingo:*:feynman"])
    assert not sp.match_selector("udfs:hash:feynman", ["udfs:hash"])


# ---------------------------------------------------------------------------
# The main pass must be untouched by all of the above
# ---------------------------------------------------------------------------


def test_main_plan_is_unaffected_by_the_recovery_additions() -> None:
    """The campaign is live; `build_plan` must still produce its 42 rows."""
    plan = sp.build_plan(CONFIG_DIR, list(range(1, 31)))
    assert len(plan) == 42
    assert sum(p.n_cells for p in plan) == 12_600
    by_key = {p.key: p for p in plan}
    # The bundles the 2026-08-09 audit measured, so a regression is visible here.
    assert by_key["udfs:baseline:feynman"].bundle == 27
    assert by_key["bingo:baseline:nguyen"].bundle == 164
    assert by_key["udfs:baseline:feynman_remainder"].bundle == 4


# ---------------------------------------------------------------------------
# c2_missing_cells
# ---------------------------------------------------------------------------


def _write_run_log(root: Path, cell: mc.Cell, *, valid: bool = True) -> Path:
    path = root / cell.relpath() / "run_log.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        {"method": cell.method, "variant": cell.arm, "problem": cell.problem, "seed": cell.seed}
        if valid
        else {"method": cell.method}
    )
    path.write_text(json.dumps(payload))
    return path


def test_missing_cells_axes_do_not_drift_from_the_planner() -> None:
    """`c2_missing_cells` re-declares the three axes so a DEPLOYED-tree copy of
    `c2_slot_plan` cannot shadow it (namespace-package finding on Picasso puts
    the editable install's tree first). This pins the two definitions together.
    """
    assert mc.METHODS == sp.METHODS
    assert mc.ARMS == sp.ARMS
    assert mc.SUITES == sp.SUITES
    for key, patterns in [
        ("udfs:hash:feynman", ["udfs:*:feynman"]),
        ("udfs:hash:feynman", ["*:*:*"]),
        ("udfs:hash:feynman", ["bingo:*:*"]),
        ("udfs:hash:feynman", ["udfs:hash"]),
    ]:
        assert mc.match_selector(key, patterns) == sp.match_selector(key, patterns)


def test_missing_cells_imports_nothing_from_the_planner() -> None:
    """A cross-tree import is exactly what broke on Picasso; keep it absent."""
    tree = ast.parse(Path(mc.__file__).read_text())
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any("c2_slot_plan" in name for name in imported), (
        f"c2_missing_cells must not import from c2_slot_plan; found {imported}"
    )


def test_expected_universe_is_the_campaign_universe() -> None:
    cells = mc.expected_cells(CONFIG_DIR, list(range(1, 31)))
    assert len(cells) == 12_600
    assert len({(c.method, c.arm, c.suite, c.problem, c.seed) for c in cells}) == 12_600


def test_cell_relpath_matches_the_worker_layout() -> None:
    cell = mc.Cell("bingo", "isalsr", "cherrypicked", "Vlad-7", 25)
    assert cell.relpath() == os.path.join("bingo", "cherrypicked", "vlad_7", "isalsr", "seed_25")


def test_missing_cells_finds_exactly_the_absent_ones(tmp_path: Path) -> None:
    cells = [mc.Cell("udfs", "baseline", "feynman", "I.6.2", seed) for seed in (1, 2, 3)]
    _write_run_log(tmp_path, cells[0])
    _write_run_log(tmp_path, cells[2])
    gaps = mc.missing_cells(str(tmp_path), cells)
    assert [c.seed for c in gaps] == [2]


def test_strict_mode_treats_a_truncated_run_log_as_missing(tmp_path: Path) -> None:
    cell = mc.Cell("udfs", "baseline", "feynman", "I.6.2", 1)
    path = _write_run_log(tmp_path, cell, valid=False)
    assert mc.missing_cells(str(tmp_path), [cell]) == []
    assert mc.missing_cells(str(tmp_path), [cell], strict=True) == [cell]
    path.write_text("{ this is not json")
    assert mc.missing_cells(str(tmp_path), [cell], strict=True) == [cell]


def test_selectors_names_every_array_with_a_gap_and_nothing_else() -> None:
    gaps = [
        mc.Cell("udfs", "baseline", "feynman", "I.6.2", 1),
        mc.Cell("udfs", "baseline", "feynman", "I.6.2", 2),
        mc.Cell("bingo", "isalsr", "nguyen", "Nguyen-1", 3),
    ]
    assert mc.selectors(gaps) == "udfs:baseline:feynman,bingo:isalsr:nguyen"
    assert mc.selectors([]) == ""


def test_selectors_output_is_accepted_by_the_recovery_planner() -> None:
    """The two tools must compose; that is the whole scoping workflow."""
    gaps = [
        mc.Cell("udfs", "hash", "feynman", "I.6.2", 1),
        mc.Cell("bingo", "baseline", "nguyen", "Nguyen-1", 3),
    ]
    only = [s for s in mc.selectors(gaps).split(",") if s]
    plan = sp.build_recovery_plan(CONFIG_DIR, list(range(1, 31)), only=only)
    assert sorted(p.key for p in plan) == ["bingo:baseline:nguyen", "udfs:hash:feynman"]


def test_summary_reconciles(tmp_path: Path) -> None:
    cells = [mc.Cell("udfs", "baseline", "feynman", "I.6.2", s) for s in (1, 2)]
    _write_run_log(tmp_path, cells[0])
    text = mc.summarise(cells, mc.missing_cells(str(tmp_path), cells))
    assert "udfs:baseline:feynman" in text
    assert "TOTAL" in text
