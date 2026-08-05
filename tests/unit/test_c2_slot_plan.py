"""Tests for the C2 array slot plan (``experiments/scripts/c2_slot_plan.py``).

The properties that matter are the ones a wrong plan would violate silently:
the budget must be spent exactly, no array may be starved or over-slotted, and
the work-proportional allocation must never be *worse* than the uniform one it
replaces — that last one is the whole justification for the change, so it is
asserted rather than assumed.
"""

from __future__ import annotations

import os

import pytest

from experiments.scripts.c2_slot_plan import (
    ARMS,
    DEFAULT_SLOT_BUDGET,
    MEASURED_RUNTIME_HOURS,
    MEM_GB,
    METHODS,
    RUNTIME_HOURS,
    SUITES,
    WALL,
    ArrayPlan,
    SlotPlanError,
    allocate_throttles,
    build_plan,
    format_plan_tsv,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "configs")


# --------------------------------------------------------------------------
# allocate_throttles
# --------------------------------------------------------------------------


def test_allocation_spends_the_whole_budget() -> None:
    slots = allocate_throttles([100.0, 200.0, 300.0], [1000, 1000, 1000], 600)
    assert sum(slots) == 600


@pytest.mark.parametrize("budget", [42, 100, 336, 1008, 2016, 4032])
def test_allocation_spends_the_whole_budget_at_every_scale(budget: int) -> None:
    works = [float(w) for w in (280, 240, 200, 200, 200, 160, 120) * 6]
    caps = [10_000] * len(works)
    slots = allocate_throttles(works, caps, budget)
    assert sum(slots) == budget


def test_allocation_is_proportional_to_work() -> None:
    slots = allocate_throttles([100.0, 200.0, 400.0], [10_000] * 3, 700)
    # 1 : 2 : 4 within one slot of rounding
    assert slots[1] == pytest.approx(2 * slots[0], abs=2)
    assert slots[2] == pytest.approx(4 * slots[0], abs=2)


def test_no_array_is_starved() -> None:
    slots = allocate_throttles([1.0, 1.0, 10_000.0], [500, 500, 500], 500)
    assert all(k >= 1 for k in slots)


def test_no_array_exceeds_its_task_count() -> None:
    caps = [5, 5, 1000]
    slots = allocate_throttles([1000.0, 1000.0, 1000.0], caps, 900)
    assert all(k <= c for k, c in zip(slots, caps, strict=True))


def test_capped_slots_are_redistributed_not_lost() -> None:
    # Two tiny arrays cap out at 5; the third must absorb the rest.
    slots = allocate_throttles([10.0, 10.0, 1000.0], [5, 5, 1000], 500)
    assert slots[0] == 5 and slots[1] == 5
    assert sum(slots) == 500


def test_budget_above_total_capacity_is_clamped() -> None:
    slots = allocate_throttles([1.0, 1.0], [10, 10], 1000)
    assert slots == [10, 10]


def test_allocation_is_deterministic() -> None:
    works = [float(w) for w in (280, 240, 200, 160, 120)] * 6
    caps = [10_000] * len(works)
    a = allocate_throttles(works, caps, 2016)
    b = allocate_throttles(works, caps, 2016)
    assert a == b


@pytest.mark.parametrize(
    ("works", "caps", "budget"),
    [
        ([1.0, 2.0], [10], 10),  # length mismatch
        ([], [], 10),  # empty
        ([1.0] * 42, [10] * 42, 41),  # budget below one-per-array
        ([0.0, 0.0], [10, 10], 10),  # zero total work
    ],
)
def test_bad_inputs_raise(works: list[float], caps: list[int], budget: int) -> None:
    with pytest.raises(SlotPlanError):
        allocate_throttles(works, caps, budget)


# --------------------------------------------------------------------------
# The property the change exists for
# --------------------------------------------------------------------------


def _makespan(plan: list[ArrayPlan]) -> float:
    return max(p.finish_h for p in plan)


@pytest.mark.parametrize("n_seeds", [3, 20, 30])
@pytest.mark.parametrize("uniform_k", [8, 24, 48, 96])
def test_proportional_never_loses_to_uniform(n_seeds: int, uniform_k: int) -> None:
    """At the same total slots, the apportioned plan must not finish later.

    This is the entire justification for the module. If it ever fails, the
    uniform split should be restored.
    """
    seeds = list(range(1, n_seeds + 1))
    uni = build_plan(CONFIG_DIR, seeds, uniform=uniform_k)
    budget = sum(p.throttle for p in uni)
    prop = build_plan(CONFIG_DIR, seeds, budget=budget)

    assert sum(p.throttle for p in prop) == budget, "must spend the same slots"
    assert _makespan(prop) <= _makespan(uni) + 1e-9


def test_proportional_beats_uniform_at_the_production_shape() -> None:
    """The measured case: 20 seeds at a mean of %24 must gain at least 1.5x."""
    seeds = list(range(1, 21))
    uni = build_plan(CONFIG_DIR, seeds, uniform=24)
    prop = build_plan(CONFIG_DIR, seeds, budget=sum(p.throttle for p in uni))
    assert _makespan(uni) / _makespan(prop) >= 1.5


@pytest.mark.parametrize("t_bingo", [4.0, 5.15, 8.0, 10.0, 12.0])
def test_sensitivity_to_the_assumed_bingo_runtime(
    t_bingo: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planning at T_bingo=8 h must still beat uniform when the truth differs.

    F-19 moved three suites' ``max_evals`` tenfold and the D2 suites have no
    runtime data, so the planned ``T_bingo`` will be wrong. It must not matter.
    """
    seeds = list(range(1, 21))
    planned = build_plan(CONFIG_DIR, seeds, budget=2016)  # uses T_bingo = 8.0
    uniform = build_plan(CONFIG_DIR, seeds, uniform=48)

    def rescore(plan: list[ArrayPlan]) -> float:
        return max(
            p.n_tasks * (t_bingo if p.method == "bingo" else RUNTIME_HOURS["udfs"]) / p.throttle
            for p in plan
        )

    assert rescore(planned) <= rescore(uniform)


# --------------------------------------------------------------------------
# build_plan
# --------------------------------------------------------------------------


def test_plan_has_42_arrays_in_submission_order() -> None:
    plan = build_plan(CONFIG_DIR, list(range(1, 21)))
    assert len(plan) == 42
    assert [p.key for p in plan[:7]] == [f"udfs:baseline:{s}" for s in SUITES]
    # UDFS first: it is the long pole and should queue while fairshare is highest.
    assert {p.method for p in plan[:21]} == {"udfs"}
    assert {p.method for p in plan[21:]} == {"bingo"}


def test_plan_task_counts_match_the_launch_ledger() -> None:
    """EXECUTION-PLAN §11.3: 240/200/200/200/160/120/280 per (method, arm)."""
    plan = build_plan(CONFIG_DIR, list(range(1, 21)))
    expected = {
        "nguyen": 240,
        "feynman": 200,
        "hard": 200,
        "cherrypicked": 200,
        "roundoff": 160,
        "feynman_remainder": 120,
        "strogatz": 280,
    }
    for p in plan:
        assert p.n_tasks == expected[p.suite], p.key
    assert sum(p.n_tasks for p in plan) == 8400


def test_plan_totals_at_30_seeds() -> None:
    plan = build_plan(CONFIG_DIR, list(range(1, 31)))
    assert sum(p.n_tasks for p in plan) == 12600


def test_every_array_gets_a_workable_throttle() -> None:
    plan = build_plan(CONFIG_DIR, list(range(1, 21)))
    for p in plan:
        assert 1 <= p.throttle <= p.n_tasks, p.key


def test_default_budget_is_spent() -> None:
    plan = build_plan(CONFIG_DIR, list(range(1, 21)))
    assert sum(p.throttle for p in plan) == DEFAULT_SLOT_BUDGET


def test_missing_config_raises_rather_than_guessing() -> None:
    with pytest.raises(SlotPlanError, match="missing config"):
        build_plan("/nonexistent/configs", [1])


def test_zero_seeds_raises() -> None:
    with pytest.raises(SlotPlanError):
        build_plan(CONFIG_DIR, [])


# --------------------------------------------------------------------------
# The resource table itself
# --------------------------------------------------------------------------


CAMPAIGN_SEEDS = 30


def test_campaign_seed_count_is_declared_by_every_config() -> None:
    """The configs' ``n_seeds`` is a LIVE fallback, not documentation.

    ``orchestrator.py:641`` reads ``n_seeds`` whenever ``--seeds`` is absent. The
    launcher always passes ``--seeds``, but anything that does not — a manual
    re-run of a failed cell, a resume, an analysis script — silently gets the
    config's value instead. A config left at 20 while the campaign runs 30 is the
    ``ISALSR_LEDGER_ENABLED`` / ``shadow_hash`` shape a fourth time: a default
    that is wrong on every path that does not override it.

    Read through the YAML loader, never the file text, so a commented-out or
    shadowed key cannot pass.
    """
    import yaml

    seen: dict[str, int | None] = {}
    for method in METHODS:
        for suite in SUITES:
            path = os.path.join(CONFIG_DIR, f"{method}_{suite}.yaml")
            with open(path) as handle:
                cfg = yaml.safe_load(handle)
            seen[f"{method}_{suite}"] = cfg.get("experiment", {}).get("n_seeds")

    assert len(seen) == 14
    missing = [k for k, v in seen.items() if v is None]
    assert not missing, f"configs not declaring n_seeds (would inherit): {missing}"
    wrong = {k: v for k, v in seen.items() if v != CAMPAIGN_SEEDS}
    assert not wrong, f"n_seeds != {CAMPAIGN_SEEDS}: {wrong}"


def test_trace_config_differs_from_its_parent_by_one_key() -> None:
    """`bingo_hard_trace.yaml` must stay `bingo_hard.yaml` + `shadow_hash` alone.

    The trace config's own header states it is *"bingo_hard.yaml with ONE key
    changed: shadow_hash: true"*, and its body repeats *"everything else below is
    byte-identical ... if that file changes, this one must change with it"*. That
    contract existed only as a comment, and the 2026-08-05 seed change broke it
    silently: moving `bingo_hard.yaml` to `n_seeds: 30` left the trace config at
    20, so it differed by two keys with nothing failing.

    It matters because the split exists so the certification cells and the traced
    cell differ *visibly and only* in `shadow_hash` (audit.md §7.3) — a second
    drifting key makes `config_sha256` stop meaning "this is the shadow variant".

    The trace config is deliberately outside `CAMPAIGN_SUITES` in
    `test_budget_uniformity.py`, so nothing else checks it.
    """
    import yaml

    with open(os.path.join(CONFIG_DIR, "bingo_hard.yaml")) as handle:
        parent = yaml.safe_load(handle)
    with open(os.path.join(CONFIG_DIR, "bingo_hard_trace.yaml")) as handle:
        trace = yaml.safe_load(handle)

    def flat(obj: object, prefix: str = "") -> dict[str, object]:
        out: dict[str, object] = {}
        if isinstance(obj, dict):
            for key, value in obj.items():
                out.update(flat(value, f"{prefix}{key}."))
        else:
            out[prefix[:-1]] = obj
        return out

    fp, ft = flat(parent), flat(trace)
    differing = {k for k in set(fp) | set(ft) if fp.get(k, "<absent>") != ft.get(k, "<absent>")}
    # `experiment.name` must differ -- the two configs write to different roots.
    differing.discard("experiment.name")

    assert differing == {"bingo.shadow_hash"}, (
        "trace config must differ from bingo_hard.yaml by shadow_hash alone; "
        f"also differs in {sorted(differing - {'bingo.shadow_hash'})}"
    )
    assert trace["bingo"]["shadow_hash"] is True
    assert parent["bingo"]["shadow_hash"] is False


def test_campaign_plan_at_the_declared_seed_count() -> None:
    plan = build_plan(CONFIG_DIR, list(range(1, CAMPAIGN_SEEDS + 1)))
    assert sum(p.n_tasks for p in plan) == 70 * CAMPAIGN_SEEDS * 6 // 6 * 6
    assert sum(p.n_tasks for p in plan) == 12600
    assert sum(p.throttle for p in plan) == DEFAULT_SLOT_BUDGET


def test_expected_makespan_at_measured_runtimes_is_reported_separately() -> None:
    """The allocation is weighted pessimistically; the forecast must not be.

    Scoring the plan at ``MEASURED_RUNTIME_HOURS`` must give a strictly shorter
    makespan than scoring it at the planning weights, or the two dictionaries
    have drifted into agreement and the distinction has quietly been lost.
    """
    plan = build_plan(CONFIG_DIR, list(range(1, CAMPAIGN_SEEDS + 1)))
    planned = max(p.finish_h for p in plan)
    expected = max(p.n_tasks * MEASURED_RUNTIME_HOURS[p.method] / p.throttle for p in plan)
    assert MEASURED_RUNTIME_HOURS["bingo"] < RUNTIME_HOURS["bingo"]
    assert MEASURED_RUNTIME_HOURS["udfs"] == RUNTIME_HOURS["udfs"]
    assert expected < planned
    # Sanity: the 30-seed campaign should land near 54 h, not near a week.
    assert 40.0 < expected < 70.0


def test_bingo_isalsr_memory_covers_the_hard_ceiling() -> None:
    """32 GB must clear the max_evals-bounded worst case with real margin.

    ``canonical_seen: set[int]`` is the only unbounded container, a candidate
    cannot enter it without being scored, and Bingo stops at
    ``max_evals = 100M`` — so ``n_unique <= 100M`` on any problem. At the
    measured 81.5 B/entry (production allocator) with the measured 1.16x
    resize transient, plus the 0.42 GB no-dedup baseline, that is ~9.4 GB.
    """
    entries = 100_000_000
    bytes_per_entry = 81.5
    resize_transient = 1.16
    baseline_gb = 0.42
    ceiling_gb = entries * bytes_per_entry * resize_transient / 1024**3 + baseline_gb

    assert ceiling_gb == pytest.approx(9.4, abs=0.5)
    assert MEM_GB[("bingo", "isalsr")] >= 3 * ceiling_gb


def test_memory_table_covers_every_arm() -> None:
    for method in METHODS:
        for arm in ARMS:
            assert MEM_GB[(method, arm)] > 0


def test_wall_is_under_three_days_on_every_method() -> None:
    """Above 3 days the job drops from medium_uma to long_uma and loses 5,000
    priority points; measured 2026-08-05."""
    for method in METHODS:
        days, hms = WALL[method].split("-")
        hours = int(days) * 24 + int(hms.split(":")[0])
        assert 13 <= hours < 72, f"{method}: {WALL[method]}"


def test_udfs_runtime_is_the_measured_saturation() -> None:
    """UDFS has no max_evals and saturated 12.00 h on 600/600 C1 runs."""
    assert RUNTIME_HOURS["udfs"] == 12.0


def test_bingo_runtime_is_planned_above_the_measured_mean() -> None:
    """Planning high is the safe direction; C1 measured 5.15 h, F-19 raised it."""
    assert RUNTIME_HOURS["bingo"] > 5.15


# --------------------------------------------------------------------------
# TSV transport to the shell
# --------------------------------------------------------------------------


def test_tsv_is_parseable_and_complete() -> None:
    plan = build_plan(CONFIG_DIR, list(range(1, 21)))
    rows = format_plan_tsv(plan).splitlines()
    assert len(rows) == 42
    for row, p in zip(rows, plan, strict=True):
        fields = row.split("\t")
        assert len(fields) == 7
        assert fields[:3] == [p.method, p.arm, p.suite]
        assert int(fields[3]) == p.n_tasks
        assert int(fields[4]) == p.throttle


def test_tsv_contains_no_spaces_in_key_fields() -> None:
    """The shell reads these with `while read`; a space would split a field."""
    plan = build_plan(CONFIG_DIR, list(range(1, 21)))
    for row in format_plan_tsv(plan).splitlines():
        for field in row.split("\t"):
            assert " " not in field
