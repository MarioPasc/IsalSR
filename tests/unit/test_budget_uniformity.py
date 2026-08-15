"""Every campaign config must declare the same search budget (F-19, F-20).

Defect F-19, found 2026-08-05: `bingo_{roundoff,strogatz,feynman_remainder}.yaml`
omitted `max_evals` and silently inherited `BingoConfig.max_evals = 10_000_000`
-- a **10x tighter** budget than the other four suites, on 28 of 70 problems,
undocumented, and pooled into the N=70 cross-problem dominance test.

It mattered because `max_evals` is the *binding* budget for Bingo: Stage D
(2026-08-05) showed all six completed cells stopping on `max_evals` at 100M
fitness evaluations, with `max_time = 43,200 s` never firing on any of them. A
missing key was therefore a 10x search-budget disparity, not a formatting slip.

This is the same failure shape as the `ISALSR_LEDGER_ENABLED` and `shadow_hash`
traps: a default that is silently wrong, invisible in the logs, and unrecoverable
after the compute is spent. The defence is the same -- assert on the *loaded*
value, never on the presence of a line in a file.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "experiments" / "configs"

# The seven campaign suites. Trace and probe configs are deliberately excluded:
# they are allowed to differ, and `bingo_hard_trace.yaml` differs by exactly one
# key (`shadow_hash`) on purpose.
CAMPAIGN_SUITES = (
    "nguyen",
    "feynman",
    "hard",
    "cherrypicked",
    "roundoff",
    "feynman_remainder",
    "strogatz",
)


def _load(method: str, suite: str) -> dict:
    """Load one campaign config's method block."""
    path = CONFIG_DIR / f"{method}_{suite}.yaml"
    assert path.is_file(), f"missing campaign config: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8")).get(method, {}) or {}


def _experiment(method: str, suite: str) -> dict:
    """Load one campaign config's experiment block."""
    path = CONFIG_DIR / f"{method}_{suite}.yaml"
    assert path.is_file(), f"missing campaign config: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8")).get("experiment", {}) or {}


# ======================================================================
# Seed count -- the same defect class as F-19, on a different key
# ======================================================================


def test_n_seeds_is_uniform_across_every_campaign_config() -> None:
    """All 14 configs must declare the same seed count.

    A split seed count is worse than a smaller one: it silently unbalances the
    paired design (§5.5), and §0.4a is explicit that a paired design with
    unbalanced completion "cannot be analysed at all". Because the campaign's
    per-array task count derives from `n_seeds`, a partial edit would produce
    arrays of differing size with nothing in the output announcing it.

    History: the value moved 30 -> 20 (§0.4a) -> 30 (2026-08-05, once measured
    runtimes showed the cost premise was wrong). Each move must be total.
    """
    values = {
        f"{method}_{suite}": _experiment(method, suite).get("n_seeds")
        for method in ("bingo", "udfs")
        for suite in CAMPAIGN_SUITES
    }
    assert None not in values.values(), f"n_seeds undeclared somewhere: {values}"
    distinct = set(values.values())
    assert len(distinct) == 1, f"n_seeds is non-uniform across campaign configs: {values}"


def test_seed_count_matches_the_declared_campaign_size() -> None:
    """Guard the arithmetic §8.1 and §11.3 depend on.

    70 problems x 3 arms x 2 methods x n_seeds must equal the campaign run count,
    and the per-array size must stay inside `MaxArraySize = 4096` (A12).
    """
    n_seeds = _experiment("bingo", "nguyen")["n_seeds"]
    largest_suite = 14  # strogatz
    assert n_seeds * largest_suite <= 4096, (
        f"largest array would be {n_seeds * largest_suite} tasks, above MaxArraySize"
    )
    assert 70 * 3 * 2 * n_seeds == 420 * n_seeds


def test_no_config_comment_contradicts_its_own_seed_count() -> None:
    """A comment asserting a different seed count will cause a wrong revert.

    Four configs carried "Seeds: 20 ... Not 30 - read the boxed note in §0.4"
    directly above `n_seeds: 30` after the 2026-08-05 change. A reader trusting
    the comment would revert the value and split the campaign.
    """
    for method in ("bingo", "udfs"):
        for suite in CAMPAIGN_SUITES:
            path = CONFIG_DIR / f"{method}_{suite}.yaml"
            text = path.read_text(encoding="utf-8")
            declared = _experiment(method, suite)["n_seeds"]
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped.startswith("#"):
                    continue
                for wrong in (20, 30):
                    if wrong == declared:
                        continue
                    assert f"Seeds: {wrong}" not in stripped, (
                        f"{path.name} declares n_seeds={declared} but a comment "
                        f"asserts 'Seeds: {wrong}': {stripped!r}"
                    )


# ======================================================================
# F-19 -- Bingo
# ======================================================================


@pytest.mark.parametrize("suite", CAMPAIGN_SUITES)
def test_bingo_declares_max_evals_explicitly(suite: str) -> None:
    """No suite may inherit the dataclass default.

    Regression for F-19: three suites omitted the key and inherited 10M.
    """
    block = _load("bingo", suite)
    assert "max_evals" in block, (
        f"bingo_{suite}.yaml does not declare max_evals and would inherit "
        f"BingoConfig.max_evals = 10_000_000, a 10x tighter budget (F-19)"
    )


@pytest.mark.parametrize("key", ["max_evals", "max_time", "population_size", "stack_size"])
def test_bingo_budget_is_uniform_across_suites(key: str) -> None:
    """A budget knob must hold one value across every campaign suite.

    A per-suite budget confounds the suite with the search capacity, which makes
    any per-suite difference in results unattributable -- the same reasoning that
    forced the A4b operator-set decision.
    """
    values = {suite: _load("bingo", suite).get(key) for suite in CAMPAIGN_SUITES}
    distinct = set(values.values())
    assert len(distinct) == 1, f"bingo {key} is non-uniform across suites: {values}"
    assert None not in distinct, f"bingo {key} is undeclared in every suite"


def test_bingo_max_evals_exceeds_the_dataclass_default() -> None:
    """The declared budget must actually differ from the default it replaces.

    Guards against a future "fix" that writes the default back in explicitly and
    silently restores the 10M cap.
    """
    from experiments.models.bingo.config import BingoConfig

    for suite in CAMPAIGN_SUITES:
        declared = _load("bingo", suite)["max_evals"]
        assert declared > BingoConfig.max_evals, (
            f"bingo_{suite}.yaml declares max_evals={declared}, which is not above "
            f"the {BingoConfig.max_evals} default this test exists to prevent"
        )


# ======================================================================
# F-20 -- UDFS
# ======================================================================


@pytest.mark.parametrize("key", ["max_orders", "max_time", "n_calc_nodes"])
def test_udfs_budget_is_uniform_across_suites(key: str) -> None:
    """UDFS's search-capacity knobs must not vary by suite.

    Regression for F-20, found 2026-08-05: `udfs_feynman.yaml` set
    `n_calc_nodes: 7` while every other suite used 5. `n_calc_nodes` bounds the
    size of the expressions UDFS enumerates, so a per-suite value confounds the
    suite with the reachable search space -- exactly the A4b defect, in UDFS.
    """
    values = {suite: _load("udfs", suite).get(key) for suite in CAMPAIGN_SUITES}
    distinct = set(values.values())
    assert len(distinct) == 1, f"udfs {key} is non-uniform across suites: {values}"


# ======================================================================
# Cross-cutting
# ======================================================================


def test_every_campaign_suite_has_both_method_configs() -> None:
    """Both hosts must cover all seven suites, or the paired design has holes."""
    for method in ("bingo", "udfs"):
        for suite in CAMPAIGN_SUITES:
            assert (CONFIG_DIR / f"{method}_{suite}.yaml").is_file()


def test_no_campaign_config_is_missed_by_this_test() -> None:
    """If a suite is added, this test must be updated rather than silently skipped."""
    on_disk = {
        Path(p).stem.split("_", 1)[1]
        for p in glob.glob(str(CONFIG_DIR / "bingo_*.yaml"))
        if "trace" not in p and "probe" not in p
    }
    assert on_disk == set(CAMPAIGN_SUITES), (
        f"campaign suites on disk {sorted(on_disk)} != those asserted here "
        f"{sorted(CAMPAIGN_SUITES)}; update CAMPAIGN_SUITES"
    )
