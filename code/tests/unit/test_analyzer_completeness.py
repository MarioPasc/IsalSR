"""Tests for the Stage E integrity checks (E6 completeness, E7 provenance).

Each test states the defect it prevents. The two properties here are the ones
C1 could not state about itself: which cells are missing, and whether every run
came from one commit and one configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.models.analyzer.completeness import (
    ROOT_PROVENANCE_KEYS,
    CampaignIntegrityError,
    Cell,
    enforce_integrity,
    infer_expected_cells,
    scan_root,
)

METHODS = ("udfs", "bingo")
BENCHMARKS = ("nguyen",)
VARIANTS = ("baseline", "hash", "isalsr")
PROBLEMS = ("nguyen_1", "nguyen_2")
SEEDS = ("seed_00", "seed_101", "seed_102")


def _write_run(
    root: Path,
    method: str,
    benchmark: str,
    problem: str,
    arm: str,
    seed: str,
    describe: str = "abc1234",
    dirty: bool = False,
    build_hash: str = "298fc1188bf1b051",
    config_sha: str | None = None,
) -> Path:
    """Write one minimal run log carrying the provenance fields the scan reads."""
    seed_dir = root / method / benchmark / problem / arm / seed
    seed_dir.mkdir(parents=True, exist_ok=True)
    path = seed_dir / "run_log.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "config_sha256": config_sha or f"cfg-{method}-{benchmark}",
                    "hardware": {
                        "git_describe": describe,
                        "git_dirty": dirty,
                        "build_hash": build_hash,
                    },
                },
                "results": {"regression": {"r2_test": 0.9}},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def complete_root(tmp_path: Path) -> Path:
    """A fully populated three-arm root: 2 methods x 2 problems x 3 arms x 3 seeds."""
    root = tmp_path / "root"
    for method in METHODS:
        for problem in PROBLEMS:
            for arm in VARIANTS:
                for seed in SEEDS:
                    _write_run(root, method, "nguyen", problem, arm, seed)
    return root


# ======================================================================
# Expected-grid inference
# ======================================================================


def test_expected_grid_is_the_cross_product(complete_root: Path) -> None:
    """The grid is methods x problems x arms x seeds, not "whatever is on disk"."""
    expected = infer_expected_cells(complete_root, METHODS, BENCHMARKS, VARIANTS)
    assert len(expected) == 2 * 2 * 3 * 3 == 36


def test_expected_grid_survives_a_deletion(complete_root: Path) -> None:
    """Deleting a run must not shrink the expectation -- that is the C1 defect.

    If the grid were inferred from surviving files, a missing cell would define
    itself away and the shortfall would again be invisible.
    """
    before = infer_expected_cells(complete_root, METHODS, BENCHMARKS, VARIANTS)
    (complete_root / "udfs/nguyen/nguyen_1/isalsr/seed_101/run_log.json").unlink()
    after = infer_expected_cells(complete_root, METHODS, BENCHMARKS, VARIANTS)
    assert after == before


def test_only_requested_arms_are_expected(complete_root: Path) -> None:
    """A two-arm query on a three-arm root yields the two-arm grid."""
    expected = infer_expected_cells(complete_root, METHODS, BENCHMARKS, ("baseline", "isalsr"))
    assert len(expected) == 2 * 2 * 2 * 3 == 24
    assert all(c.arm != "hash" for c in expected)


# ======================================================================
# E6 -- completeness
# ======================================================================


def test_complete_root_reconciles(complete_root: Path) -> None:
    """A complete root reports complete, with observed equal to expected."""
    comp, _ = scan_root(complete_root, METHODS, BENCHMARKS, VARIANTS)
    assert comp.complete
    assert comp.n_observed == comp.n_expected == 36
    assert comp.missing == []


def test_missing_cell_is_named_not_merely_counted(complete_root: Path) -> None:
    """A count that matches is worth nothing unless a mismatch names the cell."""
    (complete_root / "udfs/nguyen/nguyen_1/isalsr/seed_101/run_log.json").unlink()
    comp, _ = scan_root(complete_root, METHODS, BENCHMARKS, VARIANTS)
    assert not comp.complete
    assert comp.n_observed == 35
    assert comp.missing == [Cell("udfs", "nguyen", "nguyen_1", "isalsr", "seed_101")]
    assert "udfs/nguyen/nguyen_1/isalsr/seed_101" in comp.format_report()


def test_unreadable_run_log_is_not_silently_skipped(complete_root: Path) -> None:
    """A corrupt run log is a defect, not an absence; it must be reported."""
    victim = complete_root / "bingo/nguyen/nguyen_2/hash/seed_00/run_log.json"
    victim.write_text("{ this is not json", encoding="utf-8")
    comp, _ = scan_root(complete_root, METHODS, BENCHMARKS, VARIANTS)
    assert not comp.complete
    assert Cell("bingo", "nguyen", "nguyen_2", "hash", "seed_00") in comp.unreadable


def test_enforce_raises_on_incomplete_and_the_override_permits(complete_root: Path) -> None:
    """Fail closed by default; proceed only on an explicit override."""
    (complete_root / "udfs/nguyen/nguyen_1/isalsr/seed_101/run_log.json").unlink()
    with pytest.raises(CampaignIntegrityError, match="missing"):
        enforce_integrity(complete_root, METHODS, BENCHMARKS, VARIANTS)

    comp, _ = enforce_integrity(complete_root, METHODS, BENCHMARKS, VARIANTS, allow_incomplete=True)
    assert len(comp.missing) == 1


def test_exception_carries_the_reports(complete_root: Path) -> None:
    """The caller persists the reports without re-walking 8,400 files."""
    (complete_root / "udfs/nguyen/nguyen_1/isalsr/seed_101/run_log.json").unlink()
    with pytest.raises(CampaignIntegrityError) as excinfo:
        enforce_integrity(complete_root, METHODS, BENCHMARKS, VARIANTS)
    assert excinfo.value.completeness is not None
    assert excinfo.value.provenance is not None
    assert len(excinfo.value.completeness.missing) == 1


# ======================================================================
# E7 -- provenance
# ======================================================================


def test_uniform_provenance_is_not_mixed(complete_root: Path) -> None:
    """One commit, one build, one config per suite: nothing to report."""
    _, prov = scan_root(complete_root, METHODS, BENCHMARKS, VARIANTS)
    assert not prov.mixed
    assert prov.conflicts == []
    assert prov.n_runs == 36


@pytest.mark.parametrize("key", ["git_describe", "build_hash"])
def test_two_commits_or_builds_in_one_root_conflict(complete_root: Path, key: str) -> None:
    """A mid-wave redeploy splits provenance and must be caught, per §5.1."""
    victim = complete_root / "udfs/nguyen/nguyen_1/baseline/seed_00/run_log.json"
    payload = json.loads(victim.read_text())
    payload["metadata"]["hardware"][key] = "other-value"
    victim.write_text(json.dumps(payload), encoding="utf-8")

    _, prov = scan_root(complete_root, METHODS, BENCHMARKS, VARIANTS)
    assert prov.mixed
    assert any(key in c for c in prov.conflicts)


def test_config_sha_conflicts_within_a_suite_but_not_across(complete_root: Path) -> None:
    """config_sha256 legitimately differs per suite, never inside one."""
    # Distinct per (method, benchmark) by construction -> no conflict.
    _, prov = scan_root(complete_root, METHODS, BENCHMARKS, VARIANTS)
    assert not any("config_sha256" in c for c in prov.conflicts)

    victim = complete_root / "udfs/nguyen/nguyen_2/isalsr/seed_102/run_log.json"
    payload = json.loads(victim.read_text())
    payload["metadata"]["config_sha256"] = "edited-mid-wave"
    victim.write_text(json.dumps(payload), encoding="utf-8")

    _, prov = scan_root(complete_root, METHODS, BENCHMARKS, VARIANTS)
    assert any("config_sha256" in c and "udfs/nguyen" in c for c in prov.conflicts)


def test_absent_key_is_non_informative_not_agreement(tmp_path: Path) -> None:
    """The SP-6 trap: one value because there is no value proves nothing.

    ``git_commit`` is None on every run log this project has produced. A guard
    that counted that as agreement would pass vacuously on any input, which is
    why absent keys are reported separately and never as a clean result.
    """
    root = tmp_path / "root"
    for problem in PROBLEMS:
        for arm in VARIANTS:
            for seed in SEEDS:
                path = _write_run(root, "udfs", "nguyen", problem, arm, seed)
                payload = json.loads(path.read_text())
                payload["metadata"]["hardware"].pop("build_hash")
                path.write_text(json.dumps(payload), encoding="utf-8")

    _, prov = scan_root(root, ("udfs",), BENCHMARKS, VARIANTS)
    assert "build_hash" in prov.non_informative
    assert "build_hash" not in prov.root_keys
    assert not prov.mixed
    assert "non-informative" in prov.format_report()


def test_enforce_raises_on_mixed_provenance_and_the_override_permits(
    complete_root: Path,
) -> None:
    """Fail closed; --allow-mixed-provenance is the only way through."""
    victim = complete_root / "udfs/nguyen/nguyen_1/baseline/seed_00/run_log.json"
    payload = json.loads(victim.read_text())
    payload["metadata"]["hardware"]["git_describe"] = "c0ffee1-other-campaign"
    victim.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CampaignIntegrityError, match="provenance"):
        enforce_integrity(complete_root, METHODS, BENCHMARKS, VARIANTS)

    _, prov = enforce_integrity(
        complete_root, METHODS, BENCHMARKS, VARIANTS, allow_mixed_provenance=True
    )
    assert prov.mixed


def test_all_root_keys_are_checked(complete_root: Path) -> None:
    """Every declared root key participates, so none is silently unenforced."""
    _, prov = scan_root(complete_root, METHODS, BENCHMARKS, VARIANTS)
    assert set(prov.root_keys) == set(ROOT_PROVENANCE_KEYS)


def test_reports_serialise(complete_root: Path) -> None:
    """Both reports round-trip to JSON for the certification artefact."""
    comp, prov = scan_root(complete_root, METHODS, BENCHMARKS, VARIANTS)
    assert json.loads(json.dumps(comp.to_dict()))["complete"] is True
    assert json.loads(json.dumps(prov.to_dict()))["mixed"] is False
