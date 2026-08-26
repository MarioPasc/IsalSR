"""Tests for the C2 campaign MANIFEST schema and validator (check A6).

A6 is graded on one behaviour above all: a deliberately truncated manifest must
make the validator exit non-zero. These tests pin that, the happy path, and the
writer/loader round-trip, plus the frozen campaign constants the manifest exists
to record.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.models.manifest import (  # noqa: E402
    ALPHABET_VERSION,
    ARMS,
    CAMPAIGN_TAG,
    EXPECTED_N_ARRAYS,
    NODE_CONSTRAINT,
    SEEDS,
    SUITE_SIZES,
    BuildProvenance,
    CampaignManifest,
    ConfigDigest,
    ManifestValidationError,
    OperatorSetPolicy,
    build_submission_splits,
    load_manifest,
    main,
    save_manifest,
    sha256_file,
    utc_now,
    validate_manifest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FAKE_SHA = "a" * 64


def _build() -> BuildProvenance:
    return BuildProvenance(
        git_commit="0" * 40,
        git_tag=CAMPAIGN_TAG,
        git_dirty=False,
        native_build_hash="deadbeefcafe",
        engine="native",
        compiler="GCC 12.2.0",
        compiler_flags="-O3 -march=x86-64-v3 -DNDEBUG",
        isa_level="x86-64-v3",
        avx512f=False,
        native_module_path="/x/site-packages/isalsr/core/_native.so",
        native_module_mtime="2026-08-04T10:00:00+00:00",
    )


def _policy() -> OperatorSetPolicy:
    return OperatorSetPolicy(
        policy="uniform_per_method",
        statement=(
            "A4b: the operator set is uniform per method across every problem. "
            "All seven bingo_*.yaml carry the same ten primitives on D1 and D2 alike."
        ),
        bingo_operators=["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "pow"],
        udfs_operators=["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt"],
        udfs_operator_source=(
            "vendored DAG_search NODE_ARITY table; to_dag_regressor_kwargs() never "
            "forwards the YAML field, so the UDFS set is not configurable"
        ),
        bingo_continuity_exclusion_problems=[f"D1-{i:02d}" for i in range(22)],
        corrected_definition_exclusion_problems=[
            "I.39.10",
            "I.12.4",
            "II.3.24",
            "II.11.27",
            "III.17.37",
            "I.34.27",
        ],
        continuity_note=(
            "Both exclusions are mandatory: the compared object changed, not the method."
        ),
    )


def _configs() -> list[ConfigDigest]:
    return [
        ConfigDigest(
            method=method,
            suite=suite,
            path=f"experiments/configs/{method}_{suite}.yaml",
            sha256=_FAKE_SHA,
        )
        for method in ("udfs", "bingo")
        for suite in SUITE_SIZES
    ]


def _manifest() -> CampaignManifest:
    return CampaignManifest(
        schema_version="c2.1",
        campaign="c2",
        campaign_root="/mnt/.../fscratch/results/isalsr/c2",
        created_utc=utc_now(),
        build=_build(),
        configs=_configs(),
        operator_set_policy=_policy(),
        arms=list(ARMS),
        seeds=list(SEEDS),
        alphabet_version=ALPHABET_VERSION,
        node_constraint=NODE_CONSTRAINT,
        submission_splits=build_submission_splits(),
        notes="",
    )


@pytest.fixture
def manifest() -> CampaignManifest:
    """A manifest that satisfies every A6 requirement."""
    return _manifest()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_manifest_passes(manifest: CampaignManifest) -> None:
    validate_manifest(manifest)


def test_submission_splits_match_section_11_3() -> None:
    splits = build_submission_splits()
    assert len(splits) == EXPECTED_N_ARRAYS == 42
    assert sum(s.n_tasks for s in splits) == 8400
    assert sum(SUITE_SIZES.values()) == 70
    # Per (method, arm) block: 70 problems x 20 seeds.
    per_block = [s for s in splits if s.method == "bingo" and s.arm == "isalsr"]
    assert sum(s.n_tasks for s in per_block) == 1400
    assert [s.index for s in splits] == list(range(1, 43))


def test_frozen_constants() -> None:
    assert ARMS == ("baseline", "hash", "isalsr")
    assert tuple(range(1, 21)) == SEEDS
    assert 0 not in SEEDS
    assert ALPHABET_VERSION == "decomposed"
    assert NODE_CONSTRAINT == "sr"


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_everything(manifest: CampaignManifest, tmp_path) -> None:
    path = tmp_path / "MANIFEST.json"
    save_manifest(manifest, path)
    reloaded = load_manifest(path)
    assert reloaded == manifest
    assert reloaded.to_dict() == manifest.to_dict()
    validate_manifest(reloaded)


def test_saved_manifest_is_valid_json(manifest: CampaignManifest, tmp_path) -> None:
    path = tmp_path / "MANIFEST.json"
    save_manifest(manifest, path)
    payload = json.loads(path.read_text())
    assert payload["arms"] == list(ARMS)
    assert payload["seeds"] == list(SEEDS)
    assert payload["alphabet_version"] == "decomposed"
    assert payload["node_constraint"] == "sr"
    assert len(payload["submission_splits"]) == 42
    assert not list(path.parent.glob("*.tmp")), "atomic write left a temp file behind"


# ---------------------------------------------------------------------------
# Truncation — the graded behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dropped",
    [
        "build",
        "configs",
        "operator_set_policy",
        "arms",
        "seeds",
        "alphabet_version",
        "node_constraint",
        "submission_splits",
        "campaign_root",
        "created_utc",
        "campaign",
        "schema_version",
        "notes",
    ],
)
def test_truncated_manifest_fails(manifest: CampaignManifest, tmp_path, dropped: str) -> None:
    """Removing ANY top-level field must make the manifest unloadable."""
    payload = manifest.to_dict()
    del payload[dropped]
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestValidationError) as exc:
        load_manifest(path)
    assert dropped in str(exc.value)


@pytest.mark.parametrize(
    "dropped",
    ["git_commit", "native_build_hash", "compiler", "compiler_flags", "engine", "isa_level"],
)
def test_truncated_build_block_fails(manifest: CampaignManifest, tmp_path, dropped: str) -> None:
    payload = manifest.to_dict()
    del payload["build"][dropped]
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestValidationError):
        load_manifest(path)


def test_malformed_json_fails(tmp_path) -> None:
    path = tmp_path / "MANIFEST.json"
    path.write_text('{"campaign": "c2", "arms": [')
    with pytest.raises(ManifestValidationError):
        load_manifest(path)


# ---------------------------------------------------------------------------
# Value validation
# ---------------------------------------------------------------------------


def _expect_failure(manifest: CampaignManifest, needle: str, **overrides) -> None:
    from dataclasses import replace

    bad = replace(manifest, **overrides)
    with pytest.raises(ManifestValidationError) as exc:
        validate_manifest(bad)
    assert needle in str(exc.value)


def test_seed_zero_rejected(manifest: CampaignManifest) -> None:
    _expect_failure(manifest, "seed 0", seeds=[0, *SEEDS])


def test_two_arm_manifest_rejected_for_campaign(manifest: CampaignManifest) -> None:
    _expect_failure(manifest, "arms", arms=["baseline", "isalsr"])


def test_wrong_alphabet_rejected(manifest: CampaignManifest) -> None:
    _expect_failure(manifest, "alphabet_version", alphabet_version="legacy")


def test_wrong_node_constraint_rejected(manifest: CampaignManifest) -> None:
    _expect_failure(manifest, "node_constraint", node_constraint="dgx")


def test_empty_campaign_root_rejected(manifest: CampaignManifest) -> None:
    _expect_failure(manifest, "campaign_root", campaign_root="   ")


def test_empty_configs_rejected(manifest: CampaignManifest) -> None:
    _expect_failure(manifest, "configs", configs=[])


def test_bad_sha256_rejected(manifest: CampaignManifest) -> None:
    bad = [ConfigDigest(method="bingo", suite="nguyen", path="p.yaml", sha256="nothex")]
    _expect_failure(manifest, "sha256", configs=bad)


def test_duplicate_config_entry_rejected(manifest: CampaignManifest) -> None:
    dup = [*_configs(), ConfigDigest("bingo", "nguyen", "x.yaml", _FAKE_SHA)]
    _expect_failure(manifest, "duplicate", configs=dup)


def test_missing_continuity_exclusion_rejected(manifest: CampaignManifest) -> None:
    """The 22-problem Bingo continuity exclusion must be enumerated, not implied."""
    from dataclasses import replace

    pol = replace(manifest.operator_set_policy, bingo_continuity_exclusion_problems=[])
    _expect_failure(manifest, "continuity", operator_set_policy=pol)


def test_dirty_tree_rejected(manifest: CampaignManifest) -> None:
    from dataclasses import replace

    _expect_failure(manifest, "git_dirty", build=replace(manifest.build, git_dirty=True))


def test_python_engine_rejected_for_campaign(manifest: CampaignManifest) -> None:
    from dataclasses import replace

    _expect_failure(manifest, "native engine", build=replace(manifest.build, engine="python"))


def test_wrong_tag_rejected(manifest: CampaignManifest) -> None:
    from dataclasses import replace

    _expect_failure(manifest, "git_tag", build=replace(manifest.build, git_tag="v0.1"))


def test_short_commit_rejected(manifest: CampaignManifest) -> None:
    from dataclasses import replace

    _expect_failure(manifest, "40-char", build=replace(manifest.build, git_commit="abc1234"))


def test_wrong_array_count_rejected(manifest: CampaignManifest) -> None:
    _expect_failure(manifest, "42 arrays", submission_splits=build_submission_splits()[:10])


def test_task_count_inconsistency_rejected(manifest: CampaignManifest) -> None:
    from dataclasses import replace

    splits = build_submission_splits()
    splits[0] = replace(splits[0], n_tasks=999)
    _expect_failure(manifest, "n_tasks", submission_splits=splits)


def test_non_campaign_mode_relaxes_frozen_constants(manifest: CampaignManifest) -> None:
    """A smoke manifest (seed 0 excluded, 2 arms, fewer arrays) validates loosely."""
    from dataclasses import replace

    smoke = replace(
        manifest,
        arms=["baseline", "isalsr"],
        seeds=[101, 102, 103],
        node_constraint="sr",
        build=replace(manifest.build, git_tag="probe", git_dirty=True),
        submission_splits=build_submission_splits(
            methods=("bingo",), arms=("baseline",), seeds=(101, 102, 103)
        ),
    )
    validate_manifest(smoke, strict_campaign=False)
    with pytest.raises(ManifestValidationError):
        validate_manifest(smoke, strict_campaign=True)


# ---------------------------------------------------------------------------
# CLI — A6 is graded on the exit status
# ---------------------------------------------------------------------------


def test_cli_returns_zero_on_valid(manifest: CampaignManifest, tmp_path) -> None:
    path = tmp_path / "MANIFEST.json"
    save_manifest(manifest, path)
    assert main(["validate", str(path)]) == 0


def test_cli_returns_nonzero_on_truncated(manifest: CampaignManifest, tmp_path) -> None:
    """The A6 pass criterion, verbatim."""
    payload = manifest.to_dict()
    del payload["operator_set_policy"]
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps(payload))
    assert main(["validate", str(path)]) == 1


def test_cli_returns_nonzero_on_missing_file(tmp_path) -> None:
    assert main(["validate", str(tmp_path / "absent.json")]) == 1


def test_cli_allow_non_campaign_flag(manifest: CampaignManifest, tmp_path) -> None:
    from dataclasses import replace

    smoke = replace(
        manifest,
        arms=["baseline", "isalsr"],
        seeds=[101],
        submission_splits=build_submission_splits(
            methods=("bingo",), arms=("baseline",), seeds=(101,)
        ),
    )
    path = tmp_path / "MANIFEST.json"
    save_manifest(smoke, path)
    assert main(["validate", str(path)]) == 1
    assert main(["validate", str(path), "--allow-non-campaign"]) == 0


# ---------------------------------------------------------------------------
# sha256 helper
# ---------------------------------------------------------------------------


def test_sha256_file(tmp_path) -> None:
    import hashlib

    path = tmp_path / "cfg.yaml"
    path.write_bytes(b"seed: 1\n")
    assert sha256_file(path) == hashlib.sha256(b"seed: 1\n").hexdigest()
    assert len(sha256_file(path)) == 64
