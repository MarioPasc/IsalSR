"""The black-box configs run the C2 hosts unchanged (R3.2, PLAN D7).

The reviewer-facing claim is that the supplementary experiment reuses the main
campaign's hosts, arms and budget unchanged. These tests make the claim a
property of the repository: the only permitted differences from the C2 configs
are the ``benchmarks`` block and ``experiment.n_seeds``.
"""

from __future__ import annotations

import dataclasses
import glob
from pathlib import Path

import pytest
import yaml

from experiments.models.bingo.config import BingoConfig
from experiments.models.udfs.config import UDFSConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "experiments" / "configs"
BBX_DIR = CONFIG_DIR / "blackbox"
C2_SUITES = (
    "nguyen",
    "feynman",
    "hard",
    "cherrypicked",
    "roundoff",
    "feynman_remainder",
    "strogatz",
)
#: The C2 config each black-box config was copied from (the D2/R3.1 tier).
REFERENCE_SUITE = "strogatz"
DATACLASS = {"udfs": UDFSConfig, "bingo": BingoConfig}
METHODS = ("udfs", "bingo")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _bbx(method: str) -> dict:
    return _load(BBX_DIR / f"{method}_srbench_blackbox.yaml")


def _c2(method: str, suite: str = REFERENCE_SUITE) -> dict:
    return _load(CONFIG_DIR / f"{method}_{suite}.yaml")


@pytest.mark.parametrize("method", METHODS)
class TestIdentityWithC2:
    def test_host_block_is_key_by_key_identical(self, method: str) -> None:
        bbx, ref = _bbx(method)[method], _c2(method)[method]
        assert sorted(bbx) == sorted(ref)
        for key in ref:
            assert bbx[key] == ref[key], f"{method}.{key}: {bbx[key]!r} != C2 {ref[key]!r}"

    def test_isalsr_block_is_identical(self, method: str) -> None:
        assert _bbx(method)["isalsr"] == _c2(method)["isalsr"]

    def test_only_benchmarks_and_seed_count_differ(self, method: str) -> None:
        bbx, ref = _bbx(method), _c2(method)
        assert set(bbx) == set(ref)
        assert bbx["experiment"]["method"] == ref["experiment"]["method"] == method
        assert bbx["experiment"]["n_seeds"] == 10
        assert set(bbx["benchmarks"]) == {"srbench_blackbox"}
        assert set(bbx["experiment"]) == set(ref["experiment"])

    def test_budget_keys_explicit(self, method: str) -> None:
        """Lesson F-19: the binding budget must be declared, never inherited."""
        block = _bbx(method)[method]
        assert block["max_time"] == 43_200
        assert block["shadow_hash"] is False
        if method == "bingo":
            assert block["max_evals"] == 100_000_000
        else:
            assert block["processes"] == 1  # spawn workers would bypass the dedup patch

    def test_effective_dataclass_equals_reference(self, method: str) -> None:
        cls = DATACLASS[method]
        bbx = dataclasses.asdict(cls.from_dict(_bbx(method)[method]))
        ref = dataclasses.asdict(cls.from_dict(_c2(method)[method]))
        assert bbx == ref

    def test_effective_dataclass_agrees_with_every_c2_suite_where_c2_agrees(
        self, method: str
    ) -> None:
        """Every field on which the 7 C2 configs agree takes that common value here."""
        cls = DATACLASS[method]
        c2 = [dataclasses.asdict(cls.from_dict(_c2(method, s)[method])) for s in C2_SUITES]
        bbx = dataclasses.asdict(cls.from_dict(_bbx(method)[method]))
        for key, value in bbx.items():
            c2_values = {repr(cfg[key]) for cfg in c2}
            if len(c2_values) == 1:
                assert repr(value) in c2_values, key
            else:
                assert repr(value) in c2_values, f"{key} matches no C2 suite"


def test_c2_globs_do_not_reach_the_blackbox_configs() -> None:
    """The C2 uniformity tests glob configs/*.yaml non-recursively."""
    flat = {Path(p).name for p in glob.glob(str(CONFIG_DIR / "*_*.yaml"))}
    assert "bingo_srbench_blackbox.yaml" not in flat
    assert "udfs_srbench_blackbox.yaml" not in flat
    assert (BBX_DIR / "bingo_srbench_blackbox.yaml").is_file()
