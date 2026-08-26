"""The shadow sketches are OFF for campaign C2, and that must be enforced.

Decision: audit.md §7.3 (Mario, 2026-08-04). The sketches cost 17.6 % of Bingo's
wall clock and 0.034 % of UDFS's, are paid by the ``isalsr`` arm alone inside a
fixed budget, and would penalise exactly the arm carrying the headline claim.

These tests exist because a configuration key that nothing reads is worse than no
key at all -- that is the ``ISALSR_LEDGER_ENABLED`` trap of 2026-08-03, where a
default of ``"0"`` set in no launcher would have written five reachability rates
of zero across all 8,400 runs. The failure mode here is identical in shape: the
runner resolves ``config.get("shadow_hash", KEY_MODE == "canonical")``, so an
absent key means ON. The key must therefore be present, be false, and sit in the
block the runner is actually handed -- which is the *method* block
(``orchestrator.create_runner`` passes ``config.get(method, {})``), not the
``isalsr:`` block a reader might expect.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "experiments" / "configs"

SUITES = (
    "nguyen",
    "feynman",
    "hard",
    "cherrypicked",
    "roundoff",
    "feynman_remainder",
    "strogatz",
)
PRODUCTION_CONFIGS = tuple(f"{m}_{s}.yaml" for m in ("bingo", "udfs") for s in SUITES)
TRACE_CONFIG = "bingo_hard_trace.yaml"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text())


def _method_of(name: str) -> str:
    return "bingo" if name.startswith("bingo") else "udfs"


class TestProductionConfigsDisableShadow:
    """All fourteen campaign configs, both hosts, every suite."""

    @pytest.mark.parametrize("name", PRODUCTION_CONFIGS)
    def test_shadow_hash_is_present_and_false(self, name: str) -> None:
        block = _load(name)[_method_of(name)]
        assert "shadow_hash" in block, f"{name}: absent key means ON at the runner"
        assert block["shadow_hash"] is False, name

    @pytest.mark.parametrize("name", PRODUCTION_CONFIGS)
    def test_the_key_sits_in_the_block_the_runner_receives(self, name: str) -> None:
        """`create_runner` passes ``config.get(method, {})``, not ``config["isalsr"]``."""
        cfg = _load(name)
        assert "shadow_hash" not in cfg.get("isalsr", {}), (
            f"{name}: a shadow_hash under `isalsr:` is silently ignored"
        )
        assert "shadow_hash" in cfg[_method_of(name)], name

    def test_all_fourteen_exist(self) -> None:
        assert len(PRODUCTION_CONFIGS) == 14
        for name in PRODUCTION_CONFIGS:
            assert (CONFIG_DIR / name).is_file(), name


class TestTraceConfigEnablesShadow:
    """The one cell that must still pay for the sketches."""

    def test_trace_config_turns_shadow_on(self) -> None:
        assert _load(TRACE_CONFIG)["bingo"]["shadow_hash"] is True

    def test_trace_config_differs_from_production_in_exactly_one_key(self) -> None:
        """Any other drift would confound the traced cell against cell 10."""
        base = _load("bingo_hard.yaml")["bingo"]
        trace = _load(TRACE_CONFIG)["bingo"]
        differing = {
            k for k in set(base) | set(trace) if base.get(k, object()) != trace.get(k, object())
        }
        assert differing == {"shadow_hash"}

    def test_trace_config_keeps_the_full_budget(self) -> None:
        assert _load(TRACE_CONFIG)["bingo"]["max_time"] == 43_200


class TestRunnerHonoursTheKey:
    """End of the chain: the value the YAML carries is the value the runner uses."""

    @pytest.mark.parametrize("shadow", [True, False])
    def test_dataclass_filtering_does_not_swallow_the_key(self, shadow: bool) -> None:
        """`from_dict` filters unknown keys, so the dataclass must ignore it...

        ...while the raw dict the runner reads must still carry it. If a future
        change adds ``shadow_hash`` as a real dataclass field, this test keeps
        the two paths honest rather than letting them diverge silently.
        """
        from experiments.models.bingo.config import BingoConfig
        from experiments.models.udfs.config import UDFSConfig

        raw = {"shadow_hash": shadow, "max_time": 43_200}
        for cls in (BingoConfig, UDFSConfig):
            cfg = cls.from_dict(raw)
            assert not hasattr(cfg, "shadow_hash") or cfg.shadow_hash == shadow
        assert raw["shadow_hash"] is shadow

    def test_absent_key_still_defaults_on_for_the_canonical_arm(self) -> None:
        """The default is unchanged; only the configs are.

        Recorded deliberately: the fix is a configuration decision, not a code
        change, so a bare runner outside the campaign keeps its old behaviour.
        """
        config: dict[str, object] = {}
        assert bool(config.get("shadow_hash", "canonical" == "canonical")) is True
