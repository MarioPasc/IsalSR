"""Unit tests for the orchestrator's ``--max-time`` / ``--no-shadow-hash`` CLI overrides.

The overrides exist so a ticket probe can cap a run's budget (EXECUTION-PLAN §4.0
SP-0 caps probes at 1800 s) and can switch the shadow fixed-order counters off,
without forking the production YAML configs.  Both must be inert when omitted.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from experiments.models.bingo.config import BingoConfig
from experiments.models.orchestrator import apply_cli_overrides, build_parser
from experiments.models.udfs.config import UDFSConfig


def _config(method: str) -> dict[str, Any]:
    """Build a minimal two-section config mirroring the production YAML shape."""
    return {
        "experiment": {"method": method, "n_seeds": 30},
        method: {
            "max_time": 43200,
            "population_size": 500,
            "n_calc_nodes": 5,
        },
        "benchmarks": {"nguyen": {"train_size": 20, "test_size": 100}},
    }


# --------------------------------------------------------------------------- #
# Parser surface
# --------------------------------------------------------------------------- #


def test_parser_defaults_preserve_current_behaviour() -> None:
    """Omitting both flags must leave ``max_time`` unset and shadow counting on."""
    args = build_parser().parse_args(["--config", "c.yaml"])
    assert args.max_time is None
    assert args.no_shadow_hash is False


def test_parser_accepts_max_time() -> None:
    args = build_parser().parse_args(["--config", "c.yaml", "--max-time", "60"])
    assert args.max_time == pytest.approx(60.0)
    assert isinstance(args.max_time, float)


def test_parser_accepts_no_shadow_hash() -> None:
    args = build_parser().parse_args(["--config", "c.yaml", "--no-shadow-hash"])
    assert args.no_shadow_hash is True


def test_parser_rejects_nonpositive_max_time() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--config", "c.yaml", "--max-time", "0"])


def test_existing_arguments_still_present() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "c.yaml",
            "--output-dir",
            "/tmp/o",
            "--seeds",
            "1",
            "--problems",
            "Nguyen-1",
            "--variants",
            "isalsr",
            "--atlas-dir",
            "/tmp/a",
        ]
    )
    assert (args.output_dir, args.seeds, args.problems, args.variants, args.atlas_dir) == (
        "/tmp/o",
        "1",
        "Nguyen-1",
        "isalsr",
        "/tmp/a",
    )


# --------------------------------------------------------------------------- #
# Override semantics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["bingo", "udfs"])
def test_no_overrides_is_identity(method: str) -> None:
    cfg = _config(method)
    before = copy.deepcopy(cfg)
    out = apply_cli_overrides(cfg, max_time=None, no_shadow_hash=False)
    assert out == before
    assert cfg == before  # the caller's dict is not mutated


@pytest.mark.parametrize("method", ["bingo", "udfs"])
def test_max_time_override_reaches_host_config(method: str) -> None:
    """The override must land in the section the runner rebuilds its config from."""
    cfg = _config(method)
    out = apply_cli_overrides(cfg, max_time=60.0, no_shadow_hash=False)
    assert out[method]["max_time"] == pytest.approx(60.0)

    host_cfg = (
        BingoConfig.from_dict(out[method])
        if method == "bingo"
        else UDFSConfig.from_dict(out[method])
    )
    # ``BingoBaselineRunner.fit`` passes ``cfg.max_time`` to
    # ``evolve_until_convergence``; ``UDFSConfig.to_dag_regressor_kwargs``
    # passes it to ``DAGRegressor``.
    assert host_cfg.max_time == pytest.approx(60.0)


@pytest.mark.parametrize("method", ["bingo", "udfs"])
def test_max_time_override_leaves_other_keys_untouched(method: str) -> None:
    cfg = _config(method)
    out = apply_cli_overrides(cfg, max_time=60.0, no_shadow_hash=False)
    assert out[method]["population_size"] == 500
    assert out[method]["n_calc_nodes"] == 5
    assert out["experiment"] == cfg["experiment"]
    assert out["benchmarks"] == cfg["benchmarks"]
    assert cfg[method]["max_time"] == 43200  # original untouched


@pytest.mark.parametrize("method", ["bingo", "udfs"])
def test_no_shadow_hash_sets_the_key_the_runner_reads(method: str) -> None:
    """Runners read ``config.get("shadow_hash", KEY_MODE == "canonical")``."""
    cfg = _config(method)
    out = apply_cli_overrides(cfg, max_time=None, no_shadow_hash=True)
    assert out[method]["shadow_hash"] is False
    key_mode = "canonical"  # IsalSR arm
    assert bool(out[method].get("shadow_hash", key_mode == "canonical")) is False


@pytest.mark.parametrize("method", ["bingo", "udfs"])
def test_shadow_hash_on_by_default_for_canonical_arm(method: str) -> None:
    cfg = _config(method)
    out = apply_cli_overrides(cfg, max_time=None, no_shadow_hash=False)
    assert "shadow_hash" not in out[method]
    assert bool(out[method].get("shadow_hash", "canonical" == "canonical")) is True


@pytest.mark.parametrize("method", ["bingo", "udfs"])
def test_both_overrides_compose(method: str) -> None:
    cfg = _config(method)
    out = apply_cli_overrides(cfg, max_time=1800.0, no_shadow_hash=True)
    assert out[method]["max_time"] == pytest.approx(1800.0)
    assert out[method]["shadow_hash"] is False


def test_override_creates_missing_method_section() -> None:
    """A config with no method section still receives the override."""
    cfg: dict[str, Any] = {"experiment": {"method": "bingo"}}
    out = apply_cli_overrides(cfg, max_time=60.0, no_shadow_hash=True)
    assert out["bingo"]["max_time"] == pytest.approx(60.0)
    assert out["bingo"]["shadow_hash"] is False


def test_override_requires_a_method() -> None:
    with pytest.raises(KeyError):
        apply_cli_overrides({"experiment": {}}, max_time=60.0, no_shadow_hash=False)


# --------------------------------------------------------------------------- #
# End-to-end: the override must reach the host search's own budget
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("cli_max_time", "expected"),
    [(None, 43200.0), (60.0, 60.0)],
)
def test_max_time_reaches_bingo_evolve_until_convergence(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    cli_max_time: float | None,
    expected: float,
) -> None:
    """``--max-time`` must land in ``Island.evolve_until_convergence(max_time=...)``.

    A wrapper timer around the run would not bound Bingo, which otherwise runs
    to ``max_evals`` (1e8 in the production Nguyen config). The spy aborts the
    evolution as soon as the call is made, so no search is performed.
    """
    import argparse as _argparse

    pytest.importorskip("bingo")
    from bingo.evolutionary_optimizers.island import Island

    from experiments.models.orchestrator import run_experiment

    captured: dict[str, Any] = {}

    class _StopError(Exception):
        pass

    def _spy(self: Any, *a: Any, **kw: Any) -> None:
        captured.update(kw)
        raise _StopError

    monkeypatch.setattr(Island, "evolve_until_convergence", _spy)

    args = _argparse.Namespace(
        config="experiments/configs/bingo_nguyen.yaml",
        output_dir=str(tmp_path),
        seeds="1",
        problems="Nguyen-1",
        variants="isalsr",
        atlas_dir=None,
        max_time=cli_max_time,
        no_shadow_hash=False,
    )
    with pytest.raises(_StopError):
        run_experiment(args.config, args)

    assert captured["max_time"] == pytest.approx(expected)
