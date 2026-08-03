"""Unit tests for the orchestrator's CLI overrides and the C2 array-index helper.

``--max-time`` / ``--no-shadow-hash`` exist so a ticket probe can cap a run's
budget (EXECUTION-PLAN §4.0 SP-0 caps probes at 1800 s) and can switch the shadow
fixed-order counters off, without forking the production YAML configs.  Both must
be inert when omitted.

``--postprocess`` exists because inside a SLURM array every task would otherwise
write the same ``aggregate.csv``/``paired_stats*.json`` and re-walk the whole
output tree to rebuild one shared ``status_ledger.csv``, concurrently.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pytest

from experiments.models.bingo.config import BingoConfig
from experiments.models.orchestrator import (
    apply_cli_overrides,
    build_parser,
    postprocess_output_root,
)
from experiments.models.udfs.config import UDFSConfig
from experiments.scripts import c2_task_spec


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
    from experiments.models.status_ledger import collect_status_ledger

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
    # Since EXECUTION-PLAN P4 the orchestrator does not let a cell's exception
    # escape: it records the failure in the status ledger and reports it through
    # the return code. §5.5 admits no third state, and the dominant real failure
    # (an OOM SIGKILL) reaches no handler at all, so propagating would buy
    # nothing and would let one bad cell destroy its siblings.
    assert run_experiment(args.config, args) == 1

    assert captured["max_time"] == pytest.approx(expected)

    # The spy's abort must have been recorded as a named cause, which is the
    # behaviour that replaces the old propagation.
    ledger = collect_status_ledger(tmp_path)
    assert len(ledger) == 1
    assert ledger[0].terminal_status == "failed"
    assert ledger[0].exception_class == _StopError.__name__
    assert ledger[0].arm == "isalsr"


# --------------------------------------------------------------------------- #
# --postprocess {auto,skip,only}
# --------------------------------------------------------------------------- #


def test_postprocess_defaults_to_auto() -> None:
    args = build_parser().parse_args(["--config", "c.yaml"])
    assert args.postprocess == "auto"


@pytest.mark.parametrize("mode", ["auto", "skip", "only"])
def test_postprocess_accepts_the_three_modes(mode: str) -> None:
    args = build_parser().parse_args(["--config", "c.yaml", "--postprocess", mode])
    assert args.postprocess == mode


def test_postprocess_rejects_unknown_mode() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--config", "c.yaml", "--postprocess", "sometimes"])


def _postprocess_args(tmp_path: Path, mode: str) -> argparse.Namespace:
    """Build a namespace that runs one Nguyen-1 cell under the given mode."""
    return argparse.Namespace(
        config="experiments/configs/udfs_nguyen.yaml",
        output_dir=str(tmp_path),
        seeds="1",
        problems="Nguyen-1",
        variants="isalsr",
        atlas_dir=None,
        max_time=1.0,
        no_shadow_hash=False,
        postprocess=mode,
    )


@pytest.mark.parametrize(
    ("mode", "expect_called"),
    [("auto", True), ("skip", False)],
)
def test_skip_suppresses_the_post_run_block_and_auto_keeps_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expect_called: bool,
) -> None:
    """``skip`` must not aggregate, compare or collect the ledger; ``auto`` must.

    The runner is replaced by a raiser so no search happens: the cell is recorded
    as failed (return code 1, EXECUTION-PLAN P4) and the dispatch under test is
    still reached.
    """
    from experiments.models import orchestrator

    calls: list[tuple[Any, ...]] = []

    def _spy(*a: Any, **kw: Any) -> dict[str, int]:
        calls.append(a)
        return {"aggregates": 0, "paired_stats": 0, "ledger_rows": 0}

    def _no_runner(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("no search in this test")

    monkeypatch.setattr(orchestrator, "postprocess_output_root", _spy)
    monkeypatch.setattr(orchestrator, "create_runner", _no_runner)

    args = _postprocess_args(tmp_path, mode)
    assert orchestrator.run_experiment(args.config, args) == 1  # the raiser's cell
    assert (len(calls) == 1) is expect_called

    # The cell's own artefacts are written in both modes.
    assert (tmp_path / "udfs/nguyen/nguyen_1/isalsr/seed_01/status.json").exists()
    if not expect_called:
        assert not (tmp_path / "status_ledger.csv").exists()
        assert list(tmp_path.rglob("aggregate.csv")) == []
        assert list(tmp_path.rglob("paired_stats*.json")) == []


def test_only_runs_no_search_and_generates_no_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``only`` must not construct a runner nor generate benchmark data."""
    from experiments.models import orchestrator

    def _boom(*a: Any, **kw: Any) -> Any:
        raise AssertionError("must not be reached in --postprocess only")

    monkeypatch.setattr(orchestrator, "create_runner", _boom)
    monkeypatch.setattr(orchestrator, "_generate_benchmark_data", _boom)

    args = _postprocess_args(tmp_path, "only")
    assert orchestrator.run_experiment(args.config, args) == 0
    # The ledger is collected even over an empty root, so a campaign that lost
    # every cell still produces the artefact that says so.
    assert (tmp_path / "status_ledger.csv").exists()


def test_postprocess_only_reports_failure_through_the_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.models import orchestrator

    def _boom(*a: Any, **kw: Any) -> dict[str, int]:
        raise OSError("gpfs said no")

    monkeypatch.setattr(orchestrator, "postprocess_output_root", _boom)
    args = _postprocess_args(tmp_path, "only")
    assert orchestrator.run_experiment(args.config, args) == 1


def test_postprocess_discovery_creates_nothing_for_unrun_problems(tmp_path: Path) -> None:
    """Discovery mode must not materialise a tree for a problem never run."""
    config = {
        "experiment": {"method": "udfs"},
        "benchmarks": {"nguyen": {"train_size": 20, "test_size": 100}},
    }
    counts = postprocess_output_root(tmp_path, "udfs", config)
    assert counts == {"aggregates": 0, "paired_stats": 0, "ledger_rows": 0}
    assert not (tmp_path / "udfs").exists()
    assert (tmp_path / "status_ledger.csv").exists()


# --------------------------------------------------------------------------- #
# experiments/scripts/c2_task_spec.py -- array index -> (problem, seed)
# --------------------------------------------------------------------------- #


def test_task_spec_count_is_problems_times_seeds() -> None:
    problems = c2_task_spec.load_problem_names("experiments/configs/udfs_strogatz.yaml")
    assert len(problems) == 14
    assert len(problems) * len(c2_task_spec.parse_seeds("0,101,102")) == 42


@pytest.mark.parametrize(
    ("config", "index", "expected"),
    [
        ("experiments/configs/udfs_strogatz.yaml", 1, ("Strogatz-bacres1", 0)),
        ("experiments/configs/udfs_strogatz.yaml", 3, ("Strogatz-bacres1", 102)),
        ("experiments/configs/udfs_strogatz.yaml", 4, ("Strogatz-bacres2", 0)),
        ("experiments/configs/udfs_strogatz.yaml", 42, ("Strogatz-vdp2", 102)),
        ("experiments/configs/bingo_nguyen.yaml", 4, ("Nguyen-2", 0)),
    ],
)
def test_task_spec_decode(config: str, index: int, expected: tuple[str, int]) -> None:
    problems = c2_task_spec.load_problem_names(config)
    seeds = c2_task_spec.parse_seeds("0,101,102")
    assert c2_task_spec.decode_index(problems, seeds, index) == expected


def test_task_spec_covers_every_cell_exactly_once() -> None:
    """The decode must be a bijection onto (problem, seed)."""
    problems = c2_task_spec.load_problem_names("experiments/configs/udfs_strogatz.yaml")
    seeds = c2_task_spec.parse_seeds("0,101,102")
    total = len(problems) * len(seeds)
    decoded = [c2_task_spec.decode_index(problems, seeds, i) for i in range(1, total + 1)]
    assert len(set(decoded)) == total
    assert set(decoded) == {(p, s) for p in problems for s in seeds}


@pytest.mark.parametrize("index", [0, -1, 43, 1000])
def test_task_spec_rejects_out_of_range_index(index: int) -> None:
    problems = c2_task_spec.load_problem_names("experiments/configs/udfs_strogatz.yaml")
    seeds = c2_task_spec.parse_seeds("0,101,102")
    with pytest.raises(c2_task_spec.TaskSpecError):
        c2_task_spec.decode_index(problems, seeds, index)


def test_task_spec_seed_order_follows_the_command_line() -> None:
    problems = ["A", "B"]
    assert c2_task_spec.decode_index(problems, [102, 0, 101], 1) == ("A", 102)
    assert c2_task_spec.decode_index(problems, [102, 0, 101], 2) == ("A", 0)


@pytest.mark.parametrize(
    "config_name",
    [
        "udfs_strogatz",
        "udfs_nguyen",
        "udfs_feynman",
        "udfs_feynman_remainder",
        "bingo_nguyen",
    ],
)
def test_every_c2_config_declares_exactly_one_suite(config_name: str) -> None:
    names = c2_task_spec.load_problem_names(f"experiments/configs/{config_name}.yaml")
    assert names and all(isinstance(n, str) and n for n in names)


def test_task_spec_rejects_a_multi_suite_config(tmp_path: Path) -> None:
    import yaml

    path = tmp_path / "two.yaml"
    path.write_text(
        yaml.safe_dump(
            {"experiment": {"method": "udfs"}, "benchmarks": {"nguyen": {}, "feynman": {}}}
        )
    )
    with pytest.raises(c2_task_spec.TaskSpecError):
        c2_task_spec.load_problem_names(str(path))


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            [
                "--config",
                "experiments/configs/udfs_strogatz.yaml",
                "--seeds",
                "0,101,102",
                "--count",
            ],
            "42",
        ),
        (
            [
                "--config",
                "experiments/configs/udfs_strogatz.yaml",
                "--seeds",
                "0,101,102",
                "--index",
                "1",
            ],
            "Strogatz-bacres1 0",
        ),
        (
            [
                "--config",
                "experiments/configs/bingo_nguyen.yaml",
                "--seeds",
                "0,101,102",
                "--index",
                "4",
            ],
            "Nguyen-2 0",
        ),
    ],
)
def test_task_spec_cli_prints_one_line(
    argv: list[str],
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert c2_task_spec.main(argv) == 0
    out = capsys.readouterr().out
    assert out.splitlines() == [expected]


def test_task_spec_cli_prints_nothing_to_stdout_on_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = c2_task_spec.main(
        [
            "--config",
            "experiments/configs/udfs_strogatz.yaml",
            "--seeds",
            "0,101,102",
            "--index",
            "43",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "out of range" in captured.err
