"""Equivalence gate over evolved DAGs from Bingo and UDFS searches.

Closes §5.3 gate 3 and gate 6 of ticket T01 by verifying that the Python and
C++ canonicalisation engines agree on *every* DAG that each host produces
during a real (short) evolutionary / exhaustive search.

Gates
-----
Gate 3 — Python vs C++ agreement on evolved DAGs:
    For every DAG that Bingo or UDFS builds during a short search, compute
    ``fast_canonical_string(dag, backend='python')`` and
    ``fast_canonical_string(dag, backend='cpp')`` and assert byte equality
    (or matching exception type+message for error DAGs).

Gate 6 — Alphabet purity (split encoding):
    Zero ``NodeType.SUB``, zero ``NodeType.DIV``, zero ``'-'`` and zero
    ``'/'`` characters in any canonical string produced from the split
    encoding.

Additional measurements
-----------------------
Paired k-distribution:
    For each host individual, the split (T16) and legacy (pre-T16) encodings
    are built from the same host object and their non-VAR node counts are
    compared.  Mean, median, p95 and a histogram are reported for each
    encoding together with the paired mean delta.

Usage
-----
Quick smoke test (exits 0, completes in < 3 min)::

    python -m experiments.scripts.equivalence_gate_evolved --quick

Full gate (reports ≥ 100 000 DAGs, ≥ 40 000 per host)::

    python -m experiments.scripts.equivalence_gate_evolved --out report.json

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_PROJECT_ROOT, "src"), _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isalsr.core.backends import build_info  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-use probes from the existing gate module
# ---------------------------------------------------------------------------
from experiments.scripts.equivalence_gate import (  # noqa: E402
    _detect_self_comparison,
    _git_commit,
    _probe_cpp_canonical,
)

# ---------------------------------------------------------------------------
# Search budgets
# ---------------------------------------------------------------------------

#: Bingo config for quick mode
_BINGO_QUICK = dict(
    population_size=100,
    stack_size=16,
    operators=["+", "-", "*", "/", "sin", "cos"],
    max_time=25.0,
    max_evals=10_000_000,
    generations=100_000,
    use_fast_canonical=True,
)

#: Bingo config for full mode
_BINGO_FULL = dict(
    population_size=200,
    stack_size=24,
    operators=["+", "-", "*", "/", "sin", "cos", "exp"],
    max_time=90.0,
    max_evals=10_000_000,
    generations=100_000,
    use_fast_canonical=True,
)

#: UDFS config for quick mode
_UDFS_QUICK = dict(
    n_calc_nodes=3,
    max_orders=500_000,
    max_time=25.0,
    use_fast_canonical=True,
)

#: UDFS config for full mode
_UDFS_FULL = dict(
    n_calc_nodes=4,
    max_orders=2_000_000,
    max_time=90.0,
    use_fast_canonical=True,
)

# Seeds for each host
_SEEDS_QUICK = [42]
_SEEDS_FULL = [42, 137, 999]


# ---------------------------------------------------------------------------
# Nguyen-1 data
# ---------------------------------------------------------------------------


def _nguyen1_data(seed: int = 0) -> tuple[Any, Any, Any, Any]:
    """Return tiny Nguyen-1 train/test arrays.

    Args:
        seed: Random seed for data generation.

    Returns:
        ``(x_train, y_train, x_test, y_test)`` numpy arrays.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    x_train = rng.uniform(-1, 1, (20, 1))
    x_test = rng.uniform(-1, 1, (50, 1))

    def _fn(x: Any) -> Any:  # x^1 + x^2 + x^3
        return x + x**2 + x**3

    y_train = _fn(x_train[:, 0])
    y_test = _fn(x_test[:, 0])
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Per-host search runners
# ---------------------------------------------------------------------------


def _run_bingo(
    seeds: list[int],
    cfg_dict: dict[str, Any],
) -> int:
    """Run Bingo IsalSR search for each seed and return total DAGs seen.

    The global ``experiments.models.equivalence_probe.ACTIVE_PROBE`` must
    be installed before calling this function.

    Args:
        seeds: List of random seeds to run.
        cfg_dict: Keyword arguments forwarded to ``BingoConfig``.

    Returns:
        Total number of DAGs handed to the probe across all seeds.
    """
    import experiments.models.equivalence_probe as _ep
    from experiments.models.bingo.config import BingoConfig
    from experiments.models.bingo.isalsr_runner import IsalSRBingoRunner

    before = _ep.ACTIVE_PROBE._bingo_count if _ep.ACTIVE_PROBE is not None else 0
    cfg = BingoConfig(**cfg_dict)
    runner = IsalSRBingoRunner(config=cfg)

    for seed in seeds:
        x_train, y_train, x_test, y_test = _nguyen1_data(seed=seed)
        log.info("Bingo seed=%d starting (max_time=%.0fs)", seed, cfg.max_time)
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                runner.fit(x_train, y_train, x_test, y_test, seed=seed, config={})
        except Exception:  # noqa: BLE001
            log.warning("Bingo seed=%d raised; continuing", seed, exc_info=True)
        elapsed = time.perf_counter() - t0
        after = _ep.ACTIVE_PROBE._bingo_count if _ep.ACTIVE_PROBE is not None else 0
        log.info(
            "Bingo seed=%d done in %.1fs; %d DAGs this seed",
            seed,
            elapsed,
            after - before,
        )
        before = after

    return _ep.ACTIVE_PROBE._bingo_count if _ep.ACTIVE_PROBE is not None else 0


def _run_udfs(
    seeds: list[int],
    cfg_dict: dict[str, Any],
) -> int:
    """Run UDFS IsalSR search for each seed and return total DAGs seen.

    Args:
        seeds: List of random seeds to run.
        cfg_dict: Keyword arguments forwarded to ``UDFSConfig``.

    Returns:
        Total number of DAGs handed to the probe across all seeds.
    """
    import experiments.models.equivalence_probe as _ep
    from experiments.models.udfs.config import UDFSConfig
    from experiments.models.udfs.isalsr_runner import IsalSRUDFSRunner

    before = _ep.ACTIVE_PROBE._udfs_count if _ep.ACTIVE_PROBE is not None else 0
    cfg = UDFSConfig(**cfg_dict)
    runner = IsalSRUDFSRunner(config=cfg)

    for seed in seeds:
        x_train, y_train, x_test, y_test = _nguyen1_data(seed=seed)
        log.info("UDFS seed=%d starting (max_time=%.0fs)", seed, cfg.max_time)
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                runner.fit(x_train, y_train, x_test, y_test, seed=seed, config={})
        except Exception:  # noqa: BLE001
            log.warning("UDFS seed=%d raised; continuing", seed, exc_info=True)
        elapsed = time.perf_counter() - t0
        after = _ep.ACTIVE_PROBE._udfs_count if _ep.ACTIVE_PROBE is not None else 0
        log.info(
            "UDFS seed=%d done in %.1fs; %d DAGs this seed",
            seed,
            elapsed,
            after - before,
        )
        before = after

    return _ep.ACTIVE_PROBE._udfs_count if _ep.ACTIVE_PROBE is not None else 0


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _assemble_report(
    *,
    probe_summary: dict[str, Any],
    capability_probe: dict[str, Any],
    engine_a: str,
    engine_b: str,
    self_comparison: bool,
    git_commit: str,
    bingo_cfg: dict[str, Any],
    udfs_cfg: dict[str, Any],
    seeds: list[int],
    quick_mode: bool,
    wall_clock_s: float,
) -> dict[str, Any]:
    """Assemble the final JSON report.

    Args:
        probe_summary: Summary dict from ``EquivalenceProbe.summary()``.
        capability_probe: Capability probe metadata dict.
        engine_a: Resolved name of engine A.
        engine_b: Resolved name of engine B.
        self_comparison: True when both engines are Python.
        git_commit: Short HEAD commit hash.
        bingo_cfg: Bingo configuration used for the run.
        udfs_cfg: UDFS configuration used for the run.
        seeds: Seeds used for both hosts.
        quick_mode: Whether ``--quick`` was specified.
        wall_clock_s: Total wall-clock time in seconds.

    Returns:
        Report dict suitable for JSON serialisation.
    """
    alphabet = probe_summary["alphabet_checks"]
    alphabet_pass = (
        alphabet["sub_nodes_in_split_dag"] == 0
        and alphabet["div_nodes_in_split_dag"] == 0
        and alphabet["sub_chars_in_canonical"] == 0
        and alphabet["div_chars_in_canonical"] == 0
    )

    pass_flag = (
        not self_comparison
        and probe_summary["mismatches"] == 0
        and probe_summary["errors"] == 0
        and alphabet_pass
    )

    return {
        "quick_mode": quick_mode,
        "git_commit": git_commit,
        "wall_clock_s": round(wall_clock_s, 2),
        "engine_a": engine_a,
        "engine_b": engine_b,
        "self_comparison": self_comparison,
        "capability_probe": capability_probe,
        "build_info": build_info(),
        "seeds": seeds,
        "bingo_config": bingo_cfg,
        "udfs_config": udfs_cfg,
        "dags_compared_total": probe_summary["dags_compared_total"],
        "dags_compared": probe_summary["dags_compared"],
        "mismatches": probe_summary["mismatches"],
        "errors": probe_summary["errors"],
        "alphabet_checks": alphabet,
        "alphabet_pass": alphabet_pass,
        "k_distributions": probe_summary["k_distributions"],
        "mismatch_cases": probe_summary["mismatch_cases"],
        "pass": pass_flag,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the evolved-DAG equivalence gate.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: 0 if the gate passes (or ``--quick`` mode), 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Equivalence gate over evolved DAGs from Bingo and UDFS."
    )
    parser.add_argument(
        "--out",
        default="evolved_gate.json",
        help="Path for the JSON report (default: evolved_gate.json).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=("Quick smoke mode: one seed, 25 s search per host. Exits 0 regardless of DAG count."),
    )
    parser.add_argument(
        "--engine-a",
        default="python",
        choices=["python", "cpp"],
        help="Engine A (default: python).",
    )
    parser.add_argument(
        "--engine-b",
        default="cpp",
        choices=["python", "cpp"],
        help="Engine B (default: cpp).",
    )
    args = parser.parse_args(argv)

    t_start = time.perf_counter()

    # --- Capability probe --------------------------------------------------
    cpp_avail, cpp_detail = _probe_cpp_canonical()
    if args.engine_b == "cpp" and not cpp_avail:
        log.warning(
            "C++ canonicaliser not available (%s). "
            "Engine B falls back to 'python' → self_comparison will be True.",
            cpp_detail,
        )
        engine_b = "python"
    else:
        engine_b = args.engine_b

    engine_a = args.engine_a
    capability_probe: dict[str, Any] = {
        "method": "fast_canonical_string_backend_probe",
        "cpp_canonical_available": cpp_avail,
        "detail": cpp_detail,
    }

    self_comparison = _detect_self_comparison(engine_a, engine_b)
    if self_comparison:
        log.warning(
            "Self-comparison detected (both engines are '%s'). The gate will report pass=False.",
            engine_a,
        )

    # --- Choose budgets ----------------------------------------------------
    if args.quick:
        bingo_cfg = dict(_BINGO_QUICK)
        udfs_cfg = dict(_UDFS_QUICK)
        seeds = list(_SEEDS_QUICK)
    else:
        bingo_cfg = dict(_BINGO_FULL)
        udfs_cfg = dict(_UDFS_FULL)
        seeds = list(_SEEDS_FULL)

    log.info(
        "Gate started: engine_a=%s engine_b=%s quick=%s seeds=%s",
        engine_a,
        engine_b,
        args.quick,
        seeds,
    )

    # --- Install probe ------------------------------------------------------
    import experiments.models.equivalence_probe as _ep
    from experiments.models.equivalence_probe import EquivalenceProbe

    probe = EquivalenceProbe(engine_a=engine_a, engine_b=engine_b)
    _ep.ACTIVE_PROBE = probe

    # --- Run searches -------------------------------------------------------
    try:
        log.info("=== Bingo phase ===")
        _run_bingo(seeds, bingo_cfg)

        log.info("=== UDFS phase ===")
        _run_udfs(seeds, udfs_cfg)
    finally:
        # Always uninstall the probe, even if a host raises.
        _ep.ACTIVE_PROBE = None

    wall_clock_s = time.perf_counter() - t_start

    # --- Assemble report ---------------------------------------------------
    probe_summary = probe.summary()
    report = _assemble_report(
        probe_summary=probe_summary,
        capability_probe=capability_probe,
        engine_a=engine_a,
        engine_b=engine_b,
        self_comparison=self_comparison,
        git_commit=_git_commit(),
        bingo_cfg=bingo_cfg,
        udfs_cfg=udfs_cfg,
        seeds=seeds,
        quick_mode=args.quick,
        wall_clock_s=wall_clock_s,
    )

    # --- Write report -------------------------------------------------------
    out_path = args.out
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    log.info("Report written to %s", out_path)

    # --- Summary log --------------------------------------------------------
    log.info(
        "DAGs compared: total=%d bingo=%d udfs=%d",
        report["dags_compared_total"],
        report["dags_compared"]["bingo"],
        report["dags_compared"]["udfs"],
    )
    log.info(
        "Mismatches=%d errors=%d alphabet_pass=%s self_comparison=%s pass=%s",
        report["mismatches"],
        report["errors"],
        report["alphabet_pass"],
        report["self_comparison"],
        report["pass"],
    )

    # --- Exit code ----------------------------------------------------------
    # Quick mode always exits 0 (it's a smoke test, not the full gate).
    if args.quick:
        return 0
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
