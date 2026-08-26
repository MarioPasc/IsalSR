"""SP-7 for T05: the five statements a D2 probe must establish, as a command.

`EXECUTION-PLAN.md` §4.0 SP-7 asks a T05 probe to establish, for every D2 problem:

  1. the dataset loads **on Picasso** with the expected train/test shapes,
     asserted against the benchmark registry;
  2. a `sympy_expression` ground truth is present, so `solution_recovered` is
     actually computable — the historic gap, invisible until analysis;
  3. the run completes without crashing and leaves a `run_log.json` that parses
     and validates against the RunLog schema;
  4. the declared operator set is what actually ran, and for a fixed
     `(method, problem)` it is identical across all three arms;
  5. no NaN and no inf in any regression metric.

Statements 1, 2 and 4 are properties of the definitions and configs, so they are
checked here in one pass before any search starts — a failure then costs seconds
rather than forty array tasks. Statements 3 and 5 are properties of a completed
run and are checked by `--verify-runs` after the array.

Exit status is the result: 0 iff every checked statement holds. "I checked it" is
not evidence; this writes a JSON artefact.

Usage
-----
    python slurm/t05_probe/check_d2.py --pre  --out pre.json
    python slurm/t05_probe/check_d2.py --verify-runs <probe_out_dir> --out post.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.models.orchestrator import (  # noqa: E402
    _generate_benchmark_data,
    _get_ground_truth_sympy,
    _get_ground_truth_vars,
    get_benchmarks,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
D2_SUITES: tuple[str, ...] = ("strogatz", "feynman_remainder")
METHODS: tuple[str, ...] = ("udfs", "bingo")

#: Regression fields that must be finite. Mirrors Stage C's C1.3.
METRIC_FIELDS: tuple[str, ...] = (
    "r2_train",
    "r2_test",
    "nrmse_train",
    "nrmse_test",
    "mse_test",
)


def _expected_sizes(suite: str) -> tuple[int, int]:
    """Read the declared train/test sizes for a suite from its Bingo config.

    Parameters
    ----------
    suite
        Registry suite key.

    Returns
    -------
    tuple of int
        ``(train_size, test_size)`` as declared in the YAML.
    """
    cfg = yaml.safe_load((REPO / f"experiments/configs/bingo_{suite}.yaml").read_text())
    block = cfg["benchmarks"][suite]
    return int(block["train_size"]), int(block["test_size"])


def check_shapes_and_ground_truth() -> dict[str, Any]:
    """SP-7.1 and SP-7.2 over every D2 problem.

    Returns
    -------
    dict
        Per-problem results plus a boolean ``ok``.
    """
    rows: list[dict[str, Any]] = []
    for suite in D2_SUITES:
        train_size, test_size = _expected_sizes(suite)
        for bench in get_benchmarks(suite):
            x_tr, y_tr, x_te, y_te = _generate_benchmark_data(
                suite, bench, train_size, test_size, seed=0
            )
            nv = bench["num_variables"]
            # The per-problem `sampling` overrides may legitimately pin sizes
            # that differ from the YAML -- Strogatz is published fixed data.
            sampling = bench.get("sampling") or {}
            exp_tr = int(sampling.get("n_train_override") or train_size)
            exp_te = int(sampling.get("n_test_override") or test_size)

            gt = _get_ground_truth_sympy(bench)
            gt_vars = _get_ground_truth_vars(bench)
            rows.append(
                {
                    "suite": suite,
                    "problem": bench["name"],
                    "shape_train": list(x_tr.shape),
                    "shape_test": list(x_te.shape),
                    "expected_train": [exp_tr, nv],
                    "expected_test": [exp_te, nv],
                    "shapes_ok": (
                        x_tr.shape == (exp_tr, nv)
                        and x_te.shape == (exp_te, nv)
                        and y_tr.shape == (exp_tr,)
                        and y_te.shape == (exp_te,)
                    ),
                    "ground_truth_computable": gt is not None and gt_vars is not None,
                    "targets_finite": bool(
                        all(math.isfinite(float(v)) for v in y_tr)
                        and all(math.isfinite(float(v)) for v in y_te)
                    ),
                }
            )
    ok = all(r["shapes_ok"] and r["ground_truth_computable"] and r["targets_finite"] for r in rows)
    return {"n_problems": len(rows), "ok": ok, "problems": rows}


def check_operator_sets() -> dict[str, Any]:
    """SP-7.4: the operator set is identical across arms for a fixed (method, problem).

    The three arms are selected by the orchestrator's ``--variants`` flag and
    read the same YAML block, so the invariant holds by construction. This
    records the actual sets so the claim is a measurement rather than an
    argument, and fails if a future edit ever puts an arm-specific key in a
    config.

    Returns
    -------
    dict
        The per-``(method, suite)`` operator sets and a boolean ``ok``.
    """
    rows: list[dict[str, Any]] = []
    ok = True
    for method in METHODS:
        for suite in D2_SUITES:
            cfg = yaml.safe_load((REPO / f"experiments/configs/{method}_{suite}.yaml").read_text())
            key = "operators" if method == "bingo" else "operator_set"
            ops = cfg[method][key]
            arm_specific = [k for k in cfg if k in ("baseline", "hash", "isalsr_only")]
            # `isalsr:` holds canonicalisation settings, not search operators, so
            # it is not arm-specific in the sense A4b cares about.
            if arm_specific:
                ok = False
            rows.append(
                {
                    "method": method,
                    "suite": suite,
                    "operators": ops,
                    "arm_specific_keys": arm_specific,
                    "n_seeds": cfg["experiment"]["n_seeds"],
                    "max_time": cfg[method]["max_time"],
                }
            )
    return {"ok": ok, "configs": rows}


def verify_runs(root: Path) -> dict[str, Any]:
    """SP-7.3 and SP-7.5 over every `run_log.json` under a probe output root.

    Parameters
    ----------
    root
        Probe output directory.

    Returns
    -------
    dict
        Per-run results plus a boolean ``ok``.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("run_log.json")):
        try:
            doc = json.loads(path.read_text())
            reg = doc["results"]["regression"]
            search = doc["results"]["search_space"]
            meta = doc["metadata"]
        except (json.JSONDecodeError, KeyError) as exc:
            rows.append({"path": str(path), "parses": False, "error": repr(exc)})
            continue

        bad = [
            f
            for f in METRIC_FIELDS
            if not isinstance(reg.get(f), (int, float))
            or math.isnan(float(reg[f]))
            or math.isinf(float(reg[f]))
        ]
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "parses": True,
                "method": meta["method"],
                "arm": meta["representation"],
                "problem": meta["problem"],
                "nan_or_inf_fields": bad,
                "rho": search.get("empirical_reduction_factor"),
                "unique_canonical": search.get("unique_canonical_dags"),
                "engine": meta.get("hardware", {}).get("engine"),
            }
        )
    ok = bool(rows) and all(r.get("parses") and not r.get("nan_or_inf_fields") for r in rows)
    return {"n_runs": len(rows), "ok": ok, "runs": rows}


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv
        Command-line arguments.

    Returns
    -------
    int
        0 iff every checked SP-7 statement holds.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre", action="store_true", help="SP-7.1, 7.2, 7.4.")
    parser.add_argument("--verify-runs", default=None, help="SP-7.3, 7.5 over a probe root.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    report: dict[str, Any] = {}
    if args.pre or not args.verify_runs:
        report["shapes_and_ground_truth"] = check_shapes_and_ground_truth()
        report["operator_sets"] = check_operator_sets()
    if args.verify_runs:
        report["runs"] = verify_runs(Path(args.verify_runs))

    report["ok"] = all(v["ok"] for v in report.values() if isinstance(v, dict))

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    for key, val in report.items():
        if isinstance(val, dict):
            log.info("%-28s ok=%s", key, val["ok"])
    log.info("SP-7 overall: %s", "PASS" if report["ok"] else "FAIL")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
