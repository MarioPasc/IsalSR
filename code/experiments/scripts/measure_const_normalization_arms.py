"""Compare three CONST-normalization policies on one identical DAG stream (T15).

Question
--------
``normalize_const_creation`` used to relocate every CONST in-edge onto node 0.
That could drop the edge (orphaning the CONST and making canonicalisation raise),
it merged non-isomorphic DAGs, and it was not evaluation-preserving. It has been
replaced by a minimal repair that only supplies a creation edge to CONST nodes
with in-degree 0. Three questions follow, and the paper needs all three answered:

1. How often did the old policy actually fail on the DAGs the search produces?
2. Does the new policy ever fail there?
3. Does either policy change the reduction factor rho = n_total / n_unique?

Design
------
Every arm is evaluated on the *same* DAG, at the moment the search hands it to
the canonicaliser, so the comparison is exactly paired: identical ``n_total``,
identical population, no confound from the arms steering the search differently.

  arm A  ``submitted``  old aggressive relocation, then the canonical core
  arm B  ``repair``     new minimal repair, then the canonical core  (production)
  arm C  ``none``       canonical core with no CONST handling at all

Arm B's result is what the search receives, so the trajectory is the current
production trajectory. The canonical core is reached through
``_native.testing.fast_canonical_string_raw``, which skips normalization, so all
three arms share one implementation and differ only in the policy applied first.

Usage
-----
    python -m experiments.scripts.measure_const_normalization_arms \\
        --methods bingo,udfs --seeds 1,2,3 --max-time 120 \\
        --out /media/mpascual/Sandisk2TB/research/isalsr/results/t15_norm_arms
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("t15.arms")

# One problem per benchmark tier, spanning k, n_vars and sampling protocol.
DEFAULT_PROBLEMS: list[tuple[str, str]] = [
    ("nguyen", "Nguyen-5"),
    ("feynman", "I.6.20a"),
    ("hard", "Pagie-1"),
    ("cherrypicked", "Keijzer-11"),
    ("roundoff", "R1"),
]

# Suite key -> (defining module, benchmark-list constant). The key and the module
# name coincide except for ``cherrypicked``, a legacy suite key kept because it is
# baked into the on-disk result tree.
_BENCH_SOURCE = {
    "nguyen": ("nguyen", "NGUYEN_BENCHMARKS"),
    "feynman": ("feynman", "FEYNMAN_BENCHMARKS"),
    "hard": ("hard", "HARD_BENCHMARKS"),
    "cherrypicked": ("structural", "STRUCTURAL_BENCHMARKS"),
    "roundoff": ("roundoff", "ROUNDOFF_BENCHMARKS"),
}

ARMS = ("submitted", "repair", "none")


@dataclass
class ArmStats:
    """Per-arm outcome over one run's DAG stream."""

    n_calls: int = 0
    n_failures: int = 0
    canonical_seen: set[int] = field(default_factory=set)

    @property
    def n_unique(self) -> int:
        return len(self.canonical_seen)

    @property
    def failure_rate(self) -> float:
        return self.n_failures / self.n_calls if self.n_calls else 0.0

    @property
    def reduction_factor(self) -> float:
        """rho = n_total / n_unique, counting only DAGs this arm could classify."""
        classified = self.n_calls - self.n_failures
        return classified / self.n_unique if self.n_unique else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_calls": self.n_calls,
            "n_failures": self.n_failures,
            "failure_rate": self.failure_rate,
            "n_unique": self.n_unique,
            "reduction_factor": self.reduction_factor,
        }


def _old_aggressive_normalize(dag):  # type: ignore[no-untyped-def]
    """Verbatim reimplementation of the pre-2026-07-27 relocation policy.

    Drops every CONST in-edge and adds node_0 -> c. ``add_edge`` silently refuses
    the new edge when it would close a cycle, which is precisely how the old code
    orphaned a CONST and made D2S fail.
    """
    from isalsr.core.labeled_dag import LabeledDAG
    from isalsr.core.node_types import NodeType

    new = LabeledDAG(dag.max_nodes)
    const_nodes: set[int] = set()
    for i in range(dag.node_count):
        data = dag.node_data(i)
        new.add_node(
            dag.node_label(i),
            var_index=int(data["var_index"]) if "var_index" in data else None,
            const_value=float(data["const_value"]) if "const_value" in data else None,
        )
        if dag.node_label(i) == NodeType.CONST:
            const_nodes.add(i)
    for tgt in range(dag.node_count):
        if tgt in const_nodes:
            continue
        for src in dag.ordered_inputs(tgt):
            new.add_edge(src, tgt)
    for c in sorted(const_nodes):
        new.add_edge(0, c)
    return new


def _structural_key(dag) -> tuple[Any, ...]:  # type: ignore[no-untyped-def]
    """Labels plus ordered in-edges — equal keys mean equal canonical strings."""
    return tuple(
        (dag.node_label(i).value, tuple(dag.ordered_inputs(i))) for i in range(dag.node_count)
    )


class ThreeArmRecorder:
    """Drop-in replacement for ``fast_canonical_string`` that scores all three arms."""

    def __init__(self, timeout: float = 60.0) -> None:
        self.timeout = timeout
        self.stats: dict[str, ArmStats] = {arm: ArmStats() for arm in ARMS}
        self.disagreements = 0
        self.conversion_errors = 0
        self.identical_all_arms = 0

    def __call__(self, dag, timeout=None, mode="wl_only", backend=None, **kwargs):  # type: ignore[no-untyped-def]
        from isalsr.core import _native
        from isalsr.core.canonical import _py_dag_to_native

        budget = float(timeout) if timeout is not None else self.timeout
        raw = _native.testing.fast_canonical_string_raw

        has_const = dag._has_const_nodes()
        prepared = {
            "submitted": _old_aggressive_normalize(dag) if has_const else dag,
            "repair": dag.normalize_const_creation() if has_const else dag,
            "none": dag,
        }

        # Fast path. When the three policies produce structurally identical DAGs
        # the canonical strings are identical too, so one call answers all three.
        # This is the overwhelmingly common case on adapter output, where every
        # CONST is already an x_1-anchored leaf, and it keeps the probe's cost
        # close to production instead of tripling it.
        keys = {arm: _structural_key(prepared[arm]) for arm in ARMS}
        all_same = keys["submitted"] == keys["repair"] == keys["none"]
        if all_same:
            self.identical_all_arms += 1

        results: dict[str, str | None] = {}
        cache: dict[tuple[Any, ...], str | None] = {}
        for arm in ARMS:
            self.stats[arm].n_calls += 1
            key = keys[arm]
            if key in cache:
                text = cache[key]
                if text is None:
                    self.stats[arm].n_failures += 1
                    results[arm] = None
                    continue
                results[arm] = text
                self.stats[arm].canonical_seen.add(hash(text))
                continue
            try:
                text = raw(_py_dag_to_native(prepared[arm]), budget)
            except Exception:  # noqa: BLE001 - any failure is a failure for this arm
                self.stats[arm].n_failures += 1
                results[arm] = None
                cache[key] = None
                continue
            cache[key] = text
            results[arm] = text
            self.stats[arm].canonical_seen.add(hash(text))

        # The repair must never merge what 'none' separates, and on a DAG whose
        # CONST nodes all have in-edges the two must be byte-identical.
        if (
            results["repair"] is not None
            and results["none"] is not None
            and results["repair"] != results["none"]
        ):
            self.disagreements += 1

        if results["repair"] is None:
            raise RuntimeError("Fast canonical D2S: no valid operation found")
        return results["repair"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arms": {arm: self.stats[arm].to_dict() for arm in ARMS},
            "repair_vs_none_disagreements": self.disagreements,
            "dags_where_all_arms_identical": self.identical_all_arms,
        }


def _load_problem(suite: str, problem: str, seed: int):  # type: ignore[no-untyped-def]
    """Return (x_train, y_train, x_test, y_test) at production dataset sizes.

    Nguyen takes explicit train/test counts (240/1000); every other suite takes
    a pooled sample count with a train ratio, which the configs set to 1250/0.8
    for a 1000/250 split.
    """
    import importlib

    module_name, const_name = _BENCH_SOURCE[suite]
    module = importlib.import_module(f"benchmarks.datasets.{module_name}")
    benches = getattr(module, const_name)
    if isinstance(benches, dict):
        benches = list(benches.values())
    bench = next(b for b in benches if b["name"] == problem)

    if suite == "nguyen":
        return module.generate_data(bench, n_train=240, n_test=1000, seed=seed)
    return module.generate_data(bench, n_samples=1250, train_ratio=0.8, seed=seed)


def _run_one(task: tuple[str, str, str, int, float]) -> dict[str, Any]:
    """Execute one (method, suite, problem, seed) search with the recorder installed."""
    method, suite, problem, seed, max_time = task
    logging.basicConfig(level=logging.WARNING)

    import isalsr.core.canonical as canonical_mod
    from experiments.models.orchestrator import create_runner

    started = time.time()
    record: dict[str, Any] = {
        "method": method,
        "suite": suite,
        "problem": problem,
        "seed": seed,
        "max_time": max_time,
    }

    try:
        x_train, y_train, x_test, y_test = _load_problem(suite, problem, seed)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"data: {exc}"
        return record

    config = _build_config(method, max_time)
    recorder = ThreeArmRecorder(timeout=config[method].get("canonicalization_timeout", 60.0))

    original = canonical_mod.fast_canonical_string
    canonical_mod.fast_canonical_string = recorder  # type: ignore[assignment]
    try:
        runner = create_runner(method, "isalsr", config)
        runner.fit(x_train, y_train, x_test, y_test, seed, config)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        canonical_mod.fast_canonical_string = original  # type: ignore[assignment]

    record.update(recorder.to_dict())
    record["wall_clock_s"] = time.time() - started
    return record


def _build_config(method: str, max_time: float) -> dict[str, Any]:
    """Production hyperparameters with max_time shortened for a probe run."""
    common = {"canonicalization_timeout": 60.0, "use_fast_canonical": True}
    if method == "bingo":
        return {
            "bingo": {
                "population_size": 500,
                "stack_size": 32,
                "operators": ["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "pow"],
                "use_simplification": False,
                "crossover_prob": 0.4,
                "mutation_prob": 0.4,
                "metric": "mse",
                "clo_alg": "lm",
                "generations": 100000000,
                "fitness_threshold": 1.0e-16,
                "max_time": max_time,
                "max_evals": 100000000,
                "snapshot_frequency": 10,
                **common,
            },
            "isalsr": common,
        }
    return {
        "udfs": {
            "max_time": max_time,
            "processes": 1,
            **common,
        },
        "isalsr": common,
    }


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — well behaved at zero successes, unlike Wald."""
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool per-run counts into per-method and global arm totals."""
    summary: dict[str, Any] = {"per_method": {}, "global": {}}
    methods = sorted({r["method"] for r in records if "arms" in r})

    for scope, subset in [(m, [r for r in records if r.get("method") == m]) for m in methods] + [
        ("global", records)
    ]:
        bucket: dict[str, Any] = {}
        for arm in ARMS:
            calls = sum(r["arms"][arm]["n_calls"] for r in subset if "arms" in r)
            fails = sum(r["arms"][arm]["n_failures"] for r in subset if "arms" in r)
            lo, hi = _wilson_interval(fails, calls)
            bucket[arm] = {
                "n_calls": calls,
                "n_failures": fails,
                "failure_rate": fails / calls if calls else 0.0,
                "wilson_95ci": [lo, hi],
                "mean_reduction_factor": float(
                    np.mean(
                        [
                            r["arms"][arm]["reduction_factor"]
                            for r in subset
                            if "arms" in r and np.isfinite(r["arms"][arm]["reduction_factor"])
                        ]
                        or [float("nan")]
                    )
                ),
            }
        bucket["repair_vs_none_disagreements"] = sum(
            r.get("repair_vs_none_disagreements", 0) for r in subset
        )
        bucket["dags_where_all_arms_identical"] = sum(
            r.get("dags_where_all_arms_identical", 0) for r in subset
        )
        bucket["n_runs"] = len([r for r in subset if "arms" in r])
        if scope == "global":
            summary["global"] = bucket
        else:
            summary["per_method"][scope] = bucket
    return summary


def format_table(summary: dict[str, Any]) -> str:
    """Render the per-method arm comparison as a fixed-width table."""
    lines = [
        f"{'method':<8} {'arm':<10} {'DAGs':>10} {'fail':>7} {'fail %':>9} "
        f"{'95% CI upper':>13} {'mean rho':>9}",
        "-" * 72,
    ]
    for method, bucket in sorted(summary["per_method"].items()):
        for arm in ARMS:
            a = bucket[arm]
            lines.append(
                f"{method:<8} {arm:<10} {a['n_calls']:>10,} {a['n_failures']:>7,} "
                f"{100 * a['failure_rate']:>8.4f}% {100 * a['wilson_95ci'][1]:>12.4f}% "
                f"{a['mean_reduction_factor']:>9.3f}"
            )
        lines.append("-" * 72)
    g = summary["global"]
    for arm in ARMS:
        a = g[arm]
        lines.append(
            f"{'ALL':<8} {arm:<10} {a['n_calls']:>10,} {a['n_failures']:>7,} "
            f"{100 * a['failure_rate']:>8.4f}% {100 * a['wilson_95ci'][1]:>12.4f}% "
            f"{a['mean_reduction_factor']:>9.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="bingo,udfs")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--max-time", type=float, default=120.0)
    parser.add_argument(
        "--problems", default=None, help="suite:problem,... (default: one per tier)"
    )
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    problems = DEFAULT_PROBLEMS
    if args.problems:
        problems = [tuple(p.split(":", 1)) for p in args.problems.split(",")]  # type: ignore[misc]

    tasks = [
        (method, suite, problem, seed, args.max_time)
        for method in args.methods.split(",")
        for suite, problem in problems
        for seed in [int(s) for s in args.seeds.split(",")]
    ]
    log.info("%d runs on %d workers (max_time=%.0fs each)", len(tasks), args.workers, args.max_time)

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        records = pool.map(_run_one, tasks)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise(records)

    (out_dir / "runs.json").write_text(json.dumps(records, indent=2, default=str))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    table = format_table(summary)
    (out_dir / "summary.txt").write_text(table + "\n")

    errors = [r for r in records if "error" in r]
    print(table)
    print(f"\nruns: {len(records)}   errors: {len(errors)}")
    for r in errors[:10]:
        print(f"  {r['method']}/{r['problem']}/seed{r['seed']}: {r['error']}")
    print(f"\nwrote {out_dir}/summary.json")

    # A run that canonicalised nothing reports zero failures out of zero DAGs,
    # which is indistinguishable from the clean result this probe exists to
    # confirm. Exit non-zero so SLURM marks the task FAILED instead of
    # COMPLETED, and so the aggregate never pools an empty arm as evidence.
    n_calls = summary["global"]["submitted"]["n_calls"]
    if n_calls == 0:
        raise SystemExit(
            f"FATAL: no DAG reached the canonicaliser across {len(records)} run(s). "
            f"{len(errors)} run(s) errored. This is not a zero failure rate — it is no data."
        )
    if errors:
        raise SystemExit(
            f"FATAL: {len(errors)}/{len(records)} run(s) errored; the pooled counters "
            "cover only the runs that survived and must not be reported as a rate."
        )


if __name__ == "__main__":
    main()
