"""Why each run stopped, read from the counters the run itself recorded.

Bingo leaves ``evolve_until_convergence`` at a checkpoint for the first of these
reasons, in this order: the best fitness reached ``fitness_threshold``; Bingo's
fitness-evaluation counter reached ``max_fitness_evaluations`` (``max_evals``,
1e8 in every configuration); the elapsed time reached ``max_time``; or fewer
than a quarter of a checkpoint interval remain before ``max_time``. Stagnation
is not configured and the generation cap (1e7 or more) is never approached.
UDFS stops after a level when its best loss reaches ``stop_thresh`` or when its
``max_time`` is exhausted.

Bingo's counter is ``ExplicitRegression.eval_count``, which counts every call of
the fitness function, the Levenberg-Marquardt iterations of the constant fit
included. A skipped duplicate never calls it. Two records hold it:

* ``run_log.json`` -> ``results.search_space.total_dags_explored`` on the native
  arm only: the native runner reports the counter at termination there;
  the deduplicating arms report their candidate count in that field instead;
* ``convergence_log.npz`` -> ``n_evals`` on every arm: the counter captured at
  each generation boundary. The last capture can precede the final
  generation, so it may sit below the final counter by at most one
  generation's increment.

The classifier therefore reads the exact counter where it exists and, on the
deduplicating arms, counts a run as stopped at the cap when the last capture
plus the largest per-generation increment of that run reaches it. The rule is
checked on the native arm, where both readings exist (``validate_native``).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.scripts.blackbox.config import BINGO_MAX_EVALS, BUDGET_S, TIME_LIMIT_FRACTION
from experiments.scripts.review_campaign.config import ARMS, METHODS

#: Stop reasons in the order the report lists them.
REASONS: tuple[str, ...] = ("fitness", "evaluations", "time", "other")

#: Training R^2 at or above which a run below the caps is counted as having
#: met its fitness or loss threshold; any lower value goes to ``other``.
FITTED_R2_TRAIN = 1.0 - 1e-9

COUNTER_FIELD_NATIVE = "run_log.json:results.search_space.total_dags_explored"
COUNTER_FIELD_NPZ = "convergence_log.npz:n_evals (last capture + largest per-generation step)"


@dataclass(frozen=True)
class CounterReading:
    """Bingo's fitness-evaluation counter at the end of a run.

    Attributes:
        exact: Final counter from the run log (native arm only), else None.
        last_capture: Last value captured in ``convergence_log.npz``.
        max_step: Largest per-generation increment in that log.
    """

    exact: int | None
    last_capture: int | None
    max_step: int

    def at_cap(self) -> bool | None:
        """Whether the counter reached ``BINGO_MAX_EVALS``; None if unreadable."""
        if self.exact is not None:
            return self.exact >= BINGO_MAX_EVALS
        if self.last_capture is None:
            return None
        return self.last_capture + self.max_step >= BINGO_MAX_EVALS

    def at_cap_from_capture(self) -> bool | None:
        """The convergence-log reading alone, used to validate the rule."""
        if self.last_capture is None:
            return None
        return self.last_capture + self.max_step >= BINGO_MAX_EVALS


def read_counter(directory: Path, arm: str, run_log: dict[str, Any]) -> CounterReading:
    """Read Bingo's counter for one cell from both records.

    Args:
        directory: Cell directory.
        arm: Arm name; the run log carries the counter on the native arm only.
        run_log: Parsed ``run_log.json`` of the cell.

    Returns:
        The counter reading.
    """
    exact = (
        int(run_log["results"]["search_space"]["total_dags_explored"])
        if arm == "baseline"
        else None
    )
    npz = directory / "convergence_log.npz"
    if not npz.is_file():
        return CounterReading(exact=exact, last_capture=None, max_step=0)
    with np.load(npz) as data:
        n_evals = data["n_evals"].astype(np.int64)
    if n_evals.size == 0:
        return CounterReading(exact=exact, last_capture=None, max_step=0)
    step = int(np.diff(n_evals).max()) if n_evals.size > 1 else 0
    return CounterReading(exact=exact, last_capture=int(n_evals.max()), max_step=step)


def classify(method: str, wall_s: float, r2_train_raw: float | None, at_cap: bool | None) -> str:
    """Stop reason of one run, in Bingo's own order of checks.

    Args:
        method: Host.
        wall_s: Total wall clock of the run.
        r2_train_raw: Unclipped training R^2.
        at_cap: Whether Bingo's counter reached the cap (ignored on UDFS).

    Returns:
        One of ``REASONS``.
    """
    if method == "bingo" and at_cap:
        return "evaluations"
    if wall_s >= TIME_LIMIT_FRACTION * BUDGET_S:
        return "time"
    if r2_train_raw is not None and r2_train_raw >= FITTED_R2_TRAIN:
        return "fitness"
    return "other"


def scan(root: Path) -> list[dict[str, Any]]:
    """Classify every cell under ``<root>/<method>/<suite>/<problem>/<arm>/seed_NN``.

    Args:
        root: Campaign corpus root (read-only).

    Returns:
        One record per cell with its stop reason and the evidence used.
    """
    out: list[dict[str, Any]] = []
    for method in METHODS:
        for arm in ARMS:
            for path in sorted(root.glob(f"{method}/*/*/{arm}/seed_*/run_log.json")):
                log = json.loads(path.read_text(encoding="utf-8"))
                wall = float(log["results"]["time"]["wall_clock_total_s"])
                r2 = log["results"]["regression"]["r2_train"]
                reading = (
                    read_counter(path.parent, arm, log)
                    if method == "bingo"
                    else CounterReading(None, None, 0)
                )
                out.append(
                    {
                        "method": method,
                        "suite": path.parts[-5],
                        "problem": path.parts[-4],
                        "arm": arm,
                        "seed": int(path.parent.name.split("_")[1]),
                        "wall_s": wall,
                        "r2_train_raw": r2,
                        "counter_exact": reading.exact,
                        "counter_last_capture": reading.last_capture,
                        "counter_max_step": reading.max_step,
                        "at_cap_capture_rule": reading.at_cap_from_capture(),
                        "reason": classify(method, wall, r2, reading.at_cap()),
                    }
                )
    return out


def validate_native(records: list[dict[str, Any]]) -> dict[str, int]:
    """Agreement of the convergence-log rule with the exact counter (native Bingo).

    Args:
        records: ``scan`` output.

    Returns:
        Counts of agreeing and disagreeing native Bingo cells.
    """
    tally: Counter[str] = Counter()
    for rec in records:
        if rec["method"] != "bingo" or rec["arm"] != "baseline":
            continue
        exact = rec["counter_exact"] >= BINGO_MAX_EVALS
        tally["agree" if exact == rec["at_cap_capture_rule"] else "disagree"] += 1
        tally["exact_at_cap"] += int(exact)
    return dict(tally)


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts per host and arm, a native-by-arm cross table and the evidence ranges.

    Args:
        records: ``scan`` output.

    Returns:
        ``counts`` (host -> arm -> reason -> n), ``cross`` (host -> arm ->
        "native reason -> arm reason" -> n, seed-matched), ``evidence``
        (ranges of wall clock, training R^2 and counter per reason), and the
        native-arm validation.
    """
    counts: dict[str, dict[str, dict[str, int]]] = {
        m: {a: dict.fromkeys(REASONS, 0) for a in ARMS} for m in METHODS
    }
    evidence: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    index = {}
    for rec in records:
        counts[rec["method"]][rec["arm"]][rec["reason"]] += 1
        index[(rec["method"], rec["suite"], rec["problem"], rec["arm"], rec["seed"])] = rec
        bucket = evidence[f"{rec['method']}/{rec['reason']}"]
        bucket["wall_s"].append(rec["wall_s"])
        if rec["r2_train_raw"] is not None:
            bucket["r2_train_raw"].append(rec["r2_train_raw"])
        if rec["counter_last_capture"] is not None:
            bucket["counter_last_capture"].append(rec["counter_last_capture"])
    cross: dict[str, dict[str, dict[str, int]]] = {m: {} for m in METHODS}
    for (method, suite, problem, arm, seed), rec in index.items():
        if arm == "baseline":
            continue
        base = index.get((method, suite, problem, "baseline", seed))
        if base is None:
            continue
        key = f"{base['reason']} -> {rec['reason']}"
        cross[method].setdefault(arm, {})
        cross[method][arm][key] = cross[method][arm].get(key, 0) + 1
    ranges = {
        group: {field: [min(v), max(v)] for field, v in fields.items() if v}
        for group, fields in evidence.items()
    }
    return {
        "counts": counts,
        "cross": cross,
        "evidence": ranges,
        "native_validation": validate_native(records),
        "fields": {
            "bingo_counter_native": COUNTER_FIELD_NATIVE,
            "bingo_counter_all_arms": COUNTER_FIELD_NPZ,
            "time": f"results.time.wall_clock_total_s >= {TIME_LIMIT_FRACTION} x {BUDGET_S:.0f} s",
            "fitness": "results.regression.r2_train >= 1 - 1e-9, below both caps",
        },
    }
