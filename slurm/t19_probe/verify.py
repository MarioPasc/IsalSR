#!/usr/bin/env python
"""Verify the T19 complexity-telemetry probe.

Checks two things, and treats them as equally important:

1. **The new telemetry is present and sane** on all 24 cells, in both sampling
   modes, with the ``unique`` block populated exactly on the two arms that hold
   a deduplication cache.
2. **Nothing that was already recorded has been disturbed.** The probe would be
   worthless if it proved the new block works while the campaign's existing
   fields silently regressed, so every pre-T19 field is re-checked against the
   frozen ``RUN_LOG_FIELD_SPEC``.

Exit status is 0 only if every gate passes.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from experiments.models.schemas import SearchSpaceResults  # noqa: E402
from experiments.scripts.c2_certify import RUN_LOG_FIELD_SPEC  # noqa: E402

EXPECTED_CELLS = 24
ARMS = ("baseline", "hash", "isalsr")
METHODS = ("udfs", "bingo")
PROBLEMS = ("Nguyen-1", "Nguyen-10")
SEEDS = (0, 101)

#: Sampling mode each method must report. A mismatch means a runner was wired
#: to the wrong estimand, which would make the arms incomparable.
EXPECTED_MODE = {"bingo": "population", "udfs": "stream"}

#: Fields that existed before T19 and must still be present and populated. A
#: regression here is the failure this probe is really guarding against.
PRE_T19_REQUIRED = (
    "total_dags_explored",
    "unique_canonical_dags",
    "empirical_reduction_factor",
    "redundancy_rate",
    "max_internal_nodes_seen",
)

#: Distributional fields that must be non-None on every arm.
COMPLEXITY_REQUIRED = (
    "complexity_mean_k",
    "complexity_median_k",
    "complexity_p90_k",
    "complexity_mean_depth",
    "complexity_mean_edges",
    "complexity_mean_n_op",
    "complexity_mean_shared",
    "complexity_mean_nonlinear",
    "complexity_mean_op_entropy",
)

#: Populated only on arms holding a dedup cache; None on baseline.
COMPLEXITY_UNIQUE = (
    "complexity_unique_n_sampled",
    "complexity_unique_mean_k",
    "complexity_unique_mean_depth",
)


class Gate:
    """Accumulates pass/fail verdicts with human-readable detail."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        """Record one verdict.

        Parameters
        ----------
        name
            Short gate identifier.
        ok
            Whether the gate passed.
        detail
            Evidence, shown regardless of outcome.
        """
        self.rows.append((name, ok, detail))

    @property
    def failed(self) -> list[str]:
        """Names of gates that did not pass."""
        return [n for n, ok, _ in self.rows if not ok]

    def report(self) -> None:
        """Print the verdict table."""
        width = max(len(n) for n, _, _ in self.rows)
        print(f"\n{'gate'.ljust(width)}  verdict  detail")
        print("-" * (width + 60))
        for name, ok, detail in self.rows:
            print(f"{name.ljust(width)}  {'PASS' if ok else 'FAIL':7}  {detail}")
        print("-" * (width + 60))
        if self.failed:
            print(f"VERDICT: NO-GO -- {len(self.failed)} gate(s) failed: {', '.join(self.failed)}")
        else:
            print(f"VERDICT: GO -- {len(self.rows)}/{len(self.rows)} gates passed")


def load_cells(root: Path) -> list[dict[str, Any]]:
    """Load every ``run_log.json`` beneath *root*, with its sidecar.

    Parameters
    ----------
    root
        Results root of the probe.

    Returns
    -------
    list of dict
        One record per cell, carrying ``meta``, ``ss`` (search_space),
        ``regression``, ``sidecar`` and ``path``.
    """
    cells = []
    for path in sorted(root.rglob("run_log.json")):
        payload = json.loads(path.read_text())
        sidecar_path = path.parent / "complexity.json"
        cells.append(
            {
                "path": path,
                "meta": payload["metadata"],
                "ss": payload["results"]["search_space"],
                "regression": payload["results"]["regression"],
                "sidecar": json.loads(sidecar_path.read_text()) if sidecar_path.exists() else None,
            }
        )
    return cells


def _finite(value: Any) -> bool:
    """Return whether *value* is a finite real number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def _missing(cells: Iterable[dict[str, Any]], keys: Iterable[str]) -> list[str]:
    """Return ``cell/field`` labels for every required key that is absent or None."""
    out = []
    for cell in cells:
        tag = f"{cell['meta']['method']}/{cell['meta']['representation']}"
        for key in keys:
            if cell["ss"].get(key) is None:
                out.append(f"{tag}:{key}")
    return out


def main() -> int:
    """Run every gate and return the process exit status."""
    if len(sys.argv) != 2:
        print(__doc__)
        print("usage: verify.py <results_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"[FATAL] not a directory: {root}", file=sys.stderr)
        return 2

    cells = load_cells(root)
    gate = Gate()

    # --- G1: completeness -------------------------------------------------- #
    gate.check(
        "G1 cell count",
        len(cells) == EXPECTED_CELLS,
        f"{len(cells)}/{EXPECTED_CELLS} run_log.json found",
    )

    observed = {
        (c["meta"]["method"], c["meta"]["representation"], c["meta"]["problem"], c["meta"]["seed"])
        for c in cells
    }
    expected = {(m, a, p, s) for m in METHODS for a in ARMS for p in PROBLEMS for s in SEEDS}
    gate.check(
        "G2 full factorial",
        observed == expected,
        f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}"
        if observed != expected
        else "2 problems x 2 seeds x 3 arms x 2 methods all present",
    )

    if not cells:
        gate.report()
        return 1

    # --- G3: the sidecar exists everywhere --------------------------------- #
    no_sidecar = [str(c["path"].parent) for c in cells if c["sidecar"] is None]
    gate.check("G3 sidecar written", not no_sidecar, f"{len(cells) - len(no_sidecar)}/{len(cells)}")

    # --- G4: pre-T19 fields intact ----------------------------------------- #
    lost = _missing(cells, PRE_T19_REQUIRED)
    gate.check(
        "G4 pre-T19 fields intact",
        not lost,
        "no pre-existing field lost or nulled"
        if not lost
        else f"{len(lost)} problem(s): {lost[:6]}",
    )

    # --- G5: the whole frozen field spec still validates -------------------- #
    spec_problems: list[str] = []
    for cell in cells:
        payload = json.loads(cell["path"].read_text())
        for path_parts, types, nullable in RUN_LOG_FIELD_SPEC:
            node: Any = payload
            for part in path_parts:
                node = node.get(part) if isinstance(node, dict) else None
            key = ".".join(path_parts)
            if node is None:
                if not nullable:
                    spec_problems.append(f"{key}=missing")
            elif not isinstance(node, types):
                spec_problems.append(f"{key}={type(node).__name__}")
    gate.check(
        "G5 frozen field spec",
        not spec_problems,
        f"{len(RUN_LOG_FIELD_SPEC)} fields x {len(cells)} cells"
        if not spec_problems
        else f"{len(set(spec_problems))} distinct: {sorted(set(spec_problems))[:6]}",
    )

    # --- G6: telemetry actually fired -------------------------------------- #
    dead = [
        f"{c['meta']['method']}/{c['meta']['representation']}"
        for c in cells
        if not c["ss"].get("complexity_n_sampled")
    ]
    gate.check(
        "G6 telemetry fired",
        not dead,
        "every cell sampled >0 DAGs"
        if not dead
        else f"ZERO samples on {len(dead)} cell(s): {sorted(set(dead))}",
    )

    # --- G7: distributional fields populated and finite -------------------- #
    bad = _missing(cells, COMPLEXITY_REQUIRED)
    nonfinite = [
        f"{c['meta']['method']}/{c['meta']['representation']}:{k}"
        for c in cells
        for k in COMPLEXITY_REQUIRED
        if c["ss"].get(k) is not None and not _finite(c["ss"][k])
    ]
    gate.check(
        "G7 descriptors finite",
        not bad and not nonfinite,
        f"{len(COMPLEXITY_REQUIRED)} descriptors x {len(cells)} cells"
        if not bad and not nonfinite
        else f"missing={bad[:4]} nonfinite={nonfinite[:4]}",
    )

    # --- G8: sampling mode correct per method ------------------------------ #
    wrong_mode = [
        f"{c['meta']['method']}/{c['meta']['representation']}="
        f"{c['ss'].get('complexity_sampling_mode')}"
        for c in cells
        if c["ss"].get("complexity_sampling_mode") != EXPECTED_MODE[c["meta"]["method"]]
    ]
    gate.check(
        "G8 sampling mode",
        not wrong_mode,
        "bingo=population, udfs=stream" if not wrong_mode else str(sorted(set(wrong_mode))),
    )

    # --- G9: the sampling RULE is identical across the arms of a method ---- #
    # This is the gate the whole comparison rests on. If the three arms sampled
    # at different rates, an arm-versus-arm contrast would measure the
    # instrument rather than the search.
    rate_by_method: dict[str, set[Any]] = {}
    for cell in cells:
        rate_by_method.setdefault(cell["meta"]["method"], set()).add(
            cell["ss"].get("complexity_sample_rate")
        )
    ragged = {m: sorted(r) for m, r in rate_by_method.items() if len(r) != 1}
    gate.check(
        "G9 identical rule across arms",
        not ragged,
        " ".join(f"{m}={sorted(r)[0]}" for m, r in rate_by_method.items())
        if not ragged
        else f"RAGGED: {ragged}",
    )

    # --- G10: unique block exactly on the cached arms ---------------------- #
    wrong_unique = []
    for cell in cells:
        arm = cell["meta"]["representation"]
        has = cell["ss"].get("complexity_unique_n_sampled") is not None
        should = arm in ("hash", "isalsr")
        if has != should:
            wrong_unique.append(f"{cell['meta']['method']}/{arm}:has_unique={has}")
    gate.check(
        "G10 unique block placement",
        not wrong_unique,
        "populated on hash+isalsr, None on baseline"
        if not wrong_unique
        else str(sorted(set(wrong_unique))),
    )

    # --- G11: no telemetry failures ---------------------------------------- #
    failures = [
        (
            f"{c['meta']['method']}/{c['meta']['representation']}",
            c["ss"].get("complexity_n_failures"),
        )
        for c in cells
        if c["ss"].get("complexity_n_failures")
    ]
    gate.check(
        "G11 zero describe failures",
        not failures,
        "0 across all cells" if not failures else str(failures[:6]),
    )

    # --- G12: instrumentation cost is negligible --------------------------- #
    # Measured only on BUDGET-BOUND cells, i.e. those that ran out the clock
    # rather than stopping on convergence. A cell that converges in four seconds
    # still pays the generation-0 population sample, so its cost/wall ratio is
    # dominated by a fixed startup term and says nothing about the 12 h campaign
    # regime this gate exists to bound. Reporting that ratio as "overhead" would
    # be a measurement of how fast Nguyen-1 is solved.
    budget_bound = []
    absolute_worst = 0.0
    for cell in cells:
        wall = json.loads(cell["path"].read_text())["results"]["time"]["wall_clock_total_s"]
        cost = cell["ss"].get("complexity_time_s") or 0.0
        absolute_worst = max(absolute_worst, cost)
        budget = cell["meta"].get("hyperparameters", {}).get("max_time")
        if budget and wall >= 0.5 * float(budget):
            budget_bound.append(
                (
                    100.0 * cost / wall if wall > 0 else 0.0,
                    f"{cell['meta']['method']}/{cell['meta']['representation']}",
                )
            )

    if budget_bound:
        worst_pct, worst_tag = max(budget_bound)
        gate.check(
            "G12 overhead < 3%",
            worst_pct < 3.0,
            f"worst {worst_pct:.3f}% ({worst_tag}) over {len(budget_bound)}/{len(cells)} "
            f"budget-bound cells; max absolute cost {absolute_worst:.2f} s",
        )
    else:
        # No cell was budget-bound, so fall back to the absolute cost and say
        # so rather than passing vacuously on a ratio nobody should read.
        gate.check(
            "G12 overhead (absolute)",
            absolute_worst < 5.0,
            f"NO budget-bound cell; max absolute cost {absolute_worst:.2f} s "
            f"(percentage not reported -- every run stopped on convergence)",
        )

    # --- G13: scalars() covers exactly the schema's complexity fields ------ #
    schema_keys = {
        f.name for f in dataclass_fields(SearchSpaceResults) if f.name.startswith("complexity_")
    }
    emitted = set()
    for cell in cells:
        emitted |= {k for k in cell["ss"] if k.startswith("complexity_")}
    gate.check(
        "G13 schema coverage",
        emitted == schema_keys,
        f"{len(schema_keys)} fields"
        if emitted == schema_keys
        else f"missing={sorted(schema_keys - emitted)} extra={sorted(emitted - schema_keys)}",
    )

    # --- G14: alphabet (SP-4) ---------------------------------------------- #
    # The decomposed alphabet must not emit SUB or DIV. The label histogram in
    # the sidecar counts them over every sampled node, so this is checked on the
    # probe's own candidate stream rather than in a unit test.
    offenders = []
    for cell in cells:
        if cell["sidecar"] is None:
            continue
        counts = cell["sidecar"]["all"]["label_counts"]
        for label in ("SUB", "DIV"):
            if counts.get(label, 0):
                offenders.append(
                    f"{cell['meta']['method']}/{cell['meta']['representation']}:"
                    f"{label}={counts[label]}"
                )
    gate.check(
        "G14 SP-4 alphabet",
        not offenders,
        "0 SUB, 0 DIV over every sampled node"
        if not offenders
        else str(sorted(set(offenders))[:6]),
    )

    gate.report()

    # --- Descriptive summary (not a gate) ---------------------------------- #
    print("\nPer-cell telemetry (descriptive; a 900 s probe proves nothing scientific):")
    hdr = (
        f"{'method':6} {'arm':9} {'problem':10} {'seed':>4} {'mode':11} {'rate':>4} "
        f"{'n':>7} {'meanK':>6} {'depth':>6} {'nonlin':>6} {'share':>6} {'H':>5} "
        f"{'uqN':>7} {'cost%':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for cell in sorted(
        cells,
        key=lambda c: (
            c["meta"]["method"],
            c["meta"]["problem"],
            ARMS.index(c["meta"]["representation"]),
            c["meta"]["seed"],
        ),
    ):
        ss, meta = cell["ss"], cell["meta"]
        wall = json.loads(cell["path"].read_text())["results"]["time"]["wall_clock_total_s"]
        pct = 100.0 * (ss.get("complexity_time_s") or 0.0) / wall if wall else 0.0

        def fmt(key: str, width: int = 6, ss: dict[str, Any] = ss) -> str:
            value = ss.get(key)
            return f"{value:>{width}.2f}" if _finite(value) else f"{'-':>{width}}"

        print(
            f"{meta['method']:6} {meta['representation']:9} {meta['problem']:10} "
            f"{meta['seed']:>4} {str(ss.get('complexity_sampling_mode')):11} "
            f"{str(ss.get('complexity_sample_rate')):>4} "
            f"{str(ss.get('complexity_n_sampled')):>7} {fmt('complexity_mean_k')} "
            f"{fmt('complexity_mean_depth')} {fmt('complexity_mean_nonlinear')} "
            f"{fmt('complexity_mean_shared')} {fmt('complexity_mean_op_entropy', 5)} "
            f"{str(ss.get('complexity_unique_n_sampled')):>7} {pct:>6.3f}"
        )

    return 1 if gate.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
