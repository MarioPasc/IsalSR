"""Measure cross-architecture reproducibility of benchmark data generation.

Stage C v2 (2026-08-04) failed C4 with **35 of 210** ``(problem, seed)`` pairs
carrying more than one ``data_fingerprint``. All 35 partition **exactly** by CPU
family (0 exceptions): a cell that landed on an Intel ``sd`` node and one that
landed on an AMD ``sr``/``bl`` node generated data that is not byte-identical.

The fingerprint is byte equality over IEEE-754 bytes, which is deliberately
stricter than numerical equality, so a single-ULP difference in a transcendental
is enough to split it. This script quantifies the difference so the decision
between "pin the node family" and "relax the fingerprint" is made on a measured
magnitude rather than on the fact of a hash mismatch.

Run the same invocation on two node families and diff the two JSON outputs with
``--compare``.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.datasets.feynman import FEYNMAN_BENCHMARKS
from benchmarks.datasets.hard import HARD_BENCHMARKS
from benchmarks.datasets.nguyen import NGUYEN_BENCHMARKS
from benchmarks.datasets.structural import STRUCTURAL_BENCHMARKS
from experiments.models.orchestrator import _generate_benchmark_data
from experiments.models.provenance import data_fingerprint

#: Problems observed to split by node family in Stage C v2, plus controls that
#: did not. Keeping both means the probe can distinguish "this architecture
#: differs everywhere" from "it differs only for transcendental targets".
PROBES: list[tuple[str, str]] = [
    ("feynman", "I.12.4"),
    ("feynman", "I.6.20a"),
    ("nguyen", "Nguyen-1"),
    ("nguyen", "Nguyen-2"),
    ("cherrypicked", "Liv-14"),
    ("hard", "Vladislavleva-2"),
    ("feynman", "I.14.3"),
    ("nguyen", "Nguyen-5"),
]

_REGISTRY: dict[str, list[dict[str, Any]]] = {
    "nguyen": NGUYEN_BENCHMARKS,
    "feynman": FEYNMAN_BENCHMARKS,
    "hard": HARD_BENCHMARKS,
    "cherrypicked": STRUCTURAL_BENCHMARKS,
}


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def collect(seed: int) -> dict[str, Any]:
    """Generate every probe problem and record fingerprint plus raw bytes."""
    out: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "numpy": np.__version__,
        "problems": {},
    }
    for suite, name in PROBES:
        bench = next((b for b in _REGISTRY[suite] if b["name"] == name), None)
        if bench is None:
            out["problems"][name] = {"error": "not found"}
            continue
        try:
            x_tr, y_tr, x_te, y_te = _generate_benchmark_data(suite, bench, 1000, 250, seed)
        except Exception as exc:  # noqa: BLE001 - a probe must not die on one problem
            out["problems"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        out["problems"][name] = {
            "fingerprint": data_fingerprint(x_tr, y_tr, x_te, y_te),
            # Hex of the raw bytes lets --compare locate the differing element
            # without shipping the whole array.
            "y_train_hex_head": np.asarray(y_tr, dtype=np.float64)[:8].tobytes().hex(),
            "x_train_hex_head": np.ascontiguousarray(np.asarray(x_tr, dtype=np.float64))
            .ravel()[:8]
            .tobytes()
            .hex(),
            "y_train_sum": float(np.asarray(y_tr, dtype=np.float64).sum()),
            "y_train_shape": list(np.asarray(y_tr).shape),
        }
    return out


def compare(a_path: Path, b_path: Path) -> int:
    """Diff two collections and report the numerical size of each mismatch."""
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    print(f"A: {a['hostname']:10s} {a['cpu_model']}")
    print(f"B: {b['hostname']:10s} {b['cpu_model']}")
    print(f"numpy: {a['numpy']} vs {b['numpy']}")
    print()
    header = f"{'problem':<20}{'fingerprint':<14}{'max |Δy|':<14}{'max ULP':<10}"
    print(header)
    print("-" * len(header))
    n_diff = 0
    for name, pa in a["problems"].items():
        pb = b["problems"].get(name, {})
        if "error" in pa or "error" in pb:
            print(f"{name:<20}ERROR")
            continue
        same = pa["fingerprint"] == pb["fingerprint"]
        ya = np.frombuffer(bytes.fromhex(pa["y_train_hex_head"]), dtype=np.float64)
        yb = np.frombuffer(bytes.fromhex(pb["y_train_hex_head"]), dtype=np.float64)
        delta = float(np.max(np.abs(ya - yb))) if ya.size and ya.size == yb.size else float("nan")
        # ULP distance on the head elements, which is the scale that matters:
        # 1 ULP is libm noise, 2^20 ULP is a different computation.
        with np.errstate(invalid="ignore"):
            ulp = (
                float(np.max(np.abs(ya.view(np.int64) - yb.view(np.int64))))
                if ya.size and ya.size == yb.size
                else float("nan")
            )
        if not same:
            n_diff += 1
        print(f"{name:<20}{'MATCH' if same else 'DIFFER':<14}{delta:<14.3e}{ulp:<10.0f}")
    print()
    print(f"{n_diff} of {len(a['problems'])} problems differ between the two architectures.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Write the collection here.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("A.json", "B.json"),
        help="Diff two collections instead of generating one.",
    )
    args = parser.parse_args(argv)

    if args.compare:
        return compare(*args.compare)

    result = collect(args.seed)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}  ({result['hostname']}, {result['cpu_model']})")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
