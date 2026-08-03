"""Stage B environment and dataset probe (EXECUTION-PLAN.md §4.2, checks B1/B9).

Complements ``slurm/t04_probe/sp_probe.py`` (which establishes SP-1..SP-6) with
the two Stage B checks that need the benchmark registry rather than the engine:

B1  Environment probe -- hostname, CPU model, package and native-module paths
    with mtimes, engine, ``git describe``, library versions, free memory, and a
    **resolvability check on all 70 D1 u D2 problems**: every one must load with
    the shapes the registry declares.  Vlad-7 is 300/1200, Keijzer-6 is 50/120
    and Pagie-1 is 676/2500 -- these are the published protocols, not typos.

B9  T06 counter overhead, re-measured under the C++ engine and the decomposed
    alphabet.  Both changed underneath T06's original measurement, so an
    overhead that was negligible as a fraction of a Python canonicaliser costing
    ~24x more per DAG may not be negligible now.  This module only *summarises*
    the two runs the worker performs; it does not run them.

Writes a JSON evidence artefact and exits non-zero on failure.  "I checked it"
is not evidence; a parsed file is.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SUITES: tuple[str, ...] = (
    "nguyen",
    "feynman",
    "hard",
    "cherrypicked",
    "roundoff",
    "feynman_remainder",
    "strogatz",
)
EXPECTED_PROBLEMS = 70


def _run(cmd: list[str]) -> str:
    """Run a command and return its stripped stdout, or an error marker."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001 -- a probe never dies on its own diagnostics
        return f"<error: {type(exc).__name__}: {exc}>"


def b1_environment() -> dict[str, Any]:
    """Collect the compute node's identity, package provenance and versions."""
    info: dict[str, Any] = {
        "hostname": _run(["hostname"]),
        "cpu_model": _run(["bash", "-lc", "lscpu | sed -n 's/^Model name: *//p' | head -1"]),
        "free_g": _run(["bash", "-lc", "free -g | sed -n '2p'"]),
        "tmpdir": os.environ.get("TMPDIR", ""),
        "localscratch": os.environ.get("LOCALSCRATCH", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_describe": _run(
            ["git", "-C", str(REPO_ROOT), "describe", "--tags", "--always", "--dirty"]
        ),
        "git_head": _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "git_status_clean": _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"]) == "",
    }

    versions: dict[str, str] = {}
    for mod in ("numpy", "scipy", "sympy", "pandas", "yaml", "bingo"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            versions[mod] = f"<{type(exc).__name__}>"
    info["versions"] = versions
    info["python"] = sys.version.split()[0]
    info["python_executable"] = sys.executable

    try:
        import isalsr
        from isalsr.core import _native, backends

        so = Path(_native.__file__)
        info["isalsr_file"] = isalsr.__file__
        info["native_so"] = str(so)
        info["native_so_mtime"] = so.stat().st_mtime
        info["engine"] = backends.engine()
        info["build_info"] = dict(backends.build_info())
    except Exception as exc:  # noqa: BLE001
        info["engine_error"] = f"{type(exc).__name__}: {exc}"

    info["pass"] = info.get("engine") == "cpp"
    return info


def b1_datasets() -> dict[str, Any]:
    """Load every D1 u D2 problem and record its realised train/test shapes."""
    from experiments.models.orchestrator import (  # noqa: PLC0415
        _BENCHMARK_REGISTRY,
        _generate_benchmark_data,
        _get_ground_truth_sympy,
    )

    rows: list[dict[str, Any]] = []
    n_ok = 0
    n_sympy = 0
    for suite in SUITES:
        benches = _BENCHMARK_REGISTRY[suite][0]
        for bench in benches:
            name = bench["name"]
            row: dict[str, Any] = {"suite": suite, "problem": name}
            try:
                x_tr, y_tr, x_te, y_te = _generate_benchmark_data(suite, bench, 1000, 250, 0)
                row.update(
                    train_shape=list(x_tr.shape),
                    test_shape=list(x_te.shape),
                    y_train_shape=list(y_tr.shape),
                    y_test_shape=list(y_te.shape),
                    n_vars=int(bench["num_variables"]),
                    finite=bool(
                        __import__("numpy").isfinite(y_tr).all()
                        and __import__("numpy").isfinite(y_te).all()
                    ),
                    ok=True,
                )
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            # C1.5: solution_recovered is only computable with a sympy ground truth.
            try:
                row["has_sympy"] = _get_ground_truth_sympy(bench) is not None
            except Exception:  # noqa: BLE001
                row["has_sympy"] = False
            n_sympy += bool(row["has_sympy"])
            rows.append(row)

    return {
        "n_problems": len(rows),
        "n_resolved": n_ok,
        "n_with_sympy_ground_truth": n_sympy,
        "expected": EXPECTED_PROBLEMS,
        "failures": [r for r in rows if not r.get("ok")],
        "missing_sympy": [f"{r['suite']}/{r['problem']}" for r in rows if not r["has_sympy"]],
        "rows": rows,
        "pass": len(rows) == EXPECTED_PROBLEMS
        and n_ok == EXPECTED_PROBLEMS
        and n_sympy == EXPECTED_PROBLEMS,
    }


def b9_overhead(with_ledger: Path, without_ledger: Path) -> dict[str, Any]:
    """Compare two run logs to bound the T06 counter overhead.

    Args:
        with_ledger: Directory of the ``--ledger`` run.
        without_ledger: Directory of the same run without ``--ledger``.

    Returns:
        The two search times, the overhead as a percentage, the sampled
        denominator and the five fallback counts. A ledger whose
        ``n_ledger_sampled`` is zero is reported as **dead**, which is the whole
        point of the check -- a zero-everywhere ledger means the counters are
        not counting, not that the rates are zero.
    """

    def _load(seed_dir: Path) -> dict[str, Any] | None:
        hits = sorted(seed_dir.rglob("run_log.json"))
        if not hits:
            return None
        return json.loads(hits[0].read_text())

    on = _load(with_ledger)
    off = _load(without_ledger)
    if on is None or off is None:
        return {
            "pass": False,
            "error": f"missing run_log (on={on is not None}, off={off is not None})",
        }

    def _search_s(rl: dict[str, Any]) -> float:
        cost = rl.get("computational_cost", {})
        return float(cost.get("wall_clock_search_only_s") or cost.get("wall_clock_total_s") or 0.0)

    ss = on.get("search_space", {})
    sampled = ss.get("n_ledger_sampled")
    enabled = ss.get("ledger_enabled")
    t_on, t_off = _search_s(on), _search_s(off)
    overhead_pct = 100.0 * (t_on - t_off) / t_off if t_off > 0 else float("nan")

    live = bool(enabled) and isinstance(sampled, int) and sampled > 0
    return {
        "search_s_with_ledger": t_on,
        "search_s_without_ledger": t_off,
        "overhead_pct": overhead_pct,
        "ledger_enabled": enabled,
        "n_ledger_seen": ss.get("n_ledger_seen"),
        "n_ledger_sampled": sampled,
        "ledger_sample_rate": ss.get("ledger_sample_rate"),
        "five_paths": {
            k: ss.get(k)
            for k in (
                "n_violations_pre",
                "n_violations_post",
                "n_canon_timeouts",
                "n_conversion_failures",
                "n_canon_raised",
            )
        },
        "counters_live": live,
        # The overhead threshold is Mario's call (T06 AC-10); this records the
        # measurement and only fails on a DEAD counter, which is unambiguous.
        "pass": live,
    }


def main() -> int:
    """Run the Stage B checks selected on the command line."""
    ap = argparse.ArgumentParser(description="C2 Stage B environment/dataset/overhead probe")
    ap.add_argument("--out", required=True, help="JSON evidence path")
    ap.add_argument("--skip-datasets", action="store_true", help="Skip the 70-problem sweep")
    ap.add_argument("--ledger-on-dir", default=None, help="Run dir of the --ledger run (B9)")
    ap.add_argument("--ledger-off-dir", default=None, help="Run dir of the no-ledger run (B9)")
    args = ap.parse_args()

    evidence: dict[str, Any] = {"B1_environment": b1_environment()}
    if not args.skip_datasets:
        evidence["B1_datasets"] = b1_datasets()
    if args.ledger_on_dir and args.ledger_off_dir:
        evidence["B9_overhead"] = b9_overhead(Path(args.ledger_on_dir), Path(args.ledger_off_dir))

    failed = [k for k, v in evidence.items() if isinstance(v, dict) and not v.get("pass", True)]
    evidence["failed_checks"] = failed
    evidence["all_pass"] = not failed

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, default=str))

    for key, val in evidence.items():
        if isinstance(val, dict):
            print(f"  {key:22s} {'PASS' if val.get('pass', True) else 'FAIL'}")
    if "B1_datasets" in evidence:
        d = evidence["B1_datasets"]
        print(
            f"  datasets resolved   {d['n_resolved']}/{d['expected']}   "
            f"sympy ground truth {d['n_with_sympy_ground_truth']}/{d['expected']}"
        )
        for f in d["failures"][:10]:
            print(f"    FAIL {f['suite']}/{f['problem']}: {f.get('error')}")
        for m in d["missing_sympy"][:10]:
            print(f"    NO SYMPY {m}")
    if "B9_overhead" in evidence:
        b9 = evidence["B9_overhead"]
        print(
            f"  B9 overhead {b9.get('overhead_pct'):.2f}%  "
            f"sampled={b9.get('n_ledger_sampled')}  live={b9.get('counters_live')}"
            if "overhead_pct" in b9
            else f"  B9 {b9.get('error')}"
        )

    if failed:
        print(f"[FATAL] Stage B checks failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
