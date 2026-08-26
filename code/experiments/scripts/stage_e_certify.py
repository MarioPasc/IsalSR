"""Stage E certifier -- the analysis dry-run on pre-flight data (checks E1-E7).

EXECUTION-PLAN §4.5. The analysis pipeline had never been run on three arms;
discovering that in September, after 100,800 core-hours are spent, is the single
most expensive failure mode left in the campaign. Stage E runs the whole
downstream pipeline -- analyzer, statistics, tables, figures -- against the
Stage C smoke root, and adversarially attacks the two properties C1 got wrong.

The checks:

===== ==================================================================
E1    Full analyzer end-to-end on a three-arm root; every artefact family
      in T02 §5.5 produced.
E2    Three-arm statistics: pairwise CPDT over the three contrasts with
      Holm, Friedman/Nemenyi over three arms, and the audit §6.1 contrast
      policy actually applied (rho descriptive against the definitional
      baseline, inferential only against hash).
E3    NaN policy, adversarially tested: inject a NaN and confirm it is
      never marked better, that N drops by exactly one, and that the
      conservative-substitution sensitivity check runs.
E4    LaTeX tables emit on three arms and compile; no cell renders `nan`.
E5    Figures emit on three arms with no silent two-arm fallback.
E6    Delete a run: the analyzer names the missing cell and refuses.
E7    Mix provenance: the analyzer refuses rather than silently pooling.
===== ==================================================================

E3, E6 and E7 are the point of the stage. A pipeline that runs cleanly on clean
input proves only that the input was clean; these three prove it *notices*.

Fixtures are built as copies, never in place, and the pristine smoke root is
only ever read. Nothing here writes to a campaign root (SP-0).

Usage:
    python -m experiments.scripts.stage_e_certify \
        --smoke-root  /path/to/c2_smoke_v4 \
        --work-dir    /path/to/c2_stage_e
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage_e")

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_METHODS = ("udfs", "bingo")
DEFAULT_BENCHMARKS = (
    "nguyen",
    "feynman",
    "hard",
    "cherrypicked",
    "roundoff",
    "feynman_remainder",
    "strogatz",
)
DEFAULT_VARIANTS = ("baseline", "hash", "isalsr")

# Fixtures only need to exercise a code path, not the full portfolio, so they
# are cut to two suites. This keeps each adversarial run inside the soft-probe
# budget while still crossing both methods and all three arms.
FIXTURE_BENCHMARKS = ("nguyen", "feynman")

# Artefact families check E1 requires, per T02 §5.5.
REQUIRED_ARTEFACT_PREFIXES = (
    "benchmark_summary_",
    "computational_overhead_",
    "cross_method_",
    "reduction_comparison_",
    "three_axis_summary_",
    "cross_problem_dominance_",
)
REQUIRED_ARTEFACT_FILES = ("three_axis_global.json", "global_summary.json")

# The audit §6.1 contrast policy, as it must appear in the emitted CPDT.
# metric -> contrast -> expected `alternative`.
EXPECTED_CONTRAST_POLICY: dict[str, dict[str, str]] = {
    "r2_train": {
        "isalsr_vs_baseline": "greater",
        "isalsr_vs_hash": "two-sided",
        "hash_vs_baseline": "two-sided",
    },
    "r2_test": {
        "isalsr_vs_baseline": "greater",
        "isalsr_vs_hash": "two-sided",
        "hash_vs_baseline": "two-sided",
    },
    "nrmse_test": {
        "isalsr_vs_baseline": "less",
        "isalsr_vs_hash": "two-sided",
        "hash_vs_baseline": "two-sided",
    },
    "empirical_reduction_factor": {
        "isalsr_vs_baseline": "descriptive",
        "isalsr_vs_hash": "greater",
        "hash_vs_baseline": "descriptive",
    },
    "redundancy_rate": {
        "isalsr_vs_baseline": "descriptive",
        "isalsr_vs_hash": "greater",
        "hash_vs_baseline": "descriptive",
    },
}

PYTHON = sys.executable


@dataclass
class CheckResult:
    """Outcome of one Stage E check."""

    check: str
    title: str
    status: str = "FAIL"
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def passed(self) -> bool:
        """True when the check met its criterion."""
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the certification artefact."""
        return {
            "check": self.check,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
            "elapsed_s": round(self.elapsed_s, 2),
        }


# ======================================================================
# Subprocess helpers
# ======================================================================


def _run(cmd: list[str], log_path: Path, timeout: int = 900) -> tuple[int, str]:
    """Run a pipeline command, tee its output to a file, and return the status.

    Args:
        cmd: Argument vector.
        log_path: Where combined stdout/stderr is written.
        timeout: Hard cap in seconds. Every Stage E step is a soft probe.

    Returns:
        ``(returncode, combined_output)``. A timeout yields returncode 124.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = proc.stdout + proc.stderr
        code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = f"TIMEOUT after {timeout}s\n{exc.stdout or ''}{exc.stderr or ''}"
        code = 124
    log_path.write_text(output, encoding="utf-8")
    return code, output


def _analyze_cmd(
    root: Path,
    methods: Sequence[str],
    benchmarks: Sequence[str],
    variants: Sequence[str],
    allow_incomplete: bool = False,
    allow_mixed_provenance: bool = False,
) -> list[str]:
    """Build an ``experiments.models.analyze`` invocation."""
    cmd = [
        PYTHON,
        "-m",
        "experiments.models.analyze",
        "--results-dir",
        str(root),
        "--methods",
        ",".join(methods),
        "--benchmarks",
        ",".join(benchmarks),
        "--variants",
        ",".join(variants),
    ]
    if allow_incomplete:
        cmd.append("--allow-incomplete")
    if allow_mixed_provenance:
        cmd.append("--allow-mixed-provenance")
    return cmd


def _copy_subset(src: Path, dst: Path, methods: Sequence[str], benchmarks: Sequence[str]) -> None:
    """Copy a ``method/benchmark`` subset of a results root into a fresh dir."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for method in methods:
        for benchmark in benchmarks:
            src_dir = src / method / benchmark
            if src_dir.is_dir():
                shutil.copytree(src_dir, dst / method / benchmark)


# ======================================================================
# E1 -- analyzer end to end
# ======================================================================


def check_e1(
    root: Path,
    methods: Sequence[str],
    benchmarks: Sequence[str],
    variants: Sequence[str],
    logs: Path,
) -> CheckResult:
    """Run the analyzer end to end on the three-arm root."""
    res = CheckResult("E1", "Full analyzer end-to-end on a three-arm root")
    t0 = time.perf_counter()

    # The smoke root is known to pool two commits (161 of v4's cells recorded
    # `-dirty`), so E1 must opt in explicitly. E7 is what proves the guard bites.
    code, _ = _run(
        _analyze_cmd(root, methods, benchmarks, variants, allow_mixed_provenance=True),
        logs / "e1_analyze.log",
    )
    res.elapsed_s = time.perf_counter() - t0

    analysis = root / "analysis"
    if code != 0:
        res.detail = f"analyzer exited {code}"
        res.evidence = {"returncode": code}
        return res
    if not analysis.is_dir():
        res.detail = "no analysis/ directory produced"
        return res

    names = sorted(p.name for p in analysis.iterdir() if p.is_file())
    counts = {pre: sum(1 for n in names if n.startswith(pre)) for pre in REQUIRED_ARTEFACT_PREFIXES}
    missing_families = [pre for pre, n in counts.items() if n == 0]
    missing_files = [f for f in REQUIRED_ARTEFACT_FILES if f not in names]

    # Every arm must be represented in the aggregates, or "three arms ran" is
    # a statement about the command line rather than about the output.
    arms_seen = _arms_in_summaries(analysis, variants)
    missing_arms = sorted(set(variants) - arms_seen)

    res.evidence = {
        "returncode": code,
        "n_artefacts": len(names),
        "family_counts": counts,
        "arms_present_in_summaries": sorted(arms_seen),
        "elapsed_s": round(res.elapsed_s, 1),
    }
    if missing_families or missing_files or missing_arms:
        res.detail = (
            f"missing families={missing_families} files={missing_files} arms={missing_arms}"
        )
        return res

    res.status = "PASS"
    res.detail = (
        f"{len(names)} artefacts, all {len(REQUIRED_ARTEFACT_PREFIXES)} families present, "
        f"arms {sorted(arms_seen)}, {res.elapsed_s:.0f}s"
    )
    return res


def _arms_in_summaries(analysis: Path, variants: Sequence[str]) -> set[str]:
    """Collect the arm names that actually appear in the CPDT contrast blocks."""
    seen: set[str] = set()
    for path in analysis.glob("cross_problem_dominance_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for body in (payload.get("contrasts") or {}).values():
            for metric in body.values():
                if isinstance(metric, dict):
                    for key in ("arm_a", "arm_b"):
                        if metric.get(key) in variants:
                            seen.add(str(metric[key]))
    return seen


# ======================================================================
# E2 -- three-arm statistics
# ======================================================================


def check_e2(root: Path, logs: Path) -> CheckResult:
    """Verify the three contrasts, the Holm correction and the rho policy."""
    res = CheckResult("E2", "Three-arm CPDT with Holm, Friedman/Nemenyi over three arms")
    t0 = time.perf_counter()
    analysis = root / "analysis"

    cpdt_files = sorted(analysis.glob("cross_problem_dominance_*.json"))
    if not cpdt_files:
        res.detail = "no CPDT artefacts"
        return res

    policy_violations: list[str] = []
    missing_holm: list[str] = []
    rho_p_against_baseline: list[str] = []
    n_reported: set[int] = set()
    contrasts_seen: set[str] = set()
    sensitivity_present = 0

    for path in cpdt_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("sensitivity_conservative"):
            sensitivity_present += 1
        contrasts = payload.get("contrasts") or {}
        contrasts_seen.update(contrasts)
        for cname, body in contrasts.items():
            for metric, entry in body.items():
                if not isinstance(entry, dict):
                    continue
                expected = EXPECTED_CONTRAST_POLICY.get(metric, {}).get(cname)
                actual = entry.get("alternative")
                if expected is not None and actual != expected:
                    policy_violations.append(
                        f"{path.name}:{cname}:{metric} alternative={actual!r} want {expected!r}"
                    )
                if entry.get("n_problems") is not None:
                    n_reported.add(int(entry["n_problems"]))

                holm = entry.get("p_value_holm")
                if expected == "descriptive":
                    # rho against a baseline whose rho is 1.0 by construction is
                    # reported, never tested (audit §6.1 / F-3).
                    if holm is not None and not (isinstance(holm, float) and math.isnan(holm)):
                        rho_p_against_baseline.append(f"{path.name}:{cname}:{metric} p={holm}")
                elif holm is None:
                    missing_holm.append(f"{path.name}:{cname}:{metric}")

    # Friedman over three arms, from the cross-method artefacts.
    friedman_groups: set[int] = set()
    for path in sorted(analysis.glob("cross_method_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for value in _walk_for_key(payload, "group_names"):
            if isinstance(value, list):
                friedman_groups.add(len(value))

    res.elapsed_s = time.perf_counter() - t0
    res.evidence = {
        "n_cpdt_files": len(cpdt_files),
        "contrasts_seen": sorted(contrasts_seen),
        "n_problems_reported": sorted(n_reported),
        "friedman_group_sizes": sorted(friedman_groups),
        "sensitivity_blocks": sensitivity_present,
        "policy_violations": policy_violations[:10],
        "missing_holm": missing_holm[:10],
        "rho_tested_against_baseline": rho_p_against_baseline[:10],
    }

    problems: list[str] = []
    if contrasts_seen != {"isalsr_vs_baseline", "isalsr_vs_hash", "hash_vs_baseline"}:
        problems.append(f"contrasts {sorted(contrasts_seen)}")
    if policy_violations:
        problems.append(f"{len(policy_violations)} contrast-policy violations")
    if missing_holm:
        problems.append(f"{len(missing_holm)} entries without p_value_holm")
    if rho_p_against_baseline:
        problems.append(f"{len(rho_p_against_baseline)} rho p-values against the baseline")
    if sensitivity_present == 0:
        problems.append("no conservative-substitution sensitivity block")
    if friedman_groups and max(friedman_groups) < 6:
        problems.append(f"Friedman group sizes {sorted(friedman_groups)} < 6 (2 methods x 3 arms)")

    if problems:
        res.detail = "; ".join(problems)
        return res
    res.status = "PASS"
    res.detail = (
        f"3 contrasts x {len(EXPECTED_CONTRAST_POLICY)} metrics across {len(cpdt_files)} files; "
        f"policy matches audit §6.1; Holm present; rho descriptive vs baseline; "
        f"Friedman groups {sorted(friedman_groups)}"
    )
    return res


def _walk_for_key(node: Any, key: str) -> list[Any]:
    """Collect every value stored under ``key`` anywhere in a nested structure."""
    found: list[Any] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                found.append(v)
            found.extend(_walk_for_key(v, key))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk_for_key(item, key))
    return found


# ======================================================================
# E3 -- NaN policy, adversarially tested
# ======================================================================


def check_e3(
    smoke_root: Path,
    work: Path,
    methods: Sequence[str],
    variants: Sequence[str],
    logs: Path,
) -> CheckResult:
    """Inject a NaN and confirm it is never a winner and drops N by one."""
    res = CheckResult("E3", "NaN policy: injected NaN never wins, N drops by exactly 1")
    t0 = time.perf_counter()

    fixture = work / "e3_nan"
    _copy_subset(smoke_root, fixture, methods, FIXTURE_BENCHMARKS)

    # Pick a deterministic victim: the isalsr arm of the first problem, so the
    # NaN lands on the arm a careless comparison would be tempted to bold.
    victims = sorted((fixture / methods[0] / FIXTURE_BENCHMARKS[0]).glob("*/isalsr/*/run_log.json"))
    if not victims:
        res.detail = "no isalsr run log to poison"
        return res
    victim = victims[0]
    problem = victim.parents[2].name
    baseline_n = _paired_n(fixture, methods[0], FIXTURE_BENCHMARKS[0], problem, "r2_test")

    payload = json.loads(victim.read_text(encoding="utf-8"))
    payload["results"]["regression"]["r2_test"] = float("nan")
    victim.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    code, _ = _run(
        _analyze_cmd(fixture, methods, FIXTURE_BENCHMARKS, variants, allow_mixed_provenance=True),
        logs / "e3_analyze.log",
    )
    if code != 0:
        res.detail = f"analyzer exited {code} on the poisoned root"
        res.evidence = {"returncode": code, "victim": str(victim.relative_to(fixture))}
        return res

    tables = work / "artefacts" / "tables_e3"
    tcode, _ = _run(
        [
            PYTHON,
            "-m",
            "experiments.figures.models.generate_tables",
            "--results-dir",
            str(fixture),
            "--output-dir",
            str(tables),
            "--methods",
            ",".join(methods),
            "--benchmarks",
            ",".join(FIXTURE_BENCHMARKS),
        ],
        logs / "e3_tables.log",
    )

    nan_marked_better = _nan_marked_better(tables)
    poisoned_n = _paired_n(fixture, methods[0], FIXTURE_BENCHMARKS[0], problem, "r2_test")
    sensitivity = _sensitivity_present(fixture / "analysis")
    superscript = _table_effective_n(tables, problem)

    res.elapsed_s = time.perf_counter() - t0
    res.evidence = {
        "victim": str(victim.relative_to(fixture)),
        "poisoned_problem": problem,
        "tables_returncode": tcode,
        "n_before": baseline_n,
        "n_after": poisoned_n,
        "table_superscript_n": superscript,
        "nan_marked_better": nan_marked_better,
        "sensitivity_blocks": sensitivity,
    }

    problems: list[str] = []
    if nan_marked_better:
        problems.append(f"NaN marked better in {nan_marked_better}")
    if baseline_n is None or poisoned_n is None:
        problems.append(f"could not read paired N (before={baseline_n}, after={poisoned_n})")
    elif poisoned_n != baseline_n - 1:
        problems.append(f"N went {baseline_n} -> {poisoned_n}, expected {baseline_n - 1}")
    if sensitivity == 0:
        problems.append("conservative-substitution check absent")
    if superscript is not None and poisoned_n is not None and superscript != poisoned_n:
        problems.append(f"table reports [{superscript}] but pairwise deletion gives {poisoned_n}")

    if problems:
        res.detail = "; ".join(problems)
        return res
    res.status = "PASS"
    res.detail = (
        f"NaN never bolded; paired N {baseline_n} -> {poisoned_n} (exactly one dropped); "
        f"table discloses [{superscript}]; conservative-substitution in {sensitivity} artefacts"
    )
    return res


def _table_effective_n(tables: Path, problem: str) -> int | None:
    """Read the ``[n]`` effective-seed superscript a table prints for a problem.

    The supplementary table marks any problem whose paired seed count falls
    below the nominal one. That mark is the reader-visible half of the NaN
    policy: a dropped seed must be disclosed, not absorbed.

    Args:
        tables: Directory of emitted ``.tex`` files.
        problem: Problem directory name, e.g. ``nguyen_1``.

    Returns:
        The superscript value, or None when the table does not mark it.
    """
    if not tables.is_dir():
        return None
    # Rows are keyed by the *mapped* display label ("N-1"), not the directory
    # name ("nguyen_1"), so resolve it the same way the generator does.
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.figures.models.generate_tables import _problem_label

    label = _problem_label(problem)
    pattern = re.compile(rf"{re.escape(label)}\$\^{{\[(\d+)\]}}\$")
    for path in sorted(tables.glob("*.tex")):
        match = pattern.search(path.read_text(encoding="utf-8", errors="replace"))
        if match:
            return int(match.group(1))
    return None


def _paired_n(root: Path, method: str, benchmark: str, problem: str, metric: str) -> int | None:
    """Count seeds where *both* compared arms hold a finite value for ``metric``.

    This is pairwise deletion measured from the run logs, which is the quantity
    the tables surface as the ``[n]`` superscript. ``paired_stats.json`` records
    only a nominal top-level ``n_seeds``, so reading it would not detect the
    drop a single NaN is supposed to cause.

    Args:
        root: Results root.
        method: Method directory.
        benchmark: Benchmark directory.
        problem: Problem directory name.
        metric: Field under ``results.regression``.

    Returns:
        The number of seeds paired across baseline and isalsr, or None when the
        problem directory is absent.
    """
    prob_dir = root / method / benchmark / problem
    if not prob_dir.is_dir():
        return None

    finite_by_arm: dict[str, set[str]] = {}
    for arm in ("baseline", "isalsr"):
        arm_dir = prob_dir / arm
        if not arm_dir.is_dir():
            continue
        good: set[str] = set()
        for seed_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
            path = seed_dir / "run_log.json"
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            value = ((payload.get("results") or {}).get("regression") or {}).get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                good.add(seed_dir.name)
        finite_by_arm[arm] = good

    if len(finite_by_arm) < 2:
        return None
    return len(finite_by_arm["baseline"] & finite_by_arm["isalsr"])


def _sensitivity_present(analysis: Path) -> int:
    """Count CPDT artefacts carrying a conservative-substitution block."""
    n = 0
    for path in analysis.glob("cross_problem_dominance_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("sensitivity_conservative"):
            n += 1
    return n


# Bold/underline markup a table uses to declare a winner.
_WINNER_PATTERNS = (
    re.compile(r"\\textbf\{\s*\$?\s*nan", re.IGNORECASE),
    re.compile(r"\\mathbf\{\s*nan", re.IGNORECASE),
    re.compile(r"\\underline\{\s*\$?\s*nan", re.IGNORECASE),
)


def _nan_marked_better(tables: Path) -> list[str]:
    """Return tables where a NaN carries winner markup."""
    hits: list[str] = []
    if not tables.is_dir():
        return hits
    for path in sorted(tables.glob("*.tex")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(p.search(text) for p in _WINNER_PATTERNS):
            hits.append(path.name)
    return hits


# ======================================================================
# E4 -- tables emit and compile, and never render nan
# ======================================================================

# A bare `nan` as a rendered value, not the "nan" inside "Dominance".
_NAN_CELL = re.compile(r"(?<![A-Za-z])\$?-?nan\$?(?![A-Za-z])", re.IGNORECASE)

# xcolor is loaded once, with [table]; loading it twice raises an option clash
# and every table then "fails to compile" for a reason that is the harness's.
_LATEX_PREAMBLE = r"""\documentclass{article}
\usepackage[table]{xcolor}
\usepackage{booktabs,amsmath,amssymb,graphicx,multirow,array,longtable,siunitx}
\providecommand{\IsalSR}{IsalSR}
\providecommand{\checkmark}{x}
\begin{document}
"""


def check_e4(
    root: Path,
    work: Path,
    methods: Sequence[str],
    benchmarks: Sequence[str],
    logs: Path,
) -> CheckResult:
    """Emit the LaTeX tables on three arms, compile them, and forbid `nan`."""
    res = CheckResult("E4", "Three-arm LaTeX tables emit, compile, and render no nan")
    t0 = time.perf_counter()
    tables = work / "artefacts" / "tables"

    code, _ = _run(
        [
            PYTHON,
            "-m",
            "experiments.figures.models.generate_tables",
            "--results-dir",
            str(root),
            "--output-dir",
            str(tables),
            "--methods",
            ",".join(methods),
            "--benchmarks",
            ",".join(benchmarks),
        ],
        logs / "e4_tables.log",
    )
    if code != 0:
        res.detail = f"generate_tables exited {code}"
        return res

    emitted = sorted(tables.glob("*.tex"))
    if not emitted:
        res.detail = "no .tex emitted"
        return res

    # A `nan` in a caption is prose; a `nan` in a tabular row is a rendered
    # value and is what audit §6.1 forbids.
    nan_cells: dict[str, list[str]] = {}
    for path in emitted:
        bad: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("\\caption") or stripped.startswith("%"):
                continue
            if _NAN_CELL.search(line):
                bad.append(stripped[:120])
        if bad:
            nan_cells[path.name] = bad[:5]

    compiled, compile_errors = _compile_tables(emitted, work / "artefacts" / "latex_build", logs)

    res.elapsed_s = time.perf_counter() - t0
    res.evidence = {
        "n_tables": len(emitted),
        "tables": [p.name for p in emitted],
        "nan_cells": nan_cells,
        "compiled": compiled,
        "compile_errors": compile_errors,
    }

    problems: list[str] = []
    if nan_cells:
        problems.append(f"nan rendered in {sorted(nan_cells)}")
    if compile_errors:
        problems.append(f"{len(compile_errors)} tables failed to compile")
    if problems:
        res.detail = "; ".join(problems)
        return res
    res.status = "PASS"
    res.detail = f"{len(emitted)} tables emitted, {compiled} compiled, zero nan cells"
    return res


def _compile_tables(
    tables: Sequence[Path],
    build_dir: Path,
    logs: Path,
) -> tuple[int, list[str]]:
    """Compile each emitted table standalone; return (n_ok, failures)."""
    build_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    failures: list[str] = []
    for path in tables:
        doc = _LATEX_PREAMBLE + path.read_text(encoding="utf-8") + "\n\\end{document}\n"
        tex = build_dir / f"{path.stem}_standalone.tex"
        tex.write_text(doc, encoding="utf-8")
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if proc.returncode == 0 and tex.with_suffix(".pdf").is_file():
            ok += 1
        else:
            failures.append(path.name)
            (logs / f"e4_latex_{path.stem}.log").write_text(
                proc.stdout + proc.stderr, encoding="utf-8"
            )
    return ok, failures


# ======================================================================
# E5 -- figures, with no silent two-arm fallback
# ======================================================================

_TREATMENT_RE = re.compile(r"(UDFS|BINGO)\s+(native DAG|Naive-Hash|IsalSR)", re.IGNORECASE)


def check_e5(
    root: Path,
    work: Path,
    methods: Sequence[str],
    benchmarks: Sequence[str],
    variants: Sequence[str],
    logs: Path,
) -> CheckResult:
    """Generate the figure suite on three arms and forbid a two-arm fallback."""
    res = CheckResult("E5", "Figures on three arms, no silent two-arm fallback")
    t0 = time.perf_counter()
    figures = work / "artefacts" / "figures"

    code, output = _run(
        [
            PYTHON,
            "-m",
            "experiments.figures.models.generate_all",
            "--results-dir",
            str(root),
            "--output-dir",
            str(figures),
            "--methods",
            ",".join(methods),
            "--benchmarks",
            ",".join(benchmarks),
            "--variants",
            ",".join(variants),
        ],
        logs / "e5_figures.log",
        timeout=1800,
    )

    expected_groups = len(methods) * len(variants)
    # The CD .tex carries its treatment names, so the arm count is read from the
    # artefact rather than from the log that claims to have produced it.
    cd_groups: dict[str, int] = {}
    for path in sorted(figures.glob("cd_*.tex")):
        names = set(_TREATMENT_RE.findall(path.read_text(encoding="utf-8", errors="replace")))
        cd_groups[path.name] = len(names)
    short = {k: v for k, v in cd_groups.items() if v != expected_groups}

    forest = sorted(figures.glob("forest_plot*.pdf"))
    produced = sorted(p.name for p in figures.iterdir() if p.is_file())

    res.elapsed_s = time.perf_counter() - t0
    res.evidence = {
        "returncode": code,
        "n_files": len(produced),
        "expected_cd_groups": expected_groups,
        "cd_group_counts": cd_groups,
        "cd_with_wrong_group_count": short,
        "forest_plot": [p.name for p in forest],
        "summary_line": output.strip().splitlines()[-1][:200] if output.strip() else "",
    }

    problems: list[str] = []
    if code != 0:
        problems.append(f"generate_all exited {code}")
    if not cd_groups:
        problems.append("no CD diagrams emitted")
    if short:
        problems.append(
            f"{len(short)} CD diagrams not at {expected_groups} groups: {sorted(short)}"
        )
    if not forest:
        problems.append("no forest plot")

    if problems:
        res.detail = "; ".join(problems)
        return res
    res.status = "PASS"
    res.detail = (
        f"{len(produced)} artefacts; all {len(cd_groups)} CD diagrams carry "
        f"{expected_groups} groups (2 methods x 3 arms); forest plot present"
    )
    return res


# ======================================================================
# E6 -- a deleted run must be named, not tolerated
# ======================================================================


def check_e6(
    smoke_root: Path,
    work: Path,
    methods: Sequence[str],
    variants: Sequence[str],
    logs: Path,
) -> CheckResult:
    """Delete one run and require the analyzer to refuse and name the cell."""
    res = CheckResult("E6", "Deleted cell is named and refused, not silently tolerated")
    t0 = time.perf_counter()

    fixture = work / "e6_missing"
    _copy_subset(smoke_root, fixture, methods, FIXTURE_BENCHMARKS)

    candidates = sorted((fixture / methods[0] / FIXTURE_BENCHMARKS[0]).glob("*/*/*/run_log.json"))
    if not candidates:
        res.detail = "no run log to delete"
        return res
    n_before = len(sorted(fixture.rglob("run_log.json")))
    victim = candidates[0]
    rel = victim.relative_to(fixture)
    victim.unlink()

    # Refuse by default.
    code_strict, out_strict = _run(
        _analyze_cmd(fixture, methods, FIXTURE_BENCHMARKS, variants, allow_mixed_provenance=True),
        logs / "e6_strict.log",
    )
    # Proceed, but still name the cell, under the explicit override.
    code_allow, _ = _run(
        _analyze_cmd(
            fixture,
            methods,
            FIXTURE_BENCHMARKS,
            variants,
            allow_incomplete=True,
            allow_mixed_provenance=True,
        ),
        logs / "e6_allow.log",
    )

    # rel is <method>/<benchmark>/<problem>/<arm>/<seed>/run_log.json, and the
    # reconciler names cells in exactly that first-five-component form.
    expected_cell = "/".join(rel.parts[:5])
    named_in_log = expected_cell in out_strict

    integrity = fixture / "analysis" / "campaign_integrity.json"
    named_in_report = False
    reported = {}
    if integrity.is_file():
        payload = json.loads(integrity.read_text(encoding="utf-8"))
        comp = payload.get("completeness") or {}
        reported = {
            "n_expected": comp.get("n_expected"),
            "n_observed": comp.get("n_observed"),
            "missing": comp.get("missing", [])[:5],
        }
        named_in_report = expected_cell in (comp.get("missing") or [])

    res.elapsed_s = time.perf_counter() - t0
    res.evidence = {
        "deleted_cell": expected_cell,
        "n_runs_before": n_before,
        "returncode_strict": code_strict,
        "returncode_allow": code_allow,
        "named_in_log": named_in_log,
        "named_in_report": named_in_report,
        "reconciliation": reported,
    }

    problems: list[str] = []
    if code_strict == 0:
        problems.append("analyzer accepted an incomplete root by default")
    if not (named_in_log or named_in_report):
        problems.append("missing cell was not named")
    if reported.get("n_observed") is not None and reported["n_observed"] != n_before - 1:
        problems.append(f"observed {reported['n_observed']}, expected {n_before - 1}")
    if code_allow != 0:
        problems.append(f"--allow-incomplete still exited {code_allow}")

    if problems:
        res.detail = "; ".join(problems)
        return res
    res.status = "PASS"
    res.detail = (
        f"refused with exit {code_strict}; named {expected_cell}; "
        f"{reported.get('n_observed')}/{reported.get('n_expected')} reconciled; "
        f"--allow-incomplete proceeds"
    )
    return res


# ======================================================================
# E7 -- provenance must not pool silently
# ======================================================================


def check_e7(
    smoke_root: Path,
    work: Path,
    methods: Sequence[str],
    variants: Sequence[str],
    logs: Path,
) -> CheckResult:
    """Require the analyzer to refuse a root that pools two provenances."""
    res = CheckResult("E7", "Mixed-provenance root refused, never silently pooled")
    t0 = time.perf_counter()

    fixture = work / "e7_mixed"
    _copy_subset(smoke_root, fixture, methods, FIXTURE_BENCHMARKS)

    # Forge a second provenance on a slice of the runs: a different commit, a
    # different build and a different config hash, exactly what pooling two
    # campaign roots produces.
    poisoned = 0
    for path in sorted(fixture.rglob("run_log.json"))[:20]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload.setdefault("metadata", {})
        hardware = meta.setdefault("hardware", {})
        hardware["git_describe"] = "c0ffee1-other-campaign"
        hardware["build_hash"] = "deadbeefdeadbeef"
        meta["config_sha256"] = "0" * 64
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        poisoned += 1

    code_strict, out_strict = _run(
        _analyze_cmd(fixture, methods, FIXTURE_BENCHMARKS, variants),
        logs / "e7_strict.log",
    )
    code_allow, _ = _run(
        _analyze_cmd(fixture, methods, FIXTURE_BENCHMARKS, variants, allow_mixed_provenance=True),
        logs / "e7_allow.log",
    )

    integrity = fixture / "analysis" / "campaign_integrity.json"
    conflicts: list[str] = []
    non_informative: list[str] = []
    if integrity.is_file():
        payload = json.loads(integrity.read_text(encoding="utf-8"))
        prov = payload.get("provenance") or {}
        conflicts = prov.get("conflicts", [])
        non_informative = prov.get("non_informative_keys", [])

    keys_named = {
        key
        for key in ("git_describe", "build_hash", "config_sha256")
        if any(key in c for c in conflicts)
    }

    res.elapsed_s = time.perf_counter() - t0
    res.evidence = {
        "n_poisoned_runs": poisoned,
        "returncode_strict": code_strict,
        "returncode_allow": code_allow,
        "conflicts": conflicts[:6],
        "conflicting_keys_named": sorted(keys_named),
        "non_informative_keys": non_informative,
    }

    problems: list[str] = []
    if code_strict == 0:
        problems.append("analyzer pooled two provenances without complaint")
    if not conflicts:
        problems.append("no provenance conflict recorded")
    if not keys_named:
        problems.append("conflict did not name the offending key")
    if code_allow != 0:
        problems.append(f"--allow-mixed-provenance still exited {code_allow}")

    if problems:
        res.detail = "; ".join(problems)
        return res
    res.status = "PASS"
    res.detail = (
        f"refused with exit {code_strict}; named {sorted(keys_named)} across "
        f"{poisoned} forged runs; --allow-mixed-provenance proceeds; "
        f"non-informative keys reported: {non_informative}"
    )
    return res


# ======================================================================
# Reporting
# ======================================================================


def write_reports(results: list[CheckResult], out_dir: Path, context: dict[str, Any]) -> None:
    """Write ``stage_e_certification.json`` and its Markdown companion."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_pass = sum(1 for r in results if r.passed)
    verdict = "GO" if n_pass == len(results) else "NO-GO"

    payload = {
        "stage": "E",
        "verdict": verdict,
        "n_pass": n_pass,
        "n_checks": len(results),
        "context": context,
        "checks": [r.to_dict() for r in results],
    }
    (out_dir / "stage_e_certification.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# Stage E certification — analysis dry-run on the pre-flight data",
        "",
        f"**Verdict: {verdict}** — {n_pass}/{len(results)} checks pass.",
        "",
        "| Check | Title | Result | Detail |",
        "|---|---|---|---|",
    ]
    for r in results:
        mark = "PASS" if r.passed else "**FAIL**"
        lines.append(f"| {r.check} | {r.title} | {mark} | {r.detail} |")
    lines += ["", "## Context", ""]
    for key, value in context.items():
        lines.append(f"- **{key}**: `{value}`")
    lines += ["", "## Timings", ""]
    for r in results:
        lines.append(f"- {r.check}: {r.elapsed_s:.1f}s")
    (out_dir / "stage_e_certification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run every Stage E check and emit the certification artefacts."""
    parser = argparse.ArgumentParser(description="Stage E certifier (checks E1-E7)")
    parser.add_argument("--smoke-root", required=True, type=Path, help="Pristine Stage C root")
    parser.add_argument("--work-dir", required=True, type=Path, help="Stage E workspace")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated check ids to run (e.g. 'E3,E6'). Default: all.",
    )
    parser.add_argument(
        "--reuse-main",
        action="store_true",
        help="Reuse an existing e1_main working copy instead of re-copying.",
    )
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    only = {c.strip().upper() for c in args.only.split(",") if c.strip()}

    work = args.work_dir
    logs = work / "artefacts" / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    main_root = work / "e1_main"
    if not args.reuse_main or not main_root.exists():
        log.info("Building the E1 working copy from %s", args.smoke_root)
        if main_root.exists():
            shutil.rmtree(main_root)
        shutil.copytree(args.smoke_root, main_root)

    steps: list[tuple[str, Callable[[], CheckResult]]] = [
        ("E1", lambda: check_e1(main_root, methods, benchmarks, variants, logs)),
        ("E2", lambda: check_e2(main_root, logs)),
        ("E3", lambda: check_e3(args.smoke_root, work, methods, variants, logs)),
        ("E4", lambda: check_e4(main_root, work, methods, benchmarks, logs)),
        ("E5", lambda: check_e5(main_root, work, methods, benchmarks, variants, logs)),
        ("E6", lambda: check_e6(args.smoke_root, work, methods, variants, logs)),
        ("E7", lambda: check_e7(args.smoke_root, work, methods, variants, logs)),
    ]

    results: list[CheckResult] = []
    for name, step in steps:
        if only and name not in only:
            continue
        log.info("── %s ──", name)
        try:
            result = step()
        except Exception as exc:  # noqa: BLE001 - a crashed check is a failed check
            log.exception("%s raised", name)
            result = CheckResult(name, name, detail=f"raised {type(exc).__name__}: {exc}")
        results.append(result)
        log.info("%s: %s — %s", name, result.status, result.detail)

    context = {
        "smoke_root": str(args.smoke_root),
        "work_dir": str(work),
        "methods": methods,
        "benchmarks": benchmarks,
        "variants": variants,
        "fixture_benchmarks": list(FIXTURE_BENCHMARKS),
    }
    write_reports(results, work / "artefacts", context)

    n_pass = sum(1 for r in results if r.passed)
    log.info("Stage E: %d/%d PASS", n_pass, len(results))
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
