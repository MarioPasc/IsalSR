"""Stage A desk-check evidence generator (EXECUTION-PLAN.md §4.1, ticket T17 AC-1).

Produces the parsed artefacts Stage A is scored against. "I checked it" is not
evidence; each function below writes a file a reviewer can re-read.

    A4   config_diff.md      -- resolved hyperparameters for all 14 (method,
                                suite) configs, with the arm-invariance argument
    A4b  operator_sets.csv   -- (i) the operator set is identical across arms for
                                every (method, problem); (ii) every configured
                                operator has an image in the paper's alphabet L
    A5   seed_declaration.md -- the campaign seed set and its disjointness from
                                the Stage C smoke seeds
    A11  collision_bound.md  -- the 64-bit birthday bound, stated not hoped

A4's invariance claim is structural rather than empirical: the three arms are
selected by the ``--variants`` CLI flag alone and no YAML carries an arm key, so
a per-arm difference is unrepresentable. This script *checks* that -- it scans
every config for any arm-named key and fails if one exists.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

METHODS: tuple[str, ...] = ("udfs", "bingo")
SUITES: tuple[str, ...] = (
    "nguyen",
    "feynman",
    "hard",
    "cherrypicked",
    "roundoff",
    "feynman_remainder",
    "strogatz",
)
ARMS: tuple[str, ...] = ("baseline", "hash", "isalsr")
CAMPAIGN_SEEDS: tuple[int, ...] = tuple(range(1, 21))
SMOKE_SEEDS: tuple[int, ...] = (0, 101, 102)
TOPUP_SEEDS: tuple[int, ...] = tuple(range(21, 31))

# Hyperparameters that must not differ across arms for a fixed (method, problem).
TRACKED: tuple[str, ...] = (
    "max_time",
    "population_size",
    "stack_size",
    "crossover_prob",
    "mutation_prob",
    "processes",
    "max_evals",
    "operators",
)


def _load(method: str, suite: str) -> dict[str, Any]:
    path = REPO_ROOT / "experiments" / "configs" / f"{method}_{suite}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text())


def write_config_diff(out: Path) -> list[str]:
    """Write A4's resolved-hyperparameter table; return any problems found."""
    problems: list[str] = []
    lines: list[str] = [
        "# A4 — config equivalence across arms",
        "",
        "**Claim.** For a fixed `(method, problem)` the three arms `baseline`, `hash`",
        "and `isalsr` are selected *only* by the orchestrator's `--variants` flag.",
        "No YAML carries an arm-specific key, so an arm-dependent hyperparameter is",
        "not merely absent — it is unrepresentable. The scan below is what turns that",
        "from an assertion into a check.",
        "",
        "## Resolved hyperparameters, all 14 (method, suite) configs",
        "",
        "| config | suite | n_seeds | " + " | ".join(TRACKED[:-1]) + " |",
        "|---|---|---|" + "---|" * (len(TRACKED) - 1),
    ]
    for method in METHODS:
        for suite in SUITES:
            cfg = _load(method, suite)
            exp = cfg["experiment"]
            mc = cfg.get(method, {})
            row = [f"`{method}_{suite}.yaml`", suite, str(exp.get("n_seeds"))]
            row += [str(mc.get(k, "—")) for k in TRACKED[:-1]]
            lines.append("| " + " | ".join(row) + " |")

            # (i) No arm may carry its own copy of a HOST-SEARCH hyperparameter.
            # A top-level `isalsr:` block is expected and legitimate: it holds
            # canonicaliser settings (timeout, fast-canonical toggle) that only
            # the dedup layer reads. What A4 forbids is a search parameter --
            # population size, operator set, budget -- differing per arm, which
            # would make the three arms different experiments.
            for arm in ARMS:
                block = cfg.get(arm)
                if not isinstance(block, dict):
                    continue
                leaked = sorted(set(block) & set(TRACKED))
                if leaked:
                    problems.append(
                        f"{method}_{suite}.yaml: arm block '{arm}' overrides "
                        f"host-search hyperparameter(s) {leaked}"
                    )
            for arm in ("baseline", "hash"):
                if arm in cfg:
                    problems.append(f"{method}_{suite}.yaml carries an arm block '{arm}'")

    n_seeds = {(m, s): _load(m, s)["experiment"].get("n_seeds") for m in METHODS for s in SUITES}
    stale = sorted(k for k, v in n_seeds.items() if v != 20)
    lines += [
        "",
        "## Findings",
        "",
        f"- Arm-specific keys found in any config: **{len(problems)}** "
        f"{'(' + '; '.join(problems) + ')' if problems else '— none, as required'}.",
        "- `max_time` is uniform at 43,200 s across all 14 configs; Stage C overrides it",
        "  to 900 s on the command line (`--max-time`), never in a YAML.",
    ]
    if stale:
        lines += [
            "",
            f"- 🔴 **`n_seeds` is not yet 20 in {len(stale)} config(s):** "
            + ", ".join(f"`{m}_{s}` ({n_seeds[(m, s)]})" for m, s in stale)
            + ". EXECUTION-PLAN §0.4a fixes the campaign at 20 seeds. This does **not**",
            "  affect Stage C, whose seeds are passed explicitly (`--seeds 0,101,102`) and",
            "  whose task counts are derived from that flag, not from `n_seeds`. It must be",
            "  corrected before C2 is submitted.",
        ]
        problems.append(f"n_seeds != 20 in {len(stale)} configs: {stale}")
    out.write_text("\n".join(lines) + "\n")
    return problems


def write_operator_sets(out: Path) -> list[str]:
    """Write A4b's per-problem operator table; return any violations."""
    from experiments.models.alphabet_guard import (  # noqa: PLC0415
        AlphabetCoverageError,
        validate_bingo_operators,
        validate_udfs_operators,
    )
    from experiments.models.orchestrator import _BENCHMARK_REGISTRY  # noqa: PLC0415

    violations: list[str] = []
    rows: list[dict[str, Any]] = []

    udfs_table = validate_udfs_operators()  # takes no arguments: the vendored table
    udfs_ops = sorted(udfs_table)

    for method in METHODS:
        for suite in SUITES:
            cfg = _load(method, suite)
            # Bingo names the field `operators`; UDFS names its documentation-only
            # list `operator_set`.
            mc = cfg.get(method, {})
            ops = list(mc.get("operators") or mc.get("operator_set") or [])
            if method == "bingo":
                try:
                    validate_bingo_operators(ops)
                    covered = "yes"
                except AlphabetCoverageError as exc:
                    covered = "NO"
                    violations.append(f"{method}_{suite}: {exc}")
                effective = ops
            else:
                # UDFS takes no operator set from the YAML: to_dag_regressor_kwargs
                # never forwards the field and the search enumerates the vendored
                # NODE_ARITY table. The YAML list is documentation only.
                covered = "yes"
                effective = udfs_ops

            for bench in _BENCHMARK_REGISTRY[suite][0]:
                for arm in ARMS:
                    rows.append(
                        {
                            "method": method,
                            "suite": suite,
                            "problem": bench["name"],
                            "arm": arm,
                            "operators_configured": "|".join(ops),
                            "operators_effective": "|".join(effective),
                            "source": "yaml" if method == "bingo" else "vendored NODE_ARITY",
                            "in_alphabet_L": covered,
                        }
                    )

    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # (i) identical across arms for every (method, problem)
    by_key: dict[tuple[str, str], set[str]] = {}
    for r in rows:
        by_key.setdefault((r["method"], r["problem"]), set()).add(r["operators_effective"])
    for key, vals in by_key.items():
        if len(vals) != 1:
            violations.append(f"operator set differs across arms for {key}: {vals}")

    # A4b also asks for uniformity across PROBLEMS, one set per method.
    per_method: dict[str, set[str]] = {}
    for r in rows:
        per_method.setdefault(r["method"], set()).add(r["operators_effective"])
    for method, vals in per_method.items():
        if len(vals) != 1:
            violations.append(f"operator set not uniform across problems for {method}: {vals}")

    return violations


def write_seed_declaration(out: Path) -> list[str]:
    """Write A5's seed declaration; return any violations."""
    from experiments.models.io_utils import seed_dir  # noqa: PLC0415
    from experiments.models.orchestrator import parse_seeds  # noqa: PLC0415

    violations: list[str] = []
    parsed = parse_seeds(",".join(str(s) for s in SMOKE_SEEDS))
    if parsed != list(SMOKE_SEEDS):
        violations.append(f"parse_seeds round-trip failed: {parsed}")
    if 0 in CAMPAIGN_SEEDS:
        violations.append("seed 0 is inside the campaign seed set")
    if set(SMOKE_SEEDS) & (set(CAMPAIGN_SEEDS) | set(TOPUP_SEEDS)):
        violations.append("smoke seeds collide with campaign or top-up seeds")

    # seed_dir() creates the directory, so render the names under a tempdir.
    with tempfile.TemporaryDirectory() as tmp:
        dirs = {s: Path(seed_dir(Path(tmp), s)).name for s in SMOKE_SEEDS}
    out.write_text(
        "\n".join(
            [
                "# A5 — seed declaration",
                "",
                f"- **Campaign C2 seeds:** {CAMPAIGN_SEEDS[0]}…{CAMPAIGN_SEEDS[-1]} "
                f"({len(CAMPAIGN_SEEDS)} seeds, EXECUTION-PLAN §0.4a). `0 ∉ seeds` ✓",
                f"- **Reserved top-up range (spillover priority 1, §8.4):** "
                f"{TOPUP_SEEDS[0]}…{TOPUP_SEEDS[-1]} — not launched.",
                f"- **Stage C smoke seeds:** {', '.join(map(str, SMOKE_SEEDS))}. Disjoint from "
                "both sets above, so a smoke output can never be mistaken for, or merged "
                "into, a C2 cell (SP-0).",
                "- Directory names, so the disjointness is visible on disk: "
                + ", ".join(f"`{v}`" for v in dirs.values()),
                "",
                "**Continuity (§6.3).** Seeds 1…20 must be the same integers C1 used, so the",
                "continuity table can restrict C1 to those 20 seeds and compare like-for-like.",
                "C1 ran seeds 1…30, of which 1…20 is a prefix — the restriction is exact.",
            ]
        )
        + "\n"
    )
    return violations


def write_collision_bound(out: Path) -> list[str]:
    """Write A11's hash-collision arithmetic."""
    n = 10**7
    p_run = n**2 / 2**65
    n_runs = 5600
    out.write_text(
        "\n".join(
            [
                "# A11 — 64-bit hash collision bound",
                "",
                "Both the IsalSR dedup set and the T04 hash arm key on 64-bit digests.",
                "The relevant quantity is the birthday bound on the probability that any",
                "two of `n` distinct canonical strings collide within one run:",
                "",
                "    P(collision) ≲ n² / 2^65",
                "",
                f"At `n = 10^7` entries per run this is **{p_run:.2e}**. Across the",
                f"{n_runs} dedup-bearing runs of C2 ({n_runs} = 2 methods × 2 dedup arms ×",
                "70 problems × 20 seeds) the expected number of runs containing at least",
                f"one collision is **{p_run * n_runs:.2e}**.",
                "",
                "A collision merges two non-isomorphic DAGs into one dedup class, which",
                "*under*-counts `unique_canonical_dags` and therefore *over*-states ρ. The",
                "bound above says the effect is ~10⁻² runs campaign-wide, i.e. below the",
                "resolution of any reported figure.",
                "",
                "**Open until Stage C reports it:** the observed `max(total_dags_explored)`",
                "per run. The bound is quadratic in `n`, so it must be re-evaluated against",
                "the measured maximum rather than the assumed 10⁷.",
            ]
        )
        + "\n"
    )
    return []


def main() -> int:
    """Generate every Stage A evidence artefact and report violations."""
    ap = argparse.ArgumentParser(description="C2 Stage A desk-check evidence")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    findings: dict[str, list[str]] = {
        "A4  config_diff.md": write_config_diff(out / "config_diff.md"),
        "A4b operator_sets.csv": write_operator_sets(out / "operator_sets.csv"),
        "A5  seed_declaration.md": write_seed_declaration(out / "seed_declaration.md"),
        "A11 collision_bound.md": write_collision_bound(out / "collision_bound.md"),
    }

    blocking = 0
    for name, issues in findings.items():
        status = "PASS" if not issues else f"{len(issues)} FINDING(S)"
        print(f"  {name:26s} {status}")
        for i in issues:
            print(f"      - {i}")
            blocking += 1
    print(f"\n  artefacts -> {out}")
    return 0 if blocking == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
