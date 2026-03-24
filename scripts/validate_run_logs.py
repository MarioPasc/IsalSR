#!/usr/bin/env python
"""Validate ALL run_log.json files in model_validation results directory.

Checks:
- JSON validity
- Required sections exist
- r2_test is not NaN
- wall_clock_total_s > 0
- total_dags_explored > 0
- Reports breakdowns by method/benchmark/variant
- Specific checks for I.48.20 and previously missing Bingo/Nguyen seeds
"""

import json
import math
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path("/media/mpascual/Sandisk2TB/research/isalsr/results/model_validation")
METHODS = ["bingo", "udfs"]
BENCHMARKS = ["nguyen", "feynman"]
VARIANTS = ["baseline", "isalsr"]

# Expected Nguyen problems
NGUYEN_PROBLEMS = [
    "nguyen_1",
    "nguyen_2",
    "nguyen_3",
    "nguyen_4",
    "nguyen_5",
    "nguyen_6",
    "nguyen_7",
    "nguyen_8",
    "nguyen_9",
    "nguyen_10",
    "nguyen_11",
    "nguyen_12",
]

# Required sections in run_log.json
REQUIRED_KEYS = ["metadata", "results", "best_expression"]
REQUIRED_RESULTS_KEYS = ["regression", "time", "search_space"]


def validate_run_log(filepath: Path) -> dict:
    """Validate a single run_log.json file.

    Returns a dict with validation results.
    """
    result = {
        "path": str(filepath),
        "exists": False,
        "valid_json": False,
        "zero_bytes": False,
        "has_required_sections": False,
        "r2_test_valid": False,
        "r2_test_value": None,
        "wall_clock_valid": False,
        "total_dags_valid": False,
        "errors": [],
    }

    if not filepath.exists():
        result["errors"].append("File does not exist")
        return result

    result["exists"] = True

    # Check zero bytes
    file_size = filepath.stat().st_size
    if file_size == 0:
        result["zero_bytes"] = True
        result["errors"].append("File is 0 bytes")
        return result

    # Parse JSON
    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result["errors"].append(f"Invalid JSON: {e}")
        return result
    except Exception as e:
        result["errors"].append(f"Read error: {e}")
        return result

    result["valid_json"] = True

    # Check required sections
    missing_keys = [k for k in REQUIRED_KEYS if k not in data]
    if missing_keys:
        result["errors"].append(f"Missing top-level keys: {missing_keys}")
    else:
        # Check results sub-sections
        results = data.get("results", {})
        missing_results_keys = [k for k in REQUIRED_RESULTS_KEYS if k not in results]
        if missing_results_keys:
            result["errors"].append(f"Missing results sub-keys: {missing_results_keys}")
        else:
            result["has_required_sections"] = True

    # Check r2_test
    try:
        r2_test = data.get("results", {}).get("regression", {}).get("r2_test")
        if r2_test is None:
            result["errors"].append("r2_test is missing/null")
        elif isinstance(r2_test, (int, float)):
            result["r2_test_value"] = r2_test
            if math.isnan(r2_test):
                result["errors"].append("r2_test is NaN")
            else:
                result["r2_test_valid"] = True
        else:
            result["errors"].append(f"r2_test has unexpected type: {type(r2_test)}")
    except Exception as e:
        result["errors"].append(f"Error checking r2_test: {e}")

    # Check wall_clock_total_s
    try:
        wall_clock = data.get("results", {}).get("time", {}).get("wall_clock_total_s")
        if wall_clock is None:
            result["errors"].append("wall_clock_total_s is missing/null")
        elif isinstance(wall_clock, (int, float)) and wall_clock > 0:
            result["wall_clock_valid"] = True
        else:
            result["errors"].append(f"wall_clock_total_s invalid: {wall_clock}")
    except Exception as e:
        result["errors"].append(f"Error checking wall_clock: {e}")

    # Check total_dags_explored
    try:
        total_dags = data.get("results", {}).get("search_space", {}).get("total_dags_explored")
        if total_dags is None:
            result["errors"].append("total_dags_explored is missing/null")
        elif isinstance(total_dags, (int, float)) and total_dags > 0:
            result["total_dags_valid"] = True
        else:
            result["errors"].append(f"total_dags_explored invalid: {total_dags}")
    except Exception as e:
        result["errors"].append(f"Error checking total_dags: {e}")

    return result


def discover_run_logs(base_dir: Path) -> list[Path]:
    """Find all run_log.json files, excluding debug/ subdirectory."""
    run_logs = []
    for method in METHODS:
        method_dir = base_dir / method
        if not method_dir.exists():
            continue
        for benchmark in BENCHMARKS:
            bench_dir = method_dir / benchmark
            if not bench_dir.exists():
                continue
            for problem_dir in sorted(bench_dir.iterdir()):
                if not problem_dir.is_dir():
                    continue
                # Skip debug directory
                if problem_dir.name == "debug":
                    continue
                for variant in VARIANTS:
                    variant_dir = problem_dir / variant
                    if not variant_dir.exists():
                        continue
                    for seed_dir in sorted(variant_dir.iterdir()):
                        if not seed_dir.is_dir():
                            continue
                        if not seed_dir.name.startswith("seed_"):
                            continue
                        run_log = seed_dir / "run_log.json"
                        run_logs.append(run_log)
    return run_logs


def parse_path(filepath: Path) -> dict:
    """Extract method, benchmark, problem, variant, seed from path."""
    parts = filepath.parts
    # .../model_validation/method/benchmark/problem/variant/seed_XX/run_log.json
    idx = parts.index("model_validation")
    return {
        "method": parts[idx + 1],
        "benchmark": parts[idx + 2],
        "problem": parts[idx + 3],
        "variant": parts[idx + 4],
        "seed": parts[idx + 5],
    }


def main() -> None:
    print("=" * 80)
    print("RUN_LOG.JSON VALIDATION REPORT")
    print(f"Base directory: {BASE_DIR}")
    print("Date: 2026-03-24")
    print("=" * 80)

    # Discover all run_log.json files
    run_logs = discover_run_logs(BASE_DIR)
    print(f"\nTotal run_log.json files discovered: {len(run_logs)}")

    # Validate all
    results = []
    for rl in run_logs:
        res = validate_run_log(rl)
        res.update(parse_path(rl))
        results.append(res)

    # ---- Summary statistics ----
    total = len(results)
    existing = sum(1 for r in results if r["exists"])
    valid_json = sum(1 for r in results if r["valid_json"])
    zero_bytes = sum(1 for r in results if r["zero_bytes"])
    has_sections = sum(1 for r in results if r["has_required_sections"])
    r2_valid = sum(1 for r in results if r["r2_test_valid"])
    wall_valid = sum(1 for r in results if r["wall_clock_valid"])
    dags_valid = sum(1 for r in results if r["total_dags_valid"])

    # Fully valid = all checks pass
    fully_valid = sum(
        1
        for r in results
        if r["valid_json"]
        and r["has_required_sections"]
        and r["r2_test_valid"]
        and r["wall_clock_valid"]
        and r["total_dags_valid"]
    )

    print("\n" + "-" * 60)
    print("OVERALL SUMMARY")
    print("-" * 60)
    print(f"  Total files discovered:     {total}")
    print(f"  Files existing:             {existing}")
    print(f"  Valid JSON:                 {valid_json}")
    print(f"  Zero-byte files:            {zero_bytes}")
    print(f"  Has required sections:      {has_sections}")
    print(f"  r2_test valid (non-NaN):    {r2_valid}")
    print(f"  wall_clock_total_s > 0:     {wall_valid}")
    print(f"  total_dags_explored > 0:    {dags_valid}")
    print(f"  FULLY VALID:                {fully_valid}")
    print(f"  INVALID:                    {total - fully_valid}")

    # ---- NaN r2_test values ----
    nan_r2 = [r for r in results if r["valid_json"] and not r["r2_test_valid"]]
    print("\n" + "-" * 60)
    print(f"FILES WITH NaN / MISSING r2_test ({len(nan_r2)} files)")
    print("-" * 60)
    if nan_r2:
        for r in nan_r2:
            print(f"  {r['method']}/{r['benchmark']}/{r['problem']}/{r['variant']}/{r['seed']}")
            for err in r["errors"]:
                if "r2_test" in err:
                    print(f"    -> {err}")
    else:
        print("  None! All r2_test values are valid.")

    # ---- Zero-byte / corrupt files ----
    corrupt = [r for r in results if r["zero_bytes"] or (r["exists"] and not r["valid_json"])]
    print("\n" + "-" * 60)
    print(f"ZERO-BYTE OR CORRUPT FILES ({len(corrupt)} files)")
    print("-" * 60)
    if corrupt:
        for r in corrupt:
            print(f"  {r['path']}")
            for err in r["errors"]:
                print(f"    -> {err}")
    else:
        print("  None! All files are readable and valid JSON.")

    # ---- Missing files ----
    missing = [r for r in results if not r["exists"]]
    print("\n" + "-" * 60)
    print(f"MISSING FILES ({len(missing)} files)")
    print("-" * 60)
    if missing:
        for r in missing:
            print(f"  {r['path']}")
    else:
        print("  None! All expected files exist.")

    # ---- Files with any errors ----
    errored = [r for r in results if r["errors"]]
    print("\n" + "-" * 60)
    print(f"ALL FILES WITH ERRORS ({len(errored)} files)")
    print("-" * 60)
    if errored:
        for r in errored:
            print(f"  {r['method']}/{r['benchmark']}/{r['problem']}/{r['variant']}/{r['seed']}")
            for err in r["errors"]:
                print(f"    -> {err}")
    else:
        print("  None! All files pass all validation checks.")

    # ---- Breakdown by method / benchmark / variant ----
    print("\n" + "-" * 60)
    print("BREAKDOWN BY METHOD / BENCHMARK / VARIANT")
    print("-" * 60)
    breakdown = defaultdict(lambda: {"total": 0, "valid": 0, "invalid": 0})
    for r in results:
        key = f"{r['method']:>5s} / {r['benchmark']:>7s} / {r['variant']:>8s}"
        breakdown[key]["total"] += 1
        if (
            r["valid_json"]
            and r["has_required_sections"]
            and r["r2_test_valid"]
            and r["wall_clock_valid"]
            and r["total_dags_valid"]
        ):
            breakdown[key]["valid"] += 1
        else:
            breakdown[key]["invalid"] += 1

    print(f"  {'Method / Benchmark / Variant':<40s} {'Total':>6s} {'Valid':>6s} {'Invalid':>8s}")
    print(f"  {'-' * 40} {'-' * 6} {'-' * 6} {'-' * 8}")
    for key in sorted(breakdown.keys()):
        v = breakdown[key]
        print(f"  {key:<40s} {v['total']:>6d} {v['valid']:>6d} {v['invalid']:>8d}")

    # ---- Detailed breakdown by problem ----
    print("\n" + "-" * 60)
    print("DETAILED BREAKDOWN BY METHOD / BENCHMARK / PROBLEM / VARIANT")
    print("-" * 60)
    detail = defaultdict(lambda: {"total": 0, "valid": 0, "seeds": []})
    for r in results:
        key = (r["method"], r["benchmark"], r["problem"], r["variant"])
        detail[key]["total"] += 1
        is_valid = (
            r["valid_json"]
            and r["has_required_sections"]
            and r["r2_test_valid"]
            and r["wall_clock_valid"]
            and r["total_dags_valid"]
        )
        if is_valid:
            detail[key]["valid"] += 1
        detail[key]["seeds"].append(r["seed"])

    print(
        f"  {'Method':<6s} {'Bench':<8s} {'Problem':<15s} {'Variant':<9s} {'Total':>5s} {'Valid':>5s}"
    )
    print(f"  {'-' * 6} {'-' * 8} {'-' * 15} {'-' * 9} {'-' * 5} {'-' * 5}")
    for key in sorted(detail.keys()):
        v = detail[key]
        flag = " <-- ISSUE" if v["valid"] < v["total"] else ""
        print(
            f"  {key[0]:<6s} {key[1]:<8s} {key[2]:<15s} {key[3]:<9s} {v['total']:>5d} {v['valid']:>5d}{flag}"
        )

    # ---- SPECIFIC CHECK: I.48.20 ----
    print("\n" + "-" * 60)
    print("SPECIFIC CHECK: I.48.20 (previously had NaN r2_test)")
    print("-" * 60)
    i4820 = [r for r in results if r["problem"] == "i.48.20"]
    if not i4820:
        print("  WARNING: No runs found for I.48.20!")
    else:
        print(f"  Total runs for I.48.20: {len(i4820)}")
        valid_i4820 = sum(1 for r in i4820 if r["r2_test_valid"])
        nan_i4820 = [r for r in i4820 if r["valid_json"] and not r["r2_test_valid"]]
        print(f"  Valid r2_test: {valid_i4820}")
        print(f"  NaN/missing r2_test: {len(nan_i4820)}")
        if nan_i4820:
            for r in nan_i4820:
                print(
                    f"    {r['method']}/{r['variant']}/{r['seed']} -> r2_test={r['r2_test_value']}"
                )
        else:
            print("  All I.48.20 runs have valid (non-NaN) r2_test values.")

    # ---- SPECIFIC CHECK: Previously missing Bingo/Nguyen seeds ----
    print("\n" + "-" * 60)
    print("SPECIFIC CHECK: Previously missing Bingo/Nguyen seeds (7 expected)")
    print("-" * 60)
    # Count seeds per Bingo/Nguyen problem/variant
    bingo_nguyen = defaultdict(list)
    for r in results:
        if r["method"] == "bingo" and r["benchmark"] == "nguyen":
            key = (r["problem"], r["variant"])
            bingo_nguyen[key].append(r)

    # Check which have fewer than 30 seeds or have any invalid
    problems_with_issues = []
    total_bingo_nguyen = 0
    for key in sorted(bingo_nguyen.keys()):
        runs = bingo_nguyen[key]
        total_bingo_nguyen += len(runs)
        valid_count = sum(1 for r in runs if r["r2_test_valid"])
        if len(runs) < 30 or valid_count < len(runs):
            problems_with_issues.append((key, len(runs), valid_count))

    print(f"  Total Bingo/Nguyen runs: {total_bingo_nguyen}")
    expected_bingo_nguyen = 12 * 2 * 30  # 12 problems * 2 variants * 30 seeds
    print(f"  Expected: {expected_bingo_nguyen} (12 problems x 2 variants x 30 seeds)")

    if problems_with_issues:
        print("  Problems with incomplete or invalid seeds:")
        for (prob, var), total_count, valid_count in problems_with_issues:
            print(f"    {prob}/{var}: {total_count} runs, {valid_count} valid")
    else:
        print("  All Bingo/Nguyen problem/variant combinations have full seed sets.")

    # Show seed counts per problem
    print("\n  Bingo/Nguyen seed count per problem/variant:")
    print(f"  {'Problem':<15s} {'Variant':<9s} {'Seeds':>5s} {'Valid':>5s}")
    print(f"  {'-' * 15} {'-' * 9} {'-' * 5} {'-' * 5}")
    for key in sorted(bingo_nguyen.keys()):
        runs = bingo_nguyen[key]
        valid_count = sum(1 for r in runs if r["r2_test_valid"])
        flag = " <--" if len(runs) < 30 or valid_count < len(runs) else ""
        print(f"  {key[0]:<15s} {key[1]:<9s} {len(runs):>5d} {valid_count:>5d}{flag}")

    # ---- Expected total breakdown ----
    print("\n" + "-" * 60)
    print("EXPECTED vs ACTUAL TOTAL")
    print("-" * 60)
    # Count unique problems per method/benchmark
    problems_seen = defaultdict(set)
    for r in results:
        problems_seen[(r["method"], r["benchmark"])].add(r["problem"])

    for key in sorted(problems_seen.keys()):
        print(
            f"  {key[0]}/{key[1]}: {len(problems_seen[key])} problems: {sorted(problems_seen[key])}"
        )

    # Calculate expected
    # Each cell = n_problems * n_variants * n_seeds
    print("\n  Expected breakdown:")
    total_expected = 0
    for method in METHODS:
        for bench in BENCHMARKS:
            n_probs = len(problems_seen.get((method, bench), set()))
            n_runs = sum(1 for r in results if r["method"] == method and r["benchmark"] == bench)
            n_valid = sum(
                1
                for r in results
                if r["method"] == method and r["benchmark"] == bench and r["r2_test_valid"]
            )
            total_expected += n_runs
            print(f"    {method}/{bench}: {n_probs} problems, {n_runs} total runs, {n_valid} valid")

    print(f"\n  Grand total discovered: {total}")
    print(f"  Grand total fully valid: {fully_valid}")
    print(f"  Grand total with issues: {total - fully_valid}")

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Discovered:    {total} run_log.json files")
    print(f"  Fully valid:   {fully_valid} ({100 * fully_valid / total:.1f}%)" if total > 0 else "")
    print(f"  Invalid:       {total - fully_valid}")
    if total - fully_valid == 0:
        print("  STATUS: ALL CLEAR - Every run_log.json passes all validation checks.")
    else:
        print(f"  STATUS: {total - fully_valid} files need attention.")


if __name__ == "__main__":
    main()
