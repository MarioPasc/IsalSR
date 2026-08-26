"""Search Space Permutation Analysis -- Validates the O(k!) reduction claim.

For each labeled DAG with k internal nodes, this experiment creates all k!
(or a random sample of) permutations of internal node IDs, computes the
greedy D2S string for each permutation, and verifies that the canonical
string is invariant across all permutations.

The central claim of the IsalSR paper is that canonicalization collapses
O(k!) equivalent labelings into a single canonical string. This experiment
provides direct empirical evidence for that claim by measuring:

    1. n_distinct_d2s: The number of distinct greedy D2S strings produced
       across all k! (or sampled) permutations. This is the empirical
       reduction factor.
    2. invariant_success_rate: The fraction of permutations for which the
       canonical string matches the original DAG's canonical string.
       This should be 1.0 for a correct complete invariant.
    3. normalized_ratio: n_distinct_d2s / k!, which measures what fraction
       of the theoretical search space is actually explored.

Algorithm per DAG:
    1. Compute canonical string of the original DAG.
    2. Determine exhaustive (k! <= n_perms_sample) vs sampled mode.
    3. Generate all k! or n_perms_sample random permutations.
    4. For each permutation: permute internal nodes, compute greedy D2S,
       collect distinct strings.
    5. Verify canonical invariance on a subset of permutations.
    6. Record metrics row.

Usage:
    python experiments/scripts/search_space_permutation_analysis.py \\
        --output results.csv --k-value 5 --n-dags 100 \\
        --n-perms-sample 100000 --n-canon-verify 100 \\
        --num-vars 1 --seed 42 --canon-timeout 5.0 --include-benchmarks

Author: Mario Pascual Gonzalez (mpascual@uma.es)
Date: 2026-03-21

References:
    - Lopez-Rubio (2025). arXiv:2512.10429v2. IsalGraph.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import math
import os
import random
import sys
import time

# Ensure project root is on the path for non-installed runs.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.scripts._dag_generators import make_random_sr_dag
from isalsr.core.canonical import CanonicalTimeoutError, pruned_canonical_string
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.permutations import permute_internal_nodes, random_permutations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ======================================================================
# CSV field names
# ======================================================================

FIELDNAMES = [
    "k",
    "m",
    "dag_idx",
    "source",
    "n_nodes",
    "n_edges",
    "canonical_string",
    "n_perms_tested",
    "is_exhaustive",
    "n_distinct_representations",
    "n_distinct_d2s",
    "n_distinct_canonical",
    "reduction_factor",
    "theoretical_k_factorial",
    "normalized_ratio",
    "invariant_success_rate",
    "mean_d2s_time_us",
    "mean_canon_time_ms",
]


# ======================================================================
# Single DAG analysis
# ======================================================================


def analyze_single_dag(
    dag: LabeledDAG,
    dag_idx: int,
    source: str,
    k: int,
    m: int,
    n_perms_sample: int,
    n_canon_verify: int,
    canon_timeout: float,
    rng: random.Random,
) -> dict[str, object]:
    """Analyze a single DAG across all (or sampled) permutations of its internal nodes.

    For each permutation, computes the greedy D2S string and collects the set
    of distinct strings. Then verifies canonical invariance on a subset.

    Args:
        dag: The labeled DAG to analyze.
        dag_idx: Index of this DAG in the experiment (for logging/output).
        source: Origin label for the DAG (e.g. "random", "nguyen").
        k: Number of internal nodes.
        m: Number of variable nodes.
        n_perms_sample: Maximum number of permutations to test. If k! <= this,
            all permutations are tested exhaustively.
        n_canon_verify: Number of permutations on which to verify canonical
            invariance (subset of tested permutations).
        canon_timeout: Timeout in seconds for each canonical string computation.
        rng: Random number generator for reproducible sampling.

    Returns:
        Dictionary with one row of results (see FIELDNAMES).
    """
    # 1. Compute canonical of original.
    canon_original = pruned_canonical_string(dag, timeout=canon_timeout)

    # 2. Determine exhaustive vs sampled.
    k_factorial = math.factorial(k)
    is_exhaustive = k_factorial <= n_perms_sample

    # 3. Generate permutations.
    if is_exhaustive:
        perms: list[list[int]] = [list(p) for p in itertools.permutations(range(k))]
    else:
        perms = random_permutations(k, n_perms_sample, rng)

    # 4. For each permutation: permute, compute structural fingerprint + D2S.
    # Structural fingerprint: encodes adjacency+labels+order, sensitive to
    # node IDs. n_distinct_fingerprints = k! / |Aut(D)| (exact).
    # Greedy D2S: conservative lower bound (many permutations produce the
    # same greedy D2S due to structural determinism).
    d2s_set: set[str] = set()
    fingerprint_set: set[tuple[tuple[int, tuple[int, ...]], ...]] = set()
    d2s_times: list[int] = []  # nanoseconds
    for p in perms:
        dag_p = permute_internal_nodes(dag, p)

        # Structural fingerprint (fast, O(n+e))
        fp: list[tuple[int, tuple[int, ...]]] = []
        for v in range(dag_p.node_count):
            fp.append((dag_p.node_label(v).value, tuple(dag_p.ordered_inputs(v))))
        fingerprint_set.add(tuple(fp))

        # Greedy D2S
        t0 = time.perf_counter_ns()
        d2s_str = DAGToString(dag_p, initial_node=0).run()
        d2s_times.append(time.perf_counter_ns() - t0)
        d2s_set.add(d2s_str)

    # 5. Canonical verification on subset.
    n_verify = min(n_canon_verify, len(perms))
    verify_indices = rng.sample(range(len(perms)), n_verify)
    canon_match = 0
    canon_times: list[float] = []  # seconds
    canon_timeouts = 0
    for idx in verify_indices:
        dag_p = permute_internal_nodes(dag, perms[idx])
        t0 = time.monotonic()
        try:
            canon_p = pruned_canonical_string(dag_p, timeout=canon_timeout)
            canon_times.append(time.monotonic() - t0)
            if canon_p == canon_original:
                canon_match += 1
        except CanonicalTimeoutError:
            canon_times.append(canon_timeout)
            canon_timeouts += 1

    # Determine n_distinct_canonical:
    #   1 if all verified permutations match (invariant holds),
    #  -1 if any mismatch or timeout was encountered.
    if canon_timeouts > 0:
        n_distinct_canonical = -1
    elif canon_match == n_verify:
        n_distinct_canonical = 1
    else:
        n_distinct_canonical = -1

    # 6. Return row dict.
    mean_d2s_ns = sum(d2s_times) / max(len(d2s_times), 1)
    mean_canon_s = sum(canon_times) / max(len(canon_times), 1)

    return {
        "k": k,
        "m": m,
        "dag_idx": dag_idx,
        "source": source,
        "n_nodes": dag.node_count,
        "n_edges": dag.edge_count,
        "canonical_string": canon_original,
        "n_perms_tested": len(perms),
        "is_exhaustive": is_exhaustive,
        "n_distinct_representations": len(fingerprint_set),
        "n_distinct_d2s": len(d2s_set),
        "n_distinct_canonical": n_distinct_canonical,
        "reduction_factor": len(fingerprint_set),
        "theoretical_k_factorial": k_factorial,
        "normalized_ratio": len(fingerprint_set) / k_factorial if k_factorial > 0 else 1.0,
        "invariant_success_rate": canon_match / max(n_verify, 1),
        "mean_d2s_time_us": mean_d2s_ns / 1000,  # ns -> us
        "mean_canon_time_ms": mean_canon_s * 1000,  # s -> ms
    }


# ======================================================================
# DAG generation
# ======================================================================


def generate_source_dags(
    k: int,
    m: int,
    n_dags: int,
    seed: int,
    include_benchmarks: bool,
    canon_timeout: float,
) -> list[tuple[LabeledDAG, str]]:
    """Generate a set of structurally unique random DAGs for analysis.

    Deduplicates by canonical string to ensure each DAG represents a
    distinct isomorphism class. Uses up to n_dags * 100 attempts to
    generate the requested number of unique DAGs.

    Args:
        k: Number of internal nodes per DAG.
        m: Number of input variables per DAG.
        n_dags: Target number of unique DAGs.
        seed: Base random seed for DAG generation.
        include_benchmarks: If True, include Nguyen benchmark DAGs with
            matching k (currently a no-op; see TODO below).
        canon_timeout: Timeout for canonical string computation during
            deduplication.

    Returns:
        List of (dag, source_label) tuples.
    """
    dags: list[tuple[LabeledDAG, str]] = []
    seen_canonicals: set[str] = set()
    attempt = 0
    max_attempts = n_dags * 100

    while len(dags) < n_dags and attempt < max_attempts:
        attempt += 1
        dag = make_random_sr_dag(m, k, seed=seed + attempt)
        try:
            canon = pruned_canonical_string(dag, timeout=canon_timeout)
        except CanonicalTimeoutError:
            continue
        except Exception:
            continue
        if canon not in seen_canonicals:
            seen_canonicals.add(canon)
            dags.append((dag, "random"))

    if len(dags) < n_dags:
        log.warning(
            "Only generated %d/%d unique DAGs after %d attempts (k=%d, m=%d)",
            len(dags),
            n_dags,
            max_attempts,
            k,
            m,
        )

    # TODO: Integrate Nguyen benchmark DAGs when --include-benchmarks is set.
    # The NGUYEN_BENCHMARKS entries have 'target_fn' (callable) but not sympy
    # expressions. To include them, we would need to:
    #   1. Define the Nguyen expressions as sympy objects.
    #   2. Use SympyAdapter.from_sympy() to convert to LabeledDAG.
    #   3. Filter to those whose internal node count matches k.
    # For now, --include-benchmarks is accepted but has no effect.
    if include_benchmarks:
        log.info(
            "Note: --include-benchmarks is currently a no-op. "
            "Nguyen benchmark DAG integration is pending sympy expression definitions."
        )

    return dags


# ======================================================================
# CSV I/O
# ======================================================================


def write_csv(rows: list[dict[str, object]], output_path: str) -> None:
    """Write analysis results to CSV.

    Args:
        rows: List of row dictionaries with keys from FIELDNAMES.
        output_path: Path to the output CSV file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote %d rows to %s", len(rows), output_path)


# ======================================================================
# Summary
# ======================================================================


def _print_summary(rows: list[dict[str, object]]) -> None:
    """Log a summary table of the permutation analysis results.

    Args:
        rows: List of row dictionaries from the analysis.
    """
    log.info("")
    log.info("=" * 110)
    log.info("PERMUTATION ANALYSIS SUMMARY")
    log.info("=" * 110)
    header = (
        f"{'idx':>4} {'k':>3} {'m':>3} {'src':>8} "
        f"{'nodes':>5} {'edges':>5} {'perms':>8} {'exh':>4} "
        f"{'d2s':>8} {'repr':>8} {'k!':>8} {'ratio':>8} "
        f"{'inv_rate':>8} {'d2s_us':>10} {'canon_ms':>10}"
    )
    log.info(header)
    log.info("-" * len(header))

    for row in rows:
        log.info(
            f"{row['dag_idx']:>4} {row['k']:>3} {row['m']:>3} {row['source']:>8} "
            f"{row['n_nodes']:>5} {row['n_edges']:>5} {row['n_perms_tested']:>8} "
            f"{'Y' if row['is_exhaustive'] else 'N':>4} "
            f"{row['n_distinct_d2s']:>8} {row['n_distinct_representations']:>8} "
            f"{row['theoretical_k_factorial']:>8} "
            f"{row['normalized_ratio']:>8.4f} "
            f"{row['invariant_success_rate']:>8.4f} "
            f"{row['mean_d2s_time_us']:>10.1f} {row['mean_canon_time_ms']:>10.2f}"
        )

    # Overall aggregates.
    n_total = len(rows)
    if n_total == 0:
        log.info("No data.")
        return

    n_invariant_perfect = sum(1 for r in rows if r["invariant_success_rate"] == 1.0)
    mean_reduction = (
        sum(
            float(r["n_distinct_d2s"])
            for r in rows  # type: ignore[arg-type]
        )
        / n_total
    )
    mean_ratio = (
        sum(
            float(r["normalized_ratio"])
            for r in rows  # type: ignore[arg-type]
        )
        / n_total
    )
    mean_d2s = (
        sum(
            float(r["mean_d2s_time_us"])
            for r in rows  # type: ignore[arg-type]
        )
        / n_total
    )
    mean_canon = (
        sum(
            float(r["mean_canon_time_ms"])
            for r in rows  # type: ignore[arg-type]
        )
        / n_total
    )
    max_distinct = max(
        int(r["n_distinct_d2s"])
        for r in rows  # type: ignore[arg-type]
    )
    min_distinct = min(
        int(r["n_distinct_d2s"])
        for r in rows  # type: ignore[arg-type]
    )

    k_factorial = int(rows[0]["theoretical_k_factorial"])  # type: ignore[arg-type]

    log.info("")
    log.info("OVERALL (%d DAGs, k=%s):", n_total, rows[0]["k"])
    log.info("  Invariant perfect (rate=1.0): %d / %d", n_invariant_perfect, n_total)
    log.info("  Mean distinct D2S strings:    %.1f", mean_reduction)
    log.info("  Min/Max distinct D2S:         %d / %d", min_distinct, max_distinct)
    log.info("  k! (theoretical max):         %d", k_factorial)
    log.info("  Mean normalized ratio:        %.4f", mean_ratio)
    log.info("  Mean D2S time:                %.1f us", mean_d2s)
    log.info("  Mean canonical time:          %.2f ms", mean_canon)
    log.info("")


# ======================================================================
# CLI
# ======================================================================


DEFAULT_OUTPUT = "/media/mpascual/Sandisk2TB/research/isalsr/results/search_space_permutation.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Search space permutation analysis: validates the O(k!) reduction "
            "claim by computing D2S strings for all (or sampled) permutations "
            "of internal node IDs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--k-value",
        type=int,
        default=5,
        help="Number of internal nodes per DAG.",
    )
    parser.add_argument(
        "--n-dags",
        type=int,
        default=100,
        help="Number of structurally unique random DAGs to analyze.",
    )
    parser.add_argument(
        "--n-perms-sample",
        type=int,
        default=100000,
        help=(
            "Maximum number of permutations to test per DAG. "
            "If k! <= this value, all permutations are tested exhaustively."
        ),
    )
    parser.add_argument(
        "--n-canon-verify",
        type=int,
        default=100,
        help="Number of permutations on which to verify canonical invariance.",
    )
    parser.add_argument(
        "--num-vars",
        type=int,
        default=1,
        help="Number of input variables (m).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for DAG generation and permutation sampling.",
    )
    parser.add_argument(
        "--canon-timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds for each canonical string computation.",
    )
    parser.add_argument(
        "--include-benchmarks",
        action="store_true",
        help=(
            "Include Nguyen benchmark DAGs with matching k "
            "(currently a no-op; pending sympy integration)."
        ),
    )
    return parser.parse_args()


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    """Entry point for the search space permutation analysis."""
    args = parse_args()

    log.info("=" * 72)
    log.info("SEARCH SPACE PERMUTATION ANALYSIS")
    log.info("=" * 72)
    log.info("  output:            %s", args.output)
    log.info("  k_value:           %d", args.k_value)
    log.info("  num_vars (m):      %d", args.num_vars)
    log.info("  n_dags:            %d", args.n_dags)
    log.info("  n_perms_sample:    %d", args.n_perms_sample)
    log.info("  n_canon_verify:    %d", args.n_canon_verify)
    log.info("  seed:              %d", args.seed)
    log.info("  canon_timeout:     %.1f s", args.canon_timeout)
    log.info("  include_benchmarks:%s", args.include_benchmarks)
    log.info("  k! = %d", math.factorial(args.k_value))
    exhaustive = math.factorial(args.k_value) <= args.n_perms_sample
    log.info("  mode:              %s", "exhaustive" if exhaustive else "sampled")
    log.info("")

    # --- Generate source DAGs ---
    t_gen_start = time.perf_counter()
    dags = generate_source_dags(
        k=args.k_value,
        m=args.num_vars,
        n_dags=args.n_dags,
        seed=args.seed,
        include_benchmarks=args.include_benchmarks,
        canon_timeout=args.canon_timeout,
    )
    t_gen = time.perf_counter() - t_gen_start
    log.info("Generated %d unique DAGs in %.2f s", len(dags), t_gen)
    log.info("")

    # --- Analyze each DAG ---
    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    t_analysis_start = time.perf_counter()

    for dag_idx, (dag, source) in enumerate(dags):
        t_dag_start = time.perf_counter()
        try:
            row = analyze_single_dag(
                dag=dag,
                dag_idx=dag_idx,
                source=source,
                k=args.k_value,
                m=args.num_vars,
                n_perms_sample=args.n_perms_sample,
                n_canon_verify=args.n_canon_verify,
                canon_timeout=args.canon_timeout,
                rng=rng,
            )
            rows.append(row)
            t_dag = time.perf_counter() - t_dag_start
            log.info(
                "DAG %d/%d: k=%d, n_distinct_d2s=%d, invariant=%.4f (%.2f s)",
                dag_idx + 1,
                len(dags),
                args.k_value,
                row["n_distinct_d2s"],
                row["invariant_success_rate"],
                t_dag,
            )
        except CanonicalTimeoutError:
            log.warning(
                "DAG %d/%d: canonical timeout on original DAG, skipping.",
                dag_idx + 1,
                len(dags),
            )
        except Exception:
            log.exception(
                "DAG %d/%d: unexpected error, skipping.",
                dag_idx + 1,
                len(dags),
            )

    t_analysis = time.perf_counter() - t_analysis_start
    log.info("")
    log.info("Analysis complete: %d DAGs in %.2f s", len(rows), t_analysis)

    # --- Write CSV ---
    if rows:
        write_csv(rows, args.output)

    # --- Summary ---
    _print_summary(rows)


if __name__ == "__main__":
    main()
