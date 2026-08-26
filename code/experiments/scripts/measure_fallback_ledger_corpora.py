"""Measure Round-Trip Fidelity precondition violation rate on two static corpora.

For each corpus, reports counts (before and after CONST normalisation) of:

    n_total        Valid DAGs processed.
    violated_pre   Precondition violated BEFORE ``normalize_const_creation``.
    violated_post  Precondition still violated AFTER normalisation.
    canon_raised   Non-timeout canonicaliser exceptions on the normalised DAG.
    timeout        ``CanonicalTimeoutError`` events on the normalised DAG.

All metrics are also reported stratified by k (number of non-VAR nodes).

Corpus 1 — IsalSR string-decoded DAGs (the P1-P4 property pool).
    5,000 strings × num_vars in {1,2,3} generated with the same defaults as
    ``onetoone_properties.py`` (seed=42, max_tokens=20, all ops allowed).
    VAR-only DAGs (zero internal nodes) are excluded, matching the original
    script.  Expected: violated_pre = 0 (Critical Invariant 7).

Corpus 2 — Synthetic random SR DAGs from ``_dag_generators.make_random_sr_dag``.
    k in [1,20], num_vars in {1,2,3}, per-pair count computed from --c2-target.
    Contains no CONST nodes; normalisation is always a no-op.
    Expected: violated_pre = 0 (every node gets an edge from an existing node).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.models.fallback_ledger import count_nonvar, violates_precondition
from experiments.scripts._dag_generators import make_random_sr_dag
from isalsr.core.canonical import CanonicalTimeoutError, fast_canonical_string
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import LABEL_CHAR_MAP, OperationSet
from isalsr.core.string_to_dag import StringToDAG
from isalsr.search.random_search import random_isalsr_string

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Corpus-1 parameters — mirror onetoone_properties.py defaults exactly.
# ---------------------------------------------------------------------------
_C1_N_STRINGS: int = 5_000
_C1_MAX_TOKENS: int = 20
_C1_SEED: int = 42
_C1_NUM_VARS: list[int] = [1, 2, 3]

# ---------------------------------------------------------------------------
# Corpus-2 parameters.
# ---------------------------------------------------------------------------
_C2_SEED: int = 0
_C2_K_MIN: int = 1
_C2_K_MAX: int = 20
_C2_NUM_VARS: list[int] = [1, 2, 3]

# Canonicalisation budget — matches production (fallback_ledger default).
_CANON_TIMEOUT: float = 60.0


# ---------------------------------------------------------------------------
# Counter helpers
# ---------------------------------------------------------------------------

_METRIC_KEYS = ("n_total", "violated_pre", "violated_post", "canon_raised", "timeout")


def _new_totals() -> dict[str, int]:
    """Return a zeroed flat counter dict."""
    return dict.fromkeys(_METRIC_KEYS, 0)


def _new_by_k() -> dict[str, dict[str, int]]:
    """Return a by_k structure with defaultdict(int) leaves."""
    return {key: defaultdict(int) for key in _METRIC_KEYS}


def _serialise_by_k(by_k: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """Convert defaultdict leaves to plain dicts, keys sorted numerically."""
    return {
        metric: {str(k): cnt for k, cnt in sorted(by_k[metric].items(), key=lambda x: int(x[0]))}
        for metric in _METRIC_KEYS
    }


# ---------------------------------------------------------------------------
# Core measurement routine
# ---------------------------------------------------------------------------


def _measure_one(
    dag: LabeledDAG,
    totals: dict[str, int],
    by_k: dict[str, dict[str, int]],
) -> None:
    """Record precondition and canonicalisation metrics for one DAG.

    Measures the precondition before and after ``normalize_const_creation``,
    then attempts ``fast_canonical_string`` on the normalised DAG.  All
    counters are updated in-place.

    Args:
        dag: The DAG to measure.
        totals: Flat aggregate counters (modified in-place).
        by_k: Per-k counters (modified in-place).
    """
    k = count_nonvar(dag)
    sk = str(k)

    pre_viol: bool = violates_precondition(dag)

    repaired: LabeledDAG = dag.normalize_const_creation() if dag._has_const_nodes() else dag

    post_viol: bool = violates_precondition(repaired)

    is_timeout: bool = False
    is_raised: bool = False
    try:
        fast_canonical_string(repaired, timeout=_CANON_TIMEOUT)
    except CanonicalTimeoutError:
        is_timeout = True
    except Exception:  # noqa: BLE001
        is_raised = True

    # Flat totals.
    totals["n_total"] += 1
    totals["violated_pre"] += int(pre_viol)
    totals["violated_post"] += int(post_viol)
    totals["canon_raised"] += int(is_raised)
    totals["timeout"] += int(is_timeout)

    # Per-k histograms.
    by_k["n_total"][sk] += 1
    by_k["violated_pre"][sk] += int(pre_viol)
    by_k["violated_post"][sk] += int(post_viol)
    by_k["canon_raised"][sk] += int(is_raised)
    by_k["timeout"][sk] += int(is_timeout)


# ---------------------------------------------------------------------------
# Corpus 1 — S2D-decoded random IsalSR strings
# ---------------------------------------------------------------------------


def measure_corpus_1() -> dict[str, Any]:
    """Measure Corpus 1: S2D-decoded random IsalSR strings.

    Replicates the string generation of ``onetoone_properties.py`` with its
    default parameters: 5,000 strings per num_vars in {1,2,3}, seed=42,
    max_tokens=20, all operations allowed.  VAR-only DAGs are skipped.

    Returns:
        Dict with aggregate counters and by_k histograms.
    """
    ops = frozenset(LABEL_CHAR_MAP.values())
    allowed_ops = OperationSet(ops)

    totals = _new_totals()
    by_k = _new_by_k()
    n_skipped: int = 0
    n_parse_error: int = 0

    for num_vars in _C1_NUM_VARS:
        rng = np.random.default_rng(_C1_SEED)
        log.info(
            "Corpus 1: num_vars=%d, n_strings=%d, seed=%d",
            num_vars,
            _C1_N_STRINGS,
            _C1_SEED,
        )

        for i in range(_C1_N_STRINGS):
            string = random_isalsr_string(num_vars, _C1_MAX_TOKENS, allowed_ops, rng)
            try:
                dag = StringToDAG(string, num_vars, allowed_ops).run()
            except Exception:  # noqa: BLE001
                n_parse_error += 1
                continue

            # Skip VAR-only DAGs (no internal nodes to test).
            if dag.node_count <= num_vars:
                n_skipped += 1
                continue

            _measure_one(dag, totals, by_k)

            if (i + 1) % 2_000 == 0:
                log.info(
                    "  Corpus 1 num_vars=%d: %d/%d  n_total=%d",
                    num_vars,
                    i + 1,
                    _C1_N_STRINGS,
                    totals["n_total"],
                )

    log.info(
        "Corpus 1 complete: n_total=%d  skipped=%d  parse_err=%d  "
        "violated_pre=%d  violated_post=%d  timeout=%d  canon_raised=%d",
        totals["n_total"],
        n_skipped,
        n_parse_error,
        totals["violated_pre"],
        totals["violated_post"],
        totals["timeout"],
        totals["canon_raised"],
    )

    result: dict[str, Any] = dict(totals)
    result["n_skipped_var_only"] = n_skipped
    result["n_parse_error"] = n_parse_error
    result["by_k"] = _serialise_by_k(by_k)
    return result


# ---------------------------------------------------------------------------
# Corpus 2 — synthetic random SR DAGs
# ---------------------------------------------------------------------------


def measure_corpus_2(target: int = 50_000) -> dict[str, Any]:
    """Measure Corpus 2: synthetic random SR DAGs.

    Uses ``make_random_sr_dag`` with k in [_C2_K_MIN, _C2_K_MAX] and
    num_vars in {1,2,3}.  The per-(k, num_vars) count is chosen so the
    total is approximately ``target``.

    Args:
        target: Approximate total DAG count to generate.

    Returns:
        Dict with aggregate counters and by_k histograms.
    """
    k_values = list(range(_C2_K_MIN, _C2_K_MAX + 1))
    n_pairs = len(k_values) * len(_C2_NUM_VARS)
    per_pair = max(1, target // n_pairs)

    log.info(
        "Corpus 2: k=%d..%d, num_vars=%s, per_pair=%d, target=%d (actual=%d)",
        _C2_K_MIN,
        _C2_K_MAX,
        _C2_NUM_VARS,
        per_pair,
        target,
        per_pair * n_pairs,
    )

    totals = _new_totals()
    by_k = _new_by_k()
    seed_counter: int = _C2_SEED

    for num_vars in _C2_NUM_VARS:
        for k in k_values:
            for _ in range(per_pair):
                dag = make_random_sr_dag(num_vars, k, seed_counter)
                seed_counter += 1
                _measure_one(dag, totals, by_k)

        log.info(
            "  Corpus 2 num_vars=%d done, n_total so far=%d",
            num_vars,
            totals["n_total"],
        )

    log.info(
        "Corpus 2 complete: n_total=%d  violated_pre=%d  violated_post=%d  "
        "timeout=%d  canon_raised=%d",
        totals["n_total"],
        totals["violated_pre"],
        totals["violated_post"],
        totals["timeout"],
        totals["canon_raised"],
    )

    result: dict[str, Any] = dict(totals)
    result["per_pair"] = per_pair
    result["by_k"] = _serialise_by_k(by_k)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run both corpora and write the JSON result file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="Absolute path to the output JSON file.",
    )
    parser.add_argument(
        "--c2-target",
        type=int,
        default=50_000,
        help="Approximate DAG count for Corpus 2 (default: 50,000).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    t0 = time.perf_counter()

    log.info("=== Corpus 1: S2D-decoded random IsalSR strings ===")
    c1 = measure_corpus_1()
    t1 = time.perf_counter()
    log.info("Corpus 1 done in %.1f s", t1 - t0)

    log.info("=== Corpus 2: Synthetic random SR DAGs ===")
    c2 = measure_corpus_2(target=args.c2_target)
    t2 = time.perf_counter()
    log.info("Corpus 2 done in %.1f s", t2 - t1)

    output: dict[str, Any] = {
        "corpus_1_s2d": c1,
        "corpus_2_synthetic": c2,
        "elapsed_seconds": round(t2 - t0, 2),
        "parameters": {
            "c1_n_strings_per_num_vars": _C1_N_STRINGS,
            "c1_max_tokens": _C1_MAX_TOKENS,
            "c1_seed": _C1_SEED,
            "c1_num_vars": _C1_NUM_VARS,
            "c2_target": args.c2_target,
            "c2_seed": _C2_SEED,
            "c2_k_range": [_C2_K_MIN, _C2_K_MAX],
            "c2_num_vars": _C2_NUM_VARS,
            "canon_timeout": _CANON_TIMEOUT,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    # Summary to stdout for human review.
    sep = "=" * 62

    def _pct(num: int, denom: int) -> str:
        return f"{100 * num / max(1, denom):.2f}%"

    print(f"\n{sep}")
    print("CORPUS 1 — S2D-decoded random IsalSR strings")
    print(f"  n_total:       {c1['n_total']:>8,}  (valid count; 14,841 expected)")
    print(f"  violated_pre:  {c1['violated_pre']:>8,}  ({_pct(c1['violated_pre'], c1['n_total'])})")
    print(
        f"  violated_post: {c1['violated_post']:>8,}  ({_pct(c1['violated_post'], c1['n_total'])})"
    )
    print(f"  canon_raised:  {c1['canon_raised']:>8,}")
    print(f"  timeout:       {c1['timeout']:>8,}")

    print("\nCORPUS 2 — Synthetic random SR DAGs")
    print(f"  n_total:       {c2['n_total']:>8,}")
    print(f"  violated_pre:  {c2['violated_pre']:>8,}  ({_pct(c2['violated_pre'], c2['n_total'])})")
    print(
        f"  violated_post: {c2['violated_post']:>8,}  ({_pct(c2['violated_post'], c2['n_total'])})"
    )
    print(f"  canon_raised:  {c2['canon_raised']:>8,}")
    print(f"  timeout:       {c2['timeout']:>8,}")

    print(f"\nTotal elapsed: {t2 - t0:.1f} s")
    print(f"Output:        {out_path}")
    print(sep)


if __name__ == "__main__":
    main()
