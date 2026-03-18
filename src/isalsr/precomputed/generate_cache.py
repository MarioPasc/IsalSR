"""CLI entry point for precomputed cache generation.

Generates canonical string caches via random sampling (Phase 2 of the
design), with support for SLURM array job sharding and shard merging.

Usage:
    # Generate a sampled cache shard
    python -m isalsr.precomputed.generate_cache \\
        --mode sampled \\
        --num-variables 1 \\
        --n-strings 10000 \\
        --max-tokens 30 \\
        --seed 42 \\
        --exhaustive-timeout 60 \\
        --output /path/to/cache.h5

    # Merge shards into one file
    python -m isalsr.precomputed.generate_cache \\
        --mode merge \\
        --input-dir /path/to/shards/ \\
        --output /path/to/merged.h5

Reference: docs/design/precomputed_cache_design.md, Section 7.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from isalsr.core.node_types import LABEL_CHAR_MAP, OperationSet
from isalsr.precomputed.cache_manager import CacheManager
from isalsr.search.random_search import random_isalsr_string

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def _build_ops(ops_str: str | None) -> OperationSet:
    """Build OperationSet from comma-separated label string or use all ops."""
    if ops_str is None:
        return OperationSet(frozenset(LABEL_CHAR_MAP.values()))
    labels = [s.strip() for s in ops_str.split(",")]
    node_types = frozenset(LABEL_CHAR_MAP[label] for label in labels)
    return OperationSet(node_types)


def _generate_sampled(args: argparse.Namespace) -> None:
    """Generate cache via random string sampling."""
    ops = _build_ops(args.ops)
    seed = args.seed
    if args.run_id is not None:
        seed = args.seed + args.run_id - 1

    rng = np.random.default_rng(seed)
    manager = CacheManager(
        num_variables=args.num_variables,
        operator_set=ops,
        exhaustive_timeout=args.exhaustive_timeout,
    )

    n_added = 0
    n_skipped = 0
    t_start = time.perf_counter()

    log.info(
        "Generating sampled cache: m=%d, n_strings=%d, max_tokens=%d, seed=%d",
        args.num_variables,
        args.n_strings,
        args.max_tokens,
        seed,
    )

    for i in range(args.n_strings):
        raw = random_isalsr_string(args.num_variables, args.max_tokens, ops, rng)
        if manager.compute_and_add(raw):
            n_added += 1
        else:
            n_skipped += 1

        if (i + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            log.info(
                "  Progress: %d/%d (%.0f strings/s, %d added, %d skipped)",
                i + 1,
                args.n_strings,
                rate,
                n_added,
                n_skipped,
            )

    elapsed = time.perf_counter() - t_start
    log.info(
        "Generation complete: %d added, %d skipped in %.1fs",
        n_added,
        n_skipped,
        elapsed,
    )

    # Flush to HDF5
    output = Path(args.output)
    manager.flush_hdf5(output)
    manager.write_metadata_json(output.with_suffix(".json"))

    # Print summary
    stats = manager.stats
    log.info("=== Cache Summary ===")
    log.info("  Total entries:             %d", stats.total_entries)
    log.info("  Unique pruned canonicals:  %d", stats.unique_canonical_pruned)
    log.info("  Unique exhaustive:         %d", stats.unique_canonical_exhaustive)
    log.info("  Exhaustive timeouts:       %d", stats.exhaustive_timeout_count)
    log.info("  Exhaustive == pruned:      %d", stats.exhaustive_eq_pruned_count)
    log.info("  Avg depth:                 %.2f", stats.avg_depth)
    log.info("  Max depth:                 %d", stats.max_depth)
    log.info("  Avg internal nodes:        %.2f", stats.avg_internal_nodes)


def _merge_shards(args: argparse.Namespace) -> None:
    """Merge multiple cache shard files into one."""
    input_dir = Path(args.input_dir)
    shards = sorted(input_dir.glob("cache_shard_*.h5"))
    if not shards:
        shards = sorted(input_dir.glob("*.h5"))

    if not shards:
        log.error("No HDF5 shard files found in %s", input_dir)
        sys.exit(1)

    log.info("Merging %d shards from %s", len(shards), input_dir)

    # We need a manager to load into. Use the metadata from the first shard.
    ops = _build_ops(args.ops)
    manager = CacheManager(
        num_variables=args.num_variables,
        operator_set=ops,
    )

    for shard in shards:
        log.info("  Loading %s ...", shard.name)
        manager.load_hdf5(shard)

    log.info("  Total entries after merge: %d", len(manager))

    # Deduplicate by raw_string
    seen: set[str] = set()
    deduped: list[int] = []
    for i, e in enumerate(manager.entries):
        if e.raw_string not in seen:
            seen.add(e.raw_string)
            deduped.append(i)

    if len(deduped) < len(manager):
        log.info(
            "  Deduplicated: %d → %d entries",
            len(manager),
            len(deduped),
        )
        # Rebuild manager with deduped entries
        all_entries = manager.entries
        manager = CacheManager(
            num_variables=args.num_variables,
            operator_set=ops,
        )
        for idx in deduped:
            manager.add_entry(all_entries[idx])

    output = Path(args.output)
    manager.flush_hdf5(output)
    manager.write_metadata_json(output.with_suffix(".json"))

    stats = manager.stats
    log.info("=== Merged Cache Summary ===")
    log.info("  Total entries:             %d", stats.total_entries)
    log.info("  Unique pruned canonicals:  %d", stats.unique_canonical_pruned)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IsalSR precomputed canonical string cache generator",
    )
    parser.add_argument(
        "--mode",
        choices=["sampled", "merge"],
        required=True,
        help="Generation mode: 'sampled' for random strings, 'merge' for shard merging.",
    )
    parser.add_argument(
        "--num-variables", type=int, default=1, help="Number of input variables (m)."
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help="Comma-separated operator labels (default: all). Example: '+,*,-,/,s,c,e,l'",
    )
    parser.add_argument(
        "--n-strings", type=int, default=10000, help="Number of random strings to generate."
    )
    parser.add_argument("--max-tokens", type=int, default=30, help="Maximum tokens per string.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="SLURM array task ID (seed = base + run_id - 1).",
    )
    parser.add_argument(
        "--exhaustive-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for exhaustive canonical (default: 60).",
    )
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 file path.")
    parser.add_argument(
        "--input-dir", type=str, default=None, help="Input directory for merge mode."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="cache",
        help="Benchmark name for filename convention.",
    )

    args = parser.parse_args()

    if args.mode == "sampled":
        _generate_sampled(args)
    elif args.mode == "merge":
        if args.input_dir is None:
            parser.error("--input-dir is required for merge mode.")
        _merge_shards(args)


if __name__ == "__main__":
    main()
