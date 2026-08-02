"""Execute the pre-registered R3.1 benchmark-extension selection rule.

The rule itself is `docs/md_files/changes/r31_extension_selection.md`, committed
before this script was ever run. This module is the executable form of §3 and §4
of that document and nothing else: it takes no arguments that change the outcome,
reads no results, and has no randomness that is not derived from the eligible pool.

Running it twice, on any machine, must produce byte-identical output. That
property is the whole point — it is what lets a third party verify that the
problem list was fixed before the campaign rather than after it.

Usage
-----
    python -m experiments.scripts.r31_draw_extension --write

Without ``--write`` the draw is printed and nothing is persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.datasets.feynman_catalogue import (  # noqa: E402
    AIFEYNMAN_120,
    IN_SUITE_IDS,
    classification_table,
    eligible_extension_pool,
)

from benchmarks.datasets.strogatz import STROGATZ_BENCHMARKS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

#: Number of AI Feynman equations drawn. 14 Strogatz + 6 = 20, D2's budgeted
#: size (EXECUTION-PLAN.md §1; 1,440 core-hours per added problem).
K_FEYNMAN = 6

OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "md_files"
    / "changes"
    / "r31_extension_selection_draw.json"
)


def derive_seed(eligible_ids: list[str]) -> int:
    """Derive the draw seed from the eligible pool itself.

    The seed is a pure function of the pool, so it is not a free parameter and
    there is nothing to fish over: given the four inclusion criteria and the
    existing suite, the pool is determined, and given the pool the seed and the
    draw are determined.

    Parameters
    ----------
    eligible_ids
        The eligible canonical ids, in sorted order.

    Returns
    -------
    int
        A seed in ``[0, 2**32)``.
    """
    digest = hashlib.sha256("|".join(eligible_ids).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % 2**32


def draw_feynman_remainder(k: int = K_FEYNMAN) -> dict[str, Any]:
    """Execute §4.2 of the pre-registration.

    Parameters
    ----------
    k
        Number of equations to draw.

    Returns
    -------
    dict
        The drawn ids together with every intermediate quantity needed to
        reproduce and audit the draw.

    Raises
    ------
    ValueError
        If the eligible pool is smaller than ``k``.
    """
    eligible = sorted(eligible_extension_pool())
    if len(eligible) < k:
        raise ValueError(f"Eligible pool has {len(eligible)} ids, cannot draw {k}.")

    seed = derive_seed(eligible)
    permuted = np.random.default_rng(seed).permutation(np.asarray(eligible, dtype=object))
    drawn = sorted(str(x) for x in permuted[:k])

    by_id = {e["id"]: e for e in AIFEYNMAN_120}
    return {
        "k": k,
        "n_eligible": len(eligible),
        "eligible_sha256": hashlib.sha256("|".join(eligible).encode("utf-8")).hexdigest(),
        "seed": seed,
        "drawn_ids": drawn,
        "drawn": [
            {
                "id": i,
                "source": by_id[i]["source"],
                "output": by_id[i]["output"],
                "formula": by_id[i]["formula"],
                "num_variables": by_id[i]["num_variables"],
                "variables": by_id[i]["variables"],
                "pmlb_id": by_id[i]["pmlb_id"],
            }
            for i in drawn
        ],
        "eligible_ids": eligible,
    }


def build_record() -> dict[str, Any]:
    """Assemble the full D2 record: the Strogatz half plus the drawn half.

    Returns
    -------
    dict
        The complete, auditable description of ``D2``.
    """
    table = classification_table()
    feynman = draw_feynman_remainder()
    strogatz = sorted(b["name"] for b in STROGATZ_BENCHMARKS)

    return {
        "rule_document": "docs/md_files/changes/r31_extension_selection.md",
        "executed_by": "experiments/scripts/r31_draw_extension.py",
        "criterion_ii": {
            "reading_used": "syntactic",
            "n_total": table["n_total"],
            "n_representable_syntactic": table["n_representable_syntactic"],
            "n_representable_semantic": table["n_representable_semantic"],
            "blocked_syntactic": table["blocked_syntactic"],
            "blocked_semantic": table["blocked_semantic"],
        },
        "in_suite_ids": sorted(IN_SUITE_IDS),
        "strogatz": {"n": len(strogatz), "problems": strogatz, "selection": "all, no filter"},
        "feynman_remainder": feynman,
        "d2_size": len(strogatz) + feynman["k"],
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv
        Command-line arguments; ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist the draw to the pre-registered output path.",
    )
    args = parser.parse_args(argv)

    record = build_record()
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"

    log.info("eligible pool      : %d", record["feynman_remainder"]["n_eligible"])
    log.info("derived seed       : %d", record["feynman_remainder"]["seed"])
    log.info("Feynman remainder  : %s", ", ".join(record["feynman_remainder"]["drawn_ids"]))
    log.info("Strogatz           : %d problems (all)", record["strogatz"]["n"])
    log.info("D2 size            : %d", record["d2_size"])

    if args.write:
        OUTPUT_PATH.write_text(payload, encoding="utf-8")
        log.info("wrote %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
