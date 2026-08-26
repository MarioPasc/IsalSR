"""Shared locations and CLI plumbing for the revision-campaign analysis.

The campaign corpus and the analysis outputs live on external storage; the code
that produces them lives here. Every script takes ``--corpus`` and
``--analyses`` so that a different copy of either can be pointed at without
editing a path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: Local backup of the three-arm campaign corpus, 12,600 cells.
DEFAULT_CORPUS = Path(
    "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/results/review/c2_3arm"
)

#: Self-contained output tree beside the corpus.
DEFAULT_ANALYSES = DEFAULT_CORPUS / "analyses"

#: The seven suites, in the order the manuscript introduces them.
SUITES = (
    "nguyen",
    "feynman",
    "hard",
    "cherrypicked",
    "roundoff",
    "feynman_remainder",
    "strogatz",
)

#: The suite as it stood before the revision extended it, reported alongside the
#: full suite so that a larger denominator is not mistaken for a larger effect.
SUITES_CORE = SUITES[:5]

METHODS = ("udfs", "bingo")
ARMS = ("baseline", "hash", "isalsr")

#: Display names for the three arms, matching the manuscript's usage.
ARM_LABEL = {"baseline": "native", "hash": "naive hash", "isalsr": "IsalSR"}

#: Buckets for the k-stratified overhead table.
K_RANGES = ((0, 5), (5, 15), (15, 32))


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach ``--corpus`` and ``--analyses`` to a parser."""
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Campaign corpus root holding <method>/<suite>/<problem>/<arm>/seed_NN.",
    )
    parser.add_argument(
        "--analyses",
        type=Path,
        default=DEFAULT_ANALYSES,
        help="Output tree for derived data, tables and figures.",
    )
    return parser
