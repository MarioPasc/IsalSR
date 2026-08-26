"""SRBench standardized benchmark loader.

Reference: La Cava et al. (2021). Contemporary symbolic regression methods
and their relative performance. NeurIPS Datasets & Benchmarks.

GitHub: https://github.com/cavalab/srbench
Website: https://cavalab.org/srbench/

SRBench features 130 problems with hidden ground-truth solutions and
122 real-world datasets from the PMLB database. Full integration requires
the PMLB package; this module provides metadata and documentation.
"""

from __future__ import annotations

from typing import Any

# SRBench ground-truth benchmark names (subset from La Cava et al., 2021).
SRBENCH_GROUND_TRUTH: list[str] = [
    "Nguyen-1",
    "Nguyen-2",
    "Nguyen-3",
    "Nguyen-4",
    "Nguyen-5",
    "Nguyen-6",
    "Nguyen-7",
    "Nguyen-8",
    "Nguyen-9",
    "Nguyen-10",
    "Nguyen-11",
    "Nguyen-12",
    "Koza-1",
    "Koza-2",
    "Koza-3",
    "Keijzer-3",
    "Keijzer-4",
    "Keijzer-6",
    "Keijzer-7",
    "Keijzer-9",
    "Keijzer-11",
    "Keijzer-14",
    "Keijzer-15",
    "Livermore-4",
    "Livermore-5",
    "Livermore-9",
    "Livermore-11",
    "Livermore-12",
    "Livermore-14",
]


def list_srbench_info() -> dict[str, Any]:
    """Return metadata about the SRBench benchmark suite.

    Returns:
        Dict with keys: 'reference', 'url', 'n_ground_truth', 'n_blackbox',
        'ground_truth_names'.
    """
    return {
        "reference": "La Cava et al. (2021). NeurIPS Datasets & Benchmarks.",
        "url": "https://github.com/cavalab/srbench",
        "n_ground_truth": 130,
        "n_blackbox": 122,
        "ground_truth_names": SRBENCH_GROUND_TRUTH,
    }
