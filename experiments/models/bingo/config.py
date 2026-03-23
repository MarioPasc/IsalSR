"""Bingo experiment configuration.

Maps YAML config to Bingo pipeline construction parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BingoConfig:
    """Configuration for Bingo experiments."""

    population_size: int = 500
    stack_size: int = 32
    operators: list[str] = field(
        default_factory=lambda: ["+", "-", "*", "/", "sin", "cos", "exp", "log"],
    )
    use_simplification: bool = False
    crossover_prob: float = 0.4
    mutation_prob: float = 0.4
    metric: str = "mse"
    clo_alg: str = "lm"
    generations: int = 10_000_000  # effectively infinite; time-limited
    fitness_threshold: float = 1e-16
    max_time: float = 1800.0
    max_evals: int = 10_000_000

    # IsalSR-specific settings
    canonicalization_timeout: float = 60.0
    use_pruned: bool = True

    # Trajectory logging
    snapshot_frequency: int = 10  # snapshot every N generations

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BingoConfig:
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)
