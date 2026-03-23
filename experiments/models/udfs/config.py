"""UDFS experiment configuration.

Maps YAML config to DAGRegressor constructor parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UDFSConfig:
    """Configuration for UDFS DAGRegressor experiments."""

    n_calc_nodes: int = 5
    max_orders: int = 200_000
    max_time: float = 3600.0
    processes: int = 1
    mode: str = "hierarchical"
    k: int = 2
    stop_thresh: float = 1e-10
    use_tan: bool = False
    operator_set: list[str] = field(
        default_factory=lambda: ["+", "*", "-", "/", "sin", "cos", "exp", "log"],
    )

    # IsalSR-specific settings
    canonicalization_timeout: float = 60.0
    use_pruned: bool = True

    # Trajectory logging
    snapshot_frequency: int = 1000  # snapshot every N evaluations

    def to_dag_regressor_kwargs(self) -> dict[str, Any]:
        """Convert to DAGRegressor constructor kwargs."""
        return {
            "n_calc_nodes": self.n_calc_nodes,
            "max_orders": self.max_orders,
            "max_time": self.max_time,
            "processes": self.processes,
            "mode": self.mode,
            "k": self.k,
            "stop_thresh": self.stop_thresh,
            "use_tan": self.use_tan,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UDFSConfig:
        """Create config from dict (e.g., from YAML)."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)
