"""Abstract model runner interface.

Every SR method implements ModelRunner to provide a standard interface
for running experiments. The runner wraps the method-specific API and
returns a RawRunResult that the translator converts to unified schemas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RawRunResult:
    """Base class for model-specific raw results.

    Subclass this with model-specific fields (e.g., CompGraph for UDFS,
    adjacency matrix for GraphDSR). The translator then converts to
    unified RunLog.
    """

    wall_clock_s: float = 0.0
    seed: int = 0


class ModelRunner(ABC):
    """Abstract interface for running an SR method."""

    @abstractmethod
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        config: dict[str, Any],
    ) -> RawRunResult:
        """Run the SR method on given data.

        Args:
            x_train: Training inputs, shape (n_train, n_features).
            y_train: Training targets, shape (n_train,).
            x_test: Test inputs, shape (n_test, n_features).
            y_test: Test targets, shape (n_test,).
            seed: Random seed for reproducibility.
            config: Method-specific configuration dict.

        Returns:
            Model-specific raw result.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Method name (e.g., 'udfs', 'graphdsr')."""
        ...

    @property
    @abstractmethod
    def variant(self) -> str:
        """Representation variant: 'baseline' or 'isalsr'."""
        ...
