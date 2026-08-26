"""Abstract result translator interface.

Converts model-specific RawRunResult to unified schemas (RunLog,
TrajectoryRow). Each SR method implements its own translator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import sympy

from experiments.models.base_runner import RawRunResult
from experiments.models.schemas import RunLog, RunMetadata, TrajectoryRow


class ResultTranslator(ABC):
    """Abstract interface for translating model results to unified format."""

    @abstractmethod
    def to_run_log(
        self,
        raw: RawRunResult,
        metadata: RunMetadata,
    ) -> RunLog:
        """Convert model-specific raw result to unified RunLog.

        Args:
            raw: Model-specific raw result.
            metadata: Run metadata (method, problem, seed, etc.).

        Returns:
            Unified RunLog matching the experimental design schema.
        """
        ...

    @abstractmethod
    def to_trajectory(self, raw: RawRunResult) -> list[TrajectoryRow]:
        """Extract time-series trajectory from raw result.

        Args:
            raw: Model-specific raw result.

        Returns:
            List of trajectory rows ordered by timestamp.
        """
        ...

    @abstractmethod
    def best_expression_sympy(self, raw: RawRunResult) -> sympy.Expr | None:
        """Extract best expression as SymPy for solution verification.

        Returns None if no valid expression was found.
        """
        ...
