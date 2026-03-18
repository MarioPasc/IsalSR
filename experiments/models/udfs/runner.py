"""UDFS baseline runner.

Wraps DAGRegressor with the ModelRunner interface.
No IsalSR canonicalization — pure UDFS evaluation.
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Ensure vendored DAG_search is importable
_vendor_dir = str(Path(__file__).parent / "vendor")
if _vendor_dir not in sys.path:
    sys.path.insert(0, _vendor_dir)

from DAG_search.dag_search import DAGRegressor  # noqa: E402

from experiments.models.base_runner import ModelRunner, RawRunResult
from experiments.models.udfs.config import UDFSConfig

log = logging.getLogger(__name__)


@dataclass
class TrajectorySnapshot:
    """Periodic snapshot during UDFS search."""

    timestamp_s: float
    total_evals: int
    best_loss: float


@dataclass
class UDFSRawResult(RawRunResult):
    """Raw result from a UDFS run."""

    best_sympy: Any = None
    y_pred_train: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred_test: np.ndarray = field(default_factory=lambda: np.array([]))
    total_evals: int = 0
    best_loss: float = float("inf")
    n_top_graphs: int = 0
    trajectory_snapshots: list[TrajectorySnapshot] = field(default_factory=list)
    # IsalSR-specific (populated by IsalSR runner)
    n_total_dags: int = 0
    n_unique_canonical: int = 0
    n_skipped: int = 0
    canonicalization_time_s: float = 0.0
    search_only_time_s: float = 0.0


class UDFSBaselineRunner(ModelRunner):
    """Runs UDFS DAGRegressor without IsalSR canonicalization."""

    def __init__(self, config: UDFSConfig | None = None):
        self._config = config or UDFSConfig()

    @property
    def name(self) -> str:
        return "udfs"

    @property
    def variant(self) -> str:
        return "baseline"

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        config: dict[str, Any],
    ) -> UDFSRawResult:
        """Run UDFS on training data."""
        cfg = UDFSConfig.from_dict(config) if config else self._config
        kwargs = cfg.to_dag_regressor_kwargs()

        regressor = DAGRegressor(**kwargs)
        regressor.random_state = seed

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regressor.fit(x_train, y_train, verbose=0)
        wall_clock = time.perf_counter() - t0

        # Extract results
        total_evals = getattr(regressor, "total_evals", 0)
        best_sympy = None
        y_pred_train = np.full(len(y_train), np.nan)
        y_pred_test = np.full(len(y_test), np.nan)
        best_loss = float("inf")
        n_top = 0

        if hasattr(regressor, "cgraph") and regressor.cgraph is not None:
            try:
                best_sympy = regressor.model()
            except Exception:  # noqa: BLE001
                log.debug("Failed to extract SymPy model", exc_info=True)

            try:
                y_pred_train = regressor.predict(x_train)
                y_pred_test = regressor.predict(x_test)
            except Exception:  # noqa: BLE001
                log.debug("Prediction failed", exc_info=True)

        if hasattr(regressor, "results") and regressor.results:
            losses = regressor.results.get("losses", [])
            if losses:
                best_loss = float(min(losses))
            n_top = len(regressor.results.get("graphs", []))

        return UDFSRawResult(
            wall_clock_s=wall_clock,
            seed=seed,
            best_sympy=best_sympy,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            total_evals=total_evals,
            best_loss=best_loss,
            n_top_graphs=n_top,
            n_total_dags=total_evals,
            n_unique_canonical=total_evals,  # baseline: all unique (no dedup)
            n_skipped=0,
            canonicalization_time_s=0.0,
            search_only_time_s=wall_clock,
        )
