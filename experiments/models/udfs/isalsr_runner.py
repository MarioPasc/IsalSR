"""IsalSR-enhanced UDFS runner.

Wraps UDFS DAGRegressor with canonical string deduplication.
For each CompGraph evaluated by UDFS, converts to LabeledDAG,
computes the pruned canonical string, and skips isomorphic duplicates.

Strategy: Monkey-patch the module-level `evaluate_cgraph` function
during the UDFS run to intercept each graph evaluation. This is the
cleanest subclass-only approach (no vendored code modifications).
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

# Ensure vendored DAG_search is importable
_vendor_dir = str(Path(__file__).parent / "vendor")
if _vendor_dir not in sys.path:
    sys.path.insert(0, _vendor_dir)

import DAG_search.dag_search as dag_search_module  # noqa: E402
from DAG_search.dag_search import DAGRegressor  # noqa: E402

from experiments.models.base_runner import ModelRunner
from experiments.models.udfs.adapter import compgraph_to_labeled_dag
from experiments.models.udfs.config import UDFSConfig
from experiments.models.udfs.runner import UDFSRawResult

log = logging.getLogger(__name__)


class _CanonicalDeduplicator:
    """Tracks canonical strings and deduplication statistics.

    Uses hash-based storage (``set[int]``) instead of storing full canonical
    strings.  This reduces per-entry memory from ~150 bytes (``set[str]``)
    to ~28 bytes (``set[int]``), preventing OOM on long runs.

    The 64-bit Python hash gives collision probability < 3×10⁻⁶ for 10 M
    entries (birthday bound n²/2⁶⁵), which is negligible for our use case.
    """

    def __init__(self, use_pruned: bool = True, timeout: float = 60.0):
        self.use_pruned = use_pruned
        self.timeout = timeout
        self.canonical_seen: set[int] = set()
        self.n_total = 0
        self.n_unique = 0
        self.n_skipped = 0
        self.canon_time_total = 0.0
        self._original_evaluate: Any = None

    def wrap_evaluate_cgraph(self, original_fn: Any) -> Any:
        """Create a wrapper around evaluate_cgraph with canonical dedup."""
        self._original_evaluate = original_fn

        def wrapped(cgraph, X, loss_fkt, opt_mode="grid_zoom", loss_thresh=None):
            self.n_total += 1

            try:
                labeled_dag = compgraph_to_labeled_dag(cgraph)
            except Exception:  # noqa: BLE001
                # If conversion fails, evaluate normally
                return self._original_evaluate(
                    cgraph,
                    X,
                    loss_fkt,
                    opt_mode,
                    loss_thresh,
                )

            t0 = time.perf_counter()
            try:
                if self.use_pruned:
                    from isalsr.core.canonical import pruned_canonical_string

                    canonical = pruned_canonical_string(
                        labeled_dag,
                        timeout=self.timeout,
                    )
                else:
                    from isalsr.core.canonical import canonical_string

                    canonical = canonical_string(
                        labeled_dag,
                        timeout=self.timeout,
                    )
            except Exception:  # noqa: BLE001
                # Canonicalization failed — evaluate normally
                self.canon_time_total += time.perf_counter() - t0
                return self._original_evaluate(
                    cgraph,
                    X,
                    loss_fkt,
                    opt_mode,
                    loss_thresh,
                )

            self.canon_time_total += time.perf_counter() - t0

            canon_hash = hash(canonical)
            if canon_hash in self.canonical_seen:
                self.n_skipped += 1
                # Return infinite loss to skip this graph.
                # Use valid-shaped consts array to avoid crashes in
                # UDFS's invalid-tracking logic.
                n_consts = cgraph.n_consts
                dummy_consts = np.zeros(n_consts) if n_consts > 0 else np.array([])
                return dummy_consts, np.inf

            self.canonical_seen.add(canon_hash)
            self.n_unique += 1
            return self._original_evaluate(
                cgraph,
                X,
                loss_fkt,
                opt_mode,
                loss_thresh,
            )

        return wrapped


@contextmanager
def _patched_evaluate(deduplicator: _CanonicalDeduplicator):
    """Context manager that patches evaluate_cgraph with dedup wrapper."""
    original = dag_search_module.evaluate_cgraph
    dag_search_module.evaluate_cgraph = deduplicator.wrap_evaluate_cgraph(original)
    try:
        yield deduplicator
    finally:
        dag_search_module.evaluate_cgraph = original


class IsalSRUDFSRunner(ModelRunner):
    """Runs UDFS with IsalSR canonical deduplication."""

    def __init__(self, config: UDFSConfig | None = None):
        self._config = config or UDFSConfig()

    @property
    def name(self) -> str:
        return "udfs"

    @property
    def variant(self) -> str:
        return "isalsr"

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        seed: int,
        config: dict[str, Any],
    ) -> UDFSRawResult:
        """Run UDFS with IsalSR canonical deduplication."""
        cfg = UDFSConfig.from_dict(config) if config else self._config
        kwargs = cfg.to_dag_regressor_kwargs()

        regressor = DAGRegressor(**kwargs)
        regressor.random_state = seed

        dedup = _CanonicalDeduplicator(
            use_pruned=cfg.use_pruned,
            timeout=cfg.canonicalization_timeout,
        )

        t0 = time.perf_counter()
        with _patched_evaluate(dedup), warnings.catch_warnings():
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

        search_only = wall_clock - dedup.canon_time_total

        log.info(
            "IsalSR UDFS: total=%d unique=%d skipped=%d canon_time=%.2fs",
            dedup.n_total,
            dedup.n_unique,
            dedup.n_skipped,
            dedup.canon_time_total,
        )

        return UDFSRawResult(
            wall_clock_s=wall_clock,
            seed=seed,
            best_sympy=best_sympy,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            total_evals=total_evals,
            best_loss=best_loss,
            n_top_graphs=n_top,
            n_total_dags=dedup.n_total,
            n_unique_canonical=dedup.n_unique,
            n_skipped=dedup.n_skipped,
            canonicalization_time_s=dedup.canon_time_total,
            search_only_time_s=search_only,
        )
