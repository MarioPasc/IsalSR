"""Bingo baseline runner.

Constructs the Bingo evolutionary pipeline manually (same components
as SymbolicRegressor) with a standard Evaluation. The IsalSR variant
swaps in IsalSREvaluation while keeping everything else identical.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.symbolic_regression.agraph.component_generator import ComponentGenerator
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.explicit_regression import ExplicitRegression, ExplicitTrainingData

from experiments.models.base_runner import ModelRunner, RawRunResult
from experiments.models.bingo.config import BingoConfig

log = logging.getLogger(__name__)


@dataclass
class BingoRawResult(RawRunResult):
    """Raw result from a Bingo run."""

    best_agraph: Any = None
    best_sympy: Any = None
    y_pred_train: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred_test: np.ndarray = field(default_factory=lambda: np.array([]))
    total_evals: int = 0
    best_fitness: float = float("inf")
    n_generations: int = 0
    # IsalSR-specific (populated by IsalSR runner)
    n_total_dags: int = 0
    n_unique_canonical: int = 0
    n_skipped: int = 0
    canonicalization_time_s: float = 0.0
    search_only_time_s: float = 0.0


def build_bingo_pipeline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: BingoConfig,
    evaluation_cls: type = Evaluation,
    evaluation_kwargs: dict | None = None,
):
    """Construct the Bingo evolutionary pipeline.

    Shared between baseline and IsalSR runners to ensure identical
    conditions except for the Evaluation component.

    Args:
        x_train: Training input, shape (n_train, n_features).
        y_train: Training targets, shape (n_train,).
        cfg: BingoConfig.
        evaluation_cls: Evaluation class to use (Evaluation or IsalSREvaluation).
        evaluation_kwargs: Extra kwargs for the evaluation constructor.

    Returns:
        (island, fitness_fn, evaluation) tuple.
    """
    n_features = x_train.shape[1]

    # Component generator
    component_gen = ComponentGenerator(n_features)
    for op in cfg.operators:
        component_gen.add_operator(op)

    # AGraph generator, crossover, mutation
    agraph_gen = AGraphGenerator(
        cfg.stack_size,
        component_gen,
        use_simplification=cfg.use_simplification,
    )
    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_gen)

    # Fitness function
    training_data = ExplicitTrainingData(x_train, y_train.reshape(-1, 1))
    fitness = ExplicitRegression(training_data, metric=cfg.metric)

    # Local optimization wrapper
    optimizer = ScipyOptimizer(fitness, method=cfg.clo_alg)
    local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)

    # Evaluation
    eval_kwargs = evaluation_kwargs or {}
    evaluation = evaluation_cls(local_opt_fitness, **eval_kwargs)

    # Evolutionary algorithm
    ea = AgeFitnessEA(
        evaluation,
        agraph_gen,
        crossover,
        mutation,
        cfg.crossover_prob,
        cfg.mutation_prob,
        cfg.population_size,
    )

    # Island
    island = Island(ea, agraph_gen, cfg.population_size)

    return island, fitness, evaluation


def extract_sympy(agraph: Any) -> Any:
    """Extract SymPy expression from an AGraph.

    Tries multiple approaches:
    1. get_formatted_string("sympy") → sympify
    2. String substitution for Bingo-specific syntax (e.g., |.| for abs)
    3. Falls back to None with a warning if all methods fail.
    """
    if agraph is None:
        return None

    # Attempt 1: standard sympy format
    try:
        sympy_str = agraph.get_formatted_string("sympy")
        expr = sympy.sympify(sympy_str)
        return _remap_bingo_symbols(expr)
    except Exception:  # noqa: BLE001
        pass

    # Attempt 2: clean up common issues in Bingo's sympy string output
    try:
        sympy_str = agraph.get_formatted_string("sympy")
        # Replace |x| with Abs(x) for SymPy
        cleaned = sympy_str.replace("|", "")
        if cleaned != sympy_str:
            cleaned = f"Abs({cleaned})"
        expr = sympy.sympify(cleaned, locals={"Abs": sympy.Abs})
        return _remap_bingo_symbols(expr)
    except Exception:  # noqa: BLE001
        pass

    # Attempt 3: try console format → sympify as last resort
    try:
        console_str = agraph.get_formatted_string("console")
        # Console format is more human-readable; may need cleanup
        cleaned = console_str.replace(")(", ")*(")
        expr = sympy.sympify(cleaned)
        return _remap_bingo_symbols(expr)
    except Exception:  # noqa: BLE001
        pass

    log.warning(
        "Failed to extract SymPy from AGraph (all methods). Console repr: %s",
        _safe_agraph_str(agraph),
    )
    return None


def _remap_bingo_symbols(expr: Any) -> Any:
    """Remap Bingo's X_0, X_1, ... to IsalSR's x_0, x_1, ..."""
    subs = {}
    for sym in expr.free_symbols:
        name = str(sym)
        if name.startswith("X_"):
            idx = name[2:]
            subs[sym] = sympy.Symbol(f"x_{idx}")
    return expr.subs(subs) if subs else expr


def _safe_agraph_str(agraph: Any) -> str:
    """Get a string representation of an AGraph without crashing."""
    try:
        return agraph.get_formatted_string("console")
    except Exception:  # noqa: BLE001
        try:
            return str(agraph)
        except Exception:  # noqa: BLE001
            return "<unrepresentable>"


def get_symbolic_form(agraph: Any, best_sympy: Any = None) -> str:
    """Get a human-readable symbolic form string.

    Returns SymPy string if available, else tries console format from AGraph.
    """
    if best_sympy is not None:
        return str(best_sympy)

    if agraph is None:
        return ""

    return _safe_agraph_str(agraph)


class BingoBaselineRunner(ModelRunner):
    """Runs Bingo without IsalSR canonicalization."""

    def __init__(self, config: BingoConfig | None = None):
        self._config = config or BingoConfig()

    @property
    def name(self) -> str:
        return "bingo"

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
    ) -> BingoRawResult:
        cfg = BingoConfig.from_dict(config) if config else self._config

        np.random.seed(seed)

        island, fitness_fn, evaluation = build_bingo_pipeline(x_train, y_train, cfg)

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            island.evolve_until_convergence(
                max_generations=cfg.generations,
                fitness_threshold=cfg.fitness_threshold,
                max_fitness_evaluations=cfg.max_evals,
                convergence_check_frequency=10,
            )
        wall_clock = time.perf_counter() - t0

        # Extract results
        total_evals = fitness_fn.eval_count
        best_agraph = None
        best_sympy_expr = None
        best_fitness = float("inf")
        y_pred_train = np.full(len(y_train), np.nan)
        y_pred_test = np.full(len(y_test), np.nan)
        n_gens = island.generational_age

        try:
            best_agraph = island.get_best_individual()
            best_fitness = best_agraph.fitness
            best_sympy_expr = extract_sympy(best_agraph)
            y_pred_train = best_agraph.evaluate_equation_at(x_train).flatten()
            y_pred_test = best_agraph.evaluate_equation_at(x_test).flatten()
        except Exception:  # noqa: BLE001
            log.debug("Failed to extract Bingo results", exc_info=True)

        return BingoRawResult(
            wall_clock_s=wall_clock,
            seed=seed,
            best_agraph=best_agraph,
            best_sympy=best_sympy_expr,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            total_evals=total_evals,
            best_fitness=best_fitness,
            n_generations=n_gens,
            n_total_dags=total_evals,
            n_unique_canonical=total_evals,  # baseline: all unique
            n_skipped=0,
            canonicalization_time_s=0.0,
            search_only_time_s=wall_clock,
        )
