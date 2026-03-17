"""Population management for evolutionary search with IsalSR strings.

Full evolutionary loop: selection -> crossover -> mutation -> canonicalize
(MANDATORY) -> evaluate -> elitism. Operates in the canonical string space
where O(k!) redundant representations are eliminated.

Dependencies: numpy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from isalsr.core.canonical import canonical_string
from isalsr.core.node_types import OperationSet
from isalsr.core.string_to_dag import StringToDAG
from isalsr.evaluation.fitness import evaluate_expression
from isalsr.search.operators import one_point_crossover, point_mutation
from isalsr.search.random_search import random_isalsr_string

log = logging.getLogger(__name__)


class Population:
    """Manages a population of (canonical string, fitness) pairs.

    The key paper claim: evolutionary search in IsalSR's canonical space
    converges faster because O(k!) redundant representations are eliminated.
    """

    def __init__(
        self,
        size: int,
        num_variables: int,
        allowed_ops: OperationSet,
    ) -> None:
        self._size = size
        self._num_variables = num_variables
        self._allowed_ops = allowed_ops
        self._individuals: list[str] = []
        self._fitness: list[float] = []

    def initialize(
        self,
        x_data: np.ndarray[Any, np.dtype[Any]],
        y_true: np.ndarray[Any, np.dtype[Any]],
        max_tokens: int = 50,
        seed: int = 42,
    ) -> None:
        """Create initial population of random canonical strings."""
        rng = np.random.default_rng(seed)
        self._individuals = []
        self._fitness = []

        attempts = 0
        while len(self._individuals) < self._size and attempts < self._size * 20:
            attempts += 1
            raw = random_isalsr_string(self._num_variables, max_tokens, self._allowed_ops, rng)
            try:
                dag = StringToDAG(raw, self._num_variables, self._allowed_ops).run()
                if dag.node_count <= self._num_variables:
                    continue
                canon = canonical_string(dag)  # MANDATORY
                metrics = evaluate_expression(dag, x_data, y_true)
                self._individuals.append(canon)
                self._fitness.append(metrics["r2"])
            except Exception:  # noqa: BLE001
                continue

    def select_parents(
        self, n: int, rng: np.random.Generator, method: str = "tournament"
    ) -> list[str]:
        """Select n parents via tournament selection."""
        if method != "tournament":
            raise ValueError(f"Unknown selection method: {method}")

        parents: list[str] = []
        pop_size = len(self._individuals)
        if pop_size == 0:
            return parents

        for _ in range(n):
            i1, i2 = rng.integers(pop_size, size=2)
            winner = i1 if self._fitness[i1] >= self._fitness[i2] else i2
            parents.append(self._individuals[winner])
        return parents

    def evolve(
        self,
        x_data: np.ndarray[Any, np.dtype[Any]],
        y_true: np.ndarray[Any, np.dtype[Any]],
        n_generations: int = 50,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        seed: int = 42,
    ) -> dict[str, object]:
        """Run evolutionary search for n_generations.

        Returns dict with 'best_string', 'best_r2', 'history'.
        """
        rng = np.random.default_rng(seed)
        history: list[float] = []

        for _gen in range(n_generations):
            new_individuals: list[str] = []
            new_fitness: list[float] = []

            # Elitism: keep best individual.
            if self._individuals:
                best_idx = int(np.argmax(self._fitness))
                new_individuals.append(self._individuals[best_idx])
                new_fitness.append(self._fitness[best_idx])

            # Fill rest via crossover + mutation.
            while len(new_individuals) < self._size:
                parents = self.select_parents(2, rng)
                if len(parents) < 2:
                    break

                # Crossover.
                if rng.random() < crossover_rate:
                    c1, c2 = one_point_crossover(parents[0], parents[1], rng)
                else:
                    c1, c2 = parents[0], parents[1]

                for child_str in (c1, c2):
                    if len(new_individuals) >= self._size:
                        break
                    # Mutation.
                    if rng.random() < mutation_rate:
                        child_str = point_mutation(child_str, self._allowed_ops, rng)
                    # Canonicalize (MANDATORY) and evaluate.
                    try:
                        dag = StringToDAG(child_str, self._num_variables, self._allowed_ops).run()
                        if dag.node_count <= self._num_variables:
                            continue
                        canon = canonical_string(dag)
                        dag2 = StringToDAG(canon, self._num_variables, self._allowed_ops).run()
                        metrics = evaluate_expression(dag2, x_data, y_true)
                        new_individuals.append(canon)
                        new_fitness.append(metrics["r2"])
                    except Exception:  # noqa: BLE001
                        continue

            if new_individuals:
                self._individuals = new_individuals
                self._fitness = new_fitness

            best_r2 = max(self._fitness) if self._fitness else -1e10
            history.append(best_r2)

        best_idx = int(np.argmax(self._fitness)) if self._fitness else 0
        return {
            "best_string": self._individuals[best_idx] if self._individuals else "",
            "best_r2": self._fitness[best_idx] if self._fitness else -1e10,
            "history": history,
        }
