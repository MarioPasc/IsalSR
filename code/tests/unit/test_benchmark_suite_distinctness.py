"""No two benchmark problems may generate the same data (C4, 2026-08-04).

CPDT treats **each problem as one paired observation** and runs a sign or
Wilcoxon test over ``N`` problems, so two problems that generate identical data
are one observation counted twice and ``N`` is overstated.

That is not hypothetical. Until 2026-08-04 the D1 ``feynman`` tier contained
``I.12.1`` (``mu * N_s``) and ``I.34.27`` (``hbar * omega``), both of which
reduced to ``x_0*x_1`` on ``[1,5]^2`` because our implementation of I.34.27 had
folded the ``1/(2*pi)`` into the symbol name and dropped it from the target.
They produced **byte-identical** data at every seed, and had done so since
campaign C1. Stage C's criterion C4 found it; these tests keep it out.

Two problems are legitimately allowed to repeat data **across seeds**:
``Pagie-1`` and ``Keijzer-6`` sample deterministic grids, which is the published
protocol for both. They are named explicitly rather than tolerated by a blanket
rule, so a *new* seed-invariant problem still fails.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest
import sympy

from experiments.models.orchestrator import _BENCHMARK_REGISTRY, _generate_benchmark_data
from experiments.models.provenance import data_fingerprint

#: Sampling is a deterministic grid, so the seed cannot vary the data.
#: Mirrors ``experiments.scripts.c2_certify.SEED_INVARIANT_PROBLEMS``.
SEED_INVARIANT: frozenset[str] = frozenset({"Pagie-1", "Keijzer-6"})

_SEEDS = (0, 101)


def _all_fingerprints() -> dict[str, list[tuple[str, int]]]:
    """Map fingerprint -> the (problem, seed) pairs that produced it."""
    out: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for suite, (problems, _gen) in _BENCHMARK_REGISTRY.items():
        for bench in problems:
            for seed in _SEEDS:
                arrays = _generate_benchmark_data(suite, bench, 1000, 250, seed)
                out[data_fingerprint(*arrays)].append((bench["name"], seed))
    return out


@pytest.fixture(scope="module")
def fingerprints() -> dict[str, list[tuple[str, int]]]:
    return _all_fingerprints()


class TestNoTwoProblemsShareData:
    def test_no_cross_problem_duplicates(
        self, fingerprints: dict[str, list[tuple[str, int]]]
    ) -> None:
        """Distinct problems must never generate identical data."""
        offenders = {
            fp: sorted({name for name, _ in pairs})
            for fp, pairs in fingerprints.items()
            if len({name for name, _ in pairs}) > 1
        }
        assert not offenders, (
            "Distinct problems generate byte-identical data, so CPDT would count "
            f"one observation twice: {offenders}"
        )

    def test_seed_variation_except_declared_grids(
        self, fingerprints: dict[str, list[tuple[str, int]]]
    ) -> None:
        """A seed must change the data, unless the problem is a declared grid."""
        collapsed = {
            sorted({name for name, _ in pairs})[0]
            for pairs in fingerprints.values()
            if len(pairs) > 1 and len({name for name, _ in pairs}) == 1
        }
        unexpected = collapsed - SEED_INVARIANT
        assert not unexpected, (
            f"Seed does not reach the data generator for {sorted(unexpected)}. "
            "If the sampling is a deterministic grid by design, add it to "
            "SEED_INVARIANT here and to c2_certify.SEED_INVARIANT_PROBLEMS."
        )


class TestI3427CarriesItsConstant:
    """The specific regression: I.34.27 must not collapse onto I.12.1."""

    def _bench(self, name: str) -> dict:
        problems, _ = _BENCHMARK_REGISTRY["feynman"]
        bench = next((b for b in problems if b["name"] == name), None)
        assert bench is not None, f"{name} missing from the feynman tier"
        return bench

    def test_i34_27_differs_from_i12_1(self) -> None:
        a = _generate_benchmark_data("feynman", self._bench("I.12.1"), 1000, 250, 0)
        b = _generate_benchmark_data("feynman", self._bench("I.34.27"), 1000, 250, 0)
        assert data_fingerprint(*a) != data_fingerprint(*b)

    def test_i34_27_target_carries_one_over_two_pi(self) -> None:
        """y must equal x_0*x_1/(2*pi), not x_0*x_1."""
        bench = self._bench("I.34.27")
        x, y, _, _ = _generate_benchmark_data("feynman", bench, 1000, 250, 0)
        expected = x[:, 0] * x[:, 1] / (2.0 * np.pi)
        np.testing.assert_allclose(y, expected, rtol=1e-15, atol=0.0)

    def test_sympy_ground_truth_matches_target_fn(self) -> None:
        """``solution_recovered`` compares against this expression -- it must agree."""
        bench = self._bench("I.34.27")
        x, y, _, _ = _generate_benchmark_data("feynman", bench, 1000, 250, 0)
        fn = sympy.lambdify(bench["sympy_variables"], bench["sympy_expression"], "numpy")
        np.testing.assert_allclose(
            fn(*[x[:, i] for i in range(x.shape[1])]), y, rtol=1e-15, atol=0.0
        )
