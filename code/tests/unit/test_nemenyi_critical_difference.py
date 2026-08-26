"""The Nemenyi critical difference must match Demsar (2006), Table 5.

The constant is easy to get wrong by a factor of sqrt(2), because scipy's
studentized range is the undivided statistic while Demsar tabulates it divided.
An inflated threshold does not invalidate a diagram, it makes it declare
differences indistinguishable that the test separates, which is a silent loss
of every conclusion the figure exists to support.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.models.analyzer.statistical_tests import critical_difference_data

# Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data
# Sets. JMLR 7:1-30, Table 5, alpha = 0.05.
DEMSAR_Q_05 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}


@pytest.mark.parametrize("k", sorted(DEMSAR_Q_05))
def test_critical_difference_matches_demsar_table(k: int) -> None:
    """CD = q_alpha sqrt(k(k+1)/6N) with q_alpha as Demsar tabulates it."""
    n_problems = 70
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(n_problems, k))

    result = critical_difference_data(matrix, [f"m{i}" for i in range(k)])
    expected = DEMSAR_Q_05[k] * np.sqrt(k * (k + 1) / (6 * n_problems))

    np.testing.assert_allclose(result.cd_value, expected, rtol=1e-3)


def test_critical_difference_shrinks_with_more_problems() -> None:
    """The threshold falls as 1/sqrt(N), so pooling more problems resolves more."""
    rng = np.random.default_rng(1)
    names = ["a", "b", "c"]
    small = critical_difference_data(rng.normal(size=(30, 3)), names)
    large = critical_difference_data(rng.normal(size=(120, 3)), names)

    np.testing.assert_allclose(small.cd_value / large.cd_value, 2.0, rtol=1e-6)


def test_three_groups_resolve_more_finely_than_six() -> None:
    """Splitting a six-group comparison by host tightens the threshold.

    This is why the manuscript reports one diagram per host solver rather than
    one pooled diagram over both.
    """
    rng = np.random.default_rng(2)
    names6 = [f"m{i}" for i in range(6)]
    six = critical_difference_data(rng.normal(size=(70, 6)), names6)
    three = critical_difference_data(rng.normal(size=(70, 3)), ["a", "b", "c"])

    assert three.cd_value < six.cd_value
    np.testing.assert_allclose(three.cd_value, 0.3960, atol=5e-4)
    np.testing.assert_allclose(six.cd_value, 0.9014, atol=5e-4)
