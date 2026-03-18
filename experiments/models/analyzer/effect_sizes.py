"""Effect size computation for paired experiments.

Cohen's d (paired) with bootstrap confidence intervals.

Reference: Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def cohens_d_paired(differences: np.ndarray) -> float:
    """Compute Cohen's d for paired differences.

    d = mean(differences) / std(differences, ddof=1)

    Interpretation (Cohen, 1988):
        |d| < 0.2:  negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large

    Args:
        differences: Array of paired differences (isalsr - baseline).

    Returns:
        Cohen's d value.
    """
    if len(differences) < 2:
        return 0.0
    sd = np.std(differences, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(differences) / sd)


def cohens_d_ci_bootstrap(
    differences: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for Cohen's d.

    Args:
        differences: Array of paired differences.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        (lower, upper) bounds of the CI.
    """
    rng = np.random.default_rng(seed)
    n = len(differences)
    if n < 2:
        return 0.0, 0.0

    boot_ds = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(differences, size=n, replace=True)
        sd = np.std(sample, ddof=1)
        boot_ds[b] = np.mean(sample) / sd if sd > 0 else 0.0

    alpha = 1 - ci
    lower = float(np.percentile(boot_ds, 100 * alpha / 2))
    upper = float(np.percentile(boot_ds, 100 * (1 - alpha / 2)))
    return lower, upper


def mean_diff_ci(
    differences: np.ndarray,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Confidence interval for mean paired difference (t-distribution).

    Args:
        differences: Array of paired differences.
        ci: Confidence level (default 0.95).

    Returns:
        (mean, lower, upper).
    """
    n = len(differences)
    if n < 2:
        m = float(np.mean(differences)) if n > 0 else 0.0
        return m, m, m

    mean = float(np.mean(differences))
    se = float(np.std(differences, ddof=1) / np.sqrt(n))
    alpha = 1 - ci
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    margin = t_crit * se

    return mean, mean - margin, mean + margin
