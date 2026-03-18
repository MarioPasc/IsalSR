"""Statistical tests for paired comparison experiments.

Implements the full statistical framework from the experimental design doc:
- Per-problem: Shapiro-Wilk → paired t-test or Wilcoxon signed-rank
- Multiple comparison: Holm-Bonferroni correction
- Cross-method: Friedman test + Nemenyi post-hoc
- Binary outcome: McNemar's test for solution rate

Reference: docs/design/experimental_design/isalsr_experimental_design.md, Section C.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ======================================================================
# Per-problem paired comparisons
# ======================================================================


def shapiro_wilk(differences: np.ndarray) -> tuple[float, float]:
    """Test normality of paired differences.

    Args:
        differences: Array of paired differences (isalsr - baseline).

    Returns:
        (W-statistic, p-value).
    """
    if len(differences) < 3:
        return 0.0, 0.0
    result = stats.shapiro(differences)
    return float(result.statistic), float(result.pvalue)


def paired_ttest(
    baseline: np.ndarray,
    isalsr: np.ndarray,
) -> tuple[float, float]:
    """Two-sided paired t-test.

    Args:
        baseline: Baseline metric values (one per seed).
        isalsr: IsalSR metric values (one per seed).

    Returns:
        (t-statistic, p-value).
    """
    differences = isalsr - baseline
    if np.all(differences == 0):
        return 0.0, 1.0
    result = stats.ttest_rel(isalsr, baseline)
    return float(result.statistic), float(result.pvalue)


def wilcoxon_signed_rank(
    baseline: np.ndarray,
    isalsr: np.ndarray,
) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test.

    Falls back to paired t-test if all differences are zero.

    Args:
        baseline: Baseline metric values (one per seed).
        isalsr: IsalSR metric values (one per seed).

    Returns:
        (W-statistic, p-value).
    """
    differences = isalsr - baseline
    if np.all(differences == 0):
        return 0.0, 1.0
    result = stats.wilcoxon(differences)
    return float(result.statistic), float(result.pvalue)


def mcnemar_test(
    baseline_success: np.ndarray,
    isalsr_success: np.ndarray,
) -> tuple[float, float]:
    """McNemar's test for paired binary outcomes (solution rate).

    Args:
        baseline_success: Boolean array (True = solution recovered).
        isalsr_success: Boolean array (True = solution recovered).

    Returns:
        (chi2-statistic, p-value).
    """
    b = np.asarray(baseline_success, dtype=bool)
    i = np.asarray(isalsr_success, dtype=bool)

    # Discordant pairs
    b_not_i = np.sum(b & ~i)  # baseline yes, isalsr no
    i_not_b = np.sum(~b & i)  # isalsr yes, baseline no

    n_discordant = b_not_i + i_not_b
    if n_discordant == 0:
        return 0.0, 1.0

    # McNemar's chi-squared (with continuity correction)
    chi2 = (abs(b_not_i - i_not_b) - 1) ** 2 / n_discordant
    p = float(stats.chi2.sf(chi2, df=1))
    return float(chi2), p


# ======================================================================
# Multiple comparison correction
# ======================================================================


def holm_bonferroni(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[float]:
    """Holm-Bonferroni correction for family-wise error rate.

    Args:
        p_values: Raw p-values from multiple tests.
        alpha: Significance level (default 0.05).

    Returns:
        Adjusted p-values (same order as input).
    """
    if not p_values:
        return []
    _, adjusted, _, _ = multipletests(p_values, alpha=alpha, method="holm")
    return [float(p) for p in adjusted]


# ======================================================================
# Cross-method analysis
# ======================================================================


def friedman_test(data_matrix: np.ndarray) -> tuple[float, float]:
    """Friedman test for comparing multiple methods across problems.

    Args:
        data_matrix: Shape (n_problems, n_methods). Each row is a problem,
            each column is a method's metric value.

    Returns:
        (chi2-statistic, p-value).
    """
    n_problems, n_methods = data_matrix.shape
    if n_methods < 3:
        raise ValueError(
            f"Friedman test requires >= 3 methods, got {n_methods}. "
            "For 2-method comparison, use paired t-test or Wilcoxon."
        )
    # scipy.stats.friedmanchisquare expects each column as a separate argument
    columns = [data_matrix[:, j] for j in range(n_methods)]
    result = stats.friedmanchisquare(*columns)
    return float(result.statistic), float(result.pvalue)


def nemenyi_posthoc(
    data_matrix: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Nemenyi post-hoc test after Friedman.

    Args:
        data_matrix: Shape (n_problems, n_methods).
        alpha: Significance level.

    Returns:
        Pairwise p-value matrix, shape (n_methods, n_methods).
    """
    import scikit_posthocs as sp

    # scikit-posthocs expects a DataFrame or array with specific format
    result = sp.posthoc_nemenyi_friedman(data_matrix)
    return np.array(result)


@dataclass(frozen=True)
class CDResult:
    """Critical difference diagram data."""

    cd_value: float  # critical difference value
    avg_ranks: np.ndarray  # average rank per method
    cliques: list[list[int]]  # groups of statistically equivalent methods


def critical_difference_data(
    data_matrix: np.ndarray,
    method_names: list[str],
    alpha: float = 0.05,
) -> CDResult:
    """Compute critical difference value and method groupings.

    Based on the Nemenyi test. Two methods are in the same clique if
    their average rank difference is less than CD.

    Args:
        data_matrix: Shape (n_problems, n_methods).
        method_names: Names for each method column.
        alpha: Significance level.

    Returns:
        CDResult with CD value, average ranks, and cliques.
    """
    n_problems, n_methods = data_matrix.shape

    # Compute ranks per problem (1 = best). For metrics where higher is
    # better, the caller should negate values before passing.
    ranks = np.zeros_like(data_matrix)
    for i in range(n_problems):
        ranks[i] = stats.rankdata(-data_matrix[i])  # higher value = lower rank

    avg_ranks = ranks.mean(axis=0)

    # Critical difference (Nemenyi, 1963)
    # CD = q_alpha * sqrt(n_methods * (n_methods + 1) / (6 * n_problems))
    # q_alpha from studentized range distribution
    from scipy.stats import studentized_range

    q_alpha = studentized_range.ppf(1 - alpha, n_methods, np.inf)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_problems))

    # Find cliques: groups where all pairwise rank differences < CD
    sorted_idx = np.argsort(avg_ranks)
    cliques: list[list[int]] = []

    for i in range(n_methods):
        clique = [sorted_idx[i]]
        for j in range(i + 1, n_methods):
            if abs(avg_ranks[sorted_idx[i]] - avg_ranks[sorted_idx[j]]) < cd:
                clique.append(sorted_idx[j])
            else:
                break
        if len(clique) > 1 and clique not in cliques:
            # Check this clique isn't a subset of an existing one
            is_subset = any(set(clique).issubset(set(c)) for c in cliques)
            if not is_subset:
                cliques.append(clique)

    return CDResult(
        cd_value=float(cd),
        avg_ranks=avg_ranks,
        cliques=cliques,
    )
