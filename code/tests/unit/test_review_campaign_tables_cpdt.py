"""Unit tests for the p-value column of Table 3 (the paired test across problems).

Each contrast carries a pre-registered alternative (``CPDT_CONTRAST_POLICY``).
For a two-sided contrast the pipeline still records a one-sided p, taken in the
direction of the observed mean difference; that value describes the sample and
is not the test. Table 3 must therefore print, for every row, the p that the
row's own alternative defines, and must mark the two-sided rows so that a
reader of a column headed "one-sided" is not misled. These tests pin both.
"""

from __future__ import annotations

from typing import Any

import pytest

from experiments.models.analyzer.aggregation import CPDT_CONTRAST_POLICY
from experiments.scripts.review_campaign.config import METHODS
from experiments.scripts.review_campaign.tables import CPDT_ROWS, cpdt_summary, fmt_p

#: Arm pair of each contrast, keyed the way ``CPDT_CONTRAST_POLICY`` is.
ARMS = {"isalsr_vs_baseline": ("baseline", "isalsr"), "isalsr_vs_hash": ("hash", "isalsr")}

DAGGER = r"\dagger"


def _alternative(contrast: str, metric: str) -> str:
    policy = CPDT_CONTRAST_POLICY[ARMS[contrast]][metric]
    return "descriptive" if policy is None else policy


def _row(suite_size: int, method: str, contrast: str, metric: str, k: int) -> dict[str, Any]:
    """A cpdt.csv record whose one- and two-sided p differ, so they can be told apart."""
    alternative = _alternative(contrast, metric)
    descriptive = alternative == "descriptive"
    # Distinct, well-separated values per row; two-sided = 2 x one-sided as in SciPy.
    p_one = (k + 1) * 1.1e-4 if suite_size == 70 else (k + 1) * 3.3e-3
    return {
        "suite_size": float(suite_size),
        "method": method,
        "contrast": contrast,
        "metric": metric,
        "test": "descriptive_definitional_baseline" if descriptive else "wilcoxon_signed_rank",
        "alternative": alternative,
        "cohens_d": 0.5,
        "d_lo": 0.1,
        "d_hi": 0.9,
        "mean_delta": 0.01,
        "p_one_sided": float("nan") if descriptive else p_one,
        "p_two_sided": float("nan") if descriptive else 2 * p_one,
    }


@pytest.fixture
def cpdt_data() -> dict[str, Any]:
    rows = []
    k = 0
    for method in METHODS:
        for contrast, metric, _label in CPDT_ROWS:
            for n in (70, 50):
                rows.append(_row(n, method, contrast, metric, k))
                k += 1
    return {"cpdt": rows}


def _body_rows(tex: str) -> list[list[str]]:
    """The data rows of the tabular, split into cells (host header rows skipped)."""
    out = []
    for line in tex.splitlines():
        line = line.strip()
        if not line.endswith(r"\\") or line.startswith(r"\multicolumn") or "Metric" in line:
            continue
        out.append([c.strip() for c in line[: -len(r"\\")].split(" & ")])
    return out


def _expected(
    rows: list[dict[str, Any]], method: str, contrast: str, metric: str
) -> list[dict[str, Any]]:
    cells = []
    for n in (70.0, 50.0):
        (rec,) = [
            r
            for r in rows
            if r["suite_size"] == n
            and r["method"] == method
            and r["contrast"] == contrast
            and r["metric"] == metric
        ]
        cells.append(rec)
    return cells


def test_policy_marks_r2_test_vs_hash_two_sided() -> None:
    """The defect only exists if the policy says two-sided; pin that premise."""
    assert CPDT_CONTRAST_POLICY[("hash", "isalsr")]["r2_test"] == "two-sided"
    two_sided_rows = [(c, m) for c, m, _ in CPDT_ROWS if _alternative(c, m) == "two-sided"]
    assert two_sided_rows == [("isalsr_vs_hash", "r2_test")]


def test_every_row_prints_its_primary_p(cpdt_data: dict[str, Any]) -> None:
    """Two-sided rows print p_two_sided; directional rows print p_one_sided."""
    body = _body_rows(cpdt_summary(cpdt_data))
    assert len(body) == len(METHODS) * len(CPDT_ROWS)
    i = 0
    for method in METHODS:
        for contrast, metric, _label in CPDT_ROWS:
            cells = body[i]
            i += 1
            for printed, rec in zip(
                cells[4:6], _expected(cpdt_data["cpdt"], method, contrast, metric), strict=True
            ):
                alternative = rec["alternative"]
                if alternative == "descriptive":
                    assert printed == "---"
                    continue
                two_sided = alternative == "two-sided"
                key = "p_two_sided" if two_sided else "p_one_sided"
                # Compare the whole cell: stripping the dagger out of a cell with
                # no stars would leave an empty superscript, not the plain form.
                expected = fmt_p(rec[key], dagger=two_sided)
                assert printed == expected, (method, contrast, metric, printed)


def test_two_sided_rows_carry_the_dagger_and_only_they(cpdt_data: dict[str, Any]) -> None:
    body = _body_rows(cpdt_summary(cpdt_data))
    i = 0
    for _method in METHODS:
        for contrast, metric, _label in CPDT_ROWS:
            cells = body[i]
            i += 1
            two_sided = _alternative(contrast, metric) == "two-sided"
            for printed in cells[4:6]:
                assert (DAGGER in printed) == two_sided, (contrast, metric, printed)


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (1.8529947978981919e-09, r"$1.9{\times}10^{-9}{}^{***\dagger}$"),
        (1.24e-06, r"$1.2{\times}10^{-6}{}^{***\dagger}$"),
        (0.5229, r"$0.523{}^{\dagger}$"),
        (0.7527, r"$0.753{}^{\dagger}$"),
    ],
)
def test_dagger_sits_inside_the_math_superscript(p: float, expected: str) -> None:
    """One math group per cell: a trailing ``$^\\dagger$`` would open display math."""
    assert fmt_p(p, dagger=True) == expected


def test_fmt_p_without_dagger_is_unchanged() -> None:
    """Every other cell of every generated table keeps its exact former text."""
    assert fmt_p(1.8e-13) == r"$1.8{\times}10^{-13}{}^{***}$"
    assert fmt_p(0.261) == r"$0.261$"
    assert fmt_p(0.011) == r"$0.011{}^{*}$"
    assert fmt_p(float("nan")) == "---"
