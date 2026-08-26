"""Extract every number the synthetic-scalability appendix needs from the fragments.

Reads the per-cell ``synth_k{K}_m{M}.csv`` fragments written by
``run_synthetic_scalability.py`` and emits, as JSON and as an ``\\input``-able
LaTeX snippet, the twelve placeholder values used by the supplementary's
synthetic-scalability appendix: nine per-``k`` ``median [q1--q3]`` timing cells,
the power-law exponent, the mean per-permutation time at ``k = 9``, and the
total number of canonicalizations.

Definitions implemented here (stated because they are ambiguous):

* ``median_ms`` is the median **over expressions** of that expression's mean
  canonicalization time per permutation (``mean_canon_time_s`` x 1000). It is
  *not* the median over individual permutations, which the fragment schema does
  not record. ``q1_ms``/``q3_ms`` are the 25th/75th percentiles of the same
  per-expression quantity (linear interpolation, matching
  ``generate_fig_synthetic_scalability._aggregate_per_k``).
* ``median_us``/``q1_us``/``q3_us`` are the same three quantities in
  **microseconds** (``mean_canon_time_s`` x 1e6). Both units are emitted; a
  ``_ms`` key always holds milliseconds and a ``_us`` key always holds
  microseconds. The table and the LaTeX macros report microseconds, because on
  the compiled engine every per-``k`` median rounds to 0.01--0.02 ms at two
  decimals and the ``k`` trend disappears.
* ``powerlaw_exponent`` is the OLS slope of ``log(mean_canon_time_s)`` on
  ``log(k)`` over every row of every ``(k, m)`` cell. ``k = 1`` is **included**:
  ``log(1) = 0`` is a legitimate predictor value and does not make the design
  matrix rank-deficient as long as any ``k > 1`` row is present.

Rounding happens exactly once, in the LaTeX emitter, from the full-precision
JSON values.

Author: Mario Pascual Gonzalez (mpascual@uma.es)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants (the *expected* shape of a complete run).
# ---------------------------------------------------------------------------
EXPECTED_K_VALUES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)
EXPECTED_M_VALUES: tuple[int, ...] = (1, 2, 3)
EXPECTED_N_FRAGMENTS: int = len(EXPECTED_K_VALUES) * len(EXPECTED_M_VALUES)

FRAGMENT_GLOB: str = "synth_k*_m*.csv"
_FRAGMENT_RE = re.compile(r"^synth_k(\d+)_m(\d+)\.csv$")

METADATA_FILENAME: str = "synthetic_scalability_metadata.json"
HARDWARE_KEYS: tuple[str, ...] = ("cpu_model", "engine", "build_hash", "isa_level")

REQUIRED_COLUMNS: tuple[str, ...] = (
    "k",
    "m",
    "expr_id",
    "n_perms",
    "n_unique_canonicals",
    "rho",
    "mean_canon_time_s",
    "timeout_count",
)

_ALPHA: float = 0.05
_RHO_RTOL: float = 1e-6

_ORDINALS: tuple[str, ...] = (
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Eleven",
    "Twelve",
)


class SyntheticDataError(RuntimeError):
    """Raised when the fragment set cannot support the requested values."""


# ===================================================================
# Loading
# ===================================================================


def discover_fragments(fragments_dir: str) -> list[Path]:
    """List the per-cell fragment CSVs in ``fragments_dir``, sorted by (k, m).

    Args:
        fragments_dir: Directory holding ``synth_k{K}_m{M}.csv`` files.

    Returns:
        Fragment paths sorted by the ``(k, m)`` encoded in their names.

    Raises:
        SyntheticDataError: If the directory does not exist.
    """
    root = Path(fragments_dir)
    if not root.is_dir():
        raise SyntheticDataError(f"Fragment directory does not exist: {fragments_dir}")

    keyed: list[tuple[tuple[int, int], Path]] = []
    for path in root.glob(FRAGMENT_GLOB):
        match = _FRAGMENT_RE.match(path.name)
        if match is None:
            logger.warning("Ignoring unparseable fragment name: %s", path.name)
            continue
        keyed.append(((int(match.group(1)), int(match.group(2))), path))
    keyed.sort(key=lambda item: item[0])
    return [path for _, path in keyed]


def load_fragments(
    fragments_dir: str,
    *,
    allow_partial: bool = False,
    expected_fragments: int = EXPECTED_N_FRAGMENTS,
) -> tuple[pd.DataFrame, list[Path]]:
    """Load and concatenate the fragment CSVs.

    Args:
        fragments_dir: Directory holding the fragments.
        allow_partial: Permit fewer than ``expected_fragments`` fragments. A
            partial set yields medians that are visually indistinguishable from
            complete ones, so this must be requested explicitly.
        expected_fragments: Number of ``(k, m)`` cells a complete run produces.

    Returns:
        The concatenated rows and the list of fragment paths actually read.

    Raises:
        SyntheticDataError: If no fragments exist, if the set is incomplete and
            ``allow_partial`` is False, or if a fragment lacks a required column.
    """
    files = discover_fragments(fragments_dir)
    if not files:
        raise SyntheticDataError(
            f"No fragments matching {FRAGMENT_GLOB!r} in {fragments_dir}. Nothing to extract."
        )
    if len(files) < expected_fragments and not allow_partial:
        raise SyntheticDataError(
            f"Found {len(files)} fragments in {fragments_dir}, expected {expected_fragments}. "
            "Refusing to emit values from an incomplete run; pass --allow-partial "
            "to override (the output is then NOT publishable)."
        )
    if len(files) > expected_fragments:
        logger.warning(
            "Found %d fragments, more than the expected %d.", len(files), expected_fragments
        )

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
        if missing:
            raise SyntheticDataError(f"{path.name} is missing column(s): {missing}")
        frames.append(frame)

    rows = pd.concat(frames, ignore_index=True)
    if rows.empty:
        raise SyntheticDataError(f"All {len(files)} fragments in {fragments_dir} are empty.")
    logger.info("Loaded %d rows from %d fragments in %s", len(rows), len(files), fragments_dir)
    return rows, files


# ===================================================================
# Per-k aggregation
# ===================================================================


def aggregate_per_k(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Aggregate the rows by ``k``, pooling the ``m`` cells.

    Timing statistics are emitted twice, in milliseconds (``*_ms``) and in
    microseconds (``*_us``); each key name states its own unit.

    Args:
        rows: Concatenated fragment rows.

    Returns:
        One dict per ``k``, in ascending ``k`` order.
    """
    out: list[dict[str, Any]] = []
    for k in sorted(int(v) for v in rows["k"].unique()):
        group = rows[rows["k"] == k]
        times_s = group["mean_canon_time_s"].to_numpy(dtype=float)
        times_ms = times_s * 1000.0
        times_us = times_s * 1e6
        kfact = math.factorial(k)
        rho = group["rho"].to_numpy(dtype=float)
        n_unique = group["n_unique_canonicals"].to_numpy(dtype=int)
        perms_per_expr = sorted(int(v) for v in group["n_perms"].unique())
        rho_ok = np.isclose(rho, float(kfact), rtol=_RHO_RTOL, atol=0.0)

        # The Perms column is measured, not assumed. Flag any disagreement with
        # k! loudly instead of printing a factorial the run never performed.
        perms_matches_kfact = perms_per_expr == [kfact]
        if not perms_matches_kfact:
            logger.error(
                "k=%d: n_perms per expression is %s, not k! = %d. The table's "
                "Perms column will report the measured value.",
                k,
                perms_per_expr,
                kfact,
            )

        out.append(
            {
                "k": k,
                "k_factorial": kfact,
                "perms_per_expression": perms_per_expr[0] if len(perms_per_expr) == 1 else None,
                "perms_per_expression_values": perms_per_expr,
                "perms_equals_kfact": bool(perms_matches_kfact),
                "median_ms": float(np.median(times_ms)),
                "q1_ms": float(np.percentile(times_ms, 25)),
                "q3_ms": float(np.percentile(times_ms, 75)),
                "mean_ms": float(np.mean(times_ms)),
                "median_us": float(np.median(times_us)),
                "q1_us": float(np.percentile(times_us, 25)),
                "q3_us": float(np.percentile(times_us, 75)),
                "mean_us": float(np.mean(times_us)),
                "n_expressions": int(len(group)),
                "n_permutations": int(group["n_perms"].sum()),
                "m_values": sorted(int(v) for v in group["m"].unique()),
                "rho_equals_kfact": bool(np.all(rho_ok)),
                "n_rho_equals_kfact": int(np.count_nonzero(rho_ok)),
                "invariance_pct": float(100.0 * np.count_nonzero(n_unique == 1) / len(group)),
                "n_invariant": int(np.count_nonzero(n_unique == 1)),
                "timeouts": int(group["timeout_count"].sum()),
            }
        )
    return out


# ===================================================================
# Power-law fit
# ===================================================================


def fit_powerlaw(rows: pd.DataFrame) -> dict[str, Any]:
    """OLS fit of ``log(mean_canon_time_s)`` on ``log(k)`` over all rows.

    ``k = 1`` is retained: ``log(1) = 0`` is a valid predictor value and the fit
    stays well-posed provided at least one ``k > 1`` row is present.

    Args:
        rows: Concatenated fragment rows.

    Returns:
        Slope, its standard error, intercept, R^2, the number of rows used and
        the number dropped as non-positive.

    Raises:
        SyntheticDataError: If fewer than two distinct ``k`` values survive.
    """
    k_arr = rows["k"].to_numpy(dtype=float)
    t_arr = rows["mean_canon_time_s"].to_numpy(dtype=float)
    mask = (k_arr > 0.0) & (t_arr > 0.0) & np.isfinite(t_arr)
    n_dropped = int(np.count_nonzero(~mask))
    log_k = np.log(k_arr[mask])
    log_t = np.log(t_arr[mask])
    if len(np.unique(log_k)) < 2:
        raise SyntheticDataError("Power-law fit needs at least two distinct k values.")

    result = stats.linregress(log_k, log_t)
    return {
        "specification": "OLS: log(mean_canon_time_s) ~ log(k), one point per expression row",
        "k_equals_one_included": bool(np.any(k_arr[mask] == 1.0)),
        "exponent": float(result.slope),
        "exponent_stderr": float(result.stderr),
        "intercept_log_seconds": float(result.intercept),
        "r_squared": float(result.rvalue**2),
        "p_value": float(result.pvalue),
        "n_points": int(mask.sum()),
        "n_dropped_nonpositive": n_dropped,
    }


# ===================================================================
# m-dependence
# ===================================================================


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment of *pvals* (order preserved)."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def assess_m_dependence(rows: pd.DataFrame) -> dict[str, Any]:
    """Test the manuscript's claim that timing shows no dependence on ``m``.

    Runs a Kruskal-Wallis test across the ``m`` groups within each ``k`` (Holm
    corrected across ``k``), plus one pooled Kruskal-Wallis on log-times
    centred by their per-``k`` median, which removes the ``k`` effect that would
    otherwise dominate a naive pooled test.

    Args:
        rows: Concatenated fragment rows.

    Returns:
        Per-``k`` medians per ``m``, the per-``k`` tests, the pooled test and a
        plain-language verdict.
    """
    per_k: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for k in sorted(int(v) for v in rows["k"].unique()):
        group = rows[rows["k"] == k]
        medians: dict[str, float] = {}
        samples: list[np.ndarray] = []
        for m in sorted(int(v) for v in group["m"].unique()):
            vals = group[group["m"] == m]["mean_canon_time_s"].to_numpy(dtype=float) * 1000.0
            medians[str(m)] = float(np.median(vals))
            samples.append(vals)
        entry: dict[str, Any] = {"k": k, "median_ms_by_m": medians}
        if len(samples) >= 2 and all(len(s) > 0 for s in samples):
            h_stat, p_val = stats.kruskal(*samples)
            entry["kruskal_h"] = float(h_stat)
            entry["kruskal_p"] = float(p_val)
            raw_p.append(float(p_val))
        else:
            entry["kruskal_h"] = None
            entry["kruskal_p"] = None
        per_k.append(entry)

    adjusted = _holm(raw_p) if raw_p else []
    it = iter(adjusted)
    n_sig = 0
    for entry in per_k:
        if entry["kruskal_p"] is None:
            entry["kruskal_p_holm"] = None
            continue
        p_adj = next(it)
        entry["kruskal_p_holm"] = p_adj
        n_sig += int(p_adj < _ALPHA)

    # Pooled test on per-k median-centred log times (removes the k effect).
    log_t = np.log(rows["mean_canon_time_s"].to_numpy(dtype=float))
    k_arr = rows["k"].to_numpy(dtype=int)
    m_arr = rows["m"].to_numpy(dtype=int)
    finite = np.isfinite(log_t)
    centred = np.empty_like(log_t)
    centred[:] = np.nan
    for k in np.unique(k_arr):
        sel = (k_arr == k) & finite
        if np.any(sel):
            centred[sel] = log_t[sel] - float(np.median(log_t[sel]))
    pooled_samples = [
        centred[(m_arr == m) & np.isfinite(centred)] for m in sorted(np.unique(m_arr))
    ]
    pooled_samples = [s for s in pooled_samples if len(s) > 0]
    if len(pooled_samples) >= 2:
        h_pooled, p_pooled = stats.kruskal(*pooled_samples)
        pooled: dict[str, Any] = {
            "test": "Kruskal-Wallis on per-k median-centred log(mean_canon_time_s)",
            "h": float(h_pooled),
            "p": float(p_pooled),
            "n_groups": len(pooled_samples),
        }
    else:
        pooled = {
            "test": "Kruskal-Wallis on per-k median-centred log(mean_canon_time_s)",
            "h": None,
            "p": None,
            "n_groups": len(pooled_samples),
        }

    pooled_p = pooled["p"]
    dependent = n_sig > 0 or (pooled_p is not None and pooled_p < _ALPHA)
    verdict = (
        "DEPENDENCE DETECTED: the 'no observable dependence on m' claim is not "
        "supported by this run."
        if dependent
        else "No dependence on m detected at alpha=0.05."
    )
    return {
        "alpha": _ALPHA,
        "per_k": per_k,
        "pooled": pooled,
        "n_k_significant_after_holm": n_sig,
        "n_k_tested": len(raw_p),
        "dependence_detected": bool(dependent),
        "verdict": verdict,
    }


# ===================================================================
# Provenance
# ===================================================================


def _mtime_iso(path: Path) -> str:
    """UTC ISO-8601 modification time of *path*, to whole seconds."""
    ts = _dt.datetime.fromtimestamp(path.stat().st_mtime, tz=_dt.UTC)
    return ts.replace(microsecond=0).isoformat()


def collect_provenance(
    fragments_dir: str,
    files: list[Path],
    *,
    allow_partial: bool,
    expected_fragments: int,
) -> dict[str, Any]:
    """Gather hardware metadata and fragment file provenance.

    Args:
        fragments_dir: Directory the fragments were read from.
        files: Fragment paths actually read.
        allow_partial: Whether an incomplete fragment set was accepted.
        expected_fragments: Number of cells a complete run produces.

    Returns:
        Hardware block (if the metadata JSON is present), fragment count, cell
        coverage and per-file modification times.
    """
    meta_path = Path(fragments_dir) / METADATA_FILENAME
    hardware: dict[str, Any] | None = None
    metadata_note: str | None = None
    if meta_path.is_file():
        with meta_path.open() as handle:
            meta = json.load(handle)
        block = meta.get("hardware")
        if isinstance(block, dict):
            hardware = {key: block.get(key) for key in HARDWARE_KEYS}
            hardware["cpu_count"] = block.get("cpu_count")
            hardware["git_hash"] = block.get("git_hash")
        metadata_note = (
            "Each cell run overwrites this file, so it describes the LAST cell "
            "written, not necessarily every fragment."
        )
    else:
        metadata_note = f"{METADATA_FILENAME} not found in {fragments_dir}."

    cells = []
    for path in files:
        match = _FRAGMENT_RE.match(path.name)
        assert match is not None  # discover_fragments filtered these
        cells.append([int(match.group(1)), int(match.group(2))])

    return {
        "fragments_dir": os.path.abspath(fragments_dir),
        "n_fragments_found": len(files),
        "n_fragments_expected": expected_fragments,
        "is_complete": len(files) >= expected_fragments,
        "allow_partial": allow_partial,
        "cells": cells,
        "fragment_mtimes_utc": {path.name: _mtime_iso(path) for path in files},
        "hardware": hardware,
        "metadata_note": metadata_note,
    }


# ===================================================================
# Assembly
# ===================================================================


def build_values(
    rows: pd.DataFrame,
    files: list[Path],
    fragments_dir: str,
    *,
    allow_partial: bool,
    expected_fragments: int,
) -> dict[str, Any]:
    """Assemble the full value dictionary emitted as JSON.

    Args:
        rows: Concatenated fragment rows.
        files: Fragment paths actually read.
        fragments_dir: Directory the fragments were read from.
        allow_partial: Whether an incomplete fragment set was accepted.
        expected_fragments: Number of cells a complete run produces.

    Returns:
        Nested dict with ``per_k``, ``overall``, ``m_dependence`` and
        ``provenance`` sections.
    """
    per_k = aggregate_per_k(rows)
    times_ms_all = rows["mean_canon_time_s"].to_numpy(dtype=float) * 1000.0
    k_max = int(rows["k"].max())
    k9 = rows[rows["k"] == 9]
    mean_ms_at_k9 = (
        float(np.mean(k9["mean_canon_time_s"].to_numpy(dtype=float) * 1000.0)) if len(k9) else None
    )

    mean_ms_at_kmax = float(
        np.mean(rows[rows["k"] == k_max]["mean_canon_time_s"].to_numpy(dtype=float) * 1000.0)
    )

    overall: dict[str, Any] = {
        "powerlaw": fit_powerlaw(rows),
        "mean_ms_at_k9": mean_ms_at_k9,
        "mean_us_at_k9": None if mean_ms_at_k9 is None else mean_ms_at_k9 * 1000.0,
        "mean_ms_at_kmax": mean_ms_at_kmax,
        "mean_us_at_kmax": mean_ms_at_kmax * 1000.0,
        "k_max_present": k_max,
        "total_canonicalizations": int(rows["n_perms"].sum()),
        "total_expressions": int(len(rows)),
        "total_timeouts": int(rows["timeout_count"].sum()),
        "invariance_pct_all": float(
            100.0
            * np.count_nonzero(rows["n_unique_canonicals"].to_numpy(dtype=int) == 1)
            / len(rows)
        ),
        "rho_equals_kfact_all": bool(all(entry["rho_equals_kfact"] for entry in per_k)),
        "mean_ms_all": float(np.mean(times_ms_all)),
        "mean_us_all": float(np.mean(times_ms_all)) * 1000.0,
        "k_values_present": sorted(int(v) for v in rows["k"].unique()),
        "m_values_present": sorted(int(v) for v in rows["m"].unique()),
    }

    return {
        "median_definition": (
            "median over expressions of that expression's mean canonicalization "
            "time per permutation; q1/q3 are the 25th/75th percentiles of the "
            "same per-expression quantity with linear interpolation. Keys ending "
            "in _ms are milliseconds (mean_canon_time_s x 1e3); keys ending in "
            "_us are microseconds (mean_canon_time_s x 1e6). The LaTeX table and "
            "macros report microseconds"
        ),
        "per_k": per_k,
        "overall": overall,
        "m_dependence": assess_m_dependence(rows),
        "provenance": collect_provenance(
            fragments_dir,
            files,
            allow_partial=allow_partial,
            expected_fragments=expected_fragments,
        ),
    }


# ===================================================================
# LaTeX emission
# ===================================================================


def _ordinal_word(n: int) -> str:
    """Letters-only spelling of *n* for use inside a LaTeX macro name."""
    if 0 <= n < len(_ORDINALS):
        return _ORDINALS[n]
    return "".join(_ORDINALS[int(digit)] for digit in str(n))


def _thousands(n: int) -> str:
    """Format *n* with LaTeX-safe thousands separators."""
    return f"{n:,}".replace(",", r"{,}")


def render_latex(values: dict[str, Any]) -> str:
    """Render the ``\\newcommand`` snippet from the full-precision values.

    All rounding happens here, once, from the JSON values.

    Args:
        values: The dict produced by :func:`build_values`.

    Returns:
        The complete LaTeX snippet, newline terminated.
    """
    prov = values["provenance"]
    overall = values["overall"]
    fit = overall["powerlaw"]

    lines: list[str] = [
        "% Auto-generated by experiments/synthetic_scalability/extract_synthetic_values.py",
        "% Do not edit by hand; regenerate from the fragment CSVs.",
        f"% fragments: {prov['n_fragments_found']}/{prov['n_fragments_expected']}"
        f" (complete={prov['is_complete']})",
        f"% {values['median_definition']}",
        "",
    ]
    lines.append("% \\synthTimeK* cells are MICROSECONDS, matching the table.")
    for entry in values["per_k"]:
        name = f"synthTimeK{_ordinal_word(int(entry['k']))}"
        cell = f"{entry['median_us']:.2f}\\,[{entry['q1_us']:.2f}\\text{{--}}{entry['q3_us']:.2f}]"
        lines.append(f"\\newcommand{{\\{name}}}{{{cell}}}")

    lines.append("")
    lines.append(f"\\newcommand{{\\synthPowerlawExponent}}{{{fit['exponent']:.2f}}}")
    lines.append(f"\\newcommand{{\\synthPowerlawExponentSE}}{{{fit['exponent_stderr']:.3f}}}")
    lines.append(f"\\newcommand{{\\synthPowerlawRSquared}}{{{fit['r_squared']:.3f}}}")
    # Both units are emitted under self-describing names: \synthMeanMsKNine is
    # milliseconds, \synthMeanUsKNine is microseconds. The prose uses the latter.
    if overall["mean_ms_at_k9"] is None:
        lines.append(r"\newcommand{\synthMeanMsKNine}{N/A}")
        lines.append(r"\newcommand{\synthMeanUsKNine}{N/A}")
    else:
        lines.append(f"\\newcommand{{\\synthMeanMsKNine}}{{{overall['mean_ms_at_k9']:.2f}}}")
        lines.append(f"\\newcommand{{\\synthMeanUsKNine}}{{{overall['mean_us_at_k9']:.2f}}}")
    lines.append(
        f"\\newcommand{{\\synthTotalCanonicalizations}}"
        f"{{{_thousands(int(overall['total_canonicalizations']))}}}"
    )
    lines.append(
        f"\\newcommand{{\\synthTotalExpressions}}"
        f"{{{_thousands(int(overall['total_expressions']))}}}"
    )
    return "\n".join(lines) + "\n"


def render_table_body(rows: pd.DataFrame) -> str:
    """Render the bare ``tabular`` body for ``tab:synthetic_scalability``.

    The rows are **not** reimplemented here. They are produced by
    ``generate_fig_synthetic_scalability._export_latex_table``, which is the one
    existing implementation of this table, and the surrounding float, caption
    and label are then stripped so the supplementary keeps its curated caption.

    Args:
        rows: Concatenated fragment rows.

    Returns:
        ``\\setlength{\\tabcolsep}{3pt}`` followed by the ``tabular``
        environment, newline terminated.

    Raises:
        SyntheticDataError: If the generator's output contains no ``tabular``.
    """
    # Lazy import: pulls in matplotlib, which nothing else in this module needs.
    from experiments.synthetic_scalability.generate_fig_synthetic_scalability import (
        _aggregate_per_k,
        _export_latex_table,
    )

    records: list[dict[str, float | int]] = [
        {
            "k": int(rec["k"]),
            "m": int(rec["m"]),
            "expr_id": int(rec["expr_id"]),
            "n_perms": int(rec["n_perms"]),
            "n_unique_canonicals": int(rec["n_unique_canonicals"]),
            "rho": float(rec["rho"]),
            "rho_over_kfact": float(rec["rho_over_kfact"]),
            "mean_canon_time_s": float(rec["mean_canon_time_s"]),
            "timeout_count": int(rec["timeout_count"]),
        }
        for rec in rows.to_dict("records")
    ]

    with tempfile.TemporaryDirectory() as scratch:
        tex_path = _export_latex_table(_aggregate_per_k(records), scratch)
        generated = Path(tex_path).read_text()

    start = generated.find(r"\begin{tabular}")
    end = generated.find(r"\end{tabular}")
    if start < 0 or end < 0:
        raise SyntheticDataError(
            "generate_fig_synthetic_scalability._export_latex_table produced no tabular."
        )
    body = generated[start : end + len(r"\end{tabular}")]
    # The generator indents its rows for a float; dedent for a bare \input.
    body = "\n".join(line.strip() for line in body.splitlines())

    return (
        "% Auto-generated by experiments/synthetic_scalability/extract_synthetic_values.py\n"
        "% Rows come from generate_fig_synthetic_scalability._export_latex_table;\n"
        "% the float, caption and label are stripped so the document supplies them.\n"
        "\\setlength{\\tabcolsep}{3pt}\n" + body + "\n"
    )


# ===================================================================
# CLI
# ===================================================================


def write_outputs(
    values: dict[str, Any],
    rows: pd.DataFrame,
    out_dir: str,
) -> tuple[Path, Path, Path]:
    """Write the JSON, the ``\\newcommand`` snippet and the tabular body.

    Args:
        values: The dict produced by :func:`build_values`.
        rows: Concatenated fragment rows, for the delegated table body.
        out_dir: Destination directory, created if absent.

    Returns:
        Paths to ``synthetic_values.json``, ``synthetic_values.tex`` and
        ``tab_supp_synthetic_body.tex``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "synthetic_values.json"
    tex_path = out / "synthetic_values.tex"
    table_path = out / "tab_supp_synthetic_body.tex"
    json_path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    tex_path.write_text(render_latex(values))
    table_path.write_text(render_table_body(rows))
    return json_path, tex_path, table_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract supplementary values from synthetic-scalability fragments.",
    )
    parser.add_argument(
        "--fragments",
        type=str,
        required=True,
        help="Directory containing synth_k*_m*.csv fragments.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for synthetic_values.{json,tex}.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Emit values from fewer than the expected number of fragments. "
            "The result is exploratory and must not be published."
        ),
    )
    parser.add_argument(
        "--expected-fragments",
        type=int,
        default=EXPECTED_N_FRAGMENTS,
        help=f"Cells a complete run produces (default: {EXPECTED_N_FRAGMENTS}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        Process exit status (0 on success, 2 on a data error).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)
    try:
        rows, files = load_fragments(
            args.fragments,
            allow_partial=args.allow_partial,
            expected_fragments=args.expected_fragments,
        )
        values = build_values(
            rows,
            files,
            args.fragments,
            allow_partial=args.allow_partial,
            expected_fragments=args.expected_fragments,
        )
    except SyntheticDataError as exc:
        logger.error("%s", exc)
        return 2

    json_path, tex_path, table_path = write_outputs(values, rows, args.out)
    for path in (json_path, tex_path, table_path):
        logger.info("Wrote %s", path)

    total_timeouts = int(values["overall"]["total_timeouts"])
    if total_timeouts:
        logger.error(
            "TIMEOUTS: %d permutations timed out across %d rows. The claim "
            "'no permutation timed out' is FALSE for this run.",
            total_timeouts,
            values["overall"]["total_expressions"],
        )
    else:
        logger.info("total_timeouts = 0: no permutation timed out.")

    logger.info(
        "exponent=%.4f (SE %.4f, R2 %.4f); %s",
        values["overall"]["powerlaw"]["exponent"],
        values["overall"]["powerlaw"]["exponent_stderr"],
        values["overall"]["powerlaw"]["r_squared"],
        values["m_dependence"]["verdict"],
    )
    if not values["provenance"]["is_complete"]:
        logger.warning(
            "PARTIAL RUN: %d/%d fragments. Values are exploratory, not publishable.",
            values["provenance"]["n_fragments_found"],
            values["provenance"]["n_fragments_expected"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
