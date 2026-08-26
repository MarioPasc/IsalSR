"""Build every LaTeX table the manuscript needs from the derived data.

Writes into ``<analyses>/tables``. Each file is a bare ``tabular`` or a full
float, matching what the manuscript's existing caption expects, so that filling
a placeholder is a paste rather than a redesign.

Files
-----
tab_three_axis.tex            main paper, three-axis summary, one row per host
tab_cpdt_summary.tex          main paper, paired test across problems
tab_phi_by_host.tex           main paper, the share phi, one row per host
tab_key_cost.tex              main paper, per-candidate cost of the two keys
tab_supp_phi_per_problem.tex  supplementary, phi per host and per problem
tab_supp_k_range.tex          supplementary, Bingo overhead stratified by k
table_supplementary_udfs.tex  supplementary, per-problem UDFS results
table_supplementary_bingo.tex supplementary, per-problem Bingo results

Usage
-----
    python -m experiments.scripts.review_campaign.tables [--analyses DIR]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.scripts.review_campaign.config import (  # noqa: E402
    METHODS,
    SUITES,
    add_common_args,
)

HOST_LABEL = {"udfs": "UDFS", "bingo": "Bingo"}

#: Order the suites appear in, and the label each carries in a table.
SUITE_ORDER = {name: i for i, name in enumerate(SUITES)}
SUITE_LABEL = {
    "nguyen": "Nguyen",
    "feynman": "AI Feynman",
    "hard": "Hard",
    "cherrypicked": "Structural",
    "roundoff": "Portfolio",
    "feynman_remainder": "AI Feynman (extension)",
    "strogatz": "ODE-Strogatz",
}


def tex_escape(name: str) -> str:
    """Escape a problem name for LaTeX."""
    return name.replace("_", r"\_")


def fmt(x: float | None, digits: int = 2, *, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    return f"{x:{'+' if signed else ''}.{digits}f}"


def fmt_p(p: float | None) -> str:
    """A p-value with significance marks, in the manuscript's style."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "---"
    mark = "{}^{***}" if p < 0.001 else "{}^{**}" if p < 0.01 else "{}^{*}" if p < 0.05 else ""
    if p >= 0.01:
        return f"${p:.3f}{mark}$"
    exponent = math.floor(math.log10(p))
    mantissa = p / 10**exponent
    return f"${mantissa:.1f}{{\\times}}10^{{{exponent}}}{mark}$"


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value in {"", "None"}:
                row[key] = None
                continue
            with contextlib.suppress(ValueError):
                row[key] = float(value)
    return rows


def load(analyses: Path) -> dict[str, Any]:
    data = analyses / "data"
    with (analyses / "values" / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    return {
        "summary": summary,
        "per_problem": read_csv(data / "per_problem.csv"),
        "phi": read_csv(data / "phi.csv"),
        "cpdt": read_csv(data / "cpdt.csv"),
        "speedup": read_csv(data / "speedup.csv"),
        "k_strata": read_csv(data / "overhead_by_k.csv"),
        "paired": read_csv(data / "per_problem_paired.csv"),
    }


def pick(rows: list[dict[str, Any]], **where: Any) -> dict[str, Any] | None:
    hits = [r for r in rows if all(r[k] == v for k, v in where.items())]
    if len(hits) > 1:
        raise SystemExit(f"ambiguous lookup {where}: {len(hits)} rows")
    return hits[0] if hits else None


# ----------------------------------------------------------------------
# Main paper
# ----------------------------------------------------------------------


def three_axis(d: dict[str, Any]) -> str:
    """One row per host: reduction, quality and cost side by side."""
    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccccccccc@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Search space} & \multicolumn{3}{c}{Regression quality}"
        r" & \multicolumn{4}{c}{Computational cost} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-11}",
        r"Method & $\rho$ & $\rho_{\mathrm{ser}}$ & Red."
        r" & $R^2_{\mathrm{test}}$ (NA/NH/IS) & $d$ & $p$"
        r" & $T_{\mathrm{canon}}$/ms & $T_{\mathrm{eval}}$/ms & OH & $S$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        s = d["summary"][method]
        iso, hsh = s["isalsr"], s["hash"]
        r2 = pick(
            d["cpdt"],
            suite_size=70.0,
            method=method,
            contrast="isalsr_vs_baseline",
            metric="r2_test",
        )
        row = " & ".join(
            [
                HOST_LABEL[method],
                f"${fmt(iso['rho_over_problems']['mean'])} \\pm "
                f"{fmt(iso['rho_over_problems']['std'])}$",
                f"${fmt(hsh['rho_over_problems']['mean'])} \\pm "
                f"{fmt(hsh['rho_over_problems']['std'])}$",
                f"${fmt(100 * iso['r_over_problems']['mean'], 1)}\\%$",
                f"${fmt(s['baseline']['r2_test_mean_over_problems'], 3)}$ / "
                f"${fmt(hsh['r2_test_mean_over_problems'], 3)}$ / "
                f"${fmt(iso['r2_test_mean_over_problems'], 3)}$",
                f"${fmt(r2['cohens_d'], 2, signed=True)}$",
                fmt_p(r2["p_one_sided"]),
                f"${fmt(iso['key_ms_over_cells']['median'], 3)}$",
                f"${fmt(iso['eval_ms_over_cells']['median'], 2)}$",
                f"${fmt(iso['overhead_pct_over_cells']['median'], 2)}\\%$",
                f"${fmt(iso['S_over_cells']['median'], 2)}$",
            ]
        )
        lines.append(f"    {row} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


CPDT_ROWS = (
    ("isalsr_vs_baseline", "r2_test", r"$R^2_{\mathrm{te}}$"),
    ("isalsr_vs_baseline", "r2_train", r"$R^2_{\mathrm{tr}}$"),
    ("isalsr_vs_baseline", "nrmse_test", r"NRMSE"),
    ("isalsr_vs_baseline", "empirical_reduction_factor", r"$\rho$"),
    ("isalsr_vs_hash", "r2_test", r"$R^2_{\mathrm{te}}$"),
    ("isalsr_vs_hash", "empirical_reduction_factor", r"$\rho$"),
    ("isalsr_vs_hash", "redundancy_rate", r"$r$"),
)

CONTRAST_LABEL = {
    "isalsr_vs_baseline": "NA",
    "isalsr_vs_hash": "NH",
}


def cpdt_summary(d: dict[str, Any]) -> str:
    """The paired test across problems, both contrasts, both suite sizes."""
    lines = [
        r"\setlength{\tabcolsep}{1.2pt}",
        r"\begin{tabular}{@{}llcrll@{}}",
        r"\toprule",
        r"Metric & vs. & $d$ $[95\%$ CI$]$ & $\bar{\delta}$"
        r" & $p_{\mathrm{1s}}^{(70)}$ & $p_{\mathrm{1s}}^{(50)}$ \\",
        r"\midrule",
    ]
    for i, method in enumerate(METHODS):
        if i:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{HOST_LABEL[method]}}}}} \\")
        for contrast, metric, label in CPDT_ROWS:
            r70 = pick(d["cpdt"], suite_size=70.0, method=method, contrast=contrast, metric=metric)
            r50 = pick(d["cpdt"], suite_size=50.0, method=method, contrast=contrast, metric=metric)
            descriptive = r70["test"] == "descriptive_definitional_baseline"
            row = " & ".join(
                [
                    label,
                    CONTRAST_LABEL[contrast],
                    f"${fmt(r70['cohens_d'], 2, signed=True)}$ "
                    f"$[{fmt(r70['d_lo'], 2, signed=True)},{fmt(r70['d_hi'], 2, signed=True)}]$",
                    f"${fmt(r70['mean_delta'], 4, signed=True)}$",
                    "---" if descriptive else fmt_p(r70["p_one_sided"]),
                    "---" if descriptive else fmt_p(r50["p_one_sided"]),
                ]
            )
            lines.append(f"    {row} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def phi_by_host(d: dict[str, Any]) -> str:
    """The share of removable redundancy that only an isomorphism test reaches."""
    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrrl@{}}",
        r"\toprule",
        r"Host & $N/10^3$ & $\rho$ & $\rho_{\mathrm{ser}}$ & $\Delta r$"
        r" & $\phi$ $[\min, \max]$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        rows = [r for r in d["phi"] if r["method"] == method]
        s = d["summary"][method]
        n_is = sum(r["n_cand_isalsr"] for r in rows) / len(rows)
        phi = s["phi"]["over_problems"]
        dr = s["phi"]["delta_r_over_problems"]
        row = " & ".join(
            [
                HOST_LABEL[method],
                f"${n_is / 1000:,.0f}$".replace(",", "{,}"),
                f"${fmt(s['isalsr']['rho_over_problems']['mean'], 3)}$",
                f"${fmt(s['hash']['rho_over_problems']['mean'], 3)}$",
                f"${fmt(100 * dr['mean'], 1)}\\%$",
                f"${fmt(phi['mean'], 3)}$ $[{fmt(phi['min'], 3)}, {fmt(phi['max'], 3)}]$",
            ]
        )
        lines.append(f"    {row} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def key_cost(d: dict[str, Any]) -> str:
    """Per-candidate cost of the serialization key and of the canonical string."""
    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Host & $\mathrm{ser}(D)$ & $\fcs_D$ & Ratio"
        r" & OH$_{\mathrm{NH}}$ & OH$_{\mathrm{IS}}$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        s = d["summary"][method]
        cost = s["key_cost"]
        row = " & ".join(
            [
                HOST_LABEL[method],
                f"${fmt(cost['hash_ms']['mean'], 3)}$\\,ms",
                f"${fmt(cost['isalsr_ms']['mean'], 3)}$\\,ms",
                f"${fmt(cost['isalsr_over_hash']['mean'], 1)}\\times$",
                f"${fmt(s['hash']['overhead_pct_over_cells']['median'], 2)}\\%$",
                f"${fmt(s['isalsr']['overhead_pct_over_cells']['median'], 2)}\\%$",
            ]
        )
        lines.append(f"    {row} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Supplementary
# ----------------------------------------------------------------------


def phi_per_problem(d: dict[str, Any]) -> str:
    """phi per host and per problem, the two hosts side by side."""
    problems = sorted(
        {r["problem"] for r in d["phi"]},
        key=lambda p: (
            SUITE_ORDER[next(r["suite"] for r in d["phi"] if r["problem"] == p)],
            p,
        ),
    )
    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{4}{c}{UDFS} & \multicolumn{4}{c}{Bingo} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"Problem & $\rho_{\mathrm{ser}}$ & $\rho$ & $\Delta r$ & $\phi$"
        r" & $\rho_{\mathrm{ser}}$ & $\rho$ & $\Delta r$ & $\phi$ \\",
        r"\midrule",
    ]
    current_suite = None
    for problem in problems:
        suite = next(r["suite"] for r in d["phi"] if r["problem"] == problem)
        if suite != current_suite:
            lines.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\textit{{{SUITE_LABEL[suite]}}}}} \\")
            current_suite = suite
        cells = [tex_escape(problem)]
        for method in METHODS:
            rec = pick(d["phi"], method=method, problem=problem)
            cells += [
                f"${fmt(rec['rho_ser'], 3)}$",
                f"${fmt(rec['rho'], 3)}$",
                f"${fmt(100 * rec['delta_r'], 1)}\\%$",
                f"${fmt(rec['phi'], 3)}$",
            ]
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def k_range_overhead(d: dict[str, Any]) -> str:
    """Bingo key overhead by k-range, for both keys."""
    lines = [
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Naive hash} & \multicolumn{3}{c}{\IsalSR} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"$k$-range & Runs & OH & $T_{\mathrm{key}}$ & Runs & OH & $T_{\mathrm{key}}$ \\",
        r"\midrule",
    ]
    ranges = [r["k_range"] for r in d["k_strata"] if r["method"] == "bingo"]
    for k_range in sorted(set(ranges), key=lambda s: int(s.split(",")[0][1:])):
        lo, hi = k_range[1:-1].split(",")
        cells = [f"$[{lo}, {hi})$"]
        for arm in ("hash", "isalsr"):
            rec = pick(d["k_strata"], method="bingo", arm=arm, k_range=k_range)
            cells += [
                f"${int(rec['n_runs'])}$",
                f"${fmt(rec['overhead_pct_mean'], 1)}\\%$",
                f"${fmt(rec['key_ms_mean'], 3)}$\\,ms",
            ]
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def best_worst(
    values: dict[str, float], *, higher_is_better: bool, digits: int = 4
) -> dict[str, str]:
    """Mark the best arm bold and the worst underlined; ties stay unmarked.

    A non-finite entry is never marked in either direction: an undefined
    quantity is not comparable to a finite one.
    """
    finite = {k: v for k, v in values.items() if v is not None and math.isfinite(v)}
    out = {k: fmt(v, digits) for k, v in values.items()}
    if len(set(finite.values())) < 2:
        return out
    best = (max if higher_is_better else min)(finite, key=lambda k: finite[k])
    worst = (min if higher_is_better else max)(finite, key=lambda k: finite[k])
    out[best] = rf"\mathbf{{{fmt(finite[best], digits)}}}"
    out[worst] = rf"\underline{{{fmt(finite[worst], digits)}}}"
    return out


def per_problem_table(d: dict[str, Any], method: str) -> str:
    """One row per problem, all three arms, for one host."""
    rows = [r for r in d["per_problem"] if r["method"] == method]
    problems = sorted(
        {r["problem"] for r in rows},
        key=lambda p: (SUITE_ORDER[next(r["suite"] for r in rows if r["problem"] == p)], p),
    )
    lines = [
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}lrrrrrrlrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{$R^2$ test} & \multicolumn{3}{c}{NRMSE test}"
        r" & Effect size & $\rho$"
        r" & \multicolumn{3}{c}{Wall clock (s)} & OH \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-8}"
        r" \cmidrule(lr){9-9} \cmidrule(lr){10-12} \cmidrule(lr){13-13}",
        r"Problem & NA & NH & IS & NA & NH & IS & $d$\,[95\% CI]"
        r" & IS & NA & NH & IS & \\",
        r"\midrule",
    ]
    current_suite = None
    for problem in problems:
        arms = {a: pick(rows, problem=problem, arm=a) for a in ("baseline", "hash", "isalsr")}
        suite = arms["isalsr"]["suite"]
        if suite != current_suite:
            lines.append(rf"\multicolumn{{13}}{{@{{}}l}}{{\textit{{{SUITE_LABEL[suite]}}}}} \\")
            current_suite = suite
        r2 = best_worst({a: arms[a]["r2_test_mean"] for a in arms}, higher_is_better=True, digits=3)
        nrmse = best_worst(
            {a: arms[a]["nrmse_test_mean"] for a in arms}, higher_is_better=False, digits=3
        )
        wall = best_worst(
            {a: arms[a]["wall_s_mean"] for a in arms}, higher_is_better=False, digits=0
        )
        effect = pick(
            d["paired"],
            method=method,
            problem=problem,
            contrast="isalsr_vs_baseline",
            metric="r2_test",
        )
        phi = pick(d["phi"], method=method, problem=problem)
        cells = [
            tex_escape(problem),
            *[f"${r2[a]}$" for a in ("baseline", "hash", "isalsr")],
            *[f"${nrmse[a]}$" for a in ("baseline", "hash", "isalsr")],
            f"${fmt(effect['cohens_d'], 2, signed=True)}$\\,"
            f"$[{fmt(effect['d_lo'], 2, signed=True)},"
            f"{fmt(effect['d_hi'], 2, signed=True)}]$",
            f"${fmt(phi['rho'], 2)} \\pm {fmt(arms['isalsr']['rho_std'], 2)}$",
            *[f"${wall[a]}$" for a in ("baseline", "hash", "isalsr")],
            f"${fmt(arms['isalsr']['overhead_pct_mean'], 2)}\\%$",
        ]
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def suite_breakdown(analyses: Path, d: dict[str, Any]) -> str:
    """Response-letter table: the paired test per problem source and per suite size.

    Reviewer 3 asks whether extending the suite moved the pooled result on its
    own. The two blocks answer that directly: one row per source, and the pooled
    test at both sizes.
    """
    sizes = [(name, SUITE_LABEL[name]) for name in SUITES]
    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrlrl@{}}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{UDFS} & \multicolumn{2}{c}{Bingo} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"Problem source & $n$ & $d$ & $p_{\mathrm{1s}}$ & $d$ & $p_{\mathrm{1s}}$ \\",
        r"\midrule",
    ]
    pipeline = analyses / "pipeline" / "by_suite"
    for key, label in sizes:
        cells = []
        n_problems = 0
        for method in METHODS:
            with (pipeline / f"cross_problem_dominance_{method}_{key}.json").open(
                encoding="utf-8"
            ) as handle:
                rec = json.load(handle)["r2_test"]
            n_problems = rec["n_problems"]
            cells += [f"${fmt(rec['cohens_d'], 2, signed=True)}$", fmt_p(rec["p_value_one_sided"])]
        lines.append(f"    {label} & ${n_problems}$ & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    for size in (50, 70):
        cells = []
        for method in METHODS:
            rec = pick(
                d["cpdt"],
                suite_size=float(size),
                method=method,
                contrast="isalsr_vs_baseline",
                metric="r2_test",
            )
            cells += [f"${fmt(rec['cohens_d'], 2, signed=True)}$", fmt_p(rec["p_one_sided"])]
        label = "Pooled, submitted suite" if size == 50 else "Pooled, revised suite"
        lines.append(rf"    \textit{{{label}}} & ${size}$ & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


#: Values as the submitted manuscript printed them, recovered from the tagged
#: source at commit ``8574240^``: Table 2 (three-axis), Table 3 (paired test)
#: and Appendix D. They are historical constants and are never recomputed.
SUBMITTED = {
    "rho_udfs": "$1.56 \\pm 0.24$",
    "rho_bingo": "$1.83 \\pm 0.09$",
    "red_udfs": "$34.2\\%$",
    "red_bingo": "$45.2\\%$",
    "r2_udfs": "$0.770$ / $0.779$",
    "r2_bingo": "$0.965$ / $0.966$",
    "d_r2_udfs": "$+0.30$",
    "p_r2_udfs": "$5.9{\\times}10^{-5}$",
    "d_r2_bingo": "$+0.05$",
    "p_r2_bingo": "$4.4{\\times}10^{-4}$",
    "tcanon_udfs": "$0.296$\\,ms",
    "tcanon_bingo": "$0.817$\\,ms",
    "teval_udfs": "${\\sim}519$\\,ms",
    "teval_bingo": "$1.29$\\,ms",
    "oh_udfs": "$0.05\\%$",
    "oh_bingo": "$39.2\\%$",
    "s_udfs": "$1.07$",
    "s_bingo": "$0.93$",
    "faster_bingo": "$4$ of $50$",
    "runs": "$2{,}640$ stated, $6{,}000$ intended",
    "cells_udfs": "$1{,}500$",
    "cells_bingo": "$1{,}465$",
    "nan_bingo": "$2$",
}


def continuity(d: dict[str, Any]) -> str:
    """Response-letter appendix: submitted numbers against revised ones.

    Reviewer 2 checked the submitted values individually, so a wholesale change
    of numbers needs a map rather than an assurance. Rows whose definition or
    population changed are marked, because a reader comparing them like for like
    would be comparing two different quantities.
    """
    s = d["summary"]

    def cell(method: str, arm: str, path: tuple[str, ...], digits: int, suffix: str = "") -> str:
        node: Any = s[method][arm]
        for key in path:
            node = node[key]
        return f"${fmt(node, digits)}${suffix}"

    rows: list[tuple[str, str, str, str]] = []
    for method, tag in (("udfs", "udfs"), ("bingo", "bingo")):
        host = HOST_LABEL[method]
        iso = s[method]["isalsr"]
        r2 = pick(
            d["cpdt"],
            suite_size=70.0,
            method=method,
            contrast="isalsr_vs_baseline",
            metric="r2_test",
        )
        rows += [
            (
                rf"$\rho$, {host}",
                SUBMITTED[f"rho_{tag}"],
                f"${fmt(iso['rho_over_problems']['mean'])} \\pm "
                f"{fmt(iso['rho_over_problems']['std'])}$",
                "A, S",
            ),
            (
                rf"Redundancy rate, {host}",
                SUBMITTED[f"red_{tag}"],
                f"${fmt(100 * iso['r_over_problems']['mean'], 1)}\\%$",
                "A, S",
            ),
            (
                rf"$R^2_{{\mathrm{{test}}}}$ NA / IS, {host}",
                SUBMITTED[f"r2_{tag}"],
                f"${fmt(s[method]['baseline']['r2_test_mean_over_problems'], 3)}$ / "
                f"${fmt(iso['r2_test_mean_over_problems'], 3)}$",
                "S",
            ),
            (
                rf"$d$, $R^2_{{\mathrm{{test}}}}$, {host}",
                SUBMITTED[f"d_r2_{tag}"],
                f"${fmt(r2['cohens_d'], 2, signed=True)}$",
                "S",
            ),
            (
                rf"$p_{{\mathrm{{1s}}}}$, $R^2_{{\mathrm{{test}}}}$, {host}",
                SUBMITTED[f"p_r2_{tag}"],
                fmt_p(r2["p_one_sided"]).replace("{}^{***}", ""),
                "S",
            ),
            (
                rf"$T_{{\mathrm{{canon}}}}$, {host}",
                SUBMITTED[f"tcanon_{tag}"],
                cell(method, "isalsr", ("key_ms_over_cells", "median"), 3, "\\,ms"),
                "E, A",
            ),
            (
                rf"$T_{{\mathrm{{eval}}}}$, {host}",
                SUBMITTED[f"teval_{tag}"],
                cell(method, "isalsr", ("eval_ms_over_cells", "median"), 2, "\\,ms"),
                "S",
            ),
            (
                rf"Overhead, {host}",
                SUBMITTED[f"oh_{tag}"],
                f"${fmt(iso['overhead_pct_over_cells']['median'], 2)}\\%$",
                "E, A",
            ),
            (
                rf"$S$, {host}",
                SUBMITTED[f"s_{tag}"],
                cell(method, "isalsr", ("S_over_cells", "median"), 2),
                "E, A, S, H",
            ),
        ]
    rows += [
        (
            "Problems with $T_{\\mathrm{IS}} < T_{\\mathrm{NA}}$, Bingo",
            SUBMITTED["faster_bingo"],
            f"${s['bingo']['isalsr']['n_problems_faster_than_native']}$ of $70$",
            "E, A, S, H",
        ),
        ("Total runs", SUBMITTED["runs"], f"${s['n_cells']:,}$".replace(",", "{,}"), "D"),
        (
            "Seed-problem cells, UDFS",
            SUBMITTED["cells_udfs"],
            f"${s['udfs']['isalsr']['n_cells']:,}$".replace(",", "{,}"),
            "D",
        ),
        (
            "Seed-problem cells, Bingo",
            SUBMITTED["cells_bingo"],
            f"${s['bingo']['isalsr']['n_cells']:,}$".replace(",", "{,}"),
            "D",
        ),
        ("Cells with a non-finite metric", SUBMITTED["nan_bingo"], "$0$", "D"),
    ]

    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lllc@{}}",
        r"\toprule",
        r"Quantity & Submitted & Revised & Cause \\",
        r"\midrule",
    ]
    for label, before, after, cause in rows:
        lines.append(f"    {label} & {before} & {after} & {cause} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    d = load(args.analyses)
    out_dir = args.analyses / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {
        "tab_three_axis.tex": three_axis(d),
        "tab_cpdt_summary.tex": cpdt_summary(d),
        "tab_phi_by_host.tex": phi_by_host(d),
        "tab_key_cost.tex": key_cost(d),
        "tab_supp_phi_per_problem.tex": phi_per_problem(d),
        "tab_supp_k_range.tex": k_range_overhead(d),
        "tab_letter_suite_breakdown.tex": suite_breakdown(args.analyses, d),
        "tab_letter_continuity.tex": continuity(d),
        "table_supplementary_udfs_body.tex": per_problem_table(d, "udfs"),
        "table_supplementary_bingo_body.tex": per_problem_table(d, "bingo"),
    }
    for name, body in written.items():
        (out_dir / name).write_text(body, encoding="utf-8")
        print(f"  {name:36s} {len(body.splitlines()):4d} lines")


if __name__ == "__main__":
    main()
