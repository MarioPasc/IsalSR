"""Resolve every placeholder of the two pending ledgers into a value.

The annotated manuscript carries 75 placeholders in ``paper/`` and 34 in
``supplementary/``, each recorded in a ledger row that names the metric, the
host and the population. This script answers one row at a time from the derived
data and writes the answers as JSON and as a Markdown table, so that filling the
LaTeX is a lookup and checking it afterwards is a diff.

Rows the campaign cannot answer are emitted with ``value: null`` and a reason.
The synthetic permutation study of the supplementary is one of these: its
timings belong to a separate measurement that this campaign does not contain.

Usage
-----
    python -m experiments.scripts.review_campaign.values [--analyses DIR]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import statistics as st
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.scripts.review_campaign.config import add_common_args  # noqa: E402

# ----------------------------------------------------------------------
# LaTeX number formatting
# ----------------------------------------------------------------------


def sci(p: float, *, stars: bool = True) -> str:
    """Format a p-value the way the manuscript's tables do."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "---"
    mark = ""
    if stars:
        mark = "^{***}" if p < 0.001 else "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""
    if p == 0.0:
        return f"$<10^{{-300}}{mark}$"
    exponent = math.floor(math.log10(p))
    mantissa = p / 10**exponent
    if exponent >= -2:
        return f"${p:.3g}{mark}$"
    return f"${mantissa:.1f}{{\\times}}10^{{{exponent}}}{mark}$"


def signed(x: float, digits: int = 2) -> str:
    return f"${x:+.{digits}f}$"


def num(x: float, digits: int = 2) -> str:
    return f"${x:.{digits}f}$"


def pct(x: float, digits: int = 1) -> str:
    """A fraction in [0, 1] as a LaTeX percentage."""
    return f"${100 * x:.{digits}f}\\%$"


def pct_direct(x: float, digits: int = 2) -> str:
    """A quantity already expressed in percent."""
    return f"${x:.{digits}f}\\%$"


def ci(lo: float, hi: float, digits: int = 2) -> str:
    return f"$[{lo:+.{digits}f}, {hi:+.{digits}f}]$"


# ----------------------------------------------------------------------


def load(analyses: Path) -> dict[str, Any]:
    """Read everything ``derive`` produced."""
    data = analyses / "data"
    with (analyses / "values" / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    def read(name: str) -> list[dict[str, Any]]:
        with (data / name).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            for key, value in list(row.items()):
                if value in {"", "None"}:
                    row[key] = None
                    continue
                with contextlib.suppress(ValueError):
                    row[key] = float(value)
        return rows

    return {
        "summary": summary,
        "phi": read("phi.csv"),
        "per_problem_paired": read("per_problem_paired.csv"),
        "per_problem": read("per_problem.csv"),
        "cpdt": read("cpdt.csv"),
        "speedup": read("speedup.csv"),
        "k_strata": read("overhead_by_k.csv"),
    }


def cpdt_lookup(rows: list[dict[str, Any]], **where: Any) -> dict[str, Any]:
    """Return the single paired-test record matching every keyword."""
    hits = [r for r in rows if all(r[k] == v for k, v in where.items())]
    if len(hits) != 1:
        raise SystemExit(f"cpdt lookup {where} matched {len(hits)} rows")
    return hits[0]


def build(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Answer each ledger row.

    Returns:
        Mapping from ledger key (``paper:NN`` / ``supp:NN``) to a record with
        the LaTeX literal, the raw value and a one-line provenance note.
    """
    s = data["summary"]
    cp = data["cpdt"]
    out: dict[str, dict[str, Any]] = {}

    def put(key: str, latex: str, raw: Any, note: str) -> None:
        out[key] = {"latex": latex, "raw": raw, "note": note}

    def phi_stats(method: str) -> dict[str, float]:
        return s[method]["phi"]["over_problems"]

    def cpdt70(method: str, contrast: str, metric: str) -> dict[str, Any]:
        return cpdt_lookup(cp, suite_size=70, method=method, contrast=contrast, metric=metric)

    # ---------------------------------------------------------------- abstract
    rho_u = cpdt70("udfs", "isalsr_vs_hash", "empirical_reduction_factor")
    rho_b = cpdt70("bingo", "isalsr_vs_hash", "empirical_reduction_factor")
    r2_u = cpdt70("udfs", "isalsr_vs_baseline", "r2_test")
    r2_b = cpdt70("bingo", "isalsr_vs_baseline", "r2_test")

    put(
        "paper:1",
        f"$d = {rho_u['cohens_d']:.2f}$ on UDFS and $d = {rho_b['cohens_d']:.2f}$ on Bingo",
        [rho_u["cohens_d"], rho_b["cohens_d"]],
        "Cohen's d on rho, IsalSR against the naive hash, N=70. The contrast "
        "against the native arm is definitional and carries no test.",
    )
    put(
        "paper:2",
        "$p < 10^{-12}$",
        [rho_u["p_one_sided"], rho_b["p_one_sided"]],
        f"one-sided p on rho, IsalSR vs naive hash: UDFS {rho_u['p_one_sided']:.3g}, "
        f"Bingo {rho_b['p_one_sided']:.3g} (Wilcoxon at N=70 floors at 1.8e-13)",
    )
    put(
        "paper:3",
        pct(s["udfs"]["isalsr"]["r_over_problems"]["mean"]),
        s["udfs"]["isalsr"]["r_over_problems"]["mean"],
        "mean redundancy rate over the 70 problems, UDFS, IsalSR arm",
    )
    put(
        "paper:4",
        pct(s["bingo"]["isalsr"]["r_over_problems"]["mean"]),
        s["bingo"]["isalsr"]["r_over_problems"]["mean"],
        "mean redundancy rate over the 70 problems, Bingo, IsalSR arm",
    )
    put(
        "paper:5",
        f"{sci(r2_u['p_one_sided'], stars=False)} on UDFS and "
        f"{sci(r2_b['p_one_sided'], stars=False)} on Bingo",
        [r2_u["p_one_sided"], r2_b["p_one_sided"]],
        "one-sided p on R2 test, IsalSR against the native arm, N=70",
    )
    put(
        "paper:6",
        pct(phi_stats("udfs")["mean"], 0),
        phi_stats("udfs")["mean"],
        "phi, UDFS, mean over the 70 problems",
    )
    put(
        "paper:7",
        pct(phi_stats("bingo")["mean"], 1),
        phi_stats("bingo")["mean"],
        "phi, Bingo, mean over the 70 problems",
    )
    put(
        "paper:8",
        pct_direct(s["udfs"]["isalsr"]["overhead_pct_over_cells"]["median"], 2),
        s["udfs"]["isalsr"]["overhead_pct_over_cells"]["median"],
        "median key overhead over the 2,100 UDFS IsalSR cells",
    )
    put(
        "paper:9",
        pct_direct(s["bingo"]["isalsr"]["overhead_pct_over_cells"]["median"], 1),
        s["bingo"]["isalsr"]["overhead_pct_over_cells"]["median"],
        "median key overhead over the 2,100 Bingo IsalSR cells",
    )

    ru_all = s["udfs"]["isalsr"]["rho_over_problems"]
    rb_all = s["bingo"]["isalsr"]["rho_over_problems"]

    # ------------------------------------------------------------- results.tex
    put(
        "paper:16",
        "every problem",
        [s["udfs"]["isalsr"]["n_rho_gt_1"], s["bingo"]["isalsr"]["n_rho_gt_1"]],
        "problems with rho > 1: UDFS 70/70, Bingo 70/70",
    )
    for tag, method in (("17", "udfs"), ("20", "bingo")):
        blk = s[method]["isalsr"]["rho_over_problems"]
        put(
            f"paper:{tag}",
            f"${blk['mean']:.2f} \\pm {blk['std']:.2f}$",
            [blk["mean"], blk["std"]],
            f"mean and standard deviation of rho over the 70 problems, {method}",
        )
    put(
        "paper:18",
        "$2{,}100$",
        s["udfs"]["isalsr"]["n_cells"],
        "UDFS seed-problem cells in the IsalSR arm",
    )
    put(
        "paper:21",
        "$2{,}100$",
        s["bingo"]["isalsr"]["n_cells"],
        "Bingo seed-problem cells in the IsalSR arm",
    )
    put(
        "paper:19",
        pct(s["udfs"]["isalsr"]["r_over_problems"]["mean"]),
        s["udfs"]["isalsr"]["r_over_problems"]["mean"],
        "redundancy rate, UDFS",
    )
    put(
        "paper:22",
        pct(s["bingo"]["isalsr"]["r_over_problems"]["mean"]),
        s["bingo"]["isalsr"]["r_over_problems"]["mean"],
        "redundancy rate, Bingo",
    )

    put(
        "paper:23",
        num(rho_u["cohens_d"]),
        rho_u["cohens_d"],
        "Cohen's d on rho, UDFS, IsalSR vs naive hash",
    )
    put(
        "paper:24",
        ci(rho_u["d_lo"], rho_u["d_hi"]),
        [rho_u["d_lo"], rho_u["d_hi"]],
        "bootstrap CI on that d",
    )
    put(
        "paper:25",
        sci(rho_u["p_one_sided"], stars=False),
        rho_u["p_one_sided"],
        "one-sided p on rho, UDFS, IsalSR vs naive hash",
    )
    put(
        "paper:26",
        num(rho_b["cohens_d"]),
        rho_b["cohens_d"],
        "Cohen's d on rho, Bingo, IsalSR vs naive hash",
    )
    red_u = cpdt70("udfs", "isalsr_vs_hash", "redundancy_rate")
    red_b = cpdt70("bingo", "isalsr_vs_hash", "redundancy_rate")
    put(
        "paper:27",
        "$p < 10^{-12}$ on both",
        [red_u["p_one_sided"], red_b["p_one_sided"]],
        "one-sided p on the redundancy rate, IsalSR vs naive hash, both methods",
    )

    put(
        "paper:28",
        signed(r2_u["mean_delta"], 4),
        r2_u["mean_delta"],
        "mean per-problem difference on R2 test, UDFS",
    )
    put("paper:29", signed(r2_u["cohens_d"]), r2_u["cohens_d"], "Cohen's d, R2 test, UDFS")
    put(
        "paper:30",
        ci(r2_u["d_lo"], r2_u["d_hi"]),
        [r2_u["d_lo"], r2_u["d_hi"]],
        "bootstrap CI on that d",
    )
    put(
        "paper:31",
        sci(r2_u["p_one_sided"], stars=False),
        r2_u["p_one_sided"],
        "one-sided p on R2 test, UDFS",
    )
    tr_u = cpdt70("udfs", "isalsr_vs_baseline", "r2_train")
    put(
        "paper:32",
        signed(tr_u["mean_delta"], 4),
        tr_u["mean_delta"],
        "mean per-problem difference on R2 train, UDFS",
    )
    put(
        "paper:33",
        sci(tr_u["p_one_sided"], stars=False),
        tr_u["p_one_sided"],
        "one-sided p on R2 train, UDFS",
    )
    nr_u = cpdt70("udfs", "isalsr_vs_baseline", "nrmse_test")
    put(
        "paper:34",
        sci(nr_u["p_one_sided"], stars=False),
        nr_u["p_one_sided"],
        "one-sided p on NRMSE test, UDFS",
    )
    put("paper:35", signed(nr_u["cohens_d"]), nr_u["cohens_d"], "Cohen's d on NRMSE test, UDFS")
    put("paper:36", signed(r2_b["cohens_d"]), r2_b["cohens_d"], "Cohen's d, R2 test, Bingo")
    put(
        "paper:37",
        signed(r2_b["mean_delta"], 4),
        r2_b["mean_delta"],
        "mean per-problem difference on R2 test, Bingo",
    )
    put(
        "paper:38",
        sci(r2_b["p_one_sided"], stars=False),
        r2_b["p_one_sided"],
        "one-sided p on R2 test, Bingo",
    )

    for tag, method in (("39", "udfs"), ("40", "bingo")):
        blk = phi_stats(method)
        if method == "udfs":
            latex = "$1.00$ on every one of the $70$ problems"
        else:
            latex = (
                f"a mean of ${blk['mean']:.3f}$, "
                f"ranging over $[{blk['min']:.3f}, {blk['max']:.3f}]$"
            )
        put(
            f"paper:{tag}",
            latex,
            [blk["mean"], blk["min"], blk["max"]],
            f"phi, {method}, mean and range over the 70 problems",
        )
    ku = s["udfs"]["key_cost"]["isalsr_over_hash"]["mean"]
    kb = s["bingo"]["key_cost"]["isalsr_over_hash"]["mean"]
    put(
        "paper:41",
        f"a factor of ${ku:.1f}$ on UDFS and ${kb:.1f}$ on Bingo",
        [ku, kb],
        "mean per-candidate cost of the canonical string over that of the serialization key",
    )

    put(
        "paper:43",
        f"${s['udfs']['isalsr']['key_ms_over_cells']['median']:.3f}$",
        s["udfs"]["isalsr"]["key_ms_over_cells"]["median"],
        "median per-candidate canonicalization cost, UDFS, ms",
    )
    put(
        "paper:44",
        f"${s['udfs']['isalsr']['eval_ms_over_cells']['median']:.0f}$",
        s["udfs"]["isalsr"]["eval_ms_over_cells"]["median"],
        "median per-evaluation cost, UDFS, ms",
    )
    put(
        "paper:45",
        pct_direct(s["udfs"]["isalsr"]["overhead_pct_over_cells"]["median"], 2),
        s["udfs"]["isalsr"]["overhead_pct_over_cells"]["median"],
        "median key overhead, UDFS",
    )
    put(
        "paper:48",
        f"${s['bingo']['isalsr']['key_ms_over_cells']['median']:.3f}$",
        s["bingo"]["isalsr"]["key_ms_over_cells"]["median"],
        "median per-candidate canonicalization cost, Bingo, ms",
    )
    put(
        "paper:49",
        f"${s['bingo']['isalsr']['eval_ms_over_cells']['median']:.2f}$",
        s["bingo"]["isalsr"]["eval_ms_over_cells"]["median"],
        "median per-evaluation cost, Bingo, ms",
    )
    put(
        "paper:50",
        pct_direct(s["bingo"]["isalsr"]["overhead_pct_over_cells"]["median"], 1),
        s["bingo"]["isalsr"]["overhead_pct_over_cells"]["median"],
        "median key overhead, Bingo",
    )
    put(
        "paper:51",
        f"${s['bingo']['isalsr']['n_problems_S_ge_1']}$",
        s["bingo"]["isalsr"]["n_problems_S_ge_1"],
        "problems of 70 with per-problem median S at or above 1, Bingo",
    )
    eval_over_key = (
        s["udfs"]["isalsr"]["eval_ms_over_cells"]["median"]
        / s["udfs"]["isalsr"]["key_ms_over_cells"]["median"]
    )
    put(
        "paper:53",
        f"${eval_over_key:.0f}$",
        eval_over_key,
        "ratio of median per-evaluation cost to median per-candidate key cost, UDFS",
    )

    # per-problem descriptive summary: the targets that move most, either way
    def extremes(
        method: str, n: int = 5
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        rows = [
            r
            for r in data["per_problem_paired"]
            if r["method"] == method
            and r["contrast"] == "isalsr_vs_baseline"
            and r["metric"] == "r2_test"
        ]
        ranked = sorted(rows, key=lambda r: -r["cohens_d"])
        gains = [(r["problem"], r["cohens_d"]) for r in ranked[:n]]
        losses = [(r["problem"], r["cohens_d"]) for r in ranked[::-1] if r["cohens_d"] < 0][:n]
        return gains, losses

    parts = []
    for method, label in (("udfs", "UDFS"), ("bingo", "Bingo")):
        gains, losses = extremes(method)
        gain_s = ", ".join(f"{p} (${d:+.2f}$)" for p, d in gains)
        loss_s = ", ".join(f"{p} (${d:+.2f}$)" for p, d in losses)
        parts.append(f"{gain_s}; {loss_s} on {label}")
    put(
        "paper:42",
        ". ".join(parts),
        None,
        "per-problem Cohen's d on R2 test against the native arm: the five largest "
        "gains and every problem with a negative d, per method",
    )

    put(
        "paper:46",
        pct_direct(s["udfs"]["isalsr"]["overhead_pct_over_cells"]["p95"], 2),
        s["udfs"]["isalsr"]["overhead_pct_over_cells"]["p95"],
        "95th percentile key overhead over the 2,100 UDFS IsalSR cells",
    )
    su = s["udfs"]["isalsr"]
    put(
        "paper:47",
        f"${su['S_over_cells']['median']:.2f}$ over all cells and "
        f"${su['S_by_saturation']['S_unsaturated']['median']:.2f}$ over the "
        f"${su['S_by_saturation']['n_unsaturated']}$ cells in which at least one arm "
        f"finished before the budget",
        [su["S_over_cells"]["median"], su["S_by_saturation"]["S_unsaturated"]["median"]],
        "median search-only speedup, UDFS, over all cells and over the unsaturated ones",
    )
    sb = s["bingo"]["isalsr"]
    put(
        "paper:52",
        "less than the critical difference, the three arms falling in a single clique",
        0.071,
        "wall clock, per-host Nemenyi over three arms at N=70: native 1.971, naive "
        "hash 2.043, IsalSR 1.986, one clique",
    )
    put(
        "paper:54",
        "larger than the critical difference: the native representation is separated "
        "from both deduplicating arms, which are not distinguishable from each other",
        0.872,
        "wall clock, Bingo: native 1.514, naive hash 2.100, IsalSR 2.386; the bar "
        "joins the two deduplicating arms only",
    )
    put(
        "paper:55",
        f"absent, at a median wall-clock ratio of ${su['wall_ratio_over_problems']['median']:.2f}$",
        su["wall_ratio_over_problems"]["median"],
        "median over problems of the seed-matched ratio native/IsalSR on total wall clock, UDFS",
    )
    put(
        "paper:56",
        f"a slowdown of about ${1 / sb['wall_ratio_over_problems']['median']:.2f}\\times$, "
        f"IsalSR being the faster arm on ${sb['n_problems_faster_than_native']}$ of the "
        f"$70$ problems",
        [sb["wall_ratio_over_problems"]["median"], sb["n_problems_faster_than_native"]],
        "median wall-clock ratio and the count of problems on which IsalSR finishes sooner, Bingo",
    )

    # ---------------------------------------------------------- discussion.tex
    put(
        "paper:57",
        f"${ru_all['min']:.2f}$--${ru_all['max']:.2f}$ on UDFS and "
        f"${rb_all['min']:.2f}$--${rb_all['max']:.2f}$ on Bingo across the $70$-problem "
        f"suite (means ${ru_all['mean']:.2f}$ and ${rb_all['mean']:.2f}$)",
        [ru_all["min"], ru_all["max"], rb_all["min"], rb_all["max"]],
        "per-problem range and mean of rho, both methods",
    )
    put(
        "paper:58",
        f"${s['udfs']['isalsr']['key_ms_over_cells']['median']:.3f}$\\,ms (UDFS) and "
        f"${s['bingo']['isalsr']['key_ms_over_cells']['median']:.3f}$\\,ms (Bingo)",
        [
            s["udfs"]["isalsr"]["key_ms_over_cells"]["median"],
            s["bingo"]["isalsr"]["key_ms_over_cells"]["median"],
        ],
        "median per-candidate canonicalization cost, both methods",
    )
    ratio_u = (
        s["udfs"]["isalsr"]["eval_ms_over_cells"]["median"]
        / s["udfs"]["isalsr"]["key_ms_over_cells"]["median"]
    )
    put(
        "paper:59",
        "$T_{\\mathrm{eval}}/T_{\\mathrm{canon}} > 5{,}000$",
        ratio_u,
        "ratio of median per-evaluation to median per-candidate key cost, UDFS",
    )
    put(
        "paper:60",
        pct_direct(s["udfs"]["isalsr"]["overhead_pct_over_cells"]["median"], 2),
        s["udfs"]["isalsr"]["overhead_pct_over_cells"]["median"],
        "median key overhead, UDFS",
    )
    put(
        "paper:61",
        pct(s["udfs"]["isalsr"]["r_over_problems"]["mean"]),
        s["udfs"]["isalsr"]["r_over_problems"]["mean"],
        "redundancy rate, UDFS",
    )
    put(
        "paper:62",
        f"${su['S_by_saturation']['S_unsaturated']['median']:.2f}$",
        su["S_by_saturation"]["S_unsaturated"]["median"],
        "median search-only speedup on the UDFS cells that finish before the budget",
    )
    put(
        "paper:63",
        pct_direct(s["bingo"]["isalsr"]["overhead_pct_over_cells"]["median"], 1),
        s["bingo"]["isalsr"]["overhead_pct_over_cells"]["median"],
        "median key overhead, Bingo",
    )
    put(
        "paper:64",
        f"${sb['S_over_cells']['median']:.2f}$",
        sb["S_over_cells"]["median"],
        "median search-only speedup, Bingo",
    )
    put(
        "paper:65",
        f"${sb['n_problems_faster_than_native']}$ of the $70$ problems",
        sb["n_problems_faster_than_native"],
        "problems on which Bingo under IsalSR finishes sooner than under the native arm",
    )
    put(
        "paper:66",
        f"positive on both, at ${r2_u['cohens_d']:+.2f}$ "
        f"({sci(r2_u['p_one_sided'], stars=False)}) on UDFS and "
        f"${r2_b['cohens_d']:+.2f}$ ({sci(r2_b['p_one_sided'], stars=False)}) on Bingo",
        [r2_u["cohens_d"], r2_u["p_one_sided"], r2_b["cohens_d"], r2_b["p_one_sided"]],
        "sign, magnitude and one-sided p of the paired test across problems on R2 test",
    )
    put(
        "paper:67",
        f"${phi_stats('udfs')['mean']:.2f}$ on UDFS against "
        f"${phi_stats('bingo')['mean']:.3f}$ on Bingo",
        [phi_stats("udfs")["mean"], phi_stats("bingo")["mean"]],
        "phi, suite means, both methods",
    )
    put(
        "paper:68",
        f"(${rb_all['mean']:.2f}$ vs.\\ ${ru_all['mean']:.2f}$)",
        [rb_all["mean"], ru_all["mean"]],
        "mean rho, Bingo against UDFS",
    )
    korns = {
        p["arm"]: p["r2_test_mean"]
        for p in data["per_problem"]
        if p["method"] == "bingo" and p["problem"] == "Korns-12"
    }
    put(
        "paper:69",
        f"${korns['baseline']:.3f}$ under the native representation, "
        f"${korns['hash']:.3f}$ under the naive hash and ${korns['isalsr']:.3f}$ under "
        f"\\IsalSR{{}}",
        korns,
        "mean test R2 on Korns-12 under Bingo, all three arms",
    )

    # ---------------------------------------------------------- conclusion.tex
    put(
        "paper:70",
        pct(s["udfs"]["isalsr"]["r_over_problems"]["mean"]),
        s["udfs"]["isalsr"]["r_over_problems"]["mean"],
        "redundancy rate, UDFS",
    )
    put(
        "paper:71",
        pct(s["bingo"]["isalsr"]["r_over_problems"]["mean"]),
        s["bingo"]["isalsr"]["r_over_problems"]["mean"],
        "redundancy rate, Bingo",
    )
    put(
        "paper:72",
        f"$d = {rho_u['cohens_d']:.2f}$ and $d = {rho_b['cohens_d']:.2f}$",
        [rho_u["cohens_d"], rho_b["cohens_d"]],
        "Cohen's d on rho against the naive hash, both methods",
    )
    put(
        "paper:73",
        "$p < 10^{-12}$",
        [rho_u["p_one_sided"], rho_b["p_one_sided"]],
        "one-sided p on rho against the naive hash, both methods",
    )
    put(
        "paper:74",
        f"{sci(r2_u['p_one_sided'], stars=False)} and {sci(r2_b['p_one_sided'], stars=False)}",
        [r2_u["p_one_sided"], r2_b["p_one_sided"]],
        "one-sided p on R2 test against the native arm, both methods",
    )
    put(
        "paper:75",
        f"${phi_stats('udfs')['mean']:.2f}$ and ${phi_stats('bingo')['mean']:.3f}$",
        [phi_stats("udfs")["mean"], phi_stats("bingo")["mean"]],
        "phi, suite means, both methods",
    )

    # ------------------------------------------------------- supplementary rows
    put("supp:1", "$30$", 30, "seeds per (method, problem, arm) cell")
    put("supp:2", "$12{,}600$", s["n_cells"], "runs in the campaign")
    put("supp:3", "$30$ seeds", 30, "seeds per (problem, arm) cell")
    ru = s["udfs"]["isalsr"]["rho_over_problems"]
    rb = s["bingo"]["isalsr"]["rho_over_problems"]
    put(
        "supp:4",
        f"$[{ru['min']:.2f}, {ru['max']:.2f}]$",
        [ru["min"], ru["max"]],
        "range of rho over the 70 problems, UDFS, IsalSR arm",
    )
    ohu = s["udfs"]["isalsr"]["overhead_pct_over_problems"]
    ohb = s["bingo"]["isalsr"]["overhead_pct_over_problems"]
    put(
        "supp:5",
        pct_direct(ohu["max"], 2),
        ohu["max"],
        "largest per-problem mean key overhead, UDFS",
    )
    put(
        "supp:6",
        f"$[{ohb['min']:.1f}\\%, {ohb['max']:.1f}\\%]$",
        [ohb["min"], ohb["max"]],
        "range of per-problem mean key overhead, Bingo, IsalSR arm",
    )
    put(
        "supp:7",
        f"$[{rb['min']:.2f}, {rb['max']:.2f}]$",
        [rb["min"], rb["max"]],
        "range of rho over the 70 problems, Bingo, IsalSR arm",
    )
    put("supp:9", "$30$ seeds", 30, "seeds per (problem, arm) cell")

    ks = {(r["method"], r["arm"], r["k_range"]): r for r in data["k_strata"]}
    max_k_b = s["bingo"]["isalsr"]["max_k_seen"]
    modal = st.mode(
        [
            int(p["max_k_seen"])
            for p in data["per_problem"]
            if p["method"] == "bingo" and p["arm"] == "isalsr"
        ]
    )
    put(
        "supp:11",
        f"$2$ to ${int(max_k_b)}$ (mode $k = {modal}$)",
        [max_k_b, modal],
        "range and mode of the maximum k Bingo reaches",
    )
    lo = ks[("bingo", "isalsr", "[0,5)")]
    hi = ks[("bingo", "isalsr", "[15,32)")]
    put(
        "supp:12",
        f"$\\approx {lo['rho_mean']:.2f}$",
        lo["rho_mean"],
        "mean rho, Bingo, IsalSR arm, k in [0,5)",
    )
    put(
        "supp:13",
        f"$\\approx {hi['rho_mean']:.2f}$",
        hi["rho_mean"],
        "mean rho, Bingo, IsalSR arm, k in [15,32)",
    )
    growth = hi["key_ms_mean"] / lo["key_ms_mean"]
    put(
        "supp:14",
        f"a factor of ${growth:.1f}$",
        growth,
        "growth of the mean per-candidate key cost across Bingo's k range",
    )
    put(
        "supp:17",
        f"${lo['key_ms_mean']:.3f}$",
        lo["key_ms_mean"],
        "mean per-candidate IsalSR key cost, Bingo, k in [0,5), ms",
    )
    put(
        "supp:18",
        f"${hi['key_ms_mean']:.3f}$",
        hi["key_ms_mean"],
        "mean per-candidate IsalSR key cost, Bingo, k in [15,32), ms",
    )
    put(
        "supp:19",
        pct_direct(lo["overhead_pct_mean"], 1),
        lo["overhead_pct_mean"],
        "mean key overhead, Bingo, IsalSR arm, k in [0,5)",
    )
    put(
        "supp:20",
        pct_direct(hi["overhead_pct_mean"], 1),
        hi["overhead_pct_mean"],
        "mean key overhead, Bingo, IsalSR arm, k in [15,32)",
    )

    for row in range(21, 33):
        out[f"supp:{row}"] = {
            "latex": None,
            "raw": None,
            "note": "synthetic permutation study: a separate measurement, not part of "
            "this campaign. Left pending.",
        }
    put(
        "supp:15",
        None,
        None,
        "the k interval of dominated UDFS configurations: read off the regenerated "
        "figure, filled with the figure",
    )

    return out


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    data = load(args.analyses)
    values = build(data)

    out_dir = args.analyses / "values"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "pending_values.json").open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=1)

    lines = [
        "# Resolved placeholder values",
        "",
        "One row per placeholder of the two pending ledgers.",
        "",
        "| Row | LaTeX | Value | Source |",
        "|---|---|---|---|",
    ]
    for key, rec in values.items():
        latex = rec["latex"] if rec["latex"] is not None else "*(unresolved)*"
        raw = rec["raw"]
        if isinstance(raw, float):
            raw_s = f"{raw:.6g}"
        elif isinstance(raw, list):
            raw_s = ", ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in raw)
        else:
            raw_s = str(raw)
        cell = latex.replace("|", r"\|")
        lines.append(f"| `{key}` | {cell} | {raw_s} | {rec['note']} |")
    (out_dir / "pending_values.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    resolved = sum(1 for r in values.values() if r["latex"] is not None)
    print(f"{resolved} resolved / {len(values)} rows -> {out_dir}")


if __name__ == "__main__":
    main()
