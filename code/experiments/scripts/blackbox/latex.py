"""LaTeX tables and value macros of the black-box appendix.

Every file is a bare ``tabular`` (or a list of ``\\newcommand`` definitions)
that the supplementary ``\\input``s inside a float whose caption it owns, the
convention of the main campaign's table generator. Number formatting reuses
that generator's helpers (``fmt``, ``fmt_p``, ``primary_p``), so a value printed
here reads exactly as the same value would in the main campaign's tables.

Files
-----
tab_supp_blackbox_datasets.tex          selection: size, dimension, levels
tab_supp_blackbox_summary.tex           per host and arm: effort, time, quality
tab_supp_blackbox_cpdt.tex              paired test across problems, Table-3 rows
tab_supp_blackbox_c2.tex                governing quantities against the main campaign
tab_supp_blackbox_per_dataset_udfs.tex  per dataset, UDFS
tab_supp_blackbox_per_dataset_bingo.tex per dataset, Bingo
tab_supp_blackbox_values.tex            one macro per number quoted in the prose
"""

from __future__ import annotations

import math
import statistics as st
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.datasets.srbench_blackbox import (
    SRBENCH_BLACKBOX_DATASETS,
    load_published,
    split_indices,
)
from experiments.scripts.blackbox.config import SEEDS, SYNTHETIC_DATASETS
from experiments.scripts.blackbox.cpdt import wilcoxon_floor
from experiments.scripts.review_campaign.config import ARMS, METHODS
from experiments.scripts.review_campaign.tables import (
    CONTRAST_LABEL,
    CPDT_ROWS,
    HOST_LABEL,
    fmt,
    pick,
    primary_p,
    tex_escape,
)

ARM_SHORT = {"baseline": "NA", "hash": "NH", "isalsr": "IS"}

#: Datasets with at most this many rows are reported as small.
SMALL_N = 62


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------


def fmt_sci(x: float, digits: int = 1) -> str:
    """Mantissa-exponent form for math mode, e.g. ``8.0{\\times}10^{9}``."""
    if x == 0 or not math.isfinite(x):
        return fmt(x, digits)
    exponent = math.floor(math.log10(abs(x)))
    mantissa = x / 10**exponent
    return f"{mantissa:.{digits}f}{{\\times}}10^{{{exponent}}}"


def fmt_num(x: float | None, digits: int = 3, *, signed: bool = False) -> str:
    """Fixed point below 1,000 in magnitude, mantissa-exponent form above."""
    if x is None or not math.isfinite(x):
        return "---"
    if abs(x) >= 1000:
        sign = "+" if signed and x > 0 else ""
        return sign + fmt_sci(x, 1)
    return fmt(x, digits, signed=signed)


def fmt_int(n: float) -> str:
    """Integer with a LaTeX thousands separator."""
    return f"{int(round(n)):,}".replace(",", "{,}")


def fmt_p_plain(p: float) -> str:
    """A p-value in the manuscript's number style, without significance marks."""
    if not math.isfinite(p):
        return "---"
    if p >= 0.01:
        return f"{p:.3f}"
    return fmt_sci(p, 1)


def mark_best_worst(
    values: dict[str, float], *, higher_is_better: bool, digits: int
) -> dict[str, str]:
    """Bold the best arm and underline the worst; ties and non-finite stay plain.

    Same rule as the main campaign's per-problem tables, with ``fmt_num`` so
    that an extreme NRMSE prints in mantissa-exponent form.
    """
    finite = {k: v for k, v in values.items() if v is not None and math.isfinite(v)}
    out = {k: fmt_num(v, digits) for k, v in values.items()}
    shown = {k: fmt_num(v, digits) for k, v in finite.items()}
    if len(set(shown.values())) < 2:
        return out
    best = (max if higher_is_better else min)(finite, key=lambda k: finite[k])
    worst = (min if higher_is_better else max)(finite, key=lambda k: finite[k])
    out[best] = rf"\mathbf{{{shown[best]}}}"
    out[worst] = rf"\underline{{{shown[worst]}}}"
    return out


def _table(spec: str, header: list[str], body: list[str], tabcolsep: str = "3pt") -> str:
    lines = [rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}", rf"\begin{{tabular}}{{{spec}}}"]
    lines += [r"\toprule", *header, r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# I.1 Datasets
# ----------------------------------------------------------------------


def dataset_facts() -> list[dict[str, Any]]:
    """Size, dimension, test-fold size and value levels of every dataset.

    Levels are counted on the vendored files: the number of distinct values of
    each input (smallest and largest over the inputs) and of the target.

    Returns:
        One record per dataset, sorted by size.
    """
    out = []
    for name, n_rows, n_features in SRBENCH_BLACKBOX_DATASETS:
        features, target, _ = load_published(name)
        levels = [len(np.unique(features[:, j])) for j in range(features.shape[1])]
        test_sizes = {len(split_indices(n_rows, seed)[1]) for seed in SEEDS}
        out.append(
            {
                "dataset": name,
                "n": n_rows,
                "d": n_features,
                "n_test": max(test_sizes),
                "n_train": n_rows - max(test_sizes),
                "levels_min": min(levels),
                "levels_max": max(levels),
                "target_levels": len(np.unique(target)),
            }
        )
    return sorted(out, key=lambda r: (r["n"], r["dataset"]))


def datasets_table(facts: list[dict[str, Any]]) -> str:
    """Selected datasets with their size, dimension and value levels."""
    body = []
    for r in facts:
        lo, hi = r["levels_min"], r["levels_max"]
        levels = f"${fmt_int(lo)}$" if lo == hi else f"${fmt_int(lo)}$--${fmt_int(hi)}$"
        mark = r"$^{\ast}$" if r["dataset"] in SYNTHETIC_DATASETS else ""
        body.append(
            "    "
            + " & ".join(
                [
                    tex_escape(r["dataset"]) + mark,
                    f"${fmt_int(r['n'])}$",
                    f"${r['d']}$",
                    f"${fmt_int(r['n_test'])}$",
                    levels,
                    f"${fmt_int(r['target_levels'])}$",
                ]
            )
            + r" \\"
        )
    header = [r"Dataset & $n$ & $m$ & $n_{\mathrm{te}}$ & Input levels & Target levels \\"]
    return _table("@{}lrrrrr@{}", header, body)


# ----------------------------------------------------------------------
# I.3 Per-host summary and the paired test across problems
# ----------------------------------------------------------------------


def summary_table(values: dict[str, Any]) -> str:
    """Per host and arm: effort, cost, speed and quality."""
    header = [
        r" & & \multicolumn{2}{c}{Effort} & \multicolumn{5}{c}{Time}"
        r" & \multicolumn{2}{c}{Quality} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-9} \cmidrule(lr){10-11}",
        r"Host & Arm & $\rho$ & $r$ & $T_{\mathrm{key}}$/ms & $T_{\mathrm{eval}}$/ms & OH"
        r" & $S$ & $W$ & $R^2_{\mathrm{te}}$ & NRMSE$_{\mathrm{te}}$ \\",
    ]
    body: list[str] = []
    for i, method in enumerate(METHODS):
        if i:
            body.append(r"\midrule")
        s = values["summary"][method]
        q = values["quality"][method]
        for arm in ARMS:
            a = s[arm]
            native = arm == "baseline"
            rho = a["rho_over_problems"]
            cells = [
                HOST_LABEL[method] if arm == "baseline" else "",
                ARM_SHORT[arm],
                "$1$" if native else f"${fmt(rho['mean'], 3)} \\pm {fmt(rho['std'], 3)}$",
                "$0$" if native else f"${fmt(100 * a['r_over_problems']['mean'], 1)}\\%$",
                "---" if native else f"${fmt(a['key_ms_over_cells']['median'], 3)}$",
                "---" if native else f"${fmt(a['eval_ms_over_cells']['median'], 2)}$",
                "$0$" if native else f"${fmt(a['overhead_pct_over_cells']['median'], 2)}\\%$",
                "---" if native else f"${fmt(a['S_over_cells']['median'], 2)}$",
                "---" if native else f"${fmt(a['wall_ratio_over_problems']['median'], 2)}$",
                f"${fmt(q[arm]['r2_test_mean'], 3)}$",
                f"${fmt(q[arm]['nrmse_test_median'], 3)}$",
            ]
            body.append("    " + " & ".join(cells) + r" \\")
    return _table("@{}llrrrrrrrrr@{}", header, body)


TEST_LABEL = {
    "t_one_sample": "$t$",
    "wilcoxon_signed_rank": "$W$",
    "descriptive_definitional_baseline": "---",
}


def _test_label(row: dict[str, Any]) -> str:
    """``t``, ``W`` (exact) or ``W_n`` (normal approximation), or a dash."""
    test = str(row["test"])
    if test == "wilcoxon_signed_rank" and not int(row["wilcoxon_exact"]):
        return r"$W_{\mathrm{n}}$"
    return TEST_LABEL.get(test, test)


def _must(rows: list[dict[str, Any]], **where: Any) -> dict[str, Any]:
    """The unique row matching ``where``; a missing row is an error, not a blank."""
    row: dict[str, Any] | None = pick(rows, **where)
    if row is None:
        raise ValueError(f"no row for {where}")
    return row


def cpdt_table(values: dict[str, Any]) -> str:
    """The paired test across problems, the rows and contrasts of Table 3."""
    header = [
        r"Metric & vs. & Test & $p_{\mathrm{SW}}$ & $d$ $[95\%$ CI$]$ & $\bar{\delta}$"
        r" & W/T/L & $p$ \\",
    ]
    body: list[str] = []
    for i, method in enumerate(METHODS):
        if i:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{{HOST_LABEL[method]}}}}} \\")
        for contrast, metric, label in CPDT_ROWS:
            row = pick(values["cpdt"], method=method, contrast=contrast, metric=metric)
            if row is None:
                raise ValueError(f"no test record for {method}/{contrast}/{metric}")
            cells = [
                label,
                CONTRAST_LABEL[contrast],
                _test_label(row),
                f"${fmt_p_plain(row['sw_p'])}$",
                f"${fmt(row['cohens_d'], 2, signed=True)}$ "
                f"$[{fmt(row['d_lo'], 2, signed=True)},{fmt(row['d_hi'], 2, signed=True)}]$",
                f"${fmt_num(row['mean_delta'], 4, signed=True)}$",
                f"${row['n_wins']}/{row['n_ties']}/{row['n_losses']}$",
                primary_p(row),
            ]
            body.append("    " + " & ".join(cells) + r" \\")
    return _table("@{}llcllrcl@{}", header, body, tabcolsep="2.5pt")


# ----------------------------------------------------------------------
# I.5 Governing quantities against the main campaign
# ----------------------------------------------------------------------

#: The quantities the main campaign publishes in Tables 2 and 4, at the digits
#: those tables print; the ratio row divides the two medians before rounding.
C2_ROWS: tuple[tuple[str, str, int, str], ...] = (
    (r"$\rho$ (\IsalSR{})", "rho", 3, ""),
    (r"$\rho_{\mathrm{ser}}$ (naive hash)", "rho_ser", 3, ""),
    (r"$\phi$, mean", "phi", 3, ""),
    (r"$T_{\mathrm{canon}}$, median", "t_canon_ms", 3, r"\,ms"),
    (r"$T_{\mathrm{eval}}$, median", "t_eval_ms", 2, r"\,ms"),
    (r"$T_{\mathrm{eval}}/T_{\mathrm{canon}}$", "t_eval_over_t_canon", 0, ""),
    (r"Overhead, median", "oh_pct", 2, "%"),
    (r"$S$, median", "s_median", 2, ""),
)


def _c2_cell(gov: dict[str, float], key: str, digits: int, unit: str) -> str:
    value = gov[key]
    if key == "t_eval_over_t_canon":
        return f"${fmt_int(value)}$"
    if unit == "%":
        return f"${fmt(value, digits)}\\%$"
    return f"${fmt(value, digits)}${unit}"


def c2_table(values: dict[str, Any]) -> str:
    """The governing quantities, black-box track against the main campaign."""
    bbx = values["governing"]
    main = values["c2"]["governing"]
    header = [
        r" & \multicolumn{2}{c}{UDFS} & \multicolumn{2}{c}{Bingo} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Quantity & BB & Main & BB & Main \\",
    ]
    body = []
    for label, key, digits, unit in C2_ROWS:
        cells = [label]
        for method in METHODS:
            cells += [
                _c2_cell(bbx[method], key, digits, unit),
                _c2_cell(main[method], key, digits, unit),
            ]
        body.append("    " + " & ".join(cells) + r" \\")
    return _table("@{}lrrrr@{}", header, body)


# ----------------------------------------------------------------------
# I.4 Per dataset
# ----------------------------------------------------------------------


def per_dataset_table(derived: dict[str, Any], method: str) -> str:
    """One row per dataset for one host: quality, effort, wall clock, overhead."""
    rows = [p for p in derived["per_problem"] if p["method"] == method]
    order = [name for name, _, _ in sorted(SRBENCH_BLACKBOX_DATASETS, key=lambda t: (t[1], t[0]))]
    header = [
        r" & \multicolumn{3}{c}{$R^2_{\mathrm{te}}$} & \multicolumn{3}{c}{NRMSE$_{\mathrm{te}}$}"
        r" & & & & \multicolumn{3}{c}{Wall clock (h)} & \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){11-13}",
        r"Dataset & NA & NH & IS & NA & NH & IS & $\rho_{\mathrm{ser}}$ & $\rho$ & $\phi$"
        r" & NA & NH & IS & OH \\",
    ]
    body = []
    for problem in order:
        arms = {a: _must(rows, problem=problem, arm=a) for a in ARMS}
        if any(v is None for v in arms.values()):
            continue
        phi = _must(derived["phi"], method=method, problem=problem)
        r2 = mark_best_worst(
            {a: arms[a]["r2_test_mean"] for a in ARMS}, higher_is_better=True, digits=3
        )
        nrmse = mark_best_worst(
            {a: arms[a]["nrmse_test_mean"] for a in ARMS}, higher_is_better=False, digits=3
        )
        wall = mark_best_worst(
            {a: arms[a]["wall_s_mean"] / 3600.0 for a in ARMS}, higher_is_better=False, digits=2
        )
        cells = [
            tex_escape(problem),
            *[f"${r2[a]}$" for a in ARMS],
            *[f"${nrmse[a]}$" for a in ARMS],
            f"${fmt(phi['rho_ser'], 3)}$",
            f"${fmt(phi['rho'], 3)} \\pm {fmt(arms['isalsr']['rho_std'], 3)}$",
            f"${fmt(phi['phi'], 3)}$",
            *[f"${wall[a]}$" for a in ARMS],
            f"${fmt(arms['isalsr']['overhead_pct_mean'], 2)}\\%$",
        ]
        body.append("    " + " & ".join(cells) + r" \\")
    return _table("@{}lrrrrrrrrrrrrr@{}", header, body, tabcolsep="2.2pt")


# ----------------------------------------------------------------------
# Value macros for the prose
# ----------------------------------------------------------------------


def macro_name(key: str) -> str:
    """``udfs_rho_sd`` -> ``\\bbxUdfsRhoSd``; keys hold letters and underscores only."""
    if not all(part.isalpha() for part in key.split("_")):
        raise ValueError(f"macro key {key!r} must hold letters and underscores only")
    return r"\bbx" + "".join(part[:1].upper() + part[1:] for part in key.split("_"))


def _cpdt_values(values: dict[str, Any], method: str) -> dict[str, str]:
    """d and primary p of the Table-3 rows, keyed for the prose."""
    keys = {
        ("isalsr_vs_baseline", "r2_test"): "rtwote_na",
        ("isalsr_vs_baseline", "r2_train"): "rtwotr_na",
        ("isalsr_vs_baseline", "nrmse_test"): "nrmse_na",
        ("isalsr_vs_hash", "r2_test"): "rtwote_nh",
        ("isalsr_vs_hash", "empirical_reduction_factor"): "rho_nh",
        ("isalsr_vs_hash", "redundancy_rate"): "r_nh",
    }
    out: dict[str, str] = {}
    for (contrast, metric), stem in keys.items():
        row = pick(values["cpdt"], method=method, contrast=contrast, metric=metric)
        if row is None:
            continue
        out[f"{method}_{stem}_d"] = fmt(row["cohens_d"], 2, signed=True)
        out[f"{method}_{stem}_p"] = fmt_p_plain(row["p_primary"])
        out[f"{method}_{stem}_test"] = "t" if row["test"] == "t_one_sample" else "W"
    return out


def _host_values(values: dict[str, Any], method: str, prefix: str) -> dict[str, str]:
    """Governing quantities of one host and campaign, keyed for the prose."""
    g = values["governing"][method] if prefix == "" else values["c2"]["governing"][method]
    p = f"{prefix}{method}"
    out = {
        f"{p}_rho": fmt(g["rho"], 3),
        f"{p}_rho_sd": fmt(g["rho_std"], 3),
        f"{p}_rho_min": fmt(g["rho_min"], 3),
        f"{p}_rho_max": fmt(g["rho_max"], 3),
        f"{p}_red": fmt(100 * g["r"], 1),
        f"{p}_rho_ser": fmt(g["rho_ser"], 3),
        f"{p}_phi": fmt(g["phi"], 3),
        f"{p}_phi_min": fmt(g["phi_min"], 3),
        f"{p}_phi_max": fmt(g["phi_max"], 3),
        f"{p}_tcanon": fmt(g["t_canon_ms"], 3),
        f"{p}_tser": fmt(g["t_ser_ms"], 3),
        f"{p}_teval": fmt(g["t_eval_ms"], 2),
        f"{p}_ratio": fmt_int(g["t_eval_over_t_canon"]),
        f"{p}_oh": fmt(g["oh_pct"], 2),
        f"{p}_oh_hash": fmt(g["oh_hash_pct"], 2),
        f"{p}_s": fmt(g["s_median"], 2),
        f"{p}_wall": fmt(g["wall_ratio"], 2),
        f"{p}_slowdown": fmt(1.0 / g["wall_ratio"], 1),
        f"{p}_faster": fmt_int(g["n_faster"]),
        f"{p}_faster_hash": fmt_int(g["n_faster_hash"]),
        f"{p}_saturated": fmt_int(g["n_saturated"]),
        f"{p}_pairs": fmt_int(g["n_saturated"] + g["n_unsaturated"]),
        f"{p}_rtwo_na": fmt(g["r2_na"], 3),
        f"{p}_rtwo_nh": fmt(g["r2_nh"], 3),
        f"{p}_rtwo_is": fmt(g["r2_is"], 3),
    }
    if math.isfinite(g["s_unsaturated_median"]) and g["n_unsaturated"]:
        out[f"{p}_s_unsat"] = fmt(g["s_unsaturated_median"], 2)
    if prefix:
        # The main campaign is quoted only through the values it publishes.
        return {k: v for k, v in out.items() if k.removeprefix(f"{p}_") in PUBLISHED_KEYS}
    for arm in ARMS:
        hours = values["quality"][method][arm]["wall_h_median"]
        out[f"{p}_hours_{ARM_SHORT[arm].lower()}"] = fmt(hours, 2)
    if method == "bingo":
        # How Bingo runs end is kept out of the manuscript (user decision,
        # 6 Oct); the saturation split of S is a statement about it.
        for key in ("saturated", "pairs", "s_unsat"):
            out.pop(f"{p}_{key}", None)
    return out


#: Main-campaign quantities the prose may quote: those of Tables 2 and 4.
PUBLISHED_KEYS: frozenset[str] = frozenset(
    {"rho", "rho_ser", "phi", "tcanon", "teval", "ratio", "oh", "s"}
)


def _corpus_values(values: dict[str, Any], facts: list[dict[str, Any]]) -> dict[str, str]:
    audit = values["audit"]
    sizes = [r["n"] for r in facts]
    small = [r for r in facts if r["n"] <= SMALL_N]
    n_seeds = len(SEEDS)
    floor = wilcoxon_floor(len(facts), "greater")
    return {
        "n_datasets": fmt_int(len(facts)),
        "n_cells": fmt_int(audit["n_expected"]),
        "n_completed": fmt_int(audit["n_completed"]),
        "n_seeds": fmt_int(n_seeds),
        "n_excluded": fmt_int(sum(len(v) for v in audit["excluded"].values())),
        "n_task_logs": fmt_int(audit["slurm"]["n_task_logs"]),
        "n_gate_pass": fmt_int(audit["slurm"]["gate"].get("PASS", 0)),
        "n_min": fmt_int(min(sizes)),
        "n_max": fmt_int(max(sizes)),
        "n_median": fmt(st.median(sizes), 1) if st.median(sizes) % 1 else fmt_int(st.median(sizes)),
        "d_min": fmt_int(min(r["d"] for r in facts)),
        "d_max": fmt_int(max(r["d"] for r in facts)),
        "n_small": fmt_int(len(small)),
        "small_n": fmt_int(SMALL_N),
        "test_min": fmt_int(min(r["n_test"] for r in small)),
        "test_max": fmt_int(max(r["n_test"] for r in small)),
        "train_max": fmt_int(max(r["n_train"] for r in facts)),
        "n_binary": fmt_int(sum(1 for r in facts if r["target_levels"] == 2)),
        "n_synthetic": fmt_int(sum(1 for r in facts if r["dataset"] in SYNTHETIC_DATASETS)),
        "n_nonsynthetic": fmt_int(sum(1 for r in facts if r["dataset"] not in SYNTHETIC_DATASETS)),
        "floor": fmt_sci(floor, 1),
    }


def _nrmse_extremes(values: dict[str, Any], derived: dict[str, Any]) -> dict[str, str]:
    """The dataset with the largest Bingo NRMSE mean: its means and worst cells."""
    rows = [p for p in derived["per_problem"] if p["method"] == "bingo"]
    worst = max(rows, key=lambda p: p["nrmse_test_mean"])
    out = {"outlier_dataset": tex_escape(worst["problem"])}
    for arm in ARMS:
        short = ARM_SHORT[arm].lower()
        mean = _must(rows, problem=worst["problem"], arm=arm)["nrmse_test_mean"]
        cell = values["nrmse_max_cell"]["bingo"][arm]
        out[f"outlier_mean_{short}"] = fmt_num(mean, 3)
        out[f"outlier_cell_{short}"] = fmt_num(cell["nrmse_test"], 3)
        out[f"outlier_cell_{short}_dataset"] = tex_escape(cell["problem"])
    return out


def value_macros(
    values: dict[str, Any], derived: dict[str, Any], facts: list[dict[str, Any]]
) -> dict[str, str]:
    """Every number the appendix prose quotes, as ``macro -> expansion``."""
    flat: dict[str, str] = {}
    flat.update(_corpus_values(values, facts))
    for method in METHODS:
        flat.update(_host_values(values, method, ""))
        flat.update(_host_values(values, method, "ctwo_"))
        flat.update(_cpdt_values(values, method))
    flat.update(_nrmse_extremes(values, derived))
    return {macro_name(k): v for k, v in flat.items()}


def macros_file(macros: dict[str, str]) -> str:
    """``\\newcommand`` definitions, one per line, in a stable order."""
    lines = [
        "% Generated by experiments.scripts.blackbox.analysis; do not edit by hand.",
        *(rf"\newcommand{{{name}}}{{{value}}}" for name, value in sorted(macros.items())),
    ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def write_tables(values: dict[str, Any], derived: dict[str, Any], out_dir: Path) -> list[Path]:
    """Write every table and the value macros.

    Args:
        values: The payload assembled by ``analysis.run``.
        derived: ``measures.derive_all`` result for the black-box cells.
        out_dir: Directory for the ``.tex`` files.

    Returns:
        The written paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    facts = dataset_facts()
    files = {
        "tab_supp_blackbox_datasets.tex": datasets_table(facts),
        "tab_supp_blackbox_summary.tex": summary_table(values),
        "tab_supp_blackbox_cpdt.tex": cpdt_table(values),
        "tab_supp_blackbox_c2.tex": c2_table(values),
        "tab_supp_blackbox_per_dataset_udfs.tex": per_dataset_table(derived, "udfs"),
        "tab_supp_blackbox_per_dataset_bingo.tex": per_dataset_table(derived, "bingo"),
        "tab_supp_blackbox_values.tex": macros_file(value_macros(values, derived, facts)),
    }
    written = []
    for name, body in files.items():
        path = out_dir / name
        path.write_text(body, encoding="utf-8")
        written.append(path)
    return written
