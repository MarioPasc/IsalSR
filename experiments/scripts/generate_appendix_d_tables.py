"""Generate the Appendix D.1 benchmark documentation for all 70 campaign problems.

The generator emits two artefacts from a single pass over
``benchmarks/datasets/*.py``:

* ``appendix_d_tables.tex`` -- one LaTeX ``table`` environment per tier, meant to
  be ``\\input``-ed as a fragment into the manuscript appendix.
* ``appendix_d_benchmarks.json`` -- the same rows in machine-readable form, plus
  a citation audit block, for a downstream verification tool.

Train/test sizes are **not** taken from the ``generate_data`` defaults. They are
resolved the way the campaign resolves them: the ``benchmarks.<suite>.train_size``
and ``benchmarks.<suite>.test_size`` keys of the campaign YAML configs are fed
through ``experiments.models.orchestrator._generate_benchmark_data``, and the
realised array shapes are recorded. This is the only way to capture the
per-problem ``sampling`` overrides (Pagie-1, Korns-12, the Vladislavleva pair,
Keijzer-6, Vlad-7 and all 14 Strogatz benches) without re-implementing, and
therefore risking drift from, the dispatch logic inside each dataset module.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import sympy as sp
import yaml

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.datasets import (  # noqa: E402
    cherrypicked,
    feynman,
    feynman_remainder,
    hard,
    nguyen,
    roundoff,
    strogatz,
)
from experiments.models.orchestrator import (  # noqa: E402
    _generate_benchmark_data,
)

log = logging.getLogger(__name__)

DEFAULT_BIB_PATH: Final[str] = (
    "/media/mpascual/Sandisk2TB/research/ISAL/completed/isalsr/article/journal/"
    "69c1637a28a81fea2badda9a/article/paper/references.bib"
)

#: Fixed seed used only to realise array shapes. Sample *counts* are
#: seed-independent for every sampling protocol in use; the seed is recorded in
#: the JSON provenance block so the probe is reproducible.
SIZE_PROBE_SEED: Final[int] = 42

#: The seven top-level benchmark lists, in emission order. ``roundoff`` also
#: defines ``_FEYNMAN_ROUNDOFF`` and ``_GP_ROUNDOFF``, which are sub-lists of
#: ``ROUNDOFF_BENCHMARKS``; iterating every module-level list double-counts.
SUITES: Final[dict[str, list[dict[str, Any]]]] = {
    "nguyen": nguyen.NGUYEN_BENCHMARKS,
    "feynman": feynman.FEYNMAN_BENCHMARKS,
    "hard": hard.HARD_BENCHMARKS,
    "cherrypicked": cherrypicked.CHERRYPICKED_BENCHMARKS,
    "roundoff": roundoff.ROUNDOFF_BENCHMARKS,
    "feynman_remainder": feynman_remainder.FEYNMAN_REMAINDER_BENCHMARKS,
    "strogatz": strogatz.STROGATZ_BENCHMARKS,
}

TIER_TITLES: Final[dict[str, str]] = {
    "nguyen": "Nguyen tier",
    "feynman": "Feynman tier",
    "hard": "Hard tier",
    "cherrypicked": "Cherry-picked tier",
    "roundoff": "Round-off tier",
    "feynman_remainder": "Feynman-remainder tier",
    "strogatz": "ODE-Strogatz tier",
}

#: Sampling protocol for suites whose bench dicts carry no ``sampling`` key.
#: Nguyen and Feynman both sample i.i.d. uniformly over the box; see
#: ``nguyen.generate_data`` and ``feynman.generate_data``.
IMPLICIT_SAMPLING: Final[dict[str, str]] = {
    "nguyen": "uniform",
    "feynman": "uniform",
}

SAMPLING_LABELS: Final[dict[str, str]] = {
    "uniform": "i.i.d. uniform",
    "grid_2d_skip_zero": "2D grid, zero skipped",
    "grid_1d_train_uniform_test_grid": "1D uniform train / grid test",
    "integer_grid": "integer grid (extrapolation)",
    "published_fixed": "published fixed rows (split by seed)",
}

#: Explicit, auditable citation assignment. One entry per problem, grouped by
#: originating family. Nothing here is inferred at run time: a problem that is
#: absent raises rather than silently receiving a neighbour's key.
CITATION_MAP: Final[dict[str, str]] = {
    # --- Nguyen suite (Uy et al., 2011) ---------------------------------
    **{f"Nguyen-{i}": "uy2011" for i in range(1, 13)},
    # --- AI Feynman equations (Udrescu & Tegmark, 2020) -----------------
    "I.6.20a": "udrescu2020",
    "I.12.1": "udrescu2020",
    "I.14.3": "udrescu2020",
    "I.25.13": "udrescu2020",
    "I.34.27": "udrescu2020",
    "I.39.10": "udrescu2020",
    "I.12.4": "udrescu2020",
    "II.3.24": "udrescu2020",
    "I.10.7": "udrescu2020",
    "I.48.20": "udrescu2020",
    "I.15.10": "udrescu2020",
    "I.30.3": "udrescu2020",
    "I.37.4": "udrescu2020",
    "II.11.27": "udrescu2020",
    "III.17.37": "udrescu2020",
    "I.29.16": "udrescu2020",
    "I.50.26": "udrescu2020",
    "I.16.6": "udrescu2020",
    "II.11.28": "udrescu2020",
    "III.14.14": "udrescu2020",
    "III.10.19": "udrescu2020",
    "II.11.3": "udrescu2020",
    "I.13.12": "udrescu2020",
    "I.44.4": "udrescu2020",
    "I.12.2": "udrescu2020",
    "II.34.29a": "udrescu2020",
    "II.34.29b": "udrescu2020",
    "III.19.51": "udrescu2020",
    "III.4.32": "udrescu2020",
    "test_4": "udrescu2020",
    # --- Pagie & Hogeweg (1997) ------------------------------------------
    "Pagie-1": "pagie1997",
    "Pagie-2": "pagie1997",
    # --- Korns (2011) -----------------------------------------------------
    "Korns-12": "korns2011",
    # --- Vladislavleva et al. (2009) --------------------------------------
    "Vladislavleva-2": "vladislavleva2009",
    "Vladislavleva-4": "vladislavleva2009",
    "Vlad-7": "vladislavleva2009",
    # --- Keijzer (2003) ---------------------------------------------------
    "Keijzer-6": "keijzer2003",
    "Keijzer-11": "keijzer2003",
    # --- DSO benchmark suite (Petersen et al., 2021): R-rationals and
    #     the Livermore family. See ATTRIBUTION_NOTES for the R-rationals.
    "R1": "petersen2021",
    "R2": "petersen2021",
    "R3": "petersen2021",
    "Liv-4": "petersen2021",
    "Liv-14": "petersen2021",
    "Liv-19": "petersen2021",
    # --- ODE-Strogatz (Strogatz, 1994). NOT PRESENT IN references.bib. ----
    **{b["name"]: "strogatz1994" for b in strogatz.STROGATZ_BENCHMARKS},
}

#: Citation keys the generator asserts are missing from ``references.bib`` and
#: must be added by hand. Keeping this explicit prevents the audit from
#: silently degrading into a warning.
EXPECTED_MISSING_CITATIONS: Final[frozenset[str]] = frozenset({"strogatz1994"})

#: Suggested BibTeX stubs for the keys in ``EXPECTED_MISSING_CITATIONS``.
MISSING_CITATION_STUBS: Final[dict[str, str]] = {
    "strogatz1994": (
        "@book{strogatz1994,\n"
        "  author    = {Strogatz, Steven H.},\n"
        "  title     = {Nonlinear Dynamics and Chaos: With Applications to "
        "Physics, Biology, Chemistry, and Engineering},\n"
        "  publisher = {Addison-Wesley},\n"
        "  year      = {1994}\n"
        "}"
    ),
}

#: Attribution caveats surfaced in the JSON audit rather than resolved silently.
ATTRIBUTION_NOTES: Final[dict[str, str]] = {
    "R1": (
        "Rational benchmark distributed with the DSO suite (petersen2021); "
        "primary origin is contested in the literature."
    ),
    "R2": (
        "Rational benchmark distributed with the DSO suite (petersen2021); "
        "primary origin is contested in the literature."
    ),
    "R3": (
        "Rational benchmark distributed with the DSO suite (petersen2021); "
        "primary origin is contested in the literature."
    ),
    **{
        b["name"]: (
            "Equation from Strogatz (1994); the numerical rows are the "
            "ODE-Strogatz tables redistributed via PMLB/SRBench (lacava2021)."
        )
        for b in strogatz.STROGATZ_BENCHMARKS
    },
}

#: Campaign operator sets, uniform across all seven suites.
OPERATOR_SETS: Final[dict[str, list[str]]] = {
    "bingo": ["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "pow"],
    "udfs": ["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "neg", "inv"],
}

#: Nguyen bench dicts name their variables ``x``/``y``; every other suite uses
#: ``x_0, x_1, ...``. Renaming makes the rendered expressions consistent with
#: the positional ordering of ``variable_ranges``.
_NGUYEN_VAR_RENAME: Final[dict[str, str]] = {"x": "x_0", "y": "x_1"}


class AppendixGenerationError(Exception):
    """Raised when the appendix cannot be generated from the definitions."""


@dataclass(frozen=True)
class BenchmarkRow:
    """One documented benchmark problem.

    Attributes:
        problem_id: Benchmark name as used by the orchestrator.
        tier: Suite key the problem belongs to.
        expression: Verbatim ``expression`` string from the definition module.
        expression_latex: LaTeX rendering of the ground-truth expression.
        n_variables: Number of input variables.
        variable_ranges: Per-variable sampling interval, positionally ordered.
        n_train: Realised training-set size under the campaign configuration.
        n_test: Realised test-set size under the campaign configuration.
        sampling_protocol: Sampling-protocol key.
        sampling_protocol_label: Human-readable protocol description.
        citation_key: BibTeX key of the originating publication.
        citation_resolved: Whether the key exists in ``references.bib``.
        attribution_note: Optional caveat about the assignment.
    """

    problem_id: str
    tier: str
    expression: str
    expression_latex: str
    n_variables: int
    variable_ranges: list[list[float]]
    n_train: int
    n_test: int
    sampling_protocol: str
    sampling_protocol_label: str
    citation_key: str
    citation_resolved: bool
    attribution_note: str = field(default="")

    def required_fields(self) -> dict[str, Any]:
        """Return the fields that the acceptance check requires to be non-empty.

        Returns:
            Mapping from field name to value for the seven mandatory fields.
        """
        return {
            "problem_id": self.problem_id,
            "expression": self.expression,
            "n_variables": self.n_variables,
            "variable_ranges": self.variable_ranges,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "sampling_protocol": self.sampling_protocol,
            "citation_key": self.citation_key,
            "tier": self.tier,
        }


def load_campaign_sizes(config_dir: Path) -> dict[str, tuple[int, int]]:
    """Read ``train_size``/``test_size`` from the campaign YAML configs.

    Both the Bingo and the UDFS config of a suite are read and required to
    agree, because a disagreement would mean the two hosts saw different data
    and the paired design would be void.

    Args:
        config_dir: Directory holding ``{bingo,udfs}_<suite>.yaml``.

    Returns:
        Mapping from suite key to ``(train_size, test_size)``.

    Raises:
        AppendixGenerationError: If a config is missing, lacks the sizes, or the
            two hosts disagree.
    """
    sizes: dict[str, tuple[int, int]] = {}
    for suite in SUITES:
        per_host: dict[str, tuple[int, int]] = {}
        for host in ("bingo", "udfs"):
            path = config_dir / f"{host}_{suite}.yaml"
            if not path.exists():
                raise AppendixGenerationError(f"Missing campaign config: {path}")
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            bench_cfg = cfg.get("benchmarks", {}).get(suite, {})
            train = bench_cfg.get("train_size")
            test = bench_cfg.get("test_size")
            if train is None or test is None:
                raise AppendixGenerationError(
                    f"{path} declares no train_size/test_size for suite '{suite}'"
                )
            per_host[host] = (int(train), int(test))
        if per_host["bingo"] != per_host["udfs"]:
            raise AppendixGenerationError(
                f"Suite '{suite}': bingo sizes {per_host['bingo']} disagree with "
                f"udfs sizes {per_host['udfs']}"
            )
        sizes[suite] = per_host["bingo"]
    return sizes


def resolve_variable_ranges(bench: dict[str, Any]) -> list[list[float]]:
    """Return the per-variable sampling interval of a benchmark.

    Nguyen benches carry a single scalar ``x_range`` shared by every variable;
    it is replicated ``num_variables`` times. All other suites carry an explicit
    ``var_ranges`` list.

    Args:
        bench: A benchmark specification dict.

    Returns:
        List of ``[lo, hi]`` pairs, one per variable, positionally ordered.

    Raises:
        AppendixGenerationError: If no range information is present, or the
            recovered list does not match ``num_variables``.
    """
    n_vars = int(bench["num_variables"])
    if "var_ranges" in bench:
        ranges = [[float(lo), float(hi)] for lo, hi in bench["var_ranges"]]
    elif "x_range" in bench:
        lo, hi = bench["x_range"]
        ranges = [[float(lo), float(hi)] for _ in range(n_vars)]
    else:
        raise AppendixGenerationError(
            f"Benchmark {bench['name']!r} carries neither var_ranges nor x_range"
        )
    if len(ranges) != n_vars:
        raise AppendixGenerationError(
            f"Benchmark {bench['name']!r}: {len(ranges)} ranges for {n_vars} variables"
        )
    return ranges


def resolve_sample_sizes(
    suite: str,
    bench: dict[str, Any],
    train_size: int,
    test_size: int,
    seed: int = SIZE_PROBE_SEED,
) -> tuple[int, int]:
    """Realise the train/test sizes the campaign actually produced.

    The benchmark data is generated through the orchestrator's own dispatch so
    that per-problem ``sampling`` overrides are honoured exactly.

    Args:
        suite: Suite key.
        bench: Benchmark specification dict.
        train_size: ``train_size`` from the campaign YAML.
        test_size: ``test_size`` from the campaign YAML.
        seed: Seed passed to the data generator.

    Returns:
        ``(n_train, n_test)`` as realised array row counts.
    """
    x_train, _, x_test, _ = _generate_benchmark_data(suite, bench, train_size, test_size, seed)
    return int(x_train.shape[0]), int(x_test.shape[0])


def expression_to_latex(bench: dict[str, Any]) -> str:
    """Render a benchmark's ground-truth expression as LaTeX math.

    The ``sympy_expression`` key is preferred where present (58 of 70 problems)
    because it is the object the solution-recovery check compares against, and
    because it is free of the annotations that contaminate some ``expression``
    strings. Nguyen benches carry no ``sympy_expression``; their string is
    parsed after ``^`` is rewritten to ``**`` and ``x``/``y`` to ``x_0``/``x_1``.

    Args:
        bench: A benchmark specification dict.

    Returns:
        LaTeX math body, without surrounding ``$`` delimiters.

    Raises:
        AppendixGenerationError: If the expression cannot be rendered.
    """
    sym_expr = bench.get("sympy_expression")
    if sym_expr is None:
        text = str(bench["expression"]).replace("^", "**")
        try:
            parsed = sp.sympify(text)
        except (sp.SympifyError, SyntaxError, TypeError) as exc:
            raise AppendixGenerationError(
                f"Cannot parse expression for {bench['name']!r}: {exc}"
            ) from exc
        subs = {
            sp.Symbol(old): sp.Symbol(new)
            for old, new in _NGUYEN_VAR_RENAME.items()
            if sp.Symbol(old) in parsed.free_symbols
        }
        sym_expr = parsed.subs(subs) if subs else parsed
    try:
        return str(sp.latex(sym_expr))
    except (TypeError, ValueError) as exc:
        raise AppendixGenerationError(f"Cannot render LaTeX for {bench['name']!r}: {exc}") from exc


def parse_bib_keys(bib_path: Path) -> set[str]:
    """Extract the entry keys of a BibTeX file.

    Args:
        bib_path: Path to ``references.bib``.

    Returns:
        Set of citation keys.

    Raises:
        AppendixGenerationError: If the file cannot be read.
    """
    try:
        text = bib_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise AppendixGenerationError(f"Cannot read bib file {bib_path}: {exc}") from exc
    return set(re.findall(r"^@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,", text, flags=re.MULTILINE))


def build_rows(
    config_dir: Path,
    bib_keys: set[str],
    seed: int = SIZE_PROBE_SEED,
) -> list[BenchmarkRow]:
    """Assemble one documented row per campaign problem.

    Args:
        config_dir: Directory holding the campaign YAML configs.
        bib_keys: Citation keys available in ``references.bib``.
        seed: Seed used to realise the data-array shapes.

    Returns:
        Rows in deterministic order: suite order of ``SUITES``, then the
        declaration order of each suite's list.

    Raises:
        AppendixGenerationError: If a problem has no citation assignment or a
            mandatory field cannot be resolved.
    """
    sizes = load_campaign_sizes(config_dir)
    rows: list[BenchmarkRow] = []
    for suite, benches in SUITES.items():
        train_size, test_size = sizes[suite]
        for bench in benches:
            name = str(bench["name"])
            if name not in CITATION_MAP:
                raise AppendixGenerationError(
                    f"No citation assignment for problem {name!r}; add it to "
                    "CITATION_MAP rather than letting it default"
                )
            protocol = bench.get("sampling", {}).get("type") or IMPLICIT_SAMPLING.get(suite, "")
            if protocol not in SAMPLING_LABELS:
                raise AppendixGenerationError(
                    f"Unknown sampling protocol {protocol!r} for {name!r}"
                )
            n_train, n_test = resolve_sample_sizes(suite, bench, train_size, test_size, seed)
            citation = CITATION_MAP[name]
            rows.append(
                BenchmarkRow(
                    problem_id=name,
                    tier=suite,
                    expression=str(bench["expression"]),
                    expression_latex=expression_to_latex(bench),
                    n_variables=int(bench["num_variables"]),
                    variable_ranges=resolve_variable_ranges(bench),
                    n_train=n_train,
                    n_test=n_test,
                    sampling_protocol=protocol,
                    sampling_protocol_label=SAMPLING_LABELS[protocol],
                    citation_key=citation,
                    citation_resolved=citation in bib_keys,
                    attribution_note=ATTRIBUTION_NOTES.get(name, ""),
                )
            )
    return rows


def validate_rows(rows: list[BenchmarkRow]) -> None:
    """Fail hard if any mandatory field is empty or a problem is duplicated.

    Args:
        rows: The assembled rows.

    Raises:
        AppendixGenerationError: On an empty mandatory field or a duplicate id.
    """
    seen: set[str] = set()
    for row in rows:
        if row.problem_id in seen:
            raise AppendixGenerationError(f"Duplicate problem id: {row.problem_id}")
        seen.add(row.problem_id)
        for fname, value in row.required_fields().items():
            empty = (
                value is None
                or (isinstance(value, str) and not value.strip())
                or (isinstance(value, list) and not value)
                or (isinstance(value, int) and not isinstance(value, bool) and value <= 0)
            )
            if empty:
                raise AppendixGenerationError(
                    f"Problem {row.problem_id!r}: mandatory field {fname!r} is empty "
                    f"(value={value!r})"
                )


def audit_citations(rows: list[BenchmarkRow], bib_keys: set[str]) -> dict[str, Any]:
    """Summarise which citation keys resolve against ``references.bib``.

    Args:
        rows: The assembled rows.
        bib_keys: Citation keys available in ``references.bib``.

    Returns:
        Audit block with the used keys, the unresolved keys, the problems that
        depend on them, and suggested BibTeX stubs.
    """
    used = sorted({row.citation_key for row in rows})
    missing = sorted(k for k in used if k not in bib_keys)
    return {
        "bib_entries_total": len(bib_keys),
        "citation_keys_used": used,
        "citation_keys_resolved": sorted(k for k in used if k in bib_keys),
        "citation_keys_missing": missing,
        "problems_awaiting_bib_entry": sorted(
            row.problem_id for row in rows if not row.citation_resolved
        ),
        "suggested_bibtex": {
            k: MISSING_CITATION_STUBS[k] for k in missing if k in MISSING_CITATION_STUBS
        },
        "attribution_notes": {
            row.problem_id: row.attribution_note for row in rows if row.attribution_note
        },
    }


def _fmt_number(value: float) -> str:
    """Format a range endpoint compactly and deterministically.

    Args:
        value: The endpoint.

    Returns:
        Integer-looking string for integral values, else three decimals with
        trailing zeros stripped.
    """
    if value == int(value):
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def format_variable_ranges(ranges: list[list[float]]) -> str:
    """Render the variable ranges as compact LaTeX math.

    Args:
        ranges: Per-variable ``[lo, hi]`` pairs.

    Returns:
        LaTeX math body without ``$`` delimiters.
    """
    unique = {(lo, hi) for lo, hi in ranges}
    if len(unique) == 1:
        lo, hi = ranges[0]
        body = f"[{_fmt_number(lo)}, {_fmt_number(hi)}]"
        return body if len(ranges) == 1 else f"{body}^{{{len(ranges)}}}"
    return ",\\, ".join(f"[{_fmt_number(lo)}, {_fmt_number(hi)}]" for lo, hi in ranges)


def escape_latex_text(text: str) -> str:
    """Escape a plain string for LaTeX text mode.

    Args:
        text: Raw text.

    Returns:
        Text with the special characters escaped.
    """
    mapping = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "^": "\\^{}",
        "~": "\\~{}",
    }
    return "".join(mapping.get(char, char) for char in text)


def render_tier_table(tier: str, rows: list[BenchmarkRow]) -> str:
    """Render one tier as a standalone LaTeX ``table`` environment.

    Args:
        tier: Suite key.
        rows: Rows belonging to that suite, in emission order.

    Returns:
        The LaTeX fragment, newline-terminated.
    """
    lines: list[str] = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
        f"\\caption{{{escape_latex_text(TIER_TITLES[tier])}: ground-truth "
        f"expressions, sampling domains and dataset sizes "
        f"({len(rows)} problems).}}",
        f"\\label{{tab:appendix-d-{tier.replace('_', '-')}}}",
        "\\begin{tabular}{@{}llrlrrll@{}}",
        "\\toprule",
        "Problem & Expression & $m$ & Domain & $n_{\\mathrm{train}}$ & "
        "$n_{\\mathrm{test}}$ & Sampling & Source \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                (
                    escape_latex_text(row.problem_id),
                    f"${row.expression_latex}$",
                    str(row.n_variables),
                    f"${format_variable_ranges(row.variable_ranges)}$",
                    str(row.n_train),
                    str(row.n_test),
                    escape_latex_text(row.sampling_protocol_label),
                    f"\\cite{{{row.citation_key}}}",
                )
            )
            + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def render_latex(rows: list[BenchmarkRow]) -> str:
    """Render the whole Appendix D.1 LaTeX fragment.

    Args:
        rows: All assembled rows, in emission order.

    Returns:
        The complete fragment, ending in a newline.
    """
    header = [
        "% Appendix D.1 -- benchmark documentation.",
        "% GENERATED FILE. Regenerate with:",
        "%   python -m experiments.scripts.generate_appendix_d_tables "
        "--out docs/generated/appendix_d/",
        "% Do not edit by hand.",
        "",
    ]
    ops = [
        "% Campaign operator sets (uniform across all seven tiers):",
        "%   Bingo: " + " ".join(OPERATOR_SETS["bingo"]),
        "%   UDFS:  " + " ".join(OPERATOR_SETS["udfs"]),
        "",
    ]
    body = [render_tier_table(tier, [r for r in rows if r.tier == tier]) for tier in SUITES]
    return "\n".join(header + ops + body)


def validate_latex_fragment(text: str) -> None:
    """Structurally check the emitted LaTeX fragment.

    Verifies that ``\\begin``/``\\end`` pairs nest correctly, that braces and
    ``$`` delimiters balance, and that no ``_``, ``^`` or ``%`` appears
    unescaped in text mode (outside ``$...$`` and outside comment lines).

    Args:
        text: The fragment.

    Raises:
        AppendixGenerationError: On any structural defect.
    """
    stack: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([A-Za-z*]+)\}", text):
        kind, env = match.group(1), match.group(2)
        if kind == "begin":
            stack.append(env)
        else:
            if not stack:
                raise AppendixGenerationError(f"\\end{{{env}}} without \\begin")
            opened = stack.pop()
            if opened != env:
                raise AppendixGenerationError(
                    f"Environment mismatch: \\begin{{{opened}}} closed by \\end{{{env}}}"
                )
    if stack:
        raise AppendixGenerationError(f"Unclosed environments: {stack}")

    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("%"):
            continue
        stripped = re.sub(r"\\[\\{}$&%#_^~]", "", line)
        depth = 0
        for char in stripped:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth < 0:
                    raise AppendixGenerationError(f"Unbalanced '}}' on line {lineno}")
        if depth != 0:
            raise AppendixGenerationError(f"Unbalanced braces on line {lineno}: {line!r}")

        segments = stripped.split("$")
        if len(segments) % 2 == 0:
            raise AppendixGenerationError(f"Unbalanced '$' on line {lineno}: {line!r}")
        for idx, segment in enumerate(segments):
            if idx % 2 == 1:  # math mode
                continue
            # ``&`` is the tabular column separator and is legitimate here.
            bare = re.sub(r"\\[A-Za-z]+", "", segment)
            for char in ("_", "^", "%"):
                if char in bare:
                    raise AppendixGenerationError(
                        f"Unescaped {char!r} in text mode on line {lineno}: {line!r}"
                    )


def build_json_payload(
    rows: list[BenchmarkRow],
    audit: dict[str, Any],
    sizes: dict[str, tuple[int, int]],
    bib_path: Path,
    seed: int,
) -> dict[str, Any]:
    """Assemble the machine-readable payload.

    Args:
        rows: All assembled rows.
        audit: Citation audit block.
        sizes: Per-suite ``(train_size, test_size)`` from the campaign YAMLs.
        bib_path: Path the citation keys were checked against.
        seed: Seed used to realise the array shapes.

    Returns:
        JSON-serialisable payload.
    """
    return {
        "schema_version": "appendix-d.1",
        "provenance": {
            "generator": "experiments/scripts/generate_appendix_d_tables.py",
            "size_source": (
                "experiments/configs/{bingo,udfs}_<suite>.yaml -> "
                "benchmarks.<suite>.{train_size,test_size}, realised through "
                "experiments.models.orchestrator._generate_benchmark_data"
            ),
            "size_probe_seed": seed,
            "bib_path": str(bib_path),
            "campaign_yaml_sizes": {k: list(v) for k, v in sorted(sizes.items())},
            "operator_sets": OPERATOR_SETS,
        },
        "counts": {
            "n_problems": len(rows),
            "per_tier": {tier: sum(r.tier == tier for r in rows) for tier in SUITES},
        },
        "citation_audit": audit,
        "problems": [asdict(row) for row in rows],
    }


def generate(out_dir: Path, config_dir: Path, bib_path: Path, seed: int) -> list[Path]:
    """Generate the Appendix D.1 artefacts.

    Args:
        out_dir: Destination directory; created if absent.
        config_dir: Directory holding the campaign YAML configs.
        bib_path: Path to ``references.bib``.
        seed: Seed used to realise the array shapes.

    Returns:
        The written paths, LaTeX first.

    Raises:
        AppendixGenerationError: If any validation fails.
    """
    bib_keys = parse_bib_keys(bib_path)
    rows = build_rows(config_dir, bib_keys, seed)
    validate_rows(rows)
    audit = audit_citations(rows, bib_keys)
    unexpected = set(audit["citation_keys_missing"]) - EXPECTED_MISSING_CITATIONS
    if unexpected:
        raise AppendixGenerationError(
            f"Citation keys missing from {bib_path} and not declared in "
            f"EXPECTED_MISSING_CITATIONS: {sorted(unexpected)}"
        )
    for key in audit["citation_keys_missing"]:
        log.warning(
            "Citation key %r does not exist in %s; %d problem(s) need it. Suggested entry:\n%s",
            key,
            bib_path,
            sum(1 for r in rows if r.citation_key == key),
            MISSING_CITATION_STUBS.get(key, "<no stub available>"),
        )

    tex = render_latex(rows)
    validate_latex_fragment(tex)

    sizes = load_campaign_sizes(config_dir)
    payload = build_json_payload(rows, audit, sizes, bib_path, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / "appendix_d_tables.tex"
    json_path = out_dir / "appendix_d_benchmarks.json"
    tex_path.write_text(tex, encoding="utf-8", newline="\n")
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    log.info("Wrote %d problems to %s and %s", len(rows), tex_path, json_path)
    return [tex_path, json_path]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate the Appendix D.1 benchmark documentation."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "docs" / "generated" / "appendix_d",
        help="Output directory.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=_REPO_ROOT / "experiments" / "configs",
        help="Directory holding the campaign YAML configs.",
    )
    parser.add_argument(
        "--bib",
        type=Path,
        default=Path(DEFAULT_BIB_PATH),
        help="Path to references.bib (read-only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SIZE_PROBE_SEED,
        help="Seed used to realise the data-array shapes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        Process exit status.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    try:
        generate(args.out, args.config_dir, args.bib, args.seed)
    except AppendixGenerationError:
        log.exception("Appendix D generation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
