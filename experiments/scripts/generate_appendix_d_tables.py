"""Generate the Appendix D.1 benchmark documentation for all 70 campaign problems.

The generator emits two artefacts from a single pass over
``benchmarks/datasets/*.py``:

* ``appendix_d_tables.tex`` -- one LaTeX ``table`` environment per tier, meant to
  be ``\\input``-ed as a fragment into the manuscript appendix.
* ``appendix_d_benchmarks.json`` -- the same rows in machine-readable form, plus
  a citation audit block, for a downstream verification tool.
* ``tab_supp_bench_struct_{feynman,other}.tex`` -- two float-free ``tabular``
  bodies documenting the 28 problems that the shipped Appendix D.1 leaves
  undocumented, split 14/14 into the AI Feynman equations and the six further
  benchmark families. The including document supplies the float, caption and
  label, matching the ``tab_supp_*.tex`` idiom already in the supplementary.

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

import numpy as np
import sympy as sp
import yaml

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.datasets import (  # noqa: E402
    feynman,
    feynman_remainder,
    hard,
    nguyen,
    roundoff,
    strogatz,
    structural,
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
    "cherrypicked": structural.STRUCTURAL_BENCHMARKS,
    "roundoff": roundoff.ROUNDOFF_BENCHMARKS,
    "feynman_remainder": feynman_remainder.FEYNMAN_REMAINDER_BENCHMARKS,
    "strogatz": strogatz.STROGATZ_BENCHMARKS,
}

TIER_TITLES: Final[dict[str, str]] = {
    "nguyen": "Nguyen tier",
    "feynman": "Feynman tier",
    "hard": "Hard tier",
    "cherrypicked": "Structural tier",
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

#: The three tiers whose problems the shipped Appendix D.1 does not document.
#: Their union is the 28-problem extension targeting regimes where structural
#: search dominates. The internal tier keys are an implementation detail and are
#: never rendered into reviewer-facing LaTeX.
SUPPLEMENTARY_TIERS: Final[frozenset[str]] = frozenset({"hard", "cherrypicked", "roundoff"})

#: Citation key that splits the 28 into the AI Feynman half and the half drawn
#: from the six further benchmark families.
FEYNMAN_CITATION_KEY: Final[str] = "udrescu2020"

#: Each of the two supplementary groups must hold exactly this many problems.
SUPP_GROUP_SIZE: Final[int] = 14

#: Number of distinct BibTeX keys behind the non-Feynman group. The group draws
#: on six named benchmark families (Vladislavleva, DSO-Livermore, the Koza
#: rational functions, Pagie, Keijzer, Korns) but only five citation keys,
#: because the Livermore problems and the R-rationals are both distributed with
#: the DSO suite and therefore share ``petersen2021``. Any caption must say
#: "families", not "publications".
SUPP_OTHER_SOURCE_COUNT: Final[int] = 5

#: The AI Feynman half samples i.i.d. uniformly at these sizes for every
#: problem, so the sizes belong in the caption rather than in a column. The
#: generator asserts this rather than assuming it.
SUPP_FEYNMAN_SAMPLING: Final[tuple[int, int, str]] = (1000, 250, "uniform")

#: Column separation applied to both supplementary bodies, matching the
#: ``tab_supp_*.tex`` idiom already used in the manuscript supplementary.
SUPP_TABCOLSEP: Final[str] = "4pt"

SUPP_FEYNMAN_FILENAME: Final[str] = "tab_supp_bench_struct_feynman.tex"
SUPP_OTHER_FILENAME: Final[str] = "tab_supp_bench_struct_other.tex"

#: Substrings that must never appear in the supplementary bodies: they are
#: internal tier vocabulary, and the legacy ``cherrypicked`` key in particular
#: misdescribes how the tier was assembled. Problems enter it by a screening
#: criterion on the target expression, not by any measured solver outcome.
_FORBIDDEN_TIER_WORDS: Final[tuple[str, ...]] = (
    "hard",
    "cherrypicked",
    "cherry-picked",
    "roundoff",
    "round-off",
)

#: Nguyen bench dicts name their variables ``x``/``y``; every other suite uses
#: ``x_0, x_1, ...``. Renaming makes the rendered expressions consistent with
#: the positional ordering of ``variable_ranges``.
_NGUYEN_VAR_RENAME: Final[dict[str, str]] = {"x": "x_0", "y": "x_1"}


#: A decimal literal with more than this many significant figures is a raw float
#: dump rather than a benchmark coefficient, and is refused in emitted LaTeX.
MAX_SIGNIFICANT_FIGURES: Final[int] = 4

#: Seed for the expression-versus-data agreement audit. Shared with the size
#: probe so that both passes describe the same realisation of each dataset.
DATA_AGREEMENT_SEED: Final[int] = SIZE_PROBE_SEED

#: Rows compared per problem. The campaign's own sampling protocol and sizes are
#: used, then the leading ``n`` rows are taken so that the audit stays cheap and
#: deterministic on the four suites that generate thousands of points. The
#: realised count is recorded per problem, because several protocols (the
#: integer grid, the 2D grid, the published ODE-Strogatz rows) return fewer.
DATA_AGREEMENT_MAX_POINTS: Final[int] = 256

#: Relative-error boundary between rounding and a genuine disagreement.
#: Evaluating one expression through two code paths (``target_fn``'s NumPy
#: composition against SymPy's ``lambdify``) reassociates float64 operations, so
#: agreement is only expected to about ``kappa * eps`` with ``eps = 2.2e-16``.
#: ``1e-6`` sits eight orders of magnitude below the smallest genuine
#: discrepancy observed (Keijzer-6, ``4.2e-1``) and two orders above the largest
#: rounding artefact observed (Nguyen-11's ``x^y``, ``1.6e-8``), so no
#: measurement lies near the boundary and the classification is not sensitive to
#: the exact value.
FLOATING_POINT_TOLERANCE: Final[float] = 1e-6

VERDICT_EXACT: Final[str] = "exact"
VERDICT_FLOATING_POINT: Final[str] = "floating_point"
VERDICT_APPROXIMATION: Final[str] = "approximation"


class AppendixGenerationError(Exception):
    """Raised when the appendix cannot be generated from the definitions."""


@dataclass(frozen=True)
class DisplayOverride:
    """A hand-declared replacement for a problem's printed expression.

    Attributes:
        expression_latex: LaTeX body to print instead of the rendered
            ``sympy_expression``.
        note: One-sentence caveat explaining the substitution, emitted as a
            comment beside the table and recorded in the JSON inventory.
    """

    expression_latex: str
    note: str


#: Per-problem display overrides. Every entry exists because the definition
#: module's ``sympy_expression`` is *not* the function the campaign's ``y`` came
#: from, so rendering it verbatim would misdescribe the data. Entries are keyed
#: by problem id, are asserted to be reachable, and are never inferred.
#:
#: * ``Keijzer-6`` -- ``benchmarks/datasets/hard.py`` sets ``target_fn`` to the
#:   exact harmonic number ``H(x_0) = sum_{i=1}^{x_0} 1/i`` while its
#:   ``sympy_expression`` is the Euler-Mascheroni asymptotic form
#:   ``log(x_0) + gamma``. The two disagree by 42% at the smallest training
#:   point ``x_0 = 1`` (``H(1) = 1`` against ``gamma = 0.5772``). The exact
#:   target is printed; the asymptotic form is the recovery target, because
#:   ``H`` is not expressible in the operator set, and that is what the note
#:   records. The definition module is deliberately left untouched: the campaign
#:   executed against it, and ``solution_recovered`` is assessed against the
#:   asymptotic form.
DISPLAY_OVERRIDES: Final[dict[str, DisplayOverride]] = {
    "Keijzer-6": DisplayOverride(
        expression_latex="\\sum_{i=1}^{x_{0}} \\frac{1}{i}",
        note=(
            "the campaign target is the exact harmonic number "
            "$H(x_{0}) = \\sum_{i=1}^{x_{0}} 1/i$; solution recovery is assessed "
            "against its asymptotic form $\\log x_{0} + \\gamma$, which is the "
            "closest expression available in the operator set"
        ),
    ),
}


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
        expression_display_note: Optional caveat explaining why the printed
            expression differs from the module's ``sympy_expression``.
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
    expression_display_note: str = field(default="")

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


@dataclass(frozen=True)
class DataAgreement:
    """Agreement between a problem's recorded expression and its campaign data.

    Attributes:
        problem_id: Benchmark name.
        tier: Suite key.
        max_relative_error: Largest relative deviation over the compared rows.
        n_points: Number of rows actually compared.
        verdict: One of ``exact``, ``floating_point`` or ``approximation``.
    """

    problem_id: str
    tier: str
    max_relative_error: float
    n_points: int
    verdict: str


def classify_agreement(max_relative_error: float) -> str:
    """Classify a measured agreement.

    Args:
        max_relative_error: Largest relative deviation over the compared rows.

    Returns:
        ``exact`` when the two evaluations are bitwise identical,
        ``floating_point`` when they differ only within
        :data:`FLOATING_POINT_TOLERANCE`, otherwise ``approximation``.
    """
    if max_relative_error == 0.0:
        return VERDICT_EXACT
    if max_relative_error <= FLOATING_POINT_TOLERANCE:
        return VERDICT_FLOATING_POINT
    return VERDICT_APPROXIMATION


def measure_data_agreement(
    suite: str,
    bench: dict[str, Any],
    train_size: int,
    test_size: int,
    seed: int = DATA_AGREEMENT_SEED,
    max_points: int = DATA_AGREEMENT_MAX_POINTS,
) -> DataAgreement:
    """Compare a problem's recorded expression against the data it generated.

    The benchmark's own ``target_fn`` produces ``y`` through the orchestrator's
    dispatch, exactly as the campaign did. The recorded expression is evaluated
    independently on the same ``X`` and the two are compared row-wise. Rows on
    which either side is non-finite are excluded, because a protected operator
    and a bare SymPy evaluation are not required to agree outside the domain.

    Args:
        suite: Suite key.
        bench: Benchmark specification dict.
        train_size: ``train_size`` from the campaign YAML.
        test_size: ``test_size`` from the campaign YAML.
        seed: Seed passed to the data generator.
        max_points: Upper bound on the rows compared.

    Returns:
        The measured agreement.

    Raises:
        AppendixGenerationError: If the expression carries symbols outside
            ``x_0..x_{m-1}``, cannot be evaluated, or leaves no finite row.
    """
    name = str(bench["name"])
    n_vars = int(bench["num_variables"])
    symbols = [sp.Symbol(f"x_{i}") for i in range(n_vars)]
    expr = recorded_sympy_expression(bench)
    unexpected = {str(s) for s in expr.free_symbols} - {str(s) for s in symbols}
    if unexpected:
        raise AppendixGenerationError(
            f"Recorded expression for {name!r} carries symbols {sorted(unexpected)} "
            f"outside x_0..x_{n_vars - 1}"
        )

    x_train, y_train, _, _ = _generate_benchmark_data(suite, bench, train_size, test_size, seed)
    n_points = min(int(x_train.shape[0]), max_points)
    x_probe = x_train[:n_points]
    y_probe = np.asarray(y_train[:n_points], dtype=float).ravel()

    func = sp.lambdify(symbols, expr, "numpy")
    try:
        predicted = func(*[x_probe[:, i] for i in range(n_vars)])
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise AppendixGenerationError(
            f"Cannot evaluate the recorded expression for {name!r}: {exc}"
        ) from exc
    # ``lambdify`` returns a scalar for a constant expression; broadcast it.
    y_hat = np.asarray(predicted, dtype=float) * np.ones_like(y_probe)

    finite = np.isfinite(y_probe) & np.isfinite(y_hat)
    if not finite.any():
        raise AppendixGenerationError(
            f"No finite row to compare for {name!r}; the agreement audit cannot "
            "certify its recorded expression"
        )
    denominator = np.maximum(np.abs(y_probe[finite]), np.finfo(float).tiny)
    relative = np.abs(y_hat[finite] - y_probe[finite]) / denominator
    max_relative = float(np.max(relative))
    return DataAgreement(
        problem_id=name,
        tier=suite,
        max_relative_error=max_relative,
        n_points=int(finite.sum()),
        verdict=classify_agreement(max_relative),
    )


def audit_data_agreement(
    config_dir: Path,
    seed: int = DATA_AGREEMENT_SEED,
    max_points: int = DATA_AGREEMENT_MAX_POINTS,
) -> list[DataAgreement]:
    """Run the agreement audit over every campaign problem.

    Args:
        config_dir: Directory holding the campaign YAML configs.
        seed: Seed passed to the data generator.
        max_points: Upper bound on the rows compared per problem.

    Returns:
        One record per problem, in the emission order of :data:`SUITES`.
    """
    sizes = load_campaign_sizes(config_dir)
    records: list[DataAgreement] = []
    for suite, benches in SUITES.items():
        train_size, test_size = sizes[suite]
        for bench in benches:
            records.append(
                measure_data_agreement(suite, bench, train_size, test_size, seed, max_points)
            )
    return records


def validate_data_agreement(records: list[DataAgreement]) -> None:
    """Refuse to print an expression that disagrees with its own data.

    A problem may only be classified ``approximation`` when a
    :data:`DISPLAY_OVERRIDES` entry states what is printed instead and why. This
    is the durable guard: a benchmark whose recorded expression drifts from its
    ``target_fn`` fails the generator instead of reaching the appendix.

    Args:
        records: The measured agreements.

    Raises:
        AppendixGenerationError: If an ``approximation`` has no override.
    """
    undeclared = sorted(
        record.problem_id
        for record in records
        if record.verdict == VERDICT_APPROXIMATION and record.problem_id not in DISPLAY_OVERRIDES
    )
    if undeclared:
        detail = ", ".join(
            f"{r.problem_id} (max rel. err {r.max_relative_error:.3e} over {r.n_points} points)"
            for r in records
            if r.problem_id in undeclared
        )
        raise AppendixGenerationError(
            "Recorded expression disagrees with the generated data for "
            f"{len(undeclared)} problem(s) with no DISPLAY_OVERRIDES entry: {detail}. "
            "Add an override declaring what is printed and why, or correct the "
            "benchmark definition; do not print the recorded expression."
        )


def summarise_data_agreement(records: list[DataAgreement]) -> dict[str, int]:
    """Count the verdicts.

    Args:
        records: The measured agreements.

    Returns:
        Count per verdict plus the total, with every verdict key present.
    """
    counts = {
        verdict: sum(r.verdict == verdict for r in records)
        for verdict in (VERDICT_EXACT, VERDICT_FLOATING_POINT, VERDICT_APPROXIMATION)
    }
    counts["n_problems"] = len(records)
    return counts


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


def recorded_sympy_expression(bench: dict[str, Any]) -> sp.Expr:
    """Return the ground-truth expression a definition module records.

    The ``sympy_expression`` key is preferred where present (58 of 70 problems)
    because it is the object the solution-recovery check compares against, and
    because it is free of the annotations that contaminate some ``expression``
    strings. Nguyen benches carry no ``sympy_expression``; their string is
    parsed after ``^`` is rewritten to ``**`` and ``x``/``y`` to ``x_0``/``x_1``.

    This is the *recorded* expression, deliberately unaffected by
    ``DISPLAY_OVERRIDES``: the agreement audit has to see what the module
    claims, not the corrected form the appendix prints.

    Args:
        bench: A benchmark specification dict.

    Returns:
        The recorded expression as a SymPy object.

    Raises:
        AppendixGenerationError: If the expression string cannot be parsed.
    """
    sym_expr = bench.get("sympy_expression")
    if sym_expr is not None:
        return sp.sympify(sym_expr)
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
    return parsed.subs(subs) if subs else parsed


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
    sym_expr = recorded_sympy_expression(bench)
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
            override = DISPLAY_OVERRIDES.get(name)
            latex = override.expression_latex if override else expression_to_latex(bench)
            rows.append(
                BenchmarkRow(
                    problem_id=name,
                    tier=suite,
                    expression=str(bench["expression"]),
                    expression_latex=latex,
                    n_variables=int(bench["num_variables"]),
                    variable_ranges=resolve_variable_ranges(bench),
                    n_train=n_train,
                    n_test=n_test,
                    sampling_protocol=protocol,
                    sampling_protocol_label=SAMPLING_LABELS[protocol],
                    citation_key=citation,
                    citation_resolved=citation in bib_keys,
                    attribution_note=ATTRIBUTION_NOTES.get(name, ""),
                    expression_display_note=override.note if override else "",
                )
            )
    stale = set(DISPLAY_OVERRIDES) - {row.problem_id for row in rows}
    if stale:
        raise AppendixGenerationError(
            f"DISPLAY_OVERRIDES names problems absent from the suite: {sorted(stale)}"
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


def find_long_decimals(text: str) -> list[str]:
    """Return decimal literals carrying more than four significant figures.

    Leading zeros do not count as significant, so ``0.05`` is a two-figure
    coefficient while ``0.577215664901533`` is a fifteen-figure float dump.

    Args:
        text: Any emitted text.

    Returns:
        The offending literals, in order of appearance.
    """
    return [
        literal
        for literal in re.findall(r"\d+\.\d+", text)
        if len(literal.replace(".", "").lstrip("0")) > MAX_SIGNIFICANT_FIGURES
    ]


def validate_no_long_decimals(text: str, source: str) -> None:
    """Refuse raw float dumps in emitted LaTeX.

    Genuine benchmark coefficients are short by construction (Korns-12's
    ``2.1``, ``1.3``, ``9.8``; the Vladislavleva domain bound ``0.05``), so a
    long literal means a mathematical constant was evaluated numerically instead
    of being rendered as a symbol.

    Args:
        text: The emitted text.
        source: Name used in the error message.

    Raises:
        AppendixGenerationError: If any literal exceeds the significant-figure
            budget.
    """
    offenders = find_long_decimals(text)
    if offenders:
        raise AppendixGenerationError(
            f"{source} dumps {len(offenders)} decimal literal(s) with more than "
            f"{MAX_SIGNIFICANT_FIGURES} significant figures: {offenders}. Render "
            "the constant as a symbol via DISPLAY_OVERRIDES instead."
        )


def validate_expression_cells(rows: list[BenchmarkRow]) -> None:
    """Refuse a raw float dump in any printed expression.

    Only the expression cell is audited. The variable-range cells of six
    ODE-Strogatz rows carry five-figure literals (for example ``54.445``)
    because they are the empirical extents of the redistributed PMLB data, not
    numerically evaluated constants, and rounding them would misstate the
    sampled domain.

    Args:
        rows: All assembled rows.

    Raises:
        AppendixGenerationError: If a printed expression dumps a long decimal.
    """
    for row in rows:
        validate_no_long_decimals(row.expression_latex, f"expression of {row.problem_id!r}")


def render_display_note_block(rows: list[BenchmarkRow]) -> list[str]:
    """Render the display notes of a table as LaTeX comment lines.

    The body files carry no caption, so the notes travel beside them as
    comments for the including document to place.

    Args:
        rows: The rows of one table, in emission order.

    Returns:
        Comment lines, or an empty list when no row carries a note.
    """
    noted = [row for row in rows if row.expression_display_note]
    if not noted:
        return []
    lines = ["% Display notes for this table (place in the caption):"]
    lines += [f"%   {row.problem_id}: {row.expression_display_note}." for row in noted]
    return lines


def select_supplementary_rows(rows: list[BenchmarkRow]) -> list[BenchmarkRow]:
    """Select the problems the shipped Appendix D.1 does not document.

    Args:
        rows: All assembled rows.

    Returns:
        The rows of the three undocumented tiers, ordered by variable count and
        then by problem id. The ordering is presentational only and mirrors the
        shipped ``tab:feynman``; it deliberately does not follow tier order.
    """
    selected = [row for row in rows if row.tier in SUPPLEMENTARY_TIERS]
    return sorted(selected, key=lambda row: (row.n_variables, row.problem_id))


def split_supplementary_rows(
    rows: list[BenchmarkRow],
) -> tuple[list[BenchmarkRow], list[BenchmarkRow]]:
    """Split the supplementary rows into the AI Feynman half and the rest.

    Args:
        rows: Rows returned by :func:`select_supplementary_rows`.

    Returns:
        ``(feynman_rows, other_rows)``, each preserving the input order.
    """
    feynman = [row for row in rows if row.citation_key == FEYNMAN_CITATION_KEY]
    other = [row for row in rows if row.citation_key != FEYNMAN_CITATION_KEY]
    return feynman, other


def validate_supplementary_split(feynman: list[BenchmarkRow], other: list[BenchmarkRow]) -> None:
    """Fail hard unless both supplementary groups hold 14 problems.

    Args:
        feynman: The AI Feynman group.
        other: The group drawn from the six further benchmark families.

    Raises:
        AppendixGenerationError: If either group has the wrong size or the
            non-Feynman group does not span exactly six sources.
    """
    for label, group in (("AI Feynman", feynman), ("further-sources", other)):
        if len(group) != SUPP_GROUP_SIZE:
            raise AppendixGenerationError(
                f"Supplementary {label} group holds {len(group)} problems, "
                f"expected {SUPP_GROUP_SIZE}"
            )
    sources = {row.citation_key for row in other}
    if len(sources) != SUPP_OTHER_SOURCE_COUNT:
        raise AppendixGenerationError(
            f"Further-sources group spans {len(sources)} publications "
            f"({sorted(sources)}), expected {SUPP_OTHER_SOURCE_COUNT}"
        )


def validate_supplementary_feynman_uniformity(rows: list[BenchmarkRow]) -> None:
    """Fail hard unless every AI Feynman row shares the caption-level sampling.

    The train/test sizes of this group are stated in the caption rather than in
    a column, which is only sound while every row agrees.

    Args:
        rows: The AI Feynman group.

    Raises:
        AppendixGenerationError: If any row deviates from the shared protocol.
    """
    expected = SUPP_FEYNMAN_SAMPLING
    for row in rows:
        actual = (row.n_train, row.n_test, row.sampling_protocol)
        if actual != expected:
            raise AppendixGenerationError(
                f"Problem {row.problem_id!r} samples {actual}, but the "
                f"supplementary AI Feynman table states {expected} "
                f"({expected[0]} train / {expected[1]} test) in its caption; "
                "add per-row size columns instead of widening this guard"
            )


def validate_supplementary_body(text: str) -> None:
    """Structurally check a supplementary tabular body.

    Args:
        text: The emitted body.

    Raises:
        AppendixGenerationError: If the body carries a float, a caption, a label
            or an internal tier name, or fails the shared LaTeX checks.
    """
    validate_latex_fragment(text)
    for forbidden in ("\\begin{table}", "\\caption", "\\label"):
        if forbidden in text:
            raise AppendixGenerationError(
                f"Supplementary body must be float-free but carries {forbidden!r}"
            )
    if text.count("\\begin{tabular}") != 1 or text.count("\\end{tabular}") != 1:
        raise AppendixGenerationError("Supplementary body must hold exactly one tabular")
    lowered = text.lower()
    for word in _FORBIDDEN_TIER_WORDS:
        if word in lowered:
            raise AppendixGenerationError(
                f"Supplementary body leaks the internal tier word {word!r}"
            )


def _supp_common_cells(row: BenchmarkRow) -> tuple[str, str, str, str]:
    """Render the four columns shared by both supplementary tables.

    Args:
        row: The documented problem.

    Returns:
        ``(id, expression, m, variable ranges)`` as LaTeX cells.
    """
    return (
        escape_latex_text(row.problem_id),
        f"${row.expression_latex}$",
        str(row.n_variables),
        f"${format_variable_ranges(row.variable_ranges)}$",
    )


def render_supp_feynman_tabular(rows: list[BenchmarkRow]) -> str:
    """Render the AI Feynman supplementary table body.

    The column specification, header wording and rule set mirror ``tab:feynman``
    in the shipped supplementary. No float, caption or label is emitted: those
    are supplied by the including document.

    Args:
        rows: The AI Feynman group, in emission order.

    Returns:
        The tabular body, newline-terminated.
    """
    lines = [
        *render_display_note_block(rows),
        f"\\setlength{{\\tabcolsep}}{{{SUPP_TABCOLSEP}}}",
        "\\begin{tabular}{@{}llcl@{}}",
        "\\toprule",
        "ID & Expression & $m$ & Variable ranges \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(_supp_common_cells(row)) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def render_supp_other_tabular(rows: list[BenchmarkRow]) -> str:
    """Render the supplementary table body for the six further sources.

    These problems carry heterogeneous sampling protocols and dataset sizes, so
    the shared four-column layout is extended with per-row sizes and protocol.

    Args:
        rows: The non-Feynman group, in emission order.

    Returns:
        The tabular body, newline-terminated.
    """
    lines = [
        *render_display_note_block(rows),
        f"\\setlength{{\\tabcolsep}}{{{SUPP_TABCOLSEP}}}",
        "\\begin{tabular}{@{}llclrrl@{}}",
        "\\toprule",
        "ID & Expression & $m$ & Variable ranges & $n_{\\mathrm{train}}$ & "
        "$n_{\\mathrm{test}}$ & Sampling \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = (
            *_supp_common_cells(row),
            str(row.n_train),
            str(row.n_test),
            escape_latex_text(row.sampling_protocol_label),
        )
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def render_supplementary_bodies(rows: list[BenchmarkRow]) -> dict[str, str]:
    """Render both supplementary tabular bodies, validating them first.

    Args:
        rows: All assembled rows.

    Returns:
        Mapping from output filename to body text.

    Raises:
        AppendixGenerationError: If the selection, the group sizes, the sampling
            premise, any mandatory field or the emitted LaTeX fails validation.
    """
    selected = select_supplementary_rows(rows)
    validate_rows(selected)
    feynman, other = split_supplementary_rows(selected)
    validate_supplementary_split(feynman, other)
    validate_supplementary_feynman_uniformity(feynman)
    bodies = {
        SUPP_FEYNMAN_FILENAME: render_supp_feynman_tabular(feynman),
        SUPP_OTHER_FILENAME: render_supp_other_tabular(other),
    }
    for body in bodies.values():
        validate_supplementary_body(body)
    return bodies


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


def build_data_agreement_block(
    records: list[DataAgreement],
    seed: int,
    max_points: int,
) -> dict[str, Any]:
    """Assemble the machine-readable agreement block.

    Args:
        records: The measured agreements.
        seed: Seed used to generate the compared data.
        max_points: Upper bound on the rows compared per problem.

    Returns:
        JSON-serialisable block with the protocol, the verdict counts and the
        per-problem measurements.
    """
    return {
        "protocol": {
            "description": (
                "The recorded ground-truth expression of each problem is "
                "evaluated on the X the campaign's own sampling protocol "
                "generates, and compared row-wise against the benchmark's "
                "target_fn output y. DISPLAY_OVERRIDES is deliberately not "
                "applied: the audit measures what the definition module "
                "records, not what the appendix prints."
            ),
            "seed": seed,
            "max_points_per_problem": max_points,
            "floating_point_tolerance": FLOATING_POINT_TOLERANCE,
            "verdicts": {
                VERDICT_EXACT: "max relative error is exactly 0",
                VERDICT_FLOATING_POINT: (f"0 < max relative error <= {FLOATING_POINT_TOLERANCE:g}"),
                VERDICT_APPROXIMATION: (
                    f"max relative error > {FLOATING_POINT_TOLERANCE:g}; requires a "
                    "DISPLAY_OVERRIDES entry"
                ),
            },
        },
        "summary": summarise_data_agreement(records),
        "problems_requiring_display_override": sorted(
            r.problem_id for r in records if r.verdict == VERDICT_APPROXIMATION
        ),
        "per_problem": [asdict(record) for record in records],
    }


def build_json_payload(
    rows: list[BenchmarkRow],
    audit: dict[str, Any],
    sizes: dict[str, tuple[int, int]],
    bib_path: Path,
    seed: int,
    data_agreement: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the machine-readable payload.

    Args:
        rows: All assembled rows.
        audit: Citation audit block.
        sizes: Per-suite ``(train_size, test_size)`` from the campaign YAMLs.
        bib_path: Path the citation keys were checked against.
        seed: Seed used to realise the array shapes.
        data_agreement: Expression-versus-data agreement block.

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
        "data_agreement": data_agreement,
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
        The written paths: the appendix fragment, the JSON inventory, then the
        two supplementary tabular bodies.

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
    validate_expression_cells(rows)
    supp_bodies = render_supplementary_bodies(rows)

    agreement_records = audit_data_agreement(config_dir, seed, DATA_AGREEMENT_MAX_POINTS)
    validate_data_agreement(agreement_records)
    agreement_block = build_data_agreement_block(agreement_records, seed, DATA_AGREEMENT_MAX_POINTS)
    counts = agreement_block["summary"]
    log.info(
        "Expression-vs-data agreement: %d exact, %d floating-point, %d approximation",
        counts[VERDICT_EXACT],
        counts[VERDICT_FLOATING_POINT],
        counts[VERDICT_APPROXIMATION],
    )

    sizes = load_campaign_sizes(config_dir)
    payload = build_json_payload(rows, audit, sizes, bib_path, seed, agreement_block)

    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / "appendix_d_tables.tex"
    json_path = out_dir / "appendix_d_benchmarks.json"
    tex_path.write_text(tex, encoding="utf-8", newline="\n")
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    supp_paths: list[Path] = []
    for name in (SUPP_FEYNMAN_FILENAME, SUPP_OTHER_FILENAME):
        path = out_dir / name
        path.write_text(supp_bodies[name], encoding="utf-8", newline="\n")
        supp_paths.append(path)

    log.info("Wrote %d problems to %s and %s", len(rows), tex_path, json_path)
    log.info(
        "Wrote the %d-problem supplementary extension to %s",
        2 * SUPP_GROUP_SIZE,
        ", ".join(str(p) for p in supp_paths),
    )
    return [tex_path, json_path, *supp_paths]


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
