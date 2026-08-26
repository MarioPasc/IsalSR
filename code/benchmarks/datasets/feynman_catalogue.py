"""Complete AI Feynman equation database (100 base + 20 bonus) with a Sigma_SR classifier.

This module is *not* a benchmark suite: it defines no samplers and no ``generate_data``
entry point. It is the evidence artefact behind the answer to reviewer comment R3.1,
"how many of the 120 AI Feynman equations does the operator-compatibility criterion
actually exclude, and which ones".

Provenance of the embedded data
-------------------------------
* The 100 base equations come from ``FeynmanEquations.csv`` of the original MIT
  AI Feynman database (Udrescu & Tegmark, *Sci. Adv.* 6(16):eaay2631, 2020,
  DOI 10.1126/sciadv.aay2631), recovered from an Internet Archive snapshot.
* The 20 bonus equations, and the *variable ranges* of the 99 base equations that
  have a PMLB counterpart, come from the PMLB ``feynman_*`` dataset metadata
  (Olson et al., *BioData Mining* 10:36, 2017; La Cava et al., SRBench, NeurIPS 2021).

Reconciliation between the two sources, encoded in the literal data below:

* Base equation ``II.11.17`` is the only one with no PMLB dataset, hence
  ``pmlb_id is None`` for exactly one row. 99 base + 20 bonus = 119 PMLB datasets;
  100 base + 20 bonus = 120 catalogue rows.
* The ``Filename`` column of the distributed CSV truncates a trailing zero on six
  equation numbers. The catalogue ``id`` uses the *Feynman-lecture* equation number,
  as printed in the AI Feynman paper's table and as used by the IsalSR suite:
  ``I.6.2a -> I.6.20a``, ``I.6.2 -> I.6.20``, ``I.6.2b -> I.6.20b``,
  ``I.15.1 -> I.15.10``, ``I.39.1 -> I.39.10``, ``I.48.2 -> I.48.20``.
  The un-truncated ``pmlb_id`` is retained verbatim in each row, so the mapping back
  to the distributed files is never lost.
* Where the CSV variable block is truncated or blank (``I.18.12``, ``I.18.14``,
  ``I.38.12``, ``II.37.1``, ``III.10.19``, ``III.19.51``), the PMLB ``features``/
  ``ranges`` are authoritative. After this repair, for every one of the 120 rows the
  free symbols of the sympified formula equal exactly the declared variable names.
  The CSV and PMLB formula strings agree on all 99 shared equations.

Sigma_SR
--------
Sigma_SR is the paper's Definition 3.2 alphabet: exactly twelve labels ---
``Add, Mul, Sin, Cos, Exp, Log, Sqrt, Pow, Abs, Neg, Inv, Const`` --- plus pre-inserted
``Var`` leaves. There is no ``Sub`` and no ``Div``: ``a - b = Add(a, Neg(b))`` and
``a / b = Mul(a, Inv(b))``. Consequently sympy's ``Add``, ``Mul`` and ``Pow`` all map
into Sigma_SR, as do ``Integer``, ``Rational``, ``Float`` and ``pi`` (-> ``Const``) and
``Symbol`` (-> ``Var``). A rule that treats ``Pow`` as suspicious would be wrong: sympy
canonicalises ``a/b`` into ``Mul(a, Pow(b, -1))`` and ``sqrt(a)`` into
``Pow(a, Rational(1, 2))``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TypeAlias

import sympy

logger = logging.getLogger(__name__)

VariableSpec: TypeAlias = dict[str, Any]
EquationRecord: TypeAlias = dict[str, Any]

__all__ = [
    "AIFEYNMAN_120",
    "IN_SUITE_IDS",
    "SEMANTICALLY_DERIVABLE",
    "SIGMA_SR_FUNCTIONS",
    "SigmaSRClassification",
    "classification_table",
    "classify_sigma_sr",
    "eligible_extension_pool",
    "get_equation",
]

# --------------------------------------------------------------------------------------
# Sigma_SR alphabet, expressed at the level of sympy node types
# --------------------------------------------------------------------------------------

#: Sympy expression heads that are Sigma_SR *structure* rather than function symbols.
#: ``Add`` and ``Mul`` are the variable-arity operations; ``Pow`` is the binary one.
#: ``Neg``, ``Inv`` and ``Sqrt`` never survive sympy's canonicalisation as distinct
#: heads --- they appear as ``Mul(-1, .)``, ``Pow(., -1)`` and ``Pow(., 1/2)``.
_STRUCTURAL_HEADS: frozenset[str] = frozenset({"Add", "Mul", "Pow"})

#: Function symbols that a published formula may write and that map directly into
#: Sigma_SR. ``ln`` is sympy's alias of ``log``; ``abs``/``Abs`` are the two spellings
#: of the absolute value; ``sqrt`` is listed for completeness although sympy rewrites
#: it as a ``Pow``.
SIGMA_SR_FUNCTIONS: frozenset[str] = frozenset(
    {"sin", "cos", "exp", "log", "ln", "sqrt", "Abs", "abs"}
)

#: Function symbols outside Sigma_SR that are nevertheless *equal* to a finite Sigma_SR
#: expression, so they block the syntactic reading but not the semantic one. Each is a
#: finite composition of ``exp``/``sin``/``cos`` with ``Add``, ``Mul``, ``Neg``, ``Inv``:
#: ``tanh(u) = (e^u - e^-u) * (e^u + e^-u)^-1``, ``tan(u) = sin(u) * cos(u)^-1``, and so
#: on. The inverse trigonometric functions and ``sign`` are *not* in this set: no finite
#: composition of the twelve labels equals them.
SEMANTICALLY_DERIVABLE: frozenset[str] = frozenset(
    {"tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "coth", "sech", "csch"}
)


@dataclass(frozen=True)
class SigmaSRClassification:
    """Verdict of the Sigma_SR representability test for one formula.

    Two readings are reported because they disagree, and the disagreement is itself
    the finding: the paper's stated exclusion criterion is syntactic (it names
    ``tanh``, ``arctan``, ``sgn`` as excluded), whereas the representable class is
    strictly larger under the semantic reading.

    Attributes:
        representable_syntactic: True when every function symbol written in the
            published formula is in :data:`SIGMA_SR_FUNCTIONS`.
        blocking_ops_syntactic: Offending function symbols under the syntactic
            reading, sorted; empty when representable.
        representable_semantic: True when a finite Sigma_SR expression equal to the
            formula exists, i.e. every offending symbol is in
            :data:`SEMANTICALLY_DERIVABLE`.
        blocking_ops_semantic: Offending function symbols under the semantic reading,
            sorted; empty when representable.
        functions_used: Every non-structural expression head appearing in the parsed
            tree, sorted.
    """

    representable_syntactic: bool
    blocking_ops_syntactic: tuple[str, ...]
    representable_semantic: bool
    blocking_ops_semantic: tuple[str, ...]
    functions_used: tuple[str, ...]


def _parse(formula: str, variables: list[str]) -> sympy.Expr:
    """Sympify a formula with an explicit symbol table.

    An explicit ``locals`` mapping is required, not optional: bare ``sympify`` would
    silently reinterpret variable names such as ``E`` (Euler's number), ``I`` (the
    imaginary unit), ``beta``, ``gamma``, ``zeta``, ``O``, ``S`` or ``N`` as sympy
    objects, corrupting a row without raising. ``ln`` is bound to ``sympy.log``.

    Args:
        formula: Right-hand side of the equation, in sympy-compatible syntax.
        variables: Names of the free variables the formula is declared over.

    Returns:
        The parsed expression.

    Raises:
        sympy.SympifyError: If the formula cannot be parsed.
    """
    local_dict: dict[str, Any] = {name: sympy.Symbol(name) for name in variables}
    local_dict["ln"] = sympy.log
    return sympy.sympify(formula, locals=local_dict)


def _heads(expr: sympy.Expr) -> set[str]:
    """Collect every non-structural expression head in a parsed formula.

    Symbols and numeric literals (including ``pi``, which is a ``Const`` in Sigma_SR)
    are skipped, as are ``Add``, ``Mul`` and ``Pow``. Everything else --- applied
    sympy functions such as ``sin``, applied undefined functions such as ``arcsin``,
    and any exotic head such as ``Piecewise`` --- is reported by class name.

    Args:
        expr: Parsed expression.

    Returns:
        Set of head names.
    """
    heads: set[str] = set()
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, sympy.Symbol):
            continue
        if isinstance(node, (sympy.Number, sympy.NumberSymbol)):
            continue
        name = type(node).__name__
        if name in _STRUCTURAL_HEADS:
            continue
        heads.add(name)
    return heads


def classify_sigma_sr(formula: str, variables: list[str]) -> SigmaSRClassification:
    """Decide whether a formula is representable in Sigma_SR, under both readings.

    The formula is parsed into a sympy tree and classified by walking the tree and
    inspecting node types. Classification is deliberately *not* done by regular
    expression on the source string: the tree walk is invariant to spelling
    (``a - b`` versus ``a + (-1)*b``) and cannot be fooled by a substring match.

    Args:
        formula: Right-hand side of the equation, in sympy-compatible syntax.
        variables: Names of the free variables the formula is declared over.

    Returns:
        The two verdicts and the function symbols that produced them.

    Raises:
        sympy.SympifyError: If the formula cannot be parsed.
    """
    expr = _parse(formula, variables)
    used = _heads(expr)
    blocking_syn = tuple(sorted(h for h in used if h not in SIGMA_SR_FUNCTIONS))
    blocking_sem = tuple(h for h in blocking_syn if h not in SEMANTICALLY_DERIVABLE)
    return SigmaSRClassification(
        representable_syntactic=not blocking_syn,
        blocking_ops_syntactic=blocking_syn,
        representable_semantic=not blocking_sem,
        blocking_ops_semantic=blocking_sem,
        functions_used=tuple(sorted(used)),
    )


# --------------------------------------------------------------------------------------
# The catalogue, embedded as literal data
# --------------------------------------------------------------------------------------
# Row layout: (id, source, output, formula, ((var, low, high), ...), pmlb_id)
# Generated once from the two cached sources described in the module docstring and
# frozen here; nothing is read from disk at import time.

_RawVariable: TypeAlias = tuple[str, float, float]
_RawRow: TypeAlias = tuple[str, str, str, str, tuple[_RawVariable, ...], "str | None"]

_RAW: tuple[_RawRow, ...] = (
    (
        "I.10.7",
        "base",
        "m",
        "m_0/sqrt(1-v**2/c**2)",
        (("m_0", 1.0, 5.0), ("v", 1.0, 2.0), ("c", 3.0, 10.0)),
        "feynman_I_10_7",
    ),
    (
        "I.11.19",
        "base",
        "A",
        "x1*y1+x2*y2+x3*y3",
        (
            ("x1", 1.0, 5.0),
            ("x2", 1.0, 5.0),
            ("x3", 1.0, 5.0),
            ("y1", 1.0, 5.0),
            ("y2", 1.0, 5.0),
            ("y3", 1.0, 5.0),
        ),
        "feynman_I_11_19",
    ),
    ("I.12.1", "base", "F", "mu*Nn", (("mu", 1.0, 5.0), ("Nn", 1.0, 5.0)), "feynman_I_12_1"),
    (
        "I.12.11",
        "base",
        "F",
        "q*(Ef+B*v*sin(theta))",
        (("q", 1.0, 5.0), ("Ef", 1.0, 5.0), ("B", 1.0, 5.0), ("v", 1.0, 5.0), ("theta", 1.0, 5.0)),
        "feynman_I_12_11",
    ),
    (
        "I.12.2",
        "base",
        "F",
        "q1*q2*r/(4*pi*epsilon*r**3)",
        (("q1", 1.0, 5.0), ("q2", 1.0, 5.0), ("epsilon", 1.0, 5.0), ("r", 1.0, 5.0)),
        "feynman_I_12_2",
    ),
    (
        "I.12.4",
        "base",
        "Ef",
        "q1*r/(4*pi*epsilon*r**3)",
        (("q1", 1.0, 5.0), ("epsilon", 1.0, 5.0), ("r", 1.0, 5.0)),
        "feynman_I_12_4",
    ),
    ("I.12.5", "base", "F", "q2*Ef", (("q2", 1.0, 5.0), ("Ef", 1.0, 5.0)), "feynman_I_12_5"),
    (
        "I.13.12",
        "base",
        "U",
        "G*m1*m2*(1/r2-1/r1)",
        (("m1", 1.0, 5.0), ("m2", 1.0, 5.0), ("r1", 1.0, 5.0), ("r2", 1.0, 5.0), ("G", 1.0, 5.0)),
        "feynman_I_13_12",
    ),
    (
        "I.13.4",
        "base",
        "K",
        "1/2*m*(v**2+u**2+w**2)",
        (("m", 1.0, 5.0), ("v", 1.0, 5.0), ("u", 1.0, 5.0), ("w", 1.0, 5.0)),
        "feynman_I_13_4",
    ),
    (
        "I.14.3",
        "base",
        "U",
        "m*g*z",
        (("m", 1.0, 5.0), ("g", 1.0, 5.0), ("z", 1.0, 5.0)),
        "feynman_I_14_3",
    ),
    (
        "I.14.4",
        "base",
        "U",
        "1/2*k_spring*x**2",
        (("k_spring", 1.0, 5.0), ("x", 1.0, 5.0)),
        "feynman_I_14_4",
    ),
    (
        "I.15.10",
        "base",
        "p",
        "m_0*v/sqrt(1-v**2/c**2)",
        (("m_0", 1.0, 5.0), ("v", 1.0, 2.0), ("c", 3.0, 10.0)),
        "feynman_I_15_10",
    ),
    (
        "I.15.3t",
        "base",
        "t1",
        "(t-u*x/c**2)/sqrt(1-u**2/c**2)",
        (("x", 1.0, 5.0), ("c", 3.0, 10.0), ("u", 1.0, 2.0), ("t", 1.0, 5.0)),
        "feynman_I_15_3t",
    ),
    (
        "I.15.3x",
        "base",
        "x1",
        "(x-u*t)/sqrt(1-u**2/c**2)",
        (("x", 5.0, 10.0), ("u", 1.0, 2.0), ("c", 3.0, 20.0), ("t", 1.0, 2.0)),
        "feynman_I_15_3x",
    ),
    (
        "I.16.6",
        "base",
        "v1",
        "(u+v)/(1+u*v/c**2)",
        (("c", 1.0, 5.0), ("v", 1.0, 5.0), ("u", 1.0, 5.0)),
        "feynman_I_16_6",
    ),
    (
        "I.18.12",
        "base",
        "tau",
        "r*F*sin(theta)",
        (("r", 1.0, 5.0), ("F", 1.0, 5.0), ("theta", 0.0, 5.0)),
        "feynman_I_18_12",
    ),
    (
        "I.18.14",
        "base",
        "L",
        "m*r*v*sin(theta)",
        (("m", 1.0, 5.0), ("r", 1.0, 5.0), ("v", 1.0, 5.0), ("theta", 1.0, 5.0)),
        "feynman_I_18_14",
    ),
    (
        "I.18.4",
        "base",
        "r",
        "(m1*r1+m2*r2)/(m1+m2)",
        (("m1", 1.0, 5.0), ("m2", 1.0, 5.0), ("r1", 1.0, 5.0), ("r2", 1.0, 5.0)),
        "feynman_I_18_4",
    ),
    (
        "I.24.6",
        "base",
        "E_n",
        "1/2*m*(omega**2+omega_0**2)*1/2*x**2",
        (("m", 1.0, 3.0), ("omega", 1.0, 3.0), ("omega_0", 1.0, 3.0), ("x", 1.0, 3.0)),
        "feynman_I_24_6",
    ),
    ("I.25.13", "base", "Volt", "q/C", (("q", 1.0, 5.0), ("C", 1.0, 5.0)), "feynman_I_25_13"),
    (
        "I.26.2",
        "base",
        "theta1",
        "arcsin(n*sin(theta2))",
        (("n", 0.0, 1.0), ("theta2", 1.0, 5.0)),
        "feynman_I_26_2",
    ),
    (
        "I.27.6",
        "base",
        "foc",
        "1/(1/d1+n/d2)",
        (("d1", 1.0, 5.0), ("d2", 1.0, 5.0), ("n", 1.0, 5.0)),
        "feynman_I_27_6",
    ),
    (
        "I.29.16",
        "base",
        "x",
        "sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))",
        (("x1", 1.0, 5.0), ("x2", 1.0, 5.0), ("theta1", 1.0, 5.0), ("theta2", 1.0, 5.0)),
        "feynman_I_29_16",
    ),
    ("I.29.4", "base", "k", "omega/c", (("omega", 1.0, 10.0), ("c", 1.0, 10.0)), "feynman_I_29_4"),
    (
        "I.30.3",
        "base",
        "Int",
        "Int_0*sin(n*theta/2)**2/sin(theta/2)**2",
        (("Int_0", 1.0, 5.0), ("theta", 1.0, 5.0), ("n", 1.0, 5.0)),
        "feynman_I_30_3",
    ),
    (
        "I.30.5",
        "base",
        "theta",
        "arcsin(lambd/(n*d))",
        (("lambd", 1.0, 2.0), ("d", 2.0, 5.0), ("n", 1.0, 5.0)),
        "feynman_I_30_5",
    ),
    (
        "I.32.17",
        "base",
        "Pwr",
        "(1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)",
        (
            ("epsilon", 1.0, 2.0),
            ("c", 1.0, 2.0),
            ("Ef", 1.0, 2.0),
            ("r", 1.0, 2.0),
            ("omega", 1.0, 2.0),
            ("omega_0", 3.0, 5.0),
        ),
        "feynman_I_32_17",
    ),
    (
        "I.32.5",
        "base",
        "Pwr",
        "q**2*a**2/(6*pi*epsilon*c**3)",
        (("q", 1.0, 5.0), ("a", 1.0, 5.0), ("epsilon", 1.0, 5.0), ("c", 1.0, 5.0)),
        "feynman_I_32_5",
    ),
    (
        "I.34.1",
        "base",
        "omega",
        "omega_0/(1-v/c)",
        (("c", 3.0, 10.0), ("v", 1.0, 2.0), ("omega_0", 1.0, 5.0)),
        "feynman_I_34_1",
    ),
    (
        "I.34.14",
        "base",
        "omega",
        "(1+v/c)/sqrt(1-v**2/c**2)*omega_0",
        (("c", 3.0, 10.0), ("v", 1.0, 2.0), ("omega_0", 1.0, 5.0)),
        "feynman_I_34_14",
    ),
    (
        "I.34.27",
        "base",
        "E_n",
        "(h/(2*pi))*omega",
        (("omega", 1.0, 5.0), ("h", 1.0, 5.0)),
        "feynman_I_34_27",
    ),
    (
        "I.34.8",
        "base",
        "omega",
        "q*v*B/p",
        (("q", 1.0, 5.0), ("v", 1.0, 5.0), ("B", 1.0, 5.0), ("p", 1.0, 5.0)),
        "feynman_I_34_8",
    ),
    (
        "I.37.4",
        "base",
        "Int",
        "I1+I2+2*sqrt(I1*I2)*cos(delta)",
        (("I1", 1.0, 5.0), ("I2", 1.0, 5.0), ("delta", 1.0, 5.0)),
        "feynman_I_37_4",
    ),
    (
        "I.38.12",
        "base",
        "r",
        "4*pi*epsilon*(h/(2*pi))**2/(m*q**2)",
        (("m", 1.0, 5.0), ("q", 1.0, 5.0), ("h", 1.0, 5.0), ("epsilon", 1.0, 5.0)),
        "feynman_I_38_12",
    ),
    ("I.39.10", "base", "E_n", "3/2*pr*V", (("pr", 1.0, 5.0), ("V", 1.0, 5.0)), "feynman_I_39_1"),
    (
        "I.39.11",
        "base",
        "E_n",
        "1/(gamma-1)*pr*V",
        (("gamma", 2.0, 5.0), ("pr", 1.0, 5.0), ("V", 1.0, 5.0)),
        "feynman_I_39_11",
    ),
    (
        "I.39.22",
        "base",
        "pr",
        "n*kb*T/V",
        (("n", 1.0, 5.0), ("T", 1.0, 5.0), ("V", 1.0, 5.0), ("kb", 1.0, 5.0)),
        "feynman_I_39_22",
    ),
    (
        "I.40.1",
        "base",
        "n",
        "n_0*exp(-m*g*x/(kb*T))",
        (
            ("n_0", 1.0, 5.0),
            ("m", 1.0, 5.0),
            ("x", 1.0, 5.0),
            ("T", 1.0, 5.0),
            ("g", 1.0, 5.0),
            ("kb", 1.0, 5.0),
        ),
        "feynman_I_40_1",
    ),
    (
        "I.41.16",
        "base",
        "L_rad",
        "h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))",
        (("omega", 1.0, 5.0), ("T", 1.0, 5.0), ("h", 1.0, 5.0), ("kb", 1.0, 5.0), ("c", 1.0, 5.0)),
        "feynman_I_41_16",
    ),
    (
        "I.43.16",
        "base",
        "v",
        "mu_drift*q*Volt/d",
        (("mu_drift", 1.0, 5.0), ("q", 1.0, 5.0), ("Volt", 1.0, 5.0), ("d", 1.0, 5.0)),
        "feynman_I_43_16",
    ),
    (
        "I.43.31",
        "base",
        "D",
        "mob*kb*T",
        (("mob", 1.0, 5.0), ("T", 1.0, 5.0), ("kb", 1.0, 5.0)),
        "feynman_I_43_31",
    ),
    (
        "I.43.43",
        "base",
        "kappa",
        "1/(gamma-1)*kb*v/A",
        (("gamma", 2.0, 5.0), ("kb", 1.0, 5.0), ("A", 1.0, 5.0), ("v", 1.0, 5.0)),
        "feynman_I_43_43",
    ),
    (
        "I.44.4",
        "base",
        "E_n",
        "n*kb*T*ln(V2/V1)",
        (("n", 1.0, 5.0), ("kb", 1.0, 5.0), ("T", 1.0, 5.0), ("V1", 1.0, 5.0), ("V2", 1.0, 5.0)),
        "feynman_I_44_4",
    ),
    (
        "I.47.23",
        "base",
        "c",
        "sqrt(gamma*pr/rho)",
        (("gamma", 1.0, 5.0), ("pr", 1.0, 5.0), ("rho", 1.0, 5.0)),
        "feynman_I_47_23",
    ),
    (
        "I.48.20",
        "base",
        "E_n",
        "m*c**2/sqrt(1-v**2/c**2)",
        (("m", 1.0, 5.0), ("v", 1.0, 2.0), ("c", 3.0, 10.0)),
        "feynman_I_48_2",
    ),
    (
        "I.50.26",
        "base",
        "x",
        "x1*(cos(omega*t)+alpha*cos(omega*t)**2)",
        (("x1", 1.0, 3.0), ("omega", 1.0, 3.0), ("t", 1.0, 3.0), ("alpha", 1.0, 3.0)),
        "feynman_I_50_26",
    ),
    (
        "I.6.20",
        "base",
        "f",
        "exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)",
        (("sigma", 1.0, 3.0), ("theta", 1.0, 3.0)),
        "feynman_I_6_2",
    ),
    (
        "I.6.20a",
        "base",
        "f",
        "exp(-theta**2/2)/sqrt(2*pi)",
        (("theta", 1.0, 3.0),),
        "feynman_I_6_2a",
    ),
    (
        "I.6.20b",
        "base",
        "f",
        "exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)",
        (("sigma", 1.0, 3.0), ("theta", 1.0, 3.0), ("theta1", 1.0, 3.0)),
        "feynman_I_6_2b",
    ),
    (
        "I.8.14",
        "base",
        "d",
        "sqrt((x2-x1)**2+(y2-y1)**2)",
        (("x1", 1.0, 5.0), ("x2", 1.0, 5.0), ("y1", 1.0, 5.0), ("y2", 1.0, 5.0)),
        "feynman_I_8_14",
    ),
    (
        "I.9.18",
        "base",
        "F",
        "G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)",
        (
            ("m1", 1.0, 2.0),
            ("m2", 1.0, 2.0),
            ("G", 1.0, 2.0),
            ("x1", 3.0, 4.0),
            ("x2", 1.0, 2.0),
            ("y1", 3.0, 4.0),
            ("y2", 1.0, 2.0),
            ("z1", 3.0, 4.0),
            ("z2", 1.0, 2.0),
        ),
        "feynman_I_9_18",
    ),
    (
        "II.10.9",
        "base",
        "Ef",
        "sigma_den/epsilon*1/(1+chi)",
        (("sigma_den", 1.0, 5.0), ("epsilon", 1.0, 5.0), ("chi", 1.0, 5.0)),
        "feynman_II_10_9",
    ),
    (
        "II.11.17",
        "base",
        "n",
        "n_0*(1+p_d*Ef*cos(theta)/(kb*T))",
        (
            ("n_0", 1.0, 3.0),
            ("kb", 1.0, 3.0),
            ("T", 1.0, 3.0),
            ("theta", 1.0, 3.0),
            ("p_d", 1.0, 3.0),
            ("Ef", 1.0, 3.0),
        ),
        None,
    ),
    (
        "II.11.20",
        "base",
        "Pol",
        "n_rho*p_d**2*Ef/(3*kb*T)",
        (
            ("n_rho", 1.0, 5.0),
            ("p_d", 1.0, 5.0),
            ("Ef", 1.0, 5.0),
            ("kb", 1.0, 5.0),
            ("T", 1.0, 5.0),
        ),
        "feynman_II_11_20",
    ),
    (
        "II.11.27",
        "base",
        "Pol",
        "n*alpha/(1-(n*alpha/3))*epsilon*Ef",
        (("n", 0.0, 1.0), ("alpha", 0.0, 1.0), ("epsilon", 1.0, 2.0), ("Ef", 1.0, 2.0)),
        "feynman_II_11_27",
    ),
    (
        "II.11.28",
        "base",
        "theta",
        "1+n*alpha/(1-(n*alpha/3))",
        (("n", 0.0, 1.0), ("alpha", 0.0, 1.0)),
        "feynman_II_11_28",
    ),
    (
        "II.11.3",
        "base",
        "x",
        "q*Ef/(m*(omega_0**2-omega**2))",
        (
            ("q", 1.0, 3.0),
            ("Ef", 1.0, 3.0),
            ("m", 1.0, 3.0),
            ("omega_0", 3.0, 5.0),
            ("omega", 1.0, 2.0),
        ),
        "feynman_II_11_3",
    ),
    (
        "II.13.17",
        "base",
        "B",
        "1/(4*pi*epsilon*c**2)*2*I/r",
        (("epsilon", 1.0, 5.0), ("c", 1.0, 5.0), ("I", 1.0, 5.0), ("r", 1.0, 5.0)),
        "feynman_II_13_17",
    ),
    (
        "II.13.23",
        "base",
        "rho_c",
        "rho_c_0/sqrt(1-v**2/c**2)",
        (("rho_c_0", 1.0, 5.0), ("v", 1.0, 2.0), ("c", 3.0, 10.0)),
        "feynman_II_13_23",
    ),
    (
        "II.13.34",
        "base",
        "j",
        "rho_c_0*v/sqrt(1-v**2/c**2)",
        (("rho_c_0", 1.0, 5.0), ("v", 1.0, 2.0), ("c", 3.0, 10.0)),
        "feynman_II_13_34",
    ),
    (
        "II.15.4",
        "base",
        "E_n",
        "-mom*B*cos(theta)",
        (("mom", 1.0, 5.0), ("B", 1.0, 5.0), ("theta", 1.0, 5.0)),
        "feynman_II_15_4",
    ),
    (
        "II.15.5",
        "base",
        "E_n",
        "-p_d*Ef*cos(theta)",
        (("p_d", 1.0, 5.0), ("Ef", 1.0, 5.0), ("theta", 1.0, 5.0)),
        "feynman_II_15_5",
    ),
    (
        "II.2.42",
        "base",
        "Pwr",
        "kappa*(T2-T1)*A/d",
        (("kappa", 1.0, 5.0), ("T1", 1.0, 5.0), ("T2", 1.0, 5.0), ("A", 1.0, 5.0), ("d", 1.0, 5.0)),
        "feynman_II_2_42",
    ),
    (
        "II.21.32",
        "base",
        "Volt",
        "q/(4*pi*epsilon*r*(1-v/c))",
        (
            ("q", 1.0, 5.0),
            ("epsilon", 1.0, 5.0),
            ("r", 1.0, 5.0),
            ("v", 1.0, 2.0),
            ("c", 3.0, 10.0),
        ),
        "feynman_II_21_32",
    ),
    (
        "II.24.17",
        "base",
        "k",
        "sqrt(omega**2/c**2-pi**2/d**2)",
        (("omega", 4.0, 6.0), ("c", 1.0, 2.0), ("d", 2.0, 4.0)),
        "feynman_II_24_17",
    ),
    (
        "II.27.16",
        "base",
        "flux",
        "epsilon*c*Ef**2",
        (("epsilon", 1.0, 5.0), ("c", 1.0, 5.0), ("Ef", 1.0, 5.0)),
        "feynman_II_27_16",
    ),
    (
        "II.27.18",
        "base",
        "E_den",
        "epsilon*Ef**2",
        (("epsilon", 1.0, 5.0), ("Ef", 1.0, 5.0)),
        "feynman_II_27_18",
    ),
    (
        "II.3.24",
        "base",
        "flux",
        "Pwr/(4*pi*r**2)",
        (("Pwr", 1.0, 5.0), ("r", 1.0, 5.0)),
        "feynman_II_3_24",
    ),
    (
        "II.34.11",
        "base",
        "omega",
        "g_*q*B/(2*m)",
        (("g_", 1.0, 5.0), ("q", 1.0, 5.0), ("B", 1.0, 5.0), ("m", 1.0, 5.0)),
        "feynman_II_34_11",
    ),
    (
        "II.34.2",
        "base",
        "mom",
        "q*v*r/2",
        (("q", 1.0, 5.0), ("v", 1.0, 5.0), ("r", 1.0, 5.0)),
        "feynman_II_34_2",
    ),
    (
        "II.34.29a",
        "base",
        "mom",
        "q*h/(4*pi*m)",
        (("q", 1.0, 5.0), ("h", 1.0, 5.0), ("m", 1.0, 5.0)),
        "feynman_II_34_29a",
    ),
    (
        "II.34.29b",
        "base",
        "E_n",
        "g_*mom*B*Jz/(h/(2*pi))",
        (("g_", 1.0, 5.0), ("h", 1.0, 5.0), ("Jz", 1.0, 5.0), ("mom", 1.0, 5.0), ("B", 1.0, 5.0)),
        "feynman_II_34_29b",
    ),
    (
        "II.34.2a",
        "base",
        "I",
        "q*v/(2*pi*r)",
        (("q", 1.0, 5.0), ("v", 1.0, 5.0), ("r", 1.0, 5.0)),
        "feynman_II_34_2a",
    ),
    (
        "II.35.18",
        "base",
        "n",
        "n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))",
        (("n_0", 1.0, 3.0), ("kb", 1.0, 3.0), ("T", 1.0, 3.0), ("mom", 1.0, 3.0), ("B", 1.0, 3.0)),
        "feynman_II_35_18",
    ),
    (
        "II.35.21",
        "base",
        "M",
        "n_rho*mom*tanh(mom*B/(kb*T))",
        (
            ("n_rho", 1.0, 5.0),
            ("mom", 1.0, 5.0),
            ("B", 1.0, 5.0),
            ("kb", 1.0, 5.0),
            ("T", 1.0, 5.0),
        ),
        "feynman_II_35_21",
    ),
    (
        "II.36.38",
        "base",
        "f",
        "mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M",
        (
            ("mom", 1.0, 3.0),
            ("H", 1.0, 3.0),
            ("kb", 1.0, 3.0),
            ("T", 1.0, 3.0),
            ("alpha", 1.0, 3.0),
            ("epsilon", 1.0, 3.0),
            ("c", 1.0, 3.0),
            ("M", 1.0, 3.0),
        ),
        "feynman_II_36_38",
    ),
    (
        "II.37.1",
        "base",
        "E_n",
        "mom*(1+chi)*B",
        (("mom", 1.0, 5.0), ("B", 1.0, 5.0), ("chi", 1.0, 5.0)),
        "feynman_II_37_1",
    ),
    (
        "II.38.14",
        "base",
        "mu_S",
        "Y/(2*(1+sigma))",
        (("Y", 1.0, 5.0), ("sigma", 1.0, 5.0)),
        "feynman_II_38_14",
    ),
    (
        "II.38.3",
        "base",
        "F",
        "Y*A*x/d",
        (("Y", 1.0, 5.0), ("A", 1.0, 5.0), ("d", 1.0, 5.0), ("x", 1.0, 5.0)),
        "feynman_II_38_3",
    ),
    (
        "II.4.23",
        "base",
        "Volt",
        "q/(4*pi*epsilon*r)",
        (("q", 1.0, 5.0), ("epsilon", 1.0, 5.0), ("r", 1.0, 5.0)),
        "feynman_II_4_23",
    ),
    (
        "II.6.11",
        "base",
        "Volt",
        "1/(4*pi*epsilon)*p_d*cos(theta)/r**2",
        (("epsilon", 1.0, 3.0), ("p_d", 1.0, 3.0), ("theta", 1.0, 3.0), ("r", 1.0, 3.0)),
        "feynman_II_6_11",
    ),
    (
        "II.6.15a",
        "base",
        "Ef",
        "p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2)",
        (
            ("epsilon", 1.0, 3.0),
            ("p_d", 1.0, 3.0),
            ("r", 1.0, 3.0),
            ("x", 1.0, 3.0),
            ("y", 1.0, 3.0),
            ("z", 1.0, 3.0),
        ),
        "feynman_II_6_15a",
    ),
    (
        "II.6.15b",
        "base",
        "Ef",
        "p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3",
        (("epsilon", 1.0, 3.0), ("p_d", 1.0, 3.0), ("theta", 1.0, 3.0), ("r", 1.0, 3.0)),
        "feynman_II_6_15b",
    ),
    (
        "II.8.31",
        "base",
        "E_den",
        "epsilon*Ef**2/2",
        (("epsilon", 1.0, 5.0), ("Ef", 1.0, 5.0)),
        "feynman_II_8_31",
    ),
    (
        "II.8.7",
        "base",
        "E_n",
        "3/5*q**2/(4*pi*epsilon*d)",
        (("q", 1.0, 5.0), ("epsilon", 1.0, 5.0), ("d", 1.0, 5.0)),
        "feynman_II_8_7",
    ),
    (
        "III.10.19",
        "base",
        "E_n",
        "mom*sqrt(Bx**2+By**2+Bz**2)",
        (("mom", 1.0, 5.0), ("Bx", 1.0, 5.0), ("By", 1.0, 5.0), ("Bz", 1.0, 5.0)),
        "feynman_III_10_19",
    ),
    (
        "III.12.43",
        "base",
        "L",
        "n*(h/(2*pi))",
        (("n", 1.0, 5.0), ("h", 1.0, 5.0)),
        "feynman_III_12_43",
    ),
    (
        "III.13.18",
        "base",
        "v",
        "2*E_n*d**2*k/(h/(2*pi))",
        (("E_n", 1.0, 5.0), ("d", 1.0, 5.0), ("k", 1.0, 5.0), ("h", 1.0, 5.0)),
        "feynman_III_13_18",
    ),
    (
        "III.14.14",
        "base",
        "I",
        "I_0*(exp(q*Volt/(kb*T))-1)",
        (("I_0", 1.0, 5.0), ("q", 1.0, 2.0), ("Volt", 1.0, 2.0), ("kb", 1.0, 2.0), ("T", 1.0, 2.0)),
        "feynman_III_14_14",
    ),
    (
        "III.15.12",
        "base",
        "E_n",
        "2*U*(1-cos(k*d))",
        (("U", 1.0, 5.0), ("k", 1.0, 5.0), ("d", 1.0, 5.0)),
        "feynman_III_15_12",
    ),
    (
        "III.15.14",
        "base",
        "m",
        "(h/(2*pi))**2/(2*E_n*d**2)",
        (("h", 1.0, 5.0), ("E_n", 1.0, 5.0), ("d", 1.0, 5.0)),
        "feynman_III_15_14",
    ),
    (
        "III.15.27",
        "base",
        "k",
        "2*pi*alpha/(n*d)",
        (("alpha", 1.0, 5.0), ("n", 1.0, 5.0), ("d", 1.0, 5.0)),
        "feynman_III_15_27",
    ),
    (
        "III.17.37",
        "base",
        "f",
        "beta*(1+alpha*cos(theta))",
        (("beta", 1.0, 5.0), ("alpha", 1.0, 5.0), ("theta", 1.0, 5.0)),
        "feynman_III_17_37",
    ),
    (
        "III.19.51",
        "base",
        "E_n",
        "-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)",
        (("m", 1.0, 5.0), ("q", 1.0, 5.0), ("h", 1.0, 5.0), ("n", 1.0, 5.0), ("epsilon", 1.0, 5.0)),
        "feynman_III_19_51",
    ),
    (
        "III.21.20",
        "base",
        "j",
        "-rho_c_0*q*A_vec/m",
        (("rho_c_0", 1.0, 5.0), ("q", 1.0, 5.0), ("A_vec", 1.0, 5.0), ("m", 1.0, 5.0)),
        "feynman_III_21_20",
    ),
    (
        "III.4.32",
        "base",
        "n",
        "1/(exp((h/(2*pi))*omega/(kb*T))-1)",
        (("h", 1.0, 5.0), ("omega", 1.0, 5.0), ("kb", 1.0, 5.0), ("T", 1.0, 5.0)),
        "feynman_III_4_32",
    ),
    (
        "III.4.33",
        "base",
        "E_n",
        "(h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)",
        (("h", 1.0, 5.0), ("omega", 1.0, 5.0), ("kb", 1.0, 5.0), ("T", 1.0, 5.0)),
        "feynman_III_4_33",
    ),
    (
        "III.7.38",
        "base",
        "omega",
        "2*mom*B/(h/(2*pi))",
        (("mom", 1.0, 5.0), ("B", 1.0, 5.0), ("h", 1.0, 5.0)),
        "feynman_III_7_38",
    ),
    (
        "III.8.54",
        "base",
        "prob",
        "sin(E_n*t/(h/(2*pi)))**2",
        (("E_n", 1.0, 2.0), ("t", 1.0, 2.0), ("h", 1.0, 4.0)),
        "feynman_III_8_54",
    ),
    (
        "III.9.52",
        "base",
        "prob",
        "(p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2",
        (
            ("p_d", 1.0, 3.0),
            ("Ef", 1.0, 3.0),
            ("t", 1.0, 3.0),
            ("h", 1.0, 3.0),
            ("omega", 1.0, 5.0),
            ("omega_0", 1.0, 5.0),
        ),
        "feynman_III_9_52",
    ),
    (
        "test_1",
        "bonus",
        "A",
        "(Z_1*Z_2*alpha*hbar*c/(4*E_n*sin(theta/2)**2))**2",
        (
            ("Z_1", 1.0, 2.0),
            ("Z_2", 1.0, 2.0),
            ("alpha", 1.0, 2.0),
            ("hbar", 1.0, 2.0),
            ("c", 1.0, 2.0),
            ("E_n", 1.0, 3.0),
            ("theta", 1.0, 3.0),
        ),
        "feynman_test_1",
    ),
    (
        "test_10",
        "bonus",
        "theta1",
        "arccos((cos(theta2)-v/c)/(1-v/c*cos(theta2)))",
        (("c", 4.0, 6.0), ("v", 1.0, 3.0), ("theta2", 1.0, 3.0)),
        "feynman_test_10",
    ),
    (
        "test_11",
        "bonus",
        "I",
        "I_0*(sin(alpha/2)*sin(n*delta/2)/(alpha/2*sin(delta/2)))**2",
        (("I_0", 1.0, 3.0), ("alpha", 1.0, 3.0), ("delta", 1.0, 3.0), ("n", 1.0, 2.0)),
        "feynman_test_11",
    ),
    (
        "test_12",
        "bonus",
        "F",
        "q/(4*pi*epsilon*y**2)*(4*pi*epsilon*Volt*d-q*d*y**3/(y**2-d**2)**2)",
        (
            ("q", 1.0, 5.0),
            ("y", 1.0, 3.0),
            ("Volt", 1.0, 5.0),
            ("d", 4.0, 6.0),
            ("epsilon", 1.0, 5.0),
        ),
        "feynman_test_12",
    ),
    (
        "test_13",
        "bonus",
        "Volt",
        "1/(4*pi*epsilon)*q/sqrt(r**2+d**2-2*r*d*cos(alpha))",
        (
            ("q", 1.0, 5.0),
            ("r", 1.0, 3.0),
            ("d", 4.0, 6.0),
            ("alpha", 0.0, 6.0),
            ("epsilon", 1.0, 5.0),
        ),
        "feynman_test_13",
    ),
    (
        "test_14",
        "bonus",
        "Volt",
        "Ef*cos(theta)*(-r+d**3/r**2*(alpha-1)/(alpha+2))",
        (
            ("Ef", 1.0, 5.0),
            ("theta", 0.0, 6.0),
            ("r", 1.0, 5.0),
            ("d", 1.0, 5.0),
            ("alpha", 1.0, 5.0),
        ),
        "feynman_test_14",
    ),
    (
        "test_15",
        "bonus",
        "omega_0",
        "sqrt(1-v**2/c**2)*omega/(1+v/c*cos(theta))",
        (("c", 5.0, 20.0), ("v", 1.0, 3.0), ("omega", 1.0, 5.0), ("theta", 0.0, 6.0)),
        "feynman_test_15",
    ),
    (
        "test_16",
        "bonus",
        "E_n",
        "sqrt((p-q*A_vec)**2*c**2+m**2*c**4)+q*Volt",
        (
            ("m", 1.0, 5.0),
            ("c", 1.0, 5.0),
            ("p", 1.0, 5.0),
            ("q", 1.0, 5.0),
            ("A_vec", 1.0, 5.0),
            ("Volt", 1.0, 5.0),
        ),
        "feynman_test_16",
    ),
    (
        "test_17",
        "bonus",
        "E_n",
        "1/(2*m)*(p**2+m**2*omega**2*x**2*(1+alpha*x/y))",
        (
            ("m", 1.0, 5.0),
            ("omega", 1.0, 5.0),
            ("p", 1.0, 5.0),
            ("y", 1.0, 5.0),
            ("x", 1.0, 5.0),
            ("alpha", 1.0, 5.0),
        ),
        "feynman_test_17",
    ),
    (
        "test_18",
        "bonus",
        "rho_0",
        "3/(8*pi*G)*(c**2*k_f/r**2+H_G**2)",
        (("G", 1.0, 5.0), ("k_f", 1.0, 5.0), ("r", 1.0, 5.0), ("H_G", 1.0, 5.0), ("c", 1.0, 5.0)),
        "feynman_test_18",
    ),
    (
        "test_19",
        "bonus",
        "pr",
        "-1/(8*pi*G)*(c**4*k_f/r**2+H_G**2*c**2*(1-2*alpha))",
        (
            ("G", 1.0, 5.0),
            ("k_f", 1.0, 5.0),
            ("r", 1.0, 5.0),
            ("H_G", 1.0, 5.0),
            ("alpha", 1.0, 5.0),
            ("c", 1.0, 5.0),
        ),
        "feynman_test_19",
    ),
    (
        "test_2",
        "bonus",
        "k",
        "m*k_G/L**2*(1+sqrt(1+2*E_n*L**2/(m*k_G**2))*cos(theta1-theta2))",
        (
            ("m", 1.0, 3.0),
            ("k_G", 1.0, 3.0),
            ("L", 1.0, 3.0),
            ("E_n", 1.0, 3.0),
            ("theta1", 0.0, 6.0),
            ("theta2", 0.0, 6.0),
        ),
        "feynman_test_2",
    ),
    (
        "test_20",
        "bonus",
        "A",
        "1/(4*pi)*alpha**2*h**2/(m**2*c**2)*(omega_0/omega)**2*(omega_0/omega+omega/omega_0-sin(beta)**2)",
        (
            ("omega", 1.0, 5.0),
            ("omega_0", 1.0, 5.0),
            ("alpha", 1.0, 5.0),
            ("h", 1.0, 5.0),
            ("m", 1.0, 5.0),
            ("c", 1.0, 5.0),
            ("beta", 0.0, 6.0),
        ),
        "feynman_test_20",
    ),
    (
        "test_3",
        "bonus",
        "r",
        "d*(1-alpha**2)/(1+alpha*cos(theta1-theta2))",
        (("d", 1.0, 3.0), ("alpha", 2.0, 4.0), ("theta1", 4.0, 5.0), ("theta2", 4.0, 5.0)),
        "feynman_test_3",
    ),
    (
        "test_4",
        "bonus",
        "v",
        "sqrt(2/m*(E_n-U-L**2/(2*m*r**2)))",
        (("m", 1.0, 3.0), ("E_n", 8.0, 12.0), ("U", 1.0, 3.0), ("L", 1.0, 3.0), ("r", 1.0, 3.0)),
        "feynman_test_4",
    ),
    (
        "test_5",
        "bonus",
        "t",
        "2*pi*d**(3/2)/sqrt(G*(m1+m2))",
        (("d", 1.0, 3.0), ("G", 1.0, 3.0), ("m1", 1.0, 3.0), ("m2", 1.0, 3.0)),
        "feynman_test_5",
    ),
    (
        "test_6",
        "bonus",
        "alpha",
        "sqrt(1+2*epsilon**2*E_n*L**2/(m*(Z_1*Z_2*q**2)**2))",
        (
            ("epsilon", 1.0, 3.0),
            ("L", 1.0, 3.0),
            ("m", 1.0, 3.0),
            ("Z_1", 1.0, 3.0),
            ("Z_2", 1.0, 3.0),
            ("q", 1.0, 3.0),
            ("E_n", 1.0, 3.0),
        ),
        "feynman_test_6",
    ),
    (
        "test_7",
        "bonus",
        "H_G",
        "sqrt(8*pi*G*rho/3-alpha*c**2/d**2)",
        (("G", 1.0, 3.0), ("rho", 1.0, 3.0), ("alpha", 1.0, 2.0), ("c", 1.0, 2.0), ("d", 1.0, 3.0)),
        "feynman_test_7",
    ),
    (
        "test_8",
        "bonus",
        "K",
        "E_n/(1+E_n/(m*c**2)*(1-cos(theta)))",
        (("E_n", 1.0, 3.0), ("m", 1.0, 3.0), ("c", 1.0, 3.0), ("theta", 1.0, 3.0)),
        "feynman_test_8",
    ),
    (
        "test_9",
        "bonus",
        "Pwr",
        "-32/5*G**4/c**5*(m1*m2)**2*(m1+m2)/r**5",
        (("G", 1.0, 2.0), ("c", 1.0, 2.0), ("m1", 1.0, 5.0), ("m2", 1.0, 5.0), ("r", 1.0, 2.0)),
        "feynman_test_9",
    ),
)


def _expand(row: _RawRow) -> EquationRecord:
    """Expand one literal row into the public record schema.

    Args:
        row: A ``_RAW`` tuple.

    Returns:
        Record with keys ``id``, ``source``, ``output``, ``formula``,
        ``num_variables``, ``variables``, ``pmlb_id``.
    """
    eq_id, source, output, formula, raw_vars, pmlb_id = row
    variables: list[VariableSpec] = [
        {"name": name, "low": low, "high": high} for name, low, high in raw_vars
    ]
    return {
        "id": eq_id,
        "source": source,
        "output": output,
        "formula": formula,
        "num_variables": len(variables),
        "variables": variables,
        "pmlb_id": pmlb_id,
    }


#: The full AI Feynman database: 100 base equations plus 20 bonus equations,
#: sorted by ``source`` then ``id``.
AIFEYNMAN_120: list[EquationRecord] = [_expand(row) for row in _RAW]

# --------------------------------------------------------------------------------------
# Overlap with the equations the IsalSR suite already uses
# --------------------------------------------------------------------------------------
# Membership is keyed by the identifier **as the IsalSR benchmark suite writes it**, not
# by the identifier this catalogue assigns. Five of these 24 are known to be mislabelled
# or mistranscribed relative to the AI Feynman database (filed separately against T09).
# Keying by the suite's spelling is deliberate: the eligible extension pool must never
# be able to draw an identifier the suite already occupies, whatever the resolution of
# that labelling question turns out to be.

IN_SUITE_IDS: frozenset[str] = frozenset(
    {
        "I.6.20a",
        "I.12.1",
        "I.14.3",
        "I.25.13",
        "I.34.27",
        "I.39.10",
        "I.12.4",
        "II.3.24",
        "I.10.7",
        "I.48.20",
        "I.15.10",
        "I.30.3",
        "I.37.4",
        "II.11.27",
        "III.17.37",
        "I.29.16",
        "I.50.26",
        "I.16.6",
        "II.11.28",
        "III.14.14",
        "III.10.19",
        "II.11.3",
        "I.13.12",
        "I.44.4",
    }
)


def get_equation(eq_id: str) -> EquationRecord:
    """Look up one catalogue record by identifier.

    Args:
        eq_id: Catalogue identifier, e.g. ``"I.15.10"`` or ``"test_7"``.

    Returns:
        The matching record.

    Raises:
        KeyError: If no record carries that identifier.
    """
    for record in AIFEYNMAN_120:
        if record["id"] == eq_id:
            return record
    raise KeyError(eq_id)


def classification_table() -> dict[str, Any]:
    """Classify every catalogue equation and summarise the outcome.

    Returns:
        Dictionary with ``n_total``, ``n_representable_syntactic``,
        ``n_representable_semantic``, ``blocked_syntactic`` and ``blocked_semantic``
        (each a list of ``{"id", "blocking_ops", "formula"}`` in catalogue order), and
        ``by_function`` mapping every function symbol observed to the number of
        equations that use it.
    """
    n_syn = 0
    n_sem = 0
    blocked_syn: list[dict[str, Any]] = []
    blocked_sem: list[dict[str, Any]] = []
    by_function: dict[str, int] = {}

    for record in AIFEYNMAN_120:
        names = [variable["name"] for variable in record["variables"]]
        verdict = classify_sigma_sr(record["formula"], names)
        for function_name in verdict.functions_used:
            by_function[function_name] = by_function.get(function_name, 0) + 1
        if verdict.representable_syntactic:
            n_syn += 1
        else:
            blocked_syn.append(
                {
                    "id": record["id"],
                    "blocking_ops": list(verdict.blocking_ops_syntactic),
                    "formula": record["formula"],
                }
            )
        if verdict.representable_semantic:
            n_sem += 1
        else:
            blocked_sem.append(
                {
                    "id": record["id"],
                    "blocking_ops": list(verdict.blocking_ops_semantic),
                    "formula": record["formula"],
                }
            )

    return {
        "n_total": len(AIFEYNMAN_120),
        "n_representable_syntactic": n_syn,
        "n_representable_semantic": n_sem,
        "blocked_syntactic": blocked_syn,
        "blocked_semantic": blocked_sem,
        "by_function": dict(sorted(by_function.items())),
    }


def eligible_extension_pool() -> list[str]:
    """List the catalogue identifiers available for extending the IsalSR suite.

    The filter is outcome-blind by construction: an equation is eligible exactly when
    it is Sigma_SR-representable under the syntactic reading and its identifier is not
    already in :data:`IN_SUITE_IDS`. No notion of expression complexity, arity, sample
    budget or expected difficulty enters this function, so the pool cannot encode any
    preference for equations on which IsalSR is expected to do well.

    Returns:
        Sorted identifiers, in catalogue (``source``, ``id``) order.
    """
    pool: list[str] = []
    for record in AIFEYNMAN_120:
        if record["id"] in IN_SUITE_IDS:
            continue
        names = [variable["name"] for variable in record["variables"]]
        if classify_sigma_sr(record["formula"], names).representable_syntactic:
            pool.append(record["id"])
    return pool
