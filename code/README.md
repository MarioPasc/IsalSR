# isalsr

**Isomorphism-invariant instruction strings for the labeled DAGs of symbolic regression.**

A symbolic-regression expression is a labeled directed acyclic graph (DAG), and a DAG with *k*
internal nodes can be numbered in up to *k*! ways. A solver that stores or compares candidates
by their numbering evaluates the same expression many times. `isalsr` writes every labeled DAG
as a string over a two-tier instruction alphabet and computes a **canonical string**: two
expression DAGs get the same canonical string if and only if they are isomorphic (under the
reachability condition stated in the paper). Deduplicating candidates by that string removes
the redundant evaluations without changing the solver's search.

This package is the reference implementation of

> E. López-Rubio, M. Pascual-González and K. Thurnhofer-Hemsi, "Representation of Directed
> Acyclic Graphs by Sequences of Instructions for Symbolic Regression", *IEEE Transactions on
> Pattern Analysis and Machine Intelligence*, accepted 2026.

## Install

```bash
pip install isalsr
```

The core has **no dependencies**. Two engines compute identical strings: a C++17 engine
(about 19× faster) and a pure-Python one. Wheels ship the C++ engine compiled. A source build
compiles it when a C++17 compiler is available and installs the Python engine alone
otherwise, so the install itself never fails for lack of a toolchain. Check which engine is
active:

```python
>>> import isalsr
>>> isalsr.engine()
'cpp'
```

Optional extras: `isalsr[sympy]` and `isalsr[networkx]` (adapters), `isalsr[eval]` (fitness
metrics and constant optimization), `isalsr[viz]` (figures).

## Quick start

```python
import isalsr

# Two numberings of sin(x1) + cos(x1) give two different instruction strings...
a = isalsr.StringToDAG("VsVcpv+Ppc", num_variables=1).run()
b = isalsr.StringToDAG("VcVspv+Ppc", num_variables=1).run()
assert a.is_isomorphic(b)

# ...and one canonical string.
print(isalsr.fast_canonical_string(a))   # VcVspv+Ppc
print(isalsr.fast_canonical_string(b))   # VcVspv+Ppc

# Strings decode to DAGs that evaluate like the expression they encode.
print(isalsr.evaluate_dag(a, {0: 0.5}))  # sin(0.5) + cos(0.5) = 1.357...
```

To deduplicate a solver's candidates, convert each candidate to a `LabeledDAG`, compute
`fast_canonical_string`, and skip the evaluation when the string has been seen. No
hyperparameter, search operator or termination criterion of the solver changes.

Set `ISALSR_ENGINE=python` to force the Python engine, or `ISALSR_ENGINE=cpp` to fail loudly
if the C++ engine is missing.

## Reproducing the paper

The experiments (the UDFS and Bingo integrations, the 70-problem benchmark suite and the
analysis pipeline) are in the [GitHub repository](https://github.com/MarioPasc/IsalSR) and
its Code Ocean capsule, not in this package. See `REPRODUCING.md` there.

## License

MIT.
