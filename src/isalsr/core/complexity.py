"""Cheap structural descriptors of a labeled DAG, and a constant-memory accumulator.

The word "complexity" has no single referent for an expression DAG, so this
module replaces it with an explicit descriptor vector (:class:`DagComplexity`)
whose components are each a documented proxy for one facet of structural
difficulty: raw size, compositional depth, subexpression reuse, transcendental
content and operator-mix heterogeneity.

Every descriptor is obtained from a single ``O(|V| + |E|)`` sweep, so the whole
vector costs one topological sort plus one linear pass.
:class:`ComplexityAccumulator` folds a stream of descriptor vectors into exact
sums, exact integer histograms and extrema at constant memory, which is what
makes it usable inside a search loop that visits many millions of candidates.

Anchors for the individual descriptors are given on :class:`DagComplexity`.

This module is part of ``isalsr.core`` and therefore imports nothing outside the
standard library.
"""

from __future__ import annotations

from math import log2
from typing import Final, NamedTuple

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

__all__ = [
    "DESCRIPTOR_FIELDS",
    "HEADLINE_FIELDS",
    "NONLINEAR_OPS",
    "ComplexityAccumulator",
    "DagComplexity",
    "describe_dag",
    "describe_dag_with_labels",
]


#: Dense index for each ``NodeType``, in declaration order. ``NodeType`` is a
#: string-valued ``Enum``, so its members cannot index a list directly.
_LABEL_INDEX: Final[dict[NodeType, int]] = {
    node_type: index for index, node_type in enumerate(NodeType)
}

#: ``NodeType`` members in the order of :data:`_LABEL_INDEX`.
_LABEL_ORDER: Final[tuple[NodeType, ...]] = tuple(NodeType)

#: Number of ``NodeType`` members, used to size the label histogram.
_N_NODE_TYPES: Final[int] = len(_LABEL_ORDER)

#: Operators counted by :attr:`DagComplexity.n_nonlinear`.
#:
#: These are the labels that published symbolic-regression complexity measures
#: weight above the polynomial ring ``{ADD, SUB, MUL, NEG}``: Vladislavleva et
#: al. (IEEE TEC 13(2), 2009) assign them a higher order of nonlinearity, and
#: Kommenda et al. (EUROCAST 2015) give them the multiplicative or exponential
#: branch of the recursive-complexity recursion. The count is a deliberately
#: cheap surrogate -- the order of nonlinearity itself needs a Chebyshev
#: approximation per node and has no place in a hot loop.
#:
#: ``DIV`` is present for legacy safety only. Under the decomposed alphabet (T16)
#: it must never be emitted, and SP-4 checks that it is not.
NONLINEAR_OPS: Final[frozenset[NodeType]] = frozenset(
    {
        NodeType.SIN,
        NodeType.COS,
        NodeType.EXP,
        NodeType.LOG,
        NodeType.SQRT,
        NodeType.POW,
        NodeType.ABS,
        NodeType.INV,
        NodeType.DIV,
    }
)

#: Tolerance below which a computed entropy is clamped to exactly zero.
_EPS: Final[float] = 1e-12


class DagComplexity(NamedTuple):
    """Structural descriptors of one labeled DAG.

    All components except :attr:`op_label_entropy` are non-negative integers.
    The field order is fixed and is the order used by
    :data:`DESCRIPTOR_FIELDS`, by :class:`ComplexityAccumulator` and by the
    on-disk sidecar, so it must not be permuted.

    Attributes:
        n_nodes: ``|V|``. Raw size.
        n_edges: ``|E|``. Raw size.
        n_var: Number of ``VAR`` nodes.
        n_internal: ``|V| - n_var``. This is **the paper's** ``k``, matching the
            convention of ``isalsr.precomputed.cache_entry.CacheEntry``.
        n_const: Number of ``CONST`` nodes, i.e. the free-parameter count that
            constant optimisation has to fit.
        n_op: ``n_internal - n_const``. Operator count; the "model size" reported
            by SRBench (La Cava et al., NeurIPS Datasets and Benchmarks 2021).
        depth: Longest directed path, in edges, from a source to any node. The
            standard proxy for compositional nesting (Koza, 1992).
        max_in_degree: Largest in-degree. Saturation of variadic operators.
        max_out_degree: Largest out-degree. Heaviest single reuse site.
        n_shared: Number of nodes with out-degree at least two, i.e. the number
            of genuinely shared subexpressions. This is the quantity that
            separates a DAG representation from a tree, and is what the
            graph-based symbolic-regression literature (GraphSR, GraphDSR)
            claims as its advantage.
        sharing_surplus: ``sum(max(0, outdeg(v) - 1))``. The number of edges the
            DAG saves relative to unfolding it into a tree.
        n_nonlinear: Number of nodes labelled with a member of
            :data:`NONLINEAR_OPS`.
        n_var_used: Number of ``VAR`` nodes with out-degree at least one, i.e.
            variables the expression actually reads. With :attr:`n_var` this is
            the feature-selection signal.
        n_distinct_op_labels: Number of distinct operator labels present.
        op_label_entropy: Shannon entropy in bits of the operator-label
            histogram, excluding ``VAR`` and ``CONST``. Zero when the support has
            fewer than two members. Measures operator-mix heterogeneity
            (Shannon, 1948).
    """

    n_nodes: int
    n_edges: int
    n_var: int
    n_internal: int
    n_const: int
    n_op: int
    depth: int
    max_in_degree: int
    max_out_degree: int
    n_shared: int
    sharing_surplus: int
    n_nonlinear: int
    n_var_used: int
    n_distinct_op_labels: int
    op_label_entropy: float


#: Descriptor names in :class:`DagComplexity` order.
DESCRIPTOR_FIELDS: Final[tuple[str, ...]] = DagComplexity._fields

#: Descriptors that get a full exact histogram rather than only moments. Chosen
#: because these four carry the paper's claims, and because distribution-level
#: tests (Kolmogorov-Smirnov, Mann-Whitney) need the empirical distribution
#: rather than a mean.
HEADLINE_FIELDS: Final[tuple[str, ...]] = ("n_internal", "depth", "n_edges", "n_op")

#: Inclusive upper bin index per histogrammed descriptor. A value above the cap
#: lands in a single overflow bin whose count is reported, so a reader can tell
#: whether the quantiles are exact.
_HIST_CAPS: Final[dict[str, int]] = {
    "n_internal": 128,
    "depth": 64,
    "n_edges": 160,
    "n_op": 128,
}

#: ``DagComplexity`` for an empty DAG. Hoisted so the empty case allocates nothing.
_EMPTY: Final[DagComplexity] = DagComplexity(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)


def describe_dag(dag: LabeledDAG) -> DagComplexity:
    """Compute the structural descriptor vector of *dag*.

    Args:
        dag: The labeled DAG to describe. Must be acyclic, which
            :class:`LabeledDAG` guarantees for graphs built through
            :meth:`LabeledDAG.add_edge`.

    Returns:
        The descriptor vector.

    Raises:
        ValueError: If *dag* contains a cycle, propagated from
            :meth:`LabeledDAG.topological_sort`.
    """
    return describe_dag_with_labels(dag)[0]


def describe_dag_with_labels(dag: LabeledDAG) -> tuple[DagComplexity, list[int]]:
    """Compute the descriptor vector of *dag* together with its label histogram.

    One :meth:`LabeledDAG.topological_sort` (Kahn, ``O(|V| + |E|)``) followed by
    one linear pass that folds the longest-path dynamic program, the degree
    extrema and the label histogram together. The unchecked and raw accessors are
    used throughout: they return the live adjacency set with no bounds check and
    no ``frozenset`` copy.

    Args:
        dag: The labeled DAG to describe.

    Returns:
        A pair ``(descriptor, label_counts)`` where *label_counts* is indexed by
        the dense ``NodeType`` index used throughout this module.

    Raises:
        ValueError: If *dag* contains a cycle.
    """
    n_nodes = dag.node_count
    label_counts = [0] * _N_NODE_TYPES

    if n_nodes == 0:
        return _EMPTY, label_counts

    label_index = _LABEL_INDEX
    nonlinear = NONLINEAR_OPS
    depth = [0] * n_nodes

    max_in = 0
    max_out = 0
    n_shared = 0
    sharing_surplus = 0
    n_nonlinear = 0
    n_var = 0
    n_const = 0
    n_var_used = 0
    max_depth = 0

    # Topological order guarantees every predecessor of `node` has its final
    # depth before `node` is visited, so the longest-path DP needs no second
    # pass and no memoisation beyond the array.
    for node in dag.topological_sort():
        label = dag.node_label_unchecked(node)
        preds = dag.in_neighbors_raw(node)
        succs = dag.out_neighbors_raw(node)

        in_deg = len(preds)
        out_deg = len(succs)

        label_counts[label_index[label]] += 1

        if in_deg > max_in:
            max_in = in_deg
        if out_deg > max_out:
            max_out = out_deg
        if out_deg > 1:
            n_shared += 1
            sharing_surplus += out_deg - 1

        if label is NodeType.VAR:
            n_var += 1
            if out_deg:
                n_var_used += 1
        elif label is NodeType.CONST:
            n_const += 1
        elif label in nonlinear:
            n_nonlinear += 1

        if in_deg:
            best = 0
            for pred in preds:
                pred_depth = depth[pred]
                if pred_depth > best:
                    best = pred_depth
            node_depth = best + 1
            depth[node] = node_depth
            if node_depth > max_depth:
                max_depth = node_depth

    n_internal = n_nodes - n_var
    n_op = n_internal - n_const
    entropy, n_distinct = _operator_entropy(label_counts, n_op)

    descriptor = DagComplexity(
        n_nodes=n_nodes,
        n_edges=dag.edge_count,
        n_var=n_var,
        n_internal=n_internal,
        n_const=n_const,
        n_op=n_op,
        depth=max_depth,
        max_in_degree=max_in,
        max_out_degree=max_out,
        n_shared=n_shared,
        sharing_surplus=sharing_surplus,
        n_nonlinear=n_nonlinear,
        n_var_used=n_var_used,
        n_distinct_op_labels=n_distinct,
        op_label_entropy=entropy,
    )
    return descriptor, label_counts


def _operator_entropy(label_counts: list[int], n_op: int) -> tuple[float, int]:
    """Return the Shannon entropy in bits of the operator histogram, and its support.

    ``VAR`` and ``CONST`` are excluded: the quantity of interest is how varied
    the *operator* mix is, and leaves would otherwise dominate the distribution
    for small expressions.

    Args:
        label_counts: Counts indexed by the dense ``NodeType`` index.
        n_op: Total number of operator nodes, i.e. the sum over the operator
            entries of *label_counts*.

    Returns:
        A pair ``(entropy_bits, n_distinct_operator_labels)``. Entropy is
        ``0.0`` when the support has fewer than two members, which is the correct
        value rather than a sentinel.
    """
    if n_op <= 0:
        return 0.0, 0

    var_idx = _LABEL_INDEX[NodeType.VAR]
    const_idx = _LABEL_INDEX[NodeType.CONST]

    entropy = 0.0
    n_distinct = 0
    inv_total = 1.0 / n_op
    for idx, count in enumerate(label_counts):
        if count == 0 or idx in (var_idx, const_idx):
            continue
        n_distinct += 1
        p = count * inv_total
        entropy -= p * log2(p)

    # A single-support histogram can land a few ulp below zero.
    if entropy < _EPS:
        entropy = 0.0
    return entropy, n_distinct


class ComplexityAccumulator:
    """Fold a stream of :class:`DagComplexity` vectors at constant memory.

    Retains, for every descriptor, the sum, the sum of squares, the extrema and
    the count; and for each of :data:`HEADLINE_FIELDS` an exact integer histogram
    with a single overflow bin. The integer descriptors are accumulated as Python
    integers, so their moments carry no floating-point drift however long the
    stream is.

    Nothing is retained per candidate, so the footprint is a few kilobytes
    regardless of how many DAGs are folded in.

    Attributes:
        n: Number of descriptor vectors folded in.
        label_counts: Cumulative ``NodeType`` histogram over every node of every
            folded DAG, indexed by the dense ``NodeType`` index.
    """

    __slots__ = ("_hist", "_max", "_min", "_sum", "_sumsq", "label_counts", "n")

    def __init__(self) -> None:
        n_fields = len(DESCRIPTOR_FIELDS)
        self.n: int = 0
        self._sum: list[float] = [0] * n_fields
        self._sumsq: list[float] = [0] * n_fields
        self._min: list[float] = [0] * n_fields
        self._max: list[float] = [0] * n_fields
        self._hist: dict[str, list[int]] = {
            name: [0] * (_HIST_CAPS[name] + 2) for name in HEADLINE_FIELDS
        }
        self.label_counts: list[int] = [0] * _N_NODE_TYPES

    def observe(self, dag: LabeledDAG) -> DagComplexity:
        """Describe *dag* and fold the result in.

        Args:
            dag: The labeled DAG to describe.

        Returns:
            The descriptor vector that was folded in, so a caller that also wants
            it does not pay for a second sweep.
        """
        descriptor, label_counts = describe_dag_with_labels(dag)
        self.update(descriptor, label_counts)
        return descriptor

    def update(self, desc: DagComplexity, label_counts: list[int] | None = None) -> None:
        """Fold one descriptor vector into the accumulator.

        Args:
            desc: The descriptor vector.
            label_counts: Optional per-``NodeType`` counts for the same DAG, to
                be added to :attr:`label_counts`. Passing ``None`` skips the
                label histogram for this DAG.
        """
        sums = self._sum
        sumsq = self._sumsq
        mins = self._min
        maxs = self._max
        first = self.n == 0

        for i, value in enumerate(desc):
            sums[i] += value
            sumsq[i] += value * value
            if first:
                mins[i] = value
                maxs[i] = value
            elif value < mins[i]:
                mins[i] = value
            elif value > maxs[i]:
                maxs[i] = value

        hist = self._hist
        for name in HEADLINE_FIELDS:
            bins = hist[name]
            value = getattr(desc, name)
            cap = len(bins) - 2
            bins[value if 0 <= value <= cap else cap + 1] += 1

        if label_counts is not None:
            own = self.label_counts
            for i, count in enumerate(label_counts):
                own[i] += count

        self.n += 1

    def mean(self, field: str) -> float:
        """Return the mean of *field*, or ``nan`` if nothing was folded in."""
        if self.n == 0:
            return float("nan")
        return self._sum[DESCRIPTOR_FIELDS.index(field)] / self.n

    def std(self, field: str) -> float:
        """Return the population standard deviation of *field*.

        Computed from the sum and the sum of squares. For the integer descriptors
        both are exact Python integers, so the only rounding is the final
        division and square root.

        Args:
            field: Descriptor name.

        Returns:
            The standard deviation, or ``nan`` when fewer than two vectors were
            folded in.
        """
        if self.n < 2:
            return float("nan")
        i = DESCRIPTOR_FIELDS.index(field)
        mean = self._sum[i] / self.n
        var = self._sumsq[i] / self.n - mean * mean
        return (var if var > 0.0 else 0.0) ** 0.5

    def extremum(self, field: str, *, largest: bool) -> float:
        """Return the largest or smallest observed value of *field*.

        Args:
            field: Descriptor name.
            largest: ``True`` for the maximum, ``False`` for the minimum.

        Returns:
            The extremum, or ``nan`` if nothing was folded in.
        """
        if self.n == 0:
            return float("nan")
        i = DESCRIPTOR_FIELDS.index(field)
        return float(self._max[i] if largest else self._min[i])

    def quantile(self, field: str, q: float) -> float:
        """Return the *q*-quantile of a histogrammed descriptor.

        Uses the lower-value convention: the smallest bin whose cumulative count
        reaches ``q * n``. Exact whenever the overflow bin for *field* is empty;
        when it is not, a value that fell in the overflow bin is reported as the
        cap, which understates it. :meth:`to_dict` reports the overflow count, so
        this is visible rather than silent.

        Args:
            field: One of :data:`HEADLINE_FIELDS`.
            q: Quantile in ``[0, 1]``.

        Returns:
            The quantile, or ``nan`` if nothing was folded in.

        Raises:
            KeyError: If *field* is not histogrammed.
        """
        bins = self._hist[field]
        if self.n == 0:
            return float("nan")
        target = q * self.n
        cumulative = 0
        for value, count in enumerate(bins):
            cumulative += count
            if cumulative >= target:
                return float(value)
        return float(len(bins) - 1)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a JSON-ready mapping.

        Returns:
            A mapping with keys ``n`` (the sample count), ``moments`` (per
            descriptor: ``mean``, ``std``, ``min``, ``max``, ``sum``),
            ``histograms`` (per headline descriptor: ``bins``, ``cap``,
            ``overflow``), ``quantiles`` (per headline descriptor: ``p50``,
            ``p90``, ``p99``) and ``label_counts`` (``NodeType`` name to count).
        """
        moments: dict[str, dict[str, float]] = {}
        for i, name in enumerate(DESCRIPTOR_FIELDS):
            moments[name] = {
                "mean": self.mean(name),
                "std": self.std(name),
                "min": float(self._min[i]) if self.n else float("nan"),
                "max": float(self._max[i]) if self.n else float("nan"),
                "sum": float(self._sum[i]),
            }

        histograms: dict[str, dict[str, object]] = {}
        quantiles: dict[str, dict[str, float]] = {}
        for name in HEADLINE_FIELDS:
            bins = self._hist[name]
            cap = len(bins) - 2
            histograms[name] = {
                "cap": cap,
                "overflow": bins[cap + 1],
                "bins": list(bins[: cap + 1]),
            }
            quantiles[name] = {
                "p50": self.quantile(name, 0.50),
                "p90": self.quantile(name, 0.90),
                "p99": self.quantile(name, 0.99),
            }

        return {
            "n": self.n,
            "moments": moments,
            "histograms": histograms,
            "quantiles": quantiles,
            "label_counts": {
                node_type.name: self.label_counts[index]
                for node_type, index in _LABEL_INDEX.items()
            },
        }
