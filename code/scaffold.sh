#!/usr/bin/env bash
# scaffold.sh -- IsalSR project scaffold generator
# Creates all directories, Python stub files with docstrings, configs, and docs.
# Does NOT install dependencies. Run: pip install -e ".[dev]" after this script.
#
# Usage: bash scaffold.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "=== IsalSR Scaffold Generator ==="
echo "Project root: $PROJECT_ROOT"

# =============================================================================
# 1. Create directory structure
# =============================================================================
echo "[1/7] Creating directories..."

mkdir -p src/isalsr/core/algorithms
mkdir -p src/isalsr/adapters
mkdir -p src/isalsr/evaluation
mkdir -p src/isalsr/search
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/property
mkdir -p experiments/configs
mkdir -p experiments/scripts
mkdir -p benchmarks/datasets
mkdir -p docs/bibliography
mkdir -p docs/references
mkdir -p docs/tasks

# =============================================================================
# 2. Create pyproject.toml
# =============================================================================
echo "[2/7] Writing pyproject.toml..."

cat > pyproject.toml << 'PYPROJECT'
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm>=8.0"]
build-backend = "setuptools.build_meta"

[project]
name = "isalsr"
version = "0.1.0"
description = "Instruction Set and Language for Symbolic Regression"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [
    {name = "Ezequiel Lopez-Rubio", email = "ezeqlr@lcc.uma.es"},
    {name = "Mario Pascual Gonzalez", email = "mpascual@uma.es"},
]
# Core has zero dependencies
dependencies = []

[project.optional-dependencies]
networkx = ["networkx>=3.0"]
sympy = ["sympy>=1.12"]
eval = [
    "numpy>=1.24",
    "scipy>=1.10",
]
search = [
    "numpy>=1.24",
]
viz = ["matplotlib>=3.7"]
bench = [
    "isalsr[networkx,sympy,eval,search,viz]",
    "pandas>=2.0",
    "pyyaml>=6.0",
    "scikit-learn>=1.3",
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "hypothesis>=6.0",
    "ruff>=0.4",
    "mypy>=1.0",
]
all = ["isalsr[networkx,sympy,eval,search,viz,bench,dev]"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
markers = [
    "unit: Unit tests (no external deps)",
    "integration: Integration tests (external libs)",
    "property: Property-based tests (hypothesis)",
    "slow: Long-running tests",
]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true

[[tool.mypy.overrides]]
module = [
    "networkx", "networkx.*",
    "sympy", "sympy.*",
    "numpy", "numpy.*",
    "scipy", "scipy.*",
    "torch", "torch.*",
    "matplotlib", "matplotlib.*",
    "pandas", "pandas.*",
    "sklearn", "sklearn.*",
    "yaml", "yaml.*",
]
ignore_missing_imports = true

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "ANN", "B", "SIM"]

[tool.ruff.lint.per-file-ignores]
# E402: imports after pytest.importorskip() -- standard pattern for optional deps
"tests/integration/*.py" = ["E402"]
# N802: test names may use uppercase instruction names (V, C, N, P, W)
"tests/**/*.py" = ["N802"]
# ANN: benchmark/experiment scripts use dynamic typing for flexibility
"benchmarks/**/*.py" = ["E402", "ANN"]
"experiments/**/*.py" = ["E402", "ANN"]
PYPROJECT

# =============================================================================
# 3. Create source files
# =============================================================================
echo "[3/7] Creating source files..."

# --- src/isalsr/__init__.py ---
cat > src/isalsr/__init__.py << 'EOF'
"""IsalSR -- Instruction Set and Language for Symbolic Regression.

Represents symbolic regression expressions as labeled DAGs encoded in
isomorphism-invariant instruction strings. The canonical string representation
collapses O(k!) equivalent expression representations into one, reducing the
search space for symbolic regression by factorial factors.

Authors:
    Ezequiel Lopez-Rubio (ezeqlr@lcc.uma.es)
    Mario Pascual Gonzalez (mpascual@uma.es)

University of Malaga, 2025.
"""

__version__ = "0.1.0"
EOF

# --- src/isalsr/types.py ---
cat > src/isalsr/types.py << 'EOF'
"""Type aliases for IsalSR.

Centralizes all type aliases used across the package. These distinguish
between CDLL internal indices and graph node indices -- conflating them
is the most common source of silent corruption.

Restriction: ZERO external dependencies. Only Python stdlib + typing.
"""
EOF

# --- src/isalsr/errors.py ---
cat > src/isalsr/errors.py << 'EOF'
"""Exception hierarchy for IsalSR.

All IsalSR exceptions inherit from IsalSRError. Provides specific exceptions
for cycle detection, invalid tokens, invalid DAG structure, and evaluation errors.

Restriction: ZERO external dependencies. Only Python stdlib.

Exceptions to implement:
    - IsalSRError: Base exception for all IsalSR errors.
    - CycleDetectedError: Raised when an edge would create a cycle in the DAG.
    - InvalidTokenError: Raised for unrecognized or disallowed instruction tokens.
    - InvalidDAGError: Raised when a DAG violates structural constraints.
    - EvaluationError: Raised during numerical evaluation failures.
"""
EOF

# --- src/isalsr/core/__init__.py ---
cat > src/isalsr/core/__init__.py << 'EOF'
"""IsalSR core module -- ZERO external dependencies.

Re-exports the public core API: LabeledDAG, CDLL, StringToDAG, DAGToString,
NodeType, canonical_string, and dag_evaluator.

Restriction: This module and ALL submodules must have ZERO external dependencies.
Only Python stdlib and typing are allowed.
"""
EOF

# --- src/isalsr/core/cdll.py ---
cat > src/isalsr/core/cdll.py << 'EOF'
"""Array-backed Circular Doubly Linked List (CDLL).

Reused verbatim from IsalGraph. Provides O(1) insert_after, O(1) remove,
O(1) next/prev traversal. Uses a free-list stack for node allocation.

Reference implementation: /home/mpascual/research/code/IsalGraph/src/isalgraph/core/cdll.py

Restriction: ZERO external dependencies. Only Python stdlib.

Key interface:
    - CircularDoublyLinkedList(capacity: int)
    - insert_after(node: int, value: int) -> int
    - remove(node: int) -> None
    - get_value(node: int) -> int
    - next_node(node: int) -> int
    - prev_node(node: int) -> int

Critical invariant: CDLL indices != graph node indices.
    Pointers are CDLL indices; payloads are graph node IDs.
    Use get_value(ptr) to convert CDLL index -> graph node ID.
"""
EOF

# --- src/isalsr/core/node_types.py ---
cat > src/isalsr/core/node_types.py << 'EOF'
"""Node type registry for IsalSR.

Defines the NodeType enum and associated metadata: arity, label character mapping,
and categorization (unary, binary, variadic, leaf). The operation set is configurable
per experiment -- all possible operations are defined here, and experiments select
subsets via YAML configuration.

Restriction: ZERO external dependencies. Only Python stdlib + enum.

To implement:
    - NodeType(Enum): VAR, ADD, MUL, SUB, DIV, SIN, COS, EXP, LOG, SQRT, POW, ABS, CONST
    - LABEL_CHAR_MAP: dict[str, NodeType] -- maps label characters to NodeType
      Labels: {'+': ADD, '*': MUL, '-': SUB, '/': DIV, 's': SIN, 'c': COS,
               'e': EXP, 'l': LOG, 'r': SQRT, '^': POW, 'a': ABS, 'k': CONST}
    - ARITY_MAP: dict[NodeType, int | None] -- None means variable-arity (2+)
      ADD: None, MUL: None, SUB: 2, DIV: 2, POW: 2,
      SIN: 1, COS: 1, EXP: 1, LOG: 1, SQRT: 1, ABS: 1,
      CONST: 0, VAR: 0
    - UNARY_OPS, BINARY_OPS, VARIADIC_OPS, LEAF_TYPES: frozenset[NodeType]
    - VALID_LABEL_CHARS: frozenset[str]
    - OperationSet: class that wraps a frozenset[NodeType] for configurable op sets,
      with validation and label-char filtering.

Mathematical justification for variable-arity ADD/MUL:
    GraphDSR (Liu2025, Neural Networks 187:107405) demonstrates that variable-arity
    operations for sum and product are essential for efficient DAG-based SR,
    enabling natural representation of expressions like x_1 + x_2 + x_3
    without nested binary trees.
"""
EOF

# --- src/isalsr/core/labeled_dag.py ---
cat > src/isalsr/core/labeled_dag.py << 'EOF'
"""LabeledDAG -- Directed Acyclic Graph with node labels and cycle detection.

The central data structure of IsalSR. Extends IsalGraph's SparseGraph to:
1. Directed-only edges (expressions have data flow direction)
2. Node labels (NodeType: VAR, ADD, MUL, SIN, etc.)
3. Per-node metadata (var_index for VAR, const_value for CONST)
4. Dual adjacency lists (in + out) for efficient evaluation and cycle detection
5. Cycle detection on edge insertion (DFS reachability check)

Restriction: ZERO external dependencies. Only Python stdlib.

To implement:
    class LabeledDAG:
        __slots__ = ('_out_adj', '_in_adj', '_labels', '_node_data',
                     '_node_count', '_edge_count', '_max_nodes')

        Methods:
            add_node(label: NodeType, **kwargs) -> int
                Add node with label. kwargs: var_index (int) for VAR, const_value (float) for CONST.
                Returns new node ID (contiguous integers starting from 0).

            add_edge(source: int, target: int) -> bool
                Add directed edge source -> target.
                Returns True if added, False if would create cycle or is duplicate.
                Edge semantics: source provides input to target.

            has_cycle_if_added(source: int, target: int) -> bool
                Check if adding source -> target creates a cycle.
                Implementation: DFS/BFS from target, checking reachability to source.
                Complexity: O(V + E) per check.

            remove_edge(source: int, target: int) -> bool
                Remove edge. Returns True if existed, False otherwise.
                Needed for backtracking in canonical search.

            undo_node() -> None
                Remove the last-added node. For backtracking in canonical search.

            node_label(node: int) -> NodeType
            node_data(node: int) -> dict[str, Any]
            in_neighbors(node: int) -> frozenset[int]
            out_neighbors(node: int) -> frozenset[int]
            in_degree(node: int) -> int
            out_degree(node: int) -> int
            topological_sort() -> list[int]  # Kahn's algorithm
            node_count -> int (property)
            edge_count -> int (property)
            output_node() -> int  # The unique non-VAR node with out_degree 0

        Invariants:
            - Always directed (no undirected mode)
            - Acyclic: add_edge enforces this via has_cycle_if_added
            - Node IDs are contiguous: [0, 1, ..., node_count - 1]

    Reference: IsalGraph's SparseGraph at
        /home/mpascual/research/code/IsalGraph/src/isalgraph/core/sparse_graph.py
"""
EOF

# --- src/isalsr/core/string_to_dag.py ---
cat > src/isalsr/core/string_to_dag.py << 'EOF'
"""StringToDAG (S2D) -- Execute an IsalSR instruction string to produce a LabeledDAG.

Tokenizes the input string into IsalSR tokens and executes them sequentially,
building a LabeledDAG. Handles the two-tier encoding: single-char tokens for
movement/edge/no-op, and two-char compound tokens for labeled node insertion.

Restriction: ZERO external dependencies. Only Python stdlib.

To implement:
    class StringToDAG:
        __init__(input_string: str, num_variables: int,
                 allowed_ops: frozenset[NodeType] | None = None)
            - Tokenize the input string into a list of tokens.
            - Pre-compute capacity: max_nodes = num_variables + count of V/v tokens.
            - Validate tokens against the IsalSR alphabet.
            - If allowed_ops is set, validate that all insertion tokens use allowed ops.

        run(*, trace: bool = False) -> LabeledDAG
            Execute all tokens and return the resulting LabeledDAG.
            If trace=True, also collect snapshots after each instruction.

    Tokenizer rules:
        - If current char is 'V' or 'v', consume the NEXT char as the label.
          The compound token is the two chars together (e.g., 'V+', 'vs').
          If 'V'/'v' is at end of string with no next char, raise InvalidTokenError.
        - Otherwise, the single char is the token (N, P, n, p, C, c, W).
        - Unknown characters raise InvalidTokenError.

    Initial state (for m variables):
        - LabeledDAG with m VAR nodes (IDs 0..m-1), no edges.
        - CDLL with m nodes storing graph node IDs 0..m-1, in order.
        - Both pointers on CDLL node for x_1 (graph node 0).

    Instruction dispatch:
        N: primary_ptr = cdll.next_node(primary_ptr)
        P: primary_ptr = cdll.prev_node(primary_ptr)
        n: secondary_ptr = cdll.next_node(secondary_ptr)
        p: secondary_ptr = cdll.prev_node(secondary_ptr)
        V[label]: (1) Create new node with NodeType from label.
                  (2) Add directed edge: cdll.get_value(primary_ptr) -> new_node.
                  (3) Insert new node into CDLL after primary_ptr.
                  (4) Primary pointer does NOT move.
        v[label]: Same as V[label] but using secondary pointer.
        C: Add directed edge cdll.get_value(primary_ptr) -> cdll.get_value(secondary_ptr).
           If cycle would be created, silently skip (no-op).
        c: Add directed edge cdll.get_value(secondary_ptr) -> cdll.get_value(primary_ptr).
           If cycle would be created, silently skip (no-op).
        W: No-op.

    Reference: IsalGraph's StringToGraph at
        /home/mpascual/research/code/IsalGraph/src/isalgraph/core/string_to_graph.py
"""
EOF

# --- src/isalsr/core/dag_to_string.py ---
cat > src/isalsr/core/dag_to_string.py << 'EOF'
"""DAGToString (D2S) -- Convert a LabeledDAG to an IsalSR instruction string.

Greedy algorithm adapted from IsalGraph's GraphToString. Key differences:
1. Emits two-char labeled tokens (V+, Vs, etc.) instead of bare V/v.
2. Edge direction matters: C is primary->secondary, c is secondary->primary.
3. Candidate selection for V/v considers outgoing edges only (data flow direction).
4. DAG-specific: no self-loops, directed edges only.

Restriction: ZERO external dependencies. Only Python stdlib.

To implement:
    class DAGToString:
        __init__(input_dag: LabeledDAG, initial_node: int = 0)
            - Verify initial_node is valid (typically 0 = x_1).
            - Initialize output DAG, CDLL, pointers, index mappings.

        run(*, trace: bool = False) -> str
            Execute the greedy algorithm and return the IsalSR string.

    Helper functions:
        generate_pairs_sorted_by_sum(m: int) -> list[tuple[int, int]]
            Generate all (a, b) with a, b in [-m, m].
            Sort by |a| + |b| (total displacement cost), then |a|, then (a, b).
            CRITICAL: sort by |a|+|b|, NOT a+b. This is bug B2 from IsalGraph.

    Algorithm (greedy loop):
        While nodes or edges remain uninserted:
            For each displacement pair (a, b) sorted by cost:
                1. Compute tentative pointer positions after (a, b) movement.
                2. Try V[label]: primary has uninserted neighbor via outgoing edge.
                   If found: emit movement + V[label], update state. Break.
                3. Try v[label]: secondary has uninserted neighbor via outgoing edge.
                   If found: emit movement + v[label], update state. Break.
                4. Try C: edge from primary -> secondary in input but not output.
                   If found and no cycle: emit movement + C, update state. Break.
                5. Try c: edge from secondary -> primary in input but not output.
                   If found and no cycle: emit movement + c, update state. Break.

    Index space mapping:
        _i2o: dict[int, int] -- input DAG node ID -> output DAG node ID
        _o2i: dict[int, int] -- output DAG node ID -> input DAG node ID

    Reference: IsalGraph's GraphToString at
        /home/mpascual/research/code/IsalGraph/src/isalgraph/core/graph_to_string.py
"""
EOF

# --- src/isalsr/core/canonical.py ---
cat > src/isalsr/core/canonical.py << 'EOF'
"""Canonical string computation for labeled DAGs.

Computes the canonical IsalSR string w*_D, which is a complete labeled-DAG invariant:
    w*_D = w*_D'  iff  D and D' are isomorphic as labeled DAGs.

Key simplification from IsalGraph: since input variables are distinguishable and
x_1 is a fixed, distinguished starting node, we run D2S from x_1 ONLY.
No iteration over starting nodes is needed, reducing complexity by O(n).

The 6-component structural tuple for backtracking candidate pruning:
    (|in_N1(v)|, |out_N1(v)|, |in_N2(v)|, |out_N2(v)|, |in_N3(v)|, |out_N3(v)|)
where in_Nk(v) = nodes reachable in k steps following incoming edges,
      out_Nk(v) = nodes reachable in k steps following outgoing edges.

Restriction: ZERO external dependencies. Only Python stdlib.

To implement:
    canonical_string(dag: LabeledDAG) -> str
        Exhaustive backtracking from x_1 (node 0). At each V/v branch point,
        try all valid neighbor choices (operation/constant nodes sharing the
        maximum 6-component tuple). Return shortest, then lexmin.

    compute_structural_tuple(dag: LabeledDAG, node: int) -> tuple[int, ...]
        Compute the 6-component structural tuple for a node via truncated BFS
        in both in/out directions up to distance 3.

    levenshtein(s: str, t: str) -> int
        Standard DP Levenshtein distance, O(n*m) time, O(min(n,m)) space.

    dag_distance(d1: LabeledDAG, d2: LabeledDAG) -> int
        Levenshtein distance between canonical strings.

    Mathematical reference:
        Lopez-Rubio (2025). arXiv:2512.10429v2. Section on canonical strings.
        Adapted for labeled DAGs with fixed starting node and 6-component tuple.

    Reference implementation:
        /home/mpascual/research/code/IsalGraph/src/isalgraph/core/canonical.py
"""
EOF

# --- src/isalsr/core/dag_evaluator.py ---
cat > src/isalsr/core/dag_evaluator.py << 'EOF'
"""Numerical evaluation of expression DAGs.

Evaluates a LabeledDAG on numerical input data using topological sort.
No external dependencies -- operates on Python lists and the math module.
The evaluation/ layer wraps this for vectorized numpy evaluation.

Restriction: ZERO external dependencies. Only Python stdlib + math.

To implement:
    evaluate_dag(dag: LabeledDAG, inputs: dict[int, float]) -> float
        Evaluate the DAG on scalar inputs.
        Args:
            dag: The expression DAG.
            inputs: Mapping from VAR node IDs to their scalar values.
        Returns:
            The scalar output value.
        Algorithm:
            1. Topological sort the DAG.
            2. For each node in topological order:
                - VAR: output = inputs[node_data['var_index']]
                - CONST: output = node_data['const_value']
                - Unary (SIN, COS, EXP, LOG, SQRT, ABS): output = op(single_input)
                - Binary (SUB, DIV, POW): output = op(input1, input2)
                  Input order determined by edge insertion order or node ID order.
                - Variadic (ADD, MUL): output = reduce(op, all_inputs)
            3. Return output of the output node (unique non-VAR sink).

    Protected operations:
        - LOG: log(abs(x) + 1e-10)
        - DIV: x / y if abs(y) > 1e-10 else 1.0
        - SQRT: sqrt(abs(x))
        - EXP: exp(clip(x, -500, 500)) to avoid overflow
"""
EOF

# --- src/isalsr/core/algorithms/__init__.py ---
cat > src/isalsr/core/algorithms/__init__.py << 'EOF'
"""D2S algorithm variants for IsalSR.

Provides a strategy pattern for DAG-to-string conversion algorithms:
    - GreedySingleD2S: Greedy from x_1 only (fastest)
    - GreedyMinD2S: Greedy from all variable nodes, pick lexmin shortest
    - ExhaustiveD2S: Full backtracking (true canonical, exponential)
    - PrunedExhaustiveD2S: Pruned with 6-component structural tuple

Restriction: ZERO external dependencies.
"""
EOF

# --- src/isalsr/core/algorithms/base.py ---
cat > src/isalsr/core/algorithms/base.py << 'EOF'
"""Abstract base class for D2S (DAG-to-String) algorithms.

Restriction: ZERO external dependencies. Only Python stdlib + abc.

To implement:
    class D2SAlgorithm(ABC):
        @abstractmethod
        def encode(self, dag: LabeledDAG) -> str:
            ...

        @property
        @abstractmethod
        def name(self) -> str:
            ...

    Reference: IsalGraph's G2SAlgorithm at
        /home/mpascual/research/code/IsalGraph/src/isalgraph/core/algorithms/base.py
"""
EOF

# --- src/isalsr/core/algorithms/greedy_single.py ---
cat > src/isalsr/core/algorithms/greedy_single.py << 'EOF'
"""Greedy D2S from x_1 only (single starting node).

The simplest and fastest D2S variant. Runs the greedy DAGToString algorithm
starting from x_1 (node 0). Not a graph invariant -- different internal
node numberings may produce different strings.

Restriction: ZERO external dependencies.

To implement:
    class GreedySingleD2S(D2SAlgorithm):
        def encode(self, dag: LabeledDAG) -> str: ...
        @property
        def name(self) -> str: return "greedy-single"
"""
EOF

# --- src/isalsr/core/algorithms/greedy_min.py ---
cat > src/isalsr/core/algorithms/greedy_min.py << 'EOF'
"""Greedy D2S from all variable nodes, pick shortest then lexmin.

Runs greedy D2S from each variable node x_1, ..., x_m as starting node.
Returns the shortest string; ties broken by lexicographic order.
More robust than single-start but still not a true invariant for
operation/constant nodes.

Restriction: ZERO external dependencies.

To implement:
    class GreedyMinD2S(D2SAlgorithm):
        def encode(self, dag: LabeledDAG) -> str: ...
        @property
        def name(self) -> str: return "greedy-min"
"""
EOF

# --- src/isalsr/core/algorithms/exhaustive.py ---
cat > src/isalsr/core/algorithms/exhaustive.py << 'EOF'
"""Exhaustive backtracking D2S (true canonical string).

Explores ALL valid neighbor choices at each V/v step via backtracking.
Since x_1 is a distinguished starting node, only one starting node is needed.
Returns the shortest string; ties broken by lexicographic order.

This produces the true canonical string w*_D -- a complete labeled-DAG invariant.
Complexity: exponential in the number of V/v branch points (product of neighbor-choice
counts). Practical for DAGs up to ~15 nodes.

Restriction: ZERO external dependencies.

To implement:
    class ExhaustiveD2S(D2SAlgorithm):
        def encode(self, dag: LabeledDAG) -> str: ...
        @property
        def name(self) -> str: return "exhaustive"

    Reference: IsalGraph's ExhaustiveG2S at
        /home/mpascual/research/code/IsalGraph/src/isalgraph/core/algorithms/exhaustive.py
"""
EOF

# --- src/isalsr/core/algorithms/pruned_exhaustive.py ---
cat > src/isalsr/core/algorithms/pruned_exhaustive.py << 'EOF'
"""Pruned exhaustive D2S with 6-component structural tuple.

Like ExhaustiveD2S, but prunes the candidate set at each V/v branch point
using the 6-component structural tuple:
    (|in_N1(v)|, |out_N1(v)|, |in_N2(v)|, |out_N2(v)|, |in_N3(v)|, |out_N3(v)|)

Only candidates with the maximum tuple value (lexicographic ordering) are explored.
This dramatically reduces the branching factor while preserving the canonical property.

Mathematical justification: The 6-component tuple is a labeling-independent structural
discriminant. Among nodes with identical tuples, any can be chosen without affecting
the canonical string's completeness. See Lopez-Rubio (2025), arXiv:2512.10429v2.

Restriction: ZERO external dependencies.

To implement:
    class PrunedExhaustiveD2S(D2SAlgorithm):
        def encode(self, dag: LabeledDAG) -> str: ...
        @property
        def name(self) -> str: return "pruned-exhaustive"
"""
EOF

# --- src/isalsr/adapters/__init__.py ---
cat > src/isalsr/adapters/__init__.py << 'EOF'
"""IsalSR adapter layer -- bridges to external graph/math libraries.

Each adapter imports its library independently. All adapters are optional.
"""
EOF

# --- src/isalsr/adapters/base.py ---
cat > src/isalsr/adapters/base.py << 'EOF'
"""Abstract base class for DAG adapters (Bridge pattern).

Restriction: Only Python stdlib + abc + typing.

To implement:
    class DAGAdapter(ABC, Generic[T]):
        @abstractmethod
        def from_external(self, graph: T) -> LabeledDAG: ...

        @abstractmethod
        def to_external(self, dag: LabeledDAG) -> T: ...

        def to_isalsr_string(self, graph: T, *,
                             algorithm: D2SAlgorithm | None = None) -> str: ...

        def from_isalsr_string(self, string: str, num_variables: int) -> T: ...

    Reference: IsalGraph's GraphAdapter at
        /home/mpascual/research/code/IsalGraph/src/isalgraph/adapters/base.py
"""
EOF

# --- src/isalsr/adapters/networkx_adapter.py ---
cat > src/isalsr/adapters/networkx_adapter.py << 'EOF'
"""NetworkX DiGraph <-> LabeledDAG adapter.

Optional dependency: networkx >= 3.0

To implement:
    class NetworkXAdapter(DAGAdapter[nx.DiGraph]):
        Converts nx.DiGraph (with node attribute 'label') <-> LabeledDAG.
        Node mapping: external labels -> contiguous integer IDs.
        Preserves node labels (NodeType) as node attributes.
"""
EOF

# --- src/isalsr/adapters/sympy_adapter.py ---
cat > src/isalsr/adapters/sympy_adapter.py << 'EOF'
"""SymPy Expr <-> LabeledDAG adapter.

Optional dependency: sympy >= 1.12

Critical for:
    - Verifying expression correctness (DAG eval == SymPy eval)
    - Pretty-printing discovered expressions
    - Simplifying expressions
    - Benchmark definition (expressions defined as SymPy, converted to DAG)

To implement:
    class SympyAdapter(DAGAdapter[sympy.Expr]):
        to_sympy(dag: LabeledDAG) -> sympy.Expr
            Topological sort, build SymPy expression bottom-up.

        from_sympy(expr: sympy.Expr, variables: list[sympy.Symbol]) -> LabeledDAG
            Parse SymPy expression tree into a LabeledDAG.
            Handles: Add, Mul, sin, cos, exp, log, Pow, constants, symbols.
            Detects shared subexpressions for DAG (not just tree) conversion.
"""
EOF

# --- src/isalsr/evaluation/__init__.py ---
cat > src/isalsr/evaluation/__init__.py << 'EOF'
"""IsalSR evaluation module -- fitness metrics and constant optimization.

Dependencies: numpy, scipy.
"""
EOF

# --- src/isalsr/evaluation/fitness.py ---
cat > src/isalsr/evaluation/fitness.py << 'EOF'
"""Fitness metrics for symbolic regression.

Dependencies: numpy.

To implement:
    r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float
        Coefficient of determination. R^2 = 1 - SS_res / SS_tot.

    nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float
        Normalized Root Mean Square Error = RMSE / std(y_true).

    mse(y_true: np.ndarray, y_pred: np.ndarray) -> float
        Mean Squared Error.

    evaluate_expression(dag: LabeledDAG, X: np.ndarray, y: np.ndarray) -> dict[str, float]
        Evaluate a DAG on data, return dict with R^2, NRMSE, MSE.
        X: input matrix (N x m), y: target vector (N,).
        Uses vectorized evaluation (numpy) wrapping dag_evaluator.

    Reference: Liu2025 (GraphDSR) uses R^2 and NRMSE as primary metrics.
"""
EOF

# --- src/isalsr/evaluation/constant_optimizer.py ---
cat > src/isalsr/evaluation/constant_optimizer.py << 'EOF'
"""BFGS optimization of CONST node values.

Dependencies: numpy, scipy.

After the expression structure (DAG topology) is determined, optimizes the
scalar values of all CONST nodes to minimize NRMSE on training data.

To implement:
    optimize_constants(dag: LabeledDAG, X: np.ndarray, y: np.ndarray,
                       method: str = 'L-BFGS-B',
                       max_iter: int = 100) -> LabeledDAG
        1. Extract all CONST node IDs and their current values.
        2. Define objective: f(constants) = NRMSE(dag_eval(X, constants), y).
        3. Use scipy.optimize.minimize(method=method) to find optimal constants.
        4. Return a copy of the DAG with updated CONST values.

    Mathematical reference:
        Liu2025 (GraphDSR): Two-phase approach -- discrete structure search
        followed by continuous constant optimization via BFGS.
        Petersen et al. (2021, DSR): Risk-seeking policy gradient + BFGS.
"""
EOF

# --- src/isalsr/evaluation/protected_ops.py ---
cat > src/isalsr/evaluation/protected_ops.py << 'EOF'
"""Protected mathematical operations for safe numerical evaluation.

Dependencies: numpy (for vectorized versions).

Provides both scalar (for core dag_evaluator) and vectorized (for numpy evaluation)
versions of protected operations that handle domain errors gracefully.

To implement:
    protected_log(x): log(abs(x) + 1e-10)
    protected_div(x, y): x / y if |y| > 1e-10 else 1.0
    protected_sqrt(x): sqrt(abs(x))
    protected_exp(x): exp(clip(x, -500, 500))
    protected_pow(x, y): |x|^y with overflow protection

    Standard in symbolic regression literature. See:
    - Koza (1992). Genetic Programming.
    - Schmidt & Lipson (2009). Science 324(5923):81-85.
"""
EOF

# --- src/isalsr/search/__init__.py ---
cat > src/isalsr/search/__init__.py << 'EOF'
"""IsalSR search module -- string-level search operators and algorithms.

Dependencies: numpy.

Provides mutation, crossover, and search algorithms that operate directly
on IsalSR instruction strings. The key advantage: because strings are
canonicalized, every point in the search space corresponds to a structurally
unique expression, eliminating O(k!) redundancy.
"""
EOF

# --- src/isalsr/search/operators.py ---
cat > src/isalsr/search/operators.py << 'EOF'
"""Mutation and crossover operators for IsalSR strings.

Dependencies: numpy (for random number generation).

To implement:
    Mutation operators:
        point_mutation(string: str, allowed_ops: frozenset[NodeType]) -> str
            Replace a random token with another valid token of the same category.

        insertion_mutation(string: str, allowed_ops: frozenset[NodeType]) -> str
            Insert a random valid token at a random position.

        deletion_mutation(string: str) -> str
            Remove a random token.

        subsequence_mutation(string: str, max_len: int,
                             allowed_ops: frozenset[NodeType]) -> str
            Replace a contiguous subsequence with a random one.

    Crossover operators:
        one_point_crossover(parent1: str, parent2: str) -> tuple[str, str]
            Split at random positions, swap tails.

        two_point_crossover(parent1: str, parent2: str) -> tuple[str, str]
            Swap a random substring between parents.

    All operators must be token-aware: mutations must respect the two-char
    token structure (never split a V+ into V and +).

    After any mutation/crossover, the resulting string should be canonicalized
    to maintain the isomorphism-invariant property.
"""
EOF

# --- src/isalsr/search/random_search.py ---
cat > src/isalsr/search/random_search.py << 'EOF'
"""Random search for symbolic regression using IsalSR strings.

Dependencies: numpy.

To implement:
    random_isalsr_string(num_variables: int, max_tokens: int,
                         allowed_ops: frozenset[NodeType],
                         rng: np.random.Generator) -> str
        Generate a random valid IsalSR string.

    random_search(X: np.ndarray, y: np.ndarray,
                  num_variables: int,
                  allowed_ops: frozenset[NodeType],
                  n_iterations: int = 10000,
                  max_tokens: int = 50,
                  seed: int = 42) -> list[dict]
        Run random search: generate random strings, evaluate fitness,
        track best. Return sorted list of (string, fitness) results.
"""
EOF

# --- src/isalsr/search/hill_climbing.py ---
cat > src/isalsr/search/hill_climbing.py << 'EOF'
"""Hill climbing search for symbolic regression using IsalSR strings.

Dependencies: numpy.

To implement:
    hill_climbing(X: np.ndarray, y: np.ndarray,
                  num_variables: int,
                  allowed_ops: frozenset[NodeType],
                  n_iterations: int = 10000,
                  max_tokens: int = 50,
                  n_restarts: int = 10,
                  seed: int = 42) -> list[dict]
        Multi-restart hill climbing:
        1. Generate random initial string.
        2. Apply random mutations, keep improvements.
        3. Canonicalize after each mutation.
        4. Track best across restarts.
"""
EOF

# --- src/isalsr/search/population.py ---
cat > src/isalsr/search/population.py << 'EOF'
"""Population management for evolutionary search with IsalSR strings.

Dependencies: numpy.

To implement:
    class Population:
        Manages a population of (string, fitness) pairs for evolutionary algorithms.

        __init__(size: int, num_variables: int, allowed_ops: frozenset[NodeType])
        initialize(X, y, max_tokens, seed) -> None
        select_parents(n: int, method: str = 'tournament') -> list[str]
        evolve(X, y, n_generations, mutation_rate, crossover_rate) -> dict
            Full evolutionary loop with selection, crossover, mutation,
            canonicalization, evaluation, and elitism.

    The key paper claim: evolutionary search in IsalSR's canonical space
    converges faster because O(k!) redundant representations are eliminated.
"""
EOF

# =============================================================================
# 4. Create test files
# =============================================================================
echo "[4/7] Creating test files..."

# --- tests/conftest.py ---
cat > tests/conftest.py << 'EOF'
"""Shared test fixtures for IsalSR.

Provides reusable DAGs, expressions, and test data for all test modules.
Fixtures should cover: simple expressions (x, x+y, sin(x)), multi-variable
expressions, constant expressions, and edge cases.
"""
EOF

# --- Unit tests ---
cat > tests/unit/test_cdll.py << 'EOF'
"""Unit tests for CircularDoublyLinkedList.

Tests: insert_after, remove, get_value, next_node, prev_node,
circular traversal, capacity limits, free list recycling.
Adapted from IsalGraph's test_cdll.py.
"""
EOF

cat > tests/unit/test_labeled_dag.py << 'EOF'
"""Unit tests for LabeledDAG.

Tests:
    - Empty DAG creation
    - Add nodes with labels (VAR, ADD, SIN, CONST, etc.)
    - Add edges with cycle detection (refused when path target->source exists)
    - Duplicate edge is no-op (returns False)
    - in_neighbors / out_neighbors correctness
    - in_degree / out_degree correctness
    - topological_sort on known DAGs
    - Node label and data retrieval
    - remove_edge and undo_node (for backtracking)
    - Specific expression DAGs: x+y (3 nodes, 2 edges), sin(x*y) (4 nodes, 3 edges)
    - output_node() returns correct sink node
"""
EOF

cat > tests/unit/test_node_types.py << 'EOF'
"""Unit tests for NodeType enum and registry.

Tests:
    - All NodeType values exist and are unique
    - LABEL_CHAR_MAP is complete and bijective
    - ARITY_MAP covers all NodeTypes
    - Category sets (UNARY_OPS, BINARY_OPS, VARIADIC_OPS, LEAF_TYPES) are disjoint and complete
    - OperationSet validation and filtering
"""
EOF

cat > tests/unit/test_string_to_dag.py << 'EOF'
"""Unit tests for StringToDAG (S2D).

Tests:
    - Empty string with 1 variable: DAG with 1 VAR node, 0 edges
    - Empty string with 2 variables: DAG with 2 VAR nodes, 0 edges
    - "V+": creates ADD node, edge x_1 -> ADD
    - "V+NnncVs": multi-step construction
    - Cycle detection via C/c: construct situation where C would create cycle, verify skip
    - Tokenization correctness (V+, vs, bare c vs Vc, etc.)
    - Invalid token rejection
    - allowed_ops filtering
    - Pointer immobility after V/v
    - Multi-variable initial state: verify CDLL order and pointer positions
"""
EOF

cat > tests/unit/test_dag_to_string.py << 'EOF'
"""Unit tests for DAGToString (D2S).

Tests:
    - Known expression DAGs -> verify output string reconstructs to isomorphic DAG
    - Simple expressions: x + c, sin(x), x * y + x
    - Pair generation and sorting by |a|+|b|
    - Label preservation in emitted tokens
    - Edge direction correctness
"""
EOF

cat > tests/unit/test_dag_evaluator.py << 'EOF'
"""Unit tests for DAG numerical evaluation.

Tests:
    - x evaluated at x=3.0 returns 3.0
    - x + y evaluated at x=1, y=2 returns 3.0
    - sin(x) evaluated at x=pi/2 returns 1.0
    - x * x + k (k=1) evaluated at x=2 returns 5.0
    - Protected log: log(0) does not crash
    - Protected div: div(x, 0) returns 1.0
    - Protected exp: exp(1000) does not overflow
    - Variable-arity: x + y + z evaluated correctly
    - Nested operations: sin(x + y) * exp(z)
"""
EOF

cat > tests/unit/test_roundtrip.py << 'EOF'
"""Unit tests for the round-trip property.

For known IsalSR strings w:
    1. S2D(w) -> D
    2. D2S(D, x_1) -> w'
    3. S2D(w') -> D'
    4. Assert D ~ D' (labeled DAG isomorphism)

Phase 1: Short, manually inspectable strings (1-5 tokens).
Phase 2: Longer strings covering all operation types.

Tests both the greedy D2S and the fact that the representation is faithful.
"""
EOF

cat > tests/unit/test_canonical.py << 'EOF'
"""Unit tests for canonical string computation.

Tests:
    - Two DAGs representing the same expression with different internal node
      numbering produce the same canonical string.
    - Different expressions produce different canonical strings.
    - Canonical of x+y is the same regardless of internal node IDs.
    - Canonical from x_1 (fixed start) matches expected.
    - 6-component structural tuple computation correctness.
    - Levenshtein distance: known pairs.
    - dag_distance: metric properties (symmetry, triangle inequality, identity).
"""
EOF

# --- Integration tests ---
cat > tests/integration/test_networkx_adapter.py << 'EOF'
"""Integration tests for NetworkX adapter.

Requires: networkx >= 3.0

Tests:
    - LabeledDAG -> nx.DiGraph -> LabeledDAG round-trip
    - Node attributes (labels) preserved
    - Edge direction preserved
"""
EOF

cat > tests/integration/test_sympy_adapter.py << 'EOF'
"""Integration tests for SymPy adapter.

Requires: sympy >= 1.12

Tests:
    - LabeledDAG for sin(x+y) -> SymPy -> verify sympy.sin(x + y)
    - SymPy -> LabeledDAG -> SymPy round-trip (symbolic equality)
    - Complex expressions with constants
    - Variable-arity operations (x + y + z)
    - Shared subexpressions in DAG form
"""
EOF

cat > tests/integration/test_constant_optimizer.py << 'EOF'
"""Integration tests for BFGS constant optimization.

Requires: numpy, scipy

Tests:
    - Expression c*x + c on data y = 2x + 3: after BFGS, constants ~ [2, 3]
    - Expression sin(c*x) on data y = sin(pi*x): after BFGS, c ~ pi
    - Expression with no constants: returns unchanged DAG
    - Convergence within tolerance
"""
EOF

cat > tests/integration/test_fitness.py << 'EOF'
"""Integration tests for fitness metrics.

Requires: numpy

Tests:
    - R^2 = 1.0 for perfect prediction
    - NRMSE = 0 for perfect prediction
    - Known non-perfect cases with expected values
    - evaluate_expression end-to-end: DAG + data -> metrics
"""
EOF

# --- Property tests ---
cat > tests/property/test_roundtrip_property.py << 'EOF'
"""Hypothesis property-based tests for the round-trip property.

Uses Hypothesis to generate random valid IsalSR strings and verify:
    S2D(w) ~ S2D(D2S(S2D(w), x_1)) for all valid w.

Tests both 1-variable and multi-variable cases.
"""
EOF

cat > tests/property/test_dag_acyclicity.py << 'EOF'
"""Hypothesis property-based tests for DAG acyclicity.

Uses Hypothesis to generate random IsalSR strings and verify:
    The resulting DAG from S2D is always acyclic (topological sort succeeds).

This is the fundamental safety property of the C/c cycle check.
"""
EOF

cat > tests/property/test_canonical_property.py << 'EOF'
"""Hypothesis property-based tests for canonical string invariance.

Uses Hypothesis to generate random LabeledDAGs and verify:
    - Create a permuted copy (relabel operation/constant nodes)
    - canonical_string(D) == canonical_string(D_permuted)

Also tests that non-isomorphic DAGs produce different canonical strings
(completeness of the invariant).
"""
EOF

cat > tests/property/test_evaluation_consistency.py << 'EOF'
"""Hypothesis property-based tests for evaluation consistency.

Uses Hypothesis to generate random IsalSR strings and verify:
    DAG evaluation == SymPy evaluation (within floating-point tolerance).

Requires: sympy, numpy.
"""
EOF

# =============================================================================
# 5. Create experiment and benchmark stubs
# =============================================================================
echo "[5/7] Creating experiment and benchmark files..."

# --- Experiment configs ---
cat > experiments/configs/nguyen.yaml << 'EOF'
# Nguyen benchmark configuration for IsalSR
# Reference: Uy et al. (2011). Semantically-based crossover in GP.
experiment:
  name: nguyen
  description: "12 Nguyen symbolic regression benchmarks"
  seed: 42
  n_runs: 30

data:
  train_size: 20
  test_size: 100
  x_range: [-1.0, 1.0]

search:
  operations: ["+", "*", "-", "/", "s", "c", "e", "l"]
  max_tokens: 50
  n_iterations: 100000
  population_size: 500

evaluation:
  metric: r_squared
  constant_optimization: true
  bfgs_max_iter: 100
EOF

cat > experiments/configs/feynman.yaml << 'EOF'
# Feynman equations benchmark configuration for IsalSR
# Reference: Udrescu & Tegmark (2020). AI Feynman.
experiment:
  name: feynman
  description: "Feynman physics equations benchmark"
  seed: 42
  n_runs: 10

data:
  source: "feynman"
  train_ratio: 0.8

search:
  operations: ["+", "*", "-", "/", "s", "c", "e", "l", "r", "^"]
  max_tokens: 80
  n_iterations: 200000
  population_size: 1000

evaluation:
  metric: r_squared
  constant_optimization: true
  bfgs_max_iter: 200
EOF

cat > experiments/configs/srbench.yaml << 'EOF'
# SRBench standardized benchmark configuration
# Reference: La Cava et al. (2021). Contemporary SR methods.
experiment:
  name: srbench
  description: "SRBench standardized symbolic regression benchmark"
  seed: 42
  n_runs: 10

data:
  source: "srbench"

search:
  operations: ["+", "*", "-", "/", "s", "c", "e", "l", "r", "^", "a"]
  max_tokens: 100
  n_iterations: 500000
  population_size: 2000

evaluation:
  metric: r_squared
  constant_optimization: true
  bfgs_max_iter: 200
EOF

# --- Experiment scripts (stubs) ---
cat > experiments/scripts/run_random_search.py << 'EOF'
"""Run random search experiment on SR benchmarks.

Usage: python experiments/scripts/run_random_search.py --config experiments/configs/nguyen.yaml
"""
EOF

cat > experiments/scripts/run_hill_climbing.py << 'EOF'
"""Run hill climbing experiment on SR benchmarks.

Usage: python experiments/scripts/run_hill_climbing.py --config experiments/configs/nguyen.yaml
"""
EOF

cat > experiments/scripts/run_gp.py << 'EOF'
"""Run genetic programming experiment with IsalSR string crossover/mutation.

Usage: python experiments/scripts/run_gp.py --config experiments/configs/nguyen.yaml
"""
EOF

cat > experiments/scripts/analyze_results.py << 'EOF'
"""Analyze experiment results and generate plots/tables.

Usage: python experiments/scripts/analyze_results.py --results_dir results/
"""
EOF

cat > experiments/scripts/search_space_analysis.py << 'EOF'
"""Measure search space reduction from IsalSR canonicalization.

Generates random expression DAGs, computes canonical strings, and measures:
- Number of unique canonical strings vs. total DAGs generated
- Empirical reduction factor vs. theoretical O(k!)
- Distribution of canonical string lengths

This is the core experiment for the paper's central claim.

Usage: python experiments/scripts/search_space_analysis.py --config experiments/configs/nguyen.yaml
"""
EOF

# --- Benchmark datasets ---
cat > benchmarks/datasets/__init__.py << 'EOF'
"""Benchmark dataset definitions for symbolic regression."""
EOF

cat > benchmarks/datasets/nguyen.py << 'EOF'
"""Nguyen symbolic regression benchmark definitions.

12 standard Nguyen benchmarks used across SR literature.
Reference: Uy et al. (2011). Semantically-based crossover in GP.

To implement:
    NGUYEN_BENCHMARKS: list of dicts, each containing:
        - name: str (e.g., "Nguyen-1")
        - expression: str (human-readable, e.g., "x^3 + x^2 + x")
        - num_variables: int
        - x_range: tuple[float, float]
        - target_fn: Callable (numpy vectorized)

    Nguyen-1:  x^3 + x^2 + x                    (x in [-1, 1])
    Nguyen-2:  x^4 + x^3 + x^2 + x              (x in [-1, 1])
    Nguyen-3:  x^5 + x^4 + x^3 + x^2 + x        (x in [-1, 1])
    Nguyen-4:  x^6 + x^5 + x^4 + x^3 + x^2 + x  (x in [-1, 1])
    Nguyen-5:  sin(x^2) * cos(x) - 1             (x in [-1, 1])
    Nguyen-6:  sin(x) + sin(x + x^2)             (x in [-1, 1])
    Nguyen-7:  log(x + 1) + log(x^2 + 1)         (x in [0, 2])
    Nguyen-8:  sqrt(x)                            (x in [0, 4])
    Nguyen-9:  sin(x) + sin(y^2)                  (x,y in [-1, 1])
    Nguyen-10: 2 * sin(x) * cos(y)               (x,y in [-1, 1])
    Nguyen-11: x^y                                (x,y in [0, 1])
    Nguyen-12: x^4 - x^3 + y^2/2 - y             (x,y in [-1, 1])
"""
EOF

cat > benchmarks/datasets/feynman.py << 'EOF'
"""Feynman physics equations benchmark definitions.

50+ equations from the AI Feynman dataset (fundamental physics).
Reference: Udrescu & Tegmark (2020). AI Feynman: A physics-inspired method
for symbolic regression. Science Advances 6(16).

To implement:
    FEYNMAN_BENCHMARKS: list of dicts with name, expression, num_variables,
    variable_ranges, target_fn.

    Selected examples:
    - I.6.20a:  exp(-theta^2 / 2) / sqrt(2*pi)
    - I.9.18:   G * m1 * m2 / (x2 - x1)^2
    - I.12.1:   q1 * q2 / (4*pi*epsilon*r^2)
    - I.15.10:  m0 * v / sqrt(1 - v^2/c^2)
    - II.6.15a: epsilon * E^2 / 2
"""
EOF

cat > benchmarks/datasets/srbench.py << 'EOF'
"""SRBench standardized benchmark loader.

Reference: La Cava et al. (2021). Contemporary symbolic regression methods
and their relative performance. NeurIPS Datasets & Benchmarks.

To implement:
    load_srbench_dataset(name: str) -> tuple[np.ndarray, np.ndarray]
        Load a dataset from the SRBench collection.

    list_srbench_datasets() -> list[str]
        List available SRBench dataset names.

    Note: SRBench datasets are available via the PMLB package or direct download.
"""
EOF

# =============================================================================
# 6. Create documentation files
# =============================================================================
echo "[6/7] Creating documentation..."

cat > docs/tasks/todo.md << 'EOF'
# IsalSR -- Task Tracking

## Phase 1: Core Data Structures
- [ ] types.py -- type aliases
- [ ] errors.py -- exception hierarchy
- [ ] node_types.py -- NodeType enum, arity registry, configurable operation sets
- [ ] cdll.py -- CDLL (copy from IsalGraph)
- [ ] labeled_dag.py -- LabeledDAG with cycle detection
- [ ] Unit tests for all above

## Phase 2: String-to-DAG
- [ ] string_to_dag.py -- tokenizer + executor
- [ ] dag_evaluator.py -- numerical evaluation via topological sort
- [ ] Unit tests for S2D and evaluator

## Phase 3: DAG-to-String
- [ ] dag_to_string.py -- greedy algorithm for labeled DAGs
- [ ] Round-trip tests (unit + property)

## Phase 4: Canonical String
- [ ] algorithms/base.py -- D2SAlgorithm ABC
- [ ] algorithms/greedy_single.py, greedy_min.py
- [ ] canonical.py -- 6-component tuple pruning
- [ ] algorithms/exhaustive.py, pruned_exhaustive.py
- [ ] Canonical invariant tests

## Phase 5: Adapters & Evaluation
- [ ] adapters/base.py, networkx_adapter.py, sympy_adapter.py
- [ ] evaluation/fitness.py, constant_optimizer.py, protected_ops.py
- [ ] Integration tests

## Phase 6: Search Operators
- [ ] search/operators.py -- mutation/crossover
- [ ] search/random_search.py, hill_climbing.py, population.py
- [ ] Property-based tests

## Phase 7: Benchmarks & Experiments
- [ ] benchmarks/datasets/ -- Nguyen, Feynman, SRBench
- [ ] experiments/scripts/ -- experiment runners
- [ ] Search space reduction analysis experiment
EOF

cat > docs/tasks/lessons.md << 'EOF'
# IsalSR -- Lessons Learned

Record failure modes, corrective patterns, and insights discovered during development.

## From IsalGraph (inherited lessons)

1. **B1-B9 bugs**: Index space confusion (CDLL vs graph node indices) is the #1 source
   of silent corruption. Always use `cdll.get_value(ptr)` to convert.
2. **Pair sorting**: Must sort by `|a|+|b|`, not `a+b`. The cost is total displacement.
3. **Loop termination**: Use `or` (nodes OR edges remaining), not `and`.
4. **Pointer updates**: After emitting movement instructions, update pointer fields.

## IsalSR-specific lessons

(To be filled during development)
EOF

cat > docs/DEVELOPMENT.md << 'EOF'
# IsalSR Development Guide

## Setup

```bash
conda activate isalsr
pip install -e ".[dev]"
```

## Commands

| Command | Purpose |
|---------|---------|
| `python -m pytest tests/unit/ -v` | Unit tests (fast, no external deps) |
| `python -m pytest tests/integration/ -v` | Integration tests |
| `python -m pytest tests/property/ -v` | Property-based tests (hypothesis) |
| `python -m pytest tests/ -v --cov=isalsr` | Full suite with coverage |
| `python -m ruff check --fix src/ tests/` | Lint + autofix |
| `python -m ruff format src/ tests/` | Format |
| `python -m mypy src/isalsr/` | Type checking (strict) |

## Dependency Rules

- `isalsr.core`: ZERO external deps (stdlib only)
- `isalsr.adapters`: optional (networkx, sympy)
- `isalsr.evaluation`: numpy, scipy
- `isalsr.search`: numpy
- `experiments/`, `benchmarks/`: anything

## Testing Strategy

1. **Unit tests**: Fast, no external deps. Cover all core functionality.
2. **Integration tests**: Test adapter bridges and evaluation pipeline.
3. **Property tests**: Hypothesis-based. Round-trip, acyclicity, canonical invariance.
4. **Benchmark tests**: Full pipeline on standard SR benchmarks.

## Git Workflow

- Feature branches from `main`
- All tests must pass before merge
- Ruff + mypy clean
EOF

cat > docs/ISALSR_AGENT_CONTEXT.md << 'EOF'
# IsalSR Agent Context

## What is IsalSR?

IsalSR (Instruction Set and Language for Symbolic Regression) represents
mathematical expressions as labeled DAGs encoded in instruction strings.
The canonical string representation is a complete labeled-DAG invariant,
reducing the symbolic regression search space by O(k!) for k internal nodes.

## Key Differences from IsalGraph

| Aspect | IsalGraph | IsalSR |
|--------|-----------|--------|
| Graph type | Undirected, unlabeled | Directed (DAG), labeled |
| Nodes | Unlabeled, indistinguishable | Labeled: VAR, ADD, MUL, SIN, ... |
| Initial state | 1 node | m variable nodes (x_1, ..., x_m) |
| Edge constraint | None | DAG (acyclic): C/c check for cycles |
| Start node (canonical) | Try all nodes | x_1 only (distinguished) |
| Structural tuple | 3-component | 6-component (in/out at distances 1-3) |
| Instruction alphabet | {N,n,P,p,V,v,C,c,W} | Movement: {N,n,P,p,C,c,W} + labeled: V[label], v[label] |
| Application | General graphs | Symbolic regression expressions |

## Paper Strategy

- **Central claim**: Isomorphism-invariant representation reduces SR search space by O(k!)
- **Approach**: Show existing SR methods accelerate with IsalSR canonical strings
- **NOT**: Proposing a new SR method (avoid overreach)
- **Target**: IEEE TPAMI
- **Benchmarks**: Nguyen (12), Feynman (50+), SRBench

## Sibling Project

IsalGraph: /home/mpascual/research/code/IsalGraph
CDLL implementation reused verbatim. G2S/S2G algorithms adapted for labeled DAGs.
EOF

# --- Core README ---
cat > src/isalsr/core/README.md << 'EOF'
# IsalSR Core -- Mathematical Foundation and Architecture

## 1. Introduction

IsalSR represents symbolic regression expressions as labeled Directed Acyclic Graphs (DAGs)
encoded in instruction strings. The canonical string is a complete labeled-DAG invariant,
enabling isomorphism-free search spaces for symbolic regression.

## 2. Instruction Set (Sigma_SR)

Two-tier encoding:
- Movement/structure: N, P, n, p, C, c, W (single-char)
- Labeled insertion: V[label], v[label] (two-char)
- Labels: +, *, -, /, s, c, e, l, r, ^, a, k

## 3. Initial State

For m input variables: m VAR nodes, no edges, CDLL in order, pointers on x_1.

## 4. Edge Semantics

Edge u -> v: "u provides input to v" (data flow direction).

## 5. DAG Constraint

C/c instructions check for cycles via DFS reachability before adding edges.
V/v instructions never create cycles (new node has no outgoing edges).

## 6. Canonical String

w*_D = lexmin{ w in argmin |D2S(D, x_1)| }

Computed from x_1 only (fixed, distinguished start node).
6-component structural tuple for backtracking pruning.
Complete labeled-DAG invariant: w*_D = w*_D' iff D ~ D'.

## 7. Search Space Reduction

For k internal nodes, O(k!) equivalent labelings collapse to one canonical string.
Central contribution of the paper.

(Full mathematical details to be filled during implementation.)
EOF

# --- README.md ---
cat > README.md << 'EOF'
# IsalSR

**Instruction Set and Language for Symbolic Regression**

IsalSR represents symbolic regression expressions as labeled DAGs encoded in
isomorphism-invariant instruction strings. The canonical string representation
collapses O(k!) equivalent expression representations into one, reducing the
search space for symbolic regression by factorial factors.

## Authors

- Ezequiel Lopez-Rubio (University of Malaga)
- Mario Pascual Gonzalez (University of Malaga)

## Installation

```bash
conda activate isalsr
pip install -e ".[dev]"
```

## Quick Start

```python
from isalsr.core.string_to_dag import StringToDAG
from isalsr.core.dag_to_string import DAGToString
from isalsr.core.canonical import canonical_string

# Decode: instruction string -> expression DAG
s2d = StringToDAG("V+NnncVs", num_variables=2)
dag = s2d.run()

# Encode: expression DAG -> instruction string
d2s = DAGToString(dag)
string = d2s.run()

# Canonical: isomorphism-invariant representation
canon = canonical_string(dag)
```

## References

- Lopez-Rubio (2025). arXiv:2512.10429v2. IsalGraph.
- Liu et al. (2025). Neural Networks 187:107405. GraphDSR.

## License

MIT
EOF

# =============================================================================
# 7. Final checks
# =============================================================================
echo "[7/7] Final checks..."

# Count files
PY_COUNT=$(find src/ tests/ -name "*.py" | wc -l)
echo "  Python files created: $PY_COUNT"

TOTAL_COUNT=$(find src/ tests/ experiments/ benchmarks/ docs/ -type f | wc -l)
echo "  Total files created: $TOTAL_COUNT"

echo ""
echo "=== Scaffold complete! ==="
echo ""
echo "Next steps:"
echo "  1. conda activate isalsr"
echo "  2. pip install -e \".[dev]\""
echo "  3. python -c \"import isalsr; print(isalsr.__version__)\""
echo "  4. python -m pytest tests/ -v"
echo "  5. python -m ruff check src/ tests/"
echo "  6. python -m mypy src/isalsr/"
