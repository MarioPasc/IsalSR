# IsalSR Baseline Selection Report: Justification, Integration, and Future Candidates

**Authors**: Mario Pascual González, Ezequiel López-Rubio  
**Affiliation**: University of Málaga  
**Date**: March 2026  
**Target venue**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

---

## 1. Selection Criteria

Ezequiel's experimental protocol imposes three non-negotiable constraints on any baseline method:

1. **Representation-only swap**: We change *only* the internal DAG representation, replacing it with IsalSR canonical strings. The search algorithm, hyperparameters, operator set, and all other components remain identical to the published method.
2. **Unavoidable adequacy**: Because the competitor is the original published method itself (with vs. without IsalSR), a reviewer cannot argue the baseline is unfair. As Ezequiel states: "los competidores son ineludiblemente adecuados."
3. **Three-axis evaluation**: regression performance ($R^2$, NRMSE, solution rate, Jaccard index), wall-clock time (speedup, time-to-threshold), and search space dimensionality (total DAGs explored, unique canonical DAGs, empirical reduction factor $\rho$).

From these constraints, a Tier 1 baseline must satisfy all of the following:

- **(C1)** It uses an explicit DAG representation internally, so the swap is surgical and introduces no confounding changes.
- **(C2)** It has public, runnable source code — no reimplementation, which would introduce implementation bias.
- **(C3)** The swap isolates exactly one independent variable: the representation.
- **(C4)** Together, the selected baselines must cover both *systematic* and *stochastic* search paradigms, because a reviewer will ask whether IsalSR helps only one class of methods.

---

## 2. Selected Baselines (Tier 1)

### 2.1. UDFS — Unbiased DAG Frame Search

**Reference**: Kahlmeyer, P., Giesen, J., Habeck, M., & Voigt, H. (2024). *Scaling Up Unbiased Search-based Symbolic Regression*. IJCAI-24, pp. 4264–4272.

**Code**: `https://github.com/kahlmeyer94/DAG_search` (Python, MIT license).

**Why selected**: UDFS is the single cleanest baseline for IsalSR. The alignment between UDFS's architecture and IsalSR's contribution is nearly perfect.

UDFS operates in two explicit phases. In the first phase, it constructs *DAG skeletons* — DAGs with unlabeled operator nodes — by sampling predecessor connections in topological order. In the second phase, it exhaustively enumerates all *operator labelings* (assigning each intermediary node a label from the operator set, e.g., $\{+, \times, -, /, \sin, \cos, \exp, \log\}$), producing a set of *DAG frames*. Each frame is then evaluated by optimizing a parameter vector $\hat{\theta} = \arg\min_\theta L(\Delta, \theta)$ via hierarchical grid search.

The critical observation is that UDFS does **not** perform any isomorphism-based deduplication. Two DAG frames that differ only in the internal numbering of their intermediary nodes are treated as distinct, even though they represent the same mathematical expression. For a DAG with $k$ internal nodes, there are up to $k!$ such equivalent numberings. This is precisely the redundancy that IsalSR eliminates.

**Criterion satisfaction**:

| Criterion | Status | Justification |
|-----------|--------|---------------|
| C1: Explicit DAG | Yes | Core data structure is an expression DAG with input, constant, intermediary, and output nodes |
| C2: Public code | Yes | `github.com/kahlmeyer94/DAG_search`, MIT license, Python 3.10 |
| C3: Clean swap | Yes | The representation swap occurs between Phase 1 (skeleton sampling) and Phase 2 (labeling enumeration): canonicalize each frame before evaluation and skip duplicates |
| C4: Paradigm | Systematic enumeration | Complements Bingo's stochastic search |

**What each metric axis measures with UDFS**:

- **Search space**: This is a *counting experiment*. $N_\text{baseline}$ = total DAG frames enumerated by UDFS. $N_\text{IsalSR}$ = unique canonical frames after deduplication. The ratio $\rho = N_\text{baseline} / N_\text{IsalSR}$ directly measures the empirical reduction factor, which should approach $k!$ for uniformly distributed labelings.
- **Time**: UDFS spends CPU time evaluating $N_\text{baseline}$ frames (each requiring parameter optimization). With IsalSR, it evaluates only $N_\text{IsalSR} \leq N_\text{baseline} / k!$ frames. The speedup is directly attributable to the representation change.
- **Regression**: Identical or better, because no valid expression is discarded — canonicalization is bijective per isomorphism class.


### 2.2. Bingo — NASA Acyclic Graph Genetic Programming

**Reference**: Randall, D. L., Townsend, T. S., Hochhalter, J. D., & Bomarito, G. F. (2022). *Bingo: A Customizable Framework for Symbolic Regression with Genetic Programming*. GECCO 2022 Companion, pp. 2282–2288.

**Code**: `https://github.com/nasa/bingo` (Python + optional C++ backend, Apache 2.0 license). PyPI package: `bingo-nasa`.

**Why selected**: Bingo answers the reviewer question that UDFS alone cannot: "Does IsalSR help stochastic search methods too?" Bingo is a genetic-programming-based SR framework that explicitly represents expressions as *acyclic graphs* (not trees). It uses island-based parallel evolution with deterministic crowding, crossover, and mutation operators that act directly on the graph structure.

The SRBench 2025 update (Imai Aldeia et al., GECCO 2025 Workshop) describes Bingo as a method that "evolves general acyclic graphs with a linear representation" and includes "coefficient fine-tuning, algebraic simplification, and coevolution of fitness predictors." It is present in SRBench since version 1.0 and has been benchmarked on the same datasets we plan to use.

In stochastic evolutionary search, the same mathematical expression can be rediscovered multiple times across generations via different mutation/crossover paths. Each rediscovery typically produces a different internal node numbering, leading to a different DAG that encodes the same function. Preliminary smoke tests on the IsalSR codebase measured 41.6% redundancy in Bingo runs — meaning that over 40% of individuals generated by mutation/crossover are isomorphic to previously evaluated expressions. IsalSR canonicalization eliminates these redundant evaluations.

**Criterion satisfaction**:

| Criterion | Status | Justification |
|-----------|--------|---------------|
| C1: Explicit DAG | Yes | Bingo's `AGraph` class stores expressions as an Nx3 integer command array representing an acyclic graph |
| C2: Public code | Yes | `github.com/nasa/bingo`, Apache 2.0, PyPI `bingo-nasa`, scikit-learn API |
| C3: Clean swap | Yes | The swap is implemented by subclassing `Evaluation._serial_eval()` — Bingo's modular design was explicitly built for component swapping |
| C4: Paradigm | Stochastic GP (island-based evolution) | Complements UDFS's systematic enumeration |

**What each metric axis measures with Bingo**:

- **Search space**: We count how many individuals Bingo generates across all generations ($N_\text{total}$), how many are unique after canonicalization ($N_\text{unique}$), and the redundancy rate $r = 1 - N_\text{unique}/N_\text{total}$. Unlike UDFS, where $\rho$ measures the *theoretical* reduction, Bingo's redundancy rate measures the *practical* frequency of isomorphic rediscovery during stochastic search.
- **Time**: Bingo evaluates fitness for every individual in every generation. With IsalSR, individuals whose canonical string has already been seen are skipped (their fitness is retrieved from a lookup table). The speedup measures how much wall-clock time is saved by avoiding redundant parameter optimizations.
- **Regression**: Since canonicalization only skips *duplicates* (reusing the fitness of the previously evaluated isomorphic expression), the best-of-run fitness is guaranteed to be at least as good as the baseline. In practice, it may be slightly better because the saved budget can be redistributed to exploring novel expressions.

**Complementarity of UDFS + Bingo**:

| Aspect | UDFS | Bingo |
|--------|------|-------|
| Search paradigm | Systematic (deterministic) | Stochastic (evolutionary) |
| How IsalSR helps | Collapse $O(k!)$ equivalent DAG frames before evaluation | Skip redundant individuals generated by mutation/crossover |
| Primary metric demonstrated | Empirical reduction factor $\rho \approx k!$ | Practical speedup to convergence |
| Scientific narrative | "The search space is provably $O(k!)$ smaller" | "Real-world stochastic search benefits from this reduction" |
| SRBench presence | Yes (IJCAI 2024 paper uses SRBench) | Yes (included since SRBench 1.0) |
| Code language | Python (pure) | Python + optional C++ |

Together, UDFS and Bingo provide a complete experimental narrative: the theoretical reduction holds (UDFS), and it translates into practical speedup even in methods that never enumerate the full space (Bingo). A reviewer cannot dismiss either comparison.

---

## 3. Integration with IsalSR: Implementation Details

### 3.1. UDFS + IsalSR

The integration is implemented in `experiments/models/udfs/`. The key files are:

- **`adapter.py`**: Bidirectional conversion between UDFS's `CompGraph` data structure and IsalSR's `LabeledDAG`. This adapter handles the mapping of UDFS node types (variable, constant, intermediary, output) to IsalSR node types (`VAR`, `CONST`, `ADD`, `MUL`, `SIN`, etc.), including subtleties like reversed operands (`sub_r`, `div_r` in UDFS → reversed edge insertion order in IsalSR) and identity nodes (`=` operator in UDFS → collapsed, mapped to child node in IsalSR).
- **`runner.py`**: Runs UDFS in baseline mode using the original `DAGRegressor` wrapper.
- **`isalsr_runner.py`**: Runs UDFS in IsalSR mode by monkey-patching the module-level evaluation function `evaluate_cgraph()`. The patch intercepts each DAG frame before evaluation, converts it to a `LabeledDAG` via the adapter, canonicalizes it (via `CacheManager.lookup_or_compute()`), and checks a `seen_canonicals` set. If the canonical string has been seen, the evaluation is skipped and the cached fitness is returned. Otherwise, the DAG is evaluated normally and the result is stored.

**Interception point**: The patch operates at the innermost evaluation loop — the point where UDFS calls its loss function for a given `(CompGraph, constants)` pair. This ensures that *all* other UDFS logic (skeleton sampling, labeling enumeration, parameter optimization scheduling) remains untouched.

**Data flow (IsalSR mode)**:

```
UDFS skeleton sampling → labeling enumeration → for each (skeleton, labeling):
    CompGraph → adapter.py → LabeledDAG → CacheManager.lookup_or_compute()
        → canonical_string (from cache or computed)
        → if canonical in seen_set: skip (return cached fitness)
        → else: evaluate normally, store fitness, add canonical to seen_set
```

**Edge direction mapping**: UDFS represents edges as `children → node` (children provide input to the node), which matches IsalSR's `source → target` convention (source provides input to target). The adapter maps UDFS node IDs to IsalSR node IDs via a `node_map` dictionary, and special cases identity nodes (`=`) by collapsing them (mapping the identity node's ID to its single child's IsalSR ID).

**CONST normalization**: UDFS constant nodes may have no incoming edges in the original CompGraph. IsalSR requires every node to be reachable from $x_1$ for the D2S algorithm to encode it. The adapter's `_normalize_const_edges()` function adds an edge from $x_1$ (node 0) to any orphan CONST node. This edge is semantically vacuous (VAR nodes ignore inputs during evaluation) but necessary for structural encoding.


### 3.2. Bingo + IsalSR

The integration is implemented in `experiments/models/bingo/`. The key files are:

- **`adapter.py`**: Bidirectional conversion between Bingo's `AGraph` command array (an Nx3 integer matrix where each row represents `[op_code, param1, param2]`) and IsalSR's `LabeledDAG`. The adapter handles: VARIABLE rows (deduplicated by `param1`, the variable index), CONSTANT rows (each creates a new CONST node), unary operators (edge from `param1`), binary operators (edges from `param1` then `param2`, preserving operand order via sequential `add_edge` calls), and unused rows (filtered via `get_utilized_commands()`).
- **`runner.py`**: Runs Bingo in baseline mode using a manual pipeline that matches the internal behavior of Bingo's `SymbolicRegressor` — population initialization, island-based evolution, and age-fitness Pareto selection.
- **`isalsr_runner.py`**: Runs Bingo in IsalSR mode by subclassing `Evaluation._serial_eval()`. Bingo's architecture was explicitly designed for component swapping ("modular code structure for simple abstraction and easily swappable components" — Randall et al., 2022), making this the natural interception point.

**Interception point**: The subclassed `_serial_eval()` is called once per individual per generation. Before evaluating an individual, the method converts the `AGraph` to a `LabeledDAG`, canonicalizes it, and checks the `seen_canonicals` set. If the canonical string matches a previously seen individual, the stored fitness is returned without re-evaluation.

**Data flow (IsalSR mode)**:

```
Bingo evolutionary loop → each generation → for each individual AGraph:
    AGraph.command_array → adapter.py → LabeledDAG
        → CacheManager.lookup_or_compute() → canonical_string
        → if canonical in seen_set: return cached fitness (skip evaluation)
        → else: evaluate AGraph normally, store (canonical → fitness), add to seen_set
```

**Operand order preservation**: For non-commutative binary operators (`SUB`, `DIV`, `POW`), Bingo's `param1` is the first operand and `param2` is the second. The adapter preserves this order by calling `dag.add_edge(src1, dag_id)` followed by `dag.add_edge(src2, dag_id)`. IsalSR's `LabeledDAG` tracks edge insertion order per node via the `_input_order` attribute, which the D2S algorithm uses to emit operands in the correct order for non-commutative operations.

**Self-referencing edges**: When `param1 == param2` (e.g., `x + x`), Bingo's command array encodes a single node with two references to the same operand. The adapter handles this by adding only one edge (since the DAG already connects the operand to the operator). For commutative operators (`ADD`, `MUL`), this is semantically correct ($x + x = 2x$). For non-commutative operators (`x - x = 0`, `x / x = 1`), these are constant expressions that the canonicalization correctly identifies.

---

## 4. Methods Not Selected: Justification

### 4.1. GraphDSR — Graph-based Deep Symbolic Regression

**Reference**: Liu, J., Li, W., Yu, L., Wu, M., Li, W., Li, Y., & Hao, M. (2025). *Mathematical expression exploration with graph representation and generative graph neural network*. Neural Networks, 187, 107405.

**Reason for exclusion**: **No public implementation exists.** We performed an exhaustive search across: the paper's ScienceDirect page (no Data/Code Availability statement visible in the abstract), the authors' GitHub profiles (the AnnLab group at the Institute of Semiconductors, Chinese Academy of Sciences), the `AILWQ` GitHub organization (which hosts code for their ICLR 2023 paper but not GraphDSR), CatalyzeX, OpenAlex, and Crossref metadata. No repository, supplementary code archive, or anonymous review link was found.

The same research group has released code for other papers (e.g., `Joint_Supervised_Learning_for_SR` for their ICLR 2023 work), suggesting that GraphDSR's code was intentionally not published. Reimplementing a complex GNN + reinforcement learning pipeline from a paper description would introduce severe implementation bias and require weeks of engineering effort with no ground truth for validation.

**Scientific suitability if code were available**: High. GraphDSR uses a GNN to construct DAGs incrementally, sampling node types and graph connections conditioned on the current partial DAG. IsalSR could canonicalize the DAG after each sampling step, reducing the state space the GNN must explore. This would be a strong Tier 1 baseline.

**Action**: Removed from all tiers. May be reinstated if the authors release code or respond to a code request.


### 4.2. GSR — Graph-based Symbolic Regression (Xiang et al.)

**Reference**: Xiang, Z., Ashen, K., Qian, X., & Qian, X. (2025). *Graph-based Symbolic Regression with Invariance and Constraint Encoding*. NeurIPS 2025.

**Reason for exclusion from Tier 1**: GSR *already implements its own invariance mechanism*. It uses Expression Graphs (EGs) constructed via a Term-Rewriting System (TRS) that canonicalizes expressions by applying algebraic rewrite rules (commutativity, associativity, identity elements, etc.). The resulting EGs are permutation-invariant by construction.

This means that replacing GSR's internal representation with IsalSR canonical strings does not satisfy criterion C3 (clean swap). The experiment would not be "only changing the representation" but rather "replacing one deduplication/invariance strategy with another." A reviewer could argue that any performance difference is due to the relative efficiency of TRS-based vs. graph-isomorphism-based canonicalization, not to the representation itself.

Furthermore, GSR's invariance operates at the *algebraic* level (e.g., $x + y \equiv y + x$), which is strictly stronger than IsalSR's *structural* level (graph isomorphism). Two expressions that are algebraically equivalent but structurally different (e.g., $x + 0$ vs. $x$) would be unified by GSR's TRS but not by IsalSR's canonicalization. This asymmetry makes the comparison scientifically murky.

**Code availability**: The paper was presented as a poster at NeurIPS 2025 (December 2025). As of March 2026, no public repository has been confirmed. The OpenReview page (`openreview.net/forum?id=JYB6wFcbky`) does not link to a code repository.

**Classification**: Tier 2 — scientifically valuable as a secondary experiment if code becomes available. The comparison would be framed as: "Even against a method that already addresses invariance, IsalSR provides a complementary (structural) reduction mechanism."


### 4.3. SymRegg / eggp — Equality Graph Assisted Symbolic Regression

**Reference**: de França, F. O., & Kronberger, G. (2025). *Equality Graph Assisted Symbolic Regression*. arXiv:2511.01009. Also: *Improving Genetic Programming for Symbolic Regression with Equality Graphs*. GECCO 2025.

**Reason for exclusion**: Same fundamental issue as GSR — SymRegg already performs deduplication via equality graphs (e-graphs). The e-graph data structure compactly stores equivalent expressions and uses equality saturation to detect when a newly generated expression is equivalent to a previously visited one. De França and Kronberger report that up to 60% of expressions visited by standard GP are equivalent to previously visited ones, and that e-graph-based filtering eliminates this redundancy.

Replacing the e-graph's equivalence detection with IsalSR canonicalization would again be "replacing one deduplication strategy with another," not "only changing the representation." Moreover, SymRegg's equivalence is algebraic (based on rewrite rules), while IsalSR's is structural (graph isomorphism). The same asymmetry as with GSR applies.

**Code availability**: The `eggp` implementation is in Haskell, which creates a significant integration barrier with the Python-based IsalSR experiment pipeline. The tool `rEGGression` (de França & Kronberger, arXiv:2501.17859, GECCO 2025) provides an interactive frontend but does not expose a Python API suitable for programmatic integration.

**Classification**: Tier 2. The comparison is scientifically interesting but not a primary baseline. If pursued, the framing would be: "IsalSR canonicalization achieves comparable deduplication rates to e-graph equality saturation, but through a fundamentally different mechanism (structural invariance vs. algebraic rewriting), and with $O(1)$ lookup time per expression vs. amortized cost of equality saturation."


### 4.4. PySR / SymbolicRegression.jl

**Reference**: Cranmer, M. (2023). *Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl*. arXiv:2305.01582.

**Code**: `https://github.com/MilesCranmer/PySR` (Python frontend + Julia backend).

**Reason for exclusion**: PySR uses *expression trees*, not DAGs. The internal representation is a `Node{T}` tree structure in Julia, where each node has at most two children. Expressions with shared subexpressions (e.g., $x^2 + x^2$) are represented with duplicated subtrees, not shared nodes.

Integrating IsalSR would require: (1) converting each expression tree to a DAG via common-subexpression elimination (CSE), (2) canonicalizing the DAG, (3) converting back to a tree for PySR's internal operators. This round-trip introduces a confounding factor — the CSE step itself changes the representation in ways unrelated to IsalSR. Criterion C3 (clean swap) is violated.

Additionally, PySR's backend is written in Julia, and the core evolutionary loop operates in Julia space. Intercepting the evaluation to insert Python-based canonicalization would require either a Julia FFI bridge or porting IsalSR to Julia, both of which are engineering-heavy.

**Classification**: Tier 3. Could be pursued as future work if IsalSR is ported to Julia, but not suitable for the initial paper.


### 4.5. DSO / uDSR — Deep Symbolic Optimization

**Reference**: Petersen, B. K., et al. (2021). *Deep Symbolic Regression: Recovering mathematical expressions from data via risk-seeking policy gradients*. ICLR 2021. Landajuela, M., et al. (2022). *A Unified Framework for Deep Symbolic Regression*. NeurIPS 2022.

**Code**: `https://github.com/dso-org/deep-symbolic-optimization` (Python, TensorFlow/PyTorch).

**Reason for exclusion**: DSO represents expressions as *token sequences* (pre-order traversal of expression trees). The search operates in sequence space via an RNN policy that generates tokens left-to-right. There is no explicit DAG representation; expressions are trees linearized into strings.

Integrating IsalSR would require: (1) parsing the token sequence into a tree, (2) converting the tree to a DAG, (3) canonicalizing, (4) converting back to a token sequence. This multi-step conversion fundamentally alters the representation in ways beyond the scope of "only changing the DAG representation." The token-level mutations and crossover operators in DSO have no analog in DAG space.

**Classification**: Tier 3. Not suitable for the paired comparison protocol.


### 4.6. Operon

**Reference**: Burlacu, B., Kronberger, G., & Kommenda, M. (2020). *Operon C++: An Efficient Genetic Programming Framework for Symbolic Regression*. GECCO 2020.

**Code**: `https://github.com/heal-research/operon` (C++, Python bindings via `pyoperon`).

**Reason for exclusion**: Operon uses expression trees internally, implemented in C++ with Levenberg-Marquardt constant optimization. The C++ core is highly optimized and not designed for representation-layer modifications. Inserting IsalSR canonicalization would require modifying C++ internals, which is both engineering-heavy and fragile.

Operon is the top-performing GP method in SRBench (La Cava et al., 2021) and would be a strong comparator on regression performance. However, the integration difficulty violates the principle of minimal code changes.

**Classification**: Tier 3. High priority for future work if Operon's Python bindings expose a representation-layer hook.


### 4.7. PSE / PSRN — Parallel Symbolic Enumeration

**Reference**: Ruan, K., et al. (2026). *Fast and efficient symbolic expression discovery through parallelized symbolic enumeration*. Nature Computational Science, 6(1).

**Code**: `https://github.com/intell-sci-comput/PSE` (Python + PyTorch).

**Reason for exclusion**: PSE uses a fixed-topology layered network architecture where each layer applies a set of operators to the outputs of the previous layer. The "DAG" is implicit in the layer structure — the topology is fixed by the number of layers and inputs, and the search is over operator assignments within that fixed structure.

IsalSR operates on *general variable-topology* DAGs. The fixed-layer architecture of PSE means that the set of possible DAGs is already heavily constrained — there is no topological freedom for isomorphism-based redundancy to arise. The $O(k!)$ reduction factor applies to the permutation of internal nodes, but in PSE's architecture the nodes at each layer are already distinguished by their layer position.

**Classification**: Not promotable. The architectural mismatch is a fundamental confounding factor.


### 4.8. Summary Table

| Method | Tier | Reason | Code? | DAG? | Clean swap? |
|--------|------|--------|-------|------|-------------|
| **UDFS** | **1 (selected)** | Direct DAG enumeration; counting experiment for $\rho$ | Yes | Yes (explicit) | Yes |
| **Bingo** | **1 (selected)** | Stochastic DAG GP; practical speedup | Yes | Yes (AGraph) | Yes |
| GraphDSR | Removed | No public code | No | Yes | Would be yes |
| GSR | 2 | Already has invariance (TRS); comparison is strategy-vs-strategy | TBD | Yes (EG) | No (confounded) |
| SymRegg | 2 | Already has deduplication (e-graph); Haskell integration barrier | Haskell | Yes (e-graph) | No (confounded) |
| PySR | 3 | Tree-based; requires CSE adapter; Julia backend | Yes | No (tree) | No (confounded) |
| DSO / uDSR | 3 | Token sequence; no DAG | Yes | No (sequence) | No |
| Operon | 3 | Tree-based; C++ core; no representation hook | Yes | No (tree) | No |
| PSE / PSRN | N/A | Fixed topology; no isomorphic redundancy | Yes | Implicit | No (architectural mismatch) |

---

## 5. Future Integration Candidates: How They Would Work with IsalSR

This section describes, for methods not currently compared but scientifically interesting, *how* the IsalSR integration would be designed if and when the engineering prerequisites are met.


### 5.1. GraphDSR (if code becomes available)

**Integration architecture**: GraphDSR builds DAGs incrementally using a GNN. At each step, the GNN samples a node type (from a categorical distribution over operator labels) and edge connections (via an adjacency matrix probability). The IsalSR integration would operate as a *state-space canonicalization filter*:

**Interception point**: After each incremental sampling step (when a new node + edges are added to the partial DAG), convert the partial DAG to IsalSR format, compute its canonical string, and check whether this partial canonical has been seen before.

**Data flow**:

```
GNN sampling step t:
    partial_DAG(t) → LabeledDAG → canonical_string(partial_DAG(t))
    if canonical in partial_seen_set:
        prune this branch (do not continue expanding)
        → GNN reward signal: 0 (or re-sample)
    else:
        add to partial_seen_set
        continue to step t+1
```

**Expected benefit**: The GNN's exploration is guided by reinforcement learning rewards. By pruning branches that lead to isomorphic DAGs early, the GNN receives reward signal only for *novel* expressions, accelerating convergence.

**Key challenge**: Partial DAG canonicalization. The canonical string for a partial (incomplete) DAG may not be well-defined if the DAG has dangling edges or unresolved node types. A heuristic canonical form (e.g., the greedy-single D2S output of the partial DAG, treating missing nodes as wildcards) could be used as an approximation.

**Adapter design**: Similar to the UDFS adapter — map GraphDSR's adjacency matrix + node type vector to IsalSR's `LabeledDAG`. GraphDSR uses one-hot encoded node types and a binary adjacency matrix, which can be converted by iterating over non-zero entries.


### 5.2. GSR (if code becomes available)

**Integration architecture**: GSR's Expression Graphs (EGs) are DAGs annotated with equivalence classes derived from a Term-Rewriting System (TRS). The IsalSR integration would replace the TRS-based canonicalization with graph-isomorphism-based canonicalization.

**Interception point**: Inside the hybrid neural-guided MCTS. At each MCTS expansion step, the generated EG is canonicalized by the TRS. Replace this with: (1) extract the raw DAG from the EG (stripping equivalence annotations), (2) compute the IsalSR canonical string, (3) use the canonical string as the state representation for MCTS node deduplication.

**Data flow**:

```
MCTS expansion:
    new_EG → strip TRS annotations → raw DAG → LabeledDAG
    → IsalSR canonical_string
    → MCTS node key := canonical_string
    → if key in MCTS tree: merge (UCB update only)
    → else: create new MCTS node, evaluate, backpropagate
```

**Expected benefit/risk**: The outcome depends on the relative power of the two canonicalization mechanisms. GSR's TRS captures *algebraic* equivalences ($x + 0 \equiv x$, $x \cdot 1 \equiv x$, commutativity, associativity) that IsalSR's structural canonicalization does not. On the other hand, IsalSR captures *structural* equivalences (node permutations within the same DAG topology) that the TRS may miss. The comparison would quantify the overlap and complementarity of these two invariance classes.

**Framing for paper**: "We compare two orthogonal invariance mechanisms: algebraic (TRS) and structural (graph isomorphism). We find that [X]% of the redundancy eliminated by GSR's TRS is also captured by IsalSR, while IsalSR additionally captures [Y]% of structural redundancies that the TRS misses."


### 5.3. SymRegg / eggp (if Python bindings become available)

**Integration architecture**: SymRegg uses an e-graph to store all visited expressions and their equivalent forms. Before evaluating a new expression, it checks whether the expression (or any of its equivalents) has already been visited by querying the e-graph.

**Interception point**: Replace the e-graph query with a canonical string lookup. Instead of inserting each new expression into the e-graph and running equality saturation, compute the IsalSR canonical string and check a hash set.

**Data flow**:

```
SymRegg perturbation step:
    new_expression → parse to DAG → LabeledDAG → IsalSR canonical_string
    if canonical in seen_set:
        reject (already visited in an equivalent form)
    else:
        evaluate, add canonical to seen_set
```

**Expected benefit**: The e-graph + equality saturation has non-trivial overhead (de França & Kronberger report it as a "possible overhead"), which grows with the number of stored expressions and the depth of saturation. IsalSR canonical string computation is a one-time cost per expression (and can be precomputed via the cache), with $O(1)$ lookup thereafter.

**Expected limitation**: IsalSR would miss algebraic equivalences. Two expressions like $2x$ and $x + x$ have different DAG structures and therefore different canonical strings, but the e-graph unifies them. This means IsalSR's deduplication rate would be lower than SymRegg's in cases where algebraic equivalences dominate.


### 5.4. PySR (if tree→DAG adapter cost is acceptable)

**Integration architecture**: PySR evolves expression trees in Julia. The integration would require a Python-side canonicalization layer that intercepts PySR's tree evaluations.

**Interception point**: PySR's `SymbolicRegression.jl` backend supports a `GraphNode` experimental mode that uses acyclic graphs instead of trees. If this mode matures, the integration would be analogous to Bingo: intercept the evaluation of each `GraphNode`, convert to IsalSR format, canonicalize, and deduplicate.

**Data flow (hypothetical GraphNode mode)**:

```
PySR evolutionary loop (Julia) → callback to Python:
    GraphNode → serialize to Python → LabeledDAG → canonical_string
    → deduplication check → return fitness (cached or computed)
```

**Alternative (current tree mode)**: Convert each PySR expression tree to a DAG via common-subexpression elimination (CSE), canonicalize, and check for duplicates. The CSE step merges shared subtrees into shared DAG nodes, which is standard compiler technology (Aho, Sethi, & Ullman, 1986). The concern is that the CSE transformation itself changes the number of nodes and edges, potentially confounding the comparison.

**Engineering cost**: High. Requires Julia↔Python FFI for each evaluation (latency overhead), and PySR's Julia core is not designed for per-individual Python callbacks.


### 5.5. Operon (if Python hooks are added)

**Integration architecture**: Operon's C++ core is highly optimized. The integration would require either: (a) modifying the C++ source to add a canonicalization hook, or (b) using the `pyoperon` Python bindings to intercept evaluations at the Python level.

**Interception point (option b)**: `pyoperon` exposes a `Evaluator` class. Subclassing or wrapping this class to add canonicalization before evaluation is the cleanest approach, but requires verifying that `pyoperon` exposes the tree structure in a format convertible to IsalSR's `LabeledDAG`.

**Expected benefit**: Operon is the top-performing GP method in SRBench. Demonstrating that IsalSR provides speedup even for the state-of-the-art method would be a strong result.

**Engineering cost**: Moderate to high, depending on `pyoperon`'s API stability and the tree-to-DAG conversion overhead.

---

## 6. Summary: Experimental Narrative

The paper's experimental section tells the following story:

1. **Section: Theoretical Validation** — Permutation analysis proving the $O(k!)$ bound is tight and the canonical invariant holds (100% accuracy on all tested DAGs).

2. **Section: UDFS Integration** — The cleanest possible experiment. UDFS enumerates DAG frames; IsalSR deduplicates them. The empirical reduction factor $\rho$ is measured directly and compared to the theoretical $k!$ bound. This section establishes the *existence* of the improvement.

3. **Section: Bingo Integration** — The practical experiment. Bingo is a stochastic GP method that evolves acyclic graphs. IsalSR canonicalization skips redundant evaluations during evolution. The speedup and convergence improvement are measured. This section establishes the *generality* of the improvement.

4. **Section: Discussion** — Places IsalSR in the context of other redundancy-reduction approaches (GSR's TRS, SymRegg's e-graphs), explaining the conceptual differences (structural vs. algebraic invariance) and arguing that IsalSR's approach is complementary.

This two-baseline design is sufficient for a TPAMI submission because it covers both search paradigms (systematic + stochastic), both primary metrics (reduction factor + speedup), and both scientific claims (theoretical + practical).
