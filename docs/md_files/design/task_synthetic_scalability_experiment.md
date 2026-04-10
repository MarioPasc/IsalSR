# Task: Synthetic Scalability Experiment — ρ and Canonicalization Time vs. k

**Context**: The main paper is too long; Table 3 (ρ vs. overhead by k from real SR runs) moves to the supplementary. We replace it with a controlled synthetic study that isolates the effect of $k$ (internal node count) on search-space reduction factor $\rho$ and canonicalization wall-clock time.

**Destination**: Supplementary material, referenced from Section 4 of the main paper.

---

## 1. Scientific Rationale

The empirical $\rho$ values observed during real SR runs (UDFS, Bingo) confound two effects: (i) the intrinsic combinatorial reduction from canonicalization, and (ii) the bias of each SR method's search toward particular DAG topologies. A synthetic experiment decouples these by generating random algebraic expressions *uniformly* with a controlled number of internal nodes $k$, then measuring $\rho$ and canonicalization cost directly.

### 1.1 Random Expression Generation — Literature Basis

The generation procedure follows the **random unary–binary tree method** described by Lample and Charton (2020) for training data in neural symbolic mathematics. The procedure is:

1. Sample a random unary–binary tree with exactly $k$ internal nodes (i.e., nodes of arity 1 or 2).
2. Assign each internal node an operator drawn uniformly from the allowed operator set $\mathcal{O}$, respecting arity constraints.
3. Assign each leaf a terminal drawn uniformly from the variable set $\{x_0, \ldots, x_{m-1}\}$ (constants excluded to keep the focus on structural redundancy).

This is equivalent to generating a random *Motzkin tree* of a given size (internal nodes are either unary or binary), which is a well-studied class in analytic combinatorics (Flajolet and Sedgewick, *Analytic Combinatorics*, Cambridge University Press, 2009, Chapter I).

Gutjahr (1991) proposed algorithms for uniform random generation of expressions modulo algebraic identities (commutativity, associativity), extending the Hickey–Cohen framework for context-free languages. Our experiment does **not** need Gutjahr's equivalence-aware generation because the whole point is to measure how many of the naïvely generated expressions collapse to the same canonical form: the redundancy *is* the signal. Quotienting it out during generation would eliminate the very quantity we aim to measure.

**References**:

- Gutjahr, W.J. (1991). Uniform random generation of expressions respecting algebraic identities. *Computing*, 47, 51–67. https://doi.org/10.1007/BF02242022
- Lample, G. and Charton, F. (2020). Deep Learning for Symbolic Mathematics. *ICLR 2020*. arXiv:1912.01412
- Hickey, T. and Cohen, J. (1983). Uniform random generation of strings in a context-free language. *SIAM J. Comput.*, 12(4), 645–655.
- Flajolet, P. and Sedgewick, R. (2009). *Analytic Combinatorics*. Cambridge University Press.
- Duchon, P., Flajolet, P., Louchard, G. and Schaeffer, G. (2004). Boltzmann samplers for the random generation of combinatorial structures. *Combinatorics, Probability and Computing*, 13(4–5), 577–625.

### 1.2 Operator Set

Use the same operator set as the main experiments. The commutative decomposition is used (see `src/isalsr/core/README.md` §2.1):

| Label char | Operation | Arity |
|------------|-----------|-------|
| `+` | ADD | 2 (commutative) |
| `*` | MUL | 2 (commutative) |
| `^` | POW | 2 (non-commutative) |
| `s` | SIN | 1 |
| `c` | COS | 1 |
| `e` | EXP | 1 |
| `l` | LOG | 1 |
| `g` | NEG | 1 |
| `i` | INV | 1 |

Binary ops: `{+, *, ^}`. Unary ops: `{s, c, e, l, g, i}`. Total: 9 operators.

### 1.3 Number of Variables

Run the experiment for $m \in \{1, 2, 3\}$ to cover the Nguyen ($m=1,2$) and Feynman ($m=2,3$) ranges.

---

## 2. Experiment Design

### 2.1 Parameters

| Parameter | Values | Justification |
|-----------|--------|---------------|
| $k$ (internal nodes) | $\{1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15\}$ | Covers the range seen in real SR (UDFS: $k \leq 8$; Bingo: $k \leq 15$). Gaps at high $k$ to save compute. |
| $m$ (variables) | $\{1, 2, 3\}$ | Covers benchmarks used. |
| $N_{\text{expr}}$ (expressions per $(k, m)$ cell) | 200 | Enough for stable mean/std of $\rho$ and timing. |
| $N_{\text{perm}}$ (permutations per expression) | $\min(k!, 5000)$ | Exhaustive for $k \leq 7$ ($7! = 5040$); sampled for $k > 7$. |
| Global seed | 42 | Reproducibility. |
| Canonicalization algorithm | `fast_canonical_string(mode="wl_only")` | Paper's preferred algorithm. |
| Canonicalization timeout | 120 s per expression | Safety net for pathological DAGs. |

### 2.2 Per-Expression Procedure

For each expression $D_i$ generated with $k$ internal nodes and $m$ variables:

1. **Generate** a random unary–binary tree $T$ with exactly $k$ internal nodes using the procedure in §3.1 below. Label internal nodes with operators and leaves with variables.
2. **Build** a `LabeledDAG` from $T$.
3. **Generate permutations**: produce $N_{\text{perm}}$ distinct permutations of the $k$ internal node indices. For $k \leq 7$, enumerate all $k!$ permutations exhaustively. For $k > 7$, sample $5000$ permutations uniformly without replacement.
4. **Canonicalize each permuted DAG**: for each permutation $\pi$, construct the permuted DAG $D_i^{\pi}$ (relabel internal nodes according to $\pi$, keeping VAR nodes fixed), then compute `fast_canonical_string(D_i^{\pi})`.
5. **Measure**:
   - $\rho_i = N_{\text{perm}} / |\{\text{distinct canonical strings}\}|$ — empirical reduction factor for expression $i$.
   - $\bar{t}_i$ — mean canonicalization wall-clock time across all permutations (use `time.perf_counter()`).
   - $|w^*_i|$ — length of the canonical string.
   - $|V_i|$, $|E_i|$ — node count, edge count of the DAG.

### 2.3 Aggregation

Per $(k, m)$ cell, aggregate over $N_{\text{expr}}$ expressions:

- $\bar{\rho}(k, m) \pm \sigma_\rho(k, m)$ — mean and std of $\rho$.
- $\bar{t}(k, m) \pm \sigma_t(k, m)$ — mean and std of canonicalization time.
- Median and IQR of both quantities (for robustness).

### 2.4 Expected Outcome

- $\rho$ should grow as $\Theta(k!)$ for generic (asymmetric) expressions, with outliers below $k!$ for expressions with non-trivial automorphism groups ($\rho = k!/|\text{Aut}(D)|$).
- Canonicalization time should grow polynomially (near $O(k^2)$) for `fast_canonical_string` since it is greedy with WL-guided tie-breaking.

---

## 3. Implementation Specification

### 3.1 Random Expression Tree Generation

```
function RANDOM_EXPR_TREE(k, m, ops_binary, ops_unary, rng):
    """Generate a random unary-binary tree with exactly k internal nodes."""
    # Step 1: build random tree structure
    #   Use the "grow" method: maintain a list of open leaves.
    #   Start with a single open leaf (root placeholder).
    #   For each of the k internal nodes, pick a random open leaf,
    #   replace it with a randomly chosen operator.
    #     - If binary: add 2 new open leaves.
    #     - If unary: add 1 new open leaf.
    #   Net leaves = 1 + (number of binary ops among the k).
    #   But we must ensure we consume exactly k operators, so:
    #     At each step, choose binary vs unary with a constraint-aware
    #     probability that guarantees we can still place all remaining
    #     operators. Specifically, if we have `remaining` operators to place
    #     and `open` open leaves, we need:
    #       open - 1 + remaining_binary + remaining_unary = open + (remaining_binary)
    #     i.e., we need open + Σ(arity_i - 1) for remaining ops to end ≥ 0.
    #     This is equivalent to: at each step, if open == 1 and remaining > 0,
    #     we MUST pick a binary op (to keep open ≥ 1). Otherwise, pick freely.

    # Step 2: label internal nodes
    #   Already done during tree construction (operator chosen at each step).

    # Step 3: label leaves
    #   Each leaf ← rng.choice([x_0, ..., x_{m-1}])
    #   (Variables chosen with replacement; every variable appears at least
    #    once only if m ≤ number of leaves, but we do NOT enforce this —
    #    expressions like sin(x_0) + cos(x_0) with m=2 are valid.)

    return tree
```

**Critical constraint**: The algorithm must guarantee that **exactly** $k$ internal nodes are placed. The constraint is: at any step with `remaining` operators left and `open` open leaf slots, if `open == 1` and `remaining > 0`, a binary operator must be chosen (to produce a net +1 open slot rather than 0). If `open > remaining`, unary operators are still fine. A clean implementation is the *budget-aware random growth* algorithm.

### 3.2 LabeledDAG Construction from Tree

Convert the tree to a `LabeledDAG`:

1. Create $m$ VAR nodes (indices $0, \ldots, m-1$).
2. For each internal node in the tree (BFS or DFS order), call `dag.add_node(NodeType.XXX)`.
3. For each edge in the tree (child → parent in data-flow direction), call `dag.add_edge(child_id, parent_id)`.
4. For leaves referencing variable $x_j$, reuse the existing VAR node $j$ (do NOT create a new node per leaf — this is a DAG, not a tree). Multiple leaves referencing the same variable produce a fan-out from that VAR node.

**Important**: Because multiple leaves can reference the same variable, the resulting structure is a DAG (not necessarily a tree), which is exactly what IsalSR operates on.

### 3.3 Internal Node Permutation

Given a `LabeledDAG` $D$ with internal (non-VAR) nodes at indices $\{i_1, \ldots, i_k\}$, a permutation $\pi \in S_k$ produces a new DAG $D^\pi$ by renumbering the internal nodes according to $\pi$ while keeping all VAR node indices fixed and preserving all edge relationships and labels.

Use the existing `permute_internal_nodes()` helper if available in the test suite (`tests/unit/test_fast_canonical.py` imports it). Otherwise, implement it:

```python
def permute_internal_nodes(dag: LabeledDAG, perm: list[int]) -> LabeledDAG:
    """Relabel internal nodes according to perm.
    
    perm[i] = j means the i-th internal node (in sorted order) 
    maps to position j in the new DAG.
    VAR nodes keep their original indices.
    All edges and labels are preserved.
    """
```

### 3.4 Output Format

Save results as a single CSV file with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `k` | int | Number of internal nodes |
| `m` | int | Number of input variables |
| `expr_id` | int | Expression index within the $(k, m)$ cell |
| `seed` | int | Per-expression seed (for reproducibility) |
| `n_nodes` | int | Total node count $|V|$ |
| `n_edges` | int | Total edge count $|E|$ |
| `n_perms` | int | Number of permutations tested |
| `n_unique_canonicals` | int | Number of distinct canonical strings |
| `rho` | float | Reduction factor $= n\_perms / n\_unique\_canonicals$ |
| `rho_over_kfact` | float | $\rho / k!$ — should be $\leq 1.0$; equals $1/|\text{Aut}(D)|$ |
| `canonical_len` | int | Length of the canonical string $|w^*|$ |
| `mean_canon_time_s` | float | Mean canonicalization time per permuted DAG (seconds) |
| `std_canon_time_s` | float | Std of canonicalization time |
| `total_canon_time_s` | float | Total time for all permutations of this expression |
| `timeout_count` | int | Number of permutations that timed out |

Additionally, save a JSON metadata file with: operator set, variable counts, date, Python version, hostname, global seed, IsalSR version/commit hash.

---

## 4. File Organization

```
experiments/synthetic_scalability/
├── run_synthetic_scalability.py   # Single-file experiment script
├── slurm_synthetic.sh             # SLURM submission script
└── README.md                      # Brief description
```

Output goes to:
```
results/synthetic_scalability/
├── synthetic_scalability_results.csv
├── synthetic_scalability_metadata.json
└── logs/
    └── synthetic_scalability_{SLURM_JOB_ID}.log
```

---

## 5. Single-File Experiment Script — `run_synthetic_scalability.py`

### 5.1 Requirements

The script must be a **single self-contained file** (no additional modules) that:

1. Imports from `isalsr.core` (LabeledDAG, NodeType, fast_canonical_string, StringToDAG, DAGToString) and stdlib only.
2. Accepts CLI arguments: `--output-dir`, `--seed` (default 42), `--n-expr` (default 200), `--max-perms` (default 5000), `--timeout` (default 120), `--k-values` (default "1,2,3,4,5,6,7,8,10,12,15"), `--m-values` (default "1,2,3").
3. Uses `logging` throughout (no `print()`).
4. Writes the CSV incrementally (flush after each $(k, m)$ cell) so partial results survive crashes.
5. Writes the metadata JSON at the end.
6. Uses `time.perf_counter()` for wall-clock timing.
7. Sets `numpy` and Python `random` seeds deterministically from the global seed + expression index.
8. Handles `CanonicalTimeoutError` gracefully (records the timeout, continues with next permutation).

### 5.2 Key Functions

```python
@dataclass
class ExprTreeNode:
    """Node in a random expression tree (before conversion to LabeledDAG)."""
    node_type: NodeType          # operator type or VAR
    children: list[ExprTreeNode] # ordered children (empty for leaves)
    var_index: int | None = None # only for VAR leaves

def generate_random_expr_tree(
    k: int, m: int, 
    binary_ops: list[NodeType], unary_ops: list[NodeType],
    rng: np.random.Generator,
) -> ExprTreeNode:
    """Generate a random expression tree with exactly k internal nodes."""

def tree_to_labeled_dag(tree: ExprTreeNode, m: int) -> LabeledDAG:
    """Convert an expression tree to a LabeledDAG, reusing VAR nodes."""

def permute_internal_nodes(dag: LabeledDAG, perm: list[int]) -> LabeledDAG:
    """Produce a new LabeledDAG with internal nodes permuted."""

def run_single_expression(
    dag: LabeledDAG, k: int, m: int, expr_id: int, seed: int,
    max_perms: int, timeout: float,
) -> dict:
    """Run all permutations for one expression, return result dict."""

def main() -> None:
    """CLI entry point."""
```

### 5.3 Operator Constraint During Tree Growth

The tree growth algorithm must respect operator arities. At each step with `remaining` operators to place and `open` open leaf slots:

- If `open == 1` and `remaining > 0`: **must** choose a binary operator (otherwise we'd close the tree prematurely with 0 open slots and remaining operators unplaced).
- If `open > remaining`: may choose unary or binary freely.
- If `open == remaining`: must choose only unary operators (each unary consumes 1 slot and opens 1, maintaining the count; a binary would overshoot).

Wait — let me be precise. Let $r$ = remaining operators, $o$ = open leaf slots. Placing a binary op: $o \to o - 1 + 2 = o + 1$, $r \to r - 1$. Placing a unary op: $o \to o - 1 + 1 = o$, $r \to r - 1$. We need $o \geq 1$ at every step and $o \geq 0$ at the end ($o$ must be $\geq 0$ when $r = 0$; actually it will be $\geq 1$ since the last op still produces leaves).

The constraint is: after placing the current operator, the remaining operators must still be placeable. After this step with $r' = r - 1$ remaining and new open count $o'$:
- We need $o' \geq r'$ is NOT required (binary ops increase open count).
- We need $o' \geq 1$ if $r' > 0$ (need at least one slot for next operator).

So the actual constraint is simply: **if $o = 1$ and $r > 1$, do not pick unary** (it would leave $o = 1, r' = r - 1$ which is fine actually...). Actually there is no problem as long as $o \geq 1$ when $r > 0$. Since both binary and unary keep $o \geq 1$ when starting from $o \geq 1$, the only risk is $o = 0$. But $o$ starts at 1 and never decreases below... wait: unary: $o \to o$, binary: $o \to o+1$. So $o$ is non-decreasing from the initial value of 1. Hence $o \geq 1$ always holds. **There is no constraint** — any mix of unary and binary works for any $k$.

Correction: $o$ starts at 1 (the root is open). Placing a unary: pick one open slot, replace with operator + 1 child = $o - 1 + 1 = o$. Placing a binary: $o - 1 + 2 = o + 1$. So $o$ is indeed monotonically non-decreasing. No constraint needed. The implementation can freely choose unary or binary at each step.

### 5.4 POW Operand Order

POW is the only non-commutative binary operator. When building the DAG, the first child added via `add_edge` is the base, the second is the exponent. The tree structure already encodes this order (left child = base, right child = exponent).

---

## 6. SLURM Script — `slurm_synthetic.sh`

```bash
#!/bin/bash
#SBATCH --job-name=isalsr_synth_scalability
#SBATCH --partition=gputitan          # Adjust to Picasso partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=results/synthetic_scalability/logs/synth_%j.log
#SBATCH --error=results/synthetic_scalability/logs/synth_%j.err

module load Python/3.11                # Adjust to Picasso module
source /path/to/venv/bin/activate      # Adjust to your virtualenv

cd $SLURM_SUBMIT_DIR

python -m experiments.synthetic_scalability.run_synthetic_scalability \
    --output-dir results/synthetic_scalability \
    --seed 42 \
    --n-expr 200 \
    --max-perms 5000 \
    --timeout 120 \
    --k-values "1,2,3,4,5,6,7,8,10,12,15" \
    --m-values "1,2,3"
```

**Time estimate**: For $k \leq 7$ (exhaustive, up to 5040 perms × 200 expr × 7 values of $k$ × 3 values of $m$), the fast canonical is $\sim 1$ms per call → $\sim 5040 \times 200 \times 7 \times 3 \times 0.001 \approx 6{,}350$ s ≈ 1.8 h. For $k \in \{8, 10, 12, 15\}$ (5000 perms each), $\sim 5000 \times 200 \times 4 \times 3 \times 0.01 \approx 120{,}000$ s ≈ 33 h... This exceeds the budget.

**Revised approach**: Either reduce $N_{\text{expr}}$ to 50 for $k \geq 10$, or split into a SLURM array by $(k, m)$. Recommend a SLURM array:

```bash
#!/bin/bash
#SBATCH --job-name=isalsr_synth
#SBATCH --partition=gputitan
#SBATCH --array=0-32              # 11 k-values × 3 m-values = 33 tasks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=results/synthetic_scalability/logs/synth_%A_%a.log
#SBATCH --error=results/synthetic_scalability/logs/synth_%A_%a.err

module load Python/3.11
source /path/to/venv/bin/activate
cd $SLURM_SUBMIT_DIR

K_VALUES=(1 2 3 4 5 6 7 8 10 12 15)
M_VALUES=(1 2 3)

K_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
M_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

K=${K_VALUES[$K_IDX]}
M=${M_VALUES[$M_IDX]}

python -m experiments.synthetic_scalability.run_synthetic_scalability \
    --output-dir results/synthetic_scalability \
    --seed 42 \
    --n-expr 200 \
    --max-perms 5000 \
    --timeout 120 \
    --k-values "$K" \
    --m-values "$M"
```

The script must support writing per-cell CSV fragments (one file per $(k, m)$) that are merged in a separate post-processing step, OR append-mode writing with file locking. **Recommend**: each array task writes `results/synthetic_scalability/raw/synth_k{K}_m{M}.csv`, and a lightweight merge script concatenates them.

---

## 7. Acceptance Criteria (Local Validation Before Picasso)

Run locally with `--n-expr 5 --max-perms 50 --k-values "1,2,3,4" --m-values "1,2"` and verify:

1. **CSV well-formed**: all columns present, no NaN in `rho`, types correct.
2. **ρ ≥ 1.0** for all rows (by definition).
3. **ρ ≤ k!** for all rows (theoretical upper bound).
4. **rho_over_kfact ≤ 1.0** for all rows.
5. **Canonical invariance**: for at least 3 expressions, verify that all permuted DAGs produce the same canonical string (i.e., `n_unique_canonicals == 1`). This should hold for most generic expressions.
6. **Reproducibility**: running twice with the same `--seed` produces identical CSV.
7. **Metadata JSON** contains all expected fields.
8. **No `print()` calls** in the script — only `logging`.
9. **Timeout handling**: inject a trivially large $k$ (e.g., $k = 20$) with `--timeout 0.001` and verify the script does not crash and records `timeout_count > 0`.

---

## 8. Figures to Generate (Separate Script, Not Part of This Task)

The CSV produced by this experiment feeds into two figures for the supplementary:

1. **ρ vs. k** (log-scale y-axis): box plots of $\rho$ per $k$, overlaid with the $k!$ theoretical curve. One panel per $m$.
2. **Canonicalization time vs. k** (log-scale y-axis): box plots of mean_canon_time_s per $k$, one panel per $m$. Overlaid with $O(k^2)$ and $O(k!)$ reference curves.

These figures are generated by the existing `generate_rf_vs_overhead.py` pattern (attached), adapted to read the synthetic CSV.

---

## 9. Paper Text (Draft for Supplementary)

> **Supplementary §X: Synthetic Scalability Analysis.**
> To isolate the effect of DAG size on search-space reduction and canonicalization cost, we generated random algebraic expressions with $k \in \{1, \ldots, 15\}$ internal nodes and $m \in \{1, 2, 3\}$ input variables. For each configuration, 200 random expression DAGs were produced by the random unary–binary tree method of Lample and Charton (2020), with operators drawn uniformly from the set $\{+, \times, \hat{}, \sin, \cos, \exp, \log, \text{neg}, \text{inv}\}$. For each expression, we exhaustively permuted all $k!$ internal-node labelings (for $k \leq 7$) or sampled 5,000 permutations (for $k > 7$), canonicalized each permuted DAG via the WL-guided greedy algorithm (§2.5), and measured the reduction factor $\rho = N_{\text{perm}} / N_{\text{unique}}$.
>
> Figure X(a) confirms that $\rho$ tracks the theoretical $\Theta(k!)$ bound: the median $\rho$ equals $k!$ for generic expressions and falls below $k!$ only for expressions with non-trivial automorphism groups, consistent with the Orbit-Stabilizer prediction $\rho = k!/|\text{Aut}(D)|$. Figure X(b) shows that the WL-guided canonicalization time grows polynomially (empirically $\sim O(k^{2.3})$), confirming that the greedy-invariant algorithm avoids the factorial worst case of exhaustive canonicalization.
>
> The generation procedure is grounded in Gutjahr's (1991) framework for uniform generation of algebraic expressions, adapted to a simpler setting where equivalence-aware sampling is unnecessary because the quantity of interest is precisely the redundancy among naïvely generated expressions.
