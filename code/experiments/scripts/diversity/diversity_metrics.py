"""Diversity metrics for Conjecture X.7 empirical validation.

Computes effective diversity ratio (delta), pairwise Levenshtein distance,
bipartite GED approximation (Riesen & Bunke 2009), WL hash features, and
PCA projection for Bingo populations.

References:
    Riesen, K. and Bunke, H. "Approximate graph edit distance computation
    by means of bipartite matching." Image and Vision Computing, 27(7),
    2009. DOI:10.1016/j.imavis.2008.04.004.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

from isalsr.core.canonical import fast_canonical_string, levenshtein
from isalsr.core.labeled_dag import LabeledDAG

log = logging.getLogger(__name__)

# ======================================================================
# Data structures
# ======================================================================


@dataclass
class PopulationSnapshot:
    """Raw population data at one generation."""

    generation: int
    variant: str
    seed: int
    canonical_strings: list[str]
    fitnesses: np.ndarray  # (N,)
    wl_features: np.ndarray  # (N, d)
    lev_distances: np.ndarray  # (N, N) symmetric
    bp_ged_distances: np.ndarray  # (N, N) symmetric
    exact_ged_distances: np.ndarray | None  # (N, N) symmetric or None
    exact_ged_mask: np.ndarray | None  # (N, N) bool: True = proven optimal
    pca_coords: np.ndarray  # (N, 2)
    pca_explained_variance: tuple[float, float]
    command_arrays: list[np.ndarray]  # list of (stack, 3) arrays
    constants: list[np.ndarray]  # list of 1-d constant arrays


@dataclass
class DiversitySummary:
    """Computed summary statistics for one snapshot."""

    generation: int
    variant: str
    seed: int
    delta: float
    d_bar_lev: float
    d_bar_bp: float
    d_bar_exact: float  # -1.0 if not computed
    diameter_lev: float
    diameter_bp: float
    diameter_exact: float  # -1.0 if not computed
    best_fitness: float
    best_r2: float
    n_unique_classes: int
    pca_var1: float
    pca_var2: float
    n_exact_pairs: int  # pairs where exact GED was proven optimal
    n_total_pairs: int  # total C(N,2) pairs
    # Fragmentation metrics (added v2)
    cv_ged: float = 0.0  # coefficient of variation of upper-tri BP-GED
    frac_zero_pairs: float = 0.0  # fraction of zero-distance pairs
    d_bar_nonzero: float = 0.0  # mean GED restricted to non-isomorphic pairs


# ======================================================================
# Fragmentation metrics
# ======================================================================


def compute_cv_ged(bp_matrix: np.ndarray) -> float:
    """Coefficient of variation of upper-triangular BP-GED entries.

    CV = sigma / mu. High CV indicates fragmented population (bimodal
    distance distribution); low CV indicates uniform spread.
    """
    tri = bp_matrix[np.triu_indices_from(bp_matrix, k=1)]
    if len(tri) == 0:
        return 0.0
    mu = float(np.mean(tri))
    if mu < 1e-15:
        return 0.0
    return float(np.std(tri) / mu)


def compute_frac_zero_pairs(bp_matrix: np.ndarray) -> float:
    """Fraction of upper-triangular pairs with zero BP-GED distance.

    Measures isomorphic redundancy. By construction, frac_zero ≈ 1 - delta.
    """
    tri = bp_matrix[np.triu_indices_from(bp_matrix, k=1)]
    if len(tri) == 0:
        return 0.0
    return float(np.sum(tri < 1e-10) / len(tri))


def compute_d_bar_nonzero(bp_matrix: np.ndarray) -> float:
    """Mean BP-GED restricted to non-isomorphic pairs (d > 0).

    From Remark X.6: d_bar_nonzero = d_bar / (1 - frac_zero).
    """
    tri = bp_matrix[np.triu_indices_from(bp_matrix, k=1)]
    nonzero = tri[tri > 1e-10]
    if len(nonzero) == 0:
        return 0.0
    return float(np.mean(nonzero))


# ======================================================================
# Effective diversity ratio
# ======================================================================


def compute_delta(canonical_strings: list[str]) -> tuple[float, int]:
    """Effective diversity ratio: |unique canonical strings| / N.

    Args:
        canonical_strings: N canonical strings (one per individual).

    Returns:
        (delta, n_unique) where delta in (0, 1].
    """
    n = len(canonical_strings)
    if n == 0:
        return 0.0, 0
    n_unique = len(set(canonical_strings))
    return n_unique / n, n_unique


# ======================================================================
# Levenshtein distance matrix
# ======================================================================


def compute_levenshtein_matrix(canonical_strings: list[str]) -> np.ndarray:
    """Compute NxN upper-triangular Levenshtein distance matrix.

    Args:
        canonical_strings: N canonical strings.

    Returns:
        (N, N) float32 matrix with upper triangle filled, diagonal=0.
    """
    n = len(canonical_strings)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = levenshtein(canonical_strings[i], canonical_strings[j])
            mat[i, j] = d
            mat[j, i] = d
    return mat


# ======================================================================
# Bipartite GED approximation (Riesen & Bunke 2009)
# ======================================================================


def _dag_to_nx_data(dag: LabeledDAG) -> dict:
    """Extract lightweight data dict from LabeledDAG for BP-GED.

    Returns dict with 'labels', 'in_deg', 'out_deg', 'edges' keys.
    Avoids pickling the full LabeledDAG object for multiprocessing.
    """
    n = dag.node_count
    labels = [dag.node_label_unchecked(i).value for i in range(n)]
    in_deg = [dag.in_degree(i) for i in range(n)]
    out_deg = [len(dag.out_neighbors_raw(i)) for i in range(n)]
    edges = set()
    for u in range(n):
        for v in dag.out_neighbors_raw(u):
            edges.add((u, v))
    return {
        "n": n,
        "labels": labels,
        "in_deg": in_deg,
        "out_deg": out_deg,
        "edges": edges,
    }


def _bipartite_ged(g1: dict, g2: dict) -> float:
    """Bipartite GED approximation for a single pair.

    Uses Hungarian algorithm on a cost matrix that accounts for:
    - Node label substitution cost (0 if same, 1 if different)
    - Node deletion/insertion cost (1 per node + 1 per incident edge)
    - Edge edit costs from the resulting assignment

    Satisfies identity of indiscernibles: d=0 iff graphs are identical
    in structure and labels.

    Args:
        g1, g2: Lightweight graph dicts from _dag_to_nx_data.

    Returns:
        Approximate GED (upper bound on exact GED).
    """
    n1, n2 = g1["n"], g2["n"]
    size = n1 + n2

    # Build cost matrix
    cost = np.full((size, size), np.inf, dtype=np.float64)

    # Quadrant 1 (n1 x n2): substitution costs
    for i in range(n1):
        for j in range(n2):
            label_cost = 0.0 if g1["labels"][i] == g2["labels"][j] else 1.0
            cost[i, j] = label_cost

    # Quadrant 2 (n1 x n2): deletion (row i -> epsilon column n2+i)
    for i in range(n1):
        cost[i, n2 + i] = 1.0 + g1["in_deg"][i] + g1["out_deg"][i]

    # Quadrant 3 (n2 x n1): insertion (epsilon row n1+j -> col j)
    for j in range(n2):
        cost[n1 + j, j] = 1.0 + g2["in_deg"][j] + g2["out_deg"][j]

    # Quadrant 4 ((n2 x n1) block): epsilon-epsilon = 0
    for i in range(n2):
        for j in range(n1):
            cost[n1 + i, n2 + j] = 0.0

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    node_cost = cost[row_ind, col_ind].sum()

    # Build node mapping from assignment
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r < n1 and c < n2:
            mapping[r] = c

    # Compute edge edit cost from the mapping
    # Count edges in g1 whose mapped version is NOT in g2
    edge_del = 0
    for u, v in g1["edges"]:
        if u not in mapping or v not in mapping or (mapping[u], mapping[v]) not in g2["edges"]:
            edge_del += 1

    # Count edges in g2 not covered by mapped g1 edges
    mapped_edges = set()
    for u, v in g1["edges"]:
        if u in mapping and v in mapping:
            mapped_edges.add((mapping[u], mapping[v]))
    edge_ins = len(g2["edges"] - mapped_edges)

    return node_cost + edge_del + edge_ins


def _compute_ged_pair(args):
    """Worker function for parallel GED computation."""
    i, j, g1, g2 = args
    return i, j, _bipartite_ged(g1, g2)


def compute_bipartite_ged_matrix(
    graph_data: list[dict],
    n_workers: int = 1,
) -> np.ndarray:
    """Compute NxN bipartite GED approximation matrix.

    Args:
        graph_data: N lightweight graph dicts from _dag_to_nx_data.
        n_workers: Number of parallel workers (1 = sequential).

    Returns:
        (N, N) float32 symmetric matrix.
    """
    n = len(graph_data)
    mat = np.zeros((n, n), dtype=np.float32)

    pairs = [(i, j, graph_data[i], graph_data[j]) for i, j in combinations(range(n), 2)]

    if n_workers <= 1 or len(pairs) == 0:
        for i, j, g1, g2 in pairs:
            mat[i, j] = _bipartite_ged(g1, g2)
            mat[j, i] = mat[i, j]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for i, j, d in pool.map(_compute_ged_pair, pairs, chunksize=256):
                mat[i, j] = d
                mat[j, i] = d

    return mat


# ======================================================================
# Exact GED with timeout (NetworkX optimize_graph_edit_distance)
# ======================================================================


def _dag_to_nx_digraph(dag: LabeledDAG):
    """Convert LabeledDAG to NetworkX DiGraph for exact GED."""
    import networkx as nx

    G = nx.DiGraph()
    for i in range(dag.node_count):
        G.add_node(i, label=dag.node_label_unchecked(i).value)
    for u in range(dag.node_count):
        for v in dag.out_neighbors_raw(u):
            G.add_edge(u, v)
    return G


def _nx_node_subst_cost(n1_attrs, n2_attrs):
    return 0.0 if n1_attrs["label"] == n2_attrs["label"] else 1.0


def _nx_node_del_cost(n_attrs):
    return 1.0


def _nx_node_ins_cost(n_attrs):
    return 1.0


def _nx_edge_subst_cost(e1_attrs, e2_attrs):
    return 0.0


def _nx_edge_del_cost(e_attrs):
    return 1.0


def _nx_edge_ins_cost(e_attrs):
    return 1.0


def _exact_ged_with_timeout(nx_g1, nx_g2, timeout_s: float = 10.0) -> tuple[float, bool]:
    """Compute best-effort exact GED using optimize_graph_edit_distance.

    Iterates the anytime A* algorithm for up to timeout_s seconds,
    returning the best (tightest) upper bound found.

    Args:
        nx_g1, nx_g2: NetworkX DiGraphs.
        timeout_s: Per-pair time budget in seconds.

    Returns:
        (best_ged, is_exact) where is_exact indicates if the optimal
        solution was proven (generator exhausted within timeout).
    """
    import time as _time

    import networkx as nx

    gen = nx.optimize_graph_edit_distance(
        nx_g1,
        nx_g2,
        node_subst_cost=_nx_node_subst_cost,
        node_del_cost=_nx_node_del_cost,
        node_ins_cost=_nx_node_ins_cost,
        edge_subst_cost=_nx_edge_subst_cost,
        edge_del_cost=_nx_edge_del_cost,
        edge_ins_cost=_nx_edge_ins_cost,
    )

    best = float("inf")
    is_exact = False
    t0 = _time.monotonic()

    try:
        for ub in gen:
            best = ub
            if _time.monotonic() - t0 > timeout_s:
                break
        else:
            is_exact = True  # generator exhausted = optimal found
    except StopIteration:
        is_exact = True

    return best, is_exact


def _compute_exact_ged_pair(args):
    """Worker for parallel exact GED."""
    i, j, nx_g1, nx_g2, timeout_s = args
    d, exact = _exact_ged_with_timeout(nx_g1, nx_g2, timeout_s)
    return i, j, d, exact


@dataclass
class ExactGEDResult:
    """Summary of subsampled exact GED computation."""

    d_bar: float  # mean pairwise GED over sampled pairs
    diameter: float  # max GED over sampled pairs
    n_exact: int  # pairs where optimal was proven
    n_sampled: int  # total pairs sampled
    values: np.ndarray  # (n_sampled,) raw GED values
    pair_indices: np.ndarray  # (n_sampled, 2) (i, j) indices
    exact_flags: np.ndarray  # (n_sampled,) bool: proven optimal?


def compute_exact_ged_subsample(
    dags: list[LabeledDAG | None],
    n_subsample: int = 2000,
    timeout_per_pair: float = 5.0,
    n_workers: int = 1,
    rng_seed: int = 42,
) -> ExactGEDResult:
    """Compute exact GED on a random subsample of pairs.

    Instead of all C(N,2) pairs (prohibitively expensive), randomly
    samples n_subsample pairs. Mean estimate has SE proportional to
    1/sqrt(n_subsample), which at 2000 pairs is only ~3x wider than
    the full C(200,2)=19,900 computation.

    Args:
        dags: N LabeledDAGs (None entries skipped).
        n_subsample: Number of random pairs to sample.
        timeout_per_pair: Seconds per pair for A* solver.
        n_workers: Parallel workers.
        rng_seed: RNG seed for pair selection reproducibility.

    Returns:
        ExactGEDResult with summary stats and raw values.
    """
    n = len(dags)
    rng = np.random.RandomState(rng_seed)

    # Pre-convert to NetworkX
    nx_graphs = []
    for dag in dags:
        if dag is not None:
            nx_graphs.append(_dag_to_nx_digraph(dag))
        else:
            nx_graphs.append(None)

    # Generate all valid pair indices, then subsample
    valid_pairs = [
        (i, j)
        for i, j in combinations(range(n), 2)
        if nx_graphs[i] is not None and nx_graphs[j] is not None
    ]

    if len(valid_pairs) == 0:
        return ExactGEDResult(
            d_bar=0.0,
            diameter=0.0,
            n_exact=0,
            n_sampled=0,
            values=np.array([]),
            pair_indices=np.array([]).reshape(0, 2),
            exact_flags=np.array([], dtype=bool),
        )

    n_sample = min(n_subsample, len(valid_pairs))
    sample_idx = rng.choice(len(valid_pairs), size=n_sample, replace=False)
    sampled_pairs = [valid_pairs[k] for k in sample_idx]

    # Build work items
    work = [(i, j, nx_graphs[i], nx_graphs[j], timeout_per_pair) for i, j in sampled_pairs]

    # Compute GED
    results = []
    if n_workers <= 1:
        for item in work:
            results.append(_compute_exact_ged_pair(item))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_compute_exact_ged_pair, work, chunksize=32))

    # Unpack
    values = np.array([d for _, _, d, _ in results], dtype=np.float32)
    exact_flags = np.array([ex for _, _, _, ex in results], dtype=bool)
    pair_indices = np.array([(i, j) for i, j, _, _ in results], dtype=np.int32)

    d_bar = float(np.mean(values)) if len(values) > 0 else 0.0
    diameter = float(np.max(values)) if len(values) > 0 else 0.0
    n_exact = int(np.sum(exact_flags))

    return ExactGEDResult(
        d_bar=d_bar,
        diameter=diameter,
        n_exact=n_exact,
        n_sampled=n_sample,
        values=values,
        pair_indices=pair_indices,
        exact_flags=exact_flags,
    )


# ======================================================================
# WL hash feature extraction
# ======================================================================


def compute_wl_hashes(dag: LabeledDAG) -> list[int]:
    """Compute WL subtree hashes for all nodes in a DAG.

    Replicates _compute_subtree_hashes from canonical.py:670 but exposed
    as a public function. Bottom-up processing: nodes with no out-edges
    (expression roots) first, then backward toward input variables.

    Args:
        dag: LabeledDAG instance.

    Returns:
        List of integer hashes indexed by node ID.
    """
    from collections import deque

    n = dag.node_count
    node_hash = [0] * n
    out_deg = [len(dag.out_neighbors_raw(u)) for u in range(n)]

    queue = deque(u for u in range(n) if out_deg[u] == 0)
    processed = [False] * n

    while queue:
        u = queue.popleft()
        if processed[u]:
            continue
        processed[u] = True
        children_hashes = sorted(node_hash[v] for v in dag.out_neighbors_raw(u))
        node_hash[u] = hash((dag.node_label_unchecked(u), tuple(children_hashes)))
        for v in dag.in_neighbors_raw(u):
            out_deg[v] -= 1
            if out_deg[v] == 0 and not processed[v]:
                queue.append(v)

    return node_hash


def compute_wl_features(dags: list[LabeledDAG | None]) -> np.ndarray:
    """Compute WL hash histogram feature vectors for a population.

    For each DAG, computes WL subtree hashes and builds a frequency
    histogram. All unique hashes across the population form the feature
    dimensions.

    Args:
        dags: N DAGs (None entries get zero vectors).

    Returns:
        (N, d) float32 matrix where d = number of unique WL hashes.
    """
    n = len(dags)
    # Compute per-DAG hash lists
    hash_lists: list[list[int]] = []
    all_hashes: set[int] = set()
    for dag in dags:
        if dag is None:
            hash_lists.append([])
        else:
            hashes = compute_wl_hashes(dag)
            hash_lists.append(hashes)
            all_hashes.update(hashes)

    if not all_hashes:
        return np.zeros((n, 1), dtype=np.float32)

    # Build hash -> column index mapping
    hash_to_col = {h: idx for idx, h in enumerate(sorted(all_hashes))}
    d = len(hash_to_col)

    # Build feature matrix
    features = np.zeros((n, d), dtype=np.float32)
    for i, hashes in enumerate(hash_lists):
        for h in hashes:
            features[i, hash_to_col[h]] += 1.0

    return features


# ======================================================================
# PCA projection
# ======================================================================


def compute_pca_projection(
    features: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Project feature matrix to 2D via PCA.

    Args:
        features: (N, d) feature matrix.

    Returns:
        ((N, 2) coordinates, (var_ratio_1, var_ratio_2)).
    """
    if features.shape[1] < 2:
        # Pad with zeros to have at least 2 dimensions
        features = np.hstack([features, np.zeros((features.shape[0], 2 - features.shape[1]))])

    n_components = min(2, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(features)

    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 2 - coords.shape[1]))])

    var_ratios = pca.explained_variance_ratio_
    v1 = float(var_ratios[0]) if len(var_ratios) > 0 else 0.0
    v2 = float(var_ratios[1]) if len(var_ratios) > 1 else 0.0

    return coords, (v1, v2)


# ======================================================================
# Population snapshot capture
# ======================================================================


def _safe_agraph_to_dag(agraph) -> LabeledDAG | None:
    """Convert AGraph to LabeledDAG, returning None on failure."""
    from experiments.models.bingo.adapter import agraph_to_labeled_dag

    try:
        return agraph_to_labeled_dag(agraph)
    except Exception:
        return None


def _safe_canonical(dag: LabeledDAG | None) -> str:
    """Compute canonical string, returning a unique fallback on failure."""
    if dag is None:
        return f"__FAILED_{id(dag)}__"
    try:
        return fast_canonical_string(dag, timeout=30.0)
    except Exception:
        return f"__FAILED_{id(dag)}__"


def _extract_agraph_data(agraph) -> tuple[np.ndarray, np.ndarray]:
    """Extract command_array and constants from an AGraph."""
    try:
        cmd = np.array(agraph.command_array, dtype=np.int32)
    except Exception:
        cmd = np.zeros((1, 3), dtype=np.int32)
    try:
        cst = np.asarray(agraph.constants, dtype=np.float64).flatten()
    except Exception:
        cst = np.array([], dtype=np.float64)
    return cmd, cst


def _compute_best_r2(fitnesses: np.ndarray, population, x_train, y_train) -> float:
    """Compute best R^2 on training data from the fittest individual."""
    valid = np.isfinite(fitnesses)
    if not valid.any():
        return float("nan")
    best_idx = int(np.argmin(np.where(valid, fitnesses, np.inf)))
    try:
        y_pred = population[best_idx].evaluate_equation_at(x_train).flatten()
        ss_res = np.sum((y_train - y_pred) ** 2)
        ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
        if ss_tot < 1e-15:
            return 1.0 if ss_res < 1e-15 else 0.0
        return float(1.0 - ss_res / ss_tot)
    except Exception:
        return float("nan")


def capture_population_snapshot(
    population: list,
    generation: int,
    variant: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_workers: int = 1,
    ged_mode: str = "bp",
    ged_timeout: float = 5.0,
    ged_subsample: int = 2000,
) -> tuple[PopulationSnapshot, DiversitySummary]:
    """Capture full population snapshot with all diversity metrics.

    Args:
        population: List of Bingo AGraph individuals.
        generation: Current generation number.
        variant: "baseline" or "isalsr".
        seed: Random seed for this run.
        x_train: Training inputs (n_samples, n_features).
        y_train: Training targets (n_samples,).
        n_workers: Workers for parallel GED computation.
        ged_mode: "bp" (bipartite only), "exact" (subsampled exact + bipartite).
        ged_timeout: Per-pair timeout for exact GED (seconds).
        ged_subsample: Number of random pairs for exact GED subsampling.

    Returns:
        (PopulationSnapshot, DiversitySummary) tuple.
    """
    n = len(population)
    log.debug(
        "Capturing snapshot: gen=%d variant=%s N=%d mode=%s", generation, variant, n, ged_mode
    )

    # 1. Convert AGraphs to LabeledDAGs
    dags: list[LabeledDAG | None] = [_safe_agraph_to_dag(ag) for ag in population]
    n_failed = sum(1 for d in dags if d is None)
    if n_failed > 0:
        log.warning("  %d/%d AGraph->DAG conversions failed", n_failed, n)

    # 2. Compute canonical strings
    canonical_strings = [_safe_canonical(d) for d in dags]

    # 3. Effective diversity ratio
    delta, n_unique = compute_delta(canonical_strings)

    # 4. Levenshtein distance matrix (full NxN, fast: ~200ms for N=200)
    lev_mat = compute_levenshtein_matrix(canonical_strings)

    # 5. Bipartite GED matrix (full NxN, fast: ~1s with 8 workers for N=200)
    graph_data = []
    for d in dags:
        if d is not None:
            graph_data.append(_dag_to_nx_data(d))
        else:
            graph_data.append({"n": 0, "labels": [], "in_deg": [], "out_deg": [], "edges": set()})
    bp_ged_mat = compute_bipartite_ged_matrix(graph_data, n_workers=n_workers)

    # 5b. Exact GED on SUBSAMPLED pairs (if requested)
    exact_result: ExactGEDResult | None = None
    if ged_mode == "exact":
        log.info(
            "  Computing exact GED (subsample=%d, timeout=%.1fs/pair, workers=%d)...",
            ged_subsample,
            ged_timeout,
            n_workers,
        )
        exact_result = compute_exact_ged_subsample(
            dags,
            n_subsample=ged_subsample,
            timeout_per_pair=ged_timeout,
            n_workers=n_workers,
            rng_seed=seed * 1000 + generation,
        )

    # 6. WL features
    wl_features = compute_wl_features(dags)

    # 7. PCA projection
    pca_coords, pca_var = compute_pca_projection(wl_features)

    # 8. Fitness and R^2
    fitnesses = np.array(
        [ag.fitness if hasattr(ag, "fitness") and ag.fit_set else np.inf for ag in population],
        dtype=np.float64,
    )
    best_fitness = float(np.nanmin(fitnesses)) if np.any(np.isfinite(fitnesses)) else float("inf")
    best_r2 = _compute_best_r2(fitnesses, population, x_train, y_train)

    # 9. Extract AGraph raw data for reproducibility
    cmd_arrays = []
    const_arrays = []
    for ag in population:
        cmd, cst = _extract_agraph_data(ag)
        cmd_arrays.append(cmd)
        const_arrays.append(cst)

    # 10. Summary statistics
    n_pairs = n * (n - 1) // 2
    upper_tri_lev = lev_mat[np.triu_indices(n, k=1)]
    upper_tri_bp = bp_ged_mat[np.triu_indices(n, k=1)]

    d_bar_lev = float(np.mean(upper_tri_lev)) if len(upper_tri_lev) > 0 else 0.0
    d_bar_bp = float(np.mean(upper_tri_bp)) if len(upper_tri_bp) > 0 else 0.0
    diameter_lev = float(np.max(upper_tri_lev)) if len(upper_tri_lev) > 0 else 0.0
    diameter_bp = float(np.max(upper_tri_bp)) if len(upper_tri_bp) > 0 else 0.0

    d_bar_exact = -1.0
    diameter_exact = -1.0
    n_exact_pairs = 0
    n_total_sampled = 0
    if exact_result is not None:
        d_bar_exact = exact_result.d_bar
        diameter_exact = exact_result.diameter
        n_exact_pairs = exact_result.n_exact
        n_total_sampled = exact_result.n_sampled

    snapshot = PopulationSnapshot(
        generation=generation,
        variant=variant,
        seed=seed,
        canonical_strings=canonical_strings,
        fitnesses=fitnesses,
        wl_features=wl_features,
        lev_distances=lev_mat,
        bp_ged_distances=bp_ged_mat,
        exact_ged_distances=None,  # full NxN not stored for subsampled
        exact_ged_mask=None,
        pca_coords=pca_coords,
        pca_explained_variance=pca_var,
        command_arrays=cmd_arrays,
        constants=const_arrays,
    )
    # Attach subsampled exact GED result for serialization
    snapshot._exact_result = exact_result  # type: ignore[attr-defined]

    summary = DiversitySummary(
        generation=generation,
        variant=variant,
        seed=seed,
        delta=delta,
        d_bar_lev=d_bar_lev,
        d_bar_bp=d_bar_bp,
        d_bar_exact=d_bar_exact,
        diameter_lev=diameter_lev,
        diameter_bp=diameter_bp,
        diameter_exact=diameter_exact,
        best_fitness=best_fitness,
        best_r2=best_r2,
        n_unique_classes=n_unique,
        pca_var1=pca_var[0],
        pca_var2=pca_var[1],
        n_exact_pairs=n_exact_pairs,
        n_total_pairs=n_total_sampled,
        cv_ged=compute_cv_ged(bp_ged_mat),
        frac_zero_pairs=compute_frac_zero_pairs(bp_ged_mat),
        d_bar_nonzero=compute_d_bar_nonzero(bp_ged_mat),
    )

    return snapshot, summary
