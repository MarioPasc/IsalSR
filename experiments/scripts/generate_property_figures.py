"""Generate visualizations for IsalSR's 5 fundamental mathematical properties.

Each figure empirically demonstrates a key property of the IsalSR representation:
  P1: Round-trip fidelity (S2D(D2S(S2D(w))) ~ S2D(w))
  P2: DAG acyclicity (all S2D outputs are acyclic)
  P3: Canonical invariance (isomorphic DAGs -> same canonical string)
  P4: Evaluation preservation (eval before == eval after canonicalization)
  P5: Search space reduction (canonical deduplication)

Usage:
    python experiments/scripts/generate_property_figures.py

Outputs 5 PNG files in docs/technical_report/figures/
"""

from __future__ import annotations

import logging
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from isalsr.core.canonical import pruned_canonical_string  # noqa: E402
from isalsr.core.dag_evaluator import evaluate_dag  # noqa: E402
from isalsr.core.dag_to_string import DAGToString  # noqa: E402
from isalsr.core.node_types import OperationSet  # noqa: E402
from isalsr.core.string_to_dag import StringToDAG  # noqa: E402
from isalsr.search.operators import detokenize, random_token, tokenize  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FIGURE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "docs", "technical_report", "figures"
)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Paul Tol colorblind-safe palette
C_BLUE = "#4477AA"
C_CYAN = "#66CCEE"
C_GREEN = "#228833"
C_YELLOW = "#CCBB44"
C_RED = "#EE6677"
C_PURPLE = "#AA3377"
C_GREY = "#BBBBBB"

N_STRINGS = 300  # Moderate N for reasonable runtime
MAX_TOKENS = 8  # Keep small for fast canonical computation
SEED = 42


def _random_string(nv: int, rng: np.random.Generator) -> str:
    ops = OperationSet()
    n = int(rng.integers(1, MAX_TOKENS + 1))
    return detokenize([random_token(ops, rng) for _ in range(n)])


# ======================================================================
# P1: Round-Trip Fidelity
# ======================================================================
def property_roundtrip() -> None:
    """S2D(D2S(S2D(w))) ~ S2D(w) for all valid w."""
    log.info("P1: Round-Trip Fidelity")
    rng = np.random.default_rng(SEED)

    nodes_before: list[int] = []
    nodes_after: list[int] = []
    edges_before: list[int] = []
    edges_after: list[int] = []
    successes = 0
    failures = 0
    total_tested = 0

    for _ in range(N_STRINGS):
        w = _random_string(1, rng)
        try:
            dag1 = StringToDAG(w, 1).run()
            if dag1.node_count <= 1:
                continue
            total_tested += 1
            w2 = DAGToString(dag1).run()
            dag2 = StringToDAG(w2, 1).run()

            nodes_before.append(dag1.node_count)
            nodes_after.append(dag2.node_count)
            edges_before.append(dag1.edge_count)
            edges_after.append(dag2.edge_count)

            if dag1.is_isomorphic(dag2):
                successes += 1
            else:
                failures += 1
        except Exception:
            continue

    log.info("  Tested: %d, Success: %d, Failure: %d", total_tested, successes, failures)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Left: node count scatter
    axes[0].scatter(nodes_before, nodes_after, alpha=0.5, s=20, c=C_BLUE, edgecolors="none")
    lim = max(max(nodes_before, default=1), max(nodes_after, default=1)) + 1
    axes[0].plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    axes[0].set_xlabel("Node count (before)")
    axes[0].set_ylabel("Node count (after)")
    axes[0].set_title("Node Count Preservation")
    axes[0].set_aspect("equal")

    # Center: edge count scatter
    axes[1].scatter(edges_before, edges_after, alpha=0.5, s=20, c=C_GREEN, edgecolors="none")
    lim = max(max(edges_before, default=1), max(edges_after, default=1)) + 1
    axes[1].plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    axes[1].set_xlabel("Edge count (before)")
    axes[1].set_ylabel("Edge count (after)")
    axes[1].set_title("Edge Count Preservation")
    axes[1].set_aspect("equal")

    # Right: success rate
    rate = successes / max(total_tested, 1) * 100
    bars = axes[2].bar(
        ["Isomorphic\n(success)", "Not isomorphic\n(failure)"],
        [successes, failures],
        color=[C_GREEN, C_RED],
    )
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Round-Trip Success: {rate:.1f}%")
    for bar, val in zip(bars, [successes, failures]):
        if val > 0:
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(val),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    fig.suptitle(
        "Property 1: Round-Trip Fidelity — S2D(D2S(S2D(w))) ≅ S2D(w)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "P1_roundtrip_fidelity.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info("  Saved: %s", path)


# ======================================================================
# P2: DAG Acyclicity
# ======================================================================
def property_acyclicity() -> None:
    """All DAGs produced by S2D are acyclic."""
    log.info("P2: DAG Acyclicity")
    rng = np.random.default_rng(SEED)

    acyclic_count = 0
    cyclic_count = 0
    string_lengths: list[int] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    c_instruction_counts: list[int] = []

    for _ in range(N_STRINGS):
        w = _random_string(2, rng)  # 2 vars = more C/c opportunities
        try:
            dag = StringToDAG(w, 2).run()
            topo = dag.topological_sort()
            if len(topo) == dag.node_count:
                acyclic_count += 1
            else:
                cyclic_count += 1
            tokens = tokenize(w)
            string_lengths.append(len(tokens))
            node_counts.append(dag.node_count)
            edge_counts.append(dag.edge_count)
            c_count = sum(1 for t in tokens if t in ("C", "c"))
            c_instruction_counts.append(c_count)
        except Exception:
            continue

    total = acyclic_count + cyclic_count
    log.info("  Acyclic: %d/%d (%.1f%%)", acyclic_count, total, 100 * acyclic_count / max(total, 1))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Left: acyclicity rate
    axes[0].bar(
        ["Acyclic\n(valid DAG)", "Cyclic\n(impossible)"],
        [acyclic_count, cyclic_count],
        color=[C_GREEN, C_RED],
    )
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"DAG Acyclicity: {100 * acyclic_count / max(total, 1):.0f}%")
    axes[0].text(
        0,
        acyclic_count + 2,
        str(acyclic_count),
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=C_GREEN,
    )

    # Center: distribution of C/c instructions
    axes[1].hist(
        c_instruction_counts,
        bins=range(0, max(c_instruction_counts, default=5) + 2),
        color=C_CYAN,
        edgecolor="white",
        alpha=0.8,
    )
    axes[1].set_xlabel("Number of C/c edge instructions per string")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Edge Instruction Distribution")

    # Right: node vs edge count (all acyclic)
    axes[2].scatter(node_counts, edge_counts, alpha=0.4, s=15, c=C_BLUE, edgecolors="none")
    axes[2].set_xlabel("Node count")
    axes[2].set_ylabel("Edge count")
    axes[2].set_title("DAG Complexity Distribution")

    fig.suptitle(
        "Property 2: DAG Acyclicity — All S2D outputs are acyclic", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "P2_dag_acyclicity.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info("  Saved: %s", path)


# ======================================================================
# P3: Canonical Invariance
# ======================================================================
def property_canonical_invariance() -> None:
    """Isomorphic DAGs produce identical canonical strings."""
    log.info("P3: Canonical Invariance")
    rng = np.random.default_rng(SEED)

    # Build pairs: (dag_original, dag_relabeled, is_isomorphic, canonical_match)
    iso_and_match = 0  # isomorphic AND same canonical
    iso_not_match = 0  # isomorphic BUT different canonical (BUG!)
    not_iso_and_diff = 0  # not isomorphic AND different canonical
    not_iso_but_match = 0  # not isomorphic BUT same canonical (BUG!)

    tested = 0
    for _ in range(N_STRINGS):
        w = _random_string(1, rng)
        try:
            dag1 = StringToDAG(w, 1).run()
            if dag1.node_count <= 1:
                continue

            # Create isomorphic pair: same DAG via D2S roundtrip
            w2 = DAGToString(dag1).run()
            dag2 = StringToDAG(w2, 1).run()

            canon1 = pruned_canonical_string(dag1)
            canon2 = pruned_canonical_string(dag2)
            is_iso = dag1.is_isomorphic(dag2)
            canons_match = canon1 == canon2

            if is_iso and canons_match:
                iso_and_match += 1
            elif is_iso and not canons_match:
                iso_not_match += 1
            elif not is_iso and not canons_match:
                not_iso_and_diff += 1
            else:
                not_iso_but_match += 1
            tested += 1
        except Exception:
            continue

    # Also test non-isomorphic pairs
    for _ in range(N_STRINGS // 2):
        w1 = _random_string(1, rng)
        w2 = _random_string(1, rng)
        try:
            dag1 = StringToDAG(w1, 1).run()
            dag2 = StringToDAG(w2, 1).run()
            if dag1.node_count <= 1 or dag2.node_count <= 1:
                continue

            canon1 = pruned_canonical_string(dag1)
            canon2 = pruned_canonical_string(dag2)
            is_iso = dag1.is_isomorphic(dag2)
            canons_match = canon1 == canon2

            if is_iso and canons_match:
                iso_and_match += 1
            elif is_iso and not canons_match:
                iso_not_match += 1
            elif not is_iso and not canons_match:
                not_iso_and_diff += 1
            else:
                not_iso_but_match += 1
            tested += 1
        except Exception:
            continue

    log.info("  Tested: %d pairs", tested)
    log.info(
        "  Iso+Match: %d, Iso+NoMatch: %d, NotIso+Diff: %d, NotIso+Match: %d",
        iso_and_match,
        iso_not_match,
        not_iso_and_diff,
        not_iso_but_match,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Confusion matrix
    matrix = np.array([[iso_and_match, iso_not_match], [not_iso_but_match, not_iso_and_diff]])
    im = axes[0].imshow(matrix, cmap="RdYlGn", aspect="auto")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Canonical\nMATCH", "Canonical\nDIFFER"])
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Isomorphic\n(true)", "Not isomorphic\n(true)"])
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            axes[0].text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=color,
            )
    axes[0].set_title("Canonical ↔ Isomorphism Agreement")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # Right: Summary bar
    correct = iso_and_match + not_iso_and_diff
    incorrect = iso_not_match + not_iso_but_match
    accuracy = 100 * correct / max(tested, 1)
    bars = axes[1].bar(
        ["Correct\n(diagonal)", "Incorrect\n(off-diagonal)"],
        [correct, incorrect],
        color=[C_GREEN, C_RED],
    )
    axes[1].set_ylabel("Pair count")
    axes[1].set_title(f"Invariant Accuracy: {accuracy:.1f}%")
    for bar, val in zip(bars, [correct, incorrect]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(val),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    fig.suptitle(
        "Property 3: Canonical Invariance — canonical(D₁) = canonical(D₂) iff D₁ ≅ D₂",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "P3_canonical_invariance.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info("  Saved: %s", path)


# ======================================================================
# P4: Evaluation Preservation
# ======================================================================
def property_evaluation_preservation() -> None:
    """eval(D, x) == eval(S2D(canonical(D)), x) for all inputs."""
    log.info("P4: Evaluation Preservation")
    rng = np.random.default_rng(SEED)

    eval_original: list[float] = []
    eval_canonical: list[float] = []
    errors: list[float] = []

    for _ in range(N_STRINGS):
        w = _random_string(1, rng)
        try:
            dag1 = StringToDAG(w, 1).run()
            if dag1.node_count <= 1:
                continue

            # Evaluate at 5 random input points
            for _ in range(5):
                x_val = float(rng.uniform(-2, 2))
                try:
                    v1 = evaluate_dag(dag1, {0: x_val})
                except Exception:
                    continue

                canon = pruned_canonical_string(dag1)
                dag2 = StringToDAG(canon, 1).run()
                try:
                    v2 = evaluate_dag(dag2, {0: x_val})
                except Exception:
                    continue

                if math.isfinite(v1) and math.isfinite(v2):
                    eval_original.append(v1)
                    eval_canonical.append(v2)
                    errors.append(abs(v1 - v2))
        except Exception:
            continue

    log.info("  Evaluated: %d points, Max error: %.2e", len(errors), max(errors) if errors else 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Left: original vs canonical scatter
    eo = np.array(eval_original)
    ec = np.array(eval_canonical)
    # Clip for visualization
    clip = 50
    mask = (np.abs(eo) < clip) & (np.abs(ec) < clip)
    axes[0].scatter(eo[mask], ec[mask], alpha=0.3, s=8, c=C_BLUE, edgecolors="none")
    lim = max(np.abs(eo[mask]).max(), np.abs(ec[mask]).max()) if mask.any() else 10
    axes[0].plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
    axes[0].set_xlabel("eval(original)")
    axes[0].set_ylabel("eval(canonical)")
    axes[0].set_title("Evaluation: Original vs Canonical")

    # Center: Bland-Altman (difference vs mean)
    means = (eo[mask] + ec[mask]) / 2
    diffs = eo[mask] - ec[mask]
    axes[1].scatter(means, diffs, alpha=0.3, s=8, c=C_PURPLE, edgecolors="none")
    axes[1].axhline(y=0, color="k", linestyle="--", lw=0.8)
    axes[1].set_xlabel("Mean of evaluations")
    axes[1].set_ylabel("Difference (original - canonical)")
    axes[1].set_title("Bland-Altman: Should be flat at y=0")

    # Right: error histogram
    errs = np.array(errors)
    axes[2].hist(
        errs[errs < 1e-8],
        bins=50,
        color=C_GREEN,
        edgecolor="white",
        alpha=0.8,
        label=f"< 1e-8: {(errs < 1e-8).sum()}",
    )
    axes[2].hist(
        errs[errs >= 1e-8],
        bins=20,
        color=C_RED,
        edgecolor="white",
        alpha=0.8,
        label=f"≥ 1e-8: {(errs >= 1e-8).sum()}",
    )
    axes[2].set_xlabel("Absolute error |eval_orig - eval_canon|")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Error Distribution (max: {errs.max():.2e})")
    axes[2].legend(fontsize=8)

    fig.suptitle(
        "Property 4: Evaluation Preservation — eval(D, x) = eval(S2D(canonical(D)), x)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "P4_evaluation_preservation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info("  Saved: %s", path)


# ======================================================================
# P5: Search Space Reduction
# ======================================================================
def property_search_space_reduction() -> None:
    """Canonical deduplication reduces search space by collapsing O(k!) equivalents."""
    log.info("P5: Search Space Reduction")
    rng = np.random.default_rng(SEED)

    num_vars_list = [1, 1, 1, 2, 2, 2]
    benchmark_names = [
        "1-var (short)",
        "1-var (med)",
        "1-var (long)",
        "2-var (short)",
        "2-var (med)",
        "2-var (long)",
    ]
    max_tok_list = [5, 8, 12, 5, 8, 12]

    total_valid: list[int] = []
    unique_raw: list[int] = []
    unique_canonical: list[int] = []
    avg_k_vals: list[float] = []

    for nv, max_tok, name in zip(num_vars_list, max_tok_list, benchmark_names):
        canon_set: set[str] = set()
        raw_set: set[str] = set()
        valid = 0
        k_values: list[int] = []

        for _ in range(N_STRINGS):
            ops = OperationSet()
            n = int(rng.integers(1, max_tok + 1))
            w = detokenize([random_token(ops, rng) for _ in range(n)])
            try:
                dag = StringToDAG(w, nv).run()
                if dag.node_count <= nv:
                    continue
                valid += 1
                # Raw deduplication (by greedy D2S string)
                greedy_str = DAGToString(dag).run()
                raw_set.add(greedy_str)
                # Canonical deduplication
                canon = pruned_canonical_string(dag)
                canon_set.add(canon)
                k_values.append(dag.node_count - nv)
            except Exception:
                continue

        total_valid.append(valid)
        unique_raw.append(len(raw_set))
        unique_canonical.append(len(canon_set))
        avg_k_vals.append(float(np.mean(k_values)) if k_values else 0)
        log.info(
            "  %s: %d valid, %d unique_raw, %d unique_canon (k=%.1f)",
            name,
            valid,
            len(raw_set),
            len(canon_set),
            float(np.mean(k_values)) if k_values else 0,
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: grouped bar chart
    x = np.arange(len(benchmark_names))
    w_bar = 0.25
    axes[0].bar(x - w_bar, total_valid, w_bar, label="Total valid", color=C_BLUE, alpha=0.8)
    axes[0].bar(x, unique_raw, w_bar, label="Unique (greedy D2S)", color=C_CYAN, alpha=0.8)
    axes[0].bar(
        x + w_bar, unique_canonical, w_bar, label="Unique (canonical)", color=C_GREEN, alpha=0.8
    )
    axes[0].set_xlabel("Configuration")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Search Space Size: Total vs Deduplicated")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(benchmark_names, rotation=30, ha="right", fontsize=8)
    axes[0].legend(fontsize=8)

    # Right: reduction factor
    reduction_greedy = [t / max(u, 1) for t, u in zip(total_valid, unique_raw)]
    reduction_canon = [t / max(u, 1) for t, u in zip(total_valid, unique_canonical)]
    axes[1].bar(x - 0.15, reduction_greedy, 0.3, label="Greedy D2S dedup", color=C_CYAN, alpha=0.8)
    axes[1].bar(x + 0.15, reduction_canon, 0.3, label="Canonical dedup", color=C_GREEN, alpha=0.8)
    axes[1].set_xlabel("Configuration")
    axes[1].set_ylabel("Reduction factor (total / unique)")
    axes[1].set_title("Deduplication Effectiveness")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(benchmark_names, rotation=30, ha="right", fontsize=8)
    axes[1].legend(fontsize=8)
    for i, (rg, rc, k) in enumerate(zip(reduction_greedy, reduction_canon, avg_k_vals)):
        axes[1].text(i + 0.15, rc + 0.05, f"k̄={k:.1f}", ha="center", fontsize=7, color=C_GREEN)

    fig.suptitle(
        "Property 5: Search Space Reduction — O(k!) deduplication via canonicalization",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "P5_search_space_reduction.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info("  Saved: %s", path)


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    """Generate all 5 property visualization figures."""
    log.info("=" * 60)
    log.info("IsalSR Fundamental Property Visualizations")
    log.info("N_STRINGS=%d, MAX_TOKENS=%d, SEED=%d", N_STRINGS, MAX_TOKENS, SEED)
    log.info("Output: %s", FIGURE_DIR)
    log.info("=" * 60)

    property_roundtrip()
    property_acyclicity()
    property_canonical_invariance()
    property_evaluation_preservation()
    property_search_space_reduction()

    log.info("=" * 60)
    log.info("All 5 property figures generated successfully!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
