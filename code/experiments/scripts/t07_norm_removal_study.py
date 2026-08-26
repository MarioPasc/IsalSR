"""T07: Two-arm study — keep vs drop of 𝒩 (normalize_const_creation) in canonicalisation.

Two arms evaluated on identical DAG streams:

  keep  fast_canonical_string(dag, timeout, backend="cpp")     [production: 𝒩 applied in C++]
  drop  _native.testing.fast_canonical_string_raw(native, t)   [no normalization]

On adapter-sourced DAGs (bingo/udfs) the adapters repair orphan CONSTs before
the canonicaliser receives the DAG.  𝒩 is therefore the identity on those DAGs
and both arms are predicted to agree on 100 % of calls.  The study confirms or
refutes that — it does not assume it.

On adversarial DAGs (constructed to satisfy all three 𝒩-inequivariance
conditions) the keep arm gives silent wrong answers (equivariance failures)
while the drop arm gives loud refusals (RuntimeError).

Populations
-----------
  synthetic   Random S2D-decoded DAGs.  num_variables drawn from {1, 2, 3}.
  adversarial Constructed DAGs hitting the 𝒩-inequivariance counterexample.
  bingo       Live Bingo search output (adapter-sourced DAGs).
  udfs        Live UDFS search output (adapter-sourced DAGs).

Usage (smoke test, < 3 min locally)
-------------------------------------
    python -m experiments.scripts.t07_norm_removal_study \\
        --population synthetic --n 2000 --out /tmp/t07_smoke

Usage (Picasso, per-task array)
--------------------------------
    python -m experiments.scripts.t07_norm_removal_study \\
        --population synthetic --n 100000 --seed 31 --task-id 0 \\
        --out /results/t07/synthetic

    python -m experiments.scripts.t07_norm_removal_study \\
        --population bingo --suite nguyen --problem Nguyen-5 --seed 1 \\
        --max-time 600 --out /results/t07/bingo/nguyen_Nguyen-5_seed1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType
from isalsr.core.permutations import permute_internal_nodes
from isalsr.core.string_to_dag import StringToDAG

log = logging.getLogger("t07.study")

ARMS: tuple[str, str] = ("keep", "drop")

# Random-string alphabet reused from validate_const_repair_synthetic.py
ALPHABET: list[str] = list("NPnpCcW") + [f"{ptr}{lbl}" for ptr in "Vv" for lbl in "+*-/scelr^agik"]

Outcome = Literal["ok", "raised", "timeout"]
ScoreFn = Callable[[LabeledDAG], tuple[Outcome, str | None]]


# ---------------------------------------------------------------------------
# Per-arm statistics accumulator
# ---------------------------------------------------------------------------


@dataclass
class ArmStats:
    """Aggregate statistics for one arm over a DAG population.

    Args:
        round_trip_comparator: Human-readable label for the round-trip method.
    """

    round_trip_comparator: str = ""
    n_total: int = 0
    n_ok: int = 0
    n_raised: int = 0
    n_timeout: int = 0
    _seen_hashes: set[int] = field(default_factory=set, repr=False)
    n_reachable: int = 0
    n_round_trip_ok: int = 0
    n_round_trip_checked: int = 0
    n_equivariance_samples: int = 0
    n_equivariance_failures: int = 0
    # k -> {n_total, n_ok, n_raised, n_timeout, n_unique (_seen set dropped in JSON)}
    per_k: dict[int, dict[str, Any]] = field(default_factory=dict)

    def record(
        self,
        outcome: Outcome,
        canonical: str | None,
        k: int,
        reachable: bool,
    ) -> None:
        """Record one DAG canonicalisation outcome.

        Args:
            outcome: Result category.
            canonical: Canonical string (None when outcome != 'ok').
            k: Number of internal nodes in the DAG.
            reachable: True if all non-VAR nodes are reachable from VAR nodes.
        """
        self.n_total += 1
        if reachable:
            self.n_reachable += 1

        if outcome == "ok":
            self.n_ok += 1
            assert canonical is not None
            self._seen_hashes.add(hash(canonical))
        elif outcome == "raised":
            self.n_raised += 1
        else:
            self.n_timeout += 1

        if k not in self.per_k:
            self.per_k[k] = {
                "n_total": 0,
                "n_ok": 0,
                "n_raised": 0,
                "n_timeout": 0,
                "_seen": set(),
            }
        pk = self.per_k[k]
        pk["n_total"] += 1
        if outcome == "ok" and canonical is not None:
            pk["n_ok"] += 1
            pk["_seen"].add(hash(canonical))
        elif outcome == "raised":
            pk["n_raised"] += 1
        else:
            pk["n_timeout"] += 1

    @property
    def n_unique(self) -> int:
        """Distinct canonical strings seen."""
        return len(self._seen_hashes)

    @property
    def rho(self) -> float:
        """Reduction factor: n_ok / n_unique."""
        return self.n_ok / self.n_unique if self.n_unique else float("nan")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        per_k_out: dict[str, dict[str, Any]] = {}
        for k in sorted(self.per_k):
            pk = self.per_k[k]
            n_ok_k = pk.get("n_ok", 0)
            seen_k: set[int] = pk.get("_seen", set())
            n_unique_k = len(seen_k)
            per_k_out[str(k)] = {
                "n_total": pk.get("n_total", 0),
                "n_ok": n_ok_k,
                "n_raised": pk.get("n_raised", 0),
                "n_timeout": pk.get("n_timeout", 0),
                "n_unique": n_unique_k,
                "rho": (n_ok_k / n_unique_k) if n_unique_k else float("nan"),
            }
        return {
            "n_total": self.n_total,
            "n_ok": self.n_ok,
            "n_raised": self.n_raised,
            "n_timeout": self.n_timeout,
            "n_unique": self.n_unique,
            "rho": self.rho,
            "n_reachable": self.n_reachable,
            "n_round_trip_ok": self.n_round_trip_ok,
            "n_round_trip_checked": self.n_round_trip_checked,
            "round_trip_comparator": self.round_trip_comparator,
            "n_equivariance_samples": self.n_equivariance_samples,
            "n_equivariance_failures": self.n_equivariance_failures,
            "per_k": per_k_out,
        }


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _num_vars(dag: LabeledDAG) -> int:
    """Count VAR nodes in *dag*.

    Args:
        dag: The labeled DAG to inspect.

    Returns:
        Number of nodes with label NodeType.VAR.
    """
    return sum(1 for i in range(dag.node_count) if dag.node_label(i) == NodeType.VAR)


def _num_internal(dag: LabeledDAG) -> int:
    """Count non-VAR (internal) nodes in *dag*.

    Args:
        dag: The labeled DAG to inspect.

    Returns:
        Number of nodes with label != NodeType.VAR.
    """
    return int(dag.node_count) - _num_vars(dag)


def _structural_key(
    dag: LabeledDAG,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Structural fingerprint: label values + ordered in-edges per node.

    Identical keys imply identical canonical strings when 𝒩 is the identity.
    Used as the round-trip comparator for the drop arm.

    Args:
        dag: The labeled DAG to fingerprint.

    Returns:
        Tuple of (label_value, ordered_inputs) per node.
    """
    return tuple(
        (dag.node_label(i).value, tuple(dag.ordered_inputs(i))) for i in range(dag.node_count)
    )


# ---------------------------------------------------------------------------
# Arm scoring
# ---------------------------------------------------------------------------


def _score_keep_with(
    fn: Callable[..., str],
    dag: LabeledDAG,
    timeout: float,
) -> tuple[Outcome, str | None]:
    """Score keep arm using a provided fast_canonical_string callable.

    Args:
        fn: The canonical-string function to call (production or saved original).
        dag: The DAG to canonicalise.
        timeout: Time budget in seconds.

    Returns:
        (outcome, canonical_string_or_None).
    """
    from isalsr.core.canonical import CanonicalTimeoutError

    try:
        s = fn(dag, timeout=timeout, backend="cpp")
        return "ok", s
    except CanonicalTimeoutError:
        return "timeout", None
    except Exception:  # noqa: BLE001
        return "raised", None


def _score_keep(dag: LabeledDAG, timeout: float) -> tuple[Outcome, str | None]:
    """Score the keep arm (standalone, for synthetic/adversarial populations).

    Calls ``fast_canonical_string(dag, timeout=timeout, backend='cpp')``.
    The production C++ engine applies 𝒩 internally.

    Args:
        dag: The DAG to canonicalise.
        timeout: Time budget in seconds.

    Returns:
        (outcome, canonical_string_or_None).
    """
    from isalsr.core.canonical import fast_canonical_string

    return _score_keep_with(fast_canonical_string, dag, timeout)


def _score_drop(dag: LabeledDAG, timeout: float) -> tuple[Outcome, str | None]:
    """Score the drop arm: raw canonicalisation with no normalization.

    Calls ``_native.testing.fast_canonical_string_raw`` directly.  This entry
    point skips 𝒩, so CONST nodes with in-degree 0 cause a RuntimeError.

    Args:
        dag: The DAG to canonicalise.
        timeout: Time budget in seconds.

    Returns:
        (outcome, canonical_string_or_None).
    """
    from isalsr.core import _native
    from isalsr.core.canonical import _py_dag_to_native

    raw = _native.testing.fast_canonical_string_raw
    try:
        s = raw(_py_dag_to_native(dag), timeout)
        return "ok", s
    except Exception as exc:  # noqa: BLE001
        if "timeout" in str(exc).lower():
            return "timeout", None
        return "raised", None


# ---------------------------------------------------------------------------
# Equivariance check
# ---------------------------------------------------------------------------


def _check_equivariance(
    dag: LabeledDAG,
    score_fn: ScoreFn,
    rng: random.Random,
    k_perms: int = 8,
) -> tuple[int, int]:
    """Check isomorphism-equivariance by applying random internal-node permutations.

    Draws ``k_perms`` seeded permutations of internal nodes, canonicalises each,
    and counts mismatches against the original.  A mismatch is a failure: either
    the string changed (silent wrong answer for keep) or the outcome changed
    (unexpected behaviour for drop).

    Args:
        dag: The DAG to check.
        score_fn: Arm-specific scoring function.
        rng: Random number generator.
        k_perms: Number of permutations to draw.

    Returns:
        (n_failures, k_perms_tried).  k_perms_tried == 0 if k < 2.
    """
    n_internal = _num_internal(dag)
    if n_internal < 2:
        return 0, 0  # only one permutation exists (identity) — trivially equivariant

    baseline = score_fn(dag)
    failures = 0
    for _ in range(k_perms):
        perm = list(range(n_internal))
        rng.shuffle(perm)
        dag2 = permute_internal_nodes(dag, perm)
        result = score_fn(dag2)
        if result != baseline:
            failures += 1
    return failures, k_perms


# ---------------------------------------------------------------------------
# Round-trip checks
# ---------------------------------------------------------------------------


def _round_trip_keep(
    keep_canonical: str,
    m: int,
    timeout: float,
    keep_fn: Callable[..., str] | None = None,
) -> bool:
    """Round-trip check for keep arm.

    Decodes *keep_canonical* via S2D, re-canonicalises with keep arm, and
    checks identity.  The comparator is ``fast_canonical_string``.

    ``keep_fn`` MUST be supplied whenever this runs inside a monkey-patched
    window (i.e. during a live bingo/udfs search).  Without it the fallback
    ``_score_keep`` re-imports ``isalsr.core.canonical.fast_canonical_string``
    *at call time*, which resolves to the installed ``TwoArmRecorder`` and
    re-enters it -- scoring the S2D-decoded string as though it were a fresh
    adapter DAG and inflating every counter.  That is the defect that
    contaminated the bingo/udfs arms of the 2026-07-29 Picasso run.

    Args:
        keep_canonical: Canonical string produced by keep arm.
        m: Number of input variables for S2D decoding.
        timeout: Time budget for re-canonicalisation.
        keep_fn: Unpatched ``fast_canonical_string`` to use for re-scoring.

    Returns:
        True if cs(S2D(cs(D))) == cs(D).
    """
    try:
        dag2 = StringToDAG(keep_canonical, num_variables=m).run()
        if keep_fn is not None:
            outcome2, cs2 = _score_keep_with(keep_fn, dag2, timeout)
        else:
            outcome2, cs2 = _score_keep(dag2, timeout)
        return outcome2 == "ok" and cs2 == keep_canonical
    except Exception:  # noqa: BLE001
        return False


def _round_trip_drop(
    drop_canonical: str,
    keep_canonical: str,
    m: int,
    timeout: float,
    keep_fn: Callable[..., str] | None = None,
) -> bool:
    """Round-trip check for drop arm using the unified ``fast_canonical_string`` comparator.

    Decodes *drop_canonical* via S2D, re-canonicalises with ``fast_canonical_string``
    (the same comparator as the keep arm), and checks that the result equals
    *keep_canonical*.  Using a single comparator for both arms makes the two
    round-trip rates directly comparable: any difference is attributable to 𝒩
    and not to the comparator itself.

    Note: the previous implementation used ``_structural_key`` which compares
    absolute node IDs and therefore fails for isomorphic DAGs with different
    node orderings (e.g. after a permutation).  The unified comparator is
    correct for all DAGs.

    Args:
        drop_canonical: Canonical string produced by drop arm.
        keep_canonical: Canonical string produced by keep arm for the same DAG.
        m: Number of input variables for S2D decoding.
        timeout: Time budget for re-canonicalisation.

    Returns:
        True if fcs(S2D(drop_cs)) == keep_cs.
    """
    try:
        dag2 = StringToDAG(drop_canonical, num_variables=m).run()
        if keep_fn is not None:
            outcome2, cs2 = _score_keep_with(keep_fn, dag2, timeout)
        else:
            outcome2, cs2 = _score_keep(dag2, timeout)
        return outcome2 == "ok" and cs2 == keep_canonical
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Core per-DAG processing
# ---------------------------------------------------------------------------


def _process_one_dag(
    dag: LabeledDAG,
    timeout: float,
    rng: random.Random,
    sample_rate: int,
    k_perms: int,
    call_idx: int,
    stats: dict[str, ArmStats],
    comparison: dict[str, Any],
    is_adversarial: bool,
    keep_fn: Callable[..., str] | None,
) -> None:
    """Score one DAG on both arms, accumulate into *stats* and *comparison*.

    Args:
        dag: The DAG to evaluate.
        timeout: Canonicalisation time budget in seconds.
        rng: Random number generator.
        sample_rate: Equivariance check every 1-in-sample_rate DAGs (0 = off).
        k_perms: Permutations per equivariance check.
        call_idx: Sequential call index (used for subsample modulo).
        stats: Mutable {arm -> ArmStats} accumulator.
        comparison: Mutable cross-arm comparison counters.
        is_adversarial: If True, track silent-wrong vs loud-refusal counts.
        keep_fn: Optional override for keep arm (used in TwoArmRecorder to pass
            the saved original fast_canonical_string before monkey-patching).
    """
    from experiments.models.fallback_ledger import violates_precondition

    m = _num_vars(dag)
    k = _num_internal(dag)
    reachable = not violates_precondition(dag)

    if keep_fn is not None:
        keep_out, keep_str = _score_keep_with(keep_fn, dag, timeout)
    else:
        keep_out, keep_str = _score_keep(dag, timeout)
    drop_out, drop_str = _score_drop(dag, timeout)

    stats["keep"].record(keep_out, keep_str, k, reachable)
    stats["drop"].record(drop_out, drop_str, k, reachable)

    # Cross-arm agreement
    if keep_out == "ok" and drop_out == "ok":
        comparison["n_both_ok"] = comparison.get("n_both_ok", 0) + 1
        if keep_str == drop_str:
            comparison["n_agreement"] = comparison.get("n_agreement", 0) + 1

    # Round-trip
    if keep_out == "ok" and keep_str is not None:
        stats["keep"].n_round_trip_checked += 1
        if _round_trip_keep(keep_str, m, timeout, keep_fn):
            stats["keep"].n_round_trip_ok += 1
    if keep_out == "ok" and keep_str is not None and drop_out == "ok" and drop_str is not None:
        stats["drop"].n_round_trip_checked += 1
        if _round_trip_drop(drop_str, keep_str, m, timeout, keep_fn):
            stats["drop"].n_round_trip_ok += 1

    # Equivariance subsample (1-in-sample_rate by call index)
    if sample_rate > 0 and call_idx % sample_rate == 0:
        eff_timeout = timeout
        _kf_keep_fn = keep_fn  # capture for closure

        def _kf_keep(d: LabeledDAG) -> tuple[Outcome, str | None]:
            if _kf_keep_fn is not None:
                return _score_keep_with(_kf_keep_fn, d, eff_timeout)
            return _score_keep(d, eff_timeout)

        def _kf_drop(d: LabeledDAG) -> tuple[Outcome, str | None]:
            return _score_drop(d, eff_timeout)

        nf_k, tried_k = _check_equivariance(dag, _kf_keep, rng, k_perms)
        nf_d, tried_d = _check_equivariance(dag, _kf_drop, rng, k_perms)
        stats["keep"].n_equivariance_samples += tried_k
        stats["keep"].n_equivariance_failures += nf_k
        stats["drop"].n_equivariance_samples += tried_d
        stats["drop"].n_equivariance_failures += nf_d

    # Adversarial-specific: silent-wrong vs loud-refusal per DAG
    if is_adversarial:
        n_int = _num_internal(dag)
        if keep_out == "ok" and keep_str is not None and n_int >= 2:
            perm = list(range(n_int))
            rng.shuffle(perm)
            dag2 = permute_internal_nodes(dag, perm)
            if keep_fn is not None:
                _, ks2 = _score_keep_with(keep_fn, dag2, timeout)
            else:
                _, ks2 = _score_keep(dag2, timeout)
            if ks2 != keep_str:
                comparison["adversarial_keep_silent_wrong"] = (
                    comparison.get("adversarial_keep_silent_wrong", 0) + 1
                )
        if drop_out == "raised":
            comparison["adversarial_drop_loud_refusal"] = (
                comparison.get("adversarial_drop_loud_refusal", 0) + 1
            )


# ---------------------------------------------------------------------------
# Population: synthetic
# ---------------------------------------------------------------------------


def _iter_synthetic(
    n: int,
    rng: random.Random,
    num_variables_choices: tuple[int, ...] = (1, 2, 3),
) -> Iterator[LabeledDAG]:
    """Generate *n* random synthetic DAGs via S2D decoding of random strings.

    Strings that fail S2D decoding are silently skipped.

    Args:
        n: Number of random strings to attempt.
        rng: Random number generator.
        num_variables_choices: Variable counts to sample uniformly.

    Yields:
        Successfully decoded LabeledDAG instances.
    """
    for _ in range(n):
        word = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(4, 30)))
        m = rng.choice(num_variables_choices)
        try:
            yield StringToDAG(word, num_variables=m).run()
        except Exception:  # noqa: BLE001
            continue


# ---------------------------------------------------------------------------
# Population: adversarial
# ---------------------------------------------------------------------------


def _build_adversarial_dags() -> list[LabeledDAG]:
    """Build the verbatim fixture counterexample plus randomised structural variants.

    The fixture satisfies all three 𝒩-inequivariance conditions:

    (a) >=2 in-degree-0 CONST nodes (c1 and c2 have no in-edges)
    (b) >=1 VAR node that is a target of an edge (SIN -> VAR)
    (c) directed path from orphan CONST to a VAR (c2 -> SIN -> x_0)

    Keep arm on (d, d2): different canonical strings (equivariance failure).
    Drop arm on (d, d2): both raise RuntimeError (loud refusal = correct).

    Returns:
        List of LabeledDAGs.  The first two are d and d2 from the fixture.
    """
    dags: list[LabeledDAG] = []

    def _make_fixture(
        op_a: NodeType,
        op_b: NodeType,
        cv: float,
    ) -> tuple[LabeledDAG, LabeledDAG]:
        """Build d and d2 for one (op_a, op_b, const_value) combination."""
        d: LabeledDAG = LabeledDAG(16)
        for i in range(3):
            d.add_node(NodeType.VAR, var_index=i)  # 0, 1, 2
        c1 = d.add_node(NodeType.CONST, const_value=cv)  # 3
        c2 = d.add_node(NodeType.CONST, const_value=cv)  # 4
        node_a = d.add_node(op_a)  # 5
        node_b = d.add_node(op_b)  # 6
        d.add_edge(c2, node_a)
        d.add_edge(node_a, 0)  # op_a -> x_0  (Var node becomes an edge target)
        d.add_edge(c1, node_b)
        d.add_edge(node_b, 1)  # op_b -> x_1
        d2 = permute_internal_nodes(d, [1, 0, 2, 3])  # swap c1 <-> c2
        return d, d2

    # Verbatim fixture from the brief (T07 / Invariant 9 counterexample)
    d, d2 = _make_fixture(NodeType.SIN, NodeType.SIN, 1.0)
    dags.extend([d, d2])

    # Randomised structural variants
    for op_a, op_b, cv in [
        (NodeType.COS, NodeType.SIN, 1.0),
        (NodeType.SIN, NodeType.COS, 1.0),
        (NodeType.EXP, NodeType.SIN, 2.0),
        (NodeType.COS, NodeType.COS, 2.0),
        (NodeType.SIN, NodeType.SIN, 3.0),
    ]:
        va, vb = _make_fixture(op_a, op_b, cv)
        dags.extend([va, vb])

    return dags


# ---------------------------------------------------------------------------
# Population: bingo / udfs (live search with intercept hook)
# ---------------------------------------------------------------------------


class TwoArmRecorder:
    """Intercept fast_canonical_string calls to score both arms simultaneously.

    Installed as a monkey-patch during a bingo/udfs search.  Returns the keep
    arm result so the search proceeds with production behaviour.  The drop arm
    result is scored in parallel for measurement only.

    Args:
        original_fn: Saved reference to the real fast_canonical_string.
        timeout: Default canonicalisation time budget.
        rng: Random number generator for equivariance sampling.
        sample_rate: Equivariance check every 1-in-sample_rate DAGs.
        k_perms: Permutations per equivariance check.
    """

    def __init__(
        self,
        original_fn: Callable[..., str],
        timeout: float,
        rng: random.Random,
        sample_rate: int,
        k_perms: int,
    ) -> None:
        self._original = original_fn
        self.timeout = timeout
        self.rng = rng
        self.sample_rate = sample_rate
        self.k_perms = k_perms
        self.stats: dict[str, ArmStats] = {
            "keep": ArmStats(round_trip_comparator="fast_canonical_string"),
            "drop": ArmStats(round_trip_comparator="fast_canonical_string"),
        }
        self.n_both_ok: int = 0
        self.n_agreement: int = 0
        self._call_count: int = 0
        # Re-entrancy guard.  This recorder is installed *as*
        # isalsr.core.canonical.fast_canonical_string, so any helper that
        # re-imports that name at call time lands back here.  Without this flag
        # a round-trip check re-enters and scores its own S2D-decoded string as
        # a fresh adapter DAG, inflating every counter.  See the 2026-07-29
        # retraction in T07.
        self._in_call: bool = False
        self.n_reentrant_calls: int = 0

    def __call__(
        self,
        dag: LabeledDAG,
        timeout: float | None = None,
        mode: str = "wl_only",
        backend: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Score both arms and return keep result.

        Args:
            dag: DAG arriving at the canonicaliser.
            timeout: Per-call budget override.
            mode: Ignored (always wl_only via C++).
            backend: Ignored (always cpp).
            **kwargs: Forwarded to original_fn.

        Returns:
            Canonical string from the keep arm.

        Raises:
            RuntimeError: If the keep arm fails.
        """
        budget = float(timeout) if timeout is not None else self.timeout

        # Re-entrancy: delegate straight to the unpatched implementation and
        # record NOTHING.  A re-entrant call is this recorder's own round-trip
        # or equivariance machinery calling back into itself; counting it would
        # score S2D-decoded strings as if they were adapter output.
        if self._in_call:
            self.n_reentrant_calls += 1
            return self._original(dag, timeout=budget, **kwargs)

        self._in_call = True
        try:
            return self._record_one(dag, budget)
        finally:
            self._in_call = False

    def _record_one(self, dag: LabeledDAG, budget: float) -> str:
        """Score both arms for one genuinely-new DAG and return the keep string.

        Args:
            dag: DAG arriving at the canonicaliser.
            budget: Canonicalisation time budget.

        Returns:
            Canonical string from the keep arm.

        Raises:
            RuntimeError: If the keep arm fails.
        """
        comparison: dict[str, Any] = {}

        _process_one_dag(
            dag=dag,
            timeout=budget,
            rng=self.rng,
            sample_rate=self.sample_rate,
            k_perms=self.k_perms,
            call_idx=self._call_count,
            stats=self.stats,
            comparison=comparison,
            is_adversarial=False,
            keep_fn=self._original,
        )
        self._call_count += 1
        self.n_both_ok += comparison.get("n_both_ok", 0)
        self.n_agreement += comparison.get("n_agreement", 0)

        # Re-score keep arm to get the return value (ArmStats does not store last string)
        keep_out, keep_ret = _score_keep_with(self._original, dag, budget)
        if keep_out == "ok" and keep_ret is not None:
            return keep_ret
        raise RuntimeError("Fast canonical D2S: no valid operation found")


def _load_problem(
    suite: str,
    problem: str,
    seed: int,
) -> tuple[Any, Any, Any, Any]:
    """Load train/test splits for a named benchmark problem.

    Args:
        suite: Benchmark suite name (e.g. 'nguyen').
        problem: Problem name (e.g. 'Nguyen-5').
        seed: Random seed for data generation.

    Returns:
        (x_train, y_train, x_test, y_test).
    """
    import importlib

    # Suite key -> (defining module, benchmark-list constant). The key and the
    # module name coincide except for ``cherrypicked``, a legacy suite key kept
    # because it is baked into the on-disk result tree.
    bench_source_map = {
        "nguyen": ("nguyen", "NGUYEN_BENCHMARKS"),
        "feynman": ("feynman", "FEYNMAN_BENCHMARKS"),
        "hard": ("hard", "HARD_BENCHMARKS"),
        "cherrypicked": ("structural", "STRUCTURAL_BENCHMARKS"),
        "roundoff": ("roundoff", "ROUNDOFF_BENCHMARKS"),
    }
    module_name, const_name = bench_source_map[suite]
    module = importlib.import_module(f"benchmarks.datasets.{module_name}")
    benches = getattr(module, const_name)
    if isinstance(benches, dict):
        benches = list(benches.values())
    bench = next(b for b in benches if b["name"] == problem)
    if suite == "nguyen":
        return cast(
            tuple[Any, Any, Any, Any],
            module.generate_data(bench, n_train=240, n_test=1000, seed=seed),
        )
    return cast(
        tuple[Any, Any, Any, Any],
        module.generate_data(bench, n_samples=1250, train_ratio=0.8, seed=seed),
    )


def _build_config(method: str, max_time: float, timeout: float) -> dict[str, Any]:
    """Build production-like search config for bingo/udfs probe runs.

    Args:
        method: 'bingo' or 'udfs'.
        max_time: Search wall-clock budget in seconds.
        timeout: Canonicalisation budget in seconds.

    Returns:
        Config dict in the format expected by orchestrator.create_runner.
    """
    common: dict[str, Any] = {
        "canonicalization_timeout": timeout,
        "use_fast_canonical": True,
    }
    if method == "bingo":
        return {
            "bingo": {
                "population_size": 500,
                "stack_size": 32,
                "operators": ["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "pow"],
                "use_simplification": False,
                "crossover_prob": 0.4,
                "mutation_prob": 0.4,
                "metric": "mse",
                "clo_alg": "lm",
                "generations": 100_000_000,
                "fitness_threshold": 1.0e-16,
                "max_time": max_time,
                "max_evals": 100_000_000,
                "snapshot_frequency": 10,
                **common,
            },
            "isalsr": common,
        }
    return {
        "udfs": {
            "max_time": max_time,
            "processes": 1,
            **common,
        },
        "isalsr": common,
    }


def _run_live_search(
    method: str,
    suite: str,
    problem: str,
    seed: int,
    max_time: float,
    timeout: float,
    rng: random.Random,
    sample_rate: int,
    k_perms: int,
) -> dict[str, Any]:
    """Execute a live bingo/udfs search with both-arm interception.

    The TwoArmRecorder monkey-patches fast_canonical_string for the duration
    of the search.  The search itself runs with production behaviour (keep arm).

    Args:
        method: 'bingo' or 'udfs'.
        suite: Benchmark suite.
        problem: Problem name.
        seed: Random seed.
        max_time: Search budget in seconds.
        timeout: Canonicalisation budget.
        rng: RNG for equivariance sampling.
        sample_rate: 1-in-N equivariance sampling rate.
        k_perms: Permutations per equivariance check.

    Returns:
        Partial results dict (merged into main output by caller).
    """
    import isalsr.core.canonical as canonical_mod
    from experiments.models.orchestrator import create_runner

    original = canonical_mod.fast_canonical_string
    recorder = TwoArmRecorder(
        original_fn=original,
        timeout=timeout,
        rng=rng,
        sample_rate=sample_rate,
        k_perms=k_perms,
    )

    error_msg: str | None = None
    try:
        x_train, y_train, x_test, y_test = _load_problem(suite, problem, seed)
        config = _build_config(method, max_time, timeout)
        canonical_mod.fast_canonical_string = recorder  # monkey-patch
        try:
            runner = create_runner(method, "isalsr", config)
            runner.fit(x_train, y_train, x_test, y_test, seed, config)
        finally:
            canonical_mod.fast_canonical_string = original  # restore
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}"
        log.warning("Search error: %s", error_msg)

    return {
        "error": error_msg,
        "n_both_ok": recorder.n_both_ok,
        "n_agreement": recorder.n_agreement,
        "stats": recorder.stats,
        # Number of calls the re-entrancy guard intercepted and did NOT record.
        # Emitted so the fix is provable from the output rather than asserted:
        # a non-zero value is direct evidence that re-entry occurs and is now
        # excluded from the counters.
        "n_reentrant_calls": recorder.n_reentrant_calls,
    }


# ---------------------------------------------------------------------------
# Result finalisation
# ---------------------------------------------------------------------------


def _build_comparisons(
    stats: dict[str, ArmStats],
    n_both_ok: int,
    n_agreement: int,
    is_adversarial: bool,
    adversarial_keep_silent_wrong: int,
    adversarial_drop_loud_refusal: int,
) -> dict[str, Any]:
    """Compute the decisive cross-arm comparisons explicitly.

    Args:
        stats: Per-arm ArmStats.
        n_both_ok: Number of DAGs where both arms succeeded.
        n_agreement: Number of those that also produced identical strings.
        is_adversarial: Whether this is the adversarial population.
        adversarial_keep_silent_wrong: Keep-arm equivariance failures.
        adversarial_drop_loud_refusal: Drop-arm RuntimeError count.

    Returns:
        JSON-serialisable comparisons dict.
    """
    rho_keep = stats["keep"].rho
    rho_drop = stats["drop"].rho
    nd_keep = stats["keep"].n_unique
    nd_drop = stats["drop"].n_unique
    agr = n_agreement / n_both_ok if n_both_ok else float("nan")

    out: dict[str, Any] = {
        "rho_keep": rho_keep,
        "rho_drop": rho_drop,
        "rho_equal": (
            abs(rho_keep - rho_drop) < 1e-9
            if not (
                rho_keep != rho_keep or rho_drop != rho_drop  # NaN check
            )
            else False
        ),
        "n_distinct_keep": nd_keep,
        "n_distinct_drop": nd_drop,
        "n_distinct_equal": nd_keep == nd_drop,
        "n_both_ok": n_both_ok,
        "n_agreement": n_agreement,
        "per_dag_agreement_rate": agr,
        "n_equivariance_samples_keep": stats["keep"].n_equivariance_samples,
        "n_equivariance_failures_keep": stats["keep"].n_equivariance_failures,
        "n_equivariance_samples_drop": stats["drop"].n_equivariance_samples,
        "n_equivariance_failures_drop": stats["drop"].n_equivariance_failures,
        "adversarial_keep_silent_wrong": (
            adversarial_keep_silent_wrong if is_adversarial else None
        ),
        "adversarial_drop_loud_refusal": (
            adversarial_drop_loud_refusal if is_adversarial else None
        ),
    }
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901
    """CLI entry point for T07 norm-removal study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population",
        choices=["synthetic", "adversarial", "bingo", "udfs"],
        required=True,
    )
    parser.add_argument(
        "--n", type=int, default=100_000, help="Strings to attempt for synthetic population."
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Canonicalisation time budget in seconds."
    )
    parser.add_argument(
        "--sample-rate", type=int, default=50, help="Equivariance check every 1-in-N DAGs (0=off)."
    )
    parser.add_argument(
        "--equivariance-k", type=int, default=8, help="Permutations per equivariance check."
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    # Bingo/udfs only
    parser.add_argument("--method", default=None, help="SR method: bingo or udfs.")
    parser.add_argument("--suite", default=None, help="Benchmark suite (e.g. nguyen).")
    parser.add_argument("--problem", default=None, help="Problem name (e.g. Nguyen-5).")
    parser.add_argument(
        "--max-time", type=float, default=600.0, help="Search budget for bingo/udfs in seconds."
    )
    # Synthetic only
    parser.add_argument(
        "--task-id", type=int, default=0, help="Array task ID for multi-task synthetic runs."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    rng = random.Random(args.seed + args.task_id)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    stats: dict[str, ArmStats] = {
        "keep": ArmStats(round_trip_comparator="fast_canonical_string"),
        "drop": ArmStats(round_trip_comparator="structural_key"),
    }
    n_both_ok = 0
    n_agreement = 0
    adversarial_keep_silent_wrong = 0
    adversarial_drop_loud_refusal = 0
    is_adversarial = args.population == "adversarial"
    error_msg: str | None = None

    if args.population in ("synthetic", "adversarial"):
        if args.population == "synthetic":
            dag_iter: Iterator[LabeledDAG] = _iter_synthetic(args.n, rng)
        else:
            dag_iter = iter(_build_adversarial_dags())

        comparison: dict[str, Any] = {}
        for idx, dag in enumerate(dag_iter):
            _process_one_dag(
                dag=dag,
                timeout=args.timeout,
                rng=rng,
                sample_rate=args.sample_rate,
                k_perms=args.equivariance_k,
                call_idx=idx,
                stats=stats,
                comparison=comparison,
                is_adversarial=is_adversarial,
                keep_fn=None,
            )
            if (idx + 1) % 5_000 == 0:
                log.info(
                    "Processed %d DAGs; keep ok=%d drop ok=%d",
                    idx + 1,
                    stats["keep"].n_ok,
                    stats["drop"].n_ok,
                )
        n_both_ok = comparison.get("n_both_ok", 0)
        n_agreement = comparison.get("n_agreement", 0)
        adversarial_keep_silent_wrong = comparison.get("adversarial_keep_silent_wrong", 0)
        adversarial_drop_loud_refusal = comparison.get("adversarial_drop_loud_refusal", 0)
        # No monkey-patch is installed for synthetic/adversarial, so re-entry is
        # impossible by construction rather than merely unobserved.
        n_reentrant_calls = None

    else:
        # bingo / udfs: live search with interception
        if args.method is None or args.suite is None or args.problem is None:
            parser.error("--method, --suite, --problem required for bingo/udfs")

        live_result = _run_live_search(
            method=args.method,
            suite=args.suite,
            problem=args.problem,
            seed=args.seed,
            max_time=args.max_time,
            timeout=args.timeout,
            rng=rng,
            sample_rate=args.sample_rate,
            k_perms=args.equivariance_k,
        )
        stats = live_result["stats"]
        n_both_ok = live_result["n_both_ok"]
        n_agreement = live_result["n_agreement"]
        error_msg = live_result.get("error")
        n_reentrant_calls = live_result.get("n_reentrant_calls")

    elapsed = time.monotonic() - t_start
    comparisons = _build_comparisons(
        stats=stats,
        n_both_ok=n_both_ok,
        n_agreement=n_agreement,
        is_adversarial=is_adversarial,
        adversarial_keep_silent_wrong=adversarial_keep_silent_wrong,
        adversarial_drop_loud_refusal=adversarial_drop_loud_refusal,
    )

    result: dict[str, Any] = {
        "population": args.population,
        "config": {
            "seed": args.seed,
            "task_id": args.task_id,
            "timeout_sec": args.timeout,
            "sample_rate": args.sample_rate,
            "equivariance_k": args.equivariance_k,
            "n_synthetic": args.n if args.population == "synthetic" else None,
            "method": args.method,
            "suite": args.suite,
            "problem": args.problem,
            "max_time": args.max_time if args.population in ("bingo", "udfs") else None,
        },
        "elapsed_sec": elapsed,
        "error": error_msg,
        # Calls the re-entrancy guard intercepted and excluded from the
        # counters.  None for populations that install no monkey-patch
        # (synthetic, adversarial), where re-entry is impossible by
        # construction.  Non-zero here is direct evidence that the 2026-07-29
        # contamination mechanism is real and is now being excluded.
        "n_reentrant_calls": n_reentrant_calls,
        "arms": {arm: stats[arm].to_dict() for arm in ARMS},
        "comparisons": comparisons,
    }

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    log.info("Wrote %s in %.1f s", out_path, elapsed)

    # Print summary table
    print(f"\nT07 norm-removal study  population={args.population}  elapsed={elapsed:.1f}s")
    print(
        f"{'arm':<6} {'n_total':>8} {'n_ok':>7} {'n_raised':>9} "
        f"{'n_unique':>9} {'rho':>7} {'eq_fail':>8}"
    )
    print("-" * 58)
    for arm in ARMS:
        s = stats[arm]
        print(
            f"{arm:<6} {s.n_total:>8,} {s.n_ok:>7,} {s.n_raised:>9,} "
            f"{s.n_unique:>9,} {s.rho:>7.3f} "
            f"{s.n_equivariance_failures:>8,}"
        )
    print(f"\nPer-DAG agreement (both ok): {comparisons['per_dag_agreement_rate']:.6f}")
    print(
        f"rho_equal: {comparisons['rho_equal']}  "
        f"n_distinct_equal: {comparisons['n_distinct_equal']}"
    )
    if is_adversarial:
        print(
            f"Adversarial: keep silent-wrong={adversarial_keep_silent_wrong}  "
            f"drop loud-refusal={adversarial_drop_loud_refusal}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
