"""Domain restriction of the canonical-string invariant to structural candidates.

Why this module exists
----------------------
The IsalSR canonical string encodes the **instruction sequence that builds the
internal nodes** of a labeled DAG.  The :math:`m` input variables are
pre-inserted into the initial state and are never created by a ``V``/``v``
instruction (``CLAUDE.md`` invariant 7), so a DAG with **zero internal nodes**
has an empty instruction sequence and canonicalises to ``""`` --- for every
:math:`m`, and for every choice of which variable is the output.

That is not a defect of the canonicaliser.  ``""`` is the faithful encoding of
*the initial state*, exactly as Definition 3.2 specifies it.  What it cannot do
is name an output: a :class:`~isalsr.core.labeled_dag.LabeledDAG` carries no
output marker, and for :math:`k \\ge 1` the output is recoverable as the unique
sink, while at :math:`k = 0` there are :math:`m` sinks and nothing distinguishes
them.  Two host candidates ``f(x) = x_0`` and ``f(x) = x_1`` therefore share the
key ``""`` while computing different functions.

Why the restriction is principled, not a workaround
---------------------------------------------------
IsalSR's claim is that the :math:`O(k!)` relabelings of the :math:`k` internal
nodes collapse to one canonical string.  At :math:`k = 0` that group is trivial
(:math:`0! = 1`): there is no redundancy to collapse, no equivalence class with
more than one member, and :math:`\\rho` --- evaluations per distinct *structural*
class --- is undefined on an object with no structure.  A bare terminal is a
well-formed SR hypothesis, and the host is right to score it; it is simply not
an object our invariant is about.

So the canonical string is a complete labeled-DAG invariant **for** :math:`k \\ge 1`,
and candidates with :math:`k = 0` are passed through: evaluated normally by the
host, never deduplicated, never entered into a fitness cache, and excluded from
the :math:`\\rho` accounting so they neither inflate nor deflate it.

Measured incidence (Stage D, Pagie-1, Bingo/isalsr, seed 102, 119,795 sampled
candidates): **0.0593 %** at :math:`k = 0`, i.e. ~7,100 in the full stream.  UDFS
cannot reach :math:`k = 0` at all --- all seven campaign configs pin
``n_calc_nodes = 5`` --- so this is a Bingo-only effect in practice.

Discovered by the T04 Mode 1 replay (EXECUTION-PLAN §4.4 D3), 2026-08-06, before
the campaign committed any core-hours.  That is what D3 is for.
"""

from __future__ import annotations

from isalsr.baselines import FixedOrder, serialise
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

__all__ = [
    "NONSTRUCTURAL_KEY_PREFIX",
    "STRUCTURAL_SCOPE_REASON",
    "count_internal_nodes",
    "is_structural",
    "nonstructural_key",
    "recorded_key",
]

#: Recorded on the trace record of every k=0 candidate, so the substitution is
#: visible in the stream rather than inferred from a counter.
STRUCTURAL_SCOPE_REASON = "k0_nonstructural"

#: Marks a key that is NOT a Sigma_SR word.  No canonical string can collide
#: with it: every Sigma_SR word is over {N,P,n,p,C,c,W} plus V/v-labelled pairs,
#: so none can begin with ``#``.
NONSTRUCTURAL_KEY_PREFIX = "#k0:"


def count_internal_nodes(dag: LabeledDAG) -> int:
    """Return the number of non-``VAR`` nodes, i.e. :math:`k`.

    ``CONST`` counts as internal: it is created by the ``Vk`` instruction and is
    therefore encoded, which is why a bare constant canonicalises to ``"Vk"``
    and is perfectly distinguishable.

    Args:
        dag: The labeled DAG to measure.

    Returns:
        The count of nodes whose label is not :attr:`NodeType.VAR`.
    """
    return sum(1 for node in range(dag.node_count) if dag.node_label(node) is not NodeType.VAR)


def is_structural(dag: LabeledDAG) -> bool:
    """Return whether *dag* lies in the domain of the canonical-string invariant.

    A DAG is structural iff it carries at least one internal node. Non-structural
    DAGs (bare input variables) all canonicalise to ``""`` regardless of which
    variable they return, so their canonical string must never be used as a
    deduplication or fitness-cache key.

    Args:
        dag: The labeled DAG to test.

    Returns:
        ``True`` when :math:`k \\ge 1`, ``False`` for a bare-variable candidate.
    """
    return count_internal_nodes(dag) > 0


def nonstructural_key(dag: LabeledDAG) -> str:
    """Return a sound deduplication key for a k=0 candidate.

    The canonical string cannot serve: it is ``""`` for every k=0 DAG, so it
    equates ``f(x) = x_0`` with ``f(x) = x_1``.  The fixed-order insertion
    serialisation can: it records the node count and each variable's index, so
    ``1|x0<>`` and ``2|x0<>;x1<>`` are distinct.  It is injective on DAGs with
    no edges, which is exactly this class.

    This is a *substitution*, not an exclusion, and the distinction matters for
    rho.  Bare-variable candidates are ordinary redundancy -- on a
    single-variable problem every one of them is literally the same DAG, and
    collapsing them is precisely what the deduplication is for.  Dropping them
    from the accounting instead understates rho by ~12 % (measured on Stage C
    v5b) and would be a self-inflicted penalty, not a correction.

    No asymmetry is introduced between arms: the ``hash`` arm already keys on a
    fixed-order serialisation for *every* candidate, so this gives the ``isalsr``
    arm the same treatment on the one class where Sigma_SR is not a complete
    invariant, and nothing better anywhere else.

    Args:
        dag: A candidate DAG with zero internal nodes.

    Returns:
        A key string that cannot collide with any canonical string.
    """
    try:
        return NONSTRUCTURAL_KEY_PREFIX + serialise(dag, FixedOrder.INSERTION)
    except Exception:  # noqa: BLE001
        # Never raise from the evaluation hot loop: one malformed candidate must
        # not kill a 12 h run.  Every adapter sets ``var_index`` on VAR nodes, so
        # this is unreachable in production; if it ever fires, fall back to a key
        # that is still SOUND for its purpose -- it separates DAGs by node count,
        # which is what distinguishes the colliding shapes (1|x0<> vs
        # 2|x0<>;x1<>).  Coarser than the serialisation, never wrong.
        return f"{NONSTRUCTURAL_KEY_PREFIX}n{dag.node_count}"


def recorded_key(dag: LabeledDAG, canonical: str) -> str:
    """Return the key the production runners record for *dag*.

    Both IsalSR runners substitute :func:`nonstructural_key` for the canonical
    string whenever :math:`k = 0` (``bingo/isalsr_runner.py``,
    ``udfs/isalsr_runner.py``), because ``""`` equates ``f(x) = x_0`` with
    ``f(x) = x_1``.  The Stage-D trace persists the *substituted* value, so
    every replay verifier must apply the same substitution before comparing.
    Re-canonicalising unconditionally and comparing the result against the
    recorded value reports a mismatch on every k=0 record -- a false engine
    disagreement, not a real one, since both engines return ``""`` there.

    Args:
        dag: The candidate DAG the key belongs to.
        canonical: The canonical string computed for *dag*.

    Returns:
        ``canonical`` when *dag* is structural, the non-structural substitution
        key otherwise.
    """
    return canonical if is_structural(dag) else nonstructural_key(dag)
