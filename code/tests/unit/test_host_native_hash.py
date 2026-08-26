"""Tests for the host-native deduplication key of the ``hash`` arm.

The ``hash`` arm answers reviewer comment R1.4: it is identical to the
``isalsr`` arm in every respect except the equivalence relation used to decide
whether a candidate has been seen before.  For the comparison to be a *naive*
baseline the key must be computed from the host's own representation in the
host's own node order.  Keying on the IsalSR adapter's output is not naive: both
adapters renumber (variables first, then constants, then operators in evaluation
/ row order), and that layout is itself a partial canonical form.

Two properties are pinned here, on a real candidate stream from each host:

soundness
    Equal host-native serialisation implies equal ``fast_canonical_string``.
    Zero violations are permitted -- a violation would mean the naive baseline
    merges candidates that IsalSR separates, which would invert the comparison.

strict refinement
    ``|distinct(host_native)| > |distinct(adapter INSERTION)|``.  The inequality
    ``>=`` is a theorem (host-native equality implies adapter equality, since the
    adapter is deterministic); the strictness is the empirical content, and it is
    exactly what the adapter's renumbering would otherwise hide.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from isalsr.baselines import FixedOrder, serialise
from isalsr.baselines.host_native import (
    HostNativeSerialisationError,
    host_native_digest,
    host_native_hash,
    host_native_serialise,
)
from isalsr.core.canonical import fast_canonical_string

# Stream sizes.  Both hosts produce these in well under a second; the cost of the
# fixtures is dominated by canonicalising every candidate.
_UDFS_STREAM_CAP = 1500
_BINGO_STREAM_CAP = 3000


@dataclass(frozen=True)
class Candidate:
    """The three competing keys for one candidate, plus its canonical string.

    Attributes:
        host_native: Serialisation of the host's own structure in host order.
        adapter_insertion: ``serialise(dag, FixedOrder.INSERTION)`` over the
            adapter's output -- the rung the ``hash`` arm used before this test.
        canonical: ``fast_canonical_string`` of the adapter's output.
    """

    host_native: str
    adapter_insertion: str
    canonical: str


# ----------------------------------------------------------------------
# Real candidate streams
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def udfs_stream() -> list[Candidate]:
    """Collect a real UDFS candidate stream through the patched evaluator."""
    vendor = Path(__file__).resolve().parents[2] / "experiments/models/udfs/vendor"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    dag_search_module = pytest.importorskip("DAG_search.dag_search")

    from experiments.models.udfs.adapter import compgraph_to_labeled_dag
    from experiments.models.udfs.isalsr_runner import udfs_host_native_records

    captured: list[Candidate] = []
    original = dag_search_module.evaluate_cgraph

    def wrapped(cgraph, x, loss_fkt, opt_mode="grid_zoom", loss_thresh=None):  # type: ignore[no-untyped-def] # noqa: ANN001,ANN202
        if len(captured) < _UDFS_STREAM_CAP:
            try:
                dag = compgraph_to_labeled_dag(cgraph)
                captured.append(
                    Candidate(
                        host_native=host_native_serialise(udfs_host_native_records(cgraph)),
                        adapter_insertion=serialise(dag, FixedOrder.INSERTION),
                        canonical=fast_canonical_string(dag),
                    )
                )
            except Exception:  # noqa: BLE001 - a failed candidate is simply not sampled
                pass
        return original(cgraph, x, loss_fkt, opt_mode, loss_thresh)

    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=(40, 2))
    y = x[:, 0] ** 2 + x[:, 1]

    dag_search_module.evaluate_cgraph = wrapped
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regressor = dag_search_module.DAGRegressor(
                k=1, n_calc_nodes=3, max_orders=200, processes=1, random_state=0, verbose=0
            )
            regressor.fit(x, y, verbose=0)
    finally:
        dag_search_module.evaluate_cgraph = original

    if len(captured) < 100:
        pytest.skip(f"UDFS stream too small ({len(captured)}) to be informative")
    return captured


@pytest.fixture(scope="module")
def bingo_stream() -> list[Candidate]:
    """Collect a real Bingo candidate stream through a capturing Evaluation."""
    pytest.importorskip("bingo")
    from bingo.evaluation.evaluation import Evaluation

    from experiments.models.bingo.adapter import agraph_to_labeled_dag
    from experiments.models.bingo.config import BingoConfig
    from experiments.models.bingo.isalsr_runner import bingo_host_native_records
    from experiments.models.bingo.runner import build_bingo_pipeline

    captured: list[Candidate] = []

    class _Capture(Evaluation):  # type: ignore[misc]
        def _serial_eval(self, population):  # type: ignore[no-untyped-def] # noqa: ANN001,ANN202
            for indv in population:
                if len(captured) < _BINGO_STREAM_CAP:
                    try:
                        dag = agraph_to_labeled_dag(indv)
                        captured.append(
                            Candidate(
                                host_native=host_native_serialise(bingo_host_native_records(indv)),
                                adapter_insertion=serialise(dag, FixedOrder.INSERTION),
                                canonical=fast_canonical_string(dag),
                            )
                        )
                    except Exception:  # noqa: BLE001
                        pass
                if not indv.fit_set:
                    indv.fitness = self.fitness_function(indv)

    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=(60, 1))
    y = x[:, 0] ** 3 + x[:, 0] ** 2 + x[:, 0]

    cfg = BingoConfig.from_dict(
        {
            "population_size": 60,
            "stack_size": 16,
            "generations": 25,
            "max_time": 120,
            "max_evals": 100_000,
            "use_simplification": False,
        }
    )
    np.random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        island, _fitness_fn, _ev = build_bingo_pipeline(
            x, y, cfg, evaluation_cls=_Capture, evaluation_kwargs={}
        )
        island.evolve_until_convergence(
            max_generations=25,
            fitness_threshold=-1e300,
            max_fitness_evaluations=10**7,
            convergence_check_frequency=5,
            max_time=120,
        )

    if len(captured) < 100:
        pytest.skip(f"Bingo stream too small ({len(captured)}) to be informative")
    return captured


@pytest.fixture(scope="module")
def streams(
    udfs_stream: list[Candidate], bingo_stream: list[Candidate]
) -> dict[str, list[Candidate]]:
    """Both host streams, keyed by host name."""
    return {"udfs": udfs_stream, "bingo": bingo_stream}


# ----------------------------------------------------------------------
# Property 1 -- soundness on the real stream
# ----------------------------------------------------------------------


@pytest.mark.parametrize("host", ["udfs", "bingo"])
def test_host_native_key_is_sound(streams: dict[str, list[Candidate]], host: str) -> None:
    """Equal host-native serialisation must imply equal canonical string."""
    seen: dict[str, str] = {}
    violations: list[tuple[str, str, str]] = []
    for cand in streams[host]:
        previous = seen.setdefault(cand.host_native, cand.canonical)
        if previous != cand.canonical:
            violations.append((cand.host_native, previous, cand.canonical))
    assert violations == [], f"{len(violations)} soundness violations on the {host} stream"


@pytest.mark.parametrize("host", ["udfs", "bingo"])
def test_host_native_refines_adapter_order(streams: dict[str, list[Candidate]], host: str) -> None:
    """Equal host-native must imply equal adapter-order serialisation.

    This is the refinement half of ``|distinct(host_native)| >=
    |distinct(adapter)|``: the adapter is a deterministic function of the host
    object, so it cannot separate what the host representation identifies.
    """
    seen: dict[str, str] = {}
    for cand in streams[host]:
        previous = seen.setdefault(cand.host_native, cand.adapter_insertion)
        assert previous == cand.adapter_insertion, host


# ----------------------------------------------------------------------
# Property 2 -- strictly more distinct than the adapter's own order
# ----------------------------------------------------------------------


@pytest.mark.parametrize("host", ["udfs", "bingo"])
def test_host_native_is_strictly_finer_than_adapter_order(
    streams: dict[str, list[Candidate]], host: str
) -> None:
    """The adapter's node layout must not be inherited by the naive baseline."""
    stream = streams[host]
    n_host = len({c.host_native for c in stream})
    n_adapter = len({c.adapter_insertion for c in stream})
    assert n_host > n_adapter, (
        f"{host}: host-native {n_host} distinct vs adapter INSERTION {n_adapter} "
        f"over {len(stream)} candidates -- the adapter's renumbering was inherited"
    )


@pytest.mark.parametrize("host", ["udfs", "bingo"])
def test_canonical_is_the_coarsest_of_the_three(
    streams: dict[str, list[Candidate]], host: str
) -> None:
    """The ladder must be ordered: canonical <= adapter order <= host native."""
    stream = streams[host]
    n_canonical = len({c.canonical for c in stream})
    n_adapter = len({c.adapter_insertion for c in stream})
    n_host = len({c.host_native for c in stream})
    assert n_canonical <= n_adapter <= n_host, (host, n_canonical, n_adapter, n_host)


# ----------------------------------------------------------------------
# Arm wiring
# ----------------------------------------------------------------------


@pytest.mark.parametrize("module_name", ["bingo", "udfs"])
def test_hash_arm_keys_on_host_native(module_name: str) -> None:
    """The ``hash`` arm's key mode must be ``host_native`` on both hosts."""
    mod = pytest.importorskip(f"experiments.models.{module_name}.isalsr_runner")
    runner_cls = mod.HashBingoRunner if module_name == "bingo" else mod.HashUDFSRunner
    assert runner_cls.KEY_MODE == "host_native"
    assert mod.HASH_ARM_KEY_MODE == "host_native"


@pytest.mark.parametrize("method", ["udfs", "bingo"])
def test_hash_arm_variant_name_is_unchanged(method: str) -> None:
    """``metadata.representation`` is the variant name and must stay ``hash``."""
    from experiments.models.orchestrator import create_runner

    runner = create_runner(method, "hash", {})
    assert runner.variant == "hash"


@pytest.mark.parametrize("module_name", ["bingo", "udfs"])
def test_adapter_order_key_mode_still_available(module_name: str) -> None:
    """The steel-manned adapter-order rung must remain selectable."""
    mod = pytest.importorskip(f"experiments.models.{module_name}.isalsr_runner")
    assert "hash" in mod.KEY_MODES
    assert "host_native" in mod.KEY_MODES
    assert "canonical" in mod.KEY_MODES


@pytest.mark.parametrize("module_name", ["bingo", "udfs"])
def test_host_native_key_mode_requires_the_host(module_name: str) -> None:
    """Without the host object the naive key cannot be computed; it must raise."""
    from isalsr.core.labeled_dag import LabeledDAG
    from isalsr.core.node_types import NodeType

    mod = pytest.importorskip(f"experiments.models.{module_name}.isalsr_runner")
    dag = LabeledDAG(8)
    x0 = dag.add_node(NodeType.VAR, var_index=0)
    s = dag.add_node(NodeType.SIN)
    dag.add_edge(x0, s)

    dedup = mod._CanonicalDeduplicator(key_mode="host_native")
    with pytest.raises(ValueError, match="host_native"):
        dedup.representation_string(dag)


@pytest.mark.parametrize("module_name", ["bingo", "udfs"])
def test_shadow_sketches_use_precision_16(module_name: str) -> None:
    """The shadow sketches must be p=16 (s.e. 0.41 %), not p=14 (s.e. 0.81 %)."""
    mod = pytest.importorskip(f"experiments.models.{module_name}.isalsr_runner")
    assert mod.SHADOW_HLL_PRECISION == 16
    dedup = mod._CanonicalDeduplicator(shadow_hash=True)
    for sketch in dedup._shadow.values():
        assert len(sketch._registers) == 1 << 16


# ----------------------------------------------------------------------
# The serialiser itself
# ----------------------------------------------------------------------


def test_serialise_is_order_sensitive() -> None:
    """Reordering the records must change the string: the order is the host's."""
    a = [(0, "inp", ()), (1, "sin", (0,))]
    b = [(1, "sin", (0,)), (0, "inp", ())]
    assert host_native_serialise(a) != host_native_serialise(b)


def test_serialise_is_operand_order_sensitive() -> None:
    """Operand order is emitted verbatim (Critical Invariant 8)."""
    a = [(0, "inp", ()), (1, "inp", ()), (2, "div_l", (0, 1))]
    b = [(0, "inp", ()), (1, "inp", ()), (2, "div_l", (1, 0))]
    assert host_native_serialise(a) != host_native_serialise(b)


def test_serialise_distinguishes_keys_not_positions() -> None:
    """Two record lists differing only in host keys must serialise differently."""
    a = [(0, "inp", ()), (3, "sin", (0,))]
    b = [(0, "inp", ()), (7, "sin", (0,))]
    assert host_native_serialise(a) != host_native_serialise(b)


def test_serialise_empty_stream() -> None:
    assert host_native_serialise([]) == "0|"


@pytest.mark.parametrize("tag", ["", "a;b", "a|b", "a:b", "a<b", "a>b", "a,b"])
def test_serialise_rejects_ambiguous_tags(tag: str) -> None:
    with pytest.raises(HostNativeSerialisationError):
        host_native_serialise([(0, tag, ())])


def test_serialise_accepts_udfs_operation_alphabet() -> None:
    """Every UDFS op string must survive the grammar, including ``=`` and ``+``."""
    ops = [
        "inp",
        "const",
        "=",
        "+",
        "*",
        "sub_l",
        "sub_r",
        "div_l",
        "div_r",
        "sin",
        "cos",
        "exp",
        "log",
        "sqrt",
        "inv",
        "neg",
    ]
    text = host_native_serialise([(i, op, (0,)) for i, op in enumerate(ops)])
    assert text.startswith(f"{len(ops)}|")


def test_serialise_rejects_non_integer_keys() -> None:
    with pytest.raises(HostNativeSerialisationError):
        host_native_serialise([("a", "inp", ())])  # type: ignore[list-item]


def test_serialise_accepts_numpy_integers() -> None:
    """UDFS stores children as ``np.int64``; they must not need pre-conversion."""
    records: Sequence[tuple[int, str, Sequence[int]]] = [
        (np.int64(0), "inp", ()),  # type: ignore[list-item]
        (np.int64(1), "sin", (np.int64(0),)),  # type: ignore[list-item]
    ]
    assert host_native_serialise(records) == host_native_serialise(
        [(0, "inp", ()), (1, "sin", (0,))]
    )


def test_hash_and_digest_agree_with_serialisation() -> None:
    records = [(0, "inp", ()), (1, "sin", (0,))]
    assert host_native_hash(records) == hash(host_native_serialise(records))
    other = [(0, "inp", ()), (1, "cos", (0,))]
    assert host_native_digest(records) != host_native_digest(other)
    assert 0 <= host_native_digest(records) < 2**64


# ----------------------------------------------------------------------
# Host record extraction
# ----------------------------------------------------------------------


def test_udfs_records_follow_node_dict_not_eval_order() -> None:
    """The UDFS extractor must not consult the computed ``eval_order``."""
    pytest.importorskip("DAG_search.comp_graph")
    from experiments.models.udfs.isalsr_runner import udfs_host_native_records

    class _FakeCompGraph:
        node_dict = {0: ([], "inp"), 1: ([], "inp"), 3: ([1, 0], "+"), 2: ([], "const")}
        eval_order = [0, 1, 2, 3]

    records = udfs_host_native_records(_FakeCompGraph())
    assert [key for key, _tag, _ops in records] == [0, 1, 3, 2]
    assert records[2] == (3, "+", (1, 0))


def test_bingo_records_drop_dead_rows_and_unary_param2() -> None:
    """Non-utilised rows and the ignored ``param2`` of unary rows must not key."""
    pytest.importorskip("bingo")
    from bingo.symbolic_regression.agraph.agraph import AGraph

    from experiments.models.bingo.isalsr_runner import bingo_host_native_records

    def _agraph(rows: list[list[int]]) -> AGraph:
        ag = AGraph(use_simplification=False)
        ag._command_array = np.array(rows, dtype=int)
        ag._notify_modification()
        return ag

    # Row 1 is dead code; row 2 (SIN) has junk in param2.
    a = _agraph([[0, 0, 0], [0, 1, 1], [6, 0, 0], [2, 2, 0]])
    b = _agraph([[0, 0, 0], [1, 0, 0], [6, 0, 1], [2, 2, 0]])
    assert host_native_serialise(bingo_host_native_records(a)) == host_native_serialise(
        bingo_host_native_records(b)
    )
    assert [key for key, _tag, _ops in bingo_host_native_records(a)] == [0, 2, 3]


def test_bingo_records_keep_binary_operand_order() -> None:
    """``param1``/``param2`` order must be emitted verbatim (Critical Invariant 8)."""
    pytest.importorskip("bingo")
    from bingo.symbolic_regression.agraph.agraph import AGraph

    from experiments.models.bingo.isalsr_runner import bingo_host_native_records

    def _agraph(rows: list[list[int]]) -> AGraph:
        ag = AGraph(use_simplification=False)
        ag._command_array = np.array(rows, dtype=int)
        ag._notify_modification()
        return ag

    a = _agraph([[0, 0, 0], [1, 0, 0], [3, 0, 1]])  # SUBTRACTION(x0, c)
    b = _agraph([[0, 0, 0], [1, 0, 0], [3, 1, 0]])  # SUBTRACTION(c, x0)
    assert host_native_serialise(bingo_host_native_records(a)) != host_native_serialise(
        bingo_host_native_records(b)
    )
