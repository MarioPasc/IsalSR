"""Unit tests for :class:`experiments.models.complexity_telemetry.ComplexityTelemetry`.

Three properties matter operationally and are tested here directly.

*Determinism of the sampling grid.* Both arms of a contrast must sample the same
positions of the candidate stream, otherwise the residual instrumentation cost
does not cancel between arms. The exact set of firing indices is asserted, not
just the firing rate.

*Total failure containment.* Telemetry runs inside 12-hour SLURM jobs whose
product is the run log. An exception escaping the instrument destroys the
measurement it was added to take, so every entry point is fed objects that cannot
be described and is required not to raise.

*Schema agreement.* :meth:`ComplexityTelemetry.scalars` is splatted into
:class:`~experiments.models.schemas.SearchSpaceResults`. Both sides of that
contract are derived programmatically, so the test fails if either drifts.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Final

import pytest

from experiments.models.complexity_telemetry import (
    DEFAULT_GEN_FREQ,
    DEFAULT_SAMPLE_RATE,
    MODE_POPULATION,
    MODE_STREAM,
    ComplexityTelemetry,
)
from experiments.models.schemas import SearchSpaceResults
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

#: The 15 distributional keys that must be ``None`` when nothing was sampled.
_DISTRIBUTIONAL_KEYS: Final[tuple[str, ...]] = (
    "complexity_mean_k",
    "complexity_std_k",
    "complexity_median_k",
    "complexity_p90_k",
    "complexity_max_k",
    "complexity_mean_depth",
    "complexity_median_depth",
    "complexity_mean_edges",
    "complexity_mean_n_op",
    "complexity_mean_n_const",
    "complexity_mean_shared",
    "complexity_mean_sharing_surplus",
    "complexity_mean_nonlinear",
    "complexity_mean_op_entropy",
    "complexity_mean_max_in_degree",
)

#: The secondary-accumulator keys.
_UNIQUE_KEYS: Final[tuple[str, ...]] = (
    "complexity_unique_n_sampled",
    "complexity_unique_mean_k",
    "complexity_unique_mean_depth",
    "complexity_unique_mean_nonlinear",
    "complexity_unique_mean_op_entropy",
)


def _schema_complexity_fields() -> set[str]:
    """Return the ``complexity_*`` field names declared on ``SearchSpaceResults``.

    Returns
    -------
    set of str
        The field names, read from the dataclass rather than hard-coded.
    """
    return {
        field.name
        for field in dataclasses.fields(SearchSpaceResults)
        if field.name.startswith("complexity_")
    }


def _make_dag(n_ops: int = 2) -> LabeledDAG:
    """Return a small describable DAG: one variable feeding *n_ops* unary nodes.

    Parameters
    ----------
    n_ops : int
        Number of unary operator nodes.

    Returns
    -------
    LabeledDAG
        The DAG.
    """
    dag = LabeledDAG(max_nodes=n_ops + 1)
    dag.add_node(NodeType.VAR, var_index=0)
    for i in range(n_ops):
        dag.add_node(NodeType.SIN if i % 2 == 0 else NodeType.COS)
        dag.add_edge(i, i + 1)
    return dag


class _ExplodingDag:
    """Stand-in for a DAG whose description raises on the first attribute touched."""

    @property
    def node_count(self) -> int:
        """Raise unconditionally.

        Raises
        ------
        RuntimeError
            Always.
        """
        raise RuntimeError("synthetic descriptor failure")


class _NotADag:
    """An object with no DAG interface at all."""


def _exploding_convert(_individual: Any) -> LabeledDAG:
    """Conversion callable that always fails.

    Raises
    ------
    ValueError
        Always.
    """
    raise ValueError("synthetic conversion failure")


def _identity_convert(individual: Any) -> Any:
    """Return *individual* unchanged, so a population may hold pre-built DAGs."""
    return individual


# ----------------------------------------------------------------------
# 1-2. Sampling grids
# ----------------------------------------------------------------------


class TestStreamSamplingGrid:
    @pytest.mark.parametrize("sample_rate", [1, 2, 7, 31, 97])
    def test_fires_exactly_every_nth_call(self, sample_rate: int) -> None:
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=True, sample_rate=sample_rate)
        fired = {i for i in range(1000) if telemetry.should_sample()}
        expected = {i for i in range(1000) if (i + 1) % sample_rate == 0}
        assert fired == expected
        assert len(fired) == 1000 // sample_rate

    def test_disabled_never_fires(self) -> None:
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=False, sample_rate=2)
        assert not any(telemetry.should_sample() for _ in range(100))

    @pytest.mark.parametrize("sample_rate", [3, 31])
    def test_unique_substream_has_its_own_independent_grid(self, sample_rate: int) -> None:
        telemetry = ComplexityTelemetry(
            MODE_STREAM, enabled=True, sample_rate=sample_rate, track_unique=True
        )
        telemetry.should_sample()  # advance the primary counter only
        fired = {i for i in range(300) if telemetry.should_sample_unique()}
        assert fired == {i for i in range(300) if (i + 1) % sample_rate == 0}

    def test_unique_grid_is_dead_without_the_secondary_accumulator(self) -> None:
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=True, sample_rate=1)
        assert not any(telemetry.should_sample_unique() for _ in range(50))

    def test_two_instances_sample_identical_positions(self) -> None:
        # Cross-arm comparability: an identical stream must yield an identical grid.
        left = ComplexityTelemetry(MODE_STREAM, enabled=True, sample_rate=13)
        right = ComplexityTelemetry(MODE_STREAM, enabled=True, sample_rate=13, track_unique=True)
        assert [left.should_sample() for _ in range(200)] == [
            right.should_sample() for _ in range(200)
        ]


class TestGenerationSamplingGrid:
    @pytest.mark.parametrize("gen_freq", [1, 5, 25, 50])
    def test_fires_iff_generation_is_a_multiple(self, gen_freq: int) -> None:
        telemetry = ComplexityTelemetry(MODE_POPULATION, enabled=True, gen_freq=gen_freq)
        for generation in range(200):
            assert telemetry.should_sample_generation(generation) == (generation % gen_freq == 0)

    @pytest.mark.parametrize("gen_freq", [1, 5, 25, 50])
    def test_generation_zero_always_fires(self, gen_freq: int) -> None:
        telemetry = ComplexityTelemetry(MODE_POPULATION, enabled=True, gen_freq=gen_freq)
        assert telemetry.should_sample_generation(0)

    def test_disabled_never_fires(self) -> None:
        telemetry = ComplexityTelemetry(MODE_POPULATION, enabled=False, gen_freq=1)
        assert not any(telemetry.should_sample_generation(g) for g in range(50))


class TestConstruction:
    @pytest.mark.parametrize("mode", [MODE_STREAM, MODE_POPULATION])
    def test_defaults(self, mode: str) -> None:
        telemetry = ComplexityTelemetry(mode)
        assert telemetry.mode == mode
        assert telemetry.enabled is True
        assert telemetry.sample_rate == DEFAULT_SAMPLE_RATE
        assert telemetry.gen_freq == DEFAULT_GEN_FREQ
        assert telemetry.n_failures == 0
        assert telemetry.n_sampled == 0

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown complexity sampling mode"):
            ComplexityTelemetry("per_evaluation")


# ----------------------------------------------------------------------
# 3. Environment overrides
# ----------------------------------------------------------------------


class TestEnvironmentOverrides:
    def test_sample_rate_and_gen_freq_are_read_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ISALSR_COMPLEXITY_SAMPLE_RATE", "7")
        monkeypatch.setenv("ISALSR_COMPLEXITY_GEN_FREQ", "3")
        telemetry = ComplexityTelemetry(MODE_STREAM)
        assert telemetry.sample_rate == 7
        assert telemetry.gen_freq == 3

    @pytest.mark.parametrize("raw", ["0", "-4", "not-an-int", ""])
    def test_invalid_sample_rate_falls_back_to_the_default(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv("ISALSR_COMPLEXITY_SAMPLE_RATE", raw)
        assert ComplexityTelemetry(MODE_STREAM).sample_rate == DEFAULT_SAMPLE_RATE

    def test_explicit_arguments_beat_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ISALSR_COMPLEXITY_SAMPLE_RATE", "7")
        monkeypatch.setenv("ISALSR_COMPLEXITY", "0")
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=True, sample_rate=5)
        assert telemetry.enabled is True
        assert telemetry.sample_rate == 5

    @pytest.mark.parametrize("raw", ["1", "true", "yes", "anything"])
    def test_only_the_literal_zero_disables(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv("ISALSR_COMPLEXITY", raw)
        assert ComplexityTelemetry(MODE_STREAM).enabled is True

    def test_kill_switch_disables_everything(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ISALSR_COMPLEXITY", "0")
        telemetry = ComplexityTelemetry(MODE_STREAM)
        assert telemetry.enabled is False

        assert not any(telemetry.should_sample() for _ in range(200))
        assert not telemetry.should_sample_generation(0)
        telemetry.observe(_make_dag())
        telemetry.observe_population([_make_dag(), _make_dag()], _identity_convert)
        telemetry.observe_converted(object(), _identity_convert)

        scalars = telemetry.scalars()
        assert scalars["complexity_sampling_mode"] is None
        assert scalars["complexity_sample_rate"] is None
        assert scalars["complexity_n_sampled"] == 0
        assert scalars["complexity_time_s"] == 0.0
        assert all(scalars[key] is None for key in _DISTRIBUTIONAL_KEYS)
        assert all(scalars[key] is None for key in _UNIQUE_KEYS)

    def test_enabled_reports_the_active_rate_for_its_mode(self) -> None:
        stream = ComplexityTelemetry(MODE_STREAM, enabled=True, sample_rate=11, gen_freq=3)
        population = ComplexityTelemetry(MODE_POPULATION, enabled=True, sample_rate=11, gen_freq=3)
        assert stream.scalars()["complexity_sample_rate"] == 11
        assert population.scalars()["complexity_sample_rate"] == 3


# ----------------------------------------------------------------------
# 4. Failure containment
# ----------------------------------------------------------------------


class TestFailureContainment:
    @pytest.mark.parametrize(
        ("bad", "bad_id"),
        [(_ExplodingDag(), "raises"), (_NotADag(), "not-a-dag"), ("a string", "str")],
    )
    def test_observe_swallows_and_counts(self, bad: Any, bad_id: str) -> None:
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=True)
        telemetry.observe(bad)
        assert telemetry.n_failures == 1
        assert telemetry.n_sampled == 0

    def test_observe_none_is_ignored_without_counting_a_failure(self) -> None:
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=True)
        telemetry.observe(None)
        assert telemetry.n_failures == 0
        assert telemetry.n_sampled == 0

    def test_observe_unique_swallows_and_counts(self) -> None:
        telemetry = ComplexityTelemetry(MODE_POPULATION, enabled=True, track_unique=True)
        telemetry.observe_unique(_ExplodingDag())
        telemetry.observe_unique(_NotADag())
        assert telemetry.n_failures == 2
        assert telemetry.scalars()["complexity_unique_n_sampled"] == 0

    @pytest.mark.parametrize("bad", [_ExplodingDag(), _NotADag(), 3.14])
    def test_observe_converted_swallows_conversion_failures(self, bad: Any) -> None:
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=True)
        telemetry.observe_converted(bad, _exploding_convert)
        telemetry.observe_converted(bad, _identity_convert)
        assert telemetry.n_failures == 2
        assert telemetry.n_sampled == 0

    def test_a_good_individual_survives_a_bad_population(self) -> None:
        telemetry = ComplexityTelemetry(MODE_POPULATION, enabled=True)
        population = [_ExplodingDag(), _make_dag(3), _NotADag(), _make_dag(2), "junk"]
        telemetry.observe_population(population, _identity_convert)
        assert telemetry.n_failures == 3
        assert telemetry.n_sampled == 2
        # Mean over the two describable DAGs only: k = 3 and k = 2.
        assert telemetry.scalars()["complexity_mean_k"] == pytest.approx(2.5)

    def test_population_conversion_failure_loses_the_whole_population_but_not_the_run(
        self,
    ) -> None:
        telemetry = ComplexityTelemetry(MODE_POPULATION, enabled=True)
        telemetry.observe_population([_make_dag(), _make_dag()], _exploding_convert)
        assert telemetry.n_failures == 2
        assert telemetry.n_sampled == 0

    def test_failures_accumulate_beyond_the_logging_cap(self) -> None:
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=True)
        for _ in range(20):
            telemetry.observe(_NotADag())
        assert telemetry.n_failures == 20

    def test_disabled_telemetry_ignores_bad_input_entirely(self) -> None:
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=False)
        telemetry.observe(_ExplodingDag())
        telemetry.observe_converted(_NotADag(), _exploding_convert)
        telemetry.observe_population([_ExplodingDag()], _exploding_convert)
        assert telemetry.n_failures == 0


# ----------------------------------------------------------------------
# 5. Secondary (deduplication-miss) accumulator
# ----------------------------------------------------------------------


class TestUniqueAccumulator:
    def test_untracked_leaves_every_unique_field_none(self) -> None:
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=True, track_unique=False)
        for _ in range(5):
            telemetry.observe(_make_dag(3), unique=True)
        scalars = telemetry.scalars()
        assert telemetry.n_sampled == 5
        assert all(scalars[key] is None for key in _UNIQUE_KEYS)

    def test_stream_mode_populates_both_accumulators(self) -> None:
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=True, track_unique=True)
        telemetry.observe(_make_dag(2), unique=True)
        telemetry.observe(_make_dag(4), unique=False)
        scalars = telemetry.scalars()
        assert scalars["complexity_n_sampled"] == 2
        assert scalars["complexity_unique_n_sampled"] == 1
        assert scalars["complexity_mean_k"] == pytest.approx(3.0)
        assert scalars["complexity_unique_mean_k"] == pytest.approx(2.0)
        assert scalars["complexity_unique_mean_depth"] == pytest.approx(2.0)

    def test_population_mode_feeds_the_secondary_accumulator_only(self) -> None:
        telemetry = ComplexityTelemetry(MODE_POPULATION, enabled=True, track_unique=True)
        telemetry.observe_unique(_make_dag(6))
        scalars = telemetry.scalars()
        assert scalars["complexity_n_sampled"] == 0
        assert scalars["complexity_unique_n_sampled"] == 1
        assert scalars["complexity_mean_k"] is None
        assert scalars["complexity_unique_mean_k"] == pytest.approx(6.0)

    def test_tracked_but_empty_reports_zero_count_and_none_moments(self) -> None:
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=True, track_unique=True)
        telemetry.observe(_make_dag(3))
        scalars = telemetry.scalars()
        assert scalars["complexity_unique_n_sampled"] == 0
        assert scalars["complexity_unique_mean_k"] is None
        assert scalars["complexity_unique_mean_depth"] is None
        assert scalars["complexity_unique_mean_nonlinear"] is None
        assert scalars["complexity_unique_mean_op_entropy"] is None


# ----------------------------------------------------------------------
# 6. Schema agreement
# ----------------------------------------------------------------------


def _populated_telemetry() -> ComplexityTelemetry:
    """Return a stream-mode instrument with a few DAGs and one failure folded in."""
    telemetry = ComplexityTelemetry(MODE_STREAM, enabled=True, track_unique=True)
    for n_ops in (2, 3, 5):
        telemetry.observe(_make_dag(n_ops), unique=n_ops == 3)
    telemetry.observe(_NotADag())
    return telemetry


class TestScalarsSchema:
    @pytest.mark.parametrize(
        "telemetry_factory",
        [
            pytest.param(_populated_telemetry, id="populated"),
            pytest.param(lambda: ComplexityTelemetry(MODE_POPULATION, enabled=True), id="empty"),
            pytest.param(lambda: ComplexityTelemetry(MODE_STREAM, enabled=False), id="disabled"),
        ],
    )
    def test_keys_match_search_space_results_exactly(self, telemetry_factory: Any) -> None:
        assert set(telemetry_factory().scalars()) == _schema_complexity_fields()

    def test_scalars_can_be_splatted_into_the_schema(self) -> None:
        results = SearchSpaceResults(
            total_dags_explored=100,
            unique_canonical_dags=80,
            empirical_reduction_factor=1.25,
            max_internal_nodes_seen=5,
            theoretical_reduction_bound=120.0,
            redundancy_rate=0.2,
            **_populated_telemetry().scalars(),
        )
        assert results.complexity_sampling_mode == MODE_STREAM
        assert results.complexity_n_sampled == 3
        assert results.complexity_n_failures == 1

    def test_recorded_moments_are_the_expected_values(self) -> None:
        scalars = _populated_telemetry().scalars()
        # k = 2, 3, 5 -> mean 10/3, median (lower convention) 3, max 5.
        assert scalars["complexity_mean_k"] == pytest.approx(10.0 / 3.0)
        assert scalars["complexity_median_k"] == pytest.approx(3.0)
        assert scalars["complexity_max_k"] == pytest.approx(5.0)
        assert scalars["complexity_mean_depth"] == pytest.approx(10.0 / 3.0)
        assert scalars["complexity_mean_n_const"] == pytest.approx(0.0)
        assert scalars["complexity_mean_shared"] == pytest.approx(0.0)
        assert scalars["complexity_mean_max_in_degree"] == pytest.approx(1.0)

    def test_zero_is_never_used_where_no_measurement_exists(self) -> None:
        scalars = ComplexityTelemetry(MODE_STREAM, enabled=True).scalars()
        assert scalars["complexity_n_sampled"] == 0
        assert all(scalars[key] is None for key in _DISTRIBUTIONAL_KEYS)


# ----------------------------------------------------------------------
# 7. Sidecar payload
# ----------------------------------------------------------------------


class TestSidecar:
    @pytest.mark.parametrize(
        "telemetry_factory",
        [
            pytest.param(_populated_telemetry, id="populated"),
            pytest.param(
                lambda: ComplexityTelemetry(MODE_POPULATION, enabled=True, track_unique=True),
                id="empty",
            ),
            pytest.param(lambda: ComplexityTelemetry(MODE_STREAM, enabled=False), id="disabled"),
        ],
    )
    def test_json_round_trip(self, telemetry_factory: Any) -> None:
        payload = telemetry_factory().sidecar()
        restored = json.loads(json.dumps(payload))
        assert set(restored) == set(payload)
        assert restored["schema_version"] == 1

    def test_carries_the_histograms(self) -> None:
        payload = json.loads(json.dumps(_populated_telemetry().sidecar()))
        histograms = payload["all"]["histograms"]
        assert set(histograms) == {"n_internal", "depth", "n_edges", "n_op"}
        # k = 2, 3, 5 with no overflow, so the bins are the raw counts.
        assert histograms["n_internal"]["overflow"] == 0
        bins = histograms["n_internal"]["bins"]
        assert [bins[2], bins[3], bins[5]] == [1, 1, 1]
        assert sum(bins) == payload["all"]["n"] == 3

    def test_carries_the_label_counts(self) -> None:
        payload = _populated_telemetry().sidecar()
        label_counts = payload["all"]["label_counts"]
        assert label_counts["VAR"] == 3
        assert label_counts["SIN"] + label_counts["COS"] == 2 + 3 + 5

    def test_reports_the_configuration_and_the_stream_position(self) -> None:
        telemetry = ComplexityTelemetry(
            MODE_STREAM, enabled=True, sample_rate=4, gen_freq=9, track_unique=True
        )
        for _ in range(10):
            telemetry.should_sample()
        payload = telemetry.sidecar()
        assert payload["sampling_mode"] == MODE_STREAM
        assert payload["enabled"] is True
        assert payload["sample_rate"] == 4
        assert payload["gen_freq"] == 9
        assert payload["n_candidates_seen"] == 10
        assert payload["unique"] is not None

    def test_unique_block_is_none_when_untracked(self) -> None:
        payload = ComplexityTelemetry(MODE_STREAM, enabled=True).sidecar()
        assert payload["unique"] is None

    def test_empty_payload_uses_nan_for_undefined_moments(self) -> None:
        payload = ComplexityTelemetry(MODE_STREAM, enabled=True).sidecar()
        assert payload["all"]["n"] == 0
        assert math.isnan(payload["all"]["moments"]["depth"]["mean"])


# ----------------------------------------------------------------------
# 8. Instrumentation cost accounting
# ----------------------------------------------------------------------


class TestTimeAccounting:
    def test_starts_at_zero(self) -> None:
        assert ComplexityTelemetry(MODE_STREAM, enabled=True).time_s == 0.0

    @pytest.mark.parametrize(
        "call",
        ["observe", "observe_unique", "observe_converted", "observe_population"],
    )
    def test_every_entry_point_charges_wall_time(self, call: str) -> None:
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=True, track_unique=True)
        for _ in range(20):
            if call == "observe":
                telemetry.observe(_make_dag(4))
            elif call == "observe_unique":
                telemetry.observe_unique(_make_dag(4))
            elif call == "observe_converted":
                telemetry.observe_converted(_make_dag(4), _identity_convert)
            else:
                telemetry.observe_population([_make_dag(4)], _identity_convert)
        assert telemetry.time_s > 0.0

    def test_time_is_charged_even_when_the_description_fails(self) -> None:
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=True)
        for _ in range(20):
            telemetry.observe(_NotADag())
        assert telemetry.time_s > 0.0
        assert telemetry.n_failures == 20

    def test_time_is_exactly_zero_when_disabled(self) -> None:
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=False, track_unique=True)
        for _ in range(50):
            telemetry.observe(_make_dag(4))
            telemetry.observe_unique(_make_dag(4))
            telemetry.observe_converted(_make_dag(4), _identity_convert)
            telemetry.observe_population([_make_dag(4)], _identity_convert)
        assert telemetry.time_s == 0.0
        assert telemetry.scalars()["complexity_time_s"] == 0.0

    def test_time_is_monotone(self) -> None:
        telemetry = ComplexityTelemetry(MODE_STREAM, enabled=True)
        previous = 0.0
        for _ in range(10):
            telemetry.observe(_make_dag(6))
            assert telemetry.time_s >= previous
            previous = telemetry.time_s
