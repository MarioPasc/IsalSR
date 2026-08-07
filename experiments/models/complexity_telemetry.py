"""Sampled structural telemetry over the DAGs a search explores.

Records, per ``(method, arm, problem, seed)``, the distribution of the structural
descriptors defined in :mod:`isalsr.core.complexity` over the DAGs a run actually
visits. The motivating hypothesis is that the ``isalsr`` arm -- and, less
strongly, the ``hash`` arm -- explores structurally harder DAGs than ``baseline``,
because deduplication converts wall clock into search progress and because Bingo's
``isalsr`` arm actively penalises duplicates already held in the population.

Three design constraints shape this module, and each rules out something obvious.

**The baseline arm never builds a LabeledDAG.** Both baseline runners work
directly on the host representation, so any structural measurement there costs a
conversion that the arm does not otherwise pay
(``agraph_to_labeled_dag`` measures 23.8 us/DAG, ``describe_dag`` a further
6.8 us, against a Bingo fitness evaluation of roughly 140 us).

**Runs are budgeted in wall clock, not evaluations.** Instrumentation cost is
therefore not an overhead that can be subtracted from a report afterwards: it
removes evaluations from the search. On the control arm that would move R2,
solution recovery and ``n_total_dags``. Hence sampling, at a rate chosen so the
cost is a fraction of a percent.

**Comparability needs an identical rule, not an identical cost.** The sampling
rule is the same for all three arms of a method, so the residual cost is common
to all three and cancels in every arm-versus-arm contrast.

Two sampling modes, one per method, dictated by where a hook exists:

``population``
    Bingo. Every ``gen_freq`` generations the whole population is described. This
    is the only mode available to the Bingo baseline, whose sole hook
    (``_TrajectoryEvaluation.__call__``) is per-generation. The estimand is the
    structural regime the search is holding at generation *g*.

``stream``
    UDFS. Every ``sample_rate``-th candidate at the per-candidate hook that all
    three arms already have. The estimand is the distribution over proposed
    candidates.

Both defaults are prime so the sampling phase cannot alias with the population
size (500), the stack size (32) or the snapshot frequency (1000).

Telemetry is **enabled by default**. The T06 reachability ledger defaulted to off
and was not set in any config, so an entire 1,260-run certification wave recorded
five rates of zero -- which reads as "no fallbacks occurred" and means "nothing
was counted" (EXECUTION-PLAN SP-6). That failure is unrecoverable after the fact,
because the population being measured exists only while a search runs. The same
applies here, so the default is on and the environment variable is a kill switch.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable
from typing import Any, Final

from isalsr.core.complexity import ComplexityAccumulator

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GEN_FREQ",
    "DEFAULT_SAMPLE_RATE",
    "MODE_POPULATION",
    "MODE_STREAM",
    "ComplexityTelemetry",
]

MODE_POPULATION: Final[str] = "population"
MODE_STREAM: Final[str] = "stream"

#: Stream mode: describe every 31st candidate. Prime, so the phase cannot align
#: with any power-of-two or round batch size in either host. At UDFS's roughly
#: 2.2 M candidates per run this yields about 71 k sampled DAGs, whose standard
#: error is orders of magnitude below the between-seed dispersion the paired
#: tests operate on.
DEFAULT_SAMPLE_RATE: Final[int] = 31

#: Population mode: describe the whole population every 25 generations,
#: generation 0 included. Measured on Nguyen-1: a population of 500 at 32.7 us
#: per individual is 16.4 ms per sampling event against roughly 8 s of search
#: for 25 generations, i.e. about 0.2 % of wall clock. Generation 0 is sampled
#: unconditionally so that a run always carries at least one measurement -- at
#: the original frequency of 50 a short Bingo run recorded n = 0, which is
#: indistinguishable in the run log from an instrument that never fired.
DEFAULT_GEN_FREQ: Final[int] = 25

#: Environment kill switch. Anything other than "0" leaves telemetry enabled.
_ENV_ENABLED: Final[str] = "ISALSR_COMPLEXITY"
_ENV_SAMPLE_RATE: Final[str] = "ISALSR_COMPLEXITY_SAMPLE_RATE"
_ENV_GEN_FREQ: Final[str] = "ISALSR_COMPLEXITY_GEN_FREQ"

#: Cap on logged conversion failures, so a systematically failing problem cannot
#: fill a SLURM .err file. The count itself is always exact.
_LOG_FAILURE_CAP: Final[int] = 5


def _env_int(name: str, default: int) -> int:
    """Read a strictly positive integer from the environment, falling back quietly."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    if value < 1:
        logger.warning("%s=%d must be >= 1; using %d", name, value, default)
        return default
    return value


class ComplexityTelemetry:
    """Sampled structural telemetry for one run.

    Holds two accumulators. ``all`` is fed from every sampled DAG and is the
    cross-arm comparable measurement, defined identically in all three arms.
    ``unique`` is fed only from sampled DAGs that were a deduplication miss, so
    it is populated on the ``hash`` and ``isalsr`` arms and stays empty on
    ``baseline``, which holds no cache. That asymmetry is intrinsic, which is why
    every headline claim must rest on ``all``.

    Attributes:
        mode: :data:`MODE_POPULATION` or :data:`MODE_STREAM`.
        sample_rate: Stream mode -- describe every ``sample_rate``-th candidate.
        gen_freq: Population mode -- describe the population every ``gen_freq``
            generations.
        enabled: Whether anything is recorded at all.
        n_failures: Number of DAGs that could not be described. A conversion or
            descriptor failure is counted and swallowed: telemetry must never be
            able to kill a 12-hour run.
        time_s: Wall time charged to telemetry, including any conversion done on
            its behalf. Recorded so the instrumentation cost can be subtracted
            from a reported overhead exactly rather than estimated.
    """

    __slots__ = (
        "_all",
        "_counter",
        "_logged_failures",
        "_unique",
        "_unique_counter",
        "enabled",
        "gen_freq",
        "mode",
        "n_failures",
        "sample_rate",
        "time_s",
    )

    def __init__(
        self,
        mode: str,
        *,
        enabled: bool | None = None,
        sample_rate: int | None = None,
        gen_freq: int | None = None,
        track_unique: bool = False,
    ) -> None:
        """Initialise telemetry for one run.

        Args:
            mode: :data:`MODE_POPULATION` or :data:`MODE_STREAM`.
            enabled: Override the environment kill switch. ``None`` reads
                ``ISALSR_COMPLEXITY``.
            sample_rate: Override the stream sampling rate. ``None`` reads
                ``ISALSR_COMPLEXITY_SAMPLE_RATE``.
            gen_freq: Override the population sampling frequency. ``None`` reads
                ``ISALSR_COMPLEXITY_GEN_FREQ``.
            track_unique: Whether to maintain the secondary deduplication-miss
                accumulator. Only meaningful for arms that hold a cache.

        Raises:
            ValueError: If *mode* is not a recognised sampling mode.
        """
        if mode not in (MODE_POPULATION, MODE_STREAM):
            raise ValueError(f"unknown complexity sampling mode: {mode!r}")

        self.mode = mode
        self.enabled = (
            os.environ.get(_ENV_ENABLED, "1") != "0" if enabled is None else bool(enabled)
        )
        self.sample_rate = (
            _env_int(_ENV_SAMPLE_RATE, DEFAULT_SAMPLE_RATE) if sample_rate is None else sample_rate
        )
        self.gen_freq = _env_int(_ENV_GEN_FREQ, DEFAULT_GEN_FREQ) if gen_freq is None else gen_freq
        self.n_failures = 0
        self.time_s = 0.0
        self._counter = 0
        self._unique_counter = 0
        self._logged_failures = 0
        self._all = ComplexityAccumulator()
        self._unique = ComplexityAccumulator() if track_unique else None

    # ------------------------------------------------------------------
    # Stream mode
    # ------------------------------------------------------------------

    def should_sample(self) -> bool:
        """Advance the candidate counter and report whether this one is sampled.

        Deterministic systematic sampling: candidate *i* is taken when
        ``i % sample_rate == 0``. Deterministic rather than random so two arms
        given the same stream sample the same positions, and so a run is exactly
        reproducible.

        Returns:
            ``True`` if the current candidate should be described.
        """
        if not self.enabled:
            return False
        self._counter += 1
        return self._counter % self.sample_rate == 0

    def should_sample_unique(self) -> bool:
        """Advance the *unique*-substream counter and report whether to sample.

        Separate from :meth:`should_sample` because in ``population`` mode the
        primary accumulator is not fed from a candidate stream at all, so the
        secondary accumulator needs its own 1-in-``sample_rate`` grid over the
        deduplication misses. Callers must gate on the miss verdict *before*
        calling this, so the counter counts misses.

        Returns:
            ``True`` if the current deduplication miss should be described.
        """
        if not self.enabled or self._unique is None:
            return False
        self._unique_counter += 1
        return self._unique_counter % self.sample_rate == 0

    def observe(self, dag: Any, *, unique: bool = False) -> None:
        """Describe *dag* and fold it into the primary accumulator.

        Charges its own wall time to :attr:`time_s`. Never raises: a DAG that
        cannot be described increments :attr:`n_failures` and is dropped.

        Args:
            dag: A :class:`~isalsr.core.labeled_dag.LabeledDAG`.
            unique: Whether this DAG was a deduplication miss, and so belongs in
                the secondary accumulator as well. Only meaningful in
                ``stream`` mode, where the two accumulators share one sampling
                grid.
        """
        if not self.enabled or dag is None:
            return
        start = time.perf_counter()
        try:
            descriptor = self._all.observe(dag)
            if unique and self._unique is not None:
                self._unique.update(descriptor)
        except Exception as exc:  # telemetry must never kill a run
            self._record_failure(exc)
        finally:
            self.time_s += time.perf_counter() - start

    def observe_converted(self, individual: Any, convert: Callable[[Any], Any]) -> None:
        """Convert *individual* and fold the result into the primary accumulator.

        For the arms that hold no ``LabeledDAG`` of their own, where the
        conversion exists only to serve telemetry and must therefore be charged
        to :attr:`time_s`. A conversion failure is counted and swallowed.

        Args:
            individual: A host-native candidate.
            convert: Callable turning it into a
                :class:`~isalsr.core.labeled_dag.LabeledDAG`.
        """
        if not self.enabled:
            return
        start = time.perf_counter()
        try:
            self._all.observe(convert(individual))
        except Exception as exc:  # telemetry must never kill a run
            self._record_failure(exc)
        finally:
            self.time_s += time.perf_counter() - start

    def observe_unique(self, dag: Any) -> None:
        """Describe *dag* into the secondary accumulator only.

        Used in ``population`` mode, where the primary accumulator is fed from
        the population rather than from the candidate stream, so a candidate
        must not reach it.

        Args:
            dag: A :class:`~isalsr.core.labeled_dag.LabeledDAG`.
        """
        if not self.enabled or dag is None or self._unique is None:
            return
        start = time.perf_counter()
        try:
            self._unique.observe(dag)
        except Exception as exc:  # telemetry must never kill a run
            self._record_failure(exc)
        finally:
            self.time_s += time.perf_counter() - start

    # ------------------------------------------------------------------
    # Population mode
    # ------------------------------------------------------------------

    def should_sample_generation(self, generation: int) -> bool:
        """Report whether the population at *generation* should be described.

        Args:
            generation: The generation index, as counted by the host runner.

        Returns:
            ``True`` if this generation is a sampling point.
        """
        return self.enabled and generation % self.gen_freq == 0

    def observe_population(
        self,
        population: Iterable[Any],
        convert: Callable[[Any], Any],
    ) -> None:
        """Convert and describe every individual of *population*.

        The conversion is done on telemetry's behalf and is therefore charged to
        :attr:`time_s`. An individual that fails to convert is counted and
        skipped; the rest of the population is still described.

        Args:
            population: The host's population of individuals.
            convert: Callable turning one individual into a
                :class:`~isalsr.core.labeled_dag.LabeledDAG`.
        """
        if not self.enabled:
            return
        start = time.perf_counter()
        accumulator = self._all
        for individual in population:
            try:
                accumulator.observe(convert(individual))
            except Exception as exc:  # one bad individual must not lose the rest
                self._record_failure(exc)
        self.time_s += time.perf_counter() - start

    def _record_failure(self, exc: Exception) -> None:
        """Count a failed description, logging only the first few."""
        self.n_failures += 1
        if self._logged_failures < _LOG_FAILURE_CAP:
            self._logged_failures += 1
            logger.warning(
                "complexity telemetry: could not describe DAG (%s: %s)",
                type(exc).__name__,
                exc,
            )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @property
    def n_sampled(self) -> int:
        """Number of DAGs folded into the primary accumulator."""
        return self._all.n

    def scalars(self) -> dict[str, Any]:
        """Return the flat run-level summary written into ``SearchSpaceResults``.

        Keys match the ``complexity_*`` field names on
        :class:`~experiments.models.schemas.SearchSpaceResults` exactly, so the
        translators can splat this mapping. Every distributional field is ``None``
        when nothing was sampled, never ``0.0`` -- a zero would be read as a
        measurement.

        Returns:
            The mapping of field name to value.
        """
        out: dict[str, Any] = {
            "complexity_sampling_mode": self.mode if self.enabled else None,
            "complexity_sample_rate": (
                (self.sample_rate if self.mode == MODE_STREAM else self.gen_freq)
                if self.enabled
                else None
            ),
            "complexity_n_sampled": self._all.n,
            "complexity_time_s": self.time_s,
            "complexity_n_failures": self.n_failures,
            "complexity_unique_n_sampled": (self._unique.n if self._unique is not None else None),
        }

        primary = self._all
        if primary.n:
            out.update(
                {
                    "complexity_mean_k": primary.mean("n_internal"),
                    "complexity_std_k": primary.std("n_internal"),
                    "complexity_median_k": primary.quantile("n_internal", 0.50),
                    "complexity_p90_k": primary.quantile("n_internal", 0.90),
                    "complexity_max_k": primary.extremum("n_internal", largest=True),
                    "complexity_mean_depth": primary.mean("depth"),
                    "complexity_median_depth": primary.quantile("depth", 0.50),
                    "complexity_mean_edges": primary.mean("n_edges"),
                    "complexity_mean_n_op": primary.mean("n_op"),
                    "complexity_mean_n_const": primary.mean("n_const"),
                    "complexity_mean_shared": primary.mean("n_shared"),
                    "complexity_mean_sharing_surplus": primary.mean("sharing_surplus"),
                    "complexity_mean_nonlinear": primary.mean("n_nonlinear"),
                    "complexity_mean_op_entropy": primary.mean("op_label_entropy"),
                    "complexity_mean_max_in_degree": primary.mean("max_in_degree"),
                }
            )
        else:
            for key in (
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
            ):
                out[key] = None

        unique = self._unique
        has_unique = unique is not None and unique.n > 0
        out["complexity_unique_mean_k"] = unique.mean("n_internal") if has_unique else None
        out["complexity_unique_mean_depth"] = unique.mean("depth") if has_unique else None
        out["complexity_unique_mean_nonlinear"] = unique.mean("n_nonlinear") if has_unique else None
        out["complexity_unique_mean_op_entropy"] = (
            unique.mean("op_label_entropy") if has_unique else None
        )
        return out

    def sidecar(self) -> dict[str, Any]:
        """Return the full distributional payload for ``complexity.json``.

        Carries the exact histograms and the ``NodeType`` counts, which the flat
        scalars cannot. This is what distribution-level tests read.

        Returns:
            A JSON-ready mapping.
        """
        payload: dict[str, Any] = {
            "schema_version": 1,
            "sampling_mode": self.mode,
            "enabled": self.enabled,
            "sample_rate": self.sample_rate,
            "gen_freq": self.gen_freq,
            "n_candidates_seen": self._counter,
            "n_failures": self.n_failures,
            "time_s": self.time_s,
            "all": self._all.to_dict(),
        }
        payload["unique"] = self._unique.to_dict() if self._unique is not None else None
        return payload
