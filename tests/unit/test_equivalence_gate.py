"""Smoke tests for the equivalence_gate harness.

Verifies that the harness runs in --quick mode and produces a well-formed
JSON report with the required keys and types.

Key behavioural contract under test:
- ``_build_corpus`` returns ``(corpus, n_generated, n_discarded_unreachable)``.
- When the C++ canonicaliser is absent (current state), the harness sets
  ``self_comparison=True``, ``engine_b='python'``, ``pass=False``, and records
  a ``capability_probe`` dict in provenance.  A self-comparison MUST NOT
  produce ``pass=True``.
- Individual gate ``pass`` fields reflect correctness (invariance, round-trip)
  and may be ``True`` even during a self-comparison.
- ``dags_generated``, ``dags_discarded_unreachable``, and ``discard_rate`` are
  first-class fields in the gate-2 result.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from experiments.scripts.equivalence_gate import (
    FIXED_SEED,
    _build_corpus,
    _canonicalize,
    _detect_self_comparison,
    _gen_random_string,
    _probe_cpp_canonical,
    _run_gate1,
    _run_gate2,
    _run_gate3,
    main,
)
from isalsr.core.labeled_dag import LabeledDAG
from isalsr.core.node_types import NodeType

# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestDetectSelfComparison:
    """Tests for _detect_self_comparison."""

    def test_both_python_is_self(self) -> None:
        assert _detect_self_comparison("python", "python") is True

    def test_cpp_python_is_not_self(self) -> None:
        assert _detect_self_comparison("cpp", "python") is False

    def test_python_cpp_is_not_self(self) -> None:
        assert _detect_self_comparison("python", "cpp") is False


class TestProbeCppCanonical:
    """Tests for _probe_cpp_canonical.

    Under current tree state fast_canonical_string has no backend= param,
    so the probe must return (False, non-empty-string).
    """

    def test_returns_tuple(self) -> None:
        result = _probe_cpp_canonical()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_available_is_bool(self) -> None:
        available, _ = _probe_cpp_canonical()
        assert isinstance(available, bool)

    def test_detail_is_nonempty_string(self) -> None:
        _, detail = _probe_cpp_canonical()
        assert isinstance(detail, str)
        assert len(detail) > 0

    def test_cpp_not_available_absence_path(self) -> None:
        """When C++ canonicaliser is absent the probe must return False.

        Skip when the C++ canonicaliser is already live (another workstream
        may have landed it).  The two outcomes are mutually exclusive; the
        test is only meaningful in the absence case.
        """
        available, detail = _probe_cpp_canonical()
        if available:
            pytest.skip("C++ canonicaliser is live in this tree; absence-path test skipped")
        assert available is False, f"Expected False but got True: {detail!r}"

    def test_detail_mentions_type_error_when_absent(self) -> None:
        """When C++ canonicaliser is absent (TypeError raised), detail records it."""
        available, detail = _probe_cpp_canonical()
        if available:
            pytest.skip("C++ canonicaliser is live; TypeError-detail test skipped")
        assert "TypeError" in detail or "not" in detail.lower(), detail


class TestCanonicalizeWrapper:
    """Tests for _canonicalize — no silent fallback."""

    def _make_simple_dag(self) -> LabeledDAG:
        dag = LabeledDAG(max_nodes=4)
        dag.add_node(NodeType.VAR, var_index=0)
        dag.add_node(NodeType.SIN)
        dag.add_edge(0, 1)
        return dag

    def test_python_backend_returns_string(self) -> None:
        dag = self._make_simple_dag()
        result = _canonicalize(dag, "python")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_python_backend_is_deterministic(self) -> None:
        dag = self._make_simple_dag()
        r1 = _canonicalize(dag, "python")
        r2 = _canonicalize(dag, "python")
        assert r1 == r2

    def test_cpp_backend_raises_when_unavailable(self) -> None:
        """``_canonicalize("cpp")`` must raise when the C++ engine is absent —
        never silently return a Python result.

        Two distinct exceptions are legitimate, and which one appears depends on
        how far the tree has been built:

        * ``TypeError`` — ``fast_canonical_string`` has no ``backend`` parameter
          at all (the dispatcher has not landed yet);
        * ``RuntimeError`` — the dispatcher exists and rejects ``backend="cpp"``
          because the extension is not installed.

        Asserting only ``TypeError`` made this test fail once the dispatcher
        landed, which inverted its meaning: the stricter behaviour looked like a
        regression. What is actually being defended is that neither case falls
        back silently.
        """
        dag = self._make_simple_dag()
        available, _ = _probe_cpp_canonical()
        if available:
            pytest.skip("C++ canonicaliser is available; skip absence test")
        with pytest.raises((TypeError, RuntimeError)):
            _canonicalize(dag, "cpp")


class TestGenRandomString:
    """Tests for _gen_random_string."""

    def test_returns_nonempty_string(self) -> None:
        import random

        rng = random.Random(42)
        s = _gen_random_string(rng, 10, 1)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_different_seeds_differ(self) -> None:
        import random

        rng1 = random.Random(1)
        rng2 = random.Random(2)
        strings = {_gen_random_string(rng1, 15, 2) for _ in range(10)}
        strings2 = {_gen_random_string(rng2, 15, 2) for _ in range(10)}
        assert len(strings | strings2) > 1


class TestBuildCorpus:
    """Tests for _build_corpus — now returns (corpus, n_generated, n_discarded)."""

    def test_returns_three_tuple(self) -> None:
        result = _build_corpus(target_size=10, seed=FIXED_SEED)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_corpus_length_matches_target(self) -> None:
        corpus, _, _ = _build_corpus(target_size=20, seed=FIXED_SEED)
        assert len(corpus) == 20

    def test_n_generated_at_least_target(self) -> None:
        corpus, n_gen, n_disc = _build_corpus(target_size=20, seed=FIXED_SEED)
        assert n_gen >= len(corpus)

    def test_discard_counts_add_up(self) -> None:
        corpus, n_gen, n_disc = _build_corpus(target_size=30, seed=FIXED_SEED)
        # n_gen = accepted + discarded_unreachable
        assert n_gen == len(corpus) + n_disc

    def test_discard_rate_is_positive(self) -> None:
        """23%+ of DAGs with internals are unreachable from x_0 (verified empirically)."""
        _, n_gen, n_disc = _build_corpus(target_size=100, seed=FIXED_SEED)
        assert n_disc > 0, "Expected some unreachable DAGs in corpus"
        assert n_gen > n_disc

    def test_all_dags_have_internal_nodes(self) -> None:
        corpus, _, _ = _build_corpus(target_size=30, seed=FIXED_SEED + 1)
        for _src, _n_vars, dag in corpus:
            m = len(dag.var_nodes())
            assert dag.node_count > m, "Corpus DAG has no internal nodes"

    def test_reproducible_with_same_seed(self) -> None:
        c1, _, _ = _build_corpus(target_size=10, seed=FIXED_SEED)
        c2, _, _ = _build_corpus(target_size=10, seed=FIXED_SEED)
        for (s1, nv1, _), (s2, nv2, _) in zip(c1, c2, strict=True):
            assert s1 == s2
            assert nv1 == nv2

    def test_different_seeds_produce_different_corpora(self) -> None:
        c1, _, _ = _build_corpus(target_size=10, seed=FIXED_SEED)
        c2, _, _ = _build_corpus(target_size=10, seed=FIXED_SEED + 999)
        strings1 = [s for s, _, _ in c1]
        strings2 = [s for s, _, _ in c2]
        assert strings1 != strings2


# ---------------------------------------------------------------------------
# Gate-level smoke tests (quick mode)
# ---------------------------------------------------------------------------


class TestGate1Quick:
    """Gate 1 produces a well-formed result dict in quick mode."""

    def test_gate1_structure(self) -> None:
        result = _run_gate1(quick=True, backend_a="python", backend_b="python")
        assert isinstance(result["dags_tested"], int)
        assert isinstance(result["comparisons_made"], int)
        assert isinstance(result["mismatches_cross_engine"], int)
        assert isinstance(result["mismatches_invariance"], int)
        assert isinstance(result["errors"], int)
        assert isinstance(result["pass"], bool)
        assert isinstance(result["cap_dags_per_k"], dict)
        assert isinstance(result["k_max"], int)
        assert isinstance(result["per_k"], list)

    def test_gate1_self_comparison_passes_invariance(self) -> None:
        """Gate-level pass (invariance check) must be True in self-comparison."""
        result = _run_gate1(quick=True, backend_a="python", backend_b="python")
        assert result["pass"] is True
        assert result["mismatches_cross_engine"] == 0
        assert result["mismatches_invariance"] == 0

    def test_gate1_dags_tested_positive(self) -> None:
        result = _run_gate1(quick=True, backend_a="python", backend_b="python")
        assert result["dags_tested"] > 0

    def test_gate1_per_k_has_entries(self) -> None:
        result = _run_gate1(quick=True, backend_a="python", backend_b="python")
        assert len(result["per_k"]) == result["k_max"]
        for entry in result["per_k"]:
            assert "k" in entry
            assert "perms_per_dag" in entry
            assert "comparisons" in entry


class TestGate2Quick:
    """Gate 2 produces a well-formed result dict in quick mode."""

    def test_gate2_structure(self) -> None:
        result, corpus = _run_gate2(quick=True, backend_a="python", backend_b="python")
        assert isinstance(result["dags_tested"], int)
        assert isinstance(result["comparisons_made"], int)
        assert isinstance(result["mismatches"], int)
        assert isinstance(result["errors"], int)
        assert isinstance(result["pass"], bool)
        assert isinstance(result["corpus_seed"], int)
        assert isinstance(result["k_distribution"], list)
        # Reachability discard stats must be present.
        assert isinstance(result["dags_generated"], int)
        assert isinstance(result["dags_discarded_unreachable"], int)
        assert isinstance(result["discard_rate"], float)

    def test_gate2_discard_stats_consistent(self) -> None:
        result, corpus = _run_gate2(quick=True, backend_a="python", backend_b="python")
        assert result["dags_generated"] == (
            result["dags_tested"] + result["dags_discarded_unreachable"]
        )
        assert result["dags_discarded_unreachable"] >= 0
        assert 0.0 <= result["discard_rate"] <= 1.0

    def test_gate2_discard_rate_nonzero(self) -> None:
        """Empirically ~76% of DAGs with internals are unreachable; rate > 0."""
        result, _ = _run_gate2(quick=True, backend_a="python", backend_b="python")
        assert result["dags_discarded_unreachable"] > 0

    def test_gate2_self_comparison_passes(self) -> None:
        result, _ = _run_gate2(quick=True, backend_a="python", backend_b="python")
        assert result["pass"] is True
        assert result["mismatches"] == 0

    def test_gate2_k_distribution_covers_range(self) -> None:
        result, _ = _run_gate2(quick=True, backend_a="python", backend_b="python")
        ks = [entry["k"] for entry in result["k_distribution"]]
        assert max(ks) >= 2, f"k_distribution too low: {ks}"
        assert all(entry["count"] > 0 for entry in result["k_distribution"])


class TestGate3Quick:
    """Gate 3 produces a well-formed result dict in quick mode."""

    @pytest.fixture(scope="class")
    def corpus(self) -> list[Any]:
        _, corp = _run_gate2(quick=True, backend_a="python", backend_b="python")
        return corp

    def test_gate3_structure(self, corpus: list[Any]) -> None:
        result = _run_gate3(backend_a="python", backend_b="python", corpus=corpus)
        assert isinstance(result["dags_tested"], int)
        assert isinstance(result["comparisons_made"], int)
        assert isinstance(result["mismatches_engine_a"], int)
        assert isinstance(result["mismatches_engine_b"], int)
        assert isinstance(result["errors"], int)
        assert isinstance(result["pass"], bool)

    def test_gate3_self_comparison_passes(self, corpus: list[Any]) -> None:
        result = _run_gate3(backend_a="python", backend_b="python", corpus=corpus)
        assert result["pass"] is True
        assert result["mismatches_engine_a"] == 0
        assert result["mismatches_engine_b"] == 0


# ---------------------------------------------------------------------------
# End-to-end CLI smoke tests
# ---------------------------------------------------------------------------


class TestMainCLI:
    """End-to-end test: main() with --quick produces a well-formed JSON report."""

    def test_main_quick_all_gates_report_structure(self, tmp_path: Any) -> None:
        """Report contains all required keys regardless of self-comparison state."""
        out = str(tmp_path / "report.json")
        # Do not assert exit code here — it depends on C++ availability.
        main(["--gate", "all", "--quick", "--out", out])
        assert os.path.isfile(out)

        with open(out, encoding="utf-8") as fh:
            report = json.load(fh)

        # Top-level required keys including new ones.
        for key in (
            "schema_version",
            "pass",
            "reason",
            "self_comparison",
            "provenance",
            "gate1",
            "gate2",
            "gate3",
        ):
            assert key in report, f"Missing top-level key: {key}"

        assert isinstance(report["pass"], bool)
        assert isinstance(report["self_comparison"], bool)

        prov = report["provenance"]
        for pkey in (
            "git_commit",
            "engine_a",
            "engine_b",
            "capability_probe",
            "seed",
            "quick_mode",
            "elapsed_seconds",
            "gates_run",
        ):
            assert pkey in prov, f"Missing provenance key: {pkey}"

        assert prov["seed"] == FIXED_SEED
        assert prov["quick_mode"] is True
        assert isinstance(prov["elapsed_seconds"], float)

        # capability_probe must record how the determination was made.
        probe = prov["capability_probe"]
        assert "method" in probe
        assert "cpp_canonical_available" in probe
        assert "detail" in probe

        # Gate counts must be integers.
        for gate_key in ("gate1", "gate2", "gate3"):
            g = report[gate_key]
            assert g is not None, f"{gate_key} is None in report"
            assert isinstance(g["dags_tested"], int)
            assert isinstance(g["comparisons_made"], int)
            assert isinstance(g["pass"], bool)

        # Gate 2 must have discard stats.
        g2 = report["gate2"]
        assert isinstance(g2["dags_generated"], int)
        assert isinstance(g2["dags_discarded_unreachable"], int)
        assert isinstance(g2["discard_rate"], float)

    def test_self_comparison_forces_top_level_fail(self, tmp_path: Any) -> None:
        """When self_comparison=True the top-level pass MUST be False and
        reason MUST be a non-empty string.  This is the core correctness
        requirement: a self-comparison may not be reported as a passing
        equivalence gate."""
        out = str(tmp_path / "self.json")
        exit_code = main(
            [
                "--gate",
                "all",
                "--quick",
                "--backend-a",
                "python",
                "--backend-b",
                "python",
                "--out",
                out,
            ]
        )
        with open(out, encoding="utf-8") as fh:
            report = json.load(fh)

        assert report["self_comparison"] is True
        assert report["pass"] is False, "self_comparison=True must force pass=False; got pass=True"
        assert report["reason"] is not None
        assert len(report["reason"]) > 0
        assert exit_code == 1

    def test_self_comparison_note_present(self, tmp_path: Any) -> None:
        out = str(tmp_path / "note.json")
        main(
            [
                "--gate",
                "all",
                "--quick",
                "--backend-a",
                "python",
                "--backend-b",
                "python",
                "--out",
                out,
            ]
        )
        with open(out, encoding="utf-8") as fh:
            report = json.load(fh)
        assert report["self_comparison"] is True
        assert report["self_comparison_note"] is not None
        assert "python" in report["self_comparison_note"].lower()

    def test_default_backend_b_cpp_is_self_comparison_currently(self, tmp_path: Any) -> None:
        """With no C++ canonicaliser present, --backend-b cpp (default) must
        fall back to python and produce self_comparison=True with pass=False."""
        out = str(tmp_path / "default.json")
        available, _ = _probe_cpp_canonical()
        if available:
            pytest.skip("C++ canonicaliser is present; skip absence behaviour test")

        exit_code = main(["--gate", "all", "--quick", "--out", out])
        with open(out, encoding="utf-8") as fh:
            report = json.load(fh)

        assert report["self_comparison"] is True
        assert report["provenance"]["engine_b"] == "python"
        assert report["pass"] is False
        assert exit_code == 1

    def test_main_single_gate_structure(self, tmp_path: Any) -> None:
        out = str(tmp_path / "gate1.json")
        main(["--gate", "1", "--quick", "--out", out])

        with open(out, encoding="utf-8") as fh:
            report = json.load(fh)

        assert report["gate1"] is not None
        assert report["gate2"] is None
        assert report["gate3"] is None
        # capability_probe must still be present even for a single gate.
        assert "capability_probe" in report["provenance"]
