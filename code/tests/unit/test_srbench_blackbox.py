"""SRBench black-box suite: vendored data, SRBench protocol, registry (R3.2).

Offline: every check reads the vendored files and ``manifest.csv``; nothing here
touches the network. The network-dependent selection step is
``experiments/scripts/blackbox/select_datasets.py``.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from benchmarks.datasets import srbench_blackbox as bbx
from experiments.models import orchestrator
from experiments.models.analyzer.metrics import nrmse, r_squared

NAMES = [name for name, _, _ in bbx.SRBENCH_BLACKBOX_DATASETS]
SEEDS = tuple(range(1, 11))


# ----------------------------------------------------------------------
# Manifest and vendored files
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest() -> list[dict[str, str]]:
    with bbx.MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestManifest:
    def test_twenty_rows_matching_the_preregistered_table(self, manifest) -> None:
        assert len(manifest) == 20
        assert [r["dataset"] for r in manifest] == sorted(NAMES)
        table = {n: (rows, d) for n, rows, d in bbx.SRBENCH_BLACKBOX_DATASETS}
        for r in manifest:
            assert (int(r["n_samples"]), int(r["n_features"])) == table[r["dataset"]]

    def test_at_most_five_features(self, manifest) -> None:
        assert max(int(r["n_features"]) for r in manifest) <= 5

    def test_no_friedman_dataset(self, manifest) -> None:
        # SRBench's synthetic Friedman generators are PMLB's ``fri_c*`` datasets.
        assert not [r["dataset"] for r in manifest if r["dataset"].startswith("fri_")]

    def test_pinned_commit_and_source(self, manifest) -> None:
        for r in manifest:
            assert r["pmlb_commit"] == bbx.PMLB_COMMIT
            assert bbx.PMLB_COMMIT in r["source_url"]
            assert r["source_url"].endswith(f"/{r['dataset']}/{r['dataset']}.tsv.gz")

    @pytest.mark.parametrize("name", NAMES)
    def test_sha256_of_vendored_file(self, manifest, name: str) -> None:
        row = next(r for r in manifest if r["dataset"] == name)
        path = bbx.data_path(name)
        assert path.stat().st_size == int(row["size_bytes"])
        assert bbx.file_sha256(path) == row["sha256"]

    def test_vendored_footprint_is_small(self, manifest) -> None:
        assert sum(int(r["size_bytes"]) for r in manifest) < 5_000_000

    def test_feature_type_counts_add_up(self, manifest) -> None:
        for r in manifest:
            parts = int(r["n_binary"]) + int(r["n_categorical"]) + int(r["n_continuous"])
            assert parts == int(r["n_features"])
            assert len(r["feature_names"].split(";")) == int(r["n_features"])

    def test_upstream_replacements_are_recorded(self, manifest) -> None:
        drift = {r["dataset"]: r["drift_vs_types_ref"] for r in manifest}
        # PMLB replaced both after SRBench v2.0; the vendored files are the old ones.
        assert drift["banana"] == drift["titanic"] == "content"
        assert set(drift.values()) <= {"none", "format-only", "content"}


class TestLoading:
    @pytest.mark.parametrize("name", NAMES)
    def test_shape_finite_and_target_excluded(self, name: str) -> None:
        x, y, feature_names = bbx.load_published(name)
        _, n_rows, n_features = next(s for s in bbx.SRBENCH_BLACKBOX_DATASETS if s[0] == name)
        assert x.shape == (n_rows, n_features)
        assert y.shape == (n_rows,)
        assert bbx.TARGET_COLUMN not in feature_names
        assert np.isfinite(x).all() and np.isfinite(y).all()

    def test_unknown_dataset_raises(self) -> None:
        with pytest.raises(bbx.SRBenchBlackboxError):
            bbx.get_benchmark("fri_c0_100_5")


# ----------------------------------------------------------------------
# Split
# ----------------------------------------------------------------------


class TestSplit:
    @pytest.mark.parametrize("name", NAMES)
    def test_sizes_are_sklearn_75_25(self, name: str) -> None:
        n = next(rows for nm, rows, _ in bbx.SRBENCH_BLACKBOX_DATASETS if nm == name)
        tr, te = bbx.split_indices(n, 1)
        n_test = math.ceil(0.25 * n)
        assert (len(tr), len(te)) == (n - n_test, n_test)
        assert len(np.intersect1d(tr, te)) == 0
        assert sorted(np.concatenate([tr, te]).tolist()) == list(range(n))

    @pytest.mark.parametrize("n", [48, 380, 5300])
    @pytest.mark.parametrize("seed", [0, 1, 10, 23654])
    def test_equals_shufflesplit_permutation(self, n: int, seed: int) -> None:
        """sklearn's ShuffleSplit: test = perm[:n_test], train = perm[n_test:]."""
        perm = np.random.RandomState(seed).permutation(n)
        n_test = math.ceil(0.25 * n)
        tr, te = bbx.split_indices(n, seed)
        np.testing.assert_array_equal(te, perm[:n_test])
        np.testing.assert_array_equal(tr, perm[n_test : n_test + (n - n_test)])

    @pytest.mark.parametrize("name", ["485_analcatdata_vehicle", "banana"])
    def test_equals_srbench_call_on_the_data(self, name: str) -> None:
        """Splitting indices equals SRBench's train_test_split(features, labels)."""
        x, y, _ = bbx.load_published(name)
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, train_size=0.75, test_size=0.25, random_state=3
        )
        tr, te = bbx.split_indices(x.shape[0], 3)
        np.testing.assert_array_equal(x[tr], x_tr)
        np.testing.assert_array_equal(y[te], y_te)

    def test_deterministic_per_seed_and_distinct_across_seeds(self) -> None:
        bench = bbx.get_benchmark("228_elusage")
        first = [bbx.generate_data(bench, seed=s) for s in SEEDS]
        again = [bbx.generate_data(bench, seed=s) for s in SEEDS]
        for a, b in zip(first, again, strict=True):
            for u, v in zip(a, b, strict=True):
                np.testing.assert_array_equal(u, v)
        test_targets = {tuple(np.round(f[3], 12)) for f in first}
        assert len(test_targets) == len(SEEDS)


# ----------------------------------------------------------------------
# Scaling: training-fold statistics only
# ----------------------------------------------------------------------


class TestNoLeakage:
    def test_train_fold_is_standardised(self) -> None:
        x_tr, y_tr, _, _ = bbx.generate_data(bbx.get_benchmark("529_pollen"), seed=1)
        np.testing.assert_allclose(x_tr.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(x_tr.std(axis=0), 1.0, rtol=1e-12)
        assert abs(y_tr.mean()) < 1e-12
        assert math.isclose(y_tr.std(), 1.0, rel_tol=1e-12)

    def test_test_fold_never_reaches_the_scaler(self) -> None:
        """Corrupting the test fold must leave the training fold bit-identical."""
        x, y, _ = bbx.load_published("663_rabe_266")
        tr, te = bbx.split_indices(x.shape[0], 1)
        clean = bbx.standardize(x[tr], y[tr], x[te], y[te])
        x_bad, y_bad = x[te] * 1e6 + 7.0, y[te] * -1e6
        dirty = bbx.standardize(x[tr], y[tr], x_bad, y_bad)
        np.testing.assert_array_equal(clean[0], dirty[0])
        np.testing.assert_array_equal(clean[1], dirty[1])
        # And the test fold is mapped with the TRAINING mean and std.
        mu, sd = y[tr].mean(), y[tr].std()
        np.testing.assert_allclose(clean[3], (y[te] - mu) / sd, rtol=1e-12)
        np.testing.assert_allclose(
            clean[2], (x[te] - x[tr].mean(axis=0)) / x[tr].std(axis=0), rtol=1e-12
        )

    def test_matches_srbench_standardscaler(self) -> None:
        x, y, _ = bbx.load_published("1027_ESL")
        tr, te = bbx.split_indices(x.shape[0], 5)
        ours = bbx.standardize(x[tr], y[tr], x[te], y[te])
        sc_x = StandardScaler().fit(x[tr])
        sc_y = StandardScaler().fit(y[tr].reshape(-1, 1))
        np.testing.assert_array_equal(ours[0], sc_x.transform(x[tr]))
        np.testing.assert_array_equal(ours[2], sc_x.transform(x[te]))
        np.testing.assert_array_equal(ours[1], sc_y.transform(y[tr].reshape(-1, 1)).ravel())

    def test_constant_feature_column_in_a_small_fold(self) -> None:
        """A zero-variance column is centred and left unscaled, as in sklearn."""
        x_tr = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        y_tr = np.array([0.0, 1.0, 2.0])
        out = bbx.standardize(x_tr, y_tr, np.array([[2.0, 1.0]]), np.array([1.0]))
        np.testing.assert_array_equal(out[0][:, 0], 0.0)
        assert out[2][0, 0] == 1.0

    def test_constant_training_target_is_refused(self) -> None:
        with pytest.raises(bbx.SRBenchBlackboxError):
            bbx.standardize(np.ones((3, 1)), np.ones(3), np.ones((1, 1)), np.ones(1))


class TestSubsamplingCap:
    def test_inactive_for_every_dataset(self) -> None:
        for _, n_rows, _ in bbx.SRBENCH_BLACKBOX_DATASETS:
            tr, _ = bbx.split_indices(n_rows, 1)
            bbx.assert_subsampling_inactive(n_rows, len(tr))
        assert max(n for _, n, _ in bbx.SRBENCH_BLACKBOX_DATASETS) == 5300

    @pytest.mark.parametrize(("n_rows", "n_train"), [(10_001, 7_500), (14_000, 10_500)])
    def test_raises_when_srbench_would_subsample(self, n_rows: int, n_train: int) -> None:
        with pytest.raises(bbx.SRBenchBlackboxError):
            bbx.assert_subsampling_inactive(n_rows, n_train)


# ----------------------------------------------------------------------
# Affine invariance: scaled-space scores equal SRBench's inverse-transform scores
# ----------------------------------------------------------------------


class TestAffineInvariance:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_r2_and_nrmse_equal_srbench_scoring(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        y_train = rng.normal(37.0, 11.0, size=80)
        y_test = rng.normal(35.0, 12.0, size=27)
        pred_scaled = rng.normal(0.0, 1.0, size=27)  # a model's output in scaled space

        sc_y = StandardScaler().fit(y_train.reshape(-1, 1))
        y_test_scaled = sc_y.transform(y_test.reshape(-1, 1)).ravel()
        pred_orig = sc_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

        # SRBench: inverse-transform the prediction, score on the raw target.
        srbench_r2 = r2_score(y_test, pred_orig)
        ours_r2 = r_squared(y_test_scaled, pred_scaled)
        assert math.isclose(ours_r2, srbench_r2, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(
            nrmse(y_test_scaled, pred_scaled), nrmse(y_test, pred_orig), rel_tol=1e-12
        )

    def test_non_finite_policy_survives_the_map(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.0, np.nan, 3.0, 4.0])
        sc = StandardScaler().fit(y.reshape(-1, 1))
        ys = sc.transform(y.reshape(-1, 1)).ravel()
        ps = sc.transform(pred.reshape(-1, 1)).ravel()
        assert r_squared(ys, ps) == r_squared(y, pred) == 0.0
        assert nrmse(ys, ps) == nrmse(y, pred) == 1.0


# ----------------------------------------------------------------------
# Registry and dispatch
# ----------------------------------------------------------------------


class TestRegistry:
    def test_registered_with_twenty_problems(self) -> None:
        problems, gen = orchestrator._BENCHMARK_REGISTRY["srbench_blackbox"]
        assert gen is bbx.generate_data
        assert [p["name"] for p in problems] == NAMES
        assert len(orchestrator.get_benchmarks("srbench_blackbox", "all")) == 20

    def test_dispatch_ignores_train_and_test_size(self) -> None:
        bench = bbx.get_benchmark("192_vineyard")
        a = orchestrator._generate_benchmark_data("srbench_blackbox", bench, 20, 100, 4)
        b = orchestrator._generate_benchmark_data("srbench_blackbox", bench, 1000, 250, 4)
        c = bbx.generate_data(bench, seed=4)
        for u, v, w in zip(a, b, c, strict=True):
            np.testing.assert_array_equal(u, v)
            np.testing.assert_array_equal(u, w)

    @pytest.mark.parametrize("bench", bbx.SRBENCH_BLACKBOX_BENCHMARKS, ids=NAMES)
    def test_no_ground_truth(self, bench) -> None:
        """No expression, so solution recovery is never attempted."""
        assert orchestrator._get_ground_truth_sympy(bench) is None
