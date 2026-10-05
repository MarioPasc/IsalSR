"""SRBench black-box track: the 20 real-world PMLB datasets with at most five features.

Supplementary experiment for reviewer item R3.2 of the minor revision. These are
the regression problems of SRBench's black-box track (La Cava et al., NeurIPS
2021 Datasets & Benchmarks, Sec. 4.2), filtered by the rule pre-registered in
the revision plan (decision D6) before any run:

1. population: the 122 datasets of ``results/black-box_results.feather`` at
   SRBench tag ``v2.0``;
2. real-world only: drop the 62 rows SRBench flags ``friedman_dataset``;
3. at most five input features, the dimensional range of the 70-problem suite,
   counted in the PMLB snapshot SRBench ran (``pmlb==1.0.1.post3``).

The rule is applied by ``experiments/scripts/blackbox/select_datasets.py``, which
also vendors the files; the 20 survivors are hard-coded below exactly as the
Strogatz keys are in ``strogatz.py``, and ``tests/unit/test_srbench_blackbox.py``
checks the table against the vendored ``manifest.csv``. ``titanic`` is the
20th: upstream PMLB replaced it in 2022 by an 8-feature dataset that SRBench
never ran (see ``data/srbench_blackbox/PROVENANCE.md``).

There is **no ground-truth expression**. Problem dicts carry neither
``expression`` nor ``sympy_expression``, so the orchestrator's
``_get_ground_truth_sympy`` returns ``None`` and both translators leave
``solution_recovered=False`` and ``jaccard_index=0.0`` without attempting a
comparison. Neither field is reported for this suite.

Data protocol (SRBench v2.0, ``experiment/evaluate_model.py``)
---------------------------------------------------------------
* **Split.** ``train_test_split(X, y, train_size=0.75, test_size=0.25,
  random_state=seed)`` (L43-46 at tag v2.0; L77-80 on SRBench master). The split
  depends only on ``n`` and ``seed``, so it is computed on the row indices.
* **Subsampling.** SRBench caps training data at 10,000 rows (v2.0:
  ``len(labels) > n_samples``; master: ``len(y_train) > max_train_samples``).
  The largest dataset here has 5,300 rows, so the cap is inactive under either
  formulation; ``generate_data`` asserts it rather than implementing it.
* **Scaling.** ``scale_x = scale_y = True`` for the black-box track: X and y are
  standardised with ``StandardScaler`` fitted on the **training fold only**
  (v2.0 L56-70). The test fold is transformed with the training statistics and
  never contributes to them, so no test information reaches the search.
* **Scoring.** SRBench inverse-transforms the predictions and scores them
  against the unscaled test target (v2.0 L194). We instead score in the scaled
  space, which gives the same R2 and NRMSE. With training statistics
  :math:`\\mu, \\sigma > 0`, scaled values are :math:`y' = (y-\\mu)/\\sigma` and
  :math:`\\hat y' = (\\hat y-\\mu)/\\sigma`, so every residual scales by
  :math:`1/\\sigma` and :math:`\\bar{y'} = (\\bar y-\\mu)/\\sigma`, whence

  .. math::

     R^2(y', \\hat y') = 1 - \\frac{\\sum (y'_i-\\hat y'_i)^2}{\\sum (y'_i-\\bar{y'})^2}
       = 1 - \\frac{\\sigma^{-2}\\sum (y_i-\\hat y_i)^2}{\\sigma^{-2}\\sum (y_i-\\bar y)^2}
       = R^2(y, \\hat y),

  and :math:`\\mathrm{NRMSE} = \\mathrm{RMSE}/\\mathrm{std}(y_{test})` has
  numerator and denominator both scaled by :math:`1/\\sigma`, so it is invariant
  too. The non-finite scoring policy (``R2 = 0``, ``NRMSE = 1``) is preserved
  because a finite affine map preserves finiteness. ``mse_test`` is **not**
  invariant: it is in standardised units (:math:`\\mathrm{MSE}/\\sigma^2`) and is
  not comparable to SRBench's ``mse_test``.
* **Seeds.** One seed controls both the split and the host's RNG, as in SRBench
  ("a different random state that controlled both the train/test split and the
  seed of the algorithm"). The campaign uses seeds 1-10; SRBench's own ten random
  states (``experiment/seeds.py``) are not reused, so individual splits differ
  from SRBench's while the procedure is the same.

The feature matrix is read as SRBench's ``read_file`` reads it: every column
except ``target``, in file order, cast to ``float``.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

_DATA_DIR = Path(__file__).parent / "data" / "srbench_blackbox"
MANIFEST_PATH = _DATA_DIR / "manifest.csv"

#: PMLB commit the files were copied from: tag ``v1.0.1.post3``, the PMLB release
#: SRBench v2.0 pins in ``environment.yml`` (``pmlb==1.0.1.post3``). Choice and
#: content-identity evidence: ``data/srbench_blackbox/PROVENANCE.md``.
PMLB_COMMIT = "8eec4f9d1578c7ff1cbfa3efd8338a301adebe52"
PMLB_TAG = "v1.0.1.post3"

#: SRBench black-box protocol constants (evaluate_model.py at tag v2.0).
TRAIN_SIZE = 0.75
TEST_SIZE = 0.25
MAX_TRAIN_SAMPLES = 10_000

TARGET_COLUMN = "target"

#: The 20 datasets selected by PLAN D6, as ``(PMLB name, rows, features)``.
#: Rows and features are PMLB's summary statistics; the tests check them
#: against the vendored files and the manifest.
SRBENCH_BLACKBOX_DATASETS: tuple[tuple[str, int, int], ...] = (
    ("1027_ESL", 488, 4),
    ("1029_LEV", 1000, 4),
    ("1030_ERA", 1000, 4),
    ("1096_FacultySalaries", 50, 4),
    ("192_vineyard", 52, 2),
    ("210_cloud", 108, 5),
    ("228_elusage", 55, 2),
    ("485_analcatdata_vehicle", 48, 4),
    ("519_vinnie", 380, 2),
    ("523_analcatdata_neavote", 100, 2),
    ("529_pollen", 3848, 4),
    ("556_analcatdata_apnea2", 475, 3),
    ("557_analcatdata_apnea1", 475, 3),
    ("663_rabe_266", 120, 2),
    ("678_visualizing_environmental", 111, 3),
    ("687_sleuth_ex1605", 62, 5),
    ("690_visualizing_galaxy", 323, 4),
    ("712_chscase_geyser1", 222, 2),
    ("banana", 5300, 2),
    ("titanic", 2201, 3),
)

_Array = np.ndarray[Any, np.dtype[Any]]


class SRBenchBlackboxError(ValueError):
    """Raised when a vendored file or a split violates the black-box protocol."""


# ----------------------------------------------------------------------
# Vendored-data loading
# ----------------------------------------------------------------------


def data_path(name: str) -> Path:
    """Return the absolute path of a vendored PMLB file.

    Args:
        name: PMLB dataset name, e.g. ``"banana"``.

    Returns:
        Path to ``data/srbench_blackbox/<name>.tsv.gz``, resolved relative to
        this module so it works from any working directory (SLURM included).
    """
    return _DATA_DIR / f"{name}.tsv.gz"


@cache
def load_manifest() -> dict[str, dict[str, str]]:
    """Read ``manifest.csv`` into a mapping keyed by dataset name.

    Returns:
        ``{dataset: row}`` with every manifest column as a string.
    """
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        return {row["dataset"]: row for row in csv.DictReader(handle)}


def file_sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@cache
def load_published(name: str) -> tuple[_Array, _Array, tuple[str, ...]]:
    """Load one vendored dataset after checking its bytes against the manifest.

    The SHA-256 check runs once per process (the result is cached). It makes a
    silently altered or truncated file impossible to use.

    Args:
        name: PMLB dataset name.

    Returns:
        ``(features, target, feature_names)``: ``features`` has shape
        ``(n, d)`` with columns in file order, ``target`` has shape ``(n,)``,
        both ``float64``.

    Raises:
        SRBenchBlackboxError: If the file's SHA-256 differs from the manifest,
            the header lacks ``target``, or the shape differs from the
            pre-registered ``(n, d)``.
    """
    spec = _dataset_spec(name)
    path = data_path(name)
    expected = load_manifest()[name]["sha256"]
    actual = file_sha256(path)
    if actual != expected:
        raise SRBenchBlackboxError(f"{path}: sha256 {actual} != manifest {expected}")

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        table = np.loadtxt(handle, dtype=np.float64, delimiter="\t", ndmin=2)

    if header.count(TARGET_COLUMN) != 1:
        raise SRBenchBlackboxError(f"{path}: expected one '{TARGET_COLUMN}' column, got {header}")
    _, n_rows, n_features = spec
    if table.shape != (n_rows, n_features + 1):
        raise SRBenchBlackboxError(
            f"{path}: shape {table.shape} != pre-registered ({n_rows}, {n_features + 1})"
        )

    t_col = header.index(TARGET_COLUMN)
    f_cols = [i for i in range(len(header)) if i != t_col]
    feature_names = tuple(header[i] for i in f_cols)
    return table[:, f_cols], table[:, t_col], feature_names


def _dataset_spec(name: str) -> tuple[str, int, int]:
    """Return the pre-registered ``(name, n, d)`` entry for ``name``."""
    for spec in SRBENCH_BLACKBOX_DATASETS:
        if spec[0] == name:
            return spec
    raise SRBenchBlackboxError(f"Unknown srbench_blackbox dataset: {name}")


# ----------------------------------------------------------------------
# Protocol: split, cap, scaling
# ----------------------------------------------------------------------


def split_indices(n_rows: int, seed: int) -> tuple[_Array, _Array]:
    """Return SRBench's train/test row indices for ``n_rows`` rows.

    Calls ``sklearn.model_selection.train_test_split`` exactly as SRBench does
    (``train_size=0.75, test_size=0.25, random_state=seed``) on the index
    vector. The resulting partition depends only on ``n_rows`` and ``seed``, so
    it is the partition SRBench's call on ``(features, labels)`` would produce.

    Args:
        n_rows: Number of rows in the dataset.
        seed: The random state.

    Returns:
        ``(train_idx, test_idx)`` as integer arrays, in sklearn's order.
    """
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    train_idx, test_idx = train_test_split(
        np.arange(n_rows),
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        random_state=seed,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def assert_subsampling_inactive(n_rows: int, n_train: int) -> None:
    """Assert that SRBench's 10,000-row training cap would not fire.

    Both SRBench formulations are checked: v2.0 tests the whole dataset
    (``len(labels) > n_samples``), master tests the training fold
    (``len(y_train) > max_train_samples``).

    Raises:
        SRBenchBlackboxError: If either formulation would subsample.
    """
    if n_rows > MAX_TRAIN_SAMPLES or n_train > MAX_TRAIN_SAMPLES:
        raise SRBenchBlackboxError(
            f"{n_rows} rows / {n_train} training rows exceed the {MAX_TRAIN_SAMPLES}-row "
            "cap; SRBench would subsample and this module does not implement it"
        )


def standardize(
    x_train: _Array, y_train: _Array, x_test: _Array, y_test: _Array
) -> tuple[_Array, _Array, _Array, _Array]:
    """Standardise X and y with statistics of the training fold only.

    Uses ``sklearn.preprocessing.StandardScaler`` as SRBench does, including its
    treatment of a zero-variance feature column (scale 1, centred only), which
    can occur in a small training fold of a binary feature. The test fold is
    only ever passed to ``transform``.

    Args:
        x_train: Training features, ``(n_train, d)``.
        y_train: Training target, ``(n_train,)``.
        x_test: Test features, ``(n_test, d)``.
        y_test: Test target, ``(n_test,)``.

    Returns:
        ``(x_train_s, y_train_s, x_test_s, y_test_s)``.

    Raises:
        SRBenchBlackboxError: If the training target is constant, which leaves
            R2 undefined and the affine-invariance argument without a scale.
    """
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    if not np.std(y_train) > 0:
        raise SRBenchBlackboxError("constant training target: cannot standardise y")
    sc_x = StandardScaler().fit(x_train)
    sc_y = StandardScaler().fit(y_train.reshape(-1, 1))
    return (
        sc_x.transform(x_train),
        sc_y.transform(y_train.reshape(-1, 1)).ravel(),
        sc_x.transform(x_test),
        sc_y.transform(y_test.reshape(-1, 1)).ravel(),
    )


# ----------------------------------------------------------------------
# Benchmark registry
# ----------------------------------------------------------------------


def _make_blackbox(name: str, n_rows: int, n_features: int) -> dict[str, Any]:
    """Create a black-box benchmark specification dict (no ground truth)."""
    return {
        "name": name,
        "num_variables": n_features,
        "n_rows": n_rows,
        "sampling": {"type": "srbench_blackbox"},
    }


SRBENCH_BLACKBOX_BENCHMARKS: list[dict[str, Any]] = [
    _make_blackbox(name, n_rows, n_features)
    for name, n_rows, n_features in SRBENCH_BLACKBOX_DATASETS
]


def generate_data(
    benchmark: dict[str, Any],
    n_samples: int = 0,
    train_ratio: float = TRAIN_SIZE,
    seed: int = 42,
) -> tuple[_Array, _Array, _Array, _Array]:
    """Return SRBench's black-box train/test data for one dataset and seed.

    Same signature as ``strogatz.generate_data`` for orchestrator
    compatibility. ``n_samples`` and ``train_ratio`` are ignored: the rows are
    published and the 75/25 split is fixed by the protocol.

    Args:
        benchmark: An entry of ``SRBENCH_BLACKBOX_BENCHMARKS``.
        n_samples: Ignored.
        train_ratio: Ignored.
        seed: Random state of the split (the host receives the same seed).

    Returns:
        ``(X_train, y_train, X_test, y_test)``, standardised with training-fold
        statistics.

    Raises:
        SRBenchBlackboxError: If the sampling type is not ``srbench_blackbox``,
            the data fail their integrity checks, or the 10k cap would fire.
    """
    del n_samples, train_ratio  # fixed by the protocol, see docstring
    if benchmark["sampling"]["type"] != "srbench_blackbox":
        raise SRBenchBlackboxError(f"Unknown sampling type: {benchmark['sampling']['type']}")

    features, target, _ = load_published(benchmark["name"])
    train_idx, test_idx = split_indices(features.shape[0], seed)
    assert_subsampling_inactive(features.shape[0], train_idx.shape[0])
    return standardize(features[train_idx], target[train_idx], features[test_idx], target[test_idx])


def get_benchmark(name: str) -> dict[str, Any]:
    """Get a black-box benchmark by its PMLB name.

    Raises:
        SRBenchBlackboxError: If no benchmark carries that name.
    """
    for bench in SRBENCH_BLACKBOX_BENCHMARKS:
        if bench["name"] == name:
            return bench
    raise SRBenchBlackboxError(f"Unknown srbench_blackbox benchmark: {name}")
