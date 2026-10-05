"""Apply the pre-registered selection rule (PLAN D6) and vendor the SRBench black-box files.

Rule, fixed before any run:

1. population: the unique ``dataset`` values of SRBench's
   ``results/black-box_results.feather`` at tag ``v2.0``;
2. drop every dataset SRBench flags ``friedman_dataset``;
3. keep ``n_features <= 5``.

For each survivor the script downloads ``<name>.tsv.gz`` byte-verbatim from PMLB
at the pinned commit, records its SHA-256, checks its shape against PMLB's
summary statistics at the pinned commit, checks that SRBench's own
``read_file`` semantics (pandas, sniffed separator, ``target`` dropped, cast to
float) and this repository's loader agree (same feature order; values within
``PARSE_RTOL``, the measured maximum is recorded), and writes ``manifest.csv``.
Feature-type counts come from the pinned statistics; PMLB's later curated
counts (``--types-ref``) are added only where the upstream file is unchanged.

The selection must reproduce the 19 names hard-coded in
``benchmarks.datasets.srbench_blackbox``; any difference aborts with status 1.

Requirements: network access, ``pandas`` and ``pyarrow`` (for the feather file).
``pyarrow`` is not part of the ``isalsr`` environment; run this script from any
environment that has it, from the ``code/`` directory::

    python -m experiments.scripts.blackbox.select_datasets --write

The unit tests never run this script: they read the vendored manifest offline.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import logging
import sys
import tempfile
import urllib.request
from dataclasses import asdict, dataclass

from benchmarks.datasets.srbench_blackbox import (
    MANIFEST_PATH,
    PMLB_COMMIT,
    SRBENCH_BLACKBOX_DATASETS,
    TARGET_COLUMN,
    data_path,
)

log = logging.getLogger("select_datasets")

SRBENCH_TAG = "v2.0"
FEATHER_URL = (
    f"https://github.com/cavalab/srbench/raw/{SRBENCH_TAG}/results/black-box_results.feather"
)
PMLB_RAW = "https://github.com/EpistasisLab/pmlb/raw/{ref}/datasets/{name}/{name}.tsv.gz"
PMLB_STATS = "https://raw.githubusercontent.com/EpistasisLab/pmlb/{ref}/pmlb/all_summary_stats.tsv"
PMLB_API_COMMIT = "https://api.github.com/repos/EpistasisLab/pmlb/commits/{ref}"
MAX_FEATURES = 5
#: Tolerance for agreement between this repository's parser and SRBench's.
PARSE_RTOL = 1e-12


@dataclass(frozen=True)
class ManifestRow:
    """One vendored dataset, as written to ``manifest.csv``."""

    dataset: str
    n_samples: int
    n_features: int
    n_binary: int
    n_categorical: int
    n_continuous: int
    n_binary_curated: str
    n_categorical_curated: str
    n_continuous_curated: str
    srbench_parse_max_rel_diff: str
    drift_vs_types_ref: str
    feature_names: str
    sha256: str
    size_bytes: int
    pmlb_commit: str
    source_url: str
    feature_types_ref: str


def _fetch(url: str) -> bytes:
    """Download ``url`` and return its bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": "isalsr-select-datasets"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 -- fixed https URLs
        data: bytes = resp.read()
    return data


def resolve_ref(ref: str) -> str:
    """Resolve a PMLB branch or tag name to a commit SHA through the GitHub API."""
    payload = json.loads(_fetch(PMLB_API_COMMIT.format(ref=ref)))
    return str(payload["sha"])


def load_population(feather_source: str) -> dict[str, bool]:
    """Return ``{dataset: friedman_flag}`` for the black-box population.

    Args:
        feather_source: Local path or URL of ``black-box_results.feather``.

    Returns:
        One entry per unique dataset.
    """
    import pandas as pd  # noqa: PLC0415 -- optional dependency of this script only

    if feather_source.startswith("http"):
        raw = _fetch(feather_source)
        log.info("feather %s sha256=%s", feather_source, hashlib.sha256(raw).hexdigest())
        with tempfile.NamedTemporaryFile(suffix=".feather") as tmp:
            tmp.write(raw)
            tmp.flush()
            frame = pd.read_feather(tmp.name)
    else:
        frame = pd.read_feather(feather_source)
    flags = frame.groupby("dataset")["friedman_dataset"].first()
    return {str(k): bool(v) for k, v in flags.items()}


def load_stats(ref: str) -> dict[str, dict[str, str]]:
    """Return PMLB's ``all_summary_stats.tsv`` at ``ref``, keyed by dataset."""
    text = _fetch(PMLB_STATS.format(ref=ref)).decode("utf-8")
    return {row["dataset"]: row for row in csv.DictReader(io.StringIO(text), delimiter="\t")}


def select(population: dict[str, bool], stats: dict[str, dict[str, str]]) -> list[str]:
    """Apply rules 2 and 3 of PLAN D6 and return the sorted survivors.

    Raises:
        KeyError: If a population dataset has no PMLB summary statistics.
    """
    keep = [
        name
        for name, is_friedman in population.items()
        if not is_friedman and int(stats[name]["n_features"]) <= MAX_FEATURES
    ]
    return sorted(keep)


def srbench_read_file(raw: bytes) -> tuple[list[list[float]], list[float], list[str]]:
    """Read a PMLB file exactly as SRBench v2.0 ``experiment/read_file.py`` does."""
    import pandas as pd  # noqa: PLC0415

    frame = pd.read_csv(io.BytesIO(raw), sep=None, compression="gzip", engine="python")
    names = [c for c in frame.columns if c != TARGET_COLUMN]
    x = frame.drop(TARGET_COLUMN, axis=1).values.astype(float)
    y = frame[TARGET_COLUMN].values.astype(float)
    return x.tolist(), y.tolist(), names


def max_rel_diff(
    x_a: list[list[float]], x_b: list[list[float]], y_a: list[float], y_b: list[float]
) -> float:
    """Return the largest elementwise relative difference between two parses."""
    pairs = [(a, b) for ra, rb in zip(x_a, x_b, strict=True) for a, b in zip(ra, rb, strict=True)]
    pairs += list(zip(y_a, y_b, strict=True))
    return max((abs(a - b) / max(abs(a), 1e-300) for a, b in pairs), default=0.0)


def drift_at(name: str, ref: str, raw: bytes, header: list[str], rows: list[list[float]]) -> str:
    """Classify how PMLB's file at ``ref`` differs from the vendored one.

    Returns:
        ``"none"`` (byte-identical), ``"format-only"`` (bytes differ, same
        header and numerically identical table) or ``"content"`` (the dataset
        was changed or replaced upstream).
    """
    other_raw = _fetch(PMLB_RAW.format(ref=ref, name=name))
    if other_raw == raw:
        return "none"
    lines = [ln for ln in gzip.decompress(other_raw).decode("utf-8").splitlines() if ln]
    if lines[0].split("\t") != header:
        return "content"
    try:
        other = [[float(v) for v in ln.split("\t")] for ln in lines[1:]]
    except ValueError:  # e.g. empty cells in a replaced upstream file
        return "content"
    return "format-only" if other == rows else "content"


def vendor_one(
    name: str,
    pin_stats: dict[str, dict[str, str]],
    type_stats: dict[str, dict[str, str]],
    types_ref: str,
    write: bool,
) -> ManifestRow:
    """Download, verify and optionally write one dataset; return its manifest row.

    Raises:
        RuntimeError: If any shape or parser cross-check fails.
    """
    url = PMLB_RAW.format(ref=PMLB_COMMIT, name=name)
    raw = _fetch(url)
    text = gzip.decompress(raw).decode("utf-8")
    lines = [ln for ln in text.splitlines() if ln]
    header = lines[0].split("\t")
    n_rows, n_cols = len(lines) - 1, len(header)

    problems: list[str] = []
    row = pin_stats[name]
    if (int(row["n_instances"]), int(row["n_features"])) != (n_rows, n_cols - 1):
        problems.append(
            f"shape {(n_rows, n_cols - 1)} != pinned stats "
            f"({row['n_instances']}, {row['n_features']})"
        )
    spec = {s[0]: s for s in SRBENCH_BLACKBOX_DATASETS}[name]
    if (spec[1], spec[2]) != (n_rows, n_cols - 1):
        problems.append(f"shape {(n_rows, n_cols - 1)} != pre-registered {spec[1:]}")

    # Our loader parses each decimal string to the nearest double (np.loadtxt and
    # float() agree exactly); pandas' parser is not always correctly rounded, so
    # SRBench's arrays may differ in the last bits. Require agreement to 1e-12
    # relative and report the measured maximum.
    x_sr, y_sr, names_sr = srbench_read_file(raw)
    t_col = header.index(TARGET_COLUMN)
    ours = [[float(v) for v in ln.split("\t")] for ln in lines[1:]]
    x_ours = [[r[i] for i in range(n_cols) if i != t_col] for r in ours]
    y_ours = [r[t_col] for r in ours]
    if names_sr != [h for h in header if h != TARGET_COLUMN]:
        problems.append(f"feature order differs from SRBench read_file: {names_sr}")
    rel = max_rel_diff(x_ours, x_sr, y_ours, y_sr)
    if not rel <= PARSE_RTOL:
        problems.append(f"parse differs from SRBench read_file by {rel:.3g} (> {PARSE_RTOL})")
    log.info("%s: max relative difference to SRBench read_file = %.3g", name, rel)
    if problems:
        raise RuntimeError(f"{name}: " + "; ".join(problems))

    if write:
        data_path(name).write_bytes(raw)
    types = pin_stats[name]
    # PMLB's curated feature types describe the CURRENT upstream file, so they
    # are recorded only where that file still holds the vendored table.
    drift = drift_at(name, types_ref, raw, header, ours)
    curated = type_stats[name] if drift != "content" else None
    return ManifestRow(
        dataset=name,
        n_samples=n_rows,
        n_features=n_cols - 1,
        n_binary=int(types["n_binary_features"]),
        n_categorical=int(types["n_categorical_features"]),
        n_continuous=int(types["n_continuous_features"]),
        n_binary_curated=curated["n_binary_features"] if curated else "",
        n_categorical_curated=curated["n_categorical_features"] if curated else "",
        n_continuous_curated=curated["n_continuous_features"] if curated else "",
        srbench_parse_max_rel_diff=f"{rel:.3g}",
        drift_vs_types_ref=drift,
        feature_names=";".join(h for h in header if h != TARGET_COLUMN),
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        pmlb_commit=PMLB_COMMIT,
        source_url=url,
        feature_types_ref=types_ref,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the selection; return 0 on an exact match with the pre-registered 19."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feather", default=FEATHER_URL, help="path or URL of the feather")
    parser.add_argument(
        "--types-ref",
        default="master",
        help="PMLB ref whose summary stats give the feature-type counts (resolved to a SHA)",
    )
    parser.add_argument("--write", action="store_true", help="write the files and manifest")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    population = load_population(args.feather)
    types_sha = resolve_ref(args.types_ref)
    pin_stats = load_stats(PMLB_COMMIT)
    type_stats = load_stats(types_sha)
    log.info(
        "population=%d friedman=%d; pinned PMLB %s; feature types from %s=%s",
        len(population),
        sum(population.values()),
        PMLB_COMMIT,
        args.types_ref,
        types_sha,
    )

    selected = select(population, pin_stats)
    expected = sorted(s[0] for s in SRBENCH_BLACKBOX_DATASETS)
    if selected != expected:
        log.error("selection %s != pre-registered %s", selected, expected)
        return 1
    log.info("selection reproduces the %d pre-registered datasets", len(selected))

    rows = [vendor_one(n, pin_stats, type_stats, types_sha, args.write) for n in selected]
    total = sum(r.size_bytes for r in rows)
    log.info("all %d files verified; %d bytes in total", len(rows), total)
    if args.write:
        with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
            writer.writeheader()
            writer.writerows(asdict(r) for r in rows)
        log.info("wrote %s", MANIFEST_PATH)
    for r in rows:
        print(f"{r.dataset:32s} n={r.n_samples:5d} d={r.n_features} sha256={r.sha256[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
