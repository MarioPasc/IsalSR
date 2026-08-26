"""Unit tests for the synthetic-scalability value extractor.

The fixtures are built in ``tmp_path``; nothing here depends on a real run.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from experiments.synthetic_scalability.extract_synthetic_values import (
    EXPECTED_K_VALUES,
    EXPECTED_M_VALUES,
    EXPECTED_N_FRAGMENTS,
    SyntheticDataError,
    aggregate_per_k,
    build_values,
    fit_powerlaw,
    load_fragments,
    main,
    render_latex,
    render_table_body,
    write_outputs,
)

FIELDNAMES: list[str] = [
    "k",
    "m",
    "expr_id",
    "seed",
    "n_nodes",
    "n_edges",
    "n_perms",
    "n_unique_canonicals",
    "rho",
    "rho_over_kfact",
    "canonical_len",
    "mean_canon_time_s",
    "std_canon_time_s",
    "total_canon_time_s",
    "timeout_count",
]


def _write_cell(
    directory: Path,
    k: int,
    m: int,
    *,
    n_expr: int = 4,
    exponent: float = 2.0,
    base_s: float = 1e-5,
    m_shift: float = 0.0,
    unique_override: int | None = None,
    rho_override: float | None = None,
    n_perms_override: int | None = None,
    timeouts: int = 0,
) -> Path:
    """Write one deterministic ``synth_k{K}_m{M}.csv`` fixture fragment."""
    path = directory / f"synth_k{k}_m{m}.csv"
    kfact = math.factorial(k)
    n_perms = n_perms_override if n_perms_override is not None else kfact
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for expr_id in range(n_expr):
            # Deterministic spread around the power law so quartiles differ.
            spread = 1.0 + 0.1 * expr_id
            mean_t = base_s * (k**exponent) * spread * (1.0 + m_shift * (m - 1))
            writer.writerow(
                {
                    "k": k,
                    "m": m,
                    "expr_id": expr_id,
                    "seed": 1000 + expr_id,
                    "n_nodes": k + m,
                    "n_edges": k,
                    "n_perms": n_perms,
                    "n_unique_canonicals": (unique_override if unique_override is not None else 1),
                    "rho": rho_override if rho_override is not None else float(kfact),
                    "rho_over_kfact": 1.0 if rho_override is None else rho_override / kfact,
                    "canonical_len": 3 * k,
                    "mean_canon_time_s": mean_t,
                    "std_canon_time_s": 0.1 * mean_t,
                    "total_canon_time_s": mean_t * n_perms,
                    "timeout_count": timeouts,
                }
            )
    return path


def _write_cell_with_times(directory: Path, k: int, m: int, times_us: list[float]) -> Path:
    """Write a fragment whose per-expression mean times are given, in microseconds."""
    path = directory / f"synth_k{k}_m{m}.csv"
    kfact = math.factorial(k)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for expr_id, t_us in enumerate(times_us):
            mean_t = t_us * 1e-6  # the fragment column is in SECONDS
            writer.writerow(
                {
                    "k": k,
                    "m": m,
                    "expr_id": expr_id,
                    "seed": 1000 + expr_id,
                    "n_nodes": k + m,
                    "n_edges": k,
                    "n_perms": kfact,
                    "n_unique_canonicals": 1,
                    "rho": float(kfact),
                    "rho_over_kfact": 1.0,
                    "canonical_len": 3 * k,
                    "mean_canon_time_s": mean_t,
                    "std_canon_time_s": 0.1 * mean_t,
                    "total_canon_time_s": mean_t * kfact,
                    "timeout_count": 0,
                }
            )
    return path


def _write_full_run(directory: Path, **kwargs: object) -> Path:
    """Write all 27 fixture fragments plus a metadata JSON."""
    directory.mkdir(parents=True, exist_ok=True)
    for k in EXPECTED_K_VALUES:
        for m in EXPECTED_M_VALUES:
            _write_cell(directory, k, m, **kwargs)  # type: ignore[arg-type]
    (directory / "synthetic_scalability_metadata.json").write_text(
        json.dumps(
            {
                "experiment": "synthetic_scalability",
                "hardware": {
                    "cpu_model": "Fixture CPU",
                    "engine": "native",
                    "build_hash": "deadbeef",
                    "isa_level": "x86-64-v3",
                    "cpu_count": 8,
                    "git_hash": "abc123",
                },
            },
            indent=2,
        )
    )
    return directory


# ===================================================================
# Refusal on partial / empty fragment sets
# ===================================================================


def test_empty_directory_raises(tmp_path: Path) -> None:
    """No fragments at all is always an error, partial or not."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SyntheticDataError, match="No fragments"):
        load_fragments(str(empty))
    with pytest.raises(SyntheticDataError, match="No fragments"):
        load_fragments(str(empty), allow_partial=True)


def test_missing_directory_raises(tmp_path: Path) -> None:
    """A non-existent fragments directory is an error."""
    with pytest.raises(SyntheticDataError, match="does not exist"):
        load_fragments(str(tmp_path / "nope"))


@pytest.mark.parametrize("n_cells", [1, 3, EXPECTED_N_FRAGMENTS - 1])
def test_partial_run_refused_by_default(tmp_path: Path, n_cells: int) -> None:
    """Fewer than 27 fragments must refuse to emit unless --allow-partial."""
    part = tmp_path / "partial"
    part.mkdir()
    written = 0
    for k in EXPECTED_K_VALUES:
        for m in EXPECTED_M_VALUES:
            if written >= n_cells:
                break
            _write_cell(part, k, m)
            written += 1
        if written >= n_cells:
            break

    with pytest.raises(SyntheticDataError, match="Refusing to emit"):
        load_fragments(str(part))

    rows, files = load_fragments(str(part), allow_partial=True)
    assert len(files) == n_cells
    assert len(rows) == 4 * n_cells


def test_partial_run_cli_exits_nonzero_and_writes_nothing(tmp_path: Path) -> None:
    """The CLI refuses a partial run and leaves the output directory empty."""
    part = tmp_path / "partial"
    part.mkdir()
    _write_cell(part, 1, 1)
    _write_cell(part, 2, 1)
    out = tmp_path / "out"

    status = main(["--fragments", str(part), "--out", str(out)])
    assert status == 2
    assert not (out / "synthetic_values.json").exists()

    status_ok = main(["--fragments", str(part), "--out", str(out), "--allow-partial"])
    assert status_ok == 0
    payload = json.loads((out / "synthetic_values.json").read_text())
    assert payload["provenance"]["is_complete"] is False
    assert payload["provenance"]["allow_partial"] is True


def test_missing_column_raises(tmp_path: Path) -> None:
    """A fragment lacking a required column is reported by name."""
    bad = tmp_path / "bad"
    bad.mkdir()
    path = _write_cell(bad, 1, 1)
    text = path.read_text().replace("mean_canon_time_s", "mean_time")
    path.write_text(text)
    with pytest.raises(SyntheticDataError, match="mean_canon_time_s"):
        load_fragments(str(bad), allow_partial=True)


# ===================================================================
# Per-k aggregation
# ===================================================================


def test_per_k_medians_match_numpy(tmp_path: Path) -> None:
    """median/q1/q3 equal numpy's on the pooled per-expression means in ms."""
    run = _write_full_run(tmp_path / "run")
    rows, _ = load_fragments(str(run))
    per_k = aggregate_per_k(rows)
    assert [entry["k"] for entry in per_k] == list(EXPECTED_K_VALUES)

    for entry in per_k:
        k = int(entry["k"])
        expected = rows[rows["k"] == k]["mean_canon_time_s"].to_numpy(float) * 1000.0
        np.testing.assert_allclose(entry["median_ms"], float(np.median(expected)), rtol=0, atol=0)
        np.testing.assert_allclose(entry["q1_ms"], float(np.percentile(expected, 25)))
        np.testing.assert_allclose(entry["q3_ms"], float(np.percentile(expected, 75)))
        assert entry["n_expressions"] == 4 * len(EXPECTED_M_VALUES)
        assert entry["n_permutations"] == math.factorial(k) * 4 * len(EXPECTED_M_VALUES)
        assert entry["rho_equals_kfact"] is True
        assert entry["invariance_pct"] == pytest.approx(100.0)
        assert entry["timeouts"] == 0


def test_rho_mismatch_and_invariance_are_detected(tmp_path: Path) -> None:
    """One bad cell flips rho_equals_kfact and lowers invariance_pct."""
    run = tmp_path / "run"
    run.mkdir()
    _write_cell(run, 3, 1)
    _write_cell(run, 3, 2, unique_override=2, rho_override=3.0, timeouts=5)
    rows, _ = load_fragments(str(run), allow_partial=True)
    (entry,) = aggregate_per_k(rows)
    assert entry["rho_equals_kfact"] is False
    assert entry["invariance_pct"] == pytest.approx(50.0)
    assert entry["timeouts"] == 20


def test_no_hardcoded_protocol_counts(tmp_path: Path) -> None:
    """Counts come from the data, never from the 200/600 protocol constants."""
    run = _write_full_run(tmp_path / "run", n_expr=7)
    rows, files = load_fragments(str(run))
    values = build_values(
        rows, files, str(run), allow_partial=False, expected_fragments=EXPECTED_N_FRAGMENTS
    )
    assert values["overall"]["total_expressions"] == 7 * EXPECTED_N_FRAGMENTS
    assert all(entry["n_expressions"] == 21 for entry in values["per_k"])


# ===================================================================
# Power-law fit
# ===================================================================


def test_powerlaw_recovers_planted_exponent(tmp_path: Path) -> None:
    """A noiseless t = a k^b fixture recovers b, including the k=1 rows."""
    run = _write_full_run(tmp_path / "run", exponent=2.5)
    rows, _ = load_fragments(str(run))
    fit = fit_powerlaw(rows)
    np.testing.assert_allclose(fit["exponent"], 2.5, rtol=1e-6)
    assert fit["k_equals_one_included"] is True
    assert fit["n_points"] == len(rows)
    assert fit["r_squared"] > 0.99


def test_powerlaw_k_one_is_not_degenerate(tmp_path: Path) -> None:
    """log(1) = 0 does not break the fit; k=1-only data does."""
    run = tmp_path / "k1only"
    run.mkdir()
    for m in EXPECTED_M_VALUES:
        _write_cell(run, 1, m)
    rows, _ = load_fragments(str(run), allow_partial=True)
    with pytest.raises(SyntheticDataError, match="two distinct k"):
        fit_powerlaw(rows)


# ===================================================================
# m-dependence
# ===================================================================


def test_m_dependence_absent_when_planted_absent(tmp_path: Path) -> None:
    """With no m effect in the fixture, the verdict is 'no dependence'."""
    run = _write_full_run(tmp_path / "run")
    rows, _ = load_fragments(str(run))
    result = build_values(
        rows, [], str(run), allow_partial=False, expected_fragments=EXPECTED_N_FRAGMENTS
    )["m_dependence"]
    assert result["dependence_detected"] is False
    assert result["n_k_tested"] == len(EXPECTED_K_VALUES)


def test_m_dependence_detected_when_planted(tmp_path: Path) -> None:
    """A planted 60%-per-m time increase is reported, not silently ignored."""
    run = _write_full_run(tmp_path / "run", n_expr=30, m_shift=0.6)
    rows, _ = load_fragments(str(run))
    result = build_values(
        rows, [], str(run), allow_partial=False, expected_fragments=EXPECTED_N_FRAGMENTS
    )["m_dependence"]
    assert result["dependence_detected"] is True
    assert result["pooled"]["p"] < 0.05
    assert "DEPENDENCE DETECTED" in result["verdict"]


# ===================================================================
# Emission
# ===================================================================


def test_latex_macros_are_letters_only_and_rounded(tmp_path: Path) -> None:
    """Nine table cells + exponent + k=9 mean + call total, letters-only names."""
    run = _write_full_run(tmp_path / "run")
    rows, files = load_fragments(str(run))
    values = build_values(
        rows, files, str(run), allow_partial=False, expected_fragments=EXPECTED_N_FRAGMENTS
    )
    tex = render_latex(values)

    names = [
        line.split("}")[0].removeprefix("\\newcommand{\\")
        for line in tex.splitlines()
        if line.startswith("\\newcommand")
    ]
    assert all(name.isalpha() for name in names), names
    for word in ("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"):
        assert f"\\newcommand{{\\synthTimeK{word}}}" in tex
    assert "\\synthPowerlawExponent}" in tex
    assert "\\synthMeanMsKNine}" in tex
    assert "\\synthTotalCanonicalizations}" in tex

    # Two decimals, in microseconds, rounded once from the full-precision JSON value.
    k9 = next(e for e in values["per_k"] if e["k"] == 9)
    expected_cell = f"{k9['median_us']:.2f}\\,[{k9['q1_us']:.2f}\\text{{--}}{k9['q3_us']:.2f}]"
    assert f"\\newcommand{{\\synthTimeKNine}}{{{expected_cell}}}" in tex


def test_outputs_written_and_regeneration_is_byte_identical(tmp_path: Path) -> None:
    """Rerunning on unchanged fragments reproduces both files byte-for-byte."""
    run = _write_full_run(tmp_path / "run")
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    assert main(["--fragments", str(run), "--out", str(out_a)]) == 0
    assert main(["--fragments", str(run), "--out", str(out_b)]) == 0

    for name in (
        "synthetic_values.json",
        "synthetic_values.tex",
        "tab_supp_synthetic_body.tex",
    ):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()

    payload = json.loads((out_a / "synthetic_values.json").read_text())
    assert payload["provenance"]["is_complete"] is True
    assert payload["provenance"]["hardware"]["cpu_model"] == "Fixture CPU"
    assert payload["provenance"]["hardware"]["engine"] == "native"
    assert payload["provenance"]["hardware"]["build_hash"] == "deadbeef"
    assert payload["provenance"]["hardware"]["isa_level"] == "x86-64-v3"
    assert len(payload["provenance"]["fragment_mtimes_utc"]) == EXPECTED_N_FRAGMENTS


def test_json_totals_and_k9_mean(tmp_path: Path) -> None:
    """total_canonicalizations and mean_ms_at_k9 come from the rows."""
    run = _write_full_run(tmp_path / "run")
    rows, files = load_fragments(str(run))
    values = build_values(
        rows, files, str(run), allow_partial=False, expected_fragments=EXPECTED_N_FRAGMENTS
    )
    expected_total = sum(math.factorial(k) * 4 * len(EXPECTED_M_VALUES) for k in EXPECTED_K_VALUES)
    assert values["overall"]["total_canonicalizations"] == expected_total
    k9_ms = rows[rows["k"] == 9]["mean_canon_time_s"].to_numpy(float) * 1000.0
    np.testing.assert_allclose(values["overall"]["mean_ms_at_k9"], float(np.mean(k9_ms)))


def test_provenance_without_metadata_file(tmp_path: Path) -> None:
    """A missing metadata JSON yields hardware=None and an explicit note."""
    run = _write_full_run(tmp_path / "run")
    (run / "synthetic_scalability_metadata.json").unlink()
    rows, files = load_fragments(str(run))
    prov = build_values(
        rows, files, str(run), allow_partial=False, expected_fragments=EXPECTED_N_FRAGMENTS
    )["provenance"]
    assert prov["hardware"] is None
    assert "not found" in prov["metadata_note"]


def test_single_expression_cell(tmp_path: Path) -> None:
    """A one-row cell still yields median == q1 == q3 rather than failing."""
    run = tmp_path / "run"
    run.mkdir()
    _write_cell(run, 2, 1, n_expr=1)
    _write_cell(run, 4, 1, n_expr=1)
    rows, _ = load_fragments(str(run), allow_partial=True)
    per_k = aggregate_per_k(rows)
    for entry in per_k:
        assert entry["median_ms"] == entry["q1_ms"] == entry["q3_ms"]


def test_write_outputs_creates_directory(tmp_path: Path) -> None:
    """The output directory is created if it does not exist."""
    run = _write_full_run(tmp_path / "run")
    rows, files = load_fragments(str(run))
    values = build_values(
        rows, files, str(run), allow_partial=False, expected_fragments=EXPECTED_N_FRAGMENTS
    )
    json_path, tex_path, table_path = write_outputs(values, rows, str(tmp_path / "deep" / "nested"))
    assert json_path.is_file()
    assert tex_path.is_file()
    assert table_path.is_file()


# ===================================================================
# Delegated tabular body
# ===================================================================


def test_table_body_is_bare_and_delegates_to_the_figure_generator(tmp_path: Path) -> None:
    """The body carries no float/caption/label and reuses the existing rows."""
    from experiments.synthetic_scalability.generate_fig_synthetic_scalability import (
        _aggregate_per_k,
    )

    run = _write_full_run(tmp_path / "run")
    rows, _ = load_fragments(str(run))
    body = render_table_body(rows)

    assert r"\begin{table}" not in body
    assert r"\caption" not in body
    assert r"\label" not in body
    assert body.count(r"\begin{tabular}") == 1
    assert r"\setlength{\tabcolsep}{3pt}" in body
    assert r"$k$ & $k!$ & Perms & $\rho = k!$ & Invariance & Time ($\mu$s) \\" in body

    # Rows must be byte-identical to the generator's, module indentation.
    generated = _aggregate_per_k(rows.to_dict("records"))
    for entry in generated:
        med = float(entry["median_time_us"])
        lo = float(entry["iqr_lo_us"])
        hi = float(entry["iqr_hi_us"])
        assert f"{med:.2f} [{lo:.2f}--{hi:.2f}]" in body
    assert r"362{,}880" in body  # k = 9


def test_table_body_reports_rho_failure_instead_of_a_checkmark(tmp_path: Path) -> None:
    """A k whose rho != k! must not print \\checkmark."""
    run = tmp_path / "run"
    run.mkdir()
    _write_cell(run, 3, 1)
    _write_cell(run, 3, 2, unique_override=2, rho_override=3.0)
    rows, _ = load_fragments(str(run), allow_partial=True)
    body = render_table_body(rows)
    assert r"\checkmark" not in body
    assert r"$\times$" in body


def test_table_pins_two_real_campaign_rows_in_microseconds(tmp_path: Path) -> None:
    """Pin the k=1 and k=9 cells of the real run so a unit change fails loudly.

    The two rows reproduced here are the measured medians of the 2026-08 retimed
    campaign (600 expressions per k, exhaustive k!, AMD EPYC 7H12, C++ engine):
    k=1 -> ``12.06 [10.84--12.96]`` and k=9 -> ``20.46 [19.13--21.73]``, in
    microseconds. Five per-expression values make ``np.percentile`` land exactly
    on the 2nd and 4th order statistics, so the quartiles are exact.
    """
    run = tmp_path / "real"
    run.mkdir()
    _write_cell_with_times(run, 1, 1, [10.00, 10.84, 12.06, 12.96, 13.50])
    _write_cell_with_times(run, 9, 1, [18.00, 19.13, 20.46, 21.73, 22.90])
    rows, _ = load_fragments(str(run), allow_partial=True)

    per_k = aggregate_per_k(rows)
    np.testing.assert_allclose(per_k[0]["median_us"], 12.06, rtol=1e-9)
    np.testing.assert_allclose(per_k[0]["median_ms"], 12.06e-3, rtol=1e-9)
    np.testing.assert_allclose(per_k[1]["median_us"], 20.46, rtol=1e-9)

    body = render_table_body(rows)
    assert "12.06 [10.84--12.96]" in body
    assert "20.46 [19.13--21.73]" in body
    assert r"Time ($\mu$s)" in body


def test_microsecond_and_millisecond_keys_agree_by_a_factor_of_1000(tmp_path: Path) -> None:
    """``*_us`` is 1000x ``*_ms``; neither key name lies about its unit."""
    run = _write_full_run(tmp_path / "run")
    rows, files = load_fragments(str(run))
    values = build_values(
        rows, files, str(run), allow_partial=False, expected_fragments=EXPECTED_N_FRAGMENTS
    )
    for entry in values["per_k"]:
        for stat in ("median", "q1", "q3", "mean"):
            np.testing.assert_allclose(entry[f"{stat}_us"], entry[f"{stat}_ms"] * 1000.0, rtol=1e-9)
        expected = rows[rows["k"] == entry["k"]]["mean_canon_time_s"].to_numpy(float) * 1e6
        np.testing.assert_allclose(entry["median_us"], float(np.median(expected)), rtol=0, atol=0)

    overall = values["overall"]
    np.testing.assert_allclose(overall["mean_us_at_k9"], overall["mean_ms_at_k9"] * 1000.0)
    tex = render_latex(values)
    k9 = next(e for e in values["per_k"] if e["k"] == 9)
    expected_cell = f"{k9['median_us']:.2f}\\,[{k9['q1_us']:.2f}\\text{{--}}{k9['q3_us']:.2f}]"
    assert f"\\newcommand{{\\synthTimeKNine}}{{{expected_cell}}}" in tex
    assert "\\synthMeanUsKNine}" in tex


# ===================================================================
# Power-law overlay is opt-in (plotting only; the fit is always computed)
# ===================================================================


def _plot_and_collect(rows_records: list[dict[str, float | int]], **kwargs: bool) -> tuple:
    """Render one axes with the figure generator and return (labels, b_fit)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.synthetic_scalability.generate_fig_synthetic_scalability import plot_figure

    fig, ax = plt.subplots()
    b_fit = plot_figure(ax, rows_records, **kwargs)  # type: ignore[arg-type]
    legend = ax.get_legend()
    labels = [text.get_text() for text in legend.get_texts()] if legend is not None else []
    plt.close(fig)
    return labels, b_fit


def test_powerlaw_overlay_is_off_by_default(tmp_path: Path) -> None:
    """The default render carries no O(k^...) legend entry and no dashed fit line."""
    from experiments.synthetic_scalability.generate_fig_synthetic_scalability import (
        _build_caption,
        load_synthetic_data,
    )

    run = _write_full_run(tmp_path / "run", exponent=2.0)
    records = load_synthetic_data(str(run))

    labels, b_fit = _plot_and_collect(records)
    assert not any("O(k^" in label for label in labels), labels

    # The fit is still computed, and still available to any consumer.
    np.testing.assert_allclose(b_fit, 2.0, rtol=1e-6)

    caption = _build_caption(records, b_fit)
    assert "O(k^" not in caption


def test_powerlaw_overlay_restored_when_requested(tmp_path: Path) -> None:
    """--powerlaw-fit puts the fitted curve and its legend entry back."""
    from experiments.synthetic_scalability.generate_fig_synthetic_scalability import (
        _build_caption,
        load_synthetic_data,
    )

    run = _write_full_run(tmp_path / "run", exponent=2.0)
    records = load_synthetic_data(str(run))

    labels, b_fit = _plot_and_collect(records, powerlaw_fit=True)
    assert any("O(k^" in label for label in labels), labels
    # The legend prints the exponent to two decimals, matching the prose and the
    # JSON; one decimal printed 1.4 beside a sentence saying 1.43.
    assert any(re.search(r"O\(k\^\{\d+\.\d\d\}\)\$ fit", label) for label in labels), labels

    caption = _build_caption(records, b_fit, powerlaw_fit=True)
    assert "O(k^" in caption


def test_generate_figure_default_caption_has_no_powerlaw(tmp_path: Path) -> None:
    """The emitted caption file mentions the fit only when the flag is passed."""
    from experiments.synthetic_scalability.generate_fig_synthetic_scalability import generate_figure

    run = _write_full_run(tmp_path / "run")
    out_off = tmp_path / "off"
    generate_figure(str(run), str(out_off))
    caption_off = (out_off / "fig_synthetic_scalability.caption.txt").read_text()
    assert "O(k^" not in caption_off
    assert "power-law fit" not in caption_off

    out_on = tmp_path / "on"
    generate_figure(str(run), str(out_on), powerlaw_fit=True)
    caption_on = (out_on / "fig_synthetic_scalability.caption.txt").read_text()
    assert "O(k^" in caption_on


# ===================================================================
# The rho = k! badge must agree with the table's rho_equals_kfact column
# ===================================================================


def _render_axes(rows_records: list[dict[str, float | int]], **kwargs: bool):  # type: ignore[no-untyped-def]
    """Render one axes with the figure generator and return it (figure left open)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.synthetic_scalability.generate_fig_synthetic_scalability import plot_figure

    fig, ax = plt.subplots()
    plot_figure(ax, rows_records, **kwargs)  # type: ignore[arg-type]
    return fig, ax


def _sampled_run(directory: Path, n_perms: int = 20_000) -> Path:
    """A fixture where each expression samples ``n_perms`` orderings, so rho != k!."""
    directory.mkdir(parents=True, exist_ok=True)
    for k in (3, 4, 5):
        for m in EXPECTED_M_VALUES:
            _write_cell(
                directory,
                k,
                m,
                n_perms_override=n_perms,
                rho_override=float(n_perms),
            )
    return directory


def test_badge_absent_when_rho_does_not_equal_kfactorial(tmp_path: Path) -> None:
    """On sampled data the figure must not assert rho = k!; the table prints $\\times$."""
    import matplotlib.pyplot as plt

    from experiments.synthetic_scalability.generate_fig_synthetic_scalability import (
        _aggregate_per_k,
        load_synthetic_data,
    )

    run = _sampled_run(tmp_path / "sampled")
    records = load_synthetic_data(str(run))

    # Precondition: this is exactly the data on which the table refuses the checkmark.
    assert all(entry["rho_equals_kfact"] is False for entry in _aggregate_per_k(records))

    fig, ax = _render_axes(records)
    texts = [t.get_text() for t in ax.texts]
    plt.close(fig)

    assert not any(r"\rho = k!" in t for t in texts), texts
    # Something true is stated instead: the invariance that actually holds.
    assert any("canonical string" in t for t in texts), texts


def test_badge_present_when_rho_equals_kfactorial(tmp_path: Path) -> None:
    """On exhaustive data the rho = k! badge is kept, matching the table's checkmark."""
    import matplotlib.pyplot as plt

    from experiments.synthetic_scalability.generate_fig_synthetic_scalability import (
        _aggregate_per_k,
        load_synthetic_data,
    )

    run = _write_full_run(tmp_path / "run")
    records = load_synthetic_data(str(run))
    assert all(entry["rho_equals_kfact"] is True for entry in _aggregate_per_k(records))

    fig, ax = _render_axes(records)
    texts = [t.get_text() for t in ax.texts]
    plt.close(fig)

    assert any(r"\rho = k!" in t for t in texts), texts


def test_caption_does_not_claim_kfactorial_on_sampled_data(tmp_path: Path) -> None:
    """The caption's reduction-factor sentence is gated on the same predicate."""
    from experiments.synthetic_scalability.generate_fig_synthetic_scalability import (
        _build_caption,
        load_synthetic_data,
    )

    records = load_synthetic_data(str(_sampled_run(tmp_path / "sampled")))
    caption = _build_caption(records, 1.43)
    assert "equals $k!$ for all" not in caption
    assert "0/" not in caption


def test_x_tick_labels_are_thinned_but_exhaustive_k_is_untouched(tmp_path: Path) -> None:
    """21 k values get <= 9 printed labels; 9 k values keep all of theirs."""
    import matplotlib.pyplot as plt

    from experiments.synthetic_scalability.generate_fig_synthetic_scalability import (
        load_synthetic_data,
    )

    wide = tmp_path / "wide"
    wide.mkdir()
    k_wide = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32, 34, 36]
    for k in k_wide:
        _write_cell(wide, k, 1, n_perms_override=20_000, rho_override=20_000.0)
    fig, ax = _render_axes(load_synthetic_data(str(wide)))
    major = list(ax.get_xticks())
    minor = list(ax.get_xticks(minor=True))
    plt.close(fig)
    assert len(major) <= 9, major
    # Every k still carries a tick; matplotlib drops minors coinciding with majors.
    assert sorted({int(v) for v in major} | {int(v) for v in minor}) == k_wide

    narrow = _write_full_run(tmp_path / "narrow")
    fig2, ax2 = _render_axes(load_synthetic_data(str(narrow)))
    major2 = [int(v) for v in ax2.get_xticks()]
    plt.close(fig2)
    assert major2 == list(EXPECTED_K_VALUES)


def test_total_timeouts_is_surfaced(tmp_path: Path) -> None:
    """timeout_count is summed over every row and exposed in the JSON."""
    run = _write_full_run(tmp_path / "run")
    rows, files = load_fragments(str(run))
    clean = build_values(
        rows, files, str(run), allow_partial=False, expected_fragments=EXPECTED_N_FRAGMENTS
    )
    assert clean["overall"]["total_timeouts"] == 0

    dirty = tmp_path / "dirty"
    dirty.mkdir()
    _write_cell(dirty, 5, 1, n_expr=3, timeouts=2)
    _write_cell(dirty, 6, 1, n_expr=3, timeouts=0)
    rows_d, files_d = load_fragments(str(dirty), allow_partial=True)
    values_d = build_values(
        rows_d, files_d, str(dirty), allow_partial=True, expected_fragments=EXPECTED_N_FRAGMENTS
    )
    assert values_d["overall"]["total_timeouts"] == 6
