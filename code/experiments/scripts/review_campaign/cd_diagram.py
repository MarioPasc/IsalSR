"""Draw one critical-difference diagram per host solver.

The submitted figure ranked all configurations of both hosts together. Pooling
them makes the host difference dominate every rank -- on test R^2 Bingo occupies
ranks 1.5 to 2.7 and UDFS 4.2 to 5.1 -- so the contrast the paper is about, the
one between the three arms of a single host, is compressed into a fraction of
the axis. Pooling also widens the Nemenyi threshold: six groups over 70 problems
give a critical difference of 0.90 against 0.40 for three groups over the same
problems, which is the difference between a resolvable comparison and an
unresolvable one.

Two diagrams are drawn instead, one per host, in the style of the submitted
figure and with the same three colours, so that they sit side by side in a
single figure environment under one legend.

Usage
-----
    python -m experiments.scripts.review_campaign.cd_diagram [--analyses DIR]
        [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experiments.figures.models.generate_critical_difference import (  # noqa: E402
    _tikz_to_pdf,
    generate_cd_2d,
)
from experiments.plotting_styles import (  # noqa: E402
    COLOR_HASH,
    COLOR_ISALSR,
    COLOR_NATIVE,
)
from experiments.scripts.review_campaign.config import ARMS, METHODS, add_common_args  # noqa: E402

#: Within a single-host diagram the host is in the caption, so the group labels
#: name only the arm.
ARM_ONLY_LABELS = {
    "native DAG": "Native representation",
    "Naive-Hash": "Naive hash",
    "IsalSR": "IsalSR",
}


def labels_for(method: str) -> dict[str, str]:
    """Map this host's group labels onto arm-only display names."""
    return {f"{method.upper()} {raw}": shown for raw, shown in ARM_ONLY_LABELS.items()}


#: Colour per arm, in the order the shared legend lists them.
LEGEND_ENTRIES = (
    ("clrNative", COLOR_NATIVE, "Native representation"),
    ("clrHash", COLOR_HASH, "Naive hash"),
    ("clrIsalsr", COLOR_ISALSR, "\\textsc{IsalSR}"),
)

#: Horizontal step between legend items, in centimetres.
LEGEND_STEP = 4.6


def strip_legend(tex_path: Path) -> None:
    """Remove the per-panel legend and recompile.

    Both panels carry the same three entries, so printing them twice spends
    roughly a quarter of each panel's width on a duplicate. The legend is drawn
    once, separately, and placed above the pair.
    """
    lines = [
        line
        for line in tex_path.read_text(encoding="utf-8").splitlines()
        if "\\addlegendentry" not in line
        and "legend style" not in line
        and "legend cell align" not in line
    ]
    source = "\n".join(lines) + "\n"
    tex_path.write_text(source, encoding="utf-8")
    _tikz_to_pdf(source, tex_path.with_suffix(""))


def write_legend(out_stem: Path) -> None:
    """Draw the three arm swatches as one horizontal row."""
    colours = "\n".join(
        f"\\definecolor{{{name}}}{{HTML}}{{{value.lstrip('#')}}}"
        for name, value, _label in LEGEND_ENTRIES
    )
    items = "\n".join(
        f"  \\node[circle, draw={name}, very thick, inner sep=2.1pt] "
        f"at ({i * LEGEND_STEP}, 0) {{}};\n"
        f"  \\node[anchor=west, font=\\small] at ({i * LEGEND_STEP + 0.28}, 0) "
        f"{{{label}}};"
        for i, (name, _value, label) in enumerate(LEGEND_ENTRIES)
    )
    source = (
        "\\documentclass[tikz,margin=.05in]{standalone}\n"
        "\\usepackage{lmodern}\n"
        f"{colours}\n"
        "\\begin{document}\n\\begin{tikzpicture}\n"
        f"{items}\n"
        "\\end{tikzpicture}\n\\end{document}\n"
    )
    out_stem.with_suffix(".tex").write_text(source, encoding="utf-8")
    _tikz_to_pdf(source, out_stem)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Flattened single-benchmark root, as run_all.sh builds it.",
    )
    args = parser.parse_args()

    out_dir = args.analyses / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for method in METHODS:
        generate_cd_2d(
            args.results_dir,
            out_dir,
            [method],
            ["benchmark"],
            variants=list(ARMS),
            out_stem=f"cd_2d_{method}",
            treatment_labels=labels_for(method),
            # Circles in both panels: the host is named in the sub-caption, so
            # shape carries nothing and the two legends stay identical.
            mark_override="mark=o, mark size=4pt",
        )
        strip_legend(out_dir / f"cd_2d_{method}.tex")

    write_legend(out_dir / "cd_legend")


if __name__ == "__main__":
    main()
