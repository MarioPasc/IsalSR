"""Column/grid geometry helpers for the D2S trace figure.

Keeps all magic numbers for the 2×N figure in one place so the generator
and the composite module stay free of layout constants.

Dependency rule: zero imports beyond the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TraceLayout:
    """Geometry specification for one D2S trace figure.

    The figure uses a gridspec with ``(2 * n_sub_rows + 1)`` rows
    (separator row between the two trace rows) and ``n_cols`` columns.

    When ``show_cdll`` is ``True`` (the default) a CDLL-ring sub-row is
    inserted between the DAG and strip sub-rows for each trace row,
    giving three sub-rows per trace row and seven gridspec rows total.
    When ``False`` the figure reverts to the two-sub-row layout (five
    gridspec rows) and pointer markers are drawn directly on the DAG
    nodes instead.

    Attributes:
        n_cols: Number of columns (one per D2S step displayed).
        dag_ratio: Gridspec height weight for DAG sub-rows.
        cdll_ratio: Gridspec height weight for CDLL sub-rows.
        strip_ratio: Gridspec height weight for strip sub-rows.
        sep_ratio: Gridspec height weight for the inter-row separator.
        col_width: Width of each column in source inches.
        dag_height: Height of DAG sub-row in source inches.
        cdll_height: Height of CDLL sub-row in source inches.
        strip_height: Height of strip sub-row in source inches.
        wspace: Gridspec horizontal spacing.
        hspace: Gridspec vertical spacing within each row group.
        show_cdll: Whether to render the CDLL ring sub-row.
    """

    n_cols: int = 4
    dag_ratio: float = 3.5
    cdll_ratio: float = 0.9
    strip_ratio: float = 1.1
    sep_ratio: float = 0.35
    col_width: float = 3.4
    dag_height: float = 2.8
    cdll_height: float = 0.72
    strip_height: float = 0.88
    wspace: float = 0.04
    hspace: float = 0.06
    show_cdll: bool = True

    # Font sizes (source; divide by 0.478 to recover print pt at \linewidth).
    fs_node: float = field(default=22.0)
    fs_title: float = field(default=16.0)
    fs_strip: float = field(default=16.0)
    fs_annot: float = field(default=13.5)
    node_r: float = field(default=0.40)

    # CDLL panel font sizes and node radius (source units).
    fs_cdll_label: float = field(default=16.0)
    fs_cdll_ptr: float = field(default=11.0)
    cdll_node_r: float = field(default=0.28)

    @property
    def n_sub_rows(self) -> int:
        """Number of sub-rows per trace row (3 with CDLL, 2 without)."""
        return 3 if self.show_cdll else 2

    @property
    def height_ratios(self) -> list[float]:
        """Gridspec height ratios for all logical rows."""
        if self.show_cdll:
            return [
                self.dag_ratio,
                self.cdll_ratio,
                self.strip_ratio,
                self.sep_ratio,
                self.dag_ratio,
                self.cdll_ratio,
                self.strip_ratio,
            ]
        return [
            self.dag_ratio,
            self.strip_ratio,
            self.sep_ratio,
            self.dag_ratio,
            self.strip_ratio,
        ]

    @property
    def figsize(self) -> tuple[float, float]:
        """Matplotlib figure size in inches."""
        pair_h = self.dag_height + self.strip_height
        if self.show_cdll:
            pair_h += self.cdll_height
        total_h = 2 * pair_h + self._sep_inches
        return (self.n_cols * self.col_width, total_h)

    @property
    def _sep_inches(self) -> float:
        """Separator height derived from gridspec proportionality."""
        pair_ratio = self.dag_ratio + self.strip_ratio
        if self.show_cdll:
            pair_ratio += self.cdll_ratio
        pair_h = self.dag_height + self.strip_height
        if self.show_cdll:
            pair_h += self.cdll_height
        # sep_h / pair_h = sep_ratio / pair_ratio  (gridspec proportionality)
        return pair_h * self.sep_ratio / pair_ratio


# Default layout used by generate_fig_reachability.py.
DEFAULT_TRACE_LAYOUT = TraceLayout()
