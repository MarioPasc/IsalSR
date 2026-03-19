"""Publication-ready plot settings for IsalSR benchmark figures.

IEEE-compliant settings with Paul Tol colorblind-friendly palettes
for symbolic regression expression DAG analysis.

References:
    - Paul Tol's color schemes: https://personal.sron.nl/~pault/
    - IEEE publication guidelines
    - scienceplots: https://github.com/garrettj403/SciencePlots
"""

from __future__ import annotations

# =============================================================================
# Paul Tol Color Palettes (SRON - colorblind safe)
# =============================================================================

PAUL_TOL_BRIGHT = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "cyan": "#66CCEE",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

PAUL_TOL_HIGH_CONTRAST = {
    "blue": "#004488",
    "yellow": "#DDAA33",
    "red": "#BB5566",
}

PAUL_TOL_MUTED = [
    "#CC6677",  # rose
    "#332288",  # indigo
    "#DDCC77",  # sand
    "#117733",  # green
    "#88CCEE",  # cyan
    "#882255",  # wine
    "#44AA99",  # teal
    "#999933",  # olive
    "#AA4499",  # purple
]

# =============================================================================
# IEEE Column Width Specifications
# =============================================================================

IEEE_COLUMN_WIDTH_INCHES = 3.39  # Single column (86 mm)
IEEE_COLUMN_GAP_INCHES = 0.24  # Gap between columns (6 mm)
IEEE_TEXT_WIDTH_INCHES = 7.0  # Full print area width (178 mm)
IEEE_TEXT_HEIGHT_INCHES = 9.0  # Full print area height (229 mm)

# =============================================================================
# Main Plot Settings Dictionary
# =============================================================================

PLOT_SETTINGS = {
    # Figure dimensions (IEEE compliant)
    "figure_width_single": IEEE_COLUMN_WIDTH_INCHES,  # 3.39 inches
    "figure_width_double": IEEE_TEXT_WIDTH_INCHES,  # 7.0 inches
    "figure_height_max": IEEE_TEXT_HEIGHT_INCHES,  # 9.0 inches (max)
    "figure_height_ratio": 0.75,  # Height = width * ratio (for plots)
    # Fonts (IEEE requires Times or similar serif)
    "font_family": "serif",
    "font_serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext_fontset": "stix",  # STIX for math (matches Times)
    "text_usetex": False,  # Set True if LaTeX is installed
    # Font sizes (IEEE guidelines)
    "font_size": 10,
    "axes_labelsize": 11,
    "axes_titlesize": 12,
    "tick_labelsize": 9,
    "legend_fontsize": 9,
    "annotation_fontsize": 8,
    "panel_label_fontsize": 11,
    # Line properties
    "line_width": 1.2,
    "line_width_thick": 1.8,
    "marker_size": 5,
    "marker_edge_width": 0.5,
    # Error bars
    "errorbar_capsize": 2,
    "errorbar_capthick": 0.8,
    "errorbar_linewidth": 0.8,
    # Error bands (for confidence intervals)
    "error_band_alpha": 0.2,
    # Boxplot properties
    "boxplot_linewidth": 0.8,
    "boxplot_flier_size": 3,
    "boxplot_width": 0.6,
    # Bar plot properties
    "bar_width": 0.18,
    "bar_alpha": 0.85,
    # Grid
    "grid_alpha": 0.4,
    "grid_linestyle": ":",
    "grid_linewidth": 0.5,
    # Spines
    "spine_linewidth": 0.8,
    "spine_color": "0.2",
    # Ticks
    "tick_direction": "in",
    "tick_major_width": 0.8,
    "tick_minor_width": 0.5,
    "tick_major_length": 3.5,
    "tick_minor_length": 2.0,
    # Legend
    "legend_frameon": False,
    "legend_framealpha": 0.9,
    "legend_edgecolor": "0.8",
    "legend_borderpad": 0.4,
    "legend_columnspacing": 1.0,
    "legend_handletextpad": 0.5,
    # UMAP/t-SNE scatter
    "scatter_alpha": 0.6,
    "scatter_size": 15,
    "scatter_edgewidth": 0.3,
    # DPI for output
    "dpi_print": 300,
    "dpi_screen": 150,
    # Significance annotations
    "significance_bracket_linewidth": 0.8,
    "significance_text_fontsize": 9,
    "effect_size_fontsize": 8,
}

# =============================================================================
# Condition Colors (for consistent styling across figures)
# =============================================================================

CONDITION_COLORS: dict[str, str] = {
    "real_data": PAUL_TOL_BRIGHT["blue"],
    "synthetic_data": PAUL_TOL_BRIGHT["red"],
    "control": PAUL_TOL_BRIGHT["green"],
    "epilepsy": PAUL_TOL_BRIGHT["purple"],
    "vMF": PAUL_TOL_BRIGHT["cyan"],
    "baseline": PAUL_TOL_BRIGHT["yellow"],
    "threshold": PAUL_TOL_BRIGHT["grey"],
}


def apply_ieee_style() -> None:
    """Apply IEEE publication style using scienceplots if available.

    Falls back to manual style settings if scienceplots is not installed.
    Overrides default color cycle with Paul Tol colorblind-safe palette.
    """
    import matplotlib.pyplot as plt

    # Try to use scienceplots if available
    try:
        plt.style.use(["science", "ieee"])
        _scienceplots_available = True
    except OSError:
        _scienceplots_available = False
        _apply_fallback_ieee_style()

    # Override with condition colors and custom settings
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(color=list(CONDITION_COLORS.values())),
            # Ensure math rendering
            "mathtext.fontset": PLOT_SETTINGS["mathtext_fontset"],
            "font.family": PLOT_SETTINGS["font_family"],
            # Grid settings
            "axes.grid": True,
            "grid.alpha": PLOT_SETTINGS["grid_alpha"],
            "grid.linestyle": PLOT_SETTINGS["grid_linestyle"],
            "grid.linewidth": PLOT_SETTINGS["grid_linewidth"],
            # Tick settings
            "xtick.direction": PLOT_SETTINGS["tick_direction"],
            "ytick.direction": PLOT_SETTINGS["tick_direction"],
            "xtick.major.width": PLOT_SETTINGS["tick_major_width"],
            "ytick.major.width": PLOT_SETTINGS["tick_major_width"],
        }
    )


def _apply_fallback_ieee_style() -> None:
    """Apply IEEE-like style without scienceplots."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            # Fonts
            "font.family": PLOT_SETTINGS["font_family"],
            "font.serif": PLOT_SETTINGS["font_serif"],
            "font.size": PLOT_SETTINGS["font_size"],
            "mathtext.fontset": PLOT_SETTINGS["mathtext_fontset"],
            # Axes
            "axes.labelsize": PLOT_SETTINGS["axes_labelsize"],
            "axes.titlesize": PLOT_SETTINGS["axes_titlesize"],
            "axes.linewidth": PLOT_SETTINGS["spine_linewidth"],
            "axes.grid": True,
            # Ticks
            "xtick.labelsize": PLOT_SETTINGS["tick_labelsize"],
            "ytick.labelsize": PLOT_SETTINGS["tick_labelsize"],
            "xtick.major.width": PLOT_SETTINGS["tick_major_width"],
            "ytick.major.width": PLOT_SETTINGS["tick_major_width"],
            "xtick.minor.width": PLOT_SETTINGS["tick_minor_width"],
            "ytick.minor.width": PLOT_SETTINGS["tick_minor_width"],
            "xtick.direction": PLOT_SETTINGS["tick_direction"],
            "ytick.direction": PLOT_SETTINGS["tick_direction"],
            "xtick.major.size": PLOT_SETTINGS["tick_major_length"],
            "ytick.major.size": PLOT_SETTINGS["tick_major_length"],
            "xtick.minor.size": PLOT_SETTINGS["tick_minor_length"],
            "ytick.minor.size": PLOT_SETTINGS["tick_minor_length"],
            # Legend
            "legend.fontsize": PLOT_SETTINGS["legend_fontsize"],
            "legend.frameon": PLOT_SETTINGS["legend_frameon"],
            "legend.framealpha": PLOT_SETTINGS["legend_framealpha"],
            "legend.edgecolor": PLOT_SETTINGS["legend_edgecolor"],
            # Grid
            "grid.alpha": PLOT_SETTINGS["grid_alpha"],
            "grid.linestyle": PLOT_SETTINGS["grid_linestyle"],
            "grid.linewidth": PLOT_SETTINGS["grid_linewidth"],
            # Figure
            "figure.figsize": (
                PLOT_SETTINGS["figure_width_double"],
                PLOT_SETTINGS["figure_width_double"] * PLOT_SETTINGS["figure_height_ratio"],
            ),
            "figure.dpi": PLOT_SETTINGS["dpi_screen"],
            "savefig.dpi": PLOT_SETTINGS["dpi_print"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def get_significance_stars(p_val: float) -> str:
    """Convert p-value to significance stars.

    Args:
        p_val: P-value from statistical test.

    Returns:
        String with stars: "***" (p<0.001), "**" (p<0.01),
        "*" (p<0.05), or "n.s." (not significant).
    """
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return "n.s."


def get_effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value.

    Returns:
        Interpretation string.
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def get_figure_size(
    width: str = "single",
    height_ratio: float = None,
) -> tuple[float, float]:
    """Get figure size tuple for IEEE format.

    Args:
        width: "single" for column width, "double" for full width.
        height_ratio: Custom height/width ratio. If None, uses default.

    Returns:
        Tuple of (width, height) in inches.
    """
    if width == "single":
        w = PLOT_SETTINGS["figure_width_single"]
    elif width == "double":
        w = PLOT_SETTINGS["figure_width_double"]
    else:
        raise ValueError(f"Unknown width: {width}")

    ratio = height_ratio or PLOT_SETTINGS["figure_height_ratio"]
    return (w, w * ratio)


# =============================================================================
# IsalSR Instruction Colors (per-token coloring for two-tier encoding)
# =============================================================================

# Single-char instruction colors (movement, edge, no-op).
INSTRUCTION_COLORS: dict[str, str] = {
    "N": "#4477AA",  # blue   (primary movement next)
    "P": "#004488",  # dark blue (primary movement prev)
    "n": "#66CCEE",  # cyan   (secondary movement next)
    "p": "#88CCEE",  # light cyan (secondary movement prev)
    "C": "#EE6677",  # red    (primary->secondary edge)
    "c": "#CC6677",  # rose   (secondary->primary edge)
    "W": "#BBBBBB",  # grey   (no-op)
}

# Two-char compound token colors (V/v + label).
# Operations are colored by category for readability.
TOKEN_COLORS: dict[str, str] = {
    # Variadic operations (green shades)
    "V+": "#228833",  # green  (ADD from primary)
    "v+": "#117733",  # dark green (ADD from secondary)
    "V*": "#44AA66",  # light green (MUL from primary)
    "v*": "#338855",  # muted green (MUL from secondary)
    # Binary operations (orange/amber shades)
    "V-": "#EE7733",  # orange (SUB from primary)
    "v-": "#CC6622",  # dark orange (SUB from secondary)
    "V/": "#DDAA33",  # amber (DIV from primary)
    "v/": "#BB8822",  # dark amber (DIV from secondary)
    "V^": "#AA7744",  # brown (POW from primary)
    "v^": "#886633",  # dark brown (POW from secondary)
    # Unary operations (purple/magenta shades)
    "Vs": "#AA3377",  # magenta (SIN)
    "vs": "#882266",  # dark magenta
    "Vc": "#CC6699",  # pink (COS)
    "vc": "#AA5588",  # dark pink
    "Ve": "#6644AA",  # purple (EXP)
    "ve": "#553399",  # dark purple
    "Vl": "#8866BB",  # lavender (LOG)
    "vl": "#7755AA",  # dark lavender
    "Vr": "#9977CC",  # light purple (SQRT)
    "vr": "#8866BB",  # muted purple
    "Va": "#BB88DD",  # light magenta (ABS)
    "va": "#AA77CC",  # muted magenta
    # Constants (yellow/gold)
    "Vk": "#CCBB44",  # gold (CONST from primary)
    "vk": "#AAAA33",  # dark gold (CONST from secondary)
    # Single-char instructions (merged for unified lookup)
    **INSTRUCTION_COLORS,
}

# =============================================================================
# Graph Family Colors (consistent across all benchmark figures)
# =============================================================================

FAMILY_COLORS: dict[str, str] = {
    "tree": "#4477AA",
    "path": "#66CCEE",
    "star": "#228833",
    "cycle": "#CCBB44",
    "complete": "#EE6677",
    "barabasi_albert": "#AA3377",
    "ba_m1": "#AA3377",
    "ba_m2": "#882255",
    "ba_m3": "#332288",
    "gnp": "#DDCC77",
    "gnp_directed": "#999933",
    "watts_strogatz": "#44AA99",
    "grid": "#999933",
    "ladder": "#88CCEE",
    "petersen": "#AA4499",
    "wheel": "#117733",
    "random_string": "#332288",
    "gnp_03": "#DDCC77",
    "gnp_05": "#CC6600",
    "binary_tree": "#4477AA",
}

# =============================================================================
# Display Names (human-readable labels for plots)
# =============================================================================

FAMILY_DISPLAY_NAMES: dict[str, str] = {
    "tree": "Tree",
    "path": "Path",
    "star": "Star",
    "cycle": "Cycle",
    "complete": "Complete",
    "barabasi_albert": "Barabási–Albert",
    "ba_m1": "BA (m=1)",
    "ba_m2": "BA (m=2)",
    "ba_m3": "BA (m=3)",
    "gnp": "GNP",
    "gnp_directed": "GNP (dir.)",
    "watts_strogatz": "Watts–Strogatz",
    "grid": "Grid",
    "ladder": "Ladder",
    "petersen": "Petersen",
    "wheel": "Wheel",
    "random_string": "Random string",
    "gnp_03": "GNP (p=0.3)",
    "gnp_05": "GNP (p=0.5)",
    "binary_tree": "Binary tree",
}

EXPERIMENT_DISPLAY_NAMES: dict[str, str] = {
    "edge_edit": "Edge edit",
    "family_pair": "Family pair",
    "random_pair": "Random pair",
}

# Marker shapes for families (useful when colors alone are insufficient)
FAMILY_MARKERS: dict[str, str] = {
    "tree": "o",
    "path": "s",
    "star": "^",
    "cycle": "D",
    "complete": "v",
    "ba_m1": "P",
    "ba_m2": "X",
    "ba_m3": "h",
    "gnp": "d",
    "gnp_directed": "<",
    "watts_strogatz": ">",
    "grid": "p",
    "ladder": "H",
    "petersen": "*",
    "wheel": "8",
    "random_string": "H",
    "gnp_03": "d",
    "gnp_05": "<",
    "binary_tree": "^",
}


def family_display(name: str) -> str:
    """Return human-readable display name for a graph family."""
    if name in FAMILY_DISPLAY_NAMES:
        return FAMILY_DISPLAY_NAMES[name]
    # Handle parameterized families: gnp_p0.3 -> "GNP p=0.3", ws_k4 -> "WS k=4"
    if name.startswith("gnp_p"):
        return f"GNP p={name[5:]}"
    if name.startswith("ws_k"):
        return f"WS k={name[4:]}"
    if name.startswith("ba_m"):
        return f"BA m={name[4:]}"
    return name.replace("_", " ").title()


# =============================================================================
# IsalSR-specific helper functions
# =============================================================================


def tokenize_for_display(string: str) -> list[str]:
    """Tokenize an IsalSR instruction string for display purposes.

    Handles the two-tier encoding: V/v consume the next character as a label.
    Returns a list of tokens (single-char or two-char).

    Args:
        string: IsalSR instruction string (e.g. "V+NnncVs").

    Returns:
        List of tokens: ["V+", "N", "n", "n", "c", "Vs"]
    """
    tokens: list[str] = []
    i = 0
    while i < len(string):
        if string[i] in "Vv" and i + 1 < len(string):
            tokens.append(string[i : i + 2])
            i += 2
        else:
            tokens.append(string[i])
            i += 1
    return tokens


def render_colored_string(
    ax: object,
    string: str,
    x: float,
    y: float,
    fontsize: int = 8,
    mono: bool = True,
) -> None:
    """Render an IsalSR instruction string with per-token coloring.

    Each token is drawn in the color defined by TOKEN_COLORS. Two-char
    tokens (V+, Vs, etc.) are treated as single colored units. Tokens
    not in the color map are drawn in black.

    Args:
        ax: Matplotlib Axes object.
        string: IsalSR instruction string (e.g. "V+NnncVs").
        x: Starting x coordinate (in data coordinates).
        y: y coordinate (in data coordinates).
        fontsize: Font size for the characters.
        mono: If True, use monospace font.
    """

    font_props: dict[str, object] = {"fontsize": fontsize}
    if mono:
        font_props["fontfamily"] = "monospace"

    tokens = tokenize_for_display(string)
    renderer = ax.figure.canvas.get_renderer()  # type: ignore[union-attr]
    offset_x = x
    for token in tokens:
        color = TOKEN_COLORS.get(token, "#000000")
        txt = ax.text(offset_x, y, token, color=color, transform=ax.transData, **font_props)  # type: ignore[union-attr]
        # Advance x by the width of the rendered token
        txt.draw(renderer)
        bbox = txt.get_window_extent(renderer=renderer)
        inv = ax.transData.inverted()  # type: ignore[union-attr]
        data_bbox = inv.transform(bbox)
        offset_x += data_bbox[1][0] - data_bbox[0][0]


def binomial_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson exact binomial confidence interval.

    Args:
        k: Number of successes.
        n: Number of trials.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    from scipy import stats as sp_stats

    if n == 0:
        return (0.0, 1.0)
    lower = sp_stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    upper = sp_stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (float(lower), float(upper))


def bootstrap_ci(
    x: list[float],
    y: list[float],
    stat_func: str = "pearson",
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a correlation statistic.

    Args:
        x: First variable values.
        y: Second variable values.
        stat_func: "pearson" or "spearman".
        n_boot: Number of bootstrap resamples.
        alpha: Significance level.
        seed: Random seed.

    Returns:
        (point_estimate, ci_lower, ci_upper).
    """
    import numpy as np
    from scipy import stats as sp_stats

    rng = np.random.default_rng(seed)
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    n = len(x_arr)

    if stat_func == "pearson":
        point = float(sp_stats.pearsonr(x_arr, y_arr)[0])
    else:
        point = float(sp_stats.spearmanr(x_arr, y_arr)[0])

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x_arr[idx], y_arr[idx]
        if stat_func == "pearson":
            boot_stats[i] = sp_stats.pearsonr(xb, yb)[0]
        else:
            boot_stats[i] = sp_stats.spearmanr(xb, yb)[0]

    ci_lower = float(np.nanpercentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.nanpercentile(boot_stats, 100 * (1 - alpha / 2)))
    return (point, ci_lower, ci_upper)


def save_figure(
    fig: object,
    path: str,
    formats: tuple[str, ...] = ("pdf", "png", "svg"),
) -> list[str]:
    """Save figure in multiple formats at publication DPI.

    Args:
        fig: Matplotlib Figure object.
        path: Base path without extension (e.g. "/tmp/fig1").
        formats: Tuple of format extensions.

    Returns:
        List of saved file paths.
    """
    import os

    saved: list[str] = []
    for fmt in formats:
        fpath = f"{path}.{fmt}"
        os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)
        fig.savefig(  # type: ignore[union-attr]
            fpath,
            format=fmt,
            dpi=PLOT_SETTINGS["dpi_print"],
            bbox_inches="tight",
            pad_inches=0.02,
        )
        saved.append(fpath)
    return saved


def save_latex_table(
    df: object,
    path: str,
    caption: str,
    label: str,
) -> None:
    """Save a pandas DataFrame as a LaTeX table.

    Args:
        df: pandas DataFrame.
        path: Output .tex file path.
        caption: LaTeX table caption.
        label: LaTeX table label.
    """
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    latex = df.to_latex(  # type: ignore[union-attr]
        index=False,
        escape=True,
        caption=caption,
        label=label,
        position="htbp",
    )
    with open(path, "w") as f:
        f.write(latex)
