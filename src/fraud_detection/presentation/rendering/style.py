"""Reusable visual system for the final Chapter-5 presentation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from .. import (
    METHOD_AMOUNT_GAIN,
    METHOD_BCE,
    METHOD_FIXED,
    METHOD_ORDER,
    METHOD_P_ONLY,
)

DEFAULT_WIDTH_MM = 160.0
PNG_DPI = 300
PANEL_LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")


@dataclass(frozen=True)
class PathStyle:
    """Redundant colour, marker, and line encoding for one score path."""

    colour: str
    marker: str
    linestyle: str
    label: str


PATH_STYLES = {
    METHOD_BCE: PathStyle("#4D4D4D", "o", "-", "BCE"),
    METHOD_P_ONLY: PathStyle("#0072B2", "s", "--", "p-only"),
    METHOD_AMOUNT_GAIN: PathStyle("#E69F00", "D", "-.", "Amount-Gain"),
    METHOD_FIXED: PathStyle("#CC79A7", "^", ":", "feste Referenz"),
}

COMPARISON_METHODS = (
    METHOD_P_ONLY,
    METHOD_AMOUNT_GAIN,
    METHOD_FIXED,
)

NEUTRAL_DARK = "#4D4D4D"
NEUTRAL_MID = "#8C8C8C"
GRID_COLOUR = "#D9D9D9"
CENTRAL_BAND_COLOUR = "#F5F5F5"


def mm_to_inches(value_mm: float) -> float:
    """Convert a physical millimetre measure to inches."""

    return float(value_mm) / 25.4


def configure_presentation_style() -> None:
    """Install deterministic, final-size Matplotlib defaults."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.color": GRID_COLOUR,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.8,
            "lines.linewidth": 0.95,
            "lines.markersize": 4.0,
            "pdf.compression": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "svg.hashsalt": "fraud-detection-ch5-presentation-r7",
            "savefig.dpi": PNG_DPI,
        }
    )


def format_german(
    value: float,
    decimals: int = 2,
    *,
    signed: bool = False,
) -> str:
    """Format a finite numeric value with a German decimal comma."""

    if round(float(value), decimals) == 0:
        value = 0.0
    prefix = "+" if signed else ""
    return f"{value:{prefix}.{decimals}f}".replace(".", ",")


def decimal_comma(decimals: int = 2) -> FuncFormatter:
    """Return a German-decimal Matplotlib tick formatter."""

    return FuncFormatter(
        lambda value, _position: format_german(float(value), decimals)
    )


def method_legend_handles(
    methods: Iterable[str] = METHOD_ORDER,
) -> list[Line2D]:
    """Create compact shared-legend handles using every redundant path cue."""

    handles: list[Line2D] = []
    for method in methods:
        style = PATH_STYLES[method]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.colour,
                marker=style.marker,
                linestyle=style.linestyle,
                linewidth=0.95,
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor=style.colour,
                markeredgewidth=0.8,
                label=style.label,
            )
        )
    return handles


def method_marker_legend_handles(
    methods: Iterable[str] = METHOD_ORDER,
) -> list[Line2D]:
    """Create marker-only handles for figures containing unconnected points."""

    handles: list[Line2D] = []
    for method in methods:
        style = PATH_STYLES[method]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.colour,
                marker=style.marker,
                linestyle="none",
                markersize=4.4,
                markerfacecolor=style.colour,
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=style.label,
            )
        )
    return handles


def style_axis(axis: Axes, *, horizontal_grid: bool = True) -> None:
    """Apply the common axes treatment without introducing vertical grids."""

    axis.set_facecolor("white")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.75)
    axis.spines["bottom"].set_linewidth(0.75)
    axis.grid(False)
    if horizontal_grid:
        axis.yaxis.grid(True, color=GRID_COLOUR, linewidth=0.5, alpha=0.8)
    axis.tick_params(width=0.65, length=3.0)


def panel_label(
    axis: Axes,
    letter: str,
    *,
    x: float = 0.0,
    y: float = 1.025,
) -> None:
    """Place one lowercase bold panel letter in layout-aware coordinates."""

    axis.text(
        x,
        y,
        letter,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color="black",
        clip_on=False,
    )
