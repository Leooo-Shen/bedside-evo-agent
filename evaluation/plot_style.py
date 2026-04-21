"""Shared plotting style for evaluation scripts."""

from __future__ import annotations

from typing import Dict, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SYSTEM_PALETTE: Dict[str, str] = {
    "MedEvo": "#1B4965",
    "Baseline": "#B75A39",
}

METRIC_PALETTE: Dict[str, str] = {
    "hit": "#1B4965",
    "precision": "#7A5C2E",
    "accuracy": "#2F7F78",
}

METRIC_MEAN_PALETTE: Dict[str, str] = {
    "hit": "#0E3148",
    "precision": "#5E4523",
    "accuracy": "#1E5F59",
}

MODE_PALETTE: Sequence[str] = (
    "#1B4965",
    "#2F7F78",
    "#A25B33",
    "#6C5B7B",
    "#5F7F63",
    "#A04E6C",
    "#9C7A2E",
    "#4E6A84",
)

METRIC_BAND_ALPHA = 0.16
PATIENT_TRACE_ALPHA = 0.14
MODE_TRACE_ALPHA = 0.92

_STYLE_CONFIGURED = False


def configure_plot_style() -> None:
    global _STYLE_CONFIGURED
    if _STYLE_CONFIGURED:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "grid.color": "#8A8E99",
            "legend.frameon": False,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    _STYLE_CONFIGURED = True


def system_color(system_name: str) -> str:
    return SYSTEM_PALETTE.get(str(system_name), "#46505A")


def metric_color(metric: str) -> str:
    return METRIC_PALETTE[str(metric)]


def metric_mean_color(metric: str) -> str:
    return METRIC_MEAN_PALETTE[str(metric)]


def mode_color_map(labels: Sequence[str]) -> Dict[str, str]:
    ordered = [str(label) for label in labels]
    palette = list(MODE_PALETTE)
    size = len(palette)
    return {label: palette[index % size] for index, label in enumerate(ordered)}


def heatmap_cmap(metric: str) -> LinearSegmentedColormap:
    metric_name = str(metric)
    if metric_name == "precision":
        return LinearSegmentedColormap.from_list(
            "nm_precision",
            ["#FBF8F2", "#E8D0B0", "#C18A5A", "#7A5C2E"],
        )
    return LinearSegmentedColormap.from_list(
        "nm_primary",
        ["#F7FAFA", "#D2E5E1", "#77A9A2", "#1B4965"],
    )
