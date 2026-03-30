"""
Plot per-patient selected-window ratio distributions across action-count thresholds.
"""

from __future__ import annotations

import argparse
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _find_latest_per_patient_csv(output_dir: Path) -> Path:
    pattern = str(output_dir / "action_window_threshold_selection_rates_per_patient_*.csv")
    matches = sorted(glob.glob(pattern), reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No per-patient threshold CSV found in {output_dir}. "
            "Run the threshold sweep first."
        )
    return Path(matches[0])


def _extract_threshold_columns(df: pd.DataFrame) -> List[Tuple[int, str]]:
    columns: List[Tuple[int, str]] = []
    for col in df.columns:
        if not col.startswith("pct_gt_"):
            continue
        raw = col.replace("pct_gt_", "")
        try:
            threshold = int(raw)
        except ValueError:
            continue
        columns.append((threshold, col))
    columns.sort(key=lambda item: item[0])
    return columns


def _plot_distribution(df: pd.DataFrame, threshold_columns: List[Tuple[int, str]], output_path: Path) -> None:
    thresholds = [t for t, _ in threshold_columns]
    series = [df[col].astype(float).to_numpy() for _, col in threshold_columns]
    labels = [f">{t}" for t in thresholds]

    fig, ax = plt.subplots(figsize=(14, 7))

    parts = ax.violinplot(series, showmeans=False, showextrema=False, widths=0.9)
    for body in parts["bodies"]:
        body.set_facecolor("#9ecae1")
        body.set_edgecolor("#3182bd")
        body.set_alpha(0.55)

    ax.boxplot(
        series,
        widths=0.22,
        patch_artist=True,
        boxprops={"facecolor": "#fdd0a2", "edgecolor": "#e6550d", "alpha": 0.8},
        medianprops={"color": "#a63603", "linewidth": 2},
        whiskerprops={"color": "#e6550d"},
        capprops={"color": "#e6550d"},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.25},
    )

    # Overlay per-threshold means for easier trend reading.
    means = [float(np.mean(values)) for values in series]
    ax.plot(range(1, len(thresholds) + 1), means, color="#08519c", marker="o", linewidth=2, label="Mean")

    ax.set_xticks(range(1, len(thresholds) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Action-event threshold")
    ax.set_ylabel("Selected windows per patient (%)")
    ax.set_title("Distribution of selected-window ratio across patients")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot distribution of per-patient selected-window ratios under different thresholds."
    )
    parser.add_argument(
        "--per-patient-csv",
        type=Path,
        default=None,
        help="Path to per-patient threshold summary CSV. Defaults to latest file in analysis_outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory to save figure.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_csv = args.per_patient_csv or _find_latest_per_patient_csv(args.output_dir)
    df = pd.read_csv(input_csv)
    threshold_columns = _extract_threshold_columns(df)
    if not threshold_columns:
        raise ValueError("No threshold columns found (expected columns like pct_gt_5).")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"action_window_selection_ratio_distribution_{stamp}.png"
    _plot_distribution(df, threshold_columns, output_path)

    print(f"Input CSV: {input_csv}")
    print(f"Saved plot: {output_path}")
    print(f"Thresholds: {[t for t, _ in threshold_columns]}")
    print(f"Patients: {len(df)}")


if __name__ == "__main__":
    main()
