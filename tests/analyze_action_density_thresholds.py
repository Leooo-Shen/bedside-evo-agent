"""
Analyze window selection rates using action-density thresholds.

Action density is defined per 30-minute window as:
    action_event_count / total_event_count
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import get_config
from data_parser import MIMICDataParser
from experiments.oracle.action_validity_common import ACTIONABLE_EVENT_CODES


def _threshold_to_col_name(threshold: float) -> str:
    text = f"{threshold:g}".replace(".", "p").replace("-", "m")
    return f"pct_gt_{text}"


def _window_action_density(
    trajectory: dict,
    action_codes: Iterable[str],
    window_minutes: int,
) -> np.ndarray:
    enter_time = pd.to_datetime(trajectory.get("enter_time"), errors="coerce")
    leave_time = pd.to_datetime(trajectory.get("leave_time"), errors="coerce")
    if pd.isna(enter_time) or pd.isna(leave_time) or leave_time <= enter_time:
        return np.array([], dtype=float)

    window_delta = pd.Timedelta(minutes=window_minutes)
    total_seconds = (leave_time - enter_time).total_seconds()
    num_windows = int(np.ceil(total_seconds / window_delta.total_seconds()))
    if num_windows <= 0:
        return np.array([], dtype=float)

    action_counts = np.zeros(num_windows, dtype=int)
    total_counts = np.zeros(num_windows, dtype=int)
    action_code_set = {str(code).strip().upper() for code in action_codes}

    for event in trajectory.get("events", []):
        event_time = pd.to_datetime(event.get("time"), errors="coerce")
        if pd.isna(event_time):
            continue
        if event_time < enter_time or event_time > leave_time:
            continue

        window_index = int((event_time - enter_time) // window_delta)
        if window_index >= num_windows:
            window_index = num_windows - 1
        if window_index < 0:
            continue

        total_counts[window_index] += 1
        code = str(event.get("code") or "").strip().upper()
        if code in action_code_set:
            action_counts[window_index] += 1

    return np.divide(action_counts, total_counts, out=np.zeros(num_windows, dtype=float), where=total_counts > 0)


def _plot_distribution(
    per_patient_df: pd.DataFrame,
    thresholds: Sequence[float],
    output_path: Path,
) -> None:
    threshold_cols = [_threshold_to_col_name(th) for th in thresholds]
    series = [per_patient_df[col].astype(float).to_numpy() for col in threshold_cols]
    labels = [f">{th:g}" for th in thresholds]

    fig, ax = plt.subplots(figsize=(12, 7))

    parts = ax.violinplot(series, showmeans=False, showextrema=False, widths=0.9)
    for body in parts["bodies"]:
        body.set_facecolor("#bcbddc")
        body.set_edgecolor("#756bb1")
        body.set_alpha(0.60)

    ax.boxplot(
        series,
        widths=0.22,
        patch_artist=True,
        boxprops={"facecolor": "#c7e9c0", "edgecolor": "#238b45", "alpha": 0.85},
        medianprops={"color": "#005a32", "linewidth": 2},
        whiskerprops={"color": "#238b45"},
        capprops={"color": "#238b45"},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.25},
    )

    means = [float(np.mean(values)) for values in series]
    ax.plot(range(1, len(thresholds) + 1), means, color="#08519c", marker="o", linewidth=2, label="Mean")

    ax.set_xticks(range(1, len(thresholds) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Action-density threshold")
    ax.set_ylabel("Selected windows per patient (%)")
    ax.set_title("Distribution of selected-window ratios across patients")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _parse_thresholds(raw: str) -> List[float]:
    thresholds: List[float] = []
    for token in raw.split(","):
        text = token.strip()
        if not text:
            continue
        value = float(text)
        if value < 0:
            raise ValueError(f"Threshold must be >= 0. Got: {value}")
        thresholds.append(value)
    if not thresholds:
        raise ValueError("No thresholds provided.")
    thresholds = sorted(set(thresholds))
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze action-density threshold window selection rates.")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.3,0.4,0.5,0.6",
        help="Comma-separated density thresholds. Example: 0.3,0.4,0.5,0.6",
    )
    parser.add_argument("--window-minutes", type=int, default=30, help="Window size in minutes.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs"), help="Output directory.")
    args = parser.parse_args()

    if args.window_minutes < 1:
        raise ValueError("--window-minutes must be >= 1")

    thresholds = _parse_thresholds(args.thresholds)

    config = get_config()
    data_parser = MIMICDataParser(events_path=config.events_path, icu_stay_path=config.icu_stay_path)
    data_parser.load_data()

    action_codes = {code.strip().upper() for code in ACTIONABLE_EVENT_CODES}
    stays = data_parser.icu_stay_df.reset_index(drop=True)

    per_patient_rows = []
    all_window_density: List[float] = []

    for _, stay in stays.iterrows():
        subject_id = int(stay["subject_id"])
        icu_stay_id = int(stay["icu_stay_id"])
        trajectory = data_parser.get_patient_trajectory(subject_id, icu_stay_id, icu_stay=stay)

        density = _window_action_density(
            trajectory=trajectory,
            action_codes=action_codes,
            window_minutes=args.window_minutes,
        )
        if density.size == 0:
            continue

        all_window_density.extend(density.tolist())
        row = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "num_windows": int(density.size),
        }
        for threshold in thresholds:
            col_name = _threshold_to_col_name(threshold)
            row[col_name] = float((density > threshold).mean() * 100.0)
        per_patient_rows.append(row)

    if not per_patient_rows:
        raise ValueError("No valid trajectories/windows found.")

    per_patient_df = pd.DataFrame(per_patient_rows)
    all_density_arr = np.array(all_window_density, dtype=float)

    summary_rows = []
    for threshold in thresholds:
        col_name = _threshold_to_col_name(threshold)
        summary_rows.append(
            {
                "threshold": float(threshold),
                "threshold_rule": f"density > {threshold:g}",
                "selected_windows_global_pct": float((all_density_arr > threshold).mean() * 100.0),
                "selected_windows_mean_patient_pct": float(per_patient_df[col_name].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = args.output_dir / f"action_density_threshold_selection_rates_{stamp}.csv"
    per_patient_path = args.output_dir / f"action_density_threshold_selection_rates_per_patient_{stamp}.csv"
    plot_path = args.output_dir / f"action_density_selection_ratio_distribution_{stamp}.png"

    summary_df.to_csv(summary_path, index=False)
    per_patient_df.to_csv(per_patient_path, index=False)
    _plot_distribution(per_patient_df, thresholds=thresholds, output_path=plot_path)

    print(f"ICU stays analyzed: {len(per_patient_df)}")
    print(f"Total windows analyzed: {len(all_density_arr)}")
    print("\nSelection-rate table (%):")
    print(
        summary_df.to_string(
            index=False,
            formatters={
                "threshold": lambda x: f"{x:g}",
                "selected_windows_global_pct": lambda x: f"{x:.2f}",
                "selected_windows_mean_patient_pct": lambda x: f"{x:.2f}",
            },
        )
    )
    print(f"\nSaved summary CSV: {summary_path}")
    print(f"Saved per-patient CSV: {per_patient_path}")
    print(f"Saved distribution plot: {plot_path}")


if __name__ == "__main__":
    main()
