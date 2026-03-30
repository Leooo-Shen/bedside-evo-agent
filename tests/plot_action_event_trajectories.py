"""
Sample ICU trajectories and plot actionable event counts in 30-minute windows.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import get_config
from data_parser import MIMICDataParser
from experiments.oracle.action_validity_common import ACTIONABLE_EVENT_CODES


def _count_action_events_per_window(
    trajectory: Dict,
    action_codes: Iterable[str],
    window_minutes: int = 30,
) -> pd.DataFrame:
    enter_time = pd.to_datetime(trajectory.get("enter_time"), errors="coerce")
    leave_time = pd.to_datetime(trajectory.get("leave_time"), errors="coerce")

    if pd.isna(enter_time) or pd.isna(leave_time) or leave_time <= enter_time:
        return pd.DataFrame()

    window_delta = pd.Timedelta(minutes=window_minutes)
    action_code_set = {str(code).strip().upper() for code in action_codes}

    window_starts: List[pd.Timestamp] = []
    window_ends: List[pd.Timestamp] = []
    current_start = enter_time
    while current_start < leave_time:
        current_end = min(current_start + window_delta, leave_time)
        window_starts.append(current_start)
        window_ends.append(current_end)
        current_start = current_end

    if not window_starts:
        return pd.DataFrame()

    counts = [0] * len(window_starts)
    for event in trajectory.get("events", []):
        code = str(event.get("code") or "").strip().upper()
        if code not in action_code_set:
            continue

        event_time = pd.to_datetime(event.get("time"), errors="coerce")
        if pd.isna(event_time):
            continue

        if event_time < enter_time or event_time > leave_time:
            continue

        window_index = int((event_time - enter_time) // window_delta)
        if window_index >= len(counts):
            window_index = len(counts) - 1
        if window_index < 0:
            continue
        counts[window_index] += 1

    rows = []
    for idx, (start, end) in enumerate(zip(window_starts, window_ends)):
        rows.append(
            {
                "window_index": idx,
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "hours_since_admission": (start - enter_time).total_seconds() / 3600.0,
                "action_event_count": counts[idx],
            }
        )

    return pd.DataFrame(rows)


def _plot_patient_action_trajectories(patient_frames: List[pd.DataFrame], output_path: Path) -> None:
    n_patients = len(patient_frames)
    fig, axes = plt.subplots(n_patients, 1, figsize=(14, max(3.0, 2.8 * n_patients)))

    if n_patients == 1:
        axes = [axes]

    for ax, patient_df in zip(axes, patient_frames):
        subject_id = int(patient_df["subject_id"].iloc[0])
        icu_stay_id = int(patient_df["icu_stay_id"].iloc[0])
        survived = bool(patient_df["survived"].iloc[0])
        outcome_text = "survived" if survived else "died"

        ax.plot(
            patient_df["hours_since_admission"],
            patient_df["action_event_count"],
            color="#1f77b4",
            linewidth=1.8,
            drawstyle="steps-post",
        )
        ax.scatter(
            patient_df["hours_since_admission"],
            patient_df["action_event_count"],
            color="#1f77b4",
            s=20,
            zorder=3,
        )
        ax.set_ylabel("# Action events")
        ax.set_title(f"Patient {subject_id} | ICU {icu_stay_id} | Outcome: {outcome_text}")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Hours since ICU admission (30-minute windows)")
    fig.suptitle("Action-event trajectories for sampled ICU patients", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Plot action-event counts in 30-minute ICU windows.")
    arg_parser.add_argument("--num-patients", type=int, default=5, help="Number of ICU trajectories to sample.")
    arg_parser.add_argument("--window-minutes", type=int, default=30, help="Window size in minutes.")
    arg_parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    arg_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory to save plot and CSV.",
    )
    args = arg_parser.parse_args()

    if args.num_patients < 1:
        raise ValueError("--num-patients must be >= 1")
    if args.window_minutes < 1:
        raise ValueError("--window-minutes must be >= 1")

    config = get_config()
    parser = MIMICDataParser(events_path=config.events_path, icu_stay_path=config.icu_stay_path)
    parser.load_data()

    icu_stays = parser.icu_stay_df.copy()
    if len(icu_stays) == 0:
        raise ValueError("No ICU stays available after filtering.")

    sample_size = min(args.num_patients, len(icu_stays))
    sampled_stays = icu_stays.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)

    patient_frames: List[pd.DataFrame] = []
    for _, stay in sampled_stays.iterrows():
        subject_id = int(stay["subject_id"])
        icu_stay_id = int(stay["icu_stay_id"])
        trajectory = parser.get_patient_trajectory(subject_id, icu_stay_id, icu_stay=stay)

        counts_df = _count_action_events_per_window(
            trajectory=trajectory,
            action_codes=ACTIONABLE_EVENT_CODES,
            window_minutes=args.window_minutes,
        )

        if len(counts_df) == 0:
            continue

        counts_df["subject_id"] = subject_id
        counts_df["icu_stay_id"] = icu_stay_id
        counts_df["survived"] = bool(trajectory.get("survived"))
        patient_frames.append(counts_df)

    if len(patient_frames) == 0:
        raise ValueError("No trajectories produced valid 30-minute windows.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"sampled_action_event_counts_{args.window_minutes}min_{run_stamp}.csv"
    plot_path = args.output_dir / f"sampled_action_event_trajectories_{args.window_minutes}min_{run_stamp}.png"

    combined = pd.concat(patient_frames, ignore_index=True)
    combined.to_csv(csv_path, index=False)
    _plot_patient_action_trajectories(patient_frames, output_path=plot_path)

    sampled_ids = sorted({f"{int(df['subject_id'].iloc[0])}_{int(df['icu_stay_id'].iloc[0])}" for df in patient_frames})
    print("Sampled trajectories:")
    for patient_id in sampled_ids:
        print(f"  - {patient_id}")
    print(f"\nSaved counts CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
