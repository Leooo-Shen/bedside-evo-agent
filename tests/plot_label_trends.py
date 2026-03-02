"""
Plot vital sign values with window-averaged labels showing classification trends.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import get_config
from data_parser import MIMICDataParser
from prompts.agent_prompt import _calculate_vital_status, _get_vital_names


def plot_vitals_with_label_trend(trajectory, windows, output_path=None):
    """
    Plot vital signs with window-averaged values color-coded by classification.

    Args:
        trajectory: Patient trajectory from data parser
        windows: Time windows from data parser
        output_path: Path to save the plot (if None, displays instead)
    """
    # Extract vitals timeseries
    vital_names = _get_vital_names()
    vitals_data = {}

    for vital_name in vital_names:
        vitals_data[vital_name] = []

    # Extract vitals from events
    for event in trajectory["events"]:
        if event.get("code") == "VITALS":
            vital_name = event.get("code_specifics")
            if vital_name in vital_names:
                time = event.get("time")
                value = event.get("numeric_value")

                if time and value is not None:
                    if isinstance(time, str):
                        time = pd.to_datetime(time)
                    vitals_data[vital_name].append((time, value))

    # Filter out vitals with no data
    vitals_with_data = {k: v for k, v in vitals_data.items() if len(v) > 0}

    if not vitals_with_data:
        print("No vital signs data found")
        return

    # Extract window-level classifications
    window_classifications = {}
    for vital_name in vital_names:
        display_name = vital_name.split(",")[0]
        window_classifications[display_name] = []

    icu_admission_time = pd.to_datetime(trajectory["enter_time"])

    for idx, window in enumerate(windows):
        current_events = window.get("current_events", [])

        # Get previous window events if available
        previous_events = None
        if idx > 0:
            previous_events = windows[idx - 1].get("current_events", [])

        vital_status = _calculate_vital_status(current_events, previous_events)
        hours = window.get("hours_since_admission", 0)
        window_time = icu_admission_time + pd.Timedelta(hours=hours + 0.25)  # Middle of 30-min window

        for display_name in window_classifications.keys():
            if display_name in vital_status:
                window_data = {
                    "time": window_time,
                    "value": vital_status[display_name]["current_avg"],
                    "status": vital_status[display_name]["status"],
                }

                # Add delta and trend if available
                if "delta" in vital_status[display_name]:
                    window_data["delta"] = vital_status[display_name]["delta"]
                    window_data["trend"] = vital_status[display_name]["trend"]

                window_classifications[display_name].append(window_data)

    # Create figure
    n_vitals = len(vitals_with_data)
    fig, axes = plt.subplots(n_vitals, 1, figsize=(16, 3.5 * n_vitals), sharex=True)

    if n_vitals == 1:
        axes = [axes]

    # Get patient info
    subject_id = trajectory["subject_id"]
    icu_stay_id = trajectory["icu_stay_id"]
    age = trajectory["age_at_admission"]
    survived = trajectory["survived"]
    icu_duration = trajectory["icu_duration_hours"]

    # Color mapping
    color_map = {
        "above_normal": "red",
        "normal": "green",
        "below_normal": "blue",
    }

    # Plot each vital
    for idx, (vital_name, data) in enumerate(vitals_with_data.items()):
        ax = axes[idx]
        display_name = vital_name.split(",")[0]

        # Sort by time
        data_sorted = sorted(data, key=lambda x: x[0])
        times = [t for t, v in data_sorted]
        values = [v for t, v in data_sorted]

        # Plot the actual vital values (thin gray line)
        ax.plot(
            times, values, linestyle="-", linewidth=1, color="lightgray", alpha=0.6, zorder=1, label="All measurements"
        )

        # Add normal range shading
        if "Heart Rate" in vital_name:
            ax.axhspan(60, 100, alpha=0.08, color="green", zorder=0)
            ax.axhline(y=60, color="green", linestyle="--", linewidth=0.8, alpha=0.3)
            ax.axhline(y=100, color="green", linestyle="--", linewidth=0.8, alpha=0.3)
        elif "systolic" in vital_name:
            ax.axhspan(90, 120, alpha=0.08, color="green", zorder=0)
            ax.axhline(y=90, color="green", linestyle="--", linewidth=0.8, alpha=0.3)
            ax.axhline(y=120, color="green", linestyle="--", linewidth=0.8, alpha=0.3)
        elif "diastolic" in vital_name:
            ax.axhspan(60, 80, alpha=0.08, color="green", zorder=0)
            ax.axhline(y=60, color="green", linestyle="--", linewidth=0.8, alpha=0.3)
            ax.axhline(y=80, color="green", linestyle="--", linewidth=0.8, alpha=0.3)
        elif "Respiratory Rate" in vital_name:
            ax.axhspan(12, 20, alpha=0.08, color="green", zorder=0)
            ax.axhline(y=12, color="green", linestyle="--", linewidth=0.8, alpha=0.3)
            ax.axhline(y=20, color="green", linestyle="--", linewidth=0.8, alpha=0.3)
        elif "O2 saturation" in vital_name:
            ax.axhspan(95, 100, alpha=0.08, color="green", zorder=0)
            ax.axhline(y=95, color="green", linestyle="--", linewidth=0.8, alpha=0.3)

        # Plot window-averaged values with classification colors
        if display_name in window_classifications:
            window_data = window_classifications[display_name]

            # Separate by status for legend
            for status in ["above_normal", "normal", "below_normal"]:
                status_data = [w for w in window_data if w["status"] == status]
                if status_data:
                    status_times = [w["time"] for w in status_data]
                    status_values = [w["value"] for w in status_data]

                    label_map = {
                        "above_normal": "Above normal",
                        "normal": "Normal",
                        "below_normal": "Below normal",
                    }

                    ax.scatter(
                        status_times,
                        status_values,
                        color=color_map[status],
                        s=100,
                        marker="o",
                        edgecolors="black",
                        linewidths=1.5,
                        zorder=3,
                        label=f"{label_map[status]} (window avg)",
                    )

            # Connect window averages with line
            all_times = [w["time"] for w in window_data]
            all_values = [w["value"] for w in window_data]
            ax.plot(all_times, all_values, linestyle="--", linewidth=1.5, color="black", alpha=0.4, zorder=2)

            # Add trend arrows for windows with delta information
            for w in window_data:
                if "trend" in w and w["trend"] != "stable":
                    # Determine arrow direction and color
                    if w["trend"] == "up":
                        arrow_symbol = "↑"
                        arrow_color = "darkred"
                    else:  # down
                        arrow_symbol = "↓"
                        arrow_color = "darkblue"

                    # Add arrow annotation
                    ax.annotate(
                        arrow_symbol,
                        xy=(w["time"], w["value"]),
                        xytext=(0, 15 if w["trend"] == "up" else -15),
                        textcoords="offset points",
                        fontsize=14,
                        fontweight="bold",
                        color=arrow_color,
                        ha="center",
                        va="center",
                        zorder=4,
                    )

        # Labels
        ax.set_ylabel(display_name, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2, zorder=0)

        # Add legend for first plot
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9, ncol=2)

    # Set x-axis label
    axes[-1].set_xlabel("Time", fontsize=11, fontweight="bold")
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Title
    outcome_str = "Survived" if survived else "Died"
    fig.suptitle(
        f"Patient {subject_id} (ICU Stay {icu_stay_id}) - Vital Signs with Label Trends\n"
        f"Age: {age:.1f} years | ICU Duration: {icu_duration:.1f} hours | Outcome: {outcome_str}",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    """Plot vitals with label trends."""

    config = get_config()

    print("=" * 80)
    print("VITAL SIGNS WITH LABEL TRENDS")
    print("=" * 80)

    # Load data
    print("\n1. Loading MIMIC-demo data...")
    parser = MIMICDataParser(events_path=config.events_path, icu_stay_path=config.icu_stay_path)
    parser.load_data()

    # Select patient
    print("\n2. Selecting patient...")
    first_icu_stay = parser.icu_stay_df.iloc[3]
    subject_id = first_icu_stay["subject_id"]
    icu_stay_id = first_icu_stay["icu_stay_id"]

    print(f"   Subject ID: {subject_id}")
    print(f"   ICU Stay ID: {icu_stay_id}")
    print(f"   Duration: {first_icu_stay['icu_duration_hours']:.1f} hours")
    print(f"   Outcome: {'Survived' if first_icu_stay['survived'] else 'Died'}")

    # Get trajectory
    print("\n3. Extracting patient trajectory...")
    trajectory = parser.get_patient_trajectory(subject_id, icu_stay_id)

    # Create windows
    print("\n4. Creating time windows...")
    windows = parser.create_time_windows(
        trajectory,
        current_window_hours=config.oracle_current_window_hours,
        window_step_hours=0.5,  # 30-minute step size
        include_pre_icu_data=config.oracle_include_pre_icu_data,
    )
    print(f"   Generated {len(windows)} time windows")

    # Create plot
    print("\n5. Creating visualization...")
    output_path = f"{config.output_dir}/patient_{subject_id}_label_trends.png"
    plot_vitals_with_label_trend(trajectory, windows, output_path=output_path)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
