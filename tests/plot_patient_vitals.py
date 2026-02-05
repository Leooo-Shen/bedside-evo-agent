"""
Plot vital sign trends for a patient from the MIMIC-demo dataset.
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import get_config
from data_parser import MIMICDataParser


def extract_vitals_timeseries(trajectory):
    """
    Extract vital signs as time series from patient trajectory.

    Args:
        trajectory: Patient trajectory from data parser

    Returns:
        Dictionary of vital name -> list of (time, value) tuples
    """
    vitals_data = {}

    # Define vitals to track (using actual MIMIC-demo field names)
    vital_names = [
        "Heart Rate, bpm",
        "Non Invasive Blood Pressure systolic, mmHg",
        "Non Invasive Blood Pressure diastolic, mmHg",
        "Respiratory Rate, insp/min",
        "Temperature Fahrenheit, °F",
        "O2 saturation pulseoxymetry, %",
    ]

    # Initialize empty lists for each vital
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
                    # Convert time to datetime if needed
                    if isinstance(time, str):
                        time = pd.to_datetime(time)
                    vitals_data[vital_name].append((time, value))

    return vitals_data


def plot_vitals(trajectory, output_path=None):
    """
    Create a multi-panel plot of vital signs over time.

    Args:
        trajectory: Patient trajectory from data parser
        output_path: Path to save the plot (if None, displays instead)
    """
    vitals_data = extract_vitals_timeseries(trajectory)

    # Filter out vitals with no data
    vitals_with_data = {k: v for k, v in vitals_data.items() if len(v) > 0}

    if not vitals_with_data:
        print("No vital signs data found for this patient")
        return

    # Create figure with subplots
    n_vitals = len(vitals_with_data)
    fig, axes = plt.subplots(n_vitals, 1, figsize=(12, 3 * n_vitals), sharex=True)

    # Handle case of single vital
    if n_vitals == 1:
        axes = [axes]

    # Get patient info
    subject_id = trajectory["subject_id"]
    icu_stay_id = trajectory["icu_stay_id"]
    age = trajectory["age_at_admission"]
    survived = trajectory["survived"]
    icu_duration = trajectory["icu_duration_hours"]

    # Plot each vital
    for idx, (vital_name, data) in enumerate(vitals_with_data.items()):
        ax = axes[idx]

        # Sort by time
        data_sorted = sorted(data, key=lambda x: x[0])
        times = [t for t, v in data_sorted]
        values = [v for t, v in data_sorted]

        # Plot
        ax.plot(times, values, marker="o", linestyle="-", markersize=4, linewidth=1.5)

        # Use shorter label (remove units)
        display_name = vital_name.split(",")[0]
        ax.set_ylabel(display_name, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add reference lines for normal ranges (approximate)
        if "Heart Rate" in vital_name:
            ax.axhspan(60, 100, alpha=0.1, color="green", label="Normal range")
        elif "systolic" in vital_name:
            ax.axhspan(90, 120, alpha=0.1, color="green", label="Normal range")
        elif "diastolic" in vital_name:
            ax.axhspan(60, 80, alpha=0.1, color="green", label="Normal range")
        elif "Respiratory Rate" in vital_name:
            ax.axhspan(12, 20, alpha=0.1, color="green", label="Normal range")
        elif "Temperature" in vital_name:
            ax.axhspan(97.0, 99.0, alpha=0.1, color="green", label="Normal range")
        elif "O2 saturation" in vital_name:
            ax.axhspan(95, 100, alpha=0.1, color="green", label="Normal range")

        # Add legend for first plot
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Set x-axis label on bottom plot
    axes[-1].set_xlabel("Time", fontsize=10, fontweight="bold")

    # Rotate x-axis labels
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add title
    outcome_str = "Survived" if survived else "Died"
    fig.suptitle(
        f"Patient {subject_id} (ICU Stay {icu_stay_id}) - Vital Signs Over Time\n"
        f"Age: {age:.1f} years | ICU Duration: {icu_duration:.1f} hours | Outcome: {outcome_str}",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    """Plot vitals for a randomly selected patient."""

    # Load configuration
    config = get_config()

    print("=" * 80)
    print("PATIENT VITALS VISUALIZATION")
    print("=" * 80)

    # Initialize data parser
    print("\n1. Loading MIMIC-demo data...")
    parser = MIMICDataParser(events_path=config.events_path, icu_stay_path=config.icu_stay_path)
    parser.load_data()

    # Select a random patient (or use first one)
    print("\n2. Selecting patient...")
    # Use first patient for reproducibility
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

    # Count vitals
    vitals_count = sum(1 for e in trajectory["events"] if e.get("code") == "VITALS")
    print(f"   Found {vitals_count} vital sign measurements")

    # Create plot
    print("\n4. Creating visualization...")
    output_path = f"{config.output_dir}/patient_{subject_id}_vitals.png"
    plot_vitals(trajectory, output_path=output_path)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
