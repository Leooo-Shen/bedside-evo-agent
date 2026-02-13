"""
Experiment: Analyze Oracle patient status scores vs outcomes

This script:
1. Finds one patient who died and one who survived
2. Runs Oracle evaluation on their entire trajectories
3. Plots patient_status_score over time for comparison
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.oracle import MetaOracle, save_oracle_reports
from config.config import load_config
from data_parser import MIMICDataParser


def find_patients_by_outcome(parser: MIMICDataParser):
    """
    Find one patient who died and one who survived.

    Returns:
        tuple: (died_patient, survived_patient) as trajectory dictionaries
    """
    print("Searching for patients with different outcomes...")

    died_patient = None
    survived_patient = None

    for _, icu_stay in parser.icu_stay_df.iterrows():
        try:
            trajectory = parser.get_patient_trajectory(icu_stay["subject_id"], icu_stay["icu_stay_id"])

            if not trajectory["survived"] and died_patient is None:
                died_patient = trajectory
                print(
                    f"  Found patient who died: Subject {trajectory['subject_id']}, ICU Stay {trajectory['icu_stay_id']}"
                )
                print(f"    Duration: {trajectory['icu_duration_hours']:.1f} hours")

            elif trajectory["survived"] and survived_patient is None:
                survived_patient = trajectory
                print(
                    f"  Found patient who survived: Subject {trajectory['subject_id']}, ICU Stay {trajectory['icu_stay_id']}"
                )
                print(f"    Duration: {trajectory['icu_duration_hours']:.1f} hours")

            # Stop once we have both
            if died_patient and survived_patient:
                break

        except Exception as e:
            print(f"  Error processing ICU stay {icu_stay['icu_stay_id']}: {e}")
            continue

    return died_patient, survived_patient


def run_oracle_on_patient(
    oracle: MetaOracle, parser: MIMICDataParser, trajectory: dict, config, parallel: bool = True, max_workers: int = 10
):
    """
    Run Oracle evaluation on a single patient trajectory.

    Args:
        oracle: MetaOracle instance
        parser: MIMICDataParser instance
        trajectory: Patient trajectory dictionary
        config: Configuration object
        parallel: Whether to use parallel evaluation (default: True)
        max_workers: Maximum number of parallel workers (default: 10)

    Returns:
        list: Oracle reports for all windows
    """
    subject_id = trajectory["subject_id"]
    icu_stay_id = trajectory["icu_stay_id"]

    print(f"\nProcessing Patient {subject_id}, ICU Stay {icu_stay_id}")
    print(f"  Duration: {trajectory['icu_duration_hours']:.1f} hours")
    print(f"  Outcome: {'Survived' if trajectory['survived'] else 'Died'}")

    # Create time windows
    windows = parser.create_time_windows(
        trajectory,
        current_window_hours=config.current_window_hours,
        lookback_window_hours=config.lookback_window_hours,
        future_window_hours=config.future_window_hours,
        window_step_hours=config.window_step_hours,
        include_pre_icu_data=config.include_pre_icu_data,
    )

    print(f"  Generated {len(windows)} time windows")

    if len(windows) == 0:
        print(f"  No windows generated, skipping")
        return []

    # Evaluate windows (parallel or sequential)
    if parallel:
        print(f"  Using parallel evaluation with {max_workers} workers")
        reports = oracle.evaluate_trajectory_parallel(windows, max_workers=max_workers)
    else:
        reports = oracle.evaluate_trajectory(windows)

    print(f"  Completed: {len(reports)} evaluations")

    # Save trajectory log
    oracle.save_trajectory_log(subject_id, icu_stay_id, run_id="unblinded_experiment")

    return reports


def plot_score_trajectories(died_reports, survived_reports, output_path: str):
    """
    Plot patient_status_score over time for both patients.

    Args:
        died_reports: Oracle reports for patient who died
        survived_reports: Oracle reports for patient who survived
        output_path: Path to save the plot
    """
    # Extract data for died patient
    died_hours = [r.window_data["hours_since_admission"] for r in died_reports]
    died_scores = [r.patient_status_score for r in died_reports]

    # Extract data for survived patient
    survived_hours = [r.window_data["hours_since_admission"] for r in survived_reports]
    survived_scores = [r.patient_status_score for r in survived_reports]

    # Create plot
    plt.figure(figsize=(12, 6))

    plt.plot(
        died_hours, died_scores, "o-", color="red", label="Patient who died", linewidth=2, markersize=6, alpha=0.7
    )
    plt.plot(
        survived_hours,
        survived_scores,
        "o-",
        color="green",
        label="Patient who survived",
        linewidth=2,
        markersize=6,
        alpha=0.7,
    )

    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="Neutral status (0)")
    plt.xlabel("Hours since ICU admission", fontsize=12)
    plt.ylabel("Patient Status Score", fontsize=12)
    plt.title("Oracle Patient Status Score Over Time: Survivors vs Non-Survivors", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.1, 1.1)

    # Add annotations
    plt.text(
        0.02,
        0.98,
        "Score range: -1.0 (critically ill) to 1.0 (improving)",
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Print statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"\nPatient who died:")
    print(f"  Mean score: {sum(died_scores) / len(died_scores):.3f}")
    print(f"  Min score: {min(died_scores):.3f}")
    print(f"  Max score: {max(died_scores):.3f}")
    print(f"  Final score: {died_scores[-1]:.3f}")
    print(f"  Score trend (first to last): {died_scores[-1] - died_scores[0]:.3f}")

    print(f"\nPatient who survived:")
    print(f"  Mean score: {sum(survived_scores) / len(survived_scores):.3f}")
    print(f"  Min score: {min(survived_scores):.3f}")
    print(f"  Max score: {max(survived_scores):.3f}")
    print(f"  Final score: {survived_scores[-1]:.3f}")
    print(f"  Score trend (first to last): {survived_scores[-1] - survived_scores[0]:.3f}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("EXPERIMENT: Oracle Score Trajectory Analysis (UNBLINDED)")
    print("=" * 80)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%m%d-%H%M")
    output_dir = Path(f"experiments/result-{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Load configuration
    config = load_config()

    # Initialize data parser
    print("\nInitializing data parser...")
    parser = MIMICDataParser(config.events_path, config.icu_stay_path)
    parser.load_data()

    # Find patients with different outcomes
    died_patient, survived_patient = find_patients_by_outcome(parser)

    if not died_patient or not survived_patient:
        print("\nERROR: Could not find patients with both outcomes")
        return

    # Initialize Oracle (unblinded mode)
    print("\nInitializing Meta Oracle...")
    print(f"  Provider: {config.llm_provider}")
    print(f"  Model: {config.llm_model}")
    print(f"  Mode: UNBLINDED")
    oracle = MetaOracle(
        provider=config.llm_provider,
        model=config.llm_model,
        window_hours=config.current_window_hours,
        temperature=config.llm_temperature,
        blinded=False,  # Explicitly set unblinded mode
    )

    # Run Oracle on both patients
    print("\n" + "=" * 80)
    print("RUNNING ORACLE EVALUATIONS")
    print("=" * 80)

    died_reports = run_oracle_on_patient(oracle, parser, died_patient, config)
    survived_reports = run_oracle_on_patient(oracle, parser, survived_patient, config)

    if not died_reports or not survived_reports:
        print("\nERROR: Failed to generate reports for one or both patients")
        return

    # Save reports with unblinded suffix
    died_output = output_dir / f"died_patient_{died_patient['subject_id']}_reports_unblinded.json"
    survived_output = output_dir / f"survived_patient_{survived_patient['subject_id']}_reports_unblinded.json"

    save_oracle_reports(died_reports, str(died_output), include_window_data=True)
    save_oracle_reports(survived_reports, str(survived_output), include_window_data=True)

    # Plot trajectories with unblinded suffix
    plot_path = output_dir / "score_trajectory_comparison_unblinded.png"
    plot_score_trajectories(died_reports, survived_reports, str(plot_path))

    # Print Oracle statistics
    print("\n" + "=" * 80)
    print("ORACLE USAGE STATISTICS")
    print("=" * 80)
    stats = oracle.get_statistics()
    print(f"  Total evaluations: {stats['total_evaluations']}")
    print(f"  Total tokens used: {stats['total_tokens_used']:,}")
    print(f"  Avg tokens per evaluation: {stats['avg_tokens_per_evaluation']:.0f}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
