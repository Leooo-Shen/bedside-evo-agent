"""
Test script to run Oracle on 3 patients with only their first 2 windows after ICU entry.
"""

import json
from datetime import datetime
from pathlib import Path

from agents.oracle import MetaOracle, save_oracle_reports
from config.config import load_config
from data_parser import MIMICDataParser


def main():
    # Load configuration
    config = load_config()

    # Parameters
    max_patients = 3
    max_windows_per_patient = 2
    output_dir = "data/oracle_outputs/example_patient"

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TEST: First 2 Windows for 3 Patients")
    print("=" * 80)

    # Initialize data parser
    print(f"\nInitializing data parser...")
    parser = MIMICDataParser(config.events_path, config.icu_stay_path)
    parser.load_data()

    # Initialize Oracle
    print(f"\nInitializing Meta Oracle...")
    print(f"  Provider: {config.oracle_provider}")
    print(f"  Model: {config.oracle_model or 'default'}")
    print(f"  Current window: {config.oracle_current_window_hours} hours ({config.oracle_current_window_hours * 60:.0f} minutes)")
    print(f"  Lookback window: {config.oracle_lookback_window_hours} hours")
    print(f"  Future window: {config.oracle_future_window_hours} hours")
    print(f"  Window step: {config.oracle_window_step_hours} hours ({config.oracle_window_step_hours * 60:.0f} minutes)")

    oracle = MetaOracle(
        provider=config.oracle_provider,
        model=config.oracle_model,
        window_hours=config.oracle_current_window_hours,
        temperature=config.oracle_temperature,
    )

    # Get all trajectories
    print(f"\nExtracting patient trajectories...")
    all_trajectories = parser.get_all_trajectories()
    all_trajectories = all_trajectories[:max_patients]
    print(f"  Processing first {max_patients} patients")

    # Process each trajectory
    print(f"\n{'=' * 80}")
    print("PROCESSING TRAJECTORIES")
    print("=" * 80)

    all_reports = []
    summary_stats = {
        "total_patients": len(all_trajectories),
        "total_windows_evaluated": 0,
        "patients_processed": 0,
        "patients_failed": 0,
        "avg_score_per_patient": [],
    }

    for i, trajectory in enumerate(all_trajectories):
        subject_id = trajectory["subject_id"]
        icu_stay_id = trajectory["icu_stay_id"]

        print(f"\n[{i+1}/{len(all_trajectories)}] Processing Patient {subject_id}, ICU Stay {icu_stay_id}")
        print(f"  Duration: {trajectory['icu_duration_hours']:.1f} hours")
        print(f"  Outcome: {'Survived' if trajectory['survived'] else 'Died'}")

        try:
            # Create time windows
            windows = parser.create_time_windows(
                trajectory,
                current_window_hours=config.oracle_current_window_hours,
                lookback_window_hours=config.oracle_lookback_window_hours,
                future_window_hours=config.oracle_future_window_hours,
                window_step_hours=config.oracle_window_step_hours,
                include_pre_icu_data=config.oracle_include_pre_icu_data,
                use_first_n_hours_after_icu=config.oracle_use_first_n_hours_after_icu,
                remove_discharge_summary=True,  # Remove discharge summary from windows
                use_discharge_summary_for_history=config.oracle_use_discharge_summary_for_history,
                num_discharge_summaries=config.oracle_num_discharge_summaries
            )

            print(f"  Generated {len(windows)} time windows total")

            # Limit to first N windows
            windows = windows[:max_windows_per_patient]
            print(f"  Evaluating first {len(windows)} windows")

            if len(windows) == 0:
                print(f"  Skipping (no windows generated)")
                continue

            # Evaluate windows
            reports = oracle.evaluate_trajectory(windows)

            # Calculate average score for this patient
            scores = [r.patient_status_score for r in reports]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            summary_stats["avg_score_per_patient"].append(avg_score)

            print(f"  Completed: {len(reports)} evaluations")
            print(f"  Average patient status score: {avg_score:.3f}")

            # Save individual patient report
            patient_output = output_path / f"patient_{subject_id}_icu_{icu_stay_id}_oracle_report.json"
            save_oracle_reports(reports, str(patient_output), include_window_data=True)

            all_reports.extend(reports)
            summary_stats["total_windows_evaluated"] += len(reports)
            summary_stats["patients_processed"] += 1

        except Exception as e:
            print(f"  ERROR: Failed to process patient: {e}")
            import traceback

            traceback.print_exc()
            summary_stats["patients_failed"] += 1
            continue

    # Save combined report
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print("=" * 80)

    combined_output = output_path / "all_oracle_reports.json"
    save_oracle_reports(all_reports, str(combined_output), include_window_data=True)

    # Calculate summary statistics
    if summary_stats["avg_score_per_patient"]:
        overall_avg = sum(summary_stats["avg_score_per_patient"]) / len(summary_stats["avg_score_per_patient"])
        summary_stats["overall_avg_score"] = overall_avg
    else:
        summary_stats["overall_avg_score"] = 0.0

    # Add Oracle statistics
    oracle_stats = oracle.get_statistics()
    summary_stats.update(oracle_stats)

    # Save summary
    summary_file = output_path / "processing_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_stats, f, indent=2)

    print(f"\nSummary:")
    print(f"  Patients processed: {summary_stats['patients_processed']}/{summary_stats['total_patients']}")
    print(f"  Patients failed: {summary_stats['patients_failed']}")
    print(f"  Total windows evaluated: {summary_stats['total_windows_evaluated']}")
    print(f"  Overall average score: {summary_stats['overall_avg_score']:.3f}")
    print(f"  Total tokens used: {summary_stats['total_tokens_used']:,}")
    print(f"  Avg tokens per evaluation: {summary_stats['avg_tokens_per_evaluation']:.0f}")

    print(f"\nOutputs saved to: {output_path}")
    print(f"  - Combined report: {combined_output.name}")
    print(f"  - Summary: {summary_file.name}")
    print(f"  - Individual reports: patient_*_oracle_report.json")

    print(f"\n{'=' * 80}")
    print("PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
