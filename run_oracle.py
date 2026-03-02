"""
Batch Processing Pipeline for Meta Oracle

This script processes MIMIC-demo patient trajectories through the Oracle
to generate ground truth evaluations.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from config.config import load_config, get_config
from data_parser import MIMICDataParser
from agents.oracle import MetaOracle, save_oracle_reports


def process_batch_for_oracle(
    events_path: str,
    icu_stay_path: str,
    output_dir: str,
    provider: str = "anthropic",
    model: str = None,
    current_window_hours: float = 0.5,
    window_step_hours: float = 0.5,
    include_pre_icu_data: bool = True,
    max_patients: int = None,
    save_trajectories: bool = True
):
    """
    Process a batch of patient trajectories through the Oracle.

    Args:
        events_path: Path to events parquet file
        icu_stay_path: Path to ICU stay parquet file
        output_dir: Directory to save outputs
        provider: LLM provider ("anthropic", "openai", "google", or "gemini")
        model: Model name (optional)
        current_window_hours: Size of current observation window (default 0.5 = 30 minutes)
        window_step_hours: Step size between sliding windows (default 0.5 = 30 minutes)
        include_pre_icu_data: Whether to include pre-ICU hospital data (default True)
        max_patients: Maximum number of patients to process (None = all)
        save_trajectories: Whether to save intermediate trajectory data
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize data parser
    print("=" * 80)
    print("MIMIC-DEMO ORACLE PROCESSING PIPELINE")
    print("=" * 80)
    print(f"\nInitializing data parser...")
    parser = MIMICDataParser(events_path, icu_stay_path)
    parser.load_data()

    # Initialize Oracle
    print(f"\nInitializing Meta Oracle...")
    print(f"  Provider: {provider}")
    print(f"  Model: {model or 'default'}")
    print(f"  Current window: {current_window_hours} hours ({current_window_hours * 60:.0f} minutes)")
    print(f"  Window step size: {window_step_hours} hours ({window_step_hours * 60:.0f} minutes)")
    print(f"  Include pre-ICU data: {include_pre_icu_data}")

    oracle = MetaOracle(
        provider=provider,
        model=model,
        window_hours=current_window_hours,
        temperature=0.3  # Lower temperature for consistent evaluations
    )

    # Get all trajectories
    print(f"\nExtracting patient trajectories...")
    all_trajectories = parser.get_all_trajectories()

    if max_patients:
        all_trajectories = all_trajectories[:max_patients]
        print(f"  Limited to first {max_patients} patients")

    print(f"  Total trajectories: {len(all_trajectories)}")

    # Save trajectories if requested
    if save_trajectories:
        trajectories_file = output_path / "patient_trajectories.jsonl"
        parser.save_trajectories(all_trajectories, str(trajectories_file))

    # Process each trajectory
    print(f"\n{'=' * 80}")
    print("PROCESSING TRAJECTORIES")
    print("=" * 80)

    all_reports = []
    summary_stats = {
        'total_patients': len(all_trajectories),
        'total_windows_evaluated': 0,
        'patients_processed': 0,
        'patients_failed': 0,
        'avg_score_per_patient': []
    }

    for i, trajectory in enumerate(all_trajectories):
        subject_id = trajectory['subject_id']
        icu_stay_id = trajectory['icu_stay_id']

        print(f"\n[{i+1}/{len(all_trajectories)}] Processing Patient {subject_id}, ICU Stay {icu_stay_id}")
        print(f"  Duration: {trajectory['icu_duration_hours']:.1f} hours")
        print(f"  Outcome: {'Survived' if trajectory['survived'] else 'Died'}")

        try:
            # Create time windows
            windows = parser.create_time_windows(
                trajectory,
                current_window_hours=current_window_hours,
                window_step_hours=window_step_hours,
                include_pre_icu_data=include_pre_icu_data,
                use_first_n_hours_after_icu=config.oracle_observation_hours,
                use_discharge_summary_for_history=config.oracle_use_discharge_summary_for_history,
                num_discharge_summaries=config.oracle_num_discharge_summaries
            )

            # TODO: Extract discharge summary separately for Oracle ground truth
            # discharge_summary = parser.extract_discharge_summary(trajectory)
            # if discharge_summary:
            #     # Save or use discharge summary for Oracle evaluation
            #     pass

            print(f"  Generated {len(windows)} time windows")

            if len(windows) == 0:
                print(f"  Skipping (no windows generated)")
                continue

            # Evaluate windows
            reports = oracle.evaluate_trajectory(windows)

            # Calculate average score for this patient
            scores = [r.patient_status_score for r in reports]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            summary_stats['avg_score_per_patient'].append(avg_score)

            print(f"  Completed: {len(reports)} evaluations")
            print(f"  Average patient status score: {avg_score:.3f}")

            # Save individual patient report
            patient_output = output_path / f"patient_{subject_id}_icu_{icu_stay_id}_oracle_report.json"
            save_oracle_reports(reports, str(patient_output), include_window_data=True)

            all_reports.extend(reports)
            summary_stats['total_windows_evaluated'] += len(reports)
            summary_stats['patients_processed'] += 1

        except Exception as e:
            print(f"  ERROR: Failed to process patient: {e}")
            summary_stats['patients_failed'] += 1
            continue

    # Save combined report
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print("=" * 80)

    combined_output = output_path / "all_oracle_reports.json"
    save_oracle_reports(all_reports, str(combined_output), include_window_data=True)

    # Calculate summary statistics
    if summary_stats['avg_score_per_patient']:
        overall_avg = sum(summary_stats['avg_score_per_patient']) / len(summary_stats['avg_score_per_patient'])
        summary_stats['overall_avg_score'] = overall_avg
    else:
        summary_stats['overall_avg_score'] = 0.0

    # Add Oracle statistics
    oracle_stats = oracle.get_statistics()
    summary_stats.update(oracle_stats)

    # Save summary
    summary_file = output_path / "processing_summary.json"
    with open(summary_file, 'w') as f:
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


def main():
    """Main entry point for Oracle batch processing."""
    # Load configuration
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Process MIMIC-demo data through Meta Oracle for retrospective evaluation"
    )

    parser.add_argument(
        "--events",
        type=str,
        default=config.events_path,
        help=f"Path to events parquet file (default: {config.events_path})"
    )

    parser.add_argument(
        "--icu-stay",
        type=str,
        default=config.icu_stay_path,
        help=f"Path to ICU stay parquet file (default: {config.icu_stay_path})"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=config.output_dir,
        help=f"Output directory for Oracle reports (default: {config.output_dir})"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=config.llm_provider,
        choices=["anthropic", "openai", "google", "gemini"],
        help=f"LLM provider (default: {config.llm_provider})"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=config.llm_model,
        help=f"Model name (default: {config.llm_model or 'provider default'})"
    )

    parser.add_argument(
        "--current-window-hours",
        type=float,
        default=config.oracle_current_window_hours,
        help=f"Size of current observation window in hours (default: {config.oracle_current_window_hours})"
    )

    parser.add_argument(
        "--window-step-hours",
        type=float,
        default=config.oracle_window_step_hours,
        help=f"Step size between sliding windows in hours (default: {config.oracle_window_step_hours})"
    )

    parser.add_argument(
        "--no-pre-icu-data",
        action="store_true",
        help="Exclude pre-ICU hospital data from history context"
    )

    parser.add_argument(
        "--max-patients",
        type=int,
        default=config.max_patients,
        help="Maximum number of patients to process (for testing)"
    )

    parser.add_argument(
        "--no-save-trajectories",
        action="store_true",
        help="Don't save intermediate trajectory data"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.json file"
    )

    args = parser.parse_args()

    # Reload config if custom path provided
    if args.config:
        config = load_config(args.config)
        print(f"Loaded custom config from: {args.config}")

    # Run batch processing
    process_batch_for_oracle(
        events_path=args.events,
        icu_stay_path=args.icu_stay,
        output_dir=args.output,
        provider=args.provider,
        model=args.model,
        current_window_hours=args.current_window_hours,
        window_step_hours=args.window_step_hours,
        include_pre_icu_data=not args.no_pre_icu_data,
        max_patients=args.max_patients,
        save_trajectories=not args.no_save_trajectories
    )


if __name__ == "__main__":
    main()
