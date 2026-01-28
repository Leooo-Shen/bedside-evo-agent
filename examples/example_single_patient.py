"""
Example: Running Meta Oracle on a Single Patient

This script demonstrates how to use the Meta Oracle system to evaluate
a single patient's ICU stay.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.oracle import MetaOracle, save_oracle_reports
from config.config import get_config
from data_parser import MIMICDataParser


def main():
    """Run Oracle evaluation on a single patient example."""

    # Load configuration
    config = get_config()

    print("=" * 80)
    print("META ORACLE - SINGLE PATIENT EXAMPLE")
    print("=" * 80)

    # Initialize data parser
    print("\n1. Loading MIMIC-demo data...")
    parser = MIMICDataParser(events_path=config.events_path, icu_stay_path=config.icu_stay_path)
    parser.load_data()

    # Get first patient
    print("\n2. Extracting patient trajectory...")
    first_icu_stay = parser.icu_stay_df.iloc[0]
    subject_id = first_icu_stay["subject_id"]
    icu_stay_id = first_icu_stay["icu_stay_id"]

    print(f"   Subject ID: {subject_id}")
    print(f"   ICU Stay ID: {icu_stay_id}")
    print(f"   Duration: {first_icu_stay['icu_duration_hours']:.1f} hours")
    print(f"   Outcome: {'Survived' if first_icu_stay['survived'] else 'Died'}")

    trajectory = parser.get_patient_trajectory(subject_id, icu_stay_id)

    # Create time windows using config values
    print("\n3. Creating time windows...")
    print(f"   Current window: {config.current_window_hours} hours ({config.current_window_hours * 60:.0f} minutes)")
    print(f"   Lookback window: {config.lookback_window_hours} hours")
    print(f"   Future window: {config.future_window_hours} hours")
    print(f"   Window step size: {config.window_step_hours} hours ({config.window_step_hours * 60:.0f} minutes)")
    print(f"   Include pre-ICU data: {config.include_pre_icu_data}")
    windows = parser.create_time_windows(
        trajectory,
        current_window_hours=config.current_window_hours,
        lookback_window_hours=config.lookback_window_hours,
        future_window_hours=config.future_window_hours,
        window_step_hours=config.window_step_hours,
        include_pre_icu_data=config.include_pre_icu_data,
    )
    print(f"   Generated {len(windows)} time windows")

    if len(windows) == 0:
        print("   No windows generated. Patient stay may be too short.")
        return

    # Show first window details
    print("\n4. First window details:")
    first_window = windows[0]
    print(f"   Current window: {first_window['current_window_start']} to {first_window['current_window_end']}")
    print(f"   History start: {first_window['history_start']}")
    print(f"   Future end: {first_window['future_end']}")
    print(f"   Hours since admission: {first_window['hours_since_admission']:.1f}")
    print(f"   History events: {first_window['num_history_events']}")
    print(f"   Current events: {first_window['num_current_events']}")
    print(f"   Future events: {first_window['num_future_events']}")

    # Initialize Oracle using config
    print("\n5. Initializing Meta Oracle...")
    print(f"   Provider: {config.oracle_provider}")
    print(f"   Model: {config.oracle_model or 'default'}")
    print(f"   Temperature: {config.oracle_temperature}")
    print("   NOTE: This requires API key in environment variable")

    try:
        oracle = MetaOracle(
            provider=config.oracle_provider,
            model=config.oracle_model,
            temperature=config.oracle_temperature,
            max_tokens=config.oracle_max_tokens,
            window_hours=config.current_window_hours,
            log_dir=config.log_dir,
        )
        print("   Oracle initialized successfully")
    except Exception as e:
        print(f"   ERROR: Failed to initialize Oracle: {e}")
        print(f"   Please set {config.oracle_provider.upper()}_API_KEY environment variable")
        return

    # Evaluate first window only (for demo)
    print("\n6. Evaluating first time window...")
    print("   This may take 10-30 seconds...")

    try:
        report = oracle.evaluate_window(first_window)

        print("\n" + "=" * 80)
        print("ORACLE EVALUATION REPORT")
        print("=" * 80)
        print(f"\nPatient Status Score: {report.patient_status_score:.2f}")
        print(f"  (Scale: -1.0 = Critically ill, 0.0 = Stable, +1.0 = Improving)")

        print(f"\nStatus Rationale:")
        print(f"  {report.status_rationale}")

        print(f"\nAction Quality: {report.action_quality}")

        print(f"\nRecommended Action:")
        print(f"  {report.recommended_action}")

        print(f"\nClinical Insight:")
        print(f"  {report.clinical_insight}")

        # Save report
        print("\n7. Saving report...")
        save_oracle_reports(
            [report],
            f"{config.output_dir}/example_single_patient_report.json",
            include_window_data=config.include_window_data,
        )

        # Show statistics
        stats = oracle.get_statistics()
        print("\n8. Oracle Statistics:")
        print(f"   Total evaluations: {stats['total_evaluations']}")
        print(f"   Total tokens used: {stats['total_tokens_used']:,}")

        print("\n" + "=" * 80)
        print("EXAMPLE COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"\n   ERROR during evaluation: {e}")
        print("   This may be due to:")
        print("   - Invalid API key")
        print("   - Network issues")
        print("   - Rate limiting")
        print("   - Insufficient API credits")


if __name__ == "__main__":
    main()
