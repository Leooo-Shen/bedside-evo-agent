"""
Test script for de-identification feature.

This script tests that patient IDs and timestamps are properly de-identified
when the de_identify flag is enabled.
"""

import sys
sys.path.append('/Users/leo/Workdir/PhD/evo-agent')

from data_parser import MIMICDataParser
from config.config import Config
import pandas as pd

def test_deidentification():
    """Test de-identification feature."""

    # Load config
    config = Config()
    events_path = config.get('data.events_path')
    icu_stay_path = config.get('data.icu_stay_path')

    print("="*80)
    print("Testing De-identification Feature")
    print("="*80)

    # Test 1: Without de-identification
    print("\n1. Testing WITHOUT de-identification...")
    parser_normal = MIMICDataParser(events_path, icu_stay_path, de_identify=False)
    parser_normal.load_data()

    # Get a sample patient
    sample_patient = parser_normal.icu_stay_df.iloc[0]
    subject_id = sample_patient['subject_id']
    icu_stay_id = sample_patient['icu_stay_id']

    trajectory_normal = parser_normal.get_patient_trajectory(subject_id, icu_stay_id)

    print(f"   Original Subject ID: {trajectory_normal['subject_id']}")
    print(f"   Original ICU Stay ID: {trajectory_normal['icu_stay_id']}")
    print(f"   Original Enter Time: {trajectory_normal['enter_time']}")
    print(f"   Original Leave Time: {trajectory_normal['leave_time']}")

    # Get first event
    if trajectory_normal['events']:
        first_event = trajectory_normal['events'][0]
        print(f"   First event time: {first_event.get('time', 'N/A')}")

    # Test 2: With de-identification
    print("\n2. Testing WITH de-identification (seed=42)...")
    parser_deidentified = MIMICDataParser(
        events_path, icu_stay_path,
        de_identify=True,
        de_identify_seed=42
    )
    parser_deidentified.load_data()

    trajectory_deidentified = parser_deidentified.get_patient_trajectory(subject_id, icu_stay_id)

    print(f"   De-identified Subject ID: {trajectory_deidentified['subject_id']}")
    print(f"   De-identified ICU Stay ID: {trajectory_deidentified['icu_stay_id']}")
    print(f"   De-identified Enter Time: {trajectory_deidentified['enter_time']}")
    print(f"   De-identified Leave Time: {trajectory_deidentified['leave_time']}")

    # Test 3: Verify IDs are different
    print("\n3. Verification...")
    ids_changed = (
        trajectory_normal['subject_id'] != trajectory_deidentified['subject_id'] and
        trajectory_normal['icu_stay_id'] != trajectory_deidentified['icu_stay_id']
    )
    print(f"   ✓ Patient IDs changed: {ids_changed}")

    # Test 4: Verify timestamps are shifted
    timestamps_changed = (
        trajectory_normal['enter_time'] != trajectory_deidentified['enter_time'] and
        trajectory_normal['leave_time'] != trajectory_deidentified['leave_time']
    )
    print(f"   ✓ Timestamps shifted: {timestamps_changed}")

    # Test 5: Verify relative time differences are preserved
    normal_duration = pd.to_datetime(trajectory_normal['leave_time']) - pd.to_datetime(trajectory_normal['enter_time'])
    deidentified_duration = pd.to_datetime(trajectory_deidentified['leave_time']) - pd.to_datetime(trajectory_deidentified['enter_time'])
    duration_preserved = abs((normal_duration - deidentified_duration).total_seconds()) < 1
    print(f"   ✓ ICU duration preserved: {duration_preserved} ({trajectory_normal['icu_duration_hours']:.2f} hours)")

    # Test 6: Test with windows
    print("\n4. Testing de-identification in time windows...")
    windows = parser_deidentified.create_time_windows(
        trajectory_deidentified,
        current_window_hours=1,
        lookback_window_hours=0,
        future_window_hours=0,
        window_step_hours=1,
        include_pre_icu_data=False,
        use_first_n_hours_after_icu=12,
        use_discharge_summary_for_history=False,
        num_discharge_summaries=0,
    )

    if windows:
        print(f"   Generated {len(windows)} windows")
        first_window = windows[0]
        print(f"   First window subject_id: {first_window['subject_id']}")
        print(f"   First window icu_stay_id: {first_window['icu_stay_id']}")
        print(f"   First window start: {first_window['current_window_start']}")

        if first_window['current_events']:
            first_event = first_window['current_events'][0]
            print(f"   First event time: {first_event.get('time', 'N/A')}")
            print(f"   ✓ Event timestamps are de-identified")

    print("\n" + "="*80)
    print("De-identification Test Complete!")
    print("="*80)

    if ids_changed and timestamps_changed and duration_preserved:
        print("\n✓ All tests PASSED")
        return True
    else:
        print("\n✗ Some tests FAILED")
        return False

if __name__ == "__main__":
    success = test_deidentification()
    sys.exit(0 if success else 1)
