#!/usr/bin/env python3
"""
Check if outcome codes exist in patient events, particularly in the last window's current events.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data_parser import MIMICDataParser


def check_patient_outcome_codes(patient_id: int):
    """Check if outcome codes exist in patient events."""

    # Initialize parser
    parser = MIMICDataParser(
        events_path="data/mimic-demo/events/data_0.parquet",
        icu_stay_path="data/mimic-demo/icu_stay/data_0.parquet",
        de_identify=False,
    )

    # Load data
    print("Loading data...")
    parser.load_data()

    # Find ICU stay for this patient
    patient_stays = parser.icu_stay_df[parser.icu_stay_df["subject_id"] == patient_id]

    if len(patient_stays) == 0:
        print(f"No ICU stays found for patient {patient_id}")
        return

    print(f"\nFound {len(patient_stays)} ICU stay(s) for patient {patient_id}")

    # Check each ICU stay
    for idx, stay in patient_stays.iterrows():
        icu_stay_id = stay["icu_stay_id"]
        print(f"\n{'='*80}")
        print(f"ICU Stay ID: {icu_stay_id}")
        print(f"Enter time: {stay['enter_time']}")
        print(f"Leave time: {stay['leave_time']}")
        print(f"Duration: {stay['icu_duration_hours']:.2f} hours")
        print(f"Survived: {stay['survived']}")

        # Get trajectory
        trajectory = parser.get_patient_trajectory(patient_id, icu_stay_id)

        # Check all events for outcome codes
        outcome_codes = ["LEAVE_HOSPITALIZATION", "NOTE_DISCHARGESUMMARY", "META_DEATH", "LEAVE_ICU"]
        events_df = pd.DataFrame(trajectory["events"])

        print(f"\nTotal events: {len(events_df)}")

        if len(events_df) > 0 and "code" in events_df.columns:
            # Find outcome code events
            outcome_events = events_df[events_df["code"].isin(outcome_codes)]

            if len(outcome_events) > 0:
                print(f"\n*** Found {len(outcome_events)} outcome code event(s) ***")
                for _, event in outcome_events.iterrows():
                    print(f"  - Code: {event['code']}")
                    print(f"    Time: {event.get('time', 'N/A')}")
                    print(f"    Code specifics: {event.get('code_specifics', 'N/A')}")
                    print()
            else:
                print("\nNo outcome code events found in all events")

        # Create time windows
        print(f"\n{'='*80}")
        print("Creating time windows...")
        windows = parser.create_time_windows(
            trajectory,
            current_window_hours=0.5,
            lookback_window_hours=2.0,
            future_window_hours=2.0,
            window_step_hours=0.5,  # Using 0.5 to match agent config
            include_pre_icu_data=True,
            use_first_n_hours_after_icu=None,
        )

        print(f"Created {len(windows)} windows")

        if len(windows) > 0:
            # Check last window
            last_window = windows[-1]
            print(f"\n{'='*80}")
            print(f"LAST WINDOW (Window {len(windows)-1}):")
            print(f"  Hours since admission: {last_window['hours_since_admission']:.2f}")
            print(f"  Current window: {last_window['current_window_start']} to {last_window['current_window_end']}")
            print(f"  Num current events: {last_window['num_current_events']}")
            print(f"  Num future events: {last_window['num_future_events']}")

            # Check current events for outcome codes
            current_events = last_window["current_events"]
            if len(current_events) > 0:
                print(f"\n  Current events in last window:")
                for i, event in enumerate(current_events):
                    code = event.get("code", "N/A")
                    is_outcome = code in outcome_codes
                    marker = " *** OUTCOME CODE ***" if is_outcome else ""
                    print(f"    {i+1}. Code: {code}{marker}")
                    print(f"       Time: {event.get('time', 'N/A')}")
                    if "code_specifics" in event:
                        print(f"       Specifics: {event['code_specifics']}")
                    if "numeric_value" in event:
                        print(f"       Value: {event['numeric_value']}")
            else:
                print("\n  No current events in last window")

            # Check future events for outcome codes
            future_events = last_window["future_events"]
            if len(future_events) > 0:
                print(f"\n  Future events in last window:")
                outcome_in_future = False
                for i, event in enumerate(future_events):
                    code = event.get("code", "N/A")
                    is_outcome = code in outcome_codes
                    if is_outcome:
                        outcome_in_future = True
                        print(f"    {i+1}. Code: {code} *** OUTCOME CODE ***")
                        print(f"       Time: {event.get('time', 'N/A')}")
                        if "code_specifics" in event:
                            print(f"       Specifics: {event['code_specifics']}")

                if not outcome_in_future:
                    print(f"    No outcome codes found in {len(future_events)} future events")
            else:
                print("\n  No future events in last window")


if __name__ == "__main__":
    patient_id = 19904685
    check_patient_outcome_codes(patient_id)
