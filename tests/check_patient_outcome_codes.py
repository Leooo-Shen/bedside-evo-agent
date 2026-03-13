#!/usr/bin/env python3
"""Check outcome codes in trajectory and last Oracle-style current window."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_parser import MIMICDataParser


OUTCOME_CODES = ["LEAVE_HOSPITALIZATION", "NOTE_DISCHARGESUMMARY", "META_DEATH", "LEAVE_ICU"]


def check_patient_outcome_codes(patient_id: int) -> None:
    parser = MIMICDataParser(
        events_path="data/mimic-demo/events/data_0.parquet",
        icu_stay_path="data/mimic-demo/icu_stay/data_0.parquet",
    )

    print("Loading data...")
    parser.load_data()

    patient_stays = parser.icu_stay_df[parser.icu_stay_df["subject_id"] == patient_id]
    if len(patient_stays) == 0:
        print(f"No ICU stays found for patient {patient_id}")
        return

    print(f"\nFound {len(patient_stays)} ICU stay(s) for patient {patient_id}")

    for _, stay in patient_stays.iterrows():
        icu_stay_id = stay["icu_stay_id"]
        print(f"\n{'='*80}")
        print(f"ICU Stay ID: {icu_stay_id}")
        print(f"Enter time: {stay['enter_time']}")
        print(f"Leave time: {stay['leave_time']}")
        print(f"Duration: {stay['icu_duration_hours']:.2f} hours")
        print(f"Survived: {stay['survived']}")

        trajectory = parser.get_patient_trajectory(patient_id, icu_stay_id)
        events_df = pd.DataFrame(trajectory["events"])

        print(f"\nTotal events: {len(events_df)}")
        if len(events_df) > 0 and "code" in events_df.columns:
            outcome_events = events_df[events_df["code"].isin(OUTCOME_CODES)]
            if len(outcome_events) > 0:
                print(f"\n*** Found {len(outcome_events)} outcome code event(s) in trajectory ***")
                for _, event in outcome_events.iterrows():
                    print(f"  - Code: {event['code']}")
                    print(f"    Time: {event.get('time', 'N/A')}")
                    print(f"    Code specifics: {event.get('code_specifics', 'N/A')}")
            else:
                print("\nNo outcome code events found in all trajectory events")

        print(f"\n{'='*80}")
        print("Creating windows...")
        windows = parser.create_time_windows(
            trajectory,
            current_window_hours=0.5,
            window_step_hours=0.5,
            include_pre_icu_data=True,
            use_first_n_hours_after_icu=None,
        )

        print(f"Created {len(windows)} windows")
        if not windows:
            continue

        last_window = windows[-1]
        print(f"\n{'='*80}")
        print(f"LAST WINDOW (Window {len(windows)-1}):")
        print(f"  Hours since admission: {last_window['hours_since_admission']:.2f}")
        print(f"  Current window: {last_window['current_window_start']} to {last_window['current_window_end']}")
        print(f"  Num current events: {last_window['num_current_events']}")

        current_events = last_window["current_events"]
        if not current_events:
            print("\nNo current events in last window")
            continue

        print("\nCurrent events in last window:")
        found_outcome = False
        for i, event in enumerate(current_events, start=1):
            code = event.get("code", "N/A")
            is_outcome = code in OUTCOME_CODES
            if is_outcome:
                found_outcome = True
            marker = " *** OUTCOME CODE ***" if is_outcome else ""
            print(f"  {i}. Code: {code}{marker}")
            print(f"     Time: {event.get('time', 'N/A')}")
            if "code_specifics" in event:
                print(f"     Specifics: {event['code_specifics']}")

        if not found_outcome:
            print("\nNo outcome code appears in last current window events.")


if __name__ == "__main__":
    check_patient_outcome_codes(19904685)
