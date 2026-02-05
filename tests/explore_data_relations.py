"""
Script to explore data relationships in the MIMIC-demo dataset.

This script investigates:
1. Whether enter_time and leave_time match icu_duration_hours
2. Whether patients have multiple NOTE_DISCHARGESUMMARY events
3. Whether there's one NOTE_DISCHARGESUMMARY per ICU stay or multiple
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from config.config import load_config


def explore_icu_duration(icu_stay_df):
    """Check if enter_time and leave_time match icu_duration_hours."""
    print("\n" + "=" * 80)
    print("1. EXPLORING ICU DURATION CONSISTENCY")
    print("=" * 80)

    # Convert to datetime if not already
    icu_stay_df["enter_time"] = pd.to_datetime(icu_stay_df["enter_time"])
    icu_stay_df["leave_time"] = pd.to_datetime(icu_stay_df["leave_time"])

    # Calculate duration from timestamps
    icu_stay_df["calculated_duration_hours"] = (
        icu_stay_df["leave_time"] - icu_stay_df["enter_time"]
    ).dt.total_seconds() / 3600

    # Calculate difference
    icu_stay_df["duration_diff"] = (
        icu_stay_df["calculated_duration_hours"] - icu_stay_df["icu_duration_hours"]
    )

    print(f"\nTotal ICU stays: {len(icu_stay_df)}")
    print(f"\nDuration consistency check:")
    print(f"  Mean difference: {icu_stay_df['duration_diff'].mean():.6f} hours")
    print(f"  Std difference: {icu_stay_df['duration_diff'].std():.6f} hours")
    print(f"  Max difference: {icu_stay_df['duration_diff'].max():.6f} hours")
    print(f"  Min difference: {icu_stay_df['duration_diff'].min():.6f} hours")

    # Check if differences are negligible (< 1 minute)
    negligible_diff = icu_stay_df["duration_diff"].abs() < (1 / 60)
    print(f"\nStays with negligible difference (< 1 min): {negligible_diff.sum()} / {len(icu_stay_df)}")
    print(f"Percentage: {100 * negligible_diff.sum() / len(icu_stay_df):.2f}%")

    # Show examples with large differences
    large_diff = icu_stay_df[icu_stay_df["duration_diff"].abs() > 0.1]
    if len(large_diff) > 0:
        print(f"\nExamples with difference > 0.1 hours (6 minutes):")
        print(large_diff[["subject_id", "icu_stay_id", "enter_time", "leave_time",
                          "icu_duration_hours", "calculated_duration_hours", "duration_diff"]].head(10))
    else:
        print("\nNo stays with difference > 0.1 hours. Duration data is consistent!")

    return icu_stay_df


def explore_discharge_summaries(events_df, icu_stay_df):
    """Check NOTE_DISCHARGESUMMARY events per patient and per ICU stay."""
    print("\n" + "=" * 80)
    print("2. EXPLORING NOTE_DISCHARGESUMMARY EVENTS")
    print("=" * 80)

    # Filter for discharge summary events
    discharge_events = events_df[events_df["code"] == "NOTE_DISCHARGESUMMARY"].copy()

    print(f"\nTotal NOTE_DISCHARGESUMMARY events: {len(discharge_events)}")
    print(f"Total patients in events: {events_df['subject_id'].nunique()}")
    print(f"Total ICU stays: {len(icu_stay_df)}")

    # Count discharge summaries per patient
    discharge_per_patient = discharge_events.groupby("subject_id").size()
    print(f"\nDischarge summaries per patient:")
    print(f"  Mean: {discharge_per_patient.mean():.2f}")
    print(f"  Median: {discharge_per_patient.median():.0f}")
    print(f"  Min: {discharge_per_patient.min():.0f}")
    print(f"  Max: {discharge_per_patient.max():.0f}")

    print(f"\nDistribution of discharge summaries per patient:")
    print(discharge_per_patient.value_counts().sort_index())

    # Patients with multiple discharge summaries
    multiple_discharge = discharge_per_patient[discharge_per_patient > 1]
    print(f"\nPatients with multiple discharge summaries: {len(multiple_discharge)} / {len(discharge_per_patient)}")

    if len(multiple_discharge) > 0:
        print(f"\nExamples of patients with multiple discharge summaries:")
        for patient_id in multiple_discharge.head(5).index:
            patient_discharges = discharge_events[discharge_events["subject_id"] == patient_id]
            print(f"\n  Patient {patient_id}: {len(patient_discharges)} discharge summaries")
            if "time" in patient_discharges.columns:
                patient_discharges["time"] = pd.to_datetime(patient_discharges["time"])
                print(f"    Times: {patient_discharges['time'].tolist()}")

    # Now check discharge summaries per ICU stay
    print("\n" + "-" * 80)
    print("Checking discharge summaries per ICU stay...")
    print("-" * 80)

    # For each ICU stay, count discharge summaries within the event range
    discharge_per_icu_stay = []

    for idx, icu_stay in icu_stay_df.iterrows():
        subject_id = icu_stay["subject_id"]
        icu_stay_id = icu_stay["icu_stay_id"]
        min_event_idx = icu_stay["min_event_idx"]
        max_event_idx = icu_stay["max_event_idx"]
        enter_time = pd.to_datetime(icu_stay["enter_time"])
        leave_time = pd.to_datetime(icu_stay["leave_time"])

        # Get discharge summaries for this ICU stay (by event index)
        icu_discharge_events = discharge_events[
            (discharge_events["subject_id"] == subject_id) &
            (discharge_events.index >= min_event_idx) &
            (discharge_events.index <= max_event_idx)
        ]

        discharge_per_icu_stay.append({
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "enter_time": enter_time,
            "leave_time": leave_time,
            "num_discharge_summaries": len(icu_discharge_events),
            "discharge_times": icu_discharge_events["time"].tolist() if "time" in icu_discharge_events.columns else []
        })

    discharge_icu_df = pd.DataFrame(discharge_per_icu_stay)

    print(f"\nDischarge summaries per ICU stay:")
    print(discharge_icu_df["num_discharge_summaries"].value_counts().sort_index())

    print(f"\nStatistics:")
    print(f"  ICU stays with 0 discharge summaries: {(discharge_icu_df['num_discharge_summaries'] == 0).sum()}")
    print(f"  ICU stays with 1 discharge summary: {(discharge_icu_df['num_discharge_summaries'] == 1).sum()}")
    print(f"  ICU stays with >1 discharge summaries: {(discharge_icu_df['num_discharge_summaries'] > 1).sum()}")

    # Show examples with multiple discharge summaries
    multiple_icu = discharge_icu_df[discharge_icu_df["num_discharge_summaries"] > 1]
    if len(multiple_icu) > 0:
        print(f"\nExamples of ICU stays with multiple discharge summaries:")
        for idx, row in multiple_icu.head(5).iterrows():
            print(f"\n  Patient {row['subject_id']}, ICU Stay {row['icu_stay_id']}:")
            print(f"    Enter: {row['enter_time']}")
            print(f"    Leave: {row['leave_time']}")
            print(f"    Discharge summaries: {row['num_discharge_summaries']}")
            if row['discharge_times']:
                print(f"    Discharge times: {row['discharge_times']}")

    return discharge_icu_df


def explore_discharge_summaries_by_timestamp(events_df, icu_stay_df):
    """Check NOTE_DISCHARGESUMMARY events within ICU stay time windows."""
    print("\n" + "=" * 80)
    print("3. EXPLORING NOTE_DISCHARGESUMMARY WITHIN ICU TIME WINDOWS")
    print("=" * 80)
    print("\nFiltering events by timestamp (enter_time <= event.time <= leave_time)")
    print("This excludes pre-ICU historical data.")

    # Filter for discharge summary events
    discharge_events = events_df[events_df["code"] == "NOTE_DISCHARGESUMMARY"].copy()

    # Ensure time column is datetime
    if "time" in discharge_events.columns:
        discharge_events["time"] = pd.to_datetime(discharge_events["time"])
    else:
        print("\nWARNING: No 'time' column in events data!")
        return None

    # For each ICU stay, count discharge summaries within the time window
    discharge_per_icu_stay_time = []

    for idx, icu_stay in icu_stay_df.iterrows():
        subject_id = icu_stay["subject_id"]
        icu_stay_id = icu_stay["icu_stay_id"]
        enter_time = pd.to_datetime(icu_stay["enter_time"])
        leave_time = pd.to_datetime(icu_stay["leave_time"])

        # Get discharge summaries for this patient within the ICU time window
        icu_discharge_events = discharge_events[
            (discharge_events["subject_id"] == subject_id) &
            (discharge_events["time"] >= enter_time) &
            (discharge_events["time"] <= leave_time)
        ]

        discharge_per_icu_stay_time.append({
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "enter_time": enter_time,
            "leave_time": leave_time,
            "num_discharge_summaries": len(icu_discharge_events),
            "discharge_times": icu_discharge_events["time"].tolist()
        })

    discharge_time_df = pd.DataFrame(discharge_per_icu_stay_time)

    print(f"\nDischarge summaries per ICU stay (filtered by timestamp):")
    print(discharge_time_df["num_discharge_summaries"].value_counts().sort_index())

    print(f"\nStatistics:")
    print(f"  ICU stays with 0 discharge summaries: {(discharge_time_df['num_discharge_summaries'] == 0).sum()}")
    print(f"  ICU stays with 1 discharge summary: {(discharge_time_df['num_discharge_summaries'] == 1).sum()}")
    print(f"  ICU stays with >1 discharge summaries: {(discharge_time_df['num_discharge_summaries'] > 1).sum()}")

    # Calculate percentage
    total_stays = len(discharge_time_df)
    one_discharge = (discharge_time_df['num_discharge_summaries'] == 1).sum()
    print(f"\nPercentage with exactly 1 discharge summary: {100 * one_discharge / total_stays:.2f}%")

    # Show examples with multiple discharge summaries
    multiple_icu = discharge_time_df[discharge_time_df["num_discharge_summaries"] > 1]
    if len(multiple_icu) > 0:
        print(f"\nExamples of ICU stays with multiple discharge summaries (within time window):")
        for idx, row in multiple_icu.head(10).iterrows():
            print(f"\n  Patient {row['subject_id']}, ICU Stay {row['icu_stay_id']}:")
            print(f"    Enter: {row['enter_time']}")
            print(f"    Leave: {row['leave_time']}")
            print(f"    Duration: {(row['leave_time'] - row['enter_time']).total_seconds() / 3600:.2f} hours")
            print(f"    Discharge summaries: {row['num_discharge_summaries']}")
            if row['discharge_times']:
                print(f"    Discharge times: {row['discharge_times']}")
    else:
        print("\nNo ICU stays with multiple discharge summaries within the time window!")

    # Show examples with zero discharge summaries
    zero_icu = discharge_time_df[discharge_time_df["num_discharge_summaries"] == 0]
    if len(zero_icu) > 0:
        print(f"\nExamples of ICU stays with NO discharge summaries (within time window):")
        for idx, row in zero_icu.head(5).iterrows():
            print(f"\n  Patient {row['subject_id']}, ICU Stay {row['icu_stay_id']}:")
            print(f"    Enter: {row['enter_time']}")
            print(f"    Leave: {row['leave_time']}")
            print(f"    Duration: {(row['leave_time'] - row['enter_time']).total_seconds() / 3600:.2f} hours")

    return discharge_time_df


def explore_discharge_summary_timing(events_df, icu_stay_df):
    """Analyze when discharge summaries are generated relative to ICU exit."""
    print("\n" + "=" * 80)
    print("4. ANALYZING DISCHARGE SUMMARY TIMING")
    print("=" * 80)
    print("\nFor ICU stays with NO discharge summary during the stay,")
    print("when is the discharge summary generated after ICU exit?")

    # Filter for discharge summary events
    discharge_events = events_df[events_df["code"] == "NOTE_DISCHARGESUMMARY"].copy()

    # Ensure time column is datetime
    if "time" not in discharge_events.columns:
        print("\nWARNING: No 'time' column in events data!")
        return None

    discharge_events["time"] = pd.to_datetime(discharge_events["time"])

    # For each ICU stay, find the first discharge summary after leave_time
    timing_results = []

    for idx, icu_stay in icu_stay_df.iterrows():
        subject_id = icu_stay["subject_id"]
        icu_stay_id = icu_stay["icu_stay_id"]
        enter_time = pd.to_datetime(icu_stay["enter_time"])
        leave_time = pd.to_datetime(icu_stay["leave_time"])

        # Get discharge summaries for this patient AFTER ICU exit
        post_discharge_summaries = discharge_events[
            (discharge_events["subject_id"] == subject_id) &
            (discharge_events["time"] > leave_time)
        ].sort_values("time")

        # Get the first discharge summary after ICU exit
        if len(post_discharge_summaries) > 0:
            first_discharge = post_discharge_summaries.iloc[0]
            first_discharge_time = first_discharge["time"]
            hours_after_exit = (first_discharge_time - leave_time).total_seconds() / 3600
            days_after_exit = hours_after_exit / 24

            timing_results.append({
                "subject_id": subject_id,
                "icu_stay_id": icu_stay_id,
                "leave_time": leave_time,
                "first_discharge_time": first_discharge_time,
                "hours_after_exit": hours_after_exit,
                "days_after_exit": days_after_exit,
            })
        else:
            # No discharge summary found after this ICU stay
            timing_results.append({
                "subject_id": subject_id,
                "icu_stay_id": icu_stay_id,
                "leave_time": leave_time,
                "first_discharge_time": None,
                "hours_after_exit": None,
                "days_after_exit": None,
            })

    timing_df = pd.DataFrame(timing_results)

    # Filter to only those with discharge summaries
    with_discharge = timing_df[timing_df["hours_after_exit"].notna()]
    without_discharge = timing_df[timing_df["hours_after_exit"].isna()]

    print(f"\nTotal ICU stays analyzed: {len(timing_df)}")
    print(f"ICU stays with discharge summary after exit: {len(with_discharge)}")
    print(f"ICU stays with NO discharge summary found: {len(without_discharge)}")

    if len(with_discharge) > 0:
        print(f"\n--- TIMING STATISTICS (for stays with discharge summaries) ---")
        print(f"Mean time to discharge summary: {with_discharge['hours_after_exit'].mean():.2f} hours ({with_discharge['days_after_exit'].mean():.2f} days)")
        print(f"Median time to discharge summary: {with_discharge['hours_after_exit'].median():.2f} hours ({with_discharge['days_after_exit'].median():.2f} days)")
        print(f"Min time to discharge summary: {with_discharge['hours_after_exit'].min():.2f} hours ({with_discharge['days_after_exit'].min():.2f} days)")
        print(f"Max time to discharge summary: {with_discharge['hours_after_exit'].max():.2f} hours ({with_discharge['days_after_exit'].max():.2f} days)")
        print(f"Std time to discharge summary: {with_discharge['hours_after_exit'].std():.2f} hours ({with_discharge['days_after_exit'].std():.2f} days)")

        # Distribution analysis
        print(f"\n--- DISTRIBUTION ---")
        print(f"Within 24 hours: {(with_discharge['hours_after_exit'] <= 24).sum()} ({100 * (with_discharge['hours_after_exit'] <= 24).sum() / len(with_discharge):.1f}%)")
        print(f"Within 48 hours: {(with_discharge['hours_after_exit'] <= 48).sum()} ({100 * (with_discharge['hours_after_exit'] <= 48).sum() / len(with_discharge):.1f}%)")
        print(f"Within 72 hours (3 days): {(with_discharge['hours_after_exit'] <= 72).sum()} ({100 * (with_discharge['hours_after_exit'] <= 72).sum() / len(with_discharge):.1f}%)")
        print(f"Within 1 week: {(with_discharge['hours_after_exit'] <= 168).sum()} ({100 * (with_discharge['hours_after_exit'] <= 168).sum() / len(with_discharge):.1f}%)")
        print(f"Within 1 month: {(with_discharge['hours_after_exit'] <= 720).sum()} ({100 * (with_discharge['hours_after_exit'] <= 720).sum() / len(with_discharge):.1f}%)")
        print(f"More than 1 month: {(with_discharge['hours_after_exit'] > 720).sum()} ({100 * (with_discharge['hours_after_exit'] > 720).sum() / len(with_discharge):.1f}%)")

        # Show examples
        print(f"\n--- EXAMPLES (sorted by time to discharge summary) ---")
        sorted_discharge = with_discharge.sort_values("hours_after_exit")

        print(f"\nFastest discharge summaries (top 5):")
        for idx, row in sorted_discharge.head(5).iterrows():
            print(f"  Patient {row['subject_id']}, ICU Stay {row['icu_stay_id']}:")
            print(f"    ICU exit: {row['leave_time']}")
            print(f"    Discharge summary: {row['first_discharge_time']}")
            print(f"    Time after exit: {row['hours_after_exit']:.2f} hours ({row['days_after_exit']:.2f} days)")

        print(f"\nSlowest discharge summaries (bottom 5):")
        for idx, row in sorted_discharge.tail(5).iterrows():
            print(f"  Patient {row['subject_id']}, ICU Stay {row['icu_stay_id']}:")
            print(f"    ICU exit: {row['leave_time']}")
            print(f"    Discharge summary: {row['first_discharge_time']}")
            print(f"    Time after exit: {row['hours_after_exit']:.2f} hours ({row['days_after_exit']:.2f} days)")

    return timing_df


def explore_discharge_during_stay(events_df, icu_stay_df):
    """Analyze discharge summaries that occur DURING the ICU stay."""
    print("\n" + "=" * 80)
    print("5. ANALYZING DISCHARGE SUMMARIES DURING ICU STAY")
    print("=" * 80)
    print("\nFor ICU stays with discharge summary DURING the stay,")
    print("when is it generated relative to ICU entry and exit?")

    # Filter for discharge summary events
    discharge_events = events_df[events_df["code"] == "NOTE_DISCHARGESUMMARY"].copy()

    # Ensure time column is datetime
    if "time" not in discharge_events.columns:
        print("\nWARNING: No 'time' column in events data!")
        return None

    discharge_events["time"] = pd.to_datetime(discharge_events["time"])

    # For each ICU stay, find discharge summaries during the stay
    during_stay_results = []

    for idx, icu_stay in icu_stay_df.iterrows():
        subject_id = icu_stay["subject_id"]
        icu_stay_id = icu_stay["icu_stay_id"]
        enter_time = pd.to_datetime(icu_stay["enter_time"])
        leave_time = pd.to_datetime(icu_stay["leave_time"])
        icu_duration_hours = (leave_time - enter_time).total_seconds() / 3600

        # Get discharge summaries DURING the ICU stay
        during_stay_discharges = discharge_events[
            (discharge_events["subject_id"] == subject_id) &
            (discharge_events["time"] >= enter_time) &
            (discharge_events["time"] <= leave_time)
        ].sort_values("time")

        if len(during_stay_discharges) > 0:
            # Analyze each discharge summary during the stay
            for _, discharge in during_stay_discharges.iterrows():
                discharge_time = discharge["time"]
                hours_after_entry = (discharge_time - enter_time).total_seconds() / 3600
                hours_before_exit = (leave_time - discharge_time).total_seconds() / 3600
                percent_through_stay = (hours_after_entry / icu_duration_hours) * 100

                during_stay_results.append({
                    "subject_id": subject_id,
                    "icu_stay_id": icu_stay_id,
                    "enter_time": enter_time,
                    "leave_time": leave_time,
                    "discharge_time": discharge_time,
                    "icu_duration_hours": icu_duration_hours,
                    "hours_after_entry": hours_after_entry,
                    "hours_before_exit": hours_before_exit,
                    "percent_through_stay": percent_through_stay,
                })

    during_stay_df = pd.DataFrame(during_stay_results)

    if len(during_stay_df) == 0:
        print("\nNo discharge summaries found during ICU stays!")
        return None

    print(f"\nTotal discharge summaries during ICU stays: {len(during_stay_df)}")
    print(f"ICU stays with discharge during stay: {during_stay_df['icu_stay_id'].nunique()}")

    print(f"\n--- TIMING RELATIVE TO ICU ENTRY ---")
    print(f"Mean hours after entry: {during_stay_df['hours_after_entry'].mean():.2f} hours")
    print(f"Median hours after entry: {during_stay_df['hours_after_entry'].median():.2f} hours")
    print(f"Min hours after entry: {during_stay_df['hours_after_entry'].min():.2f} hours")
    print(f"Max hours after entry: {during_stay_df['hours_after_entry'].max():.2f} hours")
    print(f"Std hours after entry: {during_stay_df['hours_after_entry'].std():.2f} hours")

    print(f"\n--- TIMING RELATIVE TO ICU EXIT ---")
    print(f"Mean hours before exit: {during_stay_df['hours_before_exit'].mean():.2f} hours")
    print(f"Median hours before exit: {during_stay_df['hours_before_exit'].median():.2f} hours")
    print(f"Min hours before exit: {during_stay_df['hours_before_exit'].min():.2f} hours")
    print(f"Max hours before exit: {during_stay_df['hours_before_exit'].max():.2f} hours")
    print(f"Std hours before exit: {during_stay_df['hours_before_exit'].std():.2f} hours")

    print(f"\n--- POSITION WITHIN ICU STAY ---")
    print(f"Mean percent through stay: {during_stay_df['percent_through_stay'].mean():.1f}%")
    print(f"Median percent through stay: {during_stay_df['percent_through_stay'].median():.1f}%")

    # Distribution analysis
    print(f"\n--- DISTRIBUTION RELATIVE TO ICU ENTRY ---")
    print(f"Within 6 hours of entry: {(during_stay_df['hours_after_entry'] <= 6).sum()} ({100 * (during_stay_df['hours_after_entry'] <= 6).sum() / len(during_stay_df):.1f}%)")
    print(f"Within 12 hours of entry: {(during_stay_df['hours_after_entry'] <= 12).sum()} ({100 * (during_stay_df['hours_after_entry'] <= 12).sum() / len(during_stay_df):.1f}%)")
    print(f"Within 24 hours of entry: {(during_stay_df['hours_after_entry'] <= 24).sum()} ({100 * (during_stay_df['hours_after_entry'] <= 24).sum() / len(during_stay_df):.1f}%)")
    print(f"Within 48 hours of entry: {(during_stay_df['hours_after_entry'] <= 48).sum()} ({100 * (during_stay_df['hours_after_entry'] <= 48).sum() / len(during_stay_df):.1f}%)")

    print(f"\n--- DISTRIBUTION RELATIVE TO ICU EXIT ---")
    print(f"Within 6 hours of exit: {(during_stay_df['hours_before_exit'] <= 6).sum()} ({100 * (during_stay_df['hours_before_exit'] <= 6).sum() / len(during_stay_df):.1f}%)")
    print(f"Within 12 hours of exit: {(during_stay_df['hours_before_exit'] <= 12).sum()} ({100 * (during_stay_df['hours_before_exit'] <= 12).sum() / len(during_stay_df):.1f}%)")
    print(f"Within 24 hours of exit: {(during_stay_df['hours_before_exit'] <= 24).sum()} ({100 * (during_stay_df['hours_before_exit'] <= 24).sum() / len(during_stay_df):.1f}%)")
    print(f"Within 48 hours of exit: {(during_stay_df['hours_before_exit'] <= 48).sum()} ({100 * (during_stay_df['hours_before_exit'] <= 48).sum() / len(during_stay_df):.1f}%)")

    # Show examples
    print(f"\n--- EXAMPLES (sorted by time after entry) ---")
    sorted_by_entry = during_stay_df.sort_values("hours_after_entry")

    print(f"\nEarliest discharge summaries (after ICU entry):")
    for idx, row in sorted_by_entry.head(5).iterrows():
        print(f"  Patient {row['subject_id']}, ICU Stay {row['icu_stay_id']}:")
        print(f"    ICU entry: {row['enter_time']}")
        print(f"    Discharge summary: {row['discharge_time']}")
        print(f"    ICU exit: {row['leave_time']}")
        print(f"    Hours after entry: {row['hours_after_entry']:.2f} ({row['percent_through_stay']:.1f}% through stay)")
        print(f"    Hours before exit: {row['hours_before_exit']:.2f}")
        print(f"    Total ICU duration: {row['icu_duration_hours']:.2f} hours")

    print(f"\nLatest discharge summaries (closest to ICU exit):")
    sorted_by_exit = during_stay_df.sort_values("hours_before_exit")
    for idx, row in sorted_by_exit.head(5).iterrows():
        print(f"  Patient {row['subject_id']}, ICU Stay {row['icu_stay_id']}:")
        print(f"    ICU entry: {row['enter_time']}")
        print(f"    Discharge summary: {row['discharge_time']}")
        print(f"    ICU exit: {row['leave_time']}")
        print(f"    Hours after entry: {row['hours_after_entry']:.2f} ({row['percent_through_stay']:.1f}% through stay)")
        print(f"    Hours before exit: {row['hours_before_exit']:.2f}")
        print(f"    Total ICU duration: {row['icu_duration_hours']:.2f} hours")

    return during_stay_df


def explore_discharge_content(events_df, icu_stay_df):
    """Examine the content of discharge summaries that occur during ICU stays."""
    print("\n" + "=" * 80)
    print("6. EXAMINING DISCHARGE SUMMARY CONTENT")
    print("=" * 80)
    print("\nAnalyzing the actual content of discharge summaries during ICU stays")

    # Filter for discharge summary events
    discharge_events = events_df[events_df["code"] == "NOTE_DISCHARGESUMMARY"].copy()

    # Ensure time column is datetime
    if "time" not in discharge_events.columns:
        print("\nWARNING: No 'time' column in events data!")
        return None

    discharge_events["time"] = pd.to_datetime(discharge_events["time"])

    # Find discharge summaries during ICU stays
    discharge_content_results = []

    for idx, icu_stay in icu_stay_df.iterrows():
        subject_id = icu_stay["subject_id"]
        icu_stay_id = icu_stay["icu_stay_id"]
        enter_time = pd.to_datetime(icu_stay["enter_time"])
        leave_time = pd.to_datetime(icu_stay["leave_time"])

        # Get discharge summaries DURING the ICU stay
        during_stay_discharges = discharge_events[
            (discharge_events["subject_id"] == subject_id) &
            (discharge_events["time"] >= enter_time) &
            (discharge_events["time"] <= leave_time)
        ]

        if len(during_stay_discharges) > 0:
            for _, discharge in during_stay_discharges.iterrows():
                discharge_time = discharge["time"]

                # Extract content fields
                text_value = discharge.get("text_value", None)
                code_specifics = discharge.get("code_specifics", None)

                # Get a preview of the text (first 500 characters)
                text_preview = None
                text_length = 0
                if pd.notna(text_value) and text_value:
                    text_str = str(text_value)
                    text_length = len(text_str)
                    text_preview = text_str[:500] if len(text_str) > 500 else text_str

                discharge_content_results.append({
                    "subject_id": subject_id,
                    "icu_stay_id": icu_stay_id,
                    "enter_time": enter_time,
                    "leave_time": leave_time,
                    "discharge_time": discharge_time,
                    "code_specifics": code_specifics,
                    "text_length": text_length,
                    "text_preview": text_preview,
                    "has_text": pd.notna(text_value) and text_value != "",
                })

    content_df = pd.DataFrame(discharge_content_results)

    if len(content_df) == 0:
        print("\nNo discharge summaries found during ICU stays!")
        return None

    print(f"\nTotal discharge summaries analyzed: {len(content_df)}")
    print(f"\nDischarge summaries with text content: {content_df['has_text'].sum()} ({100 * content_df['has_text'].sum() / len(content_df):.1f}%)")
    print(f"Discharge summaries without text: {(~content_df['has_text']).sum()} ({100 * (~content_df['has_text']).sum() / len(content_df):.1f}%)")

    # Analyze text lengths
    with_text = content_df[content_df["has_text"]]
    if len(with_text) > 0:
        print(f"\n--- TEXT LENGTH STATISTICS ---")
        print(f"Mean text length: {with_text['text_length'].mean():.0f} characters")
        print(f"Median text length: {with_text['text_length'].median():.0f} characters")
        print(f"Min text length: {with_text['text_length'].min():.0f} characters")
        print(f"Max text length: {with_text['text_length'].max():.0f} characters")

    # Show examples with text content
    print(f"\n--- EXAMPLES OF DISCHARGE SUMMARY CONTENT ---")
    examples_with_text = content_df[content_df["has_text"]].head(5)

    for idx, row in examples_with_text.iterrows():
        print(f"\n{'='*80}")
        print(f"Patient {row['subject_id']}, ICU Stay {row['icu_stay_id']}")
        print(f"ICU Entry: {row['enter_time']}")
        print(f"Discharge Summary Time: {row['discharge_time']}")
        print(f"ICU Exit: {row['leave_time']}")
        print(f"Code Specifics: {row['code_specifics']}")
        print(f"Text Length: {row['text_length']} characters")
        print(f"\nText Preview (first 500 chars):")
        print("-" * 80)
        print(row['text_preview'])
        print("-" * 80)

    return content_df


def explore_event_idx_vs_time(events_df, icu_stay_df):
    """Check if enter_time/leave_time correspond to min_event_idx/max_event_idx."""
    print("\n" + "=" * 80)
    print("7. EXPLORING EVENT INDEX VS TIMESTAMP CORRESPONDENCE")
    print("=" * 80)
    print("\nChecking if enter_time corresponds to event at min_event_idx")
    print("and if leave_time corresponds to event at max_event_idx")

    # Ensure time column is datetime
    if "time" not in events_df.columns:
        print("\nWARNING: No 'time' column in events data!")
        return None

    events_df["time"] = pd.to_datetime(events_df["time"])

    # For each ICU stay, check the correspondence
    results = []

    for idx, icu_stay in icu_stay_df.head(20).iterrows():  # Check first 20 for detailed analysis
        subject_id = icu_stay["subject_id"]
        icu_stay_id = icu_stay["icu_stay_id"]
        enter_time = pd.to_datetime(icu_stay["enter_time"])
        leave_time = pd.to_datetime(icu_stay["leave_time"])
        min_event_idx = icu_stay["min_event_idx"]
        max_event_idx = icu_stay["max_event_idx"]

        # Get the event at min_event_idx
        try:
            min_event = events_df.loc[min_event_idx]
            min_event_time = pd.to_datetime(min_event["time"]) if pd.notna(min_event["time"]) else None
            min_event_subject = min_event["subject_id"]
        except:
            min_event_time = None
            min_event_subject = None

        # Get the event at max_event_idx
        try:
            max_event = events_df.loc[max_event_idx]
            max_event_time = pd.to_datetime(max_event["time"]) if pd.notna(max_event["time"]) else None
            max_event_subject = max_event["subject_id"]
        except:
            max_event_time = None
            max_event_subject = None

        # Calculate time differences
        enter_time_diff = None
        leave_time_diff = None
        if min_event_time:
            enter_time_diff = (min_event_time - enter_time).total_seconds() / 3600
        if max_event_time:
            leave_time_diff = (max_event_time - leave_time).total_seconds() / 3600

        results.append({
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "enter_time": enter_time,
            "min_event_idx": min_event_idx,
            "min_event_time": min_event_time,
            "min_event_subject": min_event_subject,
            "enter_time_diff_hours": enter_time_diff,
            "leave_time": leave_time,
            "max_event_idx": max_event_idx,
            "max_event_time": max_event_time,
            "max_event_subject": max_event_subject,
            "leave_time_diff_hours": leave_time_diff,
        })

    results_df = pd.DataFrame(results)

    # Print summary statistics
    print(f"\nAnalyzed {len(results_df)} ICU stays")

    # Check if min_event_idx corresponds to enter_time
    print("\n--- MIN_EVENT_IDX vs ENTER_TIME ---")
    if results_df["enter_time_diff_hours"].notna().any():
        print(f"Mean time difference: {results_df['enter_time_diff_hours'].mean():.2f} hours")
        print(f"Median time difference: {results_df['enter_time_diff_hours'].median():.2f} hours")
        print(f"Min time difference: {results_df['enter_time_diff_hours'].min():.2f} hours")
        print(f"Max time difference: {results_df['enter_time_diff_hours'].max():.2f} hours")

        # Check how many are before vs after enter_time
        before_enter = (results_df["enter_time_diff_hours"] < 0).sum()
        at_enter = (results_df["enter_time_diff_hours"].abs() < 0.01).sum()  # Within ~36 seconds
        after_enter = (results_df["enter_time_diff_hours"] > 0).sum()

        print(f"\nEvents at min_event_idx relative to enter_time:")
        print(f"  Before enter_time: {before_enter}")
        print(f"  At enter_time (±36s): {at_enter}")
        print(f"  After enter_time: {after_enter}")

    # Check if max_event_idx corresponds to leave_time
    print("\n--- MAX_EVENT_IDX vs LEAVE_TIME ---")
    if results_df["leave_time_diff_hours"].notna().any():
        print(f"Mean time difference: {results_df['leave_time_diff_hours'].mean():.2f} hours")
        print(f"Median time difference: {results_df['leave_time_diff_hours'].median():.2f} hours")
        print(f"Min time difference: {results_df['leave_time_diff_hours'].min():.2f} hours")
        print(f"Max time difference: {results_df['leave_time_diff_hours'].max():.2f} hours")

        # Check how many are before vs after leave_time
        before_leave = (results_df["leave_time_diff_hours"] < 0).sum()
        at_leave = (results_df["leave_time_diff_hours"].abs() < 0.01).sum()  # Within ~36 seconds
        after_leave = (results_df["leave_time_diff_hours"] > 0).sum()

        print(f"\nEvents at max_event_idx relative to leave_time:")
        print(f"  Before leave_time: {before_leave}")
        print(f"  At leave_time (±36s): {at_leave}")
        print(f"  After leave_time: {after_leave}")

    # Show detailed examples
    print("\n--- DETAILED EXAMPLES ---")
    for idx, row in results_df.head(5).iterrows():
        print(f"\nPatient {row['subject_id']}, ICU Stay {row['icu_stay_id']}:")
        print(f"  Enter time: {row['enter_time']}")
        print(f"  Min event idx: {row['min_event_idx']}")
        print(f"  Min event time: {row['min_event_time']}")
        print(f"  Min event subject: {row['min_event_subject']}")
        if row['enter_time_diff_hours'] is not None:
            print(f"  Time difference: {row['enter_time_diff_hours']:.2f} hours")
            if row['enter_time_diff_hours'] < 0:
                print(f"    → Event is {abs(row['enter_time_diff_hours']):.2f} hours BEFORE ICU entry")
            elif row['enter_time_diff_hours'] > 0:
                print(f"    → Event is {row['enter_time_diff_hours']:.2f} hours AFTER ICU entry")
            else:
                print(f"    → Event is AT ICU entry time")

        print(f"  Leave time: {row['leave_time']}")
        print(f"  Max event idx: {row['max_event_idx']}")
        print(f"  Max event time: {row['max_event_time']}")
        print(f"  Max event subject: {row['max_event_subject']}")
        if row['leave_time_diff_hours'] is not None:
            print(f"  Time difference: {row['leave_time_diff_hours']:.2f} hours")
            if row['leave_time_diff_hours'] < 0:
                print(f"    → Event is {abs(row['leave_time_diff_hours']):.2f} hours BEFORE ICU exit")
            elif row['leave_time_diff_hours'] > 0:
                print(f"    → Event is {row['leave_time_diff_hours']:.2f} hours AFTER ICU exit")
            else:
                print(f"    → Event is AT ICU exit time")

    return results_df


def main():
    """Main function to run all explorations."""
    print("=" * 80)
    print("MIMIC-DEMO DATA EXPLORATION")
    print("=" * 80)

    # Load configuration
    config = load_config()

    print(f"\nLoading data from:")
    print(f"  Events: {config.events_path}")
    print(f"  ICU stays: {config.icu_stay_path}")

    # Load data
    events_df = pd.read_parquet(config.events_path)
    icu_stay_df = pd.read_parquet(config.icu_stay_path)

    print(f"\nData loaded:")
    print(f"  Events: {len(events_df)} rows")
    print(f"  ICU stays: {len(icu_stay_df)} rows")

    # Run explorations
    icu_stay_df = explore_icu_duration(icu_stay_df)
    discharge_icu_df = explore_discharge_summaries(events_df, icu_stay_df)
    discharge_time_df = explore_discharge_summaries_by_timestamp(events_df, icu_stay_df)
    timing_df = explore_discharge_summary_timing(events_df, icu_stay_df)
    during_stay_df = explore_discharge_during_stay(events_df, icu_stay_df)
    content_df = explore_discharge_content(events_df, icu_stay_df)
    event_idx_df = explore_event_idx_vs_time(events_df, icu_stay_df)

    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

