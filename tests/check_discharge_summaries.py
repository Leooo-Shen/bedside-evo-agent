"""
Check for discharge summaries from previous hospital stays.
These provide comprehensive historical context about the patient.
"""
import pandas as pd
from datetime import timedelta

# Load the data
print("Loading data...")
icu_stay_df = pd.read_parquet("data/mimic-demo/icu_stay/data_0.parquet")
icu_stay_df["enter_time"] = pd.to_datetime(icu_stay_df["enter_time"])

events_df = pd.read_parquet("data/mimic-demo/events/data_0.parquet")
events_df["time"] = pd.to_datetime(events_df["time"])

# Find patients with discharge summaries BEFORE their ICU admission
print("\n" + "="*80)
print("ANALYZING DISCHARGE SUMMARIES BEFORE ICU ADMISSION")
print("="*80)

patients_with_summaries = 0
total_summaries = 0

# Sample a few patients
for idx in range(min(5, len(icu_stay_df))):
    icu_stay = icu_stay_df.iloc[idx]
    subject_id = icu_stay["subject_id"]
    enter_time = icu_stay["enter_time"]

    # Get all events for this patient
    patient_events = events_df[events_df["subject_id"] == subject_id].copy()

    # Find discharge summaries BEFORE current ICU admission
    discharge_summaries = patient_events[
        (patient_events["code"] == "NOTE_DISCHARGESUMMARY") &
        (pd.to_datetime(patient_events["time"]) < enter_time)
    ]

    if len(discharge_summaries) > 0:
        patients_with_summaries += 1
        total_summaries += len(discharge_summaries)

        print(f"\n{'='*80}")
        print(f"Patient {subject_id}, ICU Enter: {enter_time}")
        print(f"Found {len(discharge_summaries)} discharge summary(ies) before ICU")
        print(f"{'='*80}")

        # Show the most recent discharge summary
        most_recent = discharge_summaries.sort_values("time").iloc[-1]
        hours_before = (enter_time - most_recent["time"]).total_seconds() / 3600
        days_before = hours_before / 24

        print(f"\nMost recent discharge summary:")
        print(f"  Time: {most_recent['time']} ({days_before:.1f} days before ICU)")

        if pd.notna(most_recent.get("text_value")):
            text = str(most_recent["text_value"])
            print(f"\n  Content preview (first 1000 characters):")
            print(f"  {'-'*76}")
            print(f"  {text[:1000]}")
            print(f"  {'-'*76}")
            print(f"  [Total length: {len(text)} characters]")

print(f"\n{'='*80}")
print(f"Summary: {patients_with_summaries} out of {min(5, len(icu_stay_df))} patients had discharge summaries")
print(f"Total discharge summaries found: {total_summaries}")
print(f"{'='*80}")
