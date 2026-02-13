"""
Explore events data to find initial reports/summaries before ICU admission.
"""
import pandas as pd
from datetime import timedelta

# Load the data
print("Loading ICU stay data...")
icu_stay_df = pd.read_parquet("data/mimic-demo/icu_stay/data_0.parquet")
icu_stay_df["enter_time"] = pd.to_datetime(icu_stay_df["enter_time"])
icu_stay_df["leave_time"] = pd.to_datetime(icu_stay_df["leave_time"])

print("Loading events data...")
events_df = pd.read_parquet("data/mimic-demo/events/data_0.parquet")
events_df["time"] = pd.to_datetime(events_df["time"])

print(f"\nTotal ICU stays: {len(icu_stay_df)}")
print(f"Total events: {len(events_df)}")

# Analyze the first few patients
print("\n" + "="*80)
print("ANALYZING PRE-ICU EVENTS FOR SAMPLE PATIENTS")
print("="*80)

sample_patients = icu_stay_df.head(5)

for idx, icu_stay in sample_patients.iterrows():
    subject_id = icu_stay["subject_id"]
    icu_stay_id = icu_stay["icu_stay_id"]
    enter_time = icu_stay["enter_time"]

    print(f"\n{'='*80}")
    print(f"Patient: {subject_id}, ICU Stay: {icu_stay_id}")
    print(f"ICU Enter Time: {enter_time}")
    print(f"{'='*80}")

    # Get all events for this patient
    patient_events = events_df[events_df["subject_id"] == subject_id].copy()
    patient_events = patient_events.sort_values("time")

    # Filter events BEFORE ICU admission
    pre_icu_events = patient_events[patient_events["time"] < enter_time]

    print(f"\nTotal events for patient: {len(patient_events)}")
    print(f"Events BEFORE ICU admission: {len(pre_icu_events)}")

    if len(pre_icu_events) > 0:
        # Show time range of pre-ICU events
        earliest_event = pre_icu_events["time"].min()
        latest_event = pre_icu_events["time"].max()
        hours_before_icu = (enter_time - latest_event).total_seconds() / 3600

        print(f"\nPre-ICU event time range:")
        print(f"  Earliest: {earliest_event}")
        print(f"  Latest: {latest_event} ({hours_before_icu:.2f} hours before ICU)")

        # Analyze event types (codes)
        print(f"\nPre-ICU Event Types (codes):")
        code_counts = pre_icu_events["code"].value_counts()
        for code, count in code_counts.head(20).items():
            print(f"  {code}: {count}")

        # Look for NOTE events (reports/summaries)
        note_events = pre_icu_events[pre_icu_events["code"].str.startswith("NOTE_", na=False)]
        if len(note_events) > 0:
            print(f"\n*** FOUND {len(note_events)} NOTE EVENTS BEFORE ICU ***")
            for _, event in note_events.iterrows():
                hours_before = (enter_time - event["time"]).total_seconds() / 3600
                print(f"\n  Code: {event['code']}")
                print(f"  Time: {event['time']} ({hours_before:.2f} hours before ICU)")
                print(f"  Code Specifics: {event.get('code_specifics', 'N/A')}")
                if pd.notna(event.get("text_value")):
                    text_preview = str(event["text_value"])[:200]
                    print(f"  Text Preview: {text_preview}...")

        # Look for diagnosis/admission events
        diag_events = pre_icu_events[
            pre_icu_events["code"].str.contains("DIAG|ADMIT|ENTER", case=False, na=False)
        ]
        if len(diag_events) > 0:
            print(f"\n*** FOUND {len(diag_events)} DIAGNOSIS/ADMISSION EVENTS ***")
            for _, event in diag_events.iterrows():
                hours_before = (enter_time - event["time"]).total_seconds() / 3600
                print(f"\n  Code: {event['code']}")
                print(f"  Time: {event['time']} ({hours_before:.2f} hours before ICU)")
                print(f"  Code Specifics: {event.get('code_specifics', 'N/A')}")
    else:
        print("\nNo events found before ICU admission for this patient.")

# Summary statistics across all patients
print("\n" + "="*80)
print("SUMMARY STATISTICS ACROSS ALL PATIENTS")
print("="*80)

# For each ICU stay, count pre-ICU events
pre_icu_counts = []
note_counts = []
diag_counts = []

for _, icu_stay in icu_stay_df.iterrows():
    subject_id = icu_stay["subject_id"]
    enter_time = icu_stay["enter_time"]

    patient_events = events_df[events_df["subject_id"] == subject_id]
    pre_icu_events = patient_events[patient_events["time"] < enter_time]

    pre_icu_counts.append(len(pre_icu_events))

    note_events = pre_icu_events[pre_icu_events["code"].str.startswith("NOTE_", na=False)]
    note_counts.append(len(note_events))

    diag_events = pre_icu_events[
        pre_icu_events["code"].str.contains("DIAG|ADMIT|ENTER", case=False, na=False)
    ]
    diag_counts.append(len(diag_events))

print(f"\nPre-ICU events per patient:")
print(f"  Mean: {pd.Series(pre_icu_counts).mean():.2f}")
print(f"  Median: {pd.Series(pre_icu_counts).median():.2f}")
print(f"  Min: {pd.Series(pre_icu_counts).min()}")
print(f"  Max: {pd.Series(pre_icu_counts).max()}")
print(f"  Patients with pre-ICU events: {sum(1 for c in pre_icu_counts if c > 0)} / {len(pre_icu_counts)}")

print(f"\nNOTE events per patient (before ICU):")
print(f"  Mean: {pd.Series(note_counts).mean():.2f}")
print(f"  Median: {pd.Series(note_counts).median():.2f}")
print(f"  Patients with NOTE events: {sum(1 for c in note_counts if c > 0)} / {len(note_counts)}")

print(f"\nDiagnosis/Admission events per patient (before ICU):")
print(f"  Mean: {pd.Series(diag_counts).mean():.2f}")
print(f"  Median: {pd.Series(diag_counts).median():.2f}")
print(f"  Patients with diagnosis events: {sum(1 for c in diag_counts if c > 0)} / {len(diag_counts)}")

# Analyze all unique event codes in pre-ICU period
print("\n" + "="*80)
print("ALL UNIQUE EVENT CODES IN PRE-ICU PERIOD")
print("="*80)

all_pre_icu_events = []
for _, icu_stay in icu_stay_df.iterrows():
    subject_id = icu_stay["subject_id"]
    enter_time = icu_stay["enter_time"]

    patient_events = events_df[events_df["subject_id"] == subject_id]
    pre_icu_events = patient_events[patient_events["time"] < enter_time]
    all_pre_icu_events.append(pre_icu_events)

all_pre_icu_df = pd.concat(all_pre_icu_events, ignore_index=True)
print(f"\nTotal pre-ICU events across all patients: {len(all_pre_icu_df)}")

print(f"\nTop 30 most common event codes in pre-ICU period:")
code_counts = all_pre_icu_df["code"].value_counts()
for code, count in code_counts.head(30).items():
    print(f"  {code}: {count}")

print("\n" + "="*80)
print("DONE")
print("="*80)
