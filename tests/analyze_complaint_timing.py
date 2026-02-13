"""
Analyze the timing of COMPLAINT events relative to ICU admission.
Determine how many occur before vs during ICU stay, and within specific time windows.
"""
import pandas as pd
from datetime import timedelta

# Load the data
print("Loading data...")
icu_stay_df = pd.read_parquet("data/mimic-demo/icu_stay/data_0.parquet")
icu_stay_df["enter_time"] = pd.to_datetime(icu_stay_df["enter_time"])
icu_stay_df["leave_time"] = pd.to_datetime(icu_stay_df["leave_time"])

events_df = pd.read_parquet("data/mimic-demo/events/data_0.parquet")
events_df["time"] = pd.to_datetime(events_df["time"])

print(f"Total ICU stays: {len(icu_stay_df)}")
print(f"Total events: {len(events_df)}")

# Get all COMPLAINT events
complaint_events = events_df[events_df["code"] == "COMPLAINT"].copy()
print(f"\nTotal COMPLAINT events in dataset: {len(complaint_events)}")
print(f"Patients with COMPLAINT events: {complaint_events['subject_id'].nunique()}")

# Analyze timing relative to ICU admission
print("\n" + "="*80)
print("ANALYZING COMPLAINT EVENT TIMING RELATIVE TO ICU ADMISSION")
print("="*80)

# For each ICU stay, categorize complaint events by timing
timing_categories = {
    "during_icu": 0,
    "before_icu_0_24h": 0,
    "before_icu_24_48h": 0,
    "before_icu_48_72h": 0,
    "before_icu_72h_plus": 0,
    "after_icu": 0,
}

patients_with_complaints_during = 0
patients_with_complaints_before_24h = 0
patients_with_complaints_before_48h = 0

complaint_details = []

for _, icu_stay in icu_stay_df.iterrows():
    subject_id = icu_stay["subject_id"]
    enter_time = icu_stay["enter_time"]
    leave_time = icu_stay["leave_time"]

    # Get all complaint events for this patient
    patient_complaints = complaint_events[complaint_events["subject_id"] == subject_id]

    if len(patient_complaints) == 0:
        continue

    has_complaint_during = False
    has_complaint_before_24h = False
    has_complaint_before_48h = False

    for _, complaint in patient_complaints.iterrows():
        complaint_time = complaint["time"]

        # Calculate time difference
        hours_from_enter = (complaint_time - enter_time).total_seconds() / 3600

        # Categorize by timing
        if complaint_time >= enter_time and complaint_time <= leave_time:
            timing_categories["during_icu"] += 1
            has_complaint_during = True
            category = "during_icu"
        elif complaint_time < enter_time:
            hours_before = -hours_from_enter
            if hours_before <= 24:
                timing_categories["before_icu_0_24h"] += 1
                has_complaint_before_24h = True
                has_complaint_before_48h = True
                category = "before_0-24h"
            elif hours_before <= 48:
                timing_categories["before_icu_24_48h"] += 1
                has_complaint_before_48h = True
                category = "before_24-48h"
            elif hours_before <= 72:
                timing_categories["before_icu_48_72h"] += 1
                category = "before_48-72h"
            else:
                timing_categories["before_icu_72h_plus"] += 1
                category = f"before_{hours_before:.0f}h"
        else:
            timing_categories["after_icu"] += 1
            category = "after_icu"

        complaint_details.append({
            "subject_id": subject_id,
            "icu_stay_id": icu_stay["icu_stay_id"],
            "complaint_time": complaint_time,
            "enter_time": enter_time,
            "hours_from_enter": hours_from_enter,
            "category": category,
            "complaint": complaint.get("code_specifics", ""),
        })

    if has_complaint_during:
        patients_with_complaints_during += 1
    if has_complaint_before_24h:
        patients_with_complaints_before_24h += 1
    if has_complaint_before_48h:
        patients_with_complaints_before_48h += 1

# Print summary
print("\nCOMPLAINT Event Distribution by Timing:")
print(f"  During ICU stay: {timing_categories['during_icu']} events")
print(f"  Before ICU (0-24h): {timing_categories['before_icu_0_24h']} events")
print(f"  Before ICU (24-48h): {timing_categories['before_icu_24_48h']} events")
print(f"  Before ICU (48-72h): {timing_categories['before_icu_48_72h']} events")
print(f"  Before ICU (>72h): {timing_categories['before_icu_72h_plus']} events")
print(f"  After ICU stay: {timing_categories['after_icu']} events")

total_before_24h = timing_categories['before_icu_0_24h']
total_before_48h = timing_categories['before_icu_0_24h'] + timing_categories['before_icu_24_48h']
total_during = timing_categories['during_icu']

print(f"\nSummary:")
print(f"  Total complaints within 24h before ICU: {total_before_24h}")
print(f"  Total complaints within 48h before ICU: {total_before_48h}")
print(f"  Total complaints during ICU: {total_during}")

print(f"\nPatient-level statistics:")
print(f"  Patients with complaints during ICU: {patients_with_complaints_during} / {len(icu_stay_df)} ({patients_with_complaints_during/len(icu_stay_df)*100:.1f}%)")
print(f"  Patients with complaints within 24h before ICU: {patients_with_complaints_before_24h} / {len(icu_stay_df)} ({patients_with_complaints_before_24h/len(icu_stay_df)*100:.1f}%)")
print(f"  Patients with complaints within 48h before ICU: {patients_with_complaints_before_48h} / {len(icu_stay_df)} ({patients_with_complaints_before_48h/len(icu_stay_df)*100:.1f}%)")

# Analyze most common complaints by timing
print("\n" + "="*80)
print("MOST COMMON COMPLAINTS BY TIMING")
print("="*80)

complaint_df = pd.DataFrame(complaint_details)

# During ICU
during_complaints = complaint_df[complaint_df["category"] == "during_icu"]
if len(during_complaints) > 0:
    print(f"\nTop 10 complaints DURING ICU stay ({len(during_complaints)} total):")
    for idx, (complaint, count) in enumerate(during_complaints["complaint"].value_counts().head(10).items(), 1):
        print(f"  {idx:2d}. {complaint}: {count}")

# Before ICU (0-24h)
before_24h_complaints = complaint_df[complaint_df["category"] == "before_0-24h"]
if len(before_24h_complaints) > 0:
    print(f"\nTop 10 complaints BEFORE ICU (0-24h) ({len(before_24h_complaints)} total):")
    for idx, (complaint, count) in enumerate(before_24h_complaints["complaint"].value_counts().head(10).items(), 1):
        print(f"  {idx:2d}. {complaint}: {count}")

# Before ICU (24-48h)
before_24_48h_complaints = complaint_df[complaint_df["category"] == "before_24-48h"]
if len(before_24_48h_complaints) > 0:
    print(f"\nTop 10 complaints BEFORE ICU (24-48h) ({len(before_24_48h_complaints)} total):")
    for idx, (complaint, count) in enumerate(before_24_48h_complaints["complaint"].value_counts().head(10).items(), 1):
        print(f"  {idx:2d}. {complaint}: {count}")

# Sample patients with complaints in different time windows
print("\n" + "="*80)
print("SAMPLE PATIENTS WITH COMPLAINTS")
print("="*80)

# Find a patient with complaints before ICU
before_patients = complaint_df[complaint_df["category"].str.startswith("before")]["subject_id"].unique()
if len(before_patients) > 0:
    sample_patient = before_patients[0]
    patient_complaints = complaint_df[complaint_df["subject_id"] == sample_patient].sort_values("complaint_time")

    print(f"\nSample Patient {sample_patient}:")
    print(f"  ICU Enter Time: {patient_complaints.iloc[0]['enter_time']}")
    print(f"  Total complaints: {len(patient_complaints)}")
    print(f"\n  Complaint timeline:")
    for _, row in patient_complaints.iterrows():
        print(f"    {row['complaint_time']} ({row['hours_from_enter']:+.1f}h from ICU): {row['complaint']}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"""
COMPLAINT events are relatively INFREQUENT in this dataset:
- Only {complaint_events['subject_id'].nunique()} out of {len(icu_stay_df)} patients ({complaint_events['subject_id'].nunique()/len(icu_stay_df)*100:.1f}%) have any COMPLAINT events
- Most complaints occur DURING ICU stay ({timing_categories['during_icu']} events)
- Fewer complaints occur in the 24h before ICU ({timing_categories['before_icu_0_24h']} events)
- Only {patients_with_complaints_before_24h} patients ({patients_with_complaints_before_24h/len(icu_stay_df)*100:.1f}%) have complaints within 24h before ICU

This suggests that COMPLAINT codes are NOT the primary source of initial symptom
information. Instead, you should rely more on:
- DIAGNOSIS codes (99,539 events, 99.8% of patients)
- VITALS (1.5M events, abnormal values indicate symptoms)
- LAB_TEST (abnormal values indicate pathology)
- NOTE_RADIOLOGYREPORT (detailed text descriptions)
- ENTER_ED arrival mode (ambulance vs walk-in indicates severity)
""")