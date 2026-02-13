"""
Analyze initial patient status information available before ICU admission.
Focus on the most recent events that describe the patient's condition.
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

# Analyze a specific patient in detail
patient_idx = 0
icu_stay = icu_stay_df.iloc[patient_idx]
subject_id = icu_stay["subject_id"]
icu_stay_id = icu_stay["icu_stay_id"]
enter_time = icu_stay["enter_time"]

print(f"\n{'='*80}")
print(f"DETAILED ANALYSIS: Patient {subject_id}, ICU Stay {icu_stay_id}")
print(f"ICU Enter Time: {enter_time}")
print(f"{'='*80}")

# Get all events for this patient
patient_events = events_df[events_df["subject_id"] == subject_id].copy()
patient_events = patient_events.sort_values("time")

# Filter events in the 24 hours BEFORE ICU admission
time_window_hours = 24
pre_icu_window_start = enter_time - timedelta(hours=time_window_hours)
recent_pre_icu = patient_events[
    (patient_events["time"] >= pre_icu_window_start) &
    (patient_events["time"] < enter_time)
]

print(f"\n{'='*80}")
print(f"EVENTS IN THE {time_window_hours} HOURS BEFORE ICU ADMISSION")
print(f"Time window: {pre_icu_window_start} to {enter_time}")
print(f"Total events: {len(recent_pre_icu)}")
print(f"{'='*80}")

# Categorize events by type
print("\nEvent types in this window:")
for code, count in recent_pre_icu["code"].value_counts().items():
    print(f"  {code}: {count}")

# 1. ADMISSION/ENTRY EVENTS
print(f"\n{'='*80}")
print("1. ADMISSION/ENTRY EVENTS (How patient arrived)")
print(f"{'='*80}")
admission_events = recent_pre_icu[
    recent_pre_icu["code"].isin(["ENTER_ED", "ENTER_HOSPITALIZATION", "TRANSFER"])
]
for _, event in admission_events.iterrows():
    hours_before = (enter_time - event["time"]).total_seconds() / 3600
    print(f"\n  Time: {event['time']} ({hours_before:.2f}h before ICU)")
    print(f"  Code: {event['code']}")
    print(f"  Details: {event.get('code_specifics', 'N/A')}")

# 2. CHIEF COMPLAINTS
print(f"\n{'='*80}")
print("2. CHIEF COMPLAINTS (Initial symptoms)")
print(f"{'='*80}")
complaint_events = recent_pre_icu[recent_pre_icu["code"] == "COMPLAINT"]
if len(complaint_events) > 0:
    for _, event in complaint_events.iterrows():
        hours_before = (enter_time - event["time"]).total_seconds() / 3600
        print(f"\n  Time: {event['time']} ({hours_before:.2f}h before ICU)")
        print(f"  Complaint: {event.get('code_specifics', 'N/A')}")
else:
    print("\n  No chief complaints recorded in this window")

# 3. DIAGNOSIS CODES
print(f"\n{'='*80}")
print("3. DIAGNOSIS CODES (Identified conditions)")
print(f"{'='*80}")
diagnosis_events = recent_pre_icu[recent_pre_icu["code"] == "DIAGNOSIS"]
if len(diagnosis_events) > 0:
    for _, event in diagnosis_events.iterrows():
        hours_before = (enter_time - event["time"]).total_seconds() / 3600
        print(f"\n  Time: {event['time']} ({hours_before:.2f}h before ICU)")
        print(f"  Diagnosis: {event.get('code_specifics', 'N/A')}")
else:
    print("\n  No diagnosis codes in this window")

# 4. RADIOLOGY REPORTS
print(f"\n{'='*80}")
print("4. RADIOLOGY REPORTS (Imaging findings)")
print(f"{'='*80}")
radiology_events = recent_pre_icu[recent_pre_icu["code"] == "NOTE_RADIOLOGYREPORT"]
if len(radiology_events) > 0:
    for idx, (_, event) in enumerate(radiology_events.iterrows()):
        hours_before = (enter_time - event["time"]).total_seconds() / 3600
        print(f"\n  Report #{idx+1}")
        print(f"  Time: {event['time']} ({hours_before:.2f}h before ICU)")
        if pd.notna(event.get("text_value")):
            text = str(event["text_value"])
            # Show first 500 characters
            print(f"  Content: {text[:500]}...")
            print(f"  [Total length: {len(text)} characters]")
else:
    print("\n  No radiology reports in this window")

# 5. VITAL SIGNS (most recent)
print(f"\n{'='*80}")
print("5. VITAL SIGNS (Most recent before ICU)")
print(f"{'='*80}")
vitals_events = recent_pre_icu[recent_pre_icu["code"] == "VITALS"]
if len(vitals_events) > 0:
    # Show last 5 vital measurements
    for _, event in vitals_events.tail(5).iterrows():
        hours_before = (enter_time - event["time"]).total_seconds() / 3600
        print(f"\n  Time: {event['time']} ({hours_before:.2f}h before ICU)")
        print(f"  Measurement: {event.get('code_specifics', 'N/A')}")
        if pd.notna(event.get("numeric_value")):
            print(f"  Value: {event['numeric_value']} {event.get('text_value', '')}")
else:
    print("\n  No vital signs in this window")

# 6. LAB TESTS (most recent)
print(f"\n{'='*80}")
print("6. LAB TESTS (Most recent before ICU)")
print(f"{'='*80}")
lab_events = recent_pre_icu[recent_pre_icu["code"] == "LAB_TEST"]
if len(lab_events) > 0:
    print(f"\n  Total lab tests: {len(lab_events)}")
    # Show last 10 lab tests
    print(f"  Showing last 10:")
    for _, event in lab_events.tail(10).iterrows():
        hours_before = (enter_time - event["time"]).total_seconds() / 3600
        print(f"\n    Time: {event['time']} ({hours_before:.2f}h before ICU)")
        print(f"    Test: {event.get('code_specifics', 'N/A')}")
        if pd.notna(event.get("numeric_value")):
            print(f"    Value: {event['numeric_value']} {event.get('text_value', '')}")
else:
    print("\n  No lab tests in this window")

# 7. PROCEDURES
print(f"\n{'='*80}")
print("7. PROCEDURES (Interventions before ICU)")
print(f"{'='*80}")
procedure_events = recent_pre_icu[recent_pre_icu["code"] == "PROCEDURE"]
if len(procedure_events) > 0:
    for _, event in procedure_events.iterrows():
        hours_before = (enter_time - event["time"]).total_seconds() / 3600
        print(f"\n  Time: {event['time']} ({hours_before:.2f}h before ICU)")
        print(f"  Procedure: {event.get('code_specifics', 'N/A')}")
else:
    print("\n  No procedures in this window")

# 8. MEDICATIONS
print(f"\n{'='*80}")
print("8. MEDICATIONS (Drugs started before ICU)")
print(f"{'='*80}")
drug_events = recent_pre_icu[recent_pre_icu["code"].isin(["DRUG_START", "DRUG_PRESCRIPTION"])]
if len(drug_events) > 0:
    print(f"\n  Total medications: {len(drug_events)}")
    print(f"  Showing last 10:")
    for _, event in drug_events.tail(10).iterrows():
        hours_before = (enter_time - event["time"]).total_seconds() / 3600
        print(f"\n    Time: {event['time']} ({hours_before:.2f}h before ICU)")
        print(f"    Drug: {event.get('code_specifics', 'N/A')}")
        if pd.notna(event.get("text_value")):
            print(f"    Details: {event.get('text_value', 'N/A')}")
else:
    print("\n  No medications in this window")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"""
Based on this analysis, the following information is typically available
BEFORE a patient enters the ICU:

1. **Admission pathway**: How the patient arrived (ED, transfer, etc.)
2. **Chief complaints**: Initial symptoms reported by patient
3. **Diagnosis codes**: Identified medical conditions
4. **Radiology reports**: Detailed imaging findings with text descriptions
5. **Vital signs**: Blood pressure, heart rate, temperature, etc.
6. **Lab tests**: Blood work, chemistry panels, etc.
7. **Procedures**: Any interventions performed
8. **Medications**: Drugs administered before ICU

This information provides a comprehensive picture of the patient's initial
disease/symptoms/status before ICU admission.
""")
