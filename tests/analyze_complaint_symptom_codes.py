"""
Analyze complaint and symptom codes in the MIMIC-demo dataset.
Focus on identifying the most useful codes for understanding patient presentation.
"""
import pandas as pd
from collections import Counter

# Load the data
print("Loading data...")
icu_stay_df = pd.read_parquet("data/mimic-demo/icu_stay/data_0.parquet")
icu_stay_df["enter_time"] = pd.to_datetime(icu_stay_df["enter_time"])

events_df = pd.read_parquet("data/mimic-demo/events/data_0.parquet")
events_df["time"] = pd.to_datetime(events_df["time"])

print(f"Total ICU stays: {len(icu_stay_df)}")
print(f"Total events: {len(events_df)}")

# Get all pre-ICU events
print("\nCollecting pre-ICU events...")
all_pre_icu_events = []
for _, icu_stay in icu_stay_df.iterrows():
    subject_id = icu_stay["subject_id"]
    enter_time = icu_stay["enter_time"]

    patient_events = events_df[events_df["subject_id"] == subject_id]
    pre_icu_events = patient_events[patient_events["time"] < enter_time]
    all_pre_icu_events.append(pre_icu_events)

pre_icu_df = pd.concat(all_pre_icu_events, ignore_index=True)
print(f"Total pre-ICU events: {len(pre_icu_df)}")

# 1. ANALYZE COMPLAINT CODES
print("\n" + "="*80)
print("1. CHIEF COMPLAINT CODES")
print("="*80)

complaint_events = pre_icu_df[pre_icu_df["code"] == "COMPLAINT"]
print(f"\nTotal COMPLAINT events: {len(complaint_events)}")
print(f"Patients with complaints: {complaint_events['subject_id'].nunique()}")

if len(complaint_events) > 0:
    print(f"\nTop 30 most common complaints (code_specifics):")
    complaint_counts = complaint_events["code_specifics"].value_counts()
    for idx, (complaint, count) in enumerate(complaint_counts.head(30).items(), 1):
        pct = (count / len(complaint_events)) * 100
        print(f"  {idx:2d}. {complaint}: {count} ({pct:.1f}%)")

# 2. ANALYZE TRIAGE ACUITY
print("\n" + "="*80)
print("2. TRIAGE ACUITY CODES")
print("="*80)

triage_events = pre_icu_df[pre_icu_df["code"] == "TRIAGE_ACUITY"]
print(f"\nTotal TRIAGE_ACUITY events: {len(triage_events)}")

if len(triage_events) > 0:
    print(f"\nTriage acuity levels (code_specifics):")
    triage_counts = triage_events["code_specifics"].value_counts()
    for acuity, count in triage_counts.items():
        pct = (count / len(triage_events)) * 100
        print(f"  {acuity}: {count} ({pct:.1f}%)")

# 3. ANALYZE DIAGNOSIS CODES
print("\n" + "="*80)
print("3. DIAGNOSIS CODES (Most Common)")
print("="*80)

diagnosis_events = pre_icu_df[pre_icu_df["code"] == "DIAGNOSIS"]
print(f"\nTotal DIAGNOSIS events: {len(diagnosis_events)}")
print(f"Unique diagnoses: {diagnosis_events['code_specifics'].nunique()}")

if len(diagnosis_events) > 0:
    print(f"\nTop 50 most common diagnoses (code_specifics):")
    diagnosis_counts = diagnosis_events["code_specifics"].value_counts()
    for idx, (diagnosis, count) in enumerate(diagnosis_counts.head(50).items(), 1):
        pct = (count / len(diagnosis_events)) * 100
        print(f"  {idx:2d}. {diagnosis}: {count} ({pct:.1f}%)")

# 4. ANALYZE ADMISSION EVENTS
print("\n" + "="*80)
print("4. ADMISSION/ENTRY EVENTS")
print("="*80)

# ENTER_ED
enter_ed_events = pre_icu_df[pre_icu_df["code"] == "ENTER_ED"]
print(f"\nENTER_ED events: {len(enter_ed_events)}")
if len(enter_ed_events) > 0:
    print(f"  Arrival modes (code_specifics):")
    for mode, count in enter_ed_events["code_specifics"].value_counts().items():
        print(f"    {mode}: {count}")

# ENTER_HOSPITALIZATION
enter_hosp_events = pre_icu_df[pre_icu_df["code"] == "ENTER_HOSPITALIZATION"]
print(f"\nENTER_HOSPITALIZATION events: {len(enter_hosp_events)}")
if len(enter_hosp_events) > 0:
    print(f"  Admission sources (code_specifics):")
    for source, count in enter_hosp_events["code_specifics"].value_counts().items():
        print(f"    {source}: {count}")

# 5. ANALYZE SYMPTOM-RELATED VITALS
print("\n" + "="*80)
print("5. VITAL SIGNS (Symptom Indicators)")
print("="*80)

vitals_events = pre_icu_df[pre_icu_df["code"] == "VITALS"]
print(f"\nTotal VITALS events: {len(vitals_events)}")

if len(vitals_events) > 0:
    print(f"\nTop 30 vital sign types (code_specifics):")
    vitals_counts = vitals_events["code_specifics"].value_counts()
    for idx, (vital, count) in enumerate(vitals_counts.head(30).items(), 1):
        pct = (count / len(vitals_events)) * 100
        print(f"  {idx:2d}. {vital}: {count} ({pct:.1f}%)")

# 6. ANALYZE PROCEDURES (may indicate symptoms/conditions)
print("\n" + "="*80)
print("6. PROCEDURES (May Indicate Symptoms/Conditions)")
print("="*80)

procedure_events = pre_icu_df[pre_icu_df["code"] == "PROCEDURE"]
print(f"\nTotal PROCEDURE events: {len(procedure_events)}")

if len(procedure_events) > 0:
    print(f"\nTop 30 procedures (code_specifics):")
    procedure_counts = procedure_events["code_specifics"].value_counts()
    for idx, (procedure, count) in enumerate(procedure_counts.head(30).items(), 1):
        pct = (count / len(procedure_events)) * 100
        print(f"  {idx:2d}. {procedure}: {count} ({pct:.1f}%)")

# 7. ANALYZE ECG EVENTS
print("\n" + "="*80)
print("7. ECG EVENTS (Cardiac Symptoms)")
print("="*80)

ecg_events = pre_icu_df[pre_icu_df["code"] == "ECG"]
print(f"\nTotal ECG events: {len(ecg_events)}")

if len(ecg_events) > 0:
    print(f"\nECG findings (code_specifics):")
    ecg_counts = ecg_events["code_specifics"].value_counts()
    for idx, (finding, count) in enumerate(ecg_counts.head(20).items(), 1):
        print(f"  {idx:2d}. {finding}: {count}")

# 8. SAMPLE PATIENT JOURNEY
print("\n" + "="*80)
print("8. SAMPLE PATIENT JOURNEY (First Patient with Complaints)")
print("="*80)

# Find a patient with complaint events
patients_with_complaints = complaint_events["subject_id"].unique()
if len(patients_with_complaints) > 0:
    sample_patient = patients_with_complaints[0]

    # Get their ICU stay
    patient_icu = icu_stay_df[icu_stay_df["subject_id"] == sample_patient].iloc[0]
    enter_time = patient_icu["enter_time"]

    print(f"\nPatient ID: {sample_patient}")
    print(f"ICU Enter Time: {enter_time}")

    # Get events in 24h before ICU
    patient_events = events_df[events_df["subject_id"] == sample_patient]
    recent_events = patient_events[
        (patient_events["time"] < enter_time) &
        (patient_events["time"] >= enter_time - pd.Timedelta(hours=24))
    ].sort_values("time")

    print(f"\nEvents in 24h before ICU (chronological order):")
    print(f"{'Time':<20} {'Code':<25} {'Details':<60}")
    print("-" * 105)

    for _, event in recent_events.iterrows():
        time_str = str(event["time"])[:19]
        code = str(event["code"])[:24]
        details = str(event.get("code_specifics", ""))[:59]
        print(f"{time_str:<20} {code:<25} {details:<60}")

print("\n" + "="*80)
print("SUMMARY: KEY CODES FOR PATIENT COMPLAINTS/SYMPTOMS")
print("="*80)
print("""
The most useful codes for understanding patient complaints and symptoms:

1. COMPLAINT - Chief complaints reported by patient
2. DIAGNOSIS - Identified medical conditions
3. TRIAGE_ACUITY - Severity/urgency assessment
4. ENTER_ED - Emergency department arrival (indicates acute presentation)
5. VITALS - Abnormal vital signs indicate physiological distress
6. PROCEDURE - Interventions suggest underlying conditions
7. ECG - Cardiac findings
8. NOTE_RADIOLOGYREPORT - Imaging findings (text descriptions)
9. LAB_TEST - Abnormal lab values indicate pathology

These codes, especially when combined, provide a comprehensive picture
of the patient's initial presentation and symptoms.
""")
