"""
Explore static patient metadata that remains constant during ICU stay.
This includes demographic information, chronic conditions, and baseline characteristics.
"""
import pandas as pd
import numpy as np

# Load the data
print("Loading data...")
icu_stay_df = pd.read_parquet("data/mimic-demo/icu_stay/data_0.parquet")
icu_stay_df["enter_time"] = pd.to_datetime(icu_stay_df["enter_time"])
icu_stay_df["leave_time"] = pd.to_datetime(icu_stay_df["leave_time"])
icu_stay_df["birth_time"] = pd.to_datetime(icu_stay_df["birth_time"])

events_df = pd.read_parquet("data/mimic-demo/events/data_0.parquet")
events_df["time"] = pd.to_datetime(events_df["time"])

print(f"Total ICU stays: {len(icu_stay_df)}")
print(f"Total events: {len(events_df)}")

# 1. EXPLORE ICU_STAY METADATA
print("\n" + "="*80)
print("1. ICU STAY TABLE METADATA (Static Patient Information)")
print("="*80)

print("\nColumns in ICU stay table:")
for col in icu_stay_df.columns:
    print(f"  - {col}")

# Calculate age at admission
icu_stay_df["age_at_admission"] = (
    (icu_stay_df["enter_time"] - icu_stay_df["birth_time"]).dt.days / 365.25
)

print("\nAge distribution:")
print(f"  Mean: {icu_stay_df['age_at_admission'].mean():.1f} years")
print(f"  Median: {icu_stay_df['age_at_admission'].median():.1f} years")
print(f"  Min: {icu_stay_df['age_at_admission'].min():.1f} years")
print(f"  Max: {icu_stay_df['age_at_admission'].max():.1f} years")
print(f"  Std: {icu_stay_df['age_at_admission'].std():.1f} years")

# Age groups
age_bins = [0, 18, 40, 60, 80, 120]
age_labels = ['<18', '18-40', '40-60', '60-80', '80+']
icu_stay_df['age_group'] = pd.cut(icu_stay_df['age_at_admission'], bins=age_bins, labels=age_labels)
print("\nAge groups:")
for group, count in icu_stay_df['age_group'].value_counts().sort_index().items():
    print(f"  {group}: {count} ({count/len(icu_stay_df)*100:.1f}%)")

# 2. EXPLORE META_* EVENT CODES
print("\n" + "="*80)
print("2. META_* EVENT CODES (Demographic Information)")
print("="*80)

meta_events = events_df[events_df["code"].str.startswith("META_", na=False)]
print(f"\nTotal META_* events: {len(meta_events)}")
print(f"Unique META_* codes:")
for code in sorted(meta_events["code"].unique()):
    count = len(meta_events[meta_events["code"] == code])
    print(f"  {code}: {count} events")

# Analyze each META code
for meta_code in sorted(meta_events["code"].unique()):
    print(f"\n{meta_code}:")
    code_events = meta_events[meta_events["code"] == meta_code]

    # Check code_specifics
    if "code_specifics" in code_events.columns:
        specifics = code_events["code_specifics"].value_counts()
        print(f"  Values (code_specifics):")
        for value, count in specifics.head(10).items():
            print(f"    {value}: {count}")

    # Check text_value
    if "text_value" in code_events.columns:
        text_vals = code_events["text_value"].dropna()
        if len(text_vals) > 0:
            print(f"  Sample text_values: {list(text_vals.head(5))}")

# 3. CHRONIC CONDITIONS (DIAGNOSIS codes from before ICU)
print("\n" + "="*80)
print("3. CHRONIC CONDITIONS (Historical Diagnoses)")
print("="*80)

# Common chronic conditions to look for
chronic_keywords = [
    "diabetes", "hypertension", "heart failure", "chronic kidney",
    "COPD", "asthma", "coronary", "cancer", "cirrhosis", "depression",
    "anxiety", "obesity", "hyperlipidemia", "atrial fibrillation"
]

print("\nSearching for chronic conditions in pre-ICU diagnoses...")

# For each ICU stay, find chronic diagnoses from before admission
chronic_conditions_summary = {keyword: 0 for keyword in chronic_keywords}

for _, icu_stay in icu_stay_df.head(100).iterrows():  # Sample first 100 for speed
    subject_id = icu_stay["subject_id"]
    enter_time = icu_stay["enter_time"]

    # Get diagnoses from before ICU (more than 24h before to exclude acute conditions)
    patient_events = events_df[events_df["subject_id"] == subject_id]
    pre_icu_diagnoses = patient_events[
        (patient_events["code"] == "DIAGNOSIS") &
        (patient_events["time"] < enter_time - pd.Timedelta(hours=24))
    ]

    # Check for chronic conditions
    for _, diag in pre_icu_diagnoses.iterrows():
        diag_text = str(diag.get("code_specifics", "")).lower()
        for keyword in chronic_keywords:
            if keyword.lower() in diag_text:
                chronic_conditions_summary[keyword] += 1
                break

print("\nChronic conditions found (in first 100 patients):")
for condition, count in sorted(chronic_conditions_summary.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"  {condition}: {count} patients")

# 4. PREVIOUS HOSPITALIZATIONS
print("\n" + "="*80)
print("4. PREVIOUS HOSPITALIZATIONS (Readmission History)")
print("="*80)

# Check readmission field in ICU stay table
if "readmission" in icu_stay_df.columns or "readm_time" in icu_stay_df.columns:
    readmissions = icu_stay_df["readm_time"].notna().sum()
    print(f"\nPatients with readmissions: {readmissions} / {len(icu_stay_df)} ({readmissions/len(icu_stay_df)*100:.1f}%)")

    if readmissions > 0:
        print(f"\nReadmission duration statistics:")
        readm_durations = pd.to_numeric(
            icu_stay_df[icu_stay_df["readm_duration_hours"].notna()]["readm_duration_hours"],
            errors='coerce'
        )
        if len(readm_durations.dropna()) > 0:
            print(f"  Mean: {readm_durations.mean():.1f} hours")
            print(f"  Median: {readm_durations.median():.1f} hours")

# Count multiple ICU stays per patient
patient_icu_counts = icu_stay_df.groupby("subject_id").size()
multiple_stays = (patient_icu_counts > 1).sum()
print(f"\nPatients with multiple ICU stays: {multiple_stays} / {icu_stay_df['subject_id'].nunique()}")

# 5. BASELINE LAB VALUES (from before ICU)
print("\n" + "="*80)
print("5. BASELINE LAB VALUES (Pre-ICU Laboratory Data)")
print("="*80)

# Key lab tests that indicate baseline health
baseline_labs = [
    "Creatinine", "Hemoglobin", "White Blood Cells", "Platelet Count",
    "Glucose", "Sodium", "Potassium", "Albumin"
]

print("\nAnalyzing baseline lab values (24-72h before ICU)...")
print("(Sampling first 50 patients for speed)")

baseline_lab_data = {lab: [] for lab in baseline_labs}

for _, icu_stay in icu_stay_df.head(50).iterrows():
    subject_id = icu_stay["subject_id"]
    enter_time = icu_stay["enter_time"]

    # Get lab tests from 24-72h before ICU (baseline, not acute)
    patient_events = events_df[events_df["subject_id"] == subject_id]
    baseline_labs_events = patient_events[
        (patient_events["code"] == "LAB_TEST") &
        (patient_events["time"] < enter_time - pd.Timedelta(hours=24)) &
        (patient_events["time"] >= enter_time - pd.Timedelta(hours=72))
    ]

    for lab_name in baseline_labs:
        lab_values = baseline_labs_events[
            baseline_labs_events["code_specifics"].str.contains(lab_name, case=False, na=False)
        ]["numeric_value"].dropna()

        if len(lab_values) > 0:
            baseline_lab_data[lab_name].append(lab_values.iloc[-1])  # Most recent

print("\nBaseline lab value statistics (when available):")
for lab_name, values in baseline_lab_data.items():
    if len(values) > 0:
        print(f"\n  {lab_name}:")
        print(f"    Patients with data: {len(values)} / 50")
        print(f"    Mean: {np.mean(values):.2f}")
        print(f"    Median: {np.median(values):.2f}")

# 6. SUMMARY OF STATIC METADATA
print("\n" + "="*80)
print("SUMMARY: RECOMMENDED STATIC PATIENT METADATA")
print("="*80)

print("""
Based on the analysis, here are the recommended static metadata fields
that remain constant throughout the ICU stay:

1. DEMOGRAPHIC INFORMATION:
   - Age at admission (calculated from birth_time and enter_time)
   - Gender (META_GENDER)
   - Race/Ethnicity (META_RACE)
   - Language (META_LANGUAGE)
   - Marital status (META_MARTIAL_STATUS)
   - Insurance type (META_INSURANCE)

2. CHRONIC CONDITIONS (from historical diagnoses >24h before ICU):
   - Diabetes
   - Hypertension
   - Heart failure
   - Chronic kidney disease
   - COPD/Asthma
   - Coronary artery disease
   - Cancer history
   - Depression/Anxiety
   - Obesity
   - Hyperlipidemia

3. HOSPITALIZATION HISTORY:
   - Number of previous ICU stays
   - Readmission status
   - Time since last hospitalization

4. BASELINE PHYSIOLOGICAL STATE (from 24-72h before ICU):
   - Baseline creatinine (kidney function)
   - Baseline hemoglobin (anemia status)
   - Baseline white blood cell count (immune status)
   - Baseline platelet count (coagulation status)
   - Baseline glucose (metabolic status)
   - Baseline electrolytes (sodium, potassium)
   - Baseline albumin (nutritional status)

5. ADMISSION CONTEXT:
   - Admission source (ED, transfer, physician referral)
   - Admission mode (ambulance vs walk-in)
   - Primary diagnosis at admission

These metadata fields provide stable patient characteristics that can be
used throughout the ICU stay for risk stratification, decision support,
and outcome prediction.
""")