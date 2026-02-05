"""
Compare patient cohorts between baseline and survival prediction experiments.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_parser import MIMICDataParser
from config.config import get_config
import pandas as pd

seed = 1


def select_balanced_patients(icu_stay_df: pd.DataFrame, n_survived: int = 5, n_died: int = 5) -> pd.DataFrame:
    """
    Select balanced set of patients (equal numbers who survived and died).
    This is the SAME function used in both experiments.
    """
    # Separate patients by outcome
    survived_patients = icu_stay_df[icu_stay_df["survived"] == True]
    died_patients = icu_stay_df[icu_stay_df["survived"] == False]

    # Get actual available counts
    n_survived_available = len(survived_patients)
    n_died_available = len(died_patients)

    # Adjust requested numbers if they exceed available
    n_survived_actual = min(n_survived, n_survived_available)
    n_died_actual = min(n_died, n_died_available)

    # Sample patients
    selected_survived = survived_patients.sample(n=n_survived_actual, random_state=seed)
    selected_died = died_patients.sample(n=n_died_actual, random_state=seed)

    # Combine and shuffle
    balanced_df = pd.concat([selected_survived, selected_died]).sample(frac=1, random_state=seed)

    return balanced_df


def main():
    """Compare patient cohorts."""
    config = get_config()

    print("=" * 80)
    print("COMPARING PATIENT COHORTS")
    print("=" * 80)

    # Load data with current filtering
    print("\n1. Loading MIMIC-demo data with current filtering...")
    parser = MIMICDataParser(events_path=config.events_path, icu_stay_path=config.icu_stay_path)
    parser.load_data()

    # Select patients using the same logic as both experiments
    print("\n2. Selecting balanced patient cohort (same logic as experiments)...")
    n_per_class = 5
    selected_patients = select_balanced_patients(parser.icu_stay_df, n_survived=n_per_class, n_died=n_per_class)

    print(f"\n3. Selected patients (seed={seed}, n_per_class={n_per_class}):")
    print("=" * 80)
    for idx, (_, row) in enumerate(selected_patients.iterrows(), 1):
        outcome = "SURVIVED" if row["survived"] else "DIED"
        print(
            f"{idx:2d}. Subject: {row['subject_id']:8d}, "
            f"ICU Stay: {row['icu_stay_id']:8d}, "
            f"Duration: {row['icu_duration_hours']:6.1f}h, "
            f"Outcome: {outcome}"
        )

    # Check if there are existing experiment results
    print("\n4. Checking existing experiment results...")
    experiments_dir = Path("experiments")

    # Find most recent baseline experiment
    baseline_dirs = sorted(experiments_dir.glob("baseline-results-*"))
    if baseline_dirs:
        latest_baseline = baseline_dirs[-1]
        baseline_file = latest_baseline / "baseline_aggregate_results.json"
        if baseline_file.exists():
            with open(baseline_file) as f:
                baseline_results = json.load(f)
            baseline_patients = [
                (r["subject_id"], r["icu_stay_id"]) for r in baseline_results.get("individual_results", [])
            ]
            print(f"\n   Latest baseline experiment: {latest_baseline.name}")
            print(f"   Patients used: {len(baseline_patients)}")
            for i, (subj, icu) in enumerate(baseline_patients, 1):
                print(f"      {i}. Subject: {subj}, ICU Stay: {icu}")

    # Find survival prediction experiments
    survival_dirs = sorted(experiments_dir.glob("results-*"))
    if survival_dirs:
        for survival_dir in survival_dirs[-3:]:  # Check last 3
            patient_files = list(survival_dir.glob("patient_*.json"))
            if patient_files:
                print(f"\n   Survival prediction experiment: {survival_dir.name}")
                print(f"   Patients used: {len(patient_files)}")
                survival_patients = []
                for pf in sorted(patient_files):
                    # Extract subject_id and icu_stay_id from filename
                    # Format: patient_{subject_id}_{icu_stay_id}.json
                    parts = pf.stem.split("_")
                    if len(parts) >= 3:
                        subj_id = int(parts[1])
                        icu_id = int(parts[2])
                        survival_patients.append((subj_id, icu_id))
                        print(f"      {len(survival_patients)}. Subject: {subj_id}, ICU Stay: {icu_id}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(
        "\nBoth experiments use the SAME patient selection logic with the SAME seed (seed=1)."
    )
    print(
        "If they were run with the SAME filtered dataset, they should use IDENTICAL patients."
    )
    print(
        "\nHowever, if the filtering was added AFTER one experiment was run, "
    )
    print("the patient cohorts may differ due to different available patients.")
    print("=" * 80)


if __name__ == "__main__":
    main()
