"""Shared patient cohort selection utilities for experiments."""

from typing import Optional

import pandas as pd


DEFAULT_SELECTION_SEED = 1


def select_balanced_patients(
    icu_stay_df: pd.DataFrame,
    n_survived: int = 5,
    n_died: int = 5,
    random_seed: Optional[int] = DEFAULT_SELECTION_SEED,
) -> pd.DataFrame:
    """Select a balanced patient cohort by survival outcome."""
    survived_patients = icu_stay_df[icu_stay_df["survived"] == True]
    died_patients = icu_stay_df[icu_stay_df["survived"] == False]

    n_survived_actual = min(n_survived, len(survived_patients))
    n_died_actual = min(n_died, len(died_patients))

    print(f"   Requested: {n_survived} survived, {n_died} died")
    print(f"   Available: {len(survived_patients)} survived, {len(died_patients)} died")
    print(f"   Selected: {n_survived_actual} survived, {n_died_actual} died")

    selected_survived = survived_patients.sample(n=n_survived_actual, random_state=random_seed)
    selected_died = died_patients.sample(n=n_died_actual, random_state=random_seed)

    return pd.concat([selected_survived, selected_died]).sample(frac=1, random_state=random_seed)
