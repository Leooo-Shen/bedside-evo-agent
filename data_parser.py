import json
import random
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.append("..")  # Add parent directory to path for imports


class MIMICDataParser:
    """
    Parser for MIMIC-demo dataset to format patient trajectories for Oracle evaluation.

    The parser loads ICU stay metadata and clinical events, then formats them into
    time-windowed trajectories suitable for retrospective evaluation.
    """

    def __init__(
        self,
        events_path: str,
        icu_stay_path: str,
        de_identify: bool = False,
        de_identify_seed: Optional[int] = None,
    ):
        """
        Initialize the parser with paths to MIMIC-demo data files.

        Args:
            events_path: Path to events parquet file (e.g., data/mimic-demo/events/data_0.parquet)
            icu_stay_path: Path to ICU stay parquet file (e.g., data/mimic-demo/icu_stay/data_0.parquet)
            de_identify: If True, de-identify patient IDs and shift timestamps to prevent data memorization
            de_identify_seed: Random seed for de-identification (for reproducibility)
        """
        self.events_path = events_path
        self.icu_stay_path = icu_stay_path
        self.events_df = None
        self.icu_stay_df = None

        # De-identification settings
        self.de_identify = de_identify
        self.de_identify_seed = de_identify_seed
        self._deidentify_mappings = {}  # Store mappings for consistency
        self._timestamp_shift = None  # Will be set when first patient is processed

        if self.de_identify and self.de_identify_seed is not None:
            random.seed(self.de_identify_seed)

    def load_data(self):
        """Load the MIMIC-demo datasets into memory."""
        print(f"Loading events data from {self.events_path}...")
        self.events_df = pd.read_parquet(self.events_path)

        print(f"Loading ICU stay data from {self.icu_stay_path}...")
        self.icu_stay_df = pd.read_parquet(self.icu_stay_path)

        # Convert time columns to datetime
        self.icu_stay_df["enter_time"] = pd.to_datetime(self.icu_stay_df["enter_time"])
        self.icu_stay_df["leave_time"] = pd.to_datetime(self.icu_stay_df["leave_time"])
        self.icu_stay_df["birth_time"] = pd.to_datetime(self.icu_stay_df["birth_time"])

        # Filter out patients at the very beginning
        initial_count = len(self.icu_stay_df)

        # Filter 1: Remove patients with ICU duration <= 4 hours (died, regardless of treatment)
        # Filter 2: Remove patients transferred out within 24 hours (survived/status ok)
        self.icu_stay_df = self.icu_stay_df[
            ~(
                (self.icu_stay_df["icu_duration_hours"] <= 4)  # Died too quickly
                | (
                    (self.icu_stay_df["icu_duration_hours"] < 24) & (self.icu_stay_df["survived"] == True)
                )  # Transferred out quickly
            )
        ]

        filtered_count = initial_count - len(self.icu_stay_df)
        print(f"Filtered out {filtered_count} patients (Dead < 4h or Survived & Transferred < 24h)")

        # Keep ALL ICU stays for each patient (treat multiple stays independently)
        # Sort by subject_id and enter_time for consistent ordering
        self.icu_stay_df = self.icu_stay_df.sort_values(["subject_id", "enter_time"]).reset_index(drop=True)

        # Count patients with multiple ICU stays
        multiple_stays = self.icu_stay_df.groupby("subject_id").size()
        patients_with_multiple_stays = (multiple_stays > 1).sum()
        total_stays = len(self.icu_stay_df)
        unique_patients = len(multiple_stays)

        print(
            f"Loaded {len(self.events_df)} events and {total_stays} ICU stays from {unique_patients} unique patients"
        )
        print(f"  - Patients with multiple ICU stays: {patients_with_multiple_stays}")

    def _generate_timestamp_shift(self) -> timedelta:
        """
        Generate a random timestamp shift for de-identification.

        Shifts timestamps by a random number of years (±3-7 years) and days (±0-364 days)
        to prevent potential data memorization while preserving relative time differences.

        Returns:
            timedelta object representing the shift
        """
        if self._timestamp_shift is None:
            # Shift by 3-7 years (randomly positive or negative)
            years_shift = random.randint(3, 7) * random.choice([-1, 1])
            # Add random days (0-364)
            days_shift = random.randint(0, 364)

            total_days = years_shift * 365 + days_shift
            self._timestamp_shift = timedelta(days=total_days)

            if self.de_identify:
                print(f"  De-identification: Shifting timestamps by {years_shift} years and {days_shift} days")

        return self._timestamp_shift

    def _deidentify_patient_id(self, patient_id: int, id_type: str = "subject") -> int:
        """
        De-identify a patient ID by generating a consistent random ID.

        Args:
            patient_id: Original patient ID
            id_type: Type of ID ("subject" or "icu_stay")

        Returns:
            De-identified patient ID
        """
        key = f"{id_type}_{patient_id}"

        if key not in self._deidentify_mappings:
            # Generate a random ID in a different range to avoid collisions
            # Use hash to ensure consistency if seed is set
            if self.de_identify_seed is not None:
                # Deterministic based on original ID and seed
                hash_val = hash((patient_id, id_type, self.de_identify_seed))
                self._deidentify_mappings[key] = abs(hash_val) % 90000000 + 10000000
            else:
                # Random ID
                self._deidentify_mappings[key] = random.randint(10000000, 99999999)

        return self._deidentify_mappings[key]

    def _deidentify_timestamp(self, timestamp: datetime) -> datetime:
        """
        De-identify a timestamp by applying a consistent shift.

        Args:
            timestamp: Original timestamp

        Returns:
            Shifted timestamp
        """
        shift = self._generate_timestamp_shift()
        return timestamp + shift

    def get_patient_trajectory(self, subject_id: int, icu_stay_id: int) -> Dict:
        """
        Extract a complete patient trajectory for a specific ICU stay.

        Args:
            subject_id: Patient identifier
            icu_stay_id: ICU stay identifier

        Returns:
            Dictionary containing patient metadata and all events for this ICU stay
        """
        # Get ICU stay metadata
        icu_stay = self.icu_stay_df[
            (self.icu_stay_df["subject_id"] == subject_id) & (self.icu_stay_df["icu_stay_id"] == icu_stay_id)
        ].iloc[0]

        # Get all events for this ICU stay
        # Events are indexed, so we filter by event index range
        patient_events = self.events_df[
            (self.events_df["subject_id"] == subject_id)
            & (self.events_df.index >= icu_stay["min_event_idx"])
            & (self.events_df.index <= icu_stay["max_event_idx"])
        ].copy()

        # Calculate age at admission
        age_at_admission = (icu_stay["enter_time"] - icu_stay["birth_time"]).days / 365.25

        # Extract gender from META_GENDER event
        gender = None
        gender_events = patient_events[patient_events["code"] == "META_GENDER"]
        if len(gender_events) > 0:
            gender = gender_events.iloc[0].get("code_specifics", None)

        # Apply de-identification if enabled
        if self.de_identify:
            deidentified_subject_id = self._deidentify_patient_id(subject_id, "subject")
            deidentified_icu_stay_id = self._deidentify_patient_id(icu_stay_id, "icu_stay")
            enter_time_deidentified = self._deidentify_timestamp(icu_stay["enter_time"])
            leave_time_deidentified = self._deidentify_timestamp(icu_stay["leave_time"])
            death_time_deidentified = (
                self._deidentify_timestamp(icu_stay["death_time"]) if pd.notna(icu_stay["death_time"]) else None
            )
        else:
            deidentified_subject_id = int(subject_id)
            deidentified_icu_stay_id = int(icu_stay_id)
            enter_time_deidentified = icu_stay["enter_time"]
            leave_time_deidentified = icu_stay["leave_time"]
            death_time_deidentified = icu_stay["death_time"] if pd.notna(icu_stay["death_time"]) else None

        trajectory = {
            "subject_id": deidentified_subject_id,
            "icu_stay_id": deidentified_icu_stay_id,
            "enter_time": enter_time_deidentified.isoformat(),
            "leave_time": leave_time_deidentified.isoformat(),
            "age_at_admission": float(age_at_admission),
            "gender": gender,
            "icu_duration_hours": float(icu_stay["icu_duration_hours"]),
            "survived": bool(icu_stay["survived"]),
            "death_time": death_time_deidentified.isoformat() if death_time_deidentified else None,
            "readmission": pd.notna(icu_stay["readm_time"]),
            "readm_duration_hours": (
                float(icu_stay["readm_duration_hours"]) if pd.notna(icu_stay["readm_duration_hours"]) else None
            ),
            "events": patient_events.to_dict("records"),
        }

        return trajectory

    def _clean_event(self, event: Dict, prev_event_time=None) -> Dict:
        """
        Clean and filter event data to keep only relevant fields.

        This reduces context size by keeping only clinically relevant information:
        - time: Event timestamp (formatted as YYYY-MM-DD HH:MM)
        - code: Item identifier
        - numeric_value: Numeric measurement
        - code_specifics: Label/description
        - end_time: End timestamp if available (formatted as YYYY-MM-DD HH:MM)
        - text_value: Text value or unit
        - time_delta_minutes: Minutes since previous event

        Args:
            event: Raw event dictionary
            prev_event_time: Previous event timestamp for calculating delta

        Returns:
            Cleaned event dictionary with only non-null values
        """
        cleaned = {}

        if "time" in event and pd.notna(event["time"]):
            # Format time as YYYY-MM-DD HH:MM (remove seconds for readability and token efficiency)
            try:
                time_dt = pd.to_datetime(event["time"])
                # Apply de-identification if enabled
                if self.de_identify:
                    time_dt = self._deidentify_timestamp(time_dt)
                cleaned["time"] = time_dt.strftime("%Y-%m-%d %H:%M")
            except:
                cleaned["time"] = str(event["time"])

        if "end" in event and pd.notna(event["end"]):
            # Format end time as YYYY-MM-DD HH:MM
            try:
                end_dt = pd.to_datetime(event["end"])
                # Apply de-identification if enabled
                if self.de_identify:
                    end_dt = self._deidentify_timestamp(end_dt)
                cleaned["end_time"] = end_dt.strftime("%Y-%m-%d %H:%M")
            except:
                cleaned["end_time"] = str(event["end"])

        if "code" in event and pd.notna(event["code"]):
            cleaned["code"] = str(event["code"])

        # Numeric value
        if "numeric_value" in event and pd.notna(event["numeric_value"]):
            cleaned["numeric_value"] = float(event["numeric_value"])

        if "code_specifics" in event and pd.notna(event["code_specifics"]):
            cleaned["code_specifics"] = str(event["code_specifics"])

        if "text_value" in event and pd.notna(event["text_value"]):
            cleaned["text_value"] = str(event["text_value"])

        return cleaned

    def _clean_events_list(self, events: List[Dict]) -> List[Dict]:
        """
        Clean a list of events, calculating time deltas between consecutive events.

        Args:
            events: List of raw event dictionaries

        Returns:
            List of cleaned event dictionaries
        """
        cleaned_events = []
        prev_time = None

        for event in events:
            cleaned = self._clean_event(event, prev_time)
            if cleaned:  # Only add if there's any data
                cleaned_events.append(cleaned)
                # Update prev_time for next iteration
                if "time" in cleaned:
                    try:
                        prev_time = pd.to_datetime(cleaned["time"])
                    except:
                        pass

        return cleaned_events

    def filter_events(
        self,
        events_df: pd.DataFrame,
        remove_discharge_summary: bool = True,
    ) -> pd.DataFrame:
        """
        Filter events based on specified criteria.

        This method provides a centralized place for event filtering logic.
        Note: Outcome-revealing events (META_DEATH, LEAVE_*, etc.) are handled
        by stopping window creation at their first occurrence, not by filtering.

        Args:
            events_df: DataFrame containing patient events
            remove_discharge_summary: Remove NOTE_DISCHARGESUMMARY events (default True)

        Returns:
            Filtered DataFrame with events that meet the criteria
        """
        filtered_df = events_df.copy()
        initial_count = len(filtered_df)

        # Filter: Remove NOTE_DISCHARGESUMMARY events if requested
        # These are handled separately via use_discharge_summary_for_history parameter
        if remove_discharge_summary and "code" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["code"] != "NOTE_DISCHARGESUMMARY"].copy()
            removed_count = initial_count - len(filtered_df)
            if removed_count > 0:
                print(f"  Removed {removed_count} NOTE_DISCHARGESUMMARY event(s)")

        return filtered_df

    def create_time_windows(
        self,
        trajectory: Dict,
        current_window_hours: float = 0.5,
        lookback_window_hours: float = 2.0,
        future_window_hours: float = 2.0,
        window_step_hours: float = 0.5,
        include_pre_icu_data: bool = True,
        use_first_n_hours_after_icu: float = 12,
        use_discharge_summary_for_history: bool = False,
        num_discharge_summaries: int = 3,
    ) -> List[Dict]:
        """
        Split a patient trajectory into time windows for Oracle or Agent evaluation.

        Each window contains THREE segments:
        - History: Events from (current_start - lookback_window_hours) to current_start
          If include_pre_icu_data is True, includes pre-ICU hospital events
          If use_discharge_summary_for_history is True, uses discharge summary instead
        - Current: Events from current_start to (current_start + current_window_hours)
        - Future: Events from (current_start + current_window_hours) to (current_start + current_window_hours + future_window_hours)
        - Metadata: Patient info and outcome

        IMPORTANT: To prevent information leakage, window creation automatically stops at the first
        occurrence of outcome-revealing events (LEAVE_HOSPITALIZATION, NOTE_DISCHARGESUMMARY,
        META_DEATH, LEAVE_ED) within the ICU stay. This ensures predictions are made before
        the outcome is revealed.

        Args:
            trajectory: Patient trajectory from get_patient_trajectory()
            current_window_hours: Size of current observation window (default 0.5 = 30 minutes)
            lookback_window_hours: Size of historical lookback before current starts (default 6 hours)
            future_window_hours: Size of future prediction window after current ends (default 6 hours)
            window_step_hours: Step size between sliding windows (default 0.5 = 30 minutes)
            include_pre_icu_data: Whether to include pre-ICU hospital data (default True)
            use_first_n_hours_after_icu: Use only the first N hours after ICU entry (default None)
                                         When set, only creates windows within the first N hours after ICU admission
                                         Example: use_first_n_hours_after_icu=12 uses only first 12 hours
                                         When None, uses the full ICU duration (or until outcome event)
            use_discharge_summary_for_history: Use NOTE_DISCHARGESUMMARY content as history context
                                               instead of history events (default False)
                                               Falls back to history events if no discharge summary found
            num_discharge_summaries: Number of most recent discharge summaries to extract from
                                     before current ICU stay (default 3). Only used when
                                     use_discharge_summary_for_history is True

        Example with current=0.5, lookback=6, future=6, step=0.5:
            Window 0: history=[-6h, 0h], current=[0h, 0.5h], future=[0.5h, 6.5h]
            Window 1: history=[-5.5h, 0.5h], current=[0.5h, 1h], future=[1h, 7h]
            Window 2: history=[-5h, 1h], current=[1h, 1.5h], future=[1.5h, 7.5h]
            ...

        Returns:
            List of time windows, each suitable for Oracle or Agent evaluation
        """
        enter_time = pd.to_datetime(trajectory["enter_time"])
        leave_time = pd.to_datetime(trajectory["leave_time"])

        # Determine effective leave time based on use_first_n_hours_after_icu parameter
        # If use_first_n_hours_after_icu is set, only use the first N hours after ICU entry
        # Otherwise, use the full ICU duration
        if use_first_n_hours_after_icu is not None:
            effective_leave_time = enter_time + timedelta(hours=use_first_n_hours_after_icu)
            # Make sure we don't exceed the actual leave time
            effective_leave_time = min(effective_leave_time, leave_time)
        else:
            effective_leave_time = leave_time

        # Find first occurrence of outcome-revealing events and stop there
        # This prevents information leakage by stopping before outcome is revealed
        outcome_codes = ["LEAVE_HOSPITALIZATION", "NOTE_DISCHARGESUMMARY", "META_DEATH", "LEAVE_ICU"]
        events_df_temp = pd.DataFrame(trajectory["events"])

        if len(events_df_temp) > 0 and "code" in events_df_temp.columns and "time" in events_df_temp.columns:
            # Filter for outcome-revealing events within current ICU stay
            events_df_temp["event_time"] = pd.to_datetime(events_df_temp["time"])

            # Basic outcome codes
            outcome_events = events_df_temp[
                (events_df_temp["code"].isin(outcome_codes))
                & (events_df_temp["event_time"] >= enter_time)
                & (events_df_temp["event_time"] <= leave_time)
            ]

            # Add LEAVE_ED events that reveal outcome (exclude "ADMITTED" which is just transition to ICU)
            leave_ed_outcome = events_df_temp[
                (events_df_temp["code"] == "LEAVE_ED")
                & (events_df_temp["event_time"] >= enter_time)
                & (events_df_temp["event_time"] <= leave_time)
                & (events_df_temp["code_specifics"] != "ADMITTED")  # Exclude admission transitions
            ]

            # Combine outcome events
            outcome_events = pd.concat([outcome_events, leave_ed_outcome], ignore_index=True)

            if len(outcome_events) > 0:
                # Find the earliest outcome event
                first_outcome_time = outcome_events["event_time"].min()
                # Stop windows just before the outcome event (subtract 1 second to ensure we don't include it)
                stop_time = first_outcome_time - timedelta(seconds=1)
                effective_leave_time = min(effective_leave_time, stop_time)
                print(f"  Found outcome event at {first_outcome_time}, stopping windows at {stop_time}")

        # Extract discharge summaries from BEFORE current ICU stay if requested
        discharge_summary_content = None
        if use_discharge_summary_for_history:
            discharge_summaries = self.extract_discharge_summary(trajectory, k=num_discharge_summaries)
            if discharge_summaries:
                # Concatenate all discharge summaries with separators
                summary_texts = []
                for i, ds in enumerate(discharge_summaries):
                    text = ds.get("text_value", "")
                    hours_before = ds.get("hours_before_current_icu", 0)
                    if text:
                        summary_texts.append(
                            f"--- Discharge Summary {i+1} ({hours_before:.1f} hours before current ICU admission) ---\n{text}"
                        )

                if summary_texts:
                    discharge_summary_content = "\n\n".join(summary_texts)
                    total_chars = len(discharge_summary_content)
                    print(
                        f"  Using {len(discharge_summaries)} discharge summary(ies) as history context ({total_chars} characters)"
                    )
                else:
                    print(f"  Discharge summaries found but empty, falling back to history events")
            else:
                print(f"  No discharge summaries found before current ICU stay, falling back to history events")

        # Convert events to DataFrame for easier manipulation
        events_df = pd.DataFrame(trajectory["events"])

        if len(events_df) == 0:
            return []

        # Ensure events have timestamps
        if "time" in events_df.columns:
            events_df["event_time"] = pd.to_datetime(events_df["time"])
        else:
            # If no timestamp, distribute events evenly across the ICU stay
            total_duration = (leave_time - enter_time).total_seconds()
            events_df["event_time"] = [
                enter_time + timedelta(seconds=total_duration * i / len(events_df)) for i in range(len(events_df))
            ]

        # Filter events based on include_pre_icu_data setting
        if include_pre_icu_data:
            # Keep all events up to leave_time (including pre-ICU hospital data)
            events_df = events_df[events_df["event_time"] <= leave_time].copy()
        else:
            # Only keep events during ICU stay (discard pre-ICU hospital history)
            events_df = events_df[
                (events_df["event_time"] >= enter_time) & (events_df["event_time"] <= leave_time)
            ].copy()

        windows = []
        current_start = enter_time

        # Use effective_leave_time for windowing (may be earlier than actual leave_time)
        while current_start < effective_leave_time:
            # Calculate window boundaries
            # IMPORTANT: Cap current_end at effective_leave_time to prevent including outcome events
            current_end = min(current_start + timedelta(hours=current_window_hours), effective_leave_time)
            future_end = min(current_end + timedelta(hours=future_window_hours), effective_leave_time)
            history_start = current_start - timedelta(hours=lookback_window_hours)

            # Split events into three segments: history, current, future
            # History: events from history_start to current_start
            history_events = events_df[
                (events_df["event_time"] >= history_start) & (events_df["event_time"] < current_start)
            ]
            # Current: events from current_start to current_end
            current_events = events_df[
                (events_df["event_time"] >= current_start) & (events_df["event_time"] < current_end)
            ]
            # Future: events from current_end to future_end
            future_events = events_df[
                (events_df["event_time"] >= current_end) & (events_df["event_time"] < future_end)
            ]

            # Only create window if there are current or future events to evaluate
            if len(current_events) > 0 or len(future_events) > 0:
                # Clean events to reduce context size
                cleaned_current = self._clean_events_list(current_events.to_dict("records"))
                cleaned_future = self._clean_events_list(future_events.to_dict("records"))

                # Handle history: use discharge summary if available and mode is enabled
                if use_discharge_summary_for_history and discharge_summary_content:
                    # Use discharge summary as history context
                    cleaned_history = [
                        {
                            "type": "discharge_summary",
                            "content": discharge_summary_content,
                            "note": "Patient history from discharge summary",
                        }
                    ]
                else:
                    # Use history events as normal
                    cleaned_history = self._clean_events_list(history_events.to_dict("records"))

                window = {
                    "subject_id": trajectory["subject_id"],
                    "icu_stay_id": trajectory["icu_stay_id"],
                    "current_window_start": current_start.isoformat(),
                    "current_window_end": current_end.isoformat(),
                    "history_start": history_start.isoformat(),
                    "future_end": future_end.isoformat(),
                    "hours_since_admission": (current_start - enter_time).total_seconds() / 3600,
                    "current_window_hours": current_window_hours,
                    "lookback_window_hours": lookback_window_hours,
                    "future_window_hours": future_window_hours,
                    "patient_metadata": {
                        "age": trajectory["age_at_admission"],
                        "survived": trajectory["survived"],
                        "death_time": trajectory["death_time"],
                        "total_icu_duration_hours": trajectory["icu_duration_hours"],
                    },
                    "history_events": cleaned_history,
                    "current_events": cleaned_current,
                    "future_events": cleaned_future,
                    "num_history_events": len(cleaned_history),
                    "num_current_events": len(cleaned_current),
                    "num_future_events": len(cleaned_future),
                }
                windows.append(window)

            current_start += timedelta(hours=window_step_hours)

        # Drop the window if it has no current events
        windows = [w for w in windows if len(w["current_events"]) > 0]

        return windows

    def extract_discharge_summary(self, trajectory: Dict, k: int = 1) -> Optional[List[Dict]]:
        """
        Extract the k most recent NOTE_DISCHARGESUMMARY events from BEFORE the current ICU stay.

        This provides historical context from previous hospital visits/ICU stays.
        Useful for understanding patient's medical history without information leakage.

        Args:
            trajectory: Patient trajectory from get_patient_trajectory()
            k: Number of most recent discharge summaries to extract (default 1)

        Returns:
            List of discharge summary dictionaries sorted by time (most recent first),
            or None if no discharge summaries found before current ICU enter time
        """
        subject_id = trajectory["subject_id"]
        current_enter_time = pd.to_datetime(trajectory["enter_time"])

        # Get all events for this patient from the events dataframe
        if self.events_df is None:
            return None

        patient_events = self.events_df[self.events_df["subject_id"] == subject_id].copy()

        if len(patient_events) == 0:
            return None

        # Filter for discharge summary events that occurred BEFORE current ICU enter time
        discharge_summaries = patient_events[
            (patient_events["code"] == "NOTE_DISCHARGESUMMARY")
            & (pd.to_datetime(patient_events["time"]) < current_enter_time)
        ].copy()

        if len(discharge_summaries) == 0:
            return None

        # Find the most recent k discharge summaries
        discharge_summaries = discharge_summaries.sort_values("time", ascending=True).tail(k)

        # Convert to list of dictionaries with relevant metadata
        results = []
        for _, event in discharge_summaries.iterrows():
            discharge_summary = {
                "subject_id": int(subject_id),
                "time": event["time"].isoformat() if pd.notna(event["time"]) else None,
                "text_value": event.get("text_value"),
                "code_specifics": event.get("code_specifics"),
                "hours_before_current_icu": (
                    (current_enter_time - pd.to_datetime(event["time"])).total_seconds() / 3600
                    if pd.notna(event["time"])
                    else None
                ),
            }
            results.append(discharge_summary)

        return results if len(results) > 0 else None

    def get_all_trajectories(self) -> List[Dict]:
        """
        Get all patient trajectories from the dataset.

        Returns:
            List of all patient trajectories
        """
        trajectories = []

        for _, icu_stay in self.icu_stay_df.iterrows():
            try:
                trajectory = self.get_patient_trajectory(icu_stay["subject_id"], icu_stay["icu_stay_id"])
                trajectories.append(trajectory)
            except Exception as e:
                print(f"Error processing ICU stay {icu_stay['icu_stay_id']}: {e}")
                continue

        return trajectories

    def save_trajectories(self, trajectories: List[Dict], output_path: str):
        """
        Save trajectories to a JSONL file.

        Args:
            trajectories: List of patient trajectories
            output_path: Path to output JSONL file
        """
        with open(output_path, "w") as f:
            for trajectory in trajectories:
                f.write(json.dumps(trajectory) + "\n")
        print(f"Saved {len(trajectories)} trajectories to {output_path}")

    def load_trajectories(self, input_path: str) -> List[Dict]:
        """
        Load trajectories from a JSONL file.

        Args:
            input_path: Path to input JSONL file

        Returns:
            List of patient trajectories
        """
        trajectories = []
        with open(input_path, "r") as f:
            for line in f:
                trajectories.append(json.loads(line))
        print(f"Loaded {len(trajectories)} trajectories from {input_path}")
        return trajectories
