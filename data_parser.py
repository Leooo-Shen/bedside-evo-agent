import json
from datetime import timedelta
from typing import Dict, List

import pandas as pd


class MIMICDataParser:
    """
    Parser for MIMIC-demo dataset to format patient trajectories for Oracle evaluation.

    The parser loads ICU stay metadata and clinical events, then formats them into
    time-windowed trajectories suitable for retrospective evaluation.
    """

    def __init__(self, events_path: str, icu_stay_path: str):
        """
        Initialize the parser with paths to MIMIC-demo data files.

        Args:
            events_path: Path to events parquet file (e.g., data/mimic-demo/events/data_0.parquet)
            icu_stay_path: Path to ICU stay parquet file (e.g., data/mimic-demo/icu_stay/data_0.parquet)
        """
        self.events_path = events_path
        self.icu_stay_path = icu_stay_path
        self.events_df = None
        self.icu_stay_df = None

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

        print(f"Loaded {len(self.events_df)} events and {len(self.icu_stay_df)} ICU stays")

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

        trajectory = {
            "subject_id": int(subject_id),
            "icu_stay_id": int(icu_stay_id),
            "enter_time": icu_stay["enter_time"].isoformat(),
            "leave_time": icu_stay["leave_time"].isoformat(),
            "age_at_admission": float(age_at_admission),
            "icu_duration_hours": float(icu_stay["icu_duration_hours"]),
            "survived": bool(icu_stay["survived"]),
            "death_time": icu_stay["death_time"].isoformat() if pd.notna(icu_stay["death_time"]) else None,
            "readmission": pd.notna(icu_stay["readm_time"]),
            "readm_duration_hours": (
                float(icu_stay["readm_duration_hours"]) if pd.notna(icu_stay["readm_duration_hours"]) else None
            ),
            "events": patient_events.to_dict("records"),
        }

        return trajectory

    @staticmethod
    def _clean_event(event: Dict, prev_event_time=None) -> Dict:
        """
        Clean and filter event data to keep only relevant fields.

        This reduces context size by keeping only clinically relevant information:
        - time: Event timestamp
        - code: Item identifier
        - numeric_value: Numeric measurement
        - code_specifics: Label/description
        - end_time: End timestamp if available
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
            cleaned["start_time"] = str(event["time"])

        if "end" in event and pd.notna(event["end"]):
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

    @staticmethod
    def _clean_events_list(events: List[Dict]) -> List[Dict]:
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
            cleaned = MIMICDataParser._clean_event(event, prev_time)
            if cleaned:  # Only add if there's any data
                cleaned_events.append(cleaned)
                # Update prev_time for next iteration
                if "time" in cleaned:
                    try:
                        prev_time = pd.to_datetime(cleaned["time"])
                    except:
                        pass

        return cleaned_events

    def create_time_windows(
        self,
        trajectory: Dict,
        current_window_hours: float = 0.5,
        lookback_window_hours: float = 6.0,
        future_window_hours: float = 6.0,
        window_step_hours: float = 0.5,
        include_pre_icu_data: bool = True,
    ) -> List[Dict]:
        """
        Split a patient trajectory into time windows for Oracle evaluation.

        Each window contains THREE segments:
        - History: Events from (current_start - lookback_window_hours) to current_start
          If include_pre_icu_data is True, includes pre-ICU hospital events
        - Current: Events from current_start to (current_start + current_window_hours)
        - Future: Events from (current_start + current_window_hours) to (current_start + current_window_hours + future_window_hours)
        - Metadata: Patient info and outcome

        Args:
            trajectory: Patient trajectory from get_patient_trajectory()
            current_window_hours: Size of current observation window (default 0.5 = 30 minutes)
            lookback_window_hours: Size of historical lookback before current starts (default 6 hours)
            future_window_hours: Size of future prediction window after current ends (default 6 hours)
            window_step_hours: Step size between sliding windows (default 0.5 = 30 minutes)
            include_pre_icu_data: Whether to include pre-ICU hospital data (default True)

        Example with current=0.5, lookback=6, future=6, step=0.5:
            Window 0: history=[-6h, 0h], current=[0h, 0.5h], future=[0.5h, 6.5h]
            Window 1: history=[-5.5h, 0.5h], current=[0.5h, 1h], future=[1h, 7h]
            Window 2: history=[-5h, 1h], current=[1h, 1.5h], future=[1.5h, 7.5h]
            ...

        Returns:
            List of time windows, each suitable for Oracle evaluation
        """
        enter_time = pd.to_datetime(trajectory["enter_time"])
        leave_time = pd.to_datetime(trajectory["leave_time"])

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
            events_df = events_df[(events_df["event_time"] >= enter_time) & (events_df["event_time"] <= leave_time)].copy()

        windows = []
        current_start = enter_time

        while current_start < leave_time:
            # Calculate window boundaries
            current_end = current_start + timedelta(hours=current_window_hours)
            future_end = min(current_end + timedelta(hours=future_window_hours), leave_time)
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
                cleaned_history = self._clean_events_list(history_events.to_dict("records"))
                cleaned_current = self._clean_events_list(current_events.to_dict("records"))
                cleaned_future = self._clean_events_list(future_events.to_dict("records"))

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

        return windows

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
