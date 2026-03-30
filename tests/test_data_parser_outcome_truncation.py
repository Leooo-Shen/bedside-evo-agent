"""Tests for outcome-based truncation behavior in parser window creation."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_parser import MIMICDataParser


def _build_trajectory_with_outcome(hours_to_outcome: float) -> dict:
    enter = datetime(2024, 1, 1, 0, 0, 0)
    leave = enter + timedelta(hours=10)
    outcome_time = enter + timedelta(hours=hours_to_outcome)
    return {
        "subject_id": 1,
        "icu_stay_id": 10,
        "enter_time": enter.isoformat(),
        "leave_time": leave.isoformat(),
        "age_at_admission": 70.0,
        "gender": "M",
        "survived": False,
        "death_time": outcome_time.isoformat(),
        "icu_duration_hours": 10.0,
        "events": [
            {
                "time": "2024-01-01 00:10:00",
                "code": "VITALS",
                "code_specifics": "Heart Rate",
                "numeric_value": 101,
            },
            {
                "time": outcome_time.strftime("%Y-%m-%d %H:%M:%S"),
                "code": "META_DEATH",
                "code_specifics": "Outcome event",
            },
        ],
    }


def test_skip_patient_when_outcome_truncation_under_four_hours():
    parser = MIMICDataParser("unused_events.parquet", "unused_icu.parquet")
    trajectory = _build_trajectory_with_outcome(hours_to_outcome=2.5)
    windows = parser.create_time_windows(
        trajectory,
        current_window_hours=0.5,
        window_step_hours=0.5,
        include_pre_icu_data=False,
        use_discharge_summary_for_history=False,
    )

    assert windows == []


def test_keep_patient_when_outcome_truncation_at_least_four_hours():
    parser = MIMICDataParser("unused_events.parquet", "unused_icu.parquet")
    trajectory = _build_trajectory_with_outcome(hours_to_outcome=5.0)
    windows = parser.create_time_windows(
        trajectory,
        current_window_hours=0.5,
        window_step_hours=0.5,
        include_pre_icu_data=False,
        use_discharge_summary_for_history=False,
    )

    assert len(windows) >= 1

