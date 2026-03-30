"""Tests for global stable event_id behavior in parser windows."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_parser import MIMICDataParser


def _build_trajectory(include_event_idx: bool = True) -> dict:
    enter = datetime(2024, 1, 1, 0, 0, 0)
    leave = enter + timedelta(hours=2)
    events = [
        {
            "time": "2024-01-01 00:10:00",
            "code": "VITALS",
            "code_specifics": "Heart Rate",
            "numeric_value": 101,
        },
        {
            "time": "2024-01-01 00:20:00",
            "code": "LAB_TEST",
            "code_specifics": "Lactate",
            "numeric_value": 3.2,
        },
    ]
    if include_event_idx:
        events[0]["event_idx"] = 111
        events[1]["event_idx"] = 112

    return {
        "subject_id": 1,
        "icu_stay_id": 10,
        "enter_time": enter.isoformat(),
        "leave_time": leave.isoformat(),
        "age_at_admission": 70.0,
        "survived": True,
        "death_time": None,
        "icu_duration_hours": 2.0,
        "events": events,
    }


def test_cleaned_window_events_use_global_stable_event_id():
    parser = MIMICDataParser("unused_events.parquet", "unused_icu.parquet")
    windows = parser.create_time_windows(
        _build_trajectory(include_event_idx=True),
        current_window_hours=0.5,
        window_step_hours=0.5,
        include_pre_icu_data=False,
        use_first_n_hours_after_icu=1.0,
        use_discharge_summary_for_history=False,
    )

    assert windows
    first_window = windows[0]
    assert first_window["current_events"]

    first_event = first_window["current_events"][0]
    second_event = first_window["current_events"][1]

    assert first_event["event_id"] == 0
    assert second_event["event_id"] == 1
    assert first_event["event_index"] == 0
    assert second_event["event_index"] == 1
    assert isinstance(first_event["event_id"], int)
    assert isinstance(second_event["event_id"], int)


def test_cleaned_window_events_keep_global_ids_without_event_idx():
    parser = MIMICDataParser("unused_events.parquet", "unused_icu.parquet")
    windows = parser.create_time_windows(
        _build_trajectory(include_event_idx=False),
        current_window_hours=0.5,
        window_step_hours=0.5,
        include_pre_icu_data=False,
        use_first_n_hours_after_icu=1.0,
        use_discharge_summary_for_history=False,
    )

    assert windows
    first_window = windows[0]
    assert first_window["current_events"]

    first_event = first_window["current_events"][0]
    second_event = first_window["current_events"][1]

    assert first_event["event_id"] == 0
    assert second_event["event_id"] == 1
    assert first_event["event_index"] == 0
    assert second_event["event_index"] == 1


def test_window_events_keep_same_global_ids_across_windows():
    parser = MIMICDataParser("unused_events.parquet", "unused_icu.parquet")
    enter = datetime(2024, 1, 1, 0, 0, 0)
    leave = enter + timedelta(hours=2)
    trajectory = {
        "subject_id": 1,
        "icu_stay_id": 10,
        "enter_time": enter.isoformat(),
        "leave_time": leave.isoformat(),
        "age_at_admission": 70.0,
        "survived": True,
        "death_time": None,
        "icu_duration_hours": 2.0,
        "events": [
            {
                "time": "2024-01-01 00:10:00",
                "code": "VITALS",
                "code_specifics": "Heart Rate",
                "numeric_value": 100,
                "event_idx": 111,
            },
            {
                "time": "2024-01-01 00:40:00",
                "code": "VITALS",
                "code_specifics": "Heart Rate",
                "numeric_value": 98,
                "event_idx": 112,
            },
        ],
    }

    windows = parser.create_time_windows(
        trajectory,
        current_window_hours=0.5,
        window_step_hours=0.5,
        include_pre_icu_data=False,
        use_first_n_hours_after_icu=1.0,
        use_discharge_summary_for_history=False,
    )

    assert len(windows) >= 2

    first_window = windows[0]
    assert first_window["history_events"] == []
    assert first_window["current_events"][0]["event_id"] == 0

    second_window = windows[1]
    assert second_window["history_events"]
    assert second_window["history_events"][0]["event_id"] == 0
    assert second_window["current_events"][0]["event_id"] == 1
