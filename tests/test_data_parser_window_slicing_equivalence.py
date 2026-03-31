"""Regression tests for the single optimized window-slicing path."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_parser import MIMICDataParser


def _build_base_trajectory() -> dict:
    enter = datetime(2024, 1, 1, 0, 0, 0)
    leave = enter + timedelta(hours=6)
    return {
        "subject_id": 1,
        "icu_stay_id": 10,
        "enter_time": enter.isoformat(),
        "leave_time": leave.isoformat(),
        "age_at_admission": 67.0,
        "gender": "M",
        "survived": False,
        "death_time": None,
        "icu_duration_hours": 6.0,
        "events": [
            # pre-ICU
            {
                "time": "2023-12-31 23:45:00",
                "code": "LAB_TEST",
                "code_specifics": "Creatinine",
                "numeric_value": 1.5,
                "event_idx": 1,
            },
            # ICU
            {
                "time": "2024-01-01 00:10:00",
                "code": "VITALS",
                "code_specifics": "Heart Rate",
                "numeric_value": 104,
                "event_idx": 2,
            },
            {
                "time": "2024-01-01 00:25:00",
                "code": "LAB_TEST",
                "code_specifics": "Lactate",
                "numeric_value": 2.9,
                "event_idx": 3,
            },
            {
                "time": "2024-01-01 00:35:00",
                "code": "DRUG_START",
                "code_specifics": "Norepinephrine",
                "event_idx": 4,
            },
            {
                "time": "2024-01-01 01:05:00",
                "code": "PROCEDURE",
                "code_specifics": "Arterial Line",
                "event_idx": 5,
            },
            {
                "time": "2024-01-01 01:45:00",
                "code": "VITALS",
                "code_specifics": "MAP",
                "numeric_value": 66,
                "event_idx": 6,
            },
            {
                "time": "2024-01-01 02:10:00",
                "code": "BODY_INPUT",
                "code_specifics": "Urine Output",
                "numeric_value": 40,
                "event_idx": 7,
            },
            {
                "time": "2024-01-01 02:40:00",
                "code": "LAB_TEST",
                "code_specifics": "Bicarbonate",
                "numeric_value": 20,
                "event_idx": 8,
            },
        ],
    }


def _build_subject_events_df_for_history() -> pd.DataFrame:
    rows = [
        # pre-ICU report candidates
        {
            "subject_id": 1,
            "time": "2023-12-30 10:00:00",
            "code": "NOTE_DISCHARGESUMMARY",
            "code_specifics": "Old discharge summary",
            "text_value": "Past hospitalization summary.",
            "hosp_stay_id": 5001,
            "icu_stay_id": pd.NA,
        },
        {
            "subject_id": 1,
            "time": "2023-12-31 12:00:00",
            "code": "NOTE_RADIOLOGYREPORT",
            "code_specifics": "CXR report",
            "text_value": "Bibasilar opacities.",
            "hosp_stay_id": 5002,
            "icu_stay_id": pd.NA,
        },
        # pre-ICU baseline lab/vital rows
        {
            "subject_id": 1,
            "time": "2023-12-31 22:30:00",
            "code": "LAB_TEST",
            "code_specifics": "Creatinine",
            "numeric_value": 1.7,
            "text_value": None,
            "hosp_stay_id": 5002,
            "icu_stay_id": pd.NA,
        },
        {
            "subject_id": 1,
            "time": "2023-12-31 23:10:00",
            "code": "VITALS",
            "code_specifics": "Heart Rate",
            "numeric_value": 102,
            "text_value": None,
            "hosp_stay_id": 5002,
            "icu_stay_id": pd.NA,
        },
        # ICU rows
        {
            "subject_id": 1,
            "time": "2024-01-01 00:10:00",
            "code": "VITALS",
            "code_specifics": "Heart Rate",
            "numeric_value": 104,
            "text_value": None,
            "hosp_stay_id": 5002,
            "icu_stay_id": 10,
        },
    ]
    return pd.DataFrame(rows)


def test_window_slicing_without_report_history_returns_consistent_windows():
    parser = MIMICDataParser("unused_events.parquet", "unused_icu.parquet")
    trajectory = _build_base_trajectory()

    windows = parser.create_time_windows(
        trajectory=trajectory,
        current_window_hours=0.5,
        window_step_hours=0.5,
        include_pre_icu_data=True,
        use_first_n_hours_after_icu=None,
        use_discharge_summary_for_history=False,
    )

    assert windows

    first_window = windows[0]
    assert first_window["history_events"]
    assert first_window["history_events"][0]["code"] == "LAB_TEST"
    assert first_window["num_history_events"] == len(first_window["history_events"])
    assert first_window["num_current_events"] == len(first_window["current_events"])
    assert first_window["pre_icu_history"]["source"] == "disabled"

    for window in windows:
        assert window["num_history_events"] == len(window["history_events"])
        assert window["num_current_events"] == len(window["current_events"])


def test_window_slicing_with_report_history_uses_unified_pre_icu_history_events():
    parser = MIMICDataParser("unused_events.parquet", "unused_icu.parquet")
    parser.events_df = _build_subject_events_df_for_history()
    parser._selected_discharge_summary_map = {
        (1, 10): {
            "time": "2024-01-01T03:30:00",
            "text_value": "Current stay discharge summary.",
            "code_specifics": "Discharge summary",
            "selection_rule": "post_icu_same_hosp_within_7d_no_new_icu",
            "delta_hours_after_leave": 0.5,
        }
    }

    trajectory = _build_base_trajectory()
    windows = parser.create_time_windows(
        trajectory=trajectory,
        current_window_hours=0.5,
        window_step_hours=0.5,
        include_pre_icu_data=True,
        use_first_n_hours_after_icu=None,
        use_discharge_summary_for_history=True,
        num_discharge_summaries=2,
        relative_report_codes=["NOTE_RADIOLOGYREPORT"],
        pre_icu_history_hours=48.0,
    )

    assert windows
    first_window = windows[0]
    assert first_window["history_events"]
    assert first_window["history_events"][0]["code"] in {"NOTE_DISCHARGESUMMARY", "NOTE_RADIOLOGYREPORT"}
    assert first_window["pre_icu_history"]["source"] == "pre_icu_history"
    assert first_window["current_discharge_summary"] is not None
