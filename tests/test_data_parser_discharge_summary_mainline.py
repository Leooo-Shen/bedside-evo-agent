"""Tests for selector-backed discharge summary extraction in parser mainline."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_parser import MIMICDataParser


def _build_icu_stay_df() -> pd.DataFrame:
    base_birth = datetime(1950, 1, 1, 0, 0, 0)
    return pd.DataFrame(
        [
            {
                "subject_id": 1,
                "icu_stay_id": 101,
                "enter_time": "2024-01-01 00:00:00",
                "leave_time": "2024-01-02 00:00:00",
                "birth_time": base_birth,
                "icu_duration_hours": 24.0,
                "survived": True,
                "readm_time": pd.NaT,
                "readm_duration_hours": pd.NA,
                "death_time": pd.NaT,
            },
            {
                "subject_id": 1,
                "icu_stay_id": 102,
                "enter_time": "2024-01-05 00:00:00",
                "leave_time": "2024-01-06 00:00:00",
                "birth_time": base_birth,
                "icu_duration_hours": 24.0,
                "survived": True,
                "readm_time": pd.NaT,
                "readm_duration_hours": pd.NA,
                "death_time": pd.NaT,
            },
            {
                "subject_id": 1,
                "icu_stay_id": 103,
                "enter_time": "2024-01-10 00:00:00",
                "leave_time": "2024-01-11 00:00:00",
                "birth_time": base_birth,
                "icu_duration_hours": 24.0,
                "survived": True,
                "readm_time": pd.NaT,
                "readm_duration_hours": pd.NA,
                "death_time": pd.NaT,
            },
        ]
    )


def _build_events_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # ICU-hospitalization mapping bridges
            {
                "subject_id": 1,
                "time": "2024-01-01 01:00:00",
                "code": "VITALS",
                "hosp_stay_id": 5001,
                "icu_stay_id": 101,
                "code_specifics": "Heart Rate",
                "numeric_value": 88.0,
                "text_value": None,
            },
            {
                "subject_id": 1,
                "time": "2024-01-05 01:00:00",
                "code": "VITALS",
                "hosp_stay_id": 5002,
                "icu_stay_id": 102,
                "code_specifics": "Heart Rate",
                "numeric_value": 92.0,
                "text_value": None,
            },
            {
                "subject_id": 1,
                "time": "2024-01-10 01:00:00",
                "code": "VITALS",
                "hosp_stay_id": 5003,
                "icu_stay_id": 103,
                "code_specifics": "Heart Rate",
                "numeric_value": 95.0,
                "text_value": None,
            },
            # Rule1 early-in-stay summary for ICU 101 -> should be excluded
            {
                "subject_id": 1,
                "time": "2024-01-01 03:00:00",
                "code": "NOTE_DISCHARGESUMMARY",
                "hosp_stay_id": 5001,
                "icu_stay_id": pd.NA,
                "code_specifics": "summary",
                "numeric_value": pd.NA,
                "text_value": "early summary stay101",
            },
            # Rule1 late-in-stay summary for ICU 102 -> should stay
            {
                "subject_id": 1,
                "time": "2024-01-05 20:00:00",
                "code": "NOTE_DISCHARGESUMMARY",
                "hosp_stay_id": 5002,
                "icu_stay_id": pd.NA,
                "code_specifics": "summary",
                "numeric_value": pd.NA,
                "text_value": "valid summary stay102",
            },
            # In-window summary for ICU 103, but hosp id ties to previous ICU stay (5002) -> exclude
            {
                "subject_id": 1,
                "time": "2024-01-10 06:00:00",
                "code": "NOTE_DISCHARGESUMMARY",
                "hosp_stay_id": 5002,
                "icu_stay_id": pd.NA,
                "code_specifics": "summary",
                "numeric_value": pd.NA,
                "text_value": "actually belongs previous stay",
            },
        ]
    )


def _build_icu_stay_df_with_missing_summary() -> pd.DataFrame:
    base_birth = datetime(1950, 1, 1, 0, 0, 0)
    return pd.DataFrame(
        [
            {
                "subject_id": 2,
                "icu_stay_id": 201,
                "enter_time": "2024-01-01 00:00:00",
                "leave_time": "2024-01-02 00:00:00",
                "birth_time": base_birth,
                "icu_duration_hours": 24.0,
                "survived": True,
                "readm_time": pd.NaT,
                "readm_duration_hours": pd.NA,
                "death_time": pd.NaT,
            },
            {
                "subject_id": 2,
                "icu_stay_id": 202,
                "enter_time": "2024-01-04 00:00:00",
                "leave_time": "2024-01-05 00:00:00",
                "birth_time": base_birth,
                "icu_duration_hours": 24.0,
                "survived": True,
                "readm_time": pd.NaT,
                "readm_duration_hours": pd.NA,
                "death_time": pd.NaT,
            },
        ]
    )


def _build_events_df_with_missing_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": 2,
                "time": "2024-01-01 01:00:00",
                "code": "VITALS",
                "hosp_stay_id": 6001,
                "icu_stay_id": 201,
                "code_specifics": "Heart Rate",
                "numeric_value": 88.0,
                "text_value": None,
            },
            {
                "subject_id": 2,
                "time": "2024-01-04 01:00:00",
                "code": "VITALS",
                "hosp_stay_id": 6002,
                "icu_stay_id": 202,
                "code_specifics": "Heart Rate",
                "numeric_value": 90.0,
                "text_value": None,
            },
            {
                "subject_id": 2,
                "time": "2024-01-01 08:00:00",
                "code": "NOTE_DISCHARGESUMMARY",
                "hosp_stay_id": 6001,
                "icu_stay_id": pd.NA,
                "code_specifics": "summary",
                "numeric_value": pd.NA,
                "text_value": "summary for stay 201",
            },
        ]
    )


def test_load_data_keeps_all_icu_stays_with_selector_matched_discharge_summary(monkeypatch):
    events_df = _build_events_df()
    icu_stay_df = _build_icu_stay_df()

    def _fake_read_parquet(path: str):
        if "events" in str(path):
            return events_df.copy()
        if "icu_stay" in str(path):
            return icu_stay_df.copy()
        raise ValueError(path)

    monkeypatch.setattr(pd, "read_parquet", _fake_read_parquet)

    parser = MIMICDataParser("fake_events.parquet", "fake_icu_stay.parquet")
    parser.load_data()

    remaining_ids = parser.icu_stay_df["icu_stay_id"].tolist()
    assert remaining_ids == [101, 102, 103]
    assert parser.discharge_summary_selection_df is not None
    assert len(parser.discharge_summary_selection_df) == 3
    assert (1, 102) in parser._selected_discharge_summary_map


def test_load_data_skips_icu_stays_without_extractable_discharge_summary(monkeypatch):
    events_df = _build_events_df_with_missing_summary()
    icu_stay_df = _build_icu_stay_df_with_missing_summary()

    def _fake_read_parquet(path: str):
        if "events" in str(path):
            return events_df.copy()
        if "icu_stay" in str(path):
            return icu_stay_df.copy()
        raise ValueError(path)

    monkeypatch.setattr(pd, "read_parquet", _fake_read_parquet)

    parser = MIMICDataParser(
        "fake_events.parquet",
        "fake_icu_stay.parquet",
    )
    parser.load_data()

    remaining_ids = parser.icu_stay_df["icu_stay_id"].tolist()
    assert remaining_ids == [201]
    assert parser.discharge_summary_selection_df is not None
    assert len(parser.discharge_summary_selection_df) == 1
    assert bool(parser.discharge_summary_selection_df.iloc[0]["selected"]) is True
    assert (2, 201) in parser._selected_discharge_summary_map
    assert (2, 202) not in parser._selected_discharge_summary_map


def test_create_time_windows_prefers_selector_backed_discharge_summary(monkeypatch):
    events_df = _build_events_df()
    icu_stay_df = _build_icu_stay_df()

    def _fake_read_parquet(path: str):
        if "events" in str(path):
            return events_df.copy()
        if "icu_stay" in str(path):
            return icu_stay_df.copy()
        raise ValueError(path)

    monkeypatch.setattr(pd, "read_parquet", _fake_read_parquet)

    parser = MIMICDataParser("fake_events.parquet", "fake_icu_stay.parquet")
    parser.load_data()

    enter = datetime(2024, 1, 5, 0, 0, 0)
    leave = enter + timedelta(hours=12)
    trajectory = {
        "subject_id": 1,
        "icu_stay_id": 102,
        "enter_time": enter.isoformat(),
        "leave_time": leave.isoformat(),
        "age_at_admission": 70.0,
        "survived": True,
        "death_time": None,
        "icu_duration_hours": 12.0,
        "events": [
            {
                "time": (enter + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "code": "VITALS",
                "code_specifics": "Heart Rate",
                "numeric_value": 88,
                "text_value": None,
            }
        ],
    }

    windows = parser.create_time_windows(
        trajectory,
        current_window_hours=1.0,
        window_step_hours=1.0,
        include_pre_icu_data=True,
        use_discharge_summary_for_history=True,
        num_discharge_summaries=2,
        relative_report_codes=[],
        pre_icu_history_hours=72.0,
    )

    assert windows
    first = windows[0]
    current_summary = first.get("current_discharge_summary")
    assert isinstance(current_summary, dict)
    assert current_summary.get("selection_rule") == "in_icu_exactly_one"
    assert current_summary.get("text_value") == "valid summary stay102"

    content = first["pre_icu_history"]["content"]
    assert first["pre_icu_history_source"] == "reports"
    assert content is not None
    assert "valid summary stay102" not in content
    assert "early summary stay101" in content
