"""Tests for Oracle pre-ICU history report-first behavior in the parser."""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_parser import MIMICDataParser, PreICUHistoryProcessor
from prompts.oracle_prompt import format_event_line as format_prompt_event_line


def _build_trajectory(subject_id: int = 1) -> dict:
    enter = datetime(2024, 1, 10, 0, 0, 0)
    leave = enter + timedelta(hours=12)
    events = [
        {
            "time": (enter + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "code": "VITALS",
            "code_specifics": "Heart Rate",
            "numeric_value": 100,
            "text_value": None,
        }
    ]
    return {
        "subject_id": subject_id,
        "icu_stay_id": 10,
        "enter_time": enter.isoformat(),
        "leave_time": leave.isoformat(),
        "age_at_admission": 70.0,
        "survived": True,
        "death_time": None,
        "icu_duration_hours": 12.0,
        "events": events,
    }


def test_pre_icu_event_formatter_reuses_prompt_event_logic():
    event = {
        "time": "2024-01-01 12:34:56",
        "code": "LAB_TEST",
        "code_specifics": "Creatinine, mg/dL",
        "numeric_value": 1.23,
        "text_value": "final",
    }
    assert format_prompt_event_line(event) == PreICUHistoryProcessor._format_pre_icu_event_line(event)


def test_pre_icu_report_priority_includes_discharge_and_relative_codes():
    parser = MIMICDataParser("unused_events.parquet", "unused_icu.parquet")
    trajectory = _build_trajectory(subject_id=1)
    enter = pd.to_datetime(trajectory["enter_time"])

    parser.events_df = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "time": enter - timedelta(hours=30),
                "code": "NOTE_DISCHARGESUMMARY",
                "text_value": "older discharge summary",
                "code_specifics": "summary",
            },
            {
                "subject_id": 1,
                "time": enter - timedelta(hours=10),
                "code": "NOTE_DISCHARGESUMMARY",
                "text_value": "newer discharge summary",
                "code_specifics": "summary",
            },
            {
                "subject_id": 1,
                "time": enter - timedelta(hours=2),
                "code": "NOTE_RADIOLOGYREPORT",
                "text_value": "recent radiology report",
                "code_specifics": "radiology",
            },
        ]
    )

    windows = parser.create_time_windows(
        trajectory,
        current_window_hours=1.0,
        window_step_hours=1.0,
        include_pre_icu_data=True,
        use_discharge_summary_for_history=True,
        num_discharge_summaries=2,
        relative_report_codes=["NOTE_RADIOLOGYREPORT"],
        pre_icu_history_hours=72.0,
    )

    assert windows
    first = windows[0]
    assert first["pre_icu_history_source"] == "reports"
    assert first["pre_icu_history_items"] == 3
    report_content = str(first["pre_icu_history"]["content"] or "")
    assert "Discharge Summary" in report_content
    assert "NOTE_RADIOLOGYREPORT" in report_content
    # Report headers should keep minute precision only.
    assert re.search(r"timestamp:\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2};", report_content) is not None
    assert re.search(r"timestamp:\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2};", report_content) is None
    assert first["history_events"][0]["type"] == "pre_icu_reports"


def test_pre_icu_report_cap_applies_per_code():
    parser = MIMICDataParser("unused_events.parquet", "unused_icu.parquet")
    trajectory = _build_trajectory(subject_id=11)
    enter = pd.to_datetime(trajectory["enter_time"])

    parser.events_df = pd.DataFrame(
        [
            {
                "subject_id": 11,
                "time": enter - timedelta(hours=40),
                "code": "NOTE_DISCHARGESUMMARY",
                "text_value": "ds oldest",
                "code_specifics": "summary",
            },
            {
                "subject_id": 11,
                "time": enter - timedelta(hours=30),
                "code": "NOTE_DISCHARGESUMMARY",
                "text_value": "ds older",
                "code_specifics": "summary",
            },
            {
                "subject_id": 11,
                "time": enter - timedelta(hours=20),
                "code": "NOTE_DISCHARGESUMMARY",
                "text_value": "ds newest",
                "code_specifics": "summary",
            },
            {
                "subject_id": 11,
                "time": enter - timedelta(hours=4),
                "code": "NOTE_RADIOLOGYREPORT",
                "text_value": "rad newest",
                "code_specifics": "radiology",
            },
            {
                "subject_id": 11,
                "time": enter - timedelta(hours=8),
                "code": "NOTE_RADIOLOGYREPORT",
                "text_value": "rad older",
                "code_specifics": "radiology",
            },
            {
                "subject_id": 11,
                "time": enter - timedelta(hours=12),
                "code": "NOTE_RADIOLOGYREPORT",
                "text_value": "rad oldest",
                "code_specifics": "radiology",
            },
        ]
    )

    windows = parser.create_time_windows(
        trajectory,
        current_window_hours=1.0,
        window_step_hours=1.0,
        include_pre_icu_data=True,
        use_discharge_summary_for_history=True,
        num_discharge_summaries=2,
        relative_report_codes=["NOTE_RADIOLOGYREPORT"],
        pre_icu_history_hours=72.0,
    )

    assert windows
    first = windows[0]
    content = first["pre_icu_history"]["content"]
    assert first["pre_icu_history_source"] == "reports"
    assert first["pre_icu_history_items"] == 4
    assert content.count("Discharge Summary") == 2
    assert content.count("NOTE_RADIOLOGYREPORT") == 2


def test_pre_icu_fallback_uses_previous_72h_events_only():
    parser = MIMICDataParser("unused_events.parquet", "unused_icu.parquet")
    trajectory = _build_trajectory(subject_id=2)
    enter = pd.to_datetime(trajectory["enter_time"])

    parser.events_df = pd.DataFrame(
        [
            {
                "subject_id": 2,
                "time": enter - timedelta(hours=10),
                "code": "LAB_TEST",
                "numeric_value": 7.2,
                "code_specifics": "Lactate",
            },
            {
                "subject_id": 2,
                "time": enter - timedelta(hours=1),
                "code": "OTHER_EVENT",
                "code_specifics": "ED note",
            },
            {
                "subject_id": 2,
                "time": enter - timedelta(hours=80),
                "code": "LAB_TEST",
                "numeric_value": 1.1,
                "code_specifics": "Creatinine",
            },
        ]
    )

    windows = parser.create_time_windows(
        trajectory,
        current_window_hours=1.0,
        window_step_hours=1.0,
        include_pre_icu_data=True,
        use_discharge_summary_for_history=True,
        num_discharge_summaries=2,
        relative_report_codes=["NOTE_RADIOLOGYREPORT"],
        pre_icu_history_hours=72.0,
    )

    assert windows
    first = windows[0]
    assert first["pre_icu_history_source"] == "events_fallback"
    assert first["pre_icu_history_items"] == len(first["history_events"])
    fallback_content = first["pre_icu_history"]["content"]
    assert fallback_content is not None
    assert "[2024-" not in fallback_content
    assert "LAB_TEST Lactate =7.20" in fallback_content

    earliest = enter - timedelta(hours=72)
    for event in first["history_events"]:
        event_time = pd.to_datetime(event["time"])
        assert earliest <= event_time < enter


def test_pre_icu_history_opt_in_disabled():
    parser = MIMICDataParser("unused_events.parquet", "unused_icu.parquet")
    trajectory = _build_trajectory(subject_id=3)
    enter = pd.to_datetime(trajectory["enter_time"])

    parser.events_df = pd.DataFrame(
        [
            {
                "subject_id": 3,
                "time": enter - timedelta(hours=5),
                "code": "NOTE_DISCHARGESUMMARY",
                "text_value": "history should be disabled when include_pre_icu_data is false",
            }
        ]
    )

    windows = parser.create_time_windows(
        trajectory,
        current_window_hours=1.0,
        window_step_hours=1.0,
        include_pre_icu_data=False,
        use_discharge_summary_for_history=True,
        num_discharge_summaries=2,
        relative_report_codes=["NOTE_RADIOLOGYREPORT"],
        pre_icu_history_hours=72.0,
    )

    assert windows
    first = windows[0]
    assert first["pre_icu_history_source"] == "disabled"
    assert first["pre_icu_history_items"] == 0
    assert first["pre_icu_history"]["content"] is None


def test_pre_icu_baseline_snapshot_includes_all_lab_and_vital_events_in_window():
    parser = MIMICDataParser("unused_events.parquet", "unused_icu.parquet")
    trajectory = _build_trajectory(subject_id=4)
    enter = pd.to_datetime(trajectory["enter_time"])

    parser.events_df = pd.DataFrame(
        [
            {
                "subject_id": 4,
                "time": enter - timedelta(hours=6),
                "code": "LAB_TEST",
                "numeric_value": 2.3,
                "code_specifics": "Creatinine, mg/dL",
            },
            {
                "subject_id": 4,
                "time": enter - timedelta(hours=2),
                "code": "LAB_TEST",
                "numeric_value": 8.5,
                "code_specifics": "Hemoglobin, g/dL",
            },
            {
                "subject_id": 4,
                "time": enter - timedelta(hours=90),
                "code": "LAB_TEST",
                "numeric_value": 4.1,
                "code_specifics": "Lactic Acid, mmol/L",
            },
            {
                "subject_id": 4,
                "time": enter - timedelta(hours=3),
                "code": "VITALS",
                "numeric_value": 110,
                "code_specifics": "Heart Rate, bpm",
            },
            {
                "subject_id": 4,
                "time": enter - timedelta(hours=2),
                "code": "VITALS",
                "numeric_value": 62,
                "code_specifics": "Non Invasive Blood Pressure mean, mmHg",
            },
            {
                "subject_id": 4,
                "time": enter - timedelta(hours=4),
                "code": "NOTE_RADIOLOGYREPORT",
                "text_value": "radiology finding",
            },
        ]
    )

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
    baseline_content = first["pre_icu_history"]["baseline_content"]
    assert baseline_content is not None
    assert "Pre-ICU LAB/VITAL events" in baseline_content
    assert "Creatinine, mg/dL" in baseline_content
    assert "Hemoglobin, g/dL" in baseline_content
    assert "Heart Rate, bpm" in baseline_content
    assert "Non Invasive Blood Pressure mean, mmHg" in baseline_content
    assert "Lactic Acid, mmol/L" not in baseline_content
    assert "[2024-" not in baseline_content
    # Baseline event lines should keep minute precision only.
    assert re.search(r"B1\.\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+LAB_TEST", baseline_content) is not None
    assert re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+LAB_TEST", baseline_content) is None
    assert "=2.30" in baseline_content
    assert "=8.50" in baseline_content
    assert first["pre_icu_history"]["baseline_events_count"] == 4
