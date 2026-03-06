"""Tests for vital snapshot extraction in MIMICDataParser."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_parser import MIMICDataParser


def test_discover_relevant_vital_code_specifics_filters_non_vitals():
    parser = MIMICDataParser(events_path="", icu_stay_path="")
    parser.events_df = pd.DataFrame(
        [
            {"code": "VITALS", "code_specifics": "Heart Rate, bpm", "numeric_value": 90},
            {"code": "VITALS", "code_specifics": "Respiratory Rate", "numeric_value": 18},
            {"code": "VITALS", "code_specifics": "Non Invasive Blood Pressure mean, mmHg", "numeric_value": 72},
            {"code": "VITALS", "code_specifics": "Temperature Fahrenheit", "numeric_value": 99.1},
            {"code": "VITALS", "code_specifics": "Heart Rate Alarm - Low, bpm", "numeric_value": 50},
            {"code": "VITALS", "code_specifics": "Temperature Site", "numeric_value": None},
            {"code": "LAB_TEST", "code_specifics": "Heart Rate, bpm", "numeric_value": 88},
        ]
    )

    discovered = parser.discover_relevant_vital_code_specifics(min_occurrences=1)

    assert "heart rate" in discovered
    assert "Heart Rate, bpm" in discovered["heart rate"]

    assert "respiratory rate" in discovered
    assert "Respiratory Rate" in discovered["respiratory rate"]

    assert "mean arterial pressure" in discovered
    assert "Non Invasive Blood Pressure mean, mmHg" in discovered["mean arterial pressure"]

    all_discovered_labels = {label for labels in discovered.values() for label in labels}
    assert "Heart Rate Alarm - Low, bpm" not in all_discovered_labels
    assert "Temperature Site" not in all_discovered_labels


def test_extract_vitals_snapshot_first_12_hours_groups_values():
    parser = MIMICDataParser(events_path="", icu_stay_path="")

    trajectory = {
        "enter_time": "2024-01-01T00:00:00",
        "leave_time": "2024-01-02T00:00:00",
        "events": [
            {"time": "2023-12-31 23:30:00", "code": "VITALS", "code_specifics": "Heart Rate, bpm", "numeric_value": 70},
            {"time": "2024-01-01 00:05:00", "code": "VITALS", "code_specifics": "MAP", "numeric_value": 72},
            {"time": "2024-01-01 00:10:00", "code": "VITALS", "code_specifics": "Heart Rate, bpm", "numeric_value": 88},
            {"time": "2024-01-01 00:20:00", "code": "VITALS", "code_specifics": "Heart Rate Alarm - Low, bpm", "numeric_value": 50},
            {
                "time": "2024-01-01 00:30:00",
                "code": "VITALS",
                "code_specifics": "Non Invasive Blood Pressure systolic, mmHg",
                "numeric_value": 118,
            },
            {
                "time": "2024-01-01 00:40:00",
                "code": "VITALS",
                "code_specifics": "Non Invasive Blood Pressure mean, mmHg",
                "numeric_value": 75,
            },
            {"time": "2024-01-01 00:50:00", "code": "VITALS", "code_specifics": "O2 saturation", "numeric_value": 96},
            {
                "time": "2024-01-01 01:10:00",
                "code": "VITALS",
                "code_specifics": "Temperature Fahrenheit",
                "numeric_value": 99.8,
            },
            {"time": "2024-01-01 13:00:00", "code": "VITALS", "code_specifics": "Heart Rate, bpm", "numeric_value": 91},
        ],
    }

    snapshot = parser.extract_vitals_snapshot(trajectory, first_n_hours_after_icu=12)

    assert snapshot["heart rate"] == [88.0]
    assert snapshot["mean arterial pressure"] == [72.0, 75.0]
    assert snapshot["systolic blood pressure"] == [118.0]
    assert snapshot["oxygen saturation"] == [96.0]
    assert snapshot["temperature"] == [99.8]

    # Alarm row should be excluded, and out-of-window values ignored.
    assert all(value != 50.0 for value in snapshot["heart rate"])
    assert all(value != 91.0 for value in snapshot["heart rate"])
