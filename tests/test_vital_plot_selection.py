"""Tests for automatic plottable-vitals selection."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.vital_trends import select_plottable_vitals


def test_select_plottable_vitals_prefers_physiologic_numeric_labels() -> None:
    trajectory = {
        "events": [
            # physiologic numeric vitals
            {"time": "2024-01-01 00:00:00", "code": "VITALS", "code_specifics": "Heart Rate, bpm", "numeric_value": 90},
            {"time": "2024-01-01 00:10:00", "code": "VITALS", "code_specifics": "Heart Rate, bpm", "numeric_value": 94},
            {"time": "2024-01-01 00:20:00", "code": "VITALS", "code_specifics": "Heart Rate, bpm", "numeric_value": 92},
            {"time": "2024-01-01 00:30:00", "code": "VITALS", "code_specifics": "Heart Rate, bpm", "numeric_value": 97},
            {"time": "2024-01-01 00:40:00", "code": "VITALS", "code_specifics": "Heart Rate, bpm", "numeric_value": 95},
            {"time": "2024-01-01 00:50:00", "code": "VITALS", "code_specifics": "Heart Rate, bpm", "numeric_value": 93},
            {"time": "2024-01-01 00:00:00", "code": "VITALS", "code_specifics": "Respiratory Rate, insp/min", "numeric_value": 18},
            {"time": "2024-01-01 00:15:00", "code": "VITALS", "code_specifics": "Respiratory Rate, insp/min", "numeric_value": 20},
            {"time": "2024-01-01 00:30:00", "code": "VITALS", "code_specifics": "Respiratory Rate, insp/min", "numeric_value": 19},
            {"time": "2024-01-01 00:45:00", "code": "VITALS", "code_specifics": "Respiratory Rate, insp/min", "numeric_value": 21},
            {"time": "2024-01-01 01:00:00", "code": "VITALS", "code_specifics": "Respiratory Rate, insp/min", "numeric_value": 20},
            # numeric but not physiologic trend label
            {"time": "2024-01-01 00:00:00", "code": "VITALS", "code_specifics": "Braden Nutrition", "numeric_value": 3},
            {"time": "2024-01-01 00:20:00", "code": "VITALS", "code_specifics": "Braden Nutrition", "numeric_value": 3},
            {"time": "2024-01-01 00:40:00", "code": "VITALS", "code_specifics": "Braden Nutrition", "numeric_value": 4},
            {"time": "2024-01-01 01:00:00", "code": "VITALS", "code_specifics": "Braden Nutrition", "numeric_value": 3},
            {"time": "2024-01-01 01:20:00", "code": "VITALS", "code_specifics": "Braden Nutrition", "numeric_value": 4},
            # non-numeric
            {"time": "2024-01-01 00:00:00", "code": "VITALS", "code_specifics": "Heart Rhythm", "numeric_value": None},
        ]
    }

    selected = select_plottable_vitals(trajectory, min_points=5, max_vitals=10)

    assert "Heart Rate, bpm" in selected
    assert "Respiratory Rate, insp/min" in selected
    assert "Braden Nutrition" not in selected
    assert "Heart Rhythm" not in selected


def test_select_plottable_vitals_falls_back_to_numeric_labels_when_no_pattern_match() -> None:
    trajectory = {
        "events": [
            {"time": "2024-01-01 00:00:00", "code": "VITALS", "code_specifics": "Custom Signal A", "numeric_value": 1.0},
            {"time": "2024-01-01 00:10:00", "code": "VITALS", "code_specifics": "Custom Signal A", "numeric_value": 2.0},
            {"time": "2024-01-01 00:20:00", "code": "VITALS", "code_specifics": "Custom Signal A", "numeric_value": 3.0},
            {"time": "2024-01-01 00:30:00", "code": "VITALS", "code_specifics": "Custom Signal B", "numeric_value": 2.0},
            {"time": "2024-01-01 00:40:00", "code": "VITALS", "code_specifics": "Custom Signal B", "numeric_value": 5.0},
            {"time": "2024-01-01 00:50:00", "code": "VITALS", "code_specifics": "Custom Signal B", "numeric_value": 4.0},
        ]
    }

    selected = select_plottable_vitals(trajectory, min_points=3, max_vitals=5)
    assert selected == ["Custom Signal A", "Custom Signal B"] or selected == ["Custom Signal B", "Custom Signal A"]


def test_select_plottable_vitals_filters_sparse_or_single_timestamp_labels() -> None:
    trajectory = {
        "events": [
            # valid label
            {"time": "2024-01-01 00:00:00", "code": "VITALS", "code_specifics": "Heart Rate, bpm", "numeric_value": 80},
            {"time": "2024-01-01 00:10:00", "code": "VITALS", "code_specifics": "Heart Rate, bpm", "numeric_value": 81},
            {"time": "2024-01-01 00:20:00", "code": "VITALS", "code_specifics": "Heart Rate, bpm", "numeric_value": 82},
            # insufficient points
            {"time": "2024-01-01 00:00:00", "code": "VITALS", "code_specifics": "O2 saturation pulseoxymetry, %", "numeric_value": 96},
            {"time": "2024-01-01 00:10:00", "code": "VITALS", "code_specifics": "O2 saturation pulseoxymetry, %", "numeric_value": 97},
            # same timestamp repeated
            {"time": "2024-01-01 01:00:00", "code": "VITALS", "code_specifics": "Respiratory Rate, insp/min", "numeric_value": 18},
            {"time": "2024-01-01 01:00:00", "code": "VITALS", "code_specifics": "Respiratory Rate, insp/min", "numeric_value": 19},
            {"time": "2024-01-01 01:00:00", "code": "VITALS", "code_specifics": "Respiratory Rate, insp/min", "numeric_value": 20},
        ]
    }

    selected = select_plottable_vitals(trajectory, min_points=3, max_vitals=10)

    assert selected == ["Heart Rate, bpm"]
