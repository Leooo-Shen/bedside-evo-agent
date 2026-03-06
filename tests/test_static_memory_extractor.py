"""Tests for deterministic static memory extraction."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.static_memory_extractor import extract_static_memory


def _build_trajectory() -> dict:
    enter = "2024-01-10T12:00:00"
    return {
        "enter_time": enter,
        "age_at_admission": 68.0,
        "gender": "F",
        "events": [
            {"time": "2024-01-08 12:00:00", "code": "DIAGNOSIS", "code_specifics": "Chronic kidney disease"},
            {"time": "2024-01-09 08:00:00", "code": "DIAGNOSIS", "code_specifics": "Sepsis"},
            {"time": "2024-01-10 12:30:00", "code": "DIAGNOSIS", "code_specifics": "Septic shock"},
            {"time": "2024-01-08 11:00:00", "code": "LAB_TEST", "code_specifics": "Creatinine", "numeric_value": 1.4},
            {"time": "2024-01-09 10:00:00", "code": "LAB_TEST", "code_specifics": "Creatinine", "numeric_value": 2.8},
            {"time": "2024-01-09 06:00:00", "code": "LAB_TEST", "code_specifics": "Sodium", "numeric_value": 138},
            {"time": "2024-01-09 22:00:00", "code": "DRUG_PRESCRIPTION", "code_specifics": "Aspirin"},
            {"time": "2024-01-10 10:00:00", "code": "DRUG_START", "code_specifics": "Lisinopril"},
            {"time": "2024-01-10 10:30:00", "code": "META_RACE", "code_specifics": "WHITE"},
            {"time": "2024-01-10 10:31:00", "code": "META_LANGUAGE", "code_specifics": "ENGLISH"},
            {"time": "2024-01-10 10:32:00", "code": "META_INSURANCE", "code_specifics": "Medicare"},
            {"time": "2024-01-10 10:33:00", "code": "META_MARTIAL_STATUS", "code_specifics": "Married"},
            # Outcome leakage events must be ignored.
            {"time": "2024-01-10 13:00:00", "code": "META_DEATH", "code_specifics": "DECEASED"},
            {"time": "2024-01-10 14:00:00", "code": "LEAVE_ICU", "code_specifics": "Transfer"},
            {"time": "2024-01-10 15:00:00", "code": "NOTE_DISCHARGESUMMARY", "text_value": "Outcome note"},
        ],
    }


def test_extract_static_memory_core_sections():
    memory = extract_static_memory(
        trajectory=_build_trajectory(),
        baseline_lab_lookback_start_hours=72,
        baseline_lab_lookback_end_hours=24,
    )

    demographics = memory["demographics"]
    assert demographics["age"] == 68.0
    assert demographics["gender"] == "F"
    assert demographics["race"] == "WHITE"
    assert demographics["language"] == "ENGLISH"
    assert demographics["insurance"] == "Medicare"
    assert demographics["marital_status"] == "Married"
    assert "Septic shock" in demographics["admission_diagnoses"]

    assert "Chronic kidney disease" in memory["past_medical_history"]
    assert memory["baseline_labs"]["creatinine"] == 2.8
    assert memory["baseline_labs"]["sodium"] == 138.0
    assert memory["baseline_labs"]["albumin"] is None

    assert "Aspirin" in memory["admission_medications"]
    assert "Lisinopril" in memory["admission_medications"]


def test_extract_static_memory_handles_sparse_inputs():
    trajectory = {
        "enter_time": "2024-01-10T12:00:00",
        "age_at_admission": 55,
        "gender": None,
        "events": [
            {"time": "2024-01-10 12:30:00", "code": "DIAGNOSIS", "code_specifics": "Acute respiratory failure"},
        ],
    }

    memory = extract_static_memory(trajectory)
    assert memory["past_medical_history"] == []
    assert memory["admission_medications"] == []
    assert all(value is None for value in memory["baseline_labs"].values())
