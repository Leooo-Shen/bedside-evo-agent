"""Tests for MedAgent dynamic memory normalization/schema guards."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.med_agent import normalize_dynamic_memory_payload


def test_dynamic_memory_normalization_and_deduplication():
    payload = {
        "current_status": "  unstable but improving  ",
        "active_problems": [
            "Septic shock",
            "septic shock",
            {"text": "AKI"},
            "",
            "Respiratory failure",
        ],
        "critical_events_log": [
            {"time": "2024-01-01 03:00:00", "event": "NE started"},
            {"time": "2024-01-01 01:00:00", "event": "intubation"},
            "2024-01-01 02:00:00: broad-spectrum antibiotics",
            {"time": "2024-01-01 02:00:00", "event": "broad-spectrum antibiotics"},
        ],
        "trends": ["Lactate down", "lactate down", {"item": "Creatinine up"}],
        "interventions_responses": [
            "Fluid bolus -> MAP improved",
            "fluid bolus -> map improved",
            {"value": "NE titration -> MAP maintained"},
        ],
        "patient_specific_patterns": ["Fluid non-responder", "fluid non-responder", {"text": "Platelet decline"}],
    }

    normalized = normalize_dynamic_memory_payload(payload)

    assert normalized["current_status"] == "unstable but improving"
    assert normalized["active_problems"] == ["Septic shock", "AKI", "Respiratory failure"]

    # Must be sorted by time and de-duplicated.
    assert [item["event"] for item in normalized["critical_events_log"]] == [
        "intubation",
        "broad-spectrum antibiotics",
        "NE started",
    ]

    assert normalized["trends"] == ["Lactate down", "Creatinine up"]
    assert normalized["interventions_responses"] == ["Fluid bolus -> MAP improved", "NE titration -> MAP maintained"]
    assert normalized["patient_specific_patterns"] == ["Fluid non-responder", "Platelet decline"]


def test_dynamic_memory_limits_and_defaults():
    payload = {
        "active_problems": [f"p{i}" for i in range(10)],
        "critical_events_log": [
            {"time": f"2024-01-01 00:{i:02d}:00", "event": f"e{i}"} for i in range(30)
        ],
        "patient_specific_patterns": [f"pat{i}" for i in range(12)],
    }

    normalized = normalize_dynamic_memory_payload(
        payload,
        max_active_problems=5,
        max_critical_events=7,
        max_patterns=4,
    )

    assert normalized["current_status"] == "No significant update."
    assert len(normalized["active_problems"]) == 5
    assert len(normalized["critical_events_log"]) == 7
    assert len(normalized["patient_specific_patterns"]) == 4

    # Keep most recent critical events after truncation.
    assert normalized["critical_events_log"][0]["event"] == "e23"
    assert normalized["critical_events_log"][-1]["event"] == "e29"
