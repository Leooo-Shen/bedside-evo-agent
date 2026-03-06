"""Unit tests for Oracle prompt templates."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prompts.oracle_prompt import (
    format_oracle_prompt,
)


def test_format_oracle_prompt_contains_required_blocks():
    window_data = {
        "hours_since_admission": 1.0,
        "current_window_start": "2024-01-01T01:00:00",
        "current_window_end": "2024-01-01T01:30:00",
        "patient_metadata": {"age": 65.0, "total_icu_duration_hours": 120.0, "survived": False},
        "pre_icu_history": {
            "source": "reports",
            "items": 1,
            "content": "--- Report 1: Discharge Summary ---\nprior admission summary",
            "fallback_hours": 72.0,
            "baseline_content": "Baseline labs ...\n- Creatinine, mg/dL = 1.2",
            "baseline_events_count": 1,
            "history_hours": 72.0,
        },
        "current_events": [
            {
                "time": "2024-01-01 01:05:00",
                "code": "DRUG_START",
                "code_specifics": "Norepinephrine",
            }
        ],
    }

    prompt = format_oracle_prompt(
        window_data=window_data,
        context_block="Context type: full raw trajectory\nCTX1. ...",
        context_mode="raw_local_trajectory_icu_events_only",
        history_hours=12,
        future_hours=6,
        top_k=4,
    )

    assert "PATIENT ICU CONTEXT WINDOW" in prompt
    assert "CURRENT OBSERVATION WINDOW FOR EVALUATION" in prompt
    assert "{top_k}" not in prompt
    assert "{window_time}" not in prompt
    assert "top 4 recommendations" in prompt
    assert "CW1." in prompt
    assert "Total ICU Stay: 120.0 hours" in prompt
    assert "Current Hour Since ICU Admission: 1.0" in prompt
    assert "ICU Outcome: Died after ICU" in prompt
    assert "Context Mode: raw_local_trajectory_icu_events_only" in prompt
    assert "## PRE-ICU HISTORY" in prompt
    assert "--- Report 1: Discharge Summary ---" in prompt
    assert "prior admission summary" in prompt
    assert "## PRE-ICU BASELINE SNAPSHOT" in prompt
    assert "Baseline labs ..." in prompt
    assert '"action_evaluations"' in prompt
    assert '"overall_window_summary"' in prompt


def test_format_oracle_prompt_with_missing_pre_icu_history():
    window_data = {
        "hours_since_admission": 0.5,
        "current_window_start": "2024-01-01T00:30:00",
        "current_window_end": "2024-01-01T01:00:00",
        "patient_metadata": {"age": 70.0},
        "current_events": [],
    }

    prompt = format_oracle_prompt(
        window_data=window_data,
        context_block="Context type: full raw trajectory\n(No ICU trajectory events)",
        context_mode="raw_local_trajectory_icu_events_only",
    )

    assert "## PRE-ICU HISTORY" in prompt
    assert "No pre-ICU history provided." in prompt
