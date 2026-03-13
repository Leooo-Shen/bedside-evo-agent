"""Tests for Oracle local ICU context-window construction."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import agents.oracle as oracle_module


class FakeLLMClient:
    def __init__(self, provider="openai", model=None, api_key=None, temperature=0.3, max_tokens=4096):
        self.provider = provider
        self.model = model or "fake-model"

    def chat(self, prompt: str, response_format: str = "text", **kwargs):
        return {
            "content": "",
            "parsed": {
                "patient_status": {
                    "domains": {
                        "hemodynamics": {"label": "stable", "key_signals": ["MAP stable"], "rationale": "stable"},
                        "respiratory": {"label": "stable", "key_signals": ["oxygen stable"], "rationale": "stable"},
                        "renal_metabolic": {
                            "label": "insufficient_data",
                            "key_signals": ["no recent labs"],
                            "rationale": "limited",
                        },
                        "neurology": {"label": "stable", "key_signals": ["no acute changes"], "rationale": "stable"},
                    },
                    "overall": {"label": "stable", "rationale": "stable"},
                },
                "action_evaluations": [],
                "overall_window_summary": "stable",
            },
            "usage": {"input_tokens": 40, "output_tokens": 12},
            "model": self.model,
        }


def _build_trajectory() -> dict:
    enter = datetime(2024, 1, 1, 0, 0, 0)
    leave = enter + timedelta(hours=120)
    events = [
        {
            "time": (enter - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "code": "VITALS",
            "code_specifics": "MAP_pre_icu",
            "numeric_value": 50,
        },
        {
            "time": (enter + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "code": "VITALS",
            "code_specifics": "MAP_1h",
            "numeric_value": 62,
        },
        {
            "time": (enter + timedelta(hours=30)).strftime("%Y-%m-%d %H:%M:%S"),
            "code": "DRUG_START",
            "code_specifics": "Norepinephrine",
        },
        {
            "time": (enter + timedelta(hours=55)).strftime("%Y-%m-%d %H:%M:%S"),
            "code": "LAB",
            "code_specifics": "Lactate",
            "numeric_value": 4.2,
        },
        {
            "time": (enter + timedelta(hours=70)).strftime("%Y-%m-%d %H:%M:%S"),
            "code": "VENT_CHANGE",
            "code_specifics": "PEEP 10",
        },
        {
            "time": (enter + timedelta(hours=72)).strftime("%Y-%m-%d %H:%M:%S"),
            "code": "META_DEATH",
            "code_specifics": "Outcome event",
        },
        {
            "time": (enter + timedelta(hours=110)).strftime("%Y-%m-%d %H:%M:%S"),
            "code": "NOTE_DISCHARGESUMMARY",
            "code_specifics": "ICU discharge note",
            "text_value": "Discharge summary outside bounded context window.",
        },
        {
            "time": (enter + timedelta(hours=101)).strftime("%Y-%m-%d %H:%M:%S"),
            "code": "VITALS",
            "code_specifics": "MAP_101h",
            "numeric_value": 68,
        },
    ]
    return {
        "subject_id": 1,
        "icu_stay_id": 10,
        "enter_time": enter.isoformat(),
        "leave_time": leave.isoformat(),
        "events": events,
    }


def _build_trajectory_with_leaky_discharge_summary() -> dict:
    trajectory = _build_trajectory()
    for event in trajectory["events"]:
        if event.get("code") != "NOTE_DISCHARGESUMMARY":
            continue
        event["code_specifics"] = "Expired note"
        event["text_value"] = (
            "Brief Hospital Course: Patient required escalating support. "
            "Discharge Disposition: Expired "
            "Discharge Diagnosis: Septic shock and multi-organ failure. "
            "Discharge Condition: Deceased. "
            "Post-ICU comments: comfort measures were discussed and family noted patient passed away."
        )
    return trajectory


def test_context_window_uses_current_window_start_anchor(monkeypatch):
    monkeypatch.setattr(oracle_module, "LLMClient", FakeLLMClient)
    oracle = oracle_module.MetaOracle(
        provider="openai",
        model="fake-model",
        use_discharge_summary=False,
        history_context_hours=48,
        future_context_hours=48,
    )
    trajectory = _build_trajectory()
    window_data = {
        "current_window_start": "2024-01-02T06:00:00",  # ICU+30h
        "current_window_end": "2024-01-02T06:30:00",
    }

    context = oracle.prepare_context(trajectory, window_data)

    assert context["context_window_start"] == "2024-01-01T00:00:00"
    assert context["context_window_end"] == "2024-01-04T06:00:00"
    assert context["context_event_count"] == 5
    assert context["context_history_event_count"] == 1
    assert context["context_current_window_event_count"] == 1
    assert context["context_future_event_count"] == 3
    assert "MAP_pre_icu" not in context["context_text"]
    assert "MAP_101h" not in context["context_text"]
    assert "## HISTORY EVENTS OF CURRENT WINDOW" in context["context_text"]
    assert "## CURRENT OBSERVATION WINDOW FOR EVALUATION" in context["context_text"]
    assert "Current window duration (hours): 0.50" in context["context_text"]
    assert "## FUTURE EVENTS" in context["context_text"]
    assert "MAP_1h" in context["context_text"]
    assert "CW1." in context["context_text"]
    assert "Norepinephrine" in context["context_text"]
    assert "META_DEATH" in context["context_text"]
    assert "NOTE_DISCHARGESUMMARY" not in context["context_text"]


def test_use_discharge_summary_can_include_summary_block(monkeypatch):
    monkeypatch.setattr(oracle_module, "LLMClient", FakeLLMClient)
    oracle = oracle_module.MetaOracle(
        provider="openai",
        model="fake-model",
        use_discharge_summary=True,
        history_context_hours=48,
        future_context_hours=48,
    )
    trajectory = _build_trajectory()
    window_data = {
        "current_window_start": "2024-01-02T06:00:00",
        "current_window_end": "2024-01-02T06:30:00",
    }

    context = oracle.prepare_context(trajectory, window_data)

    assert context["mode"] == "raw_local_trajectory_with_icu_discharge_summary"
    assert context["use_discharge_summary"] is True
    assert context["context_event_count"] == 5
    assert context["context_history_event_count"] == 1
    assert context["context_current_window_event_count"] == 1
    assert context["context_future_event_count"] == 3
    assert context["has_icu_discharge_summary"] is True
    assert context["icu_discharge_summary_count"] == 1
    assert "## CURRENT DISCHARGE SUMMARY" in context["context_text"]
    assert "Selection rule: trajectory_event_fallback" in context["context_text"]
    assert "## HISTORY EVENTS OF CURRENT WINDOW" in context["context_text"]
    assert "## CURRENT OBSERVATION WINDOW FOR EVALUATION" in context["context_text"]
    assert "Current window duration (hours): 0.50" in context["context_text"]
    assert "## FUTURE EVENTS" in context["context_text"]
    assert "CW1." in context["context_text"]
    assert "Norepinephrine" in context["context_text"]
    assert "Discharge summary outside bounded context window." in context["context_text"]


def test_discharge_summary_not_filtered_when_outcome_is_in_prompt(monkeypatch):
    monkeypatch.setattr(oracle_module, "LLMClient", FakeLLMClient)
    oracle = oracle_module.MetaOracle(
        provider="openai",
        model="fake-model",
        use_discharge_summary=True,
        include_icu_outcome_in_prompt=True,
        history_context_hours=48,
        future_context_hours=48,
    )
    trajectory = _build_trajectory_with_leaky_discharge_summary()
    window_data = {
        "current_window_start": "2024-01-02T06:00:00",
        "current_window_end": "2024-01-02T06:30:00",
    }

    context = oracle.prepare_context(trajectory, window_data)

    assert "Discharge Disposition: Expired" in context["context_text"]
    assert "Discharge Condition: Deceased" in context["context_text"]
    assert "[OUTCOME_MASKED]" not in context["context_text"]


def test_discharge_summary_is_filtered_when_outcome_hidden(monkeypatch):
    monkeypatch.setattr(oracle_module, "LLMClient", FakeLLMClient)
    oracle = oracle_module.MetaOracle(
        provider="openai",
        model="fake-model",
        use_discharge_summary=True,
        include_icu_outcome_in_prompt=False,
        history_context_hours=48,
        future_context_hours=48,
    )
    trajectory = _build_trajectory_with_leaky_discharge_summary()
    window_data = {
        "current_window_start": "2024-01-02T06:00:00",
        "current_window_end": "2024-01-02T06:30:00",
    }

    context = oracle.prepare_context(trajectory, window_data)

    assert "Discharge Disposition:" not in context["context_text"]
    assert "Discharge Condition:" not in context["context_text"]
    assert "[OUTCOME_MASKED]" in context["context_text"]
    assert "passed away" not in context["context_text"].lower()
    assert "comfort measures" not in context["context_text"].lower()


def test_discharge_summary_is_filtered_when_mask_flag_enabled(monkeypatch):
    monkeypatch.setattr(oracle_module, "LLMClient", FakeLLMClient)
    oracle = oracle_module.MetaOracle(
        provider="openai",
        model="fake-model",
        use_discharge_summary=True,
        include_icu_outcome_in_prompt=True,
        mask_discharge_summary_outcome_terms=True,
        history_context_hours=48,
        future_context_hours=48,
    )
    trajectory = _build_trajectory_with_leaky_discharge_summary()
    window_data = {
        "current_window_start": "2024-01-02T06:00:00",
        "current_window_end": "2024-01-02T06:30:00",
    }

    context = oracle.prepare_context(trajectory, window_data)

    assert "Discharge Disposition:" not in context["context_text"]
    assert "Discharge Condition:" not in context["context_text"]
    assert "[OUTCOME_MASKED]" in context["context_text"]
    assert "passed away" not in context["context_text"].lower()
