"""Smoke test for experiments.oracle.run_oracle_conditions."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import agents.oracle as oracle_module
import experiments.oracle.run_oracle_conditions as suite_module


class FakeLLMClient:
    call_count = 0

    def __init__(self, provider="openai", model=None, api_key=None, temperature=0.3, max_tokens=4096):
        self.provider = provider
        self.model = model or "fake-model"

    def chat(self, prompt: str, response_format: str = "text", **kwargs):
        type(self).call_count += 1
        parsed_payload = {
            "patient_status": {
                "domains": {
                    "hemodynamics": {"label": "stable", "key_signals": ["MAP stable"], "rationale": "stable"},
                    "respiratory": {"label": "stable", "key_signals": ["SpO2 stable"], "rationale": "stable"},
                    "renal_metabolic": {"label": "fluctuating", "key_signals": ["variable labs"], "rationale": "mixed"},
                    "neurology": {"label": "deteriorating", "key_signals": ["GCS worse"], "rationale": "worse"},
                },
                "overall": {"label": "fluctuating", "rationale": "mixed trajectory"},
            },
            "action_evaluations": [],
            "overall_window_summary": "Synthetic summary.",
        }
        return {
            "content": json.dumps(parsed_payload),
            "parsed": parsed_payload,
            "usage": {"input_tokens": 10, "output_tokens": 10},
            "model": self.model,
        }


class FakeParser:
    def __init__(self, events_path: str, icu_stay_path: str, **kwargs):
        self.events_path = events_path
        self.icu_stay_path = icu_stay_path
        self.icu_stay_df = None
        self.discharge_summary_selection_df = None
        self._survival_by_pair = {}

    def load_data(self):
        rows = [
            {"subject_id": 101, "icu_stay_id": 9001, "survived": True},
            {"subject_id": 102, "icu_stay_id": 9002, "survived": False},
            {"subject_id": 103, "icu_stay_id": 9003, "survived": True},
            {"subject_id": 104, "icu_stay_id": 9004, "survived": False},
        ]
        self.icu_stay_df = pd.DataFrame(rows)
        self.discharge_summary_selection_df = pd.DataFrame(
            [
                {"subject_id": row["subject_id"], "icu_stay_id": row["icu_stay_id"], "selected": True}
                for row in rows
            ]
        )
        for row in rows:
            self._survival_by_pair[(int(row["subject_id"]), int(row["icu_stay_id"]))] = bool(row["survived"])

    def get_patient_trajectory(self, subject_id: int, icu_stay_id: int, icu_stay=None):
        survived = self._survival_by_pair[(int(subject_id), int(icu_stay_id))]
        enter = datetime(2024, 1, 1, 0, 0, 0)
        leave = enter + timedelta(hours=12)
        events = [
            {
                "time": (enter + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
                "code": "VITALS",
                "code_specifics": "MAP",
                "numeric_value": 65,
            },
            {
                "time": (enter + timedelta(hours=10)).strftime("%Y-%m-%d %H:%M:%S"),
                "code": "NOTE_DISCHARGESUMMARY",
                "code_specifics": "Expired summary text",
                "text_value": (
                    "Discharge Disposition: Expired\n"
                    "Discharge Condition: Deceased\n"
                    "Family says patient passed away."
                ),
            },
        ]
        return {
            "subject_id": int(subject_id),
            "icu_stay_id": int(icu_stay_id),
            "icu_duration_hours": 12.0,
            "survived": survived,
            "death_time": None if survived else (enter + timedelta(hours=11)).isoformat(),
            "enter_time": enter.isoformat(),
            "leave_time": leave.isoformat(),
            "events": events,
        }

    def create_time_windows(self, trajectory, **kwargs):
        enter = datetime.fromisoformat(trajectory["enter_time"])
        return [
            {
                "subject_id": trajectory["subject_id"],
                "icu_stay_id": trajectory["icu_stay_id"],
                "current_window_start": enter.isoformat(),
                "current_window_end": (enter + timedelta(hours=1)).isoformat(),
                "hours_since_admission": 0.0,
                "current_window_hours": 1.0,
                "patient_metadata": {
                    "age": 70,
                    "survived": trajectory["survived"],
                    "death_time": trajectory["death_time"],
                    "total_icu_duration_hours": 12.0,
                },
                "history_events": [
                    {
                        "type": "pre_icu_reports",
                        "content": "Pre-ICU note says patient deceased and passed away.",
                    }
                ],
                "current_events": [
                    {
                        "time": (enter + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
                        "code": "VITALS",
                        "code_specifics": "MAP",
                        "numeric_value": 65,
                    }
                ],
                "num_history_events": 1,
                "num_current_events": 1,
                "pre_icu_history": {
                    "source": "reports",
                    "items": 1,
                    "content": "Discharge Condition: Deceased",
                    "history_hours": 48.0,
                    "fallback_hours": 48.0,
                    "baseline_content": "Family says patient passed away.",
                    "baseline_events_count": 1,
                },
                "pre_icu_history_source": "reports",
                "pre_icu_history_items": 1,
            }
        ]


class StubConfig:
    llm_temperature = 0.1
    llm_max_tokens = 4096
    oracle_context_history_hours = 48.0
    oracle_context_future_hours = 48.0
    oracle_context_top_k_recommendations = 3
    oracle_use_discharge_summary_for_history = True
    oracle_num_discharge_summaries = 2
    oracle_relative_report_codes = ["NOTE_RADIOLOGYREPORT", "NOTE_DISCHARGESUMMARY"]
    oracle_pre_icu_history_hours = 48.0


def test_run_oracle_condition_suite_smoke(monkeypatch, tmp_path) -> None:
    FakeLLMClient.call_count = 0
    monkeypatch.setattr(oracle_module, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(suite_module, "MIMICDataParser", FakeParser)

    output_root = tmp_path / "oracle_suite"
    run_dir = suite_module.run_oracle_condition_suite(
        config=StubConfig(),
        events_path="fake_events.parquet",
        icu_stay_path="fake_icu.parquet",
        output_root=str(output_root),
        provider="openai",
        model="fake-model",
        current_window_hours=1.0,
        window_step_hours=2.0,
        include_pre_icu_data=True,
        n_survived=1,
        n_died=1,
        selection_seed=1,
        window_workers=1,
        conditions=None,
    )

    assert run_dir.exists()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert sorted(manifest["conditions"]) == ["full_visible", "masked_outcome", "reversed_outcome"]
    cohort = pd.read_csv(run_dir / "cohort_manifest.csv")
    assert len(cohort) == 2
    assert set(cohort["survived"].astype(bool).tolist()) == {True, False}

    for condition in ["full_visible", "masked_outcome", "reversed_outcome"]:
        condition_dir = run_dir / "conditions" / condition
        assert condition_dir.exists()
        assert (condition_dir / "processing_summary.json").exists()
        patient_files = list((condition_dir / "patients").glob("*/*/oracle_predictions.json"))
        if not patient_files:
            patient_files = list((condition_dir / "patients").glob("*/oracle_predictions.json"))
        assert patient_files

    full_calls = json.loads(
        next((run_dir / "conditions" / "full_visible" / "patients").glob("*/llm_calls.json")).read_text(encoding="utf-8")
    )
    masked_calls = json.loads(
        next((run_dir / "conditions" / "masked_outcome" / "patients").glob("*/llm_calls.json")).read_text(
            encoding="utf-8"
        )
    )
    reversed_calls = json.loads(
        next((run_dir / "conditions" / "reversed_outcome" / "patients").glob("*/llm_calls.json")).read_text(
            encoding="utf-8"
        )
    )

    full_prompt = full_calls["calls"][0]["prompt"]
    masked_prompt = masked_calls["calls"][0]["prompt"]
    reversed_prompt = reversed_calls["calls"][0]["prompt"]

    assert "ICU Outcome:" in full_prompt
    assert "Discharge Disposition: Expired" in full_prompt
    assert "ICU Outcome:" not in masked_prompt
    assert "[OUTCOME_MASKED]" in masked_prompt
    assert "Discharge Disposition: Expired" not in masked_prompt
    assert "ICU Outcome:" in reversed_prompt
    assert "[OUTCOME_MASKED]" in reversed_prompt

    reversed_predictions = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run_dir / "conditions" / "reversed_outcome" / "patients").glob("*/oracle_predictions.json")
    ]
    assert any(
        (item.get("trajectory_metadata", {}).get("true_survived") is False)
        and (item.get("trajectory_metadata", {}).get("prompt_survived") is True)
        for item in reversed_predictions
    )


def test_run_oracle_condition_suite_auto_resume(monkeypatch, tmp_path) -> None:
    FakeLLMClient.call_count = 0
    monkeypatch.setattr(oracle_module, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(suite_module, "MIMICDataParser", FakeParser)

    output_root = tmp_path / "oracle_suite_resume"
    run_dir = suite_module.run_oracle_condition_suite(
        config=StubConfig(),
        events_path="fake_events.parquet",
        icu_stay_path="fake_icu.parquet",
        output_root=str(output_root),
        provider="openai",
        model="fake-model",
        current_window_hours=1.0,
        window_step_hours=2.0,
        include_pre_icu_data=True,
        n_survived=1,
        n_died=1,
        selection_seed=1,
        window_workers=1,
        conditions=["full_visible"],
    )

    assert FakeLLMClient.call_count == 2

    patient_prediction_files = sorted(
        (run_dir / "conditions" / "full_visible" / "patients").glob("*/oracle_predictions.json")
    )
    assert len(patient_prediction_files) == 2
    interrupted_patient_dir = patient_prediction_files[0].parent
    for filename in ("oracle_predictions.json", "llm_calls.json", "window_contexts.json"):
        target = interrupted_patient_dir / filename
        assert target.exists()
        target.unlink()

    before_resume_calls = FakeLLMClient.call_count
    resumed_run_dir = suite_module.run_oracle_condition_suite(
        config=StubConfig(),
        events_path="fake_events.parquet",
        icu_stay_path="fake_icu.parquet",
        output_root=str(output_root),
        provider="openai",
        model="fake-model",
        current_window_hours=1.0,
        window_step_hours=2.0,
        include_pre_icu_data=True,
        n_survived=1,
        n_died=1,
        selection_seed=1,
        window_workers=1,
        conditions=["full_visible"],
    )

    assert resumed_run_dir == run_dir
    assert FakeLLMClient.call_count - before_resume_calls == 1
    summary = json.loads(
        (
            resumed_run_dir / "conditions" / "full_visible" / "processing_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary.get("patients_processed") == 2
    assert summary.get("patients_resumed") == 1
