"""Integration-style test for run_oracle.py with Oracle."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import agents.oracle as oracle_module
from run_oracle import process_batch_for_oracle


class FakeLLMClient:
    def __init__(self, provider="openai", model=None, api_key=None, temperature=0.3, max_tokens=4096):
        self.provider = provider
        self.model = model or "fake-model"

    def chat(self, prompt: str, response_format: str = "text", **kwargs):
        parsed_payload = {
            "patient_status": {
                "domains": {
                    "hemodynamics": {"label": "stable", "key_signals": ["MAP stable"], "rationale": "stable"},
                    "respiratory": {"label": "stable", "key_signals": ["SpO2 stable"], "rationale": "stable"},
                    "renal_metabolic": {
                        "label": "insufficient_data",
                        "key_signals": ["limited labs"],
                        "rationale": "limited",
                    },
                    "neurology": {"label": "stable", "key_signals": ["no change"], "rationale": "stable"},
                },
                "overall": {"label": "stable", "rationale": "patient remained stable"},
            },
            "action_evaluations": [
                {
                    "action_id": "CW1",
                    "action_description": "Norepinephrine started",
                    "guideline_adherence": {
                        "label": "adherent",
                        "guideline_reference": "Surviving Sepsis Campaign",
                        "rationale": "Pressor support in hypotension.",
                    },
                    "contextual_appropriateness": {
                        "label": "appropriate",
                        "rationale": "Consistent with hemodynamic instability.",
                        "hindsight_caveat": None,
                    },
                }
            ],
            "overall_window_summary": "Hemodynamics stabilized after pressor initiation.",
        }
        return {
            # Simulate provider response without native parsed JSON.
            "content": f"<response>{json.dumps(parsed_payload)}</response>",
            "parsed": None,
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "model": self.model,
        }


class FakeParser:
    last_create_time_windows_kwargs = None

    def __init__(self, events_path: str, icu_stay_path: str):
        self.events_path = events_path
        self.icu_stay_path = icu_stay_path

    def load_data(self):
        return None

    def get_all_trajectories(self):
        enter = datetime(2024, 1, 1, 0, 0, 0)
        leave = enter + timedelta(hours=30)
        events = []
        for i in range(30):
            t = enter + timedelta(hours=i)
            events.append(
                {
                    "time": pd.Timestamp(t),
                    "code": "VITALS",
                    "code_specifics": f"MAP_{i}",
                    "numeric_value": 65,
                }
            )

        return [
            {
                "subject_id": 123,
                "icu_stay_id": 456,
                "icu_duration_hours": 30.0,
                "survived": True,
                "enter_time": pd.Timestamp(enter),
                "leave_time": pd.Timestamp(leave),
                "events": events,
            }
        ]

    def save_trajectories(self, trajectories, output_path: str):
        with open(output_path, "w") as f:
            for t in trajectories:
                f.write(json.dumps(t) + "\n")

    def create_time_windows(self, trajectory, **kwargs):
        FakeParser.last_create_time_windows_kwargs = dict(kwargs)
        return [
            {
                "subject_id": trajectory["subject_id"],
                "icu_stay_id": trajectory["icu_stay_id"],
                "current_window_start": "2024-01-01T00:00:00",
                "current_window_end": "2024-01-01T00:30:00",
                "hours_since_admission": 0.0,
                "patient_metadata": {"age": 70, "total_icu_duration_hours": 30.0},
                "current_events": [
                    {
                        "time": "2024-01-01 00:10:00",
                        "code": "DRUG_START",
                        "code_specifics": "Norepinephrine",
                    }
                ],
            }
        ]


class FakeBalancedParser:
    def __init__(self, events_path: str, icu_stay_path: str):
        self.events_path = events_path
        self.icu_stay_path = icu_stay_path
        self.icu_stay_df = None

    def load_data(self):
        self.icu_stay_df = pd.DataFrame(
            [
                {"subject_id": 201, "icu_stay_id": 9001, "survived": True},
                {"subject_id": 202, "icu_stay_id": 9002, "survived": False},
                {"subject_id": 203, "icu_stay_id": 9003, "survived": True},
            ]
        )

    def get_patient_trajectory(self, subject_id: int, icu_stay_id: int, icu_stay=None):
        survived = bool(icu_stay["survived"]) if icu_stay is not None else True
        enter = datetime(2024, 1, 1, 0, 0, 0)
        leave = enter + timedelta(hours=24)
        return {
            "subject_id": int(subject_id),
            "icu_stay_id": int(icu_stay_id),
            "icu_duration_hours": 24.0,
            "survived": survived,
            "enter_time": enter.isoformat(),
            "leave_time": leave.isoformat(),
            "events": [],
        }

    def create_time_windows(self, trajectory, **kwargs):
        return [
            {
                "subject_id": trajectory["subject_id"],
                "icu_stay_id": trajectory["icu_stay_id"],
                "current_window_start": "2024-01-01T00:00:00",
                "current_window_end": "2024-01-01T00:30:00",
                "hours_since_admission": 0.0,
                "patient_metadata": {"age": 70, "total_icu_duration_hours": 24.0},
                "current_events": [
                    {
                        "time": "2024-01-01 00:10:00",
                        "code": "DRUG_START",
                        "code_specifics": "Norepinephrine",
                    }
                ],
            }
        ]


class StubConfig:
    oracle_observation_hours = None
    oracle_use_discharge_summary_for_history = False
    oracle_num_discharge_summaries = 2
    oracle_relative_report_codes = ["NOTE_RADIOLOGYREPORT"]
    oracle_pre_icu_history_hours = 72.0
    oracle_context_history_hours = 48.0
    oracle_context_future_hours = 48.0
    oracle_context_use_discharge_summary = False
    oracle_context_include_icu_outcome_in_prompt = True
    oracle_context_top_k_recommendations = 3
    oracle_log_dir = "logs/oracle"


def test_process_batch_for_oracle(monkeypatch, tmp_path):
    monkeypatch.setattr(oracle_module, "LLMClient", FakeLLMClient)

    import run_oracle as run_oracle_module

    monkeypatch.setattr(run_oracle_module, "MIMICDataParser", FakeParser)

    output_dir = tmp_path / "oracle_outputs"
    cfg = StubConfig()
    cfg.oracle_log_dir = str(tmp_path / "logs")

    process_batch_for_oracle(
        config=cfg,
        events_path="fake_events.parquet",
        icu_stay_path="fake_icu.parquet",
        output_dir=str(output_dir),
        provider="openai",
        model="fake-model",
        current_window_hours=0.5,
        window_step_hours=0.5,
        include_pre_icu_data=False,
        include_icu_outcome_in_prompt=False,
        max_patients=1,
    )

    run_dirs = sorted(output_dir.glob("oracle_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert run_dir.name.endswith("_without_outcome")
    summary_file = run_dir / "processing_summary.json"
    patient_dir = run_dir / "patients" / "123_456"

    assert summary_file.exists()
    assert not (run_dir / "all_oracle_reports.json").exists()
    assert not (run_dir / "patient_trajectories.jsonl").exists()
    assert patient_dir.exists()

    predictions_file = patient_dir / "oracle_predictions.json"
    llm_calls_json = patient_dir / "llm_calls.json"
    llm_calls_html = patient_dir / "llm_calls.html"
    assert predictions_file.exists()
    assert llm_calls_json.exists()
    assert llm_calls_html.exists()

    predictions = json.loads(predictions_file.read_text())
    assert predictions.get("subject_id") == 123
    assert predictions.get("icu_stay_id") == 456
    assert predictions.get("window_outputs")
    first = predictions["window_outputs"][0]["oracle_output"]
    assert "patient_status" in first
    assert "action_evaluations" in first
    assert "overall_window_summary" in first
    assert "doctor_actions" not in first
    assert "clinical_quality" not in first

    llm_calls = json.loads(llm_calls_json.read_text())
    assert llm_calls.get("patient_id") == "123_456"
    assert llm_calls.get("include_icu_outcome_in_prompt") is False
    assert llm_calls.get("prompt_outcome_mode") == "without_icu_outcome"
    assert llm_calls.get("total_calls", 0) >= 1
    assert llm_calls.get("calls")
    assert first == llm_calls["calls"][0]["parsed_response"]
    assert "ICU Outcome:" not in llm_calls["calls"][0]["prompt"]
    assert any(
        call.get("metadata", {}).get("use_discharge_summary") is False
        for call in llm_calls.get("calls", [])
    )
    assert any(
        call.get("metadata", {}).get("include_icu_outcome_in_prompt") is False
        for call in llm_calls.get("calls", [])
    )
    assert any(
        call.get("metadata", {}).get("parse_source") == "best_effort_json"
        for call in llm_calls.get("calls", [])
    )

    parser_kwargs = FakeParser.last_create_time_windows_kwargs
    assert parser_kwargs is not None
    assert parser_kwargs.get("relative_report_codes") == ["NOTE_RADIOLOGYREPORT"]
    assert parser_kwargs.get("pre_icu_history_hours") == 72.0

    summary = json.loads(summary_file.read_text())
    assert summary.get("history_context_hours") == 48.0
    assert summary.get("future_context_hours") == 48.0
    assert summary.get("use_discharge_summary") is False
    assert summary.get("include_icu_outcome_in_prompt") is False
    assert summary.get("top_k_recommendations") == 3
    assert str(run_dir) == summary.get("run_directory")


def test_process_batch_for_oracle_balanced_sampling(monkeypatch, tmp_path):
    monkeypatch.setattr(oracle_module, "LLMClient", FakeLLMClient)

    import run_oracle as run_oracle_module

    monkeypatch.setattr(run_oracle_module, "MIMICDataParser", FakeBalancedParser)

    selection_calls = {}

    def _fake_select_balanced_patients(icu_stay_df, n_survived, n_died, random_seed):
        selection_calls["n_survived"] = n_survived
        selection_calls["n_died"] = n_died
        selection_calls["random_seed"] = random_seed
        survived_rows = icu_stay_df[icu_stay_df["survived"] == True].head(n_survived)
        died_rows = icu_stay_df[icu_stay_df["survived"] == False].head(n_died)
        return pd.concat([survived_rows, died_rows]).reset_index(drop=True)

    monkeypatch.setattr(run_oracle_module, "select_balanced_patients", _fake_select_balanced_patients)

    output_dir = tmp_path / "oracle_outputs"
    cfg = StubConfig()
    cfg.oracle_log_dir = str(tmp_path / "logs")

    process_batch_for_oracle(
        config=cfg,
        events_path="fake_events.parquet",
        icu_stay_path="fake_icu.parquet",
        output_dir=str(output_dir),
        provider="openai",
        model="fake-model",
        current_window_hours=0.5,
        window_step_hours=0.5,
        include_pre_icu_data=False,
        max_patients=1,  # ignored when balanced selection is enabled
        n_survived=1,
        n_died=1,
        selection_seed=7,
    )

    assert selection_calls == {"n_survived": 1, "n_died": 1, "random_seed": 7}

    run_dirs = sorted(output_dir.glob("oracle_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert run_dir.name.endswith("_with_outcome")
    summary_file = run_dir / "processing_summary.json"
    assert summary_file.exists()

    summary = json.loads(summary_file.read_text())
    assert summary.get("total_patients") == 2
    assert summary.get("patients_processed") == 2
    assert summary.get("patients_failed") == 0

    patient_dirs = sorted((run_dir / "patients").iterdir())
    assert len(patient_dirs) == 2
    for patient_dir in patient_dirs:
        assert (patient_dir / "oracle_predictions.json").exists()
        assert (patient_dir / "llm_calls.json").exists()
