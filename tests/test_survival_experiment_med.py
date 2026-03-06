"""Integration tests for med agent path in survival_experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.med_agent import DynamicMemory, DynamicMemorySnapshot, MedAgentOutput, StaticMemory
from experiments import survival_experiment


class FakeConfig:
    events_path = "fake_events.parquet"
    icu_stay_path = "fake_icu.parquet"

    agent_observation_hours = 12
    agent_current_window_hours = 1
    agent_window_step_hours = 1
    agent_include_pre_icu_data = False

    llm_provider = "openai"
    llm_model = "fake"
    llm_temperature = 0.1
    llm_max_tokens = 256

    med_agent_use_llm_static_compression = True
    med_agent_baseline_lab_lookback_start_hours = 72
    med_agent_baseline_lab_lookback_end_hours = 24
    med_agent_max_active_problems = 8
    med_agent_max_critical_events = 20
    med_agent_max_patterns = 8
    med_agent_memory_use_thinking = False
    med_agent_predictor_use_thinking = False

    # fields used by non-med branches but read by module-level helper paths
    agent_multi_use_observer_agent = False
    agent_multi_observer_cache_enabled = False
    agent_multi_observer_cache_dir = "experiment_results/observer_cache"


class FakeParser:
    def __init__(self, events_path: str, icu_stay_path: str):
        self.events_path = events_path
        self.icu_stay_path = icu_stay_path
        self.icu_stay_df = pd.DataFrame(
            [
                {
                    "subject_id": 101,
                    "icu_stay_id": 202,
                    "survived": True,
                    "icu_duration_hours": 48.0,
                }
            ]
        )

    def load_data(self):
        return None

    def get_patient_trajectory(self, subject_id: int, icu_stay_id: int):
        return {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "enter_time": "2024-01-01T00:00:00",
            "leave_time": "2024-01-03T00:00:00",
            "age_at_admission": 70,
            "gender": "M",
            "icu_duration_hours": 48.0,
            "survived": True,
            "death_time": None,
            "events": [
                {"time": "2023-12-31 00:00:00", "code": "DIAGNOSIS", "code_specifics": "CKD"},
                {"time": "2024-01-01 00:10:00", "code": "VITALS", "code_specifics": "MAP", "numeric_value": 72},
            ],
        }

    def create_time_windows(self, trajectory: dict, **kwargs):
        return [
            {
                "window_index": 0,
                "hours_since_admission": 0.0,
                "current_events": [
                    {"time": "2024-01-01 00:10:00", "code": "VITALS", "code_specifics": "MAP", "numeric_value": 72}
                ],
                "history_events": [],
            }
        ]


class FakeMedAgent:
    def __init__(self, *args, **kwargs):
        self.enable_logging = False
        self.call_logs = []
        self.llm_client = type("_FakeClient", (), {"provider": "openai", "model": "fake"})()

    def clear_logs(self):
        self.call_logs = []

    def run_patient_trajectory(self, windows, patient_metadata, trajectory, verbose=True):
        static_memory = StaticMemory(
            age=70,
            gender="M",
            admission_diagnoses=["Septic shock"],
            summary="Static summary",
        )
        dynamic_memory = DynamicMemory(
            current_status="Stable",
            active_problems=["Septic shock"],
            critical_events_log=[{"time": "2024-01-01 00:10:00", "event": "MAP stable"}],
            trends=["MAP stable"],
            interventions_responses=["Fluids -> improved perfusion"],
            patient_specific_patterns=["Responds to fluids"],
        )
        history = [
            DynamicMemorySnapshot(
                window_index=0,
                hours_since_admission=0.0,
                num_current_events=1,
                dynamic_memory=dynamic_memory,
            )
        ]
        output = MedAgentOutput(
            static_memory=static_memory,
            final_dynamic_memory=dynamic_memory,
            dynamic_memory_history=history,
            last_window_events=windows[-1]["current_events"],
            final_state_text="final state",
        )
        prediction = {
            "survival_prediction": {
                "outcome": "survive",
                "confidence": 0.8,
                "rationale": "improving",
            }
        }
        return prediction, output

    def get_statistics(self):
        return {
            "total_patients": 0,
            "total_tokens_used": 0,
            "total_memory_calls": 0,
            "total_predictor_calls": 0,
            "total_static_compression_calls": 0,
            "total_dynamic_fallbacks": 0,
            "total_llm_calls": 0,
        }

    def get_logs(self):
        return []


def test_run_experiment_med_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(survival_experiment, "get_config", lambda: FakeConfig())
    monkeypatch.setattr(survival_experiment, "MIMICDataParser", FakeParser)
    monkeypatch.setattr(
        survival_experiment,
        "select_balanced_patients",
        lambda _df, n_survived, n_died: pd.DataFrame(
            [
                {
                    "subject_id": 101,
                    "icu_stay_id": 202,
                    "survived": True,
                    "icu_duration_hours": 48.0,
                }
            ]
        ),
    )
    monkeypatch.setattr(survival_experiment, "MedAgent", FakeMedAgent)

    aggregate = survival_experiment.run_experiment(
        agent_type="med",
        n_survived=1,
        n_died=0,
        verbose=False,
        enable_logging=False,
    )

    assert aggregate["agent_type"] == "med"
    assert aggregate["total_patients"] == 1
    assert aggregate["correct_predictions"] == 1

    result_dirs = sorted((tmp_path / "experiment_results").glob("med_*"))
    assert result_dirs, "Expected med_* results directory"
    latest = result_dirs[-1]

    patient_dir = latest / "patients" / "101_202"
    assert (patient_dir / "prediction.json").exists()
    assert (patient_dir / "patient_memory.json").exists()
    assert (patient_dir / "dynamic_memory_history.json").exists()

    prediction_payload = json.loads((patient_dir / "prediction.json").read_text())
    assert prediction_payload["predicted_outcome_normalized"] == "survive"
