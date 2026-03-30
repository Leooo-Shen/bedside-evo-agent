"""Integration test for med_evo path in survival_experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments import survival_experiment


class FakeConfig:
    events_path = "fake_events.parquet"
    icu_stay_path = "fake_icu.parquet"

    agent_observation_hours = 12
    agent_current_window_hours = 1
    agent_window_step_hours = 1
    agent_include_pre_icu_data = False
    agent_use_discharge_summary_for_history = False
    agent_num_discharge_summaries = 2
    oracle_relative_report_codes = ["NOTE_RADIOLOGYREPORT"]
    oracle_pre_icu_history_hours = 48.0

    llm_provider = "openai"
    llm_model = "fake"
    llm_max_tokens = 256

    med_evo_max_working_windows = 3
    med_evo_max_events = 100
    med_evo_max_episodes = 20
    med_evo_max_insights = 5
    med_evo_insight_recency_tau = 4.0
    med_evo_insight_every_n_windows = 1


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
                {
                    "event_idx": 11,
                    "time": "2024-01-01 00:10:00",
                    "code": "VITALS",
                    "code_specifics": "MAP",
                    "numeric_value": 72,
                }
            ],
        }

    def create_time_windows(self, trajectory: dict, **kwargs):
        return [
            {
                "window_index": 0,
                "hours_since_admission": 0.0,
                "current_events": [
                    {
                        "event_id": 11,
                        "time": "2024-01-01 00:10:00",
                        "code": "VITALS",
                        "code_specifics": "MAP",
                        "numeric_value": 72,
                    }
                ],
                "history_events": [],
            }
        ]


class FakeMemoryState:
    def to_text(self):
        return "fake final memory state"

    def to_dict(self):
        return {"insights": [], "critical_events": []}


class FakeMemoryDB:
    def save(self, path: str):
        Path(path).write_text(json.dumps({"memory_snapshots": []}, indent=2))


class FakeMedEvoAgent:
    def __init__(self, *args, **kwargs):
        self.enable_logging = False
        self.call_logs = []
        self.llm_client = type("_FakeClient", (), {"provider": "openai", "model": "fake"})()

    def clear_logs(self):
        self.call_logs = []

    def run_patient_trajectory(self, windows, patient_metadata, verbose=True):
        prediction = {
            "survival_prediction": {
                "outcome": "survive",
                "confidence": 0.8,
                "rationale": "improving",
            }
        }
        return prediction, FakeMemoryState(), FakeMemoryDB()

    def get_statistics(self):
        return {
            "total_patients": 0,
            "total_tokens_used": 0,
            "total_event_calls": 0,
            "total_insight_calls": 0,
            "total_predictor_calls": 0,
            "total_grounding_rejections": 0,
            "total_insights_pruned": 0,
            "total_llm_calls": 0,
        }

    def get_logs(self):
        return []


def test_run_experiment_med_evo_path(monkeypatch, tmp_path):
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
    monkeypatch.setattr(survival_experiment, "MedEvoAgent", FakeMedEvoAgent)

    aggregate = survival_experiment.run_experiment(
        agent_type="med_evo",
        n_survived=1,
        n_died=0,
        verbose=False,
        enable_logging=False,
    )

    assert aggregate["agent_type"] == "med_evo"
    assert aggregate["total_patients"] == 1
    assert aggregate["correct_predictions"] == 1

    result_dirs = sorted((tmp_path / "experiment_results").glob("med_evo_*"))
    assert result_dirs, "Expected med_evo_* results directory"
    latest = result_dirs[-1]

    patient_dir = latest / "patients" / "101_202"
    assert (patient_dir / "prediction.json").exists()
    assert (patient_dir / "memory_database.json").exists()
    assert (patient_dir / "final_memory.json").exists()

    prediction_payload = json.loads((patient_dir / "prediction.json").read_text())
    assert prediction_payload["predicted_outcome_normalized"] == "survive"
