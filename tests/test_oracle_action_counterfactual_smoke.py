"""Smoke test for experiments.oracle.run_oracle_action_counterfactual."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import agents.oracle as oracle_module
import experiments.oracle.run_oracle_action_counterfactual as q3_module


class FakeLLMClient:
    def __init__(self, provider="openai", model=None, api_key=None, temperature=0.0, max_tokens=1024):
        self.provider = provider
        self.model = model or "fake-model"

    def chat(self, prompt: str, response_format: str = "text", **kwargs):
        parsed = {
            "patient_status": {
                "domains": {
                    "hemodynamics": {"label": "deteriorating", "rationale": "drop"},
                    "respiratory": {"label": "stable", "rationale": "stable"},
                    "renal_metabolic": {"label": "stable", "rationale": "stable"},
                    "neurology": {"label": "stable", "rationale": "stable"},
                },
                "overall": {"label": "deteriorating", "rationale": "trend"},
            },
            "action_evaluations": [
                {
                    "action_id": "1004",
                    "action_description": "Counterfactual wrong action",
                    "guideline_adherence": {"label": "non_adherent", "rationale": "wrong"},
                    "contextual_appropriateness": {"label": "potentially_harmful", "rationale": "wrong"},
                    "overall": {"label": "potentially_harmful", "rationale": "wrong"},
                }
            ],
            "recommendations": [
                {
                    "rank": 1,
                    "action": "Increase norepinephrine",
                    "action_description": "Maintain MAP > 65",
                    "rationale": "shock",
                    "urgency": "immediate",
                }
            ],
            "overall_window_summary": "Synthetic",
        }
        return {
            "content": json.dumps(parsed),
            "parsed": parsed,
            "usage": {"input_tokens": 30, "output_tokens": 20},
            "model": self.model,
        }


class FakeParser:
    def __init__(self, events_path: str, icu_stay_path: str, **kwargs):
        self.events_path = events_path
        self.icu_stay_path = icu_stay_path
        self.icu_stay_df = None

    def load_data(self):
        self.icu_stay_df = pd.DataFrame(
            [
                {"subject_id": 101, "icu_stay_id": 9001, "survived": True},
                {"subject_id": 102, "icu_stay_id": 9002, "survived": False},
            ]
        )

    def get_patient_trajectory(self, subject_id: int, icu_stay_id: int, icu_stay=None):
        enter = datetime(2024, 1, 1, 0, 0, 0)
        leave = enter + timedelta(hours=10)
        return {
            "subject_id": int(subject_id),
            "icu_stay_id": int(icu_stay_id),
            "survived": bool(icu_stay["survived"]) if icu_stay is not None else True,
            "enter_time": enter.isoformat(),
            "leave_time": leave.isoformat(),
            "death_time": None,
            "icu_duration_hours": 10.0,
            "age_at_admission": 65,
            "events": [
                {
                    "time": (enter + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
                    "code": "VITALS",
                    "code_specifics": "MAP",
                    "numeric_value": 58,
                }
            ],
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
                    "age": 65,
                    "survived": trajectory["survived"],
                    "death_time": trajectory["death_time"],
                    "total_icu_duration_hours": 10.0,
                },
                "history_events": [],
                "current_events": [
                    {
                        "event_id": 1001,
                        "time": (enter + timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S"),
                        "code": "DRUG_START",
                        "code_specifics": "Norepinephrine infusion",
                    },
                    {
                        "event_id": 1002,
                        "time": (enter + timedelta(minutes=4)).strftime("%Y-%m-%d %H:%M:%S"),
                        "code": "PROCEDURE",
                        "code_specifics": "Arterial line insertion",
                    },
                    {
                        "event_id": 1003,
                        "time": (enter + timedelta(minutes=8)).strftime("%Y-%m-%d %H:%M:%S"),
                        "code": "TRANSFER",
                        "code_specifics": "Transfer to ICU",
                    },
                ],
                "num_history_events": 0,
                "num_current_events": 3,
                "pre_icu_history": {},
                "pre_icu_history_source": "none",
                "pre_icu_history_items": 0,
            }
        ]


class StubConfig:
    events_path = "events.parquet"
    icu_stay_path = "icu.parquet"
    llm_provider = "openai"
    llm_model = "fake-model"
    llm_temperature = 0.0
    llm_max_tokens = 1024
    oracle_context_history_hours = 12.0
    oracle_context_future_hours = 12.0
    oracle_context_top_k_recommendations = 3
    oracle_use_discharge_summary_for_history = True
    oracle_num_discharge_summaries = 2
    oracle_relative_report_codes = ["NOTE_DISCHARGESUMMARY"]
    oracle_pre_icu_history_hours = 48.0


def _write_baseline_patient(run_dir: Path, subject_id: int, icu_stay_id: int) -> None:
    patient_dir = run_dir / "conditions" / "full_visible" / "patients" / f"{subject_id}_{icu_stay_id}"
    patient_dir.mkdir(parents=True, exist_ok=True)

    baseline_payload = {
        "subject_id": subject_id,
        "icu_stay_id": icu_stay_id,
        "trajectory_metadata": {"true_survived": subject_id == 101},
        "window_outputs": [
            {
                "window_index": 1,
                "window_metadata": {"hours_since_admission": 0.0},
                "raw_current_events": [
                    {"code": "DRUG_START", "code_specifics": "Norepinephrine infusion"},
                    {"code": "PROCEDURE", "code_specifics": "Arterial line insertion"},
                    {"code": "TRANSFER", "code_specifics": "Transfer to ICU"},
                ],
                "oracle_output": {
                    "action_evaluations": [
                        {
                            "action_id": "1004",
                            "action_description": "Would correspond to injected position",
                            "overall": {"label": "appropriate"},
                        }
                    ]
                },
            }
        ],
    }

    with open(patient_dir / "oracle_predictions.json", "w", encoding="utf-8") as f:
        json.dump(baseline_payload, f)


def test_run_oracle_action_counterfactual_smoke(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(oracle_module, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(q3_module, "MIMICDataParser", FakeParser)

    run_dir = tmp_path / "oracle_conditions_20260101_000000"
    (run_dir / "conditions" / "full_visible" / "patients").mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "run_id": "oracle_conditions_20260101_000000",
        "events_path": "events.parquet",
        "icu_stay_path": "icu.parquet",
        "provider": "openai",
        "model": "fake-model",
        "current_window_hours": 1.0,
        "window_step_hours": 2.0,
        "include_pre_icu_data": True,
    }
    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f)

    cohort_df = pd.DataFrame(
        [
            {"subject_id": 101, "icu_stay_id": 9001, "survived": True, "outcome": "survived"},
            {"subject_id": 102, "icu_stay_id": 9002, "survived": False, "outcome": "died"},
        ]
    )
    cohort_df.to_csv(run_dir / "cohort_manifest.csv", index=False)

    _write_baseline_patient(run_dir, 101, 9001)
    _write_baseline_patient(run_dir, 102, 9002)

    output_dir = q3_module.run_counterfactual_action_experiment(
        run_dir=run_dir,
        config=StubConfig(),
        provider=None,
        model=None,
        min_action_events=3,
        max_patients=20,
    )

    assert output_dir.exists()
    assert (output_dir / "injection_manifest.csv").exists()
    assert (output_dir / "injection_manifest.json").exists()
    assert (output_dir / "counterfactual_llm_calls.jsonl").exists()
    assert (output_dir / "q3_window_results.csv").exists()
    assert (output_dir / "q3_summary.json").exists()

    q3_df = pd.read_csv(output_dir / "q3_window_results.csv")
    assert len(q3_df) == 2
    assert q3_df["counterfactual_label"].eq("potentially_harmful").all()

    summary = json.loads((output_dir / "q3_summary.json").read_text(encoding="utf-8"))
    assert summary["num_patients"] == 2
    assert summary["passes_primary"] is True

    jsonl_lines = (output_dir / "counterfactual_llm_calls.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 2
