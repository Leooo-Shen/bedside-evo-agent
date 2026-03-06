"""Pipeline tests for MedAgent using mocked LLM responses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.med_agent import MedAgent


class FakeLLMClient:
    def __init__(self, provider: str, model: str = None, api_key: str = None, temperature: float = 0.3, max_tokens: int = 4096):
        self.provider = provider
        self.model = model or "fake-model"
        self.memory_calls = 0

    def chat(self, prompt: str, response_format: str = "text", **kwargs):
        if "summarizing static ICU patient memory" in prompt:
            return {
                "content": '<response>{"static_summary":"Static profile: CKD with limited baseline labs unavailable."}</response>',
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

        if "updated_dynamic_memory" in prompt:
            self.memory_calls += 1
            if self.memory_calls == 1:
                content = {
                    "updated_dynamic_memory": {
                        "current_status": "Early shock state with vasopressor support.",
                        "active_problems": ["Septic shock", "AKI"],
                        "trends": ["Lactate elevated"],
                        "interventions_responses": ["Fluid bolus -> transient MAP increase"],
                        "patient_specific_patterns": ["Fluid responsiveness uncertain"],
                    },
                    "new_critical_events": [{"time": "2024-01-01 00:30:00", "event": "Norepinephrine started"}],
                }
            else:
                content = {
                    "updated_dynamic_memory": {
                        "current_status": "Hemodynamics improving with reduced pressor requirement.",
                        "active_problems": ["Septic shock", "AKI"],
                        "trends": ["Lactate down"],
                        "interventions_responses": ["NE titration -> MAP maintained"],
                        "patient_specific_patterns": ["Responds to cautious pressor weaning"],
                    },
                    "new_critical_events": [{"time": "2024-01-01 01:30:00", "event": "Lactate improved"}],
                }

            return {
                "content": f"<response>{json.dumps(content)}</response>",
                "usage": {"input_tokens": 30, "output_tokens": 12},
            }

        # Predictor
        return {
            "content": """
<response>
{
  "survival_prediction": {
    "outcome": "survive",
    "confidence": 0.81,
    "rationale": "Improving early trajectory with preserved perfusion"
  }
}
</response>
""",
            "usage": {"input_tokens": 40, "output_tokens": 15},
        }


def test_med_agent_end_to_end_with_mocked_llm(monkeypatch):
    monkeypatch.setattr("agents.med_agent.LLMClient", FakeLLMClient)

    agent = MedAgent(
        provider="openai",
        model="fake",
        enable_logging=True,
        observation_hours=12,
        use_llm_static_compression=True,
        memory_use_thinking=False,
        predictor_use_thinking=False,
    )

    trajectory = {
        "enter_time": "2024-01-01T00:00:00",
        "age_at_admission": 64,
        "gender": "M",
        "events": [
            {"time": "2023-12-30 12:00:00", "code": "DIAGNOSIS", "code_specifics": "CKD"},
            {"time": "2023-12-31 06:00:00", "code": "LAB_TEST", "code_specifics": "Creatinine", "numeric_value": 1.9},
            {"time": "2023-12-31 20:00:00", "code": "DRUG_PRESCRIPTION", "code_specifics": "Furosemide"},
            {"time": "2024-01-01 00:10:00", "code": "DIAGNOSIS", "code_specifics": "Septic shock"},
        ],
    }

    windows = [
        {
            "window_index": 0,
            "hours_since_admission": 0.0,
            "current_events": [
                {"time": "2024-01-01 00:20:00", "code": "VITALS", "code_specifics": "MAP", "numeric_value": 58}
            ],
        },
        {
            "window_index": 1,
            "hours_since_admission": 1.0,
            "current_events": [
                {"time": "2024-01-01 01:20:00", "code": "LAB_TEST", "code_specifics": "Lactate", "numeric_value": 2.1}
            ],
        },
    ]

    patient_metadata = {"age": 64, "gender": "M", "subject_id": 123, "icu_stay_id": 456}

    prediction, output = agent.run_patient_trajectory(
        windows=windows,
        patient_metadata=patient_metadata,
        trajectory=trajectory,
        verbose=False,
    )

    assert prediction["survival_prediction"]["outcome"] == "survive"
    assert output.static_memory.summary
    assert output.final_dynamic_memory.current_status.startswith("Hemodynamics improving")
    assert len(output.final_dynamic_memory.critical_events_log) == 2
    assert len(output.dynamic_memory_history) == 2

    stats = agent.get_statistics()
    assert stats["total_memory_calls"] == 2
    assert stats["total_predictor_calls"] == 1
    assert stats["total_static_compression_calls"] == 1
    assert len(agent.get_logs()) == 4  # static builder + 2 memory + predictor
