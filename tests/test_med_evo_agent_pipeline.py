"""Pipeline tests for MedEvo agent with mocked LLM responses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.med_evo_agent import MedEvoAgent


class FakeLLMClient:
    def __init__(
        self,
        provider: str,
        model: str = None,
        api_key: str = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.provider = provider
        self.model = model or "fake-model"
        self.event_calls = 0
        self.insight_calls = 0

    def chat(self, prompt: str, response_format: str = "text", **kwargs):
        if "You are an ICU EventAgent" in prompt:
            self.event_calls += 1
            if self.event_calls == 1:
                payload = {
                    "critical_event_ids": [101, 9999],
                    "window_summary": {
                        "text": "Hypotension event requiring close monitoring.",
                        "supporting_event_ids": [101, 9999],
                    },
                }
                return {
                    "content": json.dumps(payload),
                    "usage": {"input_tokens": 20, "output_tokens": 10},
                }
            else:
                payload = {
                    "critical_event_ids": [201],
                    "window_summary": {
                        "text": "Perfusion trend improving.",
                        "supporting_event_ids": [201],
                    },
                }
                return {
                    "content": "raw-content-that-is-not-json",
                    "parsed": payload,
                    "usage": {"input_tokens": 20, "output_tokens": 10},
                }

        if ("You are an ICU InsightAgent" in prompt) or ("You are a clinical insight agent" in prompt):
            self.insight_calls += 1
            if self.insight_calls == 1:
                payload = {
                    "insight_updates": [],
                    "new_insights": [
                        {
                            "hypothesis": "Hypotension responds to early intervention.",
                            "supporting_event_ids": [101],
                            "counter_event_ids": [],
                        }
                    ],
                }
            else:
                payload = {
                    "insight_updates": [],
                    "new_insights": [
                        {
                            "hypothesis": "Improving lactate suggests perfusion recovery.",
                            "supporting_event_ids": [201],
                            "counter_event_ids": [],
                        }
                    ],
                }
            return {
                "content": f"<response>{json.dumps(payload)}</response>",
                "usage": {"input_tokens": 18, "output_tokens": 8},
            }

        payload = {
            "survival_prediction": {
                "outcome": "survive",
                "confidence": 0.82,
                "rationale": "Trajectory improved over observation windows.",
            },
            "supportive_factors": ["lactate down"],
            "risk_factors": ["initial hypotension"],
        }
        return {
            "content": f"<response>{json.dumps(payload)}</response>",
            "usage": {"input_tokens": 25, "output_tokens": 12},
        }


def test_med_evo_end_to_end_with_grounding_and_pruning(monkeypatch):
    monkeypatch.setattr("agents.med_evo_agent.LLMClient", FakeLLMClient)

    agent = MedEvoAgent(
        provider="openai",
        model="fake",
        enable_logging=True,
        observation_hours=12,
        window_duration_hours=0.5,
        max_working_windows=3,
        max_events=100,
        max_insights=1,
        insight_recency_tau=4.0,
    )

    windows = [
        {
            "window_index": 0,
            "hours_since_admission": 0.0,
            "current_events": [
                {
                    "event_id": 101,
                    "time": "2024-01-01 00:10:00",
                    "code": "VITALS",
                    "code_specifics": "MAP",
                    "numeric_value": 58,
                },
                {
                    "event_id": 102,
                    "time": "2024-01-01 00:20:00",
                    "code": "LAB_TEST",
                    "code_specifics": "Lactate",
                    "numeric_value": 4.1,
                },
            ],
        },
        {
            "window_index": 1,
            "hours_since_admission": 0.5,
            "current_events": [
                {
                    "event_id": 201,
                    "time": "2024-01-01 00:40:00",
                    "code": "LAB_TEST",
                    "code_specifics": "Lactate",
                    "numeric_value": 2.3,
                }
            ],
        },
    ]

    patient_metadata = {"age": 67, "gender": "F", "subject_id": 111, "icu_stay_id": 222}

    prediction, memory, memory_db = agent.run_patient_trajectory(
        windows=windows,
        patient_metadata=patient_metadata,
        verbose=False,
    )

    assert prediction["survival_prediction"]["outcome"] == "survive"
    assert len(memory_db.memory_snapshots) == 2

    # Invalid grounded IDs should be rejected.
    assert all(item.event_id != 9999 for item in memory.critical_events)
    assert agent.get_statistics()["total_grounding_rejections"] >= 1

    # Grounded critical events should keep the formatted event string.
    first_critical = next(item for item in memory.critical_events if item.event_id == 101)
    assert first_critical.name_str.startswith("[101]")
    assert "VITALS MAP" in first_critical.name_str
    assert "=58.00" in first_critical.name_str
    second_critical = next(item for item in memory.critical_events if item.event_id == 201)
    assert second_critical.name_str.startswith("[201]")
    assert "LAB_TEST Lactate" in second_critical.name_str

    # Insight cap must be respected and pruning tracked.
    assert len(memory.insights) == 1
    assert agent.get_statistics()["total_insights_pruned"] >= 1

    # Supporting IDs in trajectory summaries must be valid known IDs.
    known_ids = {101, 102, 201}
    summary_records = []
    for item in memory.trajectory_memory:
        if item.get("type") == "window_summary":
            assert set(item.get("supporting_event_ids", [])).issubset(known_ids)
            summary_records.append(item)

    assert summary_records
    for summary in summary_records:
        support_ids = summary.get("supporting_event_ids", [])
        support_events = summary.get("supporting_events", [])
        assert [event["event_id"] for event in support_events] == support_ids
        for event in support_events:
            assert event.get("event_name")
            assert "raw_event" in event

    stats = agent.get_statistics()
    assert stats["total_event_calls"] == 2
    assert stats["total_insight_calls"] == 2
    assert stats["total_predictor_calls"] == 1
    assert stats["total_event_name_mismatches"] == 0

    # Insight evidence should be stored as grounded {id, name_str} objects.
    assert memory_db.memory_snapshots
    for insight_dict in memory_db.memory_snapshots[-1].get("insights", []):
        for evidence_key in ("supporting_evidence", "counter_evidence"):
            for evidence in insight_dict.get(evidence_key, []):
                assert "id" in evidence
                assert "name_str" in evidence
                assert evidence["name_str"].startswith(f"[{evidence['id']}]")

    logs = agent.get_logs()
    event_prompts = [
        log.get("prompt", "") for log in logs if log.get("metadata", {}).get("step_type") == "event_agent"
    ]
    assert len(event_prompts) == 2
    assert "## History windows" in event_prompts[1]
    assert "### Window 0 (Hour 0.0-0.5)" in event_prompts[1]
    assert "## Current window observation" in event_prompts[1]
    assert "### Window 1 (Hour 0.5-1.0)" in event_prompts[1]

    insight_prompts = [
        log.get("prompt", "") for log in logs if log.get("metadata", {}).get("step_type") == "insight_agent"
    ]
    assert len(insight_prompts) == 2
    assert "Hypotension event requiring close monitoring." in insight_prompts[0]
    assert "[101] 2024-01-01 00:10 VITALS MAP =58.00" in insight_prompts[0]
    assert "{window_summary}" not in insight_prompts[0]
    assert "{critical_events}" not in insight_prompts[0]
    assert "id=101 name_str=" not in insight_prompts[0]
    assert "support=" not in insight_prompts[0]
    assert "counter=" not in insight_prompts[0]
    assert "[1] Hypotension responds to early intervention." in insight_prompts[1]

    assert len(logs) == 5


def test_med_evo_insight_agent_runs_every_n_windows(monkeypatch):
    monkeypatch.setattr("agents.med_evo_agent.LLMClient", FakeLLMClient)

    agent = MedEvoAgent(
        provider="openai",
        model="fake",
        enable_logging=True,
        observation_hours=12,
        window_duration_hours=0.5,
        max_working_windows=3,
        max_events=100,
        max_insights=5,
        insight_recency_tau=4.0,
        insight_every_n_windows=2,
    )

    windows = [
        {
            "window_index": 0,
            "hours_since_admission": 0.0,
            "current_events": [
                {
                    "event_id": 101,
                    "time": "2024-01-01 00:10:00",
                    "code": "VITALS",
                    "code_specifics": "MAP",
                    "numeric_value": 58,
                }
            ],
        },
        {
            "window_index": 1,
            "hours_since_admission": 0.5,
            "current_events": [
                {
                    "event_id": 201,
                    "time": "2024-01-01 00:40:00",
                    "code": "LAB_TEST",
                    "code_specifics": "Lactate",
                    "numeric_value": 2.3,
                }
            ],
        },
    ]

    prediction, memory, memory_db = agent.run_patient_trajectory(
        windows=windows,
        patient_metadata={"subject_id": 111, "icu_stay_id": 222},
        verbose=False,
    )

    assert prediction["survival_prediction"]["outcome"] == "survive"
    assert len(memory.insights) == 1
    assert len(memory_db.memory_snapshots) == 2

    first_snapshot = memory_db.memory_snapshots[0]
    second_snapshot = memory_db.memory_snapshots[1]
    assert len(first_snapshot.get("trajectory_memory", [])) == 1
    assert len(second_snapshot.get("trajectory_memory", [])) == 2
    assert len(first_snapshot.get("insights", [])) == 1
    assert len(second_snapshot.get("insights", [])) == 1

    stats = agent.get_statistics()
    assert stats["insight_every_n_windows"] == 2
    assert stats["total_event_calls"] == 2
    assert stats["total_insight_calls"] == 1
    assert stats["total_predictor_calls"] == 1
    assert len(agent.get_logs()) == 4


def test_med_evo_insight_prompt_batches_n_windows(monkeypatch):
    monkeypatch.setattr("agents.med_evo_agent.LLMClient", FakeLLMClient)

    agent = MedEvoAgent(
        provider="openai",
        model="fake",
        enable_logging=True,
        observation_hours=12,
        window_duration_hours=0.5,
        max_working_windows=3,
        max_events=100,
        max_insights=5,
        insight_recency_tau=4.0,
        insight_every_n_windows=2,
    )

    windows = [
        {
            "window_index": 0,
            "hours_since_admission": 0.0,
            "current_events": [
                {
                    "event_id": 101,
                    "time": "2024-01-01 00:10:00",
                    "code": "VITALS",
                    "code_specifics": "MAP",
                    "numeric_value": 58,
                }
            ],
        },
        {
            "window_index": 1,
            "hours_since_admission": 0.5,
            "current_events": [
                {
                    "event_id": 201,
                    "time": "2024-01-01 00:40:00",
                    "code": "LAB_TEST",
                    "code_specifics": "Lactate",
                    "numeric_value": 2.3,
                }
            ],
        },
        {
            "window_index": 2,
            "hours_since_admission": 1.0,
            "current_events": [
                {
                    "event_id": 301,
                    "time": "2024-01-01 01:10:00",
                    "code": "LAB_TEST",
                    "code_specifics": "Lactate",
                    "numeric_value": 1.9,
                }
            ],
        },
    ]

    prediction, memory, _ = agent.run_patient_trajectory(
        windows=windows,
        patient_metadata={"subject_id": 111, "icu_stay_id": 222},
        verbose=False,
    )

    assert prediction["survival_prediction"]["outcome"] == "survive"
    assert len(memory.insights) >= 1

    insight_prompts = [
        log.get("prompt", "")
        for log in agent.get_logs()
        if log.get("metadata", {}).get("step_type") == "insight_agent"
    ]
    assert len(insight_prompts) == 2

    second_prompt = insight_prompts[1]
    assert "### Window Summary:" in second_prompt
    assert "window 1 (hour 0.5-1.0): Perfusion trend improving." in second_prompt
    assert "window 2 (hour 1.0-1.5): Perfusion trend improving." in second_prompt
    assert "### Critical Events:" in second_prompt
    assert "[201] 2024-01-01 00:40 LAB_TEST Lactate =2.30" in second_prompt
