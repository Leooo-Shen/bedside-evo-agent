"""Tests for deterministic llm_calls ordering in run_oracle."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from run_oracle import _build_oracle_llm_calls_payload, _sort_llm_calls


def test_sort_llm_calls_orders_by_window_then_hour_then_timestamp() -> None:
    calls = [
        {"window_index": 2, "hours_since_admission": 1.0, "timestamp": "2026-03-09T12:00:05"},
        {"window_index": 1, "hours_since_admission": 0.7, "timestamp": "2026-03-09T12:00:04"},
        {"window_index": 1, "hours_since_admission": 0.5, "timestamp": "2026-03-09T12:00:02"},
        {"window_index": -1, "hours_since_admission": 0.1, "timestamp": "2026-03-09T12:00:01"},
    ]

    sorted_calls = _sort_llm_calls(calls)
    assert [call["window_index"] for call in sorted_calls] == [1, 1, 2, -1]
    assert [call["hours_since_admission"] for call in sorted_calls] == [0.5, 0.7, 1.0, 0.1]


def test_build_oracle_llm_calls_payload_sorts_calls() -> None:
    payload = _build_oracle_llm_calls_payload(
        subject_id=123,
        icu_stay_id=456,
        provider="openai",
        model="gpt-test",
        include_icu_outcome_in_prompt=True,
        calls=[
            {"step_type": "oracle_evaluator", "window_index": 3, "hours_since_admission": 1.5},
            {"step_type": "oracle_evaluator", "window_index": 1, "hours_since_admission": 0.5},
            {"step_type": "oracle_evaluator", "window_index": 2, "hours_since_admission": 1.0},
        ],
    )

    assert [call["window_index"] for call in payload["calls"]] == [1, 2, 3]
