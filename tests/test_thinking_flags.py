"""Test JSON-only prompt behavior for multi-agent system."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prompts.agent_multi_prompts import (  # noqa: E402
    get_memory_agent_prompt,
    get_observer_prompt,
    get_predictor_prompt,
    get_reflection_agent_prompt,
)


def _assert_json_only_prompt(prompt: str) -> None:
    assert "<think>" not in prompt
    assert "</think>" not in prompt
    assert "<response>" not in prompt
    assert "</response>" not in prompt
    assert "ALWAYS start with thinking first" not in prompt


def test_observer_prompt_json_only():
    """Observer prompt should be JSON-only with no think/response tags."""
    prompt = get_observer_prompt()
    _assert_json_only_prompt(prompt)
    assert "clinical_summary" in prompt

    print("✓ Observer prompt is JSON-only")


def test_memory_prompt_json_only():
    """Memory prompt should be JSON-only with no think/response tags."""
    prompt = get_memory_agent_prompt()
    _assert_json_only_prompt(prompt)
    assert "memory_management" in prompt

    print("✓ Memory prompt is JSON-only")


def test_memory_prompt_variable_formatting():
    """Test memory prompt variables are substituted correctly."""
    prompt = get_memory_agent_prompt().format(
        trajectory_text="TRAJECTORY",
        window_input='{"clinical_summary":"ok"}',
        window_index=3,
        num_trajectories=2,
    )
    assert "{trajectory_text}" not in prompt
    assert "{window_input}" not in prompt
    assert "{window_index}" not in prompt
    assert "{num_trajectories}" not in prompt
    assert "TRAJECTORY" in prompt
    assert "Window 3" in prompt
    assert '{"clinical_summary":"ok"}' in prompt
    assert "existing 2 trajectory entries" in prompt

    print("✓ Memory prompt variables are formatted correctly")


def test_reflection_prompt_json_only():
    """Reflection prompt should be JSON-only with no think/response tags."""
    prompt = get_reflection_agent_prompt()
    _assert_json_only_prompt(prompt)
    assert "audit_results" in prompt

    print("✓ Reflection prompt is JSON-only")


def test_reflection_prompt_variable_formatting():
    """Test reflection prompt variables are substituted correctly."""
    prompt = get_reflection_agent_prompt().format(
        previous_trajectory_text="PREV_TRAJ",
        start_index=1,
        end_index=3,
        start_hour=2.0,
        end_hour=6.5,
        trajectory_summary="SUMMARY",
        raw_events_text="RAW_EVENTS",
    )
    assert "{previous_trajectory_text}" not in prompt
    assert "{start_index}" not in prompt
    assert "{end_index}" not in prompt
    assert "{start_hour:.1f}" not in prompt
    assert "{end_hour:.1f}" not in prompt
    assert "{trajectory_summary}" not in prompt
    assert "{raw_events_text}" not in prompt
    assert "PREV_TRAJ" in prompt
    assert "Window 1 to 3" in prompt
    assert "Hour 2.0 to 6.5" in prompt
    assert "SUMMARY" in prompt
    assert "RAW_EVENTS" in prompt

    print("✓ Reflection prompt variables are formatted correctly")


def test_predictor_prompt_json_only():
    """Predictor prompt should be JSON-only with no think/response tags."""
    prompt_8h = get_predictor_prompt(observation_hours=8)
    _assert_json_only_prompt(prompt_8h)
    assert "first 8 hours after ICU admission" in prompt_8h
    formatted_prompt = prompt_8h.format(context="CONTEXT_OK")
    assert "CONTEXT_OK" in formatted_prompt
    assert "{context}" not in formatted_prompt

    print("✓ Predictor prompt is JSON-only")


def test_multi_agent_initialization_no_thinking_flags():
    """MultiAgent should initialize without legacy thinking flags."""
    from agents.agent_fold_multi import MultiAgent

    agent = MultiAgent(
        provider="openai",
        model="gpt-4o-mini",
    )

    assert not hasattr(agent, "observer_use_thinking")
    assert not hasattr(agent, "memory_use_thinking")
    assert not hasattr(agent, "reflection_use_thinking")
    assert not hasattr(agent, "predictor_use_thinking")

    print("✓ MultiAgent initializes without thinking flags")


def test_multi_agent_log_stores_parsed_response():
    """Test _log_call stores parsed_response."""
    from agents.agent_fold_multi import MultiAgent

    agent = MultiAgent(
        provider="openai",
        model="gpt-4o-mini",
        enable_logging=True,
    )
    expected = {"ok": True}
    agent._log_call(
        "observer",
        window_index=0,
        hours=0.0,
        prompt="prompt",
        response="response",
        usage={"input_tokens": 1, "output_tokens": 2},
        parsed_response=expected,
    )

    assert len(agent.call_logs) == 1
    assert agent.call_logs[0].parsed_response == expected

    print("✓ Parsed response is stored in logs")


if __name__ == "__main__":
    test_observer_prompt_json_only()
    test_memory_prompt_json_only()
    test_memory_prompt_variable_formatting()
    test_reflection_prompt_json_only()
    test_reflection_prompt_variable_formatting()
    test_predictor_prompt_json_only()
    test_multi_agent_initialization_no_thinking_flags()
    test_multi_agent_log_stores_parsed_response()
    print("\n✓ All JSON-only prompt tests passed!")
