"""Test thinking flags for multi-agent system."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prompts.agent_multi_prompts import (
    get_memory_agent_prompt,
    get_observer_prompt,
    get_predictor_prompt,
    get_reflection_agent_prompt,
)


def test_observer_thinking():
    """Test observer prompt with and without thinking."""
    # With thinking
    prompt_with = get_observer_prompt(use_thinking=True)
    assert "<think>" in prompt_with
    assert "</think>" in prompt_with
    assert "ALWAYS start with thinking first" in prompt_with

    # Without thinking
    prompt_without = get_observer_prompt(use_thinking=False)
    assert "<think>" not in prompt_without
    assert "</think>" not in prompt_without
    assert "ALWAYS start with thinking first" not in prompt_without
    assert "<response>" in prompt_without

    print("✓ Observer thinking flags work correctly")


def test_memory_thinking():
    """Test memory agent prompt with and without thinking."""
    # With thinking
    prompt_with = get_memory_agent_prompt(use_thinking=True)
    assert "<think>" in prompt_with
    assert "</think>" in prompt_with

    # Without thinking
    prompt_without = get_memory_agent_prompt(use_thinking=False)
    assert "<think>" not in prompt_without
    assert "</think>" not in prompt_without

    print("✓ Memory agent thinking flags work correctly")


def test_memory_prompt_variable_formatting():
    """Test memory prompt variables are substituted correctly."""
    prompt = get_memory_agent_prompt(use_thinking=False).format(
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


def test_reflection_thinking():
    """Test reflection agent prompt with and without thinking."""
    # With thinking
    prompt_with = get_reflection_agent_prompt(use_thinking=True)
    assert "<think>" in prompt_with
    assert "</think>" in prompt_with

    # Without thinking
    prompt_without = get_reflection_agent_prompt(use_thinking=False)
    assert "<think>" not in prompt_without
    assert "</think>" not in prompt_without

    print("✓ Reflection agent thinking flags work correctly")


def test_reflection_prompt_variable_formatting():
    """Test reflection prompt variables are substituted correctly."""
    prompt = get_reflection_agent_prompt(use_thinking=False).format(
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


def test_predictor_thinking():
    """Test predictor prompt with and without thinking."""
    # With thinking
    prompt_with = get_predictor_prompt(use_thinking=True)
    assert "<think>" in prompt_with
    assert "</think>" in prompt_with

    # Without thinking
    prompt_without = get_predictor_prompt(use_thinking=False)
    assert "<think>" not in prompt_without
    assert "</think>" not in prompt_without

    prompt_8h = get_predictor_prompt(use_thinking=False, observation_hours=8)
    assert "first 8 hours after ICU admission" in prompt_8h
    formatted_prompt = prompt_8h.format(context="CONTEXT_OK")
    assert "CONTEXT_OK" in formatted_prompt
    assert "{context}" not in formatted_prompt

    print("✓ Predictor thinking flags work correctly")


def test_multi_agent_initialization():
    """Test MultiAgent initialization with thinking flags."""
    from agents.agent_multi import MultiAgent

    # Test with thinking disabled
    agent = MultiAgent(
        provider="openai",
        model="gpt-4o-mini",
        observer_use_thinking=False,
        memory_use_thinking=False,
        reflection_use_thinking=False,
        predictor_use_thinking=False,
    )

    assert agent.observer_use_thinking is False
    assert agent.memory_use_thinking is False
    assert agent.reflection_use_thinking is False
    assert agent.predictor_use_thinking is False

    print("✓ MultiAgent thinking flags initialized correctly")


def test_multi_agent_log_stores_parsed_response():
    """Test _log_call stores parsed_response."""
    from agents.agent_multi import MultiAgent

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
    test_observer_thinking()
    test_memory_thinking()
    test_memory_prompt_variable_formatting()
    test_reflection_thinking()
    test_reflection_prompt_variable_formatting()
    test_predictor_thinking()
    test_multi_agent_initialization()
    test_multi_agent_log_stores_parsed_response()
    print("\n✓ All thinking flag tests passed!")
