"""Test memory agent ablation mode."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.agent_fold import WorkingContext
from agents.agent_fold_multi import MultiAgent


def test_memory_agent_disabled():
    """Test that MultiAgent works with memory agent disabled."""
    # Create agent with memory agent disabled
    agent = MultiAgent(
        provider="openai",
        model="gpt-4o-mini",
        use_memory_agent=False,
        use_reflection_agent=False,
    )

    # Verify memory agent is not initialized
    assert not hasattr(agent, "memory_agent") or agent.memory_agent is None
    assert not hasattr(agent, "reflection_agent") or agent.reflection_agent is None
    assert agent.use_memory_agent is False
    assert agent.use_reflection_agent is False

    # Verify observer and predictor are still initialized
    assert agent.observer is not None
    assert agent.predictor is not None

    print("✓ Memory agent disabled mode works correctly")


def test_memory_agent_enabled():
    """Test that MultiAgent works with memory agent enabled."""
    # Create agent with memory agent enabled
    agent = MultiAgent(
        provider="openai",
        model="gpt-4o-mini",
        use_memory_agent=True,
        use_reflection_agent=False,
    )

    # Verify memory agent is initialized
    assert agent.memory_agent is not None
    assert agent.use_memory_agent is True

    # Verify all agents are initialized
    assert agent.observer is not None
    assert agent.predictor is not None

    print("✓ Memory agent enabled mode works correctly")


def test_statistics():
    """Test that statistics are properly initialized."""
    agent = MultiAgent(
        provider="openai",
        model="gpt-4o-mini",
        use_memory_agent=False,
    )

    stats = agent.get_statistics()
    assert "total_observer_calls" in stats
    assert "total_memory_calls" in stats
    assert "total_folds" in stats
    assert "total_appends" in stats

    print("✓ Statistics initialized correctly")


def test_observer_agent_disabled_uses_raw_event_passthrough():
    """When observer is disabled, process_window should pass raw events through unchanged."""
    agent = MultiAgent(
        provider="openai",
        model="gpt-4o-mini",
        use_observer_agent=False,
        use_memory_agent=False,
        use_reflection_agent=False,
        enable_logging=True,
    )

    assert agent.observer is None

    context = WorkingContext(patient_metadata={"age": 70, "gender": "F"})
    observer_parsed, memory_parsed, _ = agent.process_window(
        working_context=context,
        current_events=[
            {"time": "2144-01-01 00:00", "code_specifics": "Heart Rate", "numeric_value": 110},
            {"time": "2144-01-01 00:10", "code_specifics": "SpO2", "numeric_value": 91},
        ],
        window_index=0,
        hours_since_admission=0.0,
        observer_prompt="",
        memory_prompt="",
        precomputed_observer_output=None,
    )

    assert memory_parsed == {}
    assert observer_parsed == {}
    assert agent.total_observer_calls == 0
    assert len(agent.call_logs) == 0  # No synthetic observer-bypass logs


if __name__ == "__main__":
    test_memory_agent_disabled()
    test_memory_agent_enabled()
    test_statistics()
    test_observer_agent_disabled_uses_raw_event_passthrough()
    print("\n✓ All tests passed!")
