"""Test script for Reflection Agent functionality."""

import json
from agents.agent_multi import MultiAgent

def test_reflection_agent_initialization():
    """Test that reflection agent can be initialized."""
    # Without reflection agent
    agent_no_reflection = MultiAgent(
        provider="openai",
        model="gpt-4",
        use_reflection_agent=False,
    )
    assert not hasattr(agent_no_reflection, 'reflection_agent') or not agent_no_reflection.use_reflection_agent
    print("✓ MultiAgent without reflection agent initialized")

    # With reflection agent
    agent_with_reflection = MultiAgent(
        provider="openai",
        model="gpt-4",
        use_reflection_agent=True,
    )
    assert agent_with_reflection.use_reflection_agent
    assert hasattr(agent_with_reflection, 'reflection_agent')
    print("✓ MultiAgent with reflection agent initialized")

def test_statistics_tracking():
    """Test that reflection statistics are tracked."""
    agent = MultiAgent(
        provider="openai",
        model="gpt-4",
        use_reflection_agent=True,
    )

    stats = agent.get_statistics()
    assert "total_reflection_calls" in stats
    assert "total_revisions" in stats
    assert stats["total_reflection_calls"] == 0
    assert stats["total_revisions"] == 0
    print("✓ Reflection statistics tracked correctly")

def test_prompt_structure():
    """Test that reflection agent prompt is properly formatted."""
    from prompts.agent_multi_prompts import get_reflection_agent_prompt

    prompt = get_reflection_agent_prompt()
    assert "{previous_trajectory_text}" in prompt
    assert "{start_index}" in prompt
    assert "{end_index}" in prompt
    assert "{trajectory_summary}" in prompt
    assert "{raw_events_text}" in prompt
    assert "temporal_consistency" in prompt.lower()
    assert "evidence_grounding" in prompt.lower()
    assert "clinical_coherence" in prompt.lower()
    print("✓ Reflection agent prompt structure correct")

if __name__ == "__main__":
    print("Testing Reflection Agent Implementation...\n")

    try:
        test_reflection_agent_initialization()
        test_statistics_tracking()
        test_prompt_structure()

        print("\n✅ All tests passed!")
        print("\nReflection Agent is ready to use with:")
        print("  agent = MultiAgent(use_reflection_agent=True)")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
