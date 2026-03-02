"""Regression tests for token accounting in MultiAgent logging."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.agent_multi import MultiAgent, _normalize_token_count


def test_normalize_token_count_handles_none_and_strings():
    """Token normalization should coerce common malformed values safely."""
    assert _normalize_token_count(None) == 0
    assert _normalize_token_count("12") == 12
    assert _normalize_token_count("7.9") == 7
    assert _normalize_token_count("") == 0


def test_log_call_does_not_crash_on_none_output_tokens():
    """_log_call should treat None output tokens as zero instead of crashing."""
    agent = MultiAgent.__new__(MultiAgent)
    agent.total_tokens_used = 5
    agent.enable_logging = False

    agent._log_call(
        step_type="predictor",
        window_index=-1,
        hours=10.0,
        prompt="prompt",
        response="response",
        usage={"input_tokens": 13, "output_tokens": None},
    )

    assert agent.total_tokens_used == 18
