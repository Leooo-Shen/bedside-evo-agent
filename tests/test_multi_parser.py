"""Tests for robust JSON parsing in multi-agent responses."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.agent_multi import _parse_json_response


def test_parse_multiple_response_blocks():
    """Parser should ignore empty response blocks and parse the valid one."""
    response = """
<response></response>
<response>
{
  "clinical_summary": "stable"
}
</response>
"""
    parsed = _parse_json_response(response)
    assert parsed["clinical_summary"] == "stable"


def test_parse_without_response_tag_after_think():
    """Parser should recover when model omits <response> and returns plain JSON."""
    response = """
<think>Quick analysis.</think>
{
  "memory_management": {
    "decision": "APPEND"
  }
}
"""
    parsed = _parse_json_response(response)
    assert parsed["memory_management"]["decision"] == "APPEND"


if __name__ == "__main__":
    test_parse_multiple_response_blocks()
    test_parse_without_response_tag_after_think()
    print("All parser tests passed.")
