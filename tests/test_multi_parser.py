"""Tests for robust JSON parsing in multi-agent responses."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.agent_fold_multi import _parse_json_response


def test_parse_direct_json():
    """Parser should parse direct JSON payloads."""
    response = """
{
  "clinical_summary": "stable"
}
"""
    parsed = _parse_json_response(response)
    assert parsed["clinical_summary"] == "stable"


def test_parse_markdown_json_fence():
    """Parser should recover when model wraps JSON in fenced code blocks."""
    response = """
Here is the result:
```json
{
  "memory_management": {
    "decision": "APPEND"
  }
}
```
"""
    parsed = _parse_json_response(response)
    assert parsed["memory_management"]["decision"] == "APPEND"


if __name__ == "__main__":
    test_parse_direct_json()
    test_parse_markdown_json_fence()
    print("All parser tests passed.")
