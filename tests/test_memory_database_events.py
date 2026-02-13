"""Test that raw events are saved to memory database."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent_fold import FoldAgent, MemoryDatabase, WorkingContext


def test_memory_database_stores_raw_events():
    """Test that memory database stores raw events for each window."""

    # Create agent
    agent = FoldAgent(window_duration_hours=0.5)

    # Create memory database
    memory_db = MemoryDatabase(patient_metadata={"age": 65, "gender": "M"})

    # Simulate adding a window with raw events
    raw_events = [
        {"time": "2023-01-01 10:00", "code": "VITAL_SIGN", "value": 120},
        {"time": "2023-01-01 10:15", "code": "LAB_RESULT", "value": 7.4},
        {"time": "2023-01-01 10:30", "code": "MEDICATION", "value": "Aspirin"},
    ]

    window_record = {
        "window_index": 0,
        "hours_since_admission": 0.0,
        "num_events": len(raw_events),
        "raw_events": raw_events,
        "llm_response": {"test": "data"},
        "trajectory_state": {"num_trajectories": 1, "trajectories": []},
        "key_events": [],
        "active_concerns": [],
    }

    memory_db.add_window(window_record)

    # Verify the raw events are stored
    assert len(memory_db.window_records) == 1, "Should have 1 window record"
    stored_record = memory_db.window_records[0]

    assert "raw_events" in stored_record, "Window record should have raw_events field"
    assert len(stored_record["raw_events"]) == 3, "Should have 3 raw events"
    assert stored_record["raw_events"][0]["code"] == "VITAL_SIGN", "First event should be VITAL_SIGN"
    assert stored_record["raw_events"][1]["code"] == "LAB_RESULT", "Second event should be LAB_RESULT"
    assert stored_record["raw_events"][2]["code"] == "MEDICATION", "Third event should be MEDICATION"

    print("✓ Memory database correctly stores raw events")

    # Test serialization
    db_dict = memory_db.to_dict()
    assert "window_records" in db_dict, "Should have window_records in dict"
    assert len(db_dict["window_records"]) == 1, "Should have 1 window record in dict"
    assert "raw_events" in db_dict["window_records"][0], "Window record in dict should have raw_events"

    print("✓ Memory database serialization includes raw events")

    print("\n✓ All memory database tests passed!")


if __name__ == "__main__":
    test_memory_database_stores_raw_events()
