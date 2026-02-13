"""Test window hour calculation in FoldAgent."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent_fold import FoldAgent, WorkingContext, TrajectoryEntry


def test_window_hour_calculation():
    """Test that _get_window_start_hour calculates hours correctly."""

    # Create agent with 0.5 hour windows
    agent = FoldAgent(window_duration_hours=0.5)

    # Create working context with some trajectories
    context = WorkingContext()

    # Add T0 (window 0)
    t0 = TrajectoryEntry(
        start_window=0,
        end_window=0,
        start_hour=0.0,
        end_hour=0.5,
        summary="Window 0"
    )
    context.trajectory.append(t0)

    # Add T2-4 (merged windows 2-4)
    t2_4 = TrajectoryEntry(
        start_window=2,
        end_window=4,
        start_hour=1.0,
        end_hour=2.5,
        summary="Merged windows 2-4"
    )
    context.trajectory.append(t2_4)

    # Test: Get start hour for window 0 (exists in trajectory)
    hour_0 = agent._get_window_start_hour(context, 0)
    assert hour_0 == 0.0, f"Window 0 should start at 0.0, got {hour_0}"
    print(f"✓ Window 0 start hour: {hour_0}")

    # Test: Get start hour for window 1 (not in trajectory, should calculate)
    hour_1 = agent._get_window_start_hour(context, 1)
    assert hour_1 == 0.5, f"Window 1 should start at 0.5, got {hour_1}"
    print(f"✓ Window 1 start hour: {hour_1}")

    # Test: Get start hour for window 2 (exists in merged trajectory)
    hour_2 = agent._get_window_start_hour(context, 2)
    assert hour_2 == 1.0, f"Window 2 should start at 1.0, got {hour_2}"
    print(f"✓ Window 2 start hour: {hour_2}")

    # Test: Get start hour for window 3 (in merged range, should calculate)
    hour_3 = agent._get_window_start_hour(context, 3)
    assert hour_3 == 1.5, f"Window 3 should start at 1.5, got {hour_3}"
    print(f"✓ Window 3 start hour: {hour_3}")

    # Test: Get start hour for window 5 (not in trajectory, should calculate)
    hour_5 = agent._get_window_start_hour(context, 5)
    assert hour_5 == 2.5, f"Window 5 should start at 2.5, got {hour_5}"
    print(f"✓ Window 5 start hour: {hour_5}")

    # Test: Get start hour for window 10 (far ahead, should calculate)
    hour_10 = agent._get_window_start_hour(context, 10)
    assert hour_10 == 5.0, f"Window 10 should start at 5.0, got {hour_10}"
    print(f"✓ Window 10 start hour: {hour_10}")

    print("\n✓ All window hour calculation tests passed!")


def test_trajectory_entry_creation():
    """Test that trajectory entries are created with correct hours."""

    # Create agent with 0.5 hour windows
    agent = FoldAgent(window_duration_hours=0.5)

    # Create working context
    context = WorkingContext()

    # Simulate processing window 0 at hour 0.0
    # This should create T0 with start_hour=0.0, end_hour=0.5
    new_entry = TrajectoryEntry(
        start_window=0,
        end_window=0,
        start_hour=0.0,
        end_hour=0.0 + agent.window_duration_hours,
        summary="Window 0"
    )
    context.trajectory.append(new_entry)

    assert new_entry.start_hour == 0.0, f"Expected start_hour=0.0, got {new_entry.start_hour}"
    assert new_entry.end_hour == 0.5, f"Expected end_hour=0.5, got {new_entry.end_hour}"
    print(f"✓ Window 0: Hour {new_entry.start_hour}-{new_entry.end_hour}")

    # Simulate processing window 1 at hour 0.5
    new_entry = TrajectoryEntry(
        start_window=1,
        end_window=1,
        start_hour=0.5,
        end_hour=0.5 + agent.window_duration_hours,
        summary="Window 1"
    )
    context.trajectory.append(new_entry)

    assert new_entry.start_hour == 0.5, f"Expected start_hour=0.5, got {new_entry.start_hour}"
    assert new_entry.end_hour == 1.0, f"Expected end_hour=1.0, got {new_entry.end_hour}"
    print(f"✓ Window 1: Hour {new_entry.start_hour}-{new_entry.end_hour}")

    # Simulate merging windows 2-4 at hour 2.0
    # Start hour should be from window 2 (1.0), end hour should be 2.0 + 0.5 = 2.5
    start_hour = agent._get_window_start_hour(context, 2)
    new_entry = TrajectoryEntry(
        start_window=2,
        end_window=4,
        start_hour=start_hour,
        end_hour=2.0 + agent.window_duration_hours,
        summary="Merged windows 2-4"
    )

    assert new_entry.start_hour == 1.0, f"Expected start_hour=1.0, got {new_entry.start_hour}"
    assert new_entry.end_hour == 2.5, f"Expected end_hour=2.5, got {new_entry.end_hour}"
    print(f"✓ Windows 2-4 merged: Hour {new_entry.start_hour}-{new_entry.end_hour}")

    print("\n✓ All trajectory entry creation tests passed!")


if __name__ == "__main__":
    test_window_hour_calculation()
    print()
    test_trajectory_entry_creation()
