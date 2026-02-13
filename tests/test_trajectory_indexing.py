"""Test trajectory indexing and timestamp fixes."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent_fold import TrajectoryEntry, WorkingContext


def test_trajectory_display():
    """Test that trajectory display shows correct indices and timestamps."""

    # Create working context
    context = WorkingContext()

    # Add some trajectory entries
    # T0: Window 0 (Hour 0.0-0.5)
    t0 = TrajectoryEntry(
        start_window=0,
        end_window=0,
        start_hour=0.0,
        end_hour=0.5,
        summary="Patient admitted to MICU/SICU with acute liver disorder"
    )
    context.trajectory.append(t0)

    # T1: Window 1 (Hour 0.5-1.0)
    t1 = TrajectoryEntry(
        start_window=1,
        end_window=1,
        start_hour=0.5,
        end_hour=1.0,
        summary="Patient continues with thrombocytopenia"
    )
    context.trajectory.append(t1)

    # T2-T4: Merged windows 2-4 (Hour 1.0-2.5)
    t2_4 = TrajectoryEntry(
        start_window=2,
        end_window=4,
        start_hour=1.0,
        end_hour=2.5,
        summary="Patient remains stable with ongoing monitoring"
    )
    context.trajectory.append(t2_4)

    # T5: Window 5 (Hour 2.5-3.0)
    t5 = TrajectoryEntry(
        start_window=5,
        end_window=5,
        start_hour=2.5,
        end_hour=3.0,
        summary="New hemodynamic changes observed"
    )
    context.trajectory.append(t5)

    # Generate text representation
    text = context.to_text()

    print("Generated trajectory text:")
    print("=" * 80)
    print(text)
    print("=" * 80)

    # Verify the output
    assert "T0. Hour 0.0-0.5:" in text, "T0 should show Hour 0.0-0.5"
    assert "T1. Hour 0.5-1.0:" in text, "T1 should show Hour 0.5-1.0"
    assert "T2-4. Hour 1.0-2.5:" in text, "T2-4 should show Hour 1.0-2.5"
    assert "T5. Hour 2.5-3.0:" in text, "T5 should show Hour 2.5-3.0"

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_trajectory_display()
