"""
Test script for ManagedMemory system

Tests:
1. Basic entry creation and retrieval
2. Deduplication by organ system
3. State transitions (ACTIVE → RESOLVED)
4. Pruning with max capacity
5. JSON serialization/deserialization
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.memory_manager import ManagedMemory, ClinicalMemoryEntry


def test_basic_operations():
    """Test basic memory operations."""
    print("\n" + "="*60)
    print("TEST 1: Basic Operations")
    print("="*60)

    memory = ManagedMemory(patient_id="test_patient", max_entries=5)

    # Add entries
    entry1 = memory.add_or_update_entry(
        system="Hemodynamics",
        description="Blood pressure low, requiring vasopressor support",
        last_updated=2.0,
        status="ACTIVE"
    )
    print(f"✓ Added entry: {entry1}")

    entry2 = memory.add_or_update_entry(
        system="Respiratory",
        description="Oxygen saturation stable at 95% on 2L nasal cannula",
        last_updated=2.5,
        status="ACTIVE"
    )
    print(f"✓ Added entry: {entry2}")

    # Check entries
    assert len(memory.entries) == 2, "Should have 2 entries"
    print(f"✓ Memory has {len(memory.entries)} entries")

    # Get active entries
    active = memory.get_active_entries()
    assert len(active) == 2, "Should have 2 active entries"
    print(f"✓ {len(active)} active entries")

    print("\n✅ Test 1 PASSED\n")


def test_deduplication():
    """Test deduplication by organ system."""
    print("="*60)
    print("TEST 2: Deduplication")
    print("="*60)

    memory = ManagedMemory(max_entries=5)

    # Add initial entry
    memory.add_or_update_entry(
        system="Renal",
        description="Potassium elevated at 6.2 mmol/L",
        last_updated=3.0,
        status="ACTIVE"
    )
    print("✓ Added initial Renal entry")

    # Update same system
    memory.add_or_update_entry(
        system="Renal",
        description="Potassium corrected to 4.8 mmol/L after treatment",
        last_updated=5.0,
        status="RESOLVED"
    )
    print("✓ Updated Renal entry (should deduplicate)")

    # Check that we still have only 1 entry
    assert len(memory.entries) == 1, "Should have 1 entry (deduplicated)"
    print(f"✓ Memory has {len(memory.entries)} entry (deduplicated)")

    # Check that entry was updated
    renal_entry = memory.entries[0]
    assert renal_entry.status == "RESOLVED", "Status should be RESOLVED"
    assert renal_entry.last_updated == 5.0, "Last updated should be 5.0"
    assert "corrected" in renal_entry.description.lower(), "Description should be updated"
    print(f"✓ Entry updated: {renal_entry.status}, Hour {renal_entry.last_updated}")

    print("\n✅ Test 2 PASSED\n")


def test_state_transitions():
    """Test ACTIVE → RESOLVED transitions."""
    print("="*60)
    print("TEST 3: State Transitions")
    print("="*60)

    memory = ManagedMemory(max_entries=5)

    # Add active entry
    memory.add_or_update_entry(
        system="Neurology",
        description="Patient confused, requiring restraints",
        last_updated=4.0,
        status="ACTIVE"
    )
    print("✓ Added ACTIVE Neurology entry")

    # Resolve the entry
    memory.resolve_entry(
        system="Neurology",
        resolution_description="Confusion resolved, restraints removed, GCS 15",
        last_updated=8.0
    )
    print("✓ Resolved Neurology entry")

    # Check status
    neuro_entry = memory.entries[0]
    assert neuro_entry.status == "RESOLVED", "Status should be RESOLVED"
    assert "resolved" in neuro_entry.description.lower(), "Description should indicate resolution"
    print(f"✓ Status: {neuro_entry.status}")
    print(f"✓ Description: {neuro_entry.description}")

    print("\n✅ Test 3 PASSED\n")


def test_pruning():
    """Test pruning with max capacity."""
    print("="*60)
    print("TEST 4: Pruning")
    print("="*60)

    memory = ManagedMemory(max_entries=3)  # Small capacity for testing

    # Add 5 entries (exceeds capacity)
    systems = [
        ("Hemodynamics", "ACTIVE", 1.0),
        ("Respiratory", "ACTIVE", 2.0),
        ("Renal", "RESOLVED", 3.0),
        ("Neurology", "RESOLVED", 4.0),
        ("Hemodynamics", "ACTIVE", 5.0),  # Update existing
    ]

    for system, status, time in systems:
        memory.add_or_update_entry(
            system=system,
            description=f"{system} observation at hour {time}",
            last_updated=time,
            status=status
        )
        print(f"✓ Added/Updated {system} ({status}) at hour {time}")

    # Check that memory was pruned to max_entries
    assert len(memory.entries) <= 3, f"Should have at most 3 entries, got {len(memory.entries)}"
    print(f"✓ Memory pruned to {len(memory.entries)} entries")

    # Check that ACTIVE entries are prioritized
    active_count = len(memory.get_active_entries())
    resolved_count = len(memory.get_resolved_entries())
    print(f"✓ Active: {active_count}, Resolved: {resolved_count}")

    # ACTIVE entries should be kept over RESOLVED
    assert active_count >= resolved_count, "Should prioritize ACTIVE over RESOLVED"
    print("✓ ACTIVE entries prioritized")

    print("\n✅ Test 4 PASSED\n")


def test_serialization():
    """Test JSON serialization and deserialization."""
    print("="*60)
    print("TEST 5: Serialization")
    print("="*60)

    # Create memory with entries
    memory = ManagedMemory(patient_id="test_patient_123", max_entries=5)

    memory.add_or_update_entry(
        system="Hemodynamics",
        description="MAP stable at 70 mmHg",
        last_updated=6.0,
        status="ACTIVE"
    )

    memory.add_or_update_entry(
        system="Respiratory",
        description="Weaned off ventilator successfully",
        last_updated=7.0,
        status="RESOLVED"
    )

    print("✓ Created memory with 2 entries")

    # Save to file
    test_file = Path(__file__).parent / "test_memory_output.json"
    memory.save(str(test_file))
    print(f"✓ Saved to {test_file}")

    # Load from file
    loaded_memory = ManagedMemory.load(str(test_file))
    print(f"✓ Loaded from {test_file}")

    # Verify
    assert loaded_memory.patient_id == "test_patient_123", "Patient ID should match"
    assert len(loaded_memory.entries) == 2, "Should have 2 entries"
    assert loaded_memory.max_entries == 5, "Max entries should match"
    print("✓ All fields match")

    # Clean up
    test_file.unlink()
    print("✓ Cleaned up test file")

    print("\n✅ Test 5 PASSED\n")


def test_format_for_prompt():
    """Test formatting for LLM prompts."""
    print("="*60)
    print("TEST 6: Format for Prompt")
    print("="*60)

    memory = ManagedMemory(max_entries=5)

    memory.add_or_update_entry(
        system="Hemodynamics",
        description="Blood pressure stabilizing with vasopressor support",
        last_updated=3.5,
        status="ACTIVE"
    )

    memory.add_or_update_entry(
        system="Renal",
        description="Hyperkalemia resolved after treatment",
        last_updated=4.0,
        status="RESOLVED"
    )

    formatted = memory.format_for_prompt()
    print("Formatted output:")
    print(formatted)

    # Check that formatted string contains key information
    assert "Hemodynamics" in formatted, "Should contain Hemodynamics"
    assert "Renal" in formatted, "Should contain Renal"
    assert "ACTIVE" in formatted, "Should contain ACTIVE status"
    assert "RESOLVED" in formatted, "Should contain RESOLVED status"
    print("✓ All key information present")

    print("\n✅ Test 6 PASSED\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING MANAGED MEMORY TESTS")
    print("="*60)

    try:
        test_basic_operations()
        test_deduplication()
        test_state_transitions()
        test_pruning()
        test_serialization()
        test_format_for_prompt()

        print("="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    run_all_tests()
