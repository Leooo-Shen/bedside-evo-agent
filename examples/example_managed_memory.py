"""
Example: Managed Memory System Demo

This script demonstrates the Extract → Consolidate → Predict workflow
using the new Managed Memory system.

Usage:
    python examples/example_managed_memory.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.agent import EvoAgent
from config.config import load_config
from data_parser import DataParser


def demo_managed_memory_workflow():
    """
    Demonstrate the managed memory workflow with a single patient.
    """
    print("\n" + "="*70)
    print("MANAGED MEMORY SYSTEM DEMO")
    print("="*70 + "\n")

    # Load configuration
    config = load_config()

    # Initialize data parser
    print("Loading patient data...")
    parser = DataParser(
        events_path=config.data.events_path,
        icu_stay_path=config.data.icu_stay_path,
    )

    # Get first patient
    patient_ids = parser.get_patient_ids()
    if not patient_ids:
        print("No patients found in dataset")
        return

    patient_id = patient_ids[0]
    print(f"Selected patient: {patient_id}\n")

    # Parse patient windows
    print("Parsing patient trajectory...")
    windows = parser.parse_patient_windows(
        subject_id=patient_id,
        current_window_hours=config.agent_time_windows.current_window_hours,
        window_step_hours=config.agent_time_windows.window_step_hours,
        lookback_window_hours=config.agent_time_windows.lookback_window_hours,
        include_pre_icu_data=config.agent_time_windows.include_pre_icu_data,
    )

    print(f"Found {len(windows)} time windows\n")

    # Initialize agent with MANAGED MEMORY
    print("Initializing agent with Managed Memory...")
    agent_managed = EvoAgent(
        provider=config.agent.provider,
        model=config.agent.model,
        temperature=config.agent.temperature,
        max_tokens=config.agent.max_tokens,
        use_managed_memory=True,  # Enable managed memory
        max_memory_entries=config.memory_management.max_entries,
    )
    print(f"✓ Managed Memory enabled (max {config.memory_management.max_entries} entries)\n")

    # Run through first 3 windows to demonstrate workflow
    print("="*70)
    print("RUNNING EXTRACT → CONSOLIDATE → PREDICT WORKFLOW")
    print("="*70 + "\n")

    num_demo_windows = min(3, len(windows) - 1)

    for i in range(num_demo_windows):
        current_window = windows[i]
        hours = current_window.get("hours_since_admission", 0)

        print(f"\n{'─'*70}")
        print(f"WINDOW {i} (Hour {hours:.1f})")
        print(f"{'─'*70}\n")

        # Step 1: Extract insight
        print("STEP 1: Extracting clinical insight...")
        current_events = current_window.get("current_events", [])
        insight = agent_managed.extract_insight(current_events, hours)

        if insight:
            print(f"  System: {insight.get('system', 'N/A')}")
            print(f"  Observation: {insight.get('observation', 'N/A')}")
            print(f"  Status: {insight.get('status', 'N/A')}")
        else:
            print("  No insight extracted")

        # Step 2: Consolidate into memory
        print("\nSTEP 2: Consolidating into memory...")
        if insight:
            success = agent_managed.consolidate_memory(insight, hours)
            if success:
                print(f"  ✓ Memory updated")
                print(f"  Total entries: {len(agent_managed.memory.entries)}")
                print(f"  Active: {len(agent_managed.memory.get_active_entries())}")
                print(f"  Resolved: {len(agent_managed.memory.get_resolved_entries())}")

        # Step 3: Display current memory state
        print("\nSTEP 3: Current Memory State:")
        print(agent_managed.memory.format_for_prompt())

        # Step 4: Make prediction
        print("STEP 4: Making prediction...")
        prediction = agent_managed.predict(current_window, window_index=i)
        print(f"  Trajectory: {prediction.patient_status_prediction.get('trajectory', 'N/A')}")
        print(f"  Severity: {prediction.patient_status_prediction.get('severity_score', 'N/A')}")
        print(f"  Confidence: {prediction.confidence.get('overall_confidence', 0):.2f}")

    # Display final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)

    stats = agent_managed.get_statistics()
    print(f"Memory Type: {stats['memory_type']}")
    print(f"Total Predictions: {stats['total_predictions']}")
    print(f"Total Extractions: {stats['total_extractions']}")
    print(f"Total Consolidations: {stats['total_consolidations']}")
    print(f"Total Memory Entries: {stats['total_insights']}")
    print(f"  - Active: {stats['active_entries']}")
    print(f"  - Resolved: {stats['resolved_entries']}")
    print(f"Total Tokens Used: {stats['total_tokens_used']}")

    # Save memory
    output_dir = Path(__file__).parent.parent / "logs" / "managed_memory_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_file = output_dir / f"memory_{patient_id}.json"
    agent_managed.save_memory(str(memory_file))
    print(f"\nMemory saved to: {memory_file}")

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70 + "\n")


def compare_memory_systems():
    """
    Compare Cumulative vs Managed Memory systems side-by-side.
    """
    print("\n" + "="*70)
    print("COMPARING CUMULATIVE VS MANAGED MEMORY")
    print("="*70 + "\n")

    # Load configuration
    config = load_config()

    # Initialize data parser
    parser = DataParser(
        events_path=config.data.events_path,
        icu_stay_path=config.data.icu_stay_path,
    )

    # Get first patient
    patient_ids = parser.get_patient_ids()
    if not patient_ids:
        print("No patients found in dataset")
        return

    patient_id = patient_ids[0]

    # Parse patient windows
    windows = parser.parse_patient_windows(
        subject_id=patient_id,
        current_window_hours=config.agent_time_windows.current_window_hours,
        window_step_hours=config.agent_time_windows.window_step_hours,
        lookback_window_hours=config.agent_time_windows.lookback_window_hours,
        include_pre_icu_data=config.agent_time_windows.include_pre_icu_data,
    )

    # Initialize both agents
    print("Initializing agents...")
    agent_cumulative = EvoAgent(
        provider=config.agent.provider,
        model=config.agent.model,
        use_managed_memory=False,  # Cumulative memory
    )

    agent_managed = EvoAgent(
        provider=config.agent.provider,
        model=config.agent.model,
        use_managed_memory=True,  # Managed memory
        max_memory_entries=5,
    )

    print("✓ Cumulative Memory Agent")
    print("✓ Managed Memory Agent\n")

    # Run both through trajectory
    print("Running trajectories (first 3 windows)...\n")

    num_windows = min(3, len(windows) - 1)

    print("CUMULATIVE MEMORY:")
    results_cumulative = agent_cumulative.run_trajectory(windows, start_window=0)

    print("\nMANAGED MEMORY:")
    results_managed = agent_managed.run_trajectory(windows, start_window=0)

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    stats_cumulative = agent_cumulative.get_statistics()
    stats_managed = agent_managed.get_statistics()

    print(f"\nCumulative Memory:")
    print(f"  Total Insights: {stats_cumulative['total_insights']}")
    print(f"  Total Tokens: {stats_cumulative['total_tokens_used']}")

    print(f"\nManaged Memory:")
    print(f"  Total Entries: {stats_managed['total_insights']}")
    print(f"  Active: {stats_managed['active_entries']}")
    print(f"  Resolved: {stats_managed['resolved_entries']}")
    print(f"  Total Tokens: {stats_managed['total_tokens_used']}")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Managed Memory System Demo")
    parser.add_argument(
        "--mode",
        choices=["demo", "compare"],
        default="demo",
        help="Demo mode: 'demo' for workflow demo, 'compare' for side-by-side comparison"
    )

    args = parser.parse_args()

    if args.mode == "demo":
        demo_managed_memory_workflow()
    elif args.mode == "compare":
        compare_memory_systems()
