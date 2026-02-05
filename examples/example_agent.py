"""
Example: Running Evo-Agent on a Single Patient

This script demonstrates how to use the Evo-Agent system with evolving memory
to make predictions and learn from a patient's ICU trajectory.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.agent import EvoAgent
from config.config import get_config
from data_parser import MIMICDataParser


def main():
    """Run Evo-Agent on a single patient example."""

    # Load configuration
    config = get_config()

    print("=" * 80)
    print("EVO-AGENT - SINGLE PATIENT EXAMPLE")
    print("=" * 80)

    # Initialize data parser
    print("\n1. Loading MIMIC-demo data...")
    parser = MIMICDataParser(events_path=config.events_path, icu_stay_path=config.icu_stay_path)
    parser.load_data()

    # Get first patient
    print("\n2. Extracting patient trajectory...")
    first_icu_stay = parser.icu_stay_df.iloc[0]
    subject_id = first_icu_stay["subject_id"]
    icu_stay_id = first_icu_stay["icu_stay_id"]

    print(f"   Subject ID: {subject_id}")
    print(f"   ICU Stay ID: {icu_stay_id}")
    print(f"   Duration: {first_icu_stay['icu_duration_hours']:.1f} hours")
    print(f"   Outcome: {'Survived' if first_icu_stay['survived'] else 'Died'}")

    trajectory = parser.get_patient_trajectory(subject_id, icu_stay_id)

    # Create time windows
    print("\n3. Creating time windows...")
    print(f"   Current window: {config.agent_current_window_hours} hours ({config.agent_current_window_hours * 60:.0f} minutes)")
    print(f"   Lookback window: {config.agent_lookback_window_hours} hours")
    print(f"   Window step size: {config.agent_window_step_hours} hours ({config.agent_window_step_hours * 60:.0f} minutes)")
    print(f"   Stop 1 hour before ICU end (for prediction)")

    windows = parser.create_time_windows(
        trajectory,
        current_window_hours=config.agent_current_window_hours,
        lookback_window_hours=config.agent_lookback_window_hours,
        future_window_hours=config.get("agent_time_windows.future_window_hours", 6.0),  # For reflection
        window_step_hours=config.agent_window_step_hours,
        include_pre_icu_data=config.agent_include_pre_icu_data,
        use_first_n_hours_after_icu=config.agent_use_first_n_hours_after_icu,
        remove_discharge_summary=True,  # Remove discharge summary from windows
        use_discharge_summary_for_history=config.agent_use_discharge_summary_for_history,
        num_discharge_summaries=config.agent_num_discharge_summaries
    )
    print(f"   Generated {len(windows)} time windows")

    if len(windows) < 2:
        print("   Need at least 2 windows to run agent (for prediction + reflection)")
        return

    # Initialize Agent
    print("\n4. Initializing Evo-Agent...")
    print(f"   Provider: {config.oracle_provider}")
    print(f"   Model: {config.oracle_model or 'default'}")
    print(f"   Temperature: {config.oracle_temperature}")
    print("   NOTE: This requires API key in environment variable")

    try:
        agent = EvoAgent(
            provider=config.oracle_provider,
            model=config.oracle_model,
            temperature=config.oracle_temperature,
            max_tokens=config.oracle_max_tokens,
            log_dir=config.log_dir.replace("oracle", "agent"),
        )
        print("   Agent initialized successfully")
    except Exception as e:
        print(f"   ERROR: Failed to initialize Agent: {e}")
        print(f"   Please set {config.oracle_provider.upper()}_API_KEY environment variable")
        return

    # Run agent on first few windows (for demo)
    num_windows_to_process = min(5, len(windows) - 1)
    print(f"\n5. Running agent on first {num_windows_to_process} windows...")
    print("   This may take a few minutes...")

    try:
        results = agent.run_trajectory(windows, start_window=0)

        # Show summary
        print("\n" + "=" * 80)
        print("AGENT TRAJECTORY SUMMARY")
        print("=" * 80)

        for result in results[:num_windows_to_process]:
            window_idx = result["window_index"]
            prediction = result["prediction"]
            reflection = result["reflection"]

            print(f"\nWindow {window_idx}:")
            print(f"  Predicted trajectory: {prediction['patient_status_prediction'].get('trajectory', 'N/A')}")
            print(f"  Confidence: {prediction['confidence'].get('overall_confidence', 0):.2f}")

            if reflection:
                accuracy = reflection.get("prediction_accuracy", {})
                print(f"  Actual accuracy: {accuracy.get('overall_assessment', 'N/A')}")

                new_insight = reflection.get("new_insight", {})
                if new_insight.get("insight"):
                    print(f"  Learned: {new_insight['insight'][:100]}...")

        # Show memory
        print("\n" + "=" * 80)
        print("AGENT MEMORY")
        print("=" * 80)
        print(f"Total insights learned: {len(agent.memory.entries)}")

        if agent.memory.entries:
            print("\nRecent insights:")
            for i, entry in enumerate(agent.memory.get_recent_insights(3), 1):
                print(f"\n{i}. {entry.clinical_scenario}")
                print(f"   {entry.insight}")
                print(f"   (Window {entry.source_window}, confidence: {entry.confidence:.2f})")

        # Save memory
        print("\n6. Saving agent memory...")
        memory_path = f"{config.output_dir}/agent_memory_{subject_id}_{icu_stay_id}.json"
        agent.save_memory(memory_path)

        # Show statistics
        stats = agent.get_statistics()
        print("\n7. Agent Statistics:")
        print(f"   Total predictions: {stats['total_predictions']}")
        print(f"   Total reflections: {stats['total_reflections']}")
        print(f"   Total insights: {stats['total_insights']}")
        print(f"   Total tokens used: {stats['total_tokens_used']:,}")
        print(f"   Avg tokens per prediction: {stats['avg_tokens_per_prediction']:.0f}")

        print("\n" + "=" * 80)
        print("EXAMPLE COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"\n   ERROR during agent execution: {e}")
        import traceback

        traceback.print_exc()
        print("   This may be due to:")
        print("   - Invalid API key")
        print("   - Network issues")
        print("   - Rate limiting")
        print("   - Insufficient API credits")


if __name__ == "__main__":
    main()
