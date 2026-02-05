"""
Count tokens for a patient's entire ICU trajectory.

This script:
1. Loads a patient trajectory
2. Counts the number of events
3. Reports ICU duration
4. Formats events as strings (like in Oracle prompts)
5. Counts actual tokens using tiktoken
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import load_config
from data_parser import MIMICDataParser
from prompts.oracle_prompt import _format_event

try:
    import tiktoken
except ImportError:
    print("ERROR: tiktoken not installed. Install with: pip install tiktoken")
    sys.exit(1)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for
        model: Model name for tokenizer (default: gpt-4)

    Returns:
        Number of tokens
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def analyze_patient_trajectory(parser: MIMICDataParser, subject_id: int, icu_stay_id: int):
    """
    Analyze a patient trajectory and count tokens.

    Args:
        parser: MIMICDataParser instance
        subject_id: Patient subject ID
        icu_stay_id: ICU stay ID
    """
    print("=" * 80)
    print("PATIENT TRAJECTORY TOKEN ANALYSIS")
    print("=" * 80)

    # Get patient trajectory
    print(f"\nLoading trajectory for Subject {subject_id}, ICU Stay {icu_stay_id}...")
    trajectory = parser.get_patient_trajectory(subject_id, icu_stay_id)

    # Basic statistics
    print(f"\n{'=' * 80}")
    print("BASIC STATISTICS")
    print("=" * 80)
    print(f"Subject ID: {trajectory['subject_id']}")
    print(f"ICU Stay ID: {trajectory['icu_stay_id']}")
    print(f"ICU Duration: {trajectory['icu_duration_hours']:.2f} hours")
    print(f"Age at admission: {trajectory['age_at_admission']:.1f} years")
    print(f"Outcome: {'Survived' if trajectory['survived'] else 'Died'}")
    print(f"Total events: {len(trajectory['events'])}")

    # Clean events
    cleaned_events = parser._clean_events_list(trajectory['events'])
    print(f"Cleaned events: {len(cleaned_events)}")

    # Format events as strings (like in Oracle prompts)
    print(f"\n{'=' * 80}")
    print("FORMATTING EVENTS")
    print("=" * 80)

    formatted_events = []
    for i, event in enumerate(cleaned_events, 1):
        formatted = f"E{i}. {_format_event(event)}"
        formatted_events.append(formatted)

    # Show first 5 and last 5 events as examples
    print("\nFirst 5 events:")
    for event_str in formatted_events[:5]:
        print(f"  {event_str}")

    if len(formatted_events) > 10:
        print("\n...")
        print("\nLast 5 events:")
        for event_str in formatted_events[-5:]:
            print(f"  {event_str}")

    # Combine all events into a single text
    all_events_text = "\n".join(formatted_events)

    # Count tokens
    print(f"\n{'=' * 80}")
    print("TOKEN ANALYSIS")
    print("=" * 80)

    # Count tokens for different models
    models = ["gpt-4", "gpt-3.5-turbo"]
    for model in models:
        try:
            token_count = count_tokens(all_events_text, model)
            print(f"\n{model}:")
            print(f"  Total tokens: {token_count:,}")
            print(f"  Tokens per event: {token_count / len(cleaned_events):.2f}")
            print(f"  Tokens per hour: {token_count / trajectory['icu_duration_hours']:.2f}")
        except Exception as e:
            print(f"\n{model}: Error counting tokens - {e}")

    # Character statistics
    print(f"\n{'=' * 80}")
    print("CHARACTER STATISTICS")
    print("=" * 80)
    print(f"Total characters: {len(all_events_text):,}")
    print(f"Characters per event: {len(all_events_text) / len(cleaned_events):.2f}")
    print(f"Characters per hour: {len(all_events_text) / trajectory['icu_duration_hours']:.2f}")

    return {
        "subject_id": subject_id,
        "icu_stay_id": icu_stay_id,
        "icu_duration_hours": trajectory['icu_duration_hours'],
        "total_events": len(trajectory['events']),
        "cleaned_events": len(cleaned_events),
        "formatted_text": all_events_text,
        "total_characters": len(all_events_text),
    }


def main():
    """Main execution function."""
    print("Loading configuration...")
    config = load_config()

    # Initialize data parser
    print("Initializing data parser...")
    parser = MIMICDataParser(config.events_path, config.icu_stay_path)
    parser.load_data()

    # Get first patient from dataset
    first_icu_stay = parser.icu_stay_df.iloc[0]
    subject_id = first_icu_stay["subject_id"]
    icu_stay_id = first_icu_stay["icu_stay_id"]

    print(f"\nAnalyzing first patient in dataset:")
    print(f"  Subject ID: {subject_id}")
    print(f"  ICU Stay ID: {icu_stay_id}")

    # Analyze trajectory
    result = analyze_patient_trajectory(parser, subject_id, icu_stay_id)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
