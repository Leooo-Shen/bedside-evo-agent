"""
Baseline Prediction Experiment

This script runs a baseline experiment where a model makes a single prediction
based on the entire patient trajectory (all events before the final window).
This serves as a comparison to the incremental learning agent.

The baseline:
1. Observes all events from admission to near the end of ICU stay
2. Makes a single prediction (e.g., survival outcome)
3. No memory, no reflection - just one-shot prediction

This is extensible to different prediction tasks.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime

import pandas as pd

from config.config import get_config
from data_parser import MIMICDataParser
from model.llms import LLMClient
from prompts.predictor_prompts import get_prediction_prompt
from utils.outcome_utils import evaluate_outcome_match
from utils.patient_selection import select_balanced_patients


def format_events_for_baseline(events: List[Dict], max_events: Optional[int] = None) -> str:
    """
    Format events for display in baseline prompt.

    Args:
        events: List of event dictionaries
        max_events: Maximum number of events to display (None = all)

    Returns:
        Formatted string of events
    """
    if not events:
        return "No events recorded."

    # If max_events is None, show all events
    if max_events is None:
        display_events = events
        truncated = False
    else:
        display_events = events[:max_events]
        truncated = len(events) > max_events

    formatted = ""
    for event in display_events:
        time = event.get("time", "Unknown time")
        code = event.get("code", "Unknown")
        code_specifics = event.get("code_specifics", None)
        value = event.get("numeric_value", None)
        text = event.get("text_value", None)

        formatted += f"- {time}: {code}"
        if code_specifics is not None and pd.notna(code_specifics):
            formatted += f" ({code_specifics})"
        if value is not None and pd.notna(value):
            formatted += f" = {value}"
        if text is not None and pd.notna(text):
            formatted += f" ({text})"
        formatted += "\n"

    if truncated:
        formatted += f"\n... and {len(events) - max_events} more events\n"

    return formatted


def create_baseline_prompt(
    trajectory: Dict, all_events: List[Dict], task: str = "survival", max_events: Optional[int] = None
) -> str:
    """
    Create prompt for baseline prediction.

    Args:
        trajectory: Patient trajectory data
        all_events: All events before the final window
        task: Prediction task ("survival", "mortality", etc.)
        max_events: Maximum events to include in prompt (None = all)

    Returns:
        Formatted prompt string
    """
    subject_id = trajectory.get("subject_id")
    age = trajectory.get("age_at_admission", "Unknown")
    icu_duration = trajectory.get("icu_duration_hours", 0)

    # Format age properly
    age_str = f"{age:.1f}" if isinstance(age, (int, float)) else str(age)

    # Format events
    events_str = format_events_for_baseline(all_events, max_events=max_events)

    if task == "survival":
        # Build context string with patient info and events
        context = f"## Patient Information\n- Age: {age_str} years\n\n## Clinical Events (First 12 Hours After ICU Admission)\n{events_str}"
        prompt = get_prediction_prompt().format(context=context)

    return prompt


def make_baseline_prediction(
    llm_client: LLMClient,
    trajectory: Dict,
    all_events: List[Dict],
    task: str = "survival",
    log_dir: Optional[Path] = None,
) -> Dict:
    """
    Make a baseline prediction using all available data.

    Args:
        llm_client: LLM client for making predictions
        trajectory: Patient trajectory data
        all_events: All events before final window
        task: Prediction task
        log_dir: Optional directory to save prompt/response logs

    Returns:
        Prediction dictionary
    """
    # Create prompt
    prompt = create_baseline_prompt(trajectory, all_events, task=task)

    # Get prediction from LLM
    llm_call_error = None
    try:
        response = llm_client.chat(prompt=prompt)
    except Exception as e:
        # Preserve failed calls as explicit wrong predictions instead of dropping them.
        llm_call_error = str(e)
        response = {"content": "", "usage": {}}

    # Log prompt and response if log_dir is provided
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        subject_id = trajectory.get("subject_id")
        icu_stay_id = trajectory.get("icu_stay_id")

        log_data = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "task": task,
            "llm_provider": getattr(llm_client, "provider", None),
            "llm_model": getattr(llm_client, "model", None),
            "prompt": prompt,
            "response": response.get("content", ""),
            "usage": response.get("usage", {}),
            "llm_call_failed": llm_call_error is not None,
            "llm_call_error": llm_call_error,
        }

        log_file = log_dir / f"baseline_prediction_{subject_id}_{icu_stay_id}.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    # Parse response
    if llm_call_error is not None:
        prediction = {
            "survival_prediction": {
                "outcome": "unknown",
                "confidence": 0.0,
                "rationale": f"LLM call failed: {llm_call_error}",
            }
        }
    else:
        try:
            prediction = json.loads(response["content"])
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback if JSON parsing fails
            prediction = {
                "survival_prediction": {
                    "outcome": "unknown",
                    "confidence": 0.0,
                    "rationale": "Failed to parse prediction",
                }
            }

    if llm_call_error is not None:
        prediction["llm_call_failed"] = True
        prediction["llm_call_error"] = llm_call_error
    else:
        prediction["llm_call_failed"] = False
        prediction["llm_call_error"] = None

    if llm_call_error is not None:
        print(f"   WARNING: LLM call failed and will be counted as incorrect: {llm_call_error}")

    # Add metadata
    prediction["task"] = task
    prediction["num_events_used"] = len(all_events)
    prediction["icu_duration_hours"] = trajectory.get("icu_duration_hours", 0)

    return prediction


def process_single_patient_baseline(
    patient_row: pd.Series,
    parser: MIMICDataParser,
    config,
    results_dir: Path,
    patient_idx: int,
    total_patients: int,
    task: str = "survival",
) -> Optional[Dict]:
    """
    Process a single patient through baseline experiment.

    Args:
        patient_row: Patient data from ICU stay DataFrame
        parser: MIMICDataParser instance
        config: Configuration object
        results_dir: Directory to save results
        patient_idx: Index of this patient (for display)
        total_patients: Total number of patients (for display)
        task: Prediction task

    Returns:
        Dictionary with experiment results
    """
    subject_id = patient_row["subject_id"]
    icu_stay_id = patient_row["icu_stay_id"]
    actual_outcome = "survive" if patient_row["survived"] else "die"

    print(f"\n[Patient {patient_idx}/{total_patients}] Subject: {subject_id}, ICU Stay: {icu_stay_id}")
    print(f"   Actual Outcome: {actual_outcome.upper()}")
    print(f"   Duration: {patient_row['icu_duration_hours']:.1f} hours")

    try:
        # Create LLM client for this patient
        llm_client = LLMClient(
            provider=config.llm_provider,
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
        )

        # Get patient trajectory
        trajectory = parser.get_patient_trajectory(subject_id, icu_stay_id)

        # Check if trajectory has events
        all_trajectory_events = trajectory.get("events", [])
        if len(all_trajectory_events) == 0:
            print(f"   WARNING: No events found for this patient, skipping...")
            return None

        # Use create_time_windows to get properly filtered events (removes discharge summaries)
        # This ensures we use the same data processing pipeline as survival_prediction_experiment

        # Use the same time-window configuration as survival_experiment.py
        observation_hours = config.agent_observation_hours
        windows = parser.create_time_windows(
            trajectory,
            current_window_hours=config.agent_current_window_hours,
            window_step_hours=config.agent_window_step_hours,
            include_pre_icu_data=config.agent_include_pre_icu_data,
            use_first_n_hours_after_icu=observation_hours,
        )

        if len(windows) < 1:
            print(f"   WARNING: No windows generated for this patient, skipping...")
            return None

        # Extract all events from all windows to get complete ICU trajectory
        # Collect all unique events from current_events across all windows
        all_events = []
        for window in windows:
            current_events = window.get("current_events", [])
            for event in current_events:
                all_events.append(event)

        print(
            f"   Events after filtering: {len(all_trajectory_events)} → {len(all_events)} "
            f"(First {observation_hours:.1f} hour ICU events)"
        )

        if len(all_events) == 0:
            print(f"   WARNING: No events after filtering, skipping...")
            return None

        # Create logs directory
        log_dir = results_dir / "logs"

        # Make baseline prediction
        prediction = make_baseline_prediction(llm_client, trajectory, all_events, task=task, log_dir=log_dir)

        # Extract predicted outcome
        predicted_outcome = prediction.get("survival_prediction", {}).get("outcome", "unknown")
        confidence = prediction.get("survival_prediction", {}).get("confidence", 0.0)
        is_correct, normalized_predicted_outcome, normalized_actual_outcome = evaluate_outcome_match(
            predicted=predicted_outcome,
            actual=actual_outcome,
        )

        # Print summary
        print(f"   Predicted: {predicted_outcome.upper()} (confidence: {confidence:.2f})")
        print(f"   Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

        # Get metadata for results
        icu_duration_hours = trajectory.get("icu_duration_hours", 0)
        # Calculate cutoff time from the last window
        if windows:
            last_window = windows[-1]
            cutoff_time = last_window.get("current_window_end", "")
        else:
            cutoff_time = ""

        # Calculate how many hours before ICU end the cutoff is
        from datetime import datetime

        if cutoff_time and trajectory.get("leave_time"):
            cutoff_dt = datetime.fromisoformat(str(cutoff_time))
            leave_dt = datetime.fromisoformat(trajectory["leave_time"])
            cutoff_hours_before_end = (leave_dt - cutoff_dt).total_seconds() / 3600
        else:
            cutoff_hours_before_end = None

        # Prepare results
        results = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "actual_outcome": actual_outcome,
            "predicted_outcome": predicted_outcome,
            "actual_outcome_normalized": normalized_actual_outcome,
            "predicted_outcome_normalized": normalized_predicted_outcome,
            "is_correct": is_correct,
            "confidence": confidence,
            "llm_call_failed": prediction.get("llm_call_failed", False),
            "llm_call_error": prediction.get("llm_call_error"),
            "num_events_used": len(all_events),
            "icu_duration_hours": icu_duration_hours,
            "cutoff_time": str(cutoff_time),
            "cutoff_hours_before_end": cutoff_hours_before_end,
            "num_original_events": len(all_trajectory_events),
            "num_windows_generated": len(windows),
            "full_prediction": prediction,
        }

        # Save individual patient results
        patient_results_file = results_dir / f"baseline_patient_{subject_id}_{icu_stay_id}.json"
        with open(patient_results_file, "w") as f:
            json.dump(results, f, indent=2)

        return results

    except Exception as e:
        print(f"   ERROR: Failed to process patient: {e}")
        import traceback

        traceback.print_exc()
        return None


def main(
    max_workers: int,
    task: str = "survival",
    n_survived: int = 10,
    n_died: int = 10,
):
    """
    Run baseline prediction experiment on multiple balanced patients.

    Args:
        max_workers: Maximum number of parallel workers (default 4)
        task: Prediction task (default "survival")
        n_survived: Number of survived patients for balanced selection
        n_died: Number of died patients for balanced selection
    """
    # Load configuration
    config = get_config()

    print("=" * 80)
    print(f"BASELINE PREDICTION EXPERIMENT - {task.upper()}")
    print("=" * 80)
    print(f"Max parallel workers: {max_workers}")

    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiment_results") / f"baseline-results-{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")

    # Initialize data parser
    print("\n1. Loading MIMIC-demo data...")
    parser = MIMICDataParser(
        events_path=config.events_path,
        icu_stay_path=config.icu_stay_path,
    )
    parser.load_data()
    print(f"   Loaded {len(parser.icu_stay_df)} ICU stays")

    # Select balanced patient cohort
    print("\n2. Selecting patient cohort...")
    selected_patients = select_balanced_patients(
        parser.icu_stay_df,
        n_survived=n_survived,
        n_died=n_died,
    )
    print(f"   Total patients selected: {len(selected_patients)}")

    # Run experiments in parallel
    print(f"\n3. Running baseline experiments (up to {max_workers} in parallel)...")
    print("=" * 80)

    all_results = []
    correct_predictions = 0
    total_patients = len(selected_patients)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all patient processing tasks
        future_to_patient = {}
        for idx, (_, patient_row) in enumerate(selected_patients.iterrows(), 1):
            future = executor.submit(
                process_single_patient_baseline,
                patient_row=patient_row,
                parser=parser,
                config=config,
                results_dir=results_dir,
                patient_idx=idx,
                total_patients=total_patients,
                task=task,
            )
            future_to_patient[future] = (idx, patient_row)

        # Collect results as they complete
        for future in as_completed(future_to_patient):
            idx, patient_row = future_to_patient[future]
            try:
                results = future.result()
                if results is not None:
                    all_results.append(results)
                    if results["is_correct"]:
                        correct_predictions += 1
            except Exception as e:
                print(f"   ERROR: Patient {idx} raised an exception: {e}")
                import traceback

                traceback.print_exc()

    # Compute aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    if len(all_results) > 0:
        accuracy = correct_predictions / len(all_results)
        print(f"\nTotal Patients Processed: {len(all_results)}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Incorrect Predictions: {len(all_results) - correct_predictions}")
        print(f"Overall Accuracy: {accuracy:.2%}")

        # Breakdown by outcome
        survived_results = [r for r in all_results if r["actual_outcome"] == "survive"]
        died_results = [r for r in all_results if r["actual_outcome"] == "die"]

        if survived_results:
            survived_correct = sum(1 for r in survived_results if r["is_correct"])
            survived_accuracy = survived_correct / len(survived_results)
            print(f"\nSurvived Patients:")
            print(f"   Total: {len(survived_results)}")
            print(f"   Correct: {survived_correct}")
            print(f"   Accuracy: {survived_accuracy:.2%}")

        if died_results:
            died_correct = sum(1 for r in died_results if r["is_correct"])
            died_accuracy = died_correct / len(died_results)
            print(f"\nDied Patients:")
            print(f"   Total: {len(died_results)}")
            print(f"   Correct: {died_correct}")
            print(f"   Accuracy: {died_accuracy:.2%}")

        # Average confidence
        avg_confidence = sum(r["confidence"] for r in all_results) / len(all_results)
        print(f"\nAverage Confidence: {avg_confidence:.2f}")

        # Average events used
        avg_events = sum(r["num_events_used"] for r in all_results) / len(all_results)
        print(f"Average Events Used: {avg_events:.1f}")

        # Save aggregate results
        aggregate_results = {
            "timestamp": timestamp,
            "task": task,
            "total_patients": len(all_results),
            "correct_predictions": correct_predictions,
            "incorrect_predictions": len(all_results) - correct_predictions,
            "overall_accuracy": accuracy,
            "survived_patients": len(survived_results),
            "survived_correct": survived_correct if survived_results else 0,
            "survived_accuracy": survived_accuracy if survived_results else 0,
            "died_patients": len(died_results),
            "died_correct": died_correct if died_results else 0,
            "died_accuracy": died_accuracy if died_results else 0,
            "average_confidence": avg_confidence,
            "average_events_used": avg_events,
            "individual_results": all_results,
        }

        aggregate_file = results_dir / "baseline_aggregate_results.json"
        with open(aggregate_file, "w") as f:
            json.dump(aggregate_results, f, indent=2)
        print(f"\nAggregate results saved to: {aggregate_file}")

    print("\n" + "=" * 80)
    print("BASELINE EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"All results saved to: {results_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline Prediction Experiment")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--task", type=str, default="survival", help="Prediction task (default: survival)")
    parser.add_argument("--n-survived", type=int, default=10, help="Number of survived patients")
    parser.add_argument("--n-died", type=int, default=10, help="Number of died patients")

    args = parser.parse_args()

    main(
        max_workers=args.max_workers,
        task=args.task,
        n_survived=args.n_survived,
        n_died=args.n_died,
    )
