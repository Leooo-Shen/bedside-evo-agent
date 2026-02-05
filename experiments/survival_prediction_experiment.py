"""
Survival Prediction Experiment with Memory-Enhanced Agent

This script runs an experiment where the agent predicts patient survival/death
after ICU using a sliding 30-minute window approach. The agent can operate in two modes:
1. WITH MEMORY: Agent reflects on observations and updates memory between windows (processes all windows)
2. WITHOUT MEMORY: Agent makes predictions based only on current observations (processes only last window for efficiency)

The no_memory mode is optimized to only process the last window since previous windows
don't influence the final prediction when memory is disabled.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime

import pandas as pd

from agents.agent import EvoAgent
from config.config import get_config
from data_parser import MIMICDataParser
from prompts.agent_prompt import format_survival_prediction_prompt

seed = 1


def select_balanced_patients(icu_stay_df: pd.DataFrame, n_survived: int = 5, n_died: int = 5) -> pd.DataFrame:
    """
    Select balanced set of patients (equal numbers who survived and died).

    Args:
        icu_stay_df: DataFrame with ICU stay data
        n_survived: Number of patients who survived to select
        n_died: Number of patients who died to select

    Returns:
        DataFrame with balanced patient selection
    """
    # Separate patients by outcome
    survived_patients = icu_stay_df[icu_stay_df["survived"] == True]
    died_patients = icu_stay_df[icu_stay_df["survived"] == False]

    # Get actual available counts
    n_survived_available = len(survived_patients)
    n_died_available = len(died_patients)

    # Adjust requested numbers if they exceed available
    n_survived_actual = min(n_survived, n_survived_available)
    n_died_actual = min(n_died, n_died_available)

    print(f"   Requested: {n_survived} survived, {n_died} died")
    print(f"   Available: {n_survived_available} survived, {n_died_available} died")
    print(f"   Selected: {n_survived_actual} survived, {n_died_actual} died")

    # Sample patients
    selected_survived = survived_patients.sample(n=n_survived_actual, random_state=seed)
    selected_died = died_patients.sample(n=n_died_actual, random_state=seed)

    # Combine and shuffle
    # balanced_df = pd.concat([selected_survived, selected_died]).sample(frac=1, random_state=seed)
    balanced_df = pd.concat([selected_survived]).sample(frac=1, random_state=seed)  # TODO: only survived for testing
    return balanced_df


# Note: Prompt formatting functions moved to prompts/agent_prompt/


def _format_events(events: List[Dict], max_events: int = None) -> str:
    """Format events for display."""
    if not events:
        return "No events recorded in this period."

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
        code = event.get("code_specifics", event.get("code", "Unknown"))
        value = event.get("numeric_value")
        text = event.get("text_value")

        formatted += f"- {time}: {code}"
        if value is not None:
            formatted += f" = {value}"
        if text:
            formatted += f" ({text})"
        formatted += "\n"

    if truncated:
        formatted += f"\n... and {len(events) - max_events} more events\n"

    return formatted


def predict_survival(
    agent: EvoAgent, window_data: Dict, use_memory: bool = True, log_dir: Optional[Path] = None
) -> Dict:
    """
    Make survival prediction for a single window.

    Args:
        agent: The EvoAgent instance
        window_data: Current window data
        use_memory: Whether to use memory in predictions
        log_dir: Optional directory to save prompt/response logs

    Returns:
        Prediction dictionary with survival outcome
    """
    # Format prompt based on memory usage
    memory_context = agent.memory.format_for_prompt(max_insights=20) if use_memory else None
    prompt = format_survival_prediction_prompt(window_data, memory_context)

    # Get prediction from LLM
    response = agent.llm_client.chat(prompt=prompt, response_format="json")

    # Log prompt and response if log_dir is provided
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        subject_id = window_data.get("subject_id")
        window_index = window_data.get("window_index", 0)
        hours_since_admission = window_data.get("hours_since_admission", 0)

        log_data = {
            "subject_id": subject_id,
            "window_index": window_index,
            "hours_since_admission": hours_since_admission,
            "type": "prediction",
            "prompt": prompt,
            "response": response.get("content", ""),
            "usage": response.get("usage", {}),
        }

        log_file = log_dir / f"prediction_{subject_id}_window_{window_index:03d}.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    # Update token usage
    if "usage" in response:
        agent.total_tokens_used += response["usage"].get("input_tokens", 0)
        agent.total_tokens_used += response["usage"].get("output_tokens", 0)

    # Parse response
    try:
        prediction = json.loads(response["content"])
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        prediction = {
            "survival_prediction": {"outcome": "unknown", "confidence": 0.0, "rationale": "Failed to parse prediction"}
        }

    return prediction


def reflect_on_observation(
    agent: EvoAgent, window_data: Dict, prediction: Dict, log_dir: Optional[Path] = None
) -> Dict:
    """
    Reflect on current observations and update memory.

    Args:
        agent: The EvoAgent instance
        window_data: Current window data
        prediction: Previous prediction
        log_dir: Optional directory to save prompt/response logs

    Returns:
        Reflection dictionary
    """
    memory_context = agent.memory.format_for_prompt(max_insights=10)
    current_events = window_data.get("current_events", [])

    prompt = f"""You are reflecting on your observations to learn clinical patterns.

## Your Previous Prediction
Outcome: {prediction.get('survival_prediction', {}).get('outcome', 'N/A')}
Confidence: {prediction.get('survival_prediction', {}).get('confidence', 0.0)}
Rationale: {prediction.get('survival_prediction', {}).get('rationale', 'N/A')}

## What You Observed in This Window
{_format_events(current_events)}

## Your Current Memory
{memory_context}

## Your Task
Based on what you observed, generate a clinical insight to add to your memory.
Focus on patterns that could help predict patient survival.

Provide your response in JSON format:
{{
  "new_insight": {{
    "clinical_scenario": "Brief description of the clinical context",
    "insight": "Transferable clinical knowledge learned",
    "confidence": <float from 0.0 to 1.0>
  }},
  "observation_summary": "What key patterns did you notice?"
}}
"""

    response = agent.llm_client.chat(prompt=prompt, response_format="json")

    # Log prompt and response if log_dir is provided
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        subject_id = window_data.get("subject_id")
        window_index = window_data.get("window_index", 0)
        hours_since_admission = window_data.get("hours_since_admission", 0)

        log_data = {
            "subject_id": subject_id,
            "window_index": window_index,
            "hours_since_admission": hours_since_admission,
            "type": "reflection",
            "prompt": prompt,
            "response": response.get("content", ""),
            "usage": response.get("usage", {}),
        }

        log_file = log_dir / f"reflection_{subject_id}_window_{window_index:03d}.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    # Update token usage
    if "usage" in response:
        agent.total_tokens_used += response["usage"].get("input_tokens", 0)
        agent.total_tokens_used += response["usage"].get("output_tokens", 0)

    try:
        reflection = json.loads(response["content"])

        # Add insight to memory
        new_insight = reflection.get("new_insight", {})
        if new_insight.get("insight"):
            agent.memory.add_insight(
                insight=new_insight["insight"],
                clinical_scenario=new_insight.get("clinical_scenario", ""),
                source_window=window_data.get("window_index", 0),
                confidence=new_insight.get("confidence", 0.5),
            )
            agent.reflection_count += 1

    except json.JSONDecodeError:
        reflection = {"error": "Failed to parse reflection"}

    return reflection


def run_survival_experiment(
    patient_trajectory: Dict,
    windows: List[Dict],
    agent: EvoAgent,
    memory_mode: str = "no_memory",
    verbose: bool = True,
    log_dir: Optional[Path] = None,
) -> Dict:
    """
    Run survival prediction experiment on a patient.

    Args:
        patient_trajectory: Patient trajectory data
        windows: List of time windows
        agent: EvoAgent instance
        memory_mode: Memory system to use:
                   - "no_memory": Only process last window, no learning
                   - "cumulative_memory": Process all windows with reflection-based learning
                   - "managed_memory": Process all windows with Extract → Consolidate workflow
        verbose: Whether to print progress
        log_dir: Optional directory to save prompt/response logs

    Returns:
        Experiment results dictionary
    """
    subject_id = patient_trajectory["subject_id"]
    icu_stay_id = patient_trajectory["icu_stay_id"]
    actual_outcome = "survive" if patient_trajectory["survived"] else "die"

    if verbose:
        print(f"\n{'='*80}")
        print(f"Running Survival Prediction Experiment")
        print(f"{'='*80}")
        print(f"Patient: {subject_id} (ICU Stay: {icu_stay_id})")
        print(f"Actual Outcome: {actual_outcome.upper()}")
        print(f"Total Windows: {len(windows)}")
        print(f"Memory Mode: {memory_mode.upper()}")
        print(f"{'='*80}\n")

    predictions = []

    # Determine which windows to process based on memory mode
    if memory_mode == "no_memory":
        # In no_memory mode, only process the last window with events
        windows_to_process = []
        for i in range(len(windows) - 1, -1, -1):
            if windows[i]["current_events"] is not None and len(windows[i]["current_events"]) > 0:
                windows_to_process = [(i, windows[i])]
                break
        if verbose and windows_to_process:
            print(f"NO MEMORY MODE: Only processing last window (#{windows_to_process[0][0]+1}) with events\n")
    else:
        # Process all windows for both cumulative and managed memory modes
        windows_to_process = list(enumerate(windows))

    # Run through selected windows
    for i, window in windows_to_process:
        if window["current_events"] is None or len(window["current_events"]) == 0:
            if verbose:
                print(
                    f"Window {i+1}/{len(windows)} "
                    f"(Hour {window['hours_since_admission']:.1f})... No events, skipping."
                )
            continue
        if verbose:
            print(f"Window {i+1}/{len(windows)} " f"(Hour {window['hours_since_admission']:.1f})...", end=" ")

        # Add window index to window_data for logging
        window["window_index"] = i

        # Managed Memory Workflow: Extract → Consolidate → Predict
        if memory_mode == "managed_memory":
            current_events = window.get("current_events", [])
            hours = window.get("hours_since_admission", 0)

            # Extract insight
            insight = agent.extract_insight(current_events, hours)
            if insight and verbose:
                print(f"\n  Extracted: {insight.get('system', 'N/A')} - {insight.get('observation', 'N/A')[:50]}...")

            # Consolidate into memory
            if insight:
                success = agent.consolidate_memory(insight, hours)
                if success and verbose:
                    print(f"  Memory updated: {len(agent.memory.entries)} entries")

        # Make survival prediction
        use_memory_for_prediction = (memory_mode != "no_memory")
        prediction = predict_survival(agent, window, use_memory=use_memory_for_prediction, log_dir=log_dir)
        prediction["window_index"] = i
        prediction["hours_since_admission"] = window["hours_since_admission"]
        predictions.append(prediction)

        if verbose:
            outcome = prediction.get("survival_prediction", {}).get("outcome", "unknown")
            confidence = prediction.get("survival_prediction", {}).get("confidence", 0.0)
            print(f"Predicted: {outcome.upper()} (confidence: {confidence:.2f})")

        # Reflect and update memory (only for cumulative memory mode, except for last window)
        if memory_mode == "cumulative_memory" and i < len(windows) - 1:
            reflection = reflect_on_observation(agent, window, prediction, log_dir=log_dir)
            agent.prediction_count += 1

    # Get final prediction (last one before outcome)
    final_prediction = predictions[-1] if predictions else None

    # Evaluate final prediction
    if final_prediction:
        predicted_outcome = final_prediction.get("survival_prediction", {}).get("outcome", "unknown")
        is_correct = predicted_outcome == actual_outcome
        final_confidence = final_prediction.get("survival_prediction", {}).get("confidence", 0.0)
    else:
        is_correct = False
        predicted_outcome = "unknown"
        final_confidence = 0.0

    results = {
        "subject_id": subject_id,
        "icu_stay_id": icu_stay_id,
        "actual_outcome": actual_outcome,
        "predicted_outcome": predicted_outcome,
        "is_correct": is_correct,
        "final_confidence": final_confidence,
        "num_windows": len(windows),
        "num_predictions": len(predictions),
        "all_predictions": predictions,
        "insights_learned": len(agent.memory.entries),
    }

    if verbose:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT RESULTS")
        print(f"{'='*80}")
        print(f"Actual Outcome: {actual_outcome.upper()}")
        print(f"Final Prediction: {predicted_outcome.upper()}")
        print(f"Confidence: {final_confidence:.2f}")
        print(f"Correct: {'✓ YES' if is_correct else '✗ NO'}")
        print(f"Total Insights Learned: {len(agent.memory.entries)}")
        print(f"{'='*80}\n")

    return results


def process_single_patient(
    patient_row: pd.Series,
    parser: MIMICDataParser,
    config,
    results_dir: Path,
    patient_idx: int,
    total_patients: int,
    memory_mode: str = "no_memory",
) -> Dict:
    """
    Process a single patient through the survival prediction experiment.
    This function can be run in parallel for different patients.

    Args:
        patient_row: Patient data from ICU stay DataFrame
        parser: MIMICDataParser instance
        config: Configuration object
        results_dir: Directory to save results
        patient_idx: Index of this patient (for display)
        total_patients: Total number of patients (for display)
        memory_mode: Memory system to use ("no_memory", "cumulative_memory", "managed_memory")

    Returns:
        Dictionary with experiment results
    """
    subject_id = patient_row["subject_id"]
    icu_stay_id = patient_row["icu_stay_id"]
    actual_outcome = "survive" if patient_row["survived"] else "die"

    print(f"\n[Patient {patient_idx}/{total_patients}] Subject: {subject_id}, ICU Stay: {icu_stay_id}")
    print(f"   Actual Outcome: {actual_outcome.upper()}")
    print(f"   Duration: {patient_row['icu_duration_hours']:.1f} hours")
    print(f"   Memory Mode: {memory_mode}")

    try:
        # Create a fresh agent for this patient with appropriate memory system
        use_managed_memory = (memory_mode == "managed_memory")
        agent = EvoAgent(
            provider=config.oracle_provider,
            model=config.oracle_model,
            temperature=config.oracle_temperature,
            max_tokens=config.oracle_max_tokens,
            use_managed_memory=use_managed_memory,
            max_memory_entries=config.memory_management_max_entries if hasattr(config, 'memory_management_max_entries') else 5,
        )

        # Get patient trajectory
        trajectory = parser.get_patient_trajectory(subject_id, icu_stay_id)

        # Check if trajectory has events
        all_trajectory_events = trajectory.get("events", [])
        if len(all_trajectory_events) == 0:
            print(f"   WARNING: No events found for this patient, skipping...")
            return None

        # Create time windows from trajectory
        # Only use the first 12 hours after entering the ICU
        windows = parser.create_time_windows(
            trajectory,
            current_window_hours=config.agent_current_window_hours,
            lookback_window_hours=config.agent_lookback_window_hours,
            future_window_hours=config.agent_future_window_hours,
            window_step_hours=config.agent_window_step_hours,
            include_pre_icu_data=config.agent_include_pre_icu_data,
            use_first_n_hours_after_icu=12,
            use_discharge_summary_for_history=config.agent_use_discharge_summary_for_history,
            num_discharge_summaries=config.agent_num_discharge_summaries,
        )

        if len(windows) < 1:
            print(f"   WARNING: No windows generated for this patient, skipping...")
            return None

        print(f"   Windows: {len(windows)}")

        # Create logs directory
        log_dir = results_dir / "logs" / f"patient_{subject_id}_{icu_stay_id}"

        # Run experiment (windows are processed sequentially within this patient)
        results = run_survival_experiment(
            patient_trajectory=trajectory,
            windows=windows,
            agent=agent,
            memory_mode=memory_mode,
            verbose=False,  # Less verbose for multiple patients
            log_dir=log_dir,
        )

        # Print summary for this patient
        print(
            f"   Predicted: {results['predicted_outcome'].upper()} " f"(confidence: {results['final_confidence']:.2f})"
        )
        print(f"   Result: {'✓ CORRECT' if results['is_correct'] else '✗ INCORRECT'}")
        print(f"   Insights learned: {results['insights_learned']}")

        # Save individual patient results
        patient_results_file = results_dir / f"patient_{subject_id}_{icu_stay_id}.json"
        with open(patient_results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save patient memory
        memory_file = results_dir / f"memory_{subject_id}_{icu_stay_id}.json"
        agent.save_memory(str(memory_file))

        return results

    except Exception as e:
        print(f"   ERROR: Failed to process patient: {e}")
        import traceback

        traceback.print_exc()
        return None


def main(max_workers: int = 1, experiment_mode: str = "all"):
    """
    Run survival prediction experiment on multiple balanced patients.

    Args:
        max_workers: Maximum number of parallel workers (default 1 to avoid API rate limits)
        experiment_mode: Which experiments to run:
                        - "no_memory": No memory, only last window
                        - "cumulative_memory": Reflection-based cumulative memory
                        - "managed_memory": Extract → Consolidate managed memory
                        - "all": Run all three modes (default)
    """
    # Load configuration
    config = get_config()

    print("=" * 80)
    print("SURVIVAL PREDICTION EXPERIMENT - MULTIPLE PATIENTS (PARALLEL)")
    print("=" * 80)
    print(f"Max parallel workers: {max_workers}")
    print(f"Experiment mode: {experiment_mode}")

    # Validate experiment mode
    valid_modes = ["no_memory", "cumulative_memory", "managed_memory", "all"]
    if experiment_mode not in valid_modes:
        print(f"ERROR: Invalid experiment_mode '{experiment_mode}'. Must be one of: {valid_modes}")
        return

    # Determine which experiments to run
    if experiment_mode == "all":
        experiments_to_run = [
            ("no_memory", "no_memory"),
            ("cumulative_memory", "cumulative_memory"),
            ("managed_memory", "managed_memory"),
        ]
    else:
        experiments_to_run = [(experiment_mode, experiment_mode)]

    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = Path("experiment_results") / f"results-{timestamp}"
    base_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {base_results_dir}")

    # Initialize data parser
    print("\n1. Loading MIMIC-demo data...")
    parser = MIMICDataParser(events_path=config.events_path, icu_stay_path=config.icu_stay_path)
    parser.load_data()
    print(f"   Loaded {len(parser.icu_stay_df)} ICU stays")

    # Select balanced patients
    print("\n2. Selecting balanced patient cohort...")
    n_per_class = 2  # TODO: Number of patients per class (survived/died)
    selected_patients = select_balanced_patients(parser.icu_stay_df, n_survived=n_per_class, n_died=n_per_class)
    print(f"   Total patients selected: {len(selected_patients)}")

    # Run experiments based on mode
    for exp_name, memory_mode in experiments_to_run:
        print(f"\n{'=' * 80}")
        print(f"RUNNING EXPERIMENT: {exp_name.upper().replace('_', ' ')}")
        print(f"{'=' * 80}")

        # Create experiment-specific results directory
        results_dir = base_results_dir / exp_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Run experiments in parallel
        print(f"\n3. Running experiments on all patients (up to {max_workers} in parallel)...")
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
                    process_single_patient,
                    patient_row=patient_row,
                    parser=parser,
                    config=config,
                    results_dir=results_dir,
                    patient_idx=idx,
                    total_patients=total_patients,
                    memory_mode=memory_mode,
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
        print(f"AGGREGATE RESULTS - {exp_name.upper().replace('_', ' ')}")
        print("=" * 80)

        if total_patients > 0:
            accuracy = correct_predictions / total_patients
            print(f"\nTotal Patients: {total_patients}")
            print(f"Correct Predictions: {correct_predictions}")
            print(f"Incorrect Predictions: {total_patients - correct_predictions}")
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
            avg_confidence = sum(r["final_confidence"] for r in all_results) / len(all_results)
            print(f"\nAverage Confidence: {avg_confidence:.2f}")

            # Save aggregate results
            aggregate_results = {
                "timestamp": timestamp,
                "experiment_type": exp_name,
                "memory_mode": memory_mode,
                "total_patients": total_patients,
                "correct_predictions": correct_predictions,
                "incorrect_predictions": total_patients - correct_predictions,
                "overall_accuracy": accuracy,
                "survived_patients": len(survived_results),
                "survived_correct": survived_correct if survived_results else 0,
                "survived_accuracy": survived_accuracy if survived_results else 0,
                "died_patients": len(died_results),
                "died_correct": died_correct if died_results else 0,
                "died_accuracy": died_accuracy if died_results else 0,
                "average_confidence": avg_confidence,
                "individual_results": all_results,
            }

            aggregate_file = results_dir / "aggregate_results.json"
            with open(aggregate_file, "w") as f:
                json.dump(aggregate_results, f, indent=2)
            print(f"\nAggregate results saved to: {aggregate_file}")

        # Show aggregate statistics across all patients
        if all_results:
            total_predictions = sum(r["num_predictions"] for r in all_results)
            total_insights = sum(r["insights_learned"] for r in all_results)
            print("\n4. Aggregate Statistics (across all patients):")
            print(f"   Total predictions made: {total_predictions}")
            if memory_mode != "no_memory":
                print(f"   Total insights learned: {total_insights}")
                print(f"   Average insights per patient: {total_insights / len(all_results):.1f}")
            else:
                print(f"   Memory/reflection: DISABLED")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"All results saved to: {base_results_dir}")


if __name__ == "__main__":
    # Run experiments with different memory modes
    # Options: "no_memory", "cumulative_memory", "managed_memory", or "all"
    main(max_workers=10, experiment_mode="managed_memory")
