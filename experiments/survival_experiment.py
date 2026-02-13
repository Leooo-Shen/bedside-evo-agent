"""
Survival Prediction Experiment with Multiple Agent Types

This script runs the Evo-ICU experiment using different agent approaches:
1. ReMeM: Retrieval-Enhanced Memory Management (intra-patient memory only)
2. AgentFold: Hierarchical Memory with Dynamic Trajectory Folding
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.agent_fold import FoldAgent
from agents.remem import PatientState, RememAgent
from config.config import get_config
from data_parser import MIMICDataParser

SEED = 1


def select_balanced_patients(
    icu_stay_df: pd.DataFrame,
    n_survived: int = 5,
    n_died: int = 5,
) -> pd.DataFrame:
    """Select balanced set of patients (equal numbers who survived and died)."""
    survived_patients = icu_stay_df[icu_stay_df["survived"] == True]
    died_patients = icu_stay_df[icu_stay_df["survived"] == False]

    n_survived_actual = min(n_survived, len(survived_patients))
    n_died_actual = min(n_died, len(died_patients))

    print(f"   Requested: {n_survived} survived, {n_died} died")
    print(f"   Available: {len(survived_patients)} survived, {len(died_patients)} died")
    print(f"   Selected: {n_survived_actual} survived, {n_died_actual} died")

    selected_survived = survived_patients.sample(n=n_survived_actual, random_state=SEED)
    selected_died = died_patients.sample(n=n_died_actual, random_state=SEED)

    balanced_df = pd.concat([selected_survived, selected_died]).sample(frac=1, random_state=SEED)
    return balanced_df


def process_single_patient(
    patient_row: pd.Series,
    parser: MIMICDataParser,
    agent,  # Can be RememAgent or FoldAgent
    agent_type: str,
    config,
    results_dir: Path,
    patient_idx: int,
    total_patients: int,
    verbose: bool = True,
) -> Optional[Dict]:
    """
    Process a single patient through the survival prediction pipeline.

    Args:
        patient_row: Patient data from ICU stay DataFrame
        parser: MIMICDataParser instance
        agent: Agent instance (RememAgent or FoldAgent)
        agent_type: Type of agent ("remem" or "fold")
        config: Configuration object
        results_dir: Directory to save results
        patient_idx: Index of this patient
        total_patients: Total number of patients
        verbose: Print progress

    Returns:
        Dictionary with experiment results
    """
    subject_id = patient_row["subject_id"]
    icu_stay_id = patient_row["icu_stay_id"]
    actual_outcome = "survive" if patient_row["survived"] else "die"

    if verbose:
        print(f"\n[Patient {patient_idx}/{total_patients}] Subject: {subject_id}, ICU Stay: {icu_stay_id}")
        print(f"   Actual Outcome: {actual_outcome.upper()}")
        print(f"   Duration: {patient_row['icu_duration_hours']:.1f} hours")

    try:
        # Get patient trajectory
        trajectory = parser.get_patient_trajectory(subject_id, icu_stay_id)

        if len(trajectory.get("events", [])) == 0:
            print(f"   WARNING: No events found, skipping...")
            return None

        # Create time windows (first 12 hours)
        observation_hours = (
            config.remem_observation_hours if agent_type == "remem" else config.agent_fold_observation_hours
        )
        windows = parser.create_time_windows(
            trajectory,
            current_window_hours=config.agent_current_window_hours,
            lookback_window_hours=config.agent_lookback_window_hours,
            future_window_hours=config.agent_future_window_hours,
            window_step_hours=config.agent_window_step_hours,
            include_pre_icu_data=config.agent_include_pre_icu_data,
            use_first_n_hours_after_icu=observation_hours,
        )

        if len(windows) < 1:
            print(f"   WARNING: No windows generated, skipping...")
            return None

        if verbose:
            print(f"   Windows: {len(windows)}")

        # Run agent pipeline
        patient_metadata = {
            "age": trajectory.get("age_at_admission", 0),
            "gender": trajectory.get("gender", None),
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
        }

        # Clear agent logs before processing this patient
        agent.clear_logs()

        # Run appropriate agent pipeline
        if agent_type == "remem":
            prediction, final_state, window_states = agent.run_patient_trajectory(
                windows=windows,
                patient_metadata=patient_metadata,
                verbose=verbose,
            )
            final_state_text = final_state.to_text()
        else:  # agent_fold
            prediction, working_context, memory_db = agent.run_patient_trajectory(
                windows=windows,
                patient_metadata=patient_metadata,
                verbose=verbose,
            )
            final_state_text = working_context.to_text()
            window_states = []  # FoldAgent doesn't use window_states in the same way

        # Evaluate prediction
        predicted_outcome = prediction.get("survival_prediction", {}).get("outcome", "unknown")
        confidence = prediction.get("survival_prediction", {}).get("confidence", 0.0)
        is_correct = predicted_outcome == actual_outcome

        if verbose:
            print(f"   Predicted: {predicted_outcome.upper()} (confidence: {confidence:.2f})")
            print(f"   Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

        # Create patient-specific directory
        patient_dir = results_dir / "patients" / f"{subject_id}_{icu_stay_id}"
        patient_dir.mkdir(parents=True, exist_ok=True)

        # Build results
        results = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "actual_outcome": actual_outcome,
            "predicted_outcome": predicted_outcome,
            "is_correct": is_correct,
            "confidence": confidence,
            "num_windows": len(windows),
            "final_state": final_state_text,
            "prediction": prediction,
        }

        # Save prediction.json (same structure as before)
        with open(patient_dir / "prediction.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save agent-specific data
        if agent_type == "remem":
            # Save patient_specific_insights.json (window-by-window state evolution)
            insights = {
                "patient_id": f"{subject_id}_{icu_stay_id}",
                "num_windows": len(windows),
                "window_states": window_states,
            }
            with open(patient_dir / "patient_specific_insights.json", "w") as f:
                json.dump(insights, f, indent=2)
        else:  # agent_fold
            # Save memory database
            memory_db.save(str(patient_dir / "memory_database.json"))
            # Save working context
            context_data = {
                "patient_id": f"{subject_id}_{icu_stay_id}",
                "num_trajectories": len(working_context.trajectory),
                "trajectories": [
                    {
                        "start_window": t.start_window,
                        "end_window": t.end_window,
                        "start_hour": t.start_hour,
                        "end_hour": t.end_hour,
                        "summary": t.summary,
                    }
                    for t in working_context.trajectory
                ],
                "key_events": working_context.historical_key_events,
                "active_concerns": [
                    {"id": c.concern_id, "status": c.status, "note": c.note} for c in working_context.active_concerns
                ],
            }
            with open(patient_dir / "working_context.json", "w") as f:
                json.dump(context_data, f, indent=2)

        # Save patient-specific LLM calls
        if agent.enable_logging:
            patient_logs = {
                "patient_id": f"{subject_id}_{icu_stay_id}",
                "total_calls": len(agent.call_logs),
                "calls": agent.get_logs(),
            }
            with open(patient_dir / "llm_calls.json", "w") as f:
                json.dump(patient_logs, f, indent=2)

        return results

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_experiment(
    agent_type: str = "remem",
    n_survived: int = 5,
    n_died: int = 5,
    verbose: bool = True,
    enable_logging: bool = True,
) -> Dict:
    """
    Run the survival prediction experiment.

    Args:
        agent_type: "remem" or "fold"
        n_survived: Number of survived patients to include
        n_died: Number of died patients to include
        verbose: Print progress
        enable_logging: Enable detailed logging of all LLM calls

    Returns:
        Aggregate results dictionary
    """
    config = get_config()

    print("=" * 80)
    print(f"SURVIVAL PREDICTION EXPERIMENT - {agent_type.upper()}")
    print("=" * 80)
    print(f"Agent Type: {agent_type}")
    if agent_type == "remem":
        observation_hours = config.remem_observation_hours
    else:
        observation_hours = config.agent_fold_observation_hours
    print(f"Observation Window: {observation_hours} hours")

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if agent_type == "remem":
        results_dir = Path("experiment_results") / f"remem_{timestamp}"
    else:
        results_dir = Path("experiment_results") / f"fold_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results: {results_dir}")

    # Initialize agent based on type
    if agent_type == "remem":
        # Initialize ReMeM agent (intra-patient memory only)
        agent = RememAgent(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            max_state_length=config.remem_max_state_length,
            enable_logging=enable_logging,
            enable_intra_patient_refinement=config.remem_enable_intra_patient_refinement,
        )
    else:  # agent_fold
        # Initialize FoldAgent
        agent = FoldAgent(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            enable_logging=enable_logging,
            window_duration_hours=config.agent_current_window_hours,
        )

    # Load data
    print("\n1. Loading MIMIC-demo data...")
    parser = MIMICDataParser(
        events_path=config.events_path,
        icu_stay_path=config.icu_stay_path,
    )
    parser.load_data()

    # Select patients
    print("\n2. Selecting balanced patient cohort...")
    selected_patients = select_balanced_patients(
        parser.icu_stay_df,
        n_survived=n_survived,
        n_died=n_died,
    )

    # Process patients
    print("\n3. Processing patients...")
    print("=" * 80)

    all_results = []

    # Parallel processing for both remem and fold agents (patients are independent)
    print(f"   Using parallel processing (no cross-patient memory)")

    def process_patient_wrapper(args):
        idx, patient_row = args
        # Create a separate agent instance for thread safety
        if agent_type == "remem":
            patient_agent = RememAgent(
                provider=config.llm_provider,
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                max_state_length=config.remem_max_state_length,
                enable_logging=enable_logging,
                enable_intra_patient_refinement=config.remem_enable_intra_patient_refinement,
            )
        else:  # agent_fold
            patient_agent = FoldAgent(
                provider=config.llm_provider,
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                enable_logging=enable_logging,
                window_duration_hours=config.agent_current_window_hours,
            )
        return process_single_patient(
            patient_row=patient_row,
            parser=parser,
            agent=patient_agent,
            agent_type=agent_type,
            config=config,
            results_dir=results_dir,
            patient_idx=idx,
            total_patients=len(selected_patients),
            verbose=verbose,
        )

    # Prepare patient data with indices
    patient_data = [(idx, row) for idx, (_, row) in enumerate(selected_patients.iterrows(), 1)]

    # Process in parallel
    max_workers = min(4, len(selected_patients))  # Limit concurrent workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_patient_wrapper, data) for data in patient_data]
        for future in as_completed(futures):
            results = future.result()
            if results:
                all_results.append(results)

    # Compute aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    if all_results:
        correct = sum(1 for r in all_results if r["is_correct"])
        total = len(all_results)
        accuracy = correct / total

        survived_results = [r for r in all_results if r["actual_outcome"] == "survive"]
        died_results = [r for r in all_results if r["actual_outcome"] == "die"]

        print(f"\nTotal Patients: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.2%}")

        if survived_results:
            surv_correct = sum(1 for r in survived_results if r["is_correct"])
            print(
                f"\nSurvived: {len(survived_results)} patients, {surv_correct} correct ({surv_correct/len(survived_results):.2%})"
            )

        if died_results:
            died_correct = sum(1 for r in died_results if r["is_correct"])
            print(f"Died: {len(died_results)} patients, {died_correct} correct ({died_correct/len(died_results):.2%})")

        avg_confidence = sum(r["confidence"] for r in all_results) / len(all_results)
        print(f"\nAverage Confidence: {avg_confidence:.2f}")

        # Agent statistics
        stats = agent.get_statistics()
        print(f"\nAgent Statistics:")
        if agent_type == "remem":
            print(f"  State Updates: {stats['state_updates']}")
            print(f"  Refinements: {stats['refinements']}")
        else:  # agent_fold
            print(f"  Total Folds: {stats['total_folds']}")
            print(f"  Total Appends: {stats['total_appends']}")
        print(f"  Tokens Used: {stats['total_tokens_used']}")

        # Save aggregate results
        aggregate = {
            "timestamp": timestamp,
            "agent_type": agent_type,
            "total_patients": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "agent_stats": stats,
            "individual_results": all_results,
        }

        with open(results_dir / "aggregate_results.json", "w") as f:
            json.dump(aggregate, f, indent=2)

        return aggregate

    return {}


def main():
    """Run experiments with different agent types."""
    import argparse

    parser = argparse.ArgumentParser(description="Survival Prediction Experiment")
    parser.add_argument(
        "--agent-type",
        choices=["remem", "fold"],
        default="remem",
        help="Agent type to use",
    )
    parser.add_argument("--n-survived", type=int, default=10, help="Number of survived patients")
    parser.add_argument("--n-died", type=int, default=10, help="Number of died patients")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--no-logging", action="store_true", help="Disable detailed LLM call logging")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"RUNNING: {args.agent_type.upper()}")
    print(f"{'='*80}\n")

    run_experiment(
        agent_type=args.agent_type,
        n_survived=args.n_survived,
        n_died=args.n_died,
        verbose=not args.quiet,
        enable_logging=not args.no_logging,
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
