"""
Survival Prediction Experiment with Multiple Agent Types

This script runs the Evo-ICU experiment using different agent approaches:
1. ReMeM: Retrieval-Enhanced Memory Management (intra-patient memory only)
2. AgentFold: Hierarchical Memory with Dynamic Trajectory Folding
3. MedAgent: Static + Dynamic Memory with final predictor
"""

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.agent_fold import FoldAgent
from agents.agent_fold_multi import MultiAgent
from agents.med_agent import MedAgent
from agents.remem import PatientState, RememAgent
from config.config import get_config
from data_parser import MIMICDataParser
from utils.llm_log_viewer import build_pipeline_agents, save_llm_calls_html
from utils.outcome_utils import evaluate_outcome_match
from utils.patient_selection import select_balanced_patients

OBSERVER_CACHE_VERSION = 1


def _stable_json_dumps(data: Any) -> str:
    """Serialize data deterministically for hashing/matching."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_payload(data: Any) -> str:
    """Create a deterministic hash for cache keys and signatures."""
    return hashlib.sha256(_stable_json_dumps(data).encode("utf-8")).hexdigest()


def _build_window_signature(windows: List[Dict]) -> Dict[str, Any]:
    """Build strict matching metadata for window alignment and event content."""
    window_entries = []
    for idx, window in enumerate(windows):
        current_events = window.get("current_events", [])
        history_events = window.get("history_events", [])
        window_entries.append(
            {
                "window_index": idx,
                "hours_since_admission": window.get("hours_since_admission"),
                "current_window_start": window.get("current_window_start"),
                "current_window_end": window.get("current_window_end"),
                "num_current_events": len(current_events),
                "num_history_events": len(history_events),
                "current_events_hash": _hash_payload(current_events),
            }
        )

    return {
        "num_windows": len(windows),
        "windows_hash": _hash_payload(window_entries),
        "window_entries": window_entries,
    }


def _build_observer_cache_metadata(config, subject_id: int, icu_stay_id: int, windows: List[Dict]) -> Dict[str, Any]:
    """Create metadata used both for logging and strict cache matching."""
    window_signature = _build_window_signature(windows)

    metadata = {
        "cache_version": OBSERVER_CACHE_VERSION,
        "patient": {
            "subject_id": int(subject_id),
            "icu_stay_id": int(icu_stay_id),
        },
        "windowing": {
            "observation_hours": config.agent_observation_hours,
            "current_window_hours": config.agent_current_window_hours,
            "window_step_hours": config.agent_window_step_hours,
            "include_pre_icu_data": config.agent_include_pre_icu_data,
            "use_discharge_summary_for_history": config.agent_use_discharge_summary_for_history,
            "num_discharge_summaries": config.agent_num_discharge_summaries,
        },
        "observer_model": {
            "provider": config.llm_provider,
            "model": config.llm_model,
            "temperature": config.llm_temperature,
            "max_tokens": config.llm_max_tokens,
            "use_observer_agent": config.agent_multi_use_observer_agent,
            "observer_use_thinking": config.agent_multi_observer_use_thinking,
        },
        "data_source": {
            "events_path": config.events_path,
            "icu_stay_path": config.icu_stay_path,
            "de_identify": config.get("data.de_identify", False),
            "de_identify_seed": config.get("data.de_identify_seed", None),
        },
        "window_signature": window_signature,
    }

    match_key_payload = {
        "cache_version": metadata["cache_version"],
        "patient": metadata["patient"],
        "windowing": metadata["windowing"],
        "observer_model": metadata["observer_model"],
        "data_source": metadata["data_source"],
        "num_windows": window_signature["num_windows"],
        "windows_hash": window_signature["windows_hash"],
    }
    metadata["match_key"] = _hash_payload(match_key_payload)
    return metadata


def _get_observer_cache_file(cache_root: Path, metadata: Dict[str, Any]) -> Path:
    """Build cache file path from patient ID + strict match key."""
    patient = metadata["patient"]
    patient_key = f"{patient['subject_id']}_{patient['icu_stay_id']}"
    return cache_root / patient_key / f"{metadata['match_key']}.json"


def _load_observer_cache(cache_file: Path, expected_metadata: Dict[str, Any]) -> Tuple[Optional[List[Dict]], str]:
    """Load cached observer outputs when metadata matches exactly."""
    if not cache_file.exists():
        return None, "miss:no_cache_file"

    try:
        with open(cache_file, "r") as f:
            cached = json.load(f)
    except Exception as e:
        return None, f"miss:read_error:{e}"

    if cached.get("cache_version") != OBSERVER_CACHE_VERSION:
        return None, "miss:cache_version_mismatch"

    cached_metadata = cached.get("metadata", {})

    # Explicitly enforce same observer provider/model for cache reuse.
    expected_observer_model = expected_metadata.get("observer_model", {})
    cached_observer_model = cached_metadata.get("observer_model", {})
    if cached_observer_model.get("provider") != expected_observer_model.get("provider"):
        return None, "miss:observer_provider_mismatch"
    if cached_observer_model.get("model") != expected_observer_model.get("model"):
        return None, "miss:observer_model_mismatch"

    if cached_metadata.get("match_key") != expected_metadata.get("match_key"):
        return None, "miss:match_key_mismatch"

    observer_outputs = cached.get("observer_outputs")
    if not isinstance(observer_outputs, list):
        return None, "miss:invalid_observer_outputs"

    expected_windows = expected_metadata.get("window_signature", {}).get("num_windows", 0)
    if len(observer_outputs) != expected_windows:
        return None, f"miss:window_count_mismatch:{len(observer_outputs)}!={expected_windows}"

    expected_indices = set(range(expected_windows))
    actual_indices = set()
    for item in observer_outputs:
        if not isinstance(item, dict):
            return None, "miss:invalid_observer_output_item"
        idx = item.get("window_index")
        if not isinstance(idx, int):
            return None, "miss:invalid_window_index"
        if "observer_output" not in item:
            return None, "miss:missing_observer_output"
        actual_indices.add(idx)

    if actual_indices != expected_indices:
        return None, "miss:window_index_mismatch"

    return observer_outputs, "hit"


def _save_observer_cache(
    cache_file: Path,
    metadata: Dict[str, Any],
    observer_outputs: List[Dict],
    source_run_id: str,
) -> None:
    """Save observer outputs to cache for future ablation reuse."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": OBSERVER_CACHE_VERSION,
        "saved_at": datetime.now().isoformat(),
        "source_run_id": source_run_id,
        "metadata": metadata,
        "observer_outputs": observer_outputs,
    }
    with open(cache_file, "w") as f:
        json.dump(payload, f, indent=2)


def process_single_patient(
    patient_row: pd.Series,
    parser: MIMICDataParser,
    agent,  # Can be RememAgent, FoldAgent, MultiAgent, or MedAgent
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
        agent: Agent instance
        agent_type: Type of agent ("remem", "fold", "multi", or "med")
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
        observation_hours = config.agent_observation_hours
        windows = parser.create_time_windows(
            trajectory,
            current_window_hours=config.agent_current_window_hours,
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
        run_id = results_dir.name

        observer_cache_metadata = None
        observer_cache_info: Dict[str, Any] = {}
        precomputed_observer_outputs: Optional[List[Dict]] = None
        if agent_type == "multi":
            if config.agent_multi_use_observer_agent:
                observer_cache_metadata = _build_observer_cache_metadata(
                    config=config,
                    subject_id=subject_id,
                    icu_stay_id=icu_stay_id,
                    windows=windows,
                )
                observer_cache_file = _get_observer_cache_file(
                    cache_root=Path(config.agent_multi_observer_cache_dir),
                    metadata=observer_cache_metadata,
                )
                observer_cache_info = {
                    "enabled": config.agent_multi_observer_cache_enabled,
                    "cache_file": str(observer_cache_file),
                    "match_key": observer_cache_metadata["match_key"],
                    "reused": False,
                    "reuse_status": "not_attempted",
                    "saved": False,
                    "saved_count": 0,
                }

                if config.agent_multi_observer_cache_enabled:
                    precomputed_observer_outputs, reuse_status = _load_observer_cache(
                        cache_file=observer_cache_file,
                        expected_metadata=observer_cache_metadata,
                    )
                    observer_cache_info["reuse_status"] = reuse_status
                    observer_cache_info["reused"] = precomputed_observer_outputs is not None
                    if precomputed_observer_outputs is not None:
                        observer_cache_info["reused_count"] = len(precomputed_observer_outputs)
                else:
                    observer_cache_info["reuse_status"] = "disabled"
            else:
                observer_cache_info = {
                    "enabled": False,
                    "cache_file": None,
                    "match_key": None,
                    "reused": False,
                    "reuse_status": "observer_disabled",
                    "saved": False,
                    "saved_count": 0,
                }

            if verbose:
                if observer_cache_info["reused"]:
                    print(
                        f"   Observer cache: HIT ({observer_cache_info.get('reused_count', 0)} windows, key={observer_cache_info['match_key'][:12]})"
                    )
                elif observer_cache_info["reuse_status"] == "observer_disabled":
                    print("   Observer cache: DISABLED (observer agent disabled)")
                else:
                    print(f"   Observer cache: MISS ({observer_cache_info['reuse_status']})")

        # Clear agent logs before processing this patient
        agent.clear_logs()

        # Run appropriate agent pipeline
        med_output = None
        if agent_type == "remem":
            prediction, final_state, window_states = agent.run_patient_trajectory(
                windows=windows,
                patient_metadata=patient_metadata,
                verbose=verbose,
            )
            final_state_text = final_state.to_text()
        elif agent_type == "multi":
            prediction, working_context, memory_db = agent.run_patient_trajectory(
                windows=windows,
                patient_metadata=patient_metadata,
                precomputed_observer_outputs=precomputed_observer_outputs,
                verbose=verbose,
            )
            final_state_text = working_context.to_text()
            window_states = []

            if observer_cache_info.get("enabled"):
                observer_outputs_to_save = agent.get_observer_outputs()
                if observer_outputs_to_save:
                    try:
                        _save_observer_cache(
                            cache_file=Path(observer_cache_info["cache_file"]),
                            metadata=observer_cache_metadata,
                            observer_outputs=observer_outputs_to_save,
                            source_run_id=run_id,
                        )
                        observer_cache_info["saved"] = True
                        observer_cache_info["saved_count"] = len(observer_outputs_to_save)
                    except Exception as e:
                        observer_cache_info["save_error"] = str(e)
                        if verbose:
                            print(f"   Observer cache: SAVE ERROR ({e})")
        elif agent_type == "med":
            prediction, med_output = agent.run_patient_trajectory(
                windows=windows,
                patient_metadata=patient_metadata,
                trajectory=trajectory,
                verbose=verbose,
            )
            final_state_text = med_output.final_state_text
            window_states = []
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
        is_correct, normalized_predicted_outcome, normalized_actual_outcome = evaluate_outcome_match(
            predicted=predicted_outcome,
            actual=actual_outcome,
        )

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
            "actual_outcome_normalized": normalized_actual_outcome,
            "predicted_outcome_normalized": normalized_predicted_outcome,
            "is_correct": is_correct,
            "confidence": confidence,
            "num_windows": len(windows),
            "final_state": final_state_text,
            "prediction": prediction,
        }
        if agent_type == "multi":
            results["observer_cache"] = observer_cache_info

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
        elif agent_type == "med":
            if med_output is not None:
                with open(patient_dir / "patient_memory.json", "w") as f:
                    json.dump(med_output.patient_memory_dict(), f, indent=2)
                with open(patient_dir / "dynamic_memory_history.json", "w") as f:
                    json.dump(med_output.dynamic_history_dict(), f, indent=2)
        else:  # agent_fold or multi
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
                # "active_concerns": [
                #     {"id": c.concern_id, "status": c.status, "note": c.note} for c in working_context.active_concerns
                # ],
            }
            with open(patient_dir / "working_context.json", "w") as f:
                json.dump(context_data, f, indent=2)

            if agent_type == "multi":
                observer_outputs_log = {
                    "patient_id": f"{subject_id}_{icu_stay_id}",
                    "metadata": observer_cache_metadata,
                    "cache": observer_cache_info,
                    "observer_outputs": agent.get_observer_outputs(),
                }
                with open(patient_dir / "observer_outputs.json", "w") as f:
                    json.dump(observer_outputs_log, f, indent=2)

        # Save patient-specific LLM calls
        if agent.enable_logging:
            patient_logs = {
                "patient_id": f"{subject_id}_{icu_stay_id}",
                "agent_type": agent_type,
                "llm_provider": getattr(getattr(agent, "llm_client", None), "provider", None),
                "llm_model": getattr(getattr(agent, "llm_client", None), "model", None),
                "pipeline_agents": build_pipeline_agents(agent, agent_type),
                "total_calls": len(agent.call_logs),
                "calls": agent.get_logs(),
            }
            with open(patient_dir / "llm_calls.json", "w") as f:
                json.dump(patient_logs, f, indent=2)
            save_llm_calls_html(patient_logs, patient_dir / "llm_calls.html")
            if verbose:
                print("   Saved log viewer: llm_calls.html")

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
        agent_type: "remem", "fold", "multi", or "med"
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
    observation_hours = config.agent_observation_hours
    print(f"Observation Window: {observation_hours} hours")

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiment_results") / f"{agent_type}_{timestamp}"
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
    elif agent_type == "multi":
        # Initialize MultiAgent (Observer + Memory + Predictor)
        agent = MultiAgent(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            enable_logging=enable_logging,
            window_duration_hours=config.agent_current_window_hours,
            observation_hours=config.agent_observation_hours,
            use_observer_agent=config.agent_multi_use_observer_agent,
            use_memory_agent=config.agent_multi_use_memory_agent,
            use_reflection_agent=config.agent_multi_use_reflection_agent,
            observer_use_thinking=config.agent_multi_observer_use_thinking,
            memory_use_thinking=config.agent_multi_memory_use_thinking,
            reflection_use_thinking=config.agent_multi_reflection_use_thinking,
            predictor_use_thinking=config.agent_multi_predictor_use_thinking,
        )
    elif agent_type == "med":
        # Initialize MedAgent (Static Memory + Dynamic Memory + Predictor)
        agent = MedAgent(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            enable_logging=enable_logging,
            observation_hours=config.agent_observation_hours,
            use_llm_static_compression=config.med_agent_use_llm_static_compression,
            baseline_lab_lookback_start_hours=config.med_agent_baseline_lab_lookback_start_hours,
            baseline_lab_lookback_end_hours=config.med_agent_baseline_lab_lookback_end_hours,
            max_active_problems=config.med_agent_max_active_problems,
            max_critical_events=config.med_agent_max_critical_events,
            max_patterns=config.med_agent_max_patterns,
            memory_use_thinking=config.med_agent_memory_use_thinking,
            predictor_use_thinking=config.med_agent_predictor_use_thinking,
        )
    elif agent_type == "fold":
        # Initialize FoldAgent
        agent = FoldAgent(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            enable_logging=enable_logging,
            window_duration_hours=config.agent_current_window_hours,
        )
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")

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

    # Parallel processing (patients are independent)
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
        elif agent_type == "multi":
            patient_agent = MultiAgent(
                provider=config.llm_provider,
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                enable_logging=enable_logging,
                window_duration_hours=config.agent_current_window_hours,
                observation_hours=config.agent_observation_hours,
                use_observer_agent=config.agent_multi_use_observer_agent,
                use_memory_agent=config.agent_multi_use_memory_agent,
                use_reflection_agent=config.agent_multi_use_reflection_agent,
                observer_use_thinking=config.agent_multi_observer_use_thinking,
                memory_use_thinking=config.agent_multi_memory_use_thinking,
                reflection_use_thinking=config.agent_multi_reflection_use_thinking,
                predictor_use_thinking=config.agent_multi_predictor_use_thinking,
            )
        elif agent_type == "med":
            patient_agent = MedAgent(
                provider=config.llm_provider,
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                enable_logging=enable_logging,
                observation_hours=config.agent_observation_hours,
                use_llm_static_compression=config.med_agent_use_llm_static_compression,
                baseline_lab_lookback_start_hours=config.med_agent_baseline_lab_lookback_start_hours,
                baseline_lab_lookback_end_hours=config.med_agent_baseline_lab_lookback_end_hours,
                max_active_problems=config.med_agent_max_active_problems,
                max_critical_events=config.med_agent_max_critical_events,
                max_patterns=config.med_agent_max_patterns,
                memory_use_thinking=config.med_agent_memory_use_thinking,
                predictor_use_thinking=config.med_agent_predictor_use_thinking,
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
        elif agent_type == "multi":
            if config.agent_multi_use_observer_agent:
                print(f"  Observer Calls: {stats['total_observer_calls']}")
                print(f"  Observer Cache Hits: {stats.get('total_observer_cache_hits', 0)}")
            else:
                print("  Observer Agent: DISABLED (raw-event passthrough)")
            if config.agent_multi_use_memory_agent:
                print(f"  Memory Calls: {stats['total_memory_calls']}")
                print(f"  Total Folds: {stats['total_folds']}")
                print(f"  Total Appends: {stats['total_appends']}")
                if config.agent_multi_use_reflection_agent:
                    print(f"  Reflection Calls: {stats.get('total_reflection_calls', 0)}")
                    print(f"  Revisions: {stats.get('total_revisions', 0)}")
            else:
                print(f"  Memory Agent: DISABLED (ablation mode)")
        elif agent_type == "med":
            print(f"  Static Compression Calls: {stats.get('total_static_compression_calls', 0)}")
            print(f"  Dynamic Memory Calls: {stats.get('total_memory_calls', 0)}")
            print(f"  Predictor Calls: {stats.get('total_predictor_calls', 0)}")
            print(f"  Dynamic Fallbacks: {stats.get('total_dynamic_fallbacks', 0)}")
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
        choices=["remem", "fold", "multi", "med"],
        default="multi",
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
