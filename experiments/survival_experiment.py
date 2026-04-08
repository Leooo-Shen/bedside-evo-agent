"""
Survival Prediction Experiment with Multiple Agent Types

This script runs the Evo-ICU experiment using different agent approaches:
1. ReMeM: Retrieval-Enhanced Memory Management (intra-patient memory only)
2. AgentFold: Hierarchical Memory with Dynamic Trajectory Folding
3. MedEvo: Survival prediction from a precomputed MedEvo memory run
"""

import hashlib
import json
import shutil
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
from agents.med_evo_agent import MedEvoAgent
from agents.remem import PatientState, RememAgent
from config.config import get_config
from data_parser import MIMICDataParser
from experiments.create_memory import (
    collect_windowed_snapshots,
    create_med_evo_memory_snapshots,
    extract_snapshot_window_features,
    filter_memory_patient_records_by_stay_ids,
    infer_snapshot_window_index,
    load_memory_patient_records,
    load_memory_run_config,
    load_patient_memory_payload,
    render_snapshot_to_text,
    resolve_memory_run_dir,
)
from prompts.predictor_prompts import get_survival_prediction_prompt
from utils.json_parse import parse_json_dict_best_effort
from utils.llm_log_viewer import build_pipeline_agents, save_llm_calls_html
from utils.outcome_utils import evaluate_outcome_match, extract_survival_prediction_fields
from utils.patient_selection import select_balanced_patients

OBSERVER_CACHE_VERSION = 1
PRE_ICU_REPORT_CODES = ["NOTE_DISCHARGESUMMARY"]
RUN_CONFIG_FILENAME = "run_config.json"


def _normalize_token_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return 0
    return 0


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


def _build_observer_cache_metadata(
    config,
    subject_id: int,
    icu_stay_id: int,
    windows: List[Dict],
    *,
    events_path: str,
    icu_stay_path: str,
) -> Dict[str, Any]:
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
            "relative_report_codes": PRE_ICU_REPORT_CODES,
            "pre_icu_history_hours": config.agent_pre_icu_history_hours,
        },
        "observer_model": {
            "provider": config.llm_provider,
            "model": config.llm_model,
            "max_tokens": config.llm_max_tokens,
            "use_observer_agent": config.agent_multi_use_observer_agent,
        },
        "data_source": {
            "events_path": events_path,
            "icu_stay_path": icu_stay_path,
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


def _build_reuse_compatibility_payload(
    config,
    agent_type: str,
    *,
    events_path: str,
    icu_stay_path: str,
) -> Dict[str, Any]:
    """Build deterministic payload used to match reusable historical patient outputs."""
    agent_settings: Dict[str, Any]
    if agent_type == "remem":
        agent_settings = {
            "max_state_length": config.remem_max_state_length,
            "enable_intra_patient_refinement": config.remem_enable_intra_patient_refinement,
        }
    elif agent_type == "multi":
        agent_settings = {
            "use_observer_agent": config.agent_multi_use_observer_agent,
            "use_memory_agent": config.agent_multi_use_memory_agent,
            "use_reflection_agent": config.agent_multi_use_reflection_agent,
            "observer_cache_enabled": config.agent_multi_observer_cache_enabled,
        }
    elif agent_type == "med_evo":
        agent_settings = {
            "max_working_windows": config.med_evo_max_working_windows,
            "max_critical_events": config.med_evo_max_critical_events,
            "max_window_summaries": config.med_evo_max_window_summaries,
            "max_insights": config.med_evo_max_insights,
            "insight_recency_tau": config.med_evo_insight_recency_tau,
            "insight_every_n_windows": config.med_evo_insight_every_n_windows,
            "episode_every_n_windows": config.med_evo_episode_every_n_windows,
        }
    elif agent_type == "fold":
        agent_settings = {
            "agent_fold_enable_key_events_extraction": config.agent_fold_enable_key_events_extraction,
            "agent_fold_max_trajectory_entries": config.agent_fold_max_trajectory_entries,
        }
    else:
        raise ValueError(f"Invalid agent type for reuse payload: {agent_type}")

    return {
        "agent_type": agent_type,
        "data_source": {
            "events_path": events_path,
            "icu_stay_path": icu_stay_path,
        },
        "windowing": {
            "observation_hours": config.agent_observation_hours,
            "current_window_hours": config.agent_current_window_hours,
            "window_step_hours": config.agent_window_step_hours,
            "include_pre_icu_data": config.agent_include_pre_icu_data,
            "use_discharge_summary_for_history": config.agent_use_discharge_summary_for_history,
            "num_discharge_summaries": config.agent_num_discharge_summaries,
            "relative_report_codes": PRE_ICU_REPORT_CODES,
            "pre_icu_history_hours": config.agent_pre_icu_history_hours,
        },
        "llm": {
            "provider": config.llm_provider,
            "model": config.llm_model,
            "max_tokens": config.llm_max_tokens,
        },
        "agent_settings": agent_settings,
    }


def _build_run_config_payload(
    config,
    *,
    agent_type: str,
    n_survived: int,
    n_died: int,
    enable_logging: bool,
    events_path: str,
    icu_stay_path: str,
) -> Dict[str, Any]:
    compatibility_payload = _build_reuse_compatibility_payload(
        config,
        agent_type=agent_type,
        events_path=events_path,
        icu_stay_path=icu_stay_path,
    )
    return {
        "generated_at": datetime.now().isoformat(),
        "agent_type": agent_type,
        "cohort_request": {
            "n_survived": int(n_survived),
            "n_died": int(n_died),
        },
        "logging": {
            "enable_logging": bool(enable_logging),
        },
        "compatibility_payload": compatibility_payload,
        "reuse_match_key": _hash_payload(compatibility_payload),
    }


def _save_run_config(results_dir: Path, payload: Dict[str, Any]) -> None:
    """Save run-level metadata so future runs can safely reuse patient outputs."""
    with open(results_dir / RUN_CONFIG_FILENAME, "w") as f:
        json.dump(payload, f, indent=2)


def _load_run_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    config_path = run_dir / RUN_CONFIG_FILENAME
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r") as f:
            loaded = json.load(f)
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def _resolve_reuse_source_runs(
    *,
    agent_type: str,
    current_results_dir: Path,
    reuse_run: Optional[str] = None,
) -> List[Path]:
    """Resolve reuse source run; reuse is enabled only when reuse_run is provided."""
    _ = agent_type  # reserved for possible future constraints
    if not reuse_run:
        return []

    root = Path("experiment_results")
    candidate = Path(reuse_run)
    if not candidate.is_absolute():
        direct = Path.cwd() / candidate
        candidate = direct if direct.exists() else (root / candidate)
    if not candidate.exists() or not candidate.is_dir():
        raise FileNotFoundError(f"Reuse source run not found: {candidate}")
    if candidate.resolve() == current_results_dir.resolve():
        return []
    return [candidate]


def _find_reusable_patient_results(
    *,
    selected_patient_keys: List[str],
    candidate_runs: List[Path],
    expected_reuse_match_key: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Locate reusable patient outputs from historical runs.

    Matching policy (strict only):
    - run has run_config.json
    - run reuse_match_key exactly matches expected_reuse_match_key
    """
    remaining = set(selected_patient_keys)
    reusable: Dict[str, Dict[str, Any]] = {}

    required_prediction_keys = {
        "subject_id",
        "icu_stay_id",
        "actual_outcome",
        "predicted_outcome",
        "is_correct",
        "confidence",
    }

    for run_dir in candidate_runs:
        if not remaining:
            break

        run_config = _load_run_config(run_dir)
        if not isinstance(run_config, dict):
            continue

        candidate_key = str(run_config.get("reuse_match_key") or "").strip()
        if candidate_key != expected_reuse_match_key:
            continue

        patients_dir = run_dir / "patients"
        if not patients_dir.exists():
            continue

        for patient_key in list(remaining):
            patient_dir = patients_dir / patient_key
            prediction_path = patient_dir / "prediction.json"
            if not prediction_path.exists():
                continue

            try:
                with open(prediction_path, "r") as f:
                    prediction = json.load(f)
            except Exception:
                continue

            if not isinstance(prediction, dict):
                continue
            if not required_prediction_keys.issubset(set(prediction.keys())):
                continue

            reusable[patient_key] = {
                "source_run_id": run_dir.name,
                "source_patient_dir": patient_dir,
                "prediction": prediction,
            }
            remaining.remove(patient_key)

    return reusable


def _copy_reused_patient_artifacts(source_patient_dir: Path, destination_patient_dir: Path) -> None:
    """Copy patient-level artifacts from historical run into current run directory."""
    destination_patient_dir.parent.mkdir(parents=True, exist_ok=True)
    if destination_patient_dir.exists():
        return
    shutil.copytree(source_patient_dir, destination_patient_dir)


def _load_patient_stay_ids_csv(patient_stay_ids_path: str) -> pd.DataFrame:
    """Load and validate a CSV with columns: subject_id, icu_stay_id."""
    path = Path(patient_stay_ids_path)
    if not path.exists():
        raise FileNotFoundError(f"Patient-stay IDs CSV not found: {path}")

    ids_df = pd.read_csv(path)
    required_columns = {"subject_id", "icu_stay_id"}
    missing = required_columns - set(ids_df.columns)
    if missing:
        raise ValueError(f"Invalid patient-stay IDs CSV (missing columns): {sorted(missing)}")

    ids_df = ids_df[["subject_id", "icu_stay_id"]].copy()
    ids_df["subject_id"] = pd.to_numeric(ids_df["subject_id"], errors="coerce").astype("Int64")
    ids_df["icu_stay_id"] = pd.to_numeric(ids_df["icu_stay_id"], errors="coerce").astype("Int64")
    ids_df = ids_df.dropna(subset=["subject_id", "icu_stay_id"]).copy()
    ids_df["subject_id"] = ids_df["subject_id"].astype("int64")
    ids_df["icu_stay_id"] = ids_df["icu_stay_id"].astype("int64")
    ids_df = ids_df.drop_duplicates(subset=["subject_id", "icu_stay_id"]).reset_index(drop=True)

    if len(ids_df) == 0:
        raise ValueError(f"No valid subject_id/icu_stay_id rows found in: {path}")

    return ids_df


def _select_patients_by_stay_ids(icu_stay_df: pd.DataFrame, patient_stay_ids_df: pd.DataFrame) -> pd.DataFrame:
    """Select ICU stays by exact (subject_id, icu_stay_id) keys."""
    parsed = icu_stay_df.copy()
    parsed["subject_id"] = pd.to_numeric(parsed["subject_id"], errors="coerce").astype("Int64")
    parsed["icu_stay_id"] = pd.to_numeric(parsed["icu_stay_id"], errors="coerce").astype("Int64")
    parsed = parsed.dropna(subset=["subject_id", "icu_stay_id"]).copy()
    parsed["subject_id"] = parsed["subject_id"].astype("int64")
    parsed["icu_stay_id"] = parsed["icu_stay_id"].astype("int64")

    selected = parsed.merge(patient_stay_ids_df, on=["subject_id", "icu_stay_id"], how="inner")
    return selected


def process_single_patient(
    patient_row: pd.Series,
    parser: MIMICDataParser,
    agent,  # Can be RememAgent, FoldAgent, or MultiAgent
    agent_type: str,
    config,
    events_path: str,
    icu_stay_path: str,
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
    agent_type: Type of agent ("remem", "fold", "multi", or "med_evo")
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

        # Create time windows based on configured observation horizon.
        observation_hours = config.agent_observation_hours
        windows = parser.create_time_windows(
            trajectory,
            current_window_hours=config.agent_current_window_hours,
            window_step_hours=config.agent_window_step_hours,
            include_pre_icu_data=config.agent_include_pre_icu_data,
            use_first_n_hours_after_icu=observation_hours,
            use_discharge_summary_for_history=config.agent_use_discharge_summary_for_history,
            num_discharge_summaries=config.agent_num_discharge_summaries,
            relative_report_codes=PRE_ICU_REPORT_CODES,
            pre_icu_history_hours=config.agent_pre_icu_history_hours,
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
                    events_path=events_path,
                    icu_stay_path=icu_stay_path,
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
        if agent_type == "remem":
            prediction, final_state, window_states = agent.run_patient_trajectory(
                windows=windows,
                patient_metadata=patient_metadata,
                verbose=verbose,
            )
            final_state_text = final_state.to_text()
        elif agent_type == "med_evo":
            memory_state, memory_db = create_med_evo_memory_snapshots(
                agent=agent,
                windows=windows,
                patient_metadata=patient_metadata,
                verbose=verbose,
            )
            last_window = windows[-1] if windows else {}
            last_window_events = last_window.get("current_events", []) if isinstance(last_window, dict) else []
            if not isinstance(last_window_events, list):
                last_window_events = []
            last_hours = float(last_window.get("hours_since_admission", 0.0)) if isinstance(last_window, dict) else 0.0

            prediction = agent.predict_from_memory(
                memory=memory_state,
                last_window_events=last_window_events,
                hours_since_admission=last_hours,
                window_index=-1,
            )
            final_state_text = memory_state.to_text()
            window_states = []
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
        else:  # agent_fold
            prediction, working_context, memory_db = agent.run_patient_trajectory(
                windows=windows,
                patient_metadata=patient_metadata,
                verbose=verbose,
            )
            final_state_text = working_context.to_text()
            window_states = []  # FoldAgent doesn't use window_states in the same way

        # Evaluate prediction
        predicted_outcome, confidence = extract_survival_prediction_fields(prediction)
        is_correct, normalized_predicted_outcome, normalized_actual_outcome = evaluate_outcome_match(
            predicted=predicted_outcome,
            actual=actual_outcome,
        )

        if verbose:
            print(f"   Predicted: {predicted_outcome.upper()} (confidence: {confidence})")
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
        elif agent_type == "med_evo":
            memory_db.save(str(patient_dir / "memory_database.json"))
            with open(patient_dir / "final_memory.json", "w") as f:
                json.dump(memory_state.to_dict(), f, indent=2)
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


def _process_single_patient_med_evo_from_memory(
    *,
    patient_record: Dict[str, Any],
    agent: MedEvoAgent,
    memory_run_dir: Path,
    results_dir: Path,
    patient_idx: int,
    total_patients: int,
    enable_logging: bool,
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    subject_id = int(patient_record["subject_id"])
    icu_stay_id = int(patient_record["icu_stay_id"])
    actual_outcome = str(patient_record.get("actual_outcome") or "unknown")
    source_patient_dir = Path(str(patient_record["patient_dir"]))

    if verbose:
        print(f"\n[Patient {patient_idx}/{total_patients}] Subject: {subject_id}, ICU Stay: {icu_stay_id}")
        print(f"   Source memory: {source_patient_dir}")
        print(f"   Actual Outcome: {actual_outcome.upper()}")

    try:
        payload = load_patient_memory_payload(source_patient_dir)
        memory_snapshots = payload.get("memory_snapshots", [])
        if not isinstance(memory_snapshots, list):
            memory_snapshots = []
        final_memory = payload.get("final_memory", {})
        if not isinstance(final_memory, dict):
            final_memory = {}

        windowed_snapshots = collect_windowed_snapshots(memory_snapshots)
        if not windowed_snapshots and final_memory:
            final_window_index = infer_snapshot_window_index(final_memory)
            windowed_snapshots = [(int(final_window_index or -1), final_memory)]

        if not windowed_snapshots:
            print("   WARNING: No snapshots found in source memory, skipping...")
            return None

        final_window_index, final_snapshot = windowed_snapshots[-1]
        inferred_window_index, final_hours, num_events = extract_snapshot_window_features(final_snapshot)
        if inferred_window_index >= 0:
            final_window_index = inferred_window_index

        context = render_snapshot_to_text(final_snapshot)
        prompt = get_survival_prediction_prompt().format(context=context)
        response = agent.llm_client.chat(prompt=prompt, response_format="text")

        raw_response = response.get("content", "")
        usage = response.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        input_tokens = _normalize_token_count(usage.get("input_tokens"))
        output_tokens = _normalize_token_count(usage.get("output_tokens"))

        parsed_prediction = parse_json_dict_best_effort(raw_response)
        if parsed_prediction is None:
            parsed_prediction = {}

        predicted_outcome, confidence = extract_survival_prediction_fields(parsed_prediction)
        is_correct, normalized_predicted_outcome, normalized_actual_outcome = evaluate_outcome_match(
            predicted=predicted_outcome,
            actual=actual_outcome,
        )

        if verbose:
            print(f"   Final Memory Window: {final_window_index} ({num_events} events)")
            print(f"   Predicted: {predicted_outcome.upper()} (confidence: {confidence})")
            print(f"   Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

        patient_dir = results_dir / "patients" / f"{subject_id}_{icu_stay_id}"
        patient_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "actual_outcome": actual_outcome,
            "predicted_outcome": predicted_outcome,
            "actual_outcome_normalized": normalized_actual_outcome,
            "predicted_outcome_normalized": normalized_predicted_outcome,
            "is_correct": is_correct,
            "confidence": confidence,
            "memory_run": str(memory_run_dir),
            "source_patient_dir": str(source_patient_dir),
            "num_memory_snapshots": len(windowed_snapshots),
            "final_memory_window_index": int(final_window_index),
            "prediction": parsed_prediction,
            "prediction_tokens": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

        with open(patient_dir / "prediction.json", "w") as f:
            json.dump(result, f, indent=2)

        if enable_logging:
            llm_call = {
                "timestamp": datetime.now().isoformat(),
                "patient_id": f"{subject_id}_{icu_stay_id}",
                "window_index": int(final_window_index),
                "hours_since_admission": float(final_hours),
                "prompt": prompt,
                "response": raw_response,
                "parsed_response": parsed_prediction,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "metadata": {
                    "step_type": "med_evo_predictor_from_memory",
                    "llm_provider": agent.llm_client.provider,
                    "llm_model": agent.llm_client.model,
                    "memory_run": str(memory_run_dir),
                    "snapshot_source": "precomputed_med_evo_memory",
                },
            }
            patient_logs = {
                "patient_id": f"{subject_id}_{icu_stay_id}",
                "agent_type": "med_evo_survival_from_memory",
                "llm_provider": getattr(agent.llm_client, "provider", None),
                "llm_model": getattr(agent.llm_client, "model", None),
                "pipeline_agents": [{"name": "survival_predictor", "used": True}],
                "total_calls": 1,
                "calls": [llm_call],
            }
            with open(patient_dir / "llm_calls.json", "w") as f:
                json.dump(patient_logs, f, indent=2)
            save_llm_calls_html(patient_logs, patient_dir / "llm_calls.html")
            if verbose:
                print("   Saved log viewer: llm_calls.html")

        return result

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


def _run_med_evo_from_memory(
    *,
    n_survived: int,
    n_died: int,
    verbose: bool,
    enable_logging: bool,
    patient_stay_ids_path: Optional[str],
    memory_run: str,
) -> Dict[str, Any]:
    del n_survived
    del n_died

    config = get_config()
    memory_run_dir = resolve_memory_run_dir(memory_run)
    source_run_config = load_memory_run_config(memory_run_dir)

    print("=" * 80)
    print("SURVIVAL PREDICTION EXPERIMENT - MED_EVO (FROM MEMORY)")
    print("=" * 80)
    print(f"Memory Run: {memory_run_dir}")
    print(f"LLM Provider: {config.llm_provider}")
    print(f"LLM Model: {config.llm_model}")

    results_dir = memory_run_dir / "survival_experiment"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results: {results_dir}")

    run_config_payload = {
        "generated_at": datetime.now().isoformat(),
        "agent_type": "med_evo",
        "experiment": "survival_experiment",
        "cohort_request": {
            "n_survived": None,
            "n_died": None,
        },
        "logging": {
            "enable_logging": bool(enable_logging),
        },
        "memory_input": {
            "memory_run": str(memory_run_dir),
            "source_generated_at": source_run_config.get("generated_at"),
            "source_experiment": source_run_config.get("experiment"),
        },
        "llm": {
            "provider": config.llm_provider,
            "model": config.llm_model,
            "max_tokens": config.llm_max_tokens,
        },
    }
    _save_run_config(results_dir, run_config_payload)

    all_records = load_memory_patient_records(memory_run_dir)
    selected_records = filter_memory_patient_records_by_stay_ids(
        records=all_records,
        patient_stay_ids_path=patient_stay_ids_path,
    )

    print(f"Patients in memory run: {len(all_records)}")
    if patient_stay_ids_path:
        print(f"Patients after --patient-stay-ids filter: {len(selected_records)}")

    if not selected_records:
        print("No patients available for prediction.")
        return {}

    print("\nProcessing patients...")
    print("=" * 80)

    all_results: List[Dict[str, Any]] = []
    patient_data = [(idx, record) for idx, record in enumerate(selected_records, 1)]

    def process_patient_wrapper(args):
        idx, patient_record = args
        patient_agent = MedEvoAgent(
            provider=config.llm_provider,
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
            enable_logging=False,
            window_duration_hours=config.agent_current_window_hours,
            max_working_windows=config.med_evo_max_working_windows,
            max_critical_events=config.med_evo_max_critical_events,
            max_window_summaries=config.med_evo_max_window_summaries,
            max_insights=config.med_evo_max_insights,
            insight_recency_tau=config.med_evo_insight_recency_tau,
            insight_every_n_windows=config.med_evo_insight_every_n_windows,
            episode_every_n_windows=config.med_evo_episode_every_n_windows,
        )
        return _process_single_patient_med_evo_from_memory(
            patient_record=patient_record,
            agent=patient_agent,
            memory_run_dir=memory_run_dir,
            results_dir=results_dir,
            patient_idx=idx,
            total_patients=len(selected_records),
            enable_logging=enable_logging,
            verbose=verbose,
        )

    max_workers = min(4, len(patient_data))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_patient_wrapper, item) for item in patient_data]
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_results.append(result)

    if not all_results:
        print("No patient results generated.")
        return {}

    correct = sum(1 for item in all_results if bool(item.get("is_correct")))
    total = len(all_results)
    accuracy = correct / total if total > 0 else 0.0

    confidence_distribution = {"Low": 0, "Moderate": 0, "High": 0, "Unknown": 0}
    for item in all_results:
        label = str(item.get("confidence") or "Unknown")
        if label in confidence_distribution:
            confidence_distribution[label] += 1
        else:
            confidence_distribution["Unknown"] += 1

    token_totals = {
        "input_tokens": sum(int(item.get("prediction_tokens", {}).get("input_tokens", 0)) for item in all_results),
        "output_tokens": sum(int(item.get("prediction_tokens", {}).get("output_tokens", 0)) for item in all_results),
    }
    token_totals["total_tokens"] = token_totals["input_tokens"] + token_totals["output_tokens"]

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print(f"\nTotal Patients: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nConfidence Distribution:")
    print(f"  Low: {confidence_distribution['Low']}")
    print(f"  Moderate: {confidence_distribution['Moderate']}")
    print(f"  High: {confidence_distribution['High']}")
    print(f"  Unknown: {confidence_distribution['Unknown']}")
    print("\nPredictor Tokens:")
    print(f"  Input: {token_totals['input_tokens']}")
    print(f"  Output: {token_totals['output_tokens']}")
    print(f"  Total: {token_totals['total_tokens']}")

    aggregate = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "agent_type": "med_evo",
        "experiment": "survival_experiment",
        "memory_run": str(memory_run_dir),
        "total_patients": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "confidence_distribution": confidence_distribution,
        "agent_stats": {
            "total_predictor_calls": total,
            "total_tokens_used": token_totals["total_tokens"],
        },
        "individual_results": sorted(all_results, key=lambda item: (item["subject_id"], item["icu_stay_id"])),
    }

    with open(results_dir / "aggregate_results.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    return aggregate


def run_experiment(
    agent_type: str = "remem",
    n_survived: int = 5,
    n_died: int = 5,
    verbose: bool = True,
    enable_logging: bool = True,
    reuse_run: Optional[str] = None,
    patient_stay_ids_path: Optional[str] = None,
    memory_run: Optional[str] = None,
) -> Dict:
    """
    Run the survival prediction experiment.

    Args:
        agent_type: "remem", "fold", "multi", or "med_evo"
        n_survived: Number of survived patients to include
        n_died: Number of died patients to include
        verbose: Print progress
        enable_logging: Enable detailed logging of all LLM calls
        reuse_run: Optional run directory to reuse from. Reuse is enabled only when provided.
        patient_stay_ids_path: Optional CSV path with columns subject_id, icu_stay_id.
            If provided, run this exact ICU-stay cohort.
        memory_run: Path to precomputed MedEvo memory run (required when agent_type=med_evo).

    Returns:
        Aggregate results dictionary
    """
    if agent_type == "med_evo":
        if not memory_run:
            raise ValueError(
                "For med_evo survival prediction, --memory-run is required. "
                "Please run experiments/create_memory.py first."
            )
        return _run_med_evo_from_memory(
            n_survived=n_survived,
            n_died=n_died,
            verbose=verbose,
            enable_logging=enable_logging,
            patient_stay_ids_path=patient_stay_ids_path,
            memory_run=memory_run,
        )

    config = get_config()
    events_path = str(config.events_path)
    icu_stay_path = str(config.icu_stay_path)

    print("=" * 80)
    print(f"SURVIVAL PREDICTION EXPERIMENT - {agent_type.upper()}")
    print("=" * 80)
    print(f"Agent Type: {agent_type}")
    observation_hours = config.agent_observation_hours
    if observation_hours is None:
        print("Observation Window: Full ICU stay (still outcome-truncated)")
    else:
        print(f"Observation Window: First {observation_hours:g} hours")
    print(f"Events Path: {events_path}")
    print(f"ICU Stay Path: {icu_stay_path}")

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiment_results") / f"{agent_type}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results: {results_dir}")

    run_config_payload = _build_run_config_payload(
        config,
        agent_type=agent_type,
        n_survived=n_survived,
        n_died=n_died,
        enable_logging=enable_logging,
        events_path=events_path,
        icu_stay_path=icu_stay_path,
    )
    _save_run_config(results_dir, run_config_payload)
    reuse_match_key = str(run_config_payload["reuse_match_key"])

    # Initialize agent based on type
    if agent_type == "remem":
        # Initialize ReMeM agent (intra-patient memory only)
        agent = RememAgent(
            provider=config.llm_provider,
            model=config.llm_model,
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
            max_tokens=config.llm_max_tokens,
            enable_logging=enable_logging,
            window_duration_hours=config.agent_current_window_hours,
            use_observer_agent=config.agent_multi_use_observer_agent,
            use_memory_agent=config.agent_multi_use_memory_agent,
            use_reflection_agent=config.agent_multi_use_reflection_agent,
        )
    elif agent_type == "fold":
        # Initialize FoldAgent
        agent = FoldAgent(
            provider=config.llm_provider,
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
            enable_logging=enable_logging,
            window_duration_hours=config.agent_current_window_hours,
        )
    elif agent_type == "med_evo":
        agent = MedEvoAgent(
            provider=config.llm_provider,
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
            enable_logging=enable_logging,
            window_duration_hours=config.agent_current_window_hours,
            max_working_windows=config.med_evo_max_working_windows,
            max_critical_events=config.med_evo_max_critical_events,
            max_window_summaries=config.med_evo_max_window_summaries,
            max_insights=config.med_evo_max_insights,
            insight_recency_tau=config.med_evo_insight_recency_tau,
            insight_every_n_windows=config.med_evo_insight_every_n_windows,
            episode_every_n_windows=config.med_evo_episode_every_n_windows,
        )
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")

    # Load data
    print("\n1. Loading MIMIC-demo data...")
    parser = MIMICDataParser(
        events_path=events_path,
        icu_stay_path=icu_stay_path,
    )
    parser.load_data()

    # Select patients
    if patient_stay_ids_path:
        print("\n2. Loading fixed patient-stay cohort...")
        requested_ids_df = _load_patient_stay_ids_csv(patient_stay_ids_path)
        selected_patients = _select_patients_by_stay_ids(parser.icu_stay_df, requested_ids_df)
        selected_patients = selected_patients.sort_values(["subject_id", "icu_stay_id"]).reset_index(drop=True)

        selected_key_set = {
            (int(row.subject_id), int(row.icu_stay_id))
            for row in selected_patients[["subject_id", "icu_stay_id"]].itertuples(index=False)
        }
        requested_key_set = {
            (int(row.subject_id), int(row.icu_stay_id))
            for row in requested_ids_df[["subject_id", "icu_stay_id"]].itertuples(index=False)
        }
        missing_keys = sorted(requested_key_set - selected_key_set)

        print(f"   Patient-stay IDs file: {patient_stay_ids_path}")
        print(f"   Requested ICU stays: {len(requested_key_set)}")
        print(f"   Matched ICU stays: {len(selected_patients)}")

        if missing_keys:
            preview = ", ".join(f"{sid}_{stay}" for sid, stay in missing_keys[:5])
            raise RuntimeError(
                "Some requested patient-stay IDs are missing from current parser cohort. "
                f"Missing={len(missing_keys)}; first 5: {preview}"
            )

        survived_count = int((selected_patients["survived"] == True).sum())  # noqa: E712
        died_count = int((selected_patients["survived"] == False).sum())  # noqa: E712
        print(f"   Cohort outcome split: {survived_count} survived, {died_count} died")
    else:
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
    reused_patient_keys: List[str] = []
    reused_source_runs = set()

    # Prepare patient data with indices first; then drop reused patients if enabled.
    patient_data = [(idx, row) for idx, (_, row) in enumerate(selected_patients.iterrows(), 1)]

    reuse_enabled = bool(str(reuse_run).strip()) if reuse_run is not None else False

    if reuse_enabled:
        candidate_runs = _resolve_reuse_source_runs(
            agent_type=agent_type,
            current_results_dir=results_dir,
            reuse_run=reuse_run,
        )
        print(f"   Reuse scan: {len(candidate_runs)} candidate run(s)")

        selected_patient_keys = [
            f"{int(row['subject_id'])}_{int(row['icu_stay_id'])}" for _, row in selected_patients.iterrows()
        ]
        reusable_lookup = _find_reusable_patient_results(
            selected_patient_keys=selected_patient_keys,
            candidate_runs=candidate_runs,
            expected_reuse_match_key=reuse_match_key,
        )

        filtered_patient_data = []
        for idx, row in patient_data:
            patient_key = f"{int(row['subject_id'])}_{int(row['icu_stay_id'])}"
            reuse_entry = reusable_lookup.get(patient_key)
            if reuse_entry is None:
                filtered_patient_data.append((idx, row))
                continue

            source_patient_dir = Path(reuse_entry["source_patient_dir"])
            destination_patient_dir = results_dir / "patients" / patient_key
            try:
                _copy_reused_patient_artifacts(source_patient_dir, destination_patient_dir)
            except Exception as exc:
                print(f"   Reuse failed for {patient_key}: {exc}; fallback to fresh inference.")
                filtered_patient_data.append((idx, row))
                continue

            reused_result = dict(reuse_entry["prediction"])
            reused_result["reused_from"] = {
                "source_run_id": reuse_entry["source_run_id"],
                "match_mode": "strict",
            }
            all_results.append(reused_result)
            reused_patient_keys.append(patient_key)
            reused_source_runs.add(str(reuse_entry["source_run_id"]))
            print(f"   Reused patient {patient_key} from run {reuse_entry['source_run_id']} (strict)")

        patient_data = filtered_patient_data
    else:
        print("   Reuse scan: disabled (provide --reuse-run to enable)")

    reuse_summary = {
        "enabled": bool(reuse_enabled),
        "reuse_run": str(reuse_run) if reuse_run else None,
        "requested_patients": int(len(selected_patients)),
        "reused_patients": int(len(reused_patient_keys)),
        "inferred_patients": int(len(patient_data)),
        "reused_patient_ids": sorted(reused_patient_keys),
        "source_runs": sorted(reused_source_runs),
        "reuse_match_key": reuse_match_key,
    }

    # Parallel processing (patients are independent)
    print(f"   Using parallel processing (no cross-patient memory)")
    print(f"   Reused patients: {reuse_summary['reused_patients']}")
    print(f"   Patients requiring new inference: {reuse_summary['inferred_patients']}")

    def process_patient_wrapper(args):
        idx, patient_row = args
        # Create a separate agent instance for thread safety
        if agent_type == "remem":
            patient_agent = RememAgent(
                provider=config.llm_provider,
                model=config.llm_model,
                max_tokens=config.llm_max_tokens,
                max_state_length=config.remem_max_state_length,
                enable_logging=enable_logging,
                enable_intra_patient_refinement=config.remem_enable_intra_patient_refinement,
            )
        elif agent_type == "multi":
            patient_agent = MultiAgent(
                provider=config.llm_provider,
                model=config.llm_model,
                max_tokens=config.llm_max_tokens,
                enable_logging=enable_logging,
                window_duration_hours=config.agent_current_window_hours,
                use_observer_agent=config.agent_multi_use_observer_agent,
                use_memory_agent=config.agent_multi_use_memory_agent,
                use_reflection_agent=config.agent_multi_use_reflection_agent,
            )
        elif agent_type == "med_evo":
            patient_agent = MedEvoAgent(
                provider=config.llm_provider,
                model=config.llm_model,
                max_tokens=config.llm_max_tokens,
                enable_logging=enable_logging,
                window_duration_hours=config.agent_current_window_hours,
                max_working_windows=config.med_evo_max_working_windows,
                max_critical_events=config.med_evo_max_critical_events,
                max_window_summaries=config.med_evo_max_window_summaries,
                max_insights=config.med_evo_max_insights,
                insight_recency_tau=config.med_evo_insight_recency_tau,
                insight_every_n_windows=config.med_evo_insight_every_n_windows,
                episode_every_n_windows=config.med_evo_episode_every_n_windows,
            )
        else:  # agent_fold
            patient_agent = FoldAgent(
                provider=config.llm_provider,
                model=config.llm_model,
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
            events_path=events_path,
            icu_stay_path=icu_stay_path,
            results_dir=results_dir,
            patient_idx=idx,
            total_patients=len(selected_patients),
            verbose=verbose,
        )

    # Process in parallel
    if patient_data:
        max_workers = min(4, len(patient_data))  # Limit concurrent workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_patient_wrapper, data) for data in patient_data]
            for future in as_completed(futures):
                results = future.result()
                if results:
                    all_results.append(results)
    else:
        if reuse_summary["enabled"]:
            print("   All selected patients were satisfied by historical reuse.")
        else:
            print("   No patients selected for new inference.")

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
        if reuse_summary["enabled"]:
            print(
                "Reuse Summary: "
                f"{reuse_summary['reused_patients']} reused, "
                f"{reuse_summary['inferred_patients']} newly inferred"
            )

        if survived_results:
            surv_correct = sum(1 for r in survived_results if r["is_correct"])
            print(
                f"\nSurvived: {len(survived_results)} patients, {surv_correct} correct ({surv_correct/len(survived_results):.2%})"
            )

        if died_results:
            died_correct = sum(1 for r in died_results if r["is_correct"])
            print(f"Died: {len(died_results)} patients, {died_correct} correct ({died_correct/len(died_results):.2%})")

        confidence_distribution = {"Low": 0, "Moderate": 0, "High": 0, "Unknown": 0}
        for item in all_results:
            label = item.get("confidence", "Unknown")
            if label in confidence_distribution:
                confidence_distribution[label] += 1
            else:
                confidence_distribution["Unknown"] += 1
        print("\nConfidence Distribution:")
        print(f"  Low: {confidence_distribution['Low']}")
        print(f"  Moderate: {confidence_distribution['Moderate']}")
        print(f"  High: {confidence_distribution['High']}")
        print(f"  Unknown: {confidence_distribution['Unknown']}")

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
        elif agent_type == "med_evo":
            print(f"  EventAgent Calls: {stats['total_event_calls']}")
            print(f"  InsightAgent Calls: {stats['total_insight_calls']}")
            print(f"  EpisodeAgent Calls: {stats.get('total_episode_calls', 0)}")
            print(f"  Predictor Calls: {stats['total_predictor_calls']}")
            print(f"  Grounding Rejections: {stats['total_grounding_rejections']}")
            print(f"  Insights Pruned: {stats['total_insights_pruned']}")
        else:  # agent_fold
            print(f"  Total Folds: {stats['total_folds']}")
            print(f"  Total Appends: {stats['total_appends']}")
        print(f"  Tokens Used (new inference only): {stats['total_tokens_used']}")

        # Save aggregate results
        aggregate = {
            "timestamp": timestamp,
            "agent_type": agent_type,
            "total_patients": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "confidence_distribution": confidence_distribution,
            "agent_stats": stats,
            "reuse": reuse_summary,
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
        choices=["remem", "fold", "multi", "med_evo"],
        default="med_evo",
        help="Agent type to use",
    )
    parser.add_argument("--n-survived", type=int, default=1, help="Number of survived patients")
    parser.add_argument("--n-died", type=int, default=0, help="Number of died patients")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--no-logging", action="store_true", help="Disable detailed LLM call logging")
    parser.add_argument(
        "--reuse-run",
        type=str,
        default=None,
        help="Specific run directory to reuse from. Reuse is disabled unless this is provided.",
    )
    parser.add_argument(
        "--patient-stay-ids",
        type=str,
        default=None,
        help=(
            "Optional CSV with columns subject_id,icu_stay_id. "
            "If provided, run this exact ICU-stay list instead of random sampling."
        ),
    )
    parser.add_argument(
        "--memory-run",
        type=str,
        default=None,
        help=(
            "Path to precomputed memory run directory produced by experiments/create_memory.py. "
            "Required when --agent-type med_evo."
        ),
    )

    args = parser.parse_args()

    if args.agent_type == "med_evo" and not args.memory_run:
        parser.error("--memory-run is required when --agent-type med_evo.")

    print(f"\n{'='*80}")
    print(f"RUNNING: {args.agent_type.upper()}")
    print(f"{'='*80}\n")

    run_experiment(
        agent_type=args.agent_type,
        n_survived=args.n_survived,
        n_died=args.n_died,
        verbose=not args.quiet,
        enable_logging=not args.no_logging,
        reuse_run=args.reuse_run,
        patient_stay_ids_path=args.patient_stay_ids,
        memory_run=args.memory_run,
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
