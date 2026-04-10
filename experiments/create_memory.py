"""Create and load MedEvo memory runs for downstream experiments."""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.med_evo_agent import MedEvoAgent, MedEvoMemory, MedEvoMemoryDatabase
from config.config import get_config
from data_parser import MIMICDataParser
from utils.event_format import format_event_lines
from utils.llm_log_viewer import save_llm_calls_html
from utils.patient_selection import select_balanced_patients

PRE_ICU_REPORT_CODES = ["NOTE_DISCHARGESUMMARY"]
RUN_CONFIG_FILENAME = "run_config.json"
PATIENT_INFO_FILENAME = "patient_info.json"
AGGREGATE_FILENAME = "aggregate_results.json"


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sum_numeric_stats(items: List[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def _load_patient_stay_ids_csv(patient_stay_ids_path: str) -> pd.DataFrame:
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
    parsed = icu_stay_df.copy()
    parsed["subject_id"] = pd.to_numeric(parsed["subject_id"], errors="coerce").astype("Int64")
    parsed["icu_stay_id"] = pd.to_numeric(parsed["icu_stay_id"], errors="coerce").astype("Int64")
    parsed = parsed.dropna(subset=["subject_id", "icu_stay_id"]).copy()
    parsed["subject_id"] = parsed["subject_id"].astype("int64")
    parsed["icu_stay_id"] = parsed["icu_stay_id"].astype("int64")

    selected = parsed.merge(patient_stay_ids_df, on=["subject_id", "icu_stay_id"], how="inner")
    return selected


def create_med_evo_memory_snapshots(
    agent: MedEvoAgent,
    windows: List[Dict[str, Any]],
    patient_metadata: Dict[str, Any],
    verbose: bool = True,
) -> Tuple[MedEvoMemory, MedEvoMemoryDatabase]:
    """Run MedEvo memory-building once and return all window snapshots."""
    return agent.create_memory_snapshots(
        windows=windows,
        patient_metadata=patient_metadata,
        verbose=verbose,
    )


def infer_snapshot_window_index(snapshot: Dict[str, Any]) -> Optional[int]:
    """Extract window_id from a MedEvo memory snapshot."""
    working_memory = snapshot.get("working_memory")
    if not isinstance(working_memory, list) or not working_memory:
        return None

    last_window = working_memory[-1]
    if not isinstance(last_window, dict):
        return None

    try:
        return int(last_window.get("window_id"))
    except (TypeError, ValueError):
        return None


def collect_windowed_snapshots(memory_snapshots: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    """Collect snapshots keyed by window index and return them in window order."""
    snapshots_by_window: Dict[int, Dict[str, Any]] = {}
    for snapshot in memory_snapshots:
        if not isinstance(snapshot, dict):
            continue
        window_index = infer_snapshot_window_index(snapshot)
        if window_index is None:
            continue
        snapshots_by_window[int(window_index)] = snapshot

    return sorted(snapshots_by_window.items(), key=lambda item: item[0])


def select_snapshots_with_stride(
    windowed_snapshots: List[Tuple[int, Dict[str, Any]]],
    stride: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Select every k-th snapshot and always include the final snapshot."""
    normalized_stride = max(1, int(stride))
    selected = [item for idx, item in enumerate(windowed_snapshots) if idx % normalized_stride == 0]

    if not windowed_snapshots:
        return selected

    last_item = windowed_snapshots[-1]
    if not selected or selected[-1][0] != last_item[0]:
        selected.append(last_item)

    return selected


def render_snapshot_to_text(snapshot: Dict[str, Any]) -> str:
    """Render a MedEvo memory snapshot dict into predictor context text."""
    patient_metadata = snapshot.get("patient_metadata", {})
    trajectory_memory = snapshot.get("trajectory_memory", [])
    critical_events = snapshot.get("critical_events", [])
    insights = snapshot.get("insights", [])
    working_memory = snapshot.get("working_memory", [])

    parts: List[str] = ["## Patient Metadata"]
    if isinstance(patient_metadata, dict) and patient_metadata:
        for key, value in patient_metadata.items():
            if key in {"subject_id", "icu_stay_id"}:
                continue
            parts.append(f"{key}: {value}")
    else:
        parts.append("- None")

    parts.append("")
    parts.append("## Trajectory of the ICU Stay")
    if isinstance(trajectory_memory, list) and trajectory_memory:
        for item in trajectory_memory:
            if not isinstance(item, dict):
                parts.append(f"- {item}")
                continue
            item_type = item.get("type")
            if item_type == "window_summary":
                parts.append(
                    f"- Window {item.get('window_id')} "
                    f"(hour {_safe_float(item.get('start_hour')):.1f}-{_safe_float(item.get('end_hour')):.1f}): "
                    f"{item.get('text', '')}"
                )
            elif item_type == "episode":
                parts.append(
                    f"- Window {item.get('start_window')}-{item.get('end_window')} "
                    f"(hour {_safe_float(item.get('start_hour')):.1f}-{_safe_float(item.get('end_hour')):.1f}): "
                    f"{item.get('episode_summary', item.get('text', ''))}"
                )
            else:
                parts.append(f"- {item}")
    else:
        parts.append("- None")

    parts.append("")
    parts.append("## Critical Events of the ICU Stay")
    if isinstance(critical_events, list) and critical_events:
        for item in critical_events:
            if isinstance(item, dict):
                name_text = str(item.get("name_str") or "").strip()
                parts.append(name_text if name_text else str(item))
            else:
                parts.append(str(item))
    else:
        parts.append("- None")

    parts.append("")
    parts.append("## Patient Specific Insights")
    if isinstance(insights, list) and insights:
        for item in insights:
            if not isinstance(item, dict):
                parts.append(f"- {item}")
                continue
            insight_id = item.get("insight_id")
            score = item.get("score")
            hypothesis = item.get("hypothesis", "")
            if isinstance(score, (int, float)):
                parts.append(f"- I{insight_id} score={float(score):.3f}: {hypothesis}")
            else:
                parts.append(f"- I{insight_id}: {hypothesis}")
    else:
        parts.append("- None")

    parts.append("")
    parts.append("## Current Window Observation")
    if isinstance(working_memory, list) and working_memory:
        current_window = working_memory[-1]
        if isinstance(current_window, dict):
            start_hour = _safe_float(current_window.get("start_hour"))
            end_hour = _safe_float(current_window.get("end_hour"))
            parts.append(f"Window {current_window.get('window_id')} (Hour {start_hour:.1f}-{end_hour:.1f})")
            current_events = current_window.get("events", [])
            if isinstance(current_events, list):
                parts.extend(format_event_lines(current_events, empty_text="- (No events)"))
            else:
                parts.append("- (No events)")
        else:
            parts.append("- None")
    else:
        parts.append("- None")

    return "\n".join(parts)


def extract_snapshot_window_features(snapshot: Dict[str, Any]) -> Tuple[int, float, int]:
    """Return (window_index, hours_since_admission, num_current_events)."""
    window_index = -1
    hours_since_admission = 0.0
    num_current_events = 0

    working_memory = snapshot.get("working_memory")
    if not isinstance(working_memory, list) or not working_memory:
        return window_index, hours_since_admission, num_current_events

    current_window = working_memory[-1]
    if not isinstance(current_window, dict):
        return window_index, hours_since_admission, num_current_events

    window_index = _safe_int(current_window.get("window_id"), default=-1)
    hours_since_admission = _safe_float(current_window.get("start_hour"), default=0.0)
    current_events = current_window.get("events", [])
    if isinstance(current_events, list):
        num_current_events = len(current_events)

    return window_index, hours_since_admission, num_current_events


def extract_snapshot_hour_bounds(snapshot: Dict[str, Any]) -> Tuple[float, float]:
    """Return (start_hour, end_hour) for a snapshot's current window."""
    working_memory = snapshot.get("working_memory")
    if not isinstance(working_memory, list) or not working_memory:
        return 0.0, 0.0

    current_window = working_memory[-1]
    if not isinstance(current_window, dict):
        return 0.0, 0.0

    start_hour = _safe_float(current_window.get("start_hour"), default=0.0)
    end_hour = _safe_float(current_window.get("end_hour"), default=start_hour)
    if end_hour < start_hour:
        end_hour = start_hour
    return start_hour, end_hour


def select_snapshot_by_observation_hour(
    windowed_snapshots: List[Tuple[int, Dict[str, Any]]],
    observation_hour: float,
) -> Tuple[int, Dict[str, Any]]:
    """Select the latest snapshot whose current-window end_hour is <= observation_hour."""
    if not windowed_snapshots:
        raise ValueError("Cannot select snapshot by observation hour from empty snapshot list.")

    try:
        normalized_hour = float(observation_hour)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid observation_hour value: {observation_hour}") from exc
    if not math.isfinite(normalized_hour) or normalized_hour < 0:
        raise ValueError(f"observation_hour must be a finite number >= 0, got {observation_hour}")

    selected_item: Optional[Tuple[int, Dict[str, Any]]] = None
    containing_item: Optional[Tuple[int, Dict[str, Any]]] = None
    first_start_hour, first_end_hour = extract_snapshot_hour_bounds(windowed_snapshots[0][1])

    for window_index, snapshot in windowed_snapshots:
        start_hour, end_hour = extract_snapshot_hour_bounds(snapshot)
        if end_hour <= normalized_hour:
            selected_item = (int(window_index), snapshot)
        if containing_item is None and start_hour <= normalized_hour <= end_hour:
            containing_item = (int(window_index), snapshot)

    if selected_item is not None:
        return selected_item
    if containing_item is not None:
        return containing_item

    raise ValueError(
        "No snapshot is available for the requested observation hour "
        f"{normalized_hour:g}. Earliest snapshot covers hour range [{first_start_hour:g}, {first_end_hour:g}]."
    )


def resolve_memory_run_dir(memory_run: str) -> Path:
    path = Path(str(memory_run).strip())
    if path.exists() and path.is_dir():
        return path

    candidate = Path("experiment_results") / str(memory_run).strip()
    if candidate.exists() and candidate.is_dir():
        return candidate

    raise FileNotFoundError(f"Memory run directory not found: {memory_run}. Tried '{path}' and '{candidate}'.")


def load_memory_run_config(memory_run_dir: Path) -> Dict[str, Any]:
    config_path = memory_run_dir / RUN_CONFIG_FILENAME
    if not config_path.exists():
        return {}
    return _read_json(config_path)


def load_memory_patient_records(memory_run_dir: Path) -> List[Dict[str, Any]]:
    aggregate_path = memory_run_dir / AGGREGATE_FILENAME
    records: List[Dict[str, Any]] = []

    if aggregate_path.exists():
        aggregate_payload = _read_json(aggregate_path)
        candidates = aggregate_payload.get("individual_results", [])
        if isinstance(candidates, list):
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                subject_id = _safe_int(item.get("subject_id"), default=0)
                icu_stay_id = _safe_int(item.get("icu_stay_id"), default=0)
                if subject_id <= 0 or icu_stay_id <= 0:
                    continue

                patient_dir = memory_run_dir / "patients" / f"{subject_id}_{icu_stay_id}"
                if not patient_dir.exists():
                    continue

                records.append(
                    {
                        "subject_id": subject_id,
                        "icu_stay_id": icu_stay_id,
                        "actual_outcome": str(item.get("actual_outcome") or "").strip() or "unknown",
                        "num_windows": _safe_int(item.get("num_windows"), default=0),
                        "num_memory_snapshots": _safe_int(item.get("num_memory_snapshots"), default=0),
                        "patient_dir": str(patient_dir),
                    }
                )

    if records:
        records.sort(key=lambda item: (int(item["subject_id"]), int(item["icu_stay_id"])))
        return records

    patients_dir = memory_run_dir / "patients"
    if not patients_dir.exists():
        return []

    for patient_dir in sorted(patients_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        info_path = patient_dir / PATIENT_INFO_FILENAME
        if not info_path.exists():
            continue
        info = _read_json(info_path)
        subject_id = _safe_int(info.get("subject_id"), default=0)
        icu_stay_id = _safe_int(info.get("icu_stay_id"), default=0)
        if subject_id <= 0 or icu_stay_id <= 0:
            continue
        records.append(
            {
                "subject_id": subject_id,
                "icu_stay_id": icu_stay_id,
                "actual_outcome": str(info.get("actual_outcome") or "").strip() or "unknown",
                "num_windows": _safe_int(info.get("num_windows"), default=0),
                "num_memory_snapshots": _safe_int(info.get("num_memory_snapshots"), default=0),
                "patient_dir": str(patient_dir),
            }
        )

    records.sort(key=lambda item: (int(item["subject_id"]), int(item["icu_stay_id"])))
    return records


def filter_memory_patient_records_by_stay_ids(
    records: List[Dict[str, Any]],
    patient_stay_ids_path: Optional[str],
) -> List[Dict[str, Any]]:
    if not patient_stay_ids_path:
        return records

    ids_df = _load_patient_stay_ids_csv(patient_stay_ids_path)
    key_set = {(int(row.subject_id), int(row.icu_stay_id)) for row in ids_df.itertuples(index=False)}

    filtered = [
        item for item in records if (int(item.get("subject_id", 0)), int(item.get("icu_stay_id", 0))) in key_set
    ]
    filtered.sort(key=lambda item: (int(item["subject_id"]), int(item["icu_stay_id"])))
    return filtered


def load_patient_memory_payload(patient_dir: Path) -> Dict[str, Any]:
    info_path = patient_dir / PATIENT_INFO_FILENAME
    memory_db_path = patient_dir / "memory_database.json"
    final_memory_path = patient_dir / "final_memory.json"

    if not info_path.exists():
        raise FileNotFoundError(f"Missing patient info file: {info_path}")
    if not memory_db_path.exists():
        raise FileNotFoundError(f"Missing memory database file: {memory_db_path}")

    info = _read_json(info_path)
    memory_db = _read_json(memory_db_path)
    memory_snapshots = memory_db.get("memory_snapshots", [])
    if not isinstance(memory_snapshots, list):
        memory_snapshots = []

    final_memory: Dict[str, Any] = {}
    if final_memory_path.exists():
        maybe_final = _read_json(final_memory_path)
        if isinstance(maybe_final, dict):
            final_memory = maybe_final

    return {
        "patient_info": info,
        "memory_snapshots": memory_snapshots,
        "final_memory": final_memory,
    }


def _save_run_config(results_dir: Path, payload: Dict[str, Any]) -> None:
    with open(results_dir / RUN_CONFIG_FILENAME, "w") as f:
        json.dump(payload, f, indent=2)


def _is_complete_patient_result(patient_dir: Path) -> bool:
    return (
        (patient_dir / PATIENT_INFO_FILENAME).exists()
        and (patient_dir / "memory_database.json").exists()
        and (patient_dir / "final_memory.json").exists()
    )


def _collect_completed_patient_keys(results_dir: Path) -> Set[Tuple[int, int]]:
    patients_dir = results_dir / "patients"
    if not patients_dir.exists():
        return set()

    completed: Set[Tuple[int, int]] = set()
    for patient_dir in patients_dir.iterdir():
        if not patient_dir.is_dir():
            continue
        if not _is_complete_patient_result(patient_dir):
            continue
        try:
            subject_id_text, icu_stay_id_text = patient_dir.name.split("_", 1)
            completed.add((int(subject_id_text), int(icu_stay_id_text)))
        except (TypeError, ValueError):
            continue
    return completed


def _process_single_patient_memory(
    *,
    patient_row: pd.Series,
    parser: MIMICDataParser,
    config,
    enable_logging: bool,
    verbose: bool,
    patient_idx: int,
    total_patients: int,
    results_dir: Path,
) -> Optional[Dict[str, Any]]:
    subject_id = int(patient_row["subject_id"])
    icu_stay_id = int(patient_row["icu_stay_id"])
    actual_outcome = "survive" if bool(patient_row["survived"]) else "die"

    if verbose:
        print(f"\n[Patient {patient_idx}/{total_patients}] Subject: {subject_id}, ICU Stay: {icu_stay_id}")
        print(f"   Actual Outcome: {actual_outcome.upper()}")
        print(f"   Duration: {float(patient_row['icu_duration_hours']):.1f} hours")

    trajectory = parser.get_patient_trajectory(subject_id, icu_stay_id)
    if len(trajectory.get("events", [])) == 0:
        print("   WARNING: No events found, skipping...")
        return None

    windows = parser.create_time_windows(
        trajectory,
        current_window_hours=config.agent_current_window_hours,
        window_step_hours=config.agent_window_step_hours,
        include_pre_icu_data=config.agent_include_pre_icu_data,
        use_first_n_hours_after_icu=config.agent_observation_hours,
        use_discharge_summary_for_history=config.agent_use_discharge_summary_for_history,
        num_discharge_summaries=config.agent_num_discharge_summaries,
        relative_report_codes=PRE_ICU_REPORT_CODES,
        pre_icu_history_hours=config.agent_pre_icu_history_hours,
    )
    if len(windows) < 1:
        print("   WARNING: No windows generated, skipping...")
        return None

    if verbose:
        print(f"   Windows: {len(windows)}")

    patient_metadata = {
        "age": trajectory.get("age_at_admission", 0),
        "gender": trajectory.get("gender", None),
        "subject_id": subject_id,
        "icu_stay_id": icu_stay_id,
    }

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
    agent.clear_logs()

    final_memory_state, memory_db = create_med_evo_memory_snapshots(
        agent=agent,
        windows=windows,
        patient_metadata=patient_metadata,
        verbose=verbose,
    )

    windowed_snapshots = collect_windowed_snapshots(memory_db.memory_snapshots)
    if not windowed_snapshots:
        print("   WARNING: No non-empty memory snapshots generated, skipping...")
        return None

    patient_dir = results_dir / "patients" / f"{subject_id}_{icu_stay_id}"
    patient_dir.mkdir(parents=True, exist_ok=True)

    memory_db.save(str(patient_dir / "memory_database.json"))
    with open(patient_dir / "final_memory.json", "w") as f:
        json.dump(final_memory_state.to_dict(), f, indent=2)

    info_payload = {
        "subject_id": subject_id,
        "icu_stay_id": icu_stay_id,
        "actual_outcome": actual_outcome,
        "num_windows": len(windows),
        "num_memory_snapshots": len(windowed_snapshots),
    }
    with open(patient_dir / PATIENT_INFO_FILENAME, "w") as f:
        json.dump(info_payload, f, indent=2)

    if enable_logging:
        logs_payload = {
            "patient_id": f"{subject_id}_{icu_stay_id}",
            "agent_type": "med_evo_memory_creation",
            "llm_provider": getattr(agent.llm_client, "provider", None),
            "llm_model": getattr(agent.llm_client, "model", None),
            "pipeline_agents": [
                {"name": "perception_agent", "used": True},
                {"name": "event_agent", "used": True},
                {"name": "insight_agent", "used": True},
                {"name": "episode_agent", "used": agent.episode_every_n_windows > 0},
                {"name": "survival_predictor", "used": False},
            ],
            "total_calls": len(agent.get_logs()),
            "calls": agent.get_logs(),
        }
        with open(patient_dir / "llm_calls.json", "w") as f:
            json.dump(logs_payload, f, indent=2)
        save_llm_calls_html(logs_payload, patient_dir / "llm_calls.html")
        if verbose:
            print("   Saved log viewer: llm_calls.html")

    return {
        "subject_id": subject_id,
        "icu_stay_id": icu_stay_id,
        "actual_outcome": actual_outcome,
        "num_windows": len(windows),
        "num_memory_snapshots": len(windowed_snapshots),
        "agent_stats": agent.get_statistics(),
    }


def run_experiment(
    n_survived: int = 5,
    n_died: int = 5,
    verbose: bool = True,
    enable_logging: bool = True,
    patient_stay_ids_path: Optional[str] = None,
    num_workers: int = 1,
    resume_run: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        num_workers = int(num_workers)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid num_workers value: {num_workers}") from exc
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")

    config = get_config()
    events_path = str(config.events_path)
    icu_stay_path = str(config.icu_stay_path)

    print("=" * 80)
    print("CREATE MEMORY RUN - MED_EVO")
    print("=" * 80)

    observation_hours = config.agent_observation_hours
    if observation_hours is None:
        print("Observation Window: Full ICU stay")
    else:
        print(f"Observation Window: First {observation_hours:g} hours")
    print(f"Events Path: {events_path}")
    print(f"ICU Stay Path: {icu_stay_path}")
    print(f"Num Workers: {num_workers}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if resume_run:
        results_dir = resolve_memory_run_dir(resume_run)
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Resuming Existing Results: {results_dir}")
    else:
        results_dir = Path("experiment_results") / f"memory_snapshot_med_evo_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results: {results_dir}")

    run_config_payload = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "create_memory_med_evo",
        "cohort_request": {
            "n_survived": int(n_survived),
            "n_died": int(n_died),
        },
        "logging": {
            "enable_logging": bool(enable_logging),
        },
        "execution": {
            "num_workers": num_workers,
            "resume_run": str(resume_run) if resume_run else None,
        },
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
        "med_evo": {
            "max_working_windows": config.med_evo_max_working_windows,
            "max_critical_events": config.med_evo_max_critical_events,
            "max_window_summaries": config.med_evo_max_window_summaries,
            "max_insights": config.med_evo_max_insights,
            "insight_recency_tau": config.med_evo_insight_recency_tau,
            "insight_every_n_windows": config.med_evo_insight_every_n_windows,
            "episode_every_n_windows": config.med_evo_episode_every_n_windows,
        },
    }
    if not resume_run or not (results_dir / RUN_CONFIG_FILENAME).exists():
        _save_run_config(results_dir, run_config_payload)

    print("\n1. Loading MIMIC-demo data...")
    parser = MIMICDataParser(
        events_path=events_path,
        icu_stay_path=icu_stay_path,
    )
    parser.load_data()

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
    else:
        print("\n2. Selecting balanced patient cohort...")
        selected_patients = select_balanced_patients(
            parser.icu_stay_df,
            n_survived=n_survived,
            n_died=n_died,
        )

    completed_keys = _collect_completed_patient_keys(results_dir) if resume_run else set()
    if completed_keys:
        total_selected = len(selected_patients)
        pending_mask = selected_patients.apply(
            lambda row: (int(row["subject_id"]), int(row["icu_stay_id"])) not in completed_keys,
            axis=1,
        )
        selected_patients = selected_patients[pending_mask].reset_index(drop=True)
        skipped = total_selected - len(selected_patients)
        print(f"   Resume mode: skipped {skipped} already-completed patients")

    print("\n3. Creating memory snapshots...")
    print("=" * 80)

    all_results: List[Dict[str, Any]] = []
    failed_patients: List[Dict[str, Any]] = []
    if resume_run:
        existing_records = load_memory_patient_records(results_dir)
        all_results.extend(existing_records)
        if existing_records:
            print(f"   Loaded {len(existing_records)} existing patient results from resume run")
    patient_data = [(idx, row) for idx, (_, row) in enumerate(selected_patients.iterrows(), 1)]

    def process_patient_wrapper(args):
        idx, patient_row = args
        return _process_single_patient_memory(
            patient_row=patient_row,
            parser=parser,
            config=config,
            enable_logging=enable_logging,
            verbose=verbose,
            patient_idx=idx,
            total_patients=len(selected_patients),
            results_dir=results_dir,
        )

    if patient_data:
        max_workers = min(num_workers, len(patient_data))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_patient = {executor.submit(process_patient_wrapper, data): data for data in patient_data}
            for future in as_completed(future_to_patient):
                idx, patient_row = future_to_patient[future]
                subject_id = int(patient_row["subject_id"])
                icu_stay_id = int(patient_row["icu_stay_id"])
                try:
                    result = future.result()
                except Exception as exc:
                    print(
                        f"   ERROR: Failed patient {subject_id}_{icu_stay_id} "
                        f"({idx}/{len(selected_patients)}): {exc}"
                    )
                    failed_patients.append(
                        {
                            "subject_id": subject_id,
                            "icu_stay_id": icu_stay_id,
                            "error": str(exc),
                        }
                    )
                    continue
                if result:
                    all_results.append(result)

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    if not all_results:
        print("No patient memory generated.")
        return {}

    total_patients = len(all_results)
    total_windows = sum(int(item.get("num_windows", 0)) for item in all_results)
    total_snapshots = sum(int(item.get("num_memory_snapshots", 0)) for item in all_results)
    agent_stat_totals = _sum_numeric_stats([item.get("agent_stats", {}) for item in all_results])

    print(f"\nTotal Patients: {total_patients}")
    print(f"Total Windows: {total_windows}")
    print(f"Total Memory Snapshots: {total_snapshots}")
    if failed_patients:
        print(f"Failed Patients: {len(failed_patients)}")
    print("\nMedEvo Pipeline Totals:")
    print(f"  EventAgent Calls: {int(agent_stat_totals.get('total_event_calls', 0))}")
    print(f"  InsightAgent Calls: {int(agent_stat_totals.get('total_insight_calls', 0))}")
    print(f"  EpisodeAgent Calls: {int(agent_stat_totals.get('total_episode_calls', 0))}")
    print(f"  Grounding Rejections: {int(agent_stat_totals.get('total_grounding_rejections', 0))}")
    print(f"  Insights Pruned: {int(agent_stat_totals.get('total_insights_pruned', 0))}")
    print(f"  Tokens Used: {int(agent_stat_totals.get('total_tokens_used', 0))}")

    aggregate = {
        "timestamp": timestamp,
        "experiment": "create_memory_med_evo",
        "results_dir": str(results_dir),
        "num_workers": num_workers,
        "total_patients": total_patients,
        "total_windows": total_windows,
        "total_memory_snapshots": total_snapshots,
        "failed_patients": failed_patients,
        "med_evo_agent_stats": {k: int(v) for k, v in agent_stat_totals.items()},
        "individual_results": sorted(all_results, key=lambda item: (item["subject_id"], item["icu_stay_id"])),
    }

    with open(results_dir / AGGREGATE_FILENAME, "w") as f:
        json.dump(aggregate, f, indent=2)

    return aggregate


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create MedEvo memory snapshots")
    parser.add_argument("--n-survived", type=int, default=1, help="Number of survived patients")
    parser.add_argument("--n-died", type=int, default=0, help="Number of died patients")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for patient processing (default: 4).",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--no-logging", action="store_true", help="Disable detailed LLM call logging")
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
        "--resume-run",
        type=str,
        default=None,
        help=(
            "Optional existing memory run directory to resume in-place. "
            "Completed patient folders are skipped automatically."
        ),
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("RUNNING: CREATE MEMORY MED_EVO")
    print(f"{'='*80}\n")

    run_experiment(
        n_survived=args.n_survived,
        n_died=args.n_died,
        verbose=not args.quiet,
        enable_logging=not args.no_logging,
        patient_stay_ids_path=args.patient_stay_ids,
        num_workers=args.num_workers,
        resume_run=args.resume_run,
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
