"""Batch processing pipeline for Oracle offline evaluator."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from agents.oracle import MetaOracle
from config.config import Config, load_config
from data_parser import MIMICDataParser
from utils.llm_log_viewer import save_llm_calls_html
from utils.patient_selection import DEFAULT_SELECTION_SEED, select_balanced_patients


def _safe_status(report_dict: Dict[str, Any]) -> str:
    status = report_dict.get("patient_status", {}).get("overall_status")
    if isinstance(status, str) and status.strip():
        return status.strip().lower()
    return "unknown"


def _iter_trajectories_stream(
    parser: Any,
    max_patients: Optional[int],
    selected_stays: Optional[Any] = None,
) -> Iterator[Dict[str, Any]]:
    """Iterate trajectories lazily when parser supports streaming."""
    if selected_stays is not None:
        for _, icu_stay in selected_stays.iterrows():
            subject_id = int(icu_stay["subject_id"])
            icu_stay_id = int(icu_stay["icu_stay_id"])
            try:
                yield parser.get_patient_trajectory(
                    subject_id=subject_id,
                    icu_stay_id=icu_stay_id,
                    icu_stay=icu_stay,
                )
            except Exception as e:
                print(f"Error processing ICU stay {icu_stay_id}: {e}")
                continue
        return

    if hasattr(parser, "iter_trajectories"):
        yield from parser.iter_trajectories(max_patients=max_patients)
        return

    try:
        trajectories = parser.get_all_trajectories(max_patients=max_patients)
    except TypeError:
        trajectories = parser.get_all_trajectories()
        if max_patients is not None:
            trajectories = trajectories[:max_patients]

    for trajectory in trajectories:
        yield trajectory


def _planned_total_patients(
    parser: Any,
    max_patients: Optional[int],
    selected_stays: Optional[Any] = None,
) -> Optional[int]:
    """Best-effort estimate of total trajectories without materializing all trajectories."""
    if selected_stays is not None:
        return len(selected_stays)

    icu_stay_df = getattr(parser, "icu_stay_df", None)
    if icu_stay_df is None:
        return None

    total = len(icu_stay_df)
    if max_patients is not None:
        return min(total, max_patients)
    return total


def _select_balanced_oracle_cohort(
    parser: Any,
    *,
    n_survived: Optional[int],
    n_died: Optional[int],
    selection_seed: Optional[int],
) -> Optional[Any]:
    """Return selected ICU stays for balanced survive/died sampling, or None when disabled."""
    if n_survived is None and n_died is None:
        return None
    if n_survived is None or n_died is None:
        raise ValueError("Both n_survived and n_died must be provided together.")
    if n_survived < 0 or n_died < 0:
        raise ValueError("n_survived and n_died must be >= 0")

    icu_stay_df = getattr(parser, "icu_stay_df", None)
    if icu_stay_df is None:
        raise ValueError("Balanced sampling requires parser.icu_stay_df to be loaded.")

    return select_balanced_patients(
        icu_stay_df,
        n_survived=int(n_survived),
        n_died=int(n_died),
        random_seed=selection_seed,
    ).reset_index(drop=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = float("inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_float(value: Any) -> Optional[float]:
    parsed = _safe_float(value, default=float("nan"))
    if math.isfinite(parsed):
        return float(parsed)
    return None


def _normalize_gender(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    normalized = text.lower()
    if normalized in {"none", "nan", "null", "nat"}:
        return None
    if normalized in {"m", "male"}:
        return "Male"
    if normalized in {"f", "female"}:
        return "Female"
    return text


def _format_icu_outcome(survived: Any, death_time: Any) -> str:
    if survived is True:
        return "Survived after ICU"
    if survived is False:
        return "Died after ICU"

    survived_text = str(survived).strip().lower()
    if survived_text in {"true", "1", "yes", "y", "survived", "alive"}:
        return "Survived after ICU"
    if survived_text in {"false", "0", "no", "n", "died", "dead", "deceased"}:
        return "Died after ICU"

    death_time_text = str(death_time).strip().lower() if death_time is not None else ""
    if death_time_text and death_time_text not in {"none", "nan", "nat"}:
        return "Died after ICU"
    return "Unknown"


def _get_first_window_patient_metadata(windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    for raw_window in windows:
        if not isinstance(raw_window, dict):
            continue
        metadata = raw_window.get("patient_metadata")
        if isinstance(metadata, dict):
            return metadata
    return {}


def _build_patient_context_metadata(trajectory: Dict[str, Any], windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    first_window_metadata = _get_first_window_patient_metadata(windows)

    age = _coerce_optional_float(trajectory.get("age_at_admission"))
    if age is None:
        age = _coerce_optional_float(first_window_metadata.get("age"))

    gender = _normalize_gender(trajectory.get("gender"))
    if gender is None:
        gender = _normalize_gender(first_window_metadata.get("gender"))

    total_icu_stay_hours = _coerce_optional_float(trajectory.get("icu_duration_hours"))
    if total_icu_stay_hours is None:
        total_icu_stay_hours = _coerce_optional_float(first_window_metadata.get("total_icu_duration_hours"))

    survived = trajectory.get("survived")
    if survived is None:
        survived = first_window_metadata.get("survived")

    death_time = trajectory.get("death_time")
    if death_time is None:
        death_time = first_window_metadata.get("death_time")

    return {
        "age": age,
        "gender": gender,
        "total_icu_stay_hours": total_icu_stay_hours,
        "icu_outcome": _format_icu_outcome(survived=survived, death_time=death_time),
    }


def _build_trajectory_metadata(trajectory: Dict[str, Any], windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    patient_context_metadata = _build_patient_context_metadata(trajectory, windows)
    return {
        "enter_time": trajectory.get("enter_time"),
        "leave_time": trajectory.get("leave_time"),
        "age": patient_context_metadata.get("age"),
        "gender": patient_context_metadata.get("gender"),
        "total_icu_stay_hours": patient_context_metadata.get("total_icu_stay_hours"),
        "icu_duration_hours": trajectory.get("icu_duration_hours"),
        "survived": trajectory.get("survived"),
        "icu_outcome": patient_context_metadata.get("icu_outcome"),
        "death_time": trajectory.get("death_time"),
        "total_events": len(trajectory.get("events", [])),
    }


def _llm_call_sort_key(call: Dict[str, Any]) -> tuple[int, float, str, str]:
    window_index = _safe_int(call.get("window_index"), default=-1)
    if window_index < 0:
        window_index = 10**9
    hours_since_admission = _safe_float(call.get("hours_since_admission"), default=float("inf"))
    timestamp = str(call.get("timestamp") or "")
    step_type = str(call.get("step_type") or "")
    return (window_index, hours_since_admission, timestamp, step_type)


def _sort_llm_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    indexed_calls = list(enumerate(calls))
    indexed_calls.sort(
        key=lambda item: (
            0 if isinstance(item[1], dict) else 1,
            _llm_call_sort_key(item[1]) if isinstance(item[1], dict) else (10**9, float("inf"), "", ""),
            item[0],
        )
    )
    return [call for _, call in indexed_calls]


def _build_patient_predictions_payload(
    *,
    run_id: str,
    trajectory: Dict[str, Any],
    windows: List[Dict[str, Any]],
    reports: List[Any],
    llm_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    patient_context_metadata = _build_patient_context_metadata(trajectory, windows)
    trajectory_metadata = _build_trajectory_metadata(trajectory, windows)
    parsed_outputs_by_window_index: Dict[int, Dict[str, Any]] = {}
    for call in _sort_llm_calls(llm_calls):
        if not isinstance(call, dict):
            continue
        if str(call.get("step_type")) != "oracle_evaluator":
            continue
        parsed_response = call.get("parsed_response")
        if not isinstance(parsed_response, dict):
            continue
        try:
            window_index = int(call.get("window_index"))
        except (TypeError, ValueError):
            continue
        parsed_outputs_by_window_index[window_index] = parsed_response

    window_outputs = []
    for idx, window in enumerate(windows, start=1):
        raw_current_events = window.get("current_events")
        if not isinstance(raw_current_events, list):
            raw_current_events = []
        window_outputs.append(
            {
                "window_index": idx,
                "window_metadata": {
                    "subject_id": window.get("subject_id"),
                    "icu_stay_id": window.get("icu_stay_id"),
                    "age": patient_context_metadata.get("age"),
                    "gender": patient_context_metadata.get("gender"),
                    "total_icu_stay_hours": patient_context_metadata.get("total_icu_stay_hours"),
                    "icu_outcome": patient_context_metadata.get("icu_outcome"),
                    "window_start_time": window.get("current_window_start"),
                    "window_end_time": window.get("current_window_end"),
                    "hours_since_admission": window.get("hours_since_admission"),
                    "current_window_hours": window.get("current_window_hours"),
                    "num_history_events": window.get("num_history_events"),
                    "num_current_events": window.get("num_current_events"),
                    "pre_icu_history_source": window.get("pre_icu_history_source"),
                    "pre_icu_history_items": window.get("pre_icu_history_items"),
                },
                "raw_current_events": raw_current_events,
                "oracle_output": parsed_outputs_by_window_index.get(
                    idx - 1,
                    parsed_outputs_by_window_index.get(idx, {}),
                ),
            }
        )

    return {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "subject_id": trajectory.get("subject_id"),
        "icu_stay_id": trajectory.get("icu_stay_id"),
        "trajectory_metadata": trajectory_metadata,
        "num_windows_requested": len(windows),
        "num_windows_evaluated": len(reports),
        "window_outputs": window_outputs,
    }


def _build_oracle_llm_calls_payload(
    *,
    subject_id: Any,
    icu_stay_id: Any,
    provider: Optional[str],
    model: Optional[str],
    include_icu_outcome_in_prompt: Optional[bool],
    calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    sorted_calls = _sort_llm_calls(calls)
    prompt_outcome_mode = "unknown"
    if include_icu_outcome_in_prompt is True:
        prompt_outcome_mode = "with_icu_outcome"
    elif include_icu_outcome_in_prompt is False:
        prompt_outcome_mode = "without_icu_outcome"

    step_types = {str(call.get("step_type")) for call in sorted_calls if isinstance(call, dict)}
    pipeline_agents = [
        {
            "name": "oracle_pre_icu_history_compressor",
            "used": "oracle_pre_icu_history_compressor" in step_types,
            "thinking": None,
        },
        {"name": "oracle_evaluator", "used": "oracle_evaluator" in step_types, "thinking": None},
    ]
    return {
        "patient_id": f"{subject_id}_{icu_stay_id}",
        "agent_type": "oracle",
        "llm_provider": provider,
        "llm_model": model,
        "include_icu_outcome_in_prompt": include_icu_outcome_in_prompt,
        "prompt_outcome_mode": prompt_outcome_mode,
        "pipeline_agents": pipeline_agents,
        "total_calls": len(sorted_calls),
        "calls": sorted_calls,
    }


PROMPT_SECTION_HEADINGS = {
    "icu_discharge_summary": ("## CURRENT DISCHARGE SUMMARY",),
    "icu_trajectory_context_window": ("## ICU TRAJECTORY CONTEXT WINDOW",),
    # Backward-compatible heading variants.
    "previous_events_current_window": (
        "## HISTORY EVENTS OF CURRENT WINDOW",
        "## PREVIOUS EVENTS OF CURRENT WINDOW",
    ),
    "current_observation_window": ("## CURRENT OBSERVATION WINDOW FOR EVALUATION",),
}


def _extract_prompt_section(prompt_text: str, headings: tuple[str, ...]) -> str:
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return ""
    if not isinstance(headings, tuple) or len(headings) == 0:
        return ""
    for heading in headings:
        if not isinstance(heading, str) or not heading.strip():
            continue
        escaped_heading = re.escape(heading.strip())
        match = re.search(rf"{escaped_heading}\n([\s\S]*?)(?=\n##\s+|\Z)", prompt_text, flags=re.IGNORECASE)
        if match:
            return str(match.group(1) or "").strip()
    return ""


def _extract_prompt_sections(prompt_text: str) -> Dict[str, str]:
    return {key: _extract_prompt_section(prompt_text, heading) for key, heading in PROMPT_SECTION_HEADINGS.items()}


def _normalize_events(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


def _build_context_events_by_window_index(llm_calls: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    context_by_index: Dict[int, Dict[str, Any]] = {}
    for call in _sort_llm_calls(llm_calls):
        if not isinstance(call, dict):
            continue
        if str(call.get("step_type")) != "oracle_evaluator":
            continue
        try:
            window_index = int(call.get("window_index"))
        except (TypeError, ValueError):
            continue

        metadata = call.get("metadata")
        if not isinstance(metadata, dict):
            continue

        history_events = _normalize_events(metadata.get("context_history_events"))
        current_events = _normalize_events(metadata.get("context_current_window_events"))
        future_events = _normalize_events(metadata.get("context_future_events"))
        history_hours = _safe_float(metadata.get("history_hours"), default=float("nan"))
        future_hours = _safe_float(metadata.get("future_hours"), default=float("nan"))

        has_payload = (
            bool(history_events)
            or bool(current_events)
            or bool(future_events)
            or math.isfinite(history_hours)
            or math.isfinite(future_hours)
        )
        if not has_payload:
            continue

        context_by_index[window_index] = {
            "history_events": history_events,
            "current_events": current_events,
            "future_events": future_events,
            "history_hours": history_hours if math.isfinite(history_hours) else None,
            "future_hours": future_hours if math.isfinite(future_hours) else None,
        }

    return context_by_index


def _resolve_context_events_for_window(
    *,
    window_index: int,
    context_by_window_index: Dict[int, Dict[str, Any]],
) -> tuple[Optional[Dict[str, Any]], Optional[int]]:
    if not context_by_window_index:
        return None, None
    for candidate in (window_index - 1, window_index):
        if candidate in context_by_window_index:
            return context_by_window_index[candidate], candidate
    return None, None


def _build_prompt_sections_by_window_index(llm_calls: List[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
    sections_by_index: Dict[int, Dict[str, str]] = {}
    for call in _sort_llm_calls(llm_calls):
        if not isinstance(call, dict):
            continue
        if str(call.get("step_type")) != "oracle_evaluator":
            continue
        prompt_text = call.get("prompt")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            continue
        try:
            window_index = int(call.get("window_index"))
        except (TypeError, ValueError):
            continue
        sections_by_index[window_index] = _extract_prompt_sections(prompt_text)
    return sections_by_index


def _resolve_prompt_sections_for_window(
    *,
    window_index: int,
    sections_by_window_index: Dict[int, Dict[str, str]],
) -> tuple[Dict[str, str], Optional[int]]:
    empty = {key: "" for key in PROMPT_SECTION_HEADINGS}
    if not sections_by_window_index:
        return empty, None

    # LLM call window_index may be 0-based while predictions are 1-based.
    for candidate in (window_index - 1, window_index):
        if candidate in sections_by_window_index:
            sections = sections_by_window_index[candidate]
            return (
                {key: str(sections.get(key) or "") for key in PROMPT_SECTION_HEADINGS},
                candidate,
            )
    return empty, None


def _build_window_contexts_payload(
    *,
    run_id: str,
    trajectory: Dict[str, Any],
    windows: List[Dict[str, Any]],
    llm_calls: List[Dict[str, Any]],
    history_hours: Optional[float] = None,
    future_hours: Optional[float] = None,
) -> Dict[str, Any]:
    patient_context_metadata = _build_patient_context_metadata(trajectory, windows)
    trajectory_metadata = _build_trajectory_metadata(trajectory, windows)
    sections_by_window_index = _build_prompt_sections_by_window_index(llm_calls)
    context_events_by_window_index = _build_context_events_by_window_index(llm_calls)
    window_contexts: List[Dict[str, Any]] = []
    resolved_history_hours: Optional[float] = history_hours
    resolved_future_hours: Optional[float] = future_hours

    for idx, raw_window in enumerate(windows, start=1):
        window = raw_window if isinstance(raw_window, dict) else {}
        current_events = window.get("current_events")
        if not isinstance(current_events, list):
            current_events = []
        source_history_events = window.get("history_events")
        if not isinstance(source_history_events, list):
            source_history_events = []

        prompt_sections, llm_window_index = _resolve_prompt_sections_for_window(
            window_index=idx,
            sections_by_window_index=sections_by_window_index,
        )
        context_events_payload, context_events_llm_window_index = _resolve_context_events_for_window(
            window_index=idx,
            context_by_window_index=context_events_by_window_index,
        )

        oracle_context_history_events = (
            _normalize_events(context_events_payload.get("history_events"))
            if isinstance(context_events_payload, dict)
            else []
        )
        oracle_context_current_events = (
            _normalize_events(context_events_payload.get("current_events"))
            if isinstance(context_events_payload, dict)
            else []
        )
        oracle_context_future_events = (
            _normalize_events(context_events_payload.get("future_events"))
            if isinstance(context_events_payload, dict)
            else []
        )
        window_history_hours = (
            context_events_payload.get("history_hours")
            if isinstance(context_events_payload, dict)
            else None
        )
        window_future_hours = (
            context_events_payload.get("future_hours")
            if isinstance(context_events_payload, dict)
            else None
        )
        if not isinstance(window_history_hours, (int, float)) or not math.isfinite(float(window_history_hours)):
            window_history_hours = history_hours
        else:
            window_history_hours = float(window_history_hours)

        if not isinstance(window_future_hours, (int, float)) or not math.isfinite(float(window_future_hours)):
            window_future_hours = future_hours
        else:
            window_future_hours = float(window_future_hours)

        if resolved_history_hours is None and isinstance(window_history_hours, float):
            resolved_history_hours = window_history_hours
        if resolved_future_hours is None and isinstance(window_future_hours, float):
            resolved_future_hours = window_future_hours

        history_events = oracle_context_history_events if isinstance(context_events_payload, dict) else source_history_events
        context_events_source = (
            "oracle_context_events"
            if isinstance(context_events_payload, dict)
            else "window_payload_history_events"
        )

        window_contexts.append(
            {
                "window_index": idx,
                "llm_window_index": llm_window_index,
                "context_events_llm_window_index": context_events_llm_window_index,
                "window_metadata": {
                    "subject_id": window.get("subject_id"),
                    "icu_stay_id": window.get("icu_stay_id"),
                    "age": patient_context_metadata.get("age"),
                    "gender": patient_context_metadata.get("gender"),
                    "total_icu_stay_hours": patient_context_metadata.get("total_icu_stay_hours"),
                    "icu_outcome": patient_context_metadata.get("icu_outcome"),
                    "window_start_time": window.get("current_window_start"),
                    "window_end_time": window.get("current_window_end"),
                    "hours_since_admission": window.get("hours_since_admission"),
                    "current_window_hours": window.get("current_window_hours"),
                    "num_history_events": len(history_events),
                    "num_current_events": len(current_events),
                    "num_future_events": len(oracle_context_future_events),
                    "pre_icu_history_source": window.get("pre_icu_history_source"),
                    "pre_icu_history_items": window.get("pre_icu_history_items"),
                    "current_discharge_summary_selection_rule": (
                        (window.get("current_discharge_summary", {}) or {}).get("selection_rule")
                        if isinstance(window.get("current_discharge_summary"), dict)
                        else None
                    ),
                    "history_hours": window_history_hours,
                    "future_hours": window_future_hours,
                },
                "history_events": history_events,
                "current_events": current_events,
                "future_events": oracle_context_future_events,
                "source_window_history_events": source_history_events,
                "oracle_context_history_events": oracle_context_history_events,
                "oracle_context_current_events": oracle_context_current_events,
                "oracle_context_future_events": oracle_context_future_events,
                "context_events_source": context_events_source,
                "prompt_sections": prompt_sections,
                "current_discharge_summary": (
                    window.get("current_discharge_summary")
                    if isinstance(window.get("current_discharge_summary"), dict)
                    else None
                ),
                "pre_icu_history": (
                    window.get("pre_icu_history") if isinstance(window.get("pre_icu_history"), dict) else None
                ),
            }
        )

    return {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "subject_id": trajectory.get("subject_id"),
        "icu_stay_id": trajectory.get("icu_stay_id"),
        "trajectory_metadata": trajectory_metadata,
        "history_hours": resolved_history_hours,
        "future_hours": resolved_future_hours,
        "window_contexts": window_contexts,
    }


def process_batch_for_oracle(
    config: Config,
    events_path: str,
    icu_stay_path: str,
    output_dir: str,
    provider: str = "anthropic",
    model: Optional[str] = None,
    current_window_hours: float = 0.5,
    window_step_hours: float = 0.5,
    include_pre_icu_data: bool = True,
    use_discharge_summary: Optional[bool] = None,
    compress_pre_icu_history: Optional[bool] = None,
    include_icu_outcome_in_prompt: Optional[bool] = None,
    max_patients: Optional[int] = None,
    n_survived: Optional[int] = None,
    n_died: Optional[int] = None,
    selection_seed: Optional[int] = DEFAULT_SELECTION_SEED,
    window_workers: int = 4,
) -> None:
    """Process a batch of patient trajectories through Oracle."""
    if max_patients is not None and max_patients < 0:
        raise ValueError("max_patients must be >= 0")
    if window_workers < 1:
        raise ValueError("window_workers must be >= 1")

    selected_include_icu_outcome_in_prompt = (
        config.oracle_context_include_icu_outcome_in_prompt
        if include_icu_outcome_in_prompt is None
        else bool(include_icu_outcome_in_prompt)
    )
    prompt_mode_suffix = "with_outcome" if selected_include_icu_outcome_in_prompt else "without_outcome"

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"oracle_{run_timestamp}_{prompt_mode_suffix}"
    run_dir = output_root / run_id
    patients_dir = run_dir / "patients"
    patients_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MIMIC-DEMO ORACLE PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Output Run Directory: {run_dir}")

    print("\nInitializing data parser...")
    parser = MIMICDataParser(events_path, icu_stay_path)
    parser.load_data()

    print("\nInitializing Meta Oracle...")
    print(f"  Provider: {provider}")
    print(f"  Model: {model or 'default'}")
    print(f"  Current window: {current_window_hours} hours ({current_window_hours * 60:.0f} minutes)")
    print(f"  Window step size: {window_step_hours} hours ({window_step_hours * 60:.0f} minutes)")
    print(f"  Include pre-ICU data: {include_pre_icu_data}")
    print(f"  Window workers: {window_workers}")
    print(f"  Pre-ICU report codes: {config.oracle_relative_report_codes}")
    print(f"  Pre-ICU history hours: {config.oracle_pre_icu_history_hours}")
    selected_use_discharge_summary = (
        config.oracle_context_use_discharge_summary if use_discharge_summary is None else bool(use_discharge_summary)
    )
    selected_compress_pre_icu_history = (
        bool(getattr(config, "oracle_context_compress_pre_icu_history", True))
        if compress_pre_icu_history is None
        else bool(compress_pre_icu_history)
    )
    print(f"  Use ICU discharge summary in context: {selected_use_discharge_summary}")
    print(f"  Compress pre-ICU history once per patient: {selected_compress_pre_icu_history}")
    print(f"  Include ICU outcome in prompt: {selected_include_icu_outcome_in_prompt}")
    print(f"  Context history threshold (hours): {config.oracle_context_history_hours}")
    print(f"  Context future threshold (hours): {config.oracle_context_future_hours}")
    print(f"  Top-k recommendations requested: {config.oracle_context_top_k_recommendations}")

    oracle = MetaOracle(
        provider=provider,
        model=model,
        temperature=1.0,
        max_tokens=4096,
        use_discharge_summary=selected_use_discharge_summary,
        include_icu_outcome_in_prompt=selected_include_icu_outcome_in_prompt,
        compress_pre_icu_history=selected_compress_pre_icu_history,
        history_context_hours=config.oracle_context_history_hours,
        future_context_hours=config.oracle_context_future_hours,
        top_k_recommendations=config.oracle_context_top_k_recommendations,
        log_dir=config.oracle_log_dir,
    )

    print("\nPreparing trajectory stream...")
    selected_stays = _select_balanced_oracle_cohort(
        parser,
        n_survived=n_survived,
        n_died=n_died,
        selection_seed=selection_seed,
    )
    planned_total = _planned_total_patients(parser, max_patients, selected_stays=selected_stays)

    if selected_stays is not None:
        print(
            "  Using balanced cohort selection "
            f"(survived={int(n_survived or 0)}, died={int(n_died or 0)}, seed={selection_seed})"
        )
        if max_patients is not None:
            print("  NOTE: --max-patients is ignored when balanced cohort selection is enabled.")
    elif max_patients is not None:
        print(f"  Max patients: {max_patients}")
    if planned_total is not None:
        print(f"  Total trajectories to process: {planned_total}")
    else:
        print("  Total trajectories to process: unknown (streamed)")

    json_default = getattr(parser, "_json_default", _json_default)

    print(f"\n{'=' * 80}")
    print("PROCESSING TRAJECTORIES")
    print("=" * 80)

    summary_stats: Dict[str, Any] = {
        "total_patients": planned_total if planned_total is not None else 0,
        "total_windows_evaluated": 0,
        "patients_processed": 0,
        "patients_failed": 0,
        "overall_status_distribution": {},
        "total_doctor_actions": 0,
    }
    total_seen = 0

    for i, trajectory in enumerate(
        _iter_trajectories_stream(parser, max_patients, selected_stays=selected_stays),
        start=1,
    ):
        total_seen = i

        subject_id = trajectory["subject_id"]
        icu_stay_id = trajectory["icu_stay_id"]
        progress = f"{i}/{planned_total}" if planned_total is not None else str(i)

        print(f"\n[{progress}] Processing Patient {subject_id}, ICU Stay {icu_stay_id}")
        print(f"  Duration: {trajectory['icu_duration_hours']:.1f} hours")
        print(f"  Outcome: {'Survived' if trajectory['survived'] else 'Died'}")

        try:
            windows = parser.create_time_windows(
                trajectory,
                current_window_hours=current_window_hours,
                window_step_hours=window_step_hours,
                include_pre_icu_data=include_pre_icu_data,
                use_first_n_hours_after_icu=config.oracle_observation_hours,
                use_discharge_summary_for_history=config.oracle_use_discharge_summary_for_history,
                num_discharge_summaries=config.oracle_num_discharge_summaries,
                relative_report_codes=config.oracle_relative_report_codes,
                pre_icu_history_hours=config.oracle_pre_icu_history_hours,
            )

            print(f"  Generated {len(windows)} time windows")
            if len(windows) == 0:
                print("  Skipping (no windows generated)")
                continue

            if window_workers > 1 and len(windows) > 1:
                effective_workers = min(window_workers, len(windows))
                print(f"  Evaluating windows in parallel ({effective_workers} workers)")
                reports = oracle.evaluate_trajectory_parallel(
                    windows,
                    trajectory=trajectory,
                    max_workers=effective_workers,
                    show_progress=True,
                )
            else:
                reports = oracle.evaluate_trajectory(windows, trajectory=trajectory)

            print(f"  Completed: {len(reports)} evaluations")

            patient_status_counts: Dict[str, int] = {}
            doctor_actions_count = 0
            for report in reports:
                report_dict = report.to_dict()
                status = _safe_status(report_dict)
                patient_status_counts[status] = patient_status_counts.get(status, 0) + 1
                doctor_actions_count += len(report_dict.get("doctor_actions", []))

                summary_stats["overall_status_distribution"][status] = (
                    summary_stats["overall_status_distribution"].get(status, 0) + 1
                )

            summary_stats["total_doctor_actions"] += doctor_actions_count

            print(f"  Status distribution: {patient_status_counts}")
            print(f"  Total doctor actions extracted: {doctor_actions_count}")

            patient_dir = patients_dir / f"{subject_id}_{icu_stay_id}"
            patient_dir.mkdir(parents=True, exist_ok=True)

            llm_calls = oracle.pop_patient_llm_call_logs(subject_id=subject_id, icu_stay_id=icu_stay_id)
            oracle.pop_patient_trajectory_logs(subject_id=subject_id, icu_stay_id=icu_stay_id)
            patient_predictions = _build_patient_predictions_payload(
                run_id=run_id,
                trajectory=trajectory,
                windows=windows,
                reports=reports,
                llm_calls=llm_calls,
            )
            patient_predictions_path = patient_dir / "oracle_predictions.json"
            with open(patient_predictions_path, "w", encoding="utf-8") as f:
                json.dump(patient_predictions, f, indent=2, ensure_ascii=False, default=json_default)

            llm_payload = _build_oracle_llm_calls_payload(
                subject_id=subject_id,
                icu_stay_id=icu_stay_id,
                provider=getattr(oracle.llm_client, "provider", None),
                model=getattr(oracle.llm_client, "model", None),
                include_icu_outcome_in_prompt=getattr(oracle, "include_icu_outcome_in_prompt", None),
                calls=llm_calls,
            )
            llm_calls_json = patient_dir / "llm_calls.json"
            with open(llm_calls_json, "w", encoding="utf-8") as f:
                json.dump(llm_payload, f, indent=2, ensure_ascii=False, default=json_default)
            save_llm_calls_html(llm_payload, patient_dir / "llm_calls.html")

            window_contexts_payload = _build_window_contexts_payload(
                run_id=run_id,
                trajectory=trajectory,
                windows=windows,
                llm_calls=llm_calls,
                history_hours=float(config.oracle_context_history_hours),
                future_hours=float(config.oracle_context_future_hours),
            )
            window_contexts_path = patient_dir / "window_contexts.json"
            with open(window_contexts_path, "w", encoding="utf-8") as f:
                json.dump(window_contexts_payload, f, indent=2, ensure_ascii=False, default=json_default)

            summary_stats["total_windows_evaluated"] += len(reports)
            summary_stats["patients_processed"] += 1

        except Exception as e:
            print(f"  ERROR: Failed to process patient: {e}")
            summary_stats["patients_failed"] += 1
            oracle.pop_patient_trajectory_logs(subject_id=subject_id, icu_stay_id=icu_stay_id)
            oracle.pop_patient_llm_call_logs(subject_id=subject_id, icu_stay_id=icu_stay_id)
            continue

    if planned_total is None:
        summary_stats["total_patients"] = total_seen

    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print("=" * 80)

    oracle_stats = oracle.get_statistics()
    summary_stats.update(oracle_stats)
    summary_stats["run_id"] = run_id
    summary_stats["run_directory"] = str(run_dir)

    summary_file = run_dir / "processing_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)

    print("\nSummary:")
    print(f"  Patients processed: {summary_stats['patients_processed']}/{summary_stats['total_patients']}")
    print(f"  Patients failed: {summary_stats['patients_failed']}")
    print(f"  Total windows evaluated: {summary_stats['total_windows_evaluated']}")
    print(f"  Total doctor actions: {summary_stats['total_doctor_actions']}")
    print(f"  Status distribution: {summary_stats['overall_status_distribution']}")
    print(f"  Total tokens used: {summary_stats['total_tokens_used']:,}")
    print(f"  Avg tokens per evaluation: {summary_stats['avg_tokens_per_evaluation']:.0f}")

    print(f"\nOutputs saved to: {run_dir}")
    print(f"  - Summary: {summary_file.name}")
    print("  - Per-patient predictions: patients/<subject_id>_<icu_stay_id>/oracle_predictions.json")
    print("  - Per-patient window context: patients/<subject_id>_<icu_stay_id>/window_contexts.json")

    print(f"\n{'=' * 80}")
    print("PROCESSING COMPLETE")
    print("=" * 80)


def main() -> None:
    """Main entry point for Oracle batch processing."""
    config = load_config()

    parser = argparse.ArgumentParser(description="Process MIMIC-demo data through Meta Oracle for offline evaluation")

    parser.add_argument(
        "--events",
        type=str,
        default=config.events_path,
        help=f"Path to events parquet file (default: {config.events_path})",
    )

    parser.add_argument(
        "--icu-stay",
        type=str,
        default=config.icu_stay_path,
        help=f"Path to ICU stay parquet file (default: {config.icu_stay_path})",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=config.output_dir,
        help=f"Output root directory for timestamped Oracle runs (default: {config.output_dir})",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=config.llm_provider,
        choices=["anthropic", "openai", "google", "gemini"],
        help=f"LLM provider (default: {config.llm_provider})",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=config.llm_model,
        help=f"Model name (default: {config.llm_model or 'provider default'})",
    )

    parser.add_argument(
        "--current-window-hours",
        type=float,
        default=config.oracle_current_window_hours,
        help=(f"Size of current observation window in hours " f"(default: {config.oracle_current_window_hours})"),
    )

    parser.add_argument(
        "--window-step-hours",
        type=float,
        default=config.oracle_window_step_hours,
        help=f"Step size between sliding windows in hours (default: {config.oracle_window_step_hours})",
    )

    parser.add_argument(
        "--window-workers",
        type=int,
        default=4,
        help="Number of concurrent window evaluations per patient (default: 4, set 1 to disable)",
    )

    parser.add_argument(
        "--no-pre-icu-data",
        action="store_true",
        help="Exclude pre-ICU hospital data from history context",
    )

    parser.add_argument(
        "--use-discharge-summary",
        action="store_true",
        default=None,
        help=(
            "Include ICU discharge summary block in Oracle context. "
            "If omitted, uses config oracle_context.use_discharge_summary"
        ),
    )

    parser.add_argument(
        "--compress-pre-icu-history",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether to LLM-compress pre-ICU history once per patient and reuse it across windows. "
            "If omitted, uses config oracle_context.compress_pre_icu_history."
        ),
    )

    parser.add_argument(
        "--include-icu-outcome-in-prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether to include ICU outcome (survived/died) in Oracle prompt context. "
            "If omitted, uses config oracle_context.include_icu_outcome_in_prompt."
        ),
    )

    parser.add_argument(
        "--max-patients",
        type=int,
        default=config.max_patients,
        help="Maximum number of patients to process (for testing)",
    )

    parser.add_argument(
        "--n-survived",
        type=int,
        default=None,
        help="Select this many survived ICU stays (requires --n-died).",
    )

    parser.add_argument(
        "--n-died",
        type=int,
        default=None,
        help="Select this many died ICU stays (requires --n-survived).",
    )

    parser.add_argument(
        "--selection-seed",
        type=int,
        default=DEFAULT_SELECTION_SEED,
        help=f"Random seed for balanced patient sampling (default: {DEFAULT_SELECTION_SEED}).",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.json file",
    )

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        print(f"Loaded custom config from: {args.config}")

    process_batch_for_oracle(
        config=config,
        events_path=args.events,
        icu_stay_path=args.icu_stay,
        output_dir=args.output,
        provider=args.provider,
        model=args.model,
        current_window_hours=args.current_window_hours,
        window_step_hours=args.window_step_hours,
        include_pre_icu_data=not args.no_pre_icu_data,
        use_discharge_summary=args.use_discharge_summary,
        compress_pre_icu_history=args.compress_pre_icu_history,
        include_icu_outcome_in_prompt=args.include_icu_outcome_in_prompt,
        max_patients=args.max_patients,
        n_survived=args.n_survived,
        n_died=args.n_died,
        selection_seed=args.selection_seed,
        window_workers=args.window_workers,
    )


if __name__ == "__main__":
    main()
