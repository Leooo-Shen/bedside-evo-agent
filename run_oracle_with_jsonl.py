"""Run Oracle inference from JSONL windows (one window per line)."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.oracle import MetaOracle, OracleReport
from config.config import Config, load_config
from run_oracle import (
    _build_oracle_llm_calls_payload,
    _build_patient_predictions_payload,
    _build_window_contexts_payload,
    _json_default,
    _safe_status,
)
from utils.llm_log_viewer import save_llm_calls_html

TMP_PROGRESS_FILENAME = "tmp_progress.json"
REQUIRED_PATIENT_OUTPUT_FILES = (
    "oracle_predictions.json",
    "llm_calls.json",
    "window_contexts.json",
)
ORACLE_COUNTER_KEYS = (
    "total_evaluations",
    "total_llm_calls",
    "total_tokens_used",
    "total_pre_icu_compression_calls",
    "total_pre_icu_compression_tokens",
)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_events(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


def _load_jsonl_records(path: Path, max_windows: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {path}")
    if max_windows is not None and max_windows < 0:
        raise ValueError("--max-windows must be >= 0")
    if max_windows == 0:
        return []

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_no} is not a JSON object.")
            payload = dict(payload)
            payload["_line_number"] = line_no
            records.append(payload)
            if max_windows is not None and len(records) >= max_windows:
                break
    return records


def _window_sort_key(window: Dict[str, Any]) -> Tuple[int, float, int]:
    source_window_index = _safe_int(window.get("window_index"), default=10**9)
    hours_since_admission = _safe_float(window.get("hours_since_admission"), default=float("inf"))
    line_number = _safe_int(window.get("_line_number"), default=10**9)
    return (source_window_index, hours_since_admission, line_number)


def _prepare_window_for_evaluation(raw_window: Dict[str, Any], local_index: int) -> Dict[str, Any]:
    window = dict(raw_window)
    window["subject_id"] = _safe_int(raw_window.get("subject_id"))
    window["icu_stay_id"] = _safe_int(raw_window.get("icu_stay_id"))
    window["history_events"] = _normalize_events(window.get("history_events"))
    window["current_events"] = _normalize_events(window.get("current_events"))
    window["future_events"] = _normalize_events(window.get("future_events"))
    window["num_history_events"] = len(window["history_events"])
    window["num_current_events"] = len(window["current_events"])
    window["num_future_events"] = len(window["future_events"])
    window["patient_metadata"] = (
        dict(window.get("patient_metadata")) if isinstance(window.get("patient_metadata"), dict) else {}
    )
    window["pre_icu_history"] = (
        dict(window.get("pre_icu_history")) if isinstance(window.get("pre_icu_history"), dict) else {}
    )
    window["current_discharge_summary"] = (
        dict(window.get("current_discharge_summary"))
        if isinstance(window.get("current_discharge_summary"), dict)
        else None
    )

    window["source_window_index"] = raw_window.get("window_index")
    window["source_window_position"] = raw_window.get("window_position")
    window["source_line_number"] = raw_window.get("_line_number")
    window["window_index"] = int(local_index)
    window["window_position"] = int(local_index + 1)
    return window


def _event_dedupe_key(event: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
    return (
        str(event.get("event_id") or ""),
        str(event.get("time") or event.get("start_time") or ""),
        str(event.get("code") or ""),
        str(event.get("code_specifics") or ""),
        str(event.get("numeric_value") or ""),
        str(event.get("text_value") or ""),
    )


def _build_patient_level_trajectory(windows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not windows:
        return {"events": []}

    first = windows[0]
    trajectory_metadata = first.get("trajectory_metadata")
    if not isinstance(trajectory_metadata, dict):
        trajectory_metadata = {}
    patient_metadata = first.get("patient_metadata")
    if not isinstance(patient_metadata, dict):
        patient_metadata = {}

    merged_events: List[Dict[str, Any]] = []
    seen = set()
    for window in windows:
        for event in [
            *window.get("history_events", []),
            *window.get("current_events", []),
            *window.get("future_events", []),
        ]:
            if not isinstance(event, dict):
                continue
            dedupe_key = _event_dedupe_key(event)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            merged_events.append(dict(event))

    merged_events.sort(key=lambda event: str(event.get("time") or event.get("start_time") or ""))

    return {
        "subject_id": first.get("subject_id"),
        "icu_stay_id": first.get("icu_stay_id"),
        "enter_time": trajectory_metadata.get("enter_time") or first.get("current_window_start"),
        "leave_time": trajectory_metadata.get("leave_time") or first.get("current_window_end"),
        "age_at_admission": trajectory_metadata.get("age_at_admission", patient_metadata.get("age")),
        "gender": trajectory_metadata.get("gender", patient_metadata.get("gender")),
        "icu_duration_hours": trajectory_metadata.get(
            "icu_duration_hours", patient_metadata.get("total_icu_duration_hours")
        ),
        "survived": trajectory_metadata.get("survived", patient_metadata.get("survived")),
        "death_time": trajectory_metadata.get("death_time", patient_metadata.get("death_time")),
        "events": merged_events,
    }


def _build_error_report(window: Dict[str, Any], error_message: str) -> OracleReport:
    return OracleReport(
        patient_assessment={
            "overall": {"label": "insufficient_data", "rationale": "Oracle evaluation failed."},
            "active_risks": [],
        },
        action_review={"evaluations": [], "red_flags": []},
        window_data=window,
        context_mode="raw_local_trajectory_icu_events_only",
        context_stats={},
        error=error_message,
    )


def _json_dump_atomic(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
    temp_path.replace(path)


def _json_load_dict(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(Path(text).expanduser().resolve())


def _resolve_checkpoint_path(value: str) -> Path:
    raw = Path(str(value).strip()).expanduser()
    if raw.is_dir():
        return raw / TMP_PROGRESS_FILENAME
    return raw


def process_jsonl_for_oracle(
    *,
    config: Config,
    input_jsonl: str,
    output_dir: str,
    full_windows_jsonl: Optional[str] = None,
    provider: str = "anthropic",
    model: Optional[str] = None,
    use_discharge_summary: Optional[bool] = None,
    compress_pre_icu_history: Optional[bool] = None,
    include_icu_outcome_in_prompt: Optional[bool] = None,
    stay_workers: int = 4,
    max_windows: Optional[int] = None,
    tmp_save_every_lines: Optional[int] = None,
    tmp_progress_path: Optional[str] = None,
    resume_from_tmp: Optional[str] = None,
) -> None:
    if stay_workers < 1:
        raise ValueError("--stay-workers must be >= 1")
    if tmp_save_every_lines is not None and tmp_save_every_lines < 1:
        raise ValueError("--tmp-save-every-lines must be >= 1")

    selected_use_discharge_summary = (
        config.oracle_context_use_discharge_summary if use_discharge_summary is None else bool(use_discharge_summary)
    )
    selected_compress_pre_icu_history = (
        bool(getattr(config, "oracle_context_compress_pre_icu_history", True))
        if compress_pre_icu_history is None
        else bool(compress_pre_icu_history)
    )
    selected_include_icu_outcome_in_prompt = (
        config.oracle_context_include_icu_outcome_in_prompt
        if include_icu_outcome_in_prompt is None
        else bool(include_icu_outcome_in_prompt)
    )
    prompt_mode_suffix = "with_outcome" if selected_include_icu_outcome_in_prompt else "without_outcome"

    resolved_input_jsonl = _normalize_optional_path(input_jsonl)
    resolved_full_windows_jsonl = _normalize_optional_path(full_windows_jsonl)

    resume_state: Optional[Dict[str, Any]] = None
    checkpoint_path: Optional[Path]
    if resume_from_tmp is not None:
        checkpoint_path = _resolve_checkpoint_path(resume_from_tmp)
        resume_state = _json_load_dict(checkpoint_path)
        if resume_state is None:
            raise ValueError(f"Invalid or missing tmp progress file: {checkpoint_path}")
        run_id_raw = resume_state.get("run_id")
        run_directory_raw = resume_state.get("run_directory")
        if not isinstance(run_id_raw, str) or not run_id_raw.strip():
            raise ValueError(f"Missing run_id in tmp progress file: {checkpoint_path}")
        if not isinstance(run_directory_raw, str) or not run_directory_raw.strip():
            raise ValueError(f"Missing run_directory in tmp progress file: {checkpoint_path}")
        run_id = str(run_id_raw).strip()
        run_dir = Path(str(run_directory_raw).strip()).expanduser().resolve()
        if tmp_progress_path is not None:
            requested_checkpoint = Path(tmp_progress_path).expanduser().resolve()
            if requested_checkpoint != checkpoint_path.resolve():
                raise ValueError(
                    "--tmp-progress-path must match --resume-from-tmp when both are provided: "
                    f"{requested_checkpoint} != {checkpoint_path.resolve()}"
                )
    else:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"oracle_{run_timestamp}_{prompt_mode_suffix}_jsonl"
        run_dir = output_root / run_id
        checkpoint_path = (
            Path(tmp_progress_path).expanduser().resolve()
            if tmp_progress_path is not None
            else (run_dir / TMP_PROGRESS_FILENAME).resolve()
        )

    patients_dir = run_dir / "patients"
    run_dir.mkdir(parents=True, exist_ok=True)
    patients_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MIMIC-DEMO ORACLE JSONL PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Input JSONL: {input_jsonl}")
    if full_windows_jsonl:
        print(f"Full-window JSONL: {full_windows_jsonl}")
    print(f"Output Run Directory: {run_dir}")
    if resume_state is not None:
        print(f"Resume tmp progress: {checkpoint_path}")

    print("\nLoading JSONL windows...")
    raw_records = _load_jsonl_records(Path(input_jsonl), max_windows=max_windows)
    if not raw_records:
        raise ValueError(f"No windows found in JSONL: {input_jsonl}")

    grouped: Dict[str, Dict[str, Any]] = {}
    patient_order: List[str] = []
    for record in raw_records:
        subject_id = _safe_int(record.get("subject_id"), default=-1)
        icu_stay_id = _safe_int(record.get("icu_stay_id"), default=-1)
        if subject_id < 0 or icu_stay_id < 0:
            line_number = _safe_int(record.get("_line_number"), default=-1)
            raise ValueError(
                f"Invalid subject_id/icu_stay_id at JSONL line {line_number}. "
                "Every row must contain numeric subject_id and icu_stay_id."
            )
        patient_id = f"{subject_id}_{icu_stay_id}"
        if patient_id not in grouped:
            grouped[patient_id] = {
                "subject_id": subject_id,
                "icu_stay_id": icu_stay_id,
                "raw_windows": [],
                "windows": [],
                "patient_trajectory": {},
            }
            patient_order.append(patient_id)
        grouped[patient_id]["raw_windows"].append(record)

    for patient_id in patient_order:
        raw_windows = grouped[patient_id]["raw_windows"]
        raw_windows.sort(key=_window_sort_key)

        windows: List[Dict[str, Any]] = []
        for local_index, raw_window in enumerate(raw_windows):
            window = _prepare_window_for_evaluation(raw_window, local_index=local_index)
            windows.append(window)

        grouped[patient_id]["windows"] = windows
        grouped[patient_id]["patient_trajectory"] = _build_patient_level_trajectory(windows)

    full_windows_by_patient: Dict[str, List[Dict[str, Any]]] = {}
    if full_windows_jsonl:
        print("\nLoading full-window JSONL...")
        full_raw_records = _load_jsonl_records(Path(full_windows_jsonl), max_windows=None)
        full_raw_by_patient: Dict[str, List[Dict[str, Any]]] = {}
        selected_patient_ids = set(patient_order)
        for record in full_raw_records:
            subject_id = _safe_int(record.get("subject_id"), default=-1)
            icu_stay_id = _safe_int(record.get("icu_stay_id"), default=-1)
            if subject_id < 0 or icu_stay_id < 0:
                continue
            patient_id = f"{subject_id}_{icu_stay_id}"
            if patient_id not in selected_patient_ids:
                continue
            if patient_id not in full_raw_by_patient:
                full_raw_by_patient[patient_id] = []
            full_raw_by_patient[patient_id].append(record)

        for patient_id in patient_order:
            raw_windows = full_raw_by_patient.get(patient_id, [])
            raw_windows.sort(key=_window_sort_key)
            prepared_windows: List[Dict[str, Any]] = []
            for local_index, raw_window in enumerate(raw_windows):
                prepared_windows.append(_prepare_window_for_evaluation(raw_window, local_index=local_index))
            full_windows_by_patient[patient_id] = prepared_windows

        matched_patients = sum(
            1 for patient_id in patient_order if len(full_windows_by_patient.get(patient_id, [])) > 0
        )
        print(f"  Loaded full windows for selected patients: {matched_patients}/{len(patient_order)}")

    print(f"  Loaded windows: {len(raw_records)}")
    print(f"  Patients in JSONL: {len(grouped)}")

    summary_stats: Dict[str, Any] = {
        "run_id": run_id,
        "run_directory": str(run_dir),
        "input_jsonl": str(input_jsonl),
        "full_windows_jsonl": str(full_windows_jsonl) if full_windows_jsonl else None,
        "total_patients": len(grouped),
        "total_windows_input": len(raw_records),
        "total_windows_evaluated": 0,
        "windows_failed": 0,
        "patients_processed": 0,
        "patients_resumed": 0,
        "patients_failed": 0,
        "patients_with_full_window_contexts": 0,
        "overall_status_distribution": {},
        "total_action_evaluations": 0,
        "total_evaluations": 0,
        "total_llm_calls": 0,
        "total_tokens_used": 0,
        "total_pre_icu_compression_calls": 0,
        "total_pre_icu_compression_tokens": 0,
        "avg_tokens_per_evaluation": 0.0,
        "use_discharge_summary": selected_use_discharge_summary,
        "include_icu_outcome_in_prompt": selected_include_icu_outcome_in_prompt,
        "history_context_hours": float(config.oracle_context_history_hours),
        "future_context_hours": float(config.oracle_context_future_hours),
        "top_k_recommendations": int(config.oracle_context_top_k_recommendations),
        "compress_pre_icu_history": selected_compress_pre_icu_history,
    }

    completed_patient_ids: List[str] = []
    completed_patient_id_set = set()
    if resume_state is not None:
        expected_state = {
            "run_id": run_id,
            "input_jsonl": resolved_input_jsonl,
            "full_windows_jsonl": resolved_full_windows_jsonl,
            "provider": provider,
            "model": model,
            "use_discharge_summary": selected_use_discharge_summary,
            "compress_pre_icu_history": selected_compress_pre_icu_history,
            "include_icu_outcome_in_prompt": selected_include_icu_outcome_in_prompt,
            "max_windows": max_windows,
            "total_patients": len(grouped),
            "total_windows_input": len(raw_records),
        }
        mismatched_keys = [key for key, value in expected_state.items() if resume_state.get(key) != value]
        if mismatched_keys:
            mismatch_details = ", ".join(
                f"{key}: expected={expected_state[key]!r} got={resume_state.get(key)!r}" for key in mismatched_keys
            )
            raise ValueError(
                "Tmp progress is incompatible with current run settings. " f"Mismatched fields: {mismatch_details}"
            )

        raw_completed_patient_ids = resume_state.get("completed_patient_ids")
        if not isinstance(raw_completed_patient_ids, list):
            raise ValueError("Invalid tmp progress: completed_patient_ids must be a list.")
        unknown_patient_ids: List[str] = []
        for raw_patient_id in raw_completed_patient_ids:
            patient_id = str(raw_patient_id)
            if patient_id in grouped:
                if patient_id not in completed_patient_id_set:
                    completed_patient_id_set.add(patient_id)
                    completed_patient_ids.append(patient_id)
                continue
            unknown_patient_ids.append(patient_id)
        if unknown_patient_ids:
            raise ValueError(
                "Tmp progress references patients not present in current input JSONL: " f"{unknown_patient_ids}"
            )

        for patient_id in completed_patient_ids:
            patient_dir = patients_dir / patient_id
            missing_files = [name for name in REQUIRED_PATIENT_OUTPUT_FILES if not (patient_dir / name).exists()]
            if missing_files:
                raise ValueError(
                    f"Tmp progress marks patient {patient_id} complete but files are missing: {missing_files}"
                )

        resumed_summary = resume_state.get("summary_stats")
        if isinstance(resumed_summary, dict):
            summary_stats.update(resumed_summary)
        summary_stats["run_id"] = run_id
        summary_stats["run_directory"] = str(run_dir)
        summary_stats["input_jsonl"] = str(input_jsonl)
        summary_stats["full_windows_jsonl"] = str(full_windows_jsonl) if full_windows_jsonl else None
        summary_stats["total_patients"] = len(grouped)
        summary_stats["total_windows_input"] = len(raw_records)
        summary_stats["patients_resumed"] = len(completed_patient_ids)
        if not isinstance(summary_stats.get("overall_status_distribution"), dict):
            summary_stats["overall_status_distribution"] = {}
        if _safe_int(summary_stats.get("patients_processed"), default=0) < len(completed_patient_ids):
            summary_stats["patients_processed"] = len(completed_patient_ids)
        completed_windows_from_resume = sum(
            len(grouped[patient_id]["windows"]) for patient_id in completed_patient_ids
        )
        if _safe_int(summary_stats.get("total_windows_evaluated"), default=0) < completed_windows_from_resume:
            summary_stats["total_windows_evaluated"] = completed_windows_from_resume

    effective_tmp_save_every_lines = tmp_save_every_lines
    if effective_tmp_save_every_lines is None and resume_state is not None:
        resumed_interval = _safe_int(resume_state.get("tmp_save_every_lines"), default=0)
        if resumed_interval > 0:
            effective_tmp_save_every_lines = resumed_interval

    checkpoint_enabled = checkpoint_path is not None and (
        effective_tmp_save_every_lines is not None or resume_state is not None or tmp_progress_path is not None
    )

    print("\nInitializing Meta Oracle...")
    print(f"  Provider: {provider}")
    print(f"  Model: {model or 'default'}")
    print(f"  Stay workers: {stay_workers}")
    print(f"  Use ICU discharge summary in context: {selected_use_discharge_summary}")
    print(f"  Compress pre-ICU history once per patient: {selected_compress_pre_icu_history}")
    print(f"  Include ICU outcome in prompt: {selected_include_icu_outcome_in_prompt}")
    print("  Context history/future source: window payload (create_time_windows JSONL output)")
    print(f"  Top-k recommendations requested: {config.oracle_context_top_k_recommendations}")
    if effective_tmp_save_every_lines is not None:
        print(f"  Tmp save every JSONL lines: {effective_tmp_save_every_lines}")
    if checkpoint_enabled and checkpoint_path is not None:
        print(f"  Tmp progress path: {checkpoint_path}")

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

    def _evaluate_one_stay(patient_id: str) -> Dict[str, Any]:
        patient_payload = grouped[patient_id]
        subject_id = patient_payload["subject_id"]
        icu_stay_id = patient_payload["icu_stay_id"]
        windows = patient_payload["windows"]
        patient_trajectory = patient_payload["patient_trajectory"]

        reports: List[OracleReport] = []
        compression_used = False
        compression_error: Optional[str] = None

        try:
            if selected_compress_pre_icu_history:
                try:
                    compression_info = oracle.compress_pre_icu_history_for_windows(windows)
                    compression_used = compression_info is not None
                except Exception as compression_exception:
                    compression_error = str(compression_exception)

            for window_idx, window in enumerate(windows):
                try:
                    report = oracle.evaluate_window(window)
                except Exception as e:
                    report = _build_error_report(window, error_message=f"Oracle evaluation error: {e}")
                reports.append(report)
        finally:
            llm_calls = oracle.pop_patient_llm_call_logs(subject_id=subject_id, icu_stay_id=icu_stay_id)
            oracle.pop_patient_trajectory_logs(subject_id=subject_id, icu_stay_id=icu_stay_id)

        return {
            "patient_id": patient_id,
            "reports": reports,
            "llm_calls": llm_calls,
            "compression_used": compression_used,
            "compression_error": compression_error,
        }

    baseline_oracle_totals = {key: _safe_int(summary_stats.get(key), default=0) for key in ORACLE_COUNTER_KEYS}
    total_windows = sum(len(grouped[patient_id]["windows"]) for patient_id in patient_order)
    completed_windows = sum(len(grouped[patient_id]["windows"]) for patient_id in completed_patient_ids)
    summary_file = run_dir / "processing_summary.json"

    def _build_summary_snapshot() -> Dict[str, Any]:
        snapshot = dict(summary_stats)
        oracle_stats = oracle.get_statistics()
        for key in ORACLE_COUNTER_KEYS:
            snapshot[key] = baseline_oracle_totals[key] + _safe_int(oracle_stats.get(key), default=0)
        total_evaluations = _safe_int(snapshot.get("total_evaluations"), default=0)
        total_tokens = _safe_int(snapshot.get("total_tokens_used"), default=0)
        snapshot["avg_tokens_per_evaluation"] = (
            float(total_tokens) / float(total_evaluations) if total_evaluations > 0 else 0.0
        )
        for key in (
            "use_discharge_summary",
            "include_icu_outcome_in_prompt",
            "history_context_hours",
            "future_context_hours",
            "top_k_recommendations",
            "compress_pre_icu_history",
        ):
            snapshot[key] = oracle_stats.get(key, snapshot.get(key))
        return snapshot

    def _persist_progress(is_completed: bool) -> Dict[str, Any]:
        snapshot = _build_summary_snapshot()
        _json_dump_atomic(snapshot, summary_file)
        if checkpoint_enabled and checkpoint_path is not None:
            payload = {
                "run_id": run_id,
                "run_directory": str(run_dir),
                "updated_at": datetime.now().isoformat(),
                "is_completed": bool(is_completed),
                "input_jsonl": resolved_input_jsonl,
                "full_windows_jsonl": resolved_full_windows_jsonl,
                "provider": provider,
                "model": model,
                "use_discharge_summary": selected_use_discharge_summary,
                "compress_pre_icu_history": selected_compress_pre_icu_history,
                "include_icu_outcome_in_prompt": selected_include_icu_outcome_in_prompt,
                "max_windows": max_windows,
                "tmp_save_every_lines": effective_tmp_save_every_lines,
                "total_patients": len(grouped),
                "total_windows_input": len(raw_records),
                "completed_patient_ids": list(completed_patient_ids),
                "summary_stats": snapshot,
            }
            _json_dump_atomic(payload, checkpoint_path)
        return snapshot

    print(f"\n{'=' * 80}")
    print("EVALUATING + SAVING RESULTS")
    print("=" * 80)

    if checkpoint_enabled:
        summary_stats = _persist_progress(is_completed=False)

    if completed_patient_ids:
        print(f"Resume state: {len(completed_patient_ids)} stays already completed and will be skipped.")
    remaining_patient_ids = [patient_id for patient_id in patient_order if patient_id not in completed_patient_id_set]
    lines_since_checkpoint = 0

    if not remaining_patient_ids:
        print("No remaining stays to evaluate.")
    else:
        worker_count = min(stay_workers, len(remaining_patient_ids))
        print(f"Evaluating in parallel ({worker_count} workers, remaining stays={len(remaining_patient_ids)})...")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_patient = {
                executor.submit(_evaluate_one_stay, patient_id): patient_id for patient_id in remaining_patient_ids
            }
            for future in as_completed(future_to_patient):
                patient_id = future_to_patient[future]
                patient_payload = grouped[patient_id]
                subject_id = patient_payload["subject_id"]
                icu_stay_id = patient_payload["icu_stay_id"]
                windows = patient_payload["windows"]
                patient_trajectory = patient_payload["patient_trajectory"]

                try:
                    patient_eval = future.result()
                except Exception as e:
                    patient_eval = {
                        "patient_id": patient_id,
                        "reports": [
                            _build_error_report(window, error_message=f"Stay-level evaluation error: {e}")
                            for window in windows
                        ],
                        "llm_calls": [],
                        "compression_used": False,
                        "compression_error": str(e),
                    }

                print(f"\nSaving Patient {patient_id}")
                try:
                    reports_raw = patient_eval.get("reports") if isinstance(patient_eval, dict) else None
                    reports = reports_raw if isinstance(reports_raw, list) else []
                    llm_calls_raw = patient_eval.get("llm_calls") if isinstance(patient_eval, dict) else None
                    llm_calls = llm_calls_raw if isinstance(llm_calls_raw, list) else []

                    if len(reports) < len(windows):
                        for window in windows[len(reports) :]:
                            reports.append(
                                _build_error_report(
                                    window,
                                    error_message="Missing report for window after stay-level execution.",
                                )
                            )

                    windows_failed_for_patient = 0
                    patient_status_counts: Dict[str, int] = {}
                    overall_status_delta: Dict[str, int] = {}
                    action_evaluations_count = 0
                    for report in reports:
                        if report.error:
                            windows_failed_for_patient += 1
                        report_dict = report.to_dict()
                        status = _safe_status(report_dict)
                        patient_status_counts[status] = patient_status_counts.get(status, 0) + 1
                        overall_status_delta[status] = overall_status_delta.get(status, 0) + 1
                        action_review = report_dict.get("action_review")
                        evaluations = action_review.get("evaluations") if isinstance(action_review, dict) else []
                        if isinstance(evaluations, list):
                            action_evaluations_count += len(evaluations)

                    patient_dir = patients_dir / patient_id
                    patient_dir.mkdir(parents=True, exist_ok=True)

                    patient_predictions = _build_patient_predictions_payload(
                        run_id=run_id,
                        trajectory=patient_trajectory,
                        windows=windows,
                        reports=reports,
                        llm_calls=llm_calls,
                    )
                    with open(patient_dir / "oracle_predictions.json", "w", encoding="utf-8") as f:
                        json.dump(patient_predictions, f, indent=2, ensure_ascii=False, default=_json_default)

                    llm_payload = _build_oracle_llm_calls_payload(
                        subject_id=subject_id,
                        icu_stay_id=icu_stay_id,
                        provider=getattr(oracle.llm_client, "provider", None),
                        model=getattr(oracle.llm_client, "model", None),
                        include_icu_outcome_in_prompt=getattr(oracle, "include_icu_outcome_in_prompt", None),
                        calls=llm_calls,
                    )
                    with open(patient_dir / "llm_calls.json", "w", encoding="utf-8") as f:
                        json.dump(llm_payload, f, indent=2, ensure_ascii=False, default=_json_default)
                    save_llm_calls_html(llm_payload, patient_dir / "llm_calls.html")

                    window_contexts_payload = _build_window_contexts_payload(
                        run_id=run_id,
                        trajectory=patient_trajectory,
                        windows=windows,
                        llm_calls=llm_calls,
                        history_hours=float(config.oracle_context_history_hours),
                        future_hours=float(config.oracle_context_future_hours),
                    )
                    with open(patient_dir / "window_contexts.json", "w", encoding="utf-8") as f:
                        json.dump(window_contexts_payload, f, indent=2, ensure_ascii=False, default=_json_default)

                    has_full_window_context = False
                    full_windows = full_windows_by_patient.get(patient_id, [])
                    if full_windows:
                        full_window_trajectory = _build_patient_level_trajectory(full_windows)
                        full_window_contexts_payload = _build_window_contexts_payload(
                            run_id=run_id,
                            trajectory=full_window_trajectory,
                            windows=full_windows,
                            llm_calls=[],
                            history_hours=float(config.oracle_context_history_hours),
                            future_hours=float(config.oracle_context_future_hours),
                        )
                        with open(patient_dir / "full_window_contexts.json", "w", encoding="utf-8") as f:
                            json.dump(
                                full_window_contexts_payload,
                                f,
                                indent=2,
                                ensure_ascii=False,
                                default=_json_default,
                            )
                        has_full_window_context = True

                    for status, count in overall_status_delta.items():
                        summary_stats["overall_status_distribution"][status] = (
                            summary_stats["overall_status_distribution"].get(status, 0) + count
                        )
                    summary_stats["windows_failed"] += windows_failed_for_patient
                    summary_stats["total_action_evaluations"] += action_evaluations_count
                    summary_stats["total_windows_evaluated"] += len(reports)
                    summary_stats["patients_processed"] += 1
                    if has_full_window_context:
                        summary_stats["patients_with_full_window_contexts"] += 1

                    completed_patient_id_set.add(patient_id)
                    completed_patient_ids.append(patient_id)
                    completed_windows += len(reports)
                    lines_since_checkpoint += len(windows)

                    compression_used = (
                        bool(patient_eval.get("compression_used")) if isinstance(patient_eval, dict) else False
                    )
                    compression_error = (
                        str(patient_eval.get("compression_error"))
                        if isinstance(patient_eval, dict) and patient_eval.get("compression_error")
                        else ""
                    )
                    if compression_used:
                        print("  Pre-ICU compression: reused one compressed summary for all windows")
                    elif compression_error:
                        print(f"  Pre-ICU compression error (fallback to raw history): {compression_error}")
                    print(f"  Status distribution: {patient_status_counts}")
                    print(f"  Total action evaluations: {action_evaluations_count}")
                    print(
                        f"  Completed stays: {len(completed_patient_ids)}/{len(patient_order)} | "
                        f"windows: {completed_windows}/{total_windows}"
                    )

                    if (
                        checkpoint_enabled
                        and effective_tmp_save_every_lines is not None
                        and lines_since_checkpoint >= effective_tmp_save_every_lines
                    ):
                        summary_stats = _persist_progress(is_completed=False)
                        lines_since_checkpoint = 0
                        print(
                            f"  Tmp progress saved at {len(completed_patient_ids)}/{len(patient_order)} completed stays."
                        )

                except Exception as e:
                    print(f"  ERROR: Failed to save patient outputs: {e}")
                    summary_stats["patients_failed"] += 1
                    continue

    summary_stats = _persist_progress(is_completed=len(completed_patient_ids) == len(patient_order))

    print("\nSummary:")
    print(
        f"  Patients processed: {summary_stats.get('patients_processed', 0)}/{summary_stats.get('total_patients', 0)}"
    )
    print(f"  Patients resumed: {summary_stats.get('patients_resumed', 0)}")
    print(f"  Patients failed: {summary_stats.get('patients_failed', 0)}")
    print(f"  Total windows evaluated: {summary_stats.get('total_windows_evaluated', 0)}")
    print(f"  Failed windows: {summary_stats.get('windows_failed', 0)}")
    print(f"  Patients with full-window contexts: {summary_stats.get('patients_with_full_window_contexts', 0)}")
    print(f"  Total action evaluations: {summary_stats.get('total_action_evaluations', 0)}")
    print(f"  Status distribution: {summary_stats.get('overall_status_distribution', {})}")
    print(f"  Total tokens used: {_safe_int(summary_stats.get('total_tokens_used'), default=0):,}")
    print("  Avg tokens per evaluation: " f"{float(summary_stats.get('avg_tokens_per_evaluation') or 0.0):.0f}")

    print(f"\nOutputs saved to: {run_dir}")
    print(f"  - Summary: {summary_file.name}")
    print("  - Per-patient predictions: patients/<subject_id>_<icu_stay_id>/oracle_predictions.json")
    print("  - Per-patient window context: patients/<subject_id>_<icu_stay_id>/window_contexts.json")
    if full_windows_jsonl:
        print("  - Per-patient full timeline context: patients/<subject_id>_<icu_stay_id>/full_window_contexts.json")
    if checkpoint_enabled and checkpoint_path is not None:
        print(f"  - Tmp progress: {checkpoint_path}")

    print(f"\n{'=' * 80}")
    print("PROCESSING COMPLETE")
    print("=" * 80)


def main() -> None:
    config = load_config()

    parser = argparse.ArgumentParser(description="Run Meta Oracle on selected window JSONL (one window per line).")
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Path to selected window JSONL file.",
    )
    parser.add_argument(
        "--full-windows-jsonl",
        type=str,
        default=None,
        help=(
            "Optional full-window JSONL path (all windows for the same selected patients). "
            "If provided, exports per-patient full_window_contexts.json for annotation plotting."
        ),
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
        "--stay-workers",
        "--line-workers",
        "--window-workers",
        dest="stay_workers",
        type=int,
        default=1,
        help="Number of concurrent ICU-stay evaluations (default: 4, set 1 to disable).",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap for number of JSONL windows to process.",
    )
    parser.add_argument(
        "--tmp-save-every-lines",
        type=int,
        default=None,
        help=(
            "If set, saves tmp progress every N evaluated JSONL lines (windows). "
            "Use with --resume-from-tmp to continue after interruption."
        ),
    )
    parser.add_argument(
        "--tmp-progress-path",
        type=str,
        default=None,
        help=(f"Optional path for tmp progress JSON. Defaults to <run_dir>/{TMP_PROGRESS_FILENAME}."),
    )
    parser.add_argument(
        "--resume-from-tmp",
        type=str,
        default=None,
        help=(
            "Resume from an existing tmp progress JSON file or from a run directory containing "
            f"{TMP_PROGRESS_FILENAME}."
        ),
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
        "--config",
        type=str,
        default=None,
        help="Path to custom config.json file.",
    )

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        print(f"Loaded custom config from: {args.config}")

    process_jsonl_for_oracle(
        config=config,
        input_jsonl=args.input_jsonl,
        full_windows_jsonl=args.full_windows_jsonl,
        output_dir=args.output,
        provider=args.provider,
        model=args.model,
        use_discharge_summary=args.use_discharge_summary,
        compress_pre_icu_history=args.compress_pre_icu_history,
        include_icu_outcome_in_prompt=args.include_icu_outcome_in_prompt,
        stay_workers=args.stay_workers,
        max_windows=args.max_windows,
        tmp_save_every_lines=args.tmp_save_every_lines,
        tmp_progress_path=args.tmp_progress_path,
        resume_from_tmp=args.resume_from_tmp,
    )


if __name__ == "__main__":
    main()
