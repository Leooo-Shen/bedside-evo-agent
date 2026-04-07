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
        for event in [*window.get("history_events", []), *window.get("current_events", []), *window.get("future_events", [])]:
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
) -> None:
    if stay_workers < 1:
        raise ValueError("--stay-workers must be >= 1")

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

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"oracle_{run_timestamp}_{prompt_mode_suffix}_jsonl"
    run_dir = output_root / run_id
    patients_dir = run_dir / "patients"
    patients_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MIMIC-DEMO ORACLE JSONL PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Input JSONL: {input_jsonl}")
    if full_windows_jsonl:
        print(f"Full-window JSONL: {full_windows_jsonl}")
    print(f"Output Run Directory: {run_dir}")

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

        matched_patients = sum(1 for patient_id in patient_order if len(full_windows_by_patient.get(patient_id, [])) > 0)
        print(f"  Loaded full windows for selected patients: {matched_patients}/{len(patient_order)}")

    print(f"  Loaded windows: {len(raw_records)}")
    print(f"  Patients in JSONL: {len(grouped)}")

    print("\nInitializing Meta Oracle...")
    print(f"  Provider: {provider}")
    print(f"  Model: {model or 'default'}")
    print(f"  Stay workers: {stay_workers}")
    print(f"  Use ICU discharge summary in context: {selected_use_discharge_summary}")
    print(f"  Compress pre-ICU history once per patient: {selected_compress_pre_icu_history}")
    print(f"  Include ICU outcome in prompt: {selected_include_icu_outcome_in_prompt}")
    print("  Context history/future source: window payload (create_time_windows JSONL output)")
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

    worker_count = min(stay_workers, len(patient_order))
    print(f"\nEvaluating by ICU stay in parallel ({worker_count} workers, total stays={len(patient_order)})...")
    evaluated_by_patient: Dict[str, Dict[str, Any]] = {}
    total_windows = sum(len(grouped[patient_id]["windows"]) for patient_id in patient_order)
    completed_stays = 0
    completed_windows = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_patient = {
            executor.submit(_evaluate_one_stay, patient_id): patient_id for patient_id in patient_order
        }
        for future in as_completed(future_to_patient):
            patient_id = future_to_patient[future]
            completed_stays += 1
            try:
                result = future.result()
            except Exception as e:
                windows = grouped[patient_id]["windows"]
                result = {
                    "patient_id": patient_id,
                    "reports": [
                        _build_error_report(window, error_message=f"Stay-level evaluation error: {e}")
                        for window in windows
                    ],
                    "llm_calls": [],
                    "compression_used": False,
                    "compression_error": str(e),
                }
            evaluated_by_patient[patient_id] = result
            completed_windows += len(result.get("reports", []))
            print(
                f"  Completed stays: {completed_stays}/{len(patient_order)} | "
                f"windows: {completed_windows}/{total_windows}"
            )

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
        "patients_failed": 0,
        "patients_with_full_window_contexts": 0,
        "overall_status_distribution": {},
        "total_action_evaluations": 0,
    }

    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print("=" * 80)

    for index, patient_id in enumerate(patient_order, start=1):
        patient_payload = grouped[patient_id]
        subject_id = patient_payload["subject_id"]
        icu_stay_id = patient_payload["icu_stay_id"]
        windows = patient_payload["windows"]
        patient_trajectory = patient_payload["patient_trajectory"]

        print(f"\n[{index}/{len(patient_order)}] Saving Patient {patient_id}")

        try:
            patient_eval = evaluated_by_patient.get(patient_id, {})
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

            for report in reports:
                if report.error:
                    summary_stats["windows_failed"] += 1

            patient_status_counts: Dict[str, int] = {}
            action_evaluations_count = 0
            for report in reports:
                report_dict = report.to_dict()
                status = _safe_status(report_dict)
                patient_status_counts[status] = patient_status_counts.get(status, 0) + 1
                action_review = report_dict.get("action_review")
                evaluations = action_review.get("evaluations") if isinstance(action_review, dict) else []
                if isinstance(evaluations, list):
                    action_evaluations_count += len(evaluations)
                summary_stats["overall_status_distribution"][status] = (
                    summary_stats["overall_status_distribution"].get(status, 0) + 1
                )

            summary_stats["total_action_evaluations"] += action_evaluations_count
            summary_stats["total_windows_evaluated"] += len(reports)

            patient_dir = patients_dir / patient_id
            patient_dir.mkdir(parents=True, exist_ok=True)

            patient_predictions = _build_patient_predictions_payload(
                run_id=run_id,
                trajectory=patient_trajectory,
                windows=windows,
                reports=reports,
                llm_calls=llm_calls,
            )
            patient_predictions_path = patient_dir / "oracle_predictions.json"
            with open(patient_predictions_path, "w", encoding="utf-8") as f:
                json.dump(patient_predictions, f, indent=2, ensure_ascii=False, default=_json_default)

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
            window_contexts_path = patient_dir / "window_contexts.json"
            with open(window_contexts_path, "w", encoding="utf-8") as f:
                json.dump(window_contexts_payload, f, indent=2, ensure_ascii=False, default=_json_default)

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
                full_window_contexts_path = patient_dir / "full_window_contexts.json"
                with open(full_window_contexts_path, "w", encoding="utf-8") as f:
                    json.dump(full_window_contexts_payload, f, indent=2, ensure_ascii=False, default=_json_default)
                summary_stats["patients_with_full_window_contexts"] += 1

            summary_stats["patients_processed"] += 1
            compression_used = bool(patient_eval.get("compression_used")) if isinstance(patient_eval, dict) else False
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

        except Exception as e:
            print(f"  ERROR: Failed to save patient outputs: {e}")
            summary_stats["patients_failed"] += 1
            continue

    summary_stats.update(oracle.get_statistics())
    summary_file = run_dir / "processing_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)

    print("\nSummary:")
    print(f"  Patients processed: {summary_stats['patients_processed']}/{summary_stats['total_patients']}")
    print(f"  Patients failed: {summary_stats['patients_failed']}")
    print(f"  Total windows evaluated: {summary_stats['total_windows_evaluated']}")
    print(f"  Failed windows: {summary_stats['windows_failed']}")
    print(f"  Patients with full-window contexts: {summary_stats['patients_with_full_window_contexts']}")
    print(f"  Total action evaluations: {summary_stats['total_action_evaluations']}")
    print(f"  Status distribution: {summary_stats['overall_status_distribution']}")
    print(f"  Total tokens used: {summary_stats['total_tokens_used']:,}")
    print(f"  Avg tokens per evaluation: {summary_stats['avg_tokens_per_evaluation']:.0f}")

    print(f"\nOutputs saved to: {run_dir}")
    print(f"  - Summary: {summary_file.name}")
    print("  - Per-patient predictions: patients/<subject_id>_<icu_stay_id>/oracle_predictions.json")
    print("  - Per-patient window context: patients/<subject_id>_<icu_stay_id>/window_contexts.json")
    if full_windows_jsonl:
        print("  - Per-patient full timeline context: patients/<subject_id>_<icu_stay_id>/full_window_contexts.json")

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
        default=4,
        help="Number of concurrent ICU-stay evaluations (default: 4, set 1 to disable).",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap for number of JSONL windows to process.",
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
    )


if __name__ == "__main__":
    main()
