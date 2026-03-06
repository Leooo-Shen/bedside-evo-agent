"""Batch processing pipeline for Oracle offline evaluator."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from agents.oracle import MetaOracle, save_oracle_reports
from config.config import Config, load_config
from data_parser import MIMICDataParser
from utils.llm_log_viewer import save_llm_calls_html


def _safe_status(report_dict: Dict[str, Any]) -> str:
    status = report_dict.get("patient_status", {}).get("overall_status")
    if isinstance(status, str) and status.strip():
        return status.strip().lower()
    return "unknown"


def _iter_trajectories_stream(parser: Any, max_patients: Optional[int]) -> Iterator[Dict[str, Any]]:
    """Iterate trajectories lazily when parser supports streaming."""
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


def _planned_total_patients(parser: Any, max_patients: Optional[int]) -> Optional[int]:
    """Best-effort estimate of total trajectories without materializing all trajectories."""
    icu_stay_df = getattr(parser, "icu_stay_df", None)
    if icu_stay_df is None:
        return None

    total = len(icu_stay_df)
    if max_patients is not None:
        return min(total, max_patients)
    return total


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _build_patient_predictions_payload(
    *,
    run_id: str,
    trajectory: Dict[str, Any],
    windows: List[Dict[str, Any]],
    reports: List[Any],
    llm_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    parsed_outputs_by_window_index: Dict[int, Dict[str, Any]] = {}
    for call in llm_calls:
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
        window_outputs.append(
            {
                "window_index": idx,
                "window_metadata": {
                    "subject_id": window.get("subject_id"),
                    "icu_stay_id": window.get("icu_stay_id"),
                    "window_start_time": window.get("current_window_start"),
                    "window_end_time": window.get("current_window_end"),
                    "hours_since_admission": window.get("hours_since_admission"),
                    "current_window_hours": window.get("current_window_hours"),
                    "num_history_events": window.get("num_history_events"),
                    "num_current_events": window.get("num_current_events"),
                    "pre_icu_history_source": window.get("pre_icu_history_source"),
                    "pre_icu_history_items": window.get("pre_icu_history_items"),
                },
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
        "trajectory_metadata": {
            "enter_time": trajectory.get("enter_time"),
            "leave_time": trajectory.get("leave_time"),
            "icu_duration_hours": trajectory.get("icu_duration_hours"),
            "survived": trajectory.get("survived"),
            "death_time": trajectory.get("death_time"),
            "total_events": len(trajectory.get("events", [])),
        },
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
    calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    step_types = {str(call.get("step_type")) for call in calls}
    pipeline_agents = [
        {"name": "oracle_evaluator", "used": "oracle_evaluator" in step_types, "thinking": None},
    ]
    return {
        "patient_id": f"{subject_id}_{icu_stay_id}",
        "agent_type": "oracle",
        "llm_provider": provider,
        "llm_model": model,
        "pipeline_agents": pipeline_agents,
        "total_calls": len(calls),
        "calls": calls,
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
    max_patients: Optional[int] = None,
    save_trajectories: bool = True,
    window_workers: int = 4,
) -> None:
    """Process a batch of patient trajectories through Oracle."""
    if max_patients is not None and max_patients < 0:
        raise ValueError("max_patients must be >= 0")
    if window_workers < 1:
        raise ValueError("window_workers must be >= 1")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"oracle_{run_timestamp}"
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
        config.oracle_context_use_discharge_summary
        if use_discharge_summary is None
        else bool(use_discharge_summary)
    )
    print(f"  Use ICU discharge summary in context: {selected_use_discharge_summary}")
    print(f"  Context history threshold (hours): {config.oracle_context_history_hours}")
    print(f"  Context future threshold (hours): {config.oracle_context_future_hours}")
    print(f"  Top-k recommendations requested: {config.oracle_context_top_k_recommendations}")

    oracle = MetaOracle(
        provider=provider,
        model=model,
        temperature=0.3,
        use_discharge_summary=selected_use_discharge_summary,
        history_context_hours=config.oracle_context_history_hours,
        future_context_hours=config.oracle_context_future_hours,
        top_k_recommendations=config.oracle_context_top_k_recommendations,
        log_dir=config.oracle_log_dir,
    )

    print("\nPreparing trajectory stream...")
    planned_total = _planned_total_patients(parser, max_patients)
    if max_patients is not None:
        print(f"  Max patients: {max_patients}")
    if planned_total is not None:
        print(f"  Total trajectories to process: {planned_total}")
    else:
        print("  Total trajectories to process: unknown (streamed)")

    json_default = getattr(parser, "_json_default", _json_default)
    trajectory_writer = None
    trajectories_file = run_dir / "patient_trajectories.jsonl"
    if save_trajectories:
        trajectory_writer = open(trajectories_file, "w", encoding="utf-8")
        print(f"  Streaming trajectories to: {trajectories_file}")

    print(f"\n{'=' * 80}")
    print("PROCESSING TRAJECTORIES")
    print("=" * 80)

    all_reports = []
    summary_stats: Dict[str, Any] = {
        "total_patients": planned_total if planned_total is not None else 0,
        "total_windows_evaluated": 0,
        "patients_processed": 0,
        "patients_failed": 0,
        "overall_status_distribution": {},
        "total_doctor_actions": 0,
    }
    total_seen = 0

    try:
        for i, trajectory in enumerate(_iter_trajectories_stream(parser, max_patients), start=1):
            total_seen = i

            if trajectory_writer is not None:
                trajectory_writer.write(
                    json.dumps(trajectory, ensure_ascii=False, default=json_default) + "\n"
                )

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
                    calls=llm_calls,
                )
                llm_calls_json = patient_dir / "llm_calls.json"
                with open(llm_calls_json, "w", encoding="utf-8") as f:
                    json.dump(llm_payload, f, indent=2, ensure_ascii=False, default=json_default)
                save_llm_calls_html(llm_payload, patient_dir / "llm_calls.html")

                all_reports.extend(reports)
                summary_stats["total_windows_evaluated"] += len(reports)
                summary_stats["patients_processed"] += 1

            except Exception as e:
                print(f"  ERROR: Failed to process patient: {e}")
                summary_stats["patients_failed"] += 1
                oracle.pop_patient_trajectory_logs(subject_id=subject_id, icu_stay_id=icu_stay_id)
                oracle.pop_patient_llm_call_logs(subject_id=subject_id, icu_stay_id=icu_stay_id)
                continue
    finally:
        if trajectory_writer is not None:
            trajectory_writer.close()

    if planned_total is None:
        summary_stats["total_patients"] = total_seen

    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print("=" * 80)

    combined_output = run_dir / "all_oracle_reports.json"
    save_oracle_reports(all_reports, str(combined_output), include_window_data=True)

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
    print(f"  - Combined report: {combined_output.name}")
    print(f"  - Summary: {summary_file.name}")
    print("  - Patients: patients/<subject_id>_<icu_stay_id>/")

    print(f"\n{'=' * 80}")
    print("PROCESSING COMPLETE")
    print("=" * 80)


def main() -> None:
    """Main entry point for Oracle batch processing."""
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Process MIMIC-demo data through Meta Oracle for offline evaluation"
    )

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
        help=(
            f"Size of current observation window in hours "
            f"(default: {config.oracle_current_window_hours})"
        ),
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
        "--max-patients",
        type=int,
        default=config.max_patients,
        help="Maximum number of patients to process (for testing)",
    )

    parser.add_argument(
        "--no-save-trajectories",
        action="store_true",
        help="Don't save intermediate trajectory data",
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
        max_patients=args.max_patients,
        save_trajectories=not args.no_save_trajectories,
        window_workers=args.window_workers,
    )


if __name__ == "__main__":
    main()
