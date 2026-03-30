#!/usr/bin/env python3
"""Generate and filter Oracle-ready windows from sampled ICU subset patients."""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import Config, load_config
from data_parser import MIMICDataParser
from utils.key_window_selector import HIGH_IMPACT_EVENT_CODES

ORACLE_REQUIRED_WINDOW_FIELDS: Sequence[str] = (
    "subject_id",
    "icu_stay_id",
    "current_window_start",
    "current_window_end",
    "hours_since_admission",
    "current_window_hours",
    "patient_metadata",
    "history_events",
    "current_events",
    "num_history_events",
    "num_current_events",
    "pre_icu_history",
    "pre_icu_history_source",
    "pre_icu_history_items",
    "current_discharge_summary",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate windows from sampled ICU stays, filter by action-event ratio, "
            "and save Oracle-ready window JSONL."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.json",
        help="Path to config json (default: config/config.json).",
    )
    parser.add_argument(
        "--subset-dir",
        type=Path,
        default=Path("data/mimic-demo/anno_subset_100"),
        help="Directory containing sampled subset parquet files.",
    )
    parser.add_argument(
        "--events-path",
        type=Path,
        default=None,
        help="Optional explicit events parquet path. Overrides --subset-dir/events.parquet.",
    )
    parser.add_argument(
        "--icu-stay-path",
        type=Path,
        default=None,
        help="Optional explicit ICU stay parquet path. Overrides --subset-dir/icu_stay.parquet.",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.5,
        help="Action-event ratio threshold (inclusive, default: 0.5).",
    )
    parser.add_argument(
        "--max-days-after-leave",
        type=float,
        default=7.0,
        help="Discharge-summary selector window used by parser (default: 7.0 days).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Output JSONL path (default: <subset-dir>/selected_windows_action_ratio_0p5.jsonl).",
    )
    parser.add_argument(
        "--patient-stats-csv",
        type=Path,
        default=None,
        help="Per-patient selection stats CSV path (default: <subset-dir>/patient_window_selection_stats.csv).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Summary JSON path (default: <subset-dir>/window_selection_summary.json).",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Optional cap for number of ICU stays processed.",
    )
    parser.add_argument(
        "--silence-parser-window-logs",
        action="store_true",
        help="Suppress verbose parser.create_time_windows prints.",
    )
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _json_default(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _build_action_code_set(codes: Iterable[str]) -> set[str]:
    return {str(code).strip().upper() for code in codes if str(code).strip()}


def _normalize_event_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _event_code(event: Mapping[str, Any]) -> str:
    return str(event.get("code") or "").strip().upper()


def _compute_action_ratio(
    current_events: Sequence[Mapping[str, Any]],
    action_code_set: set[str],
) -> Dict[str, Any]:
    total = int(len(current_events))
    action_events = [dict(event) for event in current_events if _event_code(event) in action_code_set]
    action_count = int(len(action_events))
    ratio = (float(action_count) / float(total)) if total > 0 else 0.0
    code_counts: Dict[str, int] = {}
    for event in action_events:
        code = _event_code(event)
        code_counts[code] = code_counts.get(code, 0) + 1
    return {
        "total_event_count": total,
        "action_event_count": action_count,
        "action_event_ratio": ratio,
        "action_event_code_counts": code_counts,
        "action_events": action_events,
    }


def _window_required_field_presence(window: Mapping[str, Any]) -> Dict[str, bool]:
    return {field: field in window for field in ORACLE_REQUIRED_WINDOW_FIELDS}


def _trajectory_metadata_for_export(trajectory: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "subject_id",
        "icu_stay_id",
        "enter_time",
        "leave_time",
        "age_at_admission",
        "gender",
        "icu_duration_hours",
        "survived",
        "death_time",
        "readmission",
        "readm_duration_hours",
    )
    payload: Dict[str, Any] = {}
    for key in keys:
        payload[key] = trajectory.get(key)
    return payload


def _resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    events_path = args.events_path or (args.subset_dir / "events.parquet")
    icu_stay_path = args.icu_stay_path or (args.subset_dir / "icu_stay.parquet")

    if not Path(events_path).exists():
        raise FileNotFoundError(f"Events parquet not found: {events_path}")
    if not Path(icu_stay_path).exists():
        raise FileNotFoundError(f"ICU stay parquet not found: {icu_stay_path}")

    default_output_jsonl = args.subset_dir / f"selected_windows_action_ratio_{str(args.ratio_threshold).replace('.', 'p')}.jsonl"
    output_jsonl = args.output_jsonl or default_output_jsonl
    patient_stats_csv = args.patient_stats_csv or (args.subset_dir / "patient_window_selection_stats.csv")
    summary_json = args.summary_json or (args.subset_dir / "window_selection_summary.json")

    return {
        "events_path": Path(events_path),
        "icu_stay_path": Path(icu_stay_path),
        "output_jsonl": Path(output_jsonl),
        "patient_stats_csv": Path(patient_stats_csv),
        "summary_json": Path(summary_json),
    }


def _create_windows_for_trajectory(
    parser: MIMICDataParser,
    trajectory: Dict[str, Any],
    config: Config,
    *,
    silence_parser_logs: bool = False,
) -> List[Dict[str, Any]]:
    kwargs = {
        "trajectory": trajectory,
        "current_window_hours": float(config.oracle_current_window_hours),
        "window_step_hours": float(config.oracle_window_step_hours),
        "include_pre_icu_data": bool(config.oracle_include_pre_icu_data),
        "use_first_n_hours_after_icu": config.oracle_observation_hours,
        "use_discharge_summary_for_history": bool(config.oracle_use_discharge_summary_for_history),
        "num_discharge_summaries": int(config.oracle_num_discharge_summaries),
        "relative_report_codes": list(config.oracle_relative_report_codes),
        "pre_icu_history_hours": float(config.oracle_pre_icu_history_hours),
    }
    if not silence_parser_logs:
        return parser.create_time_windows(**kwargs)

    with redirect_stdout(io.StringIO()):
        return parser.create_time_windows(**kwargs)


def main() -> None:
    args = _parse_args()
    paths = _resolve_paths(args)
    config = load_config(args.config)

    output_parent = paths["output_jsonl"].parent
    output_parent.mkdir(parents=True, exist_ok=True)
    paths["patient_stats_csv"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_json"].parent.mkdir(parents=True, exist_ok=True)

    ratio_threshold = float(args.ratio_threshold)
    if ratio_threshold < 0:
        raise ValueError(f"--ratio-threshold must be >= 0, got {ratio_threshold}")

    parser = MIMICDataParser(
        events_path=str(paths["events_path"]),
        icu_stay_path=str(paths["icu_stay_path"]),
        discharge_summary_max_days_after_leave=float(args.max_days_after_leave),
        require_discharge_summary_for_icu_stays=True,
    )
    parser.load_data()
    if parser.icu_stay_df is None:
        raise RuntimeError("Failed to load ICU stays from subset.")

    stays_df = parser.icu_stay_df.copy()
    if args.max_patients is not None:
        if int(args.max_patients) < 0:
            raise ValueError("--max-patients must be >= 0")
        stays_df = stays_df.head(int(args.max_patients)).copy()

    action_code_set = _build_action_code_set(HIGH_IMPACT_EVENT_CODES)
    selected_records: List[Dict[str, Any]] = []
    patient_stats_rows: List[Dict[str, Any]] = []
    total_windows_generated = 0
    total_windows_selected = 0
    skipped_no_windows = 0

    print(
        "Window generation config: "
        f"current={config.oracle_current_window_hours}h, "
        f"step={config.oracle_window_step_hours}h, "
        f"include_pre_icu_data={config.oracle_include_pre_icu_data}, "
        f"use_discharge_summary_for_history={config.oracle_use_discharge_summary_for_history}, "
        f"observation_hours={config.oracle_observation_hours}"
    )
    print(
        "Window filtering config: "
        f"ratio_threshold>={ratio_threshold:.4f}, "
        f"action_codes={sorted(action_code_set)}"
    )

    for patient_idx, (_, stay) in enumerate(stays_df.iterrows(), start=1):
        subject_id = int(stay["subject_id"])
        icu_stay_id = int(stay["icu_stay_id"])
        trajectory = parser.get_patient_trajectory(subject_id, icu_stay_id, icu_stay=stay)
        windows = _create_windows_for_trajectory(
            parser,
            trajectory,
            config,
            silence_parser_logs=bool(args.silence_parser_window_logs),
        )

        if not windows:
            skipped_no_windows += 1

        selected_for_patient = 0
        for window_position, window in enumerate(windows, start=1):
            current_events = _normalize_event_list(window.get("current_events"))
            ratio_stats = _compute_action_ratio(current_events=current_events, action_code_set=action_code_set)
            ratio_value = float(ratio_stats["action_event_ratio"])
            if ratio_value < ratio_threshold:
                continue

            selected_for_patient += 1
            total_windows_selected += 1

            record = dict(window)
            record["patient_id"] = f"{subject_id}_{icu_stay_id}"
            record["window_index"] = int(window_position - 1)  # Oracle uses 0-based index internally.
            record["window_position"] = int(window_position)  # Human-friendly 1-based index.
            record["trajectory_metadata"] = _trajectory_metadata_for_export(trajectory)
            record["selection_metadata"] = {
                "selection_rule": "action_event_ratio_threshold",
                "ratio_threshold": ratio_threshold,
                "threshold_comparator": ">=",
                "action_event_codes_reference": list(HIGH_IMPACT_EVENT_CODES),
                "action_event_count": int(ratio_stats["action_event_count"]),
                "total_event_count": int(ratio_stats["total_event_count"]),
                "action_event_ratio": ratio_value,
                "action_event_code_counts": ratio_stats["action_event_code_counts"],
                "selected_at_utc": datetime.utcnow().isoformat() + "Z",
            }
            record["action_events"] = ratio_stats["action_events"]
            record["oracle_required_fields_presence"] = _window_required_field_presence(record)
            record["oracle_prompt_metadata"] = {
                "subject_id": record.get("subject_id"),
                "icu_stay_id": record.get("icu_stay_id"),
                "window_start_time": record.get("current_window_start"),
                "window_end_time": record.get("current_window_end"),
                "hours_since_admission": record.get("hours_since_admission"),
                "current_window_hours": record.get("current_window_hours"),
                "patient_metadata": record.get("patient_metadata"),
                "pre_icu_history_source": record.get("pre_icu_history_source"),
                "pre_icu_history_items": record.get("pre_icu_history_items"),
                "has_current_discharge_summary": isinstance(record.get("current_discharge_summary"), dict),
                "num_history_events": record.get("num_history_events"),
                "num_current_events": record.get("num_current_events"),
            }
            selected_records.append(record)

        generated_for_patient = int(len(windows))
        total_windows_generated += generated_for_patient
        patient_stats_rows.append(
            {
                "patient_index": int(patient_idx),
                "subject_id": subject_id,
                "icu_stay_id": icu_stay_id,
                "survived": bool(trajectory.get("survived")),
                "generated_windows": generated_for_patient,
                "selected_windows": int(selected_for_patient),
                "selected_ratio": (
                    (float(selected_for_patient) / float(generated_for_patient)) if generated_for_patient > 0 else 0.0
                ),
                "window_generation_current_hours": float(config.oracle_current_window_hours),
                "window_generation_step_hours": float(config.oracle_window_step_hours),
                "window_filter_ratio_threshold": ratio_threshold,
            }
        )

    with open(paths["output_jsonl"], "w", encoding="utf-8") as file:
        for record in selected_records:
            file.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    patient_stats_df = pd.DataFrame(patient_stats_rows)
    patient_stats_df.to_csv(paths["patient_stats_csv"], index=False)

    selected_patient_count = int((patient_stats_df["selected_windows"] > 0).sum()) if len(patient_stats_df) > 0 else 0
    generated_patient_count = int(len(patient_stats_df))
    summary_payload = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": str(Path(args.config)),
        "events_path": str(paths["events_path"]),
        "icu_stay_path": str(paths["icu_stay_path"]),
        "output_jsonl": str(paths["output_jsonl"]),
        "patient_stats_csv": str(paths["patient_stats_csv"]),
        "ratio_threshold": ratio_threshold,
        "threshold_comparator": ">=",
        "high_impact_event_codes": list(HIGH_IMPACT_EVENT_CODES),
        "window_generation_config": {
            "current_window_hours": float(config.oracle_current_window_hours),
            "window_step_hours": float(config.oracle_window_step_hours),
            "include_pre_icu_data": bool(config.oracle_include_pre_icu_data),
            "use_first_n_hours_after_icu": config.oracle_observation_hours,
            "use_discharge_summary_for_history": bool(config.oracle_use_discharge_summary_for_history),
            "num_discharge_summaries": int(config.oracle_num_discharge_summaries),
            "relative_report_codes": list(config.oracle_relative_report_codes),
            "pre_icu_history_hours": float(config.oracle_pre_icu_history_hours),
        },
        "stats": {
            "patients_processed": generated_patient_count,
            "patients_with_at_least_one_selected_window": selected_patient_count,
            "patients_skipped_no_windows_generated": int(skipped_no_windows),
            "total_windows_generated": int(total_windows_generated),
            "total_windows_selected": int(total_windows_selected),
            "overall_selected_ratio": (
                (float(total_windows_selected) / float(total_windows_generated)) if total_windows_generated > 0 else 0.0
            ),
        },
    }
    with open(paths["summary_json"], "w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2, ensure_ascii=False, default=_json_default)

    print("Wrote outputs:")
    print(f"  - Selected windows JSONL: {paths['output_jsonl']}")
    print(f"  - Patient stats CSV: {paths['patient_stats_csv']}")
    print(f"  - Summary JSON: {paths['summary_json']}")
    print(
        "Selection stats: "
        f"windows_selected={total_windows_selected} / windows_generated={total_windows_generated}, "
        f"patients_with_selected_windows={selected_patient_count} / {generated_patient_count}"
    )


if __name__ == "__main__":
    main()
