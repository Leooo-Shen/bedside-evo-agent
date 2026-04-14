#!/usr/bin/env python3
"""Generate and filter Oracle-ready windows from sampled ICU subset patients."""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from collections import deque
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "future_events",
    "num_history_events",
    "num_current_events",
    "num_future_events",
    "pre_icu_history",
    "pre_icu_history_source",
    "pre_icu_history_items",
    "current_discharge_summary",
)

PATIENT_STATS_COLUMNS: Sequence[str] = (
    "patient_index",
    "subject_id",
    "icu_stay_id",
    "survived",
    "generated_windows",
    "selected_windows",
    "selected_valid_windows",
    "selected_invalid_windows",
    "selected_valid_ratio",
    "ratio_gap_to_target",
    "meets_min_windows",
    "selected_for_final_cohort",
    "selected_ratio",
    "window_generation_current_hours",
    "window_generation_step_hours",
    "window_filter_ratio_threshold",
)

SELECTED_WINDOW_STATS_COLUMNS: Sequence[str] = (
    "patient_id",
    "subject_id",
    "icu_stay_id",
    "num_current_events",
    "num_history_events",
    "num_total_events",
    "action_event_ratio",
    "window_bucket",
    "has_current_discharge_summary",
    "icu_timeline_fraction",
    "icu_timeline_segment",
)

ICU_TIMELINE_SEGMENT_ORDER: Sequence[str] = (
    "early_1_3",
    "middle_1_3",
    "late_1_3",
    "unassigned",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate windows from all ICU stays in the dataset, apply timeline-first "
            "valid/invalid selection, and save Oracle-ready window JSONL."
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
        default=Path("data/mimic-demo/anno_subset_160"),
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
        "--output-full-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional full-window JSONL path for final selected patients. "
            "If provided, exports all windows of those patients for annotation timeline plotting."
        ),
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
        "--statistics-dir",
        type=Path,
        default=None,
        help="Statistics output directory (default: <subset-dir>/statistics).",
    )
    parser.add_argument(
        "--min-windows-per-patient",
        type=int,
        default=5,
        help="Preferred minimum selected windows per patient (default: 5).",
    )
    parser.add_argument(
        "--max-windows-per-patient",
        type=int,
        default=10,
        help="Maximum selected windows per patient (default: 10).",
    )
    parser.add_argument(
        "--selected-survived-count",
        type=int,
        default=50,
        help="Number of survived patients kept after quality ranking (default: 50).",
    )
    parser.add_argument(
        "--selected-died-count",
        type=int,
        default=50,
        help="Number of died patients kept after quality ranking (default: 50).",
    )
    parser.add_argument(
        "--valid-window-ratio-target",
        type=float,
        default=0.8,
        help=("Target fraction of selected windows that satisfy ratio>=threshold " "(default: 0.8)."),
    )
    parser.add_argument(
        "--late-segment-boost",
        type=int,
        default=2,
        help=(
            "Sampling weight multiplier for late_1_3 during timeline balancing. "
            "1 means no boost; 2 means late gets roughly double chances per round (default: 2)."
        ),
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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


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


def _has_current_discharge_summary(window: Mapping[str, Any]) -> bool:
    value = window.get("current_discharge_summary")
    return isinstance(value, dict) and len(value) > 0


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


def _series_distribution_stats(series: pd.Series) -> Dict[str, float]:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(numeric_series.mean()),
        "median": float(numeric_series.median()),
        "min": float(numeric_series.min()),
        "max": float(numeric_series.max()),
    }


def _icu_timeline_fraction_and_segment(
    *,
    hours_since_admission: Any,
    icu_duration_hours: Any,
) -> Dict[str, Any]:
    hours_value = _safe_float(hours_since_admission, default=float("nan"))
    duration_value = _safe_float(icu_duration_hours, default=float("nan"))
    if (not math.isfinite(hours_value)) or (not math.isfinite(duration_value)) or duration_value <= 0:
        return {
            "icu_timeline_fraction": None,
            "icu_timeline_segment": "unassigned",
        }

    fraction = float(hours_value / duration_value)
    if fraction < (1.0 / 3.0):
        segment = "early_1_3"
    elif fraction < (2.0 / 3.0):
        segment = "middle_1_3"
    else:
        segment = "late_1_3"

    return {
        "icu_timeline_fraction": fraction,
        "icu_timeline_segment": segment,
    }


def _save_empty_plot(output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title)
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_selected_windows_per_patient_histogram(
    patient_selected_window_counts_df: pd.DataFrame,
    output_path: Path,
) -> None:
    title = "Selected Windows Per Patient"
    if patient_selected_window_counts_df.empty:
        _save_empty_plot(output_path, title=title)
        return

    values = pd.to_numeric(patient_selected_window_counts_df["selected_windows"], errors="coerce").dropna()
    if values.empty:
        _save_empty_plot(output_path, title=title)
        return

    bins = int(min(50, max(10, values.max() - values.min() + 1)))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=bins, color="#2a9d8f", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Selected windows per patient")
    ax.set_ylabel("Patient count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_total_events_per_window_histogram(selected_windows_df: pd.DataFrame, output_path: Path) -> None:
    title = "Total Events Per Selected Window"
    if selected_windows_df.empty:
        _save_empty_plot(output_path, title=title)
        return

    values = pd.to_numeric(selected_windows_df["num_total_events"], errors="coerce").dropna()
    if values.empty:
        _save_empty_plot(output_path, title=title)
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=40, color="#e76f51", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Total events in window")
    ax.set_ylabel("Window count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_current_history_events_histogram(selected_windows_df: pd.DataFrame, output_path: Path) -> None:
    title = "Current vs History Events Per Selected Window"
    if selected_windows_df.empty:
        _save_empty_plot(output_path, title=title)
        return

    current_values = pd.to_numeric(selected_windows_df["num_current_events"], errors="coerce").dropna()
    history_values = pd.to_numeric(selected_windows_df["num_history_events"], errors="coerce").dropna()
    if current_values.empty and history_values.empty:
        _save_empty_plot(output_path, title=title)
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    if not current_values.empty:
        ax.hist(current_values, bins=40, alpha=0.65, color="#264653", label="current_events", edgecolor="white")
    if not history_values.empty:
        ax.hist(history_values, bins=40, alpha=0.65, color="#f4a261", label="history_events", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Event count per window")
    ax.set_ylabel("Window count")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_current_discharge_summary_barplot(
    current_discharge_summary_counts_df: pd.DataFrame,
    output_path: Path,
) -> None:
    title = "Selected Windows With Current Discharge Summary"
    if current_discharge_summary_counts_df.empty:
        _save_empty_plot(output_path, title=title)
        return

    labels = current_discharge_summary_counts_df["has_current_discharge_summary"].astype(str).tolist()
    values = pd.to_numeric(current_discharge_summary_counts_df["window_count"], errors="coerce").fillna(0).tolist()

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=["#8ab17d", "#457b9d"][: len(labels)])
    ax.set_title(title)
    ax.set_xlabel("Has current discharge summary")
    ax.set_ylabel("Window count")
    ax.grid(axis="y", alpha=0.2)
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_timeline_segment_barplot(timeline_segment_counts_df: pd.DataFrame, output_path: Path) -> None:
    title = "Selected Windows Across ICU Timeline Thirds"
    if timeline_segment_counts_df.empty:
        _save_empty_plot(output_path, title=title)
        return

    labels = timeline_segment_counts_df["icu_timeline_segment"].astype(str).tolist()
    values = pd.to_numeric(timeline_segment_counts_df["window_count"], errors="coerce").fillna(0).tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#3a86ff", "#ffbe0b", "#fb5607", "#8d99ae"][: len(labels)])
    ax.set_title(title)
    ax.set_xlabel("ICU timeline segment")
    ax.set_ylabel("Window count")
    ax.grid(axis="y", alpha=0.2)
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _resolve_paths(args: argparse.Namespace) -> Dict[str, Any]:
    events_path = args.events_path or (args.subset_dir / "events.parquet")
    icu_stay_path = args.icu_stay_path or (args.subset_dir / "icu_stay.parquet")

    if not Path(events_path).exists():
        raise FileNotFoundError(f"Events parquet not found: {events_path}")
    if not Path(icu_stay_path).exists():
        raise FileNotFoundError(f"ICU stay parquet not found: {icu_stay_path}")

    default_output_jsonl = (
        args.subset_dir / f"selected_windows_action_ratio_{str(args.ratio_threshold).replace('.', 'p')}.jsonl"
    )
    output_jsonl = args.output_jsonl or default_output_jsonl
    output_full_jsonl = Path(args.output_full_jsonl) if args.output_full_jsonl else None
    patient_stats_csv = args.patient_stats_csv or (args.subset_dir / "patient_window_selection_stats.csv")
    summary_json = args.summary_json or (args.subset_dir / "window_selection_summary.json")
    statistics_dir = args.statistics_dir or (args.subset_dir / "statistics")

    return {
        "events_path": Path(events_path),
        "icu_stay_path": Path(icu_stay_path),
        "output_jsonl": Path(output_jsonl),
        "output_full_jsonl": output_full_jsonl,
        "patient_stats_csv": Path(patient_stats_csv),
        "summary_json": Path(summary_json),
        "statistics_dir": Path(statistics_dir),
        "statistics_summary_json": Path(statistics_dir) / "selected_window_statistics_summary.json",
        "patient_selected_window_counts_csv": Path(statistics_dir) / "patient_selected_window_counts.csv",
        "patient_selected_window_histogram_csv": Path(statistics_dir) / "patient_selected_window_histogram.csv",
        "selected_window_total_event_histogram_csv": Path(statistics_dir)
        / "selected_window_total_event_histogram.csv",
        "selected_window_current_discharge_summary_counts_csv": (
            Path(statistics_dir) / "selected_window_current_discharge_summary_counts.csv"
        ),
        "selected_window_icu_timeline_segment_counts_csv": (
            Path(statistics_dir) / "selected_window_icu_timeline_segment_counts.csv"
        ),
        "selected_windows_per_patient_histogram_png": Path(statistics_dir)
        / "selected_windows_per_patient_histogram.png",
        "selected_window_total_events_histogram_png": Path(statistics_dir)
        / "selected_window_total_events_histogram.png",
        "selected_window_current_vs_history_histogram_png": Path(statistics_dir)
        / "selected_window_current_vs_history_histogram.png",
        "selected_window_current_discharge_summary_bar_png": Path(statistics_dir)
        / "selected_window_current_discharge_summary_bar.png",
        "selected_window_icu_timeline_segment_bar_png": Path(statistics_dir)
        / "selected_window_icu_timeline_segment_bar.png",
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
        "history_context_hours": float(config.oracle_context_history_hours),
        "future_context_hours": float(config.oracle_context_future_hours),
    }
    if not silence_parser_logs:
        return parser.create_time_windows(**kwargs)

    with redirect_stdout(io.StringIO()):
        return parser.create_time_windows(**kwargs)


def _outcome_key_from_survived(value: Any) -> str:
    return "survived" if bool(value) else "died"


def _record_action_ratio(record: Mapping[str, Any]) -> float:
    selection_metadata = record.get("selection_metadata")
    if isinstance(selection_metadata, Mapping):
        return _safe_float(selection_metadata.get("action_event_ratio"), default=0.0)
    return 0.0


def _record_timeline_segment(record: Mapping[str, Any]) -> str:
    selection_metadata = record.get("selection_metadata")
    if isinstance(selection_metadata, Mapping):
        segment = str(selection_metadata.get("icu_timeline_segment") or "unassigned")
        if segment in ICU_TIMELINE_SEGMENT_ORDER:
            return segment
    return "unassigned"


def _sample_records_evenly_across_timeline(
    records: Sequence[Dict[str, Any]],
    *,
    target_count: int,
    start_segment_offset: int = 0,
    late_segment_boost: int = 1,
) -> List[Dict[str, Any]]:
    if target_count <= 0:
        return []
    if not records:
        return []

    timeline_buckets: Dict[str, List[Dict[str, Any]]] = {segment: [] for segment in ICU_TIMELINE_SEGMENT_ORDER}
    for record in records:
        segment = _record_timeline_segment(record)
        if segment not in timeline_buckets:
            segment = "unassigned"
        timeline_buckets[segment].append(record)

    def sort_key(record: Mapping[str, Any]) -> int:
        return _safe_int(record.get("window_position"), default=10**9)

    timeline_queues: Dict[str, deque] = {}
    for segment in ICU_TIMELINE_SEGMENT_ORDER:
        timeline_queues[segment] = deque(sorted(timeline_buckets[segment], key=sort_key))

    primary_segments = list(ICU_TIMELINE_SEGMENT_ORDER[:-1])
    if len(primary_segments) > 0:
        offset = int(start_segment_offset) % len(primary_segments)
        primary_segments = primary_segments[offset:] + primary_segments[:offset]
    late_segment_name = "late_1_3"
    weighted_primary_segments: List[str] = []
    normalized_late_boost = max(int(late_segment_boost), 1)
    for segment in primary_segments:
        repeat_count = normalized_late_boost if segment == late_segment_name else 1
        for _ in range(repeat_count):
            weighted_primary_segments.append(segment)
    fallback_segment = ICU_TIMELINE_SEGMENT_ORDER[-1]

    selected_records: List[Dict[str, Any]] = []
    while len(selected_records) < target_count:
        picked_any = False
        for segment in weighted_primary_segments:
            queue = timeline_queues[segment]
            if len(queue) == 0:
                continue
            selected_records.append(queue.popleft())
            picked_any = True
            if len(selected_records) >= target_count:
                break
        if len(selected_records) >= target_count:
            break

        primary_remaining = any(len(timeline_queues[segment]) > 0 for segment in primary_segments)
        if (not primary_remaining) and (len(timeline_queues[fallback_segment]) > 0):
            selected_records.append(timeline_queues[fallback_segment].popleft())
            picked_any = True

        if not picked_any:
            break

    return selected_records


def _select_windows_for_patient(
    candidate_records: Sequence[Dict[str, Any]],
    *,
    ratio_threshold: float,
    min_windows_per_patient: int,
    max_windows_per_patient: int,
    valid_window_ratio_target: float,
    late_segment_boost: int,
) -> Dict[str, Any]:
    all_records = list(candidate_records)
    eligible_total = int(len(all_records))
    if eligible_total == 0:
        return {
            "selected_records": [],
            "eligible_total": 0,
            "eligible_valid": 0,
            "eligible_invalid": 0,
            "target_total": 0,
            "target_valid": 0,
            "target_invalid": 0,
            "selected_valid": 0,
            "selected_invalid": 0,
            "selected_valid_ratio": 0.0,
            "ratio_gap_to_target": float(valid_window_ratio_target),
            "meets_min_windows": False,
        }

    target_total = min(int(max_windows_per_patient), eligible_total)
    target_total = max(target_total, 0)

    valid_pool = [record for record in all_records if _record_action_ratio(record) >= ratio_threshold]
    invalid_pool = [record for record in all_records if _record_action_ratio(record) < ratio_threshold]

    target_valid = int(round(float(target_total) * float(valid_window_ratio_target)))
    target_valid = max(min(target_valid, target_total), 0)
    target_invalid = int(target_total - target_valid)

    subject_id_for_offset = _safe_int(all_records[0].get("subject_id"), default=0)
    icu_stay_id_for_offset = _safe_int(all_records[0].get("icu_stay_id"), default=0)
    segment_offset_base = (subject_id_for_offset * 31 + icu_stay_id_for_offset) % 3
    ordered_valid = _sample_records_evenly_across_timeline(
        valid_pool,
        target_count=len(valid_pool),
        start_segment_offset=segment_offset_base,
        late_segment_boost=late_segment_boost,
    )
    ordered_invalid = _sample_records_evenly_across_timeline(
        invalid_pool,
        target_count=len(invalid_pool),
        start_segment_offset=(segment_offset_base + 1),
        late_segment_boost=late_segment_boost,
    )

    selected_valid = list(ordered_valid[:target_valid])
    selected_invalid = list(ordered_invalid[:target_invalid])
    valid_cursor = int(len(selected_valid))
    invalid_cursor = int(len(selected_invalid))

    while (len(selected_valid) + len(selected_invalid)) < target_total:
        current_total = len(selected_valid) + len(selected_invalid)
        can_take_valid = valid_cursor < len(ordered_valid)
        can_take_invalid = invalid_cursor < len(ordered_invalid)
        if (not can_take_valid) and (not can_take_invalid):
            break
        if can_take_valid and (not can_take_invalid):
            selected_valid.append(ordered_valid[valid_cursor])
            valid_cursor += 1
            continue
        if can_take_invalid and (not can_take_valid):
            selected_invalid.append(ordered_invalid[invalid_cursor])
            invalid_cursor += 1
            continue

        ratio_if_take_valid = abs(
            (float(len(selected_valid) + 1) / float(current_total + 1)) - float(valid_window_ratio_target)
        )
        ratio_if_take_invalid = abs(
            (float(len(selected_valid)) / float(current_total + 1)) - float(valid_window_ratio_target)
        )
        if ratio_if_take_valid <= ratio_if_take_invalid:
            selected_valid.append(ordered_valid[valid_cursor])
            valid_cursor += 1
        else:
            selected_invalid.append(ordered_invalid[invalid_cursor])
            invalid_cursor += 1

    selected_records = list(selected_valid) + list(selected_invalid)
    selected_records = sorted(
        selected_records,
        key=lambda record: _safe_int(record.get("window_position"), default=10**9),
    )
    selected_valid_count = sum(1 for record in selected_records if _record_action_ratio(record) >= ratio_threshold)
    selected_invalid_count = int(len(selected_records) - selected_valid_count)
    selected_valid_ratio = (float(selected_valid_count) / float(len(selected_records))) if selected_records else 0.0
    ratio_gap_to_target = abs(float(selected_valid_ratio) - float(valid_window_ratio_target))

    return {
        "selected_records": selected_records,
        "eligible_total": eligible_total,
        "eligible_valid": int(len(valid_pool)),
        "eligible_invalid": int(len(invalid_pool)),
        "target_total": int(target_total),
        "target_valid": int(target_valid),
        "target_invalid": int(target_invalid),
        "selected_valid": int(selected_valid_count),
        "selected_invalid": int(selected_invalid_count),
        "selected_valid_ratio": float(selected_valid_ratio),
        "ratio_gap_to_target": float(ratio_gap_to_target),
        "meets_min_windows": bool(len(selected_records) >= int(min_windows_per_patient)),
    }


def _patient_quality_sort_key(patient_entry: Mapping[str, Any]) -> tuple[float, int, int, int, int]:
    return (
        _safe_float(patient_entry.get("ratio_gap_to_target"), default=1.0),
        -_safe_int(patient_entry.get("selected_windows"), default=0),
        -_safe_int(patient_entry.get("generated_windows"), default=0),
        _safe_int(patient_entry.get("subject_id"), default=10**9),
        _safe_int(patient_entry.get("icu_stay_id"), default=10**9),
    )


def _select_top_patients_by_quality(
    patient_entries: Sequence[Dict[str, Any]],
    *,
    target_count: int,
    min_windows_per_patient: int,
) -> List[Dict[str, Any]]:
    sorted_entries = sorted(patient_entries, key=_patient_quality_sort_key)
    enough_windows = [
        entry
        for entry in sorted_entries
        if _safe_int(entry.get("selected_windows"), default=0) >= min_windows_per_patient
    ]
    low_window_entries = [
        entry
        for entry in sorted_entries
        if _safe_int(entry.get("selected_windows"), default=0) < min_windows_per_patient
    ]

    selected_entries = list(enough_windows[:target_count])
    remaining_slots = int(target_count - len(selected_entries))
    if remaining_slots > 0:
        selected_entries.extend(low_window_entries[:remaining_slots])
    return selected_entries


def main() -> None:
    args = _parse_args()
    paths = _resolve_paths(args)
    config = load_config(args.config)

    output_parent = paths["output_jsonl"].parent
    output_parent.mkdir(parents=True, exist_ok=True)
    if paths["output_full_jsonl"] is not None:
        paths["output_full_jsonl"].parent.mkdir(parents=True, exist_ok=True)
    paths["patient_stats_csv"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["statistics_dir"].mkdir(parents=True, exist_ok=True)

    ratio_threshold = float(args.ratio_threshold)
    if ratio_threshold < 0:
        raise ValueError(f"--ratio-threshold must be >= 0, got {ratio_threshold}")
    min_windows_per_patient = int(args.min_windows_per_patient)
    max_windows_per_patient = int(args.max_windows_per_patient)
    if min_windows_per_patient <= 0:
        raise ValueError("--min-windows-per-patient must be > 0.")
    if max_windows_per_patient < min_windows_per_patient:
        raise ValueError("--max-windows-per-patient must be >= --min-windows-per-patient.")
    valid_window_ratio_target = float(args.valid_window_ratio_target)
    if (valid_window_ratio_target < 0.0) or (valid_window_ratio_target > 1.0):
        raise ValueError("--valid-window-ratio-target must be within [0, 1].")
    late_segment_boost = int(args.late_segment_boost)
    if late_segment_boost <= 0:
        raise ValueError("--late-segment-boost must be > 0.")
    selected_survived_count = int(args.selected_survived_count)
    selected_died_count = int(args.selected_died_count)
    if selected_survived_count < 0 or selected_died_count < 0:
        raise ValueError("--selected-survived-count and --selected-died-count must be >= 0.")

    parser = MIMICDataParser(
        events_path=str(paths["events_path"]),
        icu_stay_path=str(paths["icu_stay_path"]),
        discharge_summary_max_days_after_leave=float(args.max_days_after_leave),
        require_discharge_summary_for_icu_stays=True,
    )
    parser.load_data()
    if parser.icu_stay_df is None:
        raise RuntimeError("Failed to load ICU stays from subset.")

    all_stays_df = parser.icu_stay_df.copy()

    action_code_set = _build_action_code_set(HIGH_IMPACT_EVENT_CODES)
    selected_records: List[Dict[str, Any]] = []
    patient_entries: List[Dict[str, Any]] = []
    patient_stats_rows: List[Dict[str, Any]] = []
    total_windows_generated = 0
    skipped_no_windows = 0
    processed_stay_keys: set[tuple[int, int]] = set()

    print(
        "Window generation config: "
        f"current={config.oracle_current_window_hours}h, "
        f"step={config.oracle_window_step_hours}h, "
        f"include_pre_icu_data={config.oracle_include_pre_icu_data}, "
        f"use_discharge_summary_for_history={config.oracle_use_discharge_summary_for_history}, "
        f"observation_hours={config.oracle_observation_hours}, "
        f"history_context_hours={config.oracle_context_history_hours}, "
        f"future_context_hours={config.oracle_context_future_hours}"
    )
    print(
        "Window filtering config: "
        f"ratio_threshold>={ratio_threshold:.4f}, "
        f"action_codes={sorted(action_code_set)}, "
        f"min_windows_per_patient={min_windows_per_patient}, "
        f"max_windows_per_patient={max_windows_per_patient}, "
        f"valid_window_ratio_target={valid_window_ratio_target:.4f}, "
        f"late_segment_boost={late_segment_boost}"
    )
    print(
        "Final cohort config: " f"selected_survived={selected_survived_count}, " f"selected_died={selected_died_count}"
    )
    print(f"Candidate ICU stays loaded: {len(all_stays_df)}")

    for _, stay in all_stays_df.iterrows():
        subject_id = int(stay["subject_id"])
        icu_stay_id = int(stay["icu_stay_id"])
        stay_key = (subject_id, icu_stay_id)
        if stay_key in processed_stay_keys:
            continue
        processed_stay_keys.add(stay_key)

        trajectory = parser.get_patient_trajectory(subject_id, icu_stay_id, icu_stay=stay)
        windows = _create_windows_for_trajectory(
            parser,
            trajectory,
            config,
            silence_parser_logs=bool(args.silence_parser_window_logs),
        )
        generated_for_patient = int(len(windows))
        total_windows_generated += generated_for_patient
        if generated_for_patient == 0:
            skipped_no_windows += 1

        trajectory_metadata_payload = _trajectory_metadata_for_export(trajectory)
        candidate_records_for_patient: List[Dict[str, Any]] = []
        for window_position, window in enumerate(windows, start=1):
            current_events = _normalize_event_list(window.get("current_events"))
            ratio_stats = _compute_action_ratio(current_events=current_events, action_code_set=action_code_set)

            ratio_value = float(ratio_stats["action_event_ratio"])

            timeline_stats = _icu_timeline_fraction_and_segment(
                hours_since_admission=window.get("hours_since_admission"),
                icu_duration_hours=trajectory_metadata_payload.get("icu_duration_hours"),
            )
            ratio_bucket = "valid" if ratio_value >= ratio_threshold else "invalid"

            record = dict(window)
            record["patient_id"] = f"{subject_id}_{icu_stay_id}"
            record["window_index"] = int(window_position - 1)  # Oracle uses 0-based index internally.
            record["window_position"] = int(window_position)  # Human-friendly 1-based index.
            record["trajectory_metadata"] = trajectory_metadata_payload
            record["selection_metadata"] = {
                "selection_rule": "timeline_first_ratio_target_quality_ranked_patient_selection",
                "ratio_threshold": ratio_threshold,
                "threshold_comparator": ">=",
                "ratio_bucket": ratio_bucket,
                "valid_window_ratio_target": float(valid_window_ratio_target),
                "min_windows_per_patient": int(min_windows_per_patient),
                "max_windows_per_patient": int(max_windows_per_patient),
                "late_segment_boost": int(late_segment_boost),
                "action_event_codes_reference": list(HIGH_IMPACT_EVENT_CODES),
                "action_event_count": int(ratio_stats["action_event_count"]),
                "total_event_count": int(ratio_stats["total_event_count"]),
                "action_event_ratio": ratio_value,
                "action_event_code_counts": ratio_stats["action_event_code_counts"],
                "icu_timeline_fraction": timeline_stats["icu_timeline_fraction"],
                "icu_timeline_segment": timeline_stats["icu_timeline_segment"],
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
                "has_current_discharge_summary": _has_current_discharge_summary(record),
                "num_history_events": record.get("num_history_events"),
                "num_current_events": record.get("num_current_events"),
                "action_event_ratio": ratio_value,
                "ratio_bucket": ratio_bucket,
                "icu_timeline_segment": timeline_stats["icu_timeline_segment"],
            }
            candidate_records_for_patient.append(record)

        selection_result = _select_windows_for_patient(
            candidate_records_for_patient,
            ratio_threshold=ratio_threshold,
            min_windows_per_patient=min_windows_per_patient,
            max_windows_per_patient=max_windows_per_patient,
            valid_window_ratio_target=valid_window_ratio_target,
            late_segment_boost=late_segment_boost,
        )
        selected_records_for_patient = selection_result["selected_records"]
        selected_for_patient = int(len(selected_records_for_patient))
        if selected_for_patient > 0:
            for kept_record in selected_records_for_patient:
                selection_metadata = kept_record.get("selection_metadata")
                if isinstance(selection_metadata, dict):
                    selection_metadata["min_windows_per_patient"] = int(min_windows_per_patient)
                    selection_metadata["max_windows_per_patient"] = int(max_windows_per_patient)
                    selection_metadata["late_segment_boost"] = int(late_segment_boost)
                    selection_metadata["window_count_eligible_before_patient_selection"] = int(
                        selection_result["eligible_total"]
                    )
                    selection_metadata["window_count_after_patient_selection"] = selected_for_patient
                    selection_metadata["eligible_valid_window_count"] = int(selection_result["eligible_valid"])
                    selection_metadata["eligible_invalid_window_count"] = int(selection_result["eligible_invalid"])
                    selection_metadata["target_valid_window_count"] = int(selection_result["target_valid"])
                    selection_metadata["target_invalid_window_count"] = int(selection_result["target_invalid"])
                    selection_metadata["selected_valid_window_count"] = int(selection_result["selected_valid"])
                    selection_metadata["selected_invalid_window_count"] = int(selection_result["selected_invalid"])
                    selection_metadata["selected_valid_window_ratio"] = float(selection_result["selected_valid_ratio"])
                    selection_metadata["ratio_gap_to_target"] = float(selection_result["ratio_gap_to_target"])
                    selection_metadata["meets_min_windows"] = bool(selection_result["meets_min_windows"])

        patient_entries.append(
            {
                "subject_id": int(subject_id),
                "icu_stay_id": int(icu_stay_id),
                "survived": bool(trajectory.get("survived")),
                "generated_windows": int(generated_for_patient),
                "selected_windows": int(selected_for_patient),
                "selected_valid": int(selection_result["selected_valid"]),
                "selected_invalid": int(selection_result["selected_invalid"]),
                "selected_valid_ratio": float(selection_result["selected_valid_ratio"]),
                "ratio_gap_to_target": float(selection_result["ratio_gap_to_target"]),
                "meets_min_windows": bool(selection_result["meets_min_windows"]),
                "selected_records": selected_records_for_patient,
                "all_records": candidate_records_for_patient,
            }
        )

    survived_candidates = [
        entry for entry in patient_entries if bool(entry["survived"]) and int(entry["selected_windows"]) > 0
    ]
    died_candidates = [
        entry for entry in patient_entries if (not bool(entry["survived"])) and int(entry["selected_windows"]) > 0
    ]

    if len(survived_candidates) < selected_survived_count:
        raise RuntimeError(
            "Not enough survived patients with selected windows for final cohort selection. "
            f"Required={selected_survived_count}, available={len(survived_candidates)}."
        )
    if len(died_candidates) < selected_died_count:
        raise RuntimeError(
            "Not enough died patients with selected windows for final cohort selection. "
            f"Required={selected_died_count}, available={len(died_candidates)}."
        )

    selected_survived_entries = _select_top_patients_by_quality(
        survived_candidates,
        target_count=selected_survived_count,
        min_windows_per_patient=min_windows_per_patient,
    )
    selected_died_entries = _select_top_patients_by_quality(
        died_candidates,
        target_count=selected_died_count,
        min_windows_per_patient=min_windows_per_patient,
    )

    selected_entry_key_set: set[tuple[int, int]] = set()
    full_records_for_selected_patients: List[Dict[str, Any]] = []
    for selected_entry in selected_survived_entries + selected_died_entries:
        selected_entry_key_set.add((int(selected_entry["subject_id"]), int(selected_entry["icu_stay_id"])))
        selected_records.extend(selected_entry["selected_records"])
        all_records = selected_entry.get("all_records")
        if isinstance(all_records, list):
            sorted_all_records = sorted(
                all_records,
                key=lambda record: _safe_int(record.get("window_position"), default=10**9),
            )
            full_records_for_selected_patients.extend(sorted_all_records)

    total_windows_selected = int(len(selected_records))
    selected_patient_counts = {
        "survived": int(len(selected_survived_entries)),
        "died": int(len(selected_died_entries)),
    }

    for patient_index, entry in enumerate(patient_entries, start=1):
        stay_key = (int(entry["subject_id"]), int(entry["icu_stay_id"]))
        in_final_cohort = stay_key in selected_entry_key_set
        selected_count_for_stats = int(entry["selected_windows"]) if in_final_cohort else 0
        generated_windows = int(entry["generated_windows"])
        patient_stats_rows.append(
            {
                "patient_index": int(patient_index),
                "subject_id": int(entry["subject_id"]),
                "icu_stay_id": int(entry["icu_stay_id"]),
                "survived": bool(entry["survived"]),
                "generated_windows": generated_windows,
                "selected_windows": selected_count_for_stats,
                "selected_valid_windows": int(entry["selected_valid"]) if in_final_cohort else 0,
                "selected_invalid_windows": int(entry["selected_invalid"]) if in_final_cohort else 0,
                "selected_valid_ratio": float(entry["selected_valid_ratio"]) if in_final_cohort else 0.0,
                "ratio_gap_to_target": float(entry["ratio_gap_to_target"]) if in_final_cohort else 0.0,
                "meets_min_windows": bool(entry["meets_min_windows"]),
                "selected_for_final_cohort": bool(in_final_cohort),
                "selected_ratio": (
                    (float(selected_count_for_stats) / float(generated_windows)) if generated_windows > 0 else 0.0
                ),
                "window_generation_current_hours": float(config.oracle_current_window_hours),
                "window_generation_step_hours": float(config.oracle_window_step_hours),
                "window_filter_ratio_threshold": ratio_threshold,
            }
        )

    with open(paths["output_jsonl"], "w", encoding="utf-8") as file:
        for record in selected_records:
            file.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    if paths["output_full_jsonl"] is not None:
        with open(paths["output_full_jsonl"], "w", encoding="utf-8") as file:
            for record in full_records_for_selected_patients:
                file.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    patient_stats_df = pd.DataFrame(patient_stats_rows, columns=PATIENT_STATS_COLUMNS)
    patient_stats_df.to_csv(paths["patient_stats_csv"], index=False)

    selected_window_stats_rows: List[Dict[str, Any]] = []
    for record in selected_records:
        current_events = _normalize_event_list(record.get("current_events"))
        history_events = _normalize_event_list(record.get("history_events"))
        num_current_events = _safe_int(record.get("num_current_events"), default=len(current_events))
        num_history_events = _safe_int(record.get("num_history_events"), default=len(history_events))
        trajectory_metadata = record.get("trajectory_metadata")
        if isinstance(trajectory_metadata, Mapping):
            icu_duration_hours = trajectory_metadata.get("icu_duration_hours")
        else:
            icu_duration_hours = None
        timeline_stats = _icu_timeline_fraction_and_segment(
            hours_since_admission=record.get("hours_since_admission"),
            icu_duration_hours=icu_duration_hours,
        )
        selection_metadata = record.get("selection_metadata")
        if isinstance(selection_metadata, Mapping):
            action_event_ratio = _safe_float(selection_metadata.get("action_event_ratio"), default=0.0)
            window_bucket = str(selection_metadata.get("ratio_bucket") or "unknown")
        else:
            action_event_ratio = 0.0
            window_bucket = "unknown"
        selected_window_stats_rows.append(
            {
                "patient_id": str(record.get("patient_id") or ""),
                "subject_id": _safe_int(record.get("subject_id")),
                "icu_stay_id": _safe_int(record.get("icu_stay_id")),
                "num_current_events": int(num_current_events),
                "num_history_events": int(num_history_events),
                "num_total_events": int(num_current_events + num_history_events),
                "action_event_ratio": float(action_event_ratio),
                "window_bucket": window_bucket,
                "has_current_discharge_summary": _has_current_discharge_summary(record),
                "icu_timeline_fraction": timeline_stats["icu_timeline_fraction"],
                "icu_timeline_segment": timeline_stats["icu_timeline_segment"],
            }
        )
    selected_windows_df = pd.DataFrame(selected_window_stats_rows, columns=SELECTED_WINDOW_STATS_COLUMNS)

    patient_selected_window_counts_df = patient_stats_df[
        ["subject_id", "icu_stay_id", "generated_windows", "selected_windows"]
    ].copy()
    patient_selected_window_counts_df["patient_id"] = (
        patient_selected_window_counts_df["subject_id"].astype("Int64").astype(str)
        + "_"
        + patient_selected_window_counts_df["icu_stay_id"].astype("Int64").astype(str)
    )
    patient_selected_window_counts_df = patient_selected_window_counts_df[
        ["patient_id", "subject_id", "icu_stay_id", "generated_windows", "selected_windows"]
    ]

    if patient_selected_window_counts_df.empty:
        patient_selected_window_histogram_df = pd.DataFrame(columns=["selected_windows_per_patient", "patient_count"])
    else:
        patient_selected_window_histogram_df = (
            patient_selected_window_counts_df["selected_windows"]
            .value_counts()
            .sort_index()
            .rename_axis("selected_windows_per_patient")
            .reset_index(name="patient_count")
        )

    if selected_windows_df.empty:
        window_total_event_histogram_df = pd.DataFrame(columns=["total_events_per_window", "window_count"])
        current_discharge_summary_counts_df = pd.DataFrame(
            columns=["has_current_discharge_summary", "window_count", "window_ratio"]
        )
        icu_timeline_segment_counts_df = pd.DataFrame(columns=["icu_timeline_segment", "window_count", "window_ratio"])
    else:
        window_total_event_histogram_df = (
            selected_windows_df["num_total_events"]
            .value_counts()
            .sort_index()
            .rename_axis("total_events_per_window")
            .reset_index(name="window_count")
        )
        current_discharge_summary_counts_df = (
            selected_windows_df["has_current_discharge_summary"]
            .value_counts()
            .rename_axis("has_current_discharge_summary")
            .reset_index(name="window_count")
            .sort_values("has_current_discharge_summary")
            .reset_index(drop=True)
        )
        current_discharge_summary_counts_df["window_ratio"] = current_discharge_summary_counts_df[
            "window_count"
        ] / float(len(selected_windows_df))
        icu_timeline_segment_counts = (
            selected_windows_df["icu_timeline_segment"]
            .value_counts()
            .reindex(ICU_TIMELINE_SEGMENT_ORDER, fill_value=0)
        )
        icu_timeline_segment_counts_df = icu_timeline_segment_counts.rename_axis("icu_timeline_segment").reset_index(
            name="window_count"
        )
        icu_timeline_segment_counts_df["window_ratio"] = icu_timeline_segment_counts_df["window_count"] / float(
            len(selected_windows_df)
        )

    patient_selected_window_counts_df.to_csv(paths["patient_selected_window_counts_csv"], index=False)
    patient_selected_window_histogram_df.to_csv(paths["patient_selected_window_histogram_csv"], index=False)
    window_total_event_histogram_df.to_csv(paths["selected_window_total_event_histogram_csv"], index=False)
    current_discharge_summary_counts_df.to_csv(
        paths["selected_window_current_discharge_summary_counts_csv"],
        index=False,
    )
    icu_timeline_segment_counts_df.to_csv(
        paths["selected_window_icu_timeline_segment_counts_csv"],
        index=False,
    )
    _save_selected_windows_per_patient_histogram(
        patient_selected_window_counts_df,
        paths["selected_windows_per_patient_histogram_png"],
    )
    _save_total_events_per_window_histogram(
        selected_windows_df,
        paths["selected_window_total_events_histogram_png"],
    )
    _save_current_history_events_histogram(
        selected_windows_df,
        paths["selected_window_current_vs_history_histogram_png"],
    )
    _save_current_discharge_summary_barplot(
        current_discharge_summary_counts_df,
        paths["selected_window_current_discharge_summary_bar_png"],
    )
    _save_timeline_segment_barplot(
        icu_timeline_segment_counts_df,
        paths["selected_window_icu_timeline_segment_bar_png"],
    )

    selected_window_count = int(len(selected_windows_df))
    total_current_events = int(selected_windows_df["num_current_events"].sum()) if selected_window_count > 0 else 0
    total_history_events = int(selected_windows_df["num_history_events"].sum()) if selected_window_count > 0 else 0
    total_events = int(total_current_events + total_history_events)
    selected_valid_window_count = (
        int((selected_windows_df["window_bucket"] == "valid").sum()) if selected_window_count > 0 else 0
    )
    selected_invalid_window_count = (
        int((selected_windows_df["window_bucket"] == "invalid").sum()) if selected_window_count > 0 else 0
    )
    windows_with_current_discharge_summary = (
        int(selected_windows_df["has_current_discharge_summary"].sum()) if selected_window_count > 0 else 0
    )
    selected_window_action_ratio_stats = _series_distribution_stats(selected_windows_df["action_event_ratio"])
    selected_window_event_stats = _series_distribution_stats(selected_windows_df["num_total_events"])
    selected_window_current_event_stats = _series_distribution_stats(selected_windows_df["num_current_events"])
    selected_window_history_event_stats = _series_distribution_stats(selected_windows_df["num_history_events"])
    patient_selected_window_stats = _series_distribution_stats(patient_selected_window_counts_df["selected_windows"])
    selected_patient_rows_df = patient_stats_df[patient_stats_df["selected_windows"] > 0].copy()
    if selected_patient_rows_df.empty:
        selected_survived_patient_count = 0
        selected_died_patient_count = 0
    else:
        survived_mask = selected_patient_rows_df["survived"].astype(bool)
        selected_survived_patient_count = int(survived_mask.sum())
        selected_died_patient_count = int((~survived_mask).sum())

    statistics_summary_payload = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "statistics_dir": str(paths["statistics_dir"]),
        "window_level_stats": {
            "selected_window_count": selected_window_count,
            "selected_valid_window_count": int(selected_valid_window_count),
            "selected_invalid_window_count": int(selected_invalid_window_count),
            "selected_valid_window_ratio": (
                float(selected_valid_window_count) / float(selected_window_count) if selected_window_count > 0 else 0.0
            ),
            "selected_invalid_window_ratio": (
                float(selected_invalid_window_count) / float(selected_window_count)
                if selected_window_count > 0
                else 0.0
            ),
            "total_current_event_count": total_current_events,
            "total_history_event_count": total_history_events,
            "total_event_count": total_events,
            "windows_with_current_discharge_summary": int(windows_with_current_discharge_summary),
            "windows_with_current_discharge_summary_ratio": (
                float(windows_with_current_discharge_summary) / float(selected_window_count)
                if selected_window_count > 0
                else 0.0
            ),
            "action_event_ratio_distribution": selected_window_action_ratio_stats,
            "total_events_per_window_distribution": selected_window_event_stats,
            "current_events_per_window_distribution": selected_window_current_event_stats,
            "history_events_per_window_distribution": selected_window_history_event_stats,
            "icu_timeline_segment_distribution": icu_timeline_segment_counts_df.to_dict(orient="records"),
        },
        "patient_level_stats": {
            "patients_processed": int(len(patient_selected_window_counts_df)),
            "patients_with_at_least_one_selected_window": (
                int((patient_selected_window_counts_df["selected_windows"] > 0).sum())
                if len(patient_selected_window_counts_df) > 0
                else 0
            ),
            "selected_survived_patients": int(selected_survived_patient_count),
            "selected_died_patients": int(selected_died_patient_count),
            "selected_windows_per_patient_distribution": patient_selected_window_stats,
        },
        "artifact_paths": {
            "patient_selected_window_counts_csv": str(paths["patient_selected_window_counts_csv"]),
            "patient_selected_window_histogram_csv": str(paths["patient_selected_window_histogram_csv"]),
            "selected_window_total_event_histogram_csv": str(paths["selected_window_total_event_histogram_csv"]),
            "selected_window_current_discharge_summary_counts_csv": str(
                paths["selected_window_current_discharge_summary_counts_csv"]
            ),
            "selected_window_icu_timeline_segment_counts_csv": str(
                paths["selected_window_icu_timeline_segment_counts_csv"]
            ),
            "selected_windows_per_patient_histogram_png": str(paths["selected_windows_per_patient_histogram_png"]),
            "selected_window_total_events_histogram_png": str(paths["selected_window_total_events_histogram_png"]),
            "selected_window_current_vs_history_histogram_png": str(
                paths["selected_window_current_vs_history_histogram_png"]
            ),
            "selected_window_current_discharge_summary_bar_png": str(
                paths["selected_window_current_discharge_summary_bar_png"]
            ),
            "selected_window_icu_timeline_segment_bar_png": str(paths["selected_window_icu_timeline_segment_bar_png"]),
        },
    }
    with open(paths["statistics_summary_json"], "w", encoding="utf-8") as file:
        json.dump(statistics_summary_payload, file, indent=2, ensure_ascii=False, default=_json_default)

    selected_patient_count = int((patient_stats_df["selected_windows"] > 0).sum()) if len(patient_stats_df) > 0 else 0
    generated_patient_count = int(len(patient_stats_df))
    selected_survived_meets_min = int(
        sum(1 for entry in selected_survived_entries if bool(entry.get("meets_min_windows")))
    )
    selected_died_meets_min = int(sum(1 for entry in selected_died_entries if bool(entry.get("meets_min_windows"))))
    quality_selection_payload = {
        "target_survived_count": int(selected_survived_count),
        "target_died_count": int(selected_died_count),
        "available_survived_candidates": int(len(survived_candidates)),
        "available_died_candidates": int(len(died_candidates)),
        "selected_survived_count": int(len(selected_survived_entries)),
        "selected_died_count": int(len(selected_died_entries)),
        "min_windows_per_patient": int(min_windows_per_patient),
        "max_windows_per_patient": int(max_windows_per_patient),
        "late_segment_boost": int(late_segment_boost),
        "selected_survived_meeting_min_windows": int(selected_survived_meets_min),
        "selected_died_meeting_min_windows": int(selected_died_meets_min),
    }

    summary_payload = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": str(Path(args.config)),
        "events_path": str(paths["events_path"]),
        "icu_stay_path": str(paths["icu_stay_path"]),
        "output_jsonl": str(paths["output_jsonl"]),
        "output_full_jsonl": str(paths["output_full_jsonl"]) if paths["output_full_jsonl"] is not None else None,
        "patient_stats_csv": str(paths["patient_stats_csv"]),
        "statistics_dir": str(paths["statistics_dir"]),
        "statistics_summary_json": str(paths["statistics_summary_json"]),
        "ratio_threshold": ratio_threshold,
        "min_windows_per_patient": int(min_windows_per_patient),
        "max_windows_per_patient": int(max_windows_per_patient),
        "valid_window_ratio_target": float(valid_window_ratio_target),
        "late_segment_boost": int(late_segment_boost),
        "threshold_comparator": ">=",
        "high_impact_event_codes": list(HIGH_IMPACT_EVENT_CODES),
        "quality_based_final_cohort_selection": quality_selection_payload,
        "window_generation_config": {
            "current_window_hours": float(config.oracle_current_window_hours),
            "window_step_hours": float(config.oracle_window_step_hours),
            "include_pre_icu_data": bool(config.oracle_include_pre_icu_data),
            "use_first_n_hours_after_icu": config.oracle_observation_hours,
            "use_discharge_summary_for_history": bool(config.oracle_use_discharge_summary_for_history),
            "num_discharge_summaries": int(config.oracle_num_discharge_summaries),
            "relative_report_codes": list(config.oracle_relative_report_codes),
            "pre_icu_history_hours": float(config.oracle_pre_icu_history_hours),
            "history_context_hours": float(config.oracle_context_history_hours),
            "future_context_hours": float(config.oracle_context_future_hours),
        },
        "stats": {
            "patients_processed": generated_patient_count,
            "patients_with_at_least_one_selected_window": selected_patient_count,
            "patients_with_selected_windows_survived": int(selected_survived_patient_count),
            "patients_with_selected_windows_died": int(selected_died_patient_count),
            "patients_skipped_no_windows_generated": int(skipped_no_windows),
            "total_windows_generated": int(total_windows_generated),
            "total_windows_selected": int(total_windows_selected),
            "total_windows_exported_for_selected_patients": int(len(full_records_for_selected_patients)),
            "selected_valid_window_count": int(selected_valid_window_count),
            "selected_invalid_window_count": int(selected_invalid_window_count),
            "selected_valid_window_ratio": (
                float(selected_valid_window_count) / float(selected_window_count) if selected_window_count > 0 else 0.0
            ),
            "selected_invalid_window_ratio": (
                float(selected_invalid_window_count) / float(selected_window_count)
                if selected_window_count > 0
                else 0.0
            ),
            "overall_selected_ratio": (
                (float(total_windows_selected) / float(total_windows_generated))
                if total_windows_generated > 0
                else 0.0
            ),
        },
    }
    with open(paths["summary_json"], "w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2, ensure_ascii=False, default=_json_default)

    print("Wrote outputs:")
    print(f"  - Selected windows JSONL: {paths['output_jsonl']}")
    if paths["output_full_jsonl"] is not None:
        print(f"  - Full windows JSONL (selected patients): {paths['output_full_jsonl']}")
    print(f"  - Patient stats CSV: {paths['patient_stats_csv']}")
    print(f"  - Summary JSON: {paths['summary_json']}")
    print(f"  - Statistics summary JSON: {paths['statistics_summary_json']}")
    print(f"  - Patient selected-window counts CSV: {paths['patient_selected_window_counts_csv']}")
    print(f"  - Patient selected-window histogram CSV: {paths['patient_selected_window_histogram_csv']}")
    print(f"  - Selected-window total-event histogram CSV: {paths['selected_window_total_event_histogram_csv']}")
    print(
        "  - Selected-window current-discharge-summary counts CSV: "
        f"{paths['selected_window_current_discharge_summary_counts_csv']}"
    )
    print(
        "  - Selected-window ICU-timeline-segment counts CSV: "
        f"{paths['selected_window_icu_timeline_segment_counts_csv']}"
    )
    print("  - Selected-windows-per-patient histogram PNG: " f"{paths['selected_windows_per_patient_histogram_png']}")
    print("  - Selected-window total-events histogram PNG: " f"{paths['selected_window_total_events_histogram_png']}")
    print(
        "  - Selected-window current-vs-history histogram PNG: "
        f"{paths['selected_window_current_vs_history_histogram_png']}"
    )
    print(
        "  - Selected-window current-discharge-summary bar PNG: "
        f"{paths['selected_window_current_discharge_summary_bar_png']}"
    )
    print(
        "  - Selected-window ICU-timeline-segment bar PNG: " f"{paths['selected_window_icu_timeline_segment_bar_png']}"
    )
    print(
        "Selection stats: "
        f"windows_selected={total_windows_selected} / windows_generated={total_windows_generated}, "
        f"patients_with_selected_windows={selected_patient_count} / {generated_patient_count}"
    )
    print(
        "Selection mix stats: "
        f"valid={selected_valid_window_count}, "
        f"invalid={selected_invalid_window_count}, "
        f"valid_ratio={((float(selected_valid_window_count) / float(selected_window_count)) if selected_window_count > 0 else 0.0):.4f}"
    )
    print(
        "Final cohort stats: "
        f"survived={len(selected_survived_entries)}/{selected_survived_count}, "
        f"died={len(selected_died_entries)}/{selected_died_count}, "
        f"min_windows={min_windows_per_patient}, "
        f"max_windows={max_windows_per_patient}, "
        f"late_segment_boost={late_segment_boost}"
    )


if __name__ == "__main__":
    main()
