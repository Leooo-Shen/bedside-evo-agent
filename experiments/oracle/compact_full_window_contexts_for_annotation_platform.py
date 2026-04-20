"""Create a compact full_window_contexts JSON for annotation platform loading."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact full_window_contexts JSON that keeps only fields used by annotation platform."
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def sanitize_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        sanitized = {key: sanitize_value(item) for key, item in value.items()}
        return {key: item for key, item in sanitized.items() if item is not None and item != ""}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    return value


def first_non_null(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None


def to_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compact_event(raw_event: Any) -> dict[str, Any] | None:
    event = as_mapping(raw_event)
    if not event:
        return None

    compact = {
        "time": first_non_null(event.get("time"), event.get("start_time")),
        "code": event.get("code"),
        "code_specifics": event.get("code_specifics"),
        "numeric_value": event.get("numeric_value"),
        "text_value": event.get("text_value"),
        "event_id": first_non_null(
            event.get("event_id"),
            event.get("evidence_event_id"),
            event.get("event_ref"),
            event.get("event_reference"),
        ),
    }
    compact = sanitize_value(compact)
    if not isinstance(compact, dict):
        return None
    if len(compact) == 0:
        return None
    if not compact.get("time") and not compact.get("code") and not compact.get("code_specifics"):
        return None
    return compact


def compact_window(raw_window: Any, fallback_index: int) -> dict[str, Any] | None:
    window = as_mapping(raw_window)
    if not window:
        return None

    metadata = as_mapping(window.get("window_metadata"))
    window_index = first_non_null(
        to_int_or_none(window.get("window_index")),
        to_int_or_none(metadata.get("source_window_index")),
        fallback_index,
    )
    source_window_index = first_non_null(
        to_int_or_none(window.get("source_window_index")),
        to_int_or_none(metadata.get("source_window_index")),
        window_index,
    )
    source_window_position = first_non_null(
        to_int_or_none(window.get("source_window_position")),
        to_int_or_none(metadata.get("source_window_position")),
        source_window_index + 1 if isinstance(source_window_index, int) else None,
    )
    current_window_start = first_non_null(
        window.get("current_window_start"),
        window.get("current_window_start_time"),
        metadata.get("window_start_time"),
    )

    current_events = [event for event in (compact_event(item) for item in as_list(window.get("current_events"))) if event]

    compact = {
        "window_index": window_index,
        "source_window_index": source_window_index,
        "source_window_position": source_window_position,
        "current_window_start": current_window_start,
        "window_metadata": {
            "source_window_index": source_window_index,
            "source_window_position": source_window_position,
            "window_start_time": current_window_start,
        },
        "current_events": current_events,
    }
    compact = sanitize_value(compact)
    return compact if isinstance(compact, dict) else None


def compact_payload(raw_payload: Any, source_file: Path) -> dict[str, Any]:
    payload = as_mapping(raw_payload)
    if not payload:
        raise ValueError(f"Input is not a JSON object: {source_file}")

    raw_windows = as_list(payload.get("window_contexts"))
    compact_windows: list[dict[str, Any]] = []
    for index, item in enumerate(raw_windows):
        compact = compact_window(item, fallback_index=index)
        if compact:
            compact_windows.append(compact)

    output = {
        "run_id": payload.get("run_id"),
        "generated_at": payload.get("generated_at"),
        "subject_id": payload.get("subject_id"),
        "icu_stay_id": payload.get("icu_stay_id"),
        "trajectory_metadata": payload.get("trajectory_metadata"),
        "history_hours": payload.get("history_hours"),
        "future_hours": payload.get("future_hours"),
        "window_contexts": compact_windows,
        "compact_for_annotation_platform": True,
        "compact_generated_at": datetime.now(timezone.utc).isoformat(),
        "compact_source_file": str(source_file),
    }
    output = sanitize_value(output)
    if not isinstance(output, dict):
        raise ValueError("Failed to build compact payload.")
    return output


def main() -> None:
    args = parse_args()
    input_file = args.input_file.resolve()
    output_file = args.output_file.resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_file}. Use --overwrite to replace.")

    with input_file.open("r", encoding="utf-8") as f:
        raw_payload = json.load(f)

    compact = compact_payload(raw_payload, source_file=input_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Window contexts: {len(as_list(as_mapping(raw_payload).get('window_contexts')))} -> {len(as_list(compact.get('window_contexts')))}")
    print(f"Input bytes: {input_file.stat().st_size}")
    print(f"Output bytes: {output_file.stat().st_size}")


if __name__ == "__main__":
    main()
