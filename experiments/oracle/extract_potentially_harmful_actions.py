"""Extract potentially harmful Oracle action labels for manual investigation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all potentially_harmful action evaluations from an Oracle dataset."
    )
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--top-k", type=int, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prediction_paths(oracle_dir: Path) -> list[Path]:
    paths = sorted((oracle_dir / "patients").glob("*/oracle_predictions.json"))
    if not paths:
        raise FileNotFoundError(f"No oracle_predictions.json files found under {oracle_dir / 'patients'}")
    return paths


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalized_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}


def matching_red_flags(action_name: str, red_flag_actions: Iterable[str]) -> list[str]:
    action_tokens = normalized_tokens(action_name)
    if not action_tokens:
        return []
    matches = []
    for red_flag_action in red_flag_actions:
        red_flag_tokens = normalized_tokens(red_flag_action)
        if not red_flag_tokens:
            continue
        overlap = action_tokens & red_flag_tokens
        if len(overlap) >= 2 or action_tokens <= red_flag_tokens or red_flag_tokens <= action_tokens:
            matches.append(red_flag_action)
    return matches


def stringify_list(values: Iterable[str]) -> str:
    cleaned = [value for value in (clean_text(value) for value in values) if value]
    return " | ".join(cleaned)


def extract_rows(oracle_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for pred_path in prediction_paths(oracle_dir):
        payload = read_json(pred_path)
        subject_id = payload.get("subject_id")
        icu_stay_id = payload.get("icu_stay_id")
        patient_id = pred_path.parent.name
        patient_dir = pred_path.parent
        trajectory_metadata = as_mapping(payload.get("trajectory_metadata"))

        for window in as_list(payload.get("window_outputs")):
            window_mapping = as_mapping(window)
            window_metadata = as_mapping(window_mapping.get("window_metadata"))
            oracle_output = as_mapping(window_mapping.get("oracle_output"))
            patient_assessment = as_mapping(oracle_output.get("patient_assessment"))
            overall = as_mapping(patient_assessment.get("overall"))
            action_review = as_mapping(oracle_output.get("action_review"))
            evaluations = as_list(action_review.get("evaluations"))
            red_flags = as_list(action_review.get("red_flags"))
            active_risks = as_list(patient_assessment.get("active_risks"))

            red_flag_actions = [
                clean_text(as_mapping(red_flag).get("contraindicated_action")) for red_flag in red_flags
            ]
            active_risk_names = [clean_text(as_mapping(risk).get("risk_name")) for risk in active_risks]

            for evaluation in evaluations:
                evaluation_mapping = as_mapping(evaluation)
                if clean_text(evaluation_mapping.get("label")) != "potentially_harmful":
                    continue

                action_name = clean_text(evaluation_mapping.get("action_name"))
                matched_red_flags = matching_red_flags(action_name, red_flag_actions)
                row = {
                    "patient_id": patient_id,
                    "subject_id": subject_id,
                    "icu_stay_id": icu_stay_id,
                    "run_id": payload.get("run_id"),
                    "generated_at": payload.get("generated_at"),
                    "window_index": window_mapping.get("window_index"),
                    "source_window_index": window_mapping.get("source_window_index"),
                    "stride_source_window_index": window_mapping.get("stride_source_window_index"),
                    "hours_since_admission": window_metadata.get("hours_since_admission"),
                    "window_start_time": window_metadata.get("window_start_time"),
                    "window_end_time": window_metadata.get("window_end_time"),
                    "current_window_hours": window_metadata.get("current_window_hours"),
                    "num_current_events": window_metadata.get("num_current_events"),
                    "num_history_events": window_metadata.get("num_history_events"),
                    "patient_status_label": clean_text(overall.get("label")),
                    "patient_status_rationale": clean_text(overall.get("rationale")),
                    "active_risk_count": len(active_risk_names),
                    "active_risks": active_risk_names,
                    "action_id": clean_text(evaluation_mapping.get("action_id")),
                    "action_name": action_name,
                    "action_rationale": clean_text(evaluation_mapping.get("rationale")),
                    "red_flag_count": len(red_flag_actions),
                    "red_flag_actions": red_flag_actions,
                    "matching_red_flags": matched_red_flags,
                    "matching_red_flag_count": len(matched_red_flags),
                    "trajectory_survived": trajectory_metadata.get("survived"),
                    "trajectory_icu_outcome": clean_text(trajectory_metadata.get("icu_outcome")),
                    "oracle_predictions_path": str(pred_path),
                    "llm_calls_html_path": str(patient_dir / "llm_calls.html"),
                }
                rows.append(row)

    return rows


def counter_rows(counter: Counter, total: int, top_k: int) -> list[dict[str, Any]]:
    rows = []
    for value, count in counter.most_common(top_k):
        rows.append({"value": value, "count": count, "percent": count / total * 100 if total else None})
    return rows


def build_summary(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    action_names = Counter()
    patient_ids = Counter()
    status_labels = Counter()
    windows = Counter()
    window_hours: dict[tuple[str, Any], Any] = {}
    matched_red_flag_counter = Counter()
    trajectory_outcomes = Counter()

    for row in rows:
        action_names.update([row["action_name"]])
        patient_ids.update([row["patient_id"]])
        status_labels.update([row["patient_status_label"] or "<missing>"])
        window_key = (row["patient_id"], row["window_index"])
        windows.update([window_key])
        window_hours[window_key] = row["hours_since_admission"]
        trajectory_outcomes.update([row["trajectory_icu_outcome"] or "<missing>"])
        matched_red_flag_counter.update(["matched_red_flag" if row["matching_red_flag_count"] else "no_matched_red_flag"])

    rows_per_window = Counter(windows.values())
    top_windows = []
    for (patient_id, window_index), count in windows.most_common(top_k):
        top_windows.append(
            {
                "patient_id": patient_id,
                "window_index": window_index,
                "hours_since_admission": window_hours[(patient_id, window_index)],
                "count": count,
                "percent": count / len(rows) * 100 if rows else None,
            }
        )
    return {
        "total_potentially_harmful_actions": len(rows),
        "windows_with_potentially_harmful_actions": len(windows),
        "patients_with_potentially_harmful_actions": len(patient_ids),
        "unique_potentially_harmful_action_names": len(action_names),
        "matched_red_flag_rows": matched_red_flag_counter["matched_red_flag"],
        "no_matched_red_flag_rows": matched_red_flag_counter["no_matched_red_flag"],
        "top_action_names": counter_rows(action_names, len(rows), top_k),
        "top_patients": counter_rows(patient_ids, len(rows), top_k),
        "status_labels": counter_rows(status_labels, len(rows), top_k),
        "trajectory_outcomes": counter_rows(trajectory_outcomes, len(rows), top_k),
        "harmful_actions_per_window": counter_rows(rows_per_window, len(windows), top_k),
        "top_windows": top_windows,
    }


def render_markdown(summary: Mapping[str, Any], rows: list[dict[str, Any]], top_k: int) -> str:
    preview_rows = sorted(
        rows,
        key=lambda row: (
            str(row["patient_id"]),
            float(row["hours_since_admission"]) if row["hours_since_admission"] is not None else -1,
            str(row["action_name"]),
        ),
    )[:top_k]

    lines = [
        "# Potentially Harmful Action Extraction",
        "",
        f"- Total potentially harmful action evaluations: {summary['total_potentially_harmful_actions']:,}",
        f"- Windows with at least one potentially harmful action: {summary['windows_with_potentially_harmful_actions']:,}",
        f"- Patients with at least one potentially harmful action: {summary['patients_with_potentially_harmful_actions']:,}",
        f"- Unique potentially harmful action names: {summary['unique_potentially_harmful_action_names']:,}",
        f"- Rows with matched red flags: {summary['matched_red_flag_rows']:,}",
        f"- Rows without matched red flags: {summary['no_matched_red_flag_rows']:,}",
        "",
        "## Top Potentially Harmful Actions",
        "| Action | Count | Percent |",
        "|---|---:|---:|",
    ]

    for row in summary["top_action_names"]:
        lines.append(f"| {row['value']} | {row['count']:,} | {row['percent']:.1f} |")

    lines.extend(
        [
            "",
            "## Top Patients",
            "| Patient | Count | Percent |",
            "|---|---:|---:|",
        ]
    )
    for row in summary["top_patients"]:
        lines.append(f"| {row['value']} | {row['count']:,} | {row['percent']:.1f} |")

    lines.extend(
        [
            "",
            "## Status Labels",
            "| Status | Count | Percent |",
            "|---|---:|---:|",
        ]
    )
    for row in summary["status_labels"]:
        lines.append(f"| {row['value']} | {row['count']:,} | {row['percent']:.1f} |")

    lines.extend(
        [
            "",
            "## Harmful Actions Per Window",
            "| Harmful actions in window | Count | Percent |",
            "|---|---:|---:|",
        ]
    )
    for row in summary["harmful_actions_per_window"]:
        lines.append(f"| {row['value']} | {row['count']:,} | {row['percent']:.1f} |")

    lines.extend(
        [
            "",
            "## Top Windows",
            "| Patient | Hour | Window | Harmful actions | Percent of all harmful rows |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["top_windows"]:
        lines.append(
            f"| {row['patient_id']} | {row['hours_since_admission']} | {row['window_index']} | "
            f"{row['count']:,} | {row['percent']:.1f} |"
        )

    lines.extend(
        [
            "",
            f"## Preview Of First {len(preview_rows)} Rows",
            "| Patient | Hour | Window | Status | Action | Matching red flags |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for row in preview_rows:
        matching = stringify_list(row["matching_red_flags"])
        lines.append(
            f"| {row['patient_id']} | {row['hours_since_admission']} | {row['window_index']} | "
            f"{row['patient_status_label']} | {row['action_name']} | {matching or '<none>'} |"
        )

    return "\n".join(lines) + "\n"


def default_output_path(oracle_dir: Path, suffix: str) -> Path:
    return oracle_dir / f"potentially_harmful_actions{suffix}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id",
        "subject_id",
        "icu_stay_id",
        "run_id",
        "generated_at",
        "window_index",
        "source_window_index",
        "stride_source_window_index",
        "hours_since_admission",
        "window_start_time",
        "window_end_time",
        "current_window_hours",
        "num_current_events",
        "num_history_events",
        "patient_status_label",
        "patient_status_rationale",
        "active_risk_count",
        "active_risks",
        "action_id",
        "action_name",
        "action_rationale",
        "red_flag_count",
        "red_flag_actions",
        "matching_red_flags",
        "matching_red_flag_count",
        "trajectory_survived",
        "trajectory_icu_outcome",
        "oracle_predictions_path",
        "llm_calls_html_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["active_risks"] = stringify_list(row["active_risks"])
            csv_row["red_flag_actions"] = stringify_list(row["red_flag_actions"])
            csv_row["matching_red_flags"] = stringify_list(row["matching_red_flags"])
            writer.writerow(csv_row)


def main() -> None:
    args = parse_args()
    rows = extract_rows(args.oracle_dir)
    summary = build_summary(rows, args.top_k)

    output_csv = args.output_csv or default_output_path(args.oracle_dir, ".csv")
    output_json = args.output_json or default_output_path(args.oracle_dir, ".json")
    output_md = args.output_md or default_output_path(args.oracle_dir, ".md")

    write_csv(output_csv, rows)
    output_json.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_md.write_text(render_markdown(summary, rows, args.top_k), encoding="utf-8")


if __name__ == "__main__":
    main()
