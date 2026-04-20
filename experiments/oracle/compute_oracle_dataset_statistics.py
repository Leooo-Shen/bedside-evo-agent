"""Compute descriptive statistics for Oracle-labeled ICU trajectory datasets."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics from patients/*/oracle_predictions.json."
    )
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--window-size-hours", type=float)
    parser.add_argument("--step-size-hours", type=float)
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


def normalized_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value.strip())
    return text or None


def nested_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def nested_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (index - lower)


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / (len(clean) - 1) if len(clean) > 1 else 0.0
    return {
        "n": len(clean),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": clean[0],
        "p25": percentile(clean, 0.25),
        "median": percentile(clean, 0.5),
        "p75": percentile(clean, 0.75),
        "max": clean[-1],
    }


def counter_rows(counter: Counter, total: int | None = None, top_k: int | None = None) -> list[dict[str, Any]]:
    denominator = total if total is not None else sum(counter.values())
    rows = []
    for value, count in counter.most_common(top_k):
        row = {"value": value, "count": count}
        if denominator:
            row["percent"] = count / denominator * 100
        rows.append(row)
    return rows


def counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): value for key, value in counter.items()}


def file_id(path: Path) -> str:
    return path.parent.name


def load_completed_ids(oracle_dir: Path) -> set[str]:
    path = oracle_dir / "completed_icu_stay_ids.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def collect_statistics(
    oracle_dir: Path,
    window_size_hours: float | None,
    step_size_hours: float | None,
    top_k: int,
) -> dict[str, Any]:
    paths = prediction_paths(oracle_dir)
    completed_ids = load_completed_ids(oracle_dir)

    patient_ids = set()
    icu_stay_ids = set()
    patient_dir_ids = set()
    status_labels = Counter()
    action_labels = Counter()
    action_names = Counter()
    red_flag_actions = Counter()
    active_problems = Counter()
    event_codes = Counter()
    event_code_specifics = Counter()
    icu_outcomes = Counter()
    survival_labels = Counter()
    gender_labels = Counter()
    run_ids = Counter()
    current_window_hours = Counter()
    inferred_step_hours = Counter()
    stride_filter_labels = Counter()
    generated_at_values = []
    windows_per_stay = []
    stay_hours = []
    age_years = []
    current_events_per_window = []
    history_events_per_window = []
    hours_since_admission = []
    actions_per_window = []
    red_flags_per_window = []
    active_problems_per_window = []
    action_label_sets_per_window = Counter()
    missing_status_windows = 0
    missing_active_problem_windows = 0
    windows_with_action_labels = 0
    windows_with_red_flags = 0
    windows_with_active_problems = 0
    total_windows = 0
    total_action_evaluations = 0
    total_red_flags = 0
    total_active_problems = 0
    total_current_events = 0

    for path in paths:
        payload = read_json(path)
        patient_dir_ids.add(file_id(path))
        run_ids.update([str(payload.get("run_id"))])
        stride_filter_labels.update(["with_stride_filter" if payload.get("stride_filter") else "without_stride_filter"])
        generated_at = parse_timestamp(payload.get("generated_at"))
        if generated_at:
            generated_at_values.append(generated_at.isoformat())

        metadata = nested_mapping(payload.get("trajectory_metadata"))
        subject_id = metadata.get("subject_id", payload.get("subject_id"))
        icu_stay_id = metadata.get("icu_stay_id", payload.get("icu_stay_id"))
        if subject_id is not None:
            patient_ids.add(str(subject_id))
        if icu_stay_id is not None:
            icu_stay_ids.add(str(icu_stay_id))
        outcome = metadata.get("icu_outcome")
        if outcome is not None:
            icu_outcomes.update([str(outcome)])
        survived = metadata.get("survived")
        if isinstance(survived, bool):
            survival_labels.update(["survived" if survived else "died"])
        gender = normalized_text(metadata.get("gender"))
        if gender:
            gender_labels.update([gender])
        age = metadata.get("age")
        if isinstance(age, (int, float)):
            age_years.append(float(age))
        duration = metadata.get("total_icu_stay_hours", metadata.get("icu_duration_hours"))
        if isinstance(duration, (int, float)):
            stay_hours.append(float(duration))

        windows = nested_list(payload.get("window_outputs"))
        windows_per_stay.append(len(windows))
        previous_hour = None

        for window in windows:
            window_mapping = nested_mapping(window)
            metadata = nested_mapping(window_mapping.get("window_metadata"))
            oracle_output = nested_mapping(window_mapping.get("oracle_output"))
            assessment = nested_mapping(oracle_output.get("patient_assessment"))
            overall = nested_mapping(assessment.get("overall"))
            action_review = nested_mapping(oracle_output.get("action_review"))
            evaluations = nested_list(action_review.get("evaluations"))
            red_flags = nested_list(action_review.get("red_flags"))
            risks = nested_list(assessment.get("active_risks"))
            current_events = nested_list(window_mapping.get("raw_current_events"))

            total_windows += 1
            total_current_events += len(current_events)
            current_events_per_window.append(len(current_events))
            actions_per_window.append(len(evaluations))
            red_flags_per_window.append(len(red_flags))
            active_problems_per_window.append(len(risks))
            total_action_evaluations += len(evaluations)
            total_red_flags += len(red_flags)
            total_active_problems += len(risks)

            current_window = metadata.get("current_window_hours")
            if isinstance(current_window, (int, float)):
                current_window_hours.update([float(current_window)])
            history_events = metadata.get("num_history_events")
            if isinstance(history_events, (int, float)):
                history_events_per_window.append(float(history_events))
            current_events_count = metadata.get("num_current_events")
            if isinstance(current_events_count, (int, float)):
                current_events_per_window[-1] = float(current_events_count)
            hour = metadata.get("hours_since_admission")
            if isinstance(hour, (int, float)):
                hour = float(hour)
                hours_since_admission.append(hour)
                if previous_hour is not None:
                    inferred_step_hours.update([round(hour - previous_hour, 3)])
                previous_hour = hour

            status = normalized_text(overall.get("label"))
            if status:
                status_labels.update([status])
            else:
                missing_status_windows += 1

            if evaluations:
                windows_with_action_labels += 1
            labels_in_window = Counter()
            for evaluation in evaluations:
                evaluation_mapping = nested_mapping(evaluation)
                label = normalized_text(evaluation_mapping.get("label"))
                if label:
                    action_labels.update([label])
                    labels_in_window.update([label])
                action_name = normalized_text(evaluation_mapping.get("action_name"))
                if action_name:
                    action_names.update([action_name])
            action_label_sets_per_window.update(["; ".join(sorted(labels_in_window)) if labels_in_window else "no_action_labels"])

            if red_flags:
                windows_with_red_flags += 1
            for red_flag in red_flags:
                action = normalized_text(nested_mapping(red_flag).get("contraindicated_action"))
                if action:
                    red_flag_actions.update([action])

            if risks:
                windows_with_active_problems += 1
            else:
                missing_active_problem_windows += 1
            for risk in risks:
                problem = normalized_text(nested_mapping(risk).get("risk_name"))
                if problem:
                    active_problems.update([problem])

            for event in current_events:
                event_mapping = nested_mapping(event)
                code = normalized_text(event_mapping.get("code"))
                if code:
                    event_codes.update([code])
                code_specifics = normalized_text(event_mapping.get("code_specifics"))
                if code and code_specifics:
                    event_code_specifics.update([f"{code}: {code_specifics}"])

    completed_present_ids = patient_dir_ids & completed_ids
    generated_range = {
        "earliest": min(generated_at_values) if generated_at_values else None,
        "latest": max(generated_at_values) if generated_at_values else None,
    }
    processing_summary_path = oracle_dir / "processing_summary.json"
    processing_summary = read_json(processing_summary_path) if processing_summary_path.exists() else None
    stride_manifest_path = oracle_dir / "stride_filter_manifest.json"
    stride_manifest = read_json(stride_manifest_path) if stride_manifest_path.exists() else None

    return {
        "oracle_dir": str(oracle_dir),
        "declared_window_size_hours": window_size_hours,
        "declared_step_size_hours": step_size_hours,
        "processing_summary": processing_summary,
        "stride_filter_manifest": stride_manifest,
        "cohort": {
            "icu_stays": len(icu_stay_ids),
            "patient_directories": len(patient_dir_ids),
            "unique_subjects": len(patient_ids),
            "completed_icu_stay_ids": len(completed_ids),
            "completed_ids_present": len(completed_present_ids),
            "patient_dirs_not_in_completed_ids": len(patient_dir_ids - completed_ids) if completed_ids else None,
            "stride_filter_files": counter_dict(stride_filter_labels),
            "run_ids": counter_rows(run_ids),
            "generated_at": generated_range,
        },
        "windows": {
            "total": total_windows,
            "with_status_label": total_windows - missing_status_windows,
            "missing_status_label": missing_status_windows,
            "with_action_labels": windows_with_action_labels,
            "with_red_flags": windows_with_red_flags,
            "with_active_problems": windows_with_active_problems,
            "missing_active_problems": missing_active_problem_windows,
            "windows_per_stay": numeric_summary(windows_per_stay),
            "hours_since_admission": numeric_summary(hours_since_admission),
            "current_window_hours": counter_rows(current_window_hours),
            "inferred_step_hours": counter_rows(inferred_step_hours),
        },
        "labels": {
            "patient_status": counter_rows(status_labels, total_windows),
            "icu_outcomes": counter_rows(icu_outcomes),
            "survival": counter_rows(survival_labels),
            "gender": counter_rows(gender_labels),
            "action_labels": counter_rows(action_labels),
            "action_label_sets_per_window": counter_rows(action_label_sets_per_window, total_windows, top_k),
            "actions_per_window": counter_rows(Counter(actions_per_window), total_windows),
            "red_flags_per_window": counter_rows(Counter(red_flags_per_window), total_windows),
            "active_problems_per_window": counter_rows(Counter(active_problems_per_window), total_windows),
        },
        "clinical_content": {
            "total_action_evaluations": total_action_evaluations,
            "total_red_flags": total_red_flags,
            "total_active_problems": total_active_problems,
            "unique_action_names": len(action_names),
            "unique_red_flag_actions": len(red_flag_actions),
            "unique_active_problems": len(active_problems),
            "top_action_names": counter_rows(action_names, total_action_evaluations, top_k),
            "top_red_flag_actions": counter_rows(red_flag_actions, total_red_flags, top_k),
            "top_active_problems": counter_rows(active_problems, total_active_problems, top_k),
        },
        "events": {
            "total_current_events": total_current_events,
            "current_events_per_window": numeric_summary(current_events_per_window),
            "history_events_per_window": numeric_summary(history_events_per_window),
            "top_event_codes": counter_rows(event_codes, total_current_events, top_k),
            "top_event_code_specifics": counter_rows(event_code_specifics, total_current_events, top_k),
        },
        "demographics": {
            "age_years": numeric_summary(age_years),
            "icu_stay_hours": numeric_summary(stay_hours),
        },
    }


def fmt_number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def fmt_percent(row: Mapping[str, Any]) -> str:
    percent = row.get("percent")
    return "" if percent is None else f" ({percent:.1f}%)"


def table(title: str, rows: list[Mapping[str, Any]], value_label: str = "Value") -> list[str]:
    lines = [f"### {title}", f"| {value_label} | Count | Percent |", "|---|---:|---:|"]
    for row in rows:
        percent = row.get("percent")
        lines.append(
            f"| {row.get('value')} | {fmt_number(row.get('count'), 0)} | "
            f"{fmt_number(percent, 1) if percent is not None else 'NA'} |"
        )
    return lines


def summary_line(name: str, summary: Mapping[str, Any], unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    return (
        f"- {name}: mean {fmt_number(summary.get('mean'))}{suffix}, "
        f"median {fmt_number(summary.get('median'))}{suffix}, "
        f"IQR {fmt_number(summary.get('p25'))}-{fmt_number(summary.get('p75'))}{suffix}, "
        f"range {fmt_number(summary.get('min'))}-{fmt_number(summary.get('max'))}{suffix}"
    )


def render_markdown(stats: Mapping[str, Any]) -> str:
    cohort = nested_mapping(stats.get("cohort"))
    windows = nested_mapping(stats.get("windows"))
    labels = nested_mapping(stats.get("labels"))
    clinical = nested_mapping(stats.get("clinical_content"))
    events = nested_mapping(stats.get("events"))
    demographics = nested_mapping(stats.get("demographics"))
    stride_manifest = nested_mapping(stats.get("stride_filter_manifest"))

    lines = [
        "# Oracle Dataset Statistics",
        "",
        f"- Oracle directory: `{stats.get('oracle_dir')}`",
        f"- Declared current window size: {fmt_number(stats.get('declared_window_size_hours'))} hours",
        f"- Declared window step: {fmt_number(stats.get('declared_step_size_hours'))} hours",
        f"- ICU stays: {fmt_number(cohort.get('icu_stays'), 0)}",
        f"- Unique subjects: {fmt_number(cohort.get('unique_subjects'), 0)}",
        f"- Total labeled windows: {fmt_number(windows.get('total'), 0)}",
        f"- Completed ID file entries present: {fmt_number(cohort.get('completed_ids_present'), 0)} / {fmt_number(cohort.get('completed_icu_stay_ids'), 0)}",
        f"- Patient directories outside completed ID file: {fmt_number(cohort.get('patient_dirs_not_in_completed_ids'), 0)}",
        f"- Files with stride filter metadata: {cohort.get('stride_filter_files')}",
    ]

    if stride_manifest:
        lines.append(
            f"- Stride manifest: {fmt_number(stride_manifest.get('windows_before'), 0)} source windows "
            f"to {fmt_number(stride_manifest.get('windows_after'), 0)} windows across "
            f"{fmt_number(stride_manifest.get('patients_processed'), 0)} patients"
        )

    lines.extend(
        [
            "",
            "## Cohort Summaries",
            summary_line("Windows per ICU stay", nested_mapping(windows.get("windows_per_stay"))),
            summary_line("Age", nested_mapping(demographics.get("age_years")), "years"),
            summary_line("ICU stay length", nested_mapping(demographics.get("icu_stay_hours")), "hours"),
            summary_line("Hours since admission across windows", nested_mapping(windows.get("hours_since_admission")), "hours"),
            summary_line("Current events per window", nested_mapping(events.get("current_events_per_window"))),
            summary_line("History events per window", nested_mapping(events.get("history_events_per_window"))),
            "",
            "## Label Coverage",
            f"- Windows with patient status labels: {fmt_number(windows.get('with_status_label'), 0)} / {fmt_number(windows.get('total'), 0)}",
            f"- Windows with at least one evaluated action: {fmt_number(windows.get('with_action_labels'), 0)} / {fmt_number(windows.get('total'), 0)}",
            f"- Windows with at least one red flag action: {fmt_number(windows.get('with_red_flags'), 0)} / {fmt_number(windows.get('total'), 0)}",
            f"- Windows with at least one active problem: {fmt_number(windows.get('with_active_problems'), 0)} / {fmt_number(windows.get('total'), 0)}",
            f"- Total action evaluations: {fmt_number(clinical.get('total_action_evaluations'), 0)}",
            f"- Total red flag actions: {fmt_number(clinical.get('total_red_flags'), 0)}",
            f"- Total active problem mentions: {fmt_number(clinical.get('total_active_problems'), 0)}",
        ]
    )

    sections = [
        ("Patient Status Labels", labels.get("patient_status"), "Status"),
        ("ICU Outcomes", labels.get("icu_outcomes"), "Outcome"),
        ("Survival", labels.get("survival"), "Survival"),
        ("Gender", labels.get("gender"), "Gender"),
        ("Action Labels", labels.get("action_labels"), "Action label"),
        ("Action Label Sets Per Window", labels.get("action_label_sets_per_window"), "Label set"),
        ("Actions Per Window", labels.get("actions_per_window"), "Actions"),
        ("Red Flags Per Window", labels.get("red_flags_per_window"), "Red flags"),
        ("Active Problems Per Window", labels.get("active_problems_per_window"), "Active problems"),
        ("Top Action Names", clinical.get("top_action_names"), "Action"),
        ("Top Red Flag Actions", clinical.get("top_red_flag_actions"), "Contraindicated action"),
        ("Top Active Problems", clinical.get("top_active_problems"), "Active problem"),
        ("Top Event Codes", events.get("top_event_codes"), "Event code"),
    ]

    for title, rows, value_label in sections:
        lines.append("")
        lines.extend(table(title, list(rows or []), value_label))

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    stats = collect_statistics(
        oracle_dir=args.oracle_dir,
        window_size_hours=args.window_size_hours,
        step_size_hours=args.step_size_hours,
        top_k=args.top_k,
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

    markdown = render_markdown(stats)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
