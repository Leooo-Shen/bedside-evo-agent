#!/usr/bin/env python3
"""Build doctor-specific annotation packages with source-run-compatible layout."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, TextIO, Tuple

CaseKey = Tuple[int, int]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split one source oracle run directory into doctor-specific run directories "
            "using case assignments."
        )
    )
    parser.add_argument("--source-run-dir", type=Path, required=True, help="Source run directory to split.")
    parser.add_argument("--assignment-csv", type=Path, required=True, help="Case assignment CSV path.")
    parser.add_argument("--output-root-dir", type=Path, required=True, help="Output root directory.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite doctor output directories if they already exist.",
    )
    return parser.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _patient_id_from_case(case_key: CaseKey) -> str:
    return f"{int(case_key[0])}_{int(case_key[1])}"


def _normalize_doctor(value: Any) -> str:
    text = str(value).strip().upper()
    if not text:
        raise ValueError("Doctor id cannot be empty.")
    return text


def _case_from_row(row: Mapping[str, Any], *, source_name: str, row_number: int) -> CaseKey:
    if "subject_id" not in row or "icu_stay_id" not in row:
        raise ValueError(f"Missing subject_id/icu_stay_id at row {row_number} in {source_name}")
    return (_safe_int(row["subject_id"], default=-1), _safe_int(row["icu_stay_id"], default=-1))


def _read_assignment_csv(path: Path) -> Tuple[Dict[CaseKey, Tuple[str, str]], Dict[str, Set[CaseKey]]]:
    if not path.exists():
        raise FileNotFoundError(f"Assignment CSV not found: {path}")

    case_to_doctors: Dict[CaseKey, Tuple[str, str]] = {}
    doctor_to_cases: Dict[str, Set[CaseKey]] = defaultdict(set)

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_fields = {"subject_id", "icu_stay_id", "doctor_1", "doctor_2"}
        fieldnames = set(reader.fieldnames or [])
        missing_fields = required_fields - fieldnames
        if missing_fields:
            raise ValueError(f"Assignment CSV missing fields: {sorted(missing_fields)}")

        for row_number, row in enumerate(reader, start=2):
            case_key = _case_from_row(row, source_name=str(path), row_number=row_number)
            if case_key[0] < 0 or case_key[1] < 0:
                raise ValueError(f"Invalid subject_id/icu_stay_id at row {row_number} in {path}")
            doctor_1 = _normalize_doctor(row["doctor_1"])
            doctor_2 = _normalize_doctor(row["doctor_2"])
            if doctor_1 == doctor_2:
                raise ValueError(f"doctor_1 == doctor_2 at row {row_number} in {path}")
            doctors = tuple(sorted((doctor_1, doctor_2)))
            previous = case_to_doctors.get(case_key)
            if previous is not None and previous != doctors:
                raise ValueError(
                    f"Conflicting doctor assignment for case {case_key} in {path}: {previous} vs {doctors}"
                )
            case_to_doctors[case_key] = doctors
            doctor_to_cases[doctor_1].add(case_key)
            doctor_to_cases[doctor_2].add(case_key)

    if not case_to_doctors:
        raise ValueError(f"No assignments found in {path}")
    return case_to_doctors, dict(doctor_to_cases)


def _prepare_doctor_dirs(
    *,
    output_root_dir: Path,
    source_run_name: str,
    doctors: Sequence[str],
    overwrite: bool,
) -> Dict[str, Path]:
    output_dirs: Dict[str, Path] = {}
    for doctor in doctors:
        doctor_dir = output_root_dir / f"{source_run_name}_doctor_{doctor}"
        if doctor_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {doctor_dir}. Re-run with --overwrite to replace."
                )
            shutil.rmtree(doctor_dir)
        (doctor_dir / "patients").mkdir(parents=True, exist_ok=True)
        output_dirs[doctor] = doctor_dir
    return output_dirs


def _json_line_case(line: str, *, source_name: str, line_number: int) -> CaseKey:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at line {line_number} in {source_name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Line {line_number} in {source_name} is not a JSON object.")
    return _case_from_row(payload, source_name=source_name, row_number=line_number)


def _split_jsonl_by_doctor(
    *,
    source_path: Path,
    output_paths: Mapping[str, Path],
    case_to_doctors: Mapping[CaseKey, Tuple[str, str]],
) -> Dict[str, int]:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source JSONL: {source_path}")

    line_counts = {doctor: 0 for doctor in output_paths}
    seen_cases: Dict[str, Set[CaseKey]] = {doctor: set() for doctor in output_paths}
    handles: Dict[str, TextIO] = {}
    try:
        for doctor, output_path in output_paths.items():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            handles[doctor] = output_path.open("w", encoding="utf-8")
        with source_path.open("r", encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                text = line.strip()
                if not text:
                    continue
                case_key = _json_line_case(text, source_name=str(source_path), line_number=line_number)
                doctors = case_to_doctors.get(case_key)
                if doctors is None:
                    raise ValueError(f"Case {case_key} in {source_path} line {line_number} is not in assignments.")
                output_line = line if line.endswith("\n") else f"{line}\n"
                for doctor in doctors:
                    if doctor not in handles:
                        raise ValueError(f"Unexpected doctor '{doctor}' for case {case_key} in assignments.")
                    handles[doctor].write(output_line)
                    line_counts[doctor] += 1
                    seen_cases[doctor].add(case_key)
    finally:
        for handle in handles.values():
            handle.close()

    for doctor, count in line_counts.items():
        if count <= 0:
            raise ValueError(f"No rows written for doctor {doctor} from {source_path}.")
        if not seen_cases[doctor]:
            raise ValueError(f"No cases written for doctor {doctor} from {source_path}.")
    return line_counts


def _split_manifest_by_doctor(
    *,
    source_path: Path,
    output_paths: Mapping[str, Path],
    case_to_doctors: Mapping[CaseKey, Tuple[str, str]],
) -> Dict[str, List[Dict[str, str]]]:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {source_path}")

    rows_by_doctor: Dict[str, List[Dict[str, str]]] = {doctor: [] for doctor in output_paths}
    handles: Dict[str, TextIO] = {}
    writers: Dict[str, csv.DictWriter] = {}

    with source_path.open("r", encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"Manifest has no header: {source_path}")

        try:
            for doctor, output_path in output_paths.items():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                handle = output_path.open("w", encoding="utf-8", newline="")
                handles[doctor] = handle
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writers[doctor] = writer

            for row_number, row in enumerate(reader, start=2):
                case_key = _case_from_row(row, source_name=str(source_path), row_number=row_number)
                doctors = case_to_doctors.get(case_key)
                if doctors is None:
                    raise ValueError(f"Case {case_key} in {source_path} row {row_number} is not in assignments.")
                normalized_row = {field: row.get(field, "") for field in fieldnames}
                for doctor in doctors:
                    if doctor not in writers:
                        raise ValueError(f"Unexpected doctor '{doctor}' for case {case_key} in assignments.")
                    writers[doctor].writerow(normalized_row)
                    rows_by_doctor[doctor].append(dict(normalized_row))
        finally:
            for handle in handles.values():
                handle.close()

    for doctor, rows in rows_by_doctor.items():
        if not rows:
            raise ValueError(f"No manifest rows written for doctor {doctor}.")
    return rows_by_doctor


def _copy_patient_dirs(
    *,
    source_patients_dir: Path,
    target_patients_dir: Path,
    patient_ids: Sequence[str],
) -> None:
    if not source_patients_dir.exists():
        raise FileNotFoundError(f"Missing source patients directory: {source_patients_dir}")
    target_patients_dir.mkdir(parents=True, exist_ok=True)
    for patient_id in patient_ids:
        source_dir = source_patients_dir / patient_id
        target_dir = target_patients_dir / patient_id
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing source patient directory: {source_dir}")
        shutil.copytree(source_dir, target_dir)


def _load_json_or_empty(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _compute_sampling_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    patients_selected = len(rows)
    windows_selected_total = sum(_safe_int(row.get("selected_windows")) for row in rows)
    selected_valid_windows_total = sum(_safe_int(row.get("selected_valid_windows")) for row in rows)
    selected_invalid_windows_total = sum(_safe_int(row.get("selected_invalid_windows")) for row in rows)
    selected_valid_ratio_total = (
        float(selected_valid_windows_total) / float(windows_selected_total) if windows_selected_total > 0 else 0.0
    )
    patients_with_exact_8_valid_2_invalid = sum(
        1
        for row in rows
        if _safe_int(row.get("selected_valid_windows")) == 8 and _safe_int(row.get("selected_invalid_windows")) == 2
    )
    return {
        "patients_selected": int(patients_selected),
        "windows_selected_total": int(windows_selected_total),
        "selected_valid_windows_total": int(selected_valid_windows_total),
        "selected_invalid_windows_total": int(selected_invalid_windows_total),
        "selected_valid_ratio_total": float(selected_valid_ratio_total),
        "patients_with_exact_8_valid_2_invalid": int(patients_with_exact_8_valid_2_invalid),
    }


def _collect_processing_metrics(patient_root: Path, patient_ids: Sequence[str]) -> Dict[str, Any]:
    status_distribution: Counter[str] = Counter()
    total_action_evaluations = 0
    total_evaluations = 0
    windows_failed = 0
    total_llm_calls = 0
    total_tokens_used = 0
    total_pre_icu_compression_calls = 0
    total_pre_icu_compression_tokens = 0
    patients_failed = 0
    patients_with_full_window_contexts = 0

    for patient_id in patient_ids:
        patient_dir = patient_root / patient_id
        prediction_path = patient_dir / "oracle_predictions.json"
        if not prediction_path.exists():
            patients_failed += 1
            continue
        with prediction_path.open("r", encoding="utf-8") as file:
            prediction_payload = json.load(file)
        window_outputs = prediction_payload.get("window_outputs")
        if not isinstance(window_outputs, list):
            window_outputs = []
        total_evaluations += len(window_outputs)
        for window_output in window_outputs:
            if not isinstance(window_output, dict):
                continue
            if window_output.get("error"):
                windows_failed += 1
            oracle_output = window_output.get("oracle_output")
            if not isinstance(oracle_output, dict):
                continue
            patient_assessment = oracle_output.get("patient_assessment")
            if isinstance(patient_assessment, dict):
                overall = patient_assessment.get("overall")
                if isinstance(overall, dict):
                    label = str(overall.get("label", "")).strip()
                    if label:
                        status_distribution[label] += 1
            action_review = oracle_output.get("action_review")
            if isinstance(action_review, dict):
                evaluations = action_review.get("evaluations")
                if isinstance(evaluations, list):
                    total_action_evaluations += len(evaluations)

        llm_calls_path = patient_dir / "llm_calls.json"
        if llm_calls_path.exists():
            with llm_calls_path.open("r", encoding="utf-8") as file:
                llm_payload = json.load(file)
            calls = llm_payload.get("calls")
            if not isinstance(calls, list):
                calls = []
            llm_calls_raw = llm_payload.get("total_calls")
            llm_calls_value = _safe_int(llm_calls_raw, default=len(calls))
            if llm_calls_value < len(calls):
                llm_calls_value = len(calls)
            total_llm_calls += llm_calls_value
            for call in calls:
                if not isinstance(call, dict):
                    continue
                input_tokens = _safe_int(call.get("input_tokens"))
                output_tokens = _safe_int(call.get("output_tokens"))
                tokens_used = input_tokens + output_tokens
                total_tokens_used += tokens_used
                if str(call.get("step_type", "")).strip() == "oracle_pre_icu_history_compressor":
                    total_pre_icu_compression_calls += 1
                    total_pre_icu_compression_tokens += tokens_used

        if (patient_dir / "full_window_contexts.json").exists():
            patients_with_full_window_contexts += 1

    return {
        "overall_status_distribution": dict(sorted(status_distribution.items())),
        "total_action_evaluations": int(total_action_evaluations),
        "total_evaluations": int(total_evaluations),
        "windows_failed": int(windows_failed),
        "total_llm_calls": int(total_llm_calls),
        "total_tokens_used": int(total_tokens_used),
        "total_pre_icu_compression_calls": int(total_pre_icu_compression_calls),
        "total_pre_icu_compression_tokens": int(total_pre_icu_compression_tokens),
        "patients_failed": int(patients_failed),
        "patients_with_full_window_contexts": int(patients_with_full_window_contexts),
    }


def _build_processing_summary(
    *,
    source_summary: Mapping[str, Any],
    run_id: str,
    run_directory: Path,
    sparse_path: Path,
    full_path: Path,
    total_patients: int,
    total_windows_input: int,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = dict(source_summary)
    payload["run_id"] = run_id
    payload["run_directory"] = str(run_directory)
    payload["input_jsonl"] = str(sparse_path)
    payload["full_windows_jsonl"] = str(full_path)
    payload["total_patients"] = int(total_patients)
    payload["total_windows_input"] = int(total_windows_input)
    payload["total_windows_evaluated"] = int(metrics.get("total_evaluations", 0))
    payload["windows_failed"] = int(metrics.get("windows_failed", 0))
    payload["patients_processed"] = int(total_patients) - int(metrics.get("patients_failed", 0))
    payload["patients_resumed"] = 0
    payload["patients_failed"] = int(metrics.get("patients_failed", 0))
    payload["patients_with_full_window_contexts"] = int(metrics.get("patients_with_full_window_contexts", 0))
    payload["overall_status_distribution"] = dict(metrics.get("overall_status_distribution", {}))
    payload["total_action_evaluations"] = int(metrics.get("total_action_evaluations", 0))
    payload["total_evaluations"] = int(metrics.get("total_evaluations", 0))
    payload["total_llm_calls"] = int(metrics.get("total_llm_calls", 0))
    payload["total_tokens_used"] = int(metrics.get("total_tokens_used", 0))
    payload["total_pre_icu_compression_calls"] = int(metrics.get("total_pre_icu_compression_calls", 0))
    payload["total_pre_icu_compression_tokens"] = int(metrics.get("total_pre_icu_compression_tokens", 0))
    total_evaluations = int(metrics.get("total_evaluations", 0))
    total_tokens_used = int(metrics.get("total_tokens_used", 0))
    payload["avg_tokens_per_evaluation"] = (
        float(total_tokens_used) / float(total_evaluations) if total_evaluations > 0 else 0.0
    )
    return payload


def _build_tmp_progress(
    *,
    source_tmp_progress: Mapping[str, Any],
    processing_summary: Mapping[str, Any],
    run_id: str,
    run_directory: Path,
    sparse_path: Path,
    full_path: Path,
    patient_ids: Sequence[str],
) -> Dict[str, Any]:
    payload = dict(source_tmp_progress)
    payload["run_id"] = run_id
    payload["run_directory"] = str(run_directory)
    payload["updated_at"] = _utc_now_iso()
    payload["is_completed"] = True
    payload["input_jsonl"] = str(sparse_path)
    payload["full_windows_jsonl"] = str(full_path)
    payload["total_patients"] = int(len(patient_ids))
    payload["total_windows_input"] = int(processing_summary.get("total_windows_input", 0))
    payload["completed_patient_ids"] = list(patient_ids)
    payload["summary_stats"] = dict(processing_summary)
    return payload


def _build_sampling_summary(
    *,
    source_sampling_summary: Mapping[str, Any],
    source_run_dir: Path,
    output_run_dir: Path,
    manifest_rows: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    payload = dict(source_sampling_summary)
    patient_ids = [str(row.get("patient_id", "")).strip() for row in manifest_rows]
    payload["generated_at"] = _utc_now_iso()
    payload["source_run_dir"] = str(source_run_dir)
    payload["output_run_dir"] = str(output_run_dir)
    payload["selected_patient_ids"] = patient_ids
    payload["stats"] = _compute_sampling_stats(manifest_rows)
    payload["per_patient"] = list(manifest_rows)
    return payload


def _ensure_source_layout(source_run_dir: Path) -> None:
    required = [
        source_run_dir / "patients",
        source_run_dir / "selected_windows_sparse.jsonl",
        source_run_dir / "selected_windows_full.jsonl",
        source_run_dir / "sampling_manifest.csv",
        source_run_dir / "sampling_summary.json",
        source_run_dir / "processing_summary.json",
        source_run_dir / "tmp_progress.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required source paths: {missing}")


def main() -> None:
    args = _parse_args()
    source_run_dir = args.source_run_dir.resolve()
    assignment_csv = args.assignment_csv.resolve()
    output_root_dir = args.output_root_dir.resolve()

    _ensure_source_layout(source_run_dir)
    case_to_doctors, doctor_to_cases = _read_assignment_csv(assignment_csv)
    doctors = sorted(doctor_to_cases.keys())
    output_dirs = _prepare_doctor_dirs(
        output_root_dir=output_root_dir,
        source_run_name=source_run_dir.name,
        doctors=doctors,
        overwrite=bool(args.overwrite),
    )

    sparse_output_paths = {
        doctor: output_dir / "selected_windows_sparse.jsonl" for doctor, output_dir in output_dirs.items()
    }
    full_output_paths = {
        doctor: output_dir / "selected_windows_full.jsonl" for doctor, output_dir in output_dirs.items()
    }
    manifest_output_paths = {
        doctor: output_dir / "sampling_manifest.csv" for doctor, output_dir in output_dirs.items()
    }

    sparse_counts = _split_jsonl_by_doctor(
        source_path=source_run_dir / "selected_windows_sparse.jsonl",
        output_paths=sparse_output_paths,
        case_to_doctors=case_to_doctors,
    )
    full_counts = _split_jsonl_by_doctor(
        source_path=source_run_dir / "selected_windows_full.jsonl",
        output_paths=full_output_paths,
        case_to_doctors=case_to_doctors,
    )
    manifest_rows_by_doctor = _split_manifest_by_doctor(
        source_path=source_run_dir / "sampling_manifest.csv",
        output_paths=manifest_output_paths,
        case_to_doctors=case_to_doctors,
    )

    source_sampling_summary = _load_json_or_empty(source_run_dir / "sampling_summary.json")
    source_processing_summary = _load_json_or_empty(source_run_dir / "processing_summary.json")
    source_tmp_progress = _load_json_or_empty(source_run_dir / "tmp_progress.json")

    for doctor in doctors:
        output_dir = output_dirs[doctor]
        patient_ids = [str(row["patient_id"]).strip() for row in manifest_rows_by_doctor[doctor]]
        if len(patient_ids) != len(doctor_to_cases[doctor]):
            raise ValueError(
                f"Doctor {doctor} has {len(doctor_to_cases[doctor])} assigned cases but {len(patient_ids)} manifest rows."
            )
        _copy_patient_dirs(
            source_patients_dir=source_run_dir / "patients",
            target_patients_dir=output_dir / "patients",
            patient_ids=patient_ids,
        )

        metrics = _collect_processing_metrics(output_dir / "patients", patient_ids)
        run_id = f"{source_run_dir.name}_doctor_{doctor}"
        processing_summary = _build_processing_summary(
            source_summary=source_processing_summary,
            run_id=run_id,
            run_directory=output_dir,
            sparse_path=output_dir / "selected_windows_sparse.jsonl",
            full_path=output_dir / "selected_windows_full.jsonl",
            total_patients=len(patient_ids),
            total_windows_input=sparse_counts[doctor],
            metrics=metrics,
        )
        tmp_progress = _build_tmp_progress(
            source_tmp_progress=source_tmp_progress,
            processing_summary=processing_summary,
            run_id=run_id,
            run_directory=output_dir,
            sparse_path=output_dir / "selected_windows_sparse.jsonl",
            full_path=output_dir / "selected_windows_full.jsonl",
            patient_ids=patient_ids,
        )
        sampling_summary = _build_sampling_summary(
            source_sampling_summary=source_sampling_summary,
            source_run_dir=source_run_dir,
            output_run_dir=output_dir,
            manifest_rows=manifest_rows_by_doctor[doctor],
        )

        _write_json(output_dir / "processing_summary.json", processing_summary)
        _write_json(output_dir / "tmp_progress.json", tmp_progress)
        _write_json(output_dir / "sampling_summary.json", sampling_summary)

        print(f"[Doctor {doctor}]")
        print(f"  output_dir: {output_dir}")
        print(f"  assigned_cases: {len(patient_ids)}")
        print(f"  selected_windows_sparse: {sparse_counts[doctor]}")
        print(f"  selected_windows_full: {full_counts[doctor]}")
        print(f"  status_distribution: {json.dumps(processing_summary['overall_status_distribution'])}")


if __name__ == "__main__":
    main()
