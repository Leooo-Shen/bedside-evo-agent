"""Evaluate active-problem predictions against GT active problems."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.llms import LLMClient
from prompts.action_matcher_prompts import get_active_problem_matcher_prompt
from utils.json_parse import parse_json_dict_best_effort

KNOWN_MODES = ("memory", "full_history_events", "local_events_only")
PREDICTION_FILENAMES = (
    "active_problem_predictions.json",
    "recommendation_predictions.json",
    "patient_status_predictions.json",
)
GT_FILENAMES = (
    "oracle_predictions.json",
    "active_problem_ground_truth.json",
    "active_problem_gt.json",
    "active_problems.json",
)

LLM_PROVIDER = "google"
LLM_MODEL = "qwen/qwen3-235b-a22b-instruct-2507-maas"
LLM_MAX_TOKENS = 12800
ORACLE_INTERVAL_DECIMALS = 6

_THREAD_LOCAL = threading.local()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _normalize_token_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return 0
    return 0


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json_value(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _json_dump(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_safe_json_value(payload), f, indent=2, ensure_ascii=False)


def _parse_json_list_cell(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list cell, got: {text[:100]}")
    return parsed


def _find_first_existing_file(directory: Path, filenames: Sequence[str]) -> Optional[Path]:
    for filename in filenames:
        candidate = directory / str(filename)
        if candidate.exists():
            return candidate
    return None


def _has_prediction_files(path: Path) -> bool:
    patients_dir = path / "patients"
    if not patients_dir.exists() or not patients_dir.is_dir():
        return False
    for patient_dir in patients_dir.iterdir():
        if not patient_dir.is_dir():
            continue
        if _find_first_existing_file(patient_dir, PREDICTION_FILENAMES) is not None:
            return True
    return False


def _ensure_prediction_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {path}")
    if _has_prediction_files(path):
        return path
    if _discover_mode_dirs(path):
        return path
    nested = path / "recommendation_experiment"
    if nested.exists() and nested.is_dir() and _has_prediction_files(nested):
        return nested
    if nested.exists() and nested.is_dir() and _discover_mode_dirs(nested):
        return nested
    raise FileNotFoundError(
        f"No prediction files found under {path}. "
        "Expected patients/*/(active_problem_predictions.json|recommendation_predictions.json|patient_status_predictions.json)."
    )


def _discover_mode_dirs(path: Path) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    for mode in KNOWN_MODES:
        candidate = path / mode
        if candidate.exists() and candidate.is_dir() and _has_prediction_files(candidate):
            discovered[str(mode)] = candidate
    return discovered


def _has_eval_outputs(path: Path) -> bool:
    return (path / "window_level_windows.csv").exists()


def _discover_eval_mode_dirs(path: Path) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    for mode in KNOWN_MODES:
        candidate = path / mode
        if candidate.exists() and candidate.is_dir() and _has_eval_outputs(candidate):
            discovered[str(mode)] = candidate
    return discovered


def _ensure_reuse_eval_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Reuse eval directory not found: {path}")
    if _has_eval_outputs(path) or _discover_eval_mode_dirs(path):
        return path
    raise FileNotFoundError(
        f"No window_level_windows.csv found under {path}. "
        "Expected either <dir>/window_level_windows.csv or <dir>/<mode>/window_level_windows.csv."
    )


def _ensure_gt_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"GT directory not found: {path}")
    patients_dir = path / "patients"
    if not patients_dir.exists() or not patients_dir.is_dir():
        raise FileNotFoundError(f"GT patients directory not found: {patients_dir}")
    found = False
    for patient_dir in patients_dir.iterdir():
        if not patient_dir.is_dir():
            continue
        if _find_first_existing_file(patient_dir, GT_FILENAMES) is not None:
            found = True
            break
    if not found:
        raise FileNotFoundError(f"No GT files found under {patients_dir}")
    return path


def _default_output_root(prediction_dir: Path, gt_dir: Path) -> Path:
    return Path("evaluation_results") / "active_problem_eval" / prediction_dir.name / gt_dir.name


def _problem_text_from_item(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, Mapping):
        keys = (
            "problem",
            "problem_name",
            "name",
            "diagnosis",
            "risk_name",
            "label",
            "title",
            "description",
            "text",
        )
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _normalize_problem_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    normalized: List[str] = []
    seen = set()
    for item in value:
        text = _problem_text_from_item(item)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text)
    return normalized


def _extract_predicted_problems_from_row(row: Mapping[str, Any]) -> List[str]:
    candidates: List[Any] = [
        row.get("predicted_active_problems"),
        row.get("active_problems"),
        row.get("active_problem_list"),
        row.get("problems"),
        row.get("active_risks"),
    ]
    parsed_prediction = row.get("parsed_prediction")
    if isinstance(parsed_prediction, Mapping):
        candidates.extend(
            [
                parsed_prediction.get("predicted_active_problems"),
                parsed_prediction.get("active_problems"),
                parsed_prediction.get("active_problem_list"),
                parsed_prediction.get("problems"),
                parsed_prediction.get("active_risks"),
            ]
        )
        patient_assessment = parsed_prediction.get("patient_assessment")
        if isinstance(patient_assessment, Mapping):
            candidates.extend(
                [
                    patient_assessment.get("active_problems"),
                    patient_assessment.get("active_risks"),
                ]
            )
    patient_assessment = row.get("patient_assessment")
    if isinstance(patient_assessment, Mapping):
        candidates.extend(
            [
                patient_assessment.get("active_problems"),
                patient_assessment.get("active_risks"),
            ]
        )

    for candidate in candidates:
        normalized = _normalize_problem_list(candidate)
        if normalized:
            return normalized
    return []


def _extract_gt_problems_from_window_row(row: Mapping[str, Any]) -> List[str]:
    merged: List[str] = []
    seen = set()

    def _extend_from(value: Any) -> None:
        normalized = _normalize_problem_list(value)
        for item in normalized:
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(item)

    _extend_from(row.get("active_problems"))
    _extend_from(row.get("gt_active_problems"))
    _extend_from(row.get("oracle_active_problems"))
    _extend_from(row.get("active_risks"))

    oracle_output = row.get("oracle_output")
    if isinstance(oracle_output, Mapping):
        _extend_from(oracle_output.get("active_problems"))
        _extend_from(oracle_output.get("active_risks"))
        patient_assessment = oracle_output.get("patient_assessment")
        if isinstance(patient_assessment, Mapping):
            _extend_from(patient_assessment.get("active_problems"))
            _extend_from(patient_assessment.get("active_risks"))

    patient_assessment = row.get("patient_assessment")
    if isinstance(patient_assessment, Mapping):
        _extend_from(patient_assessment.get("active_problems"))
        _extend_from(patient_assessment.get("active_risks"))

    return merged


def _extract_window_rows_from_prediction_payload(payload: Mapping[str, Any], source_path: Path) -> List[Dict[str, Any]]:
    list_keys = ("active_problem_predictions", "recommendation_predictions", "status_predictions", "windows")
    rows_raw: Any = None
    for key in list_keys:
        value = payload.get(key)
        if isinstance(value, list):
            rows_raw = value
            break
    if not isinstance(rows_raw, list):
        raise ValueError(f"Missing prediction windows list in {source_path}")

    rows: List[Dict[str, Any]] = []
    for row in rows_raw:
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid prediction row in {source_path}")
        window_raw = row.get("window_index")
        if window_raw is None:
            source_window_raw = row.get("source_window_index")
            if source_window_raw is not None:
                window_raw = source_window_raw
        if window_raw is None:
            raise ValueError(f"Missing window_index in prediction row in {source_path}")
        window_index = int(window_raw)
        if window_index < 0:
            raise ValueError(f"Negative window_index={window_index} in {source_path}")
        normalized_row = dict(row)
        normalized_row["window_index"] = int(window_index)
        rows.append(normalized_row)
    return rows


def _extract_icu_stay_id(payload: Mapping[str, Any], source_path: Path) -> int:
    icu_stay_id_raw = payload.get("icu_stay_id")
    if icu_stay_id_raw is None:
        raise ValueError(f"Missing icu_stay_id in {source_path}")
    icu_stay_id = int(icu_stay_id_raw)
    if icu_stay_id <= 0:
        raise ValueError(f"Invalid icu_stay_id={icu_stay_id} in {source_path}")
    return icu_stay_id


def _interval_key(start_hour: float, end_hour: float) -> Tuple[float, float]:
    return (round(float(start_hour), ORACLE_INTERVAL_DECIMALS), round(float(end_hour), ORACLE_INTERVAL_DECIMALS))


def _interval_marker_id(interval_key: Tuple[float, float]) -> str:
    return f"{interval_key[0]:.{ORACLE_INTERVAL_DECIMALS}f}|{interval_key[1]:.{ORACLE_INTERVAL_DECIMALS}f}"


def _parse_window_interval(
    *,
    hours_since_admission: Any,
    current_window_hours: Any,
    source_path: Path,
    context_label: str,
) -> Tuple[float, float]:
    try:
        start_hour = float(hours_since_admission)
        duration_hours = float(current_window_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid hour fields in {source_path} ({context_label}): "
            f"hours_since_admission={hours_since_admission!r}, current_window_hours={current_window_hours!r}"
        ) from exc
    if duration_hours <= 0:
        raise ValueError(
            f"current_window_hours must be > 0 in {source_path} ({context_label}), got {duration_hours!r}"
        )
    return start_hour, start_hour + duration_hours


def _resolve_gt_window_interval(row: Mapping[str, Any], gt_file: Path) -> Tuple[float, float]:
    metadata = row.get("window_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Missing window_metadata in row in {gt_file}")
    start_hour, end_hour = _parse_window_interval(
        hours_since_admission=metadata.get("hours_since_admission"),
        current_window_hours=metadata.get("current_window_hours"),
        source_path=gt_file,
        context_label="gt_window",
    )
    return _interval_key(start_hour, end_hour)


def _resolve_prediction_interval(item: Mapping[str, Any], source_path: Path) -> Tuple[float, float]:
    marker = item.get("oracle_window_marker")
    if isinstance(marker, Mapping):
        start_raw = marker.get("interval_start_hour")
        end_raw = marker.get("interval_end_hour")
        if start_raw is not None and end_raw is not None:
            try:
                return _interval_key(float(start_raw), float(end_raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid oracle_window_marker in {source_path}: {marker}") from exc

    start_raw = item.get("window_start_hour")
    end_raw = item.get("window_end_hour")
    if start_raw is not None and end_raw is not None:
        try:
            return _interval_key(float(start_raw), float(end_raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid window_start_hour/window_end_hour in prediction row in {source_path}: "
                f"{start_raw!r}, {end_raw!r}"
            ) from exc

    hours_raw = item.get("hours_since_admission")
    duration_raw = item.get("window_duration_hours")
    if hours_raw is not None and duration_raw is not None:
        start_hour, end_hour = _parse_window_interval(
            hours_since_admission=hours_raw,
            current_window_hours=duration_raw,
            source_path=source_path,
            context_label="prediction_window",
        )
        return _interval_key(start_hour, end_hour)

    raise ValueError(
        f"Missing oracle_window_marker/window_start_hour/window_end_hour/window_duration_hours in prediction row in "
        f"{source_path}"
    )


def _load_gt_window_rows(gt_file: Path) -> List[Mapping[str, Any]]:
    payload = _load_json(gt_file)
    window_rows: List[Mapping[str, Any]] = []

    window_outputs = payload.get("window_outputs")
    if isinstance(window_outputs, list):
        window_rows = [row for row in window_outputs if isinstance(row, Mapping)]
    else:
        for key in ("windows", "active_problem_windows", "problem_windows", "items", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                window_rows = [row for row in value if isinstance(row, Mapping)]
                break

    if not window_rows:
        raise ValueError(f"Missing GT window rows in {gt_file}")

    return window_rows


def _load_gt_problems_by_interval(gt_file: Path) -> Dict[Tuple[float, float], List[str]]:
    window_rows = _load_gt_window_rows(gt_file)

    by_interval: Dict[Tuple[float, float], List[str]] = {}
    for row in window_rows:
        interval_key = _resolve_gt_window_interval(row=row, gt_file=gt_file)
        if interval_key in by_interval:
            raise ValueError(f"Duplicate GT interval={interval_key} in {gt_file}")
        problems = _extract_gt_problems_from_window_row(row)
        by_interval[interval_key] = problems

    return by_interval


def _load_gt_window_intervals(gt_file: Path) -> Set[Tuple[float, float]]:
    window_rows = _load_gt_window_rows(gt_file)
    intervals: Set[Tuple[float, float]] = set()
    for row in window_rows:
        intervals.add(_resolve_gt_window_interval(row=row, gt_file=gt_file))
    return intervals


def _match_prediction_gt_intervals_by_stay(
    *,
    prediction_windows_by_stay: Dict[int, Set[Tuple[float, float]]],
    gt_windows_by_stay: Dict[int, Set[Tuple[float, float]]],
) -> Dict[str, Any]:
    matched_windows_by_stay: Dict[int, Set[Tuple[float, float]]] = {}
    skipped_prediction_stays_missing_gt = 0
    skipped_prediction_windows_missing_gt = 0
    matched_windows = 0

    for icu_stay_id, prediction_intervals in prediction_windows_by_stay.items():
        gt_intervals = gt_windows_by_stay.get(int(icu_stay_id))
        if gt_intervals is None:
            skipped_prediction_stays_missing_gt += 1
            skipped_prediction_windows_missing_gt += int(len(prediction_intervals))
            continue
        shared_intervals = prediction_intervals.intersection(gt_intervals)
        skipped_prediction_windows_missing_gt += int(len(prediction_intervals) - len(shared_intervals))
        if shared_intervals:
            matched_windows_by_stay[int(icu_stay_id)] = set(shared_intervals)
            matched_windows += int(len(shared_intervals))

    return {
        "matched_windows_by_stay": matched_windows_by_stay,
        "prediction_stays": int(len(prediction_windows_by_stay)),
        "gt_stays": int(len(gt_windows_by_stay)),
        "matched_stays": int(len(matched_windows_by_stay)),
        "prediction_windows": int(sum(len(v) for v in prediction_windows_by_stay.values())),
        "gt_windows": int(sum(len(v) for v in gt_windows_by_stay.values())),
        "matched_windows": int(matched_windows),
        "skipped_prediction_stays_missing_gt": int(skipped_prediction_stays_missing_gt),
        "skipped_prediction_windows_missing_gt": int(skipped_prediction_windows_missing_gt),
    }


def _build_problem_matching_prompt(
    predicted_problems: Sequence[str],
    gt_problems: Sequence[str],
) -> str:
    predicted_payload = [{"pred_idx": int(i), "problem": str(text)} for i, text in enumerate(predicted_problems)]
    gt_payload = [{"gt_index": int(i), "problem": str(text)} for i, text in enumerate(gt_problems)]

    prompt_template = get_active_problem_matcher_prompt()
    return (
        prompt_template.replace("{predicted_problems}", json.dumps(predicted_payload, ensure_ascii=False, indent=2))
        .replace("{oracle_active_problems}", json.dumps(gt_payload, ensure_ascii=False, indent=2))
    )


def _extract_problem_match_pairs(
    parsed_payload: Mapping[str, Any],
    *,
    num_predictions: int,
    num_ground_truth: int,
) -> List[Dict[str, int]]:
    problem_matches = parsed_payload.get("problem_matches")
    if not isinstance(problem_matches, list):
        matches = parsed_payload.get("matches")
        if isinstance(matches, list):
            problem_matches = matches
    if not isinstance(problem_matches, list):
        return []

    pairs: List[Dict[str, int]] = []
    seen_predictions = set()
    for row in problem_matches:
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid problem_matches row: {row}")
        pred_raw = row.get("pred_idx")
        gt_indices_raw = row.get("gt_indices")
        try:
            pred_idx = int(pred_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid pred_idx in problem_matches row: {row}") from exc
        if pred_idx < 0 or pred_idx >= num_predictions:
            raise ValueError(f"pred_idx out of range: {pred_idx}")
        if pred_idx in seen_predictions:
            continue
        seen_predictions.add(pred_idx)

        if not isinstance(gt_indices_raw, list):
            raise ValueError(f"gt_indices must be list in row: {row}")
        parsed_gt_indices: List[int] = []
        for value in gt_indices_raw:
            try:
                gt_idx = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid gt index in gt_indices: {value}") from exc
            if gt_idx < 0 or gt_idx >= num_ground_truth:
                raise ValueError(f"gt index out of range: {gt_idx}")
            parsed_gt_indices.append(gt_idx)
        if not parsed_gt_indices:
            continue
        pairs.append({"prediction_index": int(pred_idx), "gt_index": int(parsed_gt_indices[0])})

    return pairs


def _match_window_with_llm(
    llm_client: LLMClient,
    *,
    predicted_problems: Sequence[str],
    gt_problems: Sequence[str],
    patient_id: str,
    window_index: int,
) -> Dict[str, Any]:
    if not predicted_problems or not gt_problems:
        return {
            "matched_pairs": [],
            "matched_prediction_indices": [],
            "matched_gt_indices": [],
            "raw_response": "",
            "input_tokens": 0,
            "output_tokens": 0,
        }

    prompt = _build_problem_matching_prompt(predicted_problems=predicted_problems, gt_problems=gt_problems)
    response = llm_client.chat(prompt=prompt, response_format="text", temperature=1)
    raw_response = response.get("content", "")
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}

    parsed = parse_json_dict_best_effort(raw_response)
    if parsed is None:
        preview = str(raw_response).strip()[:500]
        raise ValueError(
            f"Failed to parse problem matcher JSON for patient={patient_id}, window_index={window_index}. "
            f"Response preview: {preview}"
        )

    pairs = _extract_problem_match_pairs(
        parsed,
        num_predictions=len(predicted_problems),
        num_ground_truth=len(gt_problems),
    )
    matched_pred_indices = sorted({int(item["prediction_index"]) for item in pairs})
    matched_gt_indices = sorted({int(item["gt_index"]) for item in pairs if int(item["gt_index"]) >= 0})
    return {
        "matched_pairs": pairs,
        "matched_prediction_indices": matched_pred_indices,
        "matched_gt_indices": matched_gt_indices,
        "raw_response": raw_response,
        "input_tokens": _normalize_token_count(usage.get("input_tokens")),
        "output_tokens": _normalize_token_count(usage.get("output_tokens")),
    }


def _get_thread_llm_client() -> LLMClient:
    client = getattr(_THREAD_LOCAL, "llm_client", None)
    client_key = getattr(_THREAD_LOCAL, "llm_client_key", None)
    next_key = (LLM_PROVIDER, LLM_MODEL, int(LLM_MAX_TOKENS))
    if client is None or client_key != next_key:
        client = LLMClient(provider=LLM_PROVIDER, model=LLM_MODEL, max_tokens=int(LLM_MAX_TOKENS))
        _THREAD_LOCAL.llm_client = client
        _THREAD_LOCAL.llm_client_key = next_key
    return client


def _run_window_match_task(task: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    llm_client = _get_thread_llm_client()
    try:
        match_result = _match_window_with_llm(
            llm_client,
            predicted_problems=task["predicted_problems"],
            gt_problems=task["gt_problems"],
            patient_id=str(task["patient_id"]),
            window_index=int(task["window_index"]),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Active-problem matching failed for patient={task['patient_id']} window={task['window_index']}: {exc}"
        ) from exc
    return task, match_result


def _build_rows_from_match(
    *,
    task: Dict[str, Any],
    match_result: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    predicted_problems = task["predicted_problems"]
    gt_problems = task["gt_problems"]
    matched_pairs = match_result["matched_pairs"]
    matched_prediction_indices = match_result["matched_prediction_indices"]
    matched_gt_indices = match_result["matched_gt_indices"]
    matched_pred_set = {int(v) for v in matched_prediction_indices}
    matched_gt_set = {int(v) for v in matched_gt_indices}
    pair_by_prediction = {int(pair["prediction_index"]): int(pair["gt_index"]) for pair in matched_pairs}

    num_pred = int(len(predicted_problems))
    num_gt = int(len(gt_problems))
    num_matched_pred = int(len(matched_pred_set))
    num_matched_gt = int(len(matched_gt_set))
    hit = int(num_matched_pred > 0)
    precision = float(num_matched_pred) / float(num_pred) if num_pred > 0 else 0.0
    recall = float(num_matched_gt) / float(num_gt) if num_gt > 0 else 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    window_row = {
        "patient_id": task["patient_id"],
        "subject_id": int(task["subject_id"]),
        "icu_stay_id": int(task["icu_stay_id"]),
        "window_index": int(task["window_index"]),
        "oracle_window_marker_id": str(task["oracle_window_marker_id"]),
        "oracle_interval_start_hour": float(task["oracle_interval_start_hour"]),
        "oracle_interval_end_hour": float(task["oracle_interval_end_hour"]),
        "num_predicted_problems": int(num_pred),
        "num_gt_problems": int(num_gt),
        "num_matched_predictions": int(num_matched_pred),
        "num_matched_gt": int(num_matched_gt),
        "hit": int(hit),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched_prediction_indices": list(matched_prediction_indices),
        "matched_gt_indices": list(matched_gt_indices),
        "matched_pairs": list(matched_pairs),
        "matcher_input_tokens": int(match_result["input_tokens"]),
        "matcher_output_tokens": int(match_result["output_tokens"]),
        "matcher_raw_response": str(match_result["raw_response"]),
        "llm_input_tokens": int(match_result["input_tokens"]),
        "llm_output_tokens": int(match_result["output_tokens"]),
        "llm_raw_response": str(match_result["raw_response"]),
    }

    prediction_rows: List[Dict[str, Any]] = []
    for pred_idx, pred_text in enumerate(predicted_problems):
        gt_idx = pair_by_prediction.get(int(pred_idx))
        matched_gt_text = ""
        if gt_idx is not None and gt_idx >= 0 and gt_idx < len(gt_problems):
            matched_gt_text = str(gt_problems[int(gt_idx)])
        prediction_rows.append(
            {
                "patient_id": task["patient_id"],
                "subject_id": int(task["subject_id"]),
                "icu_stay_id": int(task["icu_stay_id"]),
                "window_index": int(task["window_index"]),
                "oracle_window_marker_id": str(task["oracle_window_marker_id"]),
                "oracle_interval_start_hour": float(task["oracle_interval_start_hour"]),
                "oracle_interval_end_hour": float(task["oracle_interval_end_hour"]),
                "prediction_index": int(pred_idx),
                "predicted_problem": str(pred_text),
                "is_matched": int(pred_idx in matched_pred_set),
                "matched_gt_index": gt_idx,
                "matched_gt_problem": matched_gt_text,
                "num_gt_problems": int(num_gt),
            }
        )
    return window_row, prediction_rows


def _load_window_and_prediction_rows(
    prediction_dir: Path,
    *,
    gt_dir: Path,
    num_workers: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if int(num_workers) < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")

    prediction_files: List[Path] = []
    for patient_dir in sorted((prediction_dir / "patients").glob("*")):
        if not patient_dir.is_dir():
            continue
        prediction_file = _find_first_existing_file(patient_dir, PREDICTION_FILENAMES)
        if prediction_file is not None:
            prediction_files.append(prediction_file)
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found under {prediction_dir / 'patients'}")

    prediction_records: List[Dict[str, Any]] = []
    prediction_windows_by_stay: Dict[int, Set[Tuple[float, float]]] = {}
    for pred_path in prediction_files:
        payload = _load_json(pred_path)
        icu_stay_id = _extract_icu_stay_id(payload, pred_path)
        prediction_rows = _extract_window_rows_from_prediction_payload(payload, source_path=pred_path)
        prediction_intervals: Set[Tuple[float, float]] = set()
        for row in prediction_rows:
            prediction_intervals.add(_resolve_prediction_interval(item=row, source_path=pred_path))
        existing_windows = prediction_windows_by_stay.get(int(icu_stay_id))
        if existing_windows is not None:
            raise ValueError(f"Duplicate prediction entries for icu_stay_id={icu_stay_id} in {prediction_dir}")
        prediction_windows_by_stay[int(icu_stay_id)] = set(prediction_intervals)
        prediction_records.append(
            {
                "pred_path": pred_path,
                "payload": payload,
                "icu_stay_id": int(icu_stay_id),
                "prediction_rows": prediction_rows,
            }
        )

    gt_by_stay: Dict[int, Dict[Tuple[float, float], List[str]]] = {}
    gt_windows_by_stay: Dict[int, Set[Tuple[float, float]]] = {}
    for patient_dir in sorted((gt_dir / "patients").glob("*")):
        if not patient_dir.is_dir():
            continue
        gt_file = _find_first_existing_file(patient_dir, GT_FILENAMES)
        if gt_file is None:
            continue
        gt_payload = _load_json(gt_file)
        icu_stay_id = _extract_icu_stay_id(gt_payload, gt_file)
        gt_by_interval = _load_gt_problems_by_interval(gt_file)
        if int(icu_stay_id) in gt_by_stay:
            raise ValueError(f"Duplicate GT entries for icu_stay_id={icu_stay_id} in {gt_dir}")
        gt_by_stay[int(icu_stay_id)] = gt_by_interval
        gt_windows_by_stay[int(icu_stay_id)] = _load_gt_window_intervals(gt_file)

    window_match_stats = _match_prediction_gt_intervals_by_stay(
        prediction_windows_by_stay=prediction_windows_by_stay,
        gt_windows_by_stay=gt_windows_by_stay,
    )
    matched_windows_by_stay = window_match_stats["matched_windows_by_stay"]

    tasks: List[Dict[str, Any]] = []
    skipped_missing_gt_stays = int(window_match_stats["skipped_prediction_stays_missing_gt"])
    skipped_unmatched_prediction_windows = int(window_match_stats["skipped_prediction_windows_missing_gt"])
    skipped_no_gt_windows = 0
    for record in prediction_records:
        pred_path = record["pred_path"]
        payload = record["payload"]
        patient_id = pred_path.parent.name
        icu_stay_id = int(record["icu_stay_id"])
        matched_windows = matched_windows_by_stay.get(int(icu_stay_id))
        if matched_windows is None:
            continue
        gt_by_interval = gt_by_stay.get(int(icu_stay_id))
        if gt_by_interval is None:
            continue

        subject_id_raw = payload.get("subject_id")
        if subject_id_raw is None:
            raise ValueError(f"Missing subject_id in {pred_path}")
        subject_id = int(subject_id_raw)

        prediction_rows = record["prediction_rows"]
        for row in prediction_rows:
            window_index = int(row.get("window_index"))
            interval_key = _resolve_prediction_interval(item=row, source_path=pred_path)
            if interval_key not in matched_windows:
                continue

            gt_problems = gt_by_interval.get(interval_key, [])
            if not gt_problems:
                skipped_no_gt_windows += 1
                continue
            predicted_problems = _extract_predicted_problems_from_row(row)
            tasks.append(
                {
                    "patient_id": patient_id,
                    "subject_id": int(subject_id),
                    "icu_stay_id": int(icu_stay_id),
                    "window_index": int(window_index),
                    "oracle_window_marker_id": _interval_marker_id(interval_key),
                    "oracle_interval_start_hour": float(interval_key[0]),
                    "oracle_interval_end_hour": float(interval_key[1]),
                    "predicted_problems": list(predicted_problems),
                    "gt_problems": list(gt_problems),
                }
            )

    total_tasks = int(len(tasks))
    if total_tasks == 0:
        raise ValueError("No active-problem windows found for evaluation.")

    print(
        f"Loaded active-problem windows: patients={len(prediction_files)}, windows={total_tasks}, "
        f"workers={int(num_workers)}"
    )
    print(
        "Window matching summary: "
        f"prediction_stays={window_match_stats['prediction_stays']}, "
        f"gt_stays={window_match_stats['gt_stays']}, "
        f"matched_stays={window_match_stats['matched_stays']}, "
        f"matched_windows={window_match_stats['matched_windows']}"
    )
    print(f"Skipped prediction stays with missing GT: {skipped_missing_gt_stays}")
    print(f"Skipped prediction windows not matched in GT: {skipped_unmatched_prediction_windows}")
    print(f"Skipped windows with empty GT active problems: {skipped_no_gt_windows}")

    window_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    matcher_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    def _finalize_window(completed: int, task: Dict[str, Any], match_result: Dict[str, Any]) -> None:
        nonlocal matcher_calls, total_input_tokens, total_output_tokens
        if task["predicted_problems"] and task["gt_problems"]:
            matcher_calls += 1
            total_input_tokens += int(match_result["input_tokens"])
            total_output_tokens += int(match_result["output_tokens"])
        window_row, per_prediction_rows = _build_rows_from_match(task=task, match_result=match_result)
        window_rows.append(window_row)
        prediction_rows.extend(per_prediction_rows)
        print(
            f"[{completed}/{total_tasks}] matched patient={task['patient_id']} "
            f"window={task['window_index']} pred={len(task['predicted_problems'])} "
            f"gt={len(task['gt_problems'])} matched={len(match_result['matched_prediction_indices'])}"
        )

    if int(num_workers) == 1:
        for completed, task in enumerate(tasks, start=1):
            _, match_result = _run_window_match_task(task)
            _finalize_window(completed, task, match_result)
    else:
        with ThreadPoolExecutor(max_workers=min(int(num_workers), total_tasks)) as executor:
            futures = [executor.submit(_run_window_match_task, task) for task in tasks]
            completed = 0
            for future in as_completed(futures):
                completed += 1
                task, match_result = future.result()
                _finalize_window(completed, task, match_result)

    window_frame = pd.DataFrame(window_rows)
    if window_frame.empty:
        raise ValueError("No active-problem windows found for evaluation.")
    window_frame = window_frame.sort_values(["patient_id", "window_index"]).reset_index(drop=True)

    prediction_frame = pd.DataFrame(prediction_rows)
    if not prediction_frame.empty:
        prediction_frame = prediction_frame.sort_values(
            ["patient_id", "window_index", "prediction_index"]
        ).reset_index(drop=True)

    usage = {
        "matcher_calls": int(matcher_calls),
        "input_tokens": int(total_input_tokens),
        "output_tokens": int(total_output_tokens),
        "total_tokens": int(total_input_tokens + total_output_tokens),
        "window_match_stats": {
            "prediction_stays": int(window_match_stats["prediction_stays"]),
            "gt_stays": int(window_match_stats["gt_stays"]),
            "matched_stays": int(window_match_stats["matched_stays"]),
            "prediction_windows": int(window_match_stats["prediction_windows"]),
            "gt_windows": int(window_match_stats["gt_windows"]),
            "matched_windows": int(window_match_stats["matched_windows"]),
            "skipped_prediction_stays_missing_gt": int(window_match_stats["skipped_prediction_stays_missing_gt"]),
            "skipped_prediction_windows_missing_gt": int(window_match_stats["skipped_prediction_windows_missing_gt"]),
        },
        "skipped_missing_gt_stays": int(skipped_missing_gt_stays),
        "skipped_unmatched_prediction_windows": int(skipped_unmatched_prediction_windows),
        "skipped_no_gt_windows": int(skipped_no_gt_windows),
    }
    return window_frame, prediction_frame, usage


def _compute_metrics(window_frame: pd.DataFrame) -> Dict[str, Any]:
    if window_frame.empty:
        raise ValueError("No rows to compute metrics.")
    num_windows = int(len(window_frame))
    num_patients = int(window_frame["patient_id"].nunique())

    micro_hit = float(window_frame["hit"].mean()) if num_windows > 0 else float("nan")
    micro_precision = float(window_frame["precision"].mean()) if num_windows > 0 else float("nan")
    micro_recall = float(window_frame["recall"].mean()) if num_windows > 0 else float("nan")
    micro_f1 = float(window_frame["f1"].mean()) if num_windows > 0 else float("nan")

    by_patient = (
        window_frame.groupby("patient_id")[["hit", "precision", "recall", "f1"]]
        .mean()
        .reset_index(drop=True)
    )
    macro_hit = float(by_patient["hit"].mean()) if not by_patient.empty else float("nan")
    macro_precision = float(by_patient["precision"].mean()) if not by_patient.empty else float("nan")
    macro_recall = float(by_patient["recall"].mean()) if not by_patient.empty else float("nan")
    macro_f1 = float(by_patient["f1"].mean()) if not by_patient.empty else float("nan")

    return {
        "num_patients": int(num_patients),
        "num_windows": int(num_windows),
        "micro_hit": float(micro_hit),
        "macro_hit": float(macro_hit),
        "micro_precision": float(micro_precision),
        "macro_precision": float(macro_precision),
        "micro_recall": float(micro_recall),
        "macro_recall": float(macro_recall),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "total_predicted_problems": int(window_frame["num_predicted_problems"].sum()),
        "total_gt_problems": int(window_frame["num_gt_problems"].sum()),
        "total_matched_predictions": int(window_frame["num_matched_predictions"].sum()),
        "total_matched_gt": int(window_frame["num_matched_gt"].sum()),
    }


def _load_reused_active_problem_frames(reuse_eval_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    window_path = reuse_eval_dir / "window_level_windows.csv"
    if not window_path.exists():
        raise FileNotFoundError(f"Missing reused matcher window file: {window_path}")
    window_frame = pd.read_csv(window_path)
    if window_frame.empty:
        raise ValueError(f"Reused matcher window file is empty: {window_path}")

    list_columns = ("matched_prediction_indices", "matched_gt_indices", "matched_pairs")
    for column in list_columns:
        if column not in window_frame.columns:
            window_frame[column] = [[] for _ in range(len(window_frame))]
        else:
            window_frame[column] = window_frame[column].map(_parse_json_list_cell)

    prediction_path = reuse_eval_dir / "window_level_predictions.csv"
    prediction_frame = pd.read_csv(prediction_path) if prediction_path.exists() else pd.DataFrame()

    metrics_path = reuse_eval_dir / "metrics.json"
    metrics_payload = _load_json(metrics_path) if metrics_path.exists() else {}
    matcher_usage = metrics_payload.get("matcher_usage")
    if not isinstance(matcher_usage, dict):
        matcher_usage = {
            "matcher_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "window_match_stats": {},
        }
    matcher_usage = dict(matcher_usage)
    matcher_usage["reused_from"] = str(reuse_eval_dir)
    return window_frame, prediction_frame, matcher_usage


def _write_active_problem_analysis_outputs(
    *,
    window_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    matcher_usage: Dict[str, Any],
    prediction_root: Optional[Path],
    gt_root: Optional[Path],
    output_dir: Path,
    reuse_eval_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    metrics = _compute_metrics(window_frame)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "prediction_dir": str(prediction_root) if prediction_root is not None else None,
        "gt_dir": str(gt_root) if gt_root is not None else None,
        "reuse_eval_dir": str(reuse_eval_dir) if reuse_eval_dir is not None else None,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "llm_max_tokens": int(LLM_MAX_TOKENS),
        "matcher_usage": matcher_usage,
        "metrics": metrics,
    }
    _json_dump(output_dir / "metrics.json", payload)

    window_export = window_frame.copy()
    window_export["matched_prediction_indices"] = window_export["matched_prediction_indices"].map(
        lambda v: json.dumps(v, ensure_ascii=False)
    )
    window_export["matched_gt_indices"] = window_export["matched_gt_indices"].map(
        lambda v: json.dumps(v, ensure_ascii=False)
    )
    window_export["matched_pairs"] = window_export["matched_pairs"].map(lambda v: json.dumps(v, ensure_ascii=False))
    window_export.to_csv(output_dir / "window_level_windows.csv", index=False)
    prediction_frame.to_csv(output_dir / "window_level_predictions.csv", index=False)

    print(f"Saved active-problem evaluation outputs to: {output_dir}")
    print(
        f"Active-Problem Evaluation: hit={metrics['micro_hit']:.4f}, "
        f"precision={metrics['micro_precision']:.4f}, recall={metrics['micro_recall']:.4f}, "
        f"f1={metrics['micro_f1']:.4f} (patients={metrics['num_patients']}, windows={metrics['num_windows']})"
    )
    print(
        f"LLM matcher: provider={LLM_PROVIDER}, model={LLM_MODEL}, "
        f"calls={int(matcher_usage.get('matcher_calls', 0))}, tokens={int(matcher_usage.get('total_tokens', 0))}"
    )

    return {
        "output_dir": output_dir,
        "metrics": metrics,
    }


def _evaluate_single_prediction_dir(
    *,
    prediction_root: Path,
    gt_root: Path,
    output_dir: Path,
    num_workers: int,
) -> Dict[str, Any]:
    window_frame, prediction_frame, matcher_usage = _load_window_and_prediction_rows(
        prediction_dir=prediction_root,
        gt_dir=gt_root,
        num_workers=int(num_workers),
    )
    return _write_active_problem_analysis_outputs(
        window_frame=window_frame,
        prediction_frame=prediction_frame,
        matcher_usage=matcher_usage,
        prediction_root=prediction_root,
        gt_root=gt_root,
        output_dir=output_dir,
    )


def _evaluate_reused_eval_dir(
    *,
    reuse_eval_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    print(f"Reuse eval directory: {reuse_eval_dir}")
    window_frame, prediction_frame, matcher_usage = _load_reused_active_problem_frames(reuse_eval_dir)
    return _write_active_problem_analysis_outputs(
        window_frame=window_frame,
        prediction_frame=prediction_frame,
        matcher_usage=matcher_usage,
        prediction_root=None,
        gt_root=None,
        output_dir=output_dir,
        reuse_eval_dir=reuse_eval_dir,
    )


def run_evaluation(
    *,
    prediction_dir: Optional[Path],
    gt_dir: Optional[Path],
    output_dir: Optional[Path],
    num_workers: int,
    reuse_eval_dir: Optional[Path] = None,
) -> None:
    if reuse_eval_dir is not None:
        reuse_root = _ensure_reuse_eval_dir(reuse_eval_dir)
        mode_dirs = _discover_eval_mode_dirs(reuse_root)
        base_output_root = output_dir if output_dir is not None else reuse_root

        if mode_dirs:
            base_output_root.mkdir(parents=True, exist_ok=True)
            print("Detected reused prediction modes: " + ", ".join(sorted(mode_dirs.keys())) + f" under {reuse_root}")
            for mode in sorted(mode_dirs.keys()):
                print("")
                print(f"=== Reusing matched results for mode: {mode} ===")
                _evaluate_reused_eval_dir(
                    reuse_eval_dir=mode_dirs[mode],
                    output_dir=base_output_root / mode,
                )
            print("")
            print(f"Saved reused multi-mode active-problem evaluation outputs to: {base_output_root}")
            return

        final_output_dir = base_output_root
        _evaluate_reused_eval_dir(reuse_eval_dir=reuse_root, output_dir=final_output_dir)
        return

    if prediction_dir is None:
        raise ValueError("--prediction-dir is required unless --reuse-eval-dir is provided.")
    if gt_dir is None:
        raise ValueError("--gt-dir is required unless --reuse-eval-dir is provided.")

    prediction_root = _ensure_prediction_dir(prediction_dir)
    gt_root = _ensure_gt_dir(gt_dir)

    mode_dirs = _discover_mode_dirs(prediction_root)
    base_output_root = output_dir if output_dir is not None else _default_output_root(prediction_root, gt_root)

    if mode_dirs:
        base_output_root.mkdir(parents=True, exist_ok=True)
        print("Detected prediction modes: " + ", ".join(sorted(mode_dirs.keys())) + f" under {prediction_root}")
        for mode in sorted(mode_dirs.keys()):
            print("")
            print(f"=== Evaluating mode: {mode} ===")
            _evaluate_single_prediction_dir(
                prediction_root=mode_dirs[mode],
                gt_root=gt_root,
                output_dir=base_output_root / mode,
                num_workers=int(num_workers),
            )
        print("")
        print(f"Saved multi-mode active-problem evaluation outputs to: {base_output_root}")
        return

    final_output_dir = base_output_root
    if prediction_root.parent.name == "recommendation_experiment":
        final_output_dir = base_output_root / prediction_root.name
    _evaluate_single_prediction_dir(
        prediction_root=prediction_root,
        gt_root=gt_root,
        output_dir=final_output_dir,
        num_workers=int(num_workers),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate active-problem predictions with LLM semantic matching.")
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default=None,
        help="Prediction directory containing patients/* prediction JSON files.",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=None,
        help="GT directory containing patients/* GT JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. If omitted, defaults to evaluation_results/active_problem_eval/<prediction>/<gt>.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for window-level matching.",
    )
    parser.add_argument(
        "--reuse-eval-dir",
        type=str,
        default=None,
        help=(
            "Existing evaluation output directory containing window_level_windows.csv. "
            "If provided, skips matcher inference and recomputes metrics from saved matcher results."
        ),
    )
    args = parser.parse_args()

    run_evaluation(
        prediction_dir=Path(args.prediction_dir) if args.prediction_dir else None,
        gt_dir=Path(args.gt_dir) if args.gt_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        num_workers=int(args.num_workers),
        reuse_eval_dir=Path(args.reuse_eval_dir) if args.reuse_eval_dir else None,
    )


if __name__ == "__main__":
    main()
