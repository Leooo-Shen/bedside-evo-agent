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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
        rows.append(dict(row))
    return rows


def _load_gt_problems_by_window(gt_file: Path) -> Dict[int, List[str]]:
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

    by_window: Dict[int, List[str]] = {}
    for row in window_rows:
        source_window_raw = row.get("source_window_index")
        window_raw = source_window_raw if source_window_raw is not None else row.get("window_index")
        if window_raw is None:
            continue
        window_index = int(window_raw)
        if window_index < 0:
            continue
        problems = _extract_gt_problems_from_window_row(row)
        by_window[int(window_index)] = problems

    return by_window


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
    window_stride: Optional[int],
    num_workers: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if int(num_workers) < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    normalized_window_stride: Optional[int]
    if window_stride is None:
        normalized_window_stride = None
    else:
        normalized_window_stride = int(window_stride)
        if normalized_window_stride < 1:
            raise ValueError(f"window_stride must be >= 1 when provided, got {window_stride}")

    prediction_files: List[Path] = []
    for patient_dir in sorted((prediction_dir / "patients").glob("*")):
        if not patient_dir.is_dir():
            continue
        prediction_file = _find_first_existing_file(patient_dir, PREDICTION_FILENAMES)
        if prediction_file is not None:
            prediction_files.append(prediction_file)
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found under {prediction_dir / 'patients'}")

    tasks: List[Dict[str, Any]] = []
    skipped_missing_gt_patients = 0
    skipped_no_gt_windows = 0
    for pred_path in prediction_files:
        payload = _load_json(pred_path)
        patient_id = pred_path.parent.name
        gt_file = _find_first_existing_file(gt_dir / "patients" / patient_id, GT_FILENAMES)
        if gt_file is None:
            skipped_missing_gt_patients += 1
            continue
        gt_by_window = _load_gt_problems_by_window(gt_file)

        subject_id_raw = payload.get("subject_id")
        icu_stay_id_raw = payload.get("icu_stay_id")
        if subject_id_raw is None or icu_stay_id_raw is None:
            raise ValueError(f"Missing subject_id/icu_stay_id in {pred_path}")
        subject_id = int(subject_id_raw)
        icu_stay_id = int(icu_stay_id_raw)

        prediction_rows = _extract_window_rows_from_prediction_payload(payload, source_path=pred_path)
        for row in prediction_rows:
            window_index = int(row.get("window_index"))
            if normalized_window_stride is not None and normalized_window_stride > 1:
                if int(window_index) % int(normalized_window_stride) != 0:
                    continue

            gt_problems = gt_by_window.get(window_index, [])
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
    print(f"Skipped patients with missing GT files: {skipped_missing_gt_patients}")
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
        "skipped_missing_gt_patients": int(skipped_missing_gt_patients),
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


def _evaluate_single_prediction_dir(
    *,
    prediction_root: Path,
    gt_root: Path,
    output_dir: Path,
    window_stride: Optional[int],
    num_workers: int,
) -> Dict[str, Any]:
    window_frame, prediction_frame, matcher_usage = _load_window_and_prediction_rows(
        prediction_dir=prediction_root,
        gt_dir=gt_root,
        window_stride=int(window_stride) if window_stride is not None else None,
        num_workers=int(num_workers),
    )

    metrics = _compute_metrics(window_frame)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "prediction_dir": str(prediction_root),
        "gt_dir": str(gt_root),
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
        f"calls={matcher_usage['matcher_calls']}, tokens={matcher_usage['total_tokens']}"
    )

    return {
        "output_dir": output_dir,
        "metrics": metrics,
    }


def run_evaluation(
    *,
    prediction_dir: Path,
    gt_dir: Path,
    output_dir: Optional[Path],
    window_stride: Optional[int],
    num_workers: int,
) -> None:
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
                window_stride=int(window_stride) if window_stride is not None else None,
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
        window_stride=int(window_stride) if window_stride is not None else None,
        num_workers=int(num_workers),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate active-problem predictions with LLM semantic matching.")
    parser.add_argument(
        "--prediction-dir",
        type=str,
        required=True,
        help="Prediction directory containing patients/* prediction JSON files.",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        required=True,
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
        "--window-stride",
        type=int,
        default=None,
        help="Evaluate every n-th window by window_index modulo n. If omitted, evaluate all windows.",
    )
    args = parser.parse_args()

    run_evaluation(
        prediction_dir=Path(args.prediction_dir),
        gt_dir=Path(args.gt_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        window_stride=int(args.window_stride) if args.window_stride is not None else None,
        num_workers=int(args.num_workers),
    )


if __name__ == "__main__":
    main()
