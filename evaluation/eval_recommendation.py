"""Evaluate recommendation predictions with configurable action matching."""

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import get_config
from evaluation.embedding_model import EmbeddingActionMatcher
from experiments.oracle.action_validity_common import (
    event_to_action_text,
    normalize_action_label,
    recommendation_to_text,
)
from experiments.oracle.common import assign_normalized_time_bin
from model.llms import LLMClient
from prompts.action_matcher_prompts import get_action_matcher_prompt
from utils.json_parse import parse_json_dict_best_effort

GT_SOURCE_DATASET_ACTIONS = "dataset_actions"
GT_SOURCE_ORACLE_REVIEWED_ACTIONS = "oracle_reviewed_actions"
ORACLE_POSITIVE_LABELS = frozenset({"best_practice", "acceptable"})
MATCHER_BACKEND_LLM = "llm"
MATCHER_BACKEND_EMBEDDING = "embedding"
KNOWN_RECOMMENDATION_MODES = ("memory", "full_history_events", "local_events_only")

NUM_TIME_BINS = 10
MEMORY_DEPTH_NUM_BINS = 6
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260409
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


def _ensure_predictions_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {path}")
    direct_patients = path / "patients"
    nested_patients = path / "recommendation_experiment" / "patients"
    if direct_patients.exists():
        matches = list(direct_patients.glob("*/recommendation_predictions.json"))
        if matches:
            return path
    if nested_patients.exists():
        matches = list(nested_patients.glob("*/recommendation_predictions.json"))
        if matches:
            return path / "recommendation_experiment"
    raise FileNotFoundError(
        f"No recommendation_predictions.json found under {path}. "
        "Expected either <dir>/patients/*/recommendation_predictions.json or "
        "<dir>/recommendation_experiment/patients/*/recommendation_predictions.json."
    )


def _ensure_oracle_results_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Oracle directory not found: {path}")
    patients_dir = path / "patients"
    if not patients_dir.exists() or not patients_dir.is_dir():
        raise FileNotFoundError(f"Oracle patients directory not found: {patients_dir}")
    matches = list(patients_dir.glob("*/oracle_predictions.json"))
    if not matches:
        raise FileNotFoundError(f"No oracle_predictions.json files found under {patients_dir}")
    return path


def _has_recommendation_predictions(path: Path) -> bool:
    patients_dir = path / "patients"
    if not patients_dir.exists() or not patients_dir.is_dir():
        return False
    return any(patients_dir.glob("*/recommendation_predictions.json"))


def _discover_recommendation_mode_dirs(path: Path) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    for mode in KNOWN_RECOMMENDATION_MODES:
        candidate = path / mode
        if candidate.exists() and candidate.is_dir() and _has_recommendation_predictions(candidate):
            discovered[str(mode)] = candidate
    return discovered


def _default_output_dir_from_recommendation_root(recommendation_root: Path) -> Path:
    experiment_name = recommendation_root.name
    memory_run_name = recommendation_root.parent.name if recommendation_root.parent != recommendation_root else ""
    if not memory_run_name:
        raise ValueError(f"Cannot infer memory run name from recommendation_root={recommendation_root}")
    if not experiment_name:
        raise ValueError(f"Cannot infer experiment name from recommendation_root={recommendation_root}")
    return Path("evaluation_results") / memory_run_name / experiment_name


def _default_output_root_from_recommendation_input(recommendation_input: Path) -> Path:
    if recommendation_input.name == "recommendation_experiment":
        memory_run_name = recommendation_input.parent.name
        if not memory_run_name:
            raise ValueError(f"Cannot infer memory run name from recommendation_input={recommendation_input}")
        return Path("evaluation_results") / memory_run_name / "recommendation_experiment"
    if recommendation_input.parent.name == "recommendation_experiment":
        memory_run_name = recommendation_input.parent.parent.name
        if not memory_run_name:
            raise ValueError(f"Cannot infer memory run name from recommendation_input={recommendation_input}")
        return Path("evaluation_results") / memory_run_name / "recommendation_experiment" / recommendation_input.name
    return _default_output_dir_from_recommendation_root(recommendation_input)


def _output_root_with_backend(base_output_dir: Path, matcher_backend: str) -> Path:
    backend = str(matcher_backend).strip()
    if not backend:
        raise ValueError("matcher_backend cannot be empty when constructing output directory.")
    if base_output_dir.name == backend:
        return base_output_dir
    return base_output_dir / backend


def _infer_snapshot_window_index(snapshot: Mapping[str, Any]) -> Optional[int]:
    working_memory = snapshot.get("working_memory")
    if not isinstance(working_memory, list) or not working_memory:
        return None
    current_window = working_memory[-1]
    if not isinstance(current_window, dict):
        return None
    window_id = current_window.get("window_id")
    if window_id is None:
        return None
    return int(window_id)


def _collect_windowed_snapshots(memory_snapshots: Sequence[Mapping[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    snapshots_by_window: Dict[int, Dict[str, Any]] = {}
    for snapshot in memory_snapshots:
        if not isinstance(snapshot, Mapping):
            raise ValueError("Invalid memory snapshot row; expected mapping.")
        window_index = _infer_snapshot_window_index(snapshot)
        if window_index is None:
            continue
        snapshots_by_window[int(window_index)] = dict(snapshot)
    return sorted(snapshots_by_window.items(), key=lambda item: item[0])


def _load_memory_snapshots(patient_dir: Path) -> List[Mapping[str, Any]]:
    memory_db_path = patient_dir / "memory_database.json"
    payload = _load_json(memory_db_path)
    memory_snapshots = payload.get("memory_snapshots")
    if not isinstance(memory_snapshots, list):
        raise ValueError(f"Invalid memory_snapshots list in {memory_db_path}")
    return memory_snapshots


def _memory_depth_by_window(source_patient_dir: Path) -> Dict[int, int]:
    memory_snapshots = _load_memory_snapshots(source_patient_dir)
    windowed = _collect_windowed_snapshots(memory_snapshots)
    depths: Dict[int, int] = {}
    for window_index, snapshot in windowed:
        trajectory_memory = snapshot.get("trajectory_memory")
        critical_events = snapshot.get("critical_events")
        insights = snapshot.get("insights")
        if not isinstance(trajectory_memory, list):
            trajectory_memory = []
        if not isinstance(critical_events, list):
            critical_events = []
        if not isinstance(insights, list):
            insights = []
        depth = len(trajectory_memory) + len(critical_events) + len(insights)
        depths[int(window_index)] = int(depth)
    return depths


def _ground_truth_action_text(item: Mapping[str, Any]) -> str:
    text = str(item.get("gt_action_text") or "").strip()
    if text:
        return text
    text = recommendation_to_text(item)
    if text:
        return text
    return event_to_action_text(item)


def _load_oracle_reviewed_actions_by_window(oracle_prediction_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    payload = _load_json(oracle_prediction_path)
    window_outputs = payload.get("window_outputs")
    if not isinstance(window_outputs, list):
        raise ValueError(f"Invalid window_outputs list in {oracle_prediction_path}")

    actions_by_window: Dict[int, List[Dict[str, Any]]] = {}
    for row in window_outputs:
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid row in window_outputs in {oracle_prediction_path}")
        source_window_index_raw = row.get("source_window_index")
        window_index_raw = source_window_index_raw if source_window_index_raw is not None else row.get("window_index")
        if window_index_raw is None:
            raise ValueError(f"Missing window index in window_outputs row in {oracle_prediction_path}")
        window_index = int(window_index_raw)

        oracle_output = row.get("oracle_output")
        if not isinstance(oracle_output, Mapping):
            continue
        action_review = oracle_output.get("action_review")
        if not isinstance(action_review, Mapping):
            continue
        evaluations = action_review.get("evaluations")
        if evaluations is None:
            continue
        if not isinstance(evaluations, list):
            raise ValueError(
                f"Invalid action_review.evaluations for window={window_index} in {oracle_prediction_path}"
            )

        selected_actions: List[Dict[str, Any]] = []
        for evaluation in evaluations:
            if not isinstance(evaluation, Mapping):
                raise ValueError(
                    f"Invalid evaluation row for window={window_index} in {oracle_prediction_path}: {evaluation}"
                )
            label = normalize_action_label(evaluation.get("label"))
            if label not in ORACLE_POSITIVE_LABELS:
                continue
            action_name = str(evaluation.get("action_name") or "").strip()
            if not action_name:
                raise ValueError(
                    f"Missing action_name for window={window_index} with label={label} in {oracle_prediction_path}"
                )
            rationale = str(evaluation.get("rationale") or evaluation.get("reason") or "").strip()
            gt_text = f"{action_name}. {rationale}".strip() if rationale else action_name
            selected_actions.append(
                {
                    "action_name": action_name,
                    "action_description": rationale,
                    "gt_action_text": gt_text,
                    "oracle_label": label,
                    "oracle_action_id": str(evaluation.get("action_id") or "").strip(),
                }
            )

        if not selected_actions:
            continue
        if window_index in actions_by_window:
            raise ValueError(
                f"Duplicate oracle action reviews for source window={window_index} in {oracle_prediction_path}"
            )
        actions_by_window[window_index] = selected_actions
    return actions_by_window


def _build_window_matching_prompt(
    prediction_items: Sequence[Mapping[str, Any]],
    gt_items: Sequence[Mapping[str, Any]],
) -> str:
    prediction_payload = []
    for idx, item in enumerate(prediction_items):
        prediction_payload.append(
            {
                "prediction_index": int(idx),
                "action_name": str(item.get("action_name") or "").strip(),
                "action_description": str(item.get("action_description") or "").strip(),
            }
        )

    gt_payload = []
    for idx, item in enumerate(gt_items):
        gt_payload.append(
            {
                "gt_index": int(idx),
                "action_description": _ground_truth_action_text(item),
            }
        )

    prompt_template = get_action_matcher_prompt()
    predicted_actions = json.dumps(prediction_payload, ensure_ascii=False, indent=2)
    ground_truth_actions = json.dumps(gt_payload, ensure_ascii=False, indent=2)
    return prompt_template.replace("{predicted_actions}", predicted_actions).replace(
        "{ground_truth_actions}",
        ground_truth_actions,
    )


def _extract_match_pairs(
    parsed_payload: Dict[str, Any],
    *,
    num_predictions: int,
    num_ground_truth: int,
) -> List[Dict[str, int]]:
    matched_pairs_raw = parsed_payload.get("matched_pairs")
    if not isinstance(matched_pairs_raw, list):
        matches_raw = parsed_payload.get("matches")
        if isinstance(matches_raw, list):
            matched_pairs = []
            seen_predictions = set()
            seen_gt_indices = set()
            for row in matches_raw:
                if not isinstance(row, Mapping):
                    raise ValueError(f"Invalid matches row: {row}")
                pred_raw = row.get("pred_idx")
                gt_indices_raw = row.get("gt_indices")
                try:
                    prediction_index = int(pred_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid pred_idx in matches row: {row}") from exc
                if prediction_index < 0 or prediction_index >= num_predictions:
                    raise ValueError(f"pred_idx out of range: {prediction_index}")
                if prediction_index in seen_predictions:
                    raise ValueError(f"Duplicate pred_idx in matches: {prediction_index}")
                seen_predictions.add(prediction_index)

                if not isinstance(gt_indices_raw, list):
                    raise ValueError(f"gt_indices must be a list in matches row: {row}")
                parsed_gt_indices: List[int] = []
                for value in gt_indices_raw:
                    try:
                        gt_index = int(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Invalid gt index in gt_indices: {value}") from exc
                    if gt_index < 0 or gt_index >= num_ground_truth:
                        raise ValueError(f"gt index out of range: {gt_index}")
                    if gt_index in seen_gt_indices:
                        raise ValueError(
                            "Ground-truth action index is matched more than once across predictions: " f"{gt_index}"
                        )
                    parsed_gt_indices.append(gt_index)

                if not parsed_gt_indices:
                    continue
                for gt_index in parsed_gt_indices:
                    seen_gt_indices.add(gt_index)
                matched_pairs.append(
                    {
                        "prediction_index": int(prediction_index),
                        "gt_index": int(parsed_gt_indices[0]),
                    }
                )
            return matched_pairs

    if not isinstance(matched_pairs_raw, list):
        matched_prediction_indices = parsed_payload.get("matched_prediction_indices")
        if not isinstance(matched_prediction_indices, list):
            raise ValueError("LLM output must include matched_pairs list.")
        matched_pairs: List[Dict[str, int]] = []
        seen_predictions = set()
        for value in matched_prediction_indices:
            try:
                prediction_index = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid prediction index in matched_prediction_indices: {value}") from exc
            if prediction_index < 0 or prediction_index >= num_predictions:
                raise ValueError(f"Prediction index out of range: {prediction_index}")
            if prediction_index in seen_predictions:
                raise ValueError(f"Duplicate prediction index in LLM output: {prediction_index}")
            seen_predictions.add(prediction_index)
            matched_pairs.append({"prediction_index": int(prediction_index), "gt_index": -1})
        return matched_pairs

    matched_pairs = []
    seen_predictions = set()
    seen_gt_indices = set()
    for row in matched_pairs_raw:
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid matched pair row: {row}")

        prediction_raw = row.get("prediction_index")
        gt_raw = row.get("gt_index")
        try:
            prediction_index = int(prediction_raw)
            gt_index = int(gt_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid pair indices: {row}") from exc

        if prediction_index < 0 or prediction_index >= num_predictions:
            raise ValueError(f"prediction_index out of range: {prediction_index}")
        if gt_index < 0 or gt_index >= num_ground_truth:
            raise ValueError(f"gt_index out of range: {gt_index}")
        if prediction_index in seen_predictions:
            raise ValueError(f"Duplicate prediction_index in matched_pairs: {prediction_index}")
        if gt_index in seen_gt_indices:
            raise ValueError(f"Duplicate gt_index in matched_pairs: {gt_index}")

        seen_predictions.add(prediction_index)
        seen_gt_indices.add(gt_index)
        matched_pairs.append({"prediction_index": int(prediction_index), "gt_index": int(gt_index)})

    return matched_pairs


def _match_window_with_llm(
    llm_client: LLMClient,
    *,
    prediction_items: Sequence[Mapping[str, Any]],
    gt_items: Sequence[Mapping[str, Any]],
    patient_id: str,
    window_index: int,
) -> Dict[str, Any]:
    if not prediction_items or not gt_items:
        return {
            "matched_pairs": [],
            "matched_prediction_indices": [],
            "raw_response": "",
            "input_tokens": 0,
            "output_tokens": 0,
        }

    prompt = _build_window_matching_prompt(prediction_items=prediction_items, gt_items=gt_items)
    response = llm_client.chat(prompt=prompt, response_format="text", temperature=1)
    raw_response = response.get("content", "")
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}

    parsed = parse_json_dict_best_effort(raw_response)

    if parsed is None:
        preview = str(raw_response).strip()[:500]
        raise ValueError(
            f"Failed to parse LLM JSON for patient={patient_id}, window_index={window_index}. "
            f"Response preview: {preview}"
        )

    matched_pairs = _extract_match_pairs(
        parsed,
        num_predictions=len(prediction_items),
        num_ground_truth=len(gt_items),
    )
    matched_prediction_indices = sorted({int(item["prediction_index"]) for item in matched_pairs})

    return {
        "matched_pairs": matched_pairs,
        "matched_prediction_indices": matched_prediction_indices,
        "raw_response": raw_response,
        "input_tokens": _normalize_token_count(usage.get("input_tokens")),
        "output_tokens": _normalize_token_count(usage.get("output_tokens")),
    }


def _get_thread_llm_client(
    *,
    llm_provider: str,
    llm_model: str,
    llm_max_tokens: int,
) -> LLMClient:
    client = getattr(_THREAD_LOCAL, "llm_client", None)
    client_key = getattr(_THREAD_LOCAL, "llm_client_key", None)
    next_key = (str(llm_provider), str(llm_model), int(llm_max_tokens))
    if client is None or client_key != next_key:
        client = LLMClient(
            provider=str(llm_provider),
            model=str(llm_model),
            max_tokens=int(llm_max_tokens),
        )
        _THREAD_LOCAL.llm_client = client
        _THREAD_LOCAL.llm_client_key = next_key
    return client


def _get_thread_embedding_matcher(
    *,
    embedding_model_name: str,
    embedding_similarity_threshold: float,
    embedding_device: Optional[str],
) -> EmbeddingActionMatcher:
    matcher = getattr(_THREAD_LOCAL, "embedding_matcher", None)
    matcher_key = getattr(_THREAD_LOCAL, "embedding_matcher_key", None)
    next_key = (
        str(embedding_model_name),
        float(embedding_similarity_threshold),
        str(embedding_device) if embedding_device is not None else None,
    )
    if matcher is None or matcher_key != next_key:
        matcher = EmbeddingActionMatcher(
            model_name=str(embedding_model_name),
            similarity_threshold=float(embedding_similarity_threshold),
            device=str(embedding_device) if embedding_device is not None else None,
        )
        _THREAD_LOCAL.embedding_matcher = matcher
        _THREAD_LOCAL.embedding_matcher_key = next_key
    return matcher


def _prediction_action_text(item: Mapping[str, Any]) -> str:
    text = recommendation_to_text(item)
    if text:
        return text
    action_name = str(item.get("action_name") or "").strip()
    action_description = str(item.get("action_description") or "").strip()
    if action_name and action_description and action_name.lower() not in action_description.lower():
        return f"{action_name}. {action_description}".strip()
    return (action_description or action_name).strip()


def _match_window_with_embedding(
    embedding_matcher: EmbeddingActionMatcher,
    *,
    prediction_items: Sequence[Mapping[str, Any]],
    gt_items: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not prediction_items or not gt_items:
        return {
            "matched_pairs": [],
            "matched_prediction_indices": [],
            "raw_response": "",
            "input_tokens": 0,
            "output_tokens": 0,
        }

    prediction_texts = [_prediction_action_text(item) for item in prediction_items]
    gt_texts = [_ground_truth_action_text(item) for item in gt_items]
    match_payload = embedding_matcher.match(
        prediction_texts=prediction_texts,
        gt_texts=gt_texts,
    )
    raw_response = json.dumps(match_payload, ensure_ascii=False)
    return {
        "matched_pairs": list(match_payload["matched_pairs"]),
        "matched_prediction_indices": list(match_payload["matched_prediction_indices"]),
        "raw_response": raw_response,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def _run_window_match_task(
    task: Dict[str, Any],
    *,
    matcher_backend: str,
    llm_provider: Optional[str],
    llm_model: Optional[str],
    llm_max_tokens: Optional[int],
    embedding_model_name: Optional[str],
    embedding_similarity_threshold: Optional[float],
    embedding_device: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    backend = str(matcher_backend)
    if backend == MATCHER_BACKEND_LLM:
        if llm_provider is None or llm_model is None or llm_max_tokens is None:
            raise ValueError("LLM matcher requires llm_provider, llm_model, and llm_max_tokens.")
        llm_client = _get_thread_llm_client(
            llm_provider=str(llm_provider),
            llm_model=str(llm_model),
            llm_max_tokens=int(llm_max_tokens),
        )
        try:
            match_result = _match_window_with_llm(
                llm_client,
                prediction_items=task["prediction_items"],
                gt_items=task["gt_items"],
                patient_id=str(task["patient_id"]),
                window_index=int(task["window_index"]),
            )
        except Exception as exc:
            raise RuntimeError(
                f"LLM matching failed for patient={task['patient_id']} window={task['window_index']}: {exc}"
            ) from exc
        return task, match_result

    if backend == MATCHER_BACKEND_EMBEDDING:
        if embedding_model_name is None:
            raise ValueError("Embedding matcher requires embedding_model_name.")
        if embedding_similarity_threshold is None:
            raise ValueError("Embedding matcher requires embedding_similarity_threshold.")
        embedding_matcher = _get_thread_embedding_matcher(
            embedding_model_name=str(embedding_model_name),
            embedding_similarity_threshold=float(embedding_similarity_threshold),
            embedding_device=str(embedding_device) if embedding_device is not None else None,
        )
        try:
            match_result = _match_window_with_embedding(
                embedding_matcher,
                prediction_items=task["prediction_items"],
                gt_items=task["gt_items"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Embedding matching failed for patient={task['patient_id']} window={task['window_index']}: {exc}"
            ) from exc
        return task, match_result

    raise ValueError(f"Unsupported matcher_backend={matcher_backend}")


def _build_rows_from_match(
    *,
    task: Dict[str, Any],
    match_result: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prediction_items = task["prediction_items"]
    gt_items = task["gt_items"]
    matched_pairs = match_result["matched_pairs"]
    matched_prediction_indices = match_result["matched_prediction_indices"]
    matched_index_set = set(int(value) for value in matched_prediction_indices)

    window_row = {
        "patient_id": task["patient_id"],
        "subject_id": int(task["subject_id"]),
        "icu_stay_id": int(task["icu_stay_id"]),
        "window_index": int(task["window_index"]),
        "num_windows": int(task["num_windows"]),
        "relative_time": float(task["relative_time"]),
        "time_bin": int(task["time_bin"]),
        "hours_since_admission": float(task["hours_since_admission"]),
        "top_k_actions": int(task["top_k_actions"]),
        "num_recommendations": int(task["num_recommendations"]),
        "num_ground_truth_actions": int(task["num_ground_truth_actions"]),
        "ground_truth_source": str(task["ground_truth_source"]),
        "matcher_backend": str(task["matcher_backend"]),
        "memory_depth": task["memory_depth"],
        "matched_prediction_indices": list(matched_prediction_indices),
        "matched_pairs": list(matched_pairs),
        "matcher_input_tokens": int(match_result["input_tokens"]),
        "matcher_output_tokens": int(match_result["output_tokens"]),
        "matcher_raw_response": str(match_result["raw_response"]),
        "llm_input_tokens": int(match_result["input_tokens"]),
        "llm_output_tokens": int(match_result["output_tokens"]),
        "llm_raw_response": str(match_result["raw_response"]),
    }

    prediction_rows: List[Dict[str, Any]] = []
    gt_texts = [_ground_truth_action_text(item) for item in gt_items]
    pair_by_prediction = {int(pair["prediction_index"]): int(pair["gt_index"]) for pair in matched_pairs}
    for prediction_index, prediction_item in enumerate(prediction_items):
        prediction_rows.append(
            {
                "patient_id": task["patient_id"],
                "subject_id": int(task["subject_id"]),
                "icu_stay_id": int(task["icu_stay_id"]),
                "window_index": int(task["window_index"]),
                "prediction_index": int(prediction_index),
                "is_matched": int(prediction_index in matched_index_set),
                "matched_gt_index": pair_by_prediction.get(prediction_index),
                "matcher_backend": str(task["matcher_backend"]),
                "recommended_action_name": str(prediction_item.get("action_name") or "").strip(),
                "recommended_action_description": str(prediction_item.get("action_description") or "").strip(),
                "recommended_action_text": recommendation_to_text(prediction_item),
                "matched_gt_action_text": (
                    gt_texts[pair_by_prediction[prediction_index]]
                    if prediction_index in pair_by_prediction
                    and pair_by_prediction[prediction_index] >= 0
                    and pair_by_prediction[prediction_index] < len(gt_texts)
                    else ""
                ),
                "num_ground_truth_actions": int(task["num_ground_truth_actions"]),
                "ground_truth_source": str(task["ground_truth_source"]),
            }
        )
    return window_row, prediction_rows


def _load_window_and_prediction_rows(
    prediction_dir: Path,
    *,
    gt_source: str,
    oracle_results_dir: Optional[Path],
    matcher_backend: str,
    llm_provider: Optional[str],
    llm_model: Optional[str],
    llm_max_tokens: Optional[int],
    embedding_model_name: Optional[str],
    embedding_similarity_threshold: Optional[float],
    embedding_device: Optional[str],
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
    backend = str(matcher_backend)
    if backend not in {MATCHER_BACKEND_LLM, MATCHER_BACKEND_EMBEDDING}:
        raise ValueError(f"Unsupported matcher_backend={matcher_backend}")
    if backend == MATCHER_BACKEND_LLM:
        if llm_provider is None or llm_model is None or llm_max_tokens is None:
            raise ValueError("LLM matcher requires llm_provider, llm_model, and llm_max_tokens.")
    if backend == MATCHER_BACKEND_EMBEDDING:
        if embedding_model_name is None:
            raise ValueError("Embedding matcher requires embedding_model_name.")
        if embedding_similarity_threshold is None:
            raise ValueError("Embedding matcher requires embedding_similarity_threshold.")

    prediction_files = sorted((prediction_dir / "patients").glob("*/recommendation_predictions.json"))
    if not prediction_files:
        raise FileNotFoundError(f"No recommendation_predictions.json files found under {prediction_dir / 'patients'}")

    tasks: List[Dict[str, Any]] = []
    skipped_no_oracle_gt_windows = 0
    for pred_path in prediction_files:
        payload = _load_json(pred_path)
        patient_id = pred_path.parent.name
        oracle_actions_by_window: Optional[Dict[int, List[Dict[str, Any]]]] = None
        if gt_source == GT_SOURCE_ORACLE_REVIEWED_ACTIONS:
            if oracle_results_dir is None:
                raise ValueError("oracle_results_dir is required when gt_source=oracle_reviewed_actions")
            oracle_prediction_path = oracle_results_dir / "patients" / patient_id / "oracle_predictions.json"
            if not oracle_prediction_path.exists():
                raise FileNotFoundError(
                    f"Missing oracle_predictions.json for patient={patient_id}: {oracle_prediction_path}"
                )
            oracle_actions_by_window = _load_oracle_reviewed_actions_by_window(oracle_prediction_path)

        subject_id_raw = payload.get("subject_id")
        icu_stay_id_raw = payload.get("icu_stay_id")
        if subject_id_raw is None or icu_stay_id_raw is None:
            raise ValueError(f"Missing subject_id/icu_stay_id in {pred_path}")
        subject_id = int(subject_id_raw)
        icu_stay_id = int(icu_stay_id_raw)

        num_memory_snapshots_raw = payload.get("num_memory_snapshots")
        if num_memory_snapshots_raw is None:
            raise ValueError(f"Missing num_memory_snapshots in {pred_path}")
        num_memory_snapshots = int(num_memory_snapshots_raw)
        if num_memory_snapshots <= 0:
            raise ValueError(f"num_memory_snapshots must be > 0 in {pred_path}")

        top_k_actions_raw = payload.get("top_k_actions")
        if top_k_actions_raw is None:
            raise ValueError(f"Missing top_k_actions in {pred_path}")
        top_k_actions = int(top_k_actions_raw)
        if top_k_actions < 1:
            raise ValueError(f"top_k_actions must be >= 1 in {pred_path}")

        source_patient_dir_raw = payload.get("source_patient_dir")
        if source_patient_dir_raw is None:
            raise ValueError(f"Missing source_patient_dir in {pred_path}")
        source_patient_dir = Path(str(source_patient_dir_raw))
        memory_depth_map = _memory_depth_by_window(source_patient_dir)

        predictions = payload.get("recommendation_predictions")
        if not isinstance(predictions, list):
            raise ValueError(f"Missing recommendation_predictions list in {pred_path}")

        for item in predictions:
            if not isinstance(item, Mapping):
                raise ValueError(f"Invalid prediction row in {pred_path}")

            window_raw = item.get("window_index")
            if window_raw is None:
                raise ValueError(f"Missing window_index in prediction row in {pred_path}")
            window_index = int(window_raw)
            if window_index < 0:
                raise ValueError(f"Negative window_index={window_index} in {pred_path}")
            if normalized_window_stride is not None and normalized_window_stride > 1:
                if int(window_index) % int(normalized_window_stride) != 0:
                    continue

            recommended_actions = item.get("recommended_actions")
            if not isinstance(recommended_actions, list):
                raise ValueError(f"Missing recommended_actions list in {pred_path} window={window_index}")
            prediction_items = [dict(action) for action in recommended_actions if isinstance(action, Mapping)]
            if gt_source == GT_SOURCE_DATASET_ACTIONS:
                ground_truth_action_events = item.get("ground_truth_action_events")
                if not isinstance(ground_truth_action_events, list):
                    raise ValueError(f"Missing ground_truth_action_events list in {pred_path} window={window_index}")
                gt_items = [dict(event) for event in ground_truth_action_events if isinstance(event, Mapping)]
            elif gt_source == GT_SOURCE_ORACLE_REVIEWED_ACTIONS:
                if oracle_actions_by_window is None:
                    raise ValueError(f"Oracle review cache unavailable for {pred_path}")
                if window_index not in oracle_actions_by_window:
                    skipped_no_oracle_gt_windows += 1
                    continue
                gt_items = [dict(action) for action in oracle_actions_by_window[window_index]]
            else:
                raise ValueError(f"Unsupported gt_source={gt_source}")

            num_windows = int(num_memory_snapshots)
            if num_windows > 1:
                relative_time = float(window_index) / float(num_windows - 1)
            else:
                relative_time = 0.0
            relative_time = min(max(relative_time, 0.0), 1.0)

            time_bin = assign_normalized_time_bin(relative_time, num_bins=NUM_TIME_BINS)
            hours_since_admission = float(item.get("hours_since_admission") or 0.0)
            memory_depth = memory_depth_map.get(window_index)

            tasks.append(
                {
                    "patient_id": patient_id,
                    "subject_id": int(subject_id),
                    "icu_stay_id": int(icu_stay_id),
                    "window_index": int(window_index),
                    "num_windows": int(num_windows),
                    "relative_time": float(relative_time),
                    "time_bin": int(time_bin),
                    "hours_since_admission": float(hours_since_admission),
                    "top_k_actions": int(top_k_actions),
                    "num_recommendations": int(len(prediction_items)),
                    "num_ground_truth_actions": int(len(gt_items)),
                    "memory_depth": memory_depth,
                    "ground_truth_source": str(gt_source),
                    "matcher_backend": str(backend),
                    "prediction_items": prediction_items,
                    "gt_items": gt_items,
                }
            )

    total_tasks = int(len(tasks))
    if total_tasks == 0:
        raise ValueError("No recommendation prediction windows found for evaluation.")
    print(
        f"Loaded recommendation predictions: patients={len(prediction_files)}, "
        f"windows={total_tasks}, workers={int(num_workers)}"
    )
    if gt_source == GT_SOURCE_ORACLE_REVIEWED_ACTIONS:
        print(f"Skipped windows with no Oracle best_practice/acceptable actions: {skipped_no_oracle_gt_windows}")

    window_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    matcher_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    def _finalize_window(
        *,
        completed: int,
        task: Dict[str, Any],
        match_result: Dict[str, Any],
    ) -> None:
        nonlocal matcher_calls, total_input_tokens, total_output_tokens
        if task["prediction_items"] and task["gt_items"]:
            matcher_calls += 1
            total_input_tokens += int(match_result["input_tokens"])
            total_output_tokens += int(match_result["output_tokens"])
        window_row, per_prediction_rows = _build_rows_from_match(task=task, match_result=match_result)
        window_rows.append(window_row)
        prediction_rows.extend(per_prediction_rows)
        print(
            f"[{completed}/{total_tasks}] matched patient={task['patient_id']} "
            f"window={task['window_index']} rec={task['num_recommendations']} "
            f"gt={task['num_ground_truth_actions']} matched={len(match_result['matched_prediction_indices'])}"
        )

    if int(num_workers) == 1:
        for completed, task in enumerate(tasks, start=1):
            _, match_result = _run_window_match_task(
                task,
                matcher_backend=backend,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_max_tokens=llm_max_tokens,
                embedding_model_name=embedding_model_name,
                embedding_similarity_threshold=embedding_similarity_threshold,
                embedding_device=embedding_device,
            )
            _finalize_window(completed=completed, task=task, match_result=match_result)
    else:
        with ThreadPoolExecutor(max_workers=min(int(num_workers), total_tasks)) as executor:
            futures = [
                executor.submit(
                    _run_window_match_task,
                    task,
                    matcher_backend=backend,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    llm_max_tokens=llm_max_tokens,
                    embedding_model_name=embedding_model_name,
                    embedding_similarity_threshold=embedding_similarity_threshold,
                    embedding_device=embedding_device,
                )
                for task in tasks
            ]
            completed = 0
            for future in as_completed(futures):
                completed += 1
                task, match_result = future.result()
                _finalize_window(completed=completed, task=task, match_result=match_result)

    window_frame = pd.DataFrame(window_rows)
    if window_frame.empty:
        raise ValueError("No recommendation prediction windows found for evaluation.")
    window_frame = window_frame.sort_values(["patient_id", "window_index"]).reset_index(drop=True)
    prediction_frame = pd.DataFrame(prediction_rows)
    if not prediction_frame.empty:
        prediction_frame = prediction_frame.sort_values(
            ["patient_id", "window_index", "prediction_index"]
        ).reset_index(drop=True)

    matcher_usage = {
        "matcher_backend": str(backend),
        "matcher_calls": int(matcher_calls),
        "input_tokens": int(total_input_tokens),
        "output_tokens": int(total_output_tokens),
        "total_tokens": int(total_input_tokens + total_output_tokens),
        "skipped_no_oracle_gt_windows": int(skipped_no_oracle_gt_windows),
    }
    return window_frame, prediction_frame, matcher_usage


def _k_values(window_frame: pd.DataFrame) -> List[int]:
    max_top_k = int(pd.to_numeric(window_frame["top_k_actions"], errors="coerce").max())
    if max_top_k < 1:
        raise ValueError("Cannot infer K values: top_k_actions must be >= 1.")
    return list(range(1, max_top_k + 1))


def _expand_window_rows_by_k(window_frame: pd.DataFrame, *, k_values: Sequence[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in window_frame.iterrows():
        num_recommendations = int(row["num_recommendations"])
        matched_prediction_indices = row["matched_prediction_indices"]
        if not isinstance(matched_prediction_indices, list):
            matched_prediction_indices = []
        matched_set = {int(value) for value in matched_prediction_indices}

        base = {
            "patient_id": row["patient_id"],
            "subject_id": int(row["subject_id"]),
            "icu_stay_id": int(row["icu_stay_id"]),
            "window_index": int(row["window_index"]),
            "num_windows": int(row["num_windows"]),
            "relative_time": float(row["relative_time"]),
            "time_bin": int(row["time_bin"]),
            "hours_since_admission": float(row["hours_since_admission"]),
            "memory_depth": row["memory_depth"],
            "num_recommendations": int(num_recommendations),
            "num_ground_truth_actions": int(row["num_ground_truth_actions"]),
            "top_k_actions": int(row["top_k_actions"]),
        }

        for k in k_values:
            num_considered = min(int(k), int(num_recommendations))
            num_matches_at_k = sum(1 for idx in matched_set if idx < num_considered)
            hit_at_k = int(num_matches_at_k > 0)
            if num_considered > 0:
                precision_at_k = float(num_matches_at_k) / float(num_considered)
            else:
                precision_at_k = 0.0

            rows.append(
                {
                    **base,
                    "k": int(k),
                    "num_recommendations_at_k": int(num_considered),
                    "num_matches_at_k": int(num_matches_at_k),
                    "hit_at_k": int(hit_at_k),
                    "precision_at_k": float(precision_at_k),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No K-expanded rows generated.")
    return frame


def _bootstrap_mean_ci(values: Sequence[float], *, n_samples: int, rng: np.random.Generator) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        value = float(arr[0])
        return value, value
    samples = rng.choice(arr, size=(n_samples, len(arr)), replace=True)
    means = samples.mean(axis=1)
    low = float(np.quantile(means, 0.025))
    high = float(np.quantile(means, 0.975))
    return low, high


def _compute_metrics_by_k(window_k_frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for k, subset in window_k_frame.groupby("k"):
        num_windows = int(len(subset))
        micro_hit = float(subset["hit_at_k"].mean()) if num_windows > 0 else float("nan")
        micro_precision = float(subset["precision_at_k"].mean()) if num_windows > 0 else float("nan")

        patient_hit = subset.groupby("patient_id")["hit_at_k"].mean().astype(float)
        patient_precision = subset.groupby("patient_id")["precision_at_k"].mean().astype(float)
        macro_hit = float(patient_hit.mean()) if len(patient_hit) > 0 else float("nan")
        macro_precision = float(patient_precision.mean()) if len(patient_precision) > 0 else float("nan")

        rows.append(
            {
                "k": int(k),
                "num_windows": int(num_windows),
                "num_patients": int(subset["patient_id"].nunique()),
                "micro_hit_at_k": float(micro_hit),
                "macro_hit_at_k": float(macro_hit),
                "micro_precision_at_k": float(micro_precision),
                "macro_precision_at_k": float(macro_precision),
                "total_matches_at_k": int(subset["num_matches_at_k"].sum()),
                "total_recommendations_at_k": int(subset["num_recommendations_at_k"].sum()),
            }
        )
    metrics_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    if metrics_df.empty:
        raise ValueError("No K-level metrics computed.")
    return metrics_df


def _memory_depth_bins(values: pd.Series) -> Tuple[pd.Series, List[str]]:
    valid = values.dropna().astype(float)
    if valid.empty:
        raise ValueError("Cannot build memory-depth bins from empty values.")
    unique = np.unique(valid.to_numpy(dtype=float))
    if len(unique) <= MEMORY_DEPTH_NUM_BINS:
        sorted_unique = sorted(int(value) for value in unique.tolist())
        index_map = {value: idx for idx, value in enumerate(sorted_unique)}
        mapped = values.map(lambda x: index_map.get(int(float(x))) if pd.notna(x) else pd.NA)
        labels = [f"{value}" for value in sorted_unique]
        return mapped.astype("Int64"), labels

    quantile_bins = pd.qcut(valid, q=MEMORY_DEPTH_NUM_BINS, labels=False, duplicates="drop")
    if quantile_bins.isna().all():
        raise ValueError("Failed to create quantile memory-depth bins.")

    bin_index_by_row = pd.Series(index=valid.index, data=quantile_bins.astype(int).to_numpy())
    full_codes = pd.Series(pd.NA, index=values.index, dtype="Int64")
    full_codes.loc[bin_index_by_row.index] = bin_index_by_row.astype("Int64")

    labels: List[str] = []
    for bin_id in sorted(bin_index_by_row.unique().tolist()):
        selected = valid[bin_index_by_row == bin_id]
        left = int(math.floor(float(selected.min())))
        right = int(math.ceil(float(selected.max())))
        labels.append(f"{left}-{right}")
    return full_codes.astype("Int64"), labels


def _plot_relative_time_curve(
    window_k_frame: pd.DataFrame,
    *,
    selected_k: int,
    output_path: Path,
) -> pd.DataFrame:
    subset = window_k_frame[window_k_frame["k"] == int(selected_k)].copy()
    if subset.empty:
        raise ValueError(f"No rows available for selected_k={selected_k}")

    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for time_bin in range(NUM_TIME_BINS):
        bin_rows = subset[subset["time_bin"] == int(time_bin)]
        hit_values = bin_rows["hit_at_k"].astype(float).tolist()
        precision_values = bin_rows["precision_at_k"].astype(float).tolist()

        hit_mean = float(np.mean(hit_values)) if hit_values else float("nan")
        precision_mean = float(np.mean(precision_values)) if precision_values else float("nan")
        hit_ci_low, hit_ci_high = _bootstrap_mean_ci(hit_values, n_samples=BOOTSTRAP_SAMPLES, rng=rng)
        precision_ci_low, precision_ci_high = _bootstrap_mean_ci(
            precision_values,
            n_samples=BOOTSTRAP_SAMPLES,
            rng=rng,
        )

        rows.append(
            {
                "k": int(selected_k),
                "time_bin": int(time_bin),
                "time_bin_mid_pct": float((time_bin + 0.5) * (100.0 / NUM_TIME_BINS)),
                "n_windows": int(len(bin_rows)),
                "hit_at_k": float(hit_mean),
                "hit_ci_low": float(hit_ci_low),
                "hit_ci_high": float(hit_ci_high),
                "precision_at_k": float(precision_mean),
                "precision_ci_low": float(precision_ci_low),
                "precision_ci_high": float(precision_ci_high),
            }
        )

    plot_df = pd.DataFrame(rows).sort_values("time_bin").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = plot_df["time_bin_mid_pct"].to_numpy(dtype=float)

    y_hit = plot_df["hit_at_k"].to_numpy(dtype=float)
    low_hit = plot_df["hit_ci_low"].to_numpy(dtype=float)
    high_hit = plot_df["hit_ci_high"].to_numpy(dtype=float)
    axes[0].plot(x, y_hit, marker="o", linewidth=2.2, color="#1f77b4")
    axes[0].fill_between(x, low_hit, high_hit, alpha=0.18, color="#1f77b4")
    axes[0].set_xlabel("Relative Window Position (%)")
    axes[0].set_ylabel(f"Hit@{selected_k}")
    axes[0].set_title(f"Relative-Time Hit@{selected_k}")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)

    y_precision = plot_df["precision_at_k"].to_numpy(dtype=float)
    low_precision = plot_df["precision_ci_low"].to_numpy(dtype=float)
    high_precision = plot_df["precision_ci_high"].to_numpy(dtype=float)
    axes[1].plot(x, y_precision, marker="o", linewidth=2.2, color="#ff7f0e")
    axes[1].fill_between(x, low_precision, high_precision, alpha=0.18, color="#ff7f0e")
    axes[1].set_xlabel("Relative Window Position (%)")
    axes[1].set_ylabel(f"Precision@{selected_k}")
    axes[1].set_title(f"Relative-Time Precision@{selected_k}")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_df


def _plot_prefix_curve(
    window_k_frame: pd.DataFrame,
    *,
    selected_k: int,
    output_path: Path,
) -> pd.DataFrame:
    subset = window_k_frame[window_k_frame["k"] == int(selected_k)].copy()
    if subset.empty:
        raise ValueError(f"No rows available for selected_k={selected_k}")

    subset = subset.sort_values(["patient_id", "relative_time", "window_index"]).reset_index(drop=True)
    subset["cum_hit_sum"] = subset.groupby("patient_id")["hit_at_k"].cumsum()
    subset["cum_precision_sum"] = subset.groupby("patient_id")["precision_at_k"].cumsum()
    subset["cum_count"] = subset.groupby("patient_id").cumcount() + 1
    subset["prefix_hit_at_k"] = subset["cum_hit_sum"] / subset["cum_count"]
    subset["prefix_precision_at_k"] = subset["cum_precision_sum"] / subset["cum_count"]

    thresholds = np.linspace(0.0, 1.0, NUM_TIME_BINS + 1)
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        patient_hit_values: List[float] = []
        patient_precision_values: List[float] = []

        for _, patient_df in subset.groupby("patient_id"):
            valid = patient_df[patient_df["relative_time"] <= threshold]
            if valid.empty:
                continue
            patient_hit_values.append(float(valid["prefix_hit_at_k"].iloc[-1]))
            patient_precision_values.append(float(valid["prefix_precision_at_k"].iloc[-1]))

        rows.append(
            {
                "k": int(selected_k),
                "relative_time_pct": float(threshold * 100.0),
                "num_patients": int(len(patient_hit_values)),
                "cohort_mean_prefix_hit_at_k": (
                    float(np.mean(patient_hit_values)) if patient_hit_values else float("nan")
                ),
                "cohort_mean_prefix_precision_at_k": (
                    float(np.mean(patient_precision_values)) if patient_precision_values else float("nan")
                ),
            }
        )

    mean_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for _, patient_df in subset.groupby("patient_id"):
        x = patient_df["relative_time"].to_numpy(dtype=float) * 100.0
        hit_y = patient_df["prefix_hit_at_k"].to_numpy(dtype=float)
        precision_y = patient_df["prefix_precision_at_k"].to_numpy(dtype=float)
        axes[0].plot(x, hit_y, color="#1f77b4", alpha=0.18, linewidth=1.0)
        axes[1].plot(x, precision_y, color="#ff7f0e", alpha=0.18, linewidth=1.0)

    axes[0].plot(
        mean_df["relative_time_pct"].to_numpy(dtype=float),
        mean_df["cohort_mean_prefix_hit_at_k"].to_numpy(dtype=float),
        color="#0b3c8c",
        linewidth=3.0,
        marker="o",
        label=f"Cohort Mean Hit@{selected_k}",
    )
    axes[0].set_xlabel("Relative Time (%)")
    axes[0].set_ylabel(f"Cumulative Hit@{selected_k}")
    axes[0].set_title(f"Prefix Cumulative Hit@{selected_k}")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        mean_df["relative_time_pct"].to_numpy(dtype=float),
        mean_df["cohort_mean_prefix_precision_at_k"].to_numpy(dtype=float),
        color="#b85e00",
        linewidth=3.0,
        marker="o",
        label=f"Cohort Mean Precision@{selected_k}",
    )
    axes[1].set_xlabel("Relative Time (%)")
    axes[1].set_ylabel(f"Cumulative Precision@{selected_k}")
    axes[1].set_title(f"Prefix Cumulative Precision@{selected_k}")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return mean_df


def _plot_time_memory_heatmap(
    window_k_frame: pd.DataFrame,
    *,
    selected_k: int,
    output_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subset = window_k_frame[window_k_frame["k"] == int(selected_k)].copy()
    if subset.empty:
        raise ValueError(f"No rows available for selected_k={selected_k}")
    if subset["memory_depth"].isna().all():
        raise ValueError("Time-memory heatmap requires non-empty memory depth values.")

    bin_codes, bin_labels = _memory_depth_bins(subset["memory_depth"])
    subset["memory_depth_bin"] = bin_codes
    valid = subset.dropna(subset=["memory_depth_bin"]).copy()
    valid["memory_depth_bin"] = valid["memory_depth_bin"].astype(int)

    hit_table = (
        valid.pivot_table(
            index="memory_depth_bin",
            columns="time_bin",
            values="hit_at_k",
            aggfunc="mean",
        )
        .sort_index(ascending=True)
        .reindex(columns=list(range(NUM_TIME_BINS)))
    )
    precision_table = (
        valid.pivot_table(
            index="memory_depth_bin",
            columns="time_bin",
            values="precision_at_k",
            aggfunc="mean",
        )
        .sort_index(ascending=True)
        .reindex(columns=list(range(NUM_TIME_BINS)))
    )
    count_table = (
        valid.pivot_table(
            index="memory_depth_bin",
            columns="time_bin",
            values="hit_at_k",
            aggfunc="count",
        )
        .sort_index(ascending=True)
        .reindex(columns=list(range(NUM_TIME_BINS)))
        .fillna(0)
        .astype(int)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    hit_data = hit_table.to_numpy(dtype=float)
    hit_image = axes[0].imshow(hit_data, aspect="auto", origin="lower", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    hit_cbar = fig.colorbar(hit_image, ax=axes[0])
    hit_cbar.set_label(f"Hit@{selected_k}")

    precision_data = precision_table.to_numpy(dtype=float)
    precision_image = axes[1].imshow(
        precision_data,
        aspect="auto",
        origin="lower",
        cmap="YlOrBr",
        vmin=0.0,
        vmax=1.0,
    )
    precision_cbar = fig.colorbar(precision_image, ax=axes[1])
    precision_cbar.set_label(f"Precision@{selected_k}")

    x_tick_labels = [f"{i * 10}-{(i + 1) * 10}%" for i in range(NUM_TIME_BINS)]
    y_tick_labels = [bin_labels[i] if i < len(bin_labels) else str(i) for i in hit_table.index.tolist()]

    for axis, title in (
        (axes[0], f"Time × Memory Hit@{selected_k}"),
        (axes[1], f"Time × Memory Precision@{selected_k}"),
    ):
        axis.set_xticks(np.arange(NUM_TIME_BINS))
        axis.set_xticklabels(x_tick_labels, rotation=30, ha="right")
        axis.set_yticks(np.arange(len(y_tick_labels)))
        axis.set_yticklabels(y_tick_labels)
        axis.set_xlabel("Relative-Time Decile")
        axis.set_title(title)
    axes[0].set_ylabel("Memory-Depth Bin")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    hit_long = hit_table.reset_index().melt(id_vars=["memory_depth_bin"], var_name="time_bin", value_name="hit_at_k")
    precision_long = precision_table.reset_index().melt(
        id_vars=["memory_depth_bin"], var_name="time_bin", value_name="precision_at_k"
    )
    count_long = count_table.reset_index().melt(id_vars=["memory_depth_bin"], var_name="time_bin", value_name="count")

    merged = hit_long.merge(precision_long, on=["memory_depth_bin", "time_bin"], how="outer")
    merged = merged.merge(count_long, on=["memory_depth_bin", "time_bin"], how="left")
    merged["k"] = int(selected_k)
    merged["memory_depth_label"] = merged["memory_depth_bin"].map(
        lambda value: bin_labels[int(value)] if int(value) < len(bin_labels) else str(value)
    )

    source = valid[
        [
            "patient_id",
            "window_index",
            "k",
            "time_bin",
            "memory_depth",
            "hit_at_k",
            "precision_at_k",
        ]
    ].copy()
    return merged, source


def _plot_patient_level_figures(
    window_k_frame: pd.DataFrame,
    *,
    selected_k: int,
    output_dir: Path,
) -> Tuple[int, int]:
    patient_plot_root = output_dir / "patient_plots"
    patient_plot_root.mkdir(parents=True, exist_ok=True)

    patient_ids = sorted(str(value) for value in window_k_frame["patient_id"].dropna().unique().tolist())
    if not patient_ids:
        return 0, 0

    heatmap_generated = 0
    for patient_id in patient_ids:
        patient_subset = window_k_frame[window_k_frame["patient_id"].astype(str) == str(patient_id)].copy()
        if patient_subset.empty:
            continue
        safe_patient_id = str(patient_id).replace("/", "_")
        patient_dir = patient_plot_root / safe_patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        relative_curve_df = _plot_relative_time_curve(
            patient_subset,
            selected_k=selected_k,
            output_path=patient_dir / "plot1_relative_time_hit_precision_curve.png",
        )
        relative_curve_df.to_csv(patient_dir / "plot1_relative_time_hit_precision_curve.csv", index=False)

        prefix_curve_df = _plot_prefix_curve(
            patient_subset,
            selected_k=selected_k,
            output_path=patient_dir / "plot2_prefix_cumulative_hit_precision_curve.png",
        )
        prefix_curve_df.to_csv(patient_dir / "plot2_prefix_cumulative_hit_precision_curve.csv", index=False)

        try:
            heatmap_long_df, heatmap_source_df = _plot_time_memory_heatmap(
                patient_subset,
                selected_k=selected_k,
                output_path=patient_dir / "plot3_time_memory_hit_precision_heatmap.png",
            )
        except ValueError as exc:
            with open(patient_dir / "plot3_time_memory_hit_precision_heatmap_skipped.txt", "w", encoding="utf-8") as f:
                f.write(str(exc))
        else:
            heatmap_generated += 1
            heatmap_long_df.to_csv(patient_dir / "plot3_time_memory_hit_precision_heatmap.csv", index=False)
            heatmap_source_df.to_csv(patient_dir / "plot3_time_memory_source_rows.csv", index=False)

    return len(patient_ids), int(heatmap_generated)


def _build_patient_relative_time_curve(
    window_k_frame: pd.DataFrame,
    *,
    selected_k: int,
    patient_id: str,
) -> pd.DataFrame:
    subset = window_k_frame[
        (window_k_frame["k"] == int(selected_k)) & (window_k_frame["patient_id"].astype(str) == str(patient_id))
    ].copy()
    if subset.empty:
        raise ValueError(f"No rows available for patient_id={patient_id}, selected_k={selected_k}")

    rows: List[Dict[str, Any]] = []
    for time_bin in range(NUM_TIME_BINS):
        bin_rows = subset[subset["time_bin"] == int(time_bin)]
        hit_values = bin_rows["hit_at_k"].astype(float).tolist()
        precision_values = bin_rows["precision_at_k"].astype(float).tolist()
        rows.append(
            {
                "time_bin": int(time_bin),
                "time_bin_mid_pct": float((time_bin + 0.5) * (100.0 / NUM_TIME_BINS)),
                "n_windows": int(len(bin_rows)),
                "hit_at_k": float(np.mean(hit_values)) if hit_values else float("nan"),
                "precision_at_k": float(np.mean(precision_values)) if precision_values else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("time_bin").reset_index(drop=True)


def _build_patient_prefix_curve(
    window_k_frame: pd.DataFrame,
    *,
    selected_k: int,
    patient_id: str,
) -> pd.DataFrame:
    subset = window_k_frame[
        (window_k_frame["k"] == int(selected_k)) & (window_k_frame["patient_id"].astype(str) == str(patient_id))
    ].copy()
    if subset.empty:
        raise ValueError(f"No rows available for patient_id={patient_id}, selected_k={selected_k}")

    subset = subset.sort_values(["relative_time", "window_index"]).reset_index(drop=True)
    subset["cum_hit_sum"] = subset["hit_at_k"].cumsum()
    subset["cum_precision_sum"] = subset["precision_at_k"].cumsum()
    subset["cum_count"] = subset.index.to_series() + 1
    subset["prefix_hit_at_k"] = subset["cum_hit_sum"] / subset["cum_count"]
    subset["prefix_precision_at_k"] = subset["cum_precision_sum"] / subset["cum_count"]

    thresholds = np.linspace(0.0, 1.0, NUM_TIME_BINS + 1)
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        valid = subset[subset["relative_time"] <= float(threshold)]
        if valid.empty:
            rows.append(
                {
                    "relative_time_pct": float(threshold * 100.0),
                    "prefix_hit_at_k": float("nan"),
                    "prefix_precision_at_k": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "relative_time_pct": float(threshold * 100.0),
                "prefix_hit_at_k": float(valid["prefix_hit_at_k"].iloc[-1]),
                "prefix_precision_at_k": float(valid["prefix_precision_at_k"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def _plot_patient_mode_relative_curve(
    curves_by_mode: Mapping[str, pd.DataFrame],
    *,
    patient_id: str,
    output_path: Path,
) -> pd.DataFrame:
    if not curves_by_mode:
        raise ValueError(f"No mode curves provided for patient_id={patient_id}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rows: List[Dict[str, Any]] = []
    for mode_label, curve_df in curves_by_mode.items():
        ordered = curve_df.sort_values("time_bin").reset_index(drop=True)
        x = ordered["time_bin_mid_pct"].to_numpy(dtype=float)
        hit = ordered["hit_at_k"].to_numpy(dtype=float)
        precision = ordered["precision_at_k"].to_numpy(dtype=float)
        axes[0].plot(x, hit, marker="o", linewidth=2.2, label=str(mode_label))
        axes[1].plot(x, precision, marker="o", linewidth=2.2, label=str(mode_label))
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "patient_id": str(patient_id),
                    "mode": str(mode_label),
                    "time_bin": int(row["time_bin"]),
                    "time_bin_mid_pct": float(row["time_bin_mid_pct"]),
                    "n_windows": int(row["n_windows"]),
                    "hit_at_k": float(row["hit_at_k"]),
                    "precision_at_k": float(row["precision_at_k"]),
                }
            )

    axes[0].set_xlabel("Relative Window Position (%)")
    axes[0].set_ylabel("Hit@K")
    axes[0].set_title(f"Patient {patient_id}: Relative-Time Hit@K by Mode")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Relative Window Position (%)")
    axes[1].set_ylabel("Precision@K")
    axes[1].set_title(f"Patient {patient_id}: Relative-Time Precision@K by Mode")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    output_df = pd.DataFrame(rows)
    if output_df.empty:
        raise ValueError(f"No rows generated for patient_id={patient_id}")
    return output_df


def _plot_patient_mode_prefix_curve(
    curves_by_mode: Mapping[str, pd.DataFrame],
    *,
    patient_id: str,
    output_path: Path,
) -> pd.DataFrame:
    if not curves_by_mode:
        raise ValueError(f"No mode curves provided for patient_id={patient_id}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rows: List[Dict[str, Any]] = []
    for mode_label, curve_df in curves_by_mode.items():
        ordered = curve_df.sort_values("relative_time_pct").reset_index(drop=True)
        x = ordered["relative_time_pct"].to_numpy(dtype=float)
        hit = ordered["prefix_hit_at_k"].to_numpy(dtype=float)
        precision = ordered["prefix_precision_at_k"].to_numpy(dtype=float)
        axes[0].plot(x, hit, marker="o", linewidth=2.2, label=str(mode_label))
        axes[1].plot(x, precision, marker="o", linewidth=2.2, label=str(mode_label))
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "patient_id": str(patient_id),
                    "mode": str(mode_label),
                    "relative_time_pct": float(row["relative_time_pct"]),
                    "prefix_hit_at_k": float(row["prefix_hit_at_k"]),
                    "prefix_precision_at_k": float(row["prefix_precision_at_k"]),
                }
            )

    axes[0].set_xlabel("Relative Time (%)")
    axes[0].set_ylabel("Cumulative Hit@K")
    axes[0].set_title(f"Patient {patient_id}: Prefix Cumulative Hit@K by Mode")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Relative Time (%)")
    axes[1].set_ylabel("Cumulative Precision@K")
    axes[1].set_title(f"Patient {patient_id}: Prefix Cumulative Precision@K by Mode")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    output_df = pd.DataFrame(rows)
    if output_df.empty:
        raise ValueError(f"No rows generated for patient_id={patient_id}")
    return output_df


def _plot_patient_mode_comparison_curves(
    mode_results: Mapping[str, Mapping[str, Any]],
    *,
    output_root: Path,
) -> Tuple[int, int]:
    if not mode_results:
        return 0, 0

    patient_ids: List[str] = sorted(
        {
            str(patient_id)
            for result in mode_results.values()
            for patient_id in result["window_k_frame"]["patient_id"].dropna().astype(str).unique().tolist()
        }
    )
    if not patient_ids:
        return 0, 0

    patient_root = output_root / "patient_mode_plots"
    patient_root.mkdir(parents=True, exist_ok=True)

    plotted_relative = 0
    plotted_prefix = 0
    for patient_id in patient_ids:
        relative_by_mode: Dict[str, pd.DataFrame] = {}
        prefix_by_mode: Dict[str, pd.DataFrame] = {}
        for mode_name, result in mode_results.items():
            mode_frame = result["window_k_frame"]
            selected_k = int(result["selected_k"])
            label = f"{mode_name} (K={selected_k})"
            try:
                relative_by_mode[label] = _build_patient_relative_time_curve(
                    mode_frame,
                    selected_k=selected_k,
                    patient_id=patient_id,
                )
                prefix_by_mode[label] = _build_patient_prefix_curve(
                    mode_frame,
                    selected_k=selected_k,
                    patient_id=patient_id,
                )
            except ValueError:
                continue

        if not relative_by_mode:
            continue

        patient_dir = patient_root / str(patient_id).replace("/", "_")
        patient_dir.mkdir(parents=True, exist_ok=True)

        relative_df = _plot_patient_mode_relative_curve(
            relative_by_mode,
            patient_id=patient_id,
            output_path=patient_dir / "plot1_relative_time_hit_precision_curve_by_mode.png",
        )
        relative_df.to_csv(patient_dir / "plot1_relative_time_hit_precision_curve_by_mode.csv", index=False)
        plotted_relative += 1

        prefix_df = _plot_patient_mode_prefix_curve(
            prefix_by_mode,
            patient_id=patient_id,
            output_path=patient_dir / "plot2_prefix_cumulative_hit_precision_curve_by_mode.png",
        )
        prefix_df.to_csv(patient_dir / "plot2_prefix_cumulative_hit_precision_curve_by_mode.csv", index=False)
        plotted_prefix += 1

    return int(plotted_relative), int(plotted_prefix)


def _plot_relative_time_curve_by_mode(
    curve_by_mode: Mapping[str, pd.DataFrame],
    *,
    output_path: Path,
) -> pd.DataFrame:
    if not curve_by_mode:
        raise ValueError("No mode curves provided for relative-time comparison plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rows: List[Dict[str, Any]] = []
    for mode_label, curve_df in curve_by_mode.items():
        if curve_df.empty:
            continue
        ordered = curve_df.sort_values("time_bin").reset_index(drop=True)
        x = ordered["time_bin_mid_pct"].to_numpy(dtype=float)
        y_hit = ordered["hit_at_k"].to_numpy(dtype=float)
        y_precision = ordered["precision_at_k"].to_numpy(dtype=float)
        axes[0].plot(x, y_hit, marker="o", linewidth=2.2, label=str(mode_label))
        axes[1].plot(x, y_precision, marker="o", linewidth=2.2, label=str(mode_label))
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "mode": str(mode_label),
                    "time_bin": int(row["time_bin"]),
                    "time_bin_mid_pct": float(row["time_bin_mid_pct"]),
                    "n_windows": int(row["n_windows"]),
                    "hit_at_k": float(row["hit_at_k"]),
                    "precision_at_k": float(row["precision_at_k"]),
                }
            )

    axes[0].set_xlabel("Relative Window Position (%)")
    axes[0].set_ylabel("Hit@K")
    axes[0].set_title("Relative-Time Hit@K by Mode")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Relative Window Position (%)")
    axes[1].set_ylabel("Precision@K")
    axes[1].set_title("Relative-Time Precision@K by Mode")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        raise ValueError("No rows generated for relative-time mode comparison plot.")
    return comparison_df


def _plot_prefix_curve_by_mode(
    curve_by_mode: Mapping[str, pd.DataFrame],
    *,
    output_path: Path,
) -> pd.DataFrame:
    if not curve_by_mode:
        raise ValueError("No mode curves provided for prefix comparison plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rows: List[Dict[str, Any]] = []
    for mode_label, curve_df in curve_by_mode.items():
        if curve_df.empty:
            continue
        ordered = curve_df.sort_values("relative_time_pct").reset_index(drop=True)
        x = ordered["relative_time_pct"].to_numpy(dtype=float)
        y_hit = ordered["cohort_mean_prefix_hit_at_k"].to_numpy(dtype=float)
        y_precision = ordered["cohort_mean_prefix_precision_at_k"].to_numpy(dtype=float)
        axes[0].plot(x, y_hit, marker="o", linewidth=2.2, label=str(mode_label))
        axes[1].plot(x, y_precision, marker="o", linewidth=2.2, label=str(mode_label))
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "mode": str(mode_label),
                    "relative_time_pct": float(row["relative_time_pct"]),
                    "num_patients": int(row["num_patients"]),
                    "cohort_mean_prefix_hit_at_k": float(row["cohort_mean_prefix_hit_at_k"]),
                    "cohort_mean_prefix_precision_at_k": float(row["cohort_mean_prefix_precision_at_k"]),
                }
            )

    axes[0].set_xlabel("Relative Time (%)")
    axes[0].set_ylabel("Cumulative Hit@K")
    axes[0].set_title("Prefix Cumulative Hit@K by Mode")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Relative Time (%)")
    axes[1].set_ylabel("Cumulative Precision@K")
    axes[1].set_title("Prefix Cumulative Precision@K by Mode")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        raise ValueError("No rows generated for prefix mode comparison plot.")
    return comparison_df


def _evaluate_single_prediction_dir(
    *,
    prediction_root: Path,
    output_dir: Path,
    gt_source: str,
    oracle_results_dir: Optional[Path],
    matcher_backend: str,
    embedding_model_name: Optional[str],
    embedding_similarity_threshold: Optional[float],
    embedding_device: Optional[str],
    window_stride: Optional[int],
    num_workers: int,
) -> Dict[str, Any]:
    oracle_results_root: Optional[Path]
    if gt_source == GT_SOURCE_ORACLE_REVIEWED_ACTIONS:
        if oracle_results_dir is None:
            raise ValueError("--oracle-results-dir is required when --gt-source=oracle_reviewed_actions")
        oracle_results_root = _ensure_oracle_results_dir(oracle_results_dir)
    else:
        oracle_results_root = None

    backend = str(matcher_backend)
    if backend not in {MATCHER_BACKEND_LLM, MATCHER_BACKEND_EMBEDDING}:
        raise ValueError(f"Unsupported matcher_backend={matcher_backend}")

    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_max_tokens: Optional[int] = None
    if backend == MATCHER_BACKEND_LLM:
        config = get_config()
        llm_provider = config.llm_provider
        llm_model = config.llm_model
        llm_max_tokens = config.llm_max_tokens
        if not llm_provider:
            raise ValueError("Missing llm.provider in config")
        if not llm_model:
            raise ValueError("Missing llm.model in config")
        if llm_max_tokens is None:
            raise ValueError("Missing llm.max_tokens in config")

    if backend == MATCHER_BACKEND_EMBEDDING:
        if embedding_model_name is None:
            raise ValueError("--embedding-model-name is required when --matcher-backend=embedding")
        if embedding_similarity_threshold is None:
            raise ValueError("--embedding-similarity-threshold is required when --matcher-backend=embedding")

    if int(num_workers) < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if window_stride is not None and int(window_stride) < 1:
        raise ValueError(f"window_stride must be >= 1 when provided, got {window_stride}")

    print(f"Prediction directory: {prediction_root}")
    print(f"Ground-truth source: {gt_source}")
    if oracle_results_root is not None:
        print(f"Oracle results directory: {oracle_results_root}")
    if backend == MATCHER_BACKEND_LLM:
        print(
            f"Matcher config: backend={backend}, provider={llm_provider}, model={llm_model}, "
            f"max_tokens={int(llm_max_tokens)}, workers={int(num_workers)}, window_stride={window_stride}"
        )
    else:
        print(
            f"Matcher config: backend={backend}, embedding_model={embedding_model_name}, "
            f"similarity_threshold={float(embedding_similarity_threshold)}, "
            f"device={embedding_device}, workers={int(num_workers)}, window_stride={window_stride}"
        )

    window_frame, prediction_frame, matcher_usage = _load_window_and_prediction_rows(
        prediction_dir=prediction_root,
        gt_source=str(gt_source),
        oracle_results_dir=oracle_results_root,
        matcher_backend=backend,
        llm_provider=str(llm_provider) if llm_provider is not None else None,
        llm_model=str(llm_model) if llm_model is not None else None,
        llm_max_tokens=int(llm_max_tokens) if llm_max_tokens is not None else None,
        embedding_model_name=str(embedding_model_name) if embedding_model_name is not None else None,
        embedding_similarity_threshold=(
            float(embedding_similarity_threshold) if embedding_similarity_threshold is not None else None
        ),
        embedding_device=str(embedding_device) if embedding_device is not None else None,
        window_stride=int(window_stride) if window_stride is not None else None,
        num_workers=int(num_workers),
    )

    if backend == MATCHER_BACKEND_LLM:
        llm_usage = {
            "llm_calls": int(matcher_usage["matcher_calls"]),
            "input_tokens": int(matcher_usage["input_tokens"]),
            "output_tokens": int(matcher_usage["output_tokens"]),
            "total_tokens": int(matcher_usage["total_tokens"]),
        }
    else:
        llm_usage = {
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    k_values = _k_values(window_frame)
    selected_k = int(max(k_values))

    window_k_frame = _expand_window_rows_by_k(window_frame, k_values=k_values)
    metrics_by_k_df = _compute_metrics_by_k(window_k_frame)

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "generated_at": datetime.now().isoformat(),
        "prediction_dir": str(prediction_root),
        "ground_truth_source": str(gt_source),
        "oracle_results_dir": str(oracle_results_root) if oracle_results_root is not None else None,
        "matcher_backend": str(backend),
        "llm_provider": str(llm_provider) if llm_provider is not None else None,
        "llm_model": str(llm_model) if llm_model is not None else None,
        "embedding_model_name": str(embedding_model_name) if embedding_model_name is not None else None,
        "embedding_similarity_threshold": (
            float(embedding_similarity_threshold) if embedding_similarity_threshold is not None else None
        ),
        "embedding_device": str(embedding_device) if embedding_device is not None else None,
        "matcher_usage": matcher_usage,
        "llm_usage": llm_usage,
        "window_stride": int(window_stride) if window_stride is not None else None,
        "num_patients": int(window_k_frame["patient_id"].nunique()),
        "num_windows": int(window_frame.shape[0]),
        "k_values": [int(k) for k in k_values],
        "selected_k_for_plots": int(selected_k),
        "metrics_by_k": metrics_by_k_df.to_dict(orient="records"),
    }
    _json_dump(output_dir / "metrics.json", metrics_payload)

    metrics_by_k_df.to_csv(output_dir / "metrics_by_k.csv", index=False)

    window_export_df = window_frame.copy()
    window_export_df["matched_prediction_indices"] = window_export_df["matched_prediction_indices"].map(
        lambda value: json.dumps(value, ensure_ascii=False)
    )
    window_export_df["matched_pairs"] = window_export_df["matched_pairs"].map(
        lambda value: json.dumps(value, ensure_ascii=False)
    )
    window_export_df.to_csv(output_dir / "window_level_windows.csv", index=False)

    prediction_frame.to_csv(output_dir / "window_level_predictions.csv", index=False)
    window_k_frame.to_csv(output_dir / "window_level_metrics.csv", index=False)

    relative_curve_df = _plot_relative_time_curve(
        window_k_frame,
        selected_k=selected_k,
        output_path=output_dir / "plot1_relative_time_hit_precision_curve.png",
    )
    relative_curve_df.to_csv(output_dir / "plot1_relative_time_hit_precision_curve.csv", index=False)

    prefix_curve_df = _plot_prefix_curve(
        window_k_frame,
        selected_k=selected_k,
        output_path=output_dir / "plot2_prefix_cumulative_hit_precision_curve.png",
    )
    prefix_curve_df.to_csv(output_dir / "plot2_prefix_cumulative_hit_precision_curve.csv", index=False)

    heatmap_long_df, heatmap_source_df = _plot_time_memory_heatmap(
        window_k_frame,
        selected_k=selected_k,
        output_path=output_dir / "plot3_time_memory_hit_precision_heatmap.png",
    )
    heatmap_long_df.to_csv(output_dir / "plot3_time_memory_hit_precision_heatmap.csv", index=False)
    heatmap_source_df.to_csv(output_dir / "plot3_time_memory_source_rows.csv", index=False)

    num_patient_plots, num_patient_heatmaps = _plot_patient_level_figures(
        window_k_frame,
        selected_k=selected_k,
        output_dir=output_dir,
    )

    selected_metrics = metrics_by_k_df[metrics_by_k_df["k"] == int(selected_k)]
    if selected_metrics.empty:
        raise ValueError(f"Missing selected K metrics for k={selected_k}")
    summary_row = selected_metrics.iloc[0]

    print(f"Saved evaluation outputs to: {output_dir}")
    print("Recommendation Evaluation:")
    print(
        f"  K={selected_k} "
        f"micro_hit={float(summary_row['micro_hit_at_k']):.4f} "
        f"micro_precision={float(summary_row['micro_precision_at_k']):.4f} "
        f"macro_hit={float(summary_row['macro_hit_at_k']):.4f} "
        f"macro_precision={float(summary_row['macro_precision_at_k']):.4f} "
        f"(patients={int(summary_row['num_patients'])}, windows={int(summary_row['num_windows'])})"
    )
    if backend == MATCHER_BACKEND_LLM:
        print(
            f"  LLM matcher: provider={llm_provider}, model={llm_model}, "
            f"calls={llm_usage['llm_calls']}, tokens={llm_usage['total_tokens']}"
        )
    else:
        print(
            f"  Embedding matcher: model={embedding_model_name}, "
            f"threshold={float(embedding_similarity_threshold)}, "
            f"device={embedding_device}, calls={matcher_usage['matcher_calls']}"
        )
    print(
        f"  Patient plots: patients={int(num_patient_plots)}, "
        f"heatmaps_generated={int(num_patient_heatmaps)} "
        f"(saved under {output_dir / 'patient_plots'})"
    )

    return {
        "selected_k": int(selected_k),
        "relative_curve_df": relative_curve_df,
        "prefix_curve_df": prefix_curve_df,
        "window_k_frame": window_k_frame,
        "output_dir": output_dir,
        "prediction_root": prediction_root,
    }


def run_evaluation(
    *,
    prediction_dir: Path,
    output_dir: Optional[Path],
    gt_source: str,
    oracle_results_dir: Optional[Path],
    matcher_backend: str,
    embedding_model_name: Optional[str],
    embedding_similarity_threshold: Optional[float],
    embedding_device: Optional[str],
    window_stride: Optional[int],
    num_workers: int,
) -> None:
    if gt_source not in {GT_SOURCE_DATASET_ACTIONS, GT_SOURCE_ORACLE_REVIEWED_ACTIONS}:
        raise ValueError(f"Unsupported gt_source={gt_source}")
    backend = str(matcher_backend).strip()
    if backend not in {MATCHER_BACKEND_LLM, MATCHER_BACKEND_EMBEDDING}:
        raise ValueError(f"Unsupported matcher_backend={matcher_backend}")

    mode_dirs = _discover_recommendation_mode_dirs(prediction_dir)
    if mode_dirs:
        base_output_root = (
            output_dir if output_dir is not None else _default_output_root_from_recommendation_input(prediction_dir)
        )
        output_root = _output_root_with_backend(base_output_root, matcher_backend=backend)
        output_root.mkdir(parents=True, exist_ok=True)
        print("Detected recommendation modes: " + ", ".join(sorted(mode_dirs.keys())) + f" under {prediction_dir}")

        relative_curves_by_mode: Dict[str, pd.DataFrame] = {}
        prefix_curves_by_mode: Dict[str, pd.DataFrame] = {}
        mode_results: Dict[str, Dict[str, Any]] = {}
        for mode in sorted(mode_dirs.keys()):
            prediction_root = mode_dirs[mode]
            mode_output_dir = output_root / mode
            print("")
            print(f"=== Evaluating mode: {mode} ===")
            single_result = _evaluate_single_prediction_dir(
                prediction_root=prediction_root,
                output_dir=mode_output_dir,
                gt_source=str(gt_source),
                oracle_results_dir=oracle_results_dir,
                matcher_backend=str(backend),
                embedding_model_name=str(embedding_model_name) if embedding_model_name is not None else None,
                embedding_similarity_threshold=(
                    float(embedding_similarity_threshold) if embedding_similarity_threshold is not None else None
                ),
                embedding_device=str(embedding_device) if embedding_device is not None else None,
                window_stride=int(window_stride) if window_stride is not None else None,
                num_workers=int(num_workers),
            )
            mode_label = f"{mode} (K={int(single_result['selected_k'])})"
            relative_curves_by_mode[mode_label] = single_result["relative_curve_df"]
            prefix_curves_by_mode[mode_label] = single_result["prefix_curve_df"]
            mode_results[mode] = single_result

        if len(relative_curves_by_mode) > 1:
            combined_relative_df = _plot_relative_time_curve_by_mode(
                relative_curves_by_mode,
                output_path=output_root / "plot1_relative_time_hit_precision_curve_by_mode.png",
            )
            combined_relative_df.to_csv(
                output_root / "plot1_relative_time_hit_precision_curve_by_mode.csv", index=False
            )

        if len(prefix_curves_by_mode) > 1:
            combined_prefix_df = _plot_prefix_curve_by_mode(
                prefix_curves_by_mode,
                output_path=output_root / "plot2_prefix_cumulative_hit_precision_curve_by_mode.png",
            )
            combined_prefix_df.to_csv(
                output_root / "plot2_prefix_cumulative_hit_precision_curve_by_mode.csv", index=False
            )

        patient_relative_count, patient_prefix_count = _plot_patient_mode_comparison_curves(
            mode_results,
            output_root=output_root,
        )

        print("")
        print(
            f"Saved patient mode-comparison plots: relative={patient_relative_count}, "
            f"prefix={patient_prefix_count} under {output_root / 'patient_mode_plots'}"
        )
        print(f"Saved multi-mode evaluation outputs to: {output_root}")
        return

    prediction_root = _ensure_predictions_dir(prediction_dir)
    base_final_output_dir = (
        output_dir if output_dir is not None else _default_output_root_from_recommendation_input(prediction_root)
    )
    final_output_dir = _output_root_with_backend(base_final_output_dir, matcher_backend=backend)
    _evaluate_single_prediction_dir(
        prediction_root=prediction_root,
        output_dir=final_output_dir,
        gt_source=str(gt_source),
        oracle_results_dir=oracle_results_dir,
        matcher_backend=str(backend),
        embedding_model_name=str(embedding_model_name) if embedding_model_name is not None else None,
        embedding_similarity_threshold=(
            float(embedding_similarity_threshold) if embedding_similarity_threshold is not None else None
        ),
        embedding_device=str(embedding_device) if embedding_device is not None else None,
        window_stride=int(window_stride) if window_stride is not None else None,
        num_workers=int(num_workers),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recommendation predictions with configurable matching.")
    parser.add_argument(
        "--prediction-dir",
        type=str,
        required=True,
        help="Recommendation prediction directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where metrics and plots will be saved. "
            "Results are saved under a backend subfolder (<llm|embedding>). "
            "Default for multi-mode input: "
            "evaluation_results/<memory_run_name>/recommendation_experiment/<backend>/<mode>. "
            "Default for single-mode input: "
            "evaluation_results/<memory_run_name>/recommendation_experiment/<mode>/<backend>."
        ),
    )
    parser.add_argument(
        "--gt-source",
        type=str,
        required=True,
        choices=[GT_SOURCE_DATASET_ACTIONS, GT_SOURCE_ORACLE_REVIEWED_ACTIONS],
        help="Ground-truth action source used for evaluation.",
    )
    parser.add_argument(
        "--oracle-results-dir",
        type=str,
        default=None,
        help=(
            "Oracle run directory with patients/*/oracle_predictions.json. "
            "Required when --gt-source=oracle_reviewed_actions."
        ),
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
    parser.add_argument(
        "--matcher-backend",
        type=str,
        default=MATCHER_BACKEND_LLM,
        choices=[MATCHER_BACKEND_LLM, MATCHER_BACKEND_EMBEDDING],
        help="Action matcher backend.",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default="abhinand/MedEmbed-base-v0.1",
        help="Sentence-transformers model name for embedding matcher.",
    )
    parser.add_argument(
        "--embedding-similarity-threshold",
        type=float,
        default=0.8,
        help="Cosine-similarity threshold for embedding matcher.",
    )
    parser.add_argument(
        "--embedding-device",
        type=str,
        default="mps",
        help="Embedding model device (cpu/cuda/mps).",
    )
    args = parser.parse_args()

    run_evaluation(
        prediction_dir=Path(args.prediction_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        gt_source=str(args.gt_source),
        oracle_results_dir=Path(args.oracle_results_dir) if args.oracle_results_dir else None,
        matcher_backend=str(args.matcher_backend),
        embedding_model_name=str(args.embedding_model_name) if args.embedding_model_name else None,
        embedding_similarity_threshold=(
            float(args.embedding_similarity_threshold) if args.embedding_similarity_threshold is not None else None
        ),
        embedding_device=str(args.embedding_device) if args.embedding_device else None,
        window_stride=int(args.window_stride) if args.window_stride is not None else None,
        num_workers=int(args.num_workers),
    )


if __name__ == "__main__":
    main()
