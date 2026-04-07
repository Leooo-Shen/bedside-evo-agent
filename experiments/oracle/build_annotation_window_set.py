"""Build a combined annotation window set from Oracle full_visible outputs.

This script samples windows across patients and exports a single combined pack
for annotation, while preserving source metadata for post-annotation replay.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.oracle.common import extract_overall_label

RUN_PREFIX = "oracle_conditions_"
PRIMARY_STATUS_LABELS: Tuple[str, ...] = ("improving", "stable", "deteriorating")
INSUFFICIENT_STATUS = "insufficient_data"
DEFAULT_CONDITION = "full_visible"
DEFAULT_OUTPUT_ROOT = "oracle_results/oracle-validation"
REQUIRED_PATIENT_FILES: Tuple[str, ...] = (
    "oracle_predictions.json",
    "window_contexts.json",
    "llm_calls.json",
)
REQUIRED_PROMPT_SECTION_KEYS: Tuple[str, ...] = (
    "icu_discharge_summary",
    "icu_trajectory_context_window",
    "previous_events_current_window",
    "current_observation_window",
)
AUTO_K_BUFFER_FACTOR = 1.2


@dataclass(frozen=True)
class CandidateWindow:
    """One candidate window with source payloads attached."""

    uid: str
    patient_id: str
    subject_id: int
    icu_stay_id: int
    true_survived: bool
    outcome: str
    status_label: str
    source_run_id: str
    source_condition: str
    source_window_index: int
    source_window_position: int
    window_start_time: str
    window_end_time: str
    prediction_window: Dict[str, Any]
    context_window: Optional[Dict[str, Any]]
    llm_call: Optional[Dict[str, Any]]


def _json_load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object at {path}")
    return payload


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "survived", "alive"}:
        return True
    if text in {"false", "0", "no", "n", "died", "dead", "deceased"}:
        return False
    return bool(value)


def _safe_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text else INSUFFICIENT_STATUS


def _format_patient_id(subject_id: Any, icu_stay_id: Any) -> str:
    return f"{_safe_int(subject_id)}_{_safe_int(icu_stay_id)}"


def _has_required_patient_outputs(patient_dir: Path) -> bool:
    return all((patient_dir / filename).exists() for filename in REQUIRED_PATIENT_FILES)


def _iter_condition_patient_dirs(run_dir: Path, condition: str) -> Iterable[Path]:
    patients_root = run_dir / "conditions" / condition / "patients"
    if not patients_root.exists() or not patients_root.is_dir():
        return
    for patient_dir in sorted(patients_root.iterdir()):
        if patient_dir.is_dir() and _has_required_patient_outputs(patient_dir):
            yield patient_dir


def _find_latest_complete_run(output_root: Path, condition: str) -> Path:
    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    candidates = sorted(
        [path for path in output_root.glob(f"{RUN_PREFIX}*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        if any(True for _ in _iter_condition_patient_dirs(run_dir, condition)):
            return run_dir
    raise FileNotFoundError(
        f"No complete run found under {output_root} with condition '{condition}' and required patient files."
    )


def _is_valid_oracle_output(oracle_output: Any) -> bool:
    if not isinstance(oracle_output, dict):
        return False
    required = ("patient_assessment", "action_review")
    for key in required:
        if key not in oracle_output:
            return False
    patient_assessment = oracle_output.get("patient_assessment")
    if not isinstance(patient_assessment, dict):
        return False
    overall = patient_assessment.get("overall")
    if not isinstance(overall, dict):
        return False
    label = overall.get("label")
    rationale = overall.get("rationale")
    if not isinstance(label, str) or not label.strip():
        return False
    if not isinstance(rationale, str) or not rationale.strip():
        return False
    action_review = oracle_output.get("action_review")
    if not isinstance(action_review, dict):
        return False
    if not isinstance(action_review.get("evaluations"), list):
        return False
    if not isinstance(action_review.get("red_flags"), list):
        return False
    return True


def _llm_call_sort_key(call: Mapping[str, Any], fallback_index: int) -> Tuple[int, float, str, int]:
    idx = _safe_int(call.get("window_index"), default=10**9)
    if idx < 0:
        idx = 10**9
    try:
        hours = float(call.get("hours_since_admission"))
    except (TypeError, ValueError):
        hours = float("inf")
    timestamp = str(call.get("timestamp") or "")
    return (idx, hours, timestamp, fallback_index)


def _build_llm_call_index(calls_payload: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    calls = calls_payload.get("calls")
    if not isinstance(calls, list):
        return {}
    indexed = list(enumerate(calls))
    indexed.sort(key=lambda item: _llm_call_sort_key(item[1] if isinstance(item[1], dict) else {}, item[0]))
    by_index: Dict[int, Dict[str, Any]] = {}
    for _, call in indexed:
        if not isinstance(call, dict):
            continue
        if str(call.get("step_type") or "") != "oracle_evaluator":
            continue
        idx = _safe_int(call.get("window_index"), default=-(10**9))
        if idx <= -(10**9):
            continue
        by_index[idx] = call
    return by_index


def _build_context_index(context_payload: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    contexts = context_payload.get("window_contexts")
    if not isinstance(contexts, list):
        return {}
    by_index: Dict[int, Dict[str, Any]] = {}
    for pos, item in enumerate(contexts, start=1):
        if not isinstance(item, dict):
            continue
        idx = _safe_int(item.get("window_index"), default=pos)
        by_index[idx] = item
    return by_index


def _resolve_llm_call_for_window(
    *,
    source_window_index: int,
    source_position: int,
    by_index: Mapping[int, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for candidate in (
        source_window_index - 1,
        source_window_index,
        source_position - 1,
        source_position,
    ):
        if candidate in by_index:
            return by_index[candidate]
    return None


def _resolve_context_for_window(
    *,
    source_window_index: int,
    source_position: int,
    by_index: Mapping[int, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for candidate in (
        source_window_index,
        source_window_index - 1,
        source_position,
        source_position - 1,
    ):
        if candidate in by_index:
            return by_index[candidate]
    return None


def _extract_window_times(window_output: Mapping[str, Any]) -> Tuple[str, str]:
    window_meta = window_output.get("window_metadata")
    if not isinstance(window_meta, dict):
        window_meta = {}
    start = str(window_meta.get("window_start_time") or window_output.get("current_window_start") or "")
    end = str(window_meta.get("window_end_time") or window_output.get("current_window_end") or "")
    return start, end


def _load_candidate_windows(run_dir: Path, condition: str) -> Tuple[List[CandidateWindow], Dict[str, Any]]:
    run_state_path = run_dir / "run_state.json"
    run_state = _json_load(run_state_path) if run_state_path.exists() else {}
    run_id = str(run_state.get("run_id") or run_dir.name)

    candidates: List[CandidateWindow] = []
    skipped_invalid = 0
    skipped_missing_oracle = 0
    skipped_missing_context = 0
    skipped_missing_call = 0

    for patient_dir in _iter_condition_patient_dirs(run_dir, condition):
        predictions = _json_load(patient_dir / "oracle_predictions.json")
        window_contexts = _json_load(patient_dir / "window_contexts.json")
        llm_calls = _json_load(patient_dir / "llm_calls.json")

        subject_id = _safe_int(predictions.get("subject_id"))
        icu_stay_id = _safe_int(predictions.get("icu_stay_id"))
        patient_id = _format_patient_id(subject_id, icu_stay_id)

        trajectory_meta = predictions.get("trajectory_metadata")
        if not isinstance(trajectory_meta, dict):
            trajectory_meta = {}
        true_survived = _safe_bool(
            trajectory_meta.get("true_survived", trajectory_meta.get("survived")),
            default=False,
        )
        outcome = "survived" if true_survived else "died"

        context_by_index = _build_context_index(window_contexts)
        llm_by_index = _build_llm_call_index(llm_calls)
        window_outputs = predictions.get("window_outputs")
        if not isinstance(window_outputs, list):
            continue

        for pos, window_output in enumerate(window_outputs, start=1):
            if not isinstance(window_output, dict):
                continue
            oracle_output = window_output.get("oracle_output")
            if oracle_output is None:
                skipped_missing_oracle += 1
                continue
            if not _is_valid_oracle_output(oracle_output):
                skipped_invalid += 1
                continue
            source_window_index = _safe_int(window_output.get("window_index"), default=pos)
            start_time, end_time = _extract_window_times(window_output)
            status = _safe_status(extract_overall_label(oracle_output))

            context_window = _resolve_context_for_window(
                source_window_index=source_window_index,
                source_position=pos,
                by_index=context_by_index,
            )
            if context_window is None:
                skipped_missing_context += 1

            llm_call = _resolve_llm_call_for_window(
                source_window_index=source_window_index,
                source_position=pos,
                by_index=llm_by_index,
            )
            if llm_call is None:
                skipped_missing_call += 1

            uid = f"{patient_id}_w{source_window_index}"
            candidates.append(
                CandidateWindow(
                    uid=uid,
                    patient_id=patient_id,
                    subject_id=subject_id,
                    icu_stay_id=icu_stay_id,
                    true_survived=true_survived,
                    outcome=outcome,
                    status_label=status,
                    source_run_id=run_id,
                    source_condition=condition,
                    source_window_index=source_window_index,
                    source_window_position=pos,
                    window_start_time=start_time,
                    window_end_time=end_time,
                    prediction_window=window_output,
                    context_window=context_window,
                    llm_call=llm_call,
                )
            )

    metadata = {
        "run_id": run_id,
        "skipped_invalid_windows": skipped_invalid,
        "skipped_missing_oracle": skipped_missing_oracle,
        "skipped_missing_context": skipped_missing_context,
        "skipped_missing_llm_call": skipped_missing_call,
    }
    return candidates, metadata


def _group_candidates_by_patient(candidates: Sequence[CandidateWindow]) -> Dict[str, List[CandidateWindow]]:
    grouped: Dict[str, List[CandidateWindow]] = defaultdict(list)
    for item in candidates:
        grouped[item.patient_id].append(item)
    for patient_id in grouped:
        grouped[patient_id].sort(key=lambda x: (x.source_window_index, x.source_window_position))
    return dict(grouped)


def _sample_balanced_patients(
    *,
    patients_by_outcome: Mapping[str, Sequence[str]],
    rng: random.Random,
    patients_per_outcome: Optional[int],
) -> Dict[str, List[str]]:
    survived = sorted(patients_by_outcome.get("survived", []))
    died = sorted(patients_by_outcome.get("died", []))
    if not survived or not died:
        raise ValueError("Need both survived and died patients for 50/50 patient sampling.")

    max_balanced = min(len(survived), len(died))
    n_each = max_balanced if patients_per_outcome is None else int(patients_per_outcome)
    if n_each <= 0:
        raise ValueError("patients_per_outcome must be > 0 when provided.")
    if n_each > max_balanced:
        raise ValueError(
            f"patients_per_outcome={n_each} exceeds available balanced limit={max_balanced} "
            f"(survived={len(survived)}, died={len(died)})."
        )

    return {
        "survived": sorted(rng.sample(survived, n_each)),
        "died": sorted(rng.sample(died, n_each)),
    }


def _resolve_k_per_patient(
    *,
    grouped_by_patient: Mapping[str, Sequence[CandidateWindow]],
    selected_patient_ids: Sequence[str],
    target_windows: int,
    k_per_patient: Optional[int],
    buffer_factor: float,
) -> Tuple[int, bool, int]:
    counts = [len(grouped_by_patient[patient_id]) for patient_id in selected_patient_ids]
    max_count = max(counts) if counts else 0
    if max_count <= 0:
        raise ValueError("Selected patients have no candidate windows.")

    if k_per_patient is not None:
        k = int(k_per_patient)
        if k <= 0:
            raise ValueError("k_per_patient must be > 0.")
        initial_size = sum(min(k, count) for count in counts)
        return k, False, initial_size

    target_initial_pool = int(math.ceil(float(target_windows) * float(buffer_factor)))
    k = 1
    while k < max_count and sum(min(k, count) for count in counts) < target_initial_pool:
        k += 1
    initial_size = sum(min(k, count) for count in counts)
    return k, True, initial_size


def _sample_initial_pool(
    *,
    grouped_by_patient: Mapping[str, Sequence[CandidateWindow]],
    selected_patient_ids: Sequence[str],
    k_per_patient: int,
    rng: random.Random,
) -> Tuple[List[CandidateWindow], Dict[str, str]]:
    sampled: List[CandidateWindow] = []
    stage_by_uid: Dict[str, str] = {}
    for patient_id in selected_patient_ids:
        patient_windows = list(grouped_by_patient[patient_id])
        take_n = min(k_per_patient, len(patient_windows))
        if take_n <= 0:
            continue
        picked = rng.sample(patient_windows, take_n)
        for item in picked:
            sampled.append(item)
            stage_by_uid[item.uid] = "initial"
    return sampled, stage_by_uid


def _top_up_with_remaining_windows(
    *,
    current_pool: Sequence[CandidateWindow],
    grouped_by_patient: Mapping[str, Sequence[CandidateWindow]],
    selected_patient_ids: Sequence[str],
    target_windows: int,
    rng: random.Random,
    stage_by_uid: Dict[str, str],
) -> Tuple[List[CandidateWindow], int]:
    pool = list(current_pool)
    if len(pool) >= target_windows:
        return pool, 0

    selected_uids = {item.uid for item in pool}
    extras: List[CandidateWindow] = []
    for patient_id in selected_patient_ids:
        for item in grouped_by_patient[patient_id]:
            if item.uid not in selected_uids:
                extras.append(item)
    need = target_windows - len(pool)
    if len(extras) < need:
        pool.extend(extras)
        for item in extras:
            stage_by_uid[item.uid] = "fallback_top_up"
        return pool, len(extras)

    picked = rng.sample(extras, need)
    for item in picked:
        pool.append(item)
        stage_by_uid[item.uid] = "fallback_top_up"
    return pool, need


def _build_status_buckets(windows: Sequence[CandidateWindow], rng: random.Random) -> Dict[str, List[CandidateWindow]]:
    buckets: Dict[str, List[CandidateWindow]] = defaultdict(list)
    for item in windows:
        buckets[item.status_label].append(item)
    for label in list(buckets.keys()):
        rng.shuffle(buckets[label])
    return dict(buckets)


def _pop_next_balanced_primary(
    *,
    buckets: Mapping[str, List[CandidateWindow]],
    selected_counts: Counter,
    rng: random.Random,
) -> Optional[CandidateWindow]:
    available = [label for label in PRIMARY_STATUS_LABELS if buckets.get(label)]
    if not available:
        return None
    min_count = min(selected_counts.get(label, 0) for label in available)
    candidates = [label for label in available if selected_counts.get(label, 0) == min_count]
    if len(candidates) > 1:
        # Prefer labels with more remaining windows to avoid premature depletion.
        max_remaining = max(len(buckets[label]) for label in candidates)
        candidates = [label for label in candidates if len(buckets[label]) == max_remaining]
    label = rng.choice(candidates)
    return buckets[label].pop()


def _pop_next_by_proportional_deficit(
    *,
    buckets: Mapping[str, List[CandidateWindow]],
    selected_counts: Counter,
    desired_counts: Mapping[str, float],
    rng: random.Random,
) -> Optional[CandidateWindow]:
    available = [label for label, items in buckets.items() if items]
    if not available:
        return None
    deficits: Dict[str, float] = {
        label: float(desired_counts.get(label, 0.0)) - float(selected_counts.get(label, 0)) for label in available
    }
    max_deficit = max(deficits.values())
    candidates = [label for label in available if deficits[label] == max_deficit]
    if len(candidates) > 1:
        max_remaining = max(len(buckets[label]) for label in candidates)
        candidates = [label for label in candidates if len(buckets[label]) == max_remaining]
    label = rng.choice(candidates)
    return buckets[label].pop()


def _sample_final_windows(
    *,
    candidate_pool: Sequence[CandidateWindow],
    target_windows: int,
    rng: random.Random,
) -> Tuple[List[CandidateWindow], Dict[str, Any]]:
    if len(candidate_pool) < target_windows:
        raise ValueError(f"Candidate pool ({len(candidate_pool)}) is smaller than target_windows ({target_windows}).")

    buckets = _build_status_buckets(candidate_pool, rng)
    available_counts = Counter({label: len(items) for label, items in buckets.items()})
    available_total = sum(available_counts.values())
    if available_total < target_windows:
        raise ValueError(f"Not enough windows after bucketing ({available_total}) for target ({target_windows}).")

    selected: List[CandidateWindow] = []
    selected_counts: Counter = Counter()

    insuff_available = int(available_counts.get(INSUFFICIENT_STATUS, 0))
    insuff_quota = int(round(float(target_windows) * float(insuff_available) / float(available_total)))
    insuff_quota = min(insuff_quota, insuff_available)
    primary_target = max(0, target_windows - insuff_quota)

    while len(selected) < primary_target:
        item = _pop_next_balanced_primary(buckets=buckets, selected_counts=selected_counts, rng=rng)
        if item is None:
            break
        selected.append(item)
        selected_counts[item.status_label] += 1

    insuff_bucket = buckets.get(INSUFFICIENT_STATUS, [])
    insuff_to_take = min(insuff_quota, target_windows - len(selected), len(insuff_bucket))
    for _ in range(insuff_to_take):
        item = insuff_bucket.pop()
        selected.append(item)
        selected_counts[item.status_label] += 1

    desired_counts: Dict[str, float] = {}
    for label, count in available_counts.items():
        desired_counts[label] = float(target_windows) * (float(count) / float(available_total))

    while len(selected) < target_windows:
        item = _pop_next_by_proportional_deficit(
            buckets=buckets,
            selected_counts=selected_counts,
            desired_counts=desired_counts,
            rng=rng,
        )
        if item is None:
            break
        selected.append(item)
        selected_counts[item.status_label] += 1

    if len(selected) < target_windows:
        raise ValueError(f"Unable to reach target_windows={target_windows} after final sampling; got {len(selected)}.")

    stats = {
        "candidate_pool_size": len(candidate_pool),
        "candidate_status_distribution": dict(available_counts),
        "insufficient_quota_by_proportion": insuff_quota,
        "insufficient_selected": int(selected_counts.get(INSUFFICIENT_STATUS, 0)),
        "final_status_distribution": dict(selected_counts),
    }
    return selected, stats


def _sort_windows_patient_contiguous(
    windows: Sequence[CandidateWindow], selected_patient_order: Sequence[str]
) -> List[CandidateWindow]:
    by_patient: Dict[str, List[CandidateWindow]] = defaultdict(list)
    for item in windows:
        by_patient[item.patient_id].append(item)

    ordered: List[CandidateWindow] = []
    for patient_id in selected_patient_order:
        group = by_patient.get(patient_id, [])
        if not group:
            continue
        group.sort(key=lambda x: (x.window_start_time, x.source_window_index, x.source_window_position))
        ordered.extend(group)

    remaining_ids = sorted(set(by_patient.keys()) - set(selected_patient_order))
    for patient_id in remaining_ids:
        group = by_patient[patient_id]
        group.sort(key=lambda x: (x.window_start_time, x.source_window_index, x.source_window_position))
        ordered.extend(group)
    return ordered


def _build_default_prompt_sections() -> Dict[str, str]:
    return {key: "" for key in REQUIRED_PROMPT_SECTION_KEYS}


def _ensure_prompt_sections(payload: Mapping[str, Any]) -> Dict[str, str]:
    source = payload.get("prompt_sections")
    if not isinstance(source, dict):
        return _build_default_prompt_sections()
    out = {}
    for key in REQUIRED_PROMPT_SECTION_KEYS:
        out[key] = str(source.get(key) or "")
    return out


def _merge_window_outputs(
    *,
    selected: Sequence[CandidateWindow],
    pack_id: str,
    source_run_dir: Path,
    source_condition: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    prediction_windows: List[Dict[str, Any]] = []
    context_windows: List[Dict[str, Any]] = []
    llm_calls: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    history_hours: Optional[float] = None
    future_hours: Optional[float] = None
    for item in selected:
        if history_hours is not None and future_hours is not None:
            break
        if item.context_window and isinstance(item.context_window, dict):
            value = item.context_window.get("window_metadata")
            if isinstance(value, dict):
                if history_hours is None:
                    try:
                        history_hours = float(value.get("history_hours"))
                    except (TypeError, ValueError):
                        history_hours = None
                if future_hours is None:
                    try:
                        future_hours = float(value.get("future_hours"))
                    except (TypeError, ValueError):
                        future_hours = None
            if history_hours is None:
                try:
                    history_hours = float(item.context_window.get("history_hours"))
                except (TypeError, ValueError):
                    history_hours = None
            if future_hours is None:
                try:
                    future_hours = float(item.context_window.get("future_hours"))
                except (TypeError, ValueError):
                    future_hours = None

    for new_index, item in enumerate(selected, start=1):
        pred = copy.deepcopy(item.prediction_window)
        pred.pop("annotation", None)
        pred["window_index"] = new_index
        pred_meta = pred.get("window_metadata")
        if not isinstance(pred_meta, dict):
            pred_meta = {}
        pred_meta.update(
            {
                "source_subject_id": item.subject_id,
                "source_icu_stay_id": item.icu_stay_id,
                "source_patient_id": item.patient_id,
                "source_window_index": item.source_window_index,
                "source_window_position": item.source_window_position,
                "source_condition": source_condition,
                "source_run_id": item.source_run_id,
                "source_true_survived": item.true_survived,
            }
        )
        pred["window_metadata"] = pred_meta
        prediction_windows.append(pred)

        context_payload = copy.deepcopy(item.context_window) if item.context_window else {}
        context_payload["window_index"] = new_index
        context_meta = context_payload.get("window_metadata")
        if not isinstance(context_meta, dict):
            context_meta = {}
        context_meta.update(
            {
                "source_subject_id": item.subject_id,
                "source_icu_stay_id": item.icu_stay_id,
                "source_patient_id": item.patient_id,
                "source_window_index": item.source_window_index,
                "source_window_position": item.source_window_position,
                "source_condition": source_condition,
                "source_run_id": item.source_run_id,
                "source_true_survived": item.true_survived,
            }
        )
        context_payload["window_metadata"] = context_meta
        if not isinstance(context_payload.get("history_events"), list):
            context_payload["history_events"] = []
        if not isinstance(context_payload.get("current_events"), list):
            raw_events = pred.get("raw_current_events")
            context_payload["current_events"] = raw_events if isinstance(raw_events, list) else []
        if not isinstance(context_payload.get("future_events"), list):
            context_payload["future_events"] = []
        context_payload["prompt_sections"] = _ensure_prompt_sections(context_payload)
        context_windows.append(context_payload)

        call_payload = copy.deepcopy(item.llm_call) if item.llm_call else {}
        call_payload["step_type"] = str(call_payload.get("step_type") or "oracle_evaluator")
        call_payload["window_index"] = new_index - 1
        call_payload["annotation_window_index"] = new_index
        call_payload["source_subject_id"] = item.subject_id
        call_payload["source_icu_stay_id"] = item.icu_stay_id
        call_payload["source_patient_id"] = item.patient_id
        call_payload["source_window_index"] = item.source_window_index
        call_payload["source_window_position"] = item.source_window_position
        call_payload["source_condition"] = source_condition
        call_payload["source_run_id"] = item.source_run_id
        if "parsed_response" not in call_payload:
            call_payload["parsed_response"] = copy.deepcopy(pred.get("oracle_output", {}))
        llm_calls.append(call_payload)

        manifest_rows.append(
            {
                "annotation_window_index": new_index,
                "status_label": item.status_label,
                "outcome": item.outcome,
                "true_survived": item.true_survived,
                "subject_id": item.subject_id,
                "icu_stay_id": item.icu_stay_id,
                "patient_id": item.patient_id,
                "source_window_index": item.source_window_index,
                "source_window_position": item.source_window_position,
                "window_start_time": item.window_start_time,
                "window_end_time": item.window_end_time,
                "source_condition": source_condition,
                "source_run_id": item.source_run_id,
                "has_context": bool(item.context_window),
                "has_llm_call": bool(item.llm_call),
                "has_prompt": bool(item.llm_call and str(item.llm_call.get("prompt") or "").strip()),
                "has_parsed_response": bool(item.llm_call and isinstance(item.llm_call.get("parsed_response"), dict)),
            }
        )

    predictions_payload: Dict[str, Any] = {
        "run_id": pack_id,
        "generated_at": datetime.now().isoformat(),
        "subject_id": "annotation_pack",
        "icu_stay_id": pack_id,
        "trajectory_metadata": {
            "survived": None,
            "combined_pack": True,
            "source_run_dir": str(source_run_dir),
            "source_condition": source_condition,
            "source_run_id": selected[0].source_run_id if selected else "",
            "num_source_patients": len({item.patient_id for item in selected}),
        },
        "num_windows_requested": len(selected),
        "num_windows_evaluated": len(selected),
        "window_outputs": prediction_windows,
    }

    contexts_payload: Dict[str, Any] = {
        "run_id": pack_id,
        "generated_at": datetime.now().isoformat(),
        "subject_id": "annotation_pack",
        "icu_stay_id": pack_id,
        "history_hours": history_hours,
        "future_hours": future_hours,
        "window_contexts": context_windows,
        "combined_pack_metadata": {
            "source_run_dir": str(source_run_dir),
            "source_condition": source_condition,
            "source_run_id": selected[0].source_run_id if selected else "",
        },
    }

    llm_payload: Dict[str, Any] = {
        "patient_id": "annotation_pack",
        "agent_type": "oracle",
        "llm_provider": "mixed_or_source",
        "llm_model": "mixed_or_source",
        "include_icu_outcome_in_prompt": None,
        "prompt_outcome_mode": "mixed",
        "pipeline_agents": [
            {"name": "oracle_evaluator", "used": True},
        ],
        "total_calls": len(llm_calls),
        "calls": llm_calls,
        "combined_pack_metadata": {
            "source_run_dir": str(source_run_dir),
            "source_condition": source_condition,
            "source_run_id": selected[0].source_run_id if selected else "",
        },
    }
    return predictions_payload, contexts_payload, llm_payload, manifest_rows


def _write_manifest_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def build_annotation_window_set(
    *,
    run_dir: Path,
    condition: str = DEFAULT_CONDITION,
    target_windows: int = 500,
    seed: int = 1,
    patients_per_outcome: Optional[int] = None,
    k_per_patient: Optional[int] = None,
    output_dir: Optional[Path] = None,
    buffer_factor: float = AUTO_K_BUFFER_FACTOR,
) -> Path:
    if target_windows <= 0:
        raise ValueError("target_windows must be > 0.")
    if buffer_factor <= 0:
        raise ValueError("buffer_factor must be > 0.")

    rng = random.Random(int(seed))
    run_dir = run_dir.resolve()
    candidates, load_meta = _load_candidate_windows(run_dir, condition)
    if not candidates:
        raise ValueError(f"No valid candidate windows found for run={run_dir}, condition={condition}.")

    grouped = _group_candidates_by_patient(candidates)
    patients_by_outcome: Dict[str, List[str]] = {"survived": [], "died": []}
    for patient_id, windows in grouped.items():
        if not windows:
            continue
        outcome = windows[0].outcome
        if outcome not in {"survived", "died"}:
            continue
        patients_by_outcome[outcome].append(patient_id)

    selected_by_outcome = _sample_balanced_patients(
        patients_by_outcome=patients_by_outcome,
        rng=rng,
        patients_per_outcome=patients_per_outcome,
    )
    selected_patient_ids = selected_by_outcome["survived"] + selected_by_outcome["died"]

    resolved_k, k_auto_selected, expected_initial_size = _resolve_k_per_patient(
        grouped_by_patient=grouped,
        selected_patient_ids=selected_patient_ids,
        target_windows=target_windows,
        k_per_patient=k_per_patient,
        buffer_factor=buffer_factor,
    )

    initial_pool, stage_by_uid = _sample_initial_pool(
        grouped_by_patient=grouped,
        selected_patient_ids=selected_patient_ids,
        k_per_patient=resolved_k,
        rng=rng,
    )

    candidate_pool, fallback_added = _top_up_with_remaining_windows(
        current_pool=initial_pool,
        grouped_by_patient=grouped,
        selected_patient_ids=selected_patient_ids,
        target_windows=target_windows,
        rng=rng,
        stage_by_uid=stage_by_uid,
    )
    if len(candidate_pool) < target_windows:
        raise ValueError(
            "Unable to reach target window count from selected patients. "
            f"target={target_windows}, available_from_selected_patients={len(candidate_pool)}. "
            "Increase patients_per_outcome or use a larger source run."
        )

    final_windows, final_sampling_stats = _sample_final_windows(
        candidate_pool=candidate_pool,
        target_windows=target_windows,
        rng=rng,
    )
    final_windows = _sort_windows_patient_contiguous(final_windows, selected_patient_ids)

    run_id = str(load_meta.get("run_id") or run_dir.name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pack_id = f"{condition}_annotation_set_{timestamp}"
    if output_dir is None:
        output_dir = run_dir / "annotation_sets" / pack_id
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_payload, contexts_payload, llm_payload, manifest_rows = _merge_window_outputs(
        selected=final_windows,
        pack_id=pack_id,
        source_run_dir=run_dir,
        source_condition=condition,
    )

    stage_counter = Counter(stage_by_uid.get(item.uid, "unknown") for item in final_windows)
    manifest_rows_enriched: List[Dict[str, Any]] = []
    for row, item in zip(manifest_rows, final_windows):
        copied = dict(row)
        copied["selection_stage"] = stage_by_uid.get(item.uid, "unknown")
        manifest_rows_enriched.append(copied)

    _json_dump(output_dir / "oracle_predictions.json", predictions_payload)
    _json_dump(output_dir / "window_contexts.json", contexts_payload)
    _json_dump(output_dir / "llm_calls.json", llm_payload)
    _write_manifest_csv(output_dir / "sampling_manifest.csv", manifest_rows_enriched)

    available_status_counts = Counter(item.status_label for item in candidates)
    initial_status_counts = Counter(item.status_label for item in initial_pool)
    candidate_status_counts = Counter(item.status_label for item in candidate_pool)
    final_status_counts = Counter(item.status_label for item in final_windows)
    final_outcome_counts = Counter(item.outcome for item in final_windows)
    per_patient_counts = Counter(item.patient_id for item in final_windows)

    primary_counts = [final_status_counts.get(label, 0) for label in PRIMARY_STATUS_LABELS]
    primary_balance_gap = (max(primary_counts) - min(primary_counts)) if primary_counts else 0

    missing_llm_calls = sum(1 for item in final_windows if not item.llm_call)
    missing_context_windows = sum(1 for item in final_windows if not item.context_window)

    statistics_payload: Dict[str, Any] = {
        "pack_id": pack_id,
        "generated_at": datetime.now().isoformat(),
        "source_run_dir": str(run_dir),
        "source_run_id": run_id,
        "source_condition": condition,
        "seed": int(seed),
        "target_windows": int(target_windows),
        "patients_per_outcome": (
            int(patients_per_outcome) if patients_per_outcome is not None else len(selected_by_outcome["survived"])
        ),
        "k_per_patient": int(resolved_k),
        "k_per_patient_auto_selected": bool(k_auto_selected),
        "k_auto_buffer_factor": float(buffer_factor),
        "selected_patients": {
            "survived": selected_by_outcome["survived"],
            "died": selected_by_outcome["died"],
        },
        "available_patients": {
            "survived": len(patients_by_outcome.get("survived", [])),
            "died": len(patients_by_outcome.get("died", [])),
            "total": len(grouped),
        },
        "available_windows": {
            "total": len(candidates),
            "status_distribution": dict(available_status_counts),
        },
        "initial_pool": {
            "target_size_estimate": int(expected_initial_size),
            "actual_size": len(initial_pool),
            "status_distribution": dict(initial_status_counts),
        },
        "candidate_pool": {
            "size": len(candidate_pool),
            "status_distribution": dict(candidate_status_counts),
            "fallback_windows_added": int(fallback_added),
        },
        "final_selection": {
            "size": len(final_windows),
            "status_distribution": dict(final_status_counts),
            "outcome_distribution": dict(final_outcome_counts),
            "per_patient_counts": dict(sorted(per_patient_counts.items())),
            "selection_stage_distribution": dict(stage_counter),
            "primary_balance_gap": int(primary_balance_gap),
            "missing_llm_calls": int(missing_llm_calls),
            "missing_context_windows": int(missing_context_windows),
        },
        "status_balancing": final_sampling_stats,
        "loader_warnings": load_meta,
    }
    _json_dump(output_dir / "sampling_statistics.json", statistics_payload)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a combined Oracle annotation window set from full_visible condition outputs."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to one oracle_conditions_* run directory. If omitted, auto-select latest complete run.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory to search runs when --run-dir is omitted (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the combined annotation pack. Defaults to <run_dir>/annotation_sets/<timestamped_pack>.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=DEFAULT_CONDITION,
        help=f"Condition name under run_dir/conditions (default: {DEFAULT_CONDITION}).",
    )
    parser.add_argument(
        "--target-windows",
        type=int,
        default=500,
        help="Target number of windows in final annotation pack (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for deterministic sampling (default: 1).",
    )
    parser.add_argument(
        "--patients-per-outcome",
        type=int,
        default=None,
        help="Optional fixed number of patients to sample from each outcome bucket.",
    )
    parser.add_argument(
        "--k-per-patient",
        type=int,
        default=None,
        help="Optional fixed per-patient sample size for stage-1 sampling.",
    )
    parser.add_argument(
        "--auto-k-buffer-factor",
        type=float,
        default=AUTO_K_BUFFER_FACTOR,
        help=f"Buffer factor used when auto-searching k_per_patient (default: {AUTO_K_BUFFER_FACTOR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    condition = str(args.condition).strip() or DEFAULT_CONDITION
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = _find_latest_complete_run(Path(args.output_root).expanduser().resolve(), condition=condition)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    pack_dir = build_annotation_window_set(
        run_dir=run_dir,
        condition=condition,
        target_windows=int(args.target_windows),
        seed=int(args.seed),
        patients_per_outcome=args.patients_per_outcome,
        k_per_patient=args.k_per_patient,
        output_dir=output_dir,
        buffer_factor=float(args.auto_k_buffer_factor),
    )

    print("=" * 80)
    print("ANNOTATION WINDOW SET COMPLETE")
    print("=" * 80)
    print(f"Source run: {run_dir}")
    print(f"Condition: {condition}")
    print(f"Output: {pack_dir}")
    print("Files:")
    print("  - oracle_predictions.json")
    print("  - window_contexts.json")
    print("  - llm_calls.json")
    print("  - sampling_manifest.csv")
    print("  - sampling_statistics.json")


if __name__ == "__main__":
    main()
