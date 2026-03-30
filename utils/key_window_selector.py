"""Lightweight rule-based selector for key ICU time windows."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

HIGH_IMPACT_EVENT_CODES: Tuple[str, ...] = (
    "DRUG_START",
    "DRUG_PRESCRIPTION",
    "PROCEDURE",
    "BODY_INPUT",
    "TRANSFER",
    "LAB_TEST",
    "DIAGNOSIS",
)

VITAL_SNAPSHOT_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "hr": ("heart rate",),
    "rr": ("respiratory rate",),
    "spo2": ("o2 saturation", "spo2"),
    "sbp": ("blood pressure systolic", "art bp systolic"),
    "map": ("mean arterial pressure", "art bp mean", "map"),
}

VITAL_JUMP_THRESHOLDS: Dict[str, float] = {
    "hr": 20.0,
    "rr": 8.0,
    "spo2": 5.0,
    "sbp": 20.0,
    "map": 15.0,
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_optional_float(value: Any) -> Optional[float]:
    parsed = _safe_float(value)
    return parsed


def _window_hours_since_admission(window: Mapping[str, Any]) -> Optional[float]:
    direct = _to_optional_float(window.get("hours_since_admission"))
    if direct is not None:
        return direct

    metadata = window.get("window_metadata")
    if isinstance(metadata, Mapping):
        return _to_optional_float(metadata.get("hours_since_admission"))
    return None


def _window_current_events(window: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    for key in ("current_events", "raw_current_events"):
        events = window.get(key)
        if isinstance(events, list):
            return [item for item in events if isinstance(item, Mapping)]
    return []


def _event_code(event: Mapping[str, Any]) -> str:
    return str(event.get("code") or "").strip().upper()


def _event_signature(event: Mapping[str, Any]) -> Tuple[str, str]:
    return (
        str(event.get("code") or "").strip().upper(),
        str(event.get("code_specifics") or "").strip().lower(),
    )


def _is_alarm_like(event: Mapping[str, Any]) -> bool:
    code = _event_code(event)
    specifics = str(event.get("code_specifics") or "").strip().lower()
    text = str(event.get("text_value") or "").strip().lower()

    if "alarm" in specifics or "parameters checked" in specifics:
        return True
    if code == "OTHER_EVENT" and ("alarm" in text or "parameter" in text):
        return True
    return False


def _is_critical_vital(event: Mapping[str, Any]) -> bool:
    if _event_code(event) != "VITALS":
        return False

    specifics = str(event.get("code_specifics") or "").strip().lower()
    value = _safe_float(event.get("numeric_value"))
    if value is None:
        return False

    if "heart rate" in specifics and (value < 45.0 or value > 140.0):
        return True
    if "respiratory rate" in specifics and (value < 8.0 or value > 30.0):
        return True
    if ("o2 saturation" in specifics or "spo2" in specifics) and value < 90.0:
        return True
    if ("blood pressure systolic" in specifics or "art bp systolic" in specifics) and value < 90.0:
        return True
    if ("mean arterial pressure" in specifics or "art bp mean" in specifics or specifics == "map") and value < 65.0:
        return True
    return False


def _build_vital_snapshot(events: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    snapshot: Dict[str, float] = {}
    for event in events:
        if _event_code(event) != "VITALS":
            continue

        specifics = str(event.get("code_specifics") or "").strip().lower()
        value = _safe_float(event.get("numeric_value"))
        if value is None:
            continue

        for vital_name, patterns in VITAL_SNAPSHOT_PATTERNS.items():
            if any(pattern in specifics for pattern in patterns):
                snapshot[vital_name] = value
                break
    return snapshot


def _vital_jump_count(previous: Mapping[str, float], current: Mapping[str, float]) -> int:
    if not previous:
        return 0

    jump_count = 0
    for vital_name, threshold in VITAL_JUMP_THRESHOLDS.items():
        if vital_name not in previous or vital_name not in current:
            continue
        if abs(float(current[vital_name]) - float(previous[vital_name])) >= float(threshold):
            jump_count += 1
    return jump_count


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    clipped_q = min(max(float(q), 0.0), 1.0)
    ordered = sorted(float(v) for v in values)
    position = (len(ordered) - 1) * clipped_q
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(ordered[lower_index])

    lower = float(ordered[lower_index])
    upper = float(ordered[upper_index])
    weight = float(position - lower_index)
    return lower + ((upper - lower) * weight)


def _score_feature_row(row: Mapping[str, Any]) -> float:
    score = 0.0
    score += 1.0 * min(float(row["high_impact_event_count"]), 20.0)
    score += 8.0 * float(row["high_impact_event_ratio"])
    score += 2.2 * float(row["critical_vital_count"])
    score += 1.2 * float(row["vital_jump_count"])
    score += 0.25 * min(float(row["unique_event_signature_count"]), 20.0)
    score += 0.12 * min(float(row["event_count"]), 40.0)
    score += 0.05 * min(float(row["novel_event_signature_count"]), 30.0)

    if (
        float(row["alarm_like_ratio"]) > 0.6
        and int(row["high_impact_event_count"]) == 0
        and int(row["critical_vital_count"]) == 0
    ):
        score -= 2.0
    return score


def compute_window_code_ratios(
    windows: Sequence[Mapping[str, Any]],
    *,
    target_codes: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute per-window code ratio: target-code events / all events.

    Default target set is HIGH_IMPACT_EVENT_CODES.
    """
    code_set = {
        str(code).strip().upper()
        for code in (target_codes if target_codes is not None else HIGH_IMPACT_EVENT_CODES)
        if str(code).strip()
    }

    rows: List[Dict[str, Any]] = []
    for position, window in enumerate(windows):
        window_index = _to_int(window.get("window_index"), position + 1)
        events = _window_current_events(window)

        code_counter = Counter(_event_code(event) for event in events)
        event_count = int(len(events))
        matched_count = int(sum(code_counter.get(code, 0) for code in code_set))
        ratio = (float(matched_count) / float(event_count)) if event_count > 0 else 0.0

        rows.append(
            {
                "window_index": int(window_index),
                "position": int(position),
                "hours_since_admission": _window_hours_since_admission(window),
                "event_count": event_count,
                "matched_event_count": matched_count,
                "ratio": ratio,
            }
        )

    return rows


def select_windows_by_ratio_threshold(
    windows: Sequence[Mapping[str, Any]],
    *,
    ratio_threshold: float = 0.4,
    target_codes: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Select windows with ratio strictly greater than threshold.

    Ratio is computed as:
      count(target_codes in current window) / total event count in current window

    Default target code set is HIGH_IMPACT_EVENT_CODES.
    """
    ratio_rows = compute_window_code_ratios(
        windows,
        target_codes=target_codes,
    )
    threshold = float(ratio_threshold)
    return [row for row in ratio_rows if float(row["ratio"]) > threshold]


def score_windows_by_keyness(
    windows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Compute lightweight key-window features and a composite score for each window.

    Args:
        windows: Sequence of window payloads from parser/output files.

    Returns:
        List of scored window dictionaries preserving input order.
    """
    scored: List[Dict[str, Any]] = []
    previous_signatures: Set[Tuple[str, str]] = set()
    previous_vital_snapshot: Dict[str, float] = {}

    for position, window in enumerate(windows):
        window_index = _to_int(window.get("window_index"), position + 1)
        events = _window_current_events(window)

        code_counter = Counter(_event_code(event) for event in events)
        signatures = {_event_signature(event) for event in events}

        high_impact_event_count = int(sum(code_counter.get(code, 0) for code in HIGH_IMPACT_EVENT_CODES))
        critical_vital_count = int(sum(1 for event in events if _is_critical_vital(event)))
        alarm_like_count = int(sum(1 for event in events if _is_alarm_like(event)))
        event_count = int(len(events))
        alarm_like_ratio = float(alarm_like_count / event_count) if event_count > 0 else 0.0
        high_impact_event_ratio = (
            float(high_impact_event_count / event_count) if event_count > 0 else 0.0
        )
        unique_event_signature_count = int(len(signatures))
        novel_event_signature_count = int(len(signatures - previous_signatures))

        vital_snapshot = _build_vital_snapshot(events)
        vital_jump_count = int(_vital_jump_count(previous_vital_snapshot, vital_snapshot))

        row: Dict[str, Any] = {
            "window_index": int(window_index),
            "position": int(position),
            "hours_since_admission": _window_hours_since_admission(window),
            "event_count": event_count,
            "unique_event_signature_count": unique_event_signature_count,
            "novel_event_signature_count": novel_event_signature_count,
            "high_impact_event_count": high_impact_event_count,
            "high_impact_event_ratio": high_impact_event_ratio,
            "critical_vital_count": critical_vital_count,
            "vital_jump_count": vital_jump_count,
            "alarm_like_ratio": alarm_like_ratio,
        }
        row["score"] = _score_feature_row(row)
        scored.append(row)

        previous_signatures = signatures
        previous_vital_snapshot = vital_snapshot

    return scored


def select_key_windows(
    windows: Sequence[Mapping[str, Any]],
    *,
    high_quantile: float = 0.95,
    min_windows_for_quantile: int = 20,
    neighbor_radius: int = 2,
    min_spacing: int = 4,
) -> List[Dict[str, Any]]:
    """
    Select key windows using score thresholding + local-peak filtering.

    Rule summary:
    1) Candidate windows:
       - score in top quantile for current patient, OR
       - >=2 critical vital events, OR
       - >=2 major vital jumps vs previous window.
    2) Keep local peaks (within +-neighbor_radius windows).
    3) Greedily enforce temporal spacing using min_spacing windows.
    4) Fallback to single top-score window if nothing selected.
    """
    scored = score_windows_by_keyness(windows)
    if not scored:
        return []

    scores = [float(row["score"]) for row in scored]
    if len(scores) >= int(min_windows_for_quantile):
        threshold = _percentile(scores, high_quantile)
    else:
        threshold = max(scores)

    candidate_map: Dict[int, Tuple[str, ...]] = {}
    for row in scored:
        reasons: List[str] = []
        if float(row["score"]) >= float(threshold):
            reasons.append("high_score")
        if int(row["critical_vital_count"]) >= 2:
            reasons.append("critical_vitals")
        if int(row["vital_jump_count"]) >= 2:
            reasons.append("vital_jump")
        if reasons:
            candidate_map[int(row["position"])] = tuple(reasons)

    peak_candidates: List[Dict[str, Any]] = []
    radius = max(0, int(neighbor_radius))
    for position, reasons in candidate_map.items():
        current = scored[position]
        left = max(0, position - radius)
        right = min(len(scored), position + radius + 1)
        neighbors = [scored[i] for i in range(left, right) if i != position]

        if neighbors and any(float(current["score"]) < float(neighbor["score"]) for neighbor in neighbors):
            continue

        item = dict(current)
        item["selection_reasons"] = list(reasons)
        peak_candidates.append(item)

    chosen: List[Dict[str, Any]] = []
    spacing = max(0, int(min_spacing))
    for candidate in sorted(peak_candidates, key=lambda row: float(row["score"]), reverse=True):
        position = int(candidate["position"])
        if any(abs(position - int(existing["position"])) <= spacing for existing in chosen):
            continue
        chosen.append(candidate)

    if not chosen:
        fallback = dict(max(scored, key=lambda row: float(row["score"])))
        fallback["selection_reasons"] = ["fallback_top_score"]
        chosen = [fallback]

    chosen.sort(key=lambda row: int(row["position"]))
    return chosen
