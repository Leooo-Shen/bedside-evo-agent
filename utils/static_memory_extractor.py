"""Deterministic static memory extraction for MedAgent."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

OUTCOME_LEAKAGE_CODES = {"META_DEATH"}

BASELINE_LAB_KEYWORDS = {
    "creatinine": ["creatinine"],
    "hemoglobin": ["hemoglobin"],
    "white_blood_cells": ["white blood cells", "wbc"],
    "platelet_count": ["platelet"],
    "glucose": ["glucose"],
    "sodium": ["sodium"],
    "potassium": ["potassium"],
    "albumin": ["albumin"],
}

META_FIELD_MAPPING = {
    "META_RACE": "race",
    "META_LANGUAGE": "language",
    "META_INSURANCE": "insurance",
    "META_MARTIAL_STATUS": "marital_status",
}

MEDICATION_CODES = {"DRUG_START", "DRUG_PRESCRIPTION"}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _parse_time(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value)
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        key = item.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item.strip())
    return result


def _is_outcome_leak_event(code: str, event_time: Optional[pd.Timestamp], enter_time: pd.Timestamp) -> bool:
    if code in OUTCOME_LEAKAGE_CODES:
        return True
    if code.startswith("LEAVE_"):
        return True
    if code == "NOTE_DISCHARGESUMMARY" and event_time is not None and event_time >= enter_time:
        return True
    return False


def _extract_meta_fields(pre_or_admission_events: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    values: Dict[str, Optional[str]] = {
        "race": None,
        "language": None,
        "insurance": None,
        "marital_status": None,
    }

    for event in pre_or_admission_events:
        code = _safe_text(event.get("code"))
        mapped_key = META_FIELD_MAPPING.get(code)
        if not mapped_key:
            continue

        candidate = _safe_text(event.get("code_specifics")) or _safe_text(event.get("text_value"))
        if candidate:
            values[mapped_key] = candidate

    return values


def _extract_admission_diagnoses(events_with_time: List[Tuple[pd.Timestamp, Dict[str, Any]]], enter_time: pd.Timestamp) -> List[str]:
    admission_end = enter_time + timedelta(hours=6)
    diagnoses: List[str] = []

    for event_time, event in events_with_time:
        if event_time is None:
            continue
        if not (enter_time <= event_time <= admission_end):
            continue
        if _safe_text(event.get("code")) != "DIAGNOSIS":
            continue

        diag = _safe_text(event.get("code_specifics")) or _safe_text(event.get("text_value"))
        if diag and diag.lower() != "none":
            diagnoses.append(diag)

    return _dedupe_preserve_order(diagnoses)[:10]


def _extract_past_medical_history(pre_icu_events_with_time: List[Tuple[pd.Timestamp, Dict[str, Any]]], enter_time: pd.Timestamp) -> List[str]:
    chronic_cutoff = enter_time - timedelta(hours=24)
    history: List[str] = []

    for event_time, event in pre_icu_events_with_time:
        if event_time is None:
            continue
        if event_time > chronic_cutoff:
            continue
        if _safe_text(event.get("code")) != "DIAGNOSIS":
            continue

        diag = _safe_text(event.get("code_specifics")) or _safe_text(event.get("text_value"))
        if diag and diag.lower() != "none":
            history.append(diag)

    deduped = _dedupe_preserve_order(history)
    if deduped:
        return deduped[:20]

    # Fallback: if no chronic history, include any pre-ICU diagnoses.
    fallback: List[str] = []
    for _, event in pre_icu_events_with_time:
        if _safe_text(event.get("code")) != "DIAGNOSIS":
            continue
        diag = _safe_text(event.get("code_specifics")) or _safe_text(event.get("text_value"))
        if diag and diag.lower() != "none":
            fallback.append(diag)
    return _dedupe_preserve_order(fallback)[:20]


def _extract_admission_medications(pre_icu_events_with_time: List[Tuple[pd.Timestamp, Dict[str, Any]]], enter_time: pd.Timestamp) -> List[str]:
    start_window = enter_time - timedelta(hours=24)
    meds: List[str] = []

    for event_time, event in pre_icu_events_with_time:
        if event_time is None:
            continue
        if event_time < start_window:
            continue
        code = _safe_text(event.get("code"))
        if code not in MEDICATION_CODES:
            continue

        med_name = _safe_text(event.get("code_specifics")) or _safe_text(event.get("text_value"))
        if med_name and med_name.lower() != "none":
            meds.append(med_name)

    return _dedupe_preserve_order(meds)[:20]


def _extract_baseline_labs(
    pre_icu_events_with_time: List[Tuple[pd.Timestamp, Dict[str, Any]]],
    enter_time: pd.Timestamp,
    lookback_start_hours: float,
    lookback_end_hours: float,
) -> Dict[str, Optional[float]]:
    # Example default: 72h -> 24h before ICU.
    start_time = enter_time - timedelta(hours=lookback_start_hours)
    end_time = enter_time - timedelta(hours=lookback_end_hours)

    labs_in_window: List[Tuple[pd.Timestamp, Dict[str, Any]]] = []
    for event_time, event in pre_icu_events_with_time:
        if event_time is None:
            continue
        if event_time < start_time or event_time > end_time:
            continue
        if _safe_text(event.get("code")) != "LAB_TEST":
            continue
        labs_in_window.append((event_time, event))

    baseline: Dict[str, Optional[float]] = {key: None for key in BASELINE_LAB_KEYWORDS.keys()}

    for canonical_name, keywords in BASELINE_LAB_KEYWORDS.items():
        latest_match_time: Optional[pd.Timestamp] = None
        latest_match_value: Optional[float] = None

        for event_time, event in labs_in_window:
            label = _safe_text(event.get("code_specifics")).lower()
            if not label:
                continue
            if not any(keyword in label for keyword in keywords):
                continue

            numeric_value = event.get("numeric_value")
            if numeric_value is None:
                continue
            try:
                numeric = float(numeric_value)
            except (TypeError, ValueError):
                continue

            if latest_match_time is None or event_time >= latest_match_time:
                latest_match_time = event_time
                latest_match_value = numeric

        baseline[canonical_name] = latest_match_value

    return baseline


def extract_static_memory(
    trajectory: Dict[str, Any],
    baseline_lab_lookback_start_hours: float = 72,
    baseline_lab_lookback_end_hours: float = 24,
) -> Dict[str, Any]:
    """Extract deterministic static memory from a patient trajectory.

    Outcome-leakage events are excluded:
    - META_DEATH
    - LEAVE_*
    - NOTE_DISCHARGESUMMARY events at/after current ICU admission
    """
    enter_time = _parse_time(trajectory.get("enter_time"))
    if enter_time is None:
        raise ValueError("trajectory.enter_time is required for static memory extraction")

    raw_events = trajectory.get("events", []) or []
    filtered_events_with_time: List[Tuple[Optional[pd.Timestamp], Dict[str, Any]]] = []

    for event in raw_events:
        code = _safe_text(event.get("code"))
        event_time = _parse_time(event.get("time"))
        if _is_outcome_leak_event(code, event_time, enter_time):
            continue
        filtered_events_with_time.append((event_time, event))

    filtered_events_with_time.sort(key=lambda x: x[0] if x[0] is not None else pd.Timestamp.min)

    pre_icu_events_with_time = [(t, e) for t, e in filtered_events_with_time if t is not None and t < enter_time]
    pre_or_admission_meta_events = [
        e
        for t, e in filtered_events_with_time
        if t is not None and t <= enter_time and _safe_text(e.get("code")).startswith("META_")
    ]

    demographics = {
        "age": trajectory.get("age_at_admission"),
        "gender": trajectory.get("gender"),
        "admission_diagnoses": _extract_admission_diagnoses(filtered_events_with_time, enter_time),
    }
    demographics.update(_extract_meta_fields(pre_or_admission_meta_events))

    static_memory = {
        "demographics": demographics,
        "past_medical_history": _extract_past_medical_history(pre_icu_events_with_time, enter_time),
        "baseline_labs": _extract_baseline_labs(
            pre_icu_events_with_time,
            enter_time,
            lookback_start_hours=baseline_lab_lookback_start_hours,
            lookback_end_hours=baseline_lab_lookback_end_hours,
        ),
        "admission_medications": _extract_admission_medications(pre_icu_events_with_time, enter_time),
        "extraction_metadata": {
            "baseline_lab_lookback_start_hours": baseline_lab_lookback_start_hours,
            "baseline_lab_lookback_end_hours": baseline_lab_lookback_end_hours,
            "num_total_events": len(raw_events),
            "num_events_after_leakage_filter": len(filtered_events_with_time),
            "num_pre_icu_events": len(pre_icu_events_with_time),
        },
    }

    return static_memory
