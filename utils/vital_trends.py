"""
Vital signs trend calculation and analysis utilities.

This module provides functions for:
- Calculating vital sign trends over time
- Classifying vital signs against clinical guidelines
- Formatting vital sign data for display
- Selecting plottable numeric vital labels for one patient trajectory
"""

import math
import re
from typing import Dict, List, Optional

# Clinical guideline ranges for vital signs
VITAL_GUIDELINES = {
    "Heart Rate, bpm": {"low": 60, "high": 100, "unit": "bpm"},
    "Non Invasive Blood Pressure systolic, mmHg": {"low": 90, "high": 120, "unit": "mmHg"},
    "Non Invasive Blood Pressure diastolic, mmHg": {"low": 60, "high": 80, "unit": "mmHg"},
    "Respiratory Rate, insp/min": {"low": 12, "high": 20, "unit": "insp/min"},
    "O2 saturation pulseoxymetry, %": {"low": 95, "high": 100, "unit": "%"},
}

# Numeric vital labels that are generally suitable for line-trend visualization.
# This favors physiologic trends (HR/BP/SpO2/temperature/ventilation/hemodynamics)
# and avoids many purely categorical VITALS fields.
PLOTTABLE_VITAL_LABEL_PATTERNS = (
    r"\bheart rate\b",
    r"\brespiratory rate\b",
    r"\bo2 saturation\b",
    r"\bspo2\b",
    r"\bblood pressure\b",
    r"\bart bp\b",
    r"\bmean arterial pressure\b",
    r"\bmap\b",
    r"\btemperature\b",
    r"\bfi[o0]2\b",
    r"\binspired o2 fraction\b",
    r"\bo2 flow\b",
    r"\btidal volume\b",
    r"\bminute volume\b",
    r"\bairway pressure\b",
    r"\bpeep\b",
    r"\betco2\b",
    r"\bco2 pressure\b",
    r"\bo2 pressure\b",
    r"\bcentral venous pressure\b",
    r"\bpulmonary artery pressure\b",
    r"\bintra cranial pressure\b",
    r"\bcerebral perfusion pressure\b",
    r"\bsv[o0]2\b",
)


def _safe_float(value: object) -> Optional[float]:
    """Parse finite float values; return None for invalid/NaN/inf."""
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _label_matches_patterns(label: str, patterns: List[str]) -> bool:
    normalized = str(label or "").strip().lower()
    if not normalized:
        return False
    for pattern in patterns:
        if re.search(pattern, normalized):
            return True
    return False


def select_plottable_vitals(
    trajectory: Dict,
    min_points: int = 1,
    max_vitals: int = -1,
    prefer_physiologic_labels: bool = True,
) -> List[str]:
    """
    Automatically select numeric vital labels suitable for line-trend plots.

    Selection rules:
    - code == "VITALS"
    - numeric_value can be parsed as a finite float
    - label has at least `min_points` numeric observations
    - rank by data density (more points, more timestamps, non-constant values)
    - if `prefer_physiologic_labels=True`, keep labels matching
      `PLOTTABLE_VITAL_LABEL_PATTERNS`; if that yields nothing, fallback to all numeric labels.

    Args:
        trajectory: Patient trajectory dict containing an "events" list.
        min_points: Minimum numeric points per label required for plotting.
        max_vitals: Maximum number of labels to return (<=0 means no limit).
        prefer_physiologic_labels: Prefer physiologic labels via regex allowlist.

    Returns:
        Ordered list of vital `code_specifics` labels.
    """
    if min_points < 1:
        raise ValueError("min_points must be >= 1")

    events = trajectory.get("events", [])
    if not isinstance(events, list) or not events:
        return []

    values_by_label: Dict[str, List[float]] = {}
    times_by_label: Dict[str, set] = {}

    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("code") != "VITALS":
            continue

        label = str(event.get("code_specifics") or "").strip()
        if not label:
            continue

        value = _safe_float(event.get("numeric_value"))
        if value is None:
            continue

        time_value = event.get("time")
        values_by_label.setdefault(label, []).append(value)
        times_by_label.setdefault(label, set()).add(str(time_value) if time_value is not None else "")

    if not values_by_label:
        return []

    def build_ranked(labels: List[str]) -> List[str]:
        ranked: List[tuple] = []
        for label in labels:
            values = values_by_label.get(label, [])
            if len(values) < min_points:
                continue
            unique_times = len(times_by_label.get(label, set()))
            if unique_times < 2:
                continue
            value_span = max(values) - min(values) if values else 0.0
            # Density-first ranking with a light bonus for non-flat series.
            score = (len(values) * 10) + unique_times + (1 if value_span > 0 else 0)
            ranked.append((score, len(values), unique_times, label))

        ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3].lower()))
        labels_sorted = [item[3] for item in ranked]
        if max_vitals > 0:
            return labels_sorted[:max_vitals]
        return labels_sorted

    all_labels = list(values_by_label.keys())
    if prefer_physiologic_labels:
        preferred = [
            label for label in all_labels if _label_matches_patterns(label, list(PLOTTABLE_VITAL_LABEL_PATTERNS))
        ]
        selected = build_ranked(preferred)
        if selected:
            return selected

    return build_ranked(all_labels)


def get_vital_names():
    """Get list of vital names to track (without temperature)."""
    return list(VITAL_GUIDELINES.keys())


def classify_vital_status(value: float, vital_name: str) -> str:
    """
    Classify a vital sign value against clinical guidelines.

    Args:
        value: Numeric value of the vital sign
        vital_name: Full name of the vital sign

    Returns:
        Classification: "below_normal", "normal", or "above_normal"
    """
    if vital_name not in VITAL_GUIDELINES:
        return "unknown"

    guidelines = VITAL_GUIDELINES[vital_name]
    if value < guidelines["low"]:
        return "below_normal"
    elif value > guidelines["high"]:
        return "above_normal"
    else:
        return "normal"


def calculate_vital_status(current_events: List[Dict], previous_events: List[Dict] = None) -> Dict:
    """
    Calculate current vital sign status against clinical guidelines.

    Args:
        current_events: Events from current window
        previous_events: Events from previous window (optional, for delta computation)

    Returns:
        Dictionary with vital status information including delta and trend
    """
    vital_names = get_vital_names()
    vital_status = {}

    for vital_name in vital_names:
        # Get values from current window
        current_values = [
            e.get("numeric_value")
            for e in current_events
            if e.get("code") == "VITALS"
            and e.get("code_specifics") == vital_name
            and e.get("numeric_value") is not None
        ]

        if current_values:
            current_avg = sum(current_values) / len(current_values)
            status = classify_vital_status(current_avg, vital_name)

            # Use shorter display name
            display_name = vital_name.split(",")[0]

            guidelines = VITAL_GUIDELINES[vital_name]
            vital_status[display_name] = {
                "current_avg": round(current_avg, 1),
                "status": status,
                "normal_range": f"{guidelines['low']}-{guidelines['high']} {guidelines['unit']}",
                "guideline_low": guidelines["low"],
                "guideline_high": guidelines["high"],
            }

            # Compute delta and trend if previous window data is available
            if previous_events is not None:
                previous_values = [
                    e.get("numeric_value")
                    for e in previous_events
                    if e.get("code") == "VITALS"
                    and e.get("code_specifics") == vital_name
                    and e.get("numeric_value") is not None
                ]

                if previous_values:
                    previous_avg = sum(previous_values) / len(previous_values)
                    delta = current_avg - previous_avg

                    # Determine trend direction (using 1% threshold for "stable")
                    if abs(delta) < abs(previous_avg) * 0.01:
                        trend = "stable"
                    elif delta > 0:
                        trend = "up"
                    else:
                        trend = "down"

                    vital_status[display_name]["previous_avg"] = round(previous_avg, 1)
                    vital_status[display_name]["delta"] = round(delta, 1)
                    vital_status[display_name]["trend"] = trend

    return vital_status


def format_vital_status(vital_status: Dict) -> str:
    """Format vital status for display in prompt."""
    if not vital_status:
        return "No vital sign data available."

    formatted = ""
    for vital_name, data in vital_status.items():
        status_label = data["status"].replace("_", " ").title()
        formatted += f"- {vital_name}: {data['current_avg']} "
        formatted += f"[{status_label}] "
        formatted += f"(Normal range: {data['normal_range']})\n"

    return formatted


def calculate_vital_trends(history_events: List[Dict], current_events: List[Dict]) -> Dict:
    """
    Calculate trends for key vitals by comparing history to current window.

    Args:
        history_events: Events from historical window
        current_events: Events from current window

    Returns:
        Dictionary with vital trends and summary
    """
    vital_names = get_vital_names()
    trends = {}

    for vital_name in vital_names:
        # Get values from history
        history_values = [
            e.get("numeric_value")
            for e in history_events
            if e.get("code") == "VITALS"
            and e.get("code_specifics") == vital_name
            and e.get("numeric_value") is not None
        ]

        # Get values from current window
        current_values = [
            e.get("numeric_value")
            for e in current_events
            if e.get("code") == "VITALS"
            and e.get("code_specifics") == vital_name
            and e.get("numeric_value") is not None
        ]

        if history_values and current_values:
            history_avg = sum(history_values) / len(history_values)
            current_avg = sum(current_values) / len(current_values)

            # Calculate percent change
            if history_avg != 0:
                percent_change = ((current_avg - history_avg) / abs(history_avg)) * 100
            else:
                percent_change = 0

            # Determine trend (using 5% threshold for "stable")
            if percent_change > 5:
                trend = "increasing"
            elif percent_change < -5:
                trend = "decreasing"
            else:
                trend = "stable"

            # Use shorter display name
            display_name = vital_name.split(",")[0]  # Remove units for cleaner display

            trends[display_name] = {
                "trend": trend,
                "history_avg": round(history_avg, 1),
                "current_avg": round(current_avg, 1),
                "percent_change": round(percent_change, 1),
            }

    return trends


def format_vital_trends(trends: Dict) -> str:
    """Format vital trends for display in prompt."""
    if not trends:
        return "No vital sign trends available (insufficient data)."

    formatted = ""
    for vital_name, data in trends.items():
        formatted += f"- {vital_name}: {data['trend']} "
        formatted += f"(history avg: {data['history_avg']}, current avg: {data['current_avg']}, "
        formatted += f"change: {data['percent_change']:+.1f}%)\n"

    return formatted
