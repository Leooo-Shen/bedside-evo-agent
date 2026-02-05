"""
Vital signs trend calculation and analysis utilities.

This module provides functions for:
- Calculating vital sign trends over time
- Classifying vital signs against clinical guidelines
- Formatting vital sign data for display
"""

from typing import Dict, List

# Clinical guideline ranges for vital signs
VITAL_GUIDELINES = {
    "Heart Rate, bpm": {"low": 60, "high": 100, "unit": "bpm"},
    "Non Invasive Blood Pressure systolic, mmHg": {"low": 90, "high": 120, "unit": "mmHg"},
    "Non Invasive Blood Pressure diastolic, mmHg": {"low": 60, "high": 80, "unit": "mmHg"},
    "Respiratory Rate, insp/min": {"low": 12, "high": 20, "unit": "insp/min"},
    "O2 saturation pulseoxymetry, %": {"low": 95, "high": 100, "unit": "%"},
}


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
