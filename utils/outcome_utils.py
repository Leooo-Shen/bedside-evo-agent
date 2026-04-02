"""
Utilities for robust clinical outcome label normalization and matching.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

_SURVIVE_ALIASES = {
    "survive",
    "survived",
    "survival",
    "survivor",
    "alive",
    "living",
    "discharged alive",
    "live",
    "lived",
}

_DIE_ALIASES = {
    "die",
    "died",
    "death",
    "dead",
    "deceased",
    "expired",
    "mortality",
    "non survivor",
    "nonsurvivor",
    "non survival",
    "did not survive",
    "didnt survive",
    "didn t survive",
    "not survive",
    "no survival",
}

def _clean_text(value: Any) -> str:
    """Normalize casing and separators for robust matching."""
    text = str(value).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_phrase(text: str, phrase: str) -> bool:
    """Check whether text contains phrase with token boundaries."""
    pattern = rf"\b{re.escape(phrase)}\b"
    return re.search(pattern, text) is not None


def normalize_outcome_label(value: Any) -> Optional[str]:
    """
    Convert free-form outcome strings to canonical labels: 'survive' or 'die'.

    Returns:
        'survive', 'die', or None when label is unknown/ambiguous.
    """
    if value is None:
        return None

    text = _clean_text(value)
    if not text:
        return None

    if text in _SURVIVE_ALIASES:
        return "survive"
    if text in _DIE_ALIASES:
        return "die"

    survive_hits = sum(1 for alias in _SURVIVE_ALIASES if _contains_phrase(text, alias))
    die_hits = sum(1 for alias in _DIE_ALIASES if _contains_phrase(text, alias))

    if survive_hits > 0 and die_hits == 0:
        return "survive"
    if die_hits > 0 and survive_hits == 0:
        return "die"

    return None


def evaluate_outcome_match(predicted: Any, actual: Any) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Compare predicted vs actual outcomes after robust normalization.

    Returns:
        (is_match, normalized_predicted, normalized_actual)
    """
    normalized_predicted = normalize_outcome_label(predicted)
    normalized_actual = normalize_outcome_label(actual)

    if normalized_predicted is None or normalized_actual is None:
        return False, normalized_predicted, normalized_actual

    return normalized_predicted == normalized_actual, normalized_predicted, normalized_actual


def normalize_confidence_label(value: Any) -> Optional[str]:
    """Convert confidence to canonical labels: 'Low', 'Moderate', or 'High'."""
    if not isinstance(value, str):
        return None

    text = _clean_text(value)
    if not text:
        return None
    if text == "low":
        return "Low"
    if text in {"moderate", "medium"}:
        return "Moderate"
    if text == "high":
        return "High"
    return None


def extract_survival_prediction_fields(prediction: Any) -> Tuple[str, str]:
    """
    Extract outcome and categorical confidence from either new or legacy schema.

    Supported schemas:
    - New: {"prediction": "Survive|Die", "confidence": "Low|Moderate|High", ...}
    - Legacy: {"survival_prediction": {"outcome": "...", "confidence": "...", ...}}

    Confidence is returned as categorical only. Non-categorical values map to "Unknown".
    """
    outcome: Any = "unknown"
    confidence_raw: Any = None

    if isinstance(prediction, dict):
        if isinstance(prediction.get("prediction"), str):
            outcome = prediction.get("prediction")
            confidence_raw = prediction.get("confidence")
        else:
            legacy = prediction.get("survival_prediction")
            if isinstance(legacy, dict):
                outcome = legacy.get("outcome", "unknown")
                confidence_raw = legacy.get("confidence")

    outcome_text = str(outcome).strip() if outcome is not None else "unknown"
    if not outcome_text:
        outcome_text = "unknown"

    confidence_label = normalize_confidence_label(confidence_raw)
    if confidence_label is None:
        confidence_label = "Unknown"
    return outcome_text, confidence_label
