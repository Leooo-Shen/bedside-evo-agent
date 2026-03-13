"""Unit tests for experiments.oracle.common utilities."""

from __future__ import annotations

import math
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.oracle.common import (
    apply_prompt_outcome_mode,
    assign_normalized_time_bin,
    compute_domain_consistency,
    mask_window_outcome_leakage,
    sanitize_discharge_summary_text,
    spearman_correlation,
)


def test_sanitize_discharge_summary_text_masks_outcome_and_removes_sections() -> None:
    text = (
        "Brief Hospital Course: required escalating support.\n"
        "Discharge Disposition: Expired\n"
        "Discharge Condition: Deceased\n"
        "Post-ICU comments: family said patient passed away."
    )
    sanitized = sanitize_discharge_summary_text(text)
    assert "Discharge Disposition:" not in sanitized
    assert "Discharge Condition:" not in sanitized
    assert "expired" not in sanitized.lower()
    assert "passed away" not in sanitized.lower()
    assert "Brief Hospital Course" in sanitized


def test_mask_window_outcome_leakage_masks_discharge_summary_payloads() -> None:
    window = {
        "history_events": [
            {
                "code": "NOTE_DISCHARGESUMMARY",
                "code_specifics": "Expired case",
                "text_value": "Discharge Disposition: Expired. Patient passed away.",
            },
            {
                "type": "pre_icu_reports",
                "content": "External summary says patient deceased.",
            },
        ],
        "current_events": [],
        "pre_icu_history": {
            "content": "Discharge Condition: Deceased",
            "baseline_content": "No outcome text expected here but passed away is leaked.",
        },
    }
    masked = mask_window_outcome_leakage(window)
    assert masked["history_events"][0]["code_specifics"] != window["history_events"][0]["code_specifics"]
    assert "Discharge Disposition:" not in masked["history_events"][0]["text_value"]
    assert "passed away" not in masked["history_events"][0]["text_value"].lower()
    assert "expired" not in masked["history_events"][0]["text_value"].lower()
    assert "[OUTCOME_MASKED]" in masked["history_events"][1]["content"]
    assert "deceased" not in masked["pre_icu_history"]["content"].lower()
    assert "Discharge Condition:" not in masked["pre_icu_history"]["content"]
    assert "[OUTCOME_MASKED]" in masked["pre_icu_history"]["baseline_content"]


def test_apply_prompt_outcome_mode_reverses_prompt_visible_outcome_only() -> None:
    trajectory = {"subject_id": 1, "icu_stay_id": 2, "survived": False, "events": [{"code": "X"}]}
    transformed = apply_prompt_outcome_mode(trajectory, reverse_prompt_outcome=True)
    assert transformed["survived"] is True
    assert trajectory["survived"] is False
    assert transformed["events"] == trajectory["events"]


def test_domain_consistency_uses_weighted_score_and_nearest_label() -> None:
    oracle_output = {
        "patient_status": {
            "domains": {
                "hemodynamics": {"label": "improving"},
                "respiratory": {"label": "deteriorating"},
                "renal_metabolic": {"label": "stable"},
                "neurology": {"label": "deteriorating"},
            },
            "overall": {"label": "fluctuating"},
        }
    }
    result = compute_domain_consistency(oracle_output)
    # weighted score = (1 + -1 + 0.5 + -1) / 4 = -0.125 => nearest anchor should be fluctuating (-0.5)
    assert math.isclose(result["weighted_domain_score"], -0.125, rel_tol=1e-9, abs_tol=1e-9)
    assert result["weighted_domain_label"] == "fluctuating"
    assert result["is_match"] is True


def test_normalized_time_binning_and_monotonic_spearman() -> None:
    bins = [assign_normalized_time_bin(x, num_bins=10) for x in [0.0, 0.1, 0.5, 0.999, 1.0]]
    assert bins == [0, 1, 5, 9, 9]

    x = [0, 1, 2, 3, 4]
    y = [0.1, 0.2, 0.3, 0.5, 0.8]
    rho = spearman_correlation(x, y)
    assert math.isclose(rho, 1.0, rel_tol=1e-9, abs_tol=1e-9)
