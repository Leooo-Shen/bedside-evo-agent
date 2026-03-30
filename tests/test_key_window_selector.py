"""Unit tests for lightweight key-window selection."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.key_window_selector import (
    compute_window_code_ratios,
    score_windows_by_keyness,
    select_key_windows,
    select_windows_by_ratio_threshold,
)


def _event(code: str, code_specifics: str, numeric_value: Any = None) -> Dict[str, Any]:
    return {
        "code": code,
        "code_specifics": code_specifics,
        "numeric_value": numeric_value,
    }


def _make_baseline_window(index: int) -> Dict[str, Any]:
    return {
        "window_index": index,
        "hours_since_admission": (index - 1) * 0.5,
        "current_events": [
            _event("VITALS", "Heart Rate, bpm", 92),
            _event("VITALS", "Respiratory Rate, insp/min", 18),
            _event("OTHER_EVENT", "Routine chart"),
        ],
    }


def test_select_key_windows_finds_sparse_high_signal_peaks() -> None:
    windows: List[Dict[str, Any]] = [_make_baseline_window(i + 1) for i in range(12)]

    # Peak 1: intervention-heavy burst.
    windows[3]["current_events"] = (
        [_event("PROCEDURE", "Ventilator setup") for _ in range(10)]
        + [_event("DIAGNOSIS", "Acute respiratory failure") for _ in range(6)]
        + [_event("BODY_INPUT", "Crystalloid bolus") for _ in range(4)]
    )

    # Peak 2: physiologic instability + interventions.
    windows[9]["current_events"] = [
        _event("VITALS", "Mean Arterial Pressure, mmHg", 52),
        _event("VITALS", "Heart Rate, bpm", 152),
        _event("PROCEDURE", "Urgent line placement"),
        _event("DRUG_START", "Norepinephrine"),
        _event("BODY_INPUT", "Crystalloid bolus"),
    ]

    selected = select_key_windows(
        windows,
        high_quantile=0.90,
        min_windows_for_quantile=5,
        neighbor_radius=1,
        min_spacing=2,
    )

    selected_indices = [item["window_index"] for item in selected]
    assert 4 in selected_indices
    assert 10 in selected_indices


def test_select_key_windows_returns_top_window_for_short_sequence() -> None:
    windows = [
        {
            "window_index": 1,
            "current_events": [_event("VITALS", "Heart Rate, bpm", 90)],
        },
        {
            "window_index": 2,
            "current_events": [
                _event("PROCEDURE", "Intubation"),
                _event("DRUG_START", "Norepinephrine"),
                _event("BODY_INPUT", "Crystalloid bolus"),
            ],
        },
        {
            "window_index": 3,
            "current_events": [_event("VITALS", "Heart Rate, bpm", 92)],
        },
    ]

    selected = select_key_windows(windows, min_windows_for_quantile=20)
    assert len(selected) == 1
    assert selected[0]["window_index"] == 2


def test_score_windows_by_keyness_reads_raw_current_events_fallback() -> None:
    windows = [
        {
            "window_index": 1,
            "raw_current_events": [
                _event("PROCEDURE", "Dialysis"),
                _event("BODY_INPUT", "Fluid"),
            ],
        }
    ]

    scored = score_windows_by_keyness(windows)
    assert len(scored) == 1
    assert scored[0]["event_count"] == 2
    assert scored[0]["high_impact_event_count"] == 2
    assert scored[0]["high_impact_event_ratio"] == 1.0


def test_select_windows_by_ratio_threshold_uses_high_impact_by_default() -> None:
    windows = [
        {
            "window_index": 1,
            "current_events": [
                _event("VITALS", "Heart Rate, bpm", 90),
                _event("VITALS", "Respiratory Rate, insp/min", 18),
            ],
        },
        {
            "window_index": 2,
            "current_events": [
                _event("PROCEDURE", "Intubation"),
                _event("DRUG_START", "Norepinephrine"),
                _event("DIAGNOSIS", "Acute respiratory failure"),
                _event("VITALS", "Heart Rate, bpm", 120),
            ],
        },
    ]

    ratios = compute_window_code_ratios(windows)
    assert len(ratios) == 2
    assert ratios[0]["ratio"] == 0.0
    assert ratios[1]["ratio"] == 0.75

    selected = select_windows_by_ratio_threshold(windows, ratio_threshold=0.5)
    assert len(selected) == 1
    assert selected[0]["window_index"] == 2
