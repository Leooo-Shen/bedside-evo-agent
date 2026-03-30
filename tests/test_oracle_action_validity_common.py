"""Unit tests for experiments.oracle.action_validity_common."""

from __future__ import annotations

import math
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.oracle.action_validity_common import (
    ACTIONABLE_EVENT_CODES,
    build_doctor_action_texts_from_events,
    build_recommendation_texts,
    compute_precision_recall_f1,
    count_actionable_events,
    extract_actionable_events,
    identify_action_evaluation,
    inject_counterfactual_current_event,
    match_recommendations_to_actions,
    normalize_text_tokens,
    select_wrong_action_template,
)


def test_extract_actionable_events_uses_whitelist() -> None:
    events = [
        {"code": "DRUG_START", "code_specifics": "Norepinephrine"},
        {"code": "VITALS", "code_specifics": "MAP", "numeric_value": 58},
        {"code": "PROCEDURE", "code_specifics": "Intubation"},
        {"code": "TRANSFER", "code_specifics": "MICU"},
    ]

    extracted = extract_actionable_events(events)
    assert len(extracted) == 3
    assert count_actionable_events(events) == 3
    assert {e["code"] for e in extracted}.issubset(set(ACTIONABLE_EVENT_CODES))


def test_select_wrong_action_template_priority() -> None:
    hypotension_events = [{"code": "VITALS", "code_specifics": "MAP", "numeric_value": 60}]
    hypoxemia_events = [{"code": "VITALS", "code_specifics": "SpO2", "numeric_value": 84}]
    renal_events = [{"code": "LAB", "code_specifics": "Creatinine", "numeric_value": 2.7}]
    fallback_events = [{"code": "VITALS", "code_specifics": "Heart Rate", "numeric_value": 88}]

    assert select_wrong_action_template(hypotension_events)["template_id"] == "wrong_pressor_deescalation"
    assert select_wrong_action_template(hypoxemia_events)["template_id"] == "wrong_oxygen_deescalation"
    assert select_wrong_action_template(renal_events)["template_id"] == "wrong_nephrotoxic_exposure"
    assert select_wrong_action_template(fallback_events)["template_id"] == "wrong_septic_shock_contradiction"


def test_inject_counterfactual_current_event_adds_expected_event_id() -> None:
    window = {
        "current_window_start": "2024-01-01T00:00:00",
        "current_window_end": "2024-01-01T01:00:00",
        "current_events": [
            {"event_id": 1001, "code": "PROCEDURE", "code_specifics": "Line placement"},
            {"event_id": 1002, "code": "DRUG_START", "code_specifics": "Norepinephrine"},
            {"event_id": 1003, "code": "TRANSFER", "code_specifics": "ICU"},
        ],
    }

    injected_window, injected_event, expected_action_id = inject_counterfactual_current_event(
        window,
        marker_token="CFX_MARKER_TEST",
        wrong_action_text="Decrease vasopressor despite MAP < 65.",
    )

    assert len(injected_window["current_events"]) == 4
    assert expected_action_id == "1004"
    assert int(injected_event["event_id"]) == 1004
    assert "CFX_MARKER_TEST" in str(injected_event.get("text_value"))


def test_identify_action_evaluation_by_action_id_and_marker_fallback() -> None:
    action_evaluations = [
        {
            "action_id": "1002",
            "action_description": "Titrate vasopressor",
            "overall": {"label": "appropriate"},
        },
        {
            "action_id": "A3",
            "action_description": "CFX_WRONG_ACTION_abc decrease norepinephrine",
            "overall": {"label": "potentially_harmful"},
        },
    ]

    by_id = identify_action_evaluation(
        action_evaluations,
        expected_action_id="1002",
        marker_token="CFX_WRONG_ACTION_abc",
    )
    assert by_id is not None
    assert by_id.get("action_id") == "1002"

    by_marker = identify_action_evaluation(
        action_evaluations,
        expected_action_id="9999",
        marker_token="CFX_WRONG_ACTION_abc",
    )
    assert by_marker is not None
    assert by_marker.get("overall", {}).get("label") == "potentially_harmful"


def test_normalization_matching_and_metrics() -> None:
    recommendations = [
        {"action": "Increase noradrenaline", "action_description": "Increase vasopressor support"},
        {"action": "Start broad-spectrum antibiotics", "action_description": "for septic concern"},
    ]
    events = [
        {"code": "DRUG_START", "code_specifics": "Increase norepinephrine infusion"},
        {"code": "PROCEDURE", "code_specifics": "Central line insertion"},
    ]

    rec_texts = build_recommendation_texts(recommendations)
    action_texts = build_doctor_action_texts_from_events(events)

    # Ensure synonym normalization works (noradrenaline -> norepinephrine)
    tokens = normalize_text_tokens(rec_texts[0])
    assert "norepinephrine" in tokens

    match = match_recommendations_to_actions(
        rec_texts,
        action_texts,
        jaccard_threshold=0.20,
        min_shared_tokens=1,
    )
    assert match["num_recommendations"] == 2
    assert match["num_doctor_actions"] == 2
    assert match["num_matches"] >= 1

    precision, recall, f1 = compute_precision_recall_f1(
        num_matches=match["num_matches"],
        num_recommendations=match["num_recommendations"],
        num_doctor_actions=match["num_doctor_actions"],
    )
    assert precision <= 1.0 and recall <= 1.0 and f1 <= 1.0

    p_nan, r_nan, f1_nan = compute_precision_recall_f1(
        num_matches=0,
        num_recommendations=0,
        num_doctor_actions=3,
    )
    assert math.isnan(p_nan)
    assert not math.isnan(r_nan)
    assert math.isnan(f1_nan)
