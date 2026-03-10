"""Tests for Oracle trend extraction in llm_log_viewer."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.llm_log_viewer import _build_oracle_trend_rows, save_llm_calls_html


def test_build_oracle_trend_rows_scores_actions_and_status() -> None:
    calls = [
        {
            "step_type": "oracle_evaluator",
            "window_index": 1,
            "hours_since_admission": 0.5,
            "parsed_response": {
                "patient_status": {"overall": {"label": "stable"}},
                "action_evaluations": [
                    {"overall": {"label": "appropriate"}},
                    {"overall": {"label": "suboptimal"}},
                    {"overall": {"label": "not_enough_context"}},
                ],
            },
        }
    ]

    rows = _build_oracle_trend_rows(calls)
    assert len(rows) == 1
    row = rows[0]
    assert row["window_index"] == 1
    assert row["status_label"] == "stable"
    assert row["status_score"] == 0.0
    assert row["action_total"] == 3
    assert row["action_scorable"] == 2
    assert row["action_score"] == 0.25


def test_build_oracle_trend_rows_uses_latest_duplicate_window() -> None:
    calls = [
        {
            "step_type": "oracle_evaluator",
            "window_index": 2,
            "hours_since_admission": 1.0,
            "parsed_response": {"overall": {"label": "stable"}},
        },
        {
            "step_type": "oracle_evaluator",
            "window_index": 2,
            "hours_since_admission": 1.5,
            "parsed_response": {"overall": {"label": "deteriorating"}},
        },
    ]

    rows = _build_oracle_trend_rows(calls)
    assert len(rows) == 1
    assert rows[0]["window_index"] == 2
    assert rows[0]["hours_since_admission"] == 1.5
    assert rows[0]["status_label"] == "deteriorating"
    assert rows[0]["status_score"] == -1.0


def test_build_oracle_trend_rows_keeps_zero_window_index() -> None:
    calls = [
        {
            "step_type": "oracle_evaluator",
            "window_index": 0,
            "hours_since_admission": 0.0,
            "parsed_response": {"overall": {"label": "stable"}},
        }
    ]
    rows = _build_oracle_trend_rows(calls)
    assert len(rows) == 1
    assert rows[0]["window_index"] == 0


def test_build_oracle_trend_rows_skips_non_oracle_steps() -> None:
    calls = [
        {
            "step_type": "predictor",
            "window_index": 1,
            "hours_since_admission": 0.5,
            "parsed_response": {"overall": {"label": "stable"}},
        }
    ]

    rows = _build_oracle_trend_rows(calls)
    assert rows == []


def test_save_llm_calls_html_sorts_by_window_index(tmp_path) -> None:
    payload = {
        "patient_id": "123_456",
        "calls": [
            {
                "step_type": "oracle_evaluator",
                "window_index": 2,
                "hours_since_admission": 1.0,
                "timestamp": "2026-03-09T12:00:05",
                "prompt": "p2",
                "response": "r2",
                "parsed_response": {
                    "overall": {"label": "stable"},
                    "action_evaluations": [
                        {"overall": {"label": "appropriate"}},
                        {"overall": {"label": "suboptimal"}},
                    ],
                },
                "metadata": {},
            },
            {
                "step_type": "oracle_evaluator",
                "window_index": 1,
                "hours_since_admission": 0.5,
                "timestamp": "2026-03-09T12:00:01",
                "prompt": "p1",
                "response": "r1",
                "parsed_response": {
                    "overall": {"label": "deteriorating"},
                    "action_evaluations": [{"overall": {"label": "potentially_harmful"}}],
                },
                "metadata": {},
            },
        ],
    }

    output_path = tmp_path / "llm_calls.html"
    save_llm_calls_html(payload, output_path)
    html = output_path.read_text(encoding="utf-8")

    first_window_position = html.index("window 1")
    second_window_position = html.index("window 2")
    assert first_window_position < second_window_position
    assert "Avg patient-status score: <strong>-0.50</strong>" in html
    assert "Avg doctor-action score: <strong>-0.17</strong>" in html
    assert "Doctor-action label source used for scoring" in html
    assert (tmp_path / "oracle_patient_status_trend.png").exists()
    assert (tmp_path / "oracle_doctor_action_score_trend.png").exists()
    assert not (tmp_path / "oracle_patient_status_trend.svg").exists()
    assert not (tmp_path / "oracle_doctor_action_score_trend.svg").exists()
