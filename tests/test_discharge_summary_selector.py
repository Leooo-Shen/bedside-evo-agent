"""Tests for ICU discharge-summary selection rules."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.discharge_summary_selector import (
    select_discharge_summaries_for_icu_stays,
    summarize_discharge_summary_selection,
)


def _build_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    # ICU stays for one subject across one/two hospitalizations.
    icu_df = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "icu_stay_id": 101,
                "enter_time": "2024-01-01 00:00:00",
                "leave_time": "2024-01-02 00:00:00",
            },
            {
                "subject_id": 1,
                "icu_stay_id": 102,
                "enter_time": "2024-01-10 00:00:00",
                "leave_time": "2024-01-11 00:00:00",
            },
            {
                "subject_id": 1,
                "icu_stay_id": 103,
                "enter_time": "2024-01-20 00:00:00",
                "leave_time": "2024-01-21 00:00:00",
            },
            {
                "subject_id": 1,
                "icu_stay_id": 104,
                "enter_time": "2024-02-01 00:00:00",
                "leave_time": "2024-02-02 00:00:00",
            },
            {
                "subject_id": 1,
                "icu_stay_id": 105,
                "enter_time": "2024-02-10 00:00:00",
                "leave_time": "2024-02-11 00:00:00",
            },
            # Intervening ICU stay for rule-2 negative case.
            {
                "subject_id": 1,
                "icu_stay_id": 106,
                "enter_time": "2024-01-22 00:00:00",
                "leave_time": "2024-01-22 12:00:00",
            },
        ]
    )

    # Bridge rows + discharge summaries.
    events_df = pd.DataFrame(
        [
            # Bridge rows: map ICU stay -> hosp stay
            {"subject_id": 1, "time": "2024-01-01 01:00:00", "code": "VITALS", "hosp_stay_id": 5001, "icu_stay_id": 101},
            {"subject_id": 1, "time": "2024-01-10 01:00:00", "code": "VITALS", "hosp_stay_id": 5002, "icu_stay_id": 102},
            {"subject_id": 1, "time": "2024-01-20 01:00:00", "code": "VITALS", "hosp_stay_id": 5003, "icu_stay_id": 103},
            {"subject_id": 1, "time": "2024-02-01 01:00:00", "code": "VITALS", "hosp_stay_id": 5004, "icu_stay_id": 104},
            {"subject_id": 1, "time": "2024-02-10 01:00:00", "code": "VITALS", "hosp_stay_id": 5005, "icu_stay_id": 105},
            {"subject_id": 1, "time": "2024-01-22 01:00:00", "code": "VITALS", "hosp_stay_id": 5003, "icu_stay_id": 106},
            # Rule 1: exactly one in-window summary for ICU 101
            {
                "subject_id": 1,
                "time": "2024-01-01 12:00:00",
                "code": "NOTE_DISCHARGESUMMARY",
                "hosp_stay_id": 5001,
                "icu_stay_id": pd.NA,
                "text_value": "in-window one",
                "code_specifics": "summary",
            },
            # Rule 2 positive: ICU 102 has post-ICU same-hosp summary within 7d and no intervening ICU
            {
                "subject_id": 1,
                "time": "2024-01-11 06:00:00",
                "code": "NOTE_DISCHARGESUMMARY",
                "hosp_stay_id": 5002,
                "icu_stay_id": pd.NA,
                "text_value": "post-icu valid",
                "code_specifics": "summary",
            },
            # Rule 2 negative (intervening ICU): ICU 103 has summary within 7d but ICU 106 starts before it
            {
                "subject_id": 1,
                "time": "2024-01-23 06:00:00",
                "code": "NOTE_DISCHARGESUMMARY",
                "hosp_stay_id": 5003,
                "icu_stay_id": pd.NA,
                "text_value": "post-icu but intervened",
                "code_specifics": "summary",
            },
            # Rule 2 negative (outside 7d): ICU 104 summary after 8 days
            {
                "subject_id": 1,
                "time": "2024-02-09 12:00:00",
                "code": "NOTE_DISCHARGESUMMARY",
                "hosp_stay_id": 5004,
                "icu_stay_id": pd.NA,
                "text_value": "too late",
                "code_specifics": "summary",
            },
            # Rule 1 negative (two in-window notes): ICU 105
            {
                "subject_id": 1,
                "time": "2024-02-10 06:00:00",
                "code": "NOTE_DISCHARGESUMMARY",
                "hosp_stay_id": 5005,
                "icu_stay_id": pd.NA,
                "text_value": "first in-window",
                "code_specifics": "summary",
            },
            {
                "subject_id": 1,
                "time": "2024-02-10 12:00:00",
                "code": "NOTE_DISCHARGESUMMARY",
                "hosp_stay_id": 5005,
                "icu_stay_id": pd.NA,
                "text_value": "second in-window",
                "code_specifics": "summary",
            },
        ]
    )

    return events_df, icu_df


def test_select_discharge_summaries_for_icu_stays_rules() -> None:
    events_df, icu_df = _build_fixture()
    result = select_discharge_summaries_for_icu_stays(events_df, icu_df, max_days_after_leave=7)

    by_stay = {int(row["icu_stay_id"]): row for _, row in result.iterrows()}

    # Rule 1 positive
    assert by_stay[101]["selected"] is True
    assert by_stay[101]["selection_rule"] == "in_icu_exactly_one"

    # Rule 2 positive
    assert by_stay[102]["selected"] is True
    assert by_stay[102]["selection_rule"] == "post_icu_same_hosp_within_7d_no_new_icu"

    # Rule 2 negative due to intervening ICU
    assert by_stay[103]["selected"] is False
    assert by_stay[103]["selection_rule"] is None

    # Rule 2 negative due to >7 days
    assert by_stay[104]["selected"] is False
    assert by_stay[104]["selection_rule"] is None

    # Rule 1 negative because there are two in-window summaries
    assert by_stay[105]["in_icu_note_count"] == 2
    assert by_stay[105]["selected"] is False


def test_summarize_discharge_summary_selection() -> None:
    events_df, icu_df = _build_fixture()
    result = select_discharge_summaries_for_icu_stays(events_df, icu_df, max_days_after_leave=7)
    summary = summarize_discharge_summary_selection(result)

    assert summary["total_icu_stays"] == 6
    assert summary["selected_icu_stays"] == 3
    assert abs(summary["selected_ratio"] - (3 / 6)) < 1e-9
    assert summary["in_icu_exactly_one"] == 1
    assert summary["post_icu_same_hosp_within_7d_no_new_icu"] == 2
