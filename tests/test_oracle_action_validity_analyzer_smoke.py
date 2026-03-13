"""Smoke test for experiments.oracle.analyze_oracle_action_validity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import experiments.oracle.analyze_oracle_action_validity as analyzer


def _write_full_visible_predictions(run_dir: Path) -> None:
    patient_dir = run_dir / "conditions" / "full_visible" / "patients" / "111_222"
    patient_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "subject_id": 111,
        "icu_stay_id": 222,
        "trajectory_metadata": {"true_survived": False},
        "window_outputs": [
            {
                "window_index": 1,
                "window_metadata": {"hours_since_admission": 0.0},
                "raw_current_events": [
                    {"code": "DRUG_START", "code_specifics": "Norepinephrine infusion started"},
                    {"code": "PROCEDURE", "code_specifics": "Central line insertion"},
                ],
                "oracle_output": {
                    "recommendations": [
                        {
                            "rank": 1,
                            "action": "Increase norepinephrine",
                            "action_description": "Increase vasopressor support",
                        },
                        {
                            "rank": 2,
                            "action": "Start broad-spectrum antibiotics",
                            "action_description": "Cover sepsis source",
                        },
                    ]
                },
            },
            {
                "window_index": 2,
                "window_metadata": {"hours_since_admission": 2.0},
                "raw_current_events": [
                    {"code": "TRANSFER", "code_specifics": "Transfer to MICU"}
                ],
                "oracle_output": {
                    "recommendations": [
                        {
                            "rank": 1,
                            "action": "Transfer to MICU",
                            "action_description": "Escalate level of care",
                        }
                    ]
                },
            },
        ],
    }

    with open(patient_dir / "oracle_predictions.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_q3_results(run_dir: Path) -> None:
    q3_dir = run_dir / "action_validity" / "q3_counterfactual"
    q3_dir.mkdir(parents=True, exist_ok=True)

    q3_df = pd.DataFrame(
        [
            {
                "subject_id": 111,
                "icu_stay_id": 222,
                "patient_id": "111_222",
                "counterfactual_action_found": True,
                "counterfactual_label": "potentially_harmful",
                "counterfactual_is_negative": True,
                "counterfactual_is_potentially_harmful": True,
                "baseline_score": 1.0,
                "counterfactual_score": -1.0,
                "score_delta_counterfactual_minus_baseline": -2.0,
            },
            {
                "subject_id": 111,
                "icu_stay_id": 222,
                "patient_id": "111_222",
                "counterfactual_action_found": True,
                "counterfactual_label": "suboptimal",
                "counterfactual_is_negative": True,
                "counterfactual_is_potentially_harmful": False,
                "baseline_score": 1.0,
                "counterfactual_score": -0.5,
                "score_delta_counterfactual_minus_baseline": -1.5,
            },
        ]
    )
    q3_df.to_csv(q3_dir / "q3_window_results.csv", index=False)


def test_analyze_oracle_action_validity_smoke(tmp_path) -> None:
    run_dir = tmp_path / "oracle_conditions_20260101_000000"
    (run_dir / "conditions" / "full_visible" / "patients").mkdir(parents=True, exist_ok=True)

    cohort_df = pd.DataFrame(
        [{"subject_id": 111, "icu_stay_id": 222, "survived": False, "outcome": "died"}]
    )
    cohort_df.to_csv(run_dir / "cohort_manifest.csv", index=False)

    _write_full_visible_predictions(run_dir)
    _write_q3_results(run_dir)

    result = analyzer.analyze_action_validity(
        run_dir=run_dir,
        run_q3=True,
        run_q4=True,
        jaccard_threshold=0.30,
    )

    assert "q3_analysis_dir" in result
    assert "q4_analysis_dir" in result

    q3_analysis_dir = Path(result["q3_analysis_dir"])
    q4_analysis_dir = Path(result["q4_analysis_dir"])

    assert (q3_analysis_dir / "q3_label_distribution.csv").exists()
    assert (q3_analysis_dir / "q3_paired_deltas.csv").exists()
    assert (q3_analysis_dir / "q3_summary_metrics.json").exists()
    assert (q3_analysis_dir / "q3_summary_plot.png").exists()

    assert (q4_analysis_dir / "window_overlap.csv").exists()
    assert (q4_analysis_dir / "patient_overlap.csv").exists()
    assert (q4_analysis_dir / "aggregate_metrics.csv").exists()
    assert (q4_analysis_dir / "top_unmatched_recommendations.csv").exists()
    assert (q4_analysis_dir / "q4_overlap_distributions.png").exists()

    agg_df = pd.read_csv(q4_analysis_dir / "aggregate_metrics.csv")
    metrics = {str(row["metric"]): row["value"] for _, row in agg_df.iterrows()}
    assert "micro_precision" in metrics
    assert "micro_recall" in metrics
    assert "micro_f1" in metrics
