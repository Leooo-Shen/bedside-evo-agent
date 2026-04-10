"""Analyze Oracle Q3/Q4 action-validity outputs from a full_visible baseline run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.oracle.action_validity_common import (
    DEFAULT_JACCARD_THRESHOLD,
    build_doctor_action_texts_from_events,
    build_recommendation_texts,
    compute_precision_recall_f1,
    is_finite_number,
    match_recommendations_to_actions,
)

FULL_VISIBLE_CONDITION = "full_visible"
Q3_PASS_THRESHOLD = 0.80


def _json_load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json_value(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _json_dump(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_safe_json_value(payload), f, indent=2, ensure_ascii=False)


def _iter_full_visible_prediction_paths(run_dir: Path) -> Iterable[Path]:
    patients_root = run_dir / "conditions" / FULL_VISIBLE_CONDITION / "patients"
    if not patients_root.exists():
        raise FileNotFoundError(f"Missing full_visible patient outputs: {patients_root}")

    yielded = set()
    for path in sorted(patients_root.glob("*/oracle_predictions.json")):
        yielded.add(path)
        yield path
    for path in sorted(patients_root.glob("*/*/oracle_predictions.json")):
        if path in yielded:
            continue
        yield path


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed


def _cohort_survival_map(run_dir: Path) -> Dict[str, bool]:
    cohort_path = run_dir / "cohort_manifest.csv"
    if not cohort_path.exists():
        return {}

    cohort_df = pd.read_csv(cohort_path)
    required = {"subject_id", "icu_stay_id", "survived"}
    if not required.issubset(set(cohort_df.columns)):
        return {}

    mapping: Dict[str, bool] = {}
    for _, row in cohort_df.iterrows():
        patient_id = f"{int(row['subject_id'])}_{int(row['icu_stay_id'])}"
        mapping[patient_id] = bool(row["survived"])
    return mapping


def _analyze_q3(run_dir: Path) -> Path:
    q3_root = run_dir / "action_validity" / "q3_counterfactual"
    q3_csv = q3_root / "q3_window_results.csv"
    if not q3_csv.exists():
        raise FileNotFoundError(f"Missing Q3 results file: {q3_csv}")

    q3_df = pd.read_csv(q3_csv)
    if len(q3_df) == 0:
        raise ValueError(f"Q3 results are empty: {q3_csv}")

    analysis_dir = q3_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    labels = q3_df["counterfactual_label"].fillna("<missing>").astype(str)
    label_dist = labels.value_counts().rename_axis("label").reset_index(name="count")
    label_dist["proportion"] = label_dist["count"] / float(len(q3_df))
    label_dist.to_csv(analysis_dir / "q3_label_distribution.csv", index=False)

    deltas_df = q3_df[
        pd.to_numeric(q3_df["score_delta_counterfactual_minus_baseline"], errors="coerce").notna()
    ].copy()
    deltas_df.to_csv(analysis_dir / "q3_paired_deltas.csv", index=False)

    found_mask = q3_df["counterfactual_action_found"].astype(bool)
    found_count = int(found_mask.sum())

    negative_rate = float("nan")
    harmful_rate = float("nan")
    if found_count > 0:
        negative_rate = float(q3_df.loc[found_mask, "counterfactual_is_negative"].mean())
        harmful_rate = float(q3_df.loc[found_mask, "counterfactual_is_potentially_harmful"].mean())

    finite_deltas = pd.to_numeric(
        q3_df["score_delta_counterfactual_minus_baseline"], errors="coerce"
    ).dropna()

    summary = {
        "run_dir": str(run_dir),
        "generated_at": datetime.utcnow().isoformat(),
        "num_rows": int(len(q3_df)),
        "counterfactual_action_found_count": int(found_count),
        "counterfactual_action_missing_count": int(len(q3_df) - found_count),
        "negative_label_rate": negative_rate,
        "potentially_harmful_rate": harmful_rate,
        "pass_threshold": float(Q3_PASS_THRESHOLD),
        "passes_primary": bool(is_finite_number(negative_rate) and negative_rate >= Q3_PASS_THRESHOLD),
        "mean_score_delta": float(finite_deltas.mean()) if len(finite_deltas) else None,
        "median_score_delta": float(finite_deltas.median()) if len(finite_deltas) else None,
        "score_delta_count": int(len(finite_deltas)),
    }
    _json_dump(analysis_dir / "q3_summary_metrics.json", summary)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(label_dist["label"], label_dist["count"], color="#2f4b7c")
    axes[0].set_title("Q3 Counterfactual Label Distribution")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=20)

    baseline_scores = pd.to_numeric(q3_df["baseline_score"], errors="coerce")
    counter_scores = pd.to_numeric(q3_df["counterfactual_score"], errors="coerce")
    pair_mask = baseline_scores.notna() & counter_scores.notna()

    if pair_mask.sum() > 0:
        x = np.array([0, 1])
        for base_val, counter_val in zip(baseline_scores[pair_mask], counter_scores[pair_mask]):
            axes[1].plot(x, [base_val, counter_val], color="#95a5a6", alpha=0.5)
        axes[1].scatter(
            np.zeros(pair_mask.sum()),
            baseline_scores[pair_mask],
            color="#1f77b4",
            label="baseline",
        )
        axes[1].scatter(
            np.ones(pair_mask.sum()),
            counter_scores[pair_mask],
            color="#d62728",
            label="counterfactual",
        )
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(["baseline", "counterfactual"])
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No paired score rows", ha="center", va="center")

    axes[1].set_title("Q3 Baseline vs Counterfactual Scores")
    axes[1].set_ylabel("Action score")
    axes[1].axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1)

    fig.tight_layout()
    fig.savefig(analysis_dir / "q3_summary_plot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return analysis_dir


def _build_q4_window_dataframe(
    run_dir: Path,
    *,
    jaccard_threshold: float,
) -> Tuple[pd.DataFrame, Counter]:
    cohort_map = _cohort_survival_map(run_dir)
    rows: List[Dict[str, Any]] = []
    unmatched_counter: Counter = Counter()

    for pred_path in _iter_full_visible_prediction_paths(run_dir):
        payload = _json_load(pred_path)
        subject_id = _safe_int(payload.get("subject_id"), default=-1)
        icu_stay_id = _safe_int(payload.get("icu_stay_id"), default=-1)
        if subject_id < 0 or icu_stay_id < 0:
            continue

        patient_id = f"{subject_id}_{icu_stay_id}"
        trajectory_meta = payload.get("trajectory_metadata")
        if not isinstance(trajectory_meta, dict):
            trajectory_meta = {}

        true_survived = trajectory_meta.get("true_survived")
        if true_survived is None:
            true_survived = trajectory_meta.get("survived")
        if true_survived is None and patient_id in cohort_map:
            true_survived = cohort_map[patient_id]
        true_survived = bool(true_survived)

        window_outputs = payload.get("window_outputs")
        if not isinstance(window_outputs, list):
            continue

        for idx, window_output in enumerate(window_outputs):
            if not isinstance(window_output, dict):
                continue
            window_index = _safe_int(window_output.get("window_index"), default=idx)
            window_meta = window_output.get("window_metadata")
            if not isinstance(window_meta, dict):
                window_meta = {}

            raw_current_events = window_output.get("raw_current_events")
            if not isinstance(raw_current_events, list):
                raw_current_events = []

            oracle_output = window_output.get("oracle_output")
            if not isinstance(oracle_output, dict):
                oracle_output = {}

            action_review = oracle_output.get("action_review")
            evaluations = action_review.get("evaluations") if isinstance(action_review, dict) else []
            recommendation_texts = build_recommendation_texts(evaluations)
            doctor_action_texts = build_doctor_action_texts_from_events(raw_current_events)

            match_result = match_recommendations_to_actions(
                recommendation_texts,
                doctor_action_texts,
                jaccard_threshold=float(jaccard_threshold),
                min_shared_tokens=2,
            )
            precision, recall, f1 = compute_precision_recall_f1(
                num_matches=int(match_result["num_matches"]),
                num_recommendations=int(match_result["num_recommendations"]),
                num_doctor_actions=int(match_result["num_doctor_actions"]),
            )

            for unmatched_idx in match_result["unmatched_recommendation_indices"]:
                text = recommendation_texts[unmatched_idx]
                if text:
                    unmatched_counter[text] += 1

            rows.append(
                {
                    "subject_id": subject_id,
                    "icu_stay_id": icu_stay_id,
                    "patient_id": patient_id,
                    "true_survived": true_survived,
                    "true_outcome": "survived" if true_survived else "died",
                    "window_index": window_index,
                    "hours_since_admission": _safe_float(window_meta.get("hours_since_admission")),
                    "num_recommendations": int(match_result["num_recommendations"]),
                    "num_doctor_actions": int(match_result["num_doctor_actions"]),
                    "num_matches": int(match_result["num_matches"]),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "has_recommendations": int(match_result["num_recommendations"]) > 0,
                    "has_doctor_actions": int(match_result["num_doctor_actions"]) > 0,
                    "has_both": int(match_result["num_recommendations"]) > 0
                    and int(match_result["num_doctor_actions"]) > 0,
                }
            )

    df = pd.DataFrame(rows)
    return df, unmatched_counter


def _macro_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) == 0:
        return float("nan")
    return float(numeric.mean())


def _analyze_q4(run_dir: Path, *, jaccard_threshold: float) -> Path:
    q4_dir = run_dir / "action_validity" / "q4_overlap"
    q4_dir.mkdir(parents=True, exist_ok=True)

    window_df, unmatched_counter = _build_q4_window_dataframe(
        run_dir,
        jaccard_threshold=float(jaccard_threshold),
    )
    if len(window_df) == 0:
        raise ValueError("No full_visible windows found for Q4 overlap analysis.")

    window_df.to_csv(q4_dir / "window_overlap.csv", index=False)

    patient_rows: List[Dict[str, Any]] = []
    for (patient_id, true_outcome), group in window_df.groupby(["patient_id", "true_outcome"]):
        num_recs = int(group["num_recommendations"].sum())
        num_actions = int(group["num_doctor_actions"].sum())
        num_matches = int(group["num_matches"].sum())
        precision, recall, f1 = compute_precision_recall_f1(
            num_matches=num_matches,
            num_recommendations=num_recs,
            num_doctor_actions=num_actions,
        )

        subject_id = int(group["subject_id"].iloc[0])
        icu_stay_id = int(group["icu_stay_id"].iloc[0])

        patient_rows.append(
            {
                "subject_id": subject_id,
                "icu_stay_id": icu_stay_id,
                "patient_id": patient_id,
                "true_outcome": true_outcome,
                "windows": int(len(group)),
                "num_recommendations": num_recs,
                "num_doctor_actions": num_actions,
                "num_matches": num_matches,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_window_precision": _macro_mean(group["precision"]),
                "mean_window_recall": _macro_mean(group["recall"]),
                "mean_window_f1": _macro_mean(group["f1"]),
            }
        )

    patient_df = pd.DataFrame(patient_rows)
    patient_df.to_csv(q4_dir / "patient_overlap.csv", index=False)

    total_matches = int(window_df["num_matches"].sum())
    total_recommendations = int(window_df["num_recommendations"].sum())
    total_doctor_actions = int(window_df["num_doctor_actions"].sum())
    micro_precision, micro_recall, micro_f1 = compute_precision_recall_f1(
        num_matches=total_matches,
        num_recommendations=total_recommendations,
        num_doctor_actions=total_doctor_actions,
    )

    aggregate_metrics = {
        "generated_at": datetime.utcnow().isoformat(),
        "run_dir": str(run_dir),
        "jaccard_threshold": float(jaccard_threshold),
        "num_windows": int(len(window_df)),
        "num_patients": int(window_df[["patient_id"]].drop_duplicates().shape[0]),
        "windows_with_recommendations": int(window_df["has_recommendations"].sum()),
        "windows_with_doctor_actions": int(window_df["has_doctor_actions"].sum()),
        "windows_with_both": int(window_df["has_both"].sum()),
        "total_recommendations": int(total_recommendations),
        "total_doctor_actions": int(total_doctor_actions),
        "total_matches": int(total_matches),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_window_precision": _macro_mean(window_df["precision"]),
        "macro_window_recall": _macro_mean(window_df["recall"]),
        "macro_window_f1": _macro_mean(window_df["f1"]),
        "macro_patient_precision": _macro_mean(patient_df["precision"]),
        "macro_patient_recall": _macro_mean(patient_df["recall"]),
        "macro_patient_f1": _macro_mean(patient_df["f1"]),
    }

    aggregate_df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in aggregate_metrics.items()]
    )
    aggregate_df.to_csv(q4_dir / "aggregate_metrics.csv", index=False)
    _json_dump(q4_dir / "aggregate_metrics.json", aggregate_metrics)

    top_unmatched_df = pd.DataFrame(
        [{"recommendation_text": text, "count": int(count)} for text, count in unmatched_counter.most_common(200)]
    )
    top_unmatched_df.to_csv(q4_dir / "top_unmatched_recommendations.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    window_f1 = pd.to_numeric(window_df["f1"], errors="coerce").dropna()
    if len(window_f1) > 0:
        axes[0].hist(window_f1, bins=20, color="#2a9d8f", edgecolor="#264653", alpha=0.9)
    else:
        axes[0].text(0.5, 0.5, "No defined window F1", ha="center", va="center")
    axes[0].set_title("Q4 Window F1 Distribution")
    axes[0].set_xlabel("F1")
    axes[0].set_ylabel("Count")

    patient_f1 = patient_df[["patient_id", "f1"]].copy()
    patient_f1["f1"] = pd.to_numeric(patient_f1["f1"], errors="coerce")
    patient_f1 = patient_f1.sort_values("f1", ascending=False)
    if len(patient_f1) > 0 and patient_f1["f1"].notna().any():
        axes[1].bar(patient_f1["patient_id"], patient_f1["f1"], color="#457b9d")
        axes[1].tick_params(axis="x", rotation=70)
    else:
        axes[1].text(0.5, 0.5, "No defined patient F1", ha="center", va="center")
    axes[1].set_title("Q4 Patient-Level Micro F1")
    axes[1].set_ylabel("F1")

    fig.tight_layout()
    fig.savefig(q4_dir / "q4_overlap_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return q4_dir


def analyze_action_validity(
    *,
    run_dir: Path,
    run_q3: bool,
    run_q4: bool,
    jaccard_threshold: float,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    results: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "generated_at": datetime.utcnow().isoformat(),
        "q3_enabled": bool(run_q3),
        "q4_enabled": bool(run_q4),
        "jaccard_threshold": float(jaccard_threshold),
    }

    if run_q3:
        q3_analysis_dir = _analyze_q3(run_dir)
        results["q3_analysis_dir"] = str(q3_analysis_dir)

    if run_q4:
        q4_analysis_dir = _analyze_q4(run_dir, jaccard_threshold=float(jaccard_threshold))
        results["q4_analysis_dir"] = str(q4_analysis_dir)

    summary_path = run_dir / "action_validity" / "analysis_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _json_dump(summary_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Oracle action-validity Q3/Q4 outputs.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to existing oracle_conditions_* run.")
    parser.add_argument("--q3", action="store_true", help="Run Q3 analysis only.")
    parser.add_argument("--q4", action="store_true", help="Run Q4 analysis only.")
    parser.add_argument("--all", action="store_true", help="Run all analyses (default if no mode is specified).")
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=DEFAULT_JACCARD_THRESHOLD,
        help=f"Jaccard threshold for lexical recommendation-action matching (default: {DEFAULT_JACCARD_THRESHOLD:.2f}).",
    )
    args = parser.parse_args()

    run_q3 = bool(args.q3)
    run_q4 = bool(args.q4)
    if args.all or (not args.q3 and not args.q4):
        run_q3 = True
        run_q4 = True

    results = analyze_action_validity(
        run_dir=Path(args.run_dir),
        run_q3=run_q3,
        run_q4=run_q4,
        jaccard_threshold=max(0.0, float(args.jaccard_threshold)),
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
