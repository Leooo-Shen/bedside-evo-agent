"""Run Q3 counterfactual wrong-action injection on an existing full_visible Oracle run."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.oracle import MetaOracle
from config.config import Config, load_config
from data_parser import MIMICDataParser
from experiments.oracle.action_validity_common import (
    NEGATIVE_ACTION_LABELS,
    action_label_to_score,
    count_actionable_events,
    extract_action_label,
    identify_action_evaluation,
    inject_counterfactual_current_event,
    is_finite_number,
    select_wrong_action_template,
)

FULL_VISIBLE_CONDITION = "full_visible"
PASS_THRESHOLD = 0.80


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


def _load_run_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    return _json_load(manifest_path)


def _load_cohort(run_dir: Path, max_patients: int) -> pd.DataFrame:
    cohort_path = run_dir / "cohort_manifest.csv"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Missing cohort manifest: {cohort_path}")

    cohort_df = pd.read_csv(cohort_path)
    required_cols = {"subject_id", "icu_stay_id", "survived"}
    missing = required_cols - set(cohort_df.columns)
    if missing:
        raise ValueError(f"Invalid cohort manifest missing columns: {sorted(missing)}")

    cohort_df = cohort_df[["subject_id", "icu_stay_id", "survived"]].copy()
    cohort_df["subject_id"] = cohort_df["subject_id"].astype(int)
    cohort_df["icu_stay_id"] = cohort_df["icu_stay_id"].astype(int)
    cohort_df["survived"] = cohort_df["survived"].astype(bool)

    if max_patients > 0:
        cohort_df = cohort_df.head(max_patients).copy()

    return cohort_df.reset_index(drop=True)


def _iter_full_visible_prediction_paths(run_dir: Path) -> List[Path]:
    patients_root = run_dir / "conditions" / FULL_VISIBLE_CONDITION / "patients"
    if not patients_root.exists():
        raise FileNotFoundError(f"Missing full_visible patient outputs: {patients_root}")

    direct = sorted(patients_root.glob("*/oracle_predictions.json"))
    nested = sorted(patients_root.glob("*/*/oracle_predictions.json"))
    all_paths = direct + [path for path in nested if path not in set(direct)]
    if not all_paths:
        raise FileNotFoundError(f"No full_visible oracle_predictions.json found under {patients_root}")
    return all_paths


def _load_full_visible_predictions(run_dir: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    mapping: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for path in _iter_full_visible_prediction_paths(run_dir):
        payload = _json_load(path)
        try:
            subject_id = int(payload.get("subject_id"))
            icu_stay_id = int(payload.get("icu_stay_id"))
        except (TypeError, ValueError):
            continue
        mapping[(subject_id, icu_stay_id)] = payload
    return mapping


def _select_first_action_rich_window(
    baseline_payload: Dict[str, Any],
    min_action_events: int,
) -> Optional[Dict[str, Any]]:
    window_outputs = baseline_payload.get("window_outputs")
    if not isinstance(window_outputs, list):
        return None

    for idx, window_output in enumerate(window_outputs):
        if not isinstance(window_output, dict):
            continue
        raw_current_events = window_output.get("raw_current_events")
        if not isinstance(raw_current_events, list):
            raw_current_events = []

        actionable_count = count_actionable_events(raw_current_events)
        if actionable_count < int(min_action_events):
            continue

        window_index_raw = window_output.get("window_index")
        try:
            window_index = int(window_index_raw)
        except (TypeError, ValueError):
            window_index = idx

        metadata = window_output.get("window_metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        return {
            "window_index": window_index,
            "actionable_event_count": actionable_count,
            "window_metadata": metadata,
            "window_output": window_output,
        }

    return None


def _get_trajectory_row(parser: Any, subject_id: int, icu_stay_id: int) -> Optional[pd.Series]:
    icu_stay_df = getattr(parser, "icu_stay_df", None)
    if not isinstance(icu_stay_df, pd.DataFrame):
        return None

    matched = icu_stay_df[
        (icu_stay_df["subject_id"].astype(int) == int(subject_id))
        & (icu_stay_df["icu_stay_id"].astype(int) == int(icu_stay_id))
    ]
    if len(matched) == 0:
        return None
    return matched.iloc[0]


def _sort_llm_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(item: Dict[str, Any]) -> Tuple[int, float, str]:
        try:
            window_idx = int(item.get("window_index"))
        except (TypeError, ValueError):
            window_idx = 10**9
        try:
            hours = float(item.get("hours_since_admission"))
        except (TypeError, ValueError):
            hours = float("inf")
        timestamp = str(item.get("timestamp") or "")
        return (window_idx, hours, timestamp)

    return sorted([call for call in calls if isinstance(call, dict)], key=_key)


def _build_parser(events_path: str, icu_stay_path: str) -> Any:
    try:
        parser = MIMICDataParser(
            events_path,
            icu_stay_path,
            require_discharge_summary_for_icu_stays=True,
        )
    except TypeError:
        parser = MIMICDataParser(events_path, icu_stay_path)
    parser.load_data()
    return parser


def _resolve_provider_model(
    *,
    config: Config,
    run_manifest: Dict[str, Any],
    provider: Optional[str],
    model: Optional[str],
) -> Tuple[str, Optional[str]]:
    resolved_provider = provider or str(run_manifest.get("provider") or config.llm_provider)
    resolved_model = model if model is not None else run_manifest.get("model")
    if resolved_model is None:
        resolved_model = config.llm_model
    return resolved_provider, resolved_model


def run_counterfactual_action_experiment(
    *,
    run_dir: Path,
    config: Config,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    min_action_events: int = 3,
    max_patients: int = 20,
) -> Path:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_manifest = _load_run_manifest(run_dir)
    cohort_df = _load_cohort(run_dir, max_patients=max_patients)
    baseline_map = _load_full_visible_predictions(run_dir)

    selected_rows: List[Dict[str, Any]] = []
    failures: List[str] = []

    for _, row in cohort_df.iterrows():
        subject_id = int(row["subject_id"])
        icu_stay_id = int(row["icu_stay_id"])
        patient_id = f"{subject_id}_{icu_stay_id}"

        baseline_payload = baseline_map.get((subject_id, icu_stay_id))
        if baseline_payload is None:
            failures.append(f"{patient_id}: missing full_visible oracle_predictions.json")
            continue

        candidate = _select_first_action_rich_window(
            baseline_payload,
            min_action_events=int(min_action_events),
        )
        if candidate is None:
            failures.append(
                f"{patient_id}: no window with at least {int(min_action_events)} actionable events"
            )
            continue

        selected_rows.append(
            {
                "subject_id": subject_id,
                "icu_stay_id": icu_stay_id,
                "patient_id": patient_id,
                "true_survived": bool(row["survived"]),
                "baseline_payload": baseline_payload,
                "candidate": candidate,
            }
        )

    if failures:
        joined = "\n".join(f"- {item}" for item in failures)
        raise ValueError(f"Failed eligibility checks for Q3 selection:\n{joined}")

    events_path = str(run_manifest.get("events_path") or config.events_path)
    icu_stay_path = str(run_manifest.get("icu_stay_path") or config.icu_stay_path)
    current_window_hours = float(run_manifest.get("current_window_hours", 1.0))
    window_step_hours = float(run_manifest.get("window_step_hours", 2.0))
    include_pre_icu_data = bool(run_manifest.get("include_pre_icu_data", True))

    resolved_provider, resolved_model = _resolve_provider_model(
        config=config,
        run_manifest=run_manifest,
        provider=provider,
        model=model,
    )

    parser = _build_parser(events_path=events_path, icu_stay_path=icu_stay_path)

    oracle = MetaOracle(
        provider=resolved_provider,
        model=resolved_model,
        use_discharge_summary=True,
        include_icu_outcome_in_prompt=True,
        mask_discharge_summary_outcome_terms=False,
        history_context_hours=config.oracle_context_history_hours,
        future_context_hours=config.oracle_context_future_hours,
        top_k_recommendations=config.oracle_context_top_k_recommendations,
        log_dir=str(run_dir / "action_validity" / "logs" / "oracle"),
    )

    q3_output_dir = run_dir / "action_validity" / "q3_counterfactual"
    q3_output_dir.mkdir(parents=True, exist_ok=True)

    injection_manifest_rows: List[Dict[str, Any]] = []
    q3_rows: List[Dict[str, Any]] = []
    llm_jsonl_rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(selected_rows, start=1):
        subject_id = int(item["subject_id"])
        icu_stay_id = int(item["icu_stay_id"])
        patient_id = item["patient_id"]
        true_survived = bool(item["true_survived"])
        candidate = item["candidate"]
        baseline_payload = item["baseline_payload"]
        selected_window_index = int(candidate["window_index"])

        print(
            f"[Q3] {idx}/{len(selected_rows)} patient={patient_id} "
            f"window={selected_window_index} actionable={candidate['actionable_event_count']}"
        )

        trajectory_row = _get_trajectory_row(parser, subject_id=subject_id, icu_stay_id=icu_stay_id)
        if trajectory_row is not None:
            trajectory = parser.get_patient_trajectory(subject_id, icu_stay_id, icu_stay=trajectory_row)
        else:
            trajectory = parser.get_patient_trajectory(subject_id, icu_stay_id)

        windows = parser.create_time_windows(
            trajectory,
            current_window_hours=current_window_hours,
            window_step_hours=window_step_hours,
            include_pre_icu_data=include_pre_icu_data,
            use_first_n_hours_after_icu=None,
            use_discharge_summary_for_history=config.oracle_use_discharge_summary_for_history,
            num_discharge_summaries=config.oracle_num_discharge_summaries,
            relative_report_codes=config.oracle_relative_report_codes,
            pre_icu_history_hours=config.oracle_pre_icu_history_hours,
            history_context_hours=config.oracle_context_history_hours,
            future_context_hours=config.oracle_context_future_hours,
        )

        if selected_window_index < 0 or selected_window_index >= len(windows):
            raise ValueError(
                f"Patient {patient_id} selected window {selected_window_index} not found in rebuilt windows "
                f"(available={len(windows)})"
            )

        baseline_window_output = candidate["window_output"]
        baseline_oracle_output = baseline_window_output.get("oracle_output")
        if not isinstance(baseline_oracle_output, dict):
            baseline_oracle_output = {}

        base_window = copy.deepcopy(windows[selected_window_index])
        base_window["window_index"] = selected_window_index

        template = select_wrong_action_template(base_window.get("current_events") or [])
        marker_token = f"CFX_WRONG_ACTION_{patient_id}_W{selected_window_index}"
        injected_window, injected_event, expected_action_id = inject_counterfactual_current_event(
            base_window,
            marker_token=marker_token,
            wrong_action_text=str(template["action_text"]),
        )
        injected_window["window_index"] = selected_window_index

        report = oracle.evaluate_window(injected_window)
        report_dict = report.to_dict() if hasattr(report, "to_dict") else {}
        report_action_review = report_dict.get("action_review")
        report_evaluations = report_action_review.get("evaluations") if isinstance(report_action_review, dict) else []

        llm_calls = oracle.pop_patient_llm_call_logs(subject_id=subject_id, icu_stay_id=icu_stay_id)
        oracle.pop_patient_trajectory_logs(subject_id=subject_id, icu_stay_id=icu_stay_id)
        llm_calls = _sort_llm_calls(llm_calls)

        counter_eval = identify_action_evaluation(
            report_evaluations,
            expected_action_id=expected_action_id,
            marker_token=marker_token,
        )
        counter_label = extract_action_label(counter_eval)
        counter_score = action_label_to_score(counter_label)

        baseline_action_review = baseline_oracle_output.get("action_review")
        baseline_evaluations = (
            baseline_action_review.get("evaluations") if isinstance(baseline_action_review, dict) else []
        )
        baseline_eval = identify_action_evaluation(
            baseline_evaluations,
            expected_action_id=expected_action_id,
            marker_token=None,
        )
        baseline_label = extract_action_label(baseline_eval)
        baseline_score = action_label_to_score(baseline_label)

        score_delta = float("nan")
        if is_finite_number(counter_score) and is_finite_number(baseline_score):
            score_delta = float(counter_score) - float(baseline_score)

        counter_found = isinstance(counter_eval, dict)
        baseline_found = isinstance(baseline_eval, dict)

        window_meta = candidate.get("window_metadata")
        if not isinstance(window_meta, dict):
            window_meta = {}

        q3_row = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "patient_id": patient_id,
            "true_survived": true_survived,
            "true_outcome": "survived" if true_survived else "died",
            "window_index": selected_window_index,
            "hours_since_admission": window_meta.get("hours_since_admission"),
            "actionable_events_in_baseline_window": int(candidate["actionable_event_count"]),
            "template_id": template.get("template_id"),
            "template_trigger": template.get("trigger"),
            "template_action_text": template.get("action_text"),
            "marker_token": marker_token,
            "expected_action_id": expected_action_id,
            "baseline_action_found": baseline_found,
            "baseline_label": baseline_label,
            "baseline_score": baseline_score,
            "counterfactual_action_found": counter_found,
            "counterfactual_label": counter_label,
            "counterfactual_score": counter_score,
            "counterfactual_is_negative": counter_label in NEGATIVE_ACTION_LABELS,
            "counterfactual_is_potentially_harmful": counter_label == "potentially_harmful",
            "score_delta_counterfactual_minus_baseline": score_delta,
        }
        q3_rows.append(q3_row)

        injection_manifest_rows.append(
            {
                "subject_id": subject_id,
                "icu_stay_id": icu_stay_id,
                "patient_id": patient_id,
                "window_index": selected_window_index,
                "actionable_events_in_baseline_window": int(candidate["actionable_event_count"]),
                "expected_action_id": expected_action_id,
                "template_id": template.get("template_id"),
                "template_trigger": template.get("trigger"),
                "template_action_text": template.get("action_text"),
                "marker_token": marker_token,
                "injected_event": injected_event,
            }
        )

        llm_jsonl_rows.append(
            {
                "subject_id": subject_id,
                "icu_stay_id": icu_stay_id,
                "patient_id": patient_id,
                "window_index": selected_window_index,
                "expected_action_id": expected_action_id,
                "marker_token": marker_token,
                "template_id": template.get("template_id"),
                "template_action_text": template.get("action_text"),
                "llm_calls": llm_calls,
            }
        )

    injection_manifest_df = pd.DataFrame(injection_manifest_rows)
    injection_manifest_df.to_csv(q3_output_dir / "injection_manifest.csv", index=False)
    _json_dump(q3_output_dir / "injection_manifest.json", injection_manifest_rows)

    q3_df = pd.DataFrame(q3_rows)
    q3_df.to_csv(q3_output_dir / "q3_window_results.csv", index=False)

    with open(q3_output_dir / "counterfactual_llm_calls.jsonl", "w", encoding="utf-8") as f:
        for row in llm_jsonl_rows:
            f.write(json.dumps(_safe_json_value(row), ensure_ascii=False) + "\n")

    if len(q3_df) == 0:
        raise ValueError("Q3 run produced zero rows.")

    found_mask = q3_df["counterfactual_action_found"].astype(bool)
    found_count = int(found_mask.sum())
    negative_rate = float("nan")
    harmful_rate = float("nan")
    if found_count > 0:
        negative_rate = float(q3_df.loc[found_mask, "counterfactual_is_negative"].mean())
        harmful_rate = float(q3_df.loc[found_mask, "counterfactual_is_potentially_harmful"].mean())

    delta_series = q3_df["score_delta_counterfactual_minus_baseline"]
    finite_deltas = delta_series[pd.to_numeric(delta_series, errors="coerce").notna()]

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "run_dir": str(run_dir),
        "condition_source": FULL_VISIBLE_CONDITION,
        "num_patients": int(len(q3_df)),
        "min_action_events": int(min_action_events),
        "max_patients": int(max_patients),
        "negative_label_rate": negative_rate,
        "potentially_harmful_rate": harmful_rate,
        "pass_threshold": float(PASS_THRESHOLD),
        "passes_primary": bool(is_finite_number(negative_rate) and negative_rate >= PASS_THRESHOLD),
        "counterfactual_action_found_count": int(found_count),
        "counterfactual_action_missing_count": int(len(q3_df) - found_count),
        "mean_score_delta": float(finite_deltas.mean()) if len(finite_deltas) else None,
        "median_score_delta": float(finite_deltas.median()) if len(finite_deltas) else None,
        "score_delta_count": int(len(finite_deltas)),
        "counterfactual_label_distribution": {
            str(label): int(count)
            for label, count in q3_df["counterfactual_label"].fillna("<missing>").value_counts().items()
        },
    }
    _json_dump(q3_output_dir / "q3_summary.json", summary)

    print(f"Q3 counterfactual outputs saved under: {q3_output_dir}")
    return q3_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Oracle Q3 counterfactual action-injection experiment.")
    parser.add_argument("--run-dir", type=str, required=True, help="Existing oracle_conditions_* run directory.")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider override.")
    parser.add_argument("--model", type=str, default=None, help="LLM model override.")
    parser.add_argument(
        "--min-action-events",
        type=int,
        default=3,
        help="Minimum actionable events required to select a baseline window (default: 3).",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=20,
        help="Maximum patients to run (default: 20).",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional config path override.")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()

    run_counterfactual_action_experiment(
        run_dir=Path(args.run_dir),
        config=config,
        provider=args.provider,
        model=args.model,
        min_action_events=max(1, int(args.min_action_events)),
        max_patients=max(1, int(args.max_patients)),
    )


if __name__ == "__main__":
    main()
