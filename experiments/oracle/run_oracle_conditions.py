"""Run Oracle experiments across full_visible / masked_outcome / reversed_outcome conditions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.oracle import MetaOracle
from config.config import Config, load_config
from data_parser import MIMICDataParser
from experiments.oracle.common import (
    ConditionSpec,
    apply_prompt_outcome_mode,
    build_default_condition_specs,
    extract_overall_label,
    mask_window_outcome_leakage,
)
from run_oracle import (
    _build_oracle_llm_calls_payload,
    _build_patient_predictions_payload,
    _build_window_contexts_payload,
    _json_default,
)
from utils.llm_log_viewer import save_llm_calls_html
from utils.patient_selection import DEFAULT_SELECTION_SEED, select_balanced_patients

RUN_PREFIX = "oracle_conditions_"
RUN_STATE_FILENAME = "run_state.json"
REQUIRED_PATIENT_OUTPUT_FILES = (
    "oracle_predictions.json",
    "llm_calls.json",
    "window_contexts.json",
)


def _extract_selected_discharge_pairs(selection_df: Optional[pd.DataFrame]) -> set[tuple[int, int]]:
    if selection_df is None or len(selection_df) == 0:
        return set()

    selected_rows = (
        selection_df[selection_df["selected"] == True] if "selected" in selection_df.columns else selection_df
    )  # noqa: E712
    pairs = set()
    for _, row in selected_rows.iterrows():
        if pd.isna(row.get("subject_id")) or pd.isna(row.get("icu_stay_id")):
            continue
        pairs.add((int(row["subject_id"]), int(row["icu_stay_id"])))
    return pairs


def _filter_eligible_stays_with_discharge_summary(parser: Any) -> pd.DataFrame:
    icu_stay_df = getattr(parser, "icu_stay_df", None)
    if icu_stay_df is None:
        raise ValueError("Parser does not have icu_stay_df loaded.")
    selection_df = getattr(parser, "discharge_summary_selection_df", None)
    selected_pairs = _extract_selected_discharge_pairs(selection_df)
    if not selected_pairs:
        raise ValueError("No ICU stays with selected discharge summaries were found.")

    eligible = icu_stay_df[
        icu_stay_df.apply(
            lambda row: (int(row["subject_id"]), int(row["icu_stay_id"])) in selected_pairs,
            axis=1,
        )
    ].reset_index(drop=True)
    return eligible


def _save_cohort_manifest(cohort_df: pd.DataFrame, output_dir: Path) -> None:
    manifest_df = cohort_df.copy()
    manifest_df["outcome"] = manifest_df["survived"].map(lambda x: "survived" if bool(x) else "died")
    manifest_df = manifest_df[["subject_id", "icu_stay_id", "survived", "outcome"]]
    manifest_df.to_csv(output_dir / "cohort_manifest.csv", index=False)
    with open(output_dir / "cohort_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_df.to_dict("records"), f, indent=2, ensure_ascii=False)


def _resolve_conditions(condition_names: Optional[Iterable[str]]) -> List[ConditionSpec]:
    available = build_default_condition_specs()
    if not condition_names:
        return [available[name] for name in ("full_visible", "masked_outcome", "reversed_outcome")]

    resolved: List[ConditionSpec] = []
    for raw_name in condition_names:
        name = str(raw_name).strip().lower()
        if not name:
            continue
        if name not in available:
            raise ValueError(f"Unknown condition: {name}. Available: {', '.join(sorted(available))}")
        resolved.append(available[name])
    if not resolved:
        raise ValueError("No valid conditions selected.")
    return resolved


def _json_dump(data: Any, output_path: Path, json_default: Any) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_default)


def _json_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return bool(value)


def _load_cohort_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing cohort manifest: {manifest_path}")
    cohort_df = pd.read_csv(manifest_path)
    required_columns = {"subject_id", "icu_stay_id", "survived"}
    missing_columns = required_columns - set(cohort_df.columns)
    if missing_columns:
        raise ValueError(f"Invalid cohort manifest {manifest_path}: missing columns {sorted(missing_columns)}")
    normalized = cohort_df[["subject_id", "icu_stay_id", "survived"]].copy()
    normalized["subject_id"] = normalized["subject_id"].astype(int)
    normalized["icu_stay_id"] = normalized["icu_stay_id"].astype(int)
    normalized["survived"] = normalized["survived"].map(_parse_bool)
    return normalized.reset_index(drop=True)


def _extract_existing_patient_stats(patient_dir: Path) -> Optional[Dict[str, Any]]:
    if not patient_dir.exists():
        return None
    if not all((patient_dir / filename).exists() for filename in REQUIRED_PATIENT_OUTPUT_FILES):
        return None

    predictions_payload = _json_load(patient_dir / "oracle_predictions.json")
    if predictions_payload is None:
        return None

    window_outputs = predictions_payload.get("window_outputs")
    if not isinstance(window_outputs, list):
        window_outputs = []

    status_distribution: Dict[str, int] = {}
    for window_output in window_outputs:
        if not isinstance(window_output, dict):
            continue
        oracle_output = window_output.get("oracle_output")
        label = extract_overall_label(oracle_output if isinstance(oracle_output, dict) else {})
        status_distribution[label] = status_distribution.get(label, 0) + 1

    explicit_windows = max(
        _safe_int(predictions_payload.get("num_windows_evaluated"), default=-1),
        len(window_outputs),
    )

    return {
        "num_windows_evaluated": explicit_windows,
        "overall_status_distribution": status_distribution,
    }


def _values_match(left: Any, right: Any) -> bool:
    left_text = "" if left is None else str(left).strip()
    right_text = "" if right is None else str(right).strip()
    normalized_left = None if left_text in {"", "None", "none"} else left_text
    normalized_right = None if right_text in {"", "None", "none"} else right_text
    return normalized_left == normalized_right


def _float_match(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _matches_run_metadata(
    metadata: Dict[str, Any],
    *,
    selected_conditions: List[ConditionSpec],
    provider: str,
    model: Optional[str],
    current_window_hours: float,
    window_step_hours: float,
    include_pre_icu_data: bool,
    n_survived: int,
    n_died: int,
    selection_seed: int,
    use_discharge_summary_for_history: bool,
    num_discharge_summaries: int,
    relative_report_codes: List[str],
    pre_icu_history_hours: float,
    context_history_hours: float,
    context_future_hours: float,
    require_discharge_summary_for_icu_stays: bool,
) -> tuple[bool, int]:
    checks_applied = 0
    expected_conditions = [condition.name for condition in selected_conditions]

    if "conditions" in metadata:
        checks_applied += 1
        raw_conditions = metadata.get("conditions")
        if not isinstance(raw_conditions, list):
            return False, checks_applied
        normalized_conditions = [str(name).strip().lower() for name in raw_conditions]
        if sorted(normalized_conditions) != sorted(expected_conditions):
            return False, checks_applied

    if "provider" in metadata:
        checks_applied += 1
        if not _values_match(metadata.get("provider"), provider):
            return False, checks_applied

    if "model" in metadata:
        checks_applied += 1
        if not _values_match(metadata.get("model"), model):
            return False, checks_applied

    if "selection_seed" in metadata:
        checks_applied += 1
        if _safe_int(metadata.get("selection_seed"), default=-1) != int(selection_seed):
            return False, checks_applied

    if "n_survived" in metadata:
        checks_applied += 1
        if _safe_int(metadata.get("n_survived"), default=-1) != int(n_survived):
            return False, checks_applied

    if "n_died" in metadata:
        checks_applied += 1
        if _safe_int(metadata.get("n_died"), default=-1) != int(n_died):
            return False, checks_applied

    if "current_window_hours" in metadata:
        checks_applied += 1
        if not _float_match(metadata.get("current_window_hours"), current_window_hours):
            return False, checks_applied

    if "window_step_hours" in metadata:
        checks_applied += 1
        if not _float_match(metadata.get("window_step_hours"), window_step_hours):
            return False, checks_applied

    if "include_pre_icu_data" in metadata:
        checks_applied += 1
        if _parse_bool(metadata.get("include_pre_icu_data")) != bool(include_pre_icu_data):
            return False, checks_applied

    if "use_discharge_summary_for_history" in metadata:
        checks_applied += 1
        if _parse_bool(metadata.get("use_discharge_summary_for_history")) != bool(use_discharge_summary_for_history):
            return False, checks_applied

    if "num_discharge_summaries" in metadata:
        checks_applied += 1
        if _safe_int(metadata.get("num_discharge_summaries"), default=-1) != int(num_discharge_summaries):
            return False, checks_applied

    if "relative_report_codes" in metadata:
        checks_applied += 1
        raw_codes = metadata.get("relative_report_codes")
        if not isinstance(raw_codes, list):
            return False, checks_applied
        normalized_codes = [str(code).strip() for code in raw_codes if str(code).strip()]
        expected_codes = [str(code).strip() for code in relative_report_codes if str(code).strip()]
        if normalized_codes != expected_codes:
            return False, checks_applied

    if "pre_icu_history_hours" in metadata:
        checks_applied += 1
        if not _float_match(metadata.get("pre_icu_history_hours"), pre_icu_history_hours):
            return False, checks_applied

    if "context_history_hours" in metadata:
        checks_applied += 1
        if not _float_match(metadata.get("context_history_hours"), context_history_hours):
            return False, checks_applied

    if "context_future_hours" in metadata:
        checks_applied += 1
        if not _float_match(metadata.get("context_future_hours"), context_future_hours):
            return False, checks_applied

    if "require_discharge_summary_for_icu_stays" in metadata:
        checks_applied += 1
        if _parse_bool(metadata.get("require_discharge_summary_for_icu_stays")) != bool(
            require_discharge_summary_for_icu_stays
        ):
            return False, checks_applied

    return True, checks_applied


def _cohort_matches_target_counts(manifest_path: Path, *, n_survived: int, n_died: int) -> bool:
    try:
        cohort_df = _load_cohort_manifest(manifest_path)
    except (FileNotFoundError, ValueError, TypeError):
        return False
    survived_count = int((cohort_df["survived"] == True).sum())  # noqa: E712
    died_count = int((cohort_df["survived"] == False).sum())  # noqa: E712
    return (
        len(cohort_df) == int(n_survived) + int(n_died)
        and survived_count == int(n_survived)
        and died_count == int(n_died)
    )


def _resolve_resume_run_dir(
    *,
    output_root_path: Path,
    selected_conditions: List[ConditionSpec],
    provider: str,
    model: Optional[str],
    current_window_hours: float,
    window_step_hours: float,
    include_pre_icu_data: bool,
    n_survived: int,
    n_died: int,
    selection_seed: int,
    use_discharge_summary_for_history: bool,
    num_discharge_summaries: int,
    relative_report_codes: List[str],
    pre_icu_history_hours: float,
    context_history_hours: float,
    context_future_hours: float,
    require_discharge_summary_for_icu_stays: bool,
    resume: bool,
    resume_run: Optional[str],
) -> Optional[Path]:
    if not resume and not resume_run:
        return None

    explicit = bool(resume_run and str(resume_run).strip())
    candidate_dirs: List[Path] = []
    if explicit:
        raw = str(resume_run).strip()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = output_root_path / raw
        candidate_dirs.append(candidate)
    else:
        candidate_dirs = sorted(
            [path for path in output_root_path.glob(f"{RUN_PREFIX}*") if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists() or not candidate_dir.is_dir():
            if explicit:
                raise ValueError(f"Resume run does not exist: {candidate_dir}")
            continue

        cohort_manifest_path = candidate_dir / "cohort_manifest.csv"
        if not cohort_manifest_path.exists():
            if explicit:
                raise ValueError(f"Resume run missing cohort manifest: {cohort_manifest_path}")
            continue

        metadata = _json_load(candidate_dir / RUN_STATE_FILENAME) or _json_load(candidate_dir / "run_manifest.json")
        if metadata is not None:
            compatible, checks_applied = _matches_run_metadata(
                metadata,
                selected_conditions=selected_conditions,
                provider=provider,
                model=model,
                current_window_hours=current_window_hours,
                window_step_hours=window_step_hours,
                include_pre_icu_data=include_pre_icu_data,
                n_survived=n_survived,
                n_died=n_died,
                selection_seed=selection_seed,
                use_discharge_summary_for_history=use_discharge_summary_for_history,
                num_discharge_summaries=num_discharge_summaries,
                relative_report_codes=relative_report_codes,
                pre_icu_history_hours=pre_icu_history_hours,
                context_history_hours=context_history_hours,
                context_future_hours=context_future_hours,
                require_discharge_summary_for_icu_stays=require_discharge_summary_for_icu_stays,
            )
            if compatible and checks_applied > 0:
                return candidate_dir
            if explicit:
                raise ValueError(f"Resume run is incompatible with current settings: {candidate_dir}")
            continue

        if _cohort_matches_target_counts(cohort_manifest_path, n_survived=n_survived, n_died=n_died):
            return candidate_dir

        if explicit:
            raise ValueError(
                "Resume run has no comparable metadata and cohort counts do not match requested settings: "
                f"{candidate_dir}"
            )

    return None


def _build_run_state_payload(
    *,
    run_id: str,
    events_path: str,
    icu_stay_path: str,
    provider: str,
    model: Optional[str],
    n_survived: int,
    n_died: int,
    selection_seed: int,
    current_window_hours: float,
    window_step_hours: float,
    include_pre_icu_data: bool,
    window_workers: int,
    selected_conditions: List[ConditionSpec],
    resumed_from_previous: bool,
    use_discharge_summary_for_history: bool,
    num_discharge_summaries: int,
    relative_report_codes: List[str],
    pre_icu_history_hours: float,
    context_history_hours: float,
    context_future_hours: float,
    require_discharge_summary_for_icu_stays: bool,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "updated_at": datetime.now().isoformat(),
        "events_path": events_path,
        "icu_stay_path": icu_stay_path,
        "provider": provider,
        "model": model,
        "n_survived": n_survived,
        "n_died": n_died,
        "selection_seed": selection_seed,
        "current_window_hours": current_window_hours,
        "window_step_hours": window_step_hours,
        "include_pre_icu_data": include_pre_icu_data,
        "window_workers": window_workers,
        "use_discharge_summary_for_history": use_discharge_summary_for_history,
        "num_discharge_summaries": num_discharge_summaries,
        "relative_report_codes": list(relative_report_codes),
        "pre_icu_history_hours": pre_icu_history_hours,
        "context_history_hours": context_history_hours,
        "context_future_hours": context_future_hours,
        "require_discharge_summary_for_icu_stays": require_discharge_summary_for_icu_stays,
        "conditions": [condition.name for condition in selected_conditions],
        "resumed_from_previous": resumed_from_previous,
    }


def _run_single_condition(
    *,
    config: Config,
    parser: Any,
    cohort_df: pd.DataFrame,
    condition: ConditionSpec,
    condition_dir: Path,
    provider: str,
    model: Optional[str],
    current_window_hours: float,
    window_step_hours: float,
    include_pre_icu_data: bool,
    window_workers: int,
    resume_existing_patients: bool,
) -> Dict[str, Any]:
    condition_dir.mkdir(parents=True, exist_ok=True)
    patients_dir = condition_dir / "patients"
    patients_dir.mkdir(parents=True, exist_ok=True)

    oracle = MetaOracle(
        provider=provider,
        model=model,
        max_tokens=config.llm_max_tokens,
        use_discharge_summary=True,  # Forced ON so discharge-summary masking differences are meaningful.
        include_icu_outcome_in_prompt=condition.include_icu_outcome_in_prompt,
        mask_discharge_summary_outcome_terms=condition.mask_discharge_summary_outcome_terms,
        history_context_hours=config.oracle_context_history_hours,
        future_context_hours=config.oracle_context_future_hours,
        top_k_recommendations=config.oracle_context_top_k_recommendations,
        log_dir=str(condition_dir / "logs" / "oracle"),
    )

    summary_stats: Dict[str, Any] = {
        "condition": condition.name,
        "include_icu_outcome_in_prompt": condition.include_icu_outcome_in_prompt,
        "mask_discharge_summary_outcome_terms": condition.mask_discharge_summary_outcome_terms,
        "reverse_prompt_outcome": condition.reverse_prompt_outcome,
        "total_patients": int(len(cohort_df)),
        "patients_processed": 0,
        "patients_resumed": 0,
        "patients_failed": 0,
        "total_windows_evaluated": 0,
        "overall_status_distribution": {},
    }

    for i, (_, row) in enumerate(cohort_df.iterrows(), start=1):
        subject_id = int(row["subject_id"])
        icu_stay_id = int(row["icu_stay_id"])
        true_survived = bool(row["survived"])
        patient_dir = patients_dir / f"{subject_id}_{icu_stay_id}"
        print(
            f"[{condition.name}] Patient {i}/{len(cohort_df)}: "
            f"{subject_id}_{icu_stay_id} (true outcome={'survived' if true_survived else 'died'})"
        )

        if resume_existing_patients:
            existing_stats = _extract_existing_patient_stats(patient_dir)
            if existing_stats is not None:
                print(f"  [{condition.name}] Resume: skipping existing outputs for {subject_id}_{icu_stay_id}.")
                summary_stats["patients_processed"] += 1
                summary_stats["patients_resumed"] += 1
                summary_stats["total_windows_evaluated"] += existing_stats["num_windows_evaluated"]
                for label, count in existing_stats["overall_status_distribution"].items():
                    summary_stats["overall_status_distribution"][label] = summary_stats[
                        "overall_status_distribution"
                    ].get(label, 0) + int(count)
                continue

        try:
            has_event_idx_bounds = "min_event_idx" in row.index and "max_event_idx" in row.index
            if has_event_idx_bounds:
                true_trajectory = parser.get_patient_trajectory(subject_id, icu_stay_id, icu_stay=row)
            else:
                true_trajectory = parser.get_patient_trajectory(subject_id, icu_stay_id)
            prompt_trajectory = apply_prompt_outcome_mode(
                true_trajectory,
                reverse_prompt_outcome=condition.reverse_prompt_outcome,
            )
            prompt_survived = bool(prompt_trajectory.get("survived"))

            windows = parser.create_time_windows(
                prompt_trajectory,
                current_window_hours=current_window_hours,
                window_step_hours=window_step_hours,
                include_pre_icu_data=include_pre_icu_data,
                use_first_n_hours_after_icu=None,  # full trajectory
                use_discharge_summary_for_history=config.oracle_use_discharge_summary_for_history,
                num_discharge_summaries=config.oracle_num_discharge_summaries,
                relative_report_codes=config.oracle_relative_report_codes,
                pre_icu_history_hours=config.oracle_pre_icu_history_hours,
                history_context_hours=config.oracle_context_history_hours,
                future_context_hours=config.oracle_context_future_hours,
            )
            if condition.mask_discharge_summary_outcome_terms:
                windows = [mask_window_outcome_leakage(window) for window in windows]

            if not windows:
                print(f"  [{condition.name}] Skipping {subject_id}_{icu_stay_id}: no windows generated.")
                continue

            if window_workers > 1 and len(windows) > 1:
                reports = oracle.evaluate_trajectory_parallel(
                    windows,
                    max_workers=min(window_workers, len(windows)),
                    show_progress=True,
                )
            else:
                reports = oracle.evaluate_trajectory(windows)

            patient_dir.mkdir(parents=True, exist_ok=True)

            llm_calls = oracle.pop_patient_llm_call_logs(subject_id=subject_id, icu_stay_id=icu_stay_id)
            oracle.pop_patient_trajectory_logs(subject_id=subject_id, icu_stay_id=icu_stay_id)

            patient_predictions = _build_patient_predictions_payload(
                run_id=condition.name,
                trajectory=true_trajectory,
                windows=windows,
                reports=reports,
                llm_calls=llm_calls,
            )
            patient_predictions["condition"] = condition.name
            patient_predictions["trajectory_metadata"]["true_survived"] = true_survived
            patient_predictions["trajectory_metadata"]["prompt_survived"] = prompt_survived

            for window_output in patient_predictions.get("window_outputs", []):
                meta = window_output.get("window_metadata", {})
                if isinstance(meta, dict):
                    meta["true_survived"] = true_survived
                    meta["prompt_survived"] = prompt_survived

            _json_dump(
                patient_predictions,
                patient_dir / "oracle_predictions.json",
                json_default=getattr(parser, "_json_default", _json_default),
            )

            llm_payload = _build_oracle_llm_calls_payload(
                subject_id=subject_id,
                icu_stay_id=icu_stay_id,
                provider=getattr(oracle.llm_client, "provider", None),
                model=getattr(oracle.llm_client, "model", None),
                include_icu_outcome_in_prompt=condition.include_icu_outcome_in_prompt,
                calls=llm_calls,
            )
            llm_payload["condition"] = condition.name
            llm_payload["true_survived"] = true_survived
            llm_payload["prompt_survived"] = prompt_survived
            _json_dump(
                llm_payload,
                patient_dir / "llm_calls.json",
                json_default=getattr(parser, "_json_default", _json_default),
            )
            save_llm_calls_html(llm_payload, patient_dir / "llm_calls.html")

            window_contexts = _build_window_contexts_payload(
                run_id=condition.name,
                trajectory=true_trajectory,
                windows=windows,
                llm_calls=llm_calls,
                history_hours=float(config.oracle_context_history_hours),
                future_hours=float(config.oracle_context_future_hours),
            )
            window_contexts["condition"] = condition.name
            window_contexts["true_survived"] = true_survived
            window_contexts["prompt_survived"] = prompt_survived
            _json_dump(
                window_contexts,
                patient_dir / "window_contexts.json",
                json_default=getattr(parser, "_json_default", _json_default),
            )

            for report in reports:
                label = extract_overall_label(report.to_dict())
                summary_stats["overall_status_distribution"][label] = (
                    summary_stats["overall_status_distribution"].get(label, 0) + 1
                )

            summary_stats["patients_processed"] += 1
            summary_stats["total_windows_evaluated"] += len(reports)

        except Exception as exc:
            print(f"  [{condition.name}] ERROR for {subject_id}_{icu_stay_id}: {exc}")
            summary_stats["patients_failed"] += 1
            oracle.pop_patient_trajectory_logs(subject_id=subject_id, icu_stay_id=icu_stay_id)
            oracle.pop_patient_llm_call_logs(subject_id=subject_id, icu_stay_id=icu_stay_id)

    summary_stats.update(oracle.get_statistics())
    _json_dump(
        summary_stats,
        condition_dir / "processing_summary.json",
        json_default=getattr(parser, "_json_default", _json_default),
    )
    return summary_stats


def run_oracle_condition_suite(
    *,
    config: Config,
    events_path: str,
    icu_stay_path: str,
    output_root: str,
    provider: str,
    model: Optional[str],
    current_window_hours: float,
    window_step_hours: float,
    include_pre_icu_data: bool,
    n_survived: int,
    n_died: int,
    selection_seed: int,
    window_workers: int,
    conditions: Optional[Iterable[str]] = None,
    resume: bool = True,
    resume_run: Optional[str] = None,
) -> Path:
    require_discharge_summary_for_icu_stays = True
    selected_conditions = _resolve_conditions(conditions)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    resumed_run_dir = _resolve_resume_run_dir(
        output_root_path=output_root_path,
        selected_conditions=selected_conditions,
        provider=provider,
        model=model,
        current_window_hours=current_window_hours,
        window_step_hours=window_step_hours,
        include_pre_icu_data=include_pre_icu_data,
        n_survived=n_survived,
        n_died=n_died,
        selection_seed=selection_seed,
        use_discharge_summary_for_history=bool(config.oracle_use_discharge_summary_for_history),
        num_discharge_summaries=int(config.oracle_num_discharge_summaries),
        relative_report_codes=list(config.oracle_relative_report_codes),
        pre_icu_history_hours=float(config.oracle_pre_icu_history_hours),
        context_history_hours=float(config.oracle_context_history_hours),
        context_future_hours=float(config.oracle_context_future_hours),
        require_discharge_summary_for_icu_stays=require_discharge_summary_for_icu_stays,
        resume=resume,
        resume_run=resume_run,
    )
    resumed_from_previous = resumed_run_dir is not None
    run_dir = resumed_run_dir or (output_root_path / datetime.now().strftime(f"{RUN_PREFIX}%Y%m%d_%H%M%S"))
    run_id = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "conditions").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    _json_dump(
        _build_run_state_payload(
            run_id=run_id,
            events_path=events_path,
            icu_stay_path=icu_stay_path,
            provider=provider,
            model=model,
            n_survived=n_survived,
            n_died=n_died,
            selection_seed=selection_seed,
            current_window_hours=current_window_hours,
            window_step_hours=window_step_hours,
            include_pre_icu_data=include_pre_icu_data,
            window_workers=window_workers,
            selected_conditions=selected_conditions,
            resumed_from_previous=resumed_from_previous,
            use_discharge_summary_for_history=bool(config.oracle_use_discharge_summary_for_history),
            num_discharge_summaries=int(config.oracle_num_discharge_summaries),
            relative_report_codes=list(config.oracle_relative_report_codes),
            pre_icu_history_hours=float(config.oracle_pre_icu_history_hours),
            context_history_hours=float(config.oracle_context_history_hours),
            context_future_hours=float(config.oracle_context_future_hours),
            require_discharge_summary_for_icu_stays=require_discharge_summary_for_icu_stays,
        ),
        run_dir / RUN_STATE_FILENAME,
        json_default=_json_default,
    )

    print("=" * 80)
    print("ORACLE CONDITION SUITE")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    if resumed_from_previous:
        print("Resume mode: reusing existing run outputs where possible.")
    print(f"Conditions: {[c.name for c in selected_conditions]}")
    print(f"Cohort target: survived={n_survived}, died={n_died}, seed={selection_seed}")
    print(f"Window config: current={current_window_hours}h, step={window_step_hours}h")

    try:
        parser = MIMICDataParser(
            events_path,
            icu_stay_path,
            require_discharge_summary_for_icu_stays=require_discharge_summary_for_icu_stays,
        )
    except TypeError:
        # Backward-compatible path for parser test doubles without new kwargs.
        parser = MIMICDataParser(events_path, icu_stay_path)
    parser.load_data()

    eligible_df = _filter_eligible_stays_with_discharge_summary(parser)
    survived_count = int((eligible_df["survived"] == True).sum())  # noqa: E712
    died_count = int((eligible_df["survived"] == False).sum())  # noqa: E712

    if survived_count < n_survived or died_count < n_died:
        raise ValueError(
            "Insufficient eligible stays with selected discharge summaries: "
            f"need survived={n_survived}, died={n_died}; "
            f"available survived={survived_count}, died={died_count}."
        )

    cohort_manifest_path = run_dir / "cohort_manifest.csv"
    if resumed_from_previous and cohort_manifest_path.exists():
        cohort_df = _load_cohort_manifest(cohort_manifest_path)
        existing_survived = int((cohort_df["survived"] == True).sum())  # noqa: E712
        existing_died = int((cohort_df["survived"] == False).sum())  # noqa: E712
        if existing_survived != int(n_survived) or existing_died != int(n_died):
            raise ValueError(
                "Existing cohort manifest does not match requested cohort size: "
                f"expected survived={n_survived}, died={n_died}; "
                f"found survived={existing_survived}, died={existing_died}."
            )
        cohort_pairs = {(int(row["subject_id"]), int(row["icu_stay_id"])) for _, row in cohort_df.iterrows()}
        eligible_pairs = {(int(row["subject_id"]), int(row["icu_stay_id"])) for _, row in eligible_df.iterrows()}
        missing_pairs = cohort_pairs - eligible_pairs
        if missing_pairs:
            raise ValueError(
                "Existing cohort manifest contains ICU stays no longer eligible under the current data: "
                f"{sorted(missing_pairs)}"
            )
        print(f"Loaded cohort from existing manifest: {cohort_manifest_path}")
    else:
        cohort_df = select_balanced_patients(
            eligible_df,
            n_survived=n_survived,
            n_died=n_died,
            random_seed=selection_seed,
        ).reset_index(drop=True)
        _save_cohort_manifest(cohort_df, run_dir)

    all_condition_summaries: Dict[str, Any] = {}
    for condition in selected_conditions:
        condition_dir = run_dir / "conditions" / condition.name
        print("\n" + "-" * 80)
        print(f"Running condition: {condition.name}")
        print("-" * 80)
        condition_summary = _run_single_condition(
            config=config,
            parser=parser,
            cohort_df=cohort_df,
            condition=condition,
            condition_dir=condition_dir,
            provider=provider,
            model=model,
            current_window_hours=current_window_hours,
            window_step_hours=window_step_hours,
            include_pre_icu_data=include_pre_icu_data,
            window_workers=window_workers,
            resume_existing_patients=resumed_from_previous,
        )
        all_condition_summaries[condition.name] = condition_summary

    run_manifest = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "events_path": events_path,
        "icu_stay_path": icu_stay_path,
        "provider": provider,
        "model": model,
        "n_survived": n_survived,
        "n_died": n_died,
        "selection_seed": selection_seed,
        "current_window_hours": current_window_hours,
        "window_step_hours": window_step_hours,
        "include_pre_icu_data": include_pre_icu_data,
        "window_workers": window_workers,
        "use_discharge_summary_for_history": bool(config.oracle_use_discharge_summary_for_history),
        "num_discharge_summaries": int(config.oracle_num_discharge_summaries),
        "relative_report_codes": list(config.oracle_relative_report_codes),
        "pre_icu_history_hours": float(config.oracle_pre_icu_history_hours),
        "context_history_hours": float(config.oracle_context_history_hours),
        "context_future_hours": float(config.oracle_context_future_hours),
        "require_discharge_summary_for_icu_stays": require_discharge_summary_for_icu_stays,
        "conditions": [condition.name for condition in selected_conditions],
        "condition_summaries": all_condition_summaries,
        "cohort_manifest_csv": str(run_dir / "cohort_manifest.csv"),
        "analysis_dir": str(run_dir / "analysis"),
        "resumed_from_previous": resumed_from_previous,
    }
    _json_dump(
        run_manifest, run_dir / "run_manifest.json", json_default=getattr(parser, "_json_default", _json_default)
    )

    print("\n" + "=" * 80)
    print("CONDITION SUITE COMPLETE")
    print("=" * 80)
    print(f"Outputs: {run_dir}")
    print("Next: run analysis with experiments/oracle/analyze_oracle_experiments.py")
    return run_dir


def main() -> None:
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Run Oracle condition suite (full_visible / masked_outcome / reversed_outcome)."
    )
    parser.add_argument("--events", type=str, default=config.events_path, help="Path to events parquet file.")
    parser.add_argument("--icu-stay", type=str, default=config.icu_stay_path, help="Path to ICU stay parquet file.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="oracle_results/oracle-validation",
        help="Root output directory for condition-suite runs.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=config.llm_provider,
        choices=["anthropic", "openai", "google", "gemini"],
        help="LLM provider.",
    )
    parser.add_argument("--model", type=str, default=config.llm_model, help="Model name.")
    parser.add_argument("--current-window-hours", type=float, default=1.0, help="Current window size in hours.")
    parser.add_argument("--window-step-hours", type=float, default=4.0, help="Sliding step in hours.")
    parser.add_argument("--window-workers", type=int, default=4, help="Parallel workers per patient.")
    parser.add_argument("--n-survived", type=int, default=5, help="Number of survived ICU stays in cohort.")
    parser.add_argument("--n-died", type=int, default=5, help="Number of died ICU stays in cohort.")
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=DEFAULT_SELECTION_SEED,
        help=f"Random seed for cohort selection (default: {DEFAULT_SELECTION_SEED}).",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Optional subset of conditions to run (default: all).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically resume the latest compatible run in output root (default: enabled).",
    )
    parser.add_argument(
        "--resume-run",
        type=str,
        default=None,
        help="Explicit run directory name/path to resume instead of auto-selecting the latest compatible run.",
    )
    parser.add_argument(
        "--no-pre-icu-data",
        action="store_true",
        help="Disable pre-ICU event inclusion while still requiring discharge-summary-eligible stays.",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional custom config path.")
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)

    run_oracle_condition_suite(
        config=config,
        events_path=args.events,
        icu_stay_path=args.icu_stay,
        output_root=args.output_root,
        provider=args.provider,
        model=args.model,
        current_window_hours=float(args.current_window_hours),
        window_step_hours=float(args.window_step_hours),
        include_pre_icu_data=not args.no_pre_icu_data,
        n_survived=int(args.n_survived),
        n_died=int(args.n_died),
        selection_seed=int(args.selection_seed),
        window_workers=int(args.window_workers),
        conditions=args.conditions,
        resume=bool(args.resume),
        resume_run=args.resume_run,
    )


if __name__ == "__main__":
    main()
