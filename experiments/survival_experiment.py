"""Survival prediction from precomputed MedEvo memory runs."""

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.med_evo_agent import MedEvoAgent
from config.config import get_config
from experiments.create_memory import (
    collect_windowed_snapshots,
    extract_snapshot_hour_bounds,
    extract_snapshot_window_features,
    filter_memory_patient_records_by_stay_ids,
    infer_snapshot_window_index,
    load_memory_patient_records,
    load_memory_run_config,
    load_patient_memory_payload,
    render_snapshot_to_text,
    resolve_memory_run_dir,
    select_snapshot_by_observation_hour,
)
from prompts.predictor_prompts import get_survival_prediction_prompt
from utils.json_parse import parse_json_dict_best_effort
from utils.llm_log_viewer import save_llm_calls_html
from utils.outcome_utils import evaluate_outcome_match, extract_survival_prediction_fields

RUN_CONFIG_FILENAME = "run_config.json"
AGGREGATE_FILENAME = "aggregate_results.json"


def _normalize_token_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return 0
    return 0


def _save_run_config(results_dir: Path, payload: Dict[str, Any]) -> None:
    with open(results_dir / RUN_CONFIG_FILENAME, "w") as f:
        json.dump(payload, f, indent=2)


def process_single_patient(
    patient_record: Dict[str, Any],
    agent: MedEvoAgent,
    snapshot_hour: Optional[float],
    memory_run_dir: Path,
    results_dir: Path,
    patient_idx: int,
    total_patients: int,
    enable_logging: bool,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    subject_id = int(patient_record["subject_id"])
    icu_stay_id = int(patient_record["icu_stay_id"])
    actual_outcome = str(patient_record.get("actual_outcome") or "unknown")
    source_patient_dir = Path(str(patient_record["patient_dir"]))

    if verbose:
        print(f"\n[Patient {patient_idx}/{total_patients}] Subject: {subject_id}, ICU Stay: {icu_stay_id}")
        print(f"   Source memory: {source_patient_dir}")
        print(f"   Actual Outcome: {actual_outcome.upper()}")

    try:
        payload = load_patient_memory_payload(source_patient_dir)
        memory_snapshots = payload.get("memory_snapshots", [])
        if not isinstance(memory_snapshots, list):
            memory_snapshots = []
        final_memory = payload.get("final_memory", {})
        if not isinstance(final_memory, dict):
            final_memory = {}

        windowed_snapshots = collect_windowed_snapshots(memory_snapshots)
        if not windowed_snapshots and final_memory:
            final_window_index = infer_snapshot_window_index(final_memory)
            normalized_window_index = final_window_index if final_window_index is not None else -1
            windowed_snapshots = [(int(normalized_window_index), final_memory)]

        if not windowed_snapshots:
            print("   WARNING: No snapshots found in source memory, skipping...")
            return None

        final_window_index, _ = windowed_snapshots[-1]
        if snapshot_hour is None:
            selected_window_index, snapshot_for_prediction = windowed_snapshots[-1]
            snapshot_selection_mode = "last_only"
            snapshot_selection_hour = None
        else:
            selected_window_index, snapshot_for_prediction = select_snapshot_by_observation_hour(
                windowed_snapshots=windowed_snapshots,
                observation_hour=float(snapshot_hour),
            )
            snapshot_selection_mode = "observation_hour"
            snapshot_selection_hour = float(snapshot_hour)

        inferred_window_index, selected_hours, num_current_events = extract_snapshot_window_features(snapshot_for_prediction)
        if inferred_window_index >= 0:
            selected_window_index = inferred_window_index
        selected_start_hour, selected_end_hour = extract_snapshot_hour_bounds(snapshot_for_prediction)

        context = render_snapshot_to_text(snapshot_for_prediction)
        prompt = get_survival_prediction_prompt().format(context=context)
        response = agent.llm_client.chat(prompt=prompt, response_format="text")

        raw_response = response.get("content", "")
        usage = response.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}

        input_tokens = _normalize_token_count(usage.get("input_tokens"))
        output_tokens = _normalize_token_count(usage.get("output_tokens"))

        parsed_prediction = parse_json_dict_best_effort(raw_response)
        if parsed_prediction is None:
            parsed_prediction = {}

        predicted_outcome, confidence = extract_survival_prediction_fields(parsed_prediction)
        is_correct, normalized_predicted_outcome, normalized_actual_outcome = evaluate_outcome_match(
            predicted=predicted_outcome,
            actual=actual_outcome,
        )

        if verbose:
            if snapshot_selection_hour is None:
                print(f"   Prediction Memory Window: {selected_window_index} ({num_current_events} events)")
            else:
                print(
                    f"   Prediction Memory Window: {selected_window_index} "
                    f"(hour request={snapshot_selection_hour:g}, window={selected_start_hour:g}-{selected_end_hour:g}, "
                    f"{num_current_events} events)"
                )
            print(f"   Predicted: {predicted_outcome.upper()} (confidence: {confidence})")
            print(f"   Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

        patient_dir = results_dir / "patients" / f"{subject_id}_{icu_stay_id}"
        patient_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "actual_outcome": actual_outcome,
            "predicted_outcome": predicted_outcome,
            "actual_outcome_normalized": normalized_actual_outcome,
            "predicted_outcome_normalized": normalized_predicted_outcome,
            "is_correct": is_correct,
            "confidence": confidence,
            "memory_run": str(memory_run_dir),
            "source_patient_dir": str(source_patient_dir),
            "num_memory_snapshots": len(windowed_snapshots),
            "final_memory_window_index": int(final_window_index),
            "prediction_memory_window_index": int(selected_window_index),
            "prediction_memory_window_start_hour": float(selected_start_hour),
            "prediction_memory_window_end_hour": float(selected_end_hour),
            "prediction_snapshot_selection_mode": snapshot_selection_mode,
            "prediction_snapshot_observation_hour": snapshot_selection_hour,
            "prediction": parsed_prediction,
            "prediction_tokens": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

        with open(patient_dir / "prediction.json", "w") as f:
            json.dump(result, f, indent=2)

        if enable_logging:
            llm_call = {
                "timestamp": datetime.now().isoformat(),
                "patient_id": f"{subject_id}_{icu_stay_id}",
                "window_index": int(selected_window_index),
                "hours_since_admission": float(selected_hours),
                "prompt": prompt,
                "response": raw_response,
                "parsed_response": parsed_prediction,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "metadata": {
                    "step_type": "survival_predictor",
                    "llm_provider": agent.llm_client.provider,
                    "llm_model": agent.llm_client.model,
                    "memory_run": str(memory_run_dir),
                    "snapshot_source": "precomputed_med_evo_memory",
                    "snapshot_selection": snapshot_selection_mode,
                    "snapshot_observation_hour": snapshot_selection_hour,
                    "snapshot_window_start_hour": float(selected_start_hour),
                    "snapshot_window_end_hour": float(selected_end_hour),
                },
            }
            patient_logs = {
                "patient_id": f"{subject_id}_{icu_stay_id}",
                "agent_type": "med_evo_survival_from_memory",
                "llm_provider": getattr(agent.llm_client, "provider", None),
                "llm_model": getattr(agent.llm_client, "model", None),
                "pipeline_agents": [{"name": "survival_predictor", "used": True}],
                "total_calls": 1,
                "calls": [llm_call],
            }
            with open(patient_dir / "llm_calls.json", "w") as f:
                json.dump(patient_logs, f, indent=2)
            save_llm_calls_html(patient_logs, patient_dir / "llm_calls.html")
            if verbose:
                print("   Saved log viewer: llm_calls.html")

        return result

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_experiment(
    memory_run: str,
    snapshot_hour: Optional[float] = None,
    verbose: bool = True,
    enable_logging: bool = True,
    patient_stay_ids_path: Optional[str] = None,
    num_workers: int = 1,
) -> Dict[str, Any]:
    normalized_snapshot_hour: Optional[float] = None
    if snapshot_hour is not None:
        try:
            normalized_snapshot_hour = float(snapshot_hour)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid snapshot_hour value: {snapshot_hour}") from exc
        if not math.isfinite(normalized_snapshot_hour) or normalized_snapshot_hour < 0:
            raise ValueError(f"snapshot_hour must be a finite number >= 0, got {snapshot_hour}")

    try:
        num_workers = int(num_workers)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid num_workers value: {num_workers}") from exc
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")

    config = get_config()
    memory_run_dir = resolve_memory_run_dir(memory_run)
    source_run_config = load_memory_run_config(memory_run_dir)

    print("=" * 80)
    print("SURVIVAL PREDICTION EXPERIMENT - FROM MED_EVO MEMORY")
    print("=" * 80)
    print(f"Memory Run: {memory_run_dir}")
    if normalized_snapshot_hour is None:
        print("Snapshot Selection: last_only")
    else:
        print(f"Snapshot Selection: observation_hour={normalized_snapshot_hour:g}")
    print(f"Num Workers: {num_workers}")

    results_dir = memory_run_dir / "survival_experiment"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results: {results_dir}")

    all_records = load_memory_patient_records(memory_run_dir)
    selected_records = filter_memory_patient_records_by_stay_ids(
        records=all_records,
        patient_stay_ids_path=patient_stay_ids_path,
    )

    print(f"Patients in memory run: {len(all_records)}")
    if patient_stay_ids_path:
        print(f"Patients after --patient-stay-ids filter: {len(selected_records)}")

    run_config_payload = {
        "generated_at": datetime.now().isoformat(),
        "experiment": "survival_experiment",
        "memory_input": {
            "memory_run": str(memory_run_dir),
            "source_generated_at": source_run_config.get("generated_at"),
            "source_experiment": source_run_config.get("experiment"),
            "selected_patients": len(selected_records),
        },
        "logging": {
            "enable_logging": bool(enable_logging),
        },
        "execution": {
            "num_workers": num_workers,
        },
        "survival_prediction": {
            "snapshot_selection": "last_only" if normalized_snapshot_hour is None else "observation_hour",
            "snapshot_observation_hour": normalized_snapshot_hour,
        },
        "llm": {
            "provider": config.llm_provider,
            "model": config.llm_model,
            "max_tokens": config.llm_max_tokens,
        },
    }
    _save_run_config(results_dir, run_config_payload)

    if not selected_records:
        print("No patients available for prediction.")
        return {}

    all_results: List[Dict[str, Any]] = []
    patient_data = [(idx, record) for idx, record in enumerate(selected_records, 1)]

    def process_patient_wrapper(args):
        idx, patient_record = args
        patient_agent = MedEvoAgent(
            provider=config.llm_provider,
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
            enable_logging=False,
            window_duration_hours=config.agent_current_window_hours,
            max_working_windows=config.med_evo_max_working_windows,
            max_critical_events=config.med_evo_max_critical_events,
            max_window_summaries=config.med_evo_max_window_summaries,
            max_insights=config.med_evo_max_insights,
            insight_recency_tau=config.med_evo_insight_recency_tau,
            insight_every_n_windows=config.med_evo_insight_every_n_windows,
            episode_every_n_windows=config.med_evo_episode_every_n_windows,
        )
        return process_single_patient(
            patient_record=patient_record,
            agent=patient_agent,
            snapshot_hour=normalized_snapshot_hour,
            memory_run_dir=memory_run_dir,
            results_dir=results_dir,
            patient_idx=idx,
            total_patients=len(selected_records),
            enable_logging=enable_logging,
            verbose=verbose,
        )

    max_workers = min(num_workers, len(patient_data))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_patient_wrapper, item) for item in patient_data]
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_results.append(result)

    if not all_results:
        print("No patient results generated.")
        return {}

    correct = sum(1 for item in all_results if bool(item.get("is_correct")))
    total = len(all_results)
    accuracy = correct / total if total > 0 else 0.0

    confidence_distribution = {"Low": 0, "Moderate": 0, "High": 0, "Unknown": 0}
    for item in all_results:
        label = str(item.get("confidence") or "Unknown")
        if label in confidence_distribution:
            confidence_distribution[label] += 1
        else:
            confidence_distribution["Unknown"] += 1

    token_totals = {
        "input_tokens": sum(int(item.get("prediction_tokens", {}).get("input_tokens", 0)) for item in all_results),
        "output_tokens": sum(int(item.get("prediction_tokens", {}).get("output_tokens", 0)) for item in all_results),
    }
    token_totals["total_tokens"] = token_totals["input_tokens"] + token_totals["output_tokens"]

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print(f"Total Patients: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nConfidence Distribution:")
    print(f"  Low: {confidence_distribution['Low']}")
    print(f"  Moderate: {confidence_distribution['Moderate']}")
    print(f"  High: {confidence_distribution['High']}")
    print(f"  Unknown: {confidence_distribution['Unknown']}")
    print("\nPredictor Tokens:")
    print(f"  Input: {token_totals['input_tokens']}")
    print(f"  Output: {token_totals['output_tokens']}")
    print(f"  Total: {token_totals['total_tokens']}")

    aggregate = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment": "survival_experiment",
        "memory_run": str(memory_run_dir),
        "num_workers": num_workers,
        "total_patients": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "confidence_distribution": confidence_distribution,
        "agent_stats": {
            "total_predictor_calls": total,
            "total_tokens_used": token_totals["total_tokens"],
        },
        "individual_results": sorted(all_results, key=lambda item: (item["subject_id"], item["icu_stay_id"])),
    }

    with open(results_dir / AGGREGATE_FILENAME, "w") as f:
        json.dump(aggregate, f, indent=2)

    return aggregate


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Survival prediction from precomputed MedEvo memory")
    parser.add_argument(
        "--memory-run",
        type=str,
        required=True,
        help="Path to memory run directory produced by experiments/create_memory.py.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--no-logging", action="store_true", help="Disable detailed LLM call logging")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers for patient processing (default: 1).",
    )
    parser.add_argument(
        "--patient-stay-ids",
        type=str,
        default=None,
        help="Optional CSV with columns subject_id,icu_stay_id to filter patients from memory run.",
    )
    parser.add_argument(
        "--snapshot-hour",
        type=float,
        default=None,
        help="Optional ICU hour for snapshot selection. If omitted, use the last memory snapshot.",
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("RUNNING: SURVIVAL FROM MEMORY")
    print(f"{'='*80}\n")

    run_experiment(
        memory_run=args.memory_run,
        snapshot_hour=args.snapshot_hour,
        verbose=not args.quiet,
        enable_logging=not args.no_logging,
        patient_stay_ids_path=args.patient_stay_ids,
        num_workers=args.num_workers,
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
