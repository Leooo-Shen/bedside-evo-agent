"""Patient status prediction from precomputed MedEvo memory runs and event baselines."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.med_evo_agent import MedEvoAgent
from config.config import get_config
from experiments.create_memory import (
    collect_windowed_snapshots,
    extract_snapshot_window_features,
    filter_memory_patient_records_by_stay_ids,
    infer_snapshot_window_index,
    load_memory_patient_records,
    load_memory_run_config,
    load_patient_memory_payload,
    render_snapshot_to_text,
    resolve_memory_run_dir,
    select_snapshots_with_stride,
)
from prompts.predictor_prompts import get_patient_status_prediction_prompt
from utils.event_format import format_event_lines
from utils.json_parse import parse_json_dict_best_effort
from utils.llm_log_viewer import save_llm_calls_html
from utils.status_scoring import normalize_status_label

RUN_CONFIG_FILENAME = "run_config.json"
AGGREGATE_FILENAME = "aggregate_results.json"
CONTEXT_MODE_MED_EVO_MEMORY = "med_evo_memory"
CONTEXT_MODE_FULL_HISTORY_EVENTS = "full_history_events"
CONTEXT_MODE_LOCAL_EVENTS_ONLY = "local_events_only"
SUPPORTED_CONTEXT_MODES = (
    CONTEXT_MODE_MED_EVO_MEMORY,
    CONTEXT_MODE_FULL_HISTORY_EVENTS,
    CONTEXT_MODE_LOCAL_EVENTS_ONLY,
)


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


def _extract_status_fields(parsed_prediction: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
    patient_assessment = parsed_prediction.get("patient_assessment")
    if not isinstance(patient_assessment, dict):
        return "insufficient_data", "", []

    overall = patient_assessment.get("overall")
    if isinstance(overall, dict):
        label = normalize_status_label(overall.get("label"))
        rationale = str(overall.get("rationale") or "").strip()
    else:
        label = "insufficient_data"
        rationale = ""

    active_risks = patient_assessment.get("active_risks")
    if not isinstance(active_risks, list):
        active_risks = []

    if not label:
        label = "insufficient_data"

    return label, rationale, active_risks


def _sum_numeric_stats(items: List[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def _extract_current_window(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    working_memory = snapshot.get("working_memory")
    if not isinstance(working_memory, list) or not working_memory:
        raise ValueError("Snapshot missing non-empty working_memory")
    current_window = working_memory[-1]
    if not isinstance(current_window, Mapping):
        raise ValueError("Snapshot current working_memory entry must be a mapping")
    return current_window


def _window_events(window: Mapping[str, Any]) -> List[Dict[str, Any]]:
    events = window.get("events")
    if not isinstance(events, list):
        return []
    return [dict(event) for event in events if isinstance(event, Mapping)]


def _render_flat_raw_events(events: Sequence[Mapping[str, Any]]) -> str:
    event_rows = [dict(event) for event in events if isinstance(event, Mapping)]
    return "\n".join(format_event_lines(event_rows, empty_text="(No events)"))


def _build_full_history_event_context(
    *,
    selected_window_index: int,
    selected_snapshot: Mapping[str, Any],
    snapshot_by_window: Mapping[int, Mapping[str, Any]],
) -> str:
    candidate_windows = {int(index) for index in snapshot_by_window.keys() if int(index) <= int(selected_window_index)}
    candidate_windows.add(int(selected_window_index))
    merged_events: List[Dict[str, Any]] = []
    for window_index in sorted(candidate_windows):
        if int(window_index) == int(selected_window_index):
            snapshot = selected_snapshot
        else:
            snapshot = snapshot_by_window.get(int(window_index))
            if not isinstance(snapshot, Mapping):
                continue
        merged_events.extend(_window_events(_extract_current_window(snapshot)))
    return _render_flat_raw_events(merged_events)


def _build_local_events_only_context(selected_snapshot: Mapping[str, Any]) -> str:
    current_window = _extract_current_window(selected_snapshot)
    return _render_flat_raw_events(_window_events(current_window))


def _build_status_context(
    *,
    context_mode: str,
    selected_window_index: int,
    selected_snapshot: Mapping[str, Any],
    snapshot_by_window: Mapping[int, Mapping[str, Any]],
) -> Tuple[str, str]:
    if context_mode == CONTEXT_MODE_MED_EVO_MEMORY:
        return render_snapshot_to_text(dict(selected_snapshot)), "precomputed_med_evo_memory"
    if context_mode == CONTEXT_MODE_FULL_HISTORY_EVENTS:
        return (
            _build_full_history_event_context(
                selected_window_index=int(selected_window_index),
                selected_snapshot=selected_snapshot,
                snapshot_by_window=snapshot_by_window,
            ),
            CONTEXT_MODE_FULL_HISTORY_EVENTS,
        )
    if context_mode == CONTEXT_MODE_LOCAL_EVENTS_ONLY:
        return _build_local_events_only_context(selected_snapshot), CONTEXT_MODE_LOCAL_EVENTS_ONLY
    raise ValueError(
        f"Unsupported context_mode={context_mode}. "
        f"Supported modes: {', '.join(SUPPORTED_CONTEXT_MODES)}"
    )


def _results_dir_name_for_context_mode(context_mode: str) -> str:
    if context_mode == CONTEXT_MODE_MED_EVO_MEMORY:
        return "patient_status_experiment"
    return f"patient_status_experiment_{context_mode}"


def process_single_patient(
    patient_record: Dict[str, Any],
    agent: MedEvoAgent,
    snapshot_stride: int,
    context_mode: str,
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
        print(f"   Context mode: {context_mode}")

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
            windowed_snapshots = [(int(final_window_index or -1), final_memory)]

        if not windowed_snapshots:
            print("   WARNING: No snapshots found in source memory, skipping...")
            return None

        selected_snapshots = select_snapshots_with_stride(
            windowed_snapshots=windowed_snapshots,
            stride=snapshot_stride,
        )
        snapshot_by_window = {int(window_idx): snapshot for window_idx, snapshot in windowed_snapshots}
        if not selected_snapshots:
            print("   WARNING: No selected snapshots found, skipping...")
            return None

        if verbose:
            print(
                f"   Memory snapshots: total={len(windowed_snapshots)}, "
                f"selected={len(selected_snapshots)}, stride={snapshot_stride}"
            )
            print("   Status inference uses the last selected snapshot only.")

        status_call_logs: List[Dict[str, Any]] = []
        total_status_input_tokens = 0
        total_status_output_tokens = 0

        selected_window_index, snapshot_for_status = selected_snapshots[-1]
        sequence_idx = int(len(selected_snapshots))
        inferred_window_index, hours, num_current_events = extract_snapshot_window_features(snapshot_for_status)
        if inferred_window_index >= 0:
            window_index = inferred_window_index
        else:
            window_index = int(selected_window_index)

        context, snapshot_source = _build_status_context(
            context_mode=str(context_mode),
            selected_window_index=int(selected_window_index),
            selected_snapshot=snapshot_for_status,
            snapshot_by_window=snapshot_by_window,
        )
        prompt = get_patient_status_prediction_prompt().format(context=context)
        response = agent.llm_client.chat(prompt=prompt, response_format="text")

        raw_response = response.get("content", "")
        usage = response.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}

        input_tokens = _normalize_token_count(usage.get("input_tokens"))
        output_tokens = _normalize_token_count(usage.get("output_tokens"))
        total_status_input_tokens += input_tokens
        total_status_output_tokens += output_tokens

        parsed_prediction = parse_json_dict_best_effort(raw_response)
        if parsed_prediction is None:
            parsed_prediction = {}

        status_label, status_rationale, active_risks = _extract_status_fields(parsed_prediction)
        status_predictions = [
            {
                "window_index": int(window_index),
                "snapshot_sequence": int(sequence_idx),
                "hours_since_admission": float(hours),
                "num_current_events": int(num_current_events),
                "snapshot_source": str(snapshot_source),
                "context_mode": str(context_mode),
                "status_label": status_label,
                "status_rationale": status_rationale,
                "active_risks": active_risks,
                "parsed_prediction": parsed_prediction,
            }
        ]

        status_call_logs.append(
            {
                "timestamp": datetime.now().isoformat(),
                "patient_id": f"{subject_id}_{icu_stay_id}",
                "window_index": int(window_index),
                "hours_since_admission": float(hours),
                "prompt": prompt,
                "response": raw_response,
                "parsed_response": parsed_prediction,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "metadata": {
                    "step_type": "patient_status_predictor",
                    "llm_provider": agent.llm_client.provider,
                    "llm_model": agent.llm_client.model,
                    "snapshot_source": str(snapshot_source),
                    "context_mode": str(context_mode),
                    "snapshot_sequence": int(sequence_idx),
                    "memory_run": str(memory_run_dir),
                },
            }
        )

        if verbose:
            print(f"   Status Window {window_index}: {status_label}")

        status_distribution: Dict[str, int] = {}
        for item in status_predictions:
            label = item.get("status_label", "insufficient_data")
            status_distribution[label] = status_distribution.get(label, 0) + 1

        patient_dir = results_dir / "patients" / f"{subject_id}_{icu_stay_id}"
        patient_dir.mkdir(parents=True, exist_ok=True)

        patient_status_payload = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "actual_outcome": actual_outcome,
            "memory_run": str(memory_run_dir),
            "source_patient_dir": str(source_patient_dir),
            "context_mode": str(context_mode),
            "num_memory_snapshots": len(windowed_snapshots),
            "snapshot_stride": snapshot_stride,
            "status_distribution": status_distribution,
            "status_predictions": status_predictions,
            "status_prediction_tokens": {
                "input_tokens": total_status_input_tokens,
                "output_tokens": total_status_output_tokens,
                "total_tokens": total_status_input_tokens + total_status_output_tokens,
            },
        }

        with open(patient_dir / "patient_status_predictions.json", "w") as f:
            json.dump(patient_status_payload, f, indent=2)

        summary_payload = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "actual_outcome": actual_outcome,
            "memory_run": str(memory_run_dir),
            "context_mode": str(context_mode),
            "num_memory_snapshots": len(windowed_snapshots),
            "snapshot_stride": snapshot_stride,
            "final_status_label": (
                status_predictions[-1]["status_label"] if status_predictions else "insufficient_data"
            ),
            "status_distribution": status_distribution,
        }
        with open(patient_dir / "prediction.json", "w") as f:
            json.dump(summary_payload, f, indent=2)

        if enable_logging:
            patient_logs = {
                "patient_id": f"{subject_id}_{icu_stay_id}",
                "agent_type": f"patient_status_{context_mode}",
                "llm_provider": getattr(agent.llm_client, "provider", None),
                "llm_model": getattr(agent.llm_client, "model", None),
                "context_mode": str(context_mode),
                "pipeline_agents": [{"name": "patient_status_predictor", "used": True}],
                "total_calls": len(status_call_logs),
                "calls": status_call_logs,
            }
            with open(patient_dir / "llm_calls.json", "w") as f:
                json.dump(patient_logs, f, indent=2)
            save_llm_calls_html(patient_logs, patient_dir / "llm_calls.html")
            if verbose:
                print("   Saved log viewer: llm_calls.html")

        return {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "actual_outcome": actual_outcome,
            "context_mode": str(context_mode),
            "num_memory_snapshots": len(windowed_snapshots),
            "snapshot_stride": snapshot_stride,
            "num_status_predictions": len(status_predictions),
            "status_distribution": status_distribution,
            "final_status_label": (
                status_predictions[-1]["status_label"] if status_predictions else "insufficient_data"
            ),
            "status_prediction_tokens": {
                "input_tokens": total_status_input_tokens,
                "output_tokens": total_status_output_tokens,
                "total_tokens": total_status_input_tokens + total_status_output_tokens,
            },
        }

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


def _save_run_config(results_dir: Path, payload: Dict[str, Any]) -> None:
    with open(results_dir / RUN_CONFIG_FILENAME, "w") as f:
        json.dump(payload, f, indent=2)


def run_experiment(
    memory_run: str,
    snapshot_stride: int = 1,
    context_mode: str = "",
    verbose: bool = True,
    enable_logging: bool = True,
    patient_stay_ids_path: Optional[str] = None,
    num_workers: int = 4,
) -> Dict[str, Any]:
    try:
        snapshot_stride = max(1, int(snapshot_stride))
    except (TypeError, ValueError):
        snapshot_stride = 1
    try:
        num_workers = int(num_workers)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid num_workers value: {num_workers}") from exc
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    normalized_context_mode = str(context_mode).strip()
    if normalized_context_mode not in SUPPORTED_CONTEXT_MODES:
        raise ValueError(
            f"Unsupported context_mode={context_mode}. "
            f"Supported modes: {', '.join(SUPPORTED_CONTEXT_MODES)}"
        )

    config = get_config()
    memory_run_dir = resolve_memory_run_dir(memory_run)
    source_run_config = load_memory_run_config(memory_run_dir)

    print("=" * 80)
    print("PATIENT STATUS EXPERIMENT")
    print("=" * 80)
    print(f"Memory Run: {memory_run_dir}")
    print(f"Snapshot Stride (k): {snapshot_stride}")
    print(f"Context Mode: {normalized_context_mode}")
    print(f"Num Workers: {num_workers}")

    results_dir = memory_run_dir / _results_dir_name_for_context_mode(normalized_context_mode)
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
        "experiment": "patient_status_experiment",
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
        "status_prediction": {
            "snapshot_stride": snapshot_stride,
            "context_mode": str(normalized_context_mode),
            "inference_scope": "last_selected_snapshot_only",
            "local_events_only_scope": "working_memory_last_window_only"
            if normalized_context_mode == CONTEXT_MODE_LOCAL_EVENTS_ONLY
            else None,
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
            snapshot_stride=snapshot_stride,
            context_mode=normalized_context_mode,
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

    total_patients = len(all_results)
    total_window_predictions = sum(int(item.get("num_status_predictions", 0)) for item in all_results)

    status_distribution: Dict[str, int] = {}
    for item in all_results:
        per_patient = item.get("status_distribution", {})
        if not isinstance(per_patient, dict):
            continue
        for label, count in per_patient.items():
            status_distribution[str(label)] = status_distribution.get(str(label), 0) + int(count)

    status_token_totals = _sum_numeric_stats([item.get("status_prediction_tokens", {}) for item in all_results])

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print(f"Total Patients: {total_patients}")
    print(f"Total Status Predictions: {total_window_predictions}")
    print("\nStatus Distribution:")
    for label, count in sorted(status_distribution.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {label}: {count}")
    print("\nStatus Predictor Tokens:")
    print(f"  Input: {int(status_token_totals.get('input_tokens', 0))}")
    print(f"  Output: {int(status_token_totals.get('output_tokens', 0))}")
    print(f"  Total: {int(status_token_totals.get('total_tokens', 0))}")

    aggregate = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment": "patient_status_experiment",
        "memory_run": str(memory_run_dir),
        "context_mode": str(normalized_context_mode),
        "snapshot_stride": snapshot_stride,
        "num_workers": num_workers,
        "local_events_only_scope": "working_memory_last_window_only"
        if normalized_context_mode == CONTEXT_MODE_LOCAL_EVENTS_ONLY
        else None,
        "total_patients": total_patients,
        "total_window_predictions": total_window_predictions,
        "status_distribution": status_distribution,
        "status_prediction_tokens": {
            "input_tokens": int(status_token_totals.get("input_tokens", 0)),
            "output_tokens": int(status_token_totals.get("output_tokens", 0)),
            "total_tokens": int(status_token_totals.get("total_tokens", 0)),
        },
        "individual_results": sorted(all_results, key=lambda item: (item["subject_id"], item["icu_stay_id"])),
    }

    with open(results_dir / AGGREGATE_FILENAME, "w") as f:
        json.dump(aggregate, f, indent=2)

    return aggregate


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Patient status prediction from precomputed MedEvo memory and event baselines"
    )
    parser.add_argument(
        "--memory-run",
        type=str,
        required=True,
        help="Path to memory run directory produced by experiments/create_memory.py.",
    )
    parser.add_argument(
        "--snapshot-stride",
        type=int,
        default=1,
        help="Use every k-th memory snapshot for status prediction (default: 1).",
    )
    parser.add_argument(
        "--context-mode",
        type=str,
        required=True,
        choices=list(SUPPORTED_CONTEXT_MODES),
        help=(
            "Context construction mode: "
            "med_evo_memory | full_history_events | local_events_only."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for patient processing (default: 4).",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--no-logging", action="store_true", help="Disable detailed LLM call logging")
    parser.add_argument(
        "--patient-stay-ids",
        type=str,
        default=None,
        help="Optional CSV with columns subject_id,icu_stay_id to filter patients from memory run.",
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("RUNNING: PATIENT STATUS EXPERIMENT")
    print(f"{'='*80}\n")

    run_experiment(
        memory_run=args.memory_run,
        snapshot_stride=args.snapshot_stride,
        context_mode=args.context_mode,
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
