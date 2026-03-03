"""
Re-run predictor only on an existing multi-agent experiment, while removing
selected sections from predictor context.

Default ablation removes:
- "## Historical Key Events"
- "## Status Trajectory"

This does NOT rerun observer/memory/reflection. It reuses each patient's
saved final_state and only issues a new predictor LLM call.

Usage:
    python tests/rerun_predictor_without_sections.py \
        /path/to/source_experiment_dir
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.agent_fold_multi import _parse_json_response
from config.config import get_config
from model.llms import LLMClient
from prompts.shared_prompts import get_prediction_prompt
from utils.outcome_utils import evaluate_outcome_match

REMOVED_SECTION_TITLES = {"historical key events", "status trajectory"}


def _normalize_section_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower().rstrip(":"))


def remove_selected_sections(markdown_text: str, removed_titles: set[str]) -> str:
    """Remove selected level-2 markdown sections from a context string."""
    lines = markdown_text.splitlines()
    kept: List[str] = []
    skip = False

    for line in lines:
        header_match = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if header_match:
            title = _normalize_section_title(header_match.group(1))
            skip = title in removed_titles
            if skip:
                continue

        if not skip:
            kept.append(line)

    text = "\n".join(kept).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_observation_hours_from_predictor_prompt(prompt_text: str) -> Optional[float]:
    """
    Try to recover observation hours from stored predictor prompt.

    Looks for:
    "## Clinical Contexts (First 12 Hours After ICU Admission)"
    """
    match = re.search(r"First\s+([0-9]+(?:\.[0-9]+)?)\s+Hours\s+After\s+ICU\s+Admission", prompt_text, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def load_source_runtime_settings(source_experiment_dir: Path) -> Dict[str, Any]:
    """Infer provider/model/predictor thinking/observation_hours from source logs."""
    llm_files = sorted(source_experiment_dir.glob("patients/*/llm_calls.json"))
    if not llm_files:
        raise FileNotFoundError(f"No llm_calls.json found under: {source_experiment_dir / 'patients'}")

    first_llm = json.loads(llm_files[0].read_text(encoding="utf-8"))
    provider = first_llm.get("llm_provider")
    model = first_llm.get("llm_model")
    predictor_use_thinking = True

    pipeline_agents = first_llm.get("pipeline_agents", [])
    for agent in pipeline_agents:
        if agent.get("name") == "predictor":
            thinking = agent.get("thinking")
            if isinstance(thinking, bool):
                predictor_use_thinking = thinking
            break

    # Try to recover observation_hours from stored predictor prompt.
    observation_hours = None
    for call in first_llm.get("calls", []):
        if call.get("metadata", {}).get("step_type") == "predictor":
            observation_hours = extract_observation_hours_from_predictor_prompt(call.get("prompt", ""))
            if observation_hours is not None:
                break

    return {
        "provider": provider,
        "model": model,
        "predictor_use_thinking": predictor_use_thinking,
        "observation_hours": observation_hours,
    }


def compute_group_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    survived = [r for r in rows if r.get("actual_outcome_normalized") == "survive"]
    died = [r for r in rows if r.get("actual_outcome_normalized") == "die"]

    survived_correct = sum(1 for r in survived if r.get("is_correct"))
    died_correct = sum(1 for r in died if r.get("is_correct"))
    total_correct = sum(1 for r in rows if r.get("is_correct"))

    total = len(rows)
    return {
        "total_patients": total,
        "correct_predictions": total_correct,
        "accuracy": (total_correct / total) if total else 0.0,
        "survived_patients": len(survived),
        "survived_correct": survived_correct,
        "survived_accuracy": (survived_correct / len(survived)) if survived else 0.0,
        "died_patients": len(died),
        "died_correct": died_correct,
        "died_accuracy": (died_correct / len(died)) if died else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run predictor with selected sections removed from context.")
    parser.add_argument(
        "source_experiment_dir",
        type=Path,
        help="Path to source experiment directory (e.g., experiment_results/multi_20260302_180509)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for predictor-only ablation results",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Optional cap for number of patients (for quick testing)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Number of parallel predictor workers across patients (default: 1)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Override LLM provider (default: inferred from source logs)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override LLM model (default: inferred from source logs)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override LLM temperature (default: from config)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max tokens (default: from config)",
    )
    parser.add_argument(
        "--observation-hours",
        type=float,
        default=None,
        help="Override predictor observation_hours in prompt (default: inferred from source logs, then config)",
    )
    parser.add_argument(
        "--predictor-use-thinking",
        choices=["true", "false"],
        default=None,
        help="Override predictor thinking flag (default: inferred from source logs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_experiment_dir.resolve()
    patients_root = source_dir / "patients"

    if not patients_root.exists():
        raise FileNotFoundError(f"'patients' directory not found: {patients_root}")

    runtime = load_source_runtime_settings(source_dir)
    cfg = get_config()

    provider = args.provider or runtime["provider"] or cfg.llm_provider
    model = args.model or runtime["model"] or cfg.llm_model
    temperature = args.temperature if args.temperature is not None else cfg.llm_temperature
    max_tokens = args.max_tokens if args.max_tokens is not None else cfg.llm_max_tokens
    observation_hours = (
        args.observation_hours
        if args.observation_hours is not None
        else runtime["observation_hours"] if runtime["observation_hours"] is not None else cfg.agent_observation_hours
    )
    if args.predictor_use_thinking is None:
        predictor_use_thinking = bool(runtime["predictor_use_thinking"])
    else:
        predictor_use_thinking = args.predictor_use_thinking.lower() == "true"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = source_dir.parent / f"{source_dir.name}_predictor_no_hist_status_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "patients").mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Predictor-Only Ablation Run")
    print("=" * 80)
    print(f"Source experiment: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Provider/model: {provider} / {model}")
    print(f"Temperature/max_tokens: {temperature} / {max_tokens}")
    print(f"Observation hours: {observation_hours}")
    print(f"Predictor thinking: {predictor_use_thinking}")
    print(f"Max workers: {args.max_workers}")
    print("Removed sections: Historical Key Events, Status Trajectory")
    prompt_template = get_prediction_prompt(
        use_thinking=predictor_use_thinking,
        observation_hours=observation_hours,
    )

    prediction_files = sorted(patients_root.glob("*/prediction.json"))
    if args.max_patients is not None:
        prediction_files = prediction_files[: args.max_patients]

    individual_results: List[Dict[str, Any]] = []
    source_results: List[Dict[str, Any]] = []
    total_files = len(prediction_files)

    def process_one_patient(idx: int, prediction_file: Path) -> Dict[str, Any]:
        patient_dir_name = prediction_file.parent.name
        patient_output_dir = output_dir / "patients" / patient_dir_name
        patient_output_dir.mkdir(parents=True, exist_ok=True)

        payload = json.loads(prediction_file.read_text(encoding="utf-8"))
        subject_id = payload.get("subject_id")
        icu_stay_id = payload.get("icu_stay_id")
        actual_outcome = payload.get("actual_outcome")
        source_predicted = payload.get("predicted_outcome")
        source_is_correct, source_pred_norm, source_actual_norm = evaluate_outcome_match(
            predicted=source_predicted,
            actual=actual_outcome,
        )

        final_state = payload.get("final_state")
        if not isinstance(final_state, str) or not final_state.strip():
            print(f"[{idx}/{total_files}] {patient_dir_name}: skip (missing final_state)")
            return {
                "index": idx,
                "skipped": True,
                "result_row": None,
                "source_row": None,
            }

        ablated_context = remove_selected_sections(
            final_state,
            removed_titles=REMOVED_SECTION_TITLES,
        )
        prompt = prompt_template.format(context=ablated_context)

        print(f"[{idx}/{total_files}] {patient_dir_name}: running predictor...")
        llm = LLMClient(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = llm.chat(prompt=prompt, response_format="text")
        raw_response = response.get("content", "")
        usage = response.get("usage", {})
        parsed = _parse_json_response(raw_response)

        predicted_outcome = parsed.get("survival_prediction", {}).get("outcome", "unknown")
        confidence = parsed.get("survival_prediction", {}).get("confidence", 0.0)
        is_correct, pred_norm, actual_norm = evaluate_outcome_match(
            predicted=predicted_outcome,
            actual=actual_outcome,
        )

        result_row = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "patient_id": patient_dir_name,
            "actual_outcome": actual_outcome,
            "actual_outcome_normalized": actual_norm,
            "predicted_outcome": predicted_outcome,
            "predicted_outcome_normalized": pred_norm,
            "is_correct": is_correct,
            "confidence": confidence,
            "source_predicted_outcome": source_predicted,
            "source_predicted_outcome_normalized": source_pred_norm,
            "source_is_correct": source_is_correct,
            "source_confidence": payload.get("confidence", 0.0),
            "num_windows": payload.get("num_windows"),
            "prediction": parsed,
            "ablation": {
                "removed_sections": ["Historical Key Events", "Status Trajectory"],
                "source_experiment_dir": str(source_dir),
            },
        }

        save_prediction = {
            "subject_id": subject_id,
            "icu_stay_id": icu_stay_id,
            "actual_outcome": actual_outcome,
            "actual_outcome_normalized": actual_norm,
            "predicted_outcome": predicted_outcome,
            "predicted_outcome_normalized": pred_norm,
            "is_correct": is_correct,
            "confidence": confidence,
            "num_windows": payload.get("num_windows"),
            "prediction": parsed,
            "source_prediction": {
                "predicted_outcome": source_predicted,
                "predicted_outcome_normalized": source_pred_norm,
                "is_correct": source_is_correct,
                "confidence": payload.get("confidence", 0.0),
            },
            "ablation": {
                "removed_sections": ["Historical Key Events", "Status Trajectory"],
                "source_experiment_dir": str(source_dir),
            },
        }
        (patient_output_dir / "prediction.json").write_text(
            json.dumps(save_prediction, indent=2),
            encoding="utf-8",
        )

        save_call = {
            "patient_id": patient_dir_name,
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "observation_hours": observation_hours,
            "predictor_use_thinking": predictor_use_thinking,
            "removed_sections": ["Historical Key Events", "Status Trajectory"],
            "context_after_removal": ablated_context,
            "prompt": prompt,
            "response": raw_response,
            "parsed_response": parsed,
            "usage": usage,
        }
        (patient_output_dir / "predictor_call.json").write_text(
            json.dumps(save_call, indent=2),
            encoding="utf-8",
        )
        return {
            "index": idx,
            "skipped": False,
            "result_row": result_row,
            "source_row": {
                "actual_outcome_normalized": source_actual_norm,
                "is_correct": source_is_correct,
            },
        }

    if args.max_workers <= 1:
        for idx, prediction_file in enumerate(prediction_files, 1):
            patient_result = process_one_patient(idx, prediction_file)
            if not patient_result["skipped"]:
                source_results.append(patient_result["source_row"])
                individual_results.append(patient_result["result_row"])
    else:
        completed: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(process_one_patient, idx, prediction_file)
                for idx, prediction_file in enumerate(prediction_files, 1)
            ]
            for future in as_completed(futures):
                completed.append(future.result())

        completed.sort(key=lambda row: row["index"])
        for patient_result in completed:
            if not patient_result["skipped"]:
                source_results.append(patient_result["source_row"])
                individual_results.append(patient_result["result_row"])

    new_metrics = compute_group_metrics(individual_results)
    source_metrics = compute_group_metrics(source_results)

    aggregate = {
        "timestamp": timestamp,
        "mode": "predictor_only_ablation",
        "source_experiment_dir": str(source_dir),
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "observation_hours": observation_hours,
        "predictor_use_thinking": predictor_use_thinking,
        "removed_sections": ["Historical Key Events", "Status Trajectory"],
        "new_metrics": new_metrics,
        "source_metrics": source_metrics,
        "delta": {
            "correct_predictions": new_metrics["correct_predictions"] - source_metrics["correct_predictions"],
            "accuracy": new_metrics["accuracy"] - source_metrics["accuracy"],
        },
        "individual_results": individual_results,
    }
    (output_dir / "aggregate_results.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Ablation Results (Predictor Re-run)")
    print("=" * 80)

    print(f"Total Patients: {new_metrics['total_patients']}")
    print(f"Correct: {new_metrics['correct_predictions']}")
    print(f"Accuracy: {new_metrics['accuracy']:.2%}")
    print()
    print(
        f"Survived: {new_metrics['survived_patients']} patients, "
        f"{new_metrics['survived_correct']} correct ({new_metrics['survived_accuracy']:.2%})"
    )
    print(
        f"Died: {new_metrics['died_patients']} patients, "
        f"{new_metrics['died_correct']} correct ({new_metrics['died_accuracy']:.2%})"
    )

    print("\nSource (original experiment):")
    print(f"Total Patients: {source_metrics['total_patients']}")
    print(f"Correct: {source_metrics['correct_predictions']}")
    print(f"Accuracy: {source_metrics['accuracy']:.2%}")
    print(
        f"Survived: {source_metrics['survived_patients']} patients, "
        f"{source_metrics['survived_correct']} correct ({source_metrics['survived_accuracy']:.2%})"
    )
    print(
        f"Died: {source_metrics['died_patients']} patients, "
        f"{source_metrics['died_correct']} correct ({source_metrics['died_accuracy']:.2%})"
    )

    print("\nDelta (ablation - source):")
    print(f"Correct change: {aggregate['delta']['correct_predictions']:+d}")
    print(f"Accuracy change: {aggregate['delta']['accuracy']:+.2%}")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
