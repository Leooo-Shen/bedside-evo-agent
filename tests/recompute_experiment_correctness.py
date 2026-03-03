"""
Recompute outcome-matching correctness for experiment result directories.

This script updates, in-place:
- patients/*/prediction.json
- aggregate_results.json

Usage:
    python tests/recompute_experiment_correctness.py <exp_dir1> [<exp_dir2> ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.outcome_utils import evaluate_outcome_match, normalize_outcome_label


@dataclass
class PatientRecord:
    key: str
    is_correct: bool
    actual_norm: Optional[str]
    predicted_norm: Optional[str]
    confidence: float


def _extract_predicted_outcome(payload: Dict) -> Optional[str]:
    predicted = payload.get("predicted_outcome")
    if predicted is not None:
        return str(predicted)

    nested = payload.get("prediction", {}).get("survival_prediction", {}).get("outcome")
    if nested is not None:
        return str(nested)

    return None


def _build_patient_key(payload: Dict, fallback: str) -> str:
    subject_id = payload.get("subject_id")
    icu_stay_id = payload.get("icu_stay_id")
    if subject_id is not None and icu_stay_id is not None:
        return f"{subject_id}_{icu_stay_id}"
    return fallback


def _recompute_single_payload(payload: Dict) -> tuple[bool, Optional[str], Optional[str]]:
    predicted = _extract_predicted_outcome(payload)
    actual = payload.get("actual_outcome")
    return evaluate_outcome_match(predicted=predicted, actual=actual)


def _compute_group_stats(rows: list[Dict]) -> Dict[str, float]:
    survived_total = 0
    survived_correct = 0
    died_total = 0
    died_correct = 0

    for row in rows:
        actual_norm = row.get("actual_outcome_normalized")
        if actual_norm is None:
            actual_norm = normalize_outcome_label(row.get("actual_outcome"))

        is_correct = bool(row.get("is_correct", False))

        if actual_norm == "survive":
            survived_total += 1
            if is_correct:
                survived_correct += 1
        elif actual_norm == "die":
            died_total += 1
            if is_correct:
                died_correct += 1

    survived_accuracy = survived_correct / survived_total if survived_total else 0.0
    died_accuracy = died_correct / died_total if died_total else 0.0

    return {
        "survived_total": survived_total,
        "survived_correct": survived_correct,
        "survived_accuracy": survived_accuracy,
        "died_total": died_total,
        "died_correct": died_correct,
        "died_accuracy": died_accuracy,
    }


def recompute_experiment(experiment_dir: Path, dry_run: bool = False) -> Dict[str, float]:
    patients_dir = experiment_dir / "patients"
    if not patients_dir.exists():
        raise FileNotFoundError(f"'patients' directory not found: {patients_dir}")

    patient_files = sorted(patients_dir.glob("*/prediction.json"))
    if not patient_files:
        raise FileNotFoundError(f"No prediction files found under: {patients_dir}")

    old_correct_count = 0
    new_correct_count = 0
    changed_count = 0
    patient_records: Dict[str, PatientRecord] = {}

    for path in patient_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        old_is_correct = bool(payload.get("is_correct", False))
        if old_is_correct:
            old_correct_count += 1

        is_correct, pred_norm, actual_norm = _recompute_single_payload(payload)

        if old_is_correct != is_correct:
            changed_count += 1

        if is_correct:
            new_correct_count += 1

        payload["is_correct"] = is_correct
        payload["predicted_outcome_normalized"] = pred_norm
        payload["actual_outcome_normalized"] = actual_norm

        if not dry_run:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        key = _build_patient_key(payload, fallback=path.parent.name)
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        patient_records[key] = PatientRecord(
            key=key,
            is_correct=is_correct,
            actual_norm=actual_norm,
            predicted_norm=pred_norm,
            confidence=confidence,
        )

    total = len(patient_files)
    old_accuracy = old_correct_count / total if total else 0.0
    new_accuracy = new_correct_count / total if total else 0.0

    aggregate_file = experiment_dir / "aggregate_results.json"
    aggregate_changes = 0
    aggregate_old_correct = None
    aggregate_old_accuracy = None
    aggregate_new_correct = None
    aggregate_new_accuracy = None
    final_correct = new_correct_count
    final_accuracy = new_accuracy
    survived_total = 0
    survived_correct = 0
    survived_accuracy = 0.0
    died_total = 0
    died_correct = 0
    died_accuracy = 0.0

    if aggregate_file.exists():
        aggregate = json.loads(aggregate_file.read_text(encoding="utf-8"))
        aggregate_old_correct = aggregate.get("correct_predictions")
        aggregate_old_accuracy = aggregate.get("accuracy")

        individual_results = aggregate.get("individual_results")
        if isinstance(individual_results, list):
            for row in individual_results:
                row_key = _build_patient_key(row, fallback="")
                old_row_correct = bool(row.get("is_correct", False))

                if row_key in patient_records:
                    rec = patient_records[row_key]
                    row["is_correct"] = rec.is_correct
                    row["predicted_outcome_normalized"] = rec.predicted_norm
                    row["actual_outcome_normalized"] = rec.actual_norm
                else:
                    row_is_correct, row_pred_norm, row_actual_norm = _recompute_single_payload(row)
                    row["is_correct"] = row_is_correct
                    row["predicted_outcome_normalized"] = row_pred_norm
                    row["actual_outcome_normalized"] = row_actual_norm

                if old_row_correct != bool(row["is_correct"]):
                    aggregate_changes += 1

            total_individual = len(individual_results)
            correct_individual = sum(1 for r in individual_results if bool(r.get("is_correct", False)))
            accuracy_individual = correct_individual / total_individual if total_individual else 0.0

            aggregate["total_patients"] = total_individual
            aggregate["correct_predictions"] = correct_individual
            aggregate["accuracy"] = accuracy_individual
            final_correct = correct_individual
            final_accuracy = accuracy_individual

            confidences = [float(r.get("confidence", 0.0) or 0.0) for r in individual_results]
            aggregate["avg_confidence"] = (sum(confidences) / len(confidences)) if confidences else 0.0

            group_stats = _compute_group_stats(individual_results)
            survived_total = int(group_stats["survived_total"])
            survived_correct = int(group_stats["survived_correct"])
            survived_accuracy = float(group_stats["survived_accuracy"])
            died_total = int(group_stats["died_total"])
            died_correct = int(group_stats["died_correct"])
            died_accuracy = float(group_stats["died_accuracy"])

            aggregate["survived_patients"] = survived_total
            aggregate["survived_correct"] = survived_correct
            aggregate["survived_accuracy"] = survived_accuracy
            aggregate["died_patients"] = died_total
            aggregate["died_correct"] = died_correct
            aggregate["died_accuracy"] = died_accuracy

            aggregate_new_correct = correct_individual
            aggregate_new_accuracy = accuracy_individual
        else:
            aggregate["total_patients"] = total
            aggregate["correct_predictions"] = new_correct_count
            aggregate["accuracy"] = new_accuracy
            final_correct = new_correct_count
            final_accuracy = new_accuracy
            confidences = [r.confidence for r in patient_records.values()]
            aggregate["avg_confidence"] = (sum(confidences) / len(confidences)) if confidences else 0.0

            survived_total = sum(1 for r in patient_records.values() if r.actual_norm == "survive")
            survived_correct = sum(1 for r in patient_records.values() if r.actual_norm == "survive" and r.is_correct)
            died_total = sum(1 for r in patient_records.values() if r.actual_norm == "die")
            died_correct = sum(1 for r in patient_records.values() if r.actual_norm == "die" and r.is_correct)
            survived_accuracy = survived_correct / survived_total if survived_total else 0.0
            died_accuracy = died_correct / died_total if died_total else 0.0

            aggregate["survived_patients"] = survived_total
            aggregate["survived_correct"] = survived_correct
            aggregate["survived_accuracy"] = survived_accuracy
            aggregate["died_patients"] = died_total
            aggregate["died_correct"] = died_correct
            aggregate["died_accuracy"] = died_accuracy

            aggregate_new_correct = new_correct_count
            aggregate_new_accuracy = new_accuracy

        if not dry_run:
            aggregate_file.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    else:
        survived_total = sum(1 for r in patient_records.values() if r.actual_norm == "survive")
        survived_correct = sum(1 for r in patient_records.values() if r.actual_norm == "survive" and r.is_correct)
        died_total = sum(1 for r in patient_records.values() if r.actual_norm == "die")
        died_correct = sum(1 for r in patient_records.values() if r.actual_norm == "die" and r.is_correct)
        survived_accuracy = survived_correct / survived_total if survived_total else 0.0
        died_accuracy = died_correct / died_total if died_total else 0.0
        final_correct = new_correct_count
        final_accuracy = new_accuracy

    return {
        "total_patients": total,
        "patient_old_correct": old_correct_count,
        "patient_new_correct": new_correct_count,
        "patient_old_accuracy": old_accuracy,
        "patient_new_accuracy": new_accuracy,
        "patient_changed": changed_count,
        "aggregate_old_correct": aggregate_old_correct if aggregate_old_correct is not None else -1,
        "aggregate_new_correct": aggregate_new_correct if aggregate_new_correct is not None else -1,
        "aggregate_old_accuracy": aggregate_old_accuracy if aggregate_old_accuracy is not None else -1,
        "aggregate_new_accuracy": aggregate_new_accuracy if aggregate_new_accuracy is not None else -1,
        "aggregate_changed_rows": aggregate_changes,
        "final_correct": final_correct,
        "final_accuracy": final_accuracy,
        "survived_total": survived_total,
        "survived_correct": survived_correct,
        "survived_accuracy": survived_accuracy,
        "died_total": died_total,
        "died_correct": died_correct,
        "died_accuracy": died_accuracy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute experiment correctness using robust outcome matching.")
    parser.add_argument("experiment_dirs", nargs="+", type=Path, help="One or more experiment result directories")
    parser.add_argument("--dry-run", action="store_true", help="Only report changes without writing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("Recompute Experiment Correctness")
    print("=" * 80)

    for exp_dir in args.experiment_dirs:
        exp_dir = exp_dir.resolve()
        print(f"\n[{exp_dir}]")
        stats = recompute_experiment(exp_dir, dry_run=args.dry_run)

        print(f"Total Patients: {int(stats['total_patients'])}")
        print(f"Correct: {int(stats['final_correct'])}")
        print(f"Accuracy: {stats['final_accuracy']:.2%}")
        print()
        print(
            f"Survived: {int(stats['survived_total'])} patients, "
            f"{int(stats['survived_correct'])} correct ({stats['survived_accuracy']:.2%})"
        )
        print(
            f"Died: {int(stats['died_total'])} patients, "
            f"{int(stats['died_correct'])} correct ({stats['died_accuracy']:.2%})"
        )
        print(
            f"Updated is_correct rows: patients={int(stats['patient_changed'])}, "
            f"aggregate={int(stats['aggregate_changed_rows'])}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
