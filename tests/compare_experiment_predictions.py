"""
Compare prediction correctness between two experiment result directories.

Find patient IDs where:
- experiment 1 prediction is correct
- experiment 2 prediction is incorrect

Usage:
    python tests/compare_experiment_predictions.py \
        /path/to/exp1 \
        /path/to/exp2 \
        [--output /path/to/output.txt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.outcome_utils import evaluate_outcome_match


def load_correctness_map(experiment_dir: Path) -> Dict[str, bool]:
    """
    Load patient-level correctness from prediction.json files.

    Returns:
        Mapping: patient_id (folder name) -> is_correct
    """
    patients_dir = experiment_dir / "patients"
    if not patients_dir.exists():
        raise FileNotFoundError(f"'patients' directory not found: {patients_dir}")

    correctness: Dict[str, bool] = {}
    for prediction_path in sorted(patients_dir.glob("*/prediction.json")):
        patient_id = prediction_path.parent.name
        with prediction_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        predicted = payload.get("predicted_outcome")
        if predicted is None:
            predicted = payload.get("prediction", {}).get("survival_prediction", {}).get("outcome")
        actual = payload.get("actual_outcome")

        if predicted is None or actual is None:
            is_correct = bool(payload.get("is_correct", False))
        else:
            is_correct, _, _ = evaluate_outcome_match(predicted=predicted, actual=actual)

        correctness[patient_id] = bool(is_correct)

    return correctness


def compare_experiments(exp1_dir: Path, exp2_dir: Path) -> Tuple[list[str], list[str], list[str]]:
    """
    Compare two experiment directories.

    Returns:
        target_ids: IDs where exp1 correct and exp2 incorrect
        only_in_exp1: IDs only in exp1
        only_in_exp2: IDs only in exp2
    """
    exp1 = load_correctness_map(exp1_dir)
    exp2 = load_correctness_map(exp2_dir)

    ids1 = set(exp1.keys())
    ids2 = set(exp2.keys())

    common_ids = ids1 & ids2
    only_in_exp1 = sorted(ids1 - ids2)
    only_in_exp2 = sorted(ids2 - ids1)

    target_ids = sorted([pid for pid in common_ids if exp1[pid] and not exp2[pid]])
    return target_ids, only_in_exp1, only_in_exp2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find patient IDs where exp1 is correct but exp2 is incorrect."
    )
    parser.add_argument("experiment1", type=Path, help="Path to experiment 1 directory")
    parser.add_argument("experiment2", type=Path, help="Path to experiment 2 directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output text file (one patient ID per line)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    exp1_dir = args.experiment1.resolve()
    exp2_dir = args.experiment2.resolve()

    target_ids, only_in_exp1, only_in_exp2 = compare_experiments(exp1_dir, exp2_dir)

    print("=" * 80)
    print("Experiment Comparison")
    print("=" * 80)
    print(f"Experiment 1: {exp1_dir}")
    print(f"Experiment 2: {exp2_dir}")
    print(f"Matched patient IDs: {len(target_ids)} (exp1 correct && exp2 incorrect)")

    if target_ids:
        print("\nPatient IDs:")
        for pid in target_ids:
            print(pid)
    else:
        print("\nNo matched patient IDs found.")

    if only_in_exp1:
        print(f"\nIDs only in experiment 1: {len(only_in_exp1)}")
    if only_in_exp2:
        print(f"IDs only in experiment 2: {len(only_in_exp2)}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(target_ids) + ("\n" if target_ids else ""), encoding="utf-8")
        print(f"\nSaved result to: {args.output}")


if __name__ == "__main__":
    main()
