"""Export unique (subject_id, icu_stay_id) pairs from a windows JSONL file."""

import argparse
import csv
import json
from pathlib import Path
from typing import Set, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract unique patient ICU-stay IDs from selected windows JSONL."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Path to input windows JSONL (e.g., selected_windows_full.jsonl).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional output CSV path. "
            "Default: <input-dir>/<input-stem>_patient_stay_ids.csv"
        ),
    )
    return parser.parse_args()


def _extract_pairs(input_jsonl: Path) -> Set[Tuple[int, int]]:
    pairs: Set[Tuple[int, int]] = set()
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            if "subject_id" not in record or "icu_stay_id" not in record:
                raise ValueError(
                    f"Missing subject_id/icu_stay_id at line {line_no} in {input_jsonl}"
                )
            pairs.add((int(record["subject_id"]), int(record["icu_stay_id"])))
    return pairs


def _default_output_path(input_jsonl: Path) -> Path:
    return input_jsonl.parent / f"{input_jsonl.stem}_patient_stay_ids.csv"


def _write_pairs_csv(output_csv: Path, pairs: Set[Tuple[int, int]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "icu_stay_id"])
        for subject_id, icu_stay_id in sorted(pairs):
            writer.writerow([subject_id, icu_stay_id])


def main() -> None:
    args = _parse_args()
    input_jsonl = args.input_jsonl
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    output_csv = args.output_csv if args.output_csv is not None else _default_output_path(input_jsonl)
    pairs = _extract_pairs(input_jsonl)
    _write_pairs_csv(output_csv, pairs)

    print(f"Input JSONL: {input_jsonl}")
    print(f"Unique patient-stay pairs: {len(pairs)}")
    print(f"Output CSV: {output_csv}")


if __name__ == "__main__":
    main()
