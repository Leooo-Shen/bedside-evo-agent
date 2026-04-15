#!/usr/bin/env python3
"""Split sparse annotation cases across three doctors with pairwise overlap."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

CaseKey = Tuple[int, int]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split case-level sparse windows into AB/BC/AC groups so each case is labeled by two doctors."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Path to selected_windows_sparse.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for split artifacts")
    parser.add_argument("--ab-count", type=int, required=True, help="Number of cases assigned to doctors A and B")
    parser.add_argument("--bc-count", type=int, required=True, help="Number of cases assigned to doctors B and C")
    parser.add_argument("--ac-count", type=int, required=True, help="Number of cases assigned to doctors A and C")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for deterministic case assignment")
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            if "subject_id" not in record or "icu_stay_id" not in record:
                raise ValueError(f"Missing subject_id/icu_stay_id at line {line_no} in {path}")
            records.append(record)
    if not records:
        raise ValueError(f"No JSON records found in {path}")
    return records


def _group_records_by_case(records: Sequence[Mapping[str, Any]]) -> Dict[CaseKey, List[Dict[str, Any]]]:
    grouped: Dict[CaseKey, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        case_key = (int(record["subject_id"]), int(record["icu_stay_id"]))
        grouped[case_key].append(dict(record))
    return grouped


def _shuffle_cases(case_keys: Iterable[CaseKey], seed: int) -> List[CaseKey]:
    shuffled = list(case_keys)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def _build_case_assignments(
    shuffled_case_keys: Sequence[CaseKey],
    ab_count: int,
    bc_count: int,
    ac_count: int,
) -> List[Dict[str, Any]]:
    pair_plan: List[Tuple[str, Tuple[str, str], int]] = [
        ("AB", ("A", "B"), ab_count),
        ("BC", ("B", "C"), bc_count),
        ("AC", ("A", "C"), ac_count),
    ]
    expected_total = sum(count for _, _, count in pair_plan)
    if expected_total != len(shuffled_case_keys):
        raise ValueError(
            f"Pair counts sum to {expected_total}, but input has {len(shuffled_case_keys)} unique cases."
        )

    assignments: List[Dict[str, Any]] = []
    cursor = 0
    for pair_name, doctors, count in pair_plan:
        if count < 0:
            raise ValueError(f"{pair_name} count must be non-negative, got {count}.")
        for case_key in shuffled_case_keys[cursor : cursor + count]:
            assignments.append(
                {
                    "subject_id": int(case_key[0]),
                    "icu_stay_id": int(case_key[1]),
                    "pair_group": pair_name,
                    "doctor_1": doctors[0],
                    "doctor_2": doctors[1],
                }
            )
        cursor += count
    return assignments


def _doctor_case_sets(assignments: Sequence[Mapping[str, Any]]) -> Dict[str, Set[CaseKey]]:
    doctor_cases: Dict[str, Set[CaseKey]] = {"A": set(), "B": set(), "C": set()}
    for row in assignments:
        case_key = (int(row["subject_id"]), int(row["icu_stay_id"]))
        doctor_cases[str(row["doctor_1"])].add(case_key)
        doctor_cases[str(row["doctor_2"])].add(case_key)
    return doctor_cases


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _build_doctor_case_rows(assignments: Sequence[Mapping[str, Any]], doctor: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in assignments:
        if str(row["doctor_1"]) != doctor and str(row["doctor_2"]) != doctor:
            continue
        rows.append(
            {
                "subject_id": int(row["subject_id"]),
                "icu_stay_id": int(row["icu_stay_id"]),
                "pair_group": str(row["pair_group"]),
                "other_doctor": str(row["doctor_2"] if str(row["doctor_1"]) == doctor else row["doctor_1"]),
            }
        )
    rows.sort(key=lambda item: (int(item["subject_id"]), int(item["icu_stay_id"])))
    return rows


def _doctor_windows(
    source_records: Sequence[Mapping[str, Any]],
    doctor_cases: Set[CaseKey],
) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    for record in source_records:
        case_key = (int(record["subject_id"]), int(record["icu_stay_id"]))
        if case_key in doctor_cases:
            windows.append(dict(record))
    return windows


def _validate_doctor_windows(
    doctor_windows: MutableMapping[str, Sequence[Mapping[str, Any]]],
    expected_case_count: MutableMapping[str, int],
    case_windows: Mapping[CaseKey, Sequence[Mapping[str, Any]]],
) -> None:
    for doctor, windows in doctor_windows.items():
        unique_cases = {
            (int(record["subject_id"]), int(record["icu_stay_id"]))
            for record in windows
        }
        if len(unique_cases) != int(expected_case_count[doctor]):
            raise ValueError(
                f"Doctor {doctor} expected {expected_case_count[doctor]} cases, got {len(unique_cases)}."
            )
        expected_windows = sum(len(case_windows[case_key]) for case_key in unique_cases)
        if len(windows) != expected_windows:
            raise ValueError(
                f"Doctor {doctor} expected {expected_windows} windows, got {len(windows)}."
            )


def main() -> None:
    args = _parse_args()
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")

    records = _read_jsonl(args.input_jsonl)
    case_to_records = _group_records_by_case(records)
    shuffled_case_keys = _shuffle_cases(case_to_records.keys(), seed=int(args.seed))
    assignments = _build_case_assignments(
        shuffled_case_keys,
        ab_count=int(args.ab_count),
        bc_count=int(args.bc_count),
        ac_count=int(args.ac_count),
    )
    doctor_cases = _doctor_case_sets(assignments)

    expected_case_count = {
        "A": int(args.ab_count) + int(args.ac_count),
        "B": int(args.ab_count) + int(args.bc_count),
        "C": int(args.ac_count) + int(args.bc_count),
    }
    doctor_windows = {
        doctor: _doctor_windows(records, case_keys)
        for doctor, case_keys in doctor_cases.items()
    }
    _validate_doctor_windows(doctor_windows, expected_case_count, case_to_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignment_csv = args.output_dir / "case_assignments.csv"
    _write_csv(
        assignment_csv,
        assignments,
        fieldnames=["subject_id", "icu_stay_id", "pair_group", "doctor_1", "doctor_2"],
    )

    for doctor in ("A", "B", "C"):
        case_rows = _build_doctor_case_rows(assignments, doctor)
        _write_csv(
            args.output_dir / f"doctor_{doctor}_cases.csv",
            case_rows,
            fieldnames=["subject_id", "icu_stay_id", "pair_group", "other_doctor"],
        )
        _write_jsonl(args.output_dir / f"doctor_{doctor}_selected_windows_sparse.jsonl", doctor_windows[doctor])

    pair_counts: Dict[str, int] = defaultdict(int)
    for row in assignments:
        pair_counts[str(row["pair_group"])] += 1
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_dir": str(args.output_dir),
        "seed": int(args.seed),
        "num_unique_cases": int(len(case_to_records)),
        "num_windows": int(len(records)),
        "pair_case_counts": {key: int(value) for key, value in sorted(pair_counts.items())},
        "doctor_case_counts": {doctor: int(len(cases)) for doctor, cases in doctor_cases.items()},
        "doctor_window_counts": {doctor: int(len(windows)) for doctor, windows in doctor_windows.items()},
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Input: {args.input_jsonl}")
    print(f"Output directory: {args.output_dir}")
    print(f"Unique cases: {len(case_to_records)}")
    print(f"Windows: {len(records)}")
    print(f"Pair case counts: {dict(sorted(pair_counts.items()))}")
    print(
        "Doctor case counts: "
        + ", ".join(f"{doctor}={len(doctor_cases[doctor])}" for doctor in ("A", "B", "C"))
    )
    print(
        "Doctor window counts: "
        + ", ".join(f"{doctor}={len(doctor_windows[doctor])}" for doctor in ("A", "B", "C"))
    )


if __name__ == "__main__":
    main()
