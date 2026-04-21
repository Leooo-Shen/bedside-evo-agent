from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.create_memory import resolve_memory_run_dir


LEGACY_SUPPORTING_IDS_PATTERN = re.compile(r"supporting_event_ids\s*=\s*\[([^\]]*)\]")
EVENT_PREFIX_PATTERN = re.compile(r"^\[(\-?\d+)\]\s")


@dataclass
class PatientRewriteStats:
    patient_id: str
    skipped: bool
    reason: str = ""
    event_map_size: int = 0
    snapshot_blocks: int = 0
    snapshot_items_converted: int = 0
    snapshot_items_unresolved: int = 0
    snapshot_items_unchanged: int = 0
    final_blocks: int = 0
    final_items_converted: int = 0
    final_items_unresolved: int = 0
    final_items_unchanged: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "skipped": self.skipped,
            "reason": self.reason,
            "event_map_size": self.event_map_size,
            "snapshot_blocks": self.snapshot_blocks,
            "snapshot_items_converted": self.snapshot_items_converted,
            "snapshot_items_unresolved": self.snapshot_items_unresolved,
            "snapshot_items_unchanged": self.snapshot_items_unchanged,
            "final_blocks": self.final_blocks,
            "final_items_converted": self.final_items_converted,
            "final_items_unresolved": self.final_items_unresolved,
            "final_items_unchanged": self.final_items_unchanged,
        }


def _read_json_object(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _parse_int_tokens(text: str) -> List[int]:
    values: List[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if not re.fullmatch(r"-?\d+", token):
            continue
        values.append(int(token))
    return values


def _extract_event_id(line: Any) -> Optional[int]:
    if not isinstance(line, str):
        return None
    match = EVENT_PREFIX_PATTERN.match(line.strip())
    if not match:
        return None
    return int(match.group(1))


def _add_event_line(mapping: Dict[int, str], line: Any) -> None:
    if not isinstance(line, str):
        return
    stripped = line.strip()
    match = EVENT_PREFIX_PATTERN.match(stripped)
    if not match:
        return
    mapping[int(match.group(1))] = stripped


def _ingest_payload_event_lines(mapping: Dict[int, str], payload: Dict[str, Any]) -> None:
    for window in payload.get("working_memory", []):
        if not isinstance(window, dict):
            continue
        for line in window.get("events", []):
            _add_event_line(mapping, line)

    for item in payload.get("trajectory_memory", []):
        if not isinstance(item, dict):
            continue
        for line in item.get("supporting_events", []):
            _add_event_line(mapping, line)

    for item in payload.get("insights", []):
        if not isinstance(item, dict):
            continue
        for line in item.get("supporting_evidence", []):
            _add_event_line(mapping, line)
        for line in item.get("counter_evidence", []):
            _add_event_line(mapping, line)

    for block in payload.get("critical_events_memory", []):
        if not isinstance(block, dict):
            continue
        for line in block.get("critical_events", []):
            _add_event_line(mapping, line)


def _rewrite_critical_event_lines(lines: Any, mapping: Dict[int, str]) -> Tuple[List[str], int, int, int]:
    if not isinstance(lines, list):
        return [], 0, 0, 0

    rewritten: List[str] = []
    seen = set()
    converted = 0
    unresolved = 0
    unchanged = 0

    for item in lines:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue

        legacy_match = LEGACY_SUPPORTING_IDS_PATTERN.search(text)
        if legacy_match:
            ids = _parse_int_tokens(legacy_match.group(1))
            mapped_lines = [mapping[event_id] for event_id in ids if event_id in mapping]
            if mapped_lines:
                converted += 1
                for line in mapped_lines:
                    if line in seen:
                        continue
                    seen.add(line)
                    rewritten.append(line)
            else:
                unresolved += 1
                if text not in seen:
                    seen.add(text)
                    rewritten.append(text)
            continue

        event_id = _extract_event_id(text)
        if event_id is not None and event_id in mapping:
            canonical_line = mapping[event_id]
            if canonical_line not in seen:
                seen.add(canonical_line)
                rewritten.append(canonical_line)
            continue

        unchanged += 1
        if text not in seen:
            seen.add(text)
            rewritten.append(text)

    return rewritten, converted, unresolved, unchanged


def _rewrite_payload_critical_events(payload: Dict[str, Any], mapping: Dict[int, str]) -> Tuple[int, int, int, int]:
    blocks = 0
    converted = 0
    unresolved = 0
    unchanged = 0

    critical_events_memory = payload.get("critical_events_memory")
    if not isinstance(critical_events_memory, list):
        return blocks, converted, unresolved, unchanged

    for block in critical_events_memory:
        if not isinstance(block, dict):
            continue
        new_lines, item_converted, item_unresolved, item_unchanged = _rewrite_critical_event_lines(
            block.get("critical_events", []),
            mapping,
        )
        block["critical_events"] = new_lines
        blocks += 1
        converted += item_converted
        unresolved += item_unresolved
        unchanged += item_unchanged

    return blocks, converted, unresolved, unchanged


def _ensure_backup(path: Path, backup_suffix: str) -> None:
    backup_path = path.with_name(path.name + backup_suffix)
    if backup_path.exists():
        return
    shutil.copy2(path, backup_path)


def _load_patient_key_filter(patient_stay_ids_path: Optional[str]) -> Optional[Set[Tuple[int, int]]]:
    if not patient_stay_ids_path:
        return None

    key_set: Set[Tuple[int, int]] = set()
    csv_path = Path(patient_stay_ids_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Patient-stay IDs CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        required = {"subject_id", "icu_stay_id"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV must include columns {sorted(required)}: {csv_path}")
        for row in reader:
            try:
                subject_id = int(str(row.get("subject_id", "")).strip())
                icu_stay_id = int(str(row.get("icu_stay_id", "")).strip())
            except ValueError:
                continue
            key_set.add((subject_id, icu_stay_id))

    if not key_set:
        raise ValueError(f"No valid subject_id/icu_stay_id rows found in {csv_path}")
    return key_set


def _patient_key_from_dir_name(patient_dir_name: str) -> Optional[Tuple[int, int]]:
    parts = patient_dir_name.split("_", 1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def rewrite_patient_memory(
    *,
    patient_dir: Path,
    dry_run: bool,
    backup_suffix: str,
) -> PatientRewriteStats:
    final_path = patient_dir / "final_memory.json"
    memory_db_path = patient_dir / "memory_database.json"
    if not final_path.exists() or not memory_db_path.exists():
        return PatientRewriteStats(patient_id=patient_dir.name, skipped=True, reason="missing_memory_files")

    final_payload = _read_json_object(final_path)
    memory_db_payload = _read_json_object(memory_db_path)

    event_line_map: Dict[int, str] = {}
    snapshots = memory_db_payload.get("memory_snapshots")
    if isinstance(snapshots, list):
        for snapshot in snapshots:
            if isinstance(snapshot, dict):
                _ingest_payload_event_lines(event_line_map, snapshot)
    _ingest_payload_event_lines(event_line_map, final_payload)

    snapshot_blocks = 0
    snapshot_converted = 0
    snapshot_unresolved = 0
    snapshot_unchanged = 0
    if isinstance(snapshots, list):
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            blocks, converted, unresolved, unchanged = _rewrite_payload_critical_events(snapshot, event_line_map)
            snapshot_blocks += blocks
            snapshot_converted += converted
            snapshot_unresolved += unresolved
            snapshot_unchanged += unchanged

    final_blocks, final_converted, final_unresolved, final_unchanged = _rewrite_payload_critical_events(
        final_payload,
        event_line_map,
    )

    if not dry_run:
        _ensure_backup(final_path, backup_suffix)
        _ensure_backup(memory_db_path, backup_suffix)
        _write_json(final_path, final_payload)
        _write_json(memory_db_path, memory_db_payload)

    return PatientRewriteStats(
        patient_id=patient_dir.name,
        skipped=False,
        event_map_size=len(event_line_map),
        snapshot_blocks=snapshot_blocks,
        snapshot_items_converted=snapshot_converted,
        snapshot_items_unresolved=snapshot_unresolved,
        snapshot_items_unchanged=snapshot_unchanged,
        final_blocks=final_blocks,
        final_items_converted=final_converted,
        final_items_unresolved=final_unresolved,
        final_items_unchanged=final_unchanged,
    )


def rewrite_memory_run(
    *,
    memory_run: str,
    patient_stay_ids_path: Optional[str],
    dry_run: bool,
    backup_suffix: str,
) -> Dict[str, Any]:
    run_dir = resolve_memory_run_dir(memory_run)
    patients_dir = run_dir / "patients"
    if not patients_dir.exists():
        raise FileNotFoundError(f"Missing patients directory: {patients_dir}")

    key_filter = _load_patient_key_filter(patient_stay_ids_path)
    all_patient_dirs = sorted(item for item in patients_dir.iterdir() if item.is_dir())
    selected_dirs: List[Path] = []
    for patient_dir in all_patient_dirs:
        if key_filter is None:
            selected_dirs.append(patient_dir)
            continue
        key = _patient_key_from_dir_name(patient_dir.name)
        if key is None:
            continue
        if key in key_filter:
            selected_dirs.append(patient_dir)

    if not selected_dirs:
        raise ValueError("No patient directories selected for rewrite.")

    stats: List[PatientRewriteStats] = []
    for index, patient_dir in enumerate(selected_dirs, 1):
        patient_stats = rewrite_patient_memory(
            patient_dir=patient_dir,
            dry_run=dry_run,
            backup_suffix=backup_suffix,
        )
        stats.append(patient_stats)
        if index % 25 == 0 or index == len(selected_dirs):
            print(f"processed {index}/{len(selected_dirs)}")

    rewritten = [item for item in stats if not item.skipped]
    skipped = [item for item in stats if item.skipped]

    summary = {
        "memory_run_dir": str(run_dir),
        "dry_run": dry_run,
        "backup_suffix": backup_suffix,
        "patients_total_selected": len(stats),
        "patients_rewritten": len(rewritten),
        "patients_skipped": len(skipped),
        "snapshot_blocks": sum(item.snapshot_blocks for item in rewritten),
        "snapshot_items_converted": sum(item.snapshot_items_converted for item in rewritten),
        "snapshot_items_unresolved": sum(item.snapshot_items_unresolved for item in rewritten),
        "snapshot_items_unchanged": sum(item.snapshot_items_unchanged for item in rewritten),
        "final_blocks": sum(item.final_blocks for item in rewritten),
        "final_items_converted": sum(item.final_items_converted for item in rewritten),
        "final_items_unresolved": sum(item.final_items_unresolved for item in rewritten),
        "final_items_unchanged": sum(item.final_items_unchanged for item in rewritten),
        "skipped_details": [item.to_dict() for item in skipped],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite MedEvo critical_events_memory into canonical event-line string format.",
    )
    parser.add_argument(
        "--memory-run",
        type=str,
        required=True,
        help="Memory run directory path, or a name under experiment_results/.",
    )
    parser.add_argument(
        "--patient-stay-ids",
        type=str,
        default=None,
        help="Optional CSV with columns subject_id,icu_stay_id to rewrite only selected patients.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview rewrite counts without writing files.",
    )
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=".bak_critical_event_rewrite",
        help="Suffix for one-time backup files before overwrite.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional output path for summary JSON. Defaults to <run_dir>/critical_event_rewrite_summary.json.",
    )
    args = parser.parse_args()

    summary = rewrite_memory_run(
        memory_run=args.memory_run,
        patient_stay_ids_path=args.patient_stay_ids,
        dry_run=bool(args.dry_run),
        backup_suffix=str(args.backup_suffix),
    )

    run_dir = resolve_memory_run_dir(args.memory_run)
    summary_path = (
        Path(str(args.summary_path))
        if args.summary_path is not None and str(args.summary_path).strip()
        else run_dir / "critical_event_rewrite_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
