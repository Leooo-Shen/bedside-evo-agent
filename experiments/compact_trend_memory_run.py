from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.create_memory import (
    AGGREGATE_FILENAME,
    RUN_CONFIG_FILENAME,
    compact_final_memory_trend_memory,
    compact_trend_memory_snapshot,
    resolve_memory_run_dir,
)


def _iter_memory_snapshots(path: Path) -> Iterator[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    key = '"memory_snapshots"'
    with open(path, "r") as f:
        buffer = ""
        idx = 0
        eof = False

        def read_more() -> bool:
            nonlocal buffer, eof
            chunk = f.read(1024 * 1024)
            if chunk:
                buffer += chunk
                return True
            eof = True
            return False

        while True:
            key_pos = buffer.find(key)
            if key_pos >= 0:
                bracket_pos = buffer.find("[", key_pos + len(key))
                if bracket_pos >= 0:
                    idx = bracket_pos + 1
                    break
            if eof:
                raise ValueError(f"Invalid memory database format: {path}")
            if not read_more():
                raise ValueError(f"Invalid memory database format: {path}")

        while True:
            while True:
                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1
                if idx < len(buffer):
                    break
                if eof:
                    raise ValueError(f"Unexpected EOF while reading snapshots: {path}")
                read_more()

            ch = buffer[idx]
            if ch == "]":
                break
            if ch == ",":
                idx += 1
                continue

            while True:
                try:
                    snapshot, end_idx = decoder.raw_decode(buffer, idx)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    read_more()

            if not isinstance(snapshot, dict):
                raise ValueError(f"Snapshot entry must be JSON object in {path}")
            yield snapshot
            idx = end_idx

            if idx > 2 * 1024 * 1024:
                buffer = buffer[idx:]
                idx = 0


def _write_transformed_memory_database(src_path: Path, dst_path: Path) -> int:
    count = 0
    with open(dst_path, "w") as out_f:
        out_f.write('{"memory_snapshots":[\n')
        first = True
        for snapshot in _iter_memory_snapshots(src_path):
            transformed = compact_trend_memory_snapshot(snapshot)
            if not first:
                out_f.write(",\n")
            json.dump(transformed, out_f, ensure_ascii=False, separators=(",", ":"))
            first = False
            count += 1
        out_f.write("\n]}\n")
    return count


def _copy_top_level_files(src_dir: Path, dst_dir: Path) -> None:
    for item in src_dir.iterdir():
        if not item.is_file():
            continue
        shutil.copy2(item, dst_dir / item.name)


def _rewrite_aggregate_and_run_config(dst_dir: Path, source_dir: Path) -> None:
    aggregate_path = dst_dir / AGGREGATE_FILENAME
    if aggregate_path.exists():
        with open(aggregate_path, "r") as f:
            aggregate_payload = json.load(f)
        if isinstance(aggregate_payload, dict):
            aggregate_payload["results_dir"] = str(dst_dir)
            aggregate_payload["source_results_dir"] = str(source_dir)
            aggregate_payload["trend_memory_policy"] = "current_window_plus_global_per_snapshot"
            with open(aggregate_path, "w") as f:
                json.dump(aggregate_payload, f, indent=2)

    run_config_path = dst_dir / RUN_CONFIG_FILENAME
    if run_config_path.exists():
        with open(run_config_path, "r") as f:
            run_config_payload = json.load(f)
        if isinstance(run_config_payload, dict):
            run_config_payload["source_run_dir"] = str(source_dir)
            run_config_payload["trend_memory_policy"] = "current_window_plus_global_per_snapshot"
            with open(run_config_path, "w") as f:
                json.dump(run_config_payload, f, indent=2)


def _transform_patient_dir(src_patient_dir: Path, dst_patient_dir: Path) -> int:
    dst_patient_dir.mkdir(parents=True, exist_ok=True)

    src_memory_db_path = src_patient_dir / "memory_database.json"
    src_final_memory_path = src_patient_dir / "final_memory.json"
    if not src_memory_db_path.exists():
        raise FileNotFoundError(f"Missing memory_database.json: {src_memory_db_path}")
    if not src_final_memory_path.exists():
        raise FileNotFoundError(f"Missing final_memory.json: {src_final_memory_path}")

    snapshot_count = _write_transformed_memory_database(
        src_path=src_memory_db_path,
        dst_path=dst_patient_dir / "memory_database.json",
    )

    with open(src_final_memory_path, "r") as f:
        final_memory_payload = json.load(f)
    if not isinstance(final_memory_payload, dict):
        raise ValueError(f"final_memory.json must be object: {src_final_memory_path}")
    transformed_final_memory = compact_final_memory_trend_memory(final_memory_payload)
    with open(dst_patient_dir / "final_memory.json", "w") as f:
        json.dump(transformed_final_memory, f, indent=2)

    for item in src_patient_dir.iterdir():
        if item.name in {"memory_database.json", "final_memory.json"}:
            continue
        target = dst_patient_dir / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target)

    return snapshot_count


def compact_memory_run(source_run: str, output_run: str) -> Dict[str, Any]:
    source_dir = resolve_memory_run_dir(source_run)
    output_dir = Path(str(output_run).strip())
    if not str(output_dir):
        raise ValueError("output_run must be non-empty")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=False)
    _copy_top_level_files(source_dir, output_dir)

    source_patients_dir = source_dir / "patients"
    if not source_patients_dir.exists():
        raise FileNotFoundError(f"Missing patients directory: {source_patients_dir}")
    output_patients_dir = output_dir / "patients"
    output_patients_dir.mkdir(parents=True, exist_ok=False)

    patient_dirs = [item for item in sorted(source_patients_dir.iterdir()) if item.is_dir()]
    transformed_patients = 0
    transformed_snapshots = 0

    for idx, patient_dir in enumerate(patient_dirs, 1):
        patient_id = patient_dir.name
        print(f"[{idx}/{len(patient_dirs)}] Transforming {patient_id}")
        snapshot_count = _transform_patient_dir(
            src_patient_dir=patient_dir,
            dst_patient_dir=output_patients_dir / patient_id,
        )
        transformed_patients += 1
        transformed_snapshots += snapshot_count

    _rewrite_aggregate_and_run_config(output_dir, source_dir)

    result = {
        "source_run": str(source_dir),
        "output_run": str(output_dir),
        "transformed_patients": transformed_patients,
        "transformed_snapshots": transformed_snapshots,
        "trend_memory_policy": "current_window_plus_global_per_snapshot",
    }

    with open(output_dir / "trend_memory_compaction_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compact memory run trend_memory to current+global entries")
    parser.add_argument(
        "--source-run",
        type=str,
        required=True,
        help="Source memory run directory (absolute path or experiment_results/<run_name>).",
    )
    parser.add_argument(
        "--output-run",
        type=str,
        required=True,
        help="New output directory path for transformed memory run.",
    )
    args = parser.parse_args()

    summary = compact_memory_run(source_run=args.source_run, output_run=args.output_run)
    print("\nDone")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
