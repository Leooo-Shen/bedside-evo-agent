#!/usr/bin/env python3
"""Sample a balanced, parser-compatible ICU subset from sharded MIMIC data."""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_parser import MIMICDataParser


@dataclass(frozen=True)
class ShardPaths:
    shard_id: str
    events_path: Path
    icu_stay_path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced ICU subset from data/mimic-demo "
            "with one ICU stay per subject and parser-compatible parquet outputs."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/mimic-demo"),
        help="Root directory containing 'events/' and 'icu_stay/' shard parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mimic-demo/anno_subset_160"),
        help="Output directory for sampled metadata + subset parquets.",
    )
    parser.add_argument(
        "--n-survived",
        type=int,
        default=80,
        help="Number of survived patients to sample.",
    )
    parser.add_argument(
        "--n-died",
        type=int,
        default=80,
        help="Number of died patients to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Sampling seed for deterministic cohort selection.",
    )
    parser.add_argument(
        "--max-days-after-leave",
        type=float,
        default=7.0,
        help="Discharge summary selector window used by MIMICDataParser.",
    )
    parser.add_argument(
        "--metadata-name",
        type=str,
        default="sampled_patient_metadata.csv",
        help="Output metadata CSV filename.",
    )
    parser.add_argument(
        "--events-name",
        type=str,
        default="events.parquet",
        help="Output subset events parquet filename.",
    )
    parser.add_argument(
        "--icu-stay-name",
        type=str,
        default="icu_stay.parquet",
        help="Output subset ICU stay parquet filename.",
    )
    parser.add_argument(
        "--skip-output-validation",
        action="store_true",
        help="Skip parser.load_data() compatibility validation on the generated subset files.",
    )
    return parser.parse_args()


def _shard_sort_key(path: Path) -> Tuple[int, str]:
    token = path.stem.replace("data_", "")
    try:
        return (0, f"{int(token):06d}")
    except ValueError:
        return (1, token)


def _discover_shards(input_root: Path) -> List[ShardPaths]:
    events_dir = input_root / "events"
    icu_dir = input_root / "icu_stay"

    if not events_dir.exists():
        raise FileNotFoundError(f"Missing events directory: {events_dir}")
    if not icu_dir.exists():
        raise FileNotFoundError(f"Missing icu_stay directory: {icu_dir}")

    icu_files = sorted(icu_dir.glob("data_*.parquet"), key=_shard_sort_key)
    if not icu_files:
        raise FileNotFoundError(f"No ICU shard files found under: {icu_dir}")

    shards: List[ShardPaths] = []
    for icu_path in icu_files:
        shard_id = icu_path.stem.replace("data_", "")
        events_path = events_dir / icu_path.name
        if not events_path.exists():
            raise FileNotFoundError(f"Missing events shard for {icu_path.name}: {events_path}")
        shards.append(ShardPaths(shard_id=shard_id, events_path=events_path, icu_stay_path=icu_path))
    return shards


def _collect_candidates(
    shards: Sequence[ShardPaths],
    max_days_after_leave: float,
) -> Tuple[pd.DataFrame, List[str]]:
    records: List[Dict] = []
    icu_schema_columns: Optional[List[str]] = None

    for shard in shards:
        parser = MIMICDataParser(
            events_path=str(shard.events_path),
            icu_stay_path=str(shard.icu_stay_path),
            discharge_summary_max_days_after_leave=max_days_after_leave,
            require_discharge_summary_for_icu_stays=True,
        )
        parser.load_data()

        if parser.icu_stay_df is None:
            raise RuntimeError(f"Parser returned empty ICU dataframe for shard {shard.shard_id}")
        if parser.events_df is None:
            raise RuntimeError(f"Parser returned empty events dataframe for shard {shard.shard_id}")

        shard_icu_df = parser.icu_stay_df.copy()
        if icu_schema_columns is None:
            icu_schema_columns = list(shard_icu_df.columns)
        elif list(shard_icu_df.columns) != icu_schema_columns:
            raise RuntimeError(
                f"Inconsistent ICU schema in shard {shard.shard_id}. "
                "All shards must share the same ICU parquet schema."
            )

        selected_map = dict(parser._selected_discharge_summary_map)
        for record in shard_icu_df.to_dict("records"):
            subject_id = int(record["subject_id"])
            icu_stay_id = int(record["icu_stay_id"])
            summary_meta = selected_map.get((subject_id, icu_stay_id), {})

            row = dict(record)
            row["source_shard"] = shard.shard_id
            row["source_events_path"] = str(shard.events_path)
            row["source_icu_stay_path"] = str(shard.icu_stay_path)
            row["discharge_selection_rule"] = summary_meta.get("selection_rule")
            row["selected_discharge_summary_time"] = summary_meta.get("time")
            row["selected_discharge_delta_hours_after_leave"] = summary_meta.get("delta_hours_after_leave")
            records.append(row)

        del parser
        gc.collect()

    if not records:
        raise RuntimeError("No eligible ICU stays were found after parser filtering.")
    if icu_schema_columns is None:
        raise RuntimeError("Failed to infer ICU schema columns from eligible stays.")

    candidates_df = pd.DataFrame(records)
    candidates_df["survived"] = candidates_df["survived"].astype(bool)
    return candidates_df, icu_schema_columns


def _sample_one_stay_per_subject_by_outcome(
    candidates_df: pd.DataFrame,
    *,
    survived: bool,
    n_required: int,
    rng: random.Random,
    excluded_subject_ids: set,
) -> List[int]:
    outcome_df = candidates_df[candidates_df["survived"] == bool(survived)]
    grouped = outcome_df.groupby("subject_id", sort=False)

    subject_ids = [int(subject_id) for subject_id in grouped.groups.keys()]
    rng.shuffle(subject_ids)

    picked_indices: List[int] = []
    for subject_id in subject_ids:
        if subject_id in excluded_subject_ids:
            continue

        candidate_indices = grouped.get_group(subject_id).index.tolist()
        rng.shuffle(candidate_indices)
        picked_indices.append(int(candidate_indices[0]))
        excluded_subject_ids.add(subject_id)

        if len(picked_indices) >= int(n_required):
            break

    if len(picked_indices) < int(n_required):
        requested_outcome = "survived" if survived else "died"
        raise RuntimeError(
            f"Unable to sample {n_required} unique-subject ICU stays for outcome={requested_outcome}. "
            f"Only {len(picked_indices)} available after one-subject-one-stay constraints."
        )

    return picked_indices


def _sample_balanced_cohort(
    candidates_df: pd.DataFrame,
    *,
    n_survived: int,
    n_died: int,
    seed: int,
) -> pd.DataFrame:
    if n_survived < 0 or n_died < 0:
        raise ValueError("n_survived and n_died must be >= 0")

    rng = random.Random(seed)
    selected_indices: List[int] = []
    selected_subject_ids: set = set()

    # Sample died first since this pool is typically smaller.
    selected_indices.extend(
        _sample_one_stay_per_subject_by_outcome(
            candidates_df,
            survived=False,
            n_required=n_died,
            rng=rng,
            excluded_subject_ids=selected_subject_ids,
        )
    )
    selected_indices.extend(
        _sample_one_stay_per_subject_by_outcome(
            candidates_df,
            survived=True,
            n_required=n_survived,
            rng=rng,
            excluded_subject_ids=selected_subject_ids,
        )
    )

    selected_df = candidates_df.loc[selected_indices].copy()
    selected_df = selected_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    selected_df["sample_order"] = range(len(selected_df))
    selected_df["outcome_label"] = selected_df["survived"].map(lambda value: "survived" if bool(value) else "died")

    return selected_df


def _extract_stay_events(
    events_df: pd.DataFrame,
    *,
    subject_id: int,
    min_event_idx: int,
    max_event_idx: int,
) -> pd.DataFrame:
    if min_event_idx > max_event_idx:
        raise ValueError(
            f"Invalid event_idx bounds for subject_id={subject_id}: "
            f"min_event_idx={min_event_idx} > max_event_idx={max_event_idx}"
        )
    if "subject_id" not in events_df.columns:
        raise ValueError("events_df is missing required column: subject_id")
    if "event_idx" not in events_df.columns:
        raise ValueError("events_df is missing required column: event_idx")

    subject_events = events_df[events_df["subject_id"] == int(subject_id)].copy()
    event_idx_series = pd.to_numeric(subject_events["event_idx"], errors="coerce")
    stay_events = subject_events[(event_idx_series >= int(min_event_idx)) & (event_idx_series <= int(max_event_idx))]
    return stay_events.sort_values("event_idx", ascending=True, kind="mergesort").copy()


def _coerce_icu_dtypes(icu_df: pd.DataFrame) -> pd.DataFrame:
    datetime_cols = ["enter_time", "leave_time", "death_time", "birth_time", "readm_time"]
    for col in datetime_cols:
        if col in icu_df.columns:
            icu_df[col] = pd.to_datetime(icu_df[col], errors="coerce")

    if "survived" in icu_df.columns:
        icu_df["survived"] = icu_df["survived"].astype(bool)

    strict_int_cols = ["subject_id", "icu_stay_id", "min_event_idx", "max_event_idx", "n_events"]
    for col in strict_int_cols:
        if col in icu_df.columns:
            icu_df[col] = pd.to_numeric(icu_df[col], errors="raise").astype("int64")

    nullable_int_cols = ["enter_event_idx", "death_event_idx", "birth_event_idx", "readm_event_idx", "shard_idx"]
    for col in nullable_int_cols:
        if col in icu_df.columns:
            icu_df[col] = pd.to_numeric(icu_df[col], errors="coerce").astype("Int64")

    numeric_cols = ["icu_duration_hours", "readm_duration_hours"]
    for col in numeric_cols:
        if col in icu_df.columns:
            icu_df[col] = pd.to_numeric(icu_df[col], errors="coerce")

    return icu_df


def _build_subset_data(
    selected_df: pd.DataFrame,
    *,
    icu_schema_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_df = selected_df.sort_values(["source_shard", "sample_order"]).reset_index(drop=True)

    sampled_at_utc = datetime.now(timezone.utc).isoformat()

    event_frames: List[pd.DataFrame] = []
    icu_rows: List[Dict] = []
    metadata_rows: List[Dict] = []
    events_schema_columns: Optional[List[str]] = None

    for source_shard, shard_group in selected_df.groupby("source_shard", sort=False):
        events_path = Path(str(shard_group["source_events_path"].iloc[0]))
        shard_events_df = pd.read_parquet(events_path)
        if events_schema_columns is None:
            events_schema_columns = list(shard_events_df.columns)
        elif list(shard_events_df.columns) != events_schema_columns:
            raise RuntimeError(
                f"Inconsistent events schema in shard {source_shard}. "
                "All shards must share the same events parquet schema."
            )

        for _, row in shard_group.sort_values("sample_order").iterrows():
            subject_id = int(row["subject_id"])
            icu_stay_id = int(row["icu_stay_id"])
            src_min_event_idx = int(row["min_event_idx"])
            src_max_event_idx = int(row["max_event_idx"])

            stay_events = _extract_stay_events(
                shard_events_df,
                subject_id=subject_id,
                min_event_idx=src_min_event_idx,
                max_event_idx=src_max_event_idx,
            )
            if len(stay_events) == 0:
                raise RuntimeError(
                    "No events found for sampled stay "
                    f"subject_id={subject_id}, icu_stay_id={icu_stay_id}, shard={source_shard}"
                )

            subset_min_event_idx = int(src_min_event_idx)
            subset_max_event_idx = int(src_max_event_idx)

            event_frames.append(stay_events[events_schema_columns].reset_index(drop=True))

            icu_row = {column: row[column] for column in icu_schema_columns}
            icu_row["min_event_idx"] = subset_min_event_idx
            icu_row["max_event_idx"] = subset_max_event_idx
            icu_row["n_events"] = int(len(stay_events))
            icu_rows.append(icu_row)

            metadata_rows.append(
                {
                    "sample_order": int(row["sample_order"]),
                    "subject_id": subject_id,
                    "icu_stay_id": icu_stay_id,
                    "survived": bool(row["survived"]),
                    "outcome_label": row["outcome_label"],
                    "source_shard": str(row["source_shard"]),
                    "source_events_path": str(row["source_events_path"]),
                    "source_icu_stay_path": str(row["source_icu_stay_path"]),
                    "split": row.get("split"),
                    "shard_idx": row.get("shard_idx"),
                    "enter_time": row.get("enter_time"),
                    "leave_time": row.get("leave_time"),
                    "icu_duration_hours": row.get("icu_duration_hours"),
                    "death_time": row.get("death_time"),
                    "readm_time": row.get("readm_time"),
                    "source_enter_event_idx": row.get("enter_event_idx"),
                    "source_death_event_idx": row.get("death_event_idx"),
                    "source_readm_event_idx": row.get("readm_event_idx"),
                    "source_min_event_idx": src_min_event_idx,
                    "source_max_event_idx": src_max_event_idx,
                    "source_n_events": int(row["n_events"]),
                    "subset_enter_event_idx": icu_row.get("enter_event_idx"),
                    "subset_death_event_idx": icu_row.get("death_event_idx"),
                    "subset_readm_event_idx": icu_row.get("readm_event_idx"),
                    "subset_min_event_idx": subset_min_event_idx,
                    "subset_max_event_idx": subset_max_event_idx,
                    "subset_n_events": int(len(stay_events)),
                    "discharge_selection_rule": row.get("discharge_selection_rule"),
                    "selected_discharge_summary_time": row.get("selected_discharge_summary_time"),
                    "selected_discharge_delta_hours_after_leave": row.get(
                        "selected_discharge_delta_hours_after_leave"
                    ),
                    "sampling_seed": int(row.get("sampling_seed", -1)),
                    "sampled_at_utc": sampled_at_utc,
                }
            )

        del shard_events_df
        gc.collect()

    if not event_frames:
        raise RuntimeError("No event frames were assembled for sampled cohort.")
    if not icu_rows:
        raise RuntimeError("No ICU stay rows were assembled for sampled cohort.")

    events_subset_df = pd.concat(event_frames, ignore_index=True)
    events_subset_df = events_subset_df[events_schema_columns]

    icu_subset_df = pd.DataFrame(icu_rows)
    icu_subset_df = icu_subset_df[list(icu_schema_columns)]
    icu_subset_df = _coerce_icu_dtypes(icu_subset_df)

    metadata_df = pd.DataFrame(metadata_rows).sort_values("sample_order").reset_index(drop=True)
    return events_subset_df, icu_subset_df, metadata_df


def _validate_output(
    *,
    events_path: Path,
    icu_stay_path: Path,
    expected_total: int,
    expected_survived: int,
    expected_died: int,
    max_days_after_leave: float,
) -> Dict[str, int]:
    parser = MIMICDataParser(
        events_path=str(events_path),
        icu_stay_path=str(icu_stay_path),
        discharge_summary_max_days_after_leave=max_days_after_leave,
        require_discharge_summary_for_icu_stays=True,
    )
    parser.load_data()
    if parser.icu_stay_df is None:
        raise RuntimeError("Validation parser returned empty ICU dataframe for generated subset.")

    observed_total = int(len(parser.icu_stay_df))
    observed_survived = int(parser.icu_stay_df["survived"].astype(bool).sum())
    observed_died = int((~parser.icu_stay_df["survived"].astype(bool)).sum())
    observed_unique_subjects = int(parser.icu_stay_df["subject_id"].nunique())

    if observed_total != expected_total:
        raise RuntimeError(
            f"Validation failed: expected {expected_total} stays but parser loaded {observed_total} from subset."
        )
    if observed_survived != expected_survived or observed_died != expected_died:
        raise RuntimeError(
            "Validation failed: outcome balance mismatch after parser reload "
            f"(expected survived={expected_survived}, died={expected_died}; "
            f"observed survived={observed_survived}, died={observed_died})."
        )
    if observed_unique_subjects != expected_total:
        raise RuntimeError(
            "Validation failed: one-subject-one-stay constraint broken in output subset "
            f"(expected {expected_total} unique subjects, observed {observed_unique_subjects})."
        )

    return {
        "validated_total": observed_total,
        "validated_survived": observed_survived,
        "validated_died": observed_died,
        "validated_unique_subjects": observed_unique_subjects,
    }


def _compute_subject_eligibility_counts(candidates_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    grouped = candidates_df.groupby("subject_id", sort=False)
    for subject_id, group in grouped:
        survived_mask = group["survived"].astype(bool)
        rows.append(
            {
                "subject_id": int(subject_id),
                "eligible_stay_count_for_subject": int(len(group)),
                "eligible_survived_stay_count_for_subject": int(survived_mask.sum()),
                "eligible_died_stay_count_for_subject": int((~survived_mask).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()

    shards = _discover_shards(args.input_root)
    print(f"Discovered {len(shards)} shard(s) under {args.input_root}")

    candidates_df, icu_schema_columns = _collect_candidates(
        shards=shards,
        max_days_after_leave=float(args.max_days_after_leave),
    )

    subject_eligibility_df = _compute_subject_eligibility_counts(candidates_df)
    candidates_df = candidates_df.merge(subject_eligibility_df, on="subject_id", how="left")

    survived_candidates = candidates_df[candidates_df["survived"] == True]  # noqa: E712
    died_candidates = candidates_df[candidates_df["survived"] == False]  # noqa: E712
    print(
        "Eligible stays after parser filters: "
        f"{len(candidates_df)} total "
        f"(survived={len(survived_candidates)}, died={len(died_candidates)})"
    )
    print(
        "Eligible subjects by outcome availability: "
        f"survived_subjects={survived_candidates['subject_id'].nunique()}, "
        f"died_subjects={died_candidates['subject_id'].nunique()}"
    )

    selected_df = _sample_balanced_cohort(
        candidates_df,
        n_survived=int(args.n_survived),
        n_died=int(args.n_died),
        seed=int(args.seed),
    )
    selected_df["sampling_seed"] = int(args.seed)

    selected_total = int(len(selected_df))
    selected_survived = int(selected_df["survived"].astype(bool).sum())
    selected_died = int((~selected_df["survived"].astype(bool)).sum())
    selected_unique_subjects = int(selected_df["subject_id"].nunique())
    print(
        "Sampled cohort: "
        f"{selected_total} stays, {selected_unique_subjects} unique subjects, "
        f"survived={selected_survived}, died={selected_died}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events_output_path = output_dir / args.events_name
    icu_output_path = output_dir / args.icu_stay_name
    metadata_output_path = output_dir / args.metadata_name
    summary_output_path = output_dir / "selection_summary.json"

    events_subset_df, icu_subset_df, metadata_df = _build_subset_data(
        selected_df=selected_df,
        icu_schema_columns=icu_schema_columns,
    )

    events_subset_df.to_parquet(events_output_path, index=False)
    icu_subset_df.to_parquet(icu_output_path, index=False)
    metadata_df.to_csv(metadata_output_path, index=False)

    validation_stats = {}
    if not args.skip_output_validation:
        validation_stats = _validate_output(
            events_path=events_output_path,
            icu_stay_path=icu_output_path,
            expected_total=selected_total,
            expected_survived=selected_survived,
            expected_died=selected_died,
            max_days_after_leave=float(args.max_days_after_leave),
        )

    shard_outcome_distribution = (
        selected_df.groupby(["source_shard", "survived"]).size().reset_index(name="count").to_dict("records")
    )

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(args.input_root),
        "output_dir": str(output_dir),
        "events_output_path": str(events_output_path),
        "icu_stay_output_path": str(icu_output_path),
        "metadata_output_path": str(metadata_output_path),
        "selection_seed": int(args.seed),
        "requested_n_survived": int(args.n_survived),
        "requested_n_died": int(args.n_died),
        "selected_total": selected_total,
        "selected_survived": selected_survived,
        "selected_died": selected_died,
        "selected_unique_subjects": selected_unique_subjects,
        "selected_total_events": int(len(events_subset_df)),
        "selected_total_icu_rows": int(len(icu_subset_df)),
        "source_shards": [shard.shard_id for shard in shards],
        "shard_outcome_distribution": shard_outcome_distribution,
        "output_validation": validation_stats,
    }
    with open(summary_output_path, "w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2, ensure_ascii=False)

    print("Wrote outputs:")
    print(f"  - Metadata CSV: {metadata_output_path}")
    print(f"  - Events Parquet: {events_output_path}")
    print(f"  - ICU Stay Parquet: {icu_output_path}")
    print(f"  - Summary JSON: {summary_output_path}")


if __name__ == "__main__":
    main()
