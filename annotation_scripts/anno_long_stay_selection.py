#!/usr/bin/env python3
"""Sample long ICU stays with discharge summaries and export full-stay subset files."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation_scripts.anno_patient_selection import ShardPaths, _build_subset_data, _discover_shards
from annotation_scripts.anno_patient_selection import _sample_one_stay_per_subject_by_outcome
from utils.discharge_summary_selector import select_discharge_summaries_for_icu_stays


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample long ICU stays from all MIMIC-demo shards with selected discharge summaries "
            "and write parser-compatible full-stay parquet outputs."
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
        default=Path("data/mimic-demo/anno_subset_long_50"),
        help="Output directory for sampled metadata + subset parquets.",
    )
    parser.add_argument(
        "--target-stays",
        type=int,
        default=50,
        help="Target number of ICU stays to sample when outcome-specific counts are not provided.",
    )
    parser.add_argument(
        "--n-survived",
        type=int,
        default=None,
        help="Exact number of survived ICU stays to sample. Must be set together with --n-died.",
    )
    parser.add_argument(
        "--n-died",
        type=int,
        default=None,
        help="Exact number of died ICU stays to sample. Must be set together with --n-survived.",
    )
    parser.add_argument(
        "--min-icu-duration-hours",
        type=float,
        default=168.0,
        help="Strict lower bound for ICU duration filter (sampled stay must be > this value).",
    )
    parser.add_argument(
        "--max-days-after-leave",
        type=float,
        default=7.0,
        help="Discharge summary selector window in days after ICU leave_time.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Sampling seed for deterministic cohort selection.",
    )
    parser.add_argument(
        "--allow-multiple-stays-per-subject",
        action="store_true",
        help="Allow sampling more than one ICU stay from the same subject.",
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
    return parser.parse_args()


def _to_selected_summary_map(selection_df: pd.DataFrame) -> Dict[Tuple[int, int], Dict[str, object]]:
    selected_df = selection_df[selection_df["selected"] == True]  # noqa: E712
    result: Dict[Tuple[int, int], Dict[str, object]] = {}

    for _, row in selected_df.iterrows():
        if pd.isna(row.get("subject_id")) or pd.isna(row.get("icu_stay_id")):
            continue
        key = (int(row["subject_id"]), int(row["icu_stay_id"]))
        note_time = pd.to_datetime(row.get("selected_note_time"), errors="coerce")
        result[key] = {
            "selection_rule": row.get("selection_rule"),
            "time": note_time.isoformat() if pd.notna(note_time) else None,
            "delta_hours_after_leave": row.get("selected_note_delta_hours_after_leave"),
        }
    return result


def _normalize_icu_stays(icu_df: pd.DataFrame) -> pd.DataFrame:
    normalized = icu_df.copy()
    normalized["subject_id"] = pd.to_numeric(normalized["subject_id"], errors="coerce").astype("Int64")
    normalized["icu_stay_id"] = pd.to_numeric(normalized["icu_stay_id"], errors="coerce").astype("Int64")
    normalized["icu_duration_hours"] = pd.to_numeric(normalized["icu_duration_hours"], errors="coerce")
    return normalized


def _collect_candidates(
    shards: Sequence[ShardPaths],
    *,
    min_icu_duration_hours: float,
    max_days_after_leave: float,
) -> Tuple[pd.DataFrame, List[str]]:
    records: List[Dict[str, object]] = []
    icu_schema_columns: Optional[List[str]] = None

    for shard in shards:
        events_df = pd.read_parquet(shard.events_path)
        icu_df = pd.read_parquet(shard.icu_stay_path)

        if icu_schema_columns is None:
            icu_schema_columns = list(icu_df.columns)
        elif list(icu_df.columns) != icu_schema_columns:
            raise RuntimeError(
                f"Inconsistent ICU schema in shard {shard.shard_id}. "
                "All shards must share the same ICU parquet schema."
            )

        selection_df = select_discharge_summaries_for_icu_stays(
            events_df=events_df,
            icu_stay_df=icu_df,
            max_days_after_leave=max_days_after_leave,
        )
        selected_map = _to_selected_summary_map(selection_df)
        if not selected_map:
            continue

        normalized_icu_df = _normalize_icu_stays(icu_df)
        normalized_icu_df = normalized_icu_df[
            normalized_icu_df["subject_id"].notna()
            & normalized_icu_df["icu_stay_id"].notna()
            & normalized_icu_df["icu_duration_hours"].notna()
            & (normalized_icu_df["icu_duration_hours"] > float(min_icu_duration_hours))
        ]
        if len(normalized_icu_df) == 0:
            continue

        selected_pairs_df = pd.DataFrame(
            [{"subject_id": key[0], "icu_stay_id": key[1]} for key in selected_map.keys()]
        )
        selected_pairs_df["subject_id"] = pd.to_numeric(selected_pairs_df["subject_id"], errors="coerce").astype("Int64")
        selected_pairs_df["icu_stay_id"] = pd.to_numeric(selected_pairs_df["icu_stay_id"], errors="coerce").astype("Int64")

        eligible_df = normalized_icu_df.merge(selected_pairs_df, on=["subject_id", "icu_stay_id"], how="inner")
        if len(eligible_df) == 0:
            continue

        for row in eligible_df.to_dict("records"):
            key = (int(row["subject_id"]), int(row["icu_stay_id"]))
            summary_meta = selected_map[key]
            record = dict(row)
            record["source_shard"] = shard.shard_id
            record["source_events_path"] = str(shard.events_path)
            record["source_icu_stay_path"] = str(shard.icu_stay_path)
            record["discharge_selection_rule"] = summary_meta.get("selection_rule")
            record["selected_discharge_summary_time"] = summary_meta.get("time")
            record["selected_discharge_delta_hours_after_leave"] = summary_meta.get("delta_hours_after_leave")
            records.append(record)

    if not records:
        raise RuntimeError(
            "No eligible ICU stays found. Check duration threshold and discharge summary selection constraints."
        )
    if icu_schema_columns is None:
        raise RuntimeError("Failed to infer ICU schema columns from input ICU shards.")

    candidates_df = pd.DataFrame(records)
    candidates_df["survived"] = candidates_df["survived"].astype(bool)
    return candidates_df, icu_schema_columns


def _sample_candidates(
    candidates_df: pd.DataFrame,
    *,
    target_stays: int,
    n_survived: Optional[int],
    n_died: Optional[int],
    seed: int,
    allow_multiple_stays_per_subject: bool,
) -> pd.DataFrame:
    rng = random.Random(seed)
    use_outcome_targets = n_survived is not None or n_died is not None

    if use_outcome_targets:
        if n_survived is None or n_died is None:
            raise ValueError("--n-survived and --n-died must be provided together.")
        if n_survived < 0 or n_died < 0:
            raise ValueError("n_survived and n_died must be >= 0.")
        requested_total = int(n_survived) + int(n_died)
        if int(target_stays) != requested_total:
            raise ValueError(
                f"target_stays ({target_stays}) must equal n_survived + n_died ({requested_total})."
            )
        target_stays = requested_total

        if allow_multiple_stays_per_subject:
            survived_indices = candidates_df[candidates_df["survived"] == True].index.tolist()  # noqa: E712
            died_indices = candidates_df[candidates_df["survived"] == False].index.tolist()  # noqa: E712

            if len(survived_indices) < int(n_survived):
                raise RuntimeError(
                    f"Requested {n_survived} survived stays but only {len(survived_indices)} eligible stays available."
                )
            if len(died_indices) < int(n_died):
                raise RuntimeError(
                    f"Requested {n_died} died stays but only {len(died_indices)} eligible stays available."
                )

            rng.shuffle(survived_indices)
            rng.shuffle(died_indices)
            selected_indices = survived_indices[: int(n_survived)] + died_indices[: int(n_died)]
        else:
            selected_indices = []
            excluded_subject_ids: set = set()
            selected_indices.extend(
                _sample_one_stay_per_subject_by_outcome(
                    candidates_df,
                    survived=False,
                    n_required=int(n_died),
                    rng=rng,
                    excluded_subject_ids=excluded_subject_ids,
                )
            )
            selected_indices.extend(
                _sample_one_stay_per_subject_by_outcome(
                    candidates_df,
                    survived=True,
                    n_required=int(n_survived),
                    rng=rng,
                    excluded_subject_ids=excluded_subject_ids,
                )
            )
    else:
        if target_stays <= 0:
            raise ValueError("target_stays must be > 0")

        if allow_multiple_stays_per_subject:
            indices = candidates_df.index.tolist()
            if len(indices) < target_stays:
                raise RuntimeError(f"Requested {target_stays} stays but only {len(indices)} eligible stays available.")
            rng.shuffle(indices)
            selected_indices = indices[:target_stays]
        else:
            grouped = candidates_df.groupby("subject_id", sort=False)
            one_stay_per_subject_indices: List[int] = []
            for _, group in grouped:
                group_indices = group.index.tolist()
                rng.shuffle(group_indices)
                one_stay_per_subject_indices.append(int(group_indices[0]))

            if len(one_stay_per_subject_indices) < target_stays:
                raise RuntimeError(
                    f"Requested {target_stays} stays but only {len(one_stay_per_subject_indices)} "
                    "unique-subject stays available."
                )

            rng.shuffle(one_stay_per_subject_indices)
            selected_indices = one_stay_per_subject_indices[:target_stays]

    selected_df = candidates_df.loc[selected_indices].copy()
    selected_df = selected_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    selected_df["sample_order"] = range(len(selected_df))
    selected_df["outcome_label"] = selected_df["survived"].map(lambda value: "survived" if bool(value) else "died")
    selected_df["sampling_seed"] = int(seed)
    return selected_df


def _validate_subset(
    icu_subset_df: pd.DataFrame,
    *,
    expected_total: int,
    min_icu_duration_hours: float,
    enforce_unique_subject: bool,
) -> Dict[str, int]:
    observed_total = int(len(icu_subset_df))
    observed_unique_subjects = int(icu_subset_df["subject_id"].nunique())
    observed_survived = int(icu_subset_df["survived"].astype(bool).sum())
    observed_died = int((~icu_subset_df["survived"].astype(bool)).sum())

    if observed_total != expected_total:
        raise RuntimeError(f"Validation failed: expected {expected_total} stays but found {observed_total}.")
    if (icu_subset_df["icu_duration_hours"] <= float(min_icu_duration_hours)).any():
        raise RuntimeError(
            "Validation failed: subset contains ICU stays at or below the minimum duration threshold."
        )
    if enforce_unique_subject and observed_unique_subjects != expected_total:
        raise RuntimeError(
            "Validation failed: one-subject-one-stay constraint broken "
            f"(expected {expected_total} unique subjects, observed {observed_unique_subjects})."
        )

    return {
        "validated_total": observed_total,
        "validated_unique_subjects": observed_unique_subjects,
        "validated_survived": observed_survived,
        "validated_died": observed_died,
    }


def main() -> None:
    args = _parse_args()

    shards = _discover_shards(args.input_root)
    print(f"Discovered {len(shards)} shard(s) under {args.input_root}")

    candidates_df, icu_schema_columns = _collect_candidates(
        shards=shards,
        min_icu_duration_hours=float(args.min_icu_duration_hours),
        max_days_after_leave=float(args.max_days_after_leave),
    )

    selected_df = _sample_candidates(
        candidates_df,
        target_stays=int(args.target_stays),
        n_survived=args.n_survived,
        n_died=args.n_died,
        seed=int(args.seed),
        allow_multiple_stays_per_subject=bool(args.allow_multiple_stays_per_subject),
    )

    selected_total = int(len(selected_df))
    selected_survived = int(selected_df["survived"].astype(bool).sum())
    selected_died = int((~selected_df["survived"].astype(bool)).sum())
    selected_unique_subjects = int(selected_df["subject_id"].nunique())

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

    validation_stats = _validate_subset(
        icu_subset_df=icu_subset_df,
        expected_total=selected_total,
        min_icu_duration_hours=float(args.min_icu_duration_hours),
        enforce_unique_subject=not bool(args.allow_multiple_stays_per_subject),
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
        "target_stays": int(args.target_stays),
        "requested_n_survived": (None if args.n_survived is None else int(args.n_survived)),
        "requested_n_died": (None if args.n_died is None else int(args.n_died)),
        "min_icu_duration_hours_strict_gt": float(args.min_icu_duration_hours),
        "max_days_after_leave": float(args.max_days_after_leave),
        "allow_multiple_stays_per_subject": bool(args.allow_multiple_stays_per_subject),
        "eligible_total": int(len(candidates_df)),
        "eligible_survived": int(candidates_df["survived"].astype(bool).sum()),
        "eligible_died": int((~candidates_df["survived"].astype(bool)).sum()),
        "eligible_unique_subjects": int(candidates_df["subject_id"].nunique()),
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
