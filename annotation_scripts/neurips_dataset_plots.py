#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NATURE_PALETTE = [
    "#264653",
    "#2A9D8F",
    "#8AB17D",
    "#E9C46A",
    "#F4A261",
    "#E76F51",
    "#457B9D",
    "#84A59D",
    "#B56576",
    "#6D597A",
]
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10
STANDARD_FIGSIZE = (7.2, 5.4)
WIDE_FIGSIZE = (8.6, 5.4)
SQUARE_FIGSIZE = (7.6, 7.2)
EDGE_COLOR = "#FFFFFF"
TEXT_COLOR = "#1F2933"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible NeurIPS-style cohort statistics tables and "
            "publication-ready individual figures."
        )
    )
    parser.add_argument(
        "--subset-dir",
        type=Path,
        required=True,
        help="Merged subset directory containing icu_stay.parquet, events.parquet, and sampled_patient_metadata.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <subset-dir>/statistics/neurips_benchmark_stats",
    )
    parser.add_argument(
        "--pie-min-percentage",
        type=float,
        default=5.0,
        help="Merge categories below this percentage into Others for pie charts.",
    )
    return parser.parse_args()


def normalize_icd(code: Any) -> str:
    if pd.isna(code):
        return ""
    return "".join(ch for ch in str(code).upper() if ch.isalnum())


def icd9_chapter_index(code: str) -> int | None:
    digits = "".join(ch for ch in code if ch.isdigit())
    if len(digits) < 3:
        return None
    return int(digits[:3])


def disease_category(system: Any, code: Any) -> str:
    clean = normalize_icd(code)
    coding = str(system).upper()
    if not clean:
        return "Uncategorized"
    if coding == "ICD10":
        if clean.startswith(("A40", "A41", "R652", "R572", "T8144", "O85")):
            return "Sepsis & Severe Infection"
        letter = clean[0]
        if letter in ("A", "B"):
            return "Infectious Diseases"
        if letter == "C" or clean.startswith(("D0", "D1", "D2", "D3", "D4")):
            return "Oncology"
        if letter == "D":
            return "Hematologic Disorders"
        if letter == "E":
            return "Endocrine/Metabolic Disorders"
        if letter == "F":
            return "Psychiatric Disorders"
        if letter == "G":
            return "Neurological Disorders"
        if letter == "H":
            return "Sensory Disorders"
        if letter == "I":
            return "Cardiovascular Disorders"
        if letter == "J":
            return "Respiratory Disorders"
        if letter == "K":
            return "Gastrointestinal/Hepatic Disorders"
        if letter == "L":
            return "Dermatologic Disorders"
        if letter == "M":
            return "Musculoskeletal Disorders"
        if letter == "N":
            return "Renal/Genitourinary Disorders"
        if letter == "O":
            return "Pregnancy-Related Disorders"
        if letter == "P":
            return "Perinatal Disorders"
        if letter == "Q":
            return "Congenital Disorders"
        if letter == "R":
            return "Clinical Signs/Symptoms"
        if letter in ("S", "T"):
            return "Injury/Poisoning"
        if letter in ("V", "W", "X", "Y"):
            return "External Causes"
        if letter == "Z":
            return "Health Status Factors"
        return "Uncategorized"
    if coding == "ICD9":
        if clean.startswith(("038", "99591", "99592", "78552")):
            return "Sepsis & Severe Infection"
        if clean.startswith("V"):
            return "Health Status Factors"
        if clean.startswith("E"):
            return "External Causes"
        chapter = icd9_chapter_index(clean)
        if chapter is None:
            return "Uncategorized"
        if 1 <= chapter <= 139:
            return "Infectious Diseases"
        if 140 <= chapter <= 239:
            return "Oncology"
        if 240 <= chapter <= 279:
            return "Endocrine/Metabolic Disorders"
        if 280 <= chapter <= 289:
            return "Hematologic Disorders"
        if 290 <= chapter <= 319:
            return "Psychiatric Disorders"
        if 320 <= chapter <= 389:
            return "Neurological Disorders"
        if 390 <= chapter <= 459:
            return "Cardiovascular Disorders"
        if 460 <= chapter <= 519:
            return "Respiratory Disorders"
        if 520 <= chapter <= 579:
            return "Gastrointestinal/Hepatic Disorders"
        if 580 <= chapter <= 629:
            return "Renal/Genitourinary Disorders"
        if 630 <= chapter <= 679:
            return "Pregnancy-Related Disorders"
        if 680 <= chapter <= 709:
            return "Dermatologic Disorders"
        if 710 <= chapter <= 739:
            return "Musculoskeletal Disorders"
        if 740 <= chapter <= 759:
            return "Congenital Disorders"
        if 760 <= chapter <= 779:
            return "Perinatal Disorders"
        if 780 <= chapter <= 799:
            return "Clinical Signs/Symptoms"
        if 800 <= chapter <= 999:
            return "Injury/Poisoning"
        return "Uncategorized"
    return "Uncategorized"


def load_diagnoses(metadata: pd.DataFrame) -> pd.DataFrame:
    selected_subjects = metadata["subject_id"].dropna().astype(int).unique().tolist()
    source_event_paths = sorted(metadata["source_events_path"].dropna().unique().tolist())
    diagnosis_frames = []
    for path in source_event_paths:
        diagnosis_frames.append(
            pd.read_parquet(
                path,
                columns=[
                    "subject_id",
                    "hosp_stay_id",
                    "time",
                    "table",
                    "orig_coding_system",
                    "orig_code",
                    "seq_num",
                ],
                filters=[
                    ("subject_id", "in", selected_subjects),
                    ("table", "==", "hosp/diagnoses_icd"),
                ],
            )
        )
    diagnoses = pd.concat(diagnosis_frames, ignore_index=True)
    diagnoses["time"] = pd.to_datetime(diagnoses["time"], errors="coerce")
    diagnoses["seq_num_num"] = pd.to_numeric(diagnoses["seq_num"], errors="coerce")
    return diagnoses


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "font.size": TICK_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#FBFBF8",
            "figure.facecolor": "#FFFFFF",
            "grid.alpha": 0.18,
            "grid.color": "#5F6F7A",
            "axes.grid": True,
        }
    )


def style_axis(
    axis: plt.Axes,
    title: str,
    x_label: str | None = None,
    y_label: str | None = None,
) -> None:
    axis.set_title(title, pad=10, color=TEXT_COLOR)
    if x_label:
        axis.set_xlabel(x_label, color=TEXT_COLOR)
    if y_label:
        axis.set_ylabel(y_label, color=TEXT_COLOR)
    axis.tick_params(axis="both", colors=TEXT_COLOR, width=0.8, length=4)


def style_boxplot(box: dict[str, Any], colors: list[str]) -> None:
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.76)
        patch.set_edgecolor(EDGE_COLOR)
        patch.set_linewidth(0.9)
    for median in box["medians"]:
        median.set_color(TEXT_COLOR)
        median.set_linewidth(1.5)


def collapse_for_pie(
    distribution: pd.DataFrame,
    count_col: str,
    percentage_col: str,
    min_percentage: float,
) -> pd.DataFrame:
    major = distribution[distribution[percentage_col] >= min_percentage].copy()
    minor = distribution[distribution[percentage_col] < min_percentage].copy()
    if not minor.empty:
        major = pd.concat(
            [
                major,
                pd.DataFrame(
                    {
                        "disease_category": ["Others"],
                        count_col: [int(minor[count_col].sum())],
                        percentage_col: [round(float(minor[percentage_col].sum()), 2)],
                    }
                ),
            ],
            ignore_index=True,
        )
    return major.sort_values(percentage_col, ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    subset_dir = args.subset_dir
    output_dir = args.output_dir or (subset_dir / "statistics" / "neurips_benchmark_stats")
    output_dir.mkdir(parents=True, exist_ok=True)

    icu = pd.read_parquet(subset_dir / "icu_stay.parquet")
    events_subset = pd.read_parquet(subset_dir / "events.parquet", columns=["icu_stay_id"])
    metadata = pd.read_csv(subset_dir / "sampled_patient_metadata.csv")
    diagnoses = load_diagnoses(metadata)
    icu["enter_time"] = pd.to_datetime(icu["enter_time"], errors="coerce")

    primary = (
        diagnoses.sort_values(["subject_id", "seq_num_num", "orig_code"])
        .drop_duplicates(subset=["subject_id"], keep="first")
        .copy()
    )
    primary["main_icd_code"] = primary["orig_code"].map(normalize_icd)
    primary["disease_category"] = primary.apply(
        lambda row: disease_category(row["orig_coding_system"], row["orig_code"]),
        axis=1,
    )
    primary = primary.merge(
        icu[["subject_id", "icu_stay_id", "survived", "icu_duration_hours", "enter_time"]],
        on="subject_id",
        how="left",
    )

    main_diagnosis_per_patient = primary[
        [
            "subject_id",
            "icu_stay_id",
            "survived",
            "icu_duration_hours",
            "orig_coding_system",
            "orig_code",
            "main_icd_code",
            "seq_num_num",
            "disease_category",
        ]
    ].rename(
        columns={
            "orig_coding_system": "main_icd_system",
            "orig_code": "main_icd_raw_code",
            "seq_num_num": "main_diagnosis_rank",
        }
    )
    main_diagnosis_per_patient.to_csv(output_dir / "main_diagnosis_per_patient.csv", index=False)

    disease_category_distribution = (
        main_diagnosis_per_patient["disease_category"]
        .value_counts()
        .rename_axis("disease_category")
        .reset_index(name="count")
    )
    disease_category_distribution["percentage"] = (
        disease_category_distribution["count"] / len(main_diagnosis_per_patient) * 100
    ).round(2)
    disease_category_distribution.to_csv(output_dir / "disease_category_distribution.csv", index=False)

    top_primary_diagnosis_codes = (
        main_diagnosis_per_patient["main_icd_raw_code"]
        .value_counts()
        .rename_axis("main_icd_raw_code")
        .reset_index(name="count")
    )
    top_primary_diagnosis_codes["percentage"] = (
        top_primary_diagnosis_codes["count"] / len(main_diagnosis_per_patient) * 100
    ).round(2)
    top_primary_diagnosis_codes.to_csv(output_dir / "top_primary_diagnosis_codes.csv", index=False)

    events_per_stay = events_subset.groupby("icu_stay_id").size().rename("event_count").reset_index()
    events_per_stay = events_per_stay.merge(
        icu[["icu_stay_id", "survived"]],
        on="icu_stay_id",
        how="left",
    )
    events_per_stay.to_csv(output_dir / "events_per_stay.csv", index=False)

    diagnosis_burden = (
        diagnoses.groupby("subject_id").size().rename("diagnosis_code_count").reset_index()
        .merge(icu[["subject_id", "survived"]], on="subject_id", how="left")
    )
    diagnosis_burden.to_csv(output_dir / "diagnosis_code_burden_per_patient.csv", index=False)

    duration_bins = [0, 24, 48, 72, 96, 168, 336, 504, np.inf]
    duration_labels = ["0-24h", "24-48h", "48-72h", "72-96h", "96-168h", "168-336h", "336-504h", ">504h"]
    icu["duration_bin"] = pd.cut(
        icu["icu_duration_hours"],
        bins=duration_bins,
        labels=duration_labels,
        include_lowest=True,
        right=True,
    )
    outcome_by_icu_duration_bin = (
        icu.groupby(["duration_bin", "survived"], observed=False).size().rename("count").reset_index()
    )
    outcome_by_icu_duration_bin["outcome"] = outcome_by_icu_duration_bin["survived"].map(
        {True: "Survived", False: "Died"}
    )
    outcome_by_icu_duration_bin.to_csv(output_dir / "outcome_by_icu_duration_bin.csv", index=False)

    closest = diagnoses.merge(
        icu[["subject_id", "icu_stay_id", "enter_time", "survived"]],
        on="subject_id",
        how="left",
    )
    closest["abs_hours_to_icu_enter"] = (
        (closest["time"] - closest["enter_time"]).abs() / pd.Timedelta(hours=1)
    )
    closest = (
        closest.sort_values(["subject_id", "abs_hours_to_icu_enter", "seq_num_num", "orig_code"])
        .drop_duplicates("subject_id", keep="first")
        .copy()
    )
    closest["main_icd_code"] = closest["orig_code"].map(normalize_icd)
    closest["disease_category"] = closest.apply(
        lambda row: disease_category(row["orig_coding_system"], row["orig_code"]),
        axis=1,
    )

    main_diagnosis_closest_to_icu = closest[
        [
            "subject_id",
            "icu_stay_id",
            "survived",
            "enter_time",
            "time",
            "abs_hours_to_icu_enter",
            "orig_coding_system",
            "orig_code",
            "main_icd_code",
            "seq_num_num",
            "disease_category",
        ]
    ].rename(
        columns={
            "time": "diagnosis_time",
            "orig_coding_system": "main_icd_system",
            "orig_code": "main_icd_raw_code",
            "seq_num_num": "main_diagnosis_rank",
        }
    )
    main_diagnosis_closest_to_icu.to_csv(
        output_dir / "main_diagnosis_closest_to_icu_per_patient.csv",
        index=False,
    )

    disease_category_distribution_closest_to_icu = (
        main_diagnosis_closest_to_icu["disease_category"]
        .value_counts()
        .rename_axis("disease_category")
        .reset_index(name="count_closest")
    )
    disease_category_distribution_closest_to_icu["percentage_closest"] = (
        disease_category_distribution_closest_to_icu["count_closest"] / len(main_diagnosis_closest_to_icu) * 100
    ).round(2)
    disease_category_distribution_closest_to_icu.to_csv(
        output_dir / "disease_category_distribution_closest_to_icu.csv",
        index=False,
    )

    primary_vs_closest = disease_category_distribution.rename(
        columns={"count": "count_primary", "percentage": "percentage_primary"}
    ).merge(
        disease_category_distribution_closest_to_icu,
        on="disease_category",
        how="outer",
    ).fillna(0)
    primary_vs_closest["count_primary"] = primary_vs_closest["count_primary"].astype(int)
    primary_vs_closest["count_closest"] = primary_vs_closest["count_closest"].astype(int)
    primary_vs_closest["delta_count"] = (
        primary_vs_closest["count_closest"] - primary_vs_closest["count_primary"]
    )
    primary_vs_closest["delta_pct_points"] = (
        primary_vs_closest["percentage_closest"] - primary_vs_closest["percentage_primary"]
    ).round(2)
    primary_vs_closest = primary_vs_closest.sort_values(
        ["count_closest", "count_primary"],
        ascending=False,
    ).reset_index(drop=True)
    primary_vs_closest.to_csv(
        output_dir / "disease_category_distribution_comparison_primary_vs_closest.csv",
        index=False,
    )

    patients_changed = main_diagnosis_per_patient[
        ["subject_id", "disease_category"]
    ].rename(columns={"disease_category": "category_primary"}).merge(
        main_diagnosis_closest_to_icu[["subject_id", "disease_category"]].rename(
            columns={"disease_category": "category_closest"}
        ),
        on="subject_id",
        how="inner",
    )
    patients_changed = patients_changed[
        patients_changed["category_primary"] != patients_changed["category_closest"]
    ].copy()
    patients_changed.to_csv(output_dir / "patients_changed_category_primary_vs_closest.csv", index=False)

    summary = {
        "n_patients": int(len(icu)),
        "n_survived": int((icu["survived"] == True).sum()),
        "n_died": int((icu["survived"] == False).sum()),
        "icu_duration_hours": {
            "mean": float(icu["icu_duration_hours"].mean()),
            "median": float(icu["icu_duration_hours"].median()),
            "std": float(icu["icu_duration_hours"].std()),
            "min": float(icu["icu_duration_hours"].min()),
            "max": float(icu["icu_duration_hours"].max()),
        },
        "events_per_stay": {
            "mean": float(events_per_stay["event_count"].mean()),
            "median": float(events_per_stay["event_count"].median()),
            "std": float(events_per_stay["event_count"].std()),
            "min": int(events_per_stay["event_count"].min()),
            "max": int(events_per_stay["event_count"].max()),
        },
        "diagnosis_codes_per_patient": {
            "mean": float(diagnosis_burden["diagnosis_code_count"].mean()),
            "median": float(diagnosis_burden["diagnosis_code_count"].median()),
            "std": float(diagnosis_burden["diagnosis_code_count"].std()),
            "min": int(diagnosis_burden["diagnosis_code_count"].min()),
            "max": int(diagnosis_burden["diagnosis_code_count"].max()),
        },
    }
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    configure_matplotlib()

    survived_duration = icu[icu["survived"] == True]["icu_duration_hours"]
    died_duration = icu[icu["survived"] == False]["icu_duration_hours"]

    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE, constrained_layout=True)
    bins = np.linspace(0, max(icu["icu_duration_hours"].max(), 100), 22)
    ax.hist(
        survived_duration,
        bins=bins,
        color=NATURE_PALETTE[1],
        alpha=0.76,
        edgecolor=EDGE_COLOR,
        linewidth=0.7,
        label="Survived",
    )
    ax.hist(
        died_duration,
        bins=bins,
        color=NATURE_PALETTE[5],
        alpha=0.76,
        edgecolor=EDGE_COLOR,
        linewidth=0.7,
        label="Died",
    )
    ax.axvline(float(survived_duration.median()), color=NATURE_PALETTE[1], linestyle="--", linewidth=1.4)
    ax.axvline(float(died_duration.median()), color=NATURE_PALETTE[5], linestyle="--", linewidth=1.4)
    style_axis(ax, "ICU Stay Duration Distribution", "ICU Duration (hours)", "Patient Count")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    fig.savefig(output_dir / "figure_1_icu_duration_histogram.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE, constrained_layout=True)
    box = ax.boxplot(
        [survived_duration, died_duration],
        tick_labels=["Survived", "Died"],
        patch_artist=True,
        widths=0.52,
        showfliers=True,
    )
    style_boxplot(box, [NATURE_PALETTE[1], NATURE_PALETTE[5]])
    style_axis(ax, "Outcome-Stratified ICU Duration", y_label="ICU Duration (hours)")
    fig.savefig(output_dir / "figure_2_icu_duration_boxplot.png", bbox_inches="tight")
    plt.close(fig)

    overall_duration = icu["icu_duration_hours"]
    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE, constrained_layout=True)
    bins = np.linspace(0, max(overall_duration.max(), 100), 24)
    ax.hist(
        overall_duration,
        bins=bins,
        color="#4C78A8",
        alpha=0.8,
        edgecolor=EDGE_COLOR,
        linewidth=0.7,
    )
    ax.axvline(float(overall_duration.median()), color="#E76F51", linestyle="--", linewidth=1.5)
    style_axis(ax, "Overall ICU Stay Duration Distribution", "ICU Duration (hours)", "Patient Count")
    fig.savefig(output_dir / "figure_8_icu_duration_overall_histogram.png", bbox_inches="tight")
    plt.close(fig)

    primary_pie_data = collapse_for_pie(
        disease_category_distribution,
        count_col="count",
        percentage_col="percentage",
        min_percentage=args.pie_min_percentage,
    )
    primary_pie_data.to_csv(output_dir / "disease_category_distribution_primary_pie_groups.csv", index=False)

    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE, constrained_layout=True)
    wedges, labels, percentages = ax.pie(
        primary_pie_data["percentage"],
        labels=primary_pie_data["disease_category"],
        autopct="%1.1f%%",
        startangle=140,
        colors=NATURE_PALETTE[: len(primary_pie_data)],
        wedgeprops={"linewidth": 1.0, "edgecolor": EDGE_COLOR},
        textprops={"fontsize": 10},
    )
    for label in labels:
        label.set_color(TEXT_COLOR)
    for pct in percentages:
        pct.set_color(TEXT_COLOR)
    ax.set_title("Main Disease Category Distribution", pad=10, color=TEXT_COLOR)
    fig.savefig(output_dir / "figure_3_main_disease_pie_primary.png", bbox_inches="tight")
    plt.close(fig)

    outcome_counts = (
        icu["survived"].map({True: "Survived", False: "Died"}).value_counts().reindex(["Survived", "Died"])
    )
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE, constrained_layout=True)
    ax.bar(
        outcome_counts.index,
        outcome_counts.values,
        color=[NATURE_PALETTE[1], NATURE_PALETTE[5]],
        alpha=0.82,
        edgecolor=EDGE_COLOR,
        linewidth=0.9,
    )
    style_axis(ax, "Outcome Balance", y_label="Patient Count")
    for idx, value in enumerate(outcome_counts.values):
        ax.text(idx, value + 1, str(int(value)), ha="center", va="bottom", fontsize=10)
    fig.savefig(output_dir / "figure_4_outcome_balance_bar.png", bbox_inches="tight")
    plt.close(fig)

    events_survived = events_per_stay[events_per_stay["survived"] == True]["event_count"]
    events_died = events_per_stay[events_per_stay["survived"] == False]["event_count"]
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE, constrained_layout=True)
    box = ax.boxplot(
        [events_survived, events_died],
        tick_labels=["Survived", "Died"],
        patch_artist=True,
        widths=0.52,
    )
    style_boxplot(box, [NATURE_PALETTE[1], NATURE_PALETTE[5]])
    style_axis(ax, "Events per Stay", y_label="Event Count")
    fig.savefig(output_dir / "figure_5_events_per_stay_boxplot.png", bbox_inches="tight")
    plt.close(fig)

    diagnosis_survived = diagnosis_burden[diagnosis_burden["survived"] == True]["diagnosis_code_count"]
    diagnosis_died = diagnosis_burden[diagnosis_burden["survived"] == False]["diagnosis_code_count"]
    fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE, constrained_layout=True)
    box = ax.boxplot(
        [diagnosis_survived, diagnosis_died],
        tick_labels=["Survived", "Died"],
        patch_artist=True,
        widths=0.52,
    )
    style_boxplot(box, [NATURE_PALETTE[1], NATURE_PALETTE[5]])
    style_axis(ax, "Diagnosis Burden per Patient", y_label="Number of ICD Diagnoses")
    fig.savefig(output_dir / "figure_6_diagnosis_burden_boxplot.png", bbox_inches="tight")
    plt.close(fig)

    closest_pie_data = collapse_for_pie(
        disease_category_distribution_closest_to_icu,
        count_col="count_closest",
        percentage_col="percentage_closest",
        min_percentage=args.pie_min_percentage,
    )
    closest_pie_data.to_csv(output_dir / "disease_category_distribution_closest_to_icu_pie_groups.csv", index=False)

    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE, constrained_layout=True)
    wedges, labels, percentages = ax.pie(
        closest_pie_data["percentage_closest"],
        labels=closest_pie_data["disease_category"],
        autopct="%1.1f%%",
        startangle=140,
        colors=NATURE_PALETTE[: len(closest_pie_data)],
        wedgeprops={"linewidth": 1.0, "edgecolor": EDGE_COLOR},
        textprops={"fontsize": 10},
    )
    for label in labels:
        label.set_color(TEXT_COLOR)
    for pct in percentages:
        pct.set_color(TEXT_COLOR)
    ax.set_title("Disease Category Distribution", pad=10, color=TEXT_COLOR)
    fig.savefig(output_dir / "figure_7_disease_category_pie_closest_to_icu.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Output directory: {output_dir}")
    print(f"Patients: {len(icu)}")
    print(f"Patients changed primary->closest: {len(patients_changed)}")


if __name__ == "__main__":
    main()
