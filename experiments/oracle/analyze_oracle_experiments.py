"""Analyze Oracle condition-suite outputs for Q1/Q2 outcome-bias and scoring validity."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.oracle.common import (
    DOMAIN_KEYS,
    NEGATIVE_STATUS_LABELS,
    POSITIVE_STATUS_LABELS,
    assign_normalized_time_bin,
    auc_from_scores,
    compute_domain_consistency,
    extract_domain_labels,
    extract_overall_label,
    normalize_time_position,
    spearman_correlation,
)
from utils.status_scoring import status_to_score

FULL_VISIBLE_CONDITION = "full_visible"


def _find_latest_run(output_root: Path) -> Optional[Path]:
    candidates = sorted([p for p in output_root.glob("oracle_conditions_*") if p.is_dir()], reverse=True)
    return candidates[0] if candidates else None


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_condition_prediction_files(run_dir: Path) -> Iterable[Tuple[str, Path]]:
    conditions_dir = run_dir / "conditions"
    if not conditions_dir.exists():
        return
    for condition_dir in sorted(conditions_dir.iterdir()):
        if not condition_dir.is_dir():
            continue
        condition = condition_dir.name
        patients_dir = condition_dir / "patients"
        if not patients_dir.exists():
            continue
        for patient_dir in sorted(patients_dir.iterdir()):
            pred_path = patient_dir / "oracle_predictions.json"
            if pred_path.exists():
                yield condition, pred_path


def _build_window_level_dataframe(run_dir: Path, num_time_bins: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for condition, pred_path in _iter_condition_prediction_files(run_dir):
        payload = _load_json(pred_path)
        trajectory_meta = payload.get("trajectory_metadata", {})
        true_survived = bool(trajectory_meta.get("true_survived", trajectory_meta.get("survived")))
        prompt_survived = bool(trajectory_meta.get("prompt_survived", trajectory_meta.get("survived")))
        subject_id = int(payload.get("subject_id"))
        icu_stay_id = int(payload.get("icu_stay_id"))
        window_outputs = payload.get("window_outputs", [])
        total_windows = len(window_outputs)

        for index, window_output in enumerate(window_outputs):
            window_meta = window_output.get("window_metadata", {})
            oracle_output = window_output.get("oracle_output", {})
            if not isinstance(oracle_output, dict):
                oracle_output = {}

            overall_label = extract_overall_label(oracle_output)
            overall_score = status_to_score(overall_label)
            domain_labels = extract_domain_labels(oracle_output)
            domain_consistency = compute_domain_consistency(oracle_output)

            hours_since_admission = float(window_meta.get("hours_since_admission") or 0.0)
            window_index_raw = window_output.get("window_index")
            try:
                window_index = int(window_index_raw)
            except (TypeError, ValueError):
                window_index = int(index)

            row = {
                "condition": condition,
                "subject_id": subject_id,
                "icu_stay_id": icu_stay_id,
                "patient_id": f"{subject_id}_{icu_stay_id}",
                "true_survived": true_survived,
                "prompt_survived": prompt_survived,
                "true_outcome": "survived" if true_survived else "died",
                "prompt_outcome": "survived" if prompt_survived else "died",
                "window_index": window_index,
                "window_index_0": max(0, window_index),
                "num_windows": total_windows,
                "hours_since_admission": hours_since_admission,
                "overall_label": overall_label,
                "overall_score": overall_score,
                "is_positive_label": overall_label in POSITIVE_STATUS_LABELS,
                "is_negative_label": overall_label in NEGATIVE_STATUS_LABELS,
                "is_positive_score": overall_score > 0,
                "is_negative_score": overall_score < 0,
                "weighted_domain_score": float(domain_consistency["weighted_domain_score"]),
                "weighted_domain_label": str(domain_consistency["weighted_domain_label"]),
                "domain_consistency_match": bool(domain_consistency["is_match"]),
                "domain_consistency_evaluable": bool(domain_consistency["is_evaluable"]),
                "domain_score_abs_gap": float(domain_consistency["score_abs_gap"]),
            }

            for key in DOMAIN_KEYS:
                label = domain_labels.get(key, "insufficient_data")
                row[f"domain_{key}_label"] = label
                row[f"domain_{key}_score"] = status_to_score(label)

            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    group_cols = ["condition", "subject_id", "icu_stay_id"]
    max_hours = df.groupby(group_cols)["hours_since_admission"].transform("max")
    normalized = []
    for _, row in df.iterrows():
        max_h = float(max_hours.loc[row.name])
        if max_h > 0:
            normalized.append(normalize_time_position(float(row["hours_since_admission"]), max_h))
            continue

        num_windows = int(row.get("num_windows") or 0)
        if num_windows > 1:
            normalized.append(float(row["window_index_0"]) / float(num_windows - 1))
        else:
            normalized.append(0.0)

    df["normalized_time"] = normalized
    df["time_bin"] = df["normalized_time"].map(lambda value: assign_normalized_time_bin(value, num_bins=num_time_bins))
    return df


def _save_dataframe(df: pd.DataFrame, path_csv: Path, path_parquet: Optional[Path] = None) -> None:
    df.to_csv(path_csv, index=False)
    if path_parquet is not None:
        try:
            df.to_parquet(path_parquet, index=False)
        except Exception:
            pass


def _save_q1_outputs(df: pd.DataFrame, analysis_dir: Path) -> None:
    q1_dir = analysis_dir / "q1"
    q1_dir.mkdir(parents=True, exist_ok=True)

    # Status distribution.
    status_dist = df.groupby(["condition", "true_outcome", "overall_label"]).size().reset_index(name="count")
    status_dist["proportion"] = status_dist.groupby(["condition", "true_outcome"])["count"].transform(
        lambda col: col / col.sum() if col.sum() > 0 else 0.0
    )
    _save_dataframe(status_dist, q1_dir / "status_distribution.csv")

    # Score distribution summary.
    score_dist = (
        df.groupby(["condition", "true_outcome"])["overall_score"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    _save_dataframe(score_dist, q1_dir / "score_distribution.csv")

    # Patient-level means and paired deltas.
    patient_means = (
        df.groupby(["condition", "subject_id", "icu_stay_id", "patient_id", "true_outcome"])["overall_score"]
        .mean()
        .reset_index(name="patient_mean_score")
    )
    _save_dataframe(patient_means, q1_dir / "patient_mean_scores.csv")

    pivot = patient_means.pivot_table(
        index=["subject_id", "icu_stay_id", "patient_id", "true_outcome"],
        columns="condition",
        values="patient_mean_score",
        aggfunc="first",
    ).reset_index()
    _save_dataframe(pivot, q1_dir / "patient_mean_scores_pivot.csv")

    delta_rows: List[Dict[str, Any]] = []
    condition_names = sorted(df["condition"].unique().tolist())
    for left, right in itertools.combinations(condition_names, 2):
        valid = pivot.dropna(subset=[left, right]).copy()
        valid["delta"] = valid[left] - valid[right]
        for _, row in valid.iterrows():
            delta_rows.append(
                {
                    "condition_left": left,
                    "condition_right": right,
                    "subject_id": int(row["subject_id"]),
                    "icu_stay_id": int(row["icu_stay_id"]),
                    "patient_id": row["patient_id"],
                    "true_outcome": row["true_outcome"],
                    "left_score": float(row[left]),
                    "right_score": float(row[right]),
                    "delta": float(row["delta"]),
                }
            )
    deltas_df = pd.DataFrame(delta_rows)
    if not deltas_df.empty:
        _save_dataframe(deltas_df, q1_dir / "paired_patient_deltas.csv")
        delta_summary = (
            deltas_df.groupby(["condition_left", "condition_right", "true_outcome"])["delta"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
        )
        _save_dataframe(delta_summary, q1_dir / "paired_delta_summary.csv")

    # All-positive / all-negative diagnostics.
    patient_diag = (
        df.groupby(["condition", "subject_id", "icu_stay_id", "patient_id", "true_outcome"])
        .agg(
            num_windows=("overall_score", "count"),
            positive_rate=("is_positive_score", "mean"),
            negative_rate=("is_negative_score", "mean"),
            all_positive=("is_positive_score", "all"),
            all_negative=("is_negative_score", "all"),
        )
        .reset_index()
    )
    _save_dataframe(patient_diag, q1_dir / "patient_all_positive_negative_diagnostics.csv")
    diag_summary = (
        patient_diag.groupby(["condition", "true_outcome"])
        .agg(
            patients=("patient_id", "count"),
            mean_positive_rate=("positive_rate", "mean"),
            mean_negative_rate=("negative_rate", "mean"),
            pct_all_positive=("all_positive", "mean"),
            pct_all_negative=("all_negative", "mean"),
        )
        .reset_index()
    )
    _save_dataframe(diag_summary, q1_dir / "all_positive_negative_summary.csv")

    # Plot: patient-level mean scores by condition/outcome.
    patient_group_stats = (
        patient_means.groupby(["condition", "true_outcome"])["patient_mean_score"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    _save_dataframe(patient_group_stats, q1_dir / "patient_mean_score_summary.csv")

    condition_order = [FULL_VISIBLE_CONDITION, "masked_outcome"]
    outcome_order = ["died", "survived"]
    condition_rank = {name: i for i, name in enumerate(condition_order)}
    outcome_rank = {name: i for i, name in enumerate(outcome_order)}

    plot_df = patient_group_stats.copy()
    plot_df["condition_rank"] = plot_df["condition"].map(lambda value: condition_rank.get(str(value), 999))
    plot_df["outcome_rank"] = plot_df["true_outcome"].map(lambda value: outcome_rank.get(str(value), 999))
    plot_df = plot_df.sort_values(["condition_rank", "outcome_rank", "condition", "true_outcome"]).reset_index(
        drop=True
    )

    labels = [f"{row.condition}\n{row.true_outcome}" for row in plot_df.itertuples()]
    means = plot_df["mean"].astype(float).to_numpy()
    errs = plot_df["std"].fillna(0.0).astype(float).to_numpy()
    colors = [
        "#c0392b" if outcome == "died" else "#1e8449" for outcome in plot_df["true_outcome"].astype(str).tolist()
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(plot_df), dtype=float)
    ax.bar(x, means, yerr=errs, capsize=6, color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Patient mean overall score")
    ax.set_title("Q1: Patient Mean Overall Score (Mean ± SD)")
    y_min = float(np.nanmin(means - errs)) if len(means) else 0.0
    y_max = float(np.nanmax(means + errs)) if len(means) else 0.0
    ax.set_ylim(min(0.0, y_min) - 0.05, max(0.0, y_max) + 0.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(q1_dir / "q1_score_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot: mean patient trajectories by outcome across ICU progression.
    trajectory_conditions = [
        condition
        for condition in [FULL_VISIBLE_CONDITION, "masked_outcome"]
        if condition in df["condition"].unique().tolist()
    ]
    if not trajectory_conditions:
        trajectory_conditions = sorted(df["condition"].dropna().astype(str).unique().tolist())
    if trajectory_conditions:
        max_time_bin = int(df["time_bin"].max()) if len(df) else 0
        trajectory_summary_frames: List[pd.DataFrame] = []
        fig, axes = plt.subplots(
            1,
            len(trajectory_conditions),
            figsize=(7 * len(trajectory_conditions), 5),
            sharey=True,
            squeeze=False,
        )
        for ax, condition in zip(axes[0], trajectory_conditions):
            condition_df = df[df["condition"] == condition].copy()

            # Average within each patient/bin first, then across patients.
            patient_bin_means = (
                condition_df.groupby(["true_outcome", "patient_id", "time_bin"])["overall_score"]
                .mean()
                .reset_index(name="patient_bin_mean_score")
            )
            trajectory_summary = (
                patient_bin_means.groupby(["true_outcome", "time_bin"])["patient_bin_mean_score"]
                .agg(mean_score="mean", std_score="std", num_patients="count")
                .reset_index()
            )
            trajectory_summary["condition"] = condition
            trajectory_summary = trajectory_summary[
                ["condition", "true_outcome", "time_bin", "num_patients", "mean_score", "std_score"]
            ]
            trajectory_summary_frames.append(trajectory_summary)

            for outcome, color in [("died", "#c0392b"), ("survived", "#1e8449")]:
                outcome_df = trajectory_summary[trajectory_summary["true_outcome"] == outcome].sort_values("time_bin")
                if outcome_df.empty:
                    continue
                x = outcome_df["time_bin"].astype(float).to_numpy()
                mean = outcome_df["mean_score"].astype(float).to_numpy()
                std = outcome_df["std_score"].fillna(0.0).astype(float).to_numpy()
                ax.plot(x, mean, color=color, linewidth=2.2, marker="o", markersize=4.0, label=outcome)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

            patient_counts = condition_df.groupby("true_outcome")["patient_id"].nunique().to_dict()
            ax.set_title(
                f"{condition} (died n={int(patient_counts.get('died', 0))}, survived n={int(patient_counts.get('survived', 0))})"
            )
            ax.set_xlabel("Normalized time bin (0=entry, higher=later)")
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlim(-0.2, max_time_bin + 0.2)
            tick_step = 1 if max_time_bin <= 12 else 2
            ax.set_xticks(np.arange(0, max_time_bin + 1, tick_step))
            ax.grid(alpha=0.3)
            ax.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1)
        axes[0][0].set_ylabel("Overall score (mean ± SD)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", ncol=2)
        fig.suptitle("Q1: Mean Patient Trajectories by Outcome")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(q1_dir / "q1_patient_trajectories.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        trajectory_summary_df = pd.concat(trajectory_summary_frames, ignore_index=True)
        _save_dataframe(trajectory_summary_df, q1_dir / "q1_trajectory_mean_std_by_bin.csv")

    # Plot: status proportions (stacked) per condition/outcome.
    status_pivot = status_dist.pivot_table(
        index=["condition", "true_outcome"],
        columns="overall_label",
        values="proportion",
        fill_value=0.0,
        aggfunc="sum",
    )
    status_pivot = status_pivot.sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(status_pivot.index))
    bottom = np.zeros(len(status_pivot.index))
    labels = list(status_pivot.columns)
    colors = {
        "deteriorating": "#c0392b",
        "fluctuating": "#d35400",
        "stable": "#2980b9",
        "improving": "#27ae60",
        "insufficient_data": "#7f8c8d",
    }
    for label in labels:
        values = status_pivot[label].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=label, color=colors.get(label, None))
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cond}\n{outcome}" for cond, outcome in status_pivot.index], rotation=20)
    ax.set_ylabel("Proportion")
    ax.set_title("Q1: Status Label Proportions")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(q1_dir / "q1_status_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_q2_outputs(df: pd.DataFrame, analysis_dir: Path, num_time_bins: int) -> None:
    q2_dir = analysis_dir / "q2"
    q2_dir.mkdir(parents=True, exist_ok=True)

    q2_df = df[df["condition"] == FULL_VISIBLE_CONDITION].copy()
    if q2_df.empty:
        raise ValueError(f"Q2 requires '{FULL_VISIBLE_CONDITION}' condition, but no rows were found.")

    patient_means = (
        q2_df.groupby(["subject_id", "icu_stay_id", "patient_id", "true_outcome"])["overall_score"]
        .mean()
        .reset_index(name="patient_mean_score")
    )

    y_true = patient_means["true_outcome"].map(lambda x: 1 if x == "survived" else 0).tolist()
    y_score = patient_means["patient_mean_score"].astype(float).tolist()
    auc = auc_from_scores(y_true, y_score)
    separability_df = pd.DataFrame(
        [
            {
                "condition": FULL_VISIBLE_CONDITION,
                "num_patients": int(len(patient_means)),
                "num_survived": int((patient_means["true_outcome"] == "survived").sum()),
                "num_died": int((patient_means["true_outcome"] == "died").sum()),
                "auroc_survived_vs_died": auc,
                "mean_survived_score": float(
                    patient_means.loc[patient_means["true_outcome"] == "survived", "patient_mean_score"].mean()
                ),
                "mean_died_score": float(
                    patient_means.loc[patient_means["true_outcome"] == "died", "patient_mean_score"].mean()
                ),
            }
        ]
    )
    _save_dataframe(separability_df, q2_dir / "q2_separability.csv")
    _save_dataframe(patient_means, q2_dir / "q2_patient_mean_scores.csv")

    window_score_distribution = (
        q2_df.groupby(["true_outcome"])["overall_score"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    _save_dataframe(window_score_distribution, q2_dir / "q2_window_score_distribution.csv")

    # Plot patient-mean scores by true outcome (full_visible only).
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["died", "survived"]
    positions = np.arange(len(order), dtype=float)
    means = []
    errs = []
    colors = ["#c0392b", "#1e8449"]
    for outcome in order:
        subset = patient_means[patient_means["true_outcome"] == outcome]["patient_mean_score"]
        means.append(float(subset.mean()) if len(subset) else float("nan"))
        errs.append(float(subset.std()) if len(subset) else 0.0)
    ax.bar(positions, means, width=0.6, yerr=errs, color=colors, alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylabel("Patient mean overall score")
    ax.set_title("Q2 Task1: Full Visible Score Separability")
    ax.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(q2_dir / "q2_patient_mean_separability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # End-of-trajectory deterioration trend.
    trend_bins = (
        q2_df.groupby(["true_outcome", "time_bin"])
        .agg(
            windows=("overall_label", "count"),
            deteriorating_count=("overall_label", lambda col: int((col == "deteriorating").sum())),
        )
        .reset_index()
    )
    trend_bins["deteriorating_proportion"] = trend_bins["deteriorating_count"] / trend_bins["windows"]
    trend_bins["condition"] = FULL_VISIBLE_CONDITION
    trend_bins = trend_bins[
        ["condition", "true_outcome", "time_bin", "windows", "deteriorating_count", "deteriorating_proportion"]
    ]
    _save_dataframe(trend_bins, q2_dir / "q2_deterioration_trend_by_bin.csv")

    trend_summary_rows: List[Dict[str, Any]] = []
    for outcome, group in trend_bins.groupby("true_outcome"):
        group = group.sort_values("time_bin")
        x = group["time_bin"].astype(float).tolist()
        y = group["deteriorating_proportion"].astype(float).tolist()
        rho = spearman_correlation(x, y)
        slope = float("nan")
        if len(group) >= 2:
            slope = float(
                np.polyfit(
                    group["time_bin"].to_numpy(dtype=float), group["deteriorating_proportion"].to_numpy(dtype=float), 1
                )[0]
            )
        expected_direction = "increasing" if outcome == "died" else "decreasing"
        monotonic_violations = 0
        if len(y) > 1:
            for left, right in zip(y[:-1], y[1:]):
                if expected_direction == "increasing" and right < left:
                    monotonic_violations += 1
                if expected_direction == "decreasing" and right > left:
                    monotonic_violations += 1
        trend_summary_rows.append(
            {
                "condition": FULL_VISIBLE_CONDITION,
                "true_outcome": outcome,
                "expected_monotonic_direction": expected_direction,
                "num_bins_present": int(len(group)),
                "spearman_rho": rho,
                "linear_slope": slope,
                "monotonic_violations_against_expected": int(monotonic_violations),
                "num_time_bins_requested": int(num_time_bins),
            }
        )
    trend_summary_df = pd.DataFrame(trend_summary_rows)
    _save_dataframe(trend_summary_df, q2_dir / "q2_deterioration_trend_summary.csv")

    # Plot deterioration trend by outcome for full_visible.
    fig, ax = plt.subplots(figsize=(10, 5))
    for outcome, color in [("died", "#c0392b"), ("survived", "#1e8449")]:
        group = trend_bins[trend_bins["true_outcome"] == outcome].sort_values("time_bin")
        if group.empty:
            continue
        ax.plot(
            group["time_bin"], group["deteriorating_proportion"], marker="o", linewidth=2, label=outcome, color=color
        )
    ax.set_xlabel(f"Normalized time bin (0 to {num_time_bins - 1})")
    ax.set_ylabel("Proportion labeled deteriorating")
    ax.set_title("Q2 Task1: Deteriorating Trend Over Time (Full Visible)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(q2_dir / "q2_deterioration_trend_died.png", dpi=200, bbox_inches="tight")
    fig.savefig(q2_dir / "q2_deterioration_trend_by_outcome.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Label-distribution-over-time for monotonic sanity check.
    label_time = q2_df.groupby(["true_outcome", "time_bin", "overall_label"]).size().reset_index(name="count")
    label_time["proportion"] = label_time.groupby(["true_outcome", "time_bin"])["count"].transform(
        lambda col: col / col.sum() if col.sum() > 0 else 0.0
    )
    label_time["condition"] = FULL_VISIBLE_CONDITION
    label_time = label_time[["condition", "true_outcome", "time_bin", "overall_label", "count", "proportion"]]
    _save_dataframe(label_time, q2_dir / "q2_label_distribution_by_bin.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    label_order = ["deteriorating", "fluctuating", "stable", "improving", "insufficient_data"]
    label_colors = {
        "deteriorating": "#c0392b",
        "fluctuating": "#d35400",
        "stable": "#2980b9",
        "improving": "#27ae60",
        "insufficient_data": "#7f8c8d",
    }
    for ax, outcome in zip(axes, ["died", "survived"]):
        subset = label_time[label_time["true_outcome"] == outcome]
        for label in label_order:
            group = subset[subset["overall_label"] == label].sort_values("time_bin")
            if group.empty:
                continue
            ax.plot(
                group["time_bin"], group["proportion"], marker="o", linewidth=2, label=label, color=label_colors[label]
            )
        ax.set_title(f"{outcome} (full_visible)")
        ax.set_xlabel(f"Normalized time bin (0 to {num_time_bins - 1})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Label proportion")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=5)
    fig.suptitle("Q2 Task1: Oracle Label Distribution Over Normalized Time")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(q2_dir / "q2_label_distribution_over_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Domain consistency.
    consistency = (
        q2_df.groupby(["true_outcome"])
        .agg(
            total_windows=("domain_consistency_match", "count"),
            evaluable_windows=("domain_consistency_evaluable", "sum"),
            matched_windows=("domain_consistency_match", "sum"),
            mean_abs_score_gap=("domain_score_abs_gap", "mean"),
        )
        .reset_index()
    )
    consistency["condition"] = FULL_VISIBLE_CONDITION
    consistency = consistency[
        [
            "condition",
            "true_outcome",
            "total_windows",
            "evaluable_windows",
            "matched_windows",
            "mean_abs_score_gap",
        ]
    ]
    consistency["match_rate_over_evaluable"] = consistency.apply(
        lambda row: (
            (float(row["matched_windows"]) / float(row["evaluable_windows"]))
            if float(row["evaluable_windows"]) > 0
            else float("nan")
        ),
        axis=1,
    )
    _save_dataframe(consistency, q2_dir / "q2_domain_consistency_summary.csv")

    # Save confusion tables per condition.
    confusion_dir = q2_dir / "domain_confusion"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in confusion_dir.glob("*.csv"):
        stale_path.unlink()
    confusion = pd.crosstab(
        q2_df["weighted_domain_label"],
        q2_df["overall_label"],
        dropna=False,
    )
    confusion.to_csv(confusion_dir / f"{FULL_VISIBLE_CONDITION}_confusion.csv")

    # Plot match rate by outcome (full_visible only).
    fig, ax = plt.subplots(figsize=(8, 5))
    outcomes = ["died", "survived"]
    positions = np.arange(len(outcomes), dtype=float)
    values = []
    for outcome in outcomes:
        subset = consistency[consistency["true_outcome"] == outcome]
        values.append(float(subset["match_rate_over_evaluable"].iloc[0]) if len(subset) else float("nan"))
    ax.bar(positions, values, width=0.6, color=["#c0392b", "#1e8449"], alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels(outcomes, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Domain-to-overall match rate")
    ax.set_title("Q2 Task2: Domain-Level Consistency with Overall Label (Full Visible)")
    fig.tight_layout()
    fig.savefig(q2_dir / "q2_domain_consistency_match_rate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_condition_suite(
    *,
    run_dir: Path,
    analysis_dir: Optional[Path],
    run_q1: bool,
    run_q2: bool,
    num_time_bins: int,
    min_patients_per_condition: int,
) -> None:
    if analysis_dir is None:
        analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df_all = _build_window_level_dataframe(run_dir=run_dir, num_time_bins=num_time_bins)
    if df_all.empty:
        raise ValueError(f"No window-level records found under {run_dir}")

    all_patient_counts = (
        df_all[["condition", "patient_id"]].drop_duplicates().groupby("condition").size().astype(int).to_dict()
    )
    included_conditions = sorted(
        [condition for condition, count in all_patient_counts.items() if int(count) >= int(min_patients_per_condition)]
    )
    excluded_conditions = sorted(
        [condition for condition, count in all_patient_counts.items() if int(count) < int(min_patients_per_condition)]
    )
    if not included_conditions:
        raise ValueError(
            f"No conditions meet min_patients_per_condition={min_patients_per_condition}. "
            f"Available counts: {all_patient_counts}"
        )

    df = df_all[df_all["condition"].isin(included_conditions)].copy()

    _save_dataframe(df, analysis_dir / "window_level.csv", analysis_dir / "window_level.parquet")

    if run_q1:
        _save_q1_outputs(df, analysis_dir)
    if run_q2:
        _save_q2_outputs(df, analysis_dir, num_time_bins=num_time_bins)

    summary = {
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "num_window_rows": int(len(df)),
        "num_window_rows_all_conditions": int(len(df_all)),
        "conditions": sorted(df["condition"].unique().tolist()),
        "conditions_excluded_due_to_min_patients": excluded_conditions,
        "num_patients_per_condition": (
            df[["condition", "patient_id"]].drop_duplicates().groupby("condition").size().to_dict()
        ),
        "num_patients_per_condition_all_conditions": all_patient_counts,
        "q1_enabled": bool(run_q1),
        "q2_enabled": bool(run_q2),
        "num_time_bins": int(num_time_bins),
        "min_patients_per_condition": int(min_patients_per_condition),
    }
    with open(analysis_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Oracle condition-suite outputs for Q1/Q2.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="experiment_results/oracle",
        help="Root directory containing oracle_conditions_* runs.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Specific run directory. If omitted, latest run under --output-root is used.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=None,
        help="Custom analysis output directory (default: <run_dir>/analysis).",
    )
    parser.add_argument("--q1", action="store_true", help="Run Q1 analyses only.")
    parser.add_argument("--q2", action="store_true", help="Run Q2 analyses only.")
    parser.add_argument("--all", action="store_true", help="Run all analyses (default when no mode is provided).")
    parser.add_argument("--num-time-bins", type=int, default=10, help="Number of normalized-time bins (default: 10).")
    parser.add_argument(
        "--min-patients-per-condition",
        type=int,
        default=3,
        help="Exclude conditions with fewer unique patients than this threshold (default: 3).",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    run_dir = Path(args.run_dir) if args.run_dir else _find_latest_run(output_root)
    if run_dir is None or not run_dir.exists():
        raise ValueError(f"No run directory found. Checked: {run_dir or output_root}")

    run_q1 = bool(args.q1)
    run_q2 = bool(args.q2)
    if args.all or (not args.q1 and not args.q2):
        run_q1 = True
        run_q2 = True

    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else None
    analyze_condition_suite(
        run_dir=run_dir,
        analysis_dir=analysis_dir,
        run_q1=run_q1,
        run_q2=run_q2,
        num_time_bins=max(2, int(args.num_time_bins)),
        min_patients_per_condition=max(1, int(args.min_patients_per_condition)),
    )
    print(f"Analysis complete. Results saved under: {analysis_dir or (run_dir / 'analysis')}")


if __name__ == "__main__":
    main()
