"""Evaluate patient-status predictions against Oracle labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.oracle.common import assign_normalized_time_bin, extract_overall_label
from utils.status_scoring import normalize_status_label

NUM_TIME_BINS = 10
MEMORY_DEPTH_NUM_BINS = 6
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260409
KNOWN_PATIENT_STATUS_MODES = (
    "memory",
    "full_history_events",
    "local_events_only",
)
PATIENT_STATUS_ROOT_NAMES = (
    "patient_status",
    "patient_status_experiment",
)


@dataclass(frozen=True)
class WindowLabel:
    label: str
    hours_since_admission: float
    num_windows: int


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _has_patient_status_predictions(path: Path) -> bool:
    patients_dir = path / "patients"
    if not patients_dir.exists() or not patients_dir.is_dir():
        return False
    return any(patients_dir.glob("*/patient_status_predictions.json"))


def _ensure_predictions_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {path}")
    if _has_patient_status_predictions(path):
        return path
    for root_name in PATIENT_STATUS_ROOT_NAMES:
        nested_root = path / root_name
        if nested_root.exists() and nested_root.is_dir() and _has_patient_status_predictions(nested_root):
            return nested_root
    raise FileNotFoundError(
        f"No patient_status_predictions.json found under {path}. "
        "Expected either <dir>/patients/*/patient_status_predictions.json or "
        "<dir>/<patient_status|patient_status_experiment>/patients/*/patient_status_predictions.json."
    )


def _discover_patient_status_mode_dirs(path: Path) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    for mode in KNOWN_PATIENT_STATUS_MODES:
        candidate = path / mode
        if candidate.exists() and candidate.is_dir() and _has_patient_status_predictions(candidate):
            discovered[str(mode)] = candidate
    return discovered


def _ensure_oracle_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Oracle directory not found: {path}")
    patients_dir = path / "patients"
    if not patients_dir.exists():
        raise FileNotFoundError(f"Missing patients directory under oracle path: {path}")
    if not list(patients_dir.glob("*/oracle_predictions.json")):
        raise FileNotFoundError(f"No oracle_predictions.json files found under {patients_dir}")
    return path


def _default_output_dir_from_med_evo_root(med_evo_root: Path) -> Path:
    experiment_name = med_evo_root.name
    memory_run_name = med_evo_root.parent.name if med_evo_root.parent != med_evo_root else med_evo_root.name
    if not memory_run_name:
        raise ValueError(f"Cannot infer memory run name from med_evo_root={med_evo_root}")
    if not experiment_name:
        raise ValueError(f"Cannot infer experiment name from med_evo_root={med_evo_root}")
    return Path("evaluation_results") / memory_run_name / experiment_name


def _default_output_root_from_patient_status_input(patient_status_input: Path) -> Path:
    if patient_status_input.name in PATIENT_STATUS_ROOT_NAMES:
        memory_run_name = patient_status_input.parent.name
        if not memory_run_name:
            raise ValueError(f"Cannot infer memory run name from patient_status_input={patient_status_input}")
        return Path("evaluation_results") / memory_run_name / patient_status_input.name
    if patient_status_input.parent.name in PATIENT_STATUS_ROOT_NAMES:
        memory_run_name = patient_status_input.parent.parent.name
        if not memory_run_name:
            raise ValueError(f"Cannot infer memory run name from patient_status_input={patient_status_input}")
        return Path("evaluation_results") / memory_run_name / patient_status_input.parent.name / patient_status_input.name
    return _default_output_dir_from_med_evo_root(patient_status_input)


def _infer_snapshot_window_index(snapshot: Dict[str, Any]) -> Optional[int]:
    working_memory = snapshot.get("working_memory")
    if not isinstance(working_memory, list) or not working_memory:
        return None
    current_window = working_memory[-1]
    if not isinstance(current_window, dict):
        return None
    window_id = current_window.get("window_id")
    if window_id is None:
        return None
    return int(window_id)


def _collect_windowed_snapshots(memory_snapshots: Sequence[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    snapshots_by_window: Dict[int, Dict[str, Any]] = {}
    for snapshot in memory_snapshots:
        if not isinstance(snapshot, dict):
            raise ValueError("Invalid memory snapshot row; expected dict.")
        window_index = _infer_snapshot_window_index(snapshot)
        if window_index is None:
            continue
        snapshots_by_window[int(window_index)] = snapshot
    return sorted(snapshots_by_window.items(), key=lambda item: item[0])


def _load_memory_snapshots(patient_dir: Path) -> List[Dict[str, Any]]:
    memory_db_path = patient_dir / "memory_database.json"
    payload = _load_json(memory_db_path)
    memory_snapshots = payload.get("memory_snapshots")
    if not isinstance(memory_snapshots, list):
        raise ValueError(f"Invalid memory_snapshots list in {memory_db_path}")
    return memory_snapshots


def _bootstrap_mean_ci(values: Sequence[float], *, n_samples: int, rng: np.random.Generator) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        value = float(arr[0])
        return value, value
    samples = rng.choice(arr, size=(n_samples, len(arr)), replace=True)
    means = samples.mean(axis=1)
    low = float(np.quantile(means, 0.025))
    high = float(np.quantile(means, 0.975))
    return low, high


def _window_indices_from_oracle_outputs(window_outputs: List[Dict[str, Any]], pred_path: Path) -> List[int]:
    indices: List[int] = []
    for position, item in enumerate(window_outputs):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid window output row at position={position} in {pred_path}")
        value = item.get("window_index")
        if not isinstance(value, int):
            raise ValueError(f"Missing integer window_index at position={position} in {pred_path}")
        window_index = int(value)
        if window_index < 0:
            raise ValueError(f"Negative window_index={window_index} at position={position} in {pred_path}")
        indices.append(window_index)

    unique_sorted = sorted(set(indices))
    expected = list(range(len(window_outputs)))
    if unique_sorted != expected:
        raise ValueError(
            "Oracle window_index must be strict 0-based contiguous [0..N-1]. "
            f"Found={unique_sorted[:12]} in {pred_path}. "
            "Convert the Oracle run to 0-based first."
        )
    return indices


def _load_oracle_labels(oracle_dir: Path) -> Dict[str, Dict[int, WindowLabel]]:
    labels_by_patient: Dict[str, Dict[int, WindowLabel]] = {}
    for pred_path in sorted((oracle_dir / "patients").glob("*/oracle_predictions.json")):
        patient_id = pred_path.parent.name
        payload = _load_json(pred_path)
        window_outputs = payload.get("window_outputs")
        if not isinstance(window_outputs, list) or not window_outputs:
            raise ValueError(f"Missing or empty window_outputs in {pred_path}")

        indices = _window_indices_from_oracle_outputs(window_outputs, pred_path)
        total_windows = len(indices)
        patient_labels: Dict[int, WindowLabel] = {}
        for output, window_index in zip(window_outputs, indices):
            if not isinstance(output, dict):
                raise ValueError(f"Invalid window output row in {pred_path}")
            oracle_output = output.get("oracle_output")
            if not isinstance(oracle_output, dict):
                raise ValueError(f"Missing oracle_output in {pred_path} for patient {patient_id}")
            label = normalize_status_label(extract_overall_label(oracle_output))
            metadata = output.get("window_metadata")
            hours = 0.0
            if isinstance(metadata, dict):
                try:
                    hours = float(metadata.get("hours_since_admission") or 0.0)
                except (TypeError, ValueError):
                    hours = 0.0
            patient_labels[int(window_index)] = WindowLabel(
                label=label,
                hours_since_admission=hours,
                num_windows=total_windows,
            )
        labels_by_patient[patient_id] = patient_labels
    if not labels_by_patient:
        raise ValueError(f"No oracle labels loaded from {oracle_dir}")
    return labels_by_patient


def _memory_depth_by_window(source_patient_dir: Path) -> Dict[int, int]:
    memory_snapshots = _load_memory_snapshots(source_patient_dir)
    windowed = _collect_windowed_snapshots(memory_snapshots)
    depths: Dict[int, int] = {}
    for window_index, snapshot in windowed:
        if not isinstance(snapshot, dict):
            raise ValueError(f"Invalid memory snapshot object in {source_patient_dir}")
        trajectory_memory = snapshot.get("trajectory_memory")
        critical_events = snapshot.get("critical_events")
        insights = snapshot.get("insights")
        if not isinstance(trajectory_memory, list):
            trajectory_memory = []
        if not isinstance(critical_events, list):
            critical_events = []
        if not isinstance(insights, list):
            insights = []
        depth = len(trajectory_memory) + len(critical_events) + len(insights)
        depths[int(window_index)] = int(depth)
    return depths


def _load_prediction_rows(
    prediction_dir: Path,
    oracle_labels: Dict[str, Dict[int, WindowLabel]],
    system_name: str,
    include_memory_depth: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    prediction_files = sorted((prediction_dir / "patients").glob("*/patient_status_predictions.json"))
    if not prediction_files:
        raise FileNotFoundError(f"No patient_status_predictions.json files found under {prediction_dir / 'patients'}")

    for pred_path in prediction_files:
        payload = _load_json(pred_path)
        patient_id = pred_path.parent.name
        subject_id_raw = payload.get("subject_id")
        icu_stay_id_raw = payload.get("icu_stay_id")
        if subject_id_raw is None or icu_stay_id_raw is None:
            raise ValueError(f"Missing subject_id/icu_stay_id in {pred_path}")
        subject_id = int(subject_id_raw)
        icu_stay_id = int(icu_stay_id_raw)

        status_predictions = payload.get("status_predictions")
        if not isinstance(status_predictions, list):
            raise ValueError(f"Missing status_predictions list in {pred_path}")

        oracle_map = oracle_labels.get(patient_id)
        if oracle_map is None:
            continue

        memory_depth_map: Dict[int, int] = {}
        if include_memory_depth:
            source_patient_dir_raw = payload.get("source_patient_dir")
            if source_patient_dir_raw is None:
                raise ValueError(f"Missing source_patient_dir in {pred_path}")
            memory_depth_map = _memory_depth_by_window(Path(str(source_patient_dir_raw)))

        for item in status_predictions:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid status prediction row in {pred_path}")
            if "window_index" not in item:
                raise ValueError(f"Missing window_index in status prediction row in {pred_path}")
            window_index = int(item.get("window_index"))
            if window_index < 0:
                raise ValueError(f"Negative agent window_index={window_index} in {pred_path}")
            oracle_row = oracle_map.get(window_index)
            if oracle_row is None:
                continue

            pred_label = normalize_status_label(item.get("status_label"))
            true_label = oracle_row.label
            num_windows = int(oracle_row.num_windows)
            if num_windows > 1:
                relative_time = float(window_index) / float(num_windows - 1)
            else:
                relative_time = 0.0
            relative_time = min(max(relative_time, 0.0), 1.0)
            time_bin = assign_normalized_time_bin(relative_time, num_bins=NUM_TIME_BINS)
            memory_depth = memory_depth_map.get(window_index)

            rows.append(
                {
                    "system": system_name,
                    "patient_id": patient_id,
                    "subject_id": subject_id,
                    "icu_stay_id": icu_stay_id,
                    "window_index": window_index,
                    "num_windows": num_windows,
                    "relative_time": relative_time,
                    "time_bin": time_bin,
                    "hours_since_admission": float(item.get("hours_since_admission") or 0.0),
                    "pred_label": pred_label,
                    "true_label": true_label,
                    "is_correct": int(pred_label == true_label),
                    "memory_depth": memory_depth,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(
            f"No matched prediction windows for system='{system_name}'. "
            "Check overlap between prediction windows and oracle windows."
        )
    return frame


def _compute_accuracy_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    total_windows = int(len(frame))
    per_patient = frame.groupby("patient_id")["is_correct"].mean().astype(float)
    macro_acc = float(per_patient.mean()) if len(per_patient) > 0 else float("nan")
    return {
        "num_windows": total_windows,
        "num_patients": int(frame["patient_id"].nunique()),
        "macro_acc": macro_acc,
    }


def _memory_depth_bins(values: pd.Series) -> Tuple[pd.Series, List[str]]:
    valid = values.dropna().astype(float)
    if valid.empty:
        raise ValueError("Cannot build memory-depth bins from empty values.")
    unique = np.unique(valid.to_numpy(dtype=float))
    if len(unique) <= MEMORY_DEPTH_NUM_BINS:
        sorted_unique = sorted(int(value) for value in unique.tolist())
        index_map = {value: idx for idx, value in enumerate(sorted_unique)}
        mapped = values.map(lambda x: index_map.get(int(float(x))) if pd.notna(x) else pd.NA)
        labels = [f"{value}" for value in sorted_unique]
        return mapped.astype("Int64"), labels

    quantile_bins = pd.qcut(valid, q=MEMORY_DEPTH_NUM_BINS, labels=False, duplicates="drop")
    if quantile_bins.isna().all():
        raise ValueError("Failed to create quantile memory-depth bins.")

    bin_index_by_row = pd.Series(index=valid.index, data=quantile_bins.astype(int).to_numpy())
    full_codes = pd.Series(pd.NA, index=values.index, dtype="Int64")
    full_codes.loc[bin_index_by_row.index] = bin_index_by_row.astype("Int64")

    labels: List[str] = []
    for bin_id in sorted(bin_index_by_row.unique().tolist()):
        selected = valid[bin_index_by_row == bin_id]
        left = int(math.floor(float(selected.min())))
        right = int(math.ceil(float(selected.max())))
        labels.append(f"{left}-{right}")
    return full_codes.astype("Int64"), labels


def _plot_relative_time_accuracy(
    frame: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    plot_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for system_name, system_df in frame.groupby("system"):
        for time_bin in range(NUM_TIME_BINS):
            subset = system_df[system_df["time_bin"] == time_bin]
            values = subset["is_correct"].astype(float).tolist()
            mean_acc = float(np.mean(values)) if values else float("nan")
            ci_low, ci_high = _bootstrap_mean_ci(values, n_samples=BOOTSTRAP_SAMPLES, rng=rng)
            plot_rows.append(
                {
                    "system": system_name,
                    "time_bin": int(time_bin),
                    "time_bin_mid_pct": float((time_bin + 0.5) * (100.0 / NUM_TIME_BINS)),
                    "n_windows": int(len(values)),
                    "accuracy": mean_acc,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    plot_df = pd.DataFrame(plot_rows).sort_values(["system", "time_bin"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"MedEvo": "#1f77b4", "Baseline": "#ff7f0e"}
    for system_name in plot_df["system"].unique().tolist():
        subset = plot_df[plot_df["system"] == system_name].copy()
        x = subset["time_bin_mid_pct"].to_numpy(dtype=float)
        y = subset["accuracy"].to_numpy(dtype=float)
        low = subset["ci_low"].to_numpy(dtype=float)
        high = subset["ci_high"].to_numpy(dtype=float)
        color = colors.get(system_name, None)
        ax.plot(x, y, marker="o", linewidth=2.2, label=system_name, color=color)
        ax.fill_between(x, low, high, alpha=0.18, color=color)

    ax.set_xlabel("Relative Window Position (%)")
    ax.set_ylabel("Window-Level Accuracy")
    ax.set_title("Relative-Time Accuracy Curve")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_df


def _plot_prefix_accuracy(frame: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    med_evo = frame[frame["system"] == "MedEvo"].copy()
    if med_evo.empty:
        raise ValueError("Prefix accuracy plot requires MedEvo rows.")

    med_evo = med_evo.sort_values(["patient_id", "relative_time", "window_index"]).reset_index(drop=True)
    med_evo["cum_correct"] = med_evo.groupby("patient_id")["is_correct"].cumsum()
    med_evo["cum_count"] = med_evo.groupby("patient_id").cumcount() + 1
    med_evo["prefix_accuracy"] = med_evo["cum_correct"] / med_evo["cum_count"]

    thresholds = np.linspace(0.0, 1.0, NUM_TIME_BINS + 1)
    mean_rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        patient_values: List[float] = []
        for _, patient_df in med_evo.groupby("patient_id"):
            valid = patient_df[patient_df["relative_time"] <= threshold]
            if valid.empty:
                continue
            patient_values.append(float(valid["prefix_accuracy"].iloc[-1]))
        mean_rows.append(
            {
                "relative_time_pct": float(threshold * 100.0),
                "num_patients": int(len(patient_values)),
                "cohort_mean_prefix_accuracy": float(np.mean(patient_values)) if patient_values else float("nan"),
            }
        )
    mean_df = pd.DataFrame(mean_rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    for _, patient_df in med_evo.groupby("patient_id"):
        x = patient_df["relative_time"].to_numpy(dtype=float) * 100.0
        y = patient_df["prefix_accuracy"].to_numpy(dtype=float)
        ax.plot(x, y, color="#1f77b4", alpha=0.18, linewidth=1.0)

    ax.plot(
        mean_df["relative_time_pct"].to_numpy(dtype=float),
        mean_df["cohort_mean_prefix_accuracy"].to_numpy(dtype=float),
        color="#0b3c8c",
        linewidth=3.0,
        marker="o",
        label="Cohort Mean (MedEvo)",
    )
    ax.set_xlabel("Relative Time (%)")
    ax.set_ylabel("Cumulative Accuracy")
    ax.set_title("Prefix (Cumulative) Accuracy Curve")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return mean_df


def _plot_relative_time_accuracy_by_mode(
    curves_by_mode: Dict[str, pd.DataFrame],
    output_path: Path,
) -> pd.DataFrame:
    if not curves_by_mode:
        raise ValueError("No mode curves provided for relative-time comparison plot.")

    fig, ax = plt.subplots(figsize=(10, 5))
    rows: List[Dict[str, Any]] = []
    for mode_name in sorted(curves_by_mode.keys()):
        curve_df = curves_by_mode[mode_name]
        med_evo_curve = curve_df[curve_df["system"] == "MedEvo"].copy()
        if med_evo_curve.empty:
            continue
        ordered = med_evo_curve.sort_values("time_bin").reset_index(drop=True)
        x = ordered["time_bin_mid_pct"].to_numpy(dtype=float)
        y = ordered["accuracy"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.2, label=str(mode_name))
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "mode": str(mode_name),
                    "time_bin": int(row["time_bin"]),
                    "time_bin_mid_pct": float(row["time_bin_mid_pct"]),
                    "n_windows": int(row["n_windows"]),
                    "accuracy": float(row["accuracy"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                }
            )

    ax.set_xlabel("Relative Window Position (%)")
    ax.set_ylabel("Window-Level Accuracy")
    ax.set_title("Relative-Time Accuracy by Mode")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        raise ValueError("No rows generated for relative-time mode comparison plot.")
    return comparison_df


def _plot_prefix_accuracy_by_mode(
    curves_by_mode: Dict[str, pd.DataFrame],
    output_path: Path,
) -> pd.DataFrame:
    if not curves_by_mode:
        raise ValueError("No mode curves provided for prefix comparison plot.")

    fig, ax = plt.subplots(figsize=(10, 5))
    rows: List[Dict[str, Any]] = []
    for mode_name in sorted(curves_by_mode.keys()):
        curve_df = curves_by_mode[mode_name]
        if curve_df.empty:
            continue
        ordered = curve_df.sort_values("relative_time_pct").reset_index(drop=True)
        x = ordered["relative_time_pct"].to_numpy(dtype=float)
        y = ordered["cohort_mean_prefix_accuracy"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.2, label=str(mode_name))
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "mode": str(mode_name),
                    "relative_time_pct": float(row["relative_time_pct"]),
                    "num_patients": int(row["num_patients"]),
                    "cohort_mean_prefix_accuracy": float(row["cohort_mean_prefix_accuracy"]),
                }
            )

    ax.set_xlabel("Relative Time (%)")
    ax.set_ylabel("Cumulative Accuracy")
    ax.set_title("Prefix Cumulative Accuracy by Mode")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        raise ValueError("No rows generated for prefix mode comparison plot.")
    return comparison_df


def _plot_time_memory_heatmap(frame: pd.DataFrame, output_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    med_evo = frame[frame["system"] == "MedEvo"].copy()
    if med_evo.empty:
        raise ValueError("Time-memory heatmap requires MedEvo rows.")
    if med_evo["memory_depth"].isna().all():
        raise ValueError("Time-memory heatmap requires non-empty MedEvo memory depth values.")

    bin_codes, bin_labels = _memory_depth_bins(med_evo["memory_depth"])
    med_evo["memory_depth_bin"] = bin_codes
    valid = med_evo.dropna(subset=["memory_depth_bin"]).copy()
    valid["memory_depth_bin"] = valid["memory_depth_bin"].astype(int)

    mean_table = (
        valid.pivot_table(
            index="memory_depth_bin",
            columns="time_bin",
            values="is_correct",
            aggfunc="mean",
        )
        .sort_index(ascending=True)
        .reindex(columns=list(range(NUM_TIME_BINS)))
    )
    count_table = (
        valid.pivot_table(
            index="memory_depth_bin",
            columns="time_bin",
            values="is_correct",
            aggfunc="count",
        )
        .sort_index(ascending=True)
        .reindex(columns=list(range(NUM_TIME_BINS)))
        .fillna(0)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    heatmap_data = mean_table.to_numpy(dtype=float)
    image = ax.imshow(heatmap_data, aspect="auto", origin="lower", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Accuracy")

    x_tick_labels = [f"{i * 10}-{(i + 1) * 10}%" for i in range(NUM_TIME_BINS)]
    y_tick_labels = [bin_labels[i] if i < len(bin_labels) else str(i) for i in mean_table.index.tolist()]
    ax.set_xticks(np.arange(NUM_TIME_BINS))
    ax.set_xticklabels(x_tick_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(y_tick_labels)))
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel("Relative-Time Decile")
    ax.set_ylabel("Memory-Depth Bin")
    ax.set_title("Time × Memory Accuracy Heatmap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    accuracy_long = mean_table.reset_index().melt(
        id_vars=["memory_depth_bin"], var_name="time_bin", value_name="accuracy"
    )
    counts_long = count_table.reset_index().melt(
        id_vars=["memory_depth_bin"], var_name="time_bin", value_name="count"
    )
    merged = accuracy_long.merge(counts_long, on=["memory_depth_bin", "time_bin"], how="left")
    merged["memory_depth_label"] = merged["memory_depth_bin"].map(
        lambda value: bin_labels[int(value)] if int(value) < len(bin_labels) else str(value)
    )
    return merged, med_evo[["patient_id", "window_index", "time_bin", "memory_depth", "is_correct"]].copy()


def _save_metrics(
    output_path: Path,
    *,
    med_evo_metrics: Dict[str, Any],
    baseline_metrics: Optional[Dict[str, Any]],
    med_evo_frame: pd.DataFrame,
    baseline_frame: Optional[pd.DataFrame],
) -> None:
    payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "systems": {
            "MedEvo": med_evo_metrics,
        },
        "coverage": {
            "med_evo_patients": int(med_evo_frame["patient_id"].nunique()),
            "med_evo_windows": int(len(med_evo_frame)),
        },
    }
    if baseline_metrics is not None and baseline_frame is not None:
        payload["systems"]["Baseline"] = baseline_metrics
        payload["coverage"]["baseline_patients"] = int(baseline_frame["patient_id"].nunique())
        payload["coverage"]["baseline_windows"] = int(len(baseline_frame))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _evaluate_single_prediction_dir(
    *,
    prediction_dir: Path,
    oracle_labels: Dict[str, Dict[int, WindowLabel]],
    output_dir: Path,
    baseline_root: Optional[Path],
) -> Dict[str, Any]:
    med_evo_frame = _load_prediction_rows(
        prediction_dir=prediction_dir,
        oracle_labels=oracle_labels,
        system_name="MedEvo",
        include_memory_depth=True,
    )
    baseline_frame: Optional[pd.DataFrame] = None
    if baseline_root is not None:
        baseline_frame = _load_prediction_rows(
            prediction_dir=baseline_root,
            oracle_labels=oracle_labels,
            system_name="Baseline",
            include_memory_depth=False,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    med_evo_metrics = _compute_accuracy_metrics(med_evo_frame)
    baseline_metrics = _compute_accuracy_metrics(baseline_frame) if baseline_frame is not None else None

    all_frames = [med_evo_frame]
    if baseline_frame is not None:
        all_frames.append(baseline_frame)
    combined_frame = pd.concat(all_frames, ignore_index=True)

    metrics_path = output_dir / "metrics.json"
    _save_metrics(
        metrics_path,
        med_evo_metrics=med_evo_metrics,
        baseline_metrics=baseline_metrics,
        med_evo_frame=med_evo_frame,
        baseline_frame=baseline_frame,
    )

    combined_frame.sort_values(["system", "patient_id", "window_index"]).to_csv(
        output_dir / "window_level_matches.csv", index=False
    )

    relative_curve_df = _plot_relative_time_accuracy(
        combined_frame,
        output_path=output_dir / "plot1_relative_time_accuracy_curve.png",
    )
    relative_curve_df.to_csv(output_dir / "plot1_relative_time_accuracy_curve.csv", index=False)

    prefix_curve_df = _plot_prefix_accuracy(
        combined_frame,
        output_path=output_dir / "plot2_prefix_cumulative_accuracy_curve.png",
    )
    prefix_curve_df.to_csv(output_dir / "plot2_prefix_cumulative_accuracy_curve.csv", index=False)

    heatmap_long_df, med_evo_heatmap_source = _plot_time_memory_heatmap(
        combined_frame,
        output_path=output_dir / "plot3_time_memory_accuracy_heatmap.png",
    )
    heatmap_long_df.to_csv(output_dir / "plot3_time_memory_accuracy_heatmap.csv", index=False)
    med_evo_heatmap_source.to_csv(output_dir / "plot3_time_memory_source_rows.csv", index=False)

    print(f"Saved evaluation outputs to: {output_dir}")
    print("Macro ACC:")
    print(
        f"  MedEvo macro={med_evo_metrics['macro_acc']:.4f} "
        f"(patients={med_evo_metrics['num_patients']}, windows={med_evo_metrics['num_windows']})"
    )
    if baseline_metrics is not None:
        print(
            f"  Baseline macro={baseline_metrics['macro_acc']:.4f} "
            f"(patients={baseline_metrics['num_patients']}, windows={baseline_metrics['num_windows']})"
        )

    return {
        "relative_curve_df": relative_curve_df,
        "prefix_curve_df": prefix_curve_df,
        "output_dir": output_dir,
    }


def run_evaluation(
    *,
    med_evo_dir: Path,
    oracle_dir: Path,
    output_dir: Optional[Path],
    baseline_dir: Optional[Path],
) -> None:
    oracle_root = _ensure_oracle_dir(oracle_dir)
    oracle_labels = _load_oracle_labels(oracle_root)

    mode_root_candidates: List[Path] = [med_evo_dir]
    for root_name in PATIENT_STATUS_ROOT_NAMES:
        nested_candidate = med_evo_dir / root_name
        if nested_candidate not in mode_root_candidates:
            mode_root_candidates.append(nested_candidate)

    discovered_mode_dirs: Dict[str, Path] = {}
    resolved_mode_root: Optional[Path] = None
    for candidate in mode_root_candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        candidate_modes = _discover_patient_status_mode_dirs(candidate)
        if candidate_modes:
            discovered_mode_dirs = candidate_modes
            resolved_mode_root = candidate
            break

    if discovered_mode_dirs:
        if baseline_dir is not None:
            raise ValueError(
                "--baseline-dir is not supported when --med-evo-dir points to a multi-mode patient-status root."
            )
        if resolved_mode_root is None:
            raise ValueError(f"Failed to resolve multi-mode root from {med_evo_dir}")
        output_root = (
            output_dir
            if output_dir is not None
            else _default_output_root_from_patient_status_input(resolved_mode_root)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        print(
            "Detected patient-status modes: "
            + ", ".join(sorted(discovered_mode_dirs.keys()))
            + f" under {resolved_mode_root}"
        )
        relative_curves_by_mode: Dict[str, pd.DataFrame] = {}
        prefix_curves_by_mode: Dict[str, pd.DataFrame] = {}

        for mode in sorted(discovered_mode_dirs.keys()):
            mode_prediction_dir = discovered_mode_dirs[mode]
            mode_output_dir = output_root / mode
            print(f"=== Evaluating mode: {mode} ===")
            mode_result = _evaluate_single_prediction_dir(
                prediction_dir=mode_prediction_dir,
                oracle_labels=oracle_labels,
                output_dir=mode_output_dir,
                baseline_root=None,
            )
            relative_curves_by_mode[mode] = mode_result["relative_curve_df"]
            prefix_curves_by_mode[mode] = mode_result["prefix_curve_df"]

        if len(relative_curves_by_mode) > 1:
            combined_relative_df = _plot_relative_time_accuracy_by_mode(
                relative_curves_by_mode,
                output_path=output_root / "plot1_relative_time_accuracy_curve_by_mode.png",
            )
            combined_relative_df.to_csv(
                output_root / "plot1_relative_time_accuracy_curve_by_mode.csv", index=False
            )
        if len(prefix_curves_by_mode) > 1:
            combined_prefix_df = _plot_prefix_accuracy_by_mode(
                prefix_curves_by_mode,
                output_path=output_root / "plot2_prefix_cumulative_accuracy_curve_by_mode.png",
            )
            combined_prefix_df.to_csv(
                output_root / "plot2_prefix_cumulative_accuracy_curve_by_mode.csv", index=False
            )

        print(f"Saved multi-mode evaluation outputs to: {output_root}")
        return

    med_evo_root = _ensure_predictions_dir(med_evo_dir)
    baseline_root = _ensure_predictions_dir(baseline_dir) if baseline_dir is not None else None
    final_output_dir = output_dir if output_dir is not None else _default_output_dir_from_med_evo_root(med_evo_root)
    _evaluate_single_prediction_dir(
        prediction_dir=med_evo_root,
        oracle_labels=oracle_labels,
        output_dir=final_output_dir,
        baseline_root=baseline_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate patient-status predictions against Oracle labels.")
    parser.add_argument(
        "--med-evo-dir",
        type=str,
        required=True,
        help=(
            "Patient-status prediction directory for MedEvo. "
            "Supports single-mode input (<dir>/patients) or multi-mode root "
            "(<dir>/{memory,full_history_events,local_events_only})."
        ),
    )
    parser.add_argument(
        "--oracle-dir",
        type=str,
        required=True,
        help="Oracle output directory containing patients/*/oracle_predictions.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where metrics and plots will be saved. "
            "Default single-mode: evaluation_results/<memory_run_name>/<experiment_name>. "
            "Default multi-mode: evaluation_results/<memory_run_name>/<patient_status_root>/<mode>."
        ),
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        required=False,
        help="Optional patient-status prediction directory for baseline comparison (single-mode only).",
    )
    args = parser.parse_args()

    run_evaluation(
        med_evo_dir=Path(args.med_evo_dir),
        oracle_dir=Path(args.oracle_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        baseline_dir=Path(args.baseline_dir) if args.baseline_dir else None,
    )


if __name__ == "__main__":
    main()
