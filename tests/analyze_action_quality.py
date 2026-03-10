"""Experiment: Doctor action activity analysis for Oracle outputs.

This script correlates extracted doctor-action volume with the mapped patient status index.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATUS_TO_NUM = {
    "deteriorating": -1.0,
    "fluctuating": -0.5,
    "stable": 0.0,
    "improving": 1.0,
    "insufficient_data": 0.0,
}


def spearman_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    if len(x) != len(y) or len(x) < 2:
        return 0.0, 1.0

    x_rank = pd.Series(x).rank()
    y_rank = pd.Series(y).rank()

    n = len(x)
    x_mean = x_rank.mean()
    y_mean = y_rank.mean()

    numerator = sum((x_rank[i] - x_mean) * (y_rank[i] - y_mean) for i in range(n))
    denominator = np.sqrt(
        sum((x_rank[i] - x_mean) ** 2 for i in range(n)) * sum((y_rank[i] - y_mean) ** 2 for i in range(n))
    )
    if denominator == 0:
        return 0.0, 1.0

    rho = float(numerator / denominator)
    t_stat = rho * np.sqrt((n - 2) / (1 - rho**2)) if abs(rho) < 1 else 0
    p_value = float(2 * (1 - 0.5 * (1 + np.sign(t_stat) * np.sqrt(1 - 1 / (1 + t_stat**2 / (n - 2))))))
    return rho, p_value


def load_reports(file_path: Path) -> List[Dict]:
    with open(file_path, "r") as f:
        return json.load(f)


def extract_data(reports: List[Dict]) -> Tuple[List[float], List[float], List[int], List[List[str]]]:
    hours, status_values, action_counts, action_categories = [], [], [], []
    for r in reports:
        hours.append(float(r.get("window_metadata", {}).get("hours_since_admission", 0.0)))

        status = str(r.get("patient_status", {}).get("overall_status", "insufficient_data")).lower()
        status_values.append(STATUS_TO_NUM.get(status, 0.0))

        actions = r.get("doctor_actions", []) if isinstance(r.get("doctor_actions"), list) else []
        action_counts.append(len(actions))
        action_categories.append([str(a.get("category", "other")) for a in actions if isinstance(a, dict)])

    return hours, status_values, action_counts, action_categories


def _flatten_categories(category_lists: List[List[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for cats in category_lists:
        for c in cats:
            counts[c] = counts.get(c, 0) + 1
    return counts


def plot_combined_analysis(
    died_data: Tuple[List[float], List[float], List[int], List[List[str]]],
    survived_data: Tuple[List[float], List[float], List[int], List[List[str]]],
    output_path: Path,
    title_suffix: str = "",
) -> Tuple[float, float, float, float]:
    died_hours, died_status, died_actions, _ = died_data
    surv_hours, surv_status, surv_actions, _ = survived_data

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(died_hours, died_status, "o-", color="darkred", label="Status Index", linewidth=2)
    line2 = ax1_twin.plot(died_hours, died_actions, "s--", color="orange", label="#Doctor Actions", linewidth=2)
    ax1.set_xlabel("Hours since ICU admission")
    ax1.set_ylabel("Status Index", color="darkred")
    ax1_twin.set_ylabel("Action count", color="orange")
    ax1.set_title(f"Patient who DIED{title_suffix}")
    ax1.grid(True, alpha=0.2)
    lines = line1 + line2
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")

    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(surv_hours, surv_status, "o-", color="darkgreen", label="Status Index", linewidth=2)
    line2 = ax2_twin.plot(surv_hours, surv_actions, "s--", color="orange", label="#Doctor Actions", linewidth=2)
    ax2.set_xlabel("Hours since ICU admission")
    ax2.set_ylabel("Status Index", color="darkgreen")
    ax2_twin.set_ylabel("Action count", color="orange")
    ax2.set_title(f"Patient who SURVIVED{title_suffix}")
    ax2.grid(True, alpha=0.2)
    lines = line1 + line2
    ax2.legend(lines, [l.get_label() for l in lines], loc="upper left")

    ax3 = axes[1, 0]
    ax3.scatter(died_actions, died_status, c=died_hours, cmap="Reds", s=90, alpha=0.7, edgecolors="black")
    ax3.set_xlabel("Doctor action count")
    ax3.set_ylabel("Status Index")
    ax3.grid(True, alpha=0.3)
    corr_died, p_died = spearman_correlation(died_actions, died_status)
    ax3.set_title(f"DIED correlation (ρ={corr_died:.3f}, p={p_died:.3f})")

    ax4 = axes[1, 1]
    ax4.scatter(surv_actions, surv_status, c=surv_hours, cmap="Greens", s=90, alpha=0.7, edgecolors="black")
    ax4.set_xlabel("Doctor action count")
    ax4.set_ylabel("Status Index")
    ax4.grid(True, alpha=0.3)
    corr_surv, p_surv = spearman_correlation(surv_actions, surv_status)
    ax4.set_title(f"SURVIVED correlation (ρ={corr_surv:.3f}, p={p_surv:.3f})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    return corr_died, p_died, corr_surv, p_surv


def find_latest_result_dir() -> Path | None:
    result_dirs = sorted(glob.glob("experiments/result-*"), reverse=True)
    if not result_dirs:
        return None
    return Path(result_dirs[0])


def find_report_files(results_dir: Path, mode: str) -> Tuple[Path | None, Path | None]:
    died_files = list(results_dir.glob(f"died_patient_*_reports_{mode}.json"))
    surv_files = list(results_dir.glob(f"survived_patient_*_reports_{mode}.json"))
    if not died_files or not surv_files:
        return None, None
    return died_files[0], surv_files[0]


def main() -> None:
    print("=" * 80)
    print("DOCTOR ACTION ACTIVITY ANALYSIS (Oracle)")
    print("=" * 80)

    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = find_latest_result_dir()
        if results_dir is None:
            print("\nERROR: No result directories found")
            return

    if not results_dir.exists():
        print(f"\nERROR: Results directory not found: {results_dir}")
        return

    print(f"\nAnalyzing results from: {results_dir}")

    mode_labels = {
        "with_outcome": "WITH ICU OUTCOME IN PROMPT",
        "without_outcome": "WITHOUT ICU OUTCOME IN PROMPT",
    }
    for mode in ["with_outcome", "without_outcome"]:
        print(f"\n{'=' * 80}")
        print(f"ANALYZING ORACLE: {mode_labels.get(mode, mode.upper())}")
        print("=" * 80)

        died_file, surv_file = find_report_files(results_dir, mode)
        if not died_file or not surv_file:
            print(f"\n⚠ Skipping {mode} analysis: report files not found")
            continue

        died_reports = load_reports(died_file)
        surv_reports = load_reports(surv_file)

        died_data = extract_data(died_reports)
        surv_data = extract_data(surv_reports)

        died_cat = _flatten_categories(died_data[3])
        surv_cat = _flatten_categories(surv_data[3])

        print("\nDoctor action category counts (DIED):", died_cat)
        print("Doctor action category counts (SURVIVED):", surv_cat)

        output_path = results_dir / f"doctor_action_activity_analysis_{mode}.png"
        corr_died, p_died, corr_surv, p_surv = plot_combined_analysis(
            died_data,
            surv_data,
            output_path,
            title_suffix=f" ({mode_labels.get(mode, mode)})",
        )

        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)
        print(f"\nPatient who DIED: ρ={corr_died:.3f}, p={p_died:.4f}")
        print(f"Patient who SURVIVED: ρ={corr_surv:.3f}, p={p_surv:.4f}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
