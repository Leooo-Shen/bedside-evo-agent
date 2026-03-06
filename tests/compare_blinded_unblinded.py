"""Comparison analysis: blinded vs unblinded Oracle status trajectories."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

STATUS_TO_NUM = {
    "deteriorating": -1.0,
    "fluctuating": -0.5,
    "stable": 0.0,
    "improving": 1.0,
    "insufficient_data": 0.0,
}


def load_reports(file_path: Path) -> List[Dict]:
    with open(file_path, "r") as f:
        return json.load(f)


def extract_statuses(reports: List[Dict]) -> List[str]:
    statuses = []
    for r in reports:
        status = r.get("patient_status", {}).get("overall_status", "insufficient_data")
        statuses.append(str(status).lower())
    return statuses


def extract_status_values(reports: List[Dict]) -> List[float]:
    return [STATUS_TO_NUM.get(s, 0.0) for s in extract_statuses(reports)]


def extract_hours(reports: List[Dict]) -> List[float]:
    return [float(r.get("window_metadata", {}).get("hours_since_admission", 0.0)) for r in reports]


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    arr = np.asarray(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def find_latest_result_dir() -> Path | None:
    result_dirs = sorted(glob.glob("experiments/result-*"), reverse=True)
    if not result_dirs:
        return None
    return Path(result_dirs[0])


def find_report_files(results_dir: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}

    died_unblinded = list(results_dir.glob("died_patient_*_reports_unblinded.json"))
    surv_unblinded = list(results_dir.glob("survived_patient_*_reports_unblinded.json"))
    died_blinded = list(results_dir.glob("died_patient_*_reports_blinded.json"))
    surv_blinded = list(results_dir.glob("survived_patient_*_reports_blinded.json"))

    if died_unblinded:
        files["died_unblinded"] = died_unblinded[0]
    if surv_unblinded:
        files["survived_unblinded"] = surv_unblinded[0]
    if died_blinded:
        files["died_blinded"] = died_blinded[0]
    if surv_blinded:
        files["survived_blinded"] = surv_blinded[0]

    return files


def _status_distribution(statuses: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in statuses:
        counts[s] = counts.get(s, 0) + 1
    return counts


def main() -> None:
    print("=" * 80)
    print("COMPARISON ANALYSIS: Blinded vs Unblinded Oracle")
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

    files = find_report_files(results_dir)
    required = ["died_unblinded", "survived_unblinded", "died_blinded", "survived_blinded"]
    missing = [k for k in required if k not in files]
    if missing:
        print(f"\nERROR: Missing report files for: {', '.join(missing)}")
        return

    died_unblinded = load_reports(files["died_unblinded"])
    survived_unblinded = load_reports(files["survived_unblinded"])
    died_blinded = load_reports(files["died_blinded"])
    survived_blinded = load_reports(files["survived_blinded"])

    died_unblinded_values = extract_status_values(died_unblinded)
    survived_unblinded_values = extract_status_values(survived_unblinded)
    died_blinded_values = extract_status_values(died_blinded)
    survived_blinded_values = extract_status_values(survived_blinded)

    died_hours = extract_hours(died_unblinded)
    survived_hours = extract_hours(survived_unblinded)

    print("\n" + "=" * 80)
    print("STATUS DISTRIBUTIONS")
    print("=" * 80)

    print("\nPatient who DIED:")
    print("  Unblinded:", _status_distribution(extract_statuses(died_unblinded)))
    print("  Blinded:", _status_distribution(extract_statuses(died_blinded)))

    print("\nPatient who SURVIVED:")
    print("  Unblinded:", _status_distribution(extract_statuses(survived_unblinded)))
    print("  Blinded:", _status_distribution(extract_statuses(survived_blinded)))

    print("\n" + "=" * 80)
    print("NUMERIC STATUS INDEX COMPARISON")
    print("=" * 80)

    died_unblinded_stats = calculate_statistics(died_unblinded_values)
    died_blinded_stats = calculate_statistics(died_blinded_values)
    survived_unblinded_stats = calculate_statistics(survived_unblinded_values)
    survived_blinded_stats = calculate_statistics(survived_blinded_values)

    print("\nDIED patient mean index:")
    print(f"  Unblinded: {died_unblinded_stats['mean']:.3f}")
    print(f"  Blinded: {died_blinded_stats['mean']:.3f}")
    print(f"  Difference: {died_unblinded_stats['mean'] - died_blinded_stats['mean']:.3f}")

    print("\nSURVIVED patient mean index:")
    print(f"  Unblinded: {survived_unblinded_stats['mean']:.3f}")
    print(f"  Blinded: {survived_blinded_stats['mean']:.3f}")
    print(f"  Difference: {survived_unblinded_stats['mean'] - survived_blinded_stats['mean']:.3f}")

    died_corr = float(np.corrcoef(died_unblinded_values, died_blinded_values)[0, 1])
    survived_corr = float(np.corrcoef(survived_unblinded_values, survived_blinded_values)[0, 1])

    print("\nCorrelations:")
    print(f"  DIED: {died_corr:.3f}")
    print(f"  SURVIVED: {survived_corr:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax1 = axes[0, 0]
    ax1.plot(died_hours, died_unblinded_values, "o-", color="darkred", label="Unblinded", linewidth=2)
    ax1.plot(died_hours, died_blinded_values, "s--", color="lightcoral", label="Blinded", linewidth=2)
    ax1.set_xlabel("Hours since ICU admission")
    ax1.set_ylabel("Status Index (mapped)")
    ax1.set_title("Patient who DIED")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(survived_hours, survived_unblinded_values, "o-", color="darkgreen", label="Unblinded", linewidth=2)
    ax2.plot(survived_hours, survived_blinded_values, "s--", color="lightgreen", label="Blinded", linewidth=2)
    ax2.set_xlabel("Hours since ICU admission")
    ax2.set_ylabel("Status Index (mapped)")
    ax2.set_title("Patient who SURVIVED")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.scatter(died_blinded_values, died_unblinded_values, alpha=0.7, color="red")
    ax3.plot([-1, 1], [-1, 1], "k--", alpha=0.3)
    ax3.set_xlabel("Blinded")
    ax3.set_ylabel("Unblinded")
    ax3.set_title(f"DIED agreement (r={died_corr:.3f})")
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.scatter(survived_blinded_values, survived_unblinded_values, alpha=0.7, color="green")
    ax4.plot([-1, 1], [-1, 1], "k--", alpha=0.3)
    ax4.set_xlabel("Blinded")
    ax4.set_ylabel("Unblinded")
    ax4.set_title(f"SURVIVED agreement (r={survived_corr:.3f})")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = results_dir / "comparison_blinded_vs_unblinded.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nComparison plot saved to: {output_path}")


if __name__ == "__main__":
    main()
