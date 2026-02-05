"""
Experiment: Action Quality Analysis

This script analyzes the Oracle's action_quality assessments alongside
patient_status_score to identify correlations and patterns.

Usage:
    python analyze_action_quality.py [result_dir]

    If result_dir is not provided, uses the most recent result-* folder.
"""

import json
import sys
from pathlib import Path
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def spearman_correlation(x, y):
    """Calculate Spearman correlation coefficient manually."""
    # Convert to ranks
    x_rank = pd.Series(x).rank()
    y_rank = pd.Series(y).rank()

    # Calculate Pearson correlation on ranks
    n = len(x)
    x_mean = x_rank.mean()
    y_mean = y_rank.mean()

    numerator = sum((x_rank[i] - x_mean) * (y_rank[i] - y_mean) for i in range(n))
    denominator = np.sqrt(
        sum((x_rank[i] - x_mean) ** 2 for i in range(n)) * sum((y_rank[i] - y_mean) ** 2 for i in range(n))
    )

    if denominator == 0:
        return 0.0, 1.0

    rho = numerator / denominator

    # Simple p-value approximation (not exact but reasonable for n > 10)
    t_stat = rho * np.sqrt((n - 2) / (1 - rho**2)) if abs(rho) < 1 else 0
    # Approximate p-value using t-distribution approximation
    p_value = 2 * (1 - 0.5 * (1 + np.sign(t_stat) * np.sqrt(1 - 1 / (1 + t_stat**2 / (n - 2)))))

    return rho, p_value


def load_reports(file_path):
    """Load Oracle reports from JSON file."""
    with open(file_path, "r") as f:
        reports = json.load(f)
    return reports


def action_quality_to_numeric(action_quality):
    """Convert action quality to numeric score."""
    mapping = {"sub-optimal": -1, "neutral": 0, "optimal": 1}
    return mapping.get(action_quality, 0)


def extract_data(reports):
    """Extract relevant data from reports."""
    hours = [r["window_metadata"]["hours_since_admission"] for r in reports]
    scores = [r["patient_status_score"] for r in reports]
    action_qualities = [r["action_quality"] for r in reports]
    action_numeric = [action_quality_to_numeric(aq) for aq in action_qualities]

    return hours, scores, action_qualities, action_numeric


def plot_combined_analysis(died_data, survived_data, output_path, title_suffix=""):
    """
    Create comprehensive visualization of scores and action quality.

    Args:
        died_data: Tuple of (hours, scores, action_qualities, action_numeric) for patient who died
        survived_data: Tuple of (hours, scores, action_qualities, action_numeric) for patient who survived
        output_path: Path to save the plot
        title_suffix: Additional text for title
    """
    died_hours, died_scores, died_aq, died_aq_num = died_data
    surv_hours, surv_scores, surv_aq, surv_aq_num = survived_data

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Patient who died - Score and Action Quality
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()

    # Plot score
    line1 = ax1.plot(died_hours, died_scores, "o-", color="darkred", label="Status Score", linewidth=2, markersize=6)
    ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax1.set_xlabel("Hours since ICU admission")
    ax1.set_ylabel("Patient Status Score", color="darkred")
    ax1.tick_params(axis="y", labelcolor="darkred")
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, alpha=0.2)

    # Plot action quality
    line2 = ax1_twin.plot(
        died_hours, died_aq_num, "s--", color="orange", label="Action Quality", linewidth=2, markersize=6, alpha=0.7
    )
    ax1_twin.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax1_twin.set_ylabel("Action Quality", color="orange")
    ax1_twin.tick_params(axis="y", labelcolor="orange")
    ax1_twin.set_ylim(-1.5, 1.5)
    ax1_twin.set_yticks([-1, 0, 1])
    ax1_twin.set_yticklabels(["Sub-optimal", "Neutral", "Optimal"])

    ax1.set_title(f"Patient who DIED: Score vs Action Quality{title_suffix}")

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    # Plot 2: Patient who survived - Score and Action Quality
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()

    # Plot score
    line1 = ax2.plot(
        surv_hours, surv_scores, "o-", color="darkgreen", label="Status Score", linewidth=2, markersize=6
    )
    ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax2.set_xlabel("Hours since ICU admission")
    ax2.set_ylabel("Patient Status Score", color="darkgreen")
    ax2.tick_params(axis="y", labelcolor="darkgreen")
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.2)

    # Plot action quality
    line2 = ax2_twin.plot(
        surv_hours, surv_aq_num, "s--", color="orange", label="Action Quality", linewidth=2, markersize=6, alpha=0.7
    )
    ax2_twin.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax2_twin.set_ylabel("Action Quality", color="orange")
    ax2_twin.tick_params(axis="y", labelcolor="orange")
    ax2_twin.set_ylim(-1.5, 1.5)
    ax2_twin.set_yticks([-1, 0, 1])
    ax2_twin.set_yticklabels(["Sub-optimal", "Neutral", "Optimal"])

    ax2.set_title(f"Patient who SURVIVED: Score vs Action Quality{title_suffix}")

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left")

    # Plot 3: Scatter - Patient who died
    ax3 = axes[1, 0]
    scatter = ax3.scatter(died_aq_num, died_scores, c=died_hours, cmap="Reds", s=100, alpha=0.6, edgecolors="black")
    ax3.set_xlabel("Action Quality")
    ax3.set_ylabel("Patient Status Score")
    ax3.set_xticks([-1, 0, 1])
    ax3.set_xticklabels(["Sub-optimal", "Neutral", "Optimal"])
    ax3.set_ylim(-1.1, 1.1)
    ax3.grid(True, alpha=0.3)

    # Calculate correlation
    corr_died, p_died = spearman_correlation(died_aq_num, died_scores)
    ax3.set_title(f"Patient who DIED: Correlation (ρ={corr_died:.3f}, p={p_died:.3f})")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label("Hours since admission")

    # Plot 4: Scatter - Patient who survived
    ax4 = axes[1, 1]
    scatter = ax4.scatter(
        surv_aq_num, surv_scores, c=surv_hours, cmap="Greens", s=100, alpha=0.6, edgecolors="black"
    )
    ax4.set_xlabel("Action Quality")
    ax4.set_ylabel("Patient Status Score")
    ax4.set_xticks([-1, 0, 1])
    ax4.set_xticklabels(["Sub-optimal", "Neutral", "Optimal"])
    ax4.set_ylim(-1.1, 1.1)
    ax4.grid(True, alpha=0.3)

    # Calculate correlation
    corr_surv, p_surv = spearman_correlation(surv_aq_num, surv_scores)
    ax4.set_title(f"Patient who SURVIVED: Correlation (ρ={corr_surv:.3f}, p={p_surv:.3f})")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label("Hours since admission")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    return corr_died, p_died, corr_surv, p_surv


def analyze_action_quality_distribution(died_aq, survived_aq):
    """Analyze distribution of action quality assessments."""
    print("\n" + "=" * 80)
    print("ACTION QUALITY DISTRIBUTION")
    print("=" * 80)

    # Count occurrences
    died_counts = pd.Series(died_aq).value_counts()
    surv_counts = pd.Series(survived_aq).value_counts()

    print("\nPatient who DIED:")
    for quality in ["optimal", "neutral", "sub-optimal"]:
        count = died_counts.get(quality, 0)
        pct = (count / len(died_aq)) * 100
        print(f"  {quality:12s}: {count:3d} ({pct:5.1f}%)")

    print("\nPatient who SURVIVED:")
    for quality in ["optimal", "neutral", "sub-optimal"]:
        count = surv_counts.get(quality, 0)
        pct = (count / len(survived_aq)) * 100
        print(f"  {quality:12s}: {count:3d} ({pct:5.1f}%)")


def find_latest_result_dir():
    """Find the most recent result-* directory."""
    result_dirs = sorted(glob.glob("experiments/result-*"), reverse=True)
    if not result_dirs:
        return None
    return Path(result_dirs[0])


def find_report_files(results_dir, mode):
    """
    Find report files in the results directory for a given mode.

    Args:
        results_dir: Path to results directory
        mode: "blinded" or "unblinded"

    Returns:
        tuple: (died_file, survived_file) paths
    """
    # Find files with the mode suffix
    died_files = list(results_dir.glob(f"died_patient_*_reports_{mode}.json"))
    surv_files = list(results_dir.glob(f"survived_patient_*_reports_{mode}.json"))

    if not died_files or not surv_files:
        return None, None

    return died_files[0], surv_files[0]


def main():
    """Main analysis function."""
    print("=" * 80)
    print("ACTION QUALITY ANALYSIS")
    print("=" * 80)

    # Determine results directory
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = find_latest_result_dir()
        if results_dir is None:
            print("\nERROR: No result directories found")
            print("Usage: python analyze_action_quality.py [result_dir]")
            return

    if not results_dir.exists():
        print(f"\nERROR: Results directory not found: {results_dir}")
        return

    print(f"\nAnalyzing results from: {results_dir}")

    # Analyze both blinded and unblinded if available
    for mode in ["unblinded", "blinded"]:
        print(f"\n{'=' * 80}")
        print(f"ANALYZING {mode.upper()} ORACLE")
        print("=" * 80)

        # Find report files
        died_file, surv_file = find_report_files(results_dir, mode)

        if not died_file or not surv_file:
            print(f"\n⚠ Skipping {mode} analysis: Report files not found")
            continue

        print(f"\nLoading {mode} reports...")
        print(f"  Died patient: {died_file.name}")
        print(f"  Survived patient: {surv_file.name}")

        died_reports = load_reports(died_file)
        surv_reports = load_reports(surv_file)

        # Extract data
        died_data = extract_data(died_reports)
        surv_data = extract_data(surv_reports)

        # Analyze distribution
        analyze_action_quality_distribution(died_data[2], surv_data[2])

        # Create visualization
        output_path = results_dir / f"action_quality_analysis_{mode}.png"
        title_suffix = f" ({mode.capitalize()})"

        corr_died, p_died, corr_surv, p_surv = plot_combined_analysis(died_data, surv_data, output_path, title_suffix)

        # Print correlation analysis
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)

        print(f"\nPatient who DIED:")
        print(f"  Spearman correlation: ρ = {corr_died:.3f}")
        print(f"  P-value: {p_died:.4f}")
        if p_died < 0.05:
            print(f"  → Statistically significant correlation")
        else:
            print(f"  → No significant correlation")

        print(f"\nPatient who SURVIVED:")
        print(f"  Spearman correlation: ρ = {corr_surv:.3f}")
        print(f"  P-value: {p_surv:.4f}")
        if p_surv < 0.05:
            print(f"  → Statistically significant correlation")
        else:
            print(f"  → No significant correlation")

        # Interpretation
        print("\n" + "=" * 80)
        print("INTERPRETATION")
        print("=" * 80)

        avg_corr = (abs(corr_died) + abs(corr_surv)) / 2

        if avg_corr > 0.7:
            print("\n✓ STRONG CORRELATION: Action quality strongly correlates with patient status")
        elif avg_corr > 0.4:
            print("\n✓ MODERATE CORRELATION: Action quality moderately correlates with patient status")
        elif avg_corr > 0.2:
            print("\n⚠ WEAK CORRELATION: Action quality weakly correlates with patient status")
        else:
            print("\n✗ NO CORRELATION: Action quality does not correlate with patient status")

        print(
            f"\nAverage absolute correlation: {avg_corr:.3f}"
            f"\n\nThis suggests that the Oracle's assessment of action quality"
            f"\n{'is' if avg_corr > 0.4 else 'is not'} closely tied to its evaluation of patient status."
        )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
