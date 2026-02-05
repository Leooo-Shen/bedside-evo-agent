"""
Comparison Analysis: Blinded vs Unblinded Oracle Evaluation

This script compares the results from blinded and unblinded Oracle evaluations
to assess whether knowing the patient outcome biases the Oracle's scoring.

Usage:
    python compare_blinded_unblinded.py [result_dir]

    If result_dir is not provided, uses the most recent result-* folder.
"""

import json
import sys
from pathlib import Path
import glob

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def load_reports(file_path):
    """Load Oracle reports from JSON file."""
    with open(file_path, "r") as f:
        reports = json.load(f)
    return reports


def extract_scores(reports):
    """Extract patient_status_scores from reports."""
    return [r["patient_status_score"] for r in reports]


def extract_hours(reports):
    """Extract hours_since_admission from reports."""
    return [r["window_metadata"]["hours_since_admission"] for r in reports]


def calculate_statistics(scores):
    """Calculate statistics for a list of scores."""
    return {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "min": np.min(scores),
        "max": np.max(scores),
        "median": np.median(scores),
    }


def find_latest_result_dir():
    """Find the most recent result-* directory."""
    result_dirs = sorted(glob.glob("experiments/result-*"), reverse=True)
    if not result_dirs:
        return None
    return Path(result_dirs[0])


def find_report_files(results_dir):
    """
    Find both blinded and unblinded report files in the results directory.

    Returns:
        dict: Dictionary with keys 'died_unblinded', 'survived_unblinded',
              'died_blinded', 'survived_blinded' mapping to file paths
    """
    files = {}

    # Find unblinded files
    died_unblinded = list(results_dir.glob("died_patient_*_reports_unblinded.json"))
    surv_unblinded = list(results_dir.glob("survived_patient_*_reports_unblinded.json"))

    # Find blinded files
    died_blinded = list(results_dir.glob("died_patient_*_reports_blinded.json"))
    surv_blinded = list(results_dir.glob("survived_patient_*_reports_blinded.json"))

    if died_unblinded:
        files['died_unblinded'] = died_unblinded[0]
    if surv_unblinded:
        files['survived_unblinded'] = surv_unblinded[0]
    if died_blinded:
        files['died_blinded'] = died_blinded[0]
    if surv_blinded:
        files['survived_blinded'] = surv_blinded[0]

    return files


def main():
    """Main comparison analysis."""
    print("=" * 80)
    print("COMPARISON ANALYSIS: Blinded vs Unblinded Oracle Evaluation")
    print("=" * 80)

    # Determine results directory
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = find_latest_result_dir()
        if results_dir is None:
            print("\nERROR: No result directories found")
            print("Usage: python compare_blinded_unblinded.py [result_dir]")
            return

    if not results_dir.exists():
        print(f"\nERROR: Results directory not found: {results_dir}")
        return

    print(f"\nAnalyzing results from: {results_dir}")

    # Find report files
    files = find_report_files(results_dir)

    # Check if we have all required files
    required_keys = ['died_unblinded', 'survived_unblinded', 'died_blinded', 'survived_blinded']
    missing = [k for k in required_keys if k not in files]

    if missing:
        print(f"\nERROR: Missing report files for: {', '.join(missing)}")
        print("\nThis comparison requires both blinded and unblinded results.")
        print("Run both analyze_score_trajectory_blinded.py and analyze_score_trajectory_unblinded.py first.")
        return

    # Load unblinded results
    print("\nLoading unblinded results...")
    print(f"  Died: {files['died_unblinded'].name}")
    print(f"  Survived: {files['survived_unblinded'].name}")
    died_unblinded = load_reports(files['died_unblinded'])
    survived_unblinded = load_reports(files['survived_unblinded'])

    # Load blinded results
    print("\nLoading blinded results...")
    print(f"  Died: {files['died_blinded'].name}")
    print(f"  Survived: {files['survived_blinded'].name}")
    died_blinded = load_reports(files['died_blinded'])
    survived_blinded = load_reports(files['survived_blinded'])

    # Extract scores
    died_unblinded_scores = extract_scores(died_unblinded)
    survived_unblinded_scores = extract_scores(survived_unblinded)
    died_blinded_scores = extract_scores(died_blinded)
    survived_blinded_scores = extract_scores(survived_blinded)

    # Extract hours
    died_hours = extract_hours(died_unblinded)
    survived_hours = extract_hours(survived_unblinded)

    # Calculate statistics
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON")
    print("=" * 80)

    print("\n### Patient who DIED ###")
    print("\nUnblinded (outcome known):")
    died_unblinded_stats = calculate_statistics(died_unblinded_scores)
    for key, value in died_unblinded_stats.items():
        print(f"  {key}: {value:.3f}")

    print("\nBlinded (outcome hidden):")
    died_blinded_stats = calculate_statistics(died_blinded_scores)
    for key, value in died_blinded_stats.items():
        print(f"  {key}: {value:.3f}")

    print("\nDifference (Unblinded - Blinded):")
    print(f"  Mean difference: {died_unblinded_stats['mean'] - died_blinded_stats['mean']:.3f}")
    print(f"  Std difference: {died_unblinded_stats['std'] - died_blinded_stats['std']:.3f}")

    print("\n### Patient who SURVIVED ###")
    print("\nUnblinded (outcome known):")
    survived_unblinded_stats = calculate_statistics(survived_unblinded_scores)
    for key, value in survived_unblinded_stats.items():
        print(f"  {key}: {value:.3f}")

    print("\nBlinded (outcome hidden):")
    survived_blinded_stats = calculate_statistics(survived_blinded_scores)
    for key, value in survived_blinded_stats.items():
        print(f"  {key}: {value:.3f}")

    print("\nDifference (Unblinded - Blinded):")
    print(f"  Mean difference: {survived_unblinded_stats['mean'] - survived_blinded_stats['mean']:.3f}")
    print(f"  Std difference: {survived_unblinded_stats['std'] - survived_blinded_stats['std']:.3f}")

    # Calculate correlation between blinded and unblinded scores
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    died_correlation = np.corrcoef(died_unblinded_scores, died_blinded_scores)[0, 1]
    survived_correlation = np.corrcoef(survived_unblinded_scores, survived_blinded_scores)[0, 1]

    print(f"\nPatient who died - Correlation: {died_correlation:.3f}")
    print(f"Patient who survived - Correlation: {survived_correlation:.3f}")

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Died patient - trajectory comparison
    ax1 = axes[0, 0]
    ax1.plot(died_hours, died_unblinded_scores, "o-", color="darkred", label="Unblinded", linewidth=2, markersize=6)
    ax1.plot(died_hours, died_blinded_scores, "s--", color="lightcoral", label="Blinded", linewidth=2, markersize=6)
    ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Hours since ICU admission")
    ax1.set_ylabel("Patient Status Score")
    ax1.set_title("Patient who DIED: Blinded vs Unblinded")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)

    # Plot 2: Survived patient - trajectory comparison
    ax2 = axes[0, 1]
    ax2.plot(
        survived_hours, survived_unblinded_scores, "o-", color="darkgreen", label="Unblinded", linewidth=2, markersize=6
    )
    ax2.plot(
        survived_hours, survived_blinded_scores, "s--", color="lightgreen", label="Blinded", linewidth=2, markersize=6
    )
    ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Hours since ICU admission")
    ax2.set_ylabel("Patient Status Score")
    ax2.set_title("Patient who SURVIVED: Blinded vs Unblinded")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)

    # Plot 3: Scatter plot - Died patient
    ax3 = axes[1, 0]
    ax3.scatter(died_blinded_scores, died_unblinded_scores, alpha=0.6, s=50, color="red")
    ax3.plot([-1, 1], [-1, 1], "k--", alpha=0.3, label="Perfect agreement")
    ax3.set_xlabel("Blinded Score")
    ax3.set_ylabel("Unblinded Score")
    ax3.set_title(f"Patient who DIED: Score Agreement (r={died_correlation:.3f})")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1.1, 1.1)
    ax3.set_ylim(-1.1, 1.1)
    ax3.set_aspect("equal")

    # Plot 4: Scatter plot - Survived patient
    ax4 = axes[1, 1]
    ax4.scatter(survived_blinded_scores, survived_unblinded_scores, alpha=0.6, s=50, color="green")
    ax4.plot([-1, 1], [-1, 1], "k--", alpha=0.3, label="Perfect agreement")
    ax4.set_xlabel("Blinded Score")
    ax4.set_ylabel("Unblinded Score")
    ax4.set_title(f"Patient who SURVIVED: Score Agreement (r={survived_correlation:.3f})")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-1.1, 1.1)
    ax4.set_ylim(-1.1, 1.1)
    ax4.set_aspect("equal")

    plt.tight_layout()
    output_path = results_dir / "comparison_blinded_vs_unblinded.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nComparison plot saved to: {output_path}")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print("\n### Key Findings ###")

    # Check if there's significant bias
    died_mean_diff = abs(died_unblinded_stats["mean"] - died_blinded_stats["mean"])
    survived_mean_diff = abs(survived_unblinded_stats["mean"] - survived_blinded_stats["mean"])

    print(f"\n1. Mean Score Differences:")
    print(f"   - Patient who died: {died_mean_diff:.3f}")
    print(f"   - Patient who survived: {survived_mean_diff:.3f}")

    if died_mean_diff < 0.2 and survived_mean_diff < 0.2:
        print("   → MINIMAL BIAS: Differences are small (<0.2)")
    elif died_mean_diff < 0.4 and survived_mean_diff < 0.4:
        print("   → MODERATE BIAS: Differences are moderate (0.2-0.4)")
    else:
        print("   → SIGNIFICANT BIAS: Differences are large (>0.4)")

    print(f"\n2. Score Correlations:")
    print(f"   - Patient who died: r={died_correlation:.3f}")
    print(f"   - Patient who survived: r={survived_correlation:.3f}")

    avg_correlation = (died_correlation + survived_correlation) / 2
    if avg_correlation > 0.8:
        print("   → STRONG AGREEMENT: Blinded and unblinded scores are highly correlated")
    elif avg_correlation > 0.6:
        print("   → MODERATE AGREEMENT: Scores show moderate correlation")
    else:
        print("   → WEAK AGREEMENT: Scores diverge significantly")

    print("\n### Conclusion ###")
    if died_mean_diff < 0.3 and survived_mean_diff < 0.3 and avg_correlation > 0.7:
        print(
            "\n✓ The Oracle's evaluations appear ROBUST to outcome knowledge."
            "\n  Blinded and unblinded scores are similar, suggesting the Oracle"
            "\n  evaluates based on clinical trajectory patterns rather than"
            "\n  simply using the outcome as a shortcut."
        )
    else:
        print(
            "\n⚠ The Oracle's evaluations show OUTCOME BIAS."
            "\n  Significant differences between blinded and unblinded scores"
            "\n  suggest the Oracle may be influenced by knowing the final outcome."
        )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
