"""
Analysis Script for Survival Prediction Experiments

This script analyzes the results from survival prediction experiments,
providing visualizations and insights into:
- Prediction stability over time
- Confidence patterns
- Patient status trajectories
- Prediction flips and changes
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_experiment_results(results_dir: Path) -> Dict:
    """
    Load experiment results from directory.

    Args:
        results_dir: Path to results directory

    Returns:
        Dictionary with aggregate and individual results
    """
    # Load aggregate results
    aggregate_file = results_dir / "aggregate_results.json"
    if not aggregate_file.exists():
        raise FileNotFoundError(f"Aggregate results not found: {aggregate_file}")

    with open(aggregate_file, "r") as f:
        aggregate_results = json.load(f)

    return aggregate_results


def plot_prediction_trajectory(patient_results: Dict, output_path: Optional[str] = None):
    """
    Plot prediction trajectory for a single patient.

    Shows:
    - Predicted outcome over time
    - Confidence scores
    - Final actual outcome

    Args:
        patient_results: Results dictionary for one patient
        output_path: Path to save plot (optional)
    """
    predictions = patient_results.get("all_predictions", [])
    actual_outcome = patient_results.get("actual_outcome")
    subject_id = patient_results.get("subject_id")
    icu_stay_id = patient_results.get("icu_stay_id")

    if not predictions:
        print(f"No predictions found for patient {subject_id}")
        return

    # Extract data
    windows = [p.get("window_index", i) for i, p in enumerate(predictions)]
    hours = [p.get("hours_since_admission", 0) for p in predictions]
    outcomes = [p.get("survival_prediction", {}).get("outcome", "unknown") for p in predictions]
    confidences = [p.get("survival_prediction", {}).get("confidence", 0.0) for p in predictions]

    # Convert outcomes to numeric (1 = survive, 0 = die)
    outcome_numeric = [1 if o == "survive" else 0 if o == "die" else 0.5 for o in outcomes]
    actual_numeric = 1 if actual_outcome == "survive" else 0

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Predicted outcome over time
    ax1.plot(windows, outcome_numeric, marker="o", linewidth=2, markersize=8, label="Predicted Outcome")
    ax1.axhline(y=actual_numeric, color="red", linestyle="--", linewidth=2, label=f"Actual: {actual_outcome.upper()}")
    ax1.set_ylabel("Predicted Outcome", fontsize=12)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["DIE", "SURVIVE"])
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_title(
        f"Prediction Trajectory - Patient {subject_id} (ICU Stay {icu_stay_id})\n"
        f"Final Prediction: {outcomes[-1].upper()} | Actual: {actual_outcome.upper()}",
        fontsize=14,
        fontweight="bold",
    )

    # Plot 2: Confidence over time
    ax2.plot(windows, confidences, marker="s", linewidth=2, markersize=8, color="green", label="Confidence")
    ax2.fill_between(windows, confidences, alpha=0.3, color="green")
    ax2.set_xlabel("Window Index", fontsize=12)
    ax2.set_ylabel("Confidence Score", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved prediction trajectory plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_patient_status_trajectory(patient_results: Dict, output_path: Optional[str] = None):
    """
    Plot patient status trajectory over time.

    Shows:
    - Severity score
    - Trajectory assessment (improving/stable/deteriorating)

    Args:
        patient_results: Results dictionary for one patient
        output_path: Path to save plot (optional)
    """
    predictions = patient_results.get("all_predictions", [])
    actual_outcome = patient_results.get("actual_outcome")
    subject_id = patient_results.get("subject_id")

    if not predictions:
        return

    # Extract data
    windows = [p.get("window_index", i) for i, p in enumerate(predictions)]
    severity_scores = [p.get("patient_status_prediction", {}).get("severity_score", 0.0) for p in predictions]
    trajectories = [p.get("patient_status_prediction", {}).get("trajectory", "unknown") for p in predictions]

    # Convert trajectory to numeric
    trajectory_map = {"improving": 1, "stable": 0, "deteriorating": -1, "unknown": 0}
    trajectory_numeric = [trajectory_map.get(t, 0) for t in trajectories]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Severity score
    ax1.plot(windows, severity_scores, marker="o", linewidth=2, markersize=8, color="purple")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.fill_between(windows, severity_scores, alpha=0.3, color="purple")
    ax1.set_ylabel("Severity Score", fontsize=12)
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        f"Patient Status Trajectory - Patient {subject_id}\n" f"Actual Outcome: {actual_outcome.upper()}",
        fontsize=14,
        fontweight="bold",
    )

    # Plot 2: Trajectory assessment
    colors = ["red" if t == -1 else "yellow" if t == 0 else "green" for t in trajectory_numeric]
    ax2.bar(windows, [1] * len(windows), color=colors, alpha=0.6, edgecolor="black")
    ax2.set_xlabel("Window Index", fontsize=12)
    ax2.set_ylabel("Trajectory", fontsize=12)
    ax2.set_yticks([])
    ax2.set_ylim(0, 1.2)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", alpha=0.6, label="Improving"),
        Patch(facecolor="yellow", alpha=0.6, label="Stable"),
        Patch(facecolor="red", alpha=0.6, label="Deteriorating"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved patient status plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def analyze_prediction_stability(patient_results: Dict) -> Dict:
    """
    Analyze prediction stability for a patient.

    Returns:
        Dictionary with stability metrics
    """
    predictions = patient_results.get("all_predictions", [])

    if len(predictions) < 2:
        return {"num_predictions": len(predictions), "num_flips": 0, "stability_score": 1.0}

    # Count prediction flips
    outcomes = [p.get("survival_prediction", {}).get("outcome", "unknown") for p in predictions]
    num_flips = sum(1 for i in range(1, len(outcomes)) if outcomes[i] != outcomes[i - 1])

    # Calculate stability score (1.0 = no flips, 0.0 = flip every time)
    stability_score = 1.0 - (num_flips / (len(predictions) - 1))

    # Confidence trend
    confidences = [p.get("survival_prediction", {}).get("confidence", 0.0) for p in predictions]
    confidence_trend = "increasing" if confidences[-1] > confidences[0] else "decreasing"

    return {
        "num_predictions": len(predictions),
        "num_flips": num_flips,
        "stability_score": stability_score,
        "confidence_trend": confidence_trend,
        "initial_confidence": confidences[0] if confidences else 0.0,
        "final_confidence": confidences[-1] if confidences else 0.0,
    }


def plot_aggregate_analysis(aggregate_results: Dict, output_dir: Path):
    """
    Create aggregate analysis plots across all patients.

    Args:
        aggregate_results: Aggregate results dictionary
        output_dir: Directory to save plots
    """
    individual_results = aggregate_results.get("individual_results", [])

    if not individual_results:
        print("No individual results found for aggregate analysis")
        return

    # Analyze each patient
    stability_metrics = []
    for result in individual_results:
        metrics = analyze_prediction_stability(result)
        metrics["subject_id"] = result.get("subject_id")
        metrics["is_correct"] = result.get("is_correct")
        metrics["actual_outcome"] = result.get("actual_outcome")
        stability_metrics.append(metrics)

    df = pd.DataFrame(stability_metrics)

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Stability score distribution
    ax = axes[0, 0]
    correct_stability = df[df["is_correct"] == True]["stability_score"]
    incorrect_stability = df[df["is_correct"] == False]["stability_score"]

    ax.hist([correct_stability, incorrect_stability], bins=10, label=["Correct", "Incorrect"], alpha=0.7, color=["green", "red"])
    ax.set_xlabel("Stability Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Prediction Stability Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Number of prediction flips
    ax = axes[0, 1]
    flip_counts = df["num_flips"].value_counts().sort_index()
    ax.bar(flip_counts.index, flip_counts.values, color="steelblue", alpha=0.7)
    ax.set_xlabel("Number of Prediction Flips", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Prediction Flip Distribution", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Confidence progression
    ax = axes[1, 0]
    ax.scatter(df["initial_confidence"], df["final_confidence"], c=df["is_correct"].map({True: "green", False: "red"}), alpha=0.6, s=100)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("Initial Confidence", fontsize=12)
    ax.set_ylabel("Final Confidence", fontsize=12)
    ax.set_title("Confidence Progression", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Plot 4: Accuracy by stability
    ax = axes[1, 1]
    stability_bins = pd.cut(df["stability_score"], bins=[0, 0.5, 0.75, 1.0], labels=["Low", "Medium", "High"])
    accuracy_by_stability = df.groupby(stability_bins)["is_correct"].mean()
    ax.bar(range(len(accuracy_by_stability)), accuracy_by_stability.values, color="orange", alpha=0.7)
    ax.set_xticks(range(len(accuracy_by_stability)))
    ax.set_xticklabels(accuracy_by_stability.index)
    ax.set_xlabel("Stability Level", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Prediction Stability", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "aggregate_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved aggregate analysis plot to: {output_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("STABILITY ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nAverage Stability Score: {df['stability_score'].mean():.3f}")
    print(f"Average Number of Flips: {df['num_flips'].mean():.2f}")
    print(f"\nCorrect Predictions:")
    print(f"  Average Stability: {correct_stability.mean():.3f}")
    print(f"  Average Flips: {df[df['is_correct'] == True]['num_flips'].mean():.2f}")
    print(f"\nIncorrect Predictions:")
    print(f"  Average Stability: {incorrect_stability.mean():.3f}")
    print(f"  Average Flips: {df[df['is_correct'] == False]['num_flips'].mean():.2f}")


def main(results_dir: str):
    """
    Main analysis function.

    Args:
        results_dir: Path to experiment results directory
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_path}")
        return

    print("=" * 80)
    print("EXPERIMENT RESULTS ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing results from: {results_path}")

    # Load results
    print("\n1. Loading experiment results...")
    aggregate_results = load_experiment_results(results_path)
    individual_results = aggregate_results.get("individual_results", [])
    print(f"   Found {len(individual_results)} patient results")

    # Create analysis output directory
    analysis_dir = results_path / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    print(f"\n2. Saving analysis to: {analysis_dir}")

    # Generate individual patient plots
    print("\n3. Generating individual patient plots...")
    for i, result in enumerate(individual_results, 1):
        subject_id = result.get("subject_id")
        icu_stay_id = result.get("icu_stay_id")
        print(f"   [{i}/{len(individual_results)}] Patient {subject_id} (ICU Stay {icu_stay_id})")

        # Prediction trajectory plot
        pred_output = analysis_dir / f"prediction_trajectory_{subject_id}_{icu_stay_id}.png"
        plot_prediction_trajectory(result, output_path=str(pred_output))

        # Patient status plot
        status_output = analysis_dir / f"patient_status_{subject_id}_{icu_stay_id}.png"
        plot_patient_status_trajectory(result, output_path=str(status_output))

    # Generate aggregate analysis
    print("\n4. Generating aggregate analysis...")
    plot_aggregate_analysis(aggregate_results, analysis_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All plots saved to: {analysis_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_directory>")
        print("\nExample:")
        print("  python tests/analyze_results.py experiments/results-20260130_105708")
        sys.exit(1)

    main(sys.argv[1])
