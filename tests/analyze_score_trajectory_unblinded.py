"""Experiment: Analyze Oracle patient status trajectories (prompt includes ICU outcome)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from agents.oracle import MetaOracle, OracleReport, save_oracle_reports
from config.config import load_config
from data_parser import MIMICDataParser

STATUS_TO_NUM = {
    "deteriorating": -1.0,
    "fluctuating": -0.5,
    "stable": 0.0,
    "improving": 1.0,
    "insufficient_data": 0.0,
}


def _status_of(report: OracleReport) -> str:
    return report.patient_status.get("overall_status", "insufficient_data")


def _status_to_num(status: str) -> float:
    return STATUS_TO_NUM.get(status, 0.0)


def find_patients_by_outcome(parser: MIMICDataParser) -> Tuple[Dict, Dict]:
    print("Searching for patients with different outcomes...")
    died_patient = None
    survived_patient = None

    for _, icu_stay in parser.icu_stay_df.iterrows():
        try:
            trajectory = parser.get_patient_trajectory(icu_stay["subject_id"], icu_stay["icu_stay_id"])
            if not trajectory["survived"] and died_patient is None:
                died_patient = trajectory
                print(
                    f"  Found patient who died: Subject {trajectory['subject_id']}, ICU Stay {trajectory['icu_stay_id']}"
                )
            elif trajectory["survived"] and survived_patient is None:
                survived_patient = trajectory
                print(
                    f"  Found patient who survived: Subject {trajectory['subject_id']}, ICU Stay {trajectory['icu_stay_id']}"
                )
            if died_patient and survived_patient:
                break
        except Exception as e:
            print(f"  Error processing ICU stay {icu_stay['icu_stay_id']}: {e}")

    return died_patient, survived_patient


def run_oracle_on_patient(
    oracle: MetaOracle,
    parser: MIMICDataParser,
    trajectory: Dict,
    config,
    parallel: bool = True,
    max_workers: int = 10,
) -> List[OracleReport]:
    subject_id = trajectory["subject_id"]
    icu_stay_id = trajectory["icu_stay_id"]

    print(f"\nProcessing Patient {subject_id}, ICU Stay {icu_stay_id}")

    windows = parser.create_time_windows(
        trajectory,
        current_window_hours=config.oracle_current_window_hours,
        window_step_hours=config.oracle_window_step_hours,
        include_pre_icu_data=config.oracle_include_pre_icu_data,
        use_first_n_hours_after_icu=config.oracle_observation_hours,
        use_discharge_summary_for_history=config.oracle_use_discharge_summary_for_history,
        num_discharge_summaries=config.oracle_num_discharge_summaries,
    )

    print(f"  Generated {len(windows)} time windows")
    if len(windows) == 0:
        return []

    if parallel:
        print(f"  Using parallel evaluation with {max_workers} workers")
        reports = oracle.evaluate_trajectory_parallel(windows, trajectory=trajectory, max_workers=max_workers)
    else:
        reports = oracle.evaluate_trajectory(windows, trajectory=trajectory)

    print(f"  Completed: {len(reports)} evaluations")
    oracle.save_trajectory_log(subject_id, icu_stay_id, run_id="with_outcome_experiment")
    return reports


def plot_status_trajectories(died_reports: List[OracleReport], survived_reports: List[OracleReport], output_path: str) -> None:
    died_hours = [r.window_data["hours_since_admission"] for r in died_reports]
    died_status = [_status_of(r) for r in died_reports]
    died_values = [_status_to_num(s) for s in died_status]

    survived_hours = [r.window_data["hours_since_admission"] for r in survived_reports]
    survived_status = [_status_of(r) for r in survived_reports]
    survived_values = [_status_to_num(s) for s in survived_status]

    plt.figure(figsize=(12, 6))
    plt.plot(died_hours, died_values, "o-", color="red", label="Patient who died", linewidth=2, markersize=6)
    plt.plot(
        survived_hours,
        survived_values,
        "o-",
        color="green",
        label="Patient who survived",
        linewidth=2,
        markersize=6,
    )

    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Hours since ICU admission", fontsize=12)
    plt.ylabel("Status Index (mapped)", fontsize=12)
    plt.title("Oracle Status Trajectory: Survivors vs Non-Survivors (With Outcome)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.1, 1.1)

    plt.text(
        0.02,
        0.98,
        "Mapping: deteriorating=-1, fluctuating=-0.5, stable=0, improving=1",
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    print("\n" + "=" * 60)
    print("STATUS DISTRIBUTION")
    print("=" * 60)

    print("\nPatient who died:")
    for status in sorted(set(died_status)):
        print(f"  {status}: {died_status.count(status)}")

    print("\nPatient who survived:")
    for status in sorted(set(survived_status)):
        print(f"  {status}: {survived_status.count(status)}")


def main() -> None:
    print("=" * 80)
    print("EXPERIMENT: Oracle Status Trajectory Analysis (WITH ICU OUTCOME IN PROMPT)")
    print("=" * 80)

    timestamp = datetime.now().strftime("%m%d-%H%M")
    output_dir = Path(f"experiments/result-{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    config = load_config()

    print("\nInitializing data parser...")
    parser = MIMICDataParser(config.events_path, config.icu_stay_path)
    parser.load_data()

    died_patient, survived_patient = find_patients_by_outcome(parser)
    if not died_patient or not survived_patient:
        print("\nERROR: Could not find patients with both outcomes")
        return

    print("\nInitializing Meta Oracle...")
    oracle = MetaOracle(
        provider=config.llm_provider,
        model=config.llm_model,
        temperature=config.llm_temperature,
        include_icu_outcome_in_prompt=True,
        use_discharge_summary=config.oracle_context_use_discharge_summary,
        history_context_hours=config.oracle_context_history_hours,
        future_context_hours=config.oracle_context_future_hours,
    )

    print("\n" + "=" * 80)
    print("RUNNING ORACLE EVALUATIONS")
    print("=" * 80)

    died_reports = run_oracle_on_patient(oracle, parser, died_patient, config)
    survived_reports = run_oracle_on_patient(oracle, parser, survived_patient, config)

    if not died_reports or not survived_reports:
        print("\nERROR: Failed to generate reports for one or both patients")
        return

    died_output = output_dir / f"died_patient_{died_patient['subject_id']}_reports_with_outcome.json"
    survived_output = output_dir / f"survived_patient_{survived_patient['subject_id']}_reports_with_outcome.json"

    save_oracle_reports(died_reports, str(died_output), include_window_data=True)
    save_oracle_reports(survived_reports, str(survived_output), include_window_data=True)

    plot_path = output_dir / "status_trajectory_comparison_with_outcome.png"
    plot_status_trajectories(died_reports, survived_reports, str(plot_path))

    print("\n" + "=" * 80)
    print("ORACLE USAGE STATISTICS")
    print("=" * 80)
    stats = oracle.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
