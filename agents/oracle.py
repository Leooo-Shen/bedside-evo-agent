"""
Meta Oracle: Retrospective Clinical Evaluator

The Oracle evaluates clinical decisions with the benefit of hindsight by analyzing
patient trajectories and their outcomes.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.append(str(Path(__file__).parent.parent))

from model.llms import LLMClient
from prompts.oracle_prompt import format_oracle_prompt


class OracleReport:
    """Structured output from Oracle evaluation."""

    def __init__(
        self,
        patient_status_score: float,
        status_rationale: str,
        action_quality: str,
        action_quality_rationale: str,
        clinical_insight: str,
        primary_clinical_driver: str = None,
        guideline_adherence: str = None,
        window_data: Dict = None,
    ):
        """
        Initialize Oracle report.

        Args:
            patient_status_score: Score from -1.0 to 1.0 (-1.0 critically ill, 0 stable, 1.0 improving)
            status_rationale: Medical reasoning for the patient status score
            action_quality: Clinical appropriateness of interventions (optimal, neutral, sub-optimal)
            action_quality_rationale: Detailed explanation for action quality assessment
            clinical_insight: Transferable clinical pearl for the Agent's memory
            primary_clinical_driver: Short description of the main medical issue
            guideline_adherence: Reference to standard protocols (e.g., Surviving Sepsis, ACLS)
            window_data: Original window data for reference
        """
        self.patient_status_score = patient_status_score
        self.status_rationale = status_rationale
        self.action_quality = action_quality
        self.action_quality_rationale = action_quality_rationale
        self.clinical_insight = clinical_insight
        self.primary_clinical_driver = primary_clinical_driver
        self.guideline_adherence = guideline_adherence
        self.window_data = window_data

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            "patient_status_score": self.patient_status_score,
            "status_rationale": self.status_rationale,
            "action_quality": self.action_quality,
            "action_quality_rationale": self.action_quality_rationale,
            "clinical_insight": self.clinical_insight,
            "primary_clinical_driver": self.primary_clinical_driver,
            "guideline_adherence": self.guideline_adherence,
        }

    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict, window_data: Dict = None) -> "OracleReport":
        """
        Create report from dictionary.

        Handles both the new nested format and the old flat format for backwards compatibility.

        New format:
        {
          "audit_metadata": {"primary_clinical_driver": "..."},
          "patient_status": {"score": 0.0, "rationale": "..."},
          "clinical_quality": {"rating": "...", "rationale": "...", "guideline_adherence": "..."},
          "clinical_pearl": "..."
        }

        Old format:
        {
          "patient_status_score": 0.0,
          "status_rationale": "...",
          "action_quality": "...",
          "action_quality_rationale": "...",
          "clinical_insight": "..."
        }
        """
        # Check if this is the new nested format
        if "patient_status" in data and isinstance(data["patient_status"], dict):
            # New nested format
            return cls(
                patient_status_score=data["patient_status"]["score"],
                status_rationale=data["patient_status"]["rationale"],
                action_quality=data["clinical_quality"]["rating"],
                action_quality_rationale=data["clinical_quality"]["rationale"],
                clinical_insight=data.get("clinical_pearl", ""),
                primary_clinical_driver=data.get("audit_metadata", {}).get("primary_clinical_driver"),
                guideline_adherence=data.get("clinical_quality", {}).get("guideline_adherence"),
                window_data=window_data,
            )
        else:
            # Old flat format (backwards compatibility)
            return cls(
                patient_status_score=data["patient_status_score"],
                status_rationale=data["status_rationale"],
                action_quality=data["action_quality"],
                action_quality_rationale=data["action_quality_rationale"],
                clinical_insight=data["clinical_insight"],
                primary_clinical_driver=data.get("primary_clinical_driver"),
                guideline_adherence=data.get("guideline_adherence"),
                window_data=window_data,
            )


class MetaOracle:
    """
    Meta Oracle for retrospective clinical evaluation.

    The Oracle uses a high-performance LLM to generate ground truth evaluations
    by reviewing patient cases with complete hindsight.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        api_key: str = None,
        temperature: float = 0.3,  # Lower temperature for more consistent evaluations
        max_tokens: int = 4096,
        window_hours: float = 6.0,
        log_dir: Optional[str] = None,
        blinded: bool = False,  # Whether to exclude outcome from prompts
    ):
        """
        Initialize Meta Oracle.

        Args:
            provider: LLM provider ("anthropic" or "openai")
            model: Model name (defaults to best available)
            api_key: API key (if None, uses environment variable)
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            window_hours: Size of future window for evaluation
            log_dir: Directory to save input/output logs (if None, uses 'logs/oracle')
            blinded: Whether to exclude patient outcome from prompts (for unbiased evaluation)
        """
        self.llm_client = LLMClient(
            provider=provider, model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens
        )
        self.window_hours = window_hours
        self.evaluation_count = 0
        self.total_tokens_used = 0
        self._stats_lock = Lock()  # Thread-safe counter updates
        self.blinded = blinded
        self._trajectory_logs = []  # Store logs for batch saving

        # Setup logging
        if log_dir is None:
            log_dir = str(Path(__file__).parent.parent / "logs" / "oracle")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print(f"Oracle logs will be saved to: {self.log_dir}")
        if blinded:
            print("Oracle running in BLINDED mode (outcome excluded from prompts)")

    def _save_log(
        self, window_data: Dict, prompt: str, response: Dict, report: OracleReport, error: Optional[str] = None
    ):
        """
        Save input/output log for an oracle evaluation.
        Stores in memory for batch saving per trajectory.

        Args:
            window_data: The window data that was evaluated
            prompt: The input prompt sent to the LLM
            response: The raw response from the LLM
            report: The parsed OracleReport
            error: Any error message if evaluation failed
        """
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "evaluation_number": self.evaluation_count,
            "window_metadata": {
                "subject_id": window_data.get("subject_id"),
                "icu_stay_id": window_data.get("icu_stay_id"),
                "window_start_time": window_data.get("current_window_start"),
                "window_end_time": window_data.get("current_window_end"),
                "hours_since_admission": window_data.get("hours_since_admission"),
            },
            "input": {"prompt": prompt, "window_hours": self.window_hours},
            "output": {
                "raw_content": response.get("content"),
                "parsed_json": response.get("parsed"),
                "parse_error": response.get("parse_error"),
                "model": response.get("model"),
                "usage": response.get("usage"),
            },
            "report": report.to_dict() if report else None,
            "error": error,
        }

        # Store in memory for batch saving
        self._trajectory_logs.append(log_entry)

    def save_trajectory_log(self, subject_id: int, icu_stay_id: int, run_id: str = None):
        """
        Save all accumulated logs for a trajectory to a single file.

        Args:
            subject_id: Patient subject ID
            icu_stay_id: ICU stay ID
            run_id: Optional identifier for this run (e.g., "blinded", "unblinded", "experiment1")
        """
        if not self._trajectory_logs:
            return

        timestamp = datetime.now().isoformat().replace(":", "-")
        mode = "blinded" if self.blinded else "unblinded"

        # Build filename
        filename_parts = ["oracle_trajectory", str(subject_id), str(icu_stay_id), mode]
        if run_id:
            filename_parts.append(run_id)
        filename_parts.append(timestamp)
        filename = "_".join(filename_parts) + ".json"

        log_path = self.log_dir / filename

        # Create comprehensive log
        trajectory_log = {
            "metadata": {
                "subject_id": subject_id,
                "icu_stay_id": icu_stay_id,
                "run_id": run_id,
                "mode": mode,
                "timestamp": timestamp,
                "total_windows": len(self._trajectory_logs),
            },
            "evaluations": self._trajectory_logs,
        }

        with open(log_path, "w") as f:
            json.dump(trajectory_log, f, indent=2)

        print(f"Saved trajectory log to: {log_path}")

        # Clear logs for next trajectory
        self._trajectory_logs = []

    def evaluate_window(self, window_data: Dict) -> OracleReport:
        """
        Evaluate a single time window with hindsight.

        Args:
            window_data: Dictionary containing patient metadata, history, and future events

        Returns:
            OracleReport with evaluation results
        """
        # Format the prompt (with or without outcome based on blinded mode)
        prompt = format_oracle_prompt(window_data, self.window_hours, include_outcome=not self.blinded)

        # Call LLM
        try:
            response = self.llm_client.chat(prompt=prompt, response_format="json")

            # Track usage (thread-safe)
            with self._stats_lock:
                self.evaluation_count += 1
                if "usage" in response:
                    self.total_tokens_used += response["usage"].get("input_tokens", 0)
                    self.total_tokens_used += response["usage"].get("output_tokens", 0)

            # Parse response
            if response.get("parsed"):
                report = OracleReport.from_dict(response["parsed"], window_data)
            else:
                # Fallback if JSON parsing failed
                print(f"Warning: Failed to parse JSON response. Error: {response.get('parse_error')}")
                print(f"Raw content: {response['content'][:200]}...")

                # Try to extract JSON from the content
                content = response["content"]
                try:
                    # Look for JSON block in markdown code fence
                    if "```json" in content:
                        json_start = content.find("```json") + 7
                        json_end = content.find("```", json_start)
                        json_str = content[json_start:json_end].strip()
                        parsed = json.loads(json_str)
                        report = OracleReport.from_dict(parsed, window_data)
                    else:
                        # Try parsing the whole content
                        parsed = json.loads(content)
                        report = OracleReport.from_dict(parsed, window_data)
                except Exception as e:
                    print(f"Error extracting JSON: {e}")
                    # Return a default report indicating failure
                    report = OracleReport(
                        patient_status_score=0.0,
                        status_rationale=f"Failed to parse Oracle response: {str(e)}",
                        action_quality="neutral",
                        action_quality_rationale="Unable to assess due to parsing error",
                        clinical_insight="Evaluation failed",
                        primary_clinical_driver="Parsing error",
                        guideline_adherence="N/A",
                        window_data=window_data,
                    )

            # Save log
            self._save_log(window_data, prompt, response, report, error=None)

            return report

        except Exception as e:
            print(f"Error during Oracle evaluation: {e}")
            # Return error report
            report = OracleReport(
                patient_status_score=0.0,
                status_rationale=f"Oracle evaluation error: {str(e)}",
                action_quality="neutral",
                action_quality_rationale="Unable to assess due to evaluation error",
                clinical_insight="Evaluation failed",
                primary_clinical_driver="Evaluation error",
                guideline_adherence="N/A",
                window_data=window_data,
            )

            # Save log with error
            self._save_log(window_data, prompt, {"content": None, "error": str(e)}, report, error=str(e))

            return report

    def evaluate_trajectory(self, windows: List[Dict]) -> List[OracleReport]:
        """
        Evaluate all time windows in a patient trajectory.

        Args:
            windows: List of time windows from data parser

        Returns:
            List of OracleReports, one per window
        """
        reports = []

        for i, window in enumerate(windows):
            print(f"Evaluating window {i+1}/{len(windows)} " f"(Hour {window['hours_since_admission']:.1f})...")

            report = self.evaluate_window(window)
            reports.append(report)

        return reports

    def evaluate_trajectory_parallel(
        self, windows: List[Dict], max_workers: int = 10, show_progress: bool = True
    ) -> List[OracleReport]:
        """
        Evaluate all time windows in a patient trajectory in parallel.

        Args:
            windows: List of time windows from data parser
            max_workers: Maximum number of parallel workers (default: 10)
            show_progress: Whether to print progress updates (default: True)

        Returns:
            List of OracleReports, one per window, in the same order as input windows
        """
        if show_progress:
            print(f"Starting parallel evaluation with {max_workers} workers...")

        # Create a list to store results with their original indices
        results = [None] * len(windows)
        completed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(self.evaluate_window, window): i for i, window in enumerate(windows)}

            # Process completed tasks
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                window = windows[index]

                try:
                    report = future.result()
                    results[index] = report
                    completed_count += 1

                    if show_progress:
                        print(
                            f"Completed window {completed_count}/{len(windows)} "
                            f"(Hour {window['hours_since_admission']:.1f})"
                        )

                except Exception as e:
                    print(f"Error evaluating window {index} (Hour {window['hours_since_admission']:.1f}): {e}")
                    # Create error report
                    results[index] = OracleReport(
                        patient_status_score=0.0,
                        status_rationale=f"Parallel evaluation error: {str(e)}",
                        action_quality="neutral",
                        action_quality_rationale="Unable to assess due to parallel evaluation error",
                        clinical_insight="Evaluation failed",
                        primary_clinical_driver="Parallel evaluation error",
                        guideline_adherence="N/A",
                        window_data=window,
                    )

        if show_progress:
            print(f"Parallel evaluation complete: {completed_count}/{len(windows)} windows evaluated")

        return results

    def get_statistics(self) -> Dict:
        """Get Oracle usage statistics."""
        return {
            "total_evaluations": self.evaluation_count,
            "total_tokens_used": self.total_tokens_used,
            "avg_tokens_per_evaluation": (
                self.total_tokens_used / self.evaluation_count if self.evaluation_count > 0 else 0
            ),
        }


def save_oracle_reports(reports: List[OracleReport], output_path: str, include_window_data: bool = False):
    """
    Save Oracle reports to a JSON file.

    Args:
        reports: List of OracleReports
        output_path: Path to output file
        include_window_data: Whether to include full window data in output
    """
    output_data = []

    for report in reports:
        report_dict = report.to_dict()

        if include_window_data and report.window_data:
            report_dict["window_metadata"] = {
                "subject_id": report.window_data["subject_id"],
                "icu_stay_id": report.window_data["icu_stay_id"],
                "window_start_time": report.window_data["current_window_start"],
                "window_end_time": report.window_data["current_window_end"],
                "hours_since_admission": report.window_data["hours_since_admission"],
            }

        output_data.append(report_dict)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(reports)} Oracle reports to {output_path}")


def load_oracle_reports(input_path: str) -> List[Dict]:
    """
    Load Oracle reports from a JSON file.

    Args:
        input_path: Path to input file

    Returns:
        List of report dictionaries
    """
    with open(input_path, "r") as f:
        reports = json.load(f)

    print(f"Loaded {len(reports)} Oracle reports from {input_path}")
    return reports
