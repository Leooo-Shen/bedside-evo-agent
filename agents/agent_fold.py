"""AgentFold: Hierarchical Memory with Dynamic Trajectory Folding.

Implements a memory management approach where the agent dynamically folds
historical trajectories to maintain a compact, informative working context.

Key features:
- Memory Database: Append-only storage of all raw window data
- Working Context: Dynamic context with folded trajectories
- Intelligent Folding: Agent decides when and how to fold trajectories
- Concern Tracking: Maintains active/resolved clinical concerns
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from model.llms import LLMClient


@dataclass
class TrajectoryEntry:
    """Represents a trajectory segment covering a time range."""

    start_window: int  # Starting window index (inclusive)
    end_window: int  # Ending window index (inclusive)
    start_hour: float  # Starting hour since admission
    end_hour: float  # Ending hour since admission
    summary: str  # Summary of this trajectory segment

    def to_text(self) -> str:
        """Convert to text format for context."""
        return f"Hour {self.start_hour:.1f}-{self.end_hour:.1f}: {self.summary}"

    def covers_window(self, window_index: int) -> bool:
        """Check if this trajectory covers a given window."""
        return self.start_window <= window_index <= self.end_window


@dataclass
class ClinicalConcern:
    """Represents a clinical concern."""

    concern_id: str
    status: str  # "Active" or "Resolved"
    note: str


@dataclass
class MemoryDatabase:
    """Append-only storage of all raw window data."""

    patient_metadata: Dict[str, Any] = field(default_factory=dict)
    window_records: List[Dict[str, Any]] = field(default_factory=list)

    def add_window(self, window_data: Dict[str, Any]) -> None:
        """Add a new window record (append-only)."""
        self.window_records.append(window_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "patient_metadata": self.patient_metadata,
            "window_records": self.window_records,
        }

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "MemoryDatabase":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        db = cls()
        db.patient_metadata = data.get("patient_metadata", {})
        db.window_records = data.get("window_records", [])
        return db


@dataclass
class WorkingContext:
    """Dynamic working context that the agent sees."""

    patient_metadata: Dict[str, Any] = field(default_factory=dict)
    historical_key_events: List[str] = field(default_factory=list)  # Append-only
    trajectory: List[TrajectoryEntry] = field(default_factory=list)  # Dynamically folded
    active_concerns: List[ClinicalConcern] = field(default_factory=list)  # Dynamically updated
    clinical_history: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "overall_status": [],
            "hemodynamics": [],
            "respiratory": [],
            "renal_metabolic": [],
            "neurology": [],
        }
    )  # Track clinical assessment trajectory

    def add_key_event(self, event: str) -> None:
        """Add a key event (append-only)."""
        self.historical_key_events.append(event)

    def update_trajectory(self, new_entry: TrajectoryEntry, fold_range: Optional[Tuple[int, int]] = None) -> None:
        """Update trajectory with optional folding.

        Args:
            new_entry: New trajectory entry covering current window
            fold_range: Optional (start_window, end_window) to fold. If provided,
                       removes all entries in [start_window, end_window-1] and adds new_entry
        """
        if fold_range is None:
            # Simple append
            self.trajectory.append(new_entry)
        else:
            # Fold: remove entries in range and add new merged entry
            start_win, end_win = fold_range
            # Remove trajectories that are completely within the fold range
            self.trajectory = [
                t for t in self.trajectory if not (t.start_window >= start_win and t.end_window < end_win)
            ]
            # Add the new folded entry
            self.trajectory.append(new_entry)

    def update_concerns(self, concerns: List[ClinicalConcern]) -> None:
        """Update concerns list."""
        self.active_concerns = concerns

    def get_active_concerns(self) -> List[ClinicalConcern]:
        """Get only active concerns."""
        return [c for c in self.active_concerns if c.status == "Active"]

    def add_clinical_assessment(self, clinical_assessment: Dict) -> None:
        """Add clinical assessment status to history."""
        # Extract overall status
        overall_status = clinical_assessment.get("overall_status", "").lower()
        if overall_status:
            self.clinical_history["overall_status"].append(overall_status)

        # Extract physiology trends statuses
        physiology = clinical_assessment.get("physiology_trends", {})
        for domain in ["hemodynamics", "respiratory", "renal_metabolic", "neurology"]:
            if isinstance(physiology.get(domain), dict):
                status = physiology[domain].get("status", "").lower()
            else:
                # Fallback for old format (string instead of dict)
                status = ""
            if status:
                self.clinical_history[domain].append(status)

    def to_text(
        self,
        current_events: List[Dict] = None,
        current_window_info: str = "",
        sections: Dict[str, bool] = None
    ) -> str:
        """Convert to text format for LLM prompt.

        Args:
            current_events: Optional list of current events to include
            current_window_info: Optional window info string
            sections: Dict controlling which sections to include. Defaults to all True.
                Keys: patient_metadata, historical_key_events, trajectory,
                      clinical_concerns, clinical_trajectory, current_events
        """
        if sections is None:
            sections = {
                "patient_metadata": True,
                "historical_key_events": True,
                "trajectory": True,
                "clinical_concerns": True,
                "clinical_trajectory": True,
                "current_events": True,
            }

        parts = []

        # Patient metadata - filter out IDs, format age to 1 decimal place
        if sections.get("patient_metadata", True):
            parts.append("## Patient Metadata")
            if self.patient_metadata:
                for key, value in self.patient_metadata.items():
                    # Skip IDs
                    if key in ["subject_id", "icu_stay_id"]:
                        continue
                    # Format age to 1 decimal place
                    if key == "age" and value is not None:
                        parts.append(f"{key}: {float(value):.1f}")
                    # Include gender and other fields
                    elif value is not None:
                        parts.append(f"{key}: {value}")
            parts.append("")

        # Historical key events
        if sections.get("historical_key_events", True) and self.historical_key_events:
            parts.append("## Historical Key Events")
            for event in self.historical_key_events:
                parts.append(f"- {event}")
            parts.append("")

        # Previous trajectory
        if sections.get("trajectory", True) and self.trajectory:
            parts.append("## Patient Trajectory")
            for traj in self.trajectory:
                # Format index based on window range
                if traj.start_window == traj.end_window:
                    index_str = f"T{traj.start_window}"
                else:
                    index_str = f"T{traj.start_window}-{traj.end_window}"
                parts.append(f"{index_str}. {traj.to_text()}")
            parts.append("")

        # Active concerns
        if sections.get("clinical_concerns", True):
            active = self.get_active_concerns()
            if active:
                parts.append("## Open Clinical Concerns")
                for concern in active:
                    parts.append(f"- {concern.note}")
                parts.append("")

        # Clinical trajectory (status history across windows)
        if sections.get("clinical_trajectory", True) and any(self.clinical_history.values()):
            parts.append("## Status Trajectory")
            num_windows = len(self.clinical_history["overall_status"])
            if num_windows > 0:
                parts.append(f"({num_windows} windows)")

                # Overall status
                if self.clinical_history["overall_status"]:
                    status_str = ", ".join(self.clinical_history["overall_status"])
                    parts.append(f"Overall Status: [{status_str}]")

                # Physiology domains
                for domain in ["hemodynamics", "respiratory", "renal_metabolic", "neurology"]:
                    if self.clinical_history[domain]:
                        status_str = ", ".join(self.clinical_history[domain])
                        domain_label = domain.replace("_", "/").title()
                        parts.append(f"{domain_label}: [{status_str}]")

                parts.append("")

        # Current events
        if sections.get("current_events", True) and current_events:
            parts.append(f"## Current Events {current_window_info}")
            # Format events
            for event in current_events:  # Limit to avoid overflow
                time = event.get("time", "Unknown")
                code = event.get("code_specifics", event.get("code", "Unknown"))
                numeric_value = event.get("numeric_value")
                text_value = event.get("text_value")

                line = f"- {time}: {code}"
                if numeric_value is not None:
                    line += f" = {numeric_value}"
                if text_value:
                    line += f" ({text_value})"
                parts.append(line)

            parts.append("")

        return "\n".join(parts)


@dataclass
class LLMCallLog:
    """Log entry for a single LLM call."""

    timestamp: str
    patient_id: str
    window_index: int
    hours_since_admission: float
    prompt: str
    response: str
    parsed_response: Optional[Dict]
    input_tokens: int
    output_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "patient_id": self.patient_id,
            "window_index": self.window_index,
            "hours_since_admission": self.hours_since_admission,
            "prompt": self.prompt,
            "response": self.response,
            "parsed_response": self.parsed_response,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "metadata": self.metadata,
        }


class FoldAgent:
    """
    AgentFold: Hierarchical memory with dynamic trajectory folding.

    The agent intelligently decides when and how to fold historical trajectories
    to maintain a compact, informative working context.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        api_key: str = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        enable_logging: bool = False,
        window_duration_hours: float = 0.5,
    ):
        """
        Initialize FoldAgent.

        Args:
            provider: LLM provider ("openai", "anthropic", "google", or "gemini")
            model: Model name
            api_key: API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            enable_logging: Enable detailed logging of all LLM calls
            window_duration_hours: Duration of each window in hours (default 0.5)
        """
        self.llm_client = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Window configuration
        self.window_duration_hours = window_duration_hours

        # Statistics
        self.total_patients = 0
        self.successful_predictions = 0
        self.total_tokens_used = 0
        self.total_folds = 0
        self.total_appends = 0

        # Logging
        self.enable_logging = enable_logging
        self.call_logs: List[LLMCallLog] = []
        self._current_patient_id: str = ""
        self._current_window_index: int = 0
        self._current_hours: float = 0.0

    def process_window(
        self,
        working_context: WorkingContext,
        current_events: List[Dict],
        window_index: int,
        hours_since_admission: float,
        prompt_template: str,
    ) -> Tuple[Dict, WorkingContext]:
        """
        Process a single time window.

        Args:
            working_context: Current working context
            current_events: Events in current window
            window_index: Current window index
            hours_since_admission: Hours since ICU admission
            prompt_template: Prompt template to use

        Returns:
            Tuple of (parsed_response, updated_working_context)
        """
        # Build context text
        window_info = f"(Window {window_index}, Hour {hours_since_admission:.1f})"
        context_text = working_context.to_text(
            current_events=current_events,
            current_window_info=window_info,
        )

        # Format prompt
        prompt = prompt_template.format(
            context=context_text,
            window_index=window_index,
            num_trajectories=len(working_context.trajectory),
        )

        # Call LLM
        response = self._call_llm(
            prompt,
            step_type="window_update",
            metadata={
                "window_index": window_index,
                "num_events": len(current_events),
                "num_trajectories": len(working_context.trajectory),
            },
        )

        # Parse response
        try:
            parsed = self._parse_json_response(response)
        except Exception as e:
            print(f"Error parsing response: {e}")
            # Return empty response and unchanged context
            return {}, working_context

        # Extract memory management decisions (support both old and new format)
        memory_mgmt = parsed.get("memory_management", {})

        # Support both old format (current_analysis) and new format (clinical_assessment)
        clinical_data = parsed.get("clinical_assessment", parsed.get("current_analysis", {}))

        # Execute trajectory folding
        # New format: trajectory_update with start_index/end_index
        # Old format: trajectory_folding with range
        trajectory_update = memory_mgmt.get("trajectory_update", memory_mgmt.get("trajectory_folding", {}))

        # Extract range - handle both formats
        if "start_index" in trajectory_update and "end_index" in trajectory_update:
            # New format
            fold_range = [trajectory_update["start_index"], trajectory_update["end_index"]]
            fold_summary = trajectory_update.get("refined_summary", "")
        else:
            # Old format
            fold_range = trajectory_update.get("range", [])
            fold_summary = trajectory_update.get("summary", "")

        if fold_range and len(fold_range) == 2 and fold_summary:
            k, t = fold_range
            # Validate range
            if k <= t and t == window_index:
                # Create new trajectory entry covering [k, t]
                new_entry = TrajectoryEntry(
                    start_window=k,
                    end_window=t,
                    start_hour=self._get_window_start_hour(working_context, k),
                    end_hour=hours_since_admission + self.window_duration_hours,
                    summary=fold_summary,
                )
                # Execute folding
                working_context.update_trajectory(new_entry, fold_range=(k, t))

                if k == t:
                    self.total_appends += 1
                else:
                    self.total_folds += 1
            else:
                print(f"Warning: Invalid fold range {fold_range} for window {window_index}")
        else:
            # No folding specified, just append
            # Try to get summary from various possible locations
            summary = (
                trajectory_update.get("refined_summary", "")
                or clinical_data.get("clinical_summary", "")
                or f"Window {window_index} events"
            )
            new_entry = TrajectoryEntry(
                start_window=window_index,
                end_window=window_index,
                start_hour=hours_since_admission,
                end_hour=hours_since_admission + self.window_duration_hours,
                summary=summary,
            )
            working_context.update_trajectory(new_entry, fold_range=None)
            self.total_appends += 1

        # critical_events is a list of objects with time, event, significance
        critical_events = clinical_data.get("critical_events", [])
        if critical_events and isinstance(critical_events, list):
            # New format
            for event_obj in critical_events:
                time_str = event_obj.get("time", "")
                event_name = event_obj.get("event", "")
                # significance = event_obj.get("significance", "")
                if time_str and event_name:
                    event_text = f"{event_name}"
                    # if significance:
                    #     event_text += f" - {significance}"
                    working_context.add_key_event(f"{time_str}: {event_text}")
        else:
            # Old format
            key_events = clinical_data.get("key_events", {})
            for time_str, event_desc in key_events.items():
                working_context.add_key_event(f"{time_str}: {event_desc}")

        # Update concerns - handle both formats
        concerns_update = clinical_data.get("active_concerns_update", [])
        updated_concerns = []
        for concern_data in concerns_update:
            # New format has "concern" field, old format has "note" field
            note = concern_data.get("concern", concern_data.get("note", ""))
            concern = ClinicalConcern(
                concern_id=concern_data.get("id", ""),
                status=concern_data.get("status", "Active"),
                note=note,
            )
            updated_concerns.append(concern)
        working_context.update_concerns(updated_concerns)

        return parsed, working_context

    def _get_window_start_hour(self, working_context: WorkingContext, window_index: int) -> float:
        """Get the start hour for a given window index.

        First tries to find it from existing trajectories. If not found,
        calculates it based on window_index and window_duration_hours.
        """
        # Try to find from existing trajectories
        for traj in working_context.trajectory:
            if traj.start_window == window_index:
                return traj.start_hour

        # Calculate based on window index and duration
        return window_index * self.window_duration_hours

    def predict_survival(
        self,
        working_context: WorkingContext,
        current_events: List[Dict],
        hours_since_admission: float,
        prompt_template: str,
    ) -> Dict:
        """
        Make final survival prediction.

        Args:
            working_context: Final working context
            current_events: Events from final window
            hours_since_admission: Hours since admission
            prompt_template: Prompt template for prediction

        Returns:
            Prediction dictionary
        """
        # Build context text
        context_text = working_context.to_text(
            current_events=current_events,
            current_window_info=f"(Final Window, Hour {hours_since_admission:.1f})",
        )

        # Format prompt
        prompt = prompt_template.format(context=context_text)

        # Call LLM
        response = self._call_llm(
            prompt,
            step_type="predict",
            metadata={
                "num_trajectories": len(working_context.trajectory),
                "num_concerns": len(working_context.get_active_concerns()),
            },
        )

        # Parse response
        try:
            parsed = self._parse_json_response(response)
            return parsed
        except Exception as e:
            print(f"Error parsing prediction: {e}")
            return {
                "survival_prediction": {
                    "outcome": "unknown",
                    "confidence": 0.0,
                    "rationale": f"Parsing error: {e}",
                }
            }

    def run_patient_trajectory(
        self,
        windows: List[Dict],
        patient_metadata: Dict,
        verbose: bool = True,
    ) -> Tuple[Dict, WorkingContext, MemoryDatabase]:
        """
        Run the full AgentFold pipeline on a patient trajectory.

        Args:
            windows: List of time windows from data parser
            patient_metadata: Patient metadata (age, etc.)
            verbose: Print progress

        Returns:
            Tuple of (prediction_dict, final_working_context, memory_database)
        """
        from prompts.agent_fold_prompts import get_survival_prediction_prompt, get_window_update_prompt

        self.total_patients += 1

        # Initialize memory database
        memory_db = MemoryDatabase(patient_metadata=patient_metadata)

        # Initialize working context
        working_context = WorkingContext(patient_metadata=patient_metadata)

        # Set current patient context for logging
        self._current_patient_id = (
            f"{patient_metadata.get('subject_id', 'unknown')}_{patient_metadata.get('icu_stay_id', 'unknown')}"
        )

        if verbose:
            print(f"Processing patient with {len(windows)} windows...")

        # Process all windows except the last one
        for i, window in enumerate(windows[:-1]):
            current_events = window.get("current_events", [])
            hours = window.get("hours_since_admission", 0)

            # Update logging context
            self._current_window_index = i
            self._current_hours = hours

            if not current_events:
                continue

            if verbose:
                print(f"  Window {i+1}/{len(windows)} (Hour {hours:.1f})...", end=" ")

            # Process window
            parsed, working_context = self.process_window(
                working_context=working_context,
                current_events=current_events,
                window_index=i,
                hours_since_admission=hours,
                prompt_template=get_window_update_prompt(),
            )

            # Add to memory database with all LLM-generated details
            window_record = {
                "window_index": i,
                "hours_since_admission": hours,
                "num_events": len(current_events),
                "raw_events": current_events,  # Store raw events from this window
                "llm_response": parsed,  # Store complete LLM response
                "trajectory_state": {
                    "num_trajectories": len(working_context.trajectory),
                    "trajectories": [
                        {
                            "start_window": t.start_window,
                            "end_window": t.end_window,
                            "start_hour": t.start_hour,
                            "end_hour": t.end_hour,
                            "summary": t.summary,
                        }
                        for t in working_context.trajectory
                    ],
                },
                "key_events": working_context.historical_key_events.copy(),
                "active_concerns": [
                    {
                        "id": c.concern_id,
                        "status": c.status,
                        "note": c.note,
                    }
                    for c in working_context.active_concerns
                ],
            }
            memory_db.add_window(window_record)

            if verbose:
                # Support both old and new format for status
                clinical_data = parsed.get("clinical_assessment", parsed.get("current_analysis", {}))
                status = clinical_data.get("overall_status", clinical_data.get("overall_patient_status", "unknown"))
                print(f"Status: {status}")

        # Extract last window for prediction
        last_window = windows[-1] if windows else {}
        last_window_events = last_window.get("current_events", [])
        last_window_hours = last_window.get("hours_since_admission", 0)

        # Final prediction
        self._current_window_index = -1
        if verbose:
            print(
                f"\nMaking final prediction with {len(last_window_events)} events from final window (Hour {last_window_hours:.1f})..."
            )

        prediction = self.predict_survival(
            working_context=working_context,
            current_events=last_window_events,
            hours_since_admission=last_window_hours,
            prompt_template=get_survival_prediction_prompt(),
        )

        # Add last window to memory database with prediction results
        last_window_record = {
            "window_index": len(windows) - 1,
            "hours_since_admission": last_window_hours,
            "num_events": len(last_window_events),
            "raw_events": last_window_events,  # Store raw events from final window
            "prediction": prediction,  # Store complete prediction
            "final_trajectory_state": {
                "num_trajectories": len(working_context.trajectory),
                "trajectories": [
                    {
                        "start_window": t.start_window,
                        "end_window": t.end_window,
                        "start_hour": t.start_hour,
                        "end_hour": t.end_hour,
                        "summary": t.summary,
                    }
                    for t in working_context.trajectory
                ],
            },
            "final_key_events": working_context.historical_key_events.copy(),
            "final_active_concerns": [
                {
                    "id": c.concern_id,
                    "status": c.status,
                    "note": c.note,
                }
                for c in working_context.active_concerns
            ],
        }
        memory_db.add_window(last_window_record)

        return prediction, working_context, memory_db

    def _call_llm(self, prompt: str, step_type: str = "unknown", metadata: Dict = None) -> str:
        """Call LLM and track token usage, optionally logging the call."""
        response = self.llm_client.chat(prompt=prompt, response_format="text")

        input_tokens = 0
        output_tokens = 0
        if "usage" in response:
            input_tokens = response["usage"].get("input_tokens", 0)
            output_tokens = response["usage"].get("output_tokens", 0)
            self.total_tokens_used += input_tokens + output_tokens

        content = response.get("content", "")
        log_metadata = dict(metadata or {})
        log_metadata.update(
            {
                "step_type": step_type,
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model,
            }
        )
        if response.get("model"):
            log_metadata["llm_response_model"] = response.get("model")

        # Log the call if logging is enabled
        if self.enable_logging:
            log_entry = LLMCallLog(
                timestamp=datetime.now().isoformat(),
                patient_id=self._current_patient_id,
                window_index=self._current_window_index,
                hours_since_admission=self._current_hours,
                prompt=prompt,
                response=content,
                parsed_response=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata=log_metadata,
            )
            self.call_logs.append(log_entry)

        return content

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response."""
        # Try to extract from <response> tags first
        response_match = re.search(r"<response>(.*?)</response>", response, re.DOTALL | re.IGNORECASE)
        if response_match:
            response = response_match.group(1).strip()

        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                return json.loads(response[start:end].strip())

        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return json.loads(response[start:end].strip())

        # Try finding JSON object
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            return json.loads(match.group())

        raise ValueError(f"Could not parse JSON from response: {response[:200]}...")

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_patients": self.total_patients,
            "successful_predictions": self.successful_predictions,
            "accuracy": self.successful_predictions / max(self.total_patients, 1),
            "total_tokens_used": self.total_tokens_used,
            "total_folds": self.total_folds,
            "total_appends": self.total_appends,
            "total_llm_calls": len(self.call_logs) if self.enable_logging else 0,
        }

    def get_logs(self) -> List[Dict]:
        """Get all LLM call logs as list of dictionaries."""
        return [log.to_dict() for log in self.call_logs]

    def save_logs(self, path: str) -> None:
        """Save all LLM call logs to a JSON file."""
        logs_data = {
            "total_calls": len(self.call_logs),
            "total_tokens": self.total_tokens_used,
            "calls": self.get_logs(),
        }
        with open(path, "w") as f:
            json.dump(logs_data, f, indent=2, ensure_ascii=False)

    def clear_logs(self) -> None:
        """Clear all stored logs."""
        self.call_logs = []
