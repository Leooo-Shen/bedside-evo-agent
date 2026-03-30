"""ReMeM Agent implementation.

Implements the ReMeM (Retrieval-Enhanced Memory Management) method for ICU survival prediction.
This is one of several possible memory management approaches for clinical decision support.

Key features:
- Iterative state tracking through sliding windows
- Search-Think-Act-Refine loop for each window
- Memory-augmented survival prediction with intra-patient learning only
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from memory import ICUContextBuilder
from model.llms import LLMClient
from prompts.remem_prompts import (
    format_refine_state_prompt,
    format_state_update_prompt,
    format_survival_prediction_prompt,
)


class ActionType(Enum):
    """Types of agent actions."""

    ACT = "act"
    REFINE = "refine"


@dataclass
class LLMCallLog:
    """Log entry for a single LLM call."""

    timestamp: str
    step_type: str  # "state_update", "refine", "predict"
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
            "step_type": self.step_type,
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


@dataclass
class AgentAction:
    """Represents an action taken by the agent."""

    action_type: ActionType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatientState:
    """Current state summary for a patient."""

    summary: str = ""
    key_concerns: List[str] = field(default_factory=list)
    physiology: Dict[str, str] = field(default_factory=dict)  # Renamed from organ_systems
    interventions: List[str] = field(default_factory=list)  # New field
    uncertainties: List[str] = field(default_factory=list)  # New field
    trajectory: str = "unknown"
    hours_tracked: float = 0.0

    def to_text(self) -> str:
        """Convert state to text format."""
        parts = [self.summary]
        if self.key_concerns:
            parts.append(f"Key Concerns: {', '.join(self.key_concerns)}")
        if self.physiology:
            systems_str = ", ".join([f"{k}: {v}" for k, v in self.physiology.items()])
            parts.append(f"Physiology: {systems_str}")
        if self.interventions:
            parts.append(f"Interventions: {', '.join(self.interventions)}")
        if self.uncertainties:
            parts.append(f"Uncertainties: {', '.join(self.uncertainties)}")
        parts.append(f"Trajectory: {self.trajectory}")
        return "\n".join(parts)


class RememAgent:
    """
    ReMeM Agent: Retrieval-Enhanced Memory Management for clinical decision support.

    This agent implements the ReMeM method, which uses:
    1. Iterative State Tracking (Module 1)
       - Think & Act: Update patient state summary
       - Refine: Compress state to prevent overflow
    2. Final Prediction with intra-patient memory context

    Note: This is one of several possible memory management approaches.
    The agent is designed to be decoupled from the experiment pipeline,
    allowing for easy comparison with other memory management methods.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        api_key: str = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        max_state_length: int = 1500,
        enable_logging: bool = False,
        enable_intra_patient_refinement: bool = False,
    ):
        """
        Initialize ReMeM Agent.

        Args:
            provider: LLM provider ("openai", "anthropic", "google", or "gemini")
            model: Model name
            api_key: API key
            temperature: Optional sampling temperature override
            max_tokens: Maximum tokens in response
            max_state_length: Maximum length of state summary
            enable_logging: Enable detailed logging of all LLM calls
            enable_intra_patient_refinement: Enable Think-Refine-Act loop for window state updates
        """
        self.llm_client = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Initialize context builder
        self.context_builder = ICUContextBuilder(
            max_experiences=0,  # No cross-patient experiences
            max_state_length=max_state_length,
        )

        self.max_state_length = max_state_length
        self.enable_intra_patient_refinement = enable_intra_patient_refinement

        # Statistics
        self.total_patients = 0
        self.successful_predictions = 0
        self.total_tokens_used = 0
        self.state_updates = 0
        self.refinements = 0

        # Logging
        self.enable_logging = enable_logging
        self.call_logs: List[LLMCallLog] = []
        self._current_patient_id: str = ""
        self._current_window_index: int = 0
        self._current_hours: float = 0.0

    def update_state(
        self,
        previous_state: PatientState,
        current_events: List[Dict],
        hours_since_admission: float,
    ) -> PatientState:
        """
        Update patient state based on new data.

        Implements: S_t = Think_Act(S_{t-1}, x_t)
        """
        # Build context for state update (no cross-patient experiences)
        context = self.context_builder.build_state_update_context(
            previous_state=previous_state.to_text() if previous_state.summary else "",
            current_events=current_events,
            retrieved=[],  # No cross-patient experiences
            hours_since_admission=hours_since_admission,
        )

        prompt = format_state_update_prompt(context)

        response = self._call_llm(
            prompt,
            step_type="state_update",
            metadata={
                "num_events": len(current_events),
            },
        )
        self.state_updates += 1

        try:
            parsed = self._parse_json_response(response)
            return PatientState(
                summary=parsed.get("summary", ""),
                key_concerns=parsed.get("key_concerns", []),
                physiology=parsed.get("physiology", {}),
                interventions=parsed.get("interventions", []),
                uncertainties=parsed.get("uncertainties", []),
                trajectory=parsed.get("trajectory", "unknown"),
                hours_tracked=hours_since_admission,
            )
        except Exception as e:
            print(f"Error parsing state update: {e}")
            return PatientState(
                summary=f"Hour {hours_since_admission}: State update failed",
                hours_tracked=hours_since_admission,
            )

    def update_state_with_refinement(
        self,
        previous_state: PatientState,
        current_events: List[Dict],
        hours_since_admission: float,
        window_states: List[Dict],
        max_iterations: int = 3,
    ) -> Tuple[PatientState, List[str], List[int]]:
        """
        Update patient state using Think-Refine-Act loop.

        Allows agent to:
        - Think: Reason about patient trajectory
        - Refine: Prune irrelevant previous window states
        - Act: Update current state

        Args:
            previous_state: Previous patient state
            current_events: New events in current window
            hours_since_admission: Current time
            window_states: Previous window states (can be pruned)
            max_iterations: Maximum Think-Refine iterations

        Returns:
            Tuple of (updated_state, reasoning_trace, pruned_window_ids)
        """
        reasoning_trace = []
        pruned_window_ids = []

        for iteration in range(max_iterations):
            # Build context with previous window states (no cross-patient experiences)
            context = self.context_builder.build_state_update_context(
                previous_state=previous_state.to_text() if previous_state.summary else "",
                current_events=current_events,
                retrieved=[],  # No cross-patient experiences
                hours_since_admission=hours_since_admission,
                window_states=window_states,
                reasoning_trace=reasoning_trace,
            )

            prompt = format_state_update_prompt(context)

            response = self._call_llm(
                prompt,
                step_type="state_update",
                metadata={
                    "num_events": len(current_events),
                    "num_window_states": len(window_states),
                    "iteration": iteration,
                },
            )

            print(response)

            # Parse action
            action_type, content = self._parse_action_response(response)

            if action_type == "prune":
                print(f"!!!!!!!!!! Iteration {iteration+1}: Agent is pruning...")
                # Prune window states
                ids_to_prune = self._parse_prune_ids(content, len(window_states))
                if ids_to_prune:
                    pruned_window_ids.extend([i + 1 for i in ids_to_prune])
                    window_states = [w for i, w in enumerate(window_states) if i not in ids_to_prune]
                    reasoning_trace.append(f"Pruned windows: {content} (remaining: {len(window_states)})")

            elif action_type == "predict":
                # Parse state update (using "predict" action type for state JSON)
                try:
                    parsed = self._parse_json_response(content)
                    self.state_updates += 1
                    return (
                        PatientState(
                            summary=parsed.get("summary", ""),
                            key_concerns=parsed.get("key_concerns", []),
                            physiology=parsed.get("physiology", {}),
                            interventions=parsed.get("interventions", []),
                            uncertainties=parsed.get("uncertainties", []),
                            trajectory=parsed.get("trajectory", "unknown"),
                            hours_tracked=hours_since_admission,
                        ),
                        reasoning_trace,
                        pruned_window_ids,
                    )
                except Exception as e:
                    print(f"Error parsing state update: {e}")
                    reasoning_trace.append(f"Failed to parse state, retrying...")

        # If loop exhausted, return fallback state
        print(f"Warning: Max iterations reached for state update")
        self.state_updates += 1
        return (
            PatientState(
                summary=f"Hour {hours_since_admission}: Max iterations reached",
                hours_tracked=hours_since_admission,
            ),
            reasoning_trace,
            pruned_window_ids,
        )

    def refine_state(self, state: PatientState) -> PatientState:
        """
        Compress/refine state to prevent context overflow.

        Implements: S_t = Refine(S_raw_t)
        """
        if len(state.summary) <= self.max_state_length:
            return state

        prompt = format_refine_state_prompt(state.to_text())

        response = self._call_llm(
            prompt,
            step_type="refine",
            metadata={
                "original_length": len(state.summary),
            },
        )
        self.refinements += 1

        try:
            parsed = self._parse_json_response(response)
            return PatientState(
                summary=parsed.get("summary", state.summary[: self.max_state_length]),
                key_concerns=parsed.get("key_concerns", state.key_concerns[:3]),
                physiology=parsed.get("physiology", state.physiology),
                interventions=parsed.get("interventions", state.interventions[:3]),
                uncertainties=parsed.get("uncertainties", state.uncertainties[:3]),
                trajectory=parsed.get("trajectory", state.trajectory),
                hours_tracked=state.hours_tracked,
            )
        except Exception:
            # Fallback: truncate
            return PatientState(
                summary=state.summary[: self.max_state_length],
                key_concerns=state.key_concerns[:3],
                physiology=state.physiology,
                interventions=state.interventions[:3],
                uncertainties=state.uncertainties[:3],
                trajectory=state.trajectory,
                hours_tracked=state.hours_tracked,
            )

    def _parse_action_response(self, response: str) -> Tuple[str, str]:
        """
        Parse LLM response into action type and content using XML format.

        Returns:
            Tuple of (action_type, content) where action_type is one of:
            - "prune": Prune memories (content contains IDs like "1,3")
            - "predict": Final prediction (content is JSON)

        Note: thinking is now mandatory and extracted separately, not returned as action
        """
        response = response.strip()

        # Extract thinking (mandatory but not returned as action)
        thought_match = re.search(r"<thought_process>(.*?)</thought_process>", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            # Thinking is extracted but not used as action type
            pass

        # Check for PRUNE action
        action_match = re.search(r"<action>\s*PRUNE\s*</action>", response, re.IGNORECASE)
        if action_match:
            # Extract IDs
            ids_match = re.search(r"<ids>(.*?)</ids>", response, re.DOTALL | re.IGNORECASE)
            if ids_match:
                return ("prune", ids_match.group(1).strip())
            else:
                # Fallback: look for IDs after action tag
                return ("prune", "")

        # Check for state_update (for state tracking)
        state_match = re.search(r"<state_update>(.*?)</state_update>", response, re.DOTALL | re.IGNORECASE)
        if state_match:
            return ("predict", state_match.group(1).strip())

        # Check for prediction (for survival prediction)
        pred_match = re.search(r"<prediction>(.*?)</prediction>", response, re.DOTALL | re.IGNORECASE)
        if pred_match:
            return ("predict", pred_match.group(1).strip())

        # Fallback: treat entire response as prediction
        return ("predict", response)

    def _parse_prune_ids(self, prune_str: str, max_id: int) -> List[int]:
        """Parse prune IDs from string like '1,3' or '2-4'."""
        ids = []

        # Handle comma-separated: "1,3,5"
        for part in prune_str.split(","):
            part = part.strip()

            # Handle range: "2-4"
            if "-" in part:
                try:
                    start, end = part.split("-")
                    for i in range(int(start), int(end) + 1):
                        if 1 <= i <= max_id:
                            ids.append(i - 1)  # Convert to 0-indexed
                except ValueError:
                    pass
            else:
                # Single ID
                try:
                    idx = int(part)
                    if 1 <= idx <= max_id:
                        ids.append(idx - 1)  # Convert to 0-indexed
                except ValueError:
                    pass

        return ids

    def _parse_memory_prune_ids(
        self, prune_str: str, num_windows: int, num_experiences: int
    ) -> Tuple[List[int], List[int]]:
        """
        Parse prune IDs for both window states and experiences.

        Supports formats:
        - "W1, W2" for window states
        - "E1, E2" for experiences
        - "1, 2" defaults to experiences (backward compatibility)

        Args:
            prune_str: String containing IDs to prune
            num_windows: Number of window states available
            num_experiences: Number of experiences available

        Returns:
            Tuple of (window_ids, experience_ids) as 0-indexed lists
        """
        window_ids = []
        experience_ids = []

        # Split by comma
        for part in prune_str.split(","):
            part = part.strip().upper()

            # Check for window state prefix (W)
            if part.startswith("W"):
                try:
                    idx = int(part[1:])
                    if 1 <= idx <= num_windows:
                        window_ids.append(idx - 1)  # Convert to 0-indexed
                except ValueError:
                    pass

            # Check for experience prefix (E)
            elif part.startswith("E"):
                try:
                    idx = int(part[1:])
                    if 1 <= idx <= num_experiences:
                        experience_ids.append(idx - 1)  # Convert to 0-indexed
                except ValueError:
                    pass

            # No prefix - default to experience (backward compatibility)
            else:
                try:
                    idx = int(part)
                    if 1 <= idx <= num_experiences:
                        experience_ids.append(idx - 1)  # Convert to 0-indexed
                except ValueError:
                    pass

        return window_ids, experience_ids

    def predict_survival(
        self,
        final_state: PatientState,
        patient_metadata: Dict,
        window_states: List[Dict] = None,
        current_events: List[Dict] = None,
        hours_since_admission: float = 0.0,
        max_iterations: int = 5,
    ) -> Dict:
        """
        Make final survival prediction using Think-Refine-Act loop.

        Implements: y_pred = Predict(S_final) with memory refinement

        Args:
            final_state: Final patient state after all windows
            patient_metadata: Patient metadata (age, etc.)
            window_states: Previous window states (patient-specific memory)
            current_events: Events from the final window
            hours_since_admission: Hours since admission for final window
            max_iterations: Maximum Think-Refine iterations before forcing prediction

        Returns:
            Prediction dictionary with survival outcome
        """
        # Think-Refine-Act loop
        reasoning_trace = []
        pruned_window_ids = []

        # Initialize window_states if not provided
        if window_states is None:
            window_states = []

        for iteration in range(max_iterations):
            # Build prediction context with current window states (no cross-patient experiences)
            context = self.context_builder.build_prediction_context(
                final_state=final_state.to_text(),
                retrieved=[],  # No cross-patient experiences
                patient_metadata=patient_metadata,
                window_states=window_states,
                current_events=current_events,
                hours_since_admission=hours_since_admission,
                reasoning_trace=reasoning_trace,
            )

            prompt = format_survival_prediction_prompt(context)

            response = self._call_llm(
                prompt,
                step_type="predict",
                metadata={
                    "num_window_states": len(window_states),
                    "patient_age": patient_metadata.get("age", "unknown"),
                    "iteration": iteration,
                },
            )

            # Parse action
            action_type, content = self._parse_action_response(response)

            if action_type == "prune":
                # Parse which window states to prune
                window_ids, _ = self._parse_memory_prune_ids(content, len(window_states), 0)

                # Prune window states (patient-specific memory)
                if window_ids:
                    pruned_window_ids.extend([i + 1 for i in window_ids])
                    window_states = [w for i, w in enumerate(window_states) if i not in window_ids]
                    reasoning_trace.append(
                        f"Pruned window states: {len(window_ids)} (remaining: {len(window_states)})"
                    )

            elif action_type == "predict":
                # Parse and return prediction
                try:
                    prediction = self._parse_json_response(content)
                    # Add metadata about reasoning process
                    if "prediction_metadata" not in prediction:
                        prediction["prediction_metadata"] = {}
                    prediction["prediction_metadata"]["reasoning_iterations"] = iteration + 1
                    prediction["prediction_metadata"]["window_states_pruned"] = len(pruned_window_ids)
                    prediction["prediction_metadata"]["reasoning_trace"] = reasoning_trace
                    return prediction
                except Exception as e:
                    print(f"Error parsing prediction: {e}")
                    # Continue to next iteration
                    reasoning_trace.append(f"Failed to parse prediction, retrying...")

        # If loop exhausted, force a prediction
        print(f"Warning: Max iterations reached, forcing prediction")
        return {
            "survival_prediction": {
                "outcome": "unknown",
                "confidence": 0.0,
                "rationale": "Max iterations reached without valid prediction",
            },
            "prediction_metadata": {
                "reasoning_iterations": max_iterations,
                "window_states_pruned": len(pruned_window_ids),
                "reasoning_trace": reasoning_trace,
            },
        }

    def run_patient_trajectory(
        self,
        windows: List[Dict],
        patient_metadata: Dict,
        verbose: bool = True,
    ) -> Tuple[Dict, PatientState, List[Dict]]:
        """
        Run the full ReMeM pipeline on a patient trajectory.

        Implements Module 1: Intra-patient Inference Engine
        - Iterates through time windows
        - For each window: Search → Think & Act → Refine
        - Returns final prediction and state

        Args:
            windows: List of time windows from data parser
            patient_metadata: Patient metadata (age, etc.)
            verbose: Print progress

        Returns:
            Tuple of (prediction_dict, final_state, window_states)
        """
        self.total_patients += 1
        state = PatientState()
        window_states = []  # Track state evolution through windows

        # Set current patient context for logging
        self._current_patient_id = (
            f"{patient_metadata.get('subject_id', 'unknown')}_{patient_metadata.get('icu_stay_id', 'unknown')}"
        )

        if verbose:
            print(f"Processing patient with {len(windows)} windows...")

        # Iterate through windows EXCEPT the last one (Search → Think & Act → Refine loop)
        # The last window's events will be passed directly to prediction
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

            # Step A: Retrieve - No cross-patient retrieval
            # Step B: Think & Refine & Act - Update state (with optional refinement)
            if self.enable_intra_patient_refinement and len(window_states) > 0:
                # Use Think-Refine-Act loop
                state, reasoning_trace, pruned_ids = self.update_state_with_refinement(
                    previous_state=state,
                    current_events=current_events,
                    hours_since_admission=hours,
                    window_states=window_states,
                )
                # Prune window states if any were marked for pruning
                if pruned_ids:
                    window_states = [w for i, w in enumerate(window_states) if (i + 1) not in pruned_ids]
            else:
                # Direct state update (no refinement)
                state = self.update_state(
                    previous_state=state,
                    current_events=current_events,
                    hours_since_admission=hours,
                )
                reasoning_trace = []
                pruned_ids = []

            # # Step C: Refine - Compress if needed
            # state = self.refine_state(state)

            # Track window state
            window_state_entry = {
                "window_index": i,
                "hours_since_admission": hours,
                "num_events": len(current_events),
                "state": {
                    "summary": state.summary,
                    "key_concerns": state.key_concerns,
                    "physiology": state.physiology,
                    "interventions": state.interventions,
                    "uncertainties": state.uncertainties,
                    "trajectory": state.trajectory,
                },
            }

            # Add refinement metadata if enabled
            if self.enable_intra_patient_refinement:
                window_state_entry["reasoning_trace"] = reasoning_trace
                window_state_entry["pruned_window_ids"] = pruned_ids

            window_states.append(window_state_entry)

            if verbose:
                print(f"Trajectory: {state.trajectory}")

        # Extract last window's events for prediction
        # Data parser ensures the last window always has events
        last_window = windows[-1] if windows else {}
        last_window_events = last_window.get("current_events", [])
        last_window_hours = last_window.get("hours_since_admission", 0)

        # Final Prediction
        self._current_window_index = -1  # Indicate final prediction
        if verbose:
            print(f"\nMaking final prediction with {len(last_window_events)} events from final window (Hour {last_window_hours:.1f})...")

        prediction = self.predict_survival(
            final_state=state,
            patient_metadata=patient_metadata,
            window_states=window_states,
            current_events=last_window_events,
            hours_since_admission=last_window_hours,
        )

        return prediction, state, window_states

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
            parsed = None
            # try:
            #     parsed = self._parse_json_response(content)
            # except Exception:
            #     pass

            log_entry = LLMCallLog(
                timestamp=datetime.now().isoformat(),
                step_type=step_type,
                patient_id=self._current_patient_id,
                window_index=self._current_window_index,
                hours_since_admission=self._current_hours,
                prompt=prompt,
                response=content,
                parsed_response=parsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata=log_metadata,
            )
            self.call_logs.append(log_entry)

        return content

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response."""
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
            "state_updates": self.state_updates,
            "refinements": self.refinements,
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
