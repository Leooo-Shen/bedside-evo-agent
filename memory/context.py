"""Context builder for memory synthesis.

Implements the Synthesis operation: C̃_t = C(x_t, R_t)
where retrieved information is restructured into working context.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .retriever import RetrievalResult


class ContextBuilder(ABC):
    """
    Abstract base class for context building.

    The context builder implements: C̃_t = C(x_t, R_t)
    """

    @abstractmethod
    def build(
        self,
        query: str,
        retrieved: List[RetrievalResult],
        **kwargs,
    ) -> str:
        """
        Build context from query and retrieved memories.

        Args:
            query: Current input query (x_t)
            retrieved: Retrieved memory entries (R_t)
            **kwargs: Additional context building options

        Returns:
            Constructed context string (C̃_t)
        """
        pass


class ICUContextBuilder(ContextBuilder):
    """
    Context builder for ICU patient state tracking.

    Follows the Evo-ICU spec for building prompts with:
    - Patient state summary (S_{t-1})
    - Current window data (x_t)
    - Retrieved clinical experiences (R_t)
    """

    def __init__(
        self,
        max_experiences: int = 10,
        max_state_length: int = 2000,
    ):
        """
        Initialize ICU context builder.

        Args:
            max_experiences: Maximum number of experiences to include
            max_state_length: Maximum length of state summary
        """
        self.max_experiences = max_experiences
        self.max_state_length = max_state_length

    def build(
        self,
        query: str,
        retrieved: List[RetrievalResult],
        patient_state: str = "",
        current_events: List[Dict] = None,
        hours_since_admission: float = 0.0,
        **kwargs,
    ) -> str:
        """Build context for ICU state tracking."""
        parts = []

        # Previous patient state summary
        if patient_state:
            state_text = patient_state[: self.max_state_length]
            if len(patient_state) > self.max_state_length:
                state_text += "..."
            parts.append(
                f"""==================================================
PATIENT STATE SUMMARY (Previous)
==================================================
{state_text}"""
            )

        # Retrieved clinical experiences
        if retrieved:
            exp_parts = []
            for i, result in enumerate(retrieved[: self.max_experiences]):
                entry = result.entry
                exp_text = f"""[Experience #{i + 1}] (Relevance: {result.score:.2f})
Scenario: {entry.input_text}...
Insight: {entry.output_text}...
Outcome: {'Correct' if entry.is_successful else 'Incorrect'}"""
                exp_parts.append(exp_text)

            parts.append(
                f"""==================================================
RELEVANT CLINICAL EXPERIENCES FROM MEMORY
==================================================
{chr(10).join(exp_parts)}"""
            )

        # Current window events
        if current_events:
            events_str = self._format_events(current_events)
            parts.append(
                f"""==================================================
CURRENT WINDOW DATA (Hour {hours_since_admission:.1f})
==================================================
{events_str}"""
            )

        # Current query/task
        parts.append(
            f"""==================================================
YOUR TASK
==================================================
{query}"""
        )

        return "\n\n".join(parts)

    def _format_events(self, events: List[Dict], max_events: int = 200) -> str:
        """Format events for display."""
        if not events:
            return "No events in this window."

        display_events = events[:max_events]
        formatted = []

        for event in display_events:
            time = event.get("time", "Unknown")
            code = event.get("code_specifics", event.get("code", "Unknown"))
            value = event.get("numeric_value")
            text = event.get("text_value")

            line = f"- {time}: {code}"
            if value is not None:
                line += f" = {value}"
            if text:
                line += f" ({text})"
            formatted.append(line)

        if len(events) > max_events:
            formatted.append(f"... and {len(events) - max_events} more events")

        return "\n".join(formatted)

    def build_state_update_context(
        self,
        previous_state: str,
        current_events: List[Dict],
        retrieved: List[RetrievalResult],
        hours_since_admission: float,
        window_states: List[Dict] = None,
        reasoning_trace: List[str] = None,
    ) -> str:
        """Build context for state update with optional Think-Refine-Act support."""
        parts = []

        # Previous window states (if refinement enabled)
        if window_states:
            # Filter to only show non-pruned states to the agent
            active_states = [ws for ws in window_states if not ws.get("is_pruned", False)]
            if active_states:
                memories = []
                for i, ws in enumerate(active_states):
                    # Calculate window duration
                    duration = 0.5
                    start_hour = ws["hours_since_admission"]
                    time_str = f"{start_hour:.1f}-{start_hour + duration:.1f}"

                    # Build comprehensive memory entry with all available information
                    state = ws.get("state", {})
                    memory_entry = {
                        "id": i + 1,
                        "time_range": f"Hour {time_str}",
                        "trajectory": state.get("trajectory", "unknown"),
                        "summary": state.get("summary", ""),
                    }

                    # Add optional fields only if they exist
                    if state.get("physiology"):
                        memory_entry["physiology"] = state["physiology"]
                    if state.get("key_concerns"):
                        memory_entry["key_concerns"] = state["key_concerns"]
                    if state.get("interventions"):
                        memory_entry["interventions"] = state["interventions"]
                    if state.get("uncertainties"):
                        memory_entry["uncertainties"] = state["uncertainties"]

                    memories.append(memory_entry)

                import json
                parts.append(
                    f"""## Previous Memory
{json.dumps(memories, indent=2)}"""
                )

        # Current events - calculate time range
        # The current window starts at hours_since_admission and lasts 0.5 hours
        duration = 0.5
        time_range_str = f"Hour {hours_since_admission:.1f}-{hours_since_admission + duration:.1f}"

        events_str = self._format_events(current_events)
        parts.append(
            f"""## Current ICU Events ({time_range_str})
{events_str}"""
        )

        # Retrieved experiences (cross-agent memory)
        if retrieved:
            experiences = []
            for i, result in enumerate(retrieved[: self.max_experiences]):
                entry = result.entry
                experiences.append({
                    "id": i + 1,
                    "relevance": round(result.score, 2),
                    "scenario": entry.input_text[:150] + "...",
                    "outcome": entry.output_text[:150] + "...",
                })
            parts.append(
                f"""## Relevant Clinical Experiences (Cross-Patient Memory)
{json.dumps(experiences, indent=2)}"""
            )

        # Reasoning trace (if any)
        if reasoning_trace:
            parts.append(
                f"""## Reasoning Trace
{chr(10).join(reasoning_trace)}"""
            )

        return "\n\n".join(parts)

    def build_prediction_context(
        self,
        final_state: str,
        retrieved: List[RetrievalResult],
        patient_metadata: Dict,
        window_states: List[Dict] = None,
        current_events: List[Dict] = None,
        hours_since_admission: float = 0.0,
        reasoning_trace: List[str] = None,
    ) -> str:
        """Build context for final survival prediction with Think-Refine-Act support."""
        parts = []

        # Patient metadata - filter out IDs, format age to 1 decimal place
        age = patient_metadata.get("age", None)
        gender = patient_metadata.get("gender", None)

        metadata_parts = []
        if age is not None:
            metadata_parts.append(f"- Age: {float(age):.1f} years")
        if gender is not None:
            metadata_parts.append(f"- Gender: {gender}")

        if metadata_parts:
            parts.append(
                f"""## Patient Information
{chr(10).join(metadata_parts)}"""
            )
        else:
            parts.append(
                """## Patient Information
- Age: Unknown"""
            )

        # Previous window states (patient-specific memory)
        if window_states:
            windows = []
            for i, ws in enumerate(window_states):
                # Calculate window duration
                duration = 0.5
                start_hour = ws["hours_since_admission"]
                time_str = f"{start_hour:.1f}-{start_hour + duration:.1f}"

                # Build comprehensive window entry with all available information
                state = ws.get("state", {})
                window_entry = {
                    "id": f"W{i+1}",
                    "time_range": f"Hour {time_str}",
                    "trajectory": state.get("trajectory", "unknown"),
                    "summary": state.get("summary", ""),
                }

                # Add optional fields only if they exist
                if state.get("physiology"):
                    window_entry["physiology"] = state["physiology"]
                if state.get("key_concerns"):
                    window_entry["key_concerns"] = state["key_concerns"]
                if state.get("interventions"):
                    window_entry["interventions"] = state["interventions"]
                if state.get("uncertainties"):
                    window_entry["uncertainties"] = state["uncertainties"]

                windows.append(window_entry)

            import json
            parts.append(
                f"""## Patient Trajectory
{json.dumps(windows, indent=2)}"""
            )

        # Current/final window events
        if current_events:
            # Calculate time range for final window
            # The final window starts at hours_since_admission and lasts 0.5 hours
            duration = 0.5
            time_range_str = f"Hour {hours_since_admission:.1f}-{hours_since_admission + duration:.1f}"

            events_str = self._format_events(current_events)
            parts.append(
                f"""## Final Window ICU Events ({time_range_str})
{events_str}"""
            )

        # Final state summary
        parts.append(
            f"""## Final Clinical State Summary
{final_state}"""
        )

        # Retrieved experiences for prediction (cross-agent memory)
        if retrieved:
            experiences = []
            for i, result in enumerate(retrieved[: self.max_experiences]):
                entry = result.entry
                outcome = (
                    "Survived" if "survive" in entry.feedback.lower() else "Died" if entry.feedback else "Unknown"
                )
                experiences.append({
                    "id": f"E{i+1}",
                    "relevance": round(result.score, 2),
                    "scenario": entry.input_text,
                    "outcome": outcome,
                    "details": entry.output_text,
                })
            parts.append(
                f"""## Similar Cases from Memory (Cross-Patient Memory)
{json.dumps(experiences, indent=2)}"""
            )

        # Reasoning trace (if any)
        if reasoning_trace:
            parts.append(
                f"""## Reasoning Trace
{chr(10).join(reasoning_trace)}"""
            )

        return "\n\n".join(parts)
