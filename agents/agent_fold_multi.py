"""Multi-Agent Pipeline: Observer + Memory Agent + Predictor.

Decouples the monolithic AgentFold into three specialized agents:
- ObserverAgent: Clinical assessment (what is happening)
- MemoryAgent: Trajectory folding decisions (how to remember)
- PredictorAgent: Survival prediction (final outcome)

The pipeline per window: raw events → Observer → Memory Agent → updated WorkingContext
The Predictor runs once at the end on the final WorkingContext.
"""

import json
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agents.agent_fold import ClinicalConcern, LLMCallLog, MemoryDatabase, TrajectoryEntry, WorkingContext
from model.llms import LLMClient


def _normalize_token_count(value: Any) -> int:
    """Normalize token counts from provider responses."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return 0
        try:
            return int(stripped)
        except ValueError:
            try:
                return int(float(stripped))
            except ValueError:
                return 0
    return 0


def _format_current_events_for_window(
    current_events: List[Dict], window_index: int, hours_since_admission: float
) -> str:
    """Format current events exactly as observer-context event blocks."""
    parts = [f"## Current Events (Window {window_index}, Hour {hours_since_admission:.1f})"]
    for line in _format_current_event_lines(current_events):
        parts.append(line)

    return "\n".join(parts)


def _format_current_event_lines(current_events: List[Dict]) -> List[str]:
    """Format current events as bullet lines (without section header)."""
    lines: List[str] = []
    for event in current_events:
        time = event.get("time", "Unknown")
        code = event.get("code_specifics", event.get("code", "Unknown"))
        numeric_value = event.get("numeric_value")
        text_value = event.get("text_value")

        line = f"- {time}: {code}"
        if numeric_value is not None:
            line += f" = {numeric_value}"
        if text_value:
            line += f" ({text_value})"
        lines.append(line)

    return lines


class ObserverAgent:
    """Clinical assessment agent. Reads raw events + context, outputs clinical assessment."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def observe(
        self,
        working_context: WorkingContext,
        current_events: List[Dict],
        window_index: int,
        hours_since_admission: float,
        prompt_template: str,
    ) -> Tuple[Dict, str]:
        """Produce a clinical assessment for the current window.

        Returns:
            Tuple of (parsed_assessment, raw_response, usage, formatted_prompt)
        """
        window_info = f"(Window {window_index}, Hour {hours_since_admission:.1f})"
        # Observer sees: Patient Metadata, Historical Key Events, Clinical Concerns, Current Events
        context_text = working_context.to_text(
            current_events=current_events,
            current_window_info=window_info,
            sections={
                "patient_metadata": True,
                "historical_key_events": True,
                "trajectory": False,
                "clinical_concerns": True,
                "clinical_trajectory": False,
                "current_events": True,
            },
        )
        prompt = prompt_template.format(context=context_text)
        response = self.llm_client.chat(prompt=prompt, response_format="text")
        content = response.get("content", "")
        usage = response.get("usage", {})

        parsed = _parse_json_response(content)
        return parsed, content, usage, prompt


class MemoryAgent:
    """Memory management agent. Decides APPEND/MERGE and writes trajectory summaries."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def update_memory(
        self,
        working_context: WorkingContext,
        observer_output: Optional[Dict],
        raw_events: Optional[List[Dict]],
        use_observer_input: bool,
        window_index: int,
        hours_since_admission: float,
        window_duration_hours: float,
        prompt_template: str,
    ) -> Tuple[Dict, str]:
        """Decide how to fold the observer's assessment into the trajectory.

        Returns:
            Tuple of (parsed_decision, raw_response, usage, formatted_prompt)
        """
        # Build trajectory text for the memory agent
        trajectory_text = self._build_trajectory_text(working_context)
        if use_observer_input and observer_output is not None:
            window_input = json.dumps(observer_output, indent=2, ensure_ascii=False)
        else:
            window_input = _format_current_events_for_window(
                current_events=raw_events or [],
                window_index=window_index,
                hours_since_admission=hours_since_admission,
            )

        prompt = prompt_template.format(
            trajectory_text=trajectory_text,
            window_input=window_input,
            window_index=window_index,
            num_trajectories=len(working_context.trajectory),
        )
        response = self.llm_client.chat(prompt=prompt, response_format="text")
        content = response.get("content", "")
        usage = response.get("usage", {})

        parsed = _parse_json_response(content)
        return parsed, content, usage, prompt

    def _build_trajectory_text(self, working_context: WorkingContext) -> str:
        """Build a text representation of the current trajectory state."""
        # Memory Agent sees: Patient Metadata, Trajectory entries
        return working_context.to_text(
            sections={
                "patient_metadata": True,
                "historical_key_events": False,
                "trajectory": True,
                "clinical_concerns": False,
                "clinical_trajectory": False,
                "current_events": False,
            }
        )


class PredictorAgent:
    """Survival prediction agent. Runs once at the end."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def predict(
        self,
        working_context: WorkingContext,
        hours_since_admission: float,
        prompt_template: str,
    ) -> Tuple[Dict, str]:
        """Make final survival prediction based on accumulated context.

        Returns:
            Tuple of (parsed_prediction, raw_response, usage, formatted_prompt)
        """
        # Predictor sees: Patient Metadata, Historical Key Events, Trajectory, Clinical Trajectory
        context_text = working_context.to_text(
            current_window_info=f"(Final State, Hour {hours_since_admission:.1f})",
            sections={
                "patient_metadata": True,
                "historical_key_events": True,
                "trajectory": True,
                "clinical_concerns": False,
                "clinical_trajectory": True,
                "current_events": False,
            },
        )
        prompt = prompt_template.format(context=context_text)
        response = self.llm_client.chat(prompt=prompt, response_format="text")
        content = response.get("content", "")
        usage = response.get("usage", {})

        try:
            parsed = _parse_json_response(content)
            return parsed, content, usage, prompt
        except Exception as e:
            print(f"Error parsing prediction: {e}")
            return (
                {
                    "survival_prediction": {
                        "outcome": "unknown",
                        "confidence": 0.0,
                        "rationale": f"Parsing error: {e}",
                    }
                },
                content,
                usage,
                prompt,
            )


class ReflectionAgent:
    """Clinical auditor agent. Reviews trajectory summaries for quality and accuracy."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def audit_trajectory(
        self,
        working_context: WorkingContext,
        new_trajectory: TrajectoryEntry,
        raw_events: List[Dict],
        prompt_template: str,
    ) -> Tuple[Dict, str]:
        """Audit a trajectory summary for clinical accuracy and completeness.

        Returns:
            Tuple of (parsed_audit, raw_response, usage, formatted_prompt)
        """
        # Build previous trajectory context
        previous_traj_parts = []
        for traj in working_context.trajectory:
            if traj.end_window < new_trajectory.start_window:
                if traj.start_window == traj.end_window:
                    index_str = f"T{traj.start_window}"
                else:
                    index_str = f"T{traj.start_window}-{traj.end_window}"
                previous_traj_parts.append(f"{index_str}. {traj.to_text()}")

        previous_trajectory_text = (
            "\n".join(previous_traj_parts) if previous_traj_parts else "(No previous trajectory)"
        )

        # Reflection prompt expects a compact event list for evidence checking.
        raw_events_parts = []
        for event in raw_events:
            time = event.get("time", "Unknown")
            code = event.get("code_specifics", event.get("code", "Unknown"))
            numeric_value = event.get("numeric_value")
            text_value = event.get("text_value")

            line = f"- {time}: {code}"
            if numeric_value is not None:
                line += f" = {numeric_value}"
            if text_value:
                line += f" ({text_value})"
            raw_events_parts.append(line)

        raw_events_text = "\n".join(raw_events_parts) if raw_events_parts else "(No events)"

        # Build prompt
        prompt = prompt_template.format(
            previous_trajectory_text=previous_trajectory_text,
            start_index=new_trajectory.start_window,
            end_index=new_trajectory.end_window,
            start_hour=new_trajectory.start_hour,
            end_hour=new_trajectory.end_hour,
            trajectory_summary=new_trajectory.summary,
            raw_events_text=raw_events_text,
        )

        response = self.llm_client.chat(prompt=prompt, response_format="text")
        content = response.get("content", "")
        usage = response.get("usage", {})

        parsed = _parse_json_response(content)
        return parsed, content, usage, prompt


class MultiAgent:
    """Orchestrator that coordinates Observer, Memory Agent, and Predictor.

    Pipeline per window:
        raw events → ObserverAgent(optional) → MemoryAgent(optional) → updated WorkingContext
    After all windows:
        WorkingContext → PredictorAgent → survival prediction
    """

    def __init__(
        self,
        provider: str,
        model: str = None,
        api_key: str = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        enable_logging: bool = False,
        window_duration_hours: float = 0.5,
        observation_hours: float = 12.0,
        use_observer_agent: bool = True,
        use_memory_agent: bool = True,
        use_reflection_agent: bool = True,
    ):
        self.llm_client = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.window_duration_hours = window_duration_hours
        self.observation_hours = observation_hours
        self.use_observer_agent = use_observer_agent
        self.use_memory_agent = use_memory_agent
        self.use_reflection_agent = use_reflection_agent

        # Sub-agents share the same LLM client
        self.observer = ObserverAgent(self.llm_client) if self.use_observer_agent else None
        if self.use_memory_agent:
            self.memory_agent = MemoryAgent(self.llm_client)
        self.predictor = PredictorAgent(self.llm_client)
        if self.use_reflection_agent and self.use_memory_agent:
            self.reflection_agent = ReflectionAgent(self.llm_client)

        # Statistics
        self.total_patients = 0
        self.total_tokens_used = 0
        self.total_folds = 0
        self.total_appends = 0
        self.total_observer_calls = 0
        self.total_observer_cache_hits = 0
        self.total_memory_calls = 0
        self.total_reflection_calls = 0
        self.total_revisions = 0

        # Logging
        self.enable_logging = enable_logging
        self.call_logs: List[LLMCallLog] = []
        self._current_patient_id: str = ""
        self._current_windows: List[Dict] = []  # For reflection agent
        self._observer_outputs: List[Dict] = []  # For ablation when memory agent is disabled

    def _log_call(
        self,
        step_type: str,
        window_index: int,
        hours: float,
        prompt: str,
        response: str,
        usage: Dict,
        metadata: Dict = None,
        parsed_response: Optional[Dict] = None,
    ) -> None:
        """Log an LLM call if logging is enabled."""
        usage_dict = usage if isinstance(usage, dict) else {}
        input_tokens = _normalize_token_count(usage_dict.get("input_tokens", 0))
        output_tokens = _normalize_token_count(usage_dict.get("output_tokens", 0))
        self.total_tokens_used += input_tokens + output_tokens

        if self.enable_logging:
            log_metadata = dict(metadata or {})
            log_metadata.update(
                {
                    "step_type": step_type,
                    "llm_provider": self.llm_client.provider,
                    "llm_model": self.llm_client.model,
                }
            )
            log_entry = LLMCallLog(
                timestamp=datetime.now().isoformat(),
                patient_id=self._current_patient_id,
                window_index=window_index,
                hours_since_admission=hours,
                prompt=prompt,
                response=response,
                parsed_response=parsed_response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata=log_metadata,
            )
            self.call_logs.append(log_entry)

    def _apply_observer_output(self, working_context: WorkingContext, observer_output: Dict) -> None:
        """Apply observer's clinical assessment to working context (concerns + key events)."""
        clinical = observer_output.get("clinical_assessment", {})

        # Add clinical assessment to history
        working_context.add_clinical_assessment(clinical)

        # Extract critical events → append to historical key events
        critical_events = clinical.get("critical_events", [])
        if isinstance(critical_events, list):
            for event_obj in critical_events:
                time_str = event_obj.get("time", "")
                event_name = event_obj.get("event", "")
                if time_str and event_name and event_name != "None":
                    working_context.add_key_event(f"{time_str}: {event_name}")

        # # Update concerns
        # concerns_update = clinical.get("active_concerns_update", [])
        # updated_concerns = []
        # for c in concerns_update:
        #     note = c.get("concern", c.get("note", ""))
        #     updated_concerns.append(
        #         ClinicalConcern(
        #             concern_id=c.get("id", ""),
        #             status=c.get("status", "Active"),
        #             note=note,
        #         )
        #     )
        # working_context.update_concerns(updated_concerns)

    def _reflection_loop(
        self,
        working_context: WorkingContext,
        new_entry: TrajectoryEntry,
        observer_output: Dict,
        current_events: List[Dict],
        window_index: int,
    ) -> TrajectoryEntry:
        """Run reflection agent review and handle re-summarization if needed."""
        from prompts.agent_multi_prompts import get_memory_agent_prompt, get_reflection_agent_prompt

        # Extract raw events for the trajectory time range
        raw_events = []
        for idx, window in enumerate(self._current_windows):
            win_idx = window.get("window_index", idx)
            if new_entry.start_window <= win_idx <= new_entry.end_window:
                raw_events.extend(window.get("current_events", []))

        # Call reflection agent
        reflection_prompt = get_reflection_agent_prompt()
        audit_result, audit_raw, audit_usage, audit_formatted = self.reflection_agent.audit_trajectory(
            working_context=working_context,
            new_trajectory=new_entry,
            raw_events=raw_events,
            prompt_template=reflection_prompt,
        )

        self.total_reflection_calls += 1
        self._log_call(
            "reflection_agent",
            window_index,
            new_entry.end_hour,
            audit_formatted,
            audit_raw,
            audit_usage,
            {"needs_revision": audit_result.get("needs_revision", False)},
            parsed_response=audit_result,
        )

        # If revision needed, call memory agent again with feedback
        if audit_result.get("needs_revision", False):
            self.total_revisions += 1
            revision_instructions = audit_result.get("revision_instructions", "")

            # Build enhanced memory prompt with feedback
            memory_prompt = get_memory_agent_prompt()

            # Include the trajectory being revised
            if new_entry.start_window == new_entry.end_window:
                traj_label = f"T{new_entry.start_window}"
            else:
                traj_label = f"T{new_entry.start_window}-{new_entry.end_window}"

            previous_summary_section = f"\n\n### YOUR PREVIOUS TRAJECTORY SUMMARY\n{traj_label} (Hour {new_entry.start_hour:.1f} to {new_entry.end_hour:.1f}):\n{new_entry.summary}"
            feedback_section = f"\n\n### CLINICAL AUDITOR FEEDBACK\n{revision_instructions}\n\nPlease revise your summary to address these concerns."
            enhanced_prompt = memory_prompt + previous_summary_section + feedback_section

            # Re-run memory agent
            memory_revised, memory_raw_revised, memory_usage_revised, memory_formatted_revised = (
                self.memory_agent.update_memory(
                    working_context=working_context,
                    observer_output=observer_output if self.use_observer_agent else None,
                    raw_events=current_events if not self.use_observer_agent else None,
                    use_observer_input=self.use_observer_agent,
                    window_index=window_index,
                    hours_since_admission=new_entry.end_hour,
                    window_duration_hours=self.window_duration_hours,
                    prompt_template=enhanced_prompt,
                )
            )

            self.total_memory_calls += 1
            self._log_call(
                "memory_agent_revision",
                window_index,
                new_entry.end_hour,
                memory_formatted_revised,
                memory_raw_revised,
                memory_usage_revised,
                {"revision_attempt": True},
                parsed_response=memory_revised,
            )

            # Extract revised summary
            memory_mgmt_revised = memory_revised.get("memory_management", {})
            traj_update_revised = memory_mgmt_revised.get("trajectory_update", {})
            revised_summary = traj_update_revised.get("refined_summary", new_entry.summary)

            # Update entry with revised summary
            new_entry = TrajectoryEntry(
                start_window=new_entry.start_window,
                end_window=new_entry.end_window,
                start_hour=new_entry.start_hour,
                end_hour=new_entry.end_hour,
                summary=revised_summary,
            )

        return new_entry

    def _apply_memory_output(
        self,
        working_context: WorkingContext,
        memory_output: Dict,
        observer_output: Dict,
        current_events: List[Dict],
        window_index: int,
        hours_since_admission: float,
    ) -> None:
        """Apply memory agent's folding decision to working context."""
        memory_mgmt = memory_output.get("memory_management", {})
        traj_update = memory_mgmt.get("trajectory_update", {})

        start_idx = traj_update.get("start_index")
        end_idx = traj_update.get("end_index")
        summary = traj_update.get("refined_summary", "")

        # Fallback summary from observer if memory agent didn't provide one
        if not summary:
            summary = observer_output.get("clinical_summary", f"Window {window_index} events")

        if start_idx is not None and end_idx is not None and summary:
            new_entry = TrajectoryEntry(
                start_window=start_idx,
                end_window=end_idx,
                start_hour=self._get_window_start_hour(working_context, start_idx),
                end_hour=hours_since_admission + self.window_duration_hours,
                summary=summary,
            )

            # Reflection agent review (if enabled)
            if self.use_reflection_agent:
                new_entry = self._reflection_loop(
                    working_context=working_context,
                    new_entry=new_entry,
                    observer_output=observer_output,
                    current_events=current_events,
                    window_index=window_index,
                )

            if start_idx == end_idx:
                # APPEND
                working_context.update_trajectory(new_entry, fold_range=None)
                self.total_appends += 1
            else:
                # MERGE
                working_context.update_trajectory(new_entry, fold_range=(start_idx, end_idx))
                self.total_folds += 1
        else:
            # Fallback: simple append
            new_entry = TrajectoryEntry(
                start_window=window_index,
                end_window=window_index,
                start_hour=hours_since_admission,
                end_hour=hours_since_admission + self.window_duration_hours,
                summary=summary,
            )
            working_context.update_trajectory(new_entry, fold_range=None)
            self.total_appends += 1

    def _get_window_start_hour(self, working_context: WorkingContext, window_index: int) -> float:
        """Get the start hour for a given window index."""
        for traj in working_context.trajectory:
            if traj.start_window == window_index:
                return traj.start_hour
        return window_index * self.window_duration_hours

    def _predict_without_memory(
        self,
        working_context: WorkingContext,
        hours_since_admission: float,
    ) -> Tuple[Dict, str, Dict, str]:
        """Make prediction without memory agent (ablation mode).

        Uses window-level evidence directly instead of compressed trajectory.

        Returns:
            Tuple of (parsed_prediction, raw_response, usage, formatted_prompt)
        """
        from prompts.predictor_prompts import get_prediction_prompt

        # Build context with patient metadata and observer summaries
        context_parts = []

        # Patient metadata
        metadata = working_context.patient_metadata
        context_parts.append("## Patient Information")
        context_parts.append(f"Age: {metadata.get('age', 'Unknown')}")
        context_parts.append(f"Gender: {metadata.get('gender', 'Unknown')}")
        context_parts.append("")

        # Historical key events
        if working_context.historical_key_events:
            context_parts.append("## Historical Key Events")
            for event in working_context.historical_key_events:
                context_parts.append(f"- {event}")
            context_parts.append("")

        if self.use_observer_agent:
            # Observer outputs (window-by-window clinical assessments)
            context_parts.append("## Clinical Assessments (Window-by-Window)")
            context_parts.append(f"Observation Period: First {hours_since_admission:.1f} hours after ICU admission")
            context_parts.append("")
            for obs_data in self._observer_outputs:
                window_idx = obs_data["window_index"]
                hours = obs_data["hours_since_admission"]
                obs_output = obs_data["observer_output"]

                clinical_summary = obs_output.get("clinical_summary", "")
                clinical = obs_output.get("clinical_assessment", {})
                overall_status = clinical.get("overall_status", "unknown")
                physiology_trends = clinical.get("physiology_trends", {})
                hemodynamics_status = physiology_trends.get("hemodynamics", {}).get("status", "unknown")
                respiratory_status = physiology_trends.get("respiratory", {}).get("status", "unknown")
                renal_metabolic_status = physiology_trends.get("renal_metabolic", {}).get("status", "unknown")
                neurology_status = physiology_trends.get("neurology", {}).get("status", "unknown")

                context_parts.append(f"### Window {window_idx} (Hour {hours:.1f})")
                context_parts.append(f"Status: {overall_status}")
                context_parts.append(f"Hemodynamics: {hemodynamics_status}")
                context_parts.append(f"Respiratory: {respiratory_status}")
                context_parts.append(f"Renal/Metabolic: {renal_metabolic_status}")
                context_parts.append(f"Neurology: {neurology_status}")
                context_parts.append(f"Summary: {clinical_summary}")
                context_parts.append("")
        else:
            # Observer disabled: predictor consumes one merged raw-event list.
            context_parts.append("## Current Events")
            context_parts.append(f"Observation Period: First {hours_since_admission:.1f} hours after ICU admission")
            context_parts.append("")
            merged_raw_events: List[Dict] = []
            for obs_data in self._observer_outputs:
                raw_events = obs_data.get("raw_events", [])
                if isinstance(raw_events, list):
                    merged_raw_events.extend(raw_events)

            event_lines = _format_current_event_lines(merged_raw_events)
            if event_lines:
                context_parts.extend(event_lines)
            else:
                context_parts.append("- (No events)")
            context_parts.append("")

        context_text = "\n".join(context_parts)
        prompt = get_prediction_prompt(
            observation_hours=self.observation_hours,
        ).format(context=context_text)

        response = self.llm_client.chat(prompt=prompt, response_format="text")
        content = response.get("content", "")
        usage = response.get("usage", {})

        try:
            parsed = _parse_json_response(content)
            return parsed, content, usage, prompt
        except Exception as e:
            print(f"Error parsing prediction: {e}")
            return (
                {
                    "survival_prediction": {
                        "outcome": "unknown",
                        "confidence": 0.0,
                        "rationale": f"Parsing error: {e}",
                    }
                },
                content,
                usage,
                prompt,
            )

    def process_window(
        self,
        working_context: WorkingContext,
        current_events: List[Dict],
        window_index: int,
        hours_since_admission: float,
        observer_prompt: str,
        memory_prompt: str,
        precomputed_observer_output: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict, WorkingContext]:
        """Process a single window through Observer → Memory Agent pipeline.

        Returns:
            Tuple of (observer_parsed, memory_parsed, updated_working_context)
        """
        # Step 1: Observer (or raw-event pass-through when observer is disabled)
        if self.use_observer_agent:
            if precomputed_observer_output is not None:
                observer_parsed = precomputed_observer_output
                observer_raw = json.dumps(precomputed_observer_output, ensure_ascii=False)
                observer_usage = {}
                observer_formatted = "[REUSED_OBSERVER_OUTPUT]"
                self.total_observer_cache_hits += 1
                self._log_call(
                    "observer_cache_reuse",
                    window_index,
                    hours_since_admission,
                    observer_formatted,
                    observer_raw,
                    observer_usage,
                    {"num_events": len(current_events), "cache_reuse": True},
                    parsed_response=observer_parsed,
                )
            else:
                observer_parsed, observer_raw, observer_usage, observer_formatted = self.observer.observe(
                    working_context=working_context,
                    current_events=current_events,
                    window_index=window_index,
                    hours_since_admission=hours_since_admission,
                    prompt_template=observer_prompt,
                )
                self.total_observer_calls += 1
                self._log_call(
                    "observer",
                    window_index,
                    hours_since_admission,
                    observer_formatted,
                    observer_raw,
                    observer_usage,
                    {"num_events": len(current_events)},
                    parsed_response=observer_parsed,
                )

            # Apply observer output (concerns + key events)
            self._apply_observer_output(working_context, observer_parsed)
        else:
            observer_parsed = {}

        # Step 2: Memory Agent (if enabled)
        memory_parsed = {}
        if self.use_memory_agent:
            memory_parsed, memory_raw, memory_usage, memory_formatted = self.memory_agent.update_memory(
                working_context=working_context,
                observer_output=observer_parsed if self.use_observer_agent else None,
                raw_events=current_events if not self.use_observer_agent else None,
                use_observer_input=self.use_observer_agent,
                window_index=window_index,
                hours_since_admission=hours_since_admission,
                window_duration_hours=self.window_duration_hours,
                prompt_template=memory_prompt,
            )
            self.total_memory_calls += 1
            self._log_call(
                "memory_agent",
                window_index,
                hours_since_admission,
                memory_formatted,
                memory_raw,
                memory_usage,
                {"num_trajectories": len(working_context.trajectory)},
                parsed_response=memory_parsed,
            )

            # Apply memory output (trajectory folding)
            self._apply_memory_output(
                working_context,
                memory_parsed,
                observer_parsed,
                current_events,
                window_index,
                hours_since_admission,
            )

        return observer_parsed, memory_parsed, working_context

    def run_patient_trajectory(
        self,
        windows: List[Dict],
        patient_metadata: Dict,
        precomputed_observer_outputs: Optional[List[Dict]] = None,
        verbose: bool = True,
    ) -> Tuple[Dict, WorkingContext, MemoryDatabase]:
        """Run the full multi-agent pipeline on a patient trajectory.

        Args:
            windows: List of time windows from data parser
            patient_metadata: Patient metadata (age, etc.)
            precomputed_observer_outputs: Optional observer outputs for reuse,
                formatted as a list of {"window_index", "hours_since_admission", "observer_output"}
            verbose: Print progress

        Returns:
            Tuple of (prediction_dict, final_working_context, memory_database)
        """
        from prompts.agent_multi_prompts import get_memory_agent_prompt, get_observer_prompt, get_predictor_prompt

        self.total_patients += 1
        memory_db = MemoryDatabase(patient_metadata=patient_metadata)
        working_context = WorkingContext(patient_metadata=patient_metadata)

        # Store windows for reflection agent (if enabled)
        self._current_windows = windows
        self._observer_outputs = []  # Reset for new patient

        self._current_patient_id = (
            f"{patient_metadata.get('subject_id', 'unknown')}" f"_{patient_metadata.get('icu_stay_id', 'unknown')}"
        )

        if verbose:
            print(f"Processing patient with {len(windows)} windows...")

        observer_prompt = get_observer_prompt() if self.use_observer_agent else ""
        memory_prompt = get_memory_agent_prompt()
        precomputed_by_window: Dict[int, Dict] = {}
        if self.use_observer_agent and precomputed_observer_outputs:
            for item in precomputed_observer_outputs:
                if not isinstance(item, dict):
                    continue
                idx = item.get("window_index")
                if isinstance(idx, int):
                    precomputed_by_window[idx] = item

        # Process ALL windows through Observer + Memory Agent
        for i, window in enumerate(windows):
            current_events = window.get("current_events", [])
            hours = window.get("hours_since_admission", 0)

            if not current_events:
                continue

            if verbose:
                print(f"  Window {i+1}/{len(windows)} (Hour {hours:.1f})...", end=" ")

            precomputed_observer = None
            cached_entry = precomputed_by_window.get(i) if self.use_observer_agent else None
            if self.use_observer_agent and cached_entry:
                cached_hours = cached_entry.get("hours_since_admission")
                hours_match = False
                if cached_hours is None:
                    hours_match = True
                else:
                    try:
                        hours_match = abs(float(cached_hours) - float(hours)) < 1e-6
                    except (TypeError, ValueError):
                        hours_match = False
                if hours_match:
                    precomputed_observer = cached_entry.get("observer_output")

            observer_parsed, memory_parsed, working_context = self.process_window(
                working_context=working_context,
                current_events=current_events,
                window_index=i,
                hours_since_admission=hours,
                observer_prompt=observer_prompt,
                memory_prompt=memory_prompt,
                precomputed_observer_output=precomputed_observer,
            )

            # Store observer outputs for ablation/caching.
            self._observer_outputs.append(
                {
                    "window_index": i,
                    "hours_since_admission": hours,
                    "observer_output": observer_parsed,
                    "raw_events": deepcopy(current_events),
                }
            )

            # Record to memory database
            clinical = observer_parsed.get("clinical_assessment", {})
            window_record = {
                "window_index": i,
                "hours_since_admission": hours,
                "num_events": len(current_events),
                "raw_events": current_events,
                "observer_output": observer_parsed,
                "observer_enabled": self.use_observer_agent,
                "memory_output": memory_parsed,
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
                # "active_concerns": [
                #     {"id": c.concern_id, "status": c.status, "note": c.note} for c in working_context.active_concerns
                # ],
            }
            memory_db.add_window(window_record)

            if verbose:
                status = clinical.get("overall_status", "unknown") if self.use_observer_agent else "raw_events"
                print(f"Status: {status}")

        # Final prediction
        last_hours = windows[-1].get("hours_since_admission", 0) if windows else 0
        if verbose:
            print(f"\nMaking final prediction (Hour {last_hours:.1f})...")

        # Build predictor context based on whether memory agent is enabled
        if self.use_memory_agent:
            # Standard path: use trajectory from memory agent
            prediction, pred_raw, pred_usage, pred_formatted = self.predictor.predict(
                working_context=working_context,
                hours_since_admission=last_hours,
                prompt_template=get_predictor_prompt(
                    observation_hours=self.observation_hours,
                ),
            )
        else:
            # Ablation path: use observer outputs directly
            prediction, pred_raw, pred_usage, pred_formatted = self._predict_without_memory(
                working_context=working_context,
                hours_since_admission=last_hours,
            )

        self._log_call(
            "predictor",
            -1,
            last_hours,
            pred_formatted,
            pred_raw,
            pred_usage,
            {"num_trajectories": len(working_context.trajectory)},
            parsed_response=prediction,
        )

        return prediction, working_context, memory_db

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_patients": self.total_patients,
            "total_tokens_used": self.total_tokens_used,
            "total_folds": self.total_folds,
            "total_appends": self.total_appends,
            "total_observer_calls": self.total_observer_calls,
            "total_observer_cache_hits": self.total_observer_cache_hits,
            "total_memory_calls": self.total_memory_calls,
            "total_reflection_calls": self.total_reflection_calls,
            "total_revisions": self.total_revisions,
            "total_llm_calls": len(self.call_logs) if self.enable_logging else 0,
        }

    def get_logs(self) -> List[Dict]:
        """Get all LLM call logs as list of dictionaries."""
        return [log.to_dict() for log in self.call_logs]

    def get_observer_outputs(self) -> List[Dict]:
        """Get observer outputs from the current patient run."""
        return deepcopy(self._observer_outputs)

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


# --- Shared utility ---


def _parse_json_response(response: str) -> Dict:
    """Parse JSON from LLM response."""

    def _parse_candidate(candidate: str) -> Optional[Dict]:
        candidate = candidate.strip()
        if not candidate:
            return None

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", candidate, re.IGNORECASE)
        for block in fenced_blocks:
            block = block.strip()
            if not block:
                continue
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

        decoder = json.JSONDecoder()
        for i, char in enumerate(candidate):
            if char not in "{[":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[i:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        return None

    candidates = []
    raw_response = response.strip()
    if raw_response:
        candidates.append(raw_response)

    for candidate in candidates:
        parsed = _parse_candidate(candidate)
        if parsed is not None:
            return parsed

    raise ValueError(f"Could not parse JSON from response: {response[:200]}...")
