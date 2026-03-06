"""MedAgent: static + dynamic memory for ICU survival prediction."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agents.agent_fold_multi import _normalize_token_count, _parse_json_response
from model.llms import LLMClient
from prompts.med_agent_prompts import get_dynamic_memory_update_prompt, get_med_predictor_prompt
from utils.static_memory_extractor import extract_static_memory


@dataclass
class StaticMemory:
    """Static memory extracted once per ICU stay."""

    age: Optional[float] = None
    gender: Optional[str] = None
    admission_diagnoses: List[str] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StaticMemory":
        demographics = data.get("demographics", {})
        return cls(
            age=demographics.get("age"),
            gender=demographics.get("gender"),
            admission_diagnoses=demographics.get("admission_diagnoses", []),
            summary=data.get("summary", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "age": self.age,
            "gender": self.gender,
            "admission_diagnoses": self.admission_diagnoses,
            "summary": self.summary,
        }

    def to_text(self) -> str:
        parts: List[str] = ["## Static Memory"]
        if self.age is None:
            parts.append("age: Unknown")
        else:
            try:
                parts.append(f"age: {float(self.age):.1f}")
            except (TypeError, ValueError):
                parts.append(f"age: {self.age}")

        parts.append(f"gender: {self.gender if self.gender else 'Unknown'}")
        if self.admission_diagnoses:
            parts.append("admission_diagnoses: " + "; ".join(self.admission_diagnoses))
        else:
            parts.append("admission_diagnoses: Unknown")
        parts.append("summary: " + (self.summary if self.summary else "None"))
        return "\n".join(parts)


@dataclass
class DynamicMemory:
    """Dynamic memory updated every window."""

    current_status: str = "No dynamic memory yet."
    active_problems: List[str] = field(default_factory=list)
    critical_events_log: List[Dict[str, str]] = field(default_factory=list)
    trends: List[str] = field(default_factory=list)
    interventions_responses: List[str] = field(default_factory=list)
    patient_specific_patterns: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DynamicMemory":
        normalized = normalize_dynamic_memory_payload(data)
        return cls(
            current_status=normalized["current_status"],
            active_problems=normalized["active_problems"],
            critical_events_log=normalized["critical_events_log"],
            trends=normalized["trends"],
            interventions_responses=normalized["interventions_responses"],
            patient_specific_patterns=normalized["patient_specific_patterns"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_status": self.current_status,
            "active_problems": self.active_problems,
            "critical_events_log": self.critical_events_log,
            "trends": self.trends,
            "interventions_responses": self.interventions_responses,
            "patient_specific_patterns": self.patient_specific_patterns,
        }

    def to_text(self, include_critical_events: bool = True) -> str:
        parts: List[str] = ["## Dynamic Memory"]
        parts.append("### Current Status")
        parts.append(self.current_status)

        parts.append("")
        parts.append("### Active Problems")
        if self.active_problems:
            for item in self.active_problems:
                parts.append(f"- {item}")
        else:
            parts.append("- None")

        if include_critical_events:
            parts.append("")
            parts.append("### Critical Events Log")
            if self.critical_events_log:
                for line in _format_critical_events_natural_language_lines(self.critical_events_log):
                    parts.append(f"- {line}")
            else:
                parts.append("- None")

        parts.append("")
        parts.append("### Trends")
        if self.trends:
            for item in self.trends:
                parts.append(f"- {item}")
        else:
            parts.append("- None")

        parts.append("")
        parts.append("### Interventions & Responses")
        if self.interventions_responses:
            for item in self.interventions_responses:
                parts.append(f"- {item}")
        else:
            parts.append("- None")

        parts.append("")
        parts.append("### Patient-Specific Patterns")
        if self.patient_specific_patterns:
            for item in self.patient_specific_patterns:
                parts.append(f"- {item}")
        else:
            parts.append("- None")

        return "\n".join(parts)

    def to_prompt_text(self) -> str:
        """Compact dynamic-memory view for memory-agent prompts (without historical critical events)."""
        return self.to_text(include_critical_events=False)


@dataclass
class DynamicMemorySnapshot:
    """Snapshot of dynamic memory after each window."""

    window_index: int
    hours_since_admission: float
    num_current_events: int
    dynamic_memory: DynamicMemory
    update_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_index": self.window_index,
            "hours_since_admission": self.hours_since_admission,
            "num_current_events": self.num_current_events,
            "update_error": self.update_error,
            "dynamic_memory": self.dynamic_memory.to_dict(),
        }


@dataclass
class MedAgentOutput:
    """Structured output bundle for MedAgent trajectory run."""

    static_memory: StaticMemory
    final_dynamic_memory: DynamicMemory
    dynamic_memory_history: List[DynamicMemorySnapshot]
    last_window_events: List[Dict[str, Any]]
    final_state_text: str

    def patient_memory_dict(self) -> Dict[str, Any]:
        return {
            "static_memory": self.static_memory.to_dict(),
            "final_dynamic_memory": self.final_dynamic_memory.to_dict(),
            "final_state_text": self.final_state_text,
        }

    def dynamic_history_dict(self) -> Dict[str, Any]:
        return {
            "num_snapshots": len(self.dynamic_memory_history),
            "snapshots": [snapshot.to_dict() for snapshot in self.dynamic_memory_history],
            "last_window_events": deepcopy(self.last_window_events),
        }


@dataclass
class LLMCallLog:
    """Log entry for a single LLM call."""

    timestamp: str
    patient_id: str
    window_index: int
    hours_since_admission: float
    prompt: str
    response: str
    parsed_response: Optional[Dict[str, Any]]
    input_tokens: int
    output_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _dedupe_strings(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        text = _safe_text(item)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _parse_datetime(value: str) -> Optional[datetime]:
    text = _safe_text(value)
    if not text:
        return None

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    normalized: List[str] = []
    for item in value:
        if isinstance(item, dict):
            candidate = _safe_text(item.get("text") or item.get("item") or item.get("value") or item.get("event"))
        else:
            candidate = _safe_text(item)
        if candidate:
            normalized.append(candidate)
    return _dedupe_strings(normalized)


def _normalize_critical_events(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []

    normalized: List[Dict[str, str]] = []
    seen = set()

    for item in value:
        time_text = ""
        event_text = ""

        if isinstance(item, dict):
            time_text = _safe_text(item.get("time"))
            event_text = _safe_text(item.get("event") or item.get("description") or item.get("text"))
        else:
            line = _safe_text(item)
            if not line:
                continue
            datetime_prefix = re.match(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?)\s*:\s*(.+)$", line)
            if datetime_prefix:
                time_text = _safe_text(datetime_prefix.group(1))
                event_text = _safe_text(datetime_prefix.group(2))
            elif ":" in line:
                left, right = line.split(":", 1)
                time_text = _safe_text(left)
                event_text = _safe_text(right)
            else:
                event_text = line

        if not event_text:
            continue

        key = (time_text.lower(), event_text.lower())
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"time": time_text, "event": event_text})

    normalized.sort(
        key=lambda item: (
            _parse_datetime(item.get("time", "")) is None,
            _parse_datetime(item.get("time", "")) or datetime.max,
            item.get("event", ""),
        )
    )
    return normalized


def normalize_dynamic_memory_payload(
    payload: Optional[Dict[str, Any]],
    max_active_problems: int = 8,
    max_critical_events: int = 20,
    max_patterns: int = 8,
    include_critical_events: bool = True,
) -> Dict[str, Any]:
    """Normalize/validate dynamic memory payload to strict schema."""
    data = payload if isinstance(payload, dict) else {}

    current_status = _safe_text(data.get("current_status"))
    if not current_status:
        current_status = "No significant update."

    active_problems = _normalize_string_list(data.get("active_problems"))[:max_active_problems]

    critical_events_log: List[Dict[str, str]] = []
    if include_critical_events:
        critical_events_log = _normalize_critical_events(data.get("critical_events_log"))
        if len(critical_events_log) > max_critical_events:
            critical_events_log = critical_events_log[-max_critical_events:]

    trends = _normalize_string_list(data.get("trends"))[: max_critical_events * 2]
    interventions_responses = _normalize_string_list(data.get("interventions_responses"))[: max_critical_events * 2]
    patient_specific_patterns = _normalize_string_list(data.get("patient_specific_patterns"))[:max_patterns]

    return {
        "current_status": current_status,
        "active_problems": active_problems,
        "critical_events_log": critical_events_log,
        "trends": trends,
        "interventions_responses": interventions_responses,
        "patient_specific_patterns": patient_specific_patterns,
    }


def _merge_critical_events(
    existing_events: List[Dict[str, str]],
    new_events: List[Dict[str, str]],
    max_critical_events: int,
) -> List[Dict[str, str]]:
    merged = _normalize_critical_events(list(existing_events) + list(new_events))
    if len(merged) > max_critical_events:
        merged = merged[-max_critical_events:]
    return merged


def _format_critical_events_natural_language_lines(critical_events: List[Dict[str, str]]) -> List[str]:
    normalized = _normalize_critical_events(critical_events)
    lines: List[str] = []
    for item in normalized:
        time_text = _safe_text(item.get("time"))
        event_text = _safe_text(item.get("event"))
        if not event_text:
            continue
        if time_text:
            lines.append(f"{time_text} {event_text}")
        else:
            lines.append(event_text)
    return lines


def _format_critical_events_natural_language(critical_events: List[Dict[str, str]]) -> str:
    lines = _format_critical_events_natural_language_lines(critical_events)
    if not lines:
        return "- None"
    return "\n".join(lines)


def _format_current_events(current_events: List[Dict[str, Any]]) -> str:
    if not current_events:
        return "- (No events)"

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

    return "\n".join(lines)


def _build_static_summary_fallback(static_memory: Dict[str, Any]) -> str:
    demographics = static_memory.get("demographics", {})
    pmh = static_memory.get("past_medical_history", [])
    meds = static_memory.get("admission_medications", [])

    age = demographics.get("age")
    gender = demographics.get("gender")
    intro = (
        f"Patient age {age}, gender {gender}."
        if age is not None or gender is not None
        else "Patient demographics available."
    )

    diagnosis_text = ""
    admission_dx = demographics.get("admission_diagnoses", [])
    if admission_dx:
        diagnosis_text = f" Admission diagnoses: {'; '.join(admission_dx[:5])}."

    pmh_text = ""
    if pmh:
        pmh_text = f" PMH includes: {'; '.join(pmh[:6])}."

    meds_text = ""
    if meds:
        meds_text = f" Admission medications include: {'; '.join(meds[:6])}."

    labs = static_memory.get("baseline_labs", {})
    unavailable = [name for name, value in labs.items() if value is None]
    if unavailable:
        labs_text = " Baseline labs partially unavailable: " + ", ".join(unavailable[:4]) + "."
    else:
        labs_text = " Baseline labs available for key markers."

    return (intro + diagnosis_text + pmh_text + meds_text + labs_text).strip()


class StaticMemoryBuilder:
    """Extract and optionally compress static memory."""

    def __init__(self, llm_client: LLMClient, use_llm_static_compression: bool = True):
        self.llm_client = llm_client
        self.use_llm_static_compression = use_llm_static_compression

    def build(
        self,
        trajectory: Dict[str, Any],
        baseline_lab_lookback_start_hours: float,
        baseline_lab_lookback_end_hours: float,
    ) -> Tuple[StaticMemory, Optional[Dict[str, Any]]]:
        static_payload = extract_static_memory(
            trajectory=trajectory,
            baseline_lab_lookback_start_hours=baseline_lab_lookback_start_hours,
            baseline_lab_lookback_end_hours=baseline_lab_lookback_end_hours,
        )

        llm_call: Optional[Dict[str, Any]] = None
        summary = _build_static_summary_fallback(static_payload)

        if self.use_llm_static_compression:
            prompt = self._build_static_summary_prompt(static_payload)
            response = self.llm_client.chat(prompt=prompt, response_format="text")
            raw = response.get("content", "")
            usage = response.get("usage", {})

            parsed = None
            parse_error = None
            try:
                parsed = _parse_json_response(raw)
            except Exception as exc:  # pragma: no cover - covered by fallback behavior tests
                parse_error = str(exc)

            if isinstance(parsed, dict):
                candidate = _safe_text(parsed.get("static_summary"))
                if candidate:
                    summary = candidate

            llm_call = {
                "prompt": prompt,
                "response": raw,
                "usage": usage,
                "parsed_response": parsed,
                "parse_error": parse_error,
            }

        demographics = static_payload.get("demographics", {})
        static_memory = StaticMemory(
            age=demographics.get("age"),
            gender=demographics.get("gender"),
            admission_diagnoses=demographics.get("admission_diagnoses", []),
            summary=summary,
        )
        return static_memory, llm_call

    def _build_static_summary_prompt(self, static_payload: Dict[str, Any]) -> str:
        payload_json = json.dumps(static_payload, indent=2, ensure_ascii=False)
        return f"""You are summarizing static ICU patient memory.

Return JSON only in <response></response> with this schema:
<response>
{{
  "static_summary": "2-4 sentence concise summary. Mention unavailable baseline labs explicitly if any are missing."
}}
</response>

Static memory payload:
{payload_json}
"""


class DynamicMemoryAgent:
    """LLM-powered dynamic memory updater."""

    def __init__(
        self,
        llm_client: LLMClient,
        use_thinking: bool = True,
        max_active_problems: int = 8,
        max_critical_events: int = 20,
        max_patterns: int = 8,
    ):
        self.llm_client = llm_client
        self.use_thinking = use_thinking
        self.max_active_problems = max_active_problems
        self.max_critical_events = max_critical_events
        self.max_patterns = max_patterns

    def update_memory(
        self,
        previous_memory: DynamicMemory,
        static_memory: StaticMemory,
        current_events: List[Dict[str, Any]],
        window_index: int,
        hours_since_admission: float,
    ) -> Tuple[DynamicMemory, str, Dict[str, Any], str, Optional[str], Optional[Dict[str, Any]]]:
        prompt_template = get_dynamic_memory_update_prompt(
            use_thinking=self.use_thinking,
            max_active_problems=self.max_active_problems,
            max_critical_events=self.max_critical_events,
            max_patterns=self.max_patterns,
        )

        prompt = prompt_template.format(
            static_memory_text=static_memory.to_text(),
            previous_dynamic_memory_text=previous_memory.to_prompt_text(),
            current_events_text=_format_current_events(current_events),
            window_index=window_index,
            hours_since_admission=hours_since_admission,
        )

        response = self.llm_client.chat(prompt=prompt, response_format="text")
        raw = response.get("content", "")
        usage = response.get("usage", {})

        parsed_response: Optional[Dict[str, Any]] = None
        update_error: Optional[str] = None
        normalized_payload = previous_memory.to_dict()

        try:
            parsed_response = _parse_json_response(raw)
            candidate = parsed_response.get("updated_dynamic_memory", parsed_response)
            normalized_payload = normalize_dynamic_memory_payload(
                candidate,
                max_active_problems=self.max_active_problems,
                max_critical_events=self.max_critical_events,
                max_patterns=self.max_patterns,
                include_critical_events=False,
            )

            # Memory agent outputs ONLY new events for this window; code appends cumulatively.
            new_critical_events = _normalize_critical_events(
                parsed_response.get("new_critical_events", candidate.get("new_critical_events", []))
            )
            if not new_critical_events and isinstance(candidate, dict):
                # Backward compatibility fallback for older prompt outputs.
                new_critical_events = _normalize_critical_events(candidate.get("critical_events_log", []))

            normalized_payload["critical_events_log"] = _merge_critical_events(
                existing_events=previous_memory.critical_events_log,
                new_events=new_critical_events,
                max_critical_events=self.max_critical_events,
            )
        except Exception as exc:
            update_error = f"dynamic_memory_parse_error: {exc}"

        updated_memory = DynamicMemory.from_dict(normalized_payload)
        return updated_memory, raw, usage, prompt, update_error, parsed_response


class PredictorAgent:
    """Final predictor using static + dynamic memory + last window events."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def predict(
        self,
        static_memory: StaticMemory,
        dynamic_memory: DynamicMemory,
        last_window_events: List[Dict[str, Any]],
        observation_hours: float,
        use_thinking: bool,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], str]:
        context_parts = [
            static_memory.to_text(),
            "",
            dynamic_memory.to_text(include_critical_events=False),
            "",
            "## Critical Events Timeline",
            _format_critical_events_natural_language(dynamic_memory.critical_events_log),
            "",
            "## Last Window Raw Events",
            _format_current_events(last_window_events),
        ]
        context = "\n".join(context_parts)

        prompt_template = get_med_predictor_prompt(use_thinking=use_thinking, observation_hours=observation_hours)
        prompt = prompt_template.format(context=context)

        response = self.llm_client.chat(prompt=prompt, response_format="text")
        raw = response.get("content", "")
        usage = response.get("usage", {})

        try:
            parsed = _parse_json_response(raw)
            return parsed, raw, usage, prompt
        except Exception as exc:
            fallback = {
                "survival_prediction": {
                    "outcome": "unknown",
                    "confidence": 0.0,
                    "rationale": f"Parsing error: {exc}",
                }
            }
            return fallback, raw, usage, prompt


class MedAgent:
    """MedAgent orchestrator for static + dynamic memory and final prediction."""

    def __init__(
        self,
        provider: str,
        model: str = None,
        api_key: str = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        enable_logging: bool = False,
        observation_hours: float = 12.0,
        use_llm_static_compression: bool = True,
        baseline_lab_lookback_start_hours: float = 72.0,
        baseline_lab_lookback_end_hours: float = 24.0,
        max_active_problems: int = 8,
        max_critical_events: int = 20,
        max_patterns: int = 8,
        memory_use_thinking: bool = True,
        predictor_use_thinking: bool = True,
    ):
        self.llm_client = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.observation_hours = observation_hours
        self.use_llm_static_compression = use_llm_static_compression
        self.baseline_lab_lookback_start_hours = baseline_lab_lookback_start_hours
        self.baseline_lab_lookback_end_hours = baseline_lab_lookback_end_hours
        self.memory_use_thinking = memory_use_thinking
        self.predictor_use_thinking = predictor_use_thinking

        self.static_builder = StaticMemoryBuilder(
            llm_client=self.llm_client,
            use_llm_static_compression=use_llm_static_compression,
        )
        self.dynamic_memory_agent = DynamicMemoryAgent(
            llm_client=self.llm_client,
            use_thinking=memory_use_thinking,
            max_active_problems=max_active_problems,
            max_critical_events=max_critical_events,
            max_patterns=max_patterns,
        )
        self.predictor = PredictorAgent(self.llm_client)

        self.enable_logging = enable_logging
        self.call_logs: List[LLMCallLog] = []

        # Runtime state
        self._current_patient_id = ""
        self._static_memory = StaticMemory()
        self._final_dynamic_memory = DynamicMemory()
        self._dynamic_history: List[DynamicMemorySnapshot] = []
        self._last_window_events: List[Dict[str, Any]] = []

        # Statistics
        self.total_patients = 0
        self.total_tokens_used = 0
        self.total_memory_calls = 0
        self.total_predictor_calls = 0
        self.total_static_compression_calls = 0
        self.total_dynamic_fallbacks = 0

    def _log_call(
        self,
        step_type: str,
        window_index: int,
        hours: float,
        prompt: str,
        response: str,
        usage: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        parsed_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        usage_dict = usage if isinstance(usage, dict) else {}
        input_tokens = _normalize_token_count(usage_dict.get("input_tokens", 0))
        output_tokens = _normalize_token_count(usage_dict.get("output_tokens", 0))
        self.total_tokens_used += input_tokens + output_tokens

        if not self.enable_logging:
            return

        log_metadata = dict(metadata or {})
        log_metadata.update(
            {
                "step_type": step_type,
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model,
            }
        )

        self.call_logs.append(
            LLMCallLog(
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
        )

    def run_patient_trajectory(
        self,
        windows: List[Dict[str, Any]],
        patient_metadata: Dict[str, Any],
        trajectory: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], MedAgentOutput]:
        """Run static memory build + dynamic updates + final prediction."""
        if trajectory is None:
            raise ValueError("trajectory is required for MedAgent static memory extraction")

        self.total_patients += 1
        self._current_patient_id = (
            f"{patient_metadata.get('subject_id', 'unknown')}_{patient_metadata.get('icu_stay_id', 'unknown')}"
        )

        static_memory, static_call = self.static_builder.build(
            trajectory=trajectory,
            baseline_lab_lookback_start_hours=self.baseline_lab_lookback_start_hours,
            baseline_lab_lookback_end_hours=self.baseline_lab_lookback_end_hours,
        )
        self._static_memory = static_memory

        if static_call is not None:
            self.total_static_compression_calls += 1
            self._log_call(
                "static_memory_builder",
                window_index=-1,
                hours=0.0,
                prompt=static_call.get("prompt", ""),
                response=static_call.get("response", ""),
                usage=static_call.get("usage", {}),
                metadata={"parse_error": static_call.get("parse_error")},
                parsed_response=static_call.get("parsed_response"),
            )

        dynamic_memory = DynamicMemory()
        self._dynamic_history = []
        self._last_window_events = []

        if verbose:
            print(f"Processing patient with {len(windows)} windows...")

        for index, window in enumerate(windows):
            current_events = window.get("current_events", [])
            if not current_events:
                continue

            hours = float(window.get("hours_since_admission", 0.0))

            if verbose:
                print(f"  Window {index+1}/{len(windows)} (Hour {hours:.1f})...", end=" ")

            previous_critical_count = len(dynamic_memory.critical_events_log)
            (
                updated_memory,
                memory_raw,
                memory_usage,
                memory_prompt,
                update_error,
                parsed_response,
            ) = self.dynamic_memory_agent.update_memory(
                previous_memory=dynamic_memory,
                static_memory=static_memory,
                current_events=current_events,
                window_index=index,
                hours_since_admission=hours,
            )

            self.total_memory_calls += 1
            if update_error:
                self.total_dynamic_fallbacks += 1
            new_critical_added = max(0, len(updated_memory.critical_events_log) - previous_critical_count)

            self._log_call(
                "memory_agent",
                window_index=index,
                hours=hours,
                prompt=memory_prompt,
                response=memory_raw,
                usage=memory_usage,
                metadata={
                    "num_events": len(current_events),
                    "fallback": bool(update_error),
                    "update_error": update_error,
                    "new_critical_events_added": new_critical_added,
                    "total_critical_events": len(updated_memory.critical_events_log),
                },
                parsed_response=parsed_response,
            )

            dynamic_memory = updated_memory
            self._dynamic_history.append(
                DynamicMemorySnapshot(
                    window_index=index,
                    hours_since_admission=hours,
                    num_current_events=len(current_events),
                    dynamic_memory=deepcopy(dynamic_memory),
                    update_error=update_error,
                )
            )
            self._last_window_events = deepcopy(current_events)

            if verbose:
                suffix = "fallback" if update_error else "ok"
                print(f"Dynamic memory update: {suffix}")

        self._final_dynamic_memory = dynamic_memory

        last_hours = float(windows[-1].get("hours_since_admission", 0.0)) if windows else 0.0
        prediction, pred_raw, pred_usage, pred_prompt = self.predictor.predict(
            static_memory=static_memory,
            dynamic_memory=dynamic_memory,
            last_window_events=self._last_window_events,
            observation_hours=self.observation_hours,
            use_thinking=self.predictor_use_thinking,
        )

        self.total_predictor_calls += 1
        self._log_call(
            "predictor",
            window_index=-1,
            hours=last_hours,
            prompt=pred_prompt,
            response=pred_raw,
            usage=pred_usage,
            metadata={"num_history_snapshots": len(self._dynamic_history)},
            parsed_response=prediction,
        )

        final_state_text = "\n\n".join([static_memory.to_text(), dynamic_memory.to_text()])

        output = MedAgentOutput(
            static_memory=static_memory,
            final_dynamic_memory=dynamic_memory,
            dynamic_memory_history=self._dynamic_history,
            last_window_events=self._last_window_events,
            final_state_text=final_state_text,
        )

        return prediction, output

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_patients": self.total_patients,
            "total_tokens_used": self.total_tokens_used,
            "total_memory_calls": self.total_memory_calls,
            "total_predictor_calls": self.total_predictor_calls,
            "total_static_compression_calls": self.total_static_compression_calls,
            "total_dynamic_fallbacks": self.total_dynamic_fallbacks,
            "total_llm_calls": len(self.call_logs) if self.enable_logging else 0,
        }

    def get_logs(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.call_logs]

    def get_static_memory(self) -> Dict[str, Any]:
        return self._static_memory.to_dict()

    def get_dynamic_memory(self) -> Dict[str, Any]:
        return self._final_dynamic_memory.to_dict()

    def get_dynamic_memory_history(self) -> List[Dict[str, Any]]:
        return [snapshot.to_dict() for snapshot in self._dynamic_history]

    def clear_logs(self) -> None:
        self.call_logs = []

    def save_logs(self, path: str) -> None:
        payload = {
            "total_calls": len(self.call_logs),
            "total_tokens": self.total_tokens_used,
            "calls": self.get_logs(),
        }
        with open(path, "w") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
