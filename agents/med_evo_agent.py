"""MedEvo Agent: event-grounded multi-agent pipeline with dynamic memory."""

from __future__ import annotations

import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from model.llms import LLMClient
from prompts.med_evo_prompts import get_episode_agent_prompt, get_event_agent_prompt, get_insight_agent_prompt
from prompts.oracle_prompt import format_pre_icu_compression_prompt
from prompts.predictor_prompts import get_survival_prediction_prompt
from utils.event_format import format_event_line as format_shared_event_line
from utils.event_format import format_event_lines as format_shared_event_lines
from utils.json_parse import parse_json_dict_or_raise

PRE_ICU_COMPRESSION_STEP_TYPE = "med_evo_pre_icu_history_compressor"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_token_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return 0
    return 0


def _parse_json_response(response: Any) -> Dict[str, Any]:
    """Parse JSON payloads from raw LLM output."""
    return parse_json_dict_or_raise(response)


def _normalize_int_list(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    output: List[int] = []
    seen = set()
    for item in value:
        parsed: Optional[int] = None
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            parsed = item
        elif isinstance(item, float):
            if item.is_integer():
                parsed = int(item)
        elif isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            try:
                parsed = int(text)
            except ValueError:
                try:
                    parsed_float = float(text)
                    if parsed_float.is_integer():
                        parsed = int(parsed_float)
                except ValueError:
                    matches = re.findall(r"-?\d+", text)
                    if len(matches) == 1:
                        parsed = int(matches[0])
        if parsed is None:
            continue
        if parsed in seen:
            continue
        seen.add(parsed)
        output.append(parsed)
    return output


def _format_event_lines(events: List[Dict[str, Any]], *, empty_text: str = "(No events)") -> List[str]:
    return format_shared_event_lines(events, empty_text=empty_text)


def _format_single_event_line(event: Dict[str, Any]) -> str:
    if not isinstance(event, dict):
        return ""
    return _safe_text(format_shared_event_line(event))


def _format_trajectory_observation_line(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""

    item_type = _safe_text(item.get("type"))
    if item_type == "episode":
        try:
            start_window = int(item.get("start_window"))
            end_window = int(item.get("end_window"))
        except (TypeError, ValueError):
            return ""

        start_hour = _to_optional_float(item.get("start_hour"))
        end_hour = _to_optional_float(item.get("end_hour"))
        text = _safe_text(item.get("episode_summary") or item.get("text"))
        if not text:
            text = f"Episode {start_window}-{end_window} compressed trajectory summary."
        if start_hour is None or end_hour is None:
            return f"episode {start_window}-{end_window}: {text}"
        return f"episode {start_window}-{end_window} (hour {start_hour:.1f}-{end_hour:.1f}): {text}"

    if item_type == "window_summary":
        try:
            window_id = int(item.get("window_id"))
        except (TypeError, ValueError):
            return ""

        start_hour = _to_optional_float(item.get("start_hour"))
        end_hour = _to_optional_float(item.get("end_hour"))
        text = _safe_text(item.get("text"))
        if not text:
            text = f"Window {window_id} processed with event-grounded update."
        if start_hour is None or end_hour is None:
            return f"window {window_id}: {text}"
        return f"window {window_id} (hour {start_hour:.1f}-{end_hour:.1f}): {text}"

    return ""


def _extract_event_reference_id(event: Dict[str, Any]) -> Optional[int]:
    """Resolve event references using ICU-local stable IDs across windows."""
    if not isinstance(event, dict):
        return None

    for key in ("icu_event_index", "event_index", "event_id", "event_idx", "prompt_event_id", "relative_event_id"):
        raw_value = event.get(key)
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            continue
    return None


def _derive_raw_event_name(event: Dict[str, Any]) -> str:
    if not isinstance(event, dict):
        return "Unknown event"
    name = _safe_text(event.get("event_name"))
    if name:
        return name
    name = _safe_text(event.get("code_specifics"))
    if name:
        return name
    name = _safe_text(event.get("code"))
    if name:
        return name
    return "Unknown event"


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fill_prompt_placeholder(template: str, key: str, value: str) -> str:
    """Replace both {key} and {{key}} styles used across prompt templates."""
    return template.replace(f"{{{{{key}}}}}", value).replace(f"{{{key}}}", value)


_PROMPT_PLACEHOLDER_PATTERN = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}|\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _extract_prompt_placeholders(template: str) -> List[str]:
    placeholders = set()
    for match in _PROMPT_PLACEHOLDER_PATTERN.finditer(template):
        key = match.group(1) or match.group(2)
        if key:
            placeholders.add(key)
    return sorted(placeholders)


def _render_prompt(template: str, values: Dict[str, str]) -> str:
    required_keys = _extract_prompt_placeholders(template)
    missing_keys = sorted(key for key in required_keys if key not in values)
    if missing_keys:
        raise ValueError(f"Missing prompt placeholder values: {', '.join(missing_keys)}")

    prompt = template
    for key, value in values.items():
        prompt = _fill_prompt_placeholder(prompt, key, value)

    unresolved_keys = sorted(
        key for key in required_keys if f"{{{key}}}" in prompt or f"{{{{{key}}}}}" in prompt
    )
    if unresolved_keys:
        raise ValueError(f"Unresolved prompt placeholders: {', '.join(unresolved_keys)}")
    return prompt


def _format_patient_metadata_lines(
    patient_metadata: Optional[Dict[str, Any]],
    *,
    include_ids: bool = False,
) -> List[str]:
    if not isinstance(patient_metadata, dict) or not patient_metadata:
        return ["- None"]

    lines: List[str] = []
    for key, value in patient_metadata.items():
        if not include_ids and key in {"subject_id", "icu_stay_id"}:
            continue

        value_text = _safe_text(value)
        if not value_text:
            continue

        if "\n" not in value_text:
            lines.append(f"- {key}: {value_text}")
            continue

        lines.append(f"- {key}:")
        for segment in value_text.splitlines():
            segment_text = _safe_text(segment)
            if segment_text:
                lines.append(f"  {segment_text}")

    return lines or ["- None"]


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


@dataclass
class WorkingWindow:
    """Structured working window with raw grounded events."""

    window_id: int
    start_hour: float
    end_hour: float
    events: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "events": self.events,
        }


@dataclass
class CriticalEvent:
    """Critical event identified by EventAgent."""

    event_id: int
    name_str: str
    evidence: str

    def to_dict(self) -> Dict[str, Any]:
        payload = {"id": self.event_id, "name_str": self.name_str}
        if self.evidence:
            payload["evidence"] = self.evidence
        return payload


@dataclass
class WindowSummary:
    """Event-grounded summary for one window."""

    window_id: int
    start_hour: float
    end_hour: float
    text: str
    supporting_event_ids: List[int]
    supporting_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "text": self.text,
            "supporting_event_ids": self.supporting_event_ids,
            "supporting_events": self.supporting_events,
        }


@dataclass
class Episode:
    """Compressed trajectory block spanning multiple windows."""

    episode_id: int
    start_window: int
    end_window: int
    start_hour: float
    end_hour: float
    episode_summary: str
    supporting_event_ids: List[int]
    supporting_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "start_window": self.start_window,
            "end_window": self.end_window,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "episode_summary": self.episode_summary,
            "supporting_event_ids": self.supporting_event_ids,
            "supporting_events": self.supporting_events,
        }


@dataclass
class EvidenceEvent:
    """Grounded evidence reference stored on insights."""

    event_id: int
    name_str: str

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.event_id, "name_str": self.name_str}


@dataclass
class Insight:
    """Structured hypothesis with supporting and counter evidence."""

    insight_id: int
    hypothesis: str
    supporting_event_ids: List[int]
    counter_event_ids: List[int]
    created_at: float
    updated_at: float
    supporting_evidence: List[EvidenceEvent] = field(default_factory=list)
    counter_evidence: List[EvidenceEvent] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "hypothesis": self.hypothesis,
            "supporting_event_ids": self.supporting_event_ids,
            "counter_event_ids": self.counter_event_ids,
            "supporting_evidence": [item.to_dict() for item in self.supporting_evidence],
            "counter_evidence": [item.to_dict() for item in self.counter_evidence],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "score": self.score,
        }


@dataclass
class MedEvoMemory:
    """Intra-patient memory state for MedEvo."""

    patient_metadata: Dict[str, Any] = field(default_factory=dict)
    working_memory: List[WorkingWindow] = field(default_factory=list)
    critical_events: List[CriticalEvent] = field(default_factory=list)
    trajectory_memory: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[Insight] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_metadata": self.patient_metadata,
            "working_memory": [item.to_dict() for item in self.working_memory],
            "critical_events": [item.to_dict() for item in self.critical_events],
            "trajectory_memory": self.trajectory_memory,
            "insights": [item.to_dict() for item in self.insights],
        }

    def to_text(self) -> str:
        parts: List[str] = ["## Patient Metadata"]
        for key, value in self.patient_metadata.items():
            if key in {"subject_id", "icu_stay_id"}:
                continue
            parts.append(f"{key}: {value}")

        # parts.append("")
        # parts.append("## Events from Previous Windows")
        # if self.working_memory:
        #     for window in self.working_memory[:-1]:
        #         parts.append(f"Window {window.window_id} (Hour {window.start_hour:.1f}-{window.end_hour:.1f})")
        #         parts.extend(_format_event_lines(window.events, empty_text="- (No events)"))
        # else:
        #     parts.append("(No windows)")

        parts.append("")
        parts.append("## Trajectory of the ICU Stay")
        if self.trajectory_memory:
            for item in self.trajectory_memory:
                if item.get("type") == "window_summary":
                    parts.append(
                        f"- Window {item.get('window_id')} (hour {item.get('start_hour', 0.0):.1f}-{item.get('end_hour', 0.0):.1f}): {item.get('text')} "
                    )
                elif item.get("type") == "episode":
                    parts.append(
                        f"- Window {item.get('start_window')}-{item.get('end_window')} "
                        f"(hour {item.get('start_hour', 0.0):.1f}-{item.get('end_hour', 0.0):.1f}): "
                        f"{item.get('episode_summary', item.get('text', ''))}"
                    )
                else:
                    parts.append(f"- {item}")
        else:
            parts.append("- None")

        parts.append("")
        parts.append("## Critical Events of the ICU Stay")
        if self.critical_events:
            for critical in self.critical_events:
                parts.append(f"{critical.name_str}")
        else:
            parts.append("- None")

        parts.append("")
        parts.append("## Patient Specific Insights")
        if self.insights:
            for insight in self.insights:
                parts.append(f"- I{insight.insight_id} score={insight.score:.3f}: {insight.hypothesis} ")
        else:
            parts.append("- None")

        parts.append("")
        parts.append("## Current Window Observation")
        if self.working_memory:
            current_window = self.working_memory[-1]
            parts.append(
                f"Window {current_window.window_id} (Hour {current_window.start_hour:.1f}-{current_window.end_hour:.1f})"
            )
            parts.extend(_format_event_lines(current_window.events, empty_text="- (No events)"))
        else:
            parts.append("- None")

        return "\n".join(parts)


@dataclass
class MedEvoMemoryDatabase:
    """Append-only list of per-window MedEvoMemory snapshots."""

    memory_snapshots: List[Dict[str, Any]] = field(default_factory=list)

    def add_snapshot(self, memory: MedEvoMemory) -> None:
        self.memory_snapshots.append(deepcopy(memory.to_dict()))

    def to_dict(self) -> Dict[str, Any]:
        return {"memory_snapshots": self.memory_snapshots}

    def save(self, path: str) -> None:
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=2, ensure_ascii=False)


class PerceptionAgent:
    """Deterministic window-buffer maintenance."""

    def __init__(self, max_working_windows: int = 3, window_duration_hours: float = 0.5):
        self.max_working_windows = max_working_windows
        self.window_duration_hours = window_duration_hours

    def update_working_memory(
        self,
        existing_windows: List[WorkingWindow],
        window_data: Dict[str, Any],
        window_index: int,
    ) -> List[WorkingWindow]:
        hours = float(window_data.get("hours_since_admission", 0.0))
        new_window = WorkingWindow(
            window_id=window_index,
            start_hour=hours,
            end_hour=hours + self.window_duration_hours,
            events=deepcopy(window_data.get("current_events", [])),
        )

        updated = list(existing_windows)
        updated.append(new_window)
        if len(updated) > self.max_working_windows:
            updated = updated[-self.max_working_windows :]
        return updated


class EventAgent:
    """LLM wrapper for critical event extraction and grounded summaries."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def analyze(
        self,
        working_windows: List[WorkingWindow],
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], str, Optional[str]]:
        prompt_template = get_event_agent_prompt()

        history_windows = working_windows[:-1]
        current_window = working_windows[-1] if working_windows else None

        window_lines: List[str] = ["## Patient metadata"]
        window_lines.extend(_format_patient_metadata_lines(patient_metadata))
        window_lines.append("")
        window_lines.append("## History windows")
        if history_windows:
            for window in history_windows:
                window_lines.append(
                    f"### Window {window.window_id} (Hour {window.start_hour:.1f}-{window.end_hour:.1f})"
                )
                window_lines.extend(_format_event_lines(window.events, empty_text="- (No events)"))
                window_lines.append("")
        else:
            window_lines.append("- None")
            window_lines.append("")

        window_lines.append("## Current window observation")
        if current_window is not None:
            window_lines.append(
                f"### Window {current_window.window_id} "
                f"(Hour {current_window.start_hour:.1f}-{current_window.end_hour:.1f})"
            )
            window_lines.extend(_format_event_lines(current_window.events, empty_text="- (No events)"))
        else:
            window_lines.append("- None")

        prompt = _render_prompt(
            prompt_template,
            {"working_windows_text": "\n".join(window_lines).strip()},
        )

        response = self.llm_client.chat(prompt=prompt, response_format="text")
        raw = response.get("content", "")
        usage = response.get("usage", {})

        parse_error: Optional[str] = None
        parsed: Dict[str, Any] = {}
        for candidate in (response.get("parsed"), raw):
            if candidate is None:
                continue
            try:
                parsed = _parse_json_response(candidate)
                parse_error = None
                break
            except Exception as exc:
                parse_error = str(exc)

        return parsed, raw, usage, prompt, parse_error


class InsightAgent:
    """LLM wrapper for structured insight updates."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def analyze(
        self,
        current_insights: List[Insight],
        trajectory_memory: List[Dict[str, Any]],
        recent_critical_events: List[CriticalEvent],
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], str, Optional[str]]:
        prompt_template = get_insight_agent_prompt()

        if current_insights:
            insights_text = "\n".join(
                [(f"[{insight.insight_id}] {_safe_text(insight.hypothesis)}") for insight in current_insights]
            )
        else:
            insights_text = "- None"

        if trajectory_memory:
            summary_lines = ["### Window Summary:"]
            for item in trajectory_memory:
                line = _format_trajectory_observation_line(item)
                if line:
                    summary_lines.append(line)
            if len(summary_lines) == 1:
                summary_lines.append("- None")
            summary_text = "\n".join(summary_lines)
        else:
            summary_text = "### Window Summary:\n- None"

        if recent_critical_events:
            critical_lines = ["### Critical Events:"]
            critical_lines.extend(
                [critical.name_str for critical in recent_critical_events if _safe_text(critical.name_str)]
            )
            critical_text = "\n".join(critical_lines)
        else:
            critical_text = "Critical Events:\n- None"

        patient_metadata_text = "\n".join(_format_patient_metadata_lines(patient_metadata))

        prompt = _render_prompt(
            prompt_template,
            {
                "hypothesis_bank": insights_text,
                "patient_metadata": patient_metadata_text,
                "window_summary": summary_text,
                "critical_events": critical_text,
            },
        )

        response = self.llm_client.chat(prompt=prompt, response_format="text")
        raw = response.get("content", "")
        usage = response.get("usage", {})

        parse_error: Optional[str] = None
        parsed: Dict[str, Any] = {}
        for candidate in (response.get("parsed"), raw):
            if candidate is None:
                continue
            try:
                parsed = _parse_json_response(candidate)
                parse_error = None
                break
            except Exception as exc:
                parse_error = str(exc)

        return parsed, raw, usage, prompt, parse_error


class EpisodeAgent:
    """LLM wrapper for compressing a block of window summaries into one episode."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def analyze(
        self,
        recent_window_summaries: List[WindowSummary],
        recent_critical_events: List[CriticalEvent],
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], str, Optional[str]]:
        prompt_template = get_episode_agent_prompt()

        if recent_window_summaries:
            summary_lines: List[str] = []
            for summary in recent_window_summaries:
                text = _safe_text(summary.text)
                if not text:
                    text = f"Window {summary.window_id} processed with event-grounded update."
                summary_lines.append(
                    f"- Window {summary.window_id} (hour {summary.start_hour:.1f}-{summary.end_hour:.1f}): {text}"
                )
            summaries_text = "\n".join(summary_lines)
        else:
            summaries_text = "- None"

        if recent_critical_events:
            critical_lines: List[str] = []
            for critical in recent_critical_events:
                event_name = _safe_text(critical.name_str)
                if not event_name:
                    continue
                event_id_prefix = f"[{critical.event_id}]"
                normalized_event_name = event_name.lstrip()
                if normalized_event_name.startswith(event_id_prefix):
                    event_label = normalized_event_name
                else:
                    event_label = f"{event_id_prefix} {event_name}"
                evidence = _safe_text(critical.evidence)
                if evidence:
                    critical_lines.append(f"- {event_label} | {evidence}")
                else:
                    critical_lines.append(f"- {event_label}")
            critical_text = "\n".join(critical_lines) if critical_lines else "- None"
        else:
            critical_text = "- None"

        patient_metadata_text = "\n".join(_format_patient_metadata_lines(patient_metadata))
        episode_window_count = len(recent_window_summaries)
        if recent_window_summaries:
            first_summary = recent_window_summaries[0]
            last_summary = recent_window_summaries[-1]
            episode_start_time = f"hour {first_summary.start_hour:.1f}"
            episode_end_time = f"hour {last_summary.end_hour:.1f}"
            episode_duration = max(last_summary.end_hour - first_summary.start_hour, 0.0)
        else:
            episode_start_time = "N/A"
            episode_end_time = "N/A"
            episode_duration = 0.0

        prompt = _render_prompt(
            prompt_template,
            {
                "k": str(episode_window_count),
                "duration": f"{episode_duration:.1f}",
                "episode_start_time": episode_start_time,
                "episode_end_time": episode_end_time,
                "patient_metadata": patient_metadata_text,
                "window_summaries": summaries_text,
                "critical_events": critical_text,
            },
        )

        response = self.llm_client.chat(prompt=prompt, response_format="text")
        raw = response.get("content", "")
        usage = response.get("usage", {})

        parse_error: Optional[str] = None
        parsed: Dict[str, Any] = {}
        for candidate in (response.get("parsed"), raw):
            if candidate is None:
                continue
            try:
                parsed = _parse_json_response(candidate)
                parse_error = None
                break
            except Exception as exc:
                parse_error = str(exc)

        return parsed, raw, usage, prompt, parse_error


class PredictorAgent:
    """Final ICU outcome predictor."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def predict(
        self,
        memory: MedEvoMemory,
        last_window_events: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], str, Optional[str]]:
        prompt_template = get_survival_prediction_prompt()

        # last_events_text = "\n".join(_format_event_lines(last_window_events, empty_text="- (No events)"))
        # context = "\n\n".join([memory.to_text(), "## Current Window Events", last_events_text])

        context = "\n\n".join([memory.to_text()])
        prompt = _render_prompt(prompt_template, {"context": context})

        response = self.llm_client.chat(prompt=prompt, response_format="text")
        raw = response.get("content", "")
        usage = response.get("usage", {})

        parse_error: Optional[str] = None
        parsed: Dict[str, Any] = {}
        for candidate in (response.get("parsed"), raw):
            if candidate is None:
                continue
            try:
                parsed = _parse_json_response(candidate)
                parse_error = None
                break
            except Exception as exc:
                parse_error = str(exc)

        if parse_error is not None:
            parsed = {
                "prediction": "unknown",
                "confidence": "Low",
                "supporting_evidence": [],
                "rationale": f"Parsing error: {parse_error}",
            }

        return parsed, raw, usage, prompt, parse_error


class MedEvoAgent:
    """Event-grounded multi-agent system with dynamic memory."""

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        enable_logging: bool = False,
        window_duration_hours: float = 0.5,
        max_working_windows: int = 3,
        max_critical_events: int = 100,
        max_window_summaries: int = 100,
        max_insights: int = 5,
        insight_every_n_windows: int = 1,
        episode_every_n_windows: int = 0,
    ):
        self.llm_client = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.window_duration_hours = window_duration_hours

        self.max_working_windows = max_working_windows
        self.max_critical_events = max_critical_events
        self.max_window_summaries = max_window_summaries
        self.max_insights = max_insights
        try:
            self.insight_every_n_windows = max(1, int(insight_every_n_windows))
        except (TypeError, ValueError):
            self.insight_every_n_windows = 1
        try:
            self.episode_every_n_windows = max(0, int(episode_every_n_windows))
        except (TypeError, ValueError):
            self.episode_every_n_windows = 0

        self.enable_logging = enable_logging
        self.call_logs: List[LLMCallLog] = []

        self.perception_agent = PerceptionAgent(
            max_working_windows=max_working_windows,
            window_duration_hours=window_duration_hours,
        )
        self.event_agent = EventAgent(self.llm_client)
        self.insight_agent = InsightAgent(self.llm_client)
        self.episode_agent = EpisodeAgent(self.llm_client)
        self.predictor_agent = PredictorAgent(self.llm_client)

        self.total_patients = 0
        self.total_tokens_used = 0
        self.total_event_calls = 0
        self.total_insight_calls = 0
        self.total_episode_calls = 0
        self.total_predictor_calls = 0
        self.total_pre_icu_compression_calls = 0
        self.total_pre_icu_compression_tokens = 0
        self.total_grounding_rejections = 0
        self.total_event_name_mismatches = 0
        self.total_insights_pruned = 0

        self._current_patient_id = ""
        self._event_id_to_window: Dict[int, int] = {}
        self._event_id_to_time: Dict[int, str] = {}
        self._event_id_to_raw_event: Dict[int, Dict[str, Any]] = {}
        self._event_id_to_name_str: Dict[int, str] = {}
        self._next_insight_id = 1
        self._next_episode_id = 1

    def clear_logs(self) -> None:
        self.call_logs = []

    def get_logs(self) -> List[Dict[str, Any]]:
        return [log.to_dict() for log in self.call_logs]

    def save_logs(self, path: str) -> None:
        payload = {
            "total_calls": len(self.call_logs),
            "total_tokens": self.total_tokens_used,
            "calls": self.get_logs(),
        }
        with open(path, "w") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_patients": self.total_patients,
            "total_tokens_used": self.total_tokens_used,
            "total_event_calls": self.total_event_calls,
            "total_insight_calls": self.total_insight_calls,
            "total_episode_calls": self.total_episode_calls,
            "total_predictor_calls": self.total_predictor_calls,
            "total_pre_icu_compression_calls": self.total_pre_icu_compression_calls,
            "total_pre_icu_compression_tokens": self.total_pre_icu_compression_tokens,
            "total_grounding_rejections": self.total_grounding_rejections,
            "total_event_name_mismatches": self.total_event_name_mismatches,
            "total_insights_pruned": self.total_insights_pruned,
            "insight_every_n_windows": self.insight_every_n_windows,
            "episode_every_n_windows": self.episode_every_n_windows,
            "total_llm_calls": len(self.call_logs) if self.enable_logging else 0,
        }

    def _log_call(
        self,
        step_type: str,
        window_index: int,
        hours_since_admission: float,
        prompt: str,
        response: str,
        usage: Dict[str, Any],
        parsed_response: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
                hours_since_admission=hours_since_admission,
                prompt=prompt,
                response=response,
                parsed_response=parsed_response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata=log_metadata,
            )
        )

    @staticmethod
    def _first_window_pre_icu_history(
        windows: List[Dict[str, Any]],
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        if not windows:
            return None
        first_window = windows[0]
        if not isinstance(first_window, dict):
            return None
        pre_icu_history = first_window.get("pre_icu_history")
        if not isinstance(pre_icu_history, dict):
            return None
        return first_window, pre_icu_history

    def _compress_pre_icu_history_for_windows(self, windows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        first_payload = self._first_window_pre_icu_history(windows)
        if first_payload is None:
            return None

        first_window, pre_icu_history = first_payload
        source = _safe_text(pre_icu_history.get("source")).lower()
        raw_content = _safe_text(pre_icu_history.get("content"))
        if source in {"", "none", "disabled"} or not raw_content:
            return None

        if source == "llm_compressed":
            return {
                "summary": raw_content,
                "source": source,
                "items": int(_normalize_token_count(pre_icu_history.get("items"))),
                "history_hours": _to_optional_float(pre_icu_history.get("history_hours")),
                "historical_discharge_summary_items": int(
                    _normalize_token_count(pre_icu_history.get("historical_discharge_summary_items"))
                ),
                "compression": pre_icu_history.get("compression") if isinstance(pre_icu_history, dict) else None,
            }

        prompt = format_pre_icu_compression_prompt(pre_icu_history)
        response = self.llm_client.chat(prompt=prompt, response_format="text")
        response_raw = _safe_text(response.get("content"))
        response_usage = response.get("usage", {})

        parse_error: Optional[str] = None
        parsed: Dict[str, Any] = {}
        for candidate in (response.get("parsed"), response_raw):
            if candidate is None:
                continue
            try:
                parsed = _parse_json_response(candidate)
                parse_error = None
                break
            except Exception as exc:
                parse_error = str(exc)

        compressed_text = _safe_text(parsed.get("compressed_pre_icu_history"))
        if not compressed_text and parse_error is None:
            parse_error = "compressed_pre_icu_history missing in LLM response."

        input_tokens = _normalize_token_count(response_usage.get("input_tokens", 0))
        output_tokens = _normalize_token_count(response_usage.get("output_tokens", 0))
        self.total_pre_icu_compression_calls += 1
        self.total_pre_icu_compression_tokens += input_tokens + output_tokens

        self._log_call(
            step_type=PRE_ICU_COMPRESSION_STEP_TYPE,
            window_index=-1,
            hours_since_admission=float(first_window.get("hours_since_admission", 0.0)),
            prompt=prompt,
            response=response_raw,
            usage=response_usage if isinstance(response_usage, dict) else {},
            parsed_response=parsed if isinstance(parsed, dict) else None,
            metadata={
                "pre_icu_history_source": first_window.get("pre_icu_history_source"),
                "pre_icu_history_items": first_window.get("pre_icu_history_items"),
                "original_source": source or "unknown",
                "original_content_chars": len(raw_content),
                "compressed_chars": len(compressed_text),
                "parse_error": parse_error,
            },
        )

        if not compressed_text:
            return None

        compression_metadata = {
            "method": "llm",
            "step_type": PRE_ICU_COMPRESSION_STEP_TYPE,
            "original_source": source or "unknown",
            "original_content_chars": len(raw_content),
            "compressed_chars": len(compressed_text),
        }

        compressed_pre_icu_history = dict(pre_icu_history)
        compressed_pre_icu_history["source"] = "llm_compressed"
        compressed_pre_icu_history["content"] = compressed_text
        compressed_pre_icu_history["compression"] = compression_metadata

        for window in windows:
            if not isinstance(window, dict):
                continue
            if not isinstance(window.get("pre_icu_history"), dict):
                continue
            window["pre_icu_history"] = dict(compressed_pre_icu_history)
            window["pre_icu_history_source"] = "llm_compressed"

        return {
            "summary": compressed_text,
            "source": "llm_compressed",
            "items": int(_normalize_token_count(pre_icu_history.get("items"))),
            "history_hours": _to_optional_float(pre_icu_history.get("history_hours")),
            "historical_discharge_summary_items": int(
                _normalize_token_count(pre_icu_history.get("historical_discharge_summary_items"))
            ),
            "compression": compression_metadata,
        }

    def _build_patient_metadata_with_pre_icu_summary(
        self,
        *,
        windows: List[Dict[str, Any]],
        patient_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        enriched = deepcopy(patient_metadata)
        compressed_pre_icu = self._compress_pre_icu_history_for_windows(windows)
        if not isinstance(compressed_pre_icu, dict):
            return enriched

        summary = _safe_text(compressed_pre_icu.get("summary"))
        if not summary:
            return enriched

        enriched["pre_icu_history_summary"] = summary
        enriched["pre_icu_history_source"] = _safe_text(compressed_pre_icu.get("source")) or "llm_compressed"
        enriched["pre_icu_history_items"] = int(_normalize_token_count(compressed_pre_icu.get("items")))
        history_hours = _to_optional_float(compressed_pre_icu.get("history_hours"))
        if history_hours is not None:
            enriched["pre_icu_history_hours"] = history_hours
        enriched["historical_discharge_summary_items"] = int(
            _normalize_token_count(compressed_pre_icu.get("historical_discharge_summary_items"))
        )

        return enriched

    def _register_event_ids(self, current_events: List[Dict[str, Any]], window_index: int) -> None:
        for event in current_events:
            event_id = _extract_event_reference_id(event)
            if event_id is None:
                continue

            self._event_id_to_window[event_id] = window_index
            if isinstance(event, dict):
                raw_event = deepcopy(event)
                # Canonicalize event IDs to one ICU-local coordinate system.
                raw_event["event_id"] = event_id
                raw_event["event_index"] = event_id
                raw_event["icu_event_index"] = event_id
                existing = self._event_id_to_raw_event.get(event_id)
                if existing is None or len(raw_event) >= len(existing):
                    self._event_id_to_raw_event[event_id] = raw_event

                name_str = _format_single_event_line(raw_event)
                if name_str:
                    existing_name = self._event_id_to_name_str.get(event_id, "")
                    if not existing_name or len(name_str) >= len(existing_name):
                        self._event_id_to_name_str[event_id] = name_str

            event_time = _safe_text(event.get("time"))
            if event_time:
                self._event_id_to_time[event_id] = event_time
            elif event_id not in self._event_id_to_time:
                self._event_id_to_time[event_id] = ""

            if event_id not in self._event_id_to_name_str:
                self._event_id_to_name_str[event_id] = f"[{event_id}]"

    def _resolve_event_name_str(self, event_id: int) -> str:
        resolved = _safe_text(self._event_id_to_name_str.get(event_id))
        if resolved:
            return resolved

        raw_event = self._event_id_to_raw_event.get(event_id)
        if isinstance(raw_event, dict):
            resolved = _format_single_event_line(raw_event)
            if resolved:
                self._event_id_to_name_str[event_id] = resolved
                return resolved

        return f"[{event_id}]"

    def _grounded_event_payload(self, event_id: int, event: Dict[str, Any]) -> Dict[str, Any]:
        raw_event = deepcopy(event) if isinstance(event, dict) else {}
        # Keep grounded payload IDs consistent with ICU-local indexing.
        raw_event["event_id"] = event_id
        raw_event["event_index"] = event_id
        raw_event["icu_event_index"] = event_id

        timestamp = _safe_text(raw_event.get("time")) or self._event_id_to_time.get(event_id, "")
        return {
            "event_id": event_id,
            "timestamp": timestamp,
            "event_name": _derive_raw_event_name(raw_event),
            "code": _safe_text(raw_event.get("code")),
            "code_specifics": _safe_text(raw_event.get("code_specifics")),
            "numeric_value": _to_optional_float(raw_event.get("numeric_value")),
            "text_value": _safe_text(raw_event.get("text_value")),
            "raw_event": raw_event,
        }

    def _resolve_grounded_event(self, event_id: int) -> Optional[Dict[str, Any]]:
        raw_event = self._event_id_to_raw_event.get(event_id)
        if not isinstance(raw_event, dict):
            return None
        return self._grounded_event_payload(event_id, raw_event)

    def _resolve_supporting_events(self, event_ids: List[int]) -> List[Dict[str, Any]]:
        supporting_events: List[Dict[str, Any]] = []
        seen = set()
        for event_id in event_ids:
            if event_id in seen:
                continue
            seen.add(event_id)
            grounded = self._resolve_grounded_event(event_id)
            if grounded is None:
                self.total_grounding_rejections += 1
                continue
            supporting_events.append(grounded)
        return supporting_events

    def _validate_event_ids(self, event_ids: List[int]) -> List[int]:
        valid: List[int] = []
        seen = set()
        for event_id in event_ids:
            if event_id in seen:
                continue
            seen.add(event_id)
            if event_id not in self._event_id_to_window:
                self.total_grounding_rejections += 1
                continue
            if event_id not in self._event_id_to_raw_event:
                self.total_grounding_rejections += 1
                continue
            valid.append(event_id)
        return valid

    def _ground_critical_events(
        self,
        payload: Any,
    ) -> List[CriticalEvent]:
        if not isinstance(payload, list):
            payload = []

        grounded: List[CriticalEvent] = []
        seen = set()

        for item in payload:
            evidence = ""
            raw_event_id: Any = item
            if isinstance(item, dict):
                raw_event_id = item.get("event_id", item.get("id"))
                evidence = _safe_text(item.get("evidence") or item.get("significance"))

            try:
                event_id = int(raw_event_id)
            except (TypeError, ValueError):
                self.total_grounding_rejections += 1
                continue

            if event_id not in self._event_id_to_window:
                self.total_grounding_rejections += 1
                continue

            if not evidence:
                evidence = "Clinically significant trajectory change"

            key = event_id
            if key in seen:
                continue
            seen.add(key)

            grounded.append(
                CriticalEvent(
                    event_id=event_id,
                    name_str=self._resolve_event_name_str(event_id),
                    evidence=evidence,
                )
            )

        if grounded:
            return grounded

        # If model output is empty/invalid, return an empty list.
        return []

    def _ground_window_summary(
        self,
        payload: Any,
        window_index: int,
        start_hour: float,
        end_hour: float,
        current_events: List[Dict[str, Any]],
    ) -> WindowSummary:
        summary_data = payload if isinstance(payload, dict) else {}
        text = _safe_text(summary_data.get("text"))
        if not text:
            text = f"Window {window_index} processed with event-grounded update."

        supporting_event_ids = _normalize_int_list(summary_data.get("supporting_event_ids", []))
        supporting_event_ids = self._validate_event_ids(supporting_event_ids)

        if not supporting_event_ids:
            current_ids = [
                event_id
                for event_id in (_extract_event_reference_id(event) for event in current_events)
                if event_id is not None
            ]
            supporting_event_ids = self._validate_event_ids(current_ids[:3])

        supporting_events = self._resolve_supporting_events(supporting_event_ids)
        supporting_event_ids = [
            int(item.get("event_id")) for item in supporting_events if item.get("event_id") is not None
        ]

        return WindowSummary(
            window_id=window_index,
            start_hour=float(start_hour),
            end_hour=float(end_hour),
            text=text,
            supporting_event_ids=supporting_event_ids,
            supporting_events=supporting_events,
        )

    def _recompute_insight_scores(self, insights: List[Insight]) -> None:
        for insight in insights:
            support_count = len(insight.supporting_event_ids)
            counter_count = len(insight.counter_event_ids)
            insight.score = float(support_count - counter_count)

    def _build_evidence_events(self, event_ids: List[int]) -> List[EvidenceEvent]:
        evidence_events: List[EvidenceEvent] = []
        seen = set()
        for event_id in event_ids:
            if event_id in seen:
                continue
            seen.add(event_id)
            evidence_events.append(
                EvidenceEvent(
                    event_id=event_id,
                    name_str=self._resolve_event_name_str(event_id),
                )
            )
        return evidence_events

    def _refresh_insight_evidence_objects(self, insight: Insight) -> None:
        insight.supporting_evidence = self._build_evidence_events(insight.supporting_event_ids)
        insight.counter_evidence = self._build_evidence_events(insight.counter_event_ids)

    def _apply_insight_updates(
        self,
        existing_insights: List[Insight],
        parsed_payload: Dict[str, Any],
    ) -> List[Insight]:
        now_ts = time.time()
        insights_by_id = {insight.insight_id: deepcopy(insight) for insight in existing_insights}

        updates = parsed_payload.get("insight_updates", parsed_payload.get("updated_insights", []))
        if not isinstance(updates, list):
            updates = []

        new_insights = parsed_payload.get("new_insights", [])
        if not isinstance(new_insights, list):
            new_insights = []

        for item in updates + new_insights:
            if not isinstance(item, dict):
                continue

            hypothesis = _safe_text(item.get("hypothesis"))
            supporting = self._validate_event_ids(
                _normalize_int_list(item.get("supporting_event_ids", item.get("supporting_evidence", [])))
            )
            counter = self._validate_event_ids(
                _normalize_int_list(item.get("counter_event_ids", item.get("counter_evidence", [])))
            )

            if not hypothesis and not supporting and not counter:
                continue

            raw_insight_id = item.get("insight_id", item.get("hypothesis_id"))
            insight_id: Optional[int] = None
            if raw_insight_id is not None:
                try:
                    parsed_id = int(raw_insight_id)
                    if parsed_id in insights_by_id:
                        insight_id = parsed_id
                except (TypeError, ValueError):
                    insight_id = None

            if insight_id is not None:
                insight = insights_by_id[insight_id]
                if hypothesis:
                    insight.hypothesis = hypothesis
                insight.supporting_event_ids = self._validate_event_ids(
                    _normalize_int_list(insight.supporting_event_ids + supporting)
                )
                insight.counter_event_ids = self._validate_event_ids(
                    _normalize_int_list(insight.counter_event_ids + counter)
                )
                self._refresh_insight_evidence_objects(insight)
                insight.updated_at = now_ts
                insights_by_id[insight_id] = insight
                continue

            if not hypothesis:
                continue

            insight_id = self._next_insight_id
            self._next_insight_id += 1
            new_insight = Insight(
                insight_id=insight_id,
                hypothesis=hypothesis,
                supporting_event_ids=supporting,
                counter_event_ids=counter,
                created_at=now_ts,
                updated_at=now_ts,
                score=0.0,
            )
            self._refresh_insight_evidence_objects(new_insight)
            insights_by_id[insight_id] = new_insight

        updated_insights = list(insights_by_id.values())
        for insight in updated_insights:
            self._refresh_insight_evidence_objects(insight)
        self._recompute_insight_scores(updated_insights)

        if len(updated_insights) > self.max_insights:
            updated_insights.sort(key=lambda insight: (insight.score, insight.updated_at))
            prune_count = len(updated_insights) - self.max_insights
            self.total_insights_pruned += prune_count
            updated_insights = updated_insights[prune_count:]

        updated_insights.sort(key=lambda insight: insight.insight_id)
        return updated_insights

    def _build_default_insight_payload(
        self,
        recent_summary: WindowSummary,
    ) -> Dict[str, Any]:
        if not recent_summary.supporting_event_ids:
            return {"insight_updates": [], "new_insights": []}

        return {
            "insight_updates": [],
            "new_insights": [
                {
                    "hypothesis": f"Window {recent_summary.window_id} trend requires continued monitoring.",
                    "supporting_event_ids": recent_summary.supporting_event_ids[:2],
                    "counter_event_ids": [],
                }
            ],
        }

    def _build_default_episode_payload(
        self,
        recent_window_summaries: List[WindowSummary],
        recent_critical_events: List[CriticalEvent],
    ) -> Dict[str, Any]:
        text_segments: List[str] = []
        supporting_event_ids: List[int] = []

        for summary in recent_window_summaries:
            summary_text = _safe_text(summary.text)
            if summary_text:
                text_segments.append(
                    f"Window {summary.window_id} (hour {summary.start_hour:.1f}-{summary.end_hour:.1f}): {summary_text}"
                )
            supporting_event_ids.extend(summary.supporting_event_ids)

        if not text_segments:
            text_segments.append("No clinically meaningful progression identified in this episode block.")

        for critical in recent_critical_events:
            supporting_event_ids.append(int(critical.event_id))

        supporting_event_ids = _normalize_int_list(supporting_event_ids)[:8]

        return {
            "episode_summary": {
                "text": " ".join(text_segments),
                "supporting_event_ids": supporting_event_ids,
            }
        }

    def _ground_episode(
        self,
        payload: Dict[str, Any],
        start_window: int,
        end_window: int,
        start_hour: float,
        end_hour: float,
        recent_window_summaries: List[WindowSummary],
        recent_critical_events: List[CriticalEvent],
    ) -> Episode:
        source_payload = payload if isinstance(payload, dict) else {}
        summary_payload = source_payload.get("episode_summary", source_payload.get("episode", {}))
        if not isinstance(summary_payload, dict):
            summary_payload = {}

        episode_text = _safe_text(summary_payload.get("text"))
        if not episode_text:
            episode_text = _safe_text(summary_payload.get("episode_summary"))
        if not episode_text:
            episode_text = _safe_text(source_payload.get("text"))

        default_payload = self._build_default_episode_payload(recent_window_summaries, recent_critical_events)
        if not episode_text:
            episode_text = _safe_text(default_payload.get("episode_summary", {}).get("text"))

        supporting_event_ids = _normalize_int_list(
            summary_payload.get(
                "supporting_event_ids",
                summary_payload.get("supporting_evidence", source_payload.get("supporting_event_ids", [])),
            )
        )
        supporting_event_ids = self._validate_event_ids(supporting_event_ids)

        if not supporting_event_ids:
            supporting_event_ids = self._validate_event_ids(
                _normalize_int_list(default_payload.get("episode_summary", {}).get("supporting_event_ids", []))
            )

        if not supporting_event_ids:
            supporting_event_ids = self._validate_event_ids([int(item.event_id) for item in recent_critical_events])

        supporting_events = self._resolve_supporting_events(supporting_event_ids)
        supporting_event_ids = [
            int(item.get("event_id")) for item in supporting_events if item.get("event_id") is not None
        ]

        episode = Episode(
            episode_id=self._next_episode_id,
            start_window=int(start_window),
            end_window=int(end_window),
            start_hour=float(start_hour),
            end_hour=float(end_hour),
            episode_summary=episode_text,
            supporting_event_ids=supporting_event_ids,
            supporting_events=supporting_events,
        )
        self._next_episode_id += 1
        return episode

    def _replace_window_summaries_with_episode(
        self,
        memory: MedEvoMemory,
        covered_summaries: List[WindowSummary],
        episode: Episode,
    ) -> None:
        covered_window_ids = {int(item.window_id) for item in covered_summaries}

        filtered: List[Dict[str, Any]] = []
        for item in memory.trajectory_memory:
            if item.get("type") != "window_summary":
                filtered.append(item)
                continue

            try:
                window_id = int(item.get("window_id"))
            except (TypeError, ValueError):
                filtered.append(item)
                continue

            if window_id in covered_window_ids:
                continue
            filtered.append(item)

        filtered.append({"type": "episode", **episode.to_dict(), "text": episode.episode_summary})
        memory.trajectory_memory = self._truncate_trajectory_memory(filtered)

    @staticmethod
    def _trajectory_item_index(item: Dict[str, Any]) -> int:
        raw_idx = item.get("window_id", item.get("start_window"))
        try:
            return int(raw_idx)
        except (TypeError, ValueError):
            return 10**9

    def _truncate_trajectory_memory(self, trajectory_memory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ordered = sorted(trajectory_memory, key=self._trajectory_item_index)
        if len(ordered) > self.max_window_summaries:
            return ordered[-self.max_window_summaries :]
        return ordered

    def _build_insight_context(self, memory: MedEvoMemory) -> Tuple[List[Dict[str, Any]], List[CriticalEvent]]:
        trajectory_items = [deepcopy(item) for item in memory.trajectory_memory if isinstance(item, dict)]

        critical_by_id: Dict[int, CriticalEvent] = {}
        for critical in memory.critical_events:
            critical_by_id[int(critical.event_id)] = critical

        selected_critical_events: List[CriticalEvent] = []
        selected_ids = set()
        for item in trajectory_items:
            supporting_event_ids = _normalize_int_list(item.get("supporting_event_ids", []))
            for event_id in supporting_event_ids:
                if event_id in selected_ids:
                    continue
                critical = critical_by_id.get(event_id)
                if critical is None:
                    continue
                selected_ids.add(event_id)
                selected_critical_events.append(deepcopy(critical))

        if selected_critical_events:
            return trajectory_items, selected_critical_events

        deduped_critical_events: List[CriticalEvent] = []
        seen = set()
        for critical in memory.critical_events:
            event_id = int(critical.event_id)
            if event_id in seen:
                continue
            seen.add(event_id)
            deduped_critical_events.append(deepcopy(critical))
        return trajectory_items, deduped_critical_events

    def create_memory_snapshots(
        self,
        windows: List[Dict[str, Any]],
        patient_metadata: Dict[str, Any],
        verbose: bool = True,
    ) -> Tuple[MedEvoMemory, MedEvoMemoryDatabase]:
        self.total_patients += 1
        self._current_patient_id = (
            f"{patient_metadata.get('subject_id', 'unknown')}_{patient_metadata.get('icu_stay_id', 'unknown')}"
        )
        self._event_id_to_window = {}
        self._event_id_to_time = {}
        self._event_id_to_raw_event = {}
        self._event_id_to_name_str = {}
        self._next_insight_id = 1
        self._next_episode_id = 1

        patient_metadata_with_pre_icu = self._build_patient_metadata_with_pre_icu_summary(
            windows=windows,
            patient_metadata=patient_metadata,
        )
        memory = MedEvoMemory(patient_metadata=deepcopy(patient_metadata_with_pre_icu))
        memory_db = MedEvoMemoryDatabase()

        if verbose:
            print(f"Processing patient with {len(windows)} windows...")
            pre_icu_summary = _safe_text(memory.patient_metadata.get("pre_icu_history_summary"))
            if pre_icu_summary:
                print(f"  Pre-ICU summary prepared ({len(pre_icu_summary)} chars)")

        processed_window_count = 0
        pending_summaries_for_episode: List[WindowSummary] = []
        pending_critical_for_episode: List[CriticalEvent] = []
        pending_critical_ids_for_episode = set()
        for idx, window in enumerate(windows):
            current_events = window.get("current_events", [])
            if not current_events:
                continue

            hours = float(window.get("hours_since_admission", 0.0))

            if verbose:
                print(f"  Window {idx+1}/{len(windows)} (Hour {hours:.1f})...", end=" ")

            self._register_event_ids(current_events, idx)

            memory.working_memory = self.perception_agent.update_working_memory(
                memory.working_memory,
                window,
                idx,
            )

            event_parsed, event_raw, event_usage, event_prompt, event_parse_error = self.event_agent.analyze(
                memory.working_memory,
                patient_metadata=memory.patient_metadata,
            )
            self.total_event_calls += 1

            critical_events = self._ground_critical_events(
                payload=event_parsed.get("critical_event_ids", event_parsed.get("critical_events", [])),
            )

            summary = self._ground_window_summary(
                payload=event_parsed.get("window_summary", {}),
                window_index=idx,
                start_hour=hours,
                end_hour=hours + self.window_duration_hours,
                current_events=current_events,
            )

            self._log_call(
                step_type="event_agent",
                window_index=idx,
                hours_since_admission=hours,
                prompt=event_prompt,
                response=event_raw,
                usage=event_usage,
                parsed_response=event_parsed,
                metadata={
                    "event_parse_error": event_parse_error,
                    "critical_events_count": len(critical_events),
                    "critical_name_mismatch_count": 0,
                    "summary_support_count": len(summary.supporting_event_ids),
                },
            )

            memory.critical_events.extend(critical_events)
            if len(memory.critical_events) > self.max_critical_events:
                memory.critical_events = memory.critical_events[-self.max_critical_events :]

            memory.trajectory_memory.append(
                {
                    "type": "window_summary",
                    **summary.to_dict(),
                }
            )
            memory.trajectory_memory = self._truncate_trajectory_memory(memory.trajectory_memory)

            pending_summaries_for_episode.append(deepcopy(summary))
            for critical in critical_events:
                if critical.event_id in pending_critical_ids_for_episode:
                    continue
                pending_critical_ids_for_episode.add(critical.event_id)
                pending_critical_for_episode.append(deepcopy(critical))

            run_episode_agent = self.episode_every_n_windows > 0 and (
                len(pending_summaries_for_episode) >= self.episode_every_n_windows
            )
            if run_episode_agent:
                episode_start = pending_summaries_for_episode[0]
                episode_end = pending_summaries_for_episode[-1]

                episode_parsed, episode_raw, episode_usage, episode_prompt, episode_parse_error = (
                    self.episode_agent.analyze(
                        recent_window_summaries=pending_summaries_for_episode,
                        recent_critical_events=pending_critical_for_episode,
                        patient_metadata=memory.patient_metadata,
                    )
                )
                self.total_episode_calls += 1

                if episode_parse_error is not None:
                    episode_parsed = self._build_default_episode_payload(
                        pending_summaries_for_episode,
                        pending_critical_for_episode,
                    )

                grounded_episode = self._ground_episode(
                    payload=episode_parsed,
                    start_window=episode_start.window_id,
                    end_window=episode_end.window_id,
                    start_hour=episode_start.start_hour,
                    end_hour=episode_end.end_hour,
                    recent_window_summaries=pending_summaries_for_episode,
                    recent_critical_events=pending_critical_for_episode,
                )
                self._replace_window_summaries_with_episode(
                    memory=memory,
                    covered_summaries=pending_summaries_for_episode,
                    episode=grounded_episode,
                )

                self._log_call(
                    step_type="episode_agent",
                    window_index=idx,
                    hours_since_admission=hours,
                    prompt=episode_prompt,
                    response=episode_raw,
                    usage=episode_usage,
                    parsed_response=episode_parsed,
                    metadata={
                        "episode_parse_error": episode_parse_error,
                        "episode_id": grounded_episode.episode_id,
                        "episode_start_window": grounded_episode.start_window,
                        "episode_end_window": grounded_episode.end_window,
                        "episode_support_count": len(grounded_episode.supporting_event_ids),
                    },
                )

                pending_summaries_for_episode = []
                pending_critical_for_episode = []
                pending_critical_ids_for_episode = set()

            run_insight_agent = (processed_window_count % self.insight_every_n_windows) == 0
            processed_window_count += 1

            insight_parsed: Dict[str, Any] = {"insight_updates": [], "new_insights": []}
            insight_parse_error: Optional[str] = None
            if run_insight_agent:
                trajectory_context, critical_context = self._build_insight_context(memory)
                insight_parsed, insight_raw, insight_usage, insight_prompt, insight_parse_error = (
                    self.insight_agent.analyze(
                        current_insights=memory.insights,
                        trajectory_memory=trajectory_context,
                        recent_critical_events=critical_context,
                        patient_metadata=memory.patient_metadata,
                    )
                )
                self.total_insight_calls += 1

                if insight_parse_error is not None:
                    insight_parsed = self._build_default_insight_payload(summary)

                memory.insights = self._apply_insight_updates(
                    existing_insights=memory.insights,
                    parsed_payload=insight_parsed,
                )

                self._log_call(
                    step_type="insight_agent",
                    window_index=idx,
                    hours_since_admission=hours,
                    prompt=insight_prompt,
                    response=insight_raw,
                    usage=insight_usage,
                    parsed_response=insight_parsed,
                    metadata={
                        "insight_parse_error": insight_parse_error,
                        "num_insights": len(memory.insights),
                    },
                )
            else:
                self._recompute_insight_scores(memory.insights)

            memory_db.add_snapshot(memory)

            if verbose:
                maybe_skipped = " insight_skipped=True" if not run_insight_agent else ""
                episode_count = sum(1 for item in memory.trajectory_memory if item.get("type") == "episode")
                print(
                    f"insights={len(memory.insights)} critical_events={len(memory.critical_events)} "
                    f"episodes={episode_count}{maybe_skipped}"
                )

        return memory, memory_db

    def predict_from_memory(
        self,
        memory: MedEvoMemory,
        *,
        last_window_events: Optional[List[Dict[str, Any]]] = None,
        hours_since_admission: float = 0.0,
        window_index: int = -1,
    ) -> Dict[str, Any]:
        if last_window_events is None:
            last_window_events = []

        prediction, pred_raw, pred_usage, pred_prompt, pred_parse_error = self.predictor_agent.predict(
            memory=memory,
            last_window_events=last_window_events,
        )
        self.total_predictor_calls += 1

        self._log_call(
            step_type="med_evo_predictor",
            window_index=int(window_index),
            hours_since_admission=float(hours_since_admission),
            prompt=pred_prompt,
            response=pred_raw,
            usage=pred_usage,
            parsed_response=prediction,
            metadata={"predictor_parse_error": pred_parse_error},
        )

        return prediction

    def run_patient_trajectory(
        self,
        windows: List[Dict[str, Any]],
        patient_metadata: Dict[str, Any],
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], MedEvoMemory, MedEvoMemoryDatabase]:
        memory, memory_db = self.create_memory_snapshots(
            windows=windows,
            patient_metadata=patient_metadata,
            verbose=verbose,
        )

        last_window = windows[-1] if windows else {}
        last_window_events = last_window.get("current_events", []) if isinstance(last_window, dict) else []
        if not isinstance(last_window_events, list):
            last_window_events = []
        last_hours = float(last_window.get("hours_since_admission", 0.0)) if isinstance(last_window, dict) else 0.0

        prediction = self.predict_from_memory(
            memory=memory,
            last_window_events=last_window_events,
            hours_since_admission=last_hours,
            window_index=-1,
        )

        return prediction, memory, memory_db
