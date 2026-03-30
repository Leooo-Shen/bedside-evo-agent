"""MedEvo Agent: event-grounded multi-agent pipeline with dynamic memory."""

from __future__ import annotations

import ast
import json
import math
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from model.llms import LLMClient
from prompts.med_evo_prompts import get_event_agent_prompt, get_insight_agent_prompt, get_med_evo_predictor_prompt
from utils.event_format import format_event_line as format_shared_event_line
from utils.event_format import format_event_lines as format_shared_event_lines


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

    if isinstance(response, dict):
        return response

    if isinstance(response, str):
        raw = response.strip()
    elif response is None:
        raw = ""
    else:
        raw = str(response).strip()

    def _coerce_dict(parsed: Any) -> Optional[Dict[str, Any]]:
        if isinstance(parsed, dict):
            return parsed
        return None

    def _try_parse(candidate: str) -> Optional[Dict[str, Any]]:
        candidate = candidate.strip()
        if not candidate:
            return None

        try:
            return _coerce_dict(json.loads(candidate))
        except json.JSONDecodeError:
            pass

        try:
            return _coerce_dict(ast.literal_eval(candidate))
        except (SyntaxError, ValueError):
            pass

        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", candidate, re.IGNORECASE)
        for block in fenced:
            block = block.strip()
            if not block:
                continue
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(block)
                except (SyntaxError, ValueError):
                    continue
            maybe_dict = _coerce_dict(parsed)
            if maybe_dict is not None:
                return maybe_dict

        decoder = json.JSONDecoder()
        for i, ch in enumerate(candidate):
            if ch not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[i:])
            except json.JSONDecodeError:
                continue
            maybe_dict = _coerce_dict(parsed)
            if maybe_dict is not None:
                return maybe_dict

        return None

    candidates: List[str] = []
    if raw:
        candidates.append(raw)

    tagged = re.findall(r"<response>([\s\S]*?)</response>", raw, re.IGNORECASE)
    for tag_content in tagged:
        if tag_content.strip():
            candidates.append(tag_content.strip())

    for candidate in candidates:
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed

    raise ValueError(f"Could not parse JSON from response: {raw[:200]}...")


def _normalize_int_list(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    output: List[int] = []
    seen = set()
    for item in value:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
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


def _strip_hypothesis_evidence_suffix(hypothesis: str) -> str:
    text = _safe_text(hypothesis)
    if not text:
        return ""
    # Remove deterministic evidence-id suffix if present.
    return re.sub(r"\s*\[evidence_ids \+:\[[^\]]*\] -:\[[^\]]*\]\]\s*$", "", text).strip()


def _append_hypothesis_evidence_suffix(hypothesis: str, supporting_ids: List[int], counter_ids: List[int]) -> str:
    base = _strip_hypothesis_evidence_suffix(hypothesis)
    if not base:
        base = "Clinical hypothesis"
    if not supporting_ids and not counter_ids:
        return base
    support_text = ", ".join(str(event_id) for event_id in supporting_ids)
    counter_text = ", ".join(str(event_id) for event_id in counter_ids)
    return f"{base} [evidence_ids +:[{support_text}] -:[{counter_text}]]"


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
    """Compressed trajectory block (schema scaffold for future versions)."""

    episode_id: int
    start_window: int
    end_window: int
    episode_summary: str
    supporting_event_ids: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "start_window": self.start_window,
            "end_window": self.end_window,
            "episode_summary": self.episode_summary,
            "supporting_event_ids": self.supporting_event_ids,
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

        parts.append("")
        parts.append("## Recent Working Memory")
        if self.working_memory:
            for window in self.working_memory:
                parts.append(f"Window {window.window_id} (Hour {window.start_hour:.1f}-{window.end_hour:.1f})")
                parts.extend(_format_event_lines(window.events, empty_text="- (No events)"))
        else:
            parts.append("(No windows)")

        parts.append("")
        parts.append("## Critical Events")
        if self.critical_events:
            for critical in self.critical_events[-10:]:
                evidence_suffix = f" evidence={critical.evidence}" if critical.evidence else ""
                parts.append(f"{critical.event_id} CRITICAL_EVENT {critical.name_str}{evidence_suffix}")
        else:
            parts.append("- None")

        parts.append("")
        parts.append("## Trajectory Memory")
        if self.trajectory_memory:
            for item in self.trajectory_memory[-10:]:
                if item.get("type") == "window_summary":
                    parts.append(
                        f"- Window {item.get('window_id')}: {item.get('text')} "
                        f"[support={item.get('supporting_event_ids', [])}]"
                    )
                else:
                    parts.append(f"- {item}")
        else:
            parts.append("- None")

        parts.append("")
        parts.append("## Insights")
        if self.insights:
            for insight in self.insights:
                parts.append(
                    f"- I{insight.insight_id} score={insight.score:.3f}: {insight.hypothesis} "
                    f"support={insight.supporting_event_ids} counter={insight.counter_event_ids}"
                )
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
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], str, Optional[str]]:
        prompt_template = get_event_agent_prompt()

        history_windows = working_windows[:-1]
        current_window = working_windows[-1] if working_windows else None

        window_lines: List[str] = ["## History windows"]
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

        prompt = _fill_prompt_placeholder(prompt_template, "working_windows_text", "\n".join(window_lines).strip())

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
        recent_window_summaries: List[WindowSummary],
        recent_critical_events: List[CriticalEvent],
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], str, Optional[str]]:
        prompt_template = get_insight_agent_prompt()

        if current_insights:
            insights_text = "\n".join(
                [
                    (f"[{insight.insight_id}] {_strip_hypothesis_evidence_suffix(insight.hypothesis)}")
                    for insight in current_insights
                ]
            )
        else:
            insights_text = "- None"

        if recent_window_summaries:
            summary_lines = ["### Window Summary:"]
            for summary in recent_window_summaries:
                text = _safe_text(summary.text)
                if not text:
                    text = f"Window {summary.window_id} processed with event-grounded update."
                summary_lines.append(
                    f"window {summary.window_id} (hour {summary.start_hour:.1f}-{summary.end_hour:.1f}): {text}"
                )
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

        prompt = _fill_prompt_placeholder(prompt_template, "hypothesis_bank", insights_text)
        prompt = _fill_prompt_placeholder(prompt, "window_summary", summary_text)
        prompt = _fill_prompt_placeholder(prompt, "critical_events", critical_text)

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
        observation_hours: float,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], str, Optional[str]]:
        prompt_template = get_med_evo_predictor_prompt(observation_hours=observation_hours)

        last_events_text = "\n".join(_format_event_lines(last_window_events, empty_text="- (No events)"))

        context = "\n\n".join([memory.to_text(), "## Last Window Events", last_events_text])
        prompt = _fill_prompt_placeholder(prompt_template, "context", context)

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
                "survival_prediction": {
                    "outcome": "unknown",
                    "confidence": 0.0,
                    "rationale": f"Parsing error: {parse_error}",
                }
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
        observation_hours: float = 12.0,
        window_duration_hours: float = 0.5,
        max_working_windows: int = 3,
        max_events: int = 100,
        max_episodes: int = 20,
        max_insights: int = 5,
        insight_recency_tau: float = 4.0,
        insight_every_n_windows: int = 1,
    ):
        self.llm_client = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.observation_hours = observation_hours
        self.window_duration_hours = window_duration_hours

        self.max_working_windows = max_working_windows
        self.max_events = max_events
        self.max_episodes = max_episodes
        self.max_insights = max_insights
        self.insight_recency_tau = max(float(insight_recency_tau), 0.1)
        try:
            self.insight_every_n_windows = max(1, int(insight_every_n_windows))
        except (TypeError, ValueError):
            self.insight_every_n_windows = 1

        self.enable_logging = enable_logging
        self.call_logs: List[LLMCallLog] = []

        self.perception_agent = PerceptionAgent(
            max_working_windows=max_working_windows,
            window_duration_hours=window_duration_hours,
        )
        self.event_agent = EventAgent(self.llm_client)
        self.insight_agent = InsightAgent(self.llm_client)
        self.predictor_agent = PredictorAgent(self.llm_client)

        self.total_patients = 0
        self.total_tokens_used = 0
        self.total_event_calls = 0
        self.total_insight_calls = 0
        self.total_predictor_calls = 0
        self.total_grounding_rejections = 0
        self.total_event_name_mismatches = 0
        self.total_insights_pruned = 0

        self._current_patient_id = ""
        self._event_id_to_window: Dict[int, int] = {}
        self._event_id_to_time: Dict[int, str] = {}
        self._event_id_to_raw_event: Dict[int, Dict[str, Any]] = {}
        self._event_id_to_name_str: Dict[int, str] = {}
        self._next_insight_id = 1

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
            "total_predictor_calls": self.total_predictor_calls,
            "total_grounding_rejections": self.total_grounding_rejections,
            "total_event_name_mismatches": self.total_event_name_mismatches,
            "total_insights_pruned": self.total_insights_pruned,
            "insight_every_n_windows": self.insight_every_n_windows,
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

    def _recency_weight(self, current_window: int, event_window: int) -> float:
        delta = max(current_window - event_window, 0)
        return math.exp(-float(delta) / self.insight_recency_tau)

    def _recompute_insight_scores(self, insights: List[Insight], current_window: int) -> None:
        for insight in insights:
            support_sum = 0.0
            for event_id in insight.supporting_event_ids:
                if event_id not in self._event_id_to_window:
                    continue
                support_sum += self._recency_weight(
                    current_window,
                    self._event_id_to_window[event_id],
                )

            counter_sum = 0.0
            for event_id in insight.counter_event_ids:
                if event_id not in self._event_id_to_window:
                    continue
                counter_sum += self._recency_weight(
                    current_window,
                    self._event_id_to_window[event_id],
                )

            score = support_sum - counter_sum
            insight.score = max(-10.0, min(10.0, score))

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
        current_window: int,
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
                    insight.hypothesis = _strip_hypothesis_evidence_suffix(hypothesis)
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
                hypothesis=_strip_hypothesis_evidence_suffix(hypothesis),
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
            insight.hypothesis = _append_hypothesis_evidence_suffix(
                hypothesis=insight.hypothesis,
                supporting_ids=insight.supporting_event_ids,
                counter_ids=insight.counter_event_ids,
            )
        self._recompute_insight_scores(updated_insights, current_window)

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

    def run_patient_trajectory(
        self,
        windows: List[Dict[str, Any]],
        patient_metadata: Dict[str, Any],
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], MedEvoMemory, MedEvoMemoryDatabase]:
        self.total_patients += 1
        self._current_patient_id = (
            f"{patient_metadata.get('subject_id', 'unknown')}_{patient_metadata.get('icu_stay_id', 'unknown')}"
        )
        self._event_id_to_window = {}
        self._event_id_to_time = {}
        self._event_id_to_raw_event = {}
        self._event_id_to_name_str = {}
        self._next_insight_id = 1

        memory = MedEvoMemory(patient_metadata=deepcopy(patient_metadata))
        memory_db = MedEvoMemoryDatabase()

        if verbose:
            print(f"Processing patient with {len(windows)} windows...")

        processed_window_count = 0
        pending_summaries_for_insight: List[WindowSummary] = []
        pending_critical_for_insight: List[CriticalEvent] = []
        pending_critical_ids = set()
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
                memory.working_memory
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
            if len(memory.critical_events) > self.max_events:
                memory.critical_events = memory.critical_events[-self.max_events :]

            memory.trajectory_memory.append(
                {
                    "type": "window_summary",
                    **summary.to_dict(),
                }
            )

            pending_summaries_for_insight.append(deepcopy(summary))
            for critical in critical_events:
                if critical.event_id in pending_critical_ids:
                    continue
                pending_critical_ids.add(critical.event_id)
                pending_critical_for_insight.append(deepcopy(critical))

            run_insight_agent = (processed_window_count % self.insight_every_n_windows) == 0
            processed_window_count += 1

            insight_parsed: Dict[str, Any] = {"insight_updates": [], "new_insights": []}
            insight_parse_error: Optional[str] = None
            if run_insight_agent:
                insight_parsed, insight_raw, insight_usage, insight_prompt, insight_parse_error = (
                    self.insight_agent.analyze(
                        current_insights=memory.insights,
                        recent_window_summaries=pending_summaries_for_insight,
                        recent_critical_events=pending_critical_for_insight,
                    )
                )
                self.total_insight_calls += 1

                if insight_parse_error is not None:
                    insight_parsed = self._build_default_insight_payload(summary)

                memory.insights = self._apply_insight_updates(
                    existing_insights=memory.insights,
                    parsed_payload=insight_parsed,
                    current_window=idx,
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
                pending_summaries_for_insight = []
                pending_critical_for_insight = []
                pending_critical_ids = set()
            else:
                # Keep insight recency-aware scores up to date even when skipping LLM updates.
                self._recompute_insight_scores(memory.insights, idx)

            memory_db.add_snapshot(memory)

            if verbose:
                maybe_skipped = " insight_skipped=True" if not run_insight_agent else ""
                print(f"insights={len(memory.insights)} critical_events={len(memory.critical_events)}{maybe_skipped}")

        last_window_events = windows[-1].get("current_events", []) if windows else []
        last_hours = float(windows[-1].get("hours_since_admission", 0.0)) if windows else 0.0

        prediction, pred_raw, pred_usage, pred_prompt, pred_parse_error = self.predictor_agent.predict(
            memory=memory,
            last_window_events=last_window_events,
            observation_hours=self.observation_hours,
        )
        self.total_predictor_calls += 1

        self._log_call(
            step_type="med_evo_predictor",
            window_index=-1,
            hours_since_admission=last_hours,
            prompt=pred_prompt,
            response=pred_raw,
            usage=pred_usage,
            parsed_response=prediction,
            metadata={"predictor_parse_error": pred_parse_error},
        )

        return prediction, memory, memory_db
