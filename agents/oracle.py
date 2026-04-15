"""Meta Oracle: Offline evaluator with local ICU context windows."""

from __future__ import annotations

import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from model.llms import LLMClient
from prompts.oracle_prompt import format_event_lines, format_oracle_prompt, format_pre_icu_compression_prompt
from utils.json_parse import parse_json_dict_best_effort

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None


VALID_STATUS = {"improving", "stable", "deteriorating", "insufficient_data"}
VALID_ACTION_REVIEW_LABEL = {"best_practice", "acceptable", "potentially_harmful", "insufficient_data"}
ORACLE_SCHEMA_MAX_ATTEMPTS = 4
OUTCOME_MASK_TOKEN = "[OUTCOME_MASKED]"
OUTCOME_LEAK_TERMS_PATTERN = re.compile(
    r"(?i)\b(expired|deceased|dead|died|passed\s+away|death|hospice|comfort\s+measures)\b"
)
DISCHARGE_SUMMARY_SECTION_HEADERS = (
    "Discharge Disposition:",
    "Discharge Diagnosis:",
    "Discharge Condition:",
    "Discharge Instructions:",
    "Followup Instructions:",
    "Medications on Admission:",
    "Discharge Medications:",
)
PRE_ICU_COMPRESSION_STEP_TYPE = "oracle_pre_icu_history_compressor"


class OracleReport:
    """Structured output from Oracle evaluation."""

    def __init__(
        self,
        patient_assessment: Optional[Dict[str, Any]] = None,
        action_review: Optional[Dict[str, Any]] = None,
        window_data: Optional[Dict[str, Any]] = None,
        context_mode: str = "raw_local_trajectory_icu_events_only",
        context_stats: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        self.patient_assessment = _normalize_patient_assessment(patient_assessment)
        self.action_review = _normalize_action_review(action_review)
        self.window_data = window_data
        self.context_mode = context_mode
        self.context_stats = context_stats or {}
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "patient_assessment": self.patient_assessment,
            "action_review": self.action_review,
            "context_mode": self.context_mode,
            "context_stats": self.context_stats,
        }
        if self.error:
            payload["error"] = self.error
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        window_data: Optional[Dict[str, Any]] = None,
        context_mode: str = "raw_local_trajectory_icu_events_only",
        context_stats: Optional[Dict[str, Any]] = None,
    ) -> "OracleReport":
        return cls(
            patient_assessment=data.get("patient_assessment"),
            action_review=data.get("action_review"),
            window_data=window_data,
            context_mode=context_mode,
            context_stats=context_stats or {},
            error=None,
        )


class MetaOracle:
    """Offline Oracle evaluator with bounded ICU context windows."""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_timeout_seconds: float = 300.0,
        log_dir: Optional[str] = None,
        use_discharge_summary: bool = False,
        include_icu_outcome_in_prompt: bool = True,
        mask_discharge_summary_outcome_terms: bool = False,
        history_context_hours: float = 48.0,
        future_context_hours: float = 48.0,
        top_k_recommendations: int = 3,
        compress_pre_icu_history: bool = False,
    ):
        try:
            self.llm_client = LLMClient(
                provider=provider,
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout_seconds=request_timeout_seconds,
            )
        except TypeError:
            # Backward-compatible path for tests or legacy LLM client wrappers
            # that do not yet accept request_timeout_seconds.
            self.llm_client = LLMClient(
                provider=provider,
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        self.use_discharge_summary = bool(use_discharge_summary)
        self.include_icu_outcome_in_prompt = bool(include_icu_outcome_in_prompt)
        self.mask_discharge_summary_outcome_terms = bool(mask_discharge_summary_outcome_terms)
        self.history_context_hours = float(history_context_hours)
        self.future_context_hours = float(future_context_hours)
        if self.history_context_hours < 0 or self.future_context_hours < 0:
            raise ValueError(
                "history_context_hours and future_context_hours must be >= 0, got: "
                f"{self.history_context_hours}, {self.future_context_hours}"
            )
        self.top_k_recommendations = int(top_k_recommendations)
        if self.top_k_recommendations < 1:
            self.top_k_recommendations = 1
        self.compress_pre_icu_history = bool(compress_pre_icu_history)

        self.evaluation_count = 0
        self.total_tokens_used = 0
        self.total_llm_calls = 0
        self.total_pre_icu_compression_calls = 0
        self.total_pre_icu_compression_tokens = 0
        self.request_timeout_seconds = float(request_timeout_seconds)

        self._stats_lock = Lock()
        self._trajectory_logs: List[Dict[str, Any]] = []
        self._llm_call_logs: List[Dict[str, Any]] = []

        if log_dir is None:
            log_dir = str(Path(__file__).parent.parent / "logs" / "oracle")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print(f"Oracle logs will be saved to: {self.log_dir}")
        print(
            "Oracle context config: "
            f"use_discharge_summary={self.use_discharge_summary}, "
            f"include_icu_outcome_in_prompt={self.include_icu_outcome_in_prompt}, "
            f"mask_discharge_summary_outcome_terms={self.mask_discharge_summary_outcome_terms}, "
            f"history_hours={self.history_context_hours}, "
            f"future_hours={self.future_context_hours}, "
            f"top_k_recommendations={self.top_k_recommendations}, "
            f"compress_pre_icu_history={self.compress_pre_icu_history}, "
            f"request_timeout_seconds={self.request_timeout_seconds:.1f}"
        )

    def _save_log(
        self,
        window_data: Dict[str, Any],
        prompt: str,
        response: Dict[str, Any],
        report: OracleReport,
        context_info: Dict[str, Any],
        error: Optional[str] = None,
        evaluation_number: Optional[int] = None,
    ) -> None:
        if evaluation_number is None:
            with self._stats_lock:
                evaluation_number = self.evaluation_count

        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "evaluation_number": evaluation_number,
            "window_metadata": {
                "subject_id": window_data.get("subject_id"),
                "icu_stay_id": window_data.get("icu_stay_id"),
                "window_start_time": window_data.get("current_window_start"),
                "window_end_time": window_data.get("current_window_end"),
                "hours_since_admission": window_data.get("hours_since_admission"),
                "pre_icu_history_source": window_data.get("pre_icu_history_source"),
                "pre_icu_history_items": window_data.get("pre_icu_history_items"),
                "include_icu_outcome_in_prompt": self.include_icu_outcome_in_prompt,
            },
            "context": context_info,
            "input": {"prompt": prompt},
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
        with self._stats_lock:
            self._trajectory_logs.append(log_entry)

    def save_trajectory_log(self, subject_id: int, icu_stay_id: int, run_id: Optional[str] = None) -> None:
        with self._stats_lock:
            if not self._trajectory_logs:
                return
            evaluations = list(self._trajectory_logs)
            self._trajectory_logs = []

        timestamp = datetime.now().isoformat().replace(":", "-")
        mode = "with_icu_outcome" if self.include_icu_outcome_in_prompt else "without_icu_outcome"
        filename_parts = ["oracle_trajectory", str(subject_id), str(icu_stay_id), mode]
        if run_id:
            filename_parts.append(run_id)
        filename_parts.append(timestamp)
        log_path = self.log_dir / ("_".join(filename_parts) + ".json")

        payload = {
            "metadata": {
                "subject_id": subject_id,
                "icu_stay_id": icu_stay_id,
                "run_id": run_id,
                "mode": mode,
                "timestamp": timestamp,
                "total_windows": len(evaluations),
            },
            "evaluations": evaluations,
        }
        with open(log_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"Saved trajectory log to: {log_path}")

    def _record_llm_call(
        self,
        *,
        step_type: str,
        subject_id: Any,
        icu_stay_id: Any,
        prompt: str,
        response: Dict[str, Any],
        parsed_response: Optional[Dict[str, Any]] = None,
        window_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        log_metadata = dict(metadata or {})
        if isinstance(response, dict):
            if response.get("parse_error") is not None:
                log_metadata["parse_error"] = response.get("parse_error")
            if response.get("model") is not None:
                log_metadata["response_model"] = response.get("model")
        log_metadata.update(
            {
                "step_type": step_type,
                "subject_id": subject_id,
                "icu_stay_id": icu_stay_id,
                "llm_provider": self.llm_client.provider,
                "llm_model": self.llm_client.model,
            }
        )

        window_index = log_metadata.get("window_index")
        if window_index is None and isinstance(window_data, dict):
            window_index = window_data.get("window_index", -1)
        if window_index is None:
            window_index = -1

        hours_since_admission = log_metadata.get("hours_since_admission")
        if hours_since_admission is None and isinstance(window_data, dict):
            hours_since_admission = window_data.get("hours_since_admission")

        entry = {
            "timestamp": datetime.now().isoformat(),
            "patient_id": f"{subject_id}_{icu_stay_id}",
            "step_type": step_type,
            "window_index": window_index,
            "hours_since_admission": hours_since_admission,
            "prompt": prompt,
            "response": response.get("content") if isinstance(response, dict) else None,
            "parsed_response": (
                parsed_response
                if isinstance(parsed_response, dict)
                else (response.get("parsed") if isinstance(response, dict) else None)
            ),
            "input_tokens": int(_safe_float(usage.get("input_tokens"))),
            "output_tokens": int(_safe_float(usage.get("output_tokens"))),
            "metadata": log_metadata,
        }
        with self._stats_lock:
            self._llm_call_logs.append(entry)
            self.total_llm_calls += 1

    def pop_patient_llm_call_logs(self, subject_id: Any, icu_stay_id: Any) -> List[Dict[str, Any]]:
        subject_text = str(subject_id)
        stay_text = str(icu_stay_id)
        keep: List[Dict[str, Any]] = []
        matched: List[Dict[str, Any]] = []

        with self._stats_lock:
            for entry in self._llm_call_logs:
                metadata = entry.get("metadata", {})
                if not isinstance(metadata, dict):
                    keep.append(entry)
                    continue

                entry_subject = str(metadata.get("subject_id"))
                entry_stay = str(metadata.get("icu_stay_id"))
                if entry_subject == subject_text and entry_stay == stay_text:
                    matched.append(entry)
                else:
                    keep.append(entry)

            self._llm_call_logs = keep
        return matched

    def pop_patient_trajectory_logs(self, subject_id: Any, icu_stay_id: Any) -> List[Dict[str, Any]]:
        subject_text = str(subject_id)
        stay_text = str(icu_stay_id)
        keep: List[Dict[str, Any]] = []
        matched: List[Dict[str, Any]] = []

        with self._stats_lock:
            for entry in self._trajectory_logs:
                window_metadata = entry.get("window_metadata", {})
                if not isinstance(window_metadata, dict):
                    keep.append(entry)
                    continue

                entry_subject = str(window_metadata.get("subject_id"))
                entry_stay = str(window_metadata.get("icu_stay_id"))
                if entry_subject == subject_text and entry_stay == stay_text:
                    matched.append(entry)
                else:
                    keep.append(entry)

            self._trajectory_logs = keep
        return matched

    def prepare_context(
        self,
        window_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build Oracle context directly from pre-sliced window payload events."""
        current_window_start = _parse_time(window_data.get("current_window_start"))
        current_window_end = _parse_time(window_data.get("current_window_end"))
        if (
            current_window_start is not None
            and current_window_end is not None
            and current_window_end < current_window_start
        ):
            current_window_end = current_window_start

        history_events = _normalize_window_event_payload(window_data.get("history_events"))
        current_window_events = _normalize_window_event_payload(window_data.get("current_events"))
        future_events = _normalize_window_event_payload(window_data.get("future_events"))
        context_events = [*history_events, *current_window_events, *future_events]
        context_start, context_end = _event_time_bounds(context_events)
        if context_start is None:
            context_start = current_window_start
        if context_end is None:
            context_end = current_window_end
        if context_start and context_end and context_end < context_start:
            context_end = context_start

        history_hours = _derive_history_hours(
            history_events=history_events,
            current_window_start=current_window_start,
        )
        future_hours = _derive_future_hours(
            future_events=future_events,
            current_window_end=current_window_end,
        )
        anchor_time = current_window_start or current_window_end
        current_discharge_summary = (
            window_data.get("current_discharge_summary")
            if isinstance(window_data.get("current_discharge_summary"), dict)
            else None
        )
        history_events_for_log = _sanitize_context_events(history_events)
        current_window_events_for_log = _sanitize_context_events(current_window_events)
        future_events_for_log = _sanitize_context_events(future_events)

        events_context_text = _build_raw_context_text(
            history_events=history_events,
            current_window_events=current_window_events,
            future_events=future_events,
            context_start=context_start,
            context_end=context_end,
            current_window_start=current_window_start,
            current_window_end=current_window_end,
            anchor_time=anchor_time,
            history_hours=history_hours,
            future_hours=future_hours,
        )

        current_discharge_summary_text = _build_current_discharge_summary_context_text(
            current_discharge_summary,
            sanitize_for_outcome=(
                self.use_discharge_summary
                and (not self.include_icu_outcome_in_prompt or self.mask_discharge_summary_outcome_terms)
            ),
        )

        if self.use_discharge_summary:
            context_text = "\n\n".join([current_discharge_summary_text, events_context_text]).strip()
            context_mode = "raw_local_trajectory_with_icu_discharge_summary"
        else:
            context_text = events_context_text
            context_mode = "raw_local_trajectory_icu_events_only"

        context_tokens = self._estimate_tokens(context_text)
        has_current_discharge_summary = bool(current_discharge_summary)
        current_discharge_summary_selection_rule = (
            _safe_text(current_discharge_summary.get("selection_rule"))
            if isinstance(current_discharge_summary, dict)
            else ""
        ) or None
        return {
            "mode": context_mode,
            "context_text": context_text,
            "context_tokens": context_tokens,
            "context_event_count": len(context_events),
            "context_history_event_count": len(history_events),
            "context_current_window_event_count": len(current_window_events),
            "context_future_event_count": len(future_events),
            "context_history_events": history_events_for_log,
            "context_current_window_events": current_window_events_for_log,
            "context_future_events": future_events_for_log,
            "use_discharge_summary": self.use_discharge_summary,
            "context_anchor_time": anchor_time.isoformat() if anchor_time else None,
            "context_window_start": context_start.isoformat() if context_start else None,
            "context_window_end": context_end.isoformat() if context_end else None,
            "history_hours": history_hours,
            "future_hours": future_hours,
            "has_current_discharge_summary": has_current_discharge_summary,
            "current_discharge_summary_selection_rule": current_discharge_summary_selection_rule,
            "icu_discharge_summary_count": 1 if has_current_discharge_summary else 0,
        }

    def compress_pre_icu_history_for_windows(self, windows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Compress shared pre-ICU history once and attach to all windows."""
        if not self.compress_pre_icu_history:
            return None
        if not windows:
            return None

        first_window = windows[0] if isinstance(windows[0], dict) else {}
        pre_icu_history = first_window.get("pre_icu_history")
        if not isinstance(pre_icu_history, dict):
            return None

        source = _safe_text(pre_icu_history.get("source")).lower()
        raw_content = _safe_text(pre_icu_history.get("content"))
        if source == "llm_compressed" and raw_content:
            compression_info = pre_icu_history.get("compression")
            if isinstance(compression_info, dict):
                return dict(compression_info)
            return {"status": "already_compressed"}

        if not raw_content:
            return None

        prompt = format_pre_icu_compression_prompt(pre_icu_history)
        response: Dict[str, Any] = {}
        parsed: Dict[str, Any] = {}
        parse_source = "provider"
        compression_error: Optional[str] = None

        try:
            response = self.llm_client.chat(
                prompt=prompt,
                timeout_seconds=self.request_timeout_seconds,
            )
            parsed_candidate = response.get("parsed")
            if parsed_candidate is None:
                parse_source = "best_effort_json"
                parsed_candidate = _best_effort_parse_json(response.get("content", ""))
            if isinstance(parsed_candidate, dict):
                parsed = parsed_candidate
        except Exception as exc:
            compression_error = str(exc)
            parse_source = "error_fallback"

        compressed_text = _safe_text(parsed.get("compressed_pre_icu_history"))
        if not compressed_text:
            if parse_source == "provider":
                parse_source = "heuristic_fallback"
            compressed_text = _build_fallback_pre_icu_compression(pre_icu_history)

        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        compression_tokens = _usage_tokens(usage)
        with self._stats_lock:
            self.total_pre_icu_compression_calls += 1
            self.total_pre_icu_compression_tokens += compression_tokens
            self.total_tokens_used += compression_tokens

        original_chars = len(raw_content)
        compression_metadata = {
            "method": "llm",
            "step_type": PRE_ICU_COMPRESSION_STEP_TYPE,
            "original_source": source or "unknown",
            "original_content_chars": len(raw_content),
            "compressed_chars": len(compressed_text),
            "parse_source": parse_source,
            "error": compression_error,
        }
        if original_chars > 0:
            compression_metadata["compression_ratio"] = round(len(compressed_text) / float(original_chars), 4)

        response_for_log = dict(response) if isinstance(response, dict) else {}
        parsed_for_log = parsed if isinstance(parsed, dict) else {}
        parsed_for_log = dict(parsed_for_log)
        if compressed_text and not _safe_text(parsed_for_log.get("compressed_pre_icu_history")):
            parsed_for_log["compressed_pre_icu_history"] = compressed_text
        parsed_for_log["output_source"] = parse_source

        if not _safe_text(response_for_log.get("content")):
            response_for_log["content"] = json.dumps(
                {"compressed_pre_icu_history": compressed_text},
                ensure_ascii=False,
            )

        self._record_llm_call(
            step_type=PRE_ICU_COMPRESSION_STEP_TYPE,
            subject_id=first_window.get("subject_id"),
            icu_stay_id=first_window.get("icu_stay_id"),
            prompt=prompt,
            response=response_for_log,
            parsed_response=parsed_for_log,
            window_data=first_window,
            metadata={
                "window_start_time": first_window.get("current_window_start"),
                "window_end_time": first_window.get("current_window_end"),
                "pre_icu_history_source": first_window.get("pre_icu_history_source"),
                "pre_icu_history_items": first_window.get("pre_icu_history_items"),
                "original_source": source or "unknown",
                "original_content_chars": len(raw_content),
                "compressed_chars": len(compressed_text),
                "compressed_pre_icu_history": compressed_text,
                "parse_source": parse_source,
                "error": compression_error,
            },
        )

        compressed_pre_icu_history = dict(pre_icu_history)
        compressed_pre_icu_history.pop("baseline_content", None)
        compressed_pre_icu_history.pop("baseline_events_count", None)
        compressed_pre_icu_history.pop("fallback_hours", None)
        compressed_pre_icu_history["source"] = "llm_compressed"
        compressed_pre_icu_history["content"] = compressed_text
        compressed_pre_icu_history["compression"] = compression_metadata

        applied_windows = 0
        for window in windows:
            if not isinstance(window, dict):
                continue
            if not isinstance(window.get("pre_icu_history"), dict):
                continue
            window["pre_icu_history"] = dict(compressed_pre_icu_history)
            window["pre_icu_history_source"] = "llm_compressed"
            applied_windows += 1

        compression_metadata["applied_to_windows"] = applied_windows
        return compression_metadata

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0

        if tiktoken is None:
            return math.ceil(len(text) / 4)

        model_name = self.llm_client.model or ""
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                return math.ceil(len(text) / 4)

        try:
            return len(encoding.encode(text))
        except Exception:
            return math.ceil(len(text) / 4)

    def evaluate_window(
        self,
        window_data: Dict[str, Any],
    ) -> OracleReport:
        context_info = self.prepare_context(window_data)
        prompt = format_oracle_prompt(
            window_data=window_data,
            context_block=context_info.get("context_text", ""),
            context_mode=context_info.get("mode", "raw_local_trajectory_icu_events_only"),
            history_hours=context_info.get("history_hours"),
            future_hours=context_info.get("future_hours"),
            top_k=self.top_k_recommendations,
            include_icu_outcome=self.include_icu_outcome_in_prompt,
        )
        response: Dict[str, Any] = {}
        parsed_payload: Dict[str, Any] = {}
        parse_source = "best_effort_json"
        evaluation_number: Optional[int] = None
        attempt_errors: List[str] = []
        total_attempt_tokens = 0
        try:
            for attempt_index in range(1, ORACLE_SCHEMA_MAX_ATTEMPTS + 1):
                response = {}
                parsed_payload = {}
                parse_source = "best_effort_json"
                schema_error: Optional[str] = None
                call_error: Optional[str] = None

                try:
                    response = self.llm_client.chat(
                        prompt=prompt,
                        response_format="text",
                        timeout_seconds=self.request_timeout_seconds,
                    )
                    total_attempt_tokens += _usage_tokens(response.get("usage", {}))
                    parsed_candidate = response.get("parsed")
                    if isinstance(parsed_candidate, dict):
                        parsed_payload = parsed_candidate
                        parse_source = "provider"
                    else:
                        parsed_payload = _best_effort_parse_json(response.get("content", ""))
                        parse_source = "best_effort_json"
                    schema_ok, schema_error = _validate_oracle_output_schema(parsed_payload)
                except Exception as exc:
                    call_error = str(exc)
                    schema_ok = False
                    schema_error = f"LLM call error: {call_error}"

                self._record_llm_call(
                    step_type="oracle_evaluator",
                    subject_id=window_data.get("subject_id"),
                    icu_stay_id=window_data.get("icu_stay_id"),
                    prompt=prompt,
                    response=response,
                    parsed_response=parsed_payload if isinstance(parsed_payload, dict) else None,
                    window_data=window_data,
                    metadata={
                        "window_start_time": window_data.get("current_window_start"),
                        "window_end_time": window_data.get("current_window_end"),
                        "context_mode": context_info.get("mode"),
                        "use_discharge_summary": context_info.get("use_discharge_summary", self.use_discharge_summary),
                        "context_tokens": context_info.get("context_tokens"),
                        "context_event_count": context_info.get("context_event_count"),
                        "context_history_event_count": context_info.get("context_history_event_count"),
                        "context_current_window_event_count": context_info.get("context_current_window_event_count"),
                        "context_future_event_count": context_info.get("context_future_event_count"),
                        "context_anchor_time": context_info.get("context_anchor_time"),
                        "context_window_start": context_info.get("context_window_start"),
                        "context_window_end": context_info.get("context_window_end"),
                        "history_hours": context_info.get("history_hours"),
                        "future_hours": context_info.get("future_hours"),
                        "context_history_events": context_info.get("context_history_events", []),
                        "context_current_window_events": context_info.get("context_current_window_events", []),
                        "context_future_events": context_info.get("context_future_events", []),
                        "pre_icu_history_source": window_data.get("pre_icu_history_source"),
                        "pre_icu_history_items": window_data.get("pre_icu_history_items"),
                        "current_discharge_summary_selection_rule": context_info.get(
                            "current_discharge_summary_selection_rule"
                        ),
                        "has_current_discharge_summary": context_info.get("has_current_discharge_summary", False),
                        "include_icu_outcome_in_prompt": self.include_icu_outcome_in_prompt,
                        "mask_discharge_summary_outcome_terms": self.mask_discharge_summary_outcome_terms,
                        "parse_source": parse_source,
                        "schema_valid": bool(schema_ok),
                        "schema_error": schema_error,
                        "attempt_index": attempt_index,
                        "attempt_max": ORACLE_SCHEMA_MAX_ATTEMPTS,
                    },
                )

                if schema_ok:
                    break
                attempt_errors.append(schema_error or "Schema validation failed.")
            else:
                joined_errors = " | ".join(attempt_errors)
                raise RuntimeError(
                    f"Oracle schema validation failed after {ORACLE_SCHEMA_MAX_ATTEMPTS} attempts: {joined_errors}"
                )

            with self._stats_lock:
                self.evaluation_count += 1
                self.total_tokens_used += int(total_attempt_tokens)
                evaluation_number = self.evaluation_count

            report = OracleReport.from_dict(
                data=parsed_payload if isinstance(parsed_payload, dict) else {},
                window_data=window_data,
                context_mode=context_info.get("mode", "raw_local_trajectory_icu_events_only"),
                context_stats={
                    "context_tokens": context_info.get("context_tokens"),
                    "context_event_count": context_info.get("context_event_count"),
                    "context_history_event_count": context_info.get("context_history_event_count"),
                    "context_current_window_event_count": context_info.get("context_current_window_event_count"),
                    "context_future_event_count": context_info.get("context_future_event_count"),
                    "context_anchor_time": context_info.get("context_anchor_time"),
                    "context_window_start": context_info.get("context_window_start"),
                    "context_window_end": context_info.get("context_window_end"),
                    "history_hours": context_info.get("history_hours"),
                    "future_hours": context_info.get("future_hours"),
                    "use_discharge_summary": context_info.get("use_discharge_summary", self.use_discharge_summary),
                    "has_current_discharge_summary": context_info.get("has_current_discharge_summary", False),
                    "current_discharge_summary_selection_rule": context_info.get(
                        "current_discharge_summary_selection_rule"
                    ),
                    "include_icu_outcome_in_prompt": self.include_icu_outcome_in_prompt,
                    "mask_discharge_summary_outcome_terms": self.mask_discharge_summary_outcome_terms,
                    "icu_discharge_summary_count": context_info.get("icu_discharge_summary_count", 0),
                },
            )

            self._save_log(
                window_data,
                prompt,
                response,
                report,
                context_info=context_info,
                error=None,
                evaluation_number=evaluation_number,
            )
            return report

        except Exception as e:
            self._save_log(
                window_data,
                prompt,
                response if isinstance(response, dict) else {"content": None, "error": str(e)},
                OracleReport(
                    patient_assessment=_normalize_patient_assessment({}),
                    action_review=_normalize_action_review({}),
                    window_data=window_data,
                    context_mode=context_info.get("mode", "raw_local_trajectory_icu_events_only"),
                    context_stats={
                        "context_tokens": context_info.get("context_tokens"),
                        "context_event_count": context_info.get("context_event_count"),
                        "context_history_event_count": context_info.get("context_history_event_count"),
                        "context_current_window_event_count": context_info.get("context_current_window_event_count"),
                        "context_future_event_count": context_info.get("context_future_event_count"),
                        "use_discharge_summary": context_info.get("use_discharge_summary", self.use_discharge_summary),
                        "has_current_discharge_summary": context_info.get("has_current_discharge_summary", False),
                        "current_discharge_summary_selection_rule": context_info.get(
                            "current_discharge_summary_selection_rule"
                        ),
                        "include_icu_outcome_in_prompt": self.include_icu_outcome_in_prompt,
                        "mask_discharge_summary_outcome_terms": self.mask_discharge_summary_outcome_terms,
                        "context_anchor_time": context_info.get("context_anchor_time"),
                        "context_window_start": context_info.get("context_window_start"),
                        "context_window_end": context_info.get("context_window_end"),
                    },
                    error=f"Oracle evaluation error: {e}",
                ),
                context_info=context_info,
                error=str(e),
                evaluation_number=evaluation_number,
            )
            raise

    def evaluate_trajectory(self, windows: List[Dict[str, Any]]) -> List[OracleReport]:
        reports: List[OracleReport] = []
        self.compress_pre_icu_history_for_windows(windows)

        for i, window in enumerate(windows):
            window_payload = dict(window)
            window_payload.setdefault("window_index", i)
            print(f"Evaluating window {i+1}/{len(windows)} (Hour {window_payload['hours_since_admission']:.1f})...")
            report = self.evaluate_window(window_payload)
            reports.append(report)

        return reports

    def evaluate_trajectory_parallel(
        self,
        windows: List[Dict[str, Any]],
        max_workers: int = 10,
        show_progress: bool = True,
    ) -> List[OracleReport]:
        if not windows:
            return []
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")

        worker_count = min(max_workers, len(windows))
        if show_progress:
            print(f"Starting parallel evaluation with {worker_count} workers...")

        self.compress_pre_icu_history_for_windows(windows)
        windows_with_index = []
        for i, window in enumerate(windows):
            window_payload = dict(window)
            window_payload.setdefault("window_index", i)
            windows_with_index.append(window_payload)

        results: List[Optional[OracleReport]] = [None] * len(windows)
        completed_count = 0

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(self.evaluate_window, window): i for i, window in enumerate(windows_with_index)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                window = windows_with_index[index]
                window_hour = _format_window_hour(window)

                try:
                    report = future.result()
                    results[index] = report
                    completed_count += 1
                    if show_progress:
                        print(f"Completed window {completed_count}/{len(windows)} " f"(Hour {window_hour})")
                except Exception as e:
                    for pending_future in future_to_index:
                        if not pending_future.done():
                            pending_future.cancel()
                    raise RuntimeError(f"Error evaluating window {index} (Hour {window_hour}): {e}") from e

        if show_progress:
            print(f"Parallel evaluation complete: {completed_count}/{len(windows)} windows evaluated")

        reports: List[OracleReport] = []
        for index, report in enumerate(results):
            if report is not None:
                reports.append(report)
                continue

            raise RuntimeError(f"Parallel evaluation produced no result for window index={index}.")

        return reports

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_evaluations": self.evaluation_count,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens_used": self.total_tokens_used,
            "total_pre_icu_compression_calls": self.total_pre_icu_compression_calls,
            "total_pre_icu_compression_tokens": self.total_pre_icu_compression_tokens,
            "avg_tokens_per_evaluation": (
                self.total_tokens_used / self.evaluation_count if self.evaluation_count > 0 else 0
            ),
            "use_discharge_summary": self.use_discharge_summary,
            "include_icu_outcome_in_prompt": self.include_icu_outcome_in_prompt,
            "mask_discharge_summary_outcome_terms": self.mask_discharge_summary_outcome_terms,
            "history_context_hours": self.history_context_hours,
            "future_context_hours": self.future_context_hours,
            "top_k_recommendations": self.top_k_recommendations,
            "compress_pre_icu_history": self.compress_pre_icu_history,
        }


def save_oracle_reports(reports: List[OracleReport], output_path: str, include_window_data: bool = False) -> None:
    output_data = []

    for report in reports:
        report_dict = report.to_dict()

        if include_window_data and report.window_data:
            report_dict["window_metadata"] = {
                "subject_id": report.window_data.get("subject_id"),
                "icu_stay_id": report.window_data.get("icu_stay_id"),
                "window_start_time": report.window_data.get("current_window_start"),
                "window_end_time": report.window_data.get("current_window_end"),
                "hours_since_admission": report.window_data.get("hours_since_admission"),
            }

        output_data.append(report_dict)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(reports)} Oracle reports to {output_path}")


def load_oracle_reports(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r") as f:
        reports = json.load(f)

    print(f"Loaded {len(reports)} Oracle reports from {input_path}")
    return reports


# -----------------------------
# Helpers
# -----------------------------


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _usage_tokens(usage: Dict[str, Any]) -> int:
    if not isinstance(usage, dict):
        return 0
    return int(_safe_float(usage.get("input_tokens"))) + int(_safe_float(usage.get("output_tokens")))


def _format_window_hour(window: Dict[str, Any]) -> str:
    try:
        return f"{float(window.get('hours_since_admission')):.1f}"
    except (TypeError, ValueError):
        return "unknown"


def _parse_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value

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


def _sanitize_context_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        cleaned = {key: value for key, value in event.items() if not str(key).startswith("_")}
        sanitized.append(cleaned)
    return sanitized


def _normalize_window_event_payload(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


def _event_time_from_payload(event: Dict[str, Any]) -> Optional[datetime]:
    return _parse_time(event.get("time") or event.get("start_time"))


def _event_time_bounds(events: List[Dict[str, Any]]) -> Tuple[Optional[datetime], Optional[datetime]]:
    times: List[datetime] = []
    for event in events:
        event_time = _event_time_from_payload(event)
        if isinstance(event_time, datetime):
            times.append(event_time)
    if not times:
        return None, None
    return min(times), max(times)


def _derive_history_hours(
    *,
    history_events: List[Dict[str, Any]],
    current_window_start: Optional[datetime],
) -> Optional[float]:
    if current_window_start is None:
        return None
    if not history_events:
        return 0.0
    first_history_time, _ = _event_time_bounds(history_events)
    if first_history_time is None:
        return None
    return max(0.0, (current_window_start - first_history_time).total_seconds() / 3600.0)


def _derive_future_hours(
    *,
    future_events: List[Dict[str, Any]],
    current_window_end: Optional[datetime],
) -> Optional[float]:
    if current_window_end is None:
        return None
    if not future_events:
        return 0.0
    _, last_future_time = _event_time_bounds(future_events)
    if last_future_time is None:
        return None
    return max(0.0, (last_future_time - current_window_end).total_seconds() / 3600.0)


def _format_optional_hours(value: Optional[float]) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "unknown"
    return f"{float(value):.1f}"


def _build_raw_context_text(
    *,
    history_events: List[Dict[str, Any]],
    current_window_events: List[Dict[str, Any]],
    future_events: List[Dict[str, Any]],
    context_start: Optional[datetime],
    context_end: Optional[datetime],
    current_window_start: Optional[datetime],
    current_window_end: Optional[datetime],
    anchor_time: Optional[datetime],
    history_hours: Optional[float],
    future_hours: Optional[float],
) -> str:
    anchor_text = anchor_time.isoformat() if anchor_time else "unknown"
    start_text = context_start.isoformat() if context_start else "unknown"
    end_text = context_end.isoformat() if context_end else "unknown"
    current_start_text = current_window_start.isoformat() if current_window_start else "unknown"
    current_end_text = current_window_end.isoformat() if current_window_end else "unknown"
    if (
        current_window_start is not None
        and current_window_end is not None
        and current_window_end >= current_window_start
    ):
        current_window_duration_hours = (current_window_end - current_window_start).total_seconds() / 3600.0
        current_window_duration_text = f"{current_window_duration_hours:.2f}"
    else:
        current_window_duration_text = "unknown"
    total_context_events = len(history_events) + len(current_window_events) + len(future_events)

    lines = [
        "## ICU TRAJECTORY CONTEXT WINDOW",
        f"Total ICU Context window: [{start_text}, {end_text}]",
        f"Total ICU context events: {total_context_events}",
        "",
        "## HISTORY EVENTS OF CURRENT WINDOW",
        f"Total history events: {len(history_events)}",
        f"History duration (hours): {_format_optional_hours(history_hours)}",
    ]

    lines.extend(format_event_lines(history_events, empty_text="(No history events before current window)"))

    lines.extend(
        [
            "",
            "## CURRENT OBSERVATION WINDOW FOR EVALUATION",
            f"Current observation window: [{current_start_text}, {current_end_text}]",
            f"Current window duration (hours): {current_window_duration_text}",
            f"Total trajectory events in current window: {len(current_window_events)}",
        ]
    )

    lines.extend(format_event_lines(current_window_events, empty_text="(No events in current observation window)"))

    lines.extend(
        [
            "",
            "## FUTURE EVENTS",
            f"Total future events: {len(future_events)}",
            f"Future duration (hours): {_format_optional_hours(future_hours)}",
        ]
    )

    lines.extend(format_event_lines(future_events, empty_text="(No future events after current window)"))

    return "\n".join(lines)


def _mask_outcome_terms(text: str) -> str:
    if not text:
        return ""
    return OUTCOME_LEAK_TERMS_PATTERN.sub(OUTCOME_MASK_TOKEN, text)


def _remove_summary_section(text: str, section_header: str) -> str:
    if not text:
        return ""

    section_start = re.escape(section_header)
    section_end_candidates = "|".join(
        re.escape(header) for header in DISCHARGE_SUMMARY_SECTION_HEADERS if header.lower() != section_header.lower()
    )
    pattern = re.compile(rf"(?is){section_start}\s*.*?(?=(?:{section_end_candidates})|$)")
    return pattern.sub("", text)


def _sanitize_discharge_summary_text(text: str) -> str:
    if not text:
        return ""

    sanitized = _remove_summary_section(text, "Discharge Disposition:")
    sanitized = _remove_summary_section(sanitized, "Discharge Condition:")
    sanitized = _mask_outcome_terms(sanitized)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()
    return sanitized


def _build_current_discharge_summary_context_text(
    current_discharge_summary: Optional[Dict[str, Any]],
    *,
    sanitize_for_outcome: bool = False,
) -> str:
    lines = [
        "## CURRENT DISCHARGE SUMMARY",
    ]

    if not isinstance(current_discharge_summary, dict) or not current_discharge_summary:
        lines.append("(No ICU-stay-matched discharge summary found)")
        return "\n".join(lines)

    summary_time = _safe_text(current_discharge_summary.get("time")) or "unknown"
    selection_rule = _safe_text(current_discharge_summary.get("selection_rule")) or "unknown"
    lines.append(f"Summary time: {summary_time}")
    lines.append(f"Selection rule: {selection_rule}")

    hours_after_leave = current_discharge_summary.get("hours_after_icu_leave")
    if isinstance(hours_after_leave, (int, float)):
        lines.append(f"Hours after ICU leave: {float(hours_after_leave):.2f}")

    details = _safe_text(current_discharge_summary.get("code_specifics"))
    if sanitize_for_outcome:
        details = _mask_outcome_terms(details)
    if details:
        lines.append(f"Details: {details}")

    summary_text = _safe_text(current_discharge_summary.get("text_value"))
    if sanitize_for_outcome:
        summary_text = _sanitize_discharge_summary_text(summary_text)

    lines.append("")
    lines.append(summary_text if summary_text else "(No discharge summary text)")
    return "\n".join(lines).strip()


def _truncate_text(text: str, max_chars: int) -> str:
    cleaned = _safe_text(text)
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(0, max_chars - 3)].rstrip() + "..."


def _build_fallback_pre_icu_compression(pre_icu_history: Dict[str, Any]) -> str:
    source = _safe_text(pre_icu_history.get("source")) or "unknown"
    items = int(_safe_float(pre_icu_history.get("items")))
    history_hours = _safe_float(pre_icu_history.get("history_hours"))
    content = _safe_text(pre_icu_history.get("content"))

    lines: List[str] = [
        f"Source: {source}",
        f"Historical items: {items}",
    ]
    if history_hours > 0:
        lines.append(f"History lookback hours: {history_hours:.1f}")
    if content:
        lines.append("Historical reports (truncated):")
        content_lines = [line.strip() for line in content.splitlines() if line.strip()]
        lines.extend(content_lines[:20])
    else:
        lines.append("Historical reports: none")

    return _truncate_text("\n".join(lines), max_chars=1800)


def _normalize_overall_label(value: Any) -> str:
    label = _safe_text(value).lower().replace("-", "_").replace(" ", "_")
    label = {
        "insufficient": "insufficient_data",
        "not_enough_data": "insufficient_data",
        "unknown": "insufficient_data",
    }.get(label, label)
    if label not in VALID_STATUS:
        return "insufficient_data"
    return label


def _normalize_action_review_label(value: Any) -> str:
    label = _safe_text(value).lower().replace("-", "_").replace(" ", "_")
    label = {
        "best": "best_practice",
        "good": "acceptable",
        "harmful": "potentially_harmful",
        "potential_harm": "potentially_harmful",
        "insufficient": "insufficient_data",
        "unknown": "insufficient_data",
    }.get(label, label)
    if label not in VALID_ACTION_REVIEW_LABEL:
        return "insufficient_data"
    return label


def _normalize_key_evidence(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    key_evidence: List[str] = []
    for item in value:
        if isinstance(item, dict):
            token = _safe_text(item.get("id") or item.get("event_id") or item.get("ref"))
        else:
            token = _safe_text(item)
        if token:
            key_evidence.append(token)
    return key_evidence


def _normalize_patient_assessment(value: Any) -> Dict[str, Any]:
    data = value if isinstance(value, dict) else {}
    overall_data = data.get("overall") if isinstance(data.get("overall"), dict) else {}

    overall_label = _normalize_overall_label(overall_data.get("label"))
    overall_rationale = _safe_text(overall_data.get("rationale"))
    if not overall_rationale:
        overall_rationale = "Insufficient data to assess patient trajectory at this window."

    active_risks: List[Dict[str, Any]] = []
    active_risks_raw = data.get("active_risks")
    if isinstance(active_risks_raw, list):
        for risk in active_risks_raw:
            if not isinstance(risk, dict):
                continue
            risk_name = _safe_text(risk.get("risk_name"))
            if not risk_name:
                continue
            active_risks.append(
                {
                    "risk_name": risk_name,
                    "key_evidence": _normalize_key_evidence(risk.get("key_evidence")),
                }
            )

    return {
        "overall": {
            "label": overall_label,
            "rationale": overall_rationale,
        },
        "active_risks": active_risks,
    }


def _normalize_action_review(value: Any) -> Dict[str, Any]:
    data = value if isinstance(value, dict) else {}

    evaluations: List[Dict[str, Any]] = []
    evaluations_raw = data.get("evaluations")
    if isinstance(evaluations_raw, list):
        for index, evaluation in enumerate(evaluations_raw, start=1):
            if not isinstance(evaluation, dict):
                continue

            action_name = _safe_text(evaluation.get("action_name"))
            if not action_name:
                action_name = _safe_text(evaluation.get("action"))
            if not action_name:
                action_name = _safe_text(evaluation.get("action_description"))
            if not action_name:
                continue

            action_id = _safe_text(evaluation.get("action_id")) or str(index)
            label = _normalize_action_review_label(evaluation.get("label"))
            rationale = _safe_text(evaluation.get("rationale"))
            if not rationale:
                rationale = "No rationale provided."

            evaluations.append(
                {
                    "action_id": action_id,
                    "action_name": action_name,
                    "label": label,
                    "rationale": rationale,
                }
            )

    red_flags: List[Dict[str, Any]] = []
    red_flags_raw = data.get("red_flags")
    if isinstance(red_flags_raw, list):
        for red_flag in red_flags_raw:
            if not isinstance(red_flag, dict):
                continue
            contraindicated_action = _safe_text(red_flag.get("contraindicated_action"))
            reason = _safe_text(red_flag.get("reason"))
            if not contraindicated_action or not reason:
                continue
            red_flags.append(
                {
                    "contraindicated_action": contraindicated_action,
                    "reason": reason,
                    "key_evidence": _normalize_key_evidence(red_flag.get("key_evidence")),
                }
            )

    return {
        "evaluations": evaluations,
        "red_flags": red_flags,
    }


def _validate_oracle_output_schema(payload: Any) -> Tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "Top-level payload is not a JSON object."

    patient_assessment = payload.get("patient_assessment")
    if not isinstance(patient_assessment, dict):
        return False, "Missing patient_assessment object."

    overall = patient_assessment.get("overall")
    if not isinstance(overall, dict):
        return False, "Missing patient_assessment.overall object."

    overall_label = _safe_text(overall.get("label")).lower().replace("-", "_").replace(" ", "_")
    if overall_label not in VALID_STATUS:
        return False, "Invalid patient_assessment.overall.label."

    overall_rationale = _safe_text(overall.get("rationale"))
    if not overall_rationale:
        return False, "Missing patient_assessment.overall.rationale."

    active_risks = patient_assessment.get("active_risks")
    if not isinstance(active_risks, list):
        return False, "patient_assessment.active_risks must be a list."
    for risk in active_risks:
        if not isinstance(risk, dict):
            return False, "Each active_risks item must be an object."
        risk_name = _safe_text(risk.get("risk_name"))
        if not risk_name:
            return False, "Each active_risks item must include risk_name."
        if not isinstance(risk.get("key_evidence"), list):
            return False, "Each active_risks item must include key_evidence list."

    action_review = payload.get("action_review")
    if not isinstance(action_review, dict):
        return False, "Missing action_review object."

    evaluations = action_review.get("evaluations")
    if not isinstance(evaluations, list):
        return False, "action_review.evaluations must be a list."
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            return False, "Each evaluations item must be an object."
        action_id = _safe_text(evaluation.get("action_id"))
        action_name = _safe_text(evaluation.get("action_name"))
        label = _safe_text(evaluation.get("label")).lower().replace("-", "_").replace(" ", "_")
        rationale = _safe_text(evaluation.get("rationale"))
        if not action_id:
            return False, "Each evaluations item must include action_id."
        if not action_name:
            return False, "Each evaluations item must include action_name."
        if label not in VALID_ACTION_REVIEW_LABEL:
            return False, "Invalid evaluations.label."
        if not rationale:
            return False, "Each evaluations item must include rationale."

    red_flags = action_review.get("red_flags")
    if not isinstance(red_flags, list):
        return False, "action_review.red_flags must be a list."
    for red_flag in red_flags:
        if not isinstance(red_flag, dict):
            return False, "Each red_flags item must be an object."
        contraindicated_action = _safe_text(red_flag.get("contraindicated_action"))
        reason = _safe_text(red_flag.get("reason"))
        if not contraindicated_action:
            return False, "Each red_flags item must include contraindicated_action."
        if not reason:
            return False, "Each red_flags item must include reason."
        if not isinstance(red_flag.get("key_evidence"), list):
            return False, "Each red_flags item must include key_evidence list."

    return True, ""


def _best_effort_parse_json(content: str) -> Dict[str, Any]:
    parsed = parse_json_dict_best_effort(content)
    return parsed or {}
