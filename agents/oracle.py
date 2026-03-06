"""Meta Oracle: Offline evaluator with local ICU context windows."""

from __future__ import annotations

import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from model.llms import LLMClient
from prompts.oracle_prompt import format_event_line, format_oracle_prompt

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None


DOMAIN_KEYS = ("hemodynamics", "respiratory", "renal_metabolic", "neurology")
VALID_STATUS = {"improving", "stable", "deteriorating", "fluctuating", "insufficient_data"}
VALID_ACTION_CATEGORY = {"medication_start", "medication_order", "procedure", "transfer", "other"}
VALID_QUALITY_RATING = {"optimal", "neutral", "sub_optimal"}
VALID_GUIDELINE_ADHERENCE = {"adherent", "non_adherent", "not_applicable", "guideline_unclear"}
VALID_CONTEXTUAL_APPROPRIATENESS = {"appropriate", "suboptimal", "potentially_harmful", "not_enough_context"}


class OracleReport:
    """Structured output from Oracle evaluation."""

    def __init__(
        self,
        patient_status: Dict[str, Any],
        doctor_actions: List[Dict[str, Any]],
        action_evaluations: Optional[List[Dict[str, Any]]] = None,
        overall_window_summary: str = "",
        clinical_quality: Optional[Dict[str, Any]] = None,
        primary_clinical_driver: str = "",
        clinical_pearl: str = "",
        window_data: Optional[Dict[str, Any]] = None,
        context_mode: str = "raw_local_trajectory_icu_events_only",
        context_stats: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        self.patient_status = patient_status
        self.doctor_actions = doctor_actions
        self.action_evaluations = _normalize_action_evaluations(action_evaluations)
        self.overall_window_summary = _normalize_overall_window_summary(overall_window_summary)
        self.clinical_quality = _normalize_clinical_quality(
            clinical_quality,
            action_evaluations=self.action_evaluations,
            overall_window_summary=self.overall_window_summary,
        )
        self.primary_clinical_driver = _normalize_primary_clinical_driver(primary_clinical_driver)
        self.clinical_pearl = _normalize_clinical_pearl(clinical_pearl)
        self.window_data = window_data
        self.context_mode = context_mode
        self.context_stats = context_stats or {}
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "patient_status": self.patient_status,
            "action_evaluations": self.action_evaluations,
            "overall_window_summary": self.overall_window_summary,
            "doctor_actions": self.doctor_actions,
            "clinical_quality": self.clinical_quality,
            "primary_clinical_driver": self.primary_clinical_driver,
            "clinical_pearl": self.clinical_pearl,
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
        patient_status = _normalize_patient_status(data.get("patient_status"))
        action_evaluations = _normalize_action_evaluations(data.get("action_evaluations"))
        doctor_actions = _normalize_doctor_actions(data.get("doctor_actions"))
        if not doctor_actions and action_evaluations:
            doctor_actions = _derive_doctor_actions_from_action_evaluations(action_evaluations)

        overall_window_summary = _normalize_overall_window_summary(data.get("overall_window_summary"))
        if not overall_window_summary:
            overall_window_summary = _safe_text(patient_status.get("summary"))

        clinical_quality = _normalize_clinical_quality(
            data.get("clinical_quality"),
            action_evaluations=action_evaluations,
            overall_window_summary=overall_window_summary,
        )

        primary_driver = _safe_text(data.get("primary_clinical_driver"))
        if not primary_driver:
            audit_metadata = data.get("audit_metadata")
            if isinstance(audit_metadata, dict):
                primary_driver = _safe_text(audit_metadata.get("primary_clinical_driver"))

        clinical_pearl = _normalize_clinical_pearl(data.get("clinical_pearl"))
        return cls(
            patient_status=patient_status,
            doctor_actions=doctor_actions,
            action_evaluations=action_evaluations,
            overall_window_summary=overall_window_summary,
            clinical_quality=clinical_quality,
            primary_clinical_driver=primary_driver,
            clinical_pearl=clinical_pearl,
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
        temperature: float = 0.3,
        max_tokens: int = 4096,
        log_dir: Optional[str] = None,
        blinded: bool = False,
        use_discharge_summary: bool = False,
        history_context_hours: float = 48.0,
        future_context_hours: float = 48.0,
        top_k_recommendations: int = 3,
    ):
        self.llm_client = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.blinded = blinded
        self.use_discharge_summary = bool(use_discharge_summary)
        self.history_context_hours = float(history_context_hours)
        self.future_context_hours = float(future_context_hours)
        try:
            self.top_k_recommendations = max(1, int(top_k_recommendations))
        except (TypeError, ValueError):
            self.top_k_recommendations = 3
        if self.history_context_hours < 0 or self.future_context_hours < 0:
            raise ValueError(
                "history_context_hours and future_context_hours must be >= 0, got: "
                f"{self.history_context_hours}, {self.future_context_hours}"
            )

        self.evaluation_count = 0
        self.total_tokens_used = 0
        self.total_llm_calls = 0

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
            f"history_hours={self.history_context_hours}, "
            f"future_hours={self.future_context_hours}, "
            f"top_k_recommendations={self.top_k_recommendations}"
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
        mode = "blinded" if self.blinded else "unblinded"
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
            "parsed_response": response.get("parsed") if isinstance(response, dict) else None,
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

    def _prepare_trajectory_context(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        raw_events, enter_time, leave_time = _extract_icu_events(trajectory)
        return {
            "raw_events": raw_events,
            "enter_time": enter_time,
            "leave_time": leave_time,
        }

    def prepare_context(
        self,
        trajectory: Dict[str, Any],
        window_data: Dict[str, Any],
        prepared_trajectory_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build local ICU context for one window using configured history/future bounds."""
        prepared = prepared_trajectory_context or self._prepare_trajectory_context(trajectory)
        raw_events = prepared.get("raw_events") if isinstance(prepared, dict) else None
        if not isinstance(raw_events, list):
            raw_events = []
        enter_time = prepared.get("enter_time") if isinstance(prepared, dict) else None
        leave_time = prepared.get("leave_time") if isinstance(prepared, dict) else None

        current_window_start = _parse_time(window_data.get("current_window_start"))
        current_window_end = _parse_time(window_data.get("current_window_end"))
        if (
            current_window_start is not None
            and current_window_end is not None
            and current_window_end < current_window_start
        ):
            current_window_end = current_window_start

        anchor_time = current_window_start
        if anchor_time is None:
            anchor_time = current_window_end
        if anchor_time is None:
            anchor_time = enter_time

        context_start: Optional[datetime]
        context_end: Optional[datetime]
        if anchor_time is None:
            context_start = enter_time
            context_end = leave_time
        else:
            context_start = anchor_time - timedelta(hours=self.history_context_hours)
            context_end = anchor_time + timedelta(hours=self.future_context_hours)

        if enter_time and (context_start is None or context_start < enter_time):
            context_start = enter_time
        if leave_time and (context_end is None or context_end > leave_time):
            context_end = leave_time
        if context_start and context_end and context_start > context_end:
            context_start = context_end

        context_events_all = _slice_events_by_time(raw_events, context_start, context_end)
        if self.use_discharge_summary:
            # Include all ICU discharge summaries from the full ICU trajectory, independent
            # from bounded history/future event slicing.
            discharge_summary_scope_events = raw_events
            discharge_summary_scope_start = enter_time
            discharge_summary_scope_end = leave_time
        else:
            discharge_summary_scope_events = context_events_all
            discharge_summary_scope_start = context_start
            discharge_summary_scope_end = context_end

        icu_discharge_summaries = _extract_icu_discharge_summaries_from_events(discharge_summary_scope_events)
        # NOTE_DISCHARGESUMMARY must not appear in raw ICU events context.
        context_events = _exclude_discharge_summary_events(context_events_all)
        history_events, current_window_events, future_events = _partition_events_relative_to_window(
            context_events,
            current_window_start=current_window_start,
            current_window_end=current_window_end,
        )

        events_context_text = _build_raw_context_text(
            history_events=history_events,
            current_window_events=current_window_events,
            future_events=future_events,
            context_start=context_start,
            context_end=context_end,
            current_window_start=current_window_start,
            current_window_end=current_window_end,
            anchor_time=anchor_time,
            history_hours=self.history_context_hours,
            future_hours=self.future_context_hours,
        )

        discharge_summary_context_text = _build_icu_discharge_summary_context_text(
            icu_discharge_summaries,
            context_start=discharge_summary_scope_start,
            context_end=discharge_summary_scope_end,
        )

        if self.use_discharge_summary:
            context_text = "\n\n".join([discharge_summary_context_text, events_context_text]).strip()
            context_mode = "raw_local_trajectory_with_icu_discharge_summary"
        else:
            context_text = events_context_text
            context_mode = "raw_local_trajectory_icu_events_only"

        context_tokens = self._estimate_tokens(context_text)
        return {
            "mode": context_mode,
            "context_text": context_text,
            "context_tokens": context_tokens,
            "context_event_count": len(context_events),
            "context_history_event_count": len(history_events),
            "context_current_window_event_count": len(current_window_events),
            "context_future_event_count": len(future_events),
            "use_discharge_summary": self.use_discharge_summary,
            "context_anchor_time": anchor_time.isoformat() if anchor_time else None,
            "context_window_start": context_start.isoformat() if context_start else None,
            "context_window_end": context_end.isoformat() if context_end else None,
            "history_hours": self.history_context_hours,
            "future_hours": self.future_context_hours,
            "has_icu_discharge_summary": len(icu_discharge_summaries) > 0,
            "icu_discharge_summary_count": len(icu_discharge_summaries),
        }

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
        trajectory: Dict[str, Any],
        prepared_trajectory_context: Optional[Dict[str, Any]] = None,
    ) -> OracleReport:
        context_info = self.prepare_context(
            trajectory,
            window_data,
            prepared_trajectory_context=prepared_trajectory_context,
        )
        prompt = format_oracle_prompt(
            window_data=window_data,
            context_block=context_info.get("context_text", ""),
            context_mode=context_info.get("mode", "raw_local_trajectory_icu_events_only"),
            history_hours=context_info.get("history_hours"),
            future_hours=context_info.get("future_hours"),
            top_k=self.top_k_recommendations,
        )

        response: Dict[str, Any] = {}
        call_logged = False
        evaluation_number: Optional[int] = None
        try:
            response = self.llm_client.chat(prompt=prompt, response_format="json")

            self._record_llm_call(
                step_type="oracle_evaluator",
                subject_id=window_data.get("subject_id"),
                icu_stay_id=window_data.get("icu_stay_id"),
                prompt=prompt,
                response=response,
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
                    "top_k_recommendations": self.top_k_recommendations,
                    "pre_icu_history_source": window_data.get("pre_icu_history_source"),
                    "pre_icu_history_items": window_data.get("pre_icu_history_items"),
                },
            )
            call_logged = True

            with self._stats_lock:
                self.evaluation_count += 1
                self.total_tokens_used += _usage_tokens(response.get("usage", {}))
                evaluation_number = self.evaluation_count

            parsed = response.get("parsed")
            if parsed is None:
                parsed = _best_effort_parse_json(response.get("content", ""))

            report = OracleReport.from_dict(
                data=parsed if isinstance(parsed, dict) else {},
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
                    "has_icu_discharge_summary": context_info.get("has_icu_discharge_summary", False),
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
            if not call_logged:
                self._record_llm_call(
                    step_type="oracle_evaluator",
                    subject_id=window_data.get("subject_id"),
                    icu_stay_id=window_data.get("icu_stay_id"),
                    prompt=prompt,
                    response=response if isinstance(response, dict) else {},
                    window_data=window_data,
                    metadata={
                        "window_start_time": window_data.get("current_window_start"),
                        "window_end_time": window_data.get("current_window_end"),
                        "context_mode": context_info.get("mode"),
                        "use_discharge_summary": context_info.get("use_discharge_summary", self.use_discharge_summary),
                        "pre_icu_history_source": window_data.get("pre_icu_history_source"),
                        "pre_icu_history_items": window_data.get("pre_icu_history_items"),
                        "error": str(e),
                    },
                )

            report = OracleReport(
                patient_status=_normalize_patient_status({}),
                doctor_actions=[],
                window_data=window_data,
                context_mode=context_info.get("mode", "raw_local_trajectory_icu_events_only"),
                context_stats={
                    "context_tokens": context_info.get("context_tokens"),
                    "context_event_count": context_info.get("context_event_count"),
                    "context_history_event_count": context_info.get("context_history_event_count"),
                    "context_current_window_event_count": context_info.get("context_current_window_event_count"),
                    "context_future_event_count": context_info.get("context_future_event_count"),
                    "use_discharge_summary": context_info.get("use_discharge_summary", self.use_discharge_summary),
                    "context_anchor_time": context_info.get("context_anchor_time"),
                    "context_window_start": context_info.get("context_window_start"),
                    "context_window_end": context_info.get("context_window_end"),
                },
                error=f"Oracle evaluation error: {e}",
            )
            self._save_log(
                window_data,
                prompt,
                {"content": None, "error": str(e)},
                report,
                context_info=context_info,
                error=str(e),
                evaluation_number=evaluation_number,
            )
            return report

    def evaluate_trajectory(self, windows: List[Dict[str, Any]], trajectory: Dict[str, Any]) -> List[OracleReport]:
        reports: List[OracleReport] = []
        prepared_trajectory_context = self._prepare_trajectory_context(trajectory)

        for i, window in enumerate(windows):
            window_payload = dict(window)
            window_payload.setdefault("window_index", i)
            print(f"Evaluating window {i+1}/{len(windows)} (Hour {window_payload['hours_since_admission']:.1f})...")
            report = self.evaluate_window(
                window_payload,
                trajectory=trajectory,
                prepared_trajectory_context=prepared_trajectory_context,
            )
            reports.append(report)

        return reports

    def evaluate_trajectory_parallel(
        self,
        windows: List[Dict[str, Any]],
        trajectory: Dict[str, Any],
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

        prepared_trajectory_context = self._prepare_trajectory_context(trajectory)
        windows_with_index = []
        for i, window in enumerate(windows):
            window_payload = dict(window)
            window_payload.setdefault("window_index", i)
            windows_with_index.append(window_payload)

        results: List[Optional[OracleReport]] = [None] * len(windows)
        completed_count = 0

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(self.evaluate_window, window, trajectory, prepared_trajectory_context): i
                for i, window in enumerate(windows_with_index)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                window = windows_with_index[index]

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
                    results[index] = OracleReport(
                        patient_status=_normalize_patient_status({}),
                        doctor_actions=[],
                        window_data=window,
                        context_mode="raw_local_trajectory_icu_events_only",
                        context_stats={
                            "use_discharge_summary": self.use_discharge_summary,
                        },
                        error=f"Parallel evaluation error: {e}",
                    )

        if show_progress:
            print(f"Parallel evaluation complete: {completed_count}/{len(windows)} windows evaluated")

        reports: List[OracleReport] = []
        for index, report in enumerate(results):
            if report is not None:
                reports.append(report)
                continue

            window = windows_with_index[index]
            reports.append(
                OracleReport(
                    patient_status=_normalize_patient_status({}),
                    doctor_actions=[],
                    window_data=window,
                    context_mode="raw_local_trajectory_icu_events_only",
                    context_stats={
                        "use_discharge_summary": self.use_discharge_summary,
                    },
                    error="Parallel evaluation produced no result for window.",
                )
            )

        return reports

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_evaluations": self.evaluation_count,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens_used": self.total_tokens_used,
            "avg_tokens_per_evaluation": (
                self.total_tokens_used / self.evaluation_count if self.evaluation_count > 0 else 0
            ),
            "use_discharge_summary": self.use_discharge_summary,
            "history_context_hours": self.history_context_hours,
            "future_context_hours": self.future_context_hours,
            "top_k_recommendations": self.top_k_recommendations,
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


def _extract_icu_events(
    trajectory: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Optional[datetime], Optional[datetime]]:
    enter_time = _parse_time(trajectory.get("enter_time"))
    leave_time = _parse_time(trajectory.get("leave_time"))
    events = trajectory.get("events", [])

    normalized: List[Dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue

        event_time = _parse_time(event.get("time") or event.get("start_time"))
        if enter_time and event_time and event_time < enter_time:
            continue
        if leave_time and event_time and event_time > leave_time:
            continue

        cleaned = {
            "time": event.get("time") or event.get("start_time"),
            "code": event.get("code"),
            "code_specifics": event.get("code_specifics"),
            "numeric_value": event.get("numeric_value"),
            "text_value": event.get("text_value"),
        }
        cleaned = {k: v for k, v in cleaned.items() if v is not None}
        if not cleaned:
            continue

        cleaned["_event_time"] = event_time
        normalized.append(cleaned)

    normalized.sort(key=lambda item: (item.get("_event_time") is None, item.get("_event_time") or datetime.max))
    return normalized, enter_time, leave_time


def _slice_events_by_time(
    raw_events: List[Dict[str, Any]],
    start: Optional[datetime],
    end: Optional[datetime],
) -> List[Dict[str, Any]]:
    if not raw_events:
        return []

    sliced: List[Dict[str, Any]] = []
    for event in raw_events:
        event_time = event.get("_event_time")
        if not isinstance(event_time, datetime):
            continue
        if start and event_time < start:
            continue
        if end and event_time > end:
            continue
        sliced.append(event)

    return sliced


def _exclude_discharge_summary_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for event in events:
        if _safe_text(event.get("code")) == "NOTE_DISCHARGESUMMARY":
            continue
        filtered.append(event)
    return filtered


def _partition_events_relative_to_window(
    events: List[Dict[str, Any]],
    *,
    current_window_start: Optional[datetime],
    current_window_end: Optional[datetime],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    history_events: List[Dict[str, Any]] = []
    current_window_events: List[Dict[str, Any]] = []
    future_events: List[Dict[str, Any]] = []

    for event in events:
        event_time = event.get("_event_time")
        if not isinstance(event_time, datetime):
            continue

        if current_window_start is not None and event_time < current_window_start:
            history_events.append(event)
            continue

        if current_window_end is not None and event_time > current_window_end:
            future_events.append(event)
            continue

        current_window_events.append(event)

    return history_events, current_window_events, future_events


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
    history_hours: float,
    future_hours: float,
) -> str:
    anchor_text = anchor_time.isoformat() if anchor_time else "unknown"
    start_text = context_start.isoformat() if context_start else "unknown"
    end_text = context_end.isoformat() if context_end else "unknown"
    current_start_text = current_window_start.isoformat() if current_window_start else "unknown"
    current_end_text = current_window_end.isoformat() if current_window_end else "unknown"
    total_context_events = len(history_events) + len(current_window_events) + len(future_events)

    lines = [
        "## ICU TRAJECTORY CONTEXT WINDOW",
        f"Anchor time (current_window_start): {anchor_text}",
        f"Context window: [{start_text}, {end_text}]",
        f"Current observation window: [{current_start_text}, {current_end_text}]",
        f"History threshold (hours): {history_hours:.1f}",
        f"Future threshold (hours): {future_hours:.1f}",
        f"Total ICU context events: {total_context_events}",
        "",
        "## HISTORY EVENTS OF CURRENT WINDOW",
        f"Total history events: {len(history_events)}",
    ]

    if history_events:
        for idx, event in enumerate(history_events, start=1):
            lines.append(f"HX{idx}. {format_event_line(event)}")
    else:
        lines.append("(No history events before current window)")

    lines.extend(
        [
            "",
            "## CURRENT OBSERVATION WINDOW FOR EVALUATION",
            f"Total trajectory events in current window: {len(current_window_events)}",
            "(Current-window event details are listed in the dedicated evaluation section below.)",
            "",
            "## FUTURE EVENTS",
            f"Total future events: {len(future_events)}",
        ]
    )

    if future_events:
        for idx, event in enumerate(future_events, start=1):
            lines.append(f"FX{idx}. {format_event_line(event)}")
    else:
        lines.append("(No future events after current window)")

    return "\n".join(lines)


def _extract_icu_discharge_summaries_from_events(context_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for event in context_events:
        if _safe_text(event.get("code")) != "NOTE_DISCHARGESUMMARY":
            continue
        summaries.append(
            {
                "time": event.get("time"),
                "code_specifics": event.get("code_specifics"),
                "text_value": event.get("text_value"),
                "_event_time": event.get("_event_time"),
            }
        )

    summaries.sort(key=lambda item: (item.get("_event_time") is None, item.get("_event_time") or datetime.max))
    for item in summaries:
        item.pop("_event_time", None)
    return summaries


def _build_icu_discharge_summary_context_text(
    discharge_summaries: List[Dict[str, Any]],
    *,
    context_start: Optional[datetime],
    context_end: Optional[datetime],
) -> str:
    start_text = context_start.isoformat() if context_start else "unknown"
    end_text = context_end.isoformat() if context_end else "unknown"
    lines = [
        "## ICU DISCHARGE SUMMARY",
        f"Discharge summary scope: [{start_text}, {end_text}]",
        f"Total ICU discharge summaries in scope: {len(discharge_summaries)}",
        "",
        "ICU discharge summaries:",
    ]

    if not discharge_summaries:
        lines.append("(No ICU discharge summary found in this context window)")
        return "\n".join(lines)

    for idx, item in enumerate(discharge_summaries, start=1):
        time_text = _safe_text(item.get("time")) or "unknown"
        lines.append(f"DS{idx}. [{time_text}] NOTE_DISCHARGESUMMARY")
        specifics = _safe_text(item.get("code_specifics"))
        if specifics:
            lines.append(f"Details: {specifics}")
        summary_text = _safe_text(item.get("text_value"))
        lines.append(summary_text if summary_text else "(No discharge summary text)")
        lines.append("")

    return "\n".join(lines).strip()


def _normalize_domain_assessments(
    domains_value: Any,
    fallback_physiology: Any,
) -> Dict[str, Dict[str, Any]]:
    domains_data = domains_value if isinstance(domains_value, dict) else {}
    fallback = _normalize_physiology_trends(fallback_physiology)

    normalized: Dict[str, Dict[str, Any]] = {}
    for key in DOMAIN_KEYS:
        item = domains_data.get(key)
        if not isinstance(item, dict):
            item = {}

        label = _safe_text(item.get("label")).lower()
        if label not in VALID_STATUS:
            label = fallback.get(key, {}).get("status", "insufficient_data")
        if label not in VALID_STATUS:
            label = "insufficient_data"

        rationale = _safe_text(item.get("rationale"))
        if not rationale:
            rationale = fallback.get(key, {}).get("rationale", "")
        if not rationale:
            rationale = "No rationale provided."

        key_signals_raw = item.get("key_signals")
        key_signals: List[str] = []
        if isinstance(key_signals_raw, list):
            key_signals = [_safe_text(signal) for signal in key_signals_raw if _safe_text(signal)]

        normalized[key] = {
            "label": label,
            "key_signals": key_signals,
            "rationale": rationale,
        }

    return normalized


def _domains_to_physiology_trends(domains: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    normalized: Dict[str, Dict[str, str]] = {}
    for key in DOMAIN_KEYS:
        item = domains.get(key, {})
        status = _safe_text(item.get("label")).lower()
        if status not in VALID_STATUS:
            status = "insufficient_data"

        rationale = _safe_text(item.get("rationale"))
        if not rationale:
            rationale = "No rationale provided."

        normalized[key] = {
            "status": status,
            "rationale": rationale,
        }

    return normalized


def _normalize_patient_status(value: Any) -> Dict[str, Any]:
    data = value if isinstance(value, dict) else {}

    domains = _normalize_domain_assessments(data.get("domains"), data.get("physiology_trends"))
    physiology = _domains_to_physiology_trends(domains)

    overall_data = data.get("overall") if isinstance(data.get("overall"), dict) else {}
    overall = _safe_text(overall_data.get("label")).lower()
    if overall not in VALID_STATUS:
        overall = _safe_text(data.get("overall_status")).lower()
    if overall not in VALID_STATUS:
        overall = _infer_overall_from_domains(physiology)

    overall_rationale = _safe_text(overall_data.get("rationale"))
    summary = _safe_text(data.get("summary"))
    if overall_rationale and not summary:
        summary = overall_rationale
    if summary and not overall_rationale:
        overall_rationale = summary
    if not summary:
        summary = "Insufficient data to provide detailed status summary."
    if not overall_rationale:
        overall_rationale = summary

    return {
        "domains": domains,
        "overall": {
            "label": overall,
            "rationale": overall_rationale,
        },
        # Backward-compatible fields used by existing analytics scripts.
        "overall_status": overall,
        "physiology_trends": physiology,
        "summary": summary,
    }


def _normalize_physiology_trends(value: Any) -> Dict[str, Dict[str, str]]:
    data = value if isinstance(value, dict) else {}

    normalized: Dict[str, Dict[str, str]] = {}
    for key in DOMAIN_KEYS:
        item = data.get(key)
        if not isinstance(item, dict):
            item = {}

        status = _safe_text(item.get("status")).lower()
        if status not in VALID_STATUS:
            status = "insufficient_data"

        rationale = _safe_text(item.get("rationale"))
        if not rationale:
            rationale = "No rationale provided."

        normalized[key] = {
            "status": status,
            "rationale": rationale,
        }

    return normalized


def _infer_overall_from_domains(physiology: Dict[str, Dict[str, str]]) -> str:
    statuses = [item.get("status", "insufficient_data") for item in physiology.values()]
    if all(status == "insufficient_data" for status in statuses):
        return "insufficient_data"
    if "deteriorating" in statuses:
        return "deteriorating"
    if "fluctuating" in statuses:
        return "fluctuating"
    if "improving" in statuses and "stable" not in statuses:
        return "improving"
    return "stable"


def _normalize_doctor_actions(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        action = _safe_text(item.get("action"))
        if not action:
            continue

        category = _safe_text(item.get("category")).lower()
        if category not in VALID_ACTION_CATEGORY:
            category = "other"

        refs = item.get("evidence_event_refs") if isinstance(item.get("evidence_event_refs"), list) else []
        refs = [_safe_text(ref) for ref in refs if _safe_text(ref)]

        normalized.append(
            {
                "time": item.get("time"),
                "action": action,
                "category": category,
                "evidence_event_refs": refs,
            }
        )

    return normalized


def _normalize_action_evaluations(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue

        action_description = _safe_text(item.get("action_description"))
        if not action_description:
            action_description = _safe_text(item.get("action"))
        if not action_description:
            continue

        action_id = _safe_text(item.get("action_id")) or f"A{index}"

        guideline = item.get("guideline_adherence") if isinstance(item.get("guideline_adherence"), dict) else {}
        guideline_label = _safe_text(guideline.get("label")).lower()
        if guideline_label not in VALID_GUIDELINE_ADHERENCE:
            guideline_label = "guideline_unclear"
        guideline_reference = guideline.get("guideline_reference")
        if guideline_reference is not None:
            guideline_reference = _safe_text(guideline_reference) or None
        guideline_rationale = _safe_text(guideline.get("rationale"))
        if not guideline_rationale:
            guideline_rationale = "No guideline adherence rationale provided."

        contextual = (
            item.get("contextual_appropriateness") if isinstance(item.get("contextual_appropriateness"), dict) else {}
        )
        contextual_label = _safe_text(contextual.get("label")).lower()
        if contextual_label not in VALID_CONTEXTUAL_APPROPRIATENESS:
            contextual_label = "not_enough_context"
        contextual_rationale = _safe_text(contextual.get("rationale"))
        if not contextual_rationale:
            contextual_rationale = "No contextual appropriateness rationale provided."
        hindsight_caveat = _safe_text(contextual.get("hindsight_caveat")) or None

        normalized.append(
            {
                "action_id": action_id,
                "action_description": action_description,
                "guideline_adherence": {
                    "label": guideline_label,
                    "guideline_reference": guideline_reference,
                    "rationale": guideline_rationale,
                },
                "contextual_appropriateness": {
                    "label": contextual_label,
                    "rationale": contextual_rationale,
                    "hindsight_caveat": hindsight_caveat,
                },
            }
        )

    return normalized


def _normalize_overall_window_summary(value: Any) -> str:
    return _safe_text(value)


def _extract_cw_refs(text: Any) -> List[str]:
    candidate = _safe_text(text)
    if not candidate:
        return []

    refs: List[str] = []
    seen = set()
    for ref in re.findall(r"\bCW\d+\b", candidate):
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def _derive_doctor_actions_from_action_evaluations(action_evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    derived: List[Dict[str, Any]] = []
    for action in action_evaluations:
        action_description = _safe_text(action.get("action_description"))
        if not action_description:
            continue

        refs = _extract_cw_refs(action.get("action_id"))
        if not refs:
            refs = _extract_cw_refs(action_description)

        derived.append(
            {
                "time": None,
                "action": action_description,
                "category": "other",
                "evidence_event_refs": refs,
            }
        )

    return derived


def _infer_clinical_quality_from_actions(
    action_evaluations: Optional[List[Dict[str, Any]]] = None,
    overall_window_summary: str = "",
) -> Dict[str, str]:
    actions = action_evaluations if isinstance(action_evaluations, list) else []
    summary = _safe_text(overall_window_summary)

    if not actions:
        return {
            "rating": "neutral",
            "rationale": summary or "No clinical quality rationale provided.",
            "guideline_adherence": "not_specified",
        }

    guideline_labels = [
        _safe_text(action.get("guideline_adherence", {}).get("label")).lower()
        for action in actions
        if isinstance(action, dict)
    ]
    contextual_labels = [
        _safe_text(action.get("contextual_appropriateness", {}).get("label")).lower()
        for action in actions
        if isinstance(action, dict)
    ]

    has_non_adherent = "non_adherent" in guideline_labels
    has_potential_harm = "potentially_harmful" in contextual_labels
    all_context_acceptable = all(
        label in {"appropriate", "not_enough_context"} for label in contextual_labels if label
    ) and any(label == "appropriate" for label in contextual_labels)
    all_guideline_non_negative = all(
        label in {"adherent", "not_applicable", "guideline_unclear"} for label in guideline_labels if label
    )

    if has_non_adherent or has_potential_harm:
        rating = "sub_optimal"
    elif all_context_acceptable and all_guideline_non_negative:
        rating = "optimal"
    else:
        rating = "neutral"

    guideline_counts = {
        label: guideline_labels.count(label)
        for label in ["adherent", "non_adherent", "not_applicable", "guideline_unclear"]
    }
    guideline_adherence = ", ".join(f"{k}={v}" for k, v in guideline_counts.items())

    rationale = summary
    if not rationale:
        rationale = f"Quality inferred from {len(actions)} action evaluations."

    return {
        "rating": rating,
        "rationale": rationale,
        "guideline_adherence": guideline_adherence,
    }


def _normalize_clinical_quality(
    value: Any,
    action_evaluations: Optional[List[Dict[str, Any]]] = None,
    overall_window_summary: str = "",
) -> Dict[str, str]:
    data = value if isinstance(value, dict) else {}
    inferred = _infer_clinical_quality_from_actions(action_evaluations, overall_window_summary)

    rating = _safe_text(data.get("rating")).lower().replace("-", "_")
    if rating not in VALID_QUALITY_RATING:
        rating = inferred["rating"]

    rationale = _safe_text(data.get("rationale"))
    if not rationale:
        rationale = inferred["rationale"]
    if not rationale:
        rationale = "No clinical quality rationale provided."

    guideline_adherence = _safe_text(data.get("guideline_adherence"))
    if not guideline_adherence:
        guideline_adherence = inferred["guideline_adherence"]
    if not guideline_adherence:
        guideline_adherence = "not_specified"

    return {
        "rating": rating,
        "rationale": rationale,
        "guideline_adherence": guideline_adherence,
    }


def _normalize_primary_clinical_driver(value: Any) -> str:
    text = _safe_text(value)
    if not text:
        return "Unspecified primary clinical driver."
    return text


def _normalize_clinical_pearl(value: Any) -> str:
    text = _safe_text(value)
    if not text:
        return "No clinical pearl provided."
    return text


def _best_effort_parse_json(content: str) -> Dict[str, Any]:
    text = _safe_text(content)
    if not text:
        return {}

    response_blocks = re.findall(r"<response>\s*([\s\S]*?)\s*</response>", text, re.IGNORECASE)
    for block in response_blocks:
        block = block.strip()
        if not block:
            continue
        try:
            parsed = json.loads(block)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    for block in fenced_blocks:
        block = block.strip()
        if not block:
            continue
        try:
            parsed = json.loads(block)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[i:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return {}
