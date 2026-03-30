"""Tests for one-time pre-ICU history compression in MetaOracle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import agents.oracle as oracle_module


class FakeCompressionLLMClient:
    def __init__(self, provider="openai", model=None, api_key=None, temperature=0.3, max_tokens=4096):
        self.provider = provider
        self.model = model or "fake-model"

    def chat(self, prompt: str, response_format: str = "text", **kwargs):
        if "Pre-ICU history payload" in prompt:
            parsed = {
                "compressed_pre_icu_history": (
                    "Pre-ICU: prior discharge summary indicates recurrent CHF exacerbations; "
                    "radiology showed bilateral pulmonary edema; baseline creatinine mildly elevated."
                )
            }
            return {
                "content": json.dumps(parsed),
                "parsed": parsed,
                "usage": {"input_tokens": 40, "output_tokens": 12},
                "model": self.model,
            }

        parsed_payload = {
            "patient_status": {
                "domains": {
                    "hemodynamics": {"label": "stable", "key_evidence": ["1001"], "rationale": "stable"},
                    "respiratory": {"label": "stable", "key_evidence": ["1001"], "rationale": "stable"},
                    "renal_metabolic": {
                        "label": "insufficient_data",
                        "key_evidence": ["1001"],
                        "rationale": "limited",
                    },
                    "neurology": {"label": "stable", "key_evidence": ["1001"], "rationale": "stable"},
                },
                "overall": {"label": "stable", "rationale": "stable"},
            },
            "action_evaluations": [],
            "overall_window_summary": "stable",
        }
        return {
            "content": json.dumps(parsed_payload),
            "parsed": parsed_payload,
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "model": self.model,
        }


class FakeCompressionFallbackLLMClient:
    def __init__(self, provider="openai", model=None, api_key=None, temperature=0.3, max_tokens=4096):
        self.provider = provider
        self.model = model or "fake-model"

    def chat(self, prompt: str, response_format: str = "text", **kwargs):
        if "Pre-ICU history payload" in prompt:
            return {
                "content": "",
                "parsed": None,
                "usage": {"input_tokens": 8, "output_tokens": 1},
                "model": self.model,
            }
        return {
            "content": "{}",
            "parsed": {},
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "model": self.model,
        }


def _build_windows() -> list[dict]:
    pre_icu_history = {
        "source": "reports",
        "items": 2,
        "content": (
            "--- Report 1: Discharge Summary ---\n"
            "Prior discharge: recurrent CHF exacerbation with volume overload.\n"
            "--- Report 2: NOTE_RADIOLOGYREPORT ---\n"
            "CXR: bilateral interstitial opacities concerning for pulmonary edema."
        ),
        "history_hours": 72.0,
        "fallback_hours": 72.0,
        "baseline_content": "B1. Creatinine 1.6 mg/dL, B2. Lactate 2.2 mmol/L",
        "baseline_events_count": 2,
    }
    return [
        {
            "subject_id": 1,
            "icu_stay_id": 10,
            "current_window_start": "2024-01-01T00:00:00",
            "current_window_end": "2024-01-01T00:30:00",
            "hours_since_admission": 0.0,
            "pre_icu_history_source": "reports",
            "pre_icu_history_items": 2,
            "pre_icu_history": dict(pre_icu_history),
        },
        {
            "subject_id": 1,
            "icu_stay_id": 10,
            "current_window_start": "2024-01-01T00:30:00",
            "current_window_end": "2024-01-01T01:00:00",
            "hours_since_admission": 0.5,
            "pre_icu_history_source": "reports",
            "pre_icu_history_items": 2,
            "pre_icu_history": dict(pre_icu_history),
        },
    ]


def test_pre_icu_history_is_compressed_once_and_reused(monkeypatch):
    monkeypatch.setattr(oracle_module, "LLMClient", FakeCompressionLLMClient)
    oracle = oracle_module.MetaOracle(
        provider="openai",
        model="fake-model",
        compress_pre_icu_history=True,
    )
    windows = _build_windows()

    compression = oracle.compress_pre_icu_history_for_windows(windows)

    assert compression is not None
    assert compression.get("applied_to_windows") == 2
    assert compression.get("compressed_chars", 0) > 0

    for window in windows:
        assert window["pre_icu_history_source"] == "llm_compressed"
        assert window["pre_icu_history"]["source"] == "llm_compressed"
        assert window["pre_icu_history"]["baseline_content"] == ""
        assert "compression" in window["pre_icu_history"]

    llm_calls = oracle.pop_patient_llm_call_logs(subject_id=1, icu_stay_id=10)
    assert len(llm_calls) == 1
    assert llm_calls[0]["step_type"] == "oracle_pre_icu_history_compressor"
    assert llm_calls[0]["metadata"].get("compressed_chars", 0) > 0

    stats = oracle.get_statistics()
    assert stats["total_pre_icu_compression_calls"] == 1
    assert stats["total_pre_icu_compression_tokens"] > 0


def test_pre_icu_compressor_logs_fallback_output(monkeypatch):
    monkeypatch.setattr(oracle_module, "LLMClient", FakeCompressionFallbackLLMClient)
    oracle = oracle_module.MetaOracle(
        provider="openai",
        model="fake-model",
        compress_pre_icu_history=True,
    )
    windows = _build_windows()

    compression = oracle.compress_pre_icu_history_for_windows(windows)

    assert compression is not None
    assert compression.get("parse_source") in {"best_effort_json", "heuristic_fallback", "error_fallback"}
    llm_calls = oracle.pop_patient_llm_call_logs(subject_id=1, icu_stay_id=10)
    assert len(llm_calls) == 1
    logged_call = llm_calls[0]
    assert logged_call.get("step_type") == "oracle_pre_icu_history_compressor"
    parsed_response = logged_call.get("parsed_response") or {}
    assert isinstance(parsed_response, dict)
    assert parsed_response.get("compressed_pre_icu_history")
    assert "compressed_pre_icu_history" in str(logged_call.get("response") or "")
    assert (logged_call.get("metadata") or {}).get("compressed_pre_icu_history")
