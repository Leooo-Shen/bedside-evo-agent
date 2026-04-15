from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.llms import LLMClient
from utils.json_parse import parse_json_dict_best_effort

DEFAULT_LLM_CALLS_PATH = (
    "oracle_results/oracle_pred_100_merge_v5_60_sparse/patients/10222587_38184652/llm_calls.json"
)

_THREAD_LOCAL = threading.local()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test one prompt from llm_calls.json using LLMClient.chat")
    parser.add_argument("--llm-calls-path", type=str, default=DEFAULT_LLM_CALLS_PATH)
    parser.add_argument("--window-index", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--step-type", type=str, default="oracle_evaluator")
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--response-format", type=str, choices=["text", "json"], default="text")
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--response-text-max-chars", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _select_call(payload: Dict[str, Any], *, step_type: str, window_index: int) -> Dict[str, Any]:
    calls = payload.get("calls")
    if not isinstance(calls, list):
        raise ValueError("Missing calls list in llm_calls payload")
    for call in calls:
        if not isinstance(call, dict):
            continue
        if str(call.get("step_type")) != step_type:
            continue
        try:
            idx = int(call.get("window_index"))
        except (TypeError, ValueError):
            continue
        if idx == int(window_index):
            return call
    raise ValueError(f"No call found for step_type={step_type} window_index={window_index}")


def _get_thread_client(
    *,
    provider: str,
    model: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
    timeout_seconds: Optional[float],
) -> LLMClient:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        init_kwargs: Dict[str, Any] = {
            "provider": provider,
            "model": model,
        }
        if temperature is not None:
            init_kwargs["temperature"] = float(temperature)
        if max_tokens is not None:
            init_kwargs["max_tokens"] = int(max_tokens)
        if timeout_seconds is not None:
            init_kwargs["request_timeout_seconds"] = float(timeout_seconds)
        client = LLMClient(**init_kwargs)
        _THREAD_LOCAL.client = client
    return client


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _worker(
    *,
    request_id: int,
    prompt: str,
    response_format: str,
    provider: str,
    model: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
    timeout_seconds: Optional[float],
    response_text_max_chars: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        client = _get_thread_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        response = client.chat(prompt=prompt, response_format=response_format)
        elapsed = time.perf_counter() - started
        content = str(response.get("content", "") or "")
        if response_text_max_chars > 0:
            content_for_log = content[: int(response_text_max_chars)]
        else:
            content_for_log = content
        usage = response.get("usage", {})
        parsed_best_effort = parse_json_dict_best_effort(content)
        result = {
            "request_id": int(request_id),
            "ok": True,
            "latency_seconds": elapsed,
            "response_chars": len(content),
            "input_tokens": _safe_int(usage.get("input_tokens") if isinstance(usage, dict) else 0),
            "output_tokens": _safe_int(usage.get("output_tokens") if isinstance(usage, dict) else 0),
            "ends_with_fence": content.endswith("```"),
            "json_parseable_dict": isinstance(parsed_best_effort, dict),
            "error": None,
        }
        result["response_text"] = content_for_log
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "request_id": int(request_id),
            "ok": False,
            "latency_seconds": elapsed,
            "response_chars": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "ends_with_fence": False,
            "json_parseable_dict": False,
            "response_text": "",
            "error": str(exc),
        }


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction)


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    ok_results = [row for row in results if bool(row.get("ok"))]
    err_results = [row for row in results if not bool(row.get("ok"))]
    latencies = [float(row.get("latency_seconds", 0.0)) for row in results]
    output_tokens = [int(row.get("output_tokens", 0)) for row in ok_results]
    input_tokens = [int(row.get("input_tokens", 0)) for row in ok_results]
    parseable = sum(1 for row in ok_results if bool(row.get("json_parseable_dict")))
    fenced = sum(1 for row in ok_results if bool(row.get("ends_with_fence")))
    return {
        "total_requests": total,
        "success_requests": len(ok_results),
        "error_requests": len(err_results),
        "success_rate": float(len(ok_results) / total) if total > 0 else 0.0,
        "latency_avg_seconds": float(statistics.fmean(latencies)) if latencies else 0.0,
        "latency_p50_seconds": _percentile(latencies, 0.50),
        "latency_p95_seconds": _percentile(latencies, 0.95),
        "latency_max_seconds": max(latencies) if latencies else 0.0,
        "input_tokens_avg": float(statistics.fmean(input_tokens)) if input_tokens else 0.0,
        "output_tokens_avg": float(statistics.fmean(output_tokens)) if output_tokens else 0.0,
        "json_parseable_count": int(parseable),
        "json_parseable_rate": float(parseable / len(ok_results)) if ok_results else 0.0,
        "ends_with_fence_count": int(fenced),
        "ends_with_fence_rate": float(fenced / len(ok_results)) if ok_results else 0.0,
    }


def main() -> None:
    args = _parse_args()
    if args.requests < 1:
        raise ValueError("--requests must be >= 1")
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")

    llm_calls_path = Path(args.llm_calls_path).expanduser().resolve()
    payload = _load_json(llm_calls_path)
    selected_call = _select_call(payload, step_type=str(args.step_type), window_index=int(args.window_index))
    prompt = str(selected_call.get("prompt", "") or "")
    if not prompt:
        raise ValueError("Selected call has empty prompt")

    provider = str(args.provider or payload.get("llm_provider") or "").strip()
    model = str(args.model or payload.get("llm_model") or "").strip()
    if not provider:
        raise ValueError("Provider is required. Pass --provider or ensure llm_calls.json has llm_provider")
    if not model:
        raise ValueError("Model is required. Pass --model or ensure llm_calls.json has llm_model")

    started = time.perf_counter()
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(args.threads)) as executor:
        futures = [
            executor.submit(
                _worker,
                request_id=i,
                prompt=prompt,
                response_format=str(args.response_format),
                provider=provider,
                model=model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds,
                response_text_max_chars=int(args.response_text_max_chars),
            )
            for i in range(int(args.requests))
        ]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            request_id = int(row.get("request_id", -1))
            ok = bool(row.get("ok"))
            print(f"\n===== RESPONSE {request_id} | ok={ok} =====")
            if ok:
                print(str(row.get("response_text", "")))
            else:
                print(f"ERROR: {row.get('error')}")
            print(f"===== END RESPONSE {request_id} =====")
    total_elapsed = time.perf_counter() - started

    summary = _summarize(results)
    report = {
        "generated_at": datetime.now().isoformat(),
        "llm_calls_path": str(llm_calls_path),
        "selected_step_type": str(args.step_type),
        "selected_window_index": int(args.window_index),
        "provider": provider,
        "model": model,
        "temperature_override": args.temperature,
        "max_tokens_override": args.max_tokens,
        "response_format": str(args.response_format),
        "response_text_max_chars": int(args.response_text_max_chars),
        "threads": int(args.threads),
        "requests": int(args.requests),
        "prompt_chars": len(prompt),
        "wall_time_seconds": total_elapsed,
        "summary": summary,
        "results": sorted(results, key=lambda row: int(row.get("request_id", 0))),
    }

    print(
        "Stress test done | "
        f"provider={provider} model={model} "
        f"window_index={int(args.window_index)} requests={int(args.requests)} threads={int(args.threads)}"
    )
    print(
        "Summary | "
        f"success={summary['success_requests']}/{summary['total_requests']} "
        f"success_rate={summary['success_rate']:.3f} "
        f"lat_avg={summary['latency_avg_seconds']:.2f}s "
        f"lat_p95={summary['latency_p95_seconds']:.2f}s "
        f"out_tok_avg={summary['output_tokens_avg']:.1f}"
    )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
