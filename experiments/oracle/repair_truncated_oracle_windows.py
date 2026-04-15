from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.llms import LLMClient
from utils.json_parse import parse_json_dict_best_effort
from utils.llm_log_viewer import save_llm_calls_html

_THREAD_LOCAL = threading.local()


@dataclass
class RepairTask:
    patient_id: str
    window_index: int
    call_index: int
    prompt: str


@dataclass
class RepairResult:
    patient_id: str
    window_index: int
    call_index: int
    success: bool
    attempts: int
    error: str
    response_text: str
    parsed_response: Optional[Dict[str, Any]]
    input_tokens: int
    output_tokens: int
    response_model: str


@dataclass
class PatientBundle:
    patient_dir: Path
    llm_payload: Dict[str, Any]
    predictions_payload: Dict[str, Any]
    llm_calls: List[Dict[str, Any]]
    window_outputs: List[Dict[str, Any]]
    window_output_pos_by_index: Dict[int, int]


@dataclass
class StrideFilterStats:
    patients_processed: int
    windows_before: int
    windows_after: int
    calls_before: int
    calls_after: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair truncated Oracle windows by re-running prompts from llm_calls.json."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Oracle run directory containing patients/*/llm_calls.json and oracle_predictions.json",
    )
    parser.add_argument(
        "--output-run-dir",
        type=str,
        default=None,
        help="Optional output run directory. When set, source run is copied here and repaired in the new location.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.json",
        help="Config JSON path used to resolve source oracle_time_windows.window_step_hours.",
    )
    parser.add_argument(
        "--target-window-step-hours",
        type=float,
        default=None,
        help="Target oracle window_step_hours to keep (e.g., 2.0).",
    )
    parser.add_argument(
        "--source-window-step-hours",
        type=float,
        default=None,
        help="Source oracle window_step_hours. If omitted, read from --config.",
    )
    parser.add_argument(
        "--stride-offset",
        type=int,
        default=0,
        help="Offset used by stride filtering: keep windows where window_index %% stride_multiple == stride_offset.",
    )
    parser.add_argument(
        "--stride-hours",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--provider", type=str, default="google")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now().isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_complete_oracle_output(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    required = ("patient_assessment", "action_review")
    for key in required:
        if key not in payload:
            return False
    patient_assessment = payload.get("patient_assessment")
    if not isinstance(patient_assessment, dict):
        return False
    overall = patient_assessment.get("overall")
    if not isinstance(overall, dict):
        return False
    label = overall.get("label")
    rationale = overall.get("rationale")
    if not isinstance(label, str) or not label.strip():
        return False
    if not isinstance(rationale, str) or not rationale.strip():
        return False
    action_review = payload.get("action_review")
    if not isinstance(action_review, dict):
        return False
    if not isinstance(action_review.get("evaluations"), list):
        return False
    if not isinstance(action_review.get("red_flags"), list):
        return False
    return True


def _load_json_dict(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level object in {path}")
    return payload


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _resolve_run_dirs(args: argparse.Namespace) -> Tuple[Path, Path]:
    source_run_dir = Path(args.run_dir).expanduser().resolve()
    output_run_dir = Path(args.output_run_dir).expanduser().resolve() if args.output_run_dir else source_run_dir

    target_window_step_hours = _resolve_target_window_step_hours(args)
    if target_window_step_hours is not None and source_run_dir == output_run_dir:
        raise ValueError(
            "--target-window-step-hours requires --output-run-dir different from --run-dir to avoid mutating source run."
        )
    return source_run_dir, output_run_dir


def _prepare_output_run_dir(source_run_dir: Path, output_run_dir: Path, dry_run: bool) -> None:
    if source_run_dir == output_run_dir:
        return
    if output_run_dir.exists():
        return
    if dry_run:
        raise ValueError(
            f"Output run directory does not exist in dry-run mode: {output_run_dir}. "
            "Create it by running once without --dry-run."
        )
    output_run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_run_dir, output_run_dir)


def _resolve_target_window_step_hours(args: argparse.Namespace) -> Optional[float]:
    if args.target_window_step_hours is not None:
        return float(args.target_window_step_hours)
    if args.stride_hours is not None:
        return float(args.stride_hours)
    return None


def _resolve_source_window_step_hours(args: argparse.Namespace) -> float:
    if args.source_window_step_hours is not None:
        source = float(args.source_window_step_hours)
        if source <= 0:
            raise ValueError("--source-window-step-hours must be > 0")
        return source

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config_payload = json.load(f)
    if not isinstance(config_payload, dict):
        raise ValueError(f"Invalid config format: {config_path}")
    time_window_config = config_payload.get("oracle_time_windows")
    if not isinstance(time_window_config, dict):
        raise ValueError(
            "Missing config key oracle_time_windows.window_step_hours. "
            "Pass --source-window-step-hours explicitly."
        )
    value = time_window_config.get("window_step_hours")
    try:
        source = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Could not resolve source oracle_time_windows.window_step_hours from config. "
            "Pass --source-window-step-hours explicitly."
        ) from exc
    if source <= 0:
        raise ValueError(f"Invalid oracle_time_windows.window_step_hours in config: {source}")
    return source


def _compute_stride_multiple(target_window_step_hours: float, source_window_step_hours: float) -> int:
    if target_window_step_hours <= 0:
        raise ValueError("--target-window-step-hours must be > 0")
    if source_window_step_hours <= 0:
        raise ValueError("--source-window-step-hours must be > 0")
    ratio = float(target_window_step_hours) / float(source_window_step_hours)
    rounded = int(round(ratio))
    if rounded < 1 or abs(ratio - rounded) > 1e-9:
        raise ValueError(
            "target_window_step_hours must be an integer multiple of source_window_step_hours, "
            f"got target_window_step_hours={target_window_step_hours}, "
            f"source_window_step_hours={source_window_step_hours}"
        )
    return rounded


def _select_stride_windows(
    window_outputs: List[Any],
    *,
    stride_multiple: int,
    stride_offset: int,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    selected: List[Tuple[int, int, Dict[str, Any]]] = []
    for pos, item in enumerate(window_outputs):
        if not isinstance(item, dict):
            continue
        window_index = _safe_int(item.get("window_index"), default=pos)
        if window_index < 0:
            continue
        if window_index % stride_multiple != stride_offset:
            continue
        selected.append((pos, window_index, item))
    return selected


def _filter_window_context_payload(
    payload: Dict[str, Any],
    *,
    old_to_new: Dict[int, int],
) -> Dict[str, Any]:
    contexts = payload.get("window_contexts")
    if not isinstance(contexts, list):
        return payload
    filtered: List[Dict[str, Any]] = []
    for pos, item in enumerate(contexts):
        if not isinstance(item, dict):
            continue
        old_idx = _safe_int(item.get("window_index"), default=pos)
        if old_idx not in old_to_new:
            continue
        new_item = copy.deepcopy(item)
        new_item["window_index"] = int(old_to_new[old_idx])
        new_item["stride_source_window_index"] = int(old_idx)
        filtered.append(new_item)
    payload["window_contexts"] = filtered
    return payload


def _apply_stride_filter_to_run(
    *,
    source_run_dir: Path,
    output_run_dir: Path,
    target_window_step_hours: float,
    source_window_step_hours: float,
    stride_offset: int,
    dry_run: bool,
) -> StrideFilterStats:
    manifest_path = output_run_dir / "stride_filter_manifest.json"
    stride_multiple = _compute_stride_multiple(target_window_step_hours, source_window_step_hours)
    normalized_offset = int(stride_offset) % int(stride_multiple)
    expected_manifest = {
        "target_window_step_hours": float(target_window_step_hours),
        "source_window_step_hours": float(source_window_step_hours),
        "stride_multiple": int(stride_multiple),
        "stride_offset": int(normalized_offset),
        "source_run_dir": str(source_run_dir),
    }

    if manifest_path.exists():
        existing = _load_json_dict(manifest_path)
        mismatches = []
        for key, value in expected_manifest.items():
            if existing.get(key) != value:
                mismatches.append((key, existing.get(key), value))
        if mismatches:
            details = ", ".join([f"{k}: existing={old!r} expected={new!r}" for k, old, new in mismatches])
            raise ValueError(f"Existing stride_filter_manifest.json does not match requested config: {details}")
        return StrideFilterStats(
            patients_processed=_safe_int(existing.get("patients_processed"), default=0),
            windows_before=_safe_int(existing.get("windows_before"), default=0),
            windows_after=_safe_int(existing.get("windows_after"), default=0),
            calls_before=_safe_int(existing.get("calls_before"), default=0),
            calls_after=_safe_int(existing.get("calls_after"), default=0),
        )

    if dry_run:
        raise ValueError(
            f"Stride filter manifest not found in dry-run mode: {manifest_path}. "
            "Run once without --dry-run to materialize filtered run."
        )

    patients_dir = output_run_dir / "patients"
    if not patients_dir.exists():
        raise FileNotFoundError(f"Missing patients directory: {patients_dir}")

    stats = StrideFilterStats(
        patients_processed=0,
        windows_before=0,
        windows_after=0,
        calls_before=0,
        calls_after=0,
    )

    for patient_dir in sorted(patients_dir.iterdir(), key=lambda p: p.name):
        if not patient_dir.is_dir():
            continue

        pred_path = patient_dir / "oracle_predictions.json"
        llm_path = patient_dir / "llm_calls.json"
        if not pred_path.exists() or not llm_path.exists():
            continue

        predictions_payload = _load_json_dict(pred_path)
        llm_payload = _load_json_dict(llm_path)

        window_outputs = predictions_payload.get("window_outputs")
        calls = llm_payload.get("calls")
        if not isinstance(window_outputs, list) or not isinstance(calls, list):
            continue

        selected = _select_stride_windows(
            window_outputs,
            stride_multiple=int(stride_multiple),
            stride_offset=int(normalized_offset),
        )
        old_to_new = {old_idx: new_idx for new_idx, (_, old_idx, _) in enumerate(selected)}

        filtered_window_outputs: List[Dict[str, Any]] = []
        for new_idx, (_, old_idx, row) in enumerate(selected):
            new_row = copy.deepcopy(row)
            new_row["window_index"] = int(new_idx)
            new_row["stride_source_window_index"] = int(old_idx)
            metadata = new_row.get("window_metadata")
            if isinstance(metadata, dict):
                metadata["stride_source_window_index"] = int(old_idx)
                new_row["window_metadata"] = metadata
            filtered_window_outputs.append(new_row)

        filtered_calls: List[Dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            step_type = str(call.get("step_type") or "")
            if step_type != "oracle_evaluator":
                filtered_calls.append(copy.deepcopy(call))
                continue
            old_window_index = _safe_int(call.get("window_index"), default=-1)
            if old_window_index not in old_to_new:
                continue
            new_call = copy.deepcopy(call)
            new_call["window_index"] = int(old_to_new[old_window_index])
            metadata = new_call.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["stride_source_window_index"] = int(old_window_index)
            new_call["metadata"] = metadata
            filtered_calls.append(new_call)

        predictions_payload["window_outputs"] = filtered_window_outputs
        predictions_payload["num_windows_requested"] = len(filtered_window_outputs)
        predictions_payload["num_windows_evaluated"] = len(filtered_window_outputs)
        predictions_payload["stride_filter"] = copy.deepcopy(expected_manifest)
        predictions_payload["stride_filter"]["applied_at"] = _now_iso()

        llm_payload["calls"] = filtered_calls
        llm_payload["total_calls"] = len(filtered_calls)
        llm_payload["stride_filter"] = copy.deepcopy(expected_manifest)
        llm_payload["stride_filter"]["applied_at"] = _now_iso()

        stats.patients_processed += 1
        stats.windows_before += len(window_outputs)
        stats.windows_after += len(filtered_window_outputs)
        stats.calls_before += len(calls)
        stats.calls_after += len(filtered_calls)

        _dump_json(pred_path, predictions_payload)
        _dump_json(llm_path, llm_payload)
        save_llm_calls_html(llm_payload, patient_dir / "llm_calls.html")

        context_path = patient_dir / "window_contexts.json"
        if context_path.exists():
            context_payload = _load_json_dict(context_path)
            context_payload = _filter_window_context_payload(context_payload, old_to_new=old_to_new)
            context_payload["stride_filter"] = copy.deepcopy(expected_manifest)
            context_payload["stride_filter"]["applied_at"] = _now_iso()
            _dump_json(context_path, context_payload)

        full_context_path = patient_dir / "full_window_contexts.json"
        if full_context_path.exists():
            full_context_payload = _load_json_dict(full_context_path)
            full_context_payload = _filter_window_context_payload(full_context_payload, old_to_new=old_to_new)
            full_context_payload["stride_filter"] = copy.deepcopy(expected_manifest)
            full_context_payload["stride_filter"]["applied_at"] = _now_iso()
            _dump_json(full_context_path, full_context_payload)

    manifest_payload = copy.deepcopy(expected_manifest)
    manifest_payload.update(
        {
            "applied_at": _now_iso(),
            "patients_processed": int(stats.patients_processed),
            "windows_before": int(stats.windows_before),
            "windows_after": int(stats.windows_after),
            "calls_before": int(stats.calls_before),
            "calls_after": int(stats.calls_after),
        }
    )
    _dump_json(manifest_path, manifest_payload)
    return stats


def _build_window_output_index(window_outputs: List[Dict[str, Any]]) -> Dict[int, int]:
    index_map: Dict[int, int] = {}
    for pos, row in enumerate(window_outputs):
        if not isinstance(row, dict):
            continue
        window_index = _safe_int(row.get("window_index"), default=pos)
        index_map[window_index] = pos
    return index_map


def _select_oracle_call_positions(llm_calls: List[Dict[str, Any]]) -> Dict[int, int]:
    positions: Dict[int, int] = {}
    for pos, call in enumerate(llm_calls):
        if not isinstance(call, dict):
            continue
        if str(call.get("step_type") or "") != "oracle_evaluator":
            continue
        window_index = _safe_int(call.get("window_index"), default=-1)
        if window_index < 0:
            continue
        positions[window_index] = pos
    return positions


def _collect_patient_bundles(run_dir: Path) -> Dict[str, PatientBundle]:
    bundles: Dict[str, PatientBundle] = {}
    patients_dir = run_dir / "patients"
    if not patients_dir.exists():
        raise FileNotFoundError(f"Missing patients directory: {patients_dir}")
    for patient_dir in sorted(patients_dir.iterdir(), key=lambda p: p.name):
        if not patient_dir.is_dir():
            continue
        llm_path = patient_dir / "llm_calls.json"
        pred_path = patient_dir / "oracle_predictions.json"
        if not llm_path.exists() or not pred_path.exists():
            continue
        llm_payload = _load_json_dict(llm_path)
        predictions_payload = _load_json_dict(pred_path)
        llm_calls = llm_payload.get("calls")
        window_outputs = predictions_payload.get("window_outputs")
        if not isinstance(llm_calls, list) or not isinstance(window_outputs, list):
            continue
        llm_rows = [row for row in llm_calls if isinstance(row, dict)]
        output_rows = [row for row in window_outputs if isinstance(row, dict)]
        bundle = PatientBundle(
            patient_dir=patient_dir,
            llm_payload=llm_payload,
            predictions_payload=predictions_payload,
            llm_calls=llm_rows,
            window_outputs=output_rows,
            window_output_pos_by_index=_build_window_output_index(output_rows),
        )
        bundles[patient_dir.name] = bundle
    return bundles


def _collect_repair_tasks(bundles: Dict[str, PatientBundle]) -> Tuple[List[RepairTask], List[Dict[str, Any]]]:
    tasks: List[RepairTask] = []
    skipped: List[Dict[str, Any]] = []
    for patient_id, bundle in bundles.items():
        call_pos_by_window = _select_oracle_call_positions(bundle.llm_calls)
        for window_index, call_pos in sorted(call_pos_by_window.items()):
            call = bundle.llm_calls[call_pos]
            parsed = call.get("parsed_response")
            if _is_complete_oracle_output(parsed):
                continue
            prompt = str(call.get("prompt") or "")
            if not prompt.strip():
                skipped.append(
                    {
                        "patient_id": patient_id,
                        "window_index": window_index,
                        "reason": "missing_prompt",
                    }
                )
                continue
            tasks.append(
                RepairTask(
                    patient_id=patient_id,
                    window_index=window_index,
                    call_index=call_pos,
                    prompt=prompt,
                )
            )
    return tasks, skipped


def _get_thread_client(
    *,
    provider: str,
    model: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
    timeout_seconds: float,
) -> LLMClient:
    cache_key = f"{provider}::{model}::{temperature}::{max_tokens}::{timeout_seconds}"
    current_key = getattr(_THREAD_LOCAL, "client_key", None)
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None or current_key != cache_key:
        kwargs: Dict[str, Any] = {
            "provider": provider,
            "model": model,
            "request_timeout_seconds": float(timeout_seconds),
        }
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        client = LLMClient(**kwargs)
        _THREAD_LOCAL.client = client
        _THREAD_LOCAL.client_key = cache_key
    return client


def _run_one_task(
    task: RepairTask,
    *,
    provider: str,
    model: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
    timeout_seconds: float,
    max_attempts: int,
) -> RepairResult:
    client = _get_thread_client(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )
    last_error = ""
    last_text = ""
    last_parsed: Optional[Dict[str, Any]] = None
    last_input_tokens = 0
    last_output_tokens = 0
    last_response_model = ""

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat(
                prompt=task.prompt,
                response_format="text",
                timeout_seconds=float(timeout_seconds),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = str(response.get("content") or "")
            parsed = response.get("parsed")
            if not isinstance(parsed, dict):
                parsed = parse_json_dict_best_effort(content)
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            input_tokens = _safe_int(usage.get("input_tokens"), default=0)
            output_tokens = _safe_int(usage.get("output_tokens"), default=0)
            response_model = str(response.get("model") or "")

            last_text = content
            last_parsed = parsed if isinstance(parsed, dict) else None
            last_input_tokens = input_tokens
            last_output_tokens = output_tokens
            last_response_model = response_model

            if _is_complete_oracle_output(last_parsed):
                return RepairResult(
                    patient_id=task.patient_id,
                    window_index=task.window_index,
                    call_index=task.call_index,
                    success=True,
                    attempts=attempt,
                    error="",
                    response_text=last_text,
                    parsed_response=last_parsed,
                    input_tokens=last_input_tokens,
                    output_tokens=last_output_tokens,
                    response_model=last_response_model,
                )
            last_error = "incomplete_parsed_response"
        except Exception as exc:
            last_error = str(exc)

    return RepairResult(
        patient_id=task.patient_id,
        window_index=task.window_index,
        call_index=task.call_index,
        success=False,
        attempts=max_attempts,
        error=last_error,
        response_text=last_text,
        parsed_response=last_parsed,
        input_tokens=last_input_tokens,
        output_tokens=last_output_tokens,
        response_model=last_response_model,
    )


def _apply_success_result(bundle: PatientBundle, result: RepairResult, provider: str, model: str) -> None:
    call = bundle.llm_calls[result.call_index]
    call["timestamp"] = _now_iso()
    call["response"] = result.response_text
    call["parsed_response"] = result.parsed_response
    call["input_tokens"] = int(result.input_tokens)
    call["output_tokens"] = int(result.output_tokens)
    metadata = call.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["parse_source"] = "best_effort_json_repair"
    metadata["llm_provider"] = provider
    metadata["llm_model"] = model
    if result.response_model:
        metadata["response_model"] = result.response_model
    metadata["repair_attempts"] = int(result.attempts)
    call["metadata"] = metadata

    output_pos = bundle.window_output_pos_by_index.get(result.window_index)
    if output_pos is not None and 0 <= output_pos < len(bundle.window_outputs):
        bundle.window_outputs[output_pos]["oracle_output"] = result.parsed_response

    bundle.llm_payload["llm_provider"] = provider
    bundle.llm_payload["llm_model"] = model
    bundle.llm_payload["calls"] = bundle.llm_calls
    bundle.predictions_payload["window_outputs"] = bundle.window_outputs


def _write_bundle(bundle: PatientBundle) -> None:
    llm_path = bundle.patient_dir / "llm_calls.json"
    pred_path = bundle.patient_dir / "oracle_predictions.json"
    html_path = bundle.patient_dir / "llm_calls.html"
    _dump_json(llm_path, bundle.llm_payload)
    _dump_json(pred_path, bundle.predictions_payload)
    save_llm_calls_html(bundle.llm_payload, html_path)


def main() -> None:
    args = _parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be >= 1")

    source_run_dir, output_run_dir = _resolve_run_dirs(args)
    _prepare_output_run_dir(source_run_dir, output_run_dir, bool(args.dry_run))

    target_window_step_hours = _resolve_target_window_step_hours(args)
    source_window_step_hours: Optional[float] = None

    stride_stats = None
    if target_window_step_hours is not None:
        source_window_step_hours = _resolve_source_window_step_hours(args)
        stride_stats = _apply_stride_filter_to_run(
            source_run_dir=source_run_dir,
            output_run_dir=output_run_dir,
            target_window_step_hours=float(target_window_step_hours),
            source_window_step_hours=float(source_window_step_hours),
            stride_offset=int(args.stride_offset),
            dry_run=bool(args.dry_run),
        )

    bundles = _collect_patient_bundles(output_run_dir)
    tasks, skipped = _collect_repair_tasks(bundles)

    started_at = _now_iso()
    print(f"Source run dir: {source_run_dir}")
    print(f"Working run dir: {output_run_dir}")
    if stride_stats is not None:
        print(
            "Window step filter applied: "
            f"patients={stride_stats.patients_processed} "
            f"windows_before={stride_stats.windows_before} "
            f"windows_after={stride_stats.windows_after}"
        )
    print(f"Patients loaded: {len(bundles)}")
    print(f"Truncated windows detected: {len(tasks)}")
    print(f"Skipped windows: {len(skipped)}")

    results: List[RepairResult] = []
    success_results: List[RepairResult] = []
    failed_results: List[RepairResult] = []
    touched_patient_ids = set()
    if len(tasks) > 0:
        with ThreadPoolExecutor(max_workers=int(args.threads)) as executor:
            futures = [
                executor.submit(
                    _run_one_task,
                    task,
                    provider=str(args.provider),
                    model=str(args.model),
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout_seconds=float(args.timeout_seconds),
                    max_attempts=int(args.max_attempts),
                )
                for task in tasks
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result.success:
                    success_results.append(result)
                    if not args.dry_run:
                        bundle = bundles.get(result.patient_id)
                        if bundle is not None:
                            _apply_success_result(
                                bundle=bundle,
                                result=result,
                                provider=str(args.provider),
                                model=str(args.model),
                            )
                            _write_bundle(bundle)
                            touched_patient_ids.add(result.patient_id)
                else:
                    failed_results.append(result)
                status = "OK" if result.success else "FAIL"
                print(
                    f"[{status}] patient={result.patient_id} window={result.window_index} "
                    f"attempts={result.attempts} error={result.error}"
                )

    if args.dry_run:
        print("Dry run enabled: no files were modified.")
    else:
        print(f"Updated patients written: {len(touched_patient_ids)}")

    completed_at = _now_iso()
    report = {
        "source_run_dir": str(source_run_dir),
        "working_run_dir": str(output_run_dir),
        "provider": str(args.provider),
        "model": str(args.model),
        "threads": int(args.threads),
        "max_attempts": int(args.max_attempts),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout_seconds": float(args.timeout_seconds),
        "output_run_dir": str(output_run_dir),
        "target_window_step_hours": (
            float(target_window_step_hours) if target_window_step_hours is not None else None
        ),
        "source_window_step_hours": (
            float(source_window_step_hours) if source_window_step_hours is not None else None
        ),
        "stride_offset": int(args.stride_offset),
        "dry_run": bool(args.dry_run),
        "started_at": started_at,
        "completed_at": completed_at,
        "patients_loaded": len(bundles),
        "tasks_detected": len(tasks),
        "tasks_skipped": len(skipped),
        "tasks_succeeded": len(success_results),
        "tasks_failed": len(failed_results),
        "skipped": skipped,
        "failed": [
            {
                "patient_id": row.patient_id,
                "window_index": row.window_index,
                "attempts": row.attempts,
                "error": row.error,
            }
            for row in sorted(failed_results, key=lambda item: (item.patient_id, item.window_index))
        ],
    }
    if stride_stats is not None:
        report["stride_filter_stats"] = {
            "patients_processed": int(stride_stats.patients_processed),
            "windows_before": int(stride_stats.windows_before),
            "windows_after": int(stride_stats.windows_after),
            "calls_before": int(stride_stats.calls_before),
            "calls_after": int(stride_stats.calls_after),
        }
    report_path = output_run_dir / f"repair_truncated_oracle_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _dump_json(report_path, report)
    print(f"Report saved: {report_path}")

    if len(failed_results) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
