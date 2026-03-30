"""Tests for experiments.oracle.build_annotation_window_set."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.oracle.build_annotation_window_set import build_annotation_window_set


def _make_oracle_output(label: str) -> Dict[str, object]:
    return {
        "patient_status": {
            "domains": {
                "hemodynamics": {"label": "stable", "key_signals": ["map"], "rationale": "stable"},
                "respiratory": {"label": "stable", "key_signals": ["spo2"], "rationale": "stable"},
                "renal_metabolic": {"label": "stable", "key_signals": ["uo"], "rationale": "stable"},
                "neurology": {"label": "stable", "key_signals": ["gcs"], "rationale": "stable"},
            },
            "overall": {"label": label, "rationale": f"{label} summary"},
        },
        "action_evaluations": [],
        "recommendations": [],
        "overall_window_summary": f"{label} window summary",
    }


def _write_patient_payloads(
    *,
    patient_dir: Path,
    subject_id: int,
    icu_stay_id: int,
    survived: bool,
    labels: Sequence[str],
    base_time: datetime,
) -> None:
    patient_dir.mkdir(parents=True, exist_ok=True)

    window_outputs: List[Dict[str, object]] = []
    window_contexts: List[Dict[str, object]] = []
    calls: List[Dict[str, object]] = []
    for idx, label in enumerate(labels, start=1):
        start = base_time + timedelta(hours=idx - 1)
        end = start + timedelta(hours=1)
        oracle_output = _make_oracle_output(label)

        window_outputs.append(
            {
                "window_index": idx,
                "window_metadata": {
                    "subject_id": subject_id,
                    "icu_stay_id": icu_stay_id,
                    "window_start_time": start.isoformat(),
                    "window_end_time": end.isoformat(),
                    "hours_since_admission": float(idx - 1),
                    "current_window_hours": 1.0,
                    "num_history_events": 0,
                    "num_current_events": 1,
                },
                "raw_current_events": [
                    {"time": start.isoformat(), "code": "VITALS", "code_specifics": f"HR_{idx}", "numeric_value": 80}
                ],
                "oracle_output": oracle_output,
            }
        )

        window_contexts.append(
            {
                "window_index": idx,
                "window_metadata": {
                    "subject_id": subject_id,
                    "icu_stay_id": icu_stay_id,
                    "window_start_time": start.isoformat(),
                    "window_end_time": end.isoformat(),
                    "hours_since_admission": float(idx - 1),
                    "current_window_hours": 1.0,
                    "num_history_events": 0,
                    "num_current_events": 1,
                    "history_hours": 1.0,
                    "future_hours": 1.0,
                },
                "history_events": [],
                "current_events": [
                    {"time": start.isoformat(), "code": "VITALS", "code_specifics": f"HR_{idx}", "numeric_value": 80}
                ],
                "future_events": [],
                "prompt_sections": {
                    "icu_discharge_summary": f"summary_{subject_id}_{idx}",
                    "icu_trajectory_context_window": f"traj_{subject_id}_{idx}",
                    "previous_events_current_window": f"hist_{subject_id}_{idx}",
                    "current_observation_window": f"curr_{subject_id}_{idx}",
                },
            }
        )

        # Keep 0-based window_index to mimic run_oracle llm_calls behavior.
        calls.append(
            {
                "timestamp": (start + timedelta(minutes=5)).isoformat(),
                "patient_id": f"{subject_id}_{icu_stay_id}",
                "step_type": "oracle_evaluator",
                "window_index": idx - 1,
                "hours_since_admission": float(idx - 1),
                "prompt": f"prompt_{subject_id}_{idx}",
                "response": json.dumps(oracle_output),
                "parsed_response": oracle_output,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )

    predictions_payload = {
        "run_id": "test_run",
        "generated_at": datetime.now().isoformat(),
        "subject_id": subject_id,
        "icu_stay_id": icu_stay_id,
        "trajectory_metadata": {
            "survived": survived,
            "true_survived": survived,
            "prompt_survived": survived,
        },
        "num_windows_requested": len(window_outputs),
        "num_windows_evaluated": len(window_outputs),
        "window_outputs": window_outputs,
    }

    contexts_payload = {
        "run_id": "test_run",
        "generated_at": datetime.now().isoformat(),
        "subject_id": subject_id,
        "icu_stay_id": icu_stay_id,
        "history_hours": 1.0,
        "future_hours": 1.0,
        "window_contexts": window_contexts,
    }

    llm_payload = {
        "patient_id": f"{subject_id}_{icu_stay_id}",
        "agent_type": "oracle",
        "llm_provider": "fake",
        "llm_model": "fake-model",
        "include_icu_outcome_in_prompt": True,
        "prompt_outcome_mode": "with_icu_outcome",
        "pipeline_agents": [{"name": "oracle_evaluator", "used": True}],
        "total_calls": len(calls),
        "calls": calls,
    }

    with open(patient_dir / "oracle_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions_payload, f, indent=2)
    with open(patient_dir / "window_contexts.json", "w", encoding="utf-8") as f:
        json.dump(contexts_payload, f, indent=2)
    with open(patient_dir / "llm_calls.json", "w", encoding="utf-8") as f:
        json.dump(llm_payload, f, indent=2)


def _create_run_fixture(tmp_path: Path, patient_specs: Sequence[Tuple[int, int, bool, Sequence[str]]]) -> Path:
    run_dir = tmp_path / "oracle_conditions_20260101_010101"
    (run_dir / "conditions" / "full_visible" / "patients").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_state.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_dir.name, "conditions": ["full_visible"]}, f, indent=2)

    base_time = datetime(2142, 1, 1, 0, 0, 0)
    for offset, (subject_id, icu_stay_id, survived, labels) in enumerate(patient_specs):
        patient_dir = run_dir / "conditions" / "full_visible" / "patients" / f"{subject_id}_{icu_stay_id}"
        _write_patient_payloads(
            patient_dir=patient_dir,
            subject_id=subject_id,
            icu_stay_id=icu_stay_id,
            survived=survived,
            labels=labels,
            base_time=base_time + timedelta(days=offset),
        )
    return run_dir


def _read_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_sampling_balances_patients_and_is_reproducible(tmp_path: Path) -> None:
    run_dir = _create_run_fixture(
        tmp_path,
        [
            (101, 9001, True, ["improving", "stable", "deteriorating", "insufficient_data", "stable", "improving"]),
            (102, 9002, True, ["stable", "stable", "improving", "deteriorating", "stable", "insufficient_data"]),
            (
                201,
                9101,
                False,
                ["deteriorating", "deteriorating", "stable", "insufficient_data", "improving", "stable"],
            ),
            (
                202,
                9102,
                False,
                ["deteriorating", "stable", "deteriorating", "improving", "stable", "insufficient_data"],
            ),
        ],
    )

    out_a = build_annotation_window_set(
        run_dir=run_dir,
        condition="full_visible",
        target_windows=8,
        seed=7,
        patients_per_outcome=1,
        k_per_patient=4,
        output_dir=tmp_path / "pack_a",
    )
    out_b = build_annotation_window_set(
        run_dir=run_dir,
        condition="full_visible",
        target_windows=8,
        seed=7,
        patients_per_outcome=1,
        k_per_patient=4,
        output_dir=tmp_path / "pack_b",
    )

    stats_a = json.loads((out_a / "sampling_statistics.json").read_text(encoding="utf-8"))
    stats_b = json.loads((out_b / "sampling_statistics.json").read_text(encoding="utf-8"))
    assert len(stats_a["selected_patients"]["survived"]) == 1
    assert len(stats_a["selected_patients"]["died"]) == 1
    assert stats_a["selected_patients"] == stats_b["selected_patients"]

    manifest_a = _read_manifest_rows(out_a / "sampling_manifest.csv")
    manifest_b = _read_manifest_rows(out_b / "sampling_manifest.csv")
    selected_keys_a = [(row["patient_id"], row["source_window_index"], row["status_label"]) for row in manifest_a]
    selected_keys_b = [(row["patient_id"], row["source_window_index"], row["status_label"]) for row in manifest_b]
    assert selected_keys_a == selected_keys_b


def test_auto_k_search_respects_min_k_logic(tmp_path: Path) -> None:
    run_dir = _create_run_fixture(
        tmp_path,
        [
            (101, 9001, True, ["stable", "improving"]),  # 2
            (102, 9002, True, ["stable", "stable", "improving", "deteriorating", "stable"]),  # 5
            (201, 9101, False, ["deteriorating", "stable", "deteriorating", "improving", "stable"]),  # 5
            (202, 9102, False, ["deteriorating", "deteriorating", "stable", "insufficient_data", "stable"]),  # 5
        ],
    )
    out_dir = build_annotation_window_set(
        run_dir=run_dir,
        condition="full_visible",
        target_windows=8,
        seed=1,
        patients_per_outcome=2,
        k_per_patient=None,
        output_dir=tmp_path / "auto_k_pack",
    )
    stats = json.loads((out_dir / "sampling_statistics.json").read_text(encoding="utf-8"))
    # target=8, buffer=1.2 => desired >=10. Counts [2,5,5,5], minimal k is 3 => 11.
    assert stats["k_per_patient"] == 3
    assert stats["k_per_patient_auto_selected"] is True
    assert stats["initial_pool"]["actual_size"] == 11


def test_final_target_and_insufficient_data_proportion(tmp_path: Path) -> None:
    run_dir = _create_run_fixture(
        tmp_path,
        [
            (101, 9001, True, ["improving", "stable", "deteriorating", "insufficient_data", "stable", "improving"]),
            (102, 9002, True, ["stable", "stable", "improving", "deteriorating", "stable", "insufficient_data"]),
            (
                201,
                9101,
                False,
                ["deteriorating", "deteriorating", "stable", "insufficient_data", "improving", "stable"],
            ),
            (
                202,
                9102,
                False,
                ["deteriorating", "stable", "deteriorating", "improving", "stable", "insufficient_data"],
            ),
        ],
    )
    out_dir = build_annotation_window_set(
        run_dir=run_dir,
        condition="full_visible",
        target_windows=12,
        seed=2,
        patients_per_outcome=2,
        k_per_patient=6,
        output_dir=tmp_path / "target_pack",
    )
    stats = json.loads((out_dir / "sampling_statistics.json").read_text(encoding="utf-8"))
    assert stats["final_selection"]["size"] == 12

    candidate = stats["candidate_pool"]["status_distribution"]
    candidate_total = sum(candidate.values())
    expected_insuff = round(12 * candidate.get("insufficient_data", 0) / candidate_total) if candidate_total > 0 else 0
    actual_insuff = stats["final_selection"]["status_distribution"].get("insufficient_data", 0)
    assert abs(actual_insuff - expected_insuff) <= 1


def test_outputs_are_aligned_and_patient_windows_are_contiguous(tmp_path: Path) -> None:
    run_dir = _create_run_fixture(
        tmp_path,
        [
            (101, 9001, True, ["improving", "stable", "deteriorating", "insufficient_data", "stable", "improving"]),
            (102, 9002, True, ["stable", "stable", "improving", "deteriorating", "stable", "insufficient_data"]),
            (
                201,
                9101,
                False,
                ["deteriorating", "deteriorating", "stable", "insufficient_data", "improving", "stable"],
            ),
            (
                202,
                9102,
                False,
                ["deteriorating", "stable", "deteriorating", "improving", "stable", "insufficient_data"],
            ),
        ],
    )
    out_dir = build_annotation_window_set(
        run_dir=run_dir,
        condition="full_visible",
        target_windows=10,
        seed=9,
        patients_per_outcome=2,
        k_per_patient=5,
        output_dir=tmp_path / "alignment_pack",
    )

    predictions = json.loads((out_dir / "oracle_predictions.json").read_text(encoding="utf-8"))
    contexts = json.loads((out_dir / "window_contexts.json").read_text(encoding="utf-8"))
    llm_calls = json.loads((out_dir / "llm_calls.json").read_text(encoding="utf-8"))

    window_outputs = predictions["window_outputs"]
    context_windows = contexts["window_contexts"]
    calls = llm_calls["calls"]
    assert len(window_outputs) == 10
    assert len(context_windows) == 10
    assert len(calls) == 10

    assert [item["window_index"] for item in window_outputs] == list(range(1, 11))
    assert [item["window_index"] for item in context_windows] == list(range(1, 11))
    assert [item["annotation_window_index"] for item in calls] == list(range(1, 11))

    # Patient windows must be contiguous in final ordering.
    patient_sequence = [item["window_metadata"]["source_patient_id"] for item in window_outputs]
    seen = []
    for patient_id in patient_sequence:
        if not seen or seen[-1] != patient_id:
            seen.append(patient_id)
    for patient_id in set(patient_sequence):
        assert patient_sequence.count(patient_id) == sum(1 for p in patient_sequence if p == patient_id)
        first = patient_sequence.index(patient_id)
        last = len(patient_sequence) - 1 - patient_sequence[::-1].index(patient_id)
        assert all(p == patient_id for p in patient_sequence[first : last + 1])

    # Within each patient, windows are sorted by start time.
    starts_by_patient: Dict[str, List[str]] = {}
    for item in window_outputs:
        meta = item["window_metadata"]
        starts_by_patient.setdefault(meta["source_patient_id"], []).append(meta["window_start_time"])
    for starts in starts_by_patient.values():
        assert starts == sorted(starts)

    # predictions / contexts / llm calls must align on source metadata.
    context_by_idx = {item["window_index"]: item for item in context_windows}
    call_by_idx = {item["annotation_window_index"]: item for item in calls}
    for pred in window_outputs:
        idx = pred["window_index"]
        pred_meta = pred["window_metadata"]
        ctx_meta = context_by_idx[idx]["window_metadata"]
        call_meta = call_by_idx[idx]
        assert pred_meta["source_subject_id"] == ctx_meta["source_subject_id"] == call_meta["source_subject_id"]
        assert pred_meta["source_icu_stay_id"] == ctx_meta["source_icu_stay_id"] == call_meta["source_icu_stay_id"]
        assert pred_meta["source_window_index"] == ctx_meta["source_window_index"] == call_meta["source_window_index"]
        assert isinstance(call_meta.get("prompt"), str) and call_meta["prompt"].startswith("prompt_")
        assert call_meta.get("parsed_response") == pred.get("oracle_output")
