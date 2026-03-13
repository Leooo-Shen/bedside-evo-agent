"""
TODO: not checked yet

LLM experiment: does a selected discharge summary match the current ICU stay events?

Rules:
1) Use selected discharge summary from utils.discharge_summary_selector.
2) For each sampled ICU stay, provide:
   - Selected discharge summary text
   - All events within that ICU stay time window (excluding NOTE_DISCHARGESUMMARY)
3) Ask Gemini for strict yes/no output.
4) Report yes ratio by rule and overall.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import local
from typing import Any, Dict, List, Optional

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config
from data_parser import MIMICDataParser
from model.llms import LLMClient
from utils.discharge_summary_selector import (
    select_discharge_summaries_for_icu_stays,
    summarize_discharge_summary_selection,
)

RULE_1 = "in_icu_exactly_one"
RULE_2 = "post_icu_same_hosp_within_7d_no_new_icu"
TARGET_RULES = [RULE_1, RULE_2]

_THREAD_LOCAL = local()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run yes/no LLM experiment for discharge-summary-to-ICU-event matching."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.json",
        help="Path to config json (default: config/config.json).",
    )
    parser.add_argument(
        "--events-path",
        type=str,
        default=None,
        help="Override events parquet path. Defaults to config.data.events_path.",
    )
    parser.add_argument(
        "--icu-stay-path",
        type=str,
        default=None,
        help="Override ICU stay parquet path. Defaults to config.data.icu_stay_path.",
    )
    parser.add_argument(
        "--use-raw-icu-stays",
        action="store_true",
        help=("Use raw ICU stay parquet (e.g., 594 stays) instead of parser-filtered ICU stays " "(e.g., 490 stays)."),
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.1,
        help="Sampling fraction per rule (default: 0.1).",
    )
    parser.add_argument(
        "--max-days-after-leave",
        type=float,
        default=7.0,
        help="Maximum days for rule2 post-ICU summary linkage (default: 7.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for per-rule sampling (default: 42).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=["google", "gemini", "openai", "anthropic"],
        help="LLM provider (default: google).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.1-flash-lite-preview",
        help="LLM model (default: gemini-3.1-flash-lite-preview).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0).",
    )
    parser.add_argument(
        "--response-max-tokens",
        type=int,
        default=16,
        help="Max output tokens for yes/no response (default: 16).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of threads for LLM calls (default: 8).",
    )
    parser.add_argument(
        "--max-events-per-stay",
        type=int,
        default=2000,
        help="Optional cap of ICU events per stay in prompt (default: no cap).",
    )
    parser.add_argument(
        "--max-text-chars-per-event",
        type=int,
        default=160,
        help="Max characters kept from each event text_value (default: 160).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment_results",
        help="Output root dir (default: experiment_results).",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Save prompt text in outputs (off by default due file size).",
    )
    return parser.parse_args()


def _normalize_yes_no(text: Any) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return "unknown"
    exact = raw.replace(".", "").replace("`", "").strip()
    if exact in {"yes", "no"}:
        return exact
    match = re.search(r"\b(yes|no)\b", raw)
    if match:
        return match.group(1)
    return "unknown"


def _safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _format_event_line(event: Dict[str, Any], text_char_limit: int) -> str:
    time_text = _safe_text(event.get("time")) or "unknown"
    code = _safe_text(event.get("code")) or "UNKNOWN"
    # parts = [f"[{time_text}] {code}"]
    parts = [f"{code}"]

    specifics = _safe_text(event.get("code_specifics")).strip()
    if specifics:
        parts.append(f"({specifics})")

    numeric_value = event.get("numeric_value")
    if pd.notna(numeric_value):
        try:
            parts.append(f"= {float(numeric_value):.4g}")
        except (TypeError, ValueError):
            parts.append(f"= {_safe_text(numeric_value)}")

    text_value = _safe_text(event.get("text_value")).replace("\n", " ").strip()
    if text_value:
        if text_char_limit >= 0 and len(text_value) > text_char_limit:
            text_value = text_value[:text_char_limit].rstrip() + "..."
        parts.append(f'text="{text_value}"')

    return " ".join(parts)


def _format_events_for_prompt(
    events_df: pd.DataFrame,
    *,
    max_events_per_stay: Optional[int],
    text_char_limit: int,
) -> str:
    if len(events_df) == 0:
        return "(No ICU events found in [enter_time, leave_time] after removing NOTE_DISCHARGESUMMARY.)"

    formatted_lines: List[str] = []
    total_events = len(events_df)
    display_df = events_df
    if max_events_per_stay is not None and max_events_per_stay > 0 and total_events > max_events_per_stay:
        display_df = events_df.head(max_events_per_stay)

    for idx, event in enumerate(display_df.to_dict("records"), start=1):
        formatted_lines.append(f"{idx}. {_format_event_line(event, text_char_limit=text_char_limit)}")

    if len(display_df) < total_events:
        formatted_lines.append(f"... truncated {total_events - len(display_df)} events")

    return "\n".join(formatted_lines)


def _build_prompt(
    *,
    summary_text: str,
    events_text: str,
    subject_id: int,
    icu_stay_id: int,
    enter_time: Any,
    leave_time: Any,
) -> str:
    return (
        "You are a clinical reviewer.\n\n"
        "Task: decide whether this discharge summary matches the SAME ICU stay as the ICU events below. The events in ICU stay may be partial, but if the general context is consistent, then it is likely a match.\n\n"
        "Output rule:\n"
        "- Respond with exactly one word: yes or no.\n"
        "- Do not output anything else.\n\n"
        "Decision guideline:\n"
        "- yes: the summary is broadly consistent with these ICU events.\n"
        "- no: the summary likely belongs to another stay/admission, or is clearly inconsistent.\n\n"
        f"ICU stay id: {icu_stay_id}\n"
        f"Subject id: {subject_id}\n"
        "Discharge summary:\n"
        f"{summary_text}\n\n"
        "ICU events:\n"
        f"{events_text}\n"
    )


def _get_thread_llm_client(args: argparse.Namespace) -> LLMClient:
    client = getattr(_THREAD_LOCAL, "llm_client", None)
    if client is None:
        client = LLMClient(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.response_max_tokens,
        )
        _THREAD_LOCAL.llm_client = client
    return client


def _load_dataframes(
    *,
    config: Config,
    events_path: str,
    icu_stay_path: str,
    use_raw_icu_stays: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if use_raw_icu_stays:
        events_df = pd.read_parquet(events_path)
        icu_stay_df = pd.read_parquet(icu_stay_path)
    else:
        parser = MIMICDataParser(
            events_path=events_path,
            icu_stay_path=icu_stay_path,
        )
        parser.load_data()
        events_df = parser.events_df
        icu_stay_df = parser.icu_stay_df

    if events_df is None or icu_stay_df is None:
        raise RuntimeError("Failed to load events_df or icu_stay_df.")

    return events_df, icu_stay_df


def _sample_rule_rows(
    selection_df: pd.DataFrame,
    *,
    sample_fraction: float,
    seed: int,
) -> pd.DataFrame:
    sampled_frames: List[pd.DataFrame] = []
    for idx, rule in enumerate(TARGET_RULES):
        rule_df = selection_df[
            (selection_df["selected"] == True) & (selection_df["selection_rule"] == rule)  # noqa: E712
        ].copy()
        n = len(rule_df)
        if n == 0:
            sampled_frames.append(rule_df)
            continue

        sample_n = int(math.floor(n * sample_fraction))
        if sample_fraction > 0 and sample_n == 0:
            sample_n = 1
        sample_n = min(sample_n, n)

        if sample_n == n:
            sampled_frames.append(rule_df)
        else:
            sampled_frames.append(rule_df.sample(n=sample_n, random_state=seed + idx).copy())

    if not sampled_frames:
        return pd.DataFrame()
    sampled = pd.concat(sampled_frames, ignore_index=True)
    if len(sampled) == 0:
        return sampled
    sampled = sampled.sort_values(["selection_rule", "subject_id", "icu_stay_id"]).reset_index(drop=True)
    return sampled


def _prepare_events_df(events_df: pd.DataFrame) -> pd.DataFrame:
    required = ["subject_id", "time", "code"]
    missing = [col for col in required if col not in events_df.columns]
    if missing:
        raise ValueError(f"events_df missing required columns: {missing}")

    cols = ["subject_id", "time", "code", "code_specifics", "numeric_value", "text_value", "icu_stay_id"]
    keep_cols = [col for col in cols if col in events_df.columns]
    out = events_df[keep_cols].copy()
    out["subject_id"] = pd.to_numeric(out["subject_id"], errors="coerce").astype("Int64")
    if "icu_stay_id" in out.columns:
        out["icu_stay_id"] = pd.to_numeric(out["icu_stay_id"], errors="coerce").astype("Int64")
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out[out["subject_id"].notna() & out["time"].notna()].copy()
    return out


def _extract_icu_events_for_row(
    *,
    events_by_subject: Any,
    subject_id: int,
    icu_stay_id: int,
    enter_time: Any,
    leave_time: Any,
) -> pd.DataFrame:
    try:
        subject_events = events_by_subject.get_group(subject_id)
    except KeyError:
        return pd.DataFrame(columns=["time", "code", "code_specifics", "numeric_value", "text_value"])

    enter_ts = pd.to_datetime(enter_time, errors="coerce")
    leave_ts = pd.to_datetime(leave_time, errors="coerce")
    if pd.isna(enter_ts) or pd.isna(leave_ts):
        return pd.DataFrame(columns=["time", "code", "code_specifics", "numeric_value", "text_value"])

    stay_events = subject_events[
        (subject_events["time"] >= enter_ts)
        & (subject_events["time"] <= leave_ts)
        & (subject_events["code"] != "NOTE_DISCHARGESUMMARY")
    ].copy()
    # Prefer exact ICU-stay binding when available.
    if "icu_stay_id" in stay_events.columns:
        exact_match = stay_events[stay_events["icu_stay_id"] == int(icu_stay_id)].copy()
        if len(exact_match) > 0:
            stay_events = exact_match
    stay_events = stay_events.sort_values("time", kind="stable").reset_index(drop=True)
    return stay_events


def _evaluate_one_case(
    row: Dict[str, Any],
    *,
    events_by_subject: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    subject_id = int(row["subject_id"])
    icu_stay_id = int(row["icu_stay_id"])
    selection_rule = str(row["selection_rule"])
    enter_time = row["enter_time"]
    leave_time = row["leave_time"]
    summary_text = _safe_text(row.get("selected_note_text_value")).strip()

    icu_events_df = _extract_icu_events_for_row(
        events_by_subject=events_by_subject,
        subject_id=subject_id,
        icu_stay_id=icu_stay_id,
        enter_time=enter_time,
        leave_time=leave_time,
    )
    events_text = _format_events_for_prompt(
        icu_events_df,
        max_events_per_stay=args.max_events_per_stay,
        text_char_limit=args.max_text_chars_per_event,
    )

    prompt = _build_prompt(
        summary_text=summary_text if summary_text else "(Empty discharge summary text)",
        events_text=events_text,
        subject_id=subject_id,
        icu_stay_id=icu_stay_id,
        enter_time=enter_time,
        leave_time=leave_time,
    )

    # print("===" * 10)
    # print(prompt)
    # print("===" * 10)

    # exit(0)  # TODO: remove after testing prompt format

    response_text = ""
    answer = "unknown"
    error = None
    usage: Dict[str, Any] = {}

    try:
        llm_client = _get_thread_llm_client(args)
        response = llm_client.chat(
            prompt=prompt,
            response_format="text",
            temperature=args.temperature,
            max_tokens=args.response_max_tokens,
        )
        response_text = _safe_text(response.get("content")).strip()
        answer = _normalize_yes_no(response_text)
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
    except Exception as exc:
        error = str(exc)

    result = {
        "subject_id": subject_id,
        "icu_stay_id": icu_stay_id,
        "selection_rule": selection_rule,
        "enter_time": _safe_text(enter_time),
        "leave_time": _safe_text(leave_time),
        "selected_note_time": _safe_text(row.get("selected_note_time")),
        "selected_note_hosp_stay_id": _safe_text(row.get("selected_note_hosp_stay_id")),
        "selected_note_delta_hours_after_leave": row.get("selected_note_delta_hours_after_leave"),
        "icu_event_count": int(len(icu_events_df)),
        "summary_char_count": len(summary_text),
        "answer": answer,
        "raw_response": response_text,
        "llm_error": error,
        "usage": usage,
    }
    if args.save_prompts:
        result["prompt"] = prompt

    return result


def _build_stats(results_df: pd.DataFrame) -> Dict[str, Any]:
    def _one_block(df: pd.DataFrame) -> Dict[str, Any]:
        total = int(len(df))
        yes_count = int((df["answer"] == "yes").sum()) if total else 0
        no_count = int((df["answer"] == "no").sum()) if total else 0
        unknown_count = int((df["answer"] == "unknown").sum()) if total else 0
        error_count = int(df["llm_error"].notna().sum()) if total else 0
        yes_ratio = yes_count / total if total else 0.0
        return {
            "total": total,
            "yes": yes_count,
            "no": no_count,
            "unknown": unknown_count,
            "llm_errors": error_count,
            "yes_ratio": yes_ratio,
        }

    stats = {"overall": _one_block(results_df)}
    for rule in TARGET_RULES:
        stats[rule] = _one_block(results_df[results_df["selection_rule"] == rule].copy())
    return stats


def main() -> None:
    args = _parse_args()
    if args.sample_fraction < 0 or args.sample_fraction > 1:
        raise ValueError("--sample-fraction must be in [0, 1].")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1.")
    if args.max_days_after_leave <= 0:
        raise ValueError("--max-days-after-leave must be > 0.")

    config = Config(args.config)
    events_path = args.events_path or config.events_path
    icu_stay_path = args.icu_stay_path or config.icu_stay_path

    print(f"Loading events: {events_path}")
    print(f"Loading ICU stays: {icu_stay_path}")
    events_df, icu_stay_df = _load_dataframes(
        config=config,
        events_path=events_path,
        icu_stay_path=icu_stay_path,
        use_raw_icu_stays=args.use_raw_icu_stays,
    )
    print(f"Loaded events rows: {len(events_df):,}")
    print(f"Loaded ICU stays rows: {len(icu_stay_df):,}")
    print(f"ICU source mode: {'raw parquet' if args.use_raw_icu_stays else 'parser-filtered'}")

    print("\nSelecting discharge summaries with rule1/rule2...")
    selection_df = select_discharge_summaries_for_icu_stays(
        events_df=events_df,
        icu_stay_df=icu_stay_df,
        max_days_after_leave=args.max_days_after_leave,
    )
    selection_summary = summarize_discharge_summary_selection(selection_df)
    print(json.dumps(selection_summary, indent=2, ensure_ascii=False))

    sampled_df = _sample_rule_rows(
        selection_df=selection_df,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )
    print(f"\nSampled rows: {len(sampled_df)}")
    for rule in TARGET_RULES:
        rule_total = int(
            ((selection_df["selected"] == True) & (selection_df["selection_rule"] == rule)).sum()  # noqa: E712
        )
        rule_sampled = int((sampled_df["selection_rule"] == rule).sum())
        print(f"  {rule}: sampled {rule_sampled}/{rule_total}")

    if len(sampled_df) == 0:
        print("No sampled rows. Exiting.")
        return

    events_for_lookup = _prepare_events_df(events_df)
    events_by_subject = events_for_lookup.groupby("subject_id", sort=False)

    rows = sampled_df.to_dict("records")
    results: List[Dict[str, Any]] = []
    print(f"\nRunning LLM calls with max_workers={args.max_workers} ...")
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(rows))) as executor:
        future_to_case = {
            executor.submit(
                _evaluate_one_case,
                row,
                events_by_subject=events_by_subject,
                args=args,
            ): row
            for row in rows
        }
        completed = 0
        total = len(future_to_case)
        for future in as_completed(future_to_case):
            case = future_to_case[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "subject_id": int(case["subject_id"]),
                    "icu_stay_id": int(case["icu_stay_id"]),
                    "selection_rule": str(case["selection_rule"]),
                    "answer": "unknown",
                    "raw_response": "",
                    "llm_error": f"Unhandled worker error: {exc}",
                    "usage": {},
                    "icu_event_count": 0,
                    "summary_char_count": 0,
                }
            results.append(result)
            print(
                f"[{completed}/{total}] {result['selection_rule']} "
                f"{result['subject_id']}_{result['icu_stay_id']} -> {result['answer']}"
            )

    results_df = (
        pd.DataFrame(results).sort_values(["selection_rule", "subject_id", "icu_stay_id"]).reset_index(drop=True)
    )
    stats = _build_stats(results_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"discharge_summary_match_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    sampled_df.to_csv(output_root / "sampled_cases.csv", index=False)
    results_df.to_csv(output_root / "llm_results.csv", index=False)
    (output_root / "llm_results.json").write_text(
        json.dumps(results_df.to_dict("records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    run_summary = {
        "timestamp": timestamp,
        "events_path": events_path,
        "icu_stay_path": icu_stay_path,
        "icu_source_mode": "raw_parquet" if args.use_raw_icu_stays else "parser_filtered",
        "sample_fraction": args.sample_fraction,
        "seed": args.seed,
        "max_days_after_leave": args.max_days_after_leave,
        "llm_provider": args.provider,
        "llm_model": args.model,
        "max_workers": args.max_workers,
        "response_max_tokens": args.response_max_tokens,
        "max_events_per_stay": args.max_events_per_stay,
        "max_text_chars_per_event": args.max_text_chars_per_event,
        "selection_summary": selection_summary,
        "sample_counts": {rule: int((sampled_df["selection_rule"] == rule).sum()) for rule in TARGET_RULES},
        "stats": stats,
    }
    (output_root / "summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== YES Ratio =====")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nSaved results to: {output_root}")


if __name__ == "__main__":
    main()
