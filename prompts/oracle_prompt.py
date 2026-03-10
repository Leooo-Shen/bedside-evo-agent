"""Prompt templates for Oracle offline evaluator."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List

ORACLE_PROMPT_TEMPLATE = """
You are Oracle, a clinical AI evaluator with hindsight access to a patient's full ICU trajectory including the ICU results.
Your task is to evaluate a specific local observation window {window_time} using the hindsight knowledge of the ICU trajectory. Depending on the context, your hindsight can come from ICU discharge summaries or future ICU events. 
For evaluating this observation window you must produce two assessments:

## PART 1 — PATIENT STATUS
Using the provided local context as ground truth, assess the patient's clinical direction at the time of this window across four domains, then synthesize an overall status.

Domains:
- Hemodynamics: heart rate, MAP, lactate, perfusion, vasopressor requirements
- Respiratory: SpO2, PaO2/FiO2, respiratory rate, ventilator settings
- Renal/Metabolic: creatinine, urine output, electrolytes, acid-base status
- Neurology: GCS, mental status changes, RASS sedation score

For each domain and the overall status, choose exactly one of:
- improving: indicators trending toward stability or recovery relative to the provided context.
- stable: no meaningful change in either direction.
- deteriorating: indicators trending toward worsening or decompensation relative to the provided context.
- insufficient_data: available information is not sufficient to make a reliable judgment (e.g., sparse events, conflicting signals).

Synthesize the overall label by reasoning about the relative clinical weight of each domain for this specific patient — do not apply a fixed rule.

Important nuances:
- Do NOT conflate outcome with care quality. A patient may be deteriorating despite excellent care (e.g., terminal illness, refractory septic shock managed correctly).
- A patient may be labelled stable or improving even if they eventually die, if the trajectory at this window genuinely reflects that direction.

## PART 2 — DOCTOR ACTION EVALUATION AND RECOMMENDATIONS
You will be given a list of clinical actions taken during the {window_time} window.
Evaluate EACH action individually on two dimensions:

A. Guideline adherence — does this action follow established ICU clinical guidelines (e.g., Surviving Sepsis Campaign, ARDSNet, PADIS guidelines, AHA/ACC where relevant)?
Labels: adherent | non_adherent | not_applicable | guideline_unclear

B. Contextual appropriateness — given THIS patient's specific condition, trajectory, comorbidities, and the eventual outcome known to you, was this action appropriate?
Labels: appropriate | suboptimal | potentially_harmful | not_enough_context

Use your hindsight knowledge from the provided context to inform contextual appropriateness, but be fair: judge the action against what the context reveals, not against impossible foresight.
If an action was reasonable at the time but later context revealed a missed diagnosis, note this nuance.

After evaluating all actions, provide maximum top {top_k} recommendations for what the clinical team should have prioritised at this window, grounded in hindsight knowledge of the trajectory.
Each recommendation must be concrete and actionable (e.g., "Increase norepinephrine dose — MAP trending below 65 despite fluid resuscitation over the next 4 hours"), and must reference specific trajectory evidence that justifies it. Do not recommend actions already taken correctly. 
If no further recommendations can be made, write null.


## OUTPUT FORMAT
Output your structured response inside <response></response> tags as valid JSON:

<response>
{
  "patient_status": {
    "domains": {
      "hemodynamics":    {"label": "...", "key_signals": ["..."], "rationale": "..."},
      "respiratory":     {"label": "...", "key_signals": ["..."], "rationale": "..."},
      "renal_metabolic": {"label": "...", "key_signals": ["..."], "rationale": "..."},
      "neurology":       {"label": "...", "key_signals": ["..."], "rationale": "..."}
    },
    "overall": {
      "label": "<improving | stable | deteriorating | insufficient_data>",
      "rationale": "<1–3 sentence summary grounded in trajectory evidence>"
    }
  },
  "action_evaluations": [
    {
      "action_id": "<identifier for the action, e.g., CW1 for current window event 1>",
      "action_description": "<brief restatement of the action for clarity>",
      "guideline_adherence": {
        "label": "<adherent | non_adherent | not_applicable | guideline_unclear>",
        "guideline_reference": "<name or source of the relevant guideline, or null>",
        "rationale": "<1–2 sentences>"
      },
      "contextual_appropriateness": {
        "label": "<appropriate | suboptimal | potentially_harmful | not_enough_context>",
        "rationale": "<1–3 sentences referencing patient-specific trajectory evidence>",
        "hindsight_caveat": "<note if hindsight changes the interpretation vs. real-time judgment, or null>",
      },
      "overall": {
        "label": "<appropriate | suboptimal | potentially_harmful | not_enough_context>",
        "rationale": "<1–3 sentence synthesis of the above evaluations, grounded in trajectory evidence>"
      }
    }
  ],
  "recommendations": [
    {
      "rank": 1,
      "action": "<a short name of the recommended action>",
      "action_description": "<1-2 sentences of the concrete, actionable clinical recommendation>",
      "rationale": "<1–2 sentences grounded in specific hindsight trajectory evidence>",
      "urgency": "<immediate | within_1h | longterm | optional>"
    }, // Or null if no recommendations
  ],
  "overall_window_summary": "<2–4 sentence synthesis of the window: patient direction, care quality, and any critical observations>"
}
</response>


## PATIENT ICU CONTEXT WINDOW
{patient_icu_trajectory}

Now, evaluate the CURRENT OBSERVATION WINDOW according to the instructions above.
""".strip()


def format_oracle_prompt(
    window_data: Dict[str, Any],
    context_block: str,
    context_mode: str,
    history_hours: Any = None,
    future_hours: Any = None,
    top_k: Any = None,
    include_icu_outcome: bool = True,
) -> str:
    """Format Oracle prompt with local ICU context window + current raw window."""
    metadata = window_data.get("patient_metadata", {})
    window_time = _format_window_time(window_data)
    current_hour_since_admission = _safe_float(window_data.get("hours_since_admission"))
    icu_duration_hours = _safe_float(metadata.get("total_icu_duration_hours"))
    outcome_text = _format_outcome(metadata.get("survived"))
    history_hours_text = _format_hours_for_template(history_hours)
    future_hours_text = _format_hours_for_template(future_hours)
    top_k_text = _format_top_k(top_k)

    patient_context = [
        "## Patient Context",
        f"- Age: {_format_age(metadata.get('age'))}",
        f"- Total ICU Stay: {icu_duration_hours:.1f} hours",
        f"- Current Hour Since ICU Admission: {current_hour_since_admission:.1f}",
        f"- Context Mode: {context_mode}",
        f"- Current Window Start: {window_data.get('current_window_start')}",
        f"- Current Window End: {window_data.get('current_window_end')}",
        "",
    ]
    if include_icu_outcome:
        patient_context.insert(4, f"- ICU Outcome: {outcome_text}")
    pre_icu_history_lines = _format_pre_icu_history_lines(window_data.get("pre_icu_history"))

    patient_trajectory = "\n".join([*patient_context, *pre_icu_history_lines, "", context_block]).strip()

    prompt = ORACLE_PROMPT_TEMPLATE
    prompt = prompt.replace("{history_hours}", history_hours_text)
    prompt = prompt.replace("{future_hours}", future_hours_text)
    prompt = prompt.replace("{top_k}", top_k_text)
    prompt = prompt.replace("{patient_icu_trajectory}", patient_trajectory)
    prompt = prompt.replace("{window_time}", window_time)
    # Backward-compatible token replacement for legacy templates.
    prompt = prompt.replace("{window time}", window_time)
    return prompt


def _format_pre_icu_history_lines(pre_icu_history: Any) -> List[str]:
    lines = ["## PRE-ICU HISTORY"]

    if not isinstance(pre_icu_history, dict):
        lines.append("No pre-ICU history provided.")
        return lines

    source = str(pre_icu_history.get("source") or "").strip().lower()
    items = int(_safe_float(pre_icu_history.get("items")))
    content = str(pre_icu_history.get("content") or "").strip()
    fallback_hours = _safe_float(pre_icu_history.get("fallback_hours"))

    if source in {"", "none", "disabled"}:
        lines.append("No pre-ICU history provided.")
    elif source == "reports":
        lines.append(content if content else "No pre-ICU history provided.")
    elif source == "events_fallback":
        lines.append(f"({items} item(s) from previous {fallback_hours:.1f} hours)")
        lines.append(content if content else "No pre-ICU history provided.")
    else:
        lines.append(f"Source: {source} ({items} item(s))")
        lines.append(content if content else "No pre-ICU history provided.")

    baseline_content = str(pre_icu_history.get("baseline_content") or "").strip()
    baseline_events_count = int(_safe_float(pre_icu_history.get("baseline_events_count")))
    baseline_history_hours = _safe_float(pre_icu_history.get("history_hours"))
    if baseline_history_hours <= 0:
        baseline_history_hours = _safe_float(pre_icu_history.get("fallback_hours"))

    if baseline_content:
        lines.append("")
        lines.append("## PRE-ICU BASELINE SNAPSHOT")
        lines.append(
            f"({baseline_events_count} LAB/VITAL event(s) from last {baseline_history_hours:.1f}h before ICU)"
        )
        lines.append(baseline_content)

    return lines


def format_event_line(event: Dict[str, Any]) -> str:
    """Format one event into a compact, stable line for prompts."""
    parts: List[str] = []

    time_value = event.get("time") or event.get("start_time")
    if not _is_missing_value(time_value):
        parts.append(_format_time(time_value))

    code = event.get("code")
    if not _is_missing_value(code):
        parts.append(str(code))

    details = event.get("code_specifics")
    if not _is_missing_value(details):
        parts.append(str(details))

    numeric_value = event.get("numeric_value")
    if not _is_missing_value(numeric_value):
        parts.append(f"={_format_numeric_value(numeric_value)}")

    text_value = event.get("text_value")
    if not _is_missing_value(text_value):
        parts.append(str(text_value))

    if not parts:
        return json.dumps(event, ensure_ascii=False, default=str)
    return " ".join(parts)


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "null", "none"}
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _format_age(age: Any) -> str:
    if age is None:
        return "Unknown"
    try:
        return f"{float(age):.1f} years"
    except (TypeError, ValueError):
        return str(age)


def _format_outcome(survived: Any) -> str:
    if isinstance(survived, bool):
        return "Survived after ICU" if survived else "Died after ICU"

    text = str(survived).strip().lower()
    if text in {"true", "1", "yes", "y", "survived", "alive"}:
        return "Survived after ICU"
    if text in {"false", "0", "no", "n", "died", "dead", "deceased"}:
        return "Died after ICU"
    return "Unknown"


def _format_hours_for_template(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def _format_top_k(value: Any) -> str:
    if value is None:
        return "3"
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        return str(value)
    if as_int < 1:
        as_int = 1
    return str(as_int)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_numeric_value(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _format_window_time(window_data: Dict[str, Any]) -> str:
    start = window_data.get("current_window_start")
    end = window_data.get("current_window_end")
    start_text = _format_time(start) if start else "unknown_start"
    end_text = _format_time(end) if end else "unknown_end"
    return f"{start_text} to {end_text}"


def _format_time(value: Any) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")

    text = str(value).strip()
    if not text:
        return "Unknown"

    try:
        parsed = datetime.fromisoformat(text)
        return parsed.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        pass

    if len(text) >= 16:
        return text[:16]
    return text
