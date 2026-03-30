"""Prompt templates for Oracle offline evaluator."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from utils.event_format import (
    format_event_line as format_shared_event_line,
    format_event_lines as format_shared_event_lines,
)
from utils.time_format import format_timestamp_minute

ORACLE_PROMPT_TEMPLATE = """
You are Oracle, a clinical AI evaluator with hindsight access to a patient's full ICU trajectory including the ICU results.
Your task is to evaluate a specific local observation window {window_time} using the hindsight knowledge of the ICU trajectory. Depending on the context, your hindsight can come from ICU discharge summaries or future ICU events. 
For evaluating this observation window you must produce three tasks:

## PART 1 — PATIENT STATUS
Using the provided local context, assess the patient's clinical direction at the time of this window across four domains, then estimate an overall status.

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

Synthesize the overall label by reasoning about the relative clinical weight of each domain for this specific patient. 

Important nuances:
- Do NOT conflate outcome with care quality. A patient may be deteriorating despite excellent care. 
- A patient may be labelled stable or improving even if they eventually die, if the trajectory at this window genuinely reflects that direction.

## PART 2 — DOCTOR ACTION EVALUATION
You will be given a list of clinical actions taken during the {window_time} window.
Evaluate EACH action individually on two dimensions:

A. Guideline adherence — does this action follow established ICU clinical guidelines (e.g., Surviving Sepsis Campaign, ARDSNet, PADIS guidelines, AHA/ACC where relevant)?
Labels: adherent | non_adherent | not_applicable | guideline_unclear

B. Contextual appropriateness — given THIS patient's specific condition, trajectory, comorbidities, and the eventual outcome known to you, was this action appropriate?
Labels: appropriate | suboptimal | potentially_harmful | insufficient_data

Use your hindsight knowledge from the provided context to inform contextual appropriateness, but be fair: judge the action against what the context reveals, not against impossible foresight.
If an action was reasonable at the time but later context revealed a missed diagnosis, note this nuance.

## PART3 — CLINICAL RECOMMENDATIONS

Provide up to {top_k} recommendations for what the clinical team should have prioritised at this window, grounded in hindsight knowledge of the trajectory.

Requirements:
- Concrete and actionable
- Grounded in trajectory evidence (can include hindsight)
- Focus on what would have most improved patient outcome

You may:
- Reinforce correct actions
- Suggest missing interventions
- Prioritize more impactful alternatives

Ranking criteria:
1. Expected impact on outcome
2. Urgency (urgent, nice_to_have, or optional)
3. Strength of supporting evidence

If no meaningful recommendation exists, return an empty list [].

## OUTPUT FORMAT
Output in valid JSON format:
{
  "patient_status": {
    "domains": {
      "hemodynamics":    {"label": "...", "key_evidence": ["<1-3 event IDs as the key evidence from the observation>"]},
      "respiratory":     {"label": "...", "key_evidence": ["..."]},
      "renal_metabolic": {"label": "...", "key_evidence": ["..."]},
      "neurology":       {"label": "...", "key_evidence": ["..."]}
    },
    "overall": {
      "label": "<improving | stable | deteriorating | insufficient_data>",
      "rationale": "<1-2 sentence explaining the overall patient status grounded in trajectory evidence>"
    }
  },
  "action_evaluations": [
    {
      "action_id": "<ICU-local event ID within this ICU stay (0-based), e.g., 12>",
      "action_name": "<action name corresponding to the action_id>",
      "guideline_adherence": {
        "label": "<adherent | non_adherent | not_applicable | guideline_unclear>",
        "guideline_reference": "<name or source of the relevant guideline, or null>",
      },
      "contextual_appropriateness": {
        "label": "<appropriate | suboptimal | potentially_harmful | insufficient_data>",
        "hindsight_usage": "<brief note on whether hindsight changed interpretation>",
      },
      "overall": {
        "label": "<appropriate | suboptimal | potentially_harmful | insufficient_data>",
        "rationale": "<1-2 sentence integrating both dimensions>"
      }
    }
  ],
  "recommendations": [
    {
      "rank": 1,
      "action_name": "<name of the recommended action>",
      "action_description": "<short description of the concrete, actionable clinical recommendation>",
      "urgency": "<urgent | nice_to_have | optional>",
      "key_evidence": ["<1-3 event IDs as the key evidence from the observation>"],
    },
  ], // Or [] if no recommendations
  "overall_window_summary": "<1-2 sentence covering the patient direction, care quality, and any critical observations>"
}


## PATIENT ICU CONTEXT WINDOW
{patient_icu_trajectory}

Now, evaluate the CURRENT OBSERVATION WINDOW according to the instructions above.
""".strip()


def format_pre_icu_compression_prompt(pre_icu_history: Dict[str, Any], max_summary_chars: int = 1800) -> str:
    """Build prompt for one-shot pre-ICU compression reused across all windows."""
    payload = {
        "source": str(pre_icu_history.get("source") or "").strip(),
        "items": int(_safe_float(pre_icu_history.get("items"))),
        "historical_discharge_summary_items": int(
            _safe_float(pre_icu_history.get("historical_discharge_summary_items"))
        ),
        "report_content": str(pre_icu_history.get("content") or "").strip(),
        "events_count": int(_safe_float(pre_icu_history.get("baseline_events_count"))),
        "events_content": str(pre_icu_history.get("baseline_content") or "").strip(),
    }
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False)
    return f"""You compress pre-ICU history for repeated ICU window evaluation prompts.

Return JSON only:
{{
  "compressed_pre_icu_history": "Concise summary (<= {max_summary_chars} chars)."
}}

Requirements:
- Preserve clinically important signal; remove boilerplate.
- Focus on:
  1) Baseline vulnerability (major comorbidities, frailty, immunosuppression)
  2) Pre-ICU trajectory (course, worsening/improving, response to treatments)
  3) Acute severity signals (organ dysfunction, critical abnormal findings)
  4) Working diagnoses if evident

- Prefer interpreted clinical states over raw numbers (e.g. 'worsening hypoxia despite O2').
- Pre-ICU event lines may carry history IDs like `[H0]`, `[H1]`, ... where `H` means pre-ICU history.
- If citing specific pre-ICU events in the compressed text, keep those `[H#]` references.
- Mention uncertainty if source text is sparse or conflicting.
- Do not invent events or outcomes not present in input.
- Keep the summary reusable across all subsequent ICU windows.
- Include critical patient-specific constraints (e.g.treatment-limiting allergies) if present.

Pre-ICU history payload:
{payload_json}
"""


# ORACLE_PROMPT_TEMPLATE = """
# You are Oracle, a clinical AI evaluator with hindsight access to a patient's full ICU trajectory, including future events and discharge outcomes.

# Your task is to evaluate a specific observation window {window_time}.

# You must complete three parts:
# 1) Patient Status (NO hindsight)
# 2) Doctor Action Evaluation (hindsight allowed)
# 3) Clinical Recommendations (hindsight allowed)

# --------------------------------------------------
# ## INFORMATION USAGE RULES (CRITICAL)

# - PART 1 (Patient Status):
#   Use ONLY information available up to and within the current observation window.
#   DO NOT use future events, outcomes, or discharge summaries.

# - PART 2 (Action Evaluation):
#   Use the window context for primary judgment.
#   You MAY use hindsight to refine contextual appropriateness, but:
#     - Do NOT penalize actions that were reasonable given available information.
#     - Only downgrade actions if harm or error was reasonably foreseeable.

# - PART 3 (Recommendations):
#   You MAY fully use hindsight knowledge of the trajectory.

# - General rule:
#   Do NOT directly infer current status from eventual outcomes.

# --------------------------------------------------
# ## PART 1 — PATIENT STATUS

# Assess the patient's short-term clinical direction (~2–6 hour horizon around the window) across four domains:

# Domains:
# - Hemodynamics: heart rate, MAP, lactate, perfusion, vasopressor requirements
# - Respiratory: SpO2, PaO2/FiO2, respiratory rate, ventilator settings
# - Renal/Metabolic: creatinine, urine output, electrolytes, acid-base status
# - Neurology: GCS, mental status, RASS

# For each domain and overall status, choose exactly one:
# - improving: clinically meaningful movement toward stability/recovery
# - stable: no meaningful directional change
# - deteriorating: clinically meaningful worsening or decompensation
# - insufficient_data: signal is too sparse or conflicting

# Trend definition:
# - Based on changes within the window and recent prior context
# - Use clinically meaningful directionality (e.g., rising lactate, increasing vasopressor need)
# - Prefer "stable" or "insufficient_data" over weak inference

# Overall status:
# - Synthesize based on relative clinical importance of domains
# - Do NOT use outcome knowledge

# --------------------------------------------------
# ## PART 2 — DOCTOR ACTION EVALUATION

# You will be given a list of actions within the window.
# Evaluate EACH action independently.

# ### A. Guideline Adherence
# Does the action follow established ICU guidelines (e.g., Surviving Sepsis Campaign, ARDSNet, PADIS, AHA/ACC)?

# Labels:
# - adherent
# - non_adherent
# - not_applicable
# - guideline_unclear

# ### B. Contextual Appropriateness
# Was the action appropriate for THIS patient at that time?

# Labels:
# - appropriate
# - suboptimal
# - potentially_harmful
# - insufficient_data

# Hindsight usage rule:
# - If an action was reasonable given available information → label "appropriate"
# - Only downgrade if:
#   - a high-probability condition was missed
#   - or harm was reasonably foreseeable from available signals

# ### C. Overall Action Quality
# Integrate BOTH guideline adherence and contextual appropriateness:

# Labels:
# - appropriate → appropriate OR justified deviation from guidelines
# - suboptimal → minor issue, delay, or inefficiency
# - potentially_harmful → clear harm or major error
# - insufficient_data

# --------------------------------------------------
# ## PART 3 — CLINICAL RECOMMENDATIONS

# Provide up to {top_k} recommendations.

# Requirements:
# - Concrete and actionable
# - Grounded in trajectory evidence (can include hindsight)
# - Focus on what would have most improved patient outcome

# You may:
# - Reinforce correct actions
# - Suggest missing interventions
# - Prioritize more impactful alternatives

# Ranking criteria:
# 1. Expected impact on outcome
# 2. Urgency (time sensitivity)
# 3. Strength of supporting evidence

# If no meaningful recommendation exists, return an empty list [].

# Urgency labels:
# - urgent
# - nice_to_have
# - optional

# --------------------------------------------------
# ## OUTPUT FORMAT (STRICT JSON)

# Output JSON:
# {
#   "patient_status": {
#     "domains": {
#       "hemodynamics": {
#         "label": "...",
#         "key_evidence": [{"type": "vital|lab|event|action", "id": "..."}]
#       },
#       "respiratory": {
#         "label": "...",
#         "key_evidence": [{"type": "...", "id": "..."}]
#       },
#       "renal_metabolic": {
#         "label": "...",
#         "key_evidence": [{"type": "...", "id": "..."}]
#       },
#       "neurology": {
#         "label": "...",
#         "key_evidence": [{"type": "...", "id": "..."}]
#       }
#     },
#     "overall": {
#       "label": "<improving | stable | deteriorating | insufficient_data>",
#       "rationale": "<1-2 sentences based ONLY on window evidence>"
#     }
#   },
#   "action_evaluations": [
#     {
#       "action_id": "...",
#       "action_name": "...",
#       "guideline_adherence": {
#         "label": "<adherent | non_adherent | not_applicable | guideline_unclear>",
#         "guideline_reference": "<guideline name or null>"
#       },
#       "contextual_appropriateness": {
#         "label": "<appropriate | suboptimal | potentially_harmful | insufficient_data>",
#         "hindsight_usage": "<brief note on whether hindsight changed interpretation>"
#       },
#       "overall": {
#         "label": "<appropriate | suboptimal | potentially_harmful | insufficient_data>",
#         "rationale": "<1-2 sentences integrating both dimensions>"
#       }
#     }
#   ],
#   "recommendations": [
#     {
#       "rank": 1,
#       "action_name": "...",
#       "action_description": "...",
#       "urgency": "<urgent | nice_to_have | optional>",
#       "key_evidence": [{"type": "vital|lab|event|action", "id": "..."}]
#     }
#   ],
#   "overall_window_summary": "<1-2 sentences covering patient trajectory, care quality, and key observations>"
# }

# --------------------------------------------------
# ## PATIENT ICU CONTEXT
# {patient_icu_trajectory}

# Now evaluate the CURRENT OBSERVATION WINDOW.
# """.strip()


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
        f"- Gender: {_format_gender(metadata.get('gender'))}",
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
    lines = ["## HISTORICAL PRE-ICU SUMMARY"]

    if not isinstance(pre_icu_history, dict):
        lines.append("No historical pre-ICU reports provided.")
        return lines

    source = str(pre_icu_history.get("source") or "").strip().lower()
    items = int(_safe_float(pre_icu_history.get("items")))
    content = str(pre_icu_history.get("content") or "").strip()
    fallback_hours = _safe_float(pre_icu_history.get("fallback_hours"))

    if source in {"", "none", "disabled"}:
        lines.append("No historical pre-ICU reports provided.")
    elif source == "llm_compressed":
        lines.append(content if content else "No historical pre-ICU reports provided.")

    elif source == "reports":
        lines.append(content if content else "No historical pre-ICU reports provided.")
    elif source == "events_fallback":
        lines.append(f"({items} item(s) from previous {fallback_hours:.1f} hours)")
        lines.append(content if content else "No historical pre-ICU reports provided.")
    else:
        lines.append(f"Source: {source} ({items} item(s))")
        lines.append(content if content else "No historical pre-ICU reports provided.")

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
    """Format one event line for prompt context blocks."""
    return format_shared_event_line(event)


def format_event_lines(events: List[Dict[str, Any]], *, empty_text: str = "(No events)") -> List[str]:
    """Format multiple event lines for prompt context blocks."""
    return format_shared_event_lines(events, empty_text=empty_text)


def _format_age(age: Any) -> str:
    if age is None:
        return "Unknown"
    try:
        return f"{float(age):.1f} years"
    except (TypeError, ValueError):
        return str(age)


def _format_gender(gender: Any) -> str:
    text = str(gender).strip() if gender is not None else ""
    if not text:
        return "Unknown"
    normalized = text.lower()
    if normalized in {"none", "nan", "null", "nat", "unknown"}:
        return "Unknown"
    if normalized in {"m", "male"}:
        return "Male"
    if normalized in {"f", "female"}:
        return "Female"
    return text


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


def _format_window_time(window_data: Dict[str, Any]) -> str:
    start = window_data.get("current_window_start")
    end = window_data.get("current_window_end")
    start_text = _format_time(start) if start else "unknown_start"
    end_text = _format_time(end) if end else "unknown_end"
    return f"{start_text} to {end_text}"


def _format_time(value: Any) -> str:
    return format_timestamp_minute(value)
