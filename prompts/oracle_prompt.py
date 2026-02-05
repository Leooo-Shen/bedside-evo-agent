"""
Oracle System Prompt for Retrospective Clinical Evaluation

This prompt template is used by the Meta Oracle to evaluate clinical decisions
with the benefit of hindsight.
"""

ORACLE_SYSTEM_PROMPT = """
## Role
You are a Senior Retrospective Clinical Auditor. Your objective is to perform a high-fidelity "Root Cause Analysis" (RCA) and quality assessment of patient care within a specific time window. You possess "hindsight clarity," meaning you evaluate decisions made at Time T by analyzing the patient's subsequent clinical trajectory.

## Objective
Determine if clinical interventions during the Evaluation Window (T to T + {window_hours} hours) were evidence-based and appropriate, using Future Events to validate the efficacy of those decisions while strictly avoiding "Outcome Bias" (i.e., not blaming clinicians for poor outcomes that were physiologically inevitable).


## 1. Data Input Structure
You will be provided with four distinct data blocks. You must maintain strict traceability by referencing events by their indices:
* **Patient Context**: Demographics, comorbidities, and admission etiology.
* **History (Hn)**: Events preceding the evaluation window.
* **Current Window (Cn)**: The {window_hours}-hour period currently under audit.
* **Future Trajectory (Fn)**: Events occurring after the window, used to verify the impact of Cn decisions.


## 2. Evaluation Framework

### A. Patient Status Score
Assign a value from -1.0 to +1.0 based on the patient’s physiological stability at the end of the Current Window:
* -1.0 (Critical): Active decompensation; failure of multiple organ systems.
* -0.5 (Unstable): Trending toward deterioration; escalating support required.
* 0.0 (Stable): Homeostasis maintained; no immediate threat to life.
* +0.5 (Improving): Resolving pathology; weaning of support.
* +1.0 (Recovered): Baseline status achieved or ready for step-down/discharge.

### B. Action Quality Categorization
Audit the interventions in the Current Window (Cn) against the Standard of Care:
* **Optimal**: Decisions were proactive, evidence-based, and addressed the primary pathology. Future data (Fn) confirms these actions stabilized or mitigated risks.
* **Neutral**: Care was routine or observational. No critical interventions were indicated, or actions had no measurable impact on the trajectory.
* **Sub-optimal**: Actions deviated from clinical guidelines, ignored significant trends in Hn or Cn, or represented missed opportunities that led to preventable complications in Fn.


## 3. Logic & Constraint Guidelines
* **The Hindsight Rule**: Use the Future Trajectory (Fn) to diagnose the correctness of a decision, but evaluate the quality of the clinician's choice based on the data available at Time T.
* **Evidence-Based Traceability**: Every assertion in your rationale must cite at least one index (Hn, Cn, Fn).
* **Physiological Justification**: Link your assessment to specific clinical markers (e.g., MAP, SpO2, Serum Lactate, GCS).
* **Non-Determinism**: A "Sub-optimal" action might still result in a "Positive Outcome" due to patient resilience, and an "Optimal" action might result in a "Negative Outcome" due to disease severity.


## 4. Output Format
Return the analysis strictly as a JSON object:
```json
{{
  "audit_metadata": {{
    "primary_clinical_driver": "Short description of the main medical issue"
  }},
  "patient_status": {{
    "score": 0.0,
    "rationale": "Detailed reasoning referencing indices (e.g., C4, F2) and physiological trends."
  }},
  "clinical_quality": {{
    "rating": "optimal | neutral | sub-optimal",
    "rationale": "Comprehensive evaluation of interventions. Connects Current Window actions (Cn) to Future outcomes (Fn).",
    "guideline_adherence": "Reference to standard protocols (e.g., Surviving Sepsis, ACLS, etc.)"
  }},
  "clinical_pearl": "A generalizable, high-value takeaway for medical education."
}}
```
**The patient case for evaluation follows:**

"""


def format_oracle_prompt(window_data: dict, window_hours: float = 6.0, include_outcome: bool = True) -> str:
    """
    Format the Oracle prompt with specific patient window data.

    Args:
        window_data: Dictionary containing patient metadata, history, and future events
        window_hours: Size of the future window in hours
        include_outcome: Whether to include patient outcome in the prompt (default: True)

    Returns:
        Formatted prompt string ready for LLM
    """

    # Extract patient metadata
    metadata = window_data["patient_metadata"]

    # Format patient context
    patient_context = f"""
## Patient Context

- Age: {metadata['age']:.1f} years
- Hours Since Admission: {window_data['hours_since_admission']:.1f}
- Evaluation Window: {window_data['current_window_start']} to {window_data['current_window_end']}
- Total ICU Duration: {metadata['total_icu_duration_hours']:.1f} hours

"""

    # Only include outcome if requested (for blinded evaluation)
    if include_outcome:
        patient_context += f"- Outcome: {'Survived' if metadata['survived'] else 'Died'}\n"

        if metadata["death_time"]:
            patient_context += f"- Death Time: {metadata['death_time']}\n"

    # Format history events
    history_summary = f"\n## History Events (Before Time T)\n\n"
    history_summary += f"Total events in history: {window_data['num_history_events']}\n\n"

    if window_data["num_history_events"] > 0:
        # Show all history events
        all_history = window_data["history_events"]
        history_summary += "All events:\n"
        for i, event in enumerate(all_history, 1):
            history_summary += f"H{i}. {_format_event(event)}\n"
    else:
        history_summary += "No prior events (evaluation at admission)\n"

    # Format current events
    current_summary = f"\n## Current Events (Evaluation Window from Time T to T+{window_hours} hours)\n\n"
    current_summary += f"Total events in current window: {window_data['num_current_events']}\n\n"

    if window_data["num_current_events"] > 0:
        for i, event in enumerate(window_data["current_events"], 1):
            current_summary += f"C{i}. {_format_event(event)}\n"
    else:
        current_summary += "No events in current evaluation window\n"

    # Format future events
    future_summary = f"\n## Future Events (After T+{window_hours} hours)\n\n"
    future_summary += f"Total events in future window: {window_data['num_future_events']}\n\n"

    if window_data["num_future_events"] > 0:
        for i, event in enumerate(window_data["future_events"], 1):
            future_summary += f"F{i}. {_format_event(event)}\n"
    else:
        future_summary += "No events in future window\n"

    # Combine everything
    full_prompt = ORACLE_SYSTEM_PROMPT.format(window_hours=window_hours)
    full_prompt += "\n\n" + patient_context
    full_prompt += "\n" + history_summary
    full_prompt += "\n" + current_summary
    full_prompt += "\n" + future_summary
    full_prompt += "\n\n## Your Evaluation\n\nProvide your evaluation as a JSON object:"

    return full_prompt


def _format_event(event: dict) -> str:
    """
    Format a single cleaned event for display in the prompt.

    Args:
        event: Cleaned event dictionary with filtered fields

    Returns:
        Formatted event string
    """
    parts = []

    # Time (always first if available)
    if "start_time" in event:
        # Format timestamp without seconds (YYYY-MM-DD HH:MM:SS -> YYYY-MM-DD HH:MM)
        timestamp = event["start_time"]
        if len(timestamp) >= 19 and timestamp[16] == ":":  # Check if format includes seconds
            timestamp = timestamp[:16]  # Remove :SS part
        parts.append(f"[{timestamp}]")

    # Time delta (if available)
    if "time_delta_minutes" in event:
        parts.append(f"(+{event['time_delta_minutes']}min)")

    # Code specifics (label/description)
    if "code" in event:
        parts.append(f"{event['code']}")

    if "code_specifics" in event:
        parts.append(event["code_specifics"])

    # Numeric value with text value (unit)
    if "numeric_value" in event:
        value_str = str(event["numeric_value"])
        parts.append(f"={value_str}")

    if "text_value" in event:
        parts.append(f"{event['text_value']}")

    # # End time (if different from start)
    # if "end_time" in event and event.get("end_time") != event.get("time"):
    #     parts.append(f"(until {event['end_time']})")

    return " ".join(parts) if parts else str(event)
