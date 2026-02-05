"""
Memory Management Prompts for Managed Clinical Memory System

This module contains prompts for:
1. Extraction: Extract clinical insights from raw data
2. Consolidation: Update memory by merging/resolving entries
"""

from typing import Dict, List


def format_extraction_prompt(current_events: List[Dict], hours_since_admission: float) -> str:
    """
    Format prompt for extracting ONE clinical insight from current window.

    Args:
        current_events: Events from the current 30-minute window
        hours_since_admission: Current time in hours since ICU admission

    Returns:
        Formatted extraction prompt
    """
    # Format events
    events_str = _format_events(current_events)

    prompt = f"""You are a Senior ICU Physician. Extract ONE concise clinical insight from the last 30 minutes of data.

## Current Time
Hours since ICU admission: {hours_since_admission:.1f}

## Current Observations (Last 30 minutes)
{events_str}

## Extraction Rules
1. Identify if a primary organ system (Hemodynamics, Respiratory, Renal, or Neurology) is improving, stable, or failing.
2. If labs show a return to normal ranges, explicitly note it as a "Resolution."
3. Distinguish safety interventions (e.g., restraints) from life-support interventions (e.g., ventilators).
4. Focus on the MOST SIGNIFICANT clinical trend in this window.

## Output Format
Provide your response in the following JSON format:
{{
  "system": "Hemodynamics/Respiratory/Renal/Neurology",
  "observation": "1-sentence summary of trend/status",
  "status": "ACTIVE/RESOLVED"
}}

Examples:
- {{"system": "Renal", "observation": "Initial hyperkalemia (K+ 6.2) corrected; current potassium 4.8.", "status": "RESOLVED"}}
- {{"system": "Neurology", "observation": "ICU Delirium present; requiring restraints for safety. GCS remains high at 14.", "status": "ACTIVE"}}
- {{"system": "Hemodynamics", "observation": "Blood pressure stabilizing with vasopressor support; MAP consistently above 65.", "status": "ACTIVE"}}

Note:
- Use "ACTIVE" for ongoing concerns or interventions
- Use "RESOLVED" when a previously abnormal parameter has returned to normal/safe ranges
"""

    return prompt


def format_consolidation_prompt(
    existing_memory: str,
    new_insight: Dict,
    hours_since_admission: float,
    max_entries: int = 5,
) -> str:
    """
    Format prompt for consolidating new insight into existing memory.

    Args:
        existing_memory: Formatted string of existing memory entries
        new_insight: New insight from extraction (dict with system, observation, status)
        hours_since_admission: Current time in hours
        max_entries: Maximum number of entries to maintain

    Returns:
        Formatted consolidation prompt
    """
    new_insight_str = f"""System: {new_insight.get('system', 'Unknown')}
Observation: {new_insight.get('observation', 'No observation')}
Status: {new_insight.get('status', 'ACTIVE')}
Time: Hour {hours_since_admission:.1f}"""

    prompt = f"""You are a Clinical Memory Manager. Update the patient's memory context using the [New Insight].

## Existing Memory
{existing_memory}

## New Insight to Integrate
{new_insight_str}

## Consolidation Rules
1. **Deduplicate**: If the New Insight matches an Existing Entry (same organ system), update the "Last Updated" time and refresh the description.
2. **Resolve**: If the New Insight shows a previously reported risk factor is now within safe/normal limits, change its status to 'RESOLVED'.
3. **Pruning**: Maintain a maximum of {max_entries} entries. Prioritize 'ACTIVE' threats over 'RESOLVED' ones.

## Your Task
Integrate the new insight and output the COMPLETE updated memory list.

Provide your response in the following JSON format:
{{
  "clinical_memory": [
    {{
      "id": <int>,
      "system": "Hemodynamics/Respiratory/Renal/Neurology",
      "status": "ACTIVE/RESOLVED",
      "description": "Clinical observation or trend",
      "last_updated": <float>
    }}
  ],
  "consolidation_action": "added_new/updated_existing/resolved_existing/pruned",
  "rationale": "Brief explanation of what you did"
}}

Important Notes:
- If updating an existing entry, keep its original ID
- If adding a new entry, assign the next available ID
- Sort entries with ACTIVE first, then by most recent last_updated
- If pruning is needed, remove RESOLVED entries first, then oldest ACTIVE entries
- Ensure the total number of entries does not exceed {max_entries}
"""

    return prompt


def _format_events(events: List[Dict], max_events: int = 50) -> str:
    """
    Format a list of events for display in prompts.

    Args:
        events: List of event dictionaries
        max_events: Maximum number of events to display

    Returns:
        Formatted string of events
    """
    if not events:
        return "No events recorded in this period."

    # Limit number of events
    display_events = events[:max_events]
    truncated = len(events) > max_events

    formatted = ""
    for event in display_events:
        time = event.get("start_time", "Unknown time")
        code = event.get("code_specifics", event.get("code", "Unknown"))
        value = event.get("numeric_value")
        text = event.get("text_value")

        formatted += f"- {time}: {code}"
        if value is not None:
            formatted += f" = {value}"
        if text:
            formatted += f" ({text})"
        formatted += "\n"

    if truncated:
        formatted += f"\n... and {len(events) - max_events} more events (truncated for brevity)\n"

    return formatted
