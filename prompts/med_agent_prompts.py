"""Prompt templates for MedAgent (static + dynamic memory)."""

from prompts.shared_prompts import get_prediction_prompt


def get_dynamic_memory_update_prompt(
    use_thinking: bool = True,
    max_active_problems: int = 8,
    max_critical_events: int = 20,
    max_patterns: int = 8,
) -> str:
    """Prompt for dynamic memory maintenance."""
    thinking_section = (
        """ALWAYS start with thinking first in <think></think>. Then provide your response in JSON format in <response></response>.
<think>
1. Summarize the current status from the raw events.
2. Update active problems by carrying forward unresolved issues and adding new clinically meaningful problems.
3. Keep only high-signal events and trends.
4. Preserve patient-specific patterns that repeatedly influence decisions.
</think>

"""
        if use_thinking
        else "Provide your response in JSON format in <response></response>.\n\n"
    )

    return f"""You are a clinical memory management agent for ICU patients. Your role is to maintain and update a structured patient memory by processing new events from a sliding window, integrating them with existing static and dynamic memory.

## Input Format

You will receive:
1. **Static Memory**: Fixed patient information captured at admission 
2. **Current Dynamic Memory**: The most recent dynamic memory snapshot
3. **New Events**: Raw ICU events from the current time window

## Your Task

Analyze the new events and produce an **updated Dynamic Memory** that is:
- Accurate: reflects current clinical state
- Concise: no redundant or outdated information
- Clinically meaningful: highlights what matters for decision-making

## Static Memory
{{static_memory_text}}

## Previous Dynamic Memory
{{previous_dynamic_memory_text}}

## Current Events (Window {{window_index}}, Hour {{hours_since_admission:.1f}})
{{current_events_text}}

## Output Format

{thinking_section}<response>
{{{{
  "updated_dynamic_memory": {{{{
    "current_status": "One concise sentence of current overall status",
    "trends": {{{{
      "parameter (e.g. blood pressure)": [value1, value2, ...],
    }}}}
    "interventions_responses": {{{{
      "intervention (e.g. vasopressor)": "response observed in this window, or None if no response yet",
    }}}}
    "patient_specific_patterns": 
    [patient-specific clinical insights that can inform future decisions, supported by evidence from the events],
  }}}},
  "active_concerns": [
      {{{{
        "id": "String ID",
        "concern": "Brief description of the concern with update on progression or resolution",
        "status": "Active / Resolved"
      }}}},
      ...
    ]
  "critical_events": [
      {{{{
        "time": "YY-MM-DD HH:MM, or None if no critical events",
        "event": "Event name from current window. Write None if no critical events.",
        "significance": "Why this matters clinically."
      }}}},
      ...
    ],
}}}}
</response>
    
    
## Update Rules

### What to ADD:
- New critical events that cross clinical significance thresholds
- New lab/vital trends that show directional change
- New interventions with observable responses
- Newly identified problems or diagnoses
- New patient-specific patterns supported by evidence

### What to UPDATE:
- Current Status: always rewrite to reflect most recent state
- Active Problems: revise severity, stage, and trajectory based on new data
- Trends: append the latest value; update interpretation
- Interventions & Responses: add response data if a prior intervention now has an outcome

### What to REMOVE:
- Resolved problems
- Outdated trend values beyond clinical utility (keep last 3–5 data points per parameter)
- Do NOT delete Critical Events entries — the log is append-only

    
"""


def get_med_predictor_prompt(use_thinking: bool = True, observation_hours: float = 12.0) -> str:
    """Prompt for MedAgent predictor."""
    return get_prediction_prompt(use_thinking=use_thinking, observation_hours=observation_hours)
