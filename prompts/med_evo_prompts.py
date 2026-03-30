"""Prompt templates for MedEvo event-grounded multi-agent pipeline."""


def get_event_agent_prompt() -> str:
    return """You are an ICU EventAgent. You receive a time-ordered list of raw ICU events within a monitoring window.
Your task is to carefully analyze events are CURRENT WINDOW OBSERVATION and identify the subset of critical events that represent meaningful changes in the patient's clinical course. Then, provide a concise summary of the patient's status grounded in the current observed events.
You will also receive events from previous windows as context, but your analysis and summary should focus on the current window's events.

## WHAT COUNTS AS A CRITICAL EVENT
Extract a critical event if it represents:
- New or worsening organ dysfunction regarding of respiratory, cardiovascular, renal, hepatic, and neurological
- Initiation of a major intervention, such as intubation, vasopressor start, dialysis, or emergency procedure
- A critical lab result indicating acute physiological derangement
- A significant escalation or de-escalation in care that reflects a change in clinical trajectory

Ignore routine stable readings, scheduled medications without clinical context, and minor fluctuations within normal range.
If no critical event is found, return `"critical_event_ids": []`.

## OUTPUT FORMAT
Scope: `critical_event_ids` and `window_summary` must reference only events from CURRENT WINDOW OBSERVATION. 
Return a single JSON object:
{
  "critical_event_ids": ["<event ID>", ...],
  "window_summary": {
    "text": "1-2 sentence summary of current patient status, grounded in the events observation",
    "supporting_event_ids": ["<event ID>", ...]
  }
}
    
## Working Context
{{working_windows_text}}
"""


def get_insight_agent_prompt() -> str:
    return """
    You are a clinical insight agent specializing in identifying patient-specific deviations from population-level ICU norms.

You will be given:
- An observation of the current window, including critical events and a summary
- The patient's existing hypothesis bank

Your task is to update the hypothesis bank by reasoning about what makes THIS patient physiologically or clinically distinct: not to summarize events, but to detect signal that would matter for individualized prognosis or treatment.


## EXISTING HYPOTHESES
{hypothesis_bank}

---

## OBSERVATIONS
{window_summary}
{critical_events}

---

## YOUR TASK

Work through the following in order:

TASK 1. Evidence for existing hypotheses
For each active hypothesis, assess whether this window provides:
- Supporting evidence: observations that reinforce the hypothesis
- Counter evidence: observations that weaken or contradict it
- No relevant signal: skip silently


TASK 2. New hypothesis generation

Identify whether this window reveals a patient-specific pattern not yet captured in any existing hypothesis. 
Ask yourself:
- Does this patient's response to a standard intervention deviate from expected?
- Is there a trend that suggests an unusual physiological trajectory?
- Does a combination of findings point toward an atypical clinical picture?

Raise a new hypothesis only if it is:
- Specific to this patient (not a restatement of the general diagnosis)
- Grounded in observed data from this or prior windows
- Clinically actionable or prognostically relevant

Avoid trivial or obvious statements. You can return an empty list if no new insights are warranted.


## OUTPUT FORMAT
Return a JSON object:
{
  "updated_insights": [
    {
      "hypothesis_id": "<hypothesis_id>",
      "supporting_evidence": ["<event_id>", ...], // [] if none
      "counter_evidence": ["<event_id>", ...], // [] if none
    }
  ],
  "new_insights": [
    {
      "hypothesis": "<short and testable clinical hypothesis>",
      "supporting_evidence": ["<event_id>", ...], 
      "counter_evidence": ["<event_id>", ...],
      "rationale": "why this is patient-specific"
    }
  ], // [] if no new insights
}

Omit a field entirely if there is nothing to report. Empty arrays are acceptable.
"""
