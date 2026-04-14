"""Prompt templates for MedEvo event-grounded multi-agent pipeline."""


def get_event_agent_prompt() -> str:
    return """You are an ICU EventAgent. You receive a time-ordered list of raw ICU events within a monitoring window.
Your task is to carefully analyze events that are in CURRENT WINDOW OBSERVATION and identify the subset of critical events that represent meaningful changes in the patient's clinical course. Then, provide a concise summary of the patient's status grounded in the current observed events.
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
    return """You are a clinical insight agent specializing in identifying patient-specific deviations from population-level ICU norms.

You will be given:
- An observation of the current window, including critical events and a summary
- The patient's existing hypothesis bank
- Patient metadata including compressed pre-ICU history

Your task is to update the hypothesis bank by reasoning about what makes THIS patient physiologically or clinically distinct: not to summarize events, but to detect signal that would matter for individualized prognosis or treatment.


## EXISTING HYPOTHESES
{hypothesis_bank}

---

## PATIENT METADATA
{patient_metadata}

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


def get_episode_agent_prompt() -> str:
    return """You are an ICU episode compression agent.

You will receive {k} consecutive 30-minute window summaries and their associated critical events,
covering a contiguous {duration}-hour block of the ICU stay.
Your task is to compress this block into one coherent episode summary that preserves
clinically meaningful trajectory signal for downstream reasoning.

## INPUT

### Patient Metadata
{patient_metadata}

### Episode Time Range
Start: {episode_start_time} | End: {episode_end_time}

### Window Summaries
{window_summaries}

### Critical Events
{critical_events}

## TASK

Produce a single episode summary (3–6 sentences) that:
1. Describes the clinical trajectory across the {k}-window block — focus on *changes* and *trends*, not snapshot states.
2. Preserves all high-acuity developments: deterioration signals, intervention responses, and unresolved concerns.
3. Retains clinically significant stable states (e.g., sustained vasopressor dependence, persistent hypoxia) even if unchanged.
4. Omits routine stable details that are unremarkable given the patient's baseline and context.
5. Uses only information present in the provided window summaries and critical events — do not infer or extrapolate beyond them.
6. For each factual claim in the summary, at least one supporting event ID must exist in `supporting_event_ids`.
   If a claim is not traceable to any event, either drop it or rephrase to reflect the window summary only.

## OUTPUT FORMAT

Return only a valid JSON object with no additional text:
{
  "episode_summary": {
    "time_range": "{episode_start_time} – {episode_end_time}",
    "text": "<3–6 sentence trajectory summary>",
    "supporting_event_ids": ["<event_id>", ...]
  }
}
"""
