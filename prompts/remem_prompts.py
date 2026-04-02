"""ReMeM Agent Prompts.

All prompts for the ReMeM (Retrieval-Enhanced Memory Management) agent are centralized here.
These prompts implement the Think-Act-Refine loop for patient state tracking
and survival prediction.

Note: This is specific to the ReMeM method. Other memory management approaches
may use different prompting strategies.
"""

from typing import Dict, List

from prompts.predictor_prompts import get_survival_prediction_prompt


def format_state_update_prompt(context: str) -> str:

    base_instructions = f"""## TASK
You are an ICU physician tracking the status of the patient. Based on the previous memory, current events, and your clinical expertise, provide an updated clinical state summary.

## FOCUS PHYSIOLOGY AREAS
1. Hemodynamics: Heart rate, blood pressure trends
2. Respiratory: Oxygenation, respiratory rate, ventilation status
3. Renal/Metabolic: Electrolytes, creatinine, urine output
4. Neurology: GCS, mental status changes
"""
    # Logic for Pruning/Refinement
    return f"""{base_instructions}\n\n{context}\n
## DECISION FLOW:
1. ALWAYS start with <thought_process>.
2. Analyze the Focus Areas in your thought process.
3. Identify any Memory IDs that contain outdated or conflicting clinical data.
4. If conflicts exist, output <action>PRUNE</action> followed by <ids>. You have to STOP and DO NOT proceed to state update.
5. If no conflicts exist or after pruning, output <state_update> with JSON.

## REQUIRED OUTPUT FORMATS:
### Step 1: Mandatory Thought Process (ALWAYS REQUIRED)
<thought_process>
- [Assessment]: Summary of current vitals/trends.
- [Evaluating Concerns]: Evaluate previous concerns and new events, update their status (resolved, ongoing, new).
- [Conflict Check]: List specific Memory IDs that are now medically obsolete or factually contradicted.
- [Decision]: Explain why you are either Pruning or proceeding directly to State Update.
</thought_process>

### Step 2: Choose ONE of the following actions:

#### OPTION A: PRUNE MEMORY (If data is conflicting/obsolete)
<action>PRUNE</action>
<ids>1, 2</ids>
Note: you have to STOP here.

#### OPTION B: STATE UPDATE (If memory is clean or after pruning)
<state_update>
{{
  "summary": "The updated concise clinical summary",
  "physiology": {{
    "hemodynamics": "stable/improving/deteriorating",
    "respiratory": "stable/improving/deteriorating",
    "renal": "stable/improving/deteriorating",
    "neurology": "stable/improving/deteriorating"
  }},
  "key_concerns": ["List of active clinical concerns and risk factors"],
  "interventions": ["List of crucial interventions"],
  "uncertainties": ["List of any clinical uncertainties or pending results"],
  "trajectory": "improving/stable/deteriorating"
}}
</state_update>"""


def format_refine_state_prompt(state_text: str) -> str:
    """
    TODO: not used yet
    """
    return f"""## Current Patient State (Too Long)
{state_text}

## Your Task
Compress this clinical state summary while preserving:
1. Active critical issues
2. Current trajectory
3. Key vital sign trends
4. Important physiology status
5. Active interventions
6. Clinical uncertainties

Remove:
1. Resolved issues
2. Redundant information
3. Historical details no longer relevant

Respond in JSON format:
{{
  "summary": "Compressed summary",
  "key_concerns": ["Only active concerns"],
  "physiology": {{
    "hemodynamics": "stable/improving/deteriorating",
    "respiratory": "stable/improving/deteriorating",
    "renal": "stable/improving/deteriorating",
    "neurology": "stable/improving/deteriorating"
  }},
  "interventions": ["Active interventions"],
  "uncertainties": ["Pending results or uncertainties"],
  "trajectory": "current trajectory"
}}"""


def format_survival_prediction_prompt(context: str) -> str:
    """Use the shared prediction prompt, formatted with the given context."""
    return get_survival_prediction_prompt().format(context=context)
