"""
Prompt templates for the Evo-Agent.

These prompts guide the agent through prediction, reflection, and learning.
"""

from typing import Dict, List

from utils.vital_trends import calculate_vital_status, calculate_vital_trends, format_vital_status, format_vital_trends


def format_prediction_prompt(window_data: Dict, memory_context: str) -> str:
    """
    Format prompt for agent to make predictions about next window.

    Args:
        window_data: Current window data with history and current observations
        memory_context: Formatted memory insights from previous reflections

    Returns:
        Formatted prompt string
    """
    # Extract patient metadata
    subject_id = window_data.get("subject_id")
    age = window_data.get("age_at_admission", "Unknown")
    hours_since_admission = window_data.get("hours_since_admission", 0)

    # Extract event counts
    num_history = window_data.get("num_history_events", 0)
    num_current = window_data.get("num_current_events", 0)

    # Format history events
    history_events = window_data.get("history_events", [])
    history_str = _format_events(history_events, max_events=50)

    # Format current window events
    current_events = window_data.get("current_events", [])
    current_str = _format_events(current_events, max_events=30)

    # Calculate vital trends
    vital_trends = calculate_vital_trends(history_events, current_events)
    vital_trends_str = format_vital_trends(vital_trends)

    prompt = f"""You are an ICU clinical decision support agent. Your task is to analyze the current patient state and predict what will happen in the next 30-minute window.

## Patient Information
- Patient ID: {subject_id}
- Age: {age:.1f} years
- Hours since ICU admission: {hours_since_admission:.1f}

Note: You are analyzing data from the first 12 hours after ICU admission only.

## Your Memory
{memory_context}

## Historical Context (Past {window_data.get('lookback_window_hours', 6)} hours)
Total events: {num_history}
{history_str}

## Current Window Observations (Last 30 minutes)
Total events: {num_current}
{current_str}

## Vital Sign Trends (History → Current)
{vital_trends_str}

## Your Task
Based on the above information, your accumulated insights, and the vital sign trends, predict what will happen in the NEXT 30-minute window.

Provide your response in the following JSON format:
{{
  "vitals_prediction": {{
    "overall_trend": "improving/stable/deteriorating",
    "key_vitals": {{
      "heart_rate": "increasing/stable/decreasing",
      "blood_pressure": "increasing/stable/decreasing",
      "respiratory_rate": "increasing/stable/decreasing",
      "oxygen_saturation": "increasing/stable/decreasing"
    }},
    "key_concerns": "Brief description of main vital sign concerns"
  }},
  "patient_status_prediction": {{
    "severity_score": <float from -1.0 to 1.0>,
    "trajectory": "improving/stable/deteriorating",
    "rationale": "Brief explanation for your prediction"
  }},
  "recommended_actions": [
    {{
      "action": "Specific clinical intervention",
      "rationale": "Why this action is recommended",
      "priority": "high/medium/low"
    }}
  ],
  "confidence": {{
    "overall_confidence": <float from 0.0 to 1.0>,
    "uncertainty_factors": "What makes you uncertain"
  }}
}}

Note:
- Severity score: -1.0 = critically ill, 0.0 = stable, 1.0 = improving
- Base your predictions on clinical patterns you've learned and current observations
- Be specific and actionable in your recommendations
"""

    return prompt


def format_reflection_prompt(
    prediction: Dict, actual_events: List[Dict], window_index: int, memory_context: str
) -> str:
    """
    Format prompt for agent to reflect on prediction accuracy.

    Args:
        prediction: The agent's previous prediction
        actual_events: Actual events that occurred in the window
        window_index: Index of the window being reflected on
        memory_context: Current memory context

    Returns:
        Formatted prompt string
    """
    actual_str = _format_events(actual_events, max_events=30)

    prompt = f"""You are reflecting on your previous prediction to learn and improve.

## Your Previous Prediction (Window {window_index})
{_format_prediction(prediction)}

## What Actually Happened
{actual_str}

## Your Current Memory
{memory_context}

## Your Task
Compare your prediction with what actually happened and generate a clinical insight to add to your memory.

Provide your response in the following JSON format:
{{
  "prediction_accuracy": {{
    "vitals_accuracy": "correct/partially_correct/incorrect",
    "status_accuracy": "correct/partially_correct/incorrect",
    "overall_assessment": "Brief assessment of prediction quality"
  }},
  "error_analysis": {{
    "what_was_wrong": "What did you predict incorrectly?",
    "why_was_wrong": "Why was your prediction wrong?",
    "missed_patterns": "What clinical patterns or signals did you miss?"
  }},
  "new_insight": {{
    "clinical_scenario": "Brief description of the clinical context (e.g., 'Septic patient with rising lactate')",
    "insight": "Transferable clinical knowledge learned from this experience (e.g., 'When lactate rises despite fluids, consider early vasopressor initiation')",
    "confidence": <float from 0.0 to 1.0>
  }}
}}

Note:
- Focus on generating insights that are transferable to future cases
- Be specific about cause-effect relationships you observed
- Higher confidence for insights based on clear patterns, lower for uncertain observations
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

    # Limit number of events to avoid token overflow
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


def _format_prediction(prediction: Dict) -> str:
    """
    Format a prediction dictionary for display.

    Args:
        prediction: Prediction dictionary

    Returns:
        Formatted string
    """
    formatted = "### Vitals Prediction\n"
    vitals = prediction.get("vitals_prediction", {})
    formatted += f"- Overall Trend: {vitals.get('overall_trend', 'N/A')}\n"

    key_vitals = vitals.get("key_vitals", {})
    if key_vitals:
        formatted += f"- Heart Rate: {key_vitals.get('heart_rate', 'N/A')}\n"
        formatted += f"- Blood Pressure: {key_vitals.get('blood_pressure', 'N/A')}\n"
        formatted += f"- Respiratory Rate: {key_vitals.get('respiratory_rate', 'N/A')}\n"
        formatted += f"- Oxygen Saturation: {key_vitals.get('oxygen_saturation', 'N/A')}\n"

    formatted += f"- Key Concerns: {vitals.get('key_concerns', 'N/A')}\n\n"

    formatted += "### Patient Status Prediction\n"
    status = prediction.get("patient_status_prediction", {})
    formatted += f"- Severity Score: {status.get('severity_score', 'N/A')}\n"
    formatted += f"- Trajectory: {status.get('trajectory', 'N/A')}\n"
    formatted += f"- Rationale: {status.get('rationale', 'N/A')}\n\n"

    formatted += "### Recommended Actions\n"
    actions = prediction.get("recommended_actions", [])
    if actions:
        for i, action in enumerate(actions, 1):
            formatted += f"{i}. {action.get('action', 'N/A')} (Priority: {action.get('priority', 'N/A')})\n"
            formatted += f"   Rationale: {action.get('rationale', 'N/A')}\n"
    else:
        formatted += "No specific actions recommended\n"

    formatted += f"\n### Confidence\n"
    confidence = prediction.get("confidence", {})
    formatted += f"- Overall: {confidence.get('overall_confidence', 'N/A')}\n"
    formatted += f"- Uncertainty: {confidence.get('uncertainty_factors', 'N/A')}\n"

    return formatted


def format_vital_prediction_prompt(window_data: Dict) -> str:
    """
    Format prompt for vital sign prediction experiment (guideline-based).

    This is a focused prompt specifically for predicting whether vitals will be
    above, within, or below clinical guideline ranges in the next window.

    Args:
        window_data: Current window data with history and current observations

    Returns:
        Formatted prompt string
    """
    # Extract patient metadata
    subject_id = window_data.get("subject_id")
    age = window_data.get("age_at_admission", "Unknown")
    hours_since_admission = window_data.get("hours_since_admission", 0)

    # Extract event counts
    num_current = window_data.get("num_current_events", 0)

    # Format current window events
    current_events = window_data.get("current_events", [])
    current_str = _format_events(current_events, max_events=50)

    # Calculate current vital status against guidelines
    vital_status = calculate_vital_status(current_events)
    vital_status_str = format_vital_status(vital_status)

    prompt = f"""You are an ICU clinical decision support system. Your task is to predict vital sign status in the NEXT 30-minute window.

## Patient Information
- Patient ID: {subject_id}
- Age: {age:.1f} years
- Hours since ICU admission: {hours_since_admission:.1f}

## Current Window Observations (Last 30 minutes)
Total events: {num_current}
{current_str}

## Current Vital Signs Status
{vital_status_str}

## Clinical Guidelines Reference
- Heart Rate: Normal range 60-100 bpm
- Blood Pressure Systolic: Normal range 90-120 mmHg
- Blood Pressure Diastolic: Normal range 60-80 mmHg
- Respiratory Rate: Normal range 12-20 insp/min
- O2 Saturation: Normal range 95-100%

## Your Task
Based on the current observations, predict whether each vital sign will be **above normal**, **within normal**, or **below normal** in the NEXT 30-minute window.

Provide your response in the following JSON format:
{{
  "vital_predictions": {{
    "heart_rate": {{
      "predicted_status": "above_normal/normal/below_normal",
      "rationale": "Brief explanation for prediction"
    }},
    "blood_pressure_systolic": {{
      "predicted_status": "above_normal/normal/below_normal",
      "rationale": "Brief explanation for prediction"
    }},
    "blood_pressure_diastolic": {{
      "predicted_status": "above_normal/normal/below_normal",
      "rationale": "Brief explanation for prediction"
    }},
    "respiratory_rate": {{
      "predicted_status": "above_normal/normal/below_normal",
      "rationale": "Brief explanation for prediction"
    }},
    "oxygen_saturation": {{
      "predicted_status": "above_normal/normal/below_normal",
      "rationale": "Brief explanation for prediction"
    }}
  }},
  "overall_assessment": {{
    "patient_trajectory": "improving/stable/deteriorating",
    "key_concerns": "Brief description of main concerns",
    "confidence": <float from 0.0 to 1.0>
  }}
}}

Note:
- "above_normal": Vital will be higher than the normal range upper limit
- "normal": Vital will be within the normal range
- "below_normal": Vital will be lower than the normal range lower limit
- Base predictions on clinical patterns and current vital trends
"""

    return prompt


def format_survival_prediction_prompt(window_data: Dict, memory_context: str = None) -> str:
    """
    Format prompt for survival prediction with optional memory context.

    Args:
        window_data: Current window data
        memory_context: Optional formatted memory insights. If None or empty,
                       the prompt will indicate no memory is available.

    Returns:
        Formatted prompt string
    """
    age = window_data.get("patient_metadata", {}).get("age", "Unknown")
    hours_since_admission = window_data.get("hours_since_admission", 0)
    current_events = window_data.get("current_events", [])
    history_events = window_data.get("history_events", [])

    current_str = _format_events(current_events)
    history_str = _format_events(history_events)
    age_str = f"{age:.1f}" if isinstance(age, (int, float)) else str(age)
    hours_str = f"{hours_since_admission:.1f}"

    # Format memory section based on whether memory_context is provided
    if memory_context:
        memory_section = f"""## Learned Clinical Insights (Memory)
{memory_context}
"""
    else:
        memory_section = """## Learned Clinical Insights (Memory)
No memory available. Base your prediction solely on the provided observations.
"""

    prompt = f"""You are an ICU clinical decision support system. Your goal is to predict if a patient will SURVIVE or DIE after ICU discharge by analyzing physiological trends and clinical context.

## Patient Information
- Age: {age_str} years
- Hours since ICU admission: {hours_str}

{memory_section}
## Historical Context (Past 2 Hours)
{history_str}

## Current Window Observations (Last 30 Minutes)
{current_str}

## Clinical Analysis Framework
Analyze the patient using the following four pillars of ICU stability. If "Learned Clinical Insights" are provided above, integrate them into this framework; otherwise, rely strictly on the provided observations.

1. **Hemodynamics:** Are Heart Rate and Blood Pressure stable or trending toward normal?
2. **Respiratory:** Is oxygenation (SpO2) and Respiratory Rate stable on the current level of support?
3. **Metabolism/Renal:** Are electrolytes (e.g., Potassium) and renal markers (Creatinine/Urine) improving or within safe ranges?
4. **Neurology:** Differentiate between "Reversible Delirium" (Confusion/Restraints with high GCS) and "Terminal Decline" (Significant drop in GCS/Unresponsiveness).

## Your Task
Predict the outcome (SURVIVE or DIE). Prioritize the "Current Window" and physiological "Delta" (the change from History to Current) over static risk factors like age.

Provide your response in the following JSON format:
{{
  "survival_prediction": {{
    "outcome": "survive/die",
    "confidence": <float from 0.0 to 1.0>,
    "rationale": "Directly address the 4-pillar analysis and how the patient's trajectory influenced the outcome."
  }},
  "patient_status": {{
    "severity_category": "Critical/Unstable/Guarded/Stable/Improving",
    "trajectory": "improving/stable/deteriorating",
    "key_concerns": "Identify the primary organ system of concern"
  }},
  "risk_factors": [
    {{
      "factor": "Specific physiological marker",
      "impact": "high/medium/low",
      "description": "How this affects survival"
    }}
  ]
}}

Note:
- Severity Category Definitions:
    - Critical: Life-threatening failure.
    - Unstable: Active deterioration.
    - Guarded: High risk but physiologically compensated.
    - Stable: Vitals within normal limits.
    - Improving: Resolving acute issues.
- A patient with normal vitals and high GCS should generally be predicted to SURVIVE, even if they are elderly or require safety restraints for confusion.
"""
    return prompt
