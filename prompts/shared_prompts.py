"""Shared prompt templates used across all agent types."""


def get_prediction_prompt(use_thinking: bool = True, observation_hours: float = 12.0) -> str:
    """Get the prediction prompt for survival prediction.

    Args:
        use_thinking: Whether to include explicit chain of thought section
        observation_hours: Number of hours observed after ICU admission
    """
    thinking_section = (
        """ALWAYS start with thinking first in <think></think>. Then provide your response in JSON format in <response></response>.

<think>
- Physiology status: Analyze using the four pillars:
  1. Hemodynamics: Are vitals stable?
  2. Respiratory: Is oxygenation adequate?
  3. Renal/Metabolic: Are labs improving?
  4. Neurology: Is mental status appropriate?
- Trajectory development: Is the patient improving, stable, or deteriorating over time? Focus on the trend and ignore potential noise.
</think>

"""
        if use_thinking
        else "Provide your response in JSON format in <response></response>.\n\n"
    )

    obs_hours = int(observation_hours) if float(observation_hours).is_integer() else observation_hours

    return f"""You are an ICU clinical decision support system predicting patient survival after ICU discharge.

## Clinical Contexts (First {obs_hours} Hours After ICU Admission)
Below are all clinical contexts from the first {obs_hours} hours after ICU admission:

{{context}}

## Your Task
Based on the patient context from the first {obs_hours} hours after ICU admission, predict whether this patient will survive or die after ICU discharge.

{thinking_section}<response>
{{{{
  "survival_prediction": {{{{
    "outcome": "survive/die",
    "confidence": <float from 0.0 to 1.0>,
    "rationale": "Detailed clinical reasoning for your prediction"
  }}}},
  "patient_trajectory_assessment": {{{{
    "severity_category": "improving/stable/critically_ill",
    "trajectory": "improving/stable/deteriorating",
    "rationale": "Clinical reasoning for the trajectory assessment"
  }}}},
  "supportive_factors": [
    {{{{
      "factor": "Specific supportive factor",
      "impact": "high/medium/low",
      "description": "How this supports survival"
    }}}},
    ...
  ],
  "risk_factors": [
    {{{{
      "factor": "Specific risk factor",
      "impact": "high/medium/low",
      "description": "How this may result in mortality"
    }}}},
    ...
  ]
}}}}
</response>

Note:
- Severity categories: improving (positive trajectory), stable (controlled condition), critically_ill (life-threatening, high mortality risk)
- Base your prediction on clinical patterns and vital trends
- Be honest about uncertainty - low confidence is acceptable"""
