"""Shared prompt templates used across all agent types."""


def get_prediction_prompt(observation_hours: float = 12.0) -> str:
    """Get the prediction prompt for survival prediction.

    Args:
        observation_hours: Number of hours observed after ICU admission
    """
    obs_hours = int(observation_hours) if float(observation_hours).is_integer() else observation_hours

    return f"""You are an ICU clinical decision support system predicting patient survival after ICU discharge.

## Clinical Contexts (First {obs_hours} Hours After ICU Admission)
Below are all clinical contexts from the first {obs_hours} hours after ICU admission:

{{context}}

## Your Task
Based on the patient context from the first {obs_hours} hours after ICU admission, predict whether this patient will survive or die after ICU discharge.

## Output Specification
Return valid JSON only (no XML tags, no markdown code fences):
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

Note:
- Severity categories: improving (positive trajectory), stable (controlled condition), critically_ill (life-threatening, high mortality risk)
- Base your prediction on clinical patterns and vital trends
- Be honest about uncertainty - low confidence is acceptable"""
