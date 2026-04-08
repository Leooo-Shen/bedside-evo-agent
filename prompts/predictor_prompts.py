def get_survival_prediction_prompt() -> str:
    return """You are a clinical decision support AI specializing in ICU outcome prediction. Your task is to predict whether the patient will die during their current ICU stay, based on the observed ICU timeline provided below.

## INPUT
You will receive one of:
(A) Raw ICU events in chronological order
(B) A structured Memory object with the following layers:
    - patient_metadata
    - working_memory: raw events from recent windows — current local status
    - critical_events_memory: key change points in patient trajectory
    - trajectory_memory: global progression since ICU admission
    - insight_memory: patient-specific deviations from typical ICU trajectories

## INSTRUCTIONS
1. Assess the patient's overall clinical state, trajectory, and risk of in-ICU death.
2. Prioritize high-signal information:
   - Hemodynamic instability (e.g., refractory hypotension, escalating vasopressors)
   - Respiratory failure (e.g., worsening oxygenation, ventilator dependence)
   - Organ dysfunction and progression
   - Deterioration vs. stabilization trajectory
3. If Memory is provided, use all layers — each captures a different scope of the patient's trajectory.
4. If raw events are provided, reason through key clinical states and trends internally before outputting your prediction.
5. Ground every claim strictly in the provided data.

## OUTPUT FORMAT
Respond in JSON format only:
{{
  "prediction": "Survive" | "Die",
  "confidence": "Low" | "Moderate" | "High",
  "supporting_evidence": [
    {{
      "source": "raw_event" | "critical_event" | "trajectory" | "insight" | "metadata",
      "content": "Brief summary of the evidence",
      "significance": "Why this supports the prediction"
    }}
  ],
  "rationale": "1–3 sentences of concise clinical reasoning"
}}

## CONSTRAINTS
- Every claim in the rationale must map to an entry in supporting_evidence.
- Include only the strongest 2–4 evidence items. Avoid redundancy.
- Do not hallucinate or infer data not present in the input.
- If data is insufficient for a confident prediction, set confidence to "Low" and explain the uncertainty in rationale.

## CLINICAL CONTEXT
{context}

Based on the above, predict whether this patient will die during their ICU stay.
"""


def get_survival_prediction_prompt_naive() -> str:
    return """You are a clinical decision support AI specializing in ICU outcome prediction. Your task is to predict whether the patient will die during their current ICU stay, based on the observed ICU timeline provided below.

## OUTPUT FORMAT
Respond in JSON format only:
{{
  "prediction": "Survive" | "Die",
  "confidence": "Low" | "Moderate" | "High",
  "supporting_evidence": [
    {{
      "source": "raw_event" | "critical_event" | "trajectory" | "insight" | "metadata",
      "content": "Brief summary of the evidence",
      "significance": "Why this supports the prediction"
    }}
  ],
  "rationale": "1–3 sentences of concise clinical reasoning"
}}

## CLINICAL CONTEXT
{context}

Based on the above, predict whether this patient will die during their ICU stay.
"""


def get_patient_status_prediction_prompt() -> str:
    return """You are a clinical decision support AI. Assess the patient's overall clinical status for the current ICU window.

## INPUT
You will receive one of:
(A) Raw ICU events in chronological order
(B) A structured Memory object with the following layers:
    - patient_metadata
    - working_memory: raw events from recent windows — current local status
    - critical_events_memory: key change points in patient trajectory
    - trajectory_memory: global progression since ICU admission
    - insight_memory: patient-specific deviations from typical ICU trajectories

## INSTRUCTIONS
Assess the patient's clinical direction at this window by reasoning across four domains:
- Hemodynamics: heart rate, MAP, lactate, perfusion, vasopressor requirements
- Respiratory: SpO2, PaO2/FiO2, respiratory rate, ventilator settings
- Renal/Metabolic: creatinine, urine output, electrolytes, acid-base status
- Neurology: GCS, mental status, RASS sedation score

Synthesize your domain reasoning into a single overall status label, weighted by the relative clinical importance of each domain for this specific patient:
- improving: indicators trending toward stability or recovery relative to the provided context
- stable: no meaningful change in either direction
- deteriorating: indicators trending toward worsening or decompensation relative to the provided context
- insufficient_data: available information is not sufficient to make a reliable judgment (e.g., sparse events, conflicting signals)

## OUTPUT FORMAT
Respond in JSON only:
{{
  "patient_assessment": {{
    "overall": {{
      "label": "improving" | "stable" | "deteriorating" | "insufficient_data",
      "rationale": "<1-2 sentences>"
    }}
  }}
}}

## CLINICAL CONTEXT
{context}
"""


def get_recommendataion_action_prompt() -> str:
    return """You are a clinical decision support AI. Based on the patient's current clinical status and trajectory, recommend the single most impactful clinical action that could be taken in the next ICU window to improve the patient's outcome. 

## INPUT
You will receive one of:
(A) Raw ICU events in chronological order
(B) A structured Memory object with the following layers:
    - patient_metadata
    - working_memory: raw events from recent windows — current local status
    - critical_events_memory: key change points in patient trajectory
    - trajectory_memory: global progression since ICU admission
    - insight_memory: patient-specific deviations from typical ICU trajectories


## INSTRUCTIONS
TODO

## OUTPUT FORMAT
Respond in JSON only:
TODO

## CLINICAL CONTEXT
{context}
"""
