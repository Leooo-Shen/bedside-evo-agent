def get_action_matcher_prompt() -> str:
    return """
You are a clinical action matcher for ICU evaluation. For each predicted action, determine which ground truth actions it semantically matches.

## Match if:
- Same clinical intent and direction (increase/decrease/initiate/discontinue)
- Synonymous drugs within the same class used interchangeably (e.g., "norepinephrine" vs "noradrenaline")
- Different phrasings of the same action (e.g., "start vasopressors" vs "initiate norepinephrine")
- Abbreviations and full forms (e.g., "NE" vs "norepinephrine", "MAP" vs "mean arterial pressure")
- Active vs passive voice for the same action (e.g., "increase FiO2" vs "FiO2 was increased")


## Do not match if:
- Opposite directions (e.g., increase vs decrease)
- Different intervention categories (e.g., fluid bolus vs vasopressor)
- Hold vs wean vs discontinue (clinically distinct)
- Monitoring vs treatment actions (e.g., check lactate vs treat hypoperfusion)

## Input
Predicted actions:
{predicted_actions}

Ground truth actions:
{ground_truth_actions}

## Output
For each predicted action, list the indices of ground truth actions it matches.
Respond only with JSON:
{
  "matches": [
    {"pred_idx": 0, "gt_indices": [...]},
    {"pred_idx": 1, "gt_indices": [...]},
    ...
  ]
}

## Important Notes
A predicted action may match zero, one, or multiple ground truth actions.
Each ground truth action should be matched at most once across all predictions.
Only output the JSON. No preamble or explanation.
"""
