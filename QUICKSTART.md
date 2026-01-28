# Quick Start Guide - Meta Oracle System

## Prerequisites

1. Python 3.8 or higher
2. MIMIC-demo dataset in `data/mimic-demo/`
3. OpenAI API key (or Anthropic API key for Claude)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up your API keys
# Option 1: Use .env file (recommended)
cp .env.example .env
# Then edit .env and add your actual API keys

# Option 2: Use environment variables
export ANTHROPIC_API_KEY="your-anthropic-key"
# OR
export OPENAI_API_KEY="your-openai-key"
```

**Note:** The `.env` file is automatically loaded by the system. The system is configured to use OpenAI by default. You can switch to Anthropic by using `--provider anthropic` flag.

## Verify Setup

Test your OpenAI connection:

```bash
python test_openai.py
```

This will verify your API key is working correctly.

Test event cleaning and context optimization:

```bash
python test_event_cleaning.py
```

This shows how the system reduces context size while preserving clinical information.

## Quick Test (Single Patient)

Run the example script to test the system on one patient:

```bash
python examples/example_single_patient.py
```

This will:
- Load the MIMIC-demo data
- Extract one patient's trajectory
- Create time windows
- Run Oracle evaluation on the first window
- Display the evaluation report
- Save results to `data/oracle_outputs/`

**Expected output:**
```
Patient Status Score: 0.25
Status Rationale: [Brief medical reasoning for patient condition]
Action Quality: optimal
Recommended Action: [Specific recommendation]
Clinical Insight: [Transferable clinical pearl]
```

## Process All Patients

### Test Run (10 patients)

```bash
python run_oracle.py --max-patients 10
```

### Full Dataset

```bash
python run_oracle.py
```

**Warning:** Processing all 365 patients will:
- Take several hours
- Use significant API tokens (~$50-200 depending on provider)
- Generate ~400+ ICU stay evaluations

## Understanding the Output

After running, check `data/oracle_outputs/`:

1. **`processing_summary.json`** - Overview statistics
   ```json
   {
     "total_patients": 10,
     "patients_processed": 10,
     "total_windows_evaluated": 156,
     "overall_avg_score": 0.23,
     "total_tokens_used": 1234567
   }
   ```

2. **`all_oracle_reports.json`** - All evaluations combined

3. **`patient_{id}_icu_{id}_oracle_report.json`** - Individual reports

## Customization

### Change Window Size

Evaluate with 12-hour future windows:
```bash
python run_oracle.py --window-hours 12.0
```

### Change Step Size

Create windows every 30 minutes:
```bash
python run_oracle.py --step-hours 0.5
```

### Use Different LLM

The system uses OpenAI (GPT-4o) by default. To use Anthropic Claude instead:
```bash
python run_oracle.py --provider anthropic --model claude-3-5-sonnet-20241022
```

## Programmatic Usage

```python
from config.config import get_config
from data_parser import MIMICDataParser
from agents.oracle import MetaOracle

# Load configuration
config = get_config()

# Load data
parser = MIMICDataParser(
    config.events_path,
    config.icu_stay_path
)
parser.load_data()

# Get patient
trajectory = parser.get_patient_trajectory(
    subject_id=12345,
    icu_stay_id=67890
)

# Create windows
windows = parser.create_time_windows(
    trajectory,
    window_size_hours=config.window_size_hours
)

# Evaluate
oracle = MetaOracle(
    provider=config.oracle_provider,
    model=config.oracle_model
)
reports = oracle.evaluate_trajectory(windows)

# Use results
for report in reports:
    print(f"Patient Status Score: {report.patient_status_score}")
    if report.patient_status_score < -0.5:
        print(f"⚠️  Patient critically ill!")
        print(f"Rationale: {report.status_rationale}")
    if report.action_quality == "sub-optimal":
        print(f"⚠️  Sub-optimal action detected!")
        print(f"Recommended: {report.recommended_action}")
```

## Troubleshooting

### "Cannot import anthropic/openai"
```bash
pip install anthropic openai
```

### "API key not found"
```bash
# Check environment variable
echo $ANTHROPIC_API_KEY

# Set it if missing
export ANTHROPIC_API_KEY="your-key"
```

### "No windows generated"
- Patient ICU stay may be too short
- Try reducing `--window-hours` parameter
- Check that events data is properly loaded

### "JSON parsing failed"
- LLM may have returned malformed JSON
- System will attempt fallback parsing
- Check raw response in error message

## Cost Estimation

Approximate costs per patient (varies by model and window configuration):

- **Anthropic Claude 3.5 Sonnet**: $0.10 - $0.50 per patient
- **OpenAI GPT-4o**: $0.15 - $0.60 per patient

For 365 patients: **$40 - $200 total**

Factors affecting cost:
- Number of time windows per patient
- Window size (more events = longer prompts)
- Model choice
- Step size between windows

## Next Steps

1. **Analyze Results**: Use the exploration notebook to analyze Oracle scores
2. **Build Agent**: Implement the Agent system that learns from Oracle evaluations
3. **Add Memory**: Create the Evolving Memory module for clinical insights
4. **Implement Reflection**: Compare Agent decisions vs Oracle recommendations

See [README.md](README.md) for full documentation.
