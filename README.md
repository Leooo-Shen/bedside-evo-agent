# Meta Oracle System for Bedside-Evo

A retrospective clinical evaluation system that uses LLMs to generate ground truth assessments of clinical decisions with the benefit of hindsight.

## Overview

The Meta Oracle system processes MIMIC-demo ICU patient data to evaluate clinical decisions retrospectively. Unlike real-time clinical decision-making, the Oracle has access to "future" patient outcomes, allowing it to assess whether interventions were appropriate based on what actually happened.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MIMIC-Demo Dataset                        │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  Events Data     │         │  ICU Stay Data   │         │
│  │  (Clinical       │         │  (Metadata)      │         │
│  │   Events)        │         │                  │         │
│  └──────────────────┘         └──────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    Data Parser                               │
│  • Loads patient trajectories                               │
│  • Creates time windows (history + future)                  │
│  • Formats data for Oracle evaluation                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    Meta Oracle                               │
│  • LLM-powered evaluator (GPT-4o / Claude 3.5 Sonnet)      │
│  • Chain-of-Thought retrospective reasoning                 │
│  • Generates structured evaluations                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    Oracle Reports                            │
│  • patient_status_score: -1.0 to 1.0                        │
│  • status_rationale: Medical reasoning                      │
│  • action_quality: optimal/neutral/sub-optimal              │
│  • recommended_action: What should have been done           │
│  • clinical_insight: Transferable clinical pearl            │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Parser ([data_parser.py](data_parser.py))

Processes MIMIC-demo dataset into patient trajectories with time windows:

- **Input**: Events parquet + ICU stay parquet
- **Output**: Time-windowed trajectories with history and future events
- **Key Features**:
  - Extracts complete patient trajectories per ICU stay
  - Creates sliding time windows (default: 6-hour future window, 1-hour steps)
  - Handles multiple ICU stays per patient
  - **Event Cleaning**: Filters events to keep only clinically relevant fields:
    - `time`: Event timestamp
    - `code`: Item identifier
    - `numeric_value`: Numeric measurement
    - `code_specifics`: Label/description
    - `end_time`: End timestamp (if available)
    - `text_value`: Text value or unit
    - `time_delta_minutes`: Minutes since previous event
  - Removes null/empty values to reduce context size
  - Typically achieves 60-80% reduction in context size

### 2. Meta Oracle ([agents/oracle.py](agents/oracle.py))

LLM-powered retrospective evaluator:

- **Input**: Time window with patient history and future events
- **Output**: Structured evaluation report
- **Key Features**:
  - Supports multiple LLM providers (Anthropic, OpenAI)
  - Chain-of-Thought reasoning
  - Structured JSON output
  - Error handling and fallback parsing

### 3. Oracle Prompt ([prompts/oracle_prompt.py](prompts/oracle_prompt.py))

Carefully designed prompt template:

- **System Prompt**: Defines Oracle role and evaluation framework
- **Context Formatting**: Structures patient data for LLM
- **Output Schema**: Ensures consistent structured responses
- **Example Evaluation**: Demonstrates expected reasoning

### 4. LLM Client ([model/llms.py](model/llms.py))

Unified interface for multiple LLM providers:

- Supports Anthropic (Claude) and OpenAI (GPT)
- Handles API authentication
- JSON response parsing
- Token usage tracking

### 5. Batch Processing Pipeline ([run_oracle.py](run_oracle.py))

End-to-end processing script:

- Processes all patients in dataset
- Saves individual and combined reports
- Tracks statistics and token usage
- Command-line interface

## Installation

```bash
# Install required packages
pip install -r requirements.txt

# Set up API keys
# Option 1: Use .env file (recommended)
cp .env.example .env
# Edit .env and add your actual API keys

# Option 2: Set environment variables directly
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

**Note:** You only need one API key (Anthropic or OpenAI) depending on which provider you plan to use. The system automatically loads API keys from the `.env` file if present.

## Project Structure

```
evo-agent/
├── config/                    # Configuration files
│   ├── config.json           # Main configuration (JSON format)
│   └── config.py             # Configuration loader
├── doc/                       # Documentation
│   ├── CONFIG_GUIDE.md       # Configuration guide
│   ├── CONFIG_IMPLEMENTATION.md
│   ├── FINAL_WINDOW_STRUCTURE.md
│   ├── ICU_ONLY_SOLUTION.md
│   ├── LOGGING.md
│   └── spec.md               # Project specification
├── examples/                  # Example scripts
│   ├── example_single_patient.py
│   └── example_logging.py
├── agents/                    # Agent implementations
│   ├── oracle.py             # Meta Oracle agent
│   └── memory.py             # Memory system
├── model/                     # LLM clients
│   └── llms.py               # Unified LLM interface
├── prompts/                   # Prompt templates
│   └── oracle_prompt.py      # Oracle evaluation prompt
├── data/                      # Data directory
│   └── mimic-demo/           # MIMIC-demo dataset
├── tests/                     # Test files
├── data_parser.py            # Data processing utilities
├── run_oracle.py             # Main batch processing script
└── requirements.txt          # Python dependencies
```

## Usage

### Configuration

The system uses a JSON configuration file located at [config/config.json](config/config.json). You can customize:

- Data paths (events, ICU stay data)
- Time window parameters (window size, step size)
- Oracle LLM settings (provider, model, temperature)
- Logging and output directories

See [doc/CONFIG_GUIDE.md](doc/CONFIG_GUIDE.md) for detailed configuration options.

### Basic Usage

Process all patients with default settings:

```bash
python run_oracle.py
```

### Custom Configuration

```bash
python run_oracle.py \
  --events data/mimic-demo/events/data_0.parquet \
  --icu-stay data/mimic-demo/icu_stay/data_0.parquet \
  --output data/oracle_outputs \
  --provider anthropic \
  --window-hours 6.0 \
  --step-hours 1.0 \
  --max-patients 10
```

### Parameters

- `--events`: Path to events parquet file
- `--icu-stay`: Path to ICU stay parquet file
- `--output`: Output directory for reports
- `--provider`: LLM provider (`anthropic` or `openai`)
- `--model`: Specific model name (optional)
- `--window-hours`: Size of future window (default: 6.0)
- `--step-hours`: Step between windows (default: 1.0)
- `--max-patients`: Limit number of patients (for testing)

### Programmatic Usage

```python
from config.config import get_config
from data_parser import MIMICDataParser
from agents.oracle import MetaOracle

# Load configuration
config = get_config()

# Initialize parser
parser = MIMICDataParser(
    events_path=config.events_path,
    icu_stay_path=config.icu_stay_path
)
parser.load_data()

# Get a patient trajectory
trajectory = parser.get_patient_trajectory(subject_id=12345, icu_stay_id=67890)

# Create time windows
windows = parser.create_time_windows(
    trajectory,
    window_size_hours=config.window_size_hours
)

# Initialize Oracle
oracle = MetaOracle(
    provider=config.llm_provider,
    model=config.llm_model
)

# Evaluate windows
reports = oracle.evaluate_trajectory(windows)

# Access results
for report in reports:
    print(f"Patient Status Score: {report.patient_status_score}")
    print(f"Status Rationale: {report.status_rationale}")
    print(f"Action Quality: {report.action_quality}")
    print(f"Recommended Action: {report.recommended_action}")
    print(f"Clinical Insight: {report.clinical_insight}")
```

## Output Format

### Oracle Report Structure

```json
{
  "patient_status_score": 0.5,
  "status_rationale": "Patient showing signs of improvement with stabilizing vitals...",
  "action_quality": "optimal",
  "recommended_action": "Continue current norepinephrine dose and monitor MAP closely...",
  "clinical_insight": "Early vasopressor initiation in septic shock improves outcomes..."
}
```

### Output Files

After processing, the output directory contains:

- `all_oracle_reports.json`: Combined reports for all patients
- `patient_{id}_icu_{id}_oracle_report.json`: Individual patient reports
- `patient_trajectories.jsonl`: Intermediate trajectory data
- `processing_summary.json`: Statistics and metadata

## Oracle Evaluation Framework

### Patient Status Score Scale

- **-1.0**: Critically ill, severe deterioration
- **-0.5**: Unstable, concerning trajectory
- **0.0**: Stable condition
- **+0.5**: Improving, positive trajectory
- **+1.0**: Significantly improving, excellent recovery

### Action Quality Categories

- **optimal**: Interventions were clinically appropriate and beneficial
- **neutral**: Routine care, no significant positive or negative impact
- **sub-optimal**: Decisions had minor to serious negative consequences

### Evaluation Principles

1. **Hindsight Advantage**: Oracle sees outcomes clinicians couldn't
2. **Fair Assessment**: Considers information available at decision time
3. **Focus on Decisions**: Evaluates active interventions, not just monitoring
4. **Outcome-Oriented**: Prioritizes patient outcomes
5. **Evidence-Based**: Grounds evaluation in medical evidence

## Data Insights from Exploration

Based on the MIMIC-demo dataset exploration:

- **365 unique patients** with **400+ ICU stays**
- **Survival rate**: ~90%
- **Readmission rate**: ~25%
- **Average ICU duration**: ~100 hours
- **Age range**: 20-90 years (median ~65)
- **Multiple ICU stays**: 111 patients have 2+ stays (max: 33 stays)

## Future Enhancements

- [ ] Implement Agent system with Evolving Memory
- [ ] Add reflection logic for Agent-Oracle comparison
- [ ] Create vector database for clinical insights
- [ ] Implement inference loop (Observe-Retrieve-Act-Reflect)
- [ ] Add visualization dashboard for Oracle reports
- [ ] Support for real-time streaming evaluation
- [ ] Multi-Oracle consensus mechanism

## References

- MIMIC-IV Demo Dataset
- Bedside-Evo Project Specification ([doc/spec.md](doc/spec.md))
- Configuration Guide ([doc/CONFIG_GUIDE.md](doc/CONFIG_GUIDE.md))

## License

[Add your license here]
