"""General-purpose HTML viewer for LLM call logs."""

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Tuple


def get_call_step_type(call: Dict[str, Any]) -> str:
    """Extract step type from a call record."""
    metadata = call.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("step_type"):
        return str(metadata["step_type"])
    if call.get("step_type"):
        return str(call["step_type"])
    return "unknown"


def build_pipeline_agents(agent: Any, agent_type: str) -> List[Dict[str, Any]]:
    """Build pipeline metadata (used + thinking mode) from an agent instance."""
    if agent_type == "multi":
        use_observer_agent = bool(getattr(agent, "use_observer_agent", True))
        use_memory_agent = bool(getattr(agent, "use_memory_agent", False))
        use_reflection_agent = bool(getattr(agent, "use_reflection_agent", False)) and use_memory_agent
        return [
            {
                "name": "observer",
                "used": use_observer_agent,
                "thinking": bool(getattr(agent, "observer_use_thinking", False)) if use_observer_agent else None,
            },
            {
                "name": "memory_agent",
                "used": use_memory_agent,
                "thinking": bool(getattr(agent, "memory_use_thinking", False)) if use_memory_agent else None,
            },
            {
                "name": "reflection_agent",
                "used": use_reflection_agent,
                "thinking": bool(getattr(agent, "reflection_use_thinking", False)) if use_reflection_agent else None,
            },
            {"name": "predictor", "used": True, "thinking": bool(getattr(agent, "predictor_use_thinking", False))},
        ]

    if agent_type == "fold":
        return [{"name": "fold_agent", "used": True, "thinking": None}]

    if agent_type == "remem":
        return [{"name": "remem_agent", "used": True, "thinking": None}]

    return []


def _format_json_block(value: Any) -> str:
    """Format any object as pretty JSON text."""
    if value is None:
        return "null"
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except TypeError:
        return str(value)


def _resolve_pipeline_agents(patient_logs: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
    """Resolve pipeline metadata from logs; infer from calls if missing."""
    pipeline_agents = patient_logs.get("pipeline_agents")
    if isinstance(pipeline_agents, list) and pipeline_agents:
        return pipeline_agents, False

    calls = patient_logs.get("calls", [])
    step_types = {get_call_step_type(call) for call in calls}
    inferred = [
        {"name": "observer", "used": "observer" in step_types, "thinking": None},
        {
            "name": "memory_agent",
            "used": any(step in step_types for step in ["memory_agent", "memory_agent_revision"]),
            "thinking": None,
        },
        {"name": "reflection_agent", "used": "reflection_agent" in step_types, "thinking": None},
        {"name": "predictor", "used": "predictor" in step_types, "thinking": None},
    ]
    inferred = [agent for agent in inferred if agent["used"]]
    return inferred, True


def _thinking_label(value: Any, used: bool) -> str:
    """Render thinking mode label."""
    if not used:
        return "-"
    if value is True:
        return "ON"
    if value is False:
        return "OFF"
    return "Unknown"


def _normalize_identity_value(value: Any) -> str:
    """Normalize optional identity values from logs."""
    if value is None:
        return ""
    text = str(value).strip()
    return text if text else ""


def _resolve_llm_identity(patient_logs: Dict[str, Any], calls: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Resolve LLM provider/model, falling back to per-call metadata for older logs."""
    provider = _normalize_identity_value(patient_logs.get("llm_provider"))
    model = _normalize_identity_value(patient_logs.get("llm_model"))
    if provider or model:
        return provider or "unknown", model or "unknown"

    for call in calls:
        metadata = call.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        provider = _normalize_identity_value(metadata.get("llm_provider"))
        model = _normalize_identity_value(metadata.get("llm_model"))
        if provider or model:
            return provider or "unknown", model or "unknown"

    return "unknown", "unknown"


def _format_llm_identity(provider: str, model: str) -> str:
    """Build a compact display string for provider/model."""
    if provider != "unknown" and model != "unknown":
        return f"{provider} / {model}"
    if model != "unknown":
        return model
    if provider != "unknown":
        return provider
    return "unknown"


def save_llm_calls_html(patient_logs: Dict[str, Any], output_path: Path) -> None:
    """Save an interactive HTML viewer for a patient llm_calls payload."""
    calls = patient_logs.get("calls", [])
    pipeline_agents, inferred_pipeline = _resolve_pipeline_agents(patient_logs)
    llm_provider, llm_model = _resolve_llm_identity(patient_logs, calls)
    llm_display = _format_llm_identity(llm_provider, llm_model)

    step_counts: Dict[str, int] = {}
    for call in calls:
        step = get_call_step_type(call)
        step_counts[step] = step_counts.get(step, 0) + 1

    step_badges = "".join(
        f"<span class='badge'><strong>{escape(step)}</strong>: {count}</span>"
        for step, count in sorted(step_counts.items())
    )

    pipeline_rows = "".join(
        "<tr>"
        f"<td>{escape(str(agent_info.get('name', 'unknown')))}</td>"
        f"<td>{'Yes' if agent_info.get('used') else 'No'}</td>"
        f"<td>{_thinking_label(agent_info.get('thinking'), bool(agent_info.get('used')))}</td>"
        "</tr>"
        for agent_info in pipeline_agents
    )

    call_sections = []
    for idx, call in enumerate(calls, 1):
        step_type = get_call_step_type(call)
        window_index = call.get("window_index", "n/a")
        hours = call.get("hours_since_admission", "n/a")
        input_tokens = call.get("input_tokens", 0)
        output_tokens = call.get("output_tokens", 0)
        timestamp = escape(str(call.get("timestamp", "")))

        metadata_text = _format_json_block(call.get("metadata", {}))
        parsed_text = _format_json_block(call.get("parsed_response"))
        prompt_text = call.get("prompt", "")
        response_text = call.get("response", "")

        section = f"""
<details class="call">
  <summary>
    <span class="call-index">#{idx}</span>
    <span class="call-step">{escape(step_type)}</span>
    <span>window {window_index}</span>
    <span>hour {hours}</span>
    <span>tokens {input_tokens} in / {output_tokens} out</span>
    <span class="timestamp">{timestamp}</span>
  </summary>
  <div class="call-content">
    <div class="panel">
      <h3>Metadata</h3>
      <pre>{escape(metadata_text)}</pre>
    </div>
    <div class="panel">
      <h3>Parsed Response</h3>
      <pre>{escape(parsed_text)}</pre>
    </div>
    <div class="panel">
      <h3>Prompt</h3>
      <pre>{escape(prompt_text)}</pre>
    </div>
    <div class="panel">
      <h3>Response</h3>
      <pre>{escape(response_text)}</pre>
    </div>
  </div>
</details>
"""
        call_sections.append(section)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Call Viewer - {escape(patient_logs.get("patient_id", "unknown"))}</title>
  <style>
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f3f4f6;
      color: #111827;
    }}
    .container {{
      max-width: 1100px;
      margin: 24px auto;
      padding: 0 16px 32px;
    }}
    .header {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .header h1 {{
      margin: 0 0 8px 0;
      font-size: 20px;
    }}
    .muted {{
      color: #4b5563;
      margin: 0 0 10px 0;
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .pipeline {{
      margin-top: 10px;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      overflow: hidden;
    }}
    .pipeline table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      background: #fff;
    }}
    .pipeline th {{
      text-align: left;
      background: #f9fafb;
      border-bottom: 1px solid #e5e7eb;
      padding: 8px 10px;
    }}
    .pipeline td {{
      border-top: 1px solid #f3f4f6;
      padding: 8px 10px;
    }}
    .badge {{
      background: #eef2ff;
      color: #1e40af;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      border: 1px solid #c7d2fe;
    }}
    details.call {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      margin-bottom: 10px;
      overflow: hidden;
    }}
    details.call > summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      padding: 10px 14px;
      cursor: pointer;
      list-style: none;
      border-bottom: 1px solid transparent;
      background: #fafafa;
      font-size: 13px;
    }}
    details.call[open] > summary {{
      border-bottom-color: #e5e7eb;
      background: #f9fafb;
    }}
    .call-index {{
      font-weight: 700;
      color: #374151;
    }}
    .call-step {{
      font-weight: 600;
      color: #1d4ed8;
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 6px;
      padding: 2px 8px;
    }}
    .timestamp {{
      margin-left: auto;
      color: #6b7280;
      font-size: 12px;
    }}
    .call-content {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      padding: 12px;
    }}
    .panel {{
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      background: #ffffff;
    }}
    .panel h3 {{
      margin: 0;
      padding: 8px 10px;
      font-size: 13px;
      border-bottom: 1px solid #e5e7eb;
      background: #f9fafb;
    }}
    pre {{
      margin: 0;
      padding: 10px;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.4;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      background: #fff;
    }}
    @media (min-width: 960px) {{
      .call-content {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>LLM Call Viewer</h1>
      <p class="muted">Patient: <strong>{escape(patient_logs.get("patient_id", "unknown"))}</strong> | Total calls: <strong>{len(calls)}</strong></p>
      <p class="muted">LLM: <strong>{escape(llm_display)}</strong></p>
      <div class="badges">{step_badges}</div>
      <div class="pipeline">
        <table>
          <thead>
            <tr>
              <th>Agent</th>
              <th>Used</th>
              <th>Thinking Mode</th>
            </tr>
          </thead>
          <tbody>{pipeline_rows}</tbody>
        </table>
      </div>
      {"<p class='muted'>Pipeline info inferred from call steps (thinking mode unknown).</p>" if inferred_pipeline else ""}
    </div>
    {"".join(call_sections)}
  </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def generate_html_from_json(llm_calls_json_path: Path, output_path: Path = None) -> Path:
    """Generate an HTML viewer from a llm_calls.json file."""
    if output_path is None:
        output_path = llm_calls_json_path.with_suffix(".html")

    with open(llm_calls_json_path, "r", encoding="utf-8") as f:
        patient_logs = json.load(f)
    save_llm_calls_html(patient_logs, output_path)
    return output_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML viewer from llm_calls.json")
    parser.add_argument("llm_calls_json", type=Path, help="Path to llm_calls.json")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (default: same name with .html)")
    args = parser.parse_args()

    output_path = generate_html_from_json(args.llm_calls_json, args.output)
    print(f"Generated HTML viewer: {output_path}")


if __name__ == "__main__":
    _main()
