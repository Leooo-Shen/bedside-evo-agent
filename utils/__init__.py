"""
Utility functions for the Evo-Agent project.
"""

from .vital_trends import (
    VITAL_GUIDELINES,
    calculate_vital_status,
    calculate_vital_trends,
    classify_vital_status,
    format_vital_status,
    format_vital_trends,
    get_vital_names,
    select_plottable_vitals,
)
from .llm_log_viewer import build_pipeline_agents, generate_html_from_json, save_llm_calls_html
from .status_scoring import PRIMARY_STATUS_LABELS, STATUS_SCORE_MAP, nearest_primary_status, status_to_score
from .discharge_summary_selector import (
    select_discharge_summaries_for_icu_stays,
    summarize_discharge_summary_selection,
)

__all__ = [
    "VITAL_GUIDELINES",
    "get_vital_names",
    "classify_vital_status",
    "calculate_vital_status",
    "format_vital_status",
    "calculate_vital_trends",
    "format_vital_trends",
    "select_plottable_vitals",
    "build_pipeline_agents",
    "save_llm_calls_html",
    "generate_html_from_json",
    "STATUS_SCORE_MAP",
    "PRIMARY_STATUS_LABELS",
    "status_to_score",
    "nearest_primary_status",
    "select_discharge_summaries_for_icu_stays",
    "summarize_discharge_summary_selection",
]
