"""
Configuration loader for EVO-Agent Oracle System.

This module loads configuration from config.json and provides easy access
to all configuration parameters throughout the application.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class Config:
    """Configuration manager for the Oracle system."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from JSON file.

        Args:
            config_path: Path to config.json file. If None, looks in config directory.
        """
        if config_path is None:
            # Default to config.json in config directory
            config_path = Path(__file__).parent / "config.json"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self._config = json.load(f)

        self.config_path = config_path

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., "llm.model")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            config.get("llm.model")
            config.get("time_windows.window_size_hours")  # Returns 6.0
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    # ========================================================================
    # Data Configuration
    # ========================================================================

    @property
    def events_path(self) -> str:
        """Path to events parquet file."""
        return self.get("data.events_path")

    @property
    def icu_stay_path(self) -> str:
        """Path to ICU stay parquet file."""
        return self.get("data.icu_stay_path")

    @property
    def output_dir(self) -> str:
        """Output directory for Oracle reports."""
        return self.get("data.output_dir")

    @property
    def max_patients(self) -> Optional[int]:
        """Maximum number of patients to process."""
        return self.get("data.max_patients")

    # ========================================================================
    # Data Loading Configuration
    # ========================================================================

    @property
    def agent_observation_hours(self) -> Optional[float]:
        """Hours observed after ICU admission for agent windows; None means full ICU stay."""
        value = self.get("data_loading.agent_observation_hours", 12.0)
        if value is None:
            return None
        return float(value)

    @property
    def oracle_observation_hours(self) -> Optional[float]:
        """Number of hours to observe for oracle (default None = use all data)."""
        return self.get("data_loading.oracle_observation_hours", None)

    # ========================================================================
    # Oracle Time Window Configuration
    # ========================================================================

    @property
    def oracle_current_window_hours(self) -> float:
        """Size of current observation window in hours for Oracle (default 0.5 = 30 minutes)."""
        return self.get("oracle_time_windows.current_window_hours", 0.5)

    @property
    def oracle_window_step_hours(self) -> float:
        """Step size between sliding windows for Oracle in hours (default 0.5 = 30 minutes)."""
        return self.get("oracle_time_windows.window_step_hours", 0.5)

    @property
    def oracle_include_pre_icu_data(self) -> bool:
        """Whether to include pre-ICU hospital data in history context for Oracle."""
        return self.get("oracle_time_windows.include_pre_icu_data", True)

    @property
    def oracle_use_discharge_summary_for_history(self) -> bool:
        """Whether to use discharge summary as history context for Oracle instead of history events."""
        return self.get("oracle_time_windows.use_discharge_summary_for_history", False)

    @property
    def oracle_num_discharge_summaries(self) -> int:
        """Maximum prioritized pre-ICU reports to include per NOTE_* code for Oracle."""
        return self.get("oracle_time_windows.num_discharge_summaries", 2)

    @property
    def oracle_relative_report_codes(self) -> List[str]:
        """Fixed pre-ICU report code set; standardized to discharge summary only."""
        return ["NOTE_DISCHARGESUMMARY"]

    @property
    def oracle_pre_icu_history_hours(self) -> float:
        """Pre-ICU history lookback window (hours) for fallback + LAB/VITAL baseline context."""
        value = self.get("oracle_time_windows.pre_icu_history_hours")
        if value is None:
            # Backward compatibility for older configs.
            value = self.get("oracle_time_windows.pre_icu_history_fallback_hours", 72.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 72.0

    # ========================================================================
    # Oracle Context Configuration
    # ========================================================================

    @property
    def oracle_context_history_hours(self) -> float:
        """History threshold (hours) before current window start for Oracle context."""
        return self.get("oracle_context.history_hours", 48.0)

    @property
    def oracle_context_future_hours(self) -> float:
        """Future threshold (hours) after current window start for Oracle context."""
        return self.get("oracle_context.future_hours", 48.0)

    @property
    def oracle_context_use_discharge_summary(self) -> bool:
        """Whether Oracle local context should include ICU discharge summary block."""
        return self.get("oracle_context.use_discharge_summary", False)

    @property
    def oracle_context_include_icu_outcome_in_prompt(self) -> bool:
        """Whether Oracle prompt context should include explicit ICU outcome (survived/died)."""
        value = self.get("oracle_context.include_icu_outcome_in_prompt", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "1", "yes", "y", "on"}:
                return True
            if text in {"false", "0", "no", "n", "off"}:
                return False
        return bool(value)

    @property
    def oracle_context_top_k_recommendations(self) -> int:
        """Top-k recommendation count requested in Oracle prompt output."""
        value = self.get("oracle_context.top_k_recommendations", 3)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 3
        return max(1, parsed)

    @property
    def oracle_context_compress_pre_icu_history(self) -> bool:
        """Whether to LLM-compress pre-ICU history once per patient and reuse it across windows."""
        value = self.get("oracle_context.compress_pre_icu_history", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "1", "yes", "y", "on"}:
                return True
            if text in {"false", "0", "no", "n", "off"}:
                return False
        return bool(value)

    # ========================================================================
    # Agent Time Window Configuration
    # ========================================================================

    @property
    def agent_current_window_hours(self) -> float:
        """Size of current observation window in hours for Agent (default 0.5 = 30 minutes)."""
        return self.get("agent_time_windows.current_window_hours", 0.5)

    @property
    def agent_window_step_hours(self) -> float:
        """Step size between sliding windows for Agent in hours (default 0.5 = 30 minutes)."""
        return self.get("agent_time_windows.window_step_hours", 0.5)

    @property
    def agent_include_pre_icu_data(self) -> bool:
        """Whether to include pre-ICU hospital data in history context for Agent."""
        return self.get("agent_time_windows.include_pre_icu_data", True)

    @property
    def agent_use_discharge_summary_for_history(self) -> bool:
        """Whether to use discharge summary as history context for Agent instead of history events."""
        return self.get("agent_time_windows.use_discharge_summary_for_history", False)

    @property
    def agent_num_discharge_summaries(self) -> int:
        """Maximum prioritized pre-ICU reports to include per NOTE_* code for Agent."""
        return self.get("agent_time_windows.num_discharge_summaries", 3)

    @property
    def agent_pre_icu_history_hours(self) -> float:
        """Pre-ICU history lookback window (hours) for Agent pre-ICU context construction."""
        value = self.get("agent_time_windows.pre_icu_history_hours")
        if value is None:
            # Backward compatibility: older configs may only define Oracle horizon.
            return float(self.oracle_pre_icu_history_hours)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(self.oracle_pre_icu_history_hours)

    # ========================================================================
    # LLM Configuration
    # ========================================================================

    @property
    def llm_provider(self) -> str:
        """LLM provider (e.g., 'openai', 'anthropic', 'google', 'gemini')."""
        return self.get("llm.provider")

    @property
    def llm_model(self) -> Optional[str]:
        """Model name for LLM."""
        return self.get("llm.model")

    @property
    def llm_max_tokens(self) -> int:
        """Maximum tokens for LLM responses."""
        return self.get("llm.max_tokens")

    # ========================================================================
    # Logging Configuration
    # ========================================================================

    @property
    def oracle_log_dir(self) -> str:
        """Directory for Oracle logs."""
        return self.get("oracle_logging.log_dir")

    @property
    def oracle_save_trajectories(self) -> bool:
        """Whether to save intermediate trajectory data for Oracle."""
        return self.get("oracle_logging.save_trajectories")

    @property
    def agent_log_dir(self) -> str:
        """Directory for Agent logs."""
        return self.get("agent_logging.log_dir")

    @property
    def agent_save_trajectories(self) -> bool:
        """Whether to save intermediate trajectory data for Agent."""
        return self.get("agent_logging.save_trajectories")

    # ========================================================================
    # ReMeM Configuration
    # ========================================================================

    @property
    def remem_max_state_length(self) -> int:
        """Maximum length of patient state summary (default 1500)."""
        return self.get("remem.max_state_length", 1500)

    @property
    def remem_enable_intra_patient_refinement(self) -> bool:
        """Whether to enable Think-Refine-Act loop for intra-patient state updates (default False)."""
        return self.get("remem.enable_intra_patient_refinement", False)

    # ========================================================================
    # AgentFold Configuration
    # ========================================================================

    @property
    def agent_fold_enable_key_events_extraction(self) -> bool:
        """Whether to enable key events extraction (default True)."""
        return self.get("agent_fold.enable_key_events_extraction", True)

    @property
    def agent_fold_max_trajectory_entries(self) -> int:
        """Maximum number of trajectory entries to maintain (default 20)."""
        return self.get("agent_fold.max_trajectory_entries", 20)

    # ========================================================================
    # AgentMulti Configuration
    # ========================================================================

    @property
    def agent_multi_use_observer_agent(self) -> bool:
        """Whether to enable observer agent in MultiAgent pipeline (default True)."""
        return self.get("agent_multi.use_observer_agent", True)

    @property
    def agent_multi_use_memory_agent(self) -> bool:
        """Whether to enable memory agent for trajectory management (default True)."""
        return self.get("agent_multi.use_memory_agent", True)

    @property
    def agent_multi_use_reflection_agent(self) -> bool:
        """Whether to enable reflection agent for trajectory quality review (default False)."""
        return self.get("agent_multi.use_reflection_agent", False)

    @property
    def agent_multi_observer_cache_enabled(self) -> bool:
        """Whether observer-output cache is enabled for MultiAgent ablation runs."""
        return self.get("agent_multi.observer_cache.enabled", True)

    @property
    def agent_multi_observer_cache_dir(self) -> str:
        """Directory for observer-output cache files."""
        return self.get("agent_multi.observer_cache.cache_dir", "experiment_results/observer_cache")

    # ========================================================================
    # MedEvo Configuration
    # ========================================================================

    @property
    def med_evo_max_working_windows(self) -> int:
        """Maximum number of working windows provided to EventAgent."""
        value = self.get("med_evo.max_working_windows", 3)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 3

    @property
    def med_evo_max_critical_events(self) -> int:
        """Maximum number of critical events kept in memory."""
        value = self.get("med_evo.max_critical_events", 100)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 100

    @property
    def med_evo_max_window_summaries(self) -> int:
        """Maximum number of trajectory summaries retained (window summaries + episodes)."""
        value = self.get("med_evo.max_window_summaries", 100)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 100

    @property
    def med_evo_max_insights(self) -> int:
        """Maximum number of active insights retained in memory."""
        value = self.get("med_evo.max_insights", 5)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 5

    @property
    def med_evo_insight_every_n_windows(self) -> int:
        """Run InsightAgent once every N non-empty windows (default 1 = every window)."""
        value = self.get("med_evo.insight_every_n_windows", 1)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    @property
    def med_evo_episode_every_n_windows(self) -> int:
        """Run EpisodeAgent once every N non-empty windows (default 0 = disabled)."""
        value = self.get("med_evo.episode_every_n_windows", 0)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def to_dict(self) -> Dict:
        """Return the full configuration as a dictionary."""
        return self._config.copy()

    def __repr__(self) -> str:
        """String representation of config."""
        return f"Config(config_path={self.config_path})"


# Global config instance
_global_config: Optional[Config] = None


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config.json file. If None, uses default location.

    Returns:
        Config instance
    """
    global _global_config
    _global_config = Config(config_path)
    return _global_config


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config instance

    Raises:
        RuntimeError: If config hasn't been loaded yet
    """
    global _global_config
    if _global_config is None:
        # Auto-load default config
        _global_config = Config()
    return _global_config


# Convenience function for quick access
def get(key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation.

    Args:
        key_path: Dot-separated path to config value
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    return get_config().get(key_path, default)
