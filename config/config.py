"""
Configuration loader for EVO-Agent Oracle System.

This module loads configuration from config.json and provides easy access
to all configuration parameters throughout the application.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


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
            key_path: Dot-separated path to config value (e.g., "oracle.temperature")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            config.get("oracle.temperature")  # Returns 0.3
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

    # ========================================================================
    # Oracle Time Window Configuration
    # ========================================================================

    @property
    def oracle_current_window_hours(self) -> float:
        """Size of current observation window in hours for Oracle (default 0.5 = 30 minutes)."""
        return self.get("oracle_time_windows.current_window_hours", 0.5)

    @property
    def oracle_lookback_window_hours(self) -> float:
        """Size of historical lookback window for Oracle before current window starts (default 6 hours)."""
        return self.get("oracle_time_windows.lookback_window_hours", 6.0)

    @property
    def oracle_future_window_hours(self) -> float:
        """Size of future prediction window for Oracle after current window ends (default 6 hours)."""
        return self.get("oracle_time_windows.future_window_hours", 6.0)

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
        """Number of most recent discharge summaries to extract for Oracle (default 3)."""
        return self.get("oracle_time_windows.num_discharge_summaries", 3)

    @property
    def oracle_use_first_n_hours_after_icu(self) -> Optional[float]:
        """Use only the first N hours after ICU entry for Oracle (default None = use full duration)."""
        return self.get("oracle_time_windows.use_first_n_hours_after_icu", None)

    # ========================================================================
    # Agent Time Window Configuration
    # ========================================================================

    @property
    def agent_current_window_hours(self) -> float:
        """Size of current observation window in hours for Agent (default 0.5 = 30 minutes)."""
        return self.get("agent_time_windows.current_window_hours", 0.5)

    @property
    def agent_lookback_window_hours(self) -> float:
        """Size of historical lookback window for Agent before current window starts (default 6 hours)."""
        return self.get("agent_time_windows.lookback_window_hours", 6.0)

    @property
    def agent_future_window_hours(self) -> float:
        """Size of future prediction window for Agent after current window ends (default 6 hours)."""
        return self.get("agent_time_windows.future_window_hours", 6.0)

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
        """Number of most recent discharge summaries to extract for Agent (default 3)."""
        return self.get("agent_time_windows.num_discharge_summaries", 3)

    @property
    def agent_use_first_n_hours_after_icu(self) -> Optional[float]:
        """Use only the first N hours after ICU entry for Agent (default None = use full duration)."""
        return self.get("agent_time_windows.use_first_n_hours_after_icu", None)

    # ========================================================================
    # Legacy Time Window Configuration (for backward compatibility)
    # ========================================================================

    @property
    def current_window_hours(self) -> float:
        """Size of current observation window in hours (default 0.5 = 30 minutes). DEPRECATED: Use oracle_current_window_hours or agent_current_window_hours."""
        return self.get("oracle_time_windows.current_window_hours", 0.5)

    @property
    def lookback_window_hours(self) -> float:
        """Size of historical lookback window before current window starts (default 6 hours). DEPRECATED: Use oracle_lookback_window_hours or agent_lookback_window_hours."""
        return self.get("oracle_time_windows.lookback_window_hours", 6.0)

    @property
    def future_window_hours(self) -> float:
        """Size of future prediction window after current window ends (default 6 hours). DEPRECATED: Use oracle_future_window_hours or agent_future_window_hours."""
        return self.get("oracle_time_windows.future_window_hours", 6.0)

    @property
    def window_step_hours(self) -> float:
        """Step size between sliding windows in hours (default 0.5 = 30 minutes). DEPRECATED: Use oracle_window_step_hours or agent_window_step_hours."""
        return self.get("oracle_time_windows.window_step_hours", 0.5)

    @property
    def include_pre_icu_data(self) -> bool:
        """Whether to include pre-ICU hospital data in history context. DEPRECATED: Use oracle_include_pre_icu_data or agent_include_pre_icu_data."""
        return self.get("oracle_time_windows.include_pre_icu_data", True)

    # ========================================================================
    # LLM Configuration
    # ========================================================================

    @property
    def llm_provider(self) -> str:
        """LLM provider (e.g., 'openai', 'anthropic')."""
        return self.get("llm.provider")

    @property
    def llm_model(self) -> Optional[str]:
        """Model name for LLM."""
        return self.get("llm.model")

    @property
    def llm_temperature(self) -> float:
        """Sampling temperature for LLM."""
        return self.get("llm.temperature")

    @property
    def llm_max_tokens(self) -> int:
        """Maximum tokens for LLM responses."""
        return self.get("llm.max_tokens")

    # ========================================================================
    # Logging Configuration
    # ========================================================================

    @property
    def log_dir(self) -> str:
        """Directory for Oracle logs."""
        return self.get("logging.log_dir")

    @property
    def save_trajectories(self) -> bool:
        """Whether to save intermediate trajectory data."""
        return self.get("logging.save_trajectories")

    # ========================================================================
    # Processing Configuration
    # ========================================================================

    @property
    def max_patients(self) -> Optional[int]:
        """Maximum number of patients to process."""
        return self.get("processing.max_patients")

    @property
    def include_window_data(self) -> bool:
        """Whether to include full window data in reports."""
        return self.get("processing.include_window_data")

    # ========================================================================
    # ReMeM Configuration
    # ========================================================================

    @property
    def remem_observation_hours(self) -> float:
        """Number of hours to observe before making prediction (default 12)."""
        return self.get("remem.observation_hours", 12.0)

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
    def agent_fold_observation_hours(self) -> float:
        """Number of hours to observe before making prediction (default 12)."""
        return self.get("agent_fold.observation_hours", 12.0)

    @property
    def agent_fold_enable_key_events_extraction(self) -> bool:
        """Whether to enable key events extraction (default True)."""
        return self.get("agent_fold.enable_key_events_extraction", True)

    @property
    def agent_fold_max_trajectory_entries(self) -> int:
        """Maximum number of trajectory entries to maintain (default 20)."""
        return self.get("agent_fold.max_trajectory_entries", 20)

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
