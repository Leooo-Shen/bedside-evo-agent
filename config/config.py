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
    # Time Window Configuration
    # ========================================================================

    @property
    def current_window_hours(self) -> float:
        """Size of current observation window in hours (default 0.5 = 30 minutes)."""
        return self.get("time_windows.current_window_hours", 0.5)

    @property
    def lookback_window_hours(self) -> float:
        """Size of historical lookback window before current window starts (default 6 hours)."""
        return self.get("time_windows.lookback_window_hours", 6.0)

    @property
    def future_window_hours(self) -> float:
        """Size of future prediction window after current window ends (default 6 hours)."""
        return self.get("time_windows.future_window_hours", 6.0)

    @property
    def window_step_hours(self) -> float:
        """Step size between sliding windows in hours (default 0.5 = 30 minutes)."""
        return self.get("time_windows.window_step_hours", 0.5)

    @property
    def include_pre_icu_data(self) -> bool:
        """Whether to include pre-ICU hospital data in history context."""
        return self.get("time_windows.include_pre_icu_data", True)

    # ========================================================================
    # Oracle Configuration
    # ========================================================================

    @property
    def oracle_provider(self) -> str:
        """LLM provider for Oracle."""
        return self.get("oracle.provider")

    @property
    def oracle_model(self) -> Optional[str]:
        """Model name for Oracle."""
        return self.get("oracle.model")

    @property
    def oracle_temperature(self) -> float:
        """Sampling temperature for Oracle."""
        return self.get("oracle.temperature")

    @property
    def oracle_max_tokens(self) -> int:
        """Maximum tokens for Oracle responses."""
        return self.get("oracle.max_tokens")

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
